# NOTE for future contributors (incl. AI agents): keep this file a SMALL curated
# smoke set. New coverage (shapes, dtypes, backends, randomized breadth) belongs in
# tests/gemm/test_unified_gemm_fuzz.py -- extend an adapter/axis there. Add cases
# here only as deliberate regression anchors or for paths the fuzzer cannot express.

import pytest
import torch
import torch.nn.functional as F
from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    nvfp4_quantize,
    mxfp4_quantize,
)
from flashinfer.utils import (
    BackendSupportedError,
    get_compute_capability,
    is_sm12x_supported,
    version_at_least,
    LibraryError,
)
from packaging.version import Version

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.jit.cpp_ext import get_cuda_version
from flashinfer.gemm.gemm_base import (
    CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR,
    _b12x_gemm_fp4_requirement,
    _cudnn_gemm_fp4_requirement,
    _cute_dsl_gemm_fp4_requirement,
    _cutedsl_low_latency_gemm_fp4_requirement,
    _cutlass_gemm_fp4_requirement,
    _fp4_scale_byte_order_problem,
    _fp4_short_scale_problem,
    _trtllm_gemm_fp4_requirement,
)


def _test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    use_nvfp4 = fp4_type == "nvfp4"

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_fp4.is_backend_supported(backend, compute_capability_number):
        pytest.skip(
            f"Skipping test for {backend} because it is not supported on compute capability {compute_capability_number}."
        )

    if backend == "trtllm":
        if res_dtype == torch.float16:
            pytest.skip("Skipping test for trtllm fp4 with float16")
        if compute_capability[0] in [11, 12]:
            pytest.skip("trtllm gemm does not support SM110/SM120/SM121 GPUs.")
    if backend in ("cute-dsl", "cutedsl_low_latency"):
        if not use_128x4_sf_layout:
            pytest.skip(f"{backend} backend only supports 128x4 SF layout")
        if compute_capability[0] not in [10]:
            pytest.skip(f"{backend} backend only supports SM100/SM103 GPUs.")
    if backend == "b12x":
        if not use_128x4_sf_layout:
            pytest.skip("b12x backend only supports 128x4 SF layout")
        if compute_capability[0] != 12:
            pytest.skip("b12x backend only supports SM120/SM121 GPUs.")
        min_cuda_version = "12.9" if use_nvfp4 else "13.0"
        if not version_at_least(torch.version.cuda, min_cuda_version):
            pytest.skip(
                f"b12x {'NVFP4' if use_nvfp4 else 'MXFP4'} backend requires "
                f"CUDA {min_cuda_version}+."
            )
    if not use_128x4_sf_layout and backend != "trtllm":
        pytest.skip("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
    if not use_nvfp4 and backend not in [
        "cudnn",
        "auto",
        "cute-dsl",
        "cutedsl_low_latency",
        "b12x",
    ]:
        pytest.skip(
            "mx_fp4 is only supported for cudnn, cute-dsl, b12x, cutedsl_low_latency, and auto backends"
        )

    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    # for trtllm, we need to shuffle mat2 because we swap A, B.
    do_shuffle_b = backend == "trtllm"

    block_size = 16 if use_nvfp4 else 32
    has_alpha = fp4_type == "mxfp4_alpha" or fp4_type == "nvfp4"

    if use_nvfp4:
        input_fp4, input_inv_s = nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=do_shuffle_b,
        )
    else:
        input_fp4, input_inv_s = mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = mxfp4_quantize(mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2) if has_alpha else None

    reference = torch.mm(input, mat2.T)

    res = torch.empty([m, n], device="cuda", dtype=res_dtype)

    try:
        with autotune(auto_tuning):
            mm_fp4(
                input_fp4,
                mat2_fp4.T,
                input_inv_s,
                mat2_inv_s.T,
                alpha,
                res_dtype,
                res,
                block_size=block_size,
                use_8x4_sf_layout=not use_128x4_sf_layout,
                backend=backend,
                use_nvfp4=use_nvfp4,
                skip_check=False,
            )

        cos_sim = F.cosine_similarity(
            reference.float().reshape(-1), res.float().reshape(-1), dim=0
        )
        assert cos_sim > 0.97
    except LibraryError as e:
        # TODO: Remove this check once cuDNN backend version is updated to 9.14.0
        if str(e) == CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR:
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))


# Curated smoke set. Randomized breadth over m x {n,k} x fp4-type x backend (nvfp4 +
# mxfp4, 128x4 layout, tight elementwise oracle, determinism, autotune-winner
# validation) lives in tests/gemm/test_unified_gemm_fuzz.py's mm_nvfp4/mm_mxfp4
# adapters. This file keeps, per backend: one odd-M + one large-M case, the 8x4
# scale-factor layout (NOT fuzzed yet -- fuzzer TODO C1/#2861), the trtllm
# weight-shuffle path, and each fp4_type at least once.
_SMOKE_CASES = [
    # m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
    # (every case must actually run on its target arch: trtllm is the only 8x4-layout
    #  backend and is bf16-out only; b12x is nvfp4+128x4 only; mxfp4 runs on
    #  cudnn/cute-dsl/auto only -- see the skip matrix in _test_mm_fp4)
    (7, 128, 256, torch.bfloat16, "trtllm", True, False, "nvfp4"),
    (512, 512, 512, torch.bfloat16, "trtllm", False, True, "nvfp4"),
    (48, 256, 512, torch.bfloat16, "trtllm", False, False, "nvfp4"),
    (13, 256, 128, torch.bfloat16, "cudnn", True, True, "nvfp4"),
    (256, 512, 256, torch.float16, "cudnn", True, False, "mxfp4"),
    (9, 512, 256, torch.bfloat16, "cudnn", True, True, "mxfp4_alpha"),
    (1, 128, 512, torch.bfloat16, "cutlass", True, False, "nvfp4"),
    (31, 256, 256, torch.float16, "cutlass", True, True, "nvfp4"),
    (48, 512, 128, torch.float16, "cute-dsl", True, True, "nvfp4"),
    (3, 256, 512, torch.bfloat16, "cute-dsl", True, False, "mxfp4"),
    (17, 128, 128, torch.bfloat16, "b12x", True, False, "nvfp4"),
    (128, 512, 512, torch.bfloat16, "b12x", True, True, "nvfp4"),
]


_CUTEDSL_LOW_LATENCY_MODEL_CASES = [
    # GPT-OSS-120B
    (1, 1280, 2880, torch.bfloat16, "cutedsl_low_latency", True, False, "nvfp4"),
    (8, 1280, 2944, torch.bfloat16, "cutedsl_low_latency", True, False, "mxfp4"),
    (1, 2880, 1024, torch.float16, "cutedsl_low_latency", True, False, "nvfp4"),
    (8, 2880, 1024, torch.bfloat16, "cutedsl_low_latency", True, False, "mxfp4_alpha"),
    # DeepSeek-V3
    (4, 7168, 2048, torch.bfloat16, "cutedsl_low_latency", True, False, "nvfp4"),
    (8, 7168, 2048, torch.float16, "cutedsl_low_latency", True, False, "mxfp4"),
    (1, 3072, 1536, torch.bfloat16, "cutedsl_low_latency", True, False, "nvfp4"),
    (7, 3072, 1536, torch.bfloat16, "cutedsl_low_latency", True, False, "mxfp4"),
    (3, 129, 320, torch.bfloat16, "cutedsl_low_latency", True, False, "nvfp4"),
    (7, 127, 384, torch.float16, "cutedsl_low_latency", True, False, "mxfp4"),
]


@pytest.mark.parametrize(
    "m,n,k,res_dtype,backend,use_128x4_sf_layout,auto_tuning,fp4_type",
    _SMOKE_CASES + _CUTEDSL_LOW_LATENCY_MODEL_CASES,
)
def test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    # Non-auto backends
    _test_mm_fp4(
        m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
    )


# Auto backend: one case per fp4_type, autotune on/off and boundary M covered.
_AUTO_SMOKE_CASES = [
    # m, n, k, res_dtype, auto_tuning, fp4_type
    (1, 256, 512, torch.bfloat16, False, "nvfp4"),
    (512, 512, 256, torch.float16, True, "nvfp4"),
    (48, 512, 512, torch.bfloat16, True, "mxfp4"),
    (256, 256, 256, torch.bfloat16, False, "mxfp4_alpha"),
]


@pytest.mark.parametrize("m,n,k,res_dtype,auto_tuning,fp4_type", _AUTO_SMOKE_CASES)
def test_mm_fp4_backend_auto(m, n, k, res_dtype, auto_tuning, fp4_type):
    # Some test cases for auto backend.
    _test_mm_fp4(m, n, k, res_dtype, "auto", True, auto_tuning, fp4_type)


# Regression (#3560): b12x must accept ragged K (real floor K%32==0, not tile_k=128).
# K=192 (packed_k=96) is the shape #3560 broke; both auto_tuning values hit distinct paths.
@pytest.mark.parametrize("k", [96, 192])
@pytest.mark.parametrize("auto_tuning", [False, True])
def test_mm_fp4_b12x_ragged_k(k, auto_tuning):
    _test_mm_fp4(
        m=64,
        n=512,
        k=k,
        res_dtype=torch.bfloat16,
        backend="b12x",
        use_128x4_sf_layout=True,
        auto_tuning=auto_tuning,
        fp4_type="nvfp4",
    )


# K % 32 != 0 violates TMA 16-byte alignment; explicit b12x must reject cleanly.
def test_mm_fp4_b12x_misaligned_k_raises():
    device = torch.device("cuda")
    if not (
        is_sm12x_supported(device) and version_at_least(torch.version.cuda, "12.9")
    ):
        pytest.skip("b12x backend requires SM120/SM121 + CUDA 12.9+.")
    m, n, k = 64, 512, 112  # k % 32 == 16
    _, _, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
    res = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 32"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            alpha,
            torch.bfloat16,
            res,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
            skip_check=False,
        )


def _nvfp4_operands(m, n, k):
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    g_in = (448 * 6) / a.float().abs().nan_to_num().max()
    g_w = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_s = nvfp4_quantize(
        a, g_in, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_s = nvfp4_quantize(
        b, g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    return a, b, a_fp4, a_s, b_fp4, b_s, 1.0 / (g_in * g_w)


def test_mm_fp4_b12x_short_k_multi_wave():
    # One K tile and more work tiles than SMs stress the epilogue smem
    # handoff between a persistent CTA's work tiles, a regime the
    # parametrized shapes never reach. Repeats, since a bad handoff shows
    # up as a timing-dependent mismatch.
    device = torch.device("cuda")
    if not (
        is_sm12x_supported(device) and version_at_least(torch.version.cuda, "13.0")
    ):
        pytest.skip("b12x backend requires SM120/SM121 + CUDA 13+.")
    m, n, k = 1024, 4096, 128
    for _ in range(3):
        a, b, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
        res = mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            alpha,
            torch.bfloat16,
            None,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
        )
        reference = torch.mm(a, b.T)
        cos_sim = F.cosine_similarity(
            reference.reshape(-1).float(), res.reshape(-1).float(), dim=0
        ).item()
        assert cos_sim > 0.97


def test_mm_fp4_cute_dsl_misaligned_n_raises():
    device = torch.device("cuda")
    if get_compute_capability(device)[0] != 10:
        pytest.skip("cute_dsl backend only supports SM100/SM103 GPUs.")
    m, n, k = 16, 130, 128  # n % 8 == 2
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    g_in = (448 * 6) / a.float().abs().nan_to_num().max()
    g_w = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_s = nvfp4_quantize(
        a, g_in, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_s = nvfp4_quantize(
        b, g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    res = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="N % 8 == 0"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            1.0 / (g_in * g_w),
            torch.bfloat16,
            res,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cute-dsl",
            use_nvfp4=True,
            skip_check=False,
        )


def _run_cute_dsl_fp4_tactic(m, n, k, fp4_type, tactic):
    """Launch one forced tactic through the cute-dsl FP4 runner; check output."""
    from flashinfer.gemm import gemm_base

    major, minor = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("cute-dsl FP4 GEMM needs SM100/SM103.")
    use_nvfp4 = fp4_type == "nvfp4"
    if use_nvfp4:
        a, b, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
    else:
        a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
        b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
        a_fp4, a_s = mxfp4_quantize(a)
        b_fp4, b_s = mxfp4_quantize(b)
        alpha = None
    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(1, device="cuda", dtype=torch.uint8)
    runner = gemm_base._cute_dsl_gemm_fp4_runner(  # pyright: ignore[reportPrivateUsage]
        major, minor, True, torch.bfloat16, use_nvfp4
    )
    runner(
        inputs=[
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            alpha,
            torch.bfloat16,
            out,
            16 if use_nvfp4 else 32,
            use_nvfp4,
            workspace,
        ],
        tactic=tactic,
    )
    torch.cuda.synchronize()
    cos_sim = F.cosine_similarity(
        torch.mm(a, b.T).float().reshape(-1), out.float().reshape(-1), dim=0
    )
    assert cos_sim > 0.97


@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("m", [40, 64])
@pytest.mark.parametrize("tile_n", [8, 16, 32])
def test_mm_fp4_cute_dsl_stale_narrow_tile_tactic_falls_back(fp4_type, m, tile_n):
    """A narrow swap-AB tactic tuned for M<=32 must not be replayed at M>32.

    Narrow (< 64) N tiles cover at most 32 kernel-N columns; the runner falls
    back to the untuned selector instead of faulting with
    cudaErrorMisalignedAddress.
    """
    _run_cute_dsl_fp4_tactic(
        m, 256, 2048, fp4_type, ((128, tile_n), (1, 1), True, False, "sm100", None)
    )


@pytest.mark.parametrize("m", [40, 64])
@pytest.mark.parametrize("tile_n", [8, 32])
def test_mm_fp4_cute_dsl_stale_split_k_tactic_falls_back(m, tile_n):
    """A split-K tactic tuned for M<=32 falls back at M>32 instead of raising."""
    _run_cute_dsl_fp4_tactic(
        m, 256, 2048, "nvfp4", ((128, tile_n), (1, 1), True, False, "sm100sk", 2)
    )


def _skip_unless_per_token_alpha_gpu():
    major, minor = get_compute_capability(torch.device("cuda"))
    if (major, minor) not in [(10, 0), (10, 3)]:
        pytest.skip("per-token alpha needs the cute-dsl FP4 GEMM (SM100/SM103).")


# m is swept in full because the per-row indexing is what varies with it; n and
# k only change the tile schedule, so they are kept to the minimum that still
# exercises both a single- and a multi-tile N.
@pytest.mark.parametrize("m", [4, 17, 48, 128, 257])
@pytest.mark.parametrize("n", [256, 512])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cute-dsl", "auto"])
@pytest.mark.parametrize("auto_tuning", [False, True])
def test_mm_fp4_per_token_alpha(m, n, k, res_dtype, backend, auto_tuning):
    """A per-token alpha must scale each output row by its own dequant scale."""
    _skip_unless_per_token_alpha_gpu()

    torch.manual_seed(0)
    a, b, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
    # alpha = 1 / (g_in * g_w) undoes both NVFP4 global encode scales
    # ((448 * 6) / absmax) in a @ b.T; it is what a scalar-alpha call uses.
    scalar_alpha = alpha.float().reshape(1)

    # In production the per-token alpha is the dynamic per-row scale that
    # nvfp4_quantize(..., per_token_activation=True) returns, times the weight
    # scale. Here it is the scalar alpha times a synthetic per-row factor in
    # [0.25, 1.25) that differs on every row, so an epilogue that reads alpha
    # at the wrong coordinate cannot pass.
    row = 0.25 + torch.arange(m, device="cuda", dtype=torch.float32) / m
    per_token_alpha = (scalar_alpha * row).contiguous()

    out = torch.empty([m, n], device="cuda", dtype=res_dtype)
    out_scalar = torch.empty([m, n], device="cuda", dtype=res_dtype)
    with autotune(auto_tuning):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            per_token_alpha,
            res_dtype,
            out,
            block_size=16,
            backend=backend,
            use_nvfp4=True,
            skip_check=False,
        )
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            scalar_alpha,
            res_dtype,
            out_scalar,
            block_size=16,
            backend=backend,
            use_nvfp4=True,
            skip_check=False,
        )

    reference = torch.mm(a, b.T).float() * row[:, None]
    cos_sim = F.cosine_similarity(reference.reshape(-1), out.float().reshape(-1), dim=0)
    assert cos_sim > 0.97

    # Both calls accumulate the same products, so the per-token result must be
    # the scalar-alpha result scaled row by row. Cosine similarity stays high
    # even when the epilogue reads alpha at the wrong coordinate; this does not.
    torch.testing.assert_close(
        out.float(),
        out_scalar.float() * row[:, None],
        rtol=2e-2,
        atol=2e-2 * out_scalar.float().abs().max().item(),
    )


@pytest.mark.parametrize(
    "backend", ["cutlass", "cudnn", "trtllm", "b12x", "cutedsl_low_latency"]
)
def test_mm_fp4_per_token_alpha_rejected_by_other_backends(backend):
    """A backend without a per-row epilogue would apply alpha[0] to every row;
    the implementation must refuse the call instead of running it."""
    if get_compute_capability(torch.device("cuda"))[0] < 10:
        pytest.skip("nvfp4_quantize needs SM100+.")
    m, n, k = 16, 256, 256
    torch.manual_seed(0)
    _, _, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
    per_token_alpha = alpha.float().reshape(1).expand(m).contiguous()
    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(
        (ValueError, BackendSupportedError),
        match="per-token alpha|does not support backend",
    ):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            per_token_alpha,
            torch.bfloat16,
            out,
            block_size=16,
            backend=backend,
            use_nvfp4=True,
            skip_check=False,
        )


def test_mm_fp4_per_token_alpha_auto_misaligned_n_raises():
    """backend="auto" with a per-token alpha must go through the cute-dsl
    requirement function like any other backend: n % 8 != 0 is refused before
    the runner is built (it used to reach the runner and crash with a
    TypeError once no tactic was valid)."""
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("cute_dsl backend only supports SM100/SM103 GPUs.")
    m, n, k = 16, 130, 128  # n % 8 == 2
    torch.manual_seed(0)
    _, _, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
    per_token_alpha = alpha.float().reshape(1).expand(m).contiguous()
    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(BackendSupportedError, match="No suitable auto backends"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            per_token_alpha,
            torch.bfloat16,
            out,
            block_size=16,
            backend="auto",
            use_nvfp4=True,
            skip_check=False,
        )


# The untuned low-M selector routes these shapes through the cluster split-K
# kernel (<= 20 weight tiles with an 8/16-wide token tile -> 4 K slices; long K
# -> 2 slices). Its epilogue applies the per-token alpha after the FP32
# cluster reduction, so the row scaling must be exact here too, and the
# result must still track the unquantised product.
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1536, 4096),
        (8, 2048, 8192),
        (16, 1024, 4096),
        (17, 7168, 16384),
        (32, 4096, 16384),
        # 17 <= M <= 32 with few weight tiles: 8-wide token tile over several
        # N tiles (SFB sub-tile addressing), two K slices.
        (17, 1536, 7168),
        (32, 2112, 7168),
        (24, 2048, 4096),
    ],
)
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4_per_token_alpha_splitk(m, n, k, res_dtype):
    _skip_unless_per_token_alpha_gpu()
    from flashinfer.gemm.gemm_base import _select_sm100_mm_fp4_splitk_tactic
    from flashinfer.utils import get_device_sm_count

    sm_count = get_device_sm_count(torch.device("cuda"))
    tactic = _select_sm100_mm_fp4_splitk_tactic(m, n, k, sm_count, True)
    assert tactic is not None, "shape must exercise the split-K kernel"

    torch.manual_seed(0)
    a, b, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
    scalar_alpha = alpha.float().reshape(1)
    row = 0.25 + torch.arange(m, device="cuda", dtype=torch.float32) / m
    per_token_alpha = (scalar_alpha * row).contiguous()

    out = torch.empty([m, n], device="cuda", dtype=res_dtype)
    out_scalar = torch.empty([m, n], device="cuda", dtype=res_dtype)
    for alpha_arg, dst in ((per_token_alpha, out), (scalar_alpha, out_scalar)):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            alpha_arg,
            res_dtype,
            dst,
            block_size=16,
            backend="cute-dsl",
            use_nvfp4=True,
            skip_check=False,
        )

    reference = torch.mm(a, b.T).float() * row[:, None]
    cos_sim = F.cosine_similarity(reference.reshape(-1), out.float().reshape(-1), dim=0)
    assert cos_sim > 0.97
    torch.testing.assert_close(
        out.float(),
        out_scalar.float() * row[:, None],
        rtol=2e-2,
        atol=2e-2 * out_scalar.float().abs().max().item(),
    )

    # The split-K sum must match the single-CTA persistent kernel (cutlass
    # backend as an independent reference) within the FP4 GEMM tolerance.
    out_ref = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    mm_fp4(
        a_fp4,
        b_fp4.T,
        a_s,
        b_s.T,
        scalar_alpha,
        torch.bfloat16,
        out_ref,
        block_size=16,
        backend="cutlass",
        use_nvfp4=True,
        skip_check=False,
    )
    torch.testing.assert_close(
        out_scalar.float(), out_ref.float(), rtol=1e-2, atol=1e-2
    )


# Narrow token tiles over a weight grid of about a wave or more (>= 128
# tiles) take the K tile 512 persistent variant (8 MMA K instructions per stage); same accumulation
# order, so the per-token result must match the scalar path row by row and the
# cutlass backend within the FP4 tolerance.
@pytest.mark.parametrize(
    "m,n,k",
    [
        (8, 18432, 7168),
        (17, 28672, 8192),
        (32, 18432, 7168),
    ],
)
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4_per_token_alpha_deep_k(m, n, k, res_dtype):
    _skip_unless_per_token_alpha_gpu()
    from flashinfer.gemm.gemm_base import (
        _SM100_DEEP_K_INST,
        _select_sm100_mm_fp4_splitk_tactic,
    )
    from flashinfer.utils import get_device_sm_count

    sm_count = get_device_sm_count(torch.device("cuda"))
    tactic = _select_sm100_mm_fp4_splitk_tactic(m, n, k, sm_count, True)
    assert tactic is not None and tactic[4] == "sm100", tactic
    assert tactic[5] == _SM100_DEEP_K_INST and tactic[0][1] <= 32, tactic

    _check_per_token_alpha_untuned(m, n, k, res_dtype)


def _check_per_token_alpha_untuned(m, n, k, res_dtype):
    """Per-token alpha through the untuned selector: per-token == scalar x row,
    scalar == cutlass backend within the FP4 GEMM tolerance."""
    torch.manual_seed(0)
    a, b, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
    scalar_alpha = alpha.float().reshape(1)
    row = 0.25 + torch.arange(m, device="cuda", dtype=torch.float32) / m
    per_token_alpha = (scalar_alpha * row).contiguous()

    out = torch.empty([m, n], device="cuda", dtype=res_dtype)
    out_scalar = torch.empty([m, n], device="cuda", dtype=res_dtype)
    for alpha_arg, dst in ((per_token_alpha, out), (scalar_alpha, out_scalar)):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            alpha_arg,
            res_dtype,
            dst,
            block_size=16,
            backend="cute-dsl",
            use_nvfp4=True,
            skip_check=False,
        )
    torch.testing.assert_close(
        out.float(),
        out_scalar.float() * row[:, None],
        rtol=2e-2,
        atol=2e-2 * out_scalar.float().abs().max().item(),
    )

    out_ref = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    mm_fp4(
        a_fp4,
        b_fp4.T,
        a_s,
        b_s.T,
        scalar_alpha,
        torch.bfloat16,
        out_ref,
        block_size=16,
        backend="cutlass",
        use_nvfp4=True,
        skip_check=False,
    )
    torch.testing.assert_close(
        out_scalar.float(), out_ref.float(), rtol=1e-2, atol=1e-2
    )


# SM103-only low-M rule (8192 <= K < 16384, <= sm_count/2 weight tiles): TMA
# prefetch for 17 <= M <= 32. On other SMs (and for M <= 16) the same shapes
# take the default persistent tactic; either way the result is checked.
@pytest.mark.parametrize(
    "m,n,k",
    [
        (8, 8192, 8192),
        (17, 8192, 8192),
        (32, 8192, 8192),
    ],
)
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4_per_token_alpha_low_m_untuned(m, n, k, res_dtype):
    _skip_unless_per_token_alpha_gpu()
    _check_per_token_alpha_untuned(m, n, k, res_dtype)


def _quantized_fp4_operands(m, n, k, use_nvfp4):
    major, minor = get_compute_capability(torch.device("cuda"))
    if not mm_fp4.is_backend_supported("cudnn", major * 10 + minor):
        pytest.skip(f"cudnn fp4 backend unsupported on SM{major}{minor}")
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    if use_nvfp4:
        one = torch.tensor(1.0, device="cuda")
        a_fp4, a_sf = nvfp4_quantize(a, one, sfLayout=SfLayout.layout_128x4)
        b_fp4, b_sf = nvfp4_quantize(b, one, sfLayout=SfLayout.layout_128x4)
    else:
        a_fp4, a_sf = mxfp4_quantize(a)
        b_fp4, b_sf = mxfp4_quantize(b)
    return a_fp4, a_sf, b_fp4, b_sf


def _expect_cudnn_scale_error(
    m, n, k, use_nvfp4, make_b_scale, match, make_a_scale=lambda a_sf: a_sf
):
    a_fp4, a_sf, b_fp4, b_sf = _quantized_fp4_operands(m, n, k, use_nvfp4)
    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    try:
        with pytest.raises(ValueError, match=match):
            mm_fp4(
                a_fp4,
                b_fp4.T,
                make_a_scale(a_sf),
                make_b_scale(b_sf),
                None,
                torch.bfloat16,
                out,
                block_size=16 if use_nvfp4 else 32,
                backend="cudnn",
                use_nvfp4=use_nvfp4,
            )
    except (BackendSupportedError, LibraryError) as e:
        pytest.skip(f"cudnn fp4 backend unavailable here: {e}")
    except RuntimeError as e:
        # _check_cudnn_fp4_availability raises RuntimeError when cuDNN or its FP4
        # support is missing; any other RuntimeError is a real failure.
        if "FP4 requires" in str(e) or "cuDNN is not available" in str(e):
            pytest.skip(f"cudnn fp4 backend unavailable here: {e}")
        raise


# Regression anchors: the cuDNN path derives its block-scale descriptors from the
# scale tensors' shapes and reads their bytes in the order fp4 quantization writes.
# A B scale tensor in the quantizer's (n, k // block_size) orientation (not
# transposed) used to reach cuDNN unchecked: silently wrong results for n != m (as at
# the first test's shape), and an illegal-instruction fault for MXFP4 at
# m = n = 128, k = 64. A contiguous copy of the transpose has the right shape but
# the wrong byte order and gave wrong results (NaN at 128 x 256 x 512 for MXFP4).
@pytest.mark.parametrize("use_nvfp4", [False, True])
def test_mm_fp4_cudnn_rejects_untransposed_b_descale(use_nvfp4):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    _expect_cudnn_scale_error(
        128, 256, 64, use_nvfp4, lambda b_sf: b_sf, "pass its transpose"
    )


@pytest.mark.parametrize("use_nvfp4", [False, True])
def test_mm_fp4_cudnn_rejects_contiguous_copy_of_b_descale_transpose(use_nvfp4):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    _expect_cudnn_scale_error(
        128,
        256,
        512,
        use_nvfp4,
        lambda b_sf: b_sf.T.contiguous(),
        "transposed view of those scales",
    )


@pytest.mark.parametrize("use_nvfp4", [False, True])
def test_mm_fp4_cudnn_rejects_untransposed_b_descale_square(use_nvfp4):
    # k // block_size and n both pad to 128, so the shapes of the two orientations
    # agree and only the stride tells them apart.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    block_size = 16 if use_nvfp4 else 32
    _expect_cudnn_scale_error(
        128,
        128,
        128 * block_size,
        use_nvfp4,
        lambda b_sf: b_sf,
        "transposed view of those scales",
    )


@pytest.mark.parametrize("use_nvfp4", [False, True])
def test_mm_fp4_cudnn_rejects_relaid_out_a_descale(use_nvfp4):
    # Right shape, storage re-ordered: a copy of the transpose viewed back.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    _expect_cudnn_scale_error(
        128,
        256,
        512,
        use_nvfp4,
        lambda b_sf: b_sf.T,
        "not a re-laid-out copy",
        make_a_scale=lambda a_sf: a_sf.T.contiguous().T,
    )


@pytest.mark.parametrize("use_nvfp4", [False, True])
def test_mm_fp4_cudnn_auto_declines_untransposed_b_descale(use_nvfp4):
    # With backend="auto" the cuDNN requirement returns False for a scale tensor
    # the check rejects, so auto dispatch moves on to another backend instead of
    # raising. backend="cudnn" first tells cuDNN FP4 unavailability (a library
    # error, skipped) apart from a rejected layout, so the documented scales must
    # then be accepted.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    m, n, k = 128, 256, 64
    a_fp4, a_sf, b_fp4, b_sf = _quantized_fp4_operands(m, n, k, use_nvfp4)
    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)

    def accepted(b_scale, backend="auto"):
        return _cudnn_gemm_fp4_requirement(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_scale,
            None,
            torch.bfloat16,
            out,
            block_size=16 if use_nvfp4 else 32,
            backend=backend,
            use_nvfp4=use_nvfp4,
        )

    try:
        accepted(b_sf.T, backend="cudnn")
    except (BackendSupportedError, LibraryError) as e:
        pytest.skip(f"cudnn fp4 backend unavailable here: {e}")
    except RuntimeError as e:
        if "FP4 requires" in str(e) or "cuDNN is not available" in str(e):
            pytest.skip(f"cudnn fp4 backend unavailable here: {e}")
        raise
    assert accepted(b_sf.T) is True
    assert accepted(b_sf) is False


# The CUTLASS, b12x and TRT-LLM backends, and CuTe DSL on SM100, read the block-scale
# bytes as given (the SM12x CuTe DSL path checks shape and stride itself). A
# transposed copy of the A or B scales holds the same values in a different byte order
# and used to give wrong results silently on some architecture; it now raises
# ValueError. Views over the quantizer's bytes, and reshapes to shapes a copy cannot
# have, keep their behaviour.
_BYTE_ORDER_BACKENDS = [
    ("cutlass", "nvfp4"),
    ("b12x", "nvfp4"),
    ("b12x", "mxfp4"),
    ("trtllm", "nvfp4"),
    ("cute-dsl", "nvfp4"),
    ("cute-dsl", "mxfp4"),
    ("cutedsl_low_latency", "nvfp4"),
    ("cutedsl_low_latency", "mxfp4"),
]

# (a_descale, b_descale) built from the quantizer outputs a_sf and b_sf
_BLOCK_SCALE_VIEWS = {
    "untransposed_b": lambda a_sf, b_sf: (a_sf, b_sf),
    "a_transposed_view": lambda a_sf, b_sf: (a_sf.T, b_sf.T),
    "a_reshaped_to_one_row": lambda a_sf, b_sf: (a_sf.reshape(1, -1), b_sf.T),
    "b_reshaped_to_one_row": lambda a_sf, b_sf: (a_sf, b_sf.reshape(1, -1)),
}
_BLOCK_SCALE_TRANSPOSED_COPIES = {
    "b_contiguous_transpose": lambda a_sf, b_sf: (a_sf, b_sf.T.contiguous()),
    "b_contiguous_transpose_transposed_back": lambda a_sf, b_sf: (
        a_sf,
        b_sf.T.contiguous().T,
    ),
    "a_contiguous_transpose_transposed_back": lambda a_sf, b_sf: (
        a_sf.T.contiguous().T,
        b_sf.T,
    ),
    "a_contiguous_transpose": lambda a_sf, b_sf: (a_sf.T.contiguous(), b_sf.T),
    # B is checked even when A's scales are passed flattened
    "b_contiguous_transpose_with_flat_a": lambda a_sf, b_sf: (
        a_sf.reshape(-1),
        b_sf.T.contiguous(),
    ),
}


def _byte_order_case(backend, fp4_type):
    """Skip where the backend cannot run; else quantize A and B and return the inputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cc = get_compute_capability(torch.device("cuda"))
    sm12x = cc[0] == 12
    if not mm_fp4.is_backend_supported(backend, cc[0] * 10 + cc[1]):
        pytest.skip(f"{backend} FP4 GEMM is not supported on this GPU.")
    if backend == "cute-dsl" and sm12x and fp4_type == "mxfp4":
        pytest.skip("The SM12x CuTe DSL path takes NVFP4 only.")
    if (
        backend == "cute-dsl"
        and sm12x
        and not version_at_least(torch.version.cuda, "13.0")
    ):
        pytest.skip("The SM12x CuTe DSL path requires CUDA 13+.")
    # b12x runs on CuTe DSL too
    if (
        backend in ("cute-dsl", "cutedsl_low_latency", "b12x")
        and not is_cute_dsl_available()
    ):
        pytest.skip("Please `pip install nvidia-cutlass-dsl`")
    min_cuda = "12.9" if fp4_type == "nvfp4" else "13.0"
    if backend == "b12x" and get_cuda_version() < Version(min_cuda):
        pytest.skip(f"b12x {fp4_type} FP4 GEMM requires CUDA {min_cuda}+.")
    m, n, k = (8 if backend == "cutedsl_low_latency" else 128), 256, 512
    use_nvfp4 = fp4_type == "nvfp4"
    torch.manual_seed(0)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    if use_nvfp4:
        ga = (448 * 6) / a.float().abs().nan_to_num().max()
        gb = (448 * 6) / b.float().abs().nan_to_num().max()
        a_fp4, a_sf = nvfp4_quantize(a, ga)
        # the TRT-LLM backend takes B shuffled, as in _test_mm_fp4
        b_fp4, b_sf = nvfp4_quantize(b, gb, do_shuffle=backend == "trtllm")
        alpha = 1.0 / (ga * gb)
    else:
        (a_fp4, a_sf), (b_fp4, b_sf), alpha = mxfp4_quantize(a), mxfp4_quantize(b), None
    return dict(
        a_fp4=a_fp4,
        b_fp4=b_fp4,
        a_sf=a_sf,
        b_sf=b_sf,
        alpha=alpha,
        m=m,
        n=n,
        block_size=16 if use_nvfp4 else 32,
        use_nvfp4=use_nvfp4,
        sm12x=sm12x,
    )


def _run_byte_order_case(case, backend, a_descale, b_descale):
    out = torch.empty([case["m"], case["n"]], device="cuda", dtype=torch.bfloat16)
    mm_fp4(
        case["a_fp4"],
        case["b_fp4"].T,
        a_descale,
        b_descale,
        case["alpha"],
        torch.bfloat16,
        out,
        block_size=case["block_size"],
        backend=backend,
        use_nvfp4=case["use_nvfp4"],
        skip_check=False,
    )
    return out


@pytest.mark.parametrize("form", list(_BLOCK_SCALE_VIEWS))
@pytest.mark.parametrize("backend, fp4_type", _BYTE_ORDER_BACKENDS)
def test_mm_fp4_keeps_block_scale_views(backend, fp4_type, form):
    """A view over the quantizer's bytes gives the documented result, or the
    backend's own error where it already refused it (the C++ contiguity checks on
    SM100, the SM12x CuTe DSL check), never the transposed-copy or storage error."""
    case = _byte_order_case(backend, fp4_type)
    reference = _run_byte_order_case(case, backend, case["a_sf"], case["b_sf"].T)
    a_descale, b_descale = _BLOCK_SCALE_VIEWS[form](case["a_sf"], case["b_sf"])
    try:
        out = _run_byte_order_case(case, backend, a_descale, b_descale)
    except (RuntimeError, ValueError) as err:
        assert "the layout a transposed copy of the scales has" not in str(err)
        # a view of the full quantizer storage must never trip the storage check
        assert "would read past the tensor's end" not in str(err)
    else:
        assert torch.equal(out, reference)


@pytest.mark.parametrize("form", list(_BLOCK_SCALE_TRANSPOSED_COPIES))
@pytest.mark.parametrize("backend, fp4_type", _BYTE_ORDER_BACKENDS)
def test_mm_fp4_rejects_transposed_block_scale_copy(backend, fp4_type, form):
    case = _byte_order_case(backend, fp4_type)
    a_descale, b_descale = _BLOCK_SCALE_TRANSPOSED_COPIES[form](
        case["a_sf"], case["b_sf"]
    )
    # the SM12x CuTe DSL path refuses them with its own check first
    own_check = backend == "cute-dsl" and case["sm12x"]
    with pytest.raises(
        ValueError,
        match=None if own_check else "the layout a transposed copy of the scales has",
    ):
        _run_byte_order_case(case, backend, a_descale, b_descale)


@pytest.mark.parametrize("backend, fp4_type", _BYTE_ORDER_BACKENDS)
def test_mm_fp4_auto_declines_transposed_block_scale_copy(backend, fp4_type):
    """With backend="auto" the requirement check returns False instead of raising,
    so a transposed copy takes the backend out of the candidates."""
    case = _byte_order_case(backend, fp4_type)
    if backend == "cute-dsl" and case["sm12x"]:
        pytest.skip("The SM12x CuTe DSL path is never selected by auto.")
    requirement = {
        "cutlass": _cutlass_gemm_fp4_requirement,
        "b12x": _b12x_gemm_fp4_requirement,
        "trtllm": _trtllm_gemm_fp4_requirement,
        "cute-dsl": _cute_dsl_gemm_fp4_requirement,
        "cutedsl_low_latency": _cutedsl_low_latency_gemm_fp4_requirement,
    }[backend]
    assert not requirement(
        case["a_fp4"],
        case["b_fp4"].T,
        case["a_sf"],
        case["b_sf"].T.contiguous(),
        case["alpha"],
        block_size=case["block_size"],
        backend="auto",
        use_nvfp4=case["use_nvfp4"],
    )


_SHORT_BLOCK_SCALES = {
    # a copy of the quantizer output without its last row. It is one row short of
    # what the layout reads. clone() is needed because a view of the full output
    # would pass the storage check.
    "a_one_row_short": lambda a_sf, b_sf: (a_sf[:-1].clone(), b_sf.T),
    "b_one_row_short": lambda a_sf, b_sf: (a_sf, b_sf[:-1].clone().T),
}


def test_mm_fp4_helper_rejects_scales_shorter_than_the_layout():
    """The test covers short A scales at m = 8 and k = 512. A linear A scale tensor
    is a plain row-major (8, 32) array of 8 x 32 = 256 bytes. An 8x4-layout A
    tensor is also 256 bytes. The 128x4 layout reads 128 x 32 = 4096 bytes, so a
    128x4 backend would read past the end of either 256-byte tensor. TRT-LLM's
    8x4 A layout reads 8 x 32 = 256 bytes, so the same bytes are enough there.
    The test also covers short B scales, the 8x4 row padding at m = 9, and a
    short 3-D tensor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    m, n, k = 8, 256, 512
    a = torch.empty([m, k // 2], dtype=torch.uint8, device="cuda")
    b = torch.empty([k // 2, n], dtype=torch.uint8, device="cuda")
    linear_a = torch.empty([m, k // 16], dtype=torch.float8_e4m3fn, device="cuda")
    full_b = torch.empty([n, k // 16], dtype=torch.float8_e4m3fn, device="cuda").T
    problem = _fp4_scale_byte_order_problem(a, b, linear_a, full_b, 16)
    assert problem is not None and "a_descale has 256 bytes" in problem
    assert "reads 4096 (128 padded rows of 32 scales" in problem
    assert (
        _fp4_scale_byte_order_problem(a, b, linear_a, full_b, 16, a_is_128x4=False)
        is None
    )
    full_a = torch.empty([128, k // 16], dtype=torch.float8_e4m3fn, device="cuda")
    assert _fp4_scale_byte_order_problem(a, b, full_a, full_b, 16) is None
    # B: linear scales at n = 256 already have the padded 128x4 size and pass the
    # storage check.
    # At n = 8 they hold 8 x 32 = 256 bytes. The layout reads 4096 bytes.
    linear_b = torch.empty([n, k // 16], dtype=torch.float8_e4m3fn, device="cuda")
    assert _fp4_scale_byte_order_problem(a, b, full_a, linear_b.T, 16) is None
    b_short = torch.empty([k // 2, 8], dtype=torch.uint8, device="cuda")
    linear_b_short = torch.empty([8, k // 16], dtype=torch.float8_e4m3fn, device="cuda")
    problem = _fp4_scale_byte_order_problem(a, b_short, full_a, linear_b_short.T, 16)
    assert problem is not None and "b_descale has 256 bytes" in problem
    assert (
        "reads 4096 (128 padded rows of 32 scales" in problem
        and "linear (8, 32) array" in problem
    )
    # 8x4 pads rows to a multiple of 8. At m = 9 a (9, 32) tensor is 9 x 32 = 288
    # bytes. The layout reads 16 x 32 = 512 bytes. A (16, 32) tensor is enough.
    a9 = torch.empty([9, k // 2], dtype=torch.uint8, device="cuda")
    scales_9 = torch.empty([9, k // 16], dtype=torch.float8_e4m3fn, device="cuda")
    problem = _fp4_scale_byte_order_problem(
        a9, b, scales_9, full_b, 16, a_is_128x4=False
    )
    assert problem is not None and "a_descale has 288 bytes" in problem
    assert "8x4 block-scale layout for this call reads 512 (16 padded rows" in problem
    assert "is_sf_8x4_layout=True" in problem
    scales_16 = torch.empty([16, k // 16], dtype=torch.float8_e4m3fn, device="cuda")
    assert (
        _fp4_scale_byte_order_problem(a9, b, scales_16, full_b, 16, a_is_128x4=False)
        is None
    )
    # the storage check does not look at the rank: a short 3-D tensor is rejected too
    a128 = torch.empty([128, k // 2], dtype=torch.uint8, device="cuda")
    short_3d = torch.empty([1, 8, k // 16], dtype=torch.float8_e4m3fn, device="cuda")
    problem = _fp4_scale_byte_order_problem(a128, b, short_3d, full_b, 16)
    assert problem is not None and "a_descale has 256 bytes" in problem
    # A slice that drops the first row moves the pointer 32 bytes on, leaving
    # 4096 - 32 = 4064 bytes, and fails the byte count.
    problem = _fp4_short_scale_problem(a, b, full_a[1:], full_b, 16)
    assert problem is not None and "a_descale has 4064 bytes" in problem
    b_rows = torch.empty([n, k // 16], dtype=torch.float8_e4m3fn, device="cuda")
    problem = _fp4_short_scale_problem(a, b, full_a, b_rows[1:].T, 16)
    assert problem is not None and "b_descale has 8160 bytes" in problem
    # A slice that drops the last row keeps all 4096 bytes but has 127 rows of the
    # 128 the layout pads to, so the row check rejects it.
    problem = _fp4_short_scale_problem(a, b, full_a[:-1], full_b, 16)
    assert problem is not None and "127 rows of 32 scales" in problem
    problem = _fp4_short_scale_problem(a, b, full_a, b_rows[:-1].T, 16)
    assert (
        problem is not None and "b_descale is a row-major tensor of 255 rows" in problem
    )
    # nvfp4_quantize's linear output at m = 120 is a (120, 32) array on 4096
    # bytes, because its rows are padded to a multiple of 16 in storage. The
    # byte count passes; the row check rejects it.
    a120 = torch.empty([120, k // 2], dtype=torch.uint8, device="cuda")
    linear_120 = torch.empty([128, k // 16], dtype=torch.float8_e4m3fn, device="cuda")[
        :120
    ]
    problem = _fp4_short_scale_problem(a120, b, linear_120, full_b, 16)
    assert problem is not None and "120 rows of 32 scales" in problem
    assert "128 padded rows" in problem


@pytest.mark.parametrize("form", list(_SHORT_BLOCK_SCALES))
@pytest.mark.parametrize("backend, fp4_type", _BYTE_ORDER_BACKENDS)
def test_mm_fp4_rejects_short_block_scales(backend, fp4_type, form):
    case = _byte_order_case(backend, fp4_type)
    a_descale, b_descale = _SHORT_BLOCK_SCALES[form](case["a_sf"], case["b_sf"])
    own_check = backend == "cute-dsl" and case["sm12x"]
    with pytest.raises(
        ValueError, match=None if own_check else "would read past the tensor's end"
    ):
        _run_byte_order_case(case, backend, a_descale, b_descale)
    requirement = {
        "cutlass": _cutlass_gemm_fp4_requirement,
        "b12x": _b12x_gemm_fp4_requirement,
        "trtllm": _trtllm_gemm_fp4_requirement,
        "cute-dsl": _cute_dsl_gemm_fp4_requirement,
        "cutedsl_low_latency": _cutedsl_low_latency_gemm_fp4_requirement,
    }[backend]
    if not own_check:
        assert not requirement(
            case["a_fp4"],
            case["b_fp4"].T,
            a_descale,
            b_descale,
            case["alpha"],
            block_size=case["block_size"],
            backend="auto",
            use_nvfp4=case["use_nvfp4"],
        )


def test_mm_fp4_cudnn_rejects_short_3d_block_scales():
    """cuDNN's shape check only looks at 2-D tensors and would let a short 3-D scale
    tensor through. cuDNN would then read past its end and return wrong values
    with no error. _fp4_short_scale_problem, the storage check, rejects the tensor
    first."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    _expect_cudnn_scale_error(
        128,
        256,
        512,
        True,
        make_b_scale=lambda b_sf: b_sf.T,
        match="would read past the tensor's end",
        make_a_scale=lambda a_sf: a_sf[:8].clone().reshape(1, 8, a_sf.shape[1]),
    )


def test_mm_fp4_cudnn_rejects_short_2d_block_scales():
    """The cuDNN path end to end with a 2-D A scale tensor one row short of the
    128x4 layout: the storage check rejects it before the shape check."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    _expect_cudnn_scale_error(
        128,
        256,
        512,
        True,
        make_b_scale=lambda b_sf: b_sf.T,
        match="would read past the tensor's end",
        make_a_scale=lambda a_sf: a_sf[:-1].clone(),
    )


def test_mm_fp4_auto_declines_short_3d_block_scales():
    """In auto mode every requirement check declines the short 3-D tensor, cuDNN's
    included. The call raises instead of running a backend past the tensor's end."""
    case = _byte_order_case("cutlass", "nvfp4")
    a_short_3d = case["a_sf"][:8].clone().reshape(1, 8, case["a_sf"].shape[1])
    with pytest.raises(BackendSupportedError, match="No suitable auto backends"):
        _run_byte_order_case(case, "auto", a_short_3d, case["b_sf"].T)


def test_mm_fp4_auto_raises_on_short_block_scales():
    """The 2-D tensor one row short, in auto mode, end to end: no backend accepts
    it and the call raises."""
    case = _byte_order_case("cutlass", "nvfp4")
    a_short = case["a_sf"][:-1].clone()
    with pytest.raises(BackendSupportedError, match="No suitable auto backends"):
        _run_byte_order_case(case, "auto", a_short, case["b_sf"].T)


def test_mm_fp4_auto_raises_on_linear_scales_with_padded_storage():
    """nvfp4_quantize's linear A scales at m = 120 sit on 4096 bytes, as many as
    the 128x4 layout reads, so only the row check catches them. Auto mode must
    raise instead of returning a wrong result."""
    # The same skips as the other auto-mode tests: CUDA, and a GPU and library
    # where the quantizer and an FP4 backend run.
    _byte_order_case("cutlass", "nvfp4")
    m, n, k = 120, 256, 512
    one = torch.tensor(1.0, device="cuda")
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp4, _ = nvfp4_quantize(a, one, sfLayout=SfLayout.layout_128x4)
    b_fp4, b_sf = nvfp4_quantize(b, one, sfLayout=SfLayout.layout_128x4)
    _, a_linear = nvfp4_quantize(a, one, sfLayout=SfLayout.layout_linear)
    assert tuple(a_linear.shape) == (120, 32)
    assert a_linear.untyped_storage().nbytes() == 4096
    out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(BackendSupportedError, match="No suitable auto backends"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_linear,
            b_sf.T,
            one,
            torch.bfloat16,
            out,
            block_size=16,
            backend="auto",
            use_nvfp4=True,
        )


@pytest.mark.parametrize("backend, fp4_type", _BYTE_ORDER_BACKENDS)
def test_mm_fp4_keeps_block_scales_at_a_storage_offset(backend, fp4_type):
    """The storage check _fp4_short_scale_problem counts bytes from the data
    pointer to the end of the storage. The quantizer's bytes placed after a
    256-element (256-byte) prefix in a larger buffer therefore pass the check.
    The result matches the unmodified scales."""
    case = _byte_order_case(backend, fp4_type)
    reference = _run_byte_order_case(case, backend, case["a_sf"], case["b_sf"].T)
    a_sf = case["a_sf"]
    buffer = torch.empty(a_sf.numel() + 256, dtype=a_sf.dtype, device=a_sf.device)
    buffer[256:].copy_(a_sf.reshape(-1))
    a_descale = buffer[256:].view(a_sf.shape)
    assert a_descale.storage_offset() == 256
    out = _run_byte_order_case(case, backend, a_descale, case["b_sf"].T)
    assert torch.equal(out, reference)


def test_mm_fp4_trtllm_8x4_rejects_transposed_b_scale_copy():
    """With A's scales in the 8x4 layout, B's are still 128x4 and read raw, so a
    transposed copy of them raises ValueError too."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cc = get_compute_capability(torch.device("cuda"))
    if not mm_fp4.is_backend_supported("trtllm", cc[0] * 10 + cc[1]):
        pytest.skip("trtllm FP4 GEMM is not supported on this GPU.")
    m, n, k = 128, 256, 512
    torch.manual_seed(0)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    ga = (448 * 6) / a.float().abs().nan_to_num().max()
    gb = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(a, ga, sfLayout=SfLayout.layout_8x4, do_shuffle=False)
    b_fp4, b_sf = nvfp4_quantize(b, gb, sfLayout=SfLayout.layout_128x4, do_shuffle=True)

    def run(b_descale):
        out = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_descale,
            1.0 / (ga * gb),
            torch.bfloat16,
            out,
            block_size=16,
            use_8x4_sf_layout=True,
            backend="trtllm",
            skip_check=False,
        )
        return out

    run(b_sf.T)
    for b_descale in (b_sf.T.contiguous(), b_sf.T.contiguous().T):
        with pytest.raises(
            ValueError, match="the layout a transposed copy of the scales has"
        ):
            run(b_descale)


if __name__ == "__main__":
    pytest.main([__file__])


def test_mm_fp4_l2_policy_rule():
    """Weights load evict_first while they are streamed at most twice (<= 2
    token tiles); with more token tiles both operands of the no-swap kernel
    are pinned with evict_last; other kernels keep the default."""
    from flashinfer.gemm.gemm_mm_fp4_cute_dsl import mm_fp4_l2_policy

    # swap_ab: tokens are the kernel's N extent, weights the A operand
    assert mm_fp4_l2_policy(1, (128, 8), True, "sm100") == "a_ef"
    assert mm_fp4_l2_policy(32, (128, 32), True, "sm100") == "a_ef"
    assert mm_fp4_l2_policy(64, (128, 32), True, "sm100") == "a_ef"
    assert mm_fp4_l2_policy(65, (128, 32), True, "sm100") is None
    # no swap: tokens are the M extent, weights the B operand
    assert mm_fp4_l2_policy(130, (256, 128), False, "sm100") == "b_ef"
    assert mm_fp4_l2_policy(512, (256, 256), False, "sm100") == "b_ef"
    assert mm_fp4_l2_policy(513, (256, 256), False, "sm100") == "ab_el"
    assert mm_fp4_l2_policy(8192, (256, 256), False, "sm100") == "ab_el"
    # other kernels are untouched
    assert mm_fp4_l2_policy(8, (128, 8), True, "sm100sk") is None
    assert mm_fp4_l2_policy(8, (128, 8), True, "sm103") is None
