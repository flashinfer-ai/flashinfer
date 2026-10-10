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
from flashinfer.gemm.gemm_base import CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR


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


def _cutlass_scheduler_operands(m, n=1152, k=256):
    # Integer-valued FP4 inputs make both the dot product and scaled reference exact.
    values = torch.tensor([0, 1, -1, 2, -2], device="cuda", dtype=torch.float32)
    codes = torch.tensor([0, 2, 10, 4, 12], device="cuda", dtype=torch.uint8)
    column = torch.arange(k, device="cuda")
    a_index = (torch.arange(m, device="cuda")[:, None] + column * 3) % 5
    b_index = (torch.arange(n, device="cuda")[:, None] * 2 + column) % 5
    a_codes, b_codes = codes[a_index], codes[b_index]
    a = a_codes[:, ::2] | (a_codes[:, 1::2] << 4)
    b = b_codes[:, ::2] | (b_codes[:, 1::2] << 4)
    # Uniform E4M3 scale 1 has the same bytes in every 128x4 block-scale layout.
    a_scale = torch.full(
        ((m + 127) // 128 * 128, k // 16), 0x38, device="cuda", dtype=torch.uint8
    )
    b_scale = torch.full((n, k // 16), 0x38, device="cuda", dtype=torch.uint8)
    reference = values[a_index] @ values[b_index].T
    alpha = torch.tensor([1.25], device="cuda", dtype=torch.float32)
    return a, b, a_scale, b_scale, alpha, reference


@pytest.mark.parametrize("m", [4, 257, 1025])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_mm_fp4_cutlass_scheduler_tactics(m, out_dtype):
    """All legacy and appended schedules preserve spatial output and live GPU alpha."""
    if not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("CUTLASS scheduler variants require SM120/SM121.")
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4

    module = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
    assert module.fp4_gemm_tactic_num() == 64
    a, b, sa, sb, alpha, reference = _cutlass_scheduler_operands(m)
    out = torch.empty(reference.shape, device="cuda", dtype=out_dtype)
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    saved_inputs = [t.clone() for t in (a, b, sa, sb)]

    for tactic in [-1, *range(64)]:
        # The first 32 integer IDs remain valid, and each added schedule is explicit.
        module.fp4_gemm(a, b, sa, sb, alpha, out, workspace, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            module.fp4_gemm(a, b, sa, sb, alpha, out, workspace, tactic)
        for scale in (1.25, -0.75):
            alpha.fill_(scale)
            out.fill_(float("nan"))
            graph.replay()
            expected = (reference * scale).to(out_dtype)
            assert torch.isfinite(out).all()
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (
                tactic
            )
        for actual, saved in zip((a, b, sa, sb), saved_inputs, strict=True):
            assert torch.equal(actual, saved)


@pytest.mark.parametrize("managed_cache", [False, True])
def test_mm_fp4_cutlass_scheduler_autotune_cache(monkeypatch, tmp_path, managed_cache):
    """Public tuning selects an appended integer tactic and replays it after reload."""
    if not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("CUTLASS scheduler variants require SM120/SM121.")
    from flashinfer import autotune_v2
    from flashinfer.autotuner import AutoTuner

    monkeypatch.setattr(AutoTuner, "_instance", None)
    a, b, sa, sb, alpha, reference = _cutlass_scheduler_operands(4)
    profiled = []

    def profile(self, runner, inputs, tactic, tuning_config=None, **kwargs):
        profiled.append(tactic)
        runner(inputs, tactic=tactic)
        # Deterministic selection tests cache plumbing, not hardware performance.
        return 1.0 if tactic == 54 else 2.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", profile)
    context = (
        autotune_v2(cache_root=tmp_path / "managed", tuning_buckets=(4,))
        if managed_cache
        else autotune(True, tuning_buckets=(4,))
    )
    with context:
        out = mm_fp4(a, b.T, sa, sb.T, alpha, backend="cutlass")
    assert set(profiled) == (set(range(64)) | ({-1} if managed_cache else set()))
    expected = (reference * 1.25).to(torch.bfloat16)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))

    cache = tmp_path / "tactics.json"
    if not managed_cache:
        AutoTuner.get().save_configs(cache)
    monkeypatch.setattr(AutoTuner, "_instance", None)
    if not managed_cache:
        AutoTuner.get().load_configs(cache)
    calls = []
    original_choose = AutoTuner.choose_one

    def choose(self, *args, **kwargs):
        runner, tactic = original_choose(self, *args, **kwargs)
        calls.append(tactic)
        return runner, tactic

    monkeypatch.setattr(AutoTuner, "choose_one", choose)
    alpha.fill_(-0.75)
    replay = (
        autotune_v2(cache_root=tmp_path / "managed", mode="replay", tuning_buckets=(4,))
        if managed_cache
        else autotune(True, tuning_buckets=(4,))
    )
    with replay:
        out = mm_fp4(a, b.T, sa, sb.T, alpha, backend="cutlass")
    assert calls == [54]
    assert len(profiled) == (65 if managed_cache else 64)
    expected = (reference * -0.75).to(torch.bfloat16)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


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
