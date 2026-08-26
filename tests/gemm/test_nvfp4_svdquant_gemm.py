"""Tests for the NVFP4 SVDQuant GEMM ops (Blackwell):

- mm_nvfp4_svdquant     : out = alpha * (a @ bT) + d @ l1T [+ bias], with rank r a
                          positive multiple of 32 (32-128 covered here).
- nvfp4_quantize_smooth : NVFP4-quantize(x * pre_quant_scale).
- svdquant_linear       : the full quantize -> LoRA-down -> residual/correction chain.

SM100/SM103 use fused CUTLASS. SM120/SM121 use fused CuTe DSL, with an explicit
``cute-dsl-unfused`` composition retained as a differential oracle.
"""

import pytest
import torch

from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    mm_nvfp4_svdquant,
    nvfp4_quantize,
    nvfp4_quantize_smooth,
    svdquant_linear,
)
from flashinfer.gemm.gemm_svdquant import (
    DEFAULT_WORKSPACE_SIZE,
    SVDQUANT_LORA_RANK_GRANULARITY,
    get_nvfp4_svdquant_module,
)
from flashinfer.utils import device_support_pdl, get_compute_capability

_RANK = SVDQUANT_LORA_RANK_GRANULARITY  # base rank == the collective's rank granularity


def test_nvfp4_svdquant_backend_arch_support():
    for api in (mm_nvfp4_svdquant, nvfp4_quantize_smooth):
        assert api.is_backend_supported("cutlass", 100)
        assert api.is_backend_supported("cutlass", 103)
        assert not api.is_backend_supported("cutlass", 120)
        assert api.is_backend_supported("cute-dsl", 120)
        assert api.is_backend_supported("cute-dsl", 121)
        assert not api.is_backend_supported("cute-dsl", 100)
    assert mm_nvfp4_svdquant.is_backend_supported("cute-dsl-unfused", 120)
    assert mm_nvfp4_svdquant.is_backend_supported("cute-dsl-unfused", 121)
    assert not mm_nvfp4_svdquant.is_backend_supported("cute-dsl-unfused", 100)


def test_sm120_svdquant_kernel_iket_flag_defaults_off():
    pytest.importorskip("cutlass")
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
        Sm120B12xBlockScaledDenseGemmKernel,
    )

    assert not Sm120B12xBlockScaledDenseGemmKernel(16, (64, 64), (1, 1)).enable_iket
    assert Sm120B12xBlockScaledDenseGemmKernel(
        16, (64, 64), (1, 1), enable_iket=True
    ).enable_iket


def test_sm120_svdquant_can_implement_rejects_ragged_rank():
    cutlass = pytest.importorskip("cutlass")
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
        Sm120B12xBlockScaledDenseGemmKernel,
    )

    common_args = (
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.BFloat16,
        (64, 64),
        (1, 1),
        128,
        128,
        1,
        "k",
        "k",
        "n",
    )
    assert Sm120B12xBlockScaledDenseGemmKernel.can_implement(
        *common_args, svdquant_rank=32, tile_k=128
    )
    assert not Sm120B12xBlockScaledDenseGemmKernel.can_implement(
        *common_args, svdquant_rank=18, tile_k=128
    )
    assert not Sm120B12xBlockScaledDenseGemmKernel.can_implement(
        *common_args, svdquant_rank=32, tile_k=256
    )
    assert Sm120B12xBlockScaledDenseGemmKernel.can_implement(
        *common_args, svdquant_rank=64, tile_k=256
    )


def _skip_unless_sm100():
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip(
            "NVFP4 SVDQuant kernels require SM100-class GPUs, "
            f"got compute capability {compute_capability}."
        )


def _skip_unless_sm120():
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 12:
        pytest.skip(
            "SM120 SVDQuant CuTe DSL tests require SM120-class GPUs, "
            f"got compute capability {compute_capability}."
        )


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    err = (ref - got).float()
    noise = (err**2).mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10((ref.float() ** 2).mean() / noise))


def _assert_sm120_accuracy(ref: torch.Tensor, got: torch.Tensor) -> None:
    """Guard aggregate quality and localized spikes with measured margin.

    The review sweep measured a 53.27 dB SQNR floor and a 0.347% maximum
    error/reference-peak ceiling; historical tile coverage reached 48.99 dB.
    """
    ref_f32 = ref.float()
    got_f32 = got.float()
    assert _sqnr_db(ref_f32, got_f32) > 45.0
    peak = ref_f32.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
    normalized_max_error = (ref_f32 - got_f32).abs().amax() / peak
    assert normalized_max_error < 0.01


def _nvfp4_quantize_128x4(t: torch.Tensor, backend="cuda"):
    """Stock NVFP4 quantization (ue4m3 block scales, 128x4 swizzled layout).

    Returns (packed e2m1 uint8 [r, c/2], swizzled sf uint8 2-D, global scale f32 [1]).
    """
    global_sf = ((448.0 * 6.0) / t.float().abs().nan_to_num().max()).reshape(1)
    tq, sf = nvfp4_quantize(
        t,
        global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend=backend,
    )
    return tq.view(torch.uint8), sf.view(torch.uint8), global_sf


def _mm_fp4_residual(xq, wq, x_sf, w_sf, alpha, backend="cutlass"):
    """Reference residual alpha * (a @ bT) via a generic NVFP4 GEMM backend."""
    out = torch.empty(xq.shape[0], wq.shape[0], dtype=torch.bfloat16, device=xq.device)
    mm_fp4(
        xq,
        wq.T,
        x_sf,
        w_sf.T,
        alpha,
        torch.bfloat16,
        out,
        block_size=16,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=True,
    )
    return out.float()


def _sm120_unfused_reference(p, use_bias):
    """Reproduce the exact BF16 operation order of the SM120 unfused oracle."""
    out = torch.empty(
        p["xq"].shape[0],
        p["wq"].shape[0],
        dtype=torch.bfloat16,
        device=p["xq"].device,
    )
    mm_fp4(
        p["xq"],
        p["wq"].T,
        p["x_sf"],
        p["w_sf"].T,
        p["alpha"],
        torch.bfloat16,
        out,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="b12x",
        use_nvfp4=True,
    )
    correction = torch.mm(p["d"], p["l1_scaled"].T)
    correction.mul_(p["alpha"])
    out.add_(correction)
    if use_bias:
        out.add_(p["bias"])
    return out


def _make_gemm_problem(
    m,
    n,
    k,
    rank=_RANK,
    device="cuda",
    quant_backend="cuda",
    residual_backend="cutlass",
):
    """Quantized operands and fp32 references for out = alpha*(a@bT) + D@L1T [+ bias]."""
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    xq, x_sf, gx = _nvfp4_quantize_128x4(x, backend=quant_backend)
    wq, w_sf, gw = _nvfp4_quantize_128x4(w, backend=quant_backend)
    alpha = (1.0 / (gx * gw)).reshape(1).float()
    d = torch.randn(m, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    l1 = torch.randn(n, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    # 1/alpha is folded into L1 so the epilogue out = alpha*acc [+ bias] yields the
    # unscaled D @ L1T correction.
    l1_scaled = (l1.float() / alpha).to(torch.bfloat16).contiguous()
    bias = torch.randn(n, dtype=torch.bfloat16, device=device).contiguous()

    ref = (
        _mm_fp4_residual(xq, wq, x_sf, w_sf, alpha, backend=residual_backend)
        + d.float() @ l1.float().t()
    )
    return {
        "xq": xq,
        "wq": wq,
        "x_sf": x_sf,  # 2-D swizzled layout (mm_fp4 convention)
        "w_sf": w_sf,
        "x_sf_flat": x_sf.reshape(-1),  # 1-D buffer (fused-kernel convention)
        "w_sf_flat": w_sf.reshape(-1),
        "alpha": alpha,
        "d": d,
        "l1_scaled": l1_scaled,
        "bias": bias,
        "ref": ref,
        "ref_bias": ref + bias.float(),
    }


# n=3072 / n=12288 exercise the dedicated fast-path kernels; n=4096 the legacy
# (generic-width) kernel. m values cover token tails and non-multiple-of-128 rows
# (SF row padding).
@pytest.mark.parametrize("m", [44, 129, 256, 1000])
@pytest.mark.parametrize("n", [3072, 12288, 4096])
def test_nvfp4_quantize_smooth(m, n):
    _skip_unless_sm100()
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(m, n, dtype=torch.bfloat16, device=device) / (n**0.25)
    pqs = (
        (1.0 + 0.3 * torch.randn(n, dtype=torch.bfloat16, device=device))
        .abs()
        .contiguous()
    )
    global_sf = (
        ((448.0 * 6.0) / (x.float() * pqs.float()).abs().nan_to_num().max())
        .reshape(1)
        .contiguous()
    )

    # Reference: quantize the pre-smoothed activation with the stock quantizer. The
    # kernel multiplies x * pqs in bf16, so the reference product is bf16 as well.
    xq_ref, sf_ref = nvfp4_quantize(
        (x * pqs).to(torch.bfloat16),
        global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    xq, sf = nvfp4_quantize_smooth(x, pqs, global_sf)

    assert xq.dtype == torch.uint8 and xq.shape == (m, n // 2)
    assert sf.dtype == torch.uint8 and sf.ndim == 1
    assert torch.equal(xq, xq_ref.view(torch.uint8))
    assert torch.equal(sf, sf_ref.view(torch.uint8).reshape(-1))


# Rank 32 (one storage chunk, the original kernel) and rank 128 (the widest validated
# rank: 4 chunks on K128 tiles, 2 on K256 tiles) sweep every tactic; the intermediate
# ranks are covered by test_mm_nvfp4_svdquant_rank_chunks on representative tactics.
@pytest.mark.parametrize("m", [129, 6912])
@pytest.mark.parametrize("k", [3072, 12288])
@pytest.mark.parametrize("rank", [32, 128])
def test_mm_nvfp4_svdquant_per_tactic(m, k, rank):
    _skip_unless_sm100()
    torch.manual_seed(0)
    n = 3072
    device = torch.device("cuda")
    p = _make_gemm_problem(m, n, k, rank=rank)

    module = get_nvfp4_svdquant_module()
    num_tactics = int(module.nvfp4_svdquant_gemm_tactic_num())
    assert num_tactics > 0
    enable_pdl = device_support_pdl(device)
    workspace = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=device)

    for tactic in range(num_tactics):
        # Bias is the production epilogue: exercise it for every tactic. The no-bias
        # epilogue is orthogonal to M, so cover it on the small-M problem only.
        for use_bias in [True, False] if m == 129 else [True]:
            out.fill_(float("nan"))
            module.nvfp4_svdquant_gemm(
                p["xq"],
                p["wq"],
                p["x_sf_flat"],
                p["w_sf_flat"],
                p["alpha"],
                p["d"],
                p["l1_scaled"],
                p["bias"] if use_bias else None,
                out,
                workspace,
                tactic,
                enable_pdl,
            )
            ref = p["ref_bias"] if use_bias else p["ref"]
            sqnr = _sqnr_db(ref, out.float())
            assert sqnr > 40.0, (
                f"tactic={tactic} use_bias={use_bias} m={m} n={n} k={k} rank={rank}: "
                f"SQNR={sqnr:.2f} dB <= 40 dB"
            )


# Chunked-rank coverage on representative kernel shapes: tactics 0/1 use K128 tiles
# (32-column chunks: rank 64 -> 2 chunks, 96 -> 3), tactics 19/25 use K256 tiles
# (64-column chunks: rank 64 -> 1 full-width chunk, 96 -> a full chunk plus a
# half-real TMA-zero-filled tail). Rank 128 everywhere is covered by the full
# per-tactic sweep above.
@pytest.mark.parametrize("m", [129, 6912])
@pytest.mark.parametrize("rank", [64, 96])
@pytest.mark.parametrize("tactic", [0, 1, 19, 25])
def test_mm_nvfp4_svdquant_rank_chunks(m, rank, tactic):
    _skip_unless_sm100()
    torch.manual_seed(0)
    n, k = 3072, 3072
    device = torch.device("cuda")
    p = _make_gemm_problem(m, n, k, rank=rank)

    module = get_nvfp4_svdquant_module()
    enable_pdl = device_support_pdl(device)
    workspace = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    module.nvfp4_svdquant_gemm(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        p["bias"],
        out,
        workspace,
        tactic,
        enable_pdl,
    )
    sqnr = _sqnr_db(p["ref_bias"], out.float())
    assert sqnr > 40.0, (
        f"tactic={tactic} m={m} rank={rank}: SQNR={sqnr:.2f} dB <= 40 dB"
    )


def test_mm_nvfp4_svdquant_rejects_bad_rank():
    _skip_unless_sm100()
    torch.manual_seed(0)
    m, n, k = 128, 3072, 3072
    p = _make_gemm_problem(m, n, k, rank=64)
    for d, l1 in [
        (p["d"][:, :48].contiguous(), p["l1_scaled"][:, :48].contiguous()),  # not %32
        (p["d"], p["l1_scaled"][:, :32].contiguous()),  # rank mismatch
    ]:
        with pytest.raises(ValueError):
            mm_nvfp4_svdquant(
                p["xq"],
                p["wq"],
                p["x_sf_flat"],
                p["w_sf_flat"],
                p["alpha"],
                d,
                l1,
                bias=p["bias"],
            )


def test_mm_nvfp4_svdquant_sm100_pooled_alpha_uses_first_element():
    _skip_unless_sm100()
    torch.manual_seed(0)
    p = _make_gemm_problem(129, 3072, 3072, rank=32)
    expected = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
        backend="cutlass",
    )
    pooled_alpha = torch.cat([p["alpha"], torch.tensor([2.0, 3.0, 4.0], device="cuda")])
    actual = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        pooled_alpha,
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
        backend="cutlass",
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("m", [129, 6912])
@pytest.mark.parametrize("rank", [32, 96])
def test_mm_nvfp4_svdquant_autotuned(m, rank):
    _skip_unless_sm100()
    torch.manual_seed(0)
    n, k = 3072, 3072
    p = _make_gemm_problem(m, n, k, rank=rank)

    with autotune(True):
        out = mm_nvfp4_svdquant(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            bias=p["bias"],
        )
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert _sqnr_db(p["ref_bias"], out.float()) > 40.0

    # Replay outside the tuning context: the cached tactic must also be correct.
    out_replay = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
    )
    assert _sqnr_db(p["ref_bias"], out_replay.float()) > 40.0


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("rank", [_RANK, 64])
def test_svdquant_linear_matches_reference(use_bias, rank):
    _skip_unless_sm100()
    torch.manual_seed(0)
    m, n, k = 129, 3072, 3072
    device = "cuda"

    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    pqs = (
        (1.0 + 0.3 * torch.randn(k, dtype=torch.bfloat16, device=device))
        .abs()
        .contiguous()
    )
    smoothed = (x * pqs).to(torch.bfloat16)
    global_sf = (
        ((448.0 * 6.0) / smoothed.float().abs().nan_to_num().max())
        .reshape(1)
        .contiguous()
    )

    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    wq, w_sf, gw = _nvfp4_quantize_128x4(w)
    alpha = (1.0 / (global_sf * gw)).reshape(1).float()

    lora_a = torch.randn(rank, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    l2t_smoothed = (pqs.unsqueeze(1) * lora_a.t()).contiguous()  # [k, rank] bf16
    lora_b = torch.randn(n, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    l1_scaled = (lora_b.float() / alpha).to(torch.bfloat16).contiguous()
    bias = (
        torch.randn(n, dtype=torch.bfloat16, device=device).contiguous()
        if use_bias
        else None
    )

    out = svdquant_linear(
        x,
        wq,
        w_sf.reshape(-1),
        alpha,
        pqs,
        l2t_smoothed,
        l1_scaled,
        global_sf,
        bias=bias,
    )

    # Unfused reference on byte-identical quantized operands (nvfp4_quantize_smooth
    # is byte-identical to the stock quantizer on the pre-smoothed input).
    xq_ref, x_sf_ref = nvfp4_quantize(
        smoothed, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    residual = _mm_fp4_residual(
        xq_ref.view(torch.uint8), wq, x_sf_ref.view(torch.uint8), w_sf, alpha
    )
    down = torch.mm(x, l2t_smoothed)  # same bf16 LoRA-down GEMM the chain runs
    ref = residual + down.float() @ lora_b.float().t()
    if bias is not None:
        ref = ref + bias.float()

    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert _sqnr_db(ref, out.float()) > 40.0


@pytest.mark.parametrize("m,k", [(129, 256), (129, 12288), (70, 3088)])
def test_nvfp4_quantize_smooth_sm120_cute_dsl(m, k):
    _skip_unless_sm120()
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    pqs = (
        (1.0 + 0.3 * torch.randn(k, dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    smoothed = (x * pqs).to(torch.bfloat16)
    global_sf = (
        ((448.0 * 6.0) / smoothed.float().abs().nan_to_num().max())
        .reshape(1)
        .contiguous()
    )
    xq_ref, sf_ref = nvfp4_quantize(
        smoothed,
        global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )

    xq, sf = nvfp4_quantize_smooth(x, pqs, global_sf, backend="cute-dsl")
    xq_auto, sf_auto = nvfp4_quantize_smooth(x, pqs, global_sf)

    assert xq.dtype == torch.uint8 and xq.shape == (m, k // 2)
    assert sf.dtype == torch.uint8 and sf.ndim == 1
    assert torch.equal(xq, xq_ref.view(torch.uint8))
    assert torch.equal(sf, sf_ref.view(torch.uint8).reshape(-1))
    assert torch.equal(xq_auto, xq)
    assert torch.equal(sf_auto, sf)


def test_nvfp4_quantize_smooth_sm120_misaligned_inputs():
    _skip_unless_sm120()
    torch.manual_seed(0)
    m, k = 70, 256
    x_storage = torch.randn(m * k + 1, dtype=torch.bfloat16, device="cuda")
    x = x_storage[1:].view(m, k)
    pqs_storage = torch.randn(k + 1, dtype=torch.bfloat16, device="cuda").abs()
    pqs = pqs_storage[1:]
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    assert pqs.is_contiguous() and pqs.data_ptr() % 16 != 0
    smoothed = (x * pqs).to(torch.bfloat16)
    global_sf = ((448.0 * 6.0) / smoothed.float().abs().max()).reshape(1)
    expected_q, expected_sf = nvfp4_quantize(
        smoothed,
        global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    actual_q, actual_sf = nvfp4_quantize_smooth(x, pqs, global_sf, backend="cute-dsl")
    assert torch.equal(actual_q, expected_q.view(torch.uint8))
    assert torch.equal(actual_sf, expected_sf.view(torch.uint8).reshape(-1))


@pytest.mark.parametrize("m,k", [(0, 256), (4, 0)])
def test_nvfp4_quantizers_sm120_empty_inputs(m, k):
    _skip_unless_sm120()
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_cute_dsl,
    )

    x = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
    pqs = torch.ones(k, dtype=torch.bfloat16, device="cuda")
    global_sf = torch.ones(1, dtype=torch.float32, device="cuda")
    xq, sf = nvfp4_quantize_smooth(x, pqs, global_sf, backend="cute-dsl")
    plain_xq, plain_sf = nvfp4_quantize_cute_dsl(x, global_sf)

    assert xq.shape == (m, k // 2) and xq.dtype == torch.uint8
    assert sf.shape == (0,) and sf.dtype == torch.uint8
    assert plain_xq.shape == (m, k // 2) and plain_xq.dtype == torch.uint8
    assert plain_sf.shape == (m, ((k // 16 + 3) // 4) * 4)
    assert plain_sf.dtype == torch.uint8


@pytest.mark.parametrize("use_bias", [False, True])
def test_mm_nvfp4_svdquant_sm120_fused(use_bias):
    _skip_unless_sm120()
    torch.manual_seed(0)
    m, n, k, rank = 129, 256, 256, 32
    p = _make_gemm_problem(
        m,
        n,
        k,
        rank=rank,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )
    bias = p["bias"] if use_bias else None
    expected = _sm120_unfused_reference(p, use_bias)

    out_buffer = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=bias,
        out=out_buffer,
        backend="cute-dsl",
    )
    out_auto = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=bias,
    )

    assert out.data_ptr() == out_buffer.data_ptr()
    # The fused path accumulates the BF16 rank correction in FP32 before its
    # single BF16 store, whereas the oracle rounds the residual and correction
    # in separate launches. Compare numerically, not bitwise.
    _assert_sm120_accuracy(expected, out)
    _assert_sm120_accuracy(expected, out_auto)
    fp32_ref = p["ref_bias"] if use_bias else p["ref"]
    _assert_sm120_accuracy(fp32_ref, out)


@pytest.mark.parametrize(
    "tactic,rank",
    [
        (((64, 64), 128, False), 32),
        (((128, 128), 128, False), 64),
        (((256, 64), 128, False), 32),
    ],
)
def test_mm_nvfp4_svdquant_sm120_large_m_tactic(tactic, rank):
    _skip_unless_sm120()
    from flashinfer.gemm.gemm_svdquant import _mm_nvfp4_svdquant_sm120_fused

    torch.manual_seed(tactic[0][0])
    p = _make_gemm_problem(
        257,
        128,
        512,
        rank=rank,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )
    out = torch.empty(257, 128, dtype=torch.bfloat16, device="cuda")
    _mm_nvfp4_svdquant_sm120_fused(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        p["bias"],
        out,
        device_support_pdl(torch.device("cuda")),
        tactic=tactic,
    )
    _assert_sm120_accuracy(p["ref_bias"], out)


@pytest.mark.parametrize("backend", ["cute-dsl", "auto"])
def test_mm_nvfp4_svdquant_sm120_autotuned_replay(backend):
    _skip_unless_sm120()
    torch.manual_seed(0)
    p = _make_gemm_problem(
        129,
        256,
        256,
        rank=32,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )

    with autotune(True):
        out = mm_nvfp4_svdquant(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            bias=p["bias"],
            backend=backend,
        )
    _assert_sm120_accuracy(p["ref_bias"], out)

    # Replay outside the tuning context must reuse the selected tactic.
    out_replay = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
        backend=backend,
    )
    assert torch.equal(out_replay, out)


def test_mm_nvfp4_svdquant_sm120_unfused_oracle():
    _skip_unless_sm120()
    torch.manual_seed(0)
    p = _make_gemm_problem(
        33,
        128,
        128,
        rank=32,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )
    expected = _sm120_unfused_reference(p, True)
    out = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
        backend="cute-dsl-unfused",
    )
    assert torch.equal(out, expected)


@pytest.mark.parametrize("rank", [32, 64, 96, 128])
def test_mm_nvfp4_svdquant_sm120_fused_rank_chunks(rank):
    _skip_unless_sm120()
    torch.manual_seed(rank)
    p = _make_gemm_problem(
        33,
        128,
        128,
        rank=rank,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )
    out = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
        backend="cute-dsl",
    )
    _assert_sm120_accuracy(p["ref_bias"], out)


@pytest.mark.parametrize(
    "m,n,k,rank",
    [
        (1, 512, 4096, 32),  # (64, 32) tile with swap_ab=True
        (33, 160, 192, 64),  # partial N tile and ragged K mainloop
    ],
)
def test_mm_nvfp4_svdquant_sm120_fused_boundary_plans(m, n, k, rank):
    _skip_unless_sm120()
    torch.manual_seed(m + n + k + rank)
    p = _make_gemm_problem(
        m,
        n,
        k,
        rank=rank,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )
    out = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
        backend="cute-dsl",
    )
    _assert_sm120_accuracy(p["ref_bias"], out)


@pytest.mark.parametrize("alpha_shape", [(), (1, 1), (4,)])
@pytest.mark.parametrize("backend", ["cute-dsl", "cute-dsl-unfused"])
def test_mm_nvfp4_svdquant_sm120_normalizes_alpha_shape(alpha_shape, backend):
    _skip_unless_sm120()
    torch.manual_seed(0)
    p = _make_gemm_problem(
        33,
        128,
        128,
        rank=32,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )
    expected = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
        backend=backend,
    )
    alpha = (
        p["alpha"].reshape(alpha_shape)
        if alpha_shape != (4,)
        else torch.cat([p["alpha"], torch.tensor([2.0, 3.0, 4.0], device="cuda")])
    )
    out = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        alpha,
        p["d"],
        p["l1_scaled"],
        bias=p["bias"],
        backend=backend,
    )
    assert torch.equal(out, expected)


def test_mm_nvfp4_svdquant_rejects_noncontiguous_packed_inputs():
    _skip_unless_sm120()
    torch.manual_seed(0)
    p = _make_gemm_problem(
        33,
        128,
        128,
        rank=32,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )
    storage = torch.empty(
        p["xq"].shape[0],
        p["xq"].shape[1] * 2,
        dtype=torch.uint8,
        device="cuda",
    )
    storage[:, ::2].copy_(p["xq"])
    a_noncontiguous = storage[:, ::2]
    assert not a_noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="a and b must be contiguous"):
        mm_nvfp4_svdquant(
            a_noncontiguous,
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            backend="cute-dsl",
        )


def test_mm_nvfp4_svdquant_sm120_fused_does_not_call_torch_mm(monkeypatch):
    _skip_unless_sm120()
    torch.manual_seed(0)
    p = _make_gemm_problem(
        33,
        128,
        128,
        rank=32,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )

    def fail_torch_mm(*args, **kwargs):
        raise AssertionError("the fused SM120 SVDQuant path called torch.mm")

    monkeypatch.setattr(torch, "mm", fail_torch_mm)
    out = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        backend="cute-dsl",
    )
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("use_bias", [False, True])
def test_svdquant_linear_sm120_fused(use_bias, monkeypatch):
    _skip_unless_sm120()
    torch_mm_calls = 0
    original_torch_mm = torch.mm

    def recording_torch_mm(*args, **kwargs):
        nonlocal torch_mm_calls
        torch_mm_calls += 1
        return original_torch_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "mm", recording_torch_mm)
    torch.manual_seed(0)
    m, n, k, rank = 129, 256, 256, 32
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / (k**0.25)
    pqs = (
        (1.0 + 0.3 * torch.randn(k, dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    smoothed = (x * pqs).to(torch.bfloat16)
    global_sf = (
        ((448.0 * 6.0) / smoothed.float().abs().nan_to_num().max())
        .reshape(1)
        .contiguous()
    )
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / (k**0.25)
    wq, w_sf, gw = _nvfp4_quantize_128x4(w, backend="cute-dsl")
    alpha = (1.0 / (global_sf * gw)).reshape(1).float()
    lora_a = torch.randn(rank, k, dtype=torch.bfloat16, device="cuda") / (k**0.25)
    l2t_smoothed = (pqs.unsqueeze(1) * lora_a.T).contiguous()
    lora_b = torch.randn(n, rank, dtype=torch.bfloat16, device="cuda") / (rank**0.25)
    l1_scaled = (lora_b.float() / alpha).to(torch.bfloat16).contiguous()
    bias = (
        torch.randn(n, dtype=torch.bfloat16, device="cuda").contiguous()
        if use_bias
        else None
    )

    out = svdquant_linear(
        x,
        wq,
        w_sf.reshape(-1),
        alpha,
        pqs,
        l2t_smoothed,
        l1_scaled,
        global_sf,
        bias=bias,
        backend="cute-dsl",
    )
    assert torch_mm_calls == 1

    xq, x_sf = nvfp4_quantize(
        smoothed,
        global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    residual = _mm_fp4_residual(
        xq.view(torch.uint8),
        wq,
        x_sf.view(torch.uint8),
        w_sf,
        alpha,
        backend="b12x",
    )
    down = torch.mm(x, l2t_smoothed)
    ref = residual + down.float() @ lora_b.float().T
    if bias is not None:
        ref.add_(bias.float())

    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    _assert_sm120_accuracy(ref, out)


@pytest.mark.parametrize("rank", [32, 128])
def test_mm_nvfp4_svdquant_cuda_graph(rank):
    _skip_unless_sm100()
    torch.manual_seed(0)
    m, n, k = 129, 3072, 3072
    device = torch.device("cuda")

    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    xq, x_sf2d, gx = _nvfp4_quantize_128x4(x)
    wq, w_sf, gw = _nvfp4_quantize_128x4(w)
    x_sf = x_sf2d.reshape(-1)
    w_sf_flat = w_sf.reshape(-1)
    alpha = (1.0 / (gx * gw)).reshape(1).float()
    d = torch.randn(m, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    l1 = torch.randn(n, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    l1_scaled = (l1.float() / alpha).to(torch.bfloat16).contiguous()
    bias = torch.randn(n, dtype=torch.bfloat16, device=device).contiguous()

    module = get_nvfp4_svdquant_module()
    enable_pdl = device_support_pdl(device)
    workspace = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    out_graph = torch.empty(m, n, dtype=torch.bfloat16, device=device)

    def run(out_tensor):
        # Fixed tactic 0 keeps eager and captured launches identical.
        module.nvfp4_svdquant_gemm(
            xq,
            wq,
            x_sf,
            w_sf_flat,
            alpha,
            d,
            l1_scaled,
            bias,
            out_tensor,
            workspace,
            0,
            enable_pdl,
        )

    # Warm up on a side stream so JIT loading and allocations happen outside capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(out_graph)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out_graph)

    # Refresh the captured input buffers in-place: the replay must consume the
    # current buffer contents, not the values seen at capture time.
    x_new = torch.randn(m, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    xq_new, x_sf_new = nvfp4_quantize(
        x_new, gx, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    xq.copy_(xq_new.view(torch.uint8))
    x_sf.copy_(x_sf_new.view(torch.uint8).reshape(-1))
    d.copy_(torch.randn(m, rank, dtype=torch.bfloat16, device=device) / (rank**0.25))

    out_graph.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    out_eager = torch.empty_like(out_graph)
    run(out_eager)
    torch.cuda.synchronize()

    # Same tactic and operands: the deterministic kernel must match bit-exactly.
    assert torch.equal(out_graph, out_eager)


@pytest.mark.parametrize("backend", ["cute-dsl", "auto"])
def test_mm_nvfp4_svdquant_sm120_cuda_graph_replay(backend):
    _skip_unless_sm120()
    torch.manual_seed(0)
    p = _make_gemm_problem(
        129,
        256,
        256,
        rank=32,
        quant_backend="cute-dsl",
        residual_backend="b12x",
    )
    out = torch.empty(129, 256, dtype=torch.bfloat16, device="cuda")

    def run():
        mm_nvfp4_svdquant(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            bias=p["bias"],
            out=out,
            backend=backend,
        )

    # Compile and tune before capture; capture must only replay the cached path.
    with autotune(True):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_sm120_accuracy(p["ref_bias"], out)


if __name__ == "__main__":
    pytest.main([__file__])
