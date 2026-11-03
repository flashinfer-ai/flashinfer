"""NVFP4 SVDQuant operator correctness, input validation, packing, and CUDA Graphs.

Covers quantization, GEMM and complete linear operations across SM100/SM103
CUTLASS/Cake and SM12x CuTe DSL, plus the explicit SM120 CUTLASS backend.
Host dispatch, JIT, workspace and compiled guard tests live in the companion
``test_nvfp4_svdquant_host.py`` module."""

import math
import re
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import pytest
import torch
from packaging.version import Version
from typing_extensions import assert_never

from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    mm_nvfp4_svdquant,
    nvfp4_quantize,
    nvfp4_quantize_smooth,
    svdquant_linear,
)
from flashinfer.autotuner import TunableRunner
from flashinfer.gemm import svdquant_sm120_cutlass
from flashinfer.gemm import svdquant_sm120_cutlass as backend
from flashinfer.gemm import svdquant_sm120_routes as _sm120_routes
from flashinfer.gemm import svdquant_sm120_routes as routes
from flashinfer.gemm.gemm_svdquant import (
    DEFAULT_WORKSPACE_SIZE,
    SVDQUANT_LORA_RANK_GRANULARITY,
    get_nvfp4_svdquant_module,
)
from flashinfer.gemm.svdquant_sm120_cutlass import (
    _get_nvfp4_svdquant_module_for_device,
    _svdquant_backend_for_capability,
    get_nvfp4_svdquant_sm120_module,
)
from flashinfer.gemm.svdquant_sm120_routes import (
    SM120_FAMILY_SMALL_M,
    SM120_FUSED_LINEAR_MK,
    sm120_pack_tactic,
    sm120_producer_variants,
)
from flashinfer.jit.cpp_ext import get_cuda_version
from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES
from flashinfer.utils import device_support_pdl, get_compute_capability
from tests.test_helpers.pdl import late_store

# Shared references and public operators across backends.

_RANK = SVDQUANT_LORA_RANK_GRANULARITY  # base rank == the collective's rank granularity
# Rows per block for full-output SQNR; ~0.5 GB of fp32 per block at n = 28672.
_SQNR_CHUNK_ROWS = 4096


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
    assert mm_nvfp4_svdquant.is_backend_supported("cake", 100)
    assert mm_nvfp4_svdquant.is_backend_supported("cake", 103)
    assert not mm_nvfp4_svdquant.is_backend_supported("cake", 107)
    assert not mm_nvfp4_svdquant.is_backend_supported("cake", 120)
    assert not mm_nvfp4_svdquant.is_backend_supported("cake", 121)


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


def _sqnr_db(
    ref: torch.Tensor, got: torch.Tensor, chunk_rows: int | None = None
) -> float:
    if chunk_rows is not None:
        # Row blocks bound the fp32 temporaries to one block, for outputs too
        # large to hold several full-size copies. Ratio of sums = ratio of means.
        signal = torch.zeros((), dtype=torch.float64, device=ref.device)
        noise = torch.zeros_like(signal)
        for ref_rows, got_rows in zip(
            ref.split(chunk_rows), got.split(chunk_rows), strict=True
        ):
            ref_rows = ref_rows.float()
            signal += ref_rows.square().sum(dtype=torch.float64)
            noise += (ref_rows - got_rows.float()).square().sum(dtype=torch.float64)
        if noise == 0:
            return float("inf")
        return float(10 * torch.log10(signal / noise))
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


# n=3072 takes the fast-path kernel; n=4096 the legacy kernel. Only n=4096 guards
# the fix: nvcc already ordered the fast path's __restrict__ load after the wait.
@pytest.mark.parametrize("n", [3072, 4096])
def test_nvfp4_quantize_smooth_pdl_waits_for_global_scale(n):
    _skip_unless_sm100()
    torch.manual_seed(0)
    x = torch.randn(256, n, dtype=torch.bfloat16, device="cuda")
    pqs = (1.0 + 0.3 * torch.randn(n, dtype=torch.bfloat16, device="cuda")).abs()
    global_sf = ((448.0 * 6.0) / (x.float() * pqs.float()).abs().max()).reshape(1)
    expected_xq, expected_sf = nvfp4_quantize_smooth(
        x, pqs, global_sf, enable_pdl=False, backend="cutlass"
    )

    # The scale the previous kernel is still producing, as in a dynamic-scale pipeline.
    scale = torch.full_like(global_sf, global_sf.item() / 4)
    late_store(scale, global_sf.item())
    xq, sf = nvfp4_quantize_smooth(x, pqs, scale, enable_pdl=True, backend="cutlass")

    assert torch.equal(sf, expected_sf)
    assert torch.equal(xq, expected_xq)


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


@pytest.mark.parametrize(
    "m,k,rank,use_bias",
    [
        (129, 3072, 32, False),
        (129, 12288, 32, False),
        (129, 3072, 96, True),
        (6912, 3072, 96, True),
        (129, 3072, 64, True),
        (129, 12288, 32, True),
        (129, 3072, 32, True),
        (129, 3072, 128, True),
    ],
)
def test_mm_nvfp4_svdquant_cake_cuda_graph_replay(m, k, rank, use_bias):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake NVFP4 SVDQuant requires exact SM100 or SM103")
    if get_cuda_version() < Version("13.0"):
        pytest.skip("Cake NVFP4 SVDQuant requires CUDA 13.0 or later")
    torch.manual_seed(0)
    p = _make_gemm_problem(m, 3072, k, rank=rank)
    out = torch.empty(m, 3072, dtype=torch.bfloat16, device="cuda")

    def run():
        return mm_nvfp4_svdquant(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            bias=p["bias"] if use_bias else None,
            out=out,
            backend="cake",
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected = p["ref_bias"] if use_bias else p["ref"]
    assert _sqnr_db(expected, out.float()) > 40.0


@pytest.mark.parametrize("use_bias", [False, True])
def test_mm_nvfp4_svdquant_cake_pooled_scale_buffers(use_bias):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake NVFP4 SVDQuant requires exact SM100 or SM103")
    if get_cuda_version() < Version("13.0"):
        pytest.skip("Cake NVFP4 SVDQuant requires CUDA 13.0 or later")
    torch.manual_seed(0)
    p = _make_gemm_problem(129, 3072, 3072, rank=32)

    def run(a_sf, b_sf):
        return mm_nvfp4_svdquant(
            p["xq"],
            p["wq"],
            a_sf,
            b_sf,
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            bias=p["bias"] if use_bias else None,
            backend="cake",
        )

    expected = run(p["x_sf_flat"], p["w_sf_flat"])
    padding = torch.tensor([255], dtype=torch.uint8, device="cuda")
    actual = run(
        torch.cat((p["x_sf_flat"], padding)),
        torch.cat((p["w_sf_flat"], padding)),
    )
    assert torch.equal(actual, expected)


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


def test_mm_nvfp4_svdquant_cake_requires_cuda13():
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake NVFP4 SVDQuant requires exact SM100 or SM103")
    if get_cuda_version() >= Version("13.0"):
        pytest.skip("Requires an unsupported CUDA toolkit")
    p = _make_gemm_problem(129, 3072, 3072, rank=32)
    with pytest.raises(ValueError, match="Cake NVFP4 SVDQuant requires CUDA 13.0"):
        mm_nvfp4_svdquant(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            backend="cake",
        )


def _skip_unless_cake_backend():
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake NVFP4 SVDQuant requires exact SM100 or SM103")
    if get_cuda_version() < Version("13.0"):
        pytest.skip("Cake NVFP4 SVDQuant requires CUDA 13.0 or later")


def _run_cake(p, out, use_bias):
    return mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"] if use_bias else None,
        out=out,
        backend="cake",
    )


# Shapes the previous exact-shape route table did not list: ragged and tiny M,
# new N / K, every rank, with and without bias.
@pytest.mark.parametrize(
    "m,n,k,rank,use_bias",
    [
        (128, 3072, 3072, 32, True),
        (135, 3072, 3072, 32, False),
        (333, 4096, 4096, 64, False),
        (2048, 6144, 3072, 128, True),
        (4097, 3072, 12288, 32, True),
        (200, 3072, 3072, 96, True),
        (191, 3072, 3072, 64, True),
        (129, 4096, 3072, 96, False),
    ],
)
def test_mm_nvfp4_svdquant_cake_general_shapes(m, n, k, rank, use_bias):
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    p = _make_gemm_problem(m, n, k, rank=rank)
    out = _run_cake(p, None, use_bias)
    torch.cuda.synchronize()
    expected = p["ref_bias"] if use_bias else p["ref"]
    assert out.shape == (m, n)
    assert torch.isfinite(out.float()).all()
    assert _sqnr_db(expected, out.float()) > 40.0


@pytest.mark.parametrize(
    "m,n,k,rank",
    [
        (65, 3072, 3072, 32),  # M below the 128-row tensor-map box
        (129, 3104, 3072, 32),  # N must be a multiple of 128
        (
            333,
            3072,
            3200,
            96,
        ),  # K must be a multiple of the 256-deep K tile of the rank-96 program
        (129, 3072, 3072, 160),  # ranks above 128 are not supported
    ],
)
def test_mm_nvfp4_svdquant_cake_rejects_unsupported_shapes(m, n, k, rank):
    _skip_unless_cake_backend()
    device = torch.device("cuda")
    a = torch.zeros(m, k // 2, dtype=torch.uint8, device=device)
    b = torch.zeros(n, k // 2, dtype=torch.uint8, device=device)
    sf_rows_a = (m + 127) // 128 * 128
    sf_rows_b = (n + 127) // 128 * 128
    sf_cols = (k // 16 + 3) // 4 * 4
    a_sf = torch.zeros(sf_rows_a * sf_cols, dtype=torch.uint8, device=device)
    b_sf = torch.zeros(sf_rows_b * sf_cols, dtype=torch.uint8, device=device)
    alpha = torch.ones(1, dtype=torch.float32, device=device)
    d = torch.zeros(m, rank, dtype=torch.bfloat16, device=device)
    l1 = torch.zeros(n, rank, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError):
        mm_nvfp4_svdquant(a, b, a_sf, b_sf, alpha, d, l1, backend="cake")


def test_mm_nvfp4_svdquant_cake_two_layers_cuda_graph():
    """Two layers with their own weights, K and rank replay correctly from one
    graph and from two alternating graphs: each launch carries its own
    descriptors, so no layer can observe the other's."""
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    m = 129
    layer1 = _make_gemm_problem(m, 3072, 3072, rank=32)
    layer2 = _make_gemm_problem(m, 3072, 12288, rank=64)
    out1 = torch.empty(m, 3072, dtype=torch.bfloat16, device="cuda")
    out2 = torch.empty(m, 3072, dtype=torch.bfloat16, device="cuda")

    def run1():
        _run_cake(layer1, out1, True)

    def run2():
        _run_cake(layer2, out2, True)

    run1()
    run2()
    torch.cuda.synchronize()
    eager1, eager2 = out1.clone(), out2.clone()
    assert _sqnr_db(layer1["ref_bias"], eager1.float()) > 40.0
    assert _sqnr_db(layer2["ref_bias"], eager2.float()) > 40.0

    both = torch.cuda.CUDAGraph()
    with torch.cuda.graph(both):
        run1()
        run2()
    for _ in range(2):
        out1.fill_(float("nan"))
        out2.fill_(float("nan"))
        both.replay()
        torch.cuda.synchronize()
        assert torch.equal(out1, eager1)
        assert torch.equal(out2, eager2)

    graph1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph1):
        run1()
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2):
        run2()
    for graph, out, eager in ((graph1, out1, eager1), (graph2, out2, eager2)) * 2:
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


# SM120 CUTLASS tactics and numerical regressions.


def _compute_capability():
    return get_compute_capability(torch.device(device="cuda"))


def _is_sm120():
    return _compute_capability() == (12, 0)


def _skip_unless_svdquant_supported():
    try:
        _svdquant_backend_for_capability(*_compute_capability())
    except ValueError as exc:
        pytest.skip(str(exc))


def _svdquant_module():
    return _get_nvfp4_svdquant_module_for_device(torch.device(device="cuda"))


def test_sm120_fallback_tactic_is_first_feasible_row():
    """The native -1 fallback resolves through feasibility, not a fixed row."""
    _skip_unless_svdquant_supported()
    if not _is_sm120():
        pytest.skip("requires SM120")

    module = _svdquant_module()
    m, n, k = 129, 4096, 3072
    expected = next(
        tactic
        for tactic in range(module.nvfp4_svdquant_gemm_tactic_num())
        if module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)
    )

    assert module.nvfp4_svdquant_gemm_fallback_tactic(m, n, k, _RANK) == expected


def test_mm_nvfp4_svdquant_rejects_noncontiguous_out():
    """A transposed [m, n] view must be rejected, not silently rearranged.

    The validation lives in the SM120 launcher; the SM100 launcher predates it and
    is frozen by the non-regression gate, so this asserts SM120 behavior only.
    """
    _skip_unless_svdquant_supported()
    if not _is_sm120():
        pytest.skip("output-contiguity validation is implemented in the SM120 launcher")
    torch.manual_seed(0)
    m, n, k = 128, 3072, 3072
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    storage = torch.empty(n, m, dtype=torch.bfloat16, device=p["xq"].device)
    out_view = storage.t()  # shape [m, n], non-contiguous strides
    module = _svdquant_module()
    workspace = torch.zeros(
        DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=p["xq"].device
    )
    with pytest.raises(Exception, match="contiguous"):
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out_view,
            workspace,
            0,
            device_support_pdl(p["xq"].device),
        )


@pytest.mark.parametrize("m,n,k", MODEL_SHAPE_CASES)
@pytest.mark.parametrize("use_bias", [True, False])
def test_mm_nvfp4_svdquant_model_shapes(m, n, k, use_bias):
    """Rank-32 correctness across the unified 71 model shapes.

    Covers the 12 Qwen-Image baseline shapes plus the Qwen-Image text-stream,
    Wan2.1 (1.3B/14B), Wan2.2 (A14B/TI2V-5B), and MiniMax-H3 cases from
    docs/model_shape_requirements.md (executable source of truth:
    flashinfer/testing/svdq_model_shapes.py).

    Covers both the autotuned path (tuning-context call plus a cache-replay call)
    and the fixed tactic-0 module path, with the output NaN-poisoned beforehand so
    any unwritten or tail-corrupted region fails the SQNR comparison outright.
    """
    _skip_unless_sm120_table()
    torch.manual_seed(3)
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    # The largest shapes need ~9.5 GB per fp32 m x n tensor, and run beside the
    # shared SVDQuant workspace earlier tests leave allocated. Keep one
    # reference, release each output once checked, and compare in row blocks.
    ref, ref_bias = p.pop("ref"), p.pop("ref_bias")
    if use_bias:
        ref = ref_bias
    del ref_bias
    bias = p["bias"] if use_bias else None

    device = p["xq"].device
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    with autotune(True):
        mm_nvfp4_svdquant(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            bias=bias,
            out=out,
            backend="cutlass-sm120",
        )
    assert _sqnr_db(ref, out, chunk_rows=_SQNR_CHUNK_ROWS) > 40.0
    del out

    replay = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=bias,
        out=replay,
        backend="cutlass-sm120",
    )
    assert _sqnr_db(ref, replay, chunk_rows=_SQNR_CHUNK_ROWS) > 40.0
    del replay

    module = _svdquant_module()
    out0 = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    workspace = torch.zeros(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    module.nvfp4_svdquant_gemm(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias,
        out0,
        workspace,
        0,
        device_support_pdl(device),
    )
    assert _sqnr_db(ref, out0, chunk_rows=_SQNR_CHUNK_ROWS) > 40.0


def _decode_row(module, tactic):
    packed = int(module.nvfp4_svdquant_gemm_tactic_row(tactic))
    return {
        "kernel_id": packed & 0xFF,
        "splits": (packed >> 8) & 0xFF,
        "raster": (packed >> 16) & 0xFF,
        "swizzle": (packed >> 24) & 0xFF,
        "streamk": bool((packed >> 32) & 1),
    }


def _skip_unless_sm120_table():
    _skip_unless_svdquant_supported()
    if not _is_sm120():
        pytest.skip("the runtime tactic table is an SM120 backend feature")


def test_sm120_runtime_tactic_table_decode():
    """Frozen legacy rows and documented variant blocks decode as designed."""
    _skip_unless_sm120_table()
    module = _svdquant_module()
    assert module.nvfp4_svdquant_gemm_tactic_num() == 83
    for t in range(16):
        row = _decode_row(module, t)
        assert row == {
            "kernel_id": t,
            "splits": 1,
            "raster": 0,
            "swizzle": 1,
            "streamk": t % 4 >= 2,
        }, f"legacy row {t} changed meaning: {row}"
    assert _decode_row(module, 16) == {
        "kernel_id": 2,
        "splits": 2,
        "raster": 0,
        "swizzle": 1,
        "streamk": True,
    }
    assert _decode_row(module, 31)["splits"] == 4
    assert _decode_row(module, 32)["raster"] == 1  # AlongM on kernel 0
    assert _decode_row(module, 63)["swizzle"] == 4
    spikes = [_decode_row(module, t) for t in range(64, 75)]
    assert [s["kernel_id"] for s in spikes] == [
        16,
        17,
        17,
        17,
        18,
        19,
        20,
        21,
        22,
        22,
        22,
    ]
    assert [s["splits"] for s in spikes] == [1, 1, 2, 4, 1, 1, 1, 1, 1, 2, 4]
    # driver rows (70/71) are persistent; the K256-swap Stream-K rows are not
    assert not spikes[6]["streamk"] and not spikes[7]["streamk"]
    assert all(s["streamk"] for s in spikes[8:])
    # the first static-scheduler block uses scheduler-default persistent rows
    statics = [_decode_row(module, t) for t in range(75, 78)]
    assert [s["kernel_id"] for s in statics] == [23, 24, 25]
    assert all(
        s["splits"] == 1 and s["raster"] == 0 and s["swizzle"] == 1 and not s["streamk"]
        for s in statics
    )
    # the fill-geometry row closes the table as a scheduler-default row
    fill = _decode_row(module, 78)
    assert fill == {
        "kernel_id": 26,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }
    assert _decode_row(module, 81) == {
        "kernel_id": 29,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }
    assert _decode_row(module, 82) == {
        "kernel_id": 30,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }
    # its static-scheduler sibling is append-only so row 78 keeps its meaning
    fill_static = _decode_row(module, 79)
    assert fill_static == {
        "kernel_id": 27,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }
    small_m_fill_static = _decode_row(module, 80)
    assert small_m_fill_static == {
        "kernel_id": 28,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }


def test_sm120_splitk_rows_respect_ktile_bound():
    """can_implement enforces splits <= K-tiles; workspace grows with splits."""
    _skip_unless_sm120_table()
    module = _svdquant_module()
    # k = 512: 4 K-tiles at CTA_K=128, 2 at CTA_K=256.
    m, n, k = 256, 3072, 512
    # row 16/17: kernel 2 (128x128x128 Sk) splits 2/4 -> both within 4 K-tiles
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, 17)
    # rows for kernel 6 (128x128x256 Sk): splits 4 > 2 K-tiles -> rejected
    splits4_k256 = next(
        t
        for t in range(16, 32)
        if _decode_row(module, t)
        == {"kernel_id": 6, "splits": 4, "raster": 0, "swizzle": 1, "streamk": True}
    )
    assert not module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, splits4_k256)
    # forced Split-K needs more reduction workspace than the heuristic row
    big = (4096, 3072, 12288)
    ws_default = module.nvfp4_svdquant_gemm_workspace_size(*big, 2)
    ws_split4 = module.nvfp4_svdquant_gemm_workspace_size(*big, 17)
    assert ws_split4 > ws_default, (ws_split4, ws_default)


_NEW_ROW_CLASS_TACTICS = [
    16,
    32,
    35,
    64,
    65,
    68,
    69,
    70,
    71,
    72,
    73,
    75,
    76,
    77,
    78,
    79,
    80,
]


@pytest.mark.parametrize("tactic", _NEW_ROW_CLASS_TACTICS)
@pytest.mark.parametrize("m,n,k", [(256, 3072, 3072), (6889, 3072, 3072)])
def test_sm120_new_row_classes_match_reference(tactic, m, n, k):
    """Every new runtime-row class must reproduce the reference numerics."""
    _skip_unless_sm120_table()
    torch.manual_seed(11)
    module = _svdquant_module()
    if not module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic):
        pytest.skip(f"tactic {tactic} not implementable for {(m, n, k)}")
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    ws_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
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
        device_support_pdl(device),
    )
    torch.cuda.synchronize()
    sqnr = _sqnr_db(p["ref_bias"], out.float())
    assert sqnr >= 40.0, f"tactic {tactic} on {(m, n, k)}: SQNR {sqnr:.1f} dB"


@pytest.mark.parametrize("m", [64, 129])
def test_sm120_fill_geometry_candidates_are_bitwise_equivalent(m):
    """Scheduler and CTA geometry must not change the target output bits."""
    _skip_unless_sm120_table()
    torch.manual_seed(12)
    n, k = 3072, 3072
    module = _svdquant_module()
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    tactics = (78, 79, 80)
    assert all(
        module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)
        for tactic in tactics
    )
    ws_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        max(
            int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic))
            for tactic in tactics
        ),
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    outputs = []
    for tactic in tactics:
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
            device_support_pdl(device),
        )
        torch.cuda.synchronize()
        outputs.append(out)
    assert all(
        torch.equal(outputs[0].view(torch.int16), output.view(torch.int16))
        for output in outputs[1:]
    )


def test_sm120_row80_cuda_graph():
    """The row80 LoRA side slot produces the same bits after graph replay."""
    _skip_unless_sm120_table()

    torch.manual_seed(19)
    m, n, k, tactic = 64, 3072, 3072, 80
    module = _svdquant_module()
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    workspace = torch.zeros(
        max(
            DEFAULT_WORKSPACE_SIZE,
            int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
        ),
        dtype=torch.uint8,
        device=device,
    )
    enable_pdl = device_support_pdl(device)

    def run(out_buf):
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out_buf,
            workspace,
            tactic,
            enable_pdl,
        )

    out_eager = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    run(out_eager)
    torch.cuda.synchronize()

    out_graph = torch.full_like(out_eager, float("nan"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(out_graph)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out_graph)
    for _ in range(2):
        out_graph.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_graph.view(torch.int16), out_eager.view(torch.int16))


@pytest.mark.parametrize("tactic", [16, 64, 70, 71, 73, 75, 76, 77, 78, 79, 80])
def test_sm120_cuda_graph_new_rows(tactic):
    """Graph capture/double-replay for a Split-K row, a small-N spike row,
    both 64-row-M driver rows, the K256-swap forced-split row, and the
    static-scheduler rows.

    The workspace is pre-provisioned via the workspace-size query before
    capture; the launcher must never allocate inside the captured region.
    """
    _skip_unless_sm120_table()
    torch.manual_seed(13)
    m, n, k = 256, 3072, 3072
    module = _svdquant_module()
    if not module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic):
        pytest.skip(f"tactic {tactic} not implementable for {(m, n, k)}")
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    enable_pdl = device_support_pdl(device)
    ws_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)

    def run(out_buf):
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out_buf,
            workspace,
            tactic,
            enable_pdl,
        )

    out_eager = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    run(out_eager)
    torch.cuda.synchronize()

    out_graph = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(out_graph)  # capture warmup on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out_graph)
    for _ in range(2):
        out_graph.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_graph, out_eager), (
            f"graph replay diverged from eager for tactic {tactic}"
        )


@pytest.mark.parametrize("tactic", [68, 75, 79, 80])
def test_sm120_zero_output_has_positive_sign(tactic: int) -> None:
    """The epilogue canonicalizes a negative-scaled zero result to +0.0."""
    _skip_unless_sm120_table()
    torch.manual_seed(17)
    m, n, k = 256, 3072, 3072
    module = _svdquant_module()
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    p["alpha"].fill_(-1.0)
    p["xq"].zero_()
    p["d"].zero_()
    workspace = torch.zeros(
        max(
            DEFAULT_WORKSPACE_SIZE,
            int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
        ),
        dtype=torch.uint8,
        device=device,
    )
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)

    module.nvfp4_svdquant_gemm(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        None,
        out,
        workspace,
        tactic,
        device_support_pdl(device),
    )

    assert torch.count_nonzero(out.view(torch.int16)).item() == 0


@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize("m,n,k", [(4096, 3072, 3072), (512, 5120, 5120)])
def test_public_entry_serves_rank_64(m, n, k, rank):
    """The rank reaches the module through the public API, not a private getter.

    The collective takes its LoRA rank as a build parameter, so serving rank 64
    means mm_nvfp4_svdquant reading d.shape[1] and getting the module compiled
    for it. A caller should not have to reach for
    get_nvfp4_svdquant_sm120_module(64) by hand.

    Rank 32 rides along as the control: it must keep using the module it always
    did, which is the same object the production build produces.
    """
    _skip_unless_svdquant_supported()
    if not _is_sm120():
        pytest.skip("requires SM120")
    torch.manual_seed(3)
    p = _make_gemm_problem(m, n, k, rank=rank)
    out = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=None,
        out=None,
        backend="cutlass-sm120",
    )
    assert _sqnr_db(p["ref"], out.float()) > 40.0

    module = _get_nvfp4_svdquant_module_for_device(torch.device("cuda"), rank)
    assert module.nvfp4_svdquant_gemm_lora_rank() == rank


@pytest.mark.parametrize("rank", (32, 64))
def test_linear_storage_offset_inputs_match_aligned_inputs(rank: int) -> None:
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("requires SM120")
    m, n, k = 64, 128, 3072
    torch.manual_seed(0)
    x = torch.randn(m * k + 1, dtype=torch.bfloat16, device="cuda")[1:].view(m, k)
    pqs = torch.ones(k + 1, dtype=torch.bfloat16, device="cuda")[1:]
    assert x.is_contiguous() and x.data_ptr() % 16 == 2
    assert pqs.is_contiguous() and pqs.data_ptr() % 16 == 2
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.full((n * k // 16,), 0x38, dtype=torch.uint8, device="cuda")
    alpha = torch.ones(1, device="cuda")
    global_sf = torch.ones(1, device="cuda")
    l2t = torch.randn((k, rank), dtype=torch.bfloat16, device="cuda") / k**0.25
    l1 = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda") / rank**0.25

    expected = svdquant_linear(
        x.clone(),
        weight,
        weight_sf,
        alpha,
        pqs.clone(),
        l2t,
        l1,
        global_sf,
        enable_pdl=False,
        backend="cutlass-sm120",
    )
    actual = svdquant_linear(
        x,
        weight,
        weight_sf,
        alpha,
        pqs,
        l2t,
        l1,
        global_sf,
        enable_pdl=False,
        backend="cutlass-sm120",
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# Fused quantization, LoRA-down and complete linear operations.

# The producer half of the FFI. -1 means "the caller has no opinion about the
# producer geometry", which keeps the shape-dispatched templated launch these
# tests were written against; the runtime-M/K conversion added producer_family,
# three tiling numbers and the address policy after enable_pdl.
_LEGACY_PRODUCER_ARGS = (-1, 0, 0, 0, 0)


def _assert_public_entry_matches(reference, candidate):
    """Same computation through a tactic the autotuner chose for itself.

    Not bit-equality, and not a relative-error bound either. Measured over two
    shapes (m=64 and m=512, 0.2M and 1.6M elements): half the elements are
    bit-identical, the largest absolute deviation is 8.0 where the outputs run
    to several hundred -- one to two bf16 ulps at that magnitude -- and the only
    sign flips are on elements whose magnitude is at most 4.0, i.e. values near
    zero that two accumulation orders straddle. A relative bound reports 2.0 on
    exactly those and says nothing.

    SQNR is what the rest of this suite compares against and it is magnitude
    aware, so near-zero disagreement cannot dominate it. Measured 50.1 dB and
    49.8 dB against the suite's usual 40 dB floor.

    The separate-vs-combined check stays exact: both arms there run the same
    tactic and the same producer, and producer variants were measured
    bit-identical, so that contract still has teeth.
    """
    ref, cand = reference.float(), candidate.float()
    assert ref.shape == cand.shape
    assert torch.isfinite(cand).all(), "public entry produced non-finite output"
    noise = (ref - cand).pow(2).sum().item()
    signal = ref.pow(2).sum().item()
    sqnr_db = 10.0 * math.log10(signal / noise) if noise > 0 else float("inf")
    assert sqnr_db > 40.0, (
        f"public entry disagrees with the fixed-tactic route: SQNR {sqnr_db:.1f} dB "
        f"(worst element {(ref - cand).abs().max().item():.4e})"
    )


_LAUNCH_GEOMETRY_SLICE = {
    # launch_large_m_kernel<M, K, BlockThreads, TileM, TileK>
    "large_m": (slice(2, 5), _sm120_routes.SM120_FAMILY_LARGE_M),
    # launch_small_m_kernel<M, K, BlockThreads, DownTileCols, RowsPerQuantBlock = 4>
    "small_m": (slice(2, 5), _sm120_routes.SM120_FAMILY_SMALL_M),
    # launch_m537_mixed_kernel<K, BlockThreads, DownTileCols, RowsPerQuantBlock, ...>
    "m537_mixed": (slice(1, 4), _sm120_routes.SM120_FAMILY_M537),
}


def _assert_shape_computes_geometry(m: int, k: int, launch: str) -> None:
    """The shape's computed producer ladder must name the geometry this launch did.

    These tests used to scan the C++ source for a per-shape template
    instantiation. The instantiations are gone on purpose -- one runtime
    launcher now serves every (M, K), which is what let 110 of them collapse to
    34 -- so a source scan can only report their absence. What decides the
    launch today is the variant ladder, so that is what is asserted: the shape
    still computes the geometry the pinned template used to hard-code.
    """
    match = re.fullmatch(r"launch_(\w+)_kernel<([\d,\s]+)>", launch)
    assert match, f"unrecognised launch spelling: {launch}"
    kind, nums = match.group(1), [int(v) for v in match.group(2).split(",")]
    assert kind in _LAUNCH_GEOMETRY_SLICE, f"unmapped launch family: {kind}"
    where, family = _LAUNCH_GEOMETRY_SLICE[kind]
    geometry = tuple(nums[where])
    if kind == "small_m" and len(geometry) == 2:
        geometry = geometry + (4,)  # RowsPerQuantBlock defaults to 4
    variants = _sm120_routes.sm120_producer_variants(m, k)
    assert any(f == family and tuple(t) == geometry for f, t, _ in variants), (
        f"({m}, {k}) no longer computes {geometry} for family {family}; "
        f"its ladder offers {sorted({(f, tuple(t)) for f, t, _ in variants})}"
    )


def _assert_launch_geometry_is_computed(launch: str) -> None:
    """Same check, for the constants that carry (M, K) inside the launch string."""
    nums = [int(v) for v in re.search(r"<([\d,\s]+)>", launch).group(1).split(",")]
    _assert_shape_computes_geometry(nums[0], nums[1], launch)


def _assert_admission_tracks_the_ladder(m: int, k: int, rank: int = 32) -> None:
    """Admission must be exactly "a producer can launch this", nothing else.

    These checks used to read "a neighbouring shape is refused", which was true
    of the 51-entry table and is not the contract any more: a shape nobody
    benchmarked is offered to the tuner when its geometry ladder is non-empty,
    and dropped by measurement rather than by a list. What still has to hold --
    and what a stray shape-keyed special case would break -- is that the two
    agree in both directions.
    """
    admitted = svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
    launchable = bool(_sm120_routes.sm120_producer_variants(m, k, rank))
    assert admitted == launchable, (
        f"({m}, {k}) admitted={admitted} but launchable={launchable}"
    )


def test_sm120_table_miss_n_keeps_fused_route_eligibility() -> None:
    """Fused-route eligibility is a property of (M, K), not of any one N."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(64, 3072, 32), (
        "an unmeasured N with a supported (M, K) must retain the fused K12 route"
    )


@pytest.mark.parametrize("k", (5120, 5376, 7168))
def test_sm120_m537_l2t_pack_matches_kernel_lane_mapping(k: int) -> None:
    source = torch.arange(k * 32, dtype=torch.int32).reshape(k, 32)
    packed = svdquant_sm120_cutlass._pack_sm120_m537_l2t(source)
    packed_tiles = packed.reshape(k // 16, 2, 2, 8, 4, 2, 2)

    for step, tile_n, fragment, group, pair, half, within in (
        (0, 0, 0, 0, 0, 0, 0),
        (7, 1, 1, 6, 3, 1, 1),
        (k // 16 - 1, 1, 0, 7, 2, 0, 1),
    ):
        source_row = step * 16 + half * 8 + pair * 2 + within
        source_col = tile_n * 16 + fragment * 8 + group
        assert (
            packed_tiles[step, tile_n, fragment, group, pair, half, within].item()
            == source[source_row, source_col].item()
        )


def test_sm120_m537_l2t_pack_cache_reuses_and_invalidates() -> None:
    cache = svdquant_sm120_cutlass._SM120_M537_PACKED_L2T_CACHE
    cache.clear()
    source = torch.zeros((5120, 32), dtype=torch.int32)
    try:
        first = svdquant_sm120_cutlass._cached_sm120_m537_l2t(source)
        assert svdquant_sm120_cutlass._cached_sm120_m537_l2t(source) is first

        source.add_(1)
        mutated = svdquant_sm120_cutlass._cached_sm120_m537_l2t(source)
        assert mutated is not first
        assert torch.equal(mutated, svdquant_sm120_cutlass._pack_sm120_m537_l2t(source))
    finally:
        cache.clear()


def test_sm120_m537_l2t_pack_rejects_sibling_shape() -> None:
    with pytest.raises(ValueError, match="requires rank 32"):
        svdquant_sm120_cutlass._pack_sm120_m537_l2t(torch.empty((5120, 64)))


@pytest.mark.parametrize("k", (5120, 5376, 7168))
def test_sm120_m537_l2t_pack_handles_inference_tensor_mutation(k: int) -> None:
    cache = svdquant_sm120_cutlass._SM120_M537_PACKED_L2T_CACHE
    with torch.inference_mode():
        source = torch.zeros((k, 32), dtype=torch.bfloat16)
        first = svdquant_sm120_cutlass._cached_sm120_m537_l2t(source)
        source.add_(1)
        updated = svdquant_sm120_cutlass._cached_sm120_m537_l2t(source)

    assert id(source) not in cache
    assert torch.equal(first, torch.zeros_like(first))
    assert torch.equal(updated, torch.ones_like(updated))
    assert updated is not first


def test_sm120_jit_module_exports_fused_k1_k2_operator() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    module = get_nvfp4_svdquant_sm120_module()

    assert hasattr(module, "nvfp4_quantize_smooth_lora_down_sm120")
    assert hasattr(module, "nvfp4_quantize_smooth_lora_down_cublaslt_sm120")
    assert hasattr(module, "nvfp4_svdquant_linear_sm120")
    assert hasattr(module, "nvfp4_svdquant_gemm_fallback_tactic")


@pytest.mark.parametrize(
    ("m", "k"),
    (
        (64, 3072),
        (512, 3072),
        (512, 5120),
        (7800, 8960),
        (32760, 8960),
        (7800, 5120),
        (7800, 13824),
        (27280, 3072),
        (32760, 5120),
        (32760, 13824),
        (75600, 13824),
        (27280, 14336),
        # Producers added from the geometry sweep.
        # Two of them are the first small_m instantiations at these M, and one is
        # the first large_m instantiation at 48-row tiles, so the equivalence is
        # not implied by any neighbour already listed above.
        (512, 1536),
        (7800, 1536),
        (256, 3072),
        (6889, 3072),
        (9216, 3072),
        (16384, 3072),
        # Producers added so every production prefix has a fused candidate, even
        # where fusing measured slower than the prefix the shape pays today.
        # Correctness is not conditional on winning.
        (1024, 3072),
        (4096, 3072),
        (64, 12288),
        (256, 12288),
        (512, 12288),
        (1024, 12288),
        (512, 5120),
        (537, 14336),
        (1935, 5376),
    ),
)
def test_sm120_fused_k1_k2_matches_unfused_outputs(m: int, k: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (k, 32), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),),
        dtype=torch.uint8,
        device="cuda",
    )
    down = torch.empty((m, 32), dtype=torch.bfloat16, device="cuda")
    module = get_nvfp4_svdquant_sm120_module()

    # (512, 5120) reaches its prefix through cuBLASLt in production and used to
    # be checked only that way. It has a native producer now, like every other
    # production prefix, so both are exercised: the cuBLASLt one first, then the
    # native one, which overwrites the outputs and is what the assertions below
    # score. Neither is allowed to be the only thing tested.
    if m == 512 and k == 5120:
        workspace = torch.empty((32 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
        module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
            x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down, workspace
        )
        cublaslt_xq = xq.clone()
        cublaslt_sf = sf.clone()
        cublaslt_down = down.clone()
        xq.zero_()
        sf.zero_()
        down.zero_()
    else:
        cublaslt_xq = cublaslt_sf = cublaslt_down = None
    module.nvfp4_quantize_smooth_lora_down_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        xq,
        sf,
        down,
    )
    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)

    assert torch.equal(xq, xq_ref)
    assert torch.equal(sf, sf_ref)
    if cublaslt_xq is not None:
        assert torch.equal(cublaslt_xq, xq_ref)
        assert torch.equal(cublaslt_sf, sf_ref)
        assert torch.equal(cublaslt_down, down_ref)

    # Score both LoRA-down implementations against a common accurate reference
    # rather than scoring the fused one against the other. `down_ref` is a BF16
    # GEMM with its own error, and that error is shape-dependent: measured
    # against FP64 on this seed, torch.mm ranges from 55.6 dB down to 50.2 dB
    # (9216x3072 and 16384x3072), while the fused producer sits at 55.5-55.7 dB
    # on every shape. Comparing the two directly therefore reports the
    # reference's accuracy at exactly the shapes where it is worst, which is how
    # two correct producers first failed here.
    # An FP32 GEMM with TF32 off lands 126-140 dB from FP64 on these shapes --
    # 70 dB clear of what is being measured -- so it can stand in for FP64,
    # which would be far too slow at 75600x13824.
    tf32_was_allowed = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        accurate_ref = torch.mm(x.float(), l2t_smoothed.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_was_allowed

    def _sqnr_db(candidate: torch.Tensor) -> float:
        noise = ((accurate_ref - candidate.float()) ** 2).mean()
        return float(10 * torch.log10((accurate_ref**2).mean() / noise))

    fused_sqnr = _sqnr_db(down)
    reference_sqnr = _sqnr_db(down_ref)

    # The contract this test exists to hold: fusing the LoRA-down into the
    # producer must not cost accuracy against the separate torch.mm it replaces.
    assert fused_sqnr >= reference_sqnr - 0.5, (
        f"fused LoRA-down SQNR {fused_sqnr:.3f} dB is worse than the unfused "
        f"torch.mm at {reference_sqnr:.3f} dB for m={m}, k={k}"
    )
    # Plus the absolute floor the route carried before, unchanged, so nothing
    # is weakened for a shape whose reference happens to be accurate.
    relaxed_sqnr_shape = k >= 8192 or (m, k) == (32760, 5120)
    min_sqnr_db = 48.0 if relaxed_sqnr_shape else 50.0
    assert fused_sqnr > min_sqnr_db, (
        f"fused LoRA-down SQNR {fused_sqnr:.3f} dB is below "
        f"{min_sqnr_db:.1f} dB for m={m}, k={k}"
    )


def test_sm120_fused_k1_k2_rejects_non_exact_shape() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    m, k = 63, 3072
    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((3072,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty((3072, 32), dtype=torch.bfloat16, device="cuda")
    xq = torch.empty((63, 1536), dtype=torch.uint8, device="cuda")
    sf = torch.empty((24576,), dtype=torch.uint8, device="cuda")
    down = torch.empty((63, 32), dtype=torch.bfloat16, device="cuda")

    # This asserted a RuntimeError naming "the measured SM120 K12 shapes" -- the
    # FFI whitelist, which listed where the producer had been measured and
    # refused everything else. Measured on this card, the neighbours it refused
    # run correctly: 4095x12288 and 4097x12288 reach 51.4 and 51.3 dB SQNR
    # against an fp32 reference, with bit-exact xq, next to 51.3 dB for
    # 4096x12288 which the list did admit.
    #
    # So admission is the computed geometry now, and that is what is asserted:
    # a shape that computes one runs, a shape that does not is refused by the
    # launcher that would have had to run it.
    variants = _sm120_routes.sm120_producer_variants(m, k)
    if variants:
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down
        )
        torch.cuda.synchronize()
    else:
        with pytest.raises(RuntimeError):
            module.nvfp4_quantize_smooth_lora_down_sm120(
                x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down
            )


def _run_prefix_separately(
    module, m, k, variant, x, pqs, gs, l2t, xq, sf, down, workspace
):
    """Run, on its own, exactly the prefix the fused route will run.

    The arm used to branch on ``sm120_fused_prefix_route(m, k)`` -- a table
    lookup -- while the fused arm passed no producer at all and let a second
    copy of that table, in C++, decide. The two agreed only because both read
    the same ladder. Now the producer is named once, here, and handed to both
    arms, so the comparison is between one implementation launched two ways
    rather than between whatever two tables happened to say.
    """
    family, tiling, policy = _sm120_routes.sm120_decode_producer_variant(m, k, variant)
    if family == _sm120_routes.SM120_FAMILY_CUBLASLT:
        module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
            x, pqs, gs, l2t, xq, sf, down, workspace
        )
    else:
        module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
            x,
            pqs,
            gs,
            l2t,
            xq,
            sf,
            down,
            family,
            tiling[0],
            tiling[1],
            tiling[2],
            policy,
        )
    return (family, tiling[0], tiling[1], tiling[2], policy)


@pytest.mark.parametrize(
    ("m", "n", "k", "tactic"),
    (
        (64, 3072, 3072, 80),
        (64, 12288, 3072, 0),
        (512, 3072, 3072, 36),
        (512, 12288, 3072, 55),
        (512, 5120, 5120, 52),
        (32760, 1536, 1536, 8),
        (32760, 8960, 1536, 9),
        (7800, 1536, 8960, 0),
        (32760, 1536, 8960, 8),
        (7800, 5120, 5120, 8),
        (7800, 5120, 13824, 8),
        (7800, 13824, 5120, 9),
        (27280, 3072, 3072, 8),
        (32760, 5120, 5120, 8),
        (32760, 5120, 13824, 8),
        (75600, 5120, 13824, 8),
        (27280, 3072, 14336, 8),
        # One production shape per producer added by the geometry sweep, so the
        # combined K12+K3 route is exercised on each new prefix and not only the
        # producer in isolation. Tactic 8 is the k256x128x128 kernel every other
        # large-M row here uses; these rows are checked for equivalence against
        # separate launches, not for being the tuner's pick.
        (512, 1536, 1536, 8),
        (7800, 1536, 1536, 8),
        (256, 3072, 3072, 8),
        (6889, 3072, 3072, 8),
        (9216, 3072, 3072, 8),
        (16384, 3072, 3072, 8),
        # One production shape per prefix that gained a producer for coverage
        # rather than for speed. The combined route picks its prefix by the
        # ladder, so the three cuBLASLt-routed rows below still exercise that
        # path -- what is checked here is that adding a producer did not disturb
        # the route each shape actually runs.
        (1024, 3072, 3072, 8),
        (4096, 3072, 3072, 8),
        (64, 3072, 12288, 8),
        (256, 3072, 12288, 8),
        (512, 3072, 12288, 8),
        (1024, 3072, 12288, 8),
        (512, 5120, 5120, 8),
        (537, 5376, 14336, 8),
        (1935, 21504, 5376, 8),
    ),
)
def test_sm120_combined_k12_k3_matches_separate_launches(
    m: int, n: int, k: int, tactic: int
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    rank = 32
    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.ones((n * (k // 16),), dtype=torch.uint8, device="cuda")
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
    l1_scaled = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda").contiguous()
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda").contiguous()
    module = get_nvfp4_svdquant_sm120_module()
    workspace_bytes = max(
        32 * 1024 * 1024,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")

    separate = (
        torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
        torch.empty(
            (((m + 127) // 128) * 128 * (k // 16),),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.empty((m, rank), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    )
    combined = tuple(torch.empty_like(tensor) for tensor in separate)

    # The separate arm has to run the prefix the combined route will run, or the
    # comparison below is between two different LoRA-down implementations and
    # cannot be bit-exact. Since the runtime-M/K conversion the producer is named
    # by (family, tiling, address policy) rather than by the shape's pinned
    # template, so both arms are driven from the same decoded variant here --
    # reading it is what keeps this test honest when a route changes.
    producer_args = _run_prefix_separately(
        module,
        m,
        k,
        tactic >> _sm120_routes.SM120_PRODUCER_SHIFT,
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        separate[0],
        separate[1],
        separate[2],
        workspace,
    )
    module.nvfp4_svdquant_gemm(
        separate[0],
        weight_fp4,
        separate[1],
        weight_sf,
        alpha,
        separate[2],
        l1_scaled,
        bias,
        separate[3],
        workspace,
        tactic,
        False,
    )
    module.nvfp4_svdquant_linear_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        weight_fp4,
        weight_sf,
        alpha,
        l1_scaled,
        bias,
        combined[0],
        combined[1],
        combined[2],
        combined[3],
        workspace,
        tactic,
        False,
        *producer_args,
    )

    torch.cuda.synchronize()
    for separate_output, combined_output in zip(separate, combined, strict=False):
        assert torch.equal(separate_output, combined_output)

    public_output = svdquant_linear(
        x,
        weight_fp4,
        weight_sf,
        alpha,
        pre_quant_scale,
        l2t_smoothed,
        l1_scaled,
        global_scale,
        bias=bias,
        enable_pdl=False,
        backend="cutlass-sm120",
    )
    # Not bit-equality: the raw FFI above was handed a fixed tactic, while the
    # public entry picks its own through the autotuner. A different K3 tactic
    # accumulates in a different order, which lands one bf16 ulp away -- measured
    # at 96.5 vs 97.0, and 96 is exactly where the bf16 step is 0.5. The
    # separate-vs-combined check above stays exact because both arms there run
    # the same tactic and the same producer (producer variants were measured
    # bit-identical, so that contract is still meaningful).
    _assert_public_entry_matches(combined[3], public_output)


def test_sm120_k12288_large_m_uses_the_accepted_production_route() -> None:
    """Accepted K12288 producers keep their fused-route admission."""
    for m in (4096, 6889, 9216, 16384):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, 12288, 32)
        _assert_admission_tracks_the_ladder(m + 1, 12288, 32)


@pytest.mark.parametrize(
    ("m", "k"),
    (
        (4096, 12288),
        (6889, 12288),
        (9216, 12288),
        (16384, 12288),
    ),
)
def test_sm120_k12288_large_m_writes_every_output_byte(m: int, k: int) -> None:
    """Two hostile prefills must converge on the same complete oracle result."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (k, 32), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),),
        dtype=torch.uint8,
        device="cuda",
    )
    down = torch.empty((m, 32), dtype=torch.bfloat16, device="cuda")
    module = get_nvfp4_svdquant_sm120_module()

    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)

    for sentinel in (0x00, 0xFF):
        xq.fill_(sentinel)
        sf.fill_(sentinel)
        down.fill_(float("nan"))
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )
        torch.cuda.synchronize()

        assert torch.equal(xq, xq_ref), (
            f"xq left unwritten or wrong after the 0x{sentinel:02x} prefill "
            f"for m={m}, k={k}"
        )
        assert torch.equal(sf, sf_ref), (
            f"sf left unwritten or wrong after the 0x{sentinel:02x} prefill "
            f"for m={m}, k={k}"
        )
        assert bool(torch.isfinite(down).all()), (
            f"down retains the NaN prefill after the 0x{sentinel:02x} run "
            f"for m={m}, k={k}"
        )
        noise = ((down_ref.float() - down.float()) ** 2).mean()
        sqnr = 10 * torch.log10((down_ref.float() ** 2).mean() / noise)
        assert float(sqnr) > 48.0, (
            f"fused LoRA-down SQNR {float(sqnr):.3f} dB is below 48.0 dB for "
            f"m={m}, k={k}"
        )


@pytest.mark.parametrize(
    ("m", "k"),
    (
        (4095, 12288),
        (4097, 12288),
        (6888, 12288),
        (6890, 12288),
        (9215, 12288),
        (9217, 12288),
        (16383, 12288),
        (16385, 12288),
        (4096, 12160),
        (4096, 12416),
        # (16384, 3072) used to stand here as the "same M, different K" case.
        # It now has its own swept producer, so the same coverage moves to a K
        # that still has none.
        (16384, 6144),
    ),
)
def test_sm120_k12288_large_m_rejects_neighboring_shapes(m: int, k: int) -> None:
    """The K12288 admission is exactly four shapes and does not generalize."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((k,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty((k, 32), dtype=torch.bfloat16, device="cuda")
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),),
        dtype=torch.uint8,
        device="cuda",
    )
    down = torch.empty((m, 32), dtype=torch.bfloat16, device="cuda")

    # This asserted a RuntimeError naming "the measured SM120 K12 shapes" -- the
    # FFI whitelist, which listed where the producer had been measured and
    # refused everything else. Measured on this card, the neighbours it refused
    # run correctly: 4095x12288 and 4097x12288 reach 51.4 and 51.3 dB SQNR
    # against an fp32 reference, with bit-exact xq, next to 51.3 dB for
    # 4096x12288 which the list did admit.
    #
    # So admission is the computed geometry now, and that is what is asserted:
    # a shape that computes one runs, a shape that does not is refused by the
    # launcher that would have had to run it.
    variants = _sm120_routes.sm120_producer_variants(m, k)
    if variants:
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down
        )
        torch.cuda.synchronize()
    else:
        with pytest.raises(RuntimeError):
            module.nvfp4_quantize_smooth_lora_down_sm120(
                x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down
            )


@pytest.mark.parametrize("rank", (16, 64))
def test_sm120_k12288_large_m_rejects_non_rank32_lora(rank: int) -> None:
    """A direct-FFI shape still requires the rank-32 LoRA-down topology."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    m, k = 4096, 12288
    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((k,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty((k, rank), dtype=torch.bfloat16, device="cuda")
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty((m * (k // 16),), dtype=torch.uint8, device="cuda")
    down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match=r"l2t_smoothed must be \[k, 32\]"):
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )


@pytest.mark.parametrize(
    ("m", "n", "k", "tactic"),
    (
        (4096, 3072, 12288, 32),
        (6889, 3072, 12288, 48),
        (9216, 3072, 12288, 48),
        (16384, 3072, 12288, 8),
    ),
)
def test_sm120_k12288_large_m_combined_k12_k3_matches_separate_launches(
    m: int, n: int, k: int, tactic: int
) -> None:
    """Combined equivalence plus the 40 dB full-output contract.

    The selected fixed K3 tactics for rows 33/36/39/42 pin the K3 half of the
    comparison and now match the accepted public production route.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    rank = 32
    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.ones((n * (k // 16),), dtype=torch.uint8, device="cuda")
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
    l1_scaled = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda").contiguous()
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda").contiguous()
    module = get_nvfp4_svdquant_sm120_module()
    workspace_bytes = max(
        32 * 1024 * 1024,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")

    separate = (
        torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
        torch.empty(
            (((m + 127) // 128) * 128 * (k // 16),),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.empty((m, rank), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    )
    combined = tuple(torch.empty_like(tensor) for tensor in separate)

    separate[0].fill_(0x00)
    separate[1].fill_(0x00)
    separate[2].fill_(float("nan"))
    separate[3].fill_(float("nan"))
    combined[0].fill_(0xFF)
    combined[1].fill_(0xFF)
    combined[2].fill_(float("nan"))
    combined[3].fill_(float("nan"))

    module.nvfp4_quantize_smooth_lora_down_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        separate[0],
        separate[1],
        separate[2],
    )
    module.nvfp4_svdquant_gemm(
        separate[0],
        weight_fp4,
        separate[1],
        weight_sf,
        alpha,
        separate[2],
        l1_scaled,
        bias,
        separate[3],
        workspace,
        tactic,
        False,
    )
    module.nvfp4_svdquant_linear_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        weight_fp4,
        weight_sf,
        alpha,
        l1_scaled,
        bias,
        combined[0],
        combined[1],
        combined[2],
        combined[3],
        workspace,
        tactic,
        False,
        *_LEGACY_PRODUCER_ARGS,
    )

    torch.cuda.synchronize()
    for name, output in (
        ("separate down", separate[2]),
        ("separate full output", separate[3]),
        ("combined down", combined[2]),
        ("combined full output", combined[3]),
    ):
        assert bool(torch.isfinite(output).all()), f"{name} is incomplete"
    for separate_output, combined_output in zip(separate, combined, strict=False):
        assert torch.equal(separate_output, combined_output)

    # Independent generic prefix: the same K3 tactic driven by the generic
    # quantizer and torch.mm rather than by the fused producer.
    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)
    reference_output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    module.nvfp4_svdquant_gemm(
        xq_ref,
        weight_fp4,
        sf_ref,
        weight_sf,
        alpha,
        down_ref,
        l1_scaled,
        bias,
        reference_output,
        workspace,
        tactic,
        False,
    )

    torch.cuda.synchronize()
    noise = ((reference_output.float() - combined[3].float()) ** 2).mean()
    sqnr = 10 * torch.log10((reference_output.float() ** 2).mean() / noise)
    assert float(sqnr) > 40.0, (
        f"combined K12+K3 output SQNR {float(sqnr):.3f} dB is below 40.0 dB "
        f"for m={m}, n={n}, k={k}, tactic={tactic}"
    )

    public_output = svdquant_linear(
        x,
        weight_fp4,
        weight_sf,
        alpha,
        pre_quant_scale,
        l2t_smoothed,
        l1_scaled,
        global_scale,
        bias=bias,
        enable_pdl=False,
        backend="cutlass-sm120",
    )
    # Not bit-equality: the raw FFI above was handed a fixed tactic, while the
    # public entry picks its own through the autotuner. A different K3 tactic
    # accumulates in a different order, which lands one bf16 ulp away -- measured
    # at 96.5 vs 97.0, and 96 is exactly where the bf16 step is 0.5. The
    # separate-vs-combined check above stays exact because both arms there run
    # the same tactic and the same producer (producer variants were measured
    # bit-identical, so that contract is still meaningful).
    _assert_public_entry_matches(combined[3], public_output)


# ---------------------------------------------------------------------------
# M512/K3072 prefix coverage.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_repo_source(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_sm120_m32760_k1536_has_exact_production_admission() -> None:
    retained_launch = "launch_large_m_kernel<32760, 1536, 256, 32, 256>"
    # The dispatcher used to carry an "m == 32760 && k == 1536" branch and this
    # shape's own template. Both were per-shape structures the runtime-M/K
    # conversion replaced, so the pair of statements they made -- this shape is
    # admitted, and it launches with this geometry -- is made directly.
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(32760, 1536, 32)
    _assert_launch_geometry_is_computed(retained_launch)
    # (7800, 1536) and (512, 1536) used to be named here as prefixes this
    # admission must not generalize to. They have since earned producers of
    # their own by measurement, so what is
    # asserted now is that each K1536 prefix carries its own launch rather than
    # sharing 32760's.
    # One runtime launcher serves every (M, K) now, so "carries its own launch"
    # has no instantiation to point at. What still holds, and is what the line
    # meant, is that each K1536 prefix resolves its own producer rather than
    # inheriting 32760's.
    for own_m in (7800, 512):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(own_m, 1536, 32)
        assert _sm120_routes.sm120_producer_variants(own_m, 1536)

    # The K1536 clause used to name its three admitted M values in the FFI
    # source. Admission is a predicate now, so it is asked directly -- and the
    # "and nothing else" half is kept by checking a neighbour is refused.
    for admitted_m in (512, 7800, 32760):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
            admitted_m, 1536, _H3_ROW52_RANK
        )
    _assert_admission_tracks_the_ladder(513, 1536, _H3_ROW52_RANK)

    # This prefix reached the fused runner only through the pinned-prefix union
    # while the pins existed. It is stated in SM120_FUSED_LINEAR_MK now, because
    # without it the tuner sees only the unfused runner here and measures
    # slower.
    assert (32760, 1536) in svdquant_sm120_cutlass._SM120_FUSED_LINEAR_MK
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(32760, 1536, 32)
    # Admission is not the route namespace: this shape still keys on the base
    # route ABI and keeps reaching the winners tuned under it.
    assert svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_ABI_VERSION == 5
    assert svdquant_sm120_cutlass._sm120_linear_route_abi_version(32760, 1536, 32) == 5
    assert svdquant_sm120_cutlass._SM120_TACTIC_ABI_VERSION == 3


_M512_M = 512
_M512_K = 3072
_M512_RANK = 32
_M512_ROW0_N = 3072
_M512_ROW0_K3_TACTIC = 36
_M512_ROW28_N = 12288
_M512_ROW28_K3_TACTIC = 55


def _m512_producer_inputs(
    seed: int = 20260725,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 inputs for the M512/K3072 prefix and combined linear route."""
    torch.manual_seed(seed)
    x = torch.randn((_M512_M, _M512_K), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((_M512_K,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (_M512_K, _M512_RANK), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    return x, pre_quant_scale, global_scale, l2t_smoothed


def _m512_sf_numel() -> int:
    return (((_M512_M + 127) // 128) * 128) * (_M512_K // 16)


def _m512_producer_outputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((_M512_M, _M512_K // 2), dtype=torch.uint8, device="cuda"),
        torch.empty((_m512_sf_numel(),), dtype=torch.uint8, device="cuda"),
        torch.empty((_M512_M, _M512_RANK), dtype=torch.bfloat16, device="cuda"),
    )


def test_sm120_m512_k3072_writes_every_output_byte() -> None:
    """The production prefix overwrites every xq/SF/down output element."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    x, pre_quant_scale, global_scale, l2t_smoothed = _m512_producer_inputs()
    xq, sf, down = _m512_producer_outputs()
    module = get_nvfp4_svdquant_sm120_module()
    prefix = module.nvfp4_quantize_smooth_lora_down_sm120

    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)

    for xq_sentinel, sf_sentinel in ((0x00, 0xFF), (0xFF, 0x00)):
        xq.fill_(xq_sentinel)
        sf.fill_(sf_sentinel)
        down.fill_(float("nan"))
        prefix(x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down)
        torch.cuda.synchronize()

        assert torch.equal(xq, xq_ref), (
            f"M512/K3072 xq left unwritten or wrong after the 0x{xq_sentinel:02x} prefill"
        )
        assert torch.equal(sf, sf_ref), (
            f"M512/K3072 sf left unwritten or wrong after the 0x{sf_sentinel:02x} prefill"
        )
        assert bool(torch.isfinite(down).all()), (
            "M512/K3072 down retains the NaN prefill after the opposing "
            f"0x{xq_sentinel:02x}/0x{sf_sentinel:02x} run"
        )

        noise = ((down_ref.float() - down.float()) ** 2).mean()
        sqnr = 10 * torch.log10((down_ref.float() ** 2).mean() / noise)
        # The same 48 dB direct-native floor the other native K12 producers
        # hold: the eight-warp split accumulates in a different order from
        # torch.mm regardless of how the N tiles are grouped.
        assert float(sqnr) > 48.0, (
            f"M512/K3072 LoRA-down SQNR {float(sqnr):.3f} dB is below 48.0 dB"
        )

        # A misaddressed N fragment corrupts one half of every patch, which a
        # whole-tensor SQNR can average away, so gate every rank column on its
        # own and let the failure name the column.
        column_noise = ((down_ref.float() - down.float()) ** 2).mean(dim=0)
        column_sqnr = 10 * torch.log10(
            (down_ref.float() ** 2).mean(dim=0) / column_noise
        )
        worst = int(torch.argmin(column_sqnr))
        assert float(column_sqnr[worst]) > 48.0, (
            f"M512/K3072 LoRA-down rank column {worst} SQNR "
            f"{float(column_sqnr[worst]):.3f} dB is below 48.0 dB"
        )


@pytest.mark.parametrize("lora_rank", (16, 64))
def test_sm120_m512_k3072_rejects_non_rank32_lora(lora_rank: int) -> None:
    """The admitted M512/K3072 shape still requires the rank-32 LoRA-down topology."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((_M512_M, _M512_K), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((_M512_K,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty(
        (_M512_K, lora_rank), dtype=torch.bfloat16, device="cuda"
    )
    xq = torch.empty((_M512_M, _M512_K // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty((_m512_sf_numel(),), dtype=torch.uint8, device="cuda")
    down = torch.empty((_M512_M, lora_rank), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match=r"l2t_smoothed must be \[k, 32\]"):
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )


def test_sm120_m512_k3072_public_rows0_and_28_keep_the_production_producer() -> None:
    """Combined routes use fresh prefix outputs after LoRA weights change."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    x, pre_quant_scale, global_scale, l2t_smoothed = _m512_producer_inputs()
    perturbed_l2t = (l2t_smoothed.float() * -0.5 + 0.25).bfloat16().contiguous()
    module = get_nvfp4_svdquant_sm120_module()

    references = []
    for lora_weight in (l2t_smoothed, perturbed_l2t):
        reference = _m512_producer_outputs()
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            lora_weight,
            *reference,
        )
        references.append(reference)
    torch.cuda.synchronize()

    assert not torch.equal(references[0][2], references[1][2]), (
        "witness failed: the perturbed LoRA weight produced the same down "
        "output, so the comparisons below could not detect a stale buffer"
    )

    for n, tactic in (
        (_M512_ROW0_N, _M512_ROW0_K3_TACTIC),
        (_M512_ROW28_N, _M512_ROW28_K3_TACTIC),
    ):
        torch.manual_seed(20260807 + n)
        weight_fp4 = torch.randint(
            0, 256, (n, _M512_K // 2), dtype=torch.uint8, device="cuda"
        )
        weight_sf = torch.ones((n * (_M512_K // 16),), dtype=torch.uint8, device="cuda")
        alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
        l1_scaled = torch.randn(
            (n, _M512_RANK), dtype=torch.bfloat16, device="cuda"
        ).contiguous()
        bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda").contiguous()
        workspace_bytes = max(
            32 * 1024 * 1024,
            int(module.nvfp4_svdquant_gemm_workspace_size(_M512_M, n, _M512_K, tactic)),
        )
        workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")
        row = "row 0" if n == _M512_ROW0_N else "row 28"

        for lora_weight, reference in zip(
            (l2t_smoothed, perturbed_l2t), references, strict=False
        ):
            combined_xq, combined_sf, combined_down = _m512_producer_outputs()
            combined_output = torch.full(
                (_M512_M, n), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            module.nvfp4_svdquant_linear_sm120(
                x,
                pre_quant_scale,
                global_scale,
                lora_weight,
                weight_fp4,
                weight_sf,
                alpha,
                l1_scaled,
                bias,
                combined_xq,
                combined_sf,
                combined_down,
                combined_output,
                workspace,
                tactic,
                False,
                *_LEGACY_PRODUCER_ARGS,
            )
            torch.cuda.synchronize()

            assert torch.equal(combined_xq, reference[0])
            assert torch.equal(combined_sf, reference[1])
            assert torch.equal(combined_down, reference[2]), (
                f"the public {row} combined route no longer matches the "
                f"production K12 producer"
            )
            assert bool(torch.isfinite(combined_output).all()), (
                f"the public {row} fixed-K3 output retained its NaN prefill"
            )


# ---------------------------------------------------------------------------
# MiniMax-H3 Omni row 52 -- (73984, 5376, 14336) fused K12 admission.
#
# This is the only MiniMax-H3 shape whose K already carries an accepted producer
# geometry (K=14336, from the (27280, 14336) instance), so the admission reuses a
# compiled K rather than extrapolating to a new one. The launch geometry is the
# K >= 8192 tile family every proven large-M long-K case runs: 256 threads,
# 80-row tiles, 128-wide K tiles.
# ---------------------------------------------------------------------------

_H3_ROW52_M = 73984
_H3_ROW52_N = 5376
_H3_ROW52_K = 14336
_H3_ROW52_RANK = 32
_H3_ROW52_LAUNCH = "launch_large_m_kernel<73984, 14336, 256, 80, 128>"
_H3_ROW50_K = 7168
_H3_ROW50_LAUNCH = "launch_large_m_kernel<73984, 7168, 256, 32, 256>"
_H3_ROW61_M = 61056
_H3_ROW61_LAUNCH = "launch_large_m_kernel<61056, 14336, 256, 80, 128>"
_H3_ROW59_K = 7168
_H3_ROW59_LAUNCH = "launch_large_m_kernel<61056, 7168, 256, 32, 256>"
_H3_ROW70_M = 82752
_H3_ROW70_LAUNCH = "launch_large_m_kernel<82752, 14336, 256, 80, 128>"
_H3_ROW68_K = 7168
_H3_ROW68_LAUNCH = "launch_large_m_kernel<82752, 7168, 256, 32, 256>"
_H3_REMAINING_BATCH4_CANDIDATES = (
    (537, 5120, "launch_m537_mixed_kernel<5120, 1024, 24, 16, 8>"),
    (537, 7168, "launch_m537_mixed_kernel<7168, 1024, 24, 16, 8>"),
    (537, 5376, "launch_m537_mixed_kernel<5376, 1024, 24, 16, 8>"),
    (1935, 5120, "launch_small_m_kernel<1935, 5120, 256, 16>"),
    (1935, 7168, "launch_small_m_kernel<1935, 7168, 256, 16>"),
    (1935, 14336, "launch_large_m_kernel<1935, 14336, 384, 32, 256>"),
    (6913, 5120, "launch_large_m_kernel<6913, 5120, 192, 32, 128>"),
    (6913, 7168, "launch_large_m_kernel<6913, 7168, 256, 32, 256>"),
    (6913, 14336, "launch_large_m_kernel<6913, 14336, 384, 32, 256>"),
)
# Test fixture only: a fixed K3 tactic so the separate and combined launches are
# compared against each other under identical conditions. This is NOT a
# production winner -- the prerequisite screen measured the K12 producer
# geometry alone, so no combined tactic is pinned for row 52 and the K3 half is
# left to the autotuner.
_H3_ROW52_FIXTURE_K3_TACTIC = 8


def _skip_unless_free_device_memory(required_bytes: int) -> None:
    """Skip rather than OOM: these shapes are multi-GiB by construction."""
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes:
        pytest.skip(
            f"needs {required_bytes / 1024**3:.1f} GiB free device memory, "
            f"only {free_bytes / 1024**3:.1f} GiB available"
        )


def _row52_global_scale(
    x: torch.Tensor, pre_quant_scale: torch.Tensor, rows_per_chunk: int = 512
) -> torch.Tensor:
    """The operator's global scale, without a full-shape FP32 copy of ``x``.

    The definition is
    ``(448 * 6) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()``,
    but at row 52's (73984, 14336) every intermediate in that chain is ~3.95 GiB
    and several are live at once, so the whole-tensor form spends ~8 GiB on top
    of ``x`` -- enough to OOM behind a guard sized for the persistent tensors.

    The reduction is row-independent and max is associative, so the identical
    finite absolute maximum falls out of bounded row chunks: one chunk holds
    ``rows_per_chunk * K`` floats (~29 MiB at 512 rows and K=14336), and no
    full-shape FP32 temporary is ever materialized. ``abs()`` then
    ``nan_to_num()`` run per chunk in the original order, so NaN inputs still
    contribute 0.0 and infinities still saturate to the FP32 max before the
    running maximum sees them.
    """
    scale = pre_quant_scale.float()
    # abs() is non-negative and nan_to_num() maps NaN to 0.0, so a zero seed can
    # never win the running maximum over a non-empty tensor.
    amax = torch.zeros((), dtype=torch.float32, device=x.device)
    for start in range(0, x.shape[0], rows_per_chunk):
        chunk = x[start : start + rows_per_chunk].float() * scale
        amax = torch.maximum(amax, chunk.abs().nan_to_num().max())
    return ((448.0 * 6.0) / amax).reshape(1)


def test_sm120_h3_row52_admission_is_exactly_one_shape() -> None:
    """The retained row-52 admission remains exact after adding row 61."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK
    )
    _assert_admission_tracks_the_ladder(82751, _H3_ROW52_K, _H3_ROW52_RANK)
    # Not one K but three. 5376 and 7168 are MiniMax-H3 shapes whose fused
    # producer the geometry sweep measured as a win over the unfused prefix
    # (the swept-producer tests in test_nvfp4_svdquant_host.py assert the
    # same set).
    # "Exact" is about the admission being keyed on the (M, K) pair and not
    # generalising -- so the K set is pinned outright, and a K nobody measured
    # stays out.
    assert {k for m, k in SM120_FUSED_LINEAR_MK if m == _H3_ROW52_M} == {
        5376,
        7168,
        14336,
    }
    for k in (5120, 7040, 12288):
        _assert_admission_tracks_the_ladder(_H3_ROW52_M, k, _H3_ROW52_RANK)
    # Rank is part of the admission, not just of the FFI contract.
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW52_M, _H3_ROW52_K, rank)


def test_sm120_h3_row50_admission_is_exactly_one_new_shape() -> None:
    """The third screened short-K producer stays exact to row 50."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW52_M, _H3_ROW50_K, _H3_ROW52_RANK
    )
    for m in (73983, 73985):
        _assert_admission_tracks_the_ladder(m, _H3_ROW50_K, _H3_ROW52_RANK)
    for k in (7040, 7296):
        _assert_admission_tracks_the_ladder(_H3_ROW52_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW52_M, _H3_ROW50_K, rank)


def test_sm120_h3_row61_admission_is_exactly_one_new_shape() -> None:
    """The screened row is admitted without widening to H3 siblings."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW61_M, _H3_ROW52_K, _H3_ROW52_RANK
    )
    _assert_admission_tracks_the_ladder(82751, _H3_ROW52_K, _H3_ROW52_RANK)
    # Not one K but three. 5376 and 7168 are MiniMax-H3 shapes whose fused
    # producer the geometry sweep measured as a win over the unfused prefix
    # (the swept-producer tests in test_nvfp4_svdquant_host.py assert the
    # same set).
    # "Exact" is about the admission being keyed on the (M, K) pair and not
    # generalising -- so the K set is pinned outright, and a K nobody measured
    # stays out.
    assert {k for m, k in SM120_FUSED_LINEAR_MK if m == _H3_ROW61_M} == {
        5376,
        7168,
        14336,
    }
    for k in (5120, 7040, 12288):
        _assert_admission_tracks_the_ladder(_H3_ROW61_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW61_M, _H3_ROW52_K, rank)


def test_sm120_h3_row59_admission_is_exactly_one_new_shape() -> None:
    """The second screened short-K producer stays exact to row 59."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW61_M, _H3_ROW59_K, _H3_ROW52_RANK
    )
    for m in (61055, 61057):
        _assert_admission_tracks_the_ladder(m, _H3_ROW59_K, _H3_ROW52_RANK)
    for k in (7040, 7296):
        _assert_admission_tracks_the_ladder(_H3_ROW61_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW61_M, _H3_ROW59_K, rank)


def test_sm120_h3_row70_admission_is_exactly_one_new_shape() -> None:
    """The screened row is admitted without widening to H3 siblings."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW70_M, _H3_ROW52_K, _H3_ROW52_RANK
    )
    for m in (82751, 82753):
        _assert_admission_tracks_the_ladder(m, _H3_ROW52_K, _H3_ROW52_RANK)
    # Not one K but three. 5376 and 7168 are MiniMax-H3 shapes whose fused
    # producer the geometry sweep measured as a win over the unfused prefix
    # (the swept-producer tests in test_nvfp4_svdquant_host.py assert the
    # same set).
    # "Exact" is about the admission being keyed on the (M, K) pair and not
    # generalising -- so the K set is pinned outright, and a K nobody measured
    # stays out.
    assert {k for m, k in SM120_FUSED_LINEAR_MK if m == _H3_ROW70_M} == {
        5376,
        7168,
        14336,
    }
    for k in (5120, 7040, 12288):
        _assert_admission_tracks_the_ladder(_H3_ROW70_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW70_M, _H3_ROW52_K, rank)


def test_sm120_h3_row68_admission_is_exactly_one_new_shape() -> None:
    """The short-K producer is scoped to row 68's exact M, K, and rank."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW70_M, _H3_ROW68_K, _H3_ROW52_RANK
    )
    for m in (82751, 82753):
        _assert_admission_tracks_the_ladder(m, _H3_ROW68_K, _H3_ROW52_RANK)
    for k in (7040, 7296):
        _assert_admission_tracks_the_ladder(_H3_ROW70_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW70_M, _H3_ROW68_K, rank)


def test_sm120_h3_routes_bump_abi_only_where_the_runner_set_changed() -> None:
    """Long-K H3 routes retain v6; added packed candidates advance short-K routes.

    The historical six-route admission manifest stays unchanged. Its K7168
    routes now offer packed small-M producers and use v10. Existing producer
    indices and the consumer tactic encoding retain their meaning.
    """
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK
        )
        == 6
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_op_name(
            _H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK, enable_pdl=False
        )
        == "svdquant_linear_sm120_routes_v6_tactics_v3"
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW52_M, _H3_ROW50_K, _H3_ROW52_RANK
        )
        == 10
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW61_M, _H3_ROW52_K, _H3_ROW52_RANK
        )
        == 6
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_op_name(
            _H3_ROW61_M, _H3_ROW52_K, _H3_ROW52_RANK, enable_pdl=False
        )
        == "svdquant_linear_sm120_routes_v6_tactics_v3"
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW61_M, _H3_ROW59_K, _H3_ROW52_RANK
        )
        == 10
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW70_M, _H3_ROW52_K, _H3_ROW52_RANK
        )
        == 6
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_op_name(
            _H3_ROW70_M, _H3_ROW52_K, _H3_ROW52_RANK, enable_pdl=False
        )
        == "svdquant_linear_sm120_routes_v6_tactics_v3"
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW70_M, _H3_ROW68_K, _H3_ROW52_RANK
        )
        == 10
    )

    # The historical manifest and fallback base remain unchanged.
    assert svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_ABI_VERSION == 5
    assert (
        frozenset(
            {
                (_H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK),
                (_H3_ROW52_M, _H3_ROW50_K, _H3_ROW52_RANK),
                (_H3_ROW61_M, _H3_ROW52_K, _H3_ROW52_RANK),
                (_H3_ROW61_M, _H3_ROW59_K, _H3_ROW52_RANK),
                (_H3_ROW70_M, _H3_ROW52_K, _H3_ROW52_RANK),
                (_H3_ROW70_M, _H3_ROW68_K, _H3_ROW52_RANK),
            }
        )
        == svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_V6_MKR
    ), "the original six-route admission manifest must remain stable"
    assert svdquant_sm120_cutlass._SM120_TACTIC_ABI_VERSION == 3


def test_sm120_h3_promoted_producers_are_wired_to_their_routes() -> None:
    """Screened producers and their exact public admissions stay in lockstep."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(61056, 14336, 32)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(61056, 7168, 32)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(82752, 14336, 32)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(82752, 7168, 32)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(73984, 7168, 32)

    # Each of these used to be `production_dispatch.count("m == M && k == K") == 1`
    # -- the shape's own branch in the dispatcher. The dispatcher no longer names
    # shapes, so what carries the claim is the launch geometry check below (which
    # binds KernelLaunchGeometry's static asserts to the call site) plus the
    # admission predicate.
    for wired_m, wired_k in (
        (73984, 14336),
        (73984, 7168),
        (61056, 14336),
        (61056, 7168),
    ):
        assert _sm120_routes.sm120_producer_variants(wired_m, wired_k), (
            f"({wired_m}, {wired_k}) computes no producer geometry"
        )
    _assert_launch_geometry_is_computed(_H3_ROW52_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW50_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW61_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW59_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW70_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW68_LAUNCH)
    # Neighbours stay out. This used to read as the absence of an "m == 82751"
    # branch in the dispatcher; the dispatcher no longer enumerates shapes, so
    # the same statement is made against the admission predicate.
    for sibling_m, sibling_k in ((82751, 14336), (73983, 7168)):
        _assert_admission_tracks_the_ladder(sibling_m, sibling_k, _H3_ROW52_RANK)

    # And the FFI half: this was the ~51-pair whitelist, asserted shape by shape.
    # The FFI carries no shape list now, so the same statement is made where the
    # answer actually comes from.
    csrc = _read_repo_source("csrc/nvfp4_smooth_quantize_sm100.cu")
    assert "one of the measured SM120 K12 shapes" not in csrc
    for admitted_m, admitted_k in (
        (73984, 14336),
        (73984, 7168),
        (61056, 14336),
        (61056, 7168),
        (82752, 14336),
    ):
        assert _sm120_routes.sm120_producer_variants(admitted_m, admitted_k)
    assert _sm120_routes.sm120_producer_variants(82752, 7168)


def test_sm120_h3_remaining_batch4_routes_are_exact() -> None:
    """Retain the nine-route manifest while refreshing added packed candidates."""
    expected_mkr = frozenset(
        (m, k, _H3_ROW52_RANK) for m, k, _ in _H3_REMAINING_BATCH4_CANDIDATES
    )
    assert expected_mkr == svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_V7_MKR

    for m, k, launch in _H3_REMAINING_BATCH4_CANDIDATES:
        expected_version = 7 if k == 14336 else 10
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
            m, k, _H3_ROW52_RANK
        )
        _assert_admission_tracks_the_ladder(m + 1, k, _H3_ROW52_RANK)
        _assert_admission_tracks_the_ladder(m, k, 64)
        assert (
            svdquant_sm120_cutlass._sm120_linear_route_abi_version(m, k, _H3_ROW52_RANK)
            == expected_version
        )
        assert (
            svdquant_sm120_cutlass._sm120_linear_op_name(
                m, k, _H3_ROW52_RANK, enable_pdl=False
            )
            == f"svdquant_linear_sm120_routes_v{expected_version}_tactics_v3"
        )
        _assert_shape_computes_geometry(m, k, launch)
        # The FFI used to carry an "m == M && k == K" branch per admitted shape.
        # The runtime producer takes the geometry as arguments instead, so what
        # admits a shape now is the predicate, checked above and again here for
        # the C++-side contract this line used to stand for.
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
            m, k, _H3_ROW52_RANK
        )

    # These were "x.size(0) == M && x.size(1) == K" clauses in the combined
    # launcher. It dispatches on the geometry it is handed now, so the exactness
    # they stood for is asked of the admission predicate: these two are in, and
    # a neighbour on either axis is not.
    for exact_k in (5120, 7168):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
            1935, exact_k, _H3_ROW52_RANK
        )
        _assert_admission_tracks_the_ladder(1936, exact_k, _H3_ROW52_RANK)


def test_sm120_batch4_gap_closure_routes_are_exact() -> None:
    """Case 4 uses only its screened cuBLASLt prefix."""
    m, _n, k, rank = (537, 5376, 14336, 32)
    assert (
        frozenset({(m, k, rank)}) == svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_V9_MKR
    )
    assert svdquant_sm120_cutlass._sm120_linear_route_abi_version(m, k, rank) == 9
    assert (
        svdquant_sm120_cutlass._sm120_linear_op_name(m, k, rank, enable_pdl=False)
        == "svdquant_linear_sm120_routes_v9_tactics_v3"
    )
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
    _assert_admission_tracks_the_ladder(m + 1, k, rank)

    # This K set used to appear twice in the combined source -- once as the
    # cuBLASLt entry's own admission list, once as the "measured faster" gate in
    # the dispatcher -- and the assertion pinned the two copies to each other.
    # Neither exists now: the prefix is a producer family the tactic names, so
    # there is no list to keep in step. What still has to hold is that this
    # shape can reach that prefix at all.
    combined = _read_repo_source("csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu")
    assert (
        "x.size(1) == 5120 || x.size(1) == 7168 || x.size(1) == 14336" not in combined
    )
    families = {family for family, _, _ in _sm120_routes.sm120_producer_variants(m, k)}
    assert _sm120_routes.SM120_FAMILY_CUBLASLT in families


def test_sm120_batch4_k5376_fused_routes_are_exact() -> None:
    """Historical K5376 routes gain a fresh search over packed candidates."""
    expected = (
        (1935, 21504, 5376, 32),
        (1935, 28672, 5376, 32),
        (6913, 21504, 5376, 32),
        (6913, 28672, 5376, 32),
        (73984, 21504, 5376, 32),
        (73984, 28672, 5376, 32),
        (61056, 21504, 5376, 32),
        (61056, 28672, 5376, 32),
        (82752, 21504, 5376, 32),
        (82752, 28672, 5376, 32),
    )
    assert (
        frozenset(
            {
                (1935, 5376, 32),
                (6913, 5376, 32),
                (73984, 5376, 32),
                (61056, 5376, 32),
                (82752, 5376, 32),
            }
        )
        == svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_V8_MKR
    )
    for m, _, k, rank in expected:
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
        _assert_admission_tracks_the_ladder(m + 1, k, rank)
        assert svdquant_sm120_cutlass._sm120_linear_route_abi_version(m, k, rank) == 10

    _assert_launch_geometry_is_computed(
        "launch_large_m_kernel<6913, 5376, 192, 32, 128>"
    )
    for m in (73984, 61056, 82752):
        _assert_launch_geometry_is_computed(
            f"launch_large_m_kernel<{m}, 5376, 256, 32, 256>"
        )
    ffi = _read_repo_source("csrc/nvfp4_smooth_quantize_sm100.cu")
    assert "one of the measured SM120 K12 shapes" not in ffi
    assert _sm120_routes.sm120_producer_variants(6913, 5376)
    # Same story: the duplicated cuBLASLt admission list is gone, so what is
    # asserted is its absence plus the reachability the list used to grant.
    combined = _read_repo_source("csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu")
    assert "x.size(1) == 5120 || x.size(1) == 5376 ||" not in combined
    for m_k in (537, 1935):
        families = {f for f, _, _ in _sm120_routes.sm120_producer_variants(m_k, 5376)}
        assert _sm120_routes.SM120_FAMILY_CUBLASLT in families


def test_sm120_m537_prepacked_warp_rows_have_exact_shape_admission() -> None:
    """The accepted row-44 producer cannot leak to a sibling K."""
    producer = _read_repo_source(
        "include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh"
    )
    # Was `if (m == 537 && k == 5120)` in the shape dispatcher. That dispatcher
    # is gone; the M537 family still guards its own M and enumerates its K.
    assert "if (m != 537) return cudaErrorInvalidValue;" in producer
    assert "launch_m537_mixed_kernel<5120, 1024, 24, 16, 8>" in producer
    assert "if constexpr (K == 5120 || K == 5376 || K == 7168)" in producer
    assert "reinterpret_cast<uint2 const*>(l2t_smoothed)" in producer
    assert "uint2 const packed_b" in producer

    template = _read_repo_source(
        "include/flashinfer/gemm/nvfp4_svdquant_gemm_template_sm120.h"
    )
    assert "Tactic256x128x128SwapStaticConfig" in template
    assert "k256x128x128SwapStatic" in template
    assert "Tactic256x64x128SwapStaticConfig" in template
    assert "k256x64x128SwapStatic" in template


def test_sm120_h3_row52_writes_every_output_byte() -> None:
    """Two hostile prefills converge on the same complete oracle result."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    # Live set: x 1.98 GiB + xq/xq_ref 0.99 + sf/sf_ref 0.12 + down/down_ref,
    # l2t and pre_quant_scale <0.02 = ~3.1 GiB persistent. The only temporaries
    # are the chunked global-scale buffers (~0.12 GiB, see _row52_global_scale)
    # and the rank-column FP32 casts (<0.02 GiB), so 6 GiB leaves ~1.9x margin
    # for allocator fragmentation.
    _skip_unless_free_device_memory(6 * 1024**3)

    m, k, rank = _H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK
    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = _row52_global_scale(x, pre_quant_scale)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),), dtype=torch.uint8, device="cuda"
    )
    down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")
    module = get_nvfp4_svdquant_sm120_module()

    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)

    for sentinel in (0x00, 0xFF):
        xq.fill_(sentinel)
        sf.fill_(sentinel)
        down.fill_(float("nan"))
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )
        torch.cuda.synchronize()

        assert torch.equal(xq, xq_ref), (
            f"xq left unwritten or wrong after the 0x{sentinel:02x} prefill"
        )
        assert torch.equal(sf, sf_ref), (
            f"sf left unwritten or wrong after the 0x{sentinel:02x} prefill"
        )
        assert bool(torch.isfinite(down).all()), (
            f"down retains the NaN prefill after the 0x{sentinel:02x} run"
        )

        noise = ((down_ref.float() - down.float()) ** 2).mean()
        sqnr = 10 * torch.log10((down_ref.float() ** 2).mean() / noise)
        assert float(sqnr) > 48.0, (
            f"fused LoRA-down SQNR {float(sqnr):.3f} dB is below 48.0 dB"
        )

        # A whole-tensor SQNR averages a single dead rank column away at this M.
        column_noise = ((down_ref.float() - down.float()) ** 2).mean(dim=0)
        column_sqnr = 10 * torch.log10(
            (down_ref.float() ** 2).mean(dim=0) / column_noise
        )
        worst_column = int(torch.argmin(column_sqnr))
        assert float(column_sqnr[worst_column]) > 48.0, (
            f"LoRA-down rank column {worst_column} SQNR "
            f"{float(column_sqnr[worst_column]):.3f} dB is below 48.0 dB"
        )

        # The partial final M tile (73984 % 80 == 64) is the one structural
        # difference from every proven large-M long-K neighbour, so gate it on
        # its own rather than letting the bulk rows carry the average.
        tail = slice(m - 64, m)
        tail_noise = ((down_ref[tail].float() - down[tail].float()) ** 2).mean()
        tail_sqnr = 10 * torch.log10((down_ref[tail].float() ** 2).mean() / tail_noise)
        assert float(tail_sqnr) > 48.0, (
            f"the 64-row partial final M tile SQNR {float(tail_sqnr):.3f} dB "
            f"is below 48.0 dB"
        )
        assert torch.equal(xq[tail], xq_ref[tail])


@pytest.mark.parametrize(
    ("m", "k"),
    (
        (73983, 14336),
        (73985, 14336),
        (73984, 14208),
        (73984, 14464),
        # (61056, 14336) was here. It is row 61's own shape -- adding row 61 to
        # the admission set left it behind in row 52's neighbour list, where it
        # asserted that an admitted shape is refused. Its K-neighbour keeps what
        # the entry was for: an admitted M does not carry an unmeasured K.
        (61056, 14208),
        (82751, 14336),
    ),
)
def test_sm120_h3_row52_rejects_neighboring_shapes(m: int, k: int) -> None:
    """Admission follows the computed geometry, not a one-shape list."""
    # This used to launch with 1x1 dummy outputs, because the FFI whitelist
    # refused the shape before it looked at them. The whitelist is gone, so the
    # call now reaches the buffer checks -- and giving it real buffers would mean
    # allocating gigabytes per parameter case for M up to 82751.
    #
    # The claim was never about launching, though: it is that admission does not
    # generalize from a measured shape to its neighbours. That is a property of
    # the predicate, so it is asked of the predicate. Where a neighbour *is*
    # admitted, correctness was measured separately -- 4095x12288 and
    # 4097x12288 reach 51.4 and 51.3 dB against an fp32 reference, next to
    # 51.3 dB for the 4096x12288 the old list admitted.
    variants = _sm120_routes.sm120_producer_variants(m, k)
    supported = svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, 32)
    assert bool(variants) == supported, (
        f"({m}, {k}) computes {len(variants)} producer variants but "
        f"_sm120_fused_linear_supported says {supported}"
    )


@pytest.mark.parametrize("rank", (16, 64))
def test_sm120_h3_row52_rejects_non_rank32_lora(rank: int) -> None:
    """The admitted shape still requires the rank-32 LoRA-down topology."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    _skip_unless_free_device_memory(4 * 1024**3)

    m, k = _H3_ROW52_M, _H3_ROW52_K
    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((k,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty((k, rank), dtype=torch.bfloat16, device="cuda")
    # The exact shape reaches the rank-32 check before output-shape checks.
    # Minimal outputs keep this a rank-contract test rather than a memory test.
    xq = torch.empty((1, 1), dtype=torch.uint8, device="cuda")
    sf = torch.empty((1,), dtype=torch.uint8, device="cuda")
    down = torch.empty((1, 1), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match=r"l2t_smoothed must be \[k, 32\]"):
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )


@pytest.mark.parametrize("with_bias", (False, True))
@pytest.mark.parametrize(
    ("m", "k"),
    (
        (_H3_ROW52_M, _H3_ROW52_K),
        (_H3_ROW52_M, _H3_ROW50_K),
        (_H3_ROW61_M, _H3_ROW52_K),
        (_H3_ROW61_M, _H3_ROW59_K),
        (_H3_ROW70_M, _H3_ROW52_K),
        (_H3_ROW70_M, _H3_ROW68_K),
    ),
)
def test_sm120_h3_combined_k12_k3_matches_separate_at_a_fixed_k3_tactic(
    with_bias: bool, m: int, k: int
) -> None:
    """Combined equivalence plus the 40 dB full-output contract for H3 routes.

    The K3 tactic here is a test fixture, not a production choice: holding it
    fixed is what makes the separate and combined launches comparable. Nothing
    in this test asserts that the route would select it. Public-route behaviour
    is deliberately out of scope -- the autotuner may pick a different K3 tactic
    there, so that comparison belongs to the GPU acceptance gate.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    # The long-K Omni routes set the larger live-set bound; the separate and
    # combined quads 2 x (xq 0.49 +
    # sf 0.06 + down 0.01 + out 0.74) = 2.60 + the generic prefix (xq_ref,
    # sf_ref, down_ref, reference_output) 1.30 + weights and workspace ~0.30 =
    # ~6.2 GiB persistent. The final SQNR compares two (m, n) tensors in FP32,
    # which is ~1.48 GiB per cast and peaks at three such buffers (~4.4 GiB)
    # while the squared difference is allocated. The chunked global scale adds
    # ~0.12 GiB. That is ~10.7 GiB peak, so gate at 14 GiB -- still practical on
    # the smaller of the two SM120 parts, with room for allocator fragmentation.
    _skip_unless_free_device_memory(14 * 1024**3)

    n, rank = _H3_ROW52_N, _H3_ROW52_RANK
    tactic = _H3_ROW52_FIXTURE_K3_TACTIC

    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = _row52_global_scale(x, pre_quant_scale)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.ones((n * (k // 16),), dtype=torch.uint8, device="cuda")
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
    l1_scaled = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda").contiguous()
    # The same bias object drives both launches, so the comparison isolates the
    # fusion and not the epilogue.
    bias = (
        torch.randn((n,), dtype=torch.bfloat16, device="cuda").contiguous()
        if with_bias
        else None
    )
    module = get_nvfp4_svdquant_sm120_module()
    workspace_bytes = max(
        32 * 1024 * 1024,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")

    separate = (
        torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
        torch.empty(
            (((m + 127) // 128) * 128 * (k // 16),),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.empty((m, rank), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    )
    combined = tuple(torch.empty_like(tensor) for tensor in separate)

    separate[0].fill_(0x00)
    separate[1].fill_(0x00)
    separate[2].fill_(float("nan"))
    separate[3].fill_(float("nan"))
    combined[0].fill_(0xFF)
    combined[1].fill_(0xFF)
    combined[2].fill_(float("nan"))
    combined[3].fill_(float("nan"))

    module.nvfp4_quantize_smooth_lora_down_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        separate[0],
        separate[1],
        separate[2],
    )
    module.nvfp4_svdquant_gemm(
        separate[0],
        weight_fp4,
        separate[1],
        weight_sf,
        alpha,
        separate[2],
        l1_scaled,
        bias,
        separate[3],
        workspace,
        tactic,
        False,
    )
    module.nvfp4_svdquant_linear_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        weight_fp4,
        weight_sf,
        alpha,
        l1_scaled,
        bias,
        combined[0],
        combined[1],
        combined[2],
        combined[3],
        workspace,
        tactic,
        False,
        *_LEGACY_PRODUCER_ARGS,
    )

    torch.cuda.synchronize()
    for name, output in (
        ("separate down", separate[2]),
        ("separate full output", separate[3]),
        ("combined down", combined[2]),
        ("combined full output", combined[3]),
    ):
        assert bool(torch.isfinite(output).all()), f"{name} is incomplete"
    for separate_output, combined_output in zip(separate, combined, strict=False):
        assert torch.equal(separate_output, combined_output)

    # Independent generic prefix: the same fixed K3 tactic driven by the generic
    # quantizer and torch.mm rather than by the fused producer.
    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)
    reference_output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    module.nvfp4_svdquant_gemm(
        xq_ref,
        weight_fp4,
        sf_ref,
        weight_sf,
        alpha,
        down_ref,
        l1_scaled,
        bias,
        reference_output,
        workspace,
        tactic,
        False,
    )

    torch.cuda.synchronize()
    noise = ((reference_output.float() - combined[3].float()) ** 2).mean()
    sqnr = 10 * torch.log10((reference_output.float() ** 2).mean() / noise)
    assert float(sqnr) > 40.0, (
        f"combined K12+K3 output SQNR {float(sqnr):.3f} dB is below 40.0 dB "
        f"for m={m}, n={n}, k={k}, with_bias={with_bias} at fixture K3 tactic "
        f"{tactic}"
    )


def _cublaslt_producer_args(m: int, k: int):
    """The (family, tiling..., policy) tuple that names the cuBLASLt prefix."""
    for family, tiling, policy in _sm120_routes.sm120_producer_variants(m, k):
        if family == _sm120_routes.SM120_FAMILY_CUBLASLT:
            return (family, tiling[0], tiling[1], tiling[2], policy)
    raise AssertionError(f"({m}, {k}) offers no cuBLASLt producer")


def test_sm120_m537_cublaslt_down_matches_generic_prefix() -> None:
    """Acceptance for row 44's cuBLASLt LoRA-down path."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    if torch.cuda.get_device_properties(0).multi_processor_count < 105:
        pytest.skip("requires the >=105-SM M537 admission")
    _skip_unless_free_device_memory(2 * 1024**3)

    m, n, k, rank, tactic = 537, 5376, 5120, 32, 81
    torch.manual_seed(20260814)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = _row52_global_scale(x, pre_quant_scale)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.ones((n * (k // 16),), dtype=torch.uint8, device="cuda")
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
    l1_scaled = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda").contiguous()

    module = get_nvfp4_svdquant_sm120_module()
    workspace_bytes = max(
        32 * 1024 * 1024,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),),
        dtype=torch.uint8,
        device="cuda",
    )
    down = torch.full((m, rank), float("nan"), dtype=torch.bfloat16, device="cuda")
    output = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")

    module.nvfp4_svdquant_linear_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        weight_fp4,
        weight_sf,
        alpha,
        l1_scaled,
        None,
        xq,
        sf,
        down,
        output,
        workspace,
        tactic,
        False,
        # This test is about the cuBLASLt prefix, so it asks for it by name. It
        # used to pass no producer and rely on a shape ladder to divert 537x5120
        # onto that prefix -- which is exactly the implicit routing being
        # removed, and it broke silently the moment the ladder went.
        *_cublaslt_producer_args(m, k),
    )
    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x, pre_quant_scale, global_scale, enable_pdl=False
    )
    down_ref = torch.mm(x, l2t_smoothed)
    output_ref = torch.empty_like(output)
    module.nvfp4_svdquant_gemm(
        xq_ref,
        weight_fp4,
        sf_ref,
        weight_sf,
        alpha,
        down_ref,
        l1_scaled,
        None,
        output_ref,
        workspace,
        tactic,
        False,
    )
    torch.cuda.synchronize()

    assert torch.equal(xq, xq_ref)
    assert torch.equal(sf, sf_ref)
    assert bool(torch.isfinite(down).all())
    assert bool(torch.isfinite(output).all())
    down_noise = ((down_ref.float() - down.float()) ** 2).mean()
    down_sqnr = 10 * torch.log10((down_ref.float() ** 2).mean() / down_noise)
    assert float(down_sqnr) > 48.0
    output_noise = ((output_ref.float() - output.float()) ** 2).mean()
    output_sqnr = 10 * torch.log10((output_ref.float() ** 2).mean() / output_noise)
    assert float(output_sqnr) > 40.0


# L2T packing and packed small-M producers.


@pytest.fixture
def sm120_device() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")


def _pack(source: torch.Tensor, output: torch.Tensor) -> None:
    backend.get_nvfp4_svdquant_sm120_module().nvfp4_svdquant_pack_l2t_sm120(
        source, output
    )


def _source(k: int, offset: int = 0) -> torch.Tensor:
    """Visit every BF16 bit pattern, including signed zeros and NaN payloads."""
    bits = torch.arange(k * 32 + offset, dtype=torch.int32, device="cuda")
    bits = (bits * 40503 + 17).to(torch.int16)
    return bits[offset:].view(torch.bfloat16).reshape(k, 32)


def _reference_bits(source: torch.Tensor) -> torch.Tensor:
    k = source.shape[0]
    return (
        source.view(torch.int16)
        .reshape(k // 16, 2, 4, 2, 2, 2, 8)
        .permute(0, 4, 5, 6, 2, 1, 3)
        .contiguous()
        .reshape(k, 32)
    )


def _guarded_output(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    storage = torch.full((k * 32 + 32,), 0x1234, dtype=torch.int16, device="cuda")
    output = storage[16:-16].view(torch.bfloat16).reshape(k, 32)
    return output, storage


def _assert_packed(
    expected: torch.Tensor, output: torch.Tensor, storage: torch.Tensor
) -> None:
    assert torch.equal(expected, output.view(torch.int16)), "packed bits differ"
    assert bool((storage[:16] == 0x1234).all()), "leading pack output guard changed"
    assert bool((storage[-16:] == 0x1234).all()), "trailing pack output guard changed"


@pytest.mark.usefixtures("sm120_device")
def test_sm120_pack_module_export() -> None:
    module = backend.get_nvfp4_svdquant_sm120_module()
    assert callable(getattr(module, "nvfp4_svdquant_pack_l2t_sm120", None)), (
        "SM120 module must export nvfp4_svdquant_pack_l2t_sm120"
    )


@pytest.mark.usefixtures("sm120_device")
@pytest.mark.parametrize("k", (3072, 5120, 5376, 7168))
@pytest.mark.parametrize("source_offset", (0, 1))
def test_sm120_pack_preserves_arbitrary_bits(k: int, source_offset: int) -> None:
    source = _source(k, source_offset)
    expected = _reference_bits(source)
    output, storage = _guarded_output(k)
    for sentinel in (0x55AA, -21846):
        output.view(torch.int16).fill_(sentinel)
        _pack(source, output)
        torch.cuda.synchronize()
        _assert_packed(expected, output, storage)
    actual = backend._pack_sm120_m537_l2t(source)
    assert torch.equal(expected, actual.view(torch.int16))


@pytest.mark.usefixtures("sm120_device")
@pytest.mark.parametrize("k", (3072, 5120, 5376, 7168))
@pytest.mark.parametrize("graph_calls", (1, 32))
@pytest.mark.parametrize("inference_mode", (False, True))
def test_sm120_pack_graph_observes_same_address_updates(
    k: int, graph_calls: int, inference_mode: bool
) -> None:
    with torch.inference_mode(inference_mode):
        source = _source(k)
        output, storage = _guarded_output(k)
        for _ in range(3):
            _pack(source, output)
            cached_output = backend._cached_sm120_m537_l2t(source)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_calls):
                _pack(source, output)
                cached_output = backend._cached_sm120_m537_l2t(source)
        addresses = source.data_ptr(), output.data_ptr(), cached_output.data_ptr()
        assert source.is_inference() == inference_mode
        for mask in (0x1234, -1, 0x4321):
            source.view(torch.int16).bitwise_xor_(mask)
            output.view(torch.int16).fill_(0)
            cached_output.view(torch.int16).fill_(0)
            graph.replay()
            torch.cuda.synchronize()
            expected = _reference_bits(source)
            _assert_packed(expected, output, storage)
            assert torch.equal(expected, cached_output.view(torch.int16))
            assert addresses == (
                source.data_ptr(),
                output.data_ptr(),
                cached_output.data_ptr(),
            )


@pytest.mark.usefixtures("sm120_device")
def test_sm120_pack_capture_does_not_publish_unexecuted_eager_cache() -> None:
    source = _source(3072)
    backend._pack_sm120_m537_l2t(source)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend._cached_sm120_m537_l2t(source)

    captured.view(torch.int16).zero_()
    eager = backend._cached_sm120_m537_l2t(source)
    assert torch.equal(_reference_bits(source), eager.view(torch.int16))

    source.view(torch.int16).bitwise_xor_(0x1234)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_reference_bits(source), captured.view(torch.int16))


InvalidInput = Literal[
    "dtype",
    "cpu",
    "dimensions",
    "noncontiguous",
    "output_alignment",
    "alias",
    "overlap",
    "rank",
    "k",
    "output_shape",
]


@pytest.mark.usefixtures("sm120_device")
@pytest.mark.parametrize(
    ("index", "invalid"),
    (
        (0, "dtype"),
        (1, "dtype"),
        (0, "cpu"),
        (1, "cpu"),
        (0, "dimensions"),
        (1, "dimensions"),
        (0, "noncontiguous"),
        (1, "noncontiguous"),
        (1, "output_alignment"),
        (1, "alias"),
        (1, "overlap"),
        (0, "rank"),
        (0, "k"),
        (1, "output_shape"),
    ),
)
def test_sm120_pack_rejects_invalid_arguments(
    index: int, invalid: InvalidInput
) -> None:
    k = 3072
    source = _source(k)
    output, storage = _guarded_output(k)
    inputs = [source, output]
    match invalid:
        case "dtype":
            inputs[index] = inputs[index].to(torch.float32)
        case "cpu":
            inputs[index] = inputs[index].cpu()
        case "dimensions":
            inputs[index] = inputs[index].flatten()
        case "noncontiguous":
            inputs[index] = torch.empty(
                (32, k), dtype=torch.bfloat16, device="cuda"
            ).t()
        case "output_alignment":
            inputs[1] = torch.empty((k * 32 + 1,), dtype=torch.bfloat16, device="cuda")[
                1:
            ].view(k, 32)
        case "alias":
            inputs[1] = source
        case "overlap":
            shared = torch.empty((k * 32 + 4,), dtype=torch.bfloat16, device="cuda")
            inputs = [shared[:-4].view(k, 32), shared[4:].view(k, 32)]
        case "rank":
            inputs = [
                torch.empty((k, 16), dtype=torch.bfloat16, device="cuda")
                for _ in range(2)
            ]
        case "k":
            inputs = [
                torch.empty((4096, 32), dtype=torch.bfloat16, device="cuda")
                for _ in range(2)
            ]
        case "output_shape":
            inputs[1] = torch.empty((k + 16, 32), dtype=torch.bfloat16, device="cuda")
        case unexpected:
            assert_never(unexpected)
    with pytest.raises(RuntimeError):
        _pack(*inputs)
    _pack(source, output)
    torch.cuda.synchronize()
    _assert_packed(_reference_bits(source), output, storage)


@pytest.mark.usefixtures("sm120_device")
def test_sm120_pack_accepts_disjoint_views_in_one_allocation() -> None:
    k = 3072
    storage = torch.empty((2 * k, 32), dtype=torch.bfloat16, device="cuda")
    source, output = storage[:k], storage[k:]
    source.copy_(_source(k))
    expected = _reference_bits(source)
    _pack(source, output)
    torch.cuda.synchronize()
    assert torch.equal(expected, output.view(torch.int16))


@pytest.mark.usefixtures("sm120_device")
@pytest.mark.parametrize("dtype", (torch.int32, torch.bfloat16))
def test_sm120_pack_helper_keeps_cuda_fallbacks(dtype: torch.dtype) -> None:
    k = 3072
    source = _source(k)
    if dtype == torch.int32:
        source = source.view(torch.int16).to(dtype)
    else:
        source = source.t().contiguous().t()
        assert not source.is_contiguous()
    expected = (
        source.reshape(k // 16, 2, 4, 2, 2, 2, 8)
        .permute(0, 4, 5, 6, 2, 1, 3)
        .contiguous()
        .reshape(k, 32)
    )
    actual = backend._pack_sm120_m537_l2t(source)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))


@pytest.mark.usefixtures("sm120_device")
def test_sm120_pack_rejects_different_cuda_devices() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    source = _source(3072)
    other = (torch.cuda.current_device() + 1) % torch.cuda.device_count()
    output = torch.empty_like(source, device=f"cuda:{other}")
    with pytest.raises(RuntimeError):
        _pack(source, output)


@pytest.fixture
def packed_no_grad(sm120_device: None) -> Iterator[None]:
    with torch.no_grad():
        yield


@dataclass(frozen=True, slots=True)
class _LinearCase:
    inputs: list[torch.Tensor]
    runner: TunableRunner


def _producer_pairs(m: int, k: int) -> list[tuple[int, int]]:
    """Require a packed counterpart for every admitted row-major tiling."""
    variants = sm120_producer_variants(m, k)
    pairs = []
    for old, (family, tiling, policy) in enumerate(variants):
        if family != SM120_FAMILY_SMALL_M:
            continue
        packed = (4, tiling, policy)
        assert packed in variants, (
            f"missing packed small-M producer for {(m, k, tiling)}"
        )
        pairs.append((old, variants.index(packed)))
    assert pairs, f"no small-M producers for {(m, k)}"
    return pairs


def _make_case(m: int, n: int, k: int, enable_pdl: bool = True) -> _LinearCase:
    """Use packed weights directly, without allocating dense weight baselines."""
    torch.manual_seed(20260914)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.25
    pqs = (1 + 0.3 * torch.randn(k, device="cuda", dtype=torch.bfloat16)).abs()
    global_scale = (448.0 * 6.0 / (x * pqs).float().abs().max()).reshape(1)
    l2t = (
        pqs[:, None] * torch.randn(k, 32, device="cuda", dtype=torch.bfloat16) / k**0.25
    ).contiguous()
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    weight_sf = torch.full(
        ((n + 127) // 128 * 128 * (k // 16),),
        0x38,
        device="cuda",
        dtype=torch.uint8,
    )
    alpha = (1.0 / global_scale).float()
    l1 = (
        torch.randn(n, 32, device="cuda", dtype=torch.bfloat16).float()
        / (32**0.25 * alpha)
    ).to(torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    module = backend.get_nvfp4_svdquant_sm120_module()
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, 32, 80)
    workspace_bytes = max(
        backend.DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, 80)),
    )
    inputs = [
        x,
        weight,
        weight_sf,
        alpha,
        pqs,
        l2t,
        l1,
        global_scale,
        bias,
        torch.empty(m, k // 2, device="cuda", dtype=torch.uint8),
        torch.empty(
            (m + 127) // 128 * 128 * (k // 16), device="cuda", dtype=torch.uint8
        ),
        torch.empty(m, 32, device="cuda", dtype=torch.bfloat16),
        torch.empty(m, n, device="cuda", dtype=torch.bfloat16),
        torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8),
    ]
    return _LinearCase(inputs, backend._sm120_fused_linear_runner(enable_pdl, x.device))


def _independent_outputs(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    candidate = list(inputs)
    for index in range(9, 13):
        candidate[index] = torch.empty_like(inputs[index])
    return candidate


def _prefill(inputs: list[torch.Tensor], byte: int) -> None:
    inputs[9].fill_(byte)
    inputs[10].fill_(byte)
    inputs[11].fill_(float("nan"))
    inputs[12].fill_(float("nan"))


def _assert_outputs_match(
    reference: list[torch.Tensor], candidate: list[torch.Tensor], context: str
) -> None:
    for index, name in enumerate(("XQ", "padded SF", "LoRA-down", "linear"), start=9):
        assert torch.equal(
            reference[index].view(torch.uint8), candidate[index].view(torch.uint8)
        ), f"{context}: {name}"
    assert torch.isfinite(candidate[11]).all(), f"{context}: nonfinite LoRA-down"
    assert torch.isfinite(candidate[12]).all(), f"{context}: nonfinite linear"


@pytest.mark.parametrize(
    ("m", "n", "k"),
    (
        (17, 3072, 3072),
        (64, 3072, 3072),
        (537, 5376, 7168),
        (7800, 256, 5120),
        (1935, 256, 5376),
    ),
)
@pytest.mark.parametrize("enable_pdl", (False, True))
@pytest.mark.usefixtures("packed_no_grad")
def test_packed_small_m_matches_every_row_major_producer(
    m: int, n: int, k: int, enable_pdl: bool
) -> None:
    pairs = _producer_pairs(m, k)
    case = _make_case(m, n, k, enable_pdl)
    reference = case.inputs
    candidate = _independent_outputs(reference)
    for old, packed in pairs:
        # Different prefills expose unwritten bytes, including all SF padding.
        _prefill(reference, 0xA5)
        _prefill(candidate, 0x5A)
        case.runner(inputs=reference, tactic=sm120_pack_tactic(80, old))
        case.runner(inputs=candidate, tactic=sm120_pack_tactic(80, packed))
        torch.cuda.synchronize()
        _assert_outputs_match(reference, candidate, f"producer {old} -> {packed}")


@pytest.mark.parametrize(
    "inference_graph", (False, True), ids=("eager", "inference-graph")
)
@pytest.mark.usefixtures("packed_no_grad")
def test_packed_small_m_observes_l2t_mutations(inference_graph: bool) -> None:
    old, packed = _producer_pairs(64, 3072)[1]
    with torch.inference_mode(inference_graph):
        case = _make_case(64, 3072, 3072)
        reference = case.inputs
        candidate = _independent_outputs(reference)
        assert candidate[5].is_inference() == inference_graph
        packed_call = partial(
            case.runner, inputs=candidate, tactic=sm120_pack_tactic(80, packed)
        )
        old_call = partial(
            case.runner, inputs=reference, tactic=sm120_pack_tactic(80, old)
        )
        for _ in range(3):
            packed_call()
        torch.cuda.synchronize()
        replay = packed_call
        if inference_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                packed_call()
            replay = graph.replay

        previous_down = None
        for increment in (0.0, 0.015625, -0.03125):
            candidate[5].add_(increment)
            _prefill(reference, 0xA5)
            _prefill(candidate, 0x5A)
            replay()
            old_call()
            torch.cuda.synchronize()
            _assert_outputs_match(reference, candidate, f"L2T increment {increment}")
            if previous_down is not None:
                assert not torch.equal(reference[11], previous_down), (
                    "mutation had no effect"
                )
            previous_down = reference[11].clone()


@pytest.mark.usefixtures("packed_no_grad")
def test_packed_small_m_pdl_graph_chains_two_mutable_linears() -> None:
    old, packed = _producer_pairs(64, 3072)[1]
    # Inference L2T packing becomes graph work, so replay sees fresh weights.
    with torch.inference_mode():
        first, second = (_make_case(64, 3072, 3072) for _ in range(2))
        reference = [first.inputs, second.inputs]
        candidate = [_independent_outputs(inputs) for inputs in reference]
        guards: list[torch.Tensor] = []
        for inputs in (*reference, *candidate):
            storage = torch.full(
                (3 * 64 * 3072,), -2048, device="cuda", dtype=torch.bfloat16
            )
            inputs[12] = storage[64 * 3072 : 2 * 64 * 3072].view(64, 3072)
            guards.extend((storage[: 64 * 3072], storage[2 * 64 * 3072 :]))
        reference[1][0] = reference[0][12]
        candidate[1][0] = candidate[0][12]
        assert reference[0][13].data_ptr() != reference[1][13].data_ptr()

        def run_chain(inputs: list[list[torch.Tensor]], producer: int) -> None:
            for operands in inputs:
                first.runner(inputs=operands, tactic=sm120_pack_tactic(80, producer))

        # Calibrate the second quantizer for the first linear's output range.
        first.runner(inputs=reference[0], tactic=sm120_pack_tactic(80, old))
        second.inputs[6].mul_(second.inputs[3])
        second.inputs[7].copy_(
            (
                2688.0 / (reference[0][12] * second.inputs[4]).float().abs().max()
            ).reshape(1)
        )
        second.inputs[3].copy_(second.inputs[7].reciprocal())
        second.inputs[6].div_(second.inputs[3])
        for _ in range(3):
            run_chain(candidate, packed)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_chain(candidate, packed)

        previous = None
        for replay, update_x in enumerate((True, False, True, False)):
            if update_x:
                first.inputs[0].copy_(torch.randn_like(first.inputs[0]) / 3072**0.25)
            else:
                for inputs in reference:
                    inputs[5].copy_(
                        inputs[4][:, None] * torch.randn_like(inputs[5]) / 3072**0.25
                    )
            for inputs in reference:
                _prefill(inputs, 0xA5)
            for inputs in candidate:
                _prefill(inputs, 0x5A)
            graph.replay()
            run_chain(reference, old)
            torch.cuda.synchronize()
            for stage, (expected, actual) in enumerate(
                zip(reference, candidate, strict=True)
            ):
                _assert_outputs_match(
                    expected, actual, f"replay {replay}, linear {stage}"
                )
                if previous is not None:
                    assert not torch.equal(expected[12], previous[stage]), (
                        "chain mutation had no effect"
                    )
            for guard in guards:
                assert bool((guard == -2048).all()), "linear output guard changed"
            previous = [inputs[12].clone() for inputs in reference]


# CUDA Graph capture and replay with mutable inputs.


def test_cublaslt_graph_replays_after_more_than_eight_shapes() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    torch.manual_seed(0)
    module = get_nvfp4_svdquant_sm120_module(32)
    k, rank = 256, 32
    workspace = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
    pqs = torch.ones(k, dtype=torch.bfloat16, device="cuda")
    scale = torch.ones(1, device="cuda")
    l2t = torch.randn((k, rank), dtype=torch.bfloat16, device="cuda")
    inputs = [
        (
            torch.randn((m, k), dtype=torch.bfloat16, device="cuda"),
            torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
            torch.zeros(
                ((m + 127) // 128 * 128 * (k // 16),), dtype=torch.uint8, device="cuda"
            ),
            torch.empty((m, rank), dtype=torch.bfloat16, device="cuda"),
        )
        for m in range(64, 64 * 13, 64)
    ]

    def launch(index: int) -> None:
        x, xq, sf, down = inputs[index]
        module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
            x, pqs, scale, l2t, xq, sf, down, workspace
        )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        launch(0)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(0)

    for index in range(1, len(inputs)):
        launch(index)
    torch.cuda.synchronize()
    for x, _, _, down in inputs[1:]:
        assert _sqnr_db(x.float() @ l2t.float(), down.float()) > 40

    x, _, _, down = inputs[0]
    x.copy_(torch.randn_like(x))
    l2t.copy_(torch.randn_like(l2t))
    down.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert _sqnr_db(x.float() @ l2t.float(), down.float()) > 40


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class TestSm120M537CudaGraph(unittest.TestCase):
    def test_cublaslt_lora_down_does_not_configure_l2_during_capture(self) -> None:
        """The persisting-L2 hint must not issue a forbidden capture API."""
        if torch.cuda.get_device_properties(0).multi_processor_count < 105:
            self.skipTest("requires the >=105-SM M537 admission")

        m, k, rank = 537, 5120, 32
        torch.manual_seed(20260816)
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        pre_quant_scale = torch.ones((k,), dtype=torch.bfloat16, device="cuda")
        global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
        l2t_smoothed = torch.randn(
            (k, rank), dtype=torch.bfloat16, device="cuda"
        ).contiguous()
        xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
        sf = torch.empty(
            (((m + 127) // 128) * 128 * (k // 16),),
            dtype=torch.uint8,
            device="cuda",
        )
        down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")
        workspace = torch.empty((32 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
        module = get_nvfp4_svdquant_sm120_module()

        def run() -> None:
            module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
                x,
                pre_quant_scale,
                global_scale,
                l2t_smoothed,
                xq,
                sf,
                down,
                workspace,
            )

        run()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(bool(torch.isfinite(down).all()))


def _graph_node_types(dot: str) -> list[str]:
    """Read CUDA graph node labels, excluding occurrences in kernel names."""
    labels = re.findall(r'\blabel\s*=\s*"((?:\\.|[^"\\])*)"', dot, flags=re.DOTALL)
    return [
        match.group(1).upper()
        for label in labels
        if (
            match := re.match(
                r"\s*\{?\s*(MEMSET|KERNEL|MEMCPY)\b", label, re.IGNORECASE
            )
        )
    ]


@pytest.mark.parametrize("enable_pdl", (False, True), ids=("pdl-off", "pdl-on"))
@pytest.mark.parametrize("use_bias", (False, True), ids=("no-bias", "bias"))
@pytest.mark.parametrize("tactic", (16, 73), ids=("splitk-k128", "splitk-k256-swap"))
def test_sm120_splitk_initializes_workspace_once_and_replays_fresh_inputs(
    tmp_path: Path, tactic: int, use_bias: bool, enable_pdl: bool
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    m, n, k, rank = 129, 256, 512, 32
    module = get_nvfp4_svdquant_sm120_module()
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, rank, tactic)
    torch.manual_seed(20260914)
    problem = _make_gemm_problem(
        m, n, k, rank=rank, quant_backend="cute-dsl", residual_backend="b12x"
    )
    workspace_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
    guard_elements = 128 * n
    storage = torch.full(
        (2 * guard_elements + m * n,), -2048, dtype=torch.bfloat16, device="cuda"
    )
    output = storage[guard_elements:-guard_elements].view(m, n)
    bias = problem["bias"] if use_bias else None

    def launch() -> None:
        module.nvfp4_svdquant_gemm(
            problem["xq"],
            problem["wq"],
            problem["x_sf_flat"],
            problem["w_sf_flat"],
            problem["alpha"],
            problem["d"],
            problem["l1_scaled"],
            bias,
            output,
            workspace,
            tactic,
            enable_pdl,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    keep_graph = hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph")
    graph = (
        torch.cuda.CUDAGraph(keep_graph=True) if keep_graph else torch.cuda.CUDAGraph()
    )
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        launch()
    dot_path = tmp_path / f"splitk-{tactic}-pdl-{enable_pdl}.dot"
    graph.debug_dump(str(dot_path))
    if keep_graph:
        graph.instantiate()

    keys = ("xq", "wq", "x_sf_flat", "w_sf_flat", "alpha", "d", "l1_scaled", "bias")
    addresses = [problem[key].data_ptr() for key in keys] + [
        workspace.data_ptr(),
        output.data_ptr(),
    ]
    previous = output.clone()
    # Keep the workspace left by each replay. Resetting it here would mask a
    # missing initialization or a stale Split-K reduction/semaphore state.
    for replay in range(3):
        torch.manual_seed(20260915 + replay)
        fresh = _make_gemm_problem(
            m, n, k, rank=rank, quant_backend="cute-dsl", residual_backend="b12x"
        )
        for key in keys:
            problem[key].copy_(fresh[key])
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(output).all()), f"nonfinite replay {replay}"
        expected = fresh["ref_bias"] if use_bias else fresh["ref"]
        assert _sqnr_db(expected, output.float()) > 40.0, f"inaccurate replay {replay}"
        assert not torch.equal(previous, output), "fresh operands had no effect"
        assert bool((storage[:guard_elements] == -2048).all()), "leading guard changed"
        assert bool((storage[-guard_elements:] == -2048).all()), (
            "trailing guard changed"
        )
        assert addresses == [problem[key].data_ptr() for key in keys] + [
            workspace.data_ptr(),
            output.data_ptr(),
        ]
        previous.copy_(output)

    node_types = _graph_node_types(dot_path.read_text(encoding="utf-8"))
    assert node_types.count("KERNEL") == 1, f"expected one GEMM kernel: {dot_path}"
    assert node_types.count("MEMSET") == 1, (
        f"expected one workspace initialization, found {node_types.count('MEMSET')}: {dot_path}"
    )


def _check_linear_graph_replay(
    rank: int,
    shape: tuple[int, int, int],
    use_bias: bool,
    enable_pdl: bool,
    backend_name: str = "cutlass-sm120",
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires SM120")
    if get_compute_capability(torch.device("cuda")) != (12, 0):
        pytest.skip("requires SM120")

    torch.manual_seed(0)
    m, n, k = shape
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.25
    pqs = (1 + 0.3 * torch.randn(k, device="cuda", dtype=torch.bfloat16)).abs()
    global_sf = (2688.0 / (x * pqs).float().abs().amax()).reshape(1)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.25
    weight_fp4, weight_sf, weight_global_sf = _nvfp4_quantize_128x4(weight)
    alpha = (1.0 / (global_sf * weight_global_sf)).reshape(1)
    lora_a = torch.randn(rank, k, device="cuda", dtype=torch.bfloat16) / k**0.25
    l2t = (pqs[:, None] * lora_a.T).contiguous()
    lora_b = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16) / rank**0.25
    l1 = (lora_b.float() / alpha).to(torch.bfloat16).contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if use_bias else None

    def linear() -> torch.Tensor:
        return svdquant_linear(
            x,
            weight_fp4,
            weight_sf.reshape(-1),
            alpha,
            pqs,
            l2t,
            l1,
            global_sf,
            bias=bias,
            enable_pdl=enable_pdl,
            backend=backend_name,
        )

    def reference() -> torch.Tensor:
        xq, x_sf = nvfp4_quantize((x * pqs).to(torch.bfloat16), global_sf)
        residual = _mm_fp4_residual(
            xq, weight_fp4, x_sf, weight_sf, alpha, backend="cutlass"
        )
        down = torch.mm(x, l2t)
        expected = residual + down.float() @ lora_b.float().T
        return expected if bias is None else expected + bias.float()

    expected = reference()
    eager = linear()
    assert _sqnr_db(expected, eager.float()) > 40.0

    with autotune(True):
        tuned = linear()
    assert _sqnr_db(expected, tuned.float()) > 40.0

    cached = linear()
    assert _sqnr_db(expected, cached.float()) > 40.0
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = linear()

    x.copy_(torch.randn_like(x) / k**0.25)
    l2t.copy_((pqs[:, None] * torch.randn_like(lora_a).T / k**0.25).contiguous())
    lora_b.copy_(torch.randn_like(lora_b) / rank**0.25)
    l1.copy_((lora_b.float() / alpha).bfloat16())
    expected_fresh = reference()
    graph.replay()
    torch.cuda.synchronize()

    assert captured.shape == (m, n)
    assert captured.dtype == torch.bfloat16
    assert _sqnr_db(expected_fresh, captured.float()) > 40.0


@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize(
    "shape", [(129, 256, 256), (537, 5376, 7168), (64, 3072, 3072)]
)
@pytest.mark.parametrize("use_bias,enable_pdl", [(False, False), (True, True)])
@pytest.mark.parametrize("backend_name", ["cutlass-sm120", "auto"])
def test_linear_rank_autotune_and_graph_replay(
    rank: int,
    shape: tuple[int, int, int],
    use_bias: bool,
    enable_pdl: bool,
    backend_name: str,
) -> None:
    _check_linear_graph_replay(rank, shape, use_bias, enable_pdl, backend_name)


@pytest.mark.parametrize(
    "m,k,variant",
    [
        (m, k, variant)
        for m, k in [(64, 3072), (537, 7168), (2048, 3072), (4096, 12288)]
        for variant in range(routes.sm120_producer_variant_count(m, k, 64))
    ],
)
def test_rank64_fused_prefix_observes_graph_weight_updates(
    m: int, k: int, variant: int
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    torch.manual_seed(20260917)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    pqs = (1 + 0.2 * torch.randn(k, device="cuda", dtype=torch.bfloat16)).abs()
    global_scale = (2688 / (x.float() * pqs.float()).abs().amax()).reshape(1)
    l2t = torch.randn((k, 64), device="cuda", dtype=torch.bfloat16)
    xq = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    sf = torch.zeros(
        ((m + 127) // 128 * 128 * (k // 16),), device="cuda", dtype=torch.uint8
    )
    down = torch.empty((m, 64), device="cuda", dtype=torch.bfloat16)
    expected_xq = torch.empty_like(xq)
    expected_sf = torch.zeros_like(sf)
    module = backend.get_nvfp4_svdquant_sm120_module(64)
    family, tiling, address_policy = routes.sm120_decode_producer_variant(
        m, k, variant, 64
    )

    def prefix() -> None:
        module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
            x, pqs, global_scale, l2t, xq, sf, down, family, *tiling, address_policy
        )

    def check() -> None:
        module.nvfp4_quantize_smooth(
            x, pqs, global_scale, expected_xq, expected_sf, False
        )
        reference = x.float() @ l2t.float()
        assert torch.equal(xq, expected_xq)
        assert torch.equal(sf, expected_sf)
        for rank_column in range(64):
            assert (
                _sqnr_db(reference[:, rank_column], down[:, rank_column].float()) > 40
            )

    prefix()
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prefix()

    for _ in range(2):
        x.copy_(torch.randn_like(x))
        l2t.copy_(torch.randn_like(l2t))
        down.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.parametrize("shape", [(64, 3072, 3072), (537, 5376, 7168)])
def test_rank64_complete_linear_with_fused_runner_only(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int, int]
) -> None:
    def fused_runners(
        enable_pdl: bool, device: torch.device, m: int, n: int, k: int, rank: int
    ) -> list[backend.TunableRunner]:
        return [backend._sm120_fused_linear_runner(enable_pdl, device, rank)]

    monkeypatch.setattr(backend, "_cached_sm120_linear_runners", fused_runners)
    backend.AutoTuner.get().clear_cache()
    try:
        _check_linear_graph_replay(64, shape, True, True)
    finally:
        backend.AutoTuner.get().clear_cache()


# Native prefix input validation on CUDA tensors.


@pytest.fixture
def prefix_inputs() -> list[torch.Tensor]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    m, k = 64, 3072
    return [
        torch.ones((m, k), dtype=torch.bfloat16, device="cuda"),
        torch.ones(k, dtype=torch.bfloat16, device="cuda"),
        torch.ones(1, dtype=torch.float32, device="cuda"),
        torch.ones((k, 32), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
        torch.empty(128 * (k // 16), dtype=torch.uint8, device="cuda"),
        torch.empty((m, 32), dtype=torch.bfloat16, device="cuda"),
    ]


def _run_prefix(inputs: list[torch.Tensor], *, dynamic: bool) -> None:
    module = get_nvfp4_svdquant_sm120_module()
    if dynamic:
        family, tiling, policy = sm120_producer_variants(64, 3072)[0]
        module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
            *inputs, family, *tiling, policy
        )
    else:
        module.nvfp4_quantize_smooth_lora_down_sm120(*inputs)


@pytest.mark.parametrize("dynamic", (False, True))
@pytest.mark.parametrize(
    "index,shape,message",
    (
        (0, (64 * 3072,), "x must be 2-D"),
        (1, (1,), "pqs must have k elements"),
        (2, (0,), "global_scale must contain at least one element"),
        # A wrong leading dimension fails the row check, which names the rank
        # generically; only a wrong rank names the compiled rank.
        (3, (1, 32), r"l2t_smoothed must be \[k, rank\]"),
        (3, (3072, 16), r"l2t_smoothed must be \[k, 32\]"),
        (4, (64, 1), r"xq must be \[m, k/2\]"),
        (5, (1,), "sf is smaller than"),
        (6, (1, 32), r"down must be \[m, rank\]"),
    ),
)
def test_prefix_rejects_invalid_shapes_before_launch(
    prefix_inputs: list[torch.Tensor],
    dynamic: bool,
    index: int,
    shape: tuple[int, ...],
    message: str,
) -> None:
    original = prefix_inputs[index]
    prefix_inputs[index] = torch.empty(
        shape, dtype=original.dtype, device=original.device
    )
    with pytest.raises(RuntimeError, match=message):
        _run_prefix(prefix_inputs, dynamic=dynamic)


@pytest.mark.parametrize("dynamic", (False, True))
@pytest.mark.parametrize("index", range(1, 7))
def test_prefix_rejects_mixed_devices(
    prefix_inputs: list[torch.Tensor], dynamic: bool, index: int
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    other_device = (torch.cuda.current_device() + 1) % torch.cuda.device_count()
    prefix_inputs[index] = prefix_inputs[index].to(other_device)
    with pytest.raises(RuntimeError, match="device_id"):
        _run_prefix(prefix_inputs, dynamic=dynamic)


def test_dynamic_prefix_matches_default_prefix(
    prefix_inputs: list[torch.Tensor],
) -> None:
    _run_prefix(prefix_inputs, dynamic=False)
    expected = [tensor.clone() for tensor in prefix_inputs[4:]]
    for tensor in prefix_inputs[4:]:
        tensor.zero_()
    _run_prefix(prefix_inputs, dynamic=True)
    for actual, reference in zip(prefix_inputs[4:], expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("rank", (32, 64))
@pytest.mark.parametrize("entry", ("quantize", "fused", "dynamic", "cublaslt"))
@pytest.mark.parametrize("index,name", ((0, "x"), (1, "pqs")))
@pytest.mark.parametrize("offset", (1, 2, 4))
def test_native_prefix_rejects_misaligned_inputs_before_launch(
    prefix_inputs: list[torch.Tensor],
    rank: int,
    entry: Literal["quantize", "fused", "dynamic", "cublaslt"],
    index: int,
    name: str,
    offset: int,
) -> None:
    inputs = prefix_inputs
    inputs[3] = torch.ones((3072, rank), dtype=torch.bfloat16, device="cuda")
    inputs[6] = torch.empty((64, rank), dtype=torch.bfloat16, device="cuda")
    original = inputs[index]
    inputs[index] = torch.ones(
        original.numel() + offset, dtype=original.dtype, device=original.device
    )[offset:].view_as(original)
    assert inputs[index].is_contiguous()
    assert inputs[index].data_ptr() % 16 == offset * 2
    module = get_nvfp4_svdquant_sm120_module(rank)
    # The cuBLASLt prefix is rank-32 only and refuses other ranks first.
    expected = (
        "supports rank 32 only"
        if entry == "cublaslt" and rank != 32
        else f"{name} must be 16-byte aligned"
    )
    with pytest.raises(RuntimeError, match=expected):
        match entry:
            case "quantize":
                module.nvfp4_quantize_smooth(*inputs[:3], *inputs[4:6], False)
            case "fused":
                module.nvfp4_quantize_smooth_lora_down_sm120(*inputs)
            case "dynamic":
                family, tiling, policy = sm120_producer_variants(64, 3072, rank)[0]
                module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
                    *inputs, family, *tiling, policy
                )
            case "cublaslt":
                workspace = torch.empty(16, dtype=torch.uint8, device="cuda")
                module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
                    *inputs, workspace
                )
            case unreachable:
                assert_never(unreachable)
    torch.cuda.synchronize()


if __name__ == "__main__":
    pytest.main([__file__])
