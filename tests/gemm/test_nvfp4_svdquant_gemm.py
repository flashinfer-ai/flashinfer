"""Tests for the SM100 NVFP4 SVDQuant fused GEMM ops (Blackwell):

- mm_nvfp4_svdquant     : out = alpha * (a @ bT) + d @ l1T [+ bias], the block-scaled
                          NVFP4 residual GEMM fused with the rank-r BF16 LoRA-up
                          (r a positive multiple of 32; 32-128 covered here).
- nvfp4_quantize_smooth : NVFP4-quantize(x * pre_quant_scale), byte-identical to the
                          stock quantizer run on the pre-smoothed input.
- svdquant_linear       : the full quantize -> LoRA-down -> fused GEMM chain.

The unfused reference for the residual is flashinfer.mm_fp4 (cutlass backend) on the
same quantized operands, plus the LoRA correction computed in fp32.
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


def _skip_unless_sm100():
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip(
            "NVFP4 SVDQuant kernels require SM100-class GPUs, "
            f"got compute capability {compute_capability}."
        )


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    err = (ref - got).float()
    noise = (err**2).mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10((ref.float() ** 2).mean() / noise))


def _nvfp4_quantize_128x4(t: torch.Tensor):
    """Stock NVFP4 quantization (ue4m3 block scales, 128x4 swizzled layout).

    Returns (packed e2m1 uint8 [r, c/2], swizzled sf uint8 2-D, global scale f32 [1]).
    """
    global_sf = ((448.0 * 6.0) / t.float().abs().nan_to_num().max()).reshape(1)
    tq, sf = nvfp4_quantize(
        t, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    return tq.view(torch.uint8), sf.view(torch.uint8), global_sf


def _mm_fp4_residual(xq, wq, x_sf, w_sf, alpha):
    """Unfused reference residual alpha * (a @ bT) via the stock cutlass NVFP4 GEMM."""
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
        backend="cutlass",
        use_nvfp4=True,
    )
    return out.float()


def _make_gemm_problem(m, n, k, rank=_RANK, device="cuda"):
    """Quantized operands and fp32 references for out = alpha*(a@bT) + D@L1T [+ bias]."""
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    xq, x_sf, gx = _nvfp4_quantize_128x4(x)
    wq, w_sf, gw = _nvfp4_quantize_128x4(w)
    alpha = (1.0 / (gx * gw)).reshape(1).float()
    d = torch.randn(m, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    l1 = torch.randn(n, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    # 1/alpha is folded into L1 so the epilogue out = alpha*acc [+ bias] yields the
    # unscaled D @ L1T correction.
    l1_scaled = (l1.float() / alpha).to(torch.bfloat16).contiguous()
    bias = torch.randn(n, dtype=torch.bfloat16, device=device).contiguous()

    ref = _mm_fp4_residual(xq, wq, x_sf, w_sf, alpha) + d.float() @ l1.float().t()
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


if __name__ == "__main__":
    pytest.main([__file__])
