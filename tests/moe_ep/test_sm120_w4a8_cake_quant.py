"""Single-GPU test: the fused Triton MXFP8 input quantizer / stager of the Cake SM120 W4A8 MegaMoE
is bit-identical to the eager torch reference ``quantize_activations_k32`` (codes and UE8M0 scales).

Includes exhaustive BF16 coverage: every BF16 bit pattern quantized under every reachable block
exponent, and every BF16 bit pattern as the block amax.
"""

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
    cake_staging,
)
from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_mxfp8_quant import (
    mxfp8_quantize_fused,
)
from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_staging import (
    quantize_activations_k32,
    stage_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)

DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _check(x: torch.Tensor) -> None:
    c0, s0 = quantize_activations_k32(x)
    c1, s1 = mxfp8_quantize_fused(x)
    assert c0.shape == c1.shape and s0.shape == s1.shape
    bad_c = int((c0 != c1).sum())
    bad_s = int((s0 != s1).sum())
    assert bad_c == 0 and bad_s == 0, f"{bad_c} code / {bad_s} scale mismatches"


def _make(kind: str, rows: int, hidden: int, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(rows * 131 + hidden + len(kind))
    if kind == "randn":
        x = torch.randn(rows, hidden, device="cuda", generator=g)
    elif (
        kind == "wide"
    ):  # per-block magnitudes over the whole exponent range of the dtype
        lo, hi = {torch.bfloat16: (-130, 125), torch.float16: (-24, 14)}.get(
            dtype, (-140, 125)
        )
        e = torch.randint(lo, hi, (rows, hidden // 32, 1), device="cuda", generator=g)
        x = (
            torch.randn(rows, hidden // 32, 32, device="cuda", generator=g)
            * torch.exp2(e.float())
        ).reshape(rows, hidden)
    elif kind == "denorm":  # fp32 / bf16 denormals and their neighbours
        x = torch.randn(rows, hidden, device="cuda", generator=g) * 1e-39
    elif (
        kind == "special"
    ):  # zeros, +-0, inf, nan, 448 neighbourhood, sprinkled into random data
        x = torch.randn(rows, hidden, device="cuda", generator=g) * 100
        vals = torch.tensor(
            [
                0.0,
                -0.0,
                float("inf"),
                float("-inf"),
                float("nan"),
                448.0,
                464.0,
                480.0,
                -448.0,
                1e-45,
                -1e-45,
                65504.0,
                3.0e38,
            ],
            device="cuda",
        )
        pick = torch.randint(0, 40, (rows, hidden), device="cuda", generator=g)
        x = torch.where(pick < len(vals), vals[pick.clamp(max=len(vals) - 1)], x)
        x[:, :32] = 0.0  # one all-zero block per row
    else:  # zeros
        x = torch.zeros(rows, hidden, device="cuda")
    return x.to(dtype)


@pytest.mark.parametrize("rows", [0, 1, 7, 1024, 2048, 4096])
@pytest.mark.parametrize("hidden", [4096])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["randn", "wide", "denorm", "special", "zeros"])
def test_fused_matches_reference_v4flash(rows, hidden, dtype, kind) -> None:
    _check(_make(kind, rows, hidden, dtype))


@pytest.mark.parametrize(
    "rows,hidden", [(3, 32), (5, 544), (33, 7168), (9, 2048), (17, 5120)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["randn", "wide", "special"])
def test_fused_matches_reference_other_shapes(rows, hidden, dtype, kind) -> None:
    _check(_make(kind, rows, hidden, dtype))


def _all_bf16() -> torch.Tensor:
    return (
        torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )


def test_exhaustive_bf16_codes_under_every_exponent() -> None:
    """Every BF16 bit pattern quantized in a block led by each of 256 amax values that reach every exponent."""

    allx = (
        _all_bf16()
    )  # 65536 patterns (incl. +-0, denormals, +-inf, NaNs of both signs)
    per = 31
    nblk = (allx.numel() + per - 1) // per
    pad = torch.zeros(nblk * per - allx.numel(), dtype=torch.bfloat16, device="cuda")
    xs = torch.cat([allx, pad]).view(nblk, per)
    # leaders: the largest positive finite bf16 in each binade (all 254 exponents via amax*fl(1/448)),
    # plus +0, the smallest denormals, inf and nan
    leaders = [0.0, 2.0**-133, 2.0**-126]
    leaders += [
        float(torch.tensor(2.0**k * 1.9921875, dtype=torch.bfloat16))
        for k in range(-133, 128)
    ]
    leaders += [float("inf"), float("nan")]
    lead = torch.tensor(leaders, dtype=torch.float32, device="cuda").to(torch.bfloat16)
    x = torch.cat(
        [
            lead.view(-1, 1, 1).expand(-1, nblk, 1),
            xs.unsqueeze(0).expand(lead.numel(), -1, -1),
        ],
        dim=2,
    ).reshape(lead.numel(), nblk * 32)
    _check(x.contiguous())
    _check(x.float().contiguous())


def test_exhaustive_bf16_amax() -> None:
    """Every BF16 bit pattern as a block's leading (largest-magnitude) value: all scale exponents."""

    a = _all_bf16().view(-1, 1)
    g = torch.Generator(device="cuda").manual_seed(7)
    f = torch.rand(a.shape[0], 31, device="cuda", generator=g) * 2 - 1
    x = torch.cat([a, (a.float() * f).to(torch.bfloat16)], dim=1).reshape(256, 256 * 32)
    _check(x.contiguous())
    _check(x.float().contiguous())


def test_fp32_scale_boundaries() -> None:
    """FP32 amax values one ulp around every power-of-two scale boundary (amax * fl(1/448) == 2^k)."""

    k = torch.arange(-149, 128, device="cuda", dtype=torch.float64)
    base = (torch.exp2(k) * 448.0).float()
    bits = base.view(torch.int32)
    cand = torch.stack([bits - 2, bits - 1, bits, bits + 1, bits + 2], 1).reshape(-1)
    cand = cand[(cand > 0) & (cand < 0x7F800000)].view(torch.float32)
    g = torch.Generator(device="cuda").manual_seed(11)
    f = torch.rand(cand.numel(), 31, device="cuda", generator=g) * 2 - 1
    x = torch.cat([cand.view(-1, 1), cand.view(-1, 1) * f], 1)
    _check(x.contiguous())  # [n, 32]: one block per row


def test_noncontiguous_rows_and_into_buffers() -> None:
    big = _make("randn", 40, 8192, torch.bfloat16)
    x = big[:33, 1024:5120]  # row stride 8192, unit inner stride
    codes = torch.full((64, 4096), 0xAB, dtype=torch.uint8, device="cuda")
    sf = torch.full((64, 128), 0xCD, dtype=torch.uint8, device="cuda")
    c, s = mxfp8_quantize_fused(x, codes=codes, sf=sf)
    c0, s0 = quantize_activations_k32(x)
    assert torch.equal(c, c0) and torch.equal(s, s0)
    assert bool((codes[33:] == 0xAB).all()) and bool((sf[33:] == 0xCD).all())


def _stage(fused: bool, monkeypatch, x, ids, w, cap=64):
    monkeypatch.setattr(cake_staging, "_FUSED_QUANT", fused)
    hidden, tk = x.shape[1], ids.shape[1]
    bufs = dict(
        x_fp8=torch.full((cap, hidden), 0x5A, dtype=torch.uint8, device="cuda"),
        x_sf=torch.full((cap, hidden // 32), 0x3C, dtype=torch.uint8, device="cuda"),
        topk_idx_out=torch.full((cap, tk), -1, dtype=torch.int64, device="cuda"),
        topk_weights_out=torch.full((cap, tk), 0.5, dtype=torch.float32, device="cuda"),
    )
    n = stage_inputs(x, ids, w, **bufs, quantize_input=True, scales=None)
    return n, bufs


@pytest.mark.parametrize("rows", [0, 1, 7, 64])
@pytest.mark.parametrize("routing_layout", ["contiguous", "strided", "transposed"])
def test_stage_inputs_fused_equals_eager(monkeypatch, rows, routing_layout) -> None:
    g = torch.Generator(device="cuda").manual_seed(rows + 3)
    x = _make("special", rows, 4096, torch.bfloat16)
    ids = torch.randint(-1, 256, (rows, 12), device="cuda", generator=g)
    w = torch.rand(rows, 12, device="cuda", generator=g)
    if routing_layout == "strided":
        ids, w = ids[:, ::2], w[:, ::2]
    elif routing_layout == "transposed":
        ids, w = ids[:, :6].t().contiguous().t(), w[:, :6].t().contiguous().t()
    else:
        ids, w = ids[:, :6].contiguous(), w[:, :6].contiguous()
    n0, b0 = _stage(False, monkeypatch, x, ids, w)
    n1, b1 = _stage(True, monkeypatch, x, ids, w)
    assert n0 == n1 == rows
    for k in b0:
        assert torch.equal(b0[k], b1[k]), k


def test_stage_inputs_cuda_graph(monkeypatch) -> None:
    monkeypatch.setattr(cake_staging, "_FUSED_QUANT", True)
    x = _make("randn", 128, 4096, torch.bfloat16)
    ids = torch.randint(0, 256, (128, 6), device="cuda")
    w = torch.rand(128, 6, device="cuda")
    bufs = dict(
        x_fp8=torch.zeros((256, 4096), dtype=torch.uint8, device="cuda"),
        x_sf=torch.zeros((256, 128), dtype=torch.uint8, device="cuda"),
        topk_idx_out=torch.full((256, 6), -1, dtype=torch.int64, device="cuda"),
        topk_weights_out=torch.zeros((256, 6), dtype=torch.float32, device="cuda"),
    )
    stage_inputs(
        x, ids, w, **bufs, quantize_input=True, scales=None
    )  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage_inputs(x, ids, w, **bufs, quantize_input=True, scales=None)
    x.copy_(_make("wide", 128, 4096, torch.bfloat16))
    ids.copy_(torch.randint(0, 256, (128, 6), device="cuda"))
    for b in bufs.values():
        b.zero_()
    graph.replay()
    torch.cuda.synchronize()
    c0, s0 = quantize_activations_k32(x)
    assert torch.equal(bufs["x_fp8"][:128], c0) and torch.equal(bufs["x_sf"][:128], s0)
    assert torch.equal(bufs["topk_idx_out"][:128], ids)
    assert torch.equal(bufs["topk_weights_out"][:128], w)
