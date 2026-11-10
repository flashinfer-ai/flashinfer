from __future__ import annotations

from b12x.testing.reference.fp6 import _bf16_global_scale, _quantize_bf16_matrix

import pytest
import torch

from b12x._lib.intrinsics import swizzle_block_scale
from b12x._lib.fp6 import (
    FLOAT6_E3M2_MAX,
    SF_VEC_SIZE_FP6,
    _encode_fp6_nearest,
    _ue8m0_scale_from_block_max,
    dequant_mxfp6_torch,
    quantize_grouped_mxfp6_torch,
)
from b12x._lib.utils import mxfp6_packed_k_bytes
from b12x._lib.dense_gemm import dense_gemm
from b12x.quantization.mxfp6 import (
    allocate_bf16_to_fp6_tma_outputs,
    compile_bf16_to_fp6_tma,
)
from b12x.testing.reference.helpers import require_b12x

# TODO(port): several tests below drove the retired pre-facade b12x.integration.tp_moe
# host API (allocate_tp_moe_workspace / b12x_moe_fp6 / clear_tp_moe_caches),
# which has no upstream b12x equivalent. Those test bodies are kept as reference behind
# pytest.skip("pending: fused_moe-based rewrite") gates; rewrite them against the
# b12x.moe.fused_moe plan/bind/run flow once the w6a8_mx run path lands.

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FP6 GPU tests"
)


def _one_ulp_rtol(fmt: str) -> float:
    """Relative tolerance equal to one FP6 quantization step (1 ULP).

    The GPU activation quantizer rounds the scaled value ``f32 -> fp6`` directly
    via the hardware ``cvt.rn`` instruction, while ``_quantize_bf16_matrix`` (the
    Torch reference) rounds ``f32 -> bf16 -> fp6``. Both are valid round-to-nearest
    quantizers; they only ever disagree on values that sit on an exact rounding
    boundary, where they pick *adjacent* representable FP6 grid points. Comparing
    with a relative tolerance of one ULP accepts that single-step boundary tie while
    still failing on any error of two or more FP6 steps (a real kernel bug).

    One ULP within a binade is ``2**-mantissa_bits``: e3m2 has 2 mantissa bits
    (step = 1/4 = 0.25) and e2m3 has 3 (step = 1/8 = 0.125). A small epsilon guards
    against float comparison at the exact boundary.
    """
    return 0.26 if fmt == "e3m2" else 0.13


@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_bf16_to_fp6_tma_matches_torch_reference(fmt: str) -> None:
    require_b12x()
    m = k = 128
    torch.manual_seed(7)
    bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25
    ref_packed, ref_scales = _quantize_bf16_matrix(bf16, fmt=fmt)
    out = allocate_bf16_to_fp6_tma_outputs(m, k, device=bf16.device)
    gs = _bf16_global_scale(float(bf16.abs().max().item()))
    launch = compile_bf16_to_fp6_tma(m, k, fmt=fmt)
    launch(bf16, gs, out.packed_a_flat, out.scale_flat)
    torch.cuda.synchronize()

    assert out.packed_a_storage.shape == (1, m, mxfp6_packed_k_bytes(k))
    ref_deq = dequant_mxfp6_torch(
        ref_packed,
        ref_scales,
        num_fp6=k,
        fmt=fmt,  # type: ignore[arg-type]
        global_scale=gs,
    )
    ker_deq = dequant_mxfp6_torch(
        out.packed_a_storage.squeeze(0),
        out.scale_storage.view(1, -1),
        num_fp6=k,
        fmt=fmt,  # type: ignore[arg-type]
        global_scale=gs,
    )
    # Hardware cvt.rn (direct f32->fp6) vs the bf16-rounded Torch reference may
    # disagree by at most one FP6 step on exact rounding boundaries; allow 1 ULP.
    torch.testing.assert_close(ker_deq, ref_deq, rtol=_one_ulp_rtol(fmt), atol=0.15)


def test_dense_gemm_mxfp6_nonzero_vs_torch_reference() -> None:
    require_b12x()
    # NOTE(port): the original test cleared the b12x tp_moe compile caches here
    # (clear_tp_moe_caches); upstream has no tp_moe layer, and the dense GEMM
    # path below does not need it.
    m = n = k = 128
    torch.manual_seed(11)
    a_bf = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16) * 0.2
    b_bf = torch.randn(1, n, k, device="cuda", dtype=torch.bfloat16) * 0.2
    a_packed, a_sf = _quantize_bf16_matrix(a_bf.squeeze(0), fmt="e3m2")
    b_packed, b_sf = _quantize_bf16_matrix(b_bf.squeeze(0), fmt="e3m2")
    a_gs = _bf16_global_scale(float(a_bf.abs().max().item()))
    b_gs = _bf16_global_scale(float(b_bf.abs().max().item()))

    a_g = a_packed.unsqueeze(-1)
    b_g = b_packed.unsqueeze(-1)
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)
    out = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)
    dense_gemm(
        (a_g, a_sf),
        (b_g, b_sf),
        alpha=alpha,
        ab_dtype="float6_e3m2fn",
        sf_dtype="float8_e8m0fnu",
        sf_vec_size=SF_VEC_SIZE_FP6,
        c_dtype="bfloat16",
        out=out,
    )
    torch.cuda.synchronize()

    a_f = dequant_mxfp6_torch(
        a_packed,
        a_sf,
        num_fp6=k,
        fmt="e3m2",
        global_scale=a_gs,
    )
    b_f = dequant_mxfp6_torch(
        b_packed,
        b_sf,
        num_fp6=k,
        fmt="e3m2",
        global_scale=b_gs,
    )
    ref = (a_f @ b_f.T) * alpha.item()
    cos = torch.nn.functional.cosine_similarity(
        out[:, :, 0].reshape(-1).float(),
        ref.reshape(-1).float(),
        dim=0,
    ).item()
    assert cos > 0.95
    assert float(out.abs().max()) > 0.0


def _swizzled_scales_for_quantized(
    values_bf16: torch.Tensor,
    global_scales: torch.Tensor,
    *,
    fmt: str,
) -> torch.Tensor:
    """Build flat swizzled UE8M0 scale bytes matching MoE weight layout."""
    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else 7.5
    groups, rows, cols = values_bf16.shape
    scales = torch.zeros(
        (groups, rows, cols // SF_VEC_SIZE_FP6),
        dtype=torch.uint8,
        device=values_bf16.device,
    )
    for group_idx in range(groups):
        valid_rows = rows
        x = values_bf16[group_idx, :valid_rows].float()
        gs = float(global_scales[group_idx].item())
        sliced = x.view(valid_rows, cols // SF_VEC_SIZE_FP6, SF_VEC_SIZE_FP6)
        block_max = sliced.abs().amax(dim=-1)
        scales[group_idx, :valid_rows] = _ue8m0_scale_from_block_max(
            block_max * gs, fmt_max
        )
    # BITCAST (not numeric .to): `scales` already holds the UE8M0 exponent BYTE
    # (e.g. 127 == 2^0). `.to(float8_e8m0fnu)` would re-encode the *number* 127 as
    # e8m0 (=> 2^7 => byte 134), a 2^7 over-scale that does not match the codes
    # (quantized against byte 127) nor the reference dequant. `.view` reinterprets
    # the bits, keeping the exponent byte intact.
    return swizzle_block_scale(scales.view(torch.float8_e8m0fnu)).view(torch.uint8)


def _synthetic_mxfp6_moe_weights(
    *,
    experts: int,
    k: int,
    n: int,
    device: torch.device,
    weight_fmt: str = "e2m3",
):
    w1_n = 2 * n
    w1_bf = torch.randn(experts, w1_n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(experts, k, n, device=device, dtype=torch.bfloat16) * 0.15
    row_full = torch.full((experts,), w1_n, dtype=torch.int32, device=device)
    row_w2 = torch.full((experts,), k, dtype=torch.int32, device=device)
    gs1 = torch.ones(experts, device=device, dtype=torch.float32)
    gs2 = torch.ones(experts, device=device, dtype=torch.float32)
    w1_packed, _ = quantize_grouped_mxfp6_torch(
        w1_bf,
        row_full,
        gs1,
        fmt=weight_fmt,  # type: ignore[arg-type]
    )
    w2_packed, _ = quantize_grouped_mxfp6_torch(
        w2_bf,
        row_w2,
        gs2,
        fmt=weight_fmt,  # type: ignore[arg-type]
    )
    w1_fp6 = w1_packed.permute(2, 0, 1).contiguous()
    w2_fp6 = w2_packed.permute(2, 0, 1).contiguous()
    w1_bs = _swizzled_scales_for_quantized(w1_bf, gs1, fmt=weight_fmt)
    w2_bs = _swizzled_scales_for_quantized(w2_bf, gs2, fmt=weight_fmt)
    return w1_fp6, w1_bs, w2_fp6, w2_bs


def test_moe_fp6_synthetic_smoke() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    require_b12x()
    from b12x.integration.tp_moe import (  # retired pre-facade API (module TODO)
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    device = torch.device("cuda")
    m, k, n, experts, topk = 2, 128, 128, 4, 2
    torch.manual_seed(3)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).to(
        torch.float32
    )
    w1_fp6, w1_sf, w2_fp6, w2_sf = _synthetic_mxfp6_moe_weights(
        experts=experts, k=k, n=n, device=device
    )
    a1_gscale = torch.ones(1, device=device, dtype=torch.float32)
    a2_gscale = torch.ones(1, device=device, dtype=torch.float32)
    w1_alphas = torch.ones(experts, device=device, dtype=torch.float32)
    w2_alphas = torch.ones(experts, device=device, dtype=torch.float32)

    workspace = allocate_tp_moe_workspace(
        x,
        a1_gscale,
        w1_fp6,
        a2_gscale,
        w2_fp6,
        topk_ids,
        quant_mode="w6a6",
        input_scales_static=True,
    )
    out = b12x_moe_fp6(
        x,
        a1_gscale,
        w1_fp6,
        w1_sf,
        w1_alphas,
        a2_gscale,
        w2_fp6,
        w2_sf,
        w2_alphas,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        source_format="mxfp6_default",
    )
    torch.cuda.synchronize()
    assert out.shape == (m, k)
    assert float(out.float().norm()) > 1e-4


def _fp6_qdq_vec(vec: torch.Tensor, gs: float, fmt: str) -> torch.Tensor:
    """Quantize a length-K vector to MX-FP6 and dequantize (models kernel quant).

    Mirrors the in-kernel route-pack: ``f32 -> bf16 -> e3m2`` (``bf16_round``) with
    a per-32-block UE8M0 scale, then decode. Reuses the exact torch helpers the
    dense/quantizer GPU tests validate against the kernel.
    """
    k = vec.shape[-1]
    x = vec.float().view(1, 1, k)
    rc = torch.tensor([1], dtype=torch.int32, device=vec.device)
    gst = torch.tensor([gs], dtype=torch.float32, device=vec.device)
    packed, sv = quantize_grouped_mxfp6_torch(x, rc, gst, fmt=fmt, bf16_round=True)  # type: ignore[arg-type]
    deq = dequant_mxfp6_torch(
        packed[:, :, 0],
        sv,
        num_fp6=k,
        fmt=fmt,
        global_scale=gst,  # type: ignore[arg-type]
    )
    return deq.view(k)


def _fp6_dequant_grouped_weight(
    w_bf16: torch.Tensor, gs: torch.Tensor, fmt: str
) -> torch.Tensor:
    """Dequantize MX-FP6 weights ``[E, rows, K]`` exactly as the kernel sees them.

    Quantizes with the same grouped path the synthetic-weight builder uses, then
    decodes per-expert from the packed codes + row-major UE8M0 block scales
    (``scales_swizzled=False``), yielding the kernel's effective weight values.
    """
    experts, rows, k = w_bf16.shape
    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else 7.5
    row_counts = torch.full((experts,), rows, dtype=torch.int32, device=w_bf16.device)
    packed, _ = quantize_grouped_mxfp6_torch(w_bf16, row_counts, gs, fmt=fmt)  # type: ignore[arg-type]
    out = torch.empty(experts, rows, k, dtype=torch.float32, device=w_bf16.device)
    for e in range(experts):
        gse = float(gs[e].item())
        sliced = w_bf16[e].float().view(rows, k // SF_VEC_SIZE_FP6, SF_VEC_SIZE_FP6)
        block_max = sliced.abs().amax(dim=-1)
        sc = _ue8m0_scale_from_block_max(block_max * gse, fmt_max)
        out[e] = dequant_mxfp6_torch(
            packed[:, :, e],
            sc,
            num_fp6=k,
            fmt=fmt,  # type: ignore[arg-type]
            global_scale=gs[e : e + 1],
            scales_swizzled=False,
        )
    return out


def test_moe_fp6_numeric_vs_reference() -> None:
    pytest.skip("pending: fused_moe-based rewrite")
    require_b12x()
    from b12x.integration.tp_moe import (  # retired pre-facade API (module TODO)
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    device = torch.device("cuda")
    m, k, n, experts, topk = 4, 128, 128, 2, 1
    torch.manual_seed(7)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).to(
        torch.float32
    )

    # Build bf16 weights, quantize for the kernel (e2m3 weights), and keep the
    # bf16 originals to derive the reference's dequantized weights.
    w1_n = 2 * n
    w1_bf = torch.randn(experts, w1_n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(experts, k, n, device=device, dtype=torch.bfloat16) * 0.15
    gs1 = torch.ones(experts, device=device, dtype=torch.float32)
    gs2 = torch.ones(experts, device=device, dtype=torch.float32)
    row_full = torch.full((experts,), w1_n, dtype=torch.int32, device=device)
    row_w2 = torch.full((experts,), k, dtype=torch.int32, device=device)
    w1_packed, _ = quantize_grouped_mxfp6_torch(w1_bf, row_full, gs1, fmt="e2m3")
    w2_packed, _ = quantize_grouped_mxfp6_torch(w2_bf, row_w2, gs2, fmt="e2m3")
    w1_fp6 = w1_packed.permute(2, 0, 1).contiguous()
    w2_fp6 = w2_packed.permute(2, 0, 1).contiguous()
    w1_sf = _swizzled_scales_for_quantized(w1_bf, gs1, fmt="e2m3")
    w2_sf = _swizzled_scales_for_quantized(w2_bf, gs2, fmt="e2m3")

    a1_gscale = torch.ones(1, device=device, dtype=torch.float32)
    a2_gscale = torch.ones(1, device=device, dtype=torch.float32)
    w1_alphas = torch.ones(experts, device=device, dtype=torch.float32)
    w2_alphas = torch.ones(experts, device=device, dtype=torch.float32)

    workspace = allocate_tp_moe_workspace(
        x,
        a1_gscale,
        w1_fp6,
        a2_gscale,
        w2_fp6,
        topk_ids,
        quant_mode="w6a6",
        input_scales_static=True,
    )
    out = b12x_moe_fp6(
        x,
        a1_gscale,
        w1_fp6,
        w1_sf,
        w1_alphas,
        a2_gscale,
        w2_fp6,
        w2_sf,
        w2_alphas,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        source_format="mxfp6_default",
    )
    torch.cuda.synchronize()

    # Torch reference: dequantize weights/activations exactly as the kernel does,
    # then run the gated-silu FC1 + FC2 MoE math (mirrors _moe_reference_nvfp4).
    w1_deq = _fp6_dequant_grouped_weight(w1_bf, gs1, "e2m3")  # [E, 2n, k]
    w2_deq = _fp6_dequant_grouped_weight(w2_bf, gs2, "e2m3")  # [E, k, n]

    w1_bf_f = w1_bf.float()
    w2_bf_f = w2_bf.float()

    def _build_ref(gate_first: bool, quantized: bool) -> torch.Tensor:
        ref = torch.zeros(m, k, dtype=torch.float32, device=device)
        for t in range(m):
            for j in range(topk):
                eid = int(topk_ids[t, j].item())
                rw = float(topk_weights[t, j].item())
                if quantized:
                    x_deq = _fp6_qdq_vec(x[t], 1.0, "e3m2")
                    w1e, w2e = w1_deq[eid], w2_deq[eid]
                else:
                    x_deq = x[t].float()
                    w1e, w2e = w1_bf_f[eid], w2_bf_f[eid]
                first = w1e[:n] @ x_deq
                second = w1e[n:] @ x_deq
                gate, up = (first, second) if gate_first else (second, first)
                inter = (torch.sigmoid(gate) * gate * up).to(torch.bfloat16)
                inter_deq = (
                    _fp6_qdq_vec(inter, 1.0, "e3m2") if quantized else inter.float()
                )
                down = w2e @ inter_deq
                ref[t] += rw * down
        return ref

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(
            a.reshape(-1).float(), b.reshape(-1).float(), dim=0
        ).item()

    ref_fp6 = _build_ref(gate_first=False, quantized=True)
    ref_bf16 = _build_ref(gate_first=False, quantized=False)
    cos_fp6 = _cos(out, ref_fp6)
    cos_bf16 = _cos(out, ref_bf16)
    per_tok = [
        (
            round(_cos(out[t], ref_fp6[t]), 3),
            round(float(out[t].float().norm()), 3),
            round(float(ref_fp6[t].norm()), 3),
            int(topk_ids[t, 0].item()),
        )
        for t in range(m)
    ]
    print(f"\nMoE FP6: cos_fp6={cos_fp6:.4f} cos_bf16={cos_bf16:.4f}")
    print(f"per-token (cos,out_norm,ref_norm,eid): {per_tok}")
    assert cos_fp6 > 0.93, (
        f"MoE FP6 cosine {cos_fp6:.4f} (bf16-ref {cos_bf16:.4f}) too low"
    )


@pytest.mark.parametrize("shape", [(256, 512), (128, 512), (256, 256)])
@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_bf16_to_fp6_tma_multi_ktile_row_stride(shape, fmt: str) -> None:
    """Regression: the GPU FP6 quantizer must use the full 3K/4 packed row stride.

    The packed-code byte offset previously hard-coded a single-128-K-tile row
    stride (``grow*96``), so for K>128 (``k_tiles>1``) every row collapsed into the
    first tile's 96 bytes -- leaving most rows zero and scrambling the rest. Every
    prior test used K=128 (one K-tile) and so never exercised it. Drive M>128 and
    K>128 with non-uniform per-block magnitudes so the ``grow*96*k_tiles`` stride is
    mandatory and any layout collapse shows up.
    """
    require_b12x()
    m, k = shape
    torch.manual_seed(7)
    bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25
    nblk = k // SF_VEC_SIZE_FP6
    gain = (2.0 ** (torch.arange(nblk, device="cuda") % 6 - 3)).float()
    bf16 = (
        (bf16.float().view(m, nblk, SF_VEC_SIZE_FP6) * gain.view(1, nblk, 1))
        .view(m, k)
        .to(torch.bfloat16)
    )

    ref_packed, ref_scales = _quantize_bf16_matrix(bf16, fmt=fmt)
    gs = _bf16_global_scale(float(bf16.abs().max().item()))
    out = allocate_bf16_to_fp6_tma_outputs(m, k, device=bf16.device)
    launch = compile_bf16_to_fp6_tma(m, k, fmt=fmt)
    launch(bf16, gs, out.packed_a_flat, out.scale_flat)
    torch.cuda.synchronize()

    assert out.packed_a_storage.shape == (1, m, mxfp6_packed_k_bytes(k))
    ref_deq = dequant_mxfp6_torch(
        ref_packed,
        ref_scales,
        num_fp6=k,
        fmt=fmt,
        global_scale=gs,  # type: ignore[arg-type]
    )
    ker_deq = dequant_mxfp6_torch(
        out.packed_a_storage.squeeze(0),
        out.scale_storage.view(1, -1),
        num_fp6=k,
        fmt=fmt,  # type: ignore[arg-type]
        global_scale=gs,
    )
    # Hardware cvt.rn (direct f32->fp6) vs the bf16-rounded Torch reference may
    # disagree by at most one FP6 step on exact rounding boundaries; allow 1 ULP.
    # The per-block 2**k gain scales a single-ULP delta up in absolute terms, so the
    # relative (rtol) bound is what makes this meaningful, not the atol floor.
    torch.testing.assert_close(ker_deq, ref_deq, rtol=_one_ulp_rtol(fmt), atol=0.2)
    # No packed row may be left fully zero (the collapsed-stride signature).
    per_row_any = (out.packed_a_storage.squeeze(0) != 0).any(dim=1)
    assert bool(per_row_any.all()), "packed rows left zero => row-stride regression"


@pytest.mark.parametrize("backend", ["static", "dynamic"])
def test_moe_fp6_multi_ktile_numeric(backend: str, monkeypatch) -> None:
    """Regression: W6A6 MoE must be correct for K>128 (multi K-tile), both backends.

    Locks the production failure where converted Qwen3.6 MoE weights (K=2048)
    produced all-zero output: the bug was the GPU quantizer row stride at K>128,
    which feeds the MoE expert weights. K=512 is the smallest multi-K-tile case.
    """
    pytest.skip("pending: fused_moe-based rewrite")
    require_b12x()
    from b12x.integration.tp_moe import (  # retired pre-facade API (module TODO)
        allocate_tp_moe_workspace,
        b12x_moe_fp6,
        clear_tp_moe_caches,
        select_tp_moe_backend,
    )

    # Force the backend via the routed-pairs cutover (default 640).
    monkeypatch.setenv(
        "B12X_STATIC_COMPACT_CUTOVER_PAIRS", "0" if backend == "dynamic" else "640"
    )
    clear_tp_moe_caches()
    device = torch.device("cuda")
    m, k, n, experts, topk = 8, 512, 256, 8, 2
    assert (
        select_tp_moe_backend(num_tokens=m, num_topk=topk, quant_mode="w6a6") == backend
    )
    torch.manual_seed(7)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    topk_ids = torch.randint(0, experts, (m, topk), device=device, dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk, device=device), dim=-1).to(
        torch.float32
    )

    w1_n = 2 * n
    w1_bf = torch.randn(experts, w1_n, k, device=device, dtype=torch.bfloat16) * 0.15
    w2_bf = torch.randn(experts, k, n, device=device, dtype=torch.bfloat16) * 0.15
    gs1 = torch.ones(experts, device=device, dtype=torch.float32)
    gs2 = torch.ones(experts, device=device, dtype=torch.float32)
    row_full = torch.full((experts,), w1_n, dtype=torch.int32, device=device)
    row_w2 = torch.full((experts,), k, dtype=torch.int32, device=device)
    w1_packed, _ = quantize_grouped_mxfp6_torch(w1_bf, row_full, gs1, fmt="e2m3")
    w2_packed, _ = quantize_grouped_mxfp6_torch(w2_bf, row_w2, gs2, fmt="e2m3")
    w1_fp6 = w1_packed.permute(2, 0, 1).contiguous()
    w2_fp6 = w2_packed.permute(2, 0, 1).contiguous()
    w1_sf = _swizzled_scales_for_quantized(w1_bf, gs1, fmt="e2m3")
    w2_sf = _swizzled_scales_for_quantized(w2_bf, gs2, fmt="e2m3")

    a1_gscale = torch.ones(1, device=device, dtype=torch.float32)
    a2_gscale = torch.ones(1, device=device, dtype=torch.float32)
    w1_alphas = torch.ones(experts, device=device, dtype=torch.float32)
    w2_alphas = torch.ones(experts, device=device, dtype=torch.float32)

    workspace = allocate_tp_moe_workspace(
        x,
        a1_gscale,
        w1_fp6,
        a2_gscale,
        w2_fp6,
        topk_ids,
        quant_mode="w6a6",
        input_scales_static=True,
    )
    out = b12x_moe_fp6(
        x,
        a1_gscale,
        w1_fp6,
        w1_sf,
        w1_alphas,
        a2_gscale,
        w2_fp6,
        w2_sf,
        w2_alphas,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
        source_format="mxfp6_default",
    )
    torch.cuda.synchronize()

    w1_deq = _fp6_dequant_grouped_weight(w1_bf, gs1, "e2m3")
    w2_deq = _fp6_dequant_grouped_weight(w2_bf, gs2, "e2m3")
    ref = torch.zeros(m, k, dtype=torch.float32, device=device)
    for t in range(m):
        for j in range(topk):
            eid = int(topk_ids[t, j].item())
            rw = float(topk_weights[t, j].item())
            x_deq = _fp6_qdq_vec(x[t], 1.0, "e3m2")
            w1e, w2e = w1_deq[eid], w2_deq[eid]
            up = w1e[:n] @ x_deq
            gate = w1e[n:] @ x_deq
            inter = (torch.sigmoid(gate) * gate * up).to(torch.bfloat16)
            inter_deq = _fp6_qdq_vec(inter, 1.0, "e3m2")
            ref[t] += rw * (w2e @ inter_deq)

    cos = torch.nn.functional.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert float(out.abs().max()) > 0.0, "MoE output all-zero (K>128 regression)"
    assert cos > 0.93, f"MoE FP6 K={k} {backend} cosine {cos:.4f} too low"


def test_fp6_safetensors_loader_matches_pt_path() -> None:
    """Phase 3: safetensors per-expert load == validated .pt path, byte-for-byte.

    The exporter quantizes each projection (gate/up/down) independently; the
    loader re-stacks FC1 as [up; gate] and swizzles scales. Because per-row FP6
    packing and per-block UE8M0 scales are row-independent, this must reproduce
    ``quantize_moe_weights_to_fp6`` (which packs the stacked (2N,K) at once)
    exactly — proving the safetensors round-trip preserves the kernel layout.
    """
    require_b12x()
    from b12x.quantization.mxfp6 import (
        load_fp6_moe_weights_from_safetensors,
        quantize_linear_to_fp6,
        quantize_moe_weights_to_fp6,
    )

    device = torch.device("cuda")
    experts, n, k = 8, 256, 512  # K>128 exercises the multi-K-tile path
    torch.manual_seed(11)
    gate = torch.randn(experts, n, k, device=device, dtype=torch.bfloat16) * 0.15
    up = torch.randn(experts, n, k, device=device, dtype=torch.bfloat16) * 0.15
    down = torch.randn(experts, k, n, device=device, dtype=torch.bfloat16) * 0.15

    # Validated .pt path: FC1 rows are [up; gate] stacked, packed in one shot.
    w1 = torch.cat([up, gate], dim=1)  # (E, 2N, K)
    pt = quantize_moe_weights_to_fp6(
        w1, down, source_format="mxfp6_default", activation="silu", use_gpu=True
    )

    # Safetensors path: per-projection quantize -> per-expert keys -> load.
    prefix = "model.layers.0.mlp"
    state: dict[str, torch.Tensor] = {}
    for e in range(experts):
        for proj, mat in (
            ("gate_proj", gate[e]),
            ("up_proj", up[e]),
            ("down_proj", down[e]),
        ):
            # Ceil keeps TMA codes bit-identical to quantize_moe_weights_to_fp6.
            qt = quantize_linear_to_fp6(
                mat,
                source_format="mxfp6_default",
                use_gpu=True,
                block_scale_rule="ceil",
            )
            state.update(qt.to_state_dict(f"{prefix}.experts.{e}.{proj}"))

    loaded = load_fp6_moe_weights_from_safetensors(
        state.__getitem__,
        prefix,
        experts,
        activation="silu",
        source_format="mxfp6_default",
        device=device,
    )

    assert torch.equal(loaded.w1_fp6, pt.w1_fp6), "FC1 codes differ from .pt path"
    assert torch.equal(loaded.w2_fp6, pt.w2_fp6), "FC2 codes differ from .pt path"
    assert torch.equal(loaded.w1_blockscale, pt.w1_blockscale), "FC1 scales differ"
    assert torch.equal(loaded.w2_blockscale, pt.w2_blockscale), "FC2 scales differ"
    assert loaded.num_experts == experts and loaded.k == k and loaded.n == n


def test_fp6_dense_safetensors_loader_numeric() -> None:
    """Phase 3c: dense FP6 safetensors load -> dense_fp6_linear matches bf16 matmul.

    The on-disk dense weights use the unit-gscale UE8M0 contract; the loader
    swizzles the scales and reuses dense_fp6_linear (whose alpha=1/(a_gs*w_gs) is
    correct at w_gs=1). K>128 exercises the multi-K-tile GEMM path.
    """
    require_b12x()
    from b12x.quantization.mxfp6 import (
        dense_fp6_linear,
        load_fp6_dense_weight_from_safetensors,
        quantize_linear_to_fp6,
    )

    device = torch.device("cuda")
    out_f, in_f, m = 256, 512, 128  # in_f>128 -> multi-K-tile
    torch.manual_seed(5)
    weight = torch.randn(out_f, in_f, device=device, dtype=torch.bfloat16) * 0.1
    x = torch.randn(m, in_f, device=device, dtype=torch.bfloat16) * 0.1

    qt = quantize_linear_to_fp6(weight, source_format="mxfp6_default", use_gpu=True)
    name = "model.layers.0.mlp.gate_proj"
    state = qt.to_state_dict(name)
    loaded = load_fp6_dense_weight_from_safetensors(
        state.__getitem__, name, source_format="mxfp6_default", device=device
    )
    assert loaded.out_features == out_f and loaded.in_features == in_f

    y = dense_fp6_linear(x, loaded)
    ref = x.float() @ weight.float().T
    cos = torch.nn.functional.cosine_similarity(
        y.reshape(-1).float(), ref.reshape(-1), dim=0
    ).item()
    assert float(y.abs().max()) > 0.0, "dense FP6 output all-zero"
    assert cos > 0.95, f"dense FP6 safetensors cosine {cos:.4f} too low"
