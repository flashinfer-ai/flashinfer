"""Tests for the CuTe-DSL MXFP8 x MXFP8 (E4M3 + E8M0 block-32 scales) MoE path.

Two oracles, as in the other FlashInfer MXFP8 tests:

* tier 1 -- cosine similarity against the unquantized BF16 MoE (catches layout,
  routing and activation bugs; tolerance covers quantization noise);
* tier 2 -- a PyTorch reference that applies the same E8M0 quantize/dequantize
  to the weights, the input and the GEMM1->GEMM2 intermediate (tight tolerance;
  catches scale-layout and ordering errors that cosine similarity hides).

The MoE is a non-gated ReLU^2 MoE: out = sum_k w_k * (relu(x @ W1[e].T)^2 @ W2[e].T).
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported
from flashinfer.tllm_enums import ActivationType

_requires_dsl_arch = pytest.mark.skipif(
    torch.cuda.is_available()
    and not is_cute_dsl_arch_supported(
        *torch.cuda.get_device_capability(0), native_only=True
    ),
    reason="installed CuTe DSL does not support this GPU architecture",
)

mxfp8_required = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0) in ((10, 0), (10, 3))
    ),
    reason="MXFP8 CuTe-DSL MoE requires SM100 or SM103",
)

pytestmark = [_requires_dsl_arch, mxfp8_required]

MXFP8_BLOCK = 32
FP8_MAX = 448.0
MIN_COS_SIM_VS_BF16 = 0.98


def _qdq_mxfp8(t: torch.Tensor) -> torch.Tensor:
    """Quantize + dequantize along the last dim with E8M0 scales of block 32."""
    shape = t.shape
    blocks = t.float().reshape(-1, shape[-1] // MXFP8_BLOCK, MXFP8_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exp = torch.ceil(torch.log2(torch.clamp(amax, min=1e-30) / FP8_MAX))
    exp = torch.clamp(exp, -127, 127)
    scale = torch.exp2(exp)
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).reshape(shape)


def _quantize_weights(w: torch.Tensor):
    """[E, rows, cols] BF16 -> (fp8 [E, rows, cols], MMA-layout E8M0 scales)."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp8_quantization import mxfp8_quantize

    num_experts, rows, cols = w.shape
    if rows % 128 == 0:
        q, sf = mxfp8_quantize(w.reshape(num_experts * rows, cols), True)
    else:
        # Scales are padded to 128 rows per expert, so quantize expert by expert.
        parts = [mxfp8_quantize(w[e], True) for e in range(num_experts)]
        q = torch.cat([p[0] for p in parts])
        sf = torch.cat([p[1].reshape(-1) for p in parts])
    sf_mma = convert_sf_to_mma_layout(
        sf, m=rows, k=cols, num_groups=num_experts, sf_vec_size=MXFP8_BLOCK
    )
    return q.view(num_experts, rows, cols), sf_mma


def _reference(x, w1, w2, topk_ids, topk_w, quantized: bool):
    """Non-gated ReLU^2 MoE in FP32. ``quantized`` applies MXFP8 qdq everywhere."""
    qdq = _qdq_mxfp8 if quantized else (lambda t: t.float())
    xf, w1f, w2f = qdq(x), qdq(w1), qdq(w2)
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    for e in range(w1.shape[0]):
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        z = xf[tok] @ w1f[e].T
        h = torch.relu(z) ** 2
        y = qdq(h) @ w2f[e].T
        out.index_add_(0, tok, y * topk_w[tok, slot].unsqueeze(1))
    return out


def _run_kernel(x, w1, w2, topk_ids, topk_w, num_experts, tactic=None):
    from flashinfer import cute_dsl_fused_moe
    from flashinfer.fp8_quantization import mxfp8_quantize

    num_tokens, hidden = x.shape
    w1_q, w1_sf = _quantize_weights(w1)
    w2_q, w2_sf = _quantize_weights(w2)
    x_q, x_sf = mxfp8_quantize(x, is_sf_swizzled_layout=False, alignment=128)
    x_sf = x_sf.view(torch.uint8).reshape(num_tokens, hidden // MXFP8_BLOCK)
    ones = torch.ones(num_experts, dtype=torch.float32, device=x.device)
    return cute_dsl_fused_moe(
        x=x_q,
        x_sf=x_sf,
        token_selected_experts=topk_ids,
        token_final_scales=topk_w,
        w1_weight=w1_q,
        w1_weight_sf=w1_sf,
        w1_alpha=ones,
        fc2_input_scale=None,
        w2_weight=w2_q,
        w2_weight_sf=w2_sf,
        w2_alpha=ones,
        num_experts=num_experts,
        top_k=topk_ids.shape[1],
        num_local_experts=num_experts,
        output_dtype=torch.bfloat16,
        activation_type=ActivationType.Relu2.value,
        quant_mode="mxfp8",
        tactic=tactic,
    )


def _gemm1_n192_tactic(tile):
    return (
        tile,
        ((tile, 192), (tile // 128, 1), False),
        ((tile, 128), (tile // 128, 1), False),
    )


def _make_case(num_tokens, hidden, inter, num_experts, top_k, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev) / 2
    w1 = torch.randn(num_experts, inter, hidden, dtype=torch.bfloat16, device=dev) / 8
    w2 = torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16, device=dev) / 8
    scores = torch.rand(num_tokens, num_experts, device=dev)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_w = torch.softmax(torch.randn(num_tokens, top_k, device=dev), dim=-1)
    return x, w1, w2, topk_ids, topk_w.float()


def _cos(a, b):
    return F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0)


@pytest.mark.parametrize(
    "num_tokens,hidden,inter,num_experts,top_k",
    [
        (33, 896, 1024, 16, 4),
        # Many-expert latent-MoE shape
        (64, 640, 3584, 384, 8),
    ],
)
def test_mxfp8_relu2_moe(num_tokens, hidden, inter, num_experts, top_k):
    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k
    )
    out = _run_kernel(x, w1, w2, topk_ids, topk_w, num_experts).float()

    ref_bf16 = _reference(x, w1, w2, topk_ids, topk_w, quantized=False)
    cos = _cos(out, ref_bf16)
    assert cos > MIN_COS_SIM_VS_BF16, f"cos vs BF16 {cos:.4f}"

    ref_q = _reference(x, w1, w2, topk_ids, topk_w, quantized=True)
    cos_q = _cos(out, ref_q)
    assert cos_q > 0.999, f"cos vs MXFP8 reference {cos_q:.5f}"
    torch.testing.assert_close(out, ref_q, rtol=5e-2, atol=5e-2 * ref_q.abs().max())


@pytest.mark.parametrize("tile", [128, 256])
def test_mxfp8_relu2_gemm1_n192(tile):
    """Non-gated GEMM1 may use N=192 tiles; check them against the MXFP8 reference."""
    # intermediate = 14 x 192, not a multiple of 256
    num_tokens, hidden, inter, num_experts, top_k = 256, 1024, 2688, 64, 8
    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k
    )
    out = _run_kernel(
        x, w1, w2, topk_ids, topk_w, num_experts, tactic=_gemm1_n192_tactic(tile)
    ).float()
    ref_q = _reference(x, w1, w2, topk_ids, topk_w, quantized=True)
    cos_q = _cos(out, ref_q)
    assert cos_q > 0.999, f"cos vs MXFP8 reference {cos_q:.5f}"
    torch.testing.assert_close(out, ref_q, rtol=5e-2, atol=5e-2 * ref_q.abs().max())


def test_mxfp8_relu2_gemm1_n192_rejects_partial_n_tile():
    """GEMM1 has no partial-N-tile predicate, so N=192 needs inter % 192 == 0."""
    x, w1, w2, topk_ids, topk_w = _make_case(33, 512, 512, 8, 2)
    with pytest.raises(ValueError, match="Unsupported configuration"):
        _run_kernel(x, w1, w2, topk_ids, topk_w, 8, tactic=_gemm1_n192_tactic(128))


@pytest.mark.parametrize("tile", [128, 256])
def test_w4a8_relu2_gemm1_n192_matches_default(tile):
    """W4A8 (MXFP4 weights) with GEMM1 N=192 matches the default tactic."""
    from flashinfer import cute_dsl_fused_moe
    from flashinfer.fused_moe import CuteDslConfig, QuantConfig, QuantFormat, ReLU2

    num_tokens, hidden, inter, num_experts, top_k = 64, 512, 384, 8, 2
    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k
    )
    quant = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
    view = CuteDslConfig.prepare_weights(
        w1,
        w2,
        quant=quant,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=inter,
        activation=ReLU2(),
    )
    x_q, x_sf = CuteDslConfig.prepare_activations(
        x, quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
    )

    def run(tactic):
        return cute_dsl_fused_moe(
            x=x_q,
            x_sf=x_sf,
            token_selected_experts=topk_ids,
            token_final_scales=topk_w,
            w1_weight=view["w1_weight"],
            w1_weight_sf=view["w1_weight_sf"],
            w1_alpha=view["w1_alpha"],
            fc2_input_scale=None,
            w2_weight=view["w2_weight"],
            w2_weight_sf=view["w2_weight_sf"],
            w2_alpha=view["w2_alpha"],
            num_experts=num_experts,
            top_k=top_k,
            activation_type=ActivationType.Relu2.value,
            quant_mode="w4a8",
            tactic=tactic,
        ).float()

    default = run(-1)
    out = run(_gemm1_n192_tactic(tile))
    assert _cos(out, default) > 0.999
    torch.testing.assert_close(out, default, rtol=2e-2, atol=2e-2 * default.abs().max())


def test_gemm1_n192_is_non_gated_only():
    from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
        BlockScaledContiguousGatherGroupedGemmKernel as Kernel,
    )

    for tile in (128, 256):
        cluster = (tile // 128, 1)
        assert Kernel.is_valid_mma_tiler_and_cluster_shape(
            (tile, 192), cluster, gated=False
        )
        assert not Kernel.is_valid_mma_tiler_and_cluster_shape(
            (tile, 192), cluster, gated=True
        )
        assert Kernel.is_valid_mma_tiler_and_cluster_shape(
            (tile, 128), cluster, gated=True
        )


def test_mxfp8_mode_rejects_invalid_inputs():
    from flashinfer import cute_dsl_fused_moe

    x, w1, w2, topk_ids, topk_w = _make_case(8, 256, 256, 4, 2)
    w1_q, w1_sf = _quantize_weights(w1)
    w2_q, w2_sf = _quantize_weights(w2)
    ones = torch.ones(4, dtype=torch.float32, device="cuda")
    kwargs = dict(
        token_selected_experts=topk_ids,
        token_final_scales=topk_w,
        w1_weight=w1_q,
        w1_weight_sf=w1_sf,
        w1_alpha=ones,
        fc2_input_scale=None,
        w2_weight=w2_q,
        w2_weight_sf=w2_sf,
        w2_alpha=ones,
        num_experts=4,
        top_k=2,
        activation_type=ActivationType.Relu2.value,
        quant_mode="mxfp8",
    )
    # activations must be E4M3 (the BF16 input is not pre-quantized)
    with pytest.raises(TypeError, match="float8_e4m3fn input"):
        cute_dsl_fused_moe(x=x, x_sf=None, **kwargs)
    x_q = x.to(torch.float8_e4m3fn)
    x_sf = torch.zeros(8, 256 // MXFP8_BLOCK, dtype=torch.uint8, device="cuda")
    # MXFP8 mode needs E4M3 weights, not BF16
    with pytest.raises(TypeError, match="float8_e4m3fn weights"):
        cute_dsl_fused_moe(x=x_q, x_sf=x_sf, **{**kwargs, "w1_weight": w1})


def test_row_major_scales_match_swizzled_quantizer_layout():
    """Checkpoint (row-major) scales convert to the layout mxfp8_quantize emits."""
    from flashinfer.cute_dsl.utils import convert_row_major_sf_to_mma_layout
    from flashinfer.fp8_quantization import mxfp8_quantize

    # 192 rows also exercises the padding of each expert's scales to 128 rows.
    num_experts, rows, cols = 2, 192, 384
    torch.manual_seed(1)
    w = torch.randn(num_experts, rows, cols, dtype=torch.bfloat16, device="cuda")
    flat = w.reshape(num_experts * rows, cols)
    _, sf_swizzled = mxfp8_quantize(flat, True)
    _, sf_row_major = mxfp8_quantize(flat, False)
    expected = _quantize_weights(w)[1]
    sf_row_major = sf_row_major.view(torch.uint8).reshape(
        num_experts, rows, cols // MXFP8_BLOCK
    )
    got = convert_row_major_sf_to_mma_layout(
        sf_row_major, m=rows, k=cols, num_groups=num_experts
    )
    assert got.shape == expected.shape
    assert torch.equal(got, expected)
