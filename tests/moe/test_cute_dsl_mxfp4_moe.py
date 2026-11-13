"""Tests for the CuTe-DSL MXFP4 x MXFP4 (E2M1 + E8M0 block-32 scales) MoE path.

Oracles:

* GEMM1 alone -- the MXFP4 intermediate that the epilogue writes (E2M1 values
  and E8M0 scale bytes) against a PyTorch MXFP4 quantization of an FP32
  reference intermediate built from the dequantized operands. This checks the
  E8M0 scale recipe, the scale-factor layout and the gathered A scales.
* end to end -- cosine similarity against the unquantized BF16 MoE, and a
  tight comparison against a PyTorch reference that applies the same MXFP4
  quantize/dequantize to the input, the weights and the GEMM1->GEMM2
  intermediate.

The PyTorch MXFP4 quantizer is checked against ``mxfp4_quantize`` first, so
both oracles use FlashInfer's own recipe: scale = 2^ceil(log2(amax / 6))
(E8M0, round toward +inf), values rounded to nearest-even E2M1.
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

mxfp4_required = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0) in ((10, 0), (10, 3))
    ),
    reason="MXFP4 CuTe-DSL MoE requires SM100 or SM103",
)

pytestmark = [_requires_dsl_arch, mxfp4_required]

MX_BLOCK = 32
# MXFP4 on the input, both weights and the intermediate leaves a cosine of
# about 0.955 (SwiGLU) / 0.967 (ReLU^2) to the BF16 MoE on these Gaussian
# cases, so the BF16 check only guards against layout or routing bugs.
MIN_COS_SIM_VS_BF16 = 0.9
MIN_COS_SIM_VS_MXFP4_REF = 0.999

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_round(t: torch.Tensor) -> torch.Tensor:
    """Round FP32 to E2M1 (nearest, ties to even, saturating at 6)."""
    a = t.abs()
    # Ties at 0.25, 1.25, 2.5 and 5 round down (to an even mantissa); ties at
    # 0.75, 1.75 and 3.5 round up.
    idx = (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=t.device)
    return torch.where(torch.signbit(t), -values[idx], values[idx])


def _e8m0_codes(amax: torch.Tensor) -> torch.Tensor:
    """E8M0 code of amax / 6 rounded toward +inf (``float_to_ue8m0_fast``)."""
    v = amax.float() * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    bits = v.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    bump = (mant != 0) & ~((exp == 0) & (mant <= 0x400000))
    code = torch.clamp(exp + bump.int(), max=254)
    return torch.where(v <= 0, torch.zeros_like(code), code)


def _quantize_mxfp4_ref(t: torch.Tensor):
    """PyTorch MXFP4 along the last dim: (dequantizable values, E8M0 codes)."""
    shape = t.shape
    blocks = t.float().reshape(*shape[:-1], shape[-1] // MX_BLOCK, MX_BLOCK)
    codes = _e8m0_codes(blocks.abs().amax(dim=-1))
    inv_scale = torch.where(
        codes == 0,
        torch.zeros_like(codes, dtype=torch.float32),
        torch.exp2((127 - codes).float()),
    )
    q = _e2m1_round(blocks * inv_scale.unsqueeze(-1))
    return q, codes


def _qdq_mxfp4(t: torch.Tensor) -> torch.Tensor:
    q, codes = _quantize_mxfp4_ref(t)
    scale = torch.exp2((codes - 127).float())
    return (q * scale.unsqueeze(-1)).reshape(t.shape)


def _decode_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 E2M1 pairs (low nibble first) to FP32."""
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).long()
    out = values[nibbles & 7]
    out = torch.where((nibbles & 8) != 0, -out, out)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _dequant_mxfp4(packed: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    vals = _decode_e2m1(packed)
    scale = torch.exp2((codes.int() - 127).float())
    scale = torch.where(codes == 0, torch.zeros_like(scale), scale)
    blocks = vals.reshape(*vals.shape[:-1], -1, MX_BLOCK)
    return (blocks * scale.unsqueeze(-1)).reshape(vals.shape)


def _unswizzle_mma_sf(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """(32, 4, rows/128, 4, k_tiles, 1) MMA-layout scales -> [rows, cols / 32]."""
    k_tiles = (cols // MX_BLOCK + 3) // 4
    flat = sf.contiguous().view(torch.uint8).reshape(-1)
    flat = flat[: (rows // 128) * k_tiles * 512]
    t = flat.view(rows // 128, k_tiles, 32, 4, 4).permute(0, 3, 2, 1, 4)
    return t.reshape(rows, k_tiles * 4)[:, : cols // MX_BLOCK]


def _quantize_input(x: torch.Tensor):
    from flashinfer import SfLayout, mxfp4_quantize

    x_q, x_sf = mxfp4_quantize(x, sfLayout=SfLayout.layout_linear)
    return x_q, x_sf.view(torch.uint8).reshape(x.shape[0], x.shape[1] // MX_BLOCK)


def _prepare_weights(w1, w2, activation):
    from flashinfer.fused_moe import CuteDslConfig, QuantConfig, QuantFormat

    num_experts, hidden = w2.shape[0], w2.shape[1]
    inter = w2.shape[2]
    return CuteDslConfig.prepare_weights(
        w1,
        w2,
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP4),
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=inter,
        activation=activation,
    )


def _activation(kind):
    from flashinfer.fused_moe import ReLU2, SwiGLU

    if kind == "swiglu":
        return SwiGLU(), ActivationType.Swiglu
    return ReLU2(), ActivationType.Relu2


def _apply_activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "swiglu":
        up, gate = z.chunk(2, dim=-1)
        return up * F.silu(gate)
    return torch.relu(z) ** 2


def _reference(x, w1, w2, topk_ids, topk_w, kind, quantized: bool):
    """FP32 MoE. ``w1`` is [E, 2I or I, H] with contiguous [up, gate] halves."""
    qdq = _qdq_mxfp4 if quantized else (lambda t: t.float())
    xf, w1f, w2f = qdq(x), qdq(w1), qdq(w2)
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    for e in range(w1.shape[0]):
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = _apply_activation(xf[tok] @ w1f[e].T, kind)
        y = qdq(h) @ w2f[e].T
        out.index_add_(0, tok, y * topk_w[tok, slot].unsqueeze(1))
    return out


def _make_case(
    num_tokens, hidden, inter, num_experts, top_k, kind, seed=0, skewed=False
):
    """Random MoE case. ``skewed`` routes every token's first slot to expert 0
    (several full 128-row tiles, both CTAs of a 256-row tile) and the other
    slots round-robin over the next ``top_k - 1`` * 2 experts, leaving the
    remaining experts idle."""
    torch.manual_seed(seed)
    dev = "cuda"
    gemm1_rows = 2 * inter if kind == "swiglu" else inter
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev)
    w1 = torch.randn(num_experts, gemm1_rows, hidden, dtype=torch.bfloat16, device=dev)
    w1 = w1 / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16, device=dev)
    w2 = w2 / inter**0.5
    if skewed:
        tok = torch.arange(num_tokens, device=dev)
        slots = [torch.zeros_like(tok)]
        for k in range(1, top_k):
            slots.append(1 + 2 * (k - 1) + tok % 2)
        topk_ids = torch.stack(slots, dim=1).to(torch.int32)
        assert int(topk_ids.max()) < num_experts - 1, "need idle experts"
    else:
        scores = torch.rand(num_tokens, num_experts, device=dev)
        topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_w = torch.softmax(torch.randn(num_tokens, top_k, device=dev), dim=-1)
    return x, w1, w2, topk_ids, topk_w.float()


def _tactic(tile):
    """1-CTA 128x128 tiles, or 2-CTA 256x256 tiles, for both GEMMs."""
    return (
        tile,
        ((tile, tile), (tile // 128, 1), False),
        ((tile, tile), (tile // 128, 1), False),
    )


def _run_moe(x, w1, w2, topk_ids, topk_w, kind, tactic=None, **kwargs):
    from flashinfer import cute_dsl_fused_moe

    activation, activation_type = _activation(kind)
    view = _prepare_weights(w1, w2, activation)
    x_q, x_sf = _quantize_input(x)
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
        num_experts=w1.shape[0],
        top_k=topk_ids.shape[1],
        activation_type=activation_type.value,
        quant_mode="mxfp4",
        tactic=tactic,
        **kwargs,
    )


def _cos(a, b):
    return F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0)


def test_reference_quantizer_matches_mxfp4_quantize():
    """The PyTorch oracle reproduces mxfp4_quantize's values and scale bytes."""
    torch.manual_seed(0)
    x = torch.randn(256, 1024, dtype=torch.bfloat16, device="cuda") * 3
    x[0, :32] = 0  # a zero block
    x[1, :32] = 6.0  # amax / 6 exactly a power of two
    x_q, x_sf = _quantize_input(x)
    q_ref, codes_ref = _quantize_mxfp4_ref(x)
    assert torch.equal(x_sf.int(), codes_ref.int())
    assert torch.equal(_decode_e2m1(x_q), q_ref.reshape(x.shape))


@pytest.mark.parametrize("kind", ["swiglu", "relu2"])
@pytest.mark.parametrize("tile", [128, 256])
def test_mxfp4_gemm1_epilogue_matches_reference(kind, tile):
    """GEMM1 writes the MXFP4 quantization of the activated FC1 output."""
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
        blockscaled_contiguous_gather_grouped_gemm_act_fusion,
    )
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort

    num_tokens, hidden, inter, num_experts, top_k = 300, 1024, 512, 8, 2
    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k, kind, skewed=True
    )
    activation, activation_type = _activation(kind)
    view = _prepare_weights(w1, w2, activation)
    x_q, x_sf = _quantize_input(x)
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=topk_ids,
        token_final_scales=topk_w,
        num_experts=num_experts,
        top_k=top_k,
        tile_tokens_dim=tile,
    )
    out, out_sf = blockscaled_contiguous_gather_grouped_gemm_act_fusion(
        a=x_q,
        b=view["w1_weight"],
        a_scale=x_sf,
        b_scale=view["w1_weight_sf"],
        alpha=view["w1_alpha"],
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        token_id_mapping=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        topk=top_k,
        a_dtype="float4_e2m1fn",
        b_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="float4_e2m1fn",
        sf_vec_size=MX_BLOCK,
        quantize_output=True,
        mma_tiler_mn=(tile, tile),
        cluster_shape_mn=(tile // 128, 1),
        activation_type=activation_type.value,
        gated=kind == "swiglu",
    )
    torch.cuda.synchronize()
    permuted_m = out.shape[0]
    assert out.shape == (permuted_m, inter // 2)
    sf_rows = _unswizzle_mma_sf(out_sf, permuted_m, inter)

    # moe_sort leaves padding rows of the reverse map uninitialized, so take
    # the live rows from the forward map (every expert is local here).
    valid = expanded_idx_to_permuted_idx.reshape(-1).long()
    assert valid.min() >= 0 and valid.max() < permuted_m
    token = torch.arange(num_tokens, device="cuda").repeat_interleave(top_k)
    slot = torch.arange(top_k, device="cuda").repeat(num_tokens)
    expert = topk_ids[token, slot].long()

    xf, w1f = _qdq_mxfp4(x), _qdq_mxfp4(w1)
    z = torch.einsum("rh,rnh->rn", xf[token], w1f[expert])
    h_ref = _apply_activation(z, kind)
    q_ref, codes_ref = _quantize_mxfp4_ref(h_ref)

    # The FP32 accumulation order differs from the reference, so an element or
    # a block amax that sits on a rounding boundary may land one step apart.
    codes = sf_rows[valid].int()
    assert (codes == codes_ref.int()).float().mean() > 0.995
    got = _dequant_mxfp4(out[valid], sf_rows[valid])
    want = (q_ref * torch.exp2((codes_ref - 127).float()).unsqueeze(-1)).reshape(
        h_ref.shape
    )
    assert (got == want).float().mean() > 0.99
    assert _cos(got, want) > 0.9999
    # Quantization noise only: cos(qdq(h), h) is about 0.988 / 0.992 here.
    assert _cos(got, h_ref) > 0.97


@pytest.mark.parametrize("kind", ["swiglu", "relu2"])
@pytest.mark.parametrize("tile", [128, 256])
@pytest.mark.parametrize("use_fused_finalize", [True, False])
def test_mxfp4_moe_matches_reference(kind, tile, use_fused_finalize):
    num_tokens, hidden, inter, num_experts, top_k = 300, 1024, 768, 8, 2
    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k, kind, skewed=True
    )
    out = _run_moe(
        x,
        w1,
        w2,
        topk_ids,
        topk_w,
        kind,
        tactic=_tactic(tile),
        use_fused_finalize=use_fused_finalize,
    ).float()
    if not use_fused_finalize:
        again = _run_moe(
            x,
            w1,
            w2,
            topk_ids,
            topk_w,
            kind,
            tactic=_tactic(tile),
            use_fused_finalize=False,
        ).float()
        assert torch.equal(out, again), "deterministic finalize is not run-to-run exact"

    ref_bf16 = _reference(x, w1, w2, topk_ids, topk_w, kind, quantized=False)
    cos = _cos(out, ref_bf16)
    assert cos > MIN_COS_SIM_VS_BF16, f"cos vs BF16 {cos:.4f}"

    ref_q = _reference(x, w1, w2, topk_ids, topk_w, kind, quantized=True)
    cos_q = _cos(out, ref_q)
    assert cos_q > MIN_COS_SIM_VS_MXFP4_REF, f"cos vs MXFP4 reference {cos_q:.5f}"
    torch.testing.assert_close(out, ref_q, rtol=0.1, atol=0.05 * ref_q.abs().max())


@pytest.mark.parametrize(
    "num_tokens,hidden,inter,num_experts,top_k",
    [
        (1, 512, 256, 8, 2),
        # More routed rows than one tile per expert, with idle experts.
        (300, 768, 384, 32, 6),
    ],
)
def test_mxfp4_moe_shapes_and_autotune(num_tokens, hidden, inter, num_experts, top_k):
    from flashinfer import autotune

    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k, "swiglu"
    )
    with autotune(True):
        out = _run_moe(x, w1, w2, topk_ids, topk_w, "swiglu").float()
    ref_q = _reference(x, w1, w2, topk_ids, topk_w, "swiglu", quantized=True)
    cos_q = _cos(out, ref_q)
    assert cos_q > MIN_COS_SIM_VS_MXFP4_REF, f"cos vs MXFP4 reference {cos_q:.5f}"


def test_mxfp4_wrapper_cuda_graph_replay():
    """A captured wrapper call tracks new inputs written into its buffers."""
    from flashinfer import CuteDslMoEWrapper

    num_tokens, hidden, inter, num_experts, top_k = 64, 512, 256, 8, 2
    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k, "swiglu"
    )
    x2, _, _, topk_ids2, topk_w2 = _make_case(
        num_tokens, hidden, inter, num_experts, top_k, "swiglu", seed=1
    )
    view = _prepare_weights(w1, w2, _activation("swiglu")[0])
    moe = CuteDslMoEWrapper(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=inter,
        use_cuda_graph=True,
        quant_mode="mxfp4",
    )
    x_q, x_sf = _quantize_input(x)
    weights = (
        view["w1_weight"],
        view["w1_weight_sf"],
        view["w1_alpha"],
        None,
        view["w2_weight"],
        view["w2_weight_sf"],
        view["w2_alpha"],
    )
    ids_buf, w_buf = topk_ids.clone(), topk_w.clone()
    for _ in range(2):
        moe.run(x_q, x_sf, ids_buf, w_buf, *weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = moe.run(x_q, x_sf, ids_buf, w_buf, *weights)

    x2_q, x2_sf = _quantize_input(x2)
    x_q.copy_(x2_q)
    x_sf.copy_(x2_sf)
    ids_buf.copy_(topk_ids2)
    w_buf.copy_(topk_w2)
    graph.replay()
    eager = moe.run(x2_q, x2_sf, topk_ids2, topk_w2, *weights)
    torch.cuda.synchronize()
    ref_q = _reference(x2, w1, w2, topk_ids2, topk_w2, "swiglu", quantized=True)
    assert _cos(out.float(), ref_q) > MIN_COS_SIM_VS_MXFP4_REF
    assert _cos(out.float(), eager.float()) > 0.9999


def test_mxfp4_mode_rejects_invalid_inputs():
    from flashinfer import cute_dsl_fused_moe

    num_tokens, hidden, inter, num_experts, top_k = 8, 256, 256, 4, 2
    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k, "swiglu"
    )
    view = _prepare_weights(w1, w2, _activation("swiglu")[0])
    x_q, x_sf = _quantize_input(x)
    kwargs = dict(
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
        quant_mode="mxfp4",
    )
    with pytest.raises(NotImplementedError, match="swap_ab=False"):
        cute_dsl_fused_moe(**kwargs, swap_ab=True)
    with pytest.raises(TypeError, match="packed uint8"):
        cute_dsl_fused_moe(**{**kwargs, "x": x.to(torch.float8_e4m3fn)})
    with pytest.raises(ValueError, match="fc2_input_scale must be None"):
        cute_dsl_fused_moe(
            **{**kwargs, "fc2_input_scale": torch.ones(1, device="cuda")},
            tactic=_tactic(128),
        )
    with pytest.raises(ValueError, match="x_sf must have shape"):
        cute_dsl_fused_moe(
            **{**kwargs, "x_sf": x_sf[:, : hidden // 64].contiguous()},
            tactic=_tactic(128),
        )
    with pytest.raises(ValueError, match="bfloat16 output"):
        cute_dsl_fused_moe(**kwargs, output_dtype=torch.float16)


def test_mx_gemm1_output_rejects_global_scale():
    """MXFP4 output has no global scale; passing one is an error, not ignored."""
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
        blockscaled_contiguous_gather_grouped_gemm_act_fusion,
    )

    dev = "cuda"
    a = torch.zeros(128, 128, dtype=torch.uint8, device=dev)
    b = torch.zeros(1, 256, 128, dtype=torch.uint8, device=dev)
    sf = torch.zeros(128, 256 // MX_BLOCK, dtype=torch.uint8, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="global_scale must be None"):
        blockscaled_contiguous_gather_grouped_gemm_act_fusion(
            a=a,
            b=b,
            a_scale=sf,
            b_scale=sf,
            alpha=torch.ones(1, device=dev),
            tile_idx_to_expert_idx=torch.zeros(1, **i32),
            tile_idx_to_mn_limit=torch.full((1,), 128, **i32),
            token_id_mapping=torch.arange(128, **i32),
            num_non_exiting_tiles=torch.ones(1, **i32),
            global_scale=torch.ones(1, device=dev),
            topk=1,
            sf_dtype="float8_e8m0fnu",
            c_dtype="float4_e2m1fn",
            sf_vec_size=MX_BLOCK,
            quantize_output=True,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )


def test_mxfp4_rejects_routing_weights_before_fc2(monkeypatch):
    """Pre-FC2 routing weighting quantizes the GEMM2 input as NVFP4 (one FC2
    input scale per expert), so the MXFP4 mode rejects it before any routing."""
    import importlib

    from flashinfer import CuteDslMoEWrapper, cute_dsl_fused_moe
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
        blockscaled_contiguous_gather_grouped_gemm_act_fusion,
    )

    api = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")

    def forbidden_sort(*args, **kwargs):
        raise AssertionError("MXFP4 with pre-FC2 routing reached GPU routing")

    num_tokens, hidden, inter, num_experts, top_k = 8, 256, 256, 4, 2
    x, w1, w2, topk_ids, topk_w = _make_case(
        num_tokens, hidden, inter, num_experts, top_k, "relu2"
    )
    view = _prepare_weights(w1, w2, _activation("relu2")[0])
    x_q, x_sf = _quantize_input(x)
    limit = torch.tensor([16.0], dtype=torch.float32, device="cuda")
    tensors = dict(
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
    )
    options = dict(
        num_experts=num_experts,
        top_k=top_k,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=limit,
        routing_weights_before_fc2=True,
    )
    monkeypatch.setattr(api, "moe_sort", forbidden_sort)
    match = r"quant_mode='w4a4'"
    with pytest.raises(ValueError, match=match):
        cute_dsl_fused_moe(
            **tensors, **options, quant_mode="mxfp4", tactic=_tactic(128)
        )
    with pytest.raises(ValueError, match=match):
        CuteDslMoEWrapper(
            hidden_size=hidden, intermediate_size=inter, quant_mode="mxfp4", **options
        )
    with pytest.raises(ValueError, match=match):
        api._moe_core_impl(
            **tensors,
            num_local_experts=num_experts,
            quant_mode="mxfp4",
            **options,
        )
    # The GEMM1 entry point accepts routing weights only for NVFP4 output.
    with pytest.raises(ValueError, match="routing_weights requires"):
        blockscaled_contiguous_gather_grouped_gemm_act_fusion(
            a=x_q,
            b=view["w1_weight"],
            a_scale=x_sf,
            b_scale=view["w1_weight_sf"],
            alpha=view["w1_alpha"],
            tile_idx_to_expert_idx=torch.zeros(1, dtype=torch.int32, device="cuda"),
            tile_idx_to_mn_limit=torch.zeros(1, dtype=torch.int32, device="cuda"),
            token_id_mapping=torch.full((128,), -1, dtype=torch.int32, device="cuda"),
            num_non_exiting_tiles=torch.zeros(1, dtype=torch.int32, device="cuda"),
            topk=top_k,
            sf_dtype="float8_e8m0fnu",
            c_dtype="float4_e2m1fn",
            sf_vec_size=MX_BLOCK,
            quantize_output=True,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            activation_type=ActivationType.ClampedRelu2.value,
            gated=False,
            clamped_relu2_limit=limit,
            routing_weights=torch.ones(
                num_tokens * top_k, dtype=torch.float32, device="cuda"
            ),
        )
