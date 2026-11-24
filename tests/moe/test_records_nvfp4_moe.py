"""FP4 MoE below SM100 (SM80 to SM90) on the records of prepare_records_nvfp4_weights and
prepare_records_mxfp4_weights: the flat APIs (cutlass, TRT-LLM routed and from logits,
CuTe-DSL) against a dequantized reference and against each other."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import (
    CuteDslMoEWrapper,
    SfLayout,
    block_scale_interleave,
    cute_dsl_fused_moe,
    fp4_quantize,
    nvfp4_quantize,
)
from flashinfer.fused_moe import (
    ActivationType,
    RoutingMethodType,
    cutlass_fused_moe,
    prepare_records_mxfp4_weights,
    prepare_records_nvfp4_weights,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import core
from flashinfer.fused_moe.core import _records_moe
from flashinfer.fused_moe.prepare import W4A16_SM80_TACTICS
from flashinfer.quantization.nvfp4_quantization_utils import (
    NVFP44Over6Config,
    nvfp4_4over6_cache_key,
    nvfp4_per_token_scale_inv,
)
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] not in (8, 9),
    reason="records NVFP4 MoE runs on SM80 to SM90",
)

E, H, I, K = 8, 512, 256, 2
E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)
RECIPES = [None, NVFP44Over6Config()]
RECIPES.append(NVFP44Over6Config(e4m3_max=256, err_mode="MSE", err_use_fast_math=True))


def _e2m1(codes):
    lut = torch.tensor(E2M1, device=codes.device)
    codes = codes.long()
    return torch.stack((lut[codes & 15], lut[codes >> 4]), -1).flatten(-2)


def _dq(codes, sf, scale):
    sf = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return _e2m1(codes) * sf * scale


def _quant(x, gs, recipe=None):
    """NVFP4 codes, linear E4M3 scales (uint8) and the dequantized x, under gs [1]."""
    codes, sf = fp4_quantize(
        x, gs, 16, is_sf_swizzled_layout=False, nvfp4_4over6=recipe
    )
    codes, sf = codes.view(torch.uint8), sf.view(torch.uint8).reshape(x.shape[0], -1)
    return codes, sf, _dq(codes, sf, 1 / gs)


def _base(recipe=None):
    """The per-token base inverse scale (FP32 [1]) of the recipe."""
    return torch.tensor([nvfp4_per_token_scale_inv(recipe)], device="cuda")


def _quant_pt(x, recipe=None):
    """Per token (the CUDA quantizer): codes, scales, token scales and the dequantized x."""
    codes, sf, s = nvfp4_quantize(
        x, _base(recipe), SfLayout.layout_linear, backend="cuda",
        per_token_activation=True, nvfp4_4over6=recipe,
    )  # fmt: skip
    codes, sf = codes.view(torch.uint8), sf.view(torch.uint8).reshape(x.shape[0], -1)
    return codes, sf, s, _dq(codes, sf, s[:, None])


def _model(gated, experts=E, seed=0):
    torch.manual_seed(seed)
    act = ActivationType.Swiglu if gated else ActivationType.Relu2
    m = {"act": act, "gated": gated}
    for name, rows, cols in (("w1", I * (2 if gated else 1), H), ("w2", H, I)):
        w = torch.randn(experts, rows, cols, device="cuda") / 10
        gs = 448 * 6 / w.abs().amax((1, 2))
        q = [_quant(w[e].bfloat16(), gs[e : e + 1]) for e in range(experts)]
        m[name], m[name + "_sf"], m[name + "_deq"] = (
            torch.stack(v) for v in zip(*q, strict=True)
        )
        m[name + "_g"] = 1 / gs  # the dequantizing global scale
    rec = prepare_records_nvfp4_weights(
        m["w1"], m["w1_sf"].view(torch.float8_e4m3fn), m["w1_g"],
        m["w2"], m["w2_sf"].view(torch.float8_e4m3fn), m["w2_g"],
        num_local_experts=experts, hidden_size=H, intermediate_size=I, activation_type=act,
    )  # fmt: skip
    m["r1"], m["r2"] = rec["w1"], rec["w2"]
    return m


def _reference(
    m, xr, ids, weights, gemm2_input, act=lambda g, u: F.silu(g) * u, gemm1_bf16=True
):
    """The top-k sum from the dequantized input xr, quantizing GEMM2's input by gemm2_input."""
    tokens, top_k = ids.shape
    y = torch.zeros(tokens * top_k, H, device=xr.device)
    flat = ids.flatten()
    for e in range(m["w1"].shape[0]):
        sel = (flat == e).nonzero().squeeze(1)
        if sel.numel():
            h = xr[sel // top_k] @ m["w1_deq"][e].T
            if gemm1_bf16:  # the GEMM1 output in BF16
                h = h.bfloat16().float()
            a = act(h[:, I:], h[:, :I]) if m["gated"] else F.relu(h).square()
            y[sel] = gemm2_input(a.bfloat16()) @ m["w2_deq"][e].T
    return (y.view(tokens, top_k, H) * weights.float()[..., None]).sum(1)


# 3e-2 for a quantized GEMM2 input: quantizing the activation from FP32 (the kernels) or from
# BF16 (the references) flips codes at rounding boundaries about this far apart.
def _close(out, ref, tol=3e-2):
    err = (out.float() - ref).norm() / ref.norm()
    assert err < tol, f"relative error {err:.4f}"


def _routing(tokens, top_k=K, experts=E, seed=1):
    torch.manual_seed(seed)
    logits = torch.randn(tokens, experts, device="cuda")
    weights, ids = torch.topk(logits.softmax(-1), top_k)
    return ids.int(), (weights / weights.sum(-1, keepdim=True)).float()


def _trt(m, x, x_scale, ids, weights, s1, s_gate, s2, **kw):
    b1, b2 = kw.pop("gemm1_bias", None), kw.pop("gemm2_bias", None)
    p1 = [kw.pop(f"gemm1_{k}", None) for k in ("alpha", "beta", "clamp_limit")]
    return trtllm_fp4_block_scale_routed_moe(
        (ids, weights), None, x, x_scale, m["w1"], m["r1"], b1, *p1,
        m["w2"], m["r2"], b2, s1, s_gate, s2, kw.pop("num_experts", E), ids.shape[1],
        None, None, kw.pop("inter", I), kw.pop("offset", 0), kw.pop("local", m["r1"].shape[0]),
        None, activation_type=m["act"].value, **kw,
    )  # fmt: skip


def _cutlass(m, x, ids, weights, scales, sl=..., **kw):  # sl: EP's local experts
    return cutlass_fused_moe(
        x, ids, weights, m["w1"][sl].view(torch.long), m["w2"][sl].view(torch.long),
        torch.bfloat16, scales, activation_type=m["act"], **kw,
    )[0]  # fmt: skip


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("tokens", [1, 37, 300])
def test_w4a16(gated, tokens):
    m = _model(gated)
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    ref = _reference(m, x.float(), ids, weights, lambda a: a.float())
    scales = [m["r1"], m["w1_g"], m["r2"], m["w2_g"]]
    out = _cutlass(m, x, ids, weights, scales)
    _close(out, ref)
    trt = _trt(m, x, None, ids, weights, None, m["w1_g"], m["w2_g"])[0]
    cute = cute_dsl_fused_moe(
        x, None, ids, weights, m["w1"], m["r1"], m["w1_g"], None, m["w2"], m["r2"],
        m["w2_g"], E, K, quant_mode="w4a16", activation_type=m["act"].value,
    )  # fmt: skip
    torch.testing.assert_close(trt, out, atol=0, rtol=0)
    torch.testing.assert_close(cute, out, atol=0, rtol=0)
    if not gated:  # CUTLASS-only on TRT-LLM, and the sentinel: rejected as on SM100
        with pytest.raises(ValueError, match="activation type"):
            bad = {**m, "act": ActivationType.ClampedRelu2}
            _trt(bad, x, None, ids, weights, None, m["w1_g"], m["w2_g"])
        with pytest.raises(ValueError, match="activation type"):
            _cutlass({**m, "act": ActivationType.InvalidType}, x, ids, weights, scales)
        ws = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
        for kw in ({"workspace_buffer": ws}, {"profile_ids": [0, 0]}):  # not read
            with pytest.raises(NotImplementedError, match="profile_ids"):
                _cutlass(m, x, ids, weights, scales, **kw)


@pytest.mark.parametrize("recipe", RECIPES, ids=nvfp4_4over6_cache_key)
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("tokens", [1, 37, 300])
def test_w4a4(recipe, gated, tokens):
    """BF16 input quantized by the prep and NVFP4 codes (linear and 128x4 scales) agree. The
    4over6 recipe quantizes cutlass's two GEMM inputs (TRT-LLM's and CuTe-DSL's per token only)."""
    m = _model(gated)
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    e4m3 = recipe.e4m3_max if recipe else 448
    a1 = e4m3 * 6 / x.float().abs().amax().view(1)
    a2 = torch.tensor([e4m3 * 6 / 128.0], device="cuda")
    codes, sf, xr = _quant(x, a1, recipe)
    ref = _reference(m, xr, ids, weights, lambda a: _quant(a, a2, recipe)[2])
    alpha1, alpha2 = m["w1_g"] / a1, m["w2_g"] / a2
    scales = [a1, m["r1"], alpha1, a2, m["r2"], alpha2]

    def cutlass(inp, nvfp4_4over6=recipe, **kw):
        return _cutlass(m, inp, ids, weights, scales, nvfp4_4over6=nvfp4_4over6, **kw)

    out = cutlass(x)
    _close(out, ref)
    swizzled = fp4_quantize(x, a1, 16, nvfp4_4over6=recipe)[1].view(torch.uint8)
    outs = [cutlass(codes, input_sf=sf, swizzled_input_sf=False)]
    outs.append(cutlass(codes, input_sf=swizzled))
    if tokens == 37 and not recipe:  # tuned, the E4M3 scales as vLLM passes them
        with autotune():
            sf8 = swizzled.view(torch.float8_e4m3fn)
            _close(cutlass(codes, input_sf=sf8, tune_max_num_tokens=64), ref)
    if recipe:  # the recipe reached the kernels
        assert not torch.equal(cutlass(x, None), out)
    else:
        s1 = a2 * alpha1 if gated else a2.repeat(E)
        lin = sf.view(torch.float8_e4m3fn)
        for inp, scale in ((x, a1), (codes, lin)):
            outs.append(_trt(m, inp, scale, ids, weights, s1, alpha1, alpha2)[0])
        args = (m, codes, lin, ids, weights, s1, alpha1, alpha2)
        outs.append(_trt(*args, hidden_states_scale_layout=SfLayout.layout_linear)[0])
        with pytest.raises(NotImplementedError, match="hidden_states_scale_layout"):
            _trt(*args, hidden_states_scale_layout=SfLayout.layout_128x4)
        if gated:  # SwigluStep takes its limit only, as on SM100; unclamped, SwiGLU
            step = {**m, "act": ActivationType.SwigluStep}
            ones = torch.ones(E, device="cuda")
            _close(_trt(step, *args[1:], gemm1_clamp_limit=ones * float("inf"))[0], ref)
            for k in ("gemm1_alpha", "gemm1_beta"):
                with pytest.raises(ValueError, match="SwigluStep"):
                    _trt(step, *args[1:], **{k: ones})

        def cute(inp, xsf=sf):
            return cute_dsl_fused_moe(
                inp, xsf.unsqueeze(-1), ids, weights, m["w1"], m["r1"], alpha1, a2, m["w2"],
                m["r2"], alpha2, E, K, activation_type=m["act"].value,
            )  # fmt: skip

        outs.append(cute(codes))
        i8, e5m2 = codes.view(torch.int8), sf.view(torch.float8_e5m2)  # other types
        for bad in (
            lambda: _trt(m, i8, *args[2:]), lambda: _trt(m, codes, e5m2, *args[3:]),
            lambda: cute(i8), lambda: cute(codes, e5m2),
        ):  # fmt: skip
            with pytest.raises(NotImplementedError, match="records"):
                bad()
    for o in outs:
        torch.testing.assert_close(o, out, atol=0, rtol=0)


@pytest.mark.parametrize("recipe", RECIPES, ids=nvfp4_4over6_cache_key)
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("tokens", [1, 37, 300])
def test_per_token(recipe, gated, tokens):
    """Per token: TRT-LLM NVFP4 codes, TRT-LLM BF16 (writing the token scales) and CuTe-DSL
    agree, GEMM2 quantizing its rows under the recipe's base inverse scale (SM100's scalars:
    gated output1_scale_scalar the gate's, else ones)."""
    m = _model(gated)
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    codes, sf, s, xr = _quant_pt(x, recipe)
    ref = _reference(m, xr, ids, weights, lambda a: _quant_pt(a, recipe)[3])
    s1 = m["w1_g"] if gated else torch.ones(E, device="cuda")
    sf8, base, written = (
        sf.view(torch.float8_e4m3fn),
        _base(recipe),
        torch.empty_like(s),
    )
    args = (ids, weights, s1, m["w1_g"], m["w2_g"])
    out = _trt(m, codes, sf8, *args, per_token_scale=s, nvfp4_4over6=recipe)[0]
    _close(out, ref)
    bf16 = _trt(m, x, base, *args, per_token_scale=written, nvfp4_4over6=recipe)[0]
    cute = cute_dsl_fused_moe(
        codes, sf.unsqueeze(-1), ids, weights, m["w1"], m["r1"], m["w1_g"], base, m["w2"],
        m["r2"], m["w2_g"], E, K, activation_type=m["act"].value, per_token_scale=s,
        nvfp4_4over6=recipe,
    )  # fmt: skip
    torch.testing.assert_close(written, s, atol=0, rtol=0)
    torch.testing.assert_close(bf16, out, atol=0, rtol=0)
    torch.testing.assert_close(cute, out, atol=0, rtol=0)
    if recipe:  # the recipe reached the kernels
        std = _trt(m, codes, sf8, *args, per_token_scale=s, nvfp4_4over6=None)[0]
        assert not torch.equal(std, out)


@pytest.mark.parametrize("recipe", RECIPES[:2], ids=nvfp4_4over6_cache_key)
def test_exact(recipe, monkeypatch):
    """FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH: the NVFP4 activations' quantizers as the CUDA
    quantizer's without fast math, on the modules built so."""
    monkeypatch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1")
    load, seen = core._w4a16_moe_sm80_module, []
    monkeypatch.setattr(
        core, "_w4a16_moe_sm80_module", lambda *a: seen.append(a) or load(*a)
    )
    test_w4a4(recipe, True, 37)
    test_per_token(recipe, True, 37)
    assert seen and all(exact for _, exact, *_ in seen)


@pytest.mark.parametrize("gated", [True, False])
def test_tactics(gated):
    m = _model(gated)
    tokens = 300
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    ref = _reference(m, x.float(), ids, weights, lambda a: a.float())
    for tactic in W4A16_SM80_TACTICS:
        out = x.new_empty(tokens, H)
        inputs = [x, ids, weights, m["r1"], m["w1_g"], m["r2"], m["w2_g"], out]
        _records_moe(inputs, I, gated, 8192, tactic)
        _close(out, ref)


@pytest.mark.parametrize("gated", [True, False])
def test_expert_parallel(gated):
    """Each rank's output is the full one with the other ranks' experts weighted 0."""
    m = _model(gated)
    tokens = 37
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    half = E // 2
    for rank in range(2):
        local = (ids >= rank * half) & (ids < (rank + 1) * half)
        full = _trt(m, x, None, ids, weights * local, None, m["w1_g"], m["w2_g"])[0]
        sl = slice(rank * half, (rank + 1) * half)
        scales = [m["r1"][sl], m["w1_g"][sl], m["r2"][sl], m["w2_g"][sl]]
        ep = _cutlass(m, x, ids, weights, scales, sl, ep_size=2, ep_rank=rank)
        sub = {**m, "r1": m["r1"][sl], "r2": m["r2"][sl]}
        trt = _trt(
            sub, x, None, ids, weights, None, m["w1_g"][sl], m["w2_g"][sl],
            offset=rank * half, local=half,
        )[0]  # fmt: skip
        cute = cute_dsl_fused_moe(
            x, None, ids, weights, m["w1"][sl], m["r1"][sl], m["w1_g"][sl], None,
            m["w2"][sl], m["r2"][sl], m["w2_g"][sl], E, K, num_local_experts=half,
            local_expert_offset=rank * half, quant_mode="w4a16",
            activation_type=m["act"].value,
        )  # fmt: skip
        for o in (ep, trt, cute):
            torch.testing.assert_close(o, full, atol=0, rtol=0)


def test_wrapper_tactics():
    """Below SM100 the CuTe-DSL wrapper offers the records tactics its run takes (W4A4 those
    of per token too)."""
    m, tokens = _model(True), 37
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    args = (x, None, ids, weights, m["w1"], m["r1"], m["w1_g"], None, m["w2"], m["r2"])
    args += (m["w2_g"],)
    wrapper = CuteDslMoEWrapper(E, K, H, I, quant_mode="w4a16")
    assert wrapper.get_valid_tactics() == list(W4A16_SM80_TACTICS)
    ref = _reference(m, x.float(), ids, weights, lambda a: a.float())
    for tactic in wrapper.get_valid_tactics():
        _close(wrapper.run(*args, tactic=tactic), ref)
    w4a4 = CuteDslMoEWrapper(E, K, H, I).get_valid_tactics()
    assert w4a4 == list(W4A16_SM80_TACTICS[:4])


def test_alltoall():
    """cutlass with enable_alltoall leaves the rows without a local expert as they were."""
    m = _model(True)
    tokens, half = 37, E // 2
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    sl = slice(half, E)  # rank 1 of 2
    scales = [m["r1"][sl], m["w1_g"][sl], m["r2"][sl], m["w2_g"][sl]]

    def run(**kw):
        return _cutlass(m, x, ids, weights, scales, sl, ep_size=2, ep_rank=1, **kw)

    ep = run()
    out = torch.full_like(ep, 7.0)
    run(output=out, enable_alltoall=True)
    local = (ids >= half).any(1)
    assert local.any() and not local.all()
    torch.testing.assert_close(out[local], ep[local], atol=0, rtol=0)
    assert (out[~local] == 7).all()


@pytest.mark.parametrize("gated", [True, False])
def test_valid_sizes(gated):
    """Hidden states, biases and LoRA delta padded past the records' valid sizes, as on SM100:
    the output and the (zero-padded) activation are the unpadded run's."""
    m = _model(gated)
    tokens, hp, ip, g = 37, H + 128, I + 64, 2 if gated else 1
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    b1 = torch.randn(E, g * I, device="cuda") / 10
    b2 = torch.randn(E, H, device="cuda") / 10
    # LoRA delta: gated only, as on SM100
    lora = (
        torch.randn(tokens, K, 2 * I, device="cuda").bfloat16() / 10 if gated else None
    )
    args = (m, x, None, ids, weights, None, m["w1_g"], m["w2_g"])
    base = _trt(*args, gemm1_bias=b1, gemm2_bias=b2, gemm1_lora_delta=lora)
    padded = _trt(
        m, F.pad(x, (0, hp - H)), *args[2:], inter=ip,
        gemm1_bias=F.pad(b1.view(E, g, I), (0, ip - I)).flatten(1),
        gemm2_bias=F.pad(b2, (0, hp - H)),
        gemm1_lora_delta=lora if lora is None else F.pad(lora.view(tokens, K, 2, I), (0, ip - I)).flatten(2),
        valid_hidden_size=H, valid_intermediate_size=I,
    )  # fmt: skip
    torch.testing.assert_close(padded[0], base[0], atol=0, rtol=0)
    if gated:
        torch.testing.assert_close(padded[2][:, :I], base[2], atol=0, rtol=0)
        assert padded[2].shape[1] == ip and not padded[2][:, I:].any()


@pytest.mark.parametrize("width", [H - 128, H - 2])
def test_narrow_output(width):
    """Without valid sizes a narrower TRT-LLM output receives the leading columns, as on SM100;
    an output, input or record one off 16 B, a CuTe-DSL output of other rows or type and routing of
    other shapes (CuTe-DSL, TRT-LLM, cutlass) are refused."""
    m = _model(True)
    tokens = 37
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    args = (m, x, None, ids, weights, None, m["w1_g"], m["w2_g"])
    buf = torch.full((tokens, H), 7.0, device="cuda", dtype=torch.bfloat16)
    out = _trt(*args, output=buf[:, :width])[0]
    torch.testing.assert_close(out, _trt(*args)[0][:, :width], atol=0, rtol=0)
    assert out.data_ptr() == buf.data_ptr() and (buf[:, width:] == 7).all()
    off = x.new_empty(tokens * H + 1)[1:].view(tokens, H)  # one element off 16 B
    with pytest.raises(RuntimeError, match="16 B aligned"):
        _trt(*args, output=off)
    with pytest.raises(RuntimeError, match="act and records must be 16 B aligned"):
        _trt(m, off.copy_(x), *args[2:])
    r1 = m["r1"].new_empty(m["r1"].numel() + 1)[1:].view(m["r1"].shape).copy_(m["r1"])
    with pytest.raises(RuntimeError, match="act and records must be 16 B aligned"):
        _trt({**m, "r1": r1}, *args[1:])
    scales = [m["r1"], m["w1_g"], m["r2"], m["w2_g"]]
    with pytest.raises(ValueError, match="num_tokens, top_k"):
        _trt(m, x, None, ids[1:], weights[1:], *args[5:])
    with pytest.raises(ValueError, match="num_tokens, top_k"):
        _cutlass(m, x, ids[1:], weights[1:], scales)

    def cute(ids=ids, weights=weights, **kw):
        return cute_dsl_fused_moe(
            x, None, ids, weights, m["w1"], m["r1"], m["w1_g"], None, m["w2"], m["r2"],
            m["w2_g"], E, K, quant_mode="w4a16", activation_type=m["act"].value, **kw,
        )  # fmt: skip

    for bad in (x.new_empty(tokens + 1, H), x.new_empty(tokens, H, dtype=torch.half)):
        with pytest.raises(ValueError, match="moe_output"):
            cute(moe_output=bad)
    for bad in ((ids[1:], weights[1:]), (ids, weights[:, :1])):
        with pytest.raises(ValueError, match="num_tokens, top_k"):
            cute(*bad)


def test_fc2_input_out():
    """Per token, GEMM2's NVFP4 input as SM100's fc2_input_out, in buffers of a padded
    intermediate size: the local entries decode to the rows GEMM2 ran on, the others zero."""
    m = _model(False)
    tokens, half, ip = 37, E // 2, I + 64
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    codes, sf, s, _ = _quant_pt(x)
    sub = {**m, "r1": m["r1"][:half], "r2": m["r2"][:half]}
    fo = [torch.full((tokens + 3, K, ip // n), 9, dtype=torch.uint8, device="cuda")
          for n in (2, 16)] + [torch.full((tokens + 3, K), 9.0, device="cuda")]  # fmt: skip
    out = _trt(
        sub, codes, sf.view(torch.float8_e4m3fn), ids, weights, torch.ones(half, device="cuda"),
        m["w1_g"][:half], m["w2_g"][:half], num_experts=E, local=half, inter=ip,
        valid_intermediate_size=I, per_token_scale=s, fc2_input_out=fo,
    )[0]  # fmt: skip
    local = ids < half
    assert local.any() and not local.all()
    assert all(t[tokens:].eq(9).all() for t in fo)  # rows past the tokens untouched
    rows = _dq(
        fo[0][:tokens, :, : I // 2],
        fo[1][:tokens, :, : I // 16],
        fo[2][:tokens, :, None],
    )
    assert not rows[~local].any() and not fo[0][:tokens][~local].any()
    y = torch.zeros(tokens, K, H, device="cuda")
    for e in range(half):
        y[ids == e] = rows[ids == e] @ m["w2_deq"][e].T
    _close(out, (y * weights[..., None]).sum(1), 1e-2)


def test_deferred_finalize_and_shared_experts():
    m = _model(True, experts=E + 1)
    tokens = 37
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    args = (m, x, None, ids, weights, None, m["w1_g"], m["w2_g"])
    method = RoutingMethodType.DeepSeekV3.value
    shared = _trt(
        *args, local=E, routing_method_type=method, num_fused_shared_experts=1
    )
    ids1 = torch.cat((ids, torch.full_like(ids[:, :1], E)), 1)
    w1 = torch.cat((weights, torch.ones_like(weights[:, :1])), 1)
    args1 = (m, x, None, ids1, w1, None, m["w1_g"], m["w2_g"])
    explicit = _trt(*args1, num_experts=E + 1)
    torch.testing.assert_close(shared[0], explicit[0], atol=0, rtol=0)
    y, w, idx = _trt(*args1, num_experts=E + 1, do_finalize=False)
    summed = (y[idx.long()].view(tokens, K + 1, H).float() * w[..., None]).sum(1)
    _close(explicit[0], summed, 1e-2)


@pytest.mark.parametrize("tokens", [1, 37, 300])
def test_from_logits(tokens):
    """Routing in the kernel: the slots, and with a replay buffer the gathered ids."""
    m = _model(True)
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    logits = torch.randn(tokens, E, device="cuda")

    def run(**kw):
        return trtllm_fp4_block_scale_moe(
            logits, None, x, None, m["w1"], m["r1"], None, None, None, None, m["w2"],
            m["r2"], None, None, m["w1_g"], m["w2_g"], E, K, None, None, I, 0, E, None,
            routing_method_type=RoutingMethodType.Renormalize.value, **kw,
        )[0]  # fmt: skip

    replay = torch.empty(tokens, K, dtype=torch.int16, device="cuda")
    slots, gathered = run(), run(routing_replay_out=replay)
    with pytest.raises(ValueError, match="activation type"):
        run(activation_type=ActivationType.InvalidType.value)
    with pytest.raises(NotImplementedError, match="workspace_buffer"):
        run(workspace_buffer=torch.empty(1 << 20, dtype=torch.uint8, device="cuda"))
    torch.testing.assert_close(slots, gathered, atol=0, rtol=0)
    ids = torch.topk(logits, K).indices
    assert torch.equal(replay.long().sort(1).values, ids.sort(1).values)
    weights = torch.gather(logits, 1, replay.long()).softmax(-1)
    _close(slots, _reference(m, x.float(), replay, weights, lambda a: a.float()), 2e-2)


def test_from_logits_rank0(monkeypatch):
    """EP rank 0 from logits at row tile 8 (where the records keep the slots): the other
    rank's slots are left out of the sum, as with the gathered ids of a replay buffer."""
    m, tokens, half = _model(True), 37, E // 2
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    logits = torch.randn(tokens, E, device="cuda")
    a1 = 448 * 6 / x.float().abs().amax().view(1)
    a2 = torch.tensor([448 * 6 / 128.0], device="cuda")
    codes, sf, _ = _quant(x, a1)
    alpha1, alpha2 = m["w1_g"][:half] / a1, m["w2_g"][:half] / a2
    tile8 = W4A16_SM80_TACTICS[0]
    monkeypatch.setattr(AutoTuner, "choose_one", lambda self, *a: (None, tile8))

    def run(lg=logits, **kw):
        return trtllm_fp4_block_scale_moe(
            lg, None, codes, sf.view(torch.float8_e4m3fn), m["w1"][:half],
            m["r1"][:half], None, None, None, None, m["w2"][:half], m["r2"][:half], None,
            a2 * alpha1, alpha1, alpha2, E, K, None, None, I, 0, half, None,
            routing_method_type=RoutingMethodType.Renormalize.value, **kw,
        )[0]  # fmt: skip

    # The other rank's experts first: GEMM2's rows of this rank's left behind (token, k) order
    run(logits.roll(half, 1))
    replay = torch.empty(tokens, K, dtype=torch.int16, device="cuda")
    torch.testing.assert_close(run(), run(routing_replay_out=replay), atol=0, rtol=0)


def _mx_model(gated, split=False, seed=0):
    """Checkpoint MXFP4 weights (random codes and E8M0 scales, zero blocks in expert 0) and their
    records; with ``split`` the gated halves' scales lie 40 binades apart, a shift per half."""
    torch.manual_seed(seed)
    act = ActivationType.Swiglu if gated else ActivationType.Relu2
    m = {"act": act, "gated": gated}
    for name, rows, cols in (("w1", I * (2 if gated else 1), H), ("w2", H, I)):
        w = torch.randint(256, (E, rows, cols // 2), dtype=torch.uint8, device="cuda")
        s = torch.randint(
            118, 124, (E, rows, cols // 32), dtype=torch.uint8, device="cuda"
        )
        if split and name == "w1":
            s[:, I:] += 40  # the gate half
        w[0, :, :16], s[0, :, 0] = 0, 250  # zero blocks: their scales are not read
        m[name], m[name + "_sf"] = w, s
        m[name + "_deq"] = _e2m1(w) * torch.exp2(s.float() - 127).repeat_interleave(
            32, -1
        )
    rec = prepare_records_mxfp4_weights(
        m["w1"], m["w1_sf"], m["w2"], m["w2_sf"],
        num_local_experts=E, hidden_size=H, intermediate_size=I, activation_type=act,
    )  # fmt: skip
    m["r1"], m["w1_g"], m["r2"], m["w2_g"] = rec.values()
    assert m["w1_g"].shape == ((E, 2) if split else (E,))
    return m


def _mxfp8(a):
    """MXFP8 codes (uint8), linear E8M0 scales and the dequantized a."""
    b = a.float().unflatten(-1, (-1, 32))
    e = torch.ceil(torch.log2(b.abs().amax(-1, keepdim=True) / 448)).clamp(-127, 127)
    q = (b / torch.exp2(e)).to(torch.float8_e4m3fn)
    deq = (q.float() * torch.exp2(e)).flatten(-2)
    return q.flatten(-2).view(torch.uint8), (e.squeeze(-1) + 127).to(torch.uint8), deq


MX_MODELS = [(True, False), (True, True), (False, False)]


@pytest.mark.parametrize("gated,split", MX_MODELS)
@pytest.mark.parametrize("tokens", [1, 37, 300])
def test_mxfp4_mxfp8(gated, split, tokens):
    """MXFP8 input (linear and 128x4 scales), GEMM2's input MXFP8: cutlass, TRT-LLM and
    CuTe-DSL W4A8 agree."""
    m = _mx_model(gated, split)
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    codes, sf, xr = _mxfp8(x)
    ref = _reference(m, xr, ids, weights, lambda a: _mxfp8(a)[2])
    x8, scales = (
        codes.view(torch.float8_e4m3fn),
        [m["r1"], m["w1_g"], m["r2"], m["w2_g"]],
    )
    kw = {"use_mxfp8_act_scaling": True}
    out = _cutlass(
        m, x8, ids, weights, scales, input_sf=sf, swizzled_input_sf=False, **kw
    )
    _close(out, ref)
    outs = [
        _cutlass(m, x8, ids, weights, scales, input_sf=block_scale_interleave(sf), **kw)
    ]
    outs.append(_trt(m, x8, sf, ids, weights, None, m["w1_g"], m["w2_g"])[0])

    def cute(xsf):
        return cute_dsl_fused_moe(
            x8, xsf, ids, weights, m["w1"], m["r1"], m["w1_g"], None, m["w2"], m["r2"],
            m["w2_g"], E, K, quant_mode="w4a8", activation_type=m["act"].value,
        )  # fmt: skip

    outs.append(cute(sf))
    for o in outs:
        torch.testing.assert_close(o, out, atol=0, rtol=0)
    with pytest.raises(NotImplementedError, match="records"):  # [M, H/32] as on SM100
        cute(sf.t().contiguous())


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("tokens", [1, 37, 300])
def test_mxfp4_fp8_per_tensor(gated, tokens):
    """FP8 input under one scale, GEMM2's input FP8 under q2: cutlass and TRT-LLM agree."""
    m = _mx_model(gated)
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    q1, q2 = 448 / x.float().abs().amax(), torch.tensor([4.0], device="cuda")
    x8 = (x.float() * q1).to(torch.float8_e4m3fn)

    def fp8(a):
        return (a.float() * q2).clamp(-448, 448).to(torch.float8_e4m3fn).float() / q2

    ref = _reference(m, x8.float() / q1, ids, weights, fp8)
    a1, a2 = m["w1_g"] / q1, m["w2_g"] / q2
    out = _cutlass(m, x8, ids, weights, [m["r1"], a1, q2, m["r2"], a2])
    _close(out, ref)
    s1 = q2 * a1 if gated else q2.repeat(E)  # SM100's: gated over the gate half's alpha
    trt = _trt(m, x8, None, ids, weights, s1, a1, a2)[0]
    torch.testing.assert_close(trt, out, atol=0, rtol=0)


@pytest.mark.parametrize("gated,split", MX_MODELS)
def test_mxfp4_w4a16(gated, split):
    """BF16 input: TRT-LLM and, below SM90 (SM90 runs its CUTLASS W4A16 kernels), cutlass."""
    m = _mx_model(gated, split)
    tokens = 37
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    trt = _trt(m, x, None, ids, weights, None, m["w1_g"], m["w2_g"])[0]
    _close(trt, _reference(m, x.float(), ids, weights, lambda a: a.float()), 2e-2)
    if get_compute_capability(x.device)[0] < 9:
        scales = [m["r1"], m["w1_g"], m["r2"], m["w2_g"]]
        out = cutlass_fused_moe(
            x, ids, weights, m["w1"], m["w2"], torch.bfloat16, scales,
            use_w4_group_scaling=True, activation_type=m["act"],
        )[0]  # fmt: skip
        torch.testing.assert_close(out, trt, atol=0, rtol=0)


def _fp8_rows(a):
    """Dynamic FP8 per row as SM90's Humming path: a dequantized."""
    a = a.float()
    amax = a.abs().amax(-1, keepdim=True)
    q = torch.where(amax > 0, 448 / amax, torch.ones_like(amax))
    return (a * q).clamp(-448, 448).to(torch.float8_e4m3fn).float() / q


HUMMING = [(torch.bfloat16, *model) for model in MX_MODELS]
HUMMING += [(torch.float16, gated, False) for gated in (True, False)]


@pytest.mark.parametrize("dtype,gated,split", HUMMING)
@pytest.mark.parametrize("tokens", [1, 37, 300])
def test_mxfp4_humming(dtype, gated, split, tokens):
    """use_wfp4afp8_humming: 16-bit input in FP8 per token, GEMM2's input in FP8 per row (SM90
    runs its CUTLASS kernels)."""
    if get_compute_capability(torch.device("cuda"))[0] > 8:
        pytest.skip("SM90 runs its CUTLASS Humming kernels")
    m = _mx_model(gated, split)
    x = torch.randn(tokens, H, device="cuda").to(dtype)
    ids, weights = _routing(tokens)
    scales = [m["r1"], m["w1_g"], torch.ones(E, device="cuda"), m["r2"], m["w2_g"]]
    out = cutlass_fused_moe(
        x, ids, weights, m["w1"], m["w2"], dtype, scales, use_w4_group_scaling=True,
        use_wfp4afp8_humming=True, activation_type=m["act"],
    )[0]  # fmt: skip
    _close(out, _reference(m, _fp8_rows(x), ids, weights, _fp8_rows))
