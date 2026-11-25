"""The unified MoELayer on the records backends (SM80 to SM90): each FP4 pair, NVFP4 per
token and an activation beyond TRT-LLM's against the dequantized reference of
tests/moe/test_records_nvfp4_moe.py."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    PowLU,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
    SwiGLU,
    SwiGLUStep,
)
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.api import (
    _DEFAULT_BACKEND,
    RecordsMxfp4Config,
    RecordsNvfp4Bf16Config,
    RecordsNvfp4Config,
)
from flashinfer.fused_moe.core import _trtllm_routing_sm80
from flashinfer.fused_moe.prepare import W4A16_SM80_TACTICS
from flashinfer.quantization.fp4_quantization import mxfp4_quantize
from flashinfer.tllm_enums import SfLayout
from flashinfer.utils import get_compute_capability
from tests.moe.test_records_nvfp4_moe import (
    E, H, I, K, _close, _e2m1, _model, _mx_model, _mxfp8, _quant, _quant_pt, _reference, _routing,
)  # fmt: skip

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] not in (8, 9),
    reason="records MoE runs on SM80 to SM90",
)

F4 = QuantFormat.NVFP4, QuantFormat.MXFP4
PAIRS = [(F4[0], F4[0])] + [
    (F4[1], a) for a in (QuantFormat.MXFP8, QuantFormat.BF16, F4[1])
]
# (pair, per token, activation): every pair under SwiGLU, NVFP4 per token, PowLU's typed m
# and limit
CASES = [(p, False, SwiGLU()) for p in PAIRS]
CASES += [(PAIRS[0], True, SwiGLU()), (PAIRS[0], False, PowLU(limit=4.0))]
# NVFP4 x BF16 under unpacked BF16 routing weights, and a view's limit in output units
CASES += [((F4[0], QuantFormat.BF16), False, SwiGLU())]
CASES += [(PAIRS[0], False, SwiGLUStep(limit=1.0))]


def _act(g, u, a):
    """SwiGLU, SwiGLUStep or PowLU a of the gate g and linear u. PowLU: g^(m / (sqrt(g) + 1))
    (g > 0, else g) times sigmoid(g); PowLU's and SwiGLUStep's branches clamped to the limit."""
    if isinstance(a, SwiGLUStep):
        return (g * g.sigmoid()).clamp(max=a.limit) * u.clamp(-a.limit, a.limit)
    if not isinstance(a, PowLU):
        return g * g.sigmoid() * u
    gp = g.clamp(min=0)
    y = torch.where(g > 0, gp ** (a.m / (gp.sqrt() + 1)), g) * g.sigmoid()
    return y.clamp(max=a.limit) * u.clamp(-a.limit, a.limit)


def _mxfp4(a):
    """The dequantized MXFP4 a of the TRT-LLM quantizer."""
    codes, sf = mxfp4_quantize(a.bfloat16(), sfLayout=SfLayout.layout_linear)
    sf = torch.exp2(sf.view(torch.uint8).reshape(a.shape[0], -1).float() - 127)
    return _e2m1(codes.view(torch.uint8)) * sf.repeat_interleave(32, -1)


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda c: f"{c[0][0].name}x{c[0][1].name}{'-pt' * c[1]}-{type(c[2]).__name__}",
)
@pytest.mark.parametrize("tokens", [1, 37])
def test_records_unified(case, tokens):
    (pair, per_token, activation), pt = case, None
    torch.manual_seed(0)
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    xs, sizes = None, dict(num_local_experts=E, hidden_size=H, intermediate_size=I)
    sizes["activation"], mode = activation, RoutingInputMode.PackedPrecomputed
    if pair[0] is QuantFormat.NVFP4:
        w4a16 = pair[1] is QuantFormat.BF16
        m, key = _model(True), "records_nvfp4"
        backend = RecordsNvfp4Bf16Config() if w4a16 else RecordsNvfp4Config()
        sf1, sf2 = (m[k].view(torch.float8_e4m3fn) for k in ("w1_sf", "w2_sf"))
        view = backend.prepare_weights(
            m["w1"], sf1, m["w1_g"], m["w2"], sf2, m["w2_g"], **sizes
        )
        if per_token:  # NVFP4 codes under token scales, the input global scales unread
            (x, xs, pt, xr), gemm2_input = _quant_pt(x), lambda a: _quant_pt(a)[3]
        elif w4a16:  # the routing weights in BF16, as TRT-LLM's unpacked
            xr, gemm2_input = x.float(), lambda a: a.float()
            weights, mode = weights.bfloat16(), RoutingInputMode.UnpackedPrecomputed
        else:  # BF16 input under w1_input_global_scale
            a1 = 448 * 6 / x.float().abs().amax().view(1)
            a2 = torch.tensor([448 * 6 / 128.0], device="cuda")
            view.update(w1_input_global_scale=a1, w2_input_global_scale=a2)
            xr, gemm2_input = _quant(x, a1)[2], lambda a: _quant(a, a2)[2]
    else:
        m, backend, key = _mx_model(True), RecordsMxfp4Config(), "records_mxfp4"
        view = backend.prepare_weights(
            m["w1"], m["w1_sf"], m["w2"], m["w2_sf"], **sizes
        )
        xr, gemm2_input = x.float(), lambda a: a.float()
        if pair[1] is QuantFormat.MXFP8:
            codes, xs, xr = _mxfp8(x)
            x, gemm2_input = codes.view(torch.float8_e4m3fn), lambda a: _mxfp8(a)[2]
        elif pair[1] is QuantFormat.MXFP4:  # quantized inside, GEMM2's input too
            xr, gemm2_input = _mxfp4(x), _mxfp4
    pack = MoEWeightPack()
    pack.prepare_for(key, view)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=K),
        quant=QuantConfig(
            weight=pair[0], activation=pair[1], per_token_scale=per_token
        ),
        experts=ExpertConfig(intermediate_size=I),
        activation=activation,
        backend=BackendOptions(candidates=(backend,)),
        execution=ExecutionConfig(tune_max_num_tokens=tokens),
    )
    act = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=xs,
        topk_ids=ids,
        topk_weights=weights,
        per_token_scale=pt,
        routing_input_mode=mode,
    )
    layer = MoELayer(config, device=torch.device("cuda"))
    out = layer(act, pack)
    # Tactics take GEMM1's output in FP32 or BF16 into GEMM2's quantized input, and the
    # codes that this flips cost up to 7e-2 under MXFP4's E8M0 scales: the closer reference.
    refs = [
        _reference(
            m, xr, ids, weights, gemm2_input, lambda g, u: _act(g, u, activation), b
        )
        for b in (True, False)
    ]
    _close(out, min(refs, key=lambda r: (out.float() - r).norm()))
    # The default backends hold the records ones; the pack, their view only
    default = MoELayer(replace(config, backend=_DEFAULT_BACKEND), torch.device("cuda"))
    torch.testing.assert_close(default(act, pack), out, atol=0, rtol=0)
    if key == "records_nvfp4" and not per_token:  # an inference tensor updated in place
        with torch.inference_mode():
            view["w1_global_scale"] = g1 = view["w1_global_scale"].clone()
            first = layer(act, pack)
            g1.mul_(2)
            fresh = MoELayer(config, device=torch.device("cuda"))(act, pack)
            assert not torch.equal(fresh, first)
            torch.testing.assert_close(layer(act, pack), fresh, atol=0, rtol=0)
    if key == "records_nvfp4":  # a parameter the activation reads, off the device
        step = isinstance(activation, SwiGLUStep)
        view["gemm1_clamp_limit" if step else "gemm1_alpha"] = torch.ones(E)
        with pytest.raises(ValueError, match="input's device"):
            layer(act, pack)


def test_records_unified_rank0(monkeypatch):
    """EP rank 0 from logits at row tile 8 (where the records keep the slots): the full layer
    on the same routing with the other rank's experts weighted 0."""
    m, tokens, half = _model(True), 37, E // 2
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    logits = torch.randn(tokens, E, device="cuda")
    tile8 = W4A16_SM80_TACTICS[0]
    monkeypatch.setattr(AutoTuner, "choose_one", lambda self, *a, **k: (None, tile8))
    sf1, sf2 = (m[k].view(torch.float8_e4m3fn) for k in ("w1_sf", "w2_sf"))
    a1 = 448 * 6 / x.float().abs().amax().view(1)
    a2 = torch.tensor([448 * 6 / 128.0], device="cuda")

    def layer(n, **experts):  # over the leading n experts
        view = RecordsNvfp4Config().prepare_weights(
            m["w1"][:n], sf1[:n], m["w1_g"][:n], m["w2"][:n], sf2[:n], m["w2_g"][:n],
            num_local_experts=n, hidden_size=H, intermediate_size=I, activation=SwiGLU(),
        )  # fmt: skip
        view.update(w1_input_global_scale=a1, w2_input_global_scale=a2)
        pack = MoEWeightPack()
        pack.prepare_for("records_nvfp4", view)
        config = MoEConfig(
            routing=RoutingConfig(
                num_experts=E, top_k=K, method=RoutingMethodType.Renormalize
            ),
            quant=QuantConfig(weight=F4[0], activation=F4[0]),
            experts=ExpertConfig(intermediate_size=I, **experts),
            activation=SwiGLU(),
            backend=BackendOptions(candidates=(RecordsNvfp4Config(),)),
            execution=ExecutionConfig(tune_max_num_tokens=tokens),
        )
        moe = MoELayer(config, device=torch.device("cuda"))
        return lambda act: moe(act, pack)

    def from_logits(lg):
        return MoEActivationPack(
            hidden_states_q=x,
            hidden_states_scale=None,
            routing_logits=lg,
            routing_input_mode=RoutingInputMode.FromLogits,
        )

    rank0 = layer(half, local_num_experts=half)
    rank0(from_logits(logits.roll(half, 1)))  # GEMM2's rows of the other rank's entries
    out = rank0(from_logits(logits))
    ids, w = _trtllm_routing_sm80(
        logits,
        None,
        RoutingMethodType.Renormalize.value,
        K,
        None,
        None,
        None,
        True,
        None,
    )
    routed = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        topk_ids=ids,
        topk_weights=w * (ids < half),
    )
    torch.testing.assert_close(out, layer(E)(routed), atol=0, rtol=0)


def test_records_unified_graph():
    """A captured graph reads its view's scales on every replay: an eager call on another view in
    between doesn't reach it, and a scale updated in place does, as on SM100."""
    m, tokens = _model(True), 37
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    sf1, sf2 = (m[k].view(torch.float8_e4m3fn) for k in ("w1_sf", "w2_sf"))
    a1 = 448 * 6 / x.float().abs().amax().view(1)
    a2 = torch.tensor([448 * 6 / 128.0], device="cuda")
    packs = [MoEWeightPack(), MoEWeightPack()]
    for pack, g1 in zip(packs, (m["w1_g"], m["w1_g"] * 2), strict=True):
        view = RecordsNvfp4Config().prepare_weights(
            m["w1"], sf1, g1, m["w2"], sf2, m["w2_g"],
            num_local_experts=E, hidden_size=H, intermediate_size=I, activation=SwiGLU(),
        )  # fmt: skip
        view.update(w1_input_global_scale=a1, w2_input_global_scale=a2)
        pack.prepare_for("records_nvfp4", view)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=K),
        quant=QuantConfig(weight=F4[0], activation=F4[0]),
        experts=ExpertConfig(intermediate_size=I),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(RecordsNvfp4Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=tokens),
    )
    layer = MoELayer(config, device=torch.device("cuda"))
    act = MoEActivationPack(
        hidden_states_q=x, hidden_states_scale=None, topk_ids=ids, topk_weights=weights
    )
    ref = layer(act, packs[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer(act, packs[0])
    layer(act, packs[1])  # a call on the second view, then NaN-filled tensors
    junk = [torch.full((E,), float("nan"), device="cuda") for _ in range(16)]
    graph.replay()
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    del junk
    packs[0].get_view("records_nvfp4")["w2_global_scale"].mul_(2)  # the replay reads it
    graph.replay()
    torch.testing.assert_close(out, layer(act, packs[0]), atol=0, rtol=0)
    assert not torch.equal(out, ref)
    # Records of hidden size 512 take as many bytes as of 480: the view's size tells them apart
    narrow = MoEActivationPack(
        hidden_states_q=x[:, : H - 32].contiguous(), hidden_states_scale=None,
        topk_ids=ids, topk_weights=weights,
    )  # fmt: skip
    with pytest.raises(ValueError, match="hidden_size"):
        layer(narrow, packs[0])
