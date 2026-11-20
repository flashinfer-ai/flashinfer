"""The unified MoELayer on the records backends (SM80 to SM90): each FP4 pair, NVFP4 per
token and an activation beyond TRT-LLM's against the dequantized reference of
tests/moe/test_records_nvfp4_moe.py."""

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
    SwiGLU,
)
from flashinfer.fused_moe.api import RecordsMxfp4Config, RecordsNvfp4Config
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


def _act(g, u, a):
    """SwiGLU or PowLU a of the gate g and linear u. PowLU: g^(m / (sqrt(g) + 1)) (g > 0,
    else g) times sigmoid(g), both branches clamped to the limit."""
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
    x = torch.randn(tokens, H, device="cuda").bfloat16()
    ids, weights = _routing(tokens)
    xs, sizes = None, dict(num_local_experts=E, hidden_size=H, intermediate_size=I)
    sizes["activation"] = activation
    if pair[0] is QuantFormat.NVFP4:
        m, backend, key = _model(True), RecordsNvfp4Config(), "records_nvfp4"
        sf1, sf2 = (m[k].view(torch.float8_e4m3fn) for k in ("w1_sf", "w2_sf"))
        view = backend.prepare_weights(
            m["w1"], sf1, m["w1_g"], m["w2"], sf2, m["w2_g"], **sizes
        )
        if per_token:  # NVFP4 codes under token scales, the input global scales unread
            (x, xs, pt, xr), gemm2_input = _quant_pt(x), lambda a: _quant_pt(a)[3]
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
    )
    out = MoELayer(config, device=torch.device("cuda"))(act, pack)
    ref = _reference(
        m, xr, ids, weights, gemm2_input, lambda g, u: _act(g, u, activation)
    )
    _close(out, ref)
