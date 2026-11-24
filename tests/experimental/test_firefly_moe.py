"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The experimental Firefly backend (``flashinfer.experimental.firefly``) through
``MoELayer`` with ``FireflyNvfp4Config``: NVFP4 MoE outputs (per-token and
per-tensor activations) against an FP32 torch reference that dequantizes the
same weights and inputs and rounds the intermediate as the kernels do (BF16,
then NVFP4: block scales only, or a per-token scale as well on the M <= 4 path
of 512 <= I <= 768). The reference
takes its reciprocals from the hardware rcp.approx.ftz, as the kernels do (a
correctly rounded division moves about 1e-5 of the values across an E2M1
boundary, enough to double the error), so the relative RMSE is about 0.0025
(fused finalize: BF16 additions in any order, about 0.004).
"""

from __future__ import annotations

import warnings

import pytest
import torch
import triton
import triton.language as tl

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    FireflyNvfp4Config,
    GeGLU,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    SwiGLU,
)
from flashinfer.utils import get_compute_capability

pytest.importorskip("cutlass")


def _sm100_or_sm103() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = get_compute_capability(torch.device("cuda"))
    return major * 10 + minor in (100, 103)


pytestmark = pytest.mark.skipif(
    not _sm100_or_sm103(), reason="Firefly needs SM100 / SM103"
)

E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Round to E2M1 (nearest, ties to even, saturating at 6), keeping sign."""
    levels = torch.tensor(E2M1_LEVELS, device=x.device)
    bounds = (levels[1:] + levels[:-1]) / 2
    mag = x.abs().clamp_max(6.0).contiguous()
    code = torch.bucketize(mag, bounds)
    tie = mag == bounds[code.clamp_max(len(bounds) - 1)]
    code = code + (tie & (code % 2 == 1)).long()
    return levels[code] * torch.sign(x)


@triton.jit
def _rcp_kernel(src, dst, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src + i, mask=i < n, other=1.0)
    r = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [v],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(dst + i, r, mask=i < n)


def _rcp(t: torch.Tensor) -> torch.Tensor:
    """Elementwise rcp.approx.ftz.f32, the reciprocal the kernels use."""
    t = t.contiguous().float()
    out = torch.empty_like(t)
    if t.numel():
        _rcp_kernel[(triton.cdiv(t.numel(), 1024),)](t, out, t.numel(), BLOCK=1024)
    return out


def _quantize_rows(x: torch.Tensor, per_token: bool):
    """NVFP4 of rows [M, K] as the kernels round: row scale gs = amax * (1 /
    2688) with ``per_token`` (else 1), encode = rcp(gs), block scale sf =
    e4m3(encode * (bmax * (1 / 6))), values e2m1(x * rcp(sf * gs)). Returns the
    dequantized rows, the E2M1 values, the block scales and the row scales."""
    xf = x.float()
    m, k = xf.shape
    amax = xf.abs().amax(1)
    if per_token:
        gs = amax * torch.tensor(1.0 / 2688.0, device=x.device)
        enc = torch.where(amax > 0, _rcp(gs), torch.zeros_like(gs))
    else:
        gs = enc = torch.ones_like(amax)
    blocks = xf.reshape(m, k // 16, 16)
    bmax = blocks.abs().amax(-1)
    sixth = torch.tensor(1.0 / 6.0, device=x.device)
    sf = (enc[:, None] * (bmax * sixth)).to(torch.float8_e4m3fn)
    sff = sf.float()
    inv = torch.where(sff > 0, _rcp(sff * gs[:, None]), torch.zeros_like(sff))
    vals = _e2m1_round(blocks * inv[..., None])
    deq = (vals * sff[..., None] * gs[:, None, None]).reshape(m, k)
    return deq, vals.reshape(m, k), sf, gs


def _pack_e2m1(vals: torch.Tensor) -> torch.Tensor:
    """E2M1 values [M, K] -> uint8 [M, K/2], the even element in the low nibble."""
    levels = torch.tensor(E2M1_LEVELS, device=vals.device)
    code = (vals.abs()[..., None] == levels).float().argmax(-1)
    code = code + 8 * (vals < 0).long()
    return (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)


def _quantize_input(x: torch.Tensor, per_token: bool = True):
    """The activation pack's tensors: per-token (row scales amax / 2688) or
    per-tensor (unit global scale; the row scales returned are all 1)."""
    deq, vals, sf, gs = _quantize_rows(x, per_token=per_token)
    return deq, _pack_e2m1(vals), sf, gs.float()


def _dequant_experts(w: torch.Tensor) -> torch.Tensor:
    """BF16 experts [E, N, K] quantized as the weight view does (one global
    scale per expert from its amax, block scales e4m3(bmax / 6 * scale)),
    dequantized to FP32."""
    from flashinfer.experimental.firefly.weights import quantize_expert_weights

    q, sf, gscale = quantize_expert_weights(w)
    lo, hi = (q & 0xF).long(), (q >> 4).long()
    codes = torch.stack([lo, hi], -1).reshape(*q.shape[:-1], -1)
    levels = torch.tensor(E2M1_LEVELS, device=w.device)
    vals = levels[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    e, n, k = vals.shape
    s = sf.view(torch.float8_e4m3fn).float().reshape(e, n, k // 16)
    return (vals.reshape(e, n, k // 16, 16) * s[..., None]).reshape(
        e, n, k
    ) * gscale.reshape(e, 1, 1)


def _activation_ref(act, h: torch.Tensor, inter: int) -> torch.Tensor:
    from flashinfer.experimental.firefly.activation import as_activation

    if not act.is_gated:
        return torch.relu(h) ** 2
    up, gate = h[:, :inter], h[:, inter:]
    return as_activation(act).torch_ref(gate, up)


def _reference(x_deq, ids, wts, w1, w2, act, inter, mid_per_token, offset=0):
    """FP32 output: per route, act(x @ w1_e^T) rounded to BF16 then NVFP4,
    times w2_e^T, times the route weight; ids outside the local experts add
    nothing."""
    out = torch.zeros(x_deq.shape[0], w2.shape[1], device=x_deq.device)
    local = w1.shape[0]
    for e in ids.unique().tolist():
        le = e - offset
        if not 0 <= le < local:
            continue
        token, slot = torch.where(ids == e)
        mid = _activation_ref(act, x_deq[token] @ w1[le].T, inter).bfloat16()
        mid_q = _quantize_rows(mid, per_token=mid_per_token)[0]
        out.index_add_(0, token, (mid_q @ w2[le].T) * wts[token, slot, None].float())
    return out


def _rel_rmse(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt().item()


def _setup(num_experts, top_k, hidden, inter, act, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = 2 * inter if act.is_gated else inter
    w1 = (
        torch.randn((num_experts, rows, hidden), generator=g, device="cuda") * 0.02
    ).bfloat16()
    w2 = (
        torch.randn((num_experts, hidden, inter), generator=g, device="cuda") * 0.02
    ).bfloat16()
    view = FireflyNvfp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=inter,
        activation=act,
    )
    pack = MoEWeightPack()
    pack.prepare_for("firefly_nvfp4", view)
    return g, pack, _dequant_experts(w1), _dequant_experts(w2)


def _layer(
    num_experts, top_k, inter, act, max_tokens, fused=False, per_token=True, **experts
):
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(
            weight=QuantFormat.NVFP4,
            activation=QuantFormat.NVFP4,
            per_token_scale=per_token,
        ),
        experts=ExpertConfig(intermediate_size=inter, **experts),
        activation=act,
        finalize=MoEFinalizeConfig(use_fused_finalize=fused),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
        backend=BackendOptions(candidates=(FireflyNvfp4Config(),)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MoELayer(config)


def _inputs(
    g, m, hidden, num_experts, top_k, weights_dtype=torch.float32, per_token=True
):
    """Dequantized input, q, sf, gs (None for per-tensor), ids, weights."""
    x = torch.randn((m, hidden), generator=g, device="cuda").bfloat16()
    x_deq, q, sf, gs = _quantize_input(x, per_token)
    logits = torch.randn((m, num_experts), generator=g, device="cuda")
    wts, ids = torch.sigmoid(logits).topk(top_k, dim=-1)
    return x_deq, q, sf, gs if per_token else None, ids.int(), wts.to(weights_dtype)


def _mid_per_token(layer, m, hidden, weights_dtype=torch.float32) -> bool:
    moe = layer.runners[0]._moes[hidden]
    return not moe.plan(m, weights_dtype).mid_unit


CASES = [
    # (num_experts, top_k, hidden, inter, activation, batch sizes)
    pytest.param(32, 8, 1024, 768, SwiGLU(), (1, 3, 64, 1024, 2048), id="swiglu"),
    pytest.param(16, 4, 1024, 640, ReLU2(), (2, 200, 4096), id="relu2-ntail"),
    pytest.param(
        16,
        2,
        2048,
        1024,
        SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
        (1, 32),
        id="swiglu-oa",
    ),
    pytest.param(16, 4, 1024, 512, GeGLU(), (4, 96), id="geglu"),
]


PER_TOKEN = [
    pytest.param(True, id="per-token"),
    pytest.param(False, id="per-tensor"),
]


@pytest.mark.parametrize("per_token", PER_TOKEN)
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("num_experts,top_k,hidden,inter,act,ms", CASES)
def test_firefly_matches_reference(
    num_experts, top_k, hidden, inter, act, ms, fused, per_token
):
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    layer = _layer(
        num_experts, top_k, inter, act, max(ms), fused=fused, per_token=per_token
    )
    for m in ms:
        x_deq, q, sf, gs, ids, wts = _inputs(
            g, m, hidden, num_experts, top_k, per_token=per_token
        )
        act_pack = MoEActivationPack(
            hidden_states_q=q,
            hidden_states_scale=sf,
            topk_ids=ids,
            topk_weights=wts,
            per_token_scale=gs,
        )
        out = layer(act_pack, pack)
        assert layer.winner_backend == "firefly_nvfp4"
        assert out.shape == (m, hidden) and out.dtype == torch.bfloat16
        ref = _reference(
            x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, m, hidden)
        )
        err = _rel_rmse(out, ref)
        assert torch.isfinite(out).all()
        assert err < (0.0065 if fused else 0.005), f"M={m}: relative RMSE {err}"
        if not fused:
            assert torch.equal(out, layer(act_pack, pack)), f"M={m}: not deterministic"


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
def test_firefly_configs_sharing_a_process(fused):
    """Layers that differ only in E or top_k, used alternately in one process, each
    get kernels compiled for their own configuration (the compiled kernels are
    cached per process)."""
    hidden, inter, act, ms = 1024, 768, SwiGLU(), (4, 64, 768)
    configs = [(64, 8), (128, 8), (64, 6)]
    setups = {c: _setup(c[0], c[1], hidden, inter, act) for c in configs}
    layers = {c: _layer(c[0], c[1], inter, act, max(ms), fused=fused) for c in configs}
    for m in ms:
        for (num_experts, top_k), (g, pack, w1, w2) in setups.items():
            layer = layers[(num_experts, top_k)]
            x_deq, q, sf, gs, ids, wts = _inputs(g, m, hidden, num_experts, top_k)
            out = layer(
                MoEActivationPack(
                    hidden_states_q=q,
                    hidden_states_scale=sf,
                    topk_ids=ids,
                    topk_weights=wts,
                    per_token_scale=gs,
                ),
                pack,
            )
            ref = _reference(
                x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, m, hidden)
            )
            err = _rel_rmse(out, ref)
            assert err < (0.0065 if fused else 0.005), (
                f"E={num_experts} top_k={top_k} M={m}: relative RMSE {err}"
            )


@pytest.mark.parametrize(
    "num_experts,top_k,hidden,inter,act,ms",
    [
        pytest.param(
            32, 8, 1024, 768, SwiGLU(), (1, 3, 16, 64, 512, 2048), id="swiglu"
        ),
        pytest.param(16, 4, 1024, 640, ReLU2(), (2, 200, 4096), id="relu2-ntail"),
    ],
)
def test_firefly_per_tensor_equals_unit_row_scales(
    num_experts, top_k, hidden, inter, act, ms
):
    """Per-tensor activations give the per-token kernels' output for the same
    values with every row scale 1, bit for bit (no row scale is loaded or
    multiplied; 1 * x is exact)."""
    g, pack, _, _ = _setup(num_experts, top_k, hidden, inter, act)
    per_tensor = _layer(num_experts, top_k, inter, act, max(ms), per_token=False)
    per_token = _layer(num_experts, top_k, inter, act, max(ms))
    for m in ms:
        _, q, sf, _, ids, wts = _inputs(
            g, m, hidden, num_experts, top_k, per_token=False
        )
        ones = torch.ones(m, dtype=torch.float32, device="cuda")
        out = per_tensor(
            MoEActivationPack(
                hidden_states_q=q,
                hidden_states_scale=sf,
                topk_ids=ids,
                topk_weights=wts,
            ),
            pack,
        )
        ref = per_token(
            MoEActivationPack(
                hidden_states_q=q,
                hidden_states_scale=sf,
                topk_ids=ids,
                topk_weights=wts,
                per_token_scale=ones,
            ),
            pack,
        )
        assert torch.equal(out, ref), f"M={m}"


@pytest.mark.parametrize("per_token", PER_TOKEN)
def test_firefly_skipped_routes_and_expert_parallelism(per_token):
    """Ids outside this rank's experts (negative, other ranks', >= E) add
    nothing; BF16 route weights are read as given; the unpacked mode works."""
    num_experts, top_k, hidden, inter, act = 32, 4, 1024, 512, SwiGLU()
    offset, local = 8, 16
    g, pack, w1, w2 = _setup(local, top_k, hidden, inter, act)
    layer = _layer(
        num_experts,
        top_k,
        inter,
        act,
        256,
        per_token=per_token,
        local_expert_offset=offset,
        local_num_experts=local,
    )
    for m in (3, 256):
        x_deq, q, sf, gs, ids, wts = _inputs(
            g,
            m,
            hidden,
            num_experts,
            top_k,
            weights_dtype=torch.bfloat16,
            per_token=per_token,
        )
        ids[::5, 0] = -1
        ids[1::7, 1] = num_experts + 3
        act_pack = MoEActivationPack(
            hidden_states_q=q,
            hidden_states_scale=sf,
            topk_ids=ids,
            topk_weights=wts,
            per_token_scale=gs,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        out = layer(act_pack, pack)
        ref = _reference(
            x_deq,
            ids,
            wts,
            w1,
            w2,
            act,
            inter,
            _mid_per_token(layer, m, hidden, torch.bfloat16),
            offset=offset,
        )
        assert _rel_rmse(out, ref) < 0.005


@pytest.mark.parametrize("per_token", PER_TOKEN)
def test_firefly_cuda_graph_replay(per_token):
    """A graph captured after the first call replays new inputs correctly."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    for m in (2, 48):
        layer = _layer(num_experts, top_k, inter, act, m, per_token=per_token)
        x_deq, q, sf, gs, ids, wts = _inputs(
            g, m, hidden, num_experts, top_k, per_token=per_token
        )
        act_pack = MoEActivationPack(
            hidden_states_q=q,
            hidden_states_scale=sf,
            topk_ids=ids,
            topk_weights=wts,
            per_token_scale=gs,
        )
        layer(act_pack, pack)  # plans and compiles outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = layer(act_pack, pack)
        for _ in range(2):
            x_deq, q2, sf2, gs2, ids2, wts2 = _inputs(
                g, m, hidden, num_experts, top_k, per_token=per_token
            )
            for dst, src in ((q, q2), (sf, sf2), (gs, gs2), (ids, ids2), (wts, wts2)):
                if dst is not None:
                    dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            ref = _reference(
                x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, m, hidden)
            )
            assert _rel_rmse(out, ref) < 0.005


def test_firefly_config():
    from flashinfer.fused_moe import api

    assert eval(repr(FireflyNvfp4Config()), vars(api)) == FireflyNvfp4Config()
    assert FireflyNvfp4Config.supported(100) and FireflyNvfp4Config.supported(103)
    assert not FireflyNvfp4Config.supported(90)
    assert not FireflyNvfp4Config.supported(120)


@pytest.mark.parametrize(
    "quant_kwargs",
    [
        pytest.param(
            {"per_token_scale": True, "swizzled_scale_factors": True}, id="swizzled"
        ),
        pytest.param(
            {"per_token_scale": False, "swizzled_scale_factors": True},
            id="swizzled-per-tensor",
        ),
    ],
)
def test_firefly_rejects_unsupported(quant_kwargs):
    config = MoEConfig(
        routing=RoutingConfig(num_experts=16, top_k=4),
        quant=QuantConfig(
            weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4, **quant_kwargs
        ),
        experts=ExpertConfig(intermediate_size=512),
        backend=BackendOptions(candidates=(FireflyNvfp4Config(),)),
    )
    with pytest.raises(RuntimeError):
        MoELayer(config)


def test_firefly_per_tensor_contract():
    """per_token_scale=None is per-tensor; a per-tensor layer rejects a pack
    with per-token scales, a per-token one a pack without them."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    layer = _layer(num_experts, top_k, inter, act, 8, per_token=None)
    x_deq, q, sf, gs, ids, wts = _inputs(
        g, 8, hidden, num_experts, top_k, per_token=False
    )
    out = layer(
        MoEActivationPack(
            hidden_states_q=q, hidden_states_scale=sf, topk_ids=ids, topk_weights=wts
        ),
        pack,
    )
    ref = _reference(
        x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, 8, hidden)
    )
    assert _rel_rmse(out, ref) < 0.005
    ones = torch.ones(8, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        layer(
            MoEActivationPack(
                hidden_states_q=q,
                hidden_states_scale=sf,
                topk_ids=ids,
                topk_weights=wts,
                per_token_scale=ones,
            ),
            pack,
        )
    with pytest.raises(ValueError):
        _layer(num_experts, top_k, inter, act, 8)(
            MoEActivationPack(
                hidden_states_q=q,
                hidden_states_scale=sf,
                topk_ids=ids,
                topk_weights=wts,
            ),
            pack,
        )


def test_firefly_rejects_4over6():
    from flashinfer import NVFP44Over6Config

    config = MoEConfig(
        routing=RoutingConfig(num_experts=16, top_k=4),
        quant=QuantConfig(
            weight=QuantFormat.NVFP4,
            activation=QuantFormat.NVFP4,
            per_token_scale=True,
            nvfp4_4over6=NVFP44Over6Config(),
        ),
        experts=ExpertConfig(intermediate_size=512),
        backend=BackendOptions(candidates=(FireflyNvfp4Config(),)),
    )
    with pytest.raises(RuntimeError):
        MoELayer(config)
    w = torch.zeros((16, 1024, 1024), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError):
        FireflyNvfp4Config.prepare_weights(
            w,
            w[:, :, :512].contiguous(),
            num_local_experts=16,
            hidden_size=1024,
            intermediate_size=512,
            nvfp4_4over6=NVFP44Over6Config(),
        )


def test_firefly_index_less_device():
    """MoELayer(device=torch.device("cuda")) (no index) runs on the current device."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(
            weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4, per_token_scale=True
        ),
        experts=ExpertConfig(intermediate_size=inter),
        activation=act,
        execution=ExecutionConfig(tune_max_num_tokens=8),
        backend=BackendOptions(candidates=(FireflyNvfp4Config(),)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = MoELayer(config, device=torch.device("cuda"))
    x_deq, q, sf, gs, ids, wts = _inputs(g, 8, hidden, num_experts, top_k)
    out = layer(
        MoEActivationPack(
            hidden_states_q=q,
            hidden_states_scale=sf,
            topk_ids=ids,
            topk_weights=wts,
            per_token_scale=gs,
        ),
        pack,
    )
    ref = _reference(
        x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, 8, hidden)
    )
    assert _rel_rmse(out, ref) < 0.005


def _act_pack(q, sf, gs, ids, wts):
    return MoEActivationPack(
        hidden_states_q=q,
        hidden_states_scale=sf,
        topk_ids=ids,
        topk_weights=wts,
        per_token_scale=gs,
    )


def test_firefly_rejects_inputs_off_device():
    """Input tensors on the host (or on another GPU) are rejected with
    ValueError before any kernel is launched; the layer stays usable."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    layer = _layer(num_experts, top_k, inter, act, 8)
    x_deq, q, sf, gs, ids, wts = _inputs(g, 8, hidden, num_experts, top_k)
    host = [t.cpu() for t in (q, sf, gs, ids, wts)]
    with pytest.raises(ValueError):
        layer(_act_pack(*host), pack)
    out = layer(_act_pack(q, sf, gs, ids, wts), pack)
    moe = layer.runners[0]._moes[hidden]
    view = pack.get_view("firefly_nvfp4")
    for i in range(5):  # one tensor off the device, the direct CuteMoE call
        args = [q, sf, gs, ids, wts]
        args[i] = args[i].cpu()
        with pytest.raises(ValueError):
            moe(args[0], args[1], args[2], view, topk_ids=args[3], topk_weights=args[4])
    with pytest.raises(ValueError):
        moe(q, sf, gs, view, topk_ids=ids, topk_weights=wts, out=out.cpu())
    if torch.cuda.device_count() > 1:
        other = torch.device("cuda", 1 - q.device.index)
        with pytest.raises(ValueError):
            layer(_act_pack(*(t.to(other) for t in (q, sf, gs, ids, wts))), pack)
    torch.cuda.synchronize()  # no asynchronous fault left behind
    ref = _reference(
        x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, 8, hidden)
    )
    assert _rel_rmse(layer(_act_pack(q, sf, gs, ids, wts), pack), ref) < 0.005


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_firefly_device_other_than_current():
    """MoELayer(device=cuda:1) runs on cuda:1 while cuda:0 is current (eager),
    and a CUDA graph captured on cuda:1 replays new inputs correctly."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    dev = torch.device("cuda", 1)
    with torch.cuda.device(dev):
        g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(
            weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4, per_token_scale=True
        ),
        experts=ExpertConfig(intermediate_size=inter),
        activation=act,
        execution=ExecutionConfig(tune_max_num_tokens=2048),
        backend=BackendOptions(candidates=(FireflyNvfp4Config(),)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = MoELayer(config, device=dev)
    for m in (2, 48, 2048):  # direct, short and long batches
        with torch.cuda.device(dev):
            x_deq, q, sf, gs, ids, wts = _inputs(g, m, hidden, num_experts, top_k)
        with torch.cuda.device(0):
            out = layer(_act_pack(q, sf, gs, ids, wts), pack)
        assert out.device == dev
        with torch.cuda.device(
            dev
        ):  # the reference's Triton kernel on the tensors' device
            ref = _reference(
                x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, m, hidden)
            )
            assert _rel_rmse(out, ref) < 0.005, f"M={m}"
    m = 48
    with torch.cuda.device(dev):
        x_deq, q, sf, gs, ids, wts = _inputs(g, m, hidden, num_experts, top_k)
        graph = torch.cuda.CUDAGraph()
        # a capture stream on this device: torch.cuda.graph's default capture stream is shared by all graphs and
        # lives on the device of the first capture in the process
        with torch.cuda.graph(graph, stream=torch.cuda.Stream(dev)):
            out = layer(_act_pack(q, sf, gs, ids, wts), pack)
        for _ in range(2):
            x_deq, q2, sf2, gs2, ids2, wts2 = _inputs(g, m, hidden, num_experts, top_k)
            for dst, src in ((q, q2), (sf, sf2), (gs, gs2), (ids, ids2), (wts, wts2)):
                dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize(dev)
            ref = _reference(
                x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, m, hidden)
            )
            assert _rel_rmse(out, ref) < 0.005


@pytest.mark.parametrize("local", [4, 1])
def test_firefly_expert_parallel_top_k_above_local_experts(local):
    """Expert parallelism with more route slots (top_k) than local experts:
    the direct, short and long batches."""
    num_experts, top_k, hidden, inter, act = 32, 8, 1024, 512, SwiGLU()
    offset = 8
    g, pack, w1, w2 = _setup(local, top_k, hidden, inter, act)
    layer = _layer(
        num_experts,
        top_k,
        inter,
        act,
        1024,
        local_expert_offset=offset,
        local_num_experts=local,
    )
    for m in (2, 8, 64, 1024):
        x_deq, q, sf, gs, ids, wts = _inputs(g, m, hidden, num_experts, top_k)
        out = layer(_act_pack(q, sf, gs, ids, wts), pack)
        ref = _reference(
            x_deq,
            ids,
            wts,
            w1,
            w2,
            act,
            inter,
            _mid_per_token(layer, m, hidden),
            offset=offset,
        )
        if (
            ref.abs().max() == 0
        ):  # (one local expert: possibly no route of the batch is this rank's)
            assert torch.count_nonzero(out) == 0, f"M={m}"
        else:
            assert _rel_rmse(out, ref) < 0.005, f"M={m}"


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
def test_firefly_relu2_n_tail_wide_hidden(fused):
    """ReLU2 with 256 not dividing I at H = 4096: the long batches' gate/up
    takes the 1-CTA kernel (its N tail) where the CTA-pair rule would apply."""
    num_experts, top_k, hidden, inter, act = 8, 8, 4096, 640, ReLU2()
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    layer = _layer(num_experts, top_k, inter, act, 1024, fused=fused)
    for m in (64, 1024):
        x_deq, q, sf, gs, ids, wts = _inputs(g, m, hidden, num_experts, top_k)
        out = layer(_act_pack(q, sf, gs, ids, wts), pack)
        ref = _reference(
            x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, m, hidden)
        )
        assert torch.isfinite(out).all()
        assert _rel_rmse(out, ref) < (0.0065 if fused else 0.005), f"M={m}"


def test_firefly_situ_rejects_invalid_scales():
    """SiTU scales must be positive and finite (linear_scale None: no clamp),
    checked when the activation is built, not at the first kernel trace."""
    from flashinfer.experimental.firefly.activation import parse

    for spec in (
        "situ:gate_scale=0",
        "situ:gate_scale=-4",
        "situ:gate_scale=inf",
        "situ:linear_scale=0",
        "situ:linear_scale=nan",
    ):
        with pytest.raises(ValueError):
            parse(spec)
    assert parse("situ:linear_scale=none").linear_scale is None


def test_firefly_skips_unsupported_hidden_size():
    """A pack whose hidden size Firefly cannot run (H % 256 != 0) drops the
    candidate before packing: with no other candidate the layer reports that
    no backend accepts the inputs."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    _, pack, _, _ = _setup(num_experts, top_k, hidden, inter, act)
    layer = _layer(num_experts, top_k, inter, act, 8)
    m, h = 8, 1152
    q = torch.zeros((m, h // 2), dtype=torch.uint8, device="cuda")
    sf = torch.zeros((m, h // 16), dtype=torch.float8_e4m3fn, device="cuda")
    gs = torch.ones(m, device="cuda")
    ids = torch.arange(top_k, dtype=torch.int32, device="cuda").repeat(m, 1)
    wts = torch.full((m, top_k), 0.25, device="cuda")
    with pytest.raises(NotImplementedError):
        layer(_act_pack(q, sf, gs, ids, wts), pack)


def test_firefly_rechecks_weight_layout():
    """A weight view whose tensor shares storage with a checked one but has
    another layout is checked again (and rejected)."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    g, pack, _, _ = _setup(num_experts, top_k, hidden, inter, act)
    layer = _layer(num_experts, top_k, inter, act, 8)
    _, q, sf, gs, ids, wts = _inputs(g, 8, hidden, num_experts, top_k)
    layer(_act_pack(q, sf, gs, ids, wts), pack)
    moe = layer.runners[0]._moes[hidden]
    bad = dict(pack.get_view("firefly_nvfp4"))
    bad["w1"] = bad["w1"].transpose(1, 2)
    with pytest.raises(ValueError):
        moe(q, sf, gs, bad, topk_ids=ids, topk_weights=wts)


def test_firefly_failed_capture_keeps_arguments_consistent():
    """A capture that fails on new weights' missing direct descriptors leaves
    the plan's launch arguments matching the weights of its last call."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    _, pack_b, _, _ = _setup(num_experts, top_k, hidden, inter, act, seed=1)
    layer = _layer(num_experts, top_k, inter, act, 8)
    view, view_b = pack.get_view("firefly_nvfp4"), pack_b.get_view("firefly_nvfp4")
    x_deq, q, sf, gs, ids, wts = _inputs(g, 2, hidden, num_experts, top_k)
    layer(_act_pack(q, sf, gs, ids, wts), pack)  # M = 2: the direct GEMMs
    moe = layer.runners[0]._moes[hidden]
    _, q1, sf1, gs1, ids1, wts1 = _inputs(g, 1, hidden, num_experts, top_k)
    moe(q1, sf1, gs1, view_b, topk_ids=ids1, topk_weights=wts1)  # B checked at M = 1
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError),
        torch.cuda.graph(graph, stream=torch.cuda.Stream(q.device)),
    ):
        moe(q, sf, gs, view_b, topk_ids=ids, topk_weights=wts)
    out = moe(q, sf, gs, view, topk_ids=ids, topk_weights=wts)
    ref = _reference(
        x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, 2, hidden)
    )
    assert _rel_rmse(out, ref) < 0.005


def test_firefly_preparation_builds_direct_descriptors():
    """The runner's preparation (no launch) also builds the direct GEMMs'
    descriptors of the packed weights, so the batch size can be captured."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    layer = _layer(num_experts, top_k, inter, act, 8)
    runner = layer.runners[0]
    x_deq, q, sf, gs, ids, wts = _inputs(g, 2, hidden, num_experts, top_k)
    inputs = runner.pack_inputs(_act_pack(q, sf, gs, ids, wts), pack)
    runner.forward(inputs, do_preparation=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream(q.device)):
        out = runner.forward(inputs)
    graph.replay()
    torch.cuda.synchronize()
    ref = _reference(
        x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, 2, hidden)
    )
    assert _rel_rmse(out, ref) < 0.005


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
def test_firefly_repeated_expert_ids_stay_in_bounds(fused):
    """Tokens naming one expert in every slot (not a top-k, but accepted) on a
    short batch that lists each expert's routes (block_m <= 32): the routes
    past the list are dropped, nothing is written past it, the output is
    finite and the next call is correct."""
    num_experts, top_k, hidden, inter, act = 16, 4, 1024, 512, SwiGLU()
    g, pack, w1, w2 = _setup(num_experts, top_k, hidden, inter, act)
    layer = _layer(num_experts, top_k, inter, act, 64, fused=fused)
    m = 64
    x_deq, q, sf, gs, ids, wts = _inputs(g, m, hidden, num_experts, top_k)
    rep = torch.full_like(ids, num_experts - 1)
    out = layer(_act_pack(q, sf, gs, rep, wts), pack)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    out = layer(_act_pack(q, sf, gs, ids, wts), pack)
    ref = _reference(
        x_deq, ids, wts, w1, w2, act, inter, _mid_per_token(layer, m, hidden)
    )
    assert _rel_rmse(out, ref) < (0.0065 if fused else 0.005)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_firefly_prepare_weights_on_device_other_than_current():
    """prepare_weights(device=cuda:1) with cuda:0 current puts every tensor of
    the view on cuda:1."""
    num_experts, hidden, inter = 8, 1024, 512
    dev = torch.device("cuda", 1)
    gen = torch.Generator(device=dev).manual_seed(0)
    w1 = (
        torch.randn((num_experts, 2 * inter, hidden), generator=gen, device=dev) * 0.02
    ).bfloat16()
    w2 = (
        torch.randn((num_experts, hidden, inter), generator=gen, device=dev) * 0.02
    ).bfloat16()
    with torch.cuda.device(0):
        view = FireflyNvfp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=inter,
            activation=SwiGLU(),
            device=dev,
        )
    for name, t in view.items():
        if isinstance(t, torch.Tensor):
            assert t.device == dev, name
