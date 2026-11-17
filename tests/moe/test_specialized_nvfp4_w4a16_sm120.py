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
"""

"""Tests for the "specialized" SM120 backend of ``b12x_fused_moe``
(agent-generated NVFP4-W4A16 routed-MoE kernels at the contract E=128,
top_k=6, hidden=2688, intermediate=1856, non-gated ReLU^2).

Covers the backend-dispatch semantics (auto routing on the shipped
token-count allowlist, never-swap-explicit, explicit "specialized"
off-allowlist raises, kill switch), numeric correctness against an
fp32-dequant reference MoE, CUDA-graph capture/replay, and the unified
MoELayer path (``BackendOptions((B12xW4A16Config(),))`` ->
``B12xW4A16Runner`` -> ``B12xMoEWrapper``), which must reach the same
specialized dispatch as the flat entry. CPU-side dispatch-logic tests
(mocked kernels, no sm120 needed) live in
``test_specialized_nvfp4_w4a16_sm120_dispatch.py``.
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available


def _is_sm120_exactly():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (12, 0)


def _has_cuda_13_1_toolkit():
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        v = get_cuda_version()
        return (v.major, v.minor) >= (13, 1)
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _is_sm120_exactly(),
        reason="the specialized NVFP4-W4A16 MoE backend is SM120-only",
    ),
    pytest.mark.skipif(
        not _has_cuda_13_1_toolkit(),
        reason="the specialized backend JIT-builds an sm_120a module whose "
        "cvt.rn.bf16x2.e2m1x2 needs a CUDA >= 13.1-line ptxas",
    ),
    pytest.mark.skipif(
        not is_cute_dsl_available(),
        reason="the stock b12x path (fallback arm of these tests) requires CuTe DSL",
    ),
]

DEVICE = "cuda"
E, TOPK, H, I = 128, 6, 2688, 1856
# A few bf16 ulp at the output magnitude: the kernels combine expert
# partials with bf16 atomics, so two invocations of the same call (eager vs
# a graph replay) may differ in accumulation order by 1-2 ulp of the
# largest output element (measured on the shipped allowlist).
BF16_ULP_REL = 2.0**-5
REL_TOL = 2e-2

_E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cpu").manual_seed(20260101)
    w1 = torch.randint(0, 256, (E, I, H // 2), generator=gen, dtype=torch.uint8).to(
        DEVICE
    )
    w2 = torch.randint(0, 256, (E, H, I // 2), generator=gen, dtype=torch.uint8).to(
        DEVICE
    )
    w1_sf = (
        (torch.rand(E, I, H // 16, generator=gen) * 0.75 + 0.25)
        .to(torch.float8_e4m3fn)
        .to(DEVICE)
    )
    w2_sf = (
        (torch.rand(E, H, I // 16, generator=gen) * 0.75 + 0.25)
        .to(torch.float8_e4m3fn)
        .to(DEVICE)
    )
    w1_alpha = (torch.rand(E, generator=gen) * 0.01 + 0.005).to(DEVICE)
    w2_alpha = (torch.rand(E, generator=gen) * 0.01 + 0.005).to(DEVICE)
    return w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha


def _routing(t, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(t, E, generator=gen)
    ids = torch.topk(scores, TOPK, dim=-1).indices.to(torch.int32)
    tw = torch.softmax(torch.rand(t, TOPK, generator=gen), dim=-1)
    x = torch.randn(t, H, generator=gen).to(torch.bfloat16)
    return x.to(DEVICE), ids.contiguous().to(DEVICE), tw.float().to(DEVICE)


def _dequant_nvfp4(packed, group_scale, alpha):
    lut = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    lo = lut[(packed & 0x0F).to(torch.int64)]
    hi = lut[(packed >> 4).to(torch.int64)]
    rows, half_cols = packed.shape
    w = torch.stack([lo, hi], dim=-1).view(rows, half_cols * 2)
    gs = group_scale.to(torch.float32).repeat_interleave(16, dim=1)
    return w * gs * alpha


def _reference(x, ids, tw, weights):
    w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha = weights
    t = x.shape[0]
    x32 = x.to(torch.float32)
    out = torch.zeros(t, H, dtype=torch.float32, device=x.device)
    ids64 = ids.to(torch.int64)
    for e in torch.unique(ids64).tolist():
        tok, slot = (ids64 == e).nonzero(as_tuple=True)
        a = _dequant_nvfp4(w1[e], w1_sf[e], w1_alpha[e])
        b = _dequant_nvfp4(w2[e], w2_sf[e], w2_alpha[e])
        inter = torch.clamp(x32[tok] @ a.T, min=0.0)
        inter = inter * inter
        out.index_add_(0, tok, (inter @ b.T) * tw[tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


def _call(x, ids, tw, weights, backend, out=None):
    from flashinfer.fused_moe import b12x_fused_moe

    w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha = weights
    return b12x_fused_moe(
        x,
        w1,
        w1_sf,
        w2,
        w2_sf,
        ids,
        tw,
        E,
        TOPK,
        w1_alpha=w1_alpha,
        w2_alpha=w2_alpha,
        fc2_input_scale=torch.ones(1, device=DEVICE),
        output=out,
        output_dtype=torch.bfloat16,
        activation="relu2",
        quant_mode="w4a16",
        source_format="modelopt",
        backend=backend,
    )


def _assert_close_to_reference(out, ref):
    scale = ref.float().abs().max().clamp(min=1.0)
    rel_l2 = (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    ).item()
    max_abs = (out.float() - ref.float()).abs().max().item()
    assert rel_l2 < REL_TOL, f"rel_l2 {rel_l2}"
    assert max_abs < 0.05 * scale.item(), f"max_abs {max_abs}"


def test_backends_declared_and_bogus_backend_raises(weights):
    from flashinfer.fused_moe.cute_dsl.b12x_moe import BACKENDS

    assert BACKENDS == ("auto", "specialized", "b12x")
    x, ids, tw = _routing(16, seed=1)
    with pytest.raises(ValueError):
        _call(x, ids, tw, weights, backend="fa2")


@pytest.mark.parametrize("t", [4, 16])
def test_auto_routes_on_allowlist_and_matches_reference(weights, t):
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    x, ids, tw = _routing(t, seed=100 + t)
    ref = _reference(x, ids, tw, weights)
    before = spec.dispatch_stats.get("specialized", 0)
    out = _call(x, ids, tw, weights, backend="auto")
    torch.cuda.synchronize()
    assert spec.dispatch_stats.get("specialized", 0) == before + 1
    _assert_close_to_reference(out, ref)


def test_explicit_stock_backend_never_receives_specialized(weights):
    """``backend="b12x"`` must reach the stock path even on an allowlisted
    token count (the stock path may itself reject the row-major scale
    layout; what matters is that the specialized kernels are not used)."""
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    x, ids, tw = _routing(16, seed=2)
    before = spec.dispatch_stats.get("specialized", 0)
    try:
        _call(x, ids, tw, weights, backend="b12x")
        torch.cuda.synchronize()
    except ValueError:
        pass  # stock scale-layout rejection: the stock path was reached
    assert spec.dispatch_stats.get("specialized", 0) == before


def test_auto_off_allowlist_falls_back_to_stock(weights):
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    x, ids, tw = _routing(8, seed=3)  # 8 is not an admitted token count
    before = spec.dispatch_stats.get("specialized", 0)
    try:
        _call(x, ids, tw, weights, backend="auto")
        torch.cuda.synchronize()
    except ValueError:
        pass  # stock scale-layout rejection: the stock path was reached
    assert spec.dispatch_stats.get("specialized", 0) == before
    assert any(
        any(r.startswith("token_count") for r in reasons)
        for reasons in spec.fallback_reasons
    )


def test_explicit_specialized_off_allowlist_raises(weights):
    x, ids, tw = _routing(7, seed=4)
    with pytest.raises(ValueError, match="token_count"):
        _call(x, ids, tw, weights, backend="specialized")


def test_kill_switch(weights, monkeypatch):
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    monkeypatch.setenv(spec.ENV_DISABLE, "1")
    x, ids, tw = _routing(16, seed=5)
    before = spec.dispatch_stats.get("specialized", 0)
    try:
        _call(x, ids, tw, weights, backend="auto")
        torch.cuda.synchronize()
    except ValueError:
        pass  # stock scale-layout rejection: the stock path was reached
    assert spec.dispatch_stats.get("specialized", 0) == before
    with pytest.raises(ValueError, match="kill_switch"):
        _call(x, ids, tw, weights, backend="specialized")


def _unified_layer_and_weight_pack(weights, t):
    """MoELayer over the single opt-in b12x W4A16 backend candidate."""
    from flashinfer.fused_moe import (
        B12xW4A16Config,
        MoELayer,
        MoEWeightPack,
    )
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        ReLU2,
        RoutingConfig,
    )

    w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha = weights
    prepared = B12xW4A16Config.prepare_weights(
        w1,
        w1_sf,
        w1_alpha,
        w2,
        w2_sf,
        w2_alpha,
        activation=ReLU2(),
        source_format="modelopt",
    )
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("b12x_w4a16", prepared)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=TOPK),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=I, local_num_experts=E),
        activation=ReLU2(),
        backend=BackendOptions((B12xW4A16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=t),
    )
    layer = MoELayer(config)
    assert len(layer.runners) == 1
    layer._select_winner = lambda *_: (layer.runners[0], -1)
    return layer, weight_pack


def _act_pack(x, ids, tw):
    from flashinfer.fused_moe import MoEActivationPack

    return MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=torch.empty(0, device=DEVICE),
        topk_ids=ids,
        topk_weights=tw,
    )


@pytest.mark.parametrize("t", [4, 16])
def test_moe_layer_reaches_specialized_and_matches_reference(weights, t):
    """The unified MoELayer path (B12xW4A16Runner -> B12xMoEWrapper) must
    reach the specialized kernels on allowlisted shapes, not just the flat
    b12x_fused_moe entry."""
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    x, ids, tw = _routing(t, seed=300 + t)
    ref = _reference(x, ids, tw, weights)
    layer, weight_pack = _unified_layer_and_weight_pack(weights, t)
    before = spec.dispatch_stats.get("specialized", 0)
    out = layer(_act_pack(x, ids, tw), weight_pack)
    torch.cuda.synchronize()
    assert spec.dispatch_stats.get("specialized", 0) == before + 1
    assert layer.winner_backend == "b12x_w4a16"
    _assert_close_to_reference(out, ref)


def test_moe_layer_off_allowlist_keeps_stock_and_matches_reference(weights):
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    t = 8  # not an admitted token count
    x, ids, tw = _routing(t, seed=301)
    ref = _reference(x, ids, tw, weights)
    layer, weight_pack = _unified_layer_and_weight_pack(weights, t)
    before = spec.dispatch_stats.get("specialized", 0)
    out = layer(_act_pack(x, ids, tw), weight_pack)
    torch.cuda.synchronize()
    assert spec.dispatch_stats.get("specialized", 0) == before
    assert any(
        any(r.startswith("token_count") for r in reasons)
        for reasons in spec.fallback_reasons
    )
    _assert_close_to_reference(out, ref)


def test_moe_layer_kill_switch_keeps_stock(weights, monkeypatch):
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    monkeypatch.setenv(spec.ENV_DISABLE, "1")
    t = 16
    x, ids, tw = _routing(t, seed=302)
    ref = _reference(x, ids, tw, weights)
    layer, weight_pack = _unified_layer_and_weight_pack(weights, t)
    before = spec.dispatch_stats.get("specialized", 0)
    out = layer(_act_pack(x, ids, tw), weight_pack)
    torch.cuda.synchronize()
    assert spec.dispatch_stats.get("specialized", 0) == before
    assert any("kill_switch" in reasons for reasons in spec.fallback_reasons)
    _assert_close_to_reference(out, ref)


def test_cuda_graph_replay_matches_eager(weights):
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha = weights
    spec.warmup(w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha, token_counts=(16, 4))
    t = 16
    x, ids, tw = _routing(t, seed=6)
    eager = torch.empty(t, H, dtype=torch.bfloat16, device=DEVICE)
    _call(x, ids, tw, weights, backend="auto", out=eager)
    torch.cuda.synchronize()
    eager = eager.clone()
    out = torch.empty(t, H, dtype=torch.bfloat16, device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            _call(x, ids, tw, weights, backend="auto", out=out)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            _call(x, ids, tw, weights, backend="auto", out=out)
    torch.cuda.synchronize()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    tol = BF16_ULP_REL * eager.float().abs().max().clamp(min=1.0).item()
    max_abs = (out.float() - eager.float()).abs().max().item()
    assert max_abs <= tol, f"graphed vs eager max_abs {max_abs} > {tol}"
