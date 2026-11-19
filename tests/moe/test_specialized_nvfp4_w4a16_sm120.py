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

"""Silicon tests for the specialized SM120 NVFP4-W4A16 routed-MoE kernels
(agent-generated, contract E=128, top_k=6, hidden=2688, intermediate=1856,
non-gated ReLU^2), whose sole public home is the unified MoE API
(``SpecializedW4A16Config`` / ``SpecializedW4A16Runner``).

Covers numeric correctness against an fp32-dequant reference MoE through
the unified ``MoELayer`` surface (default-visible candidate, allowlist
eligibility, kill switch), the facade negative control (``b12x_fused_moe``
never reaches the specialized kernels), and CUDA-graph capture/replay
through the runner. CPU-side dispatch-logic tests (mocked kernels, no
sm120 needed) live in ``test_specialized_nvfp4_w4a16_sm120_dispatch.py``.
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available


def _is_sm120_exactly():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (12, 0)


def _has_cuda_13_2_toolkit():
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        v = get_cuda_version()
        return (v.major, v.minor) >= (13, 2)
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _is_sm120_exactly(),
        reason="the specialized NVFP4-W4A16 MoE backend is SM120-only",
    ),
    pytest.mark.skipif(
        not _has_cuda_13_2_toolkit(),
        reason="the specialized backend JIT-builds an sm_120a module whose "
        "cvt.rn.bf16x2.e2m1x2 needs a CUDA >= 13.2-line ptxas",
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


def _facade_call(x, ids, tw, weights, out=None):
    """The stock b12x facade (negative-control arm): never the specialized
    kernels."""
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
    )


def _assert_close_to_reference(out, ref):
    scale = ref.float().abs().max().clamp(min=1.0)
    rel_l2 = (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    ).item()
    max_abs = (out.float() - ref.float()).abs().max().item()
    assert rel_l2 < REL_TOL, f"rel_l2 {rel_l2}"
    assert max_abs < 0.05 * scale.item(), f"max_abs {max_abs}"


def test_facade_negative_control_never_reaches_specialized(weights):
    """The b12x facade carries no backend kwarg and never launches the
    specialized kernels, even on an allowlisted token count (the stock path
    may itself reject the row-major scale layout; what matters is that the
    specialized kernels are not used)."""
    import inspect

    from flashinfer.fused_moe import b12x_fused_moe
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    assert "backend" not in inspect.signature(b12x_fused_moe).parameters
    x, ids, tw = _routing(16, seed=2)
    before = spec.dispatch_stats.get("specialized", 0)
    try:
        _facade_call(x, ids, tw, weights)
        torch.cuda.synchronize()
    except ValueError:
        pass  # stock scale-layout rejection: the stock path was reached
    assert spec.dispatch_stats.get("specialized", 0) == before


def test_kill_switch_rejects_candidate(weights, monkeypatch):
    from flashinfer.fused_moe import SpecializedW4A16Config
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec
    from flashinfer.fused_moe.layer import NoEligibleBackend

    monkeypatch.setenv(spec.ENV_DISABLE, "1")
    with pytest.raises(NoEligibleBackend, match="DISABLE"):
        _unified_layer_and_weight_pack(
            weights, 16, backends=(SpecializedW4A16Config(),)
        )


def _stock_swizzled_scales(sf):
    """Row-major group-16 scales [E, rows, cols/16] -> the stock b12x
    checkpoint storage: 128-row-padded, 4-col-padded, MMA-swizzled
    [E, rows_padded, cols_padded] (the exact inverse of the repo's
    ``unswizzle_block_scale``). The two layouts are NOT interchangeable, so
    the stock view is built from this converted tensor while the
    specialized view keeps the row-major original."""
    e, rows, cb = sf.shape
    rows_p = (rows + 127) // 128 * 128
    cb_p = (cb + 3) // 4 * 4
    padded = torch.zeros(e, rows_p, cb_p, dtype=torch.uint8, device=sf.device)
    padded[:, :rows, :cb] = sf.view(torch.uint8)
    r = padded.reshape(e, rows_p // 128, 4, 32, cb_p // 4, 4)
    s = r.permute(0, 1, 4, 3, 2, 5).contiguous()
    return s.reshape(e, rows_p, cb_p).view(torch.float8_e4m3fn)


def _unified_layer_and_weight_pack(weights, t, backends=None, stock_view=None):
    """MoELayer over the b12x/specialized W4A16 backend candidates.

    Each candidate gets its OWN prepared view through its public
    ``prepare_weights`` (the specialized view is row-major unpadded; the
    stock b12x view takes the swizzled 128-row-padded scales): the views
    carry different layouts and neither aliases the other."""
    from flashinfer.fused_moe import (
        B12xW4A16Config,
        MoELayer,
        MoEWeightPack,
        SpecializedW4A16Config,
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
    if backends is None:
        backends = (B12xW4A16Config(),)
    weight_pack = MoEWeightPack()
    if any(isinstance(b, SpecializedW4A16Config) for b in backends):
        weight_pack.prepare_for(
            "specialized_w4a16",
            SpecializedW4A16Config.prepare_weights(
                w1,
                w1_sf,
                w1_alpha,
                w2,
                w2_sf,
                w2_alpha,
                activation=ReLU2(),
                source_format="modelopt",
            ),
        )
    if stock_view is None:
        stock_view = any(
            isinstance(b, B12xW4A16Config) and not isinstance(b, SpecializedW4A16Config)
            for b in backends
        )
    if stock_view:
        weight_pack.prepare_for(
            "b12x_w4a16",
            B12xW4A16Config.prepare_weights(
                w1,
                _stock_swizzled_scales(w1_sf),
                w1_alpha,
                w2,
                _stock_swizzled_scales(w2_sf),
                w2_alpha,
                activation=ReLU2(),
                source_format="modelopt",
            ),
        )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=TOPK),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=I, local_num_experts=E),
        activation=ReLU2(),
        backend=BackendOptions(backends),
        execution=ExecutionConfig(tune_max_num_tokens=t),
    )
    layer = MoELayer(config)
    assert len(layer.runners) == len(backends)
    return layer, weight_pack


def _act_pack(x, ids, tw):
    from flashinfer.fused_moe import MoEActivationPack

    return MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        topk_ids=ids,
        topk_weights=tw,
    )


@pytest.mark.parametrize("t,stock_view", [(1, False), (4, True), (16, False)])
def test_public_prepared_views_match_reference(weights, t, stock_view):
    """Configure both runners. A row-major view alone must select specialized;
    adding the swizzled stock view must let both compete without forcing a winner.
    """
    from flashinfer.fused_moe import (
        B12xW4A16Config,
        SpecializedW4A16Config,
        specialized_moe_nvfp4_w4a16_sm120 as spec,
    )

    x, ids, tw = _routing(t, seed=310 + t)
    ref = _reference(x, ids, tw, weights)
    layer, weight_pack = _unified_layer_and_weight_pack(
        weights,
        t,
        backends=(B12xW4A16Config(), SpecializedW4A16Config()),
        stock_view=stock_view,
    )
    act = _act_pack(x, ids, tw)
    expected = {"specialized_w4a16"}
    if stock_view:
        expected.add("b12x_w4a16")
    eligible = layer._eligible_runners(act, weight_pack)
    assert {layer.runners[i].backend_key for i in eligible} == expected
    before = spec.dispatch_stats.get("specialized", 0)
    out = layer(act, weight_pack)
    torch.cuda.synchronize()
    # With both views, profiling itself launches the specialized kernel;
    # the final winner is a timing result, not a correctness requirement.
    assert spec.dispatch_stats.get("specialized", 0) > before
    assert layer.winner_backend in expected
    _assert_close_to_reference(out, ref)


def test_off_allowlist_declined_and_stock_serves(weights):
    """Out-of-allowlist token counts never reach the specialized kernels:
    the runner declines at eligibility time and the stock b12x candidate
    serves the call (negative control for the removed in-runner
    fallback)."""
    from flashinfer.fused_moe import (
        B12xW4A16Config,
        SpecializedW4A16Config,
        specialized_moe_nvfp4_w4a16_sm120 as spec,
    )

    t = 8  # not an admitted token count
    x, ids, tw = _routing(t, seed=311)
    ref = _reference(x, ids, tw, weights)
    layer, weight_pack = _unified_layer_and_weight_pack(
        weights, t, backends=(B12xW4A16Config(), SpecializedW4A16Config())
    )
    act = _act_pack(x, ids, tw)
    eligible = layer._eligible_runners(act, weight_pack)
    assert [layer.runners[i].backend_key for i in eligible] == ["b12x_w4a16"]
    before = spec.dispatch_stats.get("specialized", 0)
    out = layer(act, weight_pack)
    torch.cuda.synchronize()
    assert layer.winner_backend == "b12x_w4a16"
    assert spec.dispatch_stats.get("specialized", 0) == before
    _assert_close_to_reference(out, ref)


def test_default_backend_layer_constructs_specialized_candidate(weights):
    """On SM120 with the matching quant/contract, the DEFAULT candidate
    list (backend=auto, no opt-in) includes the specialized runner."""
    from flashinfer.fused_moe import MoELayer
    from flashinfer.fused_moe.api import (
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        ReLU2,
        RoutingConfig,
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=TOPK),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=I, local_num_experts=E),
        activation=ReLU2(),
        execution=ExecutionConfig(tune_max_num_tokens=16),
    )
    layer = MoELayer(config)
    assert "specialized_w4a16" in [r.backend_key for r in layer.runners]


def test_cuda_graph_replay_matches_eager(weights):
    from flashinfer.fused_moe import SpecializedW4A16Config
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha = weights
    t = 16
    x, ids, tw = _routing(t, seed=6)
    layer, weight_pack = _unified_layer_and_weight_pack(
        weights, t, backends=(SpecializedW4A16Config(),)
    )
    runner = layer.runners[0]
    act = _act_pack(x, ids, tw)

    def run():
        return runner.forward(runner.pack_inputs(act, weight_pack))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        spec.warmup(w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha, token_counts=(16, 4))
        eager = run().clone()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = run()
    torch.cuda.synchronize()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    tol = BF16_ULP_REL * eager.float().abs().max().clamp(min=1.0).item()
    max_abs = (out.float() - eager.float()).abs().max().item()
    assert max_abs <= tol, f"graphed vs eager max_abs {max_abs} > {tol}"


# ---------------------------------------------------------------------------
# Adversarial numerics: the low-precision accumulation/conversion findings.
# Each case builds a sparse weight/activation pattern whose exact answer is
# known, so a packed-BF16 FMA accumulator (which rounds every partial sum to
# 8 significand bits) or a half-precision staging conversion (which clamps at
# 65504) produces a visibly wrong output, while the FP32-accumulation paths
# reproduce the fp32-dequant reference exactly.
# ---------------------------------------------------------------------------

_E2M1_ONE = 0x02  # low nibble = even element, +1.0; 0x20 = odd element, +1.0


def _zero_pack():
    w1 = torch.zeros(E, I, H // 2, dtype=torch.uint8, device=DEVICE)
    w2 = torch.zeros(E, H, I // 2, dtype=torch.uint8, device=DEVICE)
    w1_sf = torch.ones(E, I, H // 16).to(torch.float8_e4m3fn).to(DEVICE)
    w2_sf = torch.ones(E, H, I // 16).to(torch.float8_e4m3fn).to(DEVICE)
    w1_alpha = torch.ones(E, dtype=torch.float32, device=DEVICE)
    w2_alpha = torch.ones(E, dtype=torch.float32, device=DEVICE)
    return w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha


def _run_specialized_direct(x, ids, tw, pack):
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha = pack
    out = torch.empty(x.shape[0], H, dtype=torch.bfloat16, device=DEVICE)
    spec.run_specialized_moe_nvfp4_w4a16(
        x, w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha, ids, tw, out
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("t", [1, 4, 16, 64])
def test_small_contribution_survives_large_cancelling_partials(t):
    """A packed-BF16 dot-product accumulator loses a unit contribution
    between two large cancelling values (256 + 1 - 256 accumulates to 0,
    not 1, because 257 rounds back to 256 in BF16). The kernels must keep
    FP32 partial sums so the exact answer, 1.0, comes out at every
    allowlisted token count (T=1, 4, 16 and 64 take different kernels)."""
    pack = _zero_pack()
    w1, _, _, w2, _, _ = pack
    # Within one 16-element group (elements 0, 2, 4 of group 0 -> inter[0]).
    w1[0, 0, 0] = _E2M1_ONE
    w1[0, 0, 1] = _E2M1_ONE
    w1[0, 0, 2] = _E2M1_ONE
    # Across groups that one T=1 reduction lane owns (groups 1, 33, 65 ->
    # elements 16, 528, 1040 -> inter[1]): the running per-lane sum crosses
    # group boundaries, so it must also stay FP32.
    w1[0, 1, 8] = _E2M1_ONE
    w1[0, 1, 264] = _E2M1_ONE
    w1[0, 1, 520] = _E2M1_ONE
    # FC2 reads the two intermediates straight through to out[0] and out[1].
    w2[0, 0, 0] = _E2M1_ONE
    w2[0, 1, 0] = 0x20
    x = torch.zeros(t, H, dtype=torch.bfloat16, device=DEVICE)
    x[0, 0], x[0, 2], x[0, 4] = 256.0, 1.0, -256.0
    x[0, 16], x[0, 528], x[0, 1040] = 1024.0, 1.0, -1024.0
    ids = torch.zeros(t, TOPK, dtype=torch.int32, device=DEVICE)
    tw = torch.zeros(t, TOPK, dtype=torch.float32, device=DEVICE)
    tw[:, 0] = 1.0
    out = _run_specialized_direct(x, ids, tw, pack)
    ref = _reference(x, ids, tw, pack)
    assert torch.isfinite(out.float()).all()
    # ReLU^2(256*1 + 1*1 + (-256)*1) = 1 exactly; same for the across-group
    # coordinate. BF16 represents 1.0 exactly, so the tolerance is slack.
    assert abs(out[0, 0].item() - 1.0) < 1e-2, f"within-group: {out[0, 0].item()}"
    assert abs(out[0, 1].item() - 1.0) < 1e-2, f"across-group: {out[0, 1].item()}"
    assert (out.float() - ref.float()).abs().max().item() < 1e-2


@pytest.mark.parametrize("t", [1, 4, 16, 64])
def test_relu2_intermediate_above_half_range_stays_finite(t):
    """An admitted input can legally produce a ReLU^2 intermediate above the
    half-precision maximum of 65504 (here (1 * 300)^2 = 90000). Converting
    the FC2 input to half turns it into infinity; the BF16/FP32 path must
    carry it through to a finite output matching the fp32-dequant
    reference."""
    pack = _zero_pack()
    w1, _, w1_alpha, w2, _, _ = pack
    w1[0, 0, 0] = _E2M1_ONE
    w1_alpha[0] = 300.0
    w2[0, 0, 0] = _E2M1_ONE
    x = torch.zeros(t, H, dtype=torch.bfloat16, device=DEVICE)
    x[:, 0] = 1.0
    ids = torch.zeros(t, TOPK, dtype=torch.int32, device=DEVICE)
    tw = torch.zeros(t, TOPK, dtype=torch.float32, device=DEVICE)
    tw[:, 0] = 1.0
    out = _run_specialized_direct(x, ids, tw, pack)
    ref = _reference(x, ids, tw, pack)
    assert torch.isfinite(out.float()).all(), "half-range overflow reached the output"
    scale = ref.float().abs().max().clamp(min=1.0)
    assert (out.float() - ref.float()).abs().max().item() < 0.02 * scale.item()


# ---------------------------------------------------------------------------
# Public weight-preparation contract and workspace-retention bounds (the two
# review P1s): the specialized view is row-major/unpadded and NOT
# interchangeable with the stock swizzled view, and eager token-count growth
# must not retain unbounded workspace.
# ---------------------------------------------------------------------------


def test_public_prepare_rejects_swizzled_scales(weights):
    """The specialized prepare path refuses the stock 128-row-padded
    swizzled storage at preparation time (not at the first forward)."""
    from flashinfer.fused_moe import SpecializedW4A16Config
    from flashinfer.fused_moe.api import ReLU2

    w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha = weights
    with pytest.raises(ValueError, match="row-major"):
        SpecializedW4A16Config.prepare_weights(
            w1,
            _stock_swizzled_scales(w1_sf),
            w1_alpha,
            w2,
            _stock_swizzled_scales(w2_sf),
            w2_alpha,
            activation=ReLU2(),
            source_format="modelopt",
        )


def test_stock_only_pack_leaves_specialized_ineligible(weights):
    """A pack carrying only the b12x (swizzled) view must leave the
    specialized candidate ineligible — excluded before the shootout
    profiles it — instead of raising mid-profiling."""
    from flashinfer.fused_moe import B12xW4A16Config, SpecializedW4A16Config

    x, ids, tw = _routing(16, seed=77)
    layer, weight_pack = _unified_layer_and_weight_pack(
        weights,
        16,
        backends=(B12xW4A16Config(), SpecializedW4A16Config()),
    )
    del weight_pack.native_views["specialized_w4a16"]
    act = _act_pack(x, ids, tw)
    eligible = layer._eligible_runners(act, weight_pack)
    assert [layer.runners[i].backend_key for i in eligible] == ["b12x_w4a16"]
    out = layer(act, weight_pack)
    assert layer.winner_backend == "b12x_w4a16"
    assert torch.isfinite(out.float()).all()


def test_workspace_retention_bounded_across_token_growth(weights):
    """Twelve eager calls at strictly increasing large token counts
    (T=4097..4108 after warming T=4096) must not retain one workspace per
    call: geometric per-device growth bounds the retained total near the
    live capacity instead of ~12x a single workspace."""
    from flashinfer.fused_moe import specialized_moe_nvfp4_w4a16_sm120 as spec

    w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha = weights
    ts = [4096] + list(range(4097, 4109))
    inputs = {}
    for t in ts:
        x, ids, tw = _routing(t, seed=1000 + t)
        out = torch.empty(t, H, dtype=torch.bfloat16, device=DEVICE)
        inputs[t] = (x, ids, tw, out)

    def run_t(t):
        x, ids, tw, out = inputs[t]
        spec.run_specialized_moe_nvfp4_w4a16(
            x, w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha, ids, tw, out
        )

    run_t(4096)  # warm the large-T family and its workspace
    torch.cuda.synchronize()
    free0, _ = torch.cuda.mem_get_info()
    run_t(4097)  # the one admitted geometric growth step
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    growth_step = free0 - free1
    for t in range(4098, 4109):
        run_t(t)
    torch.cuda.synchronize()
    free2, _ = torch.cuda.mem_get_info()
    tail_growth = free1 - free2
    # Pre-fix behavior retained ~one full workspace per call (11 more
    # allocations here); geometric growth admits none for +11 tokens.
    assert tail_growth <= max(64 << 20, growth_step // 2), (
        f"workspace retention grew by {tail_growth} bytes across eleven "
        f"+1-token calls (first growth step was {growth_step} bytes): "
        "retention is not bounded"
    )
