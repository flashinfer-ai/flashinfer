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

"""Unified-API tests for the specialized SM120 FP8 block-scale MoE backend.

``SpecializedFp8BlockConfig`` + ``SpecializedFp8BlockRunner`` are the
agent-generated kernels' sole public home: a DEFAULT ``MoELayer`` candidate
whose runner calls the kernels' own module directly (no
``cutlass_fused_moe`` hop). Registration, arch gating, default-list
visibility and the GPU-free support checks run anywhere; layer dispatch,
kill-switch fallback and numeric parity against the module's own dispatch
run only on the validated SM120 device profile.
"""

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SpecializedFp8BlockConfig,
    SpecializedFp8BlockRunner,
)
from flashinfer.fused_moe.api import _DEFAULT_BACKEND
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS, MoELayer, NoEligibleBackend

_KILL_SWITCH = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

E, TOPK, H, I = 256, 8, 2048, 512


def _contract_config(**overrides):
    kwargs = dict(
        routing=RoutingConfig(num_experts=E, top_k=TOPK),
        quant=QuantConfig(
            weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8
        ),
        experts=ExpertConfig(intermediate_size=I),
        backend=BackendOptions(candidates=(SpecializedFp8BlockConfig(),)),
    )
    kwargs.update(overrides)
    return MoEConfig(**kwargs)


def _is_validated_sm120_device():
    # Gate on the CURRENT device: DEVICE = "cuda" and MoELayer() both resolve
    # to it, so checking device 0 would mis-gate on multi-GPU hosts.
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.major, props.minor) == (12, 0) and props.multi_processor_count == 188


def _has_min_cuda_toolkit():
    # Shared probe with the runner's support check: a cached module artifact
    # needs no compiler; otherwise the JIT-selected nvcc (FLASHINFER_NVCC
    # override, else $CUDA_HOME/bin/nvcc) must be CUDA 13.2+. The torch CUDA
    # build version is never substituted.
    if not _is_validated_sm120_device():
        return False
    from flashinfer.fused_moe.runners import _specialized_jit_nvcc_release
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        gen_specialized_fused_moe_fp8_blockscale_sm120_module,
    )

    if gen_specialized_fused_moe_fp8_blockscale_sm120_module().is_compiled:
        return True
    v = _specialized_jit_nvcc_release()
    return v is not None and v >= (13, 2)


requires_sm120 = pytest.mark.skipif(
    not _is_validated_sm120_device(),
    reason="the specialized fp8 block-scale MoE backend serves only the "
    "validated SM120 device profile (CC 12.0, 188 SMs)",
)
requires_toolkit = pytest.mark.skipif(
    not _has_min_cuda_toolkit(),
    reason="the specialized backend JIT-builds an sm_120a module "
    "(CUDA 13.2+ toolkit required for PTX ISA 9.2)",
)


# ---------------------------------------------------------------------------
# Registration + gating (GPU-free)
# ---------------------------------------------------------------------------


def test_config_registered_as_layer_backend():
    assert _BACKEND_RUNNERS[SpecializedFp8BlockConfig] is SpecializedFp8BlockRunner
    from flashinfer.fused_moe.api import ALL_BACKEND_CONFIGS

    assert SpecializedFp8BlockConfig in ALL_BACKEND_CONFIGS


def test_supported_arch_gating():
    assert SpecializedFp8BlockConfig.supported(120)
    # The shipped dispatch surface pins CC 12.0 exactly; SM121 (and every
    # other arch) must be refused.
    for arch in (90, 100, 103, 107, 121):
        assert not SpecializedFp8BlockConfig.supported(arch)


def test_default_backend_visibility():
    """DEFAULT-VISIBLE: a matching layer reaches the kernels through
    backend=auto with no opt-in; supported() gates the arch (SM120) and the
    runner gates the generated contract."""
    assert SpecializedFp8BlockConfig in {type(c) for c in _DEFAULT_BACKEND.candidates}
    assert any(
        isinstance(c, SpecializedFp8BlockConfig)
        for c in _DEFAULT_BACKEND.valid_for(120)
    )
    for arch in (90, 100, 103, 121):
        assert not any(
            isinstance(c, SpecializedFp8BlockConfig)
            for c in _DEFAULT_BACKEND.valid_for(arch)
        )


def test_runner_quant_and_routing_declarations():
    assert SpecializedFp8BlockRunner.supports_quant(
        QuantConfig(weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8)
    )
    assert not SpecializedFp8BlockRunner.supports_quant(QuantConfig())
    from flashinfer.fused_moe.api import RoutingInputMode

    assert SpecializedFp8BlockRunner.supported_routing_modes == (
        RoutingInputMode.PackedPrecomputed,
    )
    assert not SpecializedFp8BlockRunner.supports_expert_parallelism


def test_prepare_weights_swaps_canonical_halves():
    """MoEWeightPack rule: prepare_weights takes canonical [up, gate] fc1 rows
    and performs the swap to the kernels' native [gate; up] order itself, on
    the payload and its block scales together (GPU-free)."""
    e, two_i, h = 2, 2 * I, H
    # Distinct, exactly representable fp8 constants per half: large arange
    # values (448 < v, e.g. 480) convert to NaN in e4m3fn, and torch.equal
    # rejects matching NaNs, so arange payloads would fail the swap
    # assertions below even for a correct swap. The scale assertions keep
    # their arange payload, so an incorrect scale swap is still detected.
    w13 = torch.empty(e, two_i, h, dtype=torch.float32)
    w13[:, : two_i // 2] = 1.0  # canonical up half
    w13[:, two_i // 2 :] = -2.0  # canonical gate half
    w13 = w13.to(torch.float8_e4m3fn)
    w13_sf = torch.arange(e * (two_i // 128) * (h // 128), dtype=torch.float32).reshape(
        e, two_i // 128, h // 128
    )
    w2 = torch.zeros(e, h, I, dtype=torch.float8_e4m3fn)
    w2_sf = torch.zeros(e, h // 128, I // 128, dtype=torch.float32)
    view = SpecializedFp8BlockConfig.prepare_weights(w13, w13_sf, w2, w2_sf)
    native, native_sf = view["w13_weight"], view["w13_weight_sf"]
    assert native.is_contiguous() and native_sf.is_contiguous()
    # native first half (kernel's gate) == canonical second half (gate)
    assert torch.equal(native[:, :I].to(torch.float32), w13[:, I:].to(torch.float32))
    assert torch.equal(native[:, I:].to(torch.float32), w13[:, :I].to(torch.float32))
    assert torch.equal(native_sf[:, : I // 128], w13_sf[:, I // 128 :])
    assert torch.equal(native_sf[:, I // 128 :], w13_sf[:, : I // 128])
    # fc2 passes through untouched
    assert view["w2_weight"] is w2 and view["w2_weight_sf"] is w2_sf
    with pytest.raises(ValueError, match=r"\[up, gate\]"):
        SpecializedFp8BlockConfig.prepare_weights(
            w13[:, : two_i - 1], w13_sf, w2, w2_sf
        )


def test_kill_switch_rejects_at_check_support(monkeypatch):
    """The GPU-free kill-switch check rejects the runner before any device
    query, so it runs on any host."""
    monkeypatch.setenv(_KILL_SWITCH, "1")
    runner = SpecializedFp8BlockRunner(_contract_config(), torch.device("cuda", 0))
    with pytest.raises(RuntimeError, match=_KILL_SWITCH):
        runner.check_support()


def test_kill_switch_blocks_warmup(monkeypatch):
    """The kill switch disables the backend ENTIRELY: warmup must return
    before JIT-loading the module or launching anything, and must not record
    warmed variants (GPU-free: the switch is checked before any device work)."""
    import flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 as spec

    monkeypatch.setenv(_KILL_SWITCH, "1")

    def _boom():
        raise AssertionError("warmup must not touch the kernel module while disabled")

    monkeypatch.setattr(spec, "_kernel_entry", _boom)
    warmed_before = set(spec._warmed)
    # Disabled warmup returns before reading the operands, so these are
    # small placeholders rather than checkpoint-sized weights (full geometry
    # stays in the tests that actually reach the kernel).
    w13 = torch.zeros(2, 4, 4, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(2, 4, 2, dtype=torch.float8_e4m3fn)
    w13_sf = torch.zeros(2, 1, 1, dtype=torch.float32)
    w2_sf = torch.zeros(2, 1, 1, dtype=torch.float32)
    spec.warmup(w13, w2, [w13_sf, w2_sf], device="cpu")
    assert set(spec._warmed) == warmed_before


@pytest.mark.parametrize("case", ["off_list", "wrong_cc", "wrong_sms"])
def test_warmup_refuses_before_operands_or_jit(monkeypatch, case):
    from types import SimpleNamespace
    from flashinfer.fused_moe.specialized import fp8_blockscale_moe_sm120 as mod

    monkeypatch.delenv(_KILL_SWITCH, raising=False)
    monkeypatch.setattr(mod, "_warmed", {(0, 1)})
    monkeypatch.setattr(mod, "_kernel_entry", lambda: pytest.fail("entered kernel"))
    monkeypatch.setattr(
        mod,
        "get_compute_capability",
        lambda _: (12, 1) if case == "wrong_cc" else (12, 0),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=170),
    )
    ts = (4, 7) if case == "off_list" else (4,)
    error, message = {
        "off_list": (ValueError, "off the shipped"),
        "wrong_cc": (RuntimeError, "compute capability"),
        "wrong_sms": (RuntimeError, "170 SMs"),
    }[case]
    with pytest.raises(error, match=message):
        mod.warmup(None, None, [], device="cuda:0", t_values=ts)
    assert mod._warmed == {(0, 1)}


def test_off_contract_rejected_at_check_support(monkeypatch):
    monkeypatch.delenv(_KILL_SWITCH, raising=False)
    config = _contract_config(experts=ExpertConfig(intermediate_size=2 * I))
    runner = SpecializedFp8BlockRunner(config, torch.device("cuda", 0))
    with pytest.raises(NotImplementedError, match="generated contract"):
        runner.check_support()


# ---------------------------------------------------------------------------
# Layer dispatch + numerics (SM120 silicon)
# ---------------------------------------------------------------------------

DEVICE = "cuda"


@pytest.fixture(scope="module")
def weights():
    """Canonical MoEWeightPack weights: fc1 rows in [up, gate] order."""
    gen = torch.Generator(device="cpu").manual_seed(20260102)
    w13 = torch.randn(E, 2 * I, H, generator=gen).to(torch.float8_e4m3fn)
    w2 = torch.randn(E, H, I, generator=gen).to(torch.float8_e4m3fn)
    w13_scale = (
        torch.rand(E, (2 * I) // 128, H // 128, generator=gen) * 0.02 + 0.01
    ).float()
    w2_scale = (torch.rand(E, H // 128, I // 128, generator=gen) * 0.02 + 0.01).float()
    return (
        w13.to(DEVICE),
        w2.to(DEVICE),
        w13_scale.contiguous().to(DEVICE),
        w2_scale.contiguous().to(DEVICE),
    )


def _weight_pack(weights):
    w13, w2, w13_scale, w2_scale = weights
    pack = MoEWeightPack()
    pack.prepare_for(
        "specialized_fp8_block",
        SpecializedFp8BlockConfig.prepare_weights(w13, w13_scale, w2, w2_scale),
    )
    return pack


def _routing(t, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(t, E, generator=gen)
    ids = torch.topk(scores, TOPK, dim=-1).indices.to(torch.int32)
    tw = torch.softmax(torch.rand(t, TOPK, generator=gen), dim=-1)
    x = torch.randn(t, H, generator=gen).to(torch.bfloat16)
    return (
        x.to(DEVICE),
        ids.contiguous().to(DEVICE),
        tw.float().contiguous().to(DEVICE),
    )


def _dequant_block(w_fp8, scale):
    rows, cols = w_fp8.shape
    s = scale.to(torch.float32)
    s = s.repeat_interleave(128, dim=0)[:rows]
    s = s.repeat_interleave(128, dim=1)[:, :cols]
    return w_fp8.to(torch.float32) * s


def _reference(x, ids, tw, weights):
    w13, w2, w13_scale, w2_scale = weights
    t = x.shape[0]
    x32 = x.to(torch.float32)
    out = torch.zeros(t, H, dtype=torch.float32, device=x.device)
    ids64 = ids.to(torch.int64)
    for e in torch.unique(ids64).tolist():
        tok, slot = (ids64 == e).nonzero(as_tuple=True)
        w13_e = _dequant_block(w13[e], w13_scale[e])
        w2_e = _dequant_block(w2[e], w2_scale[e])
        up_gate = x32[tok] @ w13_e.T
        # Canonical [up, gate] rows: first half up, second half gate.
        up, gate = up_gate[:, :I], up_gate[:, I:]
        inter = (gate * torch.sigmoid(gate)) * up
        out.index_add_(0, tok, (inter @ w2_e.T) * tw[tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


@requires_sm120
@requires_toolkit
@pytest.mark.parametrize("t", [4, 16])
def test_moe_layer_reaches_specialized_runner(weights, t):
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        specialized_moe_fp8_blockscale_stats,
    )

    layer = MoELayer(_contract_config())
    assert [r.backend_key for r in layer.runners] == ["specialized_fp8_block"]
    x, ids, tw = _routing(t, seed=300 + t)
    act = MoEActivationPack(x, None, ids, tw)
    before = specialized_moe_fp8_blockscale_stats()["specialized_dispatches"]
    out = layer(act, _weight_pack(weights))
    torch.cuda.synchronize()
    after = specialized_moe_fp8_blockscale_stats()["specialized_dispatches"]
    assert after > before
    assert layer._last_winner_backend == "specialized_fp8_block"
    ref = _reference(x, ids, tw, weights)
    rel_l2 = (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
    ).item()
    assert rel_l2 < 2e-2, f"rel_l2 {rel_l2}"


@requires_sm120
@requires_toolkit
def test_moe_layer_off_allowlist_token_count_refused(weights):
    """A token count off the shipped allowlist must not reach the kernels:
    with no other candidate the layer reports no eligible backend."""
    layer = MoELayer(_contract_config())
    x, ids, tw = _routing(7, seed=301)
    act = MoEActivationPack(x, None, ids, tw)
    with pytest.raises(NotImplementedError):
        layer(act, _weight_pack(weights))


@requires_sm120
@requires_toolkit
def test_moe_layer_kill_switch_fallback(weights, monkeypatch):
    """With the kill switch set, the specialized candidate drops out at layer
    construction; the layer either falls back to another configured backend
    or (with no other candidate) reports no eligible backend naming the
    switch."""
    from flashinfer.fused_moe import SM12xFp8Config

    monkeypatch.setenv(_KILL_SWITCH, "1")

    # With no other candidate, construction fails and names the switch.
    with pytest.raises(NoEligibleBackend, match=_KILL_SWITCH):
        MoELayer(_contract_config())

    # With a sibling candidate, the layer falls back to it (or, where the
    # sibling is itself unavailable — SM12x needs CUDA 13 + the CuTe DSL
    # package — the failure still carries the kill-switch reason).
    try:
        layer = MoELayer(
            _contract_config(
                backend=BackendOptions(
                    candidates=(SpecializedFp8BlockConfig(), SM12xFp8Config())
                )
            )
        )
    except NoEligibleBackend as exc:
        assert _KILL_SWITCH in str(exc)
    else:
        assert "specialized_fp8_block" not in [r.backend_key for r in layer.runners]


@requires_sm120
@requires_toolkit
@pytest.mark.parametrize("fork_join", [False, True], ids=["serial", "two-stream"])
def test_layer_same_graph_workspace_isolation(weights, fork_join):
    # Share only read-only weights. For x filled with a=1 or 2, every
    # up/gate dot is H*a/64 = 32*a. SiLU rounds to the gate itself here,
    # FC2 sums I copies with scale 1/256, and the route weights sum to 1.
    w13, w2, s13, s2 = weights
    pack = _weight_pack(
        (
            torch.ones_like(w13),
            torch.ones_like(w2),
            torch.full_like(s13, 1 / 64),
            torch.full_like(s2, 1 / 256),
        )
    )
    layers = [MoELayer(_contract_config()) for _ in range(2)]
    acts = []
    refs = []
    for a, expected in ((1, 2048), (2, 8192)):
        x, ids, tw = _routing(256, seed=9573 + a)
        acts.append(MoEActivationPack(x.fill_(a), None, ids, tw.fill_(1 / TOPK)))
        refs.append(torch.full_like(x, expected))

    main = torch.cuda.Stream()
    side = torch.cuda.Stream() if fork_join else main
    # Prime both public layers outside capture, serially, so setup/JIT is
    # neither part of the race nor a reason to skip a failed dispatch.
    for layer, act, stream in zip(layers, acts, (main, side), strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            layer(act, pack)
        stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        if fork_join:
            side.wait_stream(main)  # capture the fork before either forward
        left = layers[0](acts[0], pack)
        with torch.cuda.stream(side):
            right = layers[1](acts[1], pack)
        if fork_join:
            main.wait_stream(side)  # rejoin the origin before ending capture

    assert left.data_ptr() != right.data_ptr()
    for _ in range(30):
        left.fill_(float("nan"))
        right.fill_(float("nan"))
        graph.replay()
        for out, ref in zip((left, right), refs, strict=True):
            torch.testing.assert_close(out, ref, rtol=0, atol=0)


@requires_sm120
@requires_toolkit
@pytest.mark.parametrize("t", [4, 256])
def test_layer_matches_module_dispatch(weights, t):
    """Numerics parity: the layer route serves the same kernels as the
    module's own dispatch entry. Accumulation order is not bitwise-stable on
    every token count (bf16-atomics class), so parity is asserted at a
    few-ulp tolerance at output magnitude."""
    from flashinfer.fused_moe.specialized.fp8_blockscale_moe_sm120 import (
        maybe_run_specialized,
    )

    w13, w2, w13_scale, w2_scale = weights
    x, ids, tw = _routing(t, seed=400 + t)
    layer = MoELayer(_contract_config())
    act = MoEActivationPack(x, None, ids, tw)
    out_layer = layer(act, _weight_pack(weights))
    # The module's own entry takes the kernels' native [gate; up] layout;
    # prepare_weights performs the canonical->native swap.
    native = SpecializedFp8BlockConfig.prepare_weights(w13, w13_scale, w2, w2_scale)
    direct, reason = maybe_run_specialized(
        x,
        ids,
        tw,
        native["w13_weight"],
        native["w2_weight"],
        [native["w13_weight_sf"], native["w2_weight_sf"]],
        None,
        context_ok=True,
    )
    assert reason is None and direct is not None
    torch.cuda.synchronize()
    tol = 2.0**-5 * direct.float().abs().max().clamp(min=1e-3).item()
    max_abs = (out_layer.float() - direct.float()).abs().max().item()
    assert max_abs <= tol, f"layer vs module max_abs {max_abs} > {tol}"
