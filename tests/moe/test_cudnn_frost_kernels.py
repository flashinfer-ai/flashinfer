# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Correctness and integration tests for selected cuDNN Frost kernels."""

import ast
import functools
import gc
import weakref
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe.backends.cudnn_frost import (
    capabilities,
)
from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
    moe as bf16_moe,
)
from flashinfer.fused_moe.backends.cudnn_frost.activations import (
    ACTIVATIONS,
    activation_name,
)
from flashinfer.fused_moe.backends.cudnn_frost.mxfp8 import (
    runtime as mxfp8,
)
from flashinfer.fused_moe.backends.cudnn_frost.mxfp8_mxfp4 import (
    runtime as mxfp8_mxfp4,
)
from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import (
    runtime as nvfp4,
)
from flashinfer.fused_moe.backends.cudnn_frost.source_template import (
    extract_template,
    render_source,
)
from flashinfer.fused_moe.backends.cudnn_frost.manifest import (
    read_manifest,
    normalize_manifest,
)
from flashinfer.fused_moe import (
    BackendOptions,
    CutlassBf16Config,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    RoutingConfig,
    SwiGLU,
)
from flashinfer.fused_moe import layer as layer_module

MXFP8_ROOT = mxfp8.runtime.artifact_root("mxfp8")
MXFP8_RECORDS = read_manifest(MXFP8_ROOT / "cudnn_frost_selected_kernels.json")[
    "kernels"
]


# Every dtype packages SM107a sources and SM120a sources of the warp-MMA templates.
frost_gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 7), (12, 0)),
    reason="packaged cuDNN Frost sources target SM107a and SM120a",
)
BF16_SHORTLIST_GEOMETRIES = {
    (12, 7168, 3072, 1),
    (12, 7168, 3072, 2),
    (12, 7168, 3072, 4),
    (8, 4096, 14336, 2),
    (64, 2048, 1408, 6),
}


def _sm120():
    """SM120 kernels store through STG only: no TMA store, CTA pair or FMA."""
    return torch.cuda.get_device_capability() == (12, 0)


def _device_arch():
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


def _skip_sm120_tma(store):
    if store == "tma" and _sm120():
        pytest.skip("SM120 Frost templates store through STG only")


def bf16_config(topk=2, experts=8, intermediate=256, ceiling=16384):
    return MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=topk),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=intermediate),
        backend=BackendOptions((CutlassBf16Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=ceiling),
    )


def bf16_packs(
    tokens=129, topk=2, experts=8, hidden=128, intermediate=256, device="cuda"
):
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.1
    w1 = (
        torch.randn(experts, 2 * intermediate, hidden, dtype=x.dtype, device=device)
        * 0.1
    )
    w2 = torch.randn(experts, hidden, intermediate, dtype=x.dtype, device=device) * 0.1
    ids = torch.randint(
        0, experts // 2, (tokens, topk), dtype=torch.int32, device=device
    )
    scores = torch.rand(tokens, topk, dtype=torch.float32, device=device)
    scores /= scores.sum(dim=1, keepdim=True)
    act = MoEActivationPack(x, None, ids, scores)
    weights = MoEWeightPack(
        {"cutlass_bf16": dict(fc1_expert_weights=w1, fc2_expert_weights=w2)}
    )
    return act, weights


def _bf16_moe_reference(act, weights, activation=None):
    activation = SwiGLU() if activation is None else activation
    x, ids, scores = act.hidden_states_q, act.topk_ids, act.topk_weights
    w = weights.get_view("cutlass_bf16")
    w1, w2 = w["fc1_expert_weights"], w["fc2_expert_weights"]
    i = w2.shape[-1]
    expanded = torch.zeros(*ids.shape, x.shape[1], device=x.device, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(ids == expert)
        from tests.moe.utils import compute_reference_activation

        values = x[token].float() @ w1[expert].float().T
        # Frost fuses activation into the FP32 GEMM accumulator. The reference
        # must not introduce a BF16 rounding boundary before the epilogue.
        mid = compute_reference_activation(values, activation, i)
        down = (mid.float() @ w2[expert].float().T).bfloat16()
        expanded[token, slot] = down.float() * scores[token, slot, None]
    return expanded.sum(dim=1).bfloat16()


@frost_gpu
@pytest.mark.parametrize("name", [name for name in ACTIVATIONS if name != "swiglu"])
@pytest.mark.parametrize("tokens,topk", [(17, 1), (129, 4)])
def test_bf16_extended_activations_graph_and_routing(name, tokens, topk, monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    torch.manual_seed(81)
    activation = ACTIVATIONS[name]()
    cfg = replace(bf16_config(topk), activation=activation)
    act, weights = bf16_packs(tokens=tokens, topk=topk)
    view = weights.get_view("cutlass_bf16")
    if not activation.is_gated:
        view["fc1_expert_weights"] = view["fc1_expert_weights"][:, :256].contiguous()
    # Drive Step clamps and SiTU saturation, with both positive and negative
    # accumulator values; small random inputs would hide these semantics.
    act.hidden_states_q.mul_(8)
    view["fc1_expert_weights"].mul_(16)
    runner = bf16_moe.CudnnFrostBf16MoeRunner(cfg, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    first, _ = bf16_moe._kernels(
        tokens * topk, 128, 256, 8, act.hidden_states_q.device, activation
    )
    assert {k.activation for k in first} == {name}
    assert {k.swap_ab for k in first} == {False, True}
    reference = _bf16_moe_reference(act, weights, activation)
    for tactic in runner.get_valid_tactics(inputs, None):
        out = runner.forward(inputs, tactic)
        relative_l2 = (
            out.float() - reference.float()
        ).norm() / reference.float().norm()
        assert relative_l2 < 0.005
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.forward(inputs, tactic)
        act.topk_ids.fill_(7)
        act.topk_ids[::3] = -1
        act.topk_ids[1::3] = 8
        reference = _bf16_moe_reference(act, weights, activation)
        out.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(out).all()
        relative_l2 = (
            out.float() - reference.float()
        ).norm() / reference.float().norm()
        assert relative_l2 < 0.005


@pytest.mark.parametrize(
    "activation",
    [
        ACTIVATIONS["swiglu"](alpha=2),
        ACTIVATIONS["swiglu_step"](limit=4),
        ACTIVATIONS["situ"](gate_scale=2),
        ACTIVATIONS["situ"](linear_scale=None),
        ACTIVATIONS["situ"](clamp_limit=7),
    ],
)
def test_bf16_extended_activation_scalars_are_not_silently_dropped(activation):
    with pytest.raises(NotImplementedError, match="default activation parameters"):
        activation_name(activation)


@pytest.mark.parametrize("capability", [(10, 7), (12, 0)])
def test_bf16_activation_artifact_pools_are_disjoint(
    capability, monkeypatch, frost_metadata_only
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
    seen = set()
    shared_fc2 = None
    for name, cls in ACTIVATIONS.items():
        first, second = bf16_moe._kernels(258, 128, 256, 8, torch.device("cuda"), cls())
        assert first and second
        assert {k.activation for k in first} == {name}
        identities = {k.artifact_id for k in first}
        assert not seen.intersection(identities)
        seen.update(identities)
        fc2_ids = {k.artifact_id for k in second}
        if shared_fc2 is not None:
            assert fc2_ids == shared_fc2
        shared_fc2 = fc2_ids


@frost_gpu
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_bf16_all_compound_tactics_full_moe_graph_and_dynamic_routing(
    topk, monkeypatch
):
    # Neither the cuDNN Frost compiler nor the CUTLASS execution API is needed.
    monkeypatch.setitem(sys.modules, "cudnn", None)
    monkeypatch.setitem(sys.modules, "cudnn.gemm.frost.compiler", None)
    import flashinfer.fused_moe as fused

    def forbidden(*args, **kwargs):
        raise AssertionError("independent cuDNN Frost must not invoke CUTLASS")

    monkeypatch.setattr(fused, "cutlass_fused_moe", forbidden)
    torch.manual_seed(59)
    act, weights = bf16_packs(topk=topk)
    runner = bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(topk), "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    expected = _bf16_moe_reference(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 4  # Two bounded choices per stage outside tuning mode.
    # Both architectures retain a swap-AB alternative in the bounded pool.
    first, second = bf16_moe._kernels(129 * topk, 128, 256, 8, torch.device("cuda"))
    assert {k.swap_ab for k in first} == {False, True}
    assert {k.swap_ab for k in second} == {False, True}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for tactic in tactics:
            out = runner.forward(inputs, tactic)
            torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner.forward(inputs, tactics[-1])
        # Change contents in-place: captured routing cannot depend on host counts.
        act.topk_ids.fill_(7)
        act.topk_weights.mul_(0.5)
        act.hidden_states_q.mul_(0.5)
        expected = _bf16_moe_reference(act, weights)
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
        torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-2)
        # Invalid ids must not escape the workspace or become weight addresses.
        act.topk_ids[::2] = -1
        act.topk_ids[1::4] = 8
        graph.replay()
        torch.testing.assert_close(
            out, _bf16_moe_reference(act, weights), atol=2e-4, rtol=2e-2
        )
    torch.cuda.current_stream().wait_stream(stream)


@frost_gpu
def test_bf16_interleaved_packs_do_not_exchange_weights_and_reject_stale_tactics():
    runner = bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), "cuda")
    runner.check_support()
    runner.build()
    a, wa = bf16_packs(tokens=17)
    b, wb = bf16_packs(tokens=33)
    pb = runner.pack_inputs(b, wb)
    pa = runner.pack_inputs(a, wa)
    assert pa.tuning_config.cuda_graph_profile_replays == 3
    pa_workspace = pa.launch_state.workspace
    pb_workspace = pb.launch_state.workspace
    # Simulate the plain-list inputs synthesized by the autotuner.
    runner.forward(list(pa), **runner.launch_kwargs_for(pa))
    runner.forward(pb)
    assert pa.launch_state.workspace is pa_workspace
    assert pb.launch_state.workspace is pb_workspace
    torch.testing.assert_close(pa[0], _bf16_moe_reference(a, wa), atol=2e-4, rtol=2e-2)
    torch.testing.assert_close(pb[0], _bf16_moe_reference(b, wb), atol=2e-4, rtol=2e-2)
    with pytest.raises(ValueError, match="stale"):
        runner.forward(pa, ("missing",))
    with pytest.raises(ValueError, match="launch_state"):
        runner.forward(list(pa))
    wa.native_views["cutlass_bf16"]["gemm1_alpha"] = torch.ones(8, device="cuda")
    assert runner.accepts(a, wa)
    with pytest.raises(ValueError, match="overrides"):
        runner.pack_inputs(a, wa)


@frost_gpu
@pytest.mark.parametrize(
    "experts,hidden,intermediate",
    [(12, 7168, 3072), (8, 4096, 14336), (64, 2048, 1408)],
)
def test_bf16_model_geometry_artifacts_ragged_graph(
    experts, hidden, intermediate, monkeypatch
):
    """Exercise every packaged pair on partial tiles and initially empty experts."""
    monkeypatch.setitem(sys.modules, "cudnn.gemm.frost.compiler", None)
    torch.manual_seed(97)
    act, weights = bf16_packs(
        tokens=129, experts=experts, hidden=hidden, intermediate=intermediate
    )
    runner = bf16_moe.CudnnFrostBf16MoeRunner(
        bf16_config(experts=experts, intermediate=intermediate), "cuda"
    )
    runner.check_support()
    runner.build()
    packed = runner.pack_inputs(act, weights)
    expected = _bf16_moe_reference(act, weights)
    expected_norm = expected.float().norm()
    tactics = runner.get_valid_tactics(packed, None)
    assert tactics
    for tactic in tactics:
        actual = runner.forward(packed, tactic)
        assert torch.isfinite(actual).all().item()
        error = (actual.float() - expected.float()).norm() / expected_norm
        assert error.item() < 0.01, (tactic, error.item())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.forward(packed, tactic)
        actual.fill_(float("nan"))
        graph.replay()
        error = (actual.float() - expected.float()).norm() / expected_norm
        assert error.item() < 0.01, (tactic, error.item())
    # Captured routing must work when previously empty experts become populated.
    act.topk_ids.fill_(experts - 1)
    act.topk_weights.mul_(0.5)
    graph.replay()
    expected = _bf16_moe_reference(act, weights)
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error.item() < 0.01


@frost_gpu
@pytest.mark.parametrize(
    "overrides",
    [
        {"activation": SwiGLU(alpha=2.0)},
        {"finalize": MoEFinalizeConfig(do_finalize=False)},
        {"experts": ExpertConfig(intermediate_size=256, local_expert_offset=1)},
    ],
)
def test_bf16_unsupported_semantics_are_rejected_at_construction(overrides):
    with pytest.raises(NotImplementedError):
        bf16_moe.CudnnFrostBf16MoeRunner(
            replace(bf16_config(), **overrides), "cuda"
        ).check_support()


@pytest.mark.parametrize("dtype", ["bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4"])
@pytest.mark.parametrize("gate", [None, "0", "1"])
def test_frost_configured_selection_ignores_experimental_gate(dtype, gate, monkeypatch):
    from tests.moe.test_unified_moe_frost import FORMATS

    frost_config, canonical_config, weight_format, act_format = FORMATS[dtype]
    cfg = replace(
        bf16_config(experts=12, intermediate=3072),
        quant=QuantConfig(weight=weight_format, activation=act_format),
        backend=BackendOptions((canonical_config(), frost_config())),
    )
    runner_cls = layer_module._BACKEND_RUNNERS[frost_config]
    frost = runner_cls.__new__(runner_cls)
    frost.config = cfg
    frost._engine = SimpleNamespace(arch_number=107)
    tensor_dtype = {"bf16": torch.bfloat16, "nvfp4": torch.uint8}.get(
        dtype, torch.float8_e4m3fn
    )
    act = MoEActivationPack(
        torch.empty(
            4096,
            7168 // (2 if dtype == "nvfp4" else 1),
            dtype=tensor_dtype,
            device="meta",
        ),
        None,
        torch.empty(4096, 2, dtype=torch.int32, device="meta"),
        torch.empty(4096, 2, dtype=torch.float32, device="meta"),
    )
    weights = MoEWeightPack({f"cutlass_{dtype}": {}})
    # Keep the real shared weight-view and cheap geometry eligibility checks.
    frost.pack_inputs = lambda act, weights: [act.hidden_states_q]
    frost.launch_kwargs_for = lambda inputs: {}
    frost.forward = lambda inputs, **kwargs: inputs[0]
    original = SimpleNamespace(
        backend_key=f"cutlass_{dtype}",
        supported_routing_modes=frost.supported_routing_modes,
        accepts=lambda act, weights: True,
    )
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.device = cfg, torch.device("cuda:0")
    layer.tuner = SimpleNamespace(is_tuning_mode=True)
    layer.runners, layer._winners = [original, frost], OrderedDict()
    selections = []

    def select(act, weights, runners):
        selections.append([runner.backend_key for runner in runners])
        return runners[-1], -1

    monkeypatch.setattr(layer, "_select_winner", select)
    if gate is None:
        monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    else:
        monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", gate)
    assert layer(act, weights) is act.hidden_states_q
    assert layer.winner_backend == frost.backend_key
    assert selections == [[original.backend_key, frost.backend_key]]
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    layer.tuner.is_tuning_mode = False
    assert layer(act, weights) is act.hidden_states_q
    assert len(selections) == 1


def test_bf16_layer_adds_independent_candidate_and_separates_winner_cache(monkeypatch):
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    # Dispatch-policy test without allocating multi-GB weights or benchmarking.
    cfg = bf16_config(experts=12, intermediate=3072)
    calls = []

    class FakeRunner:
        supported_routing_modes = (
            bf16_moe.CudnnFrostBf16MoeRunner.supported_routing_modes
        )

        def __init__(self, key):
            self.backend_key = key

        def accepts(self, act, weights):
            return self.backend_key == "cutlass_bf16" or not weights.native_views.get(
                "override"
            )

        def pack_inputs(self, act, weights):
            return [act.hidden_states_q]

        def launch_kwargs_for(self, inputs):
            return {}

        def forward(self, inputs, **kwargs):
            return inputs[0]

    old, cudnn_frost = FakeRunner("cutlass_bf16"), FakeRunner("cudnn_frost_bf16")
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.device = cfg, torch.device("cuda", 0)
    layer.tuner = SimpleNamespace(is_tuning_mode=True)
    layer.runners, layer._winners = [old, cudnn_frost], OrderedDict()

    def select(act, weights, runners):
        calls.append([r.backend_key for r in runners])
        return runners[-1], -1

    monkeypatch.setattr(layer, "_select_winner", select)
    monkeypatch.setattr(layer_module, "map_to_hybrid_bucket", lambda *args: 4096)
    act, weights = bf16_packs(
        tokens=4096, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    layer(act, weights)
    assert layer.winner_backend == "cudnn_frost_bf16"
    assert calls == [["cutlass_bf16", "cudnn_frost_bf16"]]
    assert layer.runners == [old, cudnn_frost]
    layer.tuner.is_tuning_mode = False
    layer(act, weights)
    assert len(calls) == 1
    weights.native_views["override"] = {"present": True}
    layer(act, weights)
    assert layer.winner_backend == "cutlass_bf16"
    weights.native_views.pop("override")
    # Candidate membership and token bucket match the initial call, so the
    # winner is reused while packing still receives this call's actual shape.
    act2, weights2 = bf16_packs(
        tokens=4097, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    layer(act2, weights2)
    assert layer.winner_backend == "cudnn_frost_bf16"
    assert calls[-1] == ["cutlass_bf16"]
    assert len(calls) == 2
    layer.reset_winner()
    assert not layer._winners


@frost_gpu
@pytest.mark.parametrize(
    "experts,hidden,intermediate",
    [(12, 7168, 3072), (8, 4096, 14336)],
)
def test_bf16_original_layer_api_can_execute_winning_cudnn_frost_and_replay(
    experts, hidden, intermediate, monkeypatch
):
    from flashinfer.autotuner import autotune

    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    torch.manual_seed(83)
    act, weights = bf16_packs(
        tokens=4096, experts=experts, hidden=hidden, intermediate=intermediate
    )
    layer = MoELayer(bf16_config(experts=experts, intermediate=intermediate))
    visited = []

    def select(act, weights, runners):
        # Force a winner to test dispatch independently of machine performance.
        visited.extend(r.backend_key for r in runners)
        cudnn_frost = next(r for r in runners if r.backend_key == "cudnn_frost_bf16")
        packed = cudnn_frost.pack_inputs(act, weights)
        tactic = cudnn_frost.get_valid_tactics(packed, None)[0]
        cudnn_frost.forward(packed, tactic)  # warm native objects before capture
        return cudnn_frost, tactic

    monkeypatch.setattr(layer, "_select_winner", select)
    with autotune():
        actual = layer(act, weights)
    assert visited == ["cutlass_bf16", "cudnn_frost_bf16"]
    assert layer.winner_backend == "cudnn_frost_bf16"
    expected = _bf16_moe_reference(act, weights)
    rel_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel_l2.item() < 0.01
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(act, weights)
    act.topk_ids.fill_(experts - 1)
    act.topk_weights.mul_(0.5)
    graph.replay()
    expected = _bf16_moe_reference(act, weights)
    rel_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel_l2.item() < 0.01
    assert len(visited) == 2  # cached winning runner, no re-tune on replay


@pytest.mark.parametrize("activation", list(ACTIVATIONS.values()))
def test_bf16_auto_admission_uses_validated_architecture_profiles(activation):
    from flashinfer.fused_moe.backends.cudnn_frost.bf16.support import (
        large_bf16_moe,
    )

    cfg = replace(bf16_config(experts=12, intermediate=3072), activation=activation())
    for tokens, hidden, arch, expected in (
        (4095, 7168, 100, False),
        (4096, 7168, 100, False),
        (4096, 7168, 103, False),
        (4096, 7168, 107, True),
        (4095, 7168, 107, True),
        (1, 7168, 107, True),
        (12289, 7168, 107, True),
        (4096, 4096, 100, False),
        (4096, 7168, 120, True),
        (1, 7168, 120, True),
        (12289, 7168, 120, True),
        (4096, 4096, 120, True),
        (4096, 7168, 121, False),  # SM121a sources have not been selected
    ):
        act, _ = bf16_packs(
            tokens=tokens, experts=12, hidden=hidden, intermediate=3072, device="meta"
        )
        assert large_bf16_moe(cfg, act, arch) == expected
    cfg = bf16_config(experts=8, intermediate=14336)
    for tokens, expected in [(4095, False), (4096, False)]:
        act, _ = bf16_packs(
            tokens=tokens, experts=8, hidden=4096, intermediate=14336, device="meta"
        )
        assert large_bf16_moe(cfg, act, 100) == expected
    cfg = bf16_config(topk=6, experts=64, intermediate=1408)
    act, _ = bf16_packs(
        tokens=4096,
        topk=6,
        experts=64,
        hidden=2048,
        intermediate=1408,
        device="meta",
    )
    assert not large_bf16_moe(cfg, act, 100)  # unsupported architecture
    assert large_bf16_moe(cfg, act, 107)  # four shortlisted plans compete normally
    assert large_bf16_moe(cfg, act, 120)


def test_bf16_artifact_architecture_and_swap_abi_isolation(
    monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        runtime,
    )

    for capability in ((10, 0), (10, 7), (10, 3), (12, 0), (12, 1)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
        first, second = bf16_moe._kernels(258, 128, 256, 8, torch.device("cuda"))
        if capability not in ((10, 7), (12, 0)):
            assert not first and not second
        else:
            assert first and second
            assert {k.arch for k in (*first, *second)} == {
                f"sm_{capability[0]}{capability[1]}a"
            }
            if capability == (12, 0):
                # The SM120 templates have no TMA-store variant.
                assert {
                    (k.swap_ab, k.tactic_metadata["store_mode"])
                    for k in (*first, *second)
                } == {(False, "stg"), (True, "stg")}
    # Swapping changes the native signature: a flag alone cannot reinterpret
    # an old normal object, nor can a swapped object omit its orientation.
    for abi, swap in (
        ("cudnn_frost_grouped_gemm1_swiglu", False),
        ("cudnn_frost_grouped_gemm1_swiglu_swap_ab", True),
        ("cudnn_frost_grouped_gemm2", True),
        ("cudnn_frost_grouped_gemm2_swap_ab", False),
    ):
        with pytest.raises(RuntimeError, match="ABI"):
            runtime._validate_abi(
                {"id": "bad", "abi": abi, "tactic": {"swap_ab": swap}},
                "grouped_gemm2",
            )


def test_bf16_source_distribution_has_no_binary_or_cudnn_frost_dependency():
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        runtime,
    )

    root = runtime.artifact_root("bf16")
    assert json.loads((root / runtime._MANIFEST).read_text())["schema_version"] == 4
    manifest = read_manifest(root / runtime._MANIFEST)
    assert not list(root.rglob("*.o"))
    for record in manifest["kernels"]:
        assert "object" not in record and "symbol" not in record
        path, digest = runtime._read_source(root, record)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith("cudnn") for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("cudnn")
        options = [
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "frost_compile_options"
                for t in node.targets
            )
        ]
        assert options == [f"--enable-tvm-ffi --gpu-arch {record['arch']}"]


@frost_gpu
@pytest.mark.parametrize("stage", ["fc1", "fc2"])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("store", ["stg", "tma"])
def test_bf16_explicit_final_boundary_and_graph(stage, swap, store):
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        runtime,
    )
    from flashinfer.fused_moe.backends.cudnn_frost.bf16.fc2 import (
        PreparedFc2,
    )

    torch.manual_seed(23)
    s, h, i, e = 259, 128, 256, 8
    first, second = bf16_moe._kernels(s, h, i, e, torch.device("cuda"))
    kernels = [
        kernel
        for kernel in (first if stage == "fc1" else second)
        if kernel.swap_ab == swap and kernel.tactic_metadata["store_mode"] == store
    ]
    if not kernels:
        pytest.skip("this orientation/store is not in the packaged shortlist")
    n, k = (i, h) if stage == "fc1" else (h, i)
    x = torch.randn(s, k, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    up = torch.randn_like(gate) * 0.1
    offsets = torch.tensor(
        [0, 0, 1, 128, 128, 130, 130, 130, 193], device="cuda", dtype=torch.int32
    )
    scale = torch.ones(1, device="cuda")
    out = torch.full((s, n), -7.0, device="cuda", dtype=torch.bfloat16)
    for kernel in kernels:
        workspace = torch.empty(
            kernel.workspace_bytes, device="cuda", dtype=torch.uint8
        )
        if stage == "fc1":
            assert kernel in runtime.matching_kernels(x, gate, up, offsets, scale, out)

            def run():
                runtime._launch(kernel, x, gate, up, offsets, scale, out, workspace)

        else:
            run = PreparedFc2(kernel, x, gate, offsets, out, workspace).run
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for boundaries in (
            [0, 0, 1, 128, 128, 130, 130, 130, 193],
            [0, 2, 2, 2, 31, 193, 193, 193, 193],
            [0] * 9,
        ):
            offsets.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
            expected = torch.full_like(out, -7)
            for expert, (begin, end) in enumerate(
                zip(boundaries[:-1], boundaries[1:], strict=True)
            ):
                value = x[begin:end].float() @ gate[expert].float().T
                if stage == "fc1":
                    value = torch.nn.functional.silu(value) * (
                        x[begin:end].float() @ up[expert].float().T
                    )
                expected[begin:end] = value.bfloat16()
            out.fill_(-7)
            graph.replay()
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-4)
            assert torch.all(out[boundaries[-1] :] == -7)


def test_bf16_source_manifest_rejects_tampering_and_legacy_objects(tmp_path):
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        runtime,
    )

    root = runtime.artifact_root("bf16")
    manifest = read_manifest(root / runtime._MANIFEST)
    record = next(r for r in manifest["kernels"] if r["op"] == "grouped_gemm1_swiglu")
    path = tmp_path / record["source"]["path"]
    path.parent.mkdir()
    path.write_bytes((root / record["source"]["path"]).read_bytes())
    manifest["kernels"] = [record]
    manifest_path = tmp_path / runtime._MANIFEST
    manifest_path.write_text(json.dumps(manifest))
    assert len(runtime._read_root(tmp_path)) == 1
    path.write_text(path.read_text() + "\n# modified\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        runtime._read_root(tmp_path)
    record["source"]["path"] = "../escape.py"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="invalid cuDNN Frost artifact path"):
        runtime._read_root(tmp_path)
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="unsupported cuDNN Frost manifest schema"):
        runtime._read_root(tmp_path)


@pytest.mark.parametrize("dtype", ["bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4"])
def test_sm120_sources_roundtrip_without_cudnn(dtype, monkeypatch):
    """SM120 warp-MMA templates: one STG family per op/orientation, exact re-extraction."""
    from flashinfer.fused_moe.backends.cudnn_frost import runtime

    monkeypatch.setitem(sys.modules, "cudnn", None)
    root = runtime.artifact_root(dtype)
    records = [
        record
        for record in read_manifest(root / runtime._MANIFEST)["kernels"]
        if record["arch"] == "sm_120a"
    ]
    block_scale = dtype != "bf16"
    prefix = "block_scale_" if block_scale else ""
    families = {}
    for record in records:
        entry = record["source"]
        template = (root / entry["path"]).read_text()
        concrete = render_source(template, entry["parameters"])
        tactic = record["tactic"]
        swap = tactic["swap_ab"]
        quantized = record["contract"]["output_dtype"] != "bfloat16"
        families.setdefault((record["op"], quantized, swap), set()).add(entry["path"])
        # Kernel IDs are sm120_cudnn_frost_<op>_<dtype>_<digest>. The activations
        # of one topology share a template and differ only in exact fragments.
        assert record["id"].startswith(f"sm120_cudnn_frost_{record['op']}_{dtype}_")
        stage = "fc2" if record["op"].endswith("gemm2") else "fc1"
        gated = (
            stage == "fc1" and ACTIVATIONS[record["contract"]["activation"]]().is_gated
        )
        assert entry["path"] == (
            f"sources/sm120_{stage}_{'gated' if gated else 'non_gated'}"
            f"_{'swap' if swap else 'normal'}_{'quantized' if quantized else 'bf16'}_stg.py"
        )
        # Explicit G+1 group boundaries: no final endpoint inferred from S.
        assert record["abi"] == (
            f"cudnn_frost_{record['op']}{'_quantized' if quantized else ''}"
            f"{'_swap_ab' if swap else ''}"
        )
        assert "first_token_arr[my_group + 1]" in template
        assert "gemm_s" not in template
        # Unreached helpers (e.g. SM100 tcgen05 wrappers) are not frozen in.
        assert not any(
            isinstance(node, ast.FunctionDef)
            and node.name.startswith("tcgen05")
            or isinstance(node, ast.Attribute)
            and node.attr.startswith("tcgen05")
            for node in ast.walk(ast.parse(concrete))
        )
        assert "# @@FROST_TMA_STORE@@" not in template
        # PTX setmaxnreg is .sync.aligned over the warpgroup: the producer
        # warpgroup (TMA, scheduler and donor warps) releases registers once,
        # before its per-warp roles diverge, and the compute warpgroups grow once.
        assert template.count("nvvm.setmaxregister(") == 2
        assert (
            "    if warp_idx >= NUM_COMPUTE_WARPS:\n"
            "        nvvm.setmaxregister(PROD_REG_COUNT, nvvm.SetMaxRegisterAction.DECREASE)\n"
        ) in template
        assert "SCHED_REG_COUNT" not in template
        # One CTA per SM, so ptxas keeps the setmaxnreg register split.
        assert "min_blocks_per_mp=1" in template
        assert tactic["template"] == (
            f"sm120_moe_grouped{'_block_scale' if block_scale else ''}_matmul_fwd"
            f"{'_swap_ab' if swap else ''}.py"
        )
        assert (tactic["store_mode"], tactic["cta_group"]) == ("stg", 1)
        # Only the scheduler counter: the template patches no tensormap.
        assert record["workspace_bytes"] == 128
        constants = dict(entry["parameters"]["constants"])
        assert "n_tma_outputs" not in constants and "epi_n" not in constants
        assert constants["frost_compile_options"] == repr(
            "--enable-tvm-ffi --gpu-arch sm_120a"
        )
        # Occupancy, L2 budget and pipeline depth resolve on the preparation device.
        assert entry["parameters"]["device_dependent"]
        assert all(
            constants[name] == "1"
            for name in ("grid_num_clusters", "swizzle_l2_budget_bytes", "ab_stages")
        )
        # The host zeroes the tile-scheduler counter before every launch, so
        # the exported TVM-FFI function needs no caller-side reset.
        assert "_dynamic_scheduler_counter_initialization(" in concrete
        first = entry["parameters"]["constants"][0][0]
        header = "# Block-scale config:" if block_scale else "# Tile config:"
        produced = concrete.replace(
            f"\n{first} = ", f"\n{header} {tactic['tile']}\n{first} = ", 1
        )
        extracted, parameters = extract_template(produced, swap_ab=swap)
        assert ast.dump(ast.parse(render_source(extracted, parameters))) == ast.dump(
            ast.parse(concrete)
        )
    assert {op for op, _, _ in families} == {
        *(f"{prefix}grouped_gemm1_{name}" for name in ACTIVATIONS),
        f"{prefix}grouped_gemm2",
    }
    assert {swap for _, _, swap in families} == {False, True}
    # Fused intermediate quantization is a normal-orientation FC1 variant, and
    # SM120 has no packed FP4 output.
    assert all(
        not swap and op != f"{prefix}grouped_gemm2"
        for op, quantized, swap in families
        if quantized
    )
    assert any(quantized for _, quantized, _ in families) == (
        dtype in ("mxfp8", "mxfp8_mxfp4")
    )
    assert all(len(paths) == 1 for paths in families.values())


def test_frost_export_names_architecture_first_and_prunes_unreached_helpers():
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        runtime,
    )
    from flashinfer.fused_moe.backends.cudnn_frost.export import (
        _reachable_definitions,
        arch_tag,
    )

    assert (arch_tag("sm_107a"), arch_tag("sm_120a")) == ("sm107", "sm120")
    with pytest.raises(ValueError, match="architecture"):
        arch_tag("sm120")
    source = "\n".join(
        (
            "def used(x):\n    return nested(x)",
            "def nested(x):\n    return x",
            "def unused(x):\n    return orphan(x)",
            "def orphan(x):\n    return x",
            "alias = used",
            "def kernel():\n    return alias(1)",
        )
    )
    optional = {"used", "nested", "unused", "orphan"}
    assert _reachable_definitions(source, optional) == {"used", "nested"}
    gate = torch.empty(1, 3, 4, dtype=torch.bfloat16)
    args = (torch.empty(2, 4, dtype=torch.bfloat16), gate, gate)
    scale = torch.ones(1)
    # G+1 boundaries: one boundary alone describes no group.
    with pytest.raises(ValueError, match="G\\+1"):
        runtime._validate_common(*args, torch.zeros(1, dtype=torch.int32), scale, None)
    assert runtime._validate_common(
        *args, torch.tensor([0, 2], dtype=torch.int32), scale, None
    ) == (2, 3, 4, 1, 1)


@frost_gpu
def test_bf16_sm120_launches_need_no_workspace_reset(monkeypatch):
    """The SM120 host zeroes its scheduler counter, even in a garbage workspace."""
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        fc2,
    )

    if not _sm120():
        pytest.skip("SM107 sources rely on PreparedFc2.run() resetting the workspace")
    monkeypatch.setitem(sys.modules, "cudnn", None)
    torch.manual_seed(7)
    device = torch.device("cuda", torch.cuda.current_device())
    sizes = [64, 0, 200, 128, 100, 12, 196, 68]
    bounds = [sum(sizes[:g]) for g in range(len(sizes) + 1)]
    rows = bounds[-1] + 37  # routed capacity beyond every group
    x = torch.randn(rows, 256, dtype=torch.bfloat16, device=device)
    w = torch.randn(8, 128, 256, dtype=torch.bfloat16, device=device) * 0.1
    offsets = torch.tensor(bounds, dtype=torch.int32, device=device)
    expected = torch.cat(
        [
            x[a:b].float() @ w[g].float().T
            for g, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
        ]
    )
    kernels = fc2.matching_kernels(rows, 128, 256, 8, device)
    assert kernels
    for kernel in kernels:
        out = torch.full((rows, 128), float("nan"), dtype=torch.bfloat16, device=device)
        workspace = torch.full(
            (kernel.workspace_bytes,), 0x7F, dtype=torch.uint8, device=device
        )
        plan = fc2.PreparedFc2(kernel, x, w, offsets, out, workspace)
        for _ in range(2):
            # Bypass PreparedFc2.run()'s reset: the frozen host zeroes its counter.
            plan._launch(*plan._args, fc2._current_custream(device))
            error = (out[: bounds[-1]].float() - expected).norm() / expected.norm()
            assert error.item() < 0.01, (kernel.artifact_id, error.item())
        assert torch.isnan(out[bounds[-1] :]).all(), kernel.artifact_id


@pytest.fixture
def frost_engine_cpu(monkeypatch, frost_metadata_only):
    from flashinfer.fused_moe.backends.cudnn_frost import engine

    # Host preparation binds device metadata without allocating CUDA tensors.
    monkeypatch.setattr(engine.runtime, "_arch_for", lambda _: "sm_107a")
    monkeypatch.setattr(bf16_moe, "get_compute_capability", lambda _: (10, 7))
    monkeypatch.setattr(bf16_moe.runtime, "_arch_for", lambda _: "sm_107a")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=10, minor=7, multi_processor_count=148),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # These tests isolate catalog/tensor/API admission from assembler admission.
    monkeypatch.setattr(engine, "require_moe_assembler", lambda: None)


@pytest.fixture
def bf16_compiler_environment(monkeypatch, frost_engine_cpu):
    pytest.importorskip("cutlass.cute")
    monkeypatch.setattr(bf16_moe, "require_graph_resource_retention", lambda: None)
    # Isolate source API admission from the installed assembler version.
    monkeypatch.setattr(bf16_moe, "require_moe_assembler", lambda: None)
    capabilities._compiler_error.cache_clear()
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    # Compiler admission does not need GPU allocation or kernel compilation.
    monkeypatch.setattr(bf16_moe, "get_compute_capability", lambda device: (10, 7))
    try:
        yield
    finally:
        capabilities._compiler_error.cache_clear()


@pytest.fixture
def bf16_compiler_probe(bf16_compiler_environment):
    # Removing one capability only isolates that rejection when the installed
    # compiler supports the unmodified sources for both tested activations.
    for name in ("swiglu", "geglu"):
        cfg = replace(bf16_config(), activation=ACTIVATIONS[name]())
        try:
            runner = bf16_moe.CudnnFrostBf16MoeRunner(cfg, torch.device("cuda", 0))
            runner.check_support()
        except NotImplementedError as exc:
            pytest.skip(f"Compiler fault-injection tests need compatible APIs: {exc}")
    # Fault injection below must inspect the modified APIs, not this baseline.
    capabilities._compiler_error.cache_clear()


def _assert_bf16_compiler_rejected(message):
    device = torch.device("cuda", 0)
    with pytest.raises(NotImplementedError, match=message):
        bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), device).check_support()


def test_bf16_runner_rejects_missing_source_compiler(
    monkeypatch, bf16_compiler_environment
):
    monkeypatch.setitem(sys.modules, "cutlass.experimental.primitives", None)
    # Newer DSL releases import primitives through tensor_map first. Either
    # import path must report the missing dependency and decline admission.
    _assert_bf16_compiler_rejected(r"cutlass\.experimental\.primitives")


def test_bf16_compiler_admission_rejects_older_alloc_api(
    monkeypatch, bf16_compiler_environment
):
    from cutlass.experimental import primitives

    # DSL 4.7 lacks this keyword. Keep its rejection covered independently of
    # the compatible baseline required by the other fault-injection tests.
    def older_tcgen05_alloc(tmem_ptr, num_cols, *, group=None):
        raise AssertionError("compiler admission must not execute primitives")

    monkeypatch.setattr(primitives, "tcgen05_alloc", older_tcgen05_alloc)
    _assert_bf16_compiler_rejected(
        "incompatible signature.*tcgen05_alloc.*is_exclusive"
    )


def test_bf16_missing_activation_primitive_preserves_other_activations(
    monkeypatch, bf16_compiler_probe
):
    import cutlass.cute as cute

    monkeypatch.delattr(cute.math, "erf")
    device = torch.device("cuda", 0)
    with pytest.raises(NotImplementedError, match="erf"):
        bf16_moe.CudnnFrostBf16MoeRunner(
            replace(bf16_config(), activation=ACTIVATIONS["geglu"]()), device
        ).check_support()
    bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), device).check_support()


@pytest.mark.parametrize(
    "module_name,symbol",
    [
        ("cutlass.experimental.primitives", "tcgen05_mma"),
        ("cutlass.experimental.cuda.tensor_map", "create_tensor_map_tiled"),
        ("cutlass.cute.runtime", "make_fake_compact_tensor"),
        ("cutlass.cute.runtime", "load_module"),
        ("cutlass.cute", "EnableTVMFFI"),
    ],
)
def test_bf16_compiler_admission_missing_symbol(
    module_name, symbol, monkeypatch, bf16_compiler_probe
):
    module = importlib.import_module(module_name)
    monkeypatch.delattr(module, symbol)
    _assert_bf16_compiler_rejected(symbol)


def test_bf16_compiler_admission_missing_enum_member(monkeypatch, bf16_compiler_probe):
    from cutlass.experimental import primitives

    original = primitives.Tcgen05MMACollectorOp
    monkeypatch.setattr(
        primitives,
        "Tcgen05MMACollectorOp",
        SimpleNamespace(FILL=original.FILL, USE=original.USE),
    )
    _assert_bf16_compiler_rejected("Tcgen05MMACollectorOp.LASTUSE")


@pytest.mark.parametrize("case", ["primitive_keyword", "ffi_keyword", "not_callable"])
def test_bf16_compiler_admission_incompatible_signature(
    case, monkeypatch, bf16_compiler_probe
):
    import cutlass.cute as cute
    from cutlass.experimental import primitives

    if case == "primitive_keyword":
        # Older primitive without the collector_op / b_collector_op keywords.
        monkeypatch.setattr(primitives, "tcgen05_mma", lambda *args: None)
        message = "incompatible signature.*tcgen05_mma"
    elif case == "ffi_keyword":
        monkeypatch.setattr(cute.runtime, "load_module", lambda file_path: None)
        message = "incompatible signature.*load_module"
    else:
        monkeypatch.setattr(primitives, "tcgen05_mma", 123)
        message = "tcgen05_mma is not callable"
    _assert_bf16_compiler_rejected(message)


def test_bf16_compiler_admission_requires_native_arch(monkeypatch, bf16_compiler_probe):
    import cutlass.cute as cute

    gpu_arch = cute.GPUArch

    def older_gpu_arch(arch):
        if arch == "sm_107a":
            raise ValueError("unknown GPU architecture sm_107a")
        return gpu_arch(arch)

    monkeypatch.setattr(cute, "GPUArch", older_gpu_arch)
    _assert_bf16_compiler_rejected("cannot target sm_107a")
    # An unsupported device must not fall back to a different architecture.
    monkeypatch.setattr(bf16_moe, "get_compute_capability", lambda device: (10, 0))
    with pytest.raises(NotImplementedError, match="SM107a or SM120a"):
        bf16_moe.CudnnFrostBf16MoeRunner(
            bf16_config(), torch.device("cuda", 0)
        ).check_support()


def test_bf16_compiler_admission_rechecks_arch_override(
    monkeypatch, bf16_compiler_probe
):
    device = torch.device("cuda", 0)
    bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), device).check_support()
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100f")
    _assert_bf16_compiler_rejected("CUTE_DSL_ARCH=sm_100f conflicts")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm107a")
    bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), device).check_support()


def test_bf16_compiler_admission_uses_capabilities_without_compiling(
    monkeypatch, bf16_compiler_probe
):
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler capability checks must not compile kernels")

    monkeypatch.setattr(cute, "compile", forbidden)
    monkeypatch.setitem(sys.modules, "cudnn", None)
    monkeypatch.setattr(bf16_moe, "get_compute_capability", lambda device: (10, 7))
    device = torch.device("cuda", 0)
    bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), device).check_support()
    first = capabilities._compiler_error.cache_info()
    bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), device).check_support()
    assert capabilities._compiler_error.cache_info().hits == first.hits + 1


def test_bf16_missing_compiler_preserves_layer_backend(
    monkeypatch, bf16_compiler_probe
):
    from cutlass.experimental import primitives
    from flashinfer.fused_moe import CudnnFrostBf16Config, QuantFormat
    from flashinfer.fused_moe.runners import MoERunner

    monkeypatch.delattr(primitives, "tcgen05_mma")

    class Original(MoERunner):
        backend_key = "cutlass_bf16"
        supported_routing_modes = (
            bf16_moe.CudnnFrostBf16MoeRunner.supported_routing_modes
        )
        supported_quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)

        def __init__(self, config, device):
            super().__init__()
            self.config = config

        def _check_support(self):
            pass

        def get_valid_tactics(self, inputs, profile=None):
            return [-1]

        def pack_inputs(self, act, weights):
            return [act.hidden_states_q]

        def launch_kwargs_for(self, inputs):
            return {}

        def forward(self, inputs, **kwargs):
            return inputs[0]

    monkeypatch.setattr(layer_module, "get_compute_capability", lambda _: (10, 7))
    monkeypatch.setattr(
        layer_module,
        "_BACKEND_RUNNERS",
        {
            CutlassBf16Config: Original,
            CudnnFrostBf16Config: bf16_moe.CudnnFrostBf16MoeRunner,
        },
    )
    # Exercise the ordinary default pool: unavailable Frost must preserve CUTLASS.
    cfg = MoEConfig(
        routing=RoutingConfig(num_experts=12, top_k=2),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=3072),
    )
    layer = MoELayer(cfg, device=torch.device("cuda:0"))
    assert [runner.backend_key for runner in layer.runners] == ["cutlass_bf16"]
    monkeypatch.setattr(
        layer, "_select_winner", lambda act, weights, runners: (runners[0], -1)
    )
    act, weights = bf16_packs(
        tokens=4096, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    assert layer(act, weights) is act.hidden_states_q
    assert layer.winner_backend == "cutlass_bf16"


def test_bf16_moe_shortlist_limits_stages_and_maps_token_profiles(tmp_path):
    from flashinfer.fused_moe.backends.cudnn_frost import (
        shortlist,
    )

    entries = []
    for tokens, fc1 in ((16, ["a2", "a0"]), (128, ["a1", "a2"])):
        entries.append(
            dict(
                arch="sm_107a",
                activation="swiglu",
                experts=12,
                hidden=7168,
                intermediate=3072,
                top_k=2,
                tokens=tokens,
                fc1=fc1,
                fc2=["b1", "b0"],
            )
        )
    (tmp_path / "moe_shortlists.json").write_text(
        json.dumps(dict(version=1, entries=entries))
    )
    first = tuple(SimpleNamespace(artifact_id=f"a{i}") for i in range(4))
    second = tuple(SimpleNamespace(artifact_id=f"b{i}") for i in range(4))

    def select(tokens, topk=2):
        return shortlist.select(
            (tmp_path,),
            "sm_107a",
            "swiglu",
            tokens,
            7168,
            3072,
            12,
            topk,
            first,
            second,
        )

    a, b = select(16)
    assert [k.artifact_id for k in a] == ["a2", "a0"]
    assert [k.artifact_id for k in b] == ["b1", "b0"]
    assert len(a) * len(b) == 4
    for tokens in (17, 128, 129):
        assert [k.artifact_id for k in select(tokens)[0]] == ["a1", "a2"]
    assert select(16, topk=1) == (first, second)
    with pytest.raises(ValueError, match="missing or incompatible"):
        shortlist.select(
            (tmp_path,), "sm_107a", "swiglu", 16, 7168, 3072, 12, 2, first[:1], second
        )


def test_bf16_moe_shortlist_rejects_more_than_two_stage_candidates(tmp_path):
    from flashinfer.fused_moe.backends.cudnn_frost import (
        shortlist,
    )

    entry = dict(
        arch="sm_107a",
        activation="swiglu",
        experts=12,
        hidden=7168,
        intermediate=3072,
        top_k=2,
        tokens=16,
        fc1=["a0", "a1", "a2"],
        fc2=["b0"],
    )
    (tmp_path / "moe_shortlists.json").write_text(
        json.dumps(dict(version=1, entries=[entry]))
    )
    with pytest.raises(ValueError, match="one or two"):
        shortlist._read((tmp_path,))


@pytest.mark.parametrize("arch", ["sm_107a", "sm_120a"])
def test_bf16_packaged_shortlists_resolve_all_profiles(
    arch, monkeypatch, frost_metadata_only
):
    """Every measured profile maps to two FC1 and two FC2 packaged artifacts."""
    from flashinfer.fused_moe.backends.cudnn_frost import (
        shortlist,
    )

    runtime = bf16_moe.runtime
    table = {
        key: profiles
        for key, profiles in shortlist._read(runtime._artifact_roots()).items()
        if key[0] == arch
    }
    assert {key[1] for key in table} == set(ACTIVATIONS)
    assert {key[2:] for key in table} == BF16_SHORTLIST_GEOMETRIES
    monkeypatch.setattr(runtime, "_arch_for", lambda device: arch)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=128),
    )
    runtime.clear_artifact_cache()
    try:
        for (_, name, experts, hidden, intermediate, topk), profiles in table.items():
            assert sorted(profiles) == [1, 16, 128, 512, 2048, 4096, 8192, 12288]
            previous = 0
            for tokens, (fc1, fc2) in sorted(profiles.items()):
                # Measured counts and the next-bucket rule, without compiling.
                for query in {tokens, previous + 1}:
                    first, second = bf16_moe._selected_kernels(
                        query,
                        hidden,
                        intermediate,
                        experts,
                        topk,
                        "cpu",
                        ACTIVATIONS[name](),
                    )
                    assert tuple(k.artifact_id for k in first) == fc1
                    assert tuple(k.artifact_id for k in second) == fc2
                    assert all(k.arch == arch and k.activation == name for k in first)
                    assert all(k.arch == arch for k in second)
                previous = tokens
    finally:
        runtime.clear_artifact_cache()


def test_bf16_fma_tactic_is_limited_to_sm107(monkeypatch):
    """SM120 keeps its Tensor Core candidates at the SM107 FMA geometries."""
    device = torch.device("cuda", 0)
    monkeypatch.setattr(bf16_moe.fma, "check_support", lambda *_: None)
    for arch, expected in (("sm_120a", False), ("sm_107a", True)):
        monkeypatch.setattr(bf16_moe.runtime, "_arch_for", lambda _: arch)
        for geometry in ((1, 7168, 3072, 12, 2), (4, 2048, 1408, 64, 6)):
            tactic = bf16_moe._fma_tactic(*geometry, SwiGLU(), device)
            assert (tactic is not None) == expected


def test_bf16_shortlist_survives_autotuner_plain_tensor_profiles(
    monkeypatch, frost_engine_cpu
):
    runner = bf16_moe.CudnnFrostBf16MoeRunner(
        bf16_config(experts=12, intermediate=3072), "meta"
    )
    # This test exercises shape selection; static CUDA support has separate tests.
    monkeypatch.setattr(runner, "_check_support", lambda: None)
    first = tuple(
        SimpleNamespace(tactic=("fc1", n), contract={"s": {"multiple_of": 1}})
        for n in range(2)
    )
    second = tuple(
        SimpleNamespace(tactic=("fc2", n), contract={"s": {"multiple_of": 1}})
        for n in range(2)
    )
    calls = []

    def select(*args):
        calls.append(args[:5])
        return first, second

    capture_selection = bf16_moe._capture_selection
    monkeypatch.setattr(
        bf16_moe,
        "_capture_selection",
        lambda *args: capture_selection(*args)._replace(
            select_kernels=lambda shape: select(*shape)
        ),
    )
    runner.check_support()
    runner.build()
    inputs = [None, torch.empty((16, 7168), device="meta")]
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 4
    assert calls == [(16, 7168, 3072, 12, 2)]


def test_bf16_selection_cache_shared_by_admission_and_packing(
    monkeypatch, frost_engine_cpu
):
    from contextlib import nullcontext

    runtime = bf16_moe.runtime
    runtime.clear_artifact_cache()
    runner = bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), "cpu")
    # Keep CPU tensors for the alignment/override contract checks below.
    monkeypatch.setattr(runner, "_check_support", lambda: None)
    # This packing test uses a small geometry outside the candidate shortlist.
    monkeypatch.setattr(runner, "_input_support", lambda *_: True)
    act, weights = bf16_packs(tokens=16, device="cpu")
    monkeypatch.setattr(bf16_moe, "get_compute_capability", lambda _: (10, 7))
    monkeypatch.setattr(bf16_moe, "require_moe_assembler", lambda: None)
    monkeypatch.setattr(runtime, "_arch_for", lambda _: "sm_107a")
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(bf16_moe, "_Plans", lambda *args: SimpleNamespace())
    capture_selection = bf16_moe._capture_selection
    calls = []

    def capture(*args):
        snapshot = capture_selection(*args)

        def select(shape):
            calls.append(shape)
            return snapshot.select_kernels(shape)

        return snapshot._replace(select_kernels=select)

    monkeypatch.setattr(bf16_moe, "_capture_selection", capture)
    runner.check_support()
    runner.build()
    try:
        for _ in range(2):
            assert runner.accepts(act, weights)
            assert runner.pack_inputs(act, weights)[1] is act.hidden_states_q
        assert len(calls) == 1

        # Admission checks geometry; packing validates this call's layout and
        # semantics even after a selection-cache hit.
        view = weights.get_view("cutlass_bf16")
        original = view["fc1_expert_weights"]
        view["fc1_expert_weights"] = torch.empty(
            original.numel() + 1, dtype=original.dtype
        )[1:].view_as(original)
        assert runner.accepts(act, weights)
        with pytest.raises(ValueError, match="16B-aligned"):
            runner.pack_inputs(act, weights)
        view["fc1_expert_weights"] = original
        view["bias"] = torch.empty(1)
        assert runner.accepts(act, weights)
        with pytest.raises(ValueError, match="without overrides"):
            runner.pack_inputs(act, weights)
        del view["bias"]
        assert runner.accepts(act, weights)
        assert len(calls) == 1

        # Clearing shared discovery caches does not rebind a built runner,
        # including its first preparation for a previously unseen bucket.
        monkeypatch.setattr(
            bf16_moe,
            "_capture_selection",
            lambda *args: pytest.fail("built runner must use its captured selector"),
        )
        runtime.clear_artifact_cache()
        next_act, next_weights = bf16_packs(tokens=17, device="cpu")
        next_inputs = runner.pack_inputs(next_act, next_weights)
        assert next_inputs[1] is next_act.hidden_states_q
        assert len(calls) == 2
        assert calls[-1][0] == 32
    finally:
        runtime.clear_artifact_cache()


def test_bf16_selection_cache_identity_refresh_and_bound(
    tmp_path, monkeypatch, frost_metadata_only
):
    runtime = bf16_moe.runtime
    runtime.clear_artifact_cache()
    roots, arch = (tmp_path,), "sm_107a"
    compiler_key = "initial-compiler"
    props = SimpleNamespace(multi_processor_count=128)
    monkeypatch.setattr(runtime, "_artifact_roots", lambda: roots)
    monkeypatch.setattr(runtime, "_arch_for", lambda _: arch)
    monkeypatch.setattr(runtime, "_tactic_digest", lambda _: compiler_key)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)
    calls = []

    def match(*args):
        calls.append(args)
        return (), ()

    monkeypatch.setattr(bf16_moe, "_matching_kernels", match)

    def select(**changes):
        args = dict(
            tokens=16,
            hidden=128,
            intermediate=256,
            experts=8,
            topk=2,
            device="cpu",
            activation=SwiGLU(),
        )
        args.update(changes)
        return bf16_moe._selected_kernels(**args)

    try:
        assert select() == ((), ())
        select()  # Unsupported geometries are cached too.
        assert len(calls) == 1
        for key, value in (
            ("tokens", 17),
            ("hidden", 256),
            ("intermediate", 512),
            ("experts", 12),
            ("topk", 4),
            ("activation", ACTIVATIONS["relu"]()),
        ):
            select(**{key: value})
        assert len(calls) == 7
        arch = "sm_100a"
        select()
        roots = (tmp_path / "other",)
        select()
        assert len(calls) == 9
        version = runtime._artifact_cache_version
        runtime.clear_artifact_cache()
        assert runtime._artifact_cache_version != version
        select()
        assert len(calls) == 10
        compiler_key = "changed-compiler"
        select()
        props.multi_processor_count = 80
        select()
        assert len(calls) == 12

        for tokens in range(1, 130):
            select(tokens=tokens)
        info = bf16_moe._selected_kernels_cached.cache_info()
        assert info.currsize == info.maxsize == 128
        before = len(calls)
        select(tokens=129)
        assert len(calls) == before
        select(tokens=1)  # Oldest geometry was evicted.
        assert len(calls) == before + 1
    finally:
        runtime.clear_artifact_cache()


def test_mxfp8_geometry_sharing_and_roundtrip_without_cudnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    _assert_block_scale_geometry_roundtrip(
        MXFP8_ROOT, MXFP8_RECORDS, "cutlass.Float8E4M3FN", "cutlass.Float8E8M0FNU", 32
    )


def _assert_block_scale_geometry_roundtrip(
    root, records, data_dtype, sf_dtype, block, weight_dtype=None
):
    families, by_source, source_families = {}, {}, {}
    identities = set()
    for record in records:
        assert record["id"] not in identities
        identities.add(record["id"])
        entry = record["source"]
        source = (root / entry["path"]).read_text()
        assert source.count("# @@FROST_GEOMETRY@@") == 1
        assert hashlib.sha256(source.encode()).hexdigest() == entry["sha256"]
        concrete = render_source(source, entry["parameters"])
        tree = ast.parse(concrete)
        assert not any(
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("cudnn")
            or isinstance(node, ast.Import)
            and any(a.name.startswith("cudnn") for a in node.names)
            for node in ast.walk(tree)
        )
        constants = dict(entry["parameters"]["constants"])
        operands = (data_dtype, weight_dtype or data_dtype)
        if record["tactic"]["swap_ab"]:
            operands = operands[::-1]
        # SM120 warp-MMA templates name their operand storage types *_fake_dtype
        # and have no CTA pair.
        sm120 = record["arch"] == "sm_120a"
        names = ("a_fake_dtype", "b_fake_dtype") if sm120 else ("a_dtype", "b_dtype")
        assert tuple(constants[name] for name in names) == operands
        assert constants["sf_cutlass_dtype"] == sf_dtype
        assert constants["block_size"] == str(block)
        assert int(constants.get("cta_group", "1")) == record["tactic"]["cta_group"]
        for name in ("cta_tile_mnk", "cluster_shape_mnk", "cgrp_tile_mnk"):
            geometry = ast.literal_eval(constants[name])
            assert len(geometry) == 3 and all(value > 0 for value in geometry)
        assert record["arch"] in ("sm_107a", "sm_120a")
        assert record["arch"] in constants["frost_compile_options"]
        if (
            record["contract"]["output_dtype"] in ("float4_e2m1fn_x2", "float8_e4m3fn")
            and record["tactic"]["store_mode"] == "tma"
        ):
            assert constants["epi_chunk_elems"] == "32"
        family = (
            record["arch"],
            record["op"],
            record["contract"]["output_dtype"],
            record["tactic"]["swap_ab"],
            record["tactic"]["store_mode"],
        )
        families.setdefault(family, set()).add(entry["path"])
        source_families.setdefault(entry["path"], set()).add(family)
        by_source.setdefault(entry["path"], set()).add(
            hashlib.sha256(concrete.encode()).hexdigest()
        )
        # Reintroduce the producer's section header for the extraction roundtrip.
        header = "# Block-scale config: test\n"
        concrete_with_header = concrete.replace(
            f"\n{entry['parameters']['constants'][0][0]} = ",
            "\n" + header + f"{entry['parameters']['constants'][0][0]} = ",
            1,
        )
        template, parameters = extract_template(
            concrete_with_header, swap_ab=record["tactic"]["swap_ab"]
        )
        assert ast.dump(ast.parse(render_source(template, parameters))) == ast.dump(
            tree
        )
    assert {r["op"] for r in records} == {
        *(f"block_scale_grouped_gemm1_{name}" for name in ACTIVATIONS),
        "block_scale_grouped_gemm2",
    }
    # Offline selection may retain different numbers of geometries and ABI
    # families. Every retained geometry of one family must share one source;
    # source paths must not accidentally mix activations or launch ABIs.
    assert all(len(paths) == 1 for paths in families.values())
    # Activations share a topology template; layout/store/output remain fixed.
    assert all(
        len({(v[0], v[2], v[3], v[4]) for v in variants}) == 1
        for variants in source_families.values()
    )
    # Per gated/non-gated topology: SM107 normal/swap-AB x STG/TMA plus fused
    # STG/TMA FC1, SM120 STG normal/swap-AB plus fused normal FC1; then FC2.
    for arch, per_topology, fc2 in (("sm_107a", 6, 4), ("sm_120a", 3, 2)):
        sources = {
            path
            for family, paths in families.items()
            if family[0] == arch
            for path in paths
        }
        assert len(sources) <= 2 * per_topology + fc2
    assert sum(len(digests) for digests in by_source.values()) == len(records)


def test_dtype_directories_are_symmetric_and_isolated():
    from flashinfer.fused_moe.backends.cudnn_frost import (
        runtime as shared,
    )
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        runtime,
    )

    assert mxfp8.runtime is shared
    assert nvfp4.runtime is shared
    assert mxfp8_mxfp4.runtime is shared
    assert runtime._load_kernel is shared._load_kernel
    assert runtime._read_source is shared._read_source
    assert Path(runtime.__file__).parent.name == "bf16"
    assert Path(mxfp8.__file__).parent.name == "mxfp8"
    assert Path(nvfp4.__file__).parent.name == "nvfp4"
    assert Path(mxfp8_mxfp4.__file__).parent.name == "mxfp8_mxfp4"
    root = mxfp8.runtime.artifact_root("mxfp8").parent
    assert runtime._artifact_roots() == (root / "bf16",)
    assert not (root / "sources").exists()
    assert not (root / runtime._MANIFEST).exists()
    for dtype in ("bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4"):
        support = importlib.import_module(
            f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.support"
        )
        assert Path(support.__file__).parent.name == dtype
        payload = read_manifest(root / dtype / runtime._MANIFEST)
        expected = {
            "bf16": "bfloat16",
            "mxfp8": "float8_e4m3fn",
            "nvfp4": "float4_e2m1fn_x2",
            "mxfp8_mxfp4": "float8_e4m3fn",
        }[dtype]
        assert all(k["contract"]["token_dtype"] == expected for k in payload["kernels"])
        assert all(
            (root / dtype / k["source"]["path"]).is_file() for k in payload["kernels"]
        )
    assert all(k.contract["token_dtype"] == "float8_e4m3fn" for k in mxfp8.discover())


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_mxfp8_auto_admission_uses_shared_geometry_without_dtype_leakage(name):
    _assert_mxfp8_auto_admission(name)


def _assert_mxfp8_auto_admission(name, mixed=False):
    from flashinfer.fused_moe import (
        CutlassMxfp8Config,
        CutlassMxfp8Mxfp4Config,
        QuantFormat,
    )

    dtype = "mxfp8_mxfp4" if mixed else "mxfp8"
    support = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.support"
    )
    backend = CutlassMxfp8Mxfp4Config if mixed else CutlassMxfp8Config

    config = MoEConfig(
        routing=RoutingConfig(num_experts=12, top_k=2),
        quant=QuantConfig(
            weight=QuantFormat.MXFP4 if mixed else QuantFormat.MXFP8,
            activation=QuantFormat.MXFP8,
        ),
        experts=ExpertConfig(intermediate_size=3072),
        backend=BackendOptions((backend(),)),
        activation=ACTIVATIONS[name](),
        execution=ExecutionConfig(tune_max_num_tokens=16384),
    )
    for tokens, expected in (
        (0, False),
        (1, True),
        (17, True),
        (12288, True),
        (12289, True),
    ):
        act = MoEActivationPack(
            torch.empty(tokens, 7168, dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty(tokens, 224, dtype=torch.uint8, device="meta"),
            torch.empty(tokens, 2, dtype=torch.int32, device="meta"),
            torch.empty(tokens, 2, dtype=torch.float32, device="meta"),
        )
        assert support.is_eligible(config, act, 107) == expected
        assert not support.is_eligible(config, act, 100)
        assert not support.is_eligible(
            config, replace(act, hidden_states_q=act.hidden_states_q.bfloat16()), 107
        )
        assert not support.is_eligible(replace(config, quant=QuantConfig()), act, 107)
        assert (
            support.is_eligible(
                replace(config, routing=RoutingConfig(num_experts=12, top_k=3)),
                act,
                107,
            )
            == expected
        )


def _block_scale_sf_address(row, col, cols):
    return (
        (row // 128 * ((cols + 3) // 4) + col // 4) * 512
        + row % 32 * 16
        + row % 128 // 32 * 4
        + col % 4
    )


def test_mxfp8_scale_layout_uneven_empty_groups_and_experts():
    _assert_block_scale_layout(mxfp8)


def _assert_block_scale_layout(runtime):
    rows, cols = 259, 8
    scales = (torch.arange(rows * cols) % 253).to(torch.uint8).reshape(rows, cols)
    offsets = torch.tensor([0, 0, 1, 128, 128, 130, 259, 259, 259], dtype=torch.int32)
    packed = runtime.pack_token_scales(scales, offsets).view(torch.uint8)
    assert (
        packed.numel() == runtime.segmented_scale_rows(rows, offsets.numel() - 1) * cols
    )
    base = 0
    starts = offsets.tolist()
    for begin, end in zip(starts, starts[1:], strict=False):
        for row in range(end - begin):
            for col in range(cols):
                assert (
                    packed[base + _block_scale_sf_address(row, col, cols)]
                    == scales[begin + row, col]
                )
        base += ((end - begin + 127) // 128) * 128 * cols
    assert torch.count_nonzero(packed[base:]) == 0
    weight = torch.stack([scales[:129], scales[130:]])
    packed_weight = runtime.pack_weight_scales(weight).view(torch.uint8)
    for expert in range(2):
        for row in range(129):
            for col in range(cols):
                assert (
                    packed_weight[expert, _block_scale_sf_address(row, col, cols)]
                    == weight[expert, row, col]
                )


@pytest.mark.parametrize("runtime", [mxfp8, nvfp4, mxfp8_mxfp4])
@pytest.mark.parametrize("boundaries", [[0, 1, 129], [0, 129, 129], [0, 0, 0]])
def test_block_scale_explicit_final_boundary(runtime, boundaries):
    # The backing token buffer has spare capacity which belongs to no group.
    scales = torch.arange(259 * 4).remainder(253).to(torch.uint8).reshape(259, 4)
    offsets = torch.tensor(boundaries, dtype=torch.int32)
    packed = runtime.pack_token_scales(scales, offsets).view(torch.uint8)
    expected = torch.zeros_like(packed)
    base = 0
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        for row in range(end - begin):
            for col in range(4):
                expected[base + _block_scale_sf_address(row, col, 4)] = scales[
                    begin + row, col
                ]
        base += ((end - begin + 127) // 128) * 128 * 4
    torch.testing.assert_close(packed, expected)
    # Updating unused capacity cannot change the packed groups.
    scales[boundaries[-1] :] = 255
    torch.testing.assert_close(
        runtime.pack_token_scales(scales, offsets).view(torch.uint8), expected
    )


@pytest.mark.parametrize("runtime", [mxfp8, nvfp4, mxfp8_mxfp4])
@pytest.mark.parametrize("boundaries", [[], [0]])
def test_block_scale_requires_start_and_end_boundaries(runtime, boundaries):
    with pytest.raises(ValueError, match="G\\+1"):
        runtime.pack_token_scales(
            torch.ones(10, 4, dtype=torch.uint8),
            torch.tensor(boundaries, dtype=torch.int32),
        )


@frost_gpu
@pytest.mark.parametrize("runtime", [mxfp8, nvfp4, mxfp8_mxfp4])
@pytest.mark.parametrize("groups", [32, 40, 64])
@pytest.mark.parametrize("swap", [False, True])
def test_block_scale_final_boundary_across_scheduler_warps(runtime, groups, swap):
    # Exercise the scheduler's initial warp, subsequent warp, and exact-warp
    # boundary with G > E. Only the final group has tokens.
    s, k, n, e = 259, 128, 128, 8
    kernel = next(
        kernel
        for kernel in runtime.discover()
        if not kernel.fc1 and kernel.swap_ab == swap and kernel.arch == _device_arch()
    )
    nv = runtime is nvfp4
    mixed = runtime is mxfp8_mxfp4
    x = torch.full(
        (s, k // (2 if nv else 1)),
        0x22 if nv else 1,
        dtype=torch.uint8 if nv else torch.float8_e4m3fn,
        device="cuda",
    )
    weight = torch.full(
        (e, n, k // (2 if nv or mixed else 1)),
        0x22 if nv or mixed else 1,
        dtype=torch.uint8 if nv or mixed else torch.float8_e4m3fn,
        device="cuda",
    )
    block = 16 if nv else 32
    sf_dtype = torch.float8_e4m3fn if nv else torch.uint8
    x_sf = torch.full((s, k // block), 1 if nv else 127, dtype=sf_dtype, device="cuda")
    w_sf = torch.full(
        (e, n, k // block), 1 if nv else 127, dtype=sf_dtype, device="cuda"
    )
    offsets = torch.zeros(groups + 1, device="cuda", dtype=torch.int32)
    offsets[-1] = 193
    packed_sf = runtime.pack_token_scales(x_sf, offsets)
    out = torch.full((s, n), -7, dtype=torch.bfloat16, device="cuda")
    plan_type = (
        runtime.PreparedNvfp4GroupedGemm
        if nv
        else (
            runtime.PreparedMxfp8Mxfp4GroupedGemm
            if mixed
            else runtime.PreparedMxfp8GroupedGemm
        )
    )
    # G exceeds E, and only the last group is nonempty. Alpha must be indexed
    # by group_idx, rather than the scheduler's mapped expert_idx.
    kwargs = {
        "gemm_scales": (
            torch.arange(1, groups + 1, device="cuda", dtype=torch.float32),
        )
    }
    plan = plan_type(
        kernel,
        x,
        (weight,),
        offsets,
        packed_sf,
        (runtime.pack_weight_scales(w_sf),),
        out,
        **kwargs,
    )
    plan()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    for end in (193, 129, 0):
        offsets[-1] = end
        packed_sf.copy_(runtime.pack_token_scales(x_sf, offsets))
        out.fill_(-7)
        graph.replay()
        expected = torch.full_like(out, -7)
        expected[:end] = k * groups
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_mxfp8_segmented_scale_capacity_covers_routing_partitions():
    _assert_segmented_scale_capacity(mxfp8)


def _assert_segmented_scale_capacity(runtime):
    def partitions(rows, groups):
        if groups == 1:
            yield (rows,)
        else:
            for first in range(rows + 1):
                for remaining in partitions(rows - first, groups - 1):
                    yield (first, *remaining)

    # Exhaustively cover short batches, including more experts than rows and
    # empty experts; the allocation bound must be attainable as well as safe.
    for rows in range(17):
        for groups in range(1, 6):
            actual = max(
                sum((count + 127) // 128 * 128 for count in counts)
                for counts in partitions(rows, groups)
            )
            assert runtime.segmented_scale_rows(rows, groups) == actual
    for rows, groups in (
        (129, 8),
        (259, 8),
        (12288 * 2, 8),
        (12288 * 4, 12),
        (12288 * 6, 64),
    ):
        capacity = runtime.segmented_scale_rows(rows, groups)
        # Extreme skew maximizes independent padding when all other experts
        # have one token. Uniform and single-expert routes cover other edges.
        skewed = [1] * (groups - 1) + [rows - groups + 1]
        uniform = [
            rows // groups + (expert < rows % groups) for expert in range(groups)
        ]
        concentrated = [rows] + [0] * (groups - 1)
        assert sum((count + 127) // 128 * 128 for count in skewed) == capacity
        for counts in (uniform, concentrated):
            assert sum((count + 127) // 128 * 128 for count in counts) <= capacity


@pytest.mark.parametrize("offsets", [[1, 5], [0, 9, 4], [0, 11]])
def test_mxfp8_invalid_offsets(offsets):
    with pytest.raises(ValueError, match="offsets"):
        mxfp8.pack_token_scales(
            torch.ones(10, 4, dtype=torch.uint8),
            torch.tensor(offsets, dtype=torch.int32),
        )


def test_mxfp8_manifest_and_launch_abi_without_cudnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    _assert_block_scale_launch_abi(mxfp8, packed=False, block=32)


def _assert_block_scale_launch_abi(runtime, *, packed, block, packed_weights=None):
    packed_weights = packed if packed_weights is None else packed_weights
    kernels = runtime.discover()
    for kernel in kernels:
        s, n, k, e, groups = 259, 256, 128, 8, 16
        x = torch.empty(
            s,
            k // (2 if packed else 1),
            dtype=torch.uint8 if packed else torch.float8_e4m3fn,
        )
        w = tuple(
            torch.empty(
                e,
                n,
                k // (2 if packed_weights else 1),
                dtype=torch.uint8 if packed_weights else torch.float8_e4m3fn,
            )
            for _ in range(2 if kernel.gated else 1)
        )
        a_sf = torch.empty(
            runtime.segmented_scale_rows(s, groups) * (k // block), dtype=torch.uint8
        )
        b_sf = tuple(torch.empty(e, n * k // block, dtype=torch.uint8) for _ in w)
        offsets = torch.zeros(groups + 1, dtype=torch.int32)
        out = torch.empty(s, n, dtype=torch.bfloat16)
        workspace = torch.empty(kernel.workspace_bytes, dtype=torch.uint8)
        scale = torch.ones(1, 1, 1)
        activation_scales = (
            {
                "gate_scale": torch.full((1, 1, 1), 4.0),
                "linear_scale": torch.full((1, 1, 1), 25.0),
            }
            if kernel.activation == "situ"
            else {}
        )
        gemm_scales = tuple(torch.arange(groups, dtype=torch.float32) + 0.5 for _ in w)
        kwargs = {"gemm_scales": gemm_scales}
        args = runtime._launch_arguments(
            kernel,
            x,
            w,
            offsets,
            a_sf,
            b_sf,
            out,
            workspace,
            scale,
            activation_scales,
            **kwargs,
        )
        tree = ast.parse(kernel.source_path.read_text())
        host = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_host"
        )
        names = [arg.arg for arg in host.args.args]
        assert len(args) + 1 == len(names), (
            kernel.artifact_id,
            kernel.activation,
            kernel.gated,
            kernel.launch_tail,
            names,
        )
        assert args[0][:5] == (
            (n, s, k, e, groups) if kernel.swap_ab else (s, n, k, e, groups)
        )
        ptrs = {
            name: arg.data_ptr() for name, arg in zip(names[1:], args[1:], strict=False)
        }
        tensors = dict(zip(names[1:-1], args[1:], strict=True))
        assert args[0][3:5] == (e, groups)
        assert tensors["first_token_offset"].shape == (groups + 1,)
        token_arg = tensors["b_0" if kernel.swap_ab else "a_0"]
        weight_arg = tensors["a_0" if kernel.swap_ab else "b_0"]
        assert token_arg.shape == (s, k // (2 if packed else 1), 1)
        assert weight_arg.shape == (n, k // (2 if packed_weights else 1), e)
        assert token_arg.dtype == (
            torch.float4_e2m1fn_x2 if packed else torch.float8_e4m3fn
        )
        assert weight_arg.dtype == (
            torch.float4_e2m1fn_x2 if packed_weights else torch.float8_e4m3fn
        )
        assert ptrs["a_0"] == (w[0] if kernel.swap_ab else x).data_ptr()
        assert ptrs["sfa_0"] == (b_sf[0] if kernel.swap_ab else a_sf).data_ptr()
        assert ptrs["sfb_0"] == (a_sf if kernel.swap_ab else b_sf[0]).data_ptr()
        if kernel.gated:
            assert ptrs["a_1" if kernel.swap_ab else "b_1"] == w[1].data_ptr()
        if kernel.fc1:
            assert ptrs["scale"] == scale.data_ptr()
        for name, scalar in activation_scales.items():
            assert ptrs[name] == scalar.data_ptr()
        alpha_names = (
            ("gate_alpha", "up_alpha")
            if kernel.gated
            else ("gate_alpha" if kernel.fc1 else "alpha",)
        )
        for name, scalar in zip(alpha_names, gemm_scales, strict=True):
            assert ptrs[name] == scalar.data_ptr()
            assert tensors[name].shape == (groups, 1, 1)


@pytest.mark.parametrize(
    "change",
    [
        {"block_size": 16},
        {"global_scale_dtype": "bfloat16"},
        {"global_scale_layout": "per_expert"},
        {"global_scale_position": "after_activation"},
    ],
)
@pytest.mark.parametrize("schema_version", [2, 3])
def test_mxfp8_reject_incompatible_scale_contract(change, tmp_path, schema_version):
    payload = {
        "schema_version": 2,
        "kernels": [
            MXFP8_RECORDS[0] | {"contract": MXFP8_RECORDS[0]["contract"] | change}
        ],
    }
    if schema_version == 3:
        payload = normalize_manifest(payload)
    (tmp_path / "cudnn_frost_selected_kernels.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="numerical contract"):
        mxfp8.discover(tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index", range(len(mxfp8.discover())))
def test_mxfp8_grouped_kernel_and_graph_replay(index, monkeypatch):
    kernel = mxfp8.discover()[index]
    if kernel.arch != _device_arch():
        pytest.skip(f"packaged {kernel.arch} source")
    _assert_mxfp8_grouped_kernel_and_graph(mxfp8, kernel, monkeypatch)


def _assert_mxfp8_grouped_kernel_and_graph(runtime, kernel, monkeypatch, mixed=False):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    torch.manual_seed(19)
    s, n, k, e = 259, 256, 128, 8
    x = (torch.randn(s, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    if mixed:
        w = tuple(
            torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
            for _ in range(2 if kernel.gated else 1)
        )
        for weight in w:
            # Include both signs, signed zero, maxima and different even/odd K.
            weight[:, 0, :6] = torch.tensor(
                [0x00, 0x88, 0x77, 0xFF, 0x7F, 0xF7], device="cuda"
            )
    else:
        w = tuple(
            (torch.randn(e, n, k, device="cuda") * 0.1).to(x.dtype)
            for _ in range(2 if kernel.gated else 1)
        )
    if kernel.activation in ("situ", "swiglu_step"):
        # Exercise saturation/clamps, rather than only their small-signal limits.
        x.copy_((x.float() * 16).to(x.dtype))
        if not mixed:
            for weight in w:
                weight.copy_((weight.float() * 16).to(weight.dtype))
    a_sf = torch.randint(125, 128, (s, k // 32), dtype=torch.uint8, device="cuda")
    b_sf = tuple(
        torch.randint(125, 128, (e, n, k // 32), dtype=torch.uint8, device="cuda")
        for _ in w
    )
    # Repeated expert mapping (G = 2E) must retain each group's own alpha.
    offsets = torch.tensor(
        [0, 0, 1, 128, 128, 130, 193, 193, 193, 193, 194, 225, 225, 227, 259, 259, 259],
        dtype=torch.int32,
        device="cuda",
    )
    out = torch.empty(s, n, dtype=torch.bfloat16, device="cuda")
    packed_a = runtime.pack_token_scales(a_sf, offsets)
    scale = torch.full((1, 1, 1), 0.5, device="cuda") if kernel.fc1 else None
    plan_type = (
        runtime.PreparedMxfp8Mxfp4GroupedGemm
        if mixed
        else runtime.PreparedMxfp8GroupedGemm
    )
    alphas = tuple(
        torch.tensor(
            [0.0, -0.5, 0.25, 1.5, -1.0, 2.0, -0.25, 0.75], device="cuda"
        ).repeat(2)
        * torch.tensor([1.0] * e + [-1.75] * e, device="cuda")
        * (j + 1)
        for j in range(len(w))
    )
    plan = plan_type(
        kernel,
        x,
        w,
        offsets,
        packed_a,
        tuple(runtime.pack_weight_scales(sf) for sf in b_sf),
        out,
        scale=scale,
        gemm_scales=alphas,
    )

    def reference():
        xd = x.float() * a_sf.view(torch.float8_e8m0fnu).float().repeat_interleave(
            32, -1
        )
        wd = [
            (_fp4_values(v) if mixed else v.double())
            * sf.view(torch.float8_e8m0fnu).double().repeat_interleave(32, -1)
            for v, sf in zip(w, b_sf, strict=False)
        ]
        expected = torch.full((s, n), -7.0, device="cuda")
        starts = offsets.tolist()
        for group, (begin, end) in enumerate(zip(starts, starts[1:], strict=False)):
            expert = group % e
            values = (xd[begin:end].double() @ wd[0][expert].double().T) * alphas[0][
                group
            ]
            if kernel.gated:
                up = (xd[begin:end].double() @ wd[1][expert].double().T) * alphas[1][
                    group
                ]
                # The shared reference expects [up, gate] ordering.
                values = torch.cat((up, values), dim=-1)
            if kernel.fc1:
                from tests.moe.utils import compute_reference_activation

                values = (
                    compute_reference_activation(
                        values.float(),
                        ACTIVATIONS[kernel.activation](),
                        n,
                        out_dtype=torch.float32,
                    )
                    * scale.flatten()[0]
                )
            expected[begin:end] = values
        return expected.bfloat16()

    plan()
    torch.testing.assert_close(out, reference(), rtol=2e-2, atol=2e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    x.copy_((torch.randn(s, k, device="cuda") * 0.1).to(x.dtype))
    offsets.copy_(
        torch.tensor(
            [
                0,
                2,
                2,
                2,
                31,
                129,
                129,
                129,
                129,
                129,
                130,
                160,
                160,
                160,
                259,
                259,
                259,
            ],
            device="cuda",
            dtype=torch.int32,
        )
    )
    packed_a.copy_(runtime.pack_token_scales(a_sf, offsets))
    if mixed:
        for weight in w:
            weight.bitwise_xor_(0x88)
    if scale is not None:
        scale.fill_(0.25)
    for j, alpha in enumerate(alphas):
        alpha.mul_(-0.5 if j == 0 else 1.5)
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, reference(), rtol=2e-2, atol=2e-4)

    # The final offset is device data, not the static token-buffer capacity.
    # Cover a nonempty last expert, an empty last expert, and no active rows.
    for boundaries in (
        [0, 0, 1, 128, 128, 130, 130, 130, 130, 130, 131, 160, 160, 160, 160, 160, 193],
        [0, 2, 2, 2, 31, 129, 129, 129, 129, 129, 130, 160, 160, 160, 193, 193, 193],
        [0] * 17,
    ):
        offsets.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
        packed_a.copy_(runtime.pack_token_scales(a_sf, offsets))
        out.fill_(-7)
        graph.replay()
        torch.testing.assert_close(out, reference(), rtol=2e-2, atol=2e-4)
        assert torch.all(out[boundaries[-1] :] == -7)


def _block_scale_linear_scales(packed, rows, cols):
    """Independent inverse of the canonical 128x4 weight/input SF layout."""
    return (
        packed.view(torch.uint8)
        .reshape(-1, cols // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(-1, cols)[:rows]
    )


def _mxfp8_moe_reference(act, weights, activation, swizzled=False, mixed=False):
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize
    from tests.moe.utils import compute_reference_activation

    def dequant(data, scales, packed=False):
        return (_fp4_values(data) if packed else data.double()) * scales.view(
            torch.float8_e8m0fnu
        ).double().repeat_interleave(32, -1)

    x, sf = act.hidden_states_q, act.hidden_states_scale
    if swizzled:
        sf = _block_scale_linear_scales(sf, x.shape[0], x.shape[1] // 32)
    x = dequant(x, sf)
    ids, scores = act.topk_ids, act.topk_weights
    w = weights.get_view("cutlass_mxfp8_mxfp4" if mixed else "cutlass_mxfp8")
    w1, w2 = w["fc1_expert_weights"], w["fc2_expert_weights"]
    e, n, _ = w1.shape
    h, i = x.shape[-1], w2.shape[-1] * (2 if mixed else 1)
    expanded = torch.zeros((*ids.shape, h), device=x.device, dtype=torch.float32)
    for expert in range(e):
        token, slot = torch.where(ids == expert)
        if not token.numel():
            continue
        up_gate = dequant(
            w1[expert],
            _block_scale_linear_scales(w["fc1_expert_scales"][expert], n, h // 32),
            packed=mixed,
        )
        down = dequant(
            w2[expert],
            _block_scale_linear_scales(w["fc2_expert_scales"][expert], h, i // 32),
            packed=mixed,
        )
        mid = compute_reference_activation(
            (x[token] @ up_gate.T).float() * w["fc1_input_scale"][expert],
            activation,
            i,
        )
        q, sf = mxfp8_quantize(mid, is_sf_swizzled_layout=False, alignment=32)
        mid = dequant(q, sf.reshape(-1, i // 32)[: token.numel()])
        value = ((mid @ down.T).float() * w["fc2_input_scale"][expert]).bfloat16()
        expanded[token, slot] = value.float() * scores[token, slot, None]
    return expanded.sum(dim=1).bfloat16()


def _pick_stage_pair(pool):
    # Prefer different retained launch ABIs without assuming selection retained
    # a particular store, orientation, CTAMMA group, or geometry. An SM120
    # fused FC1 family may retain a single tile.
    assert pool
    if len(pool) == 1:
        return tuple(pool)
    first = pool[0]
    second = max(
        pool[1:],
        key=lambda kernel: (
            kernel.swap_ab != first.swap_ab,
            kernel.tactic_metadata["store_mode"] != first.tactic_metadata["store_mode"],
        ),
    )
    return first, second


@frost_gpu
@pytest.mark.parametrize("name", ACTIVATIONS)
@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("fc1_output", ["bf16", "fp8_stg", "fp8_tma"])
def test_mxfp8_moe_full_pipeline_and_graph(name, swizzled, monkeypatch, fc1_output):
    _assert_mxfp8_moe_full_pipeline_and_graph(
        name, swizzled, monkeypatch, fc1_output=fc1_output
    )


def _assert_mxfp8_moe_full_pipeline_and_graph(
    name,
    swizzled,
    monkeypatch,
    mixed=False,
    fc1_output="bf16",
    geometry=(17, 256, 256, 4, 2),
    require_fma=False,
):
    from flashinfer.fused_moe import (
        CutlassMxfp8Config,
        CutlassMxfp8Mxfp4Config,
        QuantFormat,
    )

    dtype = "mxfp8_mxfp4" if mixed else "mxfp8"
    moe = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.moe"
    )
    backend = CutlassMxfp8Mxfp4Config if mixed else CutlassMxfp8Config
    if fc1_output != "bf16":
        _skip_sm120_tma(fc1_output.removeprefix("fp8_"))

    monkeypatch.setitem(sys.modules, "cudnn", None)
    torch.manual_seed(77)
    t, h, i, e, k = geometry
    activation = ACTIVATIONS[name]()
    quant = QuantConfig(
        QuantFormat.MXFP4 if mixed else QuantFormat.MXFP8,
        QuantFormat.MXFP8,
        swizzled_scale_factors=swizzled,
    )
    config = replace(
        bf16_config(topk=k, experts=e, intermediate=i),
        quant=quant,
        activation=activation,
        backend=BackendOptions((backend(),)),
    )
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16) * 0.1
    w1 = (
        torch.randn(
            e,
            (2 if activation.is_gated else 1) * i,
            h,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.1
    if name in ("situ", "swiglu_step"):
        x *= 8
        w1 *= 16
    xq, xsf = backend.prepare_activations(x, quant=quant)
    view = backend.prepare_weights(
        w1,
        w2,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size=i,
        activation=activation,
    )
    view["fc1_input_scale"].copy_(
        torch.tensor([0.0, -0.5, 1.25, 2.0], device="cuda").repeat(e // 4)
    )
    view["fc2_input_scale"].copy_(
        torch.tensor([1.5, -0.75, 0.0, 0.5], device="cuda").repeat(e // 4)
    )
    ids = torch.randint(0, e // 2, (t, k), device="cuda", dtype=torch.int32)
    scores = torch.rand(t, k, device="cuda")
    scores /= scores.sum(-1, keepdim=True)
    act = MoEActivationPack(xq, xsf, ids, scores)
    weights = MoEWeightPack({f"cutlass_{dtype}": view})
    first, second = moe._kernels(t * k, h, i, e, xq.device, activation)

    if fc1_output != "bf16":
        first, _ = moe._kernels(
            t * k, h, i, e, xq.device, activation, quantized_output=True
        )
        first = tuple(
            kernel
            for kernel in first
            if kernel.tactic_metadata["store_mode"] == fc1_output.removeprefix("fp8_")
        )
    selected = _pick_stage_pair(first), _pick_stage_pair(second)
    # Test full launch mechanics on small allocations independently of the
    # measured model-size shortlist. Admission is covered separately below.
    capture_selection = moe._capture_selection
    monkeypatch.setattr(
        moe,
        "_capture_selection",
        lambda *args: capture_selection(*args)._replace(
            select_kernels=lambda shape: selected
        ),
    )
    runner_type = (
        moe.CudnnFrostMxfp8Mxfp4MoeRunner if mixed else moe.CudnnFrostMxfp8MoeRunner
    )
    runner = runner_type(config, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    # Every Tensor Core pair, optionally a small-token FMA.
    pairs = len(selected[0]) * len(selected[1])
    assert len(tactics) in (pairs, pairs + 1)
    if require_fma:
        assert len(tactics) == pairs + 1 and "fma" in tactics[-1][0]
    assert inputs[9] is view["fc1_input_scale"]
    assert inputs[10] is view["fc2_input_scale"]
    assert inputs[6].data_ptr() == view["fc1_expert_scales"].data_ptr()
    assert runner.get_valid_tactics(list(inputs), None) == tactics
    reference = _mxfp8_moe_reference(act, weights, activation, swizzled, mixed)
    assert reference.float().norm() > 0
    mult = 2 if activation.is_gated else 1
    sf_cols = h // (32 if mixed else 128)
    packed_sf1 = (
        view["fc1_expert_scales"]
        .reshape(e, mult, i, sf_cols)
        .transpose(0, 1)
        .contiguous()
    )
    packed_weights = MoEWeightPack(
        {f"cutlass_{dtype}": view | {"fc1_expert_scales": packed_sf1}}
    )
    packed_inputs = runner.pack_inputs(act, packed_weights)
    assert packed_inputs[6] is packed_sf1
    assert packed_inputs.launch_state is inputs.launch_state
    for tactic in tactics:
        out = runner.forward(packed_inputs, tactic)
        assert (
            out.float() - reference.float()
        ).norm() / reference.float().norm() < 0.02
    for tactic in tactics:
        out = runner.forward(inputs, tactic)
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out) > 0
        assert (
            out.float() - reference.float()
        ).norm() / reference.float().norm() < 0.02
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.forward(list(inputs), tactic, launch_state=inputs.launch_state)
        ids.fill_(e - 1)
        if t == 1:
            # Keep a valid route for the FMA-sized geometry while exercising
            # both invalid-ID guards on its other slots.
            ids[0, 1] = -1
            if k > 2:
                ids[0, 2] = e
        else:
            ids[::3] = -1
            ids[1::3] = e
        scores.mul_(0.5)
        xq.copy_((xq.float() * 0.5).to(torch.float8_e4m3fn))
        # E8M0 bytes, including the swizzled input, are consumed on replay.
        xsf.view(torch.uint8).add_(1)
        # Canonical weight SF buffers are read directly, including inference
        # tensors; replay observes updates without another pack or capture.
        view["fc1_expert_scales"].view(torch.uint8).add_(1)
        view["fc2_expert_scales"].view(torch.uint8).add_(1)
        view["fc1_input_scale"].mul_(-0.5)
        view["fc2_input_scale"].mul_(1.5)
        reference = _mxfp8_moe_reference(act, weights, activation, swizzled, mixed)
        assert reference.float().norm() > 0
        out.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(out).all()
        assert (
            out.float() - reference.float()
        ).norm() / reference.float().norm() < 0.02
    with pytest.raises(ValueError, match="launch_state"):
        runner.forward(list(inputs))
    if name == "swiglu" and not swizzled:
        old_scales = inputs[6].clone()
        view["fc1_expert_scales"].view(torch.uint8).add_(1)
        new_inputs = runner.pack_inputs(act, weights)
        assert new_inputs[6].data_ptr() == inputs[6].data_ptr()
        assert not torch.equal(inputs[6], old_scales)
        reference = _mxfp8_moe_reference(act, weights, activation, swizzled, mixed)
        out = runner.forward(inputs)
        assert (
            out.float() - reference.float()
        ).norm() / reference.float().norm() < 0.02
    # Replacing an inference tensor or mutating a regular alpha must be
    # consumed by the next pack, without imposing an identity-scale contract.
    if torch.is_inference(view["fc1_input_scale"]):
        view["fc1_input_scale"] = view["fc1_input_scale"] * 2
    else:
        view["fc1_input_scale"].mul_(2)
    updated = runner.pack_inputs(act, weights)
    reference = _mxfp8_moe_reference(act, weights, activation, swizzled, mixed)
    out = runner.forward(updated)
    assert (out.float() - reference.float()).norm() / reference.float().norm() < 0.02


@frost_gpu
def test_mxfp8_moe_inference_mode_preparation_and_graph(monkeypatch):
    with torch.inference_mode():
        test_mxfp8_moe_full_pipeline_and_graph("swiglu", False, monkeypatch, "fp8_stg")


def test_mxfp8_moe_shortlist_uses_measured_two_by_two_buckets(
    monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.mxfp8 import (
        moe,
    )

    _assert_moe_shortlist_buckets(moe, monkeypatch)


def _assert_moe_shortlist_buckets(moe, monkeypatch):
    moe.runtime.clear_artifact_cache()
    key = ("sm_107a", "swiglu", 8, 4096, 14336, 2)
    first = tuple(
        SimpleNamespace(artifact_id=f"first{j}", tactic_metadata={"tile": f"tile{j}"})
        for j in range(3)
    )
    second = tuple(SimpleNamespace(artifact_id=f"second{j}") for j in range(3))
    monkeypatch.setattr(moe.common, "_arch_for", lambda device: "sm_107a")
    monkeypatch.setattr(
        moe,
        "_matching_kernels",
        lambda *args, **kwargs: (
            ((), ()) if kwargs.get("quantized_output") else (first, second)
        ),
    )
    table = {key: {128: (("first2", "first0"), ("second1", "second2"))}}

    @functools.lru_cache(maxsize=8)
    def read(roots):
        return table

    monkeypatch.setattr(moe, "_read", read)
    from flashinfer.fused_moe.backends.cudnn_frost import (
        shortlist,
    )

    monkeypatch.setattr(shortlist, "_read", read)
    fallback = ((first[0],), (second[0],))
    from flashinfer.fused_moe.backends.cudnn_frost import heuristics

    monkeypatch.setattr(heuristics, "select_stages", lambda *args, **kwargs: fallback)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=128),
    )
    args = (4096, 14336, 8, 2, "cuda", SwiGLU())
    for tokens in (1, 17, 127, 128):
        assert moe._selected_kernels(tokens, *args) == (
            (first[2], first[0]),
            (second[1], second[2]),
        )
    for tokens in (0, (1 << 20) + 1):
        assert moe._selected_kernels(tokens, *args) == ((), ())
    for tokens in (129, 12288, 12289):
        assert moe._selected_kernels(tokens, *args) == fallback
    assert moe._selected_kernels(128, 4096, 14336, 9, 2, "cuda", SwiGLU()) == fallback
    table[key][128] = (("first2",), ("second1", "second2"))
    moe._selected_kernels_cached.cache_clear()
    assert moe._selected_kernels(128, *args) == ((first[2],), (second[1], second[2]))
    table.clear()
    moe._selected_kernels_cached.cache_clear()
    assert moe._selected_kernels(128, *args) == fallback
    moe._selected_kernels_cached.cache_clear()


def test_mxfp8_packaged_shortlists_resolve_all_profiles(
    monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.mxfp8 import (
        moe,
    )

    _assert_packaged_shortlists_resolve_all_profiles(moe, monkeypatch)


def test_mxfp8_plan_workspaces_preserve_prior_allocations(monkeypatch):
    from flashinfer.fused_moe.backends.cudnn_frost.mxfp8 import (
        moe,
    )

    required_sizes = iter((257, 301, 400, 129, 513, 600, 599, 450, 350, 401, 450, 449))
    retained_modules = []

    def make_runner(*args):
        module = {
            "workspace_size": lambda capacity: next(required_sizes),
            "run": lambda *unused: None,
        }
        retained_modules.append(module)
        return module

    monkeypatch.setattr(
        moe, "_module", lambda arch: SimpleNamespace(make_runner=make_runner)
    )
    monkeypatch.setattr(moe.common, "_arch_for", lambda device: "sm_107a")
    monkeypatch.setattr(moe.common, "_load_kernel", lambda kernel, device: kernel)
    pool = {}
    executor_cache = {}
    first = tuple(
        SimpleNamespace(
            tactic=("fc1", j),
            workspace_bytes=128,
            gated=True,
            launch_tail=("output", "scale", "gate_alpha", "up_alpha"),
            swap_ab=bool(j),
        )
        for j in range(2)
    )
    second = tuple(
        SimpleNamespace(
            tactic=("fc2", j),
            workspace_bytes=128,
            launch_tail=("output", "alpha"),
            swap_ab=bool(j),
        )
        for j in range(2)
    )

    def prepare(tokens):
        return moe._Plans(
            tokens,
            256,
            256,
            4,
            2,
            torch.device("cpu"),
            first,
            second,
            pool,
            executor_cache,
        )

    before = prepare(16)
    before.workspace.fill_(7)
    original = before.workspace.data_ptr()
    grown = prepare(128)
    assert grown.workspace.data_ptr() != original
    assert before.workspace.data_ptr() == original
    assert torch.all(before.workspace == 7)
    reused = prepare(17)
    assert reused.workspace.data_ptr() == original
    assert set(pool) == {512, 1024}
    assert len(before.launches) == len(grown.launches) == len(reused.launches) == 4
    # Capacities share the same fixed native executors, and each packed state
    # retains the Modules whose run Functions borrow them.
    assert len(retained_modules) == len(executor_cache) == 4
    assert list(before.plans.values()) == retained_modules[:4]
    assert list(grown.plans.values()) == retained_modules
    assert list(reused.plans.values()) == retained_modules


def test_nvfp4_geometry_sharing_and_roundtrip_without_cudnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    root = nvfp4.runtime.artifact_root("nvfp4")
    records = read_manifest(root / nvfp4.runtime._MANIFEST)["kernels"]
    _assert_block_scale_geometry_roundtrip(
        root, records, "cutlass.Float4E2M1FNx2", "cutlass.Float8E4M3FN", 16
    )


def test_nvfp4_scale_layout_uneven_empty_groups_and_experts():
    _assert_block_scale_layout(nvfp4)


def test_nvfp4_segmented_scale_capacity_covers_routing_partitions():
    _assert_segmented_scale_capacity(nvfp4)


@pytest.mark.parametrize("offsets", [[1, 5], [0, 9, 4], [0, 11]])
def test_nvfp4_invalid_offsets(offsets):
    with pytest.raises(ValueError, match="offsets"):
        nvfp4.pack_token_scales(
            torch.ones(10, 8, dtype=torch.float8_e4m3fn),
            torch.tensor(offsets, dtype=torch.int32),
        )


def test_nvfp4_manifest_and_launch_abi_without_cudnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    _assert_block_scale_launch_abi(nvfp4, packed=True, block=16)


@pytest.mark.parametrize(
    "change",
    [
        {"block_size": 32},
        {"elements_per_byte": 1},
        {"scale_dtype": "float8_e8m0fnu"},
        {"global_scale_position": "after_activation"},
    ],
)
@pytest.mark.parametrize("schema_version", [2, 3])
def test_nvfp4_reject_incompatible_numerical_contract(change, tmp_path, schema_version):
    root = nvfp4.runtime.artifact_root("nvfp4")
    record = read_manifest(root / nvfp4.runtime._MANIFEST)["kernels"][0]
    payload = {
        "schema_version": 2,
        "kernels": [record | {"contract": record["contract"] | change}],
    }
    if schema_version == 3:
        payload = normalize_manifest(payload)
    (tmp_path / nvfp4.runtime._MANIFEST).write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="numerical contract"):
        nvfp4.discover(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"intermediate_dtype": "float32"},
        {"output_scale_layout": "F8_128x4"},
        {"quant_scale_dtype": "bfloat16"},
    ],
)
@pytest.mark.parametrize("store", ["stg", "tma"])
@pytest.mark.parametrize("schema_version", [2, 3])
def test_nvfp4_fused_manifest_rejects_changed_quantization(
    change, store, tmp_path, schema_version
):
    root = nvfp4.runtime.artifact_root("nvfp4")
    records = read_manifest(root / nvfp4.runtime._MANIFEST)["kernels"]
    record = next(
        r
        for r in records
        if r["contract"]["output_dtype"] == "float4_e2m1fn_x2"
        and r["tactic"]["store_mode"] == store
    )
    payload = {
        "schema_version": 2,
        "kernels": [record | {"contract": record["contract"] | change}],
    }
    if schema_version == 3:
        payload = normalize_manifest(payload)
    (tmp_path / nvfp4.runtime._MANIFEST).write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="numerical contract"):
        nvfp4.discover(tmp_path, quantized_output=True)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_nvfp4_auto_admission_uses_logical_hidden_size(name):
    from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import (
        support,
    )
    from flashinfer.fused_moe import CutlassNvfp4Config, QuantFormat

    config = MoEConfig(
        routing=RoutingConfig(num_experts=12, top_k=2),
        quant=QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=3072),
        backend=BackendOptions((CutlassNvfp4Config(),)),
        activation=ACTIVATIONS[name](),
        execution=ExecutionConfig(tune_max_num_tokens=16384),
    )
    for tokens, expected in (
        (0, False),
        (1, True),
        (17, True),
        (12288, True),
        (12289, True),
    ):
        act = MoEActivationPack(
            torch.empty(tokens, 7168 // 2, dtype=torch.uint8, device="meta"),
            torch.empty(tokens, 7168 // 16, dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty(tokens, 2, dtype=torch.int32, device="meta"),
            torch.empty(tokens, 2, device="meta"),
        )
        assert support.is_eligible(config, act, 107) == expected
        assert not support.is_eligible(config, act, 100)
        assert not support.is_eligible(replace(config, quant=QuantConfig()), act, 107)
        assert not support.is_eligible(
            config, replace(act, hidden_states_q=act.hidden_states_q.bfloat16()), 107
        )
        assert not support.is_eligible(
            config,
            replace(
                act,
                hidden_states_q=torch.empty(
                    tokens, 7169, dtype=torch.uint8, device="meta"
                ),
            ),
            107,
        )


def _fp4_values(data):
    # Independent E2M1 decoder: the low nibble is the even logical K element.
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
        device=data.device,
    )
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2)
    return values[codes.long()]


def _nvfp4_dequant(data, scales):
    return _fp4_values(data) * scales.view(
        torch.float8_e4m3fn
    ).double().repeat_interleave(16, -1)


@frost_gpu
@pytest.mark.parametrize("name", [*ACTIVATIONS, "fc2"])
@pytest.mark.parametrize(
    "swap_ab,store", [(False, "stg"), (False, "tma"), (True, "stg"), (True, "tma")]
)
def test_nvfp4_grouped_kernel_extremes_alpha_and_graph(
    name, swap_ab, store, monkeypatch
):
    from tests.moe.utils import compute_reference_activation

    _skip_sm120_tma(store)
    monkeypatch.setitem(sys.modules, "cudnn", None)
    op = (
        "block_scale_grouped_gemm2"
        if name == "fc2"
        else f"block_scale_grouped_gemm1_{name}"
    )
    kernels = [
        kernel
        for kernel in nvfp4.discover()
        if kernel.op == op
        and kernel.arch == _device_arch()
        and kernel.swap_ab == swap_ab
        and kernel.tactic_metadata["store_mode"] == store
    ]
    if not kernels:
        pytest.skip("this launch ABI is not retained in the measured shortlist")
    kernel = kernels[0]
    torch.manual_seed(131)
    s, n, k, e = 259, 256, 128, 8
    x = torch.randint(0, 256, (s, k // 2), device="cuda", dtype=torch.uint8)
    w = tuple(
        torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
        for _ in range(2 if kernel.gated else 1)
    )
    # Include both signed zeros, maximum magnitudes, and opposite-sign pairs.
    x[0, :6] = torch.tensor(
        [0x00, 0x88, 0x77, 0xFF, 0x7F, 0xF7], device="cuda", dtype=torch.uint8
    )
    sf_values = torch.tensor([0.03125, 0.0625, 0.125], device="cuda")
    if name in ("situ", "swiglu_step"):
        sf_values *= 8
    a_sf = sf_values[torch.randint(3, (s, k // 16), device="cuda")].to(
        torch.float8_e4m3fn
    )
    b_sf = tuple(
        sf_values[torch.randint(3, (e, n, k // 16), device="cuda")].to(
            torch.float8_e4m3fn
        )
        for _ in w
    )
    offsets = torch.tensor(
        [0, 0, 1, 128, 128, 130, 259, 259, 259], dtype=torch.int32, device="cuda"
    )
    alphas = tuple(
        torch.linspace(0.25 + j, 2.0 + j, e, device="cuda") for j in range(len(w))
    )
    scale = torch.full((1, 1, 1), 0.5, device="cuda") if kernel.fc1 else None
    packed_a = nvfp4.pack_token_scales(a_sf, offsets)
    out = torch.empty(s, n, dtype=torch.bfloat16, device="cuda")
    plan = nvfp4.PreparedNvfp4GroupedGemm(
        kernel,
        x,
        w,
        offsets,
        packed_a,
        tuple(nvfp4.pack_weight_scales(sf) for sf in b_sf),
        out,
        scale=scale,
        gemm_scales=alphas,
    )

    def reference():
        xd = _nvfp4_dequant(x, a_sf)
        wd = [_nvfp4_dequant(v, sf) for v, sf in zip(w, b_sf, strict=True)]
        expected = torch.full((s, n), -7.0, device="cuda", dtype=torch.float32)
        starts = offsets.tolist()
        for expert, (begin, end) in enumerate(zip(starts, starts[1:], strict=False)):
            values = (xd[begin:end] @ wd[0][expert].T).float() * alphas[0][expert]
            if kernel.gated:
                up = (xd[begin:end] @ wd[1][expert].T).float() * alphas[1][expert]
                values = torch.cat((up, values), dim=-1)
            if kernel.fc1:
                values = (
                    compute_reference_activation(
                        values, ACTIVATIONS[name](), n, out_dtype=torch.float32
                    )
                    * scale.flatten()[0]
                )
            expected[begin:end] = values
        return expected.bfloat16()

    plan()
    torch.testing.assert_close(out, reference(), rtol=2e-2, atol=2e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    x.bitwise_xor_(0x88)
    offsets.copy_(
        torch.tensor(
            [0, 2, 2, 2, 31, 259, 259, 259, 259], device="cuda", dtype=torch.int32
        )
    )
    packed_a.copy_(nvfp4.pack_token_scales(a_sf, offsets))
    for j, alpha in enumerate(alphas):
        alpha.mul_(0.5 if j == 0 else 1.5)
    if scale is not None:
        scale.fill_(0.25)
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, reference(), rtol=2e-2, atol=2e-4)

    # The final offset is device data, not the static token-buffer capacity.
    # Cover a nonempty last expert, an empty last expert, and no active rows.
    for boundaries in (
        [0, 0, 1, 128, 128, 130, 130, 130, 193],
        [0, 2, 2, 2, 31, 193, 193, 193, 193],
        [0] * 9,
    ):
        offsets.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
        packed_a.copy_(nvfp4.pack_token_scales(a_sf, offsets))
        out.fill_(-7)
        graph.replay()
        torch.testing.assert_close(out, reference(), rtol=2e-2, atol=2e-4)
        assert torch.all(out[boundaries[-1] :] == -7)


def _nvfp4_moe_reference(act, weights, activation, swizzled=False):
    from flashinfer.quantization.fp4_quantization import fp4_quantize
    from tests.moe.utils import compute_reference_activation

    sf = act.hidden_states_scale
    if swizzled:
        sf = _block_scale_linear_scales(
            sf, act.num_tokens, act.hidden_states_q.shape[-1] // 8
        )
    x = _nvfp4_dequant(act.hidden_states_q, sf)
    ids, scores = act.topk_ids, act.topk_weights
    view = weights.get_view("cutlass_nvfp4")
    w1, w2 = view["fc1_expert_weights"], view["fc2_expert_weights"]
    e, n, _ = w1.shape
    h, i = x.shape[-1], w2.shape[-1] * 2
    expanded = torch.zeros((*ids.shape, h), device=x.device, dtype=torch.float32)
    for expert in range(e):
        token, slot = torch.where(ids == expert)
        if not token.numel():
            continue
        up_gate = _nvfp4_dequant(
            w1[expert],
            _block_scale_linear_scales(
                view["fc1_weight_block_scale"][expert], n, h // 16
            ),
        )
        down = _nvfp4_dequant(
            w2[expert],
            _block_scale_linear_scales(
                view["fc2_weight_block_scale"][expert], h, i // 16
            ),
        )
        values = (x[token] @ up_gate.T).float() * view["fc1_dequant_scale"][expert]
        mid = compute_reference_activation(values, activation, i)
        q, sf = fp4_quantize(
            mid,
            global_scale=view["fc2_act_global_scale"].reshape(1),
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        mid = _nvfp4_dequant(q, sf.reshape(-1, i // 16)[: token.numel()])
        value = ((mid @ down.T).float() * view["fc2_dequant_scale"][expert]).bfloat16()
        expanded[token, slot] = value.float() * scores[token, slot, None]
    return expanded.sum(dim=1).bfloat16()


@frost_gpu
@pytest.mark.parametrize("name", ACTIVATIONS)
@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("fc1_output", ["bf16", "fp4_stg", "fp4_tma"])
def test_nvfp4_moe_full_pipeline_and_dynamic_graph(
    name, swizzled, fc1_output, monkeypatch, geometry=(17, 256, 256, 4, 2)
):
    from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import (
        moe,
    )
    from flashinfer.fused_moe import CutlassNvfp4Config, QuantFormat

    if fc1_output != "bf16" and _sm120():
        pytest.skip("SM120 Frost templates have no packed FP4 FC1 output")
    monkeypatch.setitem(sys.modules, "cudnn", None)
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "0")
    torch.manual_seed(117)
    t, h, i, e, k = geometry
    activation = ACTIVATIONS[name]()
    quant = QuantConfig(
        QuantFormat.NVFP4, QuantFormat.NVFP4, swizzled_scale_factors=swizzled
    )
    config = replace(
        bf16_config(topk=k, experts=e, intermediate=i),
        quant=quant,
        activation=activation,
        backend=BackendOptions((CutlassNvfp4Config(),)),
    )
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            e,
            (2 if activation.is_gated else 1) * i,
            h,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.02
    if name in ("situ", "swiglu_step"):
        x *= 8
        w1 *= 16
    xq, xsf = CutlassNvfp4Config.prepare_activations(x, quant=quant)
    if swizzled:
        from flashinfer.quantization.fp4_quantization import (
            nvfp4_block_scale_interleave,
        )

        xsf = nvfp4_block_scale_interleave(xsf.view(torch.uint8))
    view = CutlassNvfp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size=i,
        activation=activation,
    )
    view["fc1_dequant_scale"].copy_(
        torch.tensor([0.5, 0.75, 1.25, 1.5], device="cuda").repeat(e // 4)
    )
    view["fc2_dequant_scale"].copy_(
        torch.tensor([1.0, 0.25, 0.75, 0.125], device="cuda").repeat(e // 4)
    )
    view["fc2_act_global_scale"].fill_(2.0)
    ids = torch.randint(0, e // 2, (t, k), device="cuda", dtype=torch.int32)
    scores = torch.rand(t, k, device="cuda")
    scores /= scores.sum(-1, keepdim=True)
    act = MoEActivationPack(xq, xsf, ids, scores)
    weights = MoEWeightPack({"cutlass_nvfp4": view})
    first, second = moe._kernels(t * k, h, i, e, xq.device, activation)
    if fc1_output != "bf16":
        first, _ = moe._kernels(
            t * k, h, i, e, xq.device, activation, quantized_output=True
        )
        first = tuple(
            kernel
            for kernel in first
            if kernel.tactic_metadata["store_mode"] == fc1_output.removeprefix("fp4_")
        )
    selected = _pick_stage_pair(first), _pick_stage_pair(second)
    capture_selection = moe._capture_selection
    monkeypatch.setattr(
        moe,
        "_capture_selection",
        lambda *args: capture_selection(*args)._replace(
            select_kernels=lambda shape: selected
        ),
    )
    runner = moe.CudnnFrostNvfp4MoeRunner(config, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 4
    assert inputs[6] is view["fc1_weight_block_scale"]
    assert runner.get_valid_tactics(list(inputs), None) == tactics

    def check(out):
        reference = _nvfp4_moe_reference(act, weights, activation, swizzled)
        assert torch.isfinite(out).all()
        assert reference.float().norm() > 0
        assert torch.count_nonzero(out) > 0
        torch.testing.assert_close(
            out.float(),
            reference.float(),
            rtol=0.05,
            atol=0.01 * reference.float().abs().max().item(),
        )
        assert (
            out.float() - reference.float()
        ).norm() / reference.float().norm().clamp_min(1e-12) < 0.02

    mult = 2 if activation.is_gated else 1
    packed_sf1 = (
        view["fc1_weight_block_scale"]
        .reshape(e, mult, i, h // 16)
        .transpose(0, 1)
        .contiguous()
    )
    packed_weights = MoEWeightPack(
        {"cutlass_nvfp4": view | {"fc1_weight_block_scale": packed_sf1}}
    )
    packed_inputs = runner.pack_inputs(act, packed_weights)
    assert packed_inputs[6] is packed_sf1
    assert packed_inputs.launch_state is inputs.launch_state
    for tactic in tactics:
        check(runner.forward(packed_inputs, tactic))

    for iteration, tactic in enumerate(tactics):
        out = runner.forward(inputs, tactic)
        check(out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.forward(list(inputs), tactic, launch_state=inputs.launch_state)
        ids.fill_(e - 1)
        ids[::3] = -1
        ids[1::3] = e
        scores.mul_(0.5)
        xq.bitwise_xor_(0x88)
        xsf.view(torch.uint8).add_(1)
        view["fc1_weight_block_scale"].view(torch.uint8).add_(1)
        view["fc2_weight_block_scale"].view(torch.uint8).add_(1)
        view["fc1_dequant_scale"].mul_(0.75)
        global_multiplier = 2.0 if iteration % 2 == 0 else 0.5
        view["fc2_dequant_scale"].mul_(1.25 / global_multiplier)
        view["fc2_act_global_scale"].mul_(global_multiplier)
        out.fill_(float("nan"))
        graph.replay()
        check(out)
    with pytest.raises(ValueError, match="launch_state"):
        runner.forward(list(inputs))


@frost_gpu
@pytest.mark.parametrize("store", ["stg", "tma"])
def test_nvfp4_moe_inference_mode_preparation_and_graph(store, monkeypatch):
    _skip_sm120_tma(store)
    # SM120 has no fused FP4 FC1 output: prepare its unfused pipeline instead.
    fc1_output = "bf16" if _sm120() else f"fp4_{store}"
    with torch.inference_mode():
        test_nvfp4_moe_full_pipeline_and_dynamic_graph(
            "swiglu", False, fc1_output, monkeypatch
        )


@frost_gpu
@pytest.mark.parametrize("tokens", [1, 17, 129])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("store", ["stg", "tma"])
def test_mxfp8_fused_quantization_preserves_bf16_intermediate(
    tokens, store, mixed, monkeypatch
):
    from flashinfer.fused_moe import (
        CutlassMxfp8Config,
        CutlassMxfp8Mxfp4Config,
        QuantFormat,
    )

    _skip_sm120_tma(store)
    dtype = "mxfp8_mxfp4" if mixed else "mxfp8"
    moe = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.moe"
    )
    backend = CutlassMxfp8Mxfp4Config if mixed else CutlassMxfp8Config
    monkeypatch.setitem(sys.modules, "cudnn", None)
    torch.manual_seed(131)
    h, i, e, k = 256, 256, 8, 2
    activation = SwiGLU()
    quant = QuantConfig(
        QuantFormat.MXFP4 if mixed else QuantFormat.MXFP8, QuantFormat.MXFP8
    )
    config = replace(
        bf16_config(topk=k, experts=e, intermediate=i),
        quant=quant,
        backend=BackendOptions((backend(),)),
    )
    x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.1
    xq, xsf = backend.prepare_activations(x, quant=quant)
    view = backend.prepare_weights(
        w1,
        w2,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size=i,
        activation=activation,
    )
    view["fc1_input_scale"].copy_(
        torch.tensor([0.0, -0.5, 1.25, 2.0], device="cuda").repeat(e // 4)
    )
    view["fc2_input_scale"].copy_(
        torch.tensor([1.5, -0.75, 0.0, 0.5], device="cuda").repeat(e // 4)
    )
    ids = torch.randint(0, e // 2, (tokens, k), device="cuda", dtype=torch.int32)
    ids[::7, 0] = -1
    scores = torch.rand(tokens, k, device="cuda")
    act = MoEActivationPack(xq, xsf, ids, scores)
    weights = MoEWeightPack({f"cutlass_{dtype}": view})
    first, second = moe._kernels(tokens * k, h, i, e, xq.device, activation)
    fused, _ = moe._kernels(
        tokens * k, h, i, e, xq.device, activation, quantized_output=True
    )
    fused = next(
        kernel for kernel in fused if kernel.tactic_metadata["store_mode"] == store
    )
    original = next(
        kernel
        for kernel in first
        if kernel.tactic_metadata["tile"] == fused.tactic_metadata["tile"]
    )
    capture_selection = moe._capture_selection
    monkeypatch.setattr(
        moe,
        "_capture_selection",
        lambda *args: capture_selection(*args)._replace(
            select_kernels=lambda shape: ((original, fused), second[:2])
        ),
    )
    runner_type = (
        moe.CudnnFrostMxfp8Mxfp4MoeRunner if mixed else moe.CudnnFrostMxfp8MoeRunner
    )
    runner = runner_type(config, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    baseline_tactic, fused_tactic = tactics[0], tactics[2]
    plans = inputs.launch_state.plans
    assert plans[baseline_tactic]["workspace_size"](tokens) - plans[fused_tactic][
        "workspace_size"
    ](tokens) == tokens * k * i * 2 + max(
        original.workspace_bytes, second[0].workspace_bytes
    ) - max(fused.workspace_bytes, second[0].workspace_bytes)

    for input_scale_byte in (127, 60, 0):
        # Include zero blocks and tiny values with E8M0 scale byte zero.
        xsf.view(torch.uint8).fill_(input_scale_byte)
        view["fc1_input_scale"].mul_(-0.5)
        view["fc2_input_scale"].mul_(1.5)
        baseline = {
            name: tensor.clone()
            for name, tensor in moe.prepared_stage_inputs(
                inputs, tactic=baseline_tactic
            ).items()
        }
        actual = moe.prepared_stage_inputs(inputs, tactic=fused_tactic)
        assert "fc1_output" not in actual
        assert torch.equal(actual["offsets"], baseline["offsets"])
        rows, columns = tokens * k, i // 32
        logical_sf = []
        for stage in (baseline, actual):
            unpacked = torch.empty(rows, columns, device="cuda", dtype=torch.uint8)
            starts = stage["offsets"].tolist()
            sf_start = 0
            for begin, end in zip(starts, starts[1:], strict=False):
                row = torch.arange(end - begin, device="cuda")[:, None]
                col = torch.arange(columns, device="cuda")[None, :]
                index = (
                    sf_start * columns
                    + (row // 128) * 128 * columns
                    + (col // 4) * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                    + col % 4
                )
                unpacked[begin:end] = stage["fc2_token_scales"][index]
                sf_start += (end - begin + 127) // 128 * 128
            logical_sf.append(unpacked)
        blocks = baseline["fc1_output"].cpu().double().reshape(rows, columns, 32)
        exponents = torch.ceil(torch.log2(blocks.abs().amax(-1) / 448)).clamp(-127, 127)
        expected_sf = (exponents + 127).to(torch.uint8)
        assert torch.equal(logical_sf[0].cpu(), expected_sf)
        reference = (
            (blocks / torch.exp2(exponents).unsqueeze(-1))
            .float()
            .to(torch.float8_e4m3fn)
            .reshape(rows, i)
        )
        torch.testing.assert_close(
            baseline["fc2_tokens"].cpu().float(), reference.float(), rtol=0, atol=0
        )
        assert torch.equal(*logical_sf)
        torch.testing.assert_close(
            actual["fc2_tokens"].float(), baseline["fc2_tokens"].float(), rtol=0, atol=0
        )
        if input_scale_byte == 127:
            reference = _mxfp8_moe_reference(act, weights, activation, mixed=mixed)
            out = runner.forward(inputs, fused_tactic)
            assert (
                out.float() - reference.float()
            ).norm() / reference.float().norm() < 0.02


@frost_gpu
@pytest.mark.parametrize("tokens", [1, 17, 129])
@pytest.mark.parametrize("store", ["stg", "tma"])
def test_nvfp4_fused_quantization_preserves_bf16_intermediate(
    tokens, store, monkeypatch
):
    from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import (
        moe,
    )
    from flashinfer.fused_moe import CutlassNvfp4Config, QuantFormat

    if _sm120():
        pytest.skip("SM120 Frost templates have no packed FP4 FC1 output")
    monkeypatch.setitem(sys.modules, "cudnn", None)
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "0")
    torch.manual_seed(131)
    h, i, e, k = 256, 256, 8, 2
    activation = SwiGLU()
    quant = QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4)
    config = replace(
        bf16_config(topk=k, experts=e, intermediate=i),
        quant=quant,
        backend=BackendOptions((CutlassNvfp4Config(),)),
    )
    x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.1
    xq, xsf = CutlassNvfp4Config.prepare_activations(x, quant=quant)
    view = CutlassNvfp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size=i,
        activation=activation,
    )
    ids = torch.randint(0, e // 2, (tokens, k), device="cuda", dtype=torch.int32)
    ids[::7, 0] = -1
    scores = torch.rand(tokens, k, device="cuda")
    act = MoEActivationPack(xq, xsf, ids, scores)
    weights = MoEWeightPack({"cutlass_nvfp4": view})
    first, second = moe._kernels(tokens * k, h, i, e, xq.device, activation)
    fused, _ = moe._kernels(
        tokens * k, h, i, e, xq.device, activation, quantized_output=True
    )
    fused = next(
        kernel for kernel in fused if kernel.tactic_metadata["store_mode"] == store
    )
    original = next(
        kernel
        for kernel in first
        if kernel.tactic_metadata["tile"] == fused.tactic_metadata["tile"]
        and kernel.tactic_metadata["store_mode"] == "stg"
    )
    capture_selection = moe._capture_selection
    monkeypatch.setattr(
        moe,
        "_capture_selection",
        lambda *args: capture_selection(*args)._replace(
            select_kernels=lambda shape: ((original, fused), second[:2])
        ),
    )
    runner = moe.CudnnFrostNvfp4MoeRunner(config, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    baseline_tactic, fused_tactic = tactics[0], tactics[2]
    plans = inputs.launch_state.plans
    assert plans[baseline_tactic]["workspace_size"](tokens) - plans[fused_tactic][
        "workspace_size"
    ](tokens) == tokens * k * i * 2 + max(
        original.workspace_bytes, second[0].workspace_bytes
    ) - max(fused.workspace_bytes, second[0].workspace_bytes)

    for global_scale in (1.0, 1.3, 3.7, 0.03125):
        view["fc2_act_global_scale"].fill_(global_scale)
        baseline = {
            name: tensor.clone()
            for name, tensor in moe.prepared_stage_inputs(
                inputs, tactic=baseline_tactic
            ).items()
        }
        actual = moe.prepared_stage_inputs(inputs, tactic=fused_tactic)
        assert "fc1_output" not in actual
        assert torch.equal(actual["offsets"], baseline["offsets"])
        rows, columns = tokens * k, i // 16
        logical_sf = []
        for stage in (baseline, actual):
            unpacked = torch.empty(rows, columns, device="cuda", dtype=torch.uint8)
            starts = stage["offsets"].tolist()
            sf_start = 0
            for begin, end in zip(starts, starts[1:], strict=False):
                row = torch.arange(end - begin, device="cuda")[:, None]
                col = torch.arange(columns, device="cuda")[None, :]
                index = (
                    sf_start * columns
                    + (row // 128) * 128 * columns
                    + (col // 4) * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                    + col % 4
                )
                unpacked[begin:end] = stage["fc2_token_scales"][index]
                sf_start += (end - begin + 127) // 128 * 128
            logical_sf.append(unpacked)
        assert torch.equal(*logical_sf)
        # Underflowed scales can produce different FP4 codes for zero values;
        # those blocks dequantize to zero in both paths.
        expected = _nvfp4_dequant(baseline["fc2_tokens"], logical_sf[0])
        result = _nvfp4_dequant(actual["fc2_tokens"], logical_sf[1])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_nvfp4_moe_shortlist_uses_measured_two_by_two_buckets(
    monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import (
        moe,
    )

    _assert_moe_shortlist_buckets(moe, monkeypatch)


def test_nvfp4_packaged_shortlists_resolve_all_profiles(
    monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import (
        moe,
    )

    _assert_packaged_shortlists_resolve_all_profiles(moe, monkeypatch)


def _assert_packaged_shortlists_resolve_all_profiles(moe, monkeypatch):
    from flashinfer.fused_moe.backends.cudnn_frost import (
        shortlist,
    )

    roots = moe._artifact_roots()
    table = shortlist._read(roots)
    assert table
    assert {key[0] for key in table} == {"sm_107a", "sm_120a"}
    for arch in ("sm_107a", "sm_120a"):
        assert {key[1] for key in table if key[0] == arch} == set(ACTIVATIONS)
    device_arch = {}
    monkeypatch.setattr(moe.common, "_arch_for", lambda device: device_arch["arch"])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=128),
    )
    for (arch, name, experts, hidden, intermediate, topk), profiles in table.items():
        device_arch["arch"] = arch
        previous = 0
        for tokens, identities in sorted(profiles.items()):
            # Check both measured counts and the next-bucket rule using the
            # installed table and actual artifact contracts, without compiling.
            for query in {tokens, previous + 1}:
                first, second = moe._selected_kernels(
                    query,
                    hidden,
                    intermediate,
                    experts,
                    topk,
                    "cpu",
                    ACTIVATIONS[name](),
                )
                assert 2 <= len(first) <= 6 and len(second) == 2
                assert tuple(k.artifact_id for k in first[:2]) == identities[0]
                assert tuple(k.artifact_id for k in second) == identities[1]
                for kernel in first[2:]:
                    assert kernel.quantizes_output and not kernel.swap_ab
                    assert kernel.tactic_metadata["tile"] in {
                        original.tactic_metadata["tile"] for original in first[:2]
                    }
                if first[2:]:
                    variants = {
                        (
                            kernel.tactic_metadata["tile"],
                            kernel.tactic_metadata["store_mode"],
                        )
                        for kernel in first[2:]
                    }
                    normal_tiles = {
                        original.tactic_metadata["tile"]
                        for original in first[:2]
                        if not original.swap_ab
                    }
                    # SM120 stores through STG only, and its producer fuses the
                    # quantized output for a subset of the warp-MMA tiles.
                    stores = ("stg", "tma") if arch == "sm_107a" else ("stg",)
                    if arch == "sm_107a":
                        assert {(tile, "stg") for tile in normal_tiles} <= variants
                    assert variants <= {
                        (tile, store) for tile in normal_tiles for store in stores
                    }
                assert all(k.fc1 and k.activation == name for k in first)
                assert all(not k.fc1 for k in second)
                assert (
                    len({(a.tactic, b.tactic) for a in first for b in second})
                    == len(first) * 2
                )
            previous = tokens


def test_mxfp8_mxfp4_geometry_sharing_and_roundtrip_without_cudnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    root = mxfp8_mxfp4.runtime.artifact_root("mxfp8_mxfp4")
    records = read_manifest(root / mxfp8_mxfp4.runtime._MANIFEST)["kernels"]
    _assert_block_scale_geometry_roundtrip(
        root,
        records,
        "cutlass.Float8E4M3FN",
        "cutlass.Float8E8M0FNU",
        32,
        weight_dtype="cutlass.Float4E2M1FNx2",
    )


def test_mxfp8_mxfp4_scale_layout_uneven_empty_groups_and_experts():
    _assert_block_scale_layout(mxfp8_mxfp4)


def test_mxfp8_mxfp4_segmented_scale_capacity_covers_routing_partitions():
    _assert_segmented_scale_capacity(mxfp8_mxfp4)


@pytest.mark.parametrize("offsets", [[1, 5], [0, 9, 4], [0, 11]])
def test_mxfp8_mxfp4_invalid_offsets(offsets):
    with pytest.raises(ValueError, match="offsets"):
        mxfp8_mxfp4.pack_token_scales(
            torch.ones(10, 8, dtype=torch.uint8),
            torch.tensor(offsets, dtype=torch.int32),
        )


def test_mxfp8_mxfp4_manifest_and_launch_abi_without_cudnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    _assert_block_scale_launch_abi(
        mxfp8_mxfp4, packed=False, packed_weights=True, block=32
    )


@pytest.mark.parametrize(
    "change",
    [
        {"block_size": 16},
        {"token_dtype": "float4_e2m1fn_x2"},
        {"weight_dtype": "float8_e4m3fn"},
        {"scale_dtype": "float8_e4m3fn"},
        {"global_scale_dtype": "bfloat16"},
        {"global_scale_layout": "per_expert"},
        {"global_scale_position": "after_activation"},
    ],
)
@pytest.mark.parametrize("schema_version", [2, 3])
def test_mxfp8_mxfp4_reject_incompatible_numerical_contract(
    change, tmp_path, schema_version
):
    root = mxfp8_mxfp4.runtime.artifact_root("mxfp8_mxfp4")
    record = read_manifest(root / mxfp8_mxfp4.runtime._MANIFEST)["kernels"][0]
    payload = {
        "schema_version": 2,
        "kernels": [record | {"contract": record["contract"] | change}],
    }
    if schema_version == 3:
        payload = normalize_manifest(payload)
    (tmp_path / mxfp8_mxfp4.runtime._MANIFEST).write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="numerical contract"):
        mxfp8_mxfp4.discover(tmp_path)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_mxfp8_mxfp4_auto_admission_uses_logical_hidden_size(name):
    _assert_mxfp8_auto_admission(name, mixed=True)


@frost_gpu
@pytest.mark.parametrize("name", [*ACTIVATIONS, "fc2"])
@pytest.mark.parametrize(
    "swap_ab,store", [(False, "stg"), (False, "tma"), (True, "stg"), (True, "tma")]
)
def test_mxfp8_mxfp4_grouped_kernel_extremes_and_graph(
    name, swap_ab, store, monkeypatch
):
    _skip_sm120_tma(store)
    candidates = [
        kernel
        for kernel in mxfp8_mxfp4.discover()
        if (kernel.activation if kernel.fc1 else "fc2") == name
        and kernel.arch == _device_arch()
        and kernel.swap_ab == swap_ab
        and kernel.tactic_metadata["store_mode"] == store
    ]
    if not candidates:
        pytest.skip(f"shortlist retains no {name}/{swap_ab=}/{store} family")
    _assert_mxfp8_grouped_kernel_and_graph(
        mxfp8_mxfp4, candidates[0], monkeypatch, mixed=True
    )


@frost_gpu
@pytest.mark.parametrize("name", ACTIVATIONS)
@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("fc1_output", ["bf16", "fp8_stg", "fp8_tma"])
def test_mxfp8_mxfp4_moe_full_pipeline_and_dynamic_graph(
    name, swizzled, monkeypatch, fc1_output
):
    _assert_mxfp8_moe_full_pipeline_and_graph(
        name, swizzled, monkeypatch, mixed=True, fc1_output=fc1_output
    )


@frost_gpu
def test_mxfp8_mxfp4_moe_inference_mode_preparation_and_graph(monkeypatch):
    with torch.inference_mode():
        test_mxfp8_mxfp4_moe_full_pipeline_and_dynamic_graph(
            "swiglu", False, monkeypatch, "fp8_tma"
        )


def test_mxfp8_mxfp4_moe_shortlist_uses_measured_two_by_two_buckets(
    monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.mxfp8_mxfp4 import (
        moe,
    )

    _assert_moe_shortlist_buckets(moe, monkeypatch)


def test_mxfp8_mxfp4_packaged_shortlists_resolve_all_profiles(
    monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.mxfp8_mxfp4 import (
        moe,
    )

    _assert_packaged_shortlists_resolve_all_profiles(moe, monkeypatch)


def test_frost_external_compiler_identity_tracks_executable_changes(
    tmp_path, monkeypatch
):
    from flashinfer.fused_moe.backends.cudnn_frost import (
        compiler,
    )

    monkeypatch.delenv(compiler._ENV, raising=False)
    assert compiler.compiler_identity() == {"backend": "bundled"}
    assert compiler.identity_key() == ""
    executable = tmp_path / "ptxas"
    executable.write_bytes(b"first executable")
    executable.chmod(0o755)
    calls = []

    def version(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="fixture ptxas 13.5\n")

    monkeypatch.setattr(compiler.subprocess, "run", version)
    monkeypatch.setenv(compiler._ENV, str(executable))
    first = compiler.compiler_identity()
    assert first["path"] == str(executable.resolve())
    assert first["sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert compiler.compiler_identity() == first
    assert len(calls) == 1
    executable.write_bytes(b"second executable with different contents")
    second = compiler.compiler_identity()
    assert second["sha256"] != first["sha256"]
    assert len(calls) == 2
    monkeypatch.setenv(compiler._ENV, str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError):
        compiler.compiler_identity()


def test_frost_compiler_switch_invalidates_memory_and_tactic_keys(monkeypatch):
    from flashinfer.fused_moe.backends.cudnn_frost import (
        compiler,
        runtime,
    )

    key = [""]
    monkeypatch.setattr(compiler, "identity_key", lambda: key[0])
    monkeypatch.setattr(runtime, "_load_source_cached", lambda *args: args[-1])
    args = (Path("source.py"), "a" * 64, "sm_107a", 0)
    first = runtime._tactic_digest(args[1])
    assert runtime._load_source(*args) == ""
    key[0] = '{"backend":"external_ptxas","sha256":"first"}'
    second = runtime._tactic_digest(args[1])
    assert runtime._load_source(*args) == key[0]
    key[0] = '{"backend":"external_ptxas","sha256":"second"}'
    third = runtime._tactic_digest(args[1])
    assert runtime._load_source(*args) == key[0]
    assert len({first, second, third}) == 3
    key[0] = ""
    assert runtime._tactic_digest(args[1]) == first


def _frost_compiler_ir(binary_count=1):
    pytest.importorskip("cutlass")
    from cutlass._mlir import ir

    # The fixture is a standard IR constant, not a loadable object or kernel.
    with ir.Context():
        globals_ = "\n".join(
            'llvm.mlir.global internal constant @binary_%d("\\50\\ed\\55\\ba")' % index
            for index in range(binary_count)
        )
        return ir.Module.parse("module {" + globals_ + "}")


@pytest.mark.parametrize("binary_count", [0, 1, 2])
def test_frost_external_binary_replacement_validates_and_restores(binary_count):
    from flashinfer.fused_moe.backends.cudnn_frost import (
        compiler,
    )

    module = _frost_compiler_ir(binary_count)
    compiled = SimpleNamespace(ir_module=module)
    before = str(module)
    if binary_count != 1:
        with (
            pytest.raises(RuntimeError, match="exactly one"),
            compiler._binary_initializer(compiled, b"\x7fELFnew-cubin"),
        ):
            pytest.fail("an ambiguous binary must not be exported")
    else:
        with (
            pytest.raises(RuntimeError, match="export failed"),
            compiler._binary_initializer(compiled, b"\x7fELFnew-cubin"),
        ):
            assert str(module) != before
            raise RuntimeError("export failed")
        with (
            pytest.raises(RuntimeError, match="ELF CUBIN"),
            compiler._binary_initializer(compiled, b"not a cubin"),
        ):
            pytest.fail("invalid external output must not be exported")
    assert str(module) == before


@pytest.mark.parametrize(
    "mode", ["persistent", "disabled", "export_failure", "hash_failure"]
)
def test_frost_external_compiler_cache_paths_use_reassembled_binary(
    tmp_path, monkeypatch, mode
):
    from flashinfer.fused_moe.backends.cudnn_frost import (
        compiler,
    )
    from flashinfer.jit import cute_dsl_core
    import cutlass.cute as cute
    import cutlass.runtime as cutlass_runtime

    module = _frost_compiler_ir()
    original = str(module)
    cubin = b"\x7fELFexternal-assembler-output"

    def export(path, function_name=None, **kwargs):
        op = module.body.operations[0]
        Path(path).write_bytes(op.attributes["value"].value_bytes)

    compiled = SimpleNamespace(ir_module=module, export_to_c=export)
    adapter = compiler._ExternalCompiledKernel(compiled, cubin)
    builds = []

    def compile_kernel():
        builds.append(True)
        return adapter

    class Loaded:
        def __init__(self, data):
            self.data = data

        def __getattr__(self, symbol):
            return lambda: self.data

    def load(path, **kwargs):
        return Loaded(Path(path).read_bytes())

    monkeypatch.setattr(cute.runtime, "load_module", load)
    monkeypatch.setattr(cutlass_runtime, "load_module", load)
    monkeypatch.setattr(cute_dsl_core.jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.setattr(cute_dsl_core, "get_tmpdir", lambda: tmp_path)
    monkeypatch.setattr(cute_dsl_core, "_get_compile_arch", lambda: "sm107a")
    monkeypatch.setattr(cute_dsl_core, "_get_cute_dsl_version", lambda: "fixture")
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setenv(
        "FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1" if mode == "disabled" else "0"
    )

    def fail(*args):
        raise OSError("synthetic persistence failure")

    if mode == "export_failure":
        monkeypatch.setattr(cute_dsl_core.JitSpecCuteDsl, "_export", fail)
    if mode == "hash_failure":
        monkeypatch.setattr(cute_dsl_core, "_hash_source_files", fail)
    result = cute_dsl_core.build_and_load_cute_dsl_kernel(
        "frost_external_fixture", "kernel", compile_kernel, extra_key_files=(__file__,)
    )
    if hasattr(result, "__tvm_ffi_object__"):
        result = result.__tvm_ffi_object__()
    assert result() == cubin
    assert str(module) == original
    if mode == "persistent":
        cached = cute_dsl_core.build_and_load_cute_dsl_kernel(
            "frost_external_fixture",
            "kernel",
            compile_kernel,
            extra_key_files=(__file__,),
        )
        assert cached() == cubin and len(builds) == 1


@pytest.mark.parametrize(
    "dtype,runner_name",
    [
        ("bf16", "CudnnFrostBf16MoeRunner"),
        ("mxfp8", "CudnnFrostMxfp8MoeRunner"),
        ("nvfp4", "CudnnFrostNvfp4MoeRunner"),
        ("mxfp8_mxfp4", "CudnnFrostMxfp8Mxfp4MoeRunner"),
    ],
)
def test_frost_moe_forward_accepts_base_runner_kwargs(dtype, runner_name, monkeypatch):
    from flashinfer.fused_moe.backends.cudnn_frost import engine

    module = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.moe"
    )
    monkeypatch.setattr(engine, "retain_graph_resources", lambda *_: None)
    runner = object.__new__(getattr(module, runner_name))
    runner._engine = object.__new__(engine.FrostEngine)
    monkeypatch.setattr(runner, "_require_built", lambda: None)
    calls = []
    output = torch.empty(1)
    workspace = torch.empty(1, dtype=torch.uint8)
    tactic = ("selected",)
    state = SimpleNamespace(
        launches={tactic: lambda *args: calls.append(args)}, workspace=workspace
    )
    result = runner.forward(
        [output],
        tactic,
        launch_state=state,
        enable_pdl=False,
    )
    assert result is output
    assert len(calls) == 1
    assert calls[0][0] is output
    assert calls[0][1] is workspace


def test_bf16_clear_artifact_cache_refreshes_fc2_manifest(
    tmp_path, monkeypatch, frost_metadata_only
):
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import (
        fc2,
        runtime,
    )

    packaged = runtime.artifact_root("bf16") / runtime._MANIFEST
    record = next(
        raw
        for raw in read_manifest(packaged)["kernels"]
        if raw["op"] == "grouped_gemm2"
    )
    manifest = tmp_path / runtime._MANIFEST
    payload = {"schema_version": 2, "kernels": [record]}
    manifest.write_text(json.dumps(payload))
    monkeypatch.setattr(fc2, "_artifact_roots", lambda: (tmp_path,))
    # This test exercises metadata cache invalidation without copying or compiling
    # frozen kernel source files.
    monkeypatch.setattr(
        fc2, "_read_source", lambda *_: (tmp_path / "unused.py", "verified-digest")
    )
    runtime.clear_artifact_cache()
    try:
        before = fc2.discover()
        record["workspace_bytes"] += 128
        manifest.write_text(json.dumps(payload))
        assert fc2.discover() is before
        runtime.clear_artifact_cache()
        after = fc2.discover()
        assert after[0].workspace_bytes == before[0].workspace_bytes + 128
    finally:
        runtime.clear_artifact_cache()


def test_frost_layer_bounds_token_bucket_winners_and_refreshes_lru(monkeypatch):
    from flashinfer.fused_moe import RoutingInputMode

    class FakeRunner:
        backend_key = "cutlass_bf16"
        supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)

        def __init__(self, config, device):
            pass

        @staticmethod
        def supports_quant(quant):
            return True

        def check_support(self):
            pass

        def build(self):
            pass

        def accepts(self, act, weights):
            return True

        def pack_inputs(self, act, weights):
            return [act.hidden_states_q]

        def launch_kwargs_for(self, inputs):
            return {}

        def forward(self, inputs, **kwargs):
            return inputs[0]

    monkeypatch.setattr(layer_module, "get_compute_capability", lambda _: (10, 7))
    monkeypatch.setattr(
        layer_module, "_BACKEND_RUNNERS", {CutlassBf16Config: FakeRunner}
    )
    # Give each token count its own bucket to exercise the winner-cache bound.
    monkeypatch.setattr(
        layer_module, "map_to_hybrid_bucket", lambda tokens, ceiling: tokens
    )
    layer = MoELayer(bf16_config(ceiling=256), device=torch.device("cpu"))
    selected = []

    def select(act, weights, runners):
        selected.append(act.num_tokens)
        assert runners == [layer.runners[0]]
        return layer.runners[0], -1

    monkeypatch.setattr(layer, "_select_winner", select)

    def run(tokens):
        act = SimpleNamespace(
            num_tokens=tokens,
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
            hidden_states_q=torch.empty(tokens, 2),
        )
        assert layer(act, MoEWeightPack({})) is act.hidden_states_q

    for tokens in range(1, 129):
        run(tokens)
    assert len(layer._winners) == 128
    assert selected == list(range(1, 129))
    run(1)  # A cache hit keeps this bucket alive across the next insertion.
    run(129)
    run(1)
    assert selected == list(range(1, 130))
    assert len(layer._winners) == 128
    run(2)  # The least recently used bucket was evicted and must be selected again.
    assert selected[-2:] == [129, 2]
    assert len(selected) == 130
    assert len(layer._winners) == 128
    assert layer.winner_backend == "cutlass_bf16"
    layer.reset_winner()
    assert not layer._winners
    assert layer.winner_backend is None
    run(1)
    assert selected[-1] == 1
    assert len(selected) == 131


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("bf16", marks=frost_gpu),
        *(
            pytest.param(dtype, marks=frost_gpu)
            for dtype in ("mxfp8", "nvfp4", "mxfp8_mxfp4")
        ),
    ],
)
def test_moe_graph_retains_evicted_plans_and_weight_scales(dtype, monkeypatch):
    """A graph owns bucket storage, executors and caller SF after cache eviction."""
    from flashinfer.fused_moe.backends.cudnn_frost.cache import LRUCache
    from flashinfer.fused_moe import (
        CutlassMxfp8Config,
        CutlassMxfp8Mxfp4Config,
        CutlassNvfp4Config,
        QuantFormat,
    )

    monkeypatch.setitem(sys.modules, "cudnn", None)
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "0")
    torch.manual_seed(239)
    t, h, i, e, k = 17, 256, 256, 4, 2
    if dtype == "bf16":
        h, e = 128, 8
    activation = SwiGLU()
    moe = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.moe"
    )
    config = bf16_config(topk=k, experts=e, intermediate=i)
    if dtype == "bf16":
        runner_type = moe.CudnnFrostBf16MoeRunner
    else:
        backend, weight_format, activation_format, runner_name = {
            "mxfp8": (
                CutlassMxfp8Config,
                QuantFormat.MXFP8,
                QuantFormat.MXFP8,
                "CudnnFrostMxfp8MoeRunner",
            ),
            "nvfp4": (
                CutlassNvfp4Config,
                QuantFormat.NVFP4,
                QuantFormat.NVFP4,
                "CudnnFrostNvfp4MoeRunner",
            ),
            "mxfp8_mxfp4": (
                CutlassMxfp8Mxfp4Config,
                QuantFormat.MXFP4,
                QuantFormat.MXFP8,
                "CudnnFrostMxfp8Mxfp4MoeRunner",
            ),
        }[dtype]
        runner_type = getattr(moe, runner_name)
        config = replace(
            config,
            quant=QuantConfig(
                weight_format, activation_format, swizzled_scale_factors=False
            ),
            backend=BackendOptions((backend(),)),
        )

    def packs(tokens):
        if dtype == "bf16":
            return bf16_packs(
                tokens=tokens, topk=k, experts=e, hidden=h, intermediate=i
            )
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
        weight_std = 0.02 if dtype == "nvfp4" else 0.1
        if dtype != "nvfp4":
            x *= 0.1
        w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) * weight_std
        w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * weight_std
        xq, xsf = backend.prepare_activations(x, quant=config.quant)
        view = backend.prepare_weights(
            w1,
            w2,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size=i,
            activation=activation,
        )
        ids = torch.randint(0, e // 2, (tokens, k), device="cuda", dtype=torch.int32)
        scores = torch.rand(tokens, k, device="cuda")
        scores /= scores.sum(-1, keepdim=True)
        return MoEActivationPack(xq, xsf, ids, scores), MoEWeightPack(
            {f"cutlass_{dtype}": view}
        )

    act, weights = packs(t)
    first, second = moe._kernels(t * k, h, i, e, act.hidden_states_q.device, activation)
    selected = _pick_stage_pair(first), _pick_stage_pair(second)
    capture_selection = moe._capture_selection
    monkeypatch.setattr(
        moe,
        "_capture_selection",
        lambda *args: capture_selection(*args)._replace(
            select_kernels=lambda shape: selected
        ),
    )
    runner = runner_type(config, "cuda")
    runner.check_support()
    runner.build()
    # Small bounds exercise lookup eviction while packed states and graphs
    # retain their original workspace and native executor modules.
    runner._engine.plans = runner._plans = LRUCache(maxsize=2)
    runner._engine.executors = LRUCache(maxsize=2)
    inputs = runner.pack_inputs(act, weights)
    state = inputs.launch_state
    original_key = next(iter(runner._plans))
    tactic = runner.get_valid_tactics(inputs, None)[0]
    expected = runner.forward(inputs, tactic).clone()
    assert torch.isfinite(expected).all()
    assert torch.count_nonzero(expected) > 0
    # Keep only a separate observed output: a layer result may be an intermediate
    # in a larger graph, so its Tensor wrapper cannot be the resource owner.
    observed = torch.empty_like(expected)
    references = [weakref.ref(state), weakref.ref(inputs[0]), weakref.ref(inputs[4])]
    if dtype != "bf16":
        references.append(weakref.ref(inputs[6]))
        if dtype in ("mxfp8", "mxfp8_mxfp4"):
            references.extend((weakref.ref(inputs[9]), weakref.ref(inputs[10])))
        scale_key = (
            "fc1_weight_block_scale" if dtype == "nvfp4" else "fc1_expert_scales"
        )
        source_scale = weights.get_view(f"cutlass_{dtype}")[scale_key]
        assert inputs[6].data_ptr() == source_scale.data_ptr()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observed.copy_(runner.forward(list(inputs), tactic, launch_state=state))

    # Keep the source tensors alive while churning: eviction must be due to the
    # bound, not merely a source-tensor weakref callback. Each preparation gets
    # freshly quantized weights and a different token bucket, outside capture.
    replacements = []
    for tokens in (33, 65, 129):
        next_act, next_weights = packs(tokens)
        next_inputs = runner.pack_inputs(next_act, next_weights)
        runner.forward(next_inputs)
        replacements.append((next_act, next_weights, next_inputs))
    assert len(runner._plans) == 2
    assert original_key not in runner._plans
    assert all(
        original is not current
        for original in state.plans.values()
        for current in runner._engine.executors.values()
    )
    if dtype != "bf16":
        assert inputs[6].data_ptr() == source_scale.data_ptr()
        del source_scale

    # Refresh releases lookup ownership while old packed calls and graphs keep
    # their original launch state and compiler-qualified tuning identity.
    preparation_key = state.preparation_key
    runner.refresh()
    assert not runner._plans
    assert state.preparation_key == preparation_key
    torch.testing.assert_close(runner.forward(inputs, tactic), expected, rtol=0, atol=0)

    del inputs, state, act, weights
    gc.collect()
    assert all(reference() is not None for reference in references)
    observed.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    # Also exercise reset with work potentially in flight; the graph retention
    # hook must drain the replay stream before it drops the final references.
    observed.fill_(float("nan"))
    graph.replay()
    graph.reset()
    gc.collect()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert all(reference() is None for reference in references)


def test_frost_tensor_cache_evicts_and_tracks_source_lifetime():
    from flashinfer.fused_moe.backends.cudnn_frost.cache import (
        TensorCache,
    )

    cache = TensorCache(maxsize=2)
    sources = [torch.ones(4) for _ in range(3)]
    first, second = torch.zeros(4), torch.zeros(4)
    first_ref, second_ref = weakref.ref(first), weakref.ref(second)
    cache.put(sources[0], sources[0]._version, first)
    cache.put(sources[1], sources[1]._version, second)
    del first, second
    assert cache.get(sources[0], sources[0]._version) is first_ref()
    cache.put(sources[2], sources[2]._version, torch.zeros(4))
    assert len(cache) == 2
    assert second_ref() is None  # Least recently used, even though its source lives.
    sources[0].add_(1)
    assert cache.get(sources[0], sources[0]._version) is None
    del sources[0]
    gc.collect()
    assert first_ref() is None
    assert len(cache) == 1
    cache.clear()
    assert not len(cache)


@pytest.mark.parametrize("dtype", ["mxfp8", "nvfp4", "mxfp8_mxfp4"])
def test_frost_scale_validation_does_not_retain_replaced_tensors(dtype, monkeypatch):
    from flashinfer.fused_moe.backends.cudnn_frost.cache import (
        TensorCache,
    )

    moe = importlib.import_module(
        f"flashinfer.fused_moe.backends.cudnn_frost.{dtype}.moe"
    )
    cls = {
        "mxfp8": "CudnnFrostMxfp8MoeRunner",
        "nvfp4": "CudnnFrostNvfp4MoeRunner",
        "mxfp8_mxfp4": "CudnnFrostMxfp8Mxfp4MoeRunner",
    }[dtype]
    runner = object.__new__(getattr(moe, cls))
    cache = TensorCache(maxsize=2)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    if dtype == "nvfp4":
        runner.device = torch.device("cpu")
        monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
        runner._validated_scales = cache
        validate = lambda value: runner._check_scale(value, positive=True)
        negative = torch.full((4,), -1.0)
        runner._check_scale(negative, positive=False)
        with pytest.raises(ValueError, match="positive"):
            validate(negative)
        del negative
    else:
        # MX formats validate only tensor metadata. Their global multipliers
        # need no value-validation cache, so retain the underlying weak-source
        # cache contract independently of those removed admission checks.
        validate = lambda value: cache.put(value, value._version, True)
    source = torch.ones(4)
    validate(source)
    reference = weakref.ref(source)
    assert len(cache) == 1
    del source
    gc.collect()
    assert reference() is None
    assert len(cache) == 0


@pytest.mark.parametrize("method", ["get_currently_capturing_graph", "retain_object"])
def test_frost_moe_declines_missing_graph_ownership(method, monkeypatch):
    from flashinfer.fused_moe.backends.cudnn_frost.cache import (
        require_graph_resource_retention,
    )

    monkeypatch.setattr(torch.cuda.CUDAGraph, method, None, raising=False)
    with pytest.raises(NotImplementedError, match=method):
        require_graph_resource_retention()
