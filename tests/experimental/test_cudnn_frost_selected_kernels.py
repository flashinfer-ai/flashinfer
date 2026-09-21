# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Correctness and integration tests for selected cuDNN Frost kernels."""

import ast
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
    capabilities,
)
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.bf16 import (
    moe as bf16_moe,
)
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.activations import (
    ACTIVATIONS,
    activation_name,
)
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8 import (
    runtime as mxfp8,
)
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8_mxfp4 import (
    runtime as mxfp8_mxfp4,
)
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.nvfp4 import (
    runtime as nvfp4,
)
from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.source_template import (
    extract_template,
    render_source,
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
MXFP8_RECORDS = json.loads(
    (MXFP8_ROOT / "cudnn_frost_selected_kernels.json").read_text()
)["kernels"]


supported_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="packaged cuDNN Frost sources target SM107a",
)


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


@supported_gpu
@pytest.mark.parametrize("name", [name for name in ACTIVATIONS if name != "swiglu"])
@pytest.mark.parametrize("tokens,topk", [(17, 1), (129, 4)])
def test_bf16_extended_activations_graph_and_routing(name, tokens, topk, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Extended activation sources have been selected on SM107")
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


def test_bf16_activation_artifact_pools_are_disjoint(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 7))
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


@supported_gpu
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
    assert len(tactics) == 16
    if torch.cuda.get_device_capability() == (10, 7):
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


@supported_gpu
def test_bf16_interleaved_packs_do_not_exchange_weights_and_reject_stale_tactics():
    runner = bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), "cuda")
    runner.check_support()
    runner.build()
    a, wa = bf16_packs(tokens=17)
    b, wb = bf16_packs(tokens=33)
    pb = runner.pack_inputs(b, wb)
    pa = runner.pack_inputs(a, wa)
    assert pa.tuning_config.cuda_graph_profile_replays == 3
    assert pa.launch_state.workspace.data_ptr() == pb.launch_state.workspace.data_ptr()
    assert len(runner._workspace_pool) == 1
    # Simulate the plain-list inputs synthesized by the autotuner.
    runner.forward(list(pa), **runner.launch_kwargs_for(pa))
    runner.forward(pb)
    torch.testing.assert_close(pa[0], _bf16_moe_reference(a, wa), atol=2e-4, rtol=2e-2)
    torch.testing.assert_close(pb[0], _bf16_moe_reference(b, wb), atol=2e-4, rtol=2e-2)
    with pytest.raises(ValueError, match="stale"):
        runner.forward(pa, ("missing",))
    with pytest.raises(ValueError, match="launch_state"):
        runner.forward(list(pa))
    wa.native_views["cutlass_bf16"]["gemm1_alpha"] = torch.ones(8, device="cuda")
    assert not runner.accepts(a, wa)
    with pytest.raises(ValueError, match="overrides"):
        runner.pack_inputs(a, wa)


@supported_gpu
@pytest.mark.parametrize(
    "experts,hidden,intermediate",
    [(12, 7168, 3072), (8, 4096, 14336), (64, 2048, 1408)],
)
def test_bf16_model_geometry_artifacts_ragged_graph(
    experts, hidden, intermediate, monkeypatch
):
    """Exercise every packaged pair on partial tiles and initially empty experts."""
    if experts == 64 and torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("E64 artifacts are packaged for Rubin only")
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


@supported_gpu
@pytest.mark.parametrize(
    "overrides",
    [
        {"activation": SwiGLU(alpha=2.0)},
        {"finalize": MoEFinalizeConfig(do_finalize=False)},
        {"experts": ExpertConfig(intermediate_size=256, local_expert_offset=1)},
    ],
)
def test_bf16_unsupported_semantics_are_not_automatic_candidates(overrides):
    assert (
        bf16_moe.automatic_candidate(replace(bf16_config(), **overrides), "cuda")
        is None
    )


def test_bf16_layer_adds_independent_candidate_and_separates_winner_cache(monkeypatch):
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
            return not weights.native_views.get("override")

        def pack_inputs(self, act, weights):
            return [act.hidden_states_q]

        def launch_kwargs_for(self, inputs):
            return {}

        def forward(self, inputs, **kwargs):
            return inputs[0]

    old, cudnn_frost = FakeRunner("cutlass_bf16"), FakeRunner("cudnn_frost_bf16")
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.device, layer._arch = cfg, torch.device("cuda", 0), 107
    layer.tuner = SimpleNamespace(is_tuning_mode=True)
    layer.runners, layer._automatic_runners, layer._winners = [old], {}, {}
    monkeypatch.setattr(bf16_moe, "automatic_candidate", lambda *args: cudnn_frost)

    def select(act, weights, runners):
        calls.append([r.backend_key for r in runners])
        return runners[-1], -1

    monkeypatch.setattr(layer, "_select_winner", select)
    act, weights = bf16_packs(
        tokens=4096, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    layer(act, weights)
    assert layer.winner_backend == "cudnn_frost_bf16"
    assert calls == [["cutlass_bf16", "cudnn_frost_bf16"]]
    assert layer.runners == [old]  # no mutation of original backend collection
    layer.tuner.is_tuning_mode = False
    layer(act, weights)
    assert len(calls) == 1
    weights.native_views["override"] = {"present": True}
    layer(act, weights)
    assert layer.winner_backend == "cutlass_bf16"
    weights.native_views.pop("override")
    # Same old tuning bucket, but another exact token count needs its own plan.
    monkeypatch.setattr(layer_module, "map_to_hybrid_bucket", lambda *args: 4096)
    act2, weights2 = bf16_packs(
        tokens=4097, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    layer(act2, weights2)
    assert calls[-1] == ["cutlass_bf16", "cudnn_frost_bf16"]
    assert len(calls) == 3
    layer.reset_winner()
    assert not layer._winners


@supported_gpu
@pytest.mark.parametrize(
    "experts,hidden,intermediate",
    [(12, 7168, 3072), (8, 4096, 14336)],
)
def test_bf16_original_layer_api_can_execute_winning_cudnn_frost_and_replay(
    experts, hidden, intermediate, monkeypatch
):
    from flashinfer.autotuner import autotune

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
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.bf16.support import (
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
        (12289, 7168, 107, False),
        (4096, 4096, 100, False),
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


def test_bf16_artifact_architecture_and_swap_abi_isolation(monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.bf16 import (
        runtime,
    )

    for capability in ((10, 0), (10, 7), (10, 3)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
        first, second = bf16_moe._kernels(258, 128, 256, 8, torch.device("cuda"))
        if capability != (10, 7):
            assert not first and not second
        else:
            assert first and second
            assert {k.arch for k in (*first, *second)} == {
                f"sm_{capability[0]}{capability[1]}a"
            }
    # Swapping changes the native signature: a flag alone cannot reinterpret
    # an old normal object, nor can a swapped object omit its orientation.
    for abi, swap in (
        ("cudnn_frost_grouped_gemm2_v1", True),
        ("cudnn_frost_grouped_gemm2_swap_ab_v1", False),
    ):
        with pytest.raises(RuntimeError, match="ABI"):
            runtime._validate_abi(
                {"id": "bad", "abi": abi, "tactic": {"swap_ab": swap}},
                "grouped_gemm2",
            )


def test_bf16_source_distribution_has_no_binary_or_cudnn_frost_dependency():
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.bf16 import (
        runtime,
    )

    root = runtime.artifact_root("bf16")
    manifest = json.loads((root / runtime._MANIFEST).read_text())
    assert manifest["schema_version"] == 2
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


def test_bf16_source_manifest_rejects_tampering_and_legacy_objects(tmp_path):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.bf16 import (
        runtime,
    )

    root = runtime.artifact_root("bf16")
    manifest = json.loads((root / runtime._MANIFEST).read_text())
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


@pytest.fixture
def bf16_compiler_probe(monkeypatch):
    pytest.importorskip("cutlass.cute")
    capabilities._compiler_error.cache_clear()
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    # Compiler admission does not need GPU allocation or kernel compilation.
    monkeypatch.setattr(bf16_moe, "get_compute_capability", lambda device: (10, 7))
    yield
    capabilities._compiler_error.cache_clear()


def _assert_bf16_compiler_rejected(message):
    device = torch.device("cuda", 0)
    with pytest.raises(NotImplementedError, match=message):
        bf16_moe.CudnnFrostBf16MoeRunner(bf16_config(), device).check_support()
    assert bf16_moe.automatic_candidate(bf16_config(), device) is None


def test_bf16_automatic_candidate_declines_missing_source_compiler(
    monkeypatch, bf16_compiler_probe
):
    monkeypatch.setitem(sys.modules, "cutlass.experimental.primitives", None)
    # Newer DSL releases import primitives through tensor_map first. Either
    # import path must report the missing dependency and decline admission.
    _assert_bf16_compiler_rejected(r"cutlass\.experimental\.primitives")


def test_bf16_missing_activation_primitive_preserves_other_activations(
    monkeypatch, bf16_compiler_probe
):
    import cutlass.cute as cute

    monkeypatch.delattr(cute.math, "erf")
    device = torch.device("cuda", 0)
    assert (
        bf16_moe.automatic_candidate(
            replace(bf16_config(), activation=ACTIVATIONS["geglu"]()), device
        )
        is None
    )
    assert bf16_moe.automatic_candidate(bf16_config(), device) is not None


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
    assert bf16_moe.automatic_candidate(bf16_config(), torch.device("cuda", 0)) is None


def test_bf16_compiler_admission_rechecks_arch_override(
    monkeypatch, bf16_compiler_probe
):
    device = torch.device("cuda", 0)
    assert bf16_moe.automatic_candidate(bf16_config(), device) is not None
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100f")
    _assert_bf16_compiler_rejected("CUTE_DSL_ARCH=sm_100f conflicts")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm107a")
    assert bf16_moe.automatic_candidate(bf16_config(), device) is not None


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
    assert bf16_moe.automatic_candidate(bf16_config(), device) is not None
    first = capabilities._compiler_error.cache_info()
    assert bf16_moe.automatic_candidate(bf16_config(), device) is not None
    assert capabilities._compiler_error.cache_info().hits == first.hits + 1


def test_bf16_missing_compiler_preserves_layer_backend(
    monkeypatch, bf16_compiler_probe
):
    from cutlass.experimental import primitives

    monkeypatch.delattr(primitives, "tcgen05_mma")
    cfg = bf16_config(experts=12, intermediate=3072)
    original = SimpleNamespace(
        backend_key="cutlass_bf16",
        supported_routing_modes=bf16_moe.CudnnFrostBf16MoeRunner.supported_routing_modes,
        pack_inputs=lambda act, weights: [act.hidden_states_q],
        launch_kwargs_for=lambda inputs: {},
        forward=lambda inputs, **kwargs: inputs[0],
    )
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.device, layer._arch = cfg, torch.device("cuda", 0), 107
    layer.tuner = SimpleNamespace(is_tuning_mode=True)
    layer.runners, layer._automatic_runners, layer._winners = [original], {}, {}

    def select(act, weights, runners):
        assert runners == [original]
        return original, -1

    monkeypatch.setattr(layer, "_select_winner", select)
    act, weights = bf16_packs(
        tokens=4096, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    assert layer(act, weights) is act.hidden_states_q
    assert layer.winner_backend == "cutlass_bf16"
    assert layer._automatic_runners == {}


def test_bf16_moe_shortlist_limits_stages_and_maps_token_profiles(tmp_path):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
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
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
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


def test_bf16_shortlist_survives_autotuner_plain_tensor_profiles(monkeypatch):
    runner = object.__new__(bf16_moe.CudnnFrostBf16MoeRunner)
    runner._built = True
    runner.config = bf16_config(experts=12, intermediate=3072)
    runner.device = torch.device("meta")
    first = tuple(SimpleNamespace(tactic=("fc1", n)) for n in range(2))
    second = tuple(SimpleNamespace(tactic=("fc2", n)) for n in range(2))
    calls = []

    def select(*args):
        calls.append(args[:5])
        return first, second

    monkeypatch.setattr(bf16_moe, "_selected_kernels", select)
    inputs = [None, torch.empty((16, 7168), device="meta")]
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 4
    assert calls == [(16, 7168, 3072, 12, 2)]


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
        assert (constants["a_dtype"], constants["b_dtype"]) == operands
        assert constants["sf_cutlass_dtype"] == sf_dtype
        assert constants["block_size"] == str(block)
        assert int(constants["cta_group"]) == record["tactic"]["cta_group"]
        for name in ("cta_tile_mnk", "cluster_shape_mnk", "cgrp_tile_mnk"):
            geometry = ast.literal_eval(constants[name])
            assert len(geometry) == 3 and all(value > 0 for value in geometry)
        assert "sm_107a" in constants["frost_compile_options"]
        family = (
            record["arch"],
            record["op"],
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
    assert all(len(variants) == 1 for variants in source_families.values())
    assert len(by_source) <= len(ACTIVATIONS) * 4 + 4
    assert sum(len(digests) for digests in by_source.values()) == len(records)


def test_dtype_directories_are_symmetric_and_isolated():
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
        runtime as shared,
    )
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.bf16 import (
        runtime,
    )
    from flashinfer.fused_moe.auto_candidates import _AUTO_CANDIDATES

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
            _AUTO_CANDIDATES[f"cudnn_frost_{dtype}"].support_module
        )
        assert Path(support.__file__).parent.name == dtype
        payload = json.loads((root / dtype / runtime._MANIFEST).read_text())
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
        f"flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.{dtype}.support"
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
    )
    for tokens, expected in (
        (0, False),
        (1, True),
        (17, True),
        (12288, True),
        (12289, False),
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
        assert not support.is_eligible(
            replace(config, routing=RoutingConfig(num_experts=12, top_k=3)), act, 107
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
    offsets = torch.tensor([0, 0, 1, 128, 128, 130, 259, 259], dtype=torch.int32)
    packed = runtime.pack_token_scales(scales, offsets).view(torch.uint8)
    assert packed.numel() == runtime.segmented_scale_rows(rows, offsets.numel()) * cols
    base = 0
    starts = offsets.tolist() + [rows]
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
        s, n, k, e = 259, 256, 128, 8
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
            runtime.segmented_scale_rows(s, e) * (k // block), dtype=torch.uint8
        )
        b_sf = tuple(torch.empty(e, n * k // block, dtype=torch.uint8) for _ in w)
        offsets = torch.zeros(e, dtype=torch.int32)
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
        gemm_scales = tuple(torch.arange(e, dtype=torch.float32) + 0.5 for _ in w)
        kwargs = {"gemm_scales": gemm_scales} if packed else {}
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
        assert len(args) + 1 == len(names)  # stream supplied at launch
        assert args[0][:5] == ((n, s, k, e, e) if kernel.swap_ab else (s, n, k, e, e))
        ptrs = {
            name: arg.data_ptr() for name, arg in zip(names[1:], args[1:], strict=False)
        }
        tensors = dict(zip(names[1:-1], args[1:], strict=True))
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
        if packed:
            alpha_names = (
                ("gate_alpha", "up_alpha")
                if kernel.gated
                else ("gate_alpha" if kernel.fc1 else "alpha",)
            )
            for name, scalar in zip(alpha_names, gemm_scales, strict=True):
                assert ptrs[name] == scalar.data_ptr()


def test_mxfp8_reject_incompatible_scale_contract(tmp_path):
    payload = {
        "schema_version": 2,
        "kernels": [
            MXFP8_RECORDS[0]
            | {"contract": MXFP8_RECORDS[0]["contract"] | {"block_size": 16}}
        ],
    }
    (tmp_path / "cudnn_frost_selected_kernels.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="numerical contract"):
        mxfp8.discover(tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index", range(len(MXFP8_RECORDS)))
def test_mxfp8_grouped_kernel_and_graph_replay(index, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("packaged block-scale templates target SM107a")
    _assert_mxfp8_grouped_kernel_and_graph(mxfp8, mxfp8.discover()[index], monkeypatch)


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
    offsets = torch.tensor(
        [0, 0, 1, 128, 128, 130, 259, 259], dtype=torch.int32, device="cuda"
    )
    out = torch.empty(s, n, dtype=torch.bfloat16, device="cuda")
    packed_a = runtime.pack_token_scales(a_sf, offsets)
    scale = torch.full((1, 1, 1), 0.5, device="cuda") if kernel.fc1 else None
    plan_type = (
        runtime.PreparedMxfp8Mxfp4GroupedGemm
        if mixed
        else runtime.PreparedMxfp8GroupedGemm
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
        expected = torch.empty(s, n, device="cuda")
        starts = offsets.tolist() + [s]
        for expert, (begin, end) in enumerate(zip(starts, starts[1:], strict=False)):
            values = xd[begin:end].double() @ wd[0][expert].double().T
            if kernel.gated:
                up = xd[begin:end].double() @ wd[1][expert].double().T
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
        torch.tensor([0, 2, 2, 2, 31, 259, 259, 259], device="cuda", dtype=torch.int32)
    )
    packed_a.copy_(runtime.pack_token_scales(a_sf, offsets))
    if mixed:
        for weight in w:
            weight.bitwise_xor_(0x88)
    if scale is not None:
        scale.fill_(0.25)
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, reference(), rtol=2e-2, atol=2e-4)


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
            (x[token] @ up_gate.T).float(), activation, i
        )
        q, sf = mxfp8_quantize(mid, is_sf_swizzled_layout=False, alignment=32)
        mid = dequant(q, sf.reshape(-1, i // 32)[: token.numel()])
        value = (mid @ down.T).bfloat16()
        expanded[token, slot] = value.float() * scores[token, slot, None]
    return expanded.sum(dim=1).bfloat16()


def _pick_stage_pair(pool):
    # Prefer different retained launch ABIs without assuming selection retained
    # a particular store, orientation, CTAMMA group, or geometry.
    assert len(pool) >= 2
    first = pool[0]
    second = max(
        pool[1:],
        key=lambda kernel: (
            kernel.swap_ab != first.swap_ab,
            kernel.tactic_metadata["store_mode"] != first.tactic_metadata["store_mode"],
        ),
    )
    return first, second


@supported_gpu
@pytest.mark.parametrize("name", ACTIVATIONS)
@pytest.mark.parametrize("swizzled", [False, True])
def test_mxfp8_moe_full_pipeline_and_graph(name, swizzled, monkeypatch):
    _assert_mxfp8_moe_full_pipeline_and_graph(name, swizzled, monkeypatch)


def _assert_mxfp8_moe_full_pipeline_and_graph(name, swizzled, monkeypatch, mixed=False):
    from flashinfer.fused_moe import (
        CutlassMxfp8Config,
        CutlassMxfp8Mxfp4Config,
        QuantFormat,
    )

    dtype = "mxfp8_mxfp4" if mixed else "mxfp8"
    moe = importlib.import_module(
        f"flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.{dtype}.moe"
    )
    backend = CutlassMxfp8Mxfp4Config if mixed else CutlassMxfp8Config

    monkeypatch.setitem(sys.modules, "cudnn", None)
    torch.manual_seed(77)
    t, h, i, e, k = 17, 256, 256, 4, 2
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
    ids = torch.randint(0, e // 2, (t, k), device="cuda", dtype=torch.int32)
    scores = torch.rand(t, k, device="cuda")
    scores /= scores.sum(-1, keepdim=True)
    act = MoEActivationPack(xq, xsf, ids, scores)
    weights = MoEWeightPack({f"cutlass_{dtype}": view})
    first, second = moe._kernels(t * k, h, i, e, xq.device, activation)

    selected = _pick_stage_pair(first), _pick_stage_pair(second)
    # Test full launch mechanics on small allocations independently of the
    # measured model-size shortlist. Admission is covered separately below.
    monkeypatch.setattr(moe, "_selected_kernels", lambda *args: selected)
    runner_type = (
        moe.CudnnFrostMxfp8Mxfp4MoeRunner if mixed else moe.CudnnFrostMxfp8MoeRunner
    )
    runner = runner_type(config, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 4
    assert runner.get_valid_tactics(list(inputs), None) == tactics
    reference = _mxfp8_moe_reference(act, weights, activation, swizzled, mixed)
    assert reference.float().norm() > 0
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
        ids[::3] = -1
        ids[1::3] = e
        scores.mul_(0.5)
        xq.copy_((xq.float() * 0.5).to(torch.float8_e4m3fn))
        # E8M0 bytes, including the swizzled input, are consumed on replay.
        xsf.view(torch.uint8).add_(1)
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
    if (
        name == "swiglu"
        and not swizzled
        and not torch.is_inference(view["fc1_expert_scales"])
    ):
        old_scales = inputs[6].clone()
        view["fc1_expert_scales"].view(torch.uint8).add_(1)
        new_inputs = runner.pack_inputs(act, weights)
        assert new_inputs[6].data_ptr() != inputs[6].data_ptr()
        assert torch.equal(inputs[6], old_scales)
        reference = _mxfp8_moe_reference(act, weights, activation, swizzled, mixed)
        out = runner.forward(new_inputs)
        assert (
            out.float() - reference.float()
        ).norm() / reference.float().norm() < 0.02
    # Inference tensors are immutable prepared weights. Replacing their
    # unversioned global scale must still trigger identity validation.
    if torch.is_inference(view["fc1_input_scale"]):
        view["fc1_input_scale"] = view["fc1_input_scale"] * 2
    else:
        view["fc1_input_scale"].mul_(2)
    with pytest.raises(ValueError, match="identity"):
        runner.pack_inputs(act, weights)


@supported_gpu
def test_mxfp8_moe_inference_mode_preparation_and_graph(monkeypatch):
    with torch.inference_mode():
        test_mxfp8_moe_full_pipeline_and_graph("swiglu", False, monkeypatch)


def test_mxfp8_moe_shortlist_uses_measured_two_by_two_buckets(monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8 import (
        moe,
    )

    _assert_moe_shortlist_buckets(moe, monkeypatch)


def _assert_moe_shortlist_buckets(moe, monkeypatch):
    key = ("sm_107a", "swiglu", 8, 4096, 14336, 2)
    first = tuple(SimpleNamespace(artifact_id=f"first{j}") for j in range(3))
    second = tuple(SimpleNamespace(artifact_id=f"second{j}") for j in range(3))
    monkeypatch.setattr(moe.common, "_arch_for", lambda device: "sm_107a")
    monkeypatch.setattr(moe, "_kernels", lambda *args: (first, second))
    table = {key: {128: (("first2", "first0"), ("second1", "second2"))}}
    monkeypatch.setattr(moe, "_read", lambda roots: table)
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
        shortlist,
    )

    monkeypatch.setattr(shortlist, "_read", lambda roots: table)
    args = (4096, 14336, 8, 2, "cuda", SwiGLU())
    for tokens in (1, 17, 127, 128, 129, 12288):
        assert moe._selected_kernels(tokens, *args) == (
            (first[2], first[0]),
            (second[1], second[2]),
        )
    for tokens in (0, 12289):
        assert moe._selected_kernels(tokens, *args) == ((), ())
    assert moe._selected_kernels(128, 4096, 14336, 9, 2, "cuda", SwiGLU()) == ((), ())
    table[key][128] = (("first2",), ("second1", "second2"))
    assert moe._selected_kernels(128, *args) == ((), ())
    table.clear()
    assert moe._selected_kernels(128, *args) == ((), ())


def test_mxfp8_plan_workspaces_preserve_prior_allocations(monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8 import (
        moe,
    )

    required_sizes = iter((257, 301, 400, 129, 513, 600, 599, 450, 350, 401, 450, 449))
    retained_modules = []

    def make_plan(*args):
        size = next(required_sizes)
        module = {"workspace_size": lambda: size, "run": lambda *unused: None}
        retained_modules.append(module)
        return module

    monkeypatch.setattr(
        moe, "_module", lambda arch: SimpleNamespace(make_plan=make_plan)
    )
    monkeypatch.setattr(moe.common, "_arch_for", lambda device: "sm_107a")
    monkeypatch.setattr(moe.common, "_load_kernel", lambda kernel, device: kernel)
    pool = {}
    first = tuple(
        SimpleNamespace(
            tactic=("fc1", j),
            workspace_bytes=128,
            gated=True,
            launch_tail=("output", "scale"),
            swap_ab=bool(j),
        )
        for j in range(2)
    )
    second = tuple(
        SimpleNamespace(tactic=("fc2", j), workspace_bytes=128, swap_ab=bool(j))
        for j in range(2)
    )

    def prepare(tokens):
        return moe._Plans(
            tokens, 256, 256, 4, 2, torch.device("cpu"), first, second, pool
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
    # Each captured launch must keep the native module that owns its function.
    assert list(before.plans.values()) == retained_modules[:4]
    assert list(grown.plans.values()) == retained_modules[4:8]


def test_nvfp4_geometry_sharing_and_roundtrip_without_cudnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    root = nvfp4.runtime.artifact_root("nvfp4")
    records = json.loads((root / nvfp4.runtime._MANIFEST).read_text())["kernels"]
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
def test_nvfp4_reject_incompatible_numerical_contract(change, tmp_path):
    root = nvfp4.runtime.artifact_root("nvfp4")
    record = json.loads((root / nvfp4.runtime._MANIFEST).read_text())["kernels"][0]
    payload = {
        "schema_version": 2,
        "kernels": [record | {"contract": record["contract"] | change}],
    }
    (tmp_path / nvfp4.runtime._MANIFEST).write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="numerical contract"):
        nvfp4.discover(tmp_path)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_nvfp4_auto_admission_uses_logical_hidden_size(name):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.nvfp4 import (
        support,
    )
    from flashinfer.fused_moe import CutlassNvfp4Config, QuantFormat

    config = MoEConfig(
        routing=RoutingConfig(num_experts=12, top_k=2),
        quant=QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=3072),
        backend=BackendOptions((CutlassNvfp4Config(),)),
        activation=ACTIVATIONS[name](),
    )
    for tokens, expected in (
        (0, False),
        (1, True),
        (17, True),
        (12288, True),
        (12289, False),
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
                    tokens, 7168, dtype=torch.uint8, device="meta"
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


@supported_gpu
@pytest.mark.parametrize("name", [*ACTIVATIONS, "fc2"])
@pytest.mark.parametrize(
    "swap_ab,store", [(False, "stg"), (False, "tma"), (True, "stg"), (True, "tma")]
)
def test_nvfp4_grouped_kernel_extremes_alpha_and_graph(
    name, swap_ab, store, monkeypatch
):
    from tests.moe.utils import compute_reference_activation

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
        [0, 0, 1, 128, 128, 130, 259, 259], dtype=torch.int32, device="cuda"
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
        expected = torch.empty(s, n, device="cuda", dtype=torch.float32)
        starts = offsets.tolist() + [s]
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
        torch.tensor([0, 2, 2, 2, 31, 259, 259, 259], device="cuda", dtype=torch.int32)
    )
    packed_a.copy_(nvfp4.pack_token_scales(a_sf, offsets))
    for j, alpha in enumerate(alphas):
        alpha.mul_(0.5 if j == 0 else 1.5)
    if scale is not None:
        scale.fill_(0.25)
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, reference(), rtol=2e-2, atol=2e-4)


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


@supported_gpu
@pytest.mark.parametrize("name", ACTIVATIONS)
@pytest.mark.parametrize("swizzled", [False, True])
def test_nvfp4_moe_full_pipeline_and_dynamic_graph(name, swizzled, monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.nvfp4 import (
        moe,
    )
    from flashinfer.fused_moe import CutlassNvfp4Config, QuantFormat

    monkeypatch.setitem(sys.modules, "cudnn", None)
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "0")
    torch.manual_seed(117)
    t, h, i, e, k = 17, 256, 256, 4, 2
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
    view["fc1_dequant_scale"].copy_(torch.tensor([0.5, 0.75, 1.25, 1.5], device="cuda"))
    view["fc2_dequant_scale"].copy_(
        torch.tensor([1.0, 0.25, 0.75, 0.125], device="cuda")
    )
    view["fc2_act_global_scale"].fill_(2.0)
    ids = torch.randint(0, e // 2, (t, k), device="cuda", dtype=torch.int32)
    scores = torch.rand(t, k, device="cuda")
    scores /= scores.sum(-1, keepdim=True)
    act = MoEActivationPack(xq, xsf, ids, scores)
    weights = MoEWeightPack({"cutlass_nvfp4": view})
    first, second = moe._kernels(t * k, h, i, e, xq.device, activation)
    selected = _pick_stage_pair(first), _pick_stage_pair(second)
    monkeypatch.setattr(moe, "_selected_kernels", lambda *args: selected)
    runner = moe.CudnnFrostNvfp4MoeRunner(config, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 4
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
        view["fc1_dequant_scale"].mul_(0.75)
        global_multiplier = 2.0 if iteration % 2 == 0 else 0.5
        view["fc2_dequant_scale"].mul_(1.25 / global_multiplier)
        view["fc2_act_global_scale"].mul_(global_multiplier)
        out.fill_(float("nan"))
        graph.replay()
        check(out)
    with pytest.raises(ValueError, match="launch_state"):
        runner.forward(list(inputs))


@supported_gpu
def test_nvfp4_moe_inference_mode_preparation_and_graph(monkeypatch):
    with torch.inference_mode():
        test_nvfp4_moe_full_pipeline_and_dynamic_graph("swiglu", False, monkeypatch)


def test_nvfp4_moe_shortlist_uses_measured_two_by_two_buckets(monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.nvfp4 import (
        moe,
    )

    _assert_moe_shortlist_buckets(moe, monkeypatch)


def test_nvfp4_packaged_shortlists_resolve_all_profiles(monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.nvfp4 import (
        moe,
    )

    _assert_packaged_shortlists_resolve_all_profiles(moe, monkeypatch)


def _assert_packaged_shortlists_resolve_all_profiles(moe, monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
        shortlist,
    )

    roots = moe._artifact_roots()
    table = shortlist._read(roots)
    assert table
    assert {key[1] for key in table} == set(ACTIVATIONS)
    monkeypatch.setattr(moe.common, "_arch_for", lambda device: "sm_107a")
    for (arch, name, experts, hidden, intermediate, topk), profiles in table.items():
        assert arch == "sm_107a"
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
                assert len(first) == len(second) == 2
                assert tuple(k.artifact_id for k in first) == identities[0]
                assert tuple(k.artifact_id for k in second) == identities[1]
                assert all(k.fc1 and k.activation == name for k in first)
                assert all(not k.fc1 for k in second)
                assert len({(a.tactic, b.tactic) for a in first for b in second}) == 4
            previous = tokens


def test_mxfp8_mxfp4_geometry_sharing_and_roundtrip_without_cudnn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    root = mxfp8_mxfp4.runtime.artifact_root("mxfp8_mxfp4")
    records = json.loads((root / mxfp8_mxfp4.runtime._MANIFEST).read_text())["kernels"]
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
    ],
)
def test_mxfp8_mxfp4_reject_incompatible_numerical_contract(change, tmp_path):
    root = mxfp8_mxfp4.runtime.artifact_root("mxfp8_mxfp4")
    record = json.loads((root / mxfp8_mxfp4.runtime._MANIFEST).read_text())["kernels"][
        0
    ]
    payload = {
        "schema_version": 2,
        "kernels": [record | {"contract": record["contract"] | change}],
    }
    (tmp_path / mxfp8_mxfp4.runtime._MANIFEST).write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="numerical contract"):
        mxfp8_mxfp4.discover(tmp_path)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_mxfp8_mxfp4_auto_admission_uses_logical_hidden_size(name):
    _assert_mxfp8_auto_admission(name, mixed=True)


@supported_gpu
@pytest.mark.parametrize("name", [*ACTIVATIONS, "fc2"])
@pytest.mark.parametrize(
    "swap_ab,store", [(False, "stg"), (False, "tma"), (True, "stg"), (True, "tma")]
)
def test_mxfp8_mxfp4_grouped_kernel_extremes_and_graph(
    name, swap_ab, store, monkeypatch
):
    candidates = [
        kernel
        for kernel in mxfp8_mxfp4.discover()
        if (kernel.activation if kernel.fc1 else "fc2") == name
        and kernel.swap_ab == swap_ab
        and kernel.tactic_metadata["store_mode"] == store
    ]
    if not candidates:
        pytest.skip(f"shortlist retains no {name}/{swap_ab=}/{store} family")
    _assert_mxfp8_grouped_kernel_and_graph(
        mxfp8_mxfp4, candidates[0], monkeypatch, mixed=True
    )


@supported_gpu
@pytest.mark.parametrize("name", ACTIVATIONS)
@pytest.mark.parametrize("swizzled", [False, True])
def test_mxfp8_mxfp4_moe_full_pipeline_and_dynamic_graph(name, swizzled, monkeypatch):
    _assert_mxfp8_moe_full_pipeline_and_graph(name, swizzled, monkeypatch, mixed=True)


@supported_gpu
def test_mxfp8_mxfp4_moe_inference_mode_preparation_and_graph(monkeypatch):
    with torch.inference_mode():
        test_mxfp8_mxfp4_moe_full_pipeline_and_dynamic_graph(
            "swiglu", False, monkeypatch
        )


def test_mxfp8_mxfp4_moe_shortlist_uses_measured_two_by_two_buckets(monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8_mxfp4 import (
        moe,
    )

    _assert_moe_shortlist_buckets(moe, monkeypatch)


def test_mxfp8_mxfp4_packaged_shortlists_resolve_all_profiles(monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.mxfp8_mxfp4 import (
        moe,
    )

    _assert_packaged_shortlists_resolve_all_profiles(moe, monkeypatch)


def test_frost_external_compiler_identity_tracks_executable_changes(
    tmp_path, monkeypatch
):
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
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
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
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
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
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
    from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm import (
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
