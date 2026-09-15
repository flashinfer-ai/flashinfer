"""Core correctness and unchanged-API integration tests for cuDNN Frost BF16 MoE."""

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

from flashinfer.experimental.cudnn_frost_selected_kernels import capabilities, moe
from flashinfer.experimental.cudnn_frost_selected_kernels.activations import (
    ACTIVATIONS,
    activation_name,
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

supported_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="packaged cuDNN Frost sources target SM107a",
)


def config(topk=2, experts=8, intermediate=256, ceiling=16384):
    return MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=topk),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=intermediate),
        backend=BackendOptions((CutlassBf16Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=ceiling),
    )


def packs(tokens=129, topk=2, experts=8, hidden=128, intermediate=256, device="cuda"):
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


def _moe_reference(act, weights, activation=None):
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
def test_extended_activations_graph_and_routing(name, tokens, topk, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Extended activation sources have been selected on SM107")
    monkeypatch.setitem(sys.modules, "cudnn", None)
    torch.manual_seed(81)
    activation = ACTIVATIONS[name]()
    cfg = replace(config(topk), activation=activation)
    act, weights = packs(tokens=tokens, topk=topk)
    view = weights.get_view("cutlass_bf16")
    if not activation.is_gated:
        view["fc1_expert_weights"] = view["fc1_expert_weights"][:, :256].contiguous()
    # Drive Step clamps and SiTU saturation, with both positive and negative
    # accumulator values; small random inputs would hide these semantics.
    act.hidden_states_q.mul_(8)
    view["fc1_expert_weights"].mul_(16)
    runner = moe.CudnnFrostBf16MoeRunner(cfg, "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    first, _ = moe._kernels(
        tokens * topk, 128, 256, 8, act.hidden_states_q.device, activation
    )
    assert {k.activation for k in first} == {name}
    assert {k.swap_ab for k in first} == {False, True}
    reference = _moe_reference(act, weights, activation)
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
        reference = _moe_reference(act, weights, activation)
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
def test_extended_activation_scalars_are_not_silently_dropped(activation):
    with pytest.raises(NotImplementedError, match="default activation parameters"):
        activation_name(activation)


def test_activation_artifact_pools_are_disjoint(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 7))
    seen = set()
    shared_fc2 = None
    for name, cls in ACTIVATIONS.items():
        first, second = moe._kernels(258, 128, 256, 8, torch.device("cuda"), cls())
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
def test_all_compound_tactics_full_moe_graph_and_dynamic_routing(topk, monkeypatch):
    # Neither the cuDNN Frost compiler nor the CUTLASS execution API is needed.
    monkeypatch.setitem(sys.modules, "cudnn", None)
    monkeypatch.setitem(sys.modules, "cudnn.gemm.frost.compiler", None)
    import flashinfer.fused_moe as fused

    def forbidden(*args, **kwargs):
        raise AssertionError("independent cuDNN Frost must not invoke CUTLASS")

    monkeypatch.setattr(fused, "cutlass_fused_moe", forbidden)
    torch.manual_seed(59)
    act, weights = packs(topk=topk)
    runner = moe.CudnnFrostBf16MoeRunner(config(topk), "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    expected = _moe_reference(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 16
    if torch.cuda.get_device_capability() == (10, 7):
        first, second = moe._kernels(129 * topk, 128, 256, 8, torch.device("cuda"))
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
        expected = _moe_reference(act, weights)
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
        torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-2)
        # Invalid ids must not escape the workspace or become weight addresses.
        act.topk_ids[::2] = -1
        act.topk_ids[1::4] = 8
        graph.replay()
        torch.testing.assert_close(
            out, _moe_reference(act, weights), atol=2e-4, rtol=2e-2
        )
    torch.cuda.current_stream().wait_stream(stream)


@supported_gpu
def test_interleaved_packs_do_not_exchange_weights_and_reject_stale_tactics():
    runner = moe.CudnnFrostBf16MoeRunner(config(), "cuda")
    runner.check_support()
    runner.build()
    a, wa = packs(tokens=17)
    b, wb = packs(tokens=33)
    pb = runner.pack_inputs(b, wb)
    pa = runner.pack_inputs(a, wa)
    assert pa.tuning_config.cuda_graph_profile_replays == 3
    assert pa.launch_state.workspace.data_ptr() == pb.launch_state.workspace.data_ptr()
    assert len(runner._workspace_pool) == 1
    # Simulate the plain-list inputs synthesized by the autotuner.
    runner.forward(list(pa), **runner.launch_kwargs_for(pa))
    runner.forward(pb)
    torch.testing.assert_close(pa[0], _moe_reference(a, wa), atol=2e-4, rtol=2e-2)
    torch.testing.assert_close(pb[0], _moe_reference(b, wb), atol=2e-4, rtol=2e-2)
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
def test_model_geometry_artifacts_ragged_graph(
    experts, hidden, intermediate, monkeypatch
):
    """Exercise every packaged pair on partial tiles and initially empty experts."""
    if experts == 64 and torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("E64 artifacts are packaged for Rubin only")
    monkeypatch.setitem(sys.modules, "cudnn.gemm.frost.compiler", None)
    torch.manual_seed(97)
    act, weights = packs(
        tokens=129, experts=experts, hidden=hidden, intermediate=intermediate
    )
    runner = moe.CudnnFrostBf16MoeRunner(
        config(experts=experts, intermediate=intermediate), "cuda"
    )
    runner.check_support()
    runner.build()
    packed = runner.pack_inputs(act, weights)
    expected = _moe_reference(act, weights)
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
    expected = _moe_reference(act, weights)
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
def test_unsupported_semantics_are_not_automatic_candidates(overrides):
    assert moe.automatic_candidate(replace(config(), **overrides), "cuda") is None


def test_layer_adds_independent_candidate_and_separates_winner_cache(monkeypatch):
    # Dispatch-policy test without allocating multi-GB weights or benchmarking.
    cfg = config(experts=12, intermediate=3072)
    calls = []

    class FakeRunner:
        supported_routing_modes = moe.CudnnFrostBf16MoeRunner.supported_routing_modes

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
    monkeypatch.setattr(moe, "automatic_candidate", lambda *args: cudnn_frost)

    def select(act, weights, runners):
        calls.append([r.backend_key for r in runners])
        return runners[-1], -1

    monkeypatch.setattr(layer, "_select_winner", select)
    act, weights = packs(
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
    act2, weights2 = packs(
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
def test_original_layer_api_can_execute_winning_cudnn_frost_and_replay(
    experts, hidden, intermediate, monkeypatch
):
    from flashinfer.autotuner import autotune

    torch.manual_seed(83)
    act, weights = packs(
        tokens=4096, experts=experts, hidden=hidden, intermediate=intermediate
    )
    layer = MoELayer(config(experts=experts, intermediate=intermediate))
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
    expected = _moe_reference(act, weights)
    rel_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel_l2.item() < 0.01
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(act, weights)
    act.topk_ids.fill_(experts - 1)
    act.topk_weights.mul_(0.5)
    graph.replay()
    expected = _moe_reference(act, weights)
    rel_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel_l2.item() < 0.01
    assert len(visited) == 2  # cached winning runner, no re-tune on replay


@pytest.mark.parametrize("activation", list(ACTIVATIONS.values()))
def test_auto_admission_uses_validated_architecture_profiles(activation):
    from flashinfer.experimental.cudnn_frost_selected_kernels.support import (
        large_bf16_moe,
    )

    cfg = replace(config(experts=12, intermediate=3072), activation=activation())
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
        act, _ = packs(
            tokens=tokens, experts=12, hidden=hidden, intermediate=3072, device="meta"
        )
        assert large_bf16_moe(cfg, act, arch) == expected
    cfg = config(experts=8, intermediate=14336)
    for tokens, expected in [(4095, False), (4096, False)]:
        act, _ = packs(
            tokens=tokens, experts=8, hidden=4096, intermediate=14336, device="meta"
        )
        assert large_bf16_moe(cfg, act, 100) == expected
    cfg = config(topk=6, experts=64, intermediate=1408)
    act, _ = packs(
        tokens=4096,
        topk=6,
        experts=64,
        hidden=2048,
        intermediate=1408,
        device="meta",
    )
    assert not large_bf16_moe(cfg, act, 100)  # unsupported architecture
    assert large_bf16_moe(cfg, act, 107)  # four shortlisted plans compete normally


def test_artifact_architecture_and_swap_abi_isolation(monkeypatch):
    from flashinfer.experimental.cudnn_frost_selected_kernels import runtime

    for capability in ((10, 0), (10, 7), (10, 3)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
        first, second = moe._kernels(258, 128, 256, 8, torch.device("cuda"))
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


def test_source_distribution_has_no_binary_or_cudnn_frost_dependency():
    from flashinfer.experimental.cudnn_frost_selected_kernels import runtime

    root = Path(runtime.__file__).parent / "artifacts"
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


def test_source_manifest_rejects_tampering_and_legacy_objects(tmp_path):
    from flashinfer.experimental.cudnn_frost_selected_kernels import runtime

    root = Path(runtime.__file__).parent / "artifacts"
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
def compiler_probe(monkeypatch):
    pytest.importorskip("cutlass.cute")
    capabilities._compiler_error.cache_clear()
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    # Compiler admission does not need GPU allocation or kernel compilation.
    monkeypatch.setattr(moe, "get_compute_capability", lambda device: (10, 7))
    yield
    capabilities._compiler_error.cache_clear()


def _assert_compiler_rejected(message):
    device = torch.device("cuda", 0)
    with pytest.raises(NotImplementedError, match=message):
        moe.CudnnFrostBf16MoeRunner(config(), device).check_support()
    assert moe.automatic_candidate(config(), device) is None


def test_automatic_candidate_declines_missing_source_compiler(
    monkeypatch, compiler_probe
):
    monkeypatch.setitem(sys.modules, "cutlass.experimental.primitives", None)
    # Newer DSL releases import primitives through tensor_map first. Either
    # import path must report the missing dependency and decline admission.
    _assert_compiler_rejected(r"cutlass\.experimental\.primitives")


def test_missing_activation_primitive_preserves_other_activations(
    monkeypatch, compiler_probe
):
    import cutlass.cute as cute

    monkeypatch.delattr(cute.math, "erf")
    device = torch.device("cuda", 0)
    assert (
        moe.automatic_candidate(
            replace(config(), activation=ACTIVATIONS["geglu"]()), device
        )
        is None
    )
    assert moe.automatic_candidate(config(), device) is not None


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
def test_compiler_admission_missing_symbol(
    module_name, symbol, monkeypatch, compiler_probe
):
    module = importlib.import_module(module_name)
    monkeypatch.delattr(module, symbol)
    _assert_compiler_rejected(symbol)


def test_compiler_admission_missing_enum_member(monkeypatch, compiler_probe):
    from cutlass.experimental import primitives

    original = primitives.Tcgen05MMACollectorOp
    monkeypatch.setattr(
        primitives,
        "Tcgen05MMACollectorOp",
        SimpleNamespace(FILL=original.FILL, USE=original.USE),
    )
    _assert_compiler_rejected("Tcgen05MMACollectorOp.LASTUSE")


@pytest.mark.parametrize("case", ["primitive_keyword", "ffi_keyword", "not_callable"])
def test_compiler_admission_incompatible_signature(case, monkeypatch, compiler_probe):
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
    _assert_compiler_rejected(message)


def test_compiler_admission_requires_native_arch(monkeypatch, compiler_probe):
    import cutlass.cute as cute

    gpu_arch = cute.GPUArch

    def older_gpu_arch(arch):
        if arch == "sm_107a":
            raise ValueError("unknown GPU architecture sm_107a")
        return gpu_arch(arch)

    monkeypatch.setattr(cute, "GPUArch", older_gpu_arch)
    _assert_compiler_rejected("cannot target sm_107a")
    # An unsupported device must not fall back to a different architecture.
    monkeypatch.setattr(moe, "get_compute_capability", lambda device: (10, 0))
    assert moe.automatic_candidate(config(), torch.device("cuda", 0)) is None


def test_compiler_admission_rechecks_arch_override(monkeypatch, compiler_probe):
    device = torch.device("cuda", 0)
    assert moe.automatic_candidate(config(), device) is not None
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100f")
    _assert_compiler_rejected("CUTE_DSL_ARCH=sm_100f conflicts")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm107a")
    assert moe.automatic_candidate(config(), device) is not None


def test_compiler_admission_uses_capabilities_without_compiling(
    monkeypatch, compiler_probe
):
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler capability checks must not compile kernels")

    monkeypatch.setattr(cute, "compile", forbidden)
    monkeypatch.setitem(sys.modules, "cudnn", None)
    monkeypatch.setattr(moe, "get_compute_capability", lambda device: (10, 7))
    device = torch.device("cuda", 0)
    assert moe.automatic_candidate(config(), device) is not None
    first = capabilities._compiler_error.cache_info()
    assert moe.automatic_candidate(config(), device) is not None
    assert capabilities._compiler_error.cache_info().hits == first.hits + 1


def test_missing_compiler_preserves_layer_backend(monkeypatch, compiler_probe):
    from cutlass.experimental import primitives

    monkeypatch.delattr(primitives, "tcgen05_mma")
    cfg = config(experts=12, intermediate=3072)
    original = SimpleNamespace(
        backend_key="cutlass_bf16",
        supported_routing_modes=moe.CudnnFrostBf16MoeRunner.supported_routing_modes,
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
    act, weights = packs(
        tokens=4096, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    assert layer(act, weights) is act.hidden_states_q
    assert layer.winner_backend == "cutlass_bf16"
    assert layer._automatic_runners == {}


def test_moe_shortlist_limits_stages_and_maps_token_profiles(tmp_path):
    from flashinfer.experimental.cudnn_frost_selected_kernels import shortlist

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


def test_moe_shortlist_rejects_more_than_two_stage_candidates(tmp_path):
    from flashinfer.experimental.cudnn_frost_selected_kernels import shortlist

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


def test_shortlist_survives_autotuner_plain_tensor_profiles(monkeypatch):
    runner = object.__new__(moe.CudnnFrostBf16MoeRunner)
    runner._built = True
    runner.config = config(experts=12, intermediate=3072)
    runner.device = torch.device("meta")
    first = tuple(SimpleNamespace(tactic=("fc1", n)) for n in range(2))
    second = tuple(SimpleNamespace(tactic=("fc2", n)) for n in range(2))
    calls = []

    def select(*args):
        calls.append(args[:5])
        return first, second

    monkeypatch.setattr(moe, "_selected_kernels", select)
    inputs = [None, torch.empty((16, 7168), device="meta")]
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 4
    assert calls == [(16, 7168, 3072, 12, 2)]
