"""Current-workload tuning contracts for the planned MLA wrapper."""

import math

import pytest
import torch

import flashinfer
from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2
from flashinfer.autotuner import AutoTuner
from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
from flashinfer.mla._batch_mla import _auto_policy as policy
from flashinfer.mla._batch_mla._backends import trtllm_gen_backend as backend
from flashinfer.mla._batch_mla._backends.cute_dsl_monolithic_backend import (
    _BatchMLAPagedAttentionCuteDslMonolithicBackend,
)


@pytest.fixture
def tuner(monkeypatch):
    # Isolate decisions without clearing another test's process-global cache.
    instance = AutoTuner(warmup=1, repeat=2)
    monkeypatch.setattr(AutoTuner, "_instance", instance)
    return instance


@pytest.fixture
def trt_only(monkeypatch):
    # Tests of TRT-specific persistent resources keep a single concrete backend.
    monkeypatch.setattr(
        policy._BatchMLAPagedAttentionAutotuneBackend,
        "_candidate_types",
        (backend._BatchMLAPagedAttentionTrtllmGenBackend,),
    )


@pytest.fixture
def case():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("requires SM100, SM103 or SM107")
    torch.manual_seed(928)
    return dict(
        query=torch.randn(3, 128, 576, device="cuda", dtype=torch.bfloat16),
        kv=torch.randn(8, 32, 576, device="cuda", dtype=torch.bfloat16),
        tables=torch.tensor(
            [[3, 0, 2, 1], [6, 4, 7, 5]], dtype=torch.int32, device="cuda"
        ),
        lengths=[47, 91],
        offsets=[0, 1, 2],
    )


def _plan(wrapper, case, *, lse=False, **changes):
    args = dict(
        metadata=MLAPlanMetadata.dense(
            torch.tensor(case["offsets"], dtype=torch.int32, device="cuda"),
            case["tables"],
            torch.tensor(case["lengths"], dtype=torch.int32, device="cuda"),
            max_q_len=max(
                b - a
                for a, b in zip(case["offsets"], case["offsets"][1:], strict=False)
            ),
        ),
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=32,
        causal=True,
        sm_scale=0.125,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        query_layout="packed",
        kv_cache_layout="packed",
        lse_mode="base2" if lse else "none",
        scale_mode="bmm-scalar",
        enable_pdl=False,
    )
    args.update(changes)
    wrapper.plan(**args)


def _wrapper(case, *, lse=False, graph=False):
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchMLAPagedAttentionWrapper(
        workspace, backend="autotune", use_cuda_graph=graph
    )
    _plan(wrapper, case, lse=lse)
    return wrapper


def _inputs(case, *, lse=False, scale=0.125):
    total = case["offsets"][-1]
    return dict(
        query=case["query"][:total],
        kv_cache=case["kv"],
        out=torch.full(
            (total, 128, 512), float("nan"), device="cuda", dtype=torch.bfloat16
        ),
        lse=(
            torch.full((total, 128), float("nan"), device="cuda", dtype=torch.float32)
            if lse
            else None
        ),
        return_lse=lse,
        bmm1_scale=scale,
        bmm2_scale=0.75,
    )


def _reference(case, scale=0.125, output_scale=0.75):
    outputs, lses = [], []
    for batch, (start, end) in enumerate(
        zip(case["offsets"], case["offsets"][1:], strict=False)
    ):
        query = case["query"][start:end].float()
        kv = case["kv"][case["tables"][batch].long()].reshape(-1, 576)
        kv = kv[: case["lengths"][batch]].float()
        scores = torch.einsum("qhd,kd->qhk", query, kv) * scale
        query_positions = (
            torch.arange(end - start, device="cuda") + len(kv) - (end - start)
        )
        mask = torch.arange(len(kv), device="cuda")[None, :] > query_positions[:, None]
        scores.masked_fill_(mask[:, None, :], -float("inf"))
        probabilities = scores.softmax(-1)
        # Bottom-right causality can leave a query row with no visible keys.
        probabilities = torch.where(
            query_positions[:, None, None] >= 0, probabilities, 0.0
        )
        outputs.append(
            torch.einsum("qhk,kd->qhd", probabilities, kv[:, :512]) * output_scale
        )
        lses.append(scores.logsumexp(-1) / math.log(2))
    return torch.cat(outputs), torch.cat(lses)


def _assert_result(case, inputs, result):
    expected, expected_lse = _reference(
        case, inputs["bmm1_scale"], inputs["bmm2_scale"]
    )
    out = result[0] if inputs["return_lse"] else result
    assert out is inputs["out"]
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    if inputs["return_lse"]:
        assert result[1] is inputs["lse"]
        torch.testing.assert_close(result[1], expected_lse, rtol=1e-2, atol=1e-2)


def _prefer_backend(tuner, monkeypatch, winner):
    measured = set()

    def measure(runner, tensors, tactic, config, **kwargs):
        # Execute real kernels with deterministic timings so selection tests
        # do not depend on GPU load or a backend's actual performance.
        runner(tensors, tactic=tactic)
        measured.add(runner._backend)
        return 1.0 if runner._backend == winner else 100.0

    monkeypatch.setattr(tuner, "_profile_single_kernel", measure)
    return measured


def _forbidden(*args, **kwargs):
    raise AssertionError("warm execution performed tuning or persistent preparation")


@pytest.mark.parametrize("ragged", [False, True])
def test_current_workload_tuning_and_warm_execution(
    case, tuner, monkeypatch, ragged, trt_only
):
    if ragged:
        case["offsets"] = [0, 1, 3]
    wrapper = _wrapper(case, lse=not ragged)
    inputs = _inputs(case, lse=not ragged)
    with pytest.raises(RuntimeError, match="flashinfer.autotune"):
        wrapper.run(**inputs)
    assert torch.isnan(inputs["out"]).all()
    with flashinfer.autotune(True):
        _assert_result(case, inputs, wrapper.run(**inputs))
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    for name in (
        "get_trtllm_gen_fmha_module",
        "_get_trtllm_gen_multi_ctas_kv_counter_buffer",
    ):
        monkeypatch.setattr(backend, name, _forbidden)
    case["query"].mul_(0.5)
    _assert_result(case, inputs, wrapper.run(**inputs))


@pytest.mark.parametrize(
    "winner", ["trtllm-gen", "cute-dsl-monolithic", "cute-dsl-modular"]
)
def test_warm_runs_keep_selection_and_use_current_options(
    case, tuner, monkeypatch, winner
):
    _prefer_backend(tuner, monkeypatch, winner)
    wrapper = _wrapper(case)
    inputs = _inputs(case)
    with flashinfer.autotune(True):
        wrapper.run(**inputs)
    selection = wrapper._planned_backend._selection
    assert selection[0]._backend == winner
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    monkeypatch.setattr(policy, "_tensor_signature", _forbidden)
    monkeypatch.setattr(selection[0], "configure_tuning", _forbidden)
    case["query"] = case["query"] * 0.5
    case["kv"] = case["kv"] * 0.75
    changed = dict(_inputs(case), bmm1_scale=0.25, bmm2_scale=1.125)
    _assert_result(case, changed, wrapper.run(**changed))
    # Entering another tuning context also keeps this plan's existing choice.
    with flashinfer.autotune(True):
        restored = _inputs(case)
        _assert_result(case, restored, wrapper.run(**restored))
    assert wrapper._planned_backend._selection is selection
    assert len(tuner.profiling_cache) == 1


def test_replan_is_transactional_and_changes_workload_identity(case, tuner):
    wrapper = _wrapper(case)
    inputs = _inputs(case)
    with flashinfer.autotune(True):
        wrapper.run(**inputs)
    previous = wrapper._planned_backend
    with pytest.raises(ValueError):
        _plan(wrapper, case, num_heads=0)
    assert wrapper._planned_backend is previous
    _assert_result(case, inputs, wrapper.run(**inputs))

    case["lengths"] = [29, 63]
    _plan(wrapper, case)
    assert wrapper._planned_backend is not previous
    with pytest.raises(RuntimeError, match="No cached MLA autotune result"):
        wrapper.run(**inputs)
    with flashinfer.autotune(True):
        _assert_result(case, inputs, wrapper.run(**inputs))
    assert len(tuner.profiling_cache) == 2


def test_cold_capture_rejected_even_when_cache_has_decision(case, tuner, monkeypatch):
    first = _wrapper(case, graph=True)
    inputs = _inputs(case)
    with flashinfer.autotune(True):
        first.run(**inputs)
    second = _wrapper(case, graph=True)
    # Exercise the guard without poisoning an actual CUDA graph capture when
    # an expected Python exception aborts it.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    with pytest.raises(
        RuntimeError, match="prepared selection before CUDA graph capture"
    ):
        second.run(**inputs)


@pytest.mark.parametrize("winner", ["trtllm-gen", "cute-dsl-monolithic"])
def test_prepared_graph_preserves_snapshot_inputs_and_options(
    case, tuner, monkeypatch, winner
):
    wrapper = _wrapper(case, graph=True, lse=True)
    expected_case = dict(case, tables=case["tables"].clone())
    case["tables"].copy_(case["tables"].flip(-1))
    assert not torch.allclose(
        _reference(case)[0], _reference(expected_case)[0], rtol=2e-2, atol=2e-2
    )
    _prefer_backend(tuner, monkeypatch, winner)
    inputs = _inputs(case, lse=True)
    with flashinfer.autotune(True):
        _assert_result(expected_case, inputs, wrapper.run(**inputs))
    assert wrapper._planned_backend._selection[0]._backend == winner
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(**inputs)
    case["query"].mul_(0.5)
    inputs["out"].fill_(float("nan"))
    graph.replay()
    _assert_result(expected_case, inputs, captured)
    with pytest.raises(RuntimeError, match="bound to a CUDA graph"):
        wrapper.run(**dict(inputs, bmm1_scale=0.25))


@pytest.mark.parametrize(
    "winner", ["trtllm-gen", "cute-dsl-monolithic", "cute-dsl-modular"]
)
def test_winner_reuses_memory_and_disk_cache_with_fresh_inputs(
    case, tuner, monkeypatch, tmp_path, winner
):
    lse = winner != "cute-dsl-modular"
    first = _wrapper(case, lse=lse)
    measured = _prefer_backend(tuner, monkeypatch, winner)
    cache = str(tmp_path / "mla-autotune.json")
    first_inputs = _inputs(case, lse=lse)
    with flashinfer.autotune(True, cache=cache):
        _assert_result(case, first_inputs, first.run(**first_inputs))
    assert measured == {
        candidate._backend for candidate in first._planned_backend._candidates
    }
    assert first._planned_backend._selection[0]._backend == winner

    for source in ("memory", "disk"):
        if source == "disk":
            tuner = AutoTuner(warmup=1, repeat=2)
            monkeypatch.setattr(AutoTuner, "_instance", tuner)
            assert not tuner.profiling_cache
        monkeypatch.setattr(tuner, "choose_one", _forbidden)
        second = _wrapper(case, lse=lse)
        changed = dict(case, query=case["query"] * 0.5, kv=case["kv"] * 0.75)
        inputs = _inputs(changed, lse=lse)
        with flashinfer.autotune(False, cache=cache if source == "disk" else None):
            _assert_result(changed, inputs, second.run(**inputs))
        selected = second._planned_backend._selection[0]
        assert selected._backend == winner
        assert selected is not first._planned_backend._selection[0]
        # Neither wrapper may retain the other's inputs or tuning resources.
        with monkeypatch.context() as warm:
            warm.setattr(tuner, "search_cache", _forbidden)
            changed["query"].mul_(0.5)
            _assert_result(changed, inputs, second.run(**inputs))
            _assert_result(case, first_inputs, first.run(**first_inputs))


def test_skipped_profiling_does_not_establish_selection(case, tuner):
    wrapper = _wrapper(case)
    inputs = _inputs(case)
    with (
        flashinfer.autotune(True, skip_ops={"batch_mla_paged_attention"}),
        pytest.raises(RuntimeError, match="produced no measured selection"),
    ):
        wrapper.run(**inputs)
    assert wrapper._planned_backend._selection is None
    assert not tuner.profiling_cache
    assert torch.isnan(inputs["out"]).all()
    with pytest.raises(RuntimeError, match="No cached MLA autotune result"):
        wrapper.run(**inputs)


def test_only_typed_unsupported_candidates_are_skipped(case, tuner, monkeypatch):
    class UnavailableBackend:
        _plan_capabilities = (
            backend._BatchMLAPagedAttentionTrtllmGenBackend._plan_capabilities
        )

        @classmethod
        def plan_from_wrapper(cls, args):
            raise policy._BackendPlanUnsupportedError("unsupported test candidate")

    monkeypatch.setattr(
        policy._BatchMLAPagedAttentionAutotuneBackend,
        "_candidate_types",
        (UnavailableBackend, backend._BatchMLAPagedAttentionTrtllmGenBackend),
    )
    wrapper = _wrapper(case)
    assert len(wrapper._planned_backend._candidates) == 1
    inputs = _inputs(case)
    with flashinfer.autotune(True):
        _assert_result(case, inputs, wrapper.run(**inputs))
    previous = wrapper._planned_backend

    def broken_plan(args):
        raise RuntimeError("unexpected compilation failure")

    monkeypatch.setattr(UnavailableBackend, "plan_from_wrapper", broken_plan)
    with pytest.raises(RuntimeError, match="unexpected compilation failure"):
        _plan(wrapper, case)
    assert wrapper._planned_backend is previous
    _assert_result(case, inputs, wrapper.run(**inputs))


def test_invalid_warm_layout_preserves_selection(case, tuner, monkeypatch):
    _prefer_backend(tuner, monkeypatch, "cute-dsl-monolithic")
    wrapper = _wrapper(case)
    inputs = _inputs(case)
    with flashinfer.autotune(True):
        wrapper.run(**inputs)
    selection = wrapper._planned_backend._selection
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    monkeypatch.setattr(policy, "_tensor_signature", _forbidden)
    out = torch.empty(
        (*inputs["out"].shape[:-1], 1024), device="cuda", dtype=inputs["out"].dtype
    )[..., ::2]
    # Planned CuTe tactics reject unsupported layouts before the launch.
    with pytest.raises(ValueError, match="Unsupported cute-dsl-monolithic tactic"):
        wrapper.run(**dict(inputs, out=out, bmm1_scale=0.25))
    assert wrapper._planned_backend._selection is selection
    changed = dict(inputs, bmm1_scale=0.0625, bmm2_scale=0.5)
    _assert_result(case, changed, wrapper.run(**changed))


@pytest.mark.parametrize("lse", [False, True])
def test_profiling_uses_current_workload_private_outputs_and_no_capture(
    case, tuner, monkeypatch, lse
):
    wrapper = _wrapper(case, lse=lse)
    inputs = _inputs(case, lse=lse)
    choose, generate = tuner.choose_one, tuner._generate_optimization_profiles
    profiles = []

    def observe_profiles(config, tensors):
        result = generate(config, tensors)
        profiles.extend(result)
        return result

    def observe(op, runners, config, tensors, *args, **kwargs):
        assert config.use_cuda_graph is False
        assert tensors[0] is inputs["query"] and tensors[1] is inputs["kv_cache"]
        for index, name in ((2, "out"), (3, "lse")):
            if inputs[name] is not None:
                assert tensors[index].shape == inputs[name].shape
                assert tensors[index].data_ptr() != inputs[name].data_ptr()
        result = choose(op, runners, config, tensors, *args, **kwargs)
        assert torch.isnan(inputs["out"]).all()
        if lse:
            assert torch.isnan(inputs["lse"]).all()
        return result

    monkeypatch.setattr(tuner, "_generate_optimization_profiles", observe_profiles)
    monkeypatch.setattr(tuner, "choose_one", observe)
    # Even the profiler must stay eager, including when modular CuTe is eligible.
    monkeypatch.setattr(torch.cuda, "graph", _forbidden)
    with flashinfer.autotune(True, tuning_buckets=(1, 4, 8)):
        _assert_result(case, inputs, wrapper.run(**inputs))
    assert len(profiles) == 1
    assert profiles[0].get_opt_shapes()[0] == tuple(inputs["query"].shape)
    # Shared scratch must leave every candidate executable, including LSE paths.
    tensors = [inputs[name] for name in ("query", "kv_cache", "out", "lse")]
    for candidate in wrapper._planned_backend._candidates:
        _assert_result(case, inputs, candidate(tensors + [None], tactic=-1))


@pytest.mark.parametrize("feature", ["fp16", "ragged-lse"])
def test_cute_only_eligible_workloads(case, tuner, feature):
    if feature == "fp16":
        case["query"] = case["query"].half()
        case["kv"] = case["kv"].half()
    else:
        case["offsets"] = [0, 1, 3]
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda"),
        backend="autotune",
    )
    _plan(
        wrapper,
        case,
        lse=True,
        q_data_type=case["query"].dtype,
        kv_data_type=case["kv"].dtype,
        output_dtype=torch.bfloat16,
    )
    candidates = wrapper._planned_backend._candidates
    assert len(candidates) == 1
    assert isinstance(candidates[0], _BatchMLAPagedAttentionCuteDslMonolithicBackend)
    inputs = _inputs(case, lse=True)
    with flashinfer.autotune(True):
        _assert_result(case, inputs, wrapper.run(**inputs))


@pytest.mark.parametrize("failure_stage", ["availability", "kernel-import"])
def test_unavailable_cute_dependency_leaves_trt_candidate(
    case, monkeypatch, failure_stage
):
    import builtins

    from flashinfer.cute_dsl import availability

    availability.is_cute_dsl_arch_supported.cache_clear()
    try:
        if failure_stage == "availability":
            find_spec = availability.importlib.util.find_spec

            def without_cutlass(name, *args, **kwargs):
                return None if name == "cutlass" else find_spec(name, *args, **kwargs)

            monkeypatch.setattr(
                availability.importlib.util, "find_spec", without_cutlass
            )
        else:
            import_module = builtins.__import__

            def without_kernel_dependency(name, *args, **kwargs):
                if name in (
                    "flashinfer.cute_dsl.attention.monolithic",
                    "flashinfer.cute_dsl.attention.wrappers",
                ):
                    raise ModuleNotFoundError("No module named 'cutlass.cute'")
                return import_module(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", without_kernel_dependency)
        wrapper = _wrapper(case)
        assert [
            type(candidate) for candidate in wrapper._planned_backend._candidates
        ] == [backend._BatchMLAPagedAttentionTrtllmGenBackend]
    finally:
        availability.is_cute_dsl_arch_supported.cache_clear()


def test_zero_workspace_keeps_monolithic_one_split_candidate(case):
    from flashinfer.cute_dsl.attention.monolithic import mla_decode

    # Two K tiles force nonzero split-reduction storage for this small query
    # grid. TRT planning uses its separately allocated counter storage.
    case["tables"] = case["tables"].repeat(1, 2)
    _, required = mla_decode._get_split_kv_and_workspace_size(
        2, 1, 128, 512, mla_decode.get_num_sm(case["query"].device), max_seq_len=256
    )
    assert required > 0
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(0, dtype=torch.uint8, device="cuda"), backend="autotune"
    )
    _plan(wrapper, case)
    assert [type(candidate) for candidate in wrapper._planned_backend._candidates] == [
        backend._BatchMLAPagedAttentionTrtllmGenBackend,
        _BatchMLAPagedAttentionCuteDslMonolithicBackend,
    ]
    candidate = wrapper._planned_backend._candidates[1]
    inputs = _inputs(case)
    assert candidate.get_valid_tactics(
        [inputs["query"], inputs["kv_cache"], inputs["out"], None, None], None
    ) == [-1, 1]


@pytest.mark.parametrize("implementation", ["monolithic", "modular"])
def test_cute_compiler_failure_is_not_a_candidate_rejection(
    case, monkeypatch, implementation
):
    if implementation == "monolithic":
        from flashinfer.cute_dsl.attention.monolithic import mla_decode as module

        name = "_get_compiled_mla_kernel"
    else:
        from flashinfer.cute_dsl.attention.wrappers import batch_mla as module

        name = "_compile_mla_kernel"
    failure = ValueError("injected compiler failure")

    def broken_compile(**kwargs):
        raise failure

    monkeypatch.setattr(module, name, broken_compile)
    with pytest.raises(ValueError) as caught:
        _wrapper(case)
    assert caught.value is failure


@pytest.mark.parametrize("graph_first", [False, True])
def test_graph_and_eager_plans_have_separate_cache_entries(
    case, tuner, monkeypatch, graph_first, trt_only
):
    # Use identical candidates to isolate the graph flag from candidate identity.
    first = _wrapper(case, graph=graph_first)
    seen = []
    choose = tuner.choose_one

    def observe(op, runners, config, tensors, *args, **kwargs):
        seen.append(config.use_cuda_graph)
        return choose(op, runners, config, tensors, *args, **kwargs)

    monkeypatch.setattr(tuner, "choose_one", observe)
    first_inputs = _inputs(case)
    with flashinfer.autotune(True):
        _assert_result(case, first_inputs, first.run(**first_inputs))
    second = _wrapper(case, graph=not graph_first)
    inputs = _inputs(case)
    with pytest.raises(RuntimeError, match="No cached MLA autotune result"):
        second.run(**inputs)
    with flashinfer.autotune(True):
        _assert_result(case, inputs, second.run(**inputs))
    assert seen == [graph_first, not graph_first]
    assert len(tuner.profiling_cache) == 2


def test_graph_plan_excludes_modular_with_eager_measurement(case, tuner, monkeypatch):
    wrapper = _wrapper(case, graph=True)
    candidates = wrapper._planned_backend._candidates
    assert [type(candidate) for candidate in candidates] == [
        backend._BatchMLAPagedAttentionTrtllmGenBackend,
        _BatchMLAPagedAttentionCuteDslMonolithicBackend,
    ]
    monkeypatch.setattr(torch.cuda, "graph", _forbidden)
    inputs = _inputs(case)
    with autotune_v2(
        persistent_cache=False,
        measurement_policy=MeasurementPolicy(execution_mode="eager"),
    ):
        _assert_result(case, inputs, wrapper.run(**inputs))


def test_eager_plan_rejects_graph_measurement_override_before_launch(
    case, tuner, monkeypatch
):
    wrapper = _wrapper(case)
    inputs = _inputs(case)
    for candidate in wrapper._planned_backend._candidates:
        monkeypatch.setattr(candidate, "forward", _forbidden)
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    monkeypatch.setattr(torch.cuda, "graph", _forbidden)
    with (
        autotune_v2(
            persistent_cache=False,
            measurement_policy=MeasurementPolicy(execution_mode="cuda_graph"),
        ),
        pytest.raises(ValueError, match="use_cuda_graph=False"),
    ):
        wrapper.run(**inputs)
    assert wrapper._planned_backend._selection is None
    assert torch.isnan(inputs["out"]).all()


@pytest.mark.parametrize("prepared", [False, True])
def test_eager_plan_rejects_manual_capture_even_with_modular_winner(
    case, tuner, monkeypatch, prepared
):
    wrapper = _wrapper(case)
    inputs = _inputs(case)
    if prepared:
        index = next(
            i
            for i, candidate in enumerate(wrapper._planned_backend._candidates)
            if candidate._backend == "cute-dsl-modular"
        )
        monkeypatch.setattr(
            tuner, "search_cache", lambda *args, **kwargs: (True, index, -1, None)
        )
        _assert_result(case, inputs, wrapper.run(**inputs))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    for candidate in wrapper._planned_backend._candidates:
        monkeypatch.setattr(candidate, "forward", _forbidden)
    with pytest.raises(RuntimeError, match="use_cuda_graph=False"):
        wrapper.run(**inputs)


@pytest.mark.parametrize("feature", ["ragged", "causal-multiquery", "lse"])
def test_modular_filter_preserves_other_candidates(case, feature):
    if feature == "ragged":
        case["offsets"] = [0, 1, 3]
    elif feature == "causal-multiquery":
        case["offsets"] = [0, 2, 4]
    wrapper = _wrapper(case, lse=feature == "lse")
    assert [type(candidate) for candidate in wrapper._planned_backend._candidates] == [
        backend._BatchMLAPagedAttentionTrtllmGenBackend,
        _BatchMLAPagedAttentionCuteDslMonolithicBackend,
    ]


def test_tensor_scales_rebind_after_tuning_and_during_graph_replay(
    case, tuner, monkeypatch, trt_only
):
    case = dict(
        case,
        query=case["query"].to(torch.float8_e4m3fn),
        kv=case["kv"].to(torch.float8_e4m3fn),
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda"),
        backend="autotune",
        use_cuda_graph=True,
    )
    _plan(
        wrapper,
        case,
        lse=True,
        scale_mode="bmm-tensor",
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        output_dtype=torch.bfloat16,
    )

    def run_and_check(inputs):
        result = wrapper.run(**inputs)
        expected, expected_lse = _reference(
            case, inputs["bmm1_scale"], inputs["bmm2_scale"]
        )
        assert result[0] is inputs["out"] and result[1] is inputs["lse"]
        torch.testing.assert_close(result[0].float(), expected, rtol=0.03, atol=0.03)
        torch.testing.assert_close(result[1], expected_lse, rtol=0.01, atol=0.01)

    inputs = _inputs(case, lse=True)
    inputs.update(
        bmm1_scale=torch.tensor([0.125], device="cuda"),
        bmm2_scale=torch.tensor([0.75], device="cuda"),
    )
    with flashinfer.autotune(True):
        run_and_check(inputs)
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    # Equal signatures must bind fresh tensors, including scale objects.
    case.update(
        query=(case["query"].float() * 0.5).to(torch.float8_e4m3fn),
        kv=(case["kv"].float() * 0.75).to(torch.float8_e4m3fn),
    )
    inputs = _inputs(case, lse=True)
    inputs.update(
        bmm1_scale=torch.tensor([0.0625], device="cuda"),
        bmm2_scale=torch.tensor([1.125], device="cuda"),
    )
    run_and_check(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = wrapper.run(**inputs)
    inputs["bmm1_scale"].fill_(0.25)
    inputs["bmm2_scale"].fill_(0.5)
    inputs["out"].fill_(float("nan"))
    graph.replay()
    expected, expected_lse = _reference(case, 0.25, 0.5)
    torch.testing.assert_close(result[0].float(), expected, rtol=0.03, atol=0.03)
    torch.testing.assert_close(result[1], expected_lse, rtol=0.01, atol=0.01)


def test_split_input_signature_distinguishes_storage_layout():
    packed = torch.randn(2, 4, 576)
    adjacent = (packed[..., :512], packed[..., 512:])
    independent = tuple(torch.empty_strided(t.shape, t.stride()) for t in adjacent)
    assert policy._tensor_signature(adjacent) != policy._tensor_signature(independent)
    fresh = packed.clone()
    assert policy._tensor_signature(adjacent) == policy._tensor_signature(
        (fresh[..., :512], fresh[..., 512:])
    )


@pytest.mark.parametrize(
    "experimental,graph", [(False, False), (False, True), (True, False)]
)
def test_supported_candidates_profile_same_operation(
    tuner, monkeypatch, experimental, graph
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (9, 0),
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("requires SM90/SM100/SM103/SM107")
    hopper = torch.cuda.get_device_capability() == (9, 0)
    if hopper and experimental:
        pytest.skip("cuTile is unsupported on SM90")
    if experimental:
        pytest.importorskip("cuda.tile.compilation")
        from flashinfer.cutile.cutile_common import is_cuda_tile_available

        if not is_cuda_tile_available():
            pytest.skip("cuTile compiler toolchain unavailable")
    monkeypatch.setenv(
        "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", str(int(experimental))
    )
    torch.manual_seed(54)
    query = torch.randn(2, 128, 576, device="cuda", dtype=torch.bfloat16) * 0.5
    kv = torch.randn(8, 32, 576, device="cuda", dtype=torch.bfloat16) * 0.5
    tables = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], device="cuda", dtype=torch.int32
    )
    lens = torch.tensor([53, 97], device="cuda", dtype=torch.int32)
    scale = 1 / math.sqrt(192)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8),
        backend="autotune",
        use_cuda_graph=graph,
    )
    wrapper.plan(
        metadata=MLAPlanMetadata.dense(
            torch.arange(3, device="cuda", dtype=torch.int32), tables, lens
        ),
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=32,
        causal=False,
        sm_scale=scale,
        q_data_type=query.dtype,
        kv_data_type=kv.dtype,
    )
    expected_names = {
        "trtllm-gen",
        "cute-dsl-monolithic",
        "cute-dsl-modular",
        "fa2",
        "cutlass",
    }
    if graph:
        expected_names.remove("cute-dsl-modular")
    if hopper:
        expected_names = {"fa2", "fa3"}
    if experimental:
        expected_names.add("cutile")
    assert {c._backend for c in wrapper._planned_backend._candidates} == expected_names
    if hopper:
        candidates = wrapper._planned_backend._candidates
        for name in ("_int_workspace_buffer", "_pin_memory_int_workspace_buffer"):
            assert len({getattr(c, name).data_ptr() for c in candidates}) == 2
    # Every candidate must own a stable metadata snapshot, independent of the
    # caller and of other candidates' native planning/scratch state.
    original_tables = tables.clone()
    tables.copy_(tables.flip(0))
    tables = original_tables
    out = torch.full((2, 128, 512), float("nan"), device="cuda", dtype=query.dtype)

    def reference():
        result = []
        for i, length in enumerate((53, 97)):
            cache = kv[tables[i].long()].reshape(-1, 576)[:length].float()
            result.append(
                (query[i].float() @ cache.T * scale).softmax(-1) @ cache[:, :512]
            )
        return torch.stack(result)

    # Check every actual candidate before measuring, including mixed scratch
    # users. Profiling alone would not expose a numerically wrong losing runner.
    signature = ("all-candidates",)
    options = dict(
        return_lse=False,
        return_lse_base_on_e=False,
        o_scale=None,
        ckv_scale=None,
        kpe_scale=None,
        skip_softmax_threshold_scale_factor=None,
        bmm1_scale=None,
        bmm2_scale=None,
    )
    for candidate in reversed(wrapper._planned_backend._candidates):
        candidate.configure_tuning(cache_key=signature, run_options=options)
        candidate([query, kv, out, None, None], tactic=-1)
        torch.testing.assert_close(out.float(), reference(), atol=1e-2, rtol=2e-2)
    with flashinfer.autotune(True):
        wrapper.run(query=query, kv_cache=kv, out=out)
    torch.testing.assert_close(out.float(), reference(), atol=1e-2, rtol=2e-2)
    query = query * 0.75
    kv = kv * 0.5
    wrapper.run(query=query, kv_cache=kv, out=out)
    torch.testing.assert_close(out.float(), reference(), atol=1e-2, rtol=2e-2)
    if hopper:
        selected = wrapper._planned_backend._selection[0]
        with monkeypatch.context() as patch:
            patch.setattr(policy, "_tensor_signature", _forbidden)
            patch.setattr(tuner, "choose_one", _forbidden)
            patch.setattr(tuner, "search_cache", _forbidden)
            # A different supported outer stride keeps the existing selection.
            strided = torch.empty((4, 128, 576), device="cuda", dtype=query.dtype)[::2]
            strided.copy_(query)
            wrapper.run(query=strided, kv_cache=kv, out=out)
            torch.testing.assert_close(out.float(), reference(), atol=1e-2, rtol=2e-2)
            patch.setattr(selected, "run_from_wrapper", _forbidden)
            misaligned = torch.empty((2, 128, 513), device="cuda", dtype=out.dtype)[
                ..., 1:
            ]
            with pytest.raises(ValueError, match="tensor layout is unsupported"):
                wrapper.run(query=query, kv_cache=kv, out=misaligned)
            with pytest.raises(ValueError, match="shape"):
                wrapper.run(query=query[:1], kv_cache=kv, out=out)
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            wrapper.run(query=query, kv_cache=kv, out=out)
        query.mul_(0.5)
        out.fill_(float("nan"))
        captured.replay()
        torch.testing.assert_close(out.float(), reference(), atol=1e-2, rtol=2e-2)


@pytest.mark.parametrize("case", ["outer-strides", "small-workspace"])
def test_candidate_rejection_preserves_supported_execution(tuner, monkeypatch, case):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
        (12, 0),
        (12, 1),
    ):
        pytest.skip("requires a Blackwell MLA backend")
    if case == "small-workspace" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires TRTLLM-GEN workspace fallback")
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    torch.manual_seed(68)
    query = torch.randn(4, 128, 576, device="cuda", dtype=torch.bfloat16)[::2]
    kv = torch.randn(16, 32, 576, device="cuda", dtype=torch.bfloat16)[::2]
    out = torch.empty(4, 128, 512, device="cuda", dtype=query.dtype)[::2]
    if case == "small-workspace":
        query, kv, out = query.contiguous(), kv.contiguous(), out.contiguous()
    tables = torch.arange(8, device="cuda", dtype=torch.int32).reshape(2, 4)
    lens = torch.tensor([53, 97], device="cuda", dtype=torch.int32)
    scale = 1 / math.sqrt(192)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(
            1024**2 if case == "small-workspace" else 128 * 1024**2,
            device="cuda",
            dtype=torch.uint8,
        ),
        backend="autotune",
    )
    wrapper.plan(
        metadata=MLAPlanMetadata.dense(
            torch.arange(3, device="cuda", dtype=torch.int32), tables, lens
        ),
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=32,
        causal=False,
        sm_scale=scale,
        q_data_type=query.dtype,
        kv_data_type=kv.dtype,
    )
    names = {c._backend for c in wrapper._planned_backend._candidates}
    if case == "small-workspace":
        assert "fa2" not in names and "trtllm-gen" in names
    else:
        assert "fa2" in names and len(names) > 1
    with flashinfer.autotune(True):
        wrapper.run(query=query, kv_cache=kv, out=out)
    if case == "outer-strides":
        assert wrapper._planned_backend._selection[0]._backend == "fa2"
    for i, length in enumerate((53, 97)):
        cache = kv[tables[i].long()].reshape(-1, 576)[:length].float()
        expected = (query[i].float() @ cache.T * scale).softmax(-1) @ cache[:, :512]
        torch.testing.assert_close(out[i].float(), expected, atol=2e-2, rtol=2e-2)


@pytest.fixture
def mtp_adapter(monkeypatch):
    """Exercise adapter contracts with a deterministic prepared launch."""
    from types import SimpleNamespace

    from flashinfer.mla._batch_mla._backends.cute_dsl_rubin_mtp_backend import (
        _BatchMLAPagedAttentionCuteDslRubinMtpBackend as Backend,
    )

    compiled, launches = [], []

    def workspace_size(B, Q, H, D, _sms, max_seq_len, num_kv_splits=None):
        splits = 4 if num_kv_splits is None else num_kv_splits
        return splits, 0 if splits == 1 else B * Q * H * splits * (D + 1) * 4

    def prepare(**options):
        compiled.append(options)

        def launch(*args):
            launches.append(args)
            args[5].zero_()
            args[6].fill_(args[-1])

        return launch

    implementation = SimpleNamespace(
        _check_can_implement=lambda **kwargs: None,
        _check_tensor_indexing=lambda tensor, name: None,
        _check_kv_tensor_indexing=lambda tensor, name: None,
        _get_split_kv_and_workspace_size=workspace_size,
        prepare_cute_dsl_mla_decode=prepare,
        get_num_sm=lambda device: 16,
        _as_cute_dsl_workspace_i8=lambda workspace: workspace.view(torch.int8),
        Float32=float,
        Int32=int,
    )
    monkeypatch.setattr(
        Backend, "_implementation", staticmethod(lambda: implementation)
    )
    runner = Backend(torch.empty(16 * 1024 * 1024, dtype=torch.uint8))
    tables = torch.arange(16, dtype=torch.int32).reshape(2, 8)
    lengths = torch.tensor([256, 256], dtype=torch.int32)
    options = dict(
        cum_seq_lens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        block_tables=tables,
        seq_lens=lengths,
        max_q_len=2,
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=64,
        sm_scale=0.125,
        q_data_type=torch.float8_e4m3fn,
        output_dtype=torch.float8_e4m3fn,
        use_cuda_graph=False,
        use_sinks=False,
        causal=True,
    )
    runner._plan(**options)
    runner._lse_scale = math.log(2)
    runner.configure_tuning(
        cache_key=("test",),
        run_options=dict(
            return_lse=True,
            return_lse_base_on_e=True,
            o_scale=None,
            ckv_scale=None,
            kpe_scale=None,
            skip_softmax_threshold_scale_factor=None,
            bmm1_scale=0.125,
            bmm2_scale=0.75,
        ),
    )
    inputs = [
        torch.empty((4, 128, 576), dtype=torch.float8_e4m3fn),
        torch.empty((16, 64, 576), dtype=torch.float8_e4m3fn),
        torch.empty((4, 128, 512), dtype=torch.float8_e4m3fn),
        torch.empty((4, 128), dtype=torch.float32),
        None,
    ]
    return SimpleNamespace(
        runner=runner,
        inputs=inputs,
        compiled=compiled,
        launches=launches,
        tables=tables,
        lengths=lengths,
        options=options,
    )


def test_mtp_split_tactics_are_capacity_bound_and_preserve_live_metadata(mtp_adapter):
    case = mtp_adapter
    runner = case.runner
    assert runner._compile_options["resolved_is_var_seq"] is True
    assert runner.get_valid_tactics(case.inputs, None) == [-1, 1, 2, 4]
    assert not runner.validate_tactic(case.inputs, 8)
    assert not runner.validate_tactic(case.inputs, True)
    runner.precompile_tactics(case.inputs, [-1, 1, 2, 4], None)
    assert len(case.compiled) == 4
    for tactic in (-1, 1, 2, 4):
        out, lse = runner(case.inputs, tactic=tactic)
        assert out is case.inputs[2] and lse is case.inputs[3]
        torch.testing.assert_close(lse, torch.full_like(lse, math.log(2)))
        args = case.launches[-1]
        assert args[4] is case.tables and args[9] is case.lengths
        assert args[-3:] == (0.125, 0.75, math.log(2))
    key = runner.get_cache_key_extras(case.inputs)
    case.lengths.copy_(torch.tensor([1, 511], dtype=torch.int32))
    case.tables.copy_(case.tables.flip(1))
    runner(case.inputs, tactic=4)
    assert len(case.compiled) == 4
    assert runner.get_cache_key_extras(case.inputs) == key
    assert runner.validate_tactic(case.inputs, 4)
    with pytest.raises(ValueError, match="Unsupported.*tactic"):
        runner(case.inputs, tactic=8)


def test_mtp_unprepared_tactic_rejects_capture_and_warm_tactic_reuses(
    mtp_adapter, monkeypatch
):
    case = mtp_adapter
    case.runner.precompile_tactics(case.inputs, [2], None)
    case.runner.device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    case.runner(case.inputs, tactic=2)
    assert len(case.compiled) == 2
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        case.runner(case.inputs, tactic=4)
    assert len(case.compiled) == 2


def test_mtp_default_falls_back_to_one_split_for_zero_workspace(mtp_adapter):
    case = mtp_adapter
    case.runner._float_workspace_buffer = torch.empty(0, dtype=torch.uint8)
    case.runner._plan(**case.options)
    assert case.runner.get_valid_tactics(case.inputs, None) == [-1, 1]
    assert case.runner._execution_state.split_kv == 1
    assert case.runner._execution_state.workspace_bytes is None


@pytest.mark.parametrize("change", ["ragged", "noncausal", "q1"])
def test_mtp_query_contract_is_independent_of_live_kv_support(mtp_adapter, change):
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    options = dict(mtp_adapter.options)
    if change == "ragged":
        options["cum_seq_lens_q"] = torch.tensor([0, 1, 4], dtype=torch.int32)
        options["max_q_len"] = 3
    elif change == "noncausal":
        options["causal"] = False
    else:
        options["cum_seq_lens_q"] = torch.tensor([0, 1, 2], dtype=torch.int32)
        options["max_q_len"] = 1
    with pytest.raises(_BackendPlanUnsupportedError):
        mtp_adapter.runner._plan(**options)


@pytest.mark.parametrize(
    "name", ["cute-dsl-rubin-mtp", "cute-dsl-monolithic", "autotune"]
)
def test_rubin_mtp_planned_fp8_output_and_snapshot(case, tuner, monkeypatch, name):
    from flashinfer.mla._batch_mla._backends.cute_dsl_rubin_mtp_backend import (
        _BatchMLAPagedAttentionCuteDslRubinMtpBackend as Mtp,
    )

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    case.update(
        query=(torch.randn(4, 128, 576, device="cuda") * 0.3).to(torch.float8_e4m3fn),
        kv=(torch.randn(16, 64, 576, device="cuda") * 0.3).to(torch.float8_e4m3fn),
        tables=torch.arange(16, device="cuda", dtype=torch.int32).reshape(2, 8),
        lengths=[257, 511],
        offsets=[0, 2, 4],
    )
    if name == "autotune":
        monkeypatch.setattr(
            policy._BatchMLAPagedAttentionAutotuneBackend,
            "_candidate_types",
            (_BatchMLAPagedAttentionCuteDslMonolithicBackend, Mtp),
        )
        _prefer_backend(tuner, monkeypatch, "cute-dsl-rubin-mtp")
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda"), backend=name
    )
    _plan(
        wrapper,
        case,
        lse=True,
        page_size=64,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        output_dtype=torch.float8_e4m3fn,
    )
    inputs = _inputs(case, lse=True)
    inputs["out"] = torch.empty((4, 128, 512), device="cuda", dtype=torch.float8_e4m3fn)
    expected, expected_lse = _reference(case)
    with flashinfer.autotune(name == "autotune"):
        out, lse = wrapper.run(**inputs)
    assert out is inputs["out"] and lse is inputs["lse"]
    torch.testing.assert_close(out.float(), expected, rtol=0.12, atol=0.03)
    torch.testing.assert_close(lse, expected_lse, rtol=0.02, atol=0.02)
    if name == "autotune":
        selection = wrapper._planned_backend._selection
        assert selection[0]._backend == "cute-dsl-rubin-mtp"
        case["tables"].copy_(case["tables"].flip(0))
        monkeypatch.setattr(tuner, "choose_one", _forbidden)
        out, lse = wrapper.run(**inputs)
        assert wrapper._planned_backend._selection is selection
        torch.testing.assert_close(out.float(), expected, rtol=0.12, atol=0.03)
        torch.testing.assert_close(lse, expected_lse, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("scale", [0.0, -0.125, 1e-50, 1e40, 3e38])
def test_mtp_rejects_unsafe_runtime_softmax_scale(mtp_adapter, scale):
    case = mtp_adapter
    options = dict(case.runner._planned_run_options, bmm1_scale=scale)
    case.runner.configure_tuning(cache_key=("bad-scale",), run_options=options)
    assert case.runner.get_valid_tactics(case.inputs, None) == []
    with pytest.raises(ValueError, match="positive softmax scale"):
        case.runner._run(
            query=case.inputs[0],
            kv_cache=case.inputs[1],
            out=case.inputs[2],
            lse=case.inputs[3],
            return_lse=True,
            bmm1_scale=scale,
            bmm2_scale=0.75,
        )
    assert not case.launches


@pytest.mark.parametrize("scale", [1e40, -1e40])
def test_mtp_rejects_unsafe_runtime_output_scale(mtp_adapter, scale):
    case = mtp_adapter
    options = dict(case.runner._planned_run_options, bmm2_scale=scale)
    case.runner.configure_tuning(cache_key=("bad-output-scale",), run_options=options)
    assert case.runner.get_valid_tactics(case.inputs, None) == []
    with pytest.raises(ValueError, match="finite FP32 output scale"):
        case.runner._run(
            query=case.inputs[0],
            kv_cache=case.inputs[1],
            out=case.inputs[2],
            lse=case.inputs[3],
            return_lse=True,
            bmm1_scale=0.125,
            bmm2_scale=scale,
        )
    assert not case.launches


@pytest.mark.parametrize("scale", [1e-50, 1e40, 3e38])
def test_mtp_rejects_unsafe_plan_softmax_scale(case, scale):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    case["offsets"] = [0, 2, 4]
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin-mtp",
    )
    with pytest.raises(ValueError, match="positive softmax scale"):
        _plan(
            wrapper,
            case,
            page_size=64,
            sm_scale=scale,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            output_dtype=torch.float8_e4m3fn,
        )
    assert wrapper._planned_backend is None


def test_mtp_family_only_compiler_target_is_typed_refusal(case, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    from flashinfer.cute_dsl import availability
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    monkeypatch.setattr(availability, "cute_dsl_compile_arch", lambda *_: "sm_100f")
    case["offsets"] = [0, 2, 4]
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin-mtp",
    )
    with pytest.raises(_BackendPlanUnsupportedError, match="native SM107"):
        _plan(
            wrapper,
            case,
            page_size=64,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            output_dtype=torch.float8_e4m3fn,
        )
    assert wrapper._planned_backend is None


@pytest.mark.parametrize(
    "softmax_scale,output_scale,message",
    [
        (1e-50, 1.0, "positive softmax scale"),
        (1e40, 1.0, "positive softmax scale"),
        (3e38, 1.0, "positive softmax scale"),
        (1 / 24, 1e40, "finite FP32 output scale"),
        (1 / 24, -1e40, "finite FP32 output scale"),
    ],
)
def test_mtp_low_level_rejects_unsafe_scales_before_launch(
    softmax_scale, output_scale, message
):
    from flashinfer.cute_dsl.attention.rubin_mtp.mla_decode import cute_dsl_mla_decode

    with pytest.raises(ValueError, match=message):
        cute_dsl_mla_decode(
            query=None,
            kv_cache=None,
            workspace_buffer=None,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=None,
            seq_lens=None,
            max_seq_len=128,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
        )


@pytest.mark.parametrize("output_scale", [0.0, -0.5, 3e38])
@pytest.mark.parametrize("softmax_scale", [1e-8, 1 / 24, 2e38])
def test_mtp_scale_validation_preserves_representable_scales(
    softmax_scale, output_scale
):
    from flashinfer.cute_dsl._mla_validation import _validate_mtp_scales

    _validate_mtp_scales(softmax_scale, output_scale)


def test_mtp_compiler_failure_propagates_without_support_fallback(
    mtp_adapter, monkeypatch
):
    failure = ValueError("injected MTP compiler failure")

    def broken_compile(**kwargs):
        raise failure

    monkeypatch.setattr(
        mtp_adapter.runner._implementation(),
        "prepare_cute_dsl_mla_decode",
        broken_compile,
    )
    with pytest.raises(ValueError) as caught:
        mtp_adapter.runner._plan(**mtp_adapter.options)
    assert caught.value is failure


def test_mtp_missing_compiler_features_are_typed_refusals(mtp_adapter, monkeypatch):
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    def unavailable(**kwargs):
        raise ImportError("missing Rubin mixed-CGA compiler features")

    monkeypatch.setattr(
        mtp_adapter.runner._implementation(), "_check_can_implement", unavailable
    )
    with pytest.raises(_BackendPlanUnsupportedError, match="mixed-CGA"):
        mtp_adapter.runner._plan(**mtp_adapter.options)


def test_mtp_filters_unsafe_split_budgets_without_clamping(mtp_adapter, monkeypatch):
    case = mtp_adapter
    implementation = case.runner._implementation()
    workspace_size = implementation._get_split_kv_and_workspace_size

    def bounded_workspace(*args, num_kv_splits=None, **kwargs):
        if num_kv_splits in (None, 4):
            raise ValueError("split scratch exceeds signed 32-bit indexing")
        return workspace_size(*args, num_kv_splits=num_kv_splits, **kwargs)

    monkeypatch.setattr(
        implementation, "_get_split_kv_and_workspace_size", bounded_workspace
    )
    case.runner._plan(**case.options)
    assert case.runner._execution_state.split_kv == 1
    assert case.runner.get_valid_tactics(case.inputs, None) == [-1, 1, 2]
    assert not case.runner.validate_tactic(case.inputs, 4)
    compiled_before = len(case.compiled)
    with pytest.raises(ValueError, match="Unsupported.*tactic"):
        case.runner.precompile_tactics(case.inputs, [4], None)
    with pytest.raises(ValueError, match="Unsupported.*tactic"):
        case.runner(case.inputs, tactic=4)
    assert len(case.compiled) == compiled_before
    assert not case.launches


def test_mtp_unsafe_shape_refuses_before_compile(mtp_adapter, monkeypatch):
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    case = mtp_adapter

    def unsafe_shape(*args, **kwargs):
        raise ValueError("query span exceeds signed 32-bit indexing")

    monkeypatch.setattr(
        case.runner._implementation(), "_get_split_kv_and_workspace_size", unsafe_shape
    )
    compiled_before = len(case.compiled)
    assert case.runner.get_valid_tactics(case.inputs, None) == []
    with pytest.raises(_BackendPlanUnsupportedError, match="query span"):
        case.runner._plan(**case.options)
    assert len(case.compiled) == compiled_before


@pytest.mark.parametrize(
    "name", ["query", "kv_cache", "out", "lse", "block_tables", "seq_lens"]
)
def test_mtp_unsafe_tensor_span_excludes_tuning_candidate(
    mtp_adapter, monkeypatch, name
):
    case = mtp_adapter

    def check_indexing(tensor, tensor_name):
        if tensor_name == name:
            raise ValueError(f"{name} span exceeds signed 32-bit indexing")

    monkeypatch.setattr(
        case.runner._implementation(),
        "_check_kv_tensor_indexing" if name == "kv_cache" else "_check_tensor_indexing",
        check_indexing,
    )
    assert case.runner.get_valid_tactics(case.inputs, None) == []
    assert not case.runner.validate_tactic(case.inputs, 1)
    assert not case.launches


def test_mtp_unsafe_plan_metadata_refuses_before_compile(mtp_adapter, monkeypatch):
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    case = mtp_adapter

    def check_indexing(tensor, name):
        if name == "block_tables":
            raise ValueError("page table span exceeds signed 32-bit indexing")

    monkeypatch.setattr(
        case.runner._implementation(), "_check_tensor_indexing", check_indexing
    )
    compiled_before = len(case.compiled)
    with pytest.raises(_BackendPlanUnsupportedError, match="page table span"):
        case.runner._plan(**case.options)
    assert len(case.compiled) == compiled_before


@pytest.mark.parametrize("q_len,page_size", [(2, 64), (4, 128)])
def test_mtp_large_kv_pool_high_pages_eager_and_graph(
    case, monkeypatch, q_len, page_size
):
    """A small request can address the final pages of a shared pool above 4 GiB."""
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    pool_bytes = 9 << 29  # 4.5 GiB of FP8, without initializing unused pages.
    if torch.cuda.mem_get_info()[0] < pool_bytes + (512 << 20):
        pytest.skip("requires 5 GiB of free GPU memory")
    page_bytes = page_size * 576
    pool_pages = pool_bytes // page_bytes
    pages = 1024 // page_size
    scale = 1 / 24
    case.update(
        query=(torch.randn(q_len, 128, 576, device="cuda") * 0.5).to(
            torch.float8_e4m3fn
        ),
        kv=(torch.randn(pages, page_size, 576, device="cuda") * 0.5).to(
            torch.float8_e4m3fn
        ),
        tables=torch.randperm(pages, device="cuda", dtype=torch.int32).view(1, pages),
        lengths=[1024],
        offsets=[0, q_len],
    )
    # Compute the oracle from the small pool before remapping physical pages.
    expected, expected_lse = _reference(case, scale)
    pool = torch.empty(
        (pool_pages, page_size, 576), dtype=torch.float8_e4m3fn, device="cuda"
    )
    first_page = pool_pages - pages
    assert first_page * page_bytes > 1 << 32
    raw = pool.view(torch.uint8).reshape(-1)
    for page in range(first_page, pool_pages):
        # A truncated 31/32-bit address must read poison, not plausible low data.
        for bits in (31, 32):
            alias = (page * page_bytes) % (1 << bits)
            raw[alias : alias + page_bytes].fill_(0x7F)  # E4M3 NaN
    pool[first_page:].copy_(case["kv"])
    case["kv"] = pool
    case["tables"] = case["tables"] + first_page
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin-mtp",
        use_cuda_graph=True,
    )
    _plan(
        wrapper,
        case,
        lse=True,
        page_size=page_size,
        sm_scale=scale,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        output_dtype=torch.float8_e4m3fn,
    )
    inputs = _inputs(case, lse=True, scale=scale)
    inputs["out"] = torch.empty(
        (q_len, 128, 512), dtype=torch.float8_e4m3fn, device="cuda"
    )

    def check(result):
        out, lse = result
        assert out is inputs["out"] and lse is inputs["lse"]
        assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all()
        torch.testing.assert_close(out.float(), expected, rtol=0.12, atol=0.03)
        torch.testing.assert_close(lse, expected_lse, rtol=0.002, atol=0.02)

    # Exercise the public runtime before checking private tuning eligibility.
    check(wrapper.run(**inputs))
    runner = wrapper._planned_backend
    tuning_inputs = [inputs[k] for k in ("query", "kv_cache", "out", "lse")] + [None]
    runner.configure_tuning(
        cache_key=("high-kv-pages",),
        run_options=dict(
            return_lse=True,
            return_lse_base_on_e=False,
            o_scale=None,
            ckv_scale=None,
            kpe_scale=None,
            skip_softmax_threshold_scale_factor=None,
            bmm1_scale=scale,
            bmm2_scale=inputs["bmm2_scale"],
        ),
    )
    assert -1 in runner.get_valid_tactics(tuning_inputs, None)
    assert runner.validate_tactic(tuning_inputs, -1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(**inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(**inputs)
    inputs["out"].view(torch.uint8).fill_(0x7F)
    inputs["lse"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(captured)


@pytest.mark.parametrize(
    "branch,q_len,page_size,scale,capacity",
    [
        (None, q_len, page_size, scale, 4096)
        for q_len in (2, 4)
        for page_size in (64, 128)
        for scale in (1 / 24, 1e-8, 0.125)
    ]
    + [
        (branch, q_len, page_size, 1e-8, 4096)
        for branch in ("preferred", "fallback")
        for q_len, page_size in ((2, 64), (4, 128))
    ]
    + [(None, q_len, 64, 1 / 24, 64) for q_len in (2, 4)],
)
def test_planned_fp8_split_tactics_and_graph_replay(
    case, monkeypatch, branch, q_len, page_size, scale, capacity
):
    """Every admitted MTP tactic must handle live tails and fully masked rows.

    Forced branches retain the empty-partition progress regression even if the
    hardware scheduler otherwise always chooses the preferred cluster size.
    """
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    from flashinfer.cute_dsl.attention.rubin_mtp import kernel, mla_decode

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    if branch is not None:
        cls = kernel.RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo
        original = cls.__init__

        def initialize(self, *args, **kwargs):
            kwargs["force_branch"] = branch
            original(self, *args, **kwargs)

        mla_decode._get_compiled_mla_kernel.cache_clear()
        monkeypatch.setattr(cls, "__init__", initialize)

    try:
        pages = capacity // page_size
        case.update(
            query=(torch.randn(2 * q_len, 128, 576, device="cuda") * 0.3).to(
                torch.float8_e4m3fn
            ),
            kv=(torch.randn(2 * pages, page_size, 576, device="cuda") * 0.3).to(
                torch.float8_e4m3fn
            ),
            tables=torch.randperm(2 * pages, device="cuda", dtype=torch.int32).view(
                2, pages
            ),
            lengths=[min(2048, capacity)] * 2,
            offsets=[0, q_len, 2 * q_len],
        )
        lengths = torch.tensor(case["lengths"], dtype=torch.int32, device="cuda")
        metadata = MLAPlanMetadata.dense(
            torch.tensor(case["offsets"], dtype=torch.int32, device="cuda"),
            case["tables"],
            lengths,
            max_q_len=q_len,
        )
        wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
            backend="cute-dsl-rubin-mtp",
            use_cuda_graph=capacity > 64,
        )
        _plan(
            wrapper,
            case,
            metadata=metadata,
            lse=True,
            page_size=page_size,
            sm_scale=scale,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            output_dtype=torch.float8_e4m3fn,
        )
        inputs = _inputs(case, lse=True, scale=scale)
        inputs["out"] = torch.empty(
            (2 * q_len, 128, 512), dtype=torch.float8_e4m3fn, device="cuda"
        )
        runner = wrapper._planned_backend
        assert runner._backend == "cute-dsl-rubin-mtp"
        assert runner._execution_state.seq_lens.data_ptr() == lengths.data_ptr()
        tuning_inputs = [inputs[k] for k in ("query", "kv_cache", "out", "lse")] + [
            None
        ]
        runner.configure_tuning(
            cache_key=("fp8-sweep",),
            run_options=dict(
                return_lse=True,
                return_lse_base_on_e=False,
                o_scale=None,
                ckv_scale=None,
                kpe_scale=None,
                skip_softmax_threshold_scale_factor=None,
                bmm1_scale=scale,
                bmm2_scale=inputs["bmm2_scale"],
            ),
        )
        tactics = runner.get_valid_tactics(tuning_inputs, None)
        assert tactics == ([-1, 1] if capacity == 64 else [-1, 1, 2, 4, 8, 16, 32])
        runner.precompile_tactics(tuning_inputs, tactics, None)
        addresses = tuple(
            t.data_ptr() for t in (lengths, case["tables"], *tuning_inputs[2:4])
        )
        # Cover the public wrapper call as well as each prepared tuning tactic.
        result = wrapper.run(**inputs)
        assert result[0] is inputs["out"] and result[1] is inputs["lse"]
        expected, expected_lse = _reference(case, scale)
        torch.testing.assert_close(result[0].float(), expected, rtol=0.12, atol=0.03)
        torch.testing.assert_close(result[1], expected_lse, rtol=0.002, atol=0.02)
        for tactic in tactics:
            graph = None
            if capacity > 64:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    runner(tuning_inputs, tactic=tactic)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = runner(tuning_inputs, tactic=tactic)
            for live in (
                [min(2048, capacity)] * 2,
                [q_len, 129],
                [1, 3],
                [0, 129],
                [0, 0],
                [63, 64],
                [128, 128],
                [4, 1024],
                [513, 1024],
                [2048, 255],
            ):
                if max(live) > capacity:
                    continue
                case["lengths"] = live
                lengths.copy_(torch.tensor(live, dtype=torch.int32, device="cuda"))
                # A one-page table is padded by plan(); that derived table
                # is intentionally independent of the caller's original.
                if capacity > 64:
                    case["tables"].copy_(case["tables"].flip(-1))
                expected, expected_lse = _reference(case, scale)
                inputs["out"].fill_(float("nan"))
                inputs["lse"].fill_(float("nan"))
                if graph is None:
                    captured = runner(tuning_inputs, tactic=tactic)
                else:
                    graph.replay()
                torch.cuda.synchronize()
                actual, lse = captured
                assert actual is inputs["out"] and lse is inputs["lse"]
                assert torch.isfinite(actual.float()).all()
                torch.testing.assert_close(
                    actual.float(), expected, rtol=0.12, atol=0.03
                )
                relative_rms = (
                    actual.float() - expected
                ).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(
                    1e-6
                )
                assert relative_rms.item() <= 0.12
                empty = torch.isneginf(expected_lse)
                assert torch.equal(torch.isneginf(lse), empty)
                assert (actual.float()[empty] == 0).all()
                torch.testing.assert_close(
                    lse[~empty], expected_lse[~empty], rtol=0.002, atol=0.02
                )
                if (~empty).any():
                    lse_error = (
                        (lse[~empty] - expected_lse[~empty]).square().mean().sqrt()
                    )
                    lse_magnitude = (
                        expected_lse[~empty].square().mean().sqrt().clamp_min(1e-6)
                    )
                    assert (lse_error / lse_magnitude).item() <= 0.003
                assert addresses == tuple(
                    t.data_ptr() for t in (lengths, case["tables"], *tuning_inputs[2:4])
                )
                assert runner.validate_tactic(tuning_inputs, tactic)
    finally:
        if branch is not None:
            mla_decode._get_compiled_mla_kernel.cache_clear()


@pytest.mark.parametrize("implementation_name", ["monolithic", "rubin_mtp"])
def test_cute_split_workspace_signed_index_boundary(implementation_name):
    """Check real sizing arithmetic; adapter-refusal tests mock that boundary."""
    import importlib

    pytest.importorskip("cutlass")
    implementation = importlib.import_module(
        f"flashinfer.cute_dsl.attention.{implementation_name}.mla_decode"
    )
    kwargs = dict(
        q_len=4,
        H=128,
        kv_lora_rank=512,
        max_active_blocks=148,
        max_seq_len=4096,
        num_kv_splits=16,
    )
    # Includes LSE partials: B=511 is 8192 FP32 elements below 2^31.
    assert implementation._get_split_kv_and_workspace_size(B=511, **kwargs) == (
        16,
        8_589_901_824,
    )
    with pytest.raises(ValueError, match="signed 32-bit indexing"):
        implementation._get_split_kv_and_workspace_size(B=512, **kwargs)
    kwargs["num_kv_splits"] = 1
    assert implementation._get_split_kv_and_workspace_size(B=512, **kwargs) == (1, 0)


def test_mtp_query_capacity_and_strided_index_boundaries():
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.rubin_mtp import mla_decode

    with pytest.raises(ValueError, match="query.*signed 32-bit indexing"):
        mla_decode._get_split_kv_and_workspace_size(
            8192, 4, 128, 512, 148, 4096, num_kv_splits=1
        )
    query = torch.empty((8192, 4, 128, 576), dtype=torch.float8_e4m3fn, device="meta")
    with pytest.raises(ValueError, match="query.*signed 32-bit indexing"):
        mla_decode._check_tensor_indexing(query, "query")
    safe = torch.empty_strided((2,), ((1 << 31) - 2,), device="meta")
    unsafe = torch.empty_strided((2,), ((1 << 31) - 1,), device="meta")
    mla_decode._check_tensor_indexing(safe, "query")
    with pytest.raises(ValueError, match="query.*signed 32-bit indexing"):
        mla_decode._check_tensor_indexing(unsafe, "query")


@pytest.mark.parametrize("pool_pages", [32768, 65536])
def test_mtp_kv_indexing_allows_large_pool_span(pool_pages):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.rubin_mtp import mla_decode

    pool = torch.empty((pool_pages, 128, 576), dtype=torch.float8_e4m3fn, device="meta")
    assert pool.numel() > 1 << 31
    mla_decode._check_kv_tensor_indexing(pool, "kv_cache")
    mla_decode._check_kv_tensor_indexing(pool[..., :512], "kv_latent")
    mla_decode._check_kv_tensor_indexing(pool[..., 512:], "kv_rope")
    # Query/output/workspace indexing keeps its original span limit.
    with pytest.raises(ValueError, match="query.*signed 32-bit indexing"):
        mla_decode._check_tensor_indexing(pool, "query")


@pytest.mark.parametrize(
    "shape,strides",
    [(((1 << 31) - 1, 1), (1, 1)), ((2, 1), ((1 << 31) - 1, 1))],
)
def test_mtp_kv_indexing_accepts_individual_signed_index_limit(shape, strides):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.rubin_mtp import mla_decode

    tensor = torch.empty_strided(
        shape, strides, dtype=torch.float8_e4m3fn, device="meta"
    )
    mla_decode._check_kv_tensor_indexing(tensor, "kv_cache")


@pytest.mark.parametrize(
    "shape,strides",
    [((1 << 31, 1), (1, 1)), ((2, 1), (1 << 31, 1)), ((0, 1), (1 << 31, 1))],
)
def test_mtp_kv_indexing_rejects_unrepresentable_dimensions_or_strides(shape, strides):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl.attention.rubin_mtp import mla_decode

    tensor = torch.empty_strided(
        shape, strides, dtype=torch.float8_e4m3fn, device="meta"
    )
    with pytest.raises(ValueError, match="kv_cache.*signed 32-bit indexing"):
        mla_decode._check_kv_tensor_indexing(tensor, "kv_cache")


@pytest.mark.parametrize(
    "graph,batch,q_len,page_size,kv_len,lse_mode",
    [
        (True, 64, 4, 64, 129, "none"),
        (True, 128, 2, 128, 1024, "basee"),
        (True, 64, 4, 128, 2048, "base2"),
        (False, 64, 4, 64, 8192, "none"),
        (False, 32, 4, 128, 32768, "base2"),
        (False, 64, 2, 128, 32768, "basee"),
    ],
)
def test_auto_rubin_mtp_planned_regions(
    case, monkeypatch, graph, batch, q_len, page_size, kv_len, lse_mode
):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    pages = (kv_len + 127) // 128 * (128 // page_size)
    scale = 1 / 24
    case.update(
        query=(torch.randn(batch * q_len, 128, 576, device="cuda") * 0.3).to(
            torch.float8_e4m3fn
        ),
        kv=(
            torch.randn(
                batch * pages, page_size, 576, device="cuda", dtype=torch.bfloat16
            )
            * 0.3
        ).to(torch.float8_e4m3fn),
        tables=torch.randperm(batch * pages, device="cuda", dtype=torch.int32).view(
            batch, pages
        ),
        lengths=[kv_len] * batch,
        offsets=list(range(0, (batch + 1) * q_len, q_len)),
    )
    lengths = torch.tensor(case["lengths"], dtype=torch.int32, device="cuda")
    metadata = MLAPlanMetadata.dense(
        torch.tensor(case["offsets"], dtype=torch.int32, device="cuda"),
        case["tables"],
        lengths,
        max_q_len=q_len,
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
        backend="auto",
        use_cuda_graph=graph,
    )
    _plan(
        wrapper,
        case,
        metadata=metadata,
        page_size=page_size,
        sm_scale=scale,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        output_dtype=torch.float8_e4m3fn,
        scale_mode="default",
        lse_mode=lse_mode,
    )
    assert wrapper._planned_backend_name == "cute-dsl-rubin-mtp"
    state = wrapper._planned_backend._execution_state
    assert state.split_kv == 1
    out = torch.empty(
        (batch * q_len, 128, 512), dtype=torch.float8_e4m3fn, device="cuda"
    )
    lse = (
        torch.empty((batch * q_len, 128), device="cuda") if lse_mode != "none" else None
    )

    def run():
        result = wrapper.run(
            query=case["query"],
            kv_cache=case["kv"],
            out=out,
            lse=lse,
            return_lse=lse is not None,
            return_lse_base_on_e=lse_mode == "basee",
        )
        assert (result[0] if lse is not None else result) is out
        if lse is not None:
            assert result[1] is lse

    def check():
        expected, expected_lse = _reference(case, scale, output_scale=1.0)
        torch.testing.assert_close(out.float(), expected, rtol=0.12, atol=0.03)
        # Small FP8 outputs can have >12% error from output rounding alone.
        # Budget that unavoidable floor from the independent FP32 oracle.
        rounding_rms = (
            (expected.to(out.dtype).float() - expected).square().mean().sqrt()
        )
        error_rms = (out.float() - expected).square().mean().sqrt()
        budget = rounding_rms + 0.12 * expected.square().mean().sqrt()
        assert error_rms.item() <= budget.item(), (error_rms.item(), budget.item())
        if lse is not None:
            if lse_mode == "basee":
                expected_lse *= math.log(2)
            torch.testing.assert_close(lse, expected_lse, rtol=0.002, atol=0.02)

    run()
    check()
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            run()
        # The initial preference need not remain optimal after live lengths
        # change, but the retained graph must continue to compute correctly.
        case["lengths"] = [q_len if i % 2 else kv_len for i in range(batch)]
        lengths.copy_(torch.tensor(case["lengths"], dtype=torch.int32, device="cuda"))
        out.view(torch.uint8).fill_(0x7F)
        if lse is not None:
            lse.fill_(float("nan"))
        captured.replay()
        torch.cuda.synchronize()
        assert wrapper._planned_backend._execution_state is state
        check()
