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
        outputs.append(
            torch.einsum("qhk,kd->qhd", scores.softmax(-1), kv[:, :512]) * output_scale
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
    with pytest.raises(ValueError, match="out must be contiguous"):
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


def test_insufficient_cute_workspace_leaves_trt_candidate(case):
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
        backend._BatchMLAPagedAttentionTrtllmGenBackend
    ]


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
