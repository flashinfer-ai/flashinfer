"""Tuner failure policy, numerical gating, and collective latency statistics."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    Sm107BlockScaledMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    autotune,
    block_scaled,
    correctness,
    knob_cache,
)


def test_candidate_deduplication_preserves_order_and_normalizes_json():
    grid = autotune.sm107_candidates()
    first, second = grid[1], grid[0]
    duplicate = dict(reversed(json.loads(json.dumps(first)).items()))
    supplied = [first, duplicate, second, first]
    snapshot = json.dumps(supplied)
    assert autotune._unique_candidates(supplied) == [first, second]
    assert json.dumps(supplied) == snapshot
    assert autotune._unique_candidates([]) == []


@pytest.mark.parametrize("variant", ["inference", "genphase"])
@pytest.mark.parametrize("limit", [2, 16])
def test_cli_deduplicates_before_limiting_candidates(monkeypatch, variant, limit):
    from flashinfer.moe_ep.backends.mega.kernel import tuning as shared
    from flashinfer.moe_ep.backends.mega.kernel.sm107 import tuning
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as package

    first, second = autotune.sm107_candidates(kernel_variant=variant)[:2]
    monkeypatch.setattr(
        package,
        "sm107_candidates",
        lambda *a, **kw: [first, json.loads(json.dumps(first)), second],
    )
    args = SimpleNamespace(
        sweep="structural",
        allow_nondeterministic=False,
        combine_dtype="bf16",
        skew=None,
        max_candidates=limit,
        dtype="nvfp4",
        warmup_iters=1,
        timed_iters=2,
    )
    candidates = tuning._candidates(args, 0, 1, 128, "nvfp4", variant)
    selected = mock.Mock(return_value=first)
    shared.finish_sweep(args, 0, 128, 4, None, None, None, None, candidates, selected)
    assert selected.call_args.kwargs["candidates"] == [first, second]


def _load_benchmark_module(path: Path):
    if not path.is_file():
        pytest.skip("requires the source-tree SM107 benchmark")
    spec = importlib.util.spec_from_file_location("sm107_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/moe_ep/backends/mega/kernel/sm107/bench_moe_ep_sm107_block_scaled_mega.py"
    )
    return _load_benchmark_module(path)


def test_missing_source_tree_benchmark_skips(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="source-tree SM107 benchmark"):
        _load_benchmark_module(tmp_path / "missing_benchmark.py")


def test_latency_reports_primary_and_rank0_medians(benchmark_module):
    # The primary median differs from rank zero and the iteration-max median.
    result = benchmark_module._summarize_samples([[1, 100], [100, 3]])
    assert result == {
        "primary_latency_statistic": "max_rank_p50_us",
        "max_rank_p50_us": 51.5,
        "p50_rank0_us": 50.5,
        "per_rank_samples_us": [[1, 100], [100, 3]],
    }


@pytest.mark.parametrize(
    "kind", ["nvfp4", "mxfp4", "mxfp4_mxfp8", "mxfp8_e4m3", "mxfp8_e5m2"]
)
def test_benchmark_default_config_preserves_standard_settings(
    benchmark_module, monkeypatch, kind
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    args, _, _, _ = benchmark_module._parse_args([])
    cfg = benchmark_module._make_backend_config(args, None, kind)
    assert cfg.kernel_variant == "inference"
    assert cfg.activation == "swiglu" and cfg.gate_up_clamp is None
    assert cfg.apply_routing_weights_before_fc2 is True
    assert cfg.knobs is None and cfg.fc2_use_bulk is False
    assert cfg.use_tanh_sigmoid is False and cfg.epilogue_warpgroups == 1


def test_benchmark_selects_situ_epilogue_options(benchmark_module, monkeypatch):
    from flashinfer.moe_ep import FleetParams
    from flashinfer.moe_ep.backends.mega.kernel.sm107.validation import (
        make_workspace_config,
    )

    monkeypatch.setenv("WORLD_SIZE", "4")
    args, _, _, _ = benchmark_module._parse_args(
        [
            "--activation",
            "situ",
            "--situ-beta",
            "4",
            "--situ-linear-beta",
            "25",
            "--use-tanh-sigmoid",
            "--epilogue-warpgroups",
            "2",
            "--routing-weight-position",
            "late",
        ]
    )
    knobs = {"fc2_use_bulk": True, "fc2_tma_stages": 1}
    cfg = benchmark_module._make_backend_config(args, knobs, "nvfp4")
    assert (cfg.activation, cfg.situ_beta, cfg.situ_linear_beta) == ("situ", 4.0, 25.0)
    assert cfg.use_tanh_sigmoid is True and cfg.epilogue_warpgroups == 2
    assert cfg.apply_routing_weights_before_fc2 is False
    assert cfg.knobs == knobs
    workspace_config = make_workspace_config(
        cfg,
        FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=3584),
        rank=0,
        world_size=4,
        quant_kind="nvfp4",
        gate_up_clamp=cfg.gate_up_clamp,
        overrides=cfg.knobs,
    )
    monkeypatch.setattr(block_scaled, "_max_active_clusters", lambda size: 128 // size)
    kernel = block_scaled.Sm107BlockScaledSymmBuffer._build_kernel(workspace_config)
    assert kernel.epilogue.use_tanh_sigmoid is True
    assert kernel.epilogue.two_epilogue_warpgroups is True


def test_benchmark_genphase_uses_admitted_launch_defaults(
    benchmark_module, monkeypatch
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    args, tokens, _, _ = benchmark_module._parse_args(
        ["--kernel-variant", "genphase", "--knobs", "heuristic"]
    )
    cfg = benchmark_module._make_backend_config(args, None, "mxfp4_mxfp8")
    assert cfg.kernel_variant == "genphase"
    assert cfg.cluster_shape_mn == (4, 1) and cfg.fallback_cluster_shape_mn is None
    assert cfg.fc2_use_bulk is True
    assert max(tokens) <= 1024


def test_benchmark_blackwell_input_profile_keeps_default_clamp(
    benchmark_module, monkeypatch
):
    monkeypatch.setenv("WORLD_SIZE", "4")
    args, _, _, _ = benchmark_module._parse_args(["--input-profile", "blackwell"])
    assert (
        benchmark_module._make_backend_config(args, None, "nvfp4").gate_up_clamp == 10.0
    )


def test_gaussian_routing_keeps_selected_scores(benchmark_module, monkeypatch):
    """Historical routing uses raw scores, including negative selected values."""
    monkeypatch.setattr(benchmark_module, "SEED", 0)
    monkeypatch.setattr(benchmark_module, "NUM_EXPERTS", 4)
    monkeypatch.setattr(benchmark_module, "TOP_K", 2)
    scores = [
        torch.tensor([[1.0, -2.0, 3.0, 0.5], [-4.0, -1.0, -3.0, -2.0]]),
        torch.tensor([[0.0, 5.0, -1.0, 4.0], [7.0, 2.0, 6.0, -3.0]]),
    ]
    seeds = []

    def randn(tokens, experts, *, dtype, device, generator):
        assert (tokens, experts, dtype, device) == (2, 4, torch.float32, "cpu")
        seeds.append(generator.initial_seed())
        return scores[len(seeds) - 1]

    monkeypatch.setattr(torch, "randn", randn)
    ids, weights = benchmark_module._make_routing(2, 2, "gaussian", 0.8, device="cpu")
    assert seeds == [17, 18]
    expected_ids = torch.tensor([[[2, 0], [1, 3]], [[1, 3], [0, 2]]])
    assert ids.dtype == torch.int32
    # Top-k is unsorted; compare expert sets and each associated raw score.
    torch.testing.assert_close(ids.long().sort(-1).values, expected_ids.sort(-1).values)
    for rank in range(2):
        torch.testing.assert_close(
            weights[rank], scores[rank].gather(1, ids[rank].long())
        )


@pytest.mark.parametrize(
    "mode,execution,no_flush,staging,ownership,allocation",
    [
        ("kernel", "eager", False, False, "workspace_view", "none"),
        ("compute", "eager", False, False, "owned", "before_warmup"),
        ("compute", "graph", True, False, "owned", "before_warmup"),
        ("forward", "eager", True, True, "owned", "per_call"),
        ("forward", "graph", False, True, "owned", "during_capture"),
    ],
)
def test_benchmark_reports_staging_and_capture_allocation(
    benchmark_module, mode, execution, no_flush, staging, ownership, allocation
):
    protocol = benchmark_module._timing_protocol(
        SimpleNamespace(
            mode=mode,
            execution=execution,
            no_l2_flush=no_flush,
            warmup=20,
            prequantized=False,
            eager_completed_wall=False,
        )
    )
    assert protocol["input_staging_in_timed_span"] is staging
    assert protocol["output_ownership"] == ownership
    assert protocol["output_allocation"] == allocation
    assert protocol["host_wall_time_measured"] is False
    assert protocol["l2_flush_bytes"] == (0 if no_flush else 300 * 1024 * 1024)
    assert protocol["l2_flush_method"] == (
        "none" if no_flush else "per_iteration_fp32_randn"
    )
    assert protocol["graph_replay_warmup"] == (20 if execution == "graph" else 0)


@pytest.mark.parametrize("execution", ["eager", "graph"])
def test_compute_reuses_owned_output_with_changing_inputs(benchmark_module, execution):
    if execution == "graph" and not torch.cuda.is_available():
        pytest.skip("CUDA graph replay requires a CUDA device")
    device = "cuda" if execution == "graph" else "cpu"
    x = torch.ones((2, 4), dtype=torch.bfloat16, device=device)
    workspace = SimpleNamespace(x=x)
    transformed = object()

    def compute(staged, weights, *, output):
        assert staged is workspace and weights is transformed
        output.copy_(staged.x)
        return output

    layer = SimpleNamespace(
        _workspace=workspace, _kernel=SimpleNamespace(compute=compute)
    )
    invoke = benchmark_module._make_compute_call(layer, transformed, x)
    output = invoke()
    assert output.data_ptr() != x.data_ptr()
    if execution == "graph":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = invoke()
        assert captured_output is output
        replay = graph.replay
    else:

        def replay():
            assert invoke() is output

    for value in (2, 7):
        x.fill_(value)
        replay()
        torch.testing.assert_close(output, x)


def test_l2_flush_uses_fresh_storage(benchmark_module, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("L2 flush requires a CUDA device")
    monkeypatch.setattr(benchmark_module, "L2_FLUSH_BYTES", 4096)
    first = benchmark_module._l2_flush()
    second = benchmark_module._l2_flush()
    assert first.data_ptr() != second.data_ptr()
    assert first.numel() * first.element_size() == 4096
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


@pytest.fixture
def fake_trials(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA events require a CUDA device; no Rubin kernel runs")
    config = Sm107BlockScaledMoeConfig(
        num_total_experts=4,
        max_tokens_per_rank=4,
        num_topk=2,
        hidden=128,
        intermediate=64,
        rank=0,
        world_size=1,
    )
    trials = []

    class Trial:
        _build_kernel = staticmethod(lambda cfg: None)
        fail = False

        def __init__(self, cfg):
            from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_state import (
                AutoTuningState,
            )

            self.config = cfg
            self._tuning_state = AutoTuningState()
            self.x = torch.zeros(4, 128, device="cuda")
            self.x_sf = torch.zeros(4, 4, device="cuda")
            self.topk_idx = torch.zeros(4, 2, device="cuda", dtype=torch.int32)
            self.topk_weights = torch.ones(4, 2, device="cuda")
            self.output_activation = torch.zeros(4, 128, device="cuda")
            self.prepared = False
            self.destroyed = False
            trials.append(self)

        def note_staged_tokens(self, count):
            self.count = count

        def staged_tokens(self):
            return self.count

        def prepare(self, *weights):
            self.prepared = True

        def launch(self, *weights):
            assert self.prepared
            if self.fail:
                raise RuntimeError("injected rank-local CUDA failure")
            self.output_activation.fill_(9 if self.config.fc2_use_bulk else 3)

        def destroy(self):
            self.destroyed = True

    source = Trial(config)
    source.note_staged_tokens(1)

    def run(y, w1, w2, trial, **kw):
        trial.launch(w1, w2)
        y.copy_(trial.output_activation[:1])

    monkeypatch.setattr(block_scaled, "Sm107BlockScaledSymmBuffer", Trial)
    monkeypatch.setattr(block_scaled, "sm107_block_scaled_mega_moe", run)
    monkeypatch.setattr(
        correctness,
        "sampled_reference",
        lambda *args, **kw: (
            torch.tensor([0], device="cuda"),
            torch.full((1, 128), 3.0, device="cuda"),
        ),
    )
    recorded = mock.Mock()
    monkeypatch.setattr(knob_cache, "write_record", recorded)
    return SimpleNamespace(source=source, trial=Trial, trials=trials, recorded=recorded)


def test_incorrect_candidate_cannot_enter_cache(fake_trials):
    state = fake_trials
    winner = autotune.autotune_sm107_block_scaled_mega_moe(
        torch.empty(1, 128, device="cuda"),
        None,
        None,
        state.source,
        candidates=[
            {"fc2_use_bulk": True},
            {"fc2_use_bulk": True},
            {"fc2_use_bulk": False},
            {"fc2_use_bulk": False},
        ],
        warmup_iters=1,
        timed_iters=2,
    )
    assert winner == {"fc2_use_bulk": False}
    assert len(state.trials) == 3  # Source workspace and two distinct candidates.
    assert all(t.destroyed for t in state.trials[1:])
    assert state.recorded.call_args.args[0]["knobs"] == winner


def test_gpu_failure_aborts_without_collective_free_or_cache_write(fake_trials):
    state = fake_trials
    state.trial.fail = True
    with pytest.raises(RuntimeError, match="rank-local CUDA failure"):
        autotune.autotune_sm107_block_scaled_mega_moe(
            torch.empty(1, 128, device="cuda"),
            None,
            None,
            state.source,
            candidates=[{"fc2_use_bulk": False}],
            warmup_iters=1,
            timed_iters=1,
        )
    assert not state.trials[-1].destroyed
    state.recorded.assert_not_called()
