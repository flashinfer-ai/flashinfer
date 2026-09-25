"""Tuner failure policy, numerical gating, and collective latency statistics."""

import importlib.util
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


@pytest.fixture
def benchmark_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/bench_moe_ep_sm107_block_scaled_mega.py"
    )
    spec = importlib.util.spec_from_file_location("sm107_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latency_reports_primary_and_rank0_medians(benchmark_module):
    # The primary median differs from rank zero and the iteration-max median.
    result = benchmark_module._summarize_samples([[1, 100], [100, 3]])
    assert result == {
        "primary_latency_statistic": "max_rank_p50_us",
        "max_rank_p50_us": 51.5,
        "p50_rank0_us": 50.5,
        "per_rank_samples_us": [[1, 100], [100, 3]],
    }


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
        SimpleNamespace(mode=mode, execution=execution, no_l2_flush=no_flush, warmup=20)
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
            self.config = cfg
            self.x = torch.zeros(4, 128, device="cuda")
            self.x_sf = torch.zeros(4, 4, device="cuda")
            self.topk_idx = torch.zeros(4, 2, device="cuda", dtype=torch.int32)
            self.topk_weights = torch.ones(4, 2, device="cuda")
            self.output_activation = torch.zeros(4, 128, device="cuda")
            self.destroyed = False
            trials.append(self)

        def note_staged_tokens(self, count):
            self.count = count

        def staged_tokens(self):
            return self.count

        def launch(self, *weights):
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
    monkeypatch.setattr(knob_cache, "record_knobs", recorded)
    return SimpleNamespace(source=source, trial=Trial, trials=trials, recorded=recorded)


def test_incorrect_candidate_cannot_enter_cache(fake_trials):
    state = fake_trials
    winner = autotune.autotune_sm107_block_scaled_mega_moe(
        torch.empty(1, 128, device="cuda"),
        None,
        None,
        state.source,
        candidates=[{"fc2_use_bulk": True}, {"fc2_use_bulk": False}],
        warmup_iters=1,
        timed_iters=2,
    )
    assert winner == {"fc2_use_bulk": False}
    assert all(t.destroyed for t in state.trials[1:])
    assert state.recorded.call_args.args[0] == winner


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
