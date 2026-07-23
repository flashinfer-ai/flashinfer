import argparse
import sys

import pytest
import torch

from benchmarks import bench_moe_deepseek


def test_run_benchmark_autotunes_once_then_uses_cache(monkeypatch):
    calls = []

    def record_single(*_args, **kwargs):
        calls.append(kwargs)
        return [], None

    monkeypatch.setattr(bench_moe_deepseek, "_benchmark_single", record_single)

    bench_moe_deepseek.run_benchmark(
        token_counts=(64, 128),
        distributions=("ddist:1", "ddist:2"),
        backends=("trtllm",),
        do_autotune=True,
        verbose=False,
    )

    assert [call["do_autotune"] for call in calls] == [True, False, False, False]
    assert all(call["tuning_buckets"] == calls[0]["tuning_buckets"] for call in calls)


def test_cuda_graph_defaults_on_and_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(bench_moe_deepseek, "is_sm100_family", lambda: True)
    monkeypatch.setattr(
        bench_moe_deepseek.torch.cuda, "get_device_name", lambda _device: "B200"
    )
    monkeypatch.setattr(
        bench_moe_deepseek, "run_benchmark", lambda **kwargs: calls.append(kwargs)
    )

    monkeypatch.setattr(
        sys, "argv", ["bench_moe_deepseek.py", "--num-tokens", "8", "--quiet"]
    )
    assert bench_moe_deepseek.main() == 0
    assert calls[-1]["use_cuda_graph"] is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_moe_deepseek.py",
            "--num-tokens",
            "8",
            "--quiet",
            "--no-cuda-graph",
        ],
    )
    assert bench_moe_deepseek.main() == 0
    assert calls[-1]["use_cuda_graph"] is False


def test_routing_input_mode_defaults_to_logits_and_accepts_routed(monkeypatch):
    calls = []
    monkeypatch.setattr(bench_moe_deepseek, "is_sm100_family", lambda: True)
    monkeypatch.setattr(
        bench_moe_deepseek.torch.cuda, "get_device_name", lambda _device: "B200"
    )
    monkeypatch.setattr(
        bench_moe_deepseek, "run_benchmark", lambda **kwargs: calls.append(kwargs)
    )

    monkeypatch.setattr(
        sys, "argv", ["bench_moe_deepseek.py", "--num-tokens", "8", "--quiet"]
    )
    assert bench_moe_deepseek.main() == 0
    assert calls[-1]["routing_input_mode"] == "logits"
    assert calls[-1]["backends"] == bench_moe_deepseek.BACKENDS

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_moe_deepseek.py",
            "--num-tokens",
            "8",
            "--quiet",
            "--routing-input-mode",
            "routed",
        ],
    )
    assert bench_moe_deepseek.main() == 0
    assert calls[-1]["routing_input_mode"] == "routed"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_moe_deepseek.py",
            "--num-tokens",
            "8",
            "--quiet",
            "--backends",
            "trtllm,cutlass",
        ],
    )
    assert bench_moe_deepseek.main() == 0
    assert calls[-1]["backends"] == ("trtllm", "cutlass")


def test_routing_input_mode_is_forwarded_to_all_backends(monkeypatch):
    calls = {}
    routed_input = (object(), object())
    monkeypatch.setattr(
        bench_moe_deepseek, "create_inputs", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "prepare_routed_input",
        lambda *_args, **_kwargs: routed_input,
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "_collect_expert_histogram",
        lambda *_args, **_kwargs: {},
    )

    def record_call(name):
        def mock_backend(*_args, **kwargs):
            calls[name] = kwargs
            return 1.0

        return mock_backend

    for name in ("bench_cute_dsl", "bench_cutlass", "bench_trtllm"):
        monkeypatch.setattr(
            bench_moe_deepseek,
            name,
            record_call(name),
        )
    bench_moe_deepseek._benchmark_single(
        8,
        1,
        1,
        16,
        0,
        True,
        False,
        routing_input_mode="routed",
    )
    assert set(calls) == {"bench_cute_dsl", "bench_cutlass", "bench_trtllm"}
    for kwargs in calls.values():
        assert kwargs["routing_input_mode"] == "routed"
        assert kwargs["routed_input"] is routed_input


def test_explicit_routed_distribution_bypasses_grouped_logits_router(monkeypatch):
    inputs = {
        "router_logits": torch.zeros(2, 16),
        "routing_bias": torch.zeros(16),
    }
    routed_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=torch.int32,
    )
    captured = {}
    monkeypatch.setattr(
        bench_moe_deepseek, "create_inputs", lambda *_args, **_kwargs: inputs
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "generate_routing_distribution",
        lambda *_args, **_kwargs: routed_ids,
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "prepare_routed_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit routed input must bypass grouped routing")
        ),
    )

    def collect_histogram(_inputs, _num_local, _local_offset, routed_ids=None):
        captured["histogram_ids"] = routed_ids
        return {
            "active_local_experts": 16,
            "min_count": 1,
            "max_count": 1,
            "median_count": 1.0,
        }

    def run_trtllm(*_args, **kwargs):
        captured["routed_input"] = kwargs["routed_input"]
        return 1.0

    monkeypatch.setattr(
        bench_moe_deepseek, "_collect_expert_histogram", collect_histogram
    )
    monkeypatch.setattr(bench_moe_deepseek, "bench_trtllm", run_trtllm)

    bench_moe_deepseek._benchmark_single(
        2,
        1,
        1,
        16,
        0,
        True,
        False,
        routing_input_mode="routed",
        backends=("trtllm",),
        distribution="ddist:3",
    )

    actual_ids, actual_weights = captured["routed_input"]
    assert actual_ids is routed_ids
    assert captured["histogram_ids"] is routed_ids
    torch.testing.assert_close(actual_weights, torch.full_like(actual_weights, 2.5 / 8))


def test_routed_distribution_is_stable_across_matched_local_shapes(monkeypatch):
    monkeypatch.setattr(bench_moe_deepseek.CFG, "num_experts", 16)
    ids_global16 = bench_moe_deepseek.generate_routing_distribution(
        {"router_logits": torch.zeros(64, 16)}, "ddist:3", 16, 0
    )
    torch.rand(1024)
    monkeypatch.setattr(bench_moe_deepseek.CFG, "num_experts", 256)
    ids_global256 = bench_moe_deepseek.generate_routing_distribution(
        {"router_logits": torch.zeros(64, 256)}, "ddist:3", 16, 0
    )
    assert torch.equal(ids_global16, ids_global256)


def test_backend_filter_skips_unselected_backends(monkeypatch):
    monkeypatch.setattr(
        bench_moe_deepseek, "create_inputs", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "_collect_expert_histogram",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "bench_cute_dsl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "bench_cutlass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        bench_moe_deepseek, "bench_trtllm", lambda *_args, **_kwargs: 1.0
    )

    results, _ = bench_moe_deepseek._benchmark_single(
        8,
        1,
        1,
        16,
        0,
        True,
        False,
        backends=("trtllm",),
    )
    assert [result.backend for result in results] == ["TRTLLM"]


def test_per_token_activation_preserves_supported_backend_controls(monkeypatch):
    calls = {}
    routed_input = (object(), object())
    monkeypatch.setattr(
        bench_moe_deepseek, "create_inputs", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "_collect_expert_histogram",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        bench_moe_deepseek,
        "prepare_routed_input",
        lambda *_args, **_kwargs: routed_input,
    )

    def record_call(name):
        def mock_backend(*_args, **kwargs):
            calls[name] = kwargs
            return 1.0

        return mock_backend

    monkeypatch.setattr(bench_moe_deepseek, "bench_cute_dsl", record_call("cutedsl"))
    monkeypatch.setattr(
        bench_moe_deepseek,
        "bench_cutlass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(bench_moe_deepseek, "bench_trtllm", record_call("trtllm"))

    bench_moe_deepseek._benchmark_single(
        8,
        1,
        1,
        16,
        0,
        True,
        False,
        routing_input_mode="routed",
        backends=("cutedsl", "cutlass", "trtllm"),
        use_per_token_activation=True,
        use_fused_finalize=False,
    )

    assert set(calls) == {"cutedsl", "trtllm"}
    assert calls["cutedsl"]["use_per_token_activation"] is True
    assert calls["cutedsl"]["use_fused_finalize"] is False
    assert calls["trtllm"]["use_per_token_activation"] is True
    assert all(call["routing_input_mode"] == "routed" for call in calls.values())


def test_backend_parser_rejects_unknown_or_empty_values():
    assert bench_moe_deepseek.parse_backends("trtllm,cutedsl,trtllm") == (
        "trtllm",
        "cutedsl",
    )
    with pytest.raises(argparse.ArgumentTypeError):
        bench_moe_deepseek.parse_backends("trtllm,unknown")
    with pytest.raises(argparse.ArgumentTypeError):
        bench_moe_deepseek.parse_backends("")


def test_print_row_supports_backend_subset(capsys):
    result = bench_moe_deepseek.BenchResult("TRTLLM", 8, 0.1, 1.0)
    bench_moe_deepseek._print_row(
        [result],
        {
            "active_local_experts": 2,
            "min_count": 0,
            "max_count": 2,
            "median_count": 1.0,
        },
    )
    output = capsys.readouterr().out
    assert "TRTLLM" in output
    assert output.count("n/a") == 6
