"""Check benchmark workload selection and timing labels without initializing CUDA."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest


def benchmark():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/moe_ep/backends/mega/kernel/sm107/bench_moe_ep_sm107_block_scaled_mega.py"
    )
    spec = importlib.util.spec_from_file_location("sm107_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules, {"torch": MagicMock(), "torch.distributed": MagicMock()}
    ):
        spec.loader.exec_module(module)
    return module


def arguments(**kwargs):
    return argparse.Namespace(
        **dict(
            {
                "testlist": None,
                "mode": "forward",
                "knobs": "heuristic",
                "prequantized": False,
                "eager_completed_wall": False,
                "execution": "graph",
                "no_l2_flush": True,
                "warmup": 20,
            },
            **kwargs,
        )
    )


def row():
    return dict(
        case_id="control",
        world_size=4,
        tokens=8,
        capacity=256,
        hidden=2048,
        intermediate=768,
        num_experts=128,
        topk=8,
        quant_kind="nvfp4",
        kernel_variant="inference",
        prequantized=False,
    )


def test_standalone_and_testlist_remain_distinct(tmp_path):
    bench = benchmark()
    args = arguments()
    assert bench._testlist(args, 4) == [args]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([row(), {**row(), "case_id": "ep2", "world_size": 2}]))
    cases = bench._testlist(arguments(testlist=str(path)), 4)
    assert len(cases) == 1 and cases[0].tokens == "8" and cases[0].capacity == 256


@pytest.mark.parametrize(
    "change",
    [
        dict(capacity=4),
        dict(topk=129),
        dict(num_experts=127),
        dict(prequantized="False"),
        dict(kernel_variant="unknown"),
        dict(quant_kind="unknown"),
    ],
)
def test_testlist_rejects_invalid_contract(tmp_path, change):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{**row(), **change}]))
    with pytest.raises(ValueError):
        benchmark()._testlist(arguments(testlist=str(path)), 4)


@pytest.mark.parametrize(
    "field", ("hidden", "intermediate", "num_experts", "topk", "capacity")
)
@pytest.mark.parametrize("value", (True, 2048.5, None, "2048"))
def test_testlist_requires_integer_dimensions(tmp_path, field, value):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{**row(), field: value}]))
    with pytest.raises(ValueError, match="must be integers"):
        benchmark()._testlist(arguments(testlist=str(path)), 4)


@pytest.mark.parametrize("knobs,capacity", [("default", 256), ("heuristic", 1025)])
def test_testlist_rejects_unsupported_genphase_settings(tmp_path, knobs, capacity):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{**row(), "kernel_variant": "genphase", "capacity": capacity}])
    )
    with pytest.raises(ValueError, match="GenPhase testlists require"):
        benchmark()._testlist(arguments(testlist=str(path), knobs=knobs), 4)


@pytest.mark.parametrize("capacity", [256, 1024])
def test_testlist_accepts_genphase_heuristics(tmp_path, capacity):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{**row(), "kernel_variant": "genphase", "capacity": capacity}])
    )
    cases = benchmark()._testlist(arguments(testlist=str(path)), 4)
    assert len(cases) == 1 and cases[0].capacity == capacity


def test_duplicate_ids_and_missing_ep_are_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([row(), row()]))
    with pytest.raises(ValueError, match="unique"):
        benchmark()._testlist(arguments(testlist=str(path)), 4)
    path.write_text(json.dumps([row()]))
    with pytest.raises(ValueError, match="no testlist cases"):
        benchmark()._testlist(arguments(testlist=str(path)), 2)


def test_host_wall_timing_is_explicit():
    bench = benchmark()
    graph = bench._timing_protocol(arguments())
    wall = bench._timing_protocol(
        arguments(execution="eager", eager_completed_wall=True)
    )
    assert (
        graph["timing_scope"] == "gpu_graph_replay"
        and not graph["host_wall_time_measured"]
    )
    assert (
        wall["timing_scope"] == "host_dispatch_and_completion"
        and wall["host_wall_time_measured"]
    )
    assert "after_each_call" in wall["synchronization"]
    assert (
        bench._timing_protocol(arguments(prequantized=True))["timed_span"]
        == "prequantized_public_forward"
    )


def test_genphase_heuristic_is_not_inference_heuristic():
    pkg = MagicMock()
    benchmark()._selected_knobs(
        pkg, "heuristic", 8, 256, "balanced", "nvfp4", 4, "genphase"
    )
    pkg.default_knobs.assert_called_once_with(
        256, quant_kind="nvfp4", kernel_variant="genphase"
    )


def main_benchmark(monkeypatch):
    bench = benchmark()
    ep = MagicMock()
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", ep)
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep.core.runtime", MagicMock())
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe",
        MagicMock(),
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    bench.torch.cuda.get_device_capability.return_value = (10, 7)
    bench.torch.cuda.max_memory_allocated.return_value = 0
    bench.torch.cuda.max_memory_reserved.return_value = 0
    monkeypatch.setattr(bench, "_environment", lambda: {})
    monkeypatch.setattr(bench, "_local_transformed_weights", MagicMock())

    def gather(gathered, local):
        gathered[:] = [local] * len(gathered)

    bench.dist.all_gather_object.side_effect = gather
    return bench, ep


def test_invalid_testlist_fails_before_runtime_setup(tmp_path, monkeypatch):
    bench, ep = main_benchmark(monkeypatch)
    path = tmp_path / "cases.json"
    path.write_text("[]")
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--testlist", str(path), "--mode", "forward"],
    )
    with pytest.raises(ValueError, match="no testlist cases"):
        bench.main()
    ep.bootstrap_moe_ep_runtime.assert_not_called()
    bench.torch.cuda.get_device_capability.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        ["--tokens", "8", "--knobs", "default"],
        ["--tokens", "8,1025", "--knobs", "heuristic"],
        ["--tokens", "8", "--capacity", "1025", "--knobs", "heuristic"],
    ],
)
def test_standalone_genphase_rejects_invalid_settings_before_setup(monkeypatch, extra):
    bench, ep = main_benchmark(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--kernel-variant", "genphase", *extra]
    )
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2
    ep.bootstrap_moe_ep_runtime.assert_not_called()
    bench.torch.cuda.get_device_capability.assert_not_called()
    bench._local_transformed_weights.assert_not_called()


@pytest.mark.parametrize(
    "variant,knobs,tokens,capacity",
    [
        ("genphase", "heuristic", "8", 1024),
        ("genphase", "cache", "8", 1024),
        (
            "genphase",
            '{"cluster_shape_mn": [4, 1], "fc2_use_bulk": true}',
            "8",
            1024,
        ),
        ("inference", "default", "2048", 2048),
    ],
)
def test_standalone_accepts_supported_policies(
    tmp_path, monkeypatch, variant, knobs, tokens, capacity
):
    bench, ep = main_benchmark(monkeypatch)
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--mode",
            "forward",
            "--kernel-variant",
            variant,
            "--knobs",
            knobs,
            "--quant-kind",
            "nvfp4",
            "--routing",
            "balanced",
            "--tokens",
            tokens,
            "--capacity",
            str(capacity),
            "--output",
            str(output),
        ],
    )
    call = MagicMock(return_value=({"execution": "graph"}, [1.0, 2.0]))
    monkeypatch.setattr(bench, "_bench_one", call)
    bench.main()
    call.assert_called_once()
    assert call.call_args.args[2:4] == (int(tokens), capacity)
    ep.finalize_moe_ep_runtime.assert_called_once()
    assert len(output.read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "routing,expected",
    [
        ("both", ["balanced", "power_law"]),
        ("balanced", ["balanced"]),
        ("power_law", ["power_law"]),
        ("gaussian", ["gaussian"]),
    ],
)
def test_testlist_executes_selected_routings(tmp_path, monkeypatch, routing, expected):
    bench, ep = main_benchmark(monkeypatch)
    path = tmp_path / "cases.json"
    output = tmp_path / "results.jsonl"
    path.write_text(json.dumps([row()]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--testlist",
            str(path),
            "--mode",
            "forward",
            "--knobs",
            "heuristic",
            "--routing",
            routing,
            "--output",
            str(output),
        ],
    )

    def run_case(rank, world, tokens, capacity, route, transformed, args, kind):
        return {"routing": route, "execution": args.execution}, [1.0, 2.0]

    monkeypatch.setattr(bench, "_bench_one", run_case)
    bench.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(r["routing"], r["execution"]) for r in records] == [
        (route, execution) for route in expected for execution in ("graph", "eager")
    ]
    ep.finalize_moe_ep_runtime.assert_called_once()
