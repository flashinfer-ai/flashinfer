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
