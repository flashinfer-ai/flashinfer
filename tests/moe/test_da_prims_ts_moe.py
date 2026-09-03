"""User-facing distribution-aware PrimsTS MoE lifecycle coverage."""

import json
from typing import Any, cast

import pytest
import torch

from benchmarks.bench_moe_da import (
    BenchmarkShape,
    _benchmark_precision,
    _matching_diagnostic,
    _prepare_precision,
    _realization,
    _temporary_environment,
)
from flashinfer.autotuner import autotune
from flashinfer.fused_moe.da_tuner import RoutingRealizationFactory
from flashinfer.fused_moe.da_runtime import _stable_runner_identity
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.utils import get_compute_capability


class _ProcessLocalModule:
    """Stand in for a loaded extension whose repr contains a process-local ID."""

    def __init__(self, process_id: int) -> None:
        self.process_id = process_id

    def __repr__(self) -> str:
        return f"Module({self.process_id})"


class _Runner:
    def __init__(self, process_id: int) -> None:
        self.moe_op = _ProcessLocalModule(process_id)
        self.top_k = 4


def test_da_runner_identity_excludes_process_local_module_repr():
    """A fresh process can restore a plan built by the same module type."""
    first = _stable_runner_identity(cast(Any, _Runner(17)))
    second = _stable_runner_identity(cast(Any, _Runner(29)))

    assert first == second
    assert json.loads(first)["fields"]["moe_op"].endswith("._ProcessLocalModule")


@pytest.mark.parametrize("routing_input_mode", ("routed", "logits"))
def test_nvfp4_da_public_graph_lifecycle(monkeypatch, routing_input_mode):
    """Routed and FromLogits PrimsTS NVFP4 DA must match their NoDA graphs."""
    if not torch.cuda.is_available():
        pytest.skip("PrimsTS DA requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("This PrimsTS DA runtime test requires SM100 or SM103")
    if not is_prims_ts_available():
        pytest.skip("PrimsTS dependencies are unavailable")

    # Admit any measured DA plan so the test exercises production lifecycle rather than a
    # machine-load-dependent baseline guard outcome.
    monkeypatch.setenv("FLASHINFER_DA_BASELINE_GUARD", "0")
    shape = BenchmarkShape(
        num_tokens=256,
        num_experts=64,
        local_num_experts=32,
        local_expert_offset=16,
        top_k=4,
        hidden_size=1024,
        intermediate_size=1024,
        n_group=1,
        topk_group=1,
        tune_max_num_tokens=256,
    )
    rows = _benchmark_precision(
        "nvfp4",
        shape,
        ("uniform", "ddist:4"),
        cache=None,
        tune=True,
        warmup=1,
        iterations=2,
        backend="prims_ts",
        routing_input_mode=routing_input_mode,
    )

    assert len(rows) == 2
    assert {row["backend"] for row in rows} == {"prims_ts"}
    assert {row["routing_input_mode"] for row in rows} == {routing_input_mode}
    assert {row["status"] for row in rows} == {"pass"}
    assert all(row["finite"] for row in rows)
    assert all(row["max_abs_difference"] == 0.0 for row in rows)
    # Tactic timing decides whether the admitted plan is singleton, switch, or deliberate
    # fallback; this lifecycle contract must not depend on the runtime plan-mode outcome.
    assert all(row["capture_policy"] != "noda_capture_fallback" for row in rows)


@pytest.mark.parametrize(
    "precision",
    ("bf16", "mxfp4", "w4a16", "fp8_per_tensor", "fp8_block", "mxfp8"),
)
def test_prims_ts_supported_dtype_da_graph_lifecycle(monkeypatch, precision):
    """Every ordinary PrimsTS dtype must tune and replay through its public DA API."""
    if not torch.cuda.is_available():
        pytest.skip("PrimsTS DA requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("This PrimsTS DA runtime test requires SM100 or SM103")
    if not is_prims_ts_available():
        pytest.skip("PrimsTS dependencies are unavailable")

    monkeypatch.setenv("FLASHINFER_DA_BASELINE_GUARD", "0")
    shape = BenchmarkShape(
        num_tokens=256,
        num_experts=32,
        local_num_experts=32,
        local_expert_offset=0,
        top_k=4,
        hidden_size=1024,
        intermediate_size=1024,
        n_group=1,
        topk_group=1,
        tune_max_num_tokens=256,
    )
    rows = _benchmark_precision(
        precision,
        shape,
        ("uniform", "ddist:4"),
        cache=None,
        tune=True,
        warmup=1,
        iterations=2,
        backend="prims_ts",
    )

    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"pass"}
    assert all(row["finite"] for row in rows)
    # Tactic timing decides whether the admitted plan is singleton, switch, or deliberate
    # fallback; this lifecycle contract must not depend on the runtime plan-mode outcome.
    assert all(row["capture_policy"] != "noda_capture_fallback" for row in rows)


def test_nvfp4_da_public_eager_uses_ddist_1_1_tactic(monkeypatch):
    """PrimsTS eager dispatch uses the DA-preferred tactic for its token bucket."""
    if not torch.cuda.is_available():
        pytest.skip("PrimsTS DA requires CUDA")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("This PrimsTS DA runtime test requires SM100 or SM103")
    if not is_prims_ts_available():
        pytest.skip("PrimsTS dependencies are unavailable")

    monkeypatch.setenv("FLASHINFER_DA_BASELINE_GUARD", "0")
    distributions = ("uniform", "ddist:1.1")
    shape = BenchmarkShape(
        num_tokens=24,
        num_experts=32,
        local_num_experts=32,
        local_expert_offset=0,
        top_k=4,
        hidden_size=1024,
        intermediate_size=1024,
        n_group=1,
        topk_group=1,
        tune_max_num_tokens=24,
    )
    prepared = _prepare_precision("nvfp4", shape, backend="prims_ts")
    ids, weights = _realization(RoutingRealizationFactory(), shape, "ddist:1.1")
    prepared.stage(ids, weights)

    with _temporary_environment(FLASHINFER_DIST_AWARE_AUTOTUNE="0"):
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            prepared.invoke()
        torch.cuda.synchronize()
        ordinary = prepared.output.clone()

    with _temporary_environment(
        FLASHINFER_DIST_AWARE_AUTOTUNE="1",
        FLASHINFER_DA_DISTRIBUTIONS=",".join(distributions),
    ):
        with autotune(True, tuning_buckets=(shape.num_tokens,)):
            prepared.invoke()
        prepared.invoke()
        torch.cuda.synchronize()
        eager = prepared.output.clone()

    torch.testing.assert_close(eager, ordinary, rtol=3e-2, atol=3e-2)
    diagnostic = _matching_diagnostic("nvfp4", shape, distributions, backend="prims_ts")
    assert diagnostic["eager_distribution"] == "ddist:1.1"
    assert diagnostic["eager_body"] is not None
    assert diagnostic["topology"] is None
