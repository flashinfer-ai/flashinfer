"""Exercise the minimum viable reference design for DA MoE capture and replay."""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator, Sequence

import torch

from flashinfer.autotuner import TuningConfig, autotune
from flashinfer.fused_moe.da_moe import DAGraphLease
from tests.moe.mock_da_moe import (
    MockDAMoERunner,
    MockMoEInputs,
)


def _expert_ids_with_active_experts(
    shape: torch.Size, active_experts: int
) -> torch.Tensor:
    """Create a deterministic expert assignment with a controlled load spectrum."""
    assignments = torch.arange(shape.numel(), dtype=torch.int32, device="cuda")
    return assignments.remainder(active_experts).view(shape)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse execution-mode and tensor-shape controls for the MVP profiler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        help="run ordinary eager autotuner dispatch without capture or replay",
    )
    parser.add_argument(
        "--num-experts", type=int, default=MockDAMoERunner.MAX_NUM_EXPERTS
    )
    return parser.parse_args(argv)


@contextlib.contextmanager
def _nvtx_phase(name: str) -> Iterator[None]:
    """Expose one end-to-end MVP lifecycle phase in CUDA profiler traces."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _replay_distribution(
    name: str,
    expert_ids: torch.Tensor,
    inputs: MockMoEInputs,
    runner: MockDAMoERunner,
    lease: DAGraphLease,
    replays: int,
    serial_inputs: Sequence[MockMoEInputs] = (),
) -> dict[str, object]:
    """Replay one live distribution and return device-observed dispatch results."""
    # Mutate only live graph inputs and clear the device observation before replay.
    inputs.expert_ids.copy_(expert_ids)
    inputs.body_trace.fill_(-1)
    for serial_input in serial_inputs:
        serial_input.expert_ids.copy_(expert_ids)
        serial_input.body_trace.fill_(-1)
    torch.cuda.synchronize()
    with _nvtx_phase(f"DA_PHASE_REPLAY_{name.upper()}"):
        for _ in range(replays):
            lease.replay()
        torch.cuda.synchronize()
    # Read both selector output and executed-body stamp after synchronized replay.
    return {
        "distribution": name,
        "selected_body": int(runner.selected_body_tensor().item()),
        "executed_tactic": int(inputs.body_trace.item()),
        "serial_executed_tactics": [
            int(serial_input.body_trace.item()) for serial_input in serial_inputs
        ],
        "replays": replays,
    }


def _run_eager_distribution(
    name: str,
    expert_ids: torch.Tensor,
    inputs: MockMoEInputs,
    runner: MockDAMoERunner,
    invocations: int,
) -> dict[str, int | str]:
    """Run one distribution through ordinary eager autotuner dispatch."""
    # Eager mode changes live input contents but launches through host AutoTuner dispatch.
    inputs.expert_ids.copy_(expert_ids)
    inputs.body_trace.fill_(-1)
    torch.cuda.synchronize()
    with _nvtx_phase(f"DA_PHASE_EAGER_{name.upper()}"):
        for _ in range(invocations):
            runner.forward(inputs.as_list())
        torch.cuda.synchronize()
    # Require and report the ordinary tactic chosen by the user-facing runner path.
    normal_tactic = runner.last_normal_tactic
    if normal_tactic is None:
        raise RuntimeError("Eager execution did not perform normal autotuner dispatch")
    return {
        "distribution": name,
        "executed_tactic": int(inputs.body_trace.item()),
        "normal_autotuner_tactic": normal_tactic,
        "invocations": invocations,
    }


def _exercise_graph_resource_lifecycle(
    runner: MockDAMoERunner,
    inputs: MockMoEInputs,
    graph: torch.cuda.CUDAGraph,
    lease: DAGraphLease,
) -> dict[str, object]:
    """Prove bounded ownership, temporary fallback, teardown, and successful retry."""
    # A second outer graph cannot reuse the live graph's lane at capacity. Its ordinary warmup and
    # capture therefore fall back without turning the lease conflict into an error.
    retry_inputs = runner.moe_runner.allocate_inputs(
        inputs.hidden_states.shape[0],
        inputs.hidden_states.shape[1],
        inputs.expert_ids.shape[1],
    )
    retry_inputs.expert_ids.copy_(inputs.expert_ids)
    runner.forward(retry_inputs.as_list())
    conflict_reason = runner.last_preparation_conflict
    counts_during_conflict = runner.resource_counts()
    fallback_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fallback_graph):
        runner.forward(retry_inputs.as_list())
    torch.cuda.synchronize()
    fallback_topology = runner.inspect_last_graph()
    fallback_graph.reset()

    # Reset the executable before releasing its lease. The lane becomes idle immediately, so the
    # same public warmup call can reuse it and capture the rejected binding.
    graph.reset()
    lease.release()
    counts_after_release = runner.resource_counts()
    runner.forward(retry_inputs.as_list())
    retry_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(retry_graph):
        runner.forward(retry_inputs.as_list())
    torch.cuda.synchronize()
    retry_topology = runner.inspect_last_graph()
    retry_lease = runner.acquire_graph_lease(retry_graph)
    retry_lease.replay()
    torch.cuda.synchronize()
    retry_trace = int(retry_inputs.body_trace.item())
    counts_after_retry = runner.resource_counts()

    # The second graph teardown makes the lane idle; explicit teardown then releases its storage.
    retry_graph.reset()
    retry_lease.release()
    released_idle = runner.release_idle_resources()
    return {
        "conflict_reason": conflict_reason,
        "counts_during_conflict": counts_during_conflict,
        "fallback_conditional_nodes": fallback_topology.conditional_node_count,
        "counts_after_release": counts_after_release,
        "retry_conditional_nodes": retry_topology.conditional_node_count,
        "retry_body_count": retry_topology.body_count,
        "retry_executed_tactic": retry_trace,
        "counts_after_retry": counts_after_retry,
        "counts_after_final_release": runner.resource_counts(),
        "released_idle_resources": released_idle,
    }


def run_mock_da_moe(args: argparse.Namespace) -> dict[str, object]:
    """Run one eager or CUDA-graph lifecycle and return its runtime observations."""
    # Build one dtype-agnostic composed runner and stable tensor bundle for all phases.
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float8_e4m3fn
    runner = MockDAMoERunner(
        num_experts=args.num_experts,
        dtype=dtype,
        tuning_config=TuningConfig(use_cuda_graph=not args.no_cuda_graph),
    )
    inputs = runner.moe_runner.allocate_inputs(
        args.num_tokens, args.hidden_size, args.top_k
    )
    uniform = _expert_ids_with_active_experts(
        inputs.expert_ids.shape, min(256, args.num_experts)
    )
    concentrated = torch.zeros_like(inputs.expert_ids)

    # Public warmup invokes ordinary forward under AutoTuner and automatically publishes the plan.
    with _nvtx_phase("DA_PHASE_AUTOTUNE_WARMUP"):
        with autotune(True):
            runner.forward(inputs.as_list())
        torch.cuda.synchronize()
    plan = runner.plan
    if plan is None:
        raise RuntimeError("Warmup did not publish a DA plan")

    exemplar_body_indices = plan.exemplar_body_indices[: plan.num_selector_exemplars]
    selected_exemplar_tactics = [
        plan.bodies[int(body_index)].tactic for body_index in exemplar_body_indices
    ]
    report: dict[str, object] = {
        "execution_mode": "eager" if args.no_cuda_graph else "cuda_graph",
        "autotune_warmup": {
            "dtype_family": runner.moe_runner.operation_family,
            "num_experts": runner.num_experts,
            "candidate_tactics": list(runner.valid_tactics),
            "exemplar_count": plan.num_selector_exemplars,
            "selected_exemplar_tactics": selected_exemplar_tactics,
            "published_body_tactics": [body.tactic for body in plan.bodies],
            "resources_prepared": runner.dispatcher.resources is not None,
            "uses_fixed_candidate_cuda_graphs": not args.no_cuda_graph,
        },
    }

    # The opt-in eager path proves ordinary host dispatch without creating DA graph resources.
    if args.no_cuda_graph:
        report["phase_sequence"] = ["autotune_warmup", "eager"]
        report["eager"] = {
            "observations": [
                _run_eager_distribution(
                    "uniform", uniform, inputs, runner, args.replays
                ),
                _run_eager_distribution(
                    "concentrated", concentrated, inputs, runner, args.replays
                ),
            ]
        }
        return report

    # Warm sixty exact bindings, then capture every serial layer into one shared workspace lane.
    serial_inputs = [
        runner.moe_runner.allocate_inputs(args.num_tokens, args.hidden_size, args.top_k)
        for _ in range(59)
    ]
    for serial_input in serial_inputs:
        serial_input.expert_ids.copy_(inputs.expert_ids)
        runner.forward(serial_input.as_list())
    with _nvtx_phase("DA_PHASE_CAPTURE"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.forward(inputs.as_list())
            for serial_input in serial_inputs:
                runner.forward(serial_input.as_list())
        torch.cuda.synchronize()
    topology = runner.inspect_last_graph()
    workspace_bindings = runner.inspect_last_workspace_bindings()
    lease = runner.acquire_graph_lease(graph)
    report["phase_sequence"] = ["autotune_warmup", "capture", "replay"]
    report["capture"] = {
        "workspace_bindings": workspace_bindings,
        "topology": {
            "capture_id": topology.capture_id,
            "outer_node_count": topology.outer_node_count,
            "outer_edge_count": topology.outer_edge_count,
            "conditional_node_count": topology.conditional_node_count,
            "body_count": topology.body_count,
            "selector_dependency_count": topology.selector_dependency_count,
            "parallel_work_dependency_count": topology.parallel_work_dependency_count,
            "is_selector_preamble_parallelizable": topology.is_selector_preamble_parallelizable,
            "body_node_counts": topology.body_node_counts,
            "is_workspace_lane_serialized": topology.is_workspace_lane_serialized,
            "workspace_lane_invocation_count": topology.workspace_lane_invocation_count,
        },
    }
    report["replay"] = {
        "observations": [
            _replay_distribution(
                "uniform", uniform, inputs, runner, lease, args.replays, serial_inputs
            ),
            _replay_distribution(
                "concentrated",
                concentrated,
                inputs,
                runner,
                lease,
                args.replays,
                serial_inputs,
            ),
        ]
    }
    report["resource_lifecycle"] = _exercise_graph_resource_lifecycle(
        runner, inputs, graph, lease
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run the reference design through its command-line-compatible interface."""
    report = run_mock_da_moe(_parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))


def test_mock_da_moe_cuda_graph_reference_design() -> None:
    """One public argument set must prove tuning, SWITCH capture, and live replay."""
    report = run_mock_da_moe(
        _parse_args(
            (
                "--replays",
                "2",
                "--num-tokens",
                "64",
                "--hidden-size",
                "64",
                "--top-k",
                "4",
                "--num-experts",
                "64",
                "--dtype",
                "bf16",
            )
        )
    )

    assert report["phase_sequence"] == ["autotune_warmup", "capture", "replay"]
    capture = report["capture"]
    assert isinstance(capture, dict)
    topology = capture["topology"]
    assert isinstance(topology, dict)
    assert topology["conditional_node_count"] == 60
    assert topology["body_count"] == 2
    assert topology["is_selector_preamble_parallelizable"] is True
    assert topology["is_workspace_lane_serialized"] is True
    assert topology["workspace_lane_invocation_count"] == 60

    workspace_bindings = capture["workspace_bindings"]
    assert isinstance(workspace_bindings, tuple)
    assert len(workspace_bindings) == 60
    weight_pointers = {binding[0] for binding in workspace_bindings}
    activation_pointers = {binding[1] for binding in workspace_bindings}
    intermediate_pointers = {binding[2] for binding in workspace_bindings}
    assert len(weight_pointers) == 60
    assert len(activation_pointers) == 1
    assert len(intermediate_pointers) == 1
    assert activation_pointers.isdisjoint(intermediate_pointers)

    replay = report["replay"]
    assert isinstance(replay, dict)
    observations = replay["observations"]
    assert isinstance(observations, list)
    assert {row["distribution"] for row in observations} == {
        "uniform",
        "concentrated",
    }
    assert {row["selected_body"] for row in observations} == {0, 1}
    assert {row["executed_tactic"] for row in observations} == {0, 1}
    assert all(
        set(row["serial_executed_tactics"]) == {row["executed_tactic"]}
        for row in observations
    )

    lifecycle = report["resource_lifecycle"]
    assert isinstance(lifecycle, dict)
    assert "live CUDA Graph leases" in str(lifecycle["conflict_reason"])
    assert lifecycle["counts_during_conflict"] == {
        "capacity": 1,
        "binding_records": 61,
        "workspace_lanes": 1,
        "leased_workspace_lanes": 1,
    }
    assert lifecycle["fallback_conditional_nodes"] == 0
    assert lifecycle["counts_after_release"] == {
        "capacity": 1,
        "binding_records": 61,
        "workspace_lanes": 1,
        "leased_workspace_lanes": 0,
    }
    assert lifecycle["retry_conditional_nodes"] == 1
    assert lifecycle["retry_body_count"] == 2
    assert lifecycle["retry_executed_tactic"] in {0, 1}
    assert lifecycle["counts_after_retry"] == {
        "capacity": 1,
        "binding_records": 61,
        "workspace_lanes": 1,
        "leased_workspace_lanes": 1,
    }
    assert lifecycle["counts_after_final_release"] == {
        "capacity": 1,
        "binding_records": 61,
        "workspace_lanes": 0,
        "leased_workspace_lanes": 0,
    }
    assert lifecycle["released_idle_resources"] == 1


if __name__ == "__main__":
    main()
