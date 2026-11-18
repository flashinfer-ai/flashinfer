"""Released NCCL EP configuration, graph replay and mask API checks."""

import gc
import json
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version

from flashinfer.moe_ep import (
    BootstrapConfig,
    EpAlgorithm,
    EpLayout,
    FleetAlgoKnobFaultTolerance,
    FleetAlgoKnobNcclEpConfig,
    FleetParams,
    MoEEpSplitLayer,
    MoEEpTensors,
    dummy_moe_weights,
)

pytestmark = [pytest.mark.nvep, pytest.mark.gpu_4]


def _assert_collective(condition, message):
    failed = torch.tensor(int(not condition), device="cuda", dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    assert not failed.item(), message


def _assert_collective_close(actual, expected):
    failure = None
    try:
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    except AssertionError as error:
        failure = str(error)
    _assert_collective(failure is None, failure or "reference failed on another rank")


def _identity_expected(inputs, *, weighted_combine):
    active = inputs.topk_ids >= 0
    if weighted_combine:
        gain = inputs.topk_weights.masked_fill(~active, 0).sum(dim=1)
    else:
        # These routes visit distinct owners. Flat and Rank-Major combine one
        # unweighted identity result per owner; real expert compute weights it.
        gain = active.sum(dim=1)
    return (inputs.hidden_states.float() * gain[:, None]).to(inputs.hidden_states.dtype)


def test_collective_assertion_reports_remote_failure(released_runtime):
    rank, _ = released_runtime
    with pytest.raises(AssertionError, match="injected rank-zero failure"):
        _assert_collective(rank != 0, "injected rank-zero failure")
    dist.barrier()


@pytest.fixture(scope="module")
def released_runtime():
    if int(os.environ.get("WORLD_SIZE", "1")) < 4 or not torch.cuda.is_available():
        pytest.skip("requires torchrun with at least four GPUs")
    ep = pytest.importorskip("nccl.ep")
    if ep.get_lib_version() < Version("0.3.0"):
        pytest.skip("requires nccl-extensions 0.2 (EP 0.3)")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", torch.cuda.current_device()),
            timeout=timedelta(minutes=60),
        )
    # The shared session hook owns teardown; NCCL cannot be reinitialized
    # between test modules in the same workers.
    yield dist.get_rank(), dist.get_world_size()


def _layer_and_inputs(
    rank,
    world,
    hidden,
    algorithm,
    *,
    policy=None,
    fault_tolerance=False,
    tokens=16,
    flow=None,
    eager_recv=False,
    capacity=16,
):
    knobs = [
        FleetAlgoKnobNcclEpConfig(
            num_topk=2,
            max_num_sms=8,
            dispatch_num_sms=4,
            combine_num_sms=4,
            ht_flow=flow,
            meta_exchange_policy=policy,
            eager_recv=eager_recv,
        )
    ]
    if fault_tolerance:
        knobs.append(FleetAlgoKnobFaultTolerance())
    layer = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(
            rank=rank, world_size=world, stream=torch.cuda.current_stream().cuda_stream
        ),
        fleet_params=FleetParams(
            num_experts=world * 2,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
            algorithm=algorithm,
            layout=EpLayout.RANK_MAJOR if fault_tolerance else EpLayout.EXPERT_MAJOR,
        ),
        weights=dummy_moe_weights(num_local_experts=2, hidden=hidden),
        fleet_knobs=knobs,
        backend="nccl_ep",
    )
    own = torch.arange(tokens, device="cuda", dtype=torch.int64) % 2 + 2 * rank
    peer = own % 2 + 2 * ((rank + 1) % world)
    ids = torch.stack((own, peer), dim=1)
    inputs = MoEEpTensors(
        hidden_states=torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16),
        topk_ids=ids,
        topk_weights=torch.full((tokens, 2), 0.5, device="cuda"),
    )
    return layer, inputs


@pytest.mark.parametrize("flow", ["push_pull", "pull_push"])
def test_release_ht_eager_receive_arrays(released_runtime, flow):
    rank, world = released_runtime
    layer, inputs = _layer_and_inputs(
        rank,
        world,
        4096,
        EpAlgorithm.HIGH_THROUGHPUT,
        tokens=8,
        capacity=128,
        flow=flow,
        eager_recv=True,
    )
    graph = None
    try:
        state = layer.create_graph_state(inputs)
        for owner in (None, 0, 1):
            if owner is not None:
                inputs.topk_ids[:, 0] = owner * 2
                inputs.topk_ids[:, 1] = -1
            output = layer.forward(inputs, graph_state=state)
            torch.cuda.synchronize()
            _assert_collective_close(
                output, _identity_expected(inputs, weighted_combine=False)
            )
            received = 16 if owner is None else (8 * world if rank == owner else 0)
            handle = state._handle
            _assert_collective(
                handle._dispatch_output_t.shape == (max(1, received), 4096),
                "receive-array shape mismatch",
            )
            _assert_collective(
                int(handle._recv_total_target.item()) == received,
                "received-token counter mismatch",
            )
            _assert_collective(
                received != 0 or handle._dispatch_out_idx.eq(-1).all().item(),
                "empty receive row contains active expert IDs",
            )
        allocated = []
        for _ in range(3):
            cycle = []
            for owner in range(world):
                inputs.topk_ids[:, 0] = owner * 2
                output = layer.forward(inputs, graph_state=state)
                _assert_collective_close(
                    output, _identity_expected(inputs, weighted_combine=False)
                )
                torch.cuda.synchronize()
                gc.collect()
                cycle.append(torch.cuda.memory_allocated())
            allocated.append(cycle)
        _assert_collective(
            max(allocated[2]) <= max(allocated[1]) + 64 * 1024,
            "receive-array retention grew after warmup",
        )
        print(
            "EAGER_RECV_RETENTION "
            + json.dumps(dict(rank=rank, flow=flow, allocated_bytes=allocated))
        )
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(ValueError, match="eager_recv.*capture"),
            torch.cuda.graph(graph, stream=torch.cuda.Stream()),
        ):
            layer.forward(inputs, graph_state=state)
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()


@pytest.mark.parametrize(
    "hidden,policy,tokens,flow",
    [(hidden, None, tokens, None) for hidden in (3072, 3584) for tokens in (16, 1, 8)]
    + [
        (4096, policy, 16, flow)
        for policy in ("auto", "force_scan", "require_fused")
        for flow in (None, "push_pull", "pull_push")
    ],
)
def test_release_forward_graph(released_runtime, hidden, policy, tokens, flow):
    rank, world = released_runtime
    algorithm = (
        EpAlgorithm.LOW_LATENCY if policy is None else EpAlgorithm.HIGH_THROUGHPUT
    )
    layer, inputs = _layer_and_inputs(
        rank, world, hidden, algorithm, policy=policy, tokens=tokens, flow=flow
    )
    graph = None
    try:
        layer.forward(inputs)
        state = layer.create_graph_state(inputs)
        layer.forward(inputs, graph_state=state)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            output = layer.forward(inputs, graph_state=state)
        routes = inputs.topk_ids.clone()
        for iteration in range(3):
            inputs.hidden_states.mul_(0.5)
            inputs.topk_ids.copy_(routes)
            if iteration == 1:
                inputs.topk_ids[::2, 1] = -1
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            expected = _identity_expected(
                inputs, weighted_combine=algorithm is EpAlgorithm.LOW_LATENCY
            )
            _assert_collective_close(output, expected)
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()


def test_release_mask_api_and_readmission(released_runtime):
    rank, world = released_runtime
    layer, inputs = _layer_and_inputs(
        rank, world, 4096, EpAlgorithm.LOW_LATENCY, fault_tolerance=True
    )
    try:
        layer.forward(inputs)
        fleet = layer._ensure_fleet()
        mixed = [1] * world
        mixed[(rank + 1) % world] = 0
        fleet.set_active_mask(mixed)
        observed_mask = fleet.query_active_mask().cpu().tolist()
        # Restore before collective work: each rank used a different local mask.
        fleet.set_active_mask([1] * world)
        fleet.set_active_mask([1] * world)
        _assert_collective(observed_mask == mixed, "mask query did not match update")
        _assert_collective(
            fleet.query_active_mask().cpu().tolist() == [1] * world,
            "mask restoration failed",
        )
        _assert_collective(not fleet.query_fault(), "unexpected fault flag")
        dist.barrier()
        fleet.clear_faults(readmit=True)
        output = layer.forward(inputs)
        torch.cuda.synchronize()
        _assert_collective_close(
            output, _identity_expected(inputs, weighted_combine=False)
        )
    finally:
        layer.destroy()
