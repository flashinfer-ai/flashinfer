"""Real BF16 rank-major MegaMoE regression tests; minimum arch SM100 (B200).

Run on one x86_64 node with eight B200 GPUs::

    bash tests/moe_ep/run_tests.sh bf16-rank-major

Only the public layer API launches the backend. The reference uses canonical
weights, PyTorch matrix products and collectives, not the backend's generated
source, transformed layouts, launch session, or another kernel implementation.
The capacity is fixed at W8/T128/H7168/I2048/E256/K8; active T is shared by ranks.
"""

from __future__ import annotations

import os
import platform
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from flashinfer.moe_ep import (
    BootstrapConfig,
    EpAlgorithm,
    EpLayout,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpMegaLayer,
    MoEEpTensors,
    MoEWeightPack,
    Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig,
)

pytestmark = [pytest.mark.gpu_8, pytest.mark.arch_blackwell]

_WORLD = 8
_CAPACITY = 128
_HIDDEN = 7168
_INTERMEDIATE = 2048
_EXPERTS = 256
_LOCAL_EXPERTS = _EXPERTS // _WORLD
_TOP_K = 8


@pytest.fixture(scope="module")
def problem():
    if int(os.environ.get("WORLD_SIZE", "0")) != _WORLD:
        pytest.skip("requires torchrun with exactly eight ranks")
    if platform.machine() != "x86_64" or torch.cuda.device_count() != _WORLD:
        pytest.skip("requires one x86_64 node with eight visible B200 GPUs")
    for device in range(_WORLD):
        if torch.cuda.get_device_capability(device) != (
            10,
            0,
        ) or "B200" not in torch.cuda.get_device_name(device):
            pytest.skip("requires eight B200 GPUs (SM100)")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["LOCAL_WORLD_SIZE"]) == _WORLD
    assert rank == local_rank, "this test requires a single node"
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    assert dist.get_world_size() == _WORLD
    assert dist.get_rank() == rank

    # The independent reference must use FP32 accumulation, not TF32 products.
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device="cuda").manual_seed(20260910 + rank)
    w13 = torch.randn(
        _LOCAL_EXPERTS,
        2 * _INTERMEDIATE,
        _HIDDEN,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).mul_(_HIDDEN**-0.5)
    w2 = torch.randn(
        _LOCAL_EXPERTS,
        _HIDDEN,
        _INTERMEDIATE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).mul_(_INTERMEDIATE**-0.5)
    weights = MoEWeightPack(w13=w13, w2=w2)
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=_WORLD,
            rank=rank,
            device=local_rank,
            process_group=dist.group.WORLD,
            auto_bootstrap=False,
        ),
        fleet_params=FleetParams(
            num_experts=_EXPERTS,
            max_tokens_per_rank=_CAPACITY,
            token_hidden_size=_HIDDEN,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.RANK_MAJOR,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm100_Bf16_Bf16_Bf16_RankMajorCuda_MegaMoeConfig(),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    assert isinstance(layer, MoEEpMegaLayer)
    try:
        yield layer, weights, rank
    finally:
        layer.destroy()
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        # The shared conftest destroys the process group once per session.


def _inputs(rank: int, tokens: int, routing: str, seed: int) -> MoEEpTensors:
    generator = torch.Generator(device="cuda").manual_seed(seed + 1009 * rank)
    hidden_states = torch.randn(
        tokens, _HIDDEN, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    token = torch.arange(tokens, device="cuda")[:, None]
    slot = torch.arange(_TOP_K, device="cuda")[None, :]
    if routing == "all_ranks":
        # Every token visits all owners; all local experts are visited as T grows.
        owner = (slot + rank + seed) % _WORLD
        local_expert = (token * 7 + slot * 3 + rank + seed) % _LOCAL_EXPERTS
        ids = (owner * _LOCAL_EXPERTS + local_expert).to(torch.int64)
    else:
        assert routing == "one_rank"
        # All source ranks choose the same owner, leaving seven owners empty.
        # The eight distinct local experts exercise the local weighted reduction.
        owner = seed % _WORLD
        local_expert = (token * 5 + slot + rank + seed) % _LOCAL_EXPERTS
        ids = (owner * _LOCAL_EXPERTS + local_expert).to(torch.int32)
    scores = torch.rand(
        tokens, _TOP_K, dtype=torch.float32, device="cuda", generator=generator
    )
    scores = (scores + 0.1) / _TOP_K
    scores[::3, 0] = 0.0
    return MoEEpTensors(hidden_states=hidden_states, topk_ids=ids, topk_weights=scores)


def _gather(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(_WORLD)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def _reference(t: MoEEpTensors, weights: MoEWeightPack, rank: int) -> torch.Tensor:
    """SwiGLU MoE with the documented BF16 intermediate/communication wire.

    Keep FC1 gate/up accumulators in FP32 through the activation. FC2 outputs,
    routing scores and each owner's local weighted sum are rounded to BF16.
    The final owner-rank sum is FP32, rounded once to the BF16 output.
    """
    x = _gather(t.hidden_states)
    ids = _gather(t.topk_ids)
    scores = _gather(t.topk_weights).to(torch.bfloat16).float()
    routes = torch.zeros(
        x.shape[0], _TOP_K, _HIDDEN, dtype=torch.bfloat16, device=x.device
    )
    for local_expert in range(_LOCAL_EXPERTS):
        token, slot = torch.where(ids == rank * _LOCAL_EXPERTS + local_expert)
        if token.numel() == 0:
            continue
        fc1 = x[token].float() @ weights.w13[local_expert].float().T
        gate, up = fc1.split(_INTERMEDIATE, dim=-1)
        activated = (F.silu(gate) * up).to(torch.bfloat16)
        fc2 = activated.float() @ weights.w2[local_expert].float().T
        routes[token, slot] = fc2.to(torch.bfloat16)
    partial = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(_TOP_K):
        partial.add_(routes[:, slot].float() * scores[:, slot, None])
    owner_partials = _gather(partial.to(torch.bfloat16)).view(
        _WORLD, x.shape[0], _HIDDEN
    )
    result = torch.zeros_like(partial)
    for owner in range(_WORLD):
        result.add_(owner_partials[owner].float())
    tokens = t.hidden_states.shape[0]
    return result[rank * tokens : (rank + 1) * tokens].to(torch.bfloat16)


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, label: str, *, exact: bool = False
) -> None:
    # Share numerical failures before any rank can enter the next fused launch.
    # Otherwise a rank-local assertion could strand its peers at device barriers.
    error = None
    try:
        tolerance = 0.0 if exact else 1e-2
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    except AssertionError as exc:
        error = f"rank {dist.get_rank()}, {label}: {exc}"
    errors = [None] * _WORLD
    dist.all_gather_object(errors, error)
    assert not any(errors), "\n".join(e for e in errors if e is not None)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    print(f"rank={dist.get_rank()} case={label} max_abs={max_abs:.9g}", flush=True)


@pytest.mark.parametrize("tokens", (1, 63, 64, 65, 127, 128))
@pytest.mark.parametrize("routing", ("all_ranks", "one_rank"))
def test_bf16_rank_major_matches_independent_reference(problem, tokens, routing):
    layer, weights, rank = problem
    t = _inputs(rank, tokens, routing, seed=101)
    expected = _reference(t, weights, rank)
    actual = layer.forward(t)
    torch.cuda.synchronize()
    _assert_close(actual, expected, f"eager/{routing}/T{tokens}")


def test_bf16_rank_major_repeated_active_prefix_and_owned_output(problem):
    layer, weights, rank = problem
    retained = snapshot = None
    for iteration, tokens in enumerate((128, 1, 127, 64, 65, 63, 128)):
        routing = "all_ranks" if iteration % 2 == 0 else "one_rank"
        t = _inputs(rank, tokens, routing, seed=211 + iteration)
        expected = _reference(t, weights, rank)
        actual = layer.forward(t)
        torch.cuda.synchronize()
        _assert_close(actual, expected, f"repeated/{iteration}/T{tokens}")
        if retained is None:
            retained, snapshot = actual, actual.clone()
        else:
            _assert_close(retained, snapshot, "retained-owned-output", exact=True)


@pytest.mark.parametrize("tokens", (1, 127, 128))
def test_bf16_rank_major_cuda_graph_replay_with_changed_inputs(problem, tokens):
    layer, weights, rank = problem
    t = _inputs(rank, tokens, "all_ranks", seed=307)
    expected = _reference(t, weights, rank)
    layer.warmup(t)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer.forward(t)
    # Capture only records work. No rank may replay until all peers finish it.
    dist.barrier()
    for replay in range(4):
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(actual, expected, f"graph/T{tokens}/replay{replay}")

    changed = _inputs(rank, tokens, "one_rank", seed=409)
    t.hidden_states.copy_(changed.hidden_states)
    t.topk_ids.copy_(changed.topk_ids)
    t.topk_weights.copy_(changed.topk_weights)
    expected = _reference(t, weights, rank)
    for replay in range(2):
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(actual, expected, f"graph-mutated/T{tokens}/replay{replay}")
    # The layer must remain usable after replay changed its device-side epochs.
    eager = layer.forward(t)
    torch.cuda.synchronize()
    _assert_close(eager, expected, f"eager-after-graph/T{tokens}")
