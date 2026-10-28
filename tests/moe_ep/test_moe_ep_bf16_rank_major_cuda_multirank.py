"""Real BF16 rank-major MegaMoE regression tests; minimum arch SM100 (B200).

Run on one x86_64 node with eight B200 GPUs::

    bash tests/moe_ep/run_tests.sh bf16-rank-major

The layer tests launch the backend only through the public layer API. Their
reference uses canonical weights, PyTorch matrix products and collectives, not
the backend's generated source, transformed layouts, launch session, or another
kernel implementation. The capacity is fixed at W8/T128/H7168/I2048/E256/K8;
active T is shared by ranks.

The flag-in-data (LL) tests at the end drive the launch session directly for
the epoch wrap and the staging lifecycle of up to 32 active rows per rank; see
the comment above them.
"""

from __future__ import annotations

import os
import platform
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
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
from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_rank_major_cuda.weights import (
    preprocess_mega_weights,
)
from flashinfer.moe_ep.kernel_src.blackwell_bf16_rank_major.session import (
    BlackwellBf16RankMajorSession,
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
    # Use the validated peer-memory transport for this eight-rank backend.
    symm_mem.set_backend("NVSHMEM")
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    assert dist.get_world_size() == _WORLD
    assert dist.get_rank() == rank

    # The independent reference must use FP32 accumulation, not TF32 products.
    old_tf32_override = os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
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
        if old_tf32_override is not None:
            os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = old_tf32_override
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
        # int64 like all_ranks: the layer's steady-state input signature includes the id dtype.
        ids = (owner * _LOCAL_EXPERTS + local_expert).to(torch.int64)
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


# Up to 32 rows per rank the flag-in-data path runs: 1, 2, 3-4 and 5-32 rows
# use 8, 4, 2 and 1 row splits; 33 rows use the pull path.
@pytest.mark.parametrize(
    "tokens", (1, 2, 3, 4, 5, 8, 16, 31, 32, 33, 63, 64, 65, 127, 128)
)
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


@pytest.mark.parametrize("tokens", (1, 8, 32, 127, 128))
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


# Flag-in-data (LL) path: up to 32 active rows per rank, the session moves rows
# and partials as LL words that carry a 32-bit forward epoch, in two banks
# selected by the epoch parity, and stage_inputs already sends the rows.  These
# tests drive the launch session directly and compare every output, bit for
# bit, with a separate reference session that never reaches the epoch wrap.  In
# the forwards that read old words, half of the ranks delay their sends, so the
# other ranks read the slots before the new words arrive.
#
# The wrap tests write every dispatch and combine slot of both banks with three
# 32-row forwards (epochs 1 to 3: bank 0 keeps epoch-2 words, bank 1 epoch-3
# words), move the session epoch next to the wrap with the session's test hook,
# and run one-row forwards up to the wrap whose routes all stay on the sending
# rank, so the slots of tokens 1-31 keep the old words, headers included.  The
# 32-row forwards after the wrap use epochs 2 and 3 and read these slots: an old
# word passes as new unless the wrap cleared both banks.

_LL_NEAR_WRAP = 0xFFFFFFFA
# GPU clock cycles a delayed rank waits before its sends (about half a millisecond).
_LL_DELAY_CYCLES = 1_000_000


def _ll_new_session(rank: int) -> BlackwellBf16RankMajorSession:
    return BlackwellBf16RankMajorSession(
        dist.group.WORLD,
        rank,
        _WORLD,
        _CAPACITY,
        _HIDDEN,
        _INTERMEDIATE,
        _EXPERTS,
        _TOP_K,
    )


def _ll_bind(s: BlackwellBf16RankMajorSession, weights) -> None:
    s.bind_weights(weights.w13_block_major, weights.w2_block_major)


@pytest.fixture(scope="module")
def ll_env(problem):
    _layer, weights, rank = problem
    transformed = preprocess_mega_weights(
        weights,
        intermediate_size=_INTERMEDIATE,
        hidden_size=_HIDDEN,
        num_local_experts=_LOCAL_EXPERTS,
    )
    reference = _ll_new_session(rank)
    _ll_bind(reference, transformed)
    dist.barrier()
    try:
        yield rank, transformed, reference
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        reference.destroy()


@pytest.fixture
def ll_session(ll_env):
    """A fresh session (epoch 0, zeroed LL buffers) without bound weights."""
    rank, _weights, _reference = ll_env
    s = _ll_new_session(rank)
    dist.barrier()
    try:
        yield s
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        s.destroy()


def _ll_inputs(rank: int, tokens: int, routing: str, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed + 1009 * rank)
    hidden = torch.randn(
        tokens, _HIDDEN, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    token = torch.arange(tokens, device="cuda")[:, None]
    slot = torch.arange(_TOP_K, device="cuda")[None, :]
    if routing == "all_ranks":
        # every token has one route on every rank
        owner = (slot + rank + seed) % _WORLD
        local_expert = (token * 7 + slot * 3 + rank + seed) % _LOCAL_EXPERTS
        ids = (owner * _LOCAL_EXPERTS + local_expert).to(torch.int64)
    elif routing == "local":
        # every route stays on the sending rank: no row or partial goes to a peer
        local_expert = (slot * 4 + token + seed) % _LOCAL_EXPERTS
        ids = (rank * _LOCAL_EXPERTS + local_expert).to(torch.int64)
    else:
        assert routing == "random"
        scores = torch.randn(
            tokens, _EXPERTS, dtype=torch.float32, device="cuda", generator=generator
        )
        ids = scores.topk(_TOP_K, dim=-1).indices.to(torch.int64)
    weights = torch.softmax(
        torch.randn(
            tokens, _TOP_K, dtype=torch.float32, device="cuda", generator=generator
        ),
        dim=-1,
    )
    return hidden, ids.contiguous(), weights.contiguous()


def _ll_delay(rank: int, step: int) -> None:
    """Delay this rank's next launches on half of the ranks (alternating per step)."""
    if (rank + step) % 2 == 0:
        torch.cuda._sleep(_LL_DELAY_CYCLES)


def _ll_settle() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _ll_output(tokens: int) -> torch.Tensor:
    return torch.empty(tokens, _HIDDEN, dtype=torch.bfloat16, device="cuda")


def _ll_forward(s, x, *, delay_step: int | None = None) -> torch.Tensor:
    out = _ll_output(x[0].shape[0])
    if delay_step is not None:
        _ll_delay(dist.get_rank(), delay_step)
    s.stage_inputs(*x)
    s.run(out)
    _ll_settle()
    return out


def _ll_assert_all(condition: bool, label: str) -> None:
    flags = [None] * _WORLD
    dist.all_gather_object(flags, bool(condition))
    assert all(flags), (
        f"{label}: ranks failing {[i for i, f in enumerate(flags) if not f]}"
    )


def _ll_assert_bitwise(actual, expected, label):
    same = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    _ll_assert_all(same, label)


def _ll_check(s, reference, x, label, *, delay_step: int | None = None):
    expected = _ll_forward(reference, x)
    _ll_assert_bitwise(_ll_forward(s, x, delay_step=delay_step), expected, label)


def _ll_fill_both_banks(s, reference, rank: int) -> None:
    """Epochs 1-3, 32 rows, every token routed to every rank: every slot is written."""
    for seed in (590, 591, 592):
        _ll_check(s, reference, _ll_inputs(rank, 32, "all_ranks", seed), f"fill/{seed}")


def _ll_last_epoch(s) -> int:
    return int(s._ll_control[0].item()) & 0xFFFFFFFF


# Kernels of one forward, in launch order (the pull path's input stage first).
_LL_FORWARD_KERNELS = (
    "ll_push",
    "ll_recv",
    "gemm1_swiglu",
    "gemm2",
    "ll_unpermute",
    "ll_combine",
)
_PULL_FORWARD_KERNELS = (
    "stage_inputs",
    "input_arrive",
    "dispatch",
    "gemm1_swiglu",
    "gemm2",
    "local_unpermute",
    "partial_arrive",
    "combine",
)


def _ll_record_launches(s, fail_at: str | None = None) -> list[str]:
    """Record the kernels the session launches from now on.

    With ``fail_at``, the next launch of that kernel raises instead, once, and
    launches nothing.
    """
    library = s._library
    launch = type(library).launch
    names: list[str] = []
    armed = [fail_at]

    def record(stage, *args, **kwargs):
        if stage["name"] == armed[0]:
            armed[0] = None
            raise RuntimeError(f"injected launch failure at {stage['name']}")
        launch(library, stage, *args, **kwargs)
        names.append(stage["name"])

    library.launch = record
    return names


# Forwards after the wrap: (tokens, routing); the first two use epochs 2 and 3
# with 32 rows and read every slot of bank 0 and bank 1.
_LL_AFTER_WRAP = (
    (32, "all_ranks"),
    (32, "all_ranks"),
    (32, "random"),
    (1, "random"),
    (1, "all_ranks"),
    (32, "random"),
)


def test_ll_forwards_across_the_epoch_wrap_are_bitwise_stable(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    _ll_fill_both_banks(s, reference, rank)
    s._ll_set_epoch_for_test(_LL_NEAR_WRAP)
    # epochs 0xfffffffb..0xffffffff: one row, every route on the sending rank;
    # of the slots other ranks read, only the token-0 headers change
    for i in range(5):
        x = _ll_inputs(rank, 1, "local", 601 + i)
        _ll_check(s, reference, x, f"eager/before-wrap/{i}")
    for i, (tokens, routing) in enumerate(_LL_AFTER_WRAP):
        x = _ll_inputs(rank, tokens, routing, 611 + i)
        _ll_check(
            s,
            reference,
            x,
            f"eager/after-wrap/epoch{2 + i}/{routing}/T{tokens}",
            delay_step=i,
        )
    last = _ll_last_epoch(s)
    assert last == 1 + len(_LL_AFTER_WRAP), f"unexpected last epoch 0x{last:08x}"


def test_ll_graph_replay_across_the_epoch_wrap_is_bitwise_stable(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    _ll_fill_both_banks(s, reference, rank)
    # one graph per active count: the one-row graph runs the forwards up to the
    # wrap, so the 32-row graph finds old words behind old headers after it
    graphs = {}
    for tokens in (1, 32):
        static = _ll_inputs(rank, tokens, "local", 700 + tokens)
        out = _ll_output(tokens)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s.stage_inputs(*static)
            s.run(out)
        graphs[tokens] = graph, static, out
    dist.barrier()
    s._ll_set_epoch_for_test(_LL_NEAR_WRAP)
    cases = [(1, "local", None)] * 5 + [
        (tokens, routing, i) for i, (tokens, routing) in enumerate(_LL_AFTER_WRAP)
    ]
    for i, (tokens, routing, delay_step) in enumerate(cases):
        graph, static, out = graphs[tokens]
        x = _ll_inputs(rank, tokens, routing, 710 + i)
        expected = _ll_forward(reference, x)
        for buffer, value in zip(static, x, strict=True):
            buffer.copy_(value)
        if delay_step is not None:
            _ll_delay(rank, delay_step)
        graph.replay()
        _ll_settle()
        _ll_assert_bitwise(out, expected, f"graph/{i}/{routing}/T{tokens}")
    last = _ll_last_epoch(s)
    assert last == 1 + len(_LL_AFTER_WRAP), f"unexpected last epoch 0x{last:08x}"


def test_restaged_ll_forward_uses_the_last_stage(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    x1 = _ll_inputs(rank, 8, "all_ranks", 801)
    x2 = _ll_inputs(rank, 8, "all_ranks", 802)
    expected = _ll_forward(reference, x2)
    out = _ll_output(8)
    _ll_delay(rank, 0)
    s.stage_inputs(*x1)
    s.stage_inputs(*x2)
    s.run(out)
    _ll_settle()
    _ll_assert_bitwise(out, expected, "restage T8")
    # mixed active counts, with one more restage between differently sized stages
    for i, (tokens, routing) in enumerate(
        ((1, "random"), (32, "all_ranks"), (33, "random"), (128, "all_ranks"))
    ):
        x = _ll_inputs(rank, tokens, routing, 810 + i)
        _ll_check(s, reference, x, f"T{tokens}")
    x3 = _ll_inputs(rank, 32, "random", 820)
    x4 = _ll_inputs(rank, 1, "all_ranks", 821)
    expected = _ll_forward(reference, x4)
    out = _ll_output(1)
    _ll_delay(rank, 1)
    s.stage_inputs(*x3)
    s.stage_inputs(*x4)
    s.run(out)
    _ll_settle()
    _ll_assert_bitwise(out, expected, "restage T32 -> T1")
    for i, tokens in enumerate((2, 8, 64)):
        x = _ll_inputs(rank, tokens, "random", 830 + i)
        _ll_check(s, reference, x, f"after restage T{tokens}", delay_step=i)


def test_rejected_run_completes_the_staged_ll_forward(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    x = _ll_inputs(rank, 8, "random", 901)
    expected = _ll_forward(reference, x)
    _ll_delay(rank, 0)
    s.stage_inputs(*x)
    with pytest.raises(ValueError, match="output must be"):
        s.run(_ll_output(9))
    _ll_settle()
    # run() completed the staged forward before raising: one epoch, its output
    # in the scratch rows, and nothing left to run
    _ll_assert_all(_ll_last_epoch(s) == 1, "epoch after the rejected run")
    _ll_assert_bitwise(s._scratch_output[:8], expected, "completed forward")
    with pytest.raises(RuntimeError, match="not staged"):
        s.run(_ll_output(8))
    # only rank 0 rejects its output; the other ranks' forwards need its partials
    x = _ll_inputs(rank, 8, "all_ranks", 902)
    expected = _ll_forward(reference, x)
    _ll_delay(rank, 1)
    s.stage_inputs(*x)
    if rank == 0:
        with pytest.raises(ValueError, match="output must be"):
            s.run(_ll_output(9))
        actual = s._scratch_output[:8]
    else:
        actual = s.run(_ll_output(8))
    _ll_settle()
    _ll_assert_bitwise(actual, expected, "rank 0 rejects")
    for i, (tokens, routing) in enumerate(
        ((8, "random"), (64, "random"), (1, "all_ranks"), (32, "random"))
    ):
        x = _ll_inputs(rank, tokens, routing, 910 + i)
        _ll_check(s, reference, x, f"after rejected run T{tokens}", delay_step=i)


def test_ll_stage_before_bound_weights_is_sent_by_run(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    names = _ll_record_launches(s)
    # without bound weights stage_inputs launches nothing and keeps a copy of
    # the inputs: a second stage replaces it, and run() sends it after its checks
    s.stage_inputs(*_ll_inputs(rank, 8, "random", 1001))
    x = _ll_inputs(rank, 8, "all_ranks", 1002)
    expected = _ll_forward(reference, x)
    s.stage_inputs(*x)
    for buffer in x:
        buffer.zero_()
    out = _ll_output(8)
    with pytest.raises(RuntimeError, match="weights are not bound"):
        s.run(out)
    _ll_settle()
    # every rank's headers go to every rank: one early push would show here
    _ll_assert_all(
        not names and torch.count_nonzero(s._ll_dispatch.local).item() == 0,
        "nothing sent before run() with bound weights",
    )
    _ll_bind(s, weights)
    _ll_delay(rank, 0)
    s.run(out)
    _ll_settle()
    _ll_assert_all(names == list(_LL_FORWARD_KERNELS), "one push, sent by run()")
    _ll_assert_bitwise(out, expected, "first forward sent by run()")
    _ll_assert_all(_ll_last_epoch(s) == 1, "LL epoch after the first forward")
    x = _ll_inputs(rank, 8, "random", 1003)
    _ll_check(s, reference, x, "next forward", delay_step=1)
    # in another fresh session only rank 0's run() rejects its output: it still
    # sends the rows the other ranks' forwards need and completes the forward
    other = _ll_new_session(rank)
    try:
        x = _ll_inputs(rank, 32, "random", 1004)
        expected = _ll_forward(reference, x)
        other.stage_inputs(*x)
        _ll_bind(other, weights)
        _ll_delay(rank, 2)
        if rank == 0:
            with pytest.raises(ValueError, match="output must be"):
                other.run(_ll_output(31))
            actual = other._scratch_output[:32]
        else:
            actual = other.run(_ll_output(32))
        _ll_settle()
        _ll_assert_bitwise(actual, expected, "deferred stage, rank 0 rejects")
        x = _ll_inputs(rank, 32, "all_ranks", 1005)
        _ll_check(other, reference, x, "after the rejected run", delay_step=3)
    finally:
        _ll_settle()
        other.destroy()


def test_failed_first_compute_after_an_ll_stage_keeps_the_session_usable(
    ll_env, ll_session
):
    rank, weights, reference = ll_env
    s = ll_session
    # a layer's first forward stages 8 rows and its compute raises before it
    # binds the weights (for example, without an owned output); the warm-up and
    # the forwards after it stage again, bind the weights and run
    s.stage_inputs(*_ll_inputs(rank, 8, "random", 1050))
    for i, (tokens, routing) in enumerate(
        ((128, "random"), (8, "all_ranks"), (32, "random"), (1, "all_ranks"))
    ):
        x = _ll_inputs(rank, tokens, routing, 1060 + i)
        expected = _ll_forward(reference, x)
        out = _ll_output(tokens)
        _ll_delay(rank, i)
        s.stage_inputs(*x)
        _ll_bind(s, weights)
        s.run(out)
        _ll_settle()
        _ll_assert_bitwise(out, expected, f"forward {i} after the failed compute")


def test_pending_ll_stage_then_pull_stage(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    x2 = _ll_inputs(rank, 64, "random", 1102)
    expected = _ll_forward(reference, x2)
    out = _ll_output(64)
    _ll_delay(rank, 0)
    s.stage_inputs(*_ll_inputs(rank, 8, "all_ranks", 1101))
    s.stage_inputs(*x2)
    s.run(out)
    _ll_settle()
    _ll_assert_bitwise(out, expected, "LL stage then pull stage")
    for i, tokens in enumerate((8, 1, 128)):
        x = _ll_inputs(rank, tokens, "random", 1110 + i)
        _ll_check(s, reference, x, f"after T{tokens}", delay_step=i)


def test_restaged_ll_forward_in_cuda_graph(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    _ll_check(s, reference, _ll_inputs(rank, 8, "random", 1200), "warm-up")
    first = _ll_inputs(rank, 8, "all_ranks", 1201)
    small = _ll_inputs(rank, 8, "all_ranks", 1202)
    large = _ll_inputs(rank, 64, "random", 1203)
    out_small = _ll_output(8)
    out_large = _ll_output(64)
    restage_small = torch.cuda.CUDAGraph()
    with torch.cuda.graph(restage_small):
        s.stage_inputs(*first)
        s.stage_inputs(*small)
        s.run(out_small)
    restage_large = torch.cuda.CUDAGraph()
    with torch.cuda.graph(restage_large):
        s.stage_inputs(*first)
        s.stage_inputs(*large)
        s.run(out_large)
    dist.barrier()
    for i in range(4):
        x_first = _ll_inputs(rank, 8, "all_ranks", 1210 + 3 * i)
        x_small = _ll_inputs(rank, 8, ("all_ranks", "random")[i % 2], 1211 + 3 * i)
        x_large = _ll_inputs(rank, 64, "random", 1212 + 3 * i)
        expected_small = _ll_forward(reference, x_small)
        expected_large = _ll_forward(reference, x_large)
        for buffer, value in zip(
            first + small + large, x_first + x_small + x_large, strict=True
        ):
            buffer.copy_(value)
        _ll_delay(rank, i)
        restage_small.replay()
        _ll_settle()
        _ll_assert_bitwise(out_small, expected_small, f"graph restage T8 {i}")
        _ll_delay(rank, i + 1)
        restage_large.replay()
        _ll_settle()
        _ll_assert_bitwise(out_large, expected_large, f"graph restage T8 -> T64 {i}")
    # an eager staged forward is neither completed nor run inside a capture
    x = _ll_inputs(rank, 8, "random", 1250)
    expected = _ll_forward(reference, x)
    s.stage_inputs(*x)
    out = _ll_output(8)
    for attempt in (lambda: s.stage_inputs(*small), lambda: s.run(out)):
        capture = torch.cuda.CUDAGraph()
        with (
            pytest.raises(RuntimeError, match="outside the active CUDA graph capture"),
            torch.cuda.graph(capture),
        ):
            attempt()
    s.run(out)
    _ll_settle()
    _ll_assert_bitwise(out, expected, "eager stage completed after refused captures")
    x = _ll_inputs(rank, 32, "random", 1251)
    _ll_check(s, reference, x, "after capture refusal")


def test_ll_stage_of_an_abandoned_capture_is_dropped(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    _ll_check(s, reference, _ll_inputs(rank, 8, "random", 1300), "warm-up")
    epoch = _ll_last_epoch(s)
    # an exception inside the capture abandons a captured 32-row stage: its
    # sends never ran, so the stage is dropped and nothing completes it
    static = _ll_inputs(rank, 32, "all_ranks", 1301)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="abandon the capture"),
        torch.cuda.graph(graph),
    ):
        s.stage_inputs(*static)
        raise RuntimeError("abandon the capture")
    _ll_settle()
    with pytest.raises(RuntimeError, match="not staged"):
        s.run(_ll_output(32))
    for i, (tokens, routing) in enumerate(
        ((8, "random"), (64, "all_ranks"), (32, "all_ranks"))
    ):
        x = _ll_inputs(rank, tokens, routing, 1310 + i)
        _ll_check(s, reference, x, f"after abandoned capture T{tokens}", delay_step=i)
    _ll_assert_all(_ll_last_epoch(s) == epoch + 2, "LL epochs after the capture")
    # a weight-descriptor miss inside the capture abandons a captured one-row stage
    other_w13 = weights.w13_block_major.clone()
    other_w2 = weights.w2_block_major.clone()
    static = _ll_inputs(rank, 1, "random", 1320)
    out = _ll_output(1)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="during CUDA graph capture"),
        torch.cuda.graph(graph),
    ):
        s.stage_inputs(*static)
        s.bind_weights(other_w13, other_w2)
        s.run(out)
    del other_w13, other_w2
    _ll_settle()
    x = _ll_inputs(rank, 1, "all_ranks", 1321)
    _ll_check(s, reference, x, "eager warm-up after the descriptor miss", delay_step=0)
    # a new capture of the same forward replays exactly
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s.stage_inputs(*static)
        s.run(out)
    dist.barrier()
    for i in range(3):
        x = _ll_inputs(rank, 1, ("random", "all_ranks")[i % 2], 1330 + i)
        expected = _ll_forward(reference, x)
        for buffer, value in zip(static, x, strict=True):
            buffer.copy_(value)
        _ll_delay(rank, i)
        graph.replay()
        _ll_settle()
        _ll_assert_bitwise(out, expected, f"recaptured graph {i}")
    x = _ll_inputs(rank, 32, "random", 1340)
    _ll_check(s, reference, x, "eager after the recaptured graph")


def test_ll_stage_of_an_abandoned_capture_is_not_completed_in_a_new_capture(
    ll_env, ll_session
):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    _ll_check(s, reference, _ll_inputs(rank, 8, "random", 1400), "eager", delay_step=0)
    # capture A records a 32-row stage and is abandoned; the next call stages
    # inside a new capture B, which must drop A's stage (its sends never ran)
    # instead of completing it
    static = _ll_inputs(rank, 32, "all_ranks", 1401)
    out = _ll_output(32)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="abandon the capture"),
        torch.cuda.graph(graph),
    ):
        s.stage_inputs(*static)
        raise RuntimeError("abandon the capture")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s.stage_inputs(*static)
        s.run(out)
    dist.barrier()
    for i in range(4):
        x = _ll_inputs(rank, 32, ("all_ranks", "random")[i % 2], 1410 + i)
        expected = _ll_forward(reference, x)
        for buffer, value in zip(static, x, strict=True):
            buffer.copy_(value)
        _ll_delay(rank, i)
        graph.replay()
        _ll_settle()
        _ll_assert_bitwise(out, expected, f"capture B replay {i}")
    _ll_assert_all(_ll_last_epoch(s) == 5, "one LL epoch per replay")
    for i, (tokens, routing) in enumerate(
        ((8, "random"), (32, "all_ranks"), (64, "random"))
    ):
        x = _ll_inputs(rank, tokens, routing, 1420 + i)
        _ll_check(s, reference, x, f"eager after capture B T{tokens}", delay_step=i)


def test_pull_stage_of_an_abandoned_capture_is_dropped(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    _ll_check(s, reference, _ll_inputs(rank, 64, "random", 1500), "eager", delay_step=0)
    # a captured 64-row stage of an abandoned capture never ran: run() must
    # report it instead of running a forward over the previous staged inputs
    static = _ll_inputs(rank, 64, "all_ranks", 1501)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="abandon the capture"),
        torch.cuda.graph(graph),
    ):
        s.stage_inputs(*static)
        raise RuntimeError("abandon the capture")
    _ll_settle()
    with pytest.raises(RuntimeError, match="not staged"):
        s.run(_ll_output(64))
    for i, (tokens, routing) in enumerate(
        ((64, "all_ranks"), (8, "random"), (128, "random"))
    ):
        x = _ll_inputs(rank, tokens, routing, 1510 + i)
        _ll_check(s, reference, x, f"eager after the dropped T{tokens}", delay_step=i)


def test_ll_and_pull_forwards_resume_after_a_failed_launch(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    # the first forward sends its deferred rows in run(), then the launch of
    # ll_recv fails on every rank; run() again launches the rest, each kernel once
    x = _ll_inputs(rank, 8, "all_ranks", 1700)
    expected = _ll_forward(reference, x)
    names = _ll_record_launches(s, fail_at="ll_recv")
    s.stage_inputs(*x)
    _ll_bind(s, weights)
    out = _ll_output(8)
    _ll_delay(rank, 0)
    with pytest.raises(RuntimeError, match="injected launch failure"):
        s.run(out)
    s.run(out)
    _ll_settle()
    _ll_assert_all(names == list(_LL_FORWARD_KERNELS), "kernels of the first forward")
    _ll_assert_bitwise(out, expected, "first forward resumed by run()")
    _ll_assert_all(_ll_last_epoch(s) == 1, "LL epoch after the first forward")
    # a later launch fails on every rank; the next stage_inputs completes the
    # started forward into the scratch output before it stages the next inputs
    for i, (tokens, routing, fail_at, kernels) in enumerate(
        (
            (32, "random", "ll_combine", _LL_FORWARD_KERNELS),
            (64, "all_ranks", "gemm2", _PULL_FORWARD_KERNELS),
        )
    ):
        x = _ll_inputs(rank, tokens, routing, 1710 + 10 * i)
        expected = _ll_forward(reference, x)
        x_next = _ll_inputs(rank, 8, "random", 1711 + 10 * i)
        expected_next = _ll_forward(reference, x_next)
        names = _ll_record_launches(s, fail_at=fail_at)
        s.stage_inputs(*x)
        _ll_delay(rank, i + 1)
        with pytest.raises(RuntimeError, match="injected launch failure"):
            s.run(_ll_output(tokens))
        s.stage_inputs(*x_next)
        _ll_settle()
        _ll_assert_all(
            names == [*kernels, "ll_push"], f"kernels of the T{tokens} forward"
        )
        _ll_assert_bitwise(
            s._scratch_output[:tokens], expected, f"completed T{tokens} forward"
        )
        out = s.run(_ll_output(8))
        _ll_settle()
        _ll_assert_bitwise(out, expected_next, f"forward after the T{tokens} failure")
    x = _ll_inputs(rank, 32, "all_ranks", 1730)
    _ll_check(s, reference, x, "after the failed launches", delay_step=3)


def test_rejected_run_completes_a_pull_forward_after_a_failed_launch(
    ll_env, ll_session
):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    # the gemm2 launch of a 64-row forward fails on every rank; then run()
    # rejects its output on every rank, and next on rank 0 only: a rejected
    # run() completes the started forward, and the other ranks resume it
    for i, rejecting in enumerate((range(_WORLD), (0,))):
        x = _ll_inputs(rank, 64, ("random", "all_ranks")[i], 1820 + i)
        expected = _ll_forward(reference, x)
        names = _ll_record_launches(s, fail_at="gemm2")
        s.stage_inputs(*x)
        _ll_delay(rank, i)
        with pytest.raises(RuntimeError, match="injected launch failure"):
            s.run(_ll_output(64))
        if rank in rejecting:
            with pytest.raises(ValueError, match="output must be"):
                s.run(_ll_output(63))
            actual = s._scratch_output[:64]
        else:
            actual = s.run(_ll_output(64))
        _ll_settle()
        _ll_assert_all(names == list(_PULL_FORWARD_KERNELS), f"kernels of case {i}")
        _ll_assert_bitwise(actual, expected, f"partly launched forward, case {i}")
        with pytest.raises(RuntimeError, match="not staged"):
            s.run(_ll_output(64))
    x = _ll_inputs(rank, 8, "random", 1830)
    _ll_check(s, reference, x, "after the rejected runs", delay_step=2)


def test_pull_forward_after_a_failed_launch_is_not_resumed_in_a_capture(
    ll_env, ll_session
):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    x = _ll_inputs(rank, 64, "all_ranks", 1860)
    expected = _ll_forward(reference, x)
    out = _ll_output(64)
    names = _ll_record_launches(s, fail_at="gemm2")
    s.stage_inputs(*x)
    _ll_delay(rank, 0)
    with pytest.raises(RuntimeError, match="injected launch failure"):
        s.run(out)
    # the kernels before gemm2 ran eagerly: the rest is not recorded into a capture
    capture = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="outside the active CUDA graph capture"),
        torch.cuda.graph(capture),
    ):
        s.run(out)
    capture = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="outside the active CUDA graph capture"),
        torch.cuda.graph(capture),
    ):
        s.stage_inputs(*x)
    del capture
    s.run(out)
    _ll_settle()
    _ll_assert_all(names == list(_PULL_FORWARD_KERNELS), "kernels of the forward")
    _ll_assert_bitwise(out, expected, "eager forward resumed after the captures")
    x = _ll_inputs(rank, 8, "random", 1861)
    _ll_check(s, reference, x, "after the refused captures", delay_step=1)


def test_pull_run_after_a_failed_launch_in_a_capture_restarts(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    _ll_bind(s, weights)
    _ll_check(s, reference, _ll_inputs(rank, 64, "random", 1800), "eager", delay_step=0)
    # an eager 64-row stage, then run() inside a capture whose launch of gemm2
    # (then dispatch) fails: the kernels recorded before it never run, so calls
    # inside that capture raise and the eager run() after it starts over
    for i, fail_at in enumerate(("gemm2", "dispatch")):
        x = _ll_inputs(rank, 64, ("all_ranks", "random")[i], 1801 + i)
        expected = _ll_forward(reference, x)
        out = _ll_output(64)
        s.stage_inputs(*x)
        _ll_record_launches(s, fail_at=fail_at)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with pytest.raises(RuntimeError, match="injected launch failure"):
                s.run(out)
            with pytest.raises(RuntimeError, match="failed in this CUDA graph capture"):
                s.run(out)
            with pytest.raises(RuntimeError, match="failed in this CUDA graph capture"):
                s.stage_inputs(*x)
        del graph
        _ll_settle()
        _ll_delay(rank, i)
        s.run(out)
        _ll_settle()
        _ll_assert_bitwise(out, expected, f"eager run after the capture, {fail_at}")
    x = _ll_inputs(rank, 8, "all_ranks", 1810)
    _ll_check(s, reference, x, "after the failed captures", delay_step=2)


def test_failed_launch_in_stage_inputs_leaves_nothing_staged(ll_env, ll_session):
    rank, weights, reference = ll_env
    s = ll_session
    # a deferred 8-row stage, then a 64-row stage whose staging kernel fails to
    # launch; next a 64-row stage, then an 8-row stage whose push fails to launch
    s.stage_inputs(*_ll_inputs(rank, 8, "random", 1840))
    _ll_bind(s, weights)
    _ll_record_launches(s, fail_at="stage_inputs")
    with pytest.raises(RuntimeError, match="injected launch failure"):
        s.stage_inputs(*_ll_inputs(rank, 64, "random", 1841))
    with pytest.raises(RuntimeError, match="not staged"):
        s.run(_ll_output(8))
    s.stage_inputs(*_ll_inputs(rank, 64, "all_ranks", 1842))
    _ll_record_launches(s, fail_at="ll_push")
    with pytest.raises(RuntimeError, match="injected launch failure"):
        s.stage_inputs(*_ll_inputs(rank, 8, "random", 1843))
    with pytest.raises(RuntimeError, match="not staged"):
        s.run(_ll_output(64))
    _ll_settle()
    for i, (tokens, routing) in enumerate(((8, "all_ranks"), (64, "random"))):
        x = _ll_inputs(rank, tokens, routing, 1850 + i)
        _ll_check(s, reference, x, f"after the failed stage T{tokens}", delay_step=i)
