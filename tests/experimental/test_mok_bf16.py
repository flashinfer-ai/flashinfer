# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU BF16 MoK kernel checks against an independent BF16-rounding reference.

Covers the fused forward/backward (ring replay, empty experts, variable
widths, CUDA Graph replay), clamped SwiGLU, FP32 weight-gradient
accumulation, GLM-5.3-Flash local expert counts, context-only recompute,
every exported scheduler layout and the epilogues, plus the host-only
communication-split tables, resolvers and configuration switches.

Backend suite: the package is imported only through
:func:`tests.test_helpers.cake_package.cake_package`; the :mod:`flashinfer.mok`
wrapper is covered by ``test_mok_bf16_api.py``.
"""

import pytest
import torch

from tests.experimental._mok_reference import require_gpu, single_rank_schedule
from tests.test_helpers.cake_package import cake_package

kernels = cake_package("cake_mok_bf16", "_kernels")
api = cake_package("cake_mok_bf16", "api")
backend = cake_package("cake_mok_bf16", "backend")
jit = cake_package("cake_mok_bf16", "jit")
MoKConfig = cake_package("cake_mok_bf16", "workspace").MoKConfig
SCHEDULER_LAYOUTS = kernels.SCHEDULER_LAYOUTS
MoKBackward = kernels.MoKBackward
MoKEpilogues = kernels.MoKEpilogues
MoKForward = kernels.MoKForward
MoKRecompute = kernels.MoKRecompute
MoKScheduler = kernels.MoKScheduler

GEOMETRIES = ((256, 256, 768), (512, 512, 768), (512, 256, 1536))


def _mlp_backward(value, gradient, weights, limit):
    gate = (value.float() @ weights[0].float().T).bfloat16()
    up = (value.float() @ weights[1].float().T).bfloat16()
    gate_f, up_f = gate.float(), up.float()
    gate_mask = up_mask = torch.ones_like(gate_f)
    if limit is not None:
        # silu(clamp(g, max=L)) * clamp(u, -L, L); clamp gradients vanish
        # outside g <= L and -L <= u <= L.
        gate_mask = (gate_f <= limit).float()
        up_mask = ((up_f >= -limit) & (up_f <= limit)).float()
        gate_f, up_f = gate_f.clamp(max=limit), up_f.clamp(-limit, limit)
    sigmoid = torch.sigmoid(gate_f)
    silu = gate_f * sigmoid
    activation = (silu * up_f).bfloat16()
    dh = (gradient.float() @ weights[2].float()).bfloat16()
    dg = (((1.0 - silu) * sigmoid + silu) * up_f * dh.float() * gate_mask).bfloat16()
    du = (silu * dh.float() * up_mask).bfloat16()
    dx = (dg.float() @ weights[0].float() + du.float() @ weights[1].float()).bfloat16()
    return dx, dg, du, dh, activation


def _check_fused_training(
    *,
    swiglu_limit=None,
    fp32_wgrad=False,
    experts=4,
    topk=2,
    tokens=512,
    geometries=GEOMETRIES,
    empty_cases=(False, True),
    mini=256,
    comm=4,
):
    clamped = swiglu_limit is not None
    forward, backward = MoKForward(clamped), MoKBackward(clamped, fp32_wgrad)
    records = []
    for empty in empty_cases:
        for hidden, intermediate, macro in geometries:
            torch.manual_seed(9173 + hidden)
            options = dict(device="cuda", dtype=torch.bfloat16)
            x = torch.randn(tokens, hidden, **options) * 0.125
            dy = torch.randn_like(x) * 0.125
            shared = [
                torch.randn(intermediate, hidden, **options) / hidden**0.5,
                torch.randn(intermediate, hidden, **options) / hidden**0.5,
                torch.randn(hidden, intermediate, **options) / intermediate**0.5,
            ]
            routed = [
                torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
                torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
                torch.randn(experts, hidden, intermediate, **options)
                / intermediate**0.5,
            ]
            scores = torch.rand(tokens, topk, device="cuda") + 0.1
            scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
            row = torch.arange(tokens, device="cuda")
            routes = torch.stack(
                [(row + slot) % (topk if empty else experts) for slot in range(topk)],
                -1,
            )
            s = single_rank_schedule(routes, experts)
            slots, count_values, actual = s["slots"], s["count_values"], s["actual"]
            combine = torch.empty(tokens * topk, hidden, **options)
            dx_peer = torch.empty(tokens * topk, hidden, **options)
            ds_peer = torch.empty_like(scores)
            context = [None]
            accumulators = initial = None
            if fp32_wgrad:
                initial = [
                    torch.randn(w.shape, device="cuda") * 0.01
                    for w in (*shared, *routed)
                ]
                accumulators = [value.clone() for value in initial]

            def sequence():
                if fp32_wgrad:
                    for accumulator, start in zip(accumulators, initial, strict=True):
                        accumulator.copy_(start)
                values = forward(
                    x, [x.data_ptr()], combine, [combine.data_ptr()],
                    shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                    s["peers"], s["schedule"], s["num_tokens"], s["counts"],
                    topk, swiglu_limit, comm, macro, mini,
                )  # fmt: skip
                context[0] = values
                return backward(
                    dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
                    scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
                    shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                    *values[:7], x, [x.data_ptr()],
                    s["peers"], s["schedule"], s["num_tokens"], s["counts"],
                    topk, swiglu_limit, comm, macro, mini,
                    weight_grad_accumulators=accumulators,
                )  # fmt: skip

            def reference():
                sx, sg, su, sh, activation = _mlp_backward(x, dy, shared, swiglu_limit)
                if fp32_wgrad:
                    dw_shared = [
                        initial[0] + sg.float().T @ x.float(),
                        initial[1] + su.float().T @ x.float(),
                        initial[2] + dy.float().T @ activation.float(),
                    ]
                    dw_routed = [value.clone() for value in initial[3:]]
                else:
                    dw_shared = [
                        (sg.float().T @ x.float()).bfloat16(),
                        (su.float().T @ x.float()).bfloat16(),
                        (dy.float().T @ activation.float()).bfloat16(),
                    ]
                    dw_routed = [torch.zeros_like(weight) for weight in routed]
                dx_reference = torch.empty_like(dx_peer)
                ds_reference = torch.empty_like(scores).flatten()
                y_routed = torch.empty_like(combine)
                offset = 0
                for expert, selected in enumerate(slots):
                    if selected.numel():
                        value = x[selected // topk]
                        score = scores.flatten()[selected]
                        gradient = (
                            dy[selected // topk].float() * score[:, None]
                        ).bfloat16()
                        rx, rg, ru, _, rhidden = _mlp_backward(
                            value, gradient, [w[expert] for w in routed], swiglu_limit
                        )
                        y_routed[selected] = (
                            rhidden.float() @ routed[2][expert].float().T
                        ).bfloat16()
                        dx_reference[selected] = rx
                        ds_reference[selected] = (
                            y_routed[selected].float() * dy[selected // topk].float()
                        ).sum(-1)
                        first = True
                        for start in range(0, actual, macro):
                            lo = max(offset, start) - offset
                            hi = min(offset + selected.numel(), start + macro) - offset
                            if lo >= hi:
                                continue
                            exact = [
                                rg[lo:hi].float().T @ value[lo:hi].float(),
                                ru[lo:hi].float().T @ value[lo:hi].float(),
                                gradient[lo:hi].float().T @ rhidden[lo:hi].float(),
                            ]
                            for destination, product in zip(
                                dw_routed, exact, strict=True
                            ):
                                if fp32_wgrad:
                                    destination[expert] += product
                                elif first:
                                    destination[expert] = product.bfloat16()
                                else:
                                    # Native BF16 accumulation across macrobatches.
                                    destination[expert] = (
                                        destination[expert].float()
                                        + product.bfloat16().float()
                                    ).bfloat16()
                            first = False
                    offset += count_values[expert]
                return {
                    0: sx, 2: sg, 4: su, 6: sh,
                    9: dw_shared[0], 10: dw_routed[0], 11: dw_shared[1],
                    12: dw_routed[1], 13: dw_shared[2], 14: dw_routed[2],
                    15: dx_reference, 16: ds_reference.reshape_as(scores),
                    17: (activation.float() @ shared[2].float().T).bfloat16(),
                    18: y_routed,
                }  # fmt: skip

            def outputs(result):
                return (*result, dx_peer, ds_peer, context[0][7], combine)

            def check(result, skip=()):
                actual_values = outputs(result)
                for index, expected in reference().items():
                    if index in skip:
                        continue
                    value = actual_values[index]
                    assert torch.isfinite(value).all().item(), (empty, hidden, index)
                    torch.testing.assert_close(
                        value.float(), expected.float(), atol=1e-2, rtol=1e-2
                    )

            compared = (0, 2, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
            saved = []
            for _ in range(3):
                result = sequence()
                check(result)
                values = outputs(result)
                saved.append(tuple(values[index].clone() for index in compared))
            assert all(
                all(torch.equal(a, b) for a, b in zip(saved[0], other, strict=True))
                for other in saved[1:]
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = sequence()
            replays = []
            for _ in range(3):
                graph.replay()
                check(result)
                values = outputs(result)
                replays.append(tuple(values[index].clone() for index in compared))
            assert all(
                all(torch.equal(a, b) for a, b in zip(replays[0], other, strict=True))
                for other in replays[1:]
            )
            # The score gradient is fused into SwiGLU backward as
            # dot(d_hidden, hidden) / score, as in native MoK. Changed positive
            # scores keep the independent dot(y_expert, dy) reference; a zero
            # score returns zero for the positive-coefficient contract.
            scores.add_(0.03125)
            graph.replay()
            check(result)
            scores.zero_()
            graph.replay()
            check(result, skip=(16,))
            assert torch.count_nonzero(ds_peer).item() == 0
            for _ in range(2):
                x.normal_(std=0.125)
                dy.normal_(std=0.125)
                scores.uniform_(0.1, 1.0)
                scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
                graph.replay()
                check(result)
            records.append(dict(empty=empty, hidden=hidden, macro=macro, routed=actual))
    return records


@pytest.fixture(autouse=True)
def _exact_matmul(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False
    )


def test_fused_training():
    require_gpu()
    assert len(_check_fused_training()) == 6


def test_fused_training_clamped_swiglu():
    require_gpu()
    # Activations have standard deviation near 0.125, so this limit clamps a
    # substantial fraction of both gate and up values.
    _check_fused_training(
        swiglu_limit=0.125, geometries=GEOMETRIES[:1], empty_cases=(False,)
    )


@pytest.mark.parametrize("geometries", [GEOMETRIES[:1], GEOMETRIES], ids=["one", "all"])
def test_fused_training_fp32_wgrad(geometries):
    require_gpu()
    _check_fused_training(fp32_wgrad=True, geometries=geometries, empty_cases=(False,))


@pytest.mark.parametrize("experts", [36, 9], ids=["ep8-local36", "ep32-local9"])
def test_fused_training_glm_flash_local_experts(experts):
    """GLM-5.3-Flash local expert counts (288 experts at EP8/EP32), top-8, clamped."""
    require_gpu()
    _check_fused_training(
        swiglu_limit=0.125,
        experts=experts,
        topk=8,
        tokens=512,
        geometries=((512, 256, 2048),),
        empty_cases=(False,),
    )


@pytest.mark.parametrize("swiglu_limit", [None, 0.125], ids=["plain", "clamped"])
@pytest.mark.parametrize("macro", [768, 4096], ids=["replay", "resident"])
def test_recompute_context(swiglu_limit, macro):
    """Context-only recompute equals the saved forward context; backward agrees bitwise."""
    require_gpu()
    clamped = swiglu_limit is not None
    forward, recompute = MoKForward(clamped), MoKRecompute(clamped)
    backward = MoKBackward(clamped)
    tokens, hidden, intermediate, experts, topk, mini, comm = (
        512,
        512,
        256,
        4,
        2,
        256,
        4,
    )
    torch.manual_seed(4111)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, **options) * 0.125
    dy = torch.randn_like(x) * 0.125
    shared = [
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    scores = torch.rand(tokens, topk, device="cuda") + 0.1
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    combine = torch.zeros(tokens * topk, hidden, **options)
    saved = forward(
        x, [x.data_ptr()], combine, [combine.data_ptr()],
        shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    rebuilt = recompute(
        x, [x.data_ptr()], shared[0], routed[0], shared[1], routed[1],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    rows = min(s["actual"], macro)
    for index in (0, 2, 4, 6):  # routed x, gate, up, hidden of the resident macrobatch
        assert torch.equal(saved[index][:rows], rebuilt[index][:rows]), index
    for index in (1, 3, 5):  # shared gate, up, hidden
        assert torch.equal(saved[index], rebuilt[index]), index

    def gradients(context):
        dx_peer = torch.zeros(tokens * topk, hidden, **options)
        ds_peer = torch.zeros_like(scores)
        values = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
            *context[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        return (dx_peer, ds_peer, values[0], *values[9:])

    for a, b in zip(gradients(saved), gradients(rebuilt), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("ep,experts", SCHEDULER_LAYOUTS)
def test_scheduler(ep, experts):
    require_gpu()
    topk = 2 if experts * ep <= 16 else 8
    scheduler = MoKScheduler(ep, experts)
    tokens = 512
    generator = torch.Generator(device="cuda").manual_seed(1107)
    ids = (
        torch.rand(ep * tokens, ep * experts, device="cuda", generator=generator)
        .argsort(-1)[:, :topk]
        .int()
        .reshape(ep, tokens, topk)
        .contiguous()
    )
    # A masked tail (expert -1) models a rank with fewer real source rows.
    ids[0, tokens - 37 :] = -1
    capacity = ep * tokens * topk + experts * 256
    owner_loads = []
    for rank in {0, ep - 1}:
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        expected_counts = torch.bincount(
            ids.flatten().long().clamp(min=0), minlength=ep * experts
        )
        expected_counts[0] -= (ids == -1).sum()
        expected_counts = expected_counts[rank * experts : (rank + 1) * experts]
        expected_counts = ((expected_counts + 255) // 256) * 256
        torch.testing.assert_close(counts, expected_counts.int(), rtol=0, atol=0)
        assert counts.shape == (experts,) and total.shape == (1,)
        assert total.item() == expected_counts.sum().item()
        offset = 0
        for expert, count in enumerate(expected_counts.tolist()):
            p = peers[offset : offset + count].long()
            t = slots[offset : offset + count].long()
            valid = p >= 0
            actual = (p[valid] * tokens * topk + t[valid]).sort().values
            expected = (ids.flatten() == rank * experts + expert).nonzero().flatten()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            offset += count
        assert (peers[offset:] == -1).all()
        # The count kernel also bins every valid gathered id by owning rank
        # (the tail of its counts buffer): exactly the host reference count.
        owner = scheduler.owner_loads()
        assert owner.dtype == torch.int32 and owner.shape == (ep,)
        assert torch.equal(owner.long(), backend.count_routed_load(ids, ep, experts))
        owner_loads.append(owner.clone())
    assert all(torch.equal(owner_loads[0], other) for other in owner_loads[1:])
    scheduler.release_counts()
    with pytest.raises(RuntimeError, match="scheduler call"):
        scheduler.owner_loads()


@pytest.mark.parametrize("ep,experts", [(4, 4), (8, 32)])
def test_scheduler_owner_loads_match_count_routed_load_with_invalid_ids(ep, experts):
    """The owner-rank bins drop padding (-1), more negative ids and ids at or
    past the limit exactly as ``count_routed_load`` does, on device and on the
    CPU reference."""
    require_gpu()
    topk = 2 if ep * experts <= 16 else 8
    limit = ep * experts
    generator = torch.Generator(device="cuda").manual_seed(2203)
    ids = torch.randint(
        -1, limit + 8, (ep, 256, topk), device="cuda", dtype=torch.int32,
        generator=generator,
    )  # fmt: skip
    edges = torch.tensor(
        [-2, -1000, limit, limit + experts - 1], device="cuda", dtype=torch.int32
    )
    ids[1, :16, 0] = edges.repeat(4)
    scheduler = MoKScheduler(ep, experts)
    scheduler(ids, ep * 256 * topk + experts * 256, 0)
    owner = scheduler.owner_loads()
    expected = backend.count_routed_load(ids, ep, experts)
    assert owner.dtype == torch.int32 and owner.shape == (ep,)
    assert torch.equal(owner.long(), expected)
    assert torch.equal(
        expected.cpu(), backend.count_routed_load(ids.cpu(), ep, experts)
    )


@pytest.mark.parametrize(
    "ep,experts,tokens,rank,real",
    [
        (8, 32, 16384, 5, (8191, 16172, 16231, 0, 8191, 16172, 16231, 0)),
        (4, 64, 32768, 3, None),
    ],
    ids=["ep8-local32-ragged", "ep4-local64-full"],
)
def test_scheduler_at_training_geometries(ep, experts, tokens, rank, real):
    """The scheduler assertions of ``test_scheduler`` and the owner-rank bins at
    full training geometries: 256 routed experts, top-8, 16384 or 32768 source
    rows per rank, ragged ranks (empty ones included) or full ranks."""
    require_gpu()
    topk = 8
    generator = torch.Generator(device="cuda").manual_seed(5309 + ep)
    ids = (
        torch.rand(ep * tokens, ep * experts, device="cuda", generator=generator)
        .argsort(-1)[:, :topk]
        .int()
        .reshape(ep, tokens, topk)
        .contiguous()
    )
    if real is not None:
        for source, rows in enumerate(real):
            ids[source, rows:] = -1
    capacity = tokens * topk * 2
    scheduler = MoKScheduler(ep, experts)
    owner_loads = []
    for r in (0, rank):
        peers, slots, total, counts = scheduler(ids, capacity, r)
        expected_counts = torch.bincount(
            ids.flatten().long().clamp(min=0), minlength=ep * experts
        )
        expected_counts[0] -= (ids == -1).sum()
        expected_counts = expected_counts[r * experts : (r + 1) * experts]
        expected_counts = ((expected_counts + 255) // 256) * 256
        torch.testing.assert_close(counts, expected_counts.int(), rtol=0, atol=0)
        assert total.item() == expected_counts.sum().item() <= capacity
        offset = 0
        for expert, count in enumerate(expected_counts.tolist()):
            p = peers[offset : offset + count].long()
            t = slots[offset : offset + count].long()
            valid = p >= 0
            actual = (p[valid] * tokens * topk + t[valid]).sort().values
            expected = (ids.flatten() == r * experts + expert).nonzero().flatten()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            offset += count
        assert (peers[offset:] == -1).all()
        owner = scheduler.owner_loads()
        assert torch.equal(owner.long(), backend.count_routed_load(ids, ep, experts))
        owner_loads.append(owner.clone())
    assert torch.equal(owner_loads[0], owner_loads[1])
    if real is not None:
        assert owner_loads[0].sum().item() == sum(real) * topk


@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("tokens,hidden", [(512, 256), (37, 6144), (1, 1032)])
def test_epilogues(topk, tokens, hidden):
    require_gpu()
    epilogues = MoKEpilogues(topk)
    shared = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
    routed = (
        torch.randn(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
    )
    scores = torch.rand(tokens, topk, device="cuda")
    # Native order: FP32 accumulation from the shared row, then ascending slots.
    expected = shared.float()
    expected_dx = shared.float()
    for k in range(topk):
        part = routed.float().view(tokens, topk, hidden)[:, k]
        expected = expected + part * scores[:, k : k + 1]
        expected_dx = expected_dx + part
    # The weighted sum may contract multiply-adds; the unweighted sum is exact.
    torch.testing.assert_close(
        epilogues.forward(shared, routed, scores),
        expected.bfloat16(),
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.equal(epilogues.backward(shared, routed), expected_dx.bfloat16())


def test_count_routed_load_matches_a_reference_count():
    """Host-only: the per-rank load histogram counts the valid gathered ids by
    owning rank and ignores padding (-1) and out-of-range ids."""
    count_routed_load = backend.count_routed_load

    ep_size, local_experts = 4, 3
    routes = torch.tensor(
        [[0, 5, 11, -1], [3, 3, 12, -7], [8, 9, 10, 11], [-1, -1, -1, 2]],
        dtype=torch.int32,
    )
    expected = [0] * ep_size
    for e in routes.reshape(-1).tolist():
        if 0 <= e < ep_size * local_experts:
            expected[e // local_experts] += 1
    loads = count_routed_load(routes, ep_size, local_experts)
    assert loads.dtype == torch.int64
    assert loads.tolist() == expected == [2, 3, 1, 4]
    empty = torch.full((6,), -1, dtype=torch.int32)
    assert count_routed_load(empty, ep_size, local_experts).tolist() == [0] * ep_size
    # Ids exactly at the limit and one owner beyond it (the last histogram bin
    # and the first value outside the range), more negative padding, and a large
    # random gather per rank count: all match the reference count exactly.
    limit = ep_size * local_experts
    edges = torch.tensor(
        [limit, limit + local_experts - 1, limit + local_experts, -2, -1000, 0],
        dtype=torch.int32,
    )
    assert count_routed_load(edges, ep_size, local_experts).tolist() == [1, 0, 0, 0]
    generator = torch.Generator().manual_seed(0)
    ep_size, local_experts = 64, 4
    random_routes = torch.randint(
        -1, ep_size * local_experts + 8, (ep_size, 2048, 8), generator=generator
    ).to(torch.int32)
    reference = torch.zeros(ep_size, dtype=torch.int64)
    valid = (random_routes >= 0) & (random_routes < ep_size * local_experts)
    reference.index_add_(
        0,
        torch.div(random_routes[valid], local_experts, rounding_mode="floor").long(),
        torch.ones(int(valid.sum()), dtype=torch.int64),
    )
    assert torch.equal(
        count_routed_load(random_routes, ep_size, local_experts), reference
    )


class _AtenRecorder:
    """Names of the aten ops dispatched inside the block (``TorchDispatchMode``)."""

    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        recorder = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                recorder.ops.append(func.overloadpacket.__name__)
                return func(*args, **(kwargs or {}))

        self.ops = []
        self._mode = Mode()

    def __enter__(self):
        self._mode.__enter__()
        return self

    def __exit__(self, *args):
        self._mode.__exit__(*args)


def test_observe_load_issues_one_copy_and_no_histogram_ops():
    """Once the workspace's load entry exists, an observation copies the count
    kernel's owner-rank tail into the pinned int32 slot through one view and
    one copy: no histogram, cast, division, fill or allocation is dispatched."""
    require_gpu()
    from collections import Counter
    from types import SimpleNamespace

    ep_size, experts = 4, 4
    functional = backend.MoKFunctional(ep_size, experts, 2)
    config = MoKConfig(
        fwd_num_comm_sms=4,
        bwd_num_comm_sms=4,
        imbalance_comm_sms=(6, 6),
        imbalance_threshold=1.5,
    )
    routes = (
        torch.arange(ep_size * 256 * 2, device="cuda", dtype=torch.int32)
        % (ep_size * experts)
    ).view(ep_size, 256, 2)
    workspace = SimpleNamespace(ep_rank=0, ep_size=ep_size)
    functional.scheduler(routes, ep_size * 256 * 2 + experts * 256, 0)
    # The first observation of a workspace creates its entry (two pinned
    # slots, two ``empty`` calls); every later observation is the steady state.
    functional._observe_load(workspace, config)
    with _AtenRecorder() as recorder:
        functional._observe_load(workspace, config)
    ops = Counter(recorder.ops)
    assert ops["copy_"] == 1, ops
    assert sum(ops[name] for name in ("slice", "narrow", "select")) <= 1, ops
    forbidden = {
        "histc",
        "div",
        "floor_divide",
        "_to_copy",
        "to",
        "fill_",
        "zeros",
        "empty",
    }
    assert not (set(ops) & forbidden), ops
    entry = functional._load_entry(workspace, create=False)
    host = entry["host"][entry["pending"]]
    assert host.dtype == torch.int32 and host.is_pinned() and host.numel() == ep_size
    torch.cuda.synchronize()
    assert host.tolist() == backend.count_routed_load(routes, ep_size, experts).tolist()


def _layout_reference(counts, peers):
    """Torch reference of the expert layout's written prefix: padded counts,
    exclusive row offsets, real rows per expert and the expert of each
    256-row block."""
    padded = counts.long()
    offsets = torch.cumsum(padded, 0) - padded
    real = torch.stack(
        [
            (peers[start : start + count] >= 0).sum()
            for start, count in zip(offsets.tolist(), padded.tolist(), strict=True)
        ]
    )
    blocks = torch.empty(
        int(padded.sum()) // 256, dtype=torch.int64, device=peers.device
    )
    for expert, (start, count) in enumerate(
        zip(offsets.tolist(), padded.tolist(), strict=True)
    ):
        blocks[start // 256 : (start + count) // 256] = expert
    return torch.cat([padded, offsets, real, blocks]).int()


@pytest.mark.parametrize("ep,experts", SCHEDULER_LAYOUTS)
def test_expert_layout_matches_torch_reference(ep, experts):
    """The layout kernel's written prefix equals the torch reference on the
    scheduler's own outputs (masked ranks and empty experts included), through
    the module helper and the scheduler's ``layout`` method alike."""
    require_gpu()
    topk = 2 if experts * ep <= 16 else 8
    scheduler = MoKScheduler(ep, experts)
    tokens = 512
    generator = torch.Generator(device="cuda").manual_seed(1107)
    ids = (
        torch.rand(ep * tokens, ep * experts, device="cuda", generator=generator)
        .argsort(-1)[:, :topk]
        .int()
        .reshape(ep, tokens, topk)
        .contiguous()
    )
    ids[0, tokens - 37 :] = -1
    if experts >= 4:
        ids[ids == experts - 1] = -1  # an empty local expert on rank 0
    capacity = ep * tokens * topk + experts * 256
    peers, _, total, counts = scheduler(ids, capacity, 0)
    layout = kernels.expert_layout(scheduler.layout_module, counts, capacity, peers)
    assert (
        layout.dtype == torch.int32 and layout.numel() == 3 * experts + capacity // 256
    )
    written = 3 * experts + total.item() // 256
    assert torch.equal(layout[:written], _layout_reference(counts, peers))
    again = scheduler.layout(counts, capacity, peers)
    assert torch.equal(again[:written], layout[:written])


def test_launchers_accept_a_precomputed_layout(monkeypatch):
    """The fused launchers derive the expert layout themselves (``layout=None``)
    or read the one passed to them; both paths are bitwise equal, eager and
    under graph replay, and a layout of another geometry is rejected."""
    require_gpu()
    forward, recompute, backward = MoKForward(), MoKRecompute(), MoKBackward()
    tokens, hidden, intermediate, experts, topk, mini, comm, macro = (
        512, 512, 256, 4, 2, 256, 4, 768,
    )  # fmt: skip
    torch.manual_seed(4111)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, **options) * 0.125
    dy = torch.randn_like(x) * 0.125
    shared = [
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    scores = torch.rand(tokens, topk, device="cuda") + 0.1
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    capacity = s["peers"].numel()
    layout = kernels.expert_layout(
        forward.layout_module, s["counts"], capacity, s["peers"]
    )
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer = torch.zeros(tokens * topk, hidden, **options)
    ds_peer = torch.zeros_like(scores)
    rows = min(s["actual"], macro)
    calls = []
    original = kernels.expert_layout

    def counting(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(kernels, "expert_layout", counting)

    def sequence(layout_arg):
        combine.zero_()
        dx_peer.zero_()
        ds_peer.zero_()
        saved = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
            *schedule, topk, None, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        rebuilt = recompute(
            x, [x.data_ptr()], shared[0], routed[0], shared[1], routed[1],
            *schedule, topk, None, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        grads = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
            *saved[:7], x, [x.data_ptr()], *schedule,
            topk, None, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        return saved, rebuilt, grads

    def outputs(result):
        saved, rebuilt, grads = result
        return (
            *(saved[index][:rows] for index in (0, 2, 4, 6)),
            *(saved[index] for index in (1, 3, 5, 7)),
            *(rebuilt[index][:rows] for index in (0, 2, 4, 6)),
            *(rebuilt[index] for index in (1, 3, 5)),
            grads[0], *grads[9:],
            combine, dx_peer, ds_peer,
        )  # fmt: skip

    runs = []
    for layout_arg in (None, None, None, layout, layout, layout):
        before = len(calls)
        runs.append(tuple(v.clone() for v in outputs(sequence(layout_arg))))
        assert len(calls) - before == (3 if layout_arg is None else 0)
    assert all(
        all(torch.equal(a, b) for a, b in zip(runs[0], other, strict=True))
        for other in runs[1:]
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sequence(layout)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert all(
            torch.equal(a, b) for a, b in zip(runs[0], outputs(captured), strict=True)
        )
    short = layout[:-1]
    with pytest.raises(ValueError, match="schedule geometry"):
        forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
            *schedule, topk, None, comm, macro, mini, layout=short,
        )  # fmt: skip


def _slab_layout(sizes):
    """Expected (offset, size) of every kept piece of ``_zero_slab(..., *sizes)``
    and the total length, in int32 elements."""
    pieces, total = [], 0
    for size in sizes:
        total += -total % 32
        for n in size if isinstance(size, tuple) else (size,):
            pieces.append((total, n))
            total += n
    return pieces, total


@pytest.mark.parametrize(
    "sizes",
    [
        (1, 64, 8704, 1088, 64, 2048, 1),
        (64, 64, 64, 8704, 1088, 8192, 1024, 1, 1),
        (1,),
        (33, 1, 31, 32),
        (264, (1, 32)),
    ],
)
def test_zero_slab_views_are_disjoint_aligned_and_zero(sizes):
    """One zero slab per call: every counter a contiguous 1-D int32 view with a
    128-byte aligned start, pairwise disjoint on one storage, zero, tuple
    groups adjacent; fresh per call and re-zeroed on every graph replay."""
    require_gpu()
    expected, total = _slab_layout(sizes)
    views = kernels._zero_slab("cuda", *sizes)
    assert len(views) == len(expected)
    base = views[0].untyped_storage().data_ptr()
    for view, (offset, size) in zip(views, expected, strict=True):
        assert view.dtype == torch.int32 and view.is_cuda and view.dim() == 1
        assert view.numel() == size and view.is_contiguous()
        assert view.untyped_storage().data_ptr() == base
        assert view.data_ptr() == base + 4 * offset
        assert torch.count_nonzero(view).item() == 0
    assert views[0].untyped_storage().nbytes() == 4 * total
    for view in views:
        if (
            view.data_ptr() % 128 != 0
        ):  # a tuple-group member: adjacent to its predecessor
            previous = [
                v for v in views if v.data_ptr() + 4 * v.numel() == view.data_ptr()
            ]
            assert len(previous) == 1
    ranges = sorted((v.data_ptr(), v.data_ptr() + 4 * v.numel()) for v in views)
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False))
    views[0].fill_(7)
    assert all(torch.count_nonzero(v).item() == 0 for v in views[1:])
    again = kernels._zero_slab("cuda", *sizes)
    assert again[0].data_ptr() != views[0].data_ptr()
    assert all(torch.count_nonzero(v).item() == 0 for v in again)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernels._zero_slab("cuda", *sizes)
        captured[0].fill_(5)
    for _ in range(2):
        captured[-1].fill_(9)
        graph.replay()
        torch.cuda.synchronize()
        assert (captured[0] == 5).all().item()
        assert all(torch.count_nonzero(v).item() == 0 for v in captured[1:])


def test_zero_slab_issues_one_zeros_and_one_split():
    """The slab is one ``zeros`` and one ``split_with_sizes``: no per-counter
    slicing, filling or copying."""
    require_gpu()
    from collections import Counter

    with _AtenRecorder() as recorder:
        kernels._zero_slab("cuda", 1, 64, 8704, 1088, 64, 2048, 1)
    assert Counter(recorder.ops) == {"zeros": 1, "split_with_sizes": 1}, recorder.ops


class _LaunchRecorder:
    """Stands in for a prepared kernel: records the launch keywords, launches nothing."""

    def __init__(self):
        self.calls = []

    def launch(self, *, grid, **kwargs):
        self.calls.append((grid, kwargs))

    def last(self, name):
        return next(kwargs for _, kwargs in reversed(self.calls) if name in kwargs)


def _check_counters(kwargs, expected):
    """Counter keywords ``expected`` (name -> numel) of one recorded fused launch:
    CUDA, contiguous, 1-D, int32 (uint32 for ``combine_next``), the exact
    numel, zero, 128-byte aligned, pairwise disjoint on one storage."""
    views = {name: kwargs[name] for name in expected}
    assert len({v.untyped_storage().data_ptr() for v in views.values()}) == 1
    for name, numel in expected.items():
        view = views[name]
        assert view.is_cuda and view.is_contiguous() and view.dim() == 1, name
        assert view.dtype == (torch.uint32 if name == "combine_next" else torch.int32)
        assert view.numel() == numel, (name, view.numel(), numel)
        assert torch.count_nonzero(view.view(torch.int32)).item() == 0, name
        assert view.data_ptr() % 128 == 0, name
    ranges = sorted(
        (v.data_ptr(), v.data_ptr() + 4 * v.numel()) for v in views.values()
    )
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False))


def _bf16_backward_clusters(
    local_tokens, hidden, intermediate, capacity, macro, mini, experts
):
    """The BF16 backward grid's cluster count, as the launcher derived it per call."""
    minis = (capacity + mini - 1) // mini
    macros = (capacity + macro - 1) // macro
    shared_rows = local_tokens // 256
    ibw, hbw = (intermediate + 511) // 512, (hidden + 511) // 512
    shared_tasks = (
        shared_rows * (ibw + hbw)
        + kernels.swiglu_backward_tasks((local_tokens // 128) * (intermediate // 128))
        + 3 * kernels.wgrad_tiles(hidden, intermediate)
    )
    mini_bwd = (mini // 256) * (ibw + hbw) + kernels.swiglu_backward_tasks(
        (mini // 128) * (intermediate // 128)
    )
    mini_replay = (
        2 * (mini // 256) * ibw + ((mini // 128) * (intermediate // 128) + 5) // 6
    )
    return (
        shared_tasks
        + minis * mini_bwd
        + max(0, minis - macro // mini) * mini_replay
        + macros * 3 * experts * kernels.wgrad_tiles(hidden, intermediate)
    )


def test_launch_counters_are_one_slab_per_launch():
    """At the launch boundary the scheduler and the BF16 launchers hand the
    kernels the counter names, dtypes and extents they always did, now as
    views of one zero slab per launch; the schedule sentinel stays separate,
    and one launcher memoises its geometry per distinct expert count."""
    require_gpu()
    ep, experts, tokens, topk = 4, 4, 512, 2
    scheduler = MoKScheduler(ep, experts)
    recorder = _LaunchRecorder()
    scheduler.count = scheduler.pad = scheduler.rows = recorder
    generator = torch.Generator(device="cuda").manual_seed(77)
    ids = torch.randint(
        0, ep * experts, (ep, tokens, topk), device="cuda", dtype=torch.int32,
        generator=generator,
    )  # fmt: skip
    capacity = ep * tokens * topk + experts * 256
    peers, _, total, counts = scheduler(ids, capacity, 0)
    count_kw, pad_kw, rows_kw = (kwargs for _, kwargs in recorder.calls)
    assert count_kw["counts"] is pad_kw["counts"] is rows_kw["counts"]
    slab = count_kw["counts"]
    assert slab.numel() == experts * ep + ep and slab.dtype == torch.int32
    assert slab.data_ptr() % 128 == 0 and torch.count_nonzero(slab).item() == 0
    assert pad_kw["num_tokens"] is total and pad_kw["tokens_per_expert"] is counts
    assert total.numel() == 1 and counts.numel() == experts
    assert total.untyped_storage().data_ptr() == slab.untyped_storage().data_ptr()
    assert total.data_ptr() % 128 == 0 and counts.data_ptr() == total.data_ptr() + 4
    assert total.data_ptr() >= slab.data_ptr() + 4 * slab.numel()
    owner = scheduler.owner_loads()
    assert (
        owner.numel() == ep and owner.data_ptr() == slab.data_ptr() + 4 * experts * ep
    )
    assert rows_kw["schedule_peer_rank"] is peers and (peers == -1).all().item()
    assert peers.untyped_storage().data_ptr() != slab.untyped_storage().data_ptr()

    hidden, intermediate, macro, mini, comm = 256, 256, 768, 256, 4
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, **options)
    dy = torch.randn_like(x)
    shared = [torch.randn(intermediate, hidden, **options) for _ in range(2)] + [
        torch.randn(hidden, intermediate, **options)
    ]
    scores = torch.rand(tokens, topk, device="cuda") + 0.1
    routes = torch.stack(
        [
            (torch.arange(tokens, device="cuda") + slot) % experts
            for slot in range(topk)
        ],
        -1,
    )
    combine = torch.empty(tokens * topk, hidden, **options)
    dx_peer, ds_peer = torch.empty_like(combine), torch.empty_like(scores)
    forward, recompute, backward = MoKForward(), MoKRecompute(), MoKBackward()
    for launcher in (forward, recompute, backward):
        launcher.module = launcher.layout_module = _LaunchRecorder()
    backward.zero_module = backward.module
    for count in (experts, 2 * experts):
        routed = [
            torch.randn(count, intermediate, hidden, **options),
            torch.randn(count, intermediate, hidden, **options),
            torch.randn(count, hidden, intermediate, **options),
        ]
        s = single_rank_schedule(routes % count, count)
        schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
        cap = s["peers"].numel()
        minis, macros = (cap + mini - 1) // mini, (cap + macro - 1) // macro
        shared_rows, routed_rows = tokens // 256, cap // 256
        ib = intermediate // 256
        for _ in range(2):
            saved = forward(
                x, [x.data_ptr()], combine, [combine.data_ptr()],
                shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                *schedule, topk, None, comm, macro, mini,
            )  # fmt: skip
            recompute(
                x, [x.data_ptr()], shared[0], routed[0], shared[1], routed[1],
                *schedule, topk, None, comm, macro, mini,
            )  # fmt: skip
            backward(
                dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
                scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
                shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                *saved[:7], x, [x.data_ptr()], *schedule,
                topk, None, comm, macro, mini,
            )  # fmt: skip
        _check_counters(
            forward.module.last("x_ready"),
            dict(
                x_ready=minis,
                hidden_ready=shared_rows + routed_rows,
                y_ready=minis,
                y_done=cap // 128,
            ),
        )
        routed_capacity = min(cap, macro)
        _check_counters(
            recompute.module.last("x_ready"),
            dict(
                x_ready=(routed_capacity + mini - 1) // mini,
                hidden_ready=shared_rows + routed_capacity // 256,
            ),
        )
        _check_counters(
            backward.module.last("dy_ready"),
            dict(
                dy_ready=minis,
                dh_ready=(shared_rows + routed_rows) * ib,
                dg_ready=shared_rows + routed_rows,
                dx_ready=minis,
                replay_x=minis,
                replay_gu=routed_rows * ib,
                replay_h=routed_rows,
                buffers_done=macros,
                weight_ready=macros,
            ),
        )
        grid, _ = next(
            call for call in reversed(backward.module.calls) if "dy_ready" in call[1]
        )
        clusters = _bf16_backward_clusters(
            tokens, hidden, intermediate, cap, macro, mini, count
        )
        assert grid == (comm + 2 * clusters, 1, 1)
    # One geometry entry per expert count on every launcher; the backward
    # geometry depends on it.
    launchers = (forward, recompute, backward)
    assert [len(launcher._geometry) for launcher in launchers] == [2, 2, 2]
    first, second = backward._geometry.values()
    assert first.clusters != second.clusters
    assert first.weight_tasks != second.weight_tasks


def _mxfp8_forward_geometry_reference(
    local_tokens, hidden, intermediate, capacity, macro, mini, tile
):
    """The MXFP8 forward launcher's per-call task arithmetic, as derived before memoisation."""
    shared_rows = local_tokens // 256
    shared_tasks = shared_rows * (intermediate // 256) + shared_rows * (
        (hidden + 511) // 512
    )
    mini_swiglu = ((mini // 128) * (intermediate // 128) + 5) // 6
    gate_passes = 1 if tile == 512 else 2
    mini_tasks = (
        gate_passes * (mini // 256) * (intermediate // 256)
        + mini_swiglu
        + (mini // 256) * ((hidden + tile - 1) // tile)
    )
    minis = (capacity + mini - 1) // mini
    return dict(
        m_tiles=macro // 128,
        h_tiles=hidden // 128,
        i_tiles=intermediate // 128,
        shared_rows=shared_rows,
        routed_rows=capacity // 256,
        shared_tasks=shared_tasks,
        mini_swiglu=mini_swiglu,
        mini_tasks=mini_tasks,
        minis=minis,
        mini_tasks_div=tuple(kernels._fast_divmod(mini_tasks)),
        minis_per_macro_div=tuple(kernels._fast_divmod(macro // mini)),
        gate_col_blocks_div=tuple(kernels._fast_divmod((intermediate + 255) // 256)),
        down_col_blocks_div=tuple(kernels._fast_divmod((hidden + tile - 1) // tile)),
    )


def _mxfp8_backward_geometry_reference(
    local_tokens, hidden, intermediate, capacity, macro, mini, experts, tile
):
    """The MXFP8 backward launcher's per-call task arithmetic, as derived before memoisation."""
    minis = (capacity + mini - 1) // mini
    macros = (capacity + macro - 1) // macro
    shared_rows, routed_rows = local_tokens // 256, capacity // 256
    ib = intermediate // 256
    ibw, hbw = (intermediate + 511) // 512, (hidden + 511) // 512
    ibr, hbr = (intermediate + tile - 1) // tile, (hidden + tile - 1) // tile
    shared_tasks = (
        shared_rows * (ibw + hbw)
        + ((local_tokens // 128) * (intermediate // 128) + 3) // 4
        + 3 * kernels.wgrad_tiles(hidden, intermediate)
    )
    mini_bwd = (mini // 256) * (ibr + hbr) + (
        (mini // 128) * (intermediate // 128) + 3
    ) // 4
    mini_replay = (
        2 * (mini // 256) * ib + ((mini // 128) * (intermediate // 128) + 5) // 6
    )
    clusters = (
        shared_tasks
        + minis * mini_bwd
        + max(0, minis - macro // mini) * mini_replay
        + macros
        * 3
        * experts
        * kernels.wgrad_tiles(hidden, intermediate, tile_cols=tile)
    )
    shared_wgrad = kernels.wgrad_tiles(hidden, intermediate)
    routed_wgrad = kernels.wgrad_tiles(hidden, intermediate, tile_cols=tile)
    weight_tasks = experts * routed_wgrad
    replay_macro_tasks = (macro // mini) * (mini_replay + mini_bwd) + 3 * weight_tasks
    return dict(
        minis=minis,
        macros=macros,
        shared_rows=shared_rows,
        routed_rows=routed_rows,
        ib=ib,
        clusters=clusters,
        weight_tasks=weight_tasks,
        replay_macro_tasks=replay_macro_tasks,
        mini_size_div=tuple(kernels._fast_divmod(mini)),
        mini_bwd_div=tuple(kernels._fast_divmod(mini_bwd)),
        mini_replay_div=tuple(kernels._fast_divmod(mini_replay)),
        replay_macro_tasks_div=tuple(kernels._fast_divmod(replay_macro_tasks)),
        weight_tasks_div=tuple(kernels._fast_divmod(weight_tasks)),
        shared_wgrad_div=tuple(kernels._fast_divmod(shared_wgrad)),
        routed_wgrad_div=tuple(kernels._fast_divmod(routed_wgrad)),
        dh_col_blocks_div=tuple(kernels._fast_divmod(ibr)),
        dx_col_blocks_div=tuple(kernels._fast_divmod(hbr)),
        replay_col_blocks_div=tuple(kernels._fast_divmod((intermediate + 255) // 256)),
    )


@pytest.mark.parametrize("tile", [256, 512])
@pytest.mark.parametrize(
    "local_tokens,capacity,macro,experts",
    [(16384, 262144, 262144, 32), (32768, 524288, 262144, 64)],
    ids=["ep8-local32", "ep4-local64"],
)
def test_launcher_geometry_memo_is_pure(tile, local_tokens, capacity, macro, experts):
    """The memoised launch geometry of the MXFP8 launchers equals a fresh
    evaluation of the per-call arithmetic at training geometries and both tile
    widths, including the expert-dependent backward cluster and task counts; a
    bounded memo clears and still serves correct values."""
    hidden, intermediate, mini = 6144, 2048, 4096
    forward = kernels._forward_mxfp8_geometry(
        local_tokens, hidden, intermediate, capacity, macro, mini, tile
    )
    for name, value in _mxfp8_forward_geometry_reference(
        local_tokens, hidden, intermediate, capacity, macro, mini, tile
    ).items():
        actual = getattr(forward, name)
        assert (tuple(actual) if name.endswith("_div") else actual) == value, name
    assert forward.tile == tile and forward.gate_passes == (1 if tile == 512 else 2)
    for count in (experts, 9):
        backward = kernels._backward_mxfp8_geometry(
            local_tokens, hidden, intermediate, capacity, macro, mini, count, tile
        )
        for name, value in _mxfp8_backward_geometry_reference(
            local_tokens, hidden, intermediate, capacity, macro, mini, count, tile
        ).items():
            actual = getattr(backward, name)
            assert (tuple(actual) if name.endswith("_div") else actual) == value, name
    cache = {}
    keys = [(i, tile) for i in range(kernels._GEOMETRY_LIMIT + 1)]
    for key in keys:
        kernels._remember(cache, key, key[0])
    assert len(cache) == 1 and cache[keys[-1]] == keys[-1][0]


def test_ffi_stream_context_follows_torch_current_stream():
    """The launchers' FFI stream context sets the environment stream of the
    current device to the current torch stream and restores it, exactly as
    ``tvm_ffi.use_torch_stream()`` does, and launches capture and replay
    bitwise under it."""
    require_gpu()
    import tvm_ffi

    device = tvm_ffi.device("cuda", torch.cuda.current_device())
    before = tvm_ffi.get_raw_stream(device)
    with kernels._ffi_stream():
        assert tvm_ffi.get_raw_stream(device) == torch.cuda.current_stream().cuda_stream
    assert tvm_ffi.get_raw_stream(device) == before
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        with tvm_ffi.use_torch_stream():
            reference = tvm_ffi.get_raw_stream(device)
        with kernels._ffi_stream():
            assert tvm_ffi.get_raw_stream(device) == reference == side.cuda_stream
        assert tvm_ffi.get_raw_stream(device) == before
    epilogues = MoKEpilogues(2)
    shared = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    routed = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(64, 2, device="cuda")
    expected = epilogues.forward(shared, routed, scores)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = epilogues.forward(shared, routed, scores)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, expected)


def test_kernel_launch_keeps_the_exported_abi_guard():
    """``_Kernel.launch`` rejects a missing, extra or renamed keyword with the
    ABI error and passes a correct set positionally in the exported order."""
    calls = []

    class Stub:
        def run(self, *args):
            calls.append(args)
            return "ran"

    kernel = jit._Kernel(Stub(), ("a", "b", "grid_x", "grid_y", "grid_z"))
    for kwargs in (dict(a=1), dict(a=1, b=2, c=3), dict(a=1, c=2)):
        with pytest.raises(ValueError, match="exported ABI"):
            kernel.launch(grid=(1, 1, 1), **kwargs)
    assert kernel.launch(grid=(2, 3, 4), b=2, a=1) == "ran"
    assert calls == [(1, 2, 2, 3, 4)]
    with pytest.raises(ValueError):
        kernel.launch(grid=(1, 1), a=1, b=2)


def test_select_comm_split_decision_table():
    """Host-only: the imbalance-aware split is a pure function of the observed
    per-rank loads, ``hot`` when max load >= threshold x mean load, else ``base``."""
    select_comm_split = backend.select_comm_split

    base, hot = (20, 24), (24, 28)
    assert select_comm_split([100, 100, 100, 100], base, hot, 1.5) == base  # 1.0
    assert select_comm_split([150, 100, 100, 100], base, hot, 1.5) == base  # 1.33
    assert select_comm_split([200, 100, 100, 100], base, hot, 1.5) == hot  # 1.6
    assert select_comm_split([150, 50, 50, 50], base, hot, 1.5) == hot  # 2.0
    assert select_comm_split([3, 1], base, hot, 1.5) == hot  # exactly 1.5
    assert select_comm_split([0, 0, 0, 0], base, hot, 1.5) == base  # every rank empty
    assert select_comm_split((), base, hot, 1.5) == base  # no observation yet
    assert select_comm_split([200, 100, 100, 100], base, None, 1.5) == base
    # The shipped threshold: 3x keeps the fixed split, 5.6x switches.
    assert (
        select_comm_split([300, 100, 100, 100, 100, 100, 0, 0], base, hot, 4.0) == base
    )  # 3.0
    assert (
        select_comm_split([560, 100, 100, 40, 0, 0, 0, 0], base, hot, 4.0) == hot
    )  # 5.6
    assert (
        select_comm_split([100, 0, 0, 0], base, hot, 4.0) == hot
    )  # exactly 4 with 4 ranks
    loads = torch.tensor([300, 100, 100, 100], dtype=torch.int32)
    assert select_comm_split(loads, base, hot, 1.5) == hot
    assert select_comm_split(loads, base, [24, 28], 1.5) == (24, 28)


@pytest.mark.parametrize(
    "capability,sm_count,precision,expected",
    [
        ((10, 0), 148, "bf16", (24, 28)),  # B200
        ((10, 0), 148, "mxfp8", (44, 36)),
        ((10, 0), 152, "bf16", None),  # GB200: same capability, no measured row
        ((10, 0), 152, "mxfp8", None),
        ((10, 3), 152, "bf16", None),  # GB300
        ((10, 3), 152, "mxfp8", None),
        ((10, 7), 160, "bf16", None),
        ((12, 0), 170, "bf16", None),
    ],
)
def test_imbalance_comm_sms_auto_follows_the_device(
    monkeypatch, capability, sm_count, precision, expected
):
    """Host-only: ``"auto"`` resolves through the per-device table (compute
    capability and SM count); devices and precisions without a measured row keep
    the fixed split (``None``)."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert api.resolve_imbalance_comm_sms(precision) == expected


def test_imbalance_comm_sms_defaults_keep_the_fixed_split():
    """Host-only: the config default disables the imbalance split, the facade
    default ``"auto"`` resolves the measured row of the device generation, and
    malformed arguments are rejected before any collective."""
    import inspect

    config = MoKConfig()
    assert config.imbalance_comm_sms is None
    assert config.imbalance_threshold == 4.0
    facade = inspect.signature(api.create_mok_bf16_workspace).parameters
    assert facade["imbalance_comm_sms"].default == "auto"
    assert facade["imbalance_threshold"].default == "auto"
    # The measured rows and thresholds (backend README); a change here is a
    # shipping change. The backend default is the BF16 threshold.
    assert api.IMBALANCE_COMM_SMS == {
        (10, 0): {148: {"bf16": (24, 28), "mxfp8": (44, 36)}}
    }
    assert api.IMBALANCE_THRESHOLDS == {"bf16": 4.0, "mxfp8": 2.7}
    assert api.IMBALANCE_THRESHOLDS["bf16"] == api.IMBALANCE_THRESHOLD
    assert api.resolve_imbalance_threshold("bf16") == config.imbalance_threshold
    assert api.resolve_imbalance_threshold("mxfp8") == 2.7
    with pytest.raises(ValueError, match="precision must be one of"):
        api.resolve_imbalance_comm_sms("fp8")
    with pytest.raises(ValueError, match="precision must be one of"):
        api.resolve_imbalance_threshold("fp8")
    kwargs = dict(
        group=None, device=None, num_local_tokens=512, hidden_size=256, topk=2
    )
    with pytest.raises(ValueError, match='"auto"'):
        api.create_mok_bf16_workspace(imbalance_comm_sms="fast", **kwargs)
    with pytest.raises(ValueError, match='"auto"'):
        api.create_mok_bf16_workspace(imbalance_threshold="fast", **kwargs)
    with pytest.raises(ValueError, match="pair"):
        api.create_mok_bf16_workspace(imbalance_comm_sms=(24,), **kwargs)
    with pytest.raises(ValueError, match="precision must be one of"):
        api.create_mok_bf16_workspace(precision="fp8", **kwargs)


@pytest.mark.parametrize(
    "capability,sm_count,precision,ep_size,expected",
    [
        ((10, 3), 152, "mxfp8", 8, (44, 36)),  # GB300, EP8 across two nodes
        ((10, 3), 152, "mxfp8", 4, None),  # GB300 EP4: no unconditional gain
        ((10, 3), 152, "bf16", 8, None),
        ((10, 3), 160, "mxfp8", 8, None),  # other GB300 SM count: factory split
        ((10, 0), 148, "mxfp8", 8, None),  # B200: the imbalance switch instead
        ((10, 0), 152, "mxfp8", 8, None),  # GB200
        ((10, 7), 160, "mxfp8", 8, None),
    ],
)
def test_comm_sms_auto_follows_the_device_and_group(
    monkeypatch, capability, sm_count, precision, ep_size, expected
):
    """Host-only: ``"auto"`` for the fixed split resolves through the per-device,
    per-group-size table; devices, precisions and group sizes without a measured
    row keep the factory split (``None``)."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert api.resolve_comm_sms(precision, ep_size) == expected


def test_comm_sms_factory_defaults_are_unchanged():
    """Host-only: the factory split stays 40 / 40, the measured rows are the
    shipped table (a change here is a shipping change) and malformed arguments
    are rejected before any collective."""
    import inspect

    facade = inspect.signature(api.create_mok_bf16_workspace).parameters
    assert facade["fwd_num_comm_sms"].default == 40
    assert facade["bwd_num_comm_sms"].default == 40
    assert api.COMM_SMS == {(10, 3): {152: {"mxfp8": {8: (44, 36)}}}}
    with pytest.raises(ValueError, match="precision must be one of"):
        api.resolve_comm_sms("fp8", 8)
    with pytest.raises(ValueError, match="ep_size must be a positive"):
        api.resolve_comm_sms("mxfp8", 0)
    with pytest.raises(ValueError, match='both be "auto"'):
        api.create_mok_bf16_workspace(
            group=None,
            device=None,
            num_local_tokens=512,
            hidden_size=512,
            topk=2,
            fwd_num_comm_sms="auto",
            bwd_num_comm_sms=40,
        )


def test_comm_split_log_prints_each_split_change(capsys):
    """``MoKConfig.comm_split_log`` prints one stderr line per rank each time the
    imbalance-aware split of the next launch changes; the default is silent and
    the switch never changes the decision itself."""
    require_gpu()
    import dataclasses
    from types import SimpleNamespace

    assert MoKConfig().comm_split_log is False
    quiet = MoKConfig(
        fwd_num_comm_sms=4,
        bwd_num_comm_sms=4,
        imbalance_comm_sms=(6, 6),
        imbalance_threshold=1.5,
    )
    loud = dataclasses.replace(quiet, comm_split_log=True)
    assert loud.comm_split_log is True
    ep_size, experts = 4, 4
    functional = backend.MoKFunctional(ep_size, experts, 2)
    # Gathered expert ids for the (4, 4) diagnostic layout, [EP, 256, 2]: every
    # expert receives the same number of routes (every rank owns 512 routed
    # rows, max/mean load 1), then the same ids all on rank 0's experts
    # (max/mean load 4, above the threshold). The observation reads the
    # owner-rank bins the scheduler's count kernel writes.
    balanced = (
        torch.arange(ep_size * 256 * 2, device="cuda", dtype=torch.int32)
        % (ep_size * experts)
    ).view(ep_size, 256, 2)
    skewed = balanced % experts
    capacity = ep_size * 256 * 2 + experts * 256
    prefix = "[cake_mok_bf16 rank 0] communication split -> "
    expected_lines = [
        prefix + "(4, 4) (max/mean routed load 1.000)",
        prefix + "(6, 6) (max/mean routed load 4.000)",
    ]
    workspaces = []  # keep every stand-in alive: the adapter keys its state by id
    for config, expected in ((quiet, []), (loud, expected_lines)):
        workspace = SimpleNamespace(ep_rank=0, ep_size=ep_size)
        workspaces.append(workspace)
        capsys.readouterr()
        splits = []
        for routes in (balanced, skewed):
            functional.scheduler(routes, capacity, 0)
            functional._observe_load(workspace, config)
            torch.cuda.synchronize()
            splits.append(functional._comm_split(config, workspace))
            # The same observation again: the split is unchanged, so no line.
            splits.append(functional._comm_split(config, workspace))
        assert splits == [(4, 4), (4, 4), (6, 6), (6, 6)]
        lines = [
            line
            for line in capsys.readouterr().err.splitlines()
            if "communication split" in line
        ]
        assert lines == expected
