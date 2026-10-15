# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU BF16 MoK kernel checks against an independent BF16-rounding reference.

Covers the fused forward/backward (ring replay, empty experts, variable
widths, CUDA Graph replay), clamped SwiGLU, FP32 weight-gradient
accumulation, GLM-5.3-Flash local expert counts, context-only recompute,
every exported scheduler layout and the epilogues.
"""

import pytest
import torch

from flashinfer.experimental.cake_mok_bf16._kernels import (
    SCHEDULER_LAYOUTS,
    MoKBackward,
    MoKEpilogues,
    MoKForward,
    MoKRecompute,
    MoKScheduler,
)
from tests.experimental._mok_reference import require_gpu, single_rank_schedule

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
    for rank in {0, ep - 1}:
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        expected_counts = torch.bincount(
            ids.flatten().long().clamp(min=0), minlength=ep * experts
        )
        expected_counts[0] -= (ids == -1).sum()
        expected_counts = expected_counts[rank * experts : (rank + 1) * experts]
        expected_counts = ((expected_counts + 255) // 256) * 256
        torch.testing.assert_close(counts, expected_counts.int(), rtol=0, atol=0)
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
    from flashinfer.experimental.cake_mok_bf16.backend import count_routed_load

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


def test_select_comm_split_decision_table():
    """Host-only: the imbalance-aware split is a pure function of the observed
    per-rank loads, ``hot`` when max load >= threshold x mean load, else ``base``."""
    from flashinfer.experimental.cake_mok_bf16.backend import select_comm_split

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

    from flashinfer import mok

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert mok.resolve_imbalance_comm_sms(precision) == expected


def test_imbalance_comm_sms_defaults_keep_the_fixed_split():
    """Host-only: the config default disables the imbalance split, the facade
    default ``"auto"`` resolves the measured row of the device generation, and
    malformed arguments are rejected before any collective."""
    import inspect

    from flashinfer import mok
    from flashinfer.experimental.cake_mok_bf16.workspace import MoKConfig

    config = MoKConfig()
    assert config.imbalance_comm_sms is None
    assert config.imbalance_threshold == 4.0
    facade = inspect.signature(mok.create_mok_bf16_workspace).parameters
    assert facade["imbalance_comm_sms"].default == "auto"
    assert facade["imbalance_threshold"].default == "auto"
    # The measured rows and thresholds (backend README); a change here is a
    # shipping change. The backend default is the BF16 threshold.
    assert mok.IMBALANCE_COMM_SMS == {
        (10, 0): {148: {"bf16": (24, 28), "mxfp8": (44, 36)}}
    }
    assert mok.IMBALANCE_THRESHOLDS == {"bf16": 4.0, "mxfp8": 2.7}
    assert mok.IMBALANCE_THRESHOLDS["bf16"] == mok.IMBALANCE_THRESHOLD
    assert mok.resolve_imbalance_threshold("bf16") == config.imbalance_threshold
    assert mok.resolve_imbalance_threshold("mxfp8") == 2.7
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.resolve_imbalance_comm_sms("fp8")
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.resolve_imbalance_threshold("fp8")
    kwargs = dict(
        group=None, device=None, num_local_tokens=512, hidden_size=256, topk=2
    )
    with pytest.raises(ValueError, match='"auto"'):
        mok.create_mok_bf16_workspace(imbalance_comm_sms="fast", **kwargs)
    with pytest.raises(ValueError, match='"auto"'):
        mok.create_mok_bf16_workspace(imbalance_threshold="fast", **kwargs)
    with pytest.raises(ValueError, match="pair"):
        mok.create_mok_bf16_workspace(imbalance_comm_sms=(24,), **kwargs)
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.create_mok_bf16_workspace(precision="fp8", **kwargs)


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

    from flashinfer import mok

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert mok.resolve_comm_sms(precision, ep_size) == expected


def test_comm_sms_factory_defaults_are_unchanged():
    """Host-only: the factory split stays 40 / 40, the measured rows are the
    shipped table (a change here is a shipping change) and malformed arguments
    are rejected before any collective."""
    import inspect

    from flashinfer import mok

    facade = inspect.signature(mok.create_mok_bf16_workspace).parameters
    assert facade["fwd_num_comm_sms"].default == 40
    assert facade["bwd_num_comm_sms"].default == 40
    assert mok.COMM_SMS == {(10, 3): {152: {"mxfp8": {8: (44, 36)}}}}
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.resolve_comm_sms("fp8", 8)
    with pytest.raises(ValueError, match="ep_size must be a positive"):
        mok.resolve_comm_sms("mxfp8", 0)
    with pytest.raises(ValueError, match='both be "auto"'):
        mok.create_mok_bf16_workspace(
            group=None,
            device=None,
            num_local_tokens=512,
            hidden_size=512,
            topk=2,
            fwd_num_comm_sms="auto",
            bwd_num_comm_sms=40,
        )
