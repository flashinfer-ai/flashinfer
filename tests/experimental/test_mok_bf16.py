# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic BF16 training checks, including later-ring recomputation."""

import math

import pytest
import torch

from flashinfer.experimental.cake_mok_bf16._kernels import (
    MoKForward,
    MoKBackward,
    MoKScheduler,
    MoKEpilogues,
)


def _require_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("Requires an SM100a-compatible CUDA device")


@pytest.mark.parametrize("variant", ["base", "ragged", "clamped", "clamped_boundary"])
def test_fused_training(monkeypatch, variant):
    _require_gpu()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    _check_fused_training(variant)


# (hidden, intermediate, macrobatch, source tokens, swiglu_limit). ``ragged``
# uses odd, non-tile-aligned source counts (including one row); ``clamped``
# uses limits that mask a large share of the gate/up elements;
# ``clamped_boundary`` pins gate/up elements exactly at the GLM-5.3-Flash limit
# L = 10, one BF16 ulp inside and outside it, and at 0 (unit-vector gate/up
# weights make the pre-activations equal to selected input entries exactly).
GEOMETRIES = {
    "base": (
        (256, 256, 768, 512, None),
        (512, 512, 768, 512, None),
        (256, 256, 1536, 512, None),
    ),
    "ragged": (
        (256, 256, 768, 501, None),
        (512, 512, 768, 129, None),
        (256, 256, 1536, 1, None),
    ),
    "clamped": (
        (256, 256, 768, 512, 0.08),
        (512, 512, 768, 501, 0.1),
        (256, 256, 1536, 512, 10.0),
    ),
    "clamped_boundary": ((256, 256, 768, 512, 10.0),),
}


def boundary_levels(limit, device):
    """``[-L-ulp, -L, -L+ulp, 0, L-ulp, L, L+ulp]`` as BF16 (ulp = BF16 spacing at L);
    every level must be exactly representable."""
    ulp = torch.finfo(torch.bfloat16).eps * 2.0 ** math.floor(math.log2(limit))
    values = [-limit - ulp, -limit, -limit + ulp, 0.0, limit - ulp, limit, limit + ulp]
    levels = torch.tensor(values, dtype=torch.bfloat16, device=device)
    assert torch.equal(levels.float(), torch.tensor(values, device=device)), values
    return levels


def pin_clamp_boundary(x, shared, routed, limit):
    """Unit-vector gate/up rows: ``gate[:, j] = x[:, (j - e) % hidden]`` exactly (one
    BF16 product, FP32 accumulation), so input entries at the boundary levels put
    gate/up exactly at +-L, one ulp inside/outside and 0 on the routed and shared
    experts; half of the input entries are pinned, the down projections stay random."""
    hidden = x.shape[1]
    assert shared[0].shape == (hidden, hidden), "needs intermediate == hidden"
    eye = torch.eye(hidden, dtype=x.dtype, device=x.device)
    shared[0], shared[1] = eye, eye.roll(1, 0)
    experts = routed[0].shape[0]
    routed[0] = torch.stack([eye.roll(e, 0) for e in range(experts)])
    routed[1] = torch.stack([eye.roll(e + 1, 0) for e in range(experts)])
    levels = boundary_levels(limit, x.device)
    pinned = torch.rand(x.shape, device=x.device) < 0.5
    draws = torch.randint(levels.numel(), (int(pinned.sum()),), device=x.device)
    x[pinned] = levels[draws]


def _check_fused_training(variant="base"):
    print("build fused backward", flush=True)
    backward = MoKBackward()
    forward = MoKForward()
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3), (10, 7))
    records = []
    for empty in (False, True):
        for hidden, intermediate, macro, tokens, swiglu_limit in GEOMETRIES[variant]:
            experts, topk, mini, comm = (4, 2, 256, 4)
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
            if variant == "clamped_boundary":
                pin_clamp_boundary(x, shared, routed, swiglu_limit)
            scores = torch.rand(tokens, topk, device="cuda") + 0.1
            scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
            row = torch.arange(tokens, device="cuda")
            routes = torch.stack(
                [(row + slot) % (topk if empty else experts) for slot in range(topk)],
                -1,
            )
            slots = [
                torch.where(routes.flatten() == expert)[0] for expert in range(experts)
            ]
            count_values = [((value.numel() + 255) // 256) * 256 for value in slots]
            schedule = torch.cat(
                [
                    torch.cat(
                        (
                            value.int(),
                            torch.full(
                                (count - value.numel(),),
                                -1,
                                dtype=torch.int32,
                                device="cuda",
                            ),
                        )
                    )
                    for value, count in zip(slots, count_values, strict=True)
                ]
            )
            actual_tokens = schedule.numel()
            schedule = torch.cat(
                (schedule, torch.full((512,), -1, dtype=torch.int32, device="cuda"))
            )
            peers = torch.where(schedule >= 0, 0, -1).int()
            num_tokens = torch.tensor([actual_tokens], device="cuda", dtype=torch.int32)
            counts = torch.tensor(count_values, device="cuda", dtype=torch.int32)
            combine, dx_peer = (
                torch.empty(tokens * topk, hidden, **options),
                torch.empty(tokens * topk, hidden, **options),
            )
            ds_peer = torch.empty_like(scores)
            forward_context = [None]

            def sequence():
                context = forward(
                    x,
                    [x.data_ptr()],
                    combine,
                    [combine.data_ptr()],
                    shared[0],
                    routed[0],
                    shared[1],
                    routed[1],
                    shared[2],
                    routed[2],
                    peers,
                    schedule,
                    num_tokens,
                    counts,
                    topk,
                    swiglu_limit,
                    comm,
                    macro,
                    mini,
                )
                forward_context[0] = context
                return backward(
                    dy,
                    [dy.data_ptr()],
                    dx_peer,
                    [dx_peer.data_ptr()],
                    scores,
                    [scores.data_ptr()],
                    ds_peer,
                    [ds_peer.data_ptr()],
                    shared[0],
                    routed[0],
                    shared[1],
                    routed[1],
                    shared[2],
                    routed[2],
                    *context[:7],
                    x,
                    [x.data_ptr()],
                    peers,
                    schedule,
                    num_tokens,
                    counts,
                    topk,
                    swiglu_limit,
                    comm,
                    macro,
                    mini,
                )

            def mlp_backward(value, gradient, weights):
                # Independent BF16-rounding-point reference; the clamp follows
                # MoK: pre-clamp inclusive masks, clamped values in the SiLU.
                # Routed experts receive the BF16 score-scaled upstream gradient
                # (the dispatched ring); the kernel rounds dh once.
                gate = (value.float() @ weights[0].float().T).bfloat16()
                up = (value.float() @ weights[1].float().T).bfloat16()
                gate_f, up_f = gate.float(), up.float()
                if swiglu_limit is not None:
                    gate_mask = gate_f <= swiglu_limit
                    up_mask = (up_f >= -swiglu_limit) & (up_f <= swiglu_limit)
                    gate_f = torch.clamp(gate_f, max=swiglu_limit)
                    up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
                sigmoid = torch.sigmoid(gate_f)
                silu = gate_f * sigmoid
                activation_unrounded = silu * up_f
                activation = activation_unrounded.bfloat16()
                dh = (gradient.float() @ weights[2].float()).bfloat16()
                dh_f = dh.float()
                dg_f = ((1.0 - silu) * sigmoid + silu) * up_f * dh_f
                du_f = silu * dh_f
                if swiglu_limit is not None:
                    dg_f = torch.where(gate_mask, dg_f, torch.zeros_like(dg_f))
                    du_f = torch.where(up_mask, du_f, torch.zeros_like(du_f))
                dg, du = dg_f.bfloat16(), du_f.bfloat16()
                dx = (
                    dg.float() @ weights[0].float() + du.float() @ weights[1].float()
                ).bfloat16()
                return dx, dg, du, dh, activation, activation_unrounded

            def reference():
                sx, sg, su, sh, activation, _ = mlp_backward(x, dy, shared)
                dw_shared = [
                    (sg.float().T @ x.float()).bfloat16(),
                    (su.float().T @ x.float()).bfloat16(),
                    (dy.float().T @ activation.float()).bfloat16(),
                ]
                dw_routed = [torch.zeros_like(weight) for weight in routed]
                dx_reference, ds_reference = (
                    torch.empty_like(dx_peer),
                    torch.empty_like(scores).flatten(),
                )
                y_routed_reference = torch.empty_like(combine)
                offset = 0
                for expert, selected in enumerate(slots):
                    if selected.numel():
                        value = x[selected // topk]
                        score = scores.flatten()[selected]
                        gradient = dy[selected // topk]
                        # The backward dispatch publishes the BF16 score-scaled dy
                        # once (MoK SCALE_ROWS); every routed gradient derives from it.
                        scaled = (gradient.float() * score[:, None]).bfloat16()
                        rx, rg, ru, rh, rhidden, unrounded = mlp_backward(
                            value, scaled, [w[expert] for w in routed]
                        )
                        y_routed_reference[selected] = (
                            rhidden.float() @ routed[2][expert].float().T
                        ).bfloat16()
                        dx_reference[selected] = rx
                        ds_reference[selected] = (
                            y_routed_reference[selected].float()
                            * dy[selected // topk].float()
                        ).sum(-1)
                        first = True
                        for start in range(0, actual_tokens, macro):
                            lo, hi = (
                                max(offset, start) - offset,
                                min(offset + selected.numel(), start + macro) - offset,
                            )
                            if lo < hi:
                                products = [
                                    (
                                        rg[lo:hi].float().T @ value[lo:hi].float()
                                    ).bfloat16(),
                                    (
                                        ru[lo:hi].float().T @ value[lo:hi].float()
                                    ).bfloat16(),
                                    (
                                        scaled[lo:hi].float().T @ rhidden[lo:hi].float()
                                    ).bfloat16(),
                                ]
                                for destination, product in zip(
                                    dw_routed, products, strict=True
                                ):
                                    destination[expert] = (
                                        product
                                        if first
                                        else (
                                            destination[expert].float()
                                            + product.float()
                                        ).bfloat16()
                                    )
                                first = False
                    offset += count_values[expert]
                return {
                    0: sx,
                    2: sg,
                    4: su,
                    6: sh,
                    9: dw_shared[0],
                    10: dw_routed[0],
                    11: dw_shared[1],
                    12: dw_routed[1],
                    13: dw_shared[2],
                    14: dw_routed[2],
                    15: dx_reference,
                    16: ds_reference.reshape_as(scores),
                    17: (activation.float() @ shared[2].float().T).bfloat16(),
                    18: y_routed_reference,
                }

            def outputs(result):
                return (*result, dx_peer, ds_peer, forward_context[0][7], combine)

            worst = {}

            def check(result):
                actual = outputs(result)
                for index, expected in reference().items():
                    value = actual[index]
                    assert torch.isfinite(value).all().item(), (
                        empty,
                        hidden,
                        index,
                        "nonfinite",
                    )
                    delta = (value.float() - expected.float()).abs()
                    maximum, norm = (
                        delta.max().item(),
                        expected.float().abs().sum().item(),
                    )
                    relative = (
                        delta.sum().item() / norm
                        if norm
                        else (0.0 if delta.sum().item() == 0 else float("inf"))
                    )
                    torch.testing.assert_close(
                        value.float(), expected.float(), atol=1e-2, rtol=1e-2
                    )
                    previous = worst.setdefault(
                        index, dict(max_abs=0.0, relative_l1=0.0)
                    )
                    previous["max_abs"] = max(previous["max_abs"], maximum)
                    previous["relative_l1"] = max(previous["relative_l1"], relative)

            print(
                dict(
                    phase="execute",
                    empty=empty,
                    hidden=hidden,
                    intermediate=intermediate,
                    macro=macro,
                    tokens=tokens,
                    swiglu_limit=swiglu_limit,
                ),
                flush=True,
            )
            saved = []
            compared = (0, 2, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
            for _ in range(3):
                result = sequence()
                check(result)
                values = outputs(result)
                saved.append(tuple(values[index].clone() for index in compared))
            assert all(
                all(torch.equal(a, b) for a, b in zip(saved[0], values, strict=True))
                for values in saved[1:]
            )
            # Context-only recompute (MoK ``recompute_forward_context``): the
            # rebuilt context and a backward from it are bitwise identical. The
            # backward replays later macrobatches into the context rings, so the
            # comparison uses a fresh forward context, not a consumed one.
            fresh = forward(
                x,
                [x.data_ptr()],
                combine,
                [combine.data_ptr()],
                shared[0],
                routed[0],
                shared[1],
                routed[1],
                shared[2],
                routed[2],
                peers,
                schedule,
                num_tokens,
                counts,
                topk,
                swiglu_limit,
                comm,
                macro,
                mini,
            )
            recomputed = forward(
                x,
                [x.data_ptr()],
                combine,
                [combine.data_ptr()],
                shared[0],
                routed[0],
                shared[1],
                routed[1],
                shared[2],
                routed[2],
                peers,
                schedule,
                num_tokens,
                counts,
                topk,
                swiglu_limit,
                comm,
                macro,
                mini,
                recompute_only=True,
            )
            assert recomputed[7] is None and recomputed[8] is None
            # Defined context rows: the routed rings hold the retained macrobatch
            # (min(macro, routed rows)) and the shared activations the real source
            # rows; neither path writes the rows beyond (allocator contents).
            routed_valid = min(macro, actual_tokens)
            for index, (a, b) in enumerate(zip(recomputed[:7], fresh[:7], strict=True)):
                valid = tokens if index in (1, 3, 5) else routed_valid
                assert torch.equal(a[:valid], b[:valid]), (
                    f"recomputed context tensor {index} differs"
                )
            result = backward(
                dy,
                [dy.data_ptr()],
                dx_peer,
                [dx_peer.data_ptr()],
                scores,
                [scores.data_ptr()],
                ds_peer,
                [ds_peer.data_ptr()],
                shared[0],
                routed[0],
                shared[1],
                routed[1],
                shared[2],
                routed[2],
                *recomputed[:7],
                x,
                [x.data_ptr()],
                peers,
                schedule,
                num_tokens,
                counts,
                topk,
                swiglu_limit,
                comm,
                macro,
                mini,
            )
            check(result)
            values = outputs(result)
            assert all(
                torch.equal(a, values[index])
                for a, index in zip(saved[0], compared, strict=True)
            ), "recomputed-context backward differs"
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = sequence()
            graph_saved = []
            for _ in range(3):
                graph.replay()
                check(result)
                values = outputs(result)
                graph_saved.append(tuple(values[index].clone() for index in compared))
            assert all(
                all(
                    torch.equal(a, b)
                    for a, b in zip(graph_saved[0], values, strict=True)
                )
                for values in graph_saved[1:]
            )
            # d(score) = dot(BF16 expert output, original upstream gradient).
            # Changing only that score leaves its own gradient unchanged up to
            # the BF16 rounding of the score-scaled upstream gradient the
            # backward dispatches (MoK form): BF16 rule, not bitwise. A
            # non-power-of-two change exercises the scale/round/unscale path.
            router_gradient = ds_peer.clone()
            scores.add_(0.03125)
            graph.replay()
            check(result)
            torch.testing.assert_close(
                ds_peer.float(), router_gradient.float(), atol=1e-2, rtol=1e-2
            )
            for _ in range(2):
                x.normal_(std=0.125)
                dy.normal_(std=0.125)
                scores.uniform_(0.1, 1.0)
                scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
                graph.replay()
                check(result)
            record = dict(
                empty=empty,
                hidden=hidden,
                intermediate=intermediate,
                macro=macro,
                errors=worst,
                tokens=tokens,
                swiglu_limit=swiglu_limit,
                variant=variant,
                experts=experts,
                topk=topk,
                mini=mini,
                comm_sms=comm,
                actual_routed_tokens=actual_tokens,
                capacity=schedule.numel(),
                three_eager_executions_bitwise_equal=True,
                three_graph_replays_bitwise_equal=True,
                recompute_context_and_backward_bitwise_equal=True,
                router_gradient_score_invariance="bf16_rule",
                changed_inputs_scores_upstream_gradients=2,
            )
            print(record, flush=True)
            records.append(record)
    result = dict(
        status="PASS",
        passed=True,
        scope="single-GPU fused forward/backward diagnostic",
        cases=records,
    )
    return result


@pytest.mark.parametrize("ep,experts,topk", [(1, 4, 2), (4, 4, 2), (16, 16, 8)])
def test_scheduler(ep, experts, topk):
    _require_gpu()
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
    capacity = ep * tokens * topk + experts * 256
    for rank in {0, ep - 1}:
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        expected_counts = torch.bincount(ids.flatten().long(), minlength=ep * experts)[
            rank * experts : (rank + 1) * experts
        ]
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
def test_epilogues(topk):
    _require_gpu()
    epilogues = MoKEpilogues(topk)
    shared = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) * 0.125
    routed = torch.randn(512 * topk, 256, device="cuda", dtype=torch.bfloat16) * 0.125
    scores = torch.rand(512, topk, device="cuda")
    expected = (
        shared.float()
        + (routed.float().view(512, topk, 256) * scores[..., None]).sum(1)
    ).bfloat16()
    torch.testing.assert_close(
        epilogues.forward(shared, routed, scores), expected, atol=0.01, rtol=0.01
    )
    expected_dx = (
        shared.float() + routed.float().view(512, topk, 256).sum(1)
    ).bfloat16()
    torch.testing.assert_close(
        epilogues.backward(shared, routed), expected_dx, atol=0.01, rtol=0.01
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_comm_sms_defaults_follow_the_device_generation():
    """The default communication-SM split comes from the row of the current device's
    compute capability, or from the B200 row when that generation has no row."""
    from flashinfer import mok

    cc = tuple(torch.cuda.get_device_capability(0))
    expected = mok.COMM_SMS_DEFAULTS.get(cc, mok.COMM_SMS_DEFAULTS[(10, 0)])
    for precision in ("bf16", "mxfp8"):
        fwd, bwd = mok.comm_sms_defaults(precision, torch.device("cuda", 0))
        assert (fwd, bwd) == expected[precision]
        assert fwd > 0 and bwd > 0
    assert mok.comm_sms_defaults("bf16") == expected["bf16"]
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.comm_sms_defaults("fp8")
    for row in mok.COMM_SMS_DEFAULTS.values():
        assert set(row) == {"bf16", "mxfp8"}
    # The measured per-generation rows (backend README); a change here is a shipping change.
    assert mok.COMM_SMS_DEFAULTS[(10, 0)] == {"bf16": (20, 24), "mxfp8": (40, 32)}
    assert mok.COMM_SMS_DEFAULTS[(10, 3)] == {"bf16": (24, 28), "mxfp8": (40, 32)}
    assert mok.COMM_SMS_DEFAULTS[(10, 7)] == {"bf16": (40, 32), "mxfp8": (72, 64)}
