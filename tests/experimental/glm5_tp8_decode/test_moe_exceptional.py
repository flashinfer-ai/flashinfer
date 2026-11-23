# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Finite inputs may overflow internally; readiness must not confuse NaN payloads.

These cases check completion and the specified exceptional-result classification.
Ordinary calls afterwards must still pass every finite stage and wire check.
"""

import time

import pytest
import torch

from . import _api, data
from . import moe_reference as ref
from ._moe_case import MoECase
from ._records import moe_previous_sender

pytestmark = pytest.mark.node_exclusive


def _exceptional_weights(device, rank, kind):
    # Every bank aliases the same full-shape tensors; the packer only reads them.
    zero = torch.zeros((2048, 6144), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    gate = (
        zero
        if kind != "gate_overflow"
        else torch.full((2048, 6144), 0x7E, dtype=torch.uint8, device=device).view(
            torch.float8_e4m3fn
        )
    )
    scales = torch.ones((16, 48), dtype=torch.float32, device=device)
    gate_scales = (
        scales
        if kind != "gate_overflow"
        else torch.full_like(scales, torch.finfo(torch.float32).max)
    )
    down = torch.zeros((6144, 2048), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    down_scales = torch.ones((48, 16), dtype=torch.float32, device=device)
    layer = {
        "post_attention_layernorm.weight": torch.full(
            (6144,),
            2.0**120 if kind == "norm_overflow" else 1.0,
            dtype=torch.float32,
            device=device,
        ),
        "mlp.gate.weight": torch.full(
            (256, 6144),
            -torch.finfo(torch.bfloat16).max if kind == "zero_routing_mass" else 0,
            dtype=torch.bfloat16,
            device=device,
        ),
        "mlp.gate.e_score_correction_bias": torch.zeros(
            256, dtype=torch.float32, device=device
        ),
    }
    for prefix in data.MOE_BANKS:
        for name, weight, scale in (
            ("gate_proj", gate, gate_scales),
            ("up_proj", zero, scales),
            ("down_proj", down, down_scales),
        ):
            layer[f"{prefix}.{name}.weight"] = weight
            layer[f"{prefix}.{name}.weight_scale_inv"] = scale
    # Inputs are finite by construction. No live packed weight is ever mutated.
    return _api.pack_moe(layer, rank)


def _require_completion(control, stream, label, seconds=60, launch_error=None):
    """Fail on every rank unless all ranks observe completion within ``seconds``.

    The wait polls an event instead of synchronizing, so a kernel that never
    finishes fails the test rather than blocking the process.
    """
    event = torch.cuda.Event()
    complete, error = False, launch_error
    try:
        if error is None:
            event.record(stream)
            deadline = time.monotonic() + seconds
            while not event.query() and time.monotonic() < deadline:
                time.sleep(0.005)
            complete = event.query()
    except RuntimeError as exc:
        error = str(exc)
    observed = control.gather((complete, error))
    if not all(done and failure is None for done, failure in observed):
        reason = (
            f"{label}: GPU completion not confirmed within {seconds}s; "
            f"(complete, error) by rank={observed}"
        )
        control.mark_incomplete(reason)
        raise AssertionError(reason)


@pytest.mark.parametrize(
    "kind", ["norm_overflow", "gate_overflow", "zero_routing_mass"]
)
def test_finite_overflow_completes_and_recovers_without_reset(
    collective, collective_watchdog, kind
):
    case = MoECase(collective, weight_seed=847223)
    graphs, plans, inputs, outputs, storage = {}, {}, {}, {}, {}
    packed = None
    try:
        case.run(4, 1)
        case.run(1, 7)
        case.capture()
        packed = collective.checked(
            lambda: _exceptional_weights(collective.device, collective.rank, kind)
        )
        for q in (1, 4):
            inputs[q] = torch.zeros(
                (q, 6144), dtype=torch.bfloat16, device=collective.device
            )
            storage[q] = torch.full(
                (q + 1, 6144), 17, dtype=torch.bfloat16, device=collective.device
            )
            outputs[q] = storage[q][:q]
            plans[q] = _api.prepare_moe(inputs[q], packed, outputs[q], case.ws)
        collective.sync()
        for q in (1, 4):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=case.stream):
                plans[q]()
            graphs[q] = graph
            del graph
        collective.sync()

        def run(q, *, exceptional, graph):
            # The same immutable packed weights and bound input addresses cover
            # both exceptional arithmetic and ordinary zero-expert recovery.
            bad = 0.0 if kind == "norm_overflow" else 1.0
            value = bad if exceptional else 1.0 - bad
            launch_error = None
            try:
                with torch.cuda.stream(case.stream):
                    inputs[q].fill_(value)
                    storage[q].fill_(17)
                    if graph:
                        graphs[q].replay()
                    else:
                        plans[q]()
            except RuntimeError as exc:
                launch_error = str(exc)
            case.counts[q] += 1
            _require_completion(
                collective, case.stream, f"{kind} q{q}", launch_error=launch_error
            )

            def check():
                plan = plans[q]
                generation = int(plan._diagnostics["generation"]())
                assert generation == case.counts[q]
                assert bool((inputs[q] == value).all())
                assert bool((storage[q][q:] == 17).all())
                logits = ref.kernel_logits(plan._diagnostics["scores"], q).contiguous()
                assert not bool((logits.view(torch.int32) == -1).any()), (
                    "pending router word survived completion"
                )
                mid = plan._diagnostics["mid"][:q]
                assert not bool((mid.view(torch.int16) == -1).any()), (
                    "pending BF16 word survived completion"
                )
                _, partial = moe_previous_sender(
                    plan._diagnostics["records"], q, collective.rank, generation
                )
                if exceptional:
                    if kind == "zero_routing_mass":
                        assert bool(torch.isneginf(logits).all())
                        assert bool((torch.sigmoid(logits) == 0).all())
                    assert bool(torch.isnan(outputs[q]).all()), (
                        "expected NaN propagation after finite-input overflow"
                    )
                    assert bool(torch.isnan(partial).all())
                else:
                    assert bool(torch.isfinite(outputs[q]).all())
                    assert torch.equal(outputs[q], inputs[q]), (
                        "zero experts must recover the residual"
                    )
                return generation

            generation = collective.checked(check)
            assert collective.gather(generation) == [generation] * 8

        for q in (1, 4):
            run(q, exceptional=True, graph=False)
            run(q, exceptional=False, graph=True)
            run(q, exceptional=True, graph=True)
            # A distinct ordinary plan reuses the same workspace and counters;
            # it must pass every stage, routing and exact-wire check.
            case.run(q, 101 + q, graph=True)
    finally:
        if collective.incomplete_gpu_work is not None:
            # Closing the case synchronizes the device and would block behind
            # the unfinished work; keep its buffers alive instead.
            collective.keep_alive(case, packed, plans, graphs, inputs, storage)
        else:
            graphs.clear()
            case.close()
