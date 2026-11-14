# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Native A16 BI plans, batch placements, masked graphs and remote oracles."""

from dataclasses import replace
import os

import pytest

from ._megamoe_bi_helpers import assert_bits, assert_reference
from .test_nvfp4_megamoe_batch_invariant import _device, _layer, _problem
from .test_nvfp4_megamoe_batch_invariant import _ieee_matmul  # noqa: F401

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("_ieee_matmul")]
CLASSES = (
    ("w4a16", False),
    ("w4a16", True),
    ("w8a16_e4m3", True),
    ("w8a16_e5m2", True),
)
BATCH_CASES = tuple((kind, early, {}) for kind, early in CLASSES) + (
    ("w4a16", True, {"swiglu_alpha": 1.702, "swiglu_beta": 1.0}),
    ("w8a16_e4m3", True, {"activation": "situ", "situ_beta": 1.25}),
    (
        "w8a16_e5m2",
        True,
        {"activation": "situ", "situ_beta": 1.25, "situ_linear_beta": 2.0},
    ),
)


def _inputs(kind, rank, world, hidden=384, intermediate=192):
    import torch
    from flashinfer.moe_ep import PrequantizedMoEWeights

    x, scores, ids, _ = _problem(rank, world, hidden, intermediate)
    generator = torch.Generator(device="cuda").manual_seed(2935 + rank)
    weights = []
    for rows, columns in ((2 * intermediate, hidden), (hidden, intermediate)):
        if kind == "w4a16":
            data = torch.randint(
                256,
                (4, rows, columns // 2),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            )
            scale = torch.randint(
                0x08,
                0x20,
                (4, rows, columns // 16),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            ).view(torch.float8_e4m3fn)
        else:
            dtype = torch.float8_e4m3fn if kind.endswith("e4m3") else torch.float8_e5m2
            data = torch.randn(4, rows, columns, device="cuda", generator=generator).to(
                dtype
            )
            scale = torch.randint(
                121,
                124,
                (4, rows, columns // 32),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            ).view(torch.float8_e8m0fnu)
        weights.append((data, scale))
    return (
        x,
        scores,
        ids,
        PrequantizedMoEWeights(
            w13=weights[0][0],
            w13_scale=weights[0][1],
            w2=weights[1][0],
            w2_scale=weights[1][1],
        ),
    )


def _config(kind, early, recipe=None, **kwargs):
    from flashinfer import moe_ep

    common = dict(
        intermediate_size=192,
        top_k=3,
        batch_invariant=True,
        knobs={},
        **(recipe or {}),
        **kwargs,
    )
    if kind == "w4a16":
        return moe_ep.Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            **common, apply_routing_weights_before_fc2=early
        )
    return moe_ep.Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        **common, kind="bf16_mxfp8_" + kind.removeprefix("w8a16_")
    )


def _decoded(weights, kind):
    if kind == "w4a16":
        from .test_helpers.w4a16_reference import _dequantize

        return (
            _dequantize(weights.w13, weights.w13_scale),
            _dequantize(weights.w2, weights.w2_scale),
        )
    return tuple(
        (data.float() * scale.float().repeat_interleave(32, -1)).bfloat16()
        for data, scale in (
            (weights.w13, weights.w13_scale),
            (weights.w2, weights.w2_scale),
        )
    )


def _reference(x, ids, scores, plain, config, early):
    """Independent Torch equations; exact BI is checked separately against plans."""
    import torch

    w13, w2 = plain
    result = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        partial = torch.zeros_like(result, dtype=torch.bfloat16)
        for expert in range(w13.shape[0]):
            selected = ids[:, slot] == expert
            rows = x[selected]
            if not rows.numel():
                continue
            fc1 = rows.float() @ w13[expert].float().T
            if getattr(config, "fc1_alpha", None) is not None:
                fc1 = fc1 * config.fc1_alpha[expert]
            gate, up = fc1.chunk(2, -1)
            clamp = config.gate_up_clamp
            if clamp is not None:
                gate, up = gate.clamp_max(abs(clamp)), up.clamp(-abs(clamp), abs(clamp))
            if config.activation == "situ":
                gate_value = (
                    config.situ_beta
                    * torch.tanh(gate / config.situ_beta)
                    * torch.sigmoid(gate)
                )
                if config.situ_linear_beta is not None:
                    up = config.situ_linear_beta * torch.tanh(
                        up / config.situ_linear_beta
                    )
                activation = up * gate_value
            elif config.swiglu_alpha is not None:
                activation = (
                    (up + config.swiglu_beta)
                    * gate
                    * torch.sigmoid(gate * config.swiglu_alpha)
                )
            else:
                activation = up * (gate * torch.sigmoid(gate))
            if early:
                activation = activation * scores[selected, slot, None]
            if getattr(config, "fc1_norm_const", None) is not None:
                activation = activation * config.fc1_norm_const[expert]
            fc2 = activation.bfloat16().float() @ w2[expert].float().T
            if getattr(config, "fc2_alpha", None) is not None:
                fc2 = fc2 * config.fc2_alpha[expert]
            partial[selected] = fc2.bfloat16()
        value = partial.float()
        if not early:
            valid = (ids[:, slot] >= 0) & (ids[:, slot] < w13.shape[0])
            value = value * torch.where(valid, scores[:, slot], 0)[:, None]
        result = result + value
    return result.bfloat16()


@pytest.mark.parametrize("kind,early,recipe", BATCH_CASES)
def test_a16_batch_invariant_batches_plans_and_graphs(
    kind, early, recipe, monkeypatch, request
):
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank case")
    _device("standard")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    x, scores, ids, weights = _inputs(kind, 0, 1)
    config = _config(kind, early, recipe)
    with _layer(config, weights, 0, 1, 257, 384) as layer:
        anchor = layer.forward(MoEEpTensors(x, ids, scores)).clone()
        receipt = layer.batch_invariant_fingerprint
    assert_reference(
        anchor[:33],
        _reference(
            x[:33], ids[:33], scores[:33], _decoded(weights, kind), config, early
        ),
        0.02,
        "A16 independent oracle",
    )
    full = request.config.getoption("--full")
    plans = (("static", "epi_warps"), ("atomic_counter", "reuse_dispatch_warps"))
    if full:
        plans += (("atomic_counter", "epi_warps"), ("static", "reuse_dispatch_warps"))
    for schedule, carrier in plans:
        knobs = dict(
            load_balance_mode=schedule,
            token_back_mode=carrier,
            flag_batch=1,
            epi_flag_batch=(1, 1),
        )
        for capacity in (17, 129) if full else (17,):
            with _layer(
                replace(config, knobs=knobs), weights, 0, 1, capacity, 384
            ) as layer:
                assert layer.batch_invariant_fingerprint == receipt
                for n in sorted({0, 1, 17, capacity}):
                    order = torch.arange(n, device="cuda").flip(0).remainder(7)
                    assert_bits(
                        layer.forward(
                            MoEEpTensors(x[order], ids[order], scores[order])
                        ),
                        anchor[order],
                        "A16 batch/plan",
                    )
                order = torch.arange(capacity, device="cuda")
                output = torch.empty_like(x[:capacity])
                inputs = MoEEpTensors(
                    x[:capacity].clone(),
                    ids[:capacity].clone(),
                    scores[:capacity].clone(),
                    output=output,
                )
                layer.forward(inputs)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        layer.forward(inputs)
                    for selected in (order.flip(0), order.remainder(5), order):
                        inputs.hidden_states.copy_(x[selected])
                        inputs.topk_ids.copy_(ids[selected])
                        inputs.topk_weights.copy_(scores[selected])
                        graph.replay()
                        assert_bits(output, anchor[selected], "A16 cohort graph")
                    inputs.topk_ids.fill_(-1)
                    inputs.topk_weights.fill_(float("nan"))
                    graph.replay()
                    assert_bits(
                        output, torch.zeros_like(output), "A16 masked stale graph"
                    )
                finally:
                    graph.reset()


@pytest.mark.gpu_2
@pytest.mark.parametrize("kind,early", CLASSES)
def test_a16_batch_invariant_expert_parallel(kind, early):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from ._megamoe_bi_helpers import (
        all_gather_rows,
        distributed_reference,
        fail_stop_distributed,
        remote_routes,
    )

    world, rank = (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
    )
    if world < 2:
        pytest.skip("requires torchrun")
    bootstrap = moe_ep.BootstrapConfig(rank=rank, world_size=world)
    moe_ep.ensure_moe_ep_cuda_device(bootstrap)
    _device("standard")
    config = _config(kind, early)
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    try:
        x, scores, ids, weights = _inputs(kind, rank, world)
        ids = remote_routes(ids, rank, world, 4)
        # Gather canonical packed bytes/scales, then decode independently.
        from flashinfer.moe_ep import PrequantizedMoEWeights

        global_weights = PrequantizedMoEWeights(
            **{
                name: all_gather_rows(getattr(weights, name))
                for name in ("w13", "w13_scale", "w2", "w2_scale")
            }
        )
        plain = _decoded(global_weights, kind)
        reference = lambda a, sf, route, score: _reference(
            a, route, score, plain, config, early
        )
        distributed_reference(
            config=config,
            weights=weights,
            packed=x[:17],
            scales=None,
            ids=ids[:17],
            scores=scores[:17],
            reference=reference,
            layer_factory=_layer,
            hidden=384,
            rank=rank,
            world=world,
            tolerance=0.02,
        )
        # An empty source must still compute incoming routes owned by its
        # experts. Keep the launch capacity equal while live rows differ.
        from flashinfer.moe_ep import MoEEpTensors

        with _layer(config, weights, rank, world, 129, 384) as layer:
            anchor = layer.forward(
                MoEEpTensors(x[:129], ids[:129], scores[:129])
            ).clone()
            for n in ((0 if rank == 0 else 129), (1 if rank % 2 == 0 else 17)):
                actual = layer.forward(MoEEpTensors(x[:n], ids[:n], scores[:n]))
                assert_bits(actual, anchor[:n], "A16 uneven/empty EP")
    except BaseException as error:
        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)


@pytest.mark.parametrize("early", (False, True))
def test_w4a16_batch_invariant_expert_scales_and_handoff_norm(early, monkeypatch):
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank case")
    _device("standard")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _inputs("w4a16", 0, 1)
    config = _config(
        "w4a16",
        early,
        fc1_alpha=torch.tensor([0.75, 1.25, 0.5, 1.5], device="cuda"),
        fc2_alpha=torch.tensor([1.25, -0.75, 1.5, 0.5], device="cuda"),
        fc1_norm_const=torch.tensor([0.5, 1.0, 1.25, 0.75], device="cuda"),
    )
    with _layer(config, weights, 0, 1, 129, 384) as layer:
        anchor = layer.forward(MoEEpTensors(x[:129], ids[:129], scores[:129])).clone()
        receipt = layer.batch_invariant_fingerprint
    assert_reference(
        anchor[:17],
        _reference(
            x[:17], ids[:17], scores[:17], _decoded(weights, "w4a16"), config, early
        ),
        0.02,
        "expert scales and handoff norm",
    )
    with _layer(
        replace(
            config,
            knobs={
                "load_balance_mode": "atomic_counter",
                "token_back_mode": "reuse_dispatch_warps",
            },
        ),
        weights,
        0,
        1,
        17,
        384,
    ) as layer:
        assert layer.batch_invariant_fingerprint == receipt
        assert_bits(
            layer.forward(MoEEpTensors(x[:17], ids[:17], scores[:17])),
            anchor[:17],
            "normalized A16 batch/plan",
        )


@pytest.mark.parametrize("kind,early", (("w4a16", False), ("w8a16_e4m3", True)))
def test_a16_batch_invariant_deferred_partials(kind, early, monkeypatch):
    import torch
    from flashinfer.moe_ep import FinalizeMode, MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank case")
    _device("standard")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _inputs(kind, 0, 1)
    config = _config(kind, early, finalize_mode=FinalizeMode.NONE)
    with _layer(config, weights, 0, 1, 129, 384) as layer:
        partials, wire_weights, _ = layer.forward_unfinalized(
            MoEEpTensors(x[:129], ids[:129], scores[:129])
        )
        anchor = partials.view(129, 3, 384).clone()
        anchor_weights = wire_weights.clone()
        receipt = layer.batch_invariant_fingerprint
    with _layer(
        replace(
            config,
            knobs={
                "load_balance_mode": "atomic_counter",
                "token_back_mode": "reuse_dispatch_warps",
            },
        ),
        weights,
        0,
        1,
        17,
        384,
    ) as layer:
        assert layer.batch_invariant_fingerprint == receipt
        inputs = MoEEpTensors(x[:17].clone(), ids[:17].clone(), scores[:17].clone())
        partials, wire_weights, _ = layer.forward_unfinalized(inputs)
        assert_bits(partials.view(17, 3, 384), anchor[:17], "A16 deferred batch/plan")
        assert torch.equal(
            wire_weights.view(torch.int32), anchor_weights[:17].view(torch.int32)
        )
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                partials, wire_weights, _ = layer.forward_unfinalized(inputs)
            for selected in (
                torch.arange(17, device="cuda").flip(0),
                torch.arange(17, device="cuda").remainder(5),
            ):
                inputs.hidden_states.copy_(x[selected])
                inputs.topk_ids.copy_(ids[selected])
                inputs.topk_weights.copy_(scores[selected])
                graph.replay()
                assert_bits(
                    partials.view(17, 3, 384), anchor[selected], "A16 deferred graph"
                )
                assert torch.equal(
                    wire_weights.view(torch.int32),
                    anchor_weights[selected].view(torch.int32),
                )
            inputs.topk_ids.fill_(-1)
            inputs.topk_weights.fill_(float("nan"))
            graph.replay()
            assert torch.count_nonzero(partials.view(torch.int16)) == 0
            assert torch.count_nonzero(wire_weights.view(torch.int32)) == 0
        finally:
            graph.reset()


@pytest.mark.parametrize("kind", ("w4a16", "nvfp4"))
def test_batch_invariant_scalar_overrides_restore_live_defaults_blackwell_only(
    kind, monkeypatch
):
    import torch
    from flashinfer.moe_ep import MoEEpTensors
    from .test_nvfp4_megamoe_batch_invariant import _config as nvfp4_config

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank scalar staging regression")
    _device("legacy")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    alpha = torch.ones(4, dtype=torch.float32, device="cuda")
    norm = torch.ones_like(alpha)
    if kind == "w4a16":
        x, scores, ids, weights = _inputs(kind, 0, 1)
        config = _config(kind, True)
    else:
        x, scores, ids, weights = _problem(0, 1, 384, 192)
        config = nvfp4_config("legacy", True, 384, 192, knobs={})
    config = replace(config, fc1_alpha=None, fc2_alpha=alpha, fc1_norm_const=norm)
    inputs = MoEEpTensors(x[:17], ids[:17], scores[:17])
    zeros = torch.zeros_like(alpha)
    other_norm = torch.full_like(norm, 2.0)
    overrides = replace(
        inputs, fc1_alpha=zeros, fc2_alpha=zeros, fc1_norm_const=other_norm
    )
    with _layer(config, weights, 0, 1, 129, 384) as layer:
        receipt = layer.batch_invariant_fingerprint
        baseline = layer(inputs).clone()
        assert torch.count_nonzero(baseline) > 0
        assert torch.count_nonzero(layer(overrides)) == 0
        assert torch.equal(alpha, torch.ones_like(alpha))
        assert torch.equal(norm, torch.ones_like(norm))
        assert_bits(
            layer(inputs), baseline, "omitted scalar overrides restore defaults"
        )
        expected_norm = layer(replace(inputs, fc1_norm_const=other_norm)).clone()
        layer(inputs)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = layer(inputs)
            # A preceding invocation may have changed the shared scratch values.
            layer(overrides)
            for alpha_value, norm_value in ((1, 1), (0, 1), (1, 2), (1, 1)):
                alpha.fill_(alpha_value)
                norm.fill_(norm_value)
                graph.replay()
                expected = (
                    torch.zeros_like(baseline)
                    if not alpha_value
                    else baseline
                    if norm_value == 1
                    else expected_norm
                )
                assert_bits(output, expected, "live config scalar graph replay")
            assert layer.batch_invariant_fingerprint == receipt
        finally:
            graph.reset()


@pytest.mark.parametrize("kind,early", (("w4a16", True), ("w8a16_e4m3", True)))
def test_a16_batch_invariant_public_autotune_disk_reload(
    kind, early, tmp_path, monkeypatch
):
    import importlib

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank case")
    _device("standard")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    from ._megamoe_bi_helpers import autotune_disk_reload

    if kind == "w4a16":
        module = importlib.import_module(
            "flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4.autotune"
        )
        name = "bf16_nvfp4_candidates"
    else:
        module = importlib.import_module(
            "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.autotune"
        )
        name = "bf16_mxfp8_candidates"
    candidates = getattr(module, name)(batch_invariant=True)[:2]
    monkeypatch.setattr(module, name, lambda **kwargs: candidates)
    x, scores, ids, weights = _inputs(kind, 0, 1)
    config = _config(kind, early, {"swiglu_alpha": 1.702, "swiglu_beta": 1.0})
    autotune_disk_reload(
        make_config=lambda knobs: replace(config, knobs=knobs),
        weights=weights,
        x=x,
        ids=ids,
        scores=scores,
        layer_factory=_layer,
        provider="legacy",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize("kind", ("w8a16_e4m3",))
def test_w8a16_batch_invariant_persistent_live_count(kind, monkeypatch):
    import torch
    from flashinfer.moe_ep import FinalizeMode, MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank case")
    _device("standard")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _inputs(kind, 0, 1)
    config = _config(kind, True)
    with _layer(config, weights, 0, 1, 129, 384) as layer:
        anchor = layer.forward(MoEEpTensors(x[:129], ids[:129], scores[:129])).clone()
        receipt = layer.batch_invariant_fingerprint
    live = torch.full((1,), 129, device="cuda", dtype=torch.int32)
    config = replace(
        config, finalize_mode=FinalizeMode.PERSISTENT, num_valid_tokens_tensor=live
    )
    with _layer(config, weights, 0, 1, 129, 384) as layer:
        assert layer.batch_invariant_fingerprint == receipt
        for n in (0, 1, 17, 32, 129):
            live.fill_(n)
            assert_bits(
                layer.forward(MoEEpTensors(x[:n], ids[:n], scores[:n])),
                anchor[:n],
                "A16 persistent eager",
            )
        live.fill_(129)
        inputs = MoEEpTensors(x[:129], ids[:129], scores[:129])
        layer.forward(inputs, return_workspace_view=True)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = layer.forward(inputs, return_workspace_view=True)
            sentinel = torch.full_like(output, 123.0)
            for n in (0, 1, 17, 32, 129, 0, 129):
                output.copy_(sentinel)
                live.fill_(n)
                graph.replay()
                expected = sentinel.clone()
                expected[:n].copy_(anchor[:n])
                assert_bits(output, expected, "A16 captured device live count")
        finally:
            graph.reset()
