# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Whole-operator BI checks for explicit numerical variants.

Representative recipes exercise the real producers, communication paths and
finalizers across batches, physical plans and graph replay.
"""

from dataclasses import replace
import os

import pytest

from ._megamoe_bi_helpers import assert_bits, assert_reference
from .test_nvfp4_megamoe_batch_invariant import _layer, _problem
from .test_nvfp4_megamoe_batch_invariant import _ieee_matmul as _precision_fixture

_ieee_matmul = _precision_fixture
pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("_ieee_matmul")]
HIDDEN = INTERMEDIATE = 512


def _require(kind, provider="standard", *, blackwell_only=False):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("requires SM100/SM103/SM107")
    if cc == (10, 7) and (blackwell_only or provider == "legacy"):
        pytest.skip("Blackwell implementation")
    if kind == "mxfp4" and cc != (10, 7):
        pytest.skip("pure MXFP4 requires SM107")
    return cc


def _recipe(name):
    if name == "minimax":
        return dict(swiglu_alpha=1.702, swiglu_beta=1.0, gate_up_clamp=2.5)
    if name == "identity_custom":
        return dict(swiglu_alpha=1.0, swiglu_beta=0.0, gate_up_clamp=2.5)
    return dict(
        activation="situ",
        situ_beta=1.25,
        situ_linear_beta=2.0 if name == "situ_linear" else None,
        gate_up_clamp=None,
    )


def _config(kind, provider, early, recipe="minimax", **overrides):
    if kind == "bf16":
        from flashinfer import moe_ep

        config = moe_ep.Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=INTERMEDIATE,
            top_k=3,
            batch_invariant=True,
            knobs={},
            apply_routing_weights_before_fc2=early,
        )
    elif kind == "nvfp4":
        from .test_nvfp4_megamoe_batch_invariant import _config as make

        config = make(provider, early, HIDDEN, INTERMEDIATE, knobs={})
    elif kind in ("mxfp4", "mxfp4_mxfp8"):
        from .test_mxfp4_megamoe_batch_invariant import _config as make

        config = make(kind, provider, early, knobs={}, intermediate=INTERMEDIATE)
    else:
        from .test_mxfp8_megamoe_batch_invariant import _config as make

        config = make(kind, provider, early, knobs={}, intermediate=INTERMEDIATE)
    return replace(config, **{**_recipe(recipe), **overrides})


def _plans(kind, provider, full=False, *, combine_dtype="bf16"):
    if provider == "genphase":
        import torch

        return (
            [{"work_id_mode": mode} for mode in ("grid_stride", "atomic_counter")]
            if torch.cuda.get_device_capability() == (10, 7)
            else [{}]
        )
    if kind == "bf16" or provider == "legacy":
        first_carrier = (
            "reuse_dispatch_warps"
            if provider == "legacy" and kind == "nvfp4" and combine_dtype != "bf16"
            else "epi_warps"
        )
        result = [
            dict(load_balance_mode="static", token_back_mode=first_carrier),
            dict(
                load_balance_mode="atomic_counter",
                token_back_mode="reuse_dispatch_warps",
            ),
        ]
    else:
        result = [
            dict(
                schedule_policy=("grouped", None),
                work_id_mode="grid_stride",
                token_back_mode="epi_warps",
            ),
            dict(
                schedule_policy=("grouped", None),
                work_id_mode="atomic_counter",
                token_back_mode="reuse_dispatch_warps",
            ),
        ]
    return result if full else result[-1:]


def _exercise(
    config,
    kind,
    provider,
    x,
    ids,
    scores,
    weights,
    *,
    rank=0,
    world=1,
    full=False,
    token_scale=None,
    quantize=True,
    input_scale=None,
    plans=None,
):
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    def inputs(order, output=None):
        values = (
            x.view(torch.uint8)[order].contiguous().view(x.dtype)
            if x.dtype != torch.bfloat16
            else x[order]
        )
        scale = (
            None
            if input_scale is None
            else input_scale.view(torch.uint8)[order]
            .contiguous()
            .view(input_scale.dtype)
        )
        return MoEEpTensors(
            values,
            ids[order],
            scores[order],
            scales=scale,
            output=output,
            fc1_activation_per_token_scale=None
            if token_scale is None
            else token_scale[order],
        )

    order = torch.arange(129, device=x.device)
    with _layer(config, weights, rank, world, 129, HIDDEN, quantize=quantize) as layer:
        anchor = layer.forward(inputs(order)).clone()
        receipt = layer.batch_invariant_fingerprint
        assert (
            receipt and torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0
        )
        if provider == "legacy":
            assert layer._workspace._frontend.batch_invariant_fingerprint == receipt
    selected_plans = (
        _plans(
            kind,
            provider,
            full,
            combine_dtype=getattr(config, "combine_dtype", "bf16"),
        )
        if plans is None
        else plans
    )
    for knobs in selected_plans:
        for capacity in (17, 129) if full else (17,):
            with _layer(
                replace(config, knobs=knobs),
                weights,
                rank,
                world,
                capacity,
                HIDDEN,
                quantize=quantize,
            ) as layer:
                assert layer.batch_invariant_fingerprint == receipt
                counts = (0, 1, 17, capacity) if world == 1 else (0, 1, capacity)
                for count in sorted(set(counts)):
                    # Empty/uneven ranks still enter every collective launch.
                    live = count if world == 1 or rank != 0 else max(0, count - 1)
                    selected = (torch.arange(live, device=x.device) * 37).remainder(129)
                    assert_bits(
                        layer.forward(inputs(selected)),
                        anchor[selected],
                        "numerical variant batch/plan",
                    )
                order = torch.arange(capacity, device=x.device)
                output = torch.empty(
                    capacity, HIDDEN, device=x.device, dtype=torch.bfloat16
                )
                args = inputs(order, output=output)
                layer.forward(args)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        layer.forward(args)
                    for selected in (
                        order.roll(2),
                        order.flip(0),
                        order.remainder(5),
                        order,
                    ):
                        staged = inputs(selected)
                        args.hidden_states.copy_(staged.hidden_states)
                        args.topk_ids.copy_(staged.topk_ids)
                        args.topk_weights.copy_(staged.topk_weights)
                        if input_scale is not None:
                            args.scales.copy_(staged.scales)
                        if token_scale is not None:
                            args.fc1_activation_per_token_scale.copy_(
                                staged.fc1_activation_per_token_scale
                            )
                        graph.replay()
                        assert_bits(
                            output, anchor[selected], "numerical variant cohort graph"
                        )
                    args.topk_ids.fill_(-1)
                    args.topk_weights.fill_(float("nan"))
                    graph.replay()
                    assert_bits(
                        output,
                        torch.zeros_like(output),
                        "masked nonzero stale variant graph",
                    )
                    # Restore real contributions into the same physical cells.
                    args.topk_ids.copy_(ids[order])
                    args.topk_weights.copy_(scores[order])
                    graph.replay()
                    assert_bits(output, anchor[order], "restored variant graph")
                finally:
                    graph.reset()
    return anchor


def _single(monkeypatch, kind, provider, early, recipe, request):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _require(kind, provider)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    config = _config(kind, provider, early, recipe)
    anchor = _exercise(
        config,
        kind,
        provider,
        x,
        ids,
        scores,
        weights,
        full=request.config.getoption("--full"),
    )
    if kind == "bf16":
        expected = _dense_reference(
            x[:33], ids[:33], scores[:33], weights.w13, weights.w2, config, early
        )
        assert_reference(anchor[:33], expected, 0.02, "BF16 variant independent oracle")
    return anchor


@pytest.mark.parametrize(
    "kind,early,recipe",
    (
        ("bf16", False, "minimax"),
        ("bf16", True, "situ_linear"),
        ("nvfp4", True, "minimax"),
        ("mxfp4_mxfp8", False, "situ"),
        ("mxfp4", True, "minimax"),
        ("mxfp8_e4m3", True, "minimax"),
        ("mxfp8_e5m2", False, "situ_linear"),
    ),
)
def test_numerical_activation_standard(kind, early, recipe, monkeypatch, request):
    _single(monkeypatch, kind, "standard", early, recipe, request)


@pytest.mark.parametrize(
    "kind,early,recipe",
    (
        ("nvfp4", False, "situ"),
        ("nvfp4", True, "minimax"),
        ("mxfp8_e4m3", True, "minimax"),
        ("mxfp8_e5m2", True, "situ_linear"),
    ),
)
def test_numerical_activation_blackwell_only_legacy(
    kind, early, recipe, monkeypatch, request
):
    _single(monkeypatch, kind, "legacy", early, recipe, request)


@pytest.mark.parametrize(
    "kind,early,recipe",
    (
        ("nvfp4", True, "minimax"),
        ("mxfp4_mxfp8", False, "situ"),
        ("mxfp4", True, "minimax"),
        ("mxfp8_e4m3", True, "minimax"),
        ("mxfp8_e5m2", False, "situ"),
    ),
)
def test_numerical_activation_genphase(kind, early, recipe, monkeypatch, request):
    _single(monkeypatch, kind, "genphase", early, recipe, request)


@pytest.mark.parametrize(
    "provider,recipe", (("legacy", "minimax"), ("standard", "situ_linear"))
)
def test_numerical_per_token_blackwell_only_nvfp4(
    provider, recipe, monkeypatch, request
):
    import torch
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )
    from .test_nvfp4_megamoe_batch_invariant import _plain_quantized_weights
    from .test_nvfp4_cutedsl_kernel_vs_reference import _torch_nvfp4_mega_reference

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _require("nvfp4", provider, blackwell_only=True)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    config = _config(
        "nvfp4", provider, True, recipe, enable_fc1_activation_per_token_scale=True
    )
    weights, plain = _plain_quantized_weights(weights, INTERMEDIATE)
    packed, scales = nvfp4_quantize_per_block_16(x.float(), config.input_norm_const)
    token_scale = torch.linspace(0.25, 2.0, x.shape[0], device="cuda")
    anchor = _exercise(
        config,
        "nvfp4",
        provider,
        packed,
        ids,
        scores,
        weights,
        full=request.config.getoption("--full"),
        token_scale=token_scale,
        quantize=False,
        input_scale=scales,
    )
    q1, sf1, q2, sf2 = plain
    expected = _torch_nvfp4_mega_reference(
        act_packed=packed[:17],
        act_sf=scales[:17],
        topk_idx=ids[:17],
        topk_weights=scores[:17],
        fc1_weight=q1,
        fc1_sf=sf1,
        fc2_weight=q2,
        fc2_sf=sf2,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        gate_up_clamp=config.gate_up_clamp,
        activation=config.activation,
        situ_beta=config.situ_beta,
        situ_linear_beta=config.situ_linear_beta,
        swiglu_alpha=config.swiglu_alpha,
        swiglu_beta=config.swiglu_beta,
        fc1_activation_per_token_scale=token_scale[:17],
        fc1_alpha=config.fc1_alpha,
        fc2_alpha=config.fc2_alpha,
        fc1_norm_const=config.fc1_norm_const,
    )
    assert_reference(
        anchor[:17], expected, 0.025, "per-token/custom independent oracle"
    )
    if provider != "standard":
        return

    from flashinfer import moe_ep
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.autotune import (
        autotune_sm100_block_scaled_mega_moe,
    )
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.correctness import (
        sampled_reference,
    )
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import (
        knob_keys,
    )

    # Both scales are finite, normal FP32 values. Their product is 2, while
    # multiplying the accumulator by expert alpha first would overflow.
    count, experts = 17, weights.w13.shape[0]
    alpha = torch.full((experts,), 1e38, dtype=torch.float32, device="cuda")
    token_scale = torch.full((count,), 2e-38, dtype=torch.float32, device="cuda")
    weights, _ = _plain_quantized_weights(
        moe_ep.MoEWeightPack(
            w13=torch.full(
                (experts, 2 * INTERMEDIATE, HIDDEN),
                0.03125,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            w2=torch.full(
                (experts, HIDDEN, INTERMEDIATE),
                0.015625,
                dtype=torch.bfloat16,
                device="cuda",
            ),
        ),
        INTERMEDIATE,
    )
    packed, scales = nvfp4_quantize_per_block_16(
        torch.full((count, HIDDEN), 0.25, device="cuda"), 1.0
    )
    config = replace(
        config,
        top_k=1,
        activation="swiglu",
        situ_beta=None,
        situ_linear_beta=None,
        gate_up_clamp=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        input_norm_const=1.0,
        fc1_alpha=alpha,
        fc2_alpha=None,
        fc1_norm_const=None,
        finalize_mode=moe_ep.FinalizeMode.DEFAULT,
    )
    inputs = moe_ep.MoEEpTensors(
        packed,
        torch.zeros((count, 1), dtype=torch.int32, device="cuda"),
        torch.ones((count, 1), dtype=torch.float32, device="cuda"),
        scales=scales,
        fc1_activation_per_token_scale=token_scale,
    )
    with _layer(config, weights, 0, 1, count, HIDDEN, quantize=False) as layer:
        actual = layer.forward(inputs).clone()
        workspace, (l1, l2) = layer._workspace, layer._transformed
        indices, expected = sampled_reference(workspace, l1, l2, count)
        assert_reference(actual[indices], expected, 0.02, "folded per-token scales")
        candidate = {
            name: getattr(workspace.config, name)
            for name in knob_keys(workspace.config)
        }
        winner = autotune_sm100_block_scaled_mega_moe(
            actual,
            l1,
            l2,
            workspace,
            num_tokens=count,
            candidates=[candidate],
            warmup_iters=1,
            timed_iters=1,
            calls_per_sample=1,
        )
        assert winner == candidate


@pytest.mark.parametrize(
    "kind,early,wire",
    (
        ("nvfp4", False, "mxfp8"),
        ("mxfp4_mxfp8", True, "nvfp4"),
        ("mxfp4", False, "mxfp8"),
        ("mxfp8_e4m3", True, "mxfp8"),
        ("mxfp8_e5m2", False, "nvfp4"),
    ),
)
def test_numerical_quantized_combine_standard(kind, early, wire, monkeypatch, request):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _require(kind)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    config = _config(kind, "standard", early, "minimax", combine_dtype=wire)
    _exercise(
        config,
        kind,
        "standard",
        x,
        ids,
        scores,
        weights,
        full=request.config.getoption("--full"),
    )


@pytest.mark.parametrize("early,wire", ((True, "nvfp4"), (False, "mxfp8")))
def test_numerical_quantized_combine_blackwell_only_legacy_nvfp4(
    early, wire, monkeypatch, request
):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _require("nvfp4", "legacy", blackwell_only=True)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    config = _config("nvfp4", "legacy", early, "minimax", combine_dtype=wire)
    _exercise(
        config,
        "nvfp4",
        "legacy",
        x,
        ids,
        scores,
        weights,
        full=request.config.getoption("--full"),
    )


@pytest.mark.parametrize(
    "kind,recipe", (("mxfp8_e4m3", "minimax"), ("mxfp8_e5m2", "situ_linear"))
)
def test_numerical_quantized_combine_blackwell_only_legacy_mxfp8(
    kind, recipe, monkeypatch, request
):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp8_batch_invariant import (
        mxfp8_batch_invariant_fingerprint,
    )

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _require(kind, "legacy", blackwell_only=True)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    config = _config(kind, "legacy", True, recipe, combine_dtype="mxfp8")
    wire_receipt = mxfp8_batch_invariant_fingerprint(
        config, hidden=HIDDEN, num_experts=4
    )
    assert wire_receipt != mxfp8_batch_invariant_fingerprint(
        replace(config, combine_dtype="bf16"), hidden=HIDDEN, num_experts=4
    )
    _exercise(
        config,
        kind,
        "legacy",
        x,
        ids,
        scores,
        weights,
        full=request.config.getoption("--full"),
        plans=_plans(kind, "legacy", True),
    )


def test_numerical_quantized_combine_blackwell_only_legacy_mxfp8_autotune_disk_reload(
    monkeypatch, tmp_path
):
    from ._megamoe_bi_helpers import autotune_disk_reload
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import autotune

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _require("mxfp8_e5m2", "legacy", blackwell_only=True)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    candidates = autotune.mxfp8_candidates(batch_invariant=True)[:2]
    monkeypatch.setattr(autotune, "mxfp8_candidates", lambda **kwargs: candidates)
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    config = _config("mxfp8_e5m2", "legacy", True, "minimax", combine_dtype="mxfp8")
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


def _deferred(kind, provider, early, monkeypatch):
    import torch
    from flashinfer.moe_ep import FinalizeMode, MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _require(kind, provider, blackwell_only=kind != "bf16")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    config = _config(kind, provider, early)
    with _layer(config, weights, 0, 1, 129, HIDDEN) as layer:
        finalized = layer.forward(
            MoEEpTensors(x[:129], ids[:129], scores[:129])
        ).clone()
        finalized_receipt = layer.batch_invariant_fingerprint
    config = replace(config, finalize_mode=FinalizeMode.NONE)
    with _layer(config, weights, 0, 1, 129, HIDDEN) as layer:
        rows, route_weights, _ = layer.forward_unfinalized(
            MoEEpTensors(x[:129], ids[:129], scores[:129])
        )
        anchor = rows.view(129, 3, HIDDEN).clone()
        weights_anchor = route_weights.clone()
        receipt = layer.batch_invariant_fingerprint
        assert receipt != finalized_receipt
    acc = torch.zeros_like(finalized, dtype=torch.float32)
    for slot in range(3):
        if early:
            acc = acc + anchor[:, slot].float()
        else:
            acc = (
                acc.double()
                + anchor[:, slot].double() * weights_anchor[:, slot, None].double()
            ).float()
    assert_bits(
        acc.bfloat16(),
        finalized,
        "deferred partial contract reconstructs finalized output",
    )
    with _layer(
        replace(config, knobs=_plans(kind, provider)[0]), weights, 0, 1, 17, HIDDEN
    ) as layer:
        assert layer.batch_invariant_fingerprint == receipt
        args = MoEEpTensors(x[:17].clone(), ids[:17].clone(), scores[:17].clone())
        rows, route_weights, _ = layer.forward_unfinalized(args)
        assert_bits(rows.view(17, 3, HIDDEN), anchor[:17], "deferred batch/plan")
        assert torch.equal(
            route_weights.view(torch.int32), weights_anchor[:17].view(torch.int32)
        )
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                rows, route_weights, _ = layer.forward_unfinalized(args)
            for selected in (
                torch.arange(17, device="cuda").roll(2),
                torch.arange(17, device="cuda").remainder(5),
            ):
                args.hidden_states.copy_(x[selected])
                args.topk_ids.copy_(ids[selected])
                args.topk_weights.copy_(scores[selected])
                graph.replay()
                assert_bits(
                    rows.view(17, 3, HIDDEN), anchor[selected], "deferred graph"
                )
            args.topk_ids.fill_(-1)
            args.topk_weights.fill_(float("nan"))
            graph.replay()
            assert torch.count_nonzero(rows.view(torch.int16)) == 0
            assert torch.count_nonzero(route_weights.view(torch.int32)) == 0
        finally:
            graph.reset()


@pytest.mark.parametrize("early", (False, True))
def test_numerical_deferred_bf16(early, monkeypatch):
    _deferred("bf16", "standard", early, monkeypatch)


@pytest.mark.parametrize(
    "kind,provider,early",
    (
        ("nvfp4", "legacy", True),
        ("nvfp4", "standard", False),
        ("mxfp4_mxfp8", "standard", False),
        ("mxfp8_e4m3", "legacy", True),
        ("mxfp8_e4m3", "standard", False),
    ),
)
def test_numerical_deferred_blackwell_only(kind, provider, early, monkeypatch):
    if kind.startswith("mxfp8") and provider == "legacy" and not early:
        pytest.skip("legacy MXFP8 is early-only")
    _deferred(kind, provider, early, monkeypatch)


def test_numerical_cake_blackwell_only(monkeypatch):
    import torch
    from flashinfer.moe_ep import FinalizeMode, MoEEpTensors
    from .test_nvfp4_megamoe_batch_invariant import _config as make

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _require("nvfp4", blackwell_only=True)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, intermediate = 4096, 512
    x, scores, ids, weights = _problem(
        0, 1, hidden, intermediate, top_k=6, local_experts=8
    )
    config = replace(
        make(
            "standard", True, hidden, intermediate, top_k=6, local_experts=8, knobs={}
        ),
        **_recipe("minimax"),
    )
    args = MoEEpTensors(x[:17], ids[:17], scores[:17])
    with _layer(config, weights, 0, 1, 256, hidden) as layer:
        anchor = layer.forward(args).clone()
        receipt = layer.batch_invariant_fingerprint
    with _layer(
        replace(config, finalize_mode=FinalizeMode.CAKE), weights, 0, 1, 256, hidden
    ) as layer:
        assert layer.batch_invariant_fingerprint == receipt
        args = MoEEpTensors(
            x[:17].clone(),
            ids[:17].clone(),
            scores[:17].clone(),
            output=torch.empty_like(x[:17]),
        )
        layer.forward(args)
        assert_bits(args.output, anchor, "CAKE/default finalized receipt")
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                layer.forward(args)
            args.topk_ids.fill_(-1)
            args.topk_weights.fill_(float("nan"))
            graph.replay()
            assert_bits(args.output, torch.zeros_like(anchor), "CAKE stale mask")
            args.topk_ids.copy_(ids[:17])
            args.topk_weights.copy_(scores[:17])
            graph.replay()
            assert_bits(args.output, anchor, "CAKE restored graph")
        finally:
            graph.reset()


def _dense_reference(x, ids, scores, w13, w2, config, early):
    """Independent BF16 whole-op oracle for the multi-rank activation variants."""
    import torch

    acc = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        wire = torch.zeros_like(x)
        for expert in range(w13.shape[0]):
            selected = ids[:, slot] == expert
            if not selected.any():
                continue
            gate, up = (x[selected].float() @ w13[expert].float().T).chunk(2, -1)
            if config.activation == "situ":
                gate = (
                    config.situ_beta
                    * torch.tanh(gate / config.situ_beta)
                    * torch.sigmoid(gate)
                )
                if config.situ_linear_beta is not None:
                    up = config.situ_linear_beta * torch.tanh(
                        up / config.situ_linear_beta
                    )
                activated = up * gate
            else:
                if config.gate_up_clamp is not None:
                    limit = abs(config.gate_up_clamp)
                    gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                activated = (
                    (up + config.swiglu_beta)
                    * gate
                    * torch.sigmoid(config.swiglu_alpha * gate)
                )
            if early:
                activated = activated * scores[selected, slot, None]
            wire[selected] = (
                activated.bfloat16().float() @ w2[expert].float().T
            ).bfloat16()
        if early:
            acc = acc + wire.float()
        else:
            valid = (ids[:, slot] >= 0) & (ids[:, slot] < w13.shape[0])
            score = torch.where(valid, scores[:, slot], 0)
            acc = (acc.double() + wire.double() * score[:, None].double()).float()
    return acc.bfloat16()


EP_CASES = (
    ("bf16", "bf16", False, "minimax"),
    ("bf16", "bf16", True, "situ_linear"),
    ("nvfp4", "bf16", True, "minimax"),
    ("nvfp4", "mxfp8", False, "situ"),
    ("mxfp4_mxfp8", "nvfp4", True, "minimax"),
    ("mxfp4", "mxfp8", False, "situ_linear"),
    ("mxfp8_e4m3", "mxfp8", True, "minimax"),
    ("mxfp8_e5m2", "nvfp4", False, "situ"),
)


@pytest.mark.gpu_2
@pytest.mark.parametrize("kind,wire,early,recipe", EP_CASES)
def test_numerical_expert_parallel(kind, wire, early, recipe, request):
    import torch
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from ._megamoe_bi_helpers import (
        all_gather_rows,
        assert_collectively,
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
    _require(kind)
    config = _config(
        kind,
        "standard",
        early,
        recipe,
        **({} if kind == "bf16" else {"combine_dtype": wire}),
    )
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    try:
        x, scores, ids, weights = _problem(rank, world, HIDDEN, INTERMEDIATE)
        ids = remote_routes(ids, rank, world, 4)
        if kind == "bf16":
            w13, w2 = all_gather_rows(weights.w13), all_gather_rows(weights.w2)
            reference = lambda values, sf, route, score: _dense_reference(
                values, route, score, w13, w2, config, early
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
                hidden=HIDDEN,
                rank=rank,
                world=world,
                tolerance=0.02,
            )
        _exercise(
            config,
            kind,
            "standard",
            x,
            ids,
            scores,
            weights,
            rank=rank,
            world=world,
            full=request.config.getoption("--full"),
        )
        # Every source has observable contributions from every remote peer.
        with _layer(config, weights, rank, world, 17, HIDDEN) as layer:
            args = moe_ep.MoEEpTensors(
                x[:17].clone(), ids[:17].clone(), scores[:17].clone()
            )
            baseline = layer.forward(args).clone()
            for peer in range(world):
                if peer == rank:
                    continue
                mask = ids[:17].div(4, rounding_mode="floor") == peer
                assert_collectively(mask.any(), x.device, "missing remote peer traffic")
                args.topk_ids.copy_(ids[:17].masked_fill(mask, -1))
                args.topk_weights.copy_(scores[:17].masked_fill(mask, float("nan")))
                masked = layer.forward(args).clone()
                assert_collectively(
                    torch.isfinite(masked).all()
                    and (masked.view(torch.int16) != baseline.view(torch.int16)).any(),
                    x.device,
                    "removing remote peer had no observable contribution",
                )
                args.topk_ids.copy_(ids[:17])
                args.topk_weights.copy_(scores[:17])
                assert_bits(
                    layer.forward(args), baseline, "remote contribution restore"
                )
    except BaseException as error:
        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_2
@pytest.mark.parametrize("provider", ("legacy", "standard"))
def test_numerical_expert_parallel_per_token_blackwell_only(provider, request):
    """Remote per-token scales are gathered independently of native pool rows."""
    from types import SimpleNamespace

    import torch
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )
    from ._megamoe_bi_helpers import (
        all_gather_rows,
        assert_collectively,
        fail_stop_distributed,
        remote_routes,
    )
    from .test_nvfp4_megamoe_batch_invariant import _plain_quantized_weights
    from .test_nvfp4_cutedsl_kernel_vs_reference import _torch_nvfp4_mega_reference

    world, rank = (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
    )
    if world < 2:
        pytest.skip("requires torchrun")
    bootstrap = moe_ep.BootstrapConfig(rank=rank, world_size=world)
    moe_ep.ensure_moe_ep_cuda_device(bootstrap)
    _require("nvfp4", provider, blackwell_only=True)
    config = _config(
        "nvfp4", provider, True, "minimax", enable_fc1_activation_per_token_scale=True
    )
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    try:
        x, scores, ids, weights = _problem(rank, world, HIDDEN, INTERMEDIATE)
        ids = remote_routes(ids, rank, world, 4)
        weights, plain = _plain_quantized_weights(weights, INTERMEDIATE)
        q, sf = nvfp4_quantize_per_block_16(x.float(), config.input_norm_const)
        token_scale = torch.linspace(
            0.25 + rank * 0.125, 2.0 + rank * 0.125, x.shape[0], device="cuda"
        )
        global_plain = tuple(all_gather_rows(value) for value in plain)
        reference_config = SimpleNamespace(
            **{
                name: all_gather_rows(getattr(config, name))
                for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
            }
        )

        def reference(data, scale, routes, route_scores, per_token):
            q1, sf1, q2, sf2 = global_plain
            return _torch_nvfp4_mega_reference(
                act_packed=data,
                act_sf=scale,
                topk_idx=routes,
                topk_weights=route_scores,
                fc1_weight=q1,
                fc1_sf=sf1,
                fc2_weight=q2,
                fc2_sf=sf2,
                hidden=HIDDEN,
                intermediate=INTERMEDIATE,
                gate_up_clamp=config.gate_up_clamp,
                swiglu_alpha=config.swiglu_alpha,
                swiglu_beta=config.swiglu_beta,
                fc1_activation_per_token_scale=per_token,
                fc1_alpha=reference_config.fc1_alpha,
                fc2_alpha=reference_config.fc2_alpha,
                fc1_norm_const=reference_config.fc1_norm_const,
            )

        global_args = [
            all_gather_rows(value[:17]) for value in (q, sf, ids, scores, token_scale)
        ]
        expected = reference(*global_args)[rank * 17 : (rank + 1) * 17]
        anchor = _exercise(
            config,
            "nvfp4",
            provider,
            q,
            ids,
            scores,
            weights,
            rank=rank,
            world=world,
            full=request.config.getoption("--full"),
            token_scale=token_scale,
            quantize=False,
            input_scale=sf,
        )
        assert_reference(
            anchor[:17], expected, 0.025, "remote per-token independent oracle"
        )
        with _layer(config, weights, rank, world, 17, HIDDEN, quantize=False) as layer:
            output = torch.empty(17, HIDDEN, device="cuda", dtype=torch.bfloat16)
            args = moe_ep.MoEEpTensors(
                q[:17].clone(),
                ids[:17].clone(),
                scores[:17].clone(),
                scales=sf[:17].clone(),
                fc1_activation_per_token_scale=token_scale[:17].clone(),
                output=output,
            )
            layer.forward(args)
            baseline = output.clone()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    layer.forward(args)
                for peer in range(world):
                    if peer == rank:
                        continue
                    mask = ids[:17].div(4, rounding_mode="floor") == peer
                    assert_collectively(
                        mask.any(), q.device, "missing per-token remote traffic"
                    )
                    masked_ids = ids[:17].masked_fill(mask, -1)
                    masked_scores = scores[:17].masked_fill(mask, 0)
                    masked_expected = reference(
                        q[:17], sf[:17], masked_ids, masked_scores, token_scale[:17]
                    )
                    assert_collectively(
                        (
                            masked_expected.view(torch.int16)
                            != expected.view(torch.int16)
                        ).any(),
                        q.device,
                        "remote per-token route has no independent contribution",
                    )
                    args.topk_ids.copy_(masked_ids)
                    args.topk_weights.copy_(scores[:17].masked_fill(mask, float("nan")))
                    graph.replay()
                    assert_reference(
                        output, masked_expected, 0.025, "masked per-token remote oracle"
                    )
                    args.topk_ids.copy_(ids[:17])
                    args.topk_weights.copy_(scores[:17])
                    graph.replay()
                    assert_bits(output, baseline, "per-token remote restored graph")
            finally:
                graph.reset()
    except BaseException as error:
        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_2
@pytest.mark.parametrize("kind", ("mxfp8_e4m3", "mxfp8_e5m2"))
def test_numerical_expert_parallel_blackwell_only_legacy_mxfp8_wire(kind, request):
    """Both legacy quantized return carriers publish the remote SF sideband."""
    import torch
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from ._megamoe_bi_helpers import (
        assert_collectively,
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
    _require(kind, "legacy", blackwell_only=True)
    config = _config(kind, "legacy", True, "minimax", combine_dtype="mxfp8")
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    try:
        x, scores, ids, weights = _problem(rank, world, HIDDEN, INTERMEDIATE)
        ids = remote_routes(ids, rank, world, 4)
        _exercise(
            config,
            kind,
            "legacy",
            x,
            ids,
            scores,
            weights,
            rank=rank,
            world=world,
            full=request.config.getoption("--full"),
            plans=_plans(kind, "legacy", True),
        )
        with _layer(config, weights, rank, world, 17, HIDDEN) as layer:
            args = moe_ep.MoEEpTensors(
                x[:17].clone(), ids[:17].clone(), scores[:17].clone()
            )
            baseline = layer.forward(args).clone()
            for peer in range(world):
                if peer == rank:
                    continue
                mask = ids[:17].div(4, rounding_mode="floor") == peer
                assert_collectively(
                    mask.any(), x.device, "missing legacy MXFP8 remote traffic"
                )
                args.topk_ids.copy_(ids[:17].masked_fill(mask, -1))
                args.topk_weights.copy_(scores[:17].masked_fill(mask, float("nan")))
                masked = layer.forward(args).clone()
                assert_collectively(
                    torch.isfinite(masked).all()
                    and (masked.view(torch.int16) != baseline.view(torch.int16)).any(),
                    x.device,
                    "removing legacy MXFP8 peer had no observable contribution",
                )
                args.topk_ids.copy_(ids[:17])
                args.topk_weights.copy_(scores[:17])
                assert_bits(
                    layer.forward(args), baseline, "legacy quantized peer restore"
                )
    except BaseException as error:
        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)
