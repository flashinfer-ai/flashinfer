"""Rubin variant selection on real inputs, persisted reuse and captured storage."""

import pytest
import torch
import torch.distributed as dist

from flashinfer import autotune_v2, moe_ep
from flashinfer.autotuner import AutoTuner
from flashinfer.autotune_cache import MeasurementPolicy
from flashinfer.moe_ep.backends.mega.kernel.sm107 import variant_selection as selection
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    autotune as native,
)
from tests.moe_ep.sm107_test_utils import (
    assert_reference,
    canonical_reference,
    quantize,
    weight_pack,
)
from tests.moe_ep.test_sm107_auto_multirank import rubin_bootstrap  # noqa: F401
from tests.moe_ep import test_moe_ep_sm107_block_scaled_mega_multirank as problem


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "kind,prequantized,scenario",
    [
        (kind, prequantized, "native")
        for kind in ("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8")
        for prequantized in (False, True)
    ]
    + [
        ("mxfp4_mxfp8", False, scenario)
        for scenario in (
            "graph-cold",
            "eager-cold",
            "cupti",
            "late-situ-empty",
            "inference-only",
            "native-skipped",
        )
    ],
)
def test_variant_search_replay_and_handoff(
    kind,
    prequantized,
    scenario,
    rubin_bootstrap,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    bootstrap = rubin_bootstrap
    rank, world = bootstrap.rank, bootstrap.world_size
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
        NAMESPACE as NATIVE_NAMESPACE,
    )

    policy = {
        "graph-cold": MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True),
        "eager-cold": MeasurementPolicy(execution_mode="eager", cold_l2=True),
        "cupti": MeasurementPolicy(execution_mode="cuda_graph", _timer="cupti"),
    }.get(scenario)
    late = scenario == "late-situ-empty"
    cfg = problem._megakernel_config(
        kind,
        knobs="auto",
        kernel_variant="auto",
        activation="situ" if late else "swiglu",
        apply_routing_weights_before_fc2=not late,
        situ_beta=1.25 if late else None,
        situ_linear_beta=0.75 if late else None,
    )
    context_options = dict(
        cache_root=str(tmp_path / "managed"),
        measurement_policy=policy,
        skip_ops={NATIVE_NAMESPACE} if scenario == "native-skipped" else None,
    )
    expected_calls = (
        []
        if scenario == "native-skipped"
        else (
            ["inference"] if scenario == "inference-only" else ["inference", "genphase"]
        )
    )
    w13, w2, x, ids, scores = problem._global_problem(world)
    pack, _, _ = weight_pack(w13, w2, kind)
    count = problem.NUM_EXPERTS // world
    local = slice(rank * count, (rank + 1) * count)
    local_pack = moe_ep.PrequantizedMoEWeights(
        *(getattr(pack, name)[local] for name in ("w13", "w2", "w13_scale", "w2_scale"))
    )
    fleet = moe_ep.FleetParams(
        num_experts=problem.NUM_EXPERTS,
        max_tokens_per_rank=1025
        if scenario == "inference-only"
        else problem.MAX_TOKENS,
        token_hidden_size=problem.HIDDEN,
    )
    live = 0 if late and rank == world - 1 else problem.NUM_TOKENS - rank
    hidden = x[rank, :live].clone()
    xq, xsf = quantize(hidden, kind)
    tensors = moe_ep.MoEEpTensors(
        xq if prequantized else hidden,
        ids[rank, :live],
        scores[rank, :live],
        scales=xsf if prequantized else None,
        output=torch.empty(
            live, problem.HIDDEN * 2, device="cuda", dtype=torch.bfloat16
        )[:, ::2]
        if late
        else None,
    )
    expected = canonical_reference(
        xq,
        xsf,
        tensors.topk_ids,
        tensors.topk_weights,
        pack,
        kind,
        situ=late,
        early=not late,
    )
    layer = moe_ep.MoEEpLayer(
        bootstrap,
        fleet,
        local_pack,
        backend=moe_ep.MegaConfig(megakernel=cfg, quantize_input=not prequantized),
    )
    layers = [layer]
    graph = None
    calls = []
    sweep = native._sweep_sm107_block_scaled_mega_moe

    def counted(*args, **kwargs):
        calls.append(args[3].config.kernel_variant)
        return sweep(*args, **kwargs)

    monkeypatch.setattr(native, "_sweep_sm107_block_scaled_mega_moe", counted)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    try:
        with autotune_v2(mode="replay", **context_options):
            layer.warmup(tensors)
            old = layer._workspace
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                old_output = layer.forward(tensors)
        assert not calls
        with autotune_v2(mode="replay", **context_options):
            assert_reference(layer.forward(tensors), expected, kind)
        assert layer._workspace is old
        assert not calls
        with autotune_v2(**context_options):
            layer.warmup(tensors)
        assert calls == expected_calls
        assert layer._workspace is not old
        assert old in layer._retired_tuning_workspaces
        fixed_cfg = layer.get_tuned_config()
        assert fixed_cfg.kernel_variant in ("inference", "genphase")
        assert fixed_cfg.knobs == layer.get_tuned_knobs()
        with autotune_v2(mode="replay", **context_options):
            assert_reference(layer.forward(tensors), expected, kind)
        if scenario == "cupti":
            assert not tuner._cupti_disabled
        graph.replay()
        torch.cuda.synchronize()
        assert_reference(old_output, expected, kind)
        transformed = layer._get_transformed_weights()
        fixed = moe_ep.MoEEpLayer(
            bootstrap,
            fleet,
            local_pack,
            backend=moe_ep.MegaConfig(
                megakernel=fixed_cfg,
                transformed_weights=transformed,
                quantize_input=not prequantized,
            ),
        )
        layers.append(fixed)
        fixed.warmup(tensors)
        assert_reference(fixed.forward(tensors), expected, kind)
        tuner.clear_cache()
        tuner._managed_cache = None
        tuner._managed_stores.clear()
        replay = moe_ep.MoEEpLayer(
            bootstrap,
            fleet,
            local_pack,
            backend=moe_ep.MegaConfig(
                megakernel=cfg,
                transformed_weights=transformed,
                quantize_input=not prequantized,
            ),
        )
        layers.append(replay)
        with autotune_v2(mode="replay", **context_options):
            replay.warmup(tensors)
        assert replay.get_tuned_config().kernel_variant == fixed_cfg.kernel_variant
        assert replay.get_tuned_knobs() == fixed_cfg.knobs
        assert calls == expected_calls
        with autotune_v2(mode="replay", **context_options):
            assert_reference(replay.forward(tensors), expected, kind)
            handle = replay.create_workspace(fleet.max_tokens_per_rank)
            replay.warmup(tensors, workspace=handle)
            assert replay.get_tuned_config(workspace=handle) == fixed_cfg
            assert (
                handle._backend_workspace.workspace is not replay._workspace.workspace
            )
            assert_reference(replay.forward(tensors, workspace=handle), expected, kind)
        assert calls == expected_calls
        handle.destroy()
        records = tuner.get_protocol_records(selection.NAMESPACE)
        choices = [replay.get_tuned_config().kernel_variant, replay.get_tuned_knobs()]
        values = [None] * world
        dist.all_gather_object(values, choices, group=bootstrap.process_group)
        assert all(value == choices for value in values)
        if rank == 0:
            assert records
        graph.reset()
        graph = None
        torch.cuda.synchronize()
        for current in reversed(layers):
            current.destroy()
    except BaseException:
        # A rank-local failure may leave peers inside a collective launch.
        # Let the worker supervisor terminate the process group.
        layers.clear()
        raise
