# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared weight ownership, workspace bindings and tuning compatibility."""

import dataclasses
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    Sm107PrequantizedMoEWeightsWithSharedExpert,
    Sm107SharedMegaWeights,
    Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp4_bf16_cutedsl.backend import (
    Sm107Mxfp8Mxfp4MegaKernelBackend,
)
from flashinfer.moe_ep.backends.mega.kernel.sm107.shared_weights import (
    bind_prepared_weights,
    validate_shared_backend,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    block_scaled,
    knob_cache,
)
from tests.moe_ep.test_sm107_shared_weights import _weights
from tests.moe_ep.test_sm107_auto_multirank import rubin_bootstrap  # noqa: F401


def _source():
    return Sm107PrequantizedMoEWeightsWithSharedExpert(
        torch.zeros((2, 256, 64), dtype=torch.uint8),
        torch.zeros((2, 128, 64), dtype=torch.uint8),
        torch.full((2, 256, 4), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        torch.full((2, 128, 4), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        *_weights(),
    )


def _backend(**overrides):
    cfg = Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128, top_k=2, shared_expert=True, **overrides
    )
    backend = Sm107Mxfp8Mxfp4MegaKernelBackend(cfg)
    bootstrap = BootstrapConfig(rank=0, world_size=2, auto_bootstrap=False)
    backend.bind_ep_bootstrap(bootstrap)
    return backend, bootstrap, FleetParams(4, 256, 128)


def test_shared_source_and_prepared_contract():
    backend, bootstrap, fleet = _backend()
    source = _source()
    prepared = backend.preprocess_weights(source, fleet)
    assert isinstance(prepared, Sm107SharedMegaWeights)
    backend.validate_transformed_weights(prepared, bootstrap, fleet)
    assert prepared.routed[0][0].shape[0] == 2
    assert prepared.shared[0][0].shape[0] == 1
    with pytest.raises(ValueError, match="prepared weight container"):
        backend.validate_transformed_weights(prepared.routed, bootstrap, fleet)
    routed_only = Sm107Mxfp8Mxfp4MegaKernelBackend(
        dataclasses.replace(backend._kernel_config, shared_expert=False)
    )
    with pytest.raises(ValueError, match="weight container"):
        routed_only.preprocess_weights(source, fleet)
    for payload in (source, prepared):
        with pytest.raises(ValueError, match="shared_expert=True"):
            validate_shared_backend(payload, None)
        with pytest.raises(ValueError, match="shared_expert=True"):
            validate_shared_backend(payload, routed_only._kernel_config)
        validate_shared_backend(payload, backend._kernel_config)


def test_shared_layer_construction_retains_both_weight_sets(monkeypatch):
    from flashinfer.moe_ep import MegaConfig, MoEEpLayer

    # This test prepares CPU weights without creating a device workspace.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend, bootstrap, fleet = _backend()
    layer = MoEEpLayer(
        bootstrap, fleet, _source(), backend=MegaConfig(backend._kernel_config)
    )
    prepared = layer._get_transformed_weights()
    assert isinstance(prepared, Sm107SharedMegaWeights)
    other = MoEEpLayer(
        bootstrap,
        fleet,
        None,
        backend=MegaConfig(
            backend._kernel_config,
            preprocess_weights=False,
            transformed_weights=prepared,
        ),
    )
    assert other._get_transformed_weights() is prepared
    assert layer._preprocessing_count == 1
    assert other._preprocessing_count == 0
    weight_refs = [
        weakref.ref(value)
        for weights in (prepared.routed, prepared.shared)
        for pair in weights
        for value in pair
    ]
    assert layer._workspace is None
    other.destroy()
    del other, prepared
    gc.collect()
    assert all(ref() is not None for ref in weight_refs)
    layer.destroy()
    del layer
    gc.collect()
    assert all(ref() is None for ref in weight_refs)


def test_routed_w4a4_inherited_weight_preparation():
    from flashinfer.moe_ep import (
        PrequantizedMoEWeights,
        Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp4_mxfp4_bf16_cutedsl.backend import (
        Sm107Mxfp4MegaKernelBackend,
    )

    config = Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128, top_k=2
    )
    backend = Sm107Mxfp4MegaKernelBackend(config)
    bootstrap = BootstrapConfig(rank=0, world_size=2, auto_bootstrap=False)
    backend.bind_ep_bootstrap(bootstrap)
    fleet = FleetParams(4, 256, 128)
    source = _source()
    routed = PrequantizedMoEWeights(
        source.w13, source.w2, source.w13_scale, source.w2_scale
    )
    prepared = backend.preprocess_weights(routed, fleet)
    backend.validate_transformed_weights(prepared, bootstrap, fleet)
    with pytest.raises(ValueError, match="supplied weight container"):
        backend.preprocess_weights(source, fleet)
    shared_backend, _, _ = _backend()
    shared = shared_backend.preprocess_weights(source, fleet)
    with pytest.raises(ValueError, match="prepared weight container"):
        backend.validate_transformed_weights(shared, bootstrap, fleet)


@pytest.mark.parametrize("knobs", [None, "auto", "cache"])
def test_shared_fallback_matches_routed(monkeypatch, knobs):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "0")
    backend, _, fleet = _backend(knobs=knobs)
    cfg = backend._resolved_config(fleet)
    assert cfg.shared_quant_kind == "mxfp8_e4m3"
    routed = Sm107Mxfp8Mxfp4MegaKernelBackend(
        dataclasses.replace(backend._kernel_config, shared_expert=False)
    )
    routed.bind_ep_bootstrap(
        BootstrapConfig(rank=0, world_size=2, auto_bootstrap=False)
    )
    expected = routed._resolved_config(fleet)
    assert cfg.fallback_cluster_shape_mn == expected.fallback_cluster_shape_mn
    assert cfg.cluster_shape_mn == expected.cluster_shape_mn
    if knobs:
        assert cfg.schedule_policy == ("phase_interleave", None)
        assert cfg.work_id_mode == "atomic_counter"


@pytest.mark.parametrize(
    "overrides",
    [
        {"combine_dtype": "nvfp4"},
        {"combine_dtype": "mxfp8"},
        {"in_kernel_fc2_reduce": True},
        {"activation": "situ", "situ_beta": 1.0, "situ_linear_beta": 1.0},
    ],
)
def test_shared_rejects_unsupported_policy(overrides):
    backend, _, fleet = _backend(**overrides)
    with pytest.raises(ValueError, match="Shared experts require"):
        backend._resolved_config(fleet)


def _workspace():
    backend, _, fleet = _backend()
    prepared = backend.preprocess_weights(_source(), fleet)
    ws = object.__new__(block_scaled.Sm107BlockScaledSymmBuffer)
    ws.config = backend._resolved_config(fleet)
    ws.device = torch.device("cpu")
    ws._shared_weights = None
    ws._launch_key = None
    ws.output_activation = torch.empty((256, 128), dtype=torch.bfloat16)
    ws.launch = Mock()
    return ws, prepared


def test_shared_binding_reuses_weights_and_rebinds_pool():
    ws, prepared = _workspace()
    assert bind_prepared_weights(ws, prepared) is prepared.routed
    assert ws._shared_weights is prepared.shared
    ws._launch_key = "already bound"
    bind_prepared_weights(ws, prepared)
    assert ws._launch_key == "already bound"
    replacement = dataclasses.replace(
        prepared,
        shared=tuple(tuple(t.clone() for t in pair) for pair in prepared.shared),
    )
    bind_prepared_weights(ws, replacement)
    assert ws._shared_weights is replacement.shared
    assert ws._launch_key is None
    with pytest.raises(ValueError, match="prepared routed and shared"):
        bind_prepared_weights(ws, prepared.routed)


@pytest.mark.parametrize("live", [0, 1, 129, 256])
@pytest.mark.parametrize("output_mode", ["caller", "view", "strided"])
def test_shared_forward_uses_live_extent(live, output_mode):
    ws, prepared = _workspace()
    bind_prepared_weights(ws, prepared)
    ws.launch.side_effect = lambda *args, output: output.fill_(3)
    output = None
    if output_mode == "caller":
        output = torch.zeros((live, 128), dtype=torch.bfloat16)
    elif output_mode == "strided":
        output = torch.zeros((live, 256), dtype=torch.bfloat16)[:, ::2]
    result = block_scaled.sm107_block_scaled_mega_moe(
        output, *prepared.routed, ws, num_tokens=live
    )
    bound = ws.launch.call_args.kwargs["output"]
    assert bound.shape == (live, 128)
    assert bound.is_contiguous()
    if output_mode == "view":
        assert result.shape == (live, 128)
        assert (
            result.untyped_storage().data_ptr()
            == ws.output_activation.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(result, torch.full_like(result, 3))
    else:
        assert result is None
        torch.testing.assert_close(output, torch.full_like(output, 3))


def test_shared_cache_records_cannot_alias_routed_only():
    query = dict(
        dtype="mxfp4_mxfp8",
        world_size=2,
        hidden=128,
        intermediate=128,
        num_experts=4,
        topk=2,
        max_tokens=256,
        device="host",
    )
    knobs = knob_cache.default_knobs(256, quant_kind="mxfp4_mxfp8")
    old = knob_cache.make_record(knobs, **query)
    old.pop("shared_quant_kind")
    shared = knob_cache.make_record(
        {**knobs, "fallback_cluster_shape_mn": None},
        **query,
        shared_quant_kind="mxfp8_e4m3",
    )
    assert knob_cache.lookup_record(entries=[old, shared], **query) is old
    assert (
        knob_cache.lookup_record(
            entries=[old, shared], **query, shared_quant_kind="mxfp8_e4m3"
        )
        is shared
    )
    assert (
        knob_cache.lookup_record(entries=[old], **query, shared_quant_kind="mxfp8_e4m3")
        is None
    )


@pytest.mark.parametrize("variant", ["inference", "genphase"])
def test_shared_candidates_match_routed(variant):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.autotune import (
        is_valid_sm107,
        sm107_candidates,
    )

    backend, _, fleet = _backend()
    cfg = backend._resolved_config(fleet)
    cfg = dataclasses.replace(
        cfg, kernel_variant=variant, cluster_shape_mn=(4, 1), fc2_use_bulk=True
    )
    candidates = sm107_candidates(
        "mxfp4_mxfp8", kernel_variant=variant, shared_quant_kind="mxfp8_e4m3"
    )
    assert {k["schedule_policy"][0] for k in candidates} == {
        "grouped",
        "phase_interleave",
    }
    assert candidates == sm107_candidates("mxfp4_mxfp8", kernel_variant=variant)
    assert any(k["fallback_cluster_shape_mn"] == (2, 1) for k in candidates) == (
        variant == "inference"
    )
    routed = dataclasses.replace(cfg, shared_quant_kind=None)
    valid = [k for k in candidates if is_valid_sm107(k, cfg)]
    assert valid == [k for k in candidates if is_valid_sm107(k, routed)]
    assert len(valid) == (16 if variant == "inference" else 10)


def test_shared_trial_preserves_prepared_payload():
    from flashinfer.moe_ep.backends.mega.kernel.sm107.forward_tuning import (
        _ForwardTrial,
    )

    ws, prepared = _workspace()
    bind_prepared_weights(ws, prepared)
    backend = SimpleNamespace(stage_inputs=Mock(), compute=Mock())
    execution = SimpleNamespace(tensors=object(), quantize_input=True, output=object())
    _ForwardTrial(execution, backend, ws).launch(*prepared.routed)
    payload = backend.compute.call_args.args[1]
    assert isinstance(payload, Sm107SharedMegaWeights)
    assert payload.routed == prepared.routed
    assert payload.shared is prepared.shared


def test_shared_oracle_adds_local_output_after_ep_reduce(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import correctness

    ws, prepared = _workspace()
    bind_prepared_weights(ws, prepared)
    ws.x = torch.zeros((4, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    ws.x_sf = torch.full((4, 4), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    ws.topk_idx = torch.full((4, 2), -1, dtype=torch.int32)
    ws.topk_weights = torch.full((4, 2), float("nan"))
    calls = []

    def reference(x, sf, ids, scores, *args, **kwargs):
        shared = kwargs["quant_kind"] == "mxfp8_e4m3"
        calls.append("shared" if shared else "routed")
        if shared:
            assert torch.equal(ids, torch.zeros_like(ids))
            assert torch.equal(scores, torch.ones_like(scores))
        return torch.full(
            (x.shape[0], 128),
            3.0 if shared else 5.0,
            dtype=torch.bfloat16 if shared else torch.float32,
        )

    linspace = torch.linspace
    monkeypatch.setattr(
        torch,
        "linspace",
        lambda *args, **kwargs: linspace(*args, **{**kwargs, "device": "cpu"}),
    )
    monkeypatch.setattr(
        correctness, "compute_megamoe_reference_sm107_block_scaled", reference
    )
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed, "reduce", lambda tensor, **kwargs: tensor.mul_(2)
    )
    _, expected = correctness.sampled_reference(ws, *prepared.routed, 2, sample_size=4)
    assert calls == ["routed", "shared", "routed"]
    assert torch.equal(expected, torch.full((2, 128), 13.0, dtype=torch.bfloat16))


@pytest.mark.parametrize("selected", ["inference", "genphase"])
def test_shared_auto_admits_variants_and_reconstructs_config(monkeypatch, selected):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.variant_selection import (
        RubinVariantBackend,
    )

    cfg = Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        shared_expert=True,
        kernel_variant="auto",
        knobs="auto",
    )
    backend = RubinVariantBackend(cfg, Sm107Mxfp8Mxfp4MegaKernelBackend)
    backend.bind_ep_bootstrap(
        BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    monkeypatch.setattr(
        block_scaled.Sm107BlockScaledSymmBuffer, "_build_kernel", Mock()
    )
    assert backend._eligible(FleetParams(4, 256, 128)) == ["inference", "genphase"]
    owner = SimpleNamespace(
        _tuning_state=SimpleNamespace(
            winner={
                "variant": selected,
                "knobs": {"cluster_shape_mn": (4, 1)},
            }
        )
    )
    fixed = backend.get_tuned_config(owner)
    assert fixed.shared_expert
    assert fixed.kernel_variant == selected
    assert fixed.knobs == {"cluster_shape_mn": (4, 1)}


def test_shared_cli_does_not_drop_weights():
    from flashinfer.moe_ep.backends.mega.kernel.sm107.tuning import run_tuning, tune_one

    args = SimpleNamespace(shared_expert=True)
    with pytest.raises(ValueError, match="Shared-expert CLI inputs"):
        run_tuning(args, "mxfp4_mxfp8")
    with pytest.raises(ValueError, match="Shared-expert CLI inputs"):
        tune_one(args, 0, 2, 256, "mxfp4_mxfp8")


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize("quantize_input", [False, True])
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize(
    "variant,tile_n,schedule,mixed",
    [
        ("inference", 128, "phase_interleave", False),
        ("inference", 128, "phase_interleave", True),
        ("genphase", 64, "grouped", False),
        ("genphase", 128, "grouped", False),
        ("genphase", 128, "phase_interleave", False),
        ("genphase", 256, "grouped", False),
        ("genphase", 256, "phase_interleave", False),
    ],
)
def test_shared_layer_pooled_replay(
    quantize_input,
    early,
    variant,
    tile_n,
    schedule,
    mixed,
    rubin_bootstrap,  # noqa: F811
):
    import torch.distributed as dist
    from flashinfer.moe_ep import MegaConfig, MoEEpLayer, MoEEpTensors
    from tests.moe_ep import test_moe_ep_sm107_block_scaled_mega_multirank as problem
    from tests.moe_ep.sm107_test_utils import (
        assert_reference,
        canonical_reference,
        quantize,
        weight_pack,
    )

    rank, world = rubin_bootstrap.rank, rubin_bootstrap.world_size
    w13, w2, x, ids, scores = problem._global_problem(world)
    capacity = 256
    repeat = (capacity + problem.NUM_TOKENS - 1) // problem.NUM_TOKENS
    x, ids, scores = (t.repeat(1, repeat, 1)[:, :capacity] for t in (x, ids, scores))
    pack, _, _ = weight_pack(w13, w2, "mxfp4_mxfp8")
    shared_pack, _, _ = weight_pack(w13[:1], w2[:1], "mxfp8_e4m3")
    count = problem.NUM_EXPERTS // world
    local = slice(rank * count, (rank + 1) * count)
    fields = ("w13", "w2", "w13_scale", "w2_scale")
    source = Sm107PrequantizedMoEWeightsWithSharedExpert(
        *(getattr(pack, name)[local] for name in fields),
        *(getattr(shared_pack, name)[0] for name in fields),
    )
    cfg = Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=problem.INTERMEDIATE,
        top_k=problem.TOP_K,
        shared_expert=True,
        kernel_variant=variant,
        mma_tiler_mnk=(256, tile_n, 128),
        schedule_policy=(schedule, None),
        work_id_mode="atomic_counter",
        fc2_use_bulk=variant == "genphase",
        fc2_tma_stages=1 if tile_n == 64 else 2,
        apply_routing_weights_before_fc2=early,
        cluster_shape_mn=(4, 1) if mixed or variant == "genphase" else (2, 1),
        fallback_cluster_shape_mn=(2, 1) if mixed else None,
    )
    fleet = FleetParams(problem.NUM_EXPERTS, capacity, problem.HIDDEN)
    layer = MoEEpLayer(
        rubin_bootstrap,
        fleet,
        source,
        backend=MegaConfig(cfg, quantize_input=quantize_input),
    )
    prepared = layer._get_transformed_weights()
    zero_shared = tuple((torch.zeros_like(w), sf) for w, sf in prepared.shared)
    other = MoEEpLayer(
        rubin_bootstrap,
        fleet,
        None,
        backend=MegaConfig(
            cfg,
            quantize_input=quantize_input,
            preprocess_weights=False,
            transformed_weights=Sm107SharedMegaWeights(prepared.routed, zero_shared),
        ),
    )
    graphs = []

    def update(t, live, step, shared_only):
        hidden = x[rank, :live] * (1.0 + step * 0.125)
        route_ids = ids[rank, :live].clone()
        route_scores = scores[rank, :live].clone()
        route_ids[::3] = -1
        route_scores[::3] = float("nan")
        if shared_only:
            route_ids.fill_(-1)
            route_scores.fill_(float("inf"))
        xq, xsf = quantize(hidden, "mxfp4_mxfp8")
        t.hidden_states.copy_(hidden if quantize_input else xq)
        if not quantize_input:
            t.scales.copy_(xsf)
        t.topk_ids.copy_(route_ids)
        t.topk_weights.copy_(route_scores)
        routed = canonical_reference(
            xq, xsf, route_ids, route_scores, pack, "mxfp4_mxfp8", early=early
        )
        shared = canonical_reference(
            xq,
            xsf,
            torch.zeros((live, 1), dtype=torch.int32, device="cuda"),
            torch.ones((live, 1), device="cuda"),
            shared_pack,
            "mxfp8_e4m3",
            early=False,
        )
        return (routed.float() + shared.float()).bfloat16(), routed

    try:
        for live, shared_only in (
            (129 - rank, False),
            (128 - rank, False),
            (127 - rank, True),
            (1 if rank == 0 else 0, True),
        ):
            hidden = x[rank, :live].clone()
            xq, xsf = quantize(hidden, "mxfp4_mxfp8")
            t = MoEEpTensors(
                hidden_states=hidden if quantize_input else xq,
                scales=None if quantize_input else xsf,
                topk_ids=ids[rank, :live].clone(),
                topk_weights=scores[rank, :live].clone(),
                output=torch.empty_like(hidden),
            )
            expected = update(t, live, 0, shared_only)
            for index, selected in enumerate((layer, other)):
                selected.forward(t)
                assert_reference(t.output, expected[index], "mxfp4_mxfp8")
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph):
                    selected.forward(t)
                graphs.append((graph, t, live, shared_only, index))
            assert layer._workspace is other._workspace
            view = layer.forward(
                dataclasses.replace(t, output=None), return_workspace_view=True
            )
            assert view.shape == (live, problem.HIDDEN)
            assert_reference(view, expected[0], "mxfp4_mxfp8")
        # Captures depend on layer-owned weights, not caller references.
        weight_refs = [weakref.ref(v) for pair in prepared.shared for v in pair]
        del source, prepared, zero_shared
        gc.collect()
        assert all(ref() is not None for ref in weight_refs)
        churn = [
            torch.empty_like(ref().view(torch.uint8)).fill_(7) for ref in weight_refs
        ]
        del churn
        for graph, t, live, shared_only, index in reversed(graphs):
            expected = update(t, live, 2, shared_only)
            graph.replay()
            torch.cuda.synchronize()
            assert_reference(t.output, expected[index], "mxfp4_mxfp8")
        prepared = layer._get_transformed_weights()
        routed_ptr = prepared.routed[1][0].data_ptr()
        shared_before = tuple(
            tuple(v.clone() for v in pair) for pair in prepared.shared
        )
        # EPLB updates routed storage in place; captures must retain shared bindings.
        prepared.routed[1][0].zero_()
        pack.w2.zero_()
        for graph, t, live, shared_only, index in graphs:
            expected = update(t, live, 3, shared_only)
            graph.replay()
            torch.cuda.synchronize()
            assert_reference(t.output, expected[index], "mxfp4_mxfp8")
        assert prepared.routed[1][0].data_ptr() == routed_ptr
        for pair, before in zip(prepared.shared, shared_before, strict=True):
            for value, original in zip(pair, before, strict=True):
                assert torch.equal(value.view(torch.uint8), original.view(torch.uint8))
    finally:
        for graph, *_ in graphs:
            graph.reset()
        graphs.clear()
        other.destroy()
        layer.destroy()


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize("variant", ["inference", "genphase", "auto"])
def test_shared_layer_auto_reload(variant, rubin_bootstrap, monkeypatch, tmp_path):  # noqa: F811
    import torch.distributed as dist
    from flashinfer import autotune, autotune_v2
    from flashinfer.autotuner import AutoTuner
    from flashinfer.autotune_cache import MeasurementPolicy
    from flashinfer.moe_ep import MegaConfig, MoEEpLayer, MoEEpTensors
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as package
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
        autotune as native,
    )
    from tests.moe_ep import test_moe_ep_sm107_block_scaled_mega_multirank as problem
    from tests.moe_ep.sm107_test_utils import (
        assert_reference,
        canonical_reference,
        quantize,
        weight_pack,
    )

    rank, world = rubin_bootstrap.rank, rubin_bootstrap.world_size
    w13, w2, x, ids, scores = problem._global_problem(world)
    pack, _, _ = weight_pack(w13, w2, "mxfp4_mxfp8")
    shared_pack, _, _ = weight_pack(w13[:1], w2[:1], "mxfp8_e4m3")
    count = problem.NUM_EXPERTS // world
    local = slice(rank * count, (rank + 1) * count)
    fields = ("w13", "w2", "w13_scale", "w2_scale")
    source = Sm107PrequantizedMoEWeightsWithSharedExpert(
        *(getattr(pack, name)[local] for name in fields),
        *(getattr(shared_pack, name)[0] for name in fields),
    )
    cfg = Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=problem.INTERMEDIATE,
        top_k=problem.TOP_K,
        shared_expert=True,
        kernel_variant=variant,
        knobs="auto",
    )
    fleet = FleetParams(problem.NUM_EXPERTS, 16, problem.HIDDEN)
    live = 0 if rank == world - 1 else 8
    t = MoEEpTensors(
        x[rank, :live].clone(),
        ids[rank, :live].clone(),
        scores[rank, :live].clone(),
        output=torch.empty_like(x[rank, :live]),
    )
    native_candidates = native.sm107_candidates

    def candidates(*args, **kwargs):
        choices = native_candidates(*args, **kwargs)
        if variant == "auto":
            # Fixed-variant tests cover both native grids. Check ownership here.
            choices = [
                choices[0],
                next(
                    k for k in choices if k["schedule_policy"][0] == "phase_interleave"
                ),
            ]
        return choices

    monkeypatch.setattr(native, "sm107_candidates", candidates)
    monkeypatch.setattr(package, "sm107_candidates", candidates)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    calls = []
    sweep = native._sweep_sm107_block_scaled_mega_moe

    def counted(*args, **kwargs):
        assert args[3]._shared_weights is not None
        assert kwargs["make_trial"] is not None
        trial_variant = args[3].config.kernel_variant
        calls.append(trial_variant)
        result = sweep(*args, **kwargs)
        assert result[2] == (
            2 if variant == "auto" else 16 if trial_variant == "inference" else 10
        )
        return result

    monkeypatch.setattr(native, "_sweep_sm107_block_scaled_mega_moe", counted)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    context = dict(
        cache_root=str(tmp_path / "managed"),
        measurement_policy=MeasurementPolicy(execution_mode="cuda_graph"),
    )
    layer = MoEEpLayer(rubin_bootstrap, fleet, source, backend=MegaConfig(cfg))
    layers, graphs = [layer], []

    def reference():
        xq, xsf = quantize(t.hidden_states, "mxfp4_mxfp8")
        routed = canonical_reference(
            xq, xsf, t.topk_ids, t.topk_weights, pack, "mxfp4_mxfp8", early=True
        )
        shared = canonical_reference(
            xq,
            xsf,
            torch.zeros((live, 1), dtype=torch.int32, device="cuda"),
            torch.ones((live, 1), device="cuda"),
            shared_pack,
            "mxfp8_e4m3",
            early=False,
        )
        return (routed.float() + shared.float()).bfloat16()

    try:
        with autotune_v2(mode="replay", **context):
            layer.warmup(t)
            previous = layer._workspace
            old_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(old_graph):
                layer.forward(t)
            graphs.append(old_graph)
        assert not calls
        with autotune_v2(**context), autotune(cuda_graph_profile_replays=3):
            layer.warmup(t)
        assert calls and layer._workspace is not previous
        assert previous in layer._retired_tuning_workspaces
        expected = reference()
        assert_reference(layer.forward(t), expected, "mxfp4_mxfp8")
        fixed_cfg = layer.get_tuned_config()
        assert fixed_cfg.shared_expert
        assert set(calls) == (
            {"inference", "genphase"} if variant == "auto" else {variant}
        )
        assert fixed_cfg.kernel_variant in calls
        assert fixed_cfg.knobs == layer.get_tuned_knobs()
        prepared = layer._get_transformed_weights()
        old_graph.replay()
        torch.cuda.synchronize()
        assert_reference(t.output, expected, "mxfp4_mxfp8")
        completed = len(calls)
        tuner.clear_cache()
        tuner._managed_cache = None
        tuner._managed_stores.clear()
        cached = MoEEpLayer(
            rubin_bootstrap,
            fleet,
            None,
            backend=MegaConfig(
                cfg, preprocess_weights=False, transformed_weights=prepared
            ),
        )
        fixed = MoEEpLayer(
            rubin_bootstrap,
            fleet,
            None,
            backend=MegaConfig(
                fixed_cfg, preprocess_weights=False, transformed_weights=prepared
            ),
        )
        layers.extend((cached, fixed))
        with autotune_v2(mode="replay", **context):
            cached.warmup(t)
        assert len(calls) == completed
        assert cached.get_tuned_config() == fixed_cfg
        for selected in (cached, fixed):
            selected.forward(t)
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                selected.forward(t)
            graphs.append(graph)
        t.hidden_states.mul_(0.75)
        t.topk_ids.fill_(-1)
        t.topk_weights.fill_(float("inf"))
        expected = reference()
        for graph in graphs:
            graph.replay()
            torch.cuda.synchronize()
            assert_reference(t.output, expected, "mxfp4_mxfp8")
    finally:
        for graph in graphs:
            graph.reset()
        graphs.clear()
        for selected in reversed(layers):
            selected.destroy()
