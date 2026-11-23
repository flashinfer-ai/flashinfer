"""Rubin shared-expert and deduplicated-combine public adapter contracts."""

from __future__ import annotations

import dataclasses
import contextlib
import os
from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
    Sm107SharedExpertConfig,
    Sm107SharedExpertWeights,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.shims.sm107 import cutedsl_megamoe as pkg
from tests.moe_ep.test_sm107_block_scaled_config import _BACKENDS


_KINDS = ["nvfp4", "mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"]


@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("raw_scales", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_prequantized_shared_input_accepts_e8m0_bytes(kind, raw_scales, explicit):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.shared_expert import (
        validate_shared_inputs,
    )

    dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
    x = torch.zeros(3, 512).to(dtype)
    scales = torch.full((3, 16), 127, dtype=torch.uint8)
    if not raw_scales:
        scales = scales.view(torch.float8_e8m0fnu)
    tensors = MoEEpTensors(
        x,
        torch.zeros(3, 4, dtype=torch.int32),
        torch.ones(3, 4),
        scales=scales,
        shared_hidden_states=x if explicit else None,
        shared_scales=scales if explicit else None,
    )
    validate_shared_inputs(
        SimpleNamespace(
            kind=kind, shared_expert=Sm107SharedExpertConfig(128, kind=kind)
        ),
        tensors,
        FleetParams(num_experts=32, max_tokens_per_rank=32, token_hidden_size=512),
        quantize_input=False,
    )


def _native_config(**overrides):
    return pkg.Sm107BlockScaledMoeConfig(
        **(
            dict(
                num_total_experts=32,
                max_tokens_per_rank=32,
                num_topk=4,
                hidden=512,
                intermediate=256,
                rank=0,
                world_size=4,
            )
            | overrides
        )
    )


@pytest.mark.parametrize("config_cls,backend_cls,name", _BACKENDS)
def test_shared_config_reaches_native_and_isolates_workspaces(
    config_cls, backend_cls, name, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    fleet = FleetParams(num_experts=32, max_tokens_per_rank=32, token_hidden_size=512)
    base = config_cls(256, 4)
    keys = []
    for shared in (
        None,
        Sm107SharedExpertConfig(128),
        Sm107SharedExpertConfig(256, kind="mxfp8_e5m2"),
    ):
        backend = create_mega_kernel(dataclasses.replace(base, shared_expert=shared))
        backend.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=4))
        assert backend._resolved_config(fleet).shared_expert == shared
        keys.append(backend._workspace_pool_key(fleet))
    assert len(set(keys)) == 3


@pytest.mark.parametrize(
    "overrides",
    [
        dict(kernel_variant="genphase"),
        dict(combine_dtype="mxfp8"),
        dict(apply_routing_weights_before_fc2=False),
        dict(fc2_use_bulk=False),
        dict(token_back_mode="epi_warps"),
        dict(reduce_topk_in_kernel=True),
        dict(epilogue_warpgroups=2),
        dict(world_size=64, num_total_experts=128),
    ],
)
def test_dedup_rejects_incompatible_execution(overrides):
    options = dict(
        combine_dedup=True, fc2_use_bulk=True, token_back_mode="reuse_dispatch_warps"
    )
    with pytest.raises(ValueError, match="combine_dedup"):
        _native_config(**(options | overrides))


@pytest.mark.parametrize("prequantized", [False, True])
def test_weight_pack_preserves_shared_ownership(prequantized):
    shared = MoEWeightPack(torch.empty(1, 256, 512), torch.empty(1, 512, 128))
    args = (torch.empty(8, 512, 512), torch.empty(8, 512, 256))
    scales = (torch.empty(8, 512, 16), torch.empty(8, 512, 8)) if prequantized else ()
    pack = MoEWeightPack(*args, *scales, shared_expert=shared)
    assert pack.shared_expert is shared
    assert MoEWeightPack(*args).shared_expert is None


def test_feature_cache_entries_do_not_alias():
    from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import knob_cache

    key = dict(
        dtype="mxfp8_e4m3",
        world_size=4,
        hidden=512,
        intermediate=256,
        num_experts=32,
        topk=4,
        max_tokens=32,
        device="test",
    )
    ordinary = knob_cache.make_record({}, p50_us=1, **key)
    shared = knob_cache.make_record(
        {}, p50_us=1, shared_expert=Sm107SharedExpertConfig(128), **key
    )
    dedup = knob_cache.make_record({}, p50_us=1, combine_dedup=True, **key)
    entries = [ordinary, shared, dedup]
    assert knob_cache.lookup_record(entries=entries, **key) == ordinary
    assert (
        knob_cache.lookup_record(
            entries=entries, shared_expert=Sm107SharedExpertConfig(128), **key
        )
        == shared
    )
    assert knob_cache.lookup_record(entries=entries, combine_dedup=True, **key) == dedup


def test_shared_scale_override_requires_an_input():
    from types import SimpleNamespace
    from flashinfer.moe_ep.backends.mega.kernel.sm107.shared_expert import (
        validate_shared_inputs,
    )

    config = SimpleNamespace(shared_expert=Sm107SharedExpertConfig(128))
    tensors = SimpleNamespace(shared_hidden_states=None, shared_scales=torch.empty(1))
    with pytest.raises(ValueError, match="shared_scales requires shared_hidden_states"):
        validate_shared_inputs(config, tensors, None, quantize_input=True)


@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("situ", [False, True])
def test_shared_reference_sampling_preserves_rows(kind, situ):
    from types import SimpleNamespace
    from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.correctness import (
        _shared_reference,
    )

    torch.manual_seed(4928)
    x = torch.randn(9, 128).bfloat16()
    w1 = (torch.randn(1, 256, 128) / 128**0.5).bfloat16()
    w2 = (torch.randn(1, 128, 128) / 128**0.5).bfloat16()
    qx, sf = _quantized_input(kind, x)
    weights = pkg.preprocess_block_scaled_weights(
        w1, w2, quant_kind=kind, intermediate_size=128
    )
    cfg = SimpleNamespace(
        shared_expert=Sm107SharedExpertConfig(128, kind=kind),
        gate_up_clamp=None,
        situ_beta=4.0 if situ else None,
        situ_linear_beta=25.0 if situ else None,
    )
    workspace = SimpleNamespace(
        config=cfg, shared_x=qx, shared_x_sf_linear=sf, _shared_weights=weights
    )
    full = pkg.compute_megamoe_reference_sm107_block_scaled(
        qx,
        sf,
        torch.zeros((9, 1), dtype=torch.int32),
        torch.ones((9, 1)),
        weights[0][0].transpose(1, 2),
        weights[0][1],
        weights[1][0].transpose(1, 2),
        weights[1][1],
        quant_kind=kind,
        local_expert_offset=0,
        gate_up_clamp=None,
        apply_routing_weights_before_fc2=False,
        weight_scales_are_swizzled=True,
        situ_beta=cfg.situ_beta,
        situ_linear_beta=cfg.situ_linear_beta,
    )
    rows = torch.tensor([0, 4, 8])
    torch.testing.assert_close(
        _shared_reference(workspace, rows), full[rows], rtol=0, atol=0
    )


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "variant,dedup,shared",
    [
        ("inference", False, True),
        ("genphase", False, True),
        ("inference", True, False),
        ("inference", True, True),
    ],
)
def test_exported_native_feature_construction(
    kind, variant, dedup, shared, monkeypatch
):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import block_scaled

    monkeypatch.setattr(block_scaled, "_max_active_clusters", lambda n: 8)
    cfg = _native_config(
        quant_kind=kind,
        kernel_variant=variant,
        fc2_use_bulk=True,
        cluster_shape_mn=(4, 1),
        combine_dedup=dedup,
        token_back_mode="reuse_dispatch_warps" if dedup else "epi_warps",
        shared_expert=Sm107SharedExpertConfig(128) if shared else None,
    )
    kernel = pkg.Sm107BlockScaledSymmBuffer._build_kernel(cfg)
    assert kernel.shared_expert is shared
    assert getattr(kernel, "combine_dedup", False) is dedup
    assert bool(kernel.name())


@pytest.mark.parametrize("topk", [31, 32, 33])
@pytest.mark.parametrize("shared", [False, True])
def test_genphase_shared_reducer_crosses_polled_limit(topk, shared, monkeypatch):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import block_scaled

    monkeypatch.setattr(block_scaled, "_max_active_clusters", lambda n: 8)
    cfg = _native_config(
        num_total_experts=64,
        num_topk=topk,
        kernel_variant="genphase",
        fc2_use_bulk=True,
        cluster_shape_mn=(4, 1),
        shared_expert=Sm107SharedExpertConfig(128) if shared else None,
    )
    kernel = pkg.Sm107BlockScaledSymmBuffer._build_kernel(cfg)
    assert kernel._topk_reduce.ll_encoded is (topk < 32)
    assert kernel._topk_reduce.outlive_producer is (shared and topk < 32)


def test_dedup_candidate_grid_preserves_tiles_and_satisfies_return_contract():
    cfg = _native_config(
        combine_dedup=True, fc2_use_bulk=True, token_back_mode="reuse_dispatch_warps"
    )
    ordinary = pkg.sm107_candidates("mxfp8_e4m3")
    candidates = pkg.sm107_candidates("mxfp8_e4m3", combine_dedup=True)
    assert candidates
    assert {c["mma_tiler_mnk"] for c in candidates} <= {
        c["mma_tiler_mnk"] for c in ordinary
    }
    assert all(pkg.is_valid_sm107(c, cfg) for c in candidates)


def test_dedup_cache_miss_returns_an_admitted_plan(monkeypatch):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    knobs, source = pkg.resolve_knobs(
        dtype="mxfp8_e4m3",
        world_size=4,
        hidden=512,
        intermediate=256,
        num_experts=32,
        topk=4,
        max_tokens=32,
        combine_dedup=True,
    )
    assert source == "heuristic"
    assert pkg.is_valid_sm107(
        knobs,
        _native_config(
            combine_dedup=True,
            fc2_use_bulk=True,
            token_back_mode="reuse_dispatch_warps",
        ),
    )


@pytest.fixture(scope="module")
def ep_runtime():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires native SM107")
    from flashinfer.moe_ep.core.runtime import (
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
        sm107_block_scaled_runtime_requirements,
    )

    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    if world < 2:
        pytest.skip("requires torchrun with at least two ranks")
    bootstrap = BootstrapConfig(rank=rank, world_size=world, auto_bootstrap=False)
    ensure_moe_ep_cuda_device(bootstrap)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, sm107_block_scaled_runtime_requirements(bootstrap)
    )
    try:
        yield bootstrap
    finally:
        finalize_moe_ep_runtime(runtime)


def _reference(workspace, weights, ids, scores, *, shared=False):
    cfg = workspace.config
    kind = cfg.shared_expert.kind if shared else cfg.quant_kind
    x = workspace.shared_x if shared else workspace.x
    sf = workspace.shared_x_sf_linear if shared else workspace.x_sf
    return pkg.compute_megamoe_reference_sm107_block_scaled(
        x,
        sf,
        ids,
        scores,
        weights[0][0].transpose(1, 2),
        weights[0][1],
        weights[1][0].transpose(1, 2),
        weights[1][1],
        quant_kind=kind,
        local_expert_offset=0,
        gate_up_clamp=None,
        apply_routing_weights_before_fc2=False
        if shared
        else cfg.apply_routing_weights_before_fc2,
        num_tokens=ids.shape[0],
        weight_scales_are_swizzled=True,
        return_fp32=True,
        situ_beta=cfg.situ_beta,
        situ_linear_beta=cfg.situ_linear_beta,
    )[: ids.shape[0]]


_BASE_FEATURE_CASES = [
    (kind, variant, shared, dedup, False, False, "mxfp8_e4m3")
    for kind in _KINDS
    for variant, shared, dedup in (
        ("inference", True, False),
        ("genphase", True, False),
        ("inference", False, True),
        ("inference", True, True),
    )
] + [
    ("nvfp4", "inference", True, False, True, False, "mxfp8_e4m3"),
    ("nvfp4", "genphase", True, False, True, True, "mxfp8_e4m3"),
    ("mxfp8_e4m3", "inference", True, False, True, True, "mxfp8_e4m3"),
    ("mxfp4_mxfp8", "genphase", True, False, True, False, "mxfp8_e5m2"),
    ("mxfp8_e4m3", "inference", True, False, False, True, "mxfp8_e5m2"),
    ("mxfp4", "genphase", True, False, False, False, "mxfp8_e5m2"),
]

_FEATURE_CASES = [(*case, "swiglu") for case in _BASE_FEATURE_CASES] + [
    ("mxfp4_mxfp8", "inference", True, True, False, False, "mxfp8_e4m3", "situ"),
    ("mxfp4_mxfp8", "genphase", True, False, False, False, "mxfp8_e4m3", "situ"),
]


def _quantized_input(kind, x):
    if kind == "nvfp4":
        return pkg.quantize_nvfp4_block16(x)
    if kind == "mxfp4":
        return pkg.quantize_mxfp4_block32(x)
    return pkg.quantize_mxfp8_block32(
        x, torch.float8_e5m2 if kind == "mxfp8_e5m2" else torch.float8_e4m3fn
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize(
    "kind,variant,shared,dedup,prequantized,shared_prequantized,shared_kind,activation",
    _FEATURE_CASES,
)
def test_public_shared_and_dedup_graph(
    ep_runtime,
    kind,
    variant,
    shared,
    dedup,
    prequantized,
    shared_prequantized,
    shared_kind,
    activation,
    geometry=(512, 256, 32, 32, 4, 128),
    tune=False,
    return_mode=None,
    shared_settings=None,
    native_defaults=False,
    zero_rank=False,
    reuse_routed_shared_input=False,
    raw_scales=False,
):
    """Compare captured public forwards after route and shared-weight changes."""
    bootstrap = ep_runtime
    rank, world = bootstrap.rank, bootstrap.world_size
    hidden, intermediate, experts, capacity, topk, shared_intermediate = geometry
    config_cls = next(
        cls
        for cls, _, name in _BACKENDS
        if (
            (kind == "nvfp4" and "nvfp4_nvfp4" in name)
            or (kind == "mxfp4" and "mxfp4_mxfp4" in name)
            or (kind == "mxfp4_mxfp8" and "mxfp8_mxfp4" in name)
            or (kind.startswith("mxfp8_") and "mxfp8_mxfp8" in name)
        )
    )
    kwargs = dict(
        kernel_variant=variant,
        fc2_use_bulk=True,
        cluster_shape_mn=(4, 1),
        apply_routing_weights_before_fc2=dedup,
        combine_dedup=dedup,
        token_back_mode=return_mode
        or ("reuse_dispatch_warps" if dedup else "epi_warps"),
        shared_expert=Sm107SharedExpertConfig(
            shared_intermediate, kind=shared_kind, **(shared_settings or {})
        )
        if shared
        else None,
        token_in_flag_batch=1 if variant == "genphase" else 2,
    )
    if activation == "situ":
        kwargs.update(activation="situ", situ_beta=4.0, situ_linear_beta=25.0)
    if tune:
        kwargs["knobs"] = "auto"
    if shared_settings and shared_settings.get("work_id_mode") == "atomic_counter":
        kwargs["work_id_mode"] = "atomic_counter"
    if native_defaults:
        kwargs = {"shared_expert": kwargs["shared_expert"]}
    if kind.startswith("mxfp8_"):
        kwargs["kind"] = kind
    torch.manual_seed(18927)
    w1 = (
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(experts, hidden, intermediate, device="cuda") / intermediate**0.5
    ).bfloat16()
    sw1 = (
        torch.randn(1, 2 * shared_intermediate, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    sw2 = (
        torch.randn(1, hidden, shared_intermediate, device="cuda")
        / shared_intermediate**0.5
    ).bfloat16()
    local = slice(rank * (experts // world), (rank + 1) * (experts // world))
    source = MoEWeightPack(
        w1[local], w2[local], shared_expert=MoEWeightPack(sw1, sw2) if shared else None
    )
    all_weights = pkg.preprocess_block_scaled_weights(
        w1, w2, quant_kind=kind, intermediate_size=intermediate
    )
    layer = MoEEpLayer(
        bootstrap,
        FleetParams(
            num_experts=experts, max_tokens_per_rank=capacity, token_hidden_size=hidden
        ),
        source,
        backend=MegaConfig(
            megakernel=config_cls(intermediate, topk, **kwargs),
            quantize_input=not prequantized,
        ),
    )
    count = 0 if zero_rank and rank == 0 else 9 + rank % 4
    torch.manual_seed(18928 + rank)
    x = torch.randn(count, hidden, device="cuda", dtype=torch.bfloat16)
    # Several experts on one remote rank exercise combine deduplication.
    owner = (rank + 1) % world
    ids = (
        torch.arange(count * topk, device="cuda", dtype=torch.int32)
        .reshape(count, topk)
        .remainder(max(experts // world, topk))
        .add(owner * (experts // world))
        .remainder(experts)
    )
    scores = torch.linspace(0.1, 0.7, count * topk, device="cuda").reshape(count, topk)
    qx, sx = _quantized_input(kind, x) if prequantized else (x, None)
    if raw_scales and sx is not None:
        sx = sx.view(torch.uint8)
    tensors = MoEEpTensors(qx, ids, scores, scales=sx)
    if (
        shared
        and (prequantized or shared_prequantized)
        and not reuse_routed_shared_input
    ):
        if shared_prequantized:
            tensors.shared_hidden_states, tensors.shared_scales = _quantized_input(
                shared_kind, x
            )
        else:
            tensors.shared_hidden_states = x
    try:
        from flashinfer import autotune

        with (
            autotune(cuda_graph_profile_replays=2) if tune else contextlib.nullcontext()
        ):
            layer.warmup(tensors)
        if tune:
            assert layer._workspace._tuning_state.winner is not None
        workspace = getattr(layer._workspace, "workspace", layer._workspace)
        if shared:
            assert isinstance(layer._transformed, Sm107SharedExpertWeights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(tensors)
        for mutation in range(3):
            if mutation == 1:
                x.mul_(-0.5)
                if prequantized:
                    q, sf = _quantized_input(kind, x)
                    tensors.hidden_states.view(torch.uint8).copy_(q.view(torch.uint8))
                    tensors.scales.view(torch.uint8).copy_(sf.view(torch.uint8))
                if shared_prequantized:
                    q, sf = _quantized_input(shared_kind, x)
                    tensors.shared_hidden_states.view(torch.uint8).copy_(
                        q.view(torch.uint8)
                    )
                    tensors.shared_scales.view(torch.uint8).copy_(sf.view(torch.uint8))
                ids[::2, 1] = -1
                scores[::2, 1] = float("nan")
                if shared:
                    replacement = pkg.preprocess_block_scaled_weights(
                        sw1 * 0.5,
                        sw2 * 1.25,
                        quant_kind=shared_kind,
                        intermediate_size=shared_intermediate,
                    )
                    for old, new in zip(
                        layer._transformed.shared, replacement, strict=True
                    ):
                        for dst, src in zip(old, new, strict=True):
                            dst.view(torch.uint8).copy_(src.view(torch.uint8))
            elif mutation == 2:
                ids.fill_(-1)
                scores.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            expected = _reference(workspace, all_weights, ids, scores)
            if shared:
                shared_ids = torch.zeros((count, 1), device="cuda", dtype=torch.int32)
                expected += _reference(
                    workspace,
                    layer._transformed.shared,
                    shared_ids,
                    torch.ones_like(shared_ids, dtype=torch.float32),
                    shared=True,
                )
            actual = output[:count].float()
            relative_l2 = torch.linalg.vector_norm(
                actual - expected
            ) / torch.linalg.vector_norm(expected).clamp_min(1e-6)
            # Agree before raising so every rank can leave collective state.
            failures = torch.tensor(
                [
                    int(
                        not torch.isfinite(actual).all()
                        or not relative_l2
                        < (0.06 if kind in ("nvfp4", "mxfp4") else 0.02)
                    )
                ],
                device="cuda",
            )
            if world > 1:
                torch.distributed.all_reduce(failures)
            assert not failures.item(), (
                kind,
                variant,
                shared,
                dedup,
                mutation,
                relative_l2.item(),
            )
        empty = dataclasses.replace(
            tensors,
            hidden_states=tensors.hidden_states[:0],
            topk_ids=ids[:0],
            topk_weights=scores[:0],
            scales=None if tensors.scales is None else tensors.scales[:0],
            shared_hidden_states=None
            if tensors.shared_hidden_states is None
            else tensors.shared_hidden_states[:0],
            shared_scales=None
            if tensors.shared_scales is None
            else tensors.shared_scales[:0],
        )
        assert layer(empty).shape == (0, hidden)
        torch.cuda.synchronize()
    finally:
        layer.destroy()


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("variant", ["inference", "genphase"])
@pytest.mark.parametrize("raw_scales", [False, True])
def test_public_shared_reuses_routed_scales(ep_runtime, kind, variant, raw_scales):
    test_public_shared_and_dedup_graph(
        ep_runtime,
        kind,
        variant,
        True,
        False,
        True,
        False,
        kind,
        "swiglu",
        reuse_routed_shared_input=True,
        raw_scales=raw_scales,
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize("topk", [31, 32, 33])
def test_public_shared_genphase_static_reduce_graph(ep_runtime, topk):
    test_public_shared_and_dedup_graph(
        ep_runtime,
        "mxfp8_e4m3",
        "genphase",
        True,
        False,
        False,
        False,
        "mxfp8_e4m3",
        "swiglu",
        geometry=(512, 256, 64, 32, topk, 128),
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4_mxfp8"])
@pytest.mark.parametrize(
    "variant,shared,dedup",
    [
        ("inference", True, False),
        ("genphase", True, False),
        ("inference", False, True),
        ("inference", True, True),
    ],
)
def test_public_shared_and_dedup_graph_padded_scales(
    ep_runtime, kind, variant, shared, dedup
):
    """Exercise wider GEMMs whose scale rows need physical padding."""
    test_public_shared_and_dedup_graph(
        ep_runtime,
        kind,
        variant,
        shared,
        dedup,
        False,
        False,
        "mxfp8_e4m3",
        "swiglu",
        geometry=(3072, 1024, 64, 128, 4, 256),
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize(
    "variant,dedup",
    [
        ("inference", False),
        ("genphase", False),
        ("auto", False),
        ("inference", True),
        ("auto", True),
    ],
)
def test_public_shared_and_dedup_graph_tuning(ep_runtime, variant, dedup, monkeypatch):
    """Keep shared bindings through candidate trials and variant installation."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import autotune as native

    candidates = pkg.sm107_candidates

    def one_candidate(*args, **kwargs):
        return [
            candidate
            for candidate in candidates(*args, **kwargs)
            if candidate["fc2_use_bulk"] and not candidate["reduce_topk_in_kernel"]
        ][:1]

    monkeypatch.setattr(pkg, "sm107_candidates", one_candidate)
    monkeypatch.setattr(native, "sm107_candidates", one_candidate)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    try:
        test_public_shared_and_dedup_graph(
            ep_runtime,
            "mxfp4_mxfp8",
            variant,
            True,
            dedup,
            False,
            False,
            "mxfp8_e4m3",
            "swiglu",
            tune=True,
        )
    finally:
        tuner.clear_cache()


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4_mxfp8"])
@pytest.mark.parametrize("shared", [False, True])
def test_public_shared_and_dedup_graph_standalone(ep_runtime, kind, shared):
    test_public_shared_and_dedup_graph(
        ep_runtime,
        kind,
        "inference",
        shared,
        True,
        False,
        False,
        "mxfp8_e4m3",
        "swiglu",
        return_mode="standalone_warps",
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp4_mxfp8"])
def test_public_shared_and_dedup_graph_defaults(ep_runtime, kind):
    test_public_shared_and_dedup_graph(
        ep_runtime,
        kind,
        "inference",
        True,
        False,
        False,
        False,
        "mxfp8_e4m3",
        "swiglu",
        native_defaults=True,
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize(
    "variant,shared,dedup",
    [
        ("inference", True, False),
        ("genphase", True, False),
        ("inference", False, True),
        ("inference", True, True),
    ],
)
def test_public_shared_and_dedup_graph_empty_rank(ep_runtime, variant, shared, dedup):
    """A rank with no local tokens still computes other ranks' routed experts."""
    test_public_shared_and_dedup_graph(
        ep_runtime,
        "mxfp4_mxfp8",
        variant,
        shared,
        dedup,
        False,
        False,
        "mxfp8_e4m3",
        "swiglu",
        zero_rank=True,
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize("variant", ["inference", "genphase"])
@pytest.mark.parametrize(
    "work_id_mode,position", [("grid_stride", "tail"), ("atomic_counter", "early")]
)
def test_public_shared_and_dedup_graph_shared_schedule(
    ep_runtime, variant, work_id_mode, position
):
    test_public_shared_and_dedup_graph(
        ep_runtime,
        "mxfp4_mxfp8",
        variant,
        True,
        False,
        False,
        False,
        "mxfp8_e4m3",
        "swiglu",
        geometry=(3072, 1024, 64, 128, 4, 256),
        shared_settings=dict(
            l2_position=position, work_id_mode=work_id_mode, fc1_swizzle=1
        ),
    )


@pytest.mark.arch_rubin
@pytest.mark.gpu_2
@pytest.mark.parametrize(
    "variant,dedup", [("inference", False), ("inference", True), ("genphase", False)]
)
def test_public_shared_and_dedup_graph_pooled_weights(
    ep_runtime, variant, dedup, monkeypatch
):
    """Two captured layers share staging, with distinct prepared shared weights."""
    from tests.moe_ep import test_moe_ep_sm107_block_scaled_mega_multirank as problem
    from tests.moe_ep.sm107_test_utils import weight_pack

    for name, value in dict(
        HIDDEN=512, INTERMEDIATE=256, NUM_EXPERTS=32, NUM_TOKENS=16, MAX_TOKENS=32
    ).items():
        monkeypatch.setattr(problem, name, value)
    rank, world = ep_runtime.rank, ep_runtime.world_size
    w1, w2, x, ids, scores = problem._global_problem(world)
    local = slice(rank * (32 // world), (rank + 1) * (32 // world))
    kind = "mxfp4_mxfp8"
    cfg = problem._megakernel_config(
        kind,
        kernel_variant=variant,
        combine_dedup=dedup,
        apply_routing_weights_before_fc2=dedup,
        fc2_use_bulk=True,
        cluster_shape_mn=(4, 1),
        token_back_mode="reuse_dispatch_warps" if dedup else "epi_warps",
        shared_expert=Sm107SharedExpertConfig(256),
    )
    routed = pkg.preprocess_block_scaled_weights(
        w1, w2, quant_kind=kind, intermediate_size=256
    )
    shared_banks = [(w1[:1], w2[:1]), (-w1[:1], w2[:1] * 0.75)]
    shared_prepared = [
        pkg.preprocess_block_scaled_weights(
            a, b, quant_kind="mxfp8_e4m3", intermediate_size=256
        )
        for a, b in shared_banks
    ]
    layers = []
    tensors = MoEEpTensors(x[rank].clone(), ids[rank].clone(), scores[rank].clone())
    try:
        for index, (a, b) in enumerate(shared_banks):
            child, _, _ = weight_pack(a, b, "mxfp8_e4m3")
            prepared = (
                Sm107SharedExpertWeights(
                    tuple((data[local], sf[local]) for data, sf in routed),
                    shared_prepared[index],
                )
                if index
                else None
            )
            layer = MoEEpLayer(
                ep_runtime,
                FleetParams(
                    num_experts=32, max_tokens_per_rank=32, token_hidden_size=512
                ),
                MoEWeightPack(w1[local], w2[local], shared_expert=child),
                backend=MegaConfig(megakernel=cfg, transformed_weights=prepared),
            )
            layers.append(layer)
            layer.warmup(tensors)
        workspace = layers[0]._workspace
        assert workspace is layers[1]._workspace
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            outputs = [layer(tensors) for layer in layers]
        for iteration in range(3):
            tensors.hidden_states.copy_(x[rank] * (1 + 0.2 * iteration))
            tensors.topk_ids.copy_((ids[rank] + iteration) % 32)
            tensors.topk_ids[::3] = -1
            tensors.topk_weights[::3] = float("nan")
            graph.replay()
            torch.cuda.synchronize()
            routed_reference = _reference(
                workspace, routed, tensors.topk_ids, tensors.topk_weights
            )
            shared_ids = torch.zeros((16, 1), device="cuda", dtype=torch.int32)
            errors = []
            for output, shared_weights in zip(outputs, shared_prepared, strict=True):
                expected = routed_reference + _reference(
                    workspace,
                    shared_weights,
                    shared_ids,
                    torch.ones_like(shared_ids, dtype=torch.float32),
                    shared=True,
                )
                errors.append(
                    (output.float() - expected).norm() / expected.norm().clamp_min(1e-6)
                )
            failed = torch.tensor(
                [int(any(not error < 0.02 for error in errors))], device="cuda"
            )
            torch.distributed.all_reduce(failed)
            assert not failed.item(), [error.item() for error in errors]
        del graph, outputs
    finally:
        for layer in layers:
            layer.destroy()
