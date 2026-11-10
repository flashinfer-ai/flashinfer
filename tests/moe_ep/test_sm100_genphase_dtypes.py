"""Blackwell GenPhase dtype admission, layouts, staging and graph replay."""

import dataclasses
import pytest
import torch

from flashinfer import moe_ep
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

KINDS = ("mxfp8_e4m3", "mxfp8_e5m2", "nvfp4", "mxfp4_mxfp8")


@pytest.mark.parametrize(
    "kind,selector",
    [
        (kind, selector)
        for kind in KINDS
        for selector in ("legacy", "next")
        if kind != "nvfp4" or selector == "next"
    ],
)
@pytest.mark.parametrize(
    "finalize_mode", [moe_ep.FinalizeMode.AUTO, moe_ep.FinalizeMode.DEFAULT]
)
@pytest.mark.parametrize("with_live_count", [False, True])
def test_genphase_finalize_live_count_admission(
    kind, selector, finalize_mode, with_live_count
):
    """AUTO must not silently discard a requested persistent live-token count."""
    count = torch.zeros(1, dtype=torch.int32) if with_live_count else None
    options = dict(
        intermediate_size=256,
        top_k=3,
        kernel_variant="genphase",
        finalize_mode=finalize_mode,
        num_valid_tokens_tensor=count,
    )
    if kind == "nvfp4":
        config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next", **options
        )
    else:
        config = moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            kind=kind,
            kernel_name="sm100_mxfp8_mxfp8_bf16_cutedsl"
            + ("_next" if selector == "next" else ""),
            **options,
        )
    if finalize_mode is moe_ep.FinalizeMode.AUTO and with_live_count:
        with pytest.raises(moe_ep.MoEEpConfigError, match="finalize"):
            create_mega_kernel(config)
    else:
        backend = create_mega_kernel(config)
        if selector == "next":
            assert backend._finalize_mode(None) is moe_ep.FinalizeMode.DEFAULT


def _config(kind, **kwargs):
    """Build a small GenPhase workload with dtype-compatible intermediate width."""
    return pkg.Sm100GenphaseMoeConfig(
        num_total_experts=8,
        max_tokens_per_rank=129,
        num_topk=3,
        hidden=384,
        intermediate=256 if kind == "mxfp4_mxfp8" else 192,
        rank=0,
        world_size=1,
        quant_kind=kind,
        **kwargs,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_genphase_dtype_geometries_and_workspace_plan(kind, monkeypatch):
    """Every tuning geometry must build a native 512-column Blackwell plan."""
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        block_scaled,
    )

    monkeypatch.setattr(
        block_scaled, "_max_active_clusters", lambda cluster: 148 // cluster
    )
    cfg = _config(kind)
    candidates = pkg.sm100_genphase_candidates(cfg, sm_count=148, live_tokens=4)
    assert cfg.instruction_k == (64 if kind == "nvfp4" else 32)
    assert cfg.gather_tile_k == (256 if kind == "nvfp4" else 128)
    geometries = {(x["mma_tiler_mnk"], x["cluster_shape_mn"]) for x in candidates}
    assert geometries == set(cfg.admitted_geometries)
    for tile, cluster in geometries:
        resolved = dataclasses.replace(
            cfg, mma_tiler_mnk=tile, cluster_shape_mn=cluster
        )
        kernel = pkg.Sm100GenphaseSymmBuffer._build_kernel(resolved)
        assert kernel._mainloop.tmem_plan.allocation_columns == 512
        assert kernel._mainloop.instruction_k == cfg.instruction_k
        assert kernel.epilogue.fc2_tma_stages == min(2, tile[1] // 64)
        assert tile[2] * (4 if kind == "nvfp4" else 8) // 8 == 128


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("selector", ["legacy", "next"])
def test_genphase_backend_dtype_selection(kind, selector, monkeypatch):
    """Public configs retain the requested dtype throughout workspace selection."""
    if selector == "legacy" and kind == "nvfp4":
        pytest.skip("NVFP4 GenPhase uses the next selector")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    options = dict(intermediate_size=256, top_k=3, kernel_variant="genphase")
    if kind == "nvfp4":
        config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next", **options
        )
    else:
        config = moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            kind=kind,
            kernel_name="sm100_mxfp8_mxfp8_bf16_cutedsl"
            + ("_next" if selector == "next" else ""),
            **options,
        )
    backend = create_mega_kernel(config)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=8, max_tokens_per_rank=129, token_hidden_size=384
    )
    cfg = backend._genphase_config(fleet)
    assert cfg.quant_kind == kind
    assert cfg.fc2_use_bulk
    assert not cfg.reduce_topk_in_kernel
    if selector == "next":
        assert backend._workspace_config(fleet) == cfg


@pytest.mark.parametrize("kind", KINDS)
def test_genphase_prequantized_weight_layout(kind):
    """Canonical prequantized scales and packed FP4 produce the same native layout."""
    cfg = _config(kind)
    torch.manual_seed(913)
    w13 = torch.randn(2, 2 * cfg.intermediate, cfg.hidden).bfloat16()
    w2 = torch.randn(2, cfg.hidden, cfg.intermediate).bfloat16()
    quantize = (
        pkg.quantize_nvfp4_block16
        if kind == "nvfp4"
        else (
            pkg.quantize_mxfp4_block32
            if kind == "mxfp4_mxfp8"
            else lambda x: pkg.quantize_mxfp8_block32(x, cfg.torch_weight_data_dtype)
        )
    )
    q13, sf13 = quantize(w13.float())
    q2, sf2 = quantize(w2.float())
    expected = pkg.preprocess_genphase_weights(
        w13, w2, quant_kind=kind, intermediate_size=cfg.intermediate
    )
    actual = pkg.preprocess_genphase_prequantized_weights(
        q13,
        q2,
        sf13,
        sf2,
        quant_kind=kind,
        hidden_size=cfg.hidden,
        intermediate_size=cfg.intermediate,
    )
    # Exercise the public backend's raw-byte scale normalization as well.
    options = dict(
        intermediate_size=cfg.intermediate,
        top_k=cfg.num_topk,
        kernel_variant="genphase",
    )
    kernel_config = (
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next", **options
        )
        if kind == "nvfp4"
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(kind=kind, **options)
    )
    backend = create_mega_kernel(kernel_config)
    raw = backend.preprocess_weights(
        moe_ep.PrequantizedMoEWeights(
            w13=q13,
            w2=q2,
            w13_scale=sf13.view(torch.uint8),
            w2_scale=sf2.view(torch.uint8),
        ),
        moe_ep.FleetParams(
            num_experts=2, max_tokens_per_rank=129, token_hidden_size=cfg.hidden
        ),
    )
    for leg_a, leg_b in zip(raw, actual, strict=True):
        for a, b in zip(leg_a, leg_b, strict=True):
            torch.testing.assert_close(
                a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0
            )
    for leg_a, leg_b in zip(actual, expected, strict=True):
        for a, b in zip(leg_a, leg_b, strict=True):
            torch.testing.assert_close(
                a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0
            )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "kind,tile_n,cluster_m",
    [
        ("mxfp8_e5m2", 128, 4),
        ("mxfp8_e5m2", 64, 4),
        ("mxfp8_e5m2", 64, 2),
        ("nvfp4", 128, 4),
        ("mxfp4_mxfp8", 128, 4),
        ("mxfp4_mxfp8", 64, 4),
    ],
)
@pytest.mark.parametrize("early", [True, False])
@pytest.mark.parametrize("capacity", [1, 129])
def test_genphase_dtype_graph(kind, tile_n, cluster_m, early, capacity, monkeypatch):
    """Check quantized bytes against Torch and replay sparse, masked and dense routing."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/103")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    from .test_nvfp4_cutedsl_kernel_vs_reference import _run_blackwell_genphase_graph

    _run_blackwell_genphase_graph(0, 1, capacity, early, tile_n, cluster_m, kind)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("early", [True, False])
@pytest.mark.parametrize("quantize_input", [True, False])
def test_genphase_layer_input_formats(kind, early, quantize_input, monkeypatch):
    """The public layer stages BF16/prequantized inputs and replays scale changes."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/103")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    cfg = _config(kind)
    options = dict(
        intermediate_size=cfg.intermediate,
        top_k=cfg.num_topk,
        kernel_variant="genphase",
        apply_routing_weights_before_fc2=early,
    )
    if kind == "nvfp4":
        kernel_config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next", **options
        )
    else:
        kernel_config = moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            kind=kind, **options
        )
    torch.manual_seed(192)
    x = torch.randn(17, cfg.hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(17 * cfg.num_topk, device="cuda").view(17, cfg.num_topk) % 8
    weights = moe_ep.MoEWeightPack(
        (
            torch.randn(8, 2 * cfg.intermediate, cfg.hidden, device="cuda")
            / cfg.hidden**0.5
        ).bfloat16(),
        (
            torch.randn(8, cfg.hidden, cfg.intermediate, device="cuda")
            / cfg.intermediate**0.5
        ).bfloat16(),
    )
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=1, rank=0),
        moe_ep.FleetParams(
            num_experts=8, max_tokens_per_rank=129, token_hidden_size=cfg.hidden
        ),
        weights,
        backend=moe_ep.MegaConfig(
            megakernel=kernel_config, quantize_input=quantize_input
        ),
    )
    tensors = moe_ep.MoEEpTensors(
        hidden_states=x,
        topk_ids=ids.clone(),
        topk_weights=torch.full((17, cfg.num_topk), 1 / cfg.num_topk, device="cuda"),
    )
    graph = None
    try:
        if not quantize_input:
            tensors.hidden_states, tensors.scales = (
                pkg.quantize_nvfp4_block16(x.float())
                if kind == "nvfp4"
                else pkg.quantize_mxfp8_block32(x.float(), cfg.torch_act_data_dtype)
            )
        if kind == "nvfp4":
            tensors.fc1_alpha = torch.linspace(0.5, 1.5, 8, device="cuda")
            tensors.fc2_alpha = torch.linspace(1.5, 0.5, 8, device="cuda")
            tensors.fc1_norm_const = torch.linspace(0.8, 1.2, 8, device="cuda")
        layer.warmup(tensors)
        ws = layer._ensure_workspace()

        def check(output):
            indices, expected = pkg.sampled_reference(
                ws, *layer._transformed, 17, sample_size=17
            )
            assert pkg.output_error(output, indices, expected) < 0.025

        check(layer.forward(tensors))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer.forward(tensors, return_workspace_view=True)
        for masked in (True, False, True, False):
            tensors.topk_ids.fill_(-1) if masked else tensors.topk_ids.copy_(ids)
            if kind == "nvfp4":
                tensors.fc1_alpha.add_(0.1)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            check(output)
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()
