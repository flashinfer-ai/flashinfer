"""Fixed knob selection for shared-weight MegaMoE workspace profiles."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe import (
    Sm100GenphaseMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import (
    default_sm100_genphase_knobs,
    is_valid_sm100_genphase_knobs,
    sm100_genphase_candidates,
)

from tests.cpu.moe_ep.test_mega_workspace_api import _inputs


def _mxfp8_backend(*, capacity=129, hidden=384, **kwargs):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    config = Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=192, top_k=3, **kwargs
    )
    backend = create_mega_kernel(config)
    bootstrap = BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    backend.bind_ep_bootstrap(bootstrap)
    fleet = FleetParams(
        num_experts=8, max_tokens_per_rank=capacity, token_hidden_size=hidden
    )
    return backend, bootstrap, fleet


@pytest.mark.parametrize(
    "tile,cluster", [(None, None), ((256, 64, 128), (2, 1)), ((256, 64, 128), (4, 1))]
)
def test_blackwell_genphase_backend_selects_own_workspace(
    tile, cluster, monkeypatch, dist_not_initialized
):
    """GenPhase allocation and pool identity stay separate from legacy inference."""
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    kernel, bootstrap, fleet = _mxfp8_backend(
        kernel_variant="genphase",
        mma_tiler_mnk=tile,
        cluster_shape_mn=cluster,
    )
    kernel.validate_init(bootstrap, fleet)
    monkeypatch.setattr(
        kernel, "_tuning_cache", mock.Mock(side_effect=AssertionError("legacy cache"))
    )
    assert kernel.live_token_buckets(fleet) == []
    allocator = mock.Mock(return_value=mock.sentinel.workspace)
    monkeypatch.setattr(pkg, "Sm100GenphaseSymmBuffer", allocator)
    assert kernel._allocate_workspace(fleet) is mock.sentinel.workspace
    config = allocator.call_args.args[0]
    assert config.resolved_mma_tiler_mnk == (tile or (256, 64, 128))
    assert config.cluster_shape_mn == (cluster or (4, 1))
    assert config.fc2_use_bulk and not config.reduce_topk_in_kernel
    assert kernel._workspace_pool_key(fleet)[-1] == config
    explicit, _, _ = _mxfp8_backend(
        kernel_variant="genphase",
        mma_tiler_mnk=config.resolved_mma_tiler_mnk,
        cluster_shape_mn=config.cluster_shape_mn,
    )
    assert kernel._workspace_pool_key(fleet) == explicit._workspace_pool_key(fleet)
    legacy, _, _ = _mxfp8_backend()
    monkeypatch.setattr(legacy, "_workspace_knobs", lambda fp: None)
    assert kernel._workspace_pool_key(fleet) != legacy._workspace_pool_key(fleet)
    other, _, _ = _mxfp8_backend(
        kernel_variant="genphase",
        mma_tiler_mnk=(256, 64, 128),
        cluster_shape_mn=(4, 1) if cluster == (2, 1) else (2, 1),
    )
    assert kernel._workspace_pool_key(fleet) != other._workspace_pool_key(fleet)


@pytest.mark.parametrize(
    "tile,cluster,hint",
    [
        (None, None, 148),
        ((256, 128, 128), (4, 1), 148),
        ((256, 64, 128), (4, 1), 148),
        ((256, 64, 128), (2, 1), 296),
    ],
)
def test_blackwell_genphase_schedule_heuristic(tile, cluster, hint, monkeypatch):
    """Explicit geometry constrains schedule estimation and workspace pooling."""
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: mock.Mock(major=10, minor=3, multi_processor_count=148),
    )

    def make(**overrides):
        """Bind a GenPhase backend with selected geometry and optional overrides."""
        params = dict(
            intermediate_size=1024,
            top_k=8,
            kernel_variant="genphase",
            mma_tiler_mnk=tile,
            cluster_shape_mn=cluster,
        )
        params.update(overrides)
        backend = create_mega_kernel(
            Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(**params)
        )
        backend.bind_ep_bootstrap(
            BootstrapConfig(world_size=2, rank=0, auto_bootstrap=False)
        )
        return backend

    fleet = FleetParams(
        num_experts=256, max_tokens_per_rank=1024, token_hidden_size=4096
    )
    kernel = make()
    resolved = kernel._genphase_config(fleet)
    assert resolved.schedule_policy == ("grouped", hint)
    explicit = make(schedule_policy=("grouped", hint))
    original = make(schedule_policy=("grouped", None))
    assert kernel._workspace_pool_key(fleet) == explicit._workspace_pool_key(fleet)
    assert kernel._workspace_pool_key(fleet) != original._workspace_pool_key(fleet)

    assert original._genphase_config(fleet).schedule_policy == ("grouped", None)
    # The rule depends on geometry and hardware, not one benchmark shape.
    for kwargs in (
        dict(top_k=6),
        dict(intermediate_size=2048),
        dict(gate_up_clamp=10.0),
    ):
        assert make(**kwargs)._genphase_config(fleet).schedule_policy == (
            "grouped",
            hint,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"kernel_variant": "unknown"},
        {"kernel_variant": "genphase", "capacity": 1025},
        {"kernel_variant": "genphase", "knobs": "bad"},
        {"kernel_variant": "genphase", "schedule_policy": ("grouped", 0)},
        {"kernel_variant": "genphase", "schedule_policy": ("grouped", -1)},
        {"kernel_variant": "genphase", "schedule_policy": ("grouped", True)},
        {"kernel_variant": "genphase", "schedule_policy": ("grouped", 1.5)},
        {"kernel_variant": "genphase", "schedule_policy": ("phase_interleave", None)},
        {"kernel_variant": "genphase", "schedule_policy": ("grouped",)},
        {"kernel_variant": "genphase", "schedule_policy": ["grouped", 148]},
        {"schedule_policy": ("grouped", None)},
        {"kernel_variant": "genphase", "knobs": {"group_hint": 4}},
        {
            "kernel_variant": "genphase",
            "mma_tiler_mnk": (256, 64, 128),
            "knobs": {"mma_tiler_mnk": (256, 128, 128)},
        },
        {"kernel_variant": "genphase", "enable_in_kernel_fc2_reduce": True},
        {"kernel_variant": "genphase", "hidden": 192},
        {
            "kernel_variant": "genphase",
            "mma_tiler_mnk": (256, 128, 128),
            "cluster_shape_mn": (2, 1),
        },
        {"mma_tiler_mnk": (256, 64, 128)},
    ],
)
def test_blackwell_genphase_backend_rejects_unsupported_config(
    overrides, monkeypatch, dist_not_initialized
):
    from flashinfer.moe_ep import MoEEpConfigError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(MoEEpConfigError):
        kernel, bootstrap, fleet = _mxfp8_backend(**overrides)
        kernel.validate_init(bootstrap, fleet)


@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.uint8])
def test_blackwell_genphase_backend_weights_and_layout_validation(
    scale_dtype, dist_not_initialized
):
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    generator = torch.Generator().manual_seed(6447)
    w13 = torch.randn(8, 384, 384, generator=generator).bfloat16()
    w2 = torch.randn(8, 384, 192, generator=generator).bfloat16()
    kernel, _, fleet = _mxfp8_backend(kernel_variant="genphase")
    transformed = kernel.preprocess_weights(MoEWeightPack(w13, w2), fleet)
    q13, sf13 = pkg.quantize_genphase_mxfp8(w13, torch.float8_e4m3fn)
    q2, sf2 = pkg.quantize_genphase_mxfp8(w2, torch.float8_e4m3fn)
    packed = kernel.preprocess_weights(
        MoEWeightPack(q13, q2, sf13.view(scale_dtype), sf2.view(scale_dtype)), fleet
    )
    for expected, actual in zip(transformed, packed, strict=True):
        for a, b in zip(expected, actual, strict=True):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    config = kernel._genphase_config(fleet)
    pkg.validate_genphase_weights(transformed, config, torch.device("cpu"))
    legacy = preprocess_mega_weights(
        MoEWeightPack(w13, w2), intermediate_size=192, hidden_size=384
    )
    with pytest.raises(ValueError, match="K-major"):
        pkg.validate_genphase_weights(legacy, config, torch.device("cpu"))
    with pytest.raises(ValueError, match="tuple"):
        pkg.validate_genphase_weights(None, config, torch.device("cpu"))
    bad = ((transformed[0][0], transformed[0][1].view(torch.uint8)), transformed[1])
    with pytest.raises(ValueError, match="scales"):
        pkg.validate_genphase_weights(bad, config, torch.device("cpu"))


@pytest.fixture
def make_layer(monkeypatch):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FinalizeMode,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl import (
        backend,
    )
    from flashinfer.moe_ep.core.kernel import workspace_pool

    cls = backend.Nvfp4CutedslMegaKernelBackend
    monkeypatch.setattr(backend, "validate_mega_arch", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cls, "_cake_reducer_supported", lambda *args: False)
    monkeypatch.setattr(cls, "validate_transformed_weights", mock.Mock())
    monkeypatch.setattr(workspace_pool, "_POOL", {})
    monkeypatch.setattr(workspace_pool, "_KEY_BY_ID", {})

    def allocate(self, fleet):
        return mock.MagicMock(
            name=f"workspace_{fleet.max_tokens_per_rank}", topk_idx=None
        )

    monkeypatch.setattr(cls, "_allocate_workspace", allocate)
    layers = []

    def make(knobs=None):
        layer = MoEEpMegaLayer(
            BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            FleetParams(num_experts=4, max_tokens_per_rank=64, token_hidden_size=128),
            MoEWeightPack(w13=torch.empty(0), w2=torch.empty(0)),
            MegaConfig(
                megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                    intermediate_size=128,
                    top_k=2,
                    knobs=knobs,
                    finalize_mode=FinalizeMode.DEFAULT,
                ),
                preprocess_weights=False,
                transformed_weights=(mock.sentinel.w1, mock.sentinel.w2),
            ),
        )
        layers.append(layer)
        return layer

    yield make
    for layer in reversed(layers):
        layer.destroy()
    assert workspace_pool.pooled_workspace_count() == 0


def test_explicit_knobs_are_copied_and_pool_by_configuration(make_layer):
    from flashinfer.moe_ep.core.kernel.workspace_pool import pooled_workspace_refcount

    owner = make_layer({"flag_batch": 8})
    other = make_layer()
    different = make_layer()
    knobs = {"flag_batch": 4, "mma_tiler_mnk": [256, 128, 256]}
    first = owner.create_workspace(128, knobs=knobs)
    shared = other.create_workspace(128, knobs=knobs)
    separate = different.create_workspace(128, knobs={"flag_batch": 8})

    raw = first._backend_workspace
    assert shared._backend_workspace is raw
    assert separate._backend_workspace is not raw
    assert pooled_workspace_refcount(raw) == 2
    knobs["mma_tiler_mnk"][0] = 1
    assert first._kernel._kernel_config.knobs["mma_tiler_mnk"] == [256, 128, 256]
    assert owner._megakernel_config.knobs == {"flag_batch": 8}
    first.destroy()
    raw.destroy.assert_not_called()
    shared.destroy()
    raw.destroy.assert_called_once()


def test_execution_uses_selected_backend_and_clears_destroyed_state(
    make_layer, monkeypatch
):
    layer = make_layer()
    default = layer.create_workspace(64, knobs={"flag_batch": 4})
    other = layer.create_workspace(64, live_tokens=8, knobs={"flag_batch": 8})
    assert other.live_tokens == 8
    methods = (
        "validate_forward",
        "validate_capture_ready",
        "stage_inputs",
        "compute",
        "compute_unfinalized",
        "set_compile_tokens_per_rank",
        "workspace_output",
        "coordinate_workspace_replacement",
    )
    for name in methods:
        monkeypatch.setattr(
            layer._kernel, name, mock.Mock(side_effect=AssertionError(name))
        )
        for handle in (default, other):
            monkeypatch.setattr(
                handle._kernel,
                name,
                mock.Mock(return_value=False)
                if name == "coordinate_workspace_replacement"
                else mock.Mock(),
            )

    tensors = _inputs(8)
    output = torch.empty(8, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for selected in (None, other, None):
        handle = default if selected is None else selected
        kernel, raw = handle._kernel, handle._backend_workspace
        kernel.compute.reset_mock()
        layer.forward(tensors, workspace=selected)
        kernel.validate_forward.assert_called_with(
            tensors, handle._fleet_params, quantize_input=True
        )
        kernel.validate_capture_ready.assert_called_with(raw, layer._transformed)
        kernel.stage_inputs.assert_called_with(tensors, raw, quantize_input=True)
        assert kernel.compute.call_args.args == (raw, layer._transformed)
        layer.stage_inputs(tensors, workspace=selected, compile_tokens_per_rank=16)
        kernel.set_compile_tokens_per_rank.assert_called_with(raw, 16)
        layer.compute_staged(output=output)
        kernel.compute.assert_called_with(raw, layer._transformed, output=output)
    assert layer.output_buffer is default._kernel.workspace_output.return_value
    default._kernel.workspace_output.assert_called_once_with(default._backend_workspace)

    monkeypatch.setattr(layer._kernel, "supports_unfinalized_output", True)
    for handle in (default, other):
        kernel, raw = handle._kernel, handle._backend_workspace
        assert (
            layer.forward_unfinalized(tensors, workspace=handle)
            is kernel.compute_unfinalized.return_value
        )
        kernel.stage_inputs.assert_called_with(tensors, raw, quantize_input=True)
        kernel.compute_unfinalized.assert_called_once_with(raw, layer._transformed)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    other.destroy()
    layer.compute_staged(output=output)
    raw = default._backend_workspace
    default._kernel._thunk_states[(id(raw),)] = mock.sentinel.thunk
    default.destroy()
    assert not default._kernel._thunk_states
    assert id(raw) not in layer._forward_signatures
    with pytest.raises(ValueError, match="requires a prior"):
        layer.compute_staged(output=output)


@pytest.mark.parametrize("live_tokens", [None, 8])
def test_override_does_not_replace_allocated_default(make_layer, live_tokens):
    layer = make_layer()
    layer._fleet_params = dataclasses.replace(
        layer._fleet_params, tuning_live_tokens=live_tokens
    )
    raw = layer._ensure_workspace()
    with pytest.raises(ValueError, match="already allocated default"):
        layer.create_workspace(64, live_tokens=live_tokens, knobs={"flag_batch": 4})
    other = layer.create_workspace(
        64, live_tokens=8 if live_tokens is None else None, knobs={"flag_batch": 4}
    )
    assert other._backend_workspace is not raw
    assert layer._ensure_workspace() is raw
    raw.destroy.assert_not_called()
    handle = layer.create_workspace(64, live_tokens=live_tokens)
    assert handle._backend_workspace is raw


@pytest.mark.parametrize(
    "world,count,ncu,expected",
    [
        (1, 128, "0", False),
        (2, 128, "0", True),
        (4, 4096, "0", True),
        (4, 4097, "0", False),
        (4, 128, "1", False),
    ],
)
def test_single_cta_reset_admission(world, count, ncu, expected, monkeypatch):
    """EP1, large prefixes and NCU retain the distributed reset protocol."""
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    cutlass = pytest.importorskip("cutlass")

    monkeypatch.setenv("MEGA_USE_NCU", ncu)
    comm = pkg.TokenInPullTokenBackPush(
        world_size=world,
        num_topk=4,
        num_experts_per_rank=4,
        num_total_experts=4 * world,
        hidden=128,
        fc1_token_dtype=cutlass.BFloat16,
        sf_uint32_per_token=0,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=64,
        cluster_shape_mn=(2, 1),
        dispatch_warp_start=8,
        num_other_warps=8,
    )
    assert comm.use_single_cta_tail_reset(count) is expected


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("tile_n,cluster_m", [(128, 4), (64, 4), (64, 2)])
def test_blackwell_genphase_workspace_plan(world, tile_n, cluster_m, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    cutlass = pytest.importorskip("cutlass")
    # DSL 4.6/4.7 expose the SMEM query through cutlass.utils only.
    monkeypatch.delattr(cutlass, "memory", raising=False)
    cfg = pkg.Sm100GenphaseMoeConfig(
        num_total_experts=8 * world,
        max_tokens_per_rank=129,
        num_topk=3,
        hidden=384,
        intermediate=192,
        rank=0,
        world_size=world,
        mma_tiler_mnk=(256, tile_n, 128),
        cluster_shape_mn=(cluster_m, 1),
    )
    monkeypatch.setattr(
        "flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.block_scaled._max_active_clusters",
        lambda size: 32,
    )
    kernel = pkg.Sm100GenphaseSymmBuffer._build_kernel(cfg)
    assert type(kernel).__module__.startswith(
        "flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.src.sources.kernel_src.blackwell."
    )
    assert kernel.architecture == "sm_100"
    assert kernel._mainloop.architecture == "sm_100"
    assert kernel._mainloop.instruction_k == 32
    assert kernel._mainloop.num_mma_instructions_per_ab_stage == 4
    plan = kernel._mainloop.tmem_plan
    assert plan.allocation_columns == 512
    assert plan.accumulator_stage_count == (3 if tile_n == 128 else 7)
    assert plan.accumulator_stage_stride_columns == tile_n
    mainloop = kernel._mainloop
    assert mainloop.gather_b_rows_per_cta == tile_n // 2
    assert mainloop.gather_b_runs_per_unit == 1
    assert mainloop.num_gather_b_warps == (2 if tile_n == 64 and cluster_m == 4 else 4)
    assert not mainloop.dynamic_gather
    assert kernel.masks_invalid_combine_slots


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
def test_blackwell_standard_backend_is_opt_in(kind, monkeypatch):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    name = "nvfp4_nvfp4" if kind == "nvfp4" else "mxfp8_mxfp8"
    config_type = (
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        if kind == "nvfp4"
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    )
    kwargs = {} if kind == "nvfp4" else {"kind": kind}
    original = config_type(intermediate_size=192, top_k=3, **kwargs)
    selected = dataclasses.replace(
        original, kernel_name=f"sm100_{name}_bf16_cutedsl_next"
    )
    backend = create_mega_kernel(selected)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=8, max_tokens_per_rank=129, token_hidden_size=384
    )
    cfg = backend._workspace_config(fleet)
    assert cfg.quant_kind == kind and cfg.kernel_variant == "inference"
    assert cfg.instruction_k == (64 if kind == "nvfp4" else 32)
    assert type(create_mega_kernel(original)) is not type(backend)
    assert hash(backend._workspace_pool_key(fleet))


@pytest.mark.parametrize("supplied_output", [False, True])
def test_standard_compute_returns_output(supplied_output, monkeypatch):
    from types import SimpleNamespace
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    backend, _, _ = _mxfp8_backend(kernel_name="sm100_mxfp8_mxfp8_bf16_cutedsl_next")
    output = torch.empty(2, 128) if supplied_output else None
    view = torch.full((2, 128), 3.0)

    def launch(y, *args, **kwargs):
        if y is None:
            return view
        y.copy_(view)
        return None

    monkeypatch.setattr(pkg, "sm100_block_scaled_mega_moe", launch)
    workspace = SimpleNamespace(config=SimpleNamespace(use_custom_finalize=False))
    result = backend.compute(workspace, (None, None), output=output)
    assert result is (output if supplied_output else view)
    torch.testing.assert_close(result, view)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("topk", [256, 257])
def test_blackwell_standard_routing_bounds(kind, topk, monkeypatch):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.backends.mega.kernel.sm100 import standard_cutedsl
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(standard_cutedsl, "validate_mega_arch", lambda: None)
    config_type = (
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        if kind == "nvfp4"
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    )
    name = "nvfp4_nvfp4" if kind == "nvfp4" else "mxfp8_mxfp8"
    backend = create_mega_kernel(
        config_type(
            intermediate_size=192,
            top_k=topk,
            kernel_name=f"sm100_{name}_bf16_cutedsl_next",
            **({} if kind == "nvfp4" else {"kind": kind}),
        )
    )
    bootstrap = moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    fleet = moe_ep.FleetParams(
        num_experts=512, max_tokens_per_rank=1, token_hidden_size=384
    )
    fields = dict(
        num_total_experts=512,
        max_tokens_per_rank=1,
        num_topk=topk,
        hidden=384,
        intermediate=192,
        rank=0,
        world_size=1,
        quant_kind=kind,
    )
    if topk == 257:
        with pytest.raises(ValueError, match="256"):
            backend.validate_init(bootstrap, fleet)
        with pytest.raises(ValueError, match="256"):
            pkg.Sm100BlockScaledMoeConfig(**fields)
    else:
        backend.validate_init(bootstrap, fleet)
        assert pkg.Sm100BlockScaledMoeConfig(**fields).num_topk == topk


@pytest.mark.parametrize("world,topk", [(1, 256), (1, 257), (257, 1)])
def test_blackwell_standard_router_metadata_bounds(world, topk):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.src.sources.api import (
        ImplDesc,
        ProblemDesc,
    )
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.src.sources.communication.nvlink_domain.token_comm import (
        _MetadataPushRouter,
    )

    problem = ProblemDesc(
        dict(
            world_size=world,
            expert_count=512 if world == 1 else world,
            topk=topk,
            max_tokens_per_rank=1,
            apply_topk_at_fc1=True,
        )
    )
    impl = ImplDesc(dict(token_padding_block=64, promised_launchable_sm_count=256))
    if topk > 256 or world > 256:
        with pytest.raises(ValueError, match="metadata"):
            _MetadataPushRouter(problem, impl)
    else:
        assert _MetadataPushRouter(problem, impl).topk == topk


@pytest.mark.parametrize("mode", ["grid_stride", "atomic_counter"])
@pytest.mark.parametrize("compute_clusters", [16, 32])
def test_blackwell_grid_stride_cluster_count(mode, compute_clusters):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.src.sources.api import (
        ImplDesc,
        ProblemDesc,
    )
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.src.sources.kernel_src.schedulers.fc12_scheduler import (
        BlackwellFusedFc12Scheduler,
    )

    problem = ProblemDesc(
        dict(expert_count=8, intermediate_gateup_size=384, hidden_size=384)
    )
    impl = ImplDesc(
        dict(
            num_scheduler_consumer_threads=224,
            mma_tiler_mnk=(256, 128, 128),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            hint=None,
            token_padding_block=64,
            sf_padding_block=128,
            work_id_mode=mode,
            is_swap_ab=True,
            launch_cluster_count=32,
            compute_cluster_count=compute_clusters,
        )
    )
    if mode == "grid_stride" and compute_clusters != 32:
        with pytest.raises(ValueError, match="grid_stride"):
            BlackwellFusedFc12Scheduler(problem, impl)
    else:
        scheduler = BlackwellFusedFc12Scheduler(problem, impl)
        assert scheduler.get_grid_shape(max_active_clusters=32) == (2, 1, 32)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3"])
def test_blackwell_standard_and_genphase_share_epilogue(kind, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    pytest.importorskip("cutlass")
    monkeypatch.setattr(
        "flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.block_scaled._max_active_clusters",
        lambda size: 32,
    )
    fields = dict(
        num_total_experts=8,
        max_tokens_per_rank=129,
        num_topk=3,
        hidden=384,
        intermediate=192,
        rank=0,
        world_size=1,
    )
    standard = pkg.Sm100BlockScaledSymmBuffer._build_kernel(
        pkg.Sm100BlockScaledMoeConfig(**fields, quant_kind=kind)
    )
    genphase = pkg.Sm100GenphaseSymmBuffer._build_kernel(
        pkg.Sm100GenphaseMoeConfig(**fields)
    )
    assert type(standard.epilogue) is type(genphase.epilogue)
    assert genphase.epilogue.fc2_tma_stages == 2
    assert standard.masks_invalid_combine_slots


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize(
    "mode,live_count,supported",
    [
        ("AUTO", False, True),
        ("AUTO", True, True),
        ("DEFAULT", False, True),
        ("DEFAULT", True, True),
        ("PERSISTENT", True, True),
        ("NONE", False, True),
    ],
)
def test_blackwell_standard_finalize_modes(kind, mode, live_count, supported):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.config import FinalizeMode
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    config_type = (
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        if kind == "nvfp4"
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    )
    config = config_type(
        intermediate_size=192,
        top_k=3,
        kernel_name=f"sm100_{kind}_{kind}_bf16_cutedsl_next",
        finalize_mode=FinalizeMode[mode],
        num_valid_tokens_tensor=torch.tensor([7], dtype=torch.int32)
        if live_count
        else None,
    )
    if supported:
        create_mega_kernel(config)
    else:
        with pytest.raises(MoEEpConfigError, match="finalize_mode"):
            create_mega_kernel(config)


@pytest.mark.parametrize("kwargs", [{"finalize_mode": "CAKE"}, {"do_finalize": False}])
def test_blackwell_standard_nvfp4_finalize_compat(kwargs):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.config import FinalizeMode
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    if "finalize_mode" in kwargs:
        kwargs = {"finalize_mode": FinalizeMode[kwargs["finalize_mode"]]}
    with (
        pytest.warns(DeprecationWarning, match="do_finalize")
        if "do_finalize" in kwargs
        else nullcontext()
    ):
        config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=192,
            top_k=3,
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
            **kwargs,
        )
    backend = create_mega_kernel(config)
    if config.finalize_mode is FinalizeMode.NONE:
        assert backend.supports_unfinalized_output
    else:
        backend.bind_ep_bootstrap(
            moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
        )
        with pytest.raises(MoEEpConfigError, match="CAKE"):
            backend._workspace_config(
                moe_ep.FleetParams(
                    num_experts=8,
                    max_tokens_per_rank=129,
                    token_hidden_size=384,
                )
            )


@pytest.mark.parametrize("early", [True, False])
def test_blackwell_standard_forwards_routing_weights(early, monkeypatch):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=192,
        top_k=3,
        kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
        apply_routing_weights_before_fc2=early,
    )
    backend = create_mega_kernel(config)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=8, max_tokens_per_rank=129, token_hidden_size=384
    )
    cfg = backend._workspace_config(fleet)
    assert cfg.apply_routing_weights_before_fc2 is early


def test_blackwell_genphase_import_is_lazy():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, types\n"
            "sentinel = types.ModuleType('sources')\n"
            "sentinel.__file__ = '/unrelated/sources/__init__.py'\n"
            "sys.modules['sources'] = sentinel\n"
            "from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg\n"
            "assert sys.modules['sources'] is sentinel\n"
            "assert 'flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe' not in sys.modules\n"
            "assert pkg.Sm100GenphaseMoeConfig.__module__.endswith('.sm100.next_cutedsl_megamoe.shim.genphase')\n"
            "assert sys.modules['sources'] is sentinel\n"
            "assert 'flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe' not in sys.modules\n"
            "from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as genphase\n"
            "assert pkg.Sm100GenphaseSymmBuffer is genphase.Sm100GenphaseSymmBuffer\n"
            "del sys.modules['sources']\n"
            "from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as rubin\n"
            "assert not issubclass(pkg.Sm100GenphaseSymmBuffer, rubin.Sm107BlockScaledSymmBuffer)\n"
            "assert not any(name.startswith('Sm100') for name in dir(rubin))\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"mma_tiler_mnk": (256, 64, 256)},
        {"mma_tiler_mnk": (256, 64, 128), "cluster_shape_mn": (1, 1)},
        {"cluster_shape_mn": (2, 1)},
        {"mma_tiler_mnk": (256, 256, 128)},
        {
            "schedule_policy": ("phase_interleave", None),
            "work_id_mode": "atomic_counter",
        },
        {"reduce_topk_in_kernel": True},
        {"max_tokens_per_rank": 1025},
        {"kernel_variant": "inference"},
        {"combine_dtype": "mxfp8"},
    ],
)
def test_blackwell_genphase_admission(override):
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    values = dict(
        num_total_experts=8,
        max_tokens_per_rank=129,
        num_topk=3,
        hidden=384,
        intermediate=192,
        rank=0,
        world_size=1,
    )
    with pytest.raises(ValueError):
        pkg.Sm100GenphaseMoeConfig(**(values | override))


def _genphase_tuning_config(**overrides):
    """Build a valid EP2 tuning workload with per-test field overrides."""
    values = dict(
        num_total_experts=256,
        max_tokens_per_rank=1024,
        num_topk=8,
        hidden=4096,
        intermediate=1024,
        rank=0,
        world_size=2,
    )
    return Sm100GenphaseMoeConfig(**(values | overrides))


@pytest.mark.parametrize(
    "tokens,n,cluster,hint",
    [
        (0, 128, 4, None),
        (1, 64, 2, 296),
        (4, 64, 4, 148),
        (17, 64, 4, 148),
        (65, 64, 4, 148),
        (1024, 128, 4, 148),
    ],
)
def test_blackwell_genphase_tuning_estimate_spans_latency_regimes(
    tokens, n, cluster, hint
):
    """Host estimates cover empty, sparse and dense workloads at fixed capacity."""
    cfg = _genphase_tuning_config()
    knobs = default_sm100_genphase_knobs(cfg, sm_count=148, live_tokens=tokens)
    assert knobs == dict(
        mma_tiler_mnk=(256, n, 128),
        cluster_shape_mn=(cluster, 1),
        schedule_policy=("grouped", hint),
    )
    assert is_valid_sm100_genphase_knobs(knobs, cfg)


@pytest.mark.parametrize("sms", [132, 148, 152])
@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize(
    "hidden,intermediate,experts,topk", [(4096, 1024, 256, 8), (7168, 3072, 384, 6)]
)
def test_blackwell_genphase_tuning_candidate_domain_and_original_fallback(
    sms, world, hidden, intermediate, experts, topk
):
    """All ranks enumerate twelve valid candidates including the original config."""
    cfg = _genphase_tuning_config(
        world_size=world,
        hidden=hidden,
        intermediate=intermediate,
        num_total_experts=experts,
        num_topk=topk,
    )
    candidates = sm100_genphase_candidates(cfg, sm_count=sms, live_tokens=16)
    assert len(candidates) == 12
    assert len({repr(x) for x in candidates}) == 12
    assert all(is_valid_sm100_genphase_knobs(x, cfg) for x in candidates)
    assert (
        dict(
            mma_tiler_mnk=(256, 128, 128),
            cluster_shape_mn=(4, 1),
            schedule_policy=("grouped", None),
        )
        in candidates
    )
    assert candidates[0] == default_sm100_genphase_knobs(
        cfg, sm_count=sms, live_tokens=16
    )
    assert candidates == sm100_genphase_candidates(
        dataclasses.replace(cfg, rank=world - 1), sm_count=sms, live_tokens=16
    )


def test_blackwell_genphase_tuning_constraints_and_resource_budget():
    """Candidate filtering respects explicit geometry, hints and limited SMs."""
    cfg = _genphase_tuning_config()
    rows = sm100_genphase_candidates(
        cfg,
        sm_count=148,
        constraints=dict(
            mma_tiler_mnk=(256, 64, 128), schedule_policy=("grouped", 333)
        ),
    )
    assert len(rows) == 2
    assert all(
        x["schedule_policy"] == ("grouped", 333)
        and x["mma_tiler_mnk"] == (256, 64, 128)
        for x in rows
    )
    limited = dataclasses.replace(
        cfg, max_sm_count=2, mma_tiler_mnk=(256, 64, 128), cluster_shape_mn=(2, 1)
    )
    rows = sm100_genphase_candidates(limited, sm_count=148, live_tokens=0)
    assert len(rows) == 4 and all(x["cluster_shape_mn"] == (2, 1) for x in rows)
    for constraints in [
        dict(mma_tiler_mnk=(256, 128, 128), cluster_shape_mn=(2, 1)),
        dict(schedule_policy=("grouped", 0)),
        dict(schedule_policy=("grouped", True)),
        dict(apply_routing_weights_before_fc2=False),
    ]:
        with pytest.raises(ValueError):
            sm100_genphase_candidates(cfg, sm_count=148, constraints=constraints)


@pytest.mark.parametrize("tokens", [-1, 1025, 1.5, True])
def test_blackwell_genphase_tuning_invalid_live_count(tokens):
    """Reject noninteger and out-of-capacity workload counts before tuning."""
    with pytest.raises(ValueError):
        sm100_genphase_candidates(
            _genphase_tuning_config(), sm_count=148, live_tokens=tokens
        )


@pytest.mark.parametrize("sms", [0, 1, 1.5, True])
def test_blackwell_genphase_tuning_invalid_hardware_count(sms):
    """Reject noninteger SM counts and devices too small for an admitted cluster."""
    with pytest.raises(ValueError):
        sm100_genphase_candidates(_genphase_tuning_config(), sm_count=sms)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("malformed", ["partial", "empty", "extra"])
def test_blackwell_genphase_autotune_rejects_incomplete_candidates_before_trials(
    monkeypatch, rank, malformed
):
    """Reject malformed shared candidates on every rank before trial collectives."""
    from flashinfer.moe_ep.core import tuning as core_tuning
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        autotune,
        correctness,
        knob_cache,
    )

    cfg = _genphase_tuning_config(rank=rank)
    valid = sm100_genphase_candidates(cfg, sm_count=148, live_tokens=16)[0]
    invalid = {
        "partial": {"schedule_policy": ("grouped", 296)},
        "empty": {},
        "extra": {**valid, "apply_routing_weights_before_fc2": False},
    }[malformed]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: mock.Mock(multi_processor_count=148),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _: rank)

    def gather(proposals, proposal, *, group):
        proposals[:] = [proposal, proposal]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(core_tuning, "tuning_live_tokens", lambda *args: 16)
    monkeypatch.setattr(knob_cache, "device_identity", lambda: "B200")
    reference = mock.Mock(side_effect=AssertionError("entered candidate trials"))
    record = mock.Mock()
    monkeypatch.setattr(correctness, "sampled_reference", reference)
    monkeypatch.setattr(knob_cache, "record_knobs", record)
    with pytest.raises(ValueError, match="candidates must set every Blackwell knob"):
        autotune._sweep_sm100_mega_moe(
            None,
            None,
            None,
            mock.Mock(config=cfg),
            num_tokens=16,
            candidates=[valid, invalid],
        )
    reference.assert_not_called()
    record.assert_not_called()


def test_blackwell_genphase_tuning_cache_exact_identity_constraints_and_invalid_record(
    tmp_path, monkeypatch
):
    """Cache reuse requires exact profiles and preserves unrelated or invalid data."""
    import json
    from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        knob_cache as cache,
    )

    path = tmp_path / "knobs.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    cfg = _genphase_tuning_config()
    winner = sm100_genphase_candidates(cfg, sm_count=148, live_tokens=16)[-1]
    path.write_text(
        json.dumps(
            dict(
                version=KNOB_CACHE_VERSION,
                entries=[dict(device="legacy", knobs={})],
            )
        )
    )
    assert cache.record_knobs(winner, cfg, live_tokens=16, p50_us=200, device="B300")
    assert cache.resolve_knobs(cfg, sm_count=148, live_tokens=16, device="B300") == (
        winner,
        "cache",
    )
    assert cache.live_token_buckets(cfg, device="B300") == [16]
    assert (
        cache.resolve_knobs(
            dataclasses.replace(cfg, rank=1),
            sm_count=148,
            live_tokens=16,
            device="B300",
        )[1]
        == "cache"
    )
    for changes in [
        dict(max_tokens_per_rank=512),
        dict(world_size=4),
        dict(gate_up_clamp=10),
        dict(apply_routing_weights_before_fc2=False),
    ]:
        assert (
            cache.resolve_knobs(
                dataclasses.replace(cfg, **changes),
                sm_count=148,
                live_tokens=16,
                device="B300",
            )[1]
            == "heuristic"
        )
    assert (
        cache.resolve_knobs(cfg, sm_count=148, live_tokens=17, device="B300")[1]
        == "heuristic"
    )
    assert (
        cache.resolve_knobs(cfg, sm_count=148, live_tokens=16, device="B200")[1]
        == "heuristic"
    )
    assert cache.resolve_knobs(
        cfg,
        sm_count=148,
        live_tokens=16,
        device="B300",
        constraints=dict(schedule_policy=("grouped", 333)),
    )[0]["schedule_policy"] == ("grouped", 333)
    data = json.loads(path.read_text())
    assert data["entries"][0]["device"] == "legacy"
    current_revision = data["entries"][-1]["backend_revision"]
    for revision in (
        "sm100-genphase-c8c67ae7-auto-v1",
        "sm100-genphase-6b9a6e2e-auto-v2",
        "sm100-genphase-3c1d1c2f-auto-v3",
        "sm100-genphase-shared-epilogue-v4",
        "sm100-genphase-33482a9d-auto-v4",
    ):
        data["entries"][-1]["backend_revision"] = revision
        path.write_text(json.dumps(data))
        assert (
            cache.resolve_knobs(cfg, sm_count=148, live_tokens=16, device="B300")[1]
            == "heuristic"
        )
        assert cache.live_token_buckets(cfg, device="B300") == []
    data["entries"][-1]["backend_revision"] = current_revision
    data["entries"][-1]["knobs"]["apply_routing_weights_before_fc2"] = False
    path.write_text(json.dumps(data))
    assert (
        cache.resolve_knobs(cfg, sm_count=148, live_tokens=16, device="B300")[1]
        == "heuristic"
    )
    assert cache.live_token_buckets(cfg, device="B300") == []
    path.write_text("invalid json")
    with pytest.warns(RuntimeWarning):
        assert not cache.record_knobs(
            winner, cfg, live_tokens=16, p50_us=200, device="B300"
        )
    assert path.read_text() == "invalid json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    assert not cache.record_knobs(
        winner, cfg, live_tokens=16, p50_us=200, device="B300"
    )


def test_blackwell_genphase_tuning_public_auto_retunes_once_and_capture_guard(
    monkeypatch,
):
    """Auto workspaces defer search to v2 and prepare once before capture."""
    from types import SimpleNamespace
    from flashinfer import autotune_v2
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MoEEpConfigError,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.core.tuning_state import AutoTuningState
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import autotune
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend = create_mega_kernel(
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            1024, 8, kernel_variant="genphase", knobs="auto"
        )
    )
    backend.bind_ep_bootstrap(
        BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    fleet = FleetParams(
        num_experts=256, max_tokens_per_rank=1024, token_hidden_size=4096
    )
    assert backend._workspace_pool_request(fleet) is None
    assert backend._workspace_pool_key(fleet) is None
    cfg = _genphase_tuning_config(world_size=1)
    workspace = SimpleNamespace(
        config=cfg,
        _compiled=None,
        _tuning_state=AutoTuningState(),
        staged_tokens=lambda: 16,
    )
    with pytest.raises(MoEEpConfigError, match="Warm up"):
        backend.validate_capture_ready(workspace, None)
    winner = sm100_genphase_candidates(cfg, sm_count=148)[-1]
    tune = mock.Mock(return_value=winner)
    launch = mock.Mock(return_value=mock.sentinel.output)
    monkeypatch.setattr(autotune, "autotune_sm100_block_scaled_mega_moe", tune)
    monkeypatch.setattr(pkg, "sm100_genphase_mega_moe", launch)
    with autotune_v2(persistent_cache=False):
        for _ in range(2):
            assert (
                backend.compute(workspace, (1, 2), output=None) is mock.sentinel.output
            )
    tune.assert_called_once()
    assert tune.call_args.kwargs["force_tune"] is False
    assert tune.call_args.kwargs["install_winner"] is True
    assert backend.get_tuned_knobs(workspace) == winner
    workspace._compiled = object()
    backend.validate_capture_ready(workspace, (1, 2))


def test_blackwell_genphase_tuning_offline_cli_dispatch_and_admission(monkeypatch):
    """The offline CLI dispatches GenPhase and rejects unsupported reduction modes."""
    from unittest import mock
    from flashinfer.moe_ep import tune

    module = mock.Mock()
    module.run_tuning.return_value = 0
    loader = mock.Mock(return_value=module)
    monkeypatch.setattr(tune.importlib, "import_module", loader)
    args = [
        "--arch",
        "sm100",
        "--dtype",
        "mxfp8_e4m3",
        "--kernel-variant",
        "genphase",
        "--hidden",
        "4096",
        "--intermediate",
        "1024",
        "--num-experts",
        "256",
        "--topk",
        "8",
        "--max-tokens",
        "1024",
        "--live-tokens",
        "65",
    ]
    assert tune.main(args) == 0
    assert loader.call_args.args[0] == ".backends.mega.kernel.sm100.next_tuning"
    assert module.run_tuning.call_args.args[0].kernel_variant == "genphase"
    loader.reset_mock()
    assert tune.main(args + ["--allow-nondeterministic"]) == 2
    loader.assert_not_called()


def test_blackwell_genphase_tuning_backend_selection_memo_separates_devices(
    monkeypatch,
):
    """Memoized knob resolution never crosses device identities."""
    from types import SimpleNamespace
    from unittest import mock
    import torch
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            major=10, minor=3, name="B300", multi_processor_count=148
        ),
    )
    selected = sm100_genphase_candidates(_genphase_tuning_config(), sm_count=148)[0]
    resolve = mock.Mock(return_value=(selected, "heuristic"))
    monkeypatch.setattr(pkg, "resolve_genphase_knobs", resolve)
    backend = create_mega_kernel(
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(1024, 8, kernel_variant="genphase")
    )
    backend.bind_ep_bootstrap(
        BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    fleet = FleetParams(
        num_experts=256, max_tokens_per_rank=1024, token_hidden_size=4096
    )
    first = backend._genphase_config(fleet)
    assert backend._genphase_config(fleet) is first
    resolve.assert_called_once()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    backend._genphase_config(fleet)
    assert resolve.call_count == 2


def test_blackwell_genphase_tuning_device_identity_separates_compiler_stacks_and_options(
    monkeypatch,
):
    """Compiler library versions and options isolate tuning cache identities."""
    from types import SimpleNamespace
    import torch
    from flashinfer.jit import cute_dsl_core
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import knob_cache

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="B300", major=10, minor=3, multi_processor_count=148
        ),
    )
    monkeypatch.setattr(
        cute_dsl_core, "_get_cute_dsl_version", lambda: "dsl==4.7.1;libs-cu13==4.7.1"
    )
    monkeypatch.delenv("CUTE_DSL_COMPILER_OPT", raising=False)
    original = knob_cache.device_identity()
    monkeypatch.setattr(
        cute_dsl_core, "_get_cute_dsl_version", lambda: "dsl==4.7.1;libs-cu13==4.7.2"
    )
    updated = knob_cache.device_identity()
    assert updated != original
    monkeypatch.setenv("CUTE_DSL_COMPILER_OPT", "iket")
    assert knob_cache.device_identity() not in (original, updated)


@pytest.mark.parametrize("minor", [0, 3])
@pytest.mark.parametrize("initial_arch", ["unset", "bare", "native"])
def test_blackwell_genphase_tuning_cache_hit_before_dsl_arch_configuration(
    tmp_path, monkeypatch, minor, initial_arch
):
    """Native target aliases reuse cached winners before DSL initialization."""
    from types import SimpleNamespace
    import json
    import torch
    from flashinfer.jit import cute_dsl_core
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        knob_cache as cache,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="Blackwell", major=10, minor=minor, multi_processor_count=148
        ),
    )
    monkeypatch.setattr(
        cute_dsl_core, "_get_cute_dsl_version", lambda: "test-dsl-stack"
    )
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    monkeypatch.setenv("CUTE_DSL_COMPILER_OPT", "preserve-options")
    native = f"sm_10{minor}a"
    cfg = _genphase_tuning_config()
    winner = sm100_genphase_candidates(cfg, sm_count=148, live_tokens=16)[-1]
    # A previous workspace configured the native compiler target before tuning.
    monkeypatch.setenv("CUTE_DSL_ARCH", native)
    assert cache.record_knobs(winner, cfg, live_tokens=16, p50_us=200)
    # A fresh caller resolves the cache before creating its first workspace.
    if initial_arch == "unset":
        monkeypatch.delenv("CUTE_DSL_ARCH")
    elif initial_arch == "bare":
        monkeypatch.setenv("CUTE_DSL_ARCH", native.removesuffix("a"))
    assert cache.resolve_knobs(cfg, sm_count=148, live_tokens=16) == (winner, "cache")
    identity = json.loads(cache.device_identity())
    assert identity["compile_arch"] == native
    assert identity["compiler_options"] == "preserve-options"
    monkeypatch.setenv("CUTE_DSL_COMPILER_OPT", "different-options")
    assert cache.resolve_knobs(cfg, sm_count=148, live_tokens=16)[1] == "heuristic"


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize(
    "activation",
    [
        {"swiglu_alpha": 1.702, "swiglu_beta": 1.0},
        {"activation": "situ", "situ_beta": 7.0},
        {"activation": "situ", "situ_beta": 7.0, "situ_linear_beta": 5.0},
    ],
)
def test_standard_activation_forwarding(kind, activation, monkeypatch):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setattr(
        "flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.block_scaled._max_active_clusters",
        lambda size: 32,
    )
    cls = (
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        if kind == "nvfp4"
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    )
    backend = create_mega_kernel(
        cls(
            intermediate_size=192,
            top_k=3,
            kernel_name=f"sm100_{kind}_{kind}_bf16_cutedsl_next",
            **activation,
        )
    )
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    cfg = backend._workspace_config(
        moe_ep.FleetParams(
            num_experts=8,
            max_tokens_per_rank=129,
            token_hidden_size=384,
        )
    )
    kernel = pkg.Sm100BlockScaledSymmBuffer._build_kernel(cfg)
    for field, value in activation.items():
        assert getattr(cfg, field) == value
        if field != "activation":
            assert getattr(kernel.epilogue, field) == value


def test_standard_persistent_pool_tracks_storage(monkeypatch):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.config import FinalizeMode
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    count = torch.tensor([7], dtype=torch.int32)
    config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=192,
        top_k=3,
        kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
        finalize_mode=FinalizeMode.PERSISTENT,
        num_valid_tokens_tensor=count,
    )
    backend = create_mega_kernel(config)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=8, max_tokens_per_rank=129, token_hidden_size=384
    )
    first = backend._workspace_pool_key(fleet)
    count.fill_(1)
    assert first == backend._workspace_pool_key(fleet)
    config.num_valid_tokens_tensor = count.clone()
    assert first != backend._workspace_pool_key(fleet)
    assert backend._workspace_config(fleet).use_persistent_finalize_kernel


def test_standard_auto_cake_matches_legacy(monkeypatch):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.config import FinalizeMode
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    monkeypatch.setattr(
        "flashinfer.jit.cake_megamoe_topk_reduce.supports_device", lambda: True
    )
    config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=6,
        kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
    )
    backend = create_mega_kernel(config)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=8, max_tokens_per_rank=256, token_hidden_size=4096
    )
    assert backend._finalize_mode(fleet) is FinalizeMode.CAKE
    assert backend._workspace_config(fleet).use_custom_finalize


def test_standard_cake_requires_reducer_warmup(monkeypatch):
    from types import SimpleNamespace
    from flashinfer.jit import cake_megamoe_topk_reduce as cake
    from flashinfer.moe_ep import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    backend = create_mega_kernel(
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=192,
            top_k=3,
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
        )
    )
    workspace = SimpleNamespace(
        _compiled=object(),
        config=SimpleNamespace(use_custom_finalize=True),
    )
    monkeypatch.setattr(
        cake, "is_cake_megamoe_topk_reduce_module_loaded", lambda: False
    )
    with pytest.raises(RuntimeError, match="terminal reducer"):
        backend.validate_capture_ready(workspace, None)
    monkeypatch.setattr(cake, "is_cake_megamoe_topk_reduce_module_loaded", lambda: True)
    backend.validate_capture_ready(workspace, None)


@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp8_e5m2"])
def test_standard_hidden_alignment_preserves_genphase_constraints(kind):
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe import (
        Sm100BlockScaledMoeConfig,
        Sm100GenphaseMoeConfig,
    )

    fields = dict(
        num_total_experts=4,
        max_tokens_per_rank=65,
        num_topk=2,
        hidden=192,
        intermediate=128,
        rank=0,
        world_size=1,
        quant_kind=kind,
    )
    assert Sm100BlockScaledMoeConfig(**fields).hidden_alignment == 64
    with pytest.raises(ValueError, match="multiple of 128"):
        Sm100GenphaseMoeConfig(**fields)
