"""Fixed knob selection for shared-weight MegaMoE workspace profiles."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from .test_mega_workspace_api import _inputs


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


def test_blackwell_mxfp8_legacy_config_and_dispatch(monkeypatch, dist_not_initialized):
    """Old positional arguments, allocator and cache key retain their contract."""
    import dataclasses

    from flashinfer.moe_ep import Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    assert [
        f.name for f in dataclasses.fields(Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig)
    ][:9] == [
        "intermediate_size",
        "top_k",
        "kernel_name",
        "kind",
        "gate_up_clamp",
        "activation_clamp",
        "fast_math",
        "enable_in_kernel_fc2_reduce",
        "knobs",
    ]
    kernel, _, fleet = _mxfp8_backend()
    assert kernel._kernel_config.kernel_variant == "inference"
    allocator = mock.Mock(return_value=mock.sentinel.workspace)
    monkeypatch.setattr(pkg, "get_symm_buffer_for_mxfp8_mega_moe", allocator)
    monkeypatch.setattr(kernel, "_workspace_knobs", lambda fp: {"flag_batch": 4})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert kernel._allocate_workspace(fleet) is mock.sentinel.workspace
    allocator.assert_called_once_with(
        8,
        129,
        3,
        384,
        192,
        0,
        1,
        kind="mxfp8_e4m3",
        gate_up_clamp=None,
        activation_clamp=None,
        enable_in_kernel_fc2_reduce=False,
        knobs={"flag_batch": 4},
    )
    assert kernel._tuning_key(fleet) == dict(
        dtype="mxfp8_e4m3",
        world_size=1,
        hidden=384,
        intermediate=192,
        num_experts=8,
        topk=3,
        max_tokens=129,
        enable_in_kernel_fc2_reduce=False,
    )
    assert kernel._workspace_pool_key(fleet) == (
        kernel.kernel_name(),
        0,
        0,
        1,
        id(None),
        8,
        129,
        3,
        384,
        192,
        "mxfp8_e4m3",
        None,
        False,
        (("flag_batch", 4),),
    )


@pytest.mark.parametrize(
    "tile,cluster", [(None, None), ((256, 64, 128), (2, 1)), ((256, 64, 128), (4, 1))]
)
def test_blackwell_genphase_backend_selects_own_workspace(
    tile, cluster, monkeypatch, dist_not_initialized
):
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
        {"kernel_variant": "genphase", "kind": "mxfp8_e5m2"},
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
    monkeypatch.setattr(cls, "_uses_native_topk_reduce", lambda *args: False)
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
                    intermediate_size=128, top_k=2, knobs=knobs
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
    )
    for name in methods:
        monkeypatch.setattr(
            layer._kernel, name, mock.Mock(side_effect=AssertionError(name))
        )
        for handle in (default, other):
            monkeypatch.setattr(handle._kernel, name, mock.Mock())

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


def test_override_does_not_replace_allocated_default(make_layer):
    layer = make_layer()
    raw = layer._ensure_workspace()
    with pytest.raises(ValueError, match="already allocated default"):
        layer.create_workspace(64, knobs={"flag_batch": 4})
    assert layer._ensure_workspace() is raw
    raw.destroy.assert_not_called()
    handle = layer.create_workspace(64)
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
        "blackwell_genphase_sources.kernel_src.blackwell."
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
        {"quant_kind": "nvfp4"},
        {"quant_kind": "mxfp4_mxfp8", "intermediate": 256},
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
