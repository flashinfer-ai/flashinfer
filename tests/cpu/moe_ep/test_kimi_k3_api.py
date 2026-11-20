"""Host contracts for the isolated Kimi K3 provider."""

from dataclasses import FrozenInstanceError, replace
from unittest import mock

import pytest
import torch

from flashinfer import moe_ep
from flashinfer.moe_ep.backends.mega.kernel.sm100.kimi_k3 import backend
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

pytestmark = pytest.mark.cpu_only


def _config(**kwargs):
    return moe_ep.Sm100_KimiK3_MegaMoeConfig(
        **{"situ_beta": 4.0, "situ_linear_beta": 25.0, **kwargs}
    )


def _fleet(**kwargs):
    return moe_ep.FleetParams(
        **{
            "num_experts": 896,
            "token_hidden_size": 3584,
            "max_tokens_per_rank": 256,
            **kwargs,
        }
    )


@pytest.mark.parametrize(
    "options",
    [
        {"intermediate_size": 1536},
        {"top_k": 8},
        {"quant_kind": "mxfp4"},
        {"combine_dtype": "nvfp4"},
        {"activation": "swiglu"},
        {"situ_beta": None},
        {"situ_beta": True},
        {"situ_beta": 0.0},
        {"situ_beta": float("nan")},
        {"situ_linear_beta": None},
        {"situ_linear_beta": -1.0},
        {"situ_linear_beta": float("inf")},
        {"input_norm_const": 0.0},
        {"fast_situ": 1},
        {"apply_routing_weights_before_fc2": True},
        {"finalize_mode": moe_ep.FinalizeMode.AUTO},
        {"finalize_mode": moe_ep.FinalizeMode.NONE},
        {"kernel_variant": "training"},
        {"enable_in_kernel_fc2_reduce": True},
        {"enable_fc1_activation_per_token_scale": True},
        {"num_valid_tokens_tensor": object()},
        {"knobs": "cache"},
        {"knobs": {"unknown": True}},
        {"quant_kind": "mxfp4_mxfp8", "input_norm_const": 2.0},
        {"quant_kind": "mxfp4_mxfp8", "fc1_alpha": torch.ones(1)},
    ],
)
def test_config_rejects_unsupported_modes(options):
    with pytest.raises(ValueError):
        _config(**options)


def test_config_and_registry_are_fixed_and_copy_knobs():
    knobs = {"mma_tiler_mnk": [128, 256, 256]}
    config = _config(knobs=knobs)
    kernel = create_mega_kernel(config)
    assert isinstance(kernel, backend.KimiK3MegaKernelBackend)
    assert config.kernel_name == "sm100_kimi_k3_megamoe"
    assert config.finalize_mode is moe_ep.FinalizeMode.DEFAULT
    assert not config.fast_situ
    with pytest.raises(FrozenInstanceError):
        config.fast_situ = True
    with pytest.raises(TypeError):
        _config(kind="nvfp4")
    knobs["mma_tiler_mnk"][0] = 256
    assert kernel._config.knobs["mma_tiler_mnk"][0] == 128


@pytest.mark.parametrize("quant_kind", ["nvfp4", "mxfp4_mxfp8"])
def test_k3_and_k31_pro_checkpoint_fields_use_routed_geometry(quant_kind, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    text_config = {
        "hidden_size": 7168,
        "routed_expert_hidden_size": 3584,
        "num_experts": 896,
        "moe_intermediate_size": 3072,
        "num_experts_per_token": 16,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
    }
    config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
        intermediate_size=text_config["moe_intermediate_size"],
        top_k=text_config["num_experts_per_token"],
        activation=text_config["hidden_act"],
        situ_beta=text_config["activation_situ_beta"],
        situ_linear_beta=text_config["activation_situ_linear_beta"],
        quant_kind=quant_kind,
    )
    fleet = moe_ep.FleetParams(
        num_experts=text_config["num_experts"],
        token_hidden_size=text_config["routed_expert_hidden_size"],
        max_tokens_per_rank=256,
    )
    bootstrap = moe_ep.BootstrapConfig(world_size=8, rank=0, auto_bootstrap=False)
    kernel = create_mega_kernel(config)
    kernel.validate_init(bootstrap, fleet)
    assert (kernel._config.situ_beta, kernel._config.situ_linear_beta) == (4.0, 25.0)
    with pytest.raises(MoEEpConfigError, match="token_hidden_size=3584"):
        kernel.validate_init(
            bootstrap, replace(fleet, token_hidden_size=text_config["hidden_size"])
        )


@pytest.mark.parametrize(
    "world,options,message",
    [
        (2, {}, "world_size"),
        (1, {"num_experts": 128}, "num_experts"),
        (1, {"token_hidden_size": 4096}, "token_hidden_size"),
        (1, {"max_tokens_per_rank": 0}, "max_tokens_per_rank"),
        (1, {"max_tokens_per_rank": True}, "max_tokens_per_rank"),
        (1, {"max_tokens_per_rank": 65536}, "int32"),
        (1, {"tuning_live_tokens": 8}, "tuning_live_tokens"),
    ],
)
def test_init_rejects_unsupported_geometry_before_allocation(world, options, message):
    kernel = create_mega_kernel(_config())
    with (
        mock.patch.object(backend, "_shim") as shim,
        pytest.raises(ValueError, match=message),
    ):
        kernel.validate_init(
            moe_ep.BootstrapConfig(world_size=world, rank=0, auto_bootstrap=False),
            _fleet(**options),
        )
    shim.assert_not_called()


@pytest.mark.parametrize("world", [1, 8, 16])
def test_native_delegation_and_runtime_requirements(world):
    bootstrap = moe_ep.BootstrapConfig(world_size=world, rank=0, auto_bootstrap=False)
    kernel = create_mega_kernel(_config())
    group = object()
    kernel.bind_ep_bootstrap(bootstrap)
    kernel._ep_comm_group = group
    fleet = _fleet(max_tokens_per_rank=33)
    weights, transformed = object(), object()
    with mock.patch.object(backend, "_shim") as shim:
        shim.return_value.preprocess_weights.return_value = transformed
        assert kernel.preprocess_weights(weights, fleet) is transformed
        shim.return_value.preprocess_weights.assert_called_once_with(
            weights, kernel._config, fleet, world_size=world
        )
        workspace = kernel.prepare_workspace(bootstrap, fleet)
        shim.return_value.create_workspace.assert_called_once_with(
            kernel._config,
            fleet,
            rank=0,
            world_size=world,
            process_group=group if world > 1 else None,
        )
        t, output = object(), object()
        workspace.fleet_params = fleet
        with mock.patch.object(kernel, "validate_forward") as validate:
            kernel.stage_inputs(t, workspace, quantize_input=True)
        validate.assert_called_once_with(t, fleet, quantize_input=True)
        workspace.stage_inputs.assert_called_once_with(t, quantize_input=True)
        kernel.validate_capture_ready(workspace, transformed)
        workspace.validate_capture_ready.assert_called_once_with(transformed)
        kernel.compute(workspace, transformed, output=output)
        workspace.compute.assert_called_once_with(transformed, output=output)
    assert kernel.runtime_requirements(bootstrap) == (
        frozenset({"torch_dist"}) if world > 1 else frozenset()
    )


def test_public_workspace_handles_keep_separate_lifetimes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    raw = [mock.MagicMock(name="small"), mock.MagicMock(name="large")]
    with mock.patch.object(backend, "_shim") as shim:
        shim.return_value.create_workspace.side_effect = raw
        layer = moe_ep.MoEEpMegaLayer(
            bootstrap=moe_ep.BootstrapConfig(
                world_size=1, rank=0, auto_bootstrap=False
            ),
            fleet_params=_fleet(),
            weights=moe_ep.MoEWeightPack(torch.empty(0), torch.empty(0)),
            backend=moe_ep.MegaConfig(
                megakernel=_config(),
                transformed_weights=object(),
                preprocess_weights=False,
            ),
        )
        try:
            small = layer.create_workspace(256)
            large = layer.create_workspace(512)
            assert layer._resolve_workspace(small)[1] is raw[0]
            assert layer._resolve_workspace(large)[1] is raw[1]
            assert layer.output_buffer is raw[0].output
            with pytest.raises(MoEEpConfigError, match="already exists"):
                layer.create_workspace(256)
            small.destroy()
            small.destroy()
            raw[0].destroy.assert_called_once()
            raw[1].destroy.assert_not_called()
            assert not large.is_destroyed
            with pytest.raises(MoEEpConfigError, match="destroyed"):
                layer._resolve_workspace(small)
            assert layer._resolve_workspace(large)[1] is raw[1]
        finally:
            layer.destroy()
        assert large.is_destroyed
        raw[1].destroy.assert_called_once()
        with pytest.raises(MoEEpConfigError, match="destroyed"):
            layer.create_workspace(1024)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("hidden_states", torch.empty(1, 3584), "hidden_states"),
        ("topk_ids", torch.empty(1, 8, dtype=torch.int32), "topk_ids"),
        ("topk_ids", torch.empty(1, 16), "topk_ids"),
        ("topk_weights", torch.empty(1, 16, dtype=torch.float64), "topk_weights"),
        ("scales", torch.empty(1, 224, dtype=torch.uint8), "scales"),
        ("fc1_alpha", torch.ones(1), "fc1_alpha"),
        ("fc1_activation_per_token_scale", torch.ones(1), "per_token_scale"),
    ],
)
def test_forward_rejects_bad_tensor_contracts(field, value, message):
    kernel = create_mega_kernel(_config())
    kernel.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    tensors = moe_ep.MoEEpTensors(
        hidden_states=torch.empty(1, 3584, dtype=torch.bfloat16),
        topk_ids=torch.zeros(1, 16, dtype=torch.int32),
        topk_weights=torch.ones(1, 16),
    )
    with pytest.raises(MoEEpConfigError, match=message):
        kernel.validate_forward(
            replace(tensors, **{field: value}), _fleet(), quantize_input=True
        )


def test_fp4_ties_saturation_and_signed_zero():
    from flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.shim.weights import (
        _pack_fp4,
        _quantize_expert,
    )

    values = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            1.75,
            -1.75,
            2.5,
            -2.5,
            3.5,
            -3.5,
            5.0,
            -5.0,
            8.0,
            -8.0,
        ]
    )
    assert _pack_fp4(values).tolist() == [
        0x80,
        0x80,
        0xA2,
        0xA2,
        0xC4,
        0xC4,
        0xE6,
        0xE6,
        0xF7,
    ]
    for magnitude in (0.0, 1e-20):
        weight = torch.tensor([-magnitude, magnitude] * 8).reshape(1, 16)
        packed, scales = _quantize_expert(weight, "nvfp4", 1.0)
        assert packed.tolist() == [[0x08] * 8]
        assert scales.tolist() == [[0]]


def test_weight_layouts_match_scalar_index_maps():
    from flashinfer.moe_ep.kernel_src.sm100 import _kimi_k3_megamoe as native

    weights = torch.arange(2 * 64 * 3).reshape(2, 64, 3)
    expected = torch.empty_like(weights)
    for row in range(64):
        target = (row % 32 // 16) * 32 + (row // 32) * 16 + row % 16
        expected[:, target] = weights[:, row]
    torch.testing.assert_close(native.interleave_gate_up(weights), expected)

    scales = (torch.arange(2 * 137 * 7) % 251).byte().reshape(2, 137, 7)
    expected = torch.zeros(2, 256 * 8, dtype=torch.uint8)
    for row in range(137):
        for col in range(7):
            target = (
                (((row // 128) * 2 + col // 4) * 32 + row % 32) * 4 + (row % 128) // 32
            ) * 4 + col % 4
            expected[:, target] = scales[:, row, col]
    torch.testing.assert_close(native.swizzle_scales(scales), expected)
