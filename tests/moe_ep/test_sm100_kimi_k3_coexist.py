"""Alternate Kimi and standard MegaMoE without shared launch or buffer state."""

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from flashinfer import moe_ep
from tests.moe_ep.test_sm100_kimi_k3 import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    TOPK,
    _alpha,
    _check,
    _inputs,
    _reference,
    _runtime,
    _weights,
)

pytestmark = [pytest.mark.arch_blackwell, pytest.mark.node_exclusive]


def test_kimi_and_standard_alternate_eager_and_graph(monkeypatch):
    rank, world = _runtime()
    if world == 1:
        monkeypatch.setenv("MEGA_NO_DIST", "1")
    expert = torch.arange(EXPERTS // world, device="cuda") + rank * (EXPERTS // world)
    alpha1, alpha2, norm = _alpha(expert)
    common = dict(
        intermediate_size=INTERMEDIATE,
        top_k=TOPK,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        combine_dtype="bf16",
        apply_routing_weights_before_fc2=False,
        finalize_mode=moe_ep.FinalizeMode.DEFAULT,
        fc1_alpha=alpha1.float(),
        fc2_alpha=alpha2.float(),
        fc1_norm_const=norm.float(),
    )
    configs = (
        moe_ep.Sm100_KimiK3_MegaMoeConfig(fast_situ=False, **common),
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
            fast_math=False,
            **common,
        ),
    )
    weights = _weights(rank, world, "nvfp4")
    layers, workspaces, graphs, inputs, outputs = [], [], [], [], []
    graph = None
    try:
        for index, config in enumerate(configs):
            layer = moe_ep.MoEEpMegaLayer(
                bootstrap=moe_ep.BootstrapConfig(
                    world_size=world,
                    rank=rank,
                    process_group=dist.group.WORLD if world > 1 else None,
                    device=torch.cuda.current_device(),
                ),
                fleet_params=moe_ep.FleetParams(
                    num_experts=EXPERTS,
                    max_tokens_per_rank=256,
                    token_hidden_size=HIDDEN,
                ),
                weights=weights,
                backend=moe_ep.MegaConfig(megakernel=config, quantize_input=True),
            )
            layers.append(layer)
            workspace = layer.create_workspace(256)
            workspaces.append(workspace)
            tensors = _inputs(17, rank, world, index, mask_routes=False)
            tensors = replace(tensors, output=torch.empty_like(tensors.hidden_states))
            inputs.append(tensors)
            layer.warmup(tensors, workspace=workspace)
            output = layer(tensors, workspace=workspace)
            assert output.data_ptr() == tensors.output.data_ptr()
            _check(output, _reference(tensors, "nvfp4", "bf16"))
            outputs.append(output)

        for name in ("hidden_states", "topk_ids", "topk_weights", "output"):
            assert (
                getattr(inputs[0], name).data_ptr()
                != getattr(inputs[1], name).data_ptr()
            )

        for index in (0, 1, 0, 1):
            other = outputs[1 - index].clone()
            output = layers[index](inputs[index], workspace=workspaces[index])
            _check(output, _reference(inputs[index], "nvfp4", "bf16"))
            torch.testing.assert_close(outputs[1 - index], other, rtol=0, atol=0)

        for index in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layers[index](inputs[index], workspace=workspaces[index])
            assert captured.data_ptr() == outputs[index].data_ptr()
            graphs.append(graph)
        graph = None

        for step, order in enumerate(((0, 1), (1, 0), (0, 1)), start=2):
            for index in order:
                changed = _inputs(17, rank, world, 2 * step + index, mask_routes=False)
                for name in ("hidden_states", "topk_ids", "topk_weights"):
                    getattr(inputs[index], name).copy_(getattr(changed, name))
                other = outputs[1 - index].clone()
                graphs[index].replay()
                torch.cuda.synchronize()
                _check(outputs[index], _reference(changed, "nvfp4", "bf16"))
                torch.testing.assert_close(outputs[1 - index], other, rtol=0, atol=0)
            for index in reversed(order):
                captured = outputs[index].clone()
                eager = layers[index](inputs[index], workspace=workspaces[index])
                torch.testing.assert_close(eager, captured, rtol=0, atol=0)
    finally:
        graph = None
        graphs.clear()
        torch.cuda.synchronize()
        for layer in reversed(layers):
            layer.destroy()
