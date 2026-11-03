"""Legacy NVFP4 GIN qualification through the formal FI layer.

Launch on two independent NVLink domains, two ranks per domain. A single
NVLink domain is intentionally rejected by the runtime's rail/LSA preflight.
This file executes correctness checks only and reports no timings.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.config import (
    Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
)

pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_blackwell, pytest.mark.node_exclusive]


def _bootstrap():
    topology = tuple(
        int(os.environ.get(name, "1"))
        for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_WORLD_SIZE")
    )
    if topology != (4, 2, 2):
        pytest.skip("requires N2R2 torchrun: two worker groups, two local ranks each")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    return BootstrapConfig(
        world_size=4,
        rank=dist.get_rank(),
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
    )


def _layer(bootstrap, *, retain_weights=False, quantize_input=True, **changes):
    generator = torch.Generator(device="cuda").manual_seed(101 + bootstrap.rank)
    config = dict(intermediate_size=1536, top_k=4, internode=True, ranks_per_node=2)
    config.update(changes)
    # Small finite weights keep the independent Torch reference well scaled.
    weights = MoEWeightPack(
        w13=torch.randn(
            (12, 3072, 7168), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        * 0.01,
        w2=torch.randn(
            (12, 7168, 1536), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        * 0.01,
    )
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=48, max_tokens_per_rank=32, token_hidden_size=7168
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(**config),
            quantize_input=quantize_input,
            preprocess_weights=True,
        ),
    )
    if retain_weights:
        layer._gin_oracle_weights = weights
    return layer


def _inputs(rank, iteration, case):
    generator = torch.Generator(device="cuda").manual_seed(301 + rank + iteration * 4)
    n = 32 if case == "balanced" else (32, 7, 32, 0 if rank == 1 else 13)[iteration % 4]
    x = torch.randn((n, 7168), dtype=torch.bfloat16, device="cuda", generator=generator)
    if case == "balanced":
        scores = torch.rand((n, 48), device="cuda", generator=generator)
        weights, ids = scores.topk(4, dim=-1)
        return MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
    node = rank // 2
    routes = (
        [rank * 12 + i for i in range(4)],
        [(1 - node) * 24 + i for i in range(4)],
        [0, 1, 24, 25],
    )[iteration % 3]
    ids = torch.tensor(routes, dtype=torch.int64, device="cuda").repeat(n, 1)
    scores = torch.tensor(
        [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device="cuda"
    ).repeat(n, 1)
    if iteration == 8:
        ids[::3, 1] = -1
        scores[::3, 1] = float("nan")
    return MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=scores)


@pytest.mark.parametrize("case", ["balanced", "matrix"])
def test_nvfp4_gin_public_layer_relaunch(case):
    bootstrap = _bootstrap()
    layer = _layer(bootstrap)
    failed = False
    try:
        for iteration in range(12):
            inputs = _inputs(bootstrap.rank, iteration, case)
            output = layer.forward(inputs)
            torch.cuda.synchronize()
            assert output.shape == inputs.hidden_states.shape
            assert torch.isfinite(output).all()
            dist.barrier()
            print(
                f"[rank {bootstrap.rank}] GIN_PUBLIC_RELAUNCH {iteration} finite PASS",
                flush=True,
            )
    except BaseException:
        failed = True
        raise
    finally:
        # Preserve the first failure rather than calling CUDA again after a
        # fatal launch error; the launcher exits the whole failed job.
        if not failed:
            layer.destroy()


@pytest.mark.parametrize(
    "changes",
    [
        {"knobs": "auto"},
        {"enable_fc1_activation_per_token_scale": True},
        {"swiglu_alpha": 1.702, "swiglu_beta": 1.0},
        {"ranks_per_node": 4},
        {"fc1_alpha": 2.0},
    ],
)
def test_nvfp4_gin_public_config_rejection(changes):
    bootstrap = _bootstrap()
    with pytest.raises((RuntimeError, ValueError), match="GIN|gin|internode"):
        _layer(bootstrap, **changes)


def test_nvfp4_gin_public_rank_local_errors():
    bootstrap = _bootstrap()
    layer = _layer(bootstrap)
    failed = False
    try:
        layer.forward(_inputs(bootstrap.rank, 0, "balanced"))
        torch.cuda.synchronize()
        for kind in ("route", "dtype", "capacity", "output", "epilogue_scale"):
            inputs = _inputs(bootstrap.rank, 1, "balanced")
            if bootstrap.rank == 1:
                if kind == "route":
                    inputs.topk_ids[0, 0] = 48
                elif kind == "dtype":
                    inputs.hidden_states = inputs.hidden_states.float()
                elif kind == "capacity":
                    inputs.hidden_states = inputs.hidden_states.repeat(2, 1)
                    inputs.topk_ids = inputs.topk_ids.repeat(2, 1)
                    inputs.topk_weights = inputs.topk_weights.repeat(2, 1)
                elif kind == "epilogue_scale":
                    inputs.fc1_alpha = torch.ones(
                        12, device="cuda", dtype=torch.float32
                    )
                else:
                    inputs.output = torch.empty(
                        (32, 1), device="cuda", dtype=torch.bfloat16
                    )
            with pytest.raises((RuntimeError, ValueError), match="GIN|gin"):
                layer.forward(inputs)
            with pytest.raises((RuntimeError, ValueError), match="GIN|gin"):
                layer.stage_inputs(inputs)
            dist.barrier()
        result = layer.forward(_inputs(bootstrap.rank, 2, "balanced"))
        torch.cuda.synchronize()
        assert torch.isfinite(result).all()
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


def test_nvfp4_gin_public_rank_local_errors_control_headers(monkeypatch):
    bootstrap = _bootstrap()
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.gin import (
        GinPreflightError,
        collective_gin_preflight,
    )

    object_reports = []
    original = dist.all_gather_object

    def detailed_report(*args, **kwargs):
        object_reports.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(dist, "all_gather_object", detailed_report)
    collective_gin_preflight(dist, phase="ready", signature=(32,), local_error=None)
    assert not object_reports
    for kind in ("phase", "configuration", "error", "subgroup"):
        different = bootstrap.rank == 1
        before = len(object_reports)
        with pytest.raises(GinPreflightError, match="GIN"):
            collective_gin_preflight(
                dist,
                phase="different-boundary"
                if different and kind == "phase"
                else "ready",
                signature=(64 if different and kind == "configuration" else 32,),
                local_error="rank-local rejection"
                if different and kind == "error"
                else None,
                requested_group=object() if different and kind == "subgroup" else None,
            )
        assert len(object_reports) == before + 1
        # Rejection must leave the common control sequence usable by all ranks.
        collective_gin_preflight(
            dist, phase="recovered", signature=(32,), local_error=None
        )
        assert len(object_reports) == before + 1


def test_nvfp4_gin_public_sessions_and_recreation():
    bootstrap = _bootstrap()
    layers = [_layer(bootstrap), _layer(bootstrap)]
    failed = False
    try:
        inputs = _inputs(bootstrap.rank, 0, "balanced")
        expected = []
        for layer in layers:
            expected.append(layer.forward(inputs).clone())
            torch.cuda.synchronize()
        identities = [
            layer._workspace._frontend._gin_session.identity for layer in layers
        ]
        assert identities[0] != identities[1]
        # Every rank enters the public layer, but one uses a different live
        # communicator. Reject collectively before launching either session.
        with pytest.raises((RuntimeError, ValueError), match="GIN|gin"):
            layers[bootstrap.rank == 1].forward(inputs)
        for layer, reference in zip(layers, expected, strict=True):
            actual = layer.forward(inputs)
            torch.cuda.synchronize()
            assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
        retired = layers.pop()
        retired.destroy()
        retired.destroy()
        layers.append(_layer(bootstrap))
        result = layers[-1].forward(inputs)
        torch.cuda.synchronize()
        assert torch.equal(result.view(torch.int16), expected[-1].view(torch.int16))
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            for layer in reversed(layers):
                layer.destroy()


def test_nvfp4_gin_public_capture_rejection():
    bootstrap = _bootstrap()
    layer = _layer(bootstrap)
    failed = False
    try:
        inputs = _inputs(bootstrap.rank, 0, "balanced")
        layer.forward(inputs)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with (
            pytest.raises((RuntimeError, ValueError), match="GIN|gin"),
            torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream),
        ):
            layer.forward(inputs)
        torch.cuda.current_stream().wait_stream(stream)
        output = layer.forward(inputs)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            layer.destroy()


def test_nvfp4_gin_public_prestaged_inputs():
    bootstrap = _bootstrap()
    layers = [_layer(bootstrap), _layer(bootstrap, quantize_input=False)]
    failed = False
    try:
        original = _inputs(bootstrap.rank, 1, "matrix")
        expected = layers[0].forward(original).clone()
        torch.cuda.synchronize()
        workspace = layers[0]._workspace
        n = original.hidden_states.shape[0]
        inputs = MoEEpTensors(
            hidden_states=workspace.x[:n].clone(),
            scales=workspace.x_sf[:n].clone(),
            topk_ids=original.topk_ids,
            topk_weights=original.topk_weights,
        )
        actual = layers[1].forward(inputs)
        torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            for layer in reversed(layers):
                layer.destroy()
