# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise retained legacy finalization while internode GIN is disabled."""

import os

import pytest
import torch

from .test_moe_ep_nvfp4_cutedsl_mega_multirank import _launcher_ranks, _run_mega_layer

pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_blackwell, pytest.mark.node_exclusive]


@pytest.mark.parametrize("live,per_token_scale", [(17, False), (64, True)])
def test_legacy_nvfp4_off_persistent_finalization(live, per_token_scale):
    from flashinfer.moe_ep import FinalizeMode

    rank, world = _launcher_ranks()
    if world != 4:
        pytest.skip("requires four ranks in one NVLink domain")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    live_count = torch.full((1,), live, dtype=torch.int32, device="cuda")
    _run_mega_layer(
        rank,
        world,
        quantize_input=True,
        num_tokens=live,
        max_tokens=64,
        per_token_scale=per_token_scale,
        config_extra=dict(
            finalize_mode=FinalizeMode.PERSISTENT,
            num_valid_tokens_tensor=live_count,
        ),
    )


@pytest.mark.parametrize("manual", [False, True], ids=["cvt", "manual"])
def test_legacy_nvfp4_fp4_decoder_environment(manual, monkeypatch, tmp_path):
    """Retain FI's environment-selected decoder and exact FP4 reduction."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    monkeypatch.setenv("MEGA_F4CVT_USE_MANUAL", "1" if manual else "0")
    tokens, topk, hidden = 3, 4, 256
    packed = (
        (torch.arange(tokens * topk * hidden // 2, device="cuda") % 256)
        .to(torch.uint8)
        .reshape(tokens, topk, hidden // 2)
    )
    sf = torch.full(
        (tokens, topk, hidden // 16), 6.0, dtype=torch.bfloat16, device="cuda"
    )
    scores = torch.tensor([0.125, 0.25, 0.5, 1.0], device="cuda").repeat(tokens, 1)
    output = torch.full(
        (tokens, hidden), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    reducer = pkg.TopkReduce(hidden, topk, pkg.CombineFormat.parse("16e2m1xbf16"))
    kwargs = dict(
        combine_quant=from_dlpack(
            packed.view(torch.float4_e2m1fn_x2), assumed_align=16
        ),
        combine_sf=from_dlpack(sf, assumed_align=16),
        reduced_output=from_dlpack(output, assumed_align=16),
        topk_score=from_dlpack(scores, assumed_align=4),
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    executable = cute.compile(
        reducer, **kwargs, options=f"--keep-ptx --dump-dir={tmp_path}"
    )
    executable(**kwargs)
    torch.cuda.synchronize()
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="cuda",
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    terms = table[codes]
    expected = torch.zeros((tokens, hidden), dtype=torch.float32, device="cuda")
    for k in range(topk):
        expected.add_(terms[:, k] * scores[:, k, None])
    assert torch.equal(output.view(torch.int16), expected.bfloat16().view(torch.int16))
    ptx = getattr(executable, "__ptx__", None) or "\n".join(
        path.read_text() for path in tmp_path.glob("*.ptx")
    )
    assert ptx, "PTX must be retained to verify actual decoder selection"
    assert ("cvt.rn.f16x2.e2m1x2" in ptx) is not manual


def test_legacy_nvfp4_off_has_no_gin_dependencies(monkeypatch):
    """Default/off eager and graph calls must never touch a GIN entry point."""
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import gin

    rank, world = _launcher_ranks()
    if world != 4:
        pytest.skip("requires four ranks in one NVLink domain")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def forbidden(*args, **kwargs):
        raise AssertionError("internode=False must not enter GIN code")

    monkeypatch.setattr(gin, "collective_gin_preflight", forbidden)
    monkeypatch.setattr(gin, "create_gin_session", forbidden)
    monkeypatch.setattr(gin.GinSession, "__init__", forbidden)
    generator = torch.Generator(device="cuda").manual_seed(1901 + rank)
    cfg = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=1024, top_k=4)
    assert cfg.internode is False
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world, rank=rank),
        fleet_params=FleetParams(
            num_experts=8, max_tokens_per_rank=64, token_hidden_size=2048
        ),
        weights=MoEWeightPack(
            w13=torch.randn(
                (2, 2048, 2048),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            * 0.01,
            w2=torch.randn(
                (2, 2048, 1024),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            * 0.01,
        ),
        backend=MegaConfig(
            megakernel=cfg, quantize_input=True, preprocess_weights=True
        ),
    )
    inputs = MoEEpTensors(
        hidden_states=torch.randn(
            (64, 2048), dtype=torch.bfloat16, device="cuda", generator=generator
        ),
        topk_ids=torch.arange(256, device="cuda", dtype=torch.int64).reshape(64, 4) % 8,
        topk_weights=torch.full((64, 4), 0.25, dtype=torch.float32, device="cuda"),
    )
    layer.warmup(inputs)
    assert not layer._kernel.requires_collective_forward_validation
    assert layer._workspace._frontend._gin_session is None
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer.forward(inputs)
    for iteration in range(3):
        inputs.hidden_states.mul_(0.5)
        expected = layer.forward(inputs).clone()
        actual.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual, expected), iteration
    del graph
    layer.destroy()
