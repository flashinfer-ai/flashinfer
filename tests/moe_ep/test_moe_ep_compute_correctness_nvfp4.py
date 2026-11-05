"""Multi-GPU functional correctness for LL EXPERT_MAJOR + RANK_MAJOR (NVFP4).

Exercises explicit TRT-LLM and CuTe/TRT-LLM selection through
``FusedMoeSplitKernelBackend``. Multi-backend cases also check graph replay
with changed activation and routing bytes.

Asserts ``EP == non-EP kernel``; the ``non-EP kernel == torch oracle`` anchor
for this dtype path is the single-GPU
``test_split_fused_moe_kernel_vs_reference.py::test_split_nvfp4_kernel_matches_torch_reference``.

Launch (4 GPU, SM100+):
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_compute_correctness_nvfp4.py -v -s \\
        -m "nvep and gpu_4"
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

# First-use JIT compile of reference kernels (e.g. fused_moe_trtllm_sm100)
# can exceed torch's 10-min default watchdog while other ranks wait in a
# collective; a cold cache is not a hang.
_PG_TIMEOUT = timedelta(minutes=60)

NUM_EXPERTS = 16
TOP_K = 8
TOKENS_PER_RANK = 128
HIDDEN = 8192
INTERMEDIATE = 2048
RTOL = 5e-2
ATOL = 5e-2


def _build_nvfp4_moe_config(*, offset, local_num_experts, max_tokens):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        TrtllmFp4Config,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=NUM_EXPERTS, top_k=TOP_K),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_expert_offset=offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(TrtllmFp4Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )


def _quantize_activation(x):
    import torch

    from flashinfer.quantization.fp4_quantization import fp4_quantize

    global_scale = torch.ones(1, dtype=torch.float32, device=x.device)
    hidden_states_q, hidden_states_scale = fp4_quantize(
        x,
        global_scale=global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    if hidden_states_scale.dim() > 2:
        hidden_states_scale = hidden_states_scale.squeeze(-1)
    return hidden_states_q, hidden_states_scale


def _kernel_full_moe_reference(x, w1_full, w2_full, topk_ids, topk_weights):
    import torch

    from flashinfer.fused_moe.api import MoEActivationPack
    from flashinfer.fused_moe.layer import MoELayer
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )

    num_tokens = x.shape[0]
    cfg = _build_nvfp4_moe_config(
        offset=0, local_num_experts=NUM_EXPERTS, max_tokens=num_tokens
    )
    canonical = MoEWeightPack(w13=w1_full, w2=w2_full)
    wp = materialize_fused_moe_weights(canonical, cfg)
    x_q, x_sf = _quantize_activation(x)
    act = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_sf,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights.to(torch.float32),
    )
    return MoELayer(cfg)(act, wp)


def _run_one_layout(layout_str, compute_backends="trtllm"):
    from dataclasses import replace

    import torch
    import torch.distributed as dist

    from flashinfer.fused_moe.api import (
        BackendOptions,
        CuteDslConfig,
        MoEFinalizeConfig,
        TrtllmFp4Config,
    )
    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        NcclEpConfig,
        SplitConfig,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    local_num_experts = NUM_EXPERTS // world_size
    offset = rank * local_num_experts

    gw = torch.Generator(device="cuda").manual_seed(2024)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gw)
        * (HIDDEN**-0.5)
    ).to(torch.bfloat16)
    w2_full = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=gw)
        * (INTERMEDIATE**-0.5)
    ).to(torch.bfloat16)

    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = torch.randn(TOKENS_PER_RANK, HIDDEN, device="cuda", generator=g).to(
        torch.bfloat16
    )
    scores = torch.randn(TOKENS_PER_RANK, NUM_EXPERTS, device="cuda", generator=g)
    topk_ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(TOKENS_PER_RANK, TOP_K, device="cuda", generator=g), dim=-1
    )

    layout = (
        EpLayout.RANK_MAJOR if layout_str == "rank_major" else EpLayout.EXPERT_MAJOR
    )
    if layout is EpLayout.RANK_MAJOR:
        max_tokens = TOKENS_PER_RANK * world_size
    else:
        max_tokens = local_num_experts * TOKENS_PER_RANK * world_size

    moe_config = _build_nvfp4_moe_config(
        offset=offset,
        local_num_experts=local_num_experts,
        max_tokens=max_tokens,
    )
    if compute_backends != "trtllm":
        candidates = (CuteDslConfig(), TrtllmFp4Config())
        if compute_backends == "trtllm_first":
            candidates = candidates[::-1]
        moe_config = replace(
            moe_config,
            backend=BackendOptions(candidates=candidates),
            finalize=MoEFinalizeConfig(use_fused_finalize=True),
        )
    canonical_weights = MoEWeightPack(
        w13=w1_full[offset : offset + local_num_experts].contiguous(),
        w2=w2_full[offset : offset + local_num_experts].contiguous(),
    )

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        stream=torch.cuda.current_stream().cuda_stream,
        nccl_comm=None,
    )
    layer = MoEEpLayer(
        bootstrap,
        FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=TOKENS_PER_RANK,
            token_hidden_size=HIDDEN,
            dtype_bytes=2,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=layout,
        ),
        weights=canonical_weights,
        backend=SplitConfig(
            comm=NcclEpConfig(),
            kernel=FusedMoeKernelConfig(moe_config=moe_config),
        ),
    )

    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()
    assert y.shape == x.shape

    y_kernel = _kernel_full_moe_reference(x, w1_full, w2_full, topk_ids, topk_weights)
    yf, kf = y.float(), y_kernel.float()

    def _rel(a, b):
        return (a - b).abs().amax().item() / b.abs().amax().clamp_min(1e-6).item()

    ep_vs_kernel = _rel(yf, kf)
    if rank == 0:
        print(f"[nvfp4 {layout_str}] EP-vs-kernel rel-err={ep_vs_kernel:.4f}")

    torch.testing.assert_close(yf, kf, rtol=RTOL, atol=ATOL)
    if compute_backends != "trtllm":
        compute = layer._kernel._compute
        assert {r.backend_key for r in compute.runners} == {
            "cute_dsl",
            "trtllm_fp4_routed",
        }
        assert compute.winner_backend in {"cute_dsl", "trtllm_fp4_routed"}
        winners = dict(compute._winners)
        state = layer.create_graph_state(t)
        layer.forward(t, graph_state=state)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_y = layer.forward(t, graph_state=state)
        # Change both activation and routing bytes at the captured addresses.
        x.mul_(0.75)
        topk_ids.add_(1).remainder_(NUM_EXPERTS)
        expected = _kernel_full_moe_reference(
            x, w1_full, w2_full, topk_ids, topk_weights
        )
        graph_y.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            graph_y.float(), expected.float(), rtol=RTOL, atol=ATOL
        )
        assert dict(compute._winners) == winners
        print(
            f"rank {rank}: {layout_str} {compute_backends} winner={compute.winner_backend}"
        )
        graph.reset()
        state.destroy()
    layer.destroy()
    return rank, ep_vs_kernel


def pytest_generate_tests(metafunc):
    if "layout" in metafunc.fixturenames:
        metafunc.parametrize("layout", ["expert_major", "rank_major"])


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize("compute_backends", ["trtllm", "cute_first", "trtllm_first"])
def test_moe_ep_nvfp4_compute_matches_dense_reference(layout, compute_backends):
    import torch

    from flashinfer.fused_moe.api import CuteDslConfig, TrtllmFp4Config

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 EP compute requires SM100+")
    major, minor = torch.cuda.get_device_capability()
    if not TrtllmFp4Config.supported(major * 10 + minor):
        pytest.skip("requires TRT-LLM NVFP4")
    if compute_backends != "trtllm":
        if not CuteDslConfig.supported(major * 10 + minor):
            pytest.skip("requires both CuTe and TRT-LLM NVFP4 backends")

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
    world_size = dist.get_world_size()
    if NUM_EXPERTS % world_size != 0:
        pytest.skip(
            f"num_experts={NUM_EXPERTS} not divisible by world_size={world_size}"
        )

    rank, ep_vs_kernel = _run_one_layout(layout, compute_backends)
    dist.barrier()
    print(f"rank {rank}: nvfp4 {layout} EP==kernel OK (rel-err={ep_vs_kernel:.4f})")
