"""Multi-GPU check: MXFP8 packed dispatch == BF16 dispatch, bit for bit.

``sm100_mxfp8_mxfp4_bf16_cutedsl`` with ``mxfp8_dispatch=True`` quantizes
tokens BEFORE EP dispatch and sends one packed uint8 row per token
(``[H]`` fp8 payload + ``[H/32]`` UE8M0 scale bytes) through nccl_ep's
single-tensor dispatch path. Per-token rows produce the same MXFP8 codes
whether quantized before or after dispatch, so the full EP forward must be
**bit-identical** to the default BF16-dispatch path — any transport-level
corruption of the narrower packed rows shows up as a hard mismatch.

Launch (4 GPU, SM100+):
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_mxfp8_dispatch_multirank.py -v -s \\
        -m "nvep and gpu_4 and arch_blackwell"
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

_PG_TIMEOUT = timedelta(minutes=60)

NUM_EXPERTS = 8
TOP_K = 2
TOKENS_PER_RANK = 64
# HIDDEN must be in the nccl_ep LL supported width set (2048, 2560, 4096,
# 5120, 6144, 7168, 8192 — probed in jobs 2390737/2390761; 1024 and 3072 are
# rejected by the device kernel). 4096 also exercises a real wire saving:
# the packed MXFP8 row pads to send width 2560 (0.625x the BF16 bytes).
HIDDEN = 4096
INTERMEDIATE = 512


def _run_one(layout_str: str, *, mxfp8_dispatch: bool):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        NcclEpConfig,
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig,
        SplitConfig,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    local_num_experts = NUM_EXPERTS // world_size
    offset = rank * local_num_experts

    gw = torch.Generator(device="cuda").manual_seed(2026)
    w1_full = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gw)
        * (HIDDEN**-0.5)
    ).to(torch.bfloat16)
    w2_full = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=gw)
        * (INTERMEDIATE**-0.5)
    ).to(torch.bfloat16)

    g = torch.Generator(device="cuda").manual_seed(3000 + rank)
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
            kernel=Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig(
                mxfp8_dispatch=mxfp8_dispatch
            ),
        ),
    )
    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    y = layer.forward(t)
    torch.cuda.synchronize()
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    layer.destroy()
    return y


def pytest_generate_tests(metafunc):
    if "layout" in metafunc.fixturenames:
        metafunc.parametrize("layout", ["expert_major", "rank_major"])


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_mxfp8_packed_dispatch_matches_bf16_dispatch(layout):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires an SM100-family GPU")
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTeDSL is not available")

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
    world_size = dist.get_world_size()
    if NUM_EXPERTS % world_size != 0:
        pytest.skip(
            f"num_experts={NUM_EXPERTS} not divisible by world_size={world_size}"
        )

    y_bf16 = _run_one(layout, mxfp8_dispatch=False)
    dist.barrier()
    y_packed = _run_one(layout, mxfp8_dispatch=True)
    dist.barrier()

    same = torch.equal(y_packed, y_bf16)
    max_delta = (y_packed.float() - y_bf16.float()).abs().max().item()
    rank = dist.get_rank()
    print(f"rank {rank}: [{layout}] packed==bf16 exact={same} max|Δ|={max_delta:.3g}")
    assert same, (
        f"[{layout}] MXFP8 packed dispatch diverged from BF16 dispatch "
        f"(max|Δ|={max_delta:.3g}) — per-token quantization commutes with "
        "dispatch, so any difference indicates payload corruption."
    )
