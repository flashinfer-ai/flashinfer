"""Four-rank split-EP coverage for the unified MegaMOE FC12 backend.

Run with:
    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_moe_ep_megamoe_fc12_split_multirank.py -v \
        -m "nvep and gpu_4 and arch_blackwell"
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest


_PG_TIMEOUT = timedelta(minutes=60)
_NUM_EXPERTS = 8
_TOP_K = 2
_TOKENS_PER_RANK = 32
_HIDDEN = 2048
_INTERMEDIATE = 1024


def _build_config(*, local_expert_offset: int, local_num_experts: int, max_tokens: int):
    from flashinfer.fused_moe import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MegaMoeFc12Config,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=_NUM_EXPERTS, top_k=_TOP_K),
        quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
        experts=ExpertConfig(
            intermediate_size=_INTERMEDIATE,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(MegaMoeFc12Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )


def _dense_reference(x, w13, w2, topk_ids, topk_weights):
    import torch
    import torch.nn.functional as F

    output = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(_TOP_K):
        for expert in range(_NUM_EXPERTS):
            rows = topk_ids[:, slot] == expert
            if not rows.any():
                continue
            gate, up = F.linear(x[rows].float(), w13[expert].float()).chunk(2, dim=-1)
            output[rows] += topk_weights[rows, slot : slot + 1].float() * F.linear(
                F.silu(gate) * up, w2[expert].float()
            )
    return output.bfloat16()


def _run_split_layer(layout_name: str):
    import torch
    import torch.distributed as dist

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
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    local_num_experts = _NUM_EXPERTS // world_size
    local_expert_offset = rank * local_num_experts
    layout = EpLayout[layout_name.upper()]
    max_compute_tokens = _TOKENS_PER_RANK * world_size
    if layout is EpLayout.EXPERT_MAJOR:
        max_compute_tokens *= local_num_experts

    weight_generator = torch.Generator(device="cuda").manual_seed(17)
    w13 = torch.randn(
        _NUM_EXPERTS,
        2 * _INTERMEDIATE,
        _HIDDEN,
        dtype=torch.bfloat16,
        device="cuda",
        generator=weight_generator,
    ).mul_(_HIDDEN**-0.5)
    w2 = torch.randn(
        _NUM_EXPERTS,
        _HIDDEN,
        _INTERMEDIATE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=weight_generator,
    ).mul_(_INTERMEDIATE**-0.5)

    tokens = torch.arange(_TOKENS_PER_RANK, device="cuda")
    topk_ids = torch.stack(
        (
            (tokens + rank) % _NUM_EXPERTS,
            (tokens + rank + local_num_experts + 1) % _NUM_EXPERTS,
        ),
        dim=1,
    ).to(torch.int64)
    topk_weights = torch.empty(
        _TOKENS_PER_RANK, _TOP_K, dtype=torch.float32, device="cuda"
    )
    topk_weights[:, 0] = 0.75
    topk_weights[:, 1] = 0.25
    input_generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = torch.randn(
        _TOKENS_PER_RANK,
        _HIDDEN,
        dtype=torch.bfloat16,
        device="cuda",
        generator=input_generator,
    )

    layer = MoEEpLayer(
        BootstrapConfig(
            world_size=world_size,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
            nccl_comm=None,
        ),
        FleetParams(
            num_experts=_NUM_EXPERTS,
            max_tokens_per_rank=_TOKENS_PER_RANK,
            token_hidden_size=_HIDDEN,
            dtype_bytes=2,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=layout,
        ),
        weights=MoEWeightPack(
            w13=w13[local_expert_offset : local_expert_offset + local_num_experts],
            w2=w2[local_expert_offset : local_expert_offset + local_num_experts],
        ),
        backend=SplitConfig(
            comm=NcclEpConfig(),
            kernel=FusedMoeKernelConfig(
                moe_config=_build_config(
                    local_expert_offset=local_expert_offset,
                    local_num_experts=local_num_experts,
                    max_tokens=max_compute_tokens,
                )
            ),
        ),
    )
    try:
        tensors = MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        actual = layer(tensors).clone()
        repeated = layer(tensors)
        expected = _dense_reference(x, w13, w2, topk_ids, topk_weights)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)
        torch.testing.assert_close(repeated, expected, atol=0.1, rtol=0.1)
    finally:
        layer.destroy()
    return rank


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("layout_name", ("expert_major", "rank_major"))
def test_moe_ep_megamoe_fc12_split_multirank_matches_dense_reference(layout_name):
    import torch
    import torch.distributed as dist

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("MegaMOE FC12 split backend requires SM100 or SM103")
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=_PG_TIMEOUT)
    if dist.get_world_size() != 4:
        pytest.skip("requires exactly four ranks")

    rank = _run_split_layer(layout_name)
    dist.barrier()
    print(f"rank {rank}: MegaMOE FC12 split {layout_name} matches dense reference")
