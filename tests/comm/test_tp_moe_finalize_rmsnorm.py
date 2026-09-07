# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team

import importlib
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer.comm import (
    init_tp_moe_finalize_rmsnorm,
    supports_tp_moe_finalize_rmsnorm,
    tp_moe_finalize_rmsnorm,
)

WORLD_SIZE = 4
TOPK = 8
HIDDEN_SIZE = 4096


def _assert_model_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    delta = actual.float() - expected.float()
    rel_l2 = delta.norm() / expected.float().norm().clamp_min(1e-12)
    assert rel_l2 < 1e-3
    assert delta.abs().max() <= 0.03125


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _reference(
    partial_rows: torch.Tensor,
    expert_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    gain_center: float,
    weight_before_norm: bool,
) -> torch.Tensor:
    rows = partial_rows.float()
    dist.all_reduce(rows)
    rows = rows.to(torch.bfloat16).float()
    if weight_before_norm:
        rows *= norm_weight.float() + gain_center
    inv_rms = torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + eps)
    normalized = rows * inv_rms
    if not weight_before_norm:
        normalized *= norm_weight.float() + gain_center
    return (
        normalized.mul(expert_weights.float().unsqueeze(-1))
        .sum(dim=1)
        .to(torch.bfloat16)
    )


def _rounded_reference(
    partial_rows: torch.Tensor,
    expert_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    gain_center: float,
    weight_before_norm: bool,
) -> torch.Tensor:
    """The obsolete contract with an expert-local BF16 normalization store."""
    rows = partial_rows.float()
    dist.all_reduce(rows)
    rows = rows.to(torch.bfloat16).float()
    if weight_before_norm:
        rows *= norm_weight.float() + gain_center
    inv_rms = torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (rows * inv_rms).to(torch.bfloat16)
    if not weight_before_norm:
        normalized = (normalized.float() * (norm_weight.float() + gain_center)).to(
            torch.bfloat16
        )
    return (
        normalized.float()
        .mul(expert_weights.float().unsqueeze(-1))
        .sum(dim=1)
        .to(torch.bfloat16)
    )


def _worker(local_rank: int, master_port: int) -> None:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        rank=local_rank,
        world_size=WORLD_SIZE,
        init_method=f"tcp://127.0.0.1:{master_port}",
    )
    assert init_tp_moe_finalize_rmsnorm()
    device = torch.device("cuda", local_rank)
    contract_rows = torch.empty(
        1, TOPK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    contract_weights = torch.empty(1, TOPK, dtype=torch.bfloat16, device=device)
    contract_norm = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    contract_output = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    misaligned_rows = torch.empty(
        TOPK * HIDDEN_SIZE + 1, dtype=torch.bfloat16, device=device
    )[1:].view(1, TOPK, HIDDEN_SIZE)
    assert misaligned_rows.is_contiguous() and misaligned_rows.data_ptr() % 16 != 0
    assert not supports_tp_moe_finalize_rmsnorm(
        misaligned_rows, contract_weights, contract_norm, contract_output
    )

    finalize_module = importlib.import_module("flashinfer.comm.tp_moe_finalize_rmsnorm")
    initialized_device = finalize_module._WS.dev_ord
    finalize_module._WS.dev_ord = (initialized_device + 1) % torch.cuda.device_count()
    try:
        assert not supports_tp_moe_finalize_rmsnorm(
            contract_rows, contract_weights, contract_norm, contract_output
        )
    finally:
        finalize_module._WS.dev_ord = initialized_device

    for weight_before_norm in (False, True):
        for num_tokens in (1, 17, 65):
            generator = torch.Generator(device=device).manual_seed(1000 + local_rank)
            partial_rows = torch.randn(
                num_tokens,
                TOPK,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            shared = torch.Generator(device=device).manual_seed(2000)
            expert_weights = torch.randn(
                num_tokens,
                TOPK,
                dtype=torch.bfloat16,
                device=device,
                generator=shared,
            )
            norm_weight = torch.randn(
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
                generator=shared,
            )
            expected = _reference(
                partial_rows,
                expert_weights,
                norm_weight,
                1e-8,
                1.0,
                weight_before_norm,
            )
            output = torch.empty_like(expected)
            tp_moe_finalize_rmsnorm(
                partial_rows,
                expert_weights,
                norm_weight,
                output,
                eps=1e-8,
                gain_center=1.0,
                weight_before_norm=weight_before_norm,
            )
            _assert_model_close(output, expected)

            gathered = [torch.empty_like(output) for _ in range(WORLD_SIZE)]
            dist.all_gather(gathered, output)
            assert all(
                torch.equal(gathered[0], rank_output) for rank_output in gathered
            )

    # Isolate post-collective arithmetic by making the other three ranks
    # contribute exact zeros. This must prefer the FP32-until-final contract
    # over the obsolete expert-local BF16 round.
    contract_tokens = 17
    shared = torch.Generator(device=device).manual_seed(7000)
    rank0_rows = torch.randn(
        contract_tokens,
        TOPK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=shared,
    )
    contract_rows = rank0_rows if local_rank == 0 else torch.zeros_like(rank0_rows)
    contract_weights = torch.randn(
        contract_tokens,
        TOPK,
        dtype=torch.bfloat16,
        device=device,
        generator=shared,
    )
    contract_norm = torch.randn(
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=shared,
    )
    for weight_before_norm in (False, True):
        expected = _reference(
            contract_rows,
            contract_weights,
            contract_norm,
            1e-8,
            1.0,
            weight_before_norm,
        )
        rounded = _rounded_reference(
            contract_rows,
            contract_weights,
            contract_norm,
            1e-8,
            1.0,
            weight_before_norm,
        )
        assert torch.count_nonzero(expected != rounded) > 0
        output = torch.empty_like(expected)
        tp_moe_finalize_rmsnorm(
            contract_rows,
            contract_weights,
            contract_norm,
            output,
            eps=1e-8,
            gain_center=1.0,
            weight_before_norm=weight_before_norm,
        )
        fp32_error = (output.float() - expected.float()).abs().mean()
        rounded_error = (output.float() - rounded.float()).abs().mean()
        assert fp32_error < rounded_error

    graph_rows_generator = torch.Generator(device=device).manual_seed(3000 + local_rank)
    graph_rows = torch.randn(
        17,
        TOPK,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=graph_rows_generator,
    )
    graph_metadata_generator = torch.Generator(device=device).manual_seed(4000)
    graph_weights = torch.randn(
        17,
        TOPK,
        dtype=torch.bfloat16,
        device=device,
        generator=graph_metadata_generator,
    )
    graph_norm = torch.randn(
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=device,
        generator=graph_metadata_generator,
    )
    graph_output = torch.empty(17, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    tp_moe_finalize_rmsnorm(
        graph_rows,
        graph_weights,
        graph_norm,
        graph_output,
        eps=1e-4,
        gain_center=1.0,
        weight_before_norm=True,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tp_moe_finalize_rmsnorm(
            graph_rows,
            graph_weights,
            graph_norm,
            graph_output,
            eps=1e-4,
            gain_center=1.0,
            weight_before_norm=True,
        )
    for iteration in range(4):
        replay_generator = torch.Generator(device=device).manual_seed(
            5000 + 10 * iteration + local_rank
        )
        graph_rows.normal_(generator=replay_generator)
        expected = _reference(graph_rows, graph_weights, graph_norm, 1e-4, 1.0, True)
        graph.replay()
        _assert_model_close(graph_output, expected)
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tp_moe_finalize_rmsnorm_eager_and_cudagraph() -> None:
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("TP MoE finalize requires SM103")
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip("Four GPUs are required")
    mp.spawn(_worker, args=(_free_port(),), nprocs=WORLD_SIZE)
