# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team

"""End-to-end accuracy of the MNNVL collectives fused with normalization.

Covers the column all-gather + gated residual dual norm, the column
all-gather + RMSNorm + NVFP4 quantize, and the MoE finalize + per-expert
RMSNorm + all-reduce against PyTorch references. Each worker reuses one
workspace across a sequence of token counts, so stale mailbox rows from a
larger previous call and the Lamport stage rotation are both exercised.
"""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer import fp4_quantize
from flashinfer.comm.mnnvl_cutedsl_ag import (
    MNNVLCuteDSLAllGatherDualNormWorkspace,
    MNNVLCuteDSLAllGatherNormQuantWorkspace,
    all_gather_gated_residual_next_rmsnorm,
    all_gather_rmsnorm_fp4_quantize,
    supports_all_gather_gated_residual_next_rmsnorm,
    supports_all_gather_rmsnorm_fp4_quantize,
)
from flashinfer.comm.mnnvl_cutedsl_moe_finalize import (
    MNNVLCuteDSLMoEFinalizeAllReduceWorkspace,
    moe_finalize_rmsnorm_allreduce,
    supports_moe_finalize_rmsnorm_allreduce,
)
from flashinfer.norm import (
    gated_residual_next_rmsnorm,
    supports_gated_residual_next_rmsnorm,
)

MAX_TOKENS = 64
# Shrink, grow and repeat so every call sees rows left by a different M and
# the sequence wraps the three Lamport stages more than once.
TOKEN_SEQUENCE = (MAX_TOKENS, 1, 7, MAX_TOKENS, 3, 33, 1)

DUAL_NORM_HIDDEN = 8192
POST_EPS = 1e-8
NEXT_EPS = 1e-5
NEXT_GAIN_CENTER = 1.0

NORM_QUANT_EPS = 1e-6
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

FINALIZE_HIDDEN = 4096
FINALIZE_ROW_STRIDE = FINALIZE_HIDDEN + 64
FINALIZE_TOPK = 8
FINALIZE_EXPERTS_PER_RANK = 8
FINALIZE_EXTRA_ROWS = 5
FINALIZE_EPS = 1e-6


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _init_group(local_rank: int, master_port: int | None, world_size: int) -> None:
    torch.cuda.set_device(local_rank)
    if master_port is None:
        store = dist.TCPStore(
            os.environ["MASTER_ADDR"],
            int(os.environ["MASTER_PORT"]),
            world_size,
            False,
            timeout=timedelta(seconds=180),
        )
        store = dist.PrefixStore(os.environ["FLASHINFER_MNNVL_FUSED_TEST_CASE"], store)
        dist.init_process_group(
            "nccl", store=store, rank=int(os.environ["RANK"]), world_size=world_size
        )
    else:
        dist.init_process_group(
            "nccl",
            rank=local_rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{master_port}",
        )


def _run_env_worker(worker, args) -> None:
    # Isolate collectives so a stalled kernel becomes a failing test ID and
    # cannot strand the rest of the suite in the same CUDA context.
    os.environ["FLASHINFER_MNNVL_FUSED_TEST_CASE"] = repr((worker.__name__, args[2:]))
    process = mp.get_context("spawn").Process(target=worker, args=args)
    process.start()
    process.join(300)
    stalled = process.is_alive()
    if stalled:
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
    assert not stalled, "collective worker timed out after 300 seconds"
    assert process.exitcode == 0, f"collective worker exited with {process.exitcode}"


def _launch(worker, world_size: int, *args) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("MNNVL fused collectives require SM100, SM103, or SM107")
    from flashinfer.comm.mnnvl import is_multicast_supported

    if not is_multicast_supported(torch.cuda.current_device()):
        pytest.skip("NVLink multicast is unavailable")
    if "LOCAL_RANK" in os.environ:
        if int(os.environ["WORLD_SIZE"]) != world_size:
            pytest.skip("different torchrun world size")
        _run_env_worker(worker, (int(os.environ["LOCAL_RANK"]), None, world_size, *args))
        return
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} GPUs are required")
    mp.spawn(worker, args=(_free_port(), world_size, *args), nprocs=world_size)


def _shared_generator(seed: int) -> torch.Generator:
    """The same stream on every rank, for replicated inputs."""
    return torch.Generator(device="cuda").manual_seed(seed)


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    return (delta.norm() / expected.float().norm().clamp_min(1e-12)).item()


# ---------------------------------------------------------------------------
# All-gather + gated residual + next RMSNorm
# ---------------------------------------------------------------------------


def _dual_norm_reference(branch, residual, alpha, beta, next_weight, post_weight):
    b = branch.float()
    if post_weight is not None:
        b = b * post_weight.float()
    b = b * torch.rsqrt(b.square().mean(dim=-1, keepdim=True) + POST_EPS)
    residual_out = alpha * residual + beta * b
    ffn_input = residual_out * torch.rsqrt(
        residual_out.square().mean(dim=-1, keepdim=True) + NEXT_EPS
    )
    ffn_input = ffn_input * (next_weight.float() + NEXT_GAIN_CENTER)
    return residual_out, ffn_input.to(torch.bfloat16)


def _dual_norm_worker(
    local_rank: int,
    master_port: int | None,
    world_size: int,
    with_post_weight: bool,
) -> None:
    _init_group(local_rank, master_port, world_size)
    rank = dist.get_rank()
    hidden = DUAL_NORM_HIDDEN
    shard_width = hidden // world_size
    columns = slice(rank * shard_width, (rank + 1) * shard_width)
    workspace = MNNVLCuteDSLAllGatherDualNormWorkspace(
        world_size, rank, MAX_TOKENS, hidden, group=dist.group.WORLD
    )
    try:
        gen = _shared_generator(6544)
        next_weight = (
            torch.randn(hidden, generator=gen, device="cuda").mul_(0.1).bfloat16()
        )
        gate = torch.empty(hidden, device="cuda").uniform_(-3, 3, generator=gen)
        beta = torch.sigmoid(gate)
        alpha = torch.sqrt(torch.clamp(1.0 - beta.square(), min=1e-3))
        post_weight = None
        if with_post_weight:
            post_weight = (
                torch.randn(hidden, generator=gen, device="cuda")
                .mul_(0.1)
                .add_(1.0)
                .bfloat16()
            )

        for step, m in enumerate(TOKEN_SEQUENCE):
            # Full rows are replicated; each rank only passes its column shard.
            gen = _shared_generator(1000 + step)
            branch = torch.randn(
                m, hidden, generator=gen, device="cuda", dtype=torch.bfloat16
            )
            residual = torch.randn(m, hidden, generator=gen, device="cuda")
            branch_shard = branch[:, columns].contiguous()
            inputs = (branch_shard, residual, alpha, beta, next_weight)

            local_ok = supports_all_gather_gated_residual_next_rmsnorm(
                workspace, *inputs, post_weight
            )
            assert workspace.check_support(local_ok), f"unsupported at M={m}"
            residual_out, ffn_input = all_gather_gated_residual_next_rmsnorm(
                *inputs,
                workspace,
                post_eps=POST_EPS,
                next_eps=NEXT_EPS,
                next_gain_center=NEXT_GAIN_CENTER,
                post_weight=post_weight,
            )
            torch.cuda.synchronize()

            ref_residual, ref_ffn = _dual_norm_reference(
                branch, residual, alpha, beta, next_weight, post_weight
            )
            context = f"rank {rank}, step {step}, M={m}"
            torch.testing.assert_close(
                residual_out, ref_residual, rtol=1e-5, atol=1e-5, msg=context
            )
            torch.testing.assert_close(
                ffn_input, ref_ffn, rtol=0.015625, atol=0.03125, msg=context
            )

            # The single-GPU kernel on the gathered rows must agree.
            local_inputs = (branch, residual, next_weight, alpha, beta)
            if supports_gated_residual_next_rmsnorm(
                *local_inputs, post_weight=post_weight
            ):
                local_residual, local_ffn = gated_residual_next_rmsnorm(
                    *local_inputs,
                    post_eps=POST_EPS,
                    next_eps=NEXT_EPS,
                    next_gain_center=NEXT_GAIN_CENTER,
                    post_weight=post_weight,
                )
                torch.testing.assert_close(
                    residual_out, local_residual, rtol=1e-5, atol=1e-5, msg=context
                )
                torch.testing.assert_close(
                    ffn_input, local_ffn, rtol=0.015625, atol=0.03125, msg=context
                )
    finally:
        workspace.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("with_post_weight", [False, True])
def test_all_gather_gated_residual_next_rmsnorm_accuracy(
    world_size: int, with_post_weight: bool
) -> None:
    _launch(_dual_norm_worker, world_size, with_post_weight)


# ---------------------------------------------------------------------------
# All-gather + RMSNorm + NVFP4 quantize
# ---------------------------------------------------------------------------


def _dequantize_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Linear-layout NVFP4 (low nibble first) to FP32."""
    lut = torch.tensor(E2M1_VALUES, device=packed.device)
    lut = torch.cat([lut, -lut])
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    values = lut[codes]
    block_scale = scales.view(torch.float8_e4m3fn).float()
    values = values.unflatten(-1, (-1, 16)) * block_scale.unsqueeze(-1)
    return values.flatten(-2) / global_scale


def _norm_quant_worker(
    local_rank: int,
    master_port: int | None,
    world_size: int,
    hidden: int,
    gain_center: float,
) -> None:
    _init_group(local_rank, master_port, world_size)
    rank = dist.get_rank()
    shard_width = hidden // world_size
    columns = slice(rank * shard_width, (rank + 1) * shard_width)
    workspace = MNNVLCuteDSLAllGatherNormQuantWorkspace(
        world_size, rank, MAX_TOKENS, hidden, group=dist.group.WORLD
    )
    try:
        gen = _shared_generator(4096 + hidden)
        weight = (
            torch.randn(hidden, generator=gen, device="cuda")
            .mul_(0.1)
            .add_(1.0 - gain_center)
            .bfloat16()
        )

        for step, m in enumerate(TOKEN_SEQUENCE):
            gen = _shared_generator(2000 + step)
            x = torch.randn(m, hidden, generator=gen, device="cuda").bfloat16()
            values = x.float()
            y = values * torch.rsqrt(
                values.square().mean(dim=-1, keepdim=True) + NORM_QUANT_EPS
            )
            y = (y * (weight.float() + gain_center)).bfloat16()
            global_scale = (
                (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / y.float().abs().amax()
            ).reshape(1)
            shard = x[:, columns].contiguous()

            local_ok = supports_all_gather_rmsnorm_fp4_quantize(
                workspace, shard, weight, global_scale
            )
            assert workspace.check_support(local_ok), f"unsupported at M={m}"
            packed, scales = all_gather_rmsnorm_fp4_quantize(
                shard,
                weight,
                global_scale,
                workspace,
                eps=NORM_QUANT_EPS,
                gain_center=gain_center,
            )
            torch.cuda.synchronize()

            ref_packed, ref_scales = fp4_quantize(
                y, global_scale, sf_vec_size=16, is_sf_swizzled_layout=False
            )
            ref_packed = ref_packed.view(torch.uint8).reshape(m, hidden // 2)
            ref_scales = ref_scales.view(torch.uint8).reshape(m, hidden // 16)
            context = f"rank {rank}, step {step}, M={m}"
            assert packed.shape == (m, hidden // 2), context
            assert scales.shape == (m, hidden // 16), context

            # The kernel's BF16 y can differ from the reference by one ULP,
            # which flips a few rounding boundaries; everything else is exact.
            scale_match = (scales == ref_scales).float().mean().item()
            packed_match = (packed == ref_packed).float().mean().item()
            assert scale_match >= 0.99, f"{context}: scale match {scale_match:.4f}"
            assert packed_match >= 0.99, f"{context}: packed match {packed_match:.4f}"

            dequant = _dequantize_nvfp4(packed, scales, global_scale)
            ref_dequant = _dequantize_nvfp4(ref_packed, ref_scales, global_scale)
            assert torch.isfinite(dequant).all(), context
            rel = _rel_l2(dequant, ref_dequant)
            assert rel < 1e-2, f"{context}: dequantized relative L2 {rel:.4g}"
            rel = _rel_l2(dequant, y)
            assert rel < 0.15, f"{context}: quantization relative L2 {rel:.4g}"
    finally:
        workspace.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("hidden", [4096, 8192])
@pytest.mark.parametrize("gain_center", [0.0, 1.0])
def test_all_gather_rmsnorm_fp4_quantize_accuracy(
    world_size: int, hidden: int, gain_center: float
) -> None:
    _launch(_norm_quant_worker, world_size, hidden, gain_center)


# ---------------------------------------------------------------------------
# MoE finalize + per-expert RMSNorm + all-reduce
# ---------------------------------------------------------------------------


def _local_routes(
    m: int, rank: int, world_size: int, step: int, num_rows: int, gen_seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicated routing, and this rank's permuted rows and route indices."""
    num_experts = FINALIZE_EXPERTS_PER_RANK * world_size
    gen = _shared_generator(gen_seed)
    logits = torch.randn(m, num_experts, generator=gen, device="cuda")
    route_weights, experts = torch.topk(logits.softmax(dim=-1), FINALIZE_TOPK, dim=-1)
    route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)

    local = (experts // FINALIZE_EXPERTS_PER_RANK) == rank
    num_local = int(local.sum().item())
    assert num_local <= num_rows
    permuted_idx = torch.full_like(experts, -1, dtype=torch.int32)
    rank_gen = torch.Generator(device="cuda").manual_seed(gen_seed * 16 + rank + 1)
    order = torch.randperm(num_rows, generator=rank_gen, device="cuda")[:num_local]
    permuted_idx[local] = order.to(torch.int32)
    # Routes at or past the row count belong to other ranks as well.
    nonlocal_routes = (~local).nonzero()
    if step % 2 and nonlocal_routes.numel():
        first = nonlocal_routes[0]
        permuted_idx[first[0], first[1]] = num_rows
    return route_weights, permuted_idx, local


def _finalize_partial(
    routed_rows: torch.Tensor,
    route_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    local: torch.Tensor,
    norm_weight: torch.Tensor,
    gain_center: float,
    weight_before_norm: bool,
) -> torch.Tensor:
    m = permuted_idx.shape[0]
    idx = permuted_idx.long().clamp(0, routed_rows.shape[0] - 1)
    rows = routed_rows[idx.flatten(), :FINALIZE_HIDDEN].float()
    rows = rows.view(m, FINALIZE_TOPK, FINALIZE_HIDDEN)
    gamma = norm_weight.float() + gain_center
    if weight_before_norm:
        rows = rows * gamma
    rows = rows * torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + FINALIZE_EPS)
    if not weight_before_norm:
        rows = rows * gamma
    weights = route_weights.float() * local
    return (rows * weights.unsqueeze(-1)).sum(dim=1)


def _finalize_worker(
    local_rank: int,
    master_port: int | None,
    world_size: int,
    routing_fp32: bool,
    weight_before_norm: bool,
) -> None:
    _init_group(local_rank, master_port, world_size)
    rank = dist.get_rank()
    gain_center = 1.0
    workspace = MNNVLCuteDSLMoEFinalizeAllReduceWorkspace(
        world_size,
        rank,
        MAX_TOKENS,
        FINALIZE_HIDDEN,
        FINALIZE_TOPK,
        group=dist.group.WORLD,
    )
    try:
        gen = _shared_generator(3000)
        norm_weight = (
            torch.randn(FINALIZE_HIDDEN, generator=gen, device="cuda")
            .mul_(0.1)
            .bfloat16()
        )
        # Per-rank row count differs, as it would for each rank's experts.
        num_rows = MAX_TOKENS * FINALIZE_TOPK + FINALIZE_EXTRA_ROWS + rank

        for step, m in enumerate(TOKEN_SEQUENCE):
            route_weights, permuted_idx, local = _local_routes(
                m, rank, world_size, step, num_rows, 4000 + step
            )
            rank_gen = torch.Generator(device="cuda").manual_seed(
                5000 + 16 * step + rank
            )
            routed_rows = torch.randn(
                num_rows,
                FINALIZE_ROW_STRIDE,
                generator=rank_gen,
                device="cuda",
                dtype=torch.bfloat16,
            )
            route_weights = route_weights.to(
                torch.float32 if routing_fp32 else torch.bfloat16
            )
            output = torch.empty(
                m, FINALIZE_HIDDEN, device="cuda", dtype=torch.bfloat16
            )
            args = (routed_rows, route_weights, permuted_idx, norm_weight, output)

            local_ok = supports_moe_finalize_rmsnorm_allreduce(workspace, *args)
            assert workspace.check_support(local_ok), f"unsupported at M={m}"
            moe_finalize_rmsnorm_allreduce(
                *args,
                workspace,
                eps=FINALIZE_EPS,
                gain_center=gain_center,
                weight_before_norm=weight_before_norm,
            )
            torch.cuda.synchronize()

            # Each rank contributes its BF16-rounded partial sum.
            reference = _finalize_partial(
                routed_rows,
                route_weights,
                permuted_idx,
                local,
                norm_weight,
                gain_center,
                weight_before_norm,
            )
            reference = reference.bfloat16().float()
            dist.all_reduce(reference)
            reference = reference.bfloat16()

            context = f"rank {rank}, step {step}, M={m}"
            delta = (output.float() - reference.float()).abs().max().item()
            rel = _rel_l2(output, reference)
            assert rel < 2e-3, f"{context}: relative L2 error {rel:.4g}"
            # A few BF16 ULPs at output magnitudes below 8.
            assert delta <= 0.0625, f"{context}: maximum absolute error {delta:.4g}"

            # Every rank must hold the identical all-reduced output.
            gathered = [torch.empty_like(output) for _ in range(world_size)]
            dist.all_gather(gathered, output)
            for other, peer_output in enumerate(gathered):
                assert torch.equal(peer_output, output), (
                    f"{context}: output differs from rank {other}"
                )
    finally:
        workspace.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("routing_fp32", [False, True])
@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_moe_finalize_rmsnorm_allreduce_accuracy(
    world_size: int, routing_fp32: bool, weight_before_norm: bool
) -> None:
    _launch(_finalize_worker, world_size, routing_fp32, weight_before_norm)
