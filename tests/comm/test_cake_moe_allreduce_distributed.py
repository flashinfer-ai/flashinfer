"""Distributed correctness coverage for the SM100 Cake MoE backend."""

from __future__ import annotations

import multiprocessing as mp
import socket
import time
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.jit import cake_moe_comm


HIDDEN_SIZE = 7168
MAX_TOKEN_NUM = 2048
ACTIVE_EXPERTS = 8
TOP_K = 8
ATOL = 1e-2
RTOL = 1e-2
WORKER_TIMEOUT_SECONDS = 20 * 60


def _get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
            return int(sock.getsockname()[1])


def _bounded_rand(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    scale: float = 0.25,
) -> torch.Tensor:
    values = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    return ((values - 0.5) * (2.0 * scale)).to(dtype).contiguous()


def _assert_distributed_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    group: dist.ProcessGroup,
) -> None:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    close = torch.isclose(
        actual_f32,
        expected_f32,
        atol=ATOL,
        rtol=RTOL,
        equal_nan=False,
    ).all()
    failure = (~close).to(torch.int32)
    max_abs = torch.nan_to_num(
        (actual_f32 - expected_f32).abs(), nan=float("inf")
    ).max()
    dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
    if failure.item():
        raise AssertionError(
            f"{label} failed distributed close check: "
            f"max_abs={max_abs.item():.6g}, atol={ATOL}, rtol={RTOL}"
        )


def _run_mode(
    call: Callable[[], None],
    mode: str,
    group: dist.ProcessGroup,
    *,
    replay_count: int = 1,
    start_delay_seconds: float = 0.0,
) -> None:
    dist.barrier(group=group)
    if start_delay_seconds:
        time.sleep(start_delay_seconds)
    call()
    torch.cuda.synchronize()
    if mode == "eager":
        for _ in range(replay_count):
            call()
    elif mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(replay_count):
            graph.replay()
    else:  # pragma: no cover - the test matrix is fixed below.
        raise AssertionError(f"unknown execution mode {mode!r}")
    torch.cuda.synchronize()


def _rank_order_state_allreduce(
    local: torch.Tensor,
    dtype: torch.dtype,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    locals_by_rank = [
        torch.empty_like(local) for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(locals_by_rank, local, group=group)
    reduced = locals_by_rank[0]
    for peer_local in locals_by_rank[1:]:
        reduced = (reduced.float() + peer_local.float()).to(dtype)
    return reduced


def _reduction_worker(
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    distributed_init_port: int,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    ipc_handles = None
    try:
        cake_moe_comm.load(rank)
        dist.barrier(group=group)
        ipc_handles, workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                MAX_TOKEN_NUM,
                HIDDEN_SIZE,
                group=group,
            )
        )

        for token_num in (1, 64, 2048):
            generator = torch.Generator(device=device).manual_seed(
                0xCA4E0000 + world_size * 10000 + rank * 100 + token_num
            )
            expert_input = _bounded_rand(
                (ACTIVE_EXPERTS, token_num, HIDDEN_SIZE),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            expert_scale = _bounded_rand(
                (ACTIVE_EXPERTS, token_num),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            token_input = _bounded_rand(
                (token_num, HIDDEN_SIZE),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            residual_in = _bounded_rand(
                (token_num, HIDDEN_SIZE),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            rms_gamma = (
                _bounded_rand(
                    (HIDDEN_SIZE,),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                    scale=0.125,
                )
                + 1
            ).contiguous()
            rms_eps = 1e-5

            local = torch.zeros_like(token_input)
            for expert in range(ACTIVE_EXPERTS):
                contribution = (
                    expert_input[expert].float()
                    * expert_scale[expert].float().unsqueeze(-1)
                ).to(dtype)
                local = (local.float() + contribution.float()).to(dtype)
            local = (local.float() + token_input.float()).to(dtype)
            allreduce_ref = _rank_order_state_allreduce(local, dtype, group)
            allreduce_ref_f32 = allreduce_ref.float()
            residual_ref = (allreduce_ref_f32 + residual_in.float()).to(dtype)
            residual_ref_f32 = residual_ref.float()
            norm_ref = (
                residual_ref_f32
                * torch.rsqrt(
                    residual_ref_f32.square().mean(dim=-1, keepdim=True)
                    + rms_eps
                )
                * rms_gamma.float()
            ).to(dtype)

            emit_allreduce_options = (True, False) if token_num == 64 else (True,)
            for emit_allreduce in emit_allreduce_options:
                for launch_with_pdl in (False, True):
                    for mode in ("eager", "graph"):
                        moe_allreduce_out = (
                            torch.empty_like(residual_in) if emit_allreduce else None
                        )
                        residual_out = torch.empty_like(residual_in)
                        norm_out = torch.empty_like(residual_in)

                        def call() -> None:
                            comm.trtllm_moe_reduction_allreduce_fusion(
                                world_size=world_size,
                                world_rank=rank,
                                token_num=token_num,
                                hidden_dim=HIDDEN_SIZE,
                                workspace_ptrs=workspace_tensor,
                                launch_with_pdl=launch_with_pdl,
                                residual_in=residual_in,
                                rms_gamma=rms_gamma,
                                rms_eps=rms_eps,
                                scale_factor=1.0,
                                moe_reduction_device_num_experts=ACTIVE_EXPERTS,
                                moe_reduction_scale_input=expert_scale,
                                moe_reduction_active_experts_token_input=expert_input,
                                moe_reduction_token_input=token_input,
                                layout_code=None,
                                moe_allreduce_out=moe_allreduce_out,
                                residual_out=residual_out,
                                norm_out=norm_out,
                                quant_out=None,
                                scale_out=None,
                                backend="cake",
                            )

                        case = (
                            f"reduction/tp{world_size}/{dtype}/tokens{token_num}/"
                            f"pdl{int(launch_with_pdl)}/{mode}/"
                            f"allreduce_out{int(emit_allreduce)}"
                        )
                        _run_mode(call, mode, group)
                        if moe_allreduce_out is not None:
                            _assert_distributed_close(
                                moe_allreduce_out,
                                allreduce_ref_f32,
                                label=f"{case}/moe_allreduce_out",
                                group=group,
                            )
                        _assert_distributed_close(
                            residual_out,
                            residual_ref,
                            label=f"{case}/residual_out",
                            group=group,
                        )
                        _assert_distributed_close(
                            norm_out,
                            norm_ref,
                            label=f"{case}/norm_out",
                            group=group,
                        )

        dist.barrier(group=group)
    finally:
        if ipc_handles is not None:
            comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                ipc_handles, group=group
            )
        if dist.is_initialized():
            dist.destroy_process_group(group=group)


def _finalize_worker(
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    distributed_init_port: int,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    ipc_handles = None
    try:
        cake_moe_comm.load(rank)
        dist.barrier(group=group)
        ipc_handles, workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                MAX_TOKEN_NUM,
                HIDDEN_SIZE,
                group=group,
            )
        )

        for token_num in (1, 64, 2048):
            generator = torch.Generator(device=device).manual_seed(
                0xCA4F0000 + world_size * 10000 + rank * 100 + token_num
            )
            allreduce_in = _bounded_rand(
                (token_num * TOP_K, HIDDEN_SIZE),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            residual_in = _bounded_rand(
                (token_num, HIDDEN_SIZE),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            norm_weight = (
                _bounded_rand(
                    (HIDDEN_SIZE,),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                    scale=0.125,
                )
                + 1
            ).contiguous()
            expert_scale = _bounded_rand(
                (token_num, TOP_K),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            inverse_indices = torch.arange(
                token_num * TOP_K, dtype=torch.int32, device=device
            ).reshape(token_num, TOP_K)
            shared_expert = _bounded_rand(
                (token_num, HIDDEN_SIZE),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            eps = 1e-5

            optional_cases = (
                ((shared_expert, 2.5), (None, None))
                if token_num == 64
                else ((shared_expert, 2.5),)
            )
            for shared_expert_output, routed_scaling_factor in optional_cases:
                routed = (
                    1.0
                    if routed_scaling_factor is None
                    else routed_scaling_factor
                )
                gathered = allreduce_in[inverse_indices]
                local = torch.zeros_like(residual_in)
                for route in range(TOP_K):
                    contribution = (
                        gathered[:, route].float()
                        * expert_scale[:, route].float().unsqueeze(-1)
                    ).to(dtype)
                    local = (local.float() + contribution.float()).to(dtype)
                local = (local.float() * routed).to(dtype)
                if shared_expert_output is not None:
                    local = (
                        local.float() + shared_expert_output.float()
                    ).to(dtype)
                finalized_ref = _rank_order_state_allreduce(
                    local, dtype, group
                )
                residual_ref = (
                    finalized_ref.float() + residual_in.float()
                ).to(dtype)
                residual_ref_f32 = residual_ref.float()
                norm_ref = (
                    residual_ref_f32
                    * torch.rsqrt(
                        residual_ref_f32.square().mean(dim=-1, keepdim=True)
                        + eps
                    )
                    * norm_weight.float()
                ).to(dtype)

                for launch_with_pdl in (False, True):
                    for mode in ("eager", "graph"):
                        residual_out = torch.empty_like(residual_in)
                        norm_out = torch.empty_like(residual_in)

                        def call() -> None:
                            comm.trtllm_moe_finalize_allreduce_fusion(
                                allreduce_in=allreduce_in,
                                residual_in=residual_in,
                                norm_weight=norm_weight,
                                expanded_idx_to_permuted_idx=inverse_indices,
                                norm_out=norm_out,
                                residual_out=residual_out,
                                quant_out=None,
                                scale_out=None,
                                workspace_ptrs=workspace_tensor,
                                launch_with_pdl=launch_with_pdl,
                                world_rank=rank,
                                world_size=world_size,
                                eps=eps,
                                shared_expert_output=shared_expert_output,
                                expert_scale_factor=expert_scale,
                                routed_scaling_factor=routed_scaling_factor,
                                backend="cake",
                            )

                        shared_label = (
                            "none" if shared_expert_output is None else "present"
                        )
                        case = (
                            f"finalize/tp{world_size}/{dtype}/tokens{token_num}/"
                            f"pdl{int(launch_with_pdl)}/{mode}/shared_{shared_label}"
                        )
                        stress_tp4_epoch_wrap = world_size == 4 and token_num == 2048
                        _run_mode(
                            call,
                            mode,
                            group,
                            replay_count=8 if stress_tp4_epoch_wrap else 1,
                            start_delay_seconds=(
                                0.05
                                if stress_tp4_epoch_wrap and rank == world_size - 1
                                else 0.0
                            ),
                        )
                        _assert_distributed_close(
                            residual_out,
                            residual_ref,
                            label=f"{case}/residual_out",
                            group=group,
                        )
                        _assert_distributed_close(
                            norm_out,
                            norm_ref,
                            label=f"{case}/norm_out",
                            group=group,
                        )

        dist.barrier(group=group)
    finally:
        if ipc_handles is not None:
            comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                ipc_handles, group=group
            )
        if dist.is_initialized():
            dist.destroy_process_group(group=group)


def _run_distributed(
    world_size: int,
    dtype: torch.dtype,
    worker: Callable[[int, int, torch.dtype, int], None],
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"test requires {world_size} CUDA devices")
    unsupported = [
        index
        for index in range(world_size)
        if torch.cuda.get_device_capability(index) != (10, 0)
    ]
    if unsupported:
        pytest.skip(f"Cake MoE communication requires SM100 devices: {unsupported=}")

    context = mp.get_context("spawn")
    distributed_init_port = _get_open_port()
    processes = [
        context.Process(
            target=worker,
            args=(world_size, rank, dtype, distributed_init_port),
            name=f"cake-moe-rank-{rank}",
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
    for process in timed_out:
        process.join(timeout=10)
    still_alive = [process for process in timed_out if process.is_alive()]
    for process in still_alive:
        process.kill()
    for process in still_alive:
        process.join(timeout=10)
    timed_out_names = {process.name for process in timed_out}
    failures = [
        (
            process.name,
            "timeout" if process.name in timed_out_names else process.exitcode,
        )
        for process in processes
        if process.exitcode != 0
    ]
    assert not failures, f"distributed Cake MoE workers failed: {failures}"


_DISTRIBUTED_CASES = (
    (2, torch.float16),
    (2, torch.bfloat16),
    (4, torch.float16),
    (4, torch.bfloat16),
)


@pytest.mark.parametrize(
    "world_size,dtype",
    _DISTRIBUTED_CASES,
    ids=("tp2-fp16", "tp2-bf16", "tp4-fp16", "tp4-bf16"),
)
def test_cake_moe_reduction_correctness(
    world_size: int, dtype: torch.dtype
) -> None:
    _run_distributed(world_size, dtype, _reduction_worker)


@pytest.mark.parametrize(
    "world_size,dtype",
    _DISTRIBUTED_CASES,
    ids=("tp2-fp16", "tp2-bf16", "tp4-fp16", "tp4-bf16"),
)
def test_cake_moe_finalize_correctness(
    world_size: int, dtype: torch.dtype
) -> None:
    _run_distributed(world_size, dtype, _finalize_worker)
