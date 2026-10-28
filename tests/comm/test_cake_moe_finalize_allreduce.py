"""Distributed correctness for the Cake indexed MoE finalize backend."""

from __future__ import annotations

import multiprocessing as mp
import socket
import time

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.jit.cake_moe_finalize_comm import SMALL_TOKEN_MAX


HIDDEN_SIZE = 7168
BACKENDS = ("trtllm", "cake")
ATOL = 1e-2
RTOL = 1e-2
FP4_ATOL = 1.0
FP4_RTOL = 0.1
PROCESS_JOIN_TIMEOUT_S = 600.0
PROCESS_CLEANUP_TIMEOUT_S = 30.0

# Program choice of the Cake binding above SMALL_TOKEN_MAX tokens: the small-token program
# still runs up to SMALL_TOKEN_EXTENDED_MAX tokens at world sizes 2 and 4 except for top_k 4
# without a shared expert (while the device holds that many small-token clusters); the
# two-shot program runs at world size 8 and at world size 4 from TWO_SHOT_MIN_TOKENS_WS4
# tokens; the large-token program runs at world size 2 for top_k 4 above
# LARGE_TOKEN_MIN_TOKENS tokens (LARGE_TOKEN_MIN_TOKENS_SHARED with a shared expert); the
# persistent program runs otherwise.
SMALL_TOKEN_EXTENDED_MAX = 71
TWO_SHOT_MIN_TOKENS_WS4 = 96
LARGE_TOKEN_MIN_TOKENS = 512
LARGE_TOKEN_MIN_TOKENS_SHARED = 1280
# SM count at which the small-token program holds SMALL_TOKEN_EXTENDED_MAX clusters (B200, B300).
SMALL_TOKEN_EXTENDED_MIN_SMS = 148
_PROGRAM_SUFFIX = {
    "small": "_small",
    "two_shot": "_ts",
    "large_token": "_lt",
    "persistent": "",
}


def _decode_fp4_output(
    quant_out: torch.Tensor,
    scale_out: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Decode E2M1 data with per-16 E4M3 scales in SWIZZLED_128x4 layout."""

    packed = quant_out.view(torch.uint8)[: rows * columns // 2].reshape(
        rows, columns // 2
    )
    decode_table = torch.tensor(
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
        dtype=torch.float32,
        device=quant_out.device,
    )
    values = torch.empty((rows, columns), dtype=torch.float32, device=quant_out.device)
    values[:, 0::2] = decode_table[(packed & 0x0F).long()]
    values[:, 1::2] = decode_table[(packed >> 4).long()]

    scale_columns = columns // 16
    padded_rows = ((rows + 127) // 128) * 128
    padded_scale_columns = ((scale_columns + 3) // 4) * 4
    scale_bytes = scale_out.view(torch.uint8)[: padded_rows * padded_scale_columns]
    logical_scale_bytes = (
        scale_bytes.reshape(
            padded_rows // 128,
            padded_scale_columns // 4,
            32,
            4,
            4,
        )
        .permute(0, 3, 2, 1, 4)
        .reshape(padded_rows, padded_scale_columns)[:rows, :scale_columns]
        .contiguous()
    )
    scales = logical_scale_bytes.view(torch.float8_e4m3fn).float()
    return (values.reshape(rows, scale_columns, 16) * scales.unsqueeze(-1)).reshape(
        rows, columns
    )


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _bounded_rand(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    return ((values - 0.5) * 0.125).to(dtype).contiguous()


def _rank_order_sum(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    peers = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, local, group=group)
    total = peers[0]
    for peer in peers[1:]:
        total = (total.float() + peer.float()).to(local.dtype)
    return total


def _reference(
    *,
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    inverse_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    shared_expert_output: torch.Tensor | None,
    routed_scaling_factor: float,
    eps: float,
    weight_bias: float,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered = allreduce_in[inverse_indices]
    local = torch.zeros_like(residual_in)
    for route in range(inverse_indices.shape[1]):
        contribution = (
            gathered[:, route].float() * expert_scales[:, route].float().unsqueeze(-1)
        ).to(local.dtype)
        local = (local.float() + contribution.float()).to(local.dtype)
    local = (local.float() * routed_scaling_factor).to(local.dtype)
    if shared_expert_output is not None:
        local = (local.float() + shared_expert_output.float()).to(local.dtype)
    reduced = _rank_order_sum(local, group)
    residual = (reduced.float() + residual_in.float()).to(local.dtype)
    residual_f32 = residual.float()
    norm = (
        residual_f32
        * torch.rsqrt(residual_f32.square().mean(dim=-1, keepdim=True) + eps)
        * (norm_weight.float() + weight_bias)
    ).to(local.dtype)
    return residual, norm


def _token_counts(world_size: int) -> tuple[int, ...]:
    """Token counts on both sides of every program boundary of the world size."""
    tokens = {
        1,
        SMALL_TOKEN_MAX,
        SMALL_TOKEN_MAX + 1,
        SMALL_TOKEN_EXTENDED_MAX,
        SMALL_TOKEN_EXTENDED_MAX + 1,
        128,
    }
    if world_size == 4:
        tokens |= {TWO_SHOT_MIN_TOKENS_WS4 - 1, TWO_SHOT_MIN_TOKENS_WS4}
    if world_size == 2:
        tokens |= {
            LARGE_TOKEN_MIN_TOKENS,
            LARGE_TOKEN_MIN_TOKENS + 1,
            LARGE_TOKEN_MIN_TOKENS_SHARED,
            LARGE_TOKEN_MIN_TOKENS_SHARED + 1,
        }
    return tuple(sorted(tokens))


def _cases(world_size: int) -> list[tuple[int, int, bool, str, float]]:
    """Every token count of _token_counts crossed with top_k, shared expert, output
    profile and weight_bias."""
    return [
        (tokens, top_k, use_shared, output_profile, weight_bias)
        for tokens in _token_counts(world_size)
        for top_k in (4, 8)
        for use_shared in (False, True)
        for output_profile in ("110", "111")
        for weight_bias in (0.0, 1.0)
    ]


def _expected_program(
    world_size: int, tokens: int, top_k: int, use_shared: bool
) -> str:
    if tokens <= SMALL_TOKEN_MAX:
        return "small"
    if (
        world_size != 8
        and (top_k != 4 or use_shared)
        and tokens <= SMALL_TOKEN_EXTENDED_MAX
    ):
        return "small"
    if world_size == 8 or (world_size == 4 and tokens >= TWO_SHOT_MIN_TOKENS_WS4):
        return "two_shot"
    large_token_min = (
        LARGE_TOKEN_MIN_TOKENS_SHARED if use_shared else LARGE_TOKEN_MIN_TOKENS
    )
    if world_size == 2 and top_k == 4 and tokens > large_token_min:
        return "large_token"
    return "persistent"


def _expected_kernel(
    world_size: int,
    tokens: int,
    top_k: int,
    use_shared: bool,
    dtype: torch.dtype,
    output_profile: str,
) -> str:
    dtype_name = "float16" if dtype == torch.float16 else "bfloat16"
    program = _expected_program(world_size, tokens, top_k, use_shared)
    return (
        f"kernel_cake_trtllm_moe_finalize_{dtype_name}_ws{world_size}"
        f"_o{output_profile}{_PROGRAM_SUFFIX[program]}"
    )


def _workspace(
    rank: int, world_size: int, group: dist.ProcessGroup
) -> tuple[list[list[int]], torch.Tensor]:
    result = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
        rank,
        world_size,
        max(_token_counts(world_size)),
        HIDDEN_SIZE,
        group=group,
    )
    assert len(result) == 2
    return result


def _run_backend(
    *,
    backend: str,
    workspace_ptrs: torch.Tensor,
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    inverse_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    shared_expert_output: torch.Tensor | None,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    output_profile: str,
    routed_scaling_factor: float | None,
    weight_bias: float,
    kernel_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(residual_in)
    quant_out = None
    scale_out = None
    if output_profile == "111":
        # Packed E2M1 pairs and E4M3 scales (SWIZZLED_128x4) are byte buffers.
        quant_out = torch.zeros(
            residual_in.numel() // 2, dtype=torch.uint8, device=residual_in.device
        )
        padded_rows = ((residual_in.shape[0] + 127) // 128) * 128
        padded_columns = ((HIDDEN_SIZE // 16 + 3) // 4) * 4
        scale_out = torch.zeros(
            padded_rows * padded_columns,
            dtype=torch.float8_e4m3fn,
            device=residual_in.device,
        )

    def launch() -> None:
        comm.trtllm_moe_finalize_allreduce_fusion(
            allreduce_in=allreduce_in,
            residual_in=residual_in,
            norm_weight=norm_weight,
            expanded_idx_to_permuted_idx=inverse_indices,
            norm_out=norm_out,
            residual_out=residual_out,
            quant_out=quant_out,
            scale_out=scale_out,
            workspace_ptrs=workspace_ptrs,
            launch_with_pdl=launch_with_pdl,
            world_rank=world_rank,
            world_size=world_size,
            eps=1e-5,
            shared_expert_output=shared_expert_output,
            expert_scale_factor=expert_scales,
            routed_scaling_factor=routed_scaling_factor,
            weight_bias=weight_bias,
            backend=backend,
        )
        torch.cuda.synchronize()

    if kernel_names is None:
        launch()
    else:
        # Record the Cake kernels this call launches (the binding's program choice).
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            launch()
        kernel_names.extend(
            event.name
            for event in profiler.events()
            if event.name.startswith("kernel_cake_trtllm_moe_finalize_")
        )
    return residual_out, norm_out, quant_out, scale_out


def _worker(world_size: int, rank: int, dtype: torch.dtype, port: int) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    cases = _cases(world_size)
    # The small-token program holds SMALL_TOKEN_EXTENDED_MAX clusters from this SM count on;
    # with fewer SMs the binding may run the persistent program there instead.
    small_token_extended = (
        torch.cuda.get_device_properties(device).multi_processor_count
        >= SMALL_TOKEN_EXTENDED_MIN_SMS
    )
    # One workspace per backend for the whole run: consecutive cases switch between
    # the Cake programs (one-shot and two-shot) on the same workspace.
    workspaces: dict[str, tuple[list[list[int]], torch.Tensor]] = {}
    try:
        for backend in BACKENDS:
            workspaces[backend] = _workspace(rank, world_size, group)
        for launch_with_pdl in (False, True):
            for case_index, case in enumerate(cases):
                tokens, top_k, use_shared, output_profile, weight_bias = case
                generator = torch.Generator(device=device).manual_seed(
                    0xCA4E0000
                    + world_size * 100000
                    + rank * 1000
                    + case_index * 2
                    + int(launch_with_pdl)
                )
                allreduce_in = _bounded_rand(
                    (tokens * top_k, HIDDEN_SIZE),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                residual_in = _bounded_rand(
                    (tokens, HIDDEN_SIZE),
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
                    )
                    + 1
                ).contiguous()
                expert_scales = _bounded_rand(
                    (tokens, top_k),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                inverse_indices = torch.arange(
                    tokens * top_k, dtype=torch.int32, device=device
                ).reshape(tokens, top_k)
                shared_expert_output = (
                    _bounded_rand(
                        (tokens, HIDDEN_SIZE),
                        dtype=dtype,
                        device=device,
                        generator=generator,
                    )
                    if use_shared
                    else None
                )
                routed_scaling_factor = 2.5 if use_shared else None
                residual_ref, norm_ref = _reference(
                    allreduce_in=allreduce_in,
                    residual_in=residual_in,
                    norm_weight=norm_weight,
                    inverse_indices=inverse_indices,
                    expert_scales=expert_scales,
                    shared_expert_output=shared_expert_output,
                    routed_scaling_factor=(
                        1.0 if routed_scaling_factor is None else routed_scaling_factor
                    ),
                    eps=1e-5,
                    weight_bias=weight_bias,
                    group=group,
                )

                outputs: dict[str, tuple[torch.Tensor, ...]] = {}
                kernel_names: list[str] | None = None
                expected_kernel = _expected_kernel(
                    world_size, tokens, top_k, use_shared, dtype, output_profile
                )
                if (
                    not launch_with_pdl
                    and weight_bias == 0.0
                    and (
                        small_token_extended
                        or tokens <= SMALL_TOKEN_MAX
                        or not expected_kernel.endswith("_small")
                    )
                ):
                    kernel_names = []
                for backend in BACKENDS:
                    result = _run_backend(
                        backend=backend,
                        workspace_ptrs=workspaces[backend][1],
                        allreduce_in=allreduce_in,
                        residual_in=residual_in,
                        norm_weight=norm_weight,
                        inverse_indices=inverse_indices,
                        expert_scales=expert_scales,
                        shared_expert_output=shared_expert_output,
                        launch_with_pdl=launch_with_pdl,
                        world_rank=rank,
                        world_size=world_size,
                        output_profile=output_profile,
                        routed_scaling_factor=routed_scaling_factor,
                        weight_bias=weight_bias,
                        kernel_names=kernel_names if backend == "cake" else None,
                    )
                    outputs[backend] = tuple(
                        tensor.clone() for tensor in result if tensor is not None
                    )

                trtllm_result = outputs["trtllm"]
                label = f"pdl={launch_with_pdl} case={case}"
                if kernel_names is not None:
                    assert kernel_names == [expected_kernel], (
                        f"cake launched {kernel_names}, expected {expected_kernel} "
                        f"({label})"
                    )
                for backend, result in outputs.items():
                    torch.testing.assert_close(
                        result[0].float(),
                        residual_ref.float(),
                        atol=ATOL,
                        rtol=RTOL,
                        msg=lambda text, backend=backend, label=label: (
                            f"{backend} residual {label}: {text}"
                        ),
                    )
                    torch.testing.assert_close(
                        result[1].float(),
                        norm_ref.float(),
                        atol=ATOL,
                        rtol=RTOL,
                        msg=lambda text, backend=backend, label=label: (
                            f"{backend} norm {label}: {text}"
                        ),
                    )
                    if output_profile == "111":
                        assert len(result) == len(trtllm_result) == 4
                        if backend == "cake":
                            decoded = _decode_fp4_output(
                                result[2],
                                result[3],
                                rows=tokens,
                                columns=HIDDEN_SIZE,
                            )
                            torch.testing.assert_close(
                                decoded,
                                norm_ref.float(),
                                atol=FP4_ATOL,
                                rtol=FP4_RTOL,
                                msg=lambda text, label=label: (
                                    f"cake decoded FP4 {label}: {text}"
                                ),
                            )
    finally:
        dist.barrier(group=group)
        for handles, _ in workspaces.values():
            comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                handles, group=group
            )
        dist.destroy_process_group(group=group)


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cake_moe_finalize_allreduce(world_size: int, dtype: torch.dtype) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"test requires {world_size} CUDA devices")
    capabilities = {
        torch.cuda.get_device_capability(index) for index in range(world_size)
    }
    if len(capabilities) != 1 or not capabilities.issubset({(10, 0), (10, 3)}):
        pytest.skip("test requires one homogeneous SM100 or SM103 node")

    port = _open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(world_size, rank, dtype, port))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + PROCESS_JOIN_TIMEOUT_S
    try:
        for rank, process in enumerate(processes):
            process.join(timeout=max(1.0, deadline - time.monotonic()))
            if process.is_alive():
                raise AssertionError(
                    f"rank {rank} did not finish within {PROCESS_JOIN_TIMEOUT_S}s"
                )
            assert process.exitcode == 0, f"rank {rank} exited with {process.exitcode}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=PROCESS_CLEANUP_TIMEOUT_S)
            if process.is_alive():
                process.kill()
                process.join(timeout=PROCESS_CLEANUP_TIMEOUT_S)
