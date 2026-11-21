import multiprocessing as mp
import socket
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.utils import get_compute_capability

from flashinfer.comm.mnnvl import TorchDistBackend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda:0"))[0] not in (9, 10, 12),
    reason="trtllm_comm kernels support SM90/SM100/SM12x only",
)


# todo(Yingyi): add benchmark and quant test

# Usage: test var
kOneShotMaxTokenNum = 128
MIN_TOKEN_NUM = 1
MAX_TOKEN_NUM = 2048
SF_VEC_SIZE = 16

# temp var
SCALE_FACTOR_RANGE = (-1, 1)


def _run_correctness_worker(
    world_size,
    rank,
    dtype,
    hidden_dim,
    distributed_init_port,
    legacy_api=True,
    weight_bias=0.0,
    gpu_offset=0,
):
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        token_nums = [1, 128, 1024, 2048]
        pattern_codes = [
            comm.AllReduceFusionPattern.kAllReduce,
            comm.AllReduceFusionPattern.kARResidualRMSNorm,
            comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        ]
        swizzled_layout_codes = [
            comm.QuantizationSFLayout.LINEAR,
            comm.QuantizationSFLayout.SWIZZLED_128x4,
            comm.QuantizationSFLayout.SWIZZLED_8x4,
        ]
        launch_with_pdls = [True, False]
        use_oneshots = [True, False, None]
        trigger_completion_at_ends = [True, False]
        fp32_accs = [True, False]

        lamport_use_fp32 = dtype == torch.float32

        # Create workspace - choose between legacy and new API
        if legacy_api:
            # Legacy API: create workspace for allreduce fusion with metadata
            ipc_handles, workspace_tensor, workspace_metadata = (
                comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                    rank,
                    world_size,
                    MAX_TOKEN_NUM,
                    hidden_dim,
                    group=group,
                    use_fp32_lamport=lamport_use_fp32,
                    create_metadata=True,  # Get metadata for validation
                )
            )
        else:
            workspace = None
            # New unified API: create workspace
            workspace = comm.create_allreduce_fusion_workspace(
                backend="trtllm",
                world_size=world_size,
                rank=rank,
                max_token_num=MAX_TOKEN_NUM,
                hidden_dim=hidden_dim,
                dtype=dtype,
                comm_backend=TorchDistBackend(),
            )

        test_loop = 5

        for token_num in token_nums:
            for pattern_code in pattern_codes:
                for swizzled_layout_code in swizzled_layout_codes:
                    for launch_with_pdl in launch_with_pdls:
                        for use_oneshot in use_oneshots:
                            for trigger_completion_at_end in trigger_completion_at_ends:
                                for fp32_acc in fp32_accs:
                                    if token_num < world_size and not use_oneshot:
                                        continue
                                    if dtype == torch.float32 and (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant
                                        or pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant
                                    ):
                                        continue

                                    dist.barrier(group=group)
                                    test_passed = True
                                    print(
                                        f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl}-isOneShot{use_oneshot}-isTriggerCompletionAtEnd{trigger_completion_at_end}-isFP32Acc{fp32_acc} start"
                                    )
                                    dist.barrier(group=group)
                                    torch.cuda.synchronize()

                                    message_size = token_num * hidden_dim

                                    allreduce_in = torch.randn(
                                        message_size, dtype=dtype, device=device
                                    )
                                    allreduce_in_clone = allreduce_in.clone()

                                    all_reduce_out = torch.zeros(
                                        message_size, dtype=dtype, device=device
                                    )

                                    residual_in = torch.randn(
                                        message_size, dtype=dtype, device=device
                                    )
                                    residual_in_clone = residual_in.clone()

                                    residual_out = torch.empty_like(residual_in)
                                    norm_out = torch.empty_like(residual_in)
                                    quant_out = torch.empty(
                                        message_size, dtype=dtype, device=device
                                    )

                                    scale_out = None
                                    assert hidden_dim % SF_VEC_SIZE == 0, (
                                        "hidden_dim must be divisible by SF_VEC_SIZE"
                                    )
                                    if (
                                        swizzled_layout_code
                                        == comm.QuantizationSFLayout.SWIZZLED_128x4
                                    ):
                                        # TODO(Yingyi): check this
                                        padded_message_size = (
                                            (token_num + 127) // 128 * 128
                                        ) * ((hidden_dim + 63) // 64 * 4)
                                        scale_out = torch.empty(
                                            padded_message_size,
                                            dtype=dtype,
                                            device=device,
                                        )
                                    else:
                                        scale_out = torch.empty(
                                            message_size // SF_VEC_SIZE,
                                            dtype=dtype,
                                            device=device,
                                        )

                                    rms_gamma = torch.randn(
                                        hidden_dim, dtype=dtype, device=device
                                    )
                                    scale_factor = (
                                        torch.rand(
                                            1, dtype=torch.float32, device=device
                                        )
                                        * (
                                            SCALE_FACTOR_RANGE[1]
                                            - SCALE_FACTOR_RANGE[0]
                                        )
                                        + SCALE_FACTOR_RANGE[0]
                                    )
                                    rms_eps = 1e-3

                                    # warmup
                                    s = torch.cuda.Stream()
                                    s.wait_stream(torch.cuda.current_stream())
                                    with torch.cuda.stream(s):
                                        for _ in range(test_loop):
                                            if legacy_api:
                                                # Legacy API - uses flattened tensors
                                                comm.trtllm_allreduce_fusion(
                                                    allreduce_in=allreduce_in,
                                                    world_size=world_size,
                                                    world_rank=rank,
                                                    token_num=token_num,
                                                    hidden_dim=hidden_dim,
                                                    workspace_ptrs=workspace_tensor,
                                                    launch_with_pdl=launch_with_pdl,
                                                    use_oneshot=use_oneshot,
                                                    trigger_completion_at_end=trigger_completion_at_end,
                                                    fp32_acc=fp32_acc,
                                                    pattern_code=pattern_code,
                                                    allreduce_out=all_reduce_out,
                                                    residual_in=residual_in,
                                                    residual_out=residual_out,
                                                    norm_out=norm_out,
                                                    quant_out=quant_out,
                                                    scale_out=scale_out,
                                                    rms_gamma=rms_gamma,
                                                    rms_eps=rms_eps,
                                                    weight_bias=weight_bias,
                                                    scale_factor=scale_factor,
                                                    layout_code=swizzled_layout_code,
                                                    metadata=workspace_metadata,
                                                )
                                            else:
                                                # New unified API - expects 2D tensors [token_num, hidden_dim]
                                                comm.allreduce_fusion(
                                                    input=allreduce_in.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    workspace=workspace,
                                                    launch_with_pdl=launch_with_pdl,
                                                    output=all_reduce_out.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    residual_in=residual_in.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    residual_out=residual_out.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    norm_out=norm_out.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    quant_out=quant_out.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    scale_out=scale_out,
                                                    rms_gamma=rms_gamma,
                                                    rms_eps=rms_eps,
                                                    weight_bias=weight_bias,
                                                    scale_factor=scale_factor,
                                                    layout_code=swizzled_layout_code,
                                                    pattern=pattern_code,
                                                    use_oneshot=use_oneshot,
                                                    fp32_acc=fp32_acc,
                                                )

                                    # NOTE: in real case, you dont have to set all optional params. You could set those required by fusion pattern.
                                    # capture
                                    g = torch.cuda.CUDAGraph()
                                    with torch.cuda.graph(g):
                                        for _ in range(test_loop):
                                            if legacy_api:
                                                # Legacy API - uses flattened tensors
                                                comm.trtllm_allreduce_fusion(
                                                    allreduce_in=allreduce_in,
                                                    world_size=world_size,
                                                    world_rank=rank,
                                                    token_num=token_num,
                                                    hidden_dim=hidden_dim,
                                                    workspace_ptrs=workspace_tensor,
                                                    launch_with_pdl=launch_with_pdl,
                                                    use_oneshot=use_oneshot,
                                                    trigger_completion_at_end=trigger_completion_at_end,
                                                    fp32_acc=fp32_acc,
                                                    pattern_code=pattern_code,
                                                    allreduce_out=all_reduce_out,
                                                    residual_in=residual_in,
                                                    residual_out=residual_out,
                                                    norm_out=norm_out,
                                                    quant_out=quant_out,
                                                    scale_out=scale_out,
                                                    rms_gamma=rms_gamma,
                                                    rms_eps=rms_eps,
                                                    weight_bias=weight_bias,
                                                    scale_factor=scale_factor,
                                                    layout_code=swizzled_layout_code,
                                                    metadata=workspace_metadata,
                                                )
                                            else:
                                                # New unified API - expects 2D tensors [token_num, hidden_dim]
                                                comm.allreduce_fusion(
                                                    input=allreduce_in.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    workspace=workspace,
                                                    launch_with_pdl=launch_with_pdl,
                                                    output=all_reduce_out.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    residual_in=residual_in.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    residual_out=residual_out.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    norm_out=norm_out.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    quant_out=quant_out.view(
                                                        token_num, hidden_dim
                                                    ),
                                                    scale_out=scale_out,
                                                    rms_gamma=rms_gamma,
                                                    rms_eps=rms_eps,
                                                    weight_bias=weight_bias,
                                                    scale_factor=scale_factor,
                                                    layout_code=swizzled_layout_code,
                                                    pattern=pattern_code,
                                                    use_oneshot=use_oneshot,
                                                    fp32_acc=fp32_acc,
                                                )
                                    # replay
                                    g.replay()
                                    torch.cuda.synchronize()

                                    # match shape
                                    all_reduce_out = all_reduce_out.view(
                                        token_num, hidden_dim
                                    )
                                    residual_out = residual_out.view(
                                        token_num, hidden_dim
                                    )
                                    norm_out = norm_out.view(token_num, hidden_dim)

                                    torch.cuda.synchronize()

                                    # calculate reference
                                    # allreduce_out
                                    dist.all_reduce(allreduce_in_clone, group=group)
                                    ref_allreduce_out = allreduce_in_clone.clone()
                                    ref_allreduce_out = ref_allreduce_out.view(
                                        token_num, hidden_dim
                                    ).to(torch.float32)

                                    # residual_out
                                    ref_residual_out = (
                                        ref_allreduce_out
                                        + residual_in_clone.view(
                                            token_num, hidden_dim
                                        ).to(torch.float32)
                                    )

                                    # norm_out
                                    variance = (
                                        ref_residual_out.to(torch.float32)
                                        .pow(2)
                                        .mean(dim=-1, keepdim=True)
                                    )
                                    hidden_states = ref_residual_out * torch.rsqrt(
                                        variance + rms_eps
                                    )
                                    ref_norm_out = (
                                        weight_bias + rms_gamma.to(torch.float32)
                                    ) * hidden_states

                                    # check correctness
                                    tolerance = 8e-2 if dtype == torch.float16 else 8e-1
                                    # compare allreduce_out
                                    if (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kAllReduce
                                    ):
                                        torch.testing.assert_close(
                                            all_reduce_out.to(torch.float32),
                                            ref_allreduce_out,
                                            atol=tolerance,
                                            rtol=1e-2,
                                        )
                                    elif (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant
                                        or pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant
                                    ):
                                        torch.testing.assert_close(
                                            residual_out.to(torch.float32),
                                            ref_residual_out,
                                            atol=tolerance,
                                            rtol=1e-2,
                                        )

                                        torch.testing.assert_close(
                                            norm_out.to(torch.float32),
                                            ref_norm_out,
                                            atol=tolerance,
                                            rtol=1e-2,
                                        )

                                        # todo(Yingyi): check quant out
                                    dist.barrier(group=group)
                                    if test_passed:
                                        print(
                                            f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl}-isOneShot{use_oneshot}-isTriggerCompletionAtEnd{trigger_completion_at_end}-isFP32Acc{fp32_acc} passed"
                                        )
                                    else:
                                        print(
                                            f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl}-isOneShot{use_oneshot}-isTriggerCompletionAtEnd{trigger_completion_at_end}-isFP32Acc{fp32_acc} failed"
                                        )
    finally:
        dist.barrier(group=group)

        # Destroy workspace - choose between legacy and new API
        if legacy_api:
            comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                ipc_handles, group=group
            )
        elif workspace is not None:
            # New unified API
            workspace.destroy()

        dist.destroy_process_group(group=group)


def _run_twoshot_growing_grid_worker(
    world_size,
    rank,
    dtype,
    hidden_dim,
    distributed_init_port,
    small_token_num,
    large_token_num,
    gpu_offset=0,
):
    # Back-to-back two-shot calls whose grid grows (small -> large token count),
    # with no host synchronization in between, as in a model forward pass.
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=MAX_TOKEN_NUM,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )
        token_nums = [small_token_num, large_token_num, large_token_num] * 4
        # Every rank generates every rank's input so it can build the reference.
        inputs = []
        for i, token_num in enumerate(token_nums):
            per_rank = []
            for r in range(world_size):
                gen = torch.Generator(device=device).manual_seed(1000 * i + r)
                per_rank.append(
                    torch.randn(token_num, hidden_dim, generator=gen, device=device).to(
                        dtype
                    )
                )
            inputs.append(per_rank)
        torch.cuda.synchronize()
        dist.barrier()

        outputs = []
        for i in range(len(token_nums)):
            out = torch.empty_like(inputs[i][rank])
            comm.allreduce_fusion(
                input=inputs[i][rank],
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kAllReduce,
                output=out,
                use_oneshot=False,
            )
            outputs.append(out)
        torch.cuda.synchronize()

        for i, token_num in enumerate(token_nums):
            ref = torch.stack([x.float() for x in inputs[i]]).sum(0)
            torch.testing.assert_close(
                outputs[i].float(),
                ref,
                atol=0.1,
                rtol=3e-2,
                msg=lambda m, i=i, t=token_num: f"rank {rank} call {i} (T={t}): {m}",
            )
    finally:
        dist.barrier()
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group()


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    dtype: torch.dtype,
    hidden_dim: int,
    test_target: Any,
    target_args: tuple = (),
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (
            (
                world_size,
                i,
                dtype,
                hidden_dim,
                distributed_init_port,
            )
            + target_args
            + (gpu_offset,)
        )
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


def _run_fp32_twoshot_worker(
    world_size,
    rank,
    dtype,
    hidden_dim,
    distributed_init_port,
    max_token_num,
    gpu_offset=0,
):
    # Two-shot at the workspace's max_token_num: the kernel writes
    # 2 * token_num * hidden_dim elements into each rank's buffer.
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )
        assert workspace.is_buffer_size_sufficient(
            world_size, max_token_num, hidden_dim, dtype
        )
        inputs = [
            torch.randn(
                max_token_num,
                hidden_dim,
                generator=torch.Generator(device=device).manual_seed(r),
                device=device,
                dtype=dtype,
            )
            for r in range(world_size)
        ]
        out = torch.empty_like(inputs[rank])
        comm.allreduce_fusion(
            input=inputs[rank],
            workspace=workspace,
            pattern=comm.AllReduceFusionPattern.kAllReduce,
            output=out,
            use_oneshot=False,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, torch.stack(inputs).sum(0), atol=1e-4, rtol=1e-4
        )
    finally:
        dist.barrier()
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_trtllm_allreduce_fusion_fp32_twoshot_max_tokens(world_size):
    """fp32 two-shot at max_token_num must fit in the workspace (sized per element)."""
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    multi_process_parallel(
        world_size,
        torch.float32,
        4096,
        _run_fp32_twoshot_worker,
        target_args=(1024,),
    )


# Run as: python tests/comm/test_trtllm_allreduce_fusion.py
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_dim", [1024, 2048, 4096, 7168, 8192])
@pytest.mark.parametrize("legacy_api", [True, False])
@pytest.mark.parametrize("weight_bias", [0.0, 1.0])
def test_trtllm_allreduce_fusion(
    world_size, dtype, hidden_dim, legacy_api, weight_bias
):
    # weight_bias=0.0 -> standard RMSNorm; weight_bias=1.0 -> Gemma / Qwen3.5 RMSNorm.
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    api_str = "legacy" if legacy_api else "unified"
    print(
        f"Running test for world_size={world_size} with {api_str} API, "
        f"weight_bias={weight_bias}"
    )

    multi_process_parallel(
        world_size,
        dtype,
        hidden_dim,
        _run_correctness_worker,
        target_args=(legacy_api, weight_bias),
    )
    print(
        f"allreduce fusion tp = {world_size} ({api_str} API, "
        f"weight_bias={weight_bias}): OK"
    )


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("legacy_api", [True, False])
def test_trtllm_allreduce_fusion_gpu_offset(world_size, dtype, legacy_api):
    """Test allreduce fusion when CUDA device index != TP rank (base_gpu_id > 0).

    Simulates sglang colocate mode where inference engines run on non-zero
    base GPUs (e.g. GPUs 4-7 with TP ranks 0-3).
    See: https://github.com/flashinfer-ai/flashinfer/pull/2662
    """
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    available_gpus = torch.cuda.device_count()
    gpu_offset = available_gpus - world_size
    if gpu_offset <= 0:
        pytest.skip(
            f"Need more than {world_size} GPUs to test gpu_offset>0 "
            f"(have {available_gpus})"
        )
    api_str = "legacy" if legacy_api else "unified"
    print(
        f"Running gpu_offset test: world_size={world_size}, gpu_offset={gpu_offset}, "
        f"{api_str} API (GPUs {gpu_offset}..{gpu_offset + world_size - 1})"
    )

    multi_process_parallel(
        world_size,
        dtype,
        1024,
        _run_correctness_worker,
        # weight_bias must be passed positionally so gpu_offset (appended after
        # target_args) reaches the worker correctly.
        target_args=(legacy_api, 0.0),
        gpu_offset=gpu_offset,
    )
    print(f"gpu_offset allreduce fusion tp={world_size} ({api_str} API): OK")


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("hidden_dim", [4096, 7168])
def test_trtllm_allreduce_fusion_twoshot_growing_grid(world_size, hidden_dim):
    """Two-shot calls whose grid grows between calls must not pass the barrier early.

    Each block of the barrier also writes the flags of the CTAs that are not
    launched, so that a later launch with a larger grid sees the current flag.
    """
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    multi_process_parallel(
        world_size,
        torch.bfloat16,
        hidden_dim,
        _run_twoshot_growing_grid_worker,
        target_args=(world_size + 1, 256),
    )


# ---------------------------------------------------------------------------
# The specialized backend of allreduce_fusion:
# create_allreduce_fusion_workspace(backend="specialized"), which backend="auto" builds
# first when it can, holds the specialized two-rank kernel for kARResidualRMSNorm
# (compute capability 10.3, world size 2, hidden size 6144, bf16, fp32_acc=True,
# allowlisted weight_bias and token counts). Every other call on that workspace runs its
# fallback backend, the stock backend "auto" would otherwise pick; backend="trtllm" and
# backend="mnnvl" never run the specialized kernel.
# ---------------------------------------------------------------------------

_WS2_HIDDEN = 6144
# The workspace vLLM creates for this hidden size (64 MiB of bf16 per rank).
_WS2_MAX_TOKENS = 5461
_WS2_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
_WS2_EPS = 1e-6
# Odd sizes, the vLLM one-shot / two-shot switch (2730 / 2731), served sizes and the
# capacity; the edges of the allowlisted ranges are added from the allowlist.
_WS2_BASE_TOKEN_NUMS = [1, 7, 513, 2730, 2731, 3706, 4097, 5373, 5445, 5461]
_WS2_ALIASINGS = ["separate", "residual_out_is_input", "norm_out_is_input"]
# Token counts a served two-rank prefill worker sends to the same workspace below the
# kernel's range (its CUDA-graph sizes): they must run exactly the fallback backend.
_WS2_OUTSIDE_TOKEN_NUMS = [1, 4, 16, 64, 128, 256, 512]


def _ws2_stats():
    from flashinfer.comm._allreduce_fusion_ws2_specialized import (
        _allreduce_fusion_ws2_specialized_stats,
    )

    return _allreduce_fusion_ws2_specialized_stats()


def _ws2_ranges():
    """The allowlisted inclusive ``[min, max]`` token-count ranges."""
    return [tuple(r) for r in _ws2_stats()["allowlist_num_tokens"]]


def _ws2_allowed(token_num):
    return any(lo <= token_num <= hi for lo, hi in _ws2_ranges())


def _ws2_token_nums():
    """Base sizes plus both sides of every guard edge, up to the workspace size."""
    nums = set(_WS2_BASE_TOKEN_NUMS)
    for lo, hi in _ws2_ranges():
        nums.update(
            t
            for t in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1)
            if 1 <= t <= _WS2_MAX_TOKENS
        )
    return sorted(nums)


def _ws2_inside(frac):
    """A dispatched token count at fraction ``frac`` of the first range."""
    lo, hi = _ws2_ranges()[0]
    hi = min(hi, _WS2_MAX_TOKENS)
    return lo + int((hi - lo) * frac)


def _ws2_outside():
    """A token count outside every range (the one just below the first), or None."""
    for t in [_ws2_ranges()[0][0] - 1, *range(_WS2_MAX_TOKENS, 0, -1)]:
        if t >= 1 and not _ws2_allowed(t):
            return t
    return None


def _ws2_is_target(device_index: int = 0) -> bool:
    from flashinfer.utils import get_compute_capability

    return torch.cuda.is_available() and get_compute_capability(
        torch.device(f"cuda:{device_index}")
    ) == (10, 3)


def _ws2_all_target(device_indices) -> bool:
    return all(_ws2_is_target(i) for i in device_indices)


def _ws2_skip_unless_target(world_size: int = 2, gpu_offset: int = 0) -> None:
    """Skip unless every GPU the workers bind (``cuda:gpu_offset`` ..
    ``cuda:gpu_offset + world_size - 1``) exists and has compute capability 10.3."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.device_count() < gpu_offset + world_size:
        pytest.skip(
            f"needs GPUs {gpu_offset}..{gpu_offset + world_size - 1}, have "
            f"{torch.cuda.device_count()}"
        )
    if not _ws2_all_target(range(gpu_offset, gpu_offset + world_size)):
        pytest.skip("the specialized two-rank kernel targets compute capability 10.3")


def _ws2_launch(
    target,
    target_args=(),
    world_size=2,
    dtype=torch.bfloat16,
    hidden_dim=_WS2_HIDDEN,
    gpu_offset=0,
    timeout_s=600,
):
    """Run ``target`` on ``world_size`` spawned ranks with the arguments of
    :func:`multi_process_parallel`. Fails as soon as one rank fails, after terminating its
    peers (a rank that fails while its peer waits in a collective or a kernel must not hang
    the test), and fails when the ranks are not done within ``timeout_s`` (a hang)."""
    import time

    ctx = mp.get_context("spawn")
    port = get_open_port()
    procs = [
        ctx.Process(
            target=target,
            args=(world_size, rank, dtype, hidden_dim, port)
            + tuple(target_args)
            + (gpu_offset,),
            name=f"ws2-rank{rank}",
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + timeout_s
    failure = None
    try:
        while failure is None:
            codes = [proc.exitcode for proc in procs]
            failed = [(rank, code) for rank, code in enumerate(codes) if code]
            if failed:
                failure = f"rank {failed[0][0]} failed (exit code {failed[0][1]})"
            elif all(code == 0 for code in codes):
                return
            elif time.monotonic() > deadline:
                running = [rank for rank, code in enumerate(codes) if code is None]
                failure = f"ranks {running} still running after {timeout_s} s (hang)"
            else:
                time.sleep(0.2)
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(30)
            if proc.is_alive():
                proc.kill()
                proc.join(30)
    pytest.fail(failure)


def _ws2_setup(world_size, rank, distributed_init_port, gpu_offset):
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return device


def _ws2_workspace(
    world_size,
    rank,
    backend="specialized",
    hidden_dim=_WS2_HIDDEN,
    dtype=torch.bfloat16,
    max_token_num=None,
):
    return comm.create_allreduce_fusion_workspace(
        backend=backend,
        world_size=world_size,
        rank=rank,
        max_token_num=_WS2_MAX_TOKENS if max_token_num is None else max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        comm_backend=TorchDistBackend(),
    )


def _ws2_stock_backend(world_size):
    """The backend ``auto`` picked before the specialized backend existed: ``mnnvl`` when
    every rank supports NVLink multicast, else ``trtllm`` (collective)."""
    from flashinfer.comm.mnnvl import is_multicast_supported

    flags = [None] * world_size
    dist.all_gather_object(flags, is_multicast_supported(torch.cuda.current_device()))
    return "mnnvl" if all(flags) else "trtllm"


def _ws2_state(workspace):
    return getattr(workspace, "_ar_ws2", None)


def _ws2_open(world_size, rank, backend):
    """``(workspace, trtllm, fallback)``: the workspace under test (``backend`` is
    ``specialized``, or ``auto``, which must pick it here), a TRT-LLM workspace (the stock
    numerics the specialized kernel reproduces) and a plain workspace of the fallback
    backend (the exact results of every call the kernel does not serve)."""
    workspace = _ws2_workspace(world_size, rank, backend)
    assert isinstance(workspace, comm.SpecializedAllReduceFusionWorkspace), type(
        workspace
    )
    assert workspace.backend == "specialized"
    assert workspace.has_specialized_kernels, _ws2_stats()["last_state_rejection"]
    assert workspace.fallback_backend == _ws2_stock_backend(world_size)
    trtllm = _ws2_workspace(world_size, rank, "trtllm")
    assert _ws2_state(trtllm) is None
    if workspace.fallback_backend == "trtllm":
        fallback = trtllm
    else:
        fallback = _ws2_workspace(world_size, rank, workspace.fallback_backend)
    assert type(fallback) is type(workspace.fallback)
    return workspace, trtllm, fallback


def _ws2_close(*workspaces):
    done = set()
    for workspace in workspaces:
        if workspace is not None and id(workspace) not in done:
            done.add(id(workspace))
            workspace.destroy()


def _ws2_inputs(
    token_num, rank, seed, device, hidden_dim=_WS2_HIDDEN, dtype=torch.bfloat16
):
    """Rank-specific ``x``; ``residual`` and ``gamma`` identical on every rank (as in TP).
    Row 0 holds -0 / +0 / subnormal / large values, row 2 is all zero on every rank."""
    gen_rank = torch.Generator(device=device).manual_seed(100003 * seed + 7 * rank + 1)
    gen_shared = torch.Generator(device=device).manual_seed(100003 * seed + 99991)
    x = torch.randn(token_num, hidden_dim, generator=gen_rank, device=device).to(dtype)
    residual = torch.randn(
        token_num, hidden_dim, generator=gen_shared, device=device
    ).to(dtype)
    gamma = (0.1 * torch.randn(hidden_dim, generator=gen_shared, device=device)).to(
        dtype
    )
    x[0, :256] = -0.0
    x[0, 256:512] = 0.0
    x[0, 512:520] = 1e-39 if dtype == torch.bfloat16 else 1e-7
    x[0, 520:528] = (3e4 if dtype == torch.bfloat16 else 3e2) * (rank + 1)
    if token_num > 2:
        x[2] = 0.0
        residual[2] = 0.0
    return x, residual, gamma


def _ws2_ordered(t):
    """bf16/fp16 bit patterns mapped to integers whose difference counts ulps (+0 == -0)."""
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def _ws2_assert_match(got, ref, what, exact_min, ulp1_min):
    assert got.shape == ref.shape, (
        f"{what}: shape {tuple(got.shape)} != {tuple(ref.shape)}"
    )
    assert not torch.isnan(got).any().item(), f"{what}: NaN in output"
    diff = (_ws2_ordered(got) - _ws2_ordered(ref)).abs()
    n = diff.numel()
    exact = (diff == 0).sum().item() / n
    within1 = (diff <= 1).sum().item() / n
    assert exact >= exact_min and within1 >= ulp1_min, (
        f"{what}: bit-exact {exact:.6f} (min {exact_min}), within 1 ulp {within1:.6f} "
        f"(min {ulp1_min}), max ulp {diff.max().item()}"
    )
    g, r = got.float(), ref.float()
    torch.testing.assert_close(
        g, r, atol=1e-3, rtol=1.6e-2, msg=lambda m: f"{what}: {m}"
    )
    # Per-row scale: no row deviates by more than ~1 ulp of its largest value.
    row_err = (g - r).abs().amax(dim=-1)
    row_scale = r.abs().amax(dim=-1)
    assert bool((row_err <= row_scale * 2.0**-7 + 1e-6).all()), f"{what}: row scale"
    cos = torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim=0).item()
    assert cos >= 0.999 or r.abs().max().item() == 0, f"{what}: cosine {cos}"


def _ws2_assert_bitwise(got, ref, what):
    """The same bits: the call ran exactly the reference's kernels."""
    for g, r in zip(got, ref, strict=True):
        assert g.shape == r.shape and g.dtype == r.dtype, what
        assert torch.equal(g.view(torch.uint8), r.view(torch.uint8)), (
            f"{what}: differs from the fallback backend's own result"
        )


def _ws2_assert_close_loose(got, ref, what):
    """A stock backend's result against the torch reference (stock rounding may differ)."""
    torch.testing.assert_close(
        got.float(), ref.float(), atol=0.1, rtol=3e-2, msg=lambda m: f"{what}: {m}"
    )


def _ws2_assert_rank_identical(tensors, what):
    for t in tensors:
        gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, t.contiguous())
        for r in range(1, len(gathered)):
            assert torch.equal(
                gathered[0].view(torch.int16), gathered[r].view(torch.int16)
            ), f"{what}: rank 0 and rank {r} outputs differ"


def _ws2_torch_reference(x_all, residual, gamma, weight_bias):
    """fp32 sum -> bf16, + residual -> bf16, RMSNorm in fp32 -> bf16."""
    s = torch.stack([x.float() for x in x_all]).sum(0).to(residual.dtype)
    h = (s.float() + residual.float()).to(residual.dtype)
    hf = h.float()
    rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + _WS2_EPS)
    return s, h, (hf * rstd * (weight_bias + gamma.float())).to(residual.dtype)


def _ws2_alias(aliasing, x, residual):
    """``(residual_out, norm_out)`` for an aliasing form; separate outputs are NaN-filled."""
    if aliasing == "separate":
        return torch.full_like(x, float("nan")), torch.full_like(x, float("nan"))
    if aliasing == "residual_out_is_input":  # vLLM 0.27 MiniMax-M3
        return x, torch.full_like(x, float("nan"))
    if aliasing == "norm_out_is_input":
        return residual, x
    raise ValueError(aliasing)


def _ws2_call(
    workspace,
    x,
    residual,
    gamma,
    residual_out,
    norm_out,
    weight_bias=1.0,
    use_oneshot=None,
    launch_with_pdl=True,
    fp32_acc=True,
):
    return comm.allreduce_fusion(
        input=x,
        workspace=workspace,
        pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=launch_with_pdl,
        trigger_completion_at_end=True,
        residual_out=residual_out,
        norm_out=norm_out,
        residual_in=residual,
        rms_gamma=gamma,
        rms_eps=_WS2_EPS,
        use_oneshot=use_oneshot,
        fp32_acc=fp32_acc,
        weight_bias=weight_bias,
    )


def _ws2_vllm_oneshot(token_num):
    return token_num * _WS2_HIDDEN * 2 <= 32 * 1024 * 1024


def _ws2_expect_dispatch(workspace, token_num, weight_bias=1.0, fp32_acc=True):
    state = _ws2_state(workspace)
    stats = _ws2_stats()
    return (
        state is not None
        and fp32_acc is True
        and weight_bias in stats["allowlist_weight_bias"]
        and token_num <= state.capacity
        and any(lo <= token_num <= hi for lo, hi in stats["allowlist_num_tokens"])
    )


def _ws2_run(workspace, aliasing, x, residual, gamma, disable=False, **kwargs):
    """One call on copies of the inputs (``disable``: with the kill switch). Returns
    ``((residual_out, norm_out), specialized launches)``."""
    import os

    xc, rc = x.clone(), residual.clone()
    residual_out, norm_out = _ws2_alias(aliasing, xc, rc)
    before = _ws2_stats()["dispatch_count"]
    if disable:
        os.environ[_WS2_DISABLE_ENV] = "1"
    try:
        ret = _ws2_call(workspace, xc, rc, gamma, residual_out, norm_out, **kwargs)
    finally:
        os.environ.pop(_WS2_DISABLE_ENV, None)
    assert ret is norm_out
    torch.cuda.synchronize()
    return (residual_out, norm_out), _ws2_stats()["dispatch_count"] - before


def _run_ws2_correctness_worker(
    world_size,
    rank,
    dtype,
    hidden_dim,
    distributed_init_port,
    backend,
    aliasing,
    token_nums,
    gpu_offset=0,
):
    """Every call on the specialized (or auto) workspace: a dispatched call matches the
    TRT-LLM kernels, every other call and every kill-switched call gives exactly the
    fallback backend's result; outputs are bitwise identical across ranks."""
    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    workspace, trtllm, fallback = _ws2_open(world_size, rank, backend)
    oneshots = [None, True, False]
    case = 0
    for token_num in token_nums:
        for weight_bias in (1.0, 0.0):
            for launch_with_pdl in (True, False):
                case += 1
                use_oneshot = oneshots[case % 3]
                if token_num <= world_size and use_oneshot is False:
                    use_oneshot = True
                x, residual, gamma = _ws2_inputs(token_num, rank, case, device)
                what = (
                    f"rank {rank} {backend} T={token_num} wb={weight_bias} "
                    f"pdl={launch_with_pdl} use_oneshot={use_oneshot} {aliasing}"
                )
                kwargs = dict(
                    weight_bias=weight_bias,
                    use_oneshot=use_oneshot,
                    launch_with_pdl=launch_with_pdl,
                )
                run = lambda ws, disable=False: _ws2_run(  # noqa: E731
                    ws, aliasing, x, residual, gamma, disable=disable, **kwargs
                )
                got, dispatched = run(workspace)
                expected = _ws2_expect_dispatch(workspace, token_num, weight_bias)
                assert dispatched == int(expected), f"{what}: dispatched {dispatched}"
                killed, killed_dispatched = run(workspace, disable=True)
                assert killed_dispatched == 0, f"{what}: kill switch did not hold"
                stock, _ = run(fallback)
                _ws2_assert_bitwise(killed, stock, f"{what} kill switch")
                if expected:
                    ref, _ = run(trtllm)
                    _ws2_assert_match(
                        got[0], ref[0], f"{what} residual_out", 0.995, 0.9999
                    )
                    _ws2_assert_match(got[1], ref[1], f"{what} norm_out", 0.95, 0.999)
                else:
                    _ws2_assert_bitwise(got, stock, f"{what} outside the envelope")
                _ws2_assert_rank_identical(got, what)
    stats = _ws2_stats()
    assert stats["guard_error"] is None, stats["guard_error"]
    assert stats["dispatch_count"] > 0
    dist.barrier()
    _ws2_close(workspace, trtllm, fallback)
    dist.destroy_process_group()


def _run_ws2_back_to_back_worker(
    world_size, rank, dtype, hidden_dim, distributed_init_port, num_calls, gpu_offset=0
):
    """Serving pattern: one workspace, many calls back to back without synchronisation, the
    token count changing every call, mixing specialized calls (two aliasing forms) with
    fallback one-shot / two-shot pattern-1 calls and pattern-0 calls; fresh inputs every
    call."""
    import random

    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    workspace = _ws2_workspace(world_size, rank, "specialized")
    state = _ws2_state(workspace)
    assert state is not None, _ws2_stats()["last_state_rejection"]
    rng = random.Random(20261112)  # same schedule on both ranks
    inside = [
        (lo, min(hi, state.capacity))
        for lo, hi in _ws2_ranges()
        if lo <= state.capacity
    ]
    outside = [t for t in range(1, _WS2_MAX_TOKENS + 1) if not _ws2_allowed(t)]
    kinds = ["spec", "spec_alias", "allreduce"]
    if [t for t in outside if t > world_size]:
        kinds += ["stock_oneshot", "stock_twoshot"]
    calls = []
    expected_dispatch = 0
    before = _ws2_stats()["dispatch_count"]
    for i in range(num_calls):
        kind = kinds[i % len(kinds)] if i < len(kinds) else rng.choice(kinds)
        if kind.startswith("spec"):
            token_num = rng.randint(*rng.choice(inside))
        elif kind == "stock_oneshot":
            token_num = rng.choice(outside)
        elif kind == "stock_twoshot":
            token_num = rng.choice([t for t in outside if t > world_size])
        else:
            token_num = rng.randint(1, _WS2_MAX_TOKENS)
        x, residual, gamma = _ws2_inputs(token_num, rank, 1000 + i, device)
        x_copy, residual_copy = x.clone(), residual.clone()
        if kind == "allreduce":
            out = torch.full_like(x, float("nan"))
            comm.allreduce_fusion(
                input=x,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kAllReduce,
                launch_with_pdl=True,
                output=out,
                use_oneshot=_ws2_vllm_oneshot(token_num) or token_num <= world_size,
                fp32_acc=True,
            )
            calls.append((kind, token_num, x_copy, residual_copy, gamma, (out,)))
            continue
        aliasing = "residual_out_is_input" if kind == "spec_alias" else "separate"
        residual_out, norm_out = _ws2_alias(aliasing, x, residual)
        use_oneshot = {
            "stock_oneshot": True,
            "stock_twoshot": False,
        }.get(kind, _ws2_vllm_oneshot(token_num))
        _ws2_call(
            workspace,
            x,
            residual,
            gamma,
            residual_out,
            norm_out,
            use_oneshot=use_oneshot,
            launch_with_pdl=bool(i % 2),
        )
        expected_dispatch += int(_ws2_expect_dispatch(workspace, token_num))
        calls.append(
            (
                kind,
                token_num,
                x_copy,
                residual_copy,
                gamma,
                (residual_out, norm_out),
            )
        )
    torch.cuda.synchronize()
    assert _ws2_stats()["dispatch_count"] - before == expected_dispatch
    assert expected_dispatch >= 10

    for i, (kind, token_num, x_copy, residual_copy, gamma, outs) in enumerate(calls):
        what = f"rank {rank} call {i} ({kind}, T={token_num})"
        x_all = [torch.empty_like(x_copy) for _ in range(world_size)]
        dist.all_gather(x_all, x_copy)
        s, h, norm = _ws2_torch_reference(x_all, residual_copy, gamma, 1.0)
        if kind == "allreduce":
            _ws2_assert_close_loose(outs[0], s, f"{what} output")
        elif kind.startswith("spec"):
            _ws2_assert_match(outs[0], h, f"{what} residual_out", 0.99, 0.9999)
            _ws2_assert_match(outs[1], norm, f"{what} norm_out", 0.98, 0.999)
        else:
            assert not torch.isnan(outs[1]).any().item(), f"{what}: NaN"
            _ws2_assert_close_loose(outs[0], h, f"{what} residual_out")
            _ws2_assert_close_loose(outs[1], norm, f"{what} norm_out")
        _ws2_assert_rank_identical(outs, what)
    dist.barrier()
    _ws2_close(workspace)
    dist.destroy_process_group()


def _run_ws2_cuda_graph_worker(
    world_size,
    rank,
    dtype,
    hidden_dim,
    distributed_init_port,
    backend,
    num_replays,
    gpu_offset=0,
):
    """Capture specialized, fallback and specialized (vLLM aliasing) calls in one graph,
    replay with inputs changed in place; every replay matches eager calls (TRT-LLM
    numerics for the specialized calls, the fallback backend's exact bits for the other)."""
    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    workspace, trtllm, fallback = _ws2_open(world_size, rank, backend)
    # (token_num, aliasing); the middle call is out of range (fallback).
    plan = [(_ws2_inside(0.25), "separate")]
    if _ws2_outside() is not None:
        plan.append((_ws2_outside(), "separate"))
    plan.append((_ws2_inside(0.75), "residual_out_is_input"))
    static = []
    for j, (token_num, aliasing) in enumerate(plan):
        x, residual, gamma = _ws2_inputs(token_num, rank, 7000 + j, device)
        residual_out, norm_out = _ws2_alias(aliasing, x, residual)
        static.append([x, residual, gamma, residual_out, norm_out])

    def run_all():
        for (token_num, _), (x, residual, gamma, residual_out, norm_out) in zip(
            plan, static, strict=True
        ):
            _ws2_call(
                workspace,
                x,
                residual,
                gamma,
                residual_out,
                norm_out,
                use_oneshot=_ws2_vllm_oneshot(token_num),
            )

    run_all()  # eager warm-up
    torch.cuda.synchronize()
    dist.barrier()
    before = _ws2_stats()["dispatch_count"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    torch.cuda.synchronize()
    captured = _ws2_stats()["dispatch_count"] - before
    expected = sum(
        int(_ws2_expect_dispatch(workspace, token_num)) for token_num, _ in plan
    )
    assert captured == expected == 2, f"captured {captured} specialized launches"
    dist.barrier()

    for rep in range(num_replays):
        fresh = []
        for j, ((token_num, _), buffers) in enumerate(zip(plan, static, strict=True)):
            x, residual, gamma = _ws2_inputs(
                token_num, rank, 8000 + 10 * rep + j, device
            )
            buffers[0].copy_(
                x
            )  # also refills the aliased residual_out of the last call
            buffers[1].copy_(residual)
            if buffers[3] is not buffers[0]:
                buffers[3].fill_(float("nan"))
            buffers[4].fill_(float("nan"))
            fresh.append((x, residual))
        graph.replay()
        torch.cuda.synchronize()
        for j, ((token_num, aliasing), buffers) in enumerate(
            zip(plan, static, strict=True)
        ):
            x, residual = fresh[j]
            dispatched = _ws2_expect_dispatch(workspace, token_num)
            ref, _ = _ws2_run(
                trtllm if dispatched else fallback,
                aliasing,
                x,
                residual,
                buffers[2],
                use_oneshot=_ws2_vllm_oneshot(token_num),
            )
            what = f"rank {rank} replay {rep} call {j} (T={token_num}, {aliasing})"
            got = (buffers[3], buffers[4])
            if dispatched:
                _ws2_assert_match(got[0], ref[0], f"{what} residual_out", 0.995, 0.9999)
                _ws2_assert_match(got[1], ref[1], f"{what} norm_out", 0.95, 0.999)
            else:
                _ws2_assert_bitwise(got, ref, f"{what} fallback")
            _ws2_assert_rank_identical(got, what)
    del graph
    dist.barrier()
    _ws2_close(workspace, trtllm, fallback)
    dist.destroy_process_group()


def _run_ws2_kill_switch_worker(
    world_size, rank, dtype, hidden_dim, distributed_init_port, gpu_offset=0
):
    import os

    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    workspaces = []
    # Call time: the env var switches between the kernel and the fallback backend on the
    # same workspace; the kill-switched call is exactly the fallback backend's.
    enabled, trtllm, fallback = _ws2_open(world_size, rank, "specialized")
    workspaces += [enabled, trtllm, fallback]
    stock_backend = enabled.fallback_backend
    x, residual, gamma = _ws2_inputs(_ws2_inside(0.5), rank, 1, device)
    on, n_on = _ws2_run(enabled, "separate", x, residual, gamma)
    off, n_off = _ws2_run(enabled, "separate", x, residual, gamma, disable=True)
    stock, _ = _ws2_run(fallback, "separate", x, residual, gamma)
    ref, _ = _ws2_run(trtllm, "separate", x, residual, gamma)
    assert (n_on, n_off) == (1, 0)
    _ws2_assert_bitwise(off, stock, "kill switch")
    _ws2_assert_match(on[0], ref[0], "kill switch residual_out", 0.995, 0.9999)
    _ws2_assert_match(on[1], ref[1], "kill switch norm_out", 0.95, 0.999)
    assert _ws2_stats()["fallback_reasons"].get("kill_switch", 0) >= 1

    # Creation time: an explicit "specialized" workspace without the kernel's state or
    # extra device memory (every call on the fallback, also once the env var is unset
    # again), and "auto" builds the stock backend.
    rejections = _ws2_stats()["state_rejections"].get("kill_switch", 0)
    torch.cuda.synchronize()
    free_0 = torch.cuda.mem_get_info(device)[0]
    os.environ[_WS2_DISABLE_ENV] = "1"
    try:
        killed = _ws2_workspace(world_size, rank, "specialized")
        workspaces.append(killed)
        torch.cuda.synchronize()
        free_1 = torch.cuda.mem_get_info(device)[0]
        killed_auto = _ws2_workspace(world_size, rank, "auto")
        workspaces.append(killed_auto)
    finally:
        os.environ.pop(_WS2_DISABLE_ENV, None)
    assert isinstance(killed, comm.SpecializedAllReduceFusionWorkspace)
    assert not killed.has_specialized_kernels and _ws2_state(killed) is None
    assert killed.fallback_backend == stock_backend
    assert not isinstance(killed_auto, comm.SpecializedAllReduceFusionWorkspace)
    assert killed_auto.backend == stock_backend
    assert _ws2_stats()["state_rejections"].get("kill_switch", 0) == rejections + 2
    torch.cuda.synchronize()
    free_1b = torch.cuda.mem_get_info(device)[0]
    again = _ws2_workspace(world_size, rank, "specialized")
    workspaces.append(again)
    torch.cuda.synchronize()
    free_2 = torch.cuda.mem_get_info(device)[0]
    assert _ws2_state(again) is not None
    region = _ws2_state(again).region_bytes
    extra = (free_1b - free_2) - (free_0 - free_1)
    assert region > (64 << 20) and abs(extra - region) <= (64 << 20), (
        f"extra device memory of an enabled workspace {extra}, region {region}"
    )
    killed_out, n_killed = _ws2_run(killed, "separate", x, residual, gamma)
    assert n_killed == 0
    _ws2_assert_bitwise(killed_out, stock, "kill switch at creation")
    dist.barrier()
    _ws2_close(*workspaces)
    dist.destroy_process_group()


def _run_ws2_guard_worker(
    world_size, rank, dtype, hidden_dim, distributed_init_port, gpu_offset=0
):
    """Calls the guard must send to the fallback, on a workspace that has the state."""
    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    workspace = _ws2_workspace(world_size, rank, "specialized")
    assert _ws2_state(workspace) is not None, _ws2_stats()["last_state_rejection"]
    token_num = _ws2_inside(0.5)
    x, residual, gamma = _ws2_inputs(token_num, rank, 3, device)
    for reason, kwargs in [
        ("fp32_acc", {"fp32_acc": False}),
        ("weight_bias", {"weight_bias": 0.5}),
    ]:
        before = _ws2_stats()
        residual_out, norm_out = _ws2_alias("separate", x, residual)
        _ws2_call(
            workspace,
            x.clone(),
            residual.clone(),
            gamma,
            residual_out,
            norm_out,
            **kwargs,
        )
        torch.cuda.synchronize()
        after = _ws2_stats()
        assert after["dispatch_count"] == before["dispatch_count"], reason
        assert (
            after["fallback_reasons"].get(reason, 0)
            == before["fallback_reasons"].get(reason, 0) + 1
        ), reason
        assert not torch.isnan(norm_out).any().item(), reason

    # Out-of-range token counts.
    for t_out in {t for t in (_ws2_outside(), 1) if t and not _ws2_allowed(t)}:
        xs, rs, gs = _ws2_inputs(t_out, rank, 4, device)
        before = _ws2_stats()
        residual_out, norm_out = _ws2_alias("separate", xs, rs)
        _ws2_call(workspace, xs, rs, gs, residual_out, norm_out, use_oneshot=True)
        torch.cuda.synchronize()
        after = _ws2_stats()
        assert after["dispatch_count"] == before["dispatch_count"]
        assert (
            after["fallback_reasons"].get("num_tokens", 0)
            == before["fallback_reasons"].get("num_tokens", 0) + 1
        )

    # Pattern 0 on the same workspace never reaches the guard.
    before = _ws2_stats()["calls"]
    out = torch.empty_like(x)
    comm.allreduce_fusion(
        input=x.clone(),
        workspace=workspace,
        pattern=comm.AllReduceFusionPattern.kAllReduce,
        output=out,
        use_oneshot=False,
    )
    torch.cuda.synchronize()
    assert _ws2_stats()["calls"] == before

    # Non-contiguous tensors go to the fallback backend: the call does exactly what a plain
    # workspace of the fallback backend does with the same strided output (the same
    # rejection, or the same bits, also in the untouched half of the strided buffer).
    fallback = _ws2_workspace(world_size, rank, workspace.fallback_backend)
    outcomes = []
    for ws in (workspace, fallback):
        before = _ws2_stats()
        wide = torch.full(
            (token_num, 2 * _WS2_HIDDEN),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        norm_out = torch.full_like(x, float("nan"))
        try:
            _ws2_call(
                ws,
                x.clone(),
                residual.clone(),
                gamma,
                wide[:, :_WS2_HIDDEN],
                norm_out,
            )
            torch.cuda.synchronize()
            outcome = ("returned", wide, norm_out)
        except (ValueError, RuntimeError, TypeError) as exc:
            outcome = ("raised", type(exc), str(exc))
        after = _ws2_stats()
        assert after["dispatch_count"] == before["dispatch_count"]
        outcomes.append((outcome, before, after))
    (got, before, after), (ref, _, _) = outcomes
    assert (
        after["fallback_reasons"].get("non_contiguous", 0)
        == before["fallback_reasons"].get("non_contiguous", 0) + 1
    )
    assert got[0] == ref[0], f"non-contiguous call {got[0]}, plain fallback {ref[0]}"
    if got[0] == "raised":
        assert got[1:] == ref[1:], (got, ref)
    else:
        _ws2_assert_bitwise(got[1:], ref[1:], f"rank {rank} non-contiguous call")
    dist.barrier()
    _ws2_close(workspace, fallback)
    dist.destroy_process_group()


def _run_ws2_fallback_equivalence_worker(
    world_size, rank, dtype, hidden_dim, distributed_init_port, backend, gpu_offset=0
):
    """Calls outside the envelope on the specialized (auto) workspace: the decode-sized
    and other out-of-range token counts, pattern-1 arguments the kernel does not serve,
    pattern 0 and an FP8 quant pattern give bitwise the result of a plain workspace of the
    fallback backend, i.e. of what "auto" built before the specialized backend existed."""
    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    workspace, trtllm, fallback = _ws2_open(world_size, rank, backend)
    inside = _ws2_inside(0.5)
    cases = [
        (t, aliasing, {"use_oneshot": _ws2_vllm_oneshot(t) or None})
        for t in _WS2_OUTSIDE_TOKEN_NUMS
        if not _ws2_allowed(t)
        for aliasing in ("separate", "residual_out_is_input")
    ]
    cases += [
        (t, "separate", {"use_oneshot": None})
        for t in _WS2_OUTSIDE_TOKEN_NUMS
        if not _ws2_allowed(t)
    ]
    cases += [
        (inside, "separate", {"fp32_acc": False}),
        (inside, "residual_out_is_input", {"weight_bias": 0.0}),
    ]
    before = _ws2_stats()["dispatch_count"]
    for i, (token_num, aliasing, kwargs) in enumerate(cases):
        what = f"rank {rank} T={token_num} {aliasing} {kwargs}"
        x, residual, gamma = _ws2_inputs(token_num, rank, 500 + i, device)
        got, n = _ws2_run(workspace, aliasing, x, residual, gamma, **kwargs)
        ref, _ = _ws2_run(fallback, aliasing, x, residual, gamma, **kwargs)
        assert n == 0, what
        _ws2_assert_bitwise(got, ref, what)

    for i, token_num in enumerate((16, inside)):
        x, _, _ = _ws2_inputs(token_num, rank, 600 + i, device)
        outs = []
        for ws in (workspace, fallback):
            out = torch.full_like(x, float("nan"))
            comm.allreduce_fusion(
                input=x.clone(),
                workspace=ws,
                pattern=comm.AllReduceFusionPattern.kAllReduce,
                launch_with_pdl=True,
                output=out,
            )
            outs.append(out)
        torch.cuda.synchronize()
        _ws2_assert_bitwise(
            [outs[0]], [outs[1]], f"rank {rank} kAllReduce T={token_num}"
        )

    for i, token_num in enumerate((64, inside)):
        x, residual, gamma = _ws2_inputs(token_num, rank, 700 + i, device)
        outs = []
        for ws in (workspace, fallback):
            residual_out = torch.full_like(x, float("nan"))
            quant_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            ret = comm.allreduce_fusion(
                input=x.clone(),
                workspace=ws,
                pattern=comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
                launch_with_pdl=True,
                residual_in=residual.clone(),
                residual_out=residual_out,
                quant_out=quant_out,
                rms_gamma=gamma,
                rms_eps=_WS2_EPS,
                scale_factor=torch.ones(1, dtype=torch.float32, device=device),
                layout_code=comm.QuantizationSFLayout.SWIZZLED_128x4,
                fp32_acc=True,
                weight_bias=1.0,
            )
            assert ret is quant_out
            outs.append((residual_out, quant_out))
        torch.cuda.synchronize()
        _ws2_assert_bitwise(
            outs[0],
            outs[1],
            f"rank {rank} kARResidualRMSNormFP8Quant T={token_num}",
        )
    assert _ws2_stats()["dispatch_count"] == before
    dist.barrier()
    _ws2_close(workspace, trtllm, fallback)
    dist.destroy_process_group()


def _run_ws2_inert_worker(
    world_size,
    rank,
    dtype,
    hidden_dim,
    distributed_init_port,
    backend,
    max_token_num,
    token_num,
    gpu_offset=0,
):
    """Outside the creation-time envelope: "auto" builds exactly the stock backend it
    picked before (no state, no extra allocation, no dispatch), an explicit
    "specialized" raises on every rank, and "trtllm" never runs the kernel."""
    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    created = _ws2_stats()["states_created"]
    if backend == "specialized":
        with pytest.raises(ValueError, match="no kernel for this workspace"):
            _ws2_workspace(
                world_size,
                rank,
                backend,
                hidden_dim=hidden_dim,
                dtype=dtype,
                max_token_num=max_token_num,
            )
        assert _ws2_stats()["states_created"] == created
        dist.barrier()
        dist.destroy_process_group()
        return
    expected_backend = _ws2_stock_backend(world_size) if backend == "auto" else backend
    workspace = _ws2_workspace(
        world_size,
        rank,
        backend,
        hidden_dim=hidden_dim,
        dtype=dtype,
        max_token_num=max_token_num,
    )
    assert not isinstance(workspace, comm.SpecializedAllReduceFusionWorkspace)
    assert workspace.backend == expected_backend
    assert _ws2_state(workspace) is None
    assert _ws2_stats()["states_created"] == created
    x, residual, gamma = _ws2_inputs(token_num, rank, 5, device, hidden_dim, dtype)
    x_copy, residual_copy = x.clone(), residual.clone()
    residual_out, norm_out = _ws2_alias("separate", x, residual)
    before = _ws2_stats()
    _ws2_call(workspace, x, residual, gamma, residual_out, norm_out)
    torch.cuda.synchronize()
    after = _ws2_stats()
    assert after["dispatch_count"] == before["dispatch_count"]
    assert after["calls"] == before["calls"]  # the guard is never consulted
    x_all = [torch.empty_like(x_copy) for _ in range(world_size)]
    dist.all_gather(x_all, x_copy)
    _, h, norm = _ws2_torch_reference(x_all, residual_copy, gamma, 1.0)
    _ws2_assert_close_loose(residual_out, h, "residual_out")
    _ws2_assert_close_loose(norm_out, norm, "norm_out")
    dist.barrier()
    _ws2_close(workspace)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Rank-divergent conditions: one rank's local condition (an allocation failure, an omitted
# output, a misaligned view, its stream schedule) must never send the two ranks down
# different paths. Each test drives the condition on one rank only and must finish within
# its time box with correct results (a split collective hangs instead).
# ---------------------------------------------------------------------------


def _ws2_stock_oneshot(workspace, token_num):
    """``use_oneshot`` as vLLM passes it to the workspace's stock backend."""
    backend = getattr(workspace, "fallback_backend", workspace.backend)
    return None if backend == "mnnvl" else _ws2_vllm_oneshot(token_num)


def _run_ws2_divergent_allocation_worker(
    world_size, rank, dtype, hidden_dim, distributed_init_port, backend, gpu_offset=0
):
    """Rank 1's allocation of the kernel's region fails locally after the collective handle
    exchange (e.g. mapping or the pointer table). Both ranks must reject the state alike,
    release the region and stay in step: auto returns the stock workspace on both ranks and
    its calls run the stock backend; an explicit specialized backend raises on both ranks;
    a later workspace gets the state on both ranks."""
    import gc
    import weakref

    from flashinfer.comm import _allreduce_fusion_ws2_specialized as ws2

    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    real = ws2.SymmDeviceMemory
    regions = []

    class _Tracked(real):
        def __init__(self, *args, **kwargs):
            regions.append(weakref.ref(self))
            super().__init__(*args, **kwargs)
            if rank == 1:
                raise RuntimeError("injected local failure after the handle exchange")

    rejected = _ws2_stats()["state_rejections"].get("allocation", 0)
    ws2.SymmDeviceMemory = _Tracked
    try:
        if backend == "specialized":
            with pytest.raises(ValueError, match="allocation"):
                _ws2_workspace(world_size, rank, "specialized")
            workspace = None
        else:
            workspace = _ws2_workspace(world_size, rank, "auto")
    finally:
        ws2.SymmDeviceMemory = real
    stats = _ws2_stats()
    assert stats["state_rejections"].get("allocation", 0) == rejected + 1
    assert "rank 1" in stats["last_state_rejection"], stats["last_state_rejection"]
    # Every rank released its region (its own allocation and its import of the peer's).
    gc.collect()
    assert len(regions) == 1 and regions[0]() is None, f"rank {rank} kept its region"
    if workspace is not None:
        assert not isinstance(workspace, comm.SpecializedAllReduceFusionWorkspace)
        assert workspace.backend == _ws2_stock_backend(world_size)
        token_num = _ws2_inside(0.5)
        x, residual, gamma = _ws2_inputs(token_num, rank, 11, device)
        x_copy, residual_copy = x.clone(), residual.clone()
        residual_out, norm_out = _ws2_alias("separate", x, residual)
        before = _ws2_stats()["dispatch_count"]
        _ws2_call(
            workspace,
            x,
            residual,
            gamma,
            residual_out,
            norm_out,
            use_oneshot=_ws2_stock_oneshot(workspace, token_num),
        )
        torch.cuda.synchronize()
        assert _ws2_stats()["dispatch_count"] == before
        x_all = [torch.empty_like(x_copy) for _ in range(world_size)]
        dist.all_gather(x_all, x_copy)
        _, h, norm = _ws2_torch_reference(x_all, residual_copy, gamma, 1.0)
        _ws2_assert_close_loose(residual_out, h, f"rank {rank} residual_out")
        _ws2_assert_close_loose(norm_out, norm, f"rank {rank} norm_out")
    # The ranks are still in step: the next workspace gets the state on both ranks.
    again = _ws2_workspace(world_size, rank, "specialized")
    assert again.has_specialized_kernels, _ws2_stats()["last_state_rejection"]
    x, residual, gamma = _ws2_inputs(_ws2_inside(0.3), rank, 12, device)
    _, dispatched = _ws2_run(again, "separate", x, residual, gamma)
    assert dispatched == 1
    dist.barrier()
    _ws2_close(workspace, again)
    dist.destroy_process_group()


def _run_ws2_divergent_outputs_worker(
    world_size, rank, dtype, hidden_dim, distributed_init_port, gpu_offset=0
):
    """One rank omits ``residual_out`` and/or ``norm_out`` while its peer passes them: both
    ranks run the specialized kernel (the omitted outputs are allocated), with correct and
    rank-identical results."""
    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    workspace = _ws2_workspace(world_size, rank, "specialized")
    assert workspace.has_specialized_kernels, _ws2_stats()["last_state_rejection"]
    cases = [
        (omitting_rank, omit_ro, omit_no)
        for omitting_rank in (1, 0)
        for omit_ro, omit_no in ((False, True), (True, False), (True, True))
    ]
    for i, (omitting_rank, omit_ro, omit_no) in enumerate(cases):
        token_num = _ws2_inside(0.2 + 0.1 * i)
        what = (
            f"rank {rank} T={token_num}: rank {omitting_rank} omits"
            f"{' residual_out' if omit_ro else ''}{' norm_out' if omit_no else ''}"
        )
        x, residual, gamma = _ws2_inputs(token_num, rank, 900 + i, device)
        x_copy, residual_copy = x.clone(), residual.clone()
        omit = rank == omitting_rank
        residual_out = None if omit and omit_ro else torch.full_like(x, float("nan"))
        norm_out = None if omit and omit_no else torch.full_like(x, float("nan"))
        before = _ws2_stats()["dispatch_count"]
        ret = _ws2_call(workspace, x, residual, gamma, residual_out, norm_out)
        torch.cuda.synchronize()
        assert _ws2_stats()["dispatch_count"] == before + 1, what
        if norm_out is not None:
            assert ret is norm_out, what
        x_all = [torch.empty_like(x_copy) for _ in range(world_size)]
        dist.all_gather(x_all, x_copy)
        _, h, norm = _ws2_torch_reference(x_all, residual_copy, gamma, 1.0)
        _ws2_assert_match(ret, norm, f"{what} norm_out", 0.98, 0.999)
        if residual_out is not None:
            _ws2_assert_match(residual_out, h, f"{what} residual_out", 0.99, 0.9999)
        _ws2_assert_rank_identical([ret], what)
    dist.barrier()
    _ws2_close(workspace)
    dist.destroy_process_group()


def _ws2_misaligned_like(t):
    """A contiguous view with ``t``'s values that starts one element (2 bytes) into an
    aligned allocation, as ``torch.narrow`` of a larger buffer gives."""
    flat = torch.empty(t.numel() + 8, dtype=t.dtype, device=t.device)
    view = flat[1 : 1 + t.numel()].view(t.shape)
    view.copy_(t)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    return view


def _run_ws2_divergent_alignment_worker(
    world_size, rank, dtype, hidden_dim, distributed_init_port, gpu_offset=0
):
    """Rank 1 passes contiguous views that are not 16-byte aligned (``rms_gamma`` alone, or
    every tensor) while rank 0's are aligned: both ranks run the specialized kernel (rank 1
    through aligned copies), and the results are bitwise those of an aligned call, eagerly
    and replayed from a CUDA graph."""
    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    workspace = _ws2_workspace(world_size, rank, "specialized")
    assert workspace.has_specialized_kernels, _ws2_stats()["last_state_rejection"]
    token_num = _ws2_inside(0.5)

    def make(x, residual, gamma, aliasing, which):
        t = {"x": x.clone(), "r": residual.clone(), "g": gamma.clone()}
        if rank == 1:
            for name in ("x", "r", "g") if which == "all" else ("g",):
                t[name] = _ws2_misaligned_like(t[name])
        residual_out, norm_out = _ws2_alias(aliasing, t["x"], t["r"])
        if rank == 1 and which == "all" and aliasing == "separate":
            residual_out = _ws2_misaligned_like(residual_out)
            norm_out = _ws2_misaligned_like(norm_out)
        return t["x"], t["r"], t["g"], residual_out, norm_out

    for i, aliasing in enumerate(_WS2_ALIASINGS):
        for which in ("gamma", "all"):
            what = f"rank {rank} {aliasing} rank 1 misaligned: {which}"
            x, residual, gamma = _ws2_inputs(token_num, rank, 950 + i, device)
            ref, n_ref = _ws2_run(workspace, aliasing, x, residual, gamma)
            assert n_ref == 1, what
            before = _ws2_stats()
            args = make(x, residual, gamma, aliasing, which)
            ret = _ws2_call(workspace, *args)
            torch.cuda.synchronize()
            after = _ws2_stats()
            assert after["dispatch_count"] == before["dispatch_count"] + 1, what
            staged = after["staged_launches"] - before["staged_launches"]
            assert staged == int(rank == 1), f"{what}: staged {staged}"
            assert ret is args[4], what
            _ws2_assert_bitwise((args[3], args[4]), ref, what)

    # A staged call captured in a CUDA graph: replays with new values in the same views.
    x, residual, gamma = _ws2_inputs(token_num, rank, 970, device)
    args = make(x, residual, gamma, "residual_out_is_input", "all")
    _ws2_call(workspace, *args)  # eager warm-up
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _ws2_call(workspace, *args)
    torch.cuda.synchronize()
    dist.barrier()
    for rep in range(2):
        x, residual, gamma = _ws2_inputs(token_num, rank, 980 + rep, device)
        args[0].copy_(x)  # also the aliased residual_out
        args[1].copy_(residual)
        args[2].copy_(gamma)
        args[4].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        ref, _ = _ws2_run(workspace, "residual_out_is_input", x, residual, gamma)
        _ws2_assert_bitwise((args[3], args[4]), ref, f"rank {rank} replay {rep}")
    del graph
    dist.barrier()
    _ws2_close(workspace)
    dist.destroy_process_group()


def _run_ws2_cross_stream_worker(
    world_size, rank, dtype, hidden_dim, distributed_init_port, gpu_offset=0
):
    """Two specialized workspaces called back to back on two streams, with one rank's first
    stream delayed by a GPU sleep, so that without ordering the ranks would start the two
    grids in opposite orders (each grid holds every SM slot it needs and waits on its peer,
    which then cannot become resident). The launcher runs the grids of one device in host
    issue order, so both calls finish, with correct results."""
    device = _ws2_setup(world_size, rank, distributed_init_port, gpu_offset)
    ws_a = _ws2_workspace(world_size, rank, "specialized")
    ws_b = _ws2_workspace(world_size, rank, "specialized")
    for ws in (ws_a, ws_b):
        assert ws.has_specialized_kernels, _ws2_stats()["last_state_rejection"]
    streams = {"a": torch.cuda.Stream(), "b": torch.cuda.Stream()}
    workspaces = {"a": ws_a, "b": ws_b}
    tokens = {"a": _ws2_inside(0.9), "b": _ws2_inside(0.7)}
    # About 50 ms of GPU sleep at any clock this test runs at.
    sleep_cycles = int(1e8)
    for rnd, (delayed_rank, order) in enumerate(((1, "ab"), (0, "ba"))):
        calls = {}
        for k, name in enumerate(order):
            x, residual, gamma = _ws2_inputs(
                tokens[name], rank, 990 + 10 * rnd + k, device
            )
            calls[name] = (x.clone(), residual.clone(), gamma, x, residual)
        torch.cuda.synchronize()
        dist.barrier()
        before = _ws2_stats()["dispatch_count"]
        for k, name in enumerate(order):
            stream = streams[name]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                if k == 0 and rank == delayed_rank:
                    torch.cuda._sleep(sleep_cycles)
                x_copy, residual_copy, gamma, x, residual = calls[name]
                residual_out, norm_out = _ws2_alias("separate", x, residual)
                _ws2_call(workspaces[name], x, residual, gamma, residual_out, norm_out)
                calls[name] += (residual_out, norm_out)
        torch.cuda.synchronize()
        assert _ws2_stats()["dispatch_count"] == before + 2
        for name in order:
            x_copy, residual_copy, gamma, _, _, residual_out, norm_out = calls[name]
            what = f"rank {rank} round {rnd} workspace {name}"
            x_all = [torch.empty_like(x_copy) for _ in range(world_size)]
            dist.all_gather(x_all, x_copy)
            _, h, norm = _ws2_torch_reference(x_all, residual_copy, gamma, 1.0)
            _ws2_assert_match(residual_out, h, f"{what} residual_out", 0.99, 0.9999)
            _ws2_assert_match(norm_out, norm, f"{what} norm_out", 0.98, 0.999)
    dist.barrier()
    _ws2_close(ws_a, ws_b)
    dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_trtllm_allreduce_fusion_ws2_specialized_divergent_allocation_failure(backend):
    _ws2_skip_unless_target()
    _ws2_launch(
        _run_ws2_divergent_allocation_worker, target_args=(backend,), timeout_s=300
    )


def test_trtllm_allreduce_fusion_ws2_specialized_divergent_optional_outputs():
    _ws2_skip_unless_target()
    _ws2_launch(_run_ws2_divergent_outputs_worker, timeout_s=300)


def test_trtllm_allreduce_fusion_ws2_specialized_divergent_alignment():
    _ws2_skip_unless_target()
    _ws2_launch(_run_ws2_divergent_alignment_worker, timeout_s=300)


def test_trtllm_allreduce_fusion_ws2_specialized_cross_stream_progress():
    _ws2_skip_unless_target()
    _ws2_launch(_run_ws2_cross_stream_worker, timeout_s=300)


@pytest.mark.parametrize(
    "backend,aliasing",
    [("specialized", aliasing) for aliasing in _WS2_ALIASINGS]
    + [("auto", "residual_out_is_input")],
)
def test_trtllm_allreduce_fusion_ws2_specialized(backend, aliasing):
    """Specialized (and auto, which picks it) vs the TRT-LLM kernels and vs the fallback
    backend on the same inputs across the guard edges, with NaN-prefilled outputs, the
    aliasing forms, the kill switch and bitwise-identical outputs across ranks."""
    _ws2_skip_unless_target()
    _ws2_launch(
        _run_ws2_correctness_worker,
        target_args=(backend, aliasing, _ws2_token_nums()),
    )


def test_trtllm_allreduce_fusion_ws2_specialized_back_to_back():
    _ws2_skip_unless_target()
    _ws2_launch(_run_ws2_back_to_back_worker, target_args=(60,))


@pytest.mark.parametrize("backend", ["specialized", "auto"])
def test_trtllm_allreduce_fusion_ws2_specialized_cuda_graph(backend):
    _ws2_skip_unless_target()
    _ws2_launch(_run_ws2_cuda_graph_worker, target_args=(backend, 4))


def test_trtllm_allreduce_fusion_ws2_specialized_kill_switch():
    _ws2_skip_unless_target()
    _ws2_launch(_run_ws2_kill_switch_worker)


def test_trtllm_allreduce_fusion_ws2_specialized_guards():
    _ws2_skip_unless_target()
    _ws2_launch(_run_ws2_guard_worker)


def test_trtllm_allreduce_fusion_ws2_specialized_fallback_equivalence():
    """Calls outside the envelope on an auto workspace run exactly what auto ran before."""
    _ws2_skip_unless_target()
    _ws2_launch(_run_ws2_fallback_equivalence_worker, target_args=("auto",))


def test_trtllm_allreduce_fusion_ws2_specialized_gpu_offset():
    """Device index != rank (e.g. the second TP2 instance of a 4-GPU tray)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 2:
        pytest.skip("needs more than 2 GPUs")
    gpu_offset = torch.cuda.device_count() - 2
    _ws2_skip_unless_target(gpu_offset=gpu_offset)
    _ws2_launch(
        _run_ws2_correctness_worker,
        target_args=(
            "specialized",
            "residual_out_is_input",
            [_ws2_inside(0.0), _ws2_inside(0.6)],
        ),
        gpu_offset=gpu_offset,
    )


@pytest.mark.parametrize("backend", ["auto", "specialized", "trtllm"])
@pytest.mark.parametrize(
    "world_size,hidden_dim,dtype,capacity",
    [
        (2, 6144, torch.float16, "served"),
        (2, 7168, torch.bfloat16, "served"),
        (4, 6144, torch.bfloat16, "served"),
        # max_token_num below the smallest allowlisted token count.
        (2, 6144, torch.bfloat16, "below_min"),
        # The target configuration: outside the envelope on every compute capability but
        # 10.3 (B200); on 10.3 only backend="trtllm" is checked here (it stays stock).
        (2, 6144, torch.bfloat16, "served"),
    ],
)
def test_trtllm_allreduce_fusion_ws2_specialized_inert(
    backend, world_size, hidden_dim, dtype, capacity
):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    t_min = min(lo for lo, _ in _ws2_ranges())
    if capacity == "below_min":
        if t_min <= 1:
            pytest.skip("the allowlist starts at one token")
        max_token_num = t_min - 1
        token_num = max_token_num
    else:
        # The vLLM workspace size (64 MiB of activations) and a token count that would
        # dispatch in the target configuration.
        max_token_num = (64 << 20) // (hidden_dim * 2)
        token_num = min(_ws2_inside(0.5), max_token_num)
    if (
        backend != "trtllm"
        and _ws2_all_target(range(world_size))
        and (world_size, hidden_dim, dtype) == (2, 6144, torch.bfloat16)
        and max_token_num >= t_min
    ):
        pytest.skip("the target configuration dispatches on compute capability 10.3")
    _ws2_launch(
        _run_ws2_inert_worker,
        target_args=(backend, max_token_num, token_num),
        world_size=world_size,
        dtype=dtype,
        hidden_dim=hidden_dim,
    )


if __name__ == "__main__":
    # Test both legacy and unified APIs, both standard and Gemma RMSNorm.
    for wb in (0.0, 1.0):
        print(f"Testing legacy API (weight_bias={wb})...")
        test_trtllm_allreduce_fusion(
            2, torch.float16, 1024, legacy_api=True, weight_bias=wb
        )
        print(f"\nTesting unified API (weight_bias={wb})...")
        test_trtllm_allreduce_fusion(
            2, torch.float16, 1024, legacy_api=False, weight_bias=wb
        )
