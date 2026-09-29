import multiprocessing as mp
import socket

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend


FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
# E4M3FN's smallest positive subnormal is 2^-9.
FP8_MIN_SUBNORMAL = 2.0**-9
MIN_FP8_SCALE = FP8_MIN_SUBNORMAL / FP8_MAX


def _get_open_port() -> int:
    """Return a free localhost port for a spawned NCCL process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _reference(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute torch references for allreduce, RMSNorm, and dynamic FP8 quant."""
    reduced = allreduce_in.clone()
    dist.all_reduce(reduced, group=group)
    h = reduced if residual is None else reduced + residual
    y = torch.nn.functional.rms_norm(h, (h.shape[-1],), weight=weight, eps=eps)
    scale = torch.clamp(
        y.float().abs().amax(dim=-1, keepdim=True) / FP8_MAX,
        min=MIN_FP8_SCALE,
    )
    q = (
        (y.float() / scale)
        .clamp(
            min=float(torch.finfo(torch.float8_e4m3fn).min),
            max=float(torch.finfo(torch.float8_e4m3fn).max),
        )
        .to(torch.float8_e4m3fn)
    )
    return reduced, h, y, q, scale


def _assert_dynamic_fp8_dequant_close(
    quant_out: torch.Tensor,
    scale_out: torch.Tensor,
    ref_quant: torch.Tensor,
    ref_scale: torch.Tensor,
) -> None:
    """Compare dequantized dynamic FP8 outputs with scale-aware tolerances."""
    scale_max = float(ref_scale.max().item())
    torch.testing.assert_close(
        quant_out.float() * scale_out,
        ref_quant.float() * ref_scale,
        atol=scale_max * 0.05,
        rtol=0.05,
    )


def _worker(
    rank: int,
    world_size: int,
    port: int,
    with_residual: bool,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    use_oneshot: bool,
) -> None:
    """Validate one TRT-LLM dynamic FP8 fusion configuration on one rank."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    workspace = None
    try:
        torch.manual_seed(1234 + rank)
        M, H = num_tokens, hidden_size
        eps = 1e-6

        allreduce_in = torch.randn(M, H, device=device, dtype=dtype)
        residual = (
            torch.randn(M, H, device=device, dtype=dtype) if with_residual else None
        )
        weight = torch.randn(H, device=device, dtype=dtype)
        allreduce_ref = allreduce_in.clone()
        residual_ref = residual.clone() if residual is not None else None

        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=M,
            hidden_dim=H,
            dtype=dtype,
            comm_backend=TorchDistBackend(group=group),
        )

        quant_out = torch.empty(M, H, device=device, dtype=torch.float8_e4m3fn)
        scale_out = torch.empty(M, 1, device=device, dtype=torch.float32)
        residual_out = torch.empty(M, H, device=device, dtype=dtype)
        norm_out = torch.empty(M, H, device=device, dtype=dtype)

        if with_residual:
            pattern = comm.AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant
            residual_arg = residual
            norm_arg = None
        else:
            pattern = comm.AllReduceFusionPattern.kARResidualRMSNormOutDynamicFP8Quant
            residual_arg = torch.zeros_like(allreduce_in)
            norm_arg = norm_out

        comm.allreduce_fusion(
            input=allreduce_in,
            workspace=workspace,
            pattern=pattern,
            launch_with_pdl=False,
            residual_in=residual_arg,
            residual_out=residual_out,
            norm_out=norm_arg,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=weight,
            rms_eps=eps,
            scale_factor=None,
            use_oneshot=use_oneshot,
            fp32_acc=True,
        )
        torch.cuda.synchronize()

        _, h_ref, y_ref, q_ref, scale_ref = _reference(
            allreduce_ref, residual_ref, weight, eps, group
        )

        expected_residual = h_ref if with_residual else y_ref
        observed_residual = residual_out if with_residual else norm_out
        torch.testing.assert_close(
            observed_residual.float(),
            expected_residual.float(),
            atol=8e-1,
            rtol=1e-2,
        )
        torch.testing.assert_close(scale_out, scale_ref, atol=1e-3, rtol=1e-3)

        _assert_dynamic_fp8_dequant_close(quant_out, scale_out, q_ref, scale_ref)
    finally:
        if workspace is not None:
            workspace.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_cuda_graph(
    rank: int,
    world_size: int,
    port: int,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    use_oneshot: bool,
) -> None:
    """Validate CUDA Graph replay for TRT-LLM dynamic FP8 fusion on one rank."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    workspace = None
    try:
        torch.manual_seed(5678 + rank)
        M, H = num_tokens, hidden_size
        eps = 1e-6

        static_input = torch.empty(M, H, device=device, dtype=dtype)
        residual = torch.randn(M, H, device=device, dtype=dtype)
        weight = torch.randn(H, device=device, dtype=dtype)
        residual_out = torch.empty(M, H, device=device, dtype=dtype)
        quant_out = torch.empty(M, H, device=device, dtype=torch.float8_e4m3fn)
        scale_out = torch.empty(M, 1, device=device, dtype=torch.float32)

        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=M,
            hidden_dim=H,
            dtype=dtype,
            comm_backend=TorchDistBackend(group=group),
        )

        def run_op() -> None:
            comm.allreduce_fusion(
                input=static_input,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kARResidualRMSNormDynamicFP8Quant,
                launch_with_pdl=False,
                residual_in=residual,
                residual_out=residual_out,
                quant_out=quant_out,
                scale_out=scale_out,
                rms_gamma=weight,
                rms_eps=eps,
                use_oneshot=use_oneshot,
                fp32_acc=True,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                static_input.copy_(torch.randn_like(static_input))
                run_op()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        dist.barrier(group=group)

        graph = torch.cuda.CUDAGraph()
        static_input.copy_(torch.randn_like(static_input))
        with torch.cuda.graph(graph):
            run_op()
        torch.cuda.synchronize()

        for step in range(3):
            torch.manual_seed(9000 + step * world_size + rank)
            replay_input = torch.randn(M, H, device=device, dtype=dtype)
            static_input.copy_(replay_input)
            dist.barrier(group=group)
            graph.replay()
            torch.cuda.synchronize()

            _, h_ref, _y_ref, q_ref, scale_ref = _reference(
                replay_input, residual, weight, eps, group
            )
            torch.testing.assert_close(
                residual_out.float(), h_ref.float(), atol=8e-1, rtol=1e-2
            )
            torch.testing.assert_close(scale_out, scale_ref, atol=1e-3, rtol=1e-3)
            _assert_dynamic_fp8_dequant_close(quant_out, scale_out, q_ref, scale_ref)
    finally:
        if workspace is not None:
            workspace.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


_TEST_CASES = [
    pytest.param(2, 1, 4096, torch.bfloat16, True, id="ws2_m1_h4096_bf16_oneshot"),
    pytest.param(2, 16, 4096, torch.bfloat16, True, id="ws2_m16_h4096_bf16_oneshot"),
    pytest.param(2, 16, 4096, torch.bfloat16, False, id="ws2_m16_h4096_bf16_twoshot"),
    pytest.param(2, 128, 8192, torch.float16, True, id="ws2_m128_h8192_fp16_oneshot"),
    pytest.param(2, 128, 8192, torch.float16, False, id="ws2_m128_h8192_fp16_twoshot"),
    pytest.param(
        2, 1024, 4096, torch.float16, False, id="ws2_m1024_h4096_fp16_twoshot"
    ),
    pytest.param(4, 16, 4096, torch.bfloat16, True, id="ws4_m16_h4096_bf16_oneshot"),
    pytest.param(4, 16, 4096, torch.bfloat16, False, id="ws4_m16_h4096_bf16_twoshot"),
]


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize(
    "world_size,num_tokens,hidden_size,dtype,use_oneshot", _TEST_CASES
)
def test_trtllm_allreduce_dynamic_fp8_quant(
    with_residual: bool,
    world_size: int,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    use_oneshot: bool,
) -> None:
    """Spawn ranks and validate TRT-LLM dynamic FP8 fusion correctness."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires at least {world_size} GPUs")
    port = _get_open_port()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                world_size,
                port,
                with_residual,
                num_tokens,
                hidden_size,
                dtype,
                use_oneshot,
            ),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for rank, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"rank {rank} failed with exit code {proc.exitcode}"


@pytest.mark.parametrize("use_oneshot", [True, False])
def test_trtllm_allreduce_dynamic_fp8_cuda_graph_replay(use_oneshot: bool) -> None:
    """Spawn ranks and validate TRT-LLM dynamic FP8 CUDA Graph replay."""
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires at least {world_size} GPUs")
    port = _get_open_port()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_worker_cuda_graph,
            args=(
                rank,
                world_size,
                port,
                16,
                4096,
                torch.bfloat16,
                use_oneshot,
            ),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for rank, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"rank {rank} failed with exit code {proc.exitcode}"
