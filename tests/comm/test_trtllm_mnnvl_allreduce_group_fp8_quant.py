import os
import socket

import pytest
import torch
import torch.distributed as dist
from mpi4py import MPI

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend
from tests.test_helpers.comm import init_torch_distributed_from_mpi


FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _prefer_mpi_launcher_env() -> None:
    """Prefer MPI rank variables over Slurm rank variables for this test."""
    mpi_comm = MPI.COMM_WORLD
    if mpi_comm.Get_size() <= 1:
        return

    local_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
    os.environ["RANK"] = str(mpi_comm.Get_rank())
    os.environ["WORLD_SIZE"] = str(mpi_comm.Get_size())
    os.environ["LOCAL_RANK"] = str(local_comm.Get_rank())
    if "MASTER_ADDR" not in os.environ:
        master_addr = mpi_comm.bcast(
            socket.gethostname() if mpi_comm.Get_rank() == 0 else None, root=0
        )
        os.environ["MASTER_ADDR"] = master_addr
    local_comm.Free()

    for key in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NODEID"):
        os.environ.pop(key, None)


def _scale_storage_size(token_num: int, groups_per_row: int) -> int:
    """Return the int32 storage size for packed, TMA-aligned UE8M0 scales."""
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    return token_num + (k_num_packed - 1) * tma_aligned_mn


def _allocate_scale_out(
    token_num: int, hidden_dim: int, group_size: int, device: torch.device
) -> torch.Tensor:
    """Allocate the packed UE8M0 scale tensor expected by the fused API."""
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    return torch.empty_strided(
        (token_num, k_num_packed),
        (1, tma_aligned_mn),
        dtype=torch.int32,
        device=device,
    )


def _unpack_scale_exponents(
    packed_scales: torch.Tensor, groups_per_row: int
) -> torch.Tensor:
    """Unpack four UE8M0 exponent bytes from each logical int32."""
    shifts = torch.arange(4, dtype=torch.int64, device=packed_scales.device) * 8
    return (
        ((packed_scales.to(torch.int64).unsqueeze(-1) >> shifts) & 0xFF)
        .flatten(1)[:, :groups_per_row]
        .to(torch.uint8)
    )


def _assert_scale_padding_zero(
    packed_scales: torch.Tensor, token_num: int, groups_per_row: int
) -> None:
    """Verify packed-K and TMA-aligned token padding are initialized."""
    valid_bytes_in_last_pack = groups_per_row % 4
    if valid_bytes_in_last_pack:
        valid_mask = (1 << (valid_bytes_in_last_pack * 8)) - 1
        last_pack = packed_scales[:, -1].to(torch.int64) & 0xFFFFFFFF
        assert torch.count_nonzero(last_pack & (0xFFFFFFFF ^ valid_mask)) == 0

    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    if token_num < tma_aligned_mn and k_num_packed > 1:
        storage_size = _scale_storage_size(token_num, groups_per_row)
        storage = torch.as_strided(packed_scales, (storage_size,), (1,))
        pack_offsets = (
            torch.arange(k_num_packed - 1, device=packed_scales.device) * tma_aligned_mn
        )
        token_padding = torch.arange(
            token_num, tma_aligned_mn, device=packed_scales.device
        )
        assert (
            torch.count_nonzero(
                storage[(pack_offsets[:, None] + token_padding).flatten()]
            )
            == 0
        )


def _reference_norm(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute distributed residual and RMSNorm references."""
    reduced = allreduce_in.clone()
    dist.all_reduce(reduced)
    residual_out = reduced + residual
    norm_out = torch.nn.functional.rms_norm(
        residual_out, (residual_out.shape[-1],), weight=weight, eps=eps
    )
    return residual_out, norm_out


def _reference_group_quant(
    norm_out: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-token-group FP8 values and UE8M0 scales."""
    token_num, hidden_dim = norm_out.shape
    groups_per_row = hidden_dim // group_size
    grouped = norm_out.float().reshape(token_num, groups_per_row, group_size)
    absmax = grouped.abs().amax(dim=-1)
    scale_input = (absmax / FP8_E4M3_MAX).clamp(min=1e-10).contiguous()
    scale_bits = scale_input.view(torch.int32)
    scales = torch.where(
        (scale_bits & 0x7FFFFF) != 0,
        (scale_bits + 0x800000) & 0x7F800000,
        scale_bits,
    ).view(torch.float32)
    quant = (
        (grouped / scales.unsqueeze(-1))
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .reshape(token_num, hidden_dim)
        .to(torch.float8_e4m3fn)
    )
    exponents = ((scales.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return quant, exponents, scales


def _assert_dequant_close(
    quant_out: torch.Tensor,
    packed_scales: torch.Tensor,
    reference: torch.Tensor,
    group_size: int,
    atol: float = 8e-1,
) -> None:
    """Compare dequantized packed-group FP8 output to an RMSNorm reference."""
    token_num, hidden_dim = reference.shape
    groups_per_row = hidden_dim // group_size
    exponents = _unpack_scale_exponents(packed_scales, groups_per_row).to(torch.int32)
    scales_t = torch.where(
        exponents > 0,
        torch.exp2(exponents.float() - 127.0),
        torch.zeros_like(exponents, dtype=torch.float32),
    )
    dequant = (
        quant_out.float().reshape(token_num, groups_per_row, group_size)
        * scales_t.unsqueeze(-1)
    ).reshape(token_num, hidden_dim)
    torch.testing.assert_close(
        dequant,
        reference.float(),
        atol=atol,
        rtol=0.05,
    )


@pytest.mark.parametrize("with_norm_out", [False, True])
@pytest.mark.parametrize(
    "use_oneshot,token_num,hidden_dim,group_size,expected_error",
    [
        (True, 15, 4096, 128, None),
        (True, 3, 4224, 128, None),
        (False, 127, 4096, 64, None),
        (False, 129, 4224, 128, None),
        (True, 3, 768, 12, None),
        (False, 127, 4096, 512, None),
        (True, 1, 32808, 24, "no exact CTA partition"),
    ],
)
def test_mnnvl_allreduce_group_fp8_quant(
    use_oneshot: bool,
    token_num: int,
    hidden_dim: int,
    group_size: int,
    expected_error: str | None,
    with_norm_out: bool,
) -> None:
    """Validate MNNVL packed per-token-group FP8 fusion."""
    if torch.cuda.device_count() == 0:
        pytest.skip("requires CUDA")

    _prefer_mpi_launcher_env()
    init_torch_distributed_from_mpi()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip(f"requires at least 2 ranks, got {world_size}")

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    eps = 1e-6

    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=world_size,
            rank=rank,
            max_token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            gpus_per_node=torch.cuda.device_count(),
            comm_backend=TorchDistBackend(),
            force_oneshot_support=use_oneshot,
        )
        torch.manual_seed(1234 + rank)
        allreduce_in = torch.randn(token_num, hidden_dim, device=device, dtype=dtype)
        residual = torch.empty(token_num, hidden_dim, device=device, dtype=dtype)
        weight = torch.empty(hidden_dim, device=device, dtype=dtype)
        if rank == 0:
            residual.copy_(torch.randn_like(residual))
            weight.copy_(torch.randn_like(weight))
        dist.broadcast(residual, src=0)
        dist.broadcast(weight, src=0)

        residual_out = torch.empty_like(allreduce_in)
        norm_out = torch.empty_like(allreduce_in) if with_norm_out else None
        quant_out = torch.empty_like(allreduce_in, dtype=torch.float8_e4m3fn)
        scale_out = _allocate_scale_out(token_num, hidden_dim, group_size, device)
        num_scale_elems = _scale_storage_size(token_num, hidden_dim // group_size)
        torch.as_strided(scale_out, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)
        pattern = (
            comm.AllReduceFusionPattern.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant
            if with_norm_out
            else comm.AllReduceFusionPattern.kARResidualRMSNormPerTokenGroupFP8PackedQuant
        )

        def run_op() -> None:
            comm.allreduce_fusion(
                input=allreduce_in,
                workspace=workspace,
                pattern=pattern,
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                quant_out=quant_out,
                scale_out=scale_out,
                rms_gamma=weight,
                rms_eps=eps,
                block_quant_group_size=group_size,
                fp32_acc=True,
                use_oneshot=use_oneshot,
            )

        if expected_error is not None:
            with pytest.raises(ValueError, match=expected_error):
                run_op()
            return

        run_op()
        torch.cuda.synchronize()
        _assert_scale_padding_zero(scale_out, token_num, hidden_dim // group_size)

        ref_residual, ref_norm = _reference_norm(allreduce_in, residual, weight, eps)
        torch.testing.assert_close(
            residual_out.float(), ref_residual.float(), atol=8e-1, rtol=1e-2
        )

        if norm_out is not None:
            torch.testing.assert_close(
                norm_out.float(), ref_norm.float(), atol=8e-1, rtol=1e-2
            )
            ref_quant, ref_exponents, _ = _reference_group_quant(norm_out, group_size)
            fused_exponents = _unpack_scale_exponents(
                scale_out, hidden_dim // group_size
            )
            assert torch.equal(fused_exponents, ref_exponents)
            assert torch.equal(quant_out.view(torch.uint8), ref_quant.view(torch.uint8))
        else:
            _assert_dequant_close(quant_out, scale_out, ref_norm, group_size)
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.barrier()


@pytest.mark.parametrize("use_oneshot", [True, False])
def test_mnnvl_allreduce_group_fp8_cuda_graph_replay(use_oneshot: bool) -> None:
    """Validate MNNVL packed group FP8 CUDA Graph replay."""
    if torch.cuda.device_count() == 0:
        pytest.skip("requires CUDA")

    _prefer_mpi_launcher_env()
    init_torch_distributed_from_mpi()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip(f"requires at least 2 ranks, got {world_size}")

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    token_num = 16 if use_oneshot else 128
    hidden_dim = 4096
    group_size = 128
    dtype = torch.bfloat16
    eps = 1e-6

    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=world_size,
            rank=rank,
            max_token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            gpus_per_node=torch.cuda.device_count(),
            comm_backend=TorchDistBackend(),
            force_oneshot_support=use_oneshot,
        )
        static_input = torch.empty(token_num, hidden_dim, device=device, dtype=dtype)
        residual = torch.empty_like(static_input)
        weight = torch.empty(hidden_dim, device=device, dtype=dtype)
        if rank == 0:
            residual.copy_(torch.randn_like(residual))
            weight.copy_(torch.randn_like(weight))
        dist.broadcast(residual, src=0)
        dist.broadcast(weight, src=0)

        residual_out = torch.empty_like(static_input)
        quant_out = torch.empty_like(static_input, dtype=torch.float8_e4m3fn)
        scale_out = _allocate_scale_out(token_num, hidden_dim, group_size, device)

        def run_op() -> None:
            comm.allreduce_fusion(
                input=static_input,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kARResidualRMSNormPerTokenGroupFP8PackedQuant,
                residual_in=residual,
                residual_out=residual_out,
                quant_out=quant_out,
                scale_out=scale_out,
                rms_gamma=weight,
                rms_eps=eps,
                block_quant_group_size=group_size,
                use_oneshot=use_oneshot,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                static_input.copy_(torch.randn_like(static_input))
                run_op()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        static_input.copy_(torch.randn_like(static_input))
        with torch.cuda.graph(graph):
            run_op()
        torch.cuda.synchronize()

        for step in range(2):
            torch.manual_seed(8000 + step * world_size + rank)
            replay_input = torch.randn_like(static_input)
            static_input.copy_(replay_input)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()

            ref_residual, ref_norm = _reference_norm(
                replay_input, residual, weight, eps
            )
            torch.testing.assert_close(
                residual_out.float(), ref_residual.float(), atol=8e-1, rtol=1e-2
            )
            _assert_dequant_close(quant_out, scale_out, ref_norm, group_size)
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.barrier()
