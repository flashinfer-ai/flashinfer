"""
Test for AllReduce + Residual + RMSNorm + Per-Token-Group FP8 Quant fusion.
"""

import multiprocessing as mp
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend

FP8_E4M3_MAX = 448.0

# All (hidden_dim, token_num, group_size) combos to test.
# Each case is run with both oneshot and twoshot (when token_num > world_size).
# Batched into a single process group to avoid per-case process spawn overhead.
TEST_CASES = [
    # === Fast path (warp shuffle): group_size % VEC_SIZE == 0, power-of-2, <= 32 ===
    # group_size=128, various padding combos
    (7168, 4, 128),  # no padding (mn%4=0, groups%4=0)
    (7168, 1, 128),  # MN padding only (tma_aligned_mn=4)
    (7168, 3, 128),  # MN padding only
    (7168, 64, 128),  # larger, no padding
    (7168, 127, 128),  # larger, MN padding
    (7168, 512, 128),  # large token count
    (768, 4, 128),  # K padding only (groups_per_row=6, 6%4=2)
    (768, 3, 128),  # both MN and K padding
    (768, 1, 128),  # MN + K padding, single token
    (640, 4, 128),  # K padding only (groups_per_row=5, 5%4=1)
    (640, 3, 128),  # both MN and K padding
    (640, 253, 128),  # larger, both padding
    (384, 4, 128),  # k_num_packed=1, no MN padding
    (384, 1, 128),  # k_num_packed=1, both padding
    (256, 4, 128),  # k_num_packed=1, no padding
    (256, 1, 128),  # k_num_packed=1, MN padding
    # group_size=64
    (7168, 4, 64),
    (7168, 3, 64),  # MN padding
    (768, 4, 64),
    (768, 512, 64),  # large token count
    (640, 3, 64),  # both MN and K padding
    # group_size=32 (group_size_in_vecs=4)
    (768, 4, 32),
    (768, 3, 32),  # MN padding
    # group_size=16 (single shuffle step, group_size_in_vecs=2)
    (768, 4, 16),
    (768, 1, 16),  # MN padding, single token
    # group_size=8 (== VEC_SIZE, no shuffle, group_size_in_vecs=1)
    (768, 4, 8),
    (768, 3, 8),  # MN padding
    # group_size=256 (group_size_in_vecs=32, warp boundary)
    (4096, 4, 256),
    (4096, 3, 256),  # MN padding
    # === Slow path (shared memory): arbitrary group sizes ===
    # group_size not divisible by VEC_SIZE
    (768, 4, 12),  # groups_per_row=64
    (768, 3, 6),  # groups_per_row=128, MN padding
    (768, 1, 6),  # single token + slow path
    (768, 4, 4),  # group_size < VEC_SIZE, groups_in_block > blockDim.x
    # extreme small group sizes
    (768, 4, 3),  # group_size=3, groups_per_row=256
    (768, 3, 3),  # MN padding
    (768, 4, 2),  # group_size=2, scale-write loop 4 iterations
    # group_size_in_vecs not power of 2
    (768, 4, 48),  # group_size_in_vecs=6
    (768, 512, 48),  # large token count
    (960, 4, 48),  # different threads_per_token
    (768, 3, 24),  # group_size_in_vecs=3, MN padding
    (768, 1, 24),  # single token
    (768, 253, 6),  # large token count + group_size < VEC_SIZE
    (768, 256, 12),  # large token count + non-VEC_SIZE-divisible
    # slow path with K padding
    (624, 4, 48),  # groups_per_row=13 (13%4=1, K padding)
    (648, 3, 12),  # groups_per_row=54 (54%4=2, K+MN padding)
    # large hidden_dim + slow path
    (7200, 4, 48),  # groups_per_row=150 (150%4=2, K padding)
    (7200, 3, 24),  # groups_per_row=300 (300%4=0, MN padding only)
    # group_size_in_vecs > 32 (spans multiple warps)
    (7168, 4, 512),  # group_size_in_vecs=64
    (7168, 256, 512),  # large token count + multi-warp
    (4096, 3, 512),  # MN padding
    (7168, 1, 512),  # single token + multi-warp
]


def _scale_storage_size(token_num: int, groups_per_row: int) -> int:
    """Number of int32 elements in the TMA-aligned packed scale storage."""
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    return token_num + (k_num_packed - 1) * tma_aligned_mn


def _unpack_scales(
    packed_scales: torch.Tensor, token_num: int, groups_per_row: int
) -> np.ndarray:
    """Unpack UE8M0 exponent bytes from TMA-aligned packed int32 layout.

    Returns uint8 array of shape [token_num, groups_per_row].
    """
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    num_elems = _scale_storage_size(token_num, groups_per_row)
    raw_bytes = (
        torch.as_strided(packed_scales, (num_elems,), (1,))
        .cpu()
        .view(torch.uint8)
        .numpy()
    )
    exponents = np.empty((token_num, groups_per_row), dtype=np.uint8)
    for t in range(token_num):
        for g in range(groups_per_row):
            elem_idx = (g // 4) * tma_aligned_mn + t
            exponents[t, g] = raw_bytes[elem_idx * 4 + g % 4]
    return exponents


def _pack_scales(
    exponents: np.ndarray,
    token_num: int,
    groups_per_row: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack UE8M0 exponent bytes into TMA-aligned int32 layout with zero padding."""
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    num_elems = _scale_storage_size(token_num, groups_per_row)

    buf = np.zeros(num_elems * 4, dtype=np.uint8)
    for t in range(token_num):
        for g in range(groups_per_row):
            elem_idx = (g // 4) * tma_aligned_mn + t
            buf[elem_idx * 4 + g % 4] = exponents[t, g]

    flat = torch.from_numpy(buf.view(np.int32).copy()).to(device)
    out = torch.empty_strided(
        (token_num, k_num_packed),
        (1, tma_aligned_mn),
        device=device,
        dtype=torch.int32,
    )
    torch.as_strided(out, (num_elems,), (1,)).copy_(flat[:num_elems])
    return out


def ref_per_token_group_quant_fp8_packed(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token-group FP8 quantization with UE8M0 packed scales.

    1. Per-group absmax
    2. UE8M0 scale = 2^ceil(log2(max(absmax / 448, 1e-10)))
    3. Quantize: clamp(x / scale, -448, 448) -> fp8_e4m3
    4. Pack exponents 4-per-int32 in TMA-aligned column-major layout
    """
    token_num, hidden_dim = x.shape
    groups_per_row = hidden_dim // group_size

    x_grouped = x.float().reshape(token_num, groups_per_row, group_size)

    # Per-group absmax -> UE8M0 power-of-2 scale
    absmax = x_grouped.abs().amax(dim=-1)
    y_s = torch.exp2(torch.ceil(torch.log2((absmax / FP8_E4M3_MAX).clamp(min=1e-10))))

    # Quantize to FP8 e4m3 using division (matches kernel)
    q = (x_grouped / y_s.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quant_out = q.reshape(token_num, hidden_dim).to(torch.float8_e4m3fn)

    # Extract UE8M0 exponent bytes via numpy reinterpret
    exponents = ((y_s.cpu().numpy().view(np.uint32) >> 23) & 0xFF).astype(np.uint8)

    return quant_out, _pack_scales(exponents, token_num, groups_per_row, x.device)


def ref_quantize_fp8_with_packed_scales(
    x: torch.Tensor, packed_scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Quantize x to FP8 e4m3 using pre-computed packed UE8M0 scales.

    Isolates the FP8 cast from the scale computation: given the same
    scale, do fused and reference produce the same FP8 values?
    """
    token_num, hidden_dim = x.shape
    groups_per_row = hidden_dim // group_size

    exponents = _unpack_scales(packed_scales, token_num, groups_per_row)
    scales = np.where(
        exponents > 0,
        np.ldexp(1.0, exponents.astype(np.int32) - 127),
        0.0,
    ).astype(np.float32)
    scales_t = torch.from_numpy(scales).to(x.device)

    x_grouped = x.float().reshape(token_num, groups_per_row, group_size)
    q = (x_grouped / scales_t.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return q.reshape(token_num, hidden_dim).to(torch.float8_e4m3fn)


def _run_single_case(
    workspace,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    hidden_dim: int,
    token_num: int,
    group_size: int,
    use_oneshot: bool,
    device: torch.device,
):
    """Run a single test case using an existing workspace (recreated if needed)."""
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    num_scale_elems = _scale_storage_size(token_num, groups_per_row)

    # Check if current workspace is large enough; recreate if not
    if not workspace[0].is_buffer_size_sufficient(
        world_size, token_num, hidden_dim, dtype
    ):
        workspace[0].destroy()
        workspace[0] = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )

    allreduce_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device) * 8
    residual_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device) * 8
    rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

    residual_out = torch.empty_like(allreduce_in)
    norm_out = torch.empty_like(allreduce_in)
    quant_out = torch.empty(
        token_num,
        hidden_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    scale_out = torch.empty_strided(
        (token_num, k_num_packed),
        (1, tma_aligned_mn),
        device=device,
        dtype=torch.int32,
    )

    # fill with non-zero values to verify kernel zeroes padding bytes
    torch.as_strided(scale_out, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)

    # run fused kernel
    comm.allreduce_fusion(
        input=allreduce_in,
        workspace=workspace[0],
        pattern=comm.AllReduceFusionPattern.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant,
        residual_in=residual_in,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=rms_gamma,
        rms_eps=1e-5,
        block_quant_group_size=group_size,
        fp32_acc=True,
        use_oneshot=use_oneshot,
    )
    torch.cuda.synchronize()

    # run reference quantization on norm_out
    ref_quant, ref_scale = ref_per_token_group_quant_fp8_packed(
        norm_out,
        group_size,
    )

    # verify packed scales match exactly
    fused_scale_flat = torch.as_strided(scale_out, (num_scale_elems,), (1,)).cpu()
    ref_scale_flat = torch.as_strided(ref_scale, (num_scale_elems,), (1,)).cpu()
    assert torch.equal(fused_scale_flat, ref_scale_flat), (
        f"[hidden={hidden_dim}, tokens={token_num}, gs={group_size}, "
        f"oneshot={use_oneshot}] Packed scale mismatch: "
        f"{(fused_scale_flat != ref_scale_flat).sum().item()}/{num_scale_elems} differ"
    )

    # verify quantized activations match exactly
    assert torch.equal(quant_out.view(torch.uint8), ref_quant.view(torch.uint8)), (
        f"[hidden={hidden_dim}, tokens={token_num}, gs={group_size}, "
        f"oneshot={use_oneshot}] FP8 quant mismatch: "
        f"{(quant_out.view(torch.uint8) != ref_quant.view(torch.uint8)).sum().item()}"
        f"/{quant_out.numel()} differ"
    )


def _run_batch_worker(
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    cases: list,
    store_path: str,
    gpu_offset: int = 0,
):
    """Worker that runs all test cases in a single process group."""
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    # Create initial workspace with the largest problem size
    max_tokens = max(c[1] for c in cases)
    max_hidden = max(c[0] for c in cases)
    workspace = [
        comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=max_tokens,
            hidden_dim=max_hidden,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )
    ]

    try:
        for hidden_dim, token_num, group_size in cases:
            for use_oneshot in [True, False]:
                if not use_oneshot and token_num <= world_size:
                    continue  # twoshot requires token_num > world_size

                np.random.seed(42)
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)

                _run_single_case(
                    workspace,
                    world_size,
                    rank,
                    dtype,
                    hidden_dim,
                    token_num,
                    group_size,
                    use_oneshot,
                    device,
                )
                dist.barrier(group=group)
    finally:
        workspace[0].destroy()
        dist.destroy_process_group(group=group)


def _multi_process_batch(
    world_size: int,
    dtype: torch.dtype,
    cases: list,
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)
    store_file = tempfile.mktemp(prefix="flashinfer_dist_store_")
    procs = []
    for i in range(world_size):
        proc = mp.Process(
            target=_run_batch_worker,
            args=(world_size, i, dtype, cases, store_file, gpu_offset),
            name=f"Worker-{i}",
        )
        proc.start()
        procs.append(proc)
    for i, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"Process {i} failed with exit code {proc.exitcode}"


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_allreduce_rmsnorm_group_fp8_quant(world_size, dtype):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"Need {world_size} GPUs, have {available_gpus}")

    _multi_process_batch(world_size, dtype, TEST_CASES)
