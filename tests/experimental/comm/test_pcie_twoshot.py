"""Correctness + micro-benchmark for PCIe two-shot fp8 SP collectives.

Run with torchrun on 2, 4 or 8 GPUs:

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
        --nproc-per-node=8 tests/distributed/test_pcie_twoshot.py
"""

import os
import time

import torch
import torch.distributed as dist

from flashinfer.experimental.sm12x.comm.pcie.pcie_twoshot import (
    PCIeTwoShotSP,
    quantize_per_row,
)

ROWS = 4096
ROW_ELEMS = 6144


def _partial(seed: int, rows: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, ROW_ELEMS, generator=gen, dtype=torch.float32)
    return x.to(device=device, dtype=torch.bfloat16)


def _check_reduce_scatter(
    pool: PCIeTwoShotSP, rank: int, world: int, step: int
) -> None:
    device = pool.device
    payloads = []
    scales = []
    for r in range(world):
        q, s = quantize_per_row(_partial(1000 * step + r, ROWS, device))
        payloads.append(q)
        scales.append(s)

    out = pool.reduce_scatter_fp8(payloads[rank], scales[rank])

    rows_per_rank = ROWS // world
    lo, hi = rank * rows_per_rank, (rank + 1) * rows_per_rank
    ref = torch.zeros(rows_per_rank, ROW_ELEMS, dtype=torch.float32, device=device)
    for r in range(world):
        ref += payloads[r][lo:hi].float() * scales[r][lo:hi, None]
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def _check_all_gather(pool: PCIeTwoShotSP, rank: int, world: int, step: int) -> None:
    device = pool.device
    rows_per_rank = ROWS // world
    shards = []
    scales = []
    for r in range(world):
        q, s = quantize_per_row(_partial(5000 * step + r, rows_per_rank, device))
        shards.append(q)
        scales.append(s)

    out = pool.all_gather_fp8(shards[rank], scales[rank])

    ref = torch.cat(
        [
            (shards[r].float() * scales[r][:, None]).to(torch.bfloat16)
            for r in range(world)
        ]
    )
    assert torch.equal(out, ref), "all_gather must be exact (dequant only)"


def _check_graph_capture(pool: PCIeTwoShotSP, rank: int, world: int) -> None:
    device = pool.device
    rows_per_rank = ROWS // world
    q_in = torch.zeros(ROWS, ROW_ELEMS, dtype=torch.float8_e4m3fn, device=device)
    s_in = torch.zeros(ROWS, dtype=torch.float32, device=device)
    rs_out = torch.empty(rows_per_rank, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    ag_q = torch.zeros(
        rows_per_rank, ROW_ELEMS, dtype=torch.float8_e4m3fn, device=device
    )
    ag_s = torch.zeros(rows_per_rank, dtype=torch.float32, device=device)
    ag_out = torch.empty(ROWS, ROW_ELEMS, dtype=torch.bfloat16, device=device)

    graph = torch.cuda.CUDAGraph()
    # Warmup on a side stream, then capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        pool.reduce_scatter_fp8(q_in, s_in, rs_out)
        pool.all_gather_fp8(ag_q, ag_s, ag_out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()

    with torch.cuda.graph(graph):
        pool.reduce_scatter_fp8(q_in, s_in, rs_out)
        pool.all_gather_fp8(ag_q, ag_s, ag_out)

    for step in (11, 12):
        payloads, scales, shards, sscales = [], [], [], []
        for r in range(world):
            q, s = quantize_per_row(_partial(7000 * step + r, ROWS, device))
            payloads.append(q)
            scales.append(s)
            qs, ss = quantize_per_row(_partial(9000 * step + r, ROWS // world, device))
            shards.append(qs)
            sscales.append(ss)
        q_in.copy_(payloads[rank])
        s_in.copy_(scales[rank])
        ag_q.copy_(shards[rank])
        ag_s.copy_(sscales[rank])
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()

        rows_per_rank = ROWS // world
        lo, hi = rank * rows_per_rank, (rank + 1) * rows_per_rank
        ref = torch.zeros(rows_per_rank, ROW_ELEMS, dtype=torch.float32, device=device)
        for r in range(world):
            ref += payloads[r][lo:hi].float() * scales[r][lo:hi, None]
        torch.testing.assert_close(rs_out.float(), ref, rtol=2e-2, atol=2e-2)
        ag_ref = torch.cat(
            [
                (shards[r].float() * sscales[r][:, None]).to(torch.bfloat16)
                for r in range(world)
            ]
        )
        assert torch.equal(ag_out, ag_ref)


def _bench(pool: PCIeTwoShotSP, rank: int, world: int) -> None:
    device = pool.device
    rows_per_rank = ROWS // world
    x = torch.randn(ROWS, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    q, s = quantize_per_row(x)
    qs, ss = quantize_per_row(x[:rows_per_rank])
    rs_out = torch.empty(rows_per_rank, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    ag_out = torch.empty(ROWS, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    nccl_rs_out = torch.empty(
        rows_per_rank, ROW_ELEMS, dtype=torch.bfloat16, device=device
    )
    nccl_ag_out = torch.empty(ROWS, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    shard_bf16 = x[:rows_per_rank].contiguous()

    def timeit(fn, iters=30) -> float:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1e6

    results = {
        "sm12x rs_fp8": timeit(lambda: pool.reduce_scatter_fp8(q, s, rs_out)),
        "sm12x ag_fp8": timeit(lambda: pool.all_gather_fp8(qs, ss, ag_out)),
        "nccl rs bf16": timeit(lambda: dist.reduce_scatter_tensor(nccl_rs_out, x)),
        "nccl ag bf16": timeit(
            lambda: dist.all_gather_into_tensor(nccl_ag_out, shard_bf16)
        ),
    }
    if rank == 0:
        payload_mb = ROWS * ROW_ELEMS / 1e6
        print(
            f"[{world} ranks, {ROWS}x{ROW_ELEMS}, payload {payload_mb:.0f} MB bf16-equiv]"
        )
        for name, us in results.items():
            print(f"  {name:14s} {us:9.1f} us")


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    pool = PCIeTwoShotSP.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_rows=ROWS,
        row_elems=ROW_ELEMS,
    )

    for step in range(4):  # exercises double-buffer slot alternation
        _check_reduce_scatter(pool, rank, world, step)
        _check_all_gather(pool, rank, world, step)
    _check_graph_capture(pool, rank, world)
    dist.barrier()
    if rank == 0:
        print("pcie_twoshot correctness OK")
    _bench(pool, rank, world)

    pool.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
