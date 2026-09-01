"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark the experimental PCIe Ulysses backend against torch's own
all_to_all_single plus the layout permutes (the same reference
UlyssesCommunicator's NCCL backend runs). Every PCIe result is validated
element-wise against that reference before it is timed.

    torchrun --standalone --nproc-per-node=8 benchmarks/comm/bench_ulysses_pcie.py \\
      --seq-len 37888 --num-heads 56 --head-dim 128

    # force a route: all-P2P, or all-RDMA at any multi-rank world size
    FLASHINFER_ULYSSES_PCIE_ROUTE=p2p torchrun ...
    FLASHINFER_ULYSSES_PCIE_ROUTE=rdma torchrun ...

``--seq-len`` is the *global* sequence length: each rank owns
``seq_len / world_size`` tokens before ``scatter_heads``.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator


def nccl_scatter(x: torch.Tensor, world: int) -> torch.Tensor:
    # Copied verbatim from UlyssesCommunicator._nccl_scatter_heads.
    B, S_local, H, D = x.shape
    H_local = H // world
    xt = x.reshape(B, S_local, world, H_local, D).permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(xt)
    dist.all_to_all_single(recv, xt)
    return recv.permute(1, 0, 2, 3, 4).reshape(B, world * S_local, H_local, D)


def nccl_gather(x: torch.Tensor, world: int) -> torch.Tensor:
    # Copied verbatim from UlyssesCommunicator._nccl_gather_heads.
    B, S_global, H_local, D = x.shape
    S_local = S_global // world
    xt = x.reshape(B, world, S_local, H_local, D).permute(1, 0, 2, 3, 4).contiguous()
    recv = torch.empty_like(xt)
    dist.all_to_all_single(recv, xt)
    return (
        recv.permute(1, 2, 0, 3, 4).reshape(B, S_local, world * H_local, D).contiguous()
    )


def timed(fn, args, device):
    """nccl-tests shape: barrier, time a whole loop with the host clock,
    divide, max-reduce across ranks, median over trials. Per-call CUDA events
    would measure enqueue queueing on the NCCL path, and the hybrid route
    progresses its completion queue on the host."""
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(args.trials):
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize(device)
        ms = (time.perf_counter() - start) / args.iters * 1e3
        value = torch.tensor([ms], dtype=torch.float64, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        samples.append(value.item())
    median = statistics.median(samples)
    spread = 100.0 * statistics.stdev(samples) / median if args.trials > 1 else 0.0
    return median, spread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=37888, help="global sequence")
    parser.add_argument("--num-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "int8"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500, help="calls per trial")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--json", type=str, default=None, help="rank-0 row dump")
    args = parser.parse_args()

    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert args.seq_len % world == 0, "--seq-len must be divisible by the world size"
    dtype = getattr(torch, args.dtype)
    local_seq = args.seq_len // world

    torch.manual_seed(1024 + rank)

    def operand(shape):
        if dtype.is_floating_point:
            return torch.randn(shape, dtype=dtype, device=device)
        return torch.randint(-128, 128, shape, dtype=dtype, device=device)

    q, k, v = (operand((1, local_seq, args.num_heads, args.head_dim)) for _ in range(3))
    attn_out = operand((1, args.seq_len, args.num_heads // world, args.head_dim))
    # Bytes that leave this GPU: the local 1/world share never crosses a link.
    remote = q.numel() * q.element_size() * (world - 1) / world

    rows = []
    with UlyssesCommunicator(
        max_bytes=q.nbytes, dtype=dtype, backend="pcie", device=device
    ) as comm:
        q_out = comm.allocate_output(q, "scatter_heads")
        k_out = comm.allocate_output(k, "scatter_heads")
        v_out = comm.allocate_output(v, "scatter_heads")
        gather_out = comm.allocate_output(attn_out, "gather_heads")

        # Correctness first, on every rank: torchrun ends the whole group when
        # one rank's assert fails.
        torch.testing.assert_close(
            comm.scatter_heads(q, out=q_out), nccl_scatter(q, world), rtol=0, atol=0
        )
        torch.testing.assert_close(
            comm.gather_heads(attn_out, out=gather_out),
            nccl_gather(attn_out, world),
            rtol=0,
            atol=0,
        )
        torch.cuda.synchronize(device)

        def qkv_layer_pcie():
            comm.scatter_heads(q, out=q_out)
            comm.scatter_heads(k, out=k_out)
            comm.scatter_heads(v, out=v_out)
            return comm.gather_heads(attn_out, out=gather_out)

        def qkv_layer_nccl():
            return (
                nccl_scatter(q, world),
                nccl_scatter(k, world),
                nccl_scatter(v, world),
                nccl_gather(attn_out, world),
            )

        for name, pcie_fn, nccl_fn, case_remote in (
            (
                "scatter_heads",
                lambda: comm.scatter_heads(q, out=q_out),
                lambda: nccl_scatter(q, world),
                remote,
            ),
            (
                "gather_heads",
                lambda: comm.gather_heads(attn_out, out=gather_out),
                lambda: nccl_gather(attn_out, world),
                remote,
            ),
            ("qkv_layer", qkv_layer_pcie, qkv_layer_nccl, remote * 4),
        ):
            pcie_ms, pcie_sd = timed(pcie_fn, args, device)
            nccl_ms, nccl_sd = timed(nccl_fn, args, device)
            rows.append(
                {
                    "case": name,
                    # A forced RDMA route can silently fall back to all-P2P
                    # (RuntimeWarning only), so the dump must state what it
                    # actually measured.
                    "transport": comm.transport,
                    "route": comm.decision.reason,
                    "pcie_ms": pcie_ms,
                    "pcie_sd_pct": pcie_sd,
                    "nccl_ms": nccl_ms,
                    "nccl_sd_pct": nccl_sd,
                    "speedup": nccl_ms / pcie_ms,
                    "remote_gbps": case_remote / (pcie_ms / 1e3) / 1e9,
                }
            )

        transport = comm.transport
        route = comm.decision.reason

    if rank == 0:
        print(
            f"# world={world} transport={transport} seq={args.seq_len} "
            f"heads={args.num_heads} head_dim={args.head_dim} dtype={args.dtype} "
            f"warmup={args.warmup} iters={args.iters} trials={args.trials}"
        )
        print(f"# route: {route}")
        print(
            f"{'case':<16} {'pcie_ms':>9} {'sd%':>5} {'nccl_ms':>9} {'sd%':>5} "
            f"{'speedup':>8} {'remote_GB/s':>12}"
        )
        for r in rows:
            print(
                f"{r['case']:<16} {r['pcie_ms']:>9.3f} {r['pcie_sd_pct']:>5.1f} "
                f"{r['nccl_ms']:>9.3f} {r['nccl_sd_pct']:>5.1f} "
                f"{r['speedup']:>7.2f}x {r['remote_gbps']:>12.1f}"
            )
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"# wrote {args.json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
