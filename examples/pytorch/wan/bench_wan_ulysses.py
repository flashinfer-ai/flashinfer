"""Benchmark Ulysses sequence parallelism in the Wan FlashInfer attention.

Runs the example's ``FlashInferWanAttention`` (Wan2.1-14B geometry by default)
with the sequence sharded across ranks, comparing the public
``flashinfer.comm.UlyssesCommunicator`` under its two backends:

- ``nccl``: forced ``backend="nccl"`` (dist.all_to_all_single + permute glue)
- ``auto``: the fused-transpose NVLink-P2P kernel on a verified full-NVLink
  topology, automatic NCCL fallback elsewhere. The report prints the
  *effective* backend and the fallback reason, so NCCL-vs-NCCL results on
  fallback machines are labeled as such.

Verifies the two backends produce bit-identical outputs, sanity-checks
against a single-GPU full-sequence reference, then reports per-iteration
latency.

Example (8 GPUs):
    python bench_wan_ulysses.py --world-size 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist

_WAN_DIR = Path(__file__).resolve().parent
for p in (str(_WAN_DIR), str(_WAN_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_attention(args, device):
    from transformer_wan_flashinfer import FlashInferWanAttention

    torch.manual_seed(0)  # identical weights on every rank
    attn = FlashInferWanAttention(
        dim=args.dim,
        heads=args.heads,
        dim_head=args.dim_head,
        gemm_backend="torch",
        attention_backend=args.attention_backend,
    )
    return attn.to(device=device, dtype=torch.bfloat16).eval()


def _time_forward(attn, x_local, iters, warmup, group):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        attn(x_local)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start.record()
    for _ in range(iters):
        attn(x_local)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _worker(world_size, rank, port, args):
    from transformer_wan_flashinfer import set_ulysses_communicator

    from flashinfer.comm import UlyssesCommunicator

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    B, H, D = args.batch, args.heads, args.dim_head
    S_global = args.seq_len
    assert S_global % world_size == 0 and H % world_size == 0
    S_local = S_global // world_size

    attn = _build_attention(args, device)

    # Same global input on every rank; each rank owns its sequence shard.
    torch.manual_seed(42)
    x_global = torch.randn(B, S_global, args.dim, dtype=torch.bfloat16, device=device)
    x_local = x_global[:, rank * S_local : (rank + 1) * S_local].contiguous()

    max_elems = B * S_local * H * D  # largest a2a operand (input numel == output numel)
    results = {}
    a2a_results = {}
    outs = {}
    backends = {}
    q_like = torch.randn(B, S_local, H, D, dtype=torch.bfloat16, device=device)
    for impl, backend in (("nccl", "nccl"), ("flashinfer", "auto")):
        comm = UlyssesCommunicator(
            group,
            max_elems=max_elems,
            dtype=torch.bfloat16,
            backend=backend,
            device=device,
        )
        backends[impl] = (comm.backend, comm.fallback_reason)
        try:
            set_ulysses_communicator(comm)
            with torch.no_grad():
                outs[impl] = attn(x_local).clone()
                results[impl] = _time_forward(
                    attn, x_local, args.iters, args.warmup, group
                )
                # a2a-only: one attention layer issues 3 scatters (q/k/v) and
                # 1 gather — time that communication pattern in isolation,
                # warming up with the exact same 3+1 unit as the timed loop.
                u = comm.scatter_heads(q_like)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                for _ in range(args.warmup):
                    comm.scatter_heads(q_like)
                    comm.scatter_heads(q_like)
                    comm.scatter_heads(q_like)
                    comm.gather_heads(u)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                start.record()
                for _ in range(args.iters):
                    comm.scatter_heads(q_like)
                    comm.scatter_heads(q_like)
                    comm.scatter_heads(q_like)
                    comm.gather_heads(u)
                end.record()
                torch.cuda.synchronize()
                a2a_results[impl] = start.elapsed_time(end) / args.iters
        finally:
            # clear the registry BEFORE the collective close so a failure
            # above cannot leave the transformer pointing at a dead comm
            set_ulysses_communicator(None)
        comm.close()

    # The two backends implement the same permutation -> bit-identical.
    assert torch.equal(outs["nccl"], outs["flashinfer"]), (
        f"rank {rank}: flashinfer ulysses output differs from NCCL baseline"
    )

    # Single-GPU full-sequence reference (numerics differ slightly: the
    # sequence-parallel run attends with H/W heads per rank, so reduction
    # order changes -> allclose, not equal).
    with torch.no_grad():
        ref = attn(x_global)[:, rank * S_local : (rank + 1) * S_local]
    torch.testing.assert_close(outs["nccl"].float(), ref.float(), atol=3e-2, rtol=3e-2)

    if rank == 0:
        n = results["nccl"]
        f = results["flashinfer"]
        fi_backend, fi_reason = backends["flashinfer"]
        assert backends["nccl"][0] == "nccl"
        print(
            f"[wan ulysses] ws={world_size} B={B} S_global={S_global} "
            f"H={H} D={D} dim={args.dim} backend={args.attention_backend}"
        )
        print(f"  flashinfer effective backend: {fi_backend}")
        if fi_reason is not None:
            print(f"  !! fallback ({fi_reason}): the comparison below is NCCL vs NCCL")
        print(f"  self-attn fwd (nccl a2a)      : {n:8.3f} ms/iter")
        print(f"  self-attn fwd (flashinfer a2a): {f:8.3f} ms/iter")
        print(f"  speedup: {n / f:.3f}x   ({(n - f) / n * 100:.1f}% faster)")
        an = a2a_results["nccl"]
        af = a2a_results["flashinfer"]
        print(f"  a2a only 3xin+1xout (nccl)      : {an:8.3f} ms/iter")
        print(f"  a2a only 3xin+1xout (flashinfer): {af:8.3f} ms/iter")
        print(f"  a2a speedup: {an / af:.3f}x")
        print(
            "  correctness: flashinfer == nccl (bit-identical); "
            "matches single-GPU reference (allclose)"
        )

    dist.barrier(group=group)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    p.add_argument("--batch", type=int, default=1)
    # Wan2.1-14B: 40 heads x 128, dim 5120; 480p/81f video -> ~32k tokens
    p.add_argument("--seq-len", type=int, default=32760, help="global sequence length")
    p.add_argument("--dim", type=int, default=5120)
    p.add_argument("--heads", type=int, default=40)
    p.add_argument("--dim-head", type=int, default=128)
    p.add_argument(
        "--attention-backend", default="torch", help="torch|cudnn|trtllm|single|auto"
    )
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    procs = [
        mp.Process(target=_worker, args=(args.world_size, r, port, args))
        for r in range(args.world_size)
    ]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
        assert pr.exitcode == 0, f"worker failed with exit code {pr.exitcode}"


if __name__ == "__main__":
    main()
