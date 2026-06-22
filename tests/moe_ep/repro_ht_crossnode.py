"""Minimal pure-nccl.ep HT (FLAT) cross-node repro — NO FlashInfer, NO FFN.

Mirrors contrib/nccl_ep/ep_test.py's HIGH_THROUGHPUT path (FLAT layout, same
GroupConfig + dispatch/combine + complete() sequence) but bootstraps via
torch.distributed instead of MPI so it runs under torchrun. Just dispatch + combine
(no expert FFN), to isolate the nccl.ep library from the FlashInfer integration.

Repro of: multi-node HT aborts with
  CUDA error nccl_ep.cc:2884 'an illegal memory access was encountered'
Single-node (1 node / 8 GPU) works; the error appears at 2+ nodes (cross-node /
MNNVL). Geometry follows ep_test.py: top_k=min(8,world),
num_experts=min(256, top_k*world), hidden=7168, bf16; per-rank tokens default 128.

Run (2 nodes, 8 GPU each = 16 ranks):
    NCCL_MNNVL_ENABLE=1 torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE \\
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \\
        tests/moe_ep/repro_ht_crossnode.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

import nccl.ep as ep
from nccl.ep.interop.torch import get_nccl_comm_from_group

PER_RANK = int(os.environ.get("REPRO_TOKENS_PER_RANK", "128"))
HIDDEN = 7168

# EP_USE_ALLOC=1: pass an explicit GroupConfig.alloc (cudaMalloc/cudaFree hooks),
# matching ep_test.py / ep_bench. Module-scope keepalive: NCCL EP stores the raw
# function pointers, so these decorated callbacks must outlive the Group.
_ALLOC_CFG = None
if os.environ.get("EP_USE_ALLOC") == "1":
    import ctypes

    from cuda.bindings import runtime as cudart  # type: ignore[import-not-found]

    @ep.AllocFn
    def _alloc_fn(out_ptr, size, _ctx):  # noqa: ANN001
        err, ptr = cudart.cudaMalloc(size)
        out_ptr[0] = ctypes.c_void_p(int(ptr))
        return int(err)

    @ep.FreeFn
    def _free_fn(ptr, _ctx):  # noqa: ANN001
        (err,) = cudart.cudaFree(ptr)
        return int(err)

    _ALLOC_CFG = ep.AllocConfig(
        alloc_fn=ctypes.cast(_alloc_fn, ctypes.c_void_p).value,
        free_fn=ctypes.cast(_free_fn, ctypes.c_void_p).value,
    )


def main():
    # EP_BOOTSTRAP=gloo bootstraps torch.distributed over host TCP so torch holds
    # NO NCCL/IB comm; nccl.ep's get_nccl_comm_from_group then builds the ONLY NCCL
    # comm (mirrors ep_test's MPI bcast). Tests whether a coexisting torch-NCCL comm
    # is what breaks cross-node HT (GIN/GDAKI) — LL cross-node tolerates it, HT may not.
    bootstrap = os.environ.get("EP_BOOTSTRAP", "nccl")
    _im = os.environ.get("EP_INIT_METHOD")
    if _im:
        dist.init_process_group(
            bootstrap,
            init_method=_im,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
    else:
        dist.init_process_group(bootstrap)
    rank = dist.get_rank()
    world = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)

    top_k = min(8, world)
    num_experts = int(os.environ.get("REPRO_NUM_EXPERTS", str(min(256, top_k * world))))
    local_n = num_experts // world
    if rank == 0:
        print(
            f"[repro] world={world} top_k={top_k} num_experts={num_experts} "
            f"local_n={local_n} per_rank={PER_RANK} hidden={HIDDEN} "
            f"MNNVL={os.environ.get('NCCL_MNNVL_ENABLE', '0')}",
            flush=True,
        )

    # max_recv_tokens_per_rank = ep_bench's uniform estimate max_tok*top_k (NOT
    # max_tok*world). At 8 GPU these coincide (world==top_k); at 16+ GPU world>top_k
    # so max_tok*world over-sizes the recv budget — testing whether that breaks the
    # cross-node GIN RDMA path (2884).
    max_recv = PER_RANK * top_k
    comm = get_nccl_comm_from_group(group=None)
    _cfg_kw = dict(
        algorithm=ep.Algorithm.HIGH_THROUGHPUT,
        num_experts=num_experts,
        max_dispatch_tokens_per_rank=PER_RANK,
        max_recv_tokens_per_rank=max_recv,
        max_token_bytes=HIDDEN * 2,  # bf16
    )
    if _ALLOC_CFG is not None:
        _cfg_kw["alloc"] = _ALLOC_CFG
        if rank == 0:
            print("[repro] using explicit AllocConfig", flush=True)
    cfg = ep.GroupConfig(**_cfg_kw)
    group = ep.Group.create(comm, cfg)

    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    x = torch.randn(PER_RANK, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g)
    # DISTINCT top-k experts per token (like ep_test.py randperm).
    scores = torch.randn(PER_RANK, num_experts, device="cuda", generator=g)
    topk_idx = scores.topk(top_k, dim=-1).indices.to(torch.int64)
    topk_w = torch.softmax(
        torch.randn(PER_RANK, top_k, device="cuda", generator=g), dim=-1
    )

    stream = torch.cuda.current_stream().cuda_stream
    handle = group.create_handle(
        ep.Layout.FLAT,
        ep.Tensor(topk_idx),
        layout_info=None,
        config=ep.HandleConfig(),
        stream=stream,
    )

    # HT recv budget = max_tokens_per_rank * top_k (ep_bench's uniform estimate;
    # NOT max_tokens_per_rank * num_local_experts, which only coincides when
    # local_n == top_k and otherwise overflows the library's max_recv buffer → 3269).
    num_recv = PER_RANK * top_k
    out_t = torch.empty(num_recv, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out_w = torch.empty(num_recv, top_k, dtype=torch.float32, device="cuda")
    out_idx = torch.empty(num_recv, top_k, dtype=torch.int64, device="cuda")

    co = torch.empty(PER_RANK, HIDDEN, dtype=torch.bfloat16, device="cuda")
    # Loop dispatch+combine to reproduce the repeated-op behavior. REPRO_ITERS>1
    # exercises the 2nd+ HT dispatch on the same handle (the path that hangs in the
    # FlashInfer benchmark). Buffers are reused across iters (like ep_bench.cu).
    iters = int(os.environ.get("REPRO_ITERS", "1"))
    for it in range(iters):
        # --- dispatch ---
        handle.dispatch(
            ep.DispatchInputs(tokens=ep.Tensor(x), topk_weights=ep.Tensor(topk_w)),
            ep.DispatchOutputs(
                tokens=ep.Tensor(out_t),
                topk_weights=ep.Tensor(out_w),
                topk_idx=ep.Tensor(out_idx),
            ),
            layout_info=None,
            config=ep.DispatchConfig(send_only=0, round_scales=0),
            stream=stream,
        )
        handle.complete(stream=stream)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"[repro] iter {it}: dispatch OK", flush=True)

        # --- combine (expert output = recv buffer unchanged; no FFN) ---
        handle.combine(
            ep.CombineInputs(tokens=ep.Tensor(out_t)),
            ep.CombineOutputs(tokens=ep.Tensor(co)),
            config=ep.CombineConfig(send_only=0),
            stream=stream,
        )
        handle.complete(stream=stream)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"[repro] iter {it}: dispatch+combine SUCCEEDED", flush=True)

    if rank == 0:
        print(
            f"[repro] HT loop x{iters} SUCCEEDED (no hang / illegal access)", flush=True
        )
    handle.destroy()
    group.destroy()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
