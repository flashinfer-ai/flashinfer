"""FlashInfer EP comm benchmark — apples-to-apples with contrib/nccl_ep/ep_bench.

Measures FlashInfer's NCCL-EP dispatch+combine (comm only, NO expert FFN) the same
way ep_bench does, so the 28-case matrix (LL em/rm @128, HT @4096/8192, x
{8,16,32,64} GPU, IB + MNNVL) is directly comparable. Emits ep_bench-compatible
text so the upstream scripts/parse_results.py parses these logs unchanged.

Per stage we report BOTH:
  * host-observed time  — wall-clock around the op (incl. dispatch's recv-count sync)
  * kernel-only time    — CUDA-event bracket of the op (CUPTI proxy; immune to the
                          host-sync that defeats bench_gpu_time, matching the layer's
                          per-stage timing)
and the derived bandwidth:
  * LL   bytes = num_valid_selections * hidden * 2  (per-rank tokens*top_k), same for
                 dispatch and combine — matches ep_bench calculateLowLatencyBytes.
  * HT   bytes = total_recv (unique (src_rank, token) with >=1 local expert) * hidden
                 * 2, averaged across ranks — matches ep_bench's recv-side total_bw.

Times/bytes are averaged across ranks (all_reduce SUM / world), like ep_bench.

Launch (per fabric env; matches scripts/run_matrix.sh COMMON args):
    NCCL_GIN_TYPE=3 [NCCL_MNNVL_ENABLE=1] torchrun --nnodes=N --nproc_per_node=8 ... \\
      benchmarks/bench_ep_matrix.py --algorithm ll --layout em --tokens 128 \\
      --hidden 7168 --top-k 8 --experts 256 --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean
from time import perf_counter


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=["ll", "ht"], default="ll")
    p.add_argument(
        "--layout",
        choices=["em", "rm", "fl"],
        default="em",
        help="em=EXPERT_MAJOR, rm=RANK_MAJOR (ll only); fl=FLAT (ht)",
    )
    p.add_argument(
        "--tokens", type=int, default=128, help="PER-RANK tokens (ep_bench -t)"
    )
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--validate", action="store_true", help="single dispatch+combine, no perf"
    )
    p.add_argument(
        "--fresh-handle",
        action="store_true",
        help="recreate the nccl.ep handle each iter (outside the timed region). HT "
        "wedges when one handle is reused across many dispatch+combine iters; LL "
        "tolerates reuse.",
    )
    return p.parse_args()


def main() -> int:
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        CombineInputParams,
        DispatchInputParams,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        HandleParams,
        create_fleet,
    )
    from flashinfer.moe_ep.algo_knobs import (
        HandleAlgoKnobTopKWeights,
        HandleAlgoKnobUserStream,
    )

    args = _parse_args()
    # File-based rendezvous (EP_INIT_METHOD=file://<shared-path>) for SLURM srun
    # without MASTER_ADDR/scontrol plumbing; falls back to env:// (torchrun).
    # Bootstrap backend: nccl (default) or gloo. nccl.ep's get_nccl_comm_from_group
    # broadcasts a unique_id over THIS PG and then builds a standalone NCCL comm.
    # With backend=nccl, torch keeps its own NCCL comm resident alongside nccl.ep's,
    # and the 2nd HT (GIN) dispatch deadlocks against it. backend=gloo bootstraps
    # over host TCP so nccl.ep owns the ONLY NCCL comm (mirrors ep_test's MPI bcast +
    # standalone Communicator.init); aux collectives below then run on CPU tensors.
    bootstrap = os.environ.get("EP_BOOTSTRAP", "nccl")
    init_method = os.environ.get("EP_INIT_METHOD")
    if init_method:
        dist.init_process_group(
            bootstrap,
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
    else:
        dist.init_process_group(bootstrap)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", min(world, 8)))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # gloo collectives need CPU tensors; nccl uses GPU tensors.
    aux_dev = torch.device("cpu") if bootstrap == "gloo" else device
    # ep_bench/ep_test run ALL nccl.ep ops on a dedicated (non-default) stream,
    # isolated from torch's default-stream activity. EP_DEDICATED_STREAM=1 makes
    # the dedicated stream current so fleet/handle/dispatch/combine all use it
    # (the HandleAlgoKnobUserStream + BootstrapConfig.stream read current_stream).
    _ep_stream = None
    if os.environ.get("EP_DEDICATED_STREAM") == "1":
        _ep_stream = torch.cuda.Stream(device)  # keepalive: set_stream does not own it
        torch.cuda.set_stream(_ep_stream)

    per_rank = args.tokens
    hidden, top_k, num_experts = args.hidden, args.top_k, args.experts
    local_n = num_experts // world
    offset = rank * local_n
    is_ht = args.algorithm == "ht"
    algo = EpAlgorithm.HIGH_THROUGHPUT if is_ht else EpAlgorithm.LOW_LATENCY
    # EpLayout exposes only the two LL receive layouts. HT ("fl") always uses the
    # library's FLAT layout internally, so FleetParams.layout is ignored for HT —
    # EXPERT_MAJOR is just an inert placeholder there.
    layout = EpLayout.RANK_MAJOR if args.layout == "rm" else EpLayout.EXPERT_MAJOR

    # Inputs: DISTINCT top-k experts per token (matches ep_bench randperm routing).
    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    x = torch.randn(per_rank, hidden, dtype=torch.bfloat16, device=device, generator=g)
    scores = torch.randn(per_rank, num_experts, device=device, generator=g)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int64)
    topk_w = torch.softmax(
        torch.randn(per_rank, top_k, device=device, generator=g), dim=-1
    )

    # --- byte accounting (ep_bench conventions) --- done BEFORE creating the nccl.ep
    # comm: get_nccl_comm_from_group makes a FRESH NCCL comm bootstrapped over torch's
    # PG; running an HT all_gather on that PG afterwards deadlocks the two NCCL
    # contexts (the single-node HT perf hang). Do the collective while the PG is clean.
    bytes_per_tok = hidden * 2
    if not is_ht:
        disp_bytes = comb_bytes = float(per_rank * top_k * bytes_per_tok)
        recv_bytes = nvl_bytes = rdma_bytes = 0.0
    else:
        topk_ids_aux = topk_ids.to(aux_dev)
        gathered = [torch.empty_like(topk_ids_aux) for _ in range(world)]
        dist.all_gather(gathered, topk_ids_aux)
        my_node = rank // ranks_per_node
        total_recv = rdma_recv = 0
        lo, hi = offset, offset + local_n
        for src in range(world):
            hits = ((gathered[src] >= lo) & (gathered[src] < hi)).any(dim=1)
            cnt = int(hits.sum().item())
            total_recv += cnt
            if src // ranks_per_node != my_node:
                rdma_recv += cnt
        recv_bytes = float(total_recv * bytes_per_tok)
        rdma_bytes = float(rdma_recv * bytes_per_tok)
        nvl_bytes = recv_bytes - rdma_bytes
        disp_bytes = comb_bytes = recv_bytes

    fleet = create_fleet(
        BootstrapConfig(
            world_size=world,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
            nccl_comm=None,
        ),
        FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=per_rank,
            token_hidden_size=hidden,
            dtype_bytes=2,
            algorithm=algo,
            layout=layout,
            # Comm-only benchmark (no expert FFN): weights are a layer-level
            # concern now — FleetParams carries only EP sizing, and create_fleet
            # never touches expert weights, so none are needed here.
        ),
        backend="nccl_ep",
    )

    def _mk_handle():
        return fleet.create_handle(
            HandleParams(topk_ids=topk_ids),
            algo_knobs=[
                HandleAlgoKnobUserStream(
                    stream=torch.cuda.current_stream().cuda_stream
                ),
                HandleAlgoKnobTopKWeights(weights=topk_w),
            ],
        )

    handle = _mk_handle()
    out_buf = torch.empty_like(x)

    # Opt #3: EP_REUSE_PARAMS=1 builds the Dispatch/CombineInputParams wrapper objects
    # ONCE and reuses them every iter, instead of reconstructing per call. Tests how
    # much of the per-call host gap is Python param construction (vs the TVM-FFI call
    # + nccl.ep host path inside dispatch()/combine() themselves).
    _reuse_params = os.environ.get("EP_REUSE_PARAMS") == "1"
    _disp_params = DispatchInputParams(x=[x]) if _reuse_params else None

    def do_dispatch():
        return handle.dispatch(
            _disp_params if _reuse_params else DispatchInputParams(x=[x])
        )

    # EP_SEPARATE_COMBINE_BUF=1: feed a SEPARATE buffer (clone of the dispatch
    # output) as the combine input, like ep_bench's distinct expert_outputs
    # tensor, instead of aliasing the dispatch output directly. Tests whether the
    # 3269 'invalid argument' at HT combine is caused by the in/out aliasing.
    _sep_combine = os.environ.get("EP_SEPARATE_COMBINE_BUF") == "1"

    _comb_params = {}

    def do_combine(d):
        cin = d.expert_tensors.clone() if _sep_combine else d.expert_tensors
        if _reuse_params:
            p = _comb_params.get("p")
            if p is None:
                p = _comb_params["p"] = CombineInputParams(x=[cin], out=out_buf)
            else:
                # Reuse the wrapper object but refresh its input with THIS iter's
                # dispatch output — `cin` is a fresh tensor each call (a new recv
                # buffer without the fast path, or a per-iter clone under
                # EP_SEPARATE_COMBINE_BUF=1); combining the cached `cin` would be stale.
                p.x[0] = cin
            handle.combine(p)
        else:
            handle.combine(CombineInputParams(x=[cin], out=out_buf))
        handle.complete()

    # --- validate: one round-trip, no perf ---
    if args.validate:
        d = do_dispatch()
        do_combine(d)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(
                f"[validate] {args.algorithm}/{args.layout} world={world} "
                f"per_rank={per_rank} dispatch+combine OK",
                flush=True,
            )
        dist.destroy_process_group()
        return 0

    # --- warmup ---
    # HT requires (a) a stream sync between dispatch and combine and after
    # combine, and (b) a per-iteration cross-rank barrier — ep_bench.cu marks
    # both "critical for HT mode": without the barrier ranks desync and the
    # next dispatch's cross-rank GIN op deadlocks. LL tolerates the looser loop,
    # but the barrier is cheap, so apply it uniformly.
    dbg = os.environ.get("EP_DEBUG")
    for _w in range(args.warmup):
        if args.fresh_handle:
            handle = _mk_handle()
        if dbg and rank == 0:
            print(f"[dbg] w{_w} dispatch", flush=True)
        d = do_dispatch()
        if is_ht:
            torch.cuda.synchronize()
        if dbg and rank == 0:
            print(f"[dbg] w{_w} combine", flush=True)
        do_combine(d)
        if is_ht:
            torch.cuda.synchronize()
        if dbg and rank == 0:
            print(f"[dbg] w{_w} barrier", flush=True)
        dist.barrier()
        if dbg and rank == 0:
            print(f"[dbg] w{_w} done", flush=True)
    torch.cuda.synchronize()
    dist.barrier()

    # --- measure: host wall-clock + CUDA-event kernel time, per stage, per iter ---
    # EP_TIMING selects the measurement loop:
    #   baseline (default): per-iter torch.cuda.synchronize() bracketing each stage.
    #     The sync drains the stream, so ev[0] fires on an IDLE GPU and the event
    #     window also captures the host launch-prep gap (Python->TVM-FFI->nccl.ep)
    #     as GPU-idle time -> the "kernel-only" number overstates kernel time.
    #   pipeline (opt #2): NO per-iter host sync. Events are recorded inline with
    #     per-iter event objects and read once after a single sync at the end. With
    #     the stream kept busy, the host runs ahead and issues the next stage's
    #     launch while the current kernel runs, so each event window collapses toward
    #     pure kernel time. The cross-rank barrier is kept for HT lockstep unless
    #     EP_NO_BARRIER=1 (LL tolerates the looser loop; dropping it lets the
    #     dispatch window overlap the prior iter's combine kernel too).
    timing = os.environ.get("EP_TIMING", "baseline")
    no_barrier = os.environ.get("EP_NO_BARRIER") == "1"
    h_disp, h_comb, k_disp, k_comb = [], [], [], []

    if timing == "pipeline":
        evd = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.iters)
        ]
        evc = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.iters)
        ]
        torch.cuda.synchronize()
        dist.barrier()
        t0 = perf_counter()
        for i in range(args.iters):
            evd[i][0].record()
            d = do_dispatch()
            evd[i][1].record()
            evc[i][0].record()
            do_combine(d)
            evc[i][1].record()
            if not no_barrier:
                dist.barrier()
        torch.cuda.synchronize()
        wall_us = (perf_counter() - t0) * 1e6 / args.iters
        k_disp = [a.elapsed_time(b) * 1e3 for a, b in evd]
        k_comb = [a.elapsed_time(b) * 1e3 for a, b in evc]
        # per-stage host wall is not separable without re-introducing a sync; report
        # the combined per-iter wall under dispatch and 0 under combine.
        h_disp = [wall_us]
        h_comb = [0.0]
    else:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        for _ in range(args.iters):
            if args.fresh_handle:
                handle = _mk_handle()  # outside the timed region below
            torch.cuda.synchronize()
            t0 = perf_counter()
            ev[0].record()
            d = do_dispatch()
            ev[1].record()
            torch.cuda.synchronize()
            h_disp.append((perf_counter() - t0) * 1e6)
            k_disp.append(ev[0].elapsed_time(ev[1]) * 1e3)

            t0 = perf_counter()
            ev[2].record()
            do_combine(d)
            ev[3].record()
            torch.cuda.synchronize()
            h_comb.append((perf_counter() - t0) * 1e6)
            k_comb.append(ev[2].elapsed_time(ev[3]) * 1e3)
            # Per-iter cross-rank barrier (outside the timed region) keeps ranks in
            # lockstep — critical for HT (matches ep_bench.cu's per-iter MPI_Barrier).
            dist.barrier()

    # per-rank means, then average across ranks (SUM / world), like ep_bench.
    def rmean(v):
        t = torch.tensor([mean(v)], device=aux_dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() / world

    hd, hc = rmean(h_disp), rmean(h_comb)
    kd, kc = rmean(k_disp), rmean(k_comb)

    def rsum(b):
        t = torch.tensor([b], device=aux_dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() / world  # avg-per-rank bytes

    a_recv, a_nvl, a_rdma = rsum(recv_bytes), rsum(nvl_bytes), rsum(rdma_bytes)

    def bw(b, us):
        return (b / 1e9) / (us / 1e6) if us > 0 else 0.0

    if rank != 0:
        dist.destroy_process_group()
        return 0

    if not is_ht:
        print(f"\n=== Summary (Low Latency, across {world} ranks) ===\n")
        print("--- Host-observed performance ---")
        print(f"Dispatch (BF16):  avg={hd:.2f} us, min={hd:.2f} us, max={hd:.2f} us")
        print(
            f"                  throughput: avg={bw(disp_bytes, hd):.2f} GB/s, "
            f"min={bw(disp_bytes, hd):.2f} GB/s (rank 0), max={bw(disp_bytes, hd):.2f} GB/s (rank 0)"
        )
        print(f"Combine (BF16):   avg={hc:.2f} us, min={hc:.2f} us, max={hc:.2f} us")
        print(
            f"                  throughput: avg={bw(comb_bytes, hc):.2f} GB/s, "
            f"min={bw(comb_bytes, hc):.2f} GB/s (rank 0), max={bw(comb_bytes, hc):.2f} GB/s (rank 0)"
        )
        print(
            f"Total (D+C):      avg={hd + hc:.2f} us, min={hd + hc:.2f} us, max={hd + hc:.2f} us"
        )
        print(
            f"                  throughput: avg={bw(disp_bytes + comb_bytes, hd + hc):.2f} GB/s, "
            f"min=0.00 GB/s (rank 0), max=0.00 GB/s (rank 0)"
        )
        print("\n--- Kernel-only performance ---")
        print(f"Dispatch:    avg={kd:.2f} us, min={kd:.2f} us, max={kd:.2f} us")
        print(
            f"                  throughput: avg={bw(disp_bytes, kd):.2f} GB/s, "
            f"min={bw(disp_bytes, kd):.2f} GB/s, max={bw(disp_bytes, kd):.2f} GB/s"
        )
        print(f"Combine:     avg={kc:.2f} us, min={kc:.2f} us, max={kc:.2f} us")
        print(
            f"                  throughput: avg={bw(comb_bytes, kc):.2f} GB/s, "
            f"min={bw(comb_bytes, kc):.2f} GB/s, max={bw(comb_bytes, kc):.2f} GB/s"
        )
        print(
            f"\nByte counts: dispatch={disp_bytes / 1e6:.2f} MB (BF16), "
            f"combine={comb_bytes / 1e6:.2f} MB (BF16), selections={per_rank * top_k}"
        )
    else:
        print(f"\n=== Summary (High Throughput BF16, across {world} ranks) ===")
        print("NOTE: total time = kernel time + memcpyD2D + misc")
        print("--- BW based on total time ---")
        print(f"Dispatch:    total={hd:.2f} us (min={hd:.2f}, max={hd:.2f})")
        print(
            f"             recv: total_bw={bw(a_recv, hd):.2f}  nvl_bw={bw(a_nvl, hd):.2f}  "
            f"rdma_bw={bw(a_rdma, hd):.2f} GB/s"
        )
        print(f"Combine:     total={hc:.2f} us (min={hc:.2f}, max={hc:.2f})")
        print(
            f"             send: total_bw={bw(a_recv, hc):.2f}  nvl_bw={bw(a_nvl, hc):.2f}  "
            f"rdma_bw={bw(a_rdma, hc):.2f} GB/s"
        )
        print(
            f"Total (D+C): avg={hd + hc:.2f} us, min={hd + hc:.2f} us, max={hd + hc:.2f} us"
        )
        print("\n--- BW based on kernel time ---")
        print(f"Dispatch:    kernel={kd:.2f} us")
        print(
            f"             recv: total_bw={bw(a_recv, kd):.2f}  nvl_bw={bw(a_nvl, kd):.2f}  "
            f"rdma_bw={bw(a_rdma, kd):.2f} GB/s"
        )
        print(f"Combine:     kernel={kc:.2f} us")
        print(
            f"             send: total_bw={bw(a_recv, kc):.2f}  nvl_bw={bw(a_nvl, kc):.2f}  "
            f"rdma_bw={bw(a_rdma, kc):.2f} GB/s"
        )
        print(f"Total (D+C): kernel={kd + kc:.2f} us")
        print(
            f"\nByte counts (per rank avg): total_recv={a_recv / 1e6:.2f} MB, "
            f"rdma_recv={a_rdma / 1e6:.2f} MB"
        )
    sys.stdout.flush()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
