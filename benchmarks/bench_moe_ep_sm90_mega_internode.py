"""Benchmark the SM90 FP8 pull MegaMoE kernel across nodes (NCCL GIN) vs one node.

Modes (one process per GPU under torchrun):

* ``--mode internode``: ``ranks_per_node = LOCAL_WORLD_SIZE`` and the NCCL GIN
  rail-proxy path (several nodes, e.g. 2 x 4 GPUs).  Needs nccl4py with the
  CuTe GIN device API and a matching NCCL (``NCCL_HOME``).
* ``--mode intranode``: the same EP world inside one NVLink domain (NVSHMEM
  symmetric heap), the all-NVLink reference for the same per-rank problem.
* ``--mode nccl_ep_ll | nccl_ep_ht | deepep``: split (non-fused) baselines --
  EP dispatch, the CUTLASS per-tensor FP8 fused MoE over the received rows,
  EP combine -- as three serial stages, ported from the Blackwell wide-EP
  harness (NCCL-EP LL rank-major, NCCL-EP HT flat, DeepEP).  Tokens travel as
  the same per-tensor E4M3 rows the mega kernel dispatches and come back BF16.
  ``nccl_ep_*`` need ``nccl.ep`` (nccl-extensions 0.1.0, whose kernels are
  built against the NCCL 2.30 device API: run it on NCCL 2.30.x, it faults on
  2.32); ``deepep`` needs DeepEP V2 (``deep_ep.ElasticBuffer``, NCCL GIN).

Each point stages fresh routed inputs, runs warmup launches, then times
``iters`` launches of ``hopper_fp8_mega_moe`` (dispatch + FC1 + FC2 + combine +
TopkReduce) with a barrier and a device sync around every sample.  The table
reports, per token count, the median over iterations of the slowest rank.  A
baseline sample spans its first dispatch launch to its combine completion
(CUDA events), with the same barrier / device-side rendezvous per sample; the
stage columns are per-rank medians, maxed over ranks.

    torchrun --nnodes 2 --node_rank $NODE --nproc_per_node 4 --master_addr $M \\
        benchmarks/bench_moe_ep_sm90_mega_internode.py --mode internode
    torchrun --nproc_per_node 8 benchmarks/bench_moe_ep_sm90_mega_internode.py --mode intranode
    torchrun --nnodes 2 ... benchmarks/bench_moe_ep_sm90_mega_internode.py --mode nccl_ep_ll --check
    python benchmarks/bench_moe_ep_sm90_mega_internode.py \\
        --summarize mega.csv ll.csv ht.csv deepep.csv --output-csv compare.csv
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist


def _args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["internode", "intranode", *_BASELINES])
    p.add_argument("--tokens", default="8,64,256,1024,4096")
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=3072)
    p.add_argument(
        "--scale-mode", choices=["per_tensor", "blockwise"], default="per_tensor"
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output-csv", default=None)
    p.add_argument(
        "--summarize",
        nargs="+",
        metavar="CSV",
        help="no GPU: merge the --output-csv files of several modes into one "
        "per-token comparison CSV written to --output-csv",
    )
    p.add_argument(
        "--stamps",
        action="store_true",
        help="internode only: record a per-phase %%globaltimer timeline of one "
        "extra launch (developer stamps compiled into the kernel)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="baselines: before timing, verify one dispatch + combine round trip "
        "with identity experts (out == dequantized x)",
    )
    args = p.parse_args()
    if args.summarize:
        if not args.output_csv:
            p.error("--summarize needs --output-csv")
    elif args.mode is None:
        p.error("--mode is required")
    return args


# Calibrated per-tensor activation dequant scale shared by every mode.
_ACT_SCALE = 8.0 / (0.95 * 448.0)


def _bf16_weights(rank, experts, hidden, intermediate):
    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13 = torch.empty(
        experts, 2 * intermediate, hidden, dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.empty(experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda")
    for e in range(experts):  # per-expert fill keeps the bf16 temporaries small
        w13[e].normal_(generator=g).mul_(hidden**-0.5)
        w2[e].normal_(generator=g).mul_(intermediate**-0.5)
    return w13, w2


def _weights(rank, experts, hidden, intermediate, scale_mode):
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.weights import (
        preprocess_mega_weights,
    )

    act_scale = _ACT_SCALE
    w13, w2 = _bf16_weights(rank, experts, hidden, intermediate)
    out = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
        kind="fp8_e4m3",
        fp8_scale_mode=scale_mode,
        fc1_activation_dequant_scale=act_scale,
        fc2_activation_dequant_scale=act_scale,
    )
    del w13, w2
    torch.cuda.empty_cache()
    return out, act_scale


def _inputs(rank, tokens, hidden, experts_total, topk):
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    hs = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g)
    scores = torch.randn(
        tokens, experts_total, dtype=torch.float32, device="cuda", generator=g
    )
    w, ids = torch.topk(scores, topk, dim=-1)
    return hs, torch.softmax(w, dim=-1).float(), ids.to(torch.int64)


_PHASES = {
    1: "own prep",
    2: "D2 sync",
    3: "D3 puts (CTA0)",
    4: "D5 META (CTA0)",
    5: "D6 sync",
    6: "D7 proxy prep",
    7: "D8 sync",
    8: "D10 node barrier",
    9: "pull pass0 done",
    10: "PAY wait (CTA0)",
    11: "payload flags",
    12: "pull done",
    20: "tail entry (GEMM done)",
    21: "B1",
    22: "node reduce",
    23: "reduce sync",
    25: "RET wait (CTA0)",
    27: "tail done",
}


def _print_timeline(stamps, rank, tokens):
    """Per phase: CTA0 time and the min / max over CTAs, us from the first stamp."""
    raw = stamps.view(256, 64)[:200].to(torch.int64)
    hit = raw != 0
    t0 = int(raw[hit].min())
    rel = ((raw - t0) % (1 << 31)).double() / 1e3
    parts = []
    for p, name in sorted(_PHASES.items()):
        col = hit[:, p]
        if not bool(col.any()):
            continue
        vals = rel[col, p]
        cta0 = f"{rel[0, p]:.1f}" if bool(hit[0, p]) else "-"
        parts.append(f"{name}: cta0 {cta0} min {vals.min():.1f} max {vals.max():.1f}")
    print(f"[timeline rank {rank} T={tokens}] " + " | ".join(parts), flush=True)


# ---------------------------------------------------------------------------
# Split baselines: EP dispatch -> CUTLASS FP8 fused MoE -> EP combine.
#
# Every engine dispatches the per-tensor E4M3 rows [T, H] (the scale is
# static; NCCL-EP's FP8 transport also forwards a 16-byte per-token scale row,
# the counterpart of the mega kernel's x_sf plane), returns the received rows
# with LOCAL expert ids (-1 = not routed here) and their routing weights, and
# combines BF16 rows of the same shape.  The compute stage applies the routing
# weights and sums this rank's experts per row (CUTLASS finalize), so combine
# only sums ranks.
# ---------------------------------------------------------------------------


class _NcclEp:
    """NCCL-EP via ``nccl.ep``: LL rank-major or HT flat (one group per T)."""

    _comm = None  # one NCCL communicator per process, shared by the groups

    def __init__(self, algorithm, tokens, x, topk_idx, topk_w, args, world):
        import nccl.ep as ep
        from nccl.ep.interop.torch import get_nccl_comm_from_group

        self.ep = ep
        self.ht = algorithm == "ht"
        if _NcclEp._comm is None:
            _NcclEp._comm = get_nccl_comm_from_group()
        self.comm = _NcclEp._comm
        h, k = args.hidden, args.top_k
        self.group = ep.Group.create(
            self.comm,
            ep.GroupConfig(
                algorithm=ep.Algorithm.HIGH_THROUGHPUT
                if self.ht
                else ep.Algorithm.LOW_LATENCY,
                num_experts=args.num_experts,
                max_dispatch_tokens_per_rank=tokens,
                max_recv_tokens_per_rank=tokens * world if self.ht else 0,
                max_token_bytes=2 * h,  # BF16 combine rows bound the FP8 rows
                num_topk=k,
            ),
        )
        # The descriptors below do not own their buffers: keep them alive.
        self.x, self.topk_idx, self.topk_w = x, topk_idx, topk_w
        self.out = torch.empty(tokens, h, dtype=torch.bfloat16, device="cuda")
        if self.ht:
            rows = tokens * world
            self.recv_total = torch.zeros(1, dtype=torch.int64, device="cuda")
            self.handle_info = ep.LayoutInfo(
                recv_total_counter=ep.Tensor(self.recv_total),
                recv_topk_idx_kind=ep.ExpertIdKind.LOCAL,
            )
            self.recv_x = torch.empty(rows, h, dtype=x.dtype, device="cuda")
            self.recv_sf = torch.empty(rows, 4, dtype=torch.float32, device="cuda")
            self.recv_w = torch.empty(rows, k, dtype=torch.float32, device="cuda")
            self.recv_idx = torch.empty(rows, k, dtype=torch.int64, device="cuda")
            self.row_ids = torch.arange(rows, device="cuda")
            self.dispatch_info = None
            layout = ep.Layout.FLAT
        else:
            self.recv_x = torch.empty(world, tokens, h, dtype=x.dtype, device="cuda")
            self.recv_sf = torch.empty(
                world, tokens, 4, dtype=torch.float32, device="cuda"
            )
            self.recv_w = torch.empty(
                world, tokens, k, dtype=torch.float32, device="cuda"
            )
            self.recv_idx = torch.empty(
                world, tokens, k, dtype=torch.int32, device="cuda"
            )
            self.src_counts = torch.zeros(world, dtype=torch.int32, device="cuda")
            self.slot_ids = torch.arange(tokens, device="cuda")
            self.handle_info = None
            self.dispatch_info = ep.LayoutInfo(
                src_rank_counters=ep.Tensor(self.src_counts),
                recv_topk_idx_kind=ep.ExpertIdKind.LOCAL,
            )
            layout = ep.Layout.RANK_MAJOR
        # Descriptors over persistent buffers are built once (host overhead
        # stays out of the comparison, as in the Blackwell harness).
        T = ep.Tensor
        self.t_idx = T(topk_idx)
        # FP8 tokens travel only under the scale-forwarding recipe.
        self.sf = torch.full((tokens, 4), _ACT_SCALE, device="cuda")
        self.d_cfg = ep.DispatchConfig(
            quantization_recipe=ep.DispatchQuantizationRecipe.FWD
        )
        self.d_in = ep.DispatchInputs(
            tokens=T(x), topk_weights=T(topk_w), scales=T(self.sf)
        )
        self.d_out = ep.DispatchOutputs(
            tokens=T(self.recv_x),
            topk_weights=T(self.recv_w),
            scales=T(self.recv_sf),
            topk_idx=T(self.recv_idx),
        )
        self.c_out = ep.CombineOutputs(tokens=T(self.out))
        self.handle = self.group.create_handle(
            layout, self.t_idx, layout_info=self.handle_info, stream=self._stream()
        )
        self.align_buf = torch.zeros(1, dtype=torch.float32, device="cuda")

    @staticmethod
    def _stream():
        return torch.cuda.current_stream().cuda_stream

    def dispatch(self):
        s = self._stream()
        if self.ht:
            # HT routing metadata is per-step work (handle update), timed as
            # part of dispatch like the Blackwell harness default.
            self.handle.update(self.t_idx, layout_info=self.handle_info, stream=s)
        self.handle.dispatch(
            self.d_in,
            self.d_out,
            layout_info=self.dispatch_info,
            config=self.d_cfg,
            stream=s,
        )
        self.handle.complete(stream=s)
        h, k = self.recv_x.shape[-1], self.recv_idx.shape[-1]
        if self.ht:
            valid = self.row_ids < self.recv_total
        else:
            valid = (self.slot_ids[None, :] < self.src_counts[:, None]).reshape(-1)
        return (
            self.recv_x.view(-1, h),
            self.recv_idx.view(-1, k),
            self.recv_w.view(-1, k),
            valid,
        )

    def combine(self, y):
        # The compute output is a fresh tensor per call, so its descriptor is too.
        s = self._stream()
        c_in = self.ep.CombineInputs(tokens=self.ep.Tensor(y.view(self.recv_x.shape)))
        self.handle.combine(c_in, self.c_out, stream=s)
        self.handle.complete(stream=s)
        return self.out

    def align(self):
        from nccl.core import NcclRedOp

        self.comm.allreduce(
            self.align_buf, self.align_buf, NcclRedOp.SUM, stream=self._stream()
        )

    def close(self):
        torch.cuda.synchronize()
        self.handle.destroy()
        self.group.destroy()


class _DeepEp:
    """DeepEP V2 ``ElasticBuffer`` (NCCL GIN), non-expanded layout."""

    def __init__(self, tokens, x, topk_idx, topk_w, args, world):
        import deep_ep

        self.buf = deep_ep.ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=tokens,
            hidden=args.hidden,
            num_topk=args.top_k,
            use_fp8_dispatch=True,
            explicitly_destroy=True,
        )
        self.num_experts = args.num_experts
        self.x, self.topk_idx, self.topk_w = x, topk_idx, topk_w
        self.row_ids = torch.arange(tokens * world, device="cuda")
        self.handle = None

    def dispatch(self):
        recv_x, recv_idx, recv_w, self.handle, _ = self.buf.dispatch(
            self.x,
            topk_idx=self.topk_idx,
            topk_weights=self.topk_w,
            num_experts=self.num_experts,
            do_cpu_sync=False,
            do_handle_copy=False,
        )
        # Without a CPU sync the outputs are sized for the worst case; the
        # received rows are the prefix counted on device by the handle.
        total = self.handle.psum_num_recv_tokens_per_scaleup_rank[-1]
        valid = self.row_ids[: recv_idx.shape[0]] < total
        return recv_x, recv_idx, recv_w, valid

    def combine(self, y):
        return self.buf.combine(y, handle=self.handle)[0]

    def align(self):
        self.buf.barrier(use_comm_stream=False)

    def close(self):
        torch.cuda.synchronize()
        self.buf.destroy()


class _Fp8Experts:
    """This rank's experts as the CUTLASS per-tensor FP8 fused MoE (SM90)."""

    def __init__(self, rank, experts, args):
        from flashinfer.fused_moe.api import (
            CutlassFp8PerTensorConfig,
            MoEWeightPack as FusedWeightPack,
            SwiGLU,
        )

        w13, w2 = _bf16_weights(rank, experts, args.hidden, args.intermediate)
        self.activation = SwiGLU(limit=10.0)
        view = CutlassFp8PerTensorConfig.prepare_weights(
            w13,
            w2,
            hidden_states_scale_global=1.0 / _ACT_SCALE,
            intermediate_scale_global=1.0 / _ACT_SCALE,
            num_local_experts=experts,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            activation=self.activation,
        )
        del w13, w2
        torch.cuda.empty_cache()
        self.pack = FusedWeightPack()
        self.pack.prepare_for("cutlass_fp8_per_tensor", view)
        self.experts, self.args = experts, args
        self.layer = None

    def build(self, max_rows):
        from flashinfer.fused_moe.api import (
            BackendOptions,
            CutlassFp8PerTensorConfig,
            ExecutionConfig,
            ExpertConfig,
            MoEConfig,
            QuantConfig,
            QuantFormat,
            RoutingConfig,
        )
        from flashinfer.fused_moe.layer import MoELayer

        fp8 = QuantFormat.FP8PerTensor
        self.layer = MoELayer(
            MoEConfig(
                routing=RoutingConfig(num_experts=self.experts, top_k=self.args.top_k),
                quant=QuantConfig(weight=fp8, activation=fp8),
                experts=ExpertConfig(
                    intermediate_size=self.args.intermediate,
                    local_expert_offset=0,
                    local_num_experts=self.experts,
                ),
                activation=self.activation,
                backend=BackendOptions(candidates=(CutlassFp8PerTensorConfig(),)),
                execution=ExecutionConfig(tune_max_num_tokens=max_rows),
            )
        )

    def __call__(self, x, ids, w):
        from flashinfer.fused_moe.api import MoEActivationPack

        return self.layer(
            MoEActivationPack(
                hidden_states_q=x,
                hidden_states_scale=None,
                topk_ids=ids,
                topk_weights=w,
            ),
            self.pack,
        )


def _local_routing(recv_idx, recv_w, valid):
    """Received LOCAL ids/weights -> runner routing; foreign picks and rows past
    the received count become id -1 (skipped by the runner) with weight 0."""
    keep = recv_idx >= 0
    if valid is not None:
        keep &= valid[:, None]
    ids = torch.where(keep, recv_idx, -1).to(torch.int32)
    return ids, torch.where(keep, recv_w.float(), 0.0)


def _identity_check(engine, x, rank, tokens, local_experts):
    """dispatch -> y = dequant(x) * sum(local weights) -> combine == dequant(x)."""
    recv_x, recv_idx, recv_w, valid = engine.dispatch()
    ids, ws = _local_routing(recv_idx, recv_w, valid)
    if not bool(((ids >= -1) & (ids < local_experts)).all()):
        raise SystemExit(f"[check] rank {rank} T={tokens}: received ids not LOCAL")
    y = (recv_x.float() * _ACT_SCALE * ws.sum(-1, keepdim=True)).to(torch.bfloat16)
    out = engine.combine(y).float()
    torch.cuda.synchronize()
    ref = x.float() * _ACT_SCALE
    err = ((out - ref).norm() / ref.norm()).item()
    if not err < 1e-2:
        raise SystemExit(f"[check] rank {rank} T={tokens}: rel_l2 {err:.3e}")
    return err


_BASELINES = ("nccl_ep_ll", "nccl_ep_ht", "deepep")


def _run_baselines(args, rank, world, per_node):
    """Time the split baselines; returns the result rows."""
    if args.scale_mode != "per_tensor":
        raise SystemExit("the split baselines run the per_tensor FP8 recipe only")
    dist.init_process_group("gloo")
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    # One rank at a time: the JIT file lock races across nodes sharing an NFS
    # cache.  The module is cached per process for the runners built below.
    for r in range(world):
        if r == rank:
            get_cutlass_fused_moe_module("90")
        dist.barrier()
    experts = _Fp8Experts(rank, args.num_experts // world, args)
    rows = []
    for tokens in [int(t) for t in args.tokens.split(",")]:
        hs, w, idx = _inputs(rank, tokens, args.hidden, args.num_experts, args.top_k)
        # The per-tensor E4M3 rows the mega kernel stages, quantized up front.
        x = (hs.float() / _ACT_SCALE).to(torch.float8_e4m3fn)
        if args.mode == "deepep":
            engine = _DeepEp(tokens, x, idx, w, args, world)
        else:
            engine = _NcclEp(args.mode[-2:], tokens, x, idx, w, args, world)
        experts.build(tokens * world)
        check_err = (
            _identity_check(engine, x, rank, tokens, experts.experts)
            if args.check
            else None
        )

        def call(ev=None):
            recv_x, recv_idx, recv_w, valid = engine.dispatch()
            if ev is not None:
                ev[1].record()
            ids, ws = _local_routing(recv_idx, recv_w, valid)
            y = experts(recv_x, ids, ws)
            if ev is not None:
                ev[2].record()
            return engine.combine(y)

        # Tune the CUTLASS tactics for this row count first: untuned, the
        # fallback tactic is several times slower.
        with autotune(True):
            call()
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        dist.barrier()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        samples = {"total": [], "dispatch": [], "compute": [], "combine": []}
        for _ in range(args.iters):
            dist.barrier()
            torch.cuda.synchronize()
            engine.align()
            ev[0].record()
            call(ev)
            ev[3].record()
            torch.cuda.synchronize()
            samples["total"].append(ev[0].elapsed_time(ev[3]) * 1e3)
            samples["dispatch"].append(ev[0].elapsed_time(ev[1]) * 1e3)
            samples["compute"].append(ev[1].elapsed_time(ev[2]) * 1e3)
            samples["combine"].append(ev[2].elapsed_time(ev[3]) * 1e3)
        meds = torch.tensor(
            [statistics.median(samples[s]) for s in samples], dtype=torch.float64
        )
        gathered = [torch.zeros_like(meds) for _ in range(world)]
        dist.all_gather(gathered, meds)
        per_rank = torch.stack(gathered)
        row = dict(
            mode=args.mode,
            world=world,
            per_node=per_node,
            tokens=tokens,
            scale_mode=args.scale_mode,
            max_rank_us=float(per_rank[:, 0].max()),
            min_rank_us=float(per_rank[:, 0].min()),
            dispatch_us=float(per_rank[:, 1].max()),
            compute_us=float(per_rank[:, 2].max()),
            combine_us=float(per_rank[:, 3].max()),
        )
        rows.append(row)
        if rank == 0:
            chk = "" if check_err is None else f" [check rel_l2 {check_err:.1e}]"
            print(
                f"BENCH {args.mode} W={world} R={per_node} T={tokens} {args.scale_mode}: "
                f"max-rank median {row['max_rank_us']:.1f} us (min-rank "
                f"{row['min_rank_us']:.1f} us) | dispatch {row['dispatch_us']:.1f} "
                f"compute {row['compute_us']:.1f} combine {row['combine_us']:.1f}{chk}",
                flush=True,
            )
        dist.barrier()
        engine.close()
        engine = None
        torch.cuda.empty_cache()
    return rows


def main():
    args = _args()
    if args.summarize:
        _summarize(args.summarize, args.output_csv)
        return
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    torch.cuda.set_device(local)
    if args.mode in _BASELINES:
        rows = _run_baselines(args, rank, world, per_node)
        _write_csv(args, rank, rows)
        dist.barrier()
        return

    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        get_symm_buffer_for_hopper_fp8_mega_moe,
        hopper_fp8_mega_moe,
    )

    stamps = None
    if args.mode == "internode":
        if world <= per_node:
            raise SystemExit("--mode internode needs a multi-node torchrun job")
        dist.init_process_group("gloo")
        gin_kwargs = dict(ranks_per_node=per_node, enable_nccl_gin=True)
        if args.stamps:
            stamps = torch.zeros(256 * 64, dtype=torch.int32, pin_memory=True)
            os.environ["MEGA_GIN_DEBUG_ADDR"] = str(stamps.data_ptr())
    else:
        if world != per_node:
            raise SystemExit("--mode intranode runs inside one node")
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
            init_dist,
        )

        init_dist()
        gin_kwargs = {}

    experts_per_rank = args.num_experts // world
    (l1, l2), act_scale = _weights(
        rank, experts_per_rank, args.hidden, args.intermediate, args.scale_mode
    )
    rows = []
    for tokens in [int(t) for t in args.tokens.split(",")]:
        symm = get_symm_buffer_for_hopper_fp8_mega_moe(
            args.num_experts,
            tokens,
            args.top_k,
            args.hidden,
            args.intermediate,
            rank,
            world,
            kind="fp8_e4m3",
            fp8_scale_mode=args.scale_mode,
            gate_up_clamp=10.0,
            **gin_kwargs,
        )
        cfg = symm._frontend.config
        hs, tw, ti = _inputs(rank, tokens, args.hidden, args.num_experts, args.top_k)
        stage_mega_moe_inputs(
            hs,
            tw,
            ti,
            symm.x,
            symm.x_sf,
            symm.topk_idx,
            symm.topk_weights,
            kind="fp8_e4m3",
            fp8_scale_mode=args.scale_mode,
            fc1_activation_dequant_scale=act_scale,
        )
        y = torch.empty(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")

        def call():
            hopper_fp8_mega_moe(
                y,
                l1,
                l2,
                symm,
                num_tokens=tokens,
                gate_up_clamp=10.0,
                fast_math=True,
            )

        align_buf = torch.zeros(1, dtype=torch.float32, device="cuda")

        def align():
            # Device-side rendezvous right before the timed launch, so no rank
            # spends its sample waiting inside the kernel for a late peer (a
            # host barrier leaves launch skew of the order of the kernel).
            if args.mode == "internode":
                from nccl.core import NcclRedOp

                symm._gin_arena.comm.allreduce(
                    align_buf,
                    align_buf,
                    NcclRedOp.SUM,
                    stream=torch.cuda.current_stream().cuda_stream,
                )
            else:
                dist.all_reduce(align_buf)

        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            dist.barrier()
            torch.cuda.synchronize()
            align()
            start.record()
            call()
            stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop) * 1e3)
        med = torch.tensor([statistics.median(samples)], dtype=torch.float64)
        gathered = [torch.zeros_like(med) for _ in range(world)]
        dist.all_gather(gathered, med)
        rank_meds = [float(g.item()) for g in gathered]
        row = dict(
            mode=args.mode,
            world=world,
            per_node=per_node,
            tokens=tokens,
            scale_mode=args.scale_mode,
            swap_ab=cfg.swap_ab,
            tile=cfg.mma_tiler_mnk,
            token_back=cfg.resolved_token_back_mode,
            max_rank_us=max(rank_meds),
            min_rank_us=min(rank_meds),
        )
        rows.append(row)
        if rank == 0:
            print(
                f"BENCH {args.mode} W={world} R={per_node} T={tokens} {args.scale_mode} "
                f"swap_ab={cfg.swap_ab} tile={cfg.mma_tiler_mnk} tb={cfg.resolved_token_back_mode}: "
                f"max-rank median {max(rank_meds):.1f} us (min-rank {min(rank_meds):.1f} us)",
                flush=True,
            )
        if stamps is not None:
            dist.barrier()
            torch.cuda.synchronize()
            align()
            stamps.zero_()
            call()
            torch.cuda.synchronize()
            _print_timeline(stamps, rank, tokens)
        torch.cuda.synchronize()
        dist.barrier()
        symm.destroy()
        torch.cuda.empty_cache()
    _write_csv(args, rank, rows)
    dist.barrier()


_FUSED = ("internode", "intranode")


def _summarize(inputs, output):
    """One row per (world, per_node, scale_mode, tokens) across the given runs:
    every mode's max-rank median, the baselines' stage breakdown, the fastest
    split baseline and each fused mode's speedup over it (> 1 = fused faster)."""
    import csv

    runs = {}
    for path in inputs:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (int(r["world"]), int(r["per_node"]), r["scale_mode"])
                runs.setdefault(key + (int(r["tokens"]),), {})[r["mode"]] = r
    order = ["internode", "intranode", *_BASELINES]
    modes = [m for m in order if any(m in v for v in runs.values())]
    rows = []
    for (world, per_node, scale_mode, tokens), by_mode in sorted(runs.items()):
        row = dict(world=world, per_node=per_node, scale_mode=scale_mode, tokens=tokens)
        for m in modes:
            r = by_mode.get(m)
            row[f"{m}_us"] = f"{float(r['max_rank_us']):.1f}" if r else ""
        for m in modes:
            if m in _BASELINES:
                r = by_mode.get(m)
                for stage in ("dispatch", "compute", "combine"):
                    row[f"{m}_{stage}_us"] = (
                        f"{float(r[stage + '_us']):.1f}" if r else ""
                    )
        timed = {
            m: float(by_mode[m]["max_rank_us"]) for m in _BASELINES if m in by_mode
        }
        best = min(timed, key=timed.get) if timed else ""
        row["best_baseline"] = best
        row["best_baseline_us"] = f"{timed[best]:.1f}" if best else ""
        for m in modes:
            if m in _FUSED:
                ok = best and m in by_mode
                row[f"{m}_speedup_vs_best"] = (
                    f"{timed[best] / float(by_mode[m]['max_rank_us']):.3f}"
                    if ok
                    else ""
                )
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows x {len(fields)} columns to {output}")


def _write_csv(args, rank, rows):
    if rank == 0 and args.output_csv and rows:
        import csv

        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
