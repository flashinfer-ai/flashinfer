"""Benchmark the SM90 FP8 pull MegaMoE kernel across nodes (NCCL GIN) vs one node.

Modes (one process per GPU under torchrun):

* ``--mode internode``: ``ranks_per_node = LOCAL_WORLD_SIZE`` and the NCCL GIN
  rail-proxy path (several nodes, e.g. 2 x 4 GPUs).  Needs nccl4py with the
  CuTe GIN device API and a matching NCCL (``NCCL_HOME``).
* ``--mode intranode``: the same EP world inside one NVLink domain (NVSHMEM
  symmetric heap), the all-NVLink reference for the same per-rank problem.
* ``--mode nccl_ep_ll | nccl_ep_ht | deepep``: split (non-fused) baselines --
  EP dispatch, the experts over the received rows, EP combine -- as serial
  stages, ported from the Blackwell wide-EP harness (NCCL-EP LL rank-major,
  NCCL-EP HT flat, DeepEP).  Tokens travel as the same E4M3 rows (and 1x128
  block scales under blockwise) the mega kernel dispatches and come back BF16.
  The experts (``--baseline-gemm``) default to the mega kernel's own
  standalone fused FC12 -- FP8 in, BF16 out, like the Blackwell harness' FC12
  baseline -- behind a Triton pack / pre-reduce handoff; ``cutlass`` uses
  flashinfer's SM90 CUTLASS FP8 GEMMs instead, also FP8 in: the per-tensor
  fused MoE, or under blockwise the CUTLASS / DeepGEMM block-scale grouped
  GEMM (FC1 fused with SwiGLU + requant, then FC2) behind the same handoff.
  ``nccl_ep_*`` need ``nccl.ep`` (nccl-extensions 0.1.0, whose kernels are
  built against the NCCL 2.30 device API: run it on NCCL 2.30.x, it faults on
  2.32); ``deepep`` needs DeepEP V2 (``deep_ep.ElasticBuffer``, NCCL GIN).
  With a home directory shared by the nodes, point ``TRITON_CACHE_DIR`` at
  node-local storage (concurrent ranks race on the shared Triton cache).

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
import functools
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
        "--baseline-gemm",
        choices=["fc12", "cutlass"],
        default="fc12",
        help="baselines' expert compute, FP8 in either way: the mega kernel's "
        "standalone fused FC12 (as the Blackwell harness' FC12 baseline) or "
        "flashinfer's SM90 CUTLASS FP8 GEMMs (per-tensor fused MoE / block-scale "
        "grouped GEMM)",
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
# Split baselines: EP dispatch -> expert compute -> EP combine.
#
# Every engine dispatches the E4M3 rows [T, H] the mega kernel stages for the
# same --scale-mode, plus a per-token scale row: the [T, H/128] fp32 1x128
# block scales under blockwise, a 16-byte filler row under per_tensor (the
# scale is static, but NCCL-EP carries FP8 tokens only under its
# scale-forwarding recipe).  It returns the received rows with LOCAL expert
# ids (-1 = not routed here) and their routing weights, and combines BF16 rows
# of the same shape.  The experts apply the routing weights and sum this
# rank's experts per row, so combine only sums ranks.
# ---------------------------------------------------------------------------


class _NcclEp:
    """NCCL-EP via ``nccl.ep``: LL rank-major or HT flat (one group per T)."""

    _comm = None  # one NCCL communicator per process, shared by the groups

    def __init__(self, algorithm, tokens, x, sf, topk_idx, topk_w, args, world):
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
        self.blockwise = sf is not None
        if sf is None:
            sf = torch.full((tokens, 4), _ACT_SCALE, device="cuda")
        # The descriptors below do not own their buffers: keep them alive.
        self.x, self.sf, self.topk_idx, self.topk_w = x, sf, topk_idx, topk_w
        self.out = torch.empty(tokens, h, dtype=torch.bfloat16, device="cuda")
        if self.ht:
            lead = (tokens * world,)
            self.recv_total = torch.zeros(1, dtype=torch.int64, device="cuda")
            self.handle_info = ep.LayoutInfo(
                recv_total_counter=ep.Tensor(self.recv_total),
                recv_topk_idx_kind=ep.ExpertIdKind.LOCAL,
            )
            self.row_ids = torch.arange(lead[0], device="cuda")
            self.dispatch_info = None
            layout = ep.Layout.FLAT
        else:
            lead = (world, tokens)
            self.src_counts = torch.zeros(world, dtype=torch.int32, device="cuda")
            self.slot_ids = torch.arange(tokens, device="cuda")
            self.handle_info = None
            self.dispatch_info = ep.LayoutInfo(
                src_rank_counters=ep.Tensor(self.src_counts),
                recv_topk_idx_kind=ep.ExpertIdKind.LOCAL,
            )
            layout = ep.Layout.RANK_MAJOR
        self.recv_x = torch.empty(*lead, h, dtype=x.dtype, device="cuda")
        self.recv_sf = torch.empty(*lead, sf.shape[1], dtype=sf.dtype, device="cuda")
        self.recv_w = torch.empty(*lead, k, dtype=torch.float32, device="cuda")
        idx_dtype = torch.int64 if self.ht else torch.int32
        self.recv_idx = torch.empty(*lead, k, dtype=idx_dtype, device="cuda")
        # Descriptors over persistent buffers are built once (host overhead
        # stays out of the comparison, as in the Blackwell harness).
        T = ep.Tensor
        self.t_idx = T(topk_idx)
        self.d_cfg = ep.DispatchConfig(
            quantization_recipe=ep.DispatchQuantizationRecipe.FWD
        )
        self.d_in = ep.DispatchInputs(tokens=T(x), topk_weights=T(topk_w), scales=T(sf))
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
        sf = self.recv_sf.view(-1, self.recv_sf.shape[-1]) if self.blockwise else None
        return (
            self.recv_x.view(-1, h),
            sf,
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

    def __init__(self, tokens, x, sf, topk_idx, topk_w, args, world):
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
        # FP32 scale factors ride along as DeepEP's FP8 (x, sf) pair.
        self.x = x if sf is None else (x, sf)
        self.topk_idx, self.topk_w = topk_idx, topk_w
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
        recv_x, recv_sf = recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        # Without a CPU sync the outputs are sized for the worst case; the
        # received rows are the prefix counted on device by the handle.
        total = self.handle.psum_num_recv_tokens_per_scaleup_rank[-1]
        valid = self.row_ids[: recv_idx.shape[0]] < total
        return recv_x, recv_sf, recv_idx, recv_w, valid

    def combine(self, y):
        return self.buf.combine(y, handle=self.handle)[0]

    def align(self):
        self.buf.barrier(use_comm_stream=False)

    def close(self):
        torch.cuda.synchronize()
        self.buf.destroy()


def _dequant(x, sf):
    """E4M3 rows (+ [M, H/128] fp32 1x128 block scales) -> BF16 (--check)."""
    if sf is None:
        return x.to(torch.bfloat16) * _ACT_SCALE
    m, h = x.shape
    blocks = x.view(m, sf.shape[1], -1).to(torch.bfloat16)
    return blocks.mul_(sf.to(torch.bfloat16).unsqueeze(-1)).view(m, h)


class _CutlassPerTensorExperts:
    """This rank's experts as flashinfer's SM90 CUTLASS per-tensor FP8 fused
    MoE: the received E4M3 rows go in as they are."""

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

    def build(self, tokens, rows):
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
                execution=ExecutionConfig(tune_max_num_tokens=rows),
            )
        )

    def handoff(self, recv_x, recv_sf, recv_idx, recv_w, valid):
        self._ids, self._ws = _local_routing(recv_idx, recv_w, valid)
        self._act = recv_x

    def compute(self):
        from flashinfer.fused_moe.api import MoEActivationPack

        self._y = self.layer(
            MoEActivationPack(
                hidden_states_q=self._act,
                hidden_states_scale=None,
                topk_ids=self._ids,
                topk_weights=self._ws,
            ),
            self.pack,
        )

    def reduce(self):
        return self._y  # finalize already summed this rank's experts per row


@functools.cache
def _handoff_triton_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def count(
        idx,
        valid,
        counts,
        n,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_VALID: tl.constexpr,
    ):
        # Per-expert counts of the local picks (LOCAL id >= 0) of received rows.
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        pm = p < n
        e = tl.load(idx + p, mask=pm, other=-1).to(tl.int32)
        ok = pm & (e >= 0)
        if HAS_VALID:
            ok = ok & (tl.load(valid + p // K, mask=pm, other=0) != 0)
        tl.atomic_add(counts + tl.where(ok, e, 0), 1, mask=ok)

    @triton.jit
    def pack(
        idx,
        valid,
        counts,
        cursor,
        offs,
        slots,
        x,
        sf,
        pool_x,
        pool_sf,
        SF_STRIDE,
        SF_P,
        E,
        K: tl.constexpr,
        TILE: tl.constexpr,
        E_BLOCK: tl.constexpr,
        WORDS: tl.constexpr,
        WBLOCK: tl.constexpr,
        SF_COLS: tl.constexpr,
        SF_BLOCK: tl.constexpr,
        HAS_VALID: tl.constexpr,
        DG_LAYOUT: tl.constexpr,
    ):
        # One program per received (row, pick): claim a slot in the pick's
        # expert segment (starts recomputed in registers from the E counts)
        # and copy the row (+ its block scales) there.  FC12 layout: segments
        # padded to the GEMM token tile, scales row-major, offs = int32
        # cumulative ends.  DG_LAYOUT (CUTLASS / DeepGEMM block-scale grouped
        # GEMM): unpadded segments (TILE=1), offs = int64 [E+1] starts, scales
        # column-major [H/128, SF_P] at deep_gemm::compute_padded_offset.
        pick = tl.program_id(0)
        row = pick // K
        e = tl.load(idx + pick).to(tl.int32)
        ok = e >= 0
        if HAS_VALID:
            ok = ok & (tl.load(valid + row) != 0)
        es = tl.where(ok, e, 0)
        r = tl.arange(0, E_BLOCK)
        rm = r < E
        c = tl.load(counts + r, mask=rm, other=0)
        padded = (c + TILE - 1) // TILE * TILE
        start = tl.sum(tl.where(r == es, tl.cumsum(padded, 0) - padded, 0), 0)
        if pick == 0:
            if DG_LAYOUT:
                tl.store(offs, 0)
                tl.store(offs + 1 + r, tl.cumsum(c, 0).to(tl.int64), mask=rm)
            else:
                tl.store(offs + r, tl.cumsum(c, 0), mask=rm)
        claim = tl.atomic_add(cursor + es, 1, mask=ok)
        slot = tl.where(ok, start + claim, -1)
        tl.store(slots + pick, slot)
        src = row.to(tl.int64)
        dst = tl.maximum(slot, 0).to(tl.int64)
        w = tl.arange(0, WBLOCK)
        wm = (w < WORDS) & ok
        tl.store(
            pool_x + dst * WORDS + w, tl.load(x + src * WORDS + w, mask=wm), mask=wm
        )
        if SF_COLS > 0:
            s = tl.arange(0, SF_BLOCK)
            sm = (s < SF_COLS) & ok
            v = tl.load(sf + src * SF_STRIDE + s, mask=sm)
            if DG_LAYOUT:
                col = ((start + es * 31) // 32 * 32 + claim).to(tl.int64)
                tl.store(pool_sf + s.to(tl.int64) * SF_P + col, v, mask=sm)
            else:
                tl.store(pool_sf + dst * SF_COLS + s, v, mask=sm)

    @triton.jit
    def reduce(
        slots,
        w,
        y,
        out,
        K: tl.constexpr,
        H: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # out[row] = sum_k w[row, k] * y[slot[row, k]] in FP32 over the packed
        # picks; rows without a local pick come out zero.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        cm = cols < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(K):
            s = tl.load(slots + row * K + k)
            wk = tl.load(w + row * K + k)
            src = tl.maximum(s, 0).to(tl.int64) * H + cols
            v = tl.load(y + src, mask=cm & (s >= 0), other=0.0).to(tl.float32)
            acc += v * tl.where(s >= 0, wk, 0.0)
        tl.store(out + row * H + cols, acc.to(tl.bfloat16), mask=cm)

    return count, pack, reduce


class _Fc12Experts:
    """This rank's experts as the mega kernel's own standalone fused
    FC1 + SwiGLU + FC2 (``kernel_fp8_glu_fc12``): E4M3 rows (+ 1x128 scales)
    in, BF16 out -- the Hopper counterpart of the Blackwell harness' NVFP4
    ``BlockScaledSwapAbFc12Kernel`` baseline.  Same weights, scale recipe and
    per-T tile heuristic as the mega kernel.

    handoff (one memset + two Triton launches, nothing leaves the device):
    per-expert counts of the received local picks, then a pack that places
    each pick's row in its expert's token-tile-padded segment (slot claimed by
    an atomic, as in the Blackwell handoff).  reduce: a Triton pre-reduce sums
    each received row's picks with their routing weights."""

    def __init__(self, rank, experts, args):
        (self.l1, self.l2), _ = _weights(
            rank, experts, args.hidden, args.intermediate, args.scale_mode
        )
        self.blockwise = args.scale_mode == "blockwise"
        self.experts, self.args = experts, args

    def build(self, tokens, rows):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as cutlass_utils

        import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel  # noqa: F401  (src/ on sys.path)
        from common.megamoe_constants import (
            Fp8E8M0SfVecSize,
            Fp8Fc2ActivationScaleK,
            SfPaddingBlock,
        )
        from moe_hopper_fp8.heuristic_config import select_heuristic_config

        args, e = self.args, self.experts
        h, k, inter = args.hidden, args.top_k, args.intermediate
        c = select_heuristic_config(args.scale_mode, tokens).config
        if c.swap_ab:
            from moe_hopper_fp8.kernel_fp8_glu_fc12_swapab import (
                Sm90SwapABSwigluFp8Fc12Kernel as Kernel,
            )
        else:
            from moe_hopper_fp8.kernel_fp8_glu_fc12 import (
                Sm90SwigluFp8Fc12Kernel as Kernel,
            )
        self.tile = c.mma_tiler_mnk[1] if c.swap_ab else c.mma_tiler_mnk[0]
        clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
            c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        )
        kernel = Kernel(
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            use_2cta_instrs=False,
            group_hint=c.group_hint if c.group_hint is not None else clusters,
            token_padding_block=self.tile,
            sf_padding_block=SfPaddingBlock,
            static_expert_shape=(e, 2 * inter, h),
            sf_vec_size=Fp8E8M0SfVecSize,
            ab_dtype=cutlass.Float8E4M3FN,
            fp8_scale_mode=args.scale_mode,
            fp8_accum_mode=c.accum_mode,
            pingpong=c.pingpong,
            gate_up_clamp=10.0,
            tail_split_pairs=c.tail_split_pairs,
        )
        self.config = c
        # Expert pool sized for the worst case (every received pick local).
        pool = rows * k + e * self.tile
        dev = "cuda"
        self.pool_x = torch.empty(pool, h, dtype=torch.float8_e4m3fn, device=dev)
        if self.blockwise:
            self.pool_sf = torch.empty(pool, h // 128, dtype=torch.float32, device=dev)
        else:
            # Per-tensor: the unit E8M0 SF plane (layout-free: every byte 127).
            cols = (h // Fp8E8M0SfVecSize + 3) // 4 * 4
            self.pool_sf = torch.full(
                (pool + e * SfPaddingBlock, cols), 127, dtype=torch.uint8, device=dev
            ).view(torch.float8_e8m0fnu)
        fc1_w, fc1_sf, fc1_act, fc1_wdq = self.l1
        fc2_w, fc2_sf, fc2_act, fc2_wdq = self.l2
        ones_e = torch.ones(e, dtype=torch.float32, device=dev)
        ones_1 = torch.ones(1, dtype=torch.float32, device=dev)
        ws = torch.zeros(
            kernel.get_workspace_size_in_bytes(self.pool_x, fc1_w),
            dtype=torch.uint8,
            device=dev,
        )
        # Workspace partition, in get_workspace_size_in_bytes order.
        o = pool * inter
        fc1_out = ws[:o].view(torch.float8_e4m3fn).reshape(pool, inter)
        if self.blockwise:
            sf_cols = inter // Fp8Fc2ActivationScaleK
            stride = (sf_cols + 3) // 4 * 4
            n = pool * stride * 4
            fc1_out_sf = ws[o : o + n].view(torch.float32).reshape(pool, stride)
            fc1_out_sf = fc1_out_sf[:, :sf_cols]
        else:
            sf_rows = pool + e * SfPaddingBlock
            stride = (inter // Fp8E8M0SfVecSize + 3) // 4 * 4
            n = sf_rows * stride
            fc1_out_sf = (
                ws[o : o + n].view(torch.float8_e8m0fnu).reshape(sf_rows, stride)
            )
        o += n
        slots_n = (pool + self.tile - 1) // self.tile + e
        self.done_counter = ws[o : o + slots_n * 4].view(torch.int32)
        self.fc2_out = torch.empty(pool, h, dtype=torch.bfloat16, device=dev)
        self.topk_scores = torch.ones(pool, dtype=torch.float32, device=dev)
        self.offs = torch.zeros(e, dtype=torch.int32, device=dev)
        self.counters = torch.zeros(2 * e, dtype=torch.int32, device=dev)
        self.slots = torch.empty(rows * k, dtype=torch.int32, device=dev)
        self.out = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
        self._ws = ws  # backs the partition views

        def to_cute(t, align=16):
            ct = cutlass_torch.from_dlpack(t, assumed_align=align)
            return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))

        self.kw = dict(
            activation=to_cute(self.pool_x),
            fc1_weight=to_cute(fc1_w),
            activation_sf=to_cute(self.pool_sf),
            fc1_weight_sf=to_cute(fc1_sf),
            fc1_activation_dequant_scale=to_cute(
                ones_1 if self.blockwise else fc1_act, 4
            ),
            fc1_weight_dequant_scale=to_cute(ones_e if self.blockwise else fc1_wdq, 4),
            fc1_output=to_cute(fc1_out),
            fc1_output_sf=to_cute(fc1_out_sf),
            fc2_weight=to_cute(fc2_w),
            fc2_weight_sf=to_cute(fc2_sf),
            fc2_activation_dequant_scale=to_cute(
                ones_1 if self.blockwise else fc2_act, 4
            ),
            fc2_weight_dequant_scale=to_cute(ones_e if self.blockwise else fc2_wdq, 4),
            fc2_output=to_cute(self.fc2_out),
            topk_scores=to_cute(self.topk_scores, 4),
            fc1_done_counter=to_cute(self.done_counter, 4),
            offs=to_cute(self.offs, 4),
            stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        self._keep = (ones_e, ones_1)
        self.compiled = cute.compile(kernel, max_active_clusters=clusters, **self.kw)

    def handoff(self, recv_x, recv_sf, recv_idx, recv_w, valid):
        import triton

        count, pack, _ = _handoff_triton_kernels()
        e, k, h = self.experts, recv_idx.shape[1], recv_x.shape[1]
        idx = recv_idx.reshape(-1)
        n = idx.numel()
        has_valid = valid is not None
        valid = valid if has_valid else idx  # unread when HAS_VALID is False
        counts, cursor = self.counters[:e], self.counters[e:]
        self.counters.zero_()
        count[(triton.cdiv(n, 1024),)](
            idx, valid, counts, n, K=k, BLOCK=1024, HAS_VALID=has_valid
        )
        # Per-tensor rows carry no scales (SF_COLS=0): any int pointer will do.
        sf = recv_sf if self.blockwise else self.slots
        pool_sf = self.pool_sf if self.blockwise else self.slots
        pack[(n,)](
            idx,
            valid,
            counts,
            cursor,
            self.offs,
            self.slots,
            recv_x.view(torch.int32),
            sf,
            self.pool_x.view(torch.int32),
            pool_sf,
            sf.stride(0),
            0,
            e,
            K=k,
            TILE=self.tile,
            E_BLOCK=triton.next_power_of_2(e),
            WORDS=h // 4,
            WBLOCK=triton.next_power_of_2(h // 4),
            SF_COLS=h // 128 if self.blockwise else 0,
            SF_BLOCK=triton.next_power_of_2(h // 128),
            HAS_VALID=has_valid,
            DG_LAYOUT=False,
        )
        self._w, self._rows = recv_w, recv_x.shape[0]

    def compute(self):
        self.done_counter.zero_()
        self.compiled(**self.kw)

    def reduce(self):
        import triton

        _, _, reduce = _handoff_triton_kernels()
        h, k = self.args.hidden, self.args.top_k
        rows, block = self._rows, 1024
        reduce[(rows, triton.cdiv(h, block))](
            self.slots, self._w, self.fc2_out, self.out, K=k, H=h, BLOCK=block
        )
        return self.out[:rows]


class _CutlassBlockExperts:
    """This rank's experts on flashinfer's SM90 CUTLASS / DeepGEMM FP8
    block-scale grouped GEMM (TRT-LLM ``fp8_blockscale_gemm``, the runner of
    the push-style MegaMoE): E4M3 rows + 1x128 scales in, 128x128 weight
    scales.  FC1 runs fused with SwiGLU and the 1x128 requant (FP8 + scales
    out), FC2 writes BF16.  The handoff packs the received picks into the
    GEMM's layout (expert-contiguous rows, column-major padded scales); the
    reduce is the FC12 path's."""

    def __init__(self, rank, experts, args):
        from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.gemm import (
            create_sm90_push_fp8_moe_gemm_runner,
        )
        from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.weights import (
            transform_weights_for_sm90_push,
        )

        w13, w2 = _bf16_weights(rank, experts, args.hidden, args.intermediate)
        self.w13, self.w13_sf, self.w2, self.w2_sf = transform_weights_for_sm90_push(
            w13, w2, interleave_gate_up=True
        )
        del w13, w2
        torch.cuda.empty_cache()
        self.runner = create_sm90_push_fp8_moe_gemm_runner()
        self.experts, self.args = experts, args

    def build(self, tokens, rows):
        e, h, inter, k = (
            self.experts,
            self.args.hidden,
            self.args.intermediate,
            self.args.top_k,
        )
        max_rows = rows * k  # worst case: every received pick local
        dev = "cuda"
        self.runner.get_moe_workspace_size(
            tokens * k, max_rows, max(2 * inter, h), max(h, inter), e, True, True
        )
        self._workspace = torch.empty(1, dtype=torch.uint8, device=dev)
        self.runner.configure_workspace(self._workspace)
        m_buf = (max_rows + 127) // 128 * 128
        # Scale column stride: the runner's deep_gemm::compute_padded_offset(max_rows, E).
        self.sf_p = (max_rows + e * 31) // 32 * 32
        self.a1 = torch.empty(m_buf, h, dtype=torch.float8_e4m3fn, device=dev)
        self.sfa1 = torch.empty(
            (h // 128) * self.sf_p + 128, dtype=torch.float32, device=dev
        )
        self.a2 = torch.empty(m_buf, inter, dtype=torch.uint8, device=dev)
        self.sfa2 = torch.empty(
            (inter // 128) * self.sf_p + 128, dtype=torch.float32, device=dev
        )
        self.y = torch.empty(m_buf, h, dtype=torch.bfloat16, device=dev)
        self.offsets = torch.zeros(e + 1, dtype=torch.int64, device=dev)
        self.counters = torch.zeros(2 * e, dtype=torch.int32, device=dev)
        self.slots = torch.empty(rows * k, dtype=torch.int32, device=dev)
        self.out = torch.empty(rows, h, dtype=torch.bfloat16, device=dev)
        # DeepGEMM JIT-compiles per (expected_m, N, K, groups) here, not timed.
        self.offsets.zero_()
        self.compute()
        torch.cuda.synchronize()

    def handoff(self, recv_x, recv_sf, recv_idx, recv_w, valid):
        import triton

        count, pack, _ = _handoff_triton_kernels()
        e, k, h = self.experts, recv_idx.shape[1], recv_x.shape[1]
        idx = recv_idx.reshape(-1)
        n = idx.numel()
        has_valid = valid is not None
        valid = valid if has_valid else idx  # unread when HAS_VALID is False
        counts, cursor = self.counters[:e], self.counters[e:]
        self.counters.zero_()
        count[(triton.cdiv(n, 1024),)](
            idx, valid, counts, n, K=k, BLOCK=1024, HAS_VALID=has_valid
        )
        pack[(n,)](
            idx,
            valid,
            counts,
            cursor,
            self.offsets,
            self.slots,
            recv_x.view(torch.int32),
            recv_sf,
            self.a1.view(torch.int32),
            self.sfa1,
            recv_sf.stride(0),
            self.sf_p,
            e,
            K=k,
            TILE=1,
            E_BLOCK=triton.next_power_of_2(e),
            WORDS=h // 4,
            WBLOCK=triton.next_power_of_2(h // 4),
            SF_COLS=h // 128,
            SF_BLOCK=triton.next_power_of_2(h // 128),
            HAS_VALID=has_valid,
            DG_LAYOUT=True,
        )
        self._w, self._rows = recv_w, recv_x.shape[0]

    def compute(self):
        h, inter = self.args.hidden, self.args.intermediate
        self.runner.moe_gemm_fc1_fused(
            self.a2,
            self.sfa2,
            self.a1,
            self.w13,
            self.offsets,
            2 * inter,
            h,
            self.sfa1,
            self.w13_sf,
            True,
        )
        self.runner.moe_gemm(
            self.y,
            self.a2.view(torch.float8_e4m3fn),
            self.w2,
            self.offsets,
            h,
            inter,
            self.sfa2,
            self.w2_sf,
            True,
        )

    def reduce(self):
        import triton

        _, _, reduce = _handoff_triton_kernels()
        h, k = self.args.hidden, self.args.top_k
        rows, block = self._rows, 1024
        reduce[(rows, triton.cdiv(h, block))](
            self.slots, self._w, self.y, self.out, K=k, H=h, BLOCK=block
        )
        return self.out[:rows]


def _local_routing(recv_idx, recv_w, valid):
    """Received LOCAL ids/weights -> runner routing; foreign picks and rows past
    the received count become id -1 (skipped by the runner) with weight 0."""
    keep = recv_idx >= 0
    if valid is not None:
        keep &= valid[:, None]
    ids = torch.where(keep, recv_idx, -1).to(torch.int32)
    return ids, torch.where(keep, recv_w.float(), 0.0)


def _identity_check(engine, x, sf, rank, tokens, local_experts):
    """dispatch -> y = dequant(x) * sum(local weights) -> combine == dequant(x)."""
    recv_x, recv_sf, recv_idx, recv_w, valid = engine.dispatch()
    ids, ws = _local_routing(recv_idx, recv_w, valid)
    if not bool(((ids >= -1) & (ids < local_experts)).all()):
        raise SystemExit(f"[check] rank {rank} T={tokens}: received ids not LOCAL")
    y = (_dequant(recv_x, recv_sf).float() * ws.sum(-1, keepdim=True)).bfloat16()
    out = engine.combine(y).float()
    torch.cuda.synchronize()
    ref = _dequant(x, sf).float()
    err = ((out - ref).norm() / ref.norm()).item()
    if not err < 1e-2:
        raise SystemExit(f"[check] rank {rank} T={tokens}: rel_l2 {err:.3e}")
    return err


_BASELINES = ("nccl_ep_ll", "nccl_ep_ht", "deepep")
_STAGES = ("dispatch", "handoff", "compute", "reduce", "combine")


def _run_baselines(args, rank, world, per_node):
    """Time the split baselines; returns the result rows."""
    dist.init_process_group("gloo")
    from flashinfer.autotuner import autotune, set_autotune_process_group
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        Fp8BlockScaleK,
        quantize_fp8_per_token_block,
    )

    # One rank at a time: the JIT file lock races across nodes sharing an NFS
    # cache.  The module is cached per process for the runners built below.
    cutlass_gemm = args.baseline_gemm == "cutlass"
    for r in range(world):
        if r == rank and cutlass_gemm and args.scale_mode == "per_tensor":
            get_cutlass_fused_moe_module("90")
        dist.barrier()
    # Average the tactic timings across ranks: tuned independently, timing
    # noise (or a failing tactic) can leave one rank on a slow tactic that
    # every other rank then waits for in combine.  All ranks tune the same
    # shapes (T * world rows), as the reduction requires.
    set_autotune_process_group(dist.group.WORLD)
    if not cutlass_gemm:
        experts_cls = _Fc12Experts
    elif args.scale_mode == "blockwise":
        experts_cls = _CutlassBlockExperts
    else:
        experts_cls = _CutlassPerTensorExperts
    experts = experts_cls(rank, args.num_experts // world, args)
    mode = args.mode if not cutlass_gemm else f"{args.mode}+cutlass"
    rows = []
    for tokens in [int(t) for t in args.tokens.split(",")]:
        hs, w, idx = _inputs(rank, tokens, args.hidden, args.num_experts, args.top_k)
        # The E4M3 rows (and block scales) the mega kernel stages, up front.
        if args.scale_mode == "blockwise":
            x, sf = quantize_fp8_per_token_block(
                hs.float(), torch.float8_e4m3fn, block_k=Fp8BlockScaleK
            )
            sf = sf.contiguous()
        else:
            x, sf = (hs.float() / _ACT_SCALE).to(torch.float8_e4m3fn), None
        if args.mode == "deepep":
            engine = _DeepEp(tokens, x, sf, idx, w, args, world)
        else:
            engine = _NcclEp(args.mode[-2:], tokens, x, sf, idx, w, args, world)
        if experts_cls is _CutlassBlockExperts:
            # One rank at a time: the DeepGEMM JIT compiles into a cache the
            # ranks of a node share.
            for r in range(world):
                if r == rank:
                    experts.build(tokens, tokens * world)
                dist.barrier()
        else:
            experts.build(tokens, tokens * world)
        check_err = (
            _identity_check(engine, x, sf, rank, tokens, experts.experts)
            if args.check
            else None
        )

        def call(ev=None):
            recv_x, recv_sf, recv_idx, recv_w, valid = engine.dispatch()
            if ev is not None:
                ev[1].record()
            experts.handoff(recv_x, recv_sf, recv_idx, recv_w, valid)
            if ev is not None:
                ev[2].record()
            experts.compute()
            if ev is not None:
                ev[3].record()
            y = experts.reduce()
            if ev is not None:
                ev[4].record()
            return engine.combine(y)

        # Tune the CUTLASS tactics for this row count first: untuned, the
        # fallback tactic is several times slower.
        with autotune(experts_cls is _CutlassPerTensorExperts):
            call()
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        dist.barrier()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(_STAGES) + 1)]
        samples = {s: [] for s in ("total", "excl_handoff", *_STAGES)}
        for _ in range(args.iters):
            dist.barrier()
            torch.cuda.synchronize()
            engine.align()
            ev[0].record()
            call(ev)
            ev[-1].record()
            torch.cuda.synchronize()
            for i, s in enumerate(_STAGES):
                samples[s].append(ev[i].elapsed_time(ev[i + 1]) * 1e3)
            samples["total"].append(ev[0].elapsed_time(ev[-1]) * 1e3)
            samples["excl_handoff"].append(
                samples["total"][-1] - samples["handoff"][-1] - samples["reduce"][-1]
            )
        meds = torch.tensor(
            [statistics.median(samples[s]) for s in samples], dtype=torch.float64
        )
        gathered = [torch.zeros_like(meds) for _ in range(world)]
        dist.all_gather(gathered, meds)
        per_rank = torch.stack(gathered)
        row = dict(
            mode=mode,
            world=world,
            per_node=per_node,
            tokens=tokens,
            scale_mode=args.scale_mode,
            max_rank_us=float(per_rank[:, 0].max()),
            min_rank_us=float(per_rank[:, 0].min()),
            # The same slowest-rank median without the handoff / reduce glue
            # around the GEMM (routing masks, expert packing or the blockwise
            # dequant the CUTLASS block runner forces, pre-reduce).
            excl_handoff_us=float(per_rank[:, 1].max()),
        )
        for i, s in enumerate(_STAGES):
            row[f"{s}_us"] = float(per_rank[:, i + 2].max())
        rows.append(row)
        if rank == 0:
            chk = "" if check_err is None else f" [check rel_l2 {check_err:.1e}]"
            stages = " ".join(f"{s} {row[s + '_us']:.1f}" for s in _STAGES)
            print(
                f"BENCH {mode} W={world} R={per_node} T={tokens} {args.scale_mode}: "
                f"max-rank median {row['max_rank_us']:.1f} us (min-rank "
                f"{row['min_rank_us']:.1f} us, excl. handoff "
                f"{row['excl_handoff_us']:.1f} us) | {stages}{chk}",
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
    split baseline and each fused mode's speedup over it (> 1 = fused faster),
    both on the full baseline time and with its handoff stage excluded."""
    import csv

    runs = {}
    for path in inputs:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (int(r["world"]), int(r["per_node"]), r["scale_mode"])
                runs.setdefault(key + (int(r["tokens"]),), {})[r["mode"]] = r
    seen = {m for v in runs.values() for m in v}
    modes = [m for m in _FUSED if m in seen] + sorted(seen - set(_FUSED))
    baselines = [m for m in modes if m not in _FUSED]
    fused = [m for m in modes if m in _FUSED]
    rows = []
    for (world, per_node, scale_mode, tokens), by_mode in sorted(runs.items()):
        row = dict(world=world, per_node=per_node, scale_mode=scale_mode, tokens=tokens)
        for m in modes:
            r = by_mode.get(m)
            row[f"{m}_us"] = f"{float(r['max_rank_us']):.1f}" if r else ""
        for m in modes:
            if m in baselines:
                r = by_mode.get(m) or {}
                for col in ("excl_handoff", *_STAGES):
                    v = r.get(f"{col}_us", "")
                    row[f"{m}_{col}_us"] = f"{float(v):.1f}" if v else ""
        for metric, suffix in (
            ("max_rank_us", ""),
            ("excl_handoff_us", "_excl_handoff"),
        ):
            timed = {
                m: float(by_mode[m][metric])
                for m in baselines
                if by_mode.get(m, {}).get(metric)
            }
            best = min(timed, key=timed.get) if timed else ""
            row[f"best_baseline{suffix}"] = best
            row[f"best_baseline{suffix}_us"] = f"{timed[best]:.1f}" if best else ""
            for m in fused:
                ok = best and m in by_mode
                row[f"{m}_speedup_vs_best{suffix}"] = (
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
