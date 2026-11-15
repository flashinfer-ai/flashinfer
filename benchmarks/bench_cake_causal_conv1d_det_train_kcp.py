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

Benchmark of the deterministic context-parallel (KCP) Cake packed width-4 depthwise causal convolution + SiLU
training step (``flashinfer.cake_causal_conv1d_det_train_kcp``: ``cake_causal_conv1d_det_kcp`` /
``cake_causal_conv1d_det_kcp_multi``) in both launch forms of its backward, against every non-Cake context-parallel
baseline at the same per-rank shapes, alone and beside kernels resident on other streams.

Arms (the Cake arms run the autograd entries; the multi entry when a row has several tensors, the baselines one call
per tensor):

* ``det_coop``: ``cooperative=True`` (one cooperative backward launch, fixed-order ``dweight`` reduction);
* ``det_split``: ``cooperative=False`` (an ordinary backward launch one SM short of the co-resident CTA count, then
  the fixed-order reduce kernel);
* ``det_split_aa``: ``det_split`` again on its own leaves, the A/A pair: the per-round ``det_split_aa / det_split``
  range is the row's noise band;
* ``fla_cp_triton``: flash-linear-attention ``causal_conv1d(..., cp_context=build_cp_context(...),
  backend="triton")`` on FLA's own partition of the pack (``contiguous`` / ``zigzag``);
* ``fla_cp_gluon``: the same call with ``FLA_CONV_GLUON=1``.  FLA's CP path names no Gluon backend (``triton`` /
  ``cuda`` / ``mix``); its Triton path hands ``causal_conv1d_fwd`` / ``causal_conv1d_bwd`` to FLA's backend dispatch,
  which runs the Gluon kernels only when their verifiers accept the call.  The arm runs when they accept it on every
  rank, else it is reported with the verifiers' reason (the call then runs the Triton kernels, i.e.
  ``fla_cp_triton``);
* ``fla_cp_cuda`` / ``fla_cp_cuda_det``: ``backend="cuda"`` (Dao-AILab causal-conv1d's channel-last kernel,
  ``contiguous`` only) with ``CAUSAL_CONV1D_DETERMINISTIC=0`` (atomic FP32 ``dweight``, the library default with
  torch deterministic algorithms off) / ``=1`` (per-chunk FP32 ``dweight`` workspace summed by torch).  The variable
  is read by every backward call (``std::getenv`` in the C++ backward, a leading ``1`` / ``0`` decides, else torch's
  deterministic-algorithms flag), so it is set around every call of the arm;
* ``megatron_cp`` / ``megatron_cp_det``: Megatron-Core ``megatron.core.ssm.causal_conv1d.causal_conv1d_cp`` (the
  previous rank's three-row tail through one autograd ``all_to_all`` per direction, Dao's kernel on the shard, the
  documents as ``global_seq_idx``): contiguous equal shards only; ``CAUSAL_CONV1D_DETERMINISTIC`` as above;
* ``megatron_a2a_cp`` / ``megatron_a2a_cp_det``: the all-to-all context parallel of Megatron-Core's Mamba mixer
  (``all_to_all_sp2hp``: every rank gets the whole pack for ``C / W`` channels; Dao's kernel over the whole pack with
  ``seq_idx``; ``all_to_all_hp2sp`` back) on this benchmark's partition: any layout with the same token count on
  every rank, the gathered rows put in pack order by an index permutation (the mixer's
  ``_undo_attention_load_balancing``) and back;
* ``attn_gym_cp``: Attention Gym's fragment-tail exchange and CuTe DSL ``causal_conv1d`` (BF16 taps: report only);
* ``torch_compile_halo``: the document-local formula in FP32 (FP32 taps and accumulation, SiLU' at the FP32
  preactivation, one BF16 rounding of ``y``) compiled by ``torch.compile`` over this rank's rows, each fragment
  preceded by its three history rows, which one autograd ``all_to_all_single`` per direction exchanges (the rows the
  Cake halo exchange moves).

Precision class.  The Cake contract: BF16 activations in and out, FP32 taps, FP32 accumulation, one BF16 rounding,
SiLU' at the FP32 preactivation.  Each arm records its activation, tap and gradient dtypes, the preactivation its
SiLU' takes, its accumulation and ``same_precision`` (FP32 taps and FP32 accumulation; SiLU' at the BF16-rounded
preactivation still qualifies, so ``fla_cp_triton`` is a candidate).  The best baseline is the fastest same-precision
arm whose error check passes.

Determinism and error (one untimed block per arm, before the timing): ``DETERMINISM_RUNS`` forward + backward passes
on the same inputs; class ``det`` when ``y``, ``dx`` and the rank's ``dweight`` are bitwise equal over the runs on
every rank, else ``nondet`` with the tensors that varied.  The first run is compared with an FP64 evaluation of the
same math (the BF16 inputs and FP32 taps, document-local) on the arm's own rows: max-abs and relative L2 of ``y`` and
``dx`` (padding rows excluded), of ``dweight`` summed over the ranks, and of the rank's ``dweight`` partial against
the FP64 sum over the arm's rows (``budget_ratio``: the largest ``|dw - ref| / (5e-2 + 1e-4 |ref|)``).  The check
passes when the three relative L2 errors are below ``NUMERICS_REL_L2``.

Phases (autograd, the leaves' gradients reset before every backward): ``fwd``; ``bwd`` (the backward of a recorded
forward, ``retain_graph=True``); ``fwdbwd`` (one forward and its backward); ``ckpt``
(``torch.utils.checkpoint(..., use_reentrant=False)`` and the backward); ``ckpt_handle`` (Cake arms only: the same
with a ``CakeCausalConv1dDetKcpRecomputeHandle``, so the recompute runs without the forward exchange).

Measures (on every rank; a row reports the slowest rank, the JSON keeps every rank's ``gpu`` and ``ahead`` values):

* ``gpu_us``: the CUDA-event span of the phase (cold L2; CUDA events, not CUPTI: CUPTI's per-measurement finalize
  races the NCCL watchdog thread).  The span includes the exchange's wait for the peers and any host gap;
* ``ahead_us``: the host enqueue hidden.  Per sample: device synchronize, barrier, L2 flush, a one-thread spin on the
  phase's stream long enough for the host to enqueue the rest of the sample behind it, a one-element ``all_reduce``
  aligning the ranks, a ``RESIDENT_US`` one-thread spin, then CUDA events around the phase (``ahead_late`` counts
  samples whose enqueue outlasted the spin; 0 expected);
* side loads (``--side-load``, ``NxDms``: N kernels of D ms): the ``ahead`` sample with one addition.  After the
  aligning ``all_reduce`` the phase's stream records an event; N other streams (created once) each wait on it and
  launch a one-block one-thread spin kernel of D ms (``torch.cuda._sleep``).  The side kernels and the
  ``RESIDENT_US`` spin become ready together, so the side kernels are resident when the phase's first kernel is
  enqueued behind that spin, with about D ms - ``RESIDENT_US`` left.  Per load: the loaded span, the ratio loaded /
  unloaded per round (both the slowest rank of the same round) and ``loaded_scaled_us`` = median ratio x unloaded
  median.  FlashInfer #8365's side-stream criterion: the backward's ratio at most ``SIDE_STREAM_LIMIT`` with a 1x2 ms
  and with a 1x4 ms load (``side_criterion``);
* ``wall_us`` / ``host_us``: the synchronized wall time of one call and its enqueue time (``--skip-wall`` drops them);
* ``first`` / ``steady`` (``det_coop`` / ``det_split``): the first call of each phase after the package caches are
  cleared and the next call: wall time, the allocator's peak above the bytes held before (``peak_temp_bytes``) and the
  net bytes held after (``retained_bytes``).

Rounds interleave the arms: round ``i`` runs the arm list rotated left by ``i mod n`` and reversed when ``i // n`` is
odd, so each arm leads once and runs last once in every ``n`` rounds (``--rounds`` defaults to ``n``).  Every value
is the median over the rounds with its ``[min, max]``.  Per row: ``best_baseline`` (per phase and measure the fastest
eligible baseline; ``best_det_baseline``: the fastest whose measured class is ``det``), ``vs_best`` (per Cake arm
``best / arm``, above 1: Cake faster, with the per-round range),
``split_vs_coop`` (``det_split / det_coop``, ``within_2pct``, and the A/A band ``det_split_aa / det_split``) and
``side_criterion``.  ``COLUMN`` (``ahead``) is the comparison column of the text summary.  Exit status 1 when a Cake
check fails: class ``det`` for every Cake arm, ``y`` and ``dx`` bitwise equal between the two launch forms, the
``ckpt`` and ``ckpt_handle`` gradients bitwise the direct backward's (each form), the error check and every rank's
``dweight`` partial within ``5e-2 + 1e-4 |ref|``.

Rows (``--rows``):

* ``customer``: ``C = 10240`` at 32768 tokens per rank (single entry) and ``3 x C = 8192`` at 16384 tokens per rank
  (multi entry), each in the ``zigzag``, ``zigzag_per_document`` and ``contiguous`` layouts (``contiguous`` only at
  ``W = 1``).  Documents: at ``W = 1`` the FlashInfer #8365 packs (12000 / 9000 / 6000 / 3768 / 2000 tokens; 9000 /
  5000 / 2384); at ``W > 1`` the pack's first document, one long document of ``floor(W / 2)`` ranks' tokens plus a
  third of a rank's, then the pack repeated from its second document up to ``W x tokens per rank`` (the last one cut);
* ``pr8365`` (``W = 1``): the FlashInfer #8365 single-GPU rows: ``C = 10240`` over 32768 tokens and ``C = 8192`` over
  16384 tokens, single, and ``3 x C = 8192`` multi;
* ``custom``: ``--channels``, ``--tokens-per-rank``, ``--docs``, ``--tensors`` and ``--layouts``.

Every rank draws the whole pack from the same seed (the baselines' own partitions and the FP64 evaluation read it)
and releases it before the timing.  The KCP partition is
``flashinfer.cake_kda_det_train_kcp.cake_kda_det_train_kcp_partition``.  A baseline runs a row only when every rank can
run it; a failing baseline is reported with its reason and the row goes on (a failure inside one rank's collective
can still stall the group until the NCCL timeout).

Launch: ``torchrun --nproc_per_node W``; one Slurm task per GPU (``SLURM_PROCID`` / ``SLURM_NTASKS`` /
``SLURM_LOCALID``, rendezvous at ``MASTER_ADDR`` / ``MASTER_PORT``); or plain ``python`` for ``W = 1``.

Example::

    python benchmarks/bench_cake_causal_conv1d_det_train_kcp.py --rows pr8365 --json kcp_w1.json
    torchrun --nproc_per_node 4 benchmarks/bench_cake_causal_conv1d_det_train_kcp.py --json kcp_w4.json
    MASTER_ADDR=node0 MASTER_PORT=29500 srun -N 2 --ntasks-per-node 4 \\
        python benchmarks/bench_cake_causal_conv1d_det_train_kcp.py --json kcp_w8.json
"""

from __future__ import annotations

import argparse
import bisect
import gc
import importlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flashinfer  # noqa: E402
import flashinfer.cake_causal_conv1d_det_train as det_conv  # noqa: E402
import flashinfer.cake_causal_conv1d_det_train_kcp as det_kcp_api  # noqa: E402
import cake_causal_conv1d_baselines as baselines  # noqa: E402
import bench_cake_causal_conv1d_nondet_train_kcp as kcp_bench  # noqa: E402
from flashinfer.cake_kda_det_train_kcp import (  # noqa: E402
    cake_kda_det_train_kcp_build_context,
    cake_kda_det_train_kcp_partition,
)
from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (  # noqa: E402
    kcp as det_kcp_backend,
)

WIDTH = 4
HALO = WIDTH - 1
SEED = 0
LAYOUTS = ("zigzag", "zigzag_per_document", "contiguous")
# The Cake arms: name -> the ``cooperative`` argument of their calls.
CAKE_ARMS = {"det_coop": True, "det_split": False, "det_split_aa": False}
BASELINE_ARMS = (
    "fla_cp_triton",
    "fla_cp_gluon",
    "fla_cp_cuda",
    "fla_cp_cuda_det",
    "megatron_cp",
    "megatron_cp_det",
    "megatron_a2a_cp",
    "megatron_a2a_cp_det",
    "attn_gym_cp",
    "torch_compile_halo",
)
PHASES = ("fwd", "bwd", "fwdbwd", "ckpt", "ckpt_handle")
BASELINE_PHASES = ("fwd", "bwd", "fwdbwd", "ckpt")
# The call of each phase in an arm's call table (``fwdbwd`` is ``autograd_calls``' ``step``).
CALL_OF = {
    "fwd": "fwd",
    "bwd": "bwd",
    "fwdbwd": "step",
    "ckpt": "ckpt",
    "ckpt_handle": "ckpt_handle",
}
COLUMN = "ahead"
DETERMINISM_RUNS = 5
# Rows per chunk of the FP64 evaluation (about six [rows, C] FP64 temporaries live at once).
REF_CHUNK = 2048
# The one-thread spin between the side kernels' release and the phase's first kernel.
RESIDENT_US = 50.0
SIDE_LOADS = "none,1x2ms,1x4ms,8x2ms"
SIDE_STREAM_LIMIT = 1.3
SPLIT_TOLERANCE = 1.02
# The deterministic KCP package's dweight bound against an FP64 oracle.
DW_ATOL, DW_RTOL = 5e-2, 1e-4
# (channels, tokens per rank, the FlashInfer #8365 single-GPU documents of that token count)
CUSTOMER_SINGLE = (10240, 32768, (12000, 9000, 6000, 3768, 2000))
CUSTOMER_MULTI = (8192, 16384, (9000, 5000, 2384))
MULTI_TENSORS = 3
# The library versions the arms were written against (the JSON records the installed ones).
LIBRARY_TARGETS = {
    "flash-linear-attention": "0.6.0",
    "causal-conv1d": "1.7.0",
    "megatron-core": "0.20.0",
}
DAO_DET_VAR = "CAUSAL_CONV1D_DETERMINISTIC"
FLA_TRITON_ENV: dict[str, Optional[str]] = {"FLA_CONV_GLUON": None, "FLA_GLUON": None}
FLA_GLUON_ENV: dict[str, Optional[str]] = {"FLA_CONV_GLUON": "1", "FLA_GLUON": None}
DAO_ENV = {False: {DAO_DET_VAR: "0"}, True: {DAO_DET_VAR: "1"}}

_median = kcp_bench._median
_gather = kcp_bench._gather
_agreed = kcp_bench._agreed
_bits = kcp_bench._bits


# ---------------------------------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    preset: str
    channels: int
    tensors: int
    tokens_per_rank: int
    pack: tuple[int, ...]
    layout: str

    def docs(self, world: int) -> tuple[int, ...]:
        """The document lengths of the W-rank pack (module docstring, ``customer``)."""

        total = world * self.tokens_per_rank
        if world == 1:
            docs: list[int] = []
            index = 0
        else:
            long_doc = self.tokens_per_rank * (world // 2) + self.tokens_per_rank // 3
            docs = [self.pack[0], long_doc]
            if sum(docs) > total:
                raise ValueError(
                    f"the pack's first document {self.pack[0]} leaves no room for the long document at W={world}"
                )
            index = 1
        remaining = total - sum(docs)
        while remaining > 0:
            n = min(self.pack[index % len(self.pack)], remaining)
            docs.append(n)
            remaining -= n
            index += 1
        return tuple(docs)


def preset_rows(args: argparse.Namespace, world: int) -> list[Row]:
    layouts = ("contiguous",) if world == 1 else tuple(args.layouts)
    if args.rows == "customer":
        rows = []
        for (channels, tokens, pack), tensors in (
            (CUSTOMER_SINGLE, 1),
            (CUSTOMER_MULTI, MULTI_TENSORS),
        ):
            rows += [
                Row("customer", channels, tensors, tokens, pack, layout)
                for layout in layouts
            ]
        return rows
    if args.rows == "pr8365":
        if world != 1:
            raise SystemExit("--rows pr8365 is the single-GPU preset (W = 1)")
        (c1, t1, p1), (c2, t2, p2) = CUSTOMER_SINGLE, CUSTOMER_MULTI
        return [
            Row("pr8365", c1, 1, t1, p1, "contiguous"),
            Row("pr8365", c2, 1, t2, p2, "contiguous"),
            Row("pr8365", c2, MULTI_TENSORS, t2, p2, "contiguous"),
        ]
    pack = tuple(args.docs) if args.docs else (args.tokens_per_rank,)
    return [
        Row("custom", args.channels, args.tensors, args.tokens_per_rank, pack, layout)
        for layout in layouts
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Process group, timing
# ---------------------------------------------------------------------------------------------------------------------


def _init() -> tuple[int, int, torch.device]:
    """The process group of a ``torchrun`` launch, of a Slurm step with one task per GPU, or a one-rank group."""

    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ and ntasks > 1:
        os.environ.update(
            RANK=os.environ["SLURM_PROCID"],
            WORLD_SIZE=str(ntasks),
            LOCAL_RANK=os.environ.get("SLURM_LOCALID", "0"),
        )
        if "MASTER_ADDR" not in os.environ:
            if int(os.environ.get("SLURM_NNODES", "1")) > 1:
                raise SystemExit(
                    "a multi-node Slurm launch needs MASTER_ADDR (the first node) and MASTER_PORT"
                )
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("MASTER_PORT", "29500")
    if "RANK" not in os.environ:
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(kcp_bench._free_port()))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


class SideLoadAhead(kcp_bench._GpuAhead):
    """The ``ahead`` sample of the KCP benchmark with a ``RESIDENT_US`` spin before the phase, and optionally side
    kernels on other streams released together with that spin (module docstring)."""

    def __init__(self, device: torch.device, side_streams: int) -> None:
        super().__init__(device)
        self.side_streams = [
            torch.cuda.Stream(device=device) for _ in range(side_streams)
        ]
        self.gate = torch.cuda.Event()
        self.resident_cycles = int(RESIDENT_US * 1e-3 * self.cycles_per_ms)
        self.load: tuple[int, float] = (0, 0.0)

    def _sample(self, fn: Callable[[], object], spin_ms: float) -> tuple[float, float]:
        kernels, kernel_ms = self.load
        torch.cuda.synchronize(self.device)
        dist.barrier()
        self.flush.zero_()
        torch.cuda._sleep(int(spin_ms * self.cycles_per_ms))
        if dist.get_world_size() > 1:
            dist.all_reduce(self.align)
        if kernels:
            self.gate.record()
            for stream in self.side_streams[:kernels]:
                stream.wait_event(self.gate)
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(int(kernel_ms * self.cycles_per_ms))
        torch.cuda._sleep(self.resident_cycles)
        self.start.record()
        t0 = time.perf_counter()
        fn()
        enqueue_ms = (time.perf_counter() - t0) * 1e3
        self.end.record()
        torch.cuda.synchronize(self.device)
        return self.start.elapsed_time(self.end), enqueue_ms

    def loaded(
        self, fn: Callable[[], object], spin_ms: float, kernels: int, kernel_ms: float
    ) -> tuple[float, int]:
        self.load = (kernels, kernel_ms)
        try:
            return self(fn, spin_ms)
        finally:
            self.load = (0, 0.0)


def parse_side_loads(text: str) -> list[tuple[str, int, float]]:
    """``"none,1x2ms,8x2ms"`` -> ``[(label, kernels, ms), ...]`` without ``none`` (the unloaded sample is always
    taken)."""

    loads = []
    for item in (part.strip() for part in text.split(",")):
        if not item or item == "none":
            continue
        match = re.fullmatch(r"(\d+)x(\d+(?:\.\d+)?)ms", item)
        if match is None or int(match.group(1)) < 1:
            raise SystemExit(
                f"--side-load entries are none or NxDms (e.g. 1x2ms), got {item!r}"
            )
        loads.append((item, int(match.group(1)), float(match.group(2))))
    return loads


def _in_env(env: dict, fn: Callable[[], object]) -> Callable[[], object]:
    if not env:
        return fn

    def run() -> object:
        with baselines.environment(env):
            return fn()

    return run


# ---------------------------------------------------------------------------------------------------------------------
# The problem of a row
# ---------------------------------------------------------------------------------------------------------------------


class Problem:
    """One row: the whole pack drawn on every rank from the same seed, this rank's Cake rows sliced out."""

    def __init__(self, row: Row, rank: int, world: int, device: torch.device) -> None:
        self.row = row
        self.device = device
        self.lens = row.docs(world)
        cu = [0]
        for n in self.lens:
            cu.append(cu[-1] + n)
        self.cu = cu
        self.total = cu[-1]
        self.part = cake_kda_det_train_kcp_partition(cu, world, row.layout)
        self.ctx = cake_kda_det_train_kcp_build_context(
            self.part, dist.group.WORLD, rank, device, conv_kernel_size=WIDTH
        )
        # Every rank's global token ranges in its local row order.
        self.fragments = [
            baselines.merged_fragments(cu, self.part.fragments[r]) for r in range(world)
        ]
        ranges = self.fragments[rank]
        self.rows = (
            torch.cat([torch.arange(s, e) for s, e in ranges])
            if ranges
            else torch.zeros(0, dtype=torch.long)
        )
        self.cu_t = torch.tensor(cu, dtype=torch.long, device=device)
        self.doc_ids = (
            torch.repeat_interleave(
                torch.arange(len(self.lens), dtype=torch.int32, device=device),
                torch.tensor(self.lens, device=device),
            )
            .unsqueeze(0)
            .contiguous()
        )
        rows_d = self.rows.to(device)
        generator = torch.Generator(device=device).manual_seed(SEED)
        self.xs_g: list[torch.Tensor] = []
        self.dys_g: list[torch.Tensor] = []
        self.ws: list[torch.Tensor] = []
        self.xs: list[torch.Tensor] = []
        self.dys: list[torch.Tensor] = []
        channels = row.channels
        for _ in range(row.tensors):
            x = torch.randn(
                self.total, channels, device=device, generator=generator, dtype=torch.bfloat16
            )  # fmt: skip
            # N(0, 0.25) taps, as the single-rank benchmarks'.
            w = torch.randn(channels, WIDTH, device=device, generator=generator) * 0.5
            dy = torch.randn(
                self.total, channels, device=device, generator=generator, dtype=torch.bfloat16
            )  # fmt: skip
            self.xs_g.append(x)
            self.dys_g.append(dy)
            self.ws.append(w.contiguous())
            self.xs.append(x.index_select(0, rows_d))
            self.dys.append(dy.index_select(0, rows_d))
        halo = self.ctx.halo
        n = row.tensors
        self.halo = {
            "fwd_send_bytes": int(sum(halo.fwd_send_rows)) * channels * 2 * n,
            "fwd_recv_bytes": int(sum(halo.fwd_recv_rows)) * channels * 2 * n,
            "bwd_send_bytes": int(sum(halo.bwd_send_rows)) * channels * 4 * n,
            "bwd_recv_bytes": int(sum(halo.bwd_recv_rows)) * channels * 4 * n,
        }

    def release_pack(self) -> None:
        self.xs_g.clear()
        self.dys_g.clear()
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------------
# Precision class, determinism and error
# ---------------------------------------------------------------------------------------------------------------------


def precision(
    *,
    taps: str,
    preactivation: str,
    accumulation: str = "fp32",
    dx_rounding: str = "once",
    notes: Sequence[str] = (),
) -> dict:
    """An arm's precision class against the Cake contract (module docstring)."""

    same = taps == "float32" and accumulation == "fp32"
    notes = list(notes)
    if taps != "float32":
        notes.append(f"{taps} taps: report only")
    return {
        "activations": "bfloat16",
        "taps": taps,
        "preactivation": preactivation,
        "accumulation": accumulation,
        "dx_rounding": dx_rounding,
        "same_precision": same,
        "notes": notes,
    }


_TAIL_NOTE = "the dx rows sent as history get the returned derivative added after their own BF16 rounding (two roundings)"
CAKE_PRECISION = precision(
    taps="float32",
    preactivation="fp32",
    notes=[
        "the returned head-row derivative is folded into the predecessor's FP32 dx before its one rounding"
    ],
)
DAO_PRECISION = precision(
    taps="float32",
    preactivation="fp32",
    dx_rounding="twice on the history rows",
    notes=[_TAIL_NOTE],
)
PRECISIONS = {
    "fla_cp_triton": precision(
        taps="float32",
        preactivation="bf16",
        dx_rounding="twice on the history rows",
        notes=[
            "SiLU' at the BF16-rounded preactivation (still a same-precision candidate)",
            _TAIL_NOTE,
        ],
    ),
    "fla_cp_gluon": precision(
        taps="float32",
        preactivation="unverified (Gluon kernels)",
        dx_rounding="twice on the history rows",
        notes=[_TAIL_NOTE],
    ),
    "fla_cp_cuda": DAO_PRECISION,
    "megatron_cp": DAO_PRECISION,
    "megatron_a2a_cp": precision(
        taps="float32",
        preactivation="fp32",
        notes=["Dao's kernel over the whole pack: no history rows, one dx rounding"],
    ),
    "attn_gym_cp": precision(taps="bfloat16", preactivation="fp32"),
    "torch_compile_halo": precision(
        taps="float32",
        preactivation="fp32",
        dx_rounding="twice on the history rows",
        notes=[_TAIL_NOTE],
    ),
}


def _snapshot(
    calls: dict[str, Callable], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """One recorded forward and its backward: copies of ``ys`` and of the leaves' gradients (``xs`` then ``ws``)."""

    ys = calls["record"]()
    grads = calls["bwd"]()
    torch.cuda.synchronize(device)
    if any(g is None for g in grads):
        raise RuntimeError("a leaf received no gradient")
    return [y.detach().clone() for y in ys], [g.detach().clone() for g in grads]


def _fp64_rows(
    x: torch.Tensor,
    w: torch.Tensor,
    dy: torch.Tensor,
    cu_t: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP64 ``(y, dx)`` of the global rows ``r`` and their ``dweight`` partial, document-local, from the whole pack
    ``x`` / ``dy`` and the taps ``w``: ``y[g] = silu(pre[g])``, ``pre[g] = sum_k w[k] x[g - 3 + k]``,
    ``dx[g] = sum_e w[3 - e] dpre[g + e]``, ``dw[k] = sum_g dpre[g] x[g - 3 + k]`` (terms outside the document zero)."""

    total = x.shape[0]
    doc = torch.bucketize(r, cu_t[1:], right=True)
    start, end = cu_t[doc], cu_t[doc + 1]
    w64 = w.double()

    def take(t: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, index.clamp(0, total - 1)).double()

    dx = torch.zeros(r.numel(), x.shape[1], dtype=torch.float64, device=x.device)
    y = dpre0 = dx
    for e in range(WIDTH):
        g = r + e
        valid = g < end
        pre = torch.zeros_like(dx)
        for k in range(WIDTH):
            src = g - (HALO - k)
            pre += take(x, src) * w64[:, k] * (valid & (src >= start)).unsqueeze(1)
        sig = torch.sigmoid(pre)
        dpre = take(dy, g) * (sig * (1 + pre * (1 - sig))) * valid.unsqueeze(1)
        dx = dx + dpre * w64[:, HALO - e]
        if e == 0:
            y, dpre0 = pre * sig, dpre
    dw = torch.stack(
        [
            (
                dpre0
                * take(x, r - (HALO - k))
                * ((r - (HALO - k)) >= start).unsqueeze(1)
            ).sum(0)
            for k in range(WIDTH)
        ],
        dim=1,
    )
    return y, dx, dw


def local_checks(
    calls: dict[str, Callable],
    rows: torch.Tensor,
    p: Problem,
    device: torch.device,
    keep_first: bool = False,
) -> dict:
    """This rank's half of the determinism and error checks of an arm (combined over the ranks by
    :func:`combine_checks`): the tensors that varied over ``DETERMINISM_RUNS`` runs, the FP64 error sums of ``y`` /
    ``dx`` on the arm's rows, and the arm's and the FP64 ``dweight`` partials.  ``keep_first``: the first run's
    ``ys`` and gradients are returned under ``"first"`` (not sent to the other ranks)."""

    first = _snapshot(calls, device)
    n = len(first[0])
    varied: set[str] = set()
    for _ in range(DETERMINISM_RUNS - 1):
        ys, grads = _snapshot(calls, device)
        for i in range(n):
            for label, a, b in (
                ("y", ys[i], first[0][i]),
                ("dx", grads[i], first[1][i]),
                ("dweight", grads[n + i], first[1][n + i]),
            ):
                if not torch.equal(_bits(a), _bits(b)):
                    varied.add(label if n == 1 else f"{label}[{i}]")
        del ys, grads
    rows_d = rows.to(device)
    valid = torch.nonzero(rows_d < p.total).squeeze(1)
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    maxima = torch.zeros(2, dtype=torch.float64, device=device)
    dws, dw_refs = [], []
    for i in range(n):
        ref_partial = torch.zeros(
            p.ws[i].shape[0], WIDTH, dtype=torch.float64, device=device
        )
        for begin in range(0, valid.numel(), REF_CHUNK):
            pos = valid[begin : begin + REF_CHUNK]
            y_ref, dx_ref, dw_part = _fp64_rows(
                p.xs_g[i], p.ws[i], p.dys_g[i], p.cu_t, rows_d.index_select(0, pos)
            )
            ref_partial += dw_part
            for slot, out, ref in ((0, first[0][i], y_ref), (1, first[1][i], dx_ref)):
                diff = out.index_select(0, pos).double() - ref
                sums[2 * slot] += diff.square().sum()
                sums[2 * slot + 1] += ref.square().sum()
                maxima[slot] = torch.maximum(maxima[slot], diff.abs().max())
            del y_ref, dx_ref, dw_part
        dws.append(first[1][n + i].double().cpu())
        dw_refs.append(ref_partial.cpu())
    out = {
        "varied": sorted(varied),
        "dtypes": {
            "x": baselines.dtype_name(p.xs[0].dtype),
            "y": baselines.dtype_name(first[0][0].dtype),
            "dx": baselines.dtype_name(first[1][0].dtype),
            "dweight": baselines.dtype_name(first[1][n].dtype),
        },
        "sums": sums.tolist(),
        "maxima": maxima.tolist(),
        "dw": dws,
        "dw_ref": dw_refs,
    }
    if keep_first:
        out["first"] = first
    return out


def _rel_l2(diff: torch.Tensor, ref: torch.Tensor) -> float:
    norm = float(ref.square().sum())
    err = float(diff.square().sum())
    return (err / norm) ** 0.5 if norm > 0 else (0.0 if err == 0 else float("inf"))


def combine_checks(ranks: list[dict], partial: bool = True) -> tuple[dict, dict, dict]:
    """``(determinism, error, dtypes)`` of an arm from every rank's :func:`local_checks`.  ``partial=False``: the
    rank's ``dweight`` is not the sum over its own rows (the all-to-all arms: its channel slice over the whole pack),
    so only the summed ``dweight`` is compared."""

    varied = sorted({label for r in ranks for label in r["varied"]})
    determinism = {
        "runs": DETERMINISM_RUNS,
        "class": "det" if not varied else "nondet",
        "varied": varied,
        "varied_per_rank": [r["varied"] for r in ranks],
    }
    sums = [sum(r["sums"][j] for r in ranks) for j in range(4)]

    def rel(err: float, norm: float) -> float:
        return (err / norm) ** 0.5 if norm > 0 else (0.0 if err == 0 else float("inf"))

    error: dict = {
        "y": {
            "max_abs": max(r["maxima"][0] for r in ranks),
            "rel_l2": rel(sums[0], sums[1]),
        },
        "dx": {
            "max_abs": max(r["maxima"][1] for r in ranks),
            "rel_l2": rel(sums[2], sums[3]),
        },
    }
    dw_max, dw_rel = 0.0, 0.0
    part_max, part_rel, part_ratio = 0.0, 0.0, 0.0
    for i in range(len(ranks[0]["dw"])):
        arm_sum = sum(r["dw"][i] for r in ranks)
        ref_sum = sum(r["dw_ref"][i] for r in ranks)
        diff = arm_sum - ref_sum
        dw_max = max(dw_max, float(diff.abs().max()))
        dw_rel = max(dw_rel, _rel_l2(diff, ref_sum))
        for r in ranks if partial else ():
            d = r["dw"][i] - r["dw_ref"][i]
            ref = r["dw_ref"][i]
            part_max = max(part_max, float(d.abs().max()))
            part_rel = max(part_rel, _rel_l2(d, ref))
            part_ratio = max(
                part_ratio, float((d.abs() / (DW_ATOL + DW_RTOL * ref.abs())).max())
            )
    error["dweight"] = {"max_abs": dw_max, "rel_l2": dw_rel, "summed_over_ranks": True}
    error["dweight_partial"] = (
        {
            "max_abs": part_max,
            "rel_l2": part_rel,
            "budget_ratio": part_ratio,
            "budget": f"{DW_ATOL:g} + {DW_RTOL:g} |ref|",
        }
        if partial
        else {
            "not_applicable": "a rank's dweight is its channel slice over the whole pack"
        }
    )
    error["budget_rel_l2"] = baselines.NUMERICS_REL_L2
    error["ok"] = all(
        error[key]["rel_l2"] < baselines.NUMERICS_REL_L2
        for key in ("y", "dx", "dweight")
    )
    return determinism, error, ranks[0]["dtypes"]


# ---------------------------------------------------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------------------------------------------------


def cake_calls(p: Problem, cooperative: bool) -> dict[str, Callable[[], object]]:
    """The call table of a Cake arm on its own leaves: :func:`baselines.autograd_calls` (``fwd`` / ``record`` /
    ``bwd`` / ``step`` / ``ckpt``) plus ``ckpt_handle`` and the memory probes' one-shot forms (``record_once`` /
    ``bwd_once``, and ``*_memory`` with the leaves' gradients dropped again)."""

    xs = [x.detach().clone().requires_grad_(True) for x in p.xs]
    ws = [w.detach().clone().requires_grad_(True) for w in p.ws]
    leaves = [*xs, *ws]
    n = len(xs)

    def run(t: Sequence[torch.Tensor], handle: Optional[Any]) -> tuple:
        if n == 1:
            return (
                det_kcp_api.cake_causal_conv1d_det_kcp(
                    t[0], t[1], p.ctx, recompute_handle=handle, cooperative=cooperative
                ),
            )
        return tuple(
            det_kcp_api.cake_causal_conv1d_det_kcp_multi(
                t[:n], t[n:], p.ctx, recompute_handle=handle, cooperative=cooperative
            )
        )

    def layer(*t: torch.Tensor) -> tuple:
        return run(t, None)

    def layer_with_handle(*args: Any) -> tuple:
        return run(args[:-1], args[-1])

    calls = baselines.autograd_calls(layer, leaves, p.dys)
    state: dict[str, tuple] = {}

    def reset() -> None:
        for leaf in leaves:
            leaf.grad = None

    def ckpt_handle() -> list:
        reset()
        handle = det_kcp_api.CakeCausalConv1dDetKcpRecomputeHandle()
        ys = checkpoint(layer_with_handle, *leaves, handle, use_reentrant=False)
        torch.autograd.backward(list(ys), list(p.dys))
        return [leaf.grad for leaf in leaves]

    def record_once() -> None:
        state["ys"] = layer(*leaves)

    def bwd_once() -> None:
        reset()
        torch.autograd.backward(list(state.pop("ys")), list(p.dys))

    def dropping(fn: Callable[[], object]) -> Callable[[], None]:
        def run_and_drop() -> None:
            fn()
            reset()

        return run_and_drop

    calls.update(
        ckpt_handle=ckpt_handle,
        record_once=record_once,
        bwd_once=bwd_once,
        step_memory=dropping(calls["step"]),
        ckpt_memory=dropping(calls["ckpt"]),
        ckpt_handle_memory=dropping(ckpt_handle),
    )
    return calls


def _fla_gluon_verdict(p: Problem) -> None:
    """``Unsupported`` unless FLA's dispatch would hand the CP call's forward and backward to its Gluon kernels on
    this rank (the verifiers of the installed FLA, called as the CP Triton path calls them: an initial state)."""

    try:
        from fla.backends import GluonBackend  # type: ignore
    except Exception as exc:  # noqa: BLE001 -- any import failure means no Gluon arm
        raise baselines.Unsupported(
            f"fla.backends is not importable: {type(exc).__name__}: {exc}"
        ) from exc
    if not GluonBackend.is_available():
        raise baselines.Unsupported(
            "FLA GluonBackend.is_available() is False (no triton.experimental.gluon or not NVIDIA)"
        )
    from fla.modules.causal_conv1d.backends import gluon  # type: ignore

    x3 = p.xs[0].unsqueeze(0)
    w = p.ws[0]
    state = torch.zeros(1, w.shape[0], WIDTH, dtype=x3.dtype, device=x3.device)
    with baselines.environment(FLA_GLUON_ENV):
        if not GluonBackend(env_var="FLA_CONV_GLUON").is_enabled():
            raise baselines.Unsupported(
                "FLA Gluon backend not enabled by FLA_CONV_GLUON=1"
            )
        verdicts = (
            gluon.causal_conv1d_fwd_verifier(x=x3, weight=w, initial_state=state),
            gluon.causal_conv1d_bwd_verifier(
                x=x3, dy=x3, dht=None, weight=w, initial_state=state
            ),
        )
    for (accepted, reason), phase in zip(
        verdicts, ("forward", "backward"), strict=True
    ):
        if not accepted:
            raise baselines.Unsupported(
                f"FLA's CP causal_conv1d has no Gluon backend name; with FLA_CONV_GLUON=1 its dispatch offers the "
                f"{phase} to Gluon, whose verifier rejects it ({reason}), so the call runs the Triton kernels "
                "(= fla_cp_triton)"
            )


def _megatron_arm(
    name: str, p: Problem, rank: int, world: int
) -> tuple[dict, tuple[dict, torch.Tensor]]:
    """Megatron-Core's halo ``causal_conv1d_cp`` or the Mamba mixer's all-to-all context parallel (module
    docstring) on Dao's kernel; the ``_det`` forms set ``CAUSAL_CONV1D_DETERMINISTIC=1``."""

    det = name.endswith("_det")
    base = name.removesuffix("_det")
    group = dist.group.WORLD
    entry: dict = {
        "library": baselines._library("megatron.core", "megatron-core"),
        "dao": baselines._library("causal_conv1d", "causal-conv1d"),
        "determinism_claim": "deterministic (CAUSAL_CONV1D_DETERMINISTIC=1: per-chunk FP32 dweight, torch sum)"
        if det
        else "non-deterministic (CAUSAL_CONV1D_DETERMINISTIC=0: FP32 atomic dweight)",
    }
    from megatron.core.ssm.causal_conv1d import (  # type: ignore
        assert_causal_conv1d_deterministic,
        causal_conv1d_cp,
    )

    if det:
        # The guard Megatron's SSM mixers apply at construction under deterministic_mode.
        try:
            with baselines.environment(DAO_ENV[True]):
                assert_causal_conv1d_deterministic(True)
            entry["deterministic_guard"] = (
                "assert_causal_conv1d_deterministic(True) passed"
            )
        except AssertionError as exc:
            entry["deterministic_guard"] = (
                f"assert_causal_conv1d_deterministic(True) failed: {exc}"
            )
    seq_idx = p.doc_ids
    channels = p.row.channels
    if base == "megatron_cp":
        entry["entry"] = (
            "megatron.core.ssm.causal_conv1d.causal_conv1d_cp(x, weight, None, 'silu', cp_group, global_seq_idx)"
        )
        if p.row.layout != "contiguous":
            raise baselines.Unsupported(
                "causal_conv1d_cp takes contiguous equal shards (rank r holds tokens [r T, (r + 1) T)) with the "
                f"documents as global_seq_idx; no {p.row.layout} form"
            )
        if p.total % world:
            raise baselines.Unsupported(
                f"causal_conv1d_cp needs equal shards: {p.total} tokens over {world} ranks"
            )
        share = p.total // world
        rows = torch.arange(rank * share, (rank + 1) * share)

        def conv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            y = causal_conv1d_cp(
                x.unsqueeze(0), w, None, "silu", group, global_seq_idx=seq_idx
            )
            return y.squeeze(0)

    else:
        from causal_conv1d import causal_conv1d_fn  # type: ignore
        from megatron.core.tensor_parallel.mappings import (  # type: ignore
            all_to_all_hp2sp,
            all_to_all_sp2hp,
        )

        entry["entry"] = (
            "megatron.core.tensor_parallel.mappings.all_to_all_sp2hp -> causal_conv1d.causal_conv1d_fn(seq_idx) "
            "-> all_to_all_hp2sp (the Mamba mixer's all-to-all context parallel)"
        )
        counts = [sum(e - s for s, e in ranges) for ranges in p.fragments]
        if len(set(counts)) != 1:
            raise baselines.Unsupported(
                "the all-to-all context parallel needs the same token count on every rank (equal all_to_all "
                f"splits); the {p.row.layout} partition gives {counts}"
            )
        if channels % world or (channels // world) % 8:
            raise baselines.Unsupported(
                f"C = {channels} does not split into {world} channel slices of a multiple of 8 (Dao's channel-last "
                "kernel)"
            )
        width = channels // world
        lo, hi = rank * width, (rank + 1) * width
        rows = p.rows
        # The gathered rows come in rank order; pack order is their sorted global index.
        gathered = torch.cat(
            [torch.arange(s, e) for ranges in p.fragments for s, e in ranges]
        )
        reorder = not torch.equal(gathered, torch.arange(p.total))
        to_pack = torch.argsort(gathered).to(p.device)
        to_ranks = gathered.to(p.device)
        entry["reorder"] = reorder

        def conv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            xh = all_to_all_sp2hp(x, group=group)
            if reorder:
                xh = xh.index_select(0, to_pack)
            yh = causal_conv1d_fn(
                xh.unsqueeze(0).transpose(1, 2),
                w[lo:hi],
                None,
                seq_idx=seq_idx,
                activation="silu",
            )
            yh = yh.transpose(1, 2).squeeze(0)
            if reorder:
                yh = yh.index_select(0, to_ranks)
            return all_to_all_hp2sp(yh.contiguous(), group=group)

    leaves = [baselines._take(x, rows).requires_grad_(True) for x in p.xs_g] + [
        w.detach().clone().requires_grad_(True) for w in p.ws
    ]
    local_dys = [baselines._take(dy, rows) for dy in p.dys_g]

    def layer(*t: torch.Tensor) -> tuple:
        n = len(t) // 2
        return tuple(conv(x, w) for x, w in zip(t[:n], t[n:], strict=True))

    return entry, (baselines.autograd_calls(layer, leaves, local_dys), rows)


class HaloPlan:
    """This rank's rows for :func:`_halo_core`: each fragment (a merged global token range) preceded by three history
    slots, filled from this rank's rows, from rows received from the ranks that hold them (one ``all_to_all_single``;
    receive order: by source rank, then in this rank's fragment order) or from a zero row."""

    def __init__(self, p: Problem, rank: int, world: int) -> None:
        cu = p.cu
        spans = sorted(
            (s, e, r, offset)
            for r, ranges in enumerate(p.fragments)
            for (s, e), offset in zip(
                ranges, _offsets([e - s for s, e in ranges]), strict=True
            )
        )
        starts = [span[0] for span in spans]

        def owner(t: int) -> tuple[int, int]:
            s, _e, r, offset = spans[bisect.bisect_right(starts, t) - 1]
            return r, offset + t - s

        def doc_start(t: int) -> int:
            return cu[bisect.bisect_right(cu, t) - 1]

        needs = []  # per rank: the (source rank, position there, token) of its history rows held elsewhere
        for q, ranges in enumerate(p.fragments):
            need = []
            for s, _e in ranges:
                for t in range(max(doc_start(s), s - HALO), s):
                    source, pos = owner(t)
                    if source != q:
                        need.append((source, pos, t))
            needs.append(need)
        send = [[pos for src, pos, _t in needs[q] if src == rank] for q in range(world)]
        mine = needs[rank]
        self.input_splits = [len(s) for s in send]
        self.output_splits = [
            sum(1 for src, _, _ in mine if src == o) for o in range(world)
        ]
        self.exchange = world > 1 and any(needs)
        self.send_index = torch.tensor(
            [pos for s in send for pos in s], dtype=torch.long, device=p.device
        )
        base = _offsets(self.output_splits)
        seen = [0] * world
        slot: dict[int, int] = {}
        for src, _pos, t in mine:
            slot[t] = base[src] + seen[src]
            seen[src] += 1
        self.n_recv = sum(self.output_splits)
        local = sum(e - s for s, e in p.fragments[rank])
        zero = local + self.n_recv
        ext_src: list[torch.Tensor] = []
        out_index: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        position, offset = 0, 0
        cu_cpu = torch.tensor(cu)
        for s, e in p.fragments[rank]:
            history = []
            for t in range(s - HALO, s):
                if t >= 0 and t >= doc_start(s):
                    source, pos = owner(t)
                    history.append(pos if source == rank else local + slot[t])
                else:
                    history.append(zero)
            ext_src.append(torch.tensor(history, dtype=torch.long))
            ext_src.append(torch.arange(offset, offset + e - s))
            out_index.append(torch.arange(position + HALO, position + HALO + e - s))
            tokens = torch.arange(s, e)
            begin = cu_cpu[torch.bucketize(tokens, cu_cpu[1:], right=True)]
            masks.append(torch.zeros(HALO, WIDTH))
            masks.append(
                torch.stack(
                    [(tokens - (HALO - k)) >= begin for k in range(WIDTH)], dim=1
                ).float()
            )
            position += HALO + e - s
            offset += e - s
        empty = torch.zeros(0, dtype=torch.long)
        self.ext_src = torch.cat(ext_src or [empty]).to(p.device)
        self.out_index = torch.cat(out_index or [empty]).to(p.device)
        self.mask = torch.cat(masks or [torch.zeros(0, WIDTH)]).to(p.device)


def _offsets(counts: Sequence[int]) -> list[int]:
    out, total = [], 0
    for c in counts:
        out.append(total)
        total += c
    return out


def _halo_core(
    x_src: torch.Tensor,
    w: torch.Tensor,
    ext_src: torch.Tensor,
    out_index: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """The document-local width-4 convolution + SiLU of the extended rows in FP32 (row ``j`` reads rows ``j - 3`` to
    ``j``; ``mask`` zeroes the taps that cross a document start), one BF16 rounding, this rank's rows selected."""

    xe = x_src.index_select(0, ext_src).float()
    xp = F.pad(xe, (0, 0, HALO, 0))
    n = xe.shape[0]
    pre = xp[0:n] * w[:, 0] * mask[:, 0:1]
    for k in range(1, WIDTH):
        pre = pre + xp[k : k + n] * w[:, k] * mask[:, k : k + 1]
    y = pre * torch.sigmoid(pre)
    return y.to(torch.bfloat16).index_select(0, out_index)


_COMPILED_HALO_CORE: Optional[Callable] = None


def _torch_compile_halo_arm(
    p: Problem, rank: int, world: int
) -> tuple[dict, tuple[dict, torch.Tensor]]:
    global _COMPILED_HALO_CORE
    if _COMPILED_HALO_CORE is None:
        # ``import torch._dynamo`` would bind ``torch`` as a local of this function (unbound on later calls)
        import torch._dynamo as torch_dynamo

        for name in ("recompile_limit", "cache_size_limit"):
            # one compiled graph per row shape; a recompile past the limit would silently run eagerly
            if hasattr(torch_dynamo.config, name):
                setattr(
                    torch_dynamo.config,
                    name,
                    max(64, getattr(torch_dynamo.config, name)),
                )
        _COMPILED_HALO_CORE = torch.compile(_halo_core, dynamic=False)
    core = _COMPILED_HALO_CORE
    plan = HaloPlan(p, rank, world)
    group = dist.group.WORLD
    entry = {
        "library": f"torch {torch.__version__} (torch.compile, inductor)",
        "entry": "torch.compile(document-local conv + SiLU over [3 history rows + fragment rows]) + "
        "torch.distributed.nn.functional.all_to_all_single halo",
        "determinism_claim": "not claimed (inductor scatter-add in the gather's backward)",
        "halo_rows_received": plan.n_recv,
    }
    leaves = [x.detach().clone().requires_grad_(True) for x in p.xs] + [
        w.detach().clone().requires_grad_(True) for w in p.ws
    ]

    def conv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        parts = [x]
        if plan.exchange:
            recv = x.new_empty((plan.n_recv, x.shape[1]))
            parts.append(
                dist_nn.all_to_all_single(
                    recv,
                    x.index_select(0, plan.send_index),
                    plan.output_splits,
                    plan.input_splits,
                    group=group,
                )
            )
        parts.append(x.new_zeros((1, x.shape[1])))
        return core(torch.cat(parts, 0), w, plan.ext_src, plan.out_index, plan.mask)

    def layer(*t: torch.Tensor) -> tuple:
        n = len(t) // 2
        return tuple(conv(x, w) for x, w in zip(t[:n], t[n:], strict=True))

    return entry, (baselines.autograd_calls(layer, leaves, p.dys), p.rows)


def build_baseline(
    name: str, p: Problem, rank: int, world: int
) -> tuple[dict, Optional[tuple[dict, torch.Tensor]], dict]:
    """``(entry, (calls, rows) or None, env)`` of one baseline on the row (``entry["unsupported"]``: the reason)."""

    entry: dict = {"name": name}
    env: dict = {}
    built = None
    try:
        if name in (
            "fla_cp_triton",
            "fla_cp_gluon",
            "fla_cp_cuda",
            "fla_cp_cuda_det",
            "attn_gym_cp",
        ):
            base = {
                "fla_cp_gluon": "fla_cp_triton",
                "fla_cp_cuda_det": "fla_cp_cuda",
            }.get(name, name)
            if name == "fla_cp_gluon":
                _fla_gluon_verdict(p)
            env = {
                "fla_cp_triton": FLA_TRITON_ENV,
                "fla_cp_gluon": FLA_GLUON_ENV,
                "fla_cp_cuda": DAO_ENV[False],
                "fla_cp_cuda_det": DAO_ENV[True],
            }.get(name, {})
            fla_entry, built = baselines.kcp_arm(
                base, xs=p.xs_g, ws=p.ws, dys=p.dys_g, cu=p.cu, fragments=p.fragments,
                layout=p.row.layout, group=dist.group.WORLD, rank=rank, world=world,
            )  # fmt: skip
            fla_entry.pop("feature", None)
            fla_entry["determinism_claim"] = fla_entry.pop("determinism", None)
            if name == "fla_cp_cuda_det":
                fla_entry["determinism_claim"] = (
                    "deterministic (CAUSAL_CONV1D_DETERMINISTIC=1: per-chunk FP32 dweight, torch sum)"
                )
            entry.update(fla_entry, name=name)
        elif name.startswith("megatron"):
            env = DAO_ENV[name.endswith("_det")]
            info, built = _megatron_arm(name, p, rank, world)
            entry.update(info)
        elif name == "torch_compile_halo":
            info, built = _torch_compile_halo_arm(p, rank, world)
            entry.update(info)
        else:
            raise ValueError(f"unknown baseline {name!r}")
    except baselines.Unsupported as exc:
        entry["unsupported"] = str(exc)
        built = None
    except Exception as exc:  # noqa: BLE001 -- a failing library is reported, the row goes on
        entry["unsupported"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        built = None
    base_name = name.removesuffix("_det")
    entry["precision"] = PRECISIONS[base_name]
    entry["env"] = env
    return entry, built, env


# ---------------------------------------------------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------------------------------------------------


def _clear_package_caches() -> None:
    backend = importlib.import_module(det_conv.CakeCausalConv1dDetFunction.__module__)
    for name in ("_FORWARD_WORKSPACES", "_BACKWARD_WORKSPACES"):
        cache = getattr(backend, name, None)
        if isinstance(cache, dict):
            cache.clear()
    det_kcp_backend.cake_causal_conv1d_det_train_kcp_host_cache_clear()
    gc.collect()
    torch.cuda.empty_cache()


def _memory(calls: dict[str, Callable], device: torch.device) -> dict:
    """First / steady probes of every phase of a Cake arm (``bwd``: of a freshly recorded forward, the graph freed)."""

    probes = {
        "fwd": calls["fwd"],
        "bwd": calls["bwd_once"],
        "fwdbwd": calls["step_memory"],
        "ckpt": calls["ckpt_memory"],
        "ckpt_handle": calls["ckpt_handle_memory"],
    }
    out = {}
    for phase in PHASES:
        _clear_package_caches()
        for when in ("first", "steady"):
            if phase == "bwd":
                calls["record_once"]()
            out[f"{phase}_{when}"] = kcp_bench._probe(probes[phase], device)
    return out


def _stats(values: Sequence[float]) -> tuple[float, list[float]]:
    return _median(values), [min(values), max(values)]


def bench_row(
    row: Row,
    *,
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    ahead: SideLoadAhead,
) -> dict:
    p = Problem(row, rank, world, device)
    result: dict = {
        "preset": row.preset,
        "layout": row.layout,
        "channels": row.channels,
        "tensors": row.tensors,
        "tokens_per_rank": row.tokens_per_rank,
        "tokens": p.total,
        "documents": list(p.lens),
        "tokens_local": _gather(int(p.rows.numel())),
        "halo": _gather(p.halo),
        "round_orders": [],
    }
    entries: dict[str, dict] = {}
    timed: dict[str, dict] = {}
    for name, coop in CAKE_ARMS.items():
        entries[name] = {
            "name": name,
            "library": f"flashinfer {flashinfer.__version__}",
            "entry": "flashinfer.cake_causal_conv1d_det_train_kcp."
            + (
                "cake_causal_conv1d_det_kcp"
                if row.tensors == 1
                else "cake_causal_conv1d_det_kcp_multi"
            )
            + f"(..., cooperative={coop})",
            "cooperative": coop,
            "cooperative_default": det_kcp_backend.COOPERATIVE_DEFAULT,
            "determinism_claim": "deterministic (fixed-order FP32 dweight partial per rank)",
            "precision": CAKE_PRECISION,
            "env": {},
            "eligible": False,
        }
        timed[name] = {"calls": cake_calls(p, coop), "env": {}, "phases": PHASES}
    # One untimed pass per Cake arm: the JIT loads and the first exchanges.
    for t in timed.values():
        calls = t["calls"]
        calls["record"]()
        for phase in PHASES:
            calls[CALL_OF[phase]]()
    torch.cuda.synchronize(device)
    # Cake determinism / error / launch-form checks.
    firsts = {}
    for name in CAKE_ARMS:
        local = local_checks(
            timed[name]["calls"], p.rows, p, device, keep_first=name in ("det_coop", "det_split")
        )  # fmt: skip
        first = local.pop("first", None)
        if first is not None:
            firsts[name] = first
        determinism, error, dtypes = combine_checks(_gather(local))
        entries[name].update(determinism=determinism, error=error, dtypes=dtypes)
    (ys_c, g_c), (ys_s, g_s) = firsts["det_coop"], firsts["det_split"]
    n = row.tensors
    same = {
        "y_bitwise": all(
            torch.equal(_bits(a), _bits(b)) for a, b in zip(ys_c, ys_s, strict=True)
        ),
        "dx_bitwise": all(
            torch.equal(_bits(a), _bits(b))
            for a, b in zip(g_c[:n], g_s[:n], strict=True)
        ),
        "dweight_bitwise": all(
            torch.equal(_bits(a), _bits(b))
            for a, b in zip(g_c[n:], g_s[n:], strict=True)
        ),
    }
    # The checkpoint (plain and with a recompute handle) gives the direct backward's gradients bit for bit.
    same["ckpt_bitwise"] = True
    for name in ("det_coop", "det_split"):
        reference = firsts[name][1]
        for call in ("ckpt", "ckpt_handle"):
            grads = timed[name]["calls"][call]()
            torch.cuda.synchronize(device)
            same["ckpt_bitwise"] &= all(
                torch.equal(_bits(a), _bits(b))
                for a, b in zip(grads, reference, strict=True)
            )
    del firsts, ys_c, g_c, ys_s, g_s
    gathered = _gather(same)
    coop_vs_split = {key: all(r[key] for r in gathered) for key in same}
    checks = {
        "determinism_class_det": all(
            entries[a]["determinism"]["class"] == "det" for a in CAKE_ARMS
        ),
        "coop_split_y_bitwise": coop_vs_split["y_bitwise"],
        "coop_split_dx_bitwise": coop_vs_split["dx_bitwise"],
        # informational: the two forms sum different partial rows, so their dweight bits may differ
        "coop_split_dweight_bitwise": coop_vs_split["dweight_bitwise"],
        "ckpt_and_ckpt_handle_bitwise_direct": coop_vs_split["ckpt_bitwise"],
        "error_ok": all(entries[a]["error"]["ok"] for a in CAKE_ARMS),
        "dweight_partial_within_budget": all(
            entries[a]["error"]["dweight_partial"]["budget_ratio"] <= 1.0
            for a in CAKE_ARMS
        ),
    }
    checks["ok"] = all(
        value for key, value in checks.items() if key != "coop_split_dweight_bitwise"
    )
    result["checks"] = checks
    # The baselines: built, warmed up and checked on every rank; one runs only when every rank can run it.
    for name in args.sota_arms:
        entry, built, env = build_baseline(name, p, rank, world)
        reason = _agreed(entry.get("unsupported"))
        if reason is None and built is not None:
            calls, rows = built
            local: Any
            try:
                with baselines.environment(env):
                    calls["record"]()
                    for phase in BASELINE_PHASES:
                        calls[CALL_OF[phase]]()
                    local = local_checks(calls, rows, p, device)
            except Exception as exc:  # noqa: BLE001 -- a failing library is reported, the row goes on
                local = f"{type(exc).__name__}: {str(exc)[:300]}"
            ranks = _gather(local)
            failures = {i: r for i, r in enumerate(ranks) if isinstance(r, str)}
            if failures:
                distinct = set(failures.values())
                reason = (
                    distinct.pop()
                    if len(distinct) == 1 and len(failures) == world
                    else "; ".join(f"rank {i}: {r}" for i, r in failures.items())
                )
            else:
                determinism, error, dtypes = combine_checks(
                    ranks, partial=not name.startswith("megatron_a2a")
                )
                entry.update(determinism=determinism, error=error, dtypes=dtypes)
                entry["tokens_local"] = _gather(int(rows.numel()))
        if reason is not None:
            entry["unsupported"] = reason
            entry["eligible"] = False
            entries[name] = entry
            continue
        entry["eligible"] = bool(
            entry["precision"]["same_precision"] and entry["error"]["ok"]
        )
        entries[name] = entry
        timed[name] = {"calls": calls, "env": env, "phases": BASELINE_PHASES}
    p.release_pack()
    # Initialization and memory of the two launch forms.
    for name in ("det_coop", "det_split"):
        entries[name]["memory"] = _memory(timed[name]["calls"], device)
    for t in timed.values():
        with baselines.environment(t["env"]):
            t["calls"]["record"]()
    torch.cuda.synchronize(device)
    _time_rounds(result, entries, timed, args=args, device=device, ahead=ahead)
    timed.clear()
    result["arms"] = entries
    _summarize(result, entries)
    del p
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _time_rounds(
    result: dict,
    entries: dict[str, dict],
    timed: dict[str, dict],
    *,
    args: argparse.Namespace,
    device: torch.device,
    ahead: SideLoadAhead,
) -> None:
    """Interleaved rounds of every phase of every timed arm (module docstring); fills each entry's ``*_us``,
    ``*_us_range``, ``per_rank_us``, ``side_load``, ``ahead_late`` and ``rounds`` (the raw per-round values)."""

    arms = list(timed)
    rounds = args.rounds or len(arms)
    result["rounds"] = rounds
    samples: dict[str, dict[str, list[float]]] = {a: {} for a in arms}
    per_rank: dict[str, dict[str, list[list[float]]]] = {a: {} for a in arms}
    late = {a: 0 for a in arms}

    def keep(
        arm: str, key: str, value: float, ranks: Optional[list[float]] = None
    ) -> None:
        samples[arm].setdefault(key, []).append(value * 1e3)
        if ranks is not None:
            per_rank[arm].setdefault(key, []).append([v * 1e3 for v in ranks])

    for index in range(rounds):
        shift = index % len(arms)
        order = arms[shift:] + arms[:shift]
        if (index // len(arms)) % 2:
            order.reverse()
        result["round_orders"].append(list(order))
        spin_ms = ahead.spin_for(
            [
                _in_env(timed[a]["env"], timed[a]["calls"][CALL_OF[phase]])
                for a in order
                for phase in timed[a]["phases"]
            ]
        )
        for arm in order:
            t = timed[arm]
            with baselines.environment(t["env"]):
                for phase in t["phases"]:
                    fn = t["calls"][CALL_OF[phase]]
                    gpu = kcp_bench._gpu_ms(fn)
                    keep(arm, f"{phase}_gpu", max(gpu), gpu)
                    pairs = _gather(ahead(fn, spin_ms))
                    late[arm] += sum(lt for _, lt in pairs)
                    values = [ms for ms, _ in pairs]
                    keep(arm, f"{phase}_ahead", max(values), values)
                    if phase in args.side_phases:
                        for label, kernels, kernel_ms in args.loads:
                            pairs = _gather(
                                ahead.loaded(fn, spin_ms, kernels, kernel_ms)
                            )
                            late[arm] += sum(lt for _, lt in pairs)
                            values = [ms for ms, _ in pairs]
                            keep(arm, f"{phase}_ahead_{label}", max(values), values)
                    if not args.skip_wall:
                        wall, host = kcp_bench._wall_ms(fn, device)
                        keep(arm, f"{phase}_wall", wall)
                        keep(arm, f"{phase}_host", host)
    for arm in arms:
        entry = entries[arm]
        for key, values in samples[arm].items():
            entry[f"{key}_us"], entry[f"{key}_us_range"] = _stats(values)
        entry["per_rank_us"] = per_rank[arm]
        entry["rounds_us"] = samples[arm]
        entry["ahead_late"] = late[arm]
        side: dict = {}
        for phase in timed[arm]["phases"]:
            unloaded = samples[arm][f"{phase}_ahead"]
            for label, kernels, kernel_ms in args.loads:
                loaded = samples[arm].get(f"{phase}_ahead_{label}")
                if loaded is None:
                    continue
                ratios = [lo / un for lo, un in zip(loaded, unloaded, strict=True)]
                ratio, ratio_range = _stats(ratios)
                loaded_us, loaded_range = _stats(loaded)
                side.setdefault(phase, {})[label] = {
                    "kernels": kernels,
                    "kernel_ms": kernel_ms,
                    "loaded_us": loaded_us,
                    "loaded_us_range": loaded_range,
                    "ratio": ratio,
                    "ratio_range": ratio_range,
                    "loaded_scaled_us": ratio * _median(unloaded),
                }
        entry["side_load"] = side


def _phase_of(key: str) -> str:
    """The phase of a ``{phase}_{measure}`` key (the longest matching phase name: ``ckpt_handle`` before ``ckpt``)."""

    return next(
        phase
        for phase in sorted(PHASES, key=len, reverse=True)
        if key.startswith(phase + "_")
    )


def _ratio_block(entries: dict, num: str, den: str, key: str) -> Optional[dict]:
    a, b = (
        entries[num].get("rounds_us", {}).get(key),
        entries[den].get("rounds_us", {}).get(key),
    )
    if not a or not b:
        return None
    ratios = [x / y for x, y in zip(a, b, strict=True)]
    return {
        "ratio_of_medians": _median(a) / _median(b),
        "median_ratio": _median(ratios),
        "range": [min(ratios), max(ratios)],
    }


def _fastest(
    entries: dict[str, dict], names: Sequence[str], key: str
) -> Optional[dict]:
    timed = [(entries[n][f"{key}_us"], n) for n in names if f"{key}_us" in entries[n]]
    if not timed:
        return None
    us, name = min(timed)
    return {
        "arm": name,
        "us": us,
        "library": entries[name].get("library"),
        "determinism": entries[name]["determinism"]["class"],
    }


def _summarize(result: dict, entries: dict[str, dict]) -> None:
    """``best_baseline`` (and ``best_det_baseline``: among the measured-``det`` ones), ``vs_best``,
    ``split_vs_coop`` and ``side_criterion`` of a row (module docstring)."""

    cake = list(CAKE_ARMS)
    eligible = [
        n for n, e in entries.items() if n not in CAKE_ARMS and e.get("eligible")
    ]
    eligible_det = [n for n in eligible if entries[n]["determinism"]["class"] == "det"]
    keys = sorted(
        {k for a in cake for k in entries[a].get("rounds_us", {})},
        key=lambda k: (PHASES.index(_phase_of(k)), k),
    )
    best: dict = {}
    best_det: dict = {}
    for key in keys:
        for table, names in ((best, eligible), (best_det, eligible_det)):
            fastest = _fastest(entries, names, key)
            if fastest is not None:
                table[key] = fastest
    result["best_baseline"] = best
    result["best_det_baseline"] = best_det
    excluded = {
        n: e.get("unsupported")
        or "; ".join(
            reason
            for reason, bad in (
                ("not same precision", not e["precision"]["same_precision"]),
                ("error check failed", not e.get("error", {}).get("ok", False)),
            )
            if bad
        )
        for n, e in entries.items()
        if n not in CAKE_ARMS and n not in eligible
    }
    result["excluded_baselines"] = excluded
    vs_best: dict = {}
    for arm in cake:
        vs_best[arm] = {}
        for key, b in best.items():
            block = _ratio_block(entries, b["arm"], arm, key)
            if block is not None:
                vs_best[arm][key] = {
                    "baseline": b["arm"],
                    "baseline_us": b["us"],
                    "arm_us": entries[arm][f"{key}_us"],
                    "best_over_arm": block["ratio_of_medians"],
                    "best_over_arm_range": block["range"],
                }
    result["vs_best"] = vs_best
    split: dict = {}
    for key in keys:
        block = _ratio_block(entries, "det_split", "det_coop", key)
        aa = _ratio_block(entries, "det_split_aa", "det_split", key)
        if block is None:
            continue
        split[key] = {
            "coop_us": entries["det_coop"][f"{key}_us"],
            "split_us": entries["det_split"][f"{key}_us"],
            "split_over_coop": block["ratio_of_medians"],
            "split_over_coop_median_ratio": block["median_ratio"],
            "split_over_coop_range": block["range"],
            "within_2pct": block["ratio_of_medians"] <= SPLIT_TOLERANCE,
            "aa_over_split": None if aa is None else aa["median_ratio"],
            "aa_over_split_range": None if aa is None else aa["range"],
        }
    result["split_vs_coop"] = split
    criterion: dict = {}
    for name, entry in entries.items():
        bwd = entry.get("side_load", {}).get("bwd", {})
        rows = {
            label: {"ratio": v["ratio"], "pass": v["ratio"] <= SIDE_STREAM_LIMIT}
            for label, v in bwd.items()
            if v["kernels"] == 1
        }
        if rows:
            criterion[name] = {
                "loads": rows,
                "pass": all(v["pass"] for v in rows.values()),
                "limit": SIDE_STREAM_LIMIT,
            }
    result["side_criterion"] = criterion


# ---------------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------------


def _fmt_range(r: Sequence[float]) -> str:
    return f"[{r[0]:.3f}, {r[1]:.3f}]"


def _describe(entry: dict) -> str:
    if "unsupported" in entry:
        return f"{entry['name']}: unsupported ({entry['unsupported']})"
    prec = entry["precision"]
    det = entry["determinism"]
    err = entry["error"]
    varied = f" (varied: {', '.join(det['varied'])})" if det["varied"] else ""
    partial = err["dweight_partial"]
    budget = f"{partial['budget_ratio']:.3f}" if "budget_ratio" in partial else "n/a"
    tail = (
        ""
        if entry["name"] in CAKE_ARMS
        else "; " + ("eligible" if entry.get("eligible") else "not eligible")
    )
    return (
        f"{entry['name']}: {entry.get('library')}; taps {prec['taps']}, SiLU' at {prec['preactivation']}, "
        f"acc {prec['accumulation']}, same_precision {prec['same_precision']}; {det['class']}{varied}; "
        f"max-abs/rel-L2 y {err['y']['max_abs']:.2e}/{err['y']['rel_l2']:.1e} dx {err['dx']['max_abs']:.2e}/"
        f"{err['dx']['rel_l2']:.1e} dw(sum) {err['dweight']['max_abs']:.2e}/{err['dweight']['rel_l2']:.1e} "
        f"dw-partial budget {budget}" + tail
    )


def format_row(
    result: dict, world: int, loads: Sequence[tuple[str, int, float]]
) -> list[str]:
    head = f"W={world} {result['preset']} {result['layout']} C={result['channels']}" + (
        f" x{result['tensors']}" if result["tensors"] > 1 else ""
    )
    lines = []
    entries = result["arms"]
    for phase in PHASES:
        for name, entry in entries.items():
            key = f"{phase}_{COLUMN}"
            if f"{key}_us" not in entry:
                continue
            line = (
                f"{head} {phase:11s} {name:19s} {COLUMN} {entry[f'{key}_us']:9.2f} "
                f"[{entry[f'{key}_us_range'][0]:9.2f}, {entry[f'{key}_us_range'][1]:9.2f}] "
                f"gpu {entry[f'{phase}_gpu_us']:9.2f}"
            )
            if f"{phase}_wall_us" in entry:
                line += f" wall {entry[f'{phase}_wall_us']:9.2f}"
            for label, _, _ in loads:
                v = entry.get("side_load", {}).get(phase, {}).get(label)
                if v is not None:
                    line += f" | {label} {v['loaded_us']:9.2f} x{v['ratio']:5.2f}"
            lines.append(line)
        key = f"{phase}_{COLUMN}"
        parts = []
        best = result["best_baseline"].get(key)
        if best is not None:
            parts.append(
                f"best {best['arm']} ({best['determinism']}) {best['us']:9.2f}"
            )
            for arm in CAKE_ARMS:
                v = result["vs_best"][arm].get(key)
                if v is not None:
                    parts.append(
                        f"best/{arm} {v['best_over_arm']:5.3f} {_fmt_range(v['best_over_arm_range'])}"
                    )
        else:
            parts.append("best - (no eligible baseline)")
        best_det = result["best_det_baseline"].get(key)
        if best_det is not None and (best is None or best_det["arm"] != best["arm"]):
            parts.append(f"best det {best_det['arm']} {best_det['us']:9.2f}")
        s = result["split_vs_coop"].get(key)
        if s is not None:
            parts.append(
                f"split/coop {s['split_over_coop']:5.3f} {_fmt_range(s['split_over_coop_range'])} "
                f"within2% {s['within_2pct']}"
            )
            if s["aa_over_split_range"] is not None:
                parts.append(f"aa/split {_fmt_range(s['aa_over_split_range'])}")
        lines.append(f"{head} {phase:11s} summary [{COLUMN}] " + " | ".join(parts))
    for name, c in result["side_criterion"].items():
        loads_text = " ".join(
            f"{label} x{v['ratio']:.2f} {'ok' if v['pass'] else 'FAIL'}"
            for label, v in c["loads"].items()
        )
        lines.append(
            f"{head} side-stream bwd {name:19s} {loads_text} -> {'pass' if c['pass'] else 'fail'}"
        )
    for name in ("det_coop", "det_split"):
        memory = entries[name].get("memory", {})
        first, steady = memory.get("bwd_first"), memory.get("bwd_steady")
        if first and steady:
            lines.append(
                f"{head} {name} bwd first {first['wall_us']:9.2f} us peak temp {first['peak_temp_bytes']} B | "
                f"steady {steady['wall_us']:9.2f} us peak temp {steady['peak_temp_bytes']} B | "
                f"ckpt_handle steady peak temp {memory['ckpt_handle_steady']['peak_temp_bytes']} B | "
                f"ahead_late {entries[name].get('ahead_late')}"
            )
    lines.extend(f"{head} arm {_describe(entry)}" for entry in entries.values())
    lines.append(
        f"{head} tokens per rank {result['tokens_local']} checks {result['checks']}"
    )
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows", choices=("customer", "pr8365", "custom"), default="customer"
    )
    parser.add_argument(
        "--layouts",
        type=lambda s: [v for v in s.split(",") if v],
        default=list(LAYOUTS),
        help=f"comma-separated KCP layouts of the customer / custom rows (default {','.join(LAYOUTS)})",
    )
    parser.add_argument(
        "--channels", type=int, default=CUSTOMER_SINGLE[0], help="custom rows: C"
    )
    parser.add_argument(
        "--tokens-per-rank",
        type=int,
        default=CUSTOMER_SINGLE[1],
        help="custom rows: tokens per rank",
    )
    parser.add_argument(
        "--docs",
        type=lambda s: [int(v) for v in s.split(",") if v],
        default=list(CUSTOMER_SINGLE[2]),
        help="custom rows: the comma-separated document pack the W-rank documents repeat",
    )
    parser.add_argument(
        "--tensors", type=int, default=1, help="custom rows: tensors per layer"
    )
    parser.add_argument(
        "--sota",
        nargs="*",
        default=["all"],
        help=f"baselines: all (default), none, or names from {list(BASELINE_ARMS)}",
    )
    parser.add_argument(
        "--side-load",
        default=SIDE_LOADS,
        help=f"comma-separated side loads NxDms (N one-thread kernels of D ms); default {SIDE_LOADS}",
    )
    parser.add_argument(
        "--side-load-phases",
        default=",".join(PHASES),
        help="comma-separated phases timed under the side loads (default: every phase)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="interleaved timing rounds (default: the number of timed arms, one balanced block)",
    )
    parser.add_argument(
        "--skip-wall", action="store_true", help="skip the wall / host measures"
    )
    parser.add_argument(
        "--json", default=None, help="rank 0 writes all rows to this JSON file"
    )
    args = parser.parse_args(argv)
    unknown = [layout for layout in args.layouts if layout not in LAYOUTS]
    if unknown:
        parser.error(f"unknown layouts {unknown}; known {list(LAYOUTS)}")
    if args.rounds < 0 or args.tensors < 1:
        parser.error("--rounds must be >= 0 and --tensors >= 1")
    args.sota_arms = baselines.parse_arms(args.sota, BASELINE_ARMS)
    args.loads = parse_side_loads(args.side_load)
    args.side_phases = [v for v in args.side_load_phases.split(",") if v]
    unknown = [phase for phase in args.side_phases if phase not in PHASES]
    if unknown:
        parser.error(f"unknown --side-load-phases {unknown}; known {list(PHASES)}")
    rank, world, device = _init()
    if not det_conv.is_cake_causal_conv1d_det_train_supported(device):
        raise SystemExit(
            "the deterministic package has no exported route on this device (or is not rendered)"
        )
    rows = preset_rows(args, world)
    ahead = SideLoadAhead(device, max([k for _, k, _ in args.loads], default=0))
    environment = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "flashinfer": flashinfer.__version__,
        "world": world,
        "hosts": sorted(set(_gather(os.uname().nodename))),
        "cooperative_default": det_kcp_backend.COOPERATIVE_DEFAULT,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "library_targets": LIBRARY_TARGETS,
        "sota_arms": args.sota_arms,
        "side_loads": [label for label, _, _ in args.loads],
        "side_load_phases": args.side_phases,
        "resident_us": RESIDENT_US,
        "gpu_timer": "cuda_event",
        "comparison_column": COLUMN,
    }
    if rank == 0:
        print(
            f"device {environment['device']} CUDA {torch.version.cuda} torch {torch.__version__} W={world} "
            f"rows {args.rows} side loads {environment['side_loads']} cooperative default "
            f"{det_kcp_backend.COOPERATIVE_DEFAULT}",
            flush=True,
        )
    results: list[dict] = []

    def write_json(complete: bool) -> None:
        # Rewritten after every row so a later failure keeps the finished rows.
        if rank != 0 or not args.json:
            return
        with open(args.json, "w") as handle:
            json.dump(
                {**environment, "complete": complete, "rows": results}, handle, indent=1
            )

    for row in rows:
        result = bench_row(
            row, args=args, rank=rank, world=world, device=device, ahead=ahead
        )
        results.append(result)
        if rank == 0:
            for line in format_row(result, world, args.loads):
                print(line, flush=True)
        write_json(complete=False)
    ok = all(r["checks"]["ok"] for r in results)
    if rank == 0 and not ok:
        print(
            "CHECK FAILED: "
            + ", ".join(
                f"{r['preset']} {r['layout']} C={r['channels']} x{r['tensors']}"
                for r in results
                if not r["checks"]["ok"]
            ),
            flush=True,
        )
    write_json(complete=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
