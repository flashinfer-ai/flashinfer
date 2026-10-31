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

Paired benchmark of the experimental MiniMax-M3 ICP indexer kernels against the MSA path on one GPU.

The timed process is rank 0 of a TP2 / TP4 group (4 index heads, head dim 128,
init 0 / local 1 forced blocks, max_model_len 172032).  Rows:

* ``decode``: one sparse layer's decode chain of rank 0 -- the scorer, the
  Top16 select-and-publish and owner 0's merge -- for TP {2, 4} x P {128, 256}
  x batch {1, 2, 4, 14} x decode query length {1, 4} x context
  {8192, 163840, 172031}.  The other ranks' rows are published before the
  timing into fresh receive slots (one slot per call), so the timed call holds
  only rank 0's kernels.  ``--protocols`` selects the receive rows: ``tagged``
  (generation-tagged int64 rows; the serving decode default) and ``legacy``
  (int32 rows with a ready flag; used when a prefill shares the step).
* ``prefill``: one 8192-token chunk at prefix {0, 32768, 81920, 155648}: the
  device plan, the scorer, the carrier select and the prefill merge of the
  exchanged rows (the all-to-all is not timed).
* ``mixed``: 14 decode tokens (context 163840, legacy rows) plus one
  8192-token chunk at prefix {81920, 155648}.

``--roles`` adds one row per kernel (score / publish / merge, plan / score /
select / merge) next to each chain.  ``--fused`` times the FI decode chain with
the fused scorer-publisher (an option; the default is the score plane and the
select publish).

Baseline ("MSA path"): vLLM's MiniMax-M3 CuTe DSL decode scorer (with
``prefetch_history``), the MSA sharded-indexer kernels of
``vllm.third_party.fmha_sm100`` (device plan, prefill scorer, prefill select,
decode select-and-publish) and vLLM's Triton merge, called as vLLM's indexer
calls them.  Both arms run on the same inputs, GPU and process, in alternating
sessions (MSA-FI-FI-MSA, then FI-MSA-MSA-FI, ``--groups`` times).  A session is
``bench_gpu_time`` with CUPTI (install ``cupti-python``; without it the
timer falls back to CUDA events and the caption says so) and a cold L2
(``--iters`` steady samples after ``--warmup``).  Each timed call replays one
CUDA graph of the row, as the serving decode step does; ``--eager`` launches
from the host instead, which adds each entry point's host time.  A row
reports the median of each arm's session medians.  The script prints a
Markdown table with both arms' absolute times and the ratio, and the GPU,
architecture, SM count, driver, CUDA version and timer.

    python benchmarks/bench_cake_msa_icp_indexer.py --rows decode prefill mixed --json out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
from typing import Callable, Optional

import torch

from flashinfer.experimental.cake_msa_icp_indexer import cake_jit
from flashinfer.msa_ops import (
    msa_icp_comm_layout,
    msa_icp_decode_index_score_sharded,
    msa_icp_decode_pool_bytes,
    msa_icp_decode_score_columns,
    msa_icp_decode_score_publish,
    msa_icp_index_score_columns,
    msa_icp_index_score_plan_storage,
    msa_icp_index_score_sharded,
    msa_icp_index_score_sharded_plan_device,
    msa_icp_merge_candidates,
    msa_icp_topk_select_prefill,
    msa_icp_topk_select_publish,
)
from flashinfer.testing import bench_gpu_time

HEADS, DIM, TOPK = 4, 128, 16
INIT, LOCAL = 0, 1
CHUNK = 8192
MAX_MODEL_LEN = 172032
SPIN = 1 << 26
BASELINE = "MSA path"


def _fp8(shape, device) -> torch.Tensor:
    return torch.randint(-4, 5, shape, device=device, dtype=torch.int8).to(
        torch.float8_e4m3fn
    )


def _reference() -> Optional[dict]:
    """The MSA path's entry points, or ``None`` when they are not importable."""
    try:
        from vllm.models.minimax_m3.nvidia.ops import index_decode_score_sharded as vd
        from vllm.third_party.fmha_sm100 import api
    except ImportError:
        return None
    return {"api": api, "vd": vd}


class Slots:
    """Fresh receive slots of one arm: call ``i`` uses slot ``i % 3`` of storage
    ``i // 3`` on every rank, so no call waits on a row a previous call left.
    Every counter of a slot advances once per use, in call order, so the peers'
    rows published before the timing carry the generation the timed merge expects."""

    def __init__(self, world, hpr, capacity, calls, device):
        layout = msa_icp_comm_layout(world, hpr, capacity)
        self.world, self.calls, self.next = world, calls, 0
        words = (layout["storage_words"] + 3) // 4 * 4  # 16-byte aligned rank storage
        n = (calls + 2) // 3
        self.storage = torch.zeros(n, world, words, dtype=torch.int32, device=device)
        legacy_n = 1
        for d in layout["legacy_shape"]:
            legacy_n *= d
        tagged_off = layout["tagged_offset_bytes"] // 4
        tagged_n = 1
        for d in layout["tagged_shape"]:
            tagged_n *= d
        self.legacy = self.storage[..., :legacy_n].unflatten(-1, layout["legacy_shape"])
        self.tagged = (
            self.storage[..., tagged_off : tagged_off + 2 * tagged_n]
            .view(torch.int64)
            .unflatten(-1, layout["tagged_shape"])
        )
        i64 = dict(dtype=torch.int64, device=device)
        self.legacy_bases = [
            torch.tensor([self.legacy[j, o].data_ptr() for o in range(world)], **i64)
            for j in range(n)
        ]
        self.tagged_bases = [
            torch.tensor([self.tagged[j, o].data_ptr() for o in range(world)], **i64)
            for j in range(n)
        ]
        i32 = dict(dtype=torch.int32, device=device)
        self.generations = torch.zeros(
            world, *layout["generations_shape"], **i32
        )  # per publishing rank
        self.expected = torch.zeros(
            world, *layout["expected_shape"], **i32
        )  # per owner

    def take(self) -> tuple[int, int]:
        if self.next >= self.calls:
            raise RuntimeError(
                "receive slots exhausted; raise --warmup/--iters headroom"
            )
        i = self.next
        self.next += 1
        return i // 3, i % 3


class Decode:
    """Rank 0's decode layer; ``part`` is ``chain`` or one role."""

    def __init__(self, shape, protocol, part, fused, inputs, calls, device):
        self.world, self.page, self.batch, self.dql, self.ctx = shape
        self.protocol, self.part, self.fused = protocol, part, fused
        self.tagged = protocol == "tagged"
        self.hpr = HEADS // self.world
        self.t = self.batch * self.dql
        for key, value in inputs.items():
            setattr(self, key, value)
        self.index_scale = 2 if self.page == 256 else 1
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        self.output = torch.empty(
            self.t, self.hpr, TOPK, dtype=torch.int32, device=device
        )
        self.pool = (
            torch.empty(
                msa_icp_decode_pool_bytes(max_decode_tokens=self.t),
                dtype=torch.uint8,
                device=device,
            )
            if fused
            else None
        )
        uses_slots = part in ("chain", "publish", "merge", "score_publish")
        self.slots = (
            Slots(self.world, self.hpr, self.t, calls, device) if uses_slots else None
        )
        self.calls = calls

    @staticmethod
    def inputs(shape, device) -> dict:
        world, page, batch, dql, ctx = shape
        t = batch * dql
        pages = (ctx + page - 1) // page
        table_cols = (MAX_MODEL_LEN + page - 1) // page
        block_table = torch.zeros(batch, table_cols, dtype=torch.int32, device=device)
        block_table[:, :pages] = torch.randperm(
            batch * pages, device=device, dtype=torch.int32
        ).view(batch, pages)
        pos = torch.arange(ctx - dql, ctx, device=device).repeat(batch)
        cols = msa_icp_decode_score_columns(MAX_MODEL_LEN, page)
        return {
            "q": _fp8((t, HEADS, DIM), device),
            "k": _fp8((batch * pages, page // world, DIM), device),
            "block_table": block_table,
            "seq_lens": torch.full((batch,), ctx, dtype=torch.int32, device=device),
            "nvalid": (pos // 128 + 1).to(torch.int32),
            "scores": torch.randn(t, HEADS, cols, device=device),
            "peer_scores": torch.randn(t, HEADS, cols, device=device),
        }

    def offset(self, rank):
        return rank * self.index_scale // self.world

    # ---- one kernel of one arm
    def _score(self, arm, ref):
        if arm == "fi":
            msa_icp_decode_index_score_sharded(
                self.q,
                self.k,
                self.block_table,
                self.seq_lens,
                self.scores,
                rank=0,
                world_size=self.world,
                page_size=self.page,
                decode_query_len=self.dql,
                max_decode_query_len=self.dql,
                enable_pdl=True,
            )
        else:
            ref["vd"].minimax_m3_index_decode_score_sharded_cutedsl(
                self.q,
                self.k,
                self.block_table,
                self.seq_lens,
                0,
                self.dql,
                self.dql,
                self.scores,
                page_size=self.page,
                world_size=self.world,
                use_pdl=True,
                prefetch_history=True,
            )

    def _bases(self, j):
        return (self.slots.tagged_bases if self.tagged else self.slots.legacy_bases)[j]

    def _publish(self, arm, ref, rank, j, slot, scores):
        gens = self.slots.generations[rank] if self.tagged else None
        if arm == "fi":
            msa_icp_topk_select_publish(
                scores,
                self.nvalid,
                self._bases(j),
                self.errors,
                slot=slot,
                rank=rank,
                world_size=self.world,
                heads_per_rank=self.hpr,
                capacity=self.t,
                force_begin_blocks=INIT,
                force_end_blocks=LOCAL,
                spin_limit=SPIN,
                enable_pdl=True,
                generations=gens,
                index_scale=self.index_scale,
                index_offset=self.offset(rank),
            )
        else:
            ref["vd"].minimax_m3_index_select_publish_candidates(
                scores,
                self.nvalid,
                self._bases(j),
                slot,
                rank,
                self.t,
                self.errors,
                world_size=self.world,
                heads_per_rank=self.hpr,
                init_blocks=INIT,
                local_blocks=LOCAL,
                spin_limit=SPIN,
                use_pdl=True,
                generations=gens,
                index_scale=self.index_scale,
                index_offset=self.offset(rank),
            )

    def _score_publish(self, j, slot):
        msa_icp_decode_score_publish(
            self.q,
            self.k,
            self.block_table,
            self.seq_lens,
            self.nvalid,
            self.pool,
            self._bases(j),
            self.errors,
            rank=0,
            world_size=self.world,
            page_size=self.page,
            decode_query_len=self.dql,
            slot=slot,
            capacity=self.t,
            force_begin_blocks=INIT,
            force_end_blocks=LOCAL,
            spin_limit=SPIN,
            generations=self.slots.generations[0] if self.tagged else None,
            enable_pdl=True,
        )

    def _merge(self, arm, ref, j, slot):
        recv = (self.slots.tagged if self.tagged else self.slots.legacy)[j, 0, slot]
        expected = self.slots.expected[0, slot] if self.tagged else None
        if arm == "fi":
            msa_icp_merge_candidates(
                recv,
                self.nvalid,
                self.output,
                self.errors,
                init_blocks=INIT,
                local_blocks=LOCAL,
                spin_limit=SPIN,
                enable_pdl=True,
                expected_generations=expected,
            )
        else:
            ref["vd"].minimax_m3_index_merge_candidates(
                recv,
                self.nvalid,
                self.output,
                self.errors,
                init_blocks=INIT,
                local_blocks=LOCAL,
                spin_limit=SPIN,
                use_pdl=True,
                expected_generations=expected,
            )

    # ---- setup and the timed call
    def prepare(self, arm, ref) -> None:
        """Publish the rows the timed merges read: peers for a chain, every rank
        for a merge row.  Run before any timed call of this arm."""
        if self.part not in ("chain", "merge"):
            return
        ranks = range(1, self.world) if self.part == "chain" else range(self.world)
        for i in range(self.calls):
            j, slot = i // 3, i % 3
            for rank in ranks:
                self._publish(
                    arm,
                    ref,
                    rank,
                    j,
                    slot,
                    self.scores if rank == 0 else self.peer_scores,
                )
        torch.cuda.synchronize()

    def run(self, arm, ref) -> None:
        part = self.part
        if part == "score":
            self._score(arm, ref)
            return
        j, slot = self.slots.take()
        if part == "publish":
            self._publish(arm, ref, 0, j, slot, self.scores)
        elif part == "score_publish":
            if arm == "fi":
                self._score_publish(j, slot)
            else:
                self._score(arm, ref)
                self._publish(arm, ref, 0, j, slot, self.scores)
        elif part == "merge":
            self._merge(arm, ref, j, slot)
        else:  # chain
            if self.fused and arm == "fi":
                self._score_publish(j, slot)
            else:
                self._score(arm, ref)
                self._publish(arm, ref, 0, j, slot, self.scores)
            self._merge(arm, ref, j, slot)


class Prefill:
    """One 8192-token chunk of one request at ``prefix`` on rank 0."""

    def __init__(self, world, page, prefix, part, device):
        self.world, self.page, self.part = world, page, part
        self.hpr = HEADS // world
        kv = prefix + CHUNK
        pages = (kv + page - 1) // page
        self.q = _fp8((CHUNK, HEADS, DIM), device)
        self.k = _fp8((pages, page // world, DIM), device)
        self.block_table = torch.randperm(pages, device=device, dtype=torch.int32).view(
            1, pages
        )
        i32 = dict(dtype=torch.int32, device=device)
        self.plan_args = (
            torch.tensor([CHUNK], **i32),
            torch.tensor([kv], **i32),
            torch.tensor([0, CHUNK], **i32),
            torch.tensor([0, kv], **i32),
            torch.tensor([prefix], **i32),
            HEADS,
        )
        storage = msa_icp_index_score_plan_storage(
            num_qo_heads=HEADS, total_q=CHUNK, batch=1, device=device
        )
        self.plan_kwargs = dict(
            total_q=CHUNK,
            max_query_len=CHUNK,
            max_seq_len=kv,
            packed_work_range=torch.empty(
                storage["work_range_numel"], dtype=torch.int64, device=device
            ),
            packed_work_info=torch.empty(
                storage["work_info_numel"], dtype=torch.int64, device=device
            ),
            workspace=torch.empty(
                storage["workspace_bytes"], dtype=torch.uint8, device=device
            ),
        )
        cols = (msa_icp_index_score_columns(kv, page) + 15) // 16 * 16
        self.scores = torch.empty(CHUNK, HEADS, cols, device=device)
        self.nvalid = ((prefix + torch.arange(CHUNK, device=device)) // 128 + 1).to(
            torch.int32
        )
        self.carrier = torch.zeros(HEADS, CHUNK, 2 * TOPK + 1, **i32)
        self.output = torch.empty(CHUNK, self.hpr, TOPK, **i32)
        self.errors = torch.zeros(1, **i32)
        self.index_scale = 2 if page == 256 else 1
        self.plans = {}

    def prepare(self, arm, ref) -> None:
        self.plans[arm] = self._plan(arm, ref)
        self.scores.uniform_(-4, 4)
        self._select(arm, ref)
        torch.cuda.synchronize()

    def _plan(self, arm, ref):
        if arm == "fi":
            return msa_icp_index_score_sharded_plan_device(
                *self.plan_args, **self.plan_kwargs
            )
        return ref["api"].fmha_index_score_sharded_plan_device(
            *self.plan_args, **self.plan_kwargs
        )

    def _score(self, arm, ref, plan):
        if arm == "fi":
            msa_icp_index_score_sharded(
                self.q,
                self.k,
                self.block_table,
                plan,
                rank=0,
                max_score=self.scores,
                initialize=False,
                tp=self.world,
                compact=self.page == 256,
            )
        else:
            ref["api"].fmha_index_score_sharded(
                self.q,
                self.k,
                self.block_table,
                plan,
                rank=0,
                max_score=self.scores,
                sm_scale=1.0,
                initialize=False,
                tp=self.world,
                compact=self.page == 256,
            )

    def _select(self, arm, ref):
        fn = (
            msa_icp_topk_select_prefill
            if arm == "fi"
            else ref["api"].sparse_topk_select_prefill
        )
        fn(
            self.scores,
            self.nvalid,
            self.carrier,
            force_begin_blocks=INIT,
            force_end_blocks=LOCAL,
            index_scale=self.index_scale,
            index_offset=0,
        )

    def _merge(self, arm, ref):
        received = self.carrier.view(self.world, self.hpr, CHUNK, 2 * TOPK + 1)
        fn = (
            msa_icp_merge_candidates
            if arm == "fi"
            else ref["vd"].minimax_m3_index_merge_candidates
        )
        fn(
            received,
            self.nvalid,
            self.output,
            self.errors,
            init_blocks=INIT,
            local_blocks=LOCAL,
            prefill=True,
        )

    def run(self, arm, ref) -> None:
        part = self.part
        if part == "plan":
            self._plan(arm, ref)
        elif part == "score":
            self._score(arm, ref, self.plans[arm])
        elif part == "select":
            self._select(arm, ref)
        elif part == "merge":
            self._merge(arm, ref)
        else:  # chain
            self._score(arm, ref, self._plan(arm, ref))
            self._select(arm, ref)
            self._merge(arm, ref)


def rows(args, calls, device):
    """Yield ``(name, factory)``; the factory builds one row's state per arm."""
    decode_parts = ["chain"] + (
        ["score_publish", "merge"] if args.fused else ["score", "publish", "merge"]
    ) * args.roles
    prefill_parts = ["chain"] + ["plan", "score", "select", "merge"] * args.roles
    if "decode" in args.rows:
        for shape in itertools.product(
            (2, 4), (128, 256), (1, 2, 4, 14), (1, 4), (8192, 163840, 172031)
        ):
            w, p, b, q, c = shape
            for protocol in args.protocols:
                inputs = {}
                for part in decode_parts:

                    def make(shape=shape, protocol=protocol, part=part, inputs=inputs):
                        if not inputs:
                            inputs.update(Decode.inputs(shape, device))
                        return [
                            Decode(
                                shape, protocol, part, args.fused, inputs, calls, device
                            )
                        ]

                    yield f"decode TP{w} P{p} B{b} q{q} ctx{c} {protocol} {part}", make
    if "prefill" in args.rows:
        for w, p, prefix in itertools.product(
            (2, 4), (128, 256), (0, 32768, 81920, 155648)
        ):
            for part in prefill_parts:
                yield (
                    f"prefill TP{w} P{p} prefix{prefix} {part}",
                    lambda a=(w, p, prefix, part): [Prefill(*a, device)],
                )
    if "mixed" in args.rows:
        for w, p, prefix in itertools.product((2, 4), (128, 256), (81920, 155648)):
            cache: dict = {}

            def make(w=w, p=p, prefix=prefix, inputs=cache):
                shape = (w, p, 14, 1, 163840)
                if not inputs:
                    inputs.update(Decode.inputs(shape, device))
                decode = Decode(
                    shape, "legacy", "chain", args.fused, inputs, calls, device
                )
                return [decode, Prefill(w, p, prefix, "chain", device)]

            yield f"mixed TP{w} P{p} d14 prefix{prefix} legacy chain", make


def timer_name() -> str:
    try:
        from cupti import cupti  # noqa: F401
    except ImportError:
        return "CUDA events"  # bench_gpu_time falls back to CUDA events without cupti-python
    return "CUPTI"


def session_us(fn: Callable[[], None], warmup: int, iters: int) -> float:
    times = bench_gpu_time(
        fn,
        dry_run_iters=warmup,
        repeat_iters=iters,
        enable_cupti=True,
        cold_l2_cache=True,
    )
    return float(statistics.median(times)) * 1e3


def graphed(
    fn: Callable[[], None], replays: int, cycle: bool = False
) -> Callable[[], None]:
    """Capture ``replays`` CUDA graphs of ``fn`` and replay them in order (one
    per timed call, so a call that takes receive slots keeps its own slot), or
    the one graph again and again with ``cycle``.  No host launch gap sits
    between the kernels of one call."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = []
    with torch.cuda.stream(stream):
        for _ in range(replays):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                fn()
            graphs.append(graph)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if cycle:
        return graphs[0].replay
    position = iter(range(replays))

    def replay() -> None:
        graphs[next(position)].replay()

    return replay


def device_label(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    arch = cake_jit.arch_for_capability(torch.cuda.get_device_capability(device))
    label = {
        "gpu": props.name,
        "arch": arch,
        "sms": props.multi_processor_count,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }
    uuid = str(getattr(props, "uuid", ""))
    if uuid:
        label["uuid"] = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    label["uuid"],
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            label["driver"] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument(
        "--rows",
        nargs="+",
        default=["decode", "prefill", "mixed"],
        choices=["decode", "prefill", "mixed"],
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="only rows whose name has every one of these words (e.g. TP2 P128)",
    )
    parser.add_argument(
        "--protocols", nargs="+", default=["tagged"], choices=["tagged", "legacy"]
    )
    parser.add_argument("--roles", action="store_true", help="add one row per kernel")
    parser.add_argument(
        "--fused", action="store_true", help="FI decode with the fused scorer-publisher"
    )
    parser.add_argument(
        "--arms", nargs="+", default=["msa", "fi"], choices=["msa", "fi"]
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="launch every kernel from the host (default: one CUDA graph replay per call)",
    )
    parser.add_argument(
        "--groups", type=int, default=3, help="alternating session groups (>= 3)"
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    ref = _reference()
    arms = [a for a in args.arms if a == "fi" or ref is not None]
    if "msa" in args.arms and ref is None:
        print(
            f"# {BASELINE} not importable (vllm.third_party.fmha_sm100, vLLM MiniMax-M3 ops): FI rows only"
        )
    label = device_label(device)
    timer = timer_name()
    label["timer"] = timer
    print("# " + json.dumps(label))
    # Calls per arm: per session bench_gpu_time's first call + 5 estimate calls + warmup + iters
    # (two sessions per group), plus the check call; headroom for a CUDA-event fallback.
    calls = 2 * args.groups * (6 + args.warmup + args.iters + 8) + 1
    results = []
    for name, make in rows(args, calls, device):
        if args.only and not all(word in name.split() for word in args.only):
            continue
        state = {arm: make() for arm in arms}
        fns = {}
        for arm in arms:
            for part in state[arm]:
                part.prepare(arm, ref)

            def fn(arm=arm, parts=state[arm]):
                for part in parts:
                    part.run(arm, ref)

            fn()  # eager check call: builds and loads every kernel
            torch.cuda.synchronize()
            if not args.eager:
                slotted = any(
                    getattr(part, "slots", None) is not None for part in state[arm]
                )
                fn = graphed(fn, calls - 1 if slotted else 1, cycle=not slotted)
            fns[arm] = fn
        torch.cuda.synchronize()
        sessions = {arm: [] for arm in arms}
        for g in range(args.groups):
            order = (
                ["msa", "fi", "fi", "msa"] if g % 2 == 0 else ["fi", "msa", "msa", "fi"]
            )
            for arm in order:
                if arm in fns:
                    sessions[arm].append(session_us(fns[arm], args.warmup, args.iters))
        torch.cuda.synchronize()
        errors = {
            arm: int(sum(int(p.errors.item()) for p in state[arm])) for arm in arms
        }
        if any(errors.values()):
            raise RuntimeError(f"{name}: protocol errors {errors}")
        row = {
            "row": name,
            **{f"{arm}_us": statistics.median(sessions[arm]) for arm in arms},
            "sessions": sessions,
        }
        if "msa" in arms and "fi" in arms:
            row["speedup"] = row["msa_us"] / row["fi_us"]
        results.append(row)
        print(json.dumps(row), flush=True)
        del state, fns
        torch.cuda.empty_cache()

    caption = (
        f"{label['gpu']} ({label['arch']}, {label['sms']} SMs), driver {label.get('driver', '?')}, "
        f"CUDA {label['cuda']}; {timer}, {'eager launches' if args.eager else 'CUDA graph replay per call'}, cold L2, "
        f"median of {2 * args.groups} sessions x {args.iters} iterations"
        + ("; FI decode: fused scorer-publisher" if args.fused else "")
    )
    print(f"\n{caption}\n")
    if "msa" in arms:
        print(f"| row | {BASELINE} (us) | FI (us) | speedup |\n|---|---:|---:|---:|")
        for r in results:
            fi = f"{r['fi_us']:.2f}" if "fi_us" in r else "-"
            ratio = f"{r['speedup']:.2f}x" if "speedup" in r else "-"
            print(f"| {r['row']} | {r['msa_us']:.2f} | {fi} | {ratio} |")
    else:
        print("| row | FI (us) |\n|---|---:|")
        for r in results:
            print(f"| {r['row']} | {r['fi_us']:.2f} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "device": label,
                    "baseline": BASELINE,
                    "caption": caption,
                    "args": vars(args),
                    "rows": results,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
