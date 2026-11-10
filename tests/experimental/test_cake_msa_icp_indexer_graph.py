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

GPU tests of the per-layer CUDA graphs of the ICP indexer
(``MSAIcpIndexerPlan.layer_graphs``): a layer replayed from a graph captured
in an earlier step writes the same words as the per-call path.

Every step keeps the caller's buffers (block tables, valid-page counts, window
plan storage and segment metadata, K pages, carriers, outputs) and changes
their contents: new histories (prefix lengths, page tables, K pages) and new
queries per layer, with the window plans built for the model limit so that
one graph key serves every step.  The first step of the key runs eagerly, the
second captures and replays, the later ones replay.

* Single GPU (the ``W`` ranks of a layout on one device): prefill steps of
  every layout, materialized and fused (caller pool).  A rank's exchange
  copies its own carrier rows from its graph's ``send`` and its peers' rows
  from the per-call path's ``send`` (the same words, checked), so every
  rank's graph holds its local launches, its exchange and its merge.
* Keys: a moved buffer gives a new key (eager, then captured), ``max_keys``
  evicts the least recently used one; argument checks.
* Distributed (``torchrun``, TP2 / TP4 x P128 / P256, materialized and
  fused): mixed steps (decode chain with legacy or tagged rows, prefill
  windows, one ``all_to_all_single`` per layer inside the graph, merge)
  compared bit for bit with the per-call path and the CPU oracle.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch

import flashinfer.msa_ops as msa_ops
from tests.experimental import test_cake_msa_icp_indexer as base
from tests.experimental.test_helpers import cake_msa_icp_inputs as ref

FP8 = torch.float8_e4m3fn
H = ref.NUM_HEADS
TOPK = ref.TOPK
LAYOUTS = base.LAYOUTS
LAYOUT_IDS = base.LAYOUT_IDS
POISON_I32 = base.POISON_I32
MODEL_LEN = 16384
MAX_TOKENS = 4 * 8192

pytestmark = base.pytestmark
DEV = base.DEV


def _plan(lay: ref.IcpLayout, rank: int, **kw: Any) -> Any:
    args = dict(
        world_size=lay.tp,
        rank=rank,
        num_index_heads=H,
        page_size=lay.page_size,
        init_blocks=0,
        local_blocks=1,
        max_decode_tokens=16,
        max_decode_query_len=4,
        max_num_batched_tokens=MAX_TOKENS,
        max_num_seqs=16,
        max_model_len=MODEL_LEN,
        num_sparse_layers=3,
        device=DEV,
    )
    args.update(kw)
    return msa_ops.prepare_msa_icp_indexer(**args)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _prefill_steps(
    page: int,
    prefixes: Sequence[Sequence[int]],
    layers: int,
    *,
    q_lens: Sequence[int],
    padding_tokens: int,
    chunk: int,
    seed: int,
) -> list[list[ref.PrefillCase]]:
    """Per step one history (``prefixes[s]``), per layer a new query and K
    page contents (same page tables)."""
    out = []
    for s, pre in enumerate(prefixes):
        case = ref.make_prefill_case(
            f"graph_s{s}_p{page}",
            page_size=page,
            q_lens=q_lens,
            prefixes=pre,
            padding_tokens=padding_tokens,
            chunk=chunk,
            seed=seed + 101 * s,
        )
        out.append(
            [
                dataclasses.replace(
                    case,
                    label=f"{case.label}_l{layer}",
                    q=torch.roll(case.q, shifts=5 * layer, dims=0),
                    cache=torch.roll(case.cache, shifts=layer, dims=0),
                )
                for layer in range(layers)
            ]
        )
    return out


class StaticPrefill:
    """The persistent prefill inputs of one rank's steps: block table,
    valid-page counts, per-window segment metadata (the window plans are
    rebuilt in the facade's plan storage each step, for the model limit)."""

    def __init__(self, plan: Any, case: ref.PrefillCase) -> None:
        self.plan = plan
        self.windows = case.windows()
        self.batch = case.batch
        self.block_table = torch.full(
            (case.batch, _cdiv(MODEL_LEN, case.page_size)),
            ref.PAD_PAGE,
            dtype=torch.int32,
            device=DEV,
        )
        self.nvalid = torch.zeros(case.total_q, dtype=torch.int32, device=DEV)
        b = case.batch
        self.meta = [
            tuple(
                torch.zeros(n, dtype=torch.int32, device=DEV)
                for n in (b, b, b + 1, b + 1, b)
            )
            for _ in self.windows
        ]
        self.wplans: list[Any] = []

    def load(self, case: ref.PrefillCase, page_offset: int = 0) -> None:
        """``case``'s history, its pages at ``page_offset`` of the K buffer."""
        windows = case.windows()
        assert [(w.start, w.end) for w in windows] == [
            (w.start, w.end) for w in self.windows
        ]
        bt = torch.full(tuple(self.block_table.shape), ref.PAD_PAGE, dtype=torch.int32)
        rows = torch.tensor(case.block_table, dtype=torch.int32).reshape(self.batch, -1)
        rows = torch.where(rows == ref.PAD_PAGE, rows, rows + page_offset)
        bt[:, : rows.shape[1]] = rows
        self.block_table.copy_(bt.to(DEV))
        self.nvalid.copy_(base._i32(case.nvalid()))
        self.wplans = []
        for wi, w in enumerate(windows):
            for dst, src in zip(self.meta[wi], base._window_meta(w), strict=True):
                dst.copy_(src)
            self.wplans.append(
                self.plan.prefill_window_plan(
                    wi,
                    *self.meta[wi],
                    total_q=w.total_q,
                    max_query_len=max(max(w.q_lens), 1),
                    max_seq_len=MODEL_LEN,
                )
            )

    def windows_arg(self) -> list[tuple]:
        return [
            (w.start, w.end, self.block_table, self.wplans[wi])
            for wi, w in enumerate(self.windows)
        ]

    def per_call_local(
        self, q: torch.Tensor, k: torch.Tensor, scratch: Any, send: torch.Tensor, **kw
    ) -> None:
        for wi, w in enumerate(self.windows):
            self.plan.prefill_local(
                q[w.start : w.end],
                k,
                self.block_table,
                self.wplans[wi],
                scratch,
                self.nvalid[w.start : w.end],
                send[:, w.start : w.end],
                **kw,
            )


def _prefill_buffers(plan: Any, total_q: int, hpr: int, world: int, fused: bool):
    send = torch.full(
        (H, total_q, ref.ROW_WORDS), POISON_I32, dtype=torch.int32, device=DEV
    )
    recv = torch.full_like(send, POISON_I32)
    return types.SimpleNamespace(
        send=send,
        recv=recv,
        recv4=recv.view(world, hpr, total_q, ref.ROW_WORDS),
        out=torch.full((total_q, hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV),
        scratch=None if fused else base._poisoned_f32(plan.prefill_scratch_shape()),
        kw=(
            {
                "pool": torch.empty(
                    plan.prefill_pool_shape(), dtype=torch.uint8, device=DEV
                )
            }
            if fused
            else {}
        ),
    )


# =========================================================================== prefill, one device


class OneDeviceRig:
    """The ``W`` ranks of a layout on one device: per rank a facade, its
    persistent prefill inputs, per-call buffers ``a`` and graph buffers ``b``,
    the K shard buffers of every layer and one layer-graph cache."""

    def __init__(
        self,
        lay: ref.IcpLayout,
        steps: list[list[ref.PrefillCase]],
        *,
        fused: bool,
        capture_after: int = 1,
        max_keys: int = 4,
    ) -> None:
        self.lay, self.steps, self.fused = lay, steps, fused
        c0 = steps[0][0]
        self.layers = len(steps[0])
        self.hpr = H // lay.tp
        self.plans = [
            _plan(
                lay,
                r,
                init_blocks=c0.init_blocks,
                local_blocks=c0.local_blocks,
                max_num_seqs=c0.batch,
            )
            for r in range(lay.tp)
        ]
        if not fused:
            rows = self.plans[0].prefill_scratch_shape()[0]
            assert c0.chunk <= rows, "the windows must fit the model-limit scratch"
        self.static = [StaticPrefill(p, c0) for p in self.plans]
        pages = max(c.cache.shape[0] for step in steps for c in step)
        self.k = [
            [
                torch.zeros((pages, lay.rows, 128), dtype=torch.uint8, device=DEV)
                for _ in range(lay.tp)
            ]
            for _ in range(self.layers)
        ]
        t = c0.total_q
        self.a = [_prefill_buffers(p, t, self.hpr, lay.tp, fused) for p in self.plans]
        self.b = [_prefill_buffers(p, t, self.hpr, lay.tp, fused) for p in self.plans]
        self.graphs = [
            p.layer_graphs(capture_after=capture_after, max_keys=max_keys)
            for p in self.plans
        ]

    def load(self, s: int) -> None:
        for layer, case in enumerate(self.steps[s]):
            for r in range(self.lay.tp):
                n = case.cache.shape[0]
                self.k[layer][r][:n].copy_(self.lay.shard(case.cache, r).to(DEV))
        for st in self.static:
            st.load(self.steps[s][0])

    def kview(self, layer: int, r: int) -> torch.Tensor:
        return self.k[layer][r].view(FP8)

    def per_call_layer(self, q: torch.Tensor, layer: int) -> None:
        w, hpr = self.lay.tp, self.hpr
        for r in range(w):
            a = self.a[r]
            self.static[r].per_call_local(
                q, self.kview(layer, r), a.scratch, a.send, **a.kw
            )
        for o in range(w):
            for r in range(w):
                self.a[o].recv4[r].copy_(self.a[r].send[o * hpr : (o + 1) * hpr])
        for o in range(w):
            self.plans[o].prefill_merge(
                self.a[o].recv4, self.static[o].nvalid, self.a[o].out
            )

    def exchange(self, r: int):
        """Rank ``r``'s exchange: its own rows from its graph's ``send``, its
        peers' from the per-call ``send`` (equal words, compared)."""
        hpr = self.hpr

        def run() -> None:
            for o in range(self.lay.tp):
                src = self.b[r].send if o == r else self.a[o].send
                self.b[r].recv4[o].copy_(src[r * hpr : (r + 1) * hpr])

        return run

    def begin(self, exchange_key: Any = "emulated") -> list[Any]:
        out = []
        for r in range(self.lay.tp):
            b = self.b[r]
            step = self.plans[r].prepare_prefill_step(
                self.static[r].windows_arg(),
                b.scratch,
                self.static[r].nvalid,
                b.send,
                b.recv4,
                b.out,
                **b.kw,
            )
            out.append(
                self.graphs[r].begin_step(
                    prefill=step, exchange=self.exchange(r), exchange_key=exchange_key
                )
            )
        return out

    def run_step(self, s: int, *, oracle: bool, what: str) -> None:
        self.load(s)
        gsteps = self.begin()
        for layer in range(self.layers):
            case = self.steps[s][layer]
            q = base._fp8(case.q)  # a new query tensor every layer
            self.per_call_layer(q, layer)
            for r in range(self.lay.tp):
                gsteps[r].run(layer, q, self.kview(layer, r))
            torch.cuda.synchronize()
            exp = (
                base.prefill_expect(case, self.lay, fast=True, with_global=False)
                if oracle and layer == 0
                else None
            )
            for r in range(self.lay.tp):
                tag = f"{what} step {s} layer {layer} rank {r}"
                ga, gb = self.a[r], self.b[r]
                base.assert_words(gb.send.cpu(), ga.send.cpu(), tag + " carrier")
                base.assert_words(gb.out.cpu(), ga.out.cpu(), tag + " ids")
                if exp is not None:
                    base.assert_words(
                        ga.send.cpu(), exp.carriers[r], tag + " oracle carrier"
                    )
                    base.assert_words(ga.out.cpu(), exp.outputs[r], tag + " oracle ids")


def _one_device_steps(page: int, layers: int = 3) -> list[list[ref.PrefillCase]]:
    return _prefill_steps(
        page,
        [[0, 4000, 700], [512, 3000, 1500], [100, 9000, 20], [2000, 12000, 7]],
        layers,
        q_lens=[300, 1, 45],
        padding_tokens=3,
        chunk=128,
        seed=8170,
    )


@pytest.mark.parametrize("fused", (False, True), ids=("materialized", "fused"))
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_layer_graph_replay_equals_the_per_call_path(
    lay: ref.IcpLayout, fused: bool
) -> None:
    """Four steps of one key (new histories, K pages and queries): the eager
    step, the capturing step and two replays write the per-call path's
    carriers and merged ids, which equal the oracle."""
    steps = _one_device_steps(lay.page_size)
    assert len(steps[0][0].windows()) == 3
    base._warm()
    rig = OneDeviceRig(lay, steps, fused=fused)
    for s in range(len(steps)):
        rig.run_step(s, oracle=True, what=f"{lay.name} fused={fused}")
    n = rig.layers
    for g in rig.graphs:
        assert len(g._graphs) == 1, "one key for every step"
        assert g.stats["eager_layers"] == n
        assert g.stats["captured_layers"] == n
        assert g.stats["replayed_layers"] == n * (len(steps) - 2)
        assert g.stats["evicted_keys"] == 0


def test_layer_graphs_new_key_capture_and_eviction() -> None:
    """A moved buffer gives a new key: its first step runs eagerly, its next
    one captures; with ``max_keys=1`` the old key is evicted.  With
    ``capture_after=0`` the first step captures."""
    lay = ref.IcpLayout(128, 2)
    steps = _one_device_steps(128, layers=2)
    base._warm()
    rig = OneDeviceRig(lay, steps, fused=False, capture_after=1, max_keys=1)
    rig.run_step(0, oracle=False, what="k0")
    rig.run_step(1, oracle=False, what="k0")
    for b in rig.b:
        b.out = torch.full_like(b.out, POISON_I32)  # moved: a new key
    rig.run_step(2, oracle=False, what="k1")
    rig.run_step(3, oracle=False, what="k1")
    for g in rig.graphs:
        assert g.stats["eager_layers"] == 4
        assert g.stats["captured_layers"] == 4
        assert g.stats["replayed_layers"] == 0
        assert g.stats["evicted_keys"] == 1
        assert len(g._graphs) == 1
    rig = OneDeviceRig(lay, steps, fused=True, capture_after=0)
    rig.run_step(0, oracle=False, what="first capture")
    rig.run_step(1, oracle=False, what="first capture")
    for g in rig.graphs:
        assert g.stats["eager_layers"] == 0
        assert g.stats["captured_layers"] == 2
        assert g.stats["replayed_layers"] == 2


def test_layer_graphs_argument_checks() -> None:
    lay = ref.IcpLayout(128, 2)
    steps = _one_device_steps(128, layers=1)
    base._warm()
    rig = OneDeviceRig(lay, steps, fused=False)
    rig.load(0)
    with pytest.raises(ValueError, match="capture_after"):
        rig.plans[0].layer_graphs(capture_after=-1)
    with pytest.raises(ValueError, match="max_keys"):
        rig.plans[0].layer_graphs(max_keys=0)
    g = rig.graphs[0]
    with pytest.raises(ValueError, match="needs a prefill or a decode step"):
        g.begin_step()
    b = rig.b[0]
    st0 = rig.plans[0].prepare_prefill_step(
        rig.static[0].windows_arg(),
        b.scratch,
        rig.static[0].nvalid,
        b.send,
        b.recv4,
        b.out,
    )
    with pytest.raises(ValueError, match="exchange goes with a prefill step"):
        g.begin_step(prefill=st0)
    with pytest.raises(ValueError, match="decode_tokens needs a decode step"):
        g.begin_step(prefill=st0, exchange=lambda: None, decode_tokens=2)
    with pytest.raises(ValueError, match="from this indexer"):
        rig.graphs[1].begin_step(prefill=st0, exchange=lambda: None)


# =========================================================================== distributed


def _dist_main(argv: Sequence[str]) -> int:
    """Body of one rank under ``torchrun``: ``--page P --fused 0|1 --out DIR``."""
    import argparse

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--fused", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    # The shared helpers allocate on their module's DEV.
    base.DEV = dev
    global DEV
    DEV = dev
    dist.init_process_group("nccl", device_id=dev)
    lay = ref.IcpLayout(args.page, world)
    hpr = H // world
    fused = bool(args.fused)
    report: dict[str, Any] = {
        "rank": rank,
        "world": world,
        "page": args.page,
        "fused": fused,
        "checks": 0,
    }
    try:
        cap, steps_n, layers, dql, max_dql, batch = 16, 4, 5, 2, 4, 6
        layout = msa_ops.msa_icp_comm_layout(world, hpr, cap)
        group = dist.group.WORLD.group_name
        if hasattr(symm_mem, "enable_symm_mem_for_group"):
            symm_mem.enable_symm_mem_for_group(group)
        storage = symm_mem.empty(layout["storage_words"], dtype=torch.int32, device=dev)
        storage.zero_()
        handle = symm_mem.rendezvous(storage, group)
        bases = torch.tensor(
            [int(p) for p in handle.buffer_ptrs], dtype=torch.int64, device=dev
        )
        lw = layout["tagged_offset_bytes"] // 4
        comm = types.SimpleNamespace(
            recv=storage[:lw].view(layout["legacy_shape"]),
            tagged_recv=storage[lw:].view(torch.int64).view(layout["tagged_shape"]),
            recv_bases=bases,
            tagged_recv_bases=bases + layout["tagged_offset_bytes"],
            publish_generations=torch.zeros(
                layout["generations_shape"], dtype=torch.int32, device=dev
            ),
            expected_generations=torch.zeros(
                layout["expected_shape"], dtype=torch.int32, device=dev
            ),
            errors=torch.zeros(1, dtype=torch.int32, device=dev),
        )
        torch.cuda.synchronize()
        dist.barrier()

        pcases = _prefill_steps(
            args.page,
            [[0, 5000, 3, 260], [1024, 700, 9000, 33], [77, 12000, 511, 4000],
             [3000, 1, 128, 8000]],
            layers, q_lens=[700, 1, 0, 90], padding_tokens=6, chunk=256, seed=777,
        )  # fmt: skip
        dcases = base._decode_steps(
            lay, steps_n, layers, batch=batch, dql=dql, seed=4711, all_padding_step=3
        )
        c0 = pcases[0][0]
        t, nd = c0.total_q, dcases[0][0].tokens
        plan = _plan(
            lay, rank, init_blocks=c0.init_blocks, local_blocks=c0.local_blocks,
            max_decode_tokens=cap, max_decode_query_len=max_dql,
            max_num_seqs=max(c0.batch, batch), num_sparse_layers=layers,
        )  # fmt: skip
        plan.warmup()
        plan.check_comm(comm)
        if not fused:
            assert c0.chunk <= plan.prefill_scratch_shape()[0]
        sp = StaticPrefill(plan, c0)
        # one K buffer per layer: the prefill pages, then the decode pages
        ppages = [pcases[s][0].cache.shape[0] for s in range(steps_n)]
        dpages = [dcases[s][0].cache.shape[0] for s in range(steps_n)]
        kbuf = [
            torch.zeros(
                (
                    max(p + d for p, d in zip(ppages, dpages, strict=True)),
                    lay.rows,
                    128,
                ),
                dtype=torch.uint8,
                device=dev,
            )
            for _ in range(layers)
        ]
        dwidth = max(len(dcases[s][0].block_table[0]) for s in range(steps_n))
        dbt = torch.full((batch, dwidth), ref.PAD_PAGE, dtype=torch.int32, device=dev)
        dseq = torch.zeros(batch, dtype=torch.int32, device=dev)
        dnv = torch.zeros(nd, dtype=torch.int32, device=dev)

        def load(s: int) -> None:
            for layer in range(layers):
                pc, dc = pcases[s][layer], dcases[s][layer]
                kbuf[layer][: ppages[s]].copy_(lay.shard(pc.cache, rank).to(dev))
                kbuf[layer][ppages[s] : ppages[s] + dpages[s]].copy_(
                    lay.shard(dc.cache, rank).to(dev)
                )
            sp.load(pcases[s][0])
            dc = dcases[s][0]
            rows = torch.tensor(dc.block_table, dtype=torch.int32).reshape(batch, -1)
            rows = torch.where(rows == ref.PAD_PAGE, rows, rows + ppages[s])
            bt = torch.full((batch, dwidth), ref.PAD_PAGE, dtype=torch.int32)
            bt[:, : rows.shape[1]] = rows
            dbt.copy_(bt.to(dev))
            dseq.copy_(base._i32(dc.seq_lens))
            dnv.copy_(base._i32(dc.nvalid))

        # oracles, once per (step, layer): the slow one for the first, the
        # suite's validated fast reference after it
        pexp = [
            [
                base.prefill_expect(pcases[s][layer], lay, fast=s + layer > 0, with_global=False)
                for layer in range(layers)
            ]
            for s in range(steps_n)
        ]  # fmt: skip
        dexp = [
            [base.decode_expect(dcases[s][layer], lay)[2][rank] for layer in range(layers)]
            for s in range(steps_n)
        ]  # fmt: skip

        for tagged in (False, True):
            arms = {}
            for arm in ("call", "graph"):
                bufs = _prefill_buffers(plan, t, hpr, world, fused)
                bufs.scores = torch.empty(
                    plan.decode_scores_shape(), dtype=torch.float32, device=dev
                )
                bufs.dout = torch.full(
                    (nd, hpr, TOPK), POISON_I32, dtype=torch.int32, device=dev
                )
                arms[arm] = bufs
            graphs = plan.layer_graphs()
            g = arms["graph"]
            exchanges = [0]

            def exchange(g=g) -> None:
                exchanges[0] += 1
                dist.all_to_all_single(g.recv, g.send)

            for s in range(steps_n):
                load(s)
                qs = [
                    torch.cat([dcases[s][layer].q, pcases[s][layer].q])
                    .to(dev)
                    .view(FP8)
                    for layer in range(layers)
                ]
                hist: dict[str, list[tuple]] = {"call": [], "graph": []}
                # per-call path, every layer
                a = arms["call"]
                for layer in range(layers):
                    q, k = qs[layer], kbuf[layer].view(FP8)
                    plan.decode(
                        q[:nd], k, dbt, dseq, dnv, a.scores, a.dout, comm=comm,
                        slot=ref.slot_set(layer, layers), tagged=tagged,
                        decode_query_len=dql,
                    )  # fmt: skip
                    sp.per_call_local(q[nd:], k, a.scratch, a.send, **a.kw)
                    dist.all_to_all_single(a.recv, a.send)
                    plan.prefill_merge(a.recv4, sp.nvalid, a.out)
                    hist["call"].append((a.send.clone(), a.out.clone(), a.dout.clone()))
                # graph path, every layer
                dstep = plan.prepare_decode_step(
                    dbt, dseq, dnv, g.scores, g.dout, comm=comm, tagged=tagged,
                    decode_query_len=dql,
                )  # fmt: skip
                pstep = plan.prepare_prefill_step(
                    sp.windows_arg(),
                    g.scratch,
                    sp.nvalid,
                    g.send,
                    g.recv4,
                    g.out,
                    **g.kw,
                )
                gstep = graphs.begin_step(
                    prefill=pstep, decode=dstep, exchange=exchange,
                    exchange_key=("world", tagged), decode_tokens=nd,
                )  # fmt: skip
                before = exchanges[0]
                for layer in range(layers):
                    gstep.run(
                        layer, qs[layer], kbuf[layer].view(FP8),
                        slot=ref.slot_set(layer, layers),
                    )  # fmt: skip
                    hist["graph"].append(
                        (g.send.clone(), g.out.clone(), g.dout.clone())
                    )
                # the exchange runs once per layer: called while eager or
                # capturing, inside the graph while replaying
                assert exchanges[0] - before == (layers if s < 2 else 0)
                torch.cuda.synchronize()
                for layer in range(layers):
                    (cs, co, cd), (gs, go, gd) = (
                        hist["call"][layer],
                        hist["graph"][layer],
                    )
                    what = f"rank {rank} tagged={tagged} step {s} layer {layer}"
                    base.assert_words(
                        cs.cpu(),
                        pexp[s][layer].carriers[rank],
                        what + " carrier oracle",
                    )
                    base.assert_words(
                        co.cpu(), pexp[s][layer].outputs[rank], what + " ids oracle"
                    )
                    base.assert_words(cd.cpu(), dexp[s][layer], what + " decode oracle")
                    base.assert_words(gs.cpu(), cs.cpu(), what + " graph carrier")
                    base.assert_words(go.cpu(), co.cpu(), what + " graph ids")
                    base.assert_words(gd.cpu(), cd.cpu(), what + " graph decode")
                    report["checks"] += 6
            st = graphs.stats
            assert st["eager_layers"] == layers and st["captured_layers"] == layers
            assert st["replayed_layers"] == layers * (steps_n - 2)
            assert len(graphs._graphs) == 1
            report[f"stats_tagged{int(tagged)}"] = dict(st)
            graphs.clear()
            dist.barrier()
        assert int(comm.errors.item()) == 0
        assert int(plan.prefill_errors.item()) == 0
        report["ok"] = True
    except BaseException as e:  # noqa: BLE001 - reported to the launching test
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"[:2000]
        raise
    finally:
        Path(args.out, f"rank{rank}.json").write_text(json.dumps(report))
        dist.destroy_process_group()
    return 0


@pytest.mark.parametrize("fused", (False, True), ids=("materialized", "fused"))
@pytest.mark.parametrize("page", (128, 256))
@pytest.mark.parametrize("world", (2, 4))
def test_distributed_layer_graphs_mixed_steps(
    world: int, page: int, fused: bool
) -> None:
    """``torchrun`` with ``world`` ranks: four mixed steps of one key (decode
    chain, prefill windows, one ``all_to_all_single`` per layer, merge; new
    histories every step) through the layer graphs equal the per-call path
    bit for bit and the oracle, legacy and tagged decode rows."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _torchrun(world, ["--page", str(page), "--fused", str(int(fused))])


def _torchrun(world: int, args: list[str]) -> None:
    import subprocess
    import tempfile

    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as out:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")])
        cmd = [
            sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world}",
            "--standalone", str(Path(__file__).resolve()), *args, "--out", out,
        ]  # fmt: skip
        proc = subprocess.run(
            cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=3600
        )
        reports = [
            json.loads(p.read_text()) for p in sorted(Path(out).glob("rank*.json"))
        ]
        tail = (proc.stdout[-4000:] + "\n" + proc.stderr[-6000:]).strip()
        errors = [(r["rank"], r.get("error")) for r in reports if not r.get("ok")]
        assert proc.returncode == 0, (
            f"torchrun rc={proc.returncode} rank errors {errors}\n{tail}"
        )
        assert len(reports) == world and all(r.get("ok") for r in reports), (
            f"{reports}\n{tail}"
        )
        assert all(r["checks"] > 0 for r in reports)


if __name__ == "__main__":
    raise SystemExit(_dist_main(sys.argv[1:]))
