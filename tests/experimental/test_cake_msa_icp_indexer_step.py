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

GPU tests of the step-prepared ICP indexer (``MSAIcpIndexerPlan.prepare_prefill_step``
and ``prepare_decode_step``): every layer of a step must make exactly the
launches of the facade's per-call path.

* Prefill, single GPU (the ``W`` ranks of a layout on one device, the
  exchange as a device copy): every prefill scenario of
  ``test_cake_msa_icp_indexer`` over several layers with a new query and new
  K pages per layer; carriers, score scratch and merged ids of the step path
  are compared bit for bit with the per-call path, with and without
  ``debug``; a captured multi-layer step replayed with new inputs.
* Decode, single GPU: the launches of the decode step (program, FFI entry,
  every argument) equal the per-call ``decode`` launches for every decode
  scenario, legacy and tagged rows, the slot rotation, the caller pool and
  the facade's own pool (a decode merge spins on its peers, so the launches
  are compared, not run, here; the distributed test runs them).
* ``debug``: a step buffer moved after the first layer is caught; a layer
  with other query / K metadata is refused; the first layer runs every
  facade check.
* Distributed (``torchrun``, TP2 / TP4 x P128 / P256): decode steps (legacy /
  tagged, eager then CUDA Graph replays) and mixed steps (decode chain, then
  windows -> ``all_to_all_single`` -> merge per layer) through the step path
  and the per-call path, both compared with each other bit for bit and with
  the CPU oracle.
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
from flashinfer.experimental.cake_msa_icp_indexer import cake_jit
from flashinfer.experimental.cake_msa_icp_indexer import cake_indexer
from tests.experimental import test_cake_msa_icp_indexer as base
from tests.experimental.test_helpers import cake_msa_icp_inputs as ref

FP8 = torch.float8_e4m3fn
H = ref.NUM_HEADS
TOPK = ref.TOPK
LAYOUTS = base.LAYOUTS
LAYOUT_IDS = base.LAYOUT_IDS
POISON_I32 = base.POISON_I32
LAYERS = 4

pytestmark = base.pytestmark
DEV = base.DEV


def _rolled(t: torch.Tensor, shifts: int) -> torch.Tensor:
    """``t`` with its rows rolled along dim 0, in a new tensor of the same
    shape, strides and dtype (other layer contents, same metadata)."""
    out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    out.view(torch.uint8).copy_(torch.roll(t.view(torch.uint8), shifts, dims=0))
    return out


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
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=16384,
        num_sparse_layers=3,
        device=DEV,
    )
    args.update(kw)
    return msa_ops.prepare_msa_icp_indexer(**args)


# =========================================================================== prefill, one device


class PrefillRig:
    """Per-rank facades, window plans and two sets of step buffers (``a``:
    per-call path, ``b``: step path) of one prefill case on one device."""

    def __init__(self, case: ref.PrefillCase, lay: ref.IcpLayout) -> None:
        self.case, self.lay = case, lay
        self.windows = case.windows()
        self.hpr = H // lay.tp
        self.plans = [
            _plan(
                lay,
                r,
                init_blocks=case.init_blocks,
                local_blocks=case.local_blocks,
                max_num_batched_tokens=8192 * len(self.windows),
                max_num_seqs=case.batch,
                max_model_len=max(max(case.seq_lens), 8192),
            )
            for r in range(lay.tp)
        ]
        inp = base.prefill_case_inputs(case, lay)
        self.q0, self.shards0 = inp["q"], inp["shards"]
        self.block_table, self.nvalid = inp["block_table"], inp["nvalid"]
        self.wplans = [
            [
                self.plans[r].prefill_window_plan(
                    wi,
                    *base._window_meta(w),
                    total_q=w.total_q,
                    max_query_len=max(max(w.q_lens), 1),
                    max_seq_len=max(max(w.kv_lens), 1),
                )
                for wi, w in enumerate(self.windows)
            ]
            for r in range(lay.tp)
        ]
        rows, _, pitch = self.plans[0].prefill_scratch_shape(max(case.seq_lens))
        t = case.total_q
        self.bufs = {}
        for side in ("a", "b"):
            self.bufs[side] = types.SimpleNamespace(
                scratch=[base._poisoned_f32((rows, H, pitch)) for _ in range(lay.tp)],
                send=[
                    torch.full(
                        (H, t, ref.ROW_WORDS), POISON_I32, dtype=torch.int32, device=DEV
                    )
                    for _ in range(lay.tp)
                ],
                recv=[
                    torch.full(
                        (lay.tp, self.hpr, t, ref.ROW_WORDS),
                        POISON_I32,
                        dtype=torch.int32,
                        device=DEV,
                    )
                    for _ in range(lay.tp)
                ],
                out=[
                    torch.full(
                        (t, self.hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV
                    )
                    for _ in range(lay.tp)
                ],
            )

    def layer_inputs(self, layer: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """A new query and new K page contents per layer (same geometry)."""
        if layer == 0:
            return self.q0, self.shards0
        q = _rolled(self.q0, 7 * layer)
        ks = [_rolled(k, layer) for k in self.shards0]
        return q, ks

    def windows_arg(self, r: int) -> list[tuple]:
        return [
            (w.start, w.end, self.block_table, self.wplans[r][wi])
            for wi, w in enumerate(self.windows)
        ]

    def steps(self, *, debug: bool) -> list[Any]:
        b = self.bufs["b"]
        return [
            self.plans[r].prepare_prefill_step(
                self.windows_arg(r),
                b.scratch[r],
                self.nvalid,
                b.send[r],
                b.recv[r],
                b.out[r],
                debug=debug,
            )
            for r in range(self.lay.tp)
        ]

    def exchange(self, side: str) -> None:
        s = self.bufs[side]
        for o in range(self.lay.tp):
            for r in range(self.lay.tp):
                s.recv[o][r].copy_(s.send[r][o * self.hpr : (o + 1) * self.hpr])

    def per_call_layer(self, q: torch.Tensor, ks: list[torch.Tensor]) -> None:
        a = self.bufs["a"]
        for r in range(self.lay.tp):
            for wi, w in enumerate(self.windows):
                self.plans[r].prefill_local(
                    q[w.start : w.end],
                    ks[r],
                    self.block_table,
                    self.wplans[r][wi],
                    a.scratch[r],
                    self.nvalid[w.start : w.end],
                    a.send[r][:, w.start : w.end],
                )
        self.exchange("a")
        for o in range(self.lay.tp):
            self.plans[o].prefill_merge(a.recv[o], self.nvalid, a.out[o])

    def step_layer(
        self, steps: list[Any], q: torch.Tensor, ks: list[torch.Tensor]
    ) -> None:
        for r in range(self.lay.tp):
            steps[r].local(q, ks[r])
        self.exchange("b")
        for o in range(self.lay.tp):
            steps[o].merge()

    def snapshot(self, side: str) -> list[torch.Tensor]:
        s = self.bufs[side]
        return [t.clone() for t in (*s.send, *s.out, *s.scratch)]


def _compare(got: list[torch.Tensor], want: list[torch.Tensor], what: str) -> None:
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if g.dtype == torch.float32:
            g, w = g.view(torch.int32), w.view(torch.int32)
        base.assert_words(g.cpu(), w.cpu(), f"{what} buffer {i}")


@pytest.mark.parametrize("debug", (False, True), ids=("fast", "debug"))
@pytest.mark.parametrize("scenario", base.PREFILL_SCENARIOS)
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_prefill_step_is_byte_identical_to_the_per_call_path(
    lay: ref.IcpLayout, scenario: str, debug: bool
) -> None:
    """Every layer of a prepared prefill step writes the same carriers, score
    scratch words and merged ids as ``prefill_local`` per window +
    ``prefill_merge`` with the same inputs (new query and K per layer)."""
    case, _ = base._make_prefill(lay.page_size, scenario)
    rig = PrefillRig(case, lay)
    base._warm()
    steps = rig.steps(debug=debug)
    for layer in range(LAYERS):
        q, ks = rig.layer_inputs(layer)
        rig.per_call_layer(q, ks)
        rig.step_layer(steps, q, ks)
        torch.cuda.synchronize()
        _compare(
            rig.snapshot("b"),
            rig.snapshot("a"),
            f"{case.label} {lay.name} layer {layer}",
        )


def _multi_window_case(page: int) -> ref.PrefillCase:
    return ref.make_prefill_case(
        f"step_facade_p{page}",
        page_size=page,
        q_lens=[300, 1, 45],
        prefixes=[0, 4000, 700],
        padding_tokens=3,
        chunk=128,
        seed=31,
    )


@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_prefill_step_graph_replays(lay: ref.IcpLayout) -> None:
    """A captured multi-layer prefill step (3 windows per layer; the step
    records its launches during capture) replayed for 3 steps with new
    queries and K pages in static buffers equals the eager per-call path."""
    case = _multi_window_case(lay.page_size)
    rig = PrefillRig(case, lay)
    assert len(rig.windows) == 3
    base._warm()
    layers, steps_n = 3, 4
    staged = []
    for s in range(steps_n):
        per_layer = [
            rig.layer_inputs(1 + s * layers + layer) for layer in range(layers)
        ]
        staged.append(per_layer)
    q_static = [torch.empty_like(rig.q0) for _ in range(layers)]
    k_static = [[torch.empty_like(k) for k in rig.shards0] for _ in range(layers)]
    b = rig.bufs["b"]
    outs = [[torch.empty_like(t) for t in (*b.send, *b.out)] for _ in range(layers)]

    def load(s: int) -> None:
        for layer, (q, ks) in enumerate(staged[s]):
            q_static[layer].copy_(q)
            for r, k in enumerate(ks):
                k_static[layer][r].copy_(k)

    def run(steps: list[Any]) -> None:
        for layer in range(layers):
            rig.step_layer(steps, q_static[layer], k_static[layer])
            for dst, src in zip(outs[layer], (*b.send, *b.out), strict=True):
                dst.copy_(src)

    load(0)
    run(rig.steps(debug=False))  # eager warm-up step
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        run(rig.steps(debug=False))
    for s in range(1, steps_n):
        load(s)
        graph.replay()
        torch.cuda.synchronize()
        got = [[t.clone() for t in per] for per in outs]
        for layer, (q, ks) in enumerate(staged[s]):
            rig.per_call_layer(q, ks)
            a = rig.bufs["a"]
            torch.cuda.synchronize()
            _compare(
                got[layer], [*a.send, *a.out], f"{lay.name} replay {s} layer {layer}"
            )


def test_debug_step_catches_a_buffer_moved_after_the_first_layer() -> None:
    lay = ref.IcpLayout(128, 2)
    rig = PrefillRig(_multi_window_case(128), lay)
    base._warm()
    nv = rig.nvalid.clone()
    other = rig.nvalid.clone()
    b = rig.bufs["b"]
    step = rig.plans[0].prepare_prefill_step(
        rig.windows_arg(0), b.scratch[0], nv, b.send[0], b.recv[0], b.out[0], debug=True
    )
    q, ks = rig.layer_inputs(0)
    step.local(q, ks[0])
    step.local(q, ks[0])  # same buffers: no difference
    nv.set_(
        other.untyped_storage(), other.storage_offset(), other.shape, other.stride()
    )
    with pytest.raises(RuntimeError, match="differs from the per-call path"):
        step.local(q, ks[0])
    torch.cuda.synchronize()


@pytest.mark.parametrize("debug", (False, True), ids=("fast", "debug"))
def test_step_refuses_query_or_k_with_other_metadata(debug: bool) -> None:
    lay = ref.IcpLayout(128, 2)
    rig = PrefillRig(_multi_window_case(128), lay)
    base._warm()
    step = rig.steps(debug=debug)[0]
    q, ks = rig.layer_inputs(0)
    step.local(q, ks[0])
    k_more_pages = torch.cat([ks[0], ks[0][:1]]).contiguous()
    with pytest.raises(ValueError, match="the step's first layer"):
        step.local(q, k_more_pages)
    with pytest.raises(ValueError, match="the step's first layer"):
        step.local(q[:-1], ks[0])
    with pytest.raises(ValueError, match="the step's first layer"):
        step.local(q.view(torch.uint8), ks[0])
    torch.cuda.synchronize()


def test_step_first_layer_runs_the_facade_checks() -> None:
    lay = ref.IcpLayout(128, 2)
    rig = PrefillRig(_multi_window_case(128), lay)
    base._warm()
    b = rig.bufs["b"]
    bad_send = b.send[0].to(torch.int64)
    step = rig.plans[0].prepare_prefill_step(
        rig.windows_arg(0), b.scratch[0], rig.nvalid, bad_send, b.recv[0], b.out[0]
    )
    q, ks = rig.layer_inputs(0)
    with pytest.raises(ValueError) as per_step:
        step.local(q, ks[0])
    w = rig.windows[0]
    with pytest.raises(ValueError) as per_call:
        rig.plans[0].prefill_local(
            q[w.start : w.end],
            ks[0],
            rig.block_table,
            rig.wplans[0][0],
            b.scratch[0],
            rig.nvalid[w.start : w.end],
            bad_send[:, w.start : w.end],
        )
    assert str(per_step.value) == str(per_call.value)
    bad_out = b.out[0][:, :1]
    step = rig.plans[0].prepare_prefill_step(
        rig.windows_arg(0), b.scratch[0], rig.nvalid, b.send[0], b.recv[0], bad_out
    )
    with pytest.raises(ValueError, match="output must be int32"):
        step.merge()
    with pytest.raises(ValueError, match="ordered and disjoint"):
        rig.plans[0].prepare_prefill_step(
            list(reversed(rig.windows_arg(0))),
            b.scratch[0],
            rig.nvalid,
            b.send[0],
            b.recv[0],
            b.out[0],
        )
    torch.cuda.synchronize()


# =========================================================================== decode, launch identity


def _snapshot_calls(calls: Sequence[cake_jit.RecordedLaunch]) -> list[tuple]:
    return [(c.name, c.entry, c.device, c.plan, list(c.args)) for c in calls]


def _assert_same_calls(got: list[tuple], want: list[tuple], what: str) -> None:
    assert len(got) == len(want), f"{what}: {len(got)} launches, want {len(want)}"
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g[0] == w[0] and g[1] is w[1] and g[2] == w[2] and g[3] == w[3], (
            f"{what}: launch {i} {g[0]} vs {w[0]}"
        )
        for j, (x, y) in enumerate(zip(g[4], w[4], strict=True)):
            assert cake_indexer._same_arg(x, y), (
                f"{what}: launch {i} {g[0]} argument {j}"
            )


@pytest.mark.parametrize("mode", ("legacy", "tagged", "pool"))
@pytest.mark.parametrize("scenario", base.DECODE_SCENARIOS)
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_decode_step_launches_equal_the_per_call_launches(
    lay: ref.IcpLayout, scenario: str, mode: str, monkeypatch
) -> None:
    """Over 7 layers (slot rotation 0, 1, 2, 0, ...; a new query and K pages
    per layer) the decode step's launches equal ``decode``'s for every rank."""
    case, extra = base._make_decode(lay.page_size, scenario)
    inp = base.DecodeInputs.build(case, lay, pitch_pad=extra.get("pitch_pad", 0))
    max_dql = extra.get("max_dql", case.decode_query_len)
    comm = base.EmulatedComm(lay.tp, case.tokens + 3)
    tagged = mode == "tagged"
    base._warm()
    launched: list[list[tuple]] = []
    monkeypatch.setattr(
        cake_jit,
        "launch_recorded",
        lambda calls, device: launched.append(_snapshot_calls(calls)),
    )
    layers = 7
    for r in range(lay.tp):
        plan = _plan(
            lay,
            r,
            init_blocks=case.init_blocks,
            local_blocks=case.local_blocks,
            max_decode_tokens=case.tokens + 3,
            max_decode_query_len=max_dql,
            max_num_seqs=max(case.batch, 1),
            num_sparse_layers=layers,
        )
        fcomm = comm.facade_comm(r)
        plan.check_comm(fcomm)
        scores = torch.empty(
            (case.tokens, H, inp.case.columns), dtype=torch.float32, device=DEV
        )
        out = torch.empty(
            (case.tokens, H // lay.tp, TOPK), dtype=torch.int32, device=DEV
        )
        pool = (
            torch.empty(
                msa_ops.msa_icp_decode_pool_bytes(max_decode_tokens=case.tokens + 3),
                dtype=torch.uint8,
                device=DEV,
            )
            if mode == "pool"
            else None
        )
        kwargs = dict(
            comm=fcomm, tagged=tagged, decode_query_len=case.decode_query_len, pool=pool
        )
        step = plan.prepare_decode_step(
            inp.block_table, inp.seq_lens, inp.nvalid, scores, out, **kwargs
        )
        for layer in range(layers):
            q = _rolled(inp.q, layer)
            k = _rolled(inp.shards[r], layer)
            slot = ref.slot_set(layer, layers)
            with cake_jit.record_launches() as want:
                plan.decode(
                    q, k, inp.block_table, inp.seq_lens, inp.nvalid, scores, out,
                    slot=slot, **kwargs,
                )  # fmt: skip
            launched.clear()
            step.run(q, k, slot=slot)
            assert len(launched) == 1
            _assert_same_calls(
                launched[0],
                _snapshot_calls(want),
                f"{case.label} {lay.name} {mode} rank {r} layer {layer}",
            )
            assert len(want) >= (2 if case.batch else 0)


# =========================================================================== distributed


def _dist_main(argv: Sequence[str]) -> int:
    """Body of one rank under ``torchrun``: ``--page P --out DIR``."""
    import argparse

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, required=True)
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
    report: dict[str, Any] = {
        "rank": rank,
        "world": world,
        "page": args.page,
        "checks": 0,
    }
    try:
        cap, steps, dql, max_dql, batch = 16, 5, 2, 4, 6
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

        # ---- decode steps: per-call eager, step eager, step captured + replays
        for layers, tagged in ((5, True), (5, False), (4, True)):
            plan = _plan(
                lay,
                rank,
                max_decode_tokens=cap,
                max_decode_query_len=max_dql,
                max_num_batched_tokens=4096,
                max_num_seqs=batch,
                num_sparse_layers=layers,
            )
            plan.warmup()
            plan.check_comm(comm)
            cases = base._decode_steps(
                lay, steps, layers, batch=batch, dql=dql, seed=9001 + layers,
                all_padding_step=3,
            )  # fmt: skip
            sd = base.StaticDecode(lay, cases, layers)
            scores = torch.empty(
                plan.decode_scores_shape(), dtype=torch.float32, device=dev
            )
            # one output for every layer (as vLLM's buffer), staged per layer
            dout = torch.full(
                (sd.tokens, hpr, TOPK), POISON_I32, dtype=torch.int32, device=dev
            )
            stage = torch.full(
                (layers, sd.tokens, hpr, TOPK),
                POISON_I32,
                dtype=torch.int32,
                device=dev,
            )
            hist = {
                arm: torch.full(
                    (steps, layers, sd.tokens, hpr, TOPK),
                    POISON_I32,
                    dtype=torch.int32,
                    device=dev,
                )  # fmt: skip
                for arm in ("call", "step")
            }

            def per_call(plan=plan, sd=sd, layers=layers, tagged=tagged, scores=scores,
                         dout=dout, stage=stage):  # fmt: skip
                for layer in range(layers):
                    plan.decode(
                        sd.q[layer].view(FP8), sd.k[rank].view(FP8), sd.block_table,
                        sd.seq_lens, sd.nvalid, scores, dout, comm=comm,
                        slot=ref.slot_set(layer, layers), tagged=tagged, decode_query_len=dql,
                    )  # fmt: skip
                    stage[layer].copy_(dout)

            def step_path(plan=plan, sd=sd, layers=layers, tagged=tagged, scores=scores,
                          dout=dout, stage=stage):  # fmt: skip
                st = plan.prepare_decode_step(
                    sd.block_table, sd.seq_lens, sd.nvalid, scores, dout, comm=comm,
                    tagged=tagged, decode_query_len=dql,
                )  # fmt: skip
                for layer in range(layers):
                    st.run(
                        sd.q[layer].view(FP8), sd.k[rank].view(FP8),
                        slot=ref.slot_set(layer, layers),
                    )  # fmt: skip
                    stage[layer].copy_(dout)

            graph = torch.cuda.CUDAGraph()
            for s in range(steps):
                sd.load(s)
                per_call()
                hist["call"][s].copy_(stage)
                if s < 2:
                    step_path()
                elif s == 2:
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        step_path()
                    graph.replay()
                else:
                    graph.replay()
                hist["step"][s].copy_(stage)
            torch.cuda.synchronize()
            got_call, got_step = hist["call"].cpu(), hist["step"].cpu()
            for s in range(steps):
                for layer in range(layers):
                    _, _, owner_outs, _ = base.decode_expect(cases[s][layer], lay)
                    what = f"rank {rank} decode L{layers} tagged={tagged} step {s} layer {layer}"
                    base.assert_words(
                        got_call[s, layer], owner_outs[rank], what + " per-call"
                    )
                    base.assert_words(
                        got_step[s, layer], got_call[s, layer], what + " step"
                    )
                    report["checks"] += 2
            assert int(comm.errors.item()) == 0
            del graph
            dist.barrier()

        # ---- mixed steps: decode chain (legacy rows), then prefill windows -> a2a -> merge
        case = ref.make_prefill_case(
            "dist_step", page_size=args.page, q_lens=[700, 1, 0, 90],
            prefixes=[0, 5000, 3, 260], padding_tokens=6, chunk=256, seed=777,
        )  # fmt: skip
        windows = case.windows()
        layers = 5  # slots 0, 1, 0, 1, 2: the decode step reuses slots 0 and 1
        plan = _plan(
            lay, rank, init_blocks=case.init_blocks, local_blocks=case.local_blocks,
            max_decode_tokens=cap, max_decode_query_len=max_dql,
            max_num_batched_tokens=8192 * len(windows), max_num_seqs=max(case.batch, batch),
            num_sparse_layers=layers,
        )  # fmt: skip
        plan.warmup()
        plan.check_comm(comm)
        dcases = base._decode_steps(lay, 1, layers, batch=batch, dql=dql, seed=4711)[0]
        sd = base.StaticDecode(lay, [dcases], layers)
        sd.load(0)
        scores = torch.empty(
            plan.decode_scores_shape(), dtype=torch.float32, device=dev
        )
        pcases = [
            dataclasses.replace(
                case,
                q=torch.roll(case.q, shifts=5 * layer, dims=0),
                cache=torch.roll(case.cache, shifts=layer, dims=0),
            )
            for layer in range(layers)
        ]
        pin = [base.prefill_case_inputs(c, lay) for c in pcases]
        bt, nv = pin[0]["block_table"], pin[0]["nvalid"]
        rows_, _, pitch = plan.prefill_scratch_shape(max(case.seq_lens))
        wplans = [
            plan.prefill_window_plan(
                wi,
                *base._window_meta(w),
                total_q=w.total_q,
                max_query_len=max(max(w.q_lens), 1),
                max_seq_len=max(max(w.kv_lens), 1),
            )  # fmt: skip
            for wi, w in enumerate(windows)
        ]
        t = case.total_q
        res = {}
        for arm in ("call", "step"):
            scratch = base._poisoned_f32((rows_, H, pitch))
            send = torch.full(
                (H, t, ref.ROW_WORDS), POISON_I32, dtype=torch.int32, device=dev
            )
            recv = torch.empty_like(send)
            recv4 = recv.view(world, hpr, t, ref.ROW_WORDS)
            out = torch.full((t, hpr, TOPK), POISON_I32, dtype=torch.int32, device=dev)
            dout = torch.full(
                (sd.tokens, hpr, TOPK), POISON_I32, dtype=torch.int32, device=dev
            )
            hp, hd = [], []
            if arm == "step":
                pstep = plan.prepare_prefill_step(
                    [(w.start, w.end, bt, wplans[wi]) for wi, w in enumerate(windows)],
                    scratch, nv, send, recv4, out,
                )  # fmt: skip
                dstep = plan.prepare_decode_step(
                    sd.block_table, sd.seq_lens, sd.nvalid, scores, dout, comm=comm,
                    tagged=False, decode_query_len=dql,
                )  # fmt: skip
            a2a = 0
            for layer in range(layers):
                q, k = pin[layer]["q"], pin[layer]["shards"][rank]
                slot = ref.slot_set(layer, layers)
                dq, dk = sd.q[layer].view(FP8), sd.k[rank].view(FP8)
                if arm == "call":
                    plan.decode(
                        dq, dk, sd.block_table, sd.seq_lens, sd.nvalid, scores, dout,
                        comm=comm, slot=slot, tagged=False, decode_query_len=dql,
                    )  # fmt: skip
                    for wi, w in enumerate(windows):
                        plan.prefill_local(
                            q[w.start : w.end], k, bt, wplans[wi], scratch,
                            nv[w.start : w.end], send[:, w.start : w.end],
                        )  # fmt: skip
                    dist.all_to_all_single(recv, send)
                    a2a += 1
                    plan.prefill_merge(recv4, nv, out)
                else:
                    dstep.run(dq, dk, slot=slot)
                    pstep.local(q, k)
                    dist.all_to_all_single(recv, send)
                    a2a += 1
                    pstep.merge()
                hp.append(
                    (send.clone(), out.clone(), scratch.view(torch.int32).clone())
                )
                hd.append(dout.clone())
            assert a2a == layers
            torch.cuda.synchronize()
            res[arm] = (hp, hd)
        for layer in range(layers):
            # the slow oracle for layer 0, the suite's validated fast reference after it
            exp = base.prefill_expect(
                pcases[layer], lay, fast=layer > 0, with_global=False
            )
            (cs, co, csc), (ss, so, ssc) = res["call"][0][layer], res["step"][0][layer]
            what = f"rank {rank} mixed layer {layer}"
            base.assert_words(cs.cpu(), exp.carriers[rank], what + " per-call carrier")
            base.assert_words(co.cpu(), exp.outputs[rank], what + " per-call ids")
            base.assert_words(ss.cpu(), cs.cpu(), what + " step carrier")
            base.assert_words(so.cpu(), co.cpu(), what + " step ids")
            base.assert_words(ssc.cpu(), csc.cpu(), what + " step score scratch")
            _, _, owner_outs, _ = base.decode_expect(dcases[layer], lay)
            base.assert_words(
                res["call"][1][layer].cpu(), owner_outs[rank], what + " decode per-call"
            )
            base.assert_words(
                res["step"][1][layer].cpu(), owner_outs[rank], what + " decode step"
            )
            report["checks"] += 7
        assert int(comm.errors.item()) == 0
        report["ok"] = True
    except BaseException as e:  # noqa: BLE001 - reported to the launching test
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"[:2000]
        raise
    finally:
        Path(args.out, f"rank{rank}.json").write_text(json.dumps(report))
        dist.destroy_process_group()
    return 0


@pytest.mark.parametrize("page", (128, 256))
@pytest.mark.parametrize("world", (2, 4))
def test_distributed_step_decode_and_mixed(world: int, page: int) -> None:
    """``torchrun`` with ``world`` ranks: decode steps (legacy and tagged,
    eager then CUDA Graph replays) and mixed steps (decode chain, prefill
    windows, one ``all_to_all_single`` per layer, merge) through the step path
    equal the per-call path bit for bit and the oracle."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _torchrun(world, ["--page", str(page)])


def _torchrun(world: int, args: list[str]) -> None:
    """:func:`base._torchrun` for this file's ``_dist_main``."""
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
