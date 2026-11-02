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

GPU tests of the MiniMax-M3 index-context-parallel (ICP) indexer kernels
(``flashinfer.msa_ops.msa_icp_*`` and ``prepare_msa_icp_indexer``) against the
CPU oracle of ``tests/test_helpers/cake_msa_icp_inputs.py``.

Single GPU: the ``W`` ranks of a TP2 / TP4 group run on one device.  Each rank
keeps its own index-K shard, score rows, carrier and receive storage; the
receive bases point at the other ranks' storage on the same device, so the
publishers store into "peer" rows exactly as over NVLink.  Launches are
serialized: every rank's scorer and publisher, then every owner's merge; the
prefill all-to-all is a device copy.  Every result is compared bit for bit:
score words (raw FP32 bits; untouched columns keep their poison), published
rows (all 33 legacy / 32 tagged words), carriers, merged ids and, after
multi-layer sequences, the complete receive storage and generation counters
against the oracle's protocol emulator.  Fixtures use integer FP8 values, so
every FP32 summation order is exact and no tolerance exists.

Coverage: TP2 / TP4 x P128 / P256 (compact and non-compact prefill scores),
causal tails at block / page / rank-row edges, mixed request lengths and the
mixed decode + prefill-chunk step, duplicate candidates (max kept), exact ties
(lower block id), forced begin / end blocks, empty shards, graph padding
(``nvalid == 0`` rows, ``seq_len == 0`` requests, all-padding steps), reuse of
poisoned score scratch / carriers / receive rows / fused-path pools, legacy and
generation-tagged rows with slot rotation over 1, 2, 3 and 5 layers and the
generation wrap, at least three consecutive CUDA Graph replays (and the eager
equivalent) with no allocation and no host synchronization between steps, the
spin-limit timeout (in a child process), the golden vectors of every kernel
role, and the transcribed vLLM merge / MSA publish rows as cross-provider
inputs.

Multi GPU (``test_distributed_*``): the same chains through the facade on 2 or
4 devices under ``torchrun`` with PyTorch symmetric-memory receive storage
(``recv_bases`` from the rendezvous) and the prefill exchange as
``all_to_all_single``; graph replays and eager steps run back to back with no
host synchronization or CPU collective between steps and are checked at the
end.

A device of compute capability 10.0, 10.3 or 10.7 must have every program; a
missing one raises ``NotImplementedError`` and fails.  Other devices skip.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch

import flashinfer.msa_ops as msa_ops
from flashinfer.experimental.cake_msa_icp_indexer import cake_backend, cake_jit
from tests.test_helpers import cake_msa_icp_inputs as ref

FP8 = torch.float8_e4m3fn
H = ref.NUM_HEADS
TOPK = ref.TOPK
SUPPORTED = ((10, 0), (10, 3), (10, 7))
LAYOUTS = tuple(ref.IcpLayout(p, w) for p in (128, 256) for w in (2, 4))
LAYOUT_IDS = [lay.name for lay in LAYOUTS]
# Poison of score words a kernel must not write (a NaN payload no scorer produces).
POISON_F32_BITS = 0x7FC0BEEF
POISON_I32 = 0x5A5A5A5A
MASK32 = 0xFFFFFFFF


def _capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability(0))


pytestmark = pytest.mark.skipif(
    _capability() not in SUPPORTED,
    reason="the Cake MSA ICP indexer needs compute capability 10.0, 10.3 or 10.7",
)

DEV = torch.device("cuda", 0)


# --------------------------------------------------------------------------- helpers


def _i32(x: Sequence[int] | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.int32).to(DEV)


def _i32_bits(bits: torch.Tensor) -> torch.Tensor:
    """int32 words of raw bits given as int64 in ``[0, 2**32)``."""
    b = bits.to(torch.int64) & MASK32
    return torch.where(b >= 1 << 31, b - (1 << 32), b).to(torch.int32)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """int32 view of FP32 words (CPU)."""
    return x.detach().cpu().contiguous().view(torch.int32)


def _poisoned_f32(shape, *, bits: int = POISON_F32_BITS) -> torch.Tensor:
    return torch.full(shape, ref.to_i32(bits), dtype=torch.int32, device=DEV).view(
        torch.float32
    )


def _first_diff(got: torch.Tensor, want: torch.Tensor) -> str:
    diff = (got != want).nonzero()
    if diff.numel() == 0:
        return "equal"
    idx = tuple(int(i) for i in diff[0])
    return f"{diff.shape[0]} differing words; first at {idx}: got {int(got[idx]):#x}, want {int(want[idx]):#x}"


def assert_words(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    got = got.detach().cpu()
    want = want.detach().cpu()
    if got.dtype == torch.float32:
        got = got.contiguous().view(torch.int32)
    if want.dtype == torch.float32:
        want = want.contiguous().view(torch.int32)
    assert tuple(got.shape) == tuple(want.shape), (
        f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    )
    got, want = got.to(torch.int64), want.to(torch.int64)
    assert torch.equal(got, want), f"{what}: {_first_diff(got, want)}"


def expected_plane(plane: torch.Tensor) -> torch.Tensor:
    """Oracle plane (NaN = untouched) -> expected int32 words with the poison."""
    bits = plane.contiguous().view(torch.int32).clone()
    bits[torch.isnan(plane)] = ref.to_i32(POISON_F32_BITS)
    return bits


def _fp8(t: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes (CPU uint8) -> float8 tensor on the device."""
    return t.contiguous().to(DEV).view(FP8)


@contextmanager
def no_alloc_no_sync(what: str):
    """No CUDA caching-allocator allocation and no synchronizing torch call inside."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(DEV).get("allocation.all.allocated", 0)
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")
    after = torch.cuda.memory_stats(DEV).get("allocation.all.allocated", 0)
    assert after == before, f"{what}: {after - before} allocations"


def _warm() -> None:
    cake_backend.warmup(DEV)


# --------------------------------------------------------------------------- emulated receive storage


class EmulatedComm:
    """Receive storage of ``W`` ranks on one device (one int32 allocation per
    rank laid out by :func:`msa_icp_comm_layout`), zeroed once (I-INIT)."""

    def __init__(self, world: int, capacity: int, *, generation_start: int = 0) -> None:
        self.world, self.hpr, self.capacity = world, H // world, capacity
        lay = msa_ops.msa_icp_comm_layout(world, self.hpr, capacity)
        self.layout = lay
        legacy_words = lay["tagged_offset_bytes"] // 4
        self.storage = [
            torch.zeros(lay["storage_words"], dtype=torch.int32, device=DEV)
            for _ in range(world)
        ]
        self.legacy = [s[:legacy_words].view(lay["legacy_shape"]) for s in self.storage]
        self.tagged = [
            s[legacy_words:].view(torch.int64).view(lay["tagged_shape"])
            for s in self.storage
        ]
        self.recv_bases = torch.tensor(
            [s.data_ptr() for s in self.storage], dtype=torch.int64, device=DEV
        )
        self.tagged_bases = self.recv_bases + lay["tagged_offset_bytes"]
        g = ref.to_i32(generation_start)
        self.generations = [
            torch.full(lay["generations_shape"], g, dtype=torch.int32, device=DEV)
            for _ in range(world)
        ]
        self.expected = [
            torch.full(lay["expected_shape"], g, dtype=torch.int32, device=DEV)
            for _ in range(world)
        ]
        self.errors = [
            torch.zeros(1, dtype=torch.int32, device=DEV) for _ in range(world)
        ]
        self.emulator = ref.CommEmulator(world, self.hpr, capacity, generation_start)

    def bases(self, tagged: bool) -> torch.Tensor:
        return self.tagged_bases if tagged else self.recv_bases

    def recv_slot(self, owner: int, slot: int, tagged: bool) -> torch.Tensor:
        return (self.tagged if tagged else self.legacy)[owner][slot]

    def facade_comm(self, rank: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            recv=self.legacy[rank],
            tagged_recv=self.tagged[rank],
            recv_bases=self.recv_bases,
            tagged_recv_bases=self.tagged_bases,
            publish_generations=self.generations[rank],
            expected_generations=self.expected[rank],
            errors=self.errors[rank],
        )

    def check_against_emulator(self, what: str) -> None:
        """Complete storage, counters and error words equal the protocol emulator's."""
        for r in range(self.world):
            assert_words(
                self.storage[r],
                self.emulator.storage[r],
                f"{what}: rank {r} receive storage",
            )
            assert_words(
                self.generations[r],
                self.emulator.generations[r],
                f"{what}: rank {r} publish generations",
            )
            assert_words(
                self.expected[r],
                self.emulator.expected[r],
                f"{what}: rank {r} expected generations",
            )
            assert int(self.errors[r].item()) == 0, f"{what}: rank {r} errors"


# --------------------------------------------------------------------------- decode chain


@dataclasses.dataclass
class DecodeInputs:
    """Device inputs of one decode case for every rank of a layout."""

    case: ref.DecodeCase
    layout: ref.IcpLayout
    q: torch.Tensor
    shards: list[torch.Tensor]
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    nvalid: torch.Tensor

    @classmethod
    def build(
        cls, case: ref.DecodeCase, lay: ref.IcpLayout, *, pitch_pad: int = 0
    ) -> "DecodeInputs":
        shards = []
        for r in range(lay.tp):
            shard = lay.shard(case.cache, r)  # [pages, R, 128] bytes
            if pitch_pad:  # compound page pitch: shard rows inside a wider page
                wide = torch.zeros(
                    (shard.shape[0], shard.shape[1] * 128 + pitch_pad),
                    dtype=torch.uint8,
                )
                wide[:, : shard.shape[1] * 128] = shard.reshape(shard.shape[0], -1)
                dev = wide.to(DEV).view(FP8)
                shards.append(dev.as_strided(shard.shape, (wide.shape[1], 128, 1)))
            else:
                shards.append(_fp8(shard))
        return cls(
            case=case,
            layout=lay,
            q=_fp8(case.q),
            shards=shards,
            block_table=_i32(case.block_table),
            seq_lens=_i32(case.seq_lens),
            nvalid=_i32(case.nvalid),
        )

    def expected(self) -> tuple:
        return decode_expect(self.case, self.layout)


def decode_expect(case: ref.DecodeCase, lay: ref.IcpLayout) -> tuple:
    """Oracle ``(planes, rows, owner outputs, unsharded output)`` of a decode case."""
    planes, rows = [], []
    for r in range(lay.tp):
        plane, _ = ref.decode_local_plane(case, lay, r)
        planes.append(plane)
        rows.append(ref.decode_local_rows(case, lay, r, plane))
    outs = [ref.decode_owner_output(case, lay, rows, o) for o in range(lay.tp)]
    return planes, rows, outs, ref.decode_global_output(case)


def decode_scores(
    inp: DecodeInputs, planes: list[torch.Tensor], *, max_dql: int, enable_pdl: bool
) -> None:
    c, lay = inp.case, inp.layout
    for r in range(lay.tp):
        msa_ops.msa_icp_decode_index_score_sharded(
            inp.q,
            inp.shards[r],
            inp.block_table,
            inp.seq_lens,
            planes[r],
            rank=r,
            world_size=lay.tp,
            page_size=c.page_size,
            decode_query_len=c.decode_query_len,
            max_decode_query_len=max_dql,
            enable_pdl=enable_pdl,
        )


def publish_all(
    comm: EmulatedComm,
    lay: ref.IcpLayout,
    scores: list[torch.Tensor],
    nvalid: torch.Tensor,
    *,
    slot: int,
    tagged: bool,
    fb: int,
    fe: int,
    enable_pdl: bool = False,
) -> None:
    for r in range(lay.tp):
        msa_ops.msa_icp_topk_select_publish(
            scores[r],
            nvalid,
            comm.bases(tagged),
            comm.errors[r],
            slot=slot,
            rank=r,
            world_size=lay.tp,
            heads_per_rank=comm.hpr,
            capacity=comm.capacity,
            force_begin_blocks=fb,
            force_end_blocks=fe,
            enable_pdl=enable_pdl,
            generations=comm.generations[r] if tagged else None,
            index_scale=lay.index_scale,
            index_offset=lay.index_offset(r),
        )


def merge_all(
    comm: EmulatedComm,
    outs: list[torch.Tensor],
    nvalid: torch.Tensor,
    *,
    slot: int,
    tagged: bool,
    init: int,
    local: int,
    enable_pdl: bool = False,
) -> None:
    for o in range(comm.world):
        msa_ops.msa_icp_merge_candidates(
            comm.recv_slot(o, slot, tagged),
            nvalid,
            outs[o],
            comm.errors[o],
            init_blocks=init,
            local_blocks=local,
            enable_pdl=enable_pdl,
            expected_generations=comm.expected[o][slot] if tagged else None,
        )


def check_published(
    comm: EmulatedComm,
    lay: ref.IcpLayout,
    rows: list[torch.Tensor],
    *,
    slot: int,
    tagged: bool,
    what: str,
) -> None:
    """Every (token, head) row of every rank sits at its destination row of
    the owner's slot set, all words included (legacy ready flag = 1, tagged
    words carry the next generation)."""
    t_count = rows[0].shape[0]
    for r in range(lay.tp):
        for h in range(H):
            owner, hl = h // comm.hpr, h % comm.hpr
            want = rows[r][:, h].to(torch.int64)
            if tagged:
                gen_prev = (
                    comm.emulator.generations[r][slot, h, :t_count].to(torch.int64)
                    & MASK32
                )
                gen = torch.tensor(
                    [ref.next_generation(int(g)) for g in gen_prev], dtype=torch.int64
                )
                want = (gen[:, None] << 32) | (
                    want[:, : 2 * TOPK] & MASK32
                )  # wraps to the int64 pattern
                got = comm.tagged[owner][slot, r, hl, :t_count].cpu()
            else:
                got = comm.legacy[owner][slot, r, hl, :t_count].cpu()
            assert_words(
                got, want, f"{what}: rows of rank {r} head {h} at owner {owner}"
            )


def run_decode_layer(
    comm: EmulatedComm,
    inp: DecodeInputs,
    planes: list[torch.Tensor],
    outs: list[torch.Tensor],
    *,
    slot: int,
    tagged: bool,
    max_dql: int,
    enable_pdl: bool,
    check: bool,
    what: str,
    expect=None,
) -> None:
    """One layer: scorers, publishers, merges for every rank.  With ``check``
    the planes, published rows, outputs and protocol state are compared."""
    c, lay = inp.case, inp.layout
    t_count = c.tokens
    decode_scores(inp, planes, max_dql=max_dql, enable_pdl=enable_pdl)
    publish_all(
        comm,
        lay,
        [p[:t_count] for p in planes],
        inp.nvalid,
        slot=slot,
        tagged=tagged,
        fb=c.init_blocks,
        fe=c.local_blocks,
        enable_pdl=enable_pdl,
    )
    exp_planes, exp_rows, exp_outs, glob = (
        expect if expect is not None else inp.expected()
    )
    if check:
        torch.cuda.synchronize()
        for r in range(lay.tp):
            assert_words(
                planes[r][:t_count],
                expected_plane(exp_planes[r]),
                f"{what}: rank {r} score plane",
            )
        check_published(comm, lay, exp_rows, slot=slot, tagged=tagged, what=what)
    for r in range(lay.tp):
        comm.emulator.publish(r, slot, exp_rows[r], tagged=tagged)
    merge_all(
        comm,
        outs,
        inp.nvalid,
        slot=slot,
        tagged=tagged,
        init=c.init_blocks,
        local=c.local_blocks,
        enable_pdl=enable_pdl,
    )
    em_outs = [
        comm.emulator.merge(
            o,
            slot,
            torch.tensor(c.nvalid),
            c.init_blocks,
            c.local_blocks,
            tagged=tagged,
        )
        for o in range(lay.tp)
    ]
    if check:
        torch.cuda.synchronize()
        for o in range(lay.tp):
            assert_words(
                outs[o][:t_count], exp_outs[o], f"{what}: owner {o} merged ids"
            )
            assert_words(em_outs[o], exp_outs[o], f"{what}: emulator owner {o}")
        assert_words(
            torch.cat([o[:t_count].cpu() for o in outs], dim=1),
            glob,
            f"{what}: unsharded oracle",
        )
        comm.check_against_emulator(what)


def new_planes(
    inp: DecodeInputs, *, poison_bits: int = POISON_F32_BITS
) -> list[torch.Tensor]:
    c = inp.case
    return [
        _poisoned_f32((c.tokens, H, c.columns), bits=poison_bits)
        for _ in range(inp.layout.tp)
    ]


def new_outputs(world: int, tokens: int) -> list[torch.Tensor]:
    return [
        torch.full(
            (tokens, H // world, TOPK), POISON_I32, dtype=torch.int32, device=DEV
        )
        for _ in range(world)
    ]


# --------------------------------------------------------------------------- prefill chain


def _scores_fast(
    case: ref.PrefillCase,
    w: ref.PrefillWindow,
    lay: ref.IcpLayout,
    rank: int,
    k: int,
    *,
    compact: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ref.prefill_window_scores`` in exact FP32 on the CPU (integer fixtures
    make every FP32 summation exact), chunked for large windows."""
    out = torch.full((w.total_q, H, k), float("-inf"), dtype=torch.float32)
    cov = torch.zeros(w.total_q, dtype=torch.int32)
    s, o = lay.index_scale, lay.index_offset(rank)
    table = ref.e4m3_table().to(torch.float32)
    for b in range(w.batch):
        if w.q_lens[b] == 0:
            continue
        n_pages = -(-w.kv_lens[b] // case.page_size)
        pages = case.request_pages(b)[:n_pages]
        keys = table[lay.shard(pages, rank).to(torch.int64)]  # [n, R, 128]
        g = lay.global_rows(rank, n_pages)  # [n, R]
        start = w.start + w.local_begin[b]
        for lo in range(0, w.q_lens[b], 512):
            hi = min(w.q_lens[b], lo + 512)
            q = table[case.q[start + lo : start + hi].to(torch.int64)]  # [N, H, 128]
            dots = torch.einsum("thd,crd->thcr", q, keys)
            q_pos = torch.arange(w.qo_offsets[b] + lo, w.qo_offsets[b] + hi)[
                :, None, None, None
            ]
            vis = (g[None, None] <= q_pos) & (g[None, None] < w.kv_lens[b])
            sc = (
                torch.where(vis, dots, torch.full_like(dots, float("-inf"))).amax(-1)
                + 0.0
            )  # [N, H, n]
            for i in range(hi - lo):
                r = w.local_begin[b] + lo + i
                u = min(w.kv_lens[b], w.qo_offsets[b] + lo + i + 1)
                nb = -(-u // ref.BLOCK)
                if lay.page_size == 256 and not compact:
                    m = min(k, nb)
                    gids = torch.arange(m)
                    held = gids[gids % 2 == o]
                    out[r, :, held] = sc[i, :, held // 2]
                    cov[r] = m
                else:
                    m = min(k, ref.local_prefix_count(nb, s, o))
                    out[r, :, :m] = sc[i, :, :m]
                    cov[r] = m
    return out, cov


@dataclasses.dataclass
class PrefillExpect:
    scores: list[list[torch.Tensor]]  # [rank][window] initialize=True rows
    coverage: list[list[torch.Tensor]]
    carriers: list[torch.Tensor]  # [rank] [H, T, 33]
    outputs: list[torch.Tensor]  # [owner] [T, hpr, 16]
    global_output: torch.Tensor | None


def prefill_expect(
    case: ref.PrefillCase, lay: ref.IcpLayout, *, fast: bool, with_global: bool
) -> PrefillExpect:
    windows = case.windows()
    nvalid = torch.tensor(case.nvalid(), dtype=torch.int64)
    scores, cover, carriers = [], [], []
    for r in range(lay.tp):
        per_s, per_c = [], []
        carrier = torch.empty((H, case.total_q, ref.ROW_WORDS), dtype=torch.int32)
        for w in windows:
            if fast:
                sc, cov = _scores_fast(case, w, lay, r, case.columns)
            else:
                sc, cov = ref.prefill_window_scores(case, w, lay, r)
            per_s.append(sc)
            per_c.append(cov)
            rows = ref.select_rows(
                ref.bits_tensor(sc).reshape(w.total_q * H, case.columns),
                nvalid[w.start : w.end].repeat_interleave(H),
                case.init_blocks,
                case.local_blocks,
                lay.index_scale,
                lay.index_offset(r),
            ).reshape(w.total_q, H, ref.ROW_WORDS)
            carrier[:, w.start : w.end] = rows.permute(1, 0, 2)
        scores.append(per_s)
        cover.append(per_c)
        carriers.append(carrier)
    outs = [ref.prefill_owner_output(case, lay, carriers, o) for o in range(lay.tp)]
    glob = ref.prefill_global_output(case) if with_global else None
    return PrefillExpect(scores, cover, carriers, outs, glob)


def plan_window(w: ref.PrefillWindow, storage: dict) -> Any:
    qo_indptr = [0]
    kv_indptr = [0]
    for q, kv in zip(w.q_lens, w.kv_lens, strict=True):
        qo_indptr.append(qo_indptr[-1] + q)
        kv_indptr.append(kv_indptr[-1] + kv)
    assert qo_indptr[:-1] == list(w.local_begin)
    return msa_ops.msa_icp_index_score_sharded_plan_device(
        _i32(w.q_lens),
        _i32(w.kv_lens),
        _i32(qo_indptr),
        _i32(kv_indptr),
        _i32(w.qo_offsets),
        H,
        total_q=w.total_q,
        max_query_len=max(max(w.q_lens), 1),
        max_seq_len=max(max(w.kv_lens), 1),
        packed_work_range=storage["work_range"],
        packed_work_info=storage["work_info"],
        workspace=storage["workspace"],
    )


def plan_buffers(total_q: int, batch: int, *, poison: bool = True) -> dict:
    sizes = msa_ops.msa_icp_index_score_plan_storage(
        num_qo_heads=H, total_q=total_q, batch=batch, device=DEV
    )
    fill = -0x5A5A5A5A5A5A5A5 if poison else 0
    return {
        "work_range": torch.full(
            (sizes["work_range_numel"],), fill, dtype=torch.int64, device=DEV
        ),
        "work_info": torch.full(
            (sizes["work_info_numel"],), fill, dtype=torch.int64, device=DEV
        ),
        "workspace": torch.full(
            (sizes["workspace_bytes"],), 0xA5, dtype=torch.uint8, device=DEV
        ),
    }


def prefill_case_inputs(case: ref.PrefillCase, lay: ref.IcpLayout) -> dict:
    return {
        "q": _fp8(case.q),
        "shards": [_fp8(lay.shard(case.cache, r)) for r in range(lay.tp)],
        "block_table": _i32(case.block_table),
        "nvalid": _i32(case.nvalid()),
    }


def run_prefill(
    case: ref.PrefillCase,
    lay: ref.IcpLayout,
    exp: PrefillExpect,
    *,
    initialize: bool,
    what: str,
    scratch: list[torch.Tensor] | None = None,
    carriers: list[torch.Tensor] | None = None,
    outputs: list[torch.Tensor] | None = None,
    check_scores: bool = True,
) -> None:
    """Per rank and window: device plan, scorer, carrier select; then the
    emulated exchange and every owner's merge.  Score scratch / carriers /
    outputs may be passed in (reuse) and are poisoned otherwise."""
    inp = prefill_case_inputs(case, lay)
    windows = case.windows()
    t_max = max(w.total_q for w in windows)
    storage = plan_buffers(t_max, case.batch)
    hpr = H // lay.tp
    if scratch is None:
        scratch = [_poisoned_f32((t_max, H, case.columns)) for _ in range(lay.tp)]
    if carriers is None:
        carriers = [
            torch.full(
                (H, case.total_q, ref.ROW_WORDS),
                POISON_I32,
                dtype=torch.int32,
                device=DEV,
            )
            for _ in range(lay.tp)
        ]
    if outputs is None:
        outputs = new_outputs(lay.tp, case.total_q)
    for r in range(lay.tp):
        for wi, w in enumerate(windows):
            plan = plan_window(w, storage)
            rows = scratch[r][: w.total_q]
            before = _bits(rows) if (check_scores and not initialize) else None
            msa_ops.msa_icp_index_score_sharded(
                inp["q"][w.start : w.end],
                inp["shards"][r],
                inp["block_table"],
                plan,
                rank=r,
                max_score=rows,
                initialize=initialize,
                tp=lay.tp,
                compact=lay.page_size == 256,
            )
            if check_scores:
                want = exp.scores[r][wi].contiguous().view(torch.int32)
                if not initialize:  # exactly the covered columns are written
                    cov = exp.coverage[r][wi]
                    col = torch.arange(case.columns)[None, None, :]
                    covered = (col < cov[:, None, None].to(torch.int64)).expand_as(want)
                    want = torch.where(covered, want, before)
                assert_words(
                    rows,
                    want,
                    f"{what}: rank {r} window {wi} scores (initialize={initialize})",
                )
            msa_ops.msa_icp_topk_select_prefill(
                rows,
                inp["nvalid"][w.start : w.end],
                carriers[r][:, w.start : w.end],
                force_begin_blocks=case.init_blocks,
                force_end_blocks=case.local_blocks,
                index_scale=lay.index_scale,
                index_offset=lay.index_offset(r),
            )
    for r in range(lay.tp):
        assert_words(carriers[r], exp.carriers[r], f"{what}: rank {r} carrier")
    errors = torch.zeros(1, dtype=torch.int32, device=DEV)
    for o in range(lay.tp):
        received = torch.stack(
            [carriers[r][o * hpr : (o + 1) * hpr] for r in range(lay.tp)]
        )
        msa_ops.msa_icp_merge_candidates(
            received,
            inp["nvalid"],
            outputs[o],
            errors,
            init_blocks=case.init_blocks,
            local_blocks=case.local_blocks,
            prefill=True,
        )
    torch.cuda.synchronize()
    for o in range(lay.tp):
        assert_words(outputs[o], exp.outputs[o], f"{what}: owner {o} merged ids")
    if exp.global_output is not None:
        assert_words(
            torch.cat([o.cpu() for o in outputs], dim=1),
            exp.global_output,
            f"{what}: unsharded oracle",
        )
    assert int(errors.item()) == 0


# =========================================================================== golden vectors


GOLDEN = None


def golden() -> dict:
    global GOLDEN
    if GOLDEN is None:
        GOLDEN = ref.load_golden()
    return GOLDEN


def _golden_names(section: str) -> list[str]:
    return [g["name"] for g in ref.load_golden(decode=False)[section]]


def _golden_entry(section: str, name: str) -> dict:
    (entry,) = [g for g in golden()[section] if g["name"] == name]
    return entry


@pytest.mark.parametrize("name", _golden_names("select"))
def test_golden_select_prefill(name: str) -> None:
    """F4: every golden select row (as all four heads of one token) into a
    carrier slice; all 33 words of every row."""
    g = _golden_entry("select", name)
    bits, nv, want = g["scores"], g["num_valid_pages"], g["expected_rows"]
    n, k = bits.shape
    scores = bits[:, None, :].expand(n, H, k).contiguous().to(DEV)
    send = torch.full(
        (H, n + 7, ref.ROW_WORDS), POISON_I32, dtype=torch.int32, device=DEV
    )
    msa_ops.msa_icp_topk_select_prefill(
        scores,
        _i32(nv),
        send[:, 3 : 3 + n],
        force_begin_blocks=g["force_begin"],
        force_end_blocks=g["force_end"],
        index_scale=g["index_scale"],
        index_offset=g["index_offset"],
    )
    torch.cuda.synchronize()
    out = send.cpu()
    for h in range(H):
        assert_words(out[h, 3 : 3 + n], want, f"{name}: head {h}")
    assert bool(
        (out[:, :3] == POISON_I32).all() and (out[:, 3 + n :] == POISON_I32).all()
    ), "rows outside the slice"


@pytest.mark.parametrize("tagged", (False, True), ids=("legacy", "tagged"))
@pytest.mark.parametrize("name", _golden_names("select"))
def test_golden_select_publish(name: str, tagged: bool) -> None:
    """F5: the same rows published from every rank of a TP2 group into slot 1."""
    g = _golden_entry("select", name)
    bits, nv, want = g["scores"], g["num_valid_pages"], g["expected_rows"]
    n, k = bits.shape
    world = 2
    comm = EmulatedComm(world, n + 2)
    scores = bits[:, None, :].expand(n, H, k).contiguous().to(DEV)
    for r in range(world):
        msa_ops.msa_icp_topk_select_publish(
            scores,
            _i32(nv),
            comm.bases(tagged),
            comm.errors[r],
            slot=1,
            rank=r,
            world_size=world,
            heads_per_rank=comm.hpr,
            capacity=comm.capacity,
            force_begin_blocks=g["force_begin"],
            force_end_blocks=g["force_end"],
            generations=comm.generations[r] if tagged else None,
            index_scale=g["index_scale"],
            index_offset=g["index_offset"],
        )
        comm.emulator.publish(
            r, 1, want[:, None, :].expand(n, H, ref.ROW_WORDS), tagged=tagged
        )
    torch.cuda.synchronize()
    comm.check_against_emulator(name)


@pytest.mark.parametrize("mode", ("legacy", "tagged", "prefill"))
@pytest.mark.parametrize("name", _golden_names("merge"))
def test_golden_merge(name: str, mode: str) -> None:
    """F7: golden candidate rows of W sources (as every owned head of one
    token) in legacy (ready flags, cleared after), tagged (next generation)
    and prefill-carrier form, for every (init, local, prefill) setting."""
    g = _golden_entry("merge", name)
    world = g["world_size"]
    hpr = H // world
    cands, nv = g["candidates"], g["num_valid_pages"]  # [N, W, 32]
    n = cands.shape[0]
    cap = n + 1
    for setting in g["settings"]:
        if setting["prefill"] != (mode == "prefill"):
            continue
        out = torch.full((n, hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV)
        errors = torch.zeros(1, dtype=torch.int32, device=DEV)
        words = cands.permute(1, 0, 2)[:, None].expand(
            world, hpr, n, 2 * TOPK
        )  # [W, hpr, N, 32]
        if mode == "tagged":
            expected = torch.full((hpr, cap), ref.to_i32(0xFFFFFFFF), dtype=torch.int32)
            recv = torch.zeros((world, hpr, cap, 2 * TOPK), dtype=torch.int64)
            recv[:, :, :n] = (
                words.to(torch.int64) & MASK32
            )  # generation 0xFFFFFFFF + 1 -> 1
            recv[:, :, :n] |= 1 << 32
            recv = recv.to(DEV)
            exp_dev = expected.to(DEV)
            msa_ops.msa_icp_merge_candidates(
                recv,
                _i32(nv),
                out,
                errors,
                init_blocks=setting["init_blocks"],
                local_blocks=setting["local_blocks"],
                expected_generations=exp_dev,
            )
        else:
            recv = torch.zeros((world, hpr, cap, ref.ROW_WORDS), dtype=torch.int32)
            recv[:, :, :n, : 2 * TOPK] = words
            recv[:, :, :n, 2 * TOPK] = 1
            recv = recv.to(DEV)
            msa_ops.msa_icp_merge_candidates(
                recv,
                _i32(nv),
                out,
                errors,
                init_blocks=setting["init_blocks"],
                local_blocks=setting["local_blocks"],
                prefill=mode == "prefill",
            )
        torch.cuda.synchronize()
        what = f"{name} {mode} init={setting['init_blocks']} local={setting['local_blocks']}"
        for hl in range(hpr):
            assert_words(out[:, hl], setting["expected_output"], f"{what}: head {hl}")
        assert int(errors.item()) == 0, what
        if mode == "legacy":
            assert bool((recv[:, :, :n, 2 * TOPK] == 0).all().item()), (
                f"{what}: ready flags not cleared"
            )
        if mode == "tagged":
            assert bool((exp_dev[:, :n] == 1).all().item()), (
                f"{what}: expected generations not advanced"
            )
            assert bool((exp_dev[:, n:] == -1).all().item()), (
                f"{what}: rows past T touched"
            )


@pytest.mark.parametrize("name", _golden_names("protocol"))
def test_golden_protocol(name: str) -> None:
    """F5 + F7 over multi-step, multi-layer sequences: slot rotation, legacy /
    tagged mixes and the generation wrap; then the complete receive storage
    equals the golden digest and the counters the golden values."""
    g = _golden_entry("protocol", name)
    world, cap, page = g["world_size"], g["capacity"], g["page_size"]
    lay = ref.IcpLayout(page, world)
    comm = EmulatedComm(world, cap, generation_start=g["generation_start"])
    for step in g["steps"]:
        slot, tagged, nv = step["slot"], step["tagged"], _i32(step["num_valid_pages"])
        t = nv.numel()
        scores = [sc.contiguous().to(DEV) for sc in step["scores"]]
        publish_all(
            comm,
            lay,
            scores,
            nv,
            slot=slot,
            tagged=tagged,
            fb=g["force_begin"],
            fe=g["force_end"],
        )
        outs = new_outputs(world, t)
        merge_all(
            comm,
            outs,
            nv,
            slot=slot,
            tagged=tagged,
            init=g["init_blocks"],
            local=g["local_blocks"],
        )
        torch.cuda.synchronize()
        what = f"{name} step {step['step']} layer {step['layer']}"
        for o in range(world):
            assert_words(outs[o], step["expected_output"][o], f"{what}: owner {o}")
    for r in range(world):
        digest = hashlib.sha256(comm.storage[r].cpu().numpy().tobytes()).hexdigest()
        assert digest == g["final_storage_sha256"][r], (
            f"{name}: rank {r} final receive storage"
        )
        assert_words(
            comm.generations[r].cpu().to(torch.int64) & MASK32,
            g["final_publish_generations"][r],
            f"{name}: rank {r} publish generations",
        )
        assert_words(
            comm.expected[r].cpu().to(torch.int64) & MASK32,
            g["final_expected_generations"][r],
            f"{name}: rank {r} expected generations",
        )
        assert int(comm.errors[r].item()) == 0


def _golden_decode_case(g: dict) -> ref.DecodeCase:
    return ref.DecodeCase(
        label=g["name"],
        page_size=g["page_size"],
        decode_query_len=g["decode_query_len"],
        seq_lens=g["seq_lens"].tolist(),
        block_table=g["block_table"].tolist(),
        columns=g["columns"],
        q=g["index_q"],
        cache=g["cache_pages"],
        nvalid=g["num_valid_pages"].tolist(),
        init_blocks=g["init_blocks"],
        local_blocks=g["local_blocks"],
    )


@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("name", _golden_names("decode"))
def test_golden_decode_chain(name: str, tp: int) -> None:
    """F3 -> F5 -> F7 on the golden decode cases: planes, published rows and
    merged ids equal the golden vectors (which equal the unsharded oracle)."""
    g = _golden_entry("decode", name)
    case = _golden_decode_case(g)
    lay = ref.IcpLayout(case.page_size, tp)
    per = g["per_tp"][str(tp)]
    inp = DecodeInputs.build(case, lay)
    planes = new_planes(inp)
    comm = EmulatedComm(tp, case.tokens)
    outs = new_outputs(tp, case.tokens)
    expect = (
        list(per["expected_scores"]),
        per["expected_rows"],
        per["expected_output"],
        g["expected_global_output"],
    )
    run_decode_layer(
        comm,
        inp,
        planes,
        outs,
        slot=0,
        tagged=False,
        max_dql=max(case.decode_query_len, 1),
        enable_pdl=False,
        check=True,
        what=f"{name} tp{tp}",
        expect=expect,
    )


def _golden_prefill_case(g: dict) -> ref.PrefillCase:
    return ref.PrefillCase(
        label=g["name"],
        page_size=g["page_size"],
        q_lens=g["q_lens"].tolist(),
        prefixes=g["prefixes"].tolist(),
        padding_tokens=g["padding_tokens"],
        chunk=g["chunk"],
        block_table=g["block_table"].tolist(),
        columns=g["columns"],
        q=g["index_q"],
        cache=g["cache_pages"],
        init_blocks=g["init_blocks"],
        local_blocks=g["local_blocks"],
    )


@pytest.mark.parametrize("initialize", (True, False), ids=("init", "noinit"))
@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("name", _golden_names("prefill"))
def test_golden_prefill_chain(name: str, tp: int, initialize: bool) -> None:
    """F1 -> F2 -> F4 -> exchange -> F7 on the golden prefill cases."""
    g = _golden_entry("prefill", name)
    case = _golden_prefill_case(g)
    lay = ref.IcpLayout(case.page_size, tp)
    per = g["per_tp"][str(tp)]
    exp = PrefillExpect(
        scores=[list(per["expected_scores_initialize_true"][r]) for r in range(tp)],
        coverage=[list(per["coverage_initialize_false"][r]) for r in range(tp)],
        carriers=list(per["expected_carrier"]),
        outputs=list(per["expected_output"]),
        global_output=g["expected_global_output"],
    )
    run_prefill(case, lay, exp, initialize=initialize, what=f"{name} tp{tp}")


@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("name", [n for n in _golden_names("prefill") if "p256" in n])
def test_golden_prefill_noncompact_scores(name: str, tp: int) -> None:
    """F2 non-compact P256 rows: one column per global B128 id, the blocks
    the rank does not hold are ``-inf`` (initialize) / untouched (covered
    columns written only)."""
    g = _golden_entry("prefill", name)
    case = _golden_prefill_case(g)
    lay = ref.IcpLayout(case.page_size, tp)
    per = g["per_tp"][str(tp)]
    k = per["noncompact_columns"]
    inp = prefill_case_inputs(case, lay)
    windows = case.windows()
    storage = plan_buffers(max(w.total_q for w in windows), case.batch)
    for r in range(tp):
        for wi, w in enumerate(windows):
            want_init = (
                per["expected_scores_noncompact"][r][wi]["scores"]
                .contiguous()
                .view(torch.int32)
            )
            cov = per["expected_scores_noncompact"][r][wi]["coverage"]
            plan = plan_window(w, storage)
            for initialize in (True, False):
                rows = _poisoned_f32((w.total_q, H, k))
                msa_ops.msa_icp_index_score_sharded(
                    inp["q"][w.start : w.end],
                    inp["shards"][r],
                    inp["block_table"],
                    plan,
                    rank=r,
                    max_score=rows,
                    initialize=initialize,
                    tp=tp,
                    compact=False,
                )
                want = want_init
                if not initialize:
                    col = torch.arange(k)[None, None, :]
                    covered = (col < cov[:, None, None].to(torch.int64)).expand_as(want)
                    want = torch.where(
                        covered,
                        want,
                        torch.full_like(want, ref.to_i32(POISON_F32_BITS)),
                    )
                assert_words(
                    rows,
                    want,
                    f"{name} tp{tp} rank {r} window {wi} initialize={initialize}",
                )


def test_fast_prefill_reference_matches_the_oracle() -> None:
    """The exact-FP32 reference used for the large cases equals the oracle."""
    for g in golden()["prefill"]:
        case = _golden_prefill_case(g)
        for lay in (ref.IcpLayout(case.page_size, 2), ref.IcpLayout(case.page_size, 4)):
            for r in range(lay.tp):
                for w in case.windows():
                    a, ca = ref.prefill_window_scores(case, w, lay, r)
                    b, cb_ = _scores_fast(case, w, lay, r, case.columns)
                    assert_words(
                        b, a, f"{case.label} {lay.name} rank {r} fast reference"
                    )
                    assert torch.equal(ca, cb_)


# =========================================================================== decode axes


def _decode_scenarios(page: int) -> list[tuple[str, dict]]:
    p = page
    return [
        (
            "causal_tails",
            dict(
                seq_lens=[
                    1,
                    2,
                    3,
                    4,
                    63,
                    64,
                    65,
                    127,
                    128,
                    129,
                    191,
                    192,
                    193,
                    255,
                    256,
                    257,
                    383,
                    384,
                    385,
                    511,
                    512,
                    513,
                ],
                k_kind="int4",
            ),
        ),
        (
            "mixed_lengths",
            dict(
                seq_lens=[
                    0,
                    1,
                    15 * 128,
                    16 * 128,
                    16 * 128 + 1,
                    33 * 128,
                    127 * 128 + 7,
                    128 * 128,
                    128 * 128 + 1,
                    0,
                    300,
                ],
                page_order="repeat",
                distinct_pages=37,
            ),
        ),
        (
            "long_rows",
            dict(
                seq_lens=[1279 * 128, 1280 * 128, 1344 * 128 + 5, 1536 * 128 + 1],
                page_order="repeat",
                distinct_pages=29,
            ),
        ),
        (
            "mtp_dql4",
            dict(seq_lens=[3, 4, 5, 2300, 129, 0], decode_query_len=4, max_dql=8),
        ),
        (
            "mtp_dql8",
            dict(seq_lens=[8, 9, 1000, 4 * p + 3], decode_query_len=8, max_dql=8),
        ),
        ("mtp_dql3", dict(seq_lens=[2, 130, 777], decode_query_len=3, max_dql=4)),
        (
            "ties_palette",
            dict(
                seq_lens=[2200, 300, 4097],
                k_kind="palette",
                page_order="repeat",
                distinct_pages=4,
                init_blocks=2,
                local_blocks=3,
            ),
        ),
        (
            "ties_all_equal",
            dict(
                seq_lens=[2100, 1, 5000],
                k_kind="all_equal",
                page_order="repeat",
                distinct_pages=2,
                q_kind="narrow",
            ),
        ),
        (
            "empty_shards",
            dict(seq_lens=[1, 32, 64, 65, 96, 128, 129, 160, 192, 193, 0, 0]),
        ),
        ("graph_padding_all", dict(seq_lens=[0, 0, 0, 0], columns=8)),
        ("compound_pitch", dict(seq_lens=[700, 1, 1500], pitch_pad=256)),
    ] + [
        (
            f"forced_{fb}_{fe}",
            dict(
                seq_lens=[1, 129, 700, 3000, 64 * 128 + 3],
                init_blocks=fb,
                local_blocks=fe,
                page_order="repeat",
                distinct_pages=7,
            ),
        )
        for fb, fe in ((0, 0), (0, 1), (1, 1), (2, 3), (3, 4), (8, 8), (16, 0), (0, 16))
    ]


def _make_decode(
    page: int, scenario: str, seed: int = 7896, *, dql: int | None = None
) -> tuple[ref.DecodeCase, dict]:
    """``dql`` overrides the scenario's ``decode_query_len`` (same requests)."""
    spec = dict(_decode_scenarios(page))[scenario]
    spec = dict(spec)
    extra = {k: spec.pop(k) for k in ("max_dql", "pitch_pad") if k in spec}
    label = f"{scenario}_p{page}"
    if dql is not None:
        spec["decode_query_len"] = dql
        label = f"{scenario}_dql{dql}_p{page}"
    case = ref.make_decode_case(label, page_size=page, seed=seed, **spec)
    return case, extra


DECODE_SCENARIOS = [name for name, _ in _decode_scenarios(128)]


@pytest.mark.parametrize("scenario", DECODE_SCENARIOS)
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_decode_chain(lay: ref.IcpLayout, scenario: str) -> None:
    """Scorer -> select/publish -> merge for every rank, legacy rows on slot 0
    then tagged rows on slot 2 over the same storage (PDL on for the second)."""
    case, extra = _make_decode(lay.page_size, scenario)
    inp = DecodeInputs.build(case, lay, pitch_pad=extra.get("pitch_pad", 0))
    expect = inp.expected()
    comm = EmulatedComm(lay.tp, case.tokens + 3)
    max_dql = extra.get("max_dql", case.decode_query_len)
    for slot, tagged, pdl in ((0, False, False), (2, True, True)):
        planes = new_planes(inp)
        outs = new_outputs(lay.tp, case.tokens)
        run_decode_layer(
            comm,
            inp,
            planes,
            outs,
            slot=slot,
            tagged=tagged,
            max_dql=max_dql,
            enable_pdl=pdl,
            check=True,
            what=f"{case.label} {lay.name} slot{slot}",
            expect=expect,
        )


@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_decode_reuses_poisoned_buffers(lay: ref.IcpLayout) -> None:
    """Score planes poisoned with NaN / +inf, receive rows poisoned in their
    payload words only (legacy ready 0, tagged tag == expected: I-INIT), the
    rows of a previous step left in place, and reversed request lengths, all
    over one receive storage."""
    a, _ = _make_decode(lay.page_size, "mixed_lengths", seed=11)
    b = ref.make_decode_case(
        "reversed",
        page_size=lay.page_size,
        seq_lens=list(reversed(a.seq_lens)),
        seed=12,
        page_order="repeat",
        distinct_pages=37,
    )
    comm = EmulatedComm(lay.tp, a.tokens)
    for st in comm.legacy:
        st[..., : 2 * TOPK] = POISON_I32
    for st in comm.tagged:
        st.fill_(0x13572468)  # tag 0 == expected generation: stale, payload poisoned
    comm.emulator.storage = [st.cpu().clone() for st in comm.storage]
    for case, poison in ((a, 0x7FC0DEAD), (b, 0x7F800000), (a, 0xFFC00000)):
        inp = DecodeInputs.build(case, lay)
        expect = inp.expected()
        planes = new_planes(inp, poison_bits=poison)
        decode_scores(inp, planes, max_dql=1, enable_pdl=False)
        torch.cuda.synchronize()
        for r in range(lay.tp):
            want = expect[0][r].contiguous().view(torch.int32).clone()
            want[torch.isnan(expect[0][r])] = ref.to_i32(poison)
            assert_words(
                planes[r], want, f"{case.label} rank {r} plane over poison {poison:#x}"
            )
        for slot, tagged in ((1, False), (0, True)):
            outs = new_outputs(lay.tp, case.tokens)
            publish_all(
                comm,
                lay,
                planes,
                inp.nvalid,
                slot=slot,
                tagged=tagged,
                fb=case.init_blocks,
                fe=case.local_blocks,
            )
            for r in range(lay.tp):
                comm.emulator.publish(r, slot, expect[1][r], tagged=tagged)
            merge_all(
                comm,
                outs,
                inp.nvalid,
                slot=slot,
                tagged=tagged,
                init=case.init_blocks,
                local=case.local_blocks,
            )
            for o in range(lay.tp):
                comm.emulator.merge(
                    o,
                    slot,
                    torch.tensor(case.nvalid),
                    case.init_blocks,
                    case.local_blocks,
                    tagged=tagged,
                )
            torch.cuda.synchronize()
            for o in range(lay.tp):
                assert_words(
                    outs[o],
                    expect[2][o],
                    f"{case.label} {lay.name} slot {slot} owner {o}",
                )
            comm.check_against_emulator(f"{case.label} {lay.name} slot {slot}")


@pytest.mark.parametrize("world", (2, 4))
def test_publish_addresses_follow_the_layout(world: int) -> None:
    """Every (slot, rank, head, token) destination of the golden addressing
    table: distinct rows per (token, head) land at their rows of the owner's
    storage and nowhere else (the complete storage equals the emulator's),
    with the ready word / generation tag at the tabulated offsets."""
    table = golden()["addressing"]
    rows = [
        dict(zip(table["columns"], r, strict=True))
        for r in table["rows"]
        if r[0] == world
    ]
    cap = rows[0]["capacity"]
    comm = EmulatedComm(world, cap)
    assert comm.layout["storage_words"] == rows[0]["storage_words"]
    k, t = 64, cap
    nv = torch.full((t,), k, dtype=torch.int32)
    for slot in range(3):
        for tagged in (False, True):
            for r in range(world):
                sc = torch.full((t, H, k), -1.0)
                for tok in range(t):
                    for h in range(H):
                        sc[tok, h, (2 * tok + h + 3 * r) % k] = 1.0
                exp_rows = ref.select_rows(
                    ref.bits_tensor(sc).reshape(t * H, k), nv.repeat_interleave(H), 0, 0
                ).reshape(t, H, ref.ROW_WORDS)
                msa_ops.msa_icp_topk_select_publish(
                    sc.to(DEV),
                    nv.to(DEV),
                    comm.bases(tagged),
                    comm.errors[r],
                    slot=slot,
                    rank=r,
                    world_size=world,
                    heads_per_rank=comm.hpr,
                    capacity=cap,
                    force_begin_blocks=0,
                    force_end_blocks=0,
                    generations=comm.generations[r] if tagged else None,
                )
                comm.emulator.publish(r, slot, exp_rows, tagged=tagged)
            torch.cuda.synchronize()
            comm.check_against_emulator(
                f"W{world} slot {slot} tagged={tagged} published"
            )
            for d in rows:
                if d["slot"] != slot:
                    continue
                flat = comm.storage[d["owner"]].cpu()
                if tagged:
                    assert int(flat[d["tagged_byte_offset"] // 4 + 1]) == 1, (
                        d
                    )  # generation tag of the first use
                else:
                    assert int(flat[d["legacy_ready_word"]]) == 1, d
                    assert int(flat[d["legacy_word_offset"] + TOPK]) >= 0, (
                        d
                    )  # first id of a full row
            outs = new_outputs(world, t)
            merge_all(comm, outs, nv.to(DEV), slot=slot, tagged=tagged, init=0, local=0)
            for o in range(world):
                comm.emulator.merge(o, slot, nv, 0, 0, tagged=tagged)
            torch.cuda.synchronize()
            comm.check_against_emulator(f"W{world} slot {slot} tagged={tagged} merged")


# =========================================================================== prefill axes


def _prefill_scenarios(page: int) -> list[tuple[str, dict, bool]]:
    """(name, make_prefill_case kwargs, fast reference)."""
    return [
        ("msa_smoke_1_0", dict(q_lens=[1], prefixes=[0], chunk=256), False),
        ("msa_smoke_1_128", dict(q_lens=[1], prefixes=[128], chunk=256), False),
        ("msa_smoke_1_256", dict(q_lens=[1], prefixes=[256], chunk=256), False),
        ("msa_smoke_1_512", dict(q_lens=[1], prefixes=[512], chunk=256), False),
        ("msa_smoke_65_0", dict(q_lens=[65], prefixes=[0], chunk=256), False),
        ("msa_smoke_257_1", dict(q_lens=[257, 1], prefixes=[0, 128], chunk=256), False),
        (
            "msa_smoke_segments",
            dict(
                q_lens=[0, 65, 0, 3],
                prefixes=[127, 194, 0, 127],
                padding_tokens=5,
                chunk=256,
            ),
            False,
        ),
        (
            "tails_rank_rows",
            dict(q_lens=[3, 3, 3, 3], prefixes=[62, 126, 190, 254], chunk=256),
            False,
        ),
        (
            "forced_2_3_ties",
            dict(
                q_lens=[40, 9],
                prefixes=[3000, 17],
                k_kind="palette",
                page_order="repeat",
                distinct_pages=3,
                init_blocks=2,
                local_blocks=3,
                chunk=32,
            ),
            False,
        ),
        (
            "forced_8_8",
            dict(
                q_lens=[16, 16],
                prefixes=[4096, 0],
                init_blocks=8,
                local_blocks=8,
                chunk=16,
            ),
            False,
        ),
        (
            "forced_16_0",
            dict(
                q_lens=[8],
                prefixes=[9000],
                init_blocks=16,
                local_blocks=0,
                chunk=8,
                page_order="repeat",
                distinct_pages=5,
            ),
            False,
        ),
        (
            "forced_0_16",
            dict(
                q_lens=[8],
                prefixes=[9000],
                init_blocks=0,
                local_blocks=16,
                chunk=8,
                page_order="repeat",
                distinct_pages=5,
            ),
            False,
        ),
        (
            "all_padding",
            dict(q_lens=[0, 0], prefixes=[0, 500], padding_tokens=9, chunk=8),
            False,
        ),
        (
            "mixed_decode_chunk",
            dict(
                q_lens=[1] * 14 + [8192],
                prefixes=[
                    129,
                    1000,
                    4095,
                    4096,
                    4097,
                    7777,
                    20000,
                    1,
                    0,
                    256,
                    511,
                    9999,
                    12345,
                    16383,
                    8192,
                ],
                chunk=8192,
                page_order="repeat",
                distinct_pages=61,
            ),
            True,
        ),
        (
            "chunk_8192_prefix_32k",
            dict(
                q_lens=[8192],
                prefixes=[32768],
                chunk=8192,
                page_order="repeat",
                distinct_pages=53,
            ),
            True,
        ),
    ]


PREFILL_SCENARIOS = [name for name, _, _ in _prefill_scenarios(128)]


def _make_prefill(
    page: int, scenario: str, seed: int = 7896
) -> tuple[ref.PrefillCase, bool]:
    for name, kw, fast in _prefill_scenarios(page):
        if name == scenario:
            return ref.make_prefill_case(
                f"{scenario}_p{page}", page_size=page, seed=seed, **kw
            ), fast
    raise KeyError(scenario)


@pytest.mark.parametrize("scenario", PREFILL_SCENARIOS)
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_prefill_chain(lay: ref.IcpLayout, scenario: str) -> None:
    """Device plan -> chunk scorer (``initialize=False`` into poisoned scratch,
    the facade's mode) -> carrier select -> exchange -> merge."""
    case, fast = _make_prefill(lay.page_size, scenario)
    exp = prefill_expect(case, lay, fast=fast, with_global=not fast)
    run_prefill(case, lay, exp, initialize=False, what=f"{case.label} {lay.name}")


@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_prefill_reuses_stale_buffers(lay: ref.IcpLayout) -> None:
    """One scratch / carrier / output set reused by a step, the step with the
    request lengths reversed, and the first step again (stale carriers and
    scores from a longer step must not leak)."""
    a = ref.make_prefill_case(
        "fwd",
        page_size=lay.page_size,
        q_lens=[200, 7, 0, 33],
        prefixes=[0, 900, 5, 2000],
        padding_tokens=4,
        chunk=64,
        seed=21,
    )
    b = ref.make_prefill_case(
        "rev",
        page_size=lay.page_size,
        q_lens=[33, 0, 7, 200],
        prefixes=[2000, 5, 900, 0],
        padding_tokens=4,
        chunk=64,
        seed=22,
        columns=a.columns,
    )
    t_max = max(w.total_q for w in a.windows())
    scratch = [
        _poisoned_f32((t_max, H, a.columns), bits=0x7F800000) for _ in range(lay.tp)
    ]
    carriers = [
        torch.full(
            (H, a.total_q, ref.ROW_WORDS), POISON_I32, dtype=torch.int32, device=DEV
        )
        for _ in range(lay.tp)
    ]
    outputs = new_outputs(lay.tp, a.total_q)
    for case in (a, b, a):
        exp = prefill_expect(case, lay, fast=False, with_global=True)
        run_prefill(
            case,
            lay,
            exp,
            initialize=False,
            what=f"{case.label} {lay.name} reuse",
            scratch=scratch,
            carriers=carriers,
            outputs=outputs,
            check_scores=False,
        )


def _window_meta(w: ref.PrefillWindow) -> tuple[torch.Tensor, ...]:
    qo_indptr, kv_indptr = [0], [0]
    for q, kv in zip(w.q_lens, w.kv_lens, strict=True):
        qo_indptr.append(qo_indptr[-1] + q)
        kv_indptr.append(kv_indptr[-1] + kv)
    return (
        _i32(w.q_lens),
        _i32(w.kv_lens),
        _i32(qo_indptr),
        _i32(kv_indptr),
        _i32(w.qo_offsets),
    )


@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_facade_prefill_matches_the_oracle(lay: ref.IcpLayout) -> None:
    """``prepare_msa_icp_indexer`` per rank: one plan per query window,
    per-chunk local scoring into the carrier (reused score scratch), the
    emulated exchange and the merge."""
    case = ref.make_prefill_case(
        "facade",
        page_size=lay.page_size,
        q_lens=[300, 1, 45],
        prefixes=[0, 4000, 700],
        padding_tokens=3,
        chunk=128,
        seed=31,
    )
    exp = prefill_expect(case, lay, fast=False, with_global=True)
    inp = prefill_case_inputs(case, lay)
    windows = case.windows()
    plans = [
        msa_ops.prepare_msa_icp_indexer(
            world_size=lay.tp,
            rank=r,
            num_index_heads=H,
            page_size=lay.page_size,
            init_blocks=case.init_blocks,
            local_blocks=case.local_blocks,
            max_decode_tokens=16,
            max_decode_query_len=4,
            max_num_batched_tokens=8192 * len(windows),
            max_num_seqs=case.batch,
            max_model_len=8192,
            num_sparse_layers=3,
            device=DEV,
        )
        for r in range(lay.tp)
    ]
    assert len(windows) <= plans[0].max_prefill_windows
    send = [
        torch.full(
            (H, case.total_q, ref.ROW_WORDS), POISON_I32, dtype=torch.int32, device=DEV
        )
        for _ in range(lay.tp)
    ]
    rows, heads, pitch = plans[0].prefill_scratch_shape(max(case.seq_lens))
    assert heads == H and pitch >= case.columns and rows >= case.chunk
    scratch = [_poisoned_f32((rows, H, pitch)) for _ in range(lay.tp)]
    for r in range(lay.tp):
        wplans = [
            plans[r].prefill_window_plan(
                wi,
                *_window_meta(w),
                total_q=w.total_q,
                max_query_len=max(max(w.q_lens), 1),
                max_seq_len=max(max(w.kv_lens), 1),
            )
            for wi, w in enumerate(windows)
        ]
        for wi, w in enumerate(windows):
            plans[r].prefill_local(
                inp["q"][w.start : w.end],
                inp["shards"][r],
                inp["block_table"],
                wplans[wi],
                scratch[r],
                inp["nvalid"][w.start : w.end],
                send[r][:, w.start : w.end],
            )
    torch.cuda.synchronize()
    for r in range(lay.tp):
        assert_words(send[r], exp.carriers[r], f"facade {lay.name} rank {r} carrier")
    hpr = H // lay.tp
    for o in range(lay.tp):
        received = torch.stack(
            [send[r][o * hpr : (o + 1) * hpr] for r in range(lay.tp)]
        )
        out = torch.full(
            (case.total_q, hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV
        )
        plans[o].prefill_merge(received, inp["nvalid"], out)
        torch.cuda.synchronize()
        assert_words(out, exp.outputs[o], f"facade {lay.name} owner {o}")


# =========================================================================== CUDA Graph replay


def _decode_steps(
    lay: ref.IcpLayout,
    steps: int,
    layers: int,
    *,
    batch: int,
    dql: int,
    seed: int,
    all_padding_step: int | None = None,
) -> list[list[ref.DecodeCase]]:
    """``steps`` x ``layers`` decode cases of one batch geometry: every step
    has new request lengths, K pages and positions; every layer a new Q."""
    out = []
    page = lay.page_size
    max_pages = 0
    raw = []
    for s in range(steps):
        rng = ref.SplitMix64(seed + 97 * s)
        if s == all_padding_step:
            seq = [0] * batch
        else:
            seq = [
                rng.randint(0, 3000) if b % 3 else rng.randint(dql, 9000)
                for b in range(batch)
            ]
        raw.append(seq)
        max_pages = max(max_pages, max(-(-x // page) for x in seq))
    columns = 4 * max(1, -(-max_pages // 4))
    for s, seq in enumerate(raw):
        base = ref.make_decode_case(
            f"step{s}",
            page_size=page,
            seq_lens=seq,
            decode_query_len=dql,
            columns=columns,
            seed=seed + 97 * s,
            page_order="shuffled",
            init_blocks=0,
            local_blocks=1,
        )
        layers_cases = [base]
        for layer in range(1, layers):
            q = ref._decode_q(
                ref.SplitMix64(seed + 97 * s + layer), base.tokens, H, "int2"
            )
            layers_cases.append(
                dataclasses.replace(base, q=q, label=f"step{s}_l{layer}")
            )
        out.append(layers_cases)
    return out


class StaticDecode:
    """Static device buffers of a captured multi-layer decode step (all ranks)."""

    def __init__(
        self, lay: ref.IcpLayout, cases: list[list[ref.DecodeCase]], layers: int
    ):
        c0 = cases[0][0]
        self.lay, self.layers = lay, layers
        self.tokens, self.batch, self.columns = c0.tokens, c0.batch, c0.columns
        self.max_pool = max(c[0].cache.shape[0] for c in cases)
        self.max_bt = max(len(c[0].block_table[0]) for c in cases)
        r_rows = lay.rows
        self.q = [
            torch.zeros((self.tokens, H, 128), dtype=torch.uint8, device=DEV)
            for _ in range(layers)
        ]
        self.k = [
            torch.zeros((self.max_pool, r_rows, 128), dtype=torch.uint8, device=DEV)
            for _ in range(lay.tp)
        ]
        self.block_table = torch.zeros(
            (self.batch, self.max_bt), dtype=torch.int32, device=DEV
        )
        self.seq_lens = torch.zeros(self.batch, dtype=torch.int32, device=DEV)
        self.nvalid = torch.zeros(self.tokens, dtype=torch.int32, device=DEV)
        self.planes = [
            _poisoned_f32((self.tokens, H, self.columns)) for _ in range(lay.tp)
        ]
        self.outs = [
            [
                torch.full(
                    (self.tokens, H // lay.tp, TOPK),
                    POISON_I32,
                    dtype=torch.int32,
                    device=DEV,
                )
                for _ in range(lay.tp)
            ]
            for _ in range(layers)
        ]
        # staged inputs of every step (device), copied into the static buffers between replays
        self.stage = []
        for step in cases:
            c = step[0]
            bt = torch.full((self.batch, self.max_bt), ref.PAD_PAGE, dtype=torch.int32)
            bt[:, : len(c.block_table[0])] = torch.tensor(
                c.block_table, dtype=torch.int32
            ).reshape(self.batch, -1)
            ks = []
            for r in range(lay.tp):
                k = torch.zeros((self.max_pool, r_rows, 128), dtype=torch.uint8)
                k[: c.cache.shape[0]] = lay.shard(c.cache, r)
                ks.append(k.to(DEV))
            self.stage.append(
                {
                    "q": [layer_case.q.to(DEV) for layer_case in step],
                    "k": ks,
                    "block_table": bt.to(DEV),
                    "seq_lens": _i32(c.seq_lens),
                    "nvalid": _i32(c.nvalid),
                }
            )

    def load(self, s: int) -> None:
        st = self.stage[s]
        for layer in range(self.layers):
            self.q[layer].copy_(st["q"][layer], non_blocking=True)
        for r in range(self.lay.tp):
            self.k[r].copy_(st["k"][r], non_blocking=True)
        self.block_table.copy_(st["block_table"], non_blocking=True)
        self.seq_lens.copy_(st["seq_lens"], non_blocking=True)
        self.nvalid.copy_(st["nvalid"], non_blocking=True)


def _decode_step(
    sd: StaticDecode,
    comm: EmulatedComm,
    *,
    tagged: bool,
    max_dql: int,
    dql: int,
    enable_pdl: bool,
) -> None:
    lay = sd.lay
    for layer in range(sd.layers):
        slot = ref.slot_set(layer, sd.layers)
        for r in range(lay.tp):
            msa_ops.msa_icp_decode_index_score_sharded(
                sd.q[layer].view(FP8),
                sd.k[r].view(FP8),
                sd.block_table,
                sd.seq_lens,
                sd.planes[r],
                rank=r,
                world_size=lay.tp,
                page_size=lay.page_size,
                decode_query_len=dql,
                max_decode_query_len=max_dql,
                enable_pdl=enable_pdl,
            )
        publish_all(
            comm,
            lay,
            sd.planes,
            sd.nvalid,
            slot=slot,
            tagged=tagged,
            fb=0,
            fe=1,
            enable_pdl=enable_pdl,
        )
        merge_all(
            comm,
            sd.outs[layer],
            sd.nvalid,
            slot=slot,
            tagged=tagged,
            init=0,
            local=1,
            enable_pdl=enable_pdl,
        )


GRAPH_CASES = [
    (mode, layers, gen)
    for mode in ("legacy", "tagged")
    for layers in (1, 2, 3, 5)
    for gen in ((0,) if mode == "legacy" else (0, 0xFFFFFFFE))
    if not (mode == "tagged" and layers == 1)
]


@pytest.mark.parametrize("graph", (True, False), ids=("graph", "eager"))
@pytest.mark.parametrize(
    "mode,layers,gen", GRAPH_CASES, ids=[f"{m}-L{n}-g{g:x}" for m, n, g in GRAPH_CASES]
)
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_decode_steps_replay(
    lay: ref.IcpLayout, mode: str, layers: int, gen: int, graph: bool
) -> None:
    """A captured multi-layer decode step (slot rotation as vLLM) replayed for
    >= 3 consecutive steps with new Q / K / lengths per step (one step all
    graph padding), with no allocation and no host synchronization between
    replays; every layer's output of every step and the final receive storage
    and counters are checked at the end against the protocol emulator."""
    tagged = mode == "tagged"
    steps, batch, dql, max_dql = 5, 6, 2, 4
    cases = _decode_steps(
        lay, steps, layers, batch=batch, dql=dql, seed=4242 + layers, all_padding_step=2
    )
    sd = StaticDecode(lay, cases, layers)
    comm = EmulatedComm(lay.tp, sd.tokens, generation_start=gen)
    hist = torch.full(
        (steps, layers, lay.tp, sd.tokens, H // lay.tp, TOPK),
        POISON_I32,
        dtype=torch.int32,
        device=DEV,
    )
    _warm()

    def run() -> None:
        _decode_step(sd, comm, tagged=tagged, max_dql=max_dql, dql=dql, enable_pdl=True)

    sd.load(0)
    run()  # eager step 0 (warm-up before capture)
    for layer in range(layers):
        for o in range(lay.tp):
            hist[0, layer, o].copy_(sd.outs[layer][o])
    if graph:
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            run()
        replay = g.replay
    else:
        replay = run
    with no_alloc_no_sync(f"{lay.name} {mode} L{layers} replays"):
        for s in range(1, steps):
            sd.load(s)
            replay()
            for layer in range(layers):
                for o in range(lay.tp):
                    hist[s, layer, o].copy_(sd.outs[layer][o], non_blocking=True)
    torch.cuda.synchronize()
    got = hist.cpu()
    for s in range(steps):
        for layer in range(layers):
            c = cases[s][layer]
            _, rows, outs, glob = decode_expect(c, lay)
            slot = ref.slot_set(layer, layers)
            for r in range(lay.tp):
                comm.emulator.publish(r, slot, rows[r], tagged=tagged)
            em = [
                comm.emulator.merge(
                    o, slot, torch.tensor(c.nvalid), 0, 1, tagged=tagged
                )
                for o in range(lay.tp)
            ]
            for o in range(lay.tp):
                assert_words(
                    got[s, layer, o],
                    outs[o],
                    f"{lay.name} {mode} step {s} layer {layer} owner {o}",
                )
                assert_words(em[o], outs[o], f"emulator step {s} layer {layer}")
            assert_words(
                torch.cat([got[s, layer, o] for o in range(lay.tp)], dim=1),
                glob,
                f"{lay.name} step {s} layer {layer} unsharded",
            )
    comm.check_against_emulator(f"{lay.name} {mode} L{layers} after {steps} steps")


@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_prefill_graph_replays(lay: ref.IcpLayout) -> None:
    """Captured device plan + scorer + carrier select + exchange copy + merge,
    replayed for 4 steps with new Q / K / lengths of one token geometry."""
    steps, chunk = 4, 64
    cases = []
    for s in range(steps):
        rng = ref.SplitMix64(900 + s)
        q_lens = [rng.randint(0, 60), rng.randint(1, 60), 0]
        q_lens.append(150 - sum(q_lens))
        prefixes = [rng.randint(0, 3000) for _ in q_lens]
        cases.append(
            ref.make_prefill_case(
                f"g{s}",
                page_size=lay.page_size,
                q_lens=q_lens,
                prefixes=prefixes,
                padding_tokens=10,
                chunk=chunk,
                seed=950 + s,
                columns=48,
            )
        )
    total = cases[0].total_q
    windows = [c.windows() for c in cases]
    nwin = len(windows[0])
    assert all(len(w) == nwin for w in windows)
    batch = cases[0].batch
    max_pool = max(c.cache.shape[0] for c in cases)
    max_bt = max(len(c.block_table[0]) for c in cases)
    hpr = H // lay.tp
    q = torch.zeros((total, H, 128), dtype=torch.uint8, device=DEV)
    k = [
        torch.zeros((max_pool, lay.rows, 128), dtype=torch.uint8, device=DEV)
        for _ in range(lay.tp)
    ]
    bt = torch.zeros((batch, max_bt), dtype=torch.int32, device=DEV)
    nvalid = torch.zeros(total, dtype=torch.int32, device=DEV)
    meta = [
        {
            n: torch.zeros(
                batch + (n in ("qo_indptr", "kv_indptr")), dtype=torch.int32, device=DEV
            )
            for n in ("q_lens", "kv_lens", "qo_indptr", "kv_indptr", "qo_offsets")
        }
        for _ in range(nwin)
    ]
    storage = [plan_buffers(chunk, batch) for _ in range(nwin)]
    scratch = [_poisoned_f32((chunk, H, 48)) for _ in range(lay.tp)]
    carriers = [
        torch.full((H, total, ref.ROW_WORDS), POISON_I32, dtype=torch.int32, device=DEV)
        for _ in range(lay.tp)
    ]
    received = [
        torch.empty((lay.tp, hpr, total, ref.ROW_WORDS), dtype=torch.int32, device=DEV)
        for _ in range(lay.tp)
    ]
    outs = new_outputs(lay.tp, total)
    errors = torch.zeros(1, dtype=torch.int32, device=DEV)
    stage = []
    for c, ws in zip(cases, windows, strict=True):
        b = torch.full((batch, max_bt), ref.PAD_PAGE, dtype=torch.int32)
        b[:, : len(c.block_table[0])] = torch.tensor(c.block_table, dtype=torch.int32)
        kk = []
        for r in range(lay.tp):
            t = torch.zeros((max_pool, lay.rows, 128), dtype=torch.uint8)
            t[: c.cache.shape[0]] = lay.shard(c.cache, r)
            kk.append(t.to(DEV))
        wm = []
        for w in ws:
            qi, ki = [0], [0]
            for a_, b_ in zip(w.q_lens, w.kv_lens, strict=True):
                qi.append(qi[-1] + a_)
                ki.append(ki[-1] + b_)
            wm.append(
                {
                    "q_lens": _i32(w.q_lens),
                    "kv_lens": _i32(w.kv_lens),
                    "qo_indptr": _i32(qi),
                    "kv_indptr": _i32(ki),
                    "qo_offsets": _i32(w.qo_offsets),
                }
            )
        stage.append(
            {
                "q": c.q.to(DEV),
                "k": kk,
                "bt": b.to(DEV),
                "nvalid": _i32(c.nvalid()),
                "meta": wm,
            }
        )
    max_seq = max(max(w.kv_lens) for ws in windows for w in ws)

    def load(s: int) -> None:
        st = stage[s]
        q.copy_(st["q"], non_blocking=True)
        bt.copy_(st["bt"], non_blocking=True)
        nvalid.copy_(st["nvalid"], non_blocking=True)
        for r in range(lay.tp):
            k[r].copy_(st["k"][r], non_blocking=True)
        for wi in range(nwin):
            for n, t in meta[wi].items():
                t.copy_(st["meta"][wi][n], non_blocking=True)

    def run() -> None:
        for r in range(lay.tp):
            for wi, w in enumerate(windows[0]):
                m = meta[wi]
                plan = msa_ops.msa_icp_index_score_sharded_plan_device(
                    m["q_lens"],
                    m["kv_lens"],
                    m["qo_indptr"],
                    m["kv_indptr"],
                    m["qo_offsets"],
                    H,
                    total_q=w.total_q,
                    max_query_len=min(chunk, w.total_q),
                    max_seq_len=max_seq,
                    packed_work_range=storage[wi]["work_range"],
                    packed_work_info=storage[wi]["work_info"],
                    workspace=storage[wi]["workspace"],
                )
                rows = scratch[r][: w.total_q]
                msa_ops.msa_icp_index_score_sharded(
                    q[w.start : w.end].view(FP8),
                    k[r].view(FP8),
                    bt,
                    plan,
                    rank=r,
                    max_score=rows,
                    initialize=False,
                    tp=lay.tp,
                    compact=lay.page_size == 256,
                )
                msa_ops.msa_icp_topk_select_prefill(
                    rows,
                    nvalid[w.start : w.end],
                    carriers[r][:, w.start : w.end],
                    force_begin_blocks=0,
                    force_end_blocks=1,
                    index_scale=lay.index_scale,
                    index_offset=lay.index_offset(r),
                )
        for o in range(lay.tp):
            for r in range(lay.tp):
                received[o][r].copy_(carriers[r][o * hpr : (o + 1) * hpr])
            msa_ops.msa_icp_merge_candidates(
                received[o],
                nvalid,
                outs[o],
                errors,
                init_blocks=0,
                local_blocks=1,
                prefill=True,
            )

    hist = torch.full(
        (steps, lay.tp, total, hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV
    )
    _warm()
    load(0)
    run()
    for o in range(lay.tp):
        hist[0, o].copy_(outs[o])
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        run()
    with no_alloc_no_sync(f"{lay.name} prefill replays"):
        for s in range(1, steps):
            load(s)
            g.replay()
            for o in range(lay.tp):
                hist[s, o].copy_(outs[o], non_blocking=True)
    torch.cuda.synchronize()
    got = hist.cpu()
    for s, c in enumerate(cases):
        exp = prefill_expect(c, lay, fast=False, with_global=True)
        for o in range(lay.tp):
            assert_words(
                got[s, o], exp.outputs[o], f"{lay.name} prefill step {s} owner {o}"
            )
        assert_words(
            torch.cat([got[s, o] for o in range(lay.tp)], dim=1),
            exp.global_output,
            f"{lay.name} prefill step {s} unsharded",
        )
    assert int(errors.item()) == 0


# =========================================================================== facade decode protocol rules


def test_facade_checks_comm_and_the_tagged_layer_rule() -> None:
    lay = ref.IcpLayout(256, 4)
    comm = EmulatedComm(4, 16)
    kw = dict(
        world_size=4,
        rank=1,
        num_index_heads=H,
        page_size=256,
        init_blocks=0,
        local_blocks=1,
        max_decode_tokens=16,
        max_decode_query_len=4,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=40000,
        num_sparse_layers=1,
        device=DEV,
    )
    plan = msa_ops.prepare_msa_icp_indexer(**kw)
    plan.check_comm(comm.facade_comm(1))
    bad = comm.facade_comm(1)
    bad.tagged_recv = comm.tagged[2]  # not in rank 1's storage
    with pytest.raises(ValueError):
        plan.check_comm(bad)
    bad = comm.facade_comm(1)
    bad.recv_bases = comm.recv_bases[:3]
    with pytest.raises(ValueError):
        plan.check_comm(bad)
    assert plan.comm_layout == msa_ops.msa_icp_comm_layout(4, 1, 16)
    assert plan.decode_scores_shape() == (
        16,
        H,
        msa_ops.msa_icp_decode_score_columns(40000, 256),
    )
    case = ref.make_decode_case("facade", page_size=256, seq_lens=[100, 2000], seed=5)
    inp = DecodeInputs.build(case, lay)
    scores = _poisoned_f32(plan.decode_scores_shape())
    out = torch.empty((case.tokens, 1, TOPK), dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError, match="two sparse layers"):
        plan.decode(
            inp.q,
            inp.shards[1],
            inp.block_table,
            inp.seq_lens,
            inp.nvalid,
            scores,
            out,
            comm=comm.facade_comm(1),
            slot=0,
            tagged=True,
            decode_query_len=1,
        )


def _comm_words_changed(comm: EmulatedComm) -> int:
    torch.cuda.synchronize()
    return sum(
        int(torch.count_nonzero(t))
        for t in (*comm.storage, *comm.generations, *comm.expected, *comm.errors)
    )


@pytest.mark.parametrize(
    "swap",
    (
        "recv_other_rank",
        "tagged_recv_other_rank",
        "recv_bases_short",
        "errors_float",
        "expected_generations_int64",
    ),
)
def test_facade_decode_catches_comm_fields_replaced_after_check_comm(swap: str) -> None:
    """A comm field replaced after ``check_comm`` (without calling it again)
    is caught by the next decode before any kernel runs, with real device
    addresses: no rank's storage, counter or error word changes."""
    lay = ref.IcpLayout(128, 2)
    comm = EmulatedComm(2, 16)
    plan = msa_ops.prepare_msa_icp_indexer(
        world_size=2,
        rank=1,
        num_index_heads=H,
        page_size=128,
        init_blocks=0,
        local_blocks=1,
        max_decode_tokens=16,
        max_decode_query_len=4,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_model_len=40000,
        num_sparse_layers=2,
        device=DEV,
    )
    plan.warmup()
    case = ref.make_decode_case("stale", page_size=128, seq_lens=[100, 2000], seed=7)
    inp = DecodeInputs.build(case, lay)
    scores = _poisoned_f32(plan.decode_scores_shape())
    out = torch.full((case.tokens, H // 2, TOPK), -7, dtype=torch.int32, device=DEV)
    c = comm.facade_comm(1)
    plan.check_comm(c)
    field, value, tagged = {
        "recv_other_rank": ("recv", comm.legacy[0], False),
        "tagged_recv_other_rank": ("tagged_recv", comm.tagged[0], True),
        "recv_bases_short": ("recv_bases", comm.recv_bases[:1], False),
        "errors_float": (
            "errors",
            torch.zeros(1, dtype=torch.float32, device=DEV),
            False,
        ),
        "expected_generations_int64": (
            "expected_generations",
            comm.expected[1].to(torch.int64),
            True,
        ),
    }[swap]
    setattr(c, field, value)
    with pytest.raises(ValueError):
        plan.decode(
            inp.q,
            inp.shards[1],
            inp.block_table,
            inp.seq_lens,
            inp.nvalid,
            scores,
            out,
            comm=c,
            slot=1,
            tagged=tagged,
            decode_query_len=1,
        )
    assert _comm_words_changed(comm) == 0
    assert bool((out == -7).all())


# =========================================================================== fused decode path (O-4)


def _fused_delivered() -> bool:
    arch = cake_jit.device_arch(0)
    return cake_jit.has_role("decode_topk", arch) and cake_jit.has_role(
        "topk16_publish", arch
    )


@pytest.mark.parametrize("dql", [1, 4], ids=["dql1", "dql4"])
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_fused_decode_is_byte_identical_to_the_modular_chain(
    lay: ref.IcpLayout, dql: int
) -> None:
    """O-4: the fused scorer + publisher writes the same receive rows as the
    modular scorer + select/publish, including after a large-split call
    leaves stale pool lists that a later small-split call must not read.
    ``dql`` 4 is the MTP decode (four query rows per request, same requests).
    Without a delivered fused program the entry raises NotImplementedError."""
    case, _ = _make_decode(lay.page_size, "mixed_lengths", dql=dql)
    inp = DecodeInputs.build(case, lay)
    pool_bytes = msa_ops.msa_icp_decode_pool_bytes(max_decode_tokens=case.tokens)
    pools = [
        torch.full((pool_bytes,), 0xFF, dtype=torch.uint8, device=DEV)
        for _ in range(lay.tp)
    ]
    comm = EmulatedComm(lay.tp, case.tokens)
    if not _fused_delivered():
        with pytest.raises(NotImplementedError):
            msa_ops.msa_icp_decode_score_publish(
                inp.q,
                inp.shards[0],
                inp.block_table,
                inp.seq_lens,
                inp.nvalid,
                pools[0],
                comm.recv_bases,
                comm.errors[0],
                rank=0,
                world_size=lay.tp,
                page_size=lay.page_size,
                decode_query_len=dql,
                slot=0,
                capacity=comm.capacity,
                force_begin_blocks=0,
                force_end_blocks=1,
            )
        return
    _, rows, outs_exp, _ = inp.expected()
    # a single-request call first: the most splits per request -> every pool list written
    one = ref.make_decode_case(
        "one", page_size=lay.page_size, seq_lens=[9000], decode_query_len=dql, seed=77
    )
    one_in = DecodeInputs.build(one, lay)
    comm_one = EmulatedComm(lay.tp, one.tokens)
    for tagged in (False, True):
        for r in range(lay.tp):
            msa_ops.msa_icp_decode_score_publish(
                one_in.q,
                one_in.shards[r],
                one_in.block_table,
                one_in.seq_lens,
                one_in.nvalid,
                pools[r],
                comm_one.bases(tagged),
                comm_one.errors[r],
                rank=r,
                world_size=lay.tp,
                page_size=lay.page_size,
                decode_query_len=dql,
                slot=0,
                capacity=one.tokens,
                force_begin_blocks=0,
                force_end_blocks=1,
                generations=comm_one.generations[r] if tagged else None,
            )
        _, one_rows, _, _ = one_in.expected()
        for r in range(lay.tp):
            comm_one.emulator.publish(r, 0, one_rows[r], tagged=tagged)
        merge_all(
            comm_one,
            new_outputs(lay.tp, one.tokens),
            one_in.nvalid,
            slot=0,
            tagged=tagged,
            init=0,
            local=1,
        )
        for o in range(lay.tp):
            comm_one.emulator.merge(o, 0, torch.tensor(one.nvalid), 0, 1, tagged=tagged)
    torch.cuda.synchronize()
    comm_one.check_against_emulator("fused single request")
    for slot, tagged in ((0, False), (1, True)):
        for r in range(lay.tp):
            msa_ops.msa_icp_decode_score_publish(
                inp.q,
                inp.shards[r],
                inp.block_table,
                inp.seq_lens,
                inp.nvalid,
                pools[r],
                comm.bases(tagged),
                comm.errors[r],
                rank=r,
                world_size=lay.tp,
                page_size=lay.page_size,
                decode_query_len=dql,
                slot=slot,
                capacity=comm.capacity,
                force_begin_blocks=case.init_blocks,
                force_end_blocks=case.local_blocks,
                generations=comm.generations[r] if tagged else None,
            )
        torch.cuda.synchronize()
        check_published(
            comm,
            lay,
            rows,
            slot=slot,
            tagged=tagged,
            what=f"fused {lay.name} slot {slot}",
        )
        for r in range(lay.tp):
            comm.emulator.publish(r, slot, rows[r], tagged=tagged)
        outs = new_outputs(lay.tp, case.tokens)
        merge_all(
            comm,
            outs,
            inp.nvalid,
            slot=slot,
            tagged=tagged,
            init=case.init_blocks,
            local=case.local_blocks,
        )
        for o in range(lay.tp):
            comm.emulator.merge(
                o,
                slot,
                torch.tensor(case.nvalid),
                case.init_blocks,
                case.local_blocks,
                tagged=tagged,
            )
        torch.cuda.synchronize()
        for o in range(lay.tp):
            assert_words(
                outs[o], outs_exp[o], f"fused {lay.name} slot {slot} owner {o}"
            )
        comm.check_against_emulator(f"fused {lay.name} slot {slot}")


# =========================================================================== cross-provider rows


def _msa_publish_row(
    bits: Sequence[int], nvalid: int, fb: int, fe: int, s: int, o: int
) -> list[int]:
    """32 words of the row MSA ``select_publish`` stores (its attribution
    model: higher local id wins exact ties when ``L <= 128``)."""
    gids, determinate = ref.msa_select_model_row(
        bits, nvalid, fb, fe, s, o, publish=True
    )
    assert determinate
    scores = [bits[(g - o) // s] & MASK32 if g >= 0 else ref.NEG_INF_BITS for g in gids]
    return [ref.to_i32(w) for w in scores] + list(gids)


@pytest.mark.parametrize("world", (2, 4))
def test_merge_matches_vllm_on_cake_and_msa_rows(world: int) -> None:
    """The Cake merge equals the transcribed vLLM merge on rows from the Cake
    selector and on rows of the MSA publish model (different exact-tie
    choices, D1), so either provider's rows can feed either merge."""
    hpr = H // world
    rng = ref.SplitMix64(4711 + world)
    t, k = 24, 100  # L <= 128: the MSA model is determinate
    nv = [rng.randint(0, k) for _ in range(t)]
    nv[0], nv[1] = 0, k
    modes = ("int_ties", "all_equal", "low_bits", "zero")
    sources: dict[str, list[torch.Tensor]] = {"cake": [], "msa": []}
    for _r in range(world):
        sc = [
            [ref.select_row_values(rng, modes[(i + h) % 4], k) for h in range(H)]
            for i in range(t)
        ]
        bits = torch.tensor(sc, dtype=torch.int64)
        cake = ref.select_rows(
            bits.reshape(t * H, k), torch.tensor(nv).repeat_interleave(H), 0, 1
        )
        sources["cake"].append(cake.reshape(t, H, ref.ROW_WORDS)[..., : 2 * TOPK])
        msa = [
            [_msa_publish_row(sc[i][h], nv[i], 0, 1, 1, 0) for h in range(H)]
            for i in range(t)
        ]
        sources["msa"].append(torch.tensor(msa, dtype=torch.int32))
    for provider, rows in sources.items():
        for o in range(world):
            recv = torch.zeros((world, hpr, t, ref.ROW_WORDS), dtype=torch.int32)
            for r in range(world):
                recv[r, :, :, : 2 * TOPK] = rows[r][:, o * hpr : (o + 1) * hpr].permute(
                    1, 0, 2
                )
            recv[..., 2 * TOPK] = 1
            out = torch.full((t, hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV)
            errors = torch.zeros(1, dtype=torch.int32, device=DEV)
            msa_ops.msa_icp_merge_candidates(
                recv.to(DEV), _i32(nv), out, errors, init_blocks=0, local_blocks=1
            )
            torch.cuda.synchronize()
            got = out.cpu()
            for i in range(t):
                for hl in range(hpr):
                    cands = [
                        (int(recv[r, hl, i, j]) & MASK32, int(recv[r, hl, i, TOPK + j]))
                        for r in range(world)
                        for j in range(TOPK)
                    ]
                    want = ref.vllm_merge_transcription(
                        cands, world, nv[i], 0, 1, False
                    )
                    assert got[i, hl].tolist() == want, (
                        f"{provider} W{world} owner {o} token {i} head {hl}"
                    )
            assert int(errors.item()) == 0


# =========================================================================== spin-limit timeout


def _host_mapped_errors() -> tuple[torch.Tensor, torch.Tensor]:
    """An int32 error word in pinned host memory and its device view: the
    count stays readable after a trap has killed the CUDA context."""
    host = torch.zeros(1, dtype=torch.int32, pin_memory=True)

    class _Cai:
        __cuda_array_interface__ = {
            "shape": (1,),
            "typestr": "<i4",
            "data": (host.data_ptr(), False),
            "version": 3,
        }

    return host, torch.as_tensor(_Cai(), device=DEV)


def _run_child(flag: str, arg: str, *, env: dict[str, str] | None = None) -> dict:
    """Run this file as ``python FILE flag arg`` and return its JSON report line."""
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), flag, arg],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    tag = "CHILD_REPORT "
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(tag)]
    assert lines, (
        f"{flag} {arg}: no report (rc={proc.returncode})\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )
    return json.loads(lines[-1][len(tag) :])


def _timeout_child(kind: str) -> int:
    """Child process: run one call that must spin out; print the error count
    read from host-mapped memory after the trap (the CUDA context is dead)."""
    torch.cuda.set_device(0)
    host, errors = _host_mapped_errors()
    status, message = "returned", ""
    try:
        world, cap, t = 2, 4, 3
        comm = EmulatedComm(world, cap)
        nv = _i32([5, 5, 5])
        out = torch.zeros((t, 2, TOPK), dtype=torch.int32, device=DEV)
        torch.cuda.synchronize()
        if kind == "legacy_merge":  # no publisher: the ready flags stay 0
            msa_ops.msa_icp_merge_candidates(
                comm.legacy[0][0],
                nv,
                out,
                errors,
                init_blocks=0,
                local_blocks=1,
                spin_limit=2000,
            )
        elif kind == "tagged_merge":  # rows carry a stale tag
            msa_ops.msa_icp_merge_candidates(
                comm.tagged[0][0],
                nv,
                out,
                errors,
                init_blocks=0,
                local_blocks=1,
                spin_limit=2000,
                expected_generations=comm.expected[0][0],
            )
        else:  # legacy publish into rows whose ready flag was never consumed
            for st in comm.legacy:
                st[..., 2 * TOPK] = 1
            scores = torch.zeros((t, H, 8), dtype=torch.float32, device=DEV)
            msa_ops.msa_icp_topk_select_publish(
                scores,
                nv,
                comm.recv_bases,
                errors,
                slot=0,
                rank=0,
                world_size=2,
                heads_per_rank=2,
                capacity=cap,
                force_begin_blocks=0,
                force_end_blocks=1,
                spin_limit=2000,
            )
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - the trap surfaces as a CUDA error
        status, message = "raised", f"{type(e).__name__}: {e}"[:400]
    print(
        "CHILD_REPORT "
        + json.dumps({"status": status, "errors": int(host[0]), "message": message}),
        flush=True,
    )
    return 0


@pytest.mark.parametrize("kind", ("legacy_merge", "tagged_merge", "legacy_publish"))
def test_spin_limit_counts_an_error_and_traps(kind: str) -> None:
    report = _run_child("--timeout-child", kind)
    assert report["status"] == "raised", (
        f"{kind}: the kernel returned instead of trapping: {report}"
    )
    assert report["errors"] >= 1, f"{kind}: errors not incremented: {report}"


# =========================================================================== first exchange after warmup

FIRST_EXCHANGE_CASE = "k300_s1_fb2_fe3"  # golden select entry published by every rank
FIRST_EXCHANGE_SLOT = 1


@dataclasses.dataclass
class FirstExchange:
    """One publish + merge exchange of ``world`` ranks: every rank publishes
    the golden select case's scores; each owner merges its slot set."""

    world: int
    tagged: bool
    scores: torch.Tensor
    nv_host: torch.Tensor
    nv: torch.Tensor
    rows: torch.Tensor  # [T, H, 33] expected select rows of every rank
    g: dict

    @classmethod
    def build(cls, world: int, tagged: bool) -> "FirstExchange":
        g = _golden_entry("select", FIRST_EXCHANGE_CASE)
        bits, nv = g["scores"], g["num_valid_pages"]
        n, k = bits.shape
        return cls(
            world=world,
            tagged=tagged,
            scores=bits[:, None, :].expand(n, H, k).contiguous().to(DEV),
            nv_host=torch.as_tensor(nv),
            nv=_i32(nv),
            rows=g["expected_rows"][:, None, :].expand(n, H, ref.ROW_WORDS),
            g=g,
        )

    @property
    def tokens(self) -> int:
        return self.scores.shape[0]

    @property
    def capacity(self) -> int:
        return self.tokens + 2

    def new_output(self) -> torch.Tensor:
        return torch.full(
            (self.tokens, H // self.world, TOPK),
            POISON_I32,
            dtype=torch.int32,
            device=DEV,
        )

    def publish(self, bases, generations, errors, rank: int) -> None:
        msa_ops.msa_icp_topk_select_publish(
            self.scores,
            self.nv,
            bases,
            errors,
            slot=FIRST_EXCHANGE_SLOT,
            rank=rank,
            world_size=self.world,
            heads_per_rank=H // self.world,
            capacity=self.capacity,
            force_begin_blocks=self.g["force_begin"],
            force_end_blocks=self.g["force_end"],
            generations=generations if self.tagged else None,
            index_scale=self.g["index_scale"],
            index_offset=self.g["index_offset"],
        )

    def merge(self, recv_slot, expected_slot, out, errors) -> None:
        msa_ops.msa_icp_merge_candidates(
            recv_slot,
            self.nv,
            out,
            errors,
            init_blocks=self.g["force_begin"],
            local_blocks=self.g["force_end"],
            expected_generations=expected_slot if self.tagged else None,
        )

    def expected(self, owner: int) -> torch.Tensor:
        em = ref.CommEmulator(self.world, H // self.world, self.capacity)
        for r in range(self.world):
            em.publish(r, FIRST_EXCHANGE_SLOT, self.rows, tagged=self.tagged)
        return em.merge(
            owner,
            FIRST_EXCHANGE_SLOT,
            self.nv_host,
            self.g["force_begin"],
            self.g["force_end"],
            tagged=self.tagged,
        )


def _first_exchange_child(mode: str) -> int:
    """Child process with lazy module loading: ``warmup()`` and then the
    process's first publish / merge launches, the merges enqueued first on
    their own stream so they spin while the publishers launch for the first
    time.  A publisher whose module still had to load would stall behind the
    spinning merge until the spin limit traps."""
    torch.cuda.set_device(0)
    host, errors = _host_mapped_errors()
    report: dict[str, Any] = {
        "status": "returned",
        "lazy": os.environ.get("CUDA_MODULE_LOADING"),
        "mismatch": [],
    }
    try:
        cake_backend.warmup(DEV)
        world = 2
        ex = FirstExchange.build(world, mode == "tagged")
        comm = EmulatedComm(world, ex.capacity)
        outs = [ex.new_output() for _ in range(world)]
        merges, publishes = torch.cuda.Stream(), torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(merges):
            for o in range(world):
                ex.merge(
                    comm.recv_slot(o, FIRST_EXCHANGE_SLOT, ex.tagged),
                    comm.expected[o][FIRST_EXCHANGE_SLOT],
                    outs[o],
                    errors,
                )
        with torch.cuda.stream(publishes):
            for r in range(world):
                ex.publish(comm.bases(ex.tagged), comm.generations[r], errors, r)
        torch.cuda.synchronize()
        report["mismatch"] = [
            o
            for o in range(world)
            if not torch.equal(outs[o].cpu(), ex.expected(o).to(torch.int32))
        ]
    except Exception as e:  # noqa: BLE001 - a trap surfaces as a CUDA error
        report["status"] = "raised"
        report["message"] = f"{type(e).__name__}: {e}"[:400]
    report["errors"] = int(host[0])
    print("CHILD_REPORT " + json.dumps(report), flush=True)
    return 0


@pytest.mark.parametrize("mode", ("legacy", "tagged"))
def test_first_exchange_after_warmup_with_lazy_loading(mode: str) -> None:
    """``CUDA_MODULE_LOADING=LAZY``: after ``warmup()`` the process's first-ever
    publish + merge exchange completes, even with the merges already spinning
    when the publishers launch (warmup loads every program up front)."""
    env = dict(os.environ, CUDA_MODULE_LOADING="LAZY")
    report = _run_child("--first-exchange-child", mode, env=env)
    assert report["lazy"] == "LAZY", report
    assert report["status"] == "returned", f"{mode}: first exchange failed: {report}"
    assert report["errors"] == 0, f"{mode}: protocol errors counted: {report}"
    assert report["mismatch"] == [], f"{mode}: merged ids differ: {report}"


# =========================================================================== multi-GPU (torchrun)


def _dist_main(argv: Sequence[str]) -> int:
    """Body of one rank under ``torchrun``: ``--page P --out DIR``."""
    import argparse

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-exchange", choices=("legacy", "tagged"))
    args = ap.parse_args(argv)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    DEV = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=DEV)
    lay = ref.IcpLayout(args.page, world)
    hpr = H // world
    report: dict[str, Any] = {
        "rank": rank,
        "world": world,
        "page": args.page,
        "checks": 0,
        "device": torch.cuda.get_device_name(DEV),
    }
    try:
        backend = getattr(symm_mem, "get_backend", lambda d: None)(DEV)
        assert backend in (None, "CUDA"), (
            f"symmetric-memory backend {backend!r}: tagged rows need CUDA peers"
        )
        report["symm_mem_backend"] = backend
        cap, steps, dql, max_dql = 16, 6, 2, 4  # 16 rows: the 8-request decode configs
        if args.first_exchange:  # rows of every token of the golden select case
            cap = max(12, FirstExchange.build(world, False).capacity)
        layout = msa_ops.msa_icp_comm_layout(world, hpr, cap)
        group = dist.group.WORLD.group_name
        if hasattr(symm_mem, "enable_symm_mem_for_group"):
            symm_mem.enable_symm_mem_for_group(group)
        storage = symm_mem.empty(layout["storage_words"], dtype=torch.int32, device=DEV)
        storage.zero_()
        handle = symm_mem.rendezvous(storage, group)
        bases = torch.tensor(
            [int(ptr) for ptr in handle.buffer_ptrs], dtype=torch.int64, device=DEV
        )
        assert int(bases[rank]) == storage.data_ptr()
        legacy_words = layout["tagged_offset_bytes"] // 4
        comm = types.SimpleNamespace(
            recv=storage[:legacy_words].view(layout["legacy_shape"]),
            tagged_recv=storage[legacy_words:]
            .view(torch.int64)
            .view(layout["tagged_shape"]),
            recv_bases=bases,
            tagged_recv_bases=bases + layout["tagged_offset_bytes"],
            publish_generations=torch.zeros(
                layout["generations_shape"], dtype=torch.int32, device=DEV
            ),
            expected_generations=torch.zeros(
                layout["expected_shape"], dtype=torch.int32, device=DEV
            ),
            errors=torch.zeros(1, dtype=torch.int32, device=DEV),
        )
        torch.cuda.synchronize()
        dist.barrier()  # I-INIT: every rank's storage is zeroed before any publish
        if args.first_exchange:
            _dist_first_exchange(
                args.first_exchange, rank, world, cap, comm, report, dist
            )
            report["ok"] = True
            return 0
        # decode: eager steps 0-1, then one captured step replayed for steps 2-5, checked at the end;
        # the default chain (score plane + select publish) and the fused option (``pool``: list
        # scorer + list publisher) share the receive rows and the oracle.  Batch 8 without a pool
        # takes the facade's chain rule (the fused chain in its own pool on sm_107a at P128 W4 and W2).
        rule = cake_backend.fused_decode_min_batch(
            cake_jit.device_arch(local), args.page, world
        )
        report["fused_rule"] = rule
        fused_calls = [0]
        fused_entry = cake_backend.decode_score_publish

        def counted_fused(*a, **k) -> None:
            fused_calls[0] += 1
            fused_entry(*a, **k)

        cake_backend.decode_score_publish = counted_fused
        for layers, tagged, fused, batch in (
            (1, False, False, 6),
            (2, True, False, 6),
            (3, True, False, 6),
            (5, True, False, 6),
            (5, False, False, 6),
            (2, False, False, 6),
            (3, True, True, 6),
            (2, False, True, 6),
            (2, True, False, 8),
            (3, False, False, 8),
        ):
            plan = msa_ops.prepare_msa_icp_indexer(
                world_size=world,
                rank=rank,
                num_index_heads=H,
                page_size=args.page,
                init_blocks=0,
                local_blocks=1,
                max_decode_tokens=cap,
                max_decode_query_len=max_dql,
                max_num_batched_tokens=4096,
                max_num_seqs=batch,
                max_model_len=16384,
                num_sparse_layers=layers,
                device=DEV,
            )
            plan.warmup()
            plan.check_comm(comm)
            assert plan.fused_decode_min_batch == rule
            assert (plan._fused_pool is None) == (rule is None)
            fused_calls[0] = 0
            cases = _decode_steps(
                lay,
                steps,
                layers,
                batch=batch,
                dql=dql,
                seed=31337 + layers,
                all_padding_step=3,
            )
            sd = StaticDecode(lay, cases, layers)
            scores = torch.empty(
                plan.decode_scores_shape(), dtype=torch.float32, device=DEV
            )
            assert scores.shape[2] >= sd.columns
            pool = (
                torch.empty(
                    msa_ops.msa_icp_decode_pool_bytes(max_decode_tokens=cap),
                    dtype=torch.uint8,
                    device=DEV,
                )
                if fused
                else None
            )
            outs = [
                torch.full(
                    (sd.tokens, hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV
                )
                for _ in range(layers)
            ]
            hist = torch.full(
                (steps, layers, sd.tokens, hpr, TOPK),
                POISON_I32,
                dtype=torch.int32,
                device=DEV,
            )

            def step(
                plan=plan,
                sd=sd,
                scores=scores,
                outs=outs,
                layers=layers,
                tagged=tagged,
                pool=pool,
            ) -> None:
                for layer in range(layers):
                    plan.decode(
                        sd.q[layer].view(FP8),
                        sd.k[rank].view(FP8),
                        sd.block_table,
                        sd.seq_lens,
                        sd.nvalid,
                        scores,
                        outs[layer],
                        comm=comm,
                        slot=ref.slot_set(layer, layers),
                        tagged=tagged,
                        decode_query_len=dql,
                        pool=pool,
                    )

            graph = torch.cuda.CUDAGraph()
            for s in range(steps):
                sd.load(s)
                if s < 2:
                    step()
                elif s == 2:
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        step()
                    graph.replay()
                else:
                    graph.replay()
                for layer in range(layers):
                    hist[s, layer].copy_(outs[layer], non_blocking=True)
            torch.cuda.synchronize()
            got = hist.cpu()
            for s in range(steps):
                for layer in range(layers):
                    _, _, owner_outs, _ = decode_expect(cases[s][layer], lay)
                    assert_words(
                        got[s, layer],
                        owner_outs[rank],
                        f"rank {rank} L{layers} tagged={tagged} fused={pool is not None} step {s} layer {layer}",
                    )
                    report["checks"] += 1
            assert int(comm.errors.item()) == 0
            by_rule = rule is not None and batch >= rule
            assert (fused_calls[0] > 0) == (fused or by_rule), (
                layers,
                tagged,
                fused,
                batch,
                fused_calls[0],
            )
            report["rule_fused_configs"] = report.get("rule_fused_configs", 0) + int(
                by_rule and not fused
            )
            dist.barrier()
            if not tagged:
                assert bool((comm.recv[..., 2 * TOPK] == 0).all().item()), (
                    "ready flags left set"
                )
            del graph
        # legacy ready-flag litmus: back-to-back publish / merge rounds with new payloads, three slot sets
        rounds, t, k = 64, cap, 96
        rng = ref.SplitMix64(2024)
        staged = []
        for _ in range(rounds):
            sc = [
                torch.tensor(
                    [
                        [
                            [
                                ref.f32_bits(float(rng.randint(-50, 50)))
                                for _ in range(k)
                            ]
                            for _ in range(H)
                        ]
                        for _ in range(t)
                    ],
                    dtype=torch.int64,
                )
                for _ in range(world)
            ]
            staged.append((sc, [rng.randint(0, k) for _ in range(t)]))
        st_scores = (
            _i32_bits(torch.stack([sc[rank] for sc, _ in staged]))
            .to(DEV)
            .view(torch.float32)
        )
        st_nv = torch.tensor([nv for _, nv in staged], dtype=torch.int32, device=DEV)
        lit_out = torch.full(
            (rounds, t, hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV
        )
        errors = torch.zeros(1, dtype=torch.int32, device=DEV)
        dist.barrier()
        for i in range(rounds):
            slot = i % 3
            msa_ops.msa_icp_topk_select_publish(
                st_scores[i],
                st_nv[i],
                bases,
                errors,
                slot=slot,
                rank=rank,
                world_size=world,
                heads_per_rank=hpr,
                capacity=cap,
                force_begin_blocks=0,
                force_end_blocks=1,
                enable_pdl=True,
                index_scale=lay.index_scale,
                index_offset=lay.index_offset(rank),
            )
            msa_ops.msa_icp_merge_candidates(
                comm.recv[slot],
                st_nv[i],
                lit_out[i],
                errors,
                init_blocks=0,
                local_blocks=1,
                enable_pdl=True,
            )
        torch.cuda.synchronize()
        got = lit_out.cpu()
        for i, (sc, nv) in enumerate(staged):
            rows = [
                ref.select_rows(
                    sc[r].reshape(t * H, k),
                    torch.tensor(nv).repeat_interleave(H),
                    0,
                    1,
                    lay.index_scale,
                    lay.index_offset(r),
                ).reshape(t, H, ref.ROW_WORDS)
                for r in range(world)
            ]
            want = ref.merge_carrier_rows(
                ref.exchange_decode_rows(rows, lay, rank), torch.tensor(nv), 0, 1, False
            )
            assert_words(got[i], want, f"litmus round {i} rank {rank}")
            report["checks"] += 1
        assert int(errors.item()) == 0
        dist.barrier()
        # prefill: per-rank chunk scoring into the carrier, one all_to_all_single, merge
        case = ref.make_prefill_case(
            "dist",
            page_size=args.page,
            q_lens=[700, 1, 0, 90],
            prefixes=[0, 5000, 3, 260],
            padding_tokens=6,
            chunk=256,
            seed=777,
        )
        exp = prefill_expect(case, lay, fast=False, with_global=False)
        windows = case.windows()
        plan = msa_ops.prepare_msa_icp_indexer(
            world_size=world,
            rank=rank,
            num_index_heads=H,
            page_size=args.page,
            init_blocks=case.init_blocks,
            local_blocks=case.local_blocks,
            max_decode_tokens=cap,
            max_decode_query_len=max_dql,
            max_num_batched_tokens=8192 * len(windows),
            max_num_seqs=case.batch,
            max_model_len=16384,
            num_sparse_layers=2,
            device=DEV,
        )
        inp = prefill_case_inputs(case, lay)
        send = torch.full(
            (H, case.total_q, ref.ROW_WORDS), POISON_I32, dtype=torch.int32, device=DEV
        )
        recv = torch.empty_like(send)
        rows_, _, pitch = plan.prefill_scratch_shape(max(case.seq_lens))
        scratch = _poisoned_f32((rows_, H, pitch))
        wplans = [
            plan.prefill_window_plan(
                wi,
                *_window_meta(w),
                total_q=w.total_q,
                max_query_len=max(max(w.q_lens), 1),
                max_seq_len=max(max(w.kv_lens), 1),
            )
            for wi, w in enumerate(windows)
        ]
        for wi, w in enumerate(windows):
            plan.prefill_local(
                inp["q"][w.start : w.end],
                inp["shards"][rank],
                inp["block_table"],
                wplans[wi],
                scratch,
                inp["nvalid"][w.start : w.end],
                send[:, w.start : w.end],
            )
        dist.all_to_all_single(recv, send)
        out = torch.full(
            (case.total_q, hpr, TOPK), POISON_I32, dtype=torch.int32, device=DEV
        )
        plan.prefill_merge(
            recv.view(world, hpr, case.total_q, ref.ROW_WORDS), inp["nvalid"], out
        )
        torch.cuda.synchronize()
        assert_words(send, exp.carriers[rank], f"rank {rank} prefill carrier")
        assert_words(out, exp.outputs[rank], f"rank {rank} prefill merged ids")
        report["checks"] += 2
        report["ok"] = True
    except BaseException as e:  # noqa: BLE001 - reported to the launching test
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"[:2000]
        raise
    finally:
        Path(args.out, f"rank{rank}.json").write_text(json.dumps(report))
        dist.destroy_process_group()
    return 0


def _dist_first_exchange(
    mode: str, rank: int, world: int, cap: int, comm, report: dict, dist
) -> None:
    """First-ever exchange of this rank after ``warmup()`` under lazy module
    loading: its merge is enqueued first on a side stream and spins on the
    peers' rows while this rank's publisher launches for the first time."""
    report["lazy"] = os.environ.get("CUDA_MODULE_LOADING")
    cake_backend.warmup(DEV)
    ex = FirstExchange.build(world, mode == "tagged")
    assert ex.capacity <= cap, (ex.capacity, cap)
    recv = comm.tagged_recv if ex.tagged else comm.recv
    out = ex.new_output()
    merges = torch.cuda.Stream()
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.stream(merges):
        ex.merge(
            recv[FIRST_EXCHANGE_SLOT],
            comm.expected_generations[FIRST_EXCHANGE_SLOT],
            out,
            comm.errors,
        )
    ex.publish(
        comm.tagged_recv_bases if ex.tagged else comm.recv_bases,
        comm.publish_generations,
        comm.errors,
        rank,
    )
    torch.cuda.synchronize()
    assert int(comm.errors.item()) == 0, f"rank {rank}: protocol errors"
    assert_words(out.cpu(), ex.expected(rank), f"rank {rank} first {mode} exchange")
    report["checks"] += 1
    dist.barrier()


@pytest.mark.parametrize("mode", ("legacy", "tagged"))
def test_distributed_first_exchange_after_warmup(mode: str) -> None:
    """``torchrun`` W=2 under ``CUDA_MODULE_LOADING=LAZY``: each rank's first-ever
    publish launches while its merge already spins on the symmetric-memory rows."""
    world = 2
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _torchrun(
        world,
        ["--page", "128", "--first-exchange", mode],
        env=dict(os.environ, CUDA_MODULE_LOADING="LAZY"),
    )


def _torchrun(
    world: int, args: list[str], *, env: dict[str, str] | None = None
) -> None:
    """Run :func:`_dist_main` on ``world`` local ranks; every rank must report ok."""
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as out:
        env = dict(os.environ if env is None else env)
        env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")])
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world}",
            "--standalone",
            str(Path(__file__).resolve()),
            *args,
            "--out",
            out,
        ]
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


@pytest.mark.parametrize("page", (128, 256))
@pytest.mark.parametrize("world", (2, 4))
def test_distributed_decode_and_prefill(world: int, page: int) -> None:
    """``torchrun`` with ``world`` ranks: facade decode chains over symmetric
    memory (legacy and tagged, 1 / 2 / 3 / 5 layers, eager then CUDA Graph
    replays with no inter-step host sync; the fused chain with a caller pool
    and, at 8 requests, by the facade's chain rule), a 64-round legacy ready-flag litmus
    and a prefill exchange, every rank checked against the oracle."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    _torchrun(world, ["--page", str(page)])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--timeout-child"]:
        raise SystemExit(_timeout_child(sys.argv[2]))
    if sys.argv[1:2] == ["--first-exchange-child"]:
        raise SystemExit(_first_exchange_child(sys.argv[2]))
    raise SystemExit(_dist_main(sys.argv[1:]))
