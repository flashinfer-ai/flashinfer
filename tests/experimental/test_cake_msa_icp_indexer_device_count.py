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

GPU tests of the device-count programs (the ``_dc`` roles): a call with
``token_count`` (int32 ``[1]``, read on the device) over a grid of ``T`` rows
writes exactly what the by-value call over the first ``token_count[0]`` rows
writes, and nothing past them.

Single device, every layout, counts 0, 1 (one decode query), about half and
all of ``T``:

* prefill: the selector's carrier rows (oracle score rows as input), then the
  emulated exchange and the prefill merge;
* decode, legacy and tagged rows: the selector-publisher's receive storage,
  publish counters and error words, then every owner's merge (outputs, merge
  counters);
* the fused decode chain's publisher (the scorer runs every request).
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.cake_msa_icp_indexer import cake_backend as cb
from tests.experimental import test_cake_msa_icp_indexer as base
from tests.experimental.test_helpers import cake_msa_icp_inputs as ref

H = ref.NUM_HEADS
TOPK = ref.TOPK
LAYOUTS = base.LAYOUTS
LAYOUT_IDS = base.LAYOUT_IDS
POISON_I32 = base.POISON_I32

pytestmark = base.pytestmark
DEV = base.DEV


def _count(n: int) -> torch.Tensor:
    return torch.tensor([n], dtype=torch.int32, device=DEV)


def _poison(*shape: int) -> torch.Tensor:
    return torch.full(shape, POISON_I32, dtype=torch.int32, device=DEV)


def _counts(t: int, step: int = 1) -> list[int]:
    half = (t // 2) // step * step
    return sorted({0, step, half, t // step * step})


def _same(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    base.assert_words(got.cpu(), want.cpu(), what)


def _untouched(t: torch.Tensor, what: str) -> None:
    assert bool((t == POISON_I32).all()), f"{what}: written past the token count"


def _prefill_case(lay: ref.IcpLayout) -> ref.PrefillCase:
    return ref.make_prefill_case(
        f"dc_p{lay.page_size}",
        page_size=lay.page_size,
        q_lens=[40, 9, 0],
        prefixes=[700, 130, 5],
        padding_tokens=3,
        chunk=64,
        init_blocks=1,
        local_blocks=1,
        seed=8170,
    )


@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_prefill_select_and_merge_device_count(lay: ref.IcpLayout) -> None:
    case = _prefill_case(lay)
    (window,) = case.windows()
    t, hpr = case.total_q, H // lay.tp
    base._warm()
    scores = [
        ref.prefill_window_scores(case, window, lay, r)[0].to(DEV)
        for r in range(lay.tp)
    ]
    nv = base._i32(case.nvalid())
    errors = torch.zeros(1, dtype=torch.int32, device=DEV)
    for c in _counts(t):
        what = f"{lay.name} count {c}/{t}"
        carriers = {"a": [_poison(H, t, ref.ROW_WORDS) for _ in range(lay.tp)]}
        carriers["b"] = [x.clone() for x in carriers["a"]]
        kw = dict(
            force_begin_blocks=case.init_blocks,
            force_end_blocks=case.local_blocks,
            index_scale=lay.index_scale,
        )
        for r in range(lay.tp):
            cb.topk_select_prefill(
                scores[r][:c], nv[:c], carriers["a"][r][:, :c],
                index_offset=lay.index_offset(r), **kw,
            )  # fmt: skip
            cb.topk_select_prefill(
                scores[r], nv, carriers["b"][r], index_offset=lay.index_offset(r),
                token_count=_count(c), **kw,
            )  # fmt: skip
        torch.cuda.synchronize()
        for r in range(lay.tp):
            _same(
                carriers["b"][r][:, :c],
                carriers["a"][r][:, :c],
                f"{what} rank {r} carrier",
            )
            _untouched(carriers["b"][r][:, c:], f"{what} rank {r} carrier")
        outs = {side: [_poison(t, hpr, TOPK) for _ in range(lay.tp)] for side in "ab"}
        for side in "ab":
            for o in range(lay.tp):
                recv = torch.stack(
                    [carriers[side][r][o * hpr : (o + 1) * hpr] for r in range(lay.tp)]
                )
                if side == "a":
                    cb.merge_candidates(
                        recv, nv[:c], outs[side][o][:c], errors,
                        init_blocks=case.init_blocks, local_blocks=case.local_blocks, prefill=True,
                    )  # fmt: skip
                else:
                    cb.merge_candidates(
                        recv, nv, outs[side][o], errors,
                        init_blocks=case.init_blocks, local_blocks=case.local_blocks, prefill=True,
                        token_count=_count(c),
                    )  # fmt: skip
        torch.cuda.synchronize()
        for o in range(lay.tp):
            _same(outs["b"][o][:c], outs["a"][o][:c], f"{what} owner {o} merged ids")
            _untouched(outs["b"][o][c:], f"{what} owner {o} merged ids")
        if c == t:
            exp = base.prefill_expect(case, lay, fast=True, with_global=False)
            for r in range(lay.tp):
                _same(
                    carriers["b"][r], exp.carriers[r], f"{what} rank {r} oracle carrier"
                )
                _same(outs["b"][r], exp.outputs[r], f"{what} owner {r} oracle ids")
    assert int(errors.item()) == 0


def _comm_state(comm: base.EmulatedComm) -> list[torch.Tensor]:
    return [*comm.storage, *comm.generations, *comm.expected, *comm.errors]


@pytest.mark.parametrize("tagged", (False, True), ids=("legacy", "tagged"))
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_decode_publish_and_merge_device_count(
    lay: ref.IcpLayout, tagged: bool
) -> None:
    case, extra = base._make_decode(lay.page_size, "mixed_lengths")
    inp = base.DecodeInputs.build(case, lay)
    t, hpr = case.tokens, H // lay.tp
    base._warm()
    planes = base.new_planes(inp)
    base.decode_scores(
        inp,
        planes,
        max_dql=extra.get("max_dql", case.decode_query_len),
        enable_pdl=False,
    )
    for c in _counts(t, case.decode_query_len):
        what = f"{lay.name} tagged={tagged} count {c}/{t}"
        comms = {side: base.EmulatedComm(lay.tp, t) for side in "ab"}
        for side, comm in comms.items():
            for r in range(lay.tp):
                n = c if side == "a" else t
                cb.topk_select_publish(
                    planes[r][:n], inp.nvalid[:n], comm.bases(tagged), comm.errors[r],
                    slot=1, rank=r, world_size=lay.tp, heads_per_rank=hpr, capacity=t,
                    force_begin_blocks=case.init_blocks, force_end_blocks=case.local_blocks,
                    generations=comm.generations[r] if tagged else None,
                    index_scale=lay.index_scale, index_offset=lay.index_offset(r),
                    token_count=None if side == "a" else _count(c),
                )  # fmt: skip
        torch.cuda.synchronize()
        for i, (g, w) in enumerate(
            zip(_comm_state(comms["b"]), _comm_state(comms["a"]), strict=True)
        ):
            _same(g, w, f"{what} published state {i}")
        outs = {side: [_poison(t, hpr, TOPK) for _ in range(lay.tp)] for side in "ab"}
        for side, comm in comms.items():
            for o in range(lay.tp):
                n = c if side == "a" else t
                cb.merge_candidates(
                    comm.recv_slot(o, 1, tagged), inp.nvalid[:n], outs[side][o][:n], comm.errors[o],
                    init_blocks=case.init_blocks, local_blocks=case.local_blocks,
                    expected_generations=comm.expected[o][1] if tagged else None,
                    token_count=None if side == "a" else _count(c),
                )  # fmt: skip
        torch.cuda.synchronize()
        for o in range(lay.tp):
            _same(outs["b"][o][:c], outs["a"][o][:c], f"{what} owner {o} merged ids")
            _untouched(outs["b"][o][c:], f"{what} owner {o} merged ids")
        for i, (g, w) in enumerate(
            zip(_comm_state(comms["b"]), _comm_state(comms["a"]), strict=True)
        ):
            _same(g, w, f"{what} merged state {i}")
        assert all(int(e.item()) == 0 for e in comms["b"].errors), what
        if c == t:
            _, _, owner_outs, _ = base.decode_expect(case, lay)
            for o in range(lay.tp):
                _same(outs["b"][o], owner_outs[o], f"{what} owner {o} oracle ids")


@pytest.mark.parametrize("tagged", (False, True), ids=("legacy", "tagged"))
@pytest.mark.parametrize("lay", LAYOUTS, ids=LAYOUT_IDS)
def test_fused_decode_publisher_device_count(lay: ref.IcpLayout, tagged: bool) -> None:
    case, _ = base._make_decode(lay.page_size, "mixed_lengths")
    inp = base.DecodeInputs.build(case, lay)
    t, dql = case.tokens, case.decode_query_len
    base._warm()
    for c in _counts(t, dql):
        what = f"{lay.name} tagged={tagged} count {c}/{t}"
        comms = {side: base.EmulatedComm(lay.tp, t) for side in "ab"}
        for side, comm in comms.items():
            b = c // dql if side == "a" else case.batch
            for r in range(lay.tp):
                pool = torch.empty(
                    cb.decode_pool_bytes(max_decode_tokens=t, num_heads=H),
                    dtype=torch.uint8,
                    device=DEV,
                )
                cb.decode_score_publish(
                    inp.q[: b * dql], inp.shards[r], inp.block_table[:b], inp.seq_lens[:b],
                    inp.nvalid[: b * dql], pool, comm.bases(tagged), comm.errors[r],
                    rank=r, world_size=lay.tp, page_size=lay.page_size, decode_query_len=dql,
                    slot=2, capacity=t, force_begin_blocks=case.init_blocks,
                    force_end_blocks=case.local_blocks,
                    generations=comm.generations[r] if tagged else None, enable_pdl=False,
                    token_count=None if side == "a" else _count(c),
                )  # fmt: skip
        torch.cuda.synchronize()
        for i, (g, w) in enumerate(
            zip(_comm_state(comms["b"]), _comm_state(comms["a"]), strict=True)
        ):
            _same(g, w, f"{what} published state {i}")


def test_token_count_is_checked() -> None:
    lay = ref.IcpLayout(128, 2)
    case = _prefill_case(lay)
    (window,) = case.windows()
    t = case.total_q
    scores = ref.prefill_window_scores(case, window, lay, 0)[0].to(DEV)
    nv = base._i32(case.nvalid())
    carrier = _poison(H, t, ref.ROW_WORDS)
    for bad in (
        torch.tensor([3], dtype=torch.int64, device=DEV),
        torch.tensor([3, 4], dtype=torch.int32, device=DEV),
        torch.tensor([3], dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="token_count"):
            cb.topk_select_prefill(scores, nv, carrier, token_count=bad)
