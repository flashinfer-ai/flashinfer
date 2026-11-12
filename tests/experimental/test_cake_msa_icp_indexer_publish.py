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

The prefill exchange as a peer publish (``prefill_list_publish`` and
``merge_prefill_tagged``) against the carrier exchange and the CPU oracle, on
one GPU: the ``W`` ranks of a TP2 / TP4 group run on one device, each with its
own receive rows (``prefill_publish_layout``), which form every rank's
``peer_rows``, so a publisher stores into "peer" rows as over NVLink.
Every rank publishes every window of a step into the layer's slot, then every
owner merges its slot: the merged ids equal the oracle's and the carrier path's
word for word, over the prefill scenarios of every layout, several layers that
rotate the slot sets, the generation wrap, poisoned reused rows and CUDA Graph
replays.  The multi-window step publishes each window at its token offset.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.cake_msa_icp_indexer import cake_backend as cb
from tests.experimental import test_cake_msa_icp_indexer as base
from tests.experimental.test_helpers import cake_msa_icp_inputs as ref

pytestmark = base.pytestmark
H = ref.NUM_HEADS
DEV = base.DEV
# Layers of a step: the slot sets of a layer count of 3 (0, 1, 2) and the odd
# count 5 (0, 1, 0, 1, 2), each consecutive pair different.
SLOT_ORDERS = {3: (0, 1, 2), 5: (0, 1, 0, 1, 2)}


def _stores(lay: ref.IcpLayout, capacity: int, gen0: int) -> list[dict]:
    """Per rank: poisoned receive rows, its generation and expected counters
    (every counter at ``gen0``), and the rows of every rank (``peer_rows``)."""
    layout = cb.prefill_publish_layout(lay.tp, H // lay.tp, capacity)
    g = gen0 - (1 << 32) if gen0 >= 1 << 31 else gen0
    out = [
        {
            "rows": torch.full(
                layout["rows_shape"], 0x0BADC0DE0BADC0DE, dtype=torch.int64, device=DEV
            ),
            "generations": torch.full(
                layout["generations_shape"], g, dtype=torch.int32, device=DEV
            ),
            "expected": torch.full(
                layout["expected_shape"], g, dtype=torch.int32, device=DEV
            ),
        }
        for _ in range(lay.tp)
    ]
    peers = tuple(s["rows"] for s in out)
    for s in out:
        s["peer_rows"] = peers
    return out


def _layer(case, lay, inp, plans, pool, stores, outputs, errors, slot) -> None:
    """One layer: every rank publishes every window into ``slot``, then every
    owner merges that slot."""
    for r in range(lay.tp):
        for w, plan in plans:
            cb.prefill_score_select(
                inp["q"][w.start : w.end],
                inp["shards"][r],
                inp["block_table"],
                plan,
                inp["nvalid"][w.start : w.end],
                pool,
                None,
                rank=r,
                tp=lay.tp,
                compact=lay.page_size == 256,
                force_begin_blocks=case.init_blocks,
                force_end_blocks=case.local_blocks,
                publish={
                    "peer_rows": stores[r]["peer_rows"],
                    "generations": stores[r]["generations"],
                    "slot": slot,
                    "t_base": w.start,
                    "capacity": stores[r]["rows"].shape[3],
                },
            )
    for o in range(lay.tp):
        cb.merge_candidates(
            stores[o]["rows"][slot],
            inp["nvalid"],
            outputs[o],
            errors,
            init_blocks=case.init_blocks,
            local_blocks=case.local_blocks,
            prefill=True,
            expected_generations=stores[o]["expected"][slot],
        )


def _run(
    case, lay, *, layers: int, gen0: int, graph: bool, what: str, fast: bool = False
) -> None:
    exp = base.prefill_expect(case, lay, fast=fast, with_global=not fast)
    inp = base.prefill_case_inputs(case, lay)
    windows = case.windows()
    t_max = max(w.total_q for w in windows)
    storage = base.plan_buffers(t_max, case.batch)
    if len(windows) > 1:
        # One plan storage per window: every window's plan lives through the step.
        plans = [
            (w, base.plan_window(w, base.plan_buffers(w.total_q, case.batch)))
            for w in windows
        ]
    else:
        plans = [(windows[0], base.plan_window(windows[0], storage))]
    pool, guard = base.prefill_pool(t_max, case.batch)
    stores = _stores(lay, case.total_q + 3, gen0)
    outputs = base.new_outputs(lay.tp, case.total_q)
    errors = torch.zeros(1, dtype=torch.int32, device=DEV)
    order = SLOT_ORDERS[layers]

    def step() -> None:
        for slot in order:
            _layer(case, lay, inp, plans, pool, stores, outputs, errors, slot)

    def check(tag: str) -> None:
        torch.cuda.synchronize()
        for o in range(lay.tp):
            base.assert_words(
                outputs[o], exp.outputs[o], f"{what} {tag}: owner {o} merged ids"
            )
        if exp.global_output is not None:
            base.assert_words(
                torch.cat([o.cpu() for o in outputs], dim=1),
                exp.global_output,
                f"{what} {tag}: unsharded oracle",
            )
        assert int(errors.item()) == 0, f"{what} {tag}: errors {int(errors.item())}"
        base.assert_guard(guard, f"{what} {tag}")

    step()
    check("eager")
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        for replay in range(3):
            for o in outputs:
                o.fill_(base.POISON_I32)
            g.replay()
            check(f"replay {replay}")
    # Each slot's counters of the step's token rows: gen0 advanced once per use
    # of the slot (0 skipped), on every rank and on both sides.
    for r in range(lay.tp):
        gens = stores[r]["generations"][:, :, : case.total_q]
        exp_slots = stores[r]["expected"][:, :, : case.total_q]
        for slot in range(3):
            n = order.count(slot) * (4 if graph else 1)
            want = gen0
            for _ in range(n):
                want = (want + 1) & 0xFFFFFFFF or 1
            w32 = want - (1 << 32) if want >= 1 << 31 else want
            assert bool((gens[slot] == w32).all()), (
                f"{what}: rank {r} slot {slot} generations"
            )
            assert bool((exp_slots[slot] == w32).all()), (
                f"{what}: rank {r} slot {slot} expected"
            )


@pytest.mark.parametrize("scenario", base.PREFILL_SCENARIOS)
@pytest.mark.parametrize("lay", base.LAYOUTS, ids=base.LAYOUT_IDS)
def test_publish_matches_the_oracle(lay: ref.IcpLayout, scenario: str) -> None:
    """Every prefill scenario: three layers (one per slot set), eager."""
    case, fast = base._make_prefill(lay.page_size, scenario)
    _run(
        case,
        lay,
        layers=3,
        gen0=0,
        graph=False,
        what=f"{case.label} {lay.name} publish",
        fast=fast,
    )


@pytest.mark.parametrize("lay", base.LAYOUTS, ids=base.LAYOUT_IDS)
def test_publish_replays_wrap_and_rotate(lay: ref.IcpLayout) -> None:
    """Five layers over the slot sets of an odd layer count, counters starting
    at 0xFFFFFFFE (the wrap through 0 within the first replay), then three CUDA
    Graph replays of the step with the outputs poisoned before each."""
    case = ref.make_prefill_case(
        f"publish_wrap_p{lay.page_size}",
        page_size=lay.page_size,
        q_lens=[200, 7, 0, 33],
        prefixes=[0, 900, 5, 2000],
        padding_tokens=4,
        chunk=64,
        seed=31,
    )
    _run(
        case,
        lay,
        layers=5,
        gen0=0xFFFFFFFE,
        graph=True,
        what=f"{lay.name} publish wrap",
    )
