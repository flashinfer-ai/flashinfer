# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transport-table independence of the Cake SM120 hybrid MegaMoE (multi-rank).

The transport only moves bytes: the combine sums the returned route partials
in slot order, so every feasible per-pair table (all GIN, all LSA, LSA inside
each host and GIN across, or the probe's choice) must produce bit-identical
outputs for the same inputs.  Each table gets its own layer (and transport
session), created and destroyed in turn; infeasible tables on the launched
topology (``all_lsa`` across hosts or PCIe sockets, any GIN cell without a GIN
transport) are skipped.  ``lsa_pairs`` (LSA only between ranks 2i and 2i+1, e.g.
the two GPUs behind one PCIe switch) runs the copier instances with a single
copier CTA.  Whenever the topology offers GIN and at least one load/store pair
(and more than two ranks), at least one mixed table -- LSA and GIN cells in the
same launch -- must run.  ``-rA`` prints the tables that ran.  Run in its own torchrun session:

    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_sm120_w4a8_cake_tables.py \
        -m "gpu_4 and arch_sm120" -v
"""

from __future__ import annotations

import os

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_transport import (
    TRANSPORT_GIN,
    TRANSPORT_LSA,
)
from tests.moe_ep.test_moe_ep_sm120_w4a8_cake_mega import (
    HIDDEN,
    Oracle,
    check_close,
    init_distributed,
    local_weights,
    make_inputs,
    make_layer,
)

TABLES = ("all_gin", "all_lsa", "lsa_where_reachable", "lsa_pairs", "auto")


def _table_spec(name: str, world_size: int):
    if name != "lsa_pairs":
        return name
    return tuple(
        0 if a == b else (1 if a // 2 == b // 2 else 2)
        for a in range(world_size)
        for b in range(world_size)
    )


ROWS = (40, 512, 3000)


def _run_table(rank, world_size, weights, variant, table):
    import torch.distributed as dist

    try:
        layer = make_layer(
            rank,
            world_size,
            weights,
            variant=variant,
            table=_table_spec(table, world_size),
        )
        layer.warmup(make_inputs(64, seed=1))
    except ValueError as error:
        # validate_table rejects a table the launched topology cannot serve; every
        # rank reaches the same verdict from the same gathered reachability matrix.
        dist.barrier()
        return None, str(error)
    outputs = []
    rows_case = [
        r if i != 2 else (r if rank % 2 == 0 else r // 3) for i, r in enumerate(ROWS)
    ]
    for index, rows in enumerate(rows_case):
        outputs.append(layer(make_inputs(rows, seed=1000 * index + rank)).clone())
    torch.cuda.synchronize()
    assert layer._kernel.protocol_error(layer._workspace) == 0
    ws = layer._workspace
    info = (
        ws.transport_table.name,
        ws.copiers_active,
        ws.transport_table.count(TRANSPORT_LSA),
        ws.transport_table.count(TRANSPORT_GIN),
    )
    layer.destroy()
    dist.barrier()
    return outputs, info


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
@pytest.mark.parametrize(
    "variant",
    [
        v.strip()
        for v in os.environ.get(
            "CAKE_SM120_MEGAMOE_VARIANTS", "noshared,fp8shared"
        ).split(",")
        if v.strip()
    ],
)
def test_outputs_are_bitwise_identical_across_transport_tables(variant) -> None:
    rank, world_size = init_distributed()
    shared = variant == "fp8shared"
    weights = local_weights(rank, world_size, seed=20261007, shared=shared)
    results, skipped = {}, {}
    for table in TABLES:
        outputs, info = _run_table(rank, world_size, weights, variant, table)
        if outputs is not None:
            results[table] = (outputs, info)
        else:
            skipped[table] = info
    assert len(results) >= 2, f"fewer than two feasible tables: {sorted(results)}"
    # Every table either ran (LSA-bearing ones on the copier instances) or was rejected
    # because the launched topology has no load/store path or no GIN for one of its pairs.
    infeasible = ("not load/store reachable", "GIN is unavailable")
    for name, reason in skipped.items():
        assert any(text in reason for text in infeasible), (name, reason)
    for name in ("all_lsa", "lsa_where_reachable", "lsa_pairs"):
        if name in results:
            _, copiers, lsa_cells, _ = results[name][1]
            assert copiers == (lsa_cells > 0), (name, results[name][1])
            assert name == "lsa_where_reachable" or lsa_cells > 0, (name, lsa_cells)
    mixed = sorted(n for n, (_, info) in results.items() if info[2] and info[3])
    gin_ok = "all_gin" in results
    lsa_ok = any(results[n][1][2] for n in results)
    if rank == 0:
        print(
            "tables:",
            {n: info for n, (_, info) in results.items()},
            "skipped:",
            sorted(skipped),
            "mixed:",
            mixed,
        )
    if gin_ok and lsa_ok and world_size > 2:
        assert mixed, f"no mixed LSA + GIN table ran: {sorted(results)}"
    names = sorted(results)
    baseline = results[names[0]][0]
    for name in names[1:]:
        for rows_index, (a, b) in enumerate(
            zip(baseline, results[name][0], strict=True)
        ):
            assert torch.equal(a, b), (
                f"{name} differs from {names[0]} at case {rows_index}"
            )
    # And one table against the oracle.
    oracle = Oracle(weights, shared)
    tensors = make_inputs(ROWS[1], seed=1000 + rank)
    check_close(baseline[1], oracle(tensors))
    assert all(o.shape[1] == HIDDEN for o in baseline)
