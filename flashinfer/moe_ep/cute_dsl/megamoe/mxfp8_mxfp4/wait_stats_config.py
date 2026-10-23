# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side layout of the wait accounting (no CuTe DSL import).

See :mod:`.wait_stats` for the device helpers.  The workspace planner uses
this module to size the mega ``wait_stats`` region, so it must stay free of
CUTLASS imports.
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import Optional

WAIT_STATS_ENV = "FLASHINFER_MXFP8_MXFP4_WAIT_STATS"
# Upper bound on persistent CTAs covered by the stats area (GB200 has 152).
MAX_CTAS = 256


class Slot(IntEnum):
    """Per-CTA int32 accumulators, in nanoseconds unless named ``*_COUNT``."""

    TMA_A_EMPTY_NS = 0
    TMA_B_EMPTY_NS = 1
    TMA_B_FC2_READY_NS = 2
    TMA_B_FC2_READY_COUNT = 3
    MMA_OPERAND_FULL_NS = 4
    MMA_ACC_EMPTY_NS = 5
    EPI_ACC_FULL_NS = 6
    EPI_FC1_TILE_NS = 7
    EPI_FC1_TILE_COUNT = 8
    EPI_FC2_TILE_NS = 9
    EPI_FC2_TILE_COUNT = 10
    MMA_LOOP_NS = 11
    EPI_LOOP_NS = 12
    TMA_B_LOOP_NS = 13
    TMA_B_FIRST_FC2_READY_NS = 14
    TMA_B_FIRST_FC2_START_NS = 15
    # Mega only: FC1 waits on dispatched-token arrival, the scheduler's wait
    # for dispatch before scheduling, and the all-warp tail rendezvous.
    TMA_B_FC1_READY_NS = 16
    TMA_B_FC1_READY_COUNT = 17
    SCHED_DISPATCH_WAIT_NS = 18
    KERNEL_TAIL_NS = 19
    # Scheduler hand-off and clock: time consumers block in consume_work(),
    # the scheduler warp's publish (blocked on consumers) vs generation time,
    # and SM clock cycles over the MMA loop (cycles / MMA_LOOP_NS = GHz).
    MMA_CONSUME_NS = 20
    TMA_A_CONSUME_NS = 21
    TMA_B_CONSUME_NS = 22
    SCHED_PUBLISH_NS = 23
    SCHED_GENERATE_NS = 24
    MMA_LOOP_CYCLES = 25
    # Start-up: the scheduler's first work generation (mega: waits for the
    # dispatched expert counts) and the MMA warp's first hand-off.
    SCHED_FIRST_GENERATE_NS = 26
    MMA_FIRST_CONSUME_NS = 27
    # Start-up timeline, each relative to that warp's own read right after
    # the kernel's pipeline-init barrier (all warps pass it together).
    SCHED_INIT_DONE_AT_NS = 28
    SCHED_FIRST_PUBLISH_AT_NS = 29
    MMA_FIRST_WORK_AT_NS = 30
    SCHED_WAIT_DONE_AT_NS = 31
    # internal_init split: a timed pass loading every local expert count,
    # then the first claim's decode (group construction and readiness peek).
    SCHED_INIT_COUNT_LOADS_NS = 32
    SCHED_INIT_DECODE_NS = 33
    # Raw %globaltimer_lo values (host takes the difference of per-CTA means):
    # just after the dispatch barrier, and at the start of the count loads.
    SCHED_WAIT_DONE_RAW = 34
    SCHED_LOADS_START_RAW = 35
    # start-up split on the dispatch warps (mega only, recorded by
    # local dispatch warp 0 of each CTA): count/slot prep, the first grid
    # sync plus CTA 0's per-expert count sends to the owner ranks, and the
    # cross-rank NVLink barrier including the closing grid sync.
    DISPATCH_PREP_NS = 36
    DISPATCH_SYNC_SEND_NS = 37
    DISPATCH_NVLINK_BARRIER_NS = 38
    SCHED_LATE_CLAIM_NS = 39
    # Retired with dispatch_dedup; kept so later slot numbers stay stable.
    DISPATCH_FETCH_COUNT = 40
    TAIL_DRAIN_BARRIER_NS = 41
    TAIL_SHARED_RESET_NS = 42
    TAIL_PUBLISH_BARRIER_NS = 43
    TAIL_LOCAL_RESET_NS = 44
    # Unlike KERNEL_TAIL_NS (scheduler warp), this includes the cross-rank
    # drain even when the reduction is external and the scheduler exits early.
    # It also includes dispatch waiting for compute at the initial rendezvous;
    # the individual barrier slots isolate cleanup from that arrival skew.
    TAIL_DISPATCH_NS = 45
    # Startup subphases. Rendezvous is meaningful on CTA 0 (other CTAs
    # return immediately and wait in the closing grid synchronization).
    DISPATCH_GRID_SYNC_NS = 46
    DISPATCH_COUNT_SEND_NS = 47
    DISPATCH_RENDEZVOUS_NS = 48
    DISPATCH_COUNT_SUM_NS = 49
    # Retired with the start-up prefetch; kept so slot numbers stay stable.
    TMA_A_STARTUP_PREFETCH_NS = 50
    # start-up split of the NVLink barrier (mega only). CTA 0 alone:
    # dispatch-prep start to its cross-rank signal send (prep, first grid
    # sync, count sends), and the spin from the send until every rank has
    # signalled (arrival skew plus one NVLink round trip). Every CTA: from
    # the end of the signal exchange to the end of the dispatch barrier (CTA
    # 0's owner-side count publication and the closing grid sync; on other
    # CTAs this includes waiting for CTA 0's signal).
    STARTUP_LOCAL_ARRIVAL_AT_NS = 51
    STARTUP_SIGNAL_WAIT_NS = 52
    STARTUP_GRID_SYNC_NS = 53
    # Last launch snapshot, overwritten before the first grid rendezvous.
    STARTUP_PHASE_RAW = 54
    MMA_OPERAND_FULL_FC2_NS = 55


SLOTS = 56
STATS_WORDS = MAX_CTAS * SLOTS
# Mega only: a per-launch ring after the per-CTA block. Word 0 counts
# launches; entry i (words 1 + 3 i ...) holds (launch number + 1, local
# arrival ns, signal wait ns) of CTA 0, launch number modulo the ring size.
STARTUP_RING_ENTRIES = 512
STARTUP_RING_WORDS = 1 + 3 * STARTUP_RING_ENTRIES
MEGA_STATS_WORDS = STATS_WORDS + STARTUP_RING_WORDS


def wait_stats_enabled() -> bool:
    return os.environ.get(WAIT_STATS_ENV, "0").strip() == "1"


def mega_stats_word_offset(plan) -> Optional[int]:
    """Word offset of the ``wait_stats`` region from ``fc1_done_counter``."""

    try:
        stats = plan.local.region("wait_stats")
    except KeyError:
        return None
    done = plan.local.region("fc1_done_counter")
    return (stats.offset - done.offset) // 4


def read_mega_wait_stats(workspace):
    """The mega per-CTA ``wait_stats`` block (``STATS_WORDS``) as an int32 view."""

    region = workspace.plan.local.region("wait_stats")
    import torch

    words = workspace.workspace_bytes[region.offset : region.end].view(torch.int32)
    return words[:STATS_WORDS]


def read_mega_startup_ring(workspace):
    """The mega per-launch start-up ring (``STARTUP_RING_WORDS``) as an int32 view."""

    region = workspace.plan.local.region("wait_stats")
    import torch

    words = workspace.workspace_bytes[region.offset : region.end].view(torch.int32)
    return words[STATS_WORDS:MEGA_STATS_WORDS]


def decode_startup_ring(ring) -> list[dict]:
    """Recorded ring entries in launch order: launch, arrival and wait in µs."""

    entries = ring[1:].reshape(STARTUP_RING_ENTRIES, 3).tolist()
    rows = [
        {
            "launch": launch - 1,
            "local_arrival_us": round(arrival / 1000, 2),
            "signal_wait_us": round(wait / 1000, 2),
        }
        for launch, arrival, wait in entries
        if launch > 0
    ]
    return sorted(rows, key=lambda row: row["launch"])


def summarize(stats, num_ctas: int) -> dict:
    """Host-side mean/max per slot (µs for times) over the first CTAs."""

    rows = stats.reshape(MAX_CTAS, SLOTS)[:num_ctas].double()
    summary = {}
    for slot in Slot:
        column = rows[:, slot.value]
        if slot.name.endswith(("_COUNT", "_CYCLES", "_RAW")):
            summary[slot.name.lower()] = {
                "mean": round(float(column.mean()), 2),
                "max": int(column.max()),
            }
        else:
            summary[slot.name.lower().removesuffix("_ns") + "_us"] = {
                "mean": round(float(column.mean()) / 1000, 2),
                "max": round(float(column.max()) / 1000, 2),
            }
    return summary


__all__ = [
    "MAX_CTAS",
    "MEGA_STATS_WORDS",
    "SLOTS",
    "STARTUP_RING_ENTRIES",
    "STARTUP_RING_WORDS",
    "STATS_WORDS",
    "Slot",
    "WAIT_STATS_ENV",
    "mega_stats_word_offset",
    "decode_startup_ring",
    "read_mega_startup_ring",
    "read_mega_wait_stats",
    "summarize",
    "wait_stats_enabled",
]
