"""Host-side contracts for repeated MegaMoE launches.

This module intentionally has no CUDA, torch, or cuTe dependency.  It models
the small pieces of the production protocol that must remain exact across
kernel launches: the four-call NVLink signal cycle and padded receive-pool
layout.  Pytest can therefore check the protocol on login/CI hosts, while GPU
integration tests compare device snapshots with the same model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..common.token_segments import merge_token_segments


NVLINK_BARRIER_CYCLE = 4
NVLINK_BARRIERS_INTRANODE_ONLY = 3
NVLINK_BARRIERS_LSA_RELAY = 4
NVLINK_BARRIERS_GIN = 5


@dataclass(frozen=True)
class NvlinkBarrierState:
    """Host-visible state of the two-slot sense-reversing barrier."""

    counter: int = 0
    signal_slot_0: int = 0
    signal_slot_1: int = 0

    @property
    def signals(self) -> tuple[int, int]:
        return (self.signal_slot_0, self.signal_slot_1)


def nvlink_calls_per_launch(
    *,
    enable_nccl_gin: bool,
    enable_lsa_relay: bool,
    ncu_padding: bool = False,
) -> int:
    """Return the production NVLink-barrier call count for one launch."""

    if enable_nccl_gin:
        calls = NVLINK_BARRIERS_GIN
    elif enable_lsa_relay:
        calls = NVLINK_BARRIERS_LSA_RELAY
    else:
        calls = NVLINK_BARRIERS_INTRANODE_ONLY
    if ncu_padding:
        calls += (-calls) % NVLINK_BARRIER_CYCLE
    return calls


def advance_nvlink_barrier(
    state: NvlinkBarrierState,
    *,
    ranks_per_node: int,
    calls: int = 1,
) -> NvlinkBarrierState:
    """Advance the production ping-pong barrier by ``calls`` invocations.

    Each rank signals every rank in its logical NVLink domain.  Consequently,
    one invocation changes the selected local signal slot by
    ``ranks_per_node`` and the target alternates between that value and zero.
    The device counter is monotonic; only its low two bits select slot/sign.
    """

    if ranks_per_node <= 0:
        raise ValueError("ranks_per_node must be positive")
    if calls < 0:
        raise ValueError("calls must be non-negative")

    counter = int(state.counter)
    signals = [int(state.signal_slot_0), int(state.signal_slot_1)]
    for _ in range(calls):
        status = counter & (NVLINK_BARRIER_CYCLE - 1)
        slot = status & 1
        descending = status >> 1
        delta = -ranks_per_node if descending else ranks_per_node
        target = 0 if descending else ranks_per_node
        signals[slot] += delta
        if signals[slot] != target:
            raise AssertionError(
                "invalid barrier starting state: "
                f"status={status} slot={slot} observed={signals[slot]} "
                f"target={target}"
            )
        counter += 1
    return NvlinkBarrierState(counter, signals[0], signals[1])


def expected_nvlink_state(
    *,
    launches: int,
    calls_per_launch: int,
    ranks_per_node: int,
) -> NvlinkBarrierState:
    """Return the exact terminal state after repeated launches."""

    if launches < 0:
        raise ValueError("launches must be non-negative")
    return advance_nvlink_barrier(
        NvlinkBarrierState(),
        ranks_per_node=ranks_per_node,
        calls=launches * calls_per_launch,
    )


def round_up(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class ExpertPoolLayout:
    """Padded DATA/SF layout reconstructed from per-expert receive counts."""

    expert_pool_base: tuple[int, ...]
    expert_sf_pool_base: tuple[int, ...]
    relay_pool_base: tuple[tuple[int, ...], ...]
    relay_sf_pool_base: tuple[tuple[int, ...], ...]
    data_end: int
    sf_end: int
    merged_experts: tuple[bool, ...]
    tile_counts: tuple[int, ...]


def build_expert_pool_layout(
    intra_counts: Sequence[int],
    relay_counts: Sequence[Sequence[int]],
    *,
    token_padding_block: int,
    sf_padding_block: int,
    merge_segments: bool = False,
) -> ExpertPoolLayout:
    """Mirror ``inter_meta_exchange`` Step 5c with unbounded host integers."""

    if len(intra_counts) != len(relay_counts):
        raise ValueError("intra_counts and relay_counts must have equal length")

    remote_slots = len(relay_counts[0]) if relay_counts else 0
    if any(len(row) != remote_slots for row in relay_counts):
        raise ValueError("relay_counts rows must have equal length")

    expert_pool_base: list[int] = []
    expert_sf_pool_base: list[int] = []
    relay_pool_base: list[tuple[int, ...]] = []
    relay_sf_pool_base: list[tuple[int, ...]] = []
    data_cursor = 0
    sf_cursor = 0
    merged_experts: list[bool] = []
    tile_counts: list[int] = []

    for intra_raw, relay_row_raw in zip(intra_counts, relay_counts):
        intra = int(intra_raw)
        relay_row = tuple(int(value) for value in relay_row_raw)
        if intra < 0 or any(value < 0 for value in relay_row):
            raise ValueError("receive counts must be non-negative")

        expert_pool_base.append(data_cursor)
        expert_sf_pool_base.append(sf_cursor)
        intra_padded = round_up(intra, token_padding_block)
        relay_total = sum(relay_row)
        relay_padded = round_up(relay_total, token_padding_block)
        intra_sf_padded = round_up(intra, sf_padding_block)
        relay_sf_padded = round_up(relay_total, sf_padding_block)
        merged = merge_segments and merge_token_segments(
            intra, relay_total, token_padding_block,
        )
        merged_experts.append(merged)
        data_width = intra_padded + relay_padded
        sf_width = intra_sf_padded + relay_sf_padded
        if merged:
            data_width = round_up(intra + relay_total, token_padding_block)
            sf_width = round_up(intra + relay_total, sf_padding_block)
            intra_padded = intra
            intra_sf_padded = intra
        tile_counts.append(data_width // token_padding_block)

        data_prefix = 0
        sf_prefix = 0
        data_bases: list[int] = []
        sf_bases: list[int] = []
        for count in relay_row:
            data_bases.append(data_cursor + intra_padded + data_prefix)
            sf_bases.append(sf_cursor + intra_sf_padded + sf_prefix)
            data_prefix += count
            sf_prefix += count
        relay_pool_base.append(tuple(data_bases))
        relay_sf_pool_base.append(tuple(sf_bases))
        data_cursor += data_width
        sf_cursor += sf_width

    return ExpertPoolLayout(
        expert_pool_base=tuple(expert_pool_base),
        expert_sf_pool_base=tuple(expert_sf_pool_base),
        relay_pool_base=tuple(relay_pool_base),
        relay_sf_pool_base=tuple(relay_sf_pool_base),
        data_end=data_cursor,
        sf_end=sf_cursor,
        merged_experts=tuple(merged_experts),
        tile_counts=tuple(tile_counts),
    )


def pool_row_slots(
    layout: ExpertPoolLayout,
    intra_counts: Sequence[int],
    relay_counts: Sequence[Sequence[int]],
    *,
    tile: int,
):
    """Yield (expert, source, row, DATA slot, SF slot, counter slot).

    ``source=-1`` denotes intranode rows; other sources index remote-node
    buckets. Counter slots are derived from physical rows, independent of
    whether a tile receives one or two kinds of writer.
    """
    tile_prefix = 0
    for expert, (intra, relay) in enumerate(zip(intra_counts, relay_counts)):
        data_base = layout.expert_pool_base[expert]
        sf_base = layout.expert_sf_pool_base[expert]
        segments = [(-1, int(intra), data_base, sf_base)]
        segments.extend(
            (source, int(count), layout.relay_pool_base[expert][source],
             layout.relay_sf_pool_base[expert][source])
            for source, count in enumerate(relay)
        )
        for source, count, data_start, sf_start in segments:
            for row in range(count):
                slot = data_start + row
                yield (expert, source, row, slot, sf_start + row,
                       tile_prefix + (slot - data_base) // tile)
        tile_prefix += layout.tile_counts[expert]


def packed_recv_sum(
    token_counts: Iterable[int],
    *,
    sm_count: int,
) -> int:
    """Model one clean launch's packed ``expert_recv_count_sum`` value."""

    counts = tuple(int(value) for value in token_counts)
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if any(value < 0 or value > 0xFFFFFFFF for value in counts):
        raise ValueError("token counts must fit uint32")
    low = sum(counts)
    if low > 0xFFFFFFFF:
        raise OverflowError("summed token count does not fit uint32")
    high = sm_count * len(counts)
    if high > 0xFFFFFFFF:
        raise OverflowError("summed publisher count does not fit uint32")
    return (high << 32) | low


def direct_relay_row_slots(
    layout: ExpertPoolLayout,
    relay_counts: Sequence[Sequence[int]],
    requests: Iterable[tuple[int, int, object]],
    *,
    tile: int,
) -> tuple[tuple[int, int, object, int, int, int], ...]:
    """Model owner-side row reservations in arbitrary relay-writer order.

    A request is (expert, source-node slot, unique hit key). Return the same
    provenance followed by DATA slot, SF slot, and static tile counter slot.
    The request stream must cover every relay hit exactly once; no source or
    relay-rank boundary is padded. This is a host oracle, not a readiness poll.
    """
    if tile <= 0 or len(relay_counts) != len(layout.relay_pool_base):
        raise ValueError("invalid tile or expert count")
    cursor = [[0] * len(row) for row in relay_counts]
    seen = set()
    result = []
    for expert, source, key in requests:
        if not 0 <= expert < len(cursor) or not 0 <= source < len(cursor[expert]):
            raise ValueError("request bucket out of bounds")
        identity = (expert, source, key)
        if identity in seen:
            raise ValueError("duplicate relay hit")
        seen.add(identity)
        row = cursor[expert][source]
        if row >= relay_counts[expert][source]:
            raise ValueError("relay source bucket overflow")
        data_slot = layout.relay_pool_base[expert][source] + row
        sf_slot = layout.relay_sf_pool_base[expert][source] + row
        result.append((expert, source, key, data_slot, sf_slot, data_slot // tile))
        cursor[expert][source] += 1
    if cursor != [list(row) for row in relay_counts]:
        raise ValueError("missing relay hits")
    return tuple(result)
