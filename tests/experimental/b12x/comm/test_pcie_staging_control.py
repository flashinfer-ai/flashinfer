from __future__ import annotations


UINT32_MASK = (1 << 32) - 1


def _advance(generation: int) -> tuple[int, int]:
    """Model the device control kernel's unsigned generation update."""

    return generation & 1, (generation + 1) & UINT32_MASK


def test_slot_control_alternates_across_replays_and_wraps() -> None:
    generation = 0
    slots = []
    for _ in range(7):
        slot, generation = _advance(generation)
        slots.append(slot)

    assert slots == [0, 1, 0, 1, 0, 1, 0]
    assert _advance(UINT32_MASK) == (1, 0)
    assert _advance(0) == (0, 1)


def test_same_stream_capture_replay_handles_variable_worker_grids() -> None:
    generation = 0
    replay_slots = []
    worker_slots = []

    for worker_blocks in (1, 64, 3, 36, 8):
        slot, generation = _advance(generation)
        replay_slots.append(slot)
        worker_slots.append([slot] * worker_blocks)

    assert replay_slots == [0, 1, 0, 1, 0]
    assert all(len(set(cta_slots)) == 1 for cta_slots in worker_slots)


def test_multistream_separate_channels_match_across_rank_interleavings() -> None:
    schedules = {
        0: ("decode", "prefill", "decode", "prefill"),
        1: ("prefill", "decode", "prefill", "decode"),
    }
    rank_observations = {}

    for rank, schedule in schedules.items():
        generations = {"decode": 0, "prefill": 0}
        observed = {"decode": [], "prefill": []}
        for channel in schedule:
            slot, generations[channel] = _advance(generations[channel])
            observed[channel].append(slot)
        rank_observations[rank] = observed

    assert rank_observations[0] == rank_observations[1]
    assert rank_observations[0] == {
        "decode": [0, 1],
        "prefill": [0, 1],
    }
