"""The workspace's byte count, checked against an independent restatement.

The arithmetic is restated rather than imported: deriving the expectation from
the function under test would make this vacuous, and nothing else in the suite
notices a layout that is correct but wasteful -- an oversized copy-engine
region still produces the right answer on every collective.

Single process, one GPU: workspace_size() is a pure function and needs no
collective.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.comm.pcie_ipc_ar import get_pcie_ipc_comm_module


def _align128(n: int) -> int:
    return (n + 127) & ~127


def _expected_total(
    world_size: int, max_numel: int, elem_size: int, max_blocks: int
) -> int:
    """Independent restatement of compute_workspace_layout()."""
    k_signal_phases, k_regions, k_ce_pieces, k_ce_stride = 8, 2, 4, 128

    signal_slots = (
        max_blocks  # epoch
        + k_signal_phases * max_blocks * world_size  # barrier phases
        + max_blocks  # barrier flags
        + 2 * k_regions  # {epoch, arrival} per region
    )
    signal_bytes = _align128(4 * signal_slots)
    max_payload = _align128(max_numel * elem_size)
    scratch_bytes = _align128(2 * world_size * max_payload)

    ce_slots = 2 * (world_size - 1) * k_ce_pieces + 2
    ce_flag_bytes = _align128(ce_slots * k_ce_stride)
    ce_counter_bytes = _align128(2 * ce_slots * 4)
    flat = 2 * (world_size - 1) * _align128(max_payload // world_size)
    island = 7 * _align128(max_payload // 4) if world_size == 8 else 0
    ce_scratch_bytes = max(flat, island)

    return (
        signal_bytes
        + 2 * scratch_bytes
        + ce_flag_bytes
        + ce_counter_bytes
        + ce_scratch_bytes
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("max_numel", [8 * 1024, 128 * 6144, 8192 * 6144])
def test_workspace_size_matches_the_documented_layout(
    world_size: int, max_numel: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU to load the module")
    module = get_pcie_ipc_comm_module()
    got = module.workspace_size(world_size, max_numel, 2, 128)
    assert got == _expected_total(world_size, max_numel, 2, 128)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_the_copy_engine_region_stays_proportional_to_the_payload(
    world_size: int,
) -> None:
    """The ring stages 2*(N-1) shards of payload/N, i.e. 2*(N-1)/N of the payload.

    Pinned by name because the failure is silent: a copy-engine region sized
    like an SM region (2 x 2 x N x payload) runs correctly and wastes gigabytes.
    At world_size 8 with a 96 MiB payload that is 1.5 GiB against 168 MiB.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU to load the module")
    module = get_pcie_ipc_comm_module()
    max_numel = 8192 * 6144
    payload = _align128(max_numel * 2)

    ce_slots = 2 * (world_size - 1) * 4 + 2
    ce_flag_bytes = _align128(ce_slots * 128)
    ce_counter_bytes = _align128(2 * ce_slots * 4)
    flat = 2 * (world_size - 1) * _align128(payload // world_size)
    island = 7 * _align128(payload // 4) if world_size == 8 else 0
    ce_scratch = max(flat, island)

    # The whole slab must equal the SM part plus exactly these three terms.
    signal_bytes = _align128(4 * (128 + 8 * 128 * world_size + 128 + 4))
    sm_bytes = signal_bytes + 2 * _align128(2 * world_size * payload)
    assert module.workspace_size(world_size, max_numel, 2, 128) == (
        sm_bytes + ce_flag_bytes + ce_counter_bytes + ce_scratch
    )

    expected_ratio = 2 * (world_size - 1) / world_size
    ratio = ce_scratch / payload
    assert abs(ratio - expected_ratio) < 0.01, (
        f"world_size={world_size}: staging is {ratio:.3f}x the payload, expected "
        f"{expected_ratio:.3f} = 2*(N-1)/N. Anything near 2*world_size means it "
        f"was sized like an SM region."
    )

    # And the flags/counters must stay negligible next to it -- they are a few
    # KiB, and a layout that made them scale with the payload would be wrong in
    # a way the ratio check above cannot see.
    assert ce_flag_bytes + ce_counter_bytes < 64 * 1024
