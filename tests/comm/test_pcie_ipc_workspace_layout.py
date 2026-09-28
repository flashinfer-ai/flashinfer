"""The workspace's byte count, checked against an independent restatement.

The arithmetic is restated rather than imported: deriving the expectation from
the function under test would make this vacuous, and nothing else in the suite
notices a layout that is correct but wasteful -- an oversized copy-engine
region still produces the right answer on every collective.

Single process, one GPU: workspace_size() accounts for the current device's
optional memory-operation flags and needs no collective.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.comm.pcie_ipc_ar import get_pcie_ipc_comm_module


def _align128(n: int) -> int:
    return (n + 127) & ~127


def _expected_binary_flag_bytes(world_size: int) -> int:
    # SM120 TP4/TP8 append one cache-line-separated flag per flat-ring step.
    if world_size in (4, 8) and torch.cuda.get_device_capability() == (12, 0):
        return 2 * (world_size - 1) * 128
    return 0


def _expected_total(
    world_size: int,
    max_numel: int,
    elem_size: int,
    max_blocks: int,
) -> int:
    """Independent restatement of the base layout and optional binary flags."""
    k_signal_phases, k_regions, k_ce_pieces, k_ce_stride = 8, 2, 4, 128

    signal_slots = (
        max_blocks  # epoch
        + k_signal_phases * max_blocks * world_size  # barrier phases
        + max_blocks  # barrier flags
        + 2 * k_regions  # {epoch, arrival} per region
        + 2  # {arrival, generation} for the fused kernels' grid barrier
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
        + _expected_binary_flag_bytes(world_size)
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

    # The whole slab includes the SM part, CE regions, and optional binary flags.
    binary_flag_bytes = _expected_binary_flag_bytes(world_size)
    # The trailing 6 is 2 per scratch region plus the fused grid barrier pair.
    # Stated rather than absorbed: those 8 bytes currently disappear into the
    # 128-byte alignment at every world size tested here, so an omission stays
    # invisible until a shape moves the total across a boundary.
    signal_bytes = _align128(4 * (128 + 8 * 128 * world_size + 128 + 6))
    sm_bytes = signal_bytes + 2 * _align128(2 * world_size * payload)
    assert module.workspace_size(world_size, max_numel, 2, 128) == (
        sm_bytes + ce_flag_bytes + ce_counter_bytes + ce_scratch + binary_flag_bytes
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
    assert ce_flag_bytes + ce_counter_bytes + binary_flag_bytes < 64 * 1024


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_the_flag_block_predicate_agrees_with_the_module(world_size: int) -> None:
    """The gate above is a second copy of one the binding already owns.

    ``_expected_binary_flag_bytes`` decides from the compute capability; the
    allocation decides from ``pcie_ipc_memop_supported()``. Two predicates for
    one fact is how the term went unmodelled in the first place -- it was absent
    from the expectation entirely, which passed on every device that cannot run
    the protocol and failed on every device that can. Restating it rather than
    importing it is deliberate, so this asserts they still agree instead.

    Also pinned: the term is a function of the world size alone. It is flags,
    not staging, so it must not move with the payload.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU to load the module")
    module = get_pcie_ipc_comm_module()
    expected = 2 * (world_size - 1) * 128 if world_size in (4, 8) else 0
    from_module = expected if module.memop_supported() and world_size in (4, 8) else 0
    assert _expected_binary_flag_bytes(world_size) == from_module

    deltas = {
        module.workspace_size(world_size, n, 2, 128)
        - _expected_total(world_size, n, 2, 128)
        for n in (8 * 1024, 128 * 6144, 8192 * 6144)
    }
    assert deltas == {0}, (
        f"the layout has a term the expectation does not model: {deltas}"
    )
