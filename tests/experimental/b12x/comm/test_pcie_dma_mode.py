"""Tests for PCIe-DMA wire-mode parsing."""

from __future__ import annotations

import pytest

from b12x.comm.pcie.pcie_dma import _eager_replay_capacities, _normalize_fp8_mode


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "no"])
def test_disabled_wire_mode_aliases(value: str | None) -> None:
    assert _normalize_fp8_mode(value) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", "ag"),
        (" AG ", "ag"),
        ("ring", "ring"),
        ("a2a", "a2a"),
        ("i8", "i8"),
        ("int8", "i8"),
        ("i8_ag", "i8"),
        ("i8-ag", "i8"),
        ("ag_i8", "i8"),
        ("int8_ag", "i8"),
        ("int8-ag", "i8"),
        ("i8_ring", "i8_ring"),
        ("i8-ring", "i8_ring"),
        ("int8_ring", "i8_ring"),
        ("int8-ring", "i8_ring"),
        ("ring_i8", "i8_ring"),
        ("i8_a2a", "i8_a2a"),
        ("i8-a2a", "i8_a2a"),
        ("int8_a2a", "i8_a2a"),
        ("int8-a2a", "i8_a2a"),
        ("a2a_i8", "i8_a2a"),
        ("  InT8-RiNg ", "i8_ring"),
        ("mx", "mx"),
        ("mxfp8", "mx"),
        ("mx_ag", "mx"),
        ("mx-ag", "mx"),
        ("mxfp8_ag", "mx"),
        ("mxfp8-ag", "mx"),
        ("ag_mx", "mx"),
        ("mx_ring", "mx_ring"),
        ("mx-ring", "mx_ring"),
        ("mxfp8_ring", "mx_ring"),
        ("mxfp8-ring", "mx_ring"),
        ("ring_mx", "mx_ring"),
        ("mx_a2a", "mx_a2a"),
        ("mx-a2a", "mx_a2a"),
        ("mxfp8_a2a", "mx_a2a"),
        ("mxfp8-a2a", "mx_a2a"),
        ("a2a_mx", "mx_a2a"),
        (" MxFp8-RiNg ", "mx_ring"),
    ],
)
def test_supported_wire_mode_aliases(value: str, expected: str) -> None:
    assert _normalize_fp8_mode(value) == expected


def test_unknown_wire_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unrecognized PCIe DMA wire mode"):
        _normalize_fp8_mode("mx_rnig")


@pytest.mark.parametrize("world_size", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("itemsize", [2, 4])
@pytest.mark.parametrize(
    "min_bytes,max_bytes", [(0, 4096), (0, 12 << 20), (6 << 20, 40 << 20)]
)
def test_eager_capacities_preserve_shard_alignment(
    world_size, itemsize, min_bytes, max_bytes
):
    capacities = _eager_replay_capacities(max_bytes, min_bytes, itemsize, world_size)
    multiple = world_size * 8
    assert capacities == tuple(sorted(set(capacities)))
    assert all(value > 0 and value % multiple == 0 for value in capacities)
    assert capacities[-1] == max_bytes // itemsize // multiple * multiple
    for smaller, larger in zip(capacities, capacities[1:], strict=False):
        assert larger <= 2 * smaller + multiple


def test_eager_capacity_rejects_less_than_one_shard_vector():
    with pytest.raises(ValueError, match="too small"):
        _eager_replay_capacities(63, 0, 2, 4)


@pytest.mark.parametrize("itemsize", [2, 4])
def test_eager_dtype_bound_retains_intermediate_capacity_plans(itemsize):
    elements = 4096 * 5120
    capacities = _eager_replay_capacities(
        elements * 4, 6 << 20, itemsize, 4, max_elements=elements
    )
    assert capacities[-1] == elements
    assert tuple(capacity * itemsize for capacity in capacities) == (
        (8 << 20, 16 << 20, 32 << 20, 40 << 20)
        if itemsize == 2
        else (8 << 20, 16 << 20, 32 << 20, 64 << 20, 80 << 20)
    )


@pytest.mark.parametrize("elements", [0, -1, 1025])
def test_eager_dtype_bound_rejects_invalid_limits(elements):
    with pytest.raises(ValueError, match="dtype capacity"):
        _eager_replay_capacities(4096, 0, 4, 4, max_elements=elements)
