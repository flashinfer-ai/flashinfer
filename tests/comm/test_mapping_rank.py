# SPDX-License-Identifier: Apache-2.0

import pytest

from flashinfer.comm.mapping import Mapping


@pytest.mark.parametrize("rank", [-1, 4, 5, 1.5, "1"])
def test_constructor_rejects_invalid_rank(rank):
    with pytest.raises(ValueError, match="Rank should be an integer between 0 and 3"):
        Mapping(world_size=4, tp_size=2, pp_size=2, rank=rank)


@pytest.mark.parametrize("rank", [-1, 4, 5, 1.5, "1"])
def test_rank_assignment_rejects_invalid_rank_without_changing_groups(rank):
    mapping = Mapping(world_size=4, tp_size=2, pp_size=2, rank=1)
    with pytest.raises(ValueError, match="Rank should be an integer between 0 and 3"):
        mapping.rank = rank
    assert mapping.rank == 1
    assert mapping.tp_group == [0, 1]
    assert mapping.pp_group == [1, 3]


@pytest.mark.parametrize("rank", [0, 3])
def test_rank_boundaries_and_reassignment_select_valid_groups(rank):
    mapping = Mapping(world_size=4, tp_size=2, pp_size=2, rank=rank)
    assert mapping.rank == rank
    assert mapping.tp_group == ([0, 1] if rank == 0 else [2, 3])
    assert mapping.pp_group == ([0, 2] if rank == 0 else [1, 3])
    mapping.rank = 3 - rank
    assert mapping.rank == 3 - rank
    assert mapping.tp_group == ([2, 3] if rank == 0 else [0, 1])


@pytest.mark.parametrize("rank", [-1, 1])
def test_single_rank_mapping_rejects_out_of_bounds_rank(rank):
    with pytest.raises(ValueError, match="Rank should be an integer between 0 and 0"):
        Mapping(rank=rank)


@pytest.mark.parametrize("rank", [-1, 4])
def test_attention_dp_keeps_existing_rank_validation_bypass(rank):
    mapping = Mapping(
        world_size=4, tp_size=2, pp_size=2, rank=rank, enable_attention_dp=True
    )
    assert mapping.rank == rank
    mapping.rank = rank + 1
    assert mapping.rank == rank + 1
