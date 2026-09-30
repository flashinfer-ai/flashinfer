from __future__ import annotations

import pytest
import torch

from b12x.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    _SIGNAL_BYTES,
    _candidate_staging_layout,
    owner_stage_reference,
)
from b12x.comm.pcie._dcp_cute_common import signal_bytes


class _FakeOwner(PCIeDCPTopKOwnerExchange):
    def __init__(self) -> None:
        super().__init__(
            rank=1,
            world_size=2,
            device="cpu",
            signal_ptrs=(10, 20),
            staging0_ptrs=(30, 40),
            staging1_ptrs=(62, 72),
            max_rows=8,
            topk=4,
        )
        shape = (self.max_owner_rows, self.world_size * self.topk)
        self._candidate_views = tuple(
            (
                torch.empty(shape, dtype=torch.int32),
                torch.empty(shape, dtype=torch.float32),
            )
            for _ in range(2)
        )
        self.stage_calls: list[tuple] = []

    def _launch_stage(
        self,
        local_indices,
        local_scores,
        *,
        slot,
        rows,
        threads,
        blocks,
        wait_for_prior_consumer,
    ):
        owner_rows = rows // self.world_size
        row_slice = slice(self.rank * owner_rows, (self.rank + 1) * owner_rows)
        views = self._candidate_views[slot]
        views[0][:owner_rows].copy_(
            local_indices[row_slice].repeat(1, self.world_size)
        )
        views[1][:owner_rows].copy_(
            local_scores[row_slice].repeat(1, self.world_size)
        )
        self.stage_calls.append((slot, threads, blocks, wait_for_prior_consumer))


def _make_owner() -> PCIeDCPTopKOwnerExchange:
    return _FakeOwner()


def test_candidate_staging_layout_is_aligned():
    owner = _candidate_staging_layout(
        signal_bytes=513,
        max_rows=8,
        topk=4,
        world_size=2,
    )
    assert owner.staging0_offset == 768
    assert owner.plane_bytes == 128
    assert owner.slot_bytes == 256
    assert owner.staging1_offset == 1024
    assert owner.slab_bytes == 1280


def test_topk_signal_layout_matches_the_shared_barrier_abi():
    assert signal_bytes(128) == _SIGNAL_BYTES


def test_owner_stage_reference_preserves_rank_major_order_and_score_bits():
    world_size, rows, topk = 4, 8, 4
    indices = torch.empty(world_size, rows, topk, dtype=torch.int32)
    score_bits = torch.empty(world_size, rows, topk, dtype=torch.int32)
    for rank in range(world_size):
        for row in range(rows):
            indices[rank, row].fill_(1000 * rank + 10 * row)
            score_bits[rank, row].fill_(0x3F000000 + rank * 16 + row)
    scores = score_bits.view(torch.float32)

    owner_indices, owner_scores = owner_stage_reference(indices, scores, 2)

    assert owner_indices.shape == (2, world_size * topk)
    assert owner_scores.shape == owner_indices.shape
    for owner_row, global_row in enumerate((4, 5)):
        for rank in range(world_size):
            rank_slice = slice(rank * topk, (rank + 1) * topk)
            assert torch.equal(
                owner_indices[owner_row, rank_slice], indices[rank, global_row]
            )
            assert torch.equal(
                owner_scores[owner_row, rank_slice].view(torch.int32),
                score_bits[rank, global_row],
            )


def test_owner_dispatches_and_disposes():
    owner = _make_owner()
    indices = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10

    candidate_indices, candidate_scores = owner.stage_candidates(
        indices,
        scores,
        threads=128,
        block_limit=32,
    )
    assert torch.equal(candidate_indices, indices[4:].repeat(1, 2))
    assert torch.equal(candidate_scores, scores[4:].repeat(1, 2))
    assert owner.stage_calls == [(0, 128, 1, False)]

    owner.close()
    assert owner._closed


@pytest.mark.parametrize("eager_calls, expected_slot", [(0, 0), (1, 1), (2, 0)])
def test_graph_capture_pins_the_next_staging_slot_and_enables_prior_wait(
    monkeypatch,
    eager_calls,
    expected_slot,
):
    owner = _make_owner()
    indices = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    capture_state = False
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_topk._is_current_stream_capturing",
        lambda _device: capture_state,
    )
    monkeypatch.setattr(owner, "prepare_graph", lambda **kwargs: None)
    monkeypatch.setattr(
        "b12x.comm.pcie._dcp_topk_cute.is_topk_stage_prepared",
        lambda *args: True,
    )

    for _ in range(eager_calls):
        owner.stage_candidates(indices, scores)
    capture_state = True
    with owner.capture():
        first_indices, first_scores = owner.stage_candidates(indices, scores)
    capture_state = False
    second_indices, second_scores = owner.stage_candidates(indices + 100, scores + 1)
    capture_state = True
    with owner.capture():
        third_indices, third_scores = owner.stage_candidates(indices + 200, scores + 2)

    assert owner._graph_slot == expected_slot
    assert first_indices.data_ptr() == second_indices.data_ptr()
    assert second_indices.data_ptr() == third_indices.data_ptr()
    assert first_scores.data_ptr() == second_scores.data_ptr()
    assert second_scores.data_ptr() == third_scores.data_ptr()
    eager_stages = [
        (slot % 2, 512, 1, False) for slot in range(eager_calls)
    ]
    assert owner.stage_calls == eager_stages + [
        (expected_slot, 512, 1, True),
        (expected_slot, 512, 1, True),
        (expected_slot, 512, 1, True),
    ]
    assert torch.equal(third_indices, (indices + 200)[4:].repeat(1, 2))


def test_owner_rejects_invalid_contracts():
    owner = _make_owner()
    indices = torch.zeros(8, 4, dtype=torch.int32)
    scores = torch.zeros(8, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="matching"):
        owner.stage_candidates(indices, scores[:4])
    with pytest.raises(ValueError, match="matching"):
        owner.stage_candidates(indices.flatten(), scores.flatten())
    with pytest.raises(ValueError, match="local_indices must be int32"):
        owner.stage_candidates(indices.float(), scores)
    with pytest.raises(ValueError, match="local_scores must be float32"):
        owner.stage_candidates(indices, scores.bfloat16())
    with pytest.raises(ValueError, match="divisible"):
        owner.stage_candidates(indices[:7], scores[:7])
    with pytest.raises(ValueError, match="threads"):
        owner.stage_candidates(indices, scores, threads=31)
    with pytest.raises(ValueError, match="block_limit"):
        owner.stage_candidates(indices, scores, block_limit=129)


def test_configuration_rejects_invalid_capacity_and_topk():
    with pytest.raises(ValueError, match="multiple of 4"):
        _candidate_staging_layout(
            signal_bytes=256,
            max_rows=8,
            topk=6,
            world_size=2,
        )
    with pytest.raises(ValueError, match="divisible"):
        PCIeDCPTopKOwnerExchange(
            rank=0,
            world_size=4,
            device="cpu",
            signal_ptrs=(1, 2, 3, 4),
            staging0_ptrs=(5, 6, 7, 8),
            staging1_ptrs=(9, 10, 11, 12),
            max_rows=6,
            topk=4,
        )
