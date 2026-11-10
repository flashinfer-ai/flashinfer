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

The shared host plan of the Cake non-deterministic GDN chunked-training package (``cake_plan``): the one-wave piece
table on the gating rows at the SM counts of B200 (148), GB300 / GB200 (152) and R200 (212), its limits, the cached
host plan entry on the launching device, the frozen tape level of each architecture and the storage-free placeholder.
CPU except the host plan entry.
"""

from __future__ import annotations

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

cake_chain = cake_package("cake_gdn_nondet_train", "cake_chain")
cake_plan = cake_package("cake_gdn_nondet_train", "cake_plan")

PAIRS = (11000, 12000, 9768)
ROWS = {
    "issue": (6, (131071, 87531, 43385)),
    "P3": (6, (128147, 131027)),
    "P1": (6, (126565, 32714, 33696, 15081, 22979, 10782)),
    "(4,12)": (12, PAIRS),
    "(2,6)": (6, PAIRS),
    "(1,3)": (3, PAIRS),
    "A16": (48, PAIRS),
    "(8,24)": (24, PAIRS),
}
# (row, SMs) -> (L, pieces per sequence, first-wave CTAs)
EXPECTED = {
    ("issue", 148): (158, (13, 9, 5), 144),
    ("P3", 148): (158, (13, 13), 144),
    ("P1", 148): (132, (15, 4, 4, 2, 3, 2), 144),
    ("(4,12)", 148): (38, (5, 5, 5), 144),
    ("(2,6)", 148): (20, (9, 10, 8), 144),
    ("(1,3)", 148): (11, (16, 18, 14), 135),
    ("A16", 148): (94, (2, 2, 2), 144),
    ("(8,24)", 148): (63, (3, 3, 3), 144),
    ("issue", 152): (152, (14, 9, 5), 150),
    ("P3", 152): (155, (13, 14), 150),
    ("(2,6)", 152): (20, (9, 10, 8), 144),
    ("(1,3)", 152): (10, (18, 19, 16), 150),
    ("issue", 212): (113, (19, 13, 6), 210),
    ("P1", 212): (99, (20, 6, 6, 3, 4, 2), 210),
    ("(4,12)", 212): (27, (7, 7, 6), 204),
    ("(2,6)", 212): (14, (13, 14, 11), 210),
    ("(1,3)", 212): (8, (22, 24, 20), 189),
}


def test_chunk_constant() -> None:
    assert cake_plan.CHUNK == cake_chain.CHUNK == 64
    assert cake_plan.chunk_counts((0, 1, 64, 65, 131071)) == (0, 1, 1, 2, 2048)


@pytest.mark.parametrize("key", sorted(EXPECTED), ids=lambda k: f"{k[0]}@{k[1]}")
def test_piece_table(key: tuple[str, int]) -> None:
    row, sms = key
    hv, lens = ROWS[row]
    length, counts, first = EXPECTED[key]
    table = cake_plan.pieces(lens, hv, sms)
    assert (table.length, table.counts, table.first_wave) == (length, counts, first)
    assert table.split
    assert table.first_wave <= sms and table.second_wave <= sms
    assert table.second_wave == sum(p - 1 for p in counts) * hv
    # the smallest L: one chunk shorter overfills the first wave
    chunks = cake_plan.chunk_counts(lens)
    assert sum(max(-(-n // (length - 1)) - 1, 1) for n in chunks) * hv > sms
    # each piece of the bounds the scans cut holds at most L chunks
    for n, p in zip(chunks, counts, strict=True):
        bounds = cake_chain.segment_bounds(n, length)
        assert len(bounds) - 1 == p
        assert (
            max(b - a for a, b in zip(bounds[:-1], bounds[1:], strict=True)) <= length
        )


def test_the_backward_split_reads_the_shared_table() -> None:
    # the one-wave split of the adjoint walk takes its piece length from the shared table
    for (row, sms), (length, _counts, _first) in EXPECTED.items():
        hv, lens = ROWS[row]
        chunks = cake_plan.chunk_counts(lens)
        assert cake_chain.bwd_piece_table(chunks, hv, sms) == length
        assert cake_chain.bwd_piece_table(chunks, hv, sms) == cake_plan.piece_length(
            chunks, hv, sms
        )
    for hv, lens, sms in ((6, (4096,) * 64, 148), (6, (64,) * 3, 148), (6, (), 148)):
        chunks = cake_plan.chunk_counts(lens)
        assert cake_chain.bwd_piece_table(chunks, hv, sms) is None


def test_more_sms_give_shorter_pieces() -> None:
    hv, lens = ROWS["issue"]
    lengths = [cake_plan.pieces(lens, hv, sms).length for sms in (148, 152, 160, 212)]
    assert lengths == sorted(lengths, reverse=True) and lengths[0] > lengths[-1]


@pytest.mark.parametrize(
    "hv, lens, sms",
    [
        (6, (4096,) * 64, 148),  # 64 x 6 sequence-heads exceed the SMs
        (6, (4096,) * 64, 212),
        (6, (64,) * 3, 148),  # one chunk per sequence
        (6, (), 148),
        (0, PAIRS, 148),
    ],
)
def test_no_split(hv: int, lens: tuple[int, ...], sms: int) -> None:
    table = cake_plan.pieces(lens, hv, sms)
    assert not table.split and table.length == 0
    assert table.counts == tuple(1 for _ in lens)
    assert table.second_wave == 0
    assert cake_plan.piece_length(cake_plan.chunk_counts(lens), hv, sms) is None


def test_the_tape_levels_are_frozen_and_keep_what_the_backward_reads() -> None:
    assert dict(cake_plan.TAPE_LEVELS) == {
        "sm_100a": "T_full",
        "sm_103a": "T_full",
        "sm_107a": "T_full",
    }
    with pytest.raises(TypeError):
        cake_plan.TAPE_LEVELS["sm_100a"] = "T_full"
    with pytest.raises(TypeError):
        cake_plan.LEVEL_STATE["T_full"] = cake_plan.STATE_NAMES
    assert cake_plan.STATE_NAMES == ("w", "u", "h", "v_new")
    for arch in cake_plan.TAPE_LEVELS:
        # u is read only to rebuild h / v_new, which T_full keeps
        assert cake_plan.tape_state(arch, True) == ("w", "h", "v_new")
        assert cake_plan.tape_state(arch, False) == ()


def test_placeholder() -> None:
    p = cake_plan.placeholder((5, 6, 128, 128), torch.float32, "cpu")
    assert (
        p.shape == (5, 6, 128, 128)
        and p.dtype == torch.float32
        and p.device.type == "cpu"
    )
    assert p.stride() == (0, 0, 0, 0)
    assert p.untyped_storage().nbytes() == 4
    empty = cake_plan.placeholder((0, 6, 128), torch.bfloat16, "cpu")
    assert empty.shape == (0, 6, 128) and empty.dtype == torch.bfloat16
    scalar = cake_plan.placeholder((), torch.float32, "cpu")
    assert scalar.shape == ()
    # every placeholder of a (device, dtype) views the same element: a call allocates nothing for it
    again = cake_plan.placeholder((3, 2), torch.float32, "cpu")
    assert again.untyped_storage().data_ptr() == p.untyped_storage().data_ptr()
    assert scalar.untyped_storage().data_ptr() == p.untyped_storage().data_ptr()
    assert empty.untyped_storage().data_ptr() != p.untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_host_plan_reads_the_device() -> None:
    device = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    hv, lens = ROWS["(2,6)"]
    try:
        plan = cake_plan.host_plan(lens, hv, device)
    except RuntimeError as err:  # not an exported architecture
        pytest.skip(str(err))
    assert plan.sm_count == sms
    assert plan.chunks == cake_plan.chunk_counts(lens)
    assert plan.pieces == cake_plan.pieces(lens, hv, sms)
    assert plan.arch in ("sm_100a", "sm_103a", "sm_107a")
    assert cake_plan.host_plan(lens, hv, device) is plan
