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

"""The window table of the affine composite's gathered-index main body (CPU; no GPU needed).

Each window's row holds the sequence whose caller state index is its pool slot
(-1 for a later window of a sequence), the row-table entry and rows before the
window that make its checkpoint row start, and the position of its sequence's
final pool offset on the merging route.  Every row the index gather writes
(``index_prep`` / ``index_prep_merge``) must come out of the table the same way.
"""

from types import SimpleNamespace

import flashinfer.cake_kda_tf32_runtime as runtime


def _gather(state_indices, cu_starts, table, num_parts, stride):
    """What the main body reads and publishes through the table."""
    slots, row_starts, offsets = [], [], {}
    for w in range(num_parts):
        first, entry, rows, position = table[4 * w : 4 * w + 4]
        slots.append(state_indices[first] if first >= 0 else -1)
        row_starts.append(cu_starts[entry] + rows)
        if position >= 0:
            offsets[position] = slots[-1] * stride
    return slots, row_starts, offsets


def test_in_place_rows_and_merge_offsets_match_the_index_gather() -> None:
    # Three sequences in 2 + 1 + 3 windows; window starts on the 64-token row grid.
    token_offsets = [0, 128, 192, 256, 320, 448, 512]
    part_offsets = [0, 2, 3, 6]
    num_parts = 6
    seq_ids, local_rows = runtime._affine_window_row_starts(token_offsets, part_offsets)
    final_positions = [
        end - 2 if end >= 2 else num_parts - 1 for end in part_offsets[1:]
    ]
    first_parts = part_offsets[:-1]
    table = runtime._affine_window_table(
        num_parts, first_parts, seq_ids, local_rows, final_positions
    )
    state_indices = [7, 3, 11]
    cu_starts = [40, 90, 130]
    stride = 1000
    slots, row_starts, offsets = _gather(
        state_indices, cu_starts, table, num_parts, stride
    )
    # index_prep: part_state_indices[first_parts[s]] = state_indices[s], -1 elsewhere.
    expected_slots = [-1] * num_parts
    for s, w in enumerate(first_parts):
        expected_slots[w] = state_indices[s]
    assert slots == expected_slots
    # part_row_starts[p] = checkpoint_start[part_seq_ids[p]] + part_local_rows[p].
    assert row_starts == [
        cu_starts[seq_ids[p]] + local_rows[p] for p in range(num_parts)
    ]
    # index_prep_merge: final_offsets[final_positions[s]] = state_indices[s] * stride.
    assert offsets == {final_positions[s]: state_indices[s] * stride for s in range(3)}


def test_rows_not_in_place_read_each_window_entry_and_no_merge_offset() -> None:
    table = runtime._affine_window_table(4, [0, 3], None, None, None)
    assert table == [0, 0, 0, -1, -1, 1, 0, -1, -1, 2, 0, -1, 1, 3, 0, -1]


def test_only_a_body_with_a_window_table_gathers() -> None:
    gathered = SimpleNamespace(
        module=SimpleNamespace(_arg_plan=[("tensor", "q"), ("tensor", "window_table")])
    )
    plain = SimpleNamespace(
        module=SimpleNamespace(
            _arg_plan=[("tensor", "q"), ("scalar", "state_indices_addr")]
        )
    )
    assert runtime._gathers_window_indices(gathered)
    assert not runtime._gathers_window_indices(plain)
    assert not runtime._gathers_window_indices(SimpleNamespace())
