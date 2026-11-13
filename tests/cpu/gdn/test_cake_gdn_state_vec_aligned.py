# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests of the Cake GDN prefill state-alignment flavors.

Every prefill record compiles with ``-DCAKE_GDN_STATE_VEC_ALIGNED=1`` (16-byte-aligned
state, the unguarded body) or ``=0`` (element-aligned state, the guarded body); the
host picks the flavor per launch from the state pointers and slot strides.  The GPU
behaviour lives in ``tests/gdn/test_cake_gdn_prefill_state_align.py``.
"""

import pytest
import torch

from flashinfer import gdn_prefill
from flashinfer.jit import cake_gdn

_HEADS = 2
_SLOTS = 3
_STATE_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
)


def _state(dtype, offset=0, pad=0, slots=_SLOTS):
    """A ``[slots, H, 128, 128]`` view ``offset`` elements into a 16-byte-aligned buffer."""

    row = _HEADS * 16384
    stride0 = row + pad
    base = torch.empty(offset + (slots - 1) * stride0 + row + 16, dtype=dtype)
    assert base.data_ptr() % 16 == 0
    view = base.as_strided((slots, _HEADS, 128, 128), (stride0, 16384, 128, 1), offset)
    return view, int(view.stride(0))


def _aligned(initial, output, checkpoints):
    return gdn_prefill._cake_gdn_state_vec_aligned((initial, output), checkpoints)


def _lanes(dtype):
    return 16 // dtype.itemsize


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", _STATE_DTYPES, ids=str)
def test_state_vec_aligned_on_16_byte_aligned_buffers(dtype) -> None:
    checkpoints, _ = _state(dtype)
    assert _aligned(_state(dtype), _state(dtype), checkpoints)
    # A pool padded by a whole 16 bytes per slot stays aligned.
    padded = _state(dtype, pad=_lanes(dtype))
    assert _aligned(padded, padded, checkpoints)
    # The one-element sentinel of an absent buffer, with the default slot stride.
    sentinel = torch.empty(1, dtype=dtype)
    assert _aligned((sentinel, _HEADS * 16384), (sentinel, _HEADS * 16384), sentinel)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", _STATE_DTYPES, ids=str)
def test_state_vec_aligned_rejects_any_misaligned_buffer(dtype) -> None:
    lanes = _lanes(dtype)
    checkpoints, _ = _state(dtype)
    for offset in sorted({1, lanes // 2, lanes - 1}):
        off = _state(dtype, offset=offset)
        assert off[0].data_ptr() % 16 != 0
        assert not _aligned(off, _state(dtype), checkpoints), ("initial", offset)
        assert not _aligned(_state(dtype), off, checkpoints), ("output", offset)
        # In place: the pool is both the initial and the output state.
        assert not _aligned(off, off, checkpoints), ("in_place", offset)
        # Padded slot stride on an aligned pointer: slots alternate in alignment.
        padded = _state(dtype, pad=offset)
        assert padded[0].data_ptr() % 16 == 0
        assert not _aligned(padded, _state(dtype), checkpoints), ("padded", offset)
        assert not _aligned(_state(dtype), padded, checkpoints), ("padded", offset)
        # Checkpoint rows are dense: only their pointer decides.
        checkpoints_off, _ = _state(dtype, offset=offset)
        assert not _aligned(_state(dtype), _state(dtype), checkpoints_off), (
            "checkpoints",
            offset,
        )


@pytest.mark.cpu_only
def test_compile_options_add_the_state_flavor_to_prefill_records_only() -> None:
    define = "-DCAKE_GDN_STATE_VEC_ALIGNED"
    domains = set()
    for record in cake_gdn._manifest()["variants"]:
        base = tuple(
            record.get("compile_options", ("--use_fast_math",))
        ) + cake_gdn._define_flags(record)
        domains.add(record["domain"])
        if record["domain"] == "prefill":
            assert cake_gdn._compile_options(record) == base + (f"{define}=1",)
            assert cake_gdn._compile_options(record, state_vec_aligned=True) == base + (
                f"{define}=1",
            )
            assert cake_gdn._compile_options(
                record, state_vec_aligned=False
            ) == base + (f"{define}=0",)
        else:
            # Decode keeps its exact options (and so its cubin digest).
            options = cake_gdn._compile_options(record)
            assert options == base
            assert not any(option.startswith(define) for option in options)
            with pytest.raises(ValueError, match="no state-alignment flavor"):
                cake_gdn._compile_options(record, state_vec_aligned=False)
    assert domains == {"decode", "prefill"}
