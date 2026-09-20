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

"""Guard-zone helper semantics (gh #3978, work order cases A3-01 .. A3-11).

The helper under test is ``tests/test_helpers/guard_zone.py``; the real-kernel
integration lives in ``test_guard_zone_kernel.py`` (A3-12).
"""

import gc
import weakref

import math

import pytest
import torch

import flashinfer
from tests.test_helpers.guard_zone import (
    PREFIX_SENTINEL,
    SUFFIX_SENTINEL,
    GuardZoneCorruption,
    allocate_guarded,
    clear_guards,
    inject_out_of_bounds_write,
    live_guarded,
    release_guarded,
    reset_guards,
    verify_guards,
)

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.fixture(autouse=True)
def isolate_guard_registry():
    """No case may leave a registered buffer behind for the next one."""
    yield
    clear_guards()


def test_layout_is_one_allocation_with_a_tight_suffix_guard():
    """A3-01: the documented layout, and a payload write leaves the guards alone."""
    buffer = allocate_guarded((8, 16), torch.float32, "cpu", alignment=64)

    assert buffer.owner.dtype == torch.uint8
    assert buffer.owner.numel() == (
        2 * buffer.guard_bytes
        + math.lcm(buffer.alignment, buffer.payload.element_size())
        + buffer.payload_nbytes
    )
    # The prefix guard ends where the payload starts -- no unscanned padding
    # between them, so a store at ``payload_start - 1`` is inside the guard.
    assert (
        buffer.prefix_guard.data_ptr() == buffer.payload.data_ptr() - buffer.guard_bytes
    )
    assert buffer.payload.data_ptr() == buffer.owner.data_ptr() + buffer.payload_start
    assert buffer.payload_start >= buffer.guard_bytes
    # Padding in front of the prefix guard is smaller than the alignment step
    # it exists to satisfy.
    assert buffer.payload_start - buffer.guard_bytes < math.lcm(
        buffer.alignment, buffer.payload.element_size()
    )
    # Payload and both guards share one storage: one allocation, one free, so
    # the guards cannot be released out from under a running kernel.
    assert (
        buffer.payload.untyped_storage().data_ptr()
        == buffer.owner.untyped_storage().data_ptr()
    )
    # The suffix guard begins at the payload's last byte + 1 -- no gap, so a
    # one-byte overrun is inside the guard.
    assert (
        buffer.suffix_guard.data_ptr()
        == buffer.payload.data_ptr() + buffer.payload_nbytes
    )

    reset_guards()
    buffer.payload.fill_(1.75)

    verify_guards()
    assert buffer.prefix_guard.tolist() == [PREFIX_SENTINEL] * buffer.guard_bytes
    assert buffer.suffix_guard.tolist() == [SUFFIX_SENTINEL] * buffer.guard_bytes


@pytest.mark.parametrize("device", DEVICES)
def test_padding_and_slack_are_initialized_but_never_compared(device):
    """A3-01: only guard bytes are armed; the rest of the owner stays 0x00."""
    buffer = allocate_guarded((3, 5), torch.uint8, device, alignment=256)

    padding = buffer.owner[: buffer.payload_start - buffer.guard_bytes]
    slack = buffer.owner[buffer.payload_end + buffer.guard_bytes :]
    assert int(padding.count_nonzero()) == 0
    assert int(slack.count_nonzero()) == 0

    # Neither sentinel is the fill value, so a scan that accidentally widened
    # past a guard region would fail instead of passing.
    assert set(buffer.owner.tolist()) == {0x00, PREFIX_SENTINEL, SUFFIX_SENTINEL}

    buffer.payload.fill_(0x00)
    verify_guards()


def test_prefix_injection_reports_region_and_offset():
    """A3-02: an 8-byte underflow is reported in the prefix guard, offset exact."""
    buffer = allocate_guarded((4, 32), torch.float32, "cuda", guard_bytes=256)
    reset_guards()

    inject_out_of_bounds_write(buffer, "prefix", size=8)

    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    (hit,) = excinfo.value.hits
    assert hit.region == "prefix"
    assert hit.offset == buffer.payload_start - 8
    assert hit.owner_offset == buffer.payload_start - 8
    assert hit.expected == PREFIX_SENTINEL
    assert hit.corrupted_bytes == 8
    assert hit.device == torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("device", DEVICES)
def test_the_byte_before_the_payload_is_guarded(device):
    """A3-07: an underrun that stops one byte short of the payload is caught.

    The payload offset is aligned, so padding can precede it.  A prefix guard
    anchored at the owner start instead of next to the payload would leave that
    padding unscanned and report a clean run for a store at ``payload_start-1``.
    """
    buffer = allocate_guarded((4, 32), torch.float32, device, alignment=256)
    buffer.owner[buffer.payload_start - 1] = 0x00

    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    (hit,) = excinfo.value.hits
    assert hit.region == "prefix"
    assert hit.offset == buffer.guard_bytes - 1
    assert hit.owner_offset == buffer.payload_start - 1
    assert hit.actual == 0x00


@pytest.mark.parametrize("device", DEVICES)
def test_alignment_below_the_element_size_keeps_the_payload_viewable(device):
    """A3-08: alignment=1 with float32 still yields a legal, aligned payload.

    The payload offset must satisfy the address alignment *and* keep the storage
    offset a multiple of the element size, which a raw ``% alignment`` can miss
    when ``guard_bytes`` is not a multiple of ``itemsize``.
    """
    buffer = allocate_guarded((5, 3), torch.float32, device, guard_bytes=7, alignment=1)

    # ``storage_offset`` is in payload elements, so the byte offset the dtype
    # view had to divide evenly is ``storage_offset * element_size``.
    assert (
        buffer.payload.storage_offset() * buffer.payload.element_size()
        == buffer.payload_start
    )
    assert buffer.payload.data_ptr() % 1 == 0
    assert buffer.payload.dtype == torch.float32
    assert buffer.payload.shape == torch.Size((5, 3))
    # The prefix guard still ends exactly at the payload.
    assert (
        buffer.prefix_guard.data_ptr() == buffer.payload.data_ptr() - buffer.guard_bytes
    )
    reset_guards()
    buffer.payload.fill_(2.5)
    verify_guards()


def test_suffix_injection_reports_region_and_offset():
    """A3-03: an 8-byte overflow is reported in the suffix guard, offset exact."""
    buffer = allocate_guarded((4, 32), torch.float32, "cuda", guard_bytes=256)
    reset_guards()

    inject_out_of_bounds_write(buffer, "suffix", size=8)

    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    (hit,) = excinfo.value.hits
    assert hit.region == "suffix"
    assert hit.offset == 0  # the byte immediately past the payload
    assert hit.owner_offset == buffer.payload_end
    assert hit.expected == SUFFIX_SENTINEL
    assert hit.corrupted_bytes == 8
    assert hit.device == torch.device("cuda", torch.cuda.current_device())


def test_injection_crossing_the_boundary_stays_inside_the_owner():
    """A3-02/A3-03: a misaligned tail store crosses the boundary but stays bounded."""
    buffer = allocate_guarded((4, 32), torch.float32, "cuda", guard_bytes=64)
    reset_guards()

    # 1 byte inside the payload, 3 bytes past its end: the classic vectorised
    # tail.  It must stay inside the owning allocation.
    inject_out_of_bounds_write(buffer, "suffix", size=4, overlap=1)

    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    (hit,) = excinfo.value.hits
    assert (hit.region, hit.offset, hit.corrupted_bytes) == ("suffix", 0, 3)


def test_corruption_is_reported_while_the_payload_stays_correct():
    """A3-04: the guard signal is independent of any output comparison."""
    buffer = allocate_guarded((16, 32), torch.float32, "cuda")
    expected = torch.randn(16, 32, device="cuda")
    reset_guards()
    buffer.payload.copy_(expected)
    torch.testing.assert_close(buffer.payload, expected)

    inject_out_of_bounds_write(buffer, "suffix", size=8)

    # The payload is still exactly right: only the guard sees the overrun.
    torch.testing.assert_close(buffer.payload, expected)
    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    assert [hit.region for hit in excinfo.value.hits] == ["suffix"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.int32,
        torch.bool,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_guards_are_compared_as_bytes_not_as_payload_values(device, dtype):
    """A3-05: a NaN payload cannot produce a false guard report."""
    buffer = allocate_guarded((4, 6), dtype, device)
    assert buffer.prefix_guard.dtype == torch.uint8
    assert buffer.suffix_guard.dtype == torch.uint8

    if dtype.is_floating_point:
        buffer.payload.fill_(float("nan"))
        assert bool(torch.isnan(buffer.payload).all())
        # Why the guards are never compared in the payload's dtype: a NaN
        # payload is not equal to itself, so a float comparison would look
        # like corruption on a healthy buffer.
        assert not bool((buffer.payload == buffer.payload).all())
    else:
        buffer.payload.fill_(7)

    reset_guards()
    verify_guards()

    inject_out_of_bounds_write(buffer, "suffix", size=4, value=0xFF)
    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    (hit,) = excinfo.value.hits
    assert (hit.region, hit.offset, hit.actual) == ("suffix", 0, 0xFF)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("alignment", [1, 16, 64, 256, 512])
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1,), torch.uint8),
        ((3,), torch.float32),
        ((7, 3), torch.float16),
        ((1, 5, 9), torch.bfloat16),
    ],
)
def test_odd_shapes_and_alignments_keep_the_payload_legal(
    device, alignment, shape, dtype
):
    """A3-06: alignment is honoured and neither guard reaches into the payload."""
    buffer = allocate_guarded(shape, dtype, device, alignment=alignment)

    assert buffer.payload.data_ptr() % alignment == 0
    assert buffer.payload.is_contiguous()
    assert buffer.payload.shape == torch.Size(shape)
    assert buffer.payload.dtype == dtype
    assert buffer.payload.data_ptr() == buffer.owner.data_ptr() + buffer.payload_start
    assert buffer.payload_start - buffer.guard_bytes < math.lcm(
        alignment, buffer.payload.element_size()
    )

    # Round-tripping the whole payload, odd byte count included, neither reads
    # a guard byte nor writes one.
    buffer.payload.fill_(3)
    assert bool((buffer.payload == 3).all())
    reset_guards()
    buffer.payload.fill_(2)
    verify_guards()
    assert bool((buffer.payload == 2).all())


def test_non_contiguous_and_aliasing_templates_are_rejected():
    """A3-07: layouts the first version cannot guard are refused, not mis-guarded."""
    base = torch.zeros(4, 8)

    with pytest.raises(ValueError, match="non-contiguous"):
        allocate_guarded(base.t())
    with pytest.raises(ValueError, match="non-contiguous"):
        allocate_guarded(base[:, ::2])
    with pytest.raises(ValueError, match="non-contiguous"):
        allocate_guarded(base[0:1].expand(4, 8))
    with pytest.raises(ValueError, match="storage offset"):
        allocate_guarded(base.narrow(0, 1, 2))
    with pytest.raises(ValueError, match="larger storage"):
        allocate_guarded(torch.zeros(64)[:8])
    with pytest.raises(ValueError, match="strided only"):
        allocate_guarded(
            torch.sparse_coo_tensor(
                torch.zeros(2, 2, dtype=torch.long), torch.zeros(2), (3, 3)
            )
        )

    # Positive control: a plain contiguous template is accepted and only
    # describes what to allocate.
    buffer = allocate_guarded(base)
    assert buffer.payload.shape == torch.Size((4, 8))
    assert buffer.payload.dtype == torch.float32
    assert buffer.payload.is_contiguous()
    assert buffer.payload.data_ptr() != base.data_ptr()
    verify_guards()


def test_verify_waits_for_work_enqueued_on_a_non_default_stream():
    """A3-08: guards clobbered on a side stream are visible to verify_guards."""
    buffer = allocate_guarded((256, 64), torch.float32, "cuda")
    side = torch.cuda.Stream()
    reset_guards()

    with torch.cuda.stream(side):
        buffer.payload.fill_(0.5)
    verify_guards()

    # One clobber enqueued on the side stream, one on the current stream; the
    # device synchronisation inside verify_guards is what makes either visible.
    with torch.cuda.stream(side):
        inject_out_of_bounds_write(buffer, "prefix", size=16)
    inject_out_of_bounds_write(buffer, "suffix", size=16)

    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    assert sorted(hit.region for hit in excinfo.value.hits) == ["prefix", "suffix"]


def test_guard_state_survives_cuda_graph_replay():
    """A3-09: reset -> capture -> replay -> verify, with fixed addresses."""
    buffer = allocate_guarded((64, 512), torch.float16, "cuda", alignment=16)
    x = torch.randn(64, 1024, dtype=torch.float16, device="cuda")
    expected = flashinfer.silu_and_mul(x, enable_pdl=False)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashinfer.silu_and_mul(x, out=buffer.payload, enable_pdl=False)

    # The reset happens outside the graph, before the replay: it can never
    # re-arm a guard on a later replay and hide a corruption.
    reset_guards()
    graph.replay()
    verify_guards()
    torch.testing.assert_close(buffer.payload, expected)

    inject_out_of_bounds_write(buffer, "suffix", size=8)
    graph.replay()
    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    (hit,) = excinfo.value.hits
    assert (hit.region, hit.offset, hit.corrupted_bytes) == ("suffix", 0, 8)


@pytest.mark.parametrize("device_index", [0, 1])
def test_guarded_allocation_is_local_to_its_device(device_index):
    """A3-10: one independent guarded allocation per device, owner matches."""
    if torch.cuda.device_count() <= device_index:
        pytest.skip(f"needs {device_index + 1} GPUs")
    device = torch.device("cuda", device_index)

    buffer = allocate_guarded((8, 64), torch.float32, device, alignment=256)

    assert buffer.owner.device == device
    assert buffer.payload.device == device
    assert buffer.prefix_guard.device == device
    assert buffer.suffix_guard.device == device
    assert live_guarded(device) == (buffer,)

    reset_guards(device=device)
    verify_guards(device=device)
    inject_out_of_bounds_write(buffer, "suffix", size=8)
    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards(device=device)
    (hit,) = excinfo.value.hits
    assert hit.device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_corruption_is_attributed_to_the_device_that_owns_it():
    """A3-10: two devices, one corruption; the untouched device stays clean."""
    first = allocate_guarded((8, 64), torch.float32, "cuda:0", alignment=256)
    second = allocate_guarded((8, 64), torch.float32, "cuda:1", alignment=256)
    reset_guards()
    inject_out_of_bounds_write(second, "prefix", size=8)

    with pytest.raises(GuardZoneCorruption) as excinfo:
        verify_guards()
    assert {hit.device for hit in excinfo.value.hits} == {torch.device("cuda:1")}

    verify_guards(device="cuda:0")
    with pytest.raises(GuardZoneCorruption):
        verify_guards(device="cuda:1")
    assert live_guarded("cuda:1") == (second,)
    assert live_guarded("cuda:0") == (first,)


def test_allocation_and_release_cycles_leave_no_registered_state():
    """A3-11: repeated allocate/verify/release leaves nothing dangling behind."""
    for _ in range(8):
        buffer = allocate_guarded((4, 32), torch.float32, "cuda", alignment=128)
        reset_guards()
        verify_guards()
        inject_out_of_bounds_write(buffer, "suffix", size=8)
        with pytest.raises(GuardZoneCorruption) as excinfo:
            verify_guards()
        # The hit belongs to the buffer still registered, not to a stale one.
        assert excinfo.value.hits[0].owner_offset == buffer.payload_end
        release_guarded(buffer)

    assert live_guarded() == ()
    verify_guards()


def test_released_allocation_is_collectable_and_the_payload_outlives_it():
    """A3-11: no retained allocation, and no dangling payload view."""
    buffer = allocate_guarded((4, 32), torch.float32, "cuda")
    payload = buffer.payload
    owner_ref = weakref.ref(buffer.owner)

    release_guarded(buffer)
    del buffer
    gc.collect()

    # The registry was the only other reference, so once the payload view goes
    # too the whole allocation is collectable: no leak.
    payload.fill_(1.0)
    assert int((payload == 1.0).all()) == 1
    del payload
    gc.collect()
    assert owner_ref() is None
