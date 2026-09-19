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

"""Device-memory red-zone (canary) helper for kernel-boundary tests (gh #3978).

A kernel can return a numerically correct result while corrupting device
memory that no test tensor covers: the out-of-bounds bytes land in a
neighbouring allocation, in a workspace, or past the end of a staging buffer.
``assert_close`` on the output cannot see that, so a test that only compares
the output is blind to it by construction.  Guarding the output makes the
boundary itself observable.

``allocate_guarded`` returns ONE owning allocation laid out as::

    owner: [ prefix guard | padding | payload | suffix guard | slack ]
            ^0             ^guard_bytes  ^payload_start ^payload_end

* ``payload`` is a view of ``owner`` -- one allocation, one free.  No separate
  guard allocation can be freed while the payload is still in flight, and the
  caller keeping ``payload`` alive keeps its bytes alive.
* ``payload.data_ptr() % alignment == 0`` holds by construction, so wrapping a
  pre-allocated output in guards never turns a legal output into a misaligned
  one.
* ``owner`` is zero-filled before the sentinels are written, so no byte of it
  is uninitialized.  Padding and slack stay 0x00 -- a value that matches
  neither sentinel -- and only the two guard regions are ever written by
  ``reset_guards`` or compared by ``verify_guards``.
* Guards are compared as ``uint8`` bytes, never as the payload's dtype: a NaN
  payload is not equal to itself, so a float comparison would report
  corruption on a perfectly healthy buffer.

Typical use::

    buffer = allocate_guarded((64, 512), torch.float16, "cuda", alignment=16)
    reset_guards()
    flashinfer.silu_and_mul(x, out=buffer.payload)
    verify_guards()  # raises GuardZoneCorruption on a hit

``reset_guards`` -> kernel -> ``verify_guards`` is the whole contract.
``verify_guards`` synchronizes the device before reading the guard bytes back,
which is the happens-before for work the kernel may have enqueued on a stream
this test never recorded an event on.  That sync is also why the helper
belongs at the test boundary: never call it inside a timing region, and never
call ``reset_guards`` between a kernel and the verification meant to catch it.

What a clean ``verify_guards`` does NOT prove, stated so a green test is not
over-read: it means *these guarded bytes* were not written.  A write that
lands far from the payload, a write of the value the guard already holds, and
a write-then-restore are all invisible here.  It complements, and does not
replace, ``compute-sanitizer --tool memcheck`` for a device-wide claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

PREFIX_SENTINEL = 0xA5
SUFFIX_SENTINEL = 0x5A

# 16 B is the widest single access a flashinfer kernel assumes for a
# pre-allocated output (see the vectorised act_and_mul tail); callers whose
# kernel wants more pass ``alignment`` explicitly.
DEFAULT_ALIGNMENT = 16
DEFAULT_GUARD_BYTES = 256


@dataclass(frozen=True)
class GuardCorruption:
    """One guard region whose sentinel bytes no longer match.

    Attributes:
        region: ``"prefix"`` or ``"suffix"``.
        offset: byte offset of the first corrupted byte inside ``region``
            (0 is the region's first byte in memory order).
        owner_offset: byte offset of that same byte inside ``owner``.
        expected: the sentinel byte that should have been there.
        actual: the byte that was read back.
        corrupted_bytes: how many bytes of this region disagree.
        device: the device the owning allocation lives on.
    """

    region: str
    offset: int
    owner_offset: int
    expected: int
    actual: int
    corrupted_bytes: int
    device: torch.device


class GuardZoneCorruption(AssertionError):
    """Raised by :func:`verify_guards`; ``hits`` is the structured report."""

    def __init__(self, hits: Sequence[GuardCorruption]) -> None:
        self.hits: tuple[GuardCorruption, ...] = tuple(hits)
        detail = "\n".join(
            f"  {hit.region} guard byte {hit.offset} (owner offset "
            f"{hit.owner_offset}, {hit.device}) is 0x{hit.actual:02x}, expected "
            f"0x{hit.expected:02x}; {hit.corrupted_bytes} corrupted byte(s) in "
            f"this region"
            for hit in self.hits
        )
        super().__init__(f"guard zone corruption:\n{detail}")


@dataclass(eq=False)
class GuardedBuffer:
    """Handle for one :func:`allocate_guarded` allocation.

    Compared by identity, so the registry and the tests can hold and compare
    handles without tensors entering the comparison.
    """

    owner: torch.Tensor
    payload: torch.Tensor
    prefix_guard: torch.Tensor
    suffix_guard: torch.Tensor
    payload_start: int
    payload_end: int
    guard_bytes: int
    alignment: int

    @property
    def payload_nbytes(self) -> int:
        return self.payload_end - self.payload_start


_REGISTRY: list[GuardedBuffer] = []


def allocate_guarded(
    shape: Sequence[int] | torch.Tensor,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    *,
    alignment: int = DEFAULT_ALIGNMENT,
    guard_bytes: int = DEFAULT_GUARD_BYTES,
) -> GuardedBuffer:
    """Allocate a guarded payload inside a single owning allocation.

    ``shape`` is either a shape or a tensor used as a template, in which case
    ``dtype`` and ``device`` default to the template's.  A template only
    describes what to allocate -- the payload is always a fresh contiguous
    allocation, never an alias of the template -- so a template whose bytes
    are not exactly ``numel * element_size`` contiguous bytes is rejected
    instead of being silently mis-guarded: non-contiguous strides (transposed,
    sliced, expanded), a nonzero storage offset, a view into a larger storage,
    and non-strided layouts (sparse / quantized / nested).

    The guards are armed before this returns, so a bare ``verify_guards()``
    right after the call is already meaningful.  ``guard_bytes`` bounds the
    largest overrun the test intends to catch.
    """
    if isinstance(shape, torch.Tensor):
        template = shape
        if template.layout != torch.strided:
            raise ValueError(
                f"guarded payloads are strided only, got layout {template.layout}"
            )
        if not template.is_contiguous():
            raise ValueError(
                "guarded payloads are contiguous only, got a non-contiguous template"
            )
        if template.storage_offset() != 0:
            raise ValueError(
                "guarded payloads start at storage offset 0, got "
                f"{template.storage_offset()}"
            )
        if (
            template.untyped_storage().nbytes()
            != template.numel() * template.element_size()
        ):
            raise ValueError(
                "template aliases a larger storage, so a guard around it would "
                "not cover the bytes it can reach"
            )
        shape = tuple(template.shape)
        dtype = template.dtype if dtype is None else dtype
        device = template.device if device is None else device
    else:
        shape = tuple(shape)

    if dtype is None or device is None:
        raise ValueError("dtype and device are required when shape is not a tensor")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if guard_bytes <= 0:
        raise ValueError(f"guard_bytes must be positive, got {guard_bytes}")

    payload_nbytes = dtype.itemsize
    for dim in shape:
        payload_nbytes *= dim

    # ``alignment`` extra bytes absorb the padding needed to align the payload
    # at whatever address the allocator hands back; the remainder past the
    # suffix guard is slack of the same size, left at the 0x00 fill.
    device = torch.device(device)
    owner = torch.zeros(
        2 * guard_bytes + alignment + payload_nbytes, dtype=torch.uint8, device=device
    )
    payload_start = guard_bytes + (-(owner.data_ptr() + guard_bytes) % alignment)
    payload_end = payload_start + payload_nbytes
    payload = owner[payload_start:payload_end].view(dtype).view(shape)
    assert payload.data_ptr() == owner.data_ptr() + payload_start
    assert payload.data_ptr() % alignment == 0

    buffer = GuardedBuffer(
        owner=owner,
        payload=payload,
        prefix_guard=owner[:guard_bytes],
        suffix_guard=owner[payload_end : payload_end + guard_bytes],
        payload_start=payload_start,
        payload_end=payload_end,
        guard_bytes=guard_bytes,
        alignment=alignment,
    )
    _write_guards(buffer)
    _REGISTRY.append(buffer)
    return buffer


def live_guarded(
    device: torch.device | str | None = None,
) -> tuple[GuardedBuffer, ...]:
    """Registered guarded buffers, optionally only those on ``device``."""
    if device is None:
        return tuple(_REGISTRY)
    device = _normalize_device(device)
    return tuple(b for b in _REGISTRY if b.payload.device == device)


def release_guarded(buffer: GuardedBuffer) -> None:
    """Forget ``buffer`` so later resets and verifications no longer scan it.

    The allocation is freed once the caller also drops its references: the
    registry holds the only one this module keeps.
    """
    _REGISTRY.remove(buffer)


def clear_guards() -> None:
    """Forget every registered guarded buffer (per-test teardown)."""
    _REGISTRY.clear()


def reset_guards(device: torch.device | str | None = None) -> None:
    """Re-arm every registered guard region, optionally only on ``device``.

    Enqueued on the current stream of each buffer's device, so it is ordered
    against work the caller already enqueued there.  Call it before the kernel
    under test -- never between that kernel and the verification meant to
    catch it.
    """
    for buffer in live_guarded(device):
        _write_guards(buffer)


def verify_guards(device: torch.device | str | None = None) -> None:
    """Read back every registered guard region and raise on any mismatch.

    Synchronizes once per involved device before the byte readback, so guards
    clobbered by work on a stream this test never synchronized with are still
    seen.  Raises :class:`GuardZoneCorruption` carrying the per-region report;
    returns ``None`` when every guarded byte still holds its sentinel.
    """
    buffers = live_guarded(device)
    for buffer_device in {b.payload.device for b in buffers}:
        if buffer_device.type == "cuda":
            torch.cuda.synchronize(buffer_device)
    hits = [hit for buffer in buffers for hit in _scan(buffer)]
    if hits:
        raise GuardZoneCorruption(hits)


def inject_out_of_bounds_write(
    buffer: GuardedBuffer,
    region: str,
    *,
    size: int = 4,
    value: int = 0x00,
    overlap: int = 0,
) -> None:
    """Emulate a ``size``-byte store that misses the payload on ``region``'s side.

    The store is issued through ``buffer.owner``, so it hits real, allocated
    bytes: it can never reach unallocated device memory and can never leave
    the owning allocation.  ``region="prefix"`` writes
    ``[payload_start - size, payload_start)`` and ``region="suffix"`` writes
    ``[payload_end, payload_end + size)`` -- each store ends at, or starts at,
    the payload's logical boundary.

    ``overlap`` is how many of the store's leading bytes land *inside* the
    payload before it runs off the boundary (the misaligned-tail case);
    ``overlap=0`` leaves the payload byte-identical, so a test can assert the
    payload is still numerically correct while the guard reports corruption.
    The ``size - overlap`` bytes past the boundary must fit ``guard_bytes``,
    which is this injection's physical bound.
    """
    if region not in ("prefix", "suffix"):
        raise ValueError(f"region must be 'prefix' or 'suffix', got {region!r}")
    if not 0 <= overlap < size:
        raise ValueError(
            f"need 0 <= overlap < size, got overlap={overlap}, size={size}"
        )
    if size - overlap > buffer.guard_bytes:
        raise ValueError(
            f"a {size - overlap}-byte overrun does not fit the "
            f"{buffer.guard_bytes}-byte {region} guard; raise guard_bytes or "
            "lower size"
        )
    if overlap > buffer.payload_nbytes:
        raise ValueError(
            f"overlap={overlap} runs past the start of the "
            f"{buffer.payload_nbytes}-byte payload"
        )
    if region == "prefix":
        start = buffer.payload_start - size + overlap
    else:
        start = buffer.payload_end - overlap
    buffer.owner[start : start + size].fill_(value)


def _write_guards(buffer: GuardedBuffer) -> None:
    buffer.prefix_guard.fill_(PREFIX_SENTINEL)
    buffer.suffix_guard.fill_(SUFFIX_SENTINEL)


def _scan(buffer: GuardedBuffer) -> list[GuardCorruption]:
    regions = (
        ("prefix", buffer.prefix_guard, PREFIX_SENTINEL, 0),
        ("suffix", buffer.suffix_guard, SUFFIX_SENTINEL, buffer.payload_end),
    )
    hits = []
    for region, guard, sentinel, region_base in regions:
        got = guard.to("cpu")
        mismatched = (got != torch.full_like(got, sentinel)).nonzero().flatten()
        if mismatched.numel() == 0:
            continue
        first = int(mismatched[0])
        hits.append(
            GuardCorruption(
                region=region,
                offset=first,
                owner_offset=region_base + first,
                expected=sentinel,
                actual=int(got[first]),
                corrupted_bytes=int(mismatched.numel()),
                device=buffer.payload.device,
            )
        )
    return hits


def _normalize_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device
