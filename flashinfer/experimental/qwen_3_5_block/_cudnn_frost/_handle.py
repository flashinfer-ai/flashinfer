# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-class ``cudnn.Handle``.

The backend ``cudnnHandle_t`` binds a device and carries the current stream, but
on the FE side the handle used to be a bare int with nowhere to hang per-handle
state — so that state accreted as side tables (the ``_handle_to_stream`` dict)
and per-engine device queries (frost's ``current_device()``). ``Handle`` gives
the handle a home for ``{backend_handle, device, stream}``. It is NOT a drop-in
int: the backend handle is extracted explicitly via ``to_backend_handle()`` at
each binding boundary (grep it to trace every handoff), so a Handle that reaches
a binding unconverted fails loudly rather than being silently coerced.
"""

from __future__ import annotations

# The device-fact layer lives in cudnn._device (the FE's single owner of a GPU's
# properties); Handle.device is the DeviceInfo for the handle's ordinal.
from ._device import DeviceInfo, device_info


class Handle:
    """A cuDNN handle as a first-class FE object: owns the backend
    ``cudnnHandle_t`` (``backend_handle``), the stream it runs on (``stream`` —
    the source of truth that used to live in the module-global
    ``_handle_to_stream`` dict), and its device (``device``, lazy, cached per
    ordinal). The naming anticipates the front end BEING "cudnn": this object is
    the ``handle``; the wrapped ``cudnnHandle_t`` is the ``backend_handle``.

    The backend handle is handed to C++ EXPLICITLY via ``to_backend_handle()``
    at the handoff sites (all in ``_pygraph`` and this module — none elsewhere),
    so there is no hidden path from a Handle to the backend: a Handle that reaches
    a C++ binding unconverted fails loudly rather than being silently coerced. The dunders are therefore minimal on purpose —
    NO ``__index__``/``__int__`` (no implicit int coercion) — and
    ``__eq__``/``__hash__``/``__bool__`` are left at the object defaults
    (identity equality, identity hash, always-truthy), which is what the
    surrounding code needs: the handle stays a valid dict key, stays truthy in
    ``if handle:`` guards, and does not raise on ``wrapper.py``'s ``== 'auto'``.
    """

    def __init__(self, backend_handle: int | None, ordinal: int | None = None, stream: int | None = None):
        # ``None`` backend_handle = a destroyed handle (destroy_handle clears it so
        # a reused Handle cannot pass a released cudnnHandle_t back to C++).
        self.backend_handle = None if backend_handle is None else int(backend_handle)
        self._ordinal = ordinal
        # Seeded from the backend's actual stream at create (a fresh handle runs on
        # stream 0), so a python plan and a backend plan on the same handle agree
        # on the stream instead of the python side falling back to torch's current.
        self.stream = stream

    @property
    def device(self) -> DeviceInfo:
        ordinal = self._ordinal
        if ordinal is None:
            from .frost.device import current_device

            ordinal = self._ordinal = current_device()
        return device_info(ordinal)

    def __repr__(self) -> str:
        backend = "None" if self.backend_handle is None else f"0x{self.backend_handle:x}"
        return f"cudnn.Handle(backend_handle={backend}, cuda:{self._ordinal})"


def to_backend_handle(handle):
    """The backend ``cudnnHandle_t`` to hand to the C++ layer: a Handle's
    ``backend_handle``, or ``None`` for the default handle. The ONE explicit
    conversion from the first-class Handle to the int the bindings take.

    A raw backend int is NOT accepted: the Python API creates handles only via
    ``cudnn.create_handle()`` (which returns a Handle), so a bare int is a
    mistake -- wrap a foreign ``cudnnHandle_t`` in
    ``cudnn.Handle(backend_handle, ordinal, stream)`` to give it a device/stream."""
    if handle is None:
        return None
    if isinstance(handle, Handle):
        return handle.backend_handle
    raise TypeError(
        f"expected a cudnn.Handle (from cudnn.create_handle()) or None, got {type(handle).__name__}; "
        "raw backend handles are no longer accepted -- wrap one with cudnn.Handle(backend_handle, ordinal, stream)"
    )
