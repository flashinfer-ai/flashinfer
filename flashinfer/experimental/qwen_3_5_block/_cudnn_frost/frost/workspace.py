# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Per-execute scratch carved out of the CALLER's workspace — shared by every
FROST engine.

The workspace contract (see ``cudnn/engines/base.py``) is that an executor
never allocates: it reports the scratch it needs as ``workspace_bytes`` and
carves that scratch out of the buffer ``execute()`` hands it, so the pointers
are caller-owned and stable across executes — which is what makes a plan safe
to capture in a CUDA graph.

Two halves, matching the two times the engine knows something:

* :class:`WorkspaceLayout` at BUILD time — reserve each region in order and
  keep the offsets; ``size`` is the number the engine reports.
* :class:`Workspace` at EXECUTE time — validate the caller's buffer once, then
  hand out :class:`~cudnn.frost.buffers.DeviceView` views over it. Regions can
  be addressed by their recorded offset (:meth:`Workspace.view`) or dealt
  sequentially (:meth:`Workspace.take`); :meth:`Workspace.remaining` passes the
  unconsumed tail down to a nested carver.

Views are DLPack objects over a raw pointer, so nothing here depends on a
tensor library.
"""

from __future__ import annotations

from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

from . import buffers

# TMA tensormap patches need their 128-byte slot alignment; every other consumer
# is looser, so this is the default rather than a per-call argument.
DEFAULT_ALIGN = 128


def align_up(nbytes: int, align: int = DEFAULT_ALIGN) -> int:
    """Round a region size up to ``align``."""
    return -(-int(nbytes) // int(align)) * int(align)


class WorkspaceLayout:
    """Build-time carve plan: regions reserved in order, each padded to its
    alignment. ``size`` is what the engine reports as ``workspace_bytes``."""

    def __init__(self, *, align: int = DEFAULT_ALIGN):
        self._align = int(align)
        self._size = 0
        self._base_align = int(align)

    def add(self, nbytes: int, *, align: int | None = None) -> int:
        """Reserve ``nbytes`` and return the region's offset from the base."""
        step = self._align if align is None else int(align)
        offset = align_up(self._size, step)
        self._size = offset + align_up(nbytes, step)
        self._base_align = max(self._base_align, step)
        return offset

    @property
    def size(self) -> int:
        return self._size

    @property
    def base_align(self) -> int:
        """Alignment the caller's buffer must satisfy for every reserved region
        to land on its own alignment."""
        return self._base_align


def carve_plan(owner: str, regions) -> "_pybind_module.WorkspaceCarve":
    """Compile a build-time carve: ``[(offset, dtype, shape), ...]``.

    The regions are fixed once :class:`WorkspaceLayout` has run; only the base
    pointer arrives per execute, so one crossing serves them all.
    """
    spec = []
    for offset, dtype, shape in regions:
        code, bits = buffers.DTYPES[dtype]
        spec.append((int(offset), code, bits, [int(extent) for extent in shape]))
    return _pybind_module.WorkspaceCarve(owner, spec)


class Workspace:
    """Execute-time view onto the caller's workspace buffer.

    Validates once — present, contiguous, large enough, aligned — then every
    carve is a bounds-checked view. When ``device`` is specified, also require
    CUDA storage on that ordinal. A buffer that fails any of these raises
    with the required size in the message; it never silently corrupts memory."""

    def __init__(self, buffer, required_bytes: int, owner: str, *, align: int = DEFAULT_ALIGN, device: int | None = None):
        required_bytes = int(required_bytes)
        if buffer is None:
            raise ValueError(
                f"{owner} requires a {required_bytes}-byte workspace but execute() received "
                f"none; allocate graph.get_workspace_size() bytes and pass the buffer to execute()"
            )
        expected_device = device
        ptr, shape, strides, dtype, device = buffers.probe(buffer)
        if expected_device is not None:
            device_attr = getattr(buffer, "device", None)
            device_kind = getattr(device_attr, "type", None)
            dlpack_device = getattr(buffer, "__dlpack_device__", None)
            if device_kind is not None and getattr(device_attr, "index", None) is not None:
                # Tensor metadata is cheaper than invoking a framework's DLPack
                # adapter again after probe; no framework import is needed.
                if device_kind != "cuda":
                    raise ValueError(f"{owner}: workspace must be a CUDA device buffer")
                device = device_attr.index
            elif dlpack_device is not None:
                device_type, device = dlpack_device()
                if int(device_type) != 2:  # kDLCUDA; CPU/pinned-host storage is not scratch.
                    raise ValueError(f"{owner}: workspace must be a CUDA device buffer")
            else:
                # CAI-only producers promise device storage but need not expose a
                # device ordinal. Query pointer metadata; never read device contents.
                # Older cuda-python driver bindings return zero for
                # CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL on every device. The
                # runtime's pointer-attributes struct carries the correct ordinal.
                from cuda.bindings import runtime as cudart

                err, attributes = cudart.cudaPointerGetAttributes(ptr)
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"{owner}: cudaPointerGetAttributes failed: {err}")
                device = attributes.device
            if int(device) != int(expected_device):
                raise ValueError(f"{owner}: workspace must be on cuda:{expected_device}, got cuda:{device}")
        if not buffers.is_contiguous(shape, strides):
            raise ValueError(f"{owner}: the workspace buffer must be contiguous")
        nbytes = buffers.DTYPE_ITEMSIZE[dtype]
        for extent in shape:
            nbytes *= int(extent)
        if nbytes < required_bytes:
            raise ValueError(f"{owner}: needs a {required_bytes}-byte workspace, got {nbytes} bytes " "(size it with graph.get_workspace_size())")
        if ptr % align != 0:
            raise ValueError(f"{owner}: the workspace buffer must be {align}-byte aligned; got 0x{ptr:x}")
        self._init(ptr, nbytes, device, owner, align)

    def _init(self, ptr, nbytes, device, owner, align):
        self._ptr = ptr
        self._device = device
        self._nbytes = nbytes
        self._owner = owner
        self._align = int(align)
        self._offset = 0

    @classmethod
    def over(cls, variant_pack, required_bytes: int, owner: str, *, align: int = DEFAULT_ALIGN) -> "Workspace":
        """The same validated carver, over a workspace the pack already read."""
        required_bytes = int(required_bytes)
        ptr, nbytes = variant_pack.workspace, variant_pack.workspace_bytes
        if not ptr:
            raise ValueError(
                f"{owner} requires a {required_bytes}-byte workspace but execute() received "
                f"none; allocate graph.get_workspace_size() bytes and pass the buffer to execute()"
            )
        # None alone means unmeasured capacity. In particular, a nonnull
        # pointer to an empty view still carries a zero-byte bound.
        if nbytes is not None and nbytes < required_bytes:
            raise ValueError(f"{owner}: needs a {required_bytes}-byte workspace, got {nbytes} bytes (size it with graph.get_workspace_size())")
        if ptr % align != 0:
            raise ValueError(f"{owner}: the workspace buffer must be {align}-byte aligned; got 0x{ptr:x}")
        self = cls.__new__(cls)
        self._init(ptr, nbytes, variant_pack.device, owner, align)
        return self

    @property
    def nbytes(self) -> int | None:
        """Observed capacity, or None for an unmeasured raw address."""
        return self._nbytes

    def view(self, offset: int, dtype: str, shape):
        """The region a :class:`WorkspaceLayout` reserved at ``offset``.

        A carve is the same kind of buffer a caller operand is, so a graph
        hands its kernels one buffer type rather than two.
        """
        count = 1
        for extent in shape:
            count *= int(extent)
        self._check_span(offset, count * buffers.DTYPE_ITEMSIZE[dtype])
        code, bits = buffers.DTYPES[dtype]
        return _pybind_module.make_operand_buffer(self._ptr + offset, list(shape), code, bits, self._device)

    def carve(self, plan):
        """Every region a :func:`carve_plan` describes, in one crossing."""
        return plan.carve(self._ptr, self._nbytes, self._device)

    def take(self, numel: int, dtype: str) -> buffers.DeviceView:
        """The next region dealt sequentially: a 1-D ``numel``-element view."""
        offset = self._offset
        span = int(numel) * buffers.DTYPE_ITEMSIZE[dtype]
        self._check_span(offset, span)
        self._offset = offset + align_up(span, self._align)
        return buffers.DeviceView(self._ptr + offset, (int(numel),), dtype, self._device)

    def remaining(self) -> buffers.DeviceView:
        """The tail no :meth:`take` has claimed, as uint8 — for a nested carver."""
        if self._nbytes is None:
            raise ValueError(
                f"{self._owner}: the workspace was passed as a bare address, so its size is unknown "
                "and the unclaimed tail cannot be measured; pass a sized buffer to execute()"
            )
        # Alignment can consume the final tail after a valid short take.
        offset = min(self._offset, self._nbytes)
        return buffers.DeviceView(self._ptr + offset, (self._nbytes - offset,), "uint8", self._device)

    def _check_span(self, offset: int, span: int) -> None:
        end = int(offset) + int(span)
        if self._nbytes is not None and end > self._nbytes:
            raise ValueError(f"{self._owner}: workspace overrun — region [{offset}, {end}) exceeds the " f"{self._nbytes}-byte buffer (sizing bug)")
