# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Common device-fact layer.

The FE's single owner of "what are this GPU's properties": :class:`DeviceInfo`
queries the driver once per fact and caches on the instance, and there is one
instance per CUDA ordinal (see :func:`device_info`), so a GPU's facts are asked
for once and shared. ``Handle.device`` is a ``DeviceInfo``; every other consumer
(the frost engines via ``frost/device.py`` shims) reads the same object rather
than re-querying the driver itself.

The driver queries live here (not in frost) so the front end has one device
concept, not a per-engine introspection stack.
"""

from __future__ import annotations

import functools

from . import _pybind_module

_native_context = getattr(_pybind_module, "_try_ensure_current_context", None)


@functools.lru_cache(maxsize=1)
def _driver():
    """The cuInit'd driver module, or ``None`` when no CUDA device is visible."""
    import cuda.bindings.driver as drv

    if int(drv.cuInit(0)[0]) != 0:
        return None
    return drv


def _ck(err, *rest):
    if int(err) != 0:
        drv = _driver()
        detail = drv.cuGetErrorString(err)[1].decode() if drv is not None else ""
        raise RuntimeError(f"cudnn: CUDA driver error {err}{f': {detail}' if detail else ''}")
    return rest[0] if len(rest) == 1 else None


def is_available() -> bool:
    """True when at least one CUDA device is visible to this process."""
    drv = _driver()
    return drv is not None and int(_ck(*drv.cuDeviceGetCount())) > 0


def device_count() -> int:
    drv = _driver()
    return 0 if drv is None else int(_ck(*drv.cuDeviceGetCount()))


def _device_handle(device: int):
    """``CUdevice`` for an ordinal. Needs only cuInit — creates no context."""
    drv = _driver()
    if drv is None:
        raise NotImplementedError("cudnn: no CUDA device visible")
    count = int(_ck(*drv.cuDeviceGetCount()))
    if not 0 <= device < count:
        raise ValueError(f"cudnn: cuda:{device} does not exist ({count} device(s) visible)")
    return _ck(*drv.cuDeviceGet(device))


@functools.lru_cache(maxsize=1)
def _default_streams() -> frozenset:
    """Handles that name no context: ``cuStreamGetCtx`` answers for the calling
    thread's current context on all of them."""
    drv = _driver()
    if drv is None:
        return frozenset({0})
    return frozenset({0, int(drv.CU_STREAM_LEGACY), int(drv.CU_STREAM_PER_THREAD)})


@functools.lru_cache(maxsize=None)
def _primary_context(device: int):
    """The retained primary context for ``device``, or ``None``. Once per
    ordinal: a retain per execute would grow the usage count without bound."""
    drv = _driver()
    if drv is None:
        return None
    err, handle = drv.cuDeviceGet(device)
    if int(err) != 0:
        return None
    err, primary = drv.cuDevicePrimaryCtxRetain(handle)
    return primary if int(err) == 0 else None


def _runtime_device():
    """Ordinal the CUDA *runtime* holds current on this thread, or ``None``.
    The driver cannot see that slot."""
    try:
        import cuda.bindings.runtime as rt
    except ImportError:
        return None

    err, device = rt.cudaGetDevice()
    return int(device) if int(err) == 0 else None


def ensure_current_context(stream=None, device=None) -> None:
    """Bind the calling thread to the work's context in one native crossing."""
    if _native_context is not None and _native_context(None if stream is None else int(stream), None if device is None else int(device)):
        return
    _ensure_current_context_python(stream, device)


def _ensure_current_context_python(stream=None, device=None) -> None:
    """Bind the context this work runs in to the calling thread.

    A driver-API launch reads the calling thread's context stack; an autograd
    backward runs on a worker whose stack is empty. A real stream names the
    right context; the default-stream handles name none, so ``device`` decides
    there. With no ``device``, a bound context is authoritative and only a cold
    thread is given one (``frost.device.ambient_device``'s rung order).
    Best-effort: what this cannot establish fails at the launch."""
    drv = _driver()
    if drv is None:
        return
    err, cur = drv.cuCtxGetCurrent()
    cur = int(cur) if int(err) == 0 else 0
    stream = 0 if stream is None else int(stream)
    if stream not in _default_streams():
        err, stream_ctx = drv.cuStreamGetCtx(stream)
        if int(err) == 0 and int(stream_ctx) != 0:
            if int(stream_ctx) != cur:
                drv.cuCtxSetCurrent(stream_ctx)
            return
    if device is None:
        if cur:
            return
        device = _runtime_device()
        if device is None:
            return
    elif cur:
        err, cur_device = drv.cuCtxGetDevice()
        if int(err) == 0 and int(cur_device) == int(device):
            return
    primary = _primary_context(int(device))
    if primary is not None:
        drv.cuCtxSetCurrent(primary)


class DeviceInfo:
    """Device facts for one CUDA ordinal, each queried from the driver on first
    access and cached on the instance. One instance per ordinal (via
    :func:`device_info`), so this is the single owner + cache of a GPU's facts.

    Exposes both forms of compute capability that callers consume — the
    ``(major, minor)`` tuple and the packed ``sm_version`` int — plus the SM
    count, the opt-in / oversized per-CTA SMEM ceilings, the L2 size and the name.
    """

    def __init__(self, ordinal: int):
        self.ordinal = int(ordinal)

    @functools.cached_property
    def compute_capability(self) -> tuple[int, int]:
        drv = _driver()
        handle = _device_handle(self.ordinal)
        attr = drv.CUdevice_attribute
        major = int(_ck(*drv.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, handle)))
        minor = int(_ck(*drv.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, handle)))
        return major, minor

    @property
    def sm_version(self) -> int:
        """Packed ``major * 10 + minor`` (derived, so it cannot drift from the tuple)."""
        major, minor = self.compute_capability
        return major * 10 + minor

    @functools.cached_property
    def sm_count(self) -> int:
        drv = _driver()
        return int(_ck(*drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, _device_handle(self.ordinal))))

    @functools.cached_property
    def shared_memory_per_block_optin(self) -> int:
        drv = _driver()
        return int(_ck(*drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, _device_handle(self.ordinal))))

    @functools.cached_property
    def oversized_shared_memory_per_block(self) -> int:
        """Per-CTA SMEM ceiling in the *oversized* carveout (327 KiB vs the 227 KiB
        opt-in limit on SM 10.7), which the part gives by shrinking L1 to 8 kB —
        free for a TMA-fed GEMM. 0 when the driver has no such mode."""
        from . import _env

        drv = _driver()
        # CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK (ordinal 150)
        # arrived in CUDA 13.4. The DRIVER decides whether the mode exists at all,
        # so an older one is 0 by design (not an error). The BINDING only decides
        # how to ask: a cuda-python older than the driver cannot name the enum
        # member but still forwards the bare ordinal, and only bindings old enough
        # to reject an int (they read attrib.value) genuinely cannot make the query
        # -> 0. A real driver failure still raises rather than being masked.
        if _env.driver_version() < 13040:
            return 0
        attr = getattr(drv.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK", 150)
        try:
            return int(_ck(*drv.cuDeviceGetAttribute(attr, _device_handle(self.ordinal))))
        except AttributeError:
            return 0

    @functools.cached_property
    def l2_cache_bytes(self) -> int:
        drv = _driver()
        return int(_ck(*drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, _device_handle(self.ordinal))))

    @functools.cached_property
    def device_name(self) -> str:
        drv = _driver()
        return _ck(*drv.cuDeviceGetName(256, _device_handle(self.ordinal))).split(b"\x00")[0].decode()

    def __repr__(self) -> str:
        return f"DeviceInfo(cuda:{self.ordinal})"


@functools.lru_cache(maxsize=None)
def device_info(ordinal: int) -> DeviceInfo:
    """The :class:`DeviceInfo` for a CUDA ordinal — one per ordinal, so a GPU's
    facts are queried once and shared. Keyed by ordinal, never by Handle: the
    driver caches key on the ordinal, and two handles on the same GPU share facts."""
    return DeviceInfo(int(ordinal))
