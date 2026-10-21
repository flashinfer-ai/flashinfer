# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Which GPU a FROST plan is built for — shared by every FROST engine.

The device is recorded on the compiled plan and compared against
:func:`current_device` at execute time, so a plan whose baked constants
describe one GPU fails loudly instead of launching on another.

That comparison is about the LAUNCH, not the buffers. cuDNN's own variant pack
carries pointers, uids and a workspace and no device at all, so an operand's
device is not something the front end has an opinion about.

Device FACTS (compute capability, SM count, the SMEM/L2 ceilings, ...) are NOT
queried here — they live in the common :class:`cudnn._device.DeviceInfo`
(``Handle.device``), queried once and cached; the fact functions below are thin
shims onto it. This module owns only the frost-runtime concerns: which device a
build targets (``current_device`` / ``build_device``) and the primary-context
helper.
"""

from __future__ import annotations

import contextlib
import contextvars

from flashinfer.experimental.qwen_3_5_block._cudnn_frost._device import _ck, _device_handle, _driver, device_info
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._device import device_count, is_available  # noqa: F401 — re-exported for frost callers

# The device a frost BUILD targets, scoped in by build_device() from the
# cudnn.Handle the graph was built with. Every device-derived kernel constant
# (arch, ab_stages, grid_num_clusters, sm_count, the L2/SMEM budgets) resolves
# through current_device(), so setting this once bakes the plan for the handle's
# GPU instead of whatever CUDA device happens to be current at build time.
_build_device: contextvars.ContextVar = contextvars.ContextVar("cudnn_frost_build_device", default=None)


@contextlib.contextmanager
def build_device(device):
    """Scope a frost build to ``device`` (a CUDA ordinal, e.g.
    ``handle.device.ordinal``). ``None`` = no override (fall back to the live
    current device), so a build with no handle keeps the classic behaviour."""
    if device is None:
        yield
        return
    token = _build_device.set(int(device))
    try:
        yield
    finally:
        _build_device.reset(token)


def build_scope_device() -> int | None:
    """The ordinal a surrounding ``build_device()`` pinned, or ``None`` when the
    build is unscoped (no handle asked for a specific GPU)."""
    return _build_device.get()


def ambient_device() -> int:
    """The live CUDA device, IGNORING any ``build_device()`` scope — the device
    cuDNN's backend and cutedsl's compile-target auto-detect actually see.

    A bound driver context wins — it is process-wide and authoritative. Before
    anything has allocated there may be none, and ``cudaSetDevice`` (what
    ``torch.cuda.set_device`` drives) has only moved the runtime's thread-local
    slot, so that is the second rung."""
    drv = _driver()
    if drv is None:
        raise NotImplementedError("cudnn.frost: no CUDA device visible")
    if int(_ck(*drv.cuCtxGetCurrent())) != 0:
        return int(_ck(*drv.cuCtxGetDevice()))
    import cuda.bindings.runtime as rt

    err, index = rt.cudaGetDevice()
    if int(err) != 0:
        raise RuntimeError(f"cudnn.frost: cudaGetDevice failed: {err}")
    return int(index)


def current_device() -> int:
    """CUDA device index a plan built right now would target.

    Inside a ``build_device()`` scope this is the handle's device (so the build
    follows the handle, not the ambient CUDA device); otherwise the live
    :func:`ambient_device`."""
    override = _build_device.get()
    if override is not None:
        return override
    return ambient_device()


def resolve_device(device=None) -> int:
    """Normalize ``None`` / int / ``"cuda:N"`` / ``torch.device`` to an index.

    ``None`` (and a device without an explicit index) means the current device."""
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    kind = getattr(device, "type", None)
    index = getattr(device, "index", None)
    if kind is None:
        kind, _, tail = str(device).partition(":")
        index = int(tail) if tail else None
    if kind != "cuda":
        raise ValueError(f"cudnn.frost: expected a CUDA device, got {device}")
    return current_device() if index is None else int(index)


# --- device facts: thin shims onto the common cudnn._device.DeviceInfo --------
# The queries + per-ordinal cache live there (Handle.device is the same object);
# frost reads through these so its callers need not thread a handle everywhere.
def compute_capability(device: int) -> tuple[int, int]:
    return device_info(device).compute_capability


def multiprocessor_count(device: int) -> int:
    return device_info(device).sm_count


def shared_memory_per_block_optin(device: int) -> int:
    return device_info(device).shared_memory_per_block_optin


def oversized_shared_memory_per_block(device: int) -> int:
    return device_info(device).oversized_shared_memory_per_block


def l2_cache_bytes(device: int) -> int:
    return device_info(device).l2_cache_bytes


def device_name(device: int) -> str:
    return device_info(device).device_name


class device_context:
    """Bind ``device``'s primary context for the enclosing block, then restore
    whatever was bound before. The retain/release pair is refcounted, so this
    composes with a process that already owns the same primary context."""

    def __init__(self, device: int):
        self._device = device
        self._drv = None
        self._handle = None
        self._previous = None

    def __enter__(self):
        self._drv = _driver()
        if self._drv is None:
            raise NotImplementedError("cudnn.frost: no CUDA device visible")
        self._handle = _device_handle(self._device)
        self._previous = _ck(*self._drv.cuCtxGetCurrent())
        _ck(*self._drv.cuCtxSetCurrent(_ck(*self._drv.cuDevicePrimaryCtxRetain(self._handle))))
        return self

    def __exit__(self, *exc):
        _ck(*self._drv.cuCtxSetCurrent(self._previous))
        _ck(*self._drv.cuDevicePrimaryCtxRelease(self._handle))
        return False
