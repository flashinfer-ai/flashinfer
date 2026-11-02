# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Process-global environment facts (CUDA driver / runtime versions).

The FE's single owner of "what CUDA are we running against". These versions are
process-global, not per-device: the installed driver and the linked runtime each
have one version for the whole process, independent of which GPU a handle is
bound to. So they live here, one layer below both the per-ordinal ``DeviceInfo``
(``_device.py``) and the per-handle ``Handle`` (``_handle.py``) — putting a
process-global fact on either would duplicate it per ordinal / per handle.

This mirrors the backend convention: cuDNN exposes its own versions as
argument-less globals (``cudnnGetVersion``, ``cudnnGetCudartVersion``), never off
a handle or a device descriptor. cuDNN's own version stays there
(``cudnn.backend_version()``); this module owns only the CUDA-side versions that
were otherwise re-queried in each engine.

The queries are ~100 ns and off the execute hot path, so the ``lru_cache`` here
is for a single owner returning a stable constant, not for speed.
"""

from __future__ import annotations

import functools

from ._device import _ck, _driver


@functools.lru_cache(maxsize=1)
def driver_version() -> int:
    """Installed CUDA driver version, e.g. ``13020`` for 13.2 (``0`` if no CUDA)."""
    drv = _driver()
    if drv is None:
        return 0
    return int(_ck(*drv.cuDriverGetVersion()))


@functools.lru_cache(maxsize=1)
def runtime_version() -> int:
    """Linked CUDA runtime (cudart) version, e.g. ``13030`` for 13.3.

    ``0`` when ``cuda.bindings`` is absent or the query fails — callers gate on a
    minimum, so an unavailable runtime declines exactly like a too-old one.
    """
    try:
        from cuda.bindings import runtime
    except ImportError:
        return 0
    err, version = runtime.cudaRuntimeGetVersion()
    return int(version) if int(err) == 0 else 0
