# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cudnn.frost: the FROST tile DSL shared by the FROST engines.

Engines themselves live next to their operation (``cudnn.gemm.frost``,
``cudnn.sdpa.fwd``) and reach the graph API through ``cudnn.engines`` — the
same BaseEngine contract every other engine uses. Importing this package has no
effect on ``cudnn.pygraph``.

The one public name at this level is :func:`cutedsl_runtime_available`: the
host-side answer to "can the FROST engines run on this install, for this
part?", which an integration (FlashInfer's cuDNN backend) reads before it
routes work here by default. Its version half reads package metadata only, so
asking costs no DSL import.
"""

import operator

__all__ = ["cutedsl_runtime_available"]


def cutedsl_runtime_available(device_cc=None) -> bool:
    """Whether the installed CuTe DSL can run the FROST engines -- on the part
    ``device_cc`` names, when one is given.

    Two halves, each the engines' own gate (``python/cudnn/AGENTS.md`` Rule 7),
    so the answer is what their ``check_support`` will decide:

    * **version** -- ``nvidia-cutlass-dsl`` is installed and not below
      :data:`cudnn.frost.buffers.CUTEDSL_MIN_VERSION` (4.7.0). This package's
      own dependency floor (4.6.2) sits below it on purpose, so an install that
      satisfies pip can still be one every FROST row declines. Read from package
      metadata (:func:`cudnn.frost.buffers.cutedsl_state` /
      :func:`~cudnn.frost.buffers.cutedsl_too_old`) without importing the DSL;
      internal DSL builds count as new enough here, as they do in the engines.
    * **target** -- with ``device_cc``, a ``(major, minor)`` compute capability
      such as ``torch.cuda.get_device_capability()``, the DSL must also know the
      part: cc 10.7 needs a DSL whose ``Arch`` has ``sm_107a`` (the public 4.8.0
      wheel is the first), so **the floor is 4.7.0 everywhere and 4.8.0 on
      cc 10.7** (:func:`cudnn.frost.buffers.cutedsl_arch_requirement_error`).
      This half imports the DSL once, on a cc 10.7 part only.

    A device-blind call (``device_cc=None``) answers the version half alone:
    True there says nothing about a part the DSL has no target for, so a caller
    that routes by GPU passes the GPU it routes to -- on a cc 10.7 part with a
    4.7.x DSL the blind form reads True while every FROST row declines.

    False when the DSL is absent or below the floor (no exception); a
    ``device_cc`` that is not a pair of integers raises ``TypeError``.
    """
    from .buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        return False
    if device_cc is None:
        return True
    try:
        cc = tuple(operator.index(x) for x in device_cc)  # integers only: a 10.5 must not read as cc 10
    except TypeError as e:
        raise TypeError(f"device_cc must be a (major, minor) compute capability, got {device_cc!r}") from e
    if len(cc) != 2:
        raise TypeError(f"device_cc must be a (major, minor) compute capability, got {device_cc!r}")
    return cutedsl_arch_requirement_error(cc) is None
