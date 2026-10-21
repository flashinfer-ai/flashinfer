# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Host-callable CuTe DSL requirement gates of the ``tile_dsl`` primitives (AGENTS.md Rule 7).

THIS MODULE IMPORTS NOTHING FROM ``cutlass`` -- that is its whole point.  The primitive modules (``tma.py``, ``mask.py``,
...) import ``cutlass.experimental`` at module level, so on a DSL below the library floor (the package floor
``nvidia-cutlass-dsl==4.6.2`` has no ``cutlass.experimental``) ``from cudnn.frost.tile_dsl.tma import ...`` dies with a
``ModuleNotFoundError`` before any function in it can run.  A host-side check (an adapter's ``check_support``, a route
check) therefore imports its gate from HERE, before it imports the primitive or its kernel module, and gets a message
naming the installed and the required version instead of the raw import failure.

The version half reads ``cudnn.frost.buffers.cutedsl_state()`` (package metadata, no DSL import); the arch half
(``cutedsl_arch_requirement_error``) imports ``cutlass.base_dsl`` lazily, and only once the version half is satisfied, on
a cc 10.7 part.  ``tile_dsl`` never queries the device: the cc is the API layer's fact, passed in.
"""

from ..buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

#: The PART floor of the gather4 TMA mode: ``cp.async.bulk.tensor ... .tile::gather4`` exists from ``sm_100a`` (cc 10.0) on.  A
#: pre-Blackwell part passes the DSL version gate and would fail only at kernel compilation, so the host gate refuses it by name.
TMA_GATHER4_MIN_CC = (10, 0)


def requirement_error(what, device_cc=None, *, min_cc=None):
    """Message refusing ``what`` (a primitive's dotted name) on the installed DSL or the given part, else ``None``.

    Below ``CUTEDSL_MIN_VERSION`` the message names the installed and the required version (``cutedsl_requirement_error``).
    At or above it, when ``device_cc`` is given: a part below the primitive's own ``min_cc`` (a ``(major, minor)`` tuple --
    the instruction does not exist there, whatever the DSL knows) is refused by name first, then a DSL without the part's
    target (``cutedsl_arch_requirement_error``: an SM107 part needs the public 4.8.0 wheel, the first whose ``Arch`` knows
    ``sm_107a``).  Host Python only: no DSL import on the version and part paths, no device query.
    """
    msg = cutedsl_requirement_error(what)
    if msg is None and device_cc is not None:
        cc = tuple(device_cc)
        if min_cc is not None and cc < tuple(min_cc):
            return f"{what} requires a cc {min_cc[0]}.{min_cc[1]}+ part (the instruction it emits does not exist below it); got cc {cc[0]}.{cc[1]}"
        msg = cutedsl_arch_requirement_error(cc)
    return msg


def tma_gather4_requirement_error(device_cc=None):
    """The host half of the Rule-7 gate of :func:`tile_dsl.tma.tma_gather4` (plus the part floor ``TMA_GATHER4_MIN_CC``);
    ``tma_gather4`` runs the version half again at trace time, before it imports the inline-asm atom.  Import it from THIS
    module on a host path -- ``tma.py`` re-exports the name for callers that already hold the DSL, but importing ``tma``
    below the floor is the failure this gate exists to pre-empt."""
    return requirement_error("tile_dsl.tma.tma_gather4", device_cc, min_cc=TMA_GATHER4_MIN_CC)
