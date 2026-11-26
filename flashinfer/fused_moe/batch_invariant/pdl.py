# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic dependent launch: where a kernel lets its dependents start. Every kernel waits for its predecessor before
its first dependent read (``griddepcontrol.wait``); the trigger (``griddepcontrol.launch_dependents``) decides when the next
kernel's CTAs may become resident. The persistent bodies do not trigger: their dependents launch at completion (``TRIGGER =
"none"``); the one-wave kernels take ``TRIGGER_SMALL`` (``none``: completion) right after their wait. The trigger sites below
mark where the other placements would go (``late``: after the CTA's last tile's loads; ``early``: right after the wait). The
placement is baked into every compiled object and the kernel names do not carry it, so it is a constant of the module, not a
runtime or environment setting; this file is in the kernels' source closure (:mod:`flashinfer.batch_invariant.jit`), so a
change to the constant rebuilds every cached object."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

#: The trigger placement of the persistent bodies: ``none`` (the dependents launch at completion). Every body waits before
#: its first dependent read -- the swapped body before its group tables read the plan's counts -- so no placement is
#: excluded for any of them.
TRIGGER = "none"
#: The trigger placement of the one-wave kernels (gather, scatters, quantizers, combine; the plan-side kernels never
#: trigger): ``none`` -- an early trigger would keep the persistent bodies' CTAs resident while these kernels run.
TRIGGER_SMALL = "none"


@cute.jit
def trigger_early():
    """A persistent body's trigger right after its wait (the placement under comparison)."""
    if cutlass.const_expr(TRIGGER == "early"):
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def trigger_late():
    """A persistent body's trigger once the CTA's last loads are issued (the producer warp, after its tile loop)."""
    if cutlass.const_expr(TRIGGER == "late"):
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def trigger_one_wave():
    """A one-wave kernel's trigger, placed right after its wait: only the early placement triggers at all (the dependents launch
    at completion otherwise)."""
    if cutlass.const_expr(TRIGGER_SMALL == "early"):
        cute.arch.griddepcontrol_launch_dependents()
