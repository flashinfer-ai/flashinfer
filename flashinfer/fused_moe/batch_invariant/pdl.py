# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic dependent launch: where a kernel lets its dependents start. Every kernel waits for its predecessor before
its first dependent read (``griddepcontrol.wait``); the trigger (``griddepcontrol.launch_dependents``) decides when the next
kernel's CTAs may become resident. No kernel of this backend triggers: the dependents launch at completion (``TRIGGER =
"none"``), the placement that measured fastest for the MoE chain (an early trigger lets the dependents' CTAs occupy SM slots
during the main work; a trigger after the persistent body's last loads still overlaps its epilogue, which at decode is most
of the kernel). The trigger sites below mark where the other placements would go (``late``: after the CTA's last tile's
loads; ``early``: right after the wait). The placement is baked into every compiled object and the kernel names do not
carry it, so it is a constant of the module, not a runtime or environment setting; this file is in the kernels' source
closure (:mod:`flashinfer.batch_invariant.jit`), so a change to the constant rebuilds every cached object."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

#: The trigger placement of every kernel of the backend: ``none`` (the dependents launch at completion).
TRIGGER = "none"
#: The placements the swapped box form compiles under (asserted by its builder, ``blocks/swap_bf16.py``): the form reads
#: the plan's group counts -- its predecessor's predecessor's output -- before its own wait, which holds while every
#: dependent launches only after its predecessor's wait has returned (the trigger after the loads, or completion). The
#: early placement is kept out as the rule's margin: a trigger that close to the wait is one edit from preceding it.
BOX_FORM_TRIGGERS = ("none", "late")


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
    """A one-wave kernel's trigger: only the early placement triggers at all (the dependents launch at completion otherwise)."""
    if cutlass.const_expr(TRIGGER == "early"):
        cute.arch.griddepcontrol_launch_dependents()
