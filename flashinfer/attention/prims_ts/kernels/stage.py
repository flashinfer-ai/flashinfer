# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-local schedule-section tag for FMHA decode/context work bodies.

Some FMHA work methods need to know which schedule section (head/loop/tail) a
call belongs to.  Rather than depend on the task-scheduling framework's
``ScheduleStageType``, the FMHA schedules pass this small kernel-local enum
explicitly as a compile-time constant (the ``section`` work argument), so bodies
branch on it with ``cutlass.const_expr(section == FmhaStage.Loop)``.
"""

import enum


class FmhaStage(enum.Enum):
    """Schedule section of a single FMHA decode/context work call."""

    Head = 0
    Loop = 1
    Tail = 2
