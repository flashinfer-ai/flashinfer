# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-local schedule-section tag for MLA decode work bodies.

MLA decode work methods sometimes need to know which schedule section
(head/loop/tail) a call belongs to.  Rather than depend on the task-scheduling
framework's ``ScheduleStageType``, the MLA schedules pass this small kernel-local
enum explicitly as a compile-time constant, so bodies branch on it with
``cutlass.const_expr(stage == MlaStage.Head)``.
"""

import enum


class MlaStage(enum.Enum):
    """Schedule section of a single MLA decode work call."""

    Head = 0
    Loop = 1
    Tail = 2
