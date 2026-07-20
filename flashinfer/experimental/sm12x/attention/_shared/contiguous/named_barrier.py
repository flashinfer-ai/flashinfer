# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/contiguous/named_barrier.py @ 87134e57 (2026-05-02) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
import enum


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()
    KVConvert = enum.auto()
