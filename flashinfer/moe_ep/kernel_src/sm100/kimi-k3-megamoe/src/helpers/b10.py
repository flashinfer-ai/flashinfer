# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-10 levers (every one bit-identical). ``MEGAMOE_B10=<letters>`` selects levers explicitly for A/B ("none" =
all off), at every bucket; unset = the adopted defaults below, each limited to the buckets it was adopted for.

  F  first dispatch round TMA-row-only: each transfer warp's first pull round carries only its TMA row, so round 0
     moves half the bytes and the first wave's rows land sooner (s16-edge lever R, K = 1).
  G  pull-gather router: one 8-byte message per (destination, local expert) replaces the size-row broadcast, the push
     tables and the metadata push; each destination gathers its pool metadata from the sources' sorted runs.
  Q  the router is a PDL dependent of the activation quantize, which triggers right after its own wait: the router's
     local work overlaps the quantize where its grid leaves SMs free (needs G).
  T  the pull loop's dense-row total by a warp reduction instead of a serial SMEM chain.
All four are default on for buckets of up to 1024 tokens per rank. Above that G loses (u2k +2.1%: runs longer than a warp
make the gather serial) and F / T were not measured as wins, so the larger buckets keep build 9's code.
"""
from ..helpers import codegen_options as _kimi_options
import os

DEFAULT = "FGQT"
# largest bucket (max_tokens_per_rank) each default lever applies to; absent = every bucket
DEFAULT_MAX_T = {"F": 1024, "G": 1024, "Q": 1024, "T": 1024}


def explicit():
    raw = _kimi_options.environ.get("MEGAMOE_B10")
    if raw is None or raw.strip() == "":
        return None
    raw = raw.strip()
    return "" if raw == "none" else raw


def on(letter: str, max_tokens_per_rank=None) -> bool:
    chosen = explicit()
    if chosen is not None:
        return letter in chosen
    if letter not in DEFAULT:
        return False
    limit = DEFAULT_MAX_T.get(letter)
    return limit is None or max_tokens_per_rank is None or max_tokens_per_rank <= limit
