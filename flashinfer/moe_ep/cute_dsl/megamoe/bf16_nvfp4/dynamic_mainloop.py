# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility imports for the shared SM100 W4A16 dynamic-N MMA helpers."""

from flashinfer.cute_dsl.sm100_dynamic_mma import (
    bf16_static_idesc,
    compute_non_leader_cta_load_shift,
    issue_dynamic_bf16_mma_tile,
)

__all__ = [
    "bf16_static_idesc",
    "compute_non_leader_cta_load_shift",
    "issue_dynamic_bf16_mma_tile",
]
