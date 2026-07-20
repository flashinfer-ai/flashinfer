# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mask policy shared by the MLA decode kernels and reference path."""

from enum import Enum

import cutlass
import cutlass.cute as cute
from cutlass import Int32


class MaskType(str, Enum):
    """Supported speculative-decode attention masks."""

    CAUSAL = "causal"
    DENSE = "dense"


def normalize_mask_type(mask_type: MaskType | str) -> str:
    """Return the canonical constexpr-safe string for ``mask_type``."""

    try:
        return MaskType(mask_type).value
    except (TypeError, ValueError) as error:
        supported = ", ".join(mask.value for mask in MaskType)
        raise ValueError(
            f"mask_type must be one of ({supported}), got {mask_type!r}"
        ) from error


@cute.jit
def mask_visible_k_length(
    mask_type: cutlass.Constexpr[str],
    seq_len_kv,
    logical_q_idx,
    logical_seq_len_q,
):
    """Return the dense or bottom-right-causal K length for one Q row."""

    seq_len_kv = Int32(seq_len_kv)
    if cutlass.const_expr(mask_type == MaskType.DENSE.value):
        return cute.math.max(seq_len_kv, Int32(0))
    safe_seq_len_q = cute.math.max(Int32(logical_seq_len_q), Int32(1))
    safe_q_idx = cute.math.max(
        Int32(0),
        cute.math.min(Int32(logical_q_idx), safe_seq_len_q - Int32(1)),
    )
    return cute.math.max(
        seq_len_kv - (safe_seq_len_q - Int32(1) - safe_q_idx),
        Int32(0),
    )
