# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Layout arithmetic shared by the SM12x MXFP8 kernels and their host code."""

# Bytes of one 128-row x 4-k32-block chunk of the F8_128x4 scale layout.
SF_CHUNK = 512

# Shared memory available to one CTA on SM120 / SM121.
SMEM_BYTES = 99 * 1024


def ceil_div(a, b):
    return (a + b - 1) // b


def sf_offset(row, kb, kt):
    """Byte offset of the UE8M0 scale of (row, k32 block) in the F8_128x4 layout.

    ``kt`` is the number of 4-block chunks per 128-row group (``ceil(K / 128)``).
    Works on Python ints and on CuTe DSL integers.
    """
    return (
        (row >> 7) * (kt * SF_CHUNK)
        + (kb >> 2) * SF_CHUNK
        + (row & 31) * 16
        + ((row >> 5) & 3) * 4
        + (kb & 3)
    )


def sf_bytes(rows, k):
    """Size of the F8_128x4 scale buffer of a [rows, k] MXFP8 tensor."""
    return ceil_div(rows, 128) * 128 * ceil_div(k // 32, 4) * 4
