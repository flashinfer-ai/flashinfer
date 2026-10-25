# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""The single-head scorers' shapes, without importing a kernel (the CuTe one needs
CuTe DSL 4.7, which the Triton path off SM10x does not).

Keys come in block-table pages of PAGE keys and are scored in tiles of TILE keys.
A key has DIM values, the MMA's K. A decode job is up to MAX_ROWS query rows, the
MMA's N (tcgen05's smallest). CTAS_PER_SM CTAs share an SM when they fit."""

PAGE, TILE, DIM, MAX_ROWS, CTAS_PER_SM = 64, 128, 128, 8, 2
