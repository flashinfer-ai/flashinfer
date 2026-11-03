# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CuTeDSL unified MXFP4 dense indexer-score kernel.

MXFP4 Q/K are packed E2M1 (two values per byte) with one E8M0 scale per 32
head_dim elements, the scale format of MXFP8.  The kernel runs the MXFP8
producer, scale path and unified indexer epilogue on tcgen05 ``kind::mxf4``,
whose MMA reads K=64 values and two E8M0 scales per row.  Q/K are launched as
their packed bytes (``Uint8``) and recast to E2M1 inside the kernel.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .indexer_score_unified_sm100_mxfp8 import IndexerScoreUnifiedSm100Mxfp8


class IndexerScoreUnifiedSm100Mxfp4(IndexerScoreUnifiedSm100Mxfp8):
    """MXFP4 dense score kernel with unified indexer output semantics."""

    format_name = "MXFP4"
    mma_inst_k = 64

    @staticmethod
    def _qk_tensor(t: cute.Tensor) -> cute.Tensor:
        """Q or K as this kernel reads it: launched as packed ``Uint8`` bytes, read as two E2M1 values per byte."""
        if t.element_type is not cutlass.Uint8:
            raise TypeError(f"SM100 unified indexer MXFP4 expects Q/K launched as Uint8 bytes, got {t.element_type}")
        return cute.recast_tensor(t, cutlass.Float4E2M1FN)

    def _make_data_tiled_mma(self, blockscaled_tiled_mma_qk, cta_group):
        """Tiled MMA that lays out and partitions the Q/K data tiles: the ``kind::mxf4`` MMA itself.

        The unscaled tcgen05 kinds read E2M1 only in the padded f8f6f4 format, not packed.
        """
        return blockscaled_tiled_mma_qk
