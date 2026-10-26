from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: Int32
    offset_k: Int32
    padded_offset_q: Int32
    padded_offset_k: Int32
    seqlen_q: Int32
    seqlen_k: Int32
    has_cu_seqlens_q: cutlass.Constexpr[bool]
    has_cu_seqlens_k: cutlass.Constexpr[bool]

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        tile_m: cutlass.Constexpr[Int32] = 128,
        tile_n: cutlass.Constexpr[Int32] = 128,
    ):
        offset_q = (
            Int32(0) if const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
        )
        offset_k = (
            Int32(0) if const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx]
        )
        padded_offset_q = (
            Int32(0)
            if const_expr(mCuSeqlensQ is None)
            else cute.assume(
                (offset_q + batch_idx * tile_m) // tile_m * tile_m, divby=tile_m
            )
        )
        padded_offset_k = (
            Int32(0)
            if const_expr(mCuSeqlensK is None)
            else cute.assume(
                (offset_k + batch_idx * tile_n) // tile_n * tile_n, divby=tile_n
            )
        )
        seqlen_q = (
            seqlen_q_static
            if const_expr(mCuSeqlensQ is None)
            else mCuSeqlensQ[batch_idx + 1] - offset_q
        )
        seqlen_k = (
            seqlen_k_static
            if const_expr(mCuSeqlensK is None)
            else mCuSeqlensK[batch_idx + 1] - offset_k
        )
        return SeqlenInfoQK(
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            seqlen_q,
            seqlen_k,
            has_cu_seqlens_q=mCuSeqlensQ is not None,
            has_cu_seqlens_k=mCuSeqlensK is not None,
        )

    def offset_batch_Q(
        self,
        mQ: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor:
        if const_expr(not self.has_cu_seqlens_q):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
            return mQ[idx]
        offset_q = self.offset_q if const_expr(not padded) else self.padded_offset_q
        offset = offset_q if const_expr(cute.rank(mQ.shape[0]) == 1) else (0, offset_q)
        idx = (offset,) + (None,) * (cute.rank(mQ) - 1)
        return cute.domain_offset(idx, mQ)

    def offset_batch_K(
        self,
        mK: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        if const_expr(not self.has_cu_seqlens_k):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
            return mK[idx]
        offset_k = self.offset_k if const_expr(not padded) else self.padded_offset_k
        offset_k *= multiple
        idx = (offset_k,) + (None,) * (cute.rank(mK) - 1)
        return cute.domain_offset(idx, mK)
