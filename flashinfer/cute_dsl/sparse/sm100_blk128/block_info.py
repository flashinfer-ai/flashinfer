# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_n_block_idx(
        self,
        block_index: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        i: Int32,
        max_i: Optional[Int32] = None,
    ) -> Int32:
        idx = cutlass.min(i, max_i) if const_expr(max_i is not None) else i
        return block_index[batch_idx, head_idx, m_block, idx]
