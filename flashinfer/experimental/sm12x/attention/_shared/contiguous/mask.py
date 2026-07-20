# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/contiguous/mask.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from dataclasses import dataclass
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from flashinfer.experimental.sm12x.attention._shared.contiguous import layout_utils
from flashinfer.experimental.sm12x.attention._shared.contiguous.seqlen_info import (
    SeqlenInfoQK,
)


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
    ) -> None:
        del batch_idx, head_idx, mask_seqlen, mask_mod, aux_tensors, fastdiv_mods
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (
            (self.tile_m, self.tile_n)
            if const_expr(not self.swap_AB)
            else (self.tile_n, self.tile_m)
        )
        cS = cute.make_identity_tensor(acc_shape)
        tScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.partition_C(cS), transpose=self.swap_AB
        )
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS),
            transpose=self.swap_AB,
        )
        row_dim = 0 if const_expr(not self.swap_AB) else 1
        col_dim = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][col_dim]
        if n_block < 0:
            n_block = 0
        for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
            row_idx = tScS_mn[r, 0][row_dim] + m_block * self.tile_m
            q_row_idx = (
                row_idx // self.qhead_per_kvhead_packgqa
                if const_expr(self.qhead_per_kvhead_packgqa != 1)
                else row_idx
            )
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                col_idx = (
                    t0ScS_mn[0, c][col_dim] + thr_col_offset + n_block * self.tile_n
                )
                masked = (q_row_idx >= self.seqlen_q) | (col_idx >= self.seqlen_k)
                if const_expr(mask_causal):
                    masked = masked | (
                        col_idx >= q_row_idx + 1 + self.seqlen_k - self.seqlen_q
                    )
                if const_expr(mask_local):
                    anchor = q_row_idx + self.seqlen_k - self.seqlen_q
                    if const_expr(self.window_size_left is not None):
                        masked = masked | (col_idx < anchor - self.window_size_left)
                    if const_expr(self.window_size_right is not None):
                        masked = masked | (
                            col_idx >= anchor + self.window_size_right + 1
                        )
                acc_S_mn[r, c] = cutlass.select_(masked, -Float32.inf, acc_S_mn[r, c])
