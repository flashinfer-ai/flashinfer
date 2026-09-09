from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute


@dataclass
class WorkDesc:
    # coord
    seq_idx: cutlass.Int32
    private_q_head_idx: cutlass.Int32
    private_v_head_idx: cutlass.Int32
    tok_offset: Any

    # shape
    seq_len: cutlass.Int32

    # update by mainloop
    tile_idx: cutlass.Int32

    # Which slice of the value dimension this block owns, when the state's V
    # rows are split across blocks. Zero, and the only value, when a block owns
    # the whole state -- which is every kernel here but the sm_80 one.
    v_slice_idx: cutlass.Int32 = 0

    def v_row_offset(self, d_v: cutlass.Constexpr):
        """First V row this block owns, in elements."""
        return self.v_slice_idx * cutlass.Int32(d_v)

    def is_valid(self, num_seqs: cutlass.Int32):
        return self.seq_idx >= cutlass.Int32(0) and self.seq_idx < num_seqs

    def q_head_idx(self):
        return self.private_q_head_idx

    @staticmethod
    @cute.jit
    def is_gva(num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        return num_v_heads > num_q_heads

    @cute.jit
    def k_head_idx(self, num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        k_head_idx = self.private_v_head_idx
        if WorkDesc.is_gva(num_q_heads, num_v_heads):
            k_head_idx = self.private_q_head_idx
        return k_head_idx

    def v_head_idx(self):
        return self.private_v_head_idx

    @cute.jit
    def o_head_idx(self, num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        o_head_idx = self.private_q_head_idx
        if WorkDesc.is_gva(num_q_heads, num_v_heads):
            o_head_idx = self.private_v_head_idx
        return o_head_idx
