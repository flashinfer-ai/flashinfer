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

    # V slice this block owns; nonzero only in the sm_80 kernel, which splits V.
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

    @cute.jit
    def left_padded(self, g_prefix_lens: cute.Tensor, chunk: cutlass.Constexpr):
        """Virtual left padding for ``prefix_lens`` (paged recurrent state).

        A sequence whose prefix is not chunk-aligned is processed from its
        aligned absolute position, ``pad`` rows before its first token, so
        every chunk ends on an absolute chunk multiple and checkpoints fall
        on chunk ends. The kernels give the padding rows gate 1 and beta 0
        (identity rows: the state passes them unchanged) and do not store
        their outputs. Returns the padded work item, ``pad`` and the absolute
        position of its first row.
        """
        prefix = cutlass.Int32(g_prefix_lens[self.seq_idx])
        pad = cutlass.Int32(0)
        if self.seq_len != cutlass.Int32(0):
            pad = prefix % cutlass.Int32(chunk)
        padded = WorkDesc(
            seq_idx=self.seq_idx,
            private_q_head_idx=self.private_q_head_idx,
            private_v_head_idx=self.private_v_head_idx,
            tok_offset=self.tok_offset - pad,
            seq_len=self.seq_len + pad,
            tile_idx=self.tile_idx,
            v_slice_idx=self.v_slice_idx,
        )
        return padded, pad, prefix - pad
