"""Variable-length row semantics shared by the three kernels: the effective length of a row
under the speculative-decode stride and the KV-index compression, and the values gather.

``next_n`` (speculative decode): rows come in groups of ``next_n`` per request, one ``lengths``
entry per request, and row ``i`` of a group sees ``i % next_n`` more tokens than the first:
``length = lengths[i // next_n] - next_n + (i % next_n) + 1``.  ``compress_ratio`` (KV-index
compression, DeepSeek-V4): the row is over compressed blocks, so the token length is divided
by the ratio.  Both are compile-time integers; with the defaults the formula folds to
``lengths[i]``.  Identical to FlashInfer's ``radix`` and ``radix_primitives`` backends.

Values: the selected logits, gathered from the row by the output indices after every index is
written (the caller's barrier); padding slots (-1) get negative infinity.
"""

import cutlass
import cutlass.cute as cute

__all__ = ["effective_length", "gather_values"]


@cute.jit
def effective_length(
    lengths,
    row,
    n_cols: cutlass.Constexpr,
    next_n: cutlass.Constexpr,
    compress_ratio: cutlass.Constexpr,
):
    """The row's valid element count in [0, n_cols] (see the module docstring)."""
    if cutlass.const_expr(next_n == 1 and compress_ratio == 1):
        length = lengths[row]
    else:
        seq = lengths[row // cutlass.Int32(next_n)]
        length = (
            seq
            - cutlass.Int32(next_n)
            + (row % cutlass.Int32(next_n))
            + cutlass.Int32(1)
        ) // cutlass.Int32(compress_ratio)
    if length < 0:
        length = cutlass.Int32(0)
    if length > cutlass.Int32(n_cols):
        length = cutlass.Int32(n_cols)
    return length


@cute.jit
def gather_values(
    dtype,
    row_ptr,
    out_row,
    values_row,
    k: cutlass.Constexpr,
    tidx,
    threads: cutlass.Constexpr,
):
    """``values_row[i] = row[out_row[i]]`` for the k outputs, negative infinity where the index
    is -1.  Call after a barrier that publishes every output index of the row."""
    for i in range(tidx, k, threads):
        idx = out_row[i]
        v = dtype(float("-inf"))
        if idx >= 0:
            v = row_ptr[idx]
        values_row[i] = v
