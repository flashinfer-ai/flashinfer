# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Split-K reduce for the W8A16 GEMM: sums the fp32 partials of the K-slices, applies
the per-tensor scale and the optional per-N bias, and casts to the activation dtype.

    out[m, n] = cast(scale * sum_l partials[l, m, n] + bias[n])

``partials`` is the GEMM kernel's C for ``splits`` slices: [splits, M_padded, N] fp32,
N contiguous (the kernel's C[n, splits * M_padded] column-major; M_padded >= M, the
Hopper kernel pads a slice to whole activation tiles).  One thread handles eight
consecutive n (one 16-byte store, two 16-byte loads per slice).  The slice count is
a compile-time constant so the slice loop unrolls and every slice's loads are in
flight at once (the kernel is latency-bound: 128 CTAs for a 1024 x 256 output)."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

_VEC = 8
_THREADS = 256


class W8A16SplitKReduce:
    def __init__(self, has_bias: bool, splits: int):
        self.has_bias = has_bias
        self.splits = splits

    @cute.jit
    def wrapper(
        self,
        partials_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        slice_vecs: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        """``slice_vecs``: elements between two slices' partials, in 8-element vectors
        (M_padded * N / 8)."""
        # Flat views in units of 8-element vectors; N is a multiple of 8.  The slice
        # stride is spelled as _VEC * slice_vecs so the compiler can see that every
        # slice's vector is 16-byte aligned and keep the 16-byte loads (spelled as an
        # element count it emitted eight 4-byte loads per slice).
        mn_vecs = m * n // _VEC
        partials = cute.make_tensor(
            partials_ptr,
            cute.make_layout(
                (_VEC, mn_vecs, self.splits), stride=(1, _VEC, _VEC * slice_vecs)
            ),
        )
        out = cute.make_tensor(
            out_ptr, cute.make_layout((_VEC, mn_vecs), stride=(1, _VEC))
        )
        bias = cute.make_tensor(
            bias_ptr, cute.make_layout((_VEC, n // _VEC), stride=(1, _VEC))
        )
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        num_blocks = cute.ceil_div(mn_vecs, _THREADS)
        self.kernel(partials, scale, bias, out, mn_vecs, n // _VEC).launch(
            grid=[num_blocks, 1, 1], block=[_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        out: cute.Tensor,
        mn_vecs: cutlass.Int64,
        n_vecs: cutlass.Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = bidx * _THREADS + tidx
        if idx < mn_vecs:
            acc = partials[(None, idx, 0)].load()
            for s in cutlass.range_constexpr(1, self.splits):
                acc = acc + partials[(None, idx, s)].load()
            acc = cutlass.Float32(scale[0]) * acc
            if cutlass.const_expr(self.has_bias):
                acc = acc + bias[(None, idx % n_vecs)].load().to(cutlass.Float32)
            out[(None, idx)].store(acc.to(out.element_type))


class W8A16Sm90TailReduce:
    """Reduce of the Hopper kernel's split tail: the tail tiles (work items from
    ``body_items`` on, in the grouped walk order) each left ``splits`` fp32 partial
    blocks of (tile_m, tile_a) in the workspace, block ``(tile - body) * splits + l``,
    weight rows contiguous.  One thread sums eight consecutive weight rows of one
    activation row of one tile over the slices, applies the per-tensor scale and the
    per-weight-row bias and stores eight 16-bit outputs; rows past the weight or
    activation count (edge tiles) are skipped."""

    def __init__(self, has_bias: bool, tile_m: int, tile_a: int):
        self.has_bias = has_bias
        self.tile_m = tile_m
        self.tile_a = tile_a

    @cute.jit
    def wrapper(
        self,
        workspace_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        weight_tiles: cutlass.Int32,
        act_tiles: cutlass.Int32,
        group: cutlass.Int32,
        body_items: cutlass.Int32,
        tail_tiles: cutlass.Int32,
        splits: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        tile_m, tile_a = self.tile_m, self.tile_a
        vec_per_col = tile_m // _VEC
        blocks = cutlass.Int64(tail_tiles) * cutlass.Int64(splits)
        # Workspace: (vec of 8 rows, row vectors per column, columns, blocks).
        workspace = cute.make_tensor(
            workspace_ptr,
            cute.make_layout(
                (_VEC, vec_per_col, tile_a, blocks),
                stride=(1, _VEC, tile_m, tile_m * tile_a),
            ),
        )
        out = cute.make_tensor(
            out_ptr, cute.make_layout((_VEC, n // _VEC, m), stride=(1, _VEC, n))
        )
        bias = cute.make_tensor(
            bias_ptr, cute.make_layout((_VEC, n // _VEC), stride=(1, _VEC))
        )
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        threads_per_tile = vec_per_col * tile_a
        total = cutlass.Int32(tail_tiles) * threads_per_tile
        num_blocks = cute.ceil_div(total, _THREADS)
        self.kernel(
            workspace,
            scale,
            bias,
            out,
            cutlass.Int32(m),
            cutlass.Int32(n),
            weight_tiles,
            act_tiles,
            group,
            body_items,
            tail_tiles,
            splits,
        ).launch(grid=[num_blocks, 1, 1], block=[_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        workspace: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        out: cute.Tensor,
        m: cutlass.Int32,
        n: cutlass.Int32,
        weight_tiles: cutlass.Int32,
        act_tiles: cutlass.Int32,
        group: cutlass.Int32,
        body_items: cutlass.Int32,
        tail_tiles: cutlass.Int32,
        splits: cutlass.Int32,
    ):
        from .w8a16_fp8_gemm_sm90 import grouped_tile_at

        tile_m, tile_a = self.tile_m, self.tile_a
        vec_per_col = tile_m // _VEC
        threads_per_tile = vec_per_col * tile_a
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = bidx * _THREADS + tidx
        if idx < tail_tiles * threads_per_tile:
            t = idx // threads_per_tile
            r = idx % threads_per_tile
            col = r // vec_per_col  # activation row within the tile
            rvec = r % vec_per_col  # weight-row vector within the tile
            m_tile, n_tile = grouped_tile_at(
                body_items + t, weight_tiles, act_tiles, group
            )
            act_row = n_tile * tile_a + col
            w_row = m_tile * tile_m + rvec * _VEC
            if act_row < m and w_row < n:
                block0 = t * splits
                acc = workspace[(None, rvec, col, block0)].load()
                for s in cutlass.range(1, splits, 1):
                    acc = acc + workspace[(None, rvec, col, block0 + s)].load()
                acc = cutlass.Float32(scale[0]) * acc
                if cutlass.const_expr(self.has_bias):
                    acc = acc + bias[(None, w_row // _VEC)].load().to(cutlass.Float32)
                out[(None, w_row // _VEC, act_row)].store(acc.to(out.element_type))
