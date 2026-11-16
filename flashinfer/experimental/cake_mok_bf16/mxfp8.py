# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch MoK MXFP8 quantization for caller-prequantized routed weights.

Recipe (MoK ``mxfp8_quantize``): 32 consecutive elements along the quantized
(K, last) dimension share one UE8M0 scale byte,
``round_up(max(amax * 0.002232142857f, 1e-12f))`` saturated to 254; data is
``cvt.rn.satfinite.e4m3(x * 2**(127 - byte))``. Scale tiles cover 128 rows x
128 K, stored as ``[rows // 128, K // 128, 32, 16]`` bytes with byte
``(row % 32) * 16 + (row // 32) * 4 + k_block`` (tensor-core scale layout);
experts are concatenated along the first dimension.
"""

import functools
import warnings

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the PyTorch path below quantizes the same bytes
    triton = None

# The oldest Triton release whose Gluon compiles the kernel below to the same
# bytes (verified with 3.5.1, 3.6.0, 3.7.1 and 3.8.0); older releases use the
# Triton dual kernel.
_GLUON_MIN_TRITON = (3, 5)
if triton is not None and (
    tuple(int(part) for part in triton.__version__.split(".")[:2]) >= _GLUON_MIN_TRITON
):
    try:
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl
    except ImportError:
        gluon = None
else:
    gluon = None

_BLOCK = 32
# The FP32 ``1 / 448`` factor of the recipe.
_SCALE_FACTOR = 0.002232142857
# Float32 temporaries of one quantization pass stay below this many bytes:
# experts are quantized together in chunks of this size.
_CHUNK_BYTES = 1 << 30


@functools.cache
def _scale_factor(device):
    """The FP32 ``1 / 448`` factor of the recipe, created once per device."""
    return torch.tensor(_SCALE_FACTOR, dtype=torch.float32, device=device)


def _e8m0_round_up(scale):
    """FP32 -> UE8M0 byte, rounding toward +inf and saturating to 254."""
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    byte = exponent + (mantissa != 0).to(torch.int32)
    return byte.clamp(max=254).to(torch.uint8)


def _quantize_rows(x):
    """Quantize ``[E, rows, cols]`` along ``cols``: ``(E4M3 data, [E, rows, cols / 32] bytes)``."""
    experts, rows, cols = x.shape
    blocks = x.float().reshape(experts, rows, cols // _BLOCK, _BLOCK)
    amax = blocks.abs().amax(-1)
    scale = torch.clamp(amax * _scale_factor(x.device), min=1e-12)
    byte = _e8m0_round_up(scale)
    inverse = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    data = (blocks * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return data.reshape(experts, rows, cols), byte


def _scale_tiles(byte):
    """[E, R, K/32] bytes -> [E * R/128, K/128, 32, 16] tensor-core scale tiles."""
    experts, rows, kb = byte.shape
    tiles = byte.reshape(
        experts, rows // 128, 4, 32, kb // 4, 4
    )  # [e, rb, row//32, row%32, kt, kb%4]
    return tiles.permute(0, 1, 4, 3, 2, 5).reshape(
        experts * rows // 128, kb // 4, 32, 16
    )


def _quantize_layout_torch(matrices, transposed):
    """One layout of ``[E, M, N]`` matrices with PyTorch operations: experts are
    quantized together in chunks that bound the FP32 temporaries to
    ``_CHUNK_BYTES``; the result does not depend on the chunking."""
    experts, rows, cols = matrices.shape
    chunk = max(1, _CHUNK_BYTES // (rows * cols * 4))
    shape = (experts, cols, rows) if transposed else (experts, rows, cols)
    data = torch.empty(shape, dtype=torch.float8_e4m3fn, device=matrices.device)
    scales = []
    for start in range(0, experts, chunk):
        part = matrices[start : start + chunk]
        source = part.transpose(1, 2).contiguous() if transposed else part
        data[start : start + chunk], byte = _quantize_rows(source)
        scales.append(_scale_tiles(byte))
    return data, torch.cat(scales)


if triton is not None:

    @triton.jit
    def _quantize_tile_kernel(
        x_ptr,
        data_ptr,
        scale_ptr,
        stride_e,
        stride_r,
        stride_c,
        row_tiles,
        col_tiles,
        scale_factor,
    ):
        """One 128 x 128 tile of the quantized layout (source element ``(e, r, c)``
        at ``e * stride_e + r * stride_r + c * stride_c``): E4M3 data and the
        tile's 512 scale bytes. NaN propagates through the block maximum and the
        clamp as in :func:`_quantize_rows`."""
        tile = tl.program_id(0)
        expert = (tile // row_tiles).to(tl.int64)
        row_tile = tile % row_tiles
        col_tile = tl.program_id(1)
        rows = row_tile * 128 + tl.arange(0, 128)
        cols = col_tile * 128 + tl.arange(0, 128)
        source = (
            expert * stride_e
            + rows[:, None].to(tl.int64) * stride_r
            + cols[None, :].to(tl.int64) * stride_c
        )
        values = tl.reshape(tl.load(x_ptr + source).to(tl.float32), (128, 4, 32))
        amax = tl.max(tl.abs(values), axis=2)
        has_nan = tl.max((values != values).to(tl.int32), axis=2) > 0
        byte, inverse = _block_bytes(
            tl.where(has_nan, float("nan"), amax), scale_factor
        )
        out_cols = col_tiles * 128
        target = (
            expert * row_tiles * 128 * out_cols
            + rows[:, None].to(tl.int64) * out_cols
            + cols[None, :]
        )
        tl.store(
            data_ptr + target,
            tl.reshape(_clamp_e4m3(values * inverse[:, :, None]), (128, 128)),
        )
        local = tl.arange(0, 128)
        tile_base = ((expert * row_tiles + row_tile) * col_tiles + col_tile) * 512
        scale_offsets = (
            tile_base
            + ((local % 32) * 16 + (local // 32) * 4)[:, None]
            + tl.arange(0, 4)[None, :]
        )
        tl.store(scale_ptr + scale_offsets, byte.to(tl.uint8))

    @triton.jit
    def _block_bytes(amax, scale_factor):
        """Block maxima -> UE8M0 bytes and the exact ``2**(127 - byte)`` factors;
        a NaN maximum propagates."""
        scale = amax * scale_factor
        scale = tl.where(scale != scale, scale, tl.maximum(scale, 1e-12))
        bits = scale.to(tl.int32, bitcast=True)
        byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        byte = tl.minimum(byte, 254)
        return byte, ((254 - byte) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _clamp_e4m3(scaled):
        return tl.where(
            scaled != scaled, scaled, tl.minimum(tl.maximum(scaled, -448.0), 448.0)
        ).to(tl.float8e4nv)

    @triton.jit
    def _quantize_tile_both_kernel(
        x_ptr,
        data_ptr,
        scale_ptr,
        data_t_ptr,
        scale_t_ptr,
        row_tiles,
        col_tiles,
        scale_factor,
    ):
        """Both layouts of one 128 x 128 tile of a contiguous ``[E, M, N]`` matrix
        from a single read: the tile of :func:`_quantize_tile_kernel` for ``x`` and
        the one for ``x.transpose(-1, -2)``, byte for byte (each output byte depends
        only on its own 32-element block, which lies inside the tile)."""
        tile = tl.program_id(0)
        expert = (tile // row_tiles).to(tl.int64)
        row_tile = tile % row_tiles
        col_tile = tl.program_id(1)
        rows = row_tile * 128 + tl.arange(0, 128)
        cols = col_tile * 128 + tl.arange(0, 128)
        m = row_tiles * 128
        n = col_tiles * 128
        source = expert * m * n + rows[:, None].to(tl.int64) * n + cols[None, :]
        values = tl.load(x_ptr + source).to(tl.float32)
        nan = (values != values).to(tl.int32)
        local = tl.arange(0, 128)
        scale_rows = ((local % 32) * 16 + (local // 32) * 4)[:, None] + tl.arange(0, 4)[
            None, :
        ]

        # x: 32-element blocks along N.
        blocks = tl.reshape(values, (128, 4, 32))
        amax = tl.max(tl.abs(blocks), axis=2)
        has_nan = tl.max(tl.reshape(nan, (128, 4, 32)), axis=2) > 0
        byte, inverse = _block_bytes(
            tl.where(has_nan, float("nan"), amax), scale_factor
        )
        data = _clamp_e4m3(blocks * inverse[:, :, None])
        tl.store(data_ptr + source, tl.reshape(data, (128, 128)))
        tile_base = ((expert * row_tiles + row_tile) * col_tiles + col_tile) * 512
        tl.store(scale_ptr + tile_base + scale_rows, byte.to(tl.uint8))

        # x.transpose(-1, -2): 32-element blocks along M.
        blocks_t = tl.reshape(values, (4, 32, 128))
        amax_t = tl.max(tl.abs(blocks_t), axis=1)
        has_nan_t = tl.max(tl.reshape(nan, (4, 32, 128)), axis=1) > 0
        byte_t, inverse_t = _block_bytes(
            tl.where(has_nan_t, float("nan"), amax_t), scale_factor
        )
        data_t = _clamp_e4m3(blocks_t * inverse_t[:, None, :])
        target_t = expert * m * n + cols[:, None].to(tl.int64) * m + rows[None, :]
        tl.store(data_t_ptr + target_t, tl.trans(tl.reshape(data_t, (128, 128))))
        tile_base_t = ((expert * col_tiles + col_tile) * row_tiles + row_tile) * 512
        tl.store(scale_t_ptr + tile_base_t + scale_rows, tl.trans(byte_t).to(tl.uint8))


if gluon is not None:

    @gluon.jit
    def _nan_max(a, b):
        return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @gluon.jit
    def _gl_block_bytes(amax, scale_factor):
        """:func:`_block_bytes` in Gluon (a Gluon kernel cannot call ``triton.jit``
        functions)."""
        scale = amax * scale_factor
        scale = gl.where(scale != scale, scale, gl.maximum(scale, 1e-12))
        bits = scale.to(gl.int32, bitcast=True)
        byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(gl.int32)
        byte = gl.minimum(byte, 254)
        return byte, ((254 - byte) << 23).to(gl.float32, bitcast=True)

    @gluon.jit
    def _gl_quantize_tile(
        tensors,
        expert,
        row_tile,
        col_tile,
        scale_factor,
        M: gl.constexpr,
        N: gl.constexpr,
        LAYOUTS: gl.constexpr,
    ):
        """Both layouts of one 128 x 128 tile of a contiguous ``[E, M, N]`` matrix,
        byte for byte those of :func:`_quantize_tile_both_kernel`.

        The tile is viewed as ``[4, 32, 4, 32]`` (row = ``rb * 32 + rr``, column =
        ``kb * 32 + cc``), so every 32-element block of either layout lies along
        one axis, held in registers and lanes of one warp: both block maxima are
        warp shuffles and no tile goes through shared memory. ``TILE`` loads 8
        consecutive columns per thread and stores the E4M3 tile with 8-byte
        vectors; ``TILE_T`` owns the same elements with 4 consecutive rows first,
        so the transposed tile is stored with 4-byte vectors. Only the 512-byte
        scale tiles of each layout are regrouped through shared memory
        (``SCALES`` / ``SCALES_T``) into 4-byte stores. The NaN-propagating block
        maximum equals the explicit NaN check of the Triton kernel, and the
        saturating E4M3 conversion equals its +-448 clamp. ``tensors`` holds the
        source, data, scales, data_t and scales_t pointers; ``LAYOUTS`` holds
        ``TILE``, ``TILE_T``, ``SCALES`` and ``SCALES_T`` of :func:`_gluon_layouts`.
        """
        x_ptr, data_ptr, scale_ptr, data_t_ptr, scale_t_ptr = tensors
        TILE: gl.constexpr = LAYOUTS[0]
        TILE_T: gl.constexpr = LAYOUTS[1]
        SCALES: gl.constexpr = LAYOUTS[2]
        SCALES_T: gl.constexpr = LAYOUTS[3]
        L3: gl.constexpr = gl.SliceLayout(3, TILE)
        L2: gl.constexpr = gl.SliceLayout(2, L3)
        rb = gl.arange(0, 4, layout=gl.SliceLayout(1, L2))[:, None, None, None]
        rr = gl.arange(0, 32, layout=gl.SliceLayout(0, L2))[None, :, None, None]
        kb = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
        kb = kb[None, None, :, None]
        L1: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, TILE))
        cc = gl.arange(0, 32, layout=gl.SliceLayout(0, L1))[None, None, None, :]
        row = rb * 32 + rr
        col = kb * 32 + cc
        # Tile corners: int64 where (M - 128) * N or (N - 128) * M can wrap int32
        # (``expert`` is int64); in-tile offsets stay int32 (see _gluon_fits).
        if M * N >= 2**31:
            row_tile = row_tile.to(gl.int64)
            col_tile = col_tile.to(gl.int64)
        source = expert * (M * N) + (row_tile * 128) * N + col_tile * 128
        offsets = row * N + col
        values = gl.load(x_ptr + source + offsets).to(gl.float32)
        magnitude = gl.abs(values)

        # x: 32-element blocks along N (axis 3).
        byte, inverse = _gl_block_bytes(gl.reduce(magnitude, 3, _nan_max), scale_factor)
        data = (values * inverse[:, :, :, None]).to(gl.float8e4nv)
        gl.store(data_ptr + source + offsets, data)
        # x.transpose(-1, -2): 32-element blocks along M (axis 1).
        byte_t, inverse_t = _gl_block_bytes(
            gl.reduce(magnitude, 1, _nan_max), scale_factor
        )
        data_t = (values * inverse_t[:, None, :, :]).to(gl.float8e4nv)
        target_t = expert * (M * N) + (col_tile * 128) * M + row_tile * 128
        offsets_t = gl.convert_layout(col * M + row, TILE_T)
        gl.store(data_t_ptr + target_t + offsets_t, gl.convert_layout(data_t, TILE_T))

        # Scale tiles: byte (row % 32) * 16 + (row // 32) * 4 + column block.
        S2: gl.constexpr = gl.SliceLayout(2, SCALES)
        s_rb = gl.arange(0, 4, layout=gl.SliceLayout(1, S2))[:, None, None]
        s_rr = gl.arange(0, 32, layout=gl.SliceLayout(0, S2))[None, :, None]
        s_kb = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, SCALES)))
        tile_base = ((expert * (M // 128) + row_tile) * (N // 128) + col_tile) * 512
        gl.store(
            scale_ptr + tile_base + (s_rr * 16 + s_rb * 4 + s_kb[None, None, :]),
            gl.convert_layout(byte.to(gl.uint8), SCALES),
        )
        T2: gl.constexpr = gl.SliceLayout(2, SCALES_T)
        t_rb = gl.arange(0, 4, layout=gl.SliceLayout(1, T2))[:, None, None]
        t_kb = gl.arange(0, 4, layout=gl.SliceLayout(0, T2))[None, :, None]
        t_cc = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, SCALES_T)))
        tile_base_t = ((expert * (N // 128) + col_tile) * (M // 128) + row_tile) * 512
        gl.store(
            scale_t_ptr + tile_base_t + (t_cc[None, None, :] * 16 + t_kb * 4 + t_rb),
            gl.convert_layout(byte_t.to(gl.uint8), SCALES_T),
        )

    @gluon.jit
    def _quantize_both_gluon_kernel(
        gate,
        up,
        down,
        scale_factor,
        M: gl.constexpr,  # constexpr M, N keep the in-tile offsets in int32
        N: gl.constexpr,
        ROUTED: gl.constexpr,
        LAYOUTS: gl.constexpr,
    ):
        """Both layouts of ``gate`` (``[E, M, N]``), or with ``ROUTED`` of ``gate``
        and ``up`` (``[E, M, N]``) and ``down`` (``[E, N, M]``) in one launch: one
        program per 128 x 128 tile over an ``(E * tiles, 1 or 3 matrices)`` grid.
        Each matrix argument is a (source, data, scales, data_t, scales_t) tuple."""
        tiles: gl.constexpr = (M // 128) * (N // 128)
        tile = gl.program_id(0)
        expert = (tile // tiles).to(gl.int64)
        local = tile % tiles
        matrix = gl.program_id(1)
        if ROUTED and matrix == 2:  # down: row tiles of N, column tiles of M
            row_tile, col_tile = local // (M // 128), local % (M // 128)
            _gl_quantize_tile(
                down, expert, row_tile, col_tile, scale_factor, N, M, LAYOUTS
            )
        else:
            row_tile, col_tile = local // (N // 128), local % (N // 128)
            if ROUTED and matrix == 1:
                _gl_quantize_tile(
                    up, expert, row_tile, col_tile, scale_factor, M, N, LAYOUTS
                )
            else:
                _gl_quantize_tile(
                    gate, expert, row_tile, col_tile, scale_factor, M, N, LAYOUTS
                )

    @functools.cache
    def _gluon_layouts():
        """``[rb, rr, kb, cc]`` tile layouts (registers, lanes, warps), the two
        ``[4, 32, 4]`` scale-tile layouts and the number of warps."""
        rb, rr, kb, cc = [[1 if i == d else 0 for i in range(4)] for d in range(4)]

        def basis(axis, value):
            return [value * a for a in axis]

        columns = [basis(cc, 1), basis(cc, 2), basis(cc, 4)]
        rows = [basis(rr, 1), basis(rr, 2)]
        lanes = [basis(cc, 8), basis(cc, 16), basis(rr, 4), basis(rr, 8), basis(rr, 16)]
        # 16 warps, one 32 x 32 block pair (rb, kb) each; faster than 8 or 4 on R200.
        warps = [basis(rb, 1), basis(rb, 2), basis(kb, 1), basis(kb, 2)]
        shape = [4, 32, 4, 32]
        tile, tile_t = (
            gl.DistributedLinearLayout(
                reg_bases=registers,
                lane_bases=lanes,
                warp_bases=warps,
                block_bases=[],
                shape=shape,
            )
            for registers in (columns + rows, rows + columns)
        )
        # [rb, rr, kb] bytes, registers kb then rb: ``rr * 16 + rb * 4 + kb``
        # [rb, kb, cc] bytes, registers rb then kb: ``cc * 16 + kb * 4 + rb``
        idle = [[0, 0, 0]] * len(warps)
        scales = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1], [0, 0, 2], [1, 0, 0], [2, 0, 0]],
            lane_bases=[[0, 1 << i, 0] for i in range(5)],
            warp_bases=idle,
            block_bases=[],
            shape=[4, 32, 4],
        )
        scales_t = gl.DistributedLinearLayout(
            reg_bases=[[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]],
            lane_bases=[[0, 0, 1 << i] for i in range(5)],
            warp_bases=idle,
            block_bases=[],
            shape=[4, 4, 32],
        )
        return (tile, tile_t, scales, scales_t), 2 ** len(warps)


def _quantize_both_ok(x):
    """Whether :func:`_quantize_both` takes ``x``: a contiguous CUDA BF16
    ``[E, M, N]`` tensor with M and N positive multiples of 128."""
    return (
        triton is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.numel() > 0
        and x.shape[-2] % 128 == 0
        and x.shape[-1] % 128 == 0
    )


def _gluon_fits(matrices):
    """The Gluon kernel keeps in-tile offsets (below ``128 * max(M, N)``) in int32."""
    return 128 * max(matrices.shape[-2:]) < 2**31


def _alloc_both(matrices):
    experts, rows, cols = matrices.shape
    options = dict(device=matrices.device)
    data = torch.empty((experts, rows, cols), dtype=torch.float8_e4m3fn, **options)
    data_t = torch.empty((experts, cols, rows), dtype=torch.float8_e4m3fn, **options)
    scales = torch.empty(
        (experts * rows // 128, cols // 128, 32, 16), dtype=torch.uint8, **options
    )
    scales_t = torch.empty(
        (experts * cols // 128, rows // 128, 32, 16), dtype=torch.uint8, **options
    )
    return data, scales, data_t, scales_t


def _quantize_both_gluon(matrices, outputs):
    """Both layouts of contiguous ``[E, M, N]`` matrices (or of routed gate, up
    ``[E, M, N]`` and down ``[E, N, M]``) into ``outputs`` in one Gluon launch."""
    assert len(matrices) in (1, 3)
    experts, rows, cols = matrices[0].shape
    layouts, num_warps = _gluon_layouts()
    args = [(m, *out) for m, out in zip(matrices, outputs, strict=True)]
    args += args[:1] * (3 - len(args))  # unused slots of the single-matrix form
    grid = (experts * (rows // 128) * (cols // 128), len(matrices))
    _quantize_both_gluon_kernel[grid](
        *args,
        _SCALE_FACTOR,
        rows,
        cols,
        len(matrices) == 3,
        layouts,
        num_warps=num_warps,
    )


# (count, E, M, N) of the matrices whose Gluon kernel has compiled and run.
_gluon_shapes: set[tuple[int, ...]] = set()


def _quantize_both(*matrices):
    """Both layouts of matrices that pass :func:`_quantize_both_ok`, as one tuple
    ``(data, scales, data_t, scales_t)`` per matrix: one Gluon launch for one
    matrix or for routed gate, up and down, else the Triton dual kernel per
    matrix. If the first Gluon launch of a shape fails to build or compile,
    Gluon is disabled for the process and the Triton kernel runs instead."""
    global gluon
    if gluon is not None and _gluon_fits(matrices[0]):
        outputs = [_alloc_both(m) for m in matrices]
        key = (len(matrices), *matrices[0].shape)
        try:
            _quantize_both_gluon(matrices, outputs)
        except Exception as error:  # Gluon is experimental: any build error falls back
            if key in _gluon_shapes:  # only the first launch of a shape falls back
                raise
            gluon = None
            warnings.warn(
                f"Gluon MXFP8 quantizer disabled, using the Triton kernel: {error!r}",
                stacklevel=2,
            )
        else:
            _gluon_shapes.add(key)
            return outputs
    return [_quantize_both_triton(m) for m in matrices]


def _quantize_both_triton(matrices):
    """Both layouts of contiguous ``[E, M, N]`` matrices in one Triton launch."""
    experts, rows, cols = matrices.shape
    data, scales, data_t, scales_t = _alloc_both(matrices)
    grid = (experts * (rows // 128), cols // 128)
    _quantize_tile_both_kernel[grid](
        matrices,
        data,
        scales,
        data_t,
        scales_t,
        rows // 128,
        cols // 128,
        _SCALE_FACTOR,
        num_warps=4,  # 0.38 ms vs 0.75 ms (8) / 0.63 ms (2) per [36, 2048, 4096] on R200
    )
    return data, scales, data_t, scales_t


def _quantize_layout_triton(matrices, transposed):
    """One layout of ``[E, M, N]`` matrices in one fused kernel launch."""
    experts, rows, cols = matrices.shape
    stride_e, stride_m, stride_n = matrices.stride()
    if transposed:
        rows, cols = cols, rows
        stride_m, stride_n = stride_n, stride_m
    data = torch.empty(
        (experts, rows, cols), dtype=torch.float8_e4m3fn, device=matrices.device
    )
    scales = torch.empty(
        (experts * rows // 128, cols // 128, 32, 16),
        dtype=torch.uint8,
        device=matrices.device,
    )
    grid = (experts * (rows // 128), cols // 128)
    _quantize_tile_kernel[grid](
        matrices,
        data,
        scales,
        stride_e,
        stride_m,
        stride_n,
        rows // 128,
        cols // 128,
        _SCALE_FACTOR,
        num_warps=8,
    )
    return data, scales


def mxfp8_quantize(x, return_normal=True, return_transposed=True):
    """MoK ``mxfp8_quantize`` for BF16 ``[M, N]`` or ``[E, M, N]`` matrices.

    Returns ``(fp8, scales, fp8_t, scales_t)``; entries not requested are
    ``None``. ``fp8`` is E4M3 with the input shape and ``scales`` is uint8
    ``[E * M // 128, N // 128, 32, 16]``; the transposed pair quantizes
    ``x.transpose(-1, -2)`` the same way. M and N must be multiples of 128.
    Contiguous CUDA tensors that request both layouts are quantized by one
    Gluon (or, without Gluon, Triton) kernel that reads each tile once; a
    single layout (or a non-contiguous input) uses one Triton kernel per
    layout. Without Triton (or off CUDA) PyTorch operations produce the same
    bytes.
    """
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        raise ValueError("x must be a BF16 tensor")
    if x.ndim not in (2, 3) or any(size <= 0 for size in x.shape):
        raise ValueError("x must have positive shape (M, N) or (E, M, N)")
    if x.shape[-2] % 128 or x.shape[-1] % 128:
        raise ValueError("x M and N dimensions must be divisible by 128")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("At least one quantized layout must be requested")
    matrices = x.unsqueeze(0) if x.ndim == 2 else x
    if return_normal and return_transposed and _quantize_both_ok(matrices):
        ((data, scales, data_t, scales_t),) = _quantize_both(matrices)
        if x.ndim == 2:
            data, data_t = data[0], data_t[0]
        return data, scales, data_t, scales_t
    fused = triton is not None and x.is_cuda
    quantize = _quantize_layout_triton if fused else _quantize_layout_torch
    outputs = []
    for transposed in (False, True):
        if not (return_transposed if transposed else return_normal):
            outputs += [None, None]
            continue
        data, scales = quantize(matrices, transposed)
        outputs += [data[0] if x.ndim == 2 else data, scales]
    return tuple(outputs)


def quantize_routed_weights(gate, up, down):
    """Prequantize BF16 routed expert weights in the MoK MXFP8 recipe.

    ``gate``/``up`` are ``[E, I, H]`` and ``down`` is ``[E, H, I]``. Returns
    ``(forward_weights, backward_weights)``: forward and context recompute use
    ``(data, scales)`` pairs for gate, up and down; backward uses
    ``(data, scales, data_t, scales_t)`` for gate and up and the transposed
    ``(data_t, scales_t)`` pair for down.
    """
    if (
        any(not isinstance(w, torch.Tensor) or w.ndim != 3 for w in (gate, up, down))
        or gate.shape != up.shape
        or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1])
    ):
        raise ValueError("Expected gate/up [E, I, H] and down [E, H, I] weights")
    weights = (gate, up, down)
    if all(_quantize_both_ok(w) for w in weights):  # one launch with Gluon
        q_gate, q_up, q_down = _quantize_both(*weights)
    else:
        q_gate, q_up, q_down = (mxfp8_quantize(w) for w in weights)
    forward = ((q_gate[0], q_gate[1]), (q_up[0], q_up[1]), (q_down[0], q_down[1]))
    backward = (q_gate, q_up, (q_down[2], q_down[3]))
    return forward, backward
