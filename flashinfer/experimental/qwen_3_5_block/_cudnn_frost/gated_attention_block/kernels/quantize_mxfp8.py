# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 block quantization pass over ``[T, H, D]``: e4m3 codes + F8_128x4 E8M0 scale factors.

The block's MXFP8 pipeline hands the production Rubin SDPA
(``sdpa/fwd/kernels/sm107/prefill_d256_mxfp8.py``) Q/K/V as e4m3 codes plus one
E8M0 scale per 32-element block, in cuDNN's F8_128x4 swizzled order.  Q and K
are quantized ROWWISE (blocks along D, the BMM1 contraction), V COLUMNWISE
(blocks along S, the BMM2 contraction).  This is that pass, forked from the
per-tensor ``kernels/quantize.py`` (same lane mapping, same ``ld_global_v4`` /
``f16x2_to_f32`` / ``fp32_to_fp8_pack`` / ``st_global_v4`` helpers, same
dynamic-token-stride fake for the slab-slice source).

**Oracle** (bit-exact is the bar): ``test/python/sdpa/mxfp8_quant.py::quantize_to_mxfp8``
after permuting the block's ``[T, H, D]`` buffers to BHSD -- it pads S to 128, the
rowwise ``_d`` triple is Q/K, the columnwise ``_s`` triple is V, and pad rows carry
SF ``0x00``.  Scale: ``e = cvt.rp.satfinite.ue8m0x2.f32(amax * fp32(1/448))``
(ONE fp32 multiply by the exact ``0x3B124925``); data: ``e4m3_rn_satfinite(x * 2^(127-e))``.
The e8m0 helpers and the fusing abs-max tree live in ``frost/tile_dsl/pointwise.py``
(``e8m0_from_amax`` / ``e8m0_pair`` / ``abs_max_tree`` on ``fmax_f32`` -- PTX ``max.f32``, which ptxas
fuses into FMNMX3 where ``cute.math.max`` gave compare + select and zero FMNMX).

SF layout contracts (PR-B plan section 2.3, quoted; every byte below is consumed
by the SDPA's TMA descriptors, so a wrong order is numerically wrong and never an error):

* F8_128x4 atom rule -- ``mxfp8_quant._swizzle_128x4``: "Atoms (128 rows x 4 cols
  = 512 B) are laid out row-major; inside an atom, scale (r, c) lives at byte
  ``(r % 32) * 16 + (r // 32) * 4 + c``."  ``swizzle_sf_rowwise`` applies it to
  ``[..., M, K//32]``; ``swizzle_sf_columnwise`` applies it to the TRANSPOSE
  ``[K, M//32]`` and returns the storage shape.  The atom-local arithmetic is
  spelled ONCE, host and device, in ``frost/tile_dsl/sf_layout.py``
  (``sf_atom_offset`` / ``sf_atom_byte``); this kernel and the FP4 quantizer
  both import it, and the host twins below add only the per-unit atom base.
* Q/K (rowwise), consumer ``_build_sf_desc``: **tile ``(b, h, s_tile)`` = ``4*D``
  (1024 at D=256) contiguous bytes at ``((b*H + h)*n_tiles + s_tile) * 4*D``**;
  inside: ``(c//4)*512 + (s%32)*16 + ((s%128)//32)*4 + c%4`` with ``c = d//32``.
* V (columnwise), consumer ``tma_v_sf_desc``: **byte ``(b, h, s_tile, d, s)`` =
  ``(d//128) * (B*KH*n_tiles*512) + ((b*KH + h)*n_tiles + s_tile)*512 +
  ((d%128)%32)*16 + ((d%128)//32)*4 + (s%128)//32``** -- D-plane-major, the plane
  stride ``v_sf_groups*512 = B*KH*ceil(S/128)*512`` GROWS with ``B*KH*S``.
* Adapter ``_reshape_sf`` binds by STORAGE order and checks only
  ``numel == b*h*n_tiles*sf_smem_size`` -- the ``torch.equal`` test against the oracle
  is the only guard on the byte order.
* **GEMM-canonical** (``sf_layout="gemm"``), consumer ``build_proj_gemm``'s SFA / SFB
  declaration (``proj_gemm.sf_padded_dims``): the PADDED F8_128x4 blob over the
  ``[rows, K]`` matrix with the BATCH FOLDED INTO THE ROWS -- atom ``(r_tile, c_atom)``
  at ``(r_tile * n_c_atoms + c_atom) * 512`` with ``n_c_atoms = ceil((K/32)/4)``,
  inside ``(r%32)*16 + (r//32)*4 + c%4``; pad rows / pad blocks ``0x00``;
  ``numel == sf_blob_bytes(rows, K)``.  Rowwise: ``(rows, K) = (T, H*D)`` (the
  ``[T, H, D]`` view of a ``[T, N]`` slab: the block-scale dgrad's A operand);
  transposed columnwise (``transposed=True``, needs ``axis="col"``):
  ``(rows, K) = (H*D, T)`` over the PHYSICALLY TRANSPOSED e4m3 ``[H*D, T]`` output
  (the block-scale wgrad's A operand) -- the columnwise GEMM blob has NO other form:
  ``validate_mode`` rejects ``axis="col", sf_layout="gemm"`` without ``transposed=True``
  (the kernel lays those atoms out over ``(H*D, T)`` whatever the flag says, so a blob
  sized over ``(T, H*D)`` would be overrun).  Host twin ``sf_byte_canonical``; oracle
  ``gated_block_reference.mx_swizzle_sf_rowwise_padded``.  The byte count is
  SYMMETRIC in ``(rows, K)`` (``ceil128(rows) * ceil128(K) / 32``), so no host check
  sees a blob's orientation -- the bitwise tests are the guard.
* **PACKED sequences** (``packed=True``, appended; the block's THD arm, consumer the Rubin d256
  MXFP8 SDPA's THD row): the SF tile axis runs over the sequences' 128-row tiles in
  ``cu_seqlens`` order -- ``n_total = sum_b ceil(s_b/128)`` tiles per head, sequence ``b``'s tiles
  from ``cu_sf[b] = sum_{i<b} ceil(s_i/128)`` -- with batch extent 1.  Q/K: slab ``(h, tile)`` =
  ``4*D`` contiguous bytes at ``(h*n_cap + tile) * 4*D``; V: the two D-planes of ``(h, tile)``
  ADJACENT at ``(h*n_cap + tile) * 4*D + plane*512`` (the SDPA's ``build_columnwise_sf_desc(
  thd_varlen=True)`` strides: plane stride one atom, tile stride the whole slab -- a tile's V bytes
  are the dense layout's two atoms, only their PLACEMENT differs).  ``n_cap = (T + 127*B) // 128``
  is the HOST bound on the live total (``sum_b ceil(s_b/128) <= (T + 127*B) / 128``): the blob's head
  stride is the slot CAPACITY, the lengths are never read on the host, and the tiles past
  ``cu_sf[B]`` (SLACK, at most ``B - 1`` per head) are zero-filled (``0x00``) and store no data.
  Every live tile's SF bytes are written, pad blocks ``0x00`` -- the SDPA's BMM2 READS a V tile's pad
  bytes against masked-to-zero P, so an E8M0 NaN there would poison the row.  Grid ``(H, n_cap)``;
  every warp of a CTA resolves the unit's ``(cu_sf[b], cu[b], s_b)`` by a chunked WARP SCAN over the
  ``[B]`` int32 lengths (or the adjacent differences of ``[B+1]`` prefix sums), 32 sequences per
  step with a carried total: ``O(B/32)`` shuffles per warp, no SMEM array, no cap on ``B`` (the block
  and the SDPA have none).  Host twins ``sf_byte_rowwise_packed`` / ``sf_byte_columnwise_packed``;
  byte count ``sf_bytes_packed(B, H, T, D)``; ``run_quantize_mxfp8(..., seq_lens=, num_sequences=,
  cu_seqlens=)``.  Refused with ``sf_layout="gemm"`` / ``transposed`` (the canonical blobs are
  token-wise already: the block-scale GEMMs contract over ``T`` rows, THD-agnostic).

Grid ``(B*H, ceil(S/128))``: one CTA owns one SF unit (Q/K: one 1024-B tile = 2 atoms;
V: 2 atoms in 2 D-planes); under the canonical layout the batch folds into the rows and
the grid is ``(H, ceil(T/128))``.  The SF bytes are staged in a ``4*D``-byte SMEM tile, one
``bar.sync``, then ``D/4`` lanes burst them out 16 B each -- the same tile and the same one
barrier in every mode (the canonical modes change only the burst's global atom base).

* **Rowwise arm** (``axis="row"``): ``quantize.py``'s mapping -- a lane moves 16
  elements (two ``ld.global.v4`` of bf16 in, one ``st.global.v4`` of e4m3 out), so
  ``D/16`` lanes cover a row (16 = half a warp at D=256), and a lane PAIR
  (``shfl.bfly(1)``) shares one 32-element block.  Both lanes of the pair compute the
  scale (one ``cvt``, cheaper than a broadcast); the even lane owns the SF byte.
* **Columnwise arm** (``axis="col"``): a warp owns 32 consecutive tokens x 64 d
  (lane = 2 adjacent d), so per token the warp reads 128 B contiguous with one
  ``ld.global.b32`` and writes 64 B with ``st.global.b16`` -- 64 live fp32 + 2 amax
  per lane, one ``e8m0_pair`` cvt for both.  The 2-byte stores are the accepted v1
  cost of the SDPA layout, MEASURED against the transposed arm below on the same
  bytes: 0.2407 ms vs 0.0730 ms per launch over a ``[8192, 17408]`` bf16 source
  (412 MiB moved; 1.8 vs 5.9 TB/s; the rowwise arm 0.0555 ms, 7.8 TB/s) on Rubin
  (cc 10.7, 204 SMs, SM clock locked at 2376 MHz; 3 rounds x 50 launches, CUDA
  events, slots shuffled per round) -- the 16-byte store form is 3.3x faster, so a
  columnwise consumer that can take the ``[N, T]`` orientation should; the SDPA's
  D-plane-major ``[T, H, D]`` layout cannot and keeps the 2-byte stores.
  **Transposed** (``transposed=True``): the same lane holds the 32 tokens of ONE
  output row ``n = h*D + d`` of the ``[H*D, T]`` matrix -- 32 CONTIGUOUS bytes at
  ``n*T + t0`` -- so the data leaves as two ``st.global.v4`` per column per lane
  instead of 32 ``st.global.b16``; ``T % 32 == 0`` is required (a 32-token block is
  then entirely live or entirely padding, and every store 16-byte aligned).  The
  canonical rowwise arm costs what the SDPA rowwise arm costs (0.0556 vs 0.0555 ms,
  the same launch).
* **Tail rows** (``s >= S`` in the last tile): the load is clamped to a valid row
  and the value zeroed (rowwise: the block amax is zeroed), the data store is
  skipped, and the SF byte is WRITTEN as ``0x00`` -- an unwritten byte would be
  E8M0 NaN under the SDPA's whole-tile SF TMA, and a zero block quantizes to
  ``0x00`` in both ``cvt.rp.satfinite.ue8m0x2`` and the oracle's ``e8m0_ceil``.

Traffic: ``2 B read + 1 B write + 1/32 B SF`` per element (``moved_bytes``).  Needs
sm_100+ (``cvt.rp.satfinite.ue8m0x2.f32``); the numerics tests run on Rubin, the
shape algebra and the sm_107a trace-compile run anywhere.

**The dual-axis arm** (``frost_quantize_mxfp8_dual`` / ``quantize_mxfp8_dual_body``; the
MXFP8 backward's launch fusion): the rowwise AND the columnwise quantization of the same
rows from ONE read -- ``do8 + do_T8`` (the SDPA pair: both payloads row-major ``[T, H, D]``,
the rowwise SF tiles + the D-plane-major atoms) and ``dqkvg8 + dqkvg_t8`` (the canonical
pair: the rowwise blob over ``(T, H*D)`` + the TRANSPOSED ``[H*D, T]`` store with its blob
over ``(H*D, T)``).  One CTA per 128-row SF unit walks its four 32-token sub-tiles: the
rows are loaded once into registers (the rowwise mapping) and staged in SMEM, a COLUMN pass
(one thread per ``d``) takes the 32-token amax of every column -- its rcp into a 256-float
SMEM array, its SF byte into the columnwise tile, and under the transposed form its 32
codes straight out as two ``st.global.v4`` --, then the ROW pass quantizes the registers
twice: by the row block's scale (the rowwise payload) and, for the SDPA pair, by the 16
column rcps of each lane's ``d`` (the columnwise payload at 16-byte stores -- a transposed
payload is NOT the transpose of the payload, only the scaling axis differs).  Every byte is
BITWISE the two standalone launches': the same helpers in the same order on the same
values, ``max`` exact and order-free, never the fused scaled cvt (it flushes fp32-subnormal
inputs that the exact ``x * rcp`` arm scales).  Tail rows contribute zero to both amaxes
(SF ``0x00``) and store no payload; every SF byte of the unit is written.  Traffic
``2 + 2 x (1 + 1/32)`` B per element (``moved_bytes_dual``) against ``2 x (3 + 1/32)`` for
the two launches, and the 2-byte-store columnwise SDPA arm leaves the backward's chain.
``quantize_mxfp8_rowwise_body`` is the standalone rowwise SDPA unit as a job body (the
fused prologue's ``v8``); the standalone kernel runs the very same ``_rowwise_pass`` /
``_sf_burst_lane`` (its cubins are byte-identical to the pre-hoist ones).
"""

from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _convert_to_cutlass_data_type
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import abs_max_tree, e8m0_from_amax, e8m0_pair, f16x2_to_f32, fmax_f32, fp32_to_fp8_pack, fp32_to_fp8x2
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.sf_layout import SF_ATOM_BYTES, SF_ATOM_COLS, SF_ATOM_ROWS, sf_atom_byte, sf_atom_offset
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global_v4

from .proj_gemm import sf_blob_bytes, sf_padded_dims
from .qk_norm_rope import fake_rowmajor_dynamic_token_stride

AXIS_ROW = "row"  # Q/K: 32-element blocks along D (the BMM1 contraction)
AXIS_COL = "col"  # V:   32-element blocks along S (the BMM2 contraction)
AXES = (AXIS_ROW, AXIS_COL)

SF_LAYOUT_SDPA = "sdpa"  # per-(b, h, 128-row tile) rowwise / D-plane-major columnwise: the SDPA's SF TMA descriptors (module docstring)
SF_LAYOUT_GEMM = "gemm"  # cuDNN's canonical F8_128x4 blob over the [rows, K] matrix, batch folded into the rows: build_proj_gemm's SFA / SFB
SF_LAYOUTS = (SF_LAYOUT_SDPA, SF_LAYOUT_GEMM)

SF_BLOCK = 32  # elements per E8M0 scale
SF_TILE_ROWS = SF_ATOM_ROWS  # 128 rows of an F8_128x4 atom == the SDPA's Q / KV tile height
# SF_ATOM_BYTES / SF_ATOM_COLS / SF_ATOM_ROWS are defined in tile_dsl.sf_layout and RE-EXPORTED from here (the
# import above is load-bearing: `from quantize_mxfp8 import SF_ATOM_BYTES` callers exist in the tests).
SF_BURST_BYTES = 16  # one st.global.v4 per burst lane

ELEMS_PER_LANE = 16  # rowwise: what fp32_to_fp8_pack converts in one call (32 B of bf16 in, 16 B of e4m3 out)
SRC_BYTES_PER_LANE = ELEMS_PER_LANE * 2
DST_BYTES_PER_LANE = ELEMS_PER_LANE * 1
LOADS_PER_LANE = SRC_BYTES_PER_LANE // 16
WARP = 32
COL_D_PER_LANE = 2  # columnwise: one ld.global.b32 = two adjacent bf16 per token
COL_D_PER_UNIT = WARP * COL_D_PER_LANE  # 64 d per warp unit
COL_TOKENS_PER_UNIT = SF_BLOCK  # one 32-token block per warp unit
COL_TOKEN_BLOCKS = SF_TILE_ROWS // COL_TOKENS_PER_UNIT  # 4 per tile

DEFAULT_THREADS_PER_CTA = 256
MAX_THREADS_PER_CTA = 1024  # the hardware CTA cap; past it validation would pass and the launch would fail untyped
COMPILE_OPTIONS = "--enable-tvm-ffi"

_FAKE_STREAM = None


# ---------------------------------------------------------------------------
# Shape algebra (pure Python; the tests pin it against the torch oracle)
# ---------------------------------------------------------------------------


def n_sf_tiles(seq_len: int) -> int:
    """SF tiles per (b, h): ``ceil(S / 128)`` -- the oracle pads S to 128 the same way."""
    return (seq_len + SF_TILE_ROWS - 1) // SF_TILE_ROWS


def sf_tile_bytes(d: int) -> int:
    """SF bytes per ``(b, h, s_tile)`` unit: ``128 * D/32`` rowwise == ``D/128`` atoms of 512 B columnwise == ``4*D`` (1024 at D=256)."""
    return SF_TILE_ROWS * d // SF_BLOCK


def sf_bytes(batch: int, h: int, seq_len: int, d: int) -> int:
    """The flat SF buffer size the SDPA adapter expects: ``B*H*ceil(S/128)*sf_tile_bytes(d)``."""
    return batch * h * n_sf_tiles(seq_len) * sf_tile_bytes(d)


def lanes_per_row(d: int) -> int:
    """Rowwise: lanes cooperating on one head row, 16 elements each, capped at a warp."""
    return min(WARP, d // ELEMS_PER_LANE)


def chunks_per_lane(d: int) -> int:
    return d // (lanes_per_row(d) * ELEMS_PER_LANE)


def sf_byte_rowwise(b: int, h: int, s: int, d_idx: int, *, n_heads: int, n_tiles: int, d: int) -> int:
    """Absolute byte of scale ``(b, h, s, d_idx // 32)`` in the flat Q/K SF buffer (plan section 2.3, Q/K row)."""
    c = d_idx // SF_BLOCK
    r = s % SF_TILE_ROWS
    tile = (b * n_heads + h) * n_tiles + s // SF_TILE_ROWS
    return tile * sf_tile_bytes(d) + sf_atom_offset(r, c)


def sf_byte_columnwise(b: int, h: int, s: int, d_idx: int, *, n_heads: int, n_tiles: int, batch: int) -> int:
    """Absolute byte of scale ``(b, h, s // 32, d_idx)`` in the flat V SF buffer (plan section 2.3, V row): D-plane-major."""
    plane = d_idx // SF_TILE_ROWS
    dm = d_idx % SF_TILE_ROWS
    tile = (b * n_heads + h) * n_tiles + s // SF_TILE_ROWS
    v_sf_groups = batch * n_heads * n_tiles
    return sf_atom_byte(dm, (s % SF_TILE_ROWS) // SF_BLOCK, base=plane * (v_sf_groups * SF_ATOM_BYTES) + tile * SF_ATOM_BYTES)


def canonical_atoms_per_band(k: int) -> int:
    """Atoms per 128-row band of the canonical blob over ``K = k``: ``ceil((K/32)/4)`` -- the device's ``n_c_atoms``
    (one source: ``proj_gemm.sf_padded_dims``)."""
    return sf_padded_dims(SF_ATOM_ROWS, k, SF_BLOCK)[1] // SF_ATOM_COLS


def sf_byte_canonical(row: int, k_block: int, *, k: int) -> int:
    """Absolute byte of scale ``(row, k_block)`` in the GEMM-canonical blob over ``[rows, K = k]``:
    ``(row // 128) * n_c_atoms * 512 + sf_atom_offset(row % 128, k_block)`` -- the host twin of the device's atom-base
    arithmetic (both modes: ``row`` is the token for the rowwise blob, the output row ``n`` for the transposed one)."""
    return (row // SF_ATOM_ROWS) * canonical_atoms_per_band(k) * SF_ATOM_BYTES + sf_atom_offset(row % SF_ATOM_ROWS, k_block)


def n_sf_tiles_packed_cap(t: int, num_sequences: int) -> int:
    """The HOST bound on a packing's SF tile total, ``(T + 127*B) // 128``: ``sum_b ceil(s_b/128) <= sum_b (s_b + 127)/128 =
    (T + 127*B)/128`` for any lengths summing to ``T`` -- the slot CAPACITY per head (the live total needs the lengths, which
    the host never reads).  At ``B = 1`` it is ``ceil(T/128)``: the dense count."""
    return (int(t) + (SF_TILE_ROWS - 1) * int(num_sequences)) // SF_TILE_ROWS


def sf_bytes_packed(num_sequences: int, h: int, t: int, d: int) -> int:
    """The flat SF buffer size of the PACKED layout: ``H * n_sf_tiles_packed_cap(T, B) * sf_tile_bytes(d)`` (the SDPA binder derives
    the packed tile extent from this byte count, so K and V -- both ``h_kv`` -- carry the same count by construction)."""
    return h * n_sf_tiles_packed_cap(t, num_sequences) * sf_tile_bytes(d)


def sf_byte_rowwise_packed(h: int, tile: int, r: int, d_idx: int, *, n_cap: int, d: int) -> int:
    """Absolute byte of scale ``(h, packed tile, row r of the tile, d_idx // 32)`` in the PACKED Q/K SF buffer: slab ``(h, tile)``
    at ``(h*n_cap + tile) * 4*D``, atom-local as the dense layout."""
    return (h * n_cap + tile) * sf_tile_bytes(d) + sf_atom_offset(r, d_idx // SF_BLOCK)


def sf_byte_columnwise_packed(h: int, tile: int, r: int, d_idx: int, *, n_cap: int, d: int) -> int:
    """Absolute byte of scale ``(h, packed tile, r // 32, d_idx)`` in the PACKED V SF buffer: the D-planes of ``(h, tile)`` ADJACENT
    at ``(h*n_cap + tile) * 4*D + plane*512`` (the THD strides of the SDPA's columnwise SF descriptor)."""
    plane = d_idx // SF_TILE_ROWS
    dm = d_idx % SF_TILE_ROWS
    return sf_atom_byte(dm, r // SF_BLOCK, base=(h * n_cap + tile) * sf_tile_bytes(d) + plane * SF_ATOM_BYTES)


def validate_packed_mode(sf_layout: str, transposed: bool, packed: bool) -> None:
    """The packed arm's contract, typed: ``packed=True`` serves the SDPA layouts only (no canonical blob, no transposed store)."""
    if not isinstance(packed, bool):
        raise ValueError(f"packed must be a bool, got {packed!r}")
    if packed and sf_layout != SF_LAYOUT_SDPA:
        raise ValueError(
            f"packed=True is the SDPA layouts' per-sequence-tile-padded arm (sf_layout='sdpa'); the GEMM-canonical blobs are token-wise already "
            f"and take no packed arm, got sf_layout={sf_layout!r}"
        )
    if packed and transposed:
        raise ValueError("packed=True serves the SDPA layouts only: transposed=True (the canonical [H*D, T] store) is refused")


def validate_mode(axis: str, sf_layout: str, transposed: bool) -> None:
    """The (axis, sf_layout, transposed) contract, typed both ways: ``transposed`` is the columnwise arm's GEMM-canonical
    store and nothing else."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES} ('row' for Q/K, 'col' for V), got {axis!r}")
    if sf_layout not in SF_LAYOUTS:
        raise ValueError(
            f"sf_layout must be one of {SF_LAYOUTS} ('sdpa': the SDPA's per-tile / D-plane-major SF; 'gemm': the canonical blob), got {sf_layout!r}"
        )
    if not isinstance(transposed, bool):
        raise ValueError(f"transposed must be a bool, got {transposed!r}")
    if transposed and axis != AXIS_COL:
        raise ValueError(f"transposed=True is the columnwise arm's [H*D, T] store (32-token blocks along T): it needs axis='col', got axis={axis!r}")
    if transposed and sf_layout != SF_LAYOUT_GEMM:
        raise ValueError(f"transposed=True writes the block-scale GEMM's A operand: it needs sf_layout='gemm', got sf_layout={sf_layout!r}")
    if axis == AXIS_COL and sf_layout == SF_LAYOUT_GEMM and not transposed:
        # The columnwise GEMM-canonical arm exists only as the transposed [H*D, T] store: its scale atoms are laid out over
        # (rows = H*D, K = T) by the kernel whatever `transposed` says, while a non-transposed blob is sized over (rows = T, K = H*D)
        # -- accepting the pair would write scales past the end of the SF buffer (reproduced in review: T=128, H=2, D=256 stores
        # reach byte 6655 of a 2048-byte blob).  The SDPA's columnwise V scales are the sf_layout='sdpa' arm.
        raise ValueError(
            "axis='col' with sf_layout='gemm' is served only as the transposed [H*D, T] store: it needs transposed=True "
            "(the columnwise scale atoms are laid out over rows = H*D, K = T, and a non-transposed blob is sized over rows = T, K = H*D); "
            "the SDPA's columnwise V scales are sf_layout='sdpa'"
        )


def validate_shape(d: int, threads_per_cta: int, axis: str) -> None:
    """Raise ``ValueError`` on any geometry this kernel cannot address (never an assert)."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES} ('row' for Q/K, 'col' for V), got {axis!r}")
    if d <= 0 or d % SF_TILE_ROWS != 0:
        raise ValueError(f"d_head must be a positive multiple of 128 (whole F8_128x4 atoms: D/32 blocks in fours rowwise, 128 d-rows columnwise), got {d}")
    if threads_per_cta <= 0 or threads_per_cta % WARP != 0 or threads_per_cta > MAX_THREADS_PER_CTA:
        raise ValueError(f"threads_per_cta must be a positive multiple of {WARP} up to {MAX_THREADS_PER_CTA} (the CTA cap), got {threads_per_cta}")
    burst_lanes = sf_tile_bytes(d) // SF_BURST_BYTES
    if threads_per_cta < burst_lanes:
        raise ValueError(f"threads_per_cta={threads_per_cta} cannot burst the {sf_tile_bytes(d)}-byte SF tile in 16-byte lanes ({burst_lanes} needed)")
    if axis == AXIS_ROW:
        lanes = lanes_per_row(d)
        if WARP % lanes != 0:
            raise ValueError(f"d_head={d} gives {lanes} lanes/row, which must divide a warp")
        if d != lanes * ELEMS_PER_LANE * chunks_per_lane(d):
            raise ValueError(f"d_head={d} is not covered exactly by {lanes} lanes x {chunks_per_lane(d)} chunks x {ELEMS_PER_LANE} elements")
        if threads_per_cta % lanes != 0:
            raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of the {lanes} lanes per row")
        rows_per_pass = threads_per_cta // lanes
        if SF_TILE_ROWS % rows_per_pass != 0:
            raise ValueError(f"threads_per_cta={threads_per_cta} covers {rows_per_pass} rows per pass, which must divide the 128-row tile")
    else:
        units = COL_TOKEN_BLOCKS * (d // COL_D_PER_UNIT)
        warps = threads_per_cta // WARP
        if units % warps != 0:
            raise ValueError(f"columnwise d_head={d} has {units} (32-token x 64-d) units per tile, not a multiple of the {warps} warps")


def moved_bytes(t: int, h: int, d: int, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one launch: the bf16/f16 read, the 1-byte e4m3 write and the 1/32-byte SF write per element."""
    return t * h * d * (src_elem_bytes + 1) + t * h * d // SF_BLOCK


def validate_dual_shape(d: int, threads_per_cta: int) -> None:
    """The dual-axis kernel's geometry contract (``quantize_mxfp8_dual_body``): the rowwise arm's row mapping, PLUS one thread per
    ``d`` in the column pass, whole 32-token sub-tiles per pass set, and both SF tiles bursting in parallel.  Raises ``ValueError``."""
    validate_shape(d, threads_per_cta, AXIS_ROW)
    if d > threads_per_cta:
        raise ValueError(f"the dual-axis quantize's column pass runs one thread per d: d_head={d} needs threads_per_cta >= {d}, got {threads_per_cta}")
    rows_per_pass = threads_per_cta // lanes_per_row(d)
    if SF_BLOCK % rows_per_pass != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} covers {rows_per_pass} rows per pass, which must divide the {SF_BLOCK}-token sub-tile")
    if threads_per_cta < 2 * (sf_tile_bytes(d) // SF_BURST_BYTES):
        raise ValueError(
            f"threads_per_cta={threads_per_cta} cannot burst the two {sf_tile_bytes(d)}-byte SF tiles in parallel ({2 * (sf_tile_bytes(d) // SF_BURST_BYTES)} 16-byte lanes needed)"
        )


def moved_bytes_dual(t: int, h: int, d: int, src_elem_bytes: int = 2) -> int:
    """HBM traffic of one dual-axis launch: ONE bf16/f16 read, TWO 1-byte e4m3 writes and two 1/32-byte SF writes per element
    (against ``2 * moved_bytes`` for the two standalone launches it replaces)."""
    return t * h * d * (src_elem_bytes + 2) + 2 * (t * h * d // SF_BLOCK)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def st_global_b16(addr, value):
    """16-bit global store of a ``Uint16`` (two packed e4m3 codes)."""
    nvvm.inline_ptx("st.global.b16 [$0], $1;", read_only_args=[addr, value])


@cute.jit
def _packed_owner(mLens: cute.Tensor, n_seq: cutlass.Int32, lens_form: cutlass.Int32, tile: cutlass.Int32):
    """``(cu_sf_b, cu_b, s_b, live)`` of the sequence ``b`` that owns PACKED SF tile ``tile``: ``cu_sf_b`` its first tile,
    ``cu_b`` its first token, ``s_b`` its length; ``live`` False for a SLACK tile past ``cu_sf[B]`` (then all three are 0).

    A chunked WARP scan over the lengths (modelled on ``tile_dsl.thd.write_thd_prefix_warp``): per 32-sequence chunk lane
    ``l`` loads ``len[start + l]`` (0 past ``B``; ``lens_form != 0`` = ``[B+1]`` prefix sums, the length is the adjacent
    difference so any base works), forms ``ceil(len/128)``, two inclusive 5-step shuffle scans give the per-lane tile and
    token prefixes, and the owner chunk is the one with ``carry_sf <= tile < carry_sf + chunk_total`` (warp-uniform); inside
    it the lanes with ``carry_sf + prefix_sf > tile`` form a SUFFIX (the prefixes are non-decreasing), so the owner lane is
    ``32 - popc(ballot)`` and its EXCLUSIVE prefixes are read back by shuffle.  No SMEM array, no cap on ``B``; ``O(B/32)``
    iterations of one 32-wide int32 load (L2-resident) + 12 shuffles.  Every lane of the CALLING warp must participate; the
    result is warp-uniform.  Pure selects, no data-dependent control flow (a traced early exit is not expressible)."""
    lane = cutlass.Int32(cute.arch.thread_idx()[0]) % cutlass.Int32(WARP)
    lens_base = mLens.iterator.toint()
    is_cu = lens_form != cutlass.Int32(0)
    cu_step = cutlass.Int64(4) if is_cu else cutlass.Int64(0)  # the second load: the next prefix entry (cu) or the same word (lengths)
    carry_sf = cutlass.Int32(0)
    carry = cutlass.Int32(0)
    cu_sf_b = cutlass.Int32(0)
    cu_b = cutlass.Int32(0)
    s_b = cutlass.Int32(0)
    live = cutlass.Boolean(False)
    for start in cutlass.range(0, n_seq, WARP, unroll=1):
        b = start + lane
        in_range = b < n_seq
        b_c = b if in_range else n_seq - cutlass.Int32(1)  # clamp the LOAD index (never past the tensor); the value is zeroed below
        addr = lens_base + b_c.to(cutlass.Int64) * cutlass.Int64(4)
        v0 = ld_global(addr, cutlass.Int32)
        v1 = ld_global(addr + cu_step, cutlass.Int32)
        length = (v1 - v0) if is_cu else v0
        length = length if in_range else cutlass.Int32(0)
        n_l = (length + cutlass.Int32(SF_TILE_ROWS - 1)) // cutlass.Int32(SF_TILE_ROWS)
        incl_sf = n_l
        incl = length
        for shift in cutlass.range_constexpr(5):
            delta = 1 << shift
            p_sf = cute.arch.shuffle_sync_up(incl_sf, offset=delta, mask_and_clamp=0)
            p = cute.arch.shuffle_sync_up(incl, offset=delta, mask_and_clamp=0)
            incl_sf = (incl_sf + p_sf) if lane >= cutlass.Int32(delta) else incl_sf
            incl = (incl + p) if lane >= cutlass.Int32(delta) else incl
        total_sf = cute.arch.shuffle_sync(incl_sf, WARP - 1)
        total = cute.arch.shuffle_sync(incl, WARP - 1)
        sel = (carry_sf <= tile) & (tile < carry_sf + total_sf)  # the owner is in this chunk (warp-uniform)
        hit = (carry_sf + incl_sf) > tile  # a SUFFIX of lanes; its first lane owns the tile (n_l >= 1 there)
        ballot = cute.arch.vote_ballot_sync(hit)
        owner = cutlass.Int32(WARP) - cutlass.Int32(cute.arch.popc(ballot))
        owner_c = owner if owner < cutlass.Int32(WARP) else cutlass.Int32(WARP - 1)  # 32 when no lane hits (a non-owner chunk): clamp for the shuffle
        excl_sf = cute.arch.shuffle_sync(incl_sf - n_l, owner_c)
        excl = cute.arch.shuffle_sync(incl - length, owner_c)
        len_o = cute.arch.shuffle_sync(length, owner_c)
        cu_sf_b = (carry_sf + excl_sf) if sel else cu_sf_b
        cu_b = (carry + excl) if sel else cu_b
        s_b = len_o if sel else s_b
        live = live | sel
        carry_sf = carry_sf + total_sf
        carry = carry + total
    return cu_sf_b, cu_b, s_b, live


@cute.jit
def _unit_coords(
    bh: cutlass.Int32,
    s_tile: cutlass.Int32,
    seq_len: cutlass.Int32,
    h: cutlass.Constexpr[int],
    packed_seq: cutlass.Constexpr[bool],
    mLens,  # packed: the [B] / [B+1] int32 lengths tensor; dense: None (never touched)
    n_seq,  # packed: Int32 B; dense: None (never touched)
    lens_form,  # packed: Int32, 0 = lengths / 1 = prefix sums; dense: None
):
    """``(head, s0, tok0, last, s_len)`` of the CTA's unit -- the ONE decode of the three kernel bodies (the standalone kernel, the
    rowwise job body, the dual-axis body), so the two layouts cannot drift between them.

    Dense (``packed_seq=False``, the padded grid ``(B*H, ceil(S/128))``): ``b = bh // H``, ``head = bh % H``, ``s0 = s_tile*128``,
    ``tok0 = b*S``, ``last = S - 1``, ``s_len = S`` -- the expressions the sites used to spell inline, in the same order.
    Packed (``packed_seq=True``, the grid ``(H, n_cap)``): ``head = bh``, the owner sequence from :func:`_packed_owner`, ``s0`` the
    tile's first row WITHIN the sequence, ``tok0 = cu[b]``, ``last = s_b - 1`` (the clamp target of a tail row's load, relative to
    ``tok0``), ``s_len = s_b``; a SLACK tile gets ``s_len = 0`` (every row a tail row: SF ``0x00``, no store) with ``tok0 = last =
    s0 = 0`` so its clamped loads read token 0."""
    if cutlass.const_expr(packed_seq):
        head = bh
        cu_sf_b, cu_b, s_b, live = _packed_owner(mLens, n_seq, lens_form, s_tile)
        s0 = ((s_tile - cu_sf_b) * cutlass.Int32(SF_TILE_ROWS)) if live else cutlass.Int32(0)
        tok0 = cu_b
        last = (s_b - cutlass.Int32(1)) if live else cutlass.Int32(0)
        s_len = s_b
    else:
        b = bh // cutlass.Int32(h)
        head = bh % cutlass.Int32(h)
        s0 = s_tile * cutlass.Int32(SF_TILE_ROWS)
        tok0 = b * seq_len
        last = seq_len - cutlass.Int32(1)
        s_len = seq_len
    return head, s0, tok0, last, s_len


@cute.jit
def _rowwise_pass(
    mSrc: cute.Tensor,  # bound for its element_type; the data moves through the raw addresses below
    seq_len: cutlass.Int32,
    s0: cutlass.Int32,  # the unit's first row (s_tile * 128)
    tok0: cutlass.Int32,  # token index of this batch's row 0
    last: cutlass.Int32,  # seq_len - 1: tail rows clamp their LOADS here
    src_base: cutlass.Int64,  # this head's byte base in the source / destination, and the token pitches in bytes
    src_tok_stride: cutlass.Int64,
    dst_base: cutlass.Int64,
    dst_tok_stride: cutlass.Int64,
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    sSF,  # Uint8 SMEM Array, sf_tile_bytes(d): the unit's SF tile, filled here, burst by the caller
) -> None:
    """The ROWWISE arm's 128-row pass (module docstring): a lane moves 16 elements of one row, a lane PAIR shares one
    32-element block (``shfl.bfly(1)`` completes the block amax), both lanes compute the scale, the even lane owns the SF byte
    (``sf_atom_offset(r, c)`` into ``sSF``); a tail row (``s >= seq_len``) clamps its load, zeroes its amax (SF ``0x00``) and
    stores no data.  The standalone kernel's rowwise arm and the fused MXFP8 prologue's ``v8`` job
    (``quantize_mxfp8_rowwise_body``) run this one body."""
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    rows_per_pass = cutlass.const_expr(threads_per_cta // lanes)
    passes = cutlass.const_expr(SF_TILE_ROWS // rows_per_pass)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    even = (lane & cutlass.Int32(1)) == cutlass.Int32(0)
    src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)
    dst_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(DST_BYTES_PER_LANE)
    for p in cutlass.range_constexpr(passes):
        r = cutlass.Int32(p * rows_per_pass) + grp  # row within the 128-row tile
        s = s0 + r
        valid = s < seq_len
        s_ld = s if valid else last
        tok = tok0 + s_ld
        src_row = src_base + tok.to(cutlass.Int64) * src_tok_stride
        dst_row = dst_base + tok.to(cutlass.Int64) * dst_tok_stride
        for c in cutlass.range_constexpr(chunks):
            base = src_row + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
            vals = []
            for j in cutlass.range_constexpr(LOADS_PER_LANE):
                for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                    lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                    vals.append(lo)
                    vals.append(hi)
            # A lane pair (lane, lane^1) shares one 32-element block: bfly(1) completes the block amax.
            amax = abs_max_tree(vals)
            partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, cutlass.Int32(1), 31, kind=nvvm.Shfl.BFLY))
            amax = fmax_f32(amax, partner)
            amax = amax if valid else cutlass.Float32(0.0)  # tail row: SF 0x00, no data store
            rcp, sf_byte = e8m0_from_amax(amax)
            if valid:
                scaled = []
                for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                    scaled.append(vals[i] * rcp)
                packed = fp32_to_fp8_pack(scaled, dtype=cutlass.Float8E4M3FN)
                st_global_v4(
                    dst_row + cutlass.Int64((c * lanes) * DST_BYTES_PER_LANE) + dst_lane_off, [packed[0], packed[1], packed[2], packed[3]], cutlass.Int32
                )
            # SF SMEM byte for block column c_idx = d//32 of row r: (c//4)*512 + (r%32)*16 + (r//32)*4 + c%4 (sf_layout).
            c_idx = (cutlass.Int32(c * lanes) + lane) // cutlass.Int32(2)
            sf_off = sf_atom_offset(r, c_idx)
            if even:
                sSF.store(sf_byte.to(cutlass.Uint8), sf_off)


@cute.jit
def _sf_burst_lane(
    mSf: cute.Tensor,
    sSF,  # Uint8 SMEM Array, sf_tile_bytes(d)
    lane: cutlass.Int32,  # the burst lane in [0, sf_tile_bytes(d) / 16): the CALLER gates the range
    bh: cutlass.Int32,
    s_tile: cutlass.Int32,
    head: cutlass.Int32,
    n_tiles: cutlass.Int32,
    v_sf_groups: cutlass.Int32,
    n_c_atoms: cutlass.Int32,
    d: cutlass.Constexpr[int],
    axis_col: cutlass.Constexpr[bool],
    sf_gemm: cutlass.Constexpr[bool],
    packed_seq: cutlass.Constexpr[bool] = False,  # APPENDED: the PACKED layout -- Q/K unchanged (bh = head, n_tiles = n_cap), V plane-ADJACENT
) -> None:
    """One 16-byte lane of the SF burst: the SMEM tile at ``lane * 16`` to its place in the blob -- the SDPA Q/K tile (4*D contiguous
    bytes), the SDPA V D-plane-major atoms, or the GEMM-canonical atoms (rowwise over ``(T, H*D)`` / transposed over ``(H*D, T)``), the
    module docstring's four layouts; under ``packed_seq`` the SDPA PACKED pair (Q/K: the same ``(bh*n_tiles + s_tile) * 4*D`` with ``bh`` the
    head and ``n_tiles`` the capacity ``n_cap``; V: plane ``p`` of the tile at ``tile_idx * 4*D + p*512``).  Only the atom BASE differs
    per layout; the SMEM staging is the same bytes in the same atom-local order."""
    tile_bytes = cutlass.const_expr(sf_tile_bytes(d))
    head_atoms = cutlass.const_expr(d // SF_TILE_ROWS)  # D/128: one head's atoms along K (rowwise) == its 128-row bands of the [H*D, T] matrix
    smem_off = lane * cutlass.Int32(SF_BURST_BYTES)
    words4 = sSF.load(smem_off, vector_size=SF_BURST_BYTES, alignment=16).bitcast(cutlass.Int32)
    tile_idx = (bh * n_tiles + s_tile).to(cutlass.Int64)
    sf_base = mSf.iterator.toint()
    if cutlass.const_expr(sf_gemm):
        # GEMM-canonical: atom (r_tile, c_atom) of the padded [rows, K] blob at (r_tile * n_c_atoms + c_atom) * 512, the batch
        # folded into the rows (b == 0, s_tile indexes T).  Only the atom BASE differs from the SDPA layouts; the SMEM staging
        # (sf_atom_offset / sf_atom_byte) is the same bytes in the same atom-local order.
        if cutlass.const_expr(not axis_col):
            # rowwise over (rows = T, K = H*D): this CTA's D/128 atoms are c_atom = head*D/128 .. +D/128-1, contiguous = the 4*D tile.
            atom0 = s_tile.to(cutlass.Int64) * n_c_atoms.to(cutlass.Int64) + (head * cutlass.Int32(head_atoms)).to(cutlass.Int64)
            gaddr = sf_base + atom0 * cutlass.Int64(SF_ATOM_BYTES) + smem_off.to(cutlass.Int64)
        else:
            # transposed over (rows = H*D, K = T): plane p (rows n = head*D + p*128 ..) is row tile head*D/128 + p, the 128 tokens
            # of this CTA are the 4 blocks of column atom s_tile.
            burst_plane = (lane // cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)).to(cutlass.Int64)
            burst_within = ((lane % cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)) * cutlass.Int32(SF_BURST_BYTES)).to(cutlass.Int64)
            atom = ((head * cutlass.Int32(head_atoms)).to(cutlass.Int64) + burst_plane) * n_c_atoms.to(cutlass.Int64) + s_tile.to(cutlass.Int64)
            gaddr = sf_base + atom * cutlass.Int64(SF_ATOM_BYTES) + burst_within
    elif cutlass.const_expr(not axis_col):
        # Q/K: the tile is 4*D contiguous bytes.
        gaddr = sf_base + tile_idx * cutlass.Int64(tile_bytes) + smem_off.to(cutlass.Int64)
    elif cutlass.const_expr(packed_seq):
        # V, PACKED: the D-planes of (head, tile) are ADJACENT -- plane p at tile_idx * 4*D + p * 512 (the SDPA's THD columnwise strides).
        burst_plane = (lane // cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)).to(cutlass.Int64)
        burst_within = ((lane % cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)) * cutlass.Int32(SF_BURST_BYTES)).to(cutlass.Int64)
        gaddr = sf_base + tile_idx * cutlass.Int64(tile_bytes) + burst_plane * cutlass.Int64(SF_ATOM_BYTES) + burst_within
    else:
        # V: plane p (512 B, lanes p*32..p*32+31) lands p * v_sf_groups atoms away.
        burst_plane = (lane // cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)).to(cutlass.Int64)
        burst_within = ((lane % cutlass.Int32(SF_ATOM_BYTES // SF_BURST_BYTES)) * cutlass.Int32(SF_BURST_BYTES)).to(cutlass.Int64)
        gaddr = sf_base + burst_plane * (v_sf_groups.to(cutlass.Int64) * cutlass.Int64(SF_ATOM_BYTES)) + tile_idx * cutlass.Int64(SF_ATOM_BYTES) + burst_within
    st_global_v4(gaddr, [words4[0], words4[1], words4[2], words4[3]], cutlass.Int32)


@cute.jit
def quantize_mxfp8_rowwise_body(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token stride (a slab band or compact)
    mDst: cute.Tensor,  # [T, H, D] e4m3 compact
    mSf: cute.Tensor,  # [B*H*ceil(S/128)*4*D] uint8: the SDPA rowwise layout (per-(b, h, s_tile) 4*D-byte tiles)
    seq_len: cutlass.Int32,
    n_tiles: cutlass.Int32,  # ceil(S/128)
    bh: cutlass.Int32,  # the unit: (b * H + h, s_tile) -- the standalone kernel's (blockIdx.x, blockIdx.y), a fused launch's job-relative index
    s_tile: cutlass.Int32,
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    sSF,  # Uint8 SMEM Array, sf_tile_bytes(d) (the CALLER's: allocated once per kernel, never per inlined arm)
    mLens=None,  # APPENDED, packed only: the [B] / [B+1] int32 lengths tensor
    n_seq=None,  # APPENDED, packed only: Int32 B
    lens_form=None,  # APPENDED, packed only: Int32 (0 lengths / 1 prefix sums)
    packed_seq: cutlass.Constexpr[bool] = False,  # APPENDED: the PACKED layout (grid (H, n_cap); bh = head, s_tile = the packed tile)
) -> None:
    """ONE unit of the standalone kernel's ROWWISE SDPA-layout arm (``axis="row", sf_layout="sdpa"``) as a job body: the 128-row
    pass (:func:`_rowwise_pass`), the barrier, the 4*D-byte SF tile's burst -- byte for byte what ``frost_quantize_mxfp8`` writes for
    that unit.  The fused MXFP8 backward prologue (``mxfp8_bwd_fused.py``) runs it for ``v8`` (V's compaction, straight from the slab's
    V band) behind its block-range dispatch; ``validate_shape(d, threads_per_cta, "row")`` is the geometry contract (128 threads at
    D = 256: 16 lanes per row, 8 rows per pass, 64 burst lanes)."""
    tile_bytes = cutlass.const_expr(sf_tile_bytes(d))
    burst_lanes = cutlass.const_expr(tile_bytes // SF_BURST_BYTES)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    head, s0, tok0, last, s_len = _unit_coords(bh, s_tile, seq_len, h, packed_seq, mLens, n_seq, lens_form)
    src_tok_stride = cutlass.Int64(mSrc.stride[0]) * cutlass.Int64(2)
    src_base = mSrc.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1]) * cutlass.Int64(2)
    dst_tok_stride = cutlass.Int64(mDst.stride[0])
    dst_base = mDst.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mDst.stride[1])
    _rowwise_pass(mSrc, s_len, s0, tok0, last, src_base, src_tok_stride, dst_base, dst_tok_stride, d, threads_per_cta, sSF)
    nvvm.barrier_cta_sync()
    if tidx < cutlass.Int32(burst_lanes):
        _sf_burst_lane(mSf, sSF, tidx, bh, s_tile, head, n_tiles, cutlass.Int32(0), cutlass.Int32(0), d, False, False, packed_seq)


@cute.jit
def quantize_mxfp8_dual_body(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token stride (a slab band or compact)
    mDst: Optional[cute.Tensor],  # [T, H, D] e4m3 compact: the ROWWISE payload (None when want_row is False)
    mSf: Optional[cute.Tensor],  # the rowwise SF blob: the SDPA per-(b, h, s_tile) tiles, or the canonical blob over (rows = T, K = H*D)
    mDstT: Optional[cute.Tensor],  # the SECOND payload: [T, H, D] e4m3 compact (the SDPA columnwise, ROW-MAJOR like the rowwise one) or the
    #                                contiguous [H*D, T] matrix (the canonical TRANSPOSED store, transposed_second); None when want_col is False
    mSfT: Optional[cute.Tensor],  # the second SF blob: the SDPA D-plane-major atoms, or the canonical blob over (rows = H*D, K = T)
    seq_len: cutlass.Int32,
    n_tiles: cutlass.Int32,  # ceil(S/128) (== ceil(T/128) under the canonical layout: the batch folds into the rows)
    v_sf_groups: cutlass.Int32,  # B*H*n_tiles: the SDPA columnwise D-plane stride in atoms (0 under the canonical layout)
    n_c_atoms: cutlass.Int32,  # canonical: atoms per 128-row band of the rowwise blob, ceil((H*D/32)/4) (0 under the SDPA layouts)
    n_c_atoms_t: cutlass.Int32,  # canonical: atoms per 128-row band of the transposed blob, ceil((T/32)/4) (0 under the SDPA layouts)
    bh: cutlass.Int32,  # the unit (b * H + h, s_tile): the standalone kernel's (blockIdx.x, blockIdx.y), a fused launch's job-relative index
    s_tile: cutlass.Int32,
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    sf_gemm: cutlass.Constexpr[bool],  # the GEMM-canonical pair (rowwise over (T, H*D) + transposed over (H*D, T)); False = the SDPA pair
    transposed_second: cutlass.Constexpr[bool],  # the second payload is the [H*D, T] matrix (sf_gemm only -- the canonical columnwise blob's one form)
    want_row: cutlass.Constexpr[bool],  # trace the rowwise half (its payload + SF tile + burst)
    want_col: cutlass.Constexpr[bool],  # trace the columnwise half (the staging tile, the column pass, its payload + SF tile + burst)
    sStage,  # Int32 SMEM Array, SF_BLOCK * d // 2 words: the 32-token staging tile [tok][d] (the column pass reads it)
    sScale,  # Float32 SMEM Array, d: the column rcps of the current sub-tile
    sSFrow,  # Uint8 SMEM Array, sf_tile_bytes(d)
    sSFcol,  # Uint8 SMEM Array, sf_tile_bytes(d)
    mLens=None,  # APPENDED, packed only: the [B] / [B+1] int32 lengths tensor
    n_seq=None,  # APPENDED, packed only: Int32 B
    lens_form=None,  # APPENDED, packed only: Int32 (0 lengths / 1 prefix sums)
    packed_seq: cutlass.Constexpr[bool] = False,  # APPENDED: the PACKED SDPA pair (sf_gemm False; grid (H, n_cap); bh = head, s_tile = the packed tile)
) -> None:
    """ONE (b, h, 128-token SF unit) of the DUAL-AXIS quantize -- the rowwise AND the columnwise MXFP8 quantization of the same
    bf16 rows from ONE read (module docstring, "The dual-axis arm").  The unit's four 32-token sub-tiles in turn; per sub-tile:

    1. LOAD: thread ``(row = p * rows_per_pass + grp, lane)`` -- the rowwise arm's mapping, 16 elements per lane -- loads its 32 B
       (a tail row ``s >= seq_len`` clamps the load and ZEROES its words: it contributes 0 to every amax) and keeps the 8 words in
       REGISTERS; then stores them to ``sStage[row][lane*16 ..]`` (16 lanes span 512 B = 4 bank cycles: conflict-free) -- ``bar.sync``;
    2. COLUMN pass: thread ``d`` reads the 32 tokens of column ``d`` from ``sStage`` (a warp reads 64 B contiguous per token),
       ``abs_max_tree`` -> ``e8m0_from_amax`` -> the rcp into ``sScale[d]`` and the SF byte into ``sSFcol`` at the standalone columnwise
       arm's atom-local byte (``sf_atom_byte(d % 128, tb, base = (d // 128) * 512)``); under ``transposed_second`` the thread ALSO stores
       its 32 scaled codes as the transposed arm does -- 32 contiguous bytes of row ``n = h*D + d`` of the ``[H*D, T]`` matrix, two
       ``st.global.v4``, a whole block live or a whole block padding (the host's ``T % 32 == 0`` rule) -- ``bar.sync``;
    3. ROW pass, from the registers of step 1: the rowwise arm's body verbatim (a lane pair's block amax, ``e8m0_from_amax``,
       ``x * rcp`` -> ``fp32_to_fp8_pack`` -> one ``st.global.v4`` per lane, the SF byte into ``sSFrow`` by the even lane) and, for the
       SDPA pair, the COLUMNWISE payload from the same registers -- ``x * rcp_col[d]`` with the 16 column rcps read from ``sScale``,
       ``fp32_to_fp8_pack``, one ``st.global.v4`` per lane into the row-major ``[T, H, D]`` second destination (a transposed payload
       is NOT the transpose of the payload: only the scaling axis differs, so it takes the 16-byte stores the standalone columnwise
       arm cannot).  Every payload store is predicated on ``s < seq_len``; every SF byte of the unit is written (pad blocks ``0x00``).

    After the four sub-tiles one ``bar.sync`` and the two SF bursts (lanes ``[0, 64)`` the rowwise tile, ``[64, 128)`` the columnwise
    one; :func:`_sf_burst_lane` with each blob's own atom base).  BITWISE the two standalone launches by construction: the same
    helpers in the same order on the same values -- ``max`` is exact and order-free, ``e8m0_pair``'s two bytes are two independent
    ``cvt.rp.satfinite.ue8m0x2`` lanes of the same product ``amax * fp32(1/448)``, and the pack is the same
    ``cvt.rn.satfinite.e4m3x2.f32`` per element pair (never the fused scaled cvt, which flushes fp32-subnormal inputs).
    Barrier table: two ``bar.sync`` per sub-tile + one before the bursts, every one reached by all ``threads_per_cta`` threads (the
    only divergence is around predicated stores and the ``tidx < d`` column-pass gate, which precedes no collective).  ``want_row`` /
    ``want_col`` fold a half out (the fused epilogue's ``need_dh`` / ``need_dw_qkvg`` arms); both False is refused by the host."""
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(chunks_per_lane(d))
    rows_per_pass = cutlass.const_expr(threads_per_cta // lanes)
    passes = cutlass.const_expr(SF_BLOCK // rows_per_pass)
    words_per_lane = cutlass.const_expr(ELEMS_PER_LANE // 2)  # 8 Int32 words = 16 bf16 elements
    words_per_row = cutlass.const_expr(d // 2)
    tile_bytes = cutlass.const_expr(sf_tile_bytes(d))
    burst_lanes = cutlass.const_expr(tile_bytes // SF_BURST_BYTES)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    even = (lane & cutlass.Int32(1)) == cutlass.Int32(0)
    head, s0, tok0, last, s_len = _unit_coords(bh, s_tile, seq_len, h, packed_seq, mLens, n_seq, lens_form)
    src_tok_stride = cutlass.Int64(mSrc.stride[0]) * cutlass.Int64(2)
    src_base = mSrc.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1]) * cutlass.Int64(2)
    # the destinations of the halves that are traced (an absent half's operands are None: no address is formed from them)
    dst_tok_stride = cutlass.Int64(mDst.stride[0]) if cutlass.const_expr(want_row) else cutlass.Int64(0)
    dst_base = (mDst.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mDst.stride[1])) if cutlass.const_expr(want_row) else cutlass.Int64(0)
    # the second destination: [T, H, D] like the first (SDPA), or the [H*D, T] matrix whose row n = head*D + d starts at head*D*T
    if cutlass.const_expr(want_col and transposed_second):
        dstT_tok_stride = cutlass.Int64(mDstT.stride[1])
        dstT_row_stride = cutlass.Int64(mDstT.stride[0])
        dstT_base = mDstT.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(d) * dstT_row_stride
    elif cutlass.const_expr(want_col):
        dstT_tok_stride = cutlass.Int64(mDstT.stride[0])
        dstT_row_stride = cutlass.Int64(1)
        dstT_base = mDstT.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mDstT.stride[1])
    else:
        dstT_tok_stride = cutlass.Int64(0)
        dstT_row_stride = cutlass.Int64(1)
        dstT_base = cutlass.Int64(0)
    src_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(SRC_BYTES_PER_LANE)
    dst_lane_off = lane.to(cutlass.Int64) * cutlass.Int64(DST_BYTES_PER_LANE)

    for tb in cutlass.range_constexpr(COL_TOKEN_BLOCKS):
        s_blk = s0 + cutlass.Int32(tb * COL_TOKENS_PER_UNIT)
        # 1. LOAD into registers (the rowwise mapping), zero the tail rows' words, stage the sub-tile
        words = []
        valids = []
        dsts = []
        dstsT = []
        for p in cutlass.range_constexpr(passes):
            r = cutlass.Int32(p * rows_per_pass) + grp  # row within the 32-token sub-tile
            s = s_blk + r
            valid = s < s_len
            s_ld = s if valid else last
            tok = tok0 + s_ld
            src_row = src_base + tok.to(cutlass.Int64) * src_tok_stride
            row_words = []
            for c in cutlass.range_constexpr(chunks):
                base = src_row + cutlass.Int64((c * lanes) * SRC_BYTES_PER_LANE) + src_lane_off
                cw = []
                for j in cutlass.range_constexpr(LOADS_PER_LANE):
                    for w in ld_global_v4(base + cutlass.Int64(j * 16), cutlass.Int32):
                        cw.append(w if valid else cutlass.Int32(0))  # tail row: contributes 0 to the row AND the column amax
                row_words.append(cw)
            words.append(row_words)
            valids.append(valid)
            dsts.append(dst_base + tok.to(cutlass.Int64) * dst_tok_stride)
            dstsT.append(dstT_base + tok.to(cutlass.Int64) * dstT_tok_stride)
        if cutlass.const_expr(want_col):
            for p in cutlass.range_constexpr(passes):
                r = cutlass.Int32(p * rows_per_pass) + grp
                for c in cutlass.range_constexpr(chunks):
                    woff = r * cutlass.Int32(words_per_row) + (cutlass.Int32(c * lanes) + lane) * cutlass.Int32(words_per_lane)
                    for j in cutlass.range_constexpr(LOADS_PER_LANE):
                        cw = words[p][c]
                        sStage.store(
                            cutlass.Vector.from_elements((cw[4 * j], cw[4 * j + 1], cw[4 * j + 2], cw[4 * j + 3]), cutlass.Int32),
                            woff + cutlass.Int32(4 * j),
                            vector_size=4,
                            alignment=16,
                        )
            nvvm.barrier_cta_sync()  # the staging tile is complete
            # 2. COLUMN pass: one thread per d over the 32 staged tokens
            if tidx < cutlass.Int32(d):
                d_idx = tidx
                par_hi = (d_idx & cutlass.Int32(1)) == cutlass.Int32(1)
                word_col = d_idx // cutlass.Int32(2)
                vals = []
                for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                    w = sStage.subview(cutlass.Int32(t * words_per_row) + word_col).load()
                    lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                    vals.append(hi if par_hi else lo)
                amax_c = abs_max_tree(vals)
                rcp_c, byte_c = e8m0_from_amax(amax_c)
                sScale.subview(d_idx).store(rcp_c)
                # the standalone columnwise arm's SF byte: plane p = d//128 at p*512 + ((d%128)%32)*16 + ((d%128)//32)*4 + tb
                plane = d_idx // cutlass.Int32(SF_TILE_ROWS)
                dm = d_idx % cutlass.Int32(SF_TILE_ROWS)
                sSFcol.store(byte_c.to(cutlass.Uint8), sf_atom_byte(dm, cutlass.Int32(tb), base=plane * cutlass.Int32(SF_ATOM_BYTES)))
                if cutlass.const_expr(transposed_second):
                    # [H*D, T]: row n = head*D + d_idx, its 32 tokens are 32 CONTIGUOUS bytes at n*T + (tok0 + s_blk); T % 32 == 0 makes the
                    # block entirely live or entirely padding (the host rule), so one branch covers it; a pad block stores nothing (SF 0x00)
                    if s_blk < s_len:
                        row_addr = dstT_base + d_idx.to(cutlass.Int64) * dstT_row_stride + (tok0 + s_blk).to(cutlass.Int64) * dstT_tok_stride
                        for half in cutlass.range_constexpr(COL_TOKENS_PER_UNIT // ELEMS_PER_LANE):
                            scaled_t = []
                            for t in cutlass.range_constexpr(ELEMS_PER_LANE):
                                scaled_t.append(vals[half * ELEMS_PER_LANE + t] * rcp_c)
                            p_t = fp32_to_fp8_pack(scaled_t, dtype=cutlass.Float8E4M3FN)
                            st_global_v4(row_addr + cutlass.Int64(half * ELEMS_PER_LANE), [p_t[0], p_t[1], p_t[2], p_t[3]], cutlass.Int32)
            nvvm.barrier_cta_sync()  # sScale / sSFcol complete; the staging tile is free for the next sub-tile
        # 3. ROW pass from the registers of step 1
        for p in cutlass.range_constexpr(passes):
            r = cutlass.Int32(p * rows_per_pass) + grp
            r_sf = cutlass.Int32(tb * COL_TOKENS_PER_UNIT) + r  # row within the 128-row SF tile
            for c in cutlass.range_constexpr(chunks):
                vals16 = []
                for w in words[p][c]:
                    lo, hi = f16x2_to_f32(w, dtype=mSrc.element_type)
                    vals16.append(lo)
                    vals16.append(hi)
                if cutlass.const_expr(want_row):
                    # the rowwise arm, verbatim: a lane pair (lane, lane^1) shares one 32-element block
                    amax = abs_max_tree(vals16)
                    partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, cutlass.Int32(1), 31, kind=nvvm.Shfl.BFLY))
                    amax = fmax_f32(amax, partner)
                    amax = amax if valids[p] else cutlass.Float32(0.0)  # tail row: SF 0x00, no data store
                    rcp, sf_byte = e8m0_from_amax(amax)
                    if valids[p]:
                        scaled = []
                        for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                            scaled.append(vals16[i] * rcp)
                        packed = fp32_to_fp8_pack(scaled, dtype=cutlass.Float8E4M3FN)
                        st_global_v4(
                            dsts[p] + cutlass.Int64((c * lanes) * DST_BYTES_PER_LANE) + dst_lane_off,
                            [packed[0], packed[1], packed[2], packed[3]],
                            cutlass.Int32,
                        )
                    c_idx = (cutlass.Int32(c * lanes) + lane) // cutlass.Int32(2)
                    if even:
                        sSFrow.store(sf_byte.to(cutlass.Uint8), sf_atom_offset(r_sf, c_idx))
                if cutlass.const_expr(want_col and not transposed_second):
                    # the SDPA columnwise payload, row-major: x * rcp_col[d] for each lane's 16 d (the column pass's rcps from sScale)
                    d_lane0 = (cutlass.Int32(c * lanes) + lane) * cutlass.Int32(ELEMS_PER_LANE)
                    rcps = []
                    for j in cutlass.range_constexpr(ELEMS_PER_LANE // 4):
                        v4 = sScale.load(d_lane0 + cutlass.Int32(4 * j), vector_size=4, alignment=16)
                        rcps.append(v4[0])
                        rcps.append(v4[1])
                        rcps.append(v4[2])
                        rcps.append(v4[3])
                    if valids[p]:
                        scaled_c = []
                        for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                            scaled_c.append(vals16[i] * rcps[i])
                        packed_c = fp32_to_fp8_pack(scaled_c, dtype=cutlass.Float8E4M3FN)
                        st_global_v4(
                            dstsT[p] + cutlass.Int64((c * lanes) * DST_BYTES_PER_LANE) + dst_lane_off,
                            [packed_c[0], packed_c[1], packed_c[2], packed_c[3]],
                            cutlass.Int32,
                        )

    # ---- the two SF bursts: lanes [0, burst_lanes) the rowwise tile, [burst_lanes, 2*burst_lanes) the columnwise one ----
    nvvm.barrier_cta_sync()
    if cutlass.const_expr(want_row):
        if tidx < cutlass.Int32(burst_lanes):
            _sf_burst_lane(mSf, sSFrow, tidx, bh, s_tile, head, n_tiles, v_sf_groups, n_c_atoms, d, False, sf_gemm, packed_seq)
    if cutlass.const_expr(want_col):
        if (tidx >= cutlass.Int32(burst_lanes)) & (tidx < cutlass.Int32(2 * burst_lanes)):
            _sf_burst_lane(mSfT, sSFcol, tidx - cutlass.Int32(burst_lanes), bh, s_tile, head, n_tiles, v_sf_groups, n_c_atoms_t, d, True, sf_gemm, packed_seq)


@cute.kernel
def frost_quantize_mxfp8(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token stride (slab slice or compact), head stride D
    mDst: cute.Tensor,  # [T, H, D] e4m3, own token stride (compact in the block), head stride D -- or the contiguous [H*D, T] matrix (transposed)
    mSf: cute.Tensor,  # [B*H*ceil(S/128)*4*D] uint8, F8_128x4 order per the module docstring (sf_blob_bytes(rows, K) bytes under sf_gemm)
    seq_len: cutlass.Int32,
    n_tiles: cutlass.Int32,  # ceil(S/128) == gridDim.y
    v_sf_groups: cutlass.Int32,  # B*H*n_tiles: the columnwise D-plane stride in atoms (unused rowwise)
    n_c_atoms: cutlass.Int32,  # APPENDED: atoms per 128-row band of the canonical blob, ceil((K/32)/4) (sf_gemm only; 0 under the SDPA layouts)
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    axis_col: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    sf_gemm: cutlass.Constexpr[bool],  # APPENDED: the GEMM-canonical SF blob over [rows, K] (the host folds the batch into the rows: n_bh == H)
    transposed: cutlass.Constexpr[bool],  # APPENDED: the columnwise arm's [H*D, T] store (sf_gemm and axis_col)
    mLens,  # APPENDED: packed -> the [B] / [B+1] int32 lengths tensor; dense -> None (a trace-time constant: the dense artifact keeps its parameter list)
    n_seq,  # APPENDED: packed -> Int32 B; dense -> None (a Python int would be a DYNAMIC Int32 argument = 4 more bytes of kernel parameters)
    lens_form,  # APPENDED: packed -> Int32, 0 = lengths / 1 = prefix sums; dense -> None
    packed_seq: cutlass.Constexpr[bool],  # APPENDED: the PACKED layout (module docstring): grid (H, n_cap), bh = head, s_tile = the packed tile
) -> None:
    tile_bytes = cutlass.const_expr(sf_tile_bytes(d))
    burst_lanes = cutlass.const_expr(tile_bytes // SF_BURST_BYTES)
    sSF = cutlass.Array(cutlass.Uint8, tile_bytes, alignment=16, space=cutlass.AddressSpace.smem)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    bh = cutlass.Int32(cute.arch.block_idx()[0])
    s_tile = cutlass.Int32(cute.arch.block_idx()[1])
    # the unit: (b, h, s_tile) of the padded grid, or (h, packed tile) -> this sequence's row 0 (tok0), the tail rows' clamp (last)
    # and the sequence length (s_len) -- ONE decode for the three bodies (_unit_coords)
    head, s0, tok0, last, s_len = _unit_coords(bh, s_tile, seq_len, h, packed_seq, mLens, n_seq, lens_form)
    src_tok_stride = cutlass.Int64(mSrc.stride[0]) * cutlass.Int64(2)
    src_base = mSrc.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1]) * cutlass.Int64(2)
    # Destination addressing.  [T, H, D]: token stride = the row pitch, head offset = head * D (stride[1]), d at +1.
    # Transposed [H*D, T] (strides (T, 1)): token stride 1, this head's first row n0 = head * D at head * D * T, d at + d * T.
    dst_tok_stride = cutlass.Int64(mDst.stride[1]) if cutlass.const_expr(transposed) else cutlass.Int64(mDst.stride[0])
    dst_d_stride = cutlass.Int64(mDst.stride[0]) if cutlass.const_expr(transposed) else cutlass.Int64(1)
    dst_base = (
        mDst.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(d) * dst_d_stride
        if cutlass.const_expr(transposed)
        else mDst.iterator.toint() + head.to(cutlass.Int64) * cutlass.Int64(mDst.stride[1])
    )

    if cutlass.const_expr(not axis_col):
        # ---- ROWWISE (Q/K): 32-element blocks along D (the hoisted body: the fused prologue's v8 job runs the same one) ----
        _rowwise_pass(mSrc, s_len, s0, tok0, last, src_base, src_tok_stride, dst_base, dst_tok_stride, d, threads_per_cta, sSF)
    else:
        # ---- COLUMNWISE (V): 32-token blocks along S -----------------------------------
        warps = cutlass.const_expr(threads_per_cta // WARP)
        slices = cutlass.const_expr(d // COL_D_PER_UNIT)
        units_per_warp = cutlass.const_expr(COL_TOKEN_BLOCKS * slices // warps)
        lane = tidx % cutlass.Int32(WARP)
        warp = tidx // cutlass.Int32(WARP)
        for i in cutlass.range_constexpr(units_per_warp):
            u = warp * cutlass.Int32(units_per_warp) + cutlass.Int32(i)
            tb = u // cutlass.Int32(slices)  # 32-token block within the tile
            ds = u % cutlass.Int32(slices)  # 64-d slice
            d0 = ds * cutlass.Int32(COL_D_PER_UNIT) + lane * cutlass.Int32(COL_D_PER_LANE)
            s_blk = s0 + tb * cutlass.Int32(COL_TOKENS_PER_UNIT)
            n_valid = s_len - s_blk  # tokens t < n_valid are in range (may be <= 0 or >= 32)
            # Token addresses walk INCREMENTALLY (one 64-bit add per token, not a 64-bit multiply each);
            # a tail token's load is redirected to the batch's last row and its word zeroed.
            src_addr = src_base + (tok0 + s_blk).to(cutlass.Int64) * src_tok_stride + d0.to(cutlass.Int64) * cutlass.Int64(2)
            src_last = src_base + (tok0 + last).to(cutlass.Int64) * src_tok_stride + d0.to(cutlass.Int64) * cutlass.Int64(2)
            dst_addr = (
                dst_base + (tok0 + s_blk).to(cutlass.Int64) * dst_tok_stride + d0.to(cutlass.Int64) * dst_d_stride
                if cutlass.const_expr(transposed)
                else dst_base + (tok0 + s_blk).to(cutlass.Int64) * dst_tok_stride + d0.to(cutlass.Int64)
            )
            words = []
            valids = []
            dsts = []
            for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                valid = cutlass.Int32(t) < n_valid  # warp-uniform
                w = ld_global(src_addr if valid else src_last, cutlass.Int32)
                words.append(w if valid else cutlass.Int32(0))  # tail token: contributes 0 to the block amax
                valids.append(valid)
                src_addr = src_addr + src_tok_stride
                if cutlass.const_expr(not transposed):
                    dsts.append(dst_addr)
                    dst_addr = dst_addr + dst_tok_stride
            los = []
            his = []
            for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                lo, hi = f16x2_to_f32(words[t], dtype=mSrc.element_type)
                los.append(lo)
                his.append(hi)
            rcp0, rcp1, sf_pair = e8m0_pair(abs_max_tree(los), abs_max_tree(his))
            if cutlass.const_expr(transposed):
                # [H*D, T]: each lane owns two output rows n = head*D + d0 and n + 1, whose 32 tokens are 32 CONTIGUOUS bytes
                # at n*T + t_blk*32 -- two st.global.v4 per row (fp32_to_fp8_pack: the same cvt.rn.satfinite.e4m3x2 as the b16
                # path, byte i = token i).  T % 32 == 0 (run_) makes every 32-token block entirely live or entirely padding
                # (n_valid >= 32 or <= 0, warp-uniform), so one branch covers the block; a pad block stores nothing and its SF
                # byte is 0x00 from the zeroed words.
                if valids[0]:
                    for half in cutlass.range_constexpr(COL_TOKENS_PER_UNIT // ELEMS_PER_LANE):
                        lo_scaled = []
                        hi_scaled = []
                        for t in cutlass.range_constexpr(ELEMS_PER_LANE):
                            lo_scaled.append(los[half * ELEMS_PER_LANE + t] * rcp0)
                            hi_scaled.append(his[half * ELEMS_PER_LANE + t] * rcp1)
                        p_lo = fp32_to_fp8_pack(lo_scaled, dtype=cutlass.Float8E4M3FN)
                        p_hi = fp32_to_fp8_pack(hi_scaled, dtype=cutlass.Float8E4M3FN)
                        half_off = cutlass.Int64(half * ELEMS_PER_LANE)
                        st_global_v4(dst_addr + half_off, [p_lo[0], p_lo[1], p_lo[2], p_lo[3]], cutlass.Int32)
                        st_global_v4(dst_addr + dst_d_stride + half_off, [p_hi[0], p_hi[1], p_hi[2], p_hi[3]], cutlass.Int32)
            else:
                for t in cutlass.range_constexpr(COL_TOKENS_PER_UNIT):
                    if valids[t]:
                        st_global_b16(dsts[t], fp32_to_fp8x2(los[t] * rcp0, his[t] * rcp1))
            # SF SMEM bytes: plane p = d//128 at p*512 + ((d%128)%32)*16 + ((d%128)//32)*4 + tb (sf_layout); d0+1 sits 16 B after d0.
            plane = d0 // cutlass.Int32(SF_TILE_ROWS)
            dm = d0 % cutlass.Int32(SF_TILE_ROWS)
            sf_off = sf_atom_byte(dm, tb, base=plane * cutlass.Int32(SF_ATOM_BYTES))
            sSF.store((sf_pair & cutlass.Int32(0xFF)).to(cutlass.Uint8), sf_off)
            sSF.store(((sf_pair >> 8) & cutlass.Int32(0xFF)).to(cutlass.Uint8), sf_off + cutlass.Int32(16))

    # ---- SF burst: the whole tile leaves SMEM as 16-B lanes (the hoisted body: one lane per call, the caller gates the range) ----
    nvvm.barrier_cta_sync()
    if tidx < cutlass.Int32(burst_lanes):
        _sf_burst_lane(mSf, sSF, tidx, bh, s_tile, head, n_tiles, v_sf_groups, n_c_atoms, d, axis_col, sf_gemm, packed_seq)


@cute.jit
def quantize_mxfp8_launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    sf: cute.Tensor,
    seq_len: cutlass.Int32,
    n_tiles: cutlass.Int32,
    v_sf_groups: cutlass.Int32,
    n_bh: cutlass.Int32,
    n_c_atoms: cutlass.Int32,
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    axis_col: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    sf_gemm: cutlass.Constexpr[bool],
    transposed: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    lens=None,  # APPENDED (packed): the lengths tensor, Int32 B, Int32 form -- dense passes None / None / None (trace-time constants)
    n_seq=None,
    lens_form=None,
    packed_seq: cutlass.Constexpr[bool] = False,
):
    frost_quantize_mxfp8(
        src, dst, sf, seq_len, n_tiles, v_sf_groups, n_c_atoms, h, d, axis_col, threads_per_cta, sf_gemm, transposed, lens, n_seq, lens_form, packed_seq
    ).launch(grid=(n_bh, n_tiles, 1), block=(threads_per_cta, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Host API (frozen in the PR-B plan, section 2.5)
# ---------------------------------------------------------------------------

compiled_cache = {}


@dataclass(frozen=True)
class QuantizeMxfp8Recipe:
    """Build-time facts of one MXFP8 quantize launch; ``batch`` / ``seq_len`` ride in at run time."""

    dtype_in: object  # torch dtype of the source (bf16 / f16)
    h: int
    d: int
    axis: str  # "row" (Q/K: 32-blocks along D) | "col" (V: 32-blocks along S)
    compiled: object = None
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA
    sf_layout: str = SF_LAYOUT_SDPA  # APPENDED: "sdpa" (the SDPA descriptors' order) | "gemm" (the canonical blob, batch folded into the rows)
    transposed: bool = False  # APPENDED: the columnwise arm's physically transposed [H*D, T] store (needs axis="col", sf_layout="gemm")
    packed: bool = False  # APPENDED: the PACKED (THD) layout -- per-sequence-tile-padded SF tiles in cu_seqlens order (sf_layout="sdpa" only)


def compile_quantize_mxfp8(
    *,
    dtype_in,
    h: int,
    d: int,
    axis: str,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    compile_options: str = COMPILE_OPTIONS,
    sf_layout: str = SF_LAYOUT_SDPA,
    transposed: bool = False,
    packed: bool = False,
) -> QuantizeMxfp8Recipe:
    """Build from SHAPES ALONE -- no allocation, no launch.  E4M3 codes + E8M0 SF only.

    ``compile_options`` is a dev knob: ``"--enable-tvm-ffi --gpu-arch sm_107a"`` trace-compiles for
    Rubin on any box (the SASS spill check); production leaves the default.

    ``sf_layout`` (appended; default = today's artifact): ``"sdpa"`` writes the SDPA descriptors' SF order,
    ``"gemm"`` the GEMM-canonical F8_128x4 blob over the ``[rows, K]`` matrix with the batch folded into the
    rows -- ``(T, H*D)`` rowwise, ``(H*D, T)`` with ``transposed=True`` (module docstring).  ``transposed``
    (appended) needs ``axis="col"`` and ``sf_layout="gemm"``: the columnwise arm stores the e4m3 codes
    PHYSICALLY TRANSPOSED as the contiguous ``[H*D, T]`` matrix (the block-scale wgrad's A operand).
    ``packed`` (appended) selects the PACKED-sequence arm of the SDPA layouts (module docstring, "PACKED sequences"):
    the lengths tensor travels at ``run_quantize_mxfp8(..., seq_lens=, num_sequences=, cu_seqlens=)``; refused with
    ``sf_layout="gemm"`` / ``transposed``.  The dense artifact is untouched: the packed parameters are trace-time
    constants there (``None`` / ``0``), so its kernel parameter list and SASS are the same.
    """
    global _FAKE_STREAM
    validate_mode(axis, sf_layout, transposed)
    validate_packed_mode(sf_layout, transposed, packed)
    validate_shape(d, threads_per_cta, axis)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"quantize_mxfp8 serves bf16/f16 sources only, got {dtype_in}")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    axis_col = axis == AXIS_COL
    sf_gemm = sf_layout == SF_LAYOUT_GEMM
    # today's key is the prefix; the three appended fields keep the default artifact's entry where it was
    key = (str(dtype_in), int(h), int(d), axis, int(threads_per_cta), compile_options, current_device(), sf_layout, bool(transposed), bool(packed))
    if key not in compiled_cache:
        tok = cute.sym_int()
        # Source: a column slice of the projection slab (token stride N_qkvg) or compact;
        # destination: compact in the block but kept symbolic so one artifact serves both -- or, transposed, the
        # contiguous [H*D, T] matrix (row pitch T symbolic: one artifact serves every T).
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        if transposed:
            dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(h * d, tok), stride=(cute.sym_int(), 1), assumed_align=16)
        else:
            dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)
        sf = cute.runtime.make_fake_tensor(dtype=_convert_to_cutlass_data_type(torch.uint8), shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        # packed: the [B] / [B+1] int32 lengths (any length, 4-byte aligned) + B + the form as RUNTIME scalars; dense: trace-time constants
        # (None x3) -- the dense artifact's kernel parameter list and SASS are unchanged (a Python int here would be a dynamic Int32).
        lens = cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride=(1,), assumed_align=4) if packed else None
        compiled_cache[key] = cute.compile(
            quantize_mxfp8_launch,
            src,
            dst,
            sf,
            cutlass.Int32(0),  # seq_len     ) runtime; the zeros pin the TYPE only
            cutlass.Int32(0),  # n_tiles     )
            cutlass.Int32(0),  # v_sf_groups )
            cutlass.Int32(0),  # n_bh        )
            cutlass.Int32(0),  # n_c_atoms   )
            int(h),
            int(d),
            bool(axis_col),
            int(threads_per_cta),
            bool(sf_gemm),
            bool(transposed),
            _FAKE_STREAM,
            lens,
            cutlass.Int32(0) if packed else None,  # n_seq     ) packed: runtime; dense: a trace-time constant (no kernel parameter)
            cutlass.Int32(0) if packed else None,  # lens_form )
            bool(packed),
            options=compile_options,
        )
    return QuantizeMxfp8Recipe(
        dtype_in=dtype_in,
        h=int(h),
        d=int(d),
        axis=axis,
        compiled=compiled_cache[key],
        threads_per_cta=int(threads_per_cta),
        sf_layout=sf_layout,
        transposed=bool(transposed),
        packed=bool(packed),
    )


def _check_packed_lengths(
    packed: bool, *, seq_lens: Optional[torch.Tensor], num_sequences: int, cu_seqlens: bool, batch: Optional[int], seq_len: Optional[int], device, t: int
) -> tuple:
    """The two length contracts, typed BOTH ways (Rule 1): a PACKED artifact takes ``seq_lens`` (a contiguous 1-D int32 CUDA tensor on
    the source's device with ``B`` entries, or ``B+1`` prefix sums under ``cu_seqlens``) and ``num_sequences >= 1`` and REFUSES
    ``batch`` / ``seq_len``; a dense one takes ``batch`` / ``seq_len`` and refuses the packed trio.  The lengths' VALUES are never read
    (CUDA-graph capturable; the caller contract is every length in ``[0, T]`` and ``sum == T``).  Returns ``(batch, seq_len)`` for the
    dense arm, ``(num_sequences, 0)`` for the packed one."""
    if not packed:
        if seq_lens is not None or num_sequences or cu_seqlens:
            raise ValueError("seq_lens / num_sequences / cu_seqlens are the PACKED arm's (compile_quantize_mxfp8(packed=True)); this artifact is dense")
        if batch is None or seq_len is None:
            raise ValueError("batch and seq_len are required for a dense artifact")
        if batch <= 0 or seq_len <= 0:
            raise ValueError(f"batch and seq_len must be positive, got batch={batch} seq_len={seq_len}")
        return int(batch), int(seq_len)
    if batch is not None or seq_len is not None:
        raise ValueError("this artifact is packed=True: the geometry is (T, num_sequences, seq_lens); batch / seq_len are refused (Rule 1)")
    if int(num_sequences) < 1:
        raise ValueError(
            f"packed=True needs num_sequences >= 1 (B: the lengths tensor has B entries, or B+1 prefix sums under cu_seqlens), got {num_sequences}"
        )
    n = int(num_sequences) + (1 if cu_seqlens else 0)
    form = "[B+1] int32 prefix sums" if cu_seqlens else "[B] int32 lengths"
    dev = torch.device(device)
    ok = (
        isinstance(seq_lens, torch.Tensor)
        and seq_lens.dtype == torch.int32
        and seq_lens.dim() == 1
        and seq_lens.is_contiguous()
        and seq_lens.numel() == n
        and seq_lens.is_cuda
        and (dev.index is None or seq_lens.device.index == dev.index)
    )
    if not ok:
        got = (
            f"dtype {seq_lens.dtype}, shape {tuple(seq_lens.shape)}, strides {tuple(seq_lens.stride())}, device {seq_lens.device}"
            if isinstance(seq_lens, torch.Tensor)
            else type(seq_lens).__name__
        )
        raise ValueError(f"packed=True: seq_lens must be a contiguous 1-D int32 CUDA tensor of {n} elements on {dev} ({form}), got {got}")
    if seq_lens.data_ptr() % 4:
        raise ValueError("seq_lens must be 4-byte aligned")
    if t < 1:
        raise ValueError(f"packed=True needs at least one token, got T={t}")
    return int(num_sequences), 0


def run_quantize_mxfp8(
    r: QuantizeMxfp8Recipe,
    src: torch.Tensor,
    dst: torch.Tensor,
    sf: torch.Tensor,
    *,
    batch: Optional[int] = None,
    seq_len: Optional[int] = None,
    stream,
    seq_lens: Optional[torch.Tensor] = None,
    num_sequences: int = 0,
    cu_seqlens: bool = False,
) -> None:
    """Launch.  ``src`` ``[T, H, D]`` bf16/f16 (strided ok), ``dst`` ``[T, H, D]`` ``float8_e4m3fn`` compact,
    ``sf`` uint8 with ``numel == B*H*ceil(S/128)*4*D`` (any shape, contiguous), ``T == batch * seq_len``.

    Under ``sf_layout="gemm"`` the batch folds into the rows: ``sf.numel() == proj_gemm.sf_blob_bytes(rows, K)``
    with ``(rows, K) = (T, H*D)`` rowwise / ``(H*D, T)`` transposed -- never the SDPA count.  Under
    ``transposed=True`` ``dst`` is the CONTIGUOUS e4m3 ``[H*D, T]`` matrix (strides ``(T, 1)``) and
    ``T % 32 == 0`` (whole 32-token blocks; a ragged ``T`` is a typed error).  Every check both ways.

    Under ``packed=True`` (appended kwargs, keyword-only): ``seq_lens`` is the ``[B]`` int32 lengths tensor (or the
    ``[B+1]`` int32 prefix sums under ``cu_seqlens=True``, any base) on ``src``'s device, ``num_sequences = B``, and
    ``sf.numel() == sf_bytes_packed(B, H, T, D)`` (``H * ((T + 127*B) // 128) * 4*D``: the capacity bound, zero-filled past
    the live total); ``batch`` / ``seq_len`` are refused, and the dense arm refuses the packed trio (Rule 1).  The
    lengths are never read on the host.

    Cheap host checks only; every one of them guards a wild write or a silent wrong
    answer at the tvm-ffi boundary, which reports neither.
    """
    if r.compiled is None:
        raise ValueError("recipe was not built by compile_quantize_mxfp8")
    gemm = r.sf_layout == SF_LAYOUT_GEMM
    packed = bool(getattr(r, "packed", False))
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    if dst.dtype != torch.float8_e4m3fn:
        raise ValueError(f"dst must be torch.float8_e4m3fn, got {dst.dtype}")
    t = int(src.shape[0])
    batch, seq_len = _check_packed_lengths(
        packed, seq_lens=seq_lens, num_sequences=num_sequences, cu_seqlens=cu_seqlens, batch=batch, seq_len=seq_len, device=src.device, t=t
    )
    if src.ndim != 3 or int(src.shape[1]) != r.h or int(src.shape[2]) != r.d:
        raise ValueError(f"src must be [T, H={r.h}, D={r.d}], got {tuple(src.shape)}")
    if src.stride(2) != 1 or src.stride(1) != r.d:
        raise ValueError(f"src must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {src.stride()}")
    if not packed and t != batch * seq_len:
        raise ValueError(f"T must equal batch*seq_len: src T={t}, batch*seq_len={batch * seq_len}")
    if (src.stride(0) * 2) % 16:
        raise ValueError(f"token strides must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16/f16)")
    if r.transposed:
        # the physically transposed [H*D, T] e4m3 matrix: K = T contiguous (the block-scale GEMM's K-major A operand)
        n_rows = r.h * r.d
        if dst.ndim != 2 or tuple(dst.shape) != (n_rows, t) or not dst.is_contiguous():
            raise ValueError(
                f"this artifact is transposed=True: dst must be the contiguous [H*D={n_rows}, T={t}] float8_e4m3fn matrix (strides ({t}, 1)), "
                f"got shape {tuple(dst.shape)} strides {dst.stride()}"
            )
        if t % SF_BLOCK:
            raise ValueError(
                f"the transposed quantization writes whole {SF_BLOCK}-token blocks (32 contiguous e4m3 codes per column): "
                f"T = batch*seq_len must be a multiple of {SF_BLOCK}, got T={t}"
            )
    else:
        if dst.ndim != 3 or int(dst.shape[1]) != r.h or int(dst.shape[2]) != r.d:
            raise ValueError(f"dst must be [T, H={r.h}, D={r.d}] (this artifact is transposed=False), got {tuple(dst.shape)}")
        if dst.stride(2) != 1 or dst.stride(1) != r.d:
            raise ValueError(f"dst must have head stride D={r.d} and element stride 1 (compact), got strides {dst.stride()}")
        if int(dst.shape[0]) != t:
            raise ValueError(
                f"T must equal batch*seq_len for src and dst: src T={t}, dst T={int(dst.shape[0])}, batch*seq_len={batch * seq_len}"
                if not packed
                else f"src and dst must hold the same T packed tokens: src T={t}, dst T={int(dst.shape[0])}"
            )
        if dst.stride(0) % 16:
            raise ValueError(f"token strides must keep every row 16-byte aligned: dst {dst.stride(0)} elems (fp8)")
    for name, ten in (("src", src), ("dst", dst)):
        if not ten.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {ten.device}")
        if ten.data_ptr() % 16:
            raise ValueError(
                f"{name} must be 16-byte aligned (its rows move as 16-byte vectors); a slab column offset that is not a multiple of 16 bytes is not"
            )
    if sf.dtype != torch.uint8:
        raise ValueError(f"sf must be torch.uint8 (E8M0 bytes in F8_128x4 order), got {sf.dtype}")
    if not sf.is_contiguous() or not sf.is_cuda:
        raise ValueError("sf must be a contiguous CUDA tensor")
    if gemm:
        rows, k = (r.h * r.d, t) if r.transposed else (t, r.h * r.d)
        need = sf_blob_bytes(rows, k, SF_BLOCK)
        if sf.numel() != need:
            raise ValueError(
                f"sf must be the padded F8_128x4 blob of sf_blob_bytes(rows={rows}, K={k}) = {need} bytes (sf_layout='gemm', "
                f"{'transposed: (H*D, T)' if r.transposed else 'rowwise: (T, H*D)'}; the batch folds into the rows), got {sf.numel()}"
            )
    elif packed:
        need = sf_bytes_packed(batch, r.h, t, r.d)
        if sf.numel() != need:
            raise ValueError(
                f"sf must hold H*((T + 127*B)//128)*{sf_tile_bytes(r.d)} = {need} bytes for the PACKED layout at T={t} B={batch} H={r.h} D={r.d} "
                f"(the capacity bound on sum_b ceil(s_b/128) tiles per head), got {sf.numel()}"
            )
    else:
        need = sf_bytes(batch, r.h, seq_len, r.d)
        if sf.numel() != need:
            raise ValueError(
                f"sf must hold B*H*ceil(S/128)*{sf_tile_bytes(r.d)} = {need} bytes for batch={batch} H={r.h} S={seq_len} D={r.d}, got {sf.numel()}"
            )
    if sf.data_ptr() % 16:
        raise ValueError("sf must be 16-byte aligned (the SF tile leaves SMEM as 16-byte bursts)")
    if not (src.device == dst.device == sf.device):
        raise ValueError(f"src, dst and sf must live on one device, got {src.device}, {dst.device}, {sf.device}")
    if gemm:
        # the batch folds into the rows: ONE sequence of T rows x H heads -> grid (H, ceil(T/128)); the atoms per 128-row band
        # of the padded blob come from the one source (proj_gemm.sf_padded_dims), never a literal
        launch_seq, launch_bh = t, r.h
        n_c_atoms = sf_padded_dims(rows, k, SF_BLOCK)[1] // SF_ATOM_COLS
        tiles = n_sf_tiles(launch_seq)
    elif packed:
        # the packed tile axis: grid (H, n_cap) over the CAPACITY bound; seq_len / v_sf_groups / n_c_atoms are unused by the arm
        launch_seq, launch_bh, n_c_atoms = 0, r.h, 0
        tiles = n_sf_tiles_packed_cap(t, batch)
    else:
        launch_seq, launch_bh, n_c_atoms = seq_len, batch * r.h, 0
        tiles = n_sf_tiles(launch_seq)
    args = [
        src,
        dst,
        sf.view(-1),
        cutlass.Int32(launch_seq),
        cutlass.Int32(tiles),
        cutlass.Int32(launch_bh * tiles),
        cutlass.Int32(launch_bh),
        cutlass.Int32(n_c_atoms),
        cuda.CUstream(int(stream)),
    ]
    if packed:
        args += [seq_lens, cutlass.Int32(batch), cutlass.Int32(1 if cu_seqlens else 0)]
    r.compiled(*args)


# ---------------------------------------------------------------------------
# The DUAL-AXIS kernel: rowwise + columnwise MXFP8 from ONE read (the standalone shape of quantize_mxfp8_dual_body)
# ---------------------------------------------------------------------------


@cute.kernel
def frost_quantize_mxfp8_dual(
    mSrc: cute.Tensor,  # [T, H, D] bf16/f16, own token stride
    mDst: cute.Tensor,  # [T, H, D] e4m3 compact: the rowwise payload
    mSf: cute.Tensor,  # the rowwise SF blob (SDPA tiles | canonical over (T, H*D))
    mDstT: cute.Tensor,  # [T, H, D] e4m3 compact (SDPA columnwise) | the contiguous [H*D, T] matrix (canonical transposed)
    mSfT: cute.Tensor,  # the columnwise SF blob (SDPA D-plane-major | canonical over (H*D, T))
    seq_len: cutlass.Int32,
    n_tiles: cutlass.Int32,
    v_sf_groups: cutlass.Int32,
    n_c_atoms: cutlass.Int32,
    n_c_atoms_t: cutlass.Int32,
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    sf_gemm: cutlass.Constexpr[bool],
    transposed_second: cutlass.Constexpr[bool],
    mLens,  # APPENDED: packed -> the [B] / [B+1] int32 lengths tensor; dense -> None (the dense artifact keeps its parameter list)
    n_seq,  # APPENDED: packed -> Int32 B; dense -> None
    lens_form,  # APPENDED: packed -> Int32, 0 = lengths / 1 = prefix sums; dense -> None
    packed_seq: cutlass.Constexpr[bool],  # APPENDED: the PACKED SDPA pair (grid (H, n_cap))
) -> None:
    """Grid ``(B*H, ceil(S/128))`` (``(H, ceil(T/128))`` canonical, ``(H, n_cap)`` packed) -- one CTA per SF unit, as the standalone
    kernel; the SMEM of the dual body's table (``sStage`` 32 x D bf16, ``sScale`` D fp32, the two SF tiles), allocated here."""
    sStage = cutlass.Array(cutlass.Int32, SF_BLOCK * d // 2, alignment=16, space=cutlass.AddressSpace.smem)
    sScale = cutlass.Array(cutlass.Float32, d, alignment=16, space=cutlass.AddressSpace.smem)
    sSFrow = cutlass.Array(cutlass.Uint8, sf_tile_bytes(d), alignment=16, space=cutlass.AddressSpace.smem)
    sSFcol = cutlass.Array(cutlass.Uint8, sf_tile_bytes(d), alignment=16, space=cutlass.AddressSpace.smem)
    quantize_mxfp8_dual_body(
        mSrc,
        mDst,
        mSf,
        mDstT,
        mSfT,
        seq_len,
        n_tiles,
        v_sf_groups,
        n_c_atoms,
        n_c_atoms_t,
        cutlass.Int32(cute.arch.block_idx()[0]),
        cutlass.Int32(cute.arch.block_idx()[1]),
        h,
        d,
        threads_per_cta,
        sf_gemm,
        transposed_second,
        True,
        True,
        sStage,
        sScale,
        sSFrow,
        sSFcol,
        mLens,
        n_seq,
        lens_form,
        packed_seq,
    )


@cute.jit
def quantize_mxfp8_dual_launch(
    src: cute.Tensor,
    dst: cute.Tensor,
    sf: cute.Tensor,
    dst_t: cute.Tensor,
    sf_t: cute.Tensor,
    seq_len: cutlass.Int32,
    n_tiles: cutlass.Int32,
    v_sf_groups: cutlass.Int32,
    n_bh: cutlass.Int32,
    n_c_atoms: cutlass.Int32,
    n_c_atoms_t: cutlass.Int32,
    h: cutlass.Constexpr[int],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    sf_gemm: cutlass.Constexpr[bool],
    transposed_second: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    lens=None,  # APPENDED (packed): the lengths tensor, Int32 B, Int32 form -- dense passes None / None / None (trace-time constants)
    n_seq=None,
    lens_form=None,
    packed_seq: cutlass.Constexpr[bool] = False,
):
    frost_quantize_mxfp8_dual(
        src,
        dst,
        sf,
        dst_t,
        sf_t,
        seq_len,
        n_tiles,
        v_sf_groups,
        n_c_atoms,
        n_c_atoms_t,
        h,
        d,
        threads_per_cta,
        sf_gemm,
        transposed_second,
        lens,
        n_seq,
        lens_form,
        packed_seq,
    ).launch(grid=(n_bh, n_tiles, 1), block=(threads_per_cta, 1, 1), stream=stream)


dual_compiled_cache = {}


@dataclass(frozen=True)
class QuantizeMxfp8DualRecipe:
    """Build-time facts of one DUAL-AXIS launch: the rowwise quantization of ``[T, H, D]`` plus its columnwise twin from one read.

    ``sf_layout="sdpa"`` pairs the SDPA rowwise tiles with the SDPA columnwise D-plane-major atoms (both payloads row-major
    ``[T, H, D]``: the row's ``do8`` / ``do_T8``); ``sf_layout="gemm"`` pairs the canonical rowwise blob over ``(T, H*D)`` with the
    canonical TRANSPOSED store (``transposed_second=True``, the only canonical columnwise form: the ``[H*D, T]`` matrix + the blob
    over ``(H*D, T)``; the block-scale wgrad's A)."""

    dtype_in: object
    h: int
    d: int
    compiled: object = None
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA
    sf_layout: str = SF_LAYOUT_SDPA
    transposed_second: bool = False
    packed: bool = False  # APPENDED: the PACKED (THD) layout of the SDPA pair (sf_layout="sdpa" only)


def validate_dual_mode(sf_layout: str, transposed_second: bool) -> None:
    """The dual arm's two modes, typed both ways: the first half is always the rowwise arm (``validate_mode("row", ...)``), the second
    the columnwise arm in the one form each layout has -- row-major under the SDPA layout, TRANSPOSED under the canonical one."""
    validate_mode(AXIS_ROW, sf_layout, False)
    validate_mode(AXIS_COL, sf_layout, transposed_second)  # transposed_second needs "gemm"; "gemm" + col needs transposed_second (one form)


def compile_quantize_mxfp8_dual(
    *,
    dtype_in,
    h: int,
    d: int,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    compile_options: str = COMPILE_OPTIONS,
    sf_layout: str = SF_LAYOUT_SDPA,
    transposed_second: bool = False,
    packed: bool = False,
) -> QuantizeMxfp8DualRecipe:
    """Build the dual-axis artifact from SHAPES ALONE -- no allocation, no launch.  The same knobs as :func:`compile_quantize_mxfp8`
    minus ``axis`` (both axes), plus ``transposed_second`` (the canonical pair's second store is the ``[H*D, T]`` matrix: REQUIRED
    under ``sf_layout="gemm"``, refused under ``"sdpa"``, both typed by :func:`validate_dual_mode`); ``packed`` (appended) the
    PACKED-sequence arm of the SDPA pair (refused under ``"gemm"``)."""
    global _FAKE_STREAM
    validate_dual_mode(sf_layout, transposed_second)
    validate_packed_mode(sf_layout, transposed_second, packed)
    validate_dual_shape(d, threads_per_cta)
    if dtype_in not in (torch.bfloat16, torch.float16):
        raise ValueError(f"quantize_mxfp8 serves bf16/f16 sources only, got {dtype_in}")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    sf_gemm = sf_layout == SF_LAYOUT_GEMM
    key = ("dual", str(dtype_in), int(h), int(d), int(threads_per_cta), compile_options, current_device(), sf_layout, bool(transposed_second), bool(packed))
    if key not in dual_compiled_cache:
        tok = cute.sym_int()
        src = fake_rowmajor_dynamic_token_stride(dtype_in, tok, h, d)
        dst = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)
        if transposed_second:
            dst_t = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(h * d, tok), stride=(cute.sym_int(), 1), assumed_align=16)
        else:
            dst_t = cute.runtime.make_fake_tensor(dtype=cutlass.Float8E4M3FN, shape=(tok, h, d), stride=(cute.sym_int(), d, 1), assumed_align=16)
        sf = cute.runtime.make_fake_tensor(dtype=_convert_to_cutlass_data_type(torch.uint8), shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        sf_t = cute.runtime.make_fake_tensor(dtype=_convert_to_cutlass_data_type(torch.uint8), shape=(cute.sym_int(),), stride=(1,), assumed_align=16)
        lens = cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride=(1,), assumed_align=4) if packed else None
        dual_compiled_cache[key] = cute.compile(
            quantize_mxfp8_dual_launch,
            src,
            dst,
            sf,
            dst_t,
            sf_t,
            cutlass.Int32(0),  # seq_len      ) runtime; the zeros pin the TYPE only
            cutlass.Int32(0),  # n_tiles      )
            cutlass.Int32(0),  # v_sf_groups  )
            cutlass.Int32(0),  # n_bh         )
            cutlass.Int32(0),  # n_c_atoms    )
            cutlass.Int32(0),  # n_c_atoms_t  )
            int(h),
            int(d),
            int(threads_per_cta),
            bool(sf_gemm),
            bool(transposed_second),
            _FAKE_STREAM,
            lens,
            cutlass.Int32(0) if packed else None,  # n_seq     ) packed: runtime; dense: a trace-time constant (no kernel parameter)
            cutlass.Int32(0) if packed else None,  # lens_form )
            bool(packed),
            options=compile_options,
        )
    return QuantizeMxfp8DualRecipe(
        dtype_in=dtype_in,
        h=int(h),
        d=int(d),
        compiled=dual_compiled_cache[key],
        threads_per_cta=int(threads_per_cta),
        sf_layout=sf_layout,
        transposed_second=bool(transposed_second),
        packed=bool(packed),
    )


def check_dual_operands(
    r,
    src,
    dst,
    sf,
    dst_t,
    sf_t,
    *,
    batch: Optional[int] = None,
    seq_len: Optional[int] = None,
    want_row: bool = True,
    want_col: bool = True,
    seq_lens: Optional[torch.Tensor] = None,
    num_sequences: int = 0,
    cu_seqlens: bool = False,
) -> tuple:
    """The dual launch's host checks (``run_quantize_mxfp8``'s for the rowwise half, the transposed arm's for a ``[H*D, T]`` second
    destination, the columnwise SDPA count for a row-major one), shared by the standalone launch and the fused epilogue.  ``want_row`` /
    ``want_col`` (appended; both True for the standalone launch, which always traces both halves) name the halves the artifact traces:
    a folded-out half passes ``None`` for its pair and is NOT checked -- no stand-in tensor is ever allocated for it -- and a tensor
    bound to a folded-out half is refused (the kernel would silently ignore it).  Returns ``(launch_seq, launch_bh, n_c_atoms,
    n_c_atoms_t)`` -- each canonical atom count is asked for ONLY when its half is traced (0 otherwise): the transposed blob's count
    needs ``T % 32 == 0``, which a rowwise-only launch at a ragged ``T`` (a dgrad-only block at T = 1000) does not have and must not be
    asked for -- the kernel reads the count inside the folded-out arm alone; raises ``ValueError``.  Under a PACKED recipe (appended
    kwargs) ``seq_lens`` / ``num_sequences`` / ``cu_seqlens`` replace ``batch`` / ``seq_len`` and both SDPA blobs are
    ``sf_bytes_packed(B, H, T, D)`` (:func:`_check_packed_lengths`, Rule 1 both ways); ``launch_seq`` is then 0 and the caller takes
    ``n_sf_tiles_packed_cap(T, B)`` as the grid's tile extent."""
    gemm = r.sf_layout == SF_LAYOUT_GEMM
    packed = bool(getattr(r, "packed", False))
    if not (want_row or want_col):
        raise ValueError("check_dual_operands: at least one half (want_row / want_col) must be traced")
    if src.dtype != r.dtype_in:
        raise ValueError(f"src is {src.dtype} but this artifact was compiled for {r.dtype_in}")
    t = int(src.shape[0])
    batch, seq_len = _check_packed_lengths(
        packed, seq_lens=seq_lens, num_sequences=num_sequences, cu_seqlens=cu_seqlens, batch=batch, seq_len=seq_len, device=src.device, t=t
    )
    if src.ndim != 3 or int(src.shape[1]) != r.h or int(src.shape[2]) != r.d:
        raise ValueError(f"src must be [T, H={r.h}, D={r.d}], got {tuple(src.shape)}")
    if src.stride(2) != 1 or src.stride(1) != r.d:
        raise ValueError(f"src must have head stride D={r.d} and element stride 1 (a column slice of the slab or compact), got strides {src.stride()}")
    if not packed and t != batch * seq_len:
        raise ValueError(f"T must equal batch*seq_len: src T={t}, batch*seq_len={batch * seq_len}")
    if (src.stride(0) * 2) % 16:
        raise ValueError(f"token strides must keep every row 16-byte aligned: src {src.stride(0)} elems (bf16/f16)")
    for name, ten, wanted in (("dst", dst, want_row), ("sf", sf, want_row), ("dst_T", dst_t, want_col), ("sf_T", sf_t, want_col)):
        if not wanted and ten is not None:
            raise ValueError(
                f"{name} is bound but its half is not traced by this artifact (want_row={want_row}, want_col={want_col}); the kernel would silently ignore it"
            )
        if wanted and ten is None:
            raise ValueError(f"{name} must be bound: its half is traced by this artifact (want_row={want_row}, want_col={want_col})")
    payloads = ([("dst", dst)] if want_row else []) + ([("dst_T", dst_t)] if want_col else [])
    for name, ten in payloads:
        if ten.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{name} must be torch.float8_e4m3fn, got {ten.dtype}")
        if not ten.is_cuda or ten.data_ptr() % 16:
            raise ValueError(f"{name} must be a 16-byte-aligned CUDA tensor (its rows move as 16-byte vectors)")
    if want_row:
        if dst.ndim != 3 or int(dst.shape[0]) != t or int(dst.shape[1]) != r.h or int(dst.shape[2]) != r.d or dst.stride(2) != 1 or dst.stride(1) != r.d:
            raise ValueError(f"dst must be the compact e4m3 [T={t}, H={r.h}, D={r.d}] (the rowwise payload), got {tuple(dst.shape)} strides {dst.stride()}")
        if dst.stride(0) % 16:
            raise ValueError(f"token strides must keep every row 16-byte aligned: dst {dst.stride(0)} elems (fp8)")
    if want_col:
        if r.transposed_second:
            n_rows = r.h * r.d
            if dst_t.ndim != 2 or tuple(dst_t.shape) != (n_rows, t) or not dst_t.is_contiguous():
                raise ValueError(
                    f"this artifact is transposed_second=True: dst_T must be the contiguous [H*D={n_rows}, T={t}] float8_e4m3fn matrix (strides ({t}, 1)), "
                    f"got shape {tuple(dst_t.shape)} strides {dst_t.stride()}"
                )
            if t % SF_BLOCK:
                raise ValueError(
                    f"the transposed quantization writes whole {SF_BLOCK}-token blocks (32 contiguous e4m3 codes per column): "
                    f"T = batch*seq_len must be a multiple of {SF_BLOCK}, got T={t}"
                )
        else:
            if (
                dst_t.ndim != 3
                or int(dst_t.shape[0]) != t
                or int(dst_t.shape[1]) != r.h
                or int(dst_t.shape[2]) != r.d
                or dst_t.stride(2) != 1
                or dst_t.stride(1) != r.d
            ):
                raise ValueError(
                    f"dst_T must be the compact e4m3 [T={t}, H={r.h}, D={r.d}] (the SDPA columnwise payload is ROW-MAJOR; this artifact is transposed_second=False), "
                    f"got {tuple(dst_t.shape)} strides {dst_t.stride()}"
                )
            if dst_t.stride(0) % 16:
                raise ValueError(f"token strides must keep every row 16-byte aligned: dst_T {dst_t.stride(0)} elems (fp8)")
    if not src.is_cuda or src.data_ptr() % 16:
        raise ValueError("src must be a 16-byte-aligned CUDA tensor (its rows move as 16-byte vectors)")
    blobs = ([("sf", sf)] if want_row else []) + ([("sf_T", sf_t)] if want_col else [])
    for name, ten in blobs:
        if ten.dtype != torch.uint8:
            raise ValueError(f"{name} must be torch.uint8 (E8M0 bytes in F8_128x4 order), got {ten.dtype}")
        if not ten.is_contiguous() or not ten.is_cuda or ten.data_ptr() % 16:
            raise ValueError(f"{name} must be a contiguous, 16-byte-aligned CUDA tensor (the SF tiles leave SMEM as 16-byte bursts)")
    if gemm:
        # each half's blob is sized (and its atoms counted) only when that half is traced: the transposed blob's K is T, which the
        # canonical builder refuses at T % 32 != 0 -- a rowwise-only launch over a ragged T (a dgrad-only block) never asks for it
        if want_row:
            need = sf_blob_bytes(t, r.h * r.d, SF_BLOCK)
            if sf.numel() != need:
                raise ValueError(
                    f"sf must be the padded F8_128x4 blob of sf_blob_bytes(rows={t}, K={r.h * r.d}) = {need} bytes (the rowwise canonical half), got {sf.numel()}"
                )
        if want_col:
            need_t = sf_blob_bytes(r.h * r.d, t, SF_BLOCK)
            if sf_t.numel() != need_t:
                raise ValueError(
                    f"sf_T must be the padded F8_128x4 blob of sf_blob_bytes(rows={r.h * r.d}, K={t}) = {need_t} bytes (the transposed canonical half), got {sf_t.numel()}"
                )
    elif packed:
        need = sf_bytes_packed(batch, r.h, t, r.d)
        for name, ten in blobs:
            if ten.numel() != need:
                raise ValueError(
                    f"{name} must hold H*((T + 127*B)//128)*{sf_tile_bytes(r.d)} = {need} bytes for the PACKED layout at T={t} B={batch} H={r.h} D={r.d}, "
                    f"got {ten.numel()}"
                )
    else:
        need = sf_bytes(batch, r.h, seq_len, r.d)
        for name, ten in blobs:
            if ten.numel() != need:
                raise ValueError(
                    f"{name} must hold B*H*ceil(S/128)*{sf_tile_bytes(r.d)} = {need} bytes for batch={batch} H={r.h} S={seq_len} D={r.d}, got {ten.numel()}"
                )
    if not all(ten.device == src.device for _, ten in payloads + blobs):
        raise ValueError("src, dst, sf, dst_T and sf_T must live on one device")
    if gemm:
        launch_seq, launch_bh = t, r.h
        n_c_atoms = sf_padded_dims(t, r.h * r.d, SF_BLOCK)[1] // SF_ATOM_COLS if want_row else 0
        n_c_atoms_t = sf_padded_dims(r.h * r.d, t, SF_BLOCK)[1] // SF_ATOM_COLS if want_col else 0
    elif packed:
        launch_seq, launch_bh, n_c_atoms, n_c_atoms_t = 0, r.h, 0, 0
    else:
        launch_seq, launch_bh, n_c_atoms, n_c_atoms_t = seq_len, batch * r.h, 0, 0
    return launch_seq, launch_bh, n_c_atoms, n_c_atoms_t


def run_quantize_mxfp8_dual(
    r: QuantizeMxfp8DualRecipe,
    src: torch.Tensor,
    dst: torch.Tensor,
    sf: torch.Tensor,
    dst_t: torch.Tensor,
    sf_t: torch.Tensor,
    *,
    batch: Optional[int] = None,
    seq_len: Optional[int] = None,
    stream,
    seq_lens: Optional[torch.Tensor] = None,
    num_sequences: int = 0,
    cu_seqlens: bool = False,
) -> None:
    """Launch.  ``src`` / ``dst`` / ``sf`` exactly as :func:`run_quantize_mxfp8` takes them for the rowwise arm of the recipe's layout;
    ``dst_T`` / ``sf_T`` the columnwise half's: under ``"sdpa"`` a second compact e4m3 ``[T, H, D]`` + the D-plane-major blob (the same
    byte count as the rowwise one), under ``"gemm"`` the contiguous ``[H*D, T]`` matrix (``T % 32 == 0``) + ``sf_blob_bytes(H*D, T)``;
    under a PACKED recipe the appended ``seq_lens`` / ``num_sequences`` / ``cu_seqlens`` replace ``batch`` / ``seq_len`` and both blobs
    are ``sf_bytes_packed`` (the V blob plane-ADJACENT).  Cheap host checks only (:func:`check_dual_operands`); every one guards a wild
    write or a silent wrong answer."""
    if r.compiled is None:
        raise ValueError("recipe was not built by compile_quantize_mxfp8_dual")
    packed = bool(getattr(r, "packed", False))
    launch_seq, launch_bh, n_c_atoms, n_c_atoms_t = check_dual_operands(
        r, src, dst, sf, dst_t, sf_t, batch=batch, seq_len=seq_len, seq_lens=seq_lens, num_sequences=num_sequences, cu_seqlens=cu_seqlens
    )
    tiles = n_sf_tiles_packed_cap(int(src.shape[0]), num_sequences) if packed else n_sf_tiles(launch_seq)
    args = [
        src,
        dst,
        sf.view(-1),
        dst_t,
        sf_t.view(-1),
        cutlass.Int32(launch_seq),
        cutlass.Int32(tiles),
        cutlass.Int32(launch_bh * tiles),
        cutlass.Int32(launch_bh),
        cutlass.Int32(n_c_atoms),
        cutlass.Int32(n_c_atoms_t),
        cuda.CUstream(int(stream)),
    ]
    if packed:
        args += [seq_lens, cutlass.Int32(num_sequences), cutlass.Int32(1 if cu_seqlens else 0)]
    r.compiled(*args)


frost_quantize_mxfp8.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_quantize_mxfp8_dual.set_name_prefix("cudnn", remove_cutlass_symbol=True)
