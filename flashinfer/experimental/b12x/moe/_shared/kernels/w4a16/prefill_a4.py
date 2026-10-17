"""NVFP4-activation (A4) prefill MoE GEMMs over the W4A16 packed weights.

Large prefill calls of a W4A16 MoE can trade activation precision for speed
without a second weight copy: these kernels contract NVFP4-quantized
activations with the W4A16 packed FP4 weights through the SM120 block-scaled
QMMA (``kind::mxf4nvf4`` ``scale_vec::4X`` m16n8k64), so decode keeps the exact
W4A16 path while prefill runs at A4 speed.

Operand plumbing
----------------
W4A16 packs each (K16, N64) tile as 32 lane words per column group ``jj``:
the word of BF16 lane ``(tc_col, r)`` holds nibbles ``[c1 k0, c1 k8, c2 k0,
c2 k8, c1 k1, c1 k9, c2 k1, c2 k9]`` of rows ``2r + {0, 1, 8, 9}``, columns
``c1 = 16 jj + tc_col`` and ``c2 = c1 + 8``. Activations are stored with the
K order inside every 16-group permuted by

    pi(8 h + i) = 4 h + [0, 8, 1, 9, 2, 10, 3, 11][i]

so that QMMA lane ``(q, c)`` (``h = c & 1``) finds its B register for column
``c1`` (``c2``) as one ``PRMT`` of the words of BF16 lanes ``(q, 2h)`` and
``(q, 2h + 1)``: bytes ``{0, 2}`` (``{1, 3}``) of each. The permutation stays
inside each 16-group, so block scales are unchanged.

The W4A16 scales are lifted E4M3 bytes (``s * f * 2**7`` as FP16 bits 14..7, so
``lifted = e4m3(s * f) + 120`` for every kept scale, 0 for flushed ones); the
kernels restore ``e4m3(s * f)`` per byte and fold ``1 / f`` back through the
packed global scale (``g * 2**119 / f``) into the per-expert alpha.


Activation layout (``x_q`` for FC1, the intermediate for FC2): per row and
K64 slice, eight u32 words of eight permuted positions each, stored in the
order ``[w0, w4, w1, w5, w2, w6, w3, w7]`` so QMMA lane ``c`` loads its two
registers of a row with one 8-byte access; one u32 of four E4M3 block scales
per row and K64 slice (byte ``g`` = 16-group ``g``).

FC1 runs the gate and up halves of a 128-column intermediate tile in one K
sweep, applies ``silu(alpha g) * (alpha u)`` and requantizes the BF16-rounded
intermediate to NVFP4 with the per-expert intermediate global scale. FC2
stores unweighted per-route BF16 rows for the FP32 top-k sum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, Uint32, Uint64, dsl_user_op

from b12x._lib.intrinsics.packed import bfloat2_to_float2_scaled, pack_f32x2_to_bfloat2
from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    shared_ptr_to_u32,
    st_global_u32,
    st_global_v4_u32,
    st_shared_v4_u32,
)
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.conversion import fp8_e4m3_to_f32
from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.intrinsics.mma import nvfp4_mma_m16n8k64_f32_e2m1
from b12x._lib.intrinsics.quantization import (
    quantize_block_fp4,
    quantize_block_fp4_fast,
)

# Position m of a 16-group holds physical K offset PI[m].
PI = (0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)
# Stored word order of a row's K64 slice.
WORD_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
# Global scale of the W4A16 packed weights: g * 2**119 / f (BF16 activations).
PACKED_GLOBAL_LIFT = 2.0**-119

BLOCK_M = 128
TILE_K = 64
NUM_WARPS = 8
A_STAGE_BYTES = BLOCK_M * 32
SFA_STAGE_BYTES = BLOCK_M * 4
# One activation plane per stage: payload rows then their scale words.
PLANE_STAGE_BYTES = A_STAGE_BYTES + SFA_STAGE_BYTES
B_STAGE_BYTES = (TILE_K // 16) * 4 * 512
S_STAGE_BYTES = (TILE_K // 16) * 4 * 64
ROUTE_BYTES = BLOCK_M * 4
# E2M1 magnitudes in half units, bytes [0, 1, 2, 3] and [4, 6, 8, 12].
_E2M1_HALF_LO = 0x03020100
_E2M1_HALF_HI = 0x0C080604
# The epilogue stages a [128, 256] BF16 tile with padded rows.
EPI_ROW_BYTES = 256 * 2 + 16


@dsl_user_op
def prmt_b32(a: Uint32, b: Uint32, sel: Uint32, *, loc=None, ip=None) -> Uint32:
    """``prmt.b32`` with an immediate selector."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Uint32(sel).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,n",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def prmt_b32_reg(a: Uint32, b: Uint32, sel: Uint32, *, loc=None, ip=None) -> Uint32:
    """``prmt.b32`` with a register selector (byte LUT lookups)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Uint32(sel).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _lifted_to_e4m3(word: Uint32) -> Uint32:
    """Bytewise ``lifted - 120`` for kept (bit 7 set) scales, 0 for flushed ones."""
    return (word & Uint32(0x7F7F7F7F)) + ((word >> Uint32(4)) & Uint32(0x08080808))


@cute.jit
def _quantize_terms(
    vals, gs: Float32, terms: cutlass.Constexpr[int], fast: cutlass.Constexpr[bool]
):
    """NVFP4-quantize 16 values (``quantize_block_fp4`` semantics) in the
    permuted position order; with ``terms == 2`` also quantize the residual
    against the first term's dequantized values with the same global scale.

    Returns ((lo_word, hi_word, scale_byte) per term).
    """
    perm = cute.make_rmem_tensor((16,), Float32)
    block_max = Float32(0.0)
    for m in cutlass.range_constexpr(16):
        perm[m] = vals[PI[m]]
        block_max = fmax_f32(block_max, fabs_f32(vals[m]))
    if cutlass.const_expr(fast):
        packed, scale_byte = quantize_block_fp4_fast(perm, block_max, gs)
    else:
        packed, scale_byte = quantize_block_fp4(perm, block_max, gs)
    out = [
        (
            Uint32(packed & Uint64(0xFFFFFFFF)),
            Uint32(packed >> Uint64(32)),
            Uint32(scale_byte) & Uint32(0xFF),
        )
    ]
    if cutlass.const_expr(terms == 2):
        # Residual of the dequantized first term: e2m1(code) * scale / gs.
        half_scale = Float32(0.0)
        if gs != Float32(0.0):
            half_scale = (
                fp8_e4m3_to_f32(Uint32(scale_byte) & Uint32(0xFF)) * Float32(0.5) / gs
            )
        res = cute.make_rmem_tensor((16,), Float32)
        res_max = Float32(0.0)
        for m in cutlass.range_constexpr(16):
            code = Uint32(packed >> Uint64(4 * m)) & Uint32(0xF)
            half = prmt_b32_reg(
                Uint32(_E2M1_HALF_LO), Uint32(_E2M1_HALF_HI), code & Uint32(7)
            ) & Uint32(0xFF)
            deq = Float32(half) * half_scale
            if (code & Uint32(8)) != Uint32(0):
                deq = -deq
            res[m] = perm[m] - deq
            res_max = fmax_f32(res_max, fabs_f32(res[m]))
        if cutlass.const_expr(fast):
            packed2, scale_byte2 = quantize_block_fp4_fast(res, res_max, gs)
        else:
            packed2, scale_byte2 = quantize_block_fp4(res, res_max, gs)
        out.append(
            (
                Uint32(packed2 & Uint64(0xFFFFFFFF)),
                Uint32(packed2 >> Uint64(32)),
                Uint32(scale_byte2) & Uint32(0xFF),
            )
        )
    return out


@cute.jit
def _store_slice(q_ptr: Int64, sf_ptr: Int64, words, scale_word: Uint32):
    """One row K64 slice: eight payload words in WORD_ORDER and its scale word."""
    st_global_v4_u32(
        q_ptr,
        words[WORD_ORDER[0]],
        words[WORD_ORDER[1]],
        words[WORD_ORDER[2]],
        words[WORD_ORDER[3]],
    )
    st_global_v4_u32(
        q_ptr + Int64(16),
        words[WORD_ORDER[4]],
        words[WORD_ORDER[5]],
        words[WORD_ORDER[6]],
        words[WORD_ORDER[7]],
    )
    st_global_u32(sf_ptr, scale_word)


class A4PackedQuantize:
    """BF16 ``[rows, K]`` -> permuted NVFP4 ``x_q`` + per-K64 scale words.

    One thread per (row, K64 slice); ``global_scale[0]`` is the shared input
    global scale (``quantize_block_fp4`` semantics). ``terms=2`` also writes the
    residual term (``x_q2``/``x_sf2``, same layout): ``x ~ q1 + q2``.
    """

    THREADS = 256

    def __init__(self, *, size_k: int, fast_math: bool = False, terms: int = 1):
        if size_k % 64:
            raise ValueError("A4 packed quantization needs K % 64 == 0")
        if terms not in (1, 2):
            raise ValueError("terms must be 1 or 2")
        self.size_k = int(size_k)
        self.slices = self.size_k // 64
        self.fast_math = bool(fast_math)
        self.terms = int(terms)

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (self.size_k, self.fast_math, self.terms)

    @cute.jit
    def __call__(
        self,
        x_bf16: cute.Tensor,
        global_scale: cute.Tensor,
        x_q: cute.Tensor,
        x_sf: cute.Tensor,
        x_q2: cute.Tensor,
        x_sf2: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        items = rows * Int32(self.slices)
        self.kernel(x_bf16, global_scale, x_q, x_sf, x_q2, x_sf2, rows).launch(
            grid=((items + Int32(self.THREADS - 1)) // Int32(self.THREADS), 1, 1),
            block=[self.THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_bf16: cute.Tensor,
        global_scale: cute.Tensor,
        x_q: cute.Tensor,
        x_sf: cute.Tensor,
        x_q2: cute.Tensor,
        x_sf2: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        item = Int32(bidx) * Int32(self.THREADS) + Int32(tidx)
        if item < rows * Int32(self.slices):
            row = item // Int32(self.slices)
            sl = item - row * Int32(self.slices)
            gs = global_scale[0].to(Float32)
            src = get_ptr_as_int64(
                x_bf16, Int64(row) * Int64(self.size_k) + Int64(sl * Int32(64))
            )
            words = cute.make_rmem_tensor((8,), Uint32)
            words2 = cute.make_rmem_tensor((8,), Uint32)
            scale_word = Uint32(0)
            scale_word2 = Uint32(0)
            for g in cutlass.range_constexpr(4):
                v0 = ld_global_nc_v4_u32(src + Int64(32 * g))
                v1 = ld_global_nc_v4_u32(src + Int64(32 * g + 16))
                raw = (v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3])
                vals = cute.make_rmem_tensor((16,), Float32)
                for i in cutlass.range_constexpr(8):
                    lo, hi = bfloat2_to_float2_scaled(raw[i], Float32(1.0))
                    vals[2 * i] = lo
                    vals[2 * i + 1] = hi
                out = _quantize_terms(vals, gs, self.terms, self.fast_math)
                words[2 * g] = out[0][0]
                words[2 * g + 1] = out[0][1]
                scale_word = scale_word | (out[0][2] << Uint32(8 * g))
                if cutlass.const_expr(self.terms == 2):
                    words2[2 * g] = out[1][0]
                    words2[2 * g + 1] = out[1][1]
                    scale_word2 = scale_word2 | (out[1][2] << Uint32(8 * g))
            q_off = Int64(row) * Int64(self.size_k // 8) + Int64(sl * Int32(8))
            sf_off = Int64(row) * Int64(self.slices) + Int64(sl)
            _store_slice(
                get_ptr_as_int64(x_q, q_off),
                get_ptr_as_int64(x_sf, sf_off),
                words,
                scale_word,
            )
            if cutlass.const_expr(self.terms == 2):
                _store_slice(
                    get_ptr_as_int64(x_q2, q_off),
                    get_ptr_as_int64(x_sf2, sf_off),
                    words2,
                    scale_word2,
                )


class A4PackedPrefillGemm:
    """One FC phase of the A4 prefill MoE over W4A16 packed weights.

    Tiles are (128-route block, N tile of four packed N64 chunks); warp ``w``
    owns rows ``32 * (w // 4)`` .. ``+31`` of chunk slot ``w % 4``.

    ``phase='fc1'``: chunk slots 0/1 are gate chunks ``2j, 2j+1`` and 2/3 the up
    chunks ``I/64 + 2j, +1``; A rows are gathered by token from ``x_q``; the
    NVFP4 intermediate is written by route id. ``phase='fc2'``: N tile = 256
    hidden columns, A rows gathered by route id, BF16 output ``[routes, H]``.

    ``terms=2`` contracts two activation planes (``q1 + q2``: NVFP4 value and
    NVFP4 residual) into the same accumulators, and FC1 writes the intermediate
    the same way.

    """

    def __init__(
        self,
        *,
        phase: str,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        stages: int = 4,
        fast_math: bool = True,
        fast_quant: bool = False,
        terms: int = 1,
        warps: int = NUM_WARPS,
        swiglu_limit: float | None = None,
        swiglu_clamp_after_silu: bool = False,
    ):
        if phase not in ("fc1", "fc2"):
            raise ValueError(f"unknown A4 prefill phase {phase!r}")
        if hidden_size % 256 or intermediate_size % 64:
            raise ValueError("A4 prefill needs H % 256 == 0 and I % 64 == 0")
        if terms not in (1, 2):
            raise ValueError("terms must be 1 or 2")
        if warps not in (8, 16):
            raise ValueError("warps must be 8 or 16")
        self.phase = phase
        self.fc1 = phase == "fc1"
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.fc1_half_slab = self.fc1 and self.intermediate_size % 128 != 0
        self.top_k = int(top_k)
        self.stages = int(stages)
        self.fast_math = bool(fast_math)
        self.fast_quant = bool(fast_quant)
        self.terms = int(terms)
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_clamp_after_silu = bool(swiglu_clamp_after_silu)
        self.swiglu_limit = 0.0 if swiglu_limit is None else float(swiglu_limit)
        # Four warps per row group, one per N64 chunk slot; 16 warps own 32 rows
        # each (64 accumulators), 8 warps own 64 rows (128 accumulators).
        self.warps = int(warps)
        self.threads = self.warps * 32
        self.m_warps = self.warps // 4
        self.rows_per_warp = BLOCK_M // self.m_warps
        self.mb_per_warp = self.rows_per_warp // 16
        self.acc_size = self.mb_per_warp * 8 * 4
        self.size_k = self.hidden_size if self.fc1 else self.intermediate_size
        self.size_n = 2 * self.intermediate_size if self.fc1 else self.hidden_size
        self.n_tiles = (
            (self.intermediate_size + 127) // 128
            if self.fc1
            else self.hidden_size // 256
        )
        self.k_tiles = self.size_k // TILE_K
        # Stage: activation planes, then the B chunks and their scales.
        self.b_off = self.terms * PLANE_STAGE_BYTES
        self.s_off = self.b_off + B_STAGE_BYTES
        self.stage_bytes = self.s_off + S_STAGE_BYTES
        self.pipeline_bytes = self.stages * self.stage_bytes
        self.route_off = max(self.pipeline_bytes, BLOCK_M * EPI_ROW_BYTES)
        self.shared_bytes = self.route_off + ROUTE_BYTES

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.phase,
            self.hidden_size,
            self.intermediate_size,
            self.top_k,
            self.stages,
            self.fast_math,
            self.fast_quant,
            self.terms,
            self.has_swiglu_limit,
            self.swiglu_limit,
            self.swiglu_clamp_after_silu,
            BLOCK_M,
            TILE_K,
            self.warps,
        )

    @cute.jit
    def __call__(
        self,
        a_q: cute.Tensor,
        a_sf: cute.Tensor,
        a_q2: cute.Tensor,
        a_sf2: cute.Tensor,
        weights_i32: cute.Tensor,
        scales_u8: cute.Tensor,
        w_global: cute.Tensor,
        a_gscale: cute.Tensor,
        q_gscale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        out_q: cute.Tensor,
        out_sf: cute.Tensor,
        out_q2: cute.Tensor,
        out_sf2: cute.Tensor,
        out_bf16: cute.Tensor,
        live_routes: cutlass.Int32,
        gs_stride: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            a_q,
            a_sf,
            a_q2,
            a_sf2,
            weights_i32,
            scales_u8,
            w_global,
            a_gscale,
            q_gscale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            out_q,
            out_sf,
            out_q2,
            out_sf2,
            out_bf16,
            live_routes,
            gs_stride,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=[self.threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _chunk_n64(self, n_tile: Int32, chunk: Int32) -> Int32:
        """Packed N64 chunk index of tile ``n_tile`` slot ``chunk`` (0..3)."""
        result = n_tile * Int32(4) + chunk
        if cutlass.const_expr(self.fc1):
            result = n_tile * Int32(2) + (chunk & Int32(1))
            if chunk >= Int32(2):
                result += Int32(self.intermediate_size // 64)
        return result

    # -- staging -------------------------------------------------------------------

    @cute.jit
    def _copy_chunk(self, dst, src, n_tile, chunk):
        if cutlass.const_expr(self.fc1_half_slab):
            if n_tile * Int32(2) + (chunk & Int32(1)) < Int32(
                self.intermediate_size // 64
            ):
                cp_async4_shared_global(dst, src)
            else:
                st_shared_v4_u32(dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
        else:
            cp_async4_shared_global(dst, src)

    @cute.jit
    def _issue(
        self,
        tid: Int32,
        stage_base: Int32,
        k_tile: Int32,
        a_desc,
        sf_desc,
        b_desc,
        s_src: Int64,
        s_dst: Int32,
        scales_u8,
        expert: Int32,
        n_tile: Int32,
    ):
        """Stage one K64 slice of live activations, packed weights, and scales."""
        for src, dst, live in a_desc:
            if live != Int32(0):
                cp_async4_shared_global(
                    stage_base + dst, src + Int64(k_tile) * Int64(32)
                )
        for src, dst, live in sf_desc:
            if live != Int32(0):
                cp_async_u32_shared_global(
                    stage_base + dst, src + Int64(k_tile) * Int64(4)
                )
        for src, dst, chunk in b_desc:
            self._copy_chunk(
                stage_base + dst,
                src + Int64(k_tile) * Int64((TILE_K // 16) * (self.size_n // 64) * 512),
                n_tile,
                chunk,
            )
        if tid < Int32((TILE_K // 16) * 4 * 4):
            self._copy_chunk(
                stage_base + s_dst,
                s_src + Int64(k_tile) * Int64((TILE_K // 16) * self.size_n),
                n_tile,
                (tid >> Int32(2)) & Int32(3),
            )

    # -- mainloop ------------------------------------------------------------------

    @cute.jit
    def _mainloop(
        self,
        a_q,
        a_sf,
        a_q2,
        a_sf2,
        weights_i32,
        scales_u8,
        smem_base: Int32,
        tid: Int32,
        lane: Int32,
        m_warp: Int32,
        slot: Int32,
        expert: Int32,
        n_tile: Int32,
        live_blocks: Int32,
        acc,
    ):
        route_base = smem_base + Int32(self.shared_bytes - ROUTE_BYTES)
        k16_rows = self.size_k // 16
        n64_chunks = self.size_n // 64
        q = lane >> Int32(2)
        c = lane & Int32(3)
        h = c & Int32(1)
        live_rows = live_blocks * Int32(16)
        # A copies: (plane, row, 16-byte half) vectors and per-plane scale words,
        # spread over the CTA's threads.
        a_desc = []
        for i in cutlass.range_constexpr(
            (self.terms * 256 + self.threads - 1) // self.threads
        ):
            idx = tid + Int32(i * self.threads)
            plane = idx >> Int32(8)
            row = (idx & Int32(255)) >> Int32(1)
            vec = idx & Int32(1)
            live = Int32(0)
            if idx < Int32(self.terms * 256):
                if row < live_rows:
                    live = Int32(1)
            src_row = Int32(ld_shared_u32(route_base + (row << Int32(2))))
            elem = Int64(src_row) * Int64(self.size_k // 8) + Int64(vec * Int32(4))
            src = get_ptr_as_int64(a_q, elem)
            if cutlass.const_expr(self.terms == 2):
                if plane != Int32(0):
                    src = get_ptr_as_int64(a_q2, elem)
            dst = plane * Int32(PLANE_STAGE_BYTES) + row * Int32(32) + (vec << Int32(4))
            a_desc.append((src, dst, live))
        sf_desc = []
        for i in cutlass.range_constexpr(
            (self.terms * 128 + self.threads - 1) // self.threads
        ):
            # Offset so that one plane's scale words use the threads A leaves idle.
            idx = (tid + Int32(i * self.threads + 256)) & Int32(self.threads - 1)
            if cutlass.const_expr(self.terms * 128 > self.threads):
                idx = tid + Int32(i * self.threads)
            plane = idx >> Int32(7)
            row = idx & Int32(127)
            live = Int32(0)
            if idx < Int32(self.terms * 128):
                if row < live_rows:
                    live = Int32(1)
            src_row = Int32(ld_shared_u32(route_base + (row << Int32(2))))
            elem = Int64(src_row) * Int64(self.size_k // 64)
            src = get_ptr_as_int64(a_sf, elem)
            if cutlass.const_expr(self.terms == 2):
                if plane != Int32(0):
                    src = get_ptr_as_int64(a_sf2, elem)
            dst = (
                plane * Int32(PLANE_STAGE_BYTES)
                + Int32(A_STAGE_BYTES)
                + (row << Int32(2))
            )
            sf_desc.append((src, dst, live))
        # B copies: (K16 row, chunk slot, lane vector); the vector lands at slot
        # (v ^ (k16 & 1)) so the fragment loads below are conflict free.
        b_desc = []
        for i in cutlass.range_constexpr(512 // self.threads):
            idx = tid + Int32(i * self.threads)
            b_k16 = idx >> Int32(7)
            b_chunk = (idx >> Int32(5)) & Int32(3)
            b_vec = idx & Int32(31)
            src = get_ptr_as_int64(
                weights_i32,
                (
                    (Int64(expert) * Int64(k16_rows) + Int64(b_k16)) * Int64(n64_chunks)
                    + Int64(self._chunk_n64(n_tile, b_chunk))
                )
                * Int64(128)
                + Int64(b_vec * Int32(4)),
            )
            dst = (
                Int32(self.b_off)
                + ((b_k16 * Int32(4) + b_chunk) << Int32(9))
                + ((b_vec ^ (b_k16 & Int32(1))) << Int32(4))
            )
            b_desc.append((src, dst, b_chunk))
        s_k16 = (tid >> Int32(4)) & Int32(3)
        s_chunk = (tid >> Int32(2)) & Int32(3)
        s_part = tid & Int32(3)
        s_src = get_ptr_as_int64(
            scales_u8,
            (Int64(expert) * Int64(k16_rows) + Int64(s_k16)) * Int64(self.size_n)
            + Int64(self._chunk_n64(n_tile, s_chunk) * Int32(64))
            + Int64(s_part * Int32(16)),
        )
        s_dst = (
            Int32(self.s_off)
            + ((s_k16 * Int32(4) + s_chunk) << Int32(6))
            + (s_part << Int32(4))
        )

        # Fragment offsets inside a stage. B: lane (q, c) reads the vectors of BF16
        # lanes 4q + 2h + e of K16 rows 2j + c/2.
        b_frag = []
        for j in cutlass.range_constexpr(2):
            g = Int32(2 * j) + (c >> Int32(1))
            row_base = Int32(self.b_off) + ((g * Int32(4) + slot) << Int32(9))
            pair = []
            for e in cutlass.range_constexpr(2):
                vec = (q << Int32(2)) + (h << Int32(1)) + Int32(e)
                pair.append(row_base + ((vec ^ (g & Int32(1))) << Int32(4)))
            b_frag.append(pair)
        s_frag = Int32(self.s_off) + (slot << Int32(6)) + (q << Int32(3))
        row0 = m_warp * Int32(self.rows_per_warp) + q
        a_frag = row0 * Int32(32) + (c << Int32(3))
        sfa_frag = Int32(A_STAGE_BYTES) + ((row0 + (h << Int32(3))) << Int32(2))
        # Live 16-row blocks among this warp's (warp-uniform).
        warp_live = live_blocks - m_warp * Int32(self.mb_per_warp)

        for p in cutlass.range_constexpr(self.stages - 1):
            if Int32(p) < Int32(self.k_tiles):
                self._issue(
                    tid,
                    smem_base + Int32(p * self.stage_bytes),
                    Int32(p),
                    a_desc,
                    sf_desc,
                    b_desc,
                    s_src,
                    s_dst,
                    scales_u8,
                    expert,
                    n_tile,
                )
            cute.arch.cp_async_commit_group()
        rd_base = smem_base
        wr_base = smem_base + Int32((self.stages - 1) * self.stage_bytes)
        stage_end = smem_base + Int32(self.stages * self.stage_bytes)
        k_tile = Int32(0)
        while k_tile < Int32(self.k_tiles):
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
            nxt = k_tile + Int32(self.stages - 1)
            if nxt < Int32(self.k_tiles):
                self._issue(
                    tid,
                    wr_base,
                    nxt,
                    a_desc,
                    sf_desc,
                    b_desc,
                    s_src,
                    s_dst,
                    scales_u8,
                    expert,
                    n_tile,
                )
            cute.arch.cp_async_commit_group()
            if warp_live > Int32(0):
                self._compute_stage(
                    rd_base, b_frag, s_frag, a_frag, sfa_frag, warp_live, acc
                )
            rd_base += Int32(self.stage_bytes)
            if rd_base == stage_end:
                rd_base = smem_base
            wr_base += Int32(self.stage_bytes)
            if wr_base == stage_end:
                wr_base = smem_base
            k_tile += Int32(1)

    @cute.jit
    def _compute_stage(
        self,
        rd_base: Int32,
        b_frag,
        s_frag: Int32,
        a_frag: Int32,
        sfa_frag: Int32,
        warp_live: Int32,
        acc,
    ):
        # B registers [jj][t][j]: one PRMT of the e = 0 / 1 words per register.
        breg = cute.make_rmem_tensor((16,), Uint32)
        for j in cutlass.range_constexpr(2):
            w0 = ld_shared_v4_u32(rd_base + b_frag[j][0])
            w1 = ld_shared_v4_u32(rd_base + b_frag[j][1])
            for jj in cutlass.range_constexpr(4):
                breg[(jj * 2 + 0) * 2 + j] = prmt_b32(w0[jj], w1[jj], Uint32(0x6420))
                breg[(jj * 2 + 1) * 2 + j] = prmt_b32(w0[jj], w1[jj], Uint32(0x7531))
        # SFB [jj][t]: byte g of K16 row g, column 16 jj + 8 t + q. Row g's 8 bytes
        # at 8 q are [jj0 c1, jj1 c1, jj0 c2, jj1 c2 | jj2 c1, jj3 c1, jj2 c2, jj3 c2].
        sfb = cute.make_rmem_tensor((8,), Uint32)
        lo = cute.make_rmem_tensor((4,), Uint32)
        hi = cute.make_rmem_tensor((4,), Uint32)
        for g in cutlass.range_constexpr(4):
            w_lo, w_hi = ld_shared_v2_u32(rd_base + s_frag + Int32(g * 4 * 64))
            lo[g] = _lifted_to_e4m3(w_lo)
            hi[g] = _lifted_to_e4m3(w_hi)
        for half in cutlass.range_constexpr(2):
            src = lo if half == 0 else hi
            p01_lo = prmt_b32(src[0], src[1], Uint32(0x5140))
            p01_hi = prmt_b32(src[0], src[1], Uint32(0x7362))
            p23_lo = prmt_b32(src[2], src[3], Uint32(0x5140))
            p23_hi = prmt_b32(src[2], src[3], Uint32(0x7362))
            # Byte m = (jj & 1) + 2 t of the half's words.
            sfb[(2 * half + 0) * 2 + 0] = prmt_b32(p01_lo, p23_lo, Uint32(0x5410))
            sfb[(2 * half + 1) * 2 + 0] = prmt_b32(p01_lo, p23_lo, Uint32(0x7632))
            sfb[(2 * half + 0) * 2 + 1] = prmt_b32(p01_hi, p23_hi, Uint32(0x5410))
            sfb[(2 * half + 1) * 2 + 1] = prmt_b32(p01_hi, p23_hi, Uint32(0x7632))
        self._mma_block(rd_base, a_frag, sfa_frag, breg, sfb, acc, 0)
        for mb in cutlass.range_constexpr(1, self.mb_per_warp):
            if warp_live > Int32(mb):
                self._mma_block(rd_base, a_frag, sfa_frag, breg, sfb, acc, mb)

    @cute.jit
    def _mma_block(
        self,
        rd_base: Int32,
        a_frag: Int32,
        sfa_frag: Int32,
        breg,
        sfb,
        acc,
        mb: cutlass.Constexpr[int],
    ):
        for plane in cutlass.range_constexpr(self.terms):
            base = rd_base + a_frag + Int32(plane * PLANE_STAGE_BYTES + mb * 16 * 32)
            a0, a2 = ld_shared_v2_u32(base)
            a1, a3 = ld_shared_v2_u32(base + Int32(8 * 32))
            sfa = ld_shared_u32(
                rd_base + sfa_frag + Int32(plane * PLANE_STAGE_BYTES + mb * 16 * 4)
            )
            for jj in cutlass.range_constexpr(4):
                for t in cutlass.range_constexpr(2):
                    idx = ((mb * 4 + jj) * 2 + t) * 4
                    d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                        acc[idx],
                        acc[idx + 1],
                        acc[idx + 2],
                        acc[idx + 3],
                        a0,
                        a1,
                        a2,
                        a3,
                        breg[(jj * 2 + t) * 2 + 0],
                        breg[(jj * 2 + t) * 2 + 1],
                        sfa,
                        sfb[jj * 2 + t],
                    )
                    acc[idx] = d0
                    acc[idx + 1] = d1
                    acc[idx + 2] = d2
                    acc[idx + 3] = d3

    # -- epilogues -----------------------------------------------------------------

    @cute.jit
    def _stage_epilogue(
        self,
        acc,
        smem_base: Int32,
        lane: Int32,
        m_warp: Int32,
        slot: Int32,
        alpha: Float32,
    ):
        """[128, 256] BF16 tile of ``acc * alpha`` (columns in chunk-slot order)."""
        q = lane >> Int32(2)
        col0 = (slot << Int32(6)) + ((lane & Int32(3)) << Int32(1))
        for mb in cutlass.range_constexpr(self.mb_per_warp):
            for half_row in cutlass.range_constexpr(2):
                row = (
                    m_warp * Int32(self.rows_per_warp)
                    + Int32(16 * mb + 8 * half_row)
                    + q
                )
                for jj in cutlass.range_constexpr(4):
                    for t in cutlass.range_constexpr(2):
                        idx = ((mb * 4 + jj) * 2 + t) * 4 + 2 * half_row
                        col = col0 + Int32(16 * jj + 8 * t)
                        st_shared_u32(
                            smem_base + row * Int32(EPI_ROW_BYTES) + (col << Int32(1)),
                            pack_f32x2_to_bfloat2(
                                acc[idx] * alpha, acc[idx + 1] * alpha
                            ),
                        )
        cute.arch.sync_threads()

    @cute.jit
    def _store_fc2(
        self,
        acc,
        out_bf16,
        smem_base: Int32,
        tid: Int32,
        lane: Int32,
        m_warp: Int32,
        slot: Int32,
        n_tile: Int32,
        live_routes: Int32,
        alpha: Float32,
    ):
        self._stage_epilogue(acc, smem_base, lane, m_warp, slot, alpha)
        route_base = smem_base + Int32(self.shared_bytes - ROUTE_BYTES)
        # A warp stores one 512-byte output row as 32 x 16 bytes.
        for i in cutlass.range_constexpr(BLOCK_M * 32 // self.threads):
            idx = tid + Int32(i * self.threads)
            row = idx >> Int32(5)
            vec = idx & Int32(31)
            route = Int32(ld_shared_u32(route_base + (row << Int32(2))))
            if route < live_routes:
                v0, v1, v2, v3 = ld_shared_v4_u32(
                    smem_base + row * Int32(EPI_ROW_BYTES) + (vec << Int32(4))
                )
                dst = Int64(route) * Int64(self.hidden_size) + Int64(
                    n_tile * Int32(256) + (vec << Int32(3))
                )
                st_global_v4_u32(get_ptr_as_int64(out_bf16, dst), v0, v1, v2, v3)

    @cute.jit
    def _store_fc1(
        self,
        acc,
        out_q,
        out_sf,
        out_q2,
        out_sf2,
        smem_base: Int32,
        tid: Int32,
        lane: Int32,
        m_warp: Int32,
        slot: Int32,
        n_tile: Int32,
        live_routes: Int32,
        alpha: Float32,
        quant_gs: Float32,
    ):
        # Stage BF16 gate (cols 0..127) and up (128..255) after alpha.
        self._stage_epilogue(acc, smem_base, lane, m_warp, slot, alpha)
        route_base = smem_base + Int32(self.shared_bytes - ROUTE_BYTES)
        # Thread -> (row, K64 half of the 128 intermediate columns).
        if tid < Int32(2 * BLOCK_M):
            row = tid & Int32(BLOCK_M - 1)
            half = tid >> Int32(7)
            route = Int32(ld_shared_u32(route_base + (row << Int32(2))))
            live = route < live_routes
            if cutlass.const_expr(self.fc1_half_slab):
                live = live and n_tile * Int32(2) + half < Int32(
                    self.intermediate_size // 64
                )
            if live:
                row_base = smem_base + row * Int32(EPI_ROW_BYTES) + (half << Int32(7))
                words = cute.make_rmem_tensor((8,), Uint32)
                words2 = cute.make_rmem_tensor((8,), Uint32)
                scale_word = Uint32(0)
                scale_word2 = Uint32(0)
                for g in cutlass.range_constexpr(4):
                    vals = cute.make_rmem_tensor((16,), Float32)
                    for v in cutlass.range_constexpr(2):
                        gw = ld_shared_v4_u32(row_base + Int32(32 * g + 16 * v))
                        uw = ld_shared_v4_u32(row_base + Int32(256 + 32 * g + 16 * v))
                        for i in cutlass.range_constexpr(4):
                            g0, g1 = bfloat2_to_float2_scaled(gw[i], Float32(1.0))
                            u0, u1 = bfloat2_to_float2_scaled(uw[i], Float32(1.0))
                            vals[8 * v + 2 * i] = self._activate(g0, u0)
                            vals[8 * v + 2 * i + 1] = self._activate(g1, u1)
                    out = _quantize_terms(vals, quant_gs, self.terms, self.fast_quant)
                    words[2 * g] = out[0][0]
                    words[2 * g + 1] = out[0][1]
                    scale_word = scale_word | (out[0][2] << Uint32(8 * g))
                    if cutlass.const_expr(self.terms == 2):
                        words2[2 * g] = out[1][0]
                        words2[2 * g + 1] = out[1][1]
                        scale_word2 = scale_word2 | (out[1][2] << Uint32(8 * g))
                k64 = n_tile * Int32(2) + half
                q_off = Int64(route) * Int64(self.intermediate_size // 8) + Int64(
                    k64 * Int32(8)
                )
                sf_off = Int64(route) * Int64(self.intermediate_size // 64) + Int64(k64)
                _store_slice(
                    get_ptr_as_int64(out_q, q_off),
                    get_ptr_as_int64(out_sf, sf_off),
                    words,
                    scale_word,
                )
                if cutlass.const_expr(self.terms == 2):
                    _store_slice(
                        get_ptr_as_int64(out_q2, q_off),
                        get_ptr_as_int64(out_sf2, sf_off),
                        words2,
                        scale_word2,
                    )

    @cute.jit
    def _round(self, x: Float32) -> Float32:
        lo, _ = bfloat2_to_float2_scaled(pack_f32x2_to_bfloat2(x, x), Float32(1.0))
        return lo

    @cute.jit
    def _activate(self, gate: Float32, up: Float32) -> Float32:
        if cutlass.const_expr(self.has_swiglu_limit):
            limit = Float32(self.swiglu_limit)
            neg_limit = Float32(-self.swiglu_limit)
            if cutlass.const_expr(not self.swiglu_clamp_after_silu):
                if gate > limit:
                    gate = limit
            if up > limit:
                up = limit
            if up < neg_limit:
                up = neg_limit
        sigmoid = cute.arch.rcp_approx(
            Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math)
        )
        activated = gate * sigmoid
        if cutlass.const_expr(self.has_swiglu_limit and self.swiglu_clamp_after_silu):
            if activated > Float32(self.swiglu_limit):
                activated = Float32(self.swiglu_limit)
        return self._round(self._round(activated) * self._round(up))

    # -- kernel --------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        a_q: cute.Tensor,
        a_sf: cute.Tensor,
        a_q2: cute.Tensor,
        a_sf2: cute.Tensor,
        weights_i32: cute.Tensor,
        scales_u8: cute.Tensor,
        w_global: cute.Tensor,
        a_gscale: cute.Tensor,
        q_gscale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        out_q: cute.Tensor,
        out_sf: cute.Tensor,
        out_q2: cute.Tensor,
        out_sf2: cute.Tensor,
        out_bf16: cute.Tensor,
        live_routes: cutlass.Int32,
        gs_stride: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        m_warp = warp >> Int32(2)
        slot = warp & Int32(3)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_bytes // 4], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        route_base = smem_base + Int32(self.shared_bytes - ROUTE_BYTES)

        blocks = packed_route_count[0].to(Int32) // Int32(BLOCK_M)
        tiles = blocks * Int32(self.n_tiles)
        tile = Int32(bidx)
        while tile < tiles:
            block = tile // Int32(self.n_tiles)
            n_tile = tile - block * Int32(self.n_tiles)
            expert = block_expert_ids[block].to(Int32)
            if expert >= Int32(0):
                # Gather rows (token for FC1, route for FC2); live routes form a
                # prefix of the block.
                if tid < Int32(BLOCK_M):
                    route = packed_route_indices[block * Int32(BLOCK_M) + tid].to(Int32)
                    src = Int32(0)
                    if route < live_routes:
                        src = route
                        if cutlass.const_expr(self.fc1):
                            src = route // Int32(self.top_k)
                    st_shared_u32(route_base + (tid << Int32(2)), Uint32(src))
                cute.arch.sync_threads()
                live_blocks = Int32(0)
                for probe in cutlass.range_constexpr(BLOCK_M // 16):
                    r = packed_route_indices[
                        block * Int32(BLOCK_M) + Int32(16 * probe)
                    ].to(Int32)
                    if r < live_routes:
                        live_blocks = Int32(probe + 1)
                # alpha = g / f / a_gscale: the packed global scale carries 2**119 / f.
                # FC1 reads the shared input scale; FC2 and the FC1 requant read the
                # intermediate scale per expert (gs_stride 1) or shared (0).
                a_gs_idx = Int32(0)
                if cutlass.const_expr(not self.fc1):
                    a_gs_idx = expert * gs_stride
                alpha = (
                    w_global[expert].to(Float32)
                    * Float32(PACKED_GLOBAL_LIFT)
                    / a_gscale[a_gs_idx].to(Float32)
                )

                acc = cute.make_rmem_tensor((self.acc_size,), Float32)
                acc.fill(0.0)
                self._mainloop(
                    a_q,
                    a_sf,
                    a_q2,
                    a_sf2,
                    weights_i32,
                    scales_u8,
                    smem_base,
                    tid,
                    lane,
                    m_warp,
                    slot,
                    expert,
                    n_tile,
                    live_blocks,
                    acc,
                )
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                # The epilogue reads route ids, not gather rows.
                if tid < Int32(BLOCK_M):
                    route = packed_route_indices[block * Int32(BLOCK_M) + tid].to(Int32)
                    st_shared_u32(route_base + (tid << Int32(2)), Uint32(route))
                cute.arch.sync_threads()
                if cutlass.const_expr(self.fc1):
                    self._store_fc1(
                        acc,
                        out_q,
                        out_sf,
                        out_q2,
                        out_sf2,
                        smem_base,
                        tid,
                        lane,
                        m_warp,
                        slot,
                        n_tile,
                        live_routes,
                        alpha,
                        q_gscale[expert * gs_stride].to(Float32),
                    )
                else:
                    self._store_fc2(
                        acc,
                        out_bf16,
                        smem_base,
                        tid,
                        lane,
                        m_warp,
                        slot,
                        n_tile,
                        live_routes,
                        alpha,
                    )
                cute.arch.sync_threads()
            tile += Int32(gdimx)


# -- host: plan-time launch set and the per-call pipeline ------------------------

A4_PREFILL_ROUTE_BLOCK = BLOCK_M
_FAKE_ELEMENTS = 1 << 30


def a4_prefill_enabled() -> bool:
    """Whether calibrated weights prepare explicit A4-prefill capability."""
    import os

    value = os.environ.get("B12X_W4A16_A4_PREFILL", "0")
    if value not in ("0", "1"):
        raise ValueError("B12X_W4A16_A4_PREFILL must be 0 or 1")
    return value == "1"


def a4_prefill_terms() -> int:
    """Activation planes of the A4 prefill path: 1 = NVFP4, 2 = NVFP4 value +
    NVFP4 residual;
    ``B12X_W4A16_A4_PREFILL_TERMS``."""
    import os

    value = int(os.environ.get("B12X_W4A16_A4_PREFILL_TERMS", "1") or 1)
    if value not in (1, 2):
        raise ValueError("B12X_W4A16_A4_PREFILL_TERMS must be 1 or 2")
    return value


def a4_prefill_warps() -> int:
    """GEMM CTA size of the A4 prefill path (``B12X_W4A16_A4_PREFILL_WARPS``): 8
    warps with 64-row warp tiles (the default) or 16 warps with 32-row tiles."""
    import os

    value = int(os.environ.get("B12X_W4A16_A4_PREFILL_WARPS", "8") or 8)
    if value not in (8, 16):
        raise ValueError("B12X_W4A16_A4_PREFILL_WARPS must be 8 or 16")
    return value


def a4_prefill_supported(
    *,
    prepared_layout: str,
    scale_format: str,
    activation: str,
    is_gated: bool,
    swiglu_limit: float | None,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    return (
        prepared_layout == "packed"
        and scale_format == "e4m3_k16"
        and activation == "silu"
        and bool(is_gated)
        and dtype == torch.bfloat16
        and (swiglu_limit is None or (math.isfinite(swiglu_limit) and swiglu_limit > 0))
        and hidden_size % 256 == 0
        and intermediate_size % 64 == 0
    )


@dataclass(frozen=True)
class W4A16A4PrefillLaunches:
    """Compiled A4 prefill pipeline for one W4A16 capacity."""

    tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int
    grid: int
    terms: int
    quant: object
    fc1: object
    fc2: object
    topk_sum: object
    route_pack: object

    def carriers(self) -> tuple[object, ...]:
        return (
            self.quant,
            self.fc1,
            self.fc2,
            self.topk_sum,
            *self.route_pack.carriers(),
        )

    def scratch_bytes(self, tokens: int) -> int:
        """Bytes of ``intermediate_cache2`` the pipeline carves for ``tokens``."""
        return _carve_layout(self, int(tokens))[-1]

    def fits_buffers(self, tokens: int, cache13_bytes: int, cache2_bytes: int) -> bool:
        return cache13_bytes >= int(
            tokens
        ) * self.topk * self.hidden_size * 2 and cache2_bytes >= self.scratch_bytes(
            tokens
        )


def _fake(dtype, align=16):
    return cute.runtime.make_fake_compact_tensor(
        dtype, (_FAKE_ELEMENTS,), assumed_align=align
    )


def _compile_a4_kernels(
    *,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    fast_math: bool,
    terms: int,
    warps: int,
    swiglu_limit: float | None,
    swiglu_clamp_after_silu: bool = False,
):
    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from b12x._lib.utils.device import current_cuda_stream

    key = (
        int(hidden_size),
        int(intermediate_size),
        int(topk),
        bool(fast_math),
        5,
        int(terms),
        warps,
        swiglu_limit,
        swiglu_clamp_after_silu,
    )
    stream = current_cuda_stream()
    quant = b12x_compile(
        A4PackedQuantize(size_k=hidden_size, terms=terms),
        _fake(cutlass.BFloat16),
        _fake(cutlass.Float32, 4),
        _fake(cutlass.Uint32),
        _fake(cutlass.Uint32, 4),
        _fake(cutlass.Uint32),
        _fake(cutlass.Uint32, 4),
        Int32(1),
        stream,
        compile_spec=KernelCompileSpec.from_key("moe.w4a16.a4_prefill.quant", 3, key),
    )
    gemms = []
    for phase in ("fc1", "fc2"):
        gemms.append(
            b12x_compile(
                A4PackedPrefillGemm(
                    phase=phase,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    top_k=topk,
                    fast_math=fast_math,
                    terms=terms,
                    warps=warps,
                    swiglu_limit=swiglu_limit,
                    swiglu_clamp_after_silu=swiglu_clamp_after_silu,
                ),
                _fake(cutlass.Uint32),
                _fake(cutlass.Uint32, 4),
                _fake(cutlass.Uint32),
                _fake(cutlass.Uint32, 4),
                _fake(cutlass.Int32),
                _fake(cutlass.Uint8),
                _fake(cutlass.Float32, 4),
                _fake(cutlass.Float32, 4),
                _fake(cutlass.Float32, 4),
                _fake(cutlass.Int32, 4),
                _fake(cutlass.Int32, 4),
                _fake(cutlass.Int32, 4),
                _fake(cutlass.Uint32),
                _fake(cutlass.Uint32, 4),
                _fake(cutlass.Uint32),
                _fake(cutlass.Uint32, 4),
                _fake(cutlass.BFloat16),
                Int32(1),
                Int32(1),
                Int32(1),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    f"moe.w4a16.a4_prefill.{phase}", 5, key
                ),
            )
        )
    return quant, gemms[0], gemms[1]


def compile_w4a16_a4_prefill(
    *,
    tokens: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    sms: int,
    ordinal: int,
    fast_math: bool,
    terms: int = 1,
    warps: int = NUM_WARPS,
    swiglu_limit: float | None = None,
    swiglu_clamp_after_silu: bool = False,
) -> W4A16A4PrefillLaunches:
    from b12x.moe._shared.kernels.w4a16.kernel.sum_launch import compile_w4a16_topk_sum
    from b12x.moe._shared.kernels.w4a16.route_pack import (
        compile_w4a16_route_pack_launches,
    )

    quant, fc1, fc2 = _compile_a4_kernels(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topk=topk,
        fast_math=fast_math,
        terms=terms,
        warps=warps,
        swiglu_limit=swiglu_limit,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
    )
    topk_sum = compile_w4a16_topk_sum(
        m=tokens,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype="bf16",
        apply_topk_weights=True,
    )
    route_pack = compile_w4a16_route_pack_launches(
        tokens=tokens,
        topk=topk,
        block_size=A4_PREFILL_ROUTE_BLOCK,
        num_experts=num_experts,
        ordinal=ordinal,
    )
    return W4A16A4PrefillLaunches(
        tokens=int(tokens),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_experts=int(num_experts),
        topk=int(topk),
        grid=int(sms),
        terms=int(terms),
        quant=quant,
        fc1=fc1,
        fc2=fc2,
        topk_sum=topk_sum,
        route_pack=route_pack,
    )


def _carve_layout(launches: W4A16A4PrefillLaunches, tokens: int):
    """Byte offsets of the pipeline buffers inside ``intermediate_cache2``."""
    h, i, e = launches.hidden_size, launches.intermediate_size, launches.num_experts
    routes = tokens * launches.topk
    planes = launches.terms
    sizes = (
        planes * tokens * h // 2,  # x_q (per plane)
        planes * tokens * h // 16,  # x_sf
        planes * routes * i // 2,  # act_q
        planes * routes * i // 16,  # act_sf
        launches.route_pack.max_packed_routes * 4,  # packed_route_indices
        launches.route_pack.max_route_blocks * 4,  # block_expert_ids
        4,  # packed_route_count
        (e + 1) * 4,  # expert_offsets
        e * 4,  # expert_counts
    )
    offsets = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset = (offset + size + 255) // 256 * 256
    return tuple(zip(offsets, sizes)), offset


def _planes(region: torch.Tensor, terms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an activation region into its planes (plane 2 aliases plane 1 when
    there is a single term; the kernels never touch it then)."""
    if terms == 1:
        return region, region
    half = region.numel() // 2
    return region.narrow(0, 0, half), region.narrow(0, half, half)


def a4_prefill_fits(
    launches: W4A16A4PrefillLaunches,
    *,
    tokens: int,
    intermediate_cache13: torch.Tensor,
    intermediate_cache2: torch.Tensor,
) -> bool:
    return launches.fits_buffers(
        tokens,
        intermediate_cache13.numel() * intermediate_cache13.element_size(),
        intermediate_cache2.numel() * intermediate_cache2.element_size(),
    )


def run_w4a16_a4_prefill(
    a: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    intermediate_cache13: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    output: torch.Tensor,
    launches: W4A16A4PrefillLaunches,
) -> torch.Tensor:
    """Route pack (128) -> NVFP4 quant -> A4 FC1 -> A4 FC2 -> FP32-weighted top-k sum.

    Every buffer is a view of caller scratch: ``intermediate_cache2`` holds the
    NVFP4 activations and the route pack, ``intermediate_cache13`` the per-route
    BF16 FC2 rows.
    """
    from cutlass.cute.runtime import make_ptr

    from b12x._lib.utils.device import current_cuda_stream
    from b12x.moe._shared.kernels.w4a16.kernel.sum_launch import (
        _w4a16_topk_sum_launch_flat,
    )
    from b12x.moe._shared.kernels.w4a16.route_pack import pack_topk_routes_by_expert

    tokens = int(a.shape[0])
    routes = tokens * launches.topk
    regions, _ = _carve_layout(launches, tokens)
    raw = intermediate_cache2.view(-1).view(torch.uint8)
    views = [raw.narrow(0, off, size) for off, size in regions]
    x_q, x_sf, act_q, act_sf, pir, beid, prc, offsets, counts = (
        v.view(torch.int32) for v in views
    )
    # Plane 2 (residual) follows plane 1 inside each activation region.
    x_q, x_q2 = _planes(x_q, launches.terms)
    x_sf, x_sf2 = _planes(x_sf, launches.terms)
    act_q, act_q2 = _planes(act_q, launches.terms)
    act_sf, act_sf2 = _planes(act_sf, launches.terms)
    y = intermediate_cache13.view(-1).narrow(0, 0, routes * launches.hidden_size)
    pack_topk_routes_by_expert(
        topk_ids,
        A4_PREFILL_ROUTE_BLOCK,
        launches.num_experts,
        packed_route_indices=pir,
        block_expert_ids=beid,
        packed_route_count=prc,
        expert_offsets=offsets,
        expert_counts=counts,
        launches=launches.route_pack,
    )

    def ptr(dtype, tensor, align=16):
        return make_ptr(
            dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
        )

    stream = current_cuda_stream()
    a1 = a1_gscale.view(-1)
    a2 = a2_gscale.view(-1)
    gs_stride = Int32(0 if a2.numel() == 1 else 1)
    w13 = prepared.w13.view(torch.int32)
    w2 = prepared.w2.view(torch.int32)
    s13 = prepared.w13_scale.view(torch.uint8)
    s2 = prepared.w2_scale.view(torch.uint8)
    u32 = cutlass.Uint32
    launches.quant(
        ptr(cutlass.BFloat16, a),
        ptr(cutlass.Float32, a1, 4),
        ptr(u32, x_q),
        ptr(u32, x_sf, 4),
        ptr(u32, x_q2),
        ptr(u32, x_sf2, 4),
        Int32(tokens),
        stream,
    )
    launches.fc1(
        ptr(u32, x_q),
        ptr(u32, x_sf, 4),
        ptr(u32, x_q2),
        ptr(u32, x_sf2, 4),
        ptr(cutlass.Int32, w13),
        ptr(cutlass.Uint8, s13),
        ptr(cutlass.Float32, prepared.w13_global_scale, 4),
        ptr(cutlass.Float32, a1, 4),
        ptr(cutlass.Float32, a2, 4),
        ptr(cutlass.Int32, pir, 4),
        ptr(cutlass.Int32, beid, 4),
        ptr(cutlass.Int32, prc, 4),
        ptr(u32, act_q),
        ptr(u32, act_sf, 4),
        ptr(u32, act_q2),
        ptr(u32, act_sf2, 4),
        ptr(cutlass.BFloat16, y),
        Int32(routes),
        gs_stride,
        Int32(launches.grid),
        stream,
    )
    launches.fc2(
        ptr(u32, act_q),
        ptr(u32, act_sf, 4),
        ptr(u32, act_q2),
        ptr(u32, act_sf2, 4),
        ptr(cutlass.Int32, w2),
        ptr(cutlass.Uint8, s2),
        ptr(cutlass.Float32, prepared.w2_global_scale, 4),
        ptr(cutlass.Float32, a2, 4),
        ptr(cutlass.Float32, a2, 4),
        ptr(cutlass.Int32, pir, 4),
        ptr(cutlass.Int32, beid, 4),
        ptr(cutlass.Int32, prc, 4),
        ptr(u32, act_q),
        ptr(u32, act_sf, 4),
        ptr(u32, act_q2),
        ptr(u32, act_sf2, 4),
        ptr(cutlass.BFloat16, y),
        Int32(routes),
        gs_stride,
        Int32(launches.grid),
        stream,
    )
    _w4a16_topk_sum_launch_flat(
        y,
        output.view(-1),
        tokens,
        launches.topk,
        launches.hidden_size,
        "bf16",
        torch.cuda.current_stream().cuda_stream,
        topk_weights=topk_weights.view(-1),
        launcher=launches.topk_sum,
    )
    return output


__all__ = [
    "A4PackedPrefillGemm",
    "A4PackedQuantize",
    "PI",
    "WORD_ORDER",
    "W4A16A4PrefillLaunches",
    "a4_prefill_fits",
    "a4_prefill_enabled",
    "a4_prefill_supported",
    "compile_w4a16_a4_prefill",
    "run_w4a16_a4_prefill",
]
