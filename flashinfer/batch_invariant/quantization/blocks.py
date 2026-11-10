"""Device building blocks of the MX quantizer, importable without the kernel's launch closure (this module imports the
engine core, ``cute_ext``, ``layout`` and ``mx_spec``; never a kernel module).

Stable names: ``cvt_f32_to_ue8m0``, ``cvt_f32x2_to_fp8x2`` (re-exported from ``cvt_fp8``), ``cvt_f32x8_to_e2m1x8``,
``block_amax``, ``pack_block_codes``, ``sf_offset_128x4_dev``, ``sf_offset_linear_dev``, ``SfTileStage``; plumbing:
``predicate``, ``tiled_copy``, ``Span``, ``LanePeers``, ``st_global_v4_u32``, ``byte_address``.

Every floating-point op here is a declared conversion primitive (inline PTX pinned by the IR tests and verified against
the CPU model over every binary32 input) or goes through ``CuteF32``; the amax tree is the engine's plan.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import canon
from .cvt_fp8 import cvt_f32x2_to_fp8x2  # noqa: F401  (re-exported: the e4m3 / e5m2 pair conversion)
from .layout import SF_COL_TILE, SF_ROW_TILE, SF_ROW_TILE_8X4
from .mx_spec import BLOCK, block_amax_placement, block_reduction, leaf_value

__all__ = [
    "cvt_f32_to_ue8m0",
    "cvt_f32x2_to_fp8x2",
    "cvt_f32x8_to_e2m1x8",
    "block_amax",
    "pack_block_codes",
    "sf_offset_128x4_dev",
    "sf_offset_8x4_dev",
    "sf_offset_linear_dev",
    "SfTileStage",
    "predicate",
    "tiled_copy",
    "Span",
    "LanePeers",
    "st_global_v4_u32",
    "byte_address",
]


# ----------------------------------------------------------------------------- declared conversions
_UE8M0_RN_PTX = """
            mov.b32 bits, $2;
            and.b32 bits, bits, 0x7FFFFFFF;
            shr.u32 e, bits, 23;
            and.b32 m, bits, 0x7FFFFF;
            setp.ge.u32 p1, m, 0x400000;
            selp.u32 b1, 1, 0, p1;
            add.u32 rn, e, b1;
            setp.ge.u32 p2, m, 0x600000;
            selp.u32 rs, 1, 0, p2;
            setp.eq.u32 ps, e, 0;
            selp.u32 r, rs, rn, ps;
            min.u32 r, r, 254;
            setp.eq.u32 pe, e, 255;
            setp.ne.u32 pm, m, 0;
            and.pred pn, pe, pm;
            selp.u32 r, 255, r, pn;
            cvt.u16.u32 pair, r;
"""


@dsl_user_op
def cvt_f32_to_ue8m0(a: Float32, rnd: cutlass.Constexpr[str], *, loc=None, ip=None):
    """The MX classes' scale conversion ``cvt_out(v, "ue8m0_<rnd>")`` of one value. Returns ``(byte, value)``: the
    UE8M0 byte (in a ``Uint32``) and its value ``2^(byte - 127)`` widened exactly to f32 (``cvt.rn.bf16x2.ue8m0x2`` --
    every UE8M0 value is a bf16 value is an f32 value, incl. the subnormal 2^-127 -- then the exact bf16 -> f32 shift).
    ``rnd`` = ``"rp"`` / ``"rz"``: the hardware ``cvt.<rnd>.satfinite.ue8m0x2.f32`` (ceil / floor to a power of two,
    saturating at byte 254, values <= 2^-127 to byte 0); ``"rn"``: the nearest power of two in linear space, ties up,
    by exact integer arithmetic on the bits of the magnitude (normal: byte = exponent + [mantissa >= 1.5]; subnormal:
    byte = [value >= 1.5 x 2^-127]; >= 2^127 and inf -> 254; NaN -> 255). Every mode converts the magnitude, as the hardware
    modes do (never reached on the domain: the amax is ``fabs(x) >= +0``). CPU model: ``fp_interface.exact.to_ue8m0``."""
    head = (
        f"            cvt.{rnd}.satfinite.ue8m0x2.f32 pair, zero, $2;\n"
        if rnd in ("rp", "rz")
        else _UE8M0_RN_PTX
    )
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [Float32(a).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 pair;
            .reg .b32 b2, lo, bits, e, m, b1, rn, rs, r;
            .reg .pred p1, p2, ps, pe, pm, pn;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
"""
        + head
        + """
            cvt.rn.bf16x2.ue8m0x2 b2, pair;
            and.b32 lo, b2, 0xFFFF;
            shl.b32 lo, lo, 16;
            mov.b32 $1, lo;
            cvt.u32.u16 $0, pair;
        }
        """,
        "=r,=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_f32x8_to_e2m1x8(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Four ``cvt.rn.satfinite.e2m1x2.f32``: element ``2j`` in the low nibble of byte ``j``, the four bytes packed
    little-endian into one 32-bit word (byte 0 = elements 0, 1). The one e2m1 conversion of the MXFP4 and the NVFP4
    classes (the strings are pinned by the IR tests; the CPU model is ``exact.to_e2m1``)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v).ir_value(loc=loc, ip=ip)
                for v in (v0, v1, v2, v3, v4, v5, v6, v7)
            ],
            """
        {
            .reg .b8 b0, b1, b2, b3;
            cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;
            cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;
            cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;
            cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;
            mov.b32 $0, {b0, b1, b2, b3};
        }
        """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def pack_block_codes(fp, ys, elem_fmt: str):
    """The element conversion of a thread's ``V`` pre-conversion values (``saturate(quant_pre(x, inv))``, fp32): the
    packed 32-bit words of their codes, low byte first. ``e4m3``: 4 values per word (two
    ``cvt.rn.satfinite.e4m3x2.f32``); ``e2m1``: 8 values per word (four ``cvt.rn.satfinite.e2m1x2.f32``)."""
    V = len(ys)
    if elem_fmt == "e4m3":
        assert V % 4 == 0, V
        out = []
        for q in range(V // 4):
            p0 = cvt_f32x2_to_fp8x2(ys[4 * q], ys[4 * q + 1], "e4m3")
            p1 = cvt_f32x2_to_fp8x2(ys[4 * q + 2], ys[4 * q + 3], "e4m3")
            out.append(p0.to(Uint32) | (p1.to(Uint32) << 16))
        return out
    assert elem_fmt == "e2m1" and V % 8 == 0, (elem_fmt, V)
    return [cvt_f32x8_to_e2m1x8(*ys[8 * q : 8 * q + 8]) for q in range(V // 8)]


# ----------------------------------------------------------------------------- the block amax (the engine's plan)
class LanePeers(canon.Peers):
    """Hardware peers for lane-level steps: xor shuffles. The quantizers' plans have no warp-level steps."""

    def shuffle_xor(self, key, value, mask):
        if value is None:
            return None
        return cute.arch.shuffle_sync_bfly(value, offset=mask)

    def exchange(
        self, step, state
    ):  # pragma: no cover - no warp factors in the block amax
        raise NotImplementedError("the block amax has no warp-level factor")


def block_amax(fp, vals, lanes_per_block: int):
    """The canonical block amax from a thread's ``V = 32 / lanes_per_block`` consecutive fp32 values of a block, under
    the placement ``[(V, reg), (lanes_per_block, lane)]`` of the ``elem`` digit: the register tree over ``vals`` (leaf
    ``fabs``), then the butterfly over the ``lanes_per_block`` ADJACENT lanes that share the block (lane ``l`` holds
    part ``l % lanes_per_block``, i.e. elements ``[(l % lpb) * V, +V)``). Returns the amax in every lane of the block.
    The plan is the engine's (``canon.init_state`` / ``canon.run``): the same tree the standalone kernels compute."""
    V = len(vals)
    assert V * lanes_per_block == BLOCK, (V, lanes_per_block)

    def no_leaf(fp_, e):  # pragma: no cover - the loader below supplies the leaves
        raise RuntimeError("block_amax: leaves come from the loader")

    plan = canon.plan(block_reduction(no_leaf), block_amax_placement(V))
    state, start = canon.init_state(plan, fp, lambda key: leaf_value(fp, vals[key[0]]))
    return canon.run(plan, state, start, LanePeers(), fp)


# ----------------------------------------------------------------------------- scale-factor stores
def sf_offset_128x4_dev(row: Int64, col, pcols: Int32) -> Int64:
    """``layout.sf_offset_128x4`` on device values (the row-tile term in 64 bits; ``col`` may be a Python int)."""
    col = Int32(col)
    inner = (col % SF_COL_TILE) + (col // SF_COL_TILE) * (SF_ROW_TILE * SF_COL_TILE)
    r32 = (row % SF_ROW_TILE).to(Int32)
    inner = inner + (r32 % 32) * 16 + (r32 // 32) * 4
    return (row // SF_ROW_TILE) * (SF_ROW_TILE * pcols).to(Int64) + inner.to(Int64)


def sf_offset_8x4_dev(row: Int64, col, pcols: Int32) -> Int64:
    """``layout.sf_offset_8x4`` on device values (``col`` may be a Python int)."""
    col = Int32(col)
    inner = (
        (col % SF_COL_TILE)
        + (row % SF_ROW_TILE_8X4).to(Int32) * SF_COL_TILE
        + (col // SF_COL_TILE) * (SF_ROW_TILE_8X4 * SF_COL_TILE)
    )
    return (row // SF_ROW_TILE_8X4) * (SF_ROW_TILE_8X4 * pcols).to(Int64) + inner.to(
        Int64
    )


def sf_offset_linear_dev(row: Int64, col, cols: Int32) -> Int64:
    """``layout.sf_offset_linear`` on device values (``col`` may be a Python int: a config whose block factors are all
    register-level gives the plan's digit value as a constant)."""
    return row * cols.to(Int64) + Int32(col).to(Int64)


@dsl_user_op
def st_global_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
):
    """One 16-byte store of four 32-bit words (the tile placement's scale copy-out)."""
    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def byte_address(tensor: cute.Tensor, offset) -> Int64:
    """The address of ``tensor[offset]`` (a global uint8 tensor) as an integer."""
    return Int64(llvm.ptrtoint(T.i64(), (tensor.iterator + offset).llvm_ptr))


class SfTileStage:
    """The 128x4 placement of a 128-row x ``tb``-block scale tile in shared memory (``128 * tb`` bytes, ``tb % 4 == 0``):
    ``local_offset(blk_local, r)`` is where row ``r``'s scale of local block ``blk_local`` goes (already in the
    swizzled order: the region is ``tb / 4`` consecutive 512-byte tiles), ``copy_out`` writes the region to its
    place in the global buffer with one 16-byte store per thread, skipping column groups beyond the padded columns."""

    def __init__(self, tb: int, num_threads: int):
        assert tb % SF_COL_TILE == 0, tb
        self.tb, self.num_threads = tb, num_threads
        self.nbytes = SF_ROW_TILE * tb

    @staticmethod
    def local_offset(blk_local, r):
        return (
            (blk_local // SF_COL_TILE) * (SF_ROW_TILE * SF_COL_TILE)
            + (blk_local % SF_COL_TILE)
            + (r % 32) * 16
            + (r // 32) * 4
        )

    @cute.jit
    def copy_out(
        self,
        sbuf: cute.Tensor,
        mS: cute.Tensor,
        tidx: Int32,
        row_tile: Int32,
        col_tile: Int32,
        pcols: Int32,
    ):
        s32 = cute.make_tensor(
            cute.recast_ptr(sbuf.iterator, dtype=Uint32),
            cute.make_layout((self.nbytes // 4,)),
        )
        n_chunks = self.nbytes // 16
        ncg = pcols // SF_COL_TILE
        base = row_tile.to(Int64) * (SF_ROW_TILE * pcols).to(Int64) + (
            col_tile * (SF_ROW_TILE * self.tb)
        ).to(Int64)
        for c0 in cutlass.range_constexpr(0, n_chunks, self.num_threads):
            c = c0 + tidx
            if (c < n_chunks) & (col_tile * (self.tb // SF_COL_TILE) + c // 32 < ncg):
                st_global_v4_u32(
                    byte_address(mS, base + (c * 16).to(Int64)),
                    s32[4 * c],
                    s32[4 * c + 1],
                    s32[4 * c + 2],
                    s32[4 * c + 3],
                )


# ----------------------------------------------------------------------------- plumbing
@cute.jit
def predicate(tXcX: cute.Tensor, M: Int64, N: Int32) -> cute.Tensor:
    """Per-vector predicate: (row, col) < (M, N)."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k], (M, N)
            )
    return tXpX


def tiled_copy(mT, bits, tv_layout, tiler_mn, vtid):
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), mT.element_type, num_bits_per_copy=bits
    )
    return cute.make_tiled_copy(atom, tv_layout, tiler_mn).get_slice(vtid), atom


class Span:
    """A thread's whole row span as one zero-filled, predicated register fragment."""

    def __init__(self, gX, atom, pred, dtype):
        self.gX, self.atom, self.pred, self.dtype = gX, atom, pred, dtype

    def load(self):
        frag = cute.make_fragment_like(self.gX)
        frag.store(cute.zeros_like(frag, dtype=self.dtype))
        cute.copy(self.atom, self.gX, frag, pred=self.pred)
        return frag
