# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Warp-specialized, persistent, TMA MXFP8 row-scale -> column-scale requant.

Data uses the padded row-major dispatch pool. Destination SF is concatenated
per expert in ``[hidden_atom][token_atom]`` order.
"""

from typing import Literal

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cutlass_dsl import (
    Float32,
    Int32,
    Int64,
    T,
    Uint8,
    dsl_user_op,
)
from cutlass._mlir.dialects import arith, llvm
from cutlass.cute.typing import AddressSpace

from common.megamoe_constants import (
    Fp8E4M3FNMax,
    Fp8E5M2Max,
    Mxfp8BlockSize,
    SfPaddingBlock,
)


def _lcm(a: int, b: int) -> int:
    from math import gcd

    return a * b // gcd(a, b)


def _smem_capacity() -> int:
    """Max dynamic SMEM per CTA, from CUTLASS's per-arch table."""
    try:
        from cutlass.utils import get_smem_capacity_in_bytes

        return int(get_smem_capacity_in_bytes())
    except Exception:
        return 227 * 1024


# ptxas 13.2 accepts ``scaled::n1`` only on sm_107a -- not sm_107, sm_107f, nor
# the later sm_110a / sm_120a -- so this is an exact set, not a floor.
_SCALED_CVT_ARCHS = frozenset({(10, 7)})


def _target_arch_tuple() -> "tuple[int, int, str]":
    """``(major, minor, suffix)`` of the active cuTeDSL compilation target."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return int(arch.major), int(arch.minor), (getattr(arch, "suffix", "") or "")


def _scaled_cvt_available() -> bool:
    """Can this target assemble ``cvt...scaled::n1::ue8m0.e4m3x2.bf16x2``?"""
    major, minor, suffix = _target_arch_tuple()
    if (major, minor) not in _SCALED_CVT_ARCHS:
        return False
    if suffix != "a":
        raise ValueError(
            f"MXFP8 column requant targets sm_{major}{minor}{suffix}, but its "
            f"block-scaled requant instruction "
            f"'cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.bf16x2' is accepted by "
            f"ptxas only for the 'a' architecture variant; sm_{major}{minor} and "
            f"sm_{major}{minor}f both fail with \"Arguments mismatch for "
            f"instruction 'cvt'\".  Compile for sm_{major}{minor}a, or pass "
            f"scaled_cvt=False to select the portable requant path."
        )
    return True


_SM_COUNT_CACHE: "list[int | None]" = [None]


def _resolve_sm_count(default: int) -> int:
    """SM count of the current device."""
    if _SM_COUNT_CACHE[0] is None:
        n = 0
        try:
            import ctypes

            lib = ctypes.CDLL("libcuda.so.1")
            lib.cuInit(0)
            dev = ctypes.c_int()
            if lib.cuDeviceGet(ctypes.byref(dev), 0) == 0:
                val = ctypes.c_int()
                # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
                if lib.cuDeviceGetAttribute(ctypes.byref(val), 16, dev) == 0:
                    n = int(val.value)
        except Exception:
            n = 0
        _SM_COUNT_CACHE[0] = n
    return _SM_COUNT_CACHE[0] or default


def _address_value(pointer_or_address, *, loc=None, ip=None):
    if isinstance(pointer_or_address, Int64):
        return pointer_or_address.ir_value()
    return pointer_or_address.toint(loc=loc, ip=ip).ir_value()


@dsl_user_op
def tma_load_1d(
    destination_smem, source_gmem, mbarrier_smem, num_bytes, *, loc=None, ip=None,
) -> None:
    """Issue a 1D GMEM-to-SMEM bulk copy."""
    llvm.inline_asm(
        None,
        [
            destination_smem.toint(loc=loc, ip=ip).ir_value(),
            _address_value(source_gmem, loc=loc, ip=ip),
            num_bytes.ir_value(),
            mbarrier_smem.toint(loc=loc, ip=ip).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_bulk_s2g(destination_gmem, source_smem, num_bytes, *, loc=None, ip=None) -> None:
    """Issue a 1D SMEM-to-GMEM bulk copy; the caller commits the group."""
    llvm.inline_asm(
        None,
        [
            _address_value(destination_gmem, loc=loc, ip=ip),
            source_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _fp8x2_mnemonic(fp8_type) -> str:
    if fp8_type is cutlass.Float8E4M3FN:
        return "e4m3x2"
    if fp8_type is cutlass.Float8E5M2:
        return "e5m2x2"
    raise TypeError(f"unsupported FP8 type {fp8_type}")


@dsl_user_op
def cvt_scaled_up_bf16x2(pair_b32, scale_b32, half: int, fp8_type, *, loc=None, ip=None) -> Int32:
    """Two FP8 values + their E8M0 scale -> BF16x2, in one SASS instruction."""
    mn = _fp8x2_mnemonic(fp8_type)
    asm = (
        "{\n"
        " .reg .b16 a0,a1,s0,s1;\n"
        " mov.b32 {a0,a1}, $1;\n"
        " mov.b32 {s0,s1}, $2;\n"
        f" cvt.rn.scaled::n2::ue8m0.bf16x2.{mn} $0, a{half}, s0;\n"
        "}"
    )
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(pair_b32).ir_value(loc=loc, ip=ip), Int32(scale_b32).ir_value(loc=loc, ip=ip)],
            asm,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_scaled_dn_fp8x2(v_bf16x2, raw_b32, fp8_type, *, loc=None, ip=None) -> Int32:
    """BF16x2 + one E8M0 scale -> two FP8 bytes, in one SASS instruction."""
    mn = _fp8x2_mnemonic(fp8_type)
    asm = (
        "{\n"
        " .reg .b16 q, s16, junk;\n"
        " .reg .b8  sb;\n"
        " mov.b32 {s16, junk}, $2;\n"
        " cvt.u8.u16 sb, s16;\n"
        f" cvt.rn.satfinite.scaled::n1::ue8m0.{mn}.bf16x2 q, $1, sb;\n"
        " cvt.u32.u16 $0, q;\n"
        "}"
    )
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(v_bf16x2).ir_value(loc=loc, ip=ip), Int32(raw_b32).ir_value(loc=loc, ip=ip)],
            asm,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_dn_fp8x2_portable(v_bf16x2, inv_lo, inv_hi, fp8_type, *, loc=None, ip=None) -> Int32:
    """Two hidden columns of one token, as BF16x2, plus those two columns'
    exact FP32 reciprocal scales -> two FP8 bytes in bits [15:0] (byte 0 from
    the low BF16 half, byte 1 from the high half).

    Same rounding as ``cvt_scaled_dn_fp8x2``, not an approximation of it: BF16
    -> FP32 is an exact left shift, the reciprocal is an exact power of two, and
    ``cvt.rn.satfinite`` is the RNE-and-saturate the hardware instruction
    applies.  Taking a scale per half is what lets the caller skip the transpose
    that the one-scale-per-pair hardware instruction forces.
    """
    mn = _fp8x2_mnemonic(fp8_type)
    asm = (
        "{\n"
        " .reg .b32 a, b;\n"
        " .reg .b16 q;\n"
        " shl.b32 a, $1, 16;\n"
        " and.b32 b, $1, 0xffff0000;\n"
        " mul.f32 a, a, $2;\n"
        " mul.f32 b, b, $3;\n"
        # ``cvt d, a, b`` yields d[15:8] = cvt(a) and d[7:0] = cvt(b), so the
        # HIGH column has to be the first source for byte 0 to be the low one.
        f" cvt.rn.satfinite.{mn}.f32 q, b, a;\n"
        " cvt.u32.u16 $0, q;\n"
        "}"
    )
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(v_bf16x2).ir_value(loc=loc, ip=ip),
                Float32(inv_lo).ir_value(loc=loc, ip=ip),
                Float32(inv_hi).ir_value(loc=loc, ip=ip),
            ],
            asm,
            "=r,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_xorsign_abs_bf16x2(a, b, *, loc=None, ip=None) -> Int32:
    """Packed magnitude max of two BF16x2."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e8m0_raw_from_bf16(bf16_bits, limit_exponent: int, *, loc=None, ip=None) -> Int32:
    """E8M0 raw byte for a non-negative BF16 magnitude."""
    bits = Int32(bf16_bits) << Int32(16)
    biased = (bits + Int32(0x1FFFFF - (limit_exponent << 23))) >> Int32(23)
    return Int32(
        arith.select(
            (bits >= Int32(0x7F800000)).ir_value(loc=loc, ip=ip),
            Int32(254).ir_value(loc=loc, ip=ip),
            cutlass.max(Int32(0), biased).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bits_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def fp8_limit_exponent(limit: float) -> int:
    """Exponent ``k`` of an FP8 limit written as ``1.75 * 2**k``."""
    scaled = limit / 1.75
    exponent = int(scaled).bit_length() - 1
    if float(1 << exponent) != scaled:
        raise ValueError(
            f"FP8 limit {limit} is not of the form 1.75 * 2**k, which the integer E8M0 encoding assumes."
        )
    return exponent


class Mxfp8ColRequant:
    """Warp-specialized persistent TMA launcher for token-axis MXFP8 requant."""

    # --- fixed MXFP8 / SF-atom geometry (do not retune) ---------------------
    TokensPerBlock: int = Mxfp8BlockSize  # 32: one E8M0 scale covers 32 values
    SfAtomBytes: int = 512
    SfAtomNonK: int = 128
    SfAtomKBanks: int = 4

    TokenPaddingBlocks: tuple = (128, 256)

    # --- this kernel's tile ------------------------------------------------
    TILE_TOK: int = 128  # == SfAtomNonK, so a tile is one whole SF atom row
    NSTAGE: int = 2  # 2 is what leaves shared memory for 2 CTAs per SM

    # Tile width and per-lane access width, per requant path.  Both are tuned:
    # the two paths have different shared-memory budgets and different
    # instruction mixes, and the pairs below are the measured optima.
    TileHidScaled: int = 512
    ColsPerLaneScaled: int = 8
    TileHidPortable: int = 256
    ColsPerLanePortable: int = 4

    # Sub-tile width of the consumer, in hidden columns.  See _sp_tile_body.
    # Must divide ColsPerLane and divide 32.
    SP_LW: int = 4

    # Declared upper bound on the block size.  ptxas budgets
    # 512//ceil(T/128) registers from it, so 576 yields 96 registers per
    # thread, which is what two resident CTAs need.
    MaxNTidTarget: int = 576

    # Padding between consecutive SF atoms in shared memory, in bytes; it is
    # what keeps the consumer's strided scale accesses off one bank.
    SfInPad: int = 16
    SfOutPad: int = 16

    # Grid depth, in resident waves.  The curve is broad and flat above this.
    GridWaves: int = 24

    ProducerWarps: int = 1
    # ConsumerWarps is derived:
    #     (TILE_TOK / TokensPerBlock) * (TILE_HID / (32 * ColsPerLane))

    SmCount: int = 148  # only a fallback; _resolve_sm_count queries the driver
    # Upper bound on the grid: below this many tiles per CTA the
    # O(num_experts) prologue dominates.  Stated per 256 experts.
    MinTilesPerCta: int = 4
    MinTilesRefExperts: int = 256

    ConsumerBarrierId: int = 1

    @classmethod
    def _require_token_padding_block(cls, block) -> int:
        """Refuse any token padding block this kernel is not built for."""
        if isinstance(block, bool) or not isinstance(block, int):
            raise ValueError(
                f"token_padding_block must be an int, got {block!r} "
                f"({type(block).__name__})."
            )
        if block not in cls.TokenPaddingBlocks:
            raise ValueError(
                f"token_padding_block must be one of "
                f"{tuple(cls.TokenPaddingBlocks)}, got {block}.  The work tile is "
                f"TILE_TOK = SfAtomNonK = {cls.SfAtomNonK} tokens, i.e. one "
                f"{cls.SfAtomBytes} B SF atom of {cls.SfAtomNonK}x{cls.SfAtomNonK} "
                f"(token x hidden), so an expert's padded extent has to be a whole "
                f"number of {cls.SfAtomNonK}-token tiles for a tile to belong to "
                f"exactly one expert; {cls.TokenPaddingBlocks} are the only blocks "
                f"the dispatch pool emits and the only ones validated."
            )
        return int(block)

    def __init__(
        self,
        hidden: int,
        num_experts: int,
        max_total_tokens: int,
        quant_type: Literal["mxfp8_e4m3", "mxfp8_e5m2"],
        num_persistent_ctas: int = -1,
        token_padding_block: int = SfPaddingBlock,
        sf_padding_block: int = SfPaddingBlock,
        *,
        scaled_cvt: "bool | None" = None,
    ) -> None:
        """``scaled_cvt`` selects the requant path: ``None`` asks the
        compilation target, ``True`` and ``False`` force the block-scaled and
        the portable path so that either can be exercised on one machine.
        Everything else is derived from the problem."""
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.max_total_tokens = int(max_total_tokens)
        self.quant_type = quant_type
        self.token_padding_block = int(token_padding_block)
        self.sf_padding_block = int(sf_padding_block)

        self._require_token_padding_block(self.token_padding_block)
        if self.sf_padding_block != self.SfAtomNonK:
            raise ValueError(
                f"sf_padding_block must be {self.SfAtomNonK} for the 32x4x4 atom layout, "
                f"got {self.sf_padding_block}."
            )

        if scaled_cvt is None:
            self.scaled_cvt = _scaled_cvt_available()
        elif scaled_cvt:
            if not _scaled_cvt_available():
                _major, _minor, _suffix = _target_arch_tuple()
                raise ValueError(
                    f"scaled_cvt=True forces the block-scaled cvt consumer, but "
                    f"the target is sm_{_major}{_minor}{_suffix} and "
                    f"'cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.bf16x2' is not "
                    f"available there.  Pass scaled_cvt=None to let the target "
                    f"choose."
                )
            self.scaled_cvt = True
        else:
            self.scaled_cvt = False

        if self.hidden <= 0:
            raise ValueError(f"hidden must be positive, got {self.hidden}.")

        # The tile has to be a multiple of lcm(SF atom, 32*C) and divide hidden,
        # so a shape the tuned pair cannot tile falls back to a narrower access.
        if self.scaled_cvt:
            _cols_choices = (self.ColsPerLaneScaled, self.ColsPerLanePortable)
            _preferred_tile_hid = self.TileHidScaled
        else:
            _cols_choices = (self.ColsPerLanePortable,)
            _preferred_tile_hid = self.TileHidPortable
        _picked = None
        for _c in _cols_choices:
            _grain = _lcm(self.SfAtomNonK, 32 * _c)
            _choices = [
                t
                for t in range(_grain, _preferred_tile_hid + 1, _grain)
                if self.hidden % t == 0
            ]
            if _choices:
                _picked = (_c, _choices)
                break
        if _picked is None:
            raise ValueError(
                f"hidden={self.hidden} is not supported: it must be a multiple "
                f"of {_lcm(self.SfAtomNonK, 32 * _cols_choices[-1])}."
            )
        self.ColsPerLane, self._tile_hid_choices = _picked
        self.TILE_HID = self._tile_hid_choices[-1]

        # Narrow shapes can end up with fewer columns per lane than the sub-tile
        # width, so clamp instead of refusing; the split is then trivial.
        self.sp_lw = min(self.SP_LW, self.ColsPerLane)
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}.")
        if self.max_total_tokens <= 0:
            raise ValueError(f"max_total_tokens must be positive, got {self.max_total_tokens}.")

        if quant_type == "mxfp8_e4m3":
            self.quant_dtype = cutlass.Float8E4M3FN
            self._data_limit_exponent = fp8_limit_exponent(float(Fp8E4M3FNMax))
        elif quant_type == "mxfp8_e5m2":
            self.quant_dtype = cutlass.Float8E5M2
            self._data_limit_exponent = fp8_limit_exponent(float(Fp8E5M2Max))
        else:
            raise ValueError(f"Unsupported quant_type: {quant_type!r}")
        self.sf_dtype = cutlass.Float8E8M0FNU

        self.sf_in_pad = self.SfInPad
        self.sf_out_pad = self.SfOutPad
        self.smem_capacity = _smem_capacity()
        while (
            self._smem_bytes_for(self.TILE_HID, self.NSTAGE) > self.smem_capacity
            and len(self._tile_hid_choices) > 1
        ):
            self._tile_hid_choices.pop()
            self.TILE_HID = self._tile_hid_choices[-1]

        self.NumStages = self.NSTAGE
        self.TmaBoxHidU32 = self.TILE_HID // 4

        self._hidden_atoms = self.hidden // self.SfAtomNonK
        self.HidAtomsPerTile = self.TILE_HID // self.SfAtomNonK
        self.HidSegs = self.TILE_HID // (32 * self.ColsPerLane)
        self.TokBlocks = self.TILE_TOK // self.TokensPerBlock  # 4
        self.ConsumerWarps = self.HidSegs * self.TokBlocks
        self.SfInStride = self.SfAtomBytes + self.sf_in_pad
        self.SfTileBytes = self.HidAtomsPerTile * self.SfInStride
        self.SfTileXferBytes = self.HidAtomsPerTile * self.SfAtomBytes

        if (self.ProducerWarps + self.ConsumerWarps) * 32 > 1024:
            raise ValueError(
                f"hidden={self.hidden} needs "
                f"{(self.ProducerWarps + self.ConsumerWarps) * 32} threads per "
                f"CTA, over the 1024-thread hardware limit."
            )
        self.WarpsPerCta = self.ProducerWarps + self.ConsumerWarps
        self.ThreadsPerCta = self.WarpsPerCta * 32
        # ``.maxntid`` is an upper bound, so it can never be below the launch.
        self.MaxNTid = max(self.MaxNTidTarget, self.ThreadsPerCta)

        # --- SMEM ------------------------------------------------------------
        self.smem_data_bytes = self.NumStages * self.TILE_TOK * self.TILE_HID
        self.smem_sf_in_bytes = self.NumStages * self.SfTileBytes
        self.SfOutStride = self.SfAtomBytes + self.sf_out_pad
        self.smem_sf_out_bytes = self.HidAtomsPerTile * self.SfOutStride
        self.smem_table_bytes = 3 * (self.num_experts + 1) * 4
        self.smem_bytes = (
            self.smem_data_bytes
            + self.smem_sf_in_bytes
            + self.smem_sf_out_bytes
            + self.smem_table_bytes
            + 2 * self.NumStages * 8
            + 256
        )
        if self.smem_bytes > self.smem_capacity:
            raise ValueError(
                f"hidden={self.hidden} needs {self.smem_bytes} B of shared "
                f"memory per CTA, over this target's "
                f"{self.smem_capacity} B limit."
            )

        self.hidden_groups = self.hidden // self.TILE_HID

        # --- grid ------------------------------------------------------------
        # The grid quantum is RES = (resident CTAs) = CtasPerSm * SM_count: a
        # grid that is not a multiple of RES leaves a fractional resident wave,
        # which is a large loss.  Across multiples of RES the curve is broad and
        # flat, so one tuned wave count serves every production size.
        self.SmCount = _resolve_sm_count(type(self).SmCount)
        # Both gates: SMEM, and the 8-warps-per-scheduler cap (warps go to the
        # 4 schedulers round robin, so one CTA occupies ceil(warps/4) slots).
        _warp_gate = 8 // -(-self.WarpsPerCta // 4)
        self.CtasPerSm = max(1, min(2, self.smem_capacity // self.smem_bytes, _warp_gate))
        self.ResidentCtas = self.CtasPerSm * self.SmCount
        # Every CTA pays an O(num_experts) prefix-table prologue, so the grid has
        # an upper bound, and that bound rises in proportion to the expert count:
        # with too few tiles per CTA the prologue dominates the tile work.
        _min_tiles = self.MinTilesPerCta * max(
            1, -(-self.num_experts // self.MinTilesRefExperts)
        )
        # The wave count is tuned for 2 resident CTAs.  A shape that gets only
        # one keeps a single-wave grid rather than extrapolating that tuning
        # point outside the regime it was taken in.
        _want = self.GridWaves if self.CtasPerSm >= 2 else 1
        _max_tiles = -(-self.max_total_tokens // self.TILE_TOK) * self.hidden_groups
        _waves = max(1, min(_want, _max_tiles // (_min_tiles * self.ResidentCtas)))
        if num_persistent_ctas > 0:
            self.num_persistent_ctas = int(num_persistent_ctas)
        else:
            self.num_persistent_ctas = _waves * self.ResidentCtas
        self.grid = self.num_persistent_ctas

        # Reported by the runner's PASS line; this kernel has a fixed split.
        self.HiddenPerCta = self.TILE_HID
        self.hidden_tiles_per_work = 1

        # Binary-search ladder over the valid-token prefix table: the powers of
        # two below num_experts, largest first.
        steps = []
        span = 1
        while span < self.num_experts:
            span <<= 1
        span >>= 1
        while span >= 1:
            steps.append(span)
            span >>= 1
        self._search_steps = tuple(steps)
        self._search_needs_guard = (self.num_experts & (self.num_experts - 1)) != 0
        self._experts_per_lane = (self.num_experts + 31) // 32

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        src_data: cute.Tensor,
        src_sf_u8: cute.Tensor,
        expert_token_sizes: cute.Tensor,
        dst_data: cute.Tensor,
        dst_sf_u8: cute.Tensor,
        cuda_stream: cuda.CUstream,
        token_padding_block: cutlass.Constexpr = None,
    ) -> None:
        TOKPAD = cutlass.const_expr(
            self.token_padding_block if token_padding_block is None else token_padding_block
        )
        self._require_token_padding_block(TOKPAD)
        if cutlass.const_expr(TOKPAD != self.token_padding_block):
            raise ValueError(
                f"token_padding_block passed to __call__ ({TOKPAD}) disagrees with "
                f"the one the Mxfp8ColRequant was constructed with "
                f"({self.token_padding_block}); the launch geometry is derived from "
                f"the constructor value, so construct a new instance instead."
            )
        if cutlass.const_expr(src_data.element_type is not self.quant_dtype):
            raise TypeError(f"src_data must use {self.quant_dtype}, got {src_data.element_type}.")
        if cutlass.const_expr(dst_data.element_type is not self.quant_dtype):
            raise TypeError(f"dst_data must use {self.quant_dtype}, got {dst_data.element_type}.")

        HID_U32 = cutlass.const_expr(self.hidden // 4)
        BOX_H = cutlass.const_expr(self.TmaBoxHidU32)
        BOX_T = cutlass.const_expr(self.TILE_TOK)
        src_u32 = cute.make_tensor(
            cute.recast_ptr(src_data.iterator, dtype=cutlass.Uint32),
            cute.make_layout((src_data.shape[0], HID_U32), stride=(HID_U32, 1)),
        )
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            src_u32,
            cute.make_layout((BOX_T, BOX_H), stride=(BOX_H, 1)),
            (BOX_T, BOX_H),
        )

        dst_u32 = cute.make_tensor(
            cute.recast_ptr(dst_data.iterator, dtype=cutlass.Uint32),
            cute.make_layout((dst_data.shape[0], HID_U32), stride=(HID_U32, 1)),
        )
        tma_atom_st, tma_tensor_st = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dst_u32,
            cute.make_layout((BOX_T, BOX_H), stride=(BOX_H, 1)),
            (BOX_T, BOX_H),
        )
        k = self.ws_kernel(
            src_data, src_sf_u8, expert_token_sizes, dst_data, dst_sf_u8,
            tma_atom, tma_tensor, tma_atom_st, tma_tensor_st, TOKPAD,
        )
        # Validated in __init__, not here: __call__ is DSL-preprocessed and a
        # plain if/raise in a traced body is rejected at trace time.
        _ntid = cutlass.const_expr(self.MaxNTid)
        k.launch(
            grid=[self.grid, 1, 1],
            block=[self.ThreadsPerCta, 1, 1],
            max_number_threads=[_ntid, 1, 1],
            stream=cuda_stream,
        )

    # ------------------------------------------------------- prefix tables
    @cute.jit
    def _pad_up(self, count, block: int):
        if cutlass.const_expr(block > 0 and (block & (block - 1)) == 0):
            return (count + Int32(block - 1)) & Int32(-block)
        return ((count + Int32(block - 1)) // Int32(block)) * Int32(block)

    @cute.jit
    def warp_prefix_sum(self, value, lane_idx):
        acc = Int32(value)
        for shift in cutlass.range_constexpr(0, 5, 1):
            step = 1 << shift
            other = Int32(cute.arch.shuffle_sync_up(acc, Int32(step), mask_and_clamp=0))
            if lane_idx >= Int32(step):
                acc = acc + other
        return acc

    @cute.jit
    def build_prefix_tables(
        self, expert_token_sizes, tbl_vend, tbl_data, tbl_sf, lane_idx,
        token_padding_block: cutlass.Constexpr = None,
    ):
        """Exclusive prefixes of the padded-data / padded-SF row counts."""
        E = cutlass.const_expr(self.num_experts)
        EPL = cutlass.const_expr(self._experts_per_lane)
        TOKPAD = cutlass.const_expr(
            self.token_padding_block if token_padding_block is None else token_padding_block
        )
        self._require_token_padding_block(TOKPAD)

        counts = cute.make_rmem_tensor((EPL,), cutlass.Int32)
        local_data = Int32(0)
        local_sf = Int32(0)
        for slot in cutlass.range_constexpr(0, EPL, 1):
            expert = lane_idx * Int32(EPL) + Int32(slot)
            count = Int32(0)
            if expert < Int32(E):
                count = Int32(expert_token_sizes[expert])
            counts[slot] = count
            local_data = local_data + self._pad_up(count, TOKPAD)
            local_sf = local_sf + self._pad_up(count, self.sf_padding_block)

        base_data = self.warp_prefix_sum(local_data, lane_idx) - local_data
        base_sf = self.warp_prefix_sum(local_sf, lane_idx) - local_sf

        for slot in cutlass.range_constexpr(0, EPL, 1):
            expert = lane_idx * Int32(EPL) + Int32(slot)
            count = Int32(counts[slot])
            if expert <= Int32(E):
                tbl_vend[expert] = base_data + count
                tbl_data[expert] = base_data
                tbl_sf[expert] = base_sf
            base_data = base_data + self._pad_up(count, TOKPAD)
            base_sf = base_sf + self._pad_up(count, self.sf_padding_block)

        # When 32 * EPL == E no lane owns index E, so it is written separately.
        if (lane_idx + Int32(1)) * Int32(EPL) == Int32(E):
            tbl_vend[E] = base_data
            tbl_data[E] = base_data
            tbl_sf[E] = base_sf

    @cute.jit
    def find_expert(self, tbl, key):
        E = cutlass.const_expr(self.num_experts)
        lo = Int32(0)
        for step in self._search_steps:
            probe = lo + Int32(step)
            if cutlass.const_expr(self._search_needs_guard):
                if probe < Int32(E) and Int32(tbl[probe]) <= key:
                    lo = probe
            else:
                if Int32(tbl[probe]) <= key:
                    lo = probe
        return lo

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def ws_kernel(
        self,
        src_data: cute.Tensor,
        src_sf_u8: cute.Tensor,
        expert_token_sizes: cute.Tensor,
        dst_data: cute.Tensor,
        dst_sf_u8: cute.Tensor,
        tma_atom=None,
        tma_tensor=None,
        tma_atom_st=None,
        tma_tensor_st=None,
        token_padding_block: cutlass.Constexpr = None,
    ) -> None:
        # Compile-time constant, folded before a single instruction is emitted.
        TOKPAD = cutlass.const_expr(
            self.token_padding_block if token_padding_block is None else token_padding_block
        )
        self._require_token_padding_block(TOKPAD)
        TOK = cutlass.const_expr(self.TILE_TOK)
        W = cutlass.const_expr(self.TILE_HID)
        S = cutlass.const_expr(self.NumStages)
        SFB = cutlass.const_expr(self.SfTileBytes)
        NCONS = cutlass.const_expr(self.ConsumerWarps)
        table_len = cutlass.const_expr(self.num_experts + 1)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()
        warp_idx = tidx // Int32(32)
        lane_idx = tidx % Int32(32)

        smem = cutlass.utils.SmemAllocator()
        mbar_full = smem.allocate_array(cutlass.Int64, S)
        mbar_empty = smem.allocate_array(cutlass.Int64, S)
        tbl_vend = smem.allocate_tensor(cutlass.Int32, cute.make_layout((table_len,)), 16)
        tbl_data = smem.allocate_tensor(cutlass.Int32, cute.make_layout((table_len,)), 16)
        tbl_sf = smem.allocate_tensor(cutlass.Int32, cute.make_layout((table_len,)), 16)
        smem_sf_in = smem.allocate_array(self.sf_dtype, S * SFB, byte_alignment=128)
        smem_sf_out = smem.allocate_array(
            self.sf_dtype, cutlass.const_expr(self.smem_sf_out_bytes), byte_alignment=128
        )
        smem_data = smem.allocate_array(
            self.quant_dtype, self.NumStages * TOK * W, byte_alignment=128
        )

        if tidx == Int32(0):
            for s in cutlass.range_constexpr(0, S, 1):
                cute.arch.mbarrier_init(mbar_full + s, 1)
                cute.arch.mbarrier_init(mbar_empty + s, NCONS)
        cute.arch.mbarrier_init_fence()

        if warp_idx == Int32(0):
            self.build_prefix_tables(
                expert_token_sizes, tbl_vend, tbl_data, tbl_sf, lane_idx, TOKPAD
            )
        cute.arch.sync_threads()

        total_tiles = Int32(tbl_data[self.num_experts]) // Int32(TOK)

        smem_data_base = smem_data.toint()
        smem_sf_in_base = smem_sf_in.toint()
        smem_sf_out_base = smem_sf_out.toint()
        src_sf_base = src_sf_u8.iterator.toint()
        dst_sf_base = dst_sf_u8.iterator.toint()

        if warp_idx < Int32(self.ProducerWarps):
            self.produce(
                smem_data_base, smem_sf_in_base, mbar_full, mbar_empty,
                tbl_data, tbl_sf, src_sf_base,
                bidx, grid_dim_x, total_tiles, lane_idx,
                tma_atom, tma_tensor, TOKPAD,
            )
        else:
            self.consume_scaled(
                smem_data_base, smem_sf_in_base, smem_sf_out_base, mbar_full, mbar_empty,
                tbl_vend, tbl_data, tbl_sf, dst_sf_base,
                bidx, grid_dim_x, total_tiles,
                warp_idx - Int32(self.ProducerWarps), lane_idx,
                tma_atom_st, tma_tensor_st, TOKPAD,
            )

    # -------------------------------------------------------------- producer
    @cute.jit
    def produce(
        self, smem_data_base, smem_sf_in_base, mbar_full, mbar_empty,
        tbl_data, tbl_sf, src_sf_base,
        bidx, grid_dim_x, total_tiles, lane_idx,
        tma_atom=None, tma_tensor=None,
        token_padding_block: cutlass.Constexpr = None,
    ):
        TOKPAD = cutlass.const_expr(
            self.token_padding_block if token_padding_block is None else token_padding_block
        )
        self._require_token_padding_block(TOKPAD)
        TOK = cutlass.const_expr(self.TILE_TOK)
        W = cutlass.const_expr(self.TILE_HID)
        S = cutlass.const_expr(self.NumStages)
        SFB = cutlass.const_expr(self.SfTileBytes)
        # data bytes + SF bytes, both fenced by bar_full[stage]
        EXPECT = cutlass.const_expr(TOK * W + self.SfTileXferBytes)
        SF_PREFIX_DIFFERS = cutlass.const_expr(TOKPAD != self.sf_padding_block)

        BOX_H = cutlass.const_expr(self.TmaBoxHidU32)
        sD = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32, smem_data_base, AddressSpace.smem, assumed_align=128,
            ),
            cute.make_layout((TOK, BOX_H, S), stride=(BOX_H, 1, TOK * BOX_H)),
        )
        gD = cute.group_modes(
            cute.local_tile(tma_tensor, (TOK, BOX_H), (None, None)), 0, 2
        )
        tDsD, tDgD = cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1), cute.group_modes(sD, 0, 2), gD,
        )
        cpasync.prefetch_descriptor(tma_atom)

        t = Int32(0)
        work_idx = Int32(bidx)
        total_work = total_tiles * Int32(self.hidden_groups)
        while work_idx < total_work:
            stage = t % Int32(S)
            token_tile = work_idx // Int32(self.hidden_groups)
            hid_begin = (work_idx % Int32(self.hidden_groups)) * Int32(W)

            data_row0 = token_tile * Int32(TOK)
            sf_row0 = data_row0
            if cutlass.const_expr(SF_PREFIX_DIFFERS):
                owner = self.find_expert(tbl_data, data_row0)
                sf_row0 = Int32(tbl_sf[owner]) + (data_row0 - Int32(tbl_data[owner]))
                sf_row0 = cutlass.min(
                    sf_row0, Int32(tbl_sf[owner + Int32(1)]) - Int32(self.SfAtomNonK)
                )

            if t >= Int32(S):
                cute.arch.mbarrier_wait(mbar_empty + stage, ((t // Int32(S)) - Int32(1)) % Int32(2))

            if lane_idx == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_full + stage, Int32(EXPECT))
            cute.arch.sync_warp()

            if lane_idx == Int32(0):
                sf_src = (
                    Int64(src_sf_base)
                    + Int64(sf_row0 // Int32(self.SfAtomNonK)) * Int64(self._hidden_atoms * self.SfAtomBytes)
                    + Int64(hid_begin // Int32(self.SfAtomNonK)) * Int64(self.SfAtomBytes)
                )
                # One copy per SF atom: the atoms are padded apart in shared
                # memory, so they are not one contiguous run.
                for a in cutlass.range_constexpr(0, cutlass.const_expr(self.HidAtomsPerTile), 1):
                    tma_load_1d(
                        cute.make_ptr(
                            self.sf_dtype,
                            smem_sf_in_base + stage * Int32(SFB)
                            + Int32(a * self.SfInStride),
                            AddressSpace.smem, assumed_align=16,
                        ),
                        sf_src + Int64(a * self.SfAtomBytes),
                        mbar_full + stage,
                        Int32(self.SfAtomBytes),
                    )

            cute.copy(
                tma_atom,
                tDgD[(None, token_tile, hid_begin // Int32(W))],
                tDsD[(None, stage)],
                tma_bar_ptr=mbar_full + stage,
            )

            t = t + Int32(1)
            work_idx = work_idx + grid_dim_x

    # ------------------------------------------------- consumer (scaled cvt)
    @cute.jit
    def consume_scaled(
        self, smem_data_base, smem_sf_in_base, smem_sf_out_base, mbar_full, mbar_empty,
        tbl_vend, tbl_data, tbl_sf, dst_sf_base,
        bidx, grid_dim_x, total_tiles, cw, lane_idx,
        tma_atom_st=None, tma_tensor_st=None,
        token_padding_block: cutlass.Constexpr = None,
    ):
        """The single-pass all-BF16 consumer, shared by every target."""
        TOKPAD = cutlass.const_expr(
            self.token_padding_block if token_padding_block is None else token_padding_block
        )
        self._require_token_padding_block(TOKPAD)
        TOK = cutlass.const_expr(self.TILE_TOK)
        W = cutlass.const_expr(self.TILE_HID)
        S = cutlass.const_expr(self.NumStages)
        SFB = cutlass.const_expr(self.SfTileBytes)
        NB = cutlass.const_expr(self.TokensPerBlock)
        CONS_THREADS = cutlass.const_expr(self.ConsumerWarps * 32)
        HATOMS = cutlass.const_expr(self.HidAtomsPerTile)
        SF_PREFIX_DIFFERS = cutlass.const_expr(TOKPAD != self.sf_padding_block)
        SFOUT_ONE = cutlass.const_expr(self.HidAtomsPerTile * self.SfOutStride)
        C = cutlass.const_expr(self.ColsPerLane)     # hidden columns per lane
        SEGW = cutlass.const_expr(32 * C)            # columns per consumer segment

        tb = cw // Int32(self.HidSegs)
        seg = cw % Int32(self.HidSegs)

        LW = cutlass.const_expr(self.sp_lw)
        ldsw = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=LW * 8
        )
        data_lane_off = tb * Int32(NB * W) + seg * Int32(SEGW) + lane_idx * Int32(LW)

        hb0 = seg * Int32(C) + (lane_idx * Int32(LW)) // Int32(32)
        sf_lane_off = tb * Int32(4)

        BOX_HS = cutlass.const_expr(self.TmaBoxHidU32)
        sDo = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32, smem_data_base, AddressSpace.smem, assumed_align=128,
            ),
            cute.make_layout((TOK, BOX_HS, S), stride=(BOX_HS, 1, TOK * BOX_HS)),
        )
        gDo = cute.group_modes(
            cute.local_tile(tma_tensor_st, (TOK, BOX_HS), (None, None)), 0, 2
        )
        tDsDo, tDgDo = cpasync.tma_partition(
            tma_atom_st, 0, cute.make_layout(1), cute.group_modes(sDo, 0, 2), gDo,
        )
        cpasync.prefetch_descriptor(tma_atom_st)

        t = Int32(0)
        work_idx = Int32(bidx)
        total_work = total_tiles * Int32(self.hidden_groups)
        while work_idx < total_work:
            stage = t % Int32(S)
            token_tile = work_idx // Int32(self.hidden_groups)
            hid_begin = (work_idx % Int32(self.hidden_groups)) * Int32(W)

            data_row0 = token_tile * Int32(TOK)
            owner = self.find_expert(tbl_data, data_row0)
            valid_rows = cutlass.min(Int32(TOK), Int32(tbl_vend[owner]) - data_row0)

            # Destination SF is concatenated by expert, with token atoms
            # contiguous inside each hidden atom.
            sf_expert_token_atom = Int32(tbl_sf[owner]) // Int32(TOK)
            sf_token_atom = (
                data_row0 - Int32(tbl_data[owner])
            ) // Int32(TOK)
            sf_token_atoms = (
                Int32(tbl_sf[owner + Int32(1)]) - Int32(tbl_sf[owner])
            ) // Int32(TOK)
            sf_live = Int32(1)
            if cutlass.const_expr(SF_PREFIX_DIFFERS):
                sf_live = cutlass.min(Int32(1), cutlass.max(Int32(0), valid_rows))

            cute.arch.mbarrier_wait(mbar_full + stage, (t // Int32(S)) % Int32(2))

            stage_data = smem_data_base + stage * Int32(TOK * W) + data_lane_off
            stage_sf = smem_sf_in_base + stage * Int32(SFB) + sf_lane_off
            # A tile's sf_out store is drained before the stage is handed
            # back, so one buffer is enough.
            sfout = smem_sf_out_base

            self._sp_tile_body(
                stage_data, stage_sf, sfout, ldsw, hb0, seg, tb, lane_idx, valid_rows,
            )

            # Cross-proxy ordering, and it is NOT optional.  The consumer warps
            # wrote this tile's SMEM through the generic proxy; the store below
            # reads it through the async proxy.  Without this fence the store may
            # observe stale bytes, and the kernel becomes nondeterministic: the
            # same input yields different outputs from run to run.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=self.ConsumerBarrierId, number_of_threads=CONS_THREADS)

            # The whole tile goes out, padding rows included; those were
            # neutralised to zero in shared memory, which is what the pool
            # expects to find there.
            if cw == Int32(0):
                cute.copy(
                    tma_atom_st,
                    tDsDo[(None, stage)],
                    tDgDo[(None, token_tile, hid_begin // Int32(W))],
                )
            if cw == Int32(0) and lane_idx >= Int32(16) and lane_idx < Int32(16) + Int32(HATOMS) * sf_live:
                atom = lane_idx - Int32(16)
                cp_async_bulk_s2g(
                    Int64(dst_sf_base)
                    + (
                        Int64(sf_expert_token_atom) * Int64(self._hidden_atoms)
                        + (
                            Int64(hid_begin // Int32(self.SfAtomNonK))
                            + Int64(atom)
                        )
                        * Int64(sf_token_atoms)
                        + Int64(sf_token_atom)
                    )
                    * Int64(self.SfAtomBytes),
                    cute.make_ptr(
                        self.sf_dtype,
                        sfout + atom * Int32(self.SfOutStride),
                        AddressSpace.smem,
                        assumed_align=16,
                    ),
                    Int32(self.SfAtomBytes),
                )
            cute.arch.cp_async_bulk_commit_group()
            # Drain every store before the stage goes back to the producer.
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(barrier_id=self.ConsumerBarrierId, number_of_threads=CONS_THREADS)
            if lane_idx == Int32(0):
                cute.arch.mbarrier_arrive(mbar_empty + stage)

            t = t + Int32(1)
            work_idx = work_idx + grid_dim_x

    @cute.jit
    def _sp_tile_body(
        self, stage_data, stage_sf_base, sfout, ldsw, hb0, seg, tb, lane_idx, valid_rows,
    ):
        """The single-pass arithmetic for one lane's share of one tile."""
        TOK = cutlass.const_expr(self.TILE_TOK)
        W = cutlass.const_expr(self.TILE_HID)
        NB = cutlass.const_expr(self.TokensPerBlock)
        C = cutlass.const_expr(self.ColsPerLane)
        LW = cutlass.const_expr(self.sp_lw)
        NWc = cutlass.const_expr(LW // 4)   # 4-byte words per sub-tile row
        NPc = cutlass.const_expr(LW // 2)   # BF16x2 registers per sub-tile row
        NCH = cutlass.const_expr(C // LW)   # sub-tiles per lane per tile
        SEGW = cutlass.const_expr(32 * C)   # columns per consumer segment
        QT = self.quant_dtype

        # Dead rows ARE reachable: token_padding_block constrains an expert's
        # PADDED extent, not its valid count, so counts like 127,127,127 leave a
        # padded tail in every expert's last tile.  Neutralising them here needs
        # no barrier: a consumer thread only reads the bytes it just wrote.
        zeros = cute.make_rmem_tensor((NWc,), cutlass.Int32)
        for w in cutlass.range_constexpr(0, NWc, 1):
            zeros[w] = Int32(0)

        for ch in cutlass.range_constexpr(0, NCH, 1):
            base = stage_data + Int32(ch * 32 * LW)
            hbc = hb0 + Int32(ch * LW)
            sfb = (
                stage_sf_base
                + (hbc // Int32(4)) * Int32(self.SfInStride)
                + hbc % Int32(4)
            )

            # The dead row's SCALE has to be neutralised as well as its data.
            # ``cvt.rn.scaled::n2::ue8m0.bf16x2`` turns a NaN scale (raw 0xFF)
            # into a NaN BF16 even from a zero payload.  The amax survives --
            # ``max.xorsign.abs`` returns the non-NaN operand -- so the payload
            # stays right and only the PADDING row is written back as 0x7F
            # instead of 0x00.  The fc1 pool really does leave 0xFF in padding
            # scale bytes.  Writing 127 is the same value the masked arm
            # substitutes, paid once per tile instead of once per read.
            if valid_rows < Int32(TOK):
                for tt in cutlass.range_constexpr(0, NB, 1):
                    if tb * Int32(NB) + Int32(tt) >= valid_rows:
                        self._store_words(
                            base + Int32(tt * W), ldsw, zeros, NWc, LW
                        )
                        sf_t = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint8, sfb + Int32(tt * 16),
                                AddressSpace.smem, assumed_align=1,
                            ),
                            cute.make_layout((1,)),
                        )
                        sf_t[0] = Uint8(127)

            # ---- the one scan: unpack-with-scale, keep BF16, accumulate amax --
            d = [[None] * NPc for _ in range(NB)]
            acc = [Int32(0)] * NPc
            for tt in cutlass.range_constexpr(0, NB, 1):
                # Both loads are issued before either is consumed, so the scale
                # load overlaps the data load.
                words = self._load_words(base + Int32(tt * W), ldsw, NWc, LW)
                raw_sf = self._src_scale_raw(sfb + Int32(tt * 16))
                s16 = raw_sf | (raw_sf << Int32(8))
                for w in cutlass.range_constexpr(0, NWc, 1):
                    qw = Int32(words[w])
                    lo = cvt_scaled_up_bf16x2(qw, s16, 0, QT)
                    hi = cvt_scaled_up_bf16x2(qw, s16, 1, QT)
                    # Kept LIVE across the amax -- this is the single pass.
                    d[tt][2 * w] = lo
                    d[tt][2 * w + 1] = hi
                    acc[2 * w] = max_xorsign_abs_bf16x2(acc[2 * w], lo)
                    acc[2 * w + 1] = max_xorsign_abs_bf16x2(acc[2 * w + 1], hi)

            # max.xorsign.abs leaves junk in every sign bit.
            raws = [None] * LW
            for k in cutlass.range_constexpr(0, NPc, 1):
                a = acc[k] & Int32(0x7FFF7FFF)
                raws[2 * k] = e8m0_raw_from_bf16(a & Int32(0xFFFF), self._data_limit_exponent)
                raws[2 * k + 1] = e8m0_raw_from_bf16(
                    (a >> Int32(16)) & Int32(0xFFFF), self._data_limit_exponent
                )
            scs = raws
            if cutlass.const_expr(self.scaled_cvt):
                invs = None
            else:
                # The portable down-convert scales by an exact FP32 reciprocal
                # instead of handing an E8M0 byte to the hardware.
                invs = [
                    bits_f32(
                        cutlass.max(
                            (Int32(254) - raws[j]) << Int32(23), Int32(0x400000)
                        )
                    )
                    for j in range(LW)
                ]

            col0 = seg * Int32(SEGW) + Int32(ch * 32 * LW) + lane_idx * Int32(LW)
            for j in cutlass.range_constexpr(0, LW, 1):
                col = col0 + Int32(j)
                off = (
                    sfout
                    + (col // Int32(128)) * Int32(self.SfOutStride)
                    + (col % Int32(32)) * Int32(16)
                    + ((col % Int32(128)) // Int32(32)) * Int32(4)
                    + tb
                )
                out_t = cute.make_tensor(
                    cute.make_ptr(cutlass.Uint8, off, AddressSpace.smem, assumed_align=1),
                    cute.make_layout((1,)),
                )
                out_t[0] = Uint8(raws[j])

            for tt in cutlass.range_constexpr(0, NB, 2):
                out0, out1 = self._requant_token_pair(
                    d[tt], d[tt + 1], scs, invs, NPc, NWc
                )
                self._store_words(base + Int32(tt * W), ldsw, out0, NWc, LW)
                self._store_words(base + Int32((tt + 1) * W), ldsw, out1, NWc, LW)

    def _requant_token_pair(self, d0, d1, scs, invs, NPc, NWc):
        """Requantise one lane's two consecutive tokens.  THE ONLY PLACE THIS
        KERNEL DEPENDS ON THE TARGET ARCHITECTURE.

        ``d0``/``d1`` are lists of NPc Int32, each a token-major BF16x2: register
        ``k`` is hidden columns ``(2k, 2k+1)`` of one token.  ``out0``/``out1``
        come back as NWc packed FP8 words per token, in the byte order
        ``_load_words`` read, so ``_store_words`` can put them straight back.
        ``scs[j]`` is column ``j``'s output E8M0 raw byte and ``invs[j]`` its
        exact FP32 reciprocal; each arm reads only the one it needs.

        Deliberately NOT ``@cute.jit``: it has to inline into the caller's trace.
        A jitted callee would fail to marshal the Python lists and would change
        the emitted IR.

        The transpose belongs inside this function: it is not a layout
        preference but a consequence of ``scaled::n1`` taking a single scale for
        both halves, which forces the two elements to share a hidden column
        while SMEM is token-major.
        """
        QT = self.quant_dtype
        if cutlass.const_expr(self.scaled_cvt):
            # Allocate the rmem tensors AFTER the cvt chain: the allocation
            # order reaches the IR, so moving them changes the emitted code.
            o = [None] * (2 * NPc)
            # token-major -> column-major: o[j] = (col j of t0, t1)
            for k in range(0, NPc, 1):
                lo = Int32(cute.arch.prmt(d0[k], d1[k], Int32(0x5410)))
                hi = Int32(cute.arch.prmt(d0[k], d1[k], Int32(0x7632)))
                o[2 * k] = cvt_scaled_dn_fp8x2(lo, scs[2 * k], QT)
                o[2 * k + 1] = cvt_scaled_dn_fp8x2(hi, scs[2 * k + 1], QT)

            # column-major -> token-major, one 4-byte word per token/w
            out0 = cute.make_rmem_tensor((NWc,), cutlass.Int32)
            out1 = cute.make_rmem_tensor((NWc,), cutlass.Int32)
            for w in range(0, NWc, 1):
                b = 4 * w
                m01 = (o[b] & Int32(0xFFFF)) | (o[b + 1] << Int32(16))
                m23 = (o[b + 2] & Int32(0xFFFF)) | (o[b + 3] << Int32(16))
                out0[w] = Int32(cute.arch.prmt(m01, m23, Int32(0x6420)))
                out1[w] = Int32(cute.arch.prmt(m01, m23, Int32(0x7531)))
        else:
            # Word w is columns 4w..4w+3; d[2w] is (4w, 4w+1) and d[2w+1] is
            # (4w+2, 4w+3), and each helper returns byte 0 = its low column, so
            # the OR below reproduces the byte order _load_words saw.
            out0 = cute.make_rmem_tensor((NWc,), cutlass.Int32)
            out1 = cute.make_rmem_tensor((NWc,), cutlass.Int32)
            for w in range(0, NWc, 1):
                k0 = 2 * w
                k1 = 2 * w + 1
                p00 = cvt_dn_fp8x2_portable(d0[k0], invs[2 * k0], invs[2 * k0 + 1], QT)
                p01 = cvt_dn_fp8x2_portable(d0[k1], invs[2 * k1], invs[2 * k1 + 1], QT)
                p10 = cvt_dn_fp8x2_portable(d1[k0], invs[2 * k0], invs[2 * k0 + 1], QT)
                p11 = cvt_dn_fp8x2_portable(d1[k1], invs[2 * k1], invs[2 * k1 + 1], QT)
                out0[w] = (p00 & Int32(0xFFFF)) | (p01 << Int32(16))
                out1[w] = (p10 & Int32(0xFFFF)) | (p11 << Int32(16))
        return out0, out1

    @cute.jit
    def _load_words(self, byte_addr, atom, NW, C):
        """One LDS of C raw bytes (no FP8 decode -- the scaled cvt does that)."""
        regs = cute.make_rmem_tensor((NW,), cutlass.Int32)
        cute.copy(
            atom,
            cute.make_tensor(
                cute.make_ptr(cutlass.Int32, byte_addr, AddressSpace.smem, assumed_align=C),
                cute.make_layout((NW,)),
            ),
            regs,
        )
        return regs

    @cute.jit
    def _store_words(self, byte_addr, atom, regs, NW, C):
        cute.copy(
            atom,
            regs,
            cute.make_tensor(
                cute.make_ptr(cutlass.Int32, byte_addr, AddressSpace.smem, assumed_align=C),
                cute.make_layout((NW,)),
            ),
        )

    def _smem_bytes_for(self, tile_hid: int, stages: int) -> int:
        """SMEM a (tile_hid, stages) pair would need, in bytes."""
        hid_atoms = tile_hid // self.SfAtomNonK
        return (
            stages * self.TILE_TOK * tile_hid
            + stages * hid_atoms * (self.SfAtomBytes + self.sf_in_pad)
            + hid_atoms * (self.SfAtomBytes + self.sf_out_pad)
            + 3 * (self.num_experts + 1) * 4
            + 2 * stages * 8
            + 256
        )

    @cute.jit
    def _src_scale_raw(self, byte_addr):
        """Raw source E8M0 byte, forced to 0..255.

        The mask is load-bearing: the pointer type says unsigned, but the
        emitted load sign-extends, and the caller packs this into both halves
        of a word with `raw | (raw << 8)`. Without the mask a scale byte >=
        0x80 poisons the E8M0 pair. Removing it took the mega suite from 28/28
        to 0/28.
        """
        return (
            Int32(
                cute.make_tensor(
                    cute.make_ptr(cutlass.Uint8, byte_addr, AddressSpace.smem, assumed_align=1),
                    cute.make_layout((1,)),
                )[0]
            )
            & Int32(0xFF)
        )
