# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import inspect
from typing import Type

import cutlass
from cutlass.base_dsl.typing import Pointer
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm
import cutlass.cute as cute
from cutlass._mlir.dialects import arith, vector
from cutlass._mlir.dialects import nvvm as nvvm_ops
from cutlass._mlir.extras import types as T_
from cutlass._mlir import ir as _ir

from .regtile import RegTile, vec_concat

# nvidia-cutlass-dsl 4.7.0a0 nightlies from 2026-07-27 (6986e65) on regenerated
# the nvvm bindings to take an explicit result type -- *_packed_f32x2(res,
# src_a, src_b) -- but the cutlass.experimental.primitives wrappers still call
# the two-positional form, so going through them raises "missing 1 required
# positional argument: 'src_b'".  Call the dialect ops directly and adapt to
# whichever binding generation is installed; older ones infer the result type.
_PACKED_RES_FIRST = "res" in inspect.signature(nvvm_ops.mul_packed_f32x2).parameters


def _packed_f32x2(op, vec_a, vec_b):
    """``op(vec_a, vec_b)`` for nvvm.{mul,add}_packed_f32x2 on f32x2 IR values."""
    if _PACKED_RES_FIRST:
        return op(vec_a.type, vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)
    return op(vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)


def tmem_load_max_reduction(tmem_addr, num: cutlass.Constexpr = 64):
    """tcgen05.ld.red.sync.aligned.32x32b.x{num}.f32.max — fused TMEM
    load + HW row-max reduction (LDTM.STAT).  ``num`` is the elements-per-
    thread count (= TILE_N/2 for the current dual-MMA softmax path).
    """
    _data_ops = ", ".join("{$w%d}" % i for i in range(num))
    _ptx = ("tcgen05.ld.red.sync.aligned.32x32b.x%d.f32.max " "{" + _data_ops + "}, {$w%d}, [{$r0}];") % (num, num)
    outs = inline_ptx(
        _ptx,
        write_only_types=[cutlass.Int32] * (num + 1),
        read_only_args=[tmem_addr],
    )
    return cutlass.Vector.from_elements(tuple(outs), cutlass.Int32)


def row_max_reduction(vec):
    n = int(vec.shape[0])
    elems = [vec[i] for i in range(n)]
    while len(elems) > 1:
        nxt = []
        for i in range(0, len(elems), 3):
            grp = elems[i : i + 3]
            acc = grp[0]
            for g in grp[1:]:
                acc = cute.math.max(acc, g, ftz=True)
            nxt.append(acc)
        elems = nxt
    return elems[0]


def row_reduction_pair(vec):
    n = int(vec.shape[0])
    assert n % 2 == 0, f"row_reduction_pair: N={n} must be even"
    half = n // 2
    paired_ty = _ir.VectorType.get([half, 2], T_.f32())
    paired = vector.shape_cast(paired_ty, vec.ir_value())

    acc = vector.extract(paired, dynamic_position=[], static_position=[0])
    for i in range(1, half):
        pair = vector.extract(paired, dynamic_position=[], static_position=[i])
        acc = arith.addf(acc, pair)
    return cutlass.Vector(acc, dtype=cutlass.Float32)


def tmem_load_max_reduction_x64(tmem_addr):
    return tmem_load_max_reduction(tmem_addr, num=64)


def row_max_reduction_64(vec64):
    return row_max_reduction(vec64)


def row_reduction_pair_64(vec64):
    return row_reduction_pair(vec64)


def tmem_load_tile(tmem_addr, num_elems: int, ld_num: int = 64) -> RegTile:
    assert num_elems % ld_num == 0, f"tmem_load_tile: num_elems={num_elems} must be a multiple of " f"ld_num={ld_num}"
    chunks = [
        nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(tmem_addr + cutlass.Int32(i * ld_num), cutlass.Float32),
            num=ld_num,
        )
        for i in range(num_elems // ld_num)
    ]
    return RegTile(vec_concat(chunks))


def tmem_load_max_reduction_tile(tmem_addr, num_elems: int):
    assert num_elems % 64 == 0, f"tmem_load_max_reduction_tile: num_elems={num_elems} must be a multiple of 64"
    raw_results = [tmem_load_max_reduction(tmem_addr + cutlass.Int32(i * 64), num=64) for i in range(num_elems // 64)]
    data_chunks = [cutlass.Vector.from_elements(tuple(res[:64]), cutlass.Int32).bitcast(cutlass.Float32) for res in raw_results]
    max_scalars = [cutlass.Vector.from_elements((res[64],), cutlass.Int32).bitcast(cutlass.Float32)[0] for res in raw_results]
    final_max = max_scalars[0]
    for m in max_scalars[1:]:
        final_max = cute.math.max(final_max, m)
    return RegTile(vec_concat(data_chunks)), final_max


@cute.jit
def fp32_to_fp16(lo, hi, *, dtype=cutlass.Float16):
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"fp32_to_fp16: dtype must be Float16 or BFloat16, got {dtype}")
    tag = "f16" if cutlass.const_expr(dtype == cutlass.Float16) else "bf16"
    return inline_ptx(
        f"cvt.rn.{tag}x2.f32 $0, $2, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


def fp32_to_fp8_pack(values, *, dtype: Type[cutlass.Numeric]):
    assert len(values) == 16, f"fp32_to_fp8_pack: expected 16 input values, got {len(values)}"
    if dtype == cutlass.Float8E4M3FN:
        dtype_tag = "e4m3"
    elif dtype == cutlass.Float8E5M2:
        dtype_tag = "e5m2"
    else:
        raise TypeError(f"fp32_to_fp8_pack: dtype must be Float8E4M3FN or Float8E5M2, got {dtype}")

    u0, u1, u2, u3 = inline_ptx(
        "{ .reg .b16 lo, hi;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $5,  $4;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $7,  $6;\n"
        "mov.b32 $0, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $9,  $8;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $11, $10;\n"
        "mov.b32 $1, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $13, $12;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $15, $14;\n"
        "mov.b32 $2, {lo, hi};\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 lo, $17, $16;\n"
        f"cvt.rn.satfinite.{dtype_tag}x2.f32 hi, $19, $18;\n"
        "mov.b32 $3, {lo, hi}; }",
        write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        read_only_args=list(values),
    )
    return cutlass.Vector.from_elements((u0, u1, u2, u3), cutlass.Int32)


# ---------------------------------------------------------------------------
# Rubin's FUSED descale-and-pack: cvt.rn.satfinite.scaled::n1::ue8m0.{e4m3,e5m2}x2.f32
# (sm_107a only) -- the block-scale quantizers' FMUL-free pack.  Semantics MEASURED on
# Rubin (cc 10.7, 204 SMs), 2026-09-29, over 2,085,583 (a, b, sf) triples incl. the edge
# grid (records retained internally):
#
#     byte(x, e) = fp8_rn_satfinite( ftz(x) * 2^(127 - e) )     e = the ue8m0 byte, x fp32
#
# i.e. THE BYTE IS A DESCALE EXPONENT: pass the block's E8M0 *scale* exponent ``e`` (the
# byte the dequant multiplies by, ``e8m0_from_amax(...)[1]`` / an ``e8m0_pair`` byte) AS IS
# and the hardware divides by 2^(e-127) -- exactly ``x * e8m0_rcp(e)`` + ``fp32_to_fp8_pack``
# minus the multiply.  Byte order is the unscaled instruction's (first source operand ->
# HIGH byte), so ``cvt lo, $5, $4`` keeps element 0 in the low byte.  Edge behaviour: +-inf
# and overflow -> +-max (never inf / NaN), NaN -> 0x7f, e = 255 (the e8m0 NaN) -> 0x7f in
# BOTH bytes whatever the inputs, e = 0 is 2^+127, e = 254 is 2^-127, and an fp32-SUBNORMAL
# input (0 < |x| < 2^-126) is FLUSHED to +-0 before scaling (the one divergence from the
# FMUL arm, unobservable for a scale derived from the block's own amax on bf16 / fp16 data).
# SASS per 16 values (MEASURED, the probe kernel of test_tile_dsl_fp8_pack_scaled.py): 8
# F2FP.SATFINITE.<FMT>.F32.PACK_AB_MERGE_C.SCALE_BY_C + 4 PRMT that merge the .b8 into the
# F2FP's C operand -- the ``cvt.u8.u32`` extraction folds into them (0 SHF in the kernel;
# the standalone instruction probe read 1 SHF.L.U32) -- replace 16 FMUL + 8 unscaled F2FP.
#
# THE .b8 SCALE OPERAND HAS A PROVENANCE RULE (MEASURED 2026-09-30, libnvptxcompiler 13.4.46 =
# the DSL's, and a second standalone ptxas build, target sm_107a):
#   * a per-lane DATA-derived register assembles: a GMEM / SMEM load (an SF atom the kernel
#     itself filled, one LDS), ``e8m0_from_amax(abs_max_tree(...))[1]``,
#     ``e8m0_from_amax(warp_abs_max_f32(x))[1]`` (a CREDUX result);
#   * a compile-time constant assembles ONLY as the IMMEDIATE of the ``cvt.u8.u32``, and only
#     in the 16-PACK shape (``cvt.u8.u32 sf, 119;`` feeding the eight lo/hi-merged cvts -> 8
#     SCALE_BY_C + 4 PRMT + 4 MOV, 0 FMUL): ptxas places the byte with a PRMT there, but for a
#     LONE ``x2`` cvt (the pair twin, alone or as two merged cvts) it reaches for an
#     IMAD.SHL of two immediates and ICEs -- so fp32_to_fp8_pack_scaled spells a constant as
#     the immediate and fp32_to_fp8x2_scaled routes a constant through the FMUL arm.  The
#     16-pack form is pinned (SASS + device) on this toolchain; should a kernel context ever
#     ICE on it, feed that call a per-lane LOAD of the byte (an SF atom the kernel fills --
#     the robust provenance) or fused=False;
#   * a register ptxas can PROVE constant (``opaque_i32(cutlass.Int32(119))``, ``tid & 0 | 119``,
#     an ``and 0xff`` / ``mov.b32 {b0..b3}`` / ``cvt.u16`` / ``shl+shr`` / ``shfl.sync.idx`` of one)
#     or a KERNEL PARAMETER (``ld.param``, warp-uniform) ICEs ptxas -- "(C7907) Missing Mercury
#     ISA version for instruction MAD::(unidentified subop) on target SM 107"; an immediate in
#     the SCALED cvt's own operand list is "Arguments mismatch".  The helper cannot see a traced
#     value's provenance, so the kernel-parameter case is the CALLER's contract, tripwired by
#     ``test_kernel_parameter_scale_byte_still_ices_ptxas`` (when it passes, relax this rule).
# ---------------------------------------------------------------------------

# The only target the scaled cvt assembles for.  MEASURED 2026-09-29: libnvptxcompiler 13.4.46 (the DSL's) and a second
# standalone ptxas accept it for sm_107a and reject it for sm_100a / 103a / 110a / 120a ("Illegal modifier '.scaled::n1::ue8m0'");
# sm_107f is UNVERIFIED and therefore absent.  Read by the trace-time backstop below -- never a device query in tile_dsl.
SCALED_FP8_CVT_ARCHS = ("sm_107a",)


def _fp8_tag(dtype: Type[cutlass.Numeric], who: str) -> str:
    if dtype == cutlass.Float8E4M3FN:
        return "e4m3"
    if dtype == cutlass.Float8E5M2:
        return "e5m2"
    raise TypeError(f"{who}: dtype must be Float8E4M3FN or Float8E5M2, got {dtype}")


def _require_scaled_fp8_cvt_target(who: str) -> None:
    """Trace-time backstop of the ``fused=True`` arm.

    ``fused`` is the API layer's decision -- ``compute_capability(dev) == (10, 7)`` folded into a ``TemplateParams`` flag
    the consuming kernel reads as a trace-time constant (the ``api_dsl._EXP2_FMA_SPLIT_CC`` idiom); ``tile_dsl`` never
    queries the device.  What it CAN see is the trace's target (``--gpu-arch`` / ``CUTE_DSL_ARCH`` / the detected part), so
    a flag set for the wrong part fails HERE, naming the arch and the helper, instead of in ptxas with
    ``Illegal modifier '.scaled::n1::ue8m0'``.  Needs cutlass-dsl >= 4.8.0 (the first wheel whose ``Arch`` knows
    ``sm_107a``, AGENTS.md Rule 7): on the 4.7.0 floor no trace can target Rubin, so the arm is unreachable there."""
    try:
        from cutlass.base_dsl.dsl import BaseDSL

        arch_name = BaseDSL._get_dsl().get_arch_enum().name
    except Exception as exc:  # the accessor is DSL-private: if a wheel moves it, fail loud here, never guess the target
        raise RuntimeError(
            f"{who}: cannot read the trace's target arch from this cutlass-dsl ({exc!r}); fused=True assembles for "
            f"{', '.join(SCALED_FP8_CVT_ARCHS)} only -- pass fused=False (the portable FMUL arm) or update this backstop."
        ) from exc
    if arch_name not in SCALED_FP8_CVT_ARCHS:
        raise ValueError(
            f"{who}: fused=True emits cvt.rn.satfinite.scaled::n1::ue8m0, which assembles for {', '.join(SCALED_FP8_CVT_ARCHS)} only; "
            f"this trace targets {arch_name}.  `fused` is the API layer's fact (compute_capability(dev) == (10, 7) -> a TemplateParams "
            f"flag); pass fused=False for the portable FMUL arm."
        )


_E8M0_RCP_MAX_E = 253  # e8m0_rcp's contract: 2^(127 - e) is an fp32 NORMAL for e <= 253


def _constant_scale_byte(sf_byte, who: str):
    """The Python int behind a compile-time-constant ``sf_byte`` -- a plain ``int`` or a ``cutlass.Int32`` still holding its
    Python value (``cutlass.Int32(119)`` before a traced op touches it) -- or ``None`` for a traced register."""
    if isinstance(sf_byte, bool):
        raise TypeError(f"{who}: the scale byte is an E8M0 exponent, not a bool")
    if isinstance(sf_byte, int):
        return sf_byte
    v = getattr(sf_byte, "value", None)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _scale_byte_operand(sf_byte, *, fused: bool, who: str, const_takes_fmul: bool = False):
    """``(imm, reg)`` -- exactly one is set.  A compile-time constant (Python int / constant ``Int32``) becomes ``imm``:
    the fused 16-pack spells it as the ``cvt.u8.u32`` IMMEDIATE (the ONE constant form ptxas assembles -- a
    register-materialized constant such as ``opaque_i32(...)`` ICEs it, see the header) and the FMUL arm folds ``2^(127 - e)``
    into one opaque register (``const_takes_fmul``: the caller routes a constant through the FMUL arm even when ``fused`` --
    the pair twin -- so the ``e <= 253`` contract of :func:`e8m0_rcp` applies).  A traced value stays ``reg`` and MUST be
    per-lane data (a load, an ``e8m0_from_amax`` byte): a kernel parameter ICEs ptxas and the helper cannot tell it from data
    (header; the caller's contract)."""
    const = _constant_scale_byte(sf_byte, who)
    if const is None:
        return None, sf_byte
    if not 0 <= const <= 255:
        raise ValueError(f"{who}: a constant scale byte must be in [0, 255], got {const}")
    if const > _E8M0_RCP_MAX_E and (not fused or const_takes_fmul):
        raise ValueError(f"{who}: a constant byte descales through the FMUL arm (e8m0_rcp), valid for e <= {_E8M0_RCP_MAX_E}; got {const}")
    return const, None


def fp32_to_fp8_pack_scaled(values, sf_byte, *, dtype: Type[cutlass.Numeric], fused: bool):
    """16 fp32 -> 16 fp8 bytes in four Int32 words (element i in byte i), every element DEscaled by ONE ue8m0 byte:

        byte_i = fp8_rn_satfinite(values[i] * 2^(127 - sf_byte))

    ``sf_byte`` -- the E8M0 exponent in bits [0, 8) (higher bits are dropped by the ``cvt.u8.u32``), one of:

    * a traced ``cutlass.Int32`` that is PER-LANE DATA: a load, or ``e8m0_from_amax(amax)[1]`` / an ``e8m0_pair`` byte
      AS IT IS (the byte is a DEscale exponent).  NOT a kernel parameter and not a register the assembler can prove
      constant -- ptxas ICEs on those (C7907, the provenance rule in the header) and the helper cannot tell them from
      data: the caller's contract, tripwired by ``test_tile_dsl_fp8_pack_scaled.py``;
    * a Python int (cuDNN's fixed P scale: byte 119 = ``x * 2^8``) or a ``cutlass.Int32(<int>)`` still holding it: the
      fused arm spells it as the ``cvt.u8.u32`` IMMEDIATE (the one constant form that assembles, and only in this 16-pack
      shape: 8 SCALE_BY_C + 4 PRMT + 4 MOV per 16 values, 0 FMUL -- header), the FMUL arm folds ``2^(127 - e)`` into one
      opaque register (``e <= 253``).  A per-lane LOAD of the same byte (an SF atom the kernel fills) is the robust
      spelling if a kernel context ever ICEs on the immediate.

    ``fused=True`` (cc 10.7 ONLY: the instruction does not assemble for any other target, and the trace-time backstop
    :func:`_require_scaled_fp8_cvt_target` refuses the arm elsewhere; cutlass-dsl >= 4.8.0, the first wheel that knows
    ``sm_107a`` -- AGENTS.md Rule 7) emits 8 scaled cvts and NO multiply.  ``fused=False``
    is the portable spelling, bit-identical for NORMAL fp32 inputs and ``sf_byte <= 253``: one FMUL by :func:`e8m0_rcp`
    per element + :func:`fp32_to_fp8_pack`.  The two arms differ (both MEASURED on Rubin) only for (1) an
    fp32-SUBNORMAL input, which the fused op flushes to +-0 and the FMUL arm scales exactly, and (2) ``sf_byte`` in
    {254, 255}, which the fused arm handles natively (2^-127 / NaN bytes) while :func:`e8m0_rcp` is only valid for
    ``e <= 253`` -- neither occurs for a block scale derived from a finite amax (``e <= 247`` on bf16 data) applied to the
    data that produced it.  The caller decides ``fused`` at the API layer from ``compute_capability(dev) == (10, 7)``.
    """
    assert len(values) == 16, f"fp32_to_fp8_pack_scaled: expected 16 input values, got {len(values)}"
    tag = _fp8_tag(dtype, "fp32_to_fp8_pack_scaled")
    imm, reg = _scale_byte_operand(sf_byte, fused=fused, who="fp32_to_fp8_pack_scaled")
    if not fused:
        # 2^(127 - e) = bits((254 - e) << 23) as a register: a folded float immediate reaching the pack's asm would be the
        # s7 ICE, so a constant e goes through opaque_f32_bits (ptxas re-folds the MOV into the FMULs), a traced e through
        # e8m0_rcp.
        rcp = opaque_f32_bits((254 - imm) << 23) if imm is not None else e8m0_rcp(reg)
        return fp32_to_fp8_pack([v * rcp for v in values], dtype=dtype)
    _require_scaled_fp8_cvt_target("fp32_to_fp8_pack_scaled")
    # $0..$3 the four words, $4..$19 the values (element 2j in the LOW byte of pair j: the first source operand is the
    # high byte), $20 the scale word -- or the constant spelled as the cvt.u8.u32 IMMEDIATE (header: the one constant form
    # ptxas assembles).  ONE .b8 register serves all eight cvts.
    sf_src = str(imm) if imm is not None else "$20"
    u0, u1, u2, u3 = inline_ptx(
        "{ .reg .b16 lo, hi; .reg .b8 sf;\n"
        f"cvt.u8.u32 sf, {sf_src};\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 lo, $5,  $4,  sf;\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 hi, $7,  $6,  sf;\n"
        "mov.b32 $0, {lo, hi};\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 lo, $9,  $8,  sf;\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 hi, $11, $10, sf;\n"
        "mov.b32 $1, {lo, hi};\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 lo, $13, $12, sf;\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 hi, $15, $14, sf;\n"
        "mov.b32 $2, {lo, hi};\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 lo, $17, $16, sf;\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 hi, $19, $18, sf;\n"
        "mov.b32 $3, {lo, hi}; }",
        write_only_types=[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        read_only_args=list(values) + ([] if imm is not None else [reg]),
    )
    return cutlass.Vector.from_elements((u0, u1, u2, u3), cutlass.Int32)


def fp32_to_fp8x2_scaled(lo, hi, sf_byte, *, dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN, fused: bool):
    """The two-element twin of :func:`fp32_to_fp8x2`: ``low byte = fp8(lo * 2^(127-e))``, ``byte 1 = fp8(hi * 2^(127-e))``
    as a ``Uint16`` -- for the per-pair pack loops that assemble MMA operands with :func:`pack_fp8x2_pairs`.  Same contract,
    arms and backstop as :func:`fp32_to_fp8_pack_scaled`, with ONE exception: a compile-time CONSTANT byte takes the FMUL
    arm even under ``fused=True`` -- a lone scaled cvt with an immediate scale ICEs ptxas (C7907, MEASURED; the header's
    provenance rule), and the two FMULs by the folded power of two are bit-identical for normal inputs."""
    tag = _fp8_tag(dtype, "fp32_to_fp8x2_scaled")
    imm, reg = _scale_byte_operand(sf_byte, fused=fused, who="fp32_to_fp8x2_scaled", const_takes_fmul=True)
    if not fused or imm is not None:
        rcp = opaque_f32_bits((254 - imm) << 23) if imm is not None else e8m0_rcp(reg)
        return fp32_to_fp8x2(lo * rcp, hi * rcp, dtype=dtype)
    _require_scaled_fp8_cvt_target("fp32_to_fp8x2_scaled")
    return inline_ptx(
        "{ .reg .b8 sf; cvt.u8.u32 sf, $3;\n" + f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 $0, $2, $1, sf; " + "}",
        write_only_types=[cutlass.Uint16],
        read_only_args=[lo, hi, reg],
    )


def fp32_to_fp8x4_scaled_pairs(values, sf01, sf23, *, dtype: Type[cutlass.Numeric], fused: bool):
    """4 fp32 -> 4 fp8 bytes in ONE Int32 word (element i in byte i), EVERY element DEscaled by its OWN E8M0 byte:

        byte_0 = fp8(values[0] * 2^(127 - sf01.byte0))   byte_1 = fp8(values[1] * 2^(127 - sf01.byte1))
        byte_2 = fp8(values[2] * 2^(127 - sf23.byte0))   byte_3 = fp8(values[3] * 2^(127 - sf23.byte1))

    The per-COLUMN twin of :func:`fp32_to_fp8_pack_scaled` -- one exponent per ELEMENT instead of one per 16 -- for a lane that
    holds one ROW of a tile whose scale blocks run along the OTHER axis (the MXFP8 backward's ``ds_dq``: lane = kv row, the 32-kv
    block scale is per q column, so each of a lane's 64 values carries a different byte).  ``sf01`` / ``sf23`` are PAIR words as
    :func:`e8m0_pair` / :func:`e8m0_pair_u` lay them (byte 0 = the even element's exponent, byte 1 = the odd one's, bits 16..31
    ignored); they MUST be per-lane DATA to ptxas (the header's provenance rule -- a word built from CREDUX / ``abs_max_tree``
    amaxes assembles; a kernel parameter or a provable constant ICEs ptxas C7907).  What sm_107a ptxas makes of it (MEASURED
    2026-10-01, standalone sm_107a SASS sweeps of the dq tail): the SASS form ``F2FP...PACK_AB_MERGE_C.SCALE_BY_C Rd, Ra, Rb, Rc`` reads the scale from
    byte 3 of ``Rc`` and returns ``(Rc << 16) | pack16``, so ptxas chains the two cvts of a pair through ``Rc`` (the second's merge is
    free) and spends ~3 ops per pair positioning the bytes (a mask, a shift, a ``PRMT``) -- the same count whether the bytes arrive
    as this 32-bit word, as a 16-bit ``mov.b16`` unpack, as a ``mov.b32`` four-byte unpack, pre-positioned at bits 24..31, or with
    the even element first; a 4-byte word per four columns costs one extra extraction per word.  ~2.5 instructions per element all
    in (cvt + positioning + the final ``prmt``), zero fp32 scale builds, zero ``FMUL``.

    ``fused=True`` (cc 10.7 ONLY, :func:`_require_scaled_fp8_cvt_target`): each element takes ONE lone
    ``cvt.rn.satfinite.scaled::n1::ue8m0`` with BOTH operands the element (the pair twin's shape, so its two result bytes are the
    same byte) and the four low bytes are merged by one ``prmt`` -- no multiply, no fp32 scale build.  ``fused=False`` is the
    portable spelling, bit-identical for NORMAL fp32 inputs and bytes ``<= 253``: one FMUL by :func:`e8m0_rcp` of the element's
    byte + the plain pair cvt (:func:`fp32_to_fp8x2`); the two arms diverge exactly as :func:`fp32_to_fp8_pack_scaled`'s do (an
    fp32-SUBNORMAL input is flushed by the fused op; a byte in {254, 255} is outside :func:`e8m0_rcp`'s contract).  Element order
    is the pack's: element 0 in the low byte."""
    if len(values) != 4:
        raise ValueError(f"fp32_to_fp8x4_scaled_pairs: expected 4 input values (two per pair word), got {len(values)}")
    tag = _fp8_tag(dtype, "fp32_to_fp8x4_scaled_pairs")
    if not fused:
        pairs = []
        for i, sf in ((0, sf01), (2, sf23)):
            r_lo = e8m0_rcp(sf & cutlass.Int32(0xFF))
            r_hi = e8m0_rcp((sf >> cutlass.Int32(8)) & cutlass.Int32(0xFF))
            pairs.append(fp32_to_fp8x2(values[i] * r_lo, values[i + 1] * r_hi, dtype=dtype))
        return pack_u16x2(pairs[0], pairs[1])
    _require_scaled_fp8_cvt_target("fp32_to_fp8x4_scaled_pairs")
    # $0 the word, $1..$4 the values, $5 / $6 the pair words.  Both operands of a cvt are the element, so byte 0 of each .b16
    # result is the element's byte; prmt 0x6420 gathers {t0.b0, t0.b2, t1.b0, t1.b2} = the four low bytes in element order.
    return inline_ptx(
        "{ .reg .b8 s0, s1, s2, s3; .reg .b32 u1, u3, t0, t1; .reg .b16 h0, h1, h2, h3;\n"
        "cvt.u8.u32 s0, $5;\n"
        "shr.u32 u1, $5, 8;\n"
        "cvt.u8.u32 s1, u1;\n"
        "cvt.u8.u32 s2, $6;\n"
        "shr.u32 u3, $6, 8;\n"
        "cvt.u8.u32 s3, u3;\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 h0, $1, $1, s0;\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 h1, $2, $2, s1;\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 h2, $3, $3, s2;\n"
        f"cvt.rn.satfinite.scaled::n1::ue8m0.{tag}x2.f32 h3, $4, $4, s3;\n"
        "mov.b32 t0, {h0, h1};\n"
        "mov.b32 t1, {h2, h3};\n"
        "prmt.b32 $0, t0, t1, 0x6420; }",
        write_only_types=[cutlass.Int32],
        read_only_args=list(values) + [sf01, sf23],
    )


def fp32_to_e2m1_pack(values):
    """Pack 16 fp32 into 8 E2M1 bytes (two Int32 words), element i in nibble i.

    ``cvt.rn.satfinite.e2m1x2.f32 d, a, b`` puts ``a`` in the upper nibble, so
    the pair (values[2j+1], values[2j]) lands as byte j.
    """
    assert len(values) == 16, f"fp32_to_e2m1_pack: expected 16 input values, got {len(values)}"
    w0, w1 = inline_ptx(
        "{ .reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, $3,  $2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, $5,  $4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, $7,  $6;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, $9,  $8;\n"
        "mov.b32 $0, {b0, b1, b2, b3};\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, $11, $10;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, $13, $12;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, $15, $14;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, $17, $16;\n"
        "mov.b32 $1, {b0, b1, b2, b3}; }",
        write_only_types=[cutlass.Int32, cutlass.Int32],
        read_only_args=list(values),
    )
    return cutlass.Vector.from_elements((w0, w1), cutlass.Int32)


@cute.jit
def fp32_to_e2m1x2(lo: cutlass.Float32, hi: cutlass.Float32) -> cutlass.Int32:
    """Two fp32 -> one E2M1 pair byte (lo in the low nibble), returned zero-extended in an Int32."""
    return inline_ptx(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $2, $1; cvt.u32.u8 $0, b; }",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


@cute.jit
def e4m3_scale_rcp(sf: cutlass.Float32):
    """Round a positive fp32 scale to E4M3 and return ``(byte, 1/decoded)``.

    The reciprocal is of the DECODED (rounded) scale so quantized data and the
    stored scale stay self-consistent; a zero scale (all-zero block) yields 0.
    """
    byte_i32, dec = inline_ptx(
        "{ .reg .b16 s16, h2lo; .reg .b32 h2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 s16, $2, $2;\n"
        "cvt.u32.u16 $0, s16;\n"
        "and.b32 $0, $0, 0xFF;\n"
        "cvt.rn.f16x2.e4m3x2 h2, s16;\n"
        "mov.b32 {h2lo, s16}, h2;\n"
        "cvt.f32.f16 $1, h2lo; }",
        write_only_types=[cutlass.Int32, cutlass.Float32],
        read_only_args=[sf],
    )
    inv = cutlass.Float32(1.0) / dec
    if dec == cutlass.Float32(0.0):
        inv = cutlass.Float32(0.0)
    return byte_i32, inv


@cute.jit
def amax_to_ue8m0_rp(amax: cutlass.Float32):
    """Encode ``amax / 448`` as a UE8M0 exponent byte (round up) and return ``(byte, 2^(127-e))``.

    Same arithmetic as the MXFP8 bprop kernels' ``cvt_amax_to_e8m0_rp``: 448 is
    ``1.75 * 2**8``, so the rounded-up scale exponent is the amax exponent minus
    eight plus one when the significand exceeds 1.75. Byte 0 (amax == 0) maps to
    an inverse of 0 so an all-zero block quantizes to zeros.
    """
    amax_bits = amax.bitcast(cutlass.Uint32)
    exponent = (amax_bits >> 23) & cutlass.Uint32(0xFF)
    mantissa = amax_bits & cutlass.Uint32(0x7FFFFF)
    scale_exp = cutlass.Int32(exponent) - cutlass.Int32(8)
    if mantissa > cutlass.Uint32(0x600000):
        scale_exp = scale_exp + cutlass.Int32(1)
    if exponent == cutlass.Uint32(0xFF):
        scale_exp = cutlass.Int32(254)
    if scale_exp < cutlass.Int32(0):
        scale_exp = cutlass.Int32(0)
    if scale_exp > cutlass.Int32(254):
        scale_exp = cutlass.Int32(254)
    inv = ((cutlass.Uint32(254) - cutlass.Uint32(scale_exp)) << 23).bitcast(cutlass.Float32)
    if scale_exp == cutlass.Int32(0):
        inv = cutlass.Float32(0.0)
    if scale_exp == cutlass.Int32(254):
        inv = cutlass.Uint32(1 << 22).bitcast(cutlass.Float32)
    return scale_exp, inv


@cute.jit
def fp32_to_fp8x2(lo: cutlass.Float32, hi: cutlass.Float32, *, dtype: cutlass.Constexpr[Type[cutlass.Numeric]] = cutlass.Float8E4M3FN) -> cutlass.Uint16:
    """Pack two fp32 into fp8 bytes: low byte = fp8(lo), byte 1 = fp8(hi)."""
    if cutlass.const_expr(dtype != cutlass.Float8E4M3FN and dtype != cutlass.Float8E5M2):
        raise TypeError(f"Invalid FP8 dtype: {dtype}")
    cvt_tag = "e4m3x2" if cutlass.const_expr(dtype == cutlass.Float8E4M3FN) else "e5m2x2"
    return cute.arch.inline_ptx(
        "{ .reg .f32 fa, fb; mov.b32 fa, {$r0}; mov.b32 fb, {$r1}; " + f"cvt.rn.satfinite.{cvt_tag}.f32 " + "{$w0}, fa, fb; }",
        write_only_types=[cutlass.Uint16],
        read_only_args=[hi.bitcast(cutlass.Int32), lo.bitcast(cutlass.Int32)],
    )


@cute.jit
def pack_fp8x2_pairs(pair0: cutlass.Uint16, pair1: cutlass.Uint16) -> cutlass.Int32:
    """Two fp8x2 halves into one 32-bit MMA A/B operand (pair0 = low half)."""
    return cute.arch.inline_ptx(
        "mov.b32 $0, {$1, $2};",
        write_only_types=[cutlass.Int32],
        read_only_args=[pair0, pair1],
    )


@cute.jit
def pack_u16x2(lo: cutlass.Uint16, hi: cutlass.Uint16) -> cutlass.Int32:
    """Two 16-bit patterns into one 32-bit word (lo = low half)."""
    return cute.arch.inline_ptx(
        "mov.b32 $0, {$1, $2};",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


def vec_scale_pair(vec, scalar, N):
    assert N % 2 == 0, f"vec_scale_pair: N={N} must be even"
    pair_ty = _ir.VectorType.get([2], T_.f32())
    paired_ty = _ir.VectorType.get([N // 2, 2], T_.f32())
    flat_ty = _ir.VectorType.get([N], T_.f32())

    scalar_pair = vector.broadcast(pair_ty, scalar.ir_value())

    paired_in = vector.shape_cast(paired_ty, vec.ir_value())
    result = paired_in
    for i in range(N // 2):
        pair = vector.extract(paired_in, dynamic_position=[], static_position=[i])
        scaled = _packed_f32x2(nvvm_ops.mul_packed_f32x2, pair, scalar_pair)
        result = vector.insert(
            scaled,
            result,
            dynamic_position=[],
            static_position=[i],
        )
    return cutlass.Vector(vector.shape_cast(flat_ty, result), dtype=cutlass.Float32)


@cutlass.cute.jit
def f16x2_to_f32(word, *, dtype=cutlass.Float16):
    """Unpack one Int32 (= 2 packed halves) into ``(lo_f32, hi_f32)`` Float32.

    The inverse of :func:`fp32_to_fp16`.  bf16 IS the top 16 bits of an fp32,
    so f32 = bf16 << 16 (bit move, no PRMT storm); masks stay 32-bit (a Python
    ``0xFFFF0000`` promotes to i64 -> mov.b32 mismatch).  fp16 needs the real
    ``cvt.f32.f16`` converts.
    """
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"f16x2_to_f32: dtype must be Float16 or BFloat16, got {dtype}")
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        lo, hi = inline_ptx(
            "{ .reg .b16 l, h; mov.b32 {l, h}, $2; mov.b32 $0, {0, l}; mov.b32 $1, {0, h}; }",
            write_only_types=[cutlass.Float32, cutlass.Float32],
            read_only_args=[word],
        )
    else:
        lo, hi = inline_ptx(
            "{ .reg .b16 h0, h1; mov.b32 {h0, h1}, $2; " "cvt.f32.f16 $0, h0; cvt.f32.f16 $1, h1; }",
            write_only_types=[cutlass.Float32, cutlass.Float32],
            read_only_args=[word],
        )
    return lo, hi


@cutlass.cute.jit
def opaque_f32_zero():
    """A 0.0f the optimizer cannot prove constant.

    Use for values that feed packed-asm operands (:func:`fmul2` /
    :func:`ffma2`) and could otherwise fold to a literal: the
    ``nvvm.inline_ptx`` lowering gives constant float operands the ``n``
    immediate constraint, which ICEs libNVVM."""
    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Float32])


@cutlass.cute.jit
def opaque_i32_zero():
    """A packed-zero b32 word the optimizer cannot prove constant.

    Same libNVVM immediate-constraint hazard as :func:`opaque_f32_zero`, for
    the packed 16x2 operands (:func:`sub_f16x2` / :func:`mul_f16x2`)."""
    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Int32])


@cutlass.cute.jit
def opaque_i32(value: cutlass.Int32) -> cutlass.Int32:
    """Identity mov.b32 that pins a per-lane loop invariant in its register."""
    return inline_ptx("mov.b32 $0, $1;", write_only_types=[cutlass.Int32], read_only_args=[value])


@cutlass.cute.jit
def fmul2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 multiply (SM100 FMUL2): ``(a_lo * b_lo, a_hi * b_hi)``.

    ``mul.f32x2`` operates on register pairs; the ``mov.b64`` packs map to
    register-pair allocation and usually fold away in SASS."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; mul.f32x2 pc, pa, pb; mov.b64 {$0, $1}, pc; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi],
    )


@cutlass.cute.jit
def fadd2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 add (SM100 FADD2): ``(a_lo + b_lo, a_hi + b_hi)``.

    Native NVVM op, not inline asm: libNVVM rejects ``add.f32x2`` in asm
    blocks (mul/fma made it in, add did not)."""
    vec_a = cutlass.Vector.from_elements((a_lo, a_hi), cutlass.Float32)
    vec_b = cutlass.Vector.from_elements((b_lo, b_hi), cutlass.Float32)
    res = cutlass.Vector(_packed_f32x2(nvvm_ops.add_packed_f32x2, vec_a.ir_value(), vec_b.ir_value()), dtype=cutlass.Float32)
    return cutlass.Float32(res[0]), cutlass.Float32(res[1])


@cutlass.cute.jit
def ffma2(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi):
    """Packed fp32 fma (SM100 FFMA2): ``(a_lo*b_lo + c_lo, a_hi*b_hi + c_hi)``."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc, pd; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; mov.b64 pc, {$6, $7}; " "fma.rn.f32x2 pd, pa, pb, pc; mov.b64 {$0, $1}, pd; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi],
    )


# exp2 on the FMA pipe: a degree-3 minimax for 2^f on f in [0, 1), evaluated on PACKED fp32 pairs.
# The three coefficients (fp32 bit patterns; c0 = 1.0) are shared with the host model in
# test/python/sdpa/frost/test_tile_dsl_exp2_emul.py, which pins the error bound stated below.
EXP2_EMUL_C1_BITS = 0x3F31F519  # 0.6951787
EXP2_EMUL_C2_BITS = 0x3E6906A4  # 0.2275357
EXP2_EMUL_C3_BITS = 0x3D9DF09D  # 0.0771246
# Floor split: add 1.5 * 2^23 (0x4B400000) with round-toward-minus-infinity, subtract it back --
# for |x| < 2^22 the sum is an integer-valued fp32, so ``t - magic == floor(x)`` exactly.
EXP2_EMUL_FLOOR_MAGIC_BITS = 0x4B400000
# 2^-127 is the smallest exponent the insertion below can encode: every x below is clamped there.
EXP2_EMUL_CLAMP_BITS = 0xC2FE0000  # -127.0
# Max relative error of the emulation vs the exact 2^x, MEASURED on the host model over every fp32
# in [-1, 0) (2.0e8 values; the error depends on the fraction only) and 1.2e8 uniform samples of
# [-126, 8]: 8.77e-5 (< 2^-13), at fraction f = 0.1008; every INTEGER x is exact (p(0) == 1.0).
EXP2_EMUL_MAX_REL_ERR = 8.8e-5


@cutlass.cute.jit
def exp2_emul_pair(x_lo, x_hi):
    """``(2**x_lo, 2**x_hi)`` on the FMA pipe -- no ``MUFU.EX2`` -- for a softmax exp burst.

    Why it exists: a softmax kv-iteration on a 128-wide S tile issues 128 ``ex2.approx`` per
    row.  On SM100 the MUFU pipe (4 lanes/clk/SMSP) is then the longest single pipe of the
    softmax warps while the FP32 pipe has slack, so evaluating a compile-time subset of the
    columns here (6 packed FP32/INT instructions per PAIR instead of one MUFU per element)
    shortens the burst.  The cuDNN backend kernel splits its exps the same way.  Four sm100
    prefill kernels consume it, each behind the per-(kind, flavor) cc 10.0 gate of
    ``api_dsl._exp2_fma_split_for`` -- MEASURED vs develop, B200, A/B/A x3, CUPTI medians: the
    d128 MXFP8 kernel (``sm100/prefill_d128_mxfp8.py``, ``_E2E_*``) +7.83 % at B=1 H=24/8 S=16K
    dense (2.217 -> 2.056 ms, together with its Amax_O fold), the d128 per-tensor FP8 kernel
    +4.48 % at S=8K (llama layer, H=64/8), the d192x128 bf16 kernel +1.89 % at S=8K (DSv3 layer,
    H=128/128), and the d128 bf16 / fp16 kernel on its dense-band builds only (``_E2E_DENSE_BAND``:
    +4..5 % dense on top of its spinning ring waits, a loss on the causal builds, llama layer).

    What it computes (bit for bit the backend's split, all ops ``.ftz``):

    1. ``x = max(x, -127)`` -- 2^-127 is the smallest exponent step 4 can insert;
    2. ``n = floor(x)`` through ``add.rm.f32x2`` with 1.5 * 2^23 and a subtract back
       (``EXP2_EMUL_FLOOR_MAGIC_BITS``); ``f = x - n`` in [0, 1), exact;
    3. ``p(f) = 1 + f * (c1 + f * (c2 + f * c3))`` as three ``fma.rn.ftz.f32x2`` -- a degree-3
       minimax for 2^f (``EXP2_EMUL_C{1,2,3}_BITS``);
    4. ``p(f) * 2^n`` by adding ``n << 23`` into p's exponent field (``shl.b32`` + ``add.s32``).

    Accuracy: max relative error **8.77e-5** (< 2^-13; ``EXP2_EMUL_MAX_REL_ERR``), MEASURED on the
    host model over every fp32 in [-1, 0) and 1.2e8 samples of [-126, 8]; integer x are exact.
    That is inside E4M3's 3 mantissa bits, and the fp32 row-sum of a 128-term softmax row moves by
    ~1e-5 relative (the sm100 d128 MXFP8 kernel's LSE reads 6.1e-6 vs fp64 with 32 of 128 columns
    emulated, the cuDNN backend kernel 6.0e-6; bar 1e-4).  For x in [-127, -126) the exponent
    insertion yields a denormal bit pattern (a value below 2^-126, not 2^x) -- irrelevant against
    a softmax row max of 2^4 and above, and the same as the backend.  Inputs must satisfy
    x < 128 - 1 = 127 for the exponent add not to overflow; the softmax feeds
    ``S * scale - (max - P_CAST_LOG2_SCALE) <= RESCALE_THRESHOLD + P_CAST_LOG2_SCALE`` (8).

    Every constant is a PTX immediate INSIDE the asm text: the libNVVM ``n``-constraint ICE of
    :func:`opaque_f32_zero` concerns OPERANDS only."""
    return inline_ptx(
        "{ .reg .f32 f1, f2, f3, f4, f5, f6, f7; .reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10; .reg .s32 r1, r2, r3, r4, r5, r6, r7, r8; "
        f"max.ftz.f32 f1, $2, 0f{EXP2_EMUL_CLAMP_BITS:08X}; max.ftz.f32 f2, $3, 0f{EXP2_EMUL_CLAMP_BITS:08X}; mov.b64 l1, {{f1, f2}}; "
        f"mov.f32 f3, 0f{EXP2_EMUL_FLOOR_MAGIC_BITS:08X}; mov.b64 l2, {{f3, f3}}; "
        "add.rm.ftz.f32x2 l7, l1, l2; sub.rn.ftz.f32x2 l8, l7, l2; sub.rn.ftz.f32x2 l9, l1, l8; "
        f"mov.f32 f7, 0f{EXP2_EMUL_C3_BITS:08X}; mov.b64 l6, {{f7, f7}}; mov.f32 f6, 0f{EXP2_EMUL_C2_BITS:08X}; mov.b64 l5, {{f6, f6}}; "
        f"mov.f32 f5, 0f{EXP2_EMUL_C1_BITS:08X}; mov.b64 l4, {{f5, f5}}; mov.f32 f4, 0f3F800000; mov.b64 l3, {{f4, f4}}; "
        "fma.rn.ftz.f32x2 l10, l9, l6, l5; fma.rn.ftz.f32x2 l10, l10, l9, l4; fma.rn.ftz.f32x2 l10, l10, l9, l3; "
        "mov.b64 {r1, r2}, l7; mov.b64 {r3, r4}, l10; "
        "shl.b32 r5, r1, 23; add.s32 r7, r5, r3; shl.b32 r6, r2, 23; add.s32 r8, r6, r4; "
        "mov.b32 $0, r7; mov.b32 $1, r8; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[x_lo, x_hi],
    )


def exp2_mixed(vec, emul_pairs, N: int):
    """Element-wise ``2**vec`` over an ``N``-wide fp32 Vector, with the pairs listed in
    ``emul_pairs`` (pair p = elements 2p, 2p+1; a trace-time frozenset of ints) evaluated by
    :func:`exp2_emul_pair` on the FMA pipe and every other element by MUFU
    (``cute.math.exp2(fastmath=True)``).  An empty ``emul_pairs`` traces the plain vector exp2,
    so an all-MUFU chunk stays IR-identical to the unsplit kernel.  Trace-time helper (plain
    Python over traced values), like :func:`row_max_reduction`.  Which pairs to emulate is a
    ptxas-schedule property of the CONSUMING kernel -- tune it there, per kernel, by A/B
    (``sm100/prefill_d128_mxfp8.py`` documents its sweep)."""
    assert N % 2 == 0, f"exp2_mixed: N={N} must be even"
    if not emul_pairs:
        return cute.math.exp2(vec, fastmath=True)
    bad = [p for p in emul_pairs if not (0 <= p < N // 2)]
    assert not bad, f"exp2_mixed: pair indices {bad} outside [0, {N // 2})"
    elems = []
    for p in range(N // 2):
        lo = cutlass.Float32(vec[2 * p])
        hi = cutlass.Float32(vec[2 * p + 1])
        if p in emul_pairs:
            lo, hi = exp2_emul_pair(lo, hi)
        else:
            lo = cute.math.exp2(lo, fastmath=True)
            hi = cute.math.exp2(hi, fastmath=True)
        elems.append(lo)
        elems.append(hi)
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


@cutlass.cute.jit
def movmatrix_16b(value: cutlass.Int32) -> cutlass.Int32:
    """Transpose one packed m8n8 b16 register fragment."""
    return inline_ptx(
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[value],
    )


@cutlass.cute.jit
def mul_fp16x2(value: cutlass.Int32, scale: cutlass.Int32) -> cutlass.Int32:
    """Multiply two packed FP16 pairs."""
    return inline_ptx(
        "mul.f16x2 $0, $1, $2;",
        write_only_types=[cutlass.Int32],
        read_only_args=[value, scale],
    )


@cutlass.cute.jit
def sub_f16x2(lhs: cutlass.Int32, rhs: cutlass.Int32, input_dtype: cutlass.Constexpr) -> cutlass.Int32:
    """Subtract two packed pairs using the compile-time input dtype."""
    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return inline_ptx("sub.bf16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs])
    return inline_ptx("sub.f16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs])


@cutlass.cute.jit
def mul_f16x2(lhs: cutlass.Int32, rhs: cutlass.Int32, input_dtype: cutlass.Constexpr) -> cutlass.Int32:
    """Multiply two packed pairs using the compile-time input dtype."""
    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return nvvm.mul_bf16x2(lhs, rhs)
    return mul_fp16x2(lhs, rhs)


@cute.jit
def sigmoid_f16x2(logit_pair: cutlass.Int32, input_dtype: cutlass.Constexpr):
    """Sigmoid of a packed 16-bit logit pair, returned as two fp32 values."""
    logit_vec_f32 = cutlass.Vector.from_elements((logit_pair,), cutlass.Int32).bitcast(input_dtype).to(cutlass.Float32)
    return sigmoid2(logit_vec_f32[0], logit_vec_f32[1])


L2_NORM_EPS = 1.0e-6


@cute.jit
def lane_group_sum(value: cutlass.Float32, lanes: cutlass.Constexpr[int]) -> cutlass.Float32:
    """Sum ``value`` across a power-of-two group of consecutive lanes via
    butterfly shuffles (every lane ends up holding the group total)."""
    offset = lanes // 2
    while offset >= 1:
        value = value + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, value, offset, 31, kind=nvvm.Shfl.BFLY))
        offset = offset // 2
    return value


@cute.jit
def l2norm_inv(sum_sq: cutlass.Float32) -> cutlass.Float32:
    """Inverse L2 norm, ``rsqrt(sum_sq + L2_NORM_EPS)``; a zero row normalizes to zero."""
    return cute.math.rsqrt(sum_sq + cutlass.Float32(L2_NORM_EPS), fastmath=True)


@cute.jit
def sigmoid(x: cutlass.Float32) -> cutlass.Float32:
    """sigmoid(x) via the tanh identity (single MUFU on Blackwell)."""
    half = cutlass.Float32(0.5)
    return cute.math.tanh(x * half, approx=True) * half + half


@cute.jit
def sigmoid2(x_lo, x_hi):
    """``(sigmoid(x_lo), sigmoid(x_hi))`` via the tanh identity, with the
    halving and the scale-bias folded into one FMUL2 and one FFMA2."""
    half = opaque_f32_zero() + cutlass.Float32(0.5)
    scaled_lo, scaled_hi = fmul2(x_lo, x_hi, half, half)
    tanh_lo = cute.math.tanh(scaled_lo, approx=True)
    tanh_hi = cute.math.tanh(scaled_hi, approx=True)
    return ffma2(tanh_lo, tanh_hi, half, half, half, half)


@cute.jit
def softplus(x: cutlass.Float32) -> cutlass.Float32:
    """log(1 + exp(x)) with the linear tail (x > 20 returns x: exp saturates
    fp32 there and log1p(exp(x)) == x to fp32 precision)."""
    result = x
    if x < cutlass.Float32(20.0):
        result = cute.math.log(cutlass.Float32(1.0) + cute.math.exp(x, fastmath=True), fastmath=True)
    return result


@cute.jit
def softplus2(x_lo, x_hi):
    """``(softplus(x_lo), softplus(x_hi))`` with the ``1 + exp`` step packed
    into one FADD2 and the linear tail applied as a select."""
    one = cutlass.Float32(1.0)
    tail = cutlass.Float32(20.0)
    exp_lo = cute.math.exp(x_lo, fastmath=True)
    exp_hi = cute.math.exp(x_hi, fastmath=True)
    sum_lo, sum_hi = fadd2(exp_lo, exp_hi, one, one)
    log_lo = cute.math.log(sum_lo, fastmath=True)
    log_hi = cute.math.log(sum_hi, fastmath=True)
    return (log_lo if x_lo < tail else x_lo), (log_hi if x_hi < tail else x_hi)


@cute.jit
def ex2_f16x2(pair: cutlass.Int32, *, dtype=cutlass.Float16) -> cutlass.Int32:
    """Packed-pair ``exp2`` in ONE MUFU op: ``ex2.approx.f16x2`` (fp16) or
    ``ex2.approx.ftz.bf16x2`` (bf16).

    PTX/target contract (ISA 9.4, 9.7.4.10):

    - ``.f16x2``: PTX ISA 7.0, sm_75+. Max relative error 2^-9.9; subnormal
      inputs are supported.
    - ``.ftz.bf16x2``: PTX ISA 7.8, sm_90+. Max relative error 2^-7; ``ftz``
      is mandatory for bf16 — subnormal inputs and results flush to
      sign-preserving zero (so +/-subnormal -> +1.0; -Inf -> +0.0;
      NaN -> NaN).
    """
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"ex2_f16x2: dtype must be Float16 or BFloat16, got {dtype}")
    op = "ex2.approx.f16x2" if cutlass.const_expr(dtype == cutlass.Float16) else "ex2.approx.ftz.bf16x2"
    return inline_ptx(
        f"{op} $0, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[pair],
    )


@cute.jit
def f16x2x2_to_fp8_word(lo_pair: cutlass.Int32, hi_pair: cutlass.Int32, dtype_tag: cutlass.Constexpr, *, dtype=cutlass.Float16) -> cutlass.Int32:
    """Two packed half-pair words (elems 0..3 in order) into one 4-byte FP8 word.

    ``cvt.rn.satfinite.{e4m3,e5m2}x2.{f16x2,bf16x2}`` converts a packed pair
    directly (no f32 round-trip): byte 0 = fp8(lo of lo_pair) ... byte 3 =
    fp8(hi of hi_pair), matching :func:`fp32_to_fp8_pack`'s element order.
    ``satfinite`` semantics: NaN converts to NaN in the destination format;
    |x| > MAX_NORM saturates to sign-preserved MAX_NORM (448 e4m3 / 57344
    e5m2) — never Inf.

    PTX/target contract (ISA 9.4, 9.7.10.24):

    - ``.f16x2`` source: PTX ISA 7.8, sm_90+ (sm_89 from ISA 8.1).
    - ``.bf16x2`` source: PTX ISA 9.1, family-specific targets only
      (sm_100f/sm_110f/sm_120f or higher in family) — a CUDA 13.1+
      toolchain floor.
    """
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"f16x2x2_to_fp8_word: dtype must be Float16 or BFloat16, got {dtype}")
    if cutlass.const_expr(dtype_tag != "e4m3" and dtype_tag != "e5m2"):
        raise ValueError(f"f16x2x2_to_fp8_word: dtype_tag must be 'e4m3' or 'e5m2', got {dtype_tag}")
    src = "f16x2" if cutlass.const_expr(dtype == cutlass.Float16) else "bf16x2"
    dst = "e4m3x2" if cutlass.const_expr(dtype_tag == "e4m3") else "e5m2x2"
    return inline_ptx(
        f"{{ .reg .b16 lo, hi; cvt.rn.satfinite.{dst}.{src} lo, $1; cvt.rn.satfinite.{dst}.{src} hi, $2; mov.b32 $0, {{lo, hi}}; }}",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo_pair, hi_pair],
    )


# ---------------------------------------------------------------------------
# MXFP8 E8M0 block scales (TransformerEngine semantics; the torch oracle is
# test/python/sdpa/mxfp8_quant.py::quantize_blocks).  Ported from the DSv3
# projection kernel's hand-rolled copy (gemm/cutedsl/dense/proj_rope_mxfp8/
# gemm_proj_rope_mxfp8.py:70-107) the moment a second kernel -- the gated
# attention block's MXFP8 quantizer -- needed the same four lines.
# ---------------------------------------------------------------------------

# The fp32 rounding of 1/448 (TE ``Quantized_Limits<E4M3>::max_norm_rcp``).  The
# scale is ``cvt.rp(amax * THIS)`` -- ONE fp32 multiply by exactly these bits, so
# the kernel and the oracle agree on every rounding corner (448 -> 0x7F, 449 -> 0x80).
E8M0_RCP_E4M3_MAX_BITS = 0x3B124925


def opaque_f32_bits(bits: int) -> cutlass.Float32:
    """The fp32 whose bit pattern is ``bits`` as a REGISTER operand the optimizer cannot fold.

    For constants that feed an ``inline_ptx`` operand (a ``cvt`` / ``max.f32`` / ``div.rn.f32``
    right after one FMUL): a folded float immediate reaching an asm operand gets the ``n``
    constraint and ICEs libNVVM (:func:`opaque_f32_zero`).  ptxas re-folds the MOV into the
    consuming FMUL.  Trace-time helper: ``bits`` is a Python int in ``[0, 2**32)``."""
    if not 0 <= int(bits) < (1 << 32):
        raise ValueError(f"opaque_f32_bits: bits must be a 32-bit pattern, got {bits!r}")
    return inline_ptx(f"mov.b32 {{$w0}}, 0x{int(bits):08X};", write_only_types=[cutlass.Float32])


@cute.jit
def opaque_e4m3_max_rcp() -> cutlass.Float32:
    """``fp32(1/448)`` as a REGISTER operand the optimizer cannot fold (:func:`opaque_f32_bits`)."""
    return opaque_f32_bits(E8M0_RCP_E4M3_MAX_BITS)


@cute.jit
def e8m0_rcp(byte: cutlass.Int32) -> cutlass.Float32:
    """Biased E8M0 byte -> the EXACT fp32 dequant reciprocal ``2^(127 - e)`` = ``bits((254 - e) << 23)``.

    Valid for ``e <= 253`` (TE ``exp2f_rcp`` special-cases ``e == 254`` to the fp32
    subnormal 2^-127; an inf/NaN amax is out of contract for the callers here)."""
    return ((cutlass.Int32(254) - byte) << 23).bitcast(cutlass.Float32)


@cute.jit
def e8m0_from_amax(amax: cutlass.Float32, *, inv_max=None):
    """``(rcp, byte)`` for one block: ``byte = cvt.rp.satfinite.ue8m0x2.f32(amax * inv_max)``,
    ``rcp = bits((254 - byte) << 23)`` -- the exact power-of-two the data is multiplied by before the code cast.

    ``inv_max`` is the reciprocal of the code format's max normal AS A REGISTER (:func:`opaque_f32_bits`);
    ``None`` (the default) is :func:`opaque_e4m3_max_rcp` = ``fp32(1/448)`` for MXFP8 e4m3 codes, and
    :func:`opaque_fp4_max_rcp` = ``fp32(1/6)`` selects the MXFP4 e2m1 scale.  The default path emits the
    SAME ops as before the parameter existed.

    Bit-exact with ``mxfp8_quant.quantize_blocks`` for FINITE inputs (max bf16 amax * 1/448 -> ``e <= 247``;
    ``amax == 0 -> 0x00``).  ``e in {254, 255}`` (inf / NaN amax) is OUT OF CONTRACT: the oracle's
    ``exp2_rcp`` special-cases ``e == 254`` and this does not.  ``byte`` is an ``Int32`` in ``[0, 255]``.
    """
    scaled = amax * (opaque_e4m3_max_rcp() if cutlass.const_expr(inv_max is None) else inv_max)
    # ue8m0x2 packs TWO scales (operand a -> upper byte, b -> lower); feeding the
    # same value twice keeps the asm immediate-free, the mask keeps the low one.
    packed = inline_ptx("cvt.rp.satfinite.ue8m0x2.f32 {$w0}, {$r0}, {$r0};", write_only_types=[cutlass.Uint16], read_only_args=[scaled])
    byte = cutlass.Int32(packed) & cutlass.Int32(0xFF)
    return e8m0_rcp(byte), byte


@cute.jit
def e8m0_pair(amax0: cutlass.Float32, amax1: cutlass.Float32, *, inv_max=None):
    """``(rcp0, rcp1, packed)`` for two blocks with ONE ``cvt``: ``byte0 = packed & 0xFF`` (amax0),
    ``byte1 = (packed >> 8) & 0xFF`` (amax1) -- little-endian, so a ``st.b16`` of ``packed`` lays byte0 first.
    Same contract (and the same ``inv_max`` selector) as :func:`e8m0_from_amax`."""
    rcp_max = opaque_e4m3_max_rcp() if cutlass.const_expr(inv_max is None) else inv_max
    scaled0 = amax0 * rcp_max
    scaled1 = amax1 * rcp_max
    # operand a lands in the UPPER byte: pass amax1's scale first so amax0's byte is the low one.
    packed16 = inline_ptx("cvt.rp.satfinite.ue8m0x2.f32 {$w0}, {$r1}, {$r0};", write_only_types=[cutlass.Uint16], read_only_args=[scaled0, scaled1])
    packed = cutlass.Int32(packed16) & cutlass.Int32(0xFFFF)
    return e8m0_rcp(packed & cutlass.Int32(0xFF)), e8m0_rcp((packed >> 8) & cutlass.Int32(0xFF)), packed


def e8m0_pair_u(amax0: cutlass.Float32, amax1: cutlass.Float32, inv_max: cutlass.Float32) -> cutlass.Int32:
    """:func:`e8m0_pair`'s packed bytes for two WARP-UNIFORM amaxes (``redux.sync`` results): ``byte 0 = amax0``'s exponent,
    ``byte 1 = amax1``'s, bits 16..31 zero -- the same rule, op for op (``amax * inv_max`` rounded to nearest, then ONE
    ``cvt.rp.satfinite.ue8m0x2.f32``), spelled as ONE packed multiply for the pair.

    A ``redux.sync`` result lands in a uniform register, and on sm_107a ptxas moves it into the vector file before ANY ALU op
    reads it (``MOV R, UR``, one per amax): a scalar ``FMUL`` by an immediate, by a lane register, a packed ``mul.f32x2``, an LLVM
    ``fmul`` outside inline asm, the ``cvt.rp`` on the uniform value itself -- 13 spellings, every one 64 moves per 64 columns
    (standalone sm_107a SASS sweeps, 2026-10-01).  The move is the floor; what this helper saves is the MULTIPLY count: the two
    moved amaxes form a register PAIR, ``mul.f32x2 {inv, inv} x {amax0, amax1}`` is ONE ``FMUL2`` whose two halves are both
    consumed by the cvt (32 ``FMUL2`` per 64 columns instead of 64 scalar ``FMUL``; a broadcast ``{amax, amax}`` pair is
    scalarised by ptxas and gains nothing).  ``inv_max`` must be a VECTOR register ptxas cannot fold into an immediate
    (:func:`opaque_e4m3_max_rcp_in_lane`, hoisted once per kernel): the packed multiply has no immediate operand form, a foldable
    constant is re-materialised per use.  Bit-identical to :func:`e8m0_pair` (``mul.rn.f32x2`` is two independent RN fp32
    multiplies; operand a of the cvt lands in the UPPER byte, so amax1's scale goes first)."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc; .reg .f32 s0, s1; .reg .b16 lo;\n"
        "mov.b64 pa, {$1, $1};\n"
        "mov.b64 pb, {$2, $3};\n"
        "mul.rn.f32x2 pc, pa, pb;\n"
        "mov.b64 {s0, s1}, pc;\n"
        "cvt.rp.satfinite.ue8m0x2.f32 lo, s1, s0;\n"
        "cvt.u32.u16 $0, lo; }",
        write_only_types=[cutlass.Int32],
        read_only_args=[inv_max, amax0, amax1],
    )


def opaque_f32_bits_in_lane(bits: int, seed: cutlass.Float32) -> cutlass.Float32:
    """The fp32 whose bit pattern is ``bits`` as a VECTOR register ptxas cannot fold into an immediate: ``fma.rn.f32(seed, 0, bits)``
    on a FINITE runtime ``seed`` (IEEE: ``x * 0`` is not provably 0 for a NaN / inf ``x``, so neither compiler folds it).

    :func:`opaque_f32_bits` only hides the constant from LLVM; ptxas re-folds its ``MOV`` into the consumer as an immediate, which
    is right for a scalar ``FMUL R, R, imm`` and impossible for the PACKED multiply of :func:`e8m0_pair_u` (``mul.f32x2`` has no
    immediate operand: a foldable constant is re-materialised per use).  A constant the assembler cannot fold stays in one vector
    register for the whole kernel.  It does NOT remove the ``MOV R, UR`` of a CREDUX amax -- sm_107a ptxas moves a uniform result
    into the vector file before any ALU op reads it, in every spelling tried (standalone sm_107a SASS sweeps, 2026-10-01).  One FFMA
    per call -- hoist it to the kernel prologue."""
    if not 0 <= int(bits) < (1 << 32):
        raise ValueError(f"opaque_f32_bits_in_lane: bits must be a 32-bit pattern, got {bits!r}")
    return inline_ptx(f"fma.rn.f32 $0, $1, 0f00000000, 0f{int(bits):08X};", write_only_types=[cutlass.Float32], read_only_args=[seed])


def opaque_e4m3_max_rcp_in_lane(seed: cutlass.Float32) -> cutlass.Float32:
    """``fp32(1/448)`` as a VECTOR register ptxas cannot fold (:func:`opaque_f32_bits_in_lane`): the ``inv_max`` for a batch of
    warp-uniform amaxes (:func:`e8m0_pair_u` over CREDUX results).  ``seed`` is any finite per-lane fp32 (its lane
    id converted); the value is bit-identical to :func:`opaque_e4m3_max_rcp`."""
    return opaque_f32_bits_in_lane(E8M0_RCP_E4M3_MAX_BITS, seed)


# ---------------------------------------------------------------------------
# FP4 (e2m1) block quantization -- NVFP4 (e2m1 x E4M3 per 16) and MXFP4 (e2m1 x
# E8M0 per 32).  The four primitives a quantizer composes, hardware-rounded where
# the hardware has the op (``cvt.rn.satfinite.e2m1x2.f32``, ``cvt.rn.satfinite.
# e4m3x2.f32``, ``div.rn.f32``), so a torch oracle that rounds the SAME fp32 value
# can be held to ``torch.equal``.  The oracle constants are the SAME bit patterns:
# the scale is ``amax * fp32(1/6)`` -- ONE fp32 multiply by exactly these bits,
# never ``amax / 6`` (one ulp apart, and an e4m3 / e8m0 rounding corner away).
# ---------------------------------------------------------------------------

# fp32(1/6) = 0x3E2AAAAB: the reciprocal of e2m1's max normal (6.0), the fp4 twin
# of ``E8M0_RCP_E4M3_MAX_BITS``.
E2M1_MAX_RCP_BITS = 0x3E2AAAAB

# 2^-9 = 0x3B000000: e4m3's minimum subnormal, the NVFP4 scale floor.  ``fp32(2^-10)``
# rounds to e4m3 ZERO, so without the floor an all-zero (or tiny) block gets scale 0
# and an infinite encode -- the floor keeps every block's scale a nonzero e4m3.
E4M3_MIN_SUBNORMAL_BITS = 0x3B000000

# ``cvt.rn.satfinite.e2m1x2.f32`` converts a PAIR per instruction.
E2M1_PER_CVT = 2


@cute.jit
def opaque_fp4_max_rcp() -> cutlass.Float32:
    """``fp32(1/6)`` as a REGISTER operand the optimizer cannot fold (:func:`opaque_f32_bits`).

    Pass it as ``e8m0_from_amax(amax, inv_max=opaque_fp4_max_rcp())`` for the MXFP4 e8m0 scale;
    :func:`e4m3_scale_from_amax` uses it for the NVFP4 e4m3 scale."""
    return opaque_f32_bits(E2M1_MAX_RCP_BITS)


@cute.jit
def div_rn_f32(a: cutlass.Float32, b: cutlass.Float32) -> cutlass.Float32:
    """PTX ``div.rn.f32`` -- the IEEE correctly-rounded division, as ONE opaque op.

    NVFP4 codes are ``e2m1(x / sf)`` where ``sf`` is an e4m3 value that is NOT a power of two, so a
    reciprocal-multiply (``x * rcp(sf)``: MUFU.RCP + FMUL, or any ``fastmath`` division) can land one
    fp32 ulp off the true quotient and flip an EXACT e2m1 rounding midpoint (``0.5859375 / 0.46875 ==
    1.25`` is a tie between the codes 1.0 and 1.5) -- a whole code, not noise, and the torch oracle
    (``x / sf.float()``, correctly rounded) would disagree.  ptxas lowers ``div.rn.f32`` to the
    MUFU.RCP seed + FFMA Newton fixup + an FCHK range check with a slow-path call, never a bare
    MUFU.RCP; the tile_dsl test pins that in the sm_107a SASS.  Both operands must be traced
    registers (a folded float immediate on an asm operand ICEs libNVVM, :func:`opaque_f32_bits`)."""
    return inline_ptx("div.rn.f32 $0, $1, $2;", write_only_types=[cutlass.Float32], read_only_args=[a, b])


@cute.jit
def e4m3_scale_from_amax(amax: cutlass.Float32):
    """``(sf_f32, byte)`` for one NVFP4 block: ``byte = cvt.rn.satfinite.e4m3x2.f32(max(amax * fp32(1/6), 2^-9))``,
    ``sf_f32`` = that e4m3 value widened EXACTLY to fp32 (e4m3 -> f16 -> f32, both conversions exact).

    The torch twin is ``(amax * fp32(1/6)).clamp_min(2**-9).to(torch.float8_e4m3fn)`` (RN in both, and BOTH
    saturate a scale above 448 -- an amax above 2688 -- to 448 (``satfinite`` here, torch's cast likewise,
    verified to 1e6), so the two agree bitwise there too: do NOT add a special case for large amax).  The ``2^-9`` floor (``E4M3_MIN_SUBNORMAL_BITS``) keeps an
    all-zero block's scale the smallest nonzero e4m3 instead of 0, so ``div_rn_f32(0, sf)`` is 0 and
    not NaN.  The floor is a ``max.f32`` on two REGISTERS (:func:`fmax_f32`); NaN amax is out of
    contract.  ``byte`` is an ``Int32`` in ``[0, 255]``; the block's codes are
    ``fp32_to_fp4_pack([div_rn_f32(x, sf_f32) ...])``."""
    scaled = fmax_f32(amax * opaque_fp4_max_rcp(), opaque_f32_bits(E4M3_MIN_SUBNORMAL_BITS))
    # e4m3x2 packs TWO scales (operand a -> upper byte, b -> lower); the same value twice keeps the asm
    # immediate-free and either half is the byte.  cvt.rn.f16x2.e4m3x2 widens the pair exactly.
    sf_f32, packed = inline_ptx(
        "{ .reg .b16 p, lo, hi; .reg .b32 w; "
        "cvt.rn.satfinite.e4m3x2.f32 p, $2, $2; "
        "cvt.rn.f16x2.e4m3x2 w, p; "
        "mov.b32 {lo, hi}, w; "
        "cvt.f32.f16 $0, lo; "
        "mov.b16 $1, p; }",
        write_only_types=[cutlass.Float32, cutlass.Uint16],
        read_only_args=[scaled],
    )
    byte = cutlass.Int32(packed) & cutlass.Int32(0xFF)
    return sf_f32, byte


def fp32_to_fp4_pack(values):
    """16 fp32 (already divided / scaled into e2m1's range) -> 2 x ``Int32`` of e2m1 codes, 8 per word.

    ``cvt.rn.satfinite.e2m1x2.f32 d, a, b`` rounds to nearest-even on the e2m1 grid
    ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` (sign kept; |x| >= 6 saturates to 6, never inf/NaN; PTX ISA 8.6,
    sm_100a+) and packs ``a`` into the UPPER nibble, ``b`` into the LOWER.  The call passes
    ``(values[2i+1], values[2i])`` so byte ``i`` = ``code(values[2i]) | code(values[2i+1]) << 4`` -- low
    nibble = even element, the ``float4_e2m1fn_x2`` / ``gemm_test_utils.unpack_fp4`` convention -- and
    ``mov.b32 {b0, b1, b2, b3}`` lays byte 0 lowest, so a little-endian store of word 0 then word 1
    writes the 16 codes in element order.  Twin of :func:`fp32_to_fp8_pack`; the byte-order claim is
    what the quantizer's bitwise oracle test holds."""
    if len(values) != 16:
        raise ValueError(f"fp32_to_fp4_pack: expected 16 input values, got {len(values)}")
    cvts = []
    for w in range(2):
        for i in range(4):
            k = 2 * (w * 4 + i)  # even element of the pair -> operands $(2 + k) (low nibble), $(3 + k) (high nibble)
            cvts.append(f"cvt.rn.satfinite.e2m1x2.f32 b{i}, ${3 + k}, ${2 + k};")
        cvts.append(f"mov.b32 ${w}, {{b0, b1, b2, b3}};")
    w0, w1 = inline_ptx(
        "{ .reg .b8 b0, b1, b2, b3;\n" + "\n".join(cvts) + " }",
        write_only_types=[cutlass.Int32, cutlass.Int32],
        read_only_args=list(values),
    )
    return cutlass.Vector.from_elements((w0, w1), cutlass.Int32)


# ---------------------------------------------------------------------------
# max.f32 that FUSES.  ``row_max_reduction`` above keeps ``cute.math.max`` (the
# production SDPA softmax; switching it is a measured A/B on those kernels, not a
# drive-by).  New reductions reach for these two.
# ---------------------------------------------------------------------------


@cutlass.cute.jit
def fmax_f32(a: cutlass.Float32, b: cutlass.Float32) -> cutlass.Float32:
    """PTX ``max.f32`` -- the max ptxas fuses into ``FMNMX`` / ``FMNMX3``.

    ``cute.math.max`` lowers to ``arith.maxnumf``, which the CuTe-DSL -> NVVM path
    emits as compare + select.  Measured on cutlass-dsl 4.8.0.dev0 / sm_107a in the
    gated attention block's MXFP8 quantizer (two ternary abs-max trees per lane):
    ``cute.math.max`` -> 256 FSETP + 224 FSEL and ZERO FMNMX in the whole kernel;
    this ``max.f32`` -> 56 (rowwise) / 60 (columnwise) FMNMX3, and the kernel shrank
    from 1162 -> 762 and 2778 -> 1163 SASS lines.  NaN semantics are the SAME as
    ``maxnumf`` (one NaN operand -> the other operand is returned; PTX ``max.f32``
    without ``.NaN``), so this is purely a lowering choice.  ``|x|`` folds into the
    operand modifier: ``fmax_f32(abs(a), abs(b))`` is one instruction."""
    return inline_ptx("max.f32 $0, $1, $2;", write_only_types=[cutlass.Float32], read_only_args=[a, b])


def abs_max_tree(vals):
    """``max |v|`` over a Python list of ``Float32`` as a TERNARY tree on :func:`fmax_f32`.

    The :func:`row_max_reduction` shape -- each node is ``max(max(a, b), c)``, the
    DEPENDENT pair FMNMX3 fuses, ~log3(N) deep -- with the abs folded into the
    operands.  Trace-time helper (plain Python over traced values), like
    :func:`row_max_reduction`.  Finite inputs; ``len(vals) >= 1``."""
    if len(vals) == 0:
        raise ValueError("abs_max_tree: need at least one value")
    elems = [cute.math.abs(v) for v in vals]
    while len(elems) > 1:
        nxt = []
        for i in range(0, len(elems), 3):
            grp = elems[i : i + 3]
            acc = grp[0]
            for g in grp[1:]:
                acc = fmax_f32(acc, g)
            nxt.append(acc)
        elems = nxt
    return elems[0]


@cutlass.cute.jit
def warp_abs_max_f32(x: cutlass.Float32) -> cutlass.Float32:
    """``max |x|`` over the 32 lanes of the (fully converged) warp in ONE ``redux.sync.max.abs.f32`` (sm_100a+, PTX 8.6;
    ``cute.arch.warp_redux_sync(x, "fmax", abs=True)``); every lane receives the result.

    The along-kv dS scale of the MXFP8 d=256 backward (one E8M0 per 32 kv ROWS of a q column, lane = kv row) is this
    reduction once per column; the shuffle twin :func:`warp_abs_max_f32_shfl` is its bit-exact oracle and the fallback
    for a part without the f32 redux.  Finite and +-inf inputs, signed zeros and fp32 denormals are in contract (bitwise
    the shuffle tree -- pinned by ``test_tile_dsl_warp_abs_max.py``); NaN is not (a softmax gradient carries none, and
    ``redux.sync.max.f32`` without ``.NaN`` and ``max.f32`` may treat it alike or not -- unmeasured).  DSL floor (AGENTS.md
    Rule 7): the ``abs=`` keyword of ``cute.arch.warp_redux_sync`` is the cutlass-dsl 4.8.0 wheel's, the only one this was
    traced with; a wheel without it takes :func:`warp_abs_max_f32_shfl`."""
    return cute.arch.warp_redux_sync(x, "fmax", abs=True)


@cutlass.cute.jit
def warp_abs_max_f32_shfl(x: cutlass.Float32) -> cutlass.Float32:
    """The shuffle-tree twin of :func:`warp_abs_max_f32`: ``|x|`` then five butterfly stages of :func:`fmax_f32`
    (``shfl.sync.bfly`` 16, 8, 4, 2, 1), so every lane ends with the warp's ``max |x|``."""
    v = cute.math.abs(x)
    for i in cutlass.range_constexpr(5):
        v = fmax_f32(v, cute.arch.shuffle_sync_bfly(v, 1 << i))
    return v


@cutlass.cute.jit
def atomic_max_f32_bits(slot: cute.Tensor, value: cutlass.Float32) -> None:
    """``slot[0] = max(slot[0], value)`` for a NON-NEGATIVE fp32 ``value`` through ONE int32 ``atomicMax`` of its bit
    pattern -- the amax fold of every quantizing epilogue (``sdpa/bwd/kernels/sm107/bprop_d256_fp8.py``, the gated block's
    gradient amax pass and gate backward).

    Non-negative IEEE fp32 patterns order exactly as int32, so the slot ends at the fp32 max of every value folded into
    it whatever order the warps arrive in: order-free and bitwise the ``max``.  Contract: ``slot`` is a 1-element fp32
    tensor (4-byte aligned is enough) the caller ZEROED before the first arrive on the same stream -- ``0.0`` is the pattern
    ``0x00000000``, the identity of this max, and a poisoned slot is the caller's bug; ``value >= +0.0`` -- a negative pattern
    (``-0.0`` = ``0x80000000`` included) is a negative int32 and loses every comparison, a NaN pattern (``0x7FC00000``) wins
    them all.  Call it from ONE lane (lane 0 after :func:`warp_abs_max_f32_shfl` / :func:`warp_abs_max_f32`)."""
    ptr = Pointer(slot.iterator.raw_ptr(), dtype=cutlass.Int32)
    nvvm.atomicrmw(nvvm.AtomicOp.MAX, ptr, value.bitcast(cutlass.Int32))
