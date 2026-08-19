# Copyright (c) 2025, Tri Dao.
"""Device helpers for the SM100 block-64 forward kernel."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass import Float32, Int32, Uint32, Boolean, const_expr
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import llvm

from . import mma_sm100_desc as sm100_desc
from .tcgen05_mma_helpers import (
    _tcgen05_mma_kind,
    i64_to_i32x2,
)


@cute.jit
def mbar_arrive_and_wait(mbar_smem_addr: Int32, phase: Int32) -> None:
    """CTA-scope SMEM mbarrier arrive+wait with a long wait timeout."""
    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(mbar_smem_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(phase)).ir_value(),
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.arrive.shared::cta.b64 _, [$0];\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1, 0x989680;\n\t"
        "@p bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def mbar_wait(mbar_smem_addr: Int32, phase: Int32) -> None:
    """CTA-scope SMEM mbarrier wait with a long wait timeout."""
    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(mbar_smem_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(phase)).ir_value(),
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1, 0x989680;\n\t"
        "@p bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tcgen05_fence_after_thread_sync() -> None:
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tcgen05_fence_before_thread_sync() -> None:
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::before_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tmem_rescale_4x32dp32b32x(tmem_addr: Int32, scale: Float32) -> None:
    regs = ", ".join(f"r{i}" for i in range(32))
    scale_ops = "\n\t".join(
        f"mov.b64 la, {{r{i}, r{i + 1}}};\n\t" "mul.rn.f32x2 la, la, lscale;\n\t" f"mov.b64 {{r{i}, r{i + 1}}}, la;" for i in range(0, 32, 2)
    )
    chunk_ops = "\n\t".join(
        "add.u32 addr_cur, addr, "
        f"{chunk * 32};\n\t"
        f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs}}}, [addr_cur];\n\t"
        f"{scale_ops}\n\t"
        f"tcgen05.st.sync.aligned.32x32b.x32.b32 [addr_cur], {{{regs}}};"
        for chunk in range(4)
    )
    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value(),
            Float32(scale).ir_value(),
        ],
        "{\n\t"
        ".reg .b32 addr;\n\t"
        ".reg .b32 addr_cur;\n\t"
        ".reg .b32 r<32>;\n\t"
        ".reg .b64 la;\n\t"
        ".reg .b64 lscale;\n\t"
        ".reg .f32 scale;\n\t"
        "mov.b32 addr, $0;\n\t"
        "mov.f32 scale, $1;\n\t"
        "mov.b64 lscale, {scale, scale};\n\t"
        f"{chunk_ops}\n\t"
        "}\n",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tmem_load_32dp32b32x(tmem_addr: Int32) -> Tuple[Float32, ...]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 32),
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()],
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{"
        "$0, $1, $2, $3, $4, $5, $6, $7, "
        "$8, $9, $10, $11, $12, $13, $14, $15, "
        "$16, $17, $18, $19, $20, $21, $22, $23, "
        "$24, $25, $26, $27, $28, $29, $30, $31"
        "}, [$32];",
        ",".join(["=r"] * 32 + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i])) for i in range(32))


@cute.jit
def cvt_f32x2_to_bf16x2(a: Float32, b: Float32) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(b).ir_value(), Float32(a).ir_value()],
            "cvt.rn.satfinite.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cvt_f32x4_to_e4m3x4(
    a: Float32, b: Float32, c: Float32, d: Float32
) -> Int32:
    """Pack four FP32 values into one E4M3x4 register."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(),
                Float32(b).ir_value(),
                Float32(c).ir_value(),
                Float32(d).ir_value(),
            ],
            "{\n\t"
            ".reg .b16 out01, out23;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 out01, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 out23, $4, $3;\n\t"
            "mov.b32 $0, {out01, out23};\n\t"
            "}",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def shr_u32(x: Uint32, shift: Uint32) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(), Uint32(shift).ir_value()],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def mask_f32x32_by_u32_branch(acc_s: cute.Tensor, mask: Uint32, base: cutlass.Constexpr[int]) -> Tuple[Float32, ...]:
    mask_ops = "\n\t".join(f"and.b32 tmp, $64, {hex(1 << i)};\n\t" "setp.eq.u32 p, tmp, 0;\n\t" f"@p mov.f32 ${i}, neg_inf;" for i in range(32))
    zero_ops = "\n\t".join(f"mov.f32 ${i}, neg_inf;" for i in range(32))
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 32),
        [Float32(acc_s[base + i]).ir_value() for i in range(32)] + [Uint32(mask).ir_value()],
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .pred full;\n\t"
        ".reg .pred zero;\n\t"
        ".reg .u32 tmp;\n\t"
        ".reg .f32 neg_inf;\n\t"
        "setp.eq.u32 full, $64, 0xffffffff;\n\t"
        "@full bra.uni MASK_DONE;\n\t"
        "mov.f32 neg_inf, 0fFF800000;\n\t"
        "setp.eq.u32 zero, $64, 0;\n\t"
        "@!zero bra.uni MASK_PARTIAL;\n\t"
        f"{zero_ops}\n\t"
        "bra.uni MASK_DONE;\n\t"
        "MASK_PARTIAL:\n\t"
        f"{mask_ops}\n\t"
        "MASK_DONE:\n\t"
        "}\n",
        ",".join(["=f"] * 32 + [str(i) for i in range(32)] + ["r"]),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i])) for i in range(32))


@cute.jit
def apply_block_size_mask_64(acc_s: cute.Tensor, block_size: Int32) -> None:
    if block_size < 64:
        for s in cutlass.range_constexpr(2):
            shift = cutlass.max((s + 1) * 32 - block_size, 0)
            mask = shr_u32(Uint32(0xFFFFFFFF), Uint32(shift))
            vals = mask_f32x32_by_u32_branch(acc_s, mask, s * 32)
            for i in cutlass.range_constexpr(32):
                idx = s * 32 + i
                acc_s[idx] = vals[i]


@cute.jit
def tmem_store_bf16x16(tmem_addr: Int32, vals: cute.Tensor) -> None:
    assert cute.size(vals) == 16
    regs = ", ".join(f"${i + 1}" for i in range(16))
    llvm.inline_asm(
        None,
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()] + [Int32(vals[i]).ir_value() for i in range(16)],
        f"tcgen05.st.sync.aligned.32x32b.x16.b32 [$0], {{{regs}}};",
        ",".join(["r"] * 17),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tmem_store_e4m3x8(tmem_addr: Int32, vals: cute.Tensor) -> None:
    """Store 32 packed E4M3 values (8 b32 registers) into TMEM."""
    assert cute.size(vals) == 8
    regs = ", ".join(f"${i + 1}" for i in range(8))
    llvm.inline_asm(
        None,
        [Int32(cute.arch.make_warp_uniform(tmem_addr)).ir_value()]
        + [Int32(vals[i]).ir_value() for i in range(8)],
        f"tcgen05.st.sync.aligned.32x32b.x8.b32 [$0], {{{regs}}};",
        ",".join(["r"] * 9),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tmem_combine_store_exchange_4x32dp32b32x(
    tmem_o0_addr: Int32,
    tmem_o1_addr: Int32,
    exchange_smem_addr: Int32,
    scale0: Float32,
    scale1: Float32,
) -> None:
    regs0 = ", ".join(f"a{i}" for i in range(32))
    regs1 = ", ".join(f"b{i}" for i in range(32))
    combine_ops = "\n\t".join(
        f"mov.b64 la, {{a{i}, a{i + 1}}};\n\t"
        f"mov.b64 lb, {{b{i}, b{i + 1}}};\n\t"
        "mul.rn.f32x2 lb, lb, lscale1;\n\t"
        "fma.rn.f32x2 la, la, lscale0, lb;\n\t"
        f"mov.b64 {{o{i}, o{i + 1}}}, la;"
        for i in range(0, 32, 2)
    )
    store_ops = "\n\t".join(
        f"add.u32 store_addr, saddr_cur, {group * 32 * 4 * 4};\n\t"
        f"st.shared.v4.b32 [store_addr], {{o{group * 4 + 0}, o{group * 4 + 1}, o{group * 4 + 2}, o{group * 4 + 3}}};"
        for group in range(8)
    )
    chunk_ops = "\n\t".join(
        "add.u32 taddr0_cur, taddr0, "
        f"{chunk * 32};\n\t"
        "add.u32 taddr1_cur, taddr1, "
        f"{chunk * 32};\n\t"
        "add.u32 saddr_cur, saddr, "
        f"{chunk * 32 * 32 * 4};\n\t"
        f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs0}}}, [taddr0_cur];\n\t"
        f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs1}}}, [taddr1_cur];\n\t"
        f"{combine_ops}\n\t"
        f"{store_ops}"
        for chunk in range(4)
    )
    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(tmem_o0_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(tmem_o1_addr)).ir_value(),
            Int32(exchange_smem_addr).ir_value(),
            Float32(scale0).ir_value(),
            Float32(scale1).ir_value(),
        ],
        "{\n\t"
        ".reg .b32 taddr0;\n\t"
        ".reg .b32 taddr1;\n\t"
        ".reg .b32 taddr0_cur;\n\t"
        ".reg .b32 taddr1_cur;\n\t"
        ".reg .b32 saddr;\n\t"
        ".reg .b32 saddr_cur;\n\t"
        ".reg .b32 store_addr;\n\t"
        ".reg .b32 a<32>;\n\t"
        ".reg .b32 b<32>;\n\t"
        ".reg .b32 o<32>;\n\t"
        ".reg .b64 la;\n\t"
        ".reg .b64 lb;\n\t"
        ".reg .b64 lscale0;\n\t"
        ".reg .b64 lscale1;\n\t"
        ".reg .f32 scale0;\n\t"
        ".reg .f32 scale1;\n\t"
        "mov.b32 taddr0, $0;\n\t"
        "mov.b32 taddr1, $1;\n\t"
        "mov.b32 saddr, $2;\n\t"
        "mov.f32 scale0, $3;\n\t"
        "mov.f32 scale1, $4;\n\t"
        "mov.b64 lscale0, {scale0, scale0};\n\t"
        "mov.b64 lscale1, {scale1, scale1};\n\t"
        f"{chunk_ops}\n\t"
        "}\n",
        "r,r,r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def smem_zero_store_exchange_4x32dp32b32x(exchange_smem_addr: Int32) -> None:
    store_ops = "\n\t".join(
        f"add.u32 addr, saddr, {chunk * 32 * 32 * 4 + group * 32 * 4 * 4};\n\t" "st.shared.v4.b32 [addr], {z, z, z, z};"
        for chunk in range(4)
        for group in range(8)
    )
    llvm.inline_asm(
        None,
        [Int32(exchange_smem_addr).ir_value()],
        "{\n\t" ".reg .b32 saddr;\n\t" ".reg .b32 addr;\n\t" ".reg .b32 z;\n\t" "mov.b32 saddr, $0;\n\t" "mov.u32 z, 0;\n\t" f"{store_ops}\n\t" "}\n",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def smem_exchange_reduce_store_bf16x32(
    own_exchange_smem_addr: Int32,
    partner_exchange_smem_addr: Int32,
    sO_smem_addr0: Int32,
    sO_smem_addr1: Int32,
    sO_smem_addr2: Int32,
    sO_smem_addr3: Int32,
) -> None:
    load_ops = "\n\t".join(
        f"add.u32 addr_own, own, {group * 32 * 4 * 4};\n\t"
        f"add.u32 addr_partner, partner, {group * 32 * 4 * 4};\n\t"
        f"ld.shared.v4.b32 {{a{group * 4 + 0}, a{group * 4 + 1}, a{group * 4 + 2}, a{group * 4 + 3}}}, [addr_own];\n\t"
        f"ld.shared.v4.b32 {{b{group * 4 + 0}, b{group * 4 + 1}, b{group * 4 + 2}, b{group * 4 + 3}}}, [addr_partner];"
        for group in range(8)
    )
    add_ops = "\n\t".join(
        f"mov.b64 la, {{a{i}, a{i + 1}}};\n\t" f"mov.b64 lb, {{b{i}, b{i + 1}}};\n\t" "add.rn.f32x2 la, la, lb;\n\t" f"mov.b64 {{a{i}, a{i + 1}}}, la;"
        for i in range(0, 32, 2)
    )
    store_ops = "\n\t".join(
        f"cvt.rn.satfinite.bf16x2.f32 p0, a{j + 1}, a{j + 0};\n\t"
        f"cvt.rn.satfinite.bf16x2.f32 p1, a{j + 3}, a{j + 2};\n\t"
        f"cvt.rn.satfinite.bf16x2.f32 p2, a{j + 5}, a{j + 4};\n\t"
        f"cvt.rn.satfinite.bf16x2.f32 p3, a{j + 7}, a{j + 6};\n\t"
        f"st.shared.v4.b32 [${2 + j // 8}], {{p0, p1, p2, p3}};"
        for j in range(0, 32, 8)
    )
    llvm.inline_asm(
        None,
        [
            Int32(own_exchange_smem_addr).ir_value(),
            Int32(partner_exchange_smem_addr).ir_value(),
            Int32(sO_smem_addr0).ir_value(),
            Int32(sO_smem_addr1).ir_value(),
            Int32(sO_smem_addr2).ir_value(),
            Int32(sO_smem_addr3).ir_value(),
        ],
        "{\n\t"
        ".reg .b32 own;\n\t"
        ".reg .b32 partner;\n\t"
        ".reg .b32 addr_own;\n\t"
        ".reg .b32 addr_partner;\n\t"
        ".reg .b32 a<32>;\n\t"
        ".reg .b32 b<32>;\n\t"
        ".reg .b32 p<4>;\n\t"
        ".reg .b64 la;\n\t"
        ".reg .b64 lb;\n\t"
        "mov.b32 own, $0;\n\t"
        "mov.b32 partner, $1;\n\t"
        f"{load_ops}\n\t"
        f"{add_ops}\n\t"
        f"{store_ops}\n\t"
        "}\n",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def smem_exchange_reduce_scale_store_bf16x32(
    own_exchange_smem_addr: Int32,
    partner_exchange_smem_addr: Int32,
    sO_smem_addr0: Int32,
    sO_smem_addr1: Int32,
    sO_smem_addr2: Int32,
    sO_smem_addr3: Int32,
    scale_smem_addr0: Int32,
    scale_smem_addr1: Int32,
    scale_smem_addr2: Int32,
    scale_smem_addr3: Int32,
) -> None:
    """Reduce two FP32 exchange tiles, apply V scale, then convert once."""
    load_ops = "\n\t".join(
        f"add.u32 addr_own, own, {group * 32 * 4 * 4};\n\t"
        f"add.u32 addr_partner, partner, {group * 32 * 4 * 4};\n\t"
        f"ld.shared.v4.b32 {{a{group * 4 + 0}, a{group * 4 + 1}, a{group * 4 + 2}, a{group * 4 + 3}}}, [addr_own];\n\t"
        f"ld.shared.v4.b32 {{b{group * 4 + 0}, b{group * 4 + 1}, b{group * 4 + 2}, b{group * 4 + 3}}}, [addr_partner];"
        for group in range(8)
    )
    scale_load_ops = "\n\t".join(
        f"ld.shared.b32 v{group * 8 + item}, [${6 + group}+{item * 4}];"
        for group in range(4)
        for item in range(8)
    )
    add_scale_ops = "\n\t".join(
        f"mov.b64 la, {{a{i}, a{i + 1}}};\n\t"
        f"mov.b64 lb, {{b{i}, b{i + 1}}};\n\t"
        f"mov.b64 lv, {{v{i}, v{i + 1}}};\n\t"
        "add.rn.f32x2 la, la, lb;\n\t"
        "mul.rn.f32x2 la, la, lv;\n\t"
        f"mov.b64 {{a{i}, a{i + 1}}}, la;"
        for i in range(0, 32, 2)
    )
    store_ops = "\n\t".join(
        f"cvt.rn.satfinite.bf16x2.f32 p0, a{j + 1}, a{j + 0};\n\t"
        f"cvt.rn.satfinite.bf16x2.f32 p1, a{j + 3}, a{j + 2};\n\t"
        f"cvt.rn.satfinite.bf16x2.f32 p2, a{j + 5}, a{j + 4};\n\t"
        f"cvt.rn.satfinite.bf16x2.f32 p3, a{j + 7}, a{j + 6};\n\t"
        f"st.shared.v4.b32 [${2 + j // 8}], {{p0, p1, p2, p3}};"
        for j in range(0, 32, 8)
    )
    llvm.inline_asm(
        None,
        [
            Int32(own_exchange_smem_addr).ir_value(),
            Int32(partner_exchange_smem_addr).ir_value(),
            Int32(sO_smem_addr0).ir_value(),
            Int32(sO_smem_addr1).ir_value(),
            Int32(sO_smem_addr2).ir_value(),
            Int32(sO_smem_addr3).ir_value(),
            Int32(scale_smem_addr0).ir_value(),
            Int32(scale_smem_addr1).ir_value(),
            Int32(scale_smem_addr2).ir_value(),
            Int32(scale_smem_addr3).ir_value(),
        ],
        "{\n\t"
        ".reg .b32 own;\n\t"
        ".reg .b32 partner;\n\t"
        ".reg .b32 addr_own;\n\t"
        ".reg .b32 addr_partner;\n\t"
        ".reg .b32 a<32>;\n\t"
        ".reg .b32 b<32>;\n\t"
        ".reg .b32 v<32>;\n\t"
        ".reg .b32 p<4>;\n\t"
        ".reg .b64 la;\n\t"
        ".reg .b64 lb;\n\t"
        ".reg .b64 lv;\n\t"
        "mov.b32 own, $0;\n\t"
        "mov.b32 partner, $1;\n\t"
        f"{load_ops}\n\t"
        f"{scale_load_ops}\n\t"
        f"{add_scale_ops}\n\t"
        f"{store_ops}\n\t"
        "}\n",
        "r,r,r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def smem_exchange_reduce_store_f32x32(
    own_exchange_smem_addr: Int32,
    partner_exchange_smem_addr: Int32,
    sO_smem_addr0a: Int32,
    sO_smem_addr0b: Int32,
    sO_smem_addr1a: Int32,
    sO_smem_addr1b: Int32,
    sO_smem_addr2a: Int32,
    sO_smem_addr2b: Int32,
    sO_smem_addr3a: Int32,
    sO_smem_addr3b: Int32,
) -> None:
    load_ops = "\n\t".join(
        f"add.u32 addr_own, own, {group * 32 * 4 * 4};\n\t"
        f"add.u32 addr_partner, partner, {group * 32 * 4 * 4};\n\t"
        f"ld.shared.v4.b32 {{a{group * 4 + 0}, a{group * 4 + 1}, a{group * 4 + 2}, a{group * 4 + 3}}}, [addr_own];\n\t"
        f"ld.shared.v4.b32 {{b{group * 4 + 0}, b{group * 4 + 1}, b{group * 4 + 2}, b{group * 4 + 3}}}, [addr_partner];"
        for group in range(8)
    )
    add_ops = "\n\t".join(
        f"mov.b64 la, {{a{i}, a{i + 1}}};\n\t" f"mov.b64 lb, {{b{i}, b{i + 1}}};\n\t" "add.rn.f32x2 la, la, lb;\n\t" f"mov.b64 {{a{i}, a{i + 1}}}, la;"
        for i in range(0, 32, 2)
    )
    store_ops = "\n\t".join(f"st.shared.v4.b32 [${2 + j // 4}], {{a{j + 0}, a{j + 1}, a{j + 2}, a{j + 3}}};" for j in range(0, 32, 4))
    llvm.inline_asm(
        None,
        [
            Int32(own_exchange_smem_addr).ir_value(),
            Int32(partner_exchange_smem_addr).ir_value(),
            Int32(sO_smem_addr0a).ir_value(),
            Int32(sO_smem_addr0b).ir_value(),
            Int32(sO_smem_addr1a).ir_value(),
            Int32(sO_smem_addr1b).ir_value(),
            Int32(sO_smem_addr2a).ir_value(),
            Int32(sO_smem_addr2b).ir_value(),
            Int32(sO_smem_addr3a).ir_value(),
            Int32(sO_smem_addr3b).ir_value(),
        ],
        "{\n\t"
        ".reg .b32 own;\n\t"
        ".reg .b32 partner;\n\t"
        ".reg .b32 addr_own;\n\t"
        ".reg .b32 addr_partner;\n\t"
        ".reg .b32 a<32>;\n\t"
        ".reg .b32 b<32>;\n\t"
        ".reg .b64 la;\n\t"
        ".reg .b64 lb;\n\t"
        "mov.b32 own, $0;\n\t"
        "mov.b32 partner, $1;\n\t"
        f"{load_ops}\n\t"
        f"{add_ops}\n\t"
        f"{store_ops}\n\t"
        "}\n",
        "r,r,r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | Boolean = False,
    tA_addr: Optional[Int32] = None,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K if const_expr(op.a_major_mode == tcgen05.OperandMajorMode.K) else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K if const_expr(op.b_major_mode == tcgen05.OperandMajorMode.K) else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = tCrA.layout if const_expr(not is_ts) else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    # zero_init may be a runtime Boolean (e.g. the loop-carried O_acc_cur flag); Python
    # `not` on it would bake a wrong constant predicate at trace time, so pass the raw
    # value through and flip the setp comparison instead.
    zero_init_is_dynamic = isinstance(zero_init, Boolean)
    pred_str = "p" if zero_init_is_dynamic else "0" if zero_init else "1"
    pred_input = zero_init if zero_init_is_dynamic else not zero_init
    pred_setp = "setp.eq.b32" if zero_init_is_dynamic else "setp.ne.b32"
    mma_kind = _tcgen05_mma_kind(op)
    mma_instr = f"tcgen05.mma.ws.cta_group::1.kind::{mma_kind}"
    mma_suffix = ", 0"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(pred_input).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            f"{pred_setp} p, $2, 0;\n\t"
            f"@leader_thread {mma_instr} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str}{mma_suffix};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread {mma_instr} [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1{mma_suffix};\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(pred_input).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            assert split_arrive is not None, "split_arrive must be provided when mbar_ptr is not None"
            assert split_arrive % op.shape_mnk[2] == 0, "split_arrive must be a multiple of the MMA K extent"
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            assert 1 <= split_arrive_idx <= cute.size(tCrA.shape[2]), "split_arrive must map to a K-tile index within [1, num_k_tiles]"
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 1; \n\t"
                "@P1 bra.uni DONE; \n\t"
                "bra.uni LAB_WAIT; \n\t"
                "DONE: \n\t"
                + (
                    "tcgen05.fence::after_thread_sync; \n\t"
                    if const_expr(op.a_dtype.width == 8)
                    else ""
                )
            )
        else:
            split_arrive_idx = 0
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 tmem_a, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            # The post-wait loop below updates smem_desc_b_lo incrementally, and the
            # pre-wait loop that would otherwise seed it is empty when
            # split_arrive_idx == 1, so initialize it from the base descriptor.
            + ("mov.b32 smem_desc_b_lo, smem_desc_b_lo_start;\n\t" if mbar_ptr is not None else "") + f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            f"{pred_setp} p, $2, 0;\n\t"
            f"@leader_thread {mma_instr} [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str}{mma_suffix};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread {mma_instr} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1{mma_suffix};\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2]) if const_expr(mbar_ptr is None) else split_arrive_idx,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread {mma_instr} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1{mma_suffix};\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
