"""Typed PTX loads, stores and atomic operations with explicit scopes."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


EVICT_NORMAL = 0x1000000000000000

EVICT_FIRST = 0x12F0000000000000

EVICT_LAST = 0x14F0000000000000


@dsl_user_op
def ld_volatile_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i32(), [address],
                            'ld.volatile.global.b32 $0, [$1];', '=r,l',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def ld_volatile_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i64(), [address],
                            'ld.volatile.global.b64 $0, [$1];', '=l,l',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(value)


@dsl_user_op
def ld_global_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    """Ordinary 64-bit load for source counts after the rank rendezvous."""
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i64(), [address],
                            'ld.global.u64 $0, [$1];', '=l,l',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(value)


@dsl_user_op
def ld_peer_param_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    """Read one field of the by-value SymBuffer kernel parameter."""
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(
        T.i64(), [address],
        '{ .reg .b64 p; cvta.to.param::entry.u64 p, $1; '
        'ld.param.u64 $0, [p]; }',
        '=l,l', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return Uint64(value)


@dsl_user_op
def ld_acquire_gpu_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i32(), [address],
                            'ld.acquire.gpu.global.u32 $0, [$1];', '=r,l,~{memory}',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


@dsl_user_op
def ld_acquire_gpu_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i64(), [address],
                            'ld.acquire.gpu.global.u64 $0, [$1];', '=l,l,~{memory}',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(value)


@dsl_user_op
def ld_acquire_sys_u64(ptr: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(T.i64(), [address],
                            'ld.acquire.sys.global.u64 $0, [$1];', '=l,l,~{memory}',
                            has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(value)


@dsl_user_op
def red_release_gpu_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.release.gpu.global.add.u32 [$0], $1;', 'l,r',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_gpu_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.gpu.global.add.u32 [$0], $1;', 'l,r',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_gpu_add_s32(ptr: cute.Pointer, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.gpu.global.add.s32 [$0], $1;', 'l,r',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def red_release_gpu_xor_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'red.release.gpu.global.xor.b64 [$0], $1;', 'l,l,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def atom_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> Uint32:
    previous = llvm.inline_asm(
        T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.global.add.u32 $0, [$1], $2;', '=r,l,r',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(previous)


@dsl_user_op
def atom_add_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> Uint64:
    previous = llvm.inline_asm(
        T.i64(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.global.add.u64 $0, [$1], $2;', '=l,l,l',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(previous)


@dsl_user_op
def atom_sys_add_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> Uint64:
    previous = llvm.inline_asm(
        T.i64(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.relaxed.sys.global.add.u64 $0, [$1], $2;', '=l,l,l,~{memory}',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint64(previous)


@dsl_user_op
def atom_release_gpu_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> Uint32:
    previous = llvm.inline_asm(
        T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.release.gpu.global.add.u32 $0, [$1], $2;', '=r,l,r,~{memory}',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(previous)


@dsl_user_op
def atom_shared_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None) -> Uint32:
    previous = llvm.inline_asm(
        T.i32(), [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
        'atom.shared.add.u32 $0, [$1], $2;', '=r,r,r',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(previous)


@dsl_user_op
def st_release_sys_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'st.release.sys.global.u64 [$0], $1;', 'l,l,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def st_global_u64(ptr: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [ptr.toint(loc=loc, ip=ip).ir_value(), value.ir_value(loc=loc, ip=ip)],
                    'st.global.u64 [$0], $1;', 'l,l',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_shared_u32x4(src_smem: cute.Pointer, *, loc=None, ip=None) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    """Read one 16-byte BF16 peer-store vector from the swizzled L2 tile."""
    values = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>'),
        [src_smem.toint(loc=loc, ip=ip).ir_value()],
        'ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];',
        '=r,=r,=r,=r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Uint32(llvm.extractvalue(T.i32(), values, [i]))
                 for i in range(4))


@dsl_user_op
def st_global_u32x4(dst_gmem: cute.Pointer, value_0: Uint32,
                    value_1: Uint32, value_2: Uint32, value_3: Uint32,
                    *, loc=None, ip=None) -> None:
    """Write 16 BF16 bytes to an already-mapped, 16-byte-aligned peer slot.

    The caller owns rank mapping, row/channel offsets, and the release/acquire
    publication after all vector writes complete.
    """
    llvm.inline_asm(
        None, [dst_gmem.toint(loc=loc, ip=ip).ir_value(),
               value_0.ir_value(loc=loc, ip=ip),
               value_1.ir_value(loc=loc, ip=ip),
               value_2.ir_value(loc=loc, ip=ip),
               value_3.ir_value(loc=loc, ip=ip)],
        'st.global.v4.b32 [$0], {$1, $2, $3, $4};',
        'l,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_global_u32x4(src: cute.Pointer, *, loc=None, ip=None) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    """Read one aligned 16-byte source scale-factor vector."""
    words = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>'),
        [src.toint(loc=loc, ip=ip).ir_value()],
        'ld.global.v4.b32 {$0, $1, $2, $3}, [$4];',
        '=r,=r,=r,=r,l', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Uint32(llvm.extractvalue(T.i32(), words, [i])) for i in range(4))


@dsl_user_op
def st_shared_u32x4(dst: cute.Pointer, a: Uint32, b: Uint32,
                    c: Uint32, d: Uint32, *, loc=None, ip=None) -> None:
    """One lane writes one contiguous 16-byte BF16 combine output vector."""
    llvm.inline_asm(
        None, [dst.toint(loc=loc, ip=ip).ir_value(),
               a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
               c.ir_value(loc=loc, ip=ip), d.ir_value(loc=loc, ip=ip)],
        'st.shared.v4.b32 [$0], {$1, $2, $3, $4};', 'r,r,r,r,r',
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_global_u32x3(src: cute.Pointer, *, loc=None,
                    ip=None) -> tuple[Uint32, Uint32, Uint32]:
    """Read one valid row's source metadata before W2 shared stores."""
    values = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32)>'),
        [src.toint(loc=loc, ip=ip).ir_value()],
        'ld.global.u32 $0, [$3]; '
        'ld.global.u32 $1, [$3+4]; '
        'ld.global.u32 $2, [$3+8];',
        '=r,=r,=r,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Uint32(llvm.extractvalue(T.i32(), values, [i]))
                 for i in range(3))


@dsl_user_op
def stmatrix_fp8x4_trans_late(base_smem: cute.Pointer, byte_offset: Uint32,
                              packed_fp8x4: Uint32, *, loc=None, ip=None) -> None:
    """Form the W1 swizzled shared address in the STSM instruction region.

    The reference forms ``smem_ptr`` inside the FP8 store loop. Keeping the
    add and STSM in one PTX block keeps the summed address local to that
    block. The base and byte offset are shared-memory byte addresses.
    """
    llvm.inline_asm(
        None, [base_smem.toint(loc=loc, ip=ip).ir_value(),
               byte_offset.ir_value(loc=loc, ip=ip),
               packed_fp8x4.ir_value(loc=loc, ip=ip)],
        '{ .reg .b32 addr; add.u32 addr, $0, $1; '
        'stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [addr], {$2}; }',
        'r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def stmatrix_bf16x8_trans(dst_smem: cute.Pointer,
                          packed_bf16x2_0: Uint32, packed_bf16x2_1: Uint32,
                          packed_bf16x2_2: Uint32, packed_bf16x2_3: Uint32,
                          *, loc=None, ip=None) -> None:
    """Reference SM90_U32x4_STSM_T: four packed BF16x2 words per lane.

    These are the reference's four register pairs after rounding and packing;
    the atom's lane/register-to-output bijection is established by the caller.
    """
    llvm.inline_asm(
        None, [dst_smem.toint(loc=loc, ip=ip).ir_value(),
               packed_bf16x2_0.ir_value(loc=loc, ip=ip),
               packed_bf16x2_1.ir_value(loc=loc, ip=ip),
               packed_bf16x2_2.ir_value(loc=loc, ip=ip),
               packed_bf16x2_3.ir_value(loc=loc, ip=ip)],
        'stmatrix.sync.aligned.x4.m8n8.shared.b16.trans '
        '[$0], {$1, $2, $3, $4};',
        'r,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def mapa_shared(ptr: cute.Pointer, peer_cta: Int32, *, loc=None, ip=None) -> Int32:
    address = ptr.toint(loc=loc, ip=ip).ir_value()
    mapped = llvm.inline_asm(
        T.i32(), [address, peer_cta.ir_value(loc=loc, ip=ip)],
        'mapa.shared::cluster.u32 $0, $1, $2;', '=r,r,r',
        has_side_effects=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(mapped)


@dsl_user_op
def st_async_cluster_u32x4(task_ptr: cute.Pointer, mbar_ptr: cute.Pointer,
                           peer_cta: Int32, word_offset: Int32,
                           a: Uint32, b: Uint32, c: Uint32, d: Uint32,
                           *, loc=None, ip=None) -> None:
    task_address = mapa_shared(task_ptr + word_offset, peer_cta, loc=loc, ip=ip)
    mbar_address = mapa_shared(mbar_ptr, peer_cta, loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [task_address.ir_value(loc=loc, ip=ip),
         a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
         c.ir_value(loc=loc, ip=ip), d.ir_value(loc=loc, ip=ip),
         mbar_address.ir_value(loc=loc, ip=ip)],
        'st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 '
        '[$0], {$1, $2, $3, $4}, [$5];',
        'r,r,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def bulk_zero_shared(dst_smem: cute.Pointer, num_bytes: Uint32,
                     *, loc=None, ip=None) -> None:
    """Zero an eight-byte-aligned shared-memory region from one issuer."""
    llvm.inline_asm(
        None, [Uint64(dst_smem.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
               Uint64(num_bytes).ir_value(loc=loc, ip=ip)],
        'st.bulk.weak.shared::cta [$0], $1, 0;',
        'l,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def st_release_u32(ptr: cute.Pointer, value: Uint32, system: cutlass.Constexpr,
                   *, loc=None, ip=None):
    scope = 'sys' if system else 'gpu'
    llvm.inline_asm(None, [ptr.toint().ir_value(), value.ir_value()],
                    f'st.release.{scope}.global.u32 [$0], $1;', 'l,r,~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def copy_global_16(dst: Uint64, src: Uint64, *, loc=None, ip=None):
    llvm.inline_asm(None, [dst.ir_value(), src.ir_value()],
                    '{ .reg .b32 a,b,c,d; '
                    'ld.global.v4.u32 {a,b,c,d}, [$1]; '
                    'st.global.v4.u32 [$0], {a,b,c,d}; }',
                    'l,l,~{memory}', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def copy_global_2x16(dst: Uint64, src: Uint64, *, loc=None, ip=None):
    # Two contiguous warp panels, separated by32 vectors, before either store.
    llvm.inline_asm(None, [dst.ir_value(), src.ir_value()],
                    '{ .reg .b32 a,b,c,d,e,f,g,h; '
                    'ld.global.v4.u32 {a,b,c,d}, [$1]; '
                    'ld.global.v4.u32 {e,f,g,h}, [$1+512]; '
                    'st.global.v4.u32 [$0], {a,b,c,d}; '
                    'st.global.v4.u32 [$0+512], {e,f,g,h}; }',
                    'l,l,~{memory}', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_global_16(src: Uint64, *, loc=None, ip=None) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load half of a split peer copy: one 16-byte remote vector.

    Separating the load from its matching store lets a warp issue a batch of
    PCIe reads before stalling on the first completion; the fused copy form
    serializes one round trip per sixteen bytes.
    """
    words = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>'),
        [src.ir_value(loc=loc, ip=ip)],
        'ld.global.v4.b32 {$0, $1, $2, $3}, [$4];',
        '=r,=r,=r,=r,l', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Uint32(llvm.extractvalue(T.i32(), words, [i]))
                 for i in range(4))


@dsl_user_op
def st_global_16(dst: Uint64, value_0: Uint32, value_1: Uint32,
                 value_2: Uint32, value_3: Uint32, *, loc=None, ip=None):
    """Store half of a split peer copy: one 16-byte local vector."""
    llvm.inline_asm(
        None, [dst.ir_value(loc=loc, ip=ip),
               value_0.ir_value(loc=loc, ip=ip),
               value_1.ir_value(loc=loc, ip=ip),
               value_2.ir_value(loc=loc, ip=ip),
               value_3.ir_value(loc=loc, ip=ip)],
        'st.global.v4.b32 [$0], {$1, $2, $3, $4};',
        'l,r,r,r,r,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def load_shared_8(ptr: cute.Pointer, paired: cutlass.Constexpr,
                   *, loc=None, ip=None):
    offset = 512 if paired else 16
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(' + ','.join(['f32'] * 8) + ')>'),
        [ptr.toint().ir_value()],
        f'ld.shared.v4.f32 {{$0,$1,$2,$3}}, [$8]; '
        f'ld.shared.v4.f32 {{$4,$5,$6,$7}}, [$8+{offset}];',
        ','.join(['=f'] * 8 + ['r','~{memory}']), has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(8))


@dsl_user_op
def load_bf16_8(ptr: Uint64, *, loc=None, ip=None):
    body = '{ .reg .b32 w<4>; .reg .b16 b<8>; '
    body += 'ld.global.v4.u32 {w0,w1,w2,w3}, [$8]; '
    for i in range(4):
        body += f'mov.b32 {{b{2*i},b{2*i+1}}}, w{i}; '
        body += f'cvt.f32.bf16 ${2*i}, b{2*i}; cvt.f32.bf16 ${2*i+1}, b{2*i+1}; '
    body += '}'
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(' + ','.join(['f32'] * 8) + ')>'),
        [ptr.ir_value()], body, ','.join(['=f'] * 8 + ['l','~{memory}']),
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(8))


@dsl_user_op
def store_bf16_8(ptr: Uint64, values: tuple, *, loc=None, ip=None):
    body = '{ .reg .b32 w<4>; '
    for i in range(4):
        body += f'cvt.rn.bf16x2.f32 w{i}, ${2*i+2}, ${2*i+1}; '
    body += 'st.global.v4.u32 [$0], {w0,w1,w2,w3}; }'
    llvm.inline_asm(None, [ptr.ir_value()] + [x.ir_value() for x in values], body,
                    ','.join(['l'] + ['f'] * 8 + ['~{memory}']),
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def atom_acq_rel_add_u32(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None):
    result = llvm.inline_asm(T.i32(), [ptr.toint().ir_value(), value.ir_value()],
                             'atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;',
                             '=r,l,r,~{memory}', has_side_effects=True,
                             asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(result)


@dsl_user_op
def load_scale_pair(ptr: cute.Pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(ir.Type.parse('!llvm.struct<(f32,f32)>'),
                              [ptr.toint().ir_value()],
                              'ld.shared.v2.f32 {$0,$1}, [$2];',
                              '=f,=f,r,~{memory}', has_side_effects=True,
                              asm_dialect=llvm.AsmDialect.AD_ATT)
    return (Float32(llvm.extractvalue(T.f32(), result, [0])),
            Float32(llvm.extractvalue(T.f32(), result, [1])))


@dsl_user_op
def load_shared_4(ptr: cute.Pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(f32,f32,f32,f32)>'),
        [ptr.toint().ir_value()],
        'ld.shared.v4.f32 {$0,$1,$2,$3}, [$4];',
        '=f,=f,=f,=f,r,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(4))


# Preserve the original SM90 compiler memory-ordering contract.
@dsl_user_op
def ld_acquire_sys_u32_sm90(ptr: cute.Pointer, *, loc=None, ip=None):
    value = llvm.inline_asm(T.i32(), [ptr.toint().ir_value()],
                            'ld.acquire.sys.global.u32 $0, [$1];',
                            '=r,l,~{memory}', has_side_effects=True,
                            asm_dialect=llvm.AsmDialect.AD_ATT)
    return Uint32(value)


# Preserve the original SM90 compiler memory-ordering contract.
@dsl_user_op
def ld_shared_u32x4_sm90(ptr: cute.Pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>'), [ptr.toint().ir_value()],
        'ld.shared.v4.u32 {$0,$1,$2,$3}, [$4];',
        '=r,=r,=r,=r,r,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i])) for i in range(4))
