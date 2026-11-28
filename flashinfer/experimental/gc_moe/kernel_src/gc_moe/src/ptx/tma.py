"""PTX asynchronous copies and TensorMap operations for GC MoE.

Raw-address 1D bulk copies serve the SM90 matrix-staging path.
TensorMap loads/stores and async-proxy fences serve the SM100 path.
The helpers expose their barrier and cache-policy operands explicitly.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Uint8, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .ld_st import (
    EVICT_FIRST,
    EVICT_NORMAL,
)


@dsl_user_op
def global_async_proxy_fence(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'fence.proxy.async.global;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc, ip=ip)


@dsl_user_op
def generic_tensormap_ptr(param_ptr: cute.Pointer,
                          *, loc=None, ip=None) -> cute.Pointer:
    """Convert a selected by-value TensorMap parameter to a generic pointer.

    The JIT value-formal bridge supplies the raw parameter-space address.
    Call this on the full loader warp after task phase selection and before
    elected TMA issue for each task.
    """
    address = param_ptr.toint(loc=loc, ip=ip).ir_value()
    generic = llvm.inline_asm(
        T.i64(), [address], 'cvta.param.u64 $0, $1;', '=l,l',
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cute.make_ptr(Uint8, Uint64(generic), cute.AddressSpace.generic,
                         assumed_align=128, loc=loc, ip=ip)


@dsl_user_op
def prefetch_tensormap(desc: cute.Pointer, *, loc=None, ip=None) -> None:
    """Prefetch a TensorMap through its generic 64-bit address."""
    llvm.inline_asm(None, [desc.toint(loc=loc, ip=ip).ir_value()],
                    'prefetch.tensormap [$0];', 'l',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT,
                    loc=loc, ip=ip)


@dsl_user_op
def bulk_g2s_1d(dst_smem: cute.Pointer, src_gmem: cute.Pointer,
                num_bytes: Uint32, mbar_smem: cute.Pointer,
                *, cache_hint: int = EVICT_FIRST, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [dst_smem.toint(loc=loc, ip=ip).ir_value(),
               src_gmem.toint(loc=loc, ip=ip).ir_value(),
               num_bytes.ir_value(loc=loc, ip=ip),
               mbar_smem.toint(loc=loc, ip=ip).ir_value(),
               Uint64(cache_hint).ir_value(loc=loc, ip=ip)],
        'cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.'
        'L2::cache_hint [$0], [$1], $2, [$3], $4;',
        'r,l,r,r,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def bulk_s2g_1d(dst_gmem: cute.Pointer, src_smem: cute.Pointer,
                num_bytes: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [dst_gmem.toint(loc=loc, ip=ip).ir_value(),
               src_smem.toint(loc=loc, ip=ip).ir_value(),
               num_bytes.ir_value(loc=loc, ip=ip),
               Uint64(EVICT_NORMAL).ir_value(loc=loc, ip=ip)],
        'cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint '
        '[$0], [$1], $2, $3;',
        'l,r,r,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tile_g2s_2cta_2d(dst_smem: cute.Pointer, desc_gmem: cute.Pointer,
                     mbar_smem: cute.Pointer, inner: Uint32, outer: Uint32,
                     cache_hint: Uint64, *, loc=None, ip=None) -> None:
    """SM100 2CTA 2D TensorMap load; both CTAs target the leader barrier."""
    bar = mbar_smem.toint(loc=loc, ip=ip).ir_value()
    bar_leader = llvm.inline_asm(T.i32(), [bar],
                                 'and.b32 $0, $1, 0xfeffffff;', '=r,r',
                                 asm_dialect=llvm.AsmDialect.AD_ATT)
    llvm.inline_asm(
        None, [dst_smem.toint(loc=loc, ip=ip).ir_value(),
               desc_gmem.toint(loc=loc, ip=ip).ir_value(), bar_leader,
               inner.ir_value(loc=loc, ip=ip), outer.ir_value(loc=loc, ip=ip),
               cache_hint.ir_value(loc=loc, ip=ip)],
        'cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.'
        'mbarrier::complete_tx::bytes.L2::cache_hint '
        '[$0], [$1, {$3, $4}], [$2], $5;',
        'r,l,r,r,r,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tile_g2s_2d(dst_smem: cute.Pointer, desc_gmem: cute.Pointer,
                mbar_smem: cute.Pointer, inner: Uint32, outer: Uint32,
                cache_hint: Uint64, *, loc=None, ip=None) -> None:
    """Single-CTA 2D TensorMap load (SM120, no cluster)."""
    llvm.inline_asm(
        None, [dst_smem.toint(loc=loc, ip=ip).ir_value(),
               desc_gmem.toint(loc=loc, ip=ip).ir_value(),
               mbar_smem.toint(loc=loc, ip=ip).ir_value(),
               inner.ir_value(loc=loc, ip=ip), outer.ir_value(loc=loc, ip=ip),
               cache_hint.ir_value(loc=loc, ip=ip)],
        'cp.async.bulk.tensor.2d.shared::cluster.global.'
        'mbarrier::complete_tx::bytes.L2::cache_hint '
        '[$0], [$1, {$3, $4}], [$2], $5;',
        'r,l,r,r,r,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tile_g2s_4d(dst_smem: cute.Pointer, desc_gmem: cute.Pointer,
                mbar_smem: cute.Pointer, k: Uint32, n: Uint32,
                domain: Uint32, expert: Uint32, cache_hint: Uint64,
                *, loc=None, ip=None) -> None:
    """Single-CTA 4D TensorMap load, including FP4->8bit SMEM unpack (SM120)."""
    llvm.inline_asm(
        None, [dst_smem.toint(loc=loc, ip=ip).ir_value(),
               desc_gmem.toint(loc=loc, ip=ip).ir_value(),
               mbar_smem.toint(loc=loc, ip=ip).ir_value(),
               k.ir_value(loc=loc, ip=ip), n.ir_value(loc=loc, ip=ip),
               domain.ir_value(loc=loc, ip=ip), expert.ir_value(loc=loc, ip=ip),
               cache_hint.ir_value(loc=loc, ip=ip)],
        'cp.async.bulk.tensor.4d.shared::cluster.global.'
        'mbarrier::complete_tx::bytes.L2::cache_hint '
        '[$0], [$1, {$3, $4, $5, $6}], [$2], $7;',
        'r,l,r,r,r,r,r,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def prefetch_g2s_4d(desc_gmem: cute.Pointer, k: Uint32, n: Uint32,
                    domain: Uint32, expert: Uint32,
                    *, loc=None, ip=None) -> None:
    """Fire-and-forget 4D TensorMap L2 prefetch."""
    llvm.inline_asm(
        None, [desc_gmem.toint(loc=loc, ip=ip).ir_value(),
               k.ir_value(loc=loc, ip=ip), n.ir_value(loc=loc, ip=ip),
               domain.ir_value(loc=loc, ip=ip), expert.ir_value(loc=loc, ip=ip)],
        'cp.async.bulk.prefetch.tensor.4d.L2.global.tile '
        '[$0, {$1, $2, $3, $4}];',
        'l,r,r,r,r,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tile_g2s_2cta_4d(dst_smem: cute.Pointer, desc_gmem: cute.Pointer,
                     mbar_smem: cute.Pointer, k: Uint32, n: Uint32,
                     domain: Uint32, expert: Uint32, cache_hint: Uint64,
                     *, loc=None, ip=None) -> None:
    """SM100 2CTA 4D load, including TensorMap FP4→8bit SMEM unpack."""
    bar = mbar_smem.toint(loc=loc, ip=ip).ir_value()
    bar_leader = llvm.inline_asm(T.i32(), [bar],
                                 'and.b32 $0, $1, 0xfeffffff;', '=r,r',
                                 asm_dialect=llvm.AsmDialect.AD_ATT)
    llvm.inline_asm(
        None, [dst_smem.toint(loc=loc, ip=ip).ir_value(),
               desc_gmem.toint(loc=loc, ip=ip).ir_value(), bar_leader,
               k.ir_value(loc=loc, ip=ip), n.ir_value(loc=loc, ip=ip),
               domain.ir_value(loc=loc, ip=ip), expert.ir_value(loc=loc, ip=ip),
               cache_hint.ir_value(loc=loc, ip=ip)],
        'cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.'
        'mbarrier::complete_tx::bytes.L2::cache_hint '
        '[$0], [$1, {$3, $4, $5, $6}], [$2], $7;',
        'r,l,r,r,r,r,r,l,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tile_s2g_2d(desc_gmem: cute.Pointer, src_smem: cute.Pointer,
                inner: Uint32, outer: Uint32,
                *, loc=None, ip=None) -> None:
    """Reference L1 epilogue's unhinted TensorMap 2D global store."""
    llvm.inline_asm(
        None, [desc_gmem.toint(loc=loc, ip=ip).ir_value(),
               src_smem.toint(loc=loc, ip=ip).ir_value(),
               inner.ir_value(loc=loc, ip=ip), outer.ir_value(loc=loc, ip=ip)],
        'cp.async.bulk.tensor.2d.global.shared::cta.bulk_group '
        '[$0, {$2, $3}], [$1];',
        'l,r,r,r,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tma_store_fence(*, loc=None, ip=None) -> None:
    """Publish completed generic shared writes to the async proxy.

    Call after copy/store completion and the required thread synchronization,
    before an async TMA store or WGMMA reads the shared stage.
    """
    llvm.inline_asm(None, [], 'fence.proxy.async.shared::cta;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tma_store_arrive(*, loc=None, ip=None) -> None:
    """Commit the caller's issued S2G copies as one TMA bulk group."""
    llvm.inline_asm(None, [], 'cp.async.bulk.commit_group;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tma_store_wait_1(*, loc=None, ip=None) -> None:
    """Wait until only the newest of two committed store groups remains.

    Used before recycling the older of two shared output stages.  This uses
    ``wait_group`` without ``.read`` to wait for store completion.
    """
    llvm.inline_asm(None, [], 'cp.async.bulk.wait_group 1;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tma_store_wait_0(*, loc=None, ip=None) -> None:
    """Drain committed S2G stores before publishing the L2 ready signal."""
    llvm.inline_asm(None, [], 'cp.async.bulk.wait_group 0;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def copy_async_16(dst: cute.Pointer, src: cute.Pointer, size: Uint32,
                  *, loc=None, ip=None):
    llvm.inline_asm(None, [dst.toint().ir_value(), src.toint().ir_value(), size.ir_value()],
                    'cp.async.cg.shared.global [$0], [$1], 16, $2;',
                    'r,l,r,~{memory}', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def copy_commit(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'cp.async.commit_group;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def copy_wait(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'cp.async.wait_group 0;', '~{memory}',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def copy_async_4(dst: cute.Pointer, src: cute.Pointer, size: Uint32,
                 *, loc=None, ip=None):
    llvm.inline_asm(None, [dst.toint().ir_value(), src.toint().ir_value(), size.ir_value()],
                    'cp.async.ca.shared.global [$0], [$1], 4, $2;',
                    'r,l,r,~{memory}', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def copy_arrive(bar: cute.Pointer, *, loc=None, ip=None):
    llvm.inline_asm(None, [bar.toint().ir_value()],
                    'cp.async.mbarrier.arrive.noinc.shared.b64 [$0];',
                    'r,~{memory}', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)
