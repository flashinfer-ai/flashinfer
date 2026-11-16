# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute


@cute.jit
def tmem_alloc(tmem_ptr_i32, num_cols: int, cta_group_kind, is_exclusive: cutlass.Constexpr[bool] = False):
    # is_exclusive is the sm_107/sm_109 >512-column mechanism (GR100's 576-col
    # TMEM; cute.arch.tmem.is_tmem_allocation_exclusive) — pass True only
    # there.  The kwarg is emitted ONLY on the True branch because FROST's
    # declared cutlass-dsl floor is 4.7.0, whose tcgen05_alloc wrapper does not
    # accept it; <=512-column allocations therefore never trace it and keep
    # working on the floor.  (Verified 2026-09-04 on 4.8.0.dev0: the PUBLIC
    # `primitives.tcgen05_alloc` DOES take is_exclusive — the older claim that
    # this needed an internal wheel is false.)
    if cutlass.const_expr(is_exclusive):
        nvvm.tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_cols),
            is_exclusive=True,
            group=cta_group_kind,
        )
    else:
        nvvm.tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_cols),
            group=cta_group_kind,
        )
    nvvm.tcgen05_relinquish_alloc_permit(group=cta_group_kind)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def tmem_dealloc(tmem_ptr_i32, num_cols: int, cta_group_kind):
    tmem_ptr_for_dealloc = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    nvvm.tcgen05_dealloc(
        tmem_ptr_for_dealloc,
        cutlass.Int32(num_cols),
        group=cta_group_kind,
    )
