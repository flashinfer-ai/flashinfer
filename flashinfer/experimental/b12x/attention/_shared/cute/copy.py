"""Minimal donor TMA copy helpers shared by attention kernels."""

from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (
        (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    )
    group_rank_smem = const_expr(
        cute.rank(smem_tensor) - (1 if not single_stage else 0)
    )
    group_rank_gmem = const_expr(
        cute.rank(gmem_tensor) - (1 if not single_stage else 0)
    )
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
        loc=loc,
        ip=ip,
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **new_kwargs):
        cute.copy(
            atom,
            src[None, src_idx],
            dst[None, dst_idx],
            **new_kwargs,
            **kwargs,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def copy_tma_single_stage(*, loc=None, ip=None, **new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs, loc=loc, ip=ip)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )

    return copy_fn
