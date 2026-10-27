# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Sequence-local Q/K/V/O maps with independent caller origins."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.handles import tma_slice_runtime_desc
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import (
    THD_SETUP_THREADS,
    emit_seq_descs,
    write_thd_batch_remap,
    write_thd_prefix_warp,
    write_thd_work_prefix,
)


@cute.jit
def origin_tma_slice(maps, slot: cutlass.Constexpr[int], batch, sequence, d, h, row):
    pointer = maps.iterator.raw_ptr() + (slot * (batch + 1) + sequence) * 16
    return tma_slice_runtime_desc(pointer.tospace(cutlass.AddressSpace.generic), d, h, row, cutlass.Int32(0))


@cute.jit
def origin_tensors(maps_ptr, offset_ptrs, batch):
    """All backing storage is supplied by the prepared plan's workspace."""
    maps = cute.make_tensor(maps_ptr, cute.make_layout((4 * (batch + 1) * 16,), stride=(1,)))
    origins = cute.make_tensor(maps_ptr + 4 * (batch + 1) * 16, cute.make_layout((5, batch), stride=(batch, 1)))
    offsets = tuple(cute.make_tensor(ptr, cute.make_layout((batch + 1,), stride=(1,))) if ptr is not None else None for ptr in offset_ptrs)
    return maps, origins, offsets


@cute.kernel
def build_thd_origins(
    tensors,
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_desc: cutlass.GridConstant[tmap.TensorMap],
    maps,
    origins,
    offsets,
    specs: cutlass.Constexpr,
    meta_t,
    q_lens,
    kv_lens,
    lens_form: cutlass.Int32,
    batch: cutlass.Int32,
    heads: cutlass.Int32,
    tile_rows: cutlass.Int32,
    clusters: cutlass.Int32,
    paged: cutlass.Constexpr[bool] = False,
    rank_by_kv: cutlass.Constexpr[bool] = False,
    release_dependents: cutlass.Constexpr[bool] = False,
    splits: cutlass.Constexpr[int] = 1,
):
    """Lengths control work and clipping; physical offsets control addresses.

    A map per sequence also keeps interior KV padding (possibly NaN) outside
    every TMA load. Descriptor writers publish to the tensor-map proxy before
    the consuming launch acquires the maps. Compact prefixes remain unchanged.
    """
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    if cutlass.const_expr(release_dependents):
        # The consumer may initialize CTA-local state while these maps are
        # built. It must wait for this grid before reading metadata or maps.
        if tid == 0:
            launch_dependent_grids()
    warp, lane = tid // 32, tid % 32
    meta = cutlass.make_array_view(meta_t)
    base_maps = (q_desc, k_desc, v_desc, o_desc)
    if warp == 0:
        write_thd_prefix_warp(meta, cutlass.make_array_view(q_lens), batch, batch, (lens_form & 1) != 0, lane, store_lengths=False)
    if warp == 1:
        write_thd_prefix_warp(meta, cutlass.make_array_view(kv_lens), batch, 2 * batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
    for port in cutlass.range_constexpr(5):
        if cutlass.const_expr(specs[port] is not None):
            spec = specs[port]
            # Q/K/V/O geometry can be rebound through graph stride overrides.
            # Divide element offsets by that invocation's token stride, never
            # a cached stride from the original graph declaration.
            token_stride = cutlass.Int64(tensors[port].stride[1]) if cutlass.const_expr(port < 4) else cutlass.Int64(spec[2])
            source = cutlass.make_array_view(offsets[port])
            destination = cutlass.make_array_view(origins)
            for b in cutlass.range(tid, batch, THD_SETUP_THREADS, unroll=1):
                destination[port, b] = cutlass.Int64(source[b]) * cutlass.Int64(spec[1]) // token_stride
    cute.arch.barrier()
    if nvvm.elect_sync():
        # Stripe warps over ports as well as sequences. A short batch must
        # not serialize all four maps on one elected writer. O/Q occupy the
        # first two slots, which are the only maps needed for paged KV.
        map_count = 2 if cutlass.const_expr(paged) else 4
        for slot in cutlass.range_constexpr(4):
            port = (3, 0, 1, 2)[slot]
            if cutlass.const_expr(not paged or port not in (1, 2)):
                if warp % map_count == slot:
                    cu0 = 2 * batch + 1 if port in (1, 2) else batch
                    emit_seq_descs(
                        base_maps[port],
                        maps,
                        meta,
                        cu0,
                        tensors[port],
                        batch,
                        cutlass.Int64(tensors[port].stride[1]),
                        2,
                        slot_base=slot * (batch + 1),
                        first_batch=warp // map_count,
                        batch_step=cutlass.Int32(THD_SETUP_THREADS // 32 // map_count),
                        origins=origins,
                        origin_row=port if specs[port] is not None else -1,
                    )
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    # Optional packed Stats without offsets keeps the compact-prefix contract.
    if cutlass.const_expr(specs[4] is None):
        for b in cutlass.range(tid, batch, THD_SETUP_THREADS, unroll=1):
            cutlass.make_array_view(origins)[4, b] = cutlass.Int64(meta[batch + b])
    write_thd_batch_remap(meta, batch, tid, cutlass.Int32(THD_SETUP_THREADS), rank_by_kv=rank_by_kv)
    cute.arch.barrier()
    write_thd_work_prefix(meta, batch, heads, tile_rows, clusters, tid, splits)


build_thd_origins.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def build_paged_split_q_maps(
    q, q_desc: cutlass.GridConstant[tmap.TensorMap], maps, origins, q_offsets, multiplier: cutlass.Constexpr[int], meta_t, batch: cutlass.Int32
):
    """Publish physical Q maps after compact split metadata has been built."""
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    origin = cutlass.make_array_view(origins)
    offsets = cutlass.make_array_view(q_offsets)
    for b in cutlass.range(tid, batch, THD_SETUP_THREADS, unroll=1):
        origin[0, b] = cutlass.Int64(offsets[b]) * cutlass.Int64(multiplier) // cutlass.Int64(q.stride[1])
    cute.arch.barrier()
    if nvvm.elect_sync():
        emit_seq_descs(
            q_desc,
            maps,
            cutlass.make_array_view(meta_t),
            batch,
            q,
            batch,
            cutlass.Int64(q.stride[1]),
            2,
            slot_base=batch + 1,
            first_batch=tid // 32,
            batch_step=cutlass.Int32(THD_SETUP_THREADS // 32),
            origins=origins,
            origin_row=0,
        )
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


build_paged_split_q_maps.set_name_prefix("cudnn", remove_cutlass_symbol=True)
