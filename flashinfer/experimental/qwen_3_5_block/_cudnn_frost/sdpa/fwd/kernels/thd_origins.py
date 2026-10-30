# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Sequence-local Q/K/V/O maps with independent caller origins."""

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.handles import tma_slice_runtime_desc
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import (
    THD_SETUP_THREADS,
    emit_clamped_desc,
    emit_seq_descs,
    write_thd_batch_remap,
    write_thd_prefix_warp,
    write_thd_work_prefix,
)

# Map arrays: slot 0 holds O, slots 1..3 hold Q, K and V. Each array has B
# sequence entries and one sentinel entry. Under compact prefixes the load
# arrays hold a single map in their first entry instead.
COMPACT_MAP_SEQUENCE = 0


@cute.jit
def origin_map_ptr(maps, slot: cutlass.Constexpr[int], batch, sequence):
    pointer = maps.iterator.raw_ptr() + (slot * (batch + 1) + sequence) * 16
    return pointer.tospace(cutlass.AddressSpace.generic)


@cute.jit
def origin_tma_slice(maps, slot: cutlass.Constexpr[int], batch, sequence, d, h, row):
    return tma_slice_runtime_desc(origin_map_ptr(maps, slot, batch, sequence), d, h, row, cutlass.Int32(0))


@cute.jit
def origin_compact_word(origins, batch):
    """The Int64 word after the five origin rows: nonzero when the load ports use single maps."""
    return Pointer(cutlass.make_array_view(origins).data_ptr() + cutlass.Int64(5) * cutlass.Int64(batch), dtype=cutlass.Int64)


@cute.jit
def thd_origins_compact_word(origins, batch) -> cutlass.Int64:
    return origin_compact_word(origins, batch).load()


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
    single_maps: cutlass.Constexpr[bool] = False,
):
    """Lengths control work and clipping; physical offsets control addresses.

    A map per sequence also keeps interior KV padding (possibly NaN) outside
    every TMA load. Descriptor writers publish to the tensor-map proxy before
    the consuming launch acquires the maps. Compact prefixes remain unchanged.

    With ``single_maps``, when every sequence's bound origins equal its
    compact prefixes in each port where it occupies rows, the buffers hold no
    interior padding, so the load ports publish one map each (clamped to the
    packed total) and the consumer acquires it once instead of three
    sequence-local maps per work unit. The word after the origin rows records
    that choice for the consuming launch; a consumer that does not read it
    keeps sequence-local maps.
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
    compact_word = origin_compact_word(origins, batch)
    if tid == 0:
        # Single maps unless a live sequence votes otherwise below.
        compact_word.store(cutlass.Int64(1 if single_maps else 0))
    cute.arch.barrier()
    destination = cutlass.make_array_view(origins)
    for b in cutlass.range(tid, batch, THD_SETUP_THREADS, unroll=1):
        # Issue every port's offset load before the first conversion.
        raw = [None] * 5
        for port in cutlass.range_constexpr(5):
            if cutlass.const_expr(specs[port] is not None):
                raw[port] = cutlass.Int64(cutlass.make_array_view(offsets[port])[b])
        cu_q_b = cutlass.Int32(meta[batch + b])
        q_len = cutlass.Int32(meta[batch + b + 1]) - cu_q_b
        cu_k_b = cutlass.Int32(meta[2 * batch + 1 + b])
        kv_len = cutlass.Int32(meta[2 * batch + 2 + b]) - cu_k_b
        mismatch = cutlass.Boolean(False)
        for port in cutlass.range_constexpr(5):
            if cutlass.const_expr(specs[port] is not None):
                spec = specs[port]
                # Q/K/V/O geometry can be rebound through graph stride overrides.
                # Divide element offsets by that invocation's token stride, never
                # a cached stride from the original graph declaration.
                token_stride = cutlass.Int64(tensors[port].stride[1]) if cutlass.const_expr(port < 4) else cutlass.Int64(spec[2])
                origin = raw[port] * cutlass.Int64(spec[1]) // token_stride
                destination[port, b] = origin
                if cutlass.const_expr(single_maps):
                    # A port's shared map covers the packed prefix, so every
                    # sequence that occupies rows in that buffer votes: Q/O/
                    # Stats when it has query rows, K/V when it has keys. A
                    # sequence with keys but no query rows still has its K/V
                    # rows loaded by its predecessor's tile tail.
                    if cutlass.const_expr(port in (1, 2)):
                        live = kv_len > 0
                        expected = cutlass.Int64(cu_k_b)
                    else:
                        live = q_len > 0
                        expected = cutlass.Int64(cu_q_b)
                    mismatch = mismatch | (live & (origin != expected))
        if cutlass.const_expr(single_maps):
            if mismatch:
                compact_word.store(cutlass.Int64(0))
    cute.arch.barrier()
    compact = compact_word.load() != cutlass.Int64(0)
    if nvvm.elect_sync():
        if cutlass.const_expr(single_maps) and compact:
            # O keeps a map per sequence for clipped stores. The load ports
            # take one map each, clamped to the packed total like the
            # compact-prefix path; the consumer acquires each once.
            emit_seq_descs(
                o_desc,
                maps,
                meta,
                batch,
                tensors[3],
                batch,
                cutlass.Int64(tensors[3].stride[1]),
                2,
                slot_base=COMPACT_MAP_SEQUENCE,
                first_batch=warp,
                batch_step=cutlass.Int32(THD_SETUP_THREADS // 32),
                origins=origins,
                origin_row=3 if specs[3] is not None else -1,
            )
            # A zero extent would make a map invalid rather than empty; an
            # all-empty side is never loaded, so one row keeps the map valid.
            total_q = cute.math.max(cutlass.Int32(meta[2 * batch]), cutlass.Int32(1))
            total_kv = cute.math.max(cutlass.Int32(meta[3 * batch + 1]), cutlass.Int32(1))
            if warp == 0:
                emit_clamped_desc(q_desc, maps, 1 * (batch + 1) + COMPACT_MAP_SEQUENCE, total_q, 2)
            if cutlass.const_expr(not paged):
                if warp == 1:
                    emit_clamped_desc(k_desc, maps, 2 * (batch + 1) + COMPACT_MAP_SEQUENCE, total_kv, 2)
                if warp == 2:
                    emit_clamped_desc(v_desc, maps, 3 * (batch + 1) + COMPACT_MAP_SEQUENCE, total_kv, 2)
        else:
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


@cute.kernel
def build_split_load_maps(
    tensors,
    q_desc: cutlass.GridConstant[tmap.TensorMap],
    k_desc: cutlass.GridConstant[tmap.TensorMap],
    v_desc: cutlass.GridConstant[tmap.TensorMap],
    maps,
    origins,
    offsets,
    specs: cutlass.Constexpr,
    meta_t,
    batch: cutlass.Int32,
):
    """Keep packed split partials compact while Q/K/V use independent origins.

    The preceding split setup owns lengths, scheduling and partial-O maps.
    Final O/Stats offsets are consumed by the packed combine, not these maps.
    """
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    origin = cutlass.make_array_view(origins)
    meta = cutlass.make_array_view(meta_t)
    descs = (q_desc, k_desc, v_desc)
    for b in cutlass.range(tid, batch, THD_SETUP_THREADS, unroll=1):
        for port in cutlass.range_constexpr(3):
            if cutlass.const_expr(specs[port] is not None):
                offset = cutlass.Int64(cutlass.make_array_view(offsets[port])[b])
                origin[port, b] = offset * cutlass.Int64(specs[port][1]) // cutlass.Int64(tensors[port].stride[1])
            else:
                prefix = batch if cutlass.const_expr(port == 0) else 2 * batch + 1
                origin[port, b] = cutlass.Int64(meta[prefix + b])
    cute.arch.barrier()
    if nvvm.elect_sync():
        warp = tid // 32
        for port in cutlass.range_constexpr(3):
            # Reuse the four-map layout; slot 0 remains the compact partial O.
            if warp % 4 == port + 1:
                prefix = batch if cutlass.const_expr(port == 0) else 2 * batch + 1
                emit_seq_descs(
                    descs[port],
                    maps,
                    meta,
                    prefix,
                    tensors[port],
                    batch,
                    cutlass.Int64(tensors[port].stride[1]),
                    2,
                    slot_base=(port + 1) * (batch + 1),
                    first_batch=warp // 4,
                    batch_step=cutlass.Int32(THD_SETUP_THREADS // 32 // 4),
                    origins=origins,
                    origin_row=port,
                )
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


build_split_load_maps.set_name_prefix("cudnn", remove_cutlass_symbol=True)
