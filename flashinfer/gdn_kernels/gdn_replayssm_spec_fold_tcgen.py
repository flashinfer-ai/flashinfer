"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""SM100 tcgen05 specialization for normalized ReplaySSM T=4..8 commit.

The checkpoint streams through two half-width shared-memory stages while
tcgen05 computes all eight ``S0 @ k_t`` vectors.  One thread then owns each V
row's low-rank recurrence, and the final update reloads the checkpoint from L2
on its way back to global memory.  The smaller shared footprint raises CTA
residency while the second logical state read remains an L2 hit.
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from .cute_dsl_cache_naming import make_kernel_name
from .device_target import gdn_compile_options, gdn_device_target
import cuda.bindings.driver as cuda


_F32 = cutlass.Float32
_BF16 = cutlass.BFloat16
_D = 128
_T = 8
_K_TILE = 32
_THREADS = 256
_STATE_STAGES = 2
_OPERAND_STAGES = 4


@cute.struct
class _SharedStorage:
    state_mbar: cute.struct.MemRange[cutlass.Int64, _STATE_STAGES * 2]
    dot_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding: cutlass.Int32


def _semantic_mn(tensor: cute.Tensor):
    stored = tensor.layout
    layout = stored.outer if isinstance(stored, cute.ComposedLayout) else stored
    shape, stride = layout.shape, layout.stride
    logical = cute.make_layout(
        ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:]),
        stride=((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:]),
    )
    if isinstance(stored, cute.ComposedLayout):
        logical = cute.make_composed_layout(stored.inner, stored.offset, logical)
    return cute.make_tensor(tensor.iterator, logical)


@cute.jit
def _store_rank_update(
    state_matrix: cute.Tensor,
    s_state_scale: cute.Tensor,
    s_coeff: cute.Tensor,
    s_keys: cute.Tensor,
    warp: cutlass.Int32,
    row_base: cutlass.Int32,
    lane: cutlass.Int32,
):
    result = cute.make_rmem_tensor(cute.make_layout((4,)), _F32)
    for row_iter in cutlass.range_constexpr(16):
        row = row_base + row_iter
        for col_group in cutlass.range_constexpr(4):
            col = lane + col_group * 32
            result[col_group] = s_state_scale[warp] * state_matrix[(row, col)]
        for step in cutlass.range_constexpr(_T):
            coefficient = s_coeff[(row, step)]
            for col_group in cutlass.range_constexpr(4):
                col = lane + col_group * 32
                result[col_group] += coefficient * s_keys[(step, col)]
        for col_group in cutlass.range_constexpr(4):
            col = lane + col_group * 32
            state_matrix[(row, col)] = result[col_group]


@cute.jit
def _store_rank_update_packed(
    state_matrix: cute.Tensor,
    s_state_scale: cute.Tensor,
    s_coeff: cute.Tensor,
    s_keys: cute.Tensor,
    warp: cutlass.Int32,
    row_base: cutlass.Int32,
    lane: cutlass.Int32,
):
    keys = cute.make_rmem_tensor(cute.make_layout((_T, 4), stride=(4, 1)), _F32)
    for step in cutlass.range_constexpr(_T):
        for col_group in cutlass.range_constexpr(4):
            keys[(step, col_group)] = s_keys[(step, lane + col_group * 32)]
    result = cute.make_rmem_tensor(cute.make_layout((4,)), _F32)
    for row_iter in cutlass.range_constexpr(16):
        row = row_base + row_iter
        for col_group in cutlass.range_constexpr(4):
            col = lane + col_group * 32
            result[col_group] = state_matrix[(row, col)]
        scale_pair = (s_state_scale[warp], s_state_scale[warp])
        result[0], result[1] = cute.arch.mul_packed_f32x2(
            (result[0], result[1]), scale_pair
        )
        result[2], result[3] = cute.arch.mul_packed_f32x2(
            (result[2], result[3]), scale_pair
        )
        for step in cutlass.range_constexpr(_T):
            coefficient = s_coeff[(row, step)]
            coefficient_pair = (coefficient, coefficient)
            result[0], result[1] = cute.arch.fma_packed_f32x2(
                coefficient_pair,
                (keys[(step, 0)], keys[(step, 1)]),
                (result[0], result[1]),
            )
            result[2], result[3] = cute.arch.fma_packed_f32x2(
                coefficient_pair,
                (keys[(step, 2)], keys[(step, 3)]),
                (result[2], result[3]),
            )
        for col_group in cutlass.range_constexpr(4):
            col = lane + col_group * 32
            state_matrix[(row, col)] = result[col_group]


@cute.kernel
def _gdn_replayssm_fold_tcgen_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_state: cute.CopyAtom,
    tma_state_vdk: cute.Tensor,
    state_vdk: cute.Tensor,
    rawv: cute.Tensor,
    rawk: cute.Tensor,
    log_g: cute.Tensor,
    beta: cute.Tensor,
    state_indices: cute.Tensor,
    accept_lens: cute.Tensor,
    state_smem_layout: cute.ComposedLayout,
    operand_smem_layout: cute.ComposedLayout,
    NUM_LAYERS: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    NULL_BLOCK_ID: cutlass.Constexpr[int],
    USE_PACKED: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    layer_head, i_n, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()

    i_layer = layer_head // HV
    i_hv = layer_head % HV
    i_h = i_hv // (HV // H)
    slots_per_layer = state_vdk.shape[2] // (NUM_LAYERS * HV)
    requested_slot = state_indices[i_n]
    n_commit = accept_lens[i_n]
    is_live = requested_slot > NULL_BLOCK_ID and n_commit > 0
    safe_slot = requested_slot
    if not is_live:
        safe_slot = cutlass.Int32(0)
    flat_slot = i_layer * slots_per_layer + safe_slot
    state_flat = flat_slot * HV + i_hv

    smem = utils.SmemAllocator()
    storage = smem.allocate(_SharedStorage)
    s_state = smem.allocate_tensor(
        _F32,
        state_smem_layout.outer,
        1024,
        swizzle=state_smem_layout.inner,
    )
    s_operand = smem.allocate_tensor(
        _F32,
        operand_smem_layout.outer,
        1024,
        swizzle=operand_smem_layout.inner,
    )
    s_keys = smem.allocate_tensor(_F32, cute.make_layout((_T, _D), stride=(_D, 1)), 16)
    s_coeff = smem.allocate_tensor(_F32, cute.make_layout((_D, _T), stride=(_T, 1)), 16)
    s_gram = smem.allocate_tensor(_F32, cute.make_layout((_T, _T), stride=(_T, 1)), 16)
    s_decay = smem.allocate_tensor(_F32, cute.make_layout((_T,), stride=(1,)), 16)
    s_beta = smem.allocate_tensor(_F32, cute.make_layout((_T,), stride=(1,)), 16)
    s_state_scale = smem.allocate_tensor(_F32, cute.make_layout((8,), stride=(1,)), 16)
    s_operand_logical = _semantic_mn(s_operand)

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=_THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding.ptr, barrier_for_retrieve=tmem_barrier
    )
    tmem.allocate(32)

    if warp == 6:
        cpasync.prefetch_descriptor(tma_atom_state)

    state_bytes = cute.size_in_bytes(
        _F32, cute.select(state_smem_layout, mode=[0, 1, 2])
    )
    state_producer, state_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=_STATE_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=state_bytes,
        barrier_storage=storage.state_mbar.data_ptr(),
    ).make_participants()
    dot_producer, dot_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, _THREADS),
        barrier_storage=storage.dot_mbar.data_ptr(),
    ).make_participants()

    tiler = (_D, _T, _K_TILE)
    tma_state_matrix = tma_state_vdk[(None, None, state_flat)]
    state_matrix = state_vdk[(None, None, state_flat)]
    g_state = cute.local_tile(tma_state_matrix, tiler, (0, 0, None), proj=(1, None, 1))
    thr_mma = tiled_mma.get_slice(0)
    t_cg_state = thr_mma.partition_A(g_state)
    t_cr_state = tiled_mma.make_fragment_A(s_state)
    t_cr_operand = tiled_mma.make_fragment_B(s_operand)
    t_ss_state, t_sg_state = cpasync.tma_partition(
        tma_atom_state,
        0,
        cute.make_layout(1),
        cute.group_modes(s_state, 0, 3),
        cute.group_modes(t_cg_state, 0, 3),
    )

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(_F32)
    t_ct_dot_fake = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((_D, _T)))
    t_ct_dot = cute.make_tensor(tmem_ptr, t_ct_dot_fake.layout)

    num_k_tiles = cute.size(g_state, mode=[2])
    if warp == 6:
        for k_idx in cutlass.range_constexpr(_STATE_STAGES):
            empty = state_producer.acquire_and_advance()
            cute.copy(
                tma_atom_state,
                t_sg_state[(None, k_idx)],
                t_ss_state[(None, empty.index)],
                tma_bar_ptr=empty.barrier,
            )

    # One warp prepares each normalized key while the checkpoint TMA runs.
    key = cute.make_rmem_tensor(cute.make_layout((4,)), _F32)
    i_t = warp
    if i_t < _T:
        key_norm = _F32(0.0)
        for elem in cutlass.range_constexpr(4):
            k_idx = lane + elem * 32
            # Rejected keys must not enter the rank update, even as 0 * NaN.
            key[elem] = _F32(0.0)
            if is_live and i_t < n_commit:
                key[elem] = rawk[(flat_slot, i_h, i_t, k_idx)].to(_F32)
            key_norm += key[elem] * key[elem]
        key_norm = cute.arch.warp_reduction_sum(key_norm)
        inv_norm = cute.rsqrt(key_norm + 1.0e-6)
        for elem in cutlass.range_constexpr(4):
            k_idx = lane + elem * 32
            normalized = key[elem] * inv_norm
            key[elem] = normalized
            s_keys[(i_t, k_idx)] = normalized
            stage = k_idx // _K_TILE
            k_local = k_idx % _K_TILE
            s_operand_logical[(i_t, k_local, stage)] = normalized
        if lane == 0:
            # The rank-eight MMA tile may be wider than the logical cache.
            s_decay[i_t] = 1.0
            s_beta[i_t] = 0.0
            if i_t < rawk.shape[2]:
                s_decay[i_t] = cute.exp(log_g[(flat_slot, i_hv, i_t)], fastmath=True)
                s_beta[i_t] = beta[(flat_slot, i_hv, i_t)]

    # Publish scalar SMEM stores to the asynchronous MMA proxy.
    cute.arch.fence_view_async_shared()
    cute.arch.barrier()

    # The eight key-owning warps independently form the eight Gram rows.
    if i_t < _T:
        for j in cutlass.range_constexpr(_T):
            if i_t < j:
                dot = _F32(0.0)
                for elem in cutlass.range_constexpr(4):
                    k_idx = lane + elem * 32
                    dot += key[elem] * s_keys[(j, k_idx)]
                dot = cute.arch.warp_reduction_sum(dot)
                if lane == 0:
                    s_gram[(i_t, j)] = dot

    if warp == 6:
        for k_idx in cutlass.range(_STATE_STAGES, num_k_tiles, 1, unroll=1):
            empty = state_producer.acquire_and_advance()
            cute.copy(
                tma_atom_state,
                t_sg_state[(None, k_idx)],
                t_ss_state[(None, empty.index)],
                tma_bar_ptr=empty.barrier,
            )

    if warp == 7:
        dot_empty = dot_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_idx in cutlass.range(num_k_tiles, unroll=1):
            full = state_consumer.wait_and_advance()
            for k_block in cutlass.range_constexpr(cute.size(t_cr_state, mode=[2])):
                cute.gemm(
                    tiled_mma,
                    t_ct_dot,
                    t_cr_state[(None, None, k_block, full.index)],
                    t_cr_operand[(None, None, k_block, k_idx)],
                    t_ct_dot,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            full.release()
        dot_empty.commit()

    dot_load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x8), _F32)
    dot_tmem_copy = tcgen05.make_tmem_copy(dot_load_atom, t_ct_dot)
    local_tidx = tidx % _D
    dot_thr_copy = dot_tmem_copy.get_slice(local_tidx)
    t_rt_dot_all = dot_thr_copy.partition_S(t_ct_dot)
    dot_identity = cute.make_identity_tensor((_D, _T))
    t_cc_dot = thr_mma.partition_C(dot_identity)
    t_rc_dot_all = dot_thr_copy.partition_D(t_cc_dot)
    t_rt_dot = t_rt_dot_all[(None, 0, None, None)]
    t_rc_dot = t_rc_dot_all[(None, 0, None, None)]
    r_base = cute.make_rmem_tensor(t_rc_dot.shape, _F32)

    source_lane = lane % 16
    row_base = warp * 32
    if warp >= 4:
        source_lane += 16
        row_base = (warp - 4) * 32 + 16
    row = row_base + lane % 16
    raw_values = cute.make_rmem_tensor(cute.make_layout((_T,)), _F32)
    for step in cutlass.range_constexpr(_T):
        if step < n_commit:
            raw_values[step] = rawv[(flat_slot, i_hv, step, row)].to(_F32)

    tmem.relinquish_alloc_permit()
    dot_consumer.wait_and_advance()
    cute.copy(dot_tmem_copy, t_rt_dot, r_base)
    for step in cutlass.range_constexpr(_T):
        r_base[step] = cute.arch.shuffle_sync(r_base[step], source_lane)
    # Gram rows are produced independently by all eight warps.
    cute.arch.barrier()
    coeff = cute.make_rmem_tensor(cute.make_layout((_T,)), _F32)
    coeff.fill(0.0)
    state_scale = _F32(1.0)
    for step in cutlass.range_constexpr(_T):
        if step < n_commit:
            gate = s_decay[step]
            state_scale *= gate
            prediction = state_scale * r_base[step]
            for j in cutlass.range_constexpr(step):
                coeff[j] *= gate
                prediction += coeff[j] * s_gram[(j, step)]
            coeff[step] = (raw_values[step] - prediction) * s_beta[step]
    if lane < 16:
        for step in cutlass.range_constexpr(_T):
            s_coeff[(row, step)] = coeff[step]
        if lane == 0:
            s_state_scale[warp] = state_scale

    cute.arch.sync_warp()

    # Each warp owns one V row at a time.  Lanes span consecutive K columns,
    # eliminating the four-way shared-bank conflicts of a per-thread float4
    # mapping while still reusing one coefficient across four K quadrants.
    if is_live:
        if cutlass.const_expr(USE_PACKED):
            _store_rank_update_packed(
                state_matrix,
                s_state_scale,
                s_coeff,
                s_keys,
                warp,
                row_base,
                lane,
            )
        else:
            _store_rank_update(
                state_matrix,
                s_state_scale,
                s_coeff,
                s_keys,
                warp,
                row_base,
                lane,
            )

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def _launch_gdn_replayssm_fold_tcgen(
    state: cute.Tensor,
    rawv: cute.Tensor,
    rawk: cute.Tensor,
    log_g: cute.Tensor,
    beta: cute.Tensor,
    state_indices: cute.Tensor,
    accept_lens: cute.Tensor,
    NUM_LAYERS: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    NULL_BLOCK_ID: cutlass.Constexpr[int],
    USE_PACKED: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    op = tcgen05.MmaTF32Op(
        (_D, _T, 8),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    tiler = (_D, _T, _K_TILE)
    state_layout = sm100_utils.make_smem_layout_a(tiled_mma, tiler, _F32, _STATE_STAGES)
    operand_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, tiler, _F32, _OPERAND_STAGES
    )

    state_vdk = cute.make_tensor(
        state.iterator,
        cute.make_layout(
            (_D, _D, state.shape[0] * HV),
            stride=(_D, 1, _D * _D),
        ),
    )
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom, tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        state_vdk,
        cute.select(state_layout, mode=[0, 1, 2]),
        tiler,
        tiled_mma,
    )

    _gdn_replayssm_fold_tcgen_kernel(
        tiled_mma,
        tma_atom,
        tma_tensor,
        state_vdk,
        rawv,
        rawk,
        log_g,
        beta,
        state_indices,
        accept_lens,
        state_layout,
        operand_layout,
        NUM_LAYERS,
        H,
        HV,
        NULL_BLOCK_ID,
        USE_PACKED,
    ).launch(
        grid=(NUM_LAYERS * HV, state_indices.shape[0], 1),
        block=(_THREADS, 1, 1),
        stream=stream,
    )


_CACHE: dict = {}


def _mark_leading_dynamic(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tuple(range(tensor.dim())),
        divisibility=1,
    )


def run_replayssm_fold_tcgen(
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    null_block_id: int = -1,
) -> None:
    """Launch normalized T=4..8 commit after the common entry validates inputs."""
    num_layers, num_slots, HV, _, _ = checkpoint_state.shape
    H = rawk_cache.shape[2]
    cache_len = rawk_cache.shape[3]
    B = ssm_state_indices.shape[0]
    target = gdn_device_target(checkpoint_state.device)
    flat_slots = num_layers * num_slots
    state = checkpoint_state.view(flat_slots, HV, _D, _D)
    rawv = rawv_cache.view(flat_slots, HV, cache_len, _D)
    rawk = rawk_cache.view(flat_slots, H, cache_len, _D)
    log_g = g_cache.view(flat_slots, HV, cache_len)
    beta = beta_cache.view(flat_slots, HV, cache_len)
    use_packed = B > 1
    key = (
        target.compile_key,
        num_layers,
        H,
        HV,
        int(null_block_id),
        use_packed,
        cache_len,
    )
    cache = _CACHE.setdefault(key, {})
    stream = cuda.CUstream(
        torch.cuda.current_stream(device=checkpoint_state.device).cuda_stream
    )
    compile_options = gdn_compile_options(
        checkpoint_state.device, cute.EnableTVMFFI(True), cute.GenerateLineInfo(True)
    )
    if "compiled" not in cache:
        cache["compiled"] = build_and_load_cute_dsl_kernel(
            "gdn_replayssm_spec_fold_tcgen",
            make_kernel_name("fold", target.arch, *key[1:]),
            lambda: cute.compile[compile_options](
                _launch_gdn_replayssm_fold_tcgen,
                _mark_leading_dynamic(state),
                _mark_leading_dynamic(rawv),
                _mark_leading_dynamic(rawk),
                _mark_leading_dynamic(log_g),
                _mark_leading_dynamic(beta),
                _mark_leading_dynamic(ssm_state_indices),
                _mark_leading_dynamic(accept_lens),
                NUM_LAYERS=num_layers,
                H=H,
                HV=HV,
                NULL_BLOCK_ID=int(null_block_id),
                USE_PACKED=use_packed,
                stream=stream,
            ),
            extra_key_files=(__file__,),
        )
    cache["compiled"](
        state,
        rawv,
        rawk,
        log_g,
        beta,
        ssm_state_indices,
        accept_lens,
        stream,
    )


__all__ = ["run_replayssm_fold_tcgen"]
