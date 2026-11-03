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

"""SM100 ReplaySSM verify specialized for a frozen FP32 128x128 state.

The state is streamed once with TMA and multiplied by all K/Q vectors with a
tcgen05 TF32 MMAs.  The small low-rank recurrence is evaluated directly
from the TMEM epilogue, so no intermediate state-dot tensor reaches GMEM.
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


_ACC = cutlass.Float32
_M = 128
_K_TILE = 64
_THREADS = 256
_STAGES = 2


@cute.struct
class _SharedStorage:
    ab_mbar: cute.struct.MemRange[cutlass.Int64, _STAGES * 2]
    acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding: cutlass.Int32


def _semantic_mn(tensor: cute.Tensor):
    """Turn a partitioned UMMA SMEM layout back into logical (MN, K, stage)."""
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
def _to_tf32(fragment: cute.Tensor):
    values = fragment.load()
    result = cute.make_rmem_tensor_like(fragment, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(fragment)):
        result[i] = cute.arch.cvt_f32_tf32(values[i])
    return result


@cute.kernel
def gdn_verify_kernel_mtp_replayssm_tcgen(
    tiled_mma: cute.TiledMma,
    tma_atom_state: cute.CopyAtom,
    state_vkl: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    output: cute.Tensor,
    state_indices: cute.Tensor,
    replayssm_rawv: cute.Tensor,
    replayssm_rawk: cute.Tensor,
    replayssm_g: cute.Tensor,
    replayssm_beta: cute.Tensor,
    state_smem_layout: cute.ComposedLayout,
    operand_smem_layout: cute.ComposedLayout,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    HEADS_PER_CTA: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    block, _, _ = cute.arch.block_idx()
    head_group = block % (HV // HEADS_PER_CTA)
    i_n = block // (HV // HEADS_PER_CTA)
    i_hv0 = head_group * HEADS_PER_CTA
    i_hv1 = i_hv0 + 1
    i_h = i_hv0 // (HV // H)
    cache_idx = state_indices[i_n]
    valid = cache_idx >= 0
    safe_cache_idx = cache_idx
    if not valid:
        safe_cache_idx = 0
    state_flat0 = safe_cache_idx * HV + i_hv0
    state_flat1 = state_flat0
    if cutlass.const_expr(HEADS_PER_CTA == 2):
        state_flat1 = safe_cache_idx * HV + i_hv1

    smem = utils.SmemAllocator()
    storage = smem.allocate(_SharedStorage)
    s_state = smem.allocate_tensor(
        _ACC,
        state_smem_layout.outer,
        1024,
        swizzle=state_smem_layout.inner,
    )
    s_operand = smem.allocate_tensor(
        _ACC,
        operand_smem_layout.outer,
        1024,
        swizzle=operand_smem_layout.inner,
    )
    s_operand_logical = _semantic_mn(s_operand)
    s_values = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((2, 8, _M), stride=(8 * _M, _M, 1)),
        16,
    )
    s_rawk = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((8, 128), stride=(128, 1)), 16
    )
    s_decay = smem.allocate_tensor(_ACC, cute.make_layout((2, 8), stride=(8, 1)), 16)
    s_log_decay = smem.allocate_tensor(
        _ACC, cute.make_layout((2, 8), stride=(8, 1)), 16
    )
    s_beta = smem.allocate_tensor(_ACC, cute.make_layout((2, 8), stride=(8, 1)), 16)
    s_gram = smem.allocate_tensor(_ACC, cute.make_layout((16, 16), stride=(16, 1)), 16)

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=_THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding.ptr, barrier_for_retrieve=tmem_barrier
    )
    # Two M128xN16 accumulators exactly fill the minimum 32-column allocation.
    tmem.allocate(32)

    if warp == 0:
        cpasync.prefetch_descriptor(tma_atom_state)

    state_bytes = cute.size_in_bytes(
        _ACC, cute.select(state_smem_layout, mode=[0, 1, 2])
    )
    state_producer, state_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=state_bytes,
        barrier_storage=storage.ab_mbar.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, _THREADS),
        barrier_storage=storage.acc_mbar.data_ptr(),
    ).make_participants()

    tiler = (_M, N, _K_TILE)
    state_matrix0 = state_vkl[(None, None, state_flat0)]
    state_matrix1 = state_vkl[(None, None, state_flat1)]
    g_state0 = cute.local_tile(state_matrix0, tiler, (0, 0, None), proj=(1, None, 1))
    g_state1 = cute.local_tile(state_matrix1, tiler, (0, 0, None), proj=(1, None, 1))
    thr_mma = tiled_mma.get_slice(0)
    t_cg_state0 = thr_mma.partition_A(g_state0)
    t_cg_state1 = thr_mma.partition_A(g_state1)
    t_cr_state = tiled_mma.make_fragment_A(s_state)
    t_cr_operand = tiled_mma.make_fragment_B(s_operand)
    t_ct_acc_fake = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((_M, N)))
    t_ss_state, t_sg_state0 = cpasync.tma_partition(
        tma_atom_state,
        0,
        cute.make_layout(1),
        cute.group_modes(s_state, 0, 3),
        cute.group_modes(t_cg_state0, 0, 3),
    )
    _, t_sg_state1 = cpasync.tma_partition(
        tma_atom_state,
        0,
        cute.make_layout(1),
        cute.group_modes(s_state, 0, 3),
        cute.group_modes(t_cg_state1, 0, 3),
    )

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(_ACC)
    t_ct_acc0 = cute.make_tensor(tmem_ptr, t_ct_acc_fake.layout)
    t_ct_acc1 = cute.make_tensor(tmem_ptr + N, t_ct_acc_fake.layout)

    # Load one padded operand tile per V row (x8 for T=4, x16 for T=5..8).
    if cutlass.const_expr(N == 8):
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x8), _ACC)
    else:
        tmem_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), _ACC
        )
    tmem_copy = tcgen05.make_tmem_copy(tmem_atom, t_ct_acc0)
    epilogue_head = tidx // 128
    local_tidx = tidx % 128
    thr_tmem = tmem_copy.get_slice(local_tidx)
    t_rt_acc0_all = thr_tmem.partition_S(t_ct_acc0)
    t_rt_acc1_all = thr_tmem.partition_S(t_ct_acc1)
    c_identity = cute.make_identity_tensor((_M, N))
    t_cc_c = thr_mma.partition_C(c_identity)
    t_rc_c_all = thr_tmem.partition_D(t_cc_c)
    t_rt_acc0 = t_rt_acc0_all[(None, 0, None, None)]
    t_rt_acc1 = t_rt_acc1_all[(None, 0, None, None)]
    t_rc_c = t_rc_c_all[(None, 0, None, None)]
    r_base = cute.make_rmem_tensor(t_rc_c.shape, _ACC)

    # Start the first checkpoint before the independent operand preparation.
    # The second head enters the same two-stage pipeline after Phase 1.
    num_k_tiles = cute.size(g_state0, mode=[2])
    if warp == 0:
        for k_idx in cutlass.range(num_k_tiles, unroll=1):
            empty = state_producer.acquire_and_advance()
            cute.copy(
                tma_atom_state,
                t_sg_state0[(None, k_idx)],
                t_ss_state[(None, empty.index)],
                tma_bar_ptr=empty.barrier,
            )

    # Zero unused MMA columns for partial tiles without reading padded tokens.
    if cutlass.const_expr(4 < T < 8):
        if tidx < _M:
            for column in cutlass.range_constexpr(2 * T, N):
                s_operand_logical[(column, tidx % _K_TILE, tidx // _K_TILE)] = 0.0

    # Phase 1: one of the eight warps owns each timestep.  Q/K are shared by
    # both adjacent HV heads; V and gate metadata remain head-specific.
    q_reg_bf16 = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.BFloat16)
    k_reg_bf16 = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.BFloat16)
    q_reg = cute.make_rmem_tensor(cute.make_layout((4,)), _ACC)
    k_reg = cute.make_rmem_tensor(cute.make_layout((4,)), _ACC)
    i_t = warp
    if i_t < T:
        q_tile = cute.local_tile(q, (1, 1, 1, 4), (i_n, i_t, i_h, lane))
        k_tile = cute.local_tile(k, (1, 1, 1, 4), (i_n, i_t, i_h, lane))
        cute.autovec_copy(q_tile[(0, 0, 0, None)], q_reg_bf16)
        cute.autovec_copy(k_tile[(0, 0, 0, None)], k_reg_bf16)
        sum_q = cutlass.Float32(0.0)
        sum_k = cutlass.Float32(0.0)
        for elem in cutlass.range_constexpr(4):
            q_reg[elem] = cutlass.Float32(q_reg_bf16[elem])
            k_reg[elem] = cutlass.Float32(k_reg_bf16[elem])
            sum_q = sum_q + q_reg[elem] * q_reg[elem]
            sum_k = sum_k + k_reg[elem] * k_reg[elem]
        for offset in [16, 8, 4, 2, 1]:
            sum_q = sum_q + cute.arch.shuffle_sync_bfly(
                sum_q, offset=offset, mask=-1, mask_and_clamp=31
            )
            sum_k = sum_k + cute.arch.shuffle_sync_bfly(
                sum_k, offset=offset, mask=-1, mask_and_clamp=31
            )
        q_norm = cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
        k_norm = cute.rsqrt(sum_k + 1e-6, fastmath=True)
        for elem in cutlass.range_constexpr(4):
            k_global = lane * 4 + elem
            stage = k_global // _K_TILE
            k_local = k_global % _K_TILE
            s_operand_logical[(i_t, k_local, stage)] = k_reg[elem] * k_norm
            s_operand_logical[(T + i_t, k_local, stage)] = q_reg[elem] * q_norm
            s_rawk[(i_t, k_global)] = k_reg_bf16[elem]

        for v_chunk in cutlass.range_constexpr(4):
            v_idx = v_chunk * 32 + lane
            s_values[(0, i_t, v_idx)] = v[(i_n, i_t, i_hv0, v_idx)]
            if cutlass.const_expr(HEADS_PER_CTA == 2):
                s_values[(1, i_t, v_idx)] = v[(i_n, i_t, i_hv1, v_idx)]

        if lane == 0:
            for head in cutlass.range_constexpr(HEADS_PER_CTA):
                i_hv = i_hv0 + head
                x = cutlass.Float32(a[(i_n, i_t, i_hv)]) + cutlass.Float32(
                    dt_bias[i_hv]
                )
                # The public call fixes beta=1 and threshold=20.  Preserve the
                # threshold branch so large positive inputs never overflow exp.
                softplus_x = x
                if x <= 20.0:
                    softplus_x = cute.log(
                        cutlass.Float32(1.0) + cute.exp(x, fastmath=True),
                        fastmath=True,
                    )
                log_decay = (
                    -cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=True) * softplus_x
                )
                beta_value = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0)
                    + cute.exp(-cutlass.Float32(b[(i_n, i_t, i_hv)]), fastmath=True)
                )
                s_log_decay[(head, i_t)] = log_decay
                s_decay[(head, i_t)] = cute.exp(log_decay, fastmath=True)
                s_beta[(head, i_t)] = beta_value

    # Publish scalar SMEM stores to the asynchronous MMA proxy.
    cute.arch.fence_view_async_shared()
    cute.arch.barrier()

    # Phase 2: compute the tiny K/K and K/Q Gram matrices while the state
    # transfer is in flight.
    if warp == 2:
        operand_full = cute.group_modes(s_operand_logical, 1, 3)
        gram_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaTF32Op((16, 8, 8)),
            cute.make_layout((1, 1, 1)),
        )
        gram_thr_mma = gram_mma.get_slice(lane)
        gram_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), _ACC, num_bits_per_copy=32
        )
        gram_copy_a = cute.make_tiled_copy_A(gram_copy_atom, gram_mma)
        gram_copy_b = cute.make_tiled_copy_B(gram_copy_atom, gram_mma)
        gram_copy_c = cute.make_tiled_copy_C(gram_copy_atom, gram_mma)
        gram_thr_a = gram_copy_a.get_slice(lane)
        gram_thr_b = gram_copy_b.get_slice(lane)
        gram_thr_c = gram_copy_c.get_slice(lane)

        gram_a = cute.local_tile(operand_full, (16, 128), (0, 0))
        gram_b0 = cute.local_tile(operand_full, (8, 128), (0, 0))
        gram_b0_src = gram_thr_b.partition_S(gram_b0)
        gram_b0_f32 = cute.make_rmem_tensor_like(
            gram_mma.make_fragment_B(gram_thr_mma.partition_B(gram_b0)), _ACC
        )
        cute.copy(gram_copy_b, gram_b0_src, gram_thr_b.retile(gram_b0_f32))
        gram_b0_tf32 = _to_tf32(gram_b0_f32)
        gram_c0 = cute.local_tile(s_gram, (16, 8), (0, 0))
        gram_c0_dst = gram_thr_c.partition_D(gram_c0)
        gram_c0_acc = gram_mma.make_fragment_C(gram_mma.partition_shape_C((16, 8)))
        gram_c0_acc.fill(0.0)

        if cutlass.const_expr(N == 16):
            gram_b1 = cute.local_tile(operand_full, (8, 128), (1, 0))
            gram_b1_src = gram_thr_b.partition_S(gram_b1)
            gram_b1_f32 = cute.make_rmem_tensor_like(
                gram_mma.make_fragment_B(gram_thr_mma.partition_B(gram_b1)), _ACC
            )
            cute.copy(gram_copy_b, gram_b1_src, gram_thr_b.retile(gram_b1_f32))
            gram_b1_tf32 = _to_tf32(gram_b1_f32)
            gram_c1 = cute.local_tile(s_gram, (16, 8), (0, 1))
            gram_c1_dst = gram_thr_c.partition_D(gram_c1)
            gram_c1_acc = gram_mma.make_fragment_C(gram_mma.partition_shape_C((16, 8)))
            gram_c1_acc.fill(0.0)

        for k_group in cutlass.range_constexpr(2):
            gram_a_group = cute.local_tile(gram_a, (16, 64), (0, k_group))
            gram_a_src = gram_thr_a.partition_S(gram_a_group)
            gram_a_f32 = cute.make_rmem_tensor_like(
                gram_mma.make_fragment_A(gram_thr_mma.partition_A(gram_a_group)),
                _ACC,
            )
            cute.copy(gram_copy_a, gram_a_src, gram_thr_a.retile(gram_a_f32))
            gram_a_tf32 = _to_tf32(gram_a_f32)
            for k_tile in cutlass.range_constexpr(8):
                global_k_tile = k_group * 8 + k_tile
                cute.gemm(
                    gram_mma,
                    gram_c0_acc,
                    gram_a_tf32[None, None, k_tile],
                    gram_b0_tf32[None, None, global_k_tile],
                    gram_c0_acc,
                )
                if cutlass.const_expr(N == 16):
                    cute.gemm(
                        gram_mma,
                        gram_c1_acc,
                        gram_a_tf32[None, None, k_tile],
                        gram_b1_tf32[None, None, global_k_tile],
                        gram_c1_acc,
                    )
        cute.copy(gram_copy_c, gram_thr_c.retile(gram_c0_acc), gram_c0_dst)
        if cutlass.const_expr(N == 16):
            cute.copy(gram_copy_c, gram_thr_c.retile(gram_c1_acc), gram_c1_dst)

    # Refill the same two-stage state pipeline with the adjacent HV head while
    # the consumer drains head 0.  Q/K and the Gram matrix are reused.
    if warp == 0 and cutlass.const_expr(HEADS_PER_CTA == 2):
        for k_idx in cutlass.range(num_k_tiles, unroll=1):
            empty = state_producer.acquire_and_advance()
            cute.copy(
                tma_atom_state,
                t_sg_state1[(None, k_idx)],
                t_ss_state[(None, empty.index)],
                tma_bar_ptr=empty.barrier,
            )

    # Four warps flush both heads' ReplaySSM payloads while warp 1 executes the
    # state MMAs.  Each worker owns one K/V element, retaining all logical IO.
    if warp >= 3 and valid:
        replay_worker = tidx - 96
        if replay_worker < _M:
            for head in cutlass.range_constexpr(HEADS_PER_CTA):
                i_hv = i_hv0 + head
                for i_t in cutlass.range_constexpr(T):
                    replayssm_rawv[(safe_cache_idx, i_hv, i_t, replay_worker)] = (
                        s_values[(head, i_t, replay_worker)]
                    )
            if i_hv0 % (HV // H) == 0:
                for i_t in cutlass.range_constexpr(T):
                    replayssm_rawk[(safe_cache_idx, i_h, i_t, replay_worker)] = s_rawk[
                        (i_t, replay_worker)
                    ]
        if replay_worker < HEADS_PER_CTA * T:
            head = replay_worker // T
            i_t = replay_worker % T
            i_hv = i_hv0 + head
            replayssm_g[(safe_cache_idx, i_hv, i_t)] = s_log_decay[(head, i_t)]
            replayssm_beta[(safe_cache_idx, i_hv, i_t)] = s_beta[(head, i_t)]

    if warp == 1:
        acc_empty = acc_producer.acquire_and_advance()
        for head in cutlass.range_constexpr(HEADS_PER_CTA):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _ in cutlass.range(num_k_tiles, unroll=1):
                full = state_consumer.wait_and_advance()
                for k_block in cutlass.range_constexpr(cute.size(t_cr_state, mode=[2])):
                    coord = (None, None, k_block, full.index)
                    if cutlass.const_expr(head == 0):
                        cute.gemm(
                            tiled_mma,
                            t_ct_acc0,
                            t_cr_state[coord],
                            t_cr_operand[coord],
                            t_ct_acc0,
                        )
                    else:
                        cute.gemm(
                            tiled_mma,
                            t_ct_acc1,
                            t_cr_state[coord],
                            t_cr_operand[coord],
                            t_ct_acc1,
                        )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                full.release()
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_consumer.wait_and_advance()
    if epilogue_head == 0:
        cute.copy(tmem_copy, t_rt_acc0, r_base)
    elif cutlass.const_expr(HEADS_PER_CTA == 2):
        cute.copy(tmem_copy, t_rt_acc1, r_base)
    cute.arch.barrier()

    # Phase 3: one thread owns one V row.  The recurrence is only O(T^2)
    # scalars and executes while other CTAs are still streaming state.
    if epilogue_head < HEADS_PER_CTA:
        m_idx = t_rc_c[0][0]
        coeff = cute.make_rmem_tensor(cute.make_layout((8,)), _ACC)
        coeff.fill(0.0)
        state_scale = cutlass.Float32(1.0)
        for i_t in cutlass.range_constexpr(T):
            gate = s_decay[(epilogue_head, i_t)]
            state_scale = state_scale * gate
            prediction = state_scale * r_base[i_t]
            for j in cutlass.range_constexpr(i_t):
                coeff[j] = coeff[j] * gate
                prediction = prediction + coeff[j] * s_gram[(j, i_t)]
            delta = (
                cutlass.Float32(s_values[(epilogue_head, i_t, m_idx)]) - prediction
            ) * s_beta[(epilogue_head, i_t)]
            coeff[i_t] = delta
            result = state_scale * r_base[T + i_t]
            for j in cutlass.range_constexpr(i_t + 1):
                result = result + coeff[j] * s_gram[(j, T + i_t)]
            if valid:
                output[(i_n, i_t, i_hv0 + epilogue_head, m_idx)] = cutlass.BFloat16(
                    result
                )

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def _launch_tcgen(
    state: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    output: cute.Tensor,
    state_indices: cute.Tensor,
    replayssm_rawv: cute.Tensor,
    replayssm_rawk: cute.Tensor,
    replayssm_g: cute.Tensor,
    replayssm_beta: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    HEADS_PER_CTA: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    n_tile = 8 if cutlass.const_expr(T == 4) else 16
    op = tcgen05.MmaTF32Op(
        (_M, n_tile, 8),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    tiler = (_M, n_tile, _K_TILE)
    state_layout = sm100_utils.make_smem_layout_a(tiled_mma, tiler, _ACC, _STAGES)
    operand_layout = sm100_utils.make_smem_layout_b(tiled_mma, tiler, _ACC, _STAGES)
    state_vkl = cute.make_tensor(
        state.iterator, cute.select(state.layout, mode=[1, 2, 0])
    )
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom, tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        state_vkl,
        cute.select(state_layout, mode=[0, 1, 2]),
        tiler,
        tiled_mma,
    )
    gdn_verify_kernel_mtp_replayssm_tcgen(
        tiled_mma,
        tma_atom,
        tma_tensor,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        state_indices,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        state_layout,
        operand_layout,
        scale,
        T,
        n_tile,
        H,
        HV,
        HEADS_PER_CTA,
    ).launch(
        grid=(q.shape[0] * (HV // HEADS_PER_CTA), 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
    )


_CACHE: dict = {}


def _slot_dynamic(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tuple(range(tensor.dim())),
        divisibility=1,
    )


def run_gdn_verify_kernel_mtp_replayssm_tcgen(
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    state_indices: torch.Tensor,
    replayssm_rawv: torch.Tensor,
    replayssm_rawk: torch.Tensor,
    replayssm_g: torch.Tensor,
    replayssm_beta: torch.Tensor,
    T: int,
    H: int,
    HV: int,
    scale: float,
) -> None:
    target = gdn_device_target(state.device)
    if target.major != 10:
        raise ValueError("tcgen05 ReplaySSM verify requires SM100/SM103")
    heads_per_cta = 1 if q.shape[0] == 1 else 2
    # mark_layout_dynamic preserves the inferred unit-stride dimension.
    # Include it so sliced/transposed views do not reuse an incompatible ABI.
    layout_key = tuple(
        tuple(i for i, stride in enumerate(t.stride()) if stride == 1)
        for t in (A_log, a, dt_bias, q, k, v, b, output, state_indices)
    )
    key = (
        T,
        H,
        HV,
        scale,
        target.compile_key,
        A_log.dtype,
        dt_bias.dtype,
        state_indices.dtype,
        heads_per_cta,
        layout_key,
    )
    cache = _CACHE.setdefault(key, {})
    stream = cuda.CUstream(torch.cuda.current_stream(device=state.device).cuda_stream)
    compile_options = gdn_compile_options(
        state.device, cute.EnableTVMFFI(True), cute.GenerateLineInfo(True)
    )
    if "compiled" not in cache:
        state_cute = from_dlpack(state, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        compiled = build_and_load_cute_dsl_kernel(
            "gdn_replayssm_verify",
            make_kernel_name("verify", target.arch, *key[:4], *key[5:]),
            lambda: cute.compile[compile_options](
                _launch_tcgen,
                state_cute,
                from_dlpack(A_log, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(a, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(q, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(k, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(v, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(b, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(output, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(state_indices, assumed_align=16).mark_layout_dynamic(),
                _slot_dynamic(replayssm_rawv),
                _slot_dynamic(replayssm_rawk),
                _slot_dynamic(replayssm_g),
                _slot_dynamic(replayssm_beta),
                scale=scale,
                T=T,
                H=H,
                HV=HV,
                HEADS_PER_CTA=heads_per_cta,
                stream=stream,
            ),
            extra_key_files=(__file__,),
        )
        cache["compiled"] = compiled
    cache["compiled"](
        state,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        state_indices,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        stream,
    )
