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

"""SM120 ReplaySSM verify using Ampere-style TF32 ``mma.sync``.

Each head's frozen FP32 128x128 state is read once, split by V rows across
``v_split`` CTAs.  In each CTA, one warp per 16-row state tile computes

    state[rows, 128] @ [K0..K(T-1), Q0..Q(T-1)].T

with ``mma.sync.m16n8k8.f32.tf32.tf32.f32`` while another warp computes the
small operand Gram matrix.  The final recurrent result is reconstructed from
those base products and the rank-one update coefficients, so the kernel never
materializes or repeatedly updates an intermediate state.

This specialization deliberately uses warp MMA rather than tcgen05: SM120
supports the former and does not support the latter.
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from .cute_dsl_cache_naming import make_kernel_name
from .device_target import gdn_compile_options, gdn_device_target
import cuda.bindings.driver as cuda


_ACC = cutlass.Float32
_M = 128
_K = 128
_MAX_T = 8
_THREADS = 256
_GRAM_WARP = 4


@cute.jit
def _to_tf32(fragment: cute.Tensor) -> cute.Tensor:
    """Round one FP32 register fragment to PTX TF32 operand bits."""
    values = fragment.load()
    result = cute.make_rmem_tensor_like(fragment, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(fragment)):
        result[i] = cute.arch.cvt_f32_tf32(values[i])
    return result


@cute.jit
def _operand_column(k_idx):
    """Return the MMA K column that holds true K index ``k_idx``.

    A dot product is invariant under a consistent permutation of K.  Within
    each 64-wide K group, lane t loads true state columns [16t, 16t + 16) as
    one contiguous run; element j of that run feeds column t + 4 * (j % 2) of
    k-tile j // 2.  K/Q are stored at the same columns, so both the state
    products and the K/Q Gram matrix are unchanged.
    """
    k_group = k_idx // 64
    p = k_idx % 64
    j = p % 16
    return k_group * 64 + (j // 2) * 8 + p // 16 + 4 * (j % 2)


@cute.kernel
def gdn_verify_kernel_mtp_replayssm_hmma(
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
    V_SPLIT: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32

    # V_SPLIT adjacent CTAs share one head and each owns a contiguous slice of
    # V rows, one 16-row state tile per state warp.  A single CTA per head
    # leaves too few state loads in flight unless B * HV is large; splitting
    # repeats only the cheap Q/K normalization and Gram work per slice.
    rows = _M // V_SPLIT
    state_warps = rows // 16
    i_split = block % V_SPLIT
    head_block = block // V_SPLIT
    i_hv = head_block % HV
    i_n = head_block // HV
    row0 = i_split * rows
    heads_per_q = HV // H
    i_h = i_hv // heads_per_q
    cache_idx = state_indices[i_n]
    valid = cache_idx >= 0
    safe_cache_idx = cache_idx
    if not valid:
        safe_cache_idx = 0
    state_flat = safe_cache_idx * HV + i_hv
    state_matrix = state[(state_flat, None, None)]

    smem = utils.SmemAllocator()
    # K0..K(T-1) followed by Q0..Q(T-1), all normalized FP32 and K
    # contiguous.  Keep the maximum shape so T=4 and T=8 share the same
    # simple warp-MMA layouts.
    s_operand = smem.allocate_tensor(
        _ACC,
        cute.make_layout((2 * _MAX_T, _K), stride=(_K, 1)),
        16,
    )
    s_values = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((_MAX_T, _M), stride=(_M, 1)),
        16,
    )
    s_decay = smem.allocate_tensor(_ACC, cute.make_layout((_MAX_T,), stride=(1,)), 16)
    s_beta = smem.allocate_tensor(_ACC, cute.make_layout((_MAX_T,), stride=(1,)), 16)
    s_gram = smem.allocate_tensor(
        _ACC,
        cute.make_layout((2 * _MAX_T, 2 * _MAX_T), stride=(2 * _MAX_T, 1)),
        16,
    )
    s_base = smem.allocate_tensor(
        _ACC,
        # The epilogue has one adjacent thread per V row.  Column-major
        # physical storage makes its fixed-column reads conflict-free.
        cute.make_layout((_M, 2 * _MAX_T), stride=(1, _M)),
        16,
    )

    # Warp 4 always evaluates an m16 Gram tile.  For T<8, initialize the
    # compile-time-known padding rows [2*T, 16) to benign zeros.  T=8 emits
    # neither this loop nor any associated instructions.
    if cutlass.const_expr(T < _MAX_T):
        for zero_iter in cutlass.range_constexpr(_MAX_T - T):
            linear = tidx + zero_iter * _THREADS
            s_operand[(2 * T + linear // _K, linear % _K)] = 0.0

    # Each warp owns one timestep.  This also keeps all ReplaySSM stores
    # naturally coalesced and overlaps them with the input normalization.
    i_t = warp
    q_reg_bf16 = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.BFloat16)
    k_reg_bf16 = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.BFloat16)
    q_reg = cute.make_rmem_tensor(cute.make_layout((4,)), _ACC)
    k_reg = cute.make_rmem_tensor(cute.make_layout((4,)), _ACC)
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
            k_idx = lane * 4 + elem
            # Store K/Q at the MMA column whose state element the A loads below
            # place there (see _operand_column).
            k_col = _operand_column(k_idx)
            s_operand[(i_t, k_col)] = k_reg[elem] * k_norm
            s_operand[(T + i_t, k_col)] = q_reg[elem] * q_norm
            if valid and i_split == 0 and i_hv % heads_per_q == 0:
                replayssm_rawk[(safe_cache_idx, i_h, i_t, k_idx)] = k_reg_bf16[elem]

        for v_chunk in cutlass.range_constexpr(rows // 32):
            v_idx = row0 + v_chunk * 32 + lane
            value = v[(i_n, i_t, i_hv, v_idx)]
            s_values[(i_t, v_idx)] = value
            if valid:
                replayssm_rawv[(safe_cache_idx, i_hv, i_t, v_idx)] = value

        if lane == 0:
            x = cutlass.Float32(a[(i_n, i_t, i_hv)]) + cutlass.Float32(dt_bias[i_hv])
            softplus_x = x
            if x <= 20.0:
                softplus_x = cute.log(
                    cutlass.Float32(1.0) + cute.exp(x, fastmath=True),
                    fastmath=True,
                )
            log_decay = (
                -cute.exp(cutlass.Float32(A_log[i_hv]), fastmath=True) * softplus_x
            )
            decay = cute.exp(log_decay, fastmath=True)
            beta_value = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0)
                + cute.exp(-cutlass.Float32(b[(i_n, i_t, i_hv)]), fastmath=True)
            )
            s_decay[i_t] = decay
            s_beta[i_t] = beta_value
            if valid and i_split == 0:
                replayssm_g[(safe_cache_idx, i_hv, i_t)] = log_decay
                replayssm_beta[(safe_cache_idx, i_hv, i_t)] = beta_value

    cute.arch.barrier()

    hmma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaTF32Op((16, 8, 8)),
        cute.make_layout((1, 1, 1)),
    )
    thr_mma = hmma.get_slice(lane)
    # m16n8k8 TF32 operand coordinates: lane holds A rows g and g+8 and the
    # K columns t and t+4 of each k-tile.
    mma_row = lane // 4
    mma_col = lane % 4
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), _ACC, num_bits_per_copy=32
    )
    copy_a = cute.make_tiled_copy_A(copy_atom, hmma)
    copy_b = cute.make_tiled_copy_B(copy_atom, hmma)
    copy_c = cute.make_tiled_copy_C(copy_atom, hmma)
    thr_a = copy_a.get_slice(lane)
    thr_b = copy_b.get_slice(lane)
    thr_c = copy_c.get_slice(lane)

    # Each state warp owns one 16-row state tile.  Stream the operand in K=64
    # groups to shorten the register live range of the B fragments.
    if warp < state_warps:
        row_tile = row0 // 16 + warp
        acc0 = hmma.make_fragment_C(hmma.partition_shape_C((16, 8)))
        acc0.fill(0.0)
        if cutlass.const_expr(T > 4):
            acc1 = hmma.make_fragment_C(hmma.partition_shape_C((16, 8)))
            acc1.fill(0.0)
        for k_group in cutlass.range_constexpr(2):
            operand_b0 = cute.local_tile(s_operand, (8, 64), (0, k_group))
            b0_f32 = cute.make_rmem_tensor_like(
                hmma.make_fragment_B(thr_mma.partition_B(operand_b0)), _ACC
            )
            cute.copy(copy_b, thr_b.partition_S(operand_b0), thr_b.retile(b0_f32))
            b0_tf32 = _to_tf32(b0_f32)
            if cutlass.const_expr(T > 4):
                operand_b1 = cute.local_tile(s_operand, (8, 64), (1, k_group))
                b1_f32 = cute.make_rmem_tensor_like(
                    hmma.make_fragment_B(thr_mma.partition_B(operand_b1)), _ACC
                )
                cute.copy(
                    copy_b,
                    thr_b.partition_S(operand_b1),
                    thr_b.retile(b1_f32),
                )
                b1_tf32 = _to_tf32(b1_f32)

            # Each lane reads 16 contiguous state columns from rows g and g+8
            # with 128-bit loads instead of the MMA layout's scattered 32-bit
            # elements; _operand_column permutes K/Q to match.
            state_a = cute.local_tile(state_matrix, (16, 64), (row_tile, k_group))
            a_f32 = cute.make_rmem_tensor_like(
                hmma.make_fragment_A(thr_mma.partition_A(state_a)), _ACC
            )
            for row_half in cutlass.range_constexpr(2):
                a_row = cute.make_rmem_tensor(cute.make_layout((1, 16)), _ACC)
                cute.autovec_copy(
                    cute.local_tile(
                        state_matrix,
                        (1, 16),
                        (row_tile * 16 + row_half * 8 + mma_row, k_group * 4 + mma_col),
                    ),
                    a_row,
                )
                for j in cutlass.range_constexpr(16):
                    # Fragment index: row half + 2 * column half + 4 * k-tile.
                    a_f32[row_half + 2 * (j % 2) + 4 * (j // 2)] = a_row[(0, j)]
            a_tf32 = _to_tf32(a_f32)
            for k_tile in cutlass.range_constexpr(8):
                cute.gemm(
                    hmma,
                    acc0,
                    a_tf32[None, None, k_tile],
                    b0_tf32[None, None, k_tile],
                    acc0,
                )
                if cutlass.const_expr(T > 4):
                    cute.gemm(
                        hmma,
                        acc1,
                        a_tf32[None, None, k_tile],
                        b1_tf32[None, None, k_tile],
                        acc1,
                    )

        base0 = cute.local_tile(s_base, (16, 8), (row_tile, 0))
        cute.copy(copy_c, thr_c.retile(acc0), thr_c.partition_D(base0))
        if cutlass.const_expr(T > 4):
            base1 = cute.local_tile(s_base, (16, 8), (row_tile, 1))
            cute.copy(copy_c, thr_c.retile(acc1), thr_c.partition_D(base1))

    # Warp 4 computes the 16x16 K/K, K/Q, Q/K and Q/Q Gram matrix in parallel
    # with the state warps.  Only its upper-left K rows and Q columns are used
    # by the recurrence, but the full two MMA column tiles are equally cheap.
    if warp == _GRAM_WARP:
        gram_a = cute.local_tile(s_operand, (16, _K), (0, 0))
        gram_acc0 = hmma.make_fragment_C(hmma.partition_shape_C((16, 8)))
        gram_acc0.fill(0.0)
        if cutlass.const_expr(T > 4):
            gram_acc1 = hmma.make_fragment_C(hmma.partition_shape_C((16, 8)))
            gram_acc1.fill(0.0)
        for k_group in cutlass.range_constexpr(2):
            gram_b0 = cute.local_tile(s_operand, (8, 64), (0, k_group))
            gram_b0_f32 = cute.make_rmem_tensor_like(
                hmma.make_fragment_B(thr_mma.partition_B(gram_b0)), _ACC
            )
            cute.copy(
                copy_b,
                thr_b.partition_S(gram_b0),
                thr_b.retile(gram_b0_f32),
            )
            gram_b0_tf32 = _to_tf32(gram_b0_f32)
            if cutlass.const_expr(T > 4):
                gram_b1 = cute.local_tile(s_operand, (8, 64), (1, k_group))
                gram_b1_f32 = cute.make_rmem_tensor_like(
                    hmma.make_fragment_B(thr_mma.partition_B(gram_b1)), _ACC
                )
                cute.copy(
                    copy_b,
                    thr_b.partition_S(gram_b1),
                    thr_b.retile(gram_b1_f32),
                )
                gram_b1_tf32 = _to_tf32(gram_b1_f32)
            gram_a_group = cute.local_tile(gram_a, (16, 64), (0, k_group))
            gram_a_src = thr_a.partition_S(gram_a_group)
            gram_a_f32 = cute.make_rmem_tensor_like(
                hmma.make_fragment_A(thr_mma.partition_A(gram_a_group)), _ACC
            )
            cute.copy(copy_a, gram_a_src, thr_a.retile(gram_a_f32))
            gram_a_tf32 = _to_tf32(gram_a_f32)
            for k_tile in cutlass.range_constexpr(8):
                cute.gemm(
                    hmma,
                    gram_acc0,
                    gram_a_tf32[None, None, k_tile],
                    gram_b0_tf32[None, None, k_tile],
                    gram_acc0,
                )
                if cutlass.const_expr(T > 4):
                    cute.gemm(
                        hmma,
                        gram_acc1,
                        gram_a_tf32[None, None, k_tile],
                        gram_b1_tf32[None, None, k_tile],
                        gram_acc1,
                    )
        gram0 = cute.local_tile(s_gram, (16, 8), (0, 0))
        cute.copy(copy_c, thr_c.retile(gram_acc0), thr_c.partition_D(gram0))
        if cutlass.const_expr(T > 4):
            gram1 = cute.local_tile(s_gram, (16, 8), (0, 1))
            cute.copy(copy_c, thr_c.retile(gram_acc1), thr_c.partition_D(gram1))

    cute.arch.barrier()

    # One thread owns one V row.  O(T^2) scalar recurrence replaces eight
    # full-state rank-one updates and consumes only CTA-local base products.
    if tidx < rows:
        row = row0 + tidx
        coeff = cute.make_rmem_tensor(cute.make_layout((_MAX_T,)), _ACC)
        coeff.fill(0.0)
        state_scale = cutlass.Float32(1.0)
        for t in cutlass.range_constexpr(T):
            gate = s_decay[t]
            state_scale = state_scale * gate
            prediction = state_scale * s_base[(row, t)]
            for j in cutlass.range_constexpr(t):
                coeff[j] = coeff[j] * gate
                prediction = prediction + coeff[j] * s_gram[(j, t)]
            delta = (cutlass.Float32(s_values[(t, row)]) - prediction) * s_beta[t]
            coeff[t] = delta
            result = state_scale * s_base[(row, T + t)]
            for j in cutlass.range_constexpr(t + 1):
                result = result + coeff[j] * s_gram[(j, T + t)]
            if valid:
                output[(i_n, t, i_hv, row)] = cutlass.BFloat16(result)


@cute.jit
def _launch_hmma(
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
    V_SPLIT: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    gdn_verify_kernel_mtp_replayssm_hmma(
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
        scale,
        T,
        H,
        HV,
        V_SPLIT,
    ).launch(
        grid=(q.shape[0] * HV * V_SPLIT, 1, 1),
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


def run_gdn_verify_kernel_mtp_replayssm_hmma(
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
    v_split: int,
) -> None:
    target = gdn_device_target(state.device)
    if (target.major, target.minor) != (12, 0):
        raise ValueError("HMMA ReplaySSM verify requires SM120")
    if not 4 <= T <= 8:
        raise ValueError(f"HMMA ReplaySSM verify supports T=4..8, got {T}")
    if v_split not in (2, 4):
        raise ValueError(
            f"HMMA ReplaySSM verify supports v_split 2 or 4, got {v_split}"
        )
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
        v_split,
        A_log.dtype,
        dt_bias.dtype,
        state_indices.dtype,
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
        cache["compiled"] = build_and_load_cute_dsl_kernel(
            "gdn_replayssm_verify_hmma",
            make_kernel_name("verify_hmma", target.arch, *key[:4], *key[5:]),
            lambda: cute.compile[compile_options](
                _launch_hmma,
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
                V_SPLIT=v_split,
                stream=stream,
            ),
            extra_key_files=(__file__,),
        )
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
