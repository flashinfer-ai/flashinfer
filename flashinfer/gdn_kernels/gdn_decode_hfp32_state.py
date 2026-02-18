"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Gated Delta Rule Decode Kernel (FP32 Hidden State) - CuTe-DSL Implementation
============================================================================

High-performance CUDA kernel implementing the Gated Delta Rule linear attention
mechanism for decode-phase inference with FP32 hidden state (T=1 only).

Key Features:
- FP32 hidden state (h_state) for higher numerical precision
- Pretranspose state layout: [B, HV, V, K]
- T=1 only (single token decode)
- BS32 optimizations with fma_packed_f32x2
- 8 CTAs per head, TILE_V=8, vec_size=4
- L2-normalized Q/K with configurable scale
- Gated exponential decay of hidden state H via softplus
- Delta rule updates: v_delta = beta * (v - pred)
- BF16 Q/K/V inputs with FP32 compute for numerical stability
- GQA (grouped-query attention) support with configurable H (query) and HV (value) heads

Optimizations:
- fma_packed_f32x2 for inner loop (2 FMAs per instruction)
- Vectorized V loading via autovec_copy (BF16 -> FP32)
- Direct output write to GMEM (no sOutput SMEM, no final barrier)
- TVM-FFI enabled for reduced launch overhead
- cp.async prefetch issued BEFORE Q/K/V loads to overlap H tile GMEM fetch
- Early gate GMEM reads (before barrier, matching FI ordering)
- Pre-computed gDst tile outside main loop
- Same architecture: 8 CTAs/head, TILE_V=8, vec_size=4, 100% occupancy
"""

import torch
import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass import utils
import cuda.bindings.driver as cuda


# ==============================================================================
# CONSTANTS
# ==============================================================================
TILE_V = 8  # V rows per tile
TILE_K = 128  # Full K dimension
NUM_STAGES = 2
NUM_CTAS_PER_HEAD = 8
VEC_SIZE = 4  # Each lane handles 4 K elements


@cute.kernel
def _gated_delta_rule_kernel_seq1_hfp32(
    gH: cute.Tensor,  # [B*HV, V, K] pretranspose layout
    gQ: cute.Tensor,  # [B, T, H, K]
    gK: cute.Tensor,  # [B, T, H, K]
    gV: cute.Tensor,  # [B, T, HV, V]
    ga: cute.Tensor,  # [B, T, HV]
    gb: cute.Tensor,  # [B, T, HV]
    gA_log: cute.Tensor,  # [HV]
    gdt_bias: cute.Tensor,  # [HV]
    gO: cute.Tensor,  # [B, T, HV, V]
    scale: cutlass.Constexpr[float],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tiled_copy_load: cute.TiledCopy,
    smem_layout_staged: cute.Layout,
):
    """
    FlashInfer-style kernel: 32 active lanes, vec_size=4, sequential V rows.
    Grid: B * HV * NUM_CTAS_PER_HEAD
    Block: 128 threads (4 warps)
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    batch_idx = bidx // NUM_CTAS_PER_HEAD
    batch_inner = bidx % NUM_CTAS_PER_HEAD
    num_v_tiles_per_block = num_v_tiles // NUM_CTAS_PER_HEAD

    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    smem = utils.SmemAllocator()

    # SMEM for H tiles (staged): [TILE_V, TILE_K, NUM_STAGES]
    # Use S<2,4,3> swizzle for FP32 to eliminate shared memory bank conflicts
    # (4-way conflicts from stride-4 access pattern across 128-element rows)
    smem_swizzle = cute.make_swizzle(2, 4, 3)
    sData = smem.allocate_tensor(
        cutlass.Float32, smem_layout_staged, 128, swizzle=smem_swizzle
    )

    # SMEM for v values
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    # Register tensors: vec_size=4 elements each
    r_h = cute.make_rmem_tensor(
        cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.Float32
    )

    k_start = lane_id * VEC_SIZE

    # Read gate-related values from GMEM EARLY (before barrier)
    # All lanes read (avoids divergent global loads)
    r_A_log = cutlass.Float32(gA_log[i_hv])
    r_a = cutlass.Float32(ga[(i_n, 0, i_hv)])
    r_dt_bias = cutlass.Float32(gdt_bias[i_hv])
    r_b = cutlass.Float32(gb[(i_n, 0, i_hv)])

    cute.arch.barrier()

    # ===================================================================
    # Setup H global tensors and issue cp.async prefetch FIRST
    # This overlaps H tile GMEM fetch with subsequent Q/K/V setup compute
    # ===================================================================
    gH_batch = gH[(batch_idx, None, None)]  # [V, K]
    gSrc = cute.local_tile(
        gH_batch, (TILE_V, TILE_K), (None, 0)
    )  # [TILE_V, TILE_K, num_v_tiles]

    # Pre-compute gDst tile outside loop (match FI's 4D local_tile pattern)
    gDst = cute.local_tile(gH, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    thr_copy = tiled_copy_load.get_slice(tidx)

    start_v_tiles = batch_inner * num_v_tiles_per_block
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
    for v_tiles in range(start_v_tiles, start_v_tiles + prefetch_count):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES
        gSrc_tile = gSrc[(None, None, v_tiles)]
        sData_stage = sData[(None, None, stage)]
        thr_gSrc = thr_copy.partition_S(gSrc_tile)
        thr_sData = thr_copy.partition_D(sData_stage)
        cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
        cute.arch.cp_async_commit_group()

    # ===================================================================
    # Load Q and K into registers, normalize
    # (while cp.async H tile fetch is in-flight - free overlap!)
    # ===================================================================
    # BF16 register tensors for vectorized loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((VEC_SIZE,), stride=(1,)), cutlass.BFloat16
    )

    q_tile = cute.local_tile(gQ, (1, 1, 1, VEC_SIZE), (i_n, 0, i_h, lane_id))
    k_tile = cute.local_tile(gK, (1, 1, 1, VEC_SIZE), (i_n, 0, i_h, lane_id))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)

    # Convert BF16 to FP32
    for i in cutlass.range_constexpr(VEC_SIZE):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    # Load V into SMEM (vectorized BF16 load, convert to FP32)
    # Only warp 0 writes to sV to reduce redundant bank-conflicting stores
    # (all warps load the same V data, so only one needs to write)
    v_tile = cute.local_tile(gV, (1, 1, 1, VEC_SIZE), (i_n, 0, i_hv, lane_id))
    cute.autovec_copy(v_tile, r_v_bf16)
    if warp_idx == 0:
        for i in cutlass.range_constexpr(VEC_SIZE):
            sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

    # ===================================================================
    # Compute gate values (g_exp, beta) on lane 0, broadcast via shuffle
    # (Gate GMEM reads already done above, overlapping with cp.async)
    # ===================================================================
    r_g = cutlass.Float32(0.0)
    r_beta = cutlass.Float32(0.0)
    if lane_id == 0:
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = cutlass.Float32(0.0)
        if beta_x <= softplus_threshold:
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0) + exp_beta_x
            softplus_x = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                log_input, fastmath=True
            )
        else:
            softplus_x = x
        r_g_val = -cute.exp(r_A_log, fastmath=True) * softplus_x
        r_g = cute.exp(r_g_val, fastmath=True)
        r_beta = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0) + cute.exp(-r_b, fastmath=True)
        )

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    # L2 normalize Q, K (after gate, matching FI ordering)
    sum_q = cutlass.Float32(0.0)
    sum_k = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(VEC_SIZE):
        sum_q = sum_q + r_q[i] * r_q[i]
        sum_k = sum_k + r_k[i] * r_k[i]

    # Warp-level reduction for L2 norms
    for offset in [16, 8, 4, 2, 1]:
        sum_q = sum_q + cute.arch.shuffle_sync_bfly(
            sum_q, offset=offset, mask=-1, mask_and_clamp=31
        )
        sum_k = sum_k + cute.arch.shuffle_sync_bfly(
            sum_k, offset=offset, mask=-1, mask_and_clamp=31
        )

    inv_norm_q = cute.rsqrt(sum_q + eps, fastmath=True)
    inv_norm_k = cute.rsqrt(sum_k + eps, fastmath=True)
    for i in cutlass.range_constexpr(VEC_SIZE):
        r_q[i] = r_q[i] * inv_norm_q * scale
        r_k[i] = r_k[i] * inv_norm_k

    # Ensure sV writes are visible to all threads
    cute.arch.barrier()

    # ===================================================================
    # Main loop over V tiles
    # ===================================================================
    end_v_tiles = start_v_tiles + num_v_tiles_per_block
    for v_tiles in range(start_v_tiles, end_v_tiles):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES

        # Wait for current stage
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Issue async load for next tile
        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < end_v_tiles:
            next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES
            gSrc_next = gSrc[(None, None, next_v_tiles)]
            sData_next = sData[(None, None, next_stage)]
            thr_gSrc = thr_copy.partition_S(gSrc_next)
            thr_sData = thr_copy.partition_D(sData_next)
            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        # Process rows: 4 rows per iteration (one per warp), TILE_V/4 iterations
        for row in cutlass.range_constexpr(0, TILE_V, 4):
            row_offset = warp_idx
            v_idx = v_tiles * TILE_V + row + row_offset

            # Load h from SMEM using local_tile + autovec_copy
            sData_tile = cute.local_tile(
                sData, (1, VEC_SIZE, 1), (row + row_offset, lane_id, stage)
            )
            cute.autovec_copy(sData_tile, r_h)

            # Decay: h *= g_exp and accumulate pred = dot(h, k) using fma_packed_f32x2
            sum_hk = cutlass.Float32(0.0)
            sum_hk2 = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(0, VEC_SIZE, 2):
                r_h[i] = r_h[i] * r_g
                r_h[i + 1] = r_h[i + 1] * r_g
                sum_hk, sum_hk2 = cute.arch.fma_packed_f32x2(
                    src_a=(r_h[i], r_h[i + 1]),
                    src_b=(r_k[i], r_k[i + 1]),
                    src_c=(sum_hk, sum_hk2),
                )
            sum_hk = sum_hk + sum_hk2

            # Warp reduction for prediction
            for offset in [16, 8, 4, 2, 1]:
                sum_hk = sum_hk + cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                )

            # Delta computation: v_new = (v - pred) * beta
            v_new = sV[v_idx] - sum_hk
            v_new = v_new * r_beta

            # Outer product: h += k * v_new, and accumulate output dot using fma_packed_f32x2
            sum_hq = cutlass.Float32(0.0)
            sum_hq2 = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(0, VEC_SIZE, 2):
                r_h[i], r_h[i + 1] = cute.arch.fma_packed_f32x2(
                    src_a=(r_k[i], r_k[i + 1]),
                    src_b=(v_new, v_new),
                    src_c=(r_h[i], r_h[i + 1]),
                )
                sum_hq, sum_hq2 = cute.arch.fma_packed_f32x2(
                    src_a=(r_h[i], r_h[i + 1]),
                    src_b=(r_q[i], r_q[i + 1]),
                    src_c=(sum_hq, sum_hq2),
                )
            sum_hq = sum_hq + sum_hq2

            # Write updated h directly to GMEM using pre-computed gDst tile
            gDst_tile = cute.local_tile(
                gDst, (1, 1, VEC_SIZE, 1), (0, row + row_offset, lane_id, v_tiles)
            )
            cute.autovec_copy(r_h, gDst_tile)

            # Warp reduction for output
            for offset in [16, 8, 4, 2, 1]:
                sum_hq = sum_hq + cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                )

            # Write output directly to GMEM (no sOutput SMEM needed)
            if lane_id == 0:
                o_out = gO[(i_n, 0, i_hv, None)]
                o_out[v_idx] = cutlass.BFloat16(sum_hq)

    # Output already written directly to GMEM in the inner loop


# ==============================================================================
# LAUNCH WRAPPER
# ==============================================================================


@cute.jit
def _gated_delta_rule_launch_hfp32(
    mH: cute.Tensor,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    ma: cute.Tensor,
    mb: cute.Tensor,
    mA_log: cute.Tensor,
    mdt_bias: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Constexpr[float],
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    num_blocks: cutlass.Int32,
    stream: cuda.CUstream,
):
    # cp.async copy atom
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    thread_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((1, VEC_SIZE))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    # SMEM layout: [TILE_V, TILE_K, NUM_STAGES]
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    _gated_delta_rule_kernel_seq1_hfp32(
        mH,
        mQ,
        mK,
        mV,
        ma,
        mb,
        mA_log,
        mdt_bias,
        mO,
        scale,
        softplus_beta,
        softplus_threshold,
        eps,
        HV,
        H,
        V,
        K,
        num_v_tiles,
        tiled_copy_load,
        smem_layout_staged,
    ).launch(
        grid=[num_blocks, 1, 1],
        block=[128, 1, 1],
        stream=stream,
    )


# ==============================================================================
# PYTHON INTERFACE
# ==============================================================================

# Cache: (B, H, HV, K, V) -> compiled kernel
_compiled_kernels_hfp32 = {}


def gated_delta_rule_hfp32(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: torch.Tensor = None,
    k: torch.Tensor = None,
    v: torch.Tensor = None,
    b: torch.Tensor = None,
    initial_state_source: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: float = None,
) -> torch.Tensor:
    """
    Gated Delta Rule decode kernel with FP32 hidden state.

    This kernel implements the Gated Delta Rule for decode-phase inference
    with FP32 hidden state for higher numerical precision.

    Args:
        A_log: Log decay parameter [HV], dtype=float32
        a: Input-dependent decay [B, T, HV], dtype=bfloat16/float16
        dt_bias: Decay bias [HV], dtype=float32
        softplus_beta: Softplus beta parameter (default: 1.0)
        softplus_threshold: Softplus threshold (default: 20.0)
        q: Query tensor [B, T, H, K], dtype=bfloat16/float16
        k: Key tensor [B, T, H, K], dtype=bfloat16/float16
        v: Value tensor [B, T, HV, V], dtype=bfloat16/float16
        b: Update gate input [B, T, HV], dtype=bfloat16/float16
        initial_state_source: Hidden state [B, HV, V, K], dtype=float32, pretranspose layout
        initial_state_indices: Not used (for API compatibility)
        use_qk_l2norm_in_kernel: Whether to use L2 normalization (must be True)
        scale: Scale factor for Q (default: 1/sqrt(K))

    Returns:
        Output tensor [B, T, HV, V], dtype=bfloat16/float16

    Note:
        - T must be 1 (single token decode only)
        - State is modified in-place
        - State layout is pretranspose: [B, HV, V, K]
    """
    global _compiled_kernels_hfp32

    B, T, H_dim, K_dim = q.shape
    assert T == 1, "gated_delta_rule_hfp32 only supports T=1"
    HV_dim = v.shape[2]
    V_dim = v.shape[3]
    if scale is None:
        scale = 1.0 / math.sqrt(K_dim)

    output = torch.empty(B, T, HV_dim, V_dim, device=q.device, dtype=q.dtype)

    # H needs to be [B*HV, V, K] pretranspose layout
    h_state = initial_state_source  # [B, HV, V, K]
    h_pretrans = h_state.reshape(B * HV_dim, V_dim, K_dim)

    h_ = from_dlpack(h_pretrans, assumed_align=32)
    q_ = from_dlpack(q, assumed_align=32)
    k_ = from_dlpack(k, assumed_align=32)
    v_ = from_dlpack(v, assumed_align=32)
    a_ = from_dlpack(a, assumed_align=32)
    b_ = from_dlpack(b, assumed_align=32)
    A_log_ = from_dlpack(A_log, assumed_align=32)
    dt_bias_ = from_dlpack(dt_bias, assumed_align=32)
    o_ = from_dlpack(output, assumed_align=32)

    num_v_tiles = V_dim // TILE_V  # 128/8 = 16
    num_blocks = cutlass.Int32(B * HV_dim * NUM_CTAS_PER_HEAD)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Cache key includes all shape dimensions to avoid incorrect reuse
    cache_key = (B, H_dim, HV_dim, K_dim, V_dim)
    if cache_key not in _compiled_kernels_hfp32:
        _compiled_kernels_hfp32[cache_key] = cute.compile(
            _gated_delta_rule_launch_hfp32,
            h_,
            q_,
            k_,
            v_,
            a_,
            b_,
            A_log_,
            dt_bias_,
            o_,
            scale,
            softplus_beta,
            softplus_threshold,
            1e-6,
            HV_dim,
            H_dim,
            V_dim,
            K_dim,
            num_v_tiles,
            num_blocks,
            stream,
            options="--generate-line-info",
        )

    _compiled_kernels_hfp32[cache_key](
        h_, q_, k_, v_, a_, b_, A_log_, dt_bias_, o_, num_blocks, stream
    )
    return output


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n{'=' * 60}")
    print(
        "Testing GDN FP32 H State kernel (128 threads, 32 active lanes, vec4, fma_f32x2)"
    )
    print("=" * 60)

    B, T, H, K = 512, 1, 16, 128
    HV, V = 32, 128

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    a = torch.randn(B, T, HV, device=device, dtype=dtype)
    b = torch.randn(B, T, HV, device=device, dtype=dtype)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=torch.float32)
    h_state = torch.randn(B, HV, V, K, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(K)
    eps = 1e-6

    # PyTorch reference
    h_ref = h_state.clone().float()
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    q_norm = q_f / (q_f.norm(dim=-1, keepdim=True) + eps)
    k_norm = k_f / (k_f.norm(dim=-1, keepdim=True) + eps)
    q_scaled = q_norm * scale
    hpg = HV // H
    q_exp = q_scaled.unsqueeze(3).expand(-1, -1, -1, hpg, -1).reshape(B, T, HV, K)
    k_exp = k_norm.unsqueeze(3).expand(-1, -1, -1, hpg, -1).reshape(B, T, HV, K)
    alpha_t = a[:, 0, :].float()
    beta_raw_t = b[:, 0, :].float()
    x_t = alpha_t + dt_bias.unsqueeze(0)
    sp = torch.where(1.0 * x_t <= 20.0, torch.log(1.0 + torch.exp(x_t)), x_t)
    g_t = -torch.exp(A_log.unsqueeze(0)) * sp
    g_exp_t = torch.exp(g_t)
    beta_sig = torch.sigmoid(beta_raw_t)
    h_ref = h_ref * g_exp_t.unsqueeze(-1).unsqueeze(-1)
    k_t = k_exp[:, 0, :, :].float()
    pred_ref = torch.einsum("bhvk,bhk->bhv", h_ref, k_t)
    v_t = v_f[:, 0, :, :]
    delta = (v_t - pred_ref) * beta_sig.unsqueeze(-1)
    h_ref = h_ref + torch.einsum("bhk,bhv->bhvk", k_t, delta)
    q_t = q_exp[:, 0, :, :].float()
    out_ref = torch.einsum("bhvk,bhk->bhv", h_ref, q_t).to(dtype).unsqueeze(1)

    # Kernel
    h_copy = h_state.clone()
    print("Compiling...")
    out = gated_delta_rule_hfp32(
        A_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=h_copy,
        scale=scale,
    )
    torch.cuda.synchronize()
    print("Done.")

    out_match = (out.to(dtype) == out_ref.to(dtype)).float().mean().item()
    h_close = torch.allclose(h_copy.reshape(B, HV, V, K), h_ref, atol=1e-4, rtol=1e-4)
    print(f"  Output BF16 exact match: {out_match * 100:.2f}%")
    print(f"  H state close (atol=1e-2): {h_close}")
    print(
        f"  Output max diff: {(out.float() - out_ref.float()).abs().max().item():.6f}"
    )
    print(
        f"  H max diff: {(h_copy.reshape(B, HV, V, K) - h_ref).abs().max().item():.6f}"
    )
    if out_match > 0.99 and h_close:
        print("  CORRECTNESS PASS!")
    else:
        print("  CORRECTNESS FAIL!")
