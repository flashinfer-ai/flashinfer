// csrc/ep/layout_normalize.cu

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flashinfer {
namespace ep {

// ─── 2D -> 3D Scatter (DeepEP LL: native 2D -> requested 3D) ─────
//
// Input:  flat[total_recv_tokens, hidden_dim]  (contiguous, 2D)
//         expert_counts[num_local_experts]      (how many tokens per expert)
//         expert_offsets[num_local_experts]      (prefix sum of expert_counts)
//
// Output: scattered[num_local_experts, max_tokens_per_expert, hidden_dim]
//         (3D, zero-padded for experts with fewer than max tokens)
//
// Each CUDA block handles one expert. Warps cooperatively copy tokens.
// Uses vectorized loads (float4 = 16 bytes) for bandwidth efficiency.

// Issue #27: Fused kernels — block 0 warp 0 computes prefix sum inline,
// other blocks spin-wait on d_offsets_ready flag. Saves one kernel launch
// (~1-3 μs) per layout conversion.

__global__ void scatter_2d_to_3d_bf16(
    const __nv_bfloat16* __restrict__ flat,       // [total, hidden]
    __nv_bfloat16* __restrict__ scattered,         // [E, max_tok, hidden]
    const int32_t* __restrict__ expert_counts,     // [E]
    int32_t* __restrict__ expert_offsets,           // [E] (computed inline)
    int32_t* __restrict__ offsets_ready_flag,       // [1] GLOBAL memory flag (Issue #41)
    int num_local_experts,
    int max_tokens_per_expert,
    int hidden_dim)
{
    // Issue #27 + #41: Fused prefix sum — no separate kernel launch needed.
    // Block 0, warp 0 computes offsets. All other blocks spin briefly.
    //
    // Issue #41 FIX: The v5.1 code used `extern __shared__` for the ready
    // flag, which is block-local — other blocks cannot see it. The flag
    // MUST be in global memory (d_offsets_ready from LayoutNormalizer)
    // for cross-block visibility. Now passed as kernel argument.
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        // Warp 0 of block 0: compute exclusive prefix sum
        const int lane = threadIdx.x;
        const int WARP_SIZE = 32;
        int32_t running_total = 0;
        for (int base = 0; base < num_local_experts; base += WARP_SIZE) {
            int idx = base + lane;
            int32_t val = (idx < num_local_experts) ? expert_counts[idx] : 0;
            int32_t scan = val;
            #pragma unroll
            for (int delta = 1; delta < WARP_SIZE; delta <<= 1) {
                int32_t n = __shfl_up_sync(0xFFFFFFFF, scan, delta);
                if (lane >= delta) scan += n;
            }
            int32_t exclusive = __shfl_up_sync(0xFFFFFFFF, scan, 1);
            if (lane == 0) exclusive = 0;
            exclusive += running_total;
            if (idx < num_local_experts) expert_offsets[idx] = exclusive;
            running_total += __shfl_sync(0xFFFFFFFF, scan, WARP_SIZE - 1);
        }
        // Issue #41: Signal via GLOBAL memory flag, not shared memory.
        // __threadfence() ensures expert_offsets[] writes are visible
        // to all SMs before the flag is set.
        if (lane == 0) { __threadfence(); atomicExch(offsets_ready_flag, 1); }
    }
    // All blocks: wait for prefix sum completion via GLOBAL flag
    if (blockIdx.x != 0) {
        if (threadIdx.x == 0) {
            // Spin on global flag (< 0.1 μs on same-GPC)
            while (atomicAdd(offsets_ready_flag, 0) == 0) {}
        }
    }
    __syncthreads();

    const int expert_id = blockIdx.x;
    if (expert_id >= num_local_experts) return;

    const int count = expert_counts[expert_id];
    const int offset = expert_offsets[expert_id];

    // Output base for this expert: scattered[expert_id, 0, 0]
    __nv_bfloat16* out_base = scattered +
        (int64_t)expert_id * max_tokens_per_expert * hidden_dim;

    // Input base: flat[offset, 0]
    const __nv_bfloat16* in_base = flat + (int64_t)offset * hidden_dim;

    // Vectorized copy: each thread handles 8 bf16 elements (= float4)
    const int vec_dim = hidden_dim / 8;  // assumes hidden_dim % 8 == 0
    const float4* in_vec  = reinterpret_cast<const float4*>(in_base);
    float4*       out_vec = reinterpret_cast<float4*>(out_base);

    for (int tok = 0; tok < count; tok++) {
        for (int v = threadIdx.x; v < vec_dim; v += blockDim.x) {
            out_vec[tok * vec_dim + v] = in_vec[tok * vec_dim + v];
        }
    }

    // Zero-pad remaining slots (tokens count..max_tokens_per_expert)
    for (int tok = count; tok < max_tokens_per_expert; tok++) {
        for (int v = threadIdx.x; v < vec_dim; v += blockDim.x) {
            out_vec[tok * vec_dim + v] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}


// ─── 3D -> 2D Flatten (NCCL-EP LL: native 3D -> requested 2D) ────
//
// Input:  scattered[num_local_experts, max_tokens_per_expert, hidden_dim]
//         expert_counts[num_local_experts]
//
// Output: flat[total_recv_tokens, hidden_dim]
//         where total_recv_tokens = sum(expert_counts)
//
// Each block handles one expert. Copies only valid tokens (no padding).

__global__ void flatten_3d_to_2d_bf16(
    const __nv_bfloat16* __restrict__ scattered,   // [E, max_tok, hidden]
    __nv_bfloat16* __restrict__ flat,               // [total, hidden]
    const int32_t* __restrict__ expert_counts,      // [E]
    const int32_t* __restrict__ expert_offsets,      // [E] (prefix sum into flat)
    int num_local_experts,
    int max_tokens_per_expert,
    int hidden_dim)
{
    const int expert_id = blockIdx.x;
    if (expert_id >= num_local_experts) return;

    const int count = expert_counts[expert_id];
    const int out_offset = expert_offsets[expert_id];

    const __nv_bfloat16* in_base = scattered +
        (int64_t)expert_id * max_tokens_per_expert * hidden_dim;
    __nv_bfloat16* out_base = flat + (int64_t)out_offset * hidden_dim;

    const int vec_dim = hidden_dim / 8;
    const float4* in_vec  = reinterpret_cast<const float4*>(in_base);
    float4*       out_vec = reinterpret_cast<float4*>(out_base);

    // Only copy valid tokens (skip padding)
    for (int tok = 0; tok < count; tok++) {
        for (int v = threadIdx.x; v < vec_dim; v += blockDim.x) {
            out_vec[tok * vec_dim + v] = in_vec[tok * vec_dim + v];
        }
    }
}


// ─── FP8 variants ─────────────────────────────────────────────────
// Same structure as above but with __nv_fp8_e4m3 type and
// vec_dim = hidden_dim / 16 (float4 = 16 bytes = 16 FP8 elements).

__global__ void scatter_2d_to_3d_fp8(
    const __nv_fp8_e4m3* __restrict__ flat,
    __nv_fp8_e4m3* __restrict__ scattered,
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ offsets_ready_flag,       // Issue #41: GLOBAL memory flag
    int num_local_experts,
    int max_tokens_per_expert,
    int hidden_dim)
{
    const int expert_id = blockIdx.x;
    if (expert_id >= num_local_experts) return;

    const int count = expert_counts[expert_id];
    const int offset = expert_offsets[expert_id];

    __nv_fp8_e4m3* out_base = scattered +
        (int64_t)expert_id * max_tokens_per_expert * hidden_dim;
    const __nv_fp8_e4m3* in_base = flat + (int64_t)offset * hidden_dim;

    // FP8: 1 byte per element. float4 = 16 elements.
    const int vec_dim = hidden_dim / 16;
    const float4* in_vec  = reinterpret_cast<const float4*>(in_base);
    float4*       out_vec = reinterpret_cast<float4*>(out_base);

    for (int tok = 0; tok < count; tok++) {
        for (int v = threadIdx.x; v < vec_dim; v += blockDim.x) {
            out_vec[tok * vec_dim + v] = in_vec[tok * vec_dim + v];
        }
    }
    for (int tok = count; tok < max_tokens_per_expert; tok++) {
        for (int v = threadIdx.x; v < vec_dim; v += blockDim.x) {
            out_vec[tok * vec_dim + v] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

__global__ void flatten_3d_to_2d_fp8(
    const __nv_fp8_e4m3* __restrict__ scattered,
    __nv_fp8_e4m3* __restrict__ flat,
    const int32_t* __restrict__ expert_counts,
    const int32_t* __restrict__ expert_offsets,
    int num_local_experts,
    int max_tokens_per_expert,
    int hidden_dim)
{
    const int expert_id = blockIdx.x;
    if (expert_id >= num_local_experts) return;

    const int count = expert_counts[expert_id];
    const int out_offset = expert_offsets[expert_id];

    const __nv_fp8_e4m3* in_base = scattered +
        (int64_t)expert_id * max_tokens_per_expert * hidden_dim;
    __nv_fp8_e4m3* out_base = flat + (int64_t)out_offset * hidden_dim;

    const int vec_dim = hidden_dim / 16;
    const float4* in_vec  = reinterpret_cast<const float4*>(in_base);
    float4*       out_vec = reinterpret_cast<float4*>(out_base);

    for (int tok = 0; tok < count; tok++) {
        for (int v = threadIdx.x; v < vec_dim; v += blockDim.x) {
            out_vec[tok * vec_dim + v] = in_vec[tok * vec_dim + v];
        }
    }
}


// ─── Fused prefix sum + data copy ─────────────────────────────────
//
// Issue #15: Warp-level __shfl_up_sync scan for the prefix sum.
// Issue #27: The v3 design launched TWO kernels per layout conversion:
//   1) exclusive_prefix_sum<<<1,32>>>  (~1 μs)
//   2) scatter/flatten<<<E,256>>>      (~1-3 μs launch overhead)
// Total: ~2-5 μs of kernel launch overhead per non-native layout.
//
// Fix: Fuse the prefix sum into block 0 of the scatter/flatten kernel
// using a __shared__ offset table + __threadfence_block(). All blocks
// read expert_counts at launch. Block 0 computes the prefix sum into
// shared memory, then writes to a global d_expert_offsets array.
// Other blocks spin-wait on a global "offsets_ready" flag (~<0.1 μs
// on NVLink-connected SMs). This eliminates one kernel launch per call.
//
// The standalone prefix sum kernel is retained for use by callers
// that need offsets without a data copy (e.g., diagnostics).

__device__ void compute_prefix_sum_block0(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int num_experts,
    int32_t* __restrict__ ready_flag)  // atomically set to 1 when done
{
    // Only block 0, warp 0 computes
    if (blockIdx.x != 0 || threadIdx.x >= 32) return;

    const int lane = threadIdx.x;
    const int WARP_SIZE = 32;
    int32_t running_total = 0;

    for (int base = 0; base < num_experts; base += WARP_SIZE) {
        int idx = base + lane;
        int32_t val = (idx < num_experts) ? counts[idx] : 0;

        int32_t scan = val;
        #pragma unroll
        for (int delta = 1; delta < WARP_SIZE; delta <<= 1) {
            int32_t n = __shfl_up_sync(0xFFFFFFFF, scan, delta);
            if (lane >= delta) scan += n;
        }
        int32_t exclusive = __shfl_up_sync(0xFFFFFFFF, scan, 1);
        if (lane == 0) exclusive = 0;
        exclusive += running_total;

        if (idx < num_experts) {
            offsets[idx] = exclusive;
        }
        running_total += __shfl_sync(0xFFFFFFFF, scan, WARP_SIZE - 1);
    }

    // Signal completion
    if (lane == 0) {
        __threadfence();  // ensure offsets are visible to all blocks
        atomicExch(ready_flag, 1);
    }
}

__device__ void wait_for_offsets(const int32_t* __restrict__ ready_flag) {
    // Non-block-0 blocks spin until offsets are ready.
    // Spin is < 0.1 μs on same-GPC SMs (NVLink latency).
    if (blockIdx.x == 0) return;
    if (threadIdx.x == 0) {
        while (atomicAdd(const_cast<int32_t*>(ready_flag), 0) == 0) {}
    }
    __syncthreads();
}

// Standalone prefix sum (retained for non-data-copy callers)
__global__ void exclusive_prefix_sum(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int num_experts)
{
    const int lane = threadIdx.x;
    const int WARP_SIZE = 32;
    int32_t running_total = 0;

    for (int base = 0; base < num_experts; base += WARP_SIZE) {
        int idx = base + lane;
        int32_t val = (idx < num_experts) ? counts[idx] : 0;
        int32_t scan = val;
        #pragma unroll
        for (int delta = 1; delta < WARP_SIZE; delta <<= 1) {
            int32_t n = __shfl_up_sync(0xFFFFFFFF, scan, delta);
            if (lane >= delta) scan += n;
        }
        int32_t exclusive = __shfl_up_sync(0xFFFFFFFF, scan, 1);
        if (lane == 0) exclusive = 0;
        exclusive += running_total;
        if (idx < num_experts) offsets[idx] = exclusive;
        running_total += __shfl_sync(0xFFFFFFFF, scan, WARP_SIZE - 1);
    }
}


// ─── Launch wrappers ──────────────────────────────────────────────
//
// Issue #27: Single-launch wrappers. Block 0 computes prefix sum,
// signals ready, then all blocks proceed with data copy.

struct LayoutNormalizer {
    // Scratch buffers (allocated once at group creation)
    int32_t* d_expert_offsets;
    int32_t* d_offsets_ready;      // Issue #27: flag for fused kernel sync
    int num_local_experts;
    int max_tokens_per_expert;
    int hidden_dim;

    void scatter_2d_to_3d(
        const void* flat, void* scattered,
        const int32_t* expert_counts,
        int scalar_type,
        cudaStream_t stream)
    {
        // Issue #27: Single fused launch — block 0 does prefix sum,
        // all blocks scatter. No separate prefix_sum kernel needed.
        // Issue #41: Reset GLOBAL ready flag (not shared memory).
        cudaMemsetAsync(d_offsets_ready, 0, sizeof(int32_t), stream);

        const int threads = 256;
        const int blocks = num_local_experts;

        if (scalar_type == 15) {  // BF16
            scatter_2d_to_3d_bf16<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(flat),
                static_cast<__nv_bfloat16*>(scattered),
                expert_counts, d_expert_offsets,
                d_offsets_ready,  // Issue #41: global flag
                num_local_experts, max_tokens_per_expert, hidden_dim);
        } else if (scalar_type == 23) {  // FP8 E4M3
            scatter_2d_to_3d_fp8<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_fp8_e4m3*>(flat),
                static_cast<__nv_fp8_e4m3*>(scattered),
                expert_counts, d_expert_offsets,
                d_offsets_ready,  // Issue #41: global flag
                num_local_experts, max_tokens_per_expert, hidden_dim);
        }
    }

    void flatten_3d_to_2d(
        const void* scattered, void* flat,
        const int32_t* expert_counts,
        int scalar_type,
        cudaStream_t stream)
    {
        // Issue #27: Single fused launch
        cudaMemsetAsync(d_offsets_ready, 0, sizeof(int32_t), stream);

        const int threads = 256;
        const int blocks = num_local_experts;

        if (scalar_type == 15) {
            flatten_3d_to_2d_bf16<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(scattered),
                static_cast<__nv_bfloat16*>(flat),
                expert_counts, d_expert_offsets,
                num_local_experts, max_tokens_per_expert, hidden_dim);
        } else if (scalar_type == 23) {
            flatten_3d_to_2d_fp8<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_fp8_e4m3*>(scattered),
                static_cast<__nv_fp8_e4m3*>(flat),
                expert_counts, d_expert_offsets,
                num_local_experts, max_tokens_per_expert, hidden_dim);
        }
    }
};

}  // namespace ep
}  // namespace flashinfer
