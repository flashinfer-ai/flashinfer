/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_lm_head_loss_c4e7e9ccc838c461277a(long long* __restrict__ labels, float* __restrict__ dlogp, float* __restrict__ lse, long long* __restrict__ order, float* __restrict__ lse_out, int* __restrict__ scan, int* __restrict__ counts, int T, int chunk, int num_chunks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* sc_live = reinterpret_cast<int*>(smem_raw + 0);
    const int _sc_live_addr = smem + 0;
    int* sc_head = reinterpret_cast<int*>(smem_raw + 128);
    const int _sc_head_addr = smem + 128;
    int* sc_valid = reinterpret_cast<int*>(smem_raw + 256);
    const int _sc_valid_addr = smem + 256;
    int* sc_tile = reinterpret_cast<int*>(smem_raw + 384);
    const int _sc_tile_addr = smem + 384;

    // === Task calls (declaration order) ===
    int tid_1 = tid;
    int tile0 = blockIdx.x * 4096;
    int n_live = 0;
    int h_live = 0;
    int h_valid = 0;
    #pragma unroll 8
    for (int i = tid_1; i < T; i += 1024) {
        long long a_lab = labels[i];
        float a_d = dlogp[i];
        int a_valid = ((a_lab >= 0ll) ? 1 : 0);
        int a_live = ((a_d != 0.0f) ? a_valid : 0);
        n_live += a_live;
        if (i < tile0) {
            h_live += a_live;
            h_valid += a_valid;
        }
    }
    // CtaScanSum: threads=1024, warps=32
    int _live_prefix;
    int live_total;
    {
        const uint32_t _live_prefix_own = (uint32_t)(n_live);
        uint32_t _live_prefix_ws = _live_prefix_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_ws, 1, 32); if (lane >= 1) { _live_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_ws, 2, 32); if (lane >= 2) { _live_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_ws, 4, 32); if (lane >= 4) { _live_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_ws, 8, 32); if (lane >= 8) { _live_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_ws, 16, 32); if (lane >= 16) { _live_prefix_ws += peer; } }
        if (lane == 31) { sc_live[warp] = (int)_live_prefix_ws; }
        __syncthreads();
        const uint32_t _live_prefix_wt = (uint32_t)sc_live[lane];
        uint32_t _live_prefix_wsum = _live_prefix_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_wsum, 1, 32); if (lane >= 1) { _live_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_wsum, 2, 32); if (lane >= 2) { _live_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_wsum, 4, 32); if (lane >= 4) { _live_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_wsum, 8, 32); if (lane >= 8) { _live_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _live_prefix_wsum, 16, 32); if (lane >= 16) { _live_prefix_wsum += peer; } }
        live_total = (int)__shfl_sync(0xFFFFFFFF, _live_prefix_wsum, 31);
        const uint32_t _live_prefix_before = __shfl_sync(0xFFFFFFFF, _live_prefix_wsum - _live_prefix_wt, warp);
        _live_prefix = (int)(_live_prefix_ws + _live_prefix_before);
    }
    // CtaScanSum: threads=1024, warps=32
    int _head_prefix;
    int live_head;
    {
        const uint32_t _head_prefix_own = (uint32_t)(h_live);
        uint32_t _head_prefix_ws = _head_prefix_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_ws, 1, 32); if (lane >= 1) { _head_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_ws, 2, 32); if (lane >= 2) { _head_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_ws, 4, 32); if (lane >= 4) { _head_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_ws, 8, 32); if (lane >= 8) { _head_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_ws, 16, 32); if (lane >= 16) { _head_prefix_ws += peer; } }
        if (lane == 31) { sc_head[warp] = (int)_head_prefix_ws; }
        __syncthreads();
        const uint32_t _head_prefix_wt = (uint32_t)sc_head[lane];
        uint32_t _head_prefix_wsum = _head_prefix_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_wsum, 1, 32); if (lane >= 1) { _head_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_wsum, 2, 32); if (lane >= 2) { _head_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_wsum, 4, 32); if (lane >= 4) { _head_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_wsum, 8, 32); if (lane >= 8) { _head_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _head_prefix_wsum, 16, 32); if (lane >= 16) { _head_prefix_wsum += peer; } }
        live_head = (int)__shfl_sync(0xFFFFFFFF, _head_prefix_wsum, 31);
        const uint32_t _head_prefix_before = __shfl_sync(0xFFFFFFFF, _head_prefix_wsum - _head_prefix_wt, warp);
        _head_prefix = (int)(_head_prefix_ws + _head_prefix_before);
    }
    // CtaScanSum: threads=1024, warps=32
    int _valid_prefix;
    int valid_head;
    {
        const uint32_t _valid_prefix_own = (uint32_t)(h_valid);
        uint32_t _valid_prefix_ws = _valid_prefix_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_ws, 1, 32); if (lane >= 1) { _valid_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_ws, 2, 32); if (lane >= 2) { _valid_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_ws, 4, 32); if (lane >= 4) { _valid_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_ws, 8, 32); if (lane >= 8) { _valid_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_ws, 16, 32); if (lane >= 16) { _valid_prefix_ws += peer; } }
        if (lane == 31) { sc_valid[warp] = (int)_valid_prefix_ws; }
        __syncthreads();
        const uint32_t _valid_prefix_wt = (uint32_t)sc_valid[lane];
        uint32_t _valid_prefix_wsum = _valid_prefix_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_wsum, 1, 32); if (lane >= 1) { _valid_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_wsum, 2, 32); if (lane >= 2) { _valid_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_wsum, 4, 32); if (lane >= 4) { _valid_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_wsum, 8, 32); if (lane >= 8) { _valid_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _valid_prefix_wsum, 16, 32); if (lane >= 16) { _valid_prefix_wsum += peer; } }
        valid_head = (int)__shfl_sync(0xFFFFFFFF, _valid_prefix_wsum, 31);
        const uint32_t _valid_prefix_before = __shfl_sync(0xFFFFFFFF, _valid_prefix_wsum - _valid_prefix_wt, warp);
        _valid_prefix = (int)(_valid_prefix_ws + _valid_prefix_before);
    }
    int r0 = tile0 + tid_1 * 4;
    int live_j[4];
    int valid_j[4];
    int t_live = 0;
    int t_valid = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        live_j[j] = 0;
        valid_j[j] = 0;
        if (r0 + j < T) {
            long long b_lab = labels[r0 + j];
            float b_d = dlogp[r0 + j];
            valid_j[j] = ((b_lab >= 0ll) ? 1 : 0);
            live_j[j] = ((b_d != 0.0f) ? valid_j[j] : 0);
        }
        t_live += live_j[j];
        t_valid += valid_j[j];
    }
    int t_packed = t_valid * 65536 + t_live;
    // CtaScanSum: threads=1024, warps=32
    int tile_prefix;
    int _tile_total;
    {
        const uint32_t tile_prefix_own = (uint32_t)(t_packed);
        uint32_t tile_prefix_ws = tile_prefix_own;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_ws, 1, 32); if (lane >= 1) { tile_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_ws, 2, 32); if (lane >= 2) { tile_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_ws, 4, 32); if (lane >= 4) { tile_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_ws, 8, 32); if (lane >= 8) { tile_prefix_ws += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_ws, 16, 32); if (lane >= 16) { tile_prefix_ws += peer; } }
        if (lane == 31) { sc_tile[warp] = (int)tile_prefix_ws; }
        __syncthreads();
        const uint32_t tile_prefix_wt = (uint32_t)sc_tile[lane];
        uint32_t tile_prefix_wsum = tile_prefix_wt;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_wsum, 1, 32); if (lane >= 1) { tile_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_wsum, 2, 32); if (lane >= 2) { tile_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_wsum, 4, 32); if (lane >= 4) { tile_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_wsum, 8, 32); if (lane >= 8) { tile_prefix_wsum += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, tile_prefix_wsum, 16, 32); if (lane >= 16) { tile_prefix_wsum += peer; } }
        _tile_total = (int)__shfl_sync(0xFFFFFFFF, tile_prefix_wsum, 31);
        const uint32_t tile_prefix_before = __shfl_sync(0xFFFFFFFF, tile_prefix_wsum - tile_prefix_wt, warp);
        tile_prefix = (int)(tile_prefix_ws - tile_prefix_own + tile_prefix_before);
    }
    int live_run = live_head + (tile_prefix & 65535);
    int valid_run = valid_head + (tile_prefix >> 16);
    #pragma unroll
    for (int j_1 = 0; j_1 < 4; j_1++) {
        if (r0 + j_1 < T) {
            live_run += live_j[j_1];
            scan[r0 + j_1] = live_run;
            if (valid_j[j_1] != 0) {
                int dest = ((live_j[j_1] != 0) ? live_run - 1 : live_total + valid_run - live_run);
                order[dest] = (long long)(r0 + j_1);
                lse_out[dest] = lse[valid_run];
                valid_run += 1;
            }
        }
    }
    if (blockIdx.x == 0) {
        for (int c = tid_1; c < num_chunks; c += 1024) {
            int left = live_total - c * chunk;
            int held = ((left < 0) ? 0 : ((left > chunk) ? chunk : left));
            counts[c] = ((c == 0) ? ((held < 1) ? 1 : held) : held);
        }
    }
}

} // extern "C"
