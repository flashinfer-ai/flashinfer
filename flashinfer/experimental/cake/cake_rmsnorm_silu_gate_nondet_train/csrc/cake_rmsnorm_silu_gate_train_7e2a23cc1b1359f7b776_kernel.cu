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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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

#define LAUNCH_MIN_BLOCKS 8

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


extern "C" __device__ float __nv_expf(float);


__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_7e2a23cc1b1359f7b776(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, unsigned long long* __restrict__ dw_win, float* __restrict__ dw_spec, unsigned int* __restrict__ dw_done, float* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad, int blocks_per_chunk)
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
    float* fold = reinterpret_cast<float*>(smem_raw + 0);
    const int fold_addr = smem + 0;
    long long* win = reinterpret_cast<long long*>(smem_raw + 8192);
    const int win_addr = smem + 8192;
    int* cell = reinterpret_cast<int*>(smem_raw + 15360);
    const int cell_addr = smem + 15360;
    int* flag = reinterpret_cast<int*>(smem_raw + 15376);
    const int flag_addr = smem + 15376;
    int* tab = reinterpret_cast<int*>(smem_raw + 15392);
    const int tab_addr = smem + 15392;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int ndb = (rows + 7) / 8;
    int nchunks = (ndb + blocks_per_chunk - 1) / blocks_per_chunk;
    float wv[8];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + col + 0);
        uint4 _vld_0[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_0[_blk] = _vptr_0[_blk];
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&wv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&wv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
    }
    int ticket = 2147483647;
    int claim_at = -1;
    if (tid == 0) {
        uint32_t _atomic_fetch_inc_1;
        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(_atomic_fetch_inc_1) : "l"(&dw_done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
        ticket = (int)_atomic_fetch_inc_1;
    }
    win[tid] = 0ll;
    win[128 + tid] = 0ll;
    win[256 + tid] = 0ll;
    win[384 + tid] = 0ll;
    win[512 + tid] = 0ll;
    win[640 + tid] = 0ll;
    win[768 + tid] = 0ll;
    float spec = 0.0f;
    float acc[8];
    int parity = 0;
    if (tid == 0) {
        int _min_2 = ((num_bids) < (nchunks) ? (num_bids) : (nchunks));
        int sl = ((_min_2) > (1) ? (_min_2) : (1));
        int share = (nchunks + sl - 1) / sl;
        int tail = (((share + 63) / 64) > (1) ? ((share + 63) / 64) : (1));
        int first_run = 0;
        int base = 0;
        int rem = nchunks;
        int size = (share + 1) / 2;
        int on = 1;
        on = on * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
        tab[0] = ((on != 0) ? first_run : 2147483647);
        tab[1] = base;
        tab[2] = size;
        first_run = first_run + on * sl;
        base = base + on * sl * size;
        rem = rem - on * sl * size;
        size = (size + 1) / 2;
        on = on * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
        tab[4] = ((on != 0) ? first_run : 2147483647);
        tab[5] = base;
        tab[6] = size;
        first_run = first_run + on * sl;
        base = base + on * sl * size;
        rem = rem - on * sl * size;
        size = (size + 1) / 2;
        on = on * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
        tab[8] = ((on != 0) ? first_run : 2147483647);
        tab[9] = base;
        tab[10] = size;
        first_run = first_run + on * sl;
        base = base + on * sl * size;
        rem = rem - on * sl * size;
        size = (size + 1) / 2;
        on = on * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
        tab[12] = ((on != 0) ? first_run : 2147483647);
        tab[13] = base;
        tab[14] = size;
        first_run = first_run + on * sl;
        base = base + on * sl * size;
        rem = rem - on * sl * size;
        size = (size + 1) / 2;
        on = on * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
        tab[16] = ((on != 0) ? first_run : 2147483647);
        tab[17] = base;
        tab[18] = size;
        first_run = first_run + on * sl;
        base = base + on * sl * size;
        rem = rem - on * sl * size;
        size = (size + 1) / 2;
        on = on * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
        tab[20] = ((on != 0) ? first_run : 2147483647);
        tab[21] = base;
        tab[22] = size;
        first_run = first_run + on * sl;
        base = base + on * sl * size;
        rem = rem - on * sl * size;
        size = (size + 1) / 2;
        tab[24] = first_run;
        tab[25] = base;
        tab[26] = 1;
        tab[28] = first_run + rem;
        tab[29] = 0;
        tab[30] = 0;
    }
    int _min_3 = ((num_bids) < (nchunks) ? (num_bids) : (nchunks));
    int t_sl = ((_min_3) > (1) ? (_min_3) : (1));
    int t_share = (nchunks + t_sl - 1) / t_sl;
    int t_tail = (((t_share + 63) / 64) > (1) ? ((t_share + 63) / 64) : (1));
    int t_rem = nchunks;
    int t_size = (t_share + 1) / 2;
    int t_on = 1;
    int trips = 1;
    t_on = t_on * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
    trips = trips + t_on;
    t_rem = t_rem - t_on * t_sl * t_size;
    t_size = (t_size + 1) / 2;
    t_on = t_on * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
    trips = trips + t_on;
    t_rem = t_rem - t_on * t_sl * t_size;
    t_size = (t_size + 1) / 2;
    t_on = t_on * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
    trips = trips + t_on;
    t_rem = t_rem - t_on * t_sl * t_size;
    t_size = (t_size + 1) / 2;
    t_on = t_on * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
    trips = trips + t_on;
    t_rem = t_rem - t_on * t_sl * t_size;
    t_size = (t_size + 1) / 2;
    t_on = t_on * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
    trips = trips + t_on;
    t_rem = t_rem - t_on * t_sl * t_size;
    t_size = (t_size + 1) / 2;
    t_on = t_on * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
    trips = trips + t_on;
    t_rem = t_rem - t_on * t_sl * t_size;
    t_size = (t_size + 1) / 2;
    trips = trips + (t_rem + t_sl - 1) / t_sl;
    #pragma unroll 1
    for (int it = 0; it < trips; it++) {
        int half = it % 2 * 2;
        if (tid == 0) {
            int begin = 0;
            int count = 0;
            int e_first_0 = tab[0];
            int e_base_0 = tab[1];
            int e_size_0 = tab[2];
            begin = ((e_first_0 <= ticket) ? e_base_0 + (ticket - e_first_0) * e_size_0 : begin);
            count = ((e_first_0 <= ticket) ? e_size_0 : count);
            int e_first_1 = tab[4];
            int e_base_1 = tab[5];
            int e_size_1 = tab[6];
            begin = ((e_first_1 <= ticket) ? e_base_1 + (ticket - e_first_1) * e_size_1 : begin);
            count = ((e_first_1 <= ticket) ? e_size_1 : count);
            int e_first_2 = tab[8];
            int e_base_2 = tab[9];
            int e_size_2 = tab[10];
            begin = ((e_first_2 <= ticket) ? e_base_2 + (ticket - e_first_2) * e_size_2 : begin);
            count = ((e_first_2 <= ticket) ? e_size_2 : count);
            int e_first_3 = tab[12];
            int e_base_3 = tab[13];
            int e_size_3 = tab[14];
            begin = ((e_first_3 <= ticket) ? e_base_3 + (ticket - e_first_3) * e_size_3 : begin);
            count = ((e_first_3 <= ticket) ? e_size_3 : count);
            int e_first_4 = tab[16];
            int e_base_4 = tab[17];
            int e_size_4 = tab[18];
            begin = ((e_first_4 <= ticket) ? e_base_4 + (ticket - e_first_4) * e_size_4 : begin);
            count = ((e_first_4 <= ticket) ? e_size_4 : count);
            int e_first_5 = tab[20];
            int e_base_5 = tab[21];
            int e_size_5 = tab[22];
            begin = ((e_first_5 <= ticket) ? e_base_5 + (ticket - e_first_5) * e_size_5 : begin);
            count = ((e_first_5 <= ticket) ? e_size_5 : count);
            int e_first_6 = tab[24];
            int e_base_6 = tab[25];
            int e_size_6 = tab[26];
            begin = ((e_first_6 <= ticket) ? e_base_6 + (ticket - e_first_6) * e_size_6 : begin);
            count = ((e_first_6 <= ticket) ? e_size_6 : count);
            int e_first_7 = tab[28];
            int e_base_7 = tab[29];
            int e_size_7 = tab[30];
            begin = ((e_first_7 <= ticket) ? e_base_7 + (ticket - e_first_7) * e_size_7 : begin);
            count = ((e_first_7 <= ticket) ? e_size_7 : count);
            ticket = 2147483647;
            claim_at = -1;
            if (count > 0) {
                if (it < trips - 1) {
                    int _min_4 = (((begin + count) * blocks_per_chunk) < (ndb) ? ((begin + count) * blocks_per_chunk) : (ndb));
                    claim_at = _min_4 - 1;
                }
            }
            cell[half] = begin;
            cell[half + 1] = count;
        }
        __syncthreads();
        int run_lo = cell[half];
        int run_n = cell[half + 1];
        if (run_n > 0) {
            int _min_5 = (((run_lo + run_n) * blocks_per_chunk) < (ndb) ? ((run_lo + run_n) * blocks_per_chunk) : (ndb));
            if (run_lo * blocks_per_chunk + 1 < _min_5) {
                if ((tid & 7) == 0) {
                    int prow = (((run_lo * blocks_per_chunk + 1) * 8 + group) < (rows - 1) ? ((run_lo * blocks_per_chunk + 1) * 8 + group) : (rows - 1));
                    uint32_t pq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)prow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)prow);
                    uint32_t ph = (uint32_t)((unsigned int)prow) - pq * (uint32_t)(heads.divisor);
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)pq * dy_tok + (long long)ph * dy_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)pq * x_tok + (long long)ph * x_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)pq * z_tok + (long long)ph * z_head + (long long)col))));
                }
            }
            acc[0] = 0.0f;
            acc[1] = 0.0f;
            acc[2] = 0.0f;
            acc[3] = 0.0f;
            acc[4] = 0.0f;
            acc[5] = 0.0f;
            acc[6] = 0.0f;
            acc[7] = 0.0f;
            int _min_6 = (((run_lo + run_n) * blocks_per_chunk) < (ndb) ? ((run_lo + run_n) * blocks_per_chunk) : (ndb));
            int cend = ((run_lo * blocks_per_chunk + blocks_per_chunk) < (_min_6) ? (run_lo * blocks_per_chunk + blocks_per_chunk) : (_min_6));
            int _min_7 = (((run_lo + run_n) * blocks_per_chunk) < (ndb) ? ((run_lo + run_n) * blocks_per_chunk) : (ndb));
            #pragma unroll 1
            for (int u = run_lo * blocks_per_chunk; u < _min_7; u++) {
                float gs[8];
                float xs[8];
                float zs[8];
                int row = u * 8 + group;
                int src = ((row) < (rows - 1) ? (row) : (rows - 1));
                uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
                uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(dy + ((long long)tq * dy_tok + (long long)th * dy_head + (long long)col) + 0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                            (&gs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + ((long long)tq * x_tok + (long long)th * x_head + (long long)col) + 0);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                            (&xs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_10 = reinterpret_cast<const uint4*>(z + ((long long)tq * z_tok + (long long)th * z_head + (long long)col) + 0);
                    uint4 _vld_10[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_10[_blk] = _vptr_10[_blk];
                        uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                            (&zs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                        }
                    }
                }
                int _min_11 = (((run_lo + run_n) * blocks_per_chunk) < (ndb) ? ((run_lo + run_n) * blocks_per_chunk) : (ndb));
                if (u + 2 < _min_11) {
                    if ((tid & 7) == 0) {
                        int prow_1 = (((u + 2) * 8 + group) < (rows - 1) ? ((u + 2) * 8 + group) : (rows - 1));
                        uint32_t pq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)prow_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)prow_1);
                        uint32_t ph_1 = (uint32_t)((unsigned int)prow_1) - pq_1 * (uint32_t)(heads.divisor);
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)pq_1 * dy_tok + (long long)ph_1 * dy_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)pq_1 * x_tok + (long long)ph_1 * x_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)pq_1 * z_tok + (long long)ph_1 * z_head + (long long)col))));
                    }
                }
                if (tid == 0) {
                    if (u == claim_at) {
                        uint32_t _atomic_fetch_inc_12;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_12) : "l"(&dw_done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        ticket = (int)_atomic_fetch_inc_12;
                    }
                }
                float total = 0.0f;
                total = __fmaf_rn(xs[0], xs[0], total);
                total = __fmaf_rn(xs[1], xs[1], total);
                total = __fmaf_rn(xs[2], xs[2], total);
                total = __fmaf_rn(xs[3], xs[3], total);
                total = __fmaf_rn(xs[4], xs[4], total);
                total = __fmaf_rn(xs[5], xs[5], total);
                total = __fmaf_rn(xs[6], xs[6], total);
                total = __fmaf_rn(xs[7], xs[7], total);
                float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
                total = total + _shfl_xor_sync_13;
                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
                total = total + _shfl_xor_sync_14;
                float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
                total = total + _shfl_xor_sync_15;
                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
                total = total + _shfl_xor_sync_16;
                float _sqrt_17;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_17) : "f"(total * 0.0078125f + eps));
                float rstd = __frcp_rn(_sqrt_17);
                float live = ((row < rows) ? 1.0f : 0.0f);
                float dot = 0.0f;
                int fast = 0;
                float _neg_18;
                asm("neg.f32 %0, %1;" : "=f"(_neg_18) : "f"(zs[0]));
                float _neg_19;
                asm("neg.f32 %0, %1;" : "=f"(_neg_19) : "f"(zs[1]));
                float _neg_20;
                asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(zs[2]));
                float mz;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz) : "f"(_neg_18), "f"(_neg_19), "f"(_neg_20));
                float _abs_22;
                asm("abs.f32 %0, %1;" : "=f"(_abs_22) : "f"(zs[0]));
                float _neg_21;
                asm("neg.f32 %0, %1;" : "=f"(_neg_21) : "f"(_abs_22));
                float _abs_24;
                asm("abs.f32 %0, %1;" : "=f"(_abs_24) : "f"(zs[1]));
                float _neg_23;
                asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(_abs_24));
                float _abs_26;
                asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(zs[2]));
                float _neg_25;
                asm("neg.f32 %0, %1;" : "=f"(_neg_25) : "f"(_abs_26));
                float ma;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma) : "f"(_neg_21), "f"(_neg_23), "f"(_neg_25));
                float _neg_28;
                asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(zs[3]));
                float _neg_29;
                asm("neg.f32 %0, %1;" : "=f"(_neg_29) : "f"(zs[4]));
                float _max3_27;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_27) : "f"(mz), "f"(_neg_28), "f"(_neg_29));
                mz = _max3_27;
                float _abs_32;
                asm("abs.f32 %0, %1;" : "=f"(_abs_32) : "f"(zs[3]));
                float _neg_31;
                asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(_abs_32));
                float _abs_34;
                asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(zs[4]));
                float _neg_33;
                asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(_abs_34));
                float _max3_30;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(ma), "f"(_neg_31), "f"(_neg_33));
                ma = _max3_30;
                float _neg_36;
                asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(zs[5]));
                float _neg_37;
                asm("neg.f32 %0, %1;" : "=f"(_neg_37) : "f"(zs[6]));
                float _max3_35;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_35) : "f"(mz), "f"(_neg_36), "f"(_neg_37));
                mz = _max3_35;
                float _abs_40;
                asm("abs.f32 %0, %1;" : "=f"(_abs_40) : "f"(zs[5]));
                float _neg_39;
                asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(_abs_40));
                float _abs_42;
                asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(zs[6]));
                float _neg_41;
                asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(_abs_42));
                float _max3_38;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_38) : "f"(ma), "f"(_neg_39), "f"(_neg_41));
                ma = _max3_38;
                float _neg_44;
                asm("neg.f32 %0, %1;" : "=f"(_neg_44) : "f"(zs[7]));
                float _max_43;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_43) : "f"(mz), "f"(_neg_44));
                mz = _max_43;
                float _abs_47;
                asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(zs[7]));
                float _neg_46;
                asm("neg.f32 %0, %1;" : "=f"(_neg_46) : "f"(_abs_47));
                float _max_45;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_45) : "f"(ma), "f"(_neg_46));
                ma = _max_45;
                float _abs_48;
                asm("abs.f32 %0, %1;" : "=f"(_abs_48) : "f"(zs[0]));
                float _abs_49;
                asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(zs[1]));
                float _abs_50;
                asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(zs[2]));
                float mp;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp) : "f"(_abs_48), "f"(_abs_49), "f"(_abs_50));
                float _abs_52;
                asm("abs.f32 %0, %1;" : "=f"(_abs_52) : "f"(zs[3]));
                float _abs_53;
                asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(zs[4]));
                float _max3_51;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_51) : "f"(mp), "f"(_abs_52), "f"(_abs_53));
                mp = _max3_51;
                float _abs_55;
                asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(zs[5]));
                float _abs_56;
                asm("abs.f32 %0, %1;" : "=f"(_abs_56) : "f"(zs[6]));
                float _max3_54;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_54) : "f"(mp), "f"(_abs_55), "f"(_abs_56));
                mp = _max3_54;
                float _abs_58;
                asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(zs[7]));
                float _max_57;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_57) : "f"(mp), "f"(_abs_58));
                mp = _max_57;
                if (mz <= 41.0f) {
                    if (ma <= -1.1754944e-38f) {
                        if (mp <= 3.4028235e+38f) {
                            fast = 1;
                        }
                    }
                }
                if (fast != 0) {
                    float2 rstd2 = make_float2(rstd, rstd);
                    float2 live2 = make_float2(live, live);
                    float2 z2_0 = make_float2(zs[0], zs[1]);
                    float ta_0;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs[0]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_0;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs[1]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_59;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(ta_0), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_60;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(tb_0), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_0 = make_float2(_fma_59, _fma_60);
                    float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_0));
                    r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
                    float2 nscale_0 = make_float2(__uint_as_float((__as_u32(t2_0.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_0.y) << 23) + 2147483648u));
                    float _exp2_61 = approx_exp2(r2_0.x);
                    float _exp2_62 = approx_exp2(r2_0.y);
                    float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_61, _exp2_62), make_float2(-1.0f, -1.0f));
                    float _neg_64;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_64) : "f"(nden_0.x));
                    float _rcp_63 = approx_rcp(_neg_64);
                    float _neg_66;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_66) : "f"(nden_0.y));
                    float _rcp_65 = approx_rcp(_neg_66);
                    float2 r0_0 = make_float2(_rcp_63, _rcp_65);
                    float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
                    float2 sig2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, r1_0, make_float2(1.0f, 1.0f)), r1_0);
                    float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
                    float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
                    float2 g2_0 = make_float2(gs[0], gs[1]);
                    float2 w2_0 = make_float2(wv[0], wv[1]);
                    float2 _f32x2_mul_a_67 = make_float2(xs[0], xs[1]);
                    float2 n2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&rstd2));
                    float2 _f32x2_mul_a_68 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
                    float2 _f32x2_mul_b_69 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
                    float2 dz2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
                    float2 a2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
                    float2 dn2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
                    float2 acc2_0 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_0, live2), n2_0, make_float2(acc[0], acc[1]));
                    acc[0] = acc2_0.x;
                    acc[1] = acc2_0.y;
                    zs[0] = dz2_0.x;
                    zs[1] = dz2_0.y;
                    gs[0] = dn2_0.x;
                    gs[1] = dn2_0.y;
                    float2 z2_1 = make_float2(zs[2], zs[3]);
                    float ta_1;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs[2]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_1;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs[3]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_70;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(ta_1), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_71;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(tb_1), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_1 = make_float2(_fma_70, _fma_71);
                    float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_1));
                    r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
                    float2 nscale_1 = make_float2(__uint_as_float((__as_u32(t2_1.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_1.y) << 23) + 2147483648u));
                    float _exp2_72 = approx_exp2(r2_1.x);
                    float _exp2_73 = approx_exp2(r2_1.y);
                    float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_72, _exp2_73), make_float2(-1.0f, -1.0f));
                    float _neg_75;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_75) : "f"(nden_1.x));
                    float _rcp_74 = approx_rcp(_neg_75);
                    float _neg_77;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_77) : "f"(nden_1.y));
                    float _rcp_76 = approx_rcp(_neg_77);
                    float2 r0_1 = make_float2(_rcp_74, _rcp_76);
                    float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
                    float2 sig2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, r1_1, make_float2(1.0f, 1.0f)), r1_1);
                    float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
                    float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
                    float2 g2_1 = make_float2(gs[2], gs[3]);
                    float2 w2_1 = make_float2(wv[2], wv[3]);
                    float2 _f32x2_mul_a_78 = make_float2(xs[2], xs[3]);
                    float2 n2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&rstd2));
                    float2 _f32x2_mul_a_79 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
                    float2 _f32x2_mul_b_80 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
                    float2 dz2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
                    float2 a2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
                    float2 dn2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
                    float2 acc2_1 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_1, live2), n2_1, make_float2(acc[2], acc[3]));
                    acc[2] = acc2_1.x;
                    acc[3] = acc2_1.y;
                    zs[2] = dz2_1.x;
                    zs[3] = dz2_1.y;
                    gs[2] = dn2_1.x;
                    gs[3] = dn2_1.y;
                    float2 z2_2 = make_float2(zs[4], zs[5]);
                    float ta_2;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs[4]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_2;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs[5]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_81;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(ta_2), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_82;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(tb_2), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_2 = make_float2(_fma_81, _fma_82);
                    float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_2));
                    r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
                    float2 nscale_2 = make_float2(__uint_as_float((__as_u32(t2_2.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_2.y) << 23) + 2147483648u));
                    float _exp2_83 = approx_exp2(r2_2.x);
                    float _exp2_84 = approx_exp2(r2_2.y);
                    float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_83, _exp2_84), make_float2(-1.0f, -1.0f));
                    float _neg_86;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_86) : "f"(nden_2.x));
                    float _rcp_85 = approx_rcp(_neg_86);
                    float _neg_88;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_88) : "f"(nden_2.y));
                    float _rcp_87 = approx_rcp(_neg_88);
                    float2 r0_2 = make_float2(_rcp_85, _rcp_87);
                    float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
                    float2 sig2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, r1_2, make_float2(1.0f, 1.0f)), r1_2);
                    float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
                    float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
                    float2 g2_2 = make_float2(gs[4], gs[5]);
                    float2 w2_2 = make_float2(wv[4], wv[5]);
                    float2 _f32x2_mul_a_89 = make_float2(xs[4], xs[5]);
                    float2 n2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&rstd2));
                    float2 _f32x2_mul_a_90 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
                    float2 _f32x2_mul_b_91 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
                    float2 dz2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_91));
                    float2 a2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
                    float2 dn2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
                    float2 acc2_2 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_2, live2), n2_2, make_float2(acc[4], acc[5]));
                    acc[4] = acc2_2.x;
                    acc[5] = acc2_2.y;
                    zs[4] = dz2_2.x;
                    zs[5] = dz2_2.y;
                    gs[4] = dn2_2.x;
                    gs[5] = dn2_2.y;
                    float2 z2_3 = make_float2(zs[6], zs[7]);
                    float ta_3;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs[6]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_3;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs[7]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_92;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(ta_3), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_93;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(tb_3), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_3 = make_float2(_fma_92, _fma_93);
                    float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_3));
                    r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
                    float2 nscale_3 = make_float2(__uint_as_float((__as_u32(t2_3.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_3.y) << 23) + 2147483648u));
                    float _exp2_94 = approx_exp2(r2_3.x);
                    float _exp2_95 = approx_exp2(r2_3.y);
                    float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_94, _exp2_95), make_float2(-1.0f, -1.0f));
                    float _neg_97;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_97) : "f"(nden_3.x));
                    float _rcp_96 = approx_rcp(_neg_97);
                    float _neg_99;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_99) : "f"(nden_3.y));
                    float _rcp_98 = approx_rcp(_neg_99);
                    float2 r0_3 = make_float2(_rcp_96, _rcp_98);
                    float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
                    float2 sig2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, r1_3, make_float2(1.0f, 1.0f)), r1_3);
                    float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
                    float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
                    float2 g2_3 = make_float2(gs[6], gs[7]);
                    float2 w2_3 = make_float2(wv[6], wv[7]);
                    float2 _f32x2_mul_a_100 = make_float2(xs[6], xs[7]);
                    float2 n2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&rstd2));
                    float2 _f32x2_mul_a_101 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
                    float2 _f32x2_mul_b_102 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
                    float2 dz2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
                    float2 a2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
                    float2 dn2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
                    float2 acc2_3 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_3, live2), n2_3, make_float2(acc[6], acc[7]));
                    acc[6] = acc2_3.x;
                    acc[7] = acc2_3.y;
                    zs[6] = dz2_3.x;
                    zs[7] = dz2_3.y;
                    gs[6] = dn2_3.x;
                    gs[7] = dn2_3.y;
                    dot = __fmaf_rn(gs[0], xs[0], dot);
                    dot = __fmaf_rn(gs[1], xs[1], dot);
                    dot = __fmaf_rn(gs[2], xs[2], dot);
                    dot = __fmaf_rn(gs[3], xs[3], dot);
                    dot = __fmaf_rn(gs[4], xs[4], dot);
                    dot = __fmaf_rn(gs[5], xs[5], dot);
                    dot = __fmaf_rn(gs[6], xs[6], dot);
                    dot = __fmaf_rn(gs[7], xs[7], dot);
                } else {
                    float _exp_103 = __nv_expf(zs[0] * -1.0f);
                    float den_0 = 1.0f + _exp_103;
                    float sig_0 = __fdiv_rn(1.0f, den_0);
                    float gate_0 = __fdiv_rn(zs[0], den_0);
                    float nj_0 = xs[0] * rstd;
                    float aj_0 = gs[0] * gate_0;
                    float dzj_0 = gs[0] * (nj_0 * wv[0]) * sig_0 * __fmaf_rn(zs[0], 1.0f - sig_0, 1.0f);
                    float dnj_0 = aj_0 * wv[0];
                    dot = __fmaf_rn(dnj_0, xs[0], dot);
                    acc[0] = __fmaf_rn(aj_0 * live, nj_0, acc[0]);
                    zs[0] = dzj_0;
                    gs[0] = dnj_0;
                    float _exp_104 = __nv_expf(zs[1] * -1.0f);
                    float den_1 = 1.0f + _exp_104;
                    float sig_1 = __fdiv_rn(1.0f, den_1);
                    float gate_1 = __fdiv_rn(zs[1], den_1);
                    float nj_1 = xs[1] * rstd;
                    float aj_1 = gs[1] * gate_1;
                    float dzj_1 = gs[1] * (nj_1 * wv[1]) * sig_1 * __fmaf_rn(zs[1], 1.0f - sig_1, 1.0f);
                    float dnj_1 = aj_1 * wv[1];
                    dot = __fmaf_rn(dnj_1, xs[1], dot);
                    acc[1] = __fmaf_rn(aj_1 * live, nj_1, acc[1]);
                    zs[1] = dzj_1;
                    gs[1] = dnj_1;
                    float _exp_105 = __nv_expf(zs[2] * -1.0f);
                    float den_2 = 1.0f + _exp_105;
                    float sig_2 = __fdiv_rn(1.0f, den_2);
                    float gate_2 = __fdiv_rn(zs[2], den_2);
                    float nj_2 = xs[2] * rstd;
                    float aj_2 = gs[2] * gate_2;
                    float dzj_2 = gs[2] * (nj_2 * wv[2]) * sig_2 * __fmaf_rn(zs[2], 1.0f - sig_2, 1.0f);
                    float dnj_2 = aj_2 * wv[2];
                    dot = __fmaf_rn(dnj_2, xs[2], dot);
                    acc[2] = __fmaf_rn(aj_2 * live, nj_2, acc[2]);
                    zs[2] = dzj_2;
                    gs[2] = dnj_2;
                    float _exp_106 = __nv_expf(zs[3] * -1.0f);
                    float den_3 = 1.0f + _exp_106;
                    float sig_3 = __fdiv_rn(1.0f, den_3);
                    float gate_3 = __fdiv_rn(zs[3], den_3);
                    float nj_3 = xs[3] * rstd;
                    float aj_3 = gs[3] * gate_3;
                    float dzj_3 = gs[3] * (nj_3 * wv[3]) * sig_3 * __fmaf_rn(zs[3], 1.0f - sig_3, 1.0f);
                    float dnj_3 = aj_3 * wv[3];
                    dot = __fmaf_rn(dnj_3, xs[3], dot);
                    acc[3] = __fmaf_rn(aj_3 * live, nj_3, acc[3]);
                    zs[3] = dzj_3;
                    gs[3] = dnj_3;
                    float _exp_107 = __nv_expf(zs[4] * -1.0f);
                    float den_4 = 1.0f + _exp_107;
                    float sig_4 = __fdiv_rn(1.0f, den_4);
                    float gate_4 = __fdiv_rn(zs[4], den_4);
                    float nj_4 = xs[4] * rstd;
                    float aj_4 = gs[4] * gate_4;
                    float dzj_4 = gs[4] * (nj_4 * wv[4]) * sig_4 * __fmaf_rn(zs[4], 1.0f - sig_4, 1.0f);
                    float dnj_4 = aj_4 * wv[4];
                    dot = __fmaf_rn(dnj_4, xs[4], dot);
                    acc[4] = __fmaf_rn(aj_4 * live, nj_4, acc[4]);
                    zs[4] = dzj_4;
                    gs[4] = dnj_4;
                    float _exp_108 = __nv_expf(zs[5] * -1.0f);
                    float den_5 = 1.0f + _exp_108;
                    float sig_5 = __fdiv_rn(1.0f, den_5);
                    float gate_5 = __fdiv_rn(zs[5], den_5);
                    float nj_5 = xs[5] * rstd;
                    float aj_5 = gs[5] * gate_5;
                    float dzj_5 = gs[5] * (nj_5 * wv[5]) * sig_5 * __fmaf_rn(zs[5], 1.0f - sig_5, 1.0f);
                    float dnj_5 = aj_5 * wv[5];
                    dot = __fmaf_rn(dnj_5, xs[5], dot);
                    acc[5] = __fmaf_rn(aj_5 * live, nj_5, acc[5]);
                    zs[5] = dzj_5;
                    gs[5] = dnj_5;
                    float _exp_109 = __nv_expf(zs[6] * -1.0f);
                    float den_6 = 1.0f + _exp_109;
                    float sig_6 = __fdiv_rn(1.0f, den_6);
                    float gate_6 = __fdiv_rn(zs[6], den_6);
                    float nj_6 = xs[6] * rstd;
                    float aj_6 = gs[6] * gate_6;
                    float dzj_6 = gs[6] * (nj_6 * wv[6]) * sig_6 * __fmaf_rn(zs[6], 1.0f - sig_6, 1.0f);
                    float dnj_6 = aj_6 * wv[6];
                    dot = __fmaf_rn(dnj_6, xs[6], dot);
                    acc[6] = __fmaf_rn(aj_6 * live, nj_6, acc[6]);
                    zs[6] = dzj_6;
                    gs[6] = dnj_6;
                    float _exp_110 = __nv_expf(zs[7] * -1.0f);
                    float den_7 = 1.0f + _exp_110;
                    float sig_7 = __fdiv_rn(1.0f, den_7);
                    float gate_7 = __fdiv_rn(zs[7], den_7);
                    float nj_7 = xs[7] * rstd;
                    float aj_7 = gs[7] * gate_7;
                    float dzj_7 = gs[7] * (nj_7 * wv[7]) * sig_7 * __fmaf_rn(zs[7], 1.0f - sig_7, 1.0f);
                    float dnj_7 = aj_7 * wv[7];
                    dot = __fmaf_rn(dnj_7, xs[7], dot);
                    acc[7] = __fmaf_rn(aj_7 * live, nj_7, acc[7]);
                    zs[7] = dzj_7;
                    gs[7] = dnj_7;
                }
                float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_111;
                float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_112;
                float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_113;
                float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
                dot = dot + _shfl_xor_sync_114;
                float coef = rstd * rstd * rstd * (dot * -0.0078125f);
                float2 coef2 = make_float2(coef, coef);
                float2 rstd2b = make_float2(rstd, rstd);
                float2 dx2_0 = fma_f32x2_rn_noftz(make_float2(xs[0], xs[1]), coef2, mul_f32x2_noftz(make_float2(gs[0], gs[1]), rstd2b));
                xs[0] = dx2_0.x;
                xs[1] = dx2_0.y;
                float2 dx2_1 = fma_f32x2_rn_noftz(make_float2(xs[2], xs[3]), coef2, mul_f32x2_noftz(make_float2(gs[2], gs[3]), rstd2b));
                xs[2] = dx2_1.x;
                xs[3] = dx2_1.y;
                float2 dx2_2 = fma_f32x2_rn_noftz(make_float2(xs[4], xs[5]), coef2, mul_f32x2_noftz(make_float2(gs[4], gs[5]), rstd2b));
                xs[4] = dx2_2.x;
                xs[5] = dx2_2.y;
                float2 dx2_3 = fma_f32x2_rn_noftz(make_float2(xs[6], xs[7]), coef2, mul_f32x2_noftz(make_float2(gs[6], gs[7]), rstd2b));
                xs[6] = dx2_3.x;
                xs[7] = dx2_3.y;
                unsigned int packed_dx[4];
                unsigned int packed_dz[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                    packed_dx[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 0], zs[_lp*2+1 + 0]));
                    packed_dz[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                    packed_dx[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 2], zs[_lp*2+1 + 2]));
                    packed_dz[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                    packed_dx[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 4], zs[_lp*2+1 + 4]));
                    packed_dz[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                    packed_dx[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 6], zs[_lp*2+1 + 6]));
                    packed_dz[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row < rows) {
                    long long orow = (long long)row * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow)[0] = reinterpret_cast<int4*>(packed_dx)[0];
                    reinterpret_cast<int4*>(dz + orow)[0] = reinterpret_cast<int4*>(packed_dz)[0];
                }
                if (u + 1 == cend) {
                    if (weight_grad != 0) {
                        fold[parity * 1024 + group * 128 + col] = acc[0];
                        fold[parity * 1024 + group * 128 + col + 1] = acc[1];
                        fold[parity * 1024 + group * 128 + col + 2] = acc[2];
                        fold[parity * 1024 + group * 128 + col + 3] = acc[3];
                        fold[parity * 1024 + group * 128 + col + 4] = acc[4];
                        fold[parity * 1024 + group * 128 + col + 5] = acc[5];
                        fold[parity * 1024 + group * 128 + col + 6] = acc[6];
                        fold[parity * 1024 + group * 128 + col + 7] = acc[7];
                        __syncthreads();
                        float colsum = 0.0f;
                        #pragma unroll
                        for (int g = 0; g < 8; g++) {
                            colsum = colsum + fold[parity * 1024 + g * 128 + tid];
                        }
                        if ((__as_u32(colsum) & 2139095040) == 2139095040) {
                            spec = spec + colsum;
                        } else if ((__as_u32(colsum) & 2147483647) != 0) {
                            unsigned int bits = __as_u32(colsum);
                            unsigned int ex = bits >> 23 & 255;
                            unsigned int _max_115 = ((ex) > (1) ? (ex) : (1));
                            unsigned int idx = _max_115 - 1;
                            unsigned int _min_116 = ((ex) < (1) ? (ex) : (1));
                            unsigned long long m = (unsigned long long)(bits & 8388607 | _min_116 << 23);
                            unsigned int wk = idx / 48;
                            unsigned int sh = idx - wk * 48;
                            long long lo = (long long)(m << (unsigned long long)sh & 281474976710655ull);
                            long long hi = (long long)(m >> (unsigned long long)(48 - sh));
                            if (bits >> 31 != 0) {
                                lo = -lo;
                                hi = -hi;
                            }
                            win[(int)wk * 128 + tid] = win[(int)wk * 128 + tid] + lo;
                            if (hi != 0ll) {
                                win[((int)wk + 1) * 128 + tid] = win[((int)wk + 1) * 128 + tid] + hi;
                            }
                        }
                    }
                    parity = 1 - parity;
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    int _min_118 = (((run_lo + run_n) * blocks_per_chunk) < (ndb) ? ((run_lo + run_n) * blocks_per_chunk) : (ndb));
                    int _min_117 = ((cend + blocks_per_chunk) < (_min_118) ? (cend + blocks_per_chunk) : (_min_118));
                    cend = _min_117;
                }
            }
        }
    }
    if (weight_grad != 0) {
        long long wsum_0 = win[tid];
        if (wsum_0 != 0ll) {
            atomicAdd(&dw_win[(bid & 3) * 7 * 128 + tid], (unsigned long long)wsum_0);
        }
        long long wsum_1 = win[128 + tid];
        if (wsum_1 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 1) * 128 + tid], (unsigned long long)wsum_1);
        }
        long long wsum_2 = win[256 + tid];
        if (wsum_2 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 2) * 128 + tid], (unsigned long long)wsum_2);
        }
        long long wsum_3 = win[384 + tid];
        if (wsum_3 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 3) * 128 + tid], (unsigned long long)wsum_3);
        }
        long long wsum_4 = win[512 + tid];
        if (wsum_4 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 4) * 128 + tid], (unsigned long long)wsum_4);
        }
        long long wsum_5 = win[640 + tid];
        if (wsum_5 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 5) * 128 + tid], (unsigned long long)wsum_5);
        }
        long long wsum_6 = win[768 + tid];
        if (wsum_6 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 6) * 128 + tid], (unsigned long long)wsum_6);
        }
        if (spec != 0.0f) {
            atomicAdd(&dw_spec[tid], spec);
        }
    }
    __syncthreads();
    if (tid == 0) {
        int is_last = 0;
        uint32_t arrival;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(arrival) : "l"(&dw_done[0]), "r"(static_cast<uint32_t>((unsigned int)num_bids - 1)) : "memory");
        if (arrival == (unsigned int)num_bids - 1) {
            is_last = 1;
            dw_done[1] = 0;
        }
        flag[0] = is_last;
    }
    __syncthreads();
    int flag_word[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&flag_word[0])) : "r"(flag_addr));
    if (flag_word[0] != 0) {
        if (weight_grad != 0) {
            unsigned long long words[28];
            unsigned long long _atomic_fetch_add_119 = atomicAdd(&dw_win[tid], 0ull);
            words[0] = _atomic_fetch_add_119;
            unsigned long long _atomic_fetch_add_120 = atomicAdd(&dw_win[128 + tid], 0ull);
            words[1] = _atomic_fetch_add_120;
            unsigned long long _atomic_fetch_add_121 = atomicAdd(&dw_win[256 + tid], 0ull);
            words[2] = _atomic_fetch_add_121;
            unsigned long long _atomic_fetch_add_122 = atomicAdd(&dw_win[384 + tid], 0ull);
            words[3] = _atomic_fetch_add_122;
            unsigned long long _atomic_fetch_add_123 = atomicAdd(&dw_win[512 + tid], 0ull);
            words[4] = _atomic_fetch_add_123;
            unsigned long long _atomic_fetch_add_124 = atomicAdd(&dw_win[640 + tid], 0ull);
            words[5] = _atomic_fetch_add_124;
            unsigned long long _atomic_fetch_add_125 = atomicAdd(&dw_win[768 + tid], 0ull);
            words[6] = _atomic_fetch_add_125;
            unsigned long long _atomic_fetch_add_126 = atomicAdd(&dw_win[896 + tid], 0ull);
            words[7] = _atomic_fetch_add_126;
            unsigned long long _atomic_fetch_add_127 = atomicAdd(&dw_win[1024 + tid], 0ull);
            words[8] = _atomic_fetch_add_127;
            unsigned long long _atomic_fetch_add_128 = atomicAdd(&dw_win[1152 + tid], 0ull);
            words[9] = _atomic_fetch_add_128;
            unsigned long long _atomic_fetch_add_129 = atomicAdd(&dw_win[1280 + tid], 0ull);
            words[10] = _atomic_fetch_add_129;
            unsigned long long _atomic_fetch_add_130 = atomicAdd(&dw_win[1408 + tid], 0ull);
            words[11] = _atomic_fetch_add_130;
            unsigned long long _atomic_fetch_add_131 = atomicAdd(&dw_win[1536 + tid], 0ull);
            words[12] = _atomic_fetch_add_131;
            unsigned long long _atomic_fetch_add_132 = atomicAdd(&dw_win[1664 + tid], 0ull);
            words[13] = _atomic_fetch_add_132;
            unsigned long long _atomic_fetch_add_133 = atomicAdd(&dw_win[1792 + tid], 0ull);
            words[14] = _atomic_fetch_add_133;
            unsigned long long _atomic_fetch_add_134 = atomicAdd(&dw_win[1920 + tid], 0ull);
            words[15] = _atomic_fetch_add_134;
            unsigned long long _atomic_fetch_add_135 = atomicAdd(&dw_win[2048 + tid], 0ull);
            words[16] = _atomic_fetch_add_135;
            unsigned long long _atomic_fetch_add_136 = atomicAdd(&dw_win[2176 + tid], 0ull);
            words[17] = _atomic_fetch_add_136;
            unsigned long long _atomic_fetch_add_137 = atomicAdd(&dw_win[2304 + tid], 0ull);
            words[18] = _atomic_fetch_add_137;
            unsigned long long _atomic_fetch_add_138 = atomicAdd(&dw_win[2432 + tid], 0ull);
            words[19] = _atomic_fetch_add_138;
            unsigned long long _atomic_fetch_add_139 = atomicAdd(&dw_win[2560 + tid], 0ull);
            words[20] = _atomic_fetch_add_139;
            unsigned long long _atomic_fetch_add_140 = atomicAdd(&dw_win[2688 + tid], 0ull);
            words[21] = _atomic_fetch_add_140;
            unsigned long long _atomic_fetch_add_141 = atomicAdd(&dw_win[2816 + tid], 0ull);
            words[22] = _atomic_fetch_add_141;
            unsigned long long _atomic_fetch_add_142 = atomicAdd(&dw_win[2944 + tid], 0ull);
            words[23] = _atomic_fetch_add_142;
            unsigned long long _atomic_fetch_add_143 = atomicAdd(&dw_win[3072 + tid], 0ull);
            words[24] = _atomic_fetch_add_143;
            unsigned long long _atomic_fetch_add_144 = atomicAdd(&dw_win[3200 + tid], 0ull);
            words[25] = _atomic_fetch_add_144;
            unsigned long long _atomic_fetch_add_145 = atomicAdd(&dw_win[3328 + tid], 0ull);
            words[26] = _atomic_fetch_add_145;
            unsigned long long _atomic_fetch_add_146 = atomicAdd(&dw_win[3456 + tid], 0ull);
            words[27] = _atomic_fetch_add_146;
            float _atomic_fetch_add_147 = atomicAdd(&dw_spec[tid], -0.0f);
            spec = _atomic_fetch_add_147;
            float old = 0.0f;
            if (accumulate != 0) {
                float prev = atomicAdd(&dw[tid], -0.0f);
                old = prev;
            }
            long long digits[7];
            digits[0] = 0ll;
            digits[1] = 0ll;
            digits[2] = 0ll;
            digits[3] = 0ll;
            digits[4] = 0ll;
            digits[5] = 0ll;
            digits[6] = 0ll;
            digits[0] = digits[0] + (long long)words[0];
            if (words[0] != 0ull) {
                dw_win[tid] = 0ull;
            }
            digits[1] = digits[1] + (long long)words[1];
            if (words[1] != 0ull) {
                dw_win[128 + tid] = 0ull;
            }
            digits[2] = digits[2] + (long long)words[2];
            if (words[2] != 0ull) {
                dw_win[256 + tid] = 0ull;
            }
            digits[3] = digits[3] + (long long)words[3];
            if (words[3] != 0ull) {
                dw_win[384 + tid] = 0ull;
            }
            digits[4] = digits[4] + (long long)words[4];
            if (words[4] != 0ull) {
                dw_win[512 + tid] = 0ull;
            }
            digits[5] = digits[5] + (long long)words[5];
            if (words[5] != 0ull) {
                dw_win[640 + tid] = 0ull;
            }
            digits[6] = digits[6] + (long long)words[6];
            if (words[6] != 0ull) {
                dw_win[768 + tid] = 0ull;
            }
            digits[0] = digits[0] + (long long)words[7];
            if (words[7] != 0ull) {
                dw_win[896 + tid] = 0ull;
            }
            digits[1] = digits[1] + (long long)words[8];
            if (words[8] != 0ull) {
                dw_win[1024 + tid] = 0ull;
            }
            digits[2] = digits[2] + (long long)words[9];
            if (words[9] != 0ull) {
                dw_win[1152 + tid] = 0ull;
            }
            digits[3] = digits[3] + (long long)words[10];
            if (words[10] != 0ull) {
                dw_win[1280 + tid] = 0ull;
            }
            digits[4] = digits[4] + (long long)words[11];
            if (words[11] != 0ull) {
                dw_win[1408 + tid] = 0ull;
            }
            digits[5] = digits[5] + (long long)words[12];
            if (words[12] != 0ull) {
                dw_win[1536 + tid] = 0ull;
            }
            digits[6] = digits[6] + (long long)words[13];
            if (words[13] != 0ull) {
                dw_win[1664 + tid] = 0ull;
            }
            digits[0] = digits[0] + (long long)words[14];
            if (words[14] != 0ull) {
                dw_win[1792 + tid] = 0ull;
            }
            digits[1] = digits[1] + (long long)words[15];
            if (words[15] != 0ull) {
                dw_win[1920 + tid] = 0ull;
            }
            digits[2] = digits[2] + (long long)words[16];
            if (words[16] != 0ull) {
                dw_win[2048 + tid] = 0ull;
            }
            digits[3] = digits[3] + (long long)words[17];
            if (words[17] != 0ull) {
                dw_win[2176 + tid] = 0ull;
            }
            digits[4] = digits[4] + (long long)words[18];
            if (words[18] != 0ull) {
                dw_win[2304 + tid] = 0ull;
            }
            digits[5] = digits[5] + (long long)words[19];
            if (words[19] != 0ull) {
                dw_win[2432 + tid] = 0ull;
            }
            digits[6] = digits[6] + (long long)words[20];
            if (words[20] != 0ull) {
                dw_win[2560 + tid] = 0ull;
            }
            digits[0] = digits[0] + (long long)words[21];
            if (words[21] != 0ull) {
                dw_win[2688 + tid] = 0ull;
            }
            digits[1] = digits[1] + (long long)words[22];
            if (words[22] != 0ull) {
                dw_win[2816 + tid] = 0ull;
            }
            digits[2] = digits[2] + (long long)words[23];
            if (words[23] != 0ull) {
                dw_win[2944 + tid] = 0ull;
            }
            digits[3] = digits[3] + (long long)words[24];
            if (words[24] != 0ull) {
                dw_win[3072 + tid] = 0ull;
            }
            digits[4] = digits[4] + (long long)words[25];
            if (words[25] != 0ull) {
                dw_win[3200 + tid] = 0ull;
            }
            digits[5] = digits[5] + (long long)words[26];
            if (words[26] != 0ull) {
                dw_win[3328 + tid] = 0ull;
            }
            digits[6] = digits[6] + (long long)words[27];
            if (words[27] != 0ull) {
                dw_win[3456 + tid] = 0ull;
            }
            if (spec != 0.0f) {
                dw_spec[tid] = 0.0f;
            }
            if ((__as_u32(old) & 2139095040) == 2139095040) {
                spec = spec + old;
            } else if ((__as_u32(old) & 2147483647) != 0) {
                unsigned int bits_1 = __as_u32(old);
                unsigned int ex_1 = bits_1 >> 23 & 255;
                unsigned int _max_148 = ((ex_1) > (1) ? (ex_1) : (1));
                unsigned int idx_1 = _max_148 - 1;
                unsigned int _min_149 = ((ex_1) < (1) ? (ex_1) : (1));
                unsigned long long m_1 = (unsigned long long)(bits_1 & 8388607 | _min_149 << 23);
                unsigned int wk_1 = idx_1 / 48;
                unsigned int sh_1 = idx_1 - wk_1 * 48;
                long long lo_1 = (long long)(m_1 << (unsigned long long)sh_1 & 281474976710655ull);
                long long hi_1 = (long long)(m_1 >> (unsigned long long)(48 - sh_1));
                if (bits_1 >> 31 != 0) {
                    lo_1 = -lo_1;
                    hi_1 = -hi_1;
                }
                digits[0] = digits[0] + (((int)wk_1 == 0) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 0) ? hi_1 : 0ll);
                digits[1] = digits[1] + (((int)wk_1 == 1) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 1) ? hi_1 : 0ll);
                digits[2] = digits[2] + (((int)wk_1 == 2) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 2) ? hi_1 : 0ll);
                digits[3] = digits[3] + (((int)wk_1 == 3) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 3) ? hi_1 : 0ll);
                digits[4] = digits[4] + (((int)wk_1 == 4) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 4) ? hi_1 : 0ll);
                digits[5] = digits[5] + (((int)wk_1 == 5) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 5) ? hi_1 : 0ll);
                digits[6] = digits[6] + (((int)wk_1 == 6) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 6) ? hi_1 : 0ll);
            }
            long long carry = 0ll;
            long long t_0 = digits[0] + carry;
            digits[0] = t_0 & 281474976710655ll;
            carry = t_0 >> 48ll;
            long long t_1 = digits[1] + carry;
            digits[1] = t_1 & 281474976710655ll;
            carry = t_1 >> 48ll;
            long long t_2 = digits[2] + carry;
            digits[2] = t_2 & 281474976710655ll;
            carry = t_2 >> 48ll;
            long long t_3 = digits[3] + carry;
            digits[3] = t_3 & 281474976710655ll;
            carry = t_3 >> 48ll;
            long long t_4 = digits[4] + carry;
            digits[4] = t_4 & 281474976710655ll;
            carry = t_4 >> 48ll;
            long long t_5 = digits[5] + carry;
            digits[5] = t_5 & 281474976710655ll;
            carry = t_5 >> 48ll;
            long long t_6 = digits[6] + carry;
            digits[6] = t_6 & 281474976710655ll;
            carry = t_6 >> 48ll;
            double exact = (double)carry;
            exact = exact * 281474976710656.0 + (double)digits[6];
            exact = exact * 281474976710656.0 + (double)digits[5];
            exact = exact * 281474976710656.0 + (double)digits[4];
            exact = exact * 281474976710656.0 + (double)digits[3];
            exact = exact * 281474976710656.0 + (double)digits[2];
            exact = exact * 281474976710656.0 + (double)digits[1];
            exact = exact * 281474976710656.0 + (double)digits[0];
            float rounded = ((spec != 0.0f) ? spec : (float)(exact * 1.401298464324817e-45));
            dw[tid] = rounded;
        }
    }
}

} // extern "C"
