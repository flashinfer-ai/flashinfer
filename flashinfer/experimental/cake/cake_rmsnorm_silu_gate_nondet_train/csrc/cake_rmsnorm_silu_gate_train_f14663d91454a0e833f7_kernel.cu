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

#define LAUNCH_MIN_BLOCKS 4

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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_f14663d91454a0e833f7(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, double* __restrict__ dw_sums, unsigned int* __restrict__ dw_done, __nv_bfloat16* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
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
    int* cell = reinterpret_cast<int*>(smem_raw + 16384);
    const int cell_addr = smem + 16384;
    int* flag = reinterpret_cast<int*>(smem_raw + 16400);
    const int flag_addr = smem + 16400;
    int* tab = reinterpret_cast<int*>(smem_raw + 16416);
    const int tab_addr = smem + 16416;
    float* fold32 = reinterpret_cast<float*>(smem_raw + 16544);
    const int fold32_addr = smem + 16544;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int nblocks = (rows + 15) / 16;
    int padded_block = ((rows % 16 != 0) ? nblocks - 1 : -1);
    int bound = 0;
    if (nblocks <= num_bids) {
        bound = nblocks;
    }
    int rest = nblocks - bound;
    unsigned int wbits[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(w + col + 0);
        wbits[0 + 0] = _uv4_0.x;
        wbits[0 + 1] = _uv4_0.y;
        wbits[0 + 2] = _uv4_0.z;
        wbits[0 + 3] = _uv4_0.w;
    }
    float wclaim[8];
    int ticket = 2147483647;
    if (tid == 0) {
        if (rest > 0) {
            uint32_t _atomic_fetch_inc_1;
            asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                : "=r"(_atomic_fetch_inc_1) : "l"(&dw_done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
            ticket = (int)_atomic_fetch_inc_1;
        }
    }
    if (rest > 0) {
        int _min_2 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int sl0 = ((_min_2) > (1) ? (_min_2) : (1));
        if (bid < sl0) {
            int pblk = bid * (((rest + sl0 - 1) / sl0 + 1) / 2);
            int prow_0 = ((pblk * 16 + group) < (rows - 1) ? (pblk * 16 + group) : (rows - 1));
            uint32_t pq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)prow_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)prow_0);
            uint32_t ph_0 = (uint32_t)((unsigned int)prow_0) - pq_0 * (uint32_t)(heads.divisor);
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)pq_0 * dy_tok + (long long)ph_0 * dy_head + (long long)col))));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)pq_0 * x_tok + (long long)ph_0 * x_head + (long long)col))));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)pq_0 * z_tok + (long long)ph_0 * z_head + (long long)col))));
        }
    }
    float acc[8];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    int run = 0;
    int folded = 0;
    long long heads64 = (long long)((uint32_t)(heads.divisor));
    int packed = 0;
    if (dy_head == 128ll) {
        if (x_head == 128ll) {
            if (z_head == 128ll) {
                if (dy_tok == heads64 * 128ll) {
                    if (x_tok == heads64 * 128ll) {
                        if (z_tok == heads64 * 128ll) {
                            packed = 1;
                        }
                    }
                }
            }
        }
    }
    if (tid == 0) {
        if (rest > 0) {
            int _min_3 = ((num_bids) < (rest) ? (num_bids) : (rest));
            int sl = ((_min_3) > (1) ? (_min_3) : (1));
            int share = (rest + sl - 1) / sl;
            int tail = (((share + 63) / 64) > (1) ? ((share + 63) / 64) : (1));
            int first_chunk = 0;
            int base = 0;
            int rem = rest;
            int size = (share + 1) / 2;
            int live = 1;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[0] = ((live != 0) ? first_chunk : 2147483647);
            tab[1] = base;
            tab[2] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[4] = ((live != 0) ? first_chunk : 2147483647);
            tab[5] = base;
            tab[6] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[8] = ((live != 0) ? first_chunk : 2147483647);
            tab[9] = base;
            tab[10] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[12] = ((live != 0) ? first_chunk : 2147483647);
            tab[13] = base;
            tab[14] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[16] = ((live != 0) ? first_chunk : 2147483647);
            tab[17] = base;
            tab[18] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            live = live * ((size >= tail) ? 1 : 0) * ((rem >= sl * size) ? 1 : 0);
            tab[20] = ((live != 0) ? first_chunk : 2147483647);
            tab[21] = base;
            tab[22] = size;
            first_chunk = first_chunk + live * sl;
            base = base + live * sl * size;
            rem = rem - live * sl * size;
            size = (size + 1) / 2;
            tab[24] = first_chunk;
            tab[25] = base;
            tab[26] = 1;
            tab[28] = first_chunk + rem;
            tab[29] = 0;
            tab[30] = 0;
        }
    }
    if (bid < bound) {
        #pragma unroll 1
        for (int i = 0; i < 1; i++) {
            int u = bid + i;
            int row_0 = u * 16 + group;
            int src_0 = ((row_0) < (rows - 1) ? (row_0) : (rows - 1));
            uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
            uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
            long long goff_0 = (long long)tq_0 * dy_tok + (long long)th_0 * dy_head + (long long)col;
            long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
            long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
            float gs_0[8];
            float xs_0[8];
            float zs_0[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(dy + goff_0 + 0);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                        (&gs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + xoff_0 + 0);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                        (&xs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(z + zoff_0 + 0);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                        (&zs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                    }
                }
            }
            if (i + 1 < 1) {
                if (packed != 0) {
                    if (tid == 0) {
                        int nfirst = (u + 1) * 16;
                        int _min_7 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                        int nbytes = _min_7 * 256;
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                    }
                } else {
                    int nrow_0 = (((u + 1) * 16 + group) < (rows - 1) ? ((u + 1) * 16 + group) : (rows - 1));
                    uint32_t nq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow_0);
                    uint32_t nh_0 = (uint32_t)((unsigned int)nrow_0) - nq_0 * (uint32_t)(heads.divisor);
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq_0 * dy_tok + (long long)nh_0 * dy_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq_0 * x_tok + (long long)nh_0 * x_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq_0 * z_tok + (long long)nh_0 * z_head + (long long)col))));
                }
            }
            float wv[8];
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wv[_pair * 2])[0]), "=f"((&wv[_pair * 2])[1])
                    : "r"(wbits[_pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wv[2 + _pair * 2])[0]), "=f"((&wv[2 + _pair * 2])[1])
                    : "r"(wbits[1 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wv[4 + _pair * 2])[0]), "=f"((&wv[4 + _pair * 2])[1])
                    : "r"(wbits[2 + _pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&wv[6 + _pair * 2])[0]), "=f"((&wv[6 + _pair * 2])[1])
                    : "r"(wbits[3 + _pair]));
            }
            if (u == padded_block) {
                float total_0 = 0.0f;
                total_0 = __fmaf_rn(xs_0[0], xs_0[0], total_0);
                total_0 = __fmaf_rn(xs_0[1], xs_0[1], total_0);
                total_0 = __fmaf_rn(xs_0[2], xs_0[2], total_0);
                total_0 = __fmaf_rn(xs_0[3], xs_0[3], total_0);
                total_0 = __fmaf_rn(xs_0[4], xs_0[4], total_0);
                total_0 = __fmaf_rn(xs_0[5], xs_0[5], total_0);
                total_0 = __fmaf_rn(xs_0[6], xs_0[6], total_0);
                total_0 = __fmaf_rn(xs_0[7], xs_0[7], total_0);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total_0, 1);
                total_0 = total_0 + _shfl_xor_sync_8;
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total_0, 2);
                total_0 = total_0 + _shfl_xor_sync_9;
                float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total_0, 4);
                total_0 = total_0 + _shfl_xor_sync_10;
                float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, total_0, 8);
                total_0 = total_0 + _shfl_xor_sync_11;
                float _sqrt_12;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_12) : "f"(total_0 * 0.0078125f + eps));
                float rstd_0 = __frcp_rn(_sqrt_12);
                unsigned int keep_0 = 4294967295u;
                unsigned int sign_0 = 2147483648u;
                keep_0 = ((row_0 < rows) ? keep_0 : sign_0);
                float dot_0 = 0.0f;
                int fast_0 = 0;
                float _neg_13;
                asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(zs_0[0]));
                float _neg_14;
                asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(zs_0[1]));
                float _neg_15;
                asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(zs_0[2]));
                float mz_0;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_0) : "f"(_neg_13), "f"(_neg_14), "f"(_neg_15));
                float _abs_17;
                asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(zs_0[0]));
                float _neg_16;
                asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(_abs_17));
                float _abs_19;
                asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(zs_0[1]));
                float _neg_18;
                asm("neg.f32 %0, %1;" : "=f"(_neg_18) : "f"(_abs_19));
                float _abs_21;
                asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(zs_0[2]));
                float _neg_20;
                asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(_abs_21));
                float ma_0;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_0) : "f"(_neg_16), "f"(_neg_18), "f"(_neg_20));
                float _neg_23;
                asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(zs_0[3]));
                float _neg_24;
                asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(zs_0[4]));
                float _max3_22;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(mz_0), "f"(_neg_23), "f"(_neg_24));
                mz_0 = _max3_22;
                float _abs_27;
                asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(zs_0[3]));
                float _neg_26;
                asm("neg.f32 %0, %1;" : "=f"(_neg_26) : "f"(_abs_27));
                float _abs_29;
                asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(zs_0[4]));
                float _neg_28;
                asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(_abs_29));
                float _max3_25;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(ma_0), "f"(_neg_26), "f"(_neg_28));
                ma_0 = _max3_25;
                float _neg_31;
                asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(zs_0[5]));
                float _neg_32;
                asm("neg.f32 %0, %1;" : "=f"(_neg_32) : "f"(zs_0[6]));
                float _max3_30;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(mz_0), "f"(_neg_31), "f"(_neg_32));
                mz_0 = _max3_30;
                float _abs_35;
                asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(zs_0[5]));
                float _neg_34;
                asm("neg.f32 %0, %1;" : "=f"(_neg_34) : "f"(_abs_35));
                float _abs_37;
                asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(zs_0[6]));
                float _neg_36;
                asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(_abs_37));
                float _max3_33;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_33) : "f"(ma_0), "f"(_neg_34), "f"(_neg_36));
                ma_0 = _max3_33;
                float _neg_39;
                asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(zs_0[7]));
                float _max_38;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_38) : "f"(mz_0), "f"(_neg_39));
                mz_0 = _max_38;
                float _abs_42;
                asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(zs_0[7]));
                float _neg_41;
                asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(_abs_42));
                float _max_40;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(ma_0), "f"(_neg_41));
                ma_0 = _max_40;
                float _abs_43;
                asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(zs_0[0]));
                float _abs_44;
                asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(zs_0[1]));
                float _abs_45;
                asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(zs_0[2]));
                float mp_0;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_0) : "f"(_abs_43), "f"(_abs_44), "f"(_abs_45));
                float _abs_47;
                asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(zs_0[3]));
                float _abs_48;
                asm("abs.f32 %0, %1;" : "=f"(_abs_48) : "f"(zs_0[4]));
                float _max3_46;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_46) : "f"(mp_0), "f"(_abs_47), "f"(_abs_48));
                mp_0 = _max3_46;
                float _abs_50;
                asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(zs_0[5]));
                float _abs_51;
                asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(zs_0[6]));
                float _max3_49;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_49) : "f"(mp_0), "f"(_abs_50), "f"(_abs_51));
                mp_0 = _max3_49;
                float _abs_53;
                asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(zs_0[7]));
                float _max_52;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(mp_0), "f"(_abs_53));
                mp_0 = _max_52;
                if (mz_0 <= 41.0f) {
                    if (ma_0 <= -1.1754944e-38f) {
                        if (mp_0 <= 3.4028235e+38f) {
                            fast_0 = 1;
                        }
                    }
                }
                if (fast_0 != 0) {
                    float2 rstd2_0 = make_float2(rstd_0, rstd_0);
                    float2 z2_0 = make_float2(zs_0[0], zs_0[1]);
                    float ta_0;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs_0[0]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_0;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs_0[1]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_54;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(ta_0), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_55;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(tb_0), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_0 = make_float2(_fma_54, _fma_55);
                    float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_0));
                    r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
                    float2 nscale_0 = make_float2(__uint_as_float((__as_u32(t2_0.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_0.y) << 23) + 2147483648u));
                    float _exp2_56 = approx_exp2(r2_0.x);
                    float _exp2_57 = approx_exp2(r2_0.y);
                    float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_56, _exp2_57), make_float2(-1.0f, -1.0f));
                    float _neg_59;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_59) : "f"(nden_0.x));
                    float _rcp_58 = approx_rcp(_neg_59);
                    float _neg_61;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_61) : "f"(nden_0.y));
                    float _rcp_60 = approx_rcp(_neg_61);
                    float2 r0_0 = make_float2(_rcp_58, _rcp_60);
                    float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
                    float2 sig2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, r1_0, make_float2(1.0f, 1.0f)), r1_0);
                    float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
                    float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
                    float2 g2_0 = make_float2(gs_0[0], gs_0[1]);
                    float2 w2_0 = make_float2(wv[0], wv[1]);
                    float2 _f32x2_mul_a_62 = make_float2(xs_0[0], xs_0[1]);
                    float2 n2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&rstd2_0));
                    float2 _f32x2_mul_a_63 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
                    float2 _f32x2_mul_b_64 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
                    float2 dz2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
                    float2 a2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
                    float2 dn2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
                    float2 acc2_0 = make_float2(acc[0], acc[1]);
                    acc2_0 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_0.x) & keep_0), __uint_as_float(__as_u32(a2_0.y) & keep_0)), make_float2(__uint_as_float(__as_u32(n2_0.x) & keep_0), __uint_as_float(__as_u32(n2_0.y) & keep_0)), acc2_0);
                    acc[0] = acc2_0.x;
                    acc[1] = acc2_0.y;
                    zs_0[0] = dz2_0.x;
                    zs_0[1] = dz2_0.y;
                    gs_0[0] = dn2_0.x;
                    gs_0[1] = dn2_0.y;
                    float2 z2_1 = make_float2(zs_0[2], zs_0[3]);
                    float ta_1;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs_0[2]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_1;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs_0[3]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_65;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(ta_1), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_66;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(tb_1), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_1 = make_float2(_fma_65, _fma_66);
                    float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_1));
                    r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
                    float2 nscale_1 = make_float2(__uint_as_float((__as_u32(t2_1.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_1.y) << 23) + 2147483648u));
                    float _exp2_67 = approx_exp2(r2_1.x);
                    float _exp2_68 = approx_exp2(r2_1.y);
                    float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_67, _exp2_68), make_float2(-1.0f, -1.0f));
                    float _neg_70;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_70) : "f"(nden_1.x));
                    float _rcp_69 = approx_rcp(_neg_70);
                    float _neg_72;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_72) : "f"(nden_1.y));
                    float _rcp_71 = approx_rcp(_neg_72);
                    float2 r0_1 = make_float2(_rcp_69, _rcp_71);
                    float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
                    float2 sig2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, r1_1, make_float2(1.0f, 1.0f)), r1_1);
                    float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
                    float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
                    float2 g2_1 = make_float2(gs_0[2], gs_0[3]);
                    float2 w2_1 = make_float2(wv[2], wv[3]);
                    float2 _f32x2_mul_a_73 = make_float2(xs_0[2], xs_0[3]);
                    float2 n2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&rstd2_0));
                    float2 _f32x2_mul_a_74 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
                    float2 _f32x2_mul_b_75 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
                    float2 dz2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                    float2 a2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
                    float2 dn2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
                    float2 acc2_1 = make_float2(acc[2], acc[3]);
                    acc2_1 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_1.x) & keep_0), __uint_as_float(__as_u32(a2_1.y) & keep_0)), make_float2(__uint_as_float(__as_u32(n2_1.x) & keep_0), __uint_as_float(__as_u32(n2_1.y) & keep_0)), acc2_1);
                    acc[2] = acc2_1.x;
                    acc[3] = acc2_1.y;
                    zs_0[2] = dz2_1.x;
                    zs_0[3] = dz2_1.y;
                    gs_0[2] = dn2_1.x;
                    gs_0[3] = dn2_1.y;
                    float2 z2_2 = make_float2(zs_0[4], zs_0[5]);
                    float ta_2;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs_0[4]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_2;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs_0[5]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_76;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(ta_2), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_77;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(tb_2), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_2 = make_float2(_fma_76, _fma_77);
                    float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_2));
                    r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
                    float2 nscale_2 = make_float2(__uint_as_float((__as_u32(t2_2.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_2.y) << 23) + 2147483648u));
                    float _exp2_78 = approx_exp2(r2_2.x);
                    float _exp2_79 = approx_exp2(r2_2.y);
                    float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_78, _exp2_79), make_float2(-1.0f, -1.0f));
                    float _neg_81;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_81) : "f"(nden_2.x));
                    float _rcp_80 = approx_rcp(_neg_81);
                    float _neg_83;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_83) : "f"(nden_2.y));
                    float _rcp_82 = approx_rcp(_neg_83);
                    float2 r0_2 = make_float2(_rcp_80, _rcp_82);
                    float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
                    float2 sig2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, r1_2, make_float2(1.0f, 1.0f)), r1_2);
                    float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
                    float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
                    float2 g2_2 = make_float2(gs_0[4], gs_0[5]);
                    float2 w2_2 = make_float2(wv[4], wv[5]);
                    float2 _f32x2_mul_a_84 = make_float2(xs_0[4], xs_0[5]);
                    float2 n2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&rstd2_0));
                    float2 _f32x2_mul_a_85 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
                    float2 _f32x2_mul_b_86 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
                    float2 dz2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                    float2 a2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
                    float2 dn2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
                    float2 acc2_2 = make_float2(acc[4], acc[5]);
                    acc2_2 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_2.x) & keep_0), __uint_as_float(__as_u32(a2_2.y) & keep_0)), make_float2(__uint_as_float(__as_u32(n2_2.x) & keep_0), __uint_as_float(__as_u32(n2_2.y) & keep_0)), acc2_2);
                    acc[4] = acc2_2.x;
                    acc[5] = acc2_2.y;
                    zs_0[4] = dz2_2.x;
                    zs_0[5] = dz2_2.y;
                    gs_0[4] = dn2_2.x;
                    gs_0[5] = dn2_2.y;
                    float2 z2_3 = make_float2(zs_0[6], zs_0[7]);
                    float ta_3;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs_0[6]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_3;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs_0[7]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_87;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(ta_3), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_88;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(tb_3), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_3 = make_float2(_fma_87, _fma_88);
                    float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_3));
                    r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
                    float2 nscale_3 = make_float2(__uint_as_float((__as_u32(t2_3.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_3.y) << 23) + 2147483648u));
                    float _exp2_89 = approx_exp2(r2_3.x);
                    float _exp2_90 = approx_exp2(r2_3.y);
                    float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_89, _exp2_90), make_float2(-1.0f, -1.0f));
                    float _neg_92;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_92) : "f"(nden_3.x));
                    float _rcp_91 = approx_rcp(_neg_92);
                    float _neg_94;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_94) : "f"(nden_3.y));
                    float _rcp_93 = approx_rcp(_neg_94);
                    float2 r0_3 = make_float2(_rcp_91, _rcp_93);
                    float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
                    float2 sig2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, r1_3, make_float2(1.0f, 1.0f)), r1_3);
                    float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
                    float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
                    float2 g2_3 = make_float2(gs_0[6], gs_0[7]);
                    float2 w2_3 = make_float2(wv[6], wv[7]);
                    float2 _f32x2_mul_a_95 = make_float2(xs_0[6], xs_0[7]);
                    float2 n2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&rstd2_0));
                    float2 _f32x2_mul_a_96 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
                    float2 _f32x2_mul_b_97 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
                    float2 dz2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_97));
                    float2 a2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
                    float2 dn2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
                    float2 acc2_3 = make_float2(acc[6], acc[7]);
                    acc2_3 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_3.x) & keep_0), __uint_as_float(__as_u32(a2_3.y) & keep_0)), make_float2(__uint_as_float(__as_u32(n2_3.x) & keep_0), __uint_as_float(__as_u32(n2_3.y) & keep_0)), acc2_3);
                    acc[6] = acc2_3.x;
                    acc[7] = acc2_3.y;
                    zs_0[6] = dz2_3.x;
                    zs_0[7] = dz2_3.y;
                    gs_0[6] = dn2_3.x;
                    gs_0[7] = dn2_3.y;
                    dot_0 = __fmaf_rn(gs_0[0], xs_0[0], dot_0);
                    dot_0 = __fmaf_rn(gs_0[1], xs_0[1], dot_0);
                    dot_0 = __fmaf_rn(gs_0[2], xs_0[2], dot_0);
                    dot_0 = __fmaf_rn(gs_0[3], xs_0[3], dot_0);
                    dot_0 = __fmaf_rn(gs_0[4], xs_0[4], dot_0);
                    dot_0 = __fmaf_rn(gs_0[5], xs_0[5], dot_0);
                    dot_0 = __fmaf_rn(gs_0[6], xs_0[6], dot_0);
                    dot_0 = __fmaf_rn(gs_0[7], xs_0[7], dot_0);
                } else {
                    float _exp_98 = __nv_expf(zs_0[0] * -1.0f);
                    float den_0 = 1.0f + _exp_98;
                    float sig_0 = __fdiv_rn(1.0f, den_0);
                    float gate_0 = __fdiv_rn(zs_0[0], den_0);
                    float nj_0 = xs_0[0] * rstd_0;
                    float aj_0 = gs_0[0] * gate_0;
                    float dzj_0 = gs_0[0] * (nj_0 * wv[0]) * sig_0 * __fmaf_rn(zs_0[0], 1.0f - sig_0, 1.0f);
                    float dnj_0 = aj_0 * wv[0];
                    dot_0 = __fmaf_rn(dnj_0, xs_0[0], dot_0);
                    acc[0] = __fmaf_rn(__uint_as_float(__as_u32(aj_0) & keep_0), __uint_as_float(__as_u32(nj_0) & keep_0), acc[0]);
                    zs_0[0] = dzj_0;
                    gs_0[0] = dnj_0;
                    float _exp_99 = __nv_expf(zs_0[1] * -1.0f);
                    float den_1 = 1.0f + _exp_99;
                    float sig_1 = __fdiv_rn(1.0f, den_1);
                    float gate_1 = __fdiv_rn(zs_0[1], den_1);
                    float nj_1 = xs_0[1] * rstd_0;
                    float aj_1 = gs_0[1] * gate_1;
                    float dzj_1 = gs_0[1] * (nj_1 * wv[1]) * sig_1 * __fmaf_rn(zs_0[1], 1.0f - sig_1, 1.0f);
                    float dnj_1 = aj_1 * wv[1];
                    dot_0 = __fmaf_rn(dnj_1, xs_0[1], dot_0);
                    acc[1] = __fmaf_rn(__uint_as_float(__as_u32(aj_1) & keep_0), __uint_as_float(__as_u32(nj_1) & keep_0), acc[1]);
                    zs_0[1] = dzj_1;
                    gs_0[1] = dnj_1;
                    float _exp_100 = __nv_expf(zs_0[2] * -1.0f);
                    float den_2 = 1.0f + _exp_100;
                    float sig_2 = __fdiv_rn(1.0f, den_2);
                    float gate_2 = __fdiv_rn(zs_0[2], den_2);
                    float nj_2 = xs_0[2] * rstd_0;
                    float aj_2 = gs_0[2] * gate_2;
                    float dzj_2 = gs_0[2] * (nj_2 * wv[2]) * sig_2 * __fmaf_rn(zs_0[2], 1.0f - sig_2, 1.0f);
                    float dnj_2 = aj_2 * wv[2];
                    dot_0 = __fmaf_rn(dnj_2, xs_0[2], dot_0);
                    acc[2] = __fmaf_rn(__uint_as_float(__as_u32(aj_2) & keep_0), __uint_as_float(__as_u32(nj_2) & keep_0), acc[2]);
                    zs_0[2] = dzj_2;
                    gs_0[2] = dnj_2;
                    float _exp_101 = __nv_expf(zs_0[3] * -1.0f);
                    float den_3 = 1.0f + _exp_101;
                    float sig_3 = __fdiv_rn(1.0f, den_3);
                    float gate_3 = __fdiv_rn(zs_0[3], den_3);
                    float nj_3 = xs_0[3] * rstd_0;
                    float aj_3 = gs_0[3] * gate_3;
                    float dzj_3 = gs_0[3] * (nj_3 * wv[3]) * sig_3 * __fmaf_rn(zs_0[3], 1.0f - sig_3, 1.0f);
                    float dnj_3 = aj_3 * wv[3];
                    dot_0 = __fmaf_rn(dnj_3, xs_0[3], dot_0);
                    acc[3] = __fmaf_rn(__uint_as_float(__as_u32(aj_3) & keep_0), __uint_as_float(__as_u32(nj_3) & keep_0), acc[3]);
                    zs_0[3] = dzj_3;
                    gs_0[3] = dnj_3;
                    float _exp_102 = __nv_expf(zs_0[4] * -1.0f);
                    float den_4 = 1.0f + _exp_102;
                    float sig_4 = __fdiv_rn(1.0f, den_4);
                    float gate_4 = __fdiv_rn(zs_0[4], den_4);
                    float nj_4 = xs_0[4] * rstd_0;
                    float aj_4 = gs_0[4] * gate_4;
                    float dzj_4 = gs_0[4] * (nj_4 * wv[4]) * sig_4 * __fmaf_rn(zs_0[4], 1.0f - sig_4, 1.0f);
                    float dnj_4 = aj_4 * wv[4];
                    dot_0 = __fmaf_rn(dnj_4, xs_0[4], dot_0);
                    acc[4] = __fmaf_rn(__uint_as_float(__as_u32(aj_4) & keep_0), __uint_as_float(__as_u32(nj_4) & keep_0), acc[4]);
                    zs_0[4] = dzj_4;
                    gs_0[4] = dnj_4;
                    float _exp_103 = __nv_expf(zs_0[5] * -1.0f);
                    float den_5 = 1.0f + _exp_103;
                    float sig_5 = __fdiv_rn(1.0f, den_5);
                    float gate_5 = __fdiv_rn(zs_0[5], den_5);
                    float nj_5 = xs_0[5] * rstd_0;
                    float aj_5 = gs_0[5] * gate_5;
                    float dzj_5 = gs_0[5] * (nj_5 * wv[5]) * sig_5 * __fmaf_rn(zs_0[5], 1.0f - sig_5, 1.0f);
                    float dnj_5 = aj_5 * wv[5];
                    dot_0 = __fmaf_rn(dnj_5, xs_0[5], dot_0);
                    acc[5] = __fmaf_rn(__uint_as_float(__as_u32(aj_5) & keep_0), __uint_as_float(__as_u32(nj_5) & keep_0), acc[5]);
                    zs_0[5] = dzj_5;
                    gs_0[5] = dnj_5;
                    float _exp_104 = __nv_expf(zs_0[6] * -1.0f);
                    float den_6 = 1.0f + _exp_104;
                    float sig_6 = __fdiv_rn(1.0f, den_6);
                    float gate_6 = __fdiv_rn(zs_0[6], den_6);
                    float nj_6 = xs_0[6] * rstd_0;
                    float aj_6 = gs_0[6] * gate_6;
                    float dzj_6 = gs_0[6] * (nj_6 * wv[6]) * sig_6 * __fmaf_rn(zs_0[6], 1.0f - sig_6, 1.0f);
                    float dnj_6 = aj_6 * wv[6];
                    dot_0 = __fmaf_rn(dnj_6, xs_0[6], dot_0);
                    acc[6] = __fmaf_rn(__uint_as_float(__as_u32(aj_6) & keep_0), __uint_as_float(__as_u32(nj_6) & keep_0), acc[6]);
                    zs_0[6] = dzj_6;
                    gs_0[6] = dnj_6;
                    float _exp_105 = __nv_expf(zs_0[7] * -1.0f);
                    float den_7 = 1.0f + _exp_105;
                    float sig_7 = __fdiv_rn(1.0f, den_7);
                    float gate_7 = __fdiv_rn(zs_0[7], den_7);
                    float nj_7 = xs_0[7] * rstd_0;
                    float aj_7 = gs_0[7] * gate_7;
                    float dzj_7 = gs_0[7] * (nj_7 * wv[7]) * sig_7 * __fmaf_rn(zs_0[7], 1.0f - sig_7, 1.0f);
                    float dnj_7 = aj_7 * wv[7];
                    dot_0 = __fmaf_rn(dnj_7, xs_0[7], dot_0);
                    acc[7] = __fmaf_rn(__uint_as_float(__as_u32(aj_7) & keep_0), __uint_as_float(__as_u32(nj_7) & keep_0), acc[7]);
                    zs_0[7] = dzj_7;
                    gs_0[7] = dnj_7;
                }
                float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_106;
                float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_107;
                float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_108;
                float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 8);
                dot_0 = dot_0 + _shfl_xor_sync_109;
                float coef_0 = rstd_0 * rstd_0 * rstd_0 * (dot_0 * -0.0078125f);
                float2 coef2_0 = make_float2(coef_0, coef_0);
                float2 rstd2b_0 = make_float2(rstd_0, rstd_0);
                float2 dx2_0 = fma_f32x2_rn_noftz(make_float2(xs_0[0], xs_0[1]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[0], gs_0[1]), rstd2b_0));
                xs_0[0] = dx2_0.x;
                xs_0[1] = dx2_0.y;
                float2 dx2_1 = fma_f32x2_rn_noftz(make_float2(xs_0[2], xs_0[3]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[2], gs_0[3]), rstd2b_0));
                xs_0[2] = dx2_1.x;
                xs_0[3] = dx2_1.y;
                float2 dx2_2 = fma_f32x2_rn_noftz(make_float2(xs_0[4], xs_0[5]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[4], gs_0[5]), rstd2b_0));
                xs_0[4] = dx2_2.x;
                xs_0[5] = dx2_2.y;
                float2 dx2_3 = fma_f32x2_rn_noftz(make_float2(xs_0[6], xs_0[7]), coef2_0, mul_f32x2_noftz(make_float2(gs_0[6], gs_0[7]), rstd2b_0));
                xs_0[6] = dx2_3.x;
                xs_0[7] = dx2_3.y;
                unsigned int packed_dx_0[4];
                unsigned int packed_dz_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 0], xs_0[_lp*2+1 + 0]));
                    packed_dx_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 0], zs_0[_lp*2+1 + 0]));
                    packed_dz_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 2], xs_0[_lp*2+1 + 2]));
                    packed_dx_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 2], zs_0[_lp*2+1 + 2]));
                    packed_dz_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 4], xs_0[_lp*2+1 + 4]));
                    packed_dx_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 4], zs_0[_lp*2+1 + 4]));
                    packed_dz_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 6], xs_0[_lp*2+1 + 6]));
                    packed_dx_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 6], zs_0[_lp*2+1 + 6]));
                    packed_dz_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row_0 < rows) {
                    long long orow_0 = (long long)row_0 * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_0)[0] = reinterpret_cast<int4*>(packed_dx_0)[0];
                    reinterpret_cast<int4*>(dz + orow_0)[0] = reinterpret_cast<int4*>(packed_dz_0)[0];
                }
            } else {
                float total_1 = 0.0f;
                total_1 = __fmaf_rn(xs_0[0], xs_0[0], total_1);
                total_1 = __fmaf_rn(xs_0[1], xs_0[1], total_1);
                total_1 = __fmaf_rn(xs_0[2], xs_0[2], total_1);
                total_1 = __fmaf_rn(xs_0[3], xs_0[3], total_1);
                total_1 = __fmaf_rn(xs_0[4], xs_0[4], total_1);
                total_1 = __fmaf_rn(xs_0[5], xs_0[5], total_1);
                total_1 = __fmaf_rn(xs_0[6], xs_0[6], total_1);
                total_1 = __fmaf_rn(xs_0[7], xs_0[7], total_1);
                float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
                total_1 = total_1 + _shfl_xor_sync_110;
                float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
                total_1 = total_1 + _shfl_xor_sync_111;
                float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
                total_1 = total_1 + _shfl_xor_sync_112;
                float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
                total_1 = total_1 + _shfl_xor_sync_113;
                float _sqrt_114;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_114) : "f"(total_1 * 0.0078125f + eps));
                float rstd_1 = __frcp_rn(_sqrt_114);
                unsigned int keep_1 = 4294967295u;
                float dot_1 = 0.0f;
                int fast_1 = 0;
                float _neg_115;
                asm("neg.f32 %0, %1;" : "=f"(_neg_115) : "f"(zs_0[0]));
                float _neg_116;
                asm("neg.f32 %0, %1;" : "=f"(_neg_116) : "f"(zs_0[1]));
                float _neg_117;
                asm("neg.f32 %0, %1;" : "=f"(_neg_117) : "f"(zs_0[2]));
                float mz_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_1) : "f"(_neg_115), "f"(_neg_116), "f"(_neg_117));
                float _abs_119;
                asm("abs.f32 %0, %1;" : "=f"(_abs_119) : "f"(zs_0[0]));
                float _neg_118;
                asm("neg.f32 %0, %1;" : "=f"(_neg_118) : "f"(_abs_119));
                float _abs_121;
                asm("abs.f32 %0, %1;" : "=f"(_abs_121) : "f"(zs_0[1]));
                float _neg_120;
                asm("neg.f32 %0, %1;" : "=f"(_neg_120) : "f"(_abs_121));
                float _abs_123;
                asm("abs.f32 %0, %1;" : "=f"(_abs_123) : "f"(zs_0[2]));
                float _neg_122;
                asm("neg.f32 %0, %1;" : "=f"(_neg_122) : "f"(_abs_123));
                float ma_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_1) : "f"(_neg_118), "f"(_neg_120), "f"(_neg_122));
                float _neg_125;
                asm("neg.f32 %0, %1;" : "=f"(_neg_125) : "f"(zs_0[3]));
                float _neg_126;
                asm("neg.f32 %0, %1;" : "=f"(_neg_126) : "f"(zs_0[4]));
                float _max3_124;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_124) : "f"(mz_1), "f"(_neg_125), "f"(_neg_126));
                mz_1 = _max3_124;
                float _abs_129;
                asm("abs.f32 %0, %1;" : "=f"(_abs_129) : "f"(zs_0[3]));
                float _neg_128;
                asm("neg.f32 %0, %1;" : "=f"(_neg_128) : "f"(_abs_129));
                float _abs_131;
                asm("abs.f32 %0, %1;" : "=f"(_abs_131) : "f"(zs_0[4]));
                float _neg_130;
                asm("neg.f32 %0, %1;" : "=f"(_neg_130) : "f"(_abs_131));
                float _max3_127;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_127) : "f"(ma_1), "f"(_neg_128), "f"(_neg_130));
                ma_1 = _max3_127;
                float _neg_133;
                asm("neg.f32 %0, %1;" : "=f"(_neg_133) : "f"(zs_0[5]));
                float _neg_134;
                asm("neg.f32 %0, %1;" : "=f"(_neg_134) : "f"(zs_0[6]));
                float _max3_132;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_132) : "f"(mz_1), "f"(_neg_133), "f"(_neg_134));
                mz_1 = _max3_132;
                float _abs_137;
                asm("abs.f32 %0, %1;" : "=f"(_abs_137) : "f"(zs_0[5]));
                float _neg_136;
                asm("neg.f32 %0, %1;" : "=f"(_neg_136) : "f"(_abs_137));
                float _abs_139;
                asm("abs.f32 %0, %1;" : "=f"(_abs_139) : "f"(zs_0[6]));
                float _neg_138;
                asm("neg.f32 %0, %1;" : "=f"(_neg_138) : "f"(_abs_139));
                float _max3_135;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_135) : "f"(ma_1), "f"(_neg_136), "f"(_neg_138));
                ma_1 = _max3_135;
                float _neg_141;
                asm("neg.f32 %0, %1;" : "=f"(_neg_141) : "f"(zs_0[7]));
                float _max_140;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_140) : "f"(mz_1), "f"(_neg_141));
                mz_1 = _max_140;
                float _abs_144;
                asm("abs.f32 %0, %1;" : "=f"(_abs_144) : "f"(zs_0[7]));
                float _neg_143;
                asm("neg.f32 %0, %1;" : "=f"(_neg_143) : "f"(_abs_144));
                float _max_142;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_142) : "f"(ma_1), "f"(_neg_143));
                ma_1 = _max_142;
                float _abs_145;
                asm("abs.f32 %0, %1;" : "=f"(_abs_145) : "f"(zs_0[0]));
                float _abs_146;
                asm("abs.f32 %0, %1;" : "=f"(_abs_146) : "f"(zs_0[1]));
                float _abs_147;
                asm("abs.f32 %0, %1;" : "=f"(_abs_147) : "f"(zs_0[2]));
                float mp_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_1) : "f"(_abs_145), "f"(_abs_146), "f"(_abs_147));
                float _abs_149;
                asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(zs_0[3]));
                float _abs_150;
                asm("abs.f32 %0, %1;" : "=f"(_abs_150) : "f"(zs_0[4]));
                float _max3_148;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_148) : "f"(mp_1), "f"(_abs_149), "f"(_abs_150));
                mp_1 = _max3_148;
                float _abs_152;
                asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(zs_0[5]));
                float _abs_153;
                asm("abs.f32 %0, %1;" : "=f"(_abs_153) : "f"(zs_0[6]));
                float _max3_151;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_151) : "f"(mp_1), "f"(_abs_152), "f"(_abs_153));
                mp_1 = _max3_151;
                float _abs_155;
                asm("abs.f32 %0, %1;" : "=f"(_abs_155) : "f"(zs_0[7]));
                float _max_154;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_154) : "f"(mp_1), "f"(_abs_155));
                mp_1 = _max_154;
                if (mz_1 <= 41.0f) {
                    if (ma_1 <= -1.1754944e-38f) {
                        if (mp_1 <= 3.4028235e+38f) {
                            fast_1 = 1;
                        }
                    }
                }
                if (fast_1 != 0) {
                    float2 rstd2_1 = make_float2(rstd_1, rstd_1);
                    float2 z2_4 = make_float2(zs_0[0], zs_0[1]);
                    float ta_4;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_4) : "f"(zs_0[0]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_4;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_4) : "f"(zs_0[1]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_156;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(ta_4), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_157;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(tb_4), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_4 = make_float2(_fma_156, _fma_157);
                    float2 r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_4));
                    r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.925963e-08f, -1.925963e-08f), r2_4);
                    float2 nscale_4 = make_float2(__uint_as_float((__as_u32(t2_4.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_4.y) << 23) + 2147483648u));
                    float _exp2_158 = approx_exp2(r2_4.x);
                    float _exp2_159 = approx_exp2(r2_4.y);
                    float2 nden_4 = fma_f32x2_rn_noftz(nscale_4, make_float2(_exp2_158, _exp2_159), make_float2(-1.0f, -1.0f));
                    float _neg_161;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_161) : "f"(nden_4.x));
                    float _rcp_160 = approx_rcp(_neg_161);
                    float _neg_163;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_163) : "f"(nden_4.y));
                    float _rcp_162 = approx_rcp(_neg_163);
                    float2 r0_4 = make_float2(_rcp_160, _rcp_162);
                    float2 r1_4 = fma_f32x2_rn_noftz(r0_4, fma_f32x2_rn_noftz(nden_4, r0_4, make_float2(1.0f, 1.0f)), r0_4);
                    float2 sig2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, r1_4, make_float2(1.0f, 1.0f)), r1_4);
                    float2 q0_4 = fma_f32x2_rn_noftz(z2_4, r1_4, make_float2(0.0f, 0.0f));
                    float2 gate2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, q0_4, z2_4), q0_4);
                    float2 g2_4 = make_float2(gs_0[0], gs_0[1]);
                    float2 w2_4 = make_float2(wv[0], wv[1]);
                    float2 _f32x2_mul_a_164 = make_float2(xs_0[0], xs_0[1]);
                    float2 n2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_165 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
                    float2 _f32x2_mul_b_166 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
                    float2 dz2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&_f32x2_mul_b_166));
                    float2 a2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&gate2_4));
                    float2 dn2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_4) : "l"(*(const unsigned long long*)&a2_4), "l"(*(const unsigned long long*)&w2_4));
                    float2 acc2_4 = make_float2(acc[0], acc[1]);
                    acc2_4 = fma_f32x2_rn_noftz(a2_4, n2_4, acc2_4);
                    acc[0] = acc2_4.x;
                    acc[1] = acc2_4.y;
                    zs_0[0] = dz2_4.x;
                    zs_0[1] = dz2_4.y;
                    gs_0[0] = dn2_4.x;
                    gs_0[1] = dn2_4.y;
                    float2 z2_5 = make_float2(zs_0[2], zs_0[3]);
                    float ta_5;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_5) : "f"(zs_0[2]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_5;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_5) : "f"(zs_0[3]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_167;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(ta_5), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_168;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(tb_5), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_5 = make_float2(_fma_167, _fma_168);
                    float2 r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_5));
                    r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.925963e-08f, -1.925963e-08f), r2_5);
                    float2 nscale_5 = make_float2(__uint_as_float((__as_u32(t2_5.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_5.y) << 23) + 2147483648u));
                    float _exp2_169 = approx_exp2(r2_5.x);
                    float _exp2_170 = approx_exp2(r2_5.y);
                    float2 nden_5 = fma_f32x2_rn_noftz(nscale_5, make_float2(_exp2_169, _exp2_170), make_float2(-1.0f, -1.0f));
                    float _neg_172;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_172) : "f"(nden_5.x));
                    float _rcp_171 = approx_rcp(_neg_172);
                    float _neg_174;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_174) : "f"(nden_5.y));
                    float _rcp_173 = approx_rcp(_neg_174);
                    float2 r0_5 = make_float2(_rcp_171, _rcp_173);
                    float2 r1_5 = fma_f32x2_rn_noftz(r0_5, fma_f32x2_rn_noftz(nden_5, r0_5, make_float2(1.0f, 1.0f)), r0_5);
                    float2 sig2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, r1_5, make_float2(1.0f, 1.0f)), r1_5);
                    float2 q0_5 = fma_f32x2_rn_noftz(z2_5, r1_5, make_float2(0.0f, 0.0f));
                    float2 gate2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, q0_5, z2_5), q0_5);
                    float2 g2_5 = make_float2(gs_0[2], gs_0[3]);
                    float2 w2_5 = make_float2(wv[2], wv[3]);
                    float2 _f32x2_mul_a_175 = make_float2(xs_0[2], xs_0[3]);
                    float2 n2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_176 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
                    float2 _f32x2_mul_b_177 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
                    float2 dz2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
                    float2 a2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&gate2_5));
                    float2 dn2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_5) : "l"(*(const unsigned long long*)&a2_5), "l"(*(const unsigned long long*)&w2_5));
                    float2 acc2_5 = make_float2(acc[2], acc[3]);
                    acc2_5 = fma_f32x2_rn_noftz(a2_5, n2_5, acc2_5);
                    acc[2] = acc2_5.x;
                    acc[3] = acc2_5.y;
                    zs_0[2] = dz2_5.x;
                    zs_0[3] = dz2_5.y;
                    gs_0[2] = dn2_5.x;
                    gs_0[3] = dn2_5.y;
                    float2 z2_6 = make_float2(zs_0[4], zs_0[5]);
                    float ta_6;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_6) : "f"(zs_0[4]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_6;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_6) : "f"(zs_0[5]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_178;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(ta_6), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_179;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(tb_6), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_6 = make_float2(_fma_178, _fma_179);
                    float2 r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_6));
                    r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.925963e-08f, -1.925963e-08f), r2_6);
                    float2 nscale_6 = make_float2(__uint_as_float((__as_u32(t2_6.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_6.y) << 23) + 2147483648u));
                    float _exp2_180 = approx_exp2(r2_6.x);
                    float _exp2_181 = approx_exp2(r2_6.y);
                    float2 nden_6 = fma_f32x2_rn_noftz(nscale_6, make_float2(_exp2_180, _exp2_181), make_float2(-1.0f, -1.0f));
                    float _neg_183;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_183) : "f"(nden_6.x));
                    float _rcp_182 = approx_rcp(_neg_183);
                    float _neg_185;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_185) : "f"(nden_6.y));
                    float _rcp_184 = approx_rcp(_neg_185);
                    float2 r0_6 = make_float2(_rcp_182, _rcp_184);
                    float2 r1_6 = fma_f32x2_rn_noftz(r0_6, fma_f32x2_rn_noftz(nden_6, r0_6, make_float2(1.0f, 1.0f)), r0_6);
                    float2 sig2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, r1_6, make_float2(1.0f, 1.0f)), r1_6);
                    float2 q0_6 = fma_f32x2_rn_noftz(z2_6, r1_6, make_float2(0.0f, 0.0f));
                    float2 gate2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, q0_6, z2_6), q0_6);
                    float2 g2_6 = make_float2(gs_0[4], gs_0[5]);
                    float2 w2_6 = make_float2(wv[4], wv[5]);
                    float2 _f32x2_mul_a_186 = make_float2(xs_0[4], xs_0[5]);
                    float2 n2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_187 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
                    float2 _f32x2_mul_b_188 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
                    float2 dz2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&_f32x2_mul_b_188));
                    float2 a2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&gate2_6));
                    float2 dn2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_6) : "l"(*(const unsigned long long*)&a2_6), "l"(*(const unsigned long long*)&w2_6));
                    float2 acc2_6 = make_float2(acc[4], acc[5]);
                    acc2_6 = fma_f32x2_rn_noftz(a2_6, n2_6, acc2_6);
                    acc[4] = acc2_6.x;
                    acc[5] = acc2_6.y;
                    zs_0[4] = dz2_6.x;
                    zs_0[5] = dz2_6.y;
                    gs_0[4] = dn2_6.x;
                    gs_0[5] = dn2_6.y;
                    float2 z2_7 = make_float2(zs_0[6], zs_0[7]);
                    float ta_7;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_7) : "f"(zs_0[6]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_7;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_7) : "f"(zs_0[7]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_189;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(ta_7), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_190;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(tb_7), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_7 = make_float2(_fma_189, _fma_190);
                    float2 r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_7));
                    r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.925963e-08f, -1.925963e-08f), r2_7);
                    float2 nscale_7 = make_float2(__uint_as_float((__as_u32(t2_7.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_7.y) << 23) + 2147483648u));
                    float _exp2_191 = approx_exp2(r2_7.x);
                    float _exp2_192 = approx_exp2(r2_7.y);
                    float2 nden_7 = fma_f32x2_rn_noftz(nscale_7, make_float2(_exp2_191, _exp2_192), make_float2(-1.0f, -1.0f));
                    float _neg_194;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_194) : "f"(nden_7.x));
                    float _rcp_193 = approx_rcp(_neg_194);
                    float _neg_196;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_196) : "f"(nden_7.y));
                    float _rcp_195 = approx_rcp(_neg_196);
                    float2 r0_7 = make_float2(_rcp_193, _rcp_195);
                    float2 r1_7 = fma_f32x2_rn_noftz(r0_7, fma_f32x2_rn_noftz(nden_7, r0_7, make_float2(1.0f, 1.0f)), r0_7);
                    float2 sig2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, r1_7, make_float2(1.0f, 1.0f)), r1_7);
                    float2 q0_7 = fma_f32x2_rn_noftz(z2_7, r1_7, make_float2(0.0f, 0.0f));
                    float2 gate2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, q0_7, z2_7), q0_7);
                    float2 g2_7 = make_float2(gs_0[6], gs_0[7]);
                    float2 w2_7 = make_float2(wv[6], wv[7]);
                    float2 _f32x2_mul_a_197 = make_float2(xs_0[6], xs_0[7]);
                    float2 n2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_198 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
                    float2 _f32x2_mul_b_199 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
                    float2 dz2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_198), "l"(*(const unsigned long long*)&_f32x2_mul_b_199));
                    float2 a2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&gate2_7));
                    float2 dn2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_7) : "l"(*(const unsigned long long*)&a2_7), "l"(*(const unsigned long long*)&w2_7));
                    float2 acc2_7 = make_float2(acc[6], acc[7]);
                    acc2_7 = fma_f32x2_rn_noftz(a2_7, n2_7, acc2_7);
                    acc[6] = acc2_7.x;
                    acc[7] = acc2_7.y;
                    zs_0[6] = dz2_7.x;
                    zs_0[7] = dz2_7.y;
                    gs_0[6] = dn2_7.x;
                    gs_0[7] = dn2_7.y;
                    dot_1 = __fmaf_rn(gs_0[0], xs_0[0], dot_1);
                    dot_1 = __fmaf_rn(gs_0[1], xs_0[1], dot_1);
                    dot_1 = __fmaf_rn(gs_0[2], xs_0[2], dot_1);
                    dot_1 = __fmaf_rn(gs_0[3], xs_0[3], dot_1);
                    dot_1 = __fmaf_rn(gs_0[4], xs_0[4], dot_1);
                    dot_1 = __fmaf_rn(gs_0[5], xs_0[5], dot_1);
                    dot_1 = __fmaf_rn(gs_0[6], xs_0[6], dot_1);
                    dot_1 = __fmaf_rn(gs_0[7], xs_0[7], dot_1);
                } else {
                    float _exp_200 = __nv_expf(zs_0[0] * -1.0f);
                    float den_8 = 1.0f + _exp_200;
                    float sig_8 = __fdiv_rn(1.0f, den_8);
                    float gate_8 = __fdiv_rn(zs_0[0], den_8);
                    float nj_8 = xs_0[0] * rstd_1;
                    float aj_8 = gs_0[0] * gate_8;
                    float dzj_8 = gs_0[0] * (nj_8 * wv[0]) * sig_8 * __fmaf_rn(zs_0[0], 1.0f - sig_8, 1.0f);
                    float dnj_8 = aj_8 * wv[0];
                    dot_1 = __fmaf_rn(dnj_8, xs_0[0], dot_1);
                    acc[0] = __fmaf_rn(aj_8, nj_8, acc[0]);
                    zs_0[0] = dzj_8;
                    gs_0[0] = dnj_8;
                    float _exp_201 = __nv_expf(zs_0[1] * -1.0f);
                    float den_9 = 1.0f + _exp_201;
                    float sig_9 = __fdiv_rn(1.0f, den_9);
                    float gate_9 = __fdiv_rn(zs_0[1], den_9);
                    float nj_9 = xs_0[1] * rstd_1;
                    float aj_9 = gs_0[1] * gate_9;
                    float dzj_9 = gs_0[1] * (nj_9 * wv[1]) * sig_9 * __fmaf_rn(zs_0[1], 1.0f - sig_9, 1.0f);
                    float dnj_9 = aj_9 * wv[1];
                    dot_1 = __fmaf_rn(dnj_9, xs_0[1], dot_1);
                    acc[1] = __fmaf_rn(aj_9, nj_9, acc[1]);
                    zs_0[1] = dzj_9;
                    gs_0[1] = dnj_9;
                    float _exp_202 = __nv_expf(zs_0[2] * -1.0f);
                    float den_10 = 1.0f + _exp_202;
                    float sig_10 = __fdiv_rn(1.0f, den_10);
                    float gate_10 = __fdiv_rn(zs_0[2], den_10);
                    float nj_10 = xs_0[2] * rstd_1;
                    float aj_10 = gs_0[2] * gate_10;
                    float dzj_10 = gs_0[2] * (nj_10 * wv[2]) * sig_10 * __fmaf_rn(zs_0[2], 1.0f - sig_10, 1.0f);
                    float dnj_10 = aj_10 * wv[2];
                    dot_1 = __fmaf_rn(dnj_10, xs_0[2], dot_1);
                    acc[2] = __fmaf_rn(aj_10, nj_10, acc[2]);
                    zs_0[2] = dzj_10;
                    gs_0[2] = dnj_10;
                    float _exp_203 = __nv_expf(zs_0[3] * -1.0f);
                    float den_11 = 1.0f + _exp_203;
                    float sig_11 = __fdiv_rn(1.0f, den_11);
                    float gate_11 = __fdiv_rn(zs_0[3], den_11);
                    float nj_11 = xs_0[3] * rstd_1;
                    float aj_11 = gs_0[3] * gate_11;
                    float dzj_11 = gs_0[3] * (nj_11 * wv[3]) * sig_11 * __fmaf_rn(zs_0[3], 1.0f - sig_11, 1.0f);
                    float dnj_11 = aj_11 * wv[3];
                    dot_1 = __fmaf_rn(dnj_11, xs_0[3], dot_1);
                    acc[3] = __fmaf_rn(aj_11, nj_11, acc[3]);
                    zs_0[3] = dzj_11;
                    gs_0[3] = dnj_11;
                    float _exp_204 = __nv_expf(zs_0[4] * -1.0f);
                    float den_12 = 1.0f + _exp_204;
                    float sig_12 = __fdiv_rn(1.0f, den_12);
                    float gate_12 = __fdiv_rn(zs_0[4], den_12);
                    float nj_12 = xs_0[4] * rstd_1;
                    float aj_12 = gs_0[4] * gate_12;
                    float dzj_12 = gs_0[4] * (nj_12 * wv[4]) * sig_12 * __fmaf_rn(zs_0[4], 1.0f - sig_12, 1.0f);
                    float dnj_12 = aj_12 * wv[4];
                    dot_1 = __fmaf_rn(dnj_12, xs_0[4], dot_1);
                    acc[4] = __fmaf_rn(aj_12, nj_12, acc[4]);
                    zs_0[4] = dzj_12;
                    gs_0[4] = dnj_12;
                    float _exp_205 = __nv_expf(zs_0[5] * -1.0f);
                    float den_13 = 1.0f + _exp_205;
                    float sig_13 = __fdiv_rn(1.0f, den_13);
                    float gate_13 = __fdiv_rn(zs_0[5], den_13);
                    float nj_13 = xs_0[5] * rstd_1;
                    float aj_13 = gs_0[5] * gate_13;
                    float dzj_13 = gs_0[5] * (nj_13 * wv[5]) * sig_13 * __fmaf_rn(zs_0[5], 1.0f - sig_13, 1.0f);
                    float dnj_13 = aj_13 * wv[5];
                    dot_1 = __fmaf_rn(dnj_13, xs_0[5], dot_1);
                    acc[5] = __fmaf_rn(aj_13, nj_13, acc[5]);
                    zs_0[5] = dzj_13;
                    gs_0[5] = dnj_13;
                    float _exp_206 = __nv_expf(zs_0[6] * -1.0f);
                    float den_14 = 1.0f + _exp_206;
                    float sig_14 = __fdiv_rn(1.0f, den_14);
                    float gate_14 = __fdiv_rn(zs_0[6], den_14);
                    float nj_14 = xs_0[6] * rstd_1;
                    float aj_14 = gs_0[6] * gate_14;
                    float dzj_14 = gs_0[6] * (nj_14 * wv[6]) * sig_14 * __fmaf_rn(zs_0[6], 1.0f - sig_14, 1.0f);
                    float dnj_14 = aj_14 * wv[6];
                    dot_1 = __fmaf_rn(dnj_14, xs_0[6], dot_1);
                    acc[6] = __fmaf_rn(aj_14, nj_14, acc[6]);
                    zs_0[6] = dzj_14;
                    gs_0[6] = dnj_14;
                    float _exp_207 = __nv_expf(zs_0[7] * -1.0f);
                    float den_15 = 1.0f + _exp_207;
                    float sig_15 = __fdiv_rn(1.0f, den_15);
                    float gate_15 = __fdiv_rn(zs_0[7], den_15);
                    float nj_15 = xs_0[7] * rstd_1;
                    float aj_15 = gs_0[7] * gate_15;
                    float dzj_15 = gs_0[7] * (nj_15 * wv[7]) * sig_15 * __fmaf_rn(zs_0[7], 1.0f - sig_15, 1.0f);
                    float dnj_15 = aj_15 * wv[7];
                    dot_1 = __fmaf_rn(dnj_15, xs_0[7], dot_1);
                    acc[7] = __fmaf_rn(aj_15, nj_15, acc[7]);
                    zs_0[7] = dzj_15;
                    gs_0[7] = dnj_15;
                }
                float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_208;
                float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_209;
                float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_210;
                float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
                dot_1 = dot_1 + _shfl_xor_sync_211;
                float coef_1 = rstd_1 * rstd_1 * rstd_1 * (dot_1 * -0.0078125f);
                float2 coef2_1 = make_float2(coef_1, coef_1);
                float2 rstd2b_1 = make_float2(rstd_1, rstd_1);
                float2 dx2_4 = fma_f32x2_rn_noftz(make_float2(xs_0[0], xs_0[1]), coef2_1, mul_f32x2_noftz(make_float2(gs_0[0], gs_0[1]), rstd2b_1));
                xs_0[0] = dx2_4.x;
                xs_0[1] = dx2_4.y;
                float2 dx2_5 = fma_f32x2_rn_noftz(make_float2(xs_0[2], xs_0[3]), coef2_1, mul_f32x2_noftz(make_float2(gs_0[2], gs_0[3]), rstd2b_1));
                xs_0[2] = dx2_5.x;
                xs_0[3] = dx2_5.y;
                float2 dx2_6 = fma_f32x2_rn_noftz(make_float2(xs_0[4], xs_0[5]), coef2_1, mul_f32x2_noftz(make_float2(gs_0[4], gs_0[5]), rstd2b_1));
                xs_0[4] = dx2_6.x;
                xs_0[5] = dx2_6.y;
                float2 dx2_7 = fma_f32x2_rn_noftz(make_float2(xs_0[6], xs_0[7]), coef2_1, mul_f32x2_noftz(make_float2(gs_0[6], gs_0[7]), rstd2b_1));
                xs_0[6] = dx2_7.x;
                xs_0[7] = dx2_7.y;
                unsigned int packed_dx_1[4];
                unsigned int packed_dz_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 0], xs_0[_lp*2+1 + 0]));
                    packed_dx_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 0], zs_0[_lp*2+1 + 0]));
                    packed_dz_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 2], xs_0[_lp*2+1 + 2]));
                    packed_dx_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 2], zs_0[_lp*2+1 + 2]));
                    packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 4], xs_0[_lp*2+1 + 4]));
                    packed_dx_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 4], zs_0[_lp*2+1 + 4]));
                    packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 6], xs_0[_lp*2+1 + 6]));
                    packed_dx_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_0[_lp*2 + 6], zs_0[_lp*2+1 + 6]));
                    packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row_0 < rows) {
                    long long orow_1 = (long long)row_0 * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                    reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
                }
            }
            run = run + 1;
            if (run == 16) {
                if (folded != 0) {
                    float s;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(fold[group * 128 + col]), "f"(acc[0]));
                    float bv;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv) : "f"(s), "f"(fold[group * 128 + col]));
                    float av;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av) : "f"(s), "f"(bv));
                    float _sub_212;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_212) : "f"(fold[group * 128 + col]), "f"(av));
                    float _sub_213;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_213) : "f"(acc[0]), "f"(bv));
                    float e;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_212), "f"(_sub_213));
                    fold[group * 128 + col] = s;
                    float _add_214;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(fold[(16 + group) * 128 + col]), "f"(e));
                    fold[(16 + group) * 128 + col] = _add_214;
                    acc[0] = 0.0f;
                    float s_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_0) : "f"(fold[group * 128 + col + 1]), "f"(acc[1]));
                    float bv_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_0) : "f"(s_0), "f"(fold[group * 128 + col + 1]));
                    float av_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_0) : "f"(s_0), "f"(bv_0));
                    float _sub_215;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_215) : "f"(fold[group * 128 + col + 1]), "f"(av_0));
                    float _sub_216;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_216) : "f"(acc[1]), "f"(bv_0));
                    float e_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_215), "f"(_sub_216));
                    fold[group * 128 + col + 1] = s_0;
                    float _add_217;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_0));
                    fold[(16 + group) * 128 + col + 1] = _add_217;
                    acc[1] = 0.0f;
                    float s_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(fold[group * 128 + col + 2]), "f"(acc[2]));
                    float bv_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_1) : "f"(s_1), "f"(fold[group * 128 + col + 2]));
                    float av_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_1) : "f"(s_1), "f"(bv_1));
                    float _sub_218;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_218) : "f"(fold[group * 128 + col + 2]), "f"(av_1));
                    float _sub_219;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_219) : "f"(acc[2]), "f"(bv_1));
                    float e_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_218), "f"(_sub_219));
                    fold[group * 128 + col + 2] = s_1;
                    float _add_220;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_1));
                    fold[(16 + group) * 128 + col + 2] = _add_220;
                    acc[2] = 0.0f;
                    float s_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_2) : "f"(fold[group * 128 + col + 3]), "f"(acc[3]));
                    float bv_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_2) : "f"(s_2), "f"(fold[group * 128 + col + 3]));
                    float av_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_2) : "f"(s_2), "f"(bv_2));
                    float _sub_221;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_221) : "f"(fold[group * 128 + col + 3]), "f"(av_2));
                    float _sub_222;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_222) : "f"(acc[3]), "f"(bv_2));
                    float e_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_221), "f"(_sub_222));
                    fold[group * 128 + col + 3] = s_2;
                    float _add_223;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_2));
                    fold[(16 + group) * 128 + col + 3] = _add_223;
                    acc[3] = 0.0f;
                    float s_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_3) : "f"(fold[group * 128 + col + 4]), "f"(acc[4]));
                    float bv_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_3) : "f"(s_3), "f"(fold[group * 128 + col + 4]));
                    float av_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_3) : "f"(s_3), "f"(bv_3));
                    float _sub_224;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_224) : "f"(fold[group * 128 + col + 4]), "f"(av_3));
                    float _sub_225;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_225) : "f"(acc[4]), "f"(bv_3));
                    float e_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_224), "f"(_sub_225));
                    fold[group * 128 + col + 4] = s_3;
                    float _add_226;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_3));
                    fold[(16 + group) * 128 + col + 4] = _add_226;
                    acc[4] = 0.0f;
                    float s_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_4) : "f"(fold[group * 128 + col + 5]), "f"(acc[5]));
                    float bv_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_4) : "f"(s_4), "f"(fold[group * 128 + col + 5]));
                    float av_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_4) : "f"(s_4), "f"(bv_4));
                    float _sub_227;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_227) : "f"(fold[group * 128 + col + 5]), "f"(av_4));
                    float _sub_228;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_228) : "f"(acc[5]), "f"(bv_4));
                    float e_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_4) : "f"(_sub_227), "f"(_sub_228));
                    fold[group * 128 + col + 5] = s_4;
                    float _add_229;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_4));
                    fold[(16 + group) * 128 + col + 5] = _add_229;
                    acc[5] = 0.0f;
                    float s_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_5) : "f"(fold[group * 128 + col + 6]), "f"(acc[6]));
                    float bv_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_5) : "f"(s_5), "f"(fold[group * 128 + col + 6]));
                    float av_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_5) : "f"(s_5), "f"(bv_5));
                    float _sub_230;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_230) : "f"(fold[group * 128 + col + 6]), "f"(av_5));
                    float _sub_231;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_231) : "f"(acc[6]), "f"(bv_5));
                    float e_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_5) : "f"(_sub_230), "f"(_sub_231));
                    fold[group * 128 + col + 6] = s_5;
                    float _add_232;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_5));
                    fold[(16 + group) * 128 + col + 6] = _add_232;
                    acc[6] = 0.0f;
                    float s_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_6) : "f"(fold[group * 128 + col + 7]), "f"(acc[7]));
                    float bv_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_6) : "f"(s_6), "f"(fold[group * 128 + col + 7]));
                    float av_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_6) : "f"(s_6), "f"(bv_6));
                    float _sub_233;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_233) : "f"(fold[group * 128 + col + 7]), "f"(av_6));
                    float _sub_234;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_234) : "f"(acc[7]), "f"(bv_6));
                    float e_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_6) : "f"(_sub_233), "f"(_sub_234));
                    fold[group * 128 + col + 7] = s_6;
                    float _add_235;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_6));
                    fold[(16 + group) * 128 + col + 7] = _add_235;
                    acc[7] = 0.0f;
                } else {
                    fold[group * 128 + col] = acc[0];
                    fold[(16 + group) * 128 + col] = 0.0f;
                    acc[0] = 0.0f;
                    fold[group * 128 + col + 1] = acc[1];
                    fold[(16 + group) * 128 + col + 1] = 0.0f;
                    acc[1] = 0.0f;
                    fold[group * 128 + col + 2] = acc[2];
                    fold[(16 + group) * 128 + col + 2] = 0.0f;
                    acc[2] = 0.0f;
                    fold[group * 128 + col + 3] = acc[3];
                    fold[(16 + group) * 128 + col + 3] = 0.0f;
                    acc[3] = 0.0f;
                    fold[group * 128 + col + 4] = acc[4];
                    fold[(16 + group) * 128 + col + 4] = 0.0f;
                    acc[4] = 0.0f;
                    fold[group * 128 + col + 5] = acc[5];
                    fold[(16 + group) * 128 + col + 5] = 0.0f;
                    acc[5] = 0.0f;
                    fold[group * 128 + col + 6] = acc[6];
                    fold[(16 + group) * 128 + col + 6] = 0.0f;
                    acc[6] = 0.0f;
                    fold[group * 128 + col + 7] = acc[7];
                    fold[(16 + group) * 128 + col + 7] = 0.0f;
                    acc[7] = 0.0f;
                }
                folded = 1;
                run = 0;
            }
        }
    }
    if (rest > 0) {
        {
            const uint4* _vptr_236 = reinterpret_cast<const uint4*>(w + col + 0);
            uint4 _vld_236[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_vld_236[_blk].x), "=r"(_vld_236[_blk].y), "=r"(_vld_236[_blk].z), "=r"(_vld_236[_blk].w) : "l"((const void*)(_vptr_236 + _blk)) : "memory");
                uint32_t* _vpairs_236 = reinterpret_cast<uint32_t*>(&_vld_236[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&wclaim[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_236[_pair]) << 16);
                    (&wclaim[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_236[_pair]) & 0xffff0000u);
                }
            }
        }
    }
    int trips = 0;
    if (rest > 0) {
        int _min_237 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_237) > (1) ? (_min_237) : (1));
        int t_share = (rest + t_sl - 1) / t_sl;
        int t_tail = (((t_share + 63) / 64) > (1) ? ((t_share + 63) / 64) : (1));
        int t_rem = rest;
        int t_size = (t_share + 1) / 2;
        int t_live = 1;
        trips = 1;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        t_live = t_live * ((t_size >= t_tail) ? 1 : 0) * ((t_rem >= t_sl * t_size) ? 1 : 0);
        trips = trips + t_live;
        t_rem = t_rem - t_live * t_sl * t_size;
        t_size = (t_size + 1) / 2;
        trips = trips + (t_rem + t_sl - 1) / t_sl;
    }
    int claim_at = -1;
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
                    claim_at = count - 1;
                }
            }
            cell[half] = begin;
            cell[half + 1] = count;
        }
        __syncthreads();
        int chunk_lo = cell[half];
        int chunk_n = cell[half + 1];
        if (chunk_n > 0) {
            #pragma unroll 1
            for (int i_1 = 0; i_1 < chunk_n; i_1++) {
                int u_1 = bound + chunk_lo + i_1;
                int row_1 = u_1 * 16 + group;
                int src_1 = ((row_1) < (rows - 1) ? (row_1) : (rows - 1));
                uint32_t tq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_1);
                uint32_t th_1 = (uint32_t)((unsigned int)src_1) - tq_1 * (uint32_t)(heads.divisor);
                long long goff_1 = (long long)tq_1 * dy_tok + (long long)th_1 * dy_head + (long long)col;
                long long xoff_1 = (long long)tq_1 * x_tok + (long long)th_1 * x_head + (long long)col;
                long long zoff_1 = (long long)tq_1 * z_tok + (long long)th_1 * z_head + (long long)col;
                float gs_1[8];
                float xs_1[8];
                float zs_1[8];
                {
                    const uint4* _vptr_238 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                    uint4 _vld_238[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_238[_blk] = _vptr_238[_blk];
                        uint32_t* _vpairs_238 = reinterpret_cast<uint32_t*>(&_vld_238[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_238[_pair]) << 16);
                            (&gs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_238[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_239 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                    uint4 _vld_239[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_239[_blk] = _vptr_239[_blk];
                        uint32_t* _vpairs_239 = reinterpret_cast<uint32_t*>(&_vld_239[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_239[_pair]) << 16);
                            (&xs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_239[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_240 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                    uint4 _vld_240[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_240[_blk] = _vptr_240[_blk];
                        uint32_t* _vpairs_240 = reinterpret_cast<uint32_t*>(&_vld_240[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_240[_pair]) << 16);
                            (&zs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_240[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (i_1 + 1 < chunk_n) {
                    if (packed != 0) {
                        if (tid == 0) {
                            int nfirst_1 = (u_1 + 1) * 16;
                            int _min_241 = ((rows - nfirst_1) < (16) ? (rows - nfirst_1) : (16));
                            int nbytes_1 = _min_241 * 256;
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst_1 * 128ll), "r"((uint32_t)(nbytes_1)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst_1 * 128ll), "r"((uint32_t)(nbytes_1)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst_1 * 128ll), "r"((uint32_t)(nbytes_1)) : "memory");
                        }
                    } else {
                        int nrow_1 = (((u_1 + 1) * 16 + group) < (rows - 1) ? ((u_1 + 1) * 16 + group) : (rows - 1));
                        uint32_t nq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow_1);
                        uint32_t nh_1 = (uint32_t)((unsigned int)nrow_1) - nq_1 * (uint32_t)(heads.divisor);
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq_1 * dy_tok + (long long)nh_1 * dy_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq_1 * x_tok + (long long)nh_1 * x_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq_1 * z_tok + (long long)nh_1 * z_head + (long long)col))));
                    }
                }
                if (tid == 0) {
                    if (i_1 == claim_at) {
                        uint32_t _atomic_fetch_inc_242;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_242) : "l"(&dw_done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        ticket = (int)_atomic_fetch_inc_242;
                    }
                }
                if (u_1 == padded_block) {
                    float total_2 = 0.0f;
                    total_2 = __fmaf_rn(xs_1[0], xs_1[0], total_2);
                    total_2 = __fmaf_rn(xs_1[1], xs_1[1], total_2);
                    total_2 = __fmaf_rn(xs_1[2], xs_1[2], total_2);
                    total_2 = __fmaf_rn(xs_1[3], xs_1[3], total_2);
                    total_2 = __fmaf_rn(xs_1[4], xs_1[4], total_2);
                    total_2 = __fmaf_rn(xs_1[5], xs_1[5], total_2);
                    total_2 = __fmaf_rn(xs_1[6], xs_1[6], total_2);
                    total_2 = __fmaf_rn(xs_1[7], xs_1[7], total_2);
                    float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, total_2, 1);
                    total_2 = total_2 + _shfl_xor_sync_243;
                    float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, total_2, 2);
                    total_2 = total_2 + _shfl_xor_sync_244;
                    float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, total_2, 4);
                    total_2 = total_2 + _shfl_xor_sync_245;
                    float _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, total_2, 8);
                    total_2 = total_2 + _shfl_xor_sync_246;
                    float _sqrt_247;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_247) : "f"(total_2 * 0.0078125f + eps));
                    float rstd_2 = __frcp_rn(_sqrt_247);
                    unsigned int keep_2 = 4294967295u;
                    unsigned int sign_1 = 2147483648u;
                    keep_2 = ((row_1 < rows) ? keep_2 : sign_1);
                    float dot_2 = 0.0f;
                    int fast_2 = 0;
                    float _neg_248;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_248) : "f"(zs_1[0]));
                    float _neg_249;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_249) : "f"(zs_1[1]));
                    float _neg_250;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_250) : "f"(zs_1[2]));
                    float mz_2;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_2) : "f"(_neg_248), "f"(_neg_249), "f"(_neg_250));
                    float _abs_252;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_252) : "f"(zs_1[0]));
                    float _neg_251;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_251) : "f"(_abs_252));
                    float _abs_254;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_254) : "f"(zs_1[1]));
                    float _neg_253;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_253) : "f"(_abs_254));
                    float _abs_256;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_256) : "f"(zs_1[2]));
                    float _neg_255;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_255) : "f"(_abs_256));
                    float ma_2;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_2) : "f"(_neg_251), "f"(_neg_253), "f"(_neg_255));
                    float _neg_258;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_258) : "f"(zs_1[3]));
                    float _neg_259;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_259) : "f"(zs_1[4]));
                    float _max3_257;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_257) : "f"(mz_2), "f"(_neg_258), "f"(_neg_259));
                    mz_2 = _max3_257;
                    float _abs_262;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_262) : "f"(zs_1[3]));
                    float _neg_261;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_261) : "f"(_abs_262));
                    float _abs_264;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_264) : "f"(zs_1[4]));
                    float _neg_263;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_263) : "f"(_abs_264));
                    float _max3_260;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_260) : "f"(ma_2), "f"(_neg_261), "f"(_neg_263));
                    ma_2 = _max3_260;
                    float _neg_266;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_266) : "f"(zs_1[5]));
                    float _neg_267;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_267) : "f"(zs_1[6]));
                    float _max3_265;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_265) : "f"(mz_2), "f"(_neg_266), "f"(_neg_267));
                    mz_2 = _max3_265;
                    float _abs_270;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_270) : "f"(zs_1[5]));
                    float _neg_269;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_269) : "f"(_abs_270));
                    float _abs_272;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_272) : "f"(zs_1[6]));
                    float _neg_271;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_271) : "f"(_abs_272));
                    float _max3_268;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_268) : "f"(ma_2), "f"(_neg_269), "f"(_neg_271));
                    ma_2 = _max3_268;
                    float _neg_274;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_274) : "f"(zs_1[7]));
                    float _max_273;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_273) : "f"(mz_2), "f"(_neg_274));
                    mz_2 = _max_273;
                    float _abs_277;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_277) : "f"(zs_1[7]));
                    float _neg_276;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_276) : "f"(_abs_277));
                    float _max_275;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_275) : "f"(ma_2), "f"(_neg_276));
                    ma_2 = _max_275;
                    float _abs_278;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_278) : "f"(zs_1[0]));
                    float _abs_279;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_279) : "f"(zs_1[1]));
                    float _abs_280;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_280) : "f"(zs_1[2]));
                    float mp_2;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_2) : "f"(_abs_278), "f"(_abs_279), "f"(_abs_280));
                    float _abs_282;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_282) : "f"(zs_1[3]));
                    float _abs_283;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_283) : "f"(zs_1[4]));
                    float _max3_281;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_281) : "f"(mp_2), "f"(_abs_282), "f"(_abs_283));
                    mp_2 = _max3_281;
                    float _abs_285;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_285) : "f"(zs_1[5]));
                    float _abs_286;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_286) : "f"(zs_1[6]));
                    float _max3_284;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_284) : "f"(mp_2), "f"(_abs_285), "f"(_abs_286));
                    mp_2 = _max3_284;
                    float _abs_288;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_288) : "f"(zs_1[7]));
                    float _max_287;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_287) : "f"(mp_2), "f"(_abs_288));
                    mp_2 = _max_287;
                    if (mz_2 <= 41.0f) {
                        if (ma_2 <= -1.1754944e-38f) {
                            if (mp_2 <= 3.4028235e+38f) {
                                fast_2 = 1;
                            }
                        }
                    }
                    if (fast_2 != 0) {
                        float2 rstd2_2 = make_float2(rstd_2, rstd_2);
                        float2 z2_8 = make_float2(zs_1[0], zs_1[1]);
                        float ta_8;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_8) : "f"(zs_1[0]), "f"(-0.00572498f), "f"(0.5f));
                        float tb_8;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_8) : "f"(zs_1[1]), "f"(-0.00572498f), "f"(0.5f));
                        float _fma_289;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(ta_8), "f"(252.0f), "f"(1.2582913e+07f));
                        float _fma_290;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(tb_8), "f"(252.0f), "f"(1.2582913e+07f));
                        float2 t2_8 = make_float2(_fma_289, _fma_290);
                        float2 r2_8 = fma_f32x2_rn_noftz(z2_8, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_8));
                        r2_8 = fma_f32x2_rn_noftz(z2_8, make_float2(-1.925963e-08f, -1.925963e-08f), r2_8);
                        float2 nscale_8 = make_float2(__uint_as_float((__as_u32(t2_8.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_8.y) << 23) + 2147483648u));
                        float _exp2_291 = approx_exp2(r2_8.x);
                        float _exp2_292 = approx_exp2(r2_8.y);
                        float2 nden_8 = fma_f32x2_rn_noftz(nscale_8, make_float2(_exp2_291, _exp2_292), make_float2(-1.0f, -1.0f));
                        float _neg_294;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_294) : "f"(nden_8.x));
                        float _rcp_293 = approx_rcp(_neg_294);
                        float _neg_296;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_296) : "f"(nden_8.y));
                        float _rcp_295 = approx_rcp(_neg_296);
                        float2 r0_8 = make_float2(_rcp_293, _rcp_295);
                        float2 r1_8 = fma_f32x2_rn_noftz(r0_8, fma_f32x2_rn_noftz(nden_8, r0_8, make_float2(1.0f, 1.0f)), r0_8);
                        float2 sig2_8 = fma_f32x2_rn_noftz(r1_8, fma_f32x2_rn_noftz(nden_8, r1_8, make_float2(1.0f, 1.0f)), r1_8);
                        float2 q0_8 = fma_f32x2_rn_noftz(z2_8, r1_8, make_float2(0.0f, 0.0f));
                        float2 gate2_8 = fma_f32x2_rn_noftz(r1_8, fma_f32x2_rn_noftz(nden_8, q0_8, z2_8), q0_8);
                        float2 g2_8 = make_float2(gs_1[0], gs_1[1]);
                        float2 w2_8 = make_float2(wclaim[0], wclaim[1]);
                        float2 _f32x2_mul_a_297 = make_float2(xs_1[0], xs_1[1]);
                        float2 n2_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_297), "l"(*(const unsigned long long*)&rstd2_2));
                        float2 _f32x2_mul_a_298 = mul_f32x2_noftz(mul_f32x2_noftz(g2_8, mul_f32x2_noftz(n2_8, w2_8)), sig2_8);
                        float2 _f32x2_mul_b_299 = fma_f32x2_rn_noftz(z2_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_8), make_float2(1.0f, 1.0f));
                        float2 dz2_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_298), "l"(*(const unsigned long long*)&_f32x2_mul_b_299));
                        float2 a2_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_8) : "l"(*(const unsigned long long*)&g2_8), "l"(*(const unsigned long long*)&gate2_8));
                        float2 dn2_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_8) : "l"(*(const unsigned long long*)&a2_8), "l"(*(const unsigned long long*)&w2_8));
                        float2 acc2_8 = make_float2(acc[0], acc[1]);
                        acc2_8 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_8.x) & keep_2), __uint_as_float(__as_u32(a2_8.y) & keep_2)), make_float2(__uint_as_float(__as_u32(n2_8.x) & keep_2), __uint_as_float(__as_u32(n2_8.y) & keep_2)), acc2_8);
                        acc[0] = acc2_8.x;
                        acc[1] = acc2_8.y;
                        zs_1[0] = dz2_8.x;
                        zs_1[1] = dz2_8.y;
                        gs_1[0] = dn2_8.x;
                        gs_1[1] = dn2_8.y;
                        float2 z2_9 = make_float2(zs_1[2], zs_1[3]);
                        float ta_9;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_9) : "f"(zs_1[2]), "f"(-0.00572498f), "f"(0.5f));
                        float tb_9;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_9) : "f"(zs_1[3]), "f"(-0.00572498f), "f"(0.5f));
                        float _fma_300;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(ta_9), "f"(252.0f), "f"(1.2582913e+07f));
                        float _fma_301;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(tb_9), "f"(252.0f), "f"(1.2582913e+07f));
                        float2 t2_9 = make_float2(_fma_300, _fma_301);
                        float2 r2_9 = fma_f32x2_rn_noftz(z2_9, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_9));
                        r2_9 = fma_f32x2_rn_noftz(z2_9, make_float2(-1.925963e-08f, -1.925963e-08f), r2_9);
                        float2 nscale_9 = make_float2(__uint_as_float((__as_u32(t2_9.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_9.y) << 23) + 2147483648u));
                        float _exp2_302 = approx_exp2(r2_9.x);
                        float _exp2_303 = approx_exp2(r2_9.y);
                        float2 nden_9 = fma_f32x2_rn_noftz(nscale_9, make_float2(_exp2_302, _exp2_303), make_float2(-1.0f, -1.0f));
                        float _neg_305;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_305) : "f"(nden_9.x));
                        float _rcp_304 = approx_rcp(_neg_305);
                        float _neg_307;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_307) : "f"(nden_9.y));
                        float _rcp_306 = approx_rcp(_neg_307);
                        float2 r0_9 = make_float2(_rcp_304, _rcp_306);
                        float2 r1_9 = fma_f32x2_rn_noftz(r0_9, fma_f32x2_rn_noftz(nden_9, r0_9, make_float2(1.0f, 1.0f)), r0_9);
                        float2 sig2_9 = fma_f32x2_rn_noftz(r1_9, fma_f32x2_rn_noftz(nden_9, r1_9, make_float2(1.0f, 1.0f)), r1_9);
                        float2 q0_9 = fma_f32x2_rn_noftz(z2_9, r1_9, make_float2(0.0f, 0.0f));
                        float2 gate2_9 = fma_f32x2_rn_noftz(r1_9, fma_f32x2_rn_noftz(nden_9, q0_9, z2_9), q0_9);
                        float2 g2_9 = make_float2(gs_1[2], gs_1[3]);
                        float2 w2_9 = make_float2(wclaim[2], wclaim[3]);
                        float2 _f32x2_mul_a_308 = make_float2(xs_1[2], xs_1[3]);
                        float2 n2_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_308), "l"(*(const unsigned long long*)&rstd2_2));
                        float2 _f32x2_mul_a_309 = mul_f32x2_noftz(mul_f32x2_noftz(g2_9, mul_f32x2_noftz(n2_9, w2_9)), sig2_9);
                        float2 _f32x2_mul_b_310 = fma_f32x2_rn_noftz(z2_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_9), make_float2(1.0f, 1.0f));
                        float2 dz2_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_309), "l"(*(const unsigned long long*)&_f32x2_mul_b_310));
                        float2 a2_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_9) : "l"(*(const unsigned long long*)&g2_9), "l"(*(const unsigned long long*)&gate2_9));
                        float2 dn2_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_9) : "l"(*(const unsigned long long*)&a2_9), "l"(*(const unsigned long long*)&w2_9));
                        float2 acc2_9 = make_float2(acc[2], acc[3]);
                        acc2_9 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_9.x) & keep_2), __uint_as_float(__as_u32(a2_9.y) & keep_2)), make_float2(__uint_as_float(__as_u32(n2_9.x) & keep_2), __uint_as_float(__as_u32(n2_9.y) & keep_2)), acc2_9);
                        acc[2] = acc2_9.x;
                        acc[3] = acc2_9.y;
                        zs_1[2] = dz2_9.x;
                        zs_1[3] = dz2_9.y;
                        gs_1[2] = dn2_9.x;
                        gs_1[3] = dn2_9.y;
                        float2 z2_10 = make_float2(zs_1[4], zs_1[5]);
                        float ta_10;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_10) : "f"(zs_1[4]), "f"(-0.00572498f), "f"(0.5f));
                        float tb_10;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_10) : "f"(zs_1[5]), "f"(-0.00572498f), "f"(0.5f));
                        float _fma_311;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(ta_10), "f"(252.0f), "f"(1.2582913e+07f));
                        float _fma_312;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(tb_10), "f"(252.0f), "f"(1.2582913e+07f));
                        float2 t2_10 = make_float2(_fma_311, _fma_312);
                        float2 r2_10 = fma_f32x2_rn_noftz(z2_10, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_10));
                        r2_10 = fma_f32x2_rn_noftz(z2_10, make_float2(-1.925963e-08f, -1.925963e-08f), r2_10);
                        float2 nscale_10 = make_float2(__uint_as_float((__as_u32(t2_10.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_10.y) << 23) + 2147483648u));
                        float _exp2_313 = approx_exp2(r2_10.x);
                        float _exp2_314 = approx_exp2(r2_10.y);
                        float2 nden_10 = fma_f32x2_rn_noftz(nscale_10, make_float2(_exp2_313, _exp2_314), make_float2(-1.0f, -1.0f));
                        float _neg_316;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_316) : "f"(nden_10.x));
                        float _rcp_315 = approx_rcp(_neg_316);
                        float _neg_318;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_318) : "f"(nden_10.y));
                        float _rcp_317 = approx_rcp(_neg_318);
                        float2 r0_10 = make_float2(_rcp_315, _rcp_317);
                        float2 r1_10 = fma_f32x2_rn_noftz(r0_10, fma_f32x2_rn_noftz(nden_10, r0_10, make_float2(1.0f, 1.0f)), r0_10);
                        float2 sig2_10 = fma_f32x2_rn_noftz(r1_10, fma_f32x2_rn_noftz(nden_10, r1_10, make_float2(1.0f, 1.0f)), r1_10);
                        float2 q0_10 = fma_f32x2_rn_noftz(z2_10, r1_10, make_float2(0.0f, 0.0f));
                        float2 gate2_10 = fma_f32x2_rn_noftz(r1_10, fma_f32x2_rn_noftz(nden_10, q0_10, z2_10), q0_10);
                        float2 g2_10 = make_float2(gs_1[4], gs_1[5]);
                        float2 w2_10 = make_float2(wclaim[4], wclaim[5]);
                        float2 _f32x2_mul_a_319 = make_float2(xs_1[4], xs_1[5]);
                        float2 n2_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_319), "l"(*(const unsigned long long*)&rstd2_2));
                        float2 _f32x2_mul_a_320 = mul_f32x2_noftz(mul_f32x2_noftz(g2_10, mul_f32x2_noftz(n2_10, w2_10)), sig2_10);
                        float2 _f32x2_mul_b_321 = fma_f32x2_rn_noftz(z2_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_10), make_float2(1.0f, 1.0f));
                        float2 dz2_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_320), "l"(*(const unsigned long long*)&_f32x2_mul_b_321));
                        float2 a2_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_10) : "l"(*(const unsigned long long*)&g2_10), "l"(*(const unsigned long long*)&gate2_10));
                        float2 dn2_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_10) : "l"(*(const unsigned long long*)&a2_10), "l"(*(const unsigned long long*)&w2_10));
                        float2 acc2_10 = make_float2(acc[4], acc[5]);
                        acc2_10 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_10.x) & keep_2), __uint_as_float(__as_u32(a2_10.y) & keep_2)), make_float2(__uint_as_float(__as_u32(n2_10.x) & keep_2), __uint_as_float(__as_u32(n2_10.y) & keep_2)), acc2_10);
                        acc[4] = acc2_10.x;
                        acc[5] = acc2_10.y;
                        zs_1[4] = dz2_10.x;
                        zs_1[5] = dz2_10.y;
                        gs_1[4] = dn2_10.x;
                        gs_1[5] = dn2_10.y;
                        float2 z2_11 = make_float2(zs_1[6], zs_1[7]);
                        float ta_11;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_11) : "f"(zs_1[6]), "f"(-0.00572498f), "f"(0.5f));
                        float tb_11;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_11) : "f"(zs_1[7]), "f"(-0.00572498f), "f"(0.5f));
                        float _fma_322;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(ta_11), "f"(252.0f), "f"(1.2582913e+07f));
                        float _fma_323;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(tb_11), "f"(252.0f), "f"(1.2582913e+07f));
                        float2 t2_11 = make_float2(_fma_322, _fma_323);
                        float2 r2_11 = fma_f32x2_rn_noftz(z2_11, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_11));
                        r2_11 = fma_f32x2_rn_noftz(z2_11, make_float2(-1.925963e-08f, -1.925963e-08f), r2_11);
                        float2 nscale_11 = make_float2(__uint_as_float((__as_u32(t2_11.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_11.y) << 23) + 2147483648u));
                        float _exp2_324 = approx_exp2(r2_11.x);
                        float _exp2_325 = approx_exp2(r2_11.y);
                        float2 nden_11 = fma_f32x2_rn_noftz(nscale_11, make_float2(_exp2_324, _exp2_325), make_float2(-1.0f, -1.0f));
                        float _neg_327;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_327) : "f"(nden_11.x));
                        float _rcp_326 = approx_rcp(_neg_327);
                        float _neg_329;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_329) : "f"(nden_11.y));
                        float _rcp_328 = approx_rcp(_neg_329);
                        float2 r0_11 = make_float2(_rcp_326, _rcp_328);
                        float2 r1_11 = fma_f32x2_rn_noftz(r0_11, fma_f32x2_rn_noftz(nden_11, r0_11, make_float2(1.0f, 1.0f)), r0_11);
                        float2 sig2_11 = fma_f32x2_rn_noftz(r1_11, fma_f32x2_rn_noftz(nden_11, r1_11, make_float2(1.0f, 1.0f)), r1_11);
                        float2 q0_11 = fma_f32x2_rn_noftz(z2_11, r1_11, make_float2(0.0f, 0.0f));
                        float2 gate2_11 = fma_f32x2_rn_noftz(r1_11, fma_f32x2_rn_noftz(nden_11, q0_11, z2_11), q0_11);
                        float2 g2_11 = make_float2(gs_1[6], gs_1[7]);
                        float2 w2_11 = make_float2(wclaim[6], wclaim[7]);
                        float2 _f32x2_mul_a_330 = make_float2(xs_1[6], xs_1[7]);
                        float2 n2_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_330), "l"(*(const unsigned long long*)&rstd2_2));
                        float2 _f32x2_mul_a_331 = mul_f32x2_noftz(mul_f32x2_noftz(g2_11, mul_f32x2_noftz(n2_11, w2_11)), sig2_11);
                        float2 _f32x2_mul_b_332 = fma_f32x2_rn_noftz(z2_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_11), make_float2(1.0f, 1.0f));
                        float2 dz2_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_331), "l"(*(const unsigned long long*)&_f32x2_mul_b_332));
                        float2 a2_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_11) : "l"(*(const unsigned long long*)&g2_11), "l"(*(const unsigned long long*)&gate2_11));
                        float2 dn2_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_11) : "l"(*(const unsigned long long*)&a2_11), "l"(*(const unsigned long long*)&w2_11));
                        float2 acc2_11 = make_float2(acc[6], acc[7]);
                        acc2_11 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_11.x) & keep_2), __uint_as_float(__as_u32(a2_11.y) & keep_2)), make_float2(__uint_as_float(__as_u32(n2_11.x) & keep_2), __uint_as_float(__as_u32(n2_11.y) & keep_2)), acc2_11);
                        acc[6] = acc2_11.x;
                        acc[7] = acc2_11.y;
                        zs_1[6] = dz2_11.x;
                        zs_1[7] = dz2_11.y;
                        gs_1[6] = dn2_11.x;
                        gs_1[7] = dn2_11.y;
                        dot_2 = __fmaf_rn(gs_1[0], xs_1[0], dot_2);
                        dot_2 = __fmaf_rn(gs_1[1], xs_1[1], dot_2);
                        dot_2 = __fmaf_rn(gs_1[2], xs_1[2], dot_2);
                        dot_2 = __fmaf_rn(gs_1[3], xs_1[3], dot_2);
                        dot_2 = __fmaf_rn(gs_1[4], xs_1[4], dot_2);
                        dot_2 = __fmaf_rn(gs_1[5], xs_1[5], dot_2);
                        dot_2 = __fmaf_rn(gs_1[6], xs_1[6], dot_2);
                        dot_2 = __fmaf_rn(gs_1[7], xs_1[7], dot_2);
                    } else {
                        float _exp_333 = __nv_expf(zs_1[0] * -1.0f);
                        float den_16 = 1.0f + _exp_333;
                        float sig_16 = __fdiv_rn(1.0f, den_16);
                        float gate_16 = __fdiv_rn(zs_1[0], den_16);
                        float nj_16 = xs_1[0] * rstd_2;
                        float aj_16 = gs_1[0] * gate_16;
                        float dzj_16 = gs_1[0] * (nj_16 * wclaim[0]) * sig_16 * __fmaf_rn(zs_1[0], 1.0f - sig_16, 1.0f);
                        float dnj_16 = aj_16 * wclaim[0];
                        dot_2 = __fmaf_rn(dnj_16, xs_1[0], dot_2);
                        acc[0] = __fmaf_rn(__uint_as_float(__as_u32(aj_16) & keep_2), __uint_as_float(__as_u32(nj_16) & keep_2), acc[0]);
                        zs_1[0] = dzj_16;
                        gs_1[0] = dnj_16;
                        float _exp_334 = __nv_expf(zs_1[1] * -1.0f);
                        float den_17 = 1.0f + _exp_334;
                        float sig_17 = __fdiv_rn(1.0f, den_17);
                        float gate_17 = __fdiv_rn(zs_1[1], den_17);
                        float nj_17 = xs_1[1] * rstd_2;
                        float aj_17 = gs_1[1] * gate_17;
                        float dzj_17 = gs_1[1] * (nj_17 * wclaim[1]) * sig_17 * __fmaf_rn(zs_1[1], 1.0f - sig_17, 1.0f);
                        float dnj_17 = aj_17 * wclaim[1];
                        dot_2 = __fmaf_rn(dnj_17, xs_1[1], dot_2);
                        acc[1] = __fmaf_rn(__uint_as_float(__as_u32(aj_17) & keep_2), __uint_as_float(__as_u32(nj_17) & keep_2), acc[1]);
                        zs_1[1] = dzj_17;
                        gs_1[1] = dnj_17;
                        float _exp_335 = __nv_expf(zs_1[2] * -1.0f);
                        float den_18 = 1.0f + _exp_335;
                        float sig_18 = __fdiv_rn(1.0f, den_18);
                        float gate_18 = __fdiv_rn(zs_1[2], den_18);
                        float nj_18 = xs_1[2] * rstd_2;
                        float aj_18 = gs_1[2] * gate_18;
                        float dzj_18 = gs_1[2] * (nj_18 * wclaim[2]) * sig_18 * __fmaf_rn(zs_1[2], 1.0f - sig_18, 1.0f);
                        float dnj_18 = aj_18 * wclaim[2];
                        dot_2 = __fmaf_rn(dnj_18, xs_1[2], dot_2);
                        acc[2] = __fmaf_rn(__uint_as_float(__as_u32(aj_18) & keep_2), __uint_as_float(__as_u32(nj_18) & keep_2), acc[2]);
                        zs_1[2] = dzj_18;
                        gs_1[2] = dnj_18;
                        float _exp_336 = __nv_expf(zs_1[3] * -1.0f);
                        float den_19 = 1.0f + _exp_336;
                        float sig_19 = __fdiv_rn(1.0f, den_19);
                        float gate_19 = __fdiv_rn(zs_1[3], den_19);
                        float nj_19 = xs_1[3] * rstd_2;
                        float aj_19 = gs_1[3] * gate_19;
                        float dzj_19 = gs_1[3] * (nj_19 * wclaim[3]) * sig_19 * __fmaf_rn(zs_1[3], 1.0f - sig_19, 1.0f);
                        float dnj_19 = aj_19 * wclaim[3];
                        dot_2 = __fmaf_rn(dnj_19, xs_1[3], dot_2);
                        acc[3] = __fmaf_rn(__uint_as_float(__as_u32(aj_19) & keep_2), __uint_as_float(__as_u32(nj_19) & keep_2), acc[3]);
                        zs_1[3] = dzj_19;
                        gs_1[3] = dnj_19;
                        float _exp_337 = __nv_expf(zs_1[4] * -1.0f);
                        float den_20 = 1.0f + _exp_337;
                        float sig_20 = __fdiv_rn(1.0f, den_20);
                        float gate_20 = __fdiv_rn(zs_1[4], den_20);
                        float nj_20 = xs_1[4] * rstd_2;
                        float aj_20 = gs_1[4] * gate_20;
                        float dzj_20 = gs_1[4] * (nj_20 * wclaim[4]) * sig_20 * __fmaf_rn(zs_1[4], 1.0f - sig_20, 1.0f);
                        float dnj_20 = aj_20 * wclaim[4];
                        dot_2 = __fmaf_rn(dnj_20, xs_1[4], dot_2);
                        acc[4] = __fmaf_rn(__uint_as_float(__as_u32(aj_20) & keep_2), __uint_as_float(__as_u32(nj_20) & keep_2), acc[4]);
                        zs_1[4] = dzj_20;
                        gs_1[4] = dnj_20;
                        float _exp_338 = __nv_expf(zs_1[5] * -1.0f);
                        float den_21 = 1.0f + _exp_338;
                        float sig_21 = __fdiv_rn(1.0f, den_21);
                        float gate_21 = __fdiv_rn(zs_1[5], den_21);
                        float nj_21 = xs_1[5] * rstd_2;
                        float aj_21 = gs_1[5] * gate_21;
                        float dzj_21 = gs_1[5] * (nj_21 * wclaim[5]) * sig_21 * __fmaf_rn(zs_1[5], 1.0f - sig_21, 1.0f);
                        float dnj_21 = aj_21 * wclaim[5];
                        dot_2 = __fmaf_rn(dnj_21, xs_1[5], dot_2);
                        acc[5] = __fmaf_rn(__uint_as_float(__as_u32(aj_21) & keep_2), __uint_as_float(__as_u32(nj_21) & keep_2), acc[5]);
                        zs_1[5] = dzj_21;
                        gs_1[5] = dnj_21;
                        float _exp_339 = __nv_expf(zs_1[6] * -1.0f);
                        float den_22 = 1.0f + _exp_339;
                        float sig_22 = __fdiv_rn(1.0f, den_22);
                        float gate_22 = __fdiv_rn(zs_1[6], den_22);
                        float nj_22 = xs_1[6] * rstd_2;
                        float aj_22 = gs_1[6] * gate_22;
                        float dzj_22 = gs_1[6] * (nj_22 * wclaim[6]) * sig_22 * __fmaf_rn(zs_1[6], 1.0f - sig_22, 1.0f);
                        float dnj_22 = aj_22 * wclaim[6];
                        dot_2 = __fmaf_rn(dnj_22, xs_1[6], dot_2);
                        acc[6] = __fmaf_rn(__uint_as_float(__as_u32(aj_22) & keep_2), __uint_as_float(__as_u32(nj_22) & keep_2), acc[6]);
                        zs_1[6] = dzj_22;
                        gs_1[6] = dnj_22;
                        float _exp_340 = __nv_expf(zs_1[7] * -1.0f);
                        float den_23 = 1.0f + _exp_340;
                        float sig_23 = __fdiv_rn(1.0f, den_23);
                        float gate_23 = __fdiv_rn(zs_1[7], den_23);
                        float nj_23 = xs_1[7] * rstd_2;
                        float aj_23 = gs_1[7] * gate_23;
                        float dzj_23 = gs_1[7] * (nj_23 * wclaim[7]) * sig_23 * __fmaf_rn(zs_1[7], 1.0f - sig_23, 1.0f);
                        float dnj_23 = aj_23 * wclaim[7];
                        dot_2 = __fmaf_rn(dnj_23, xs_1[7], dot_2);
                        acc[7] = __fmaf_rn(__uint_as_float(__as_u32(aj_23) & keep_2), __uint_as_float(__as_u32(nj_23) & keep_2), acc[7]);
                        zs_1[7] = dzj_23;
                        gs_1[7] = dnj_23;
                    }
                    float _shfl_xor_sync_341 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                    dot_2 = dot_2 + _shfl_xor_sync_341;
                    float _shfl_xor_sync_342 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                    dot_2 = dot_2 + _shfl_xor_sync_342;
                    float _shfl_xor_sync_343 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                    dot_2 = dot_2 + _shfl_xor_sync_343;
                    float _shfl_xor_sync_344 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 8);
                    dot_2 = dot_2 + _shfl_xor_sync_344;
                    float coef_2 = rstd_2 * rstd_2 * rstd_2 * (dot_2 * -0.0078125f);
                    float2 coef2_2 = make_float2(coef_2, coef_2);
                    float2 rstd2b_2 = make_float2(rstd_2, rstd_2);
                    float2 dx2_8 = fma_f32x2_rn_noftz(make_float2(xs_1[0], xs_1[1]), coef2_2, mul_f32x2_noftz(make_float2(gs_1[0], gs_1[1]), rstd2b_2));
                    xs_1[0] = dx2_8.x;
                    xs_1[1] = dx2_8.y;
                    float2 dx2_9 = fma_f32x2_rn_noftz(make_float2(xs_1[2], xs_1[3]), coef2_2, mul_f32x2_noftz(make_float2(gs_1[2], gs_1[3]), rstd2b_2));
                    xs_1[2] = dx2_9.x;
                    xs_1[3] = dx2_9.y;
                    float2 dx2_10 = fma_f32x2_rn_noftz(make_float2(xs_1[4], xs_1[5]), coef2_2, mul_f32x2_noftz(make_float2(gs_1[4], gs_1[5]), rstd2b_2));
                    xs_1[4] = dx2_10.x;
                    xs_1[5] = dx2_10.y;
                    float2 dx2_11 = fma_f32x2_rn_noftz(make_float2(xs_1[6], xs_1[7]), coef2_2, mul_f32x2_noftz(make_float2(gs_1[6], gs_1[7]), rstd2b_2));
                    xs_1[6] = dx2_11.x;
                    xs_1[7] = dx2_11.y;
                    unsigned int packed_dx_2[4];
                    unsigned int packed_dz_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 0], xs_1[_lp*2+1 + 0]));
                        packed_dx_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 0], zs_1[_lp*2+1 + 0]));
                        packed_dz_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 2], xs_1[_lp*2+1 + 2]));
                        packed_dx_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 2], zs_1[_lp*2+1 + 2]));
                        packed_dz_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 4], xs_1[_lp*2+1 + 4]));
                        packed_dx_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 4], zs_1[_lp*2+1 + 4]));
                        packed_dz_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 6], xs_1[_lp*2+1 + 6]));
                        packed_dx_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 6], zs_1[_lp*2+1 + 6]));
                        packed_dz_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    if (row_1 < rows) {
                        long long orow_2 = (long long)row_1 * 128ll + (long long)col;
                        reinterpret_cast<int4*>(dx + orow_2)[0] = reinterpret_cast<int4*>(packed_dx_2)[0];
                        reinterpret_cast<int4*>(dz + orow_2)[0] = reinterpret_cast<int4*>(packed_dz_2)[0];
                    }
                } else {
                    float total_3 = 0.0f;
                    total_3 = __fmaf_rn(xs_1[0], xs_1[0], total_3);
                    total_3 = __fmaf_rn(xs_1[1], xs_1[1], total_3);
                    total_3 = __fmaf_rn(xs_1[2], xs_1[2], total_3);
                    total_3 = __fmaf_rn(xs_1[3], xs_1[3], total_3);
                    total_3 = __fmaf_rn(xs_1[4], xs_1[4], total_3);
                    total_3 = __fmaf_rn(xs_1[5], xs_1[5], total_3);
                    total_3 = __fmaf_rn(xs_1[6], xs_1[6], total_3);
                    total_3 = __fmaf_rn(xs_1[7], xs_1[7], total_3);
                    float _shfl_xor_sync_345 = __shfl_xor_sync(0xFFFFFFFF, total_3, 1);
                    total_3 = total_3 + _shfl_xor_sync_345;
                    float _shfl_xor_sync_346 = __shfl_xor_sync(0xFFFFFFFF, total_3, 2);
                    total_3 = total_3 + _shfl_xor_sync_346;
                    float _shfl_xor_sync_347 = __shfl_xor_sync(0xFFFFFFFF, total_3, 4);
                    total_3 = total_3 + _shfl_xor_sync_347;
                    float _shfl_xor_sync_348 = __shfl_xor_sync(0xFFFFFFFF, total_3, 8);
                    total_3 = total_3 + _shfl_xor_sync_348;
                    float _sqrt_349;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_349) : "f"(total_3 * 0.0078125f + eps));
                    float rstd_3 = __frcp_rn(_sqrt_349);
                    unsigned int keep_3 = 4294967295u;
                    float dot_3 = 0.0f;
                    int fast_3 = 0;
                    float _neg_350;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_350) : "f"(zs_1[0]));
                    float _neg_351;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_351) : "f"(zs_1[1]));
                    float _neg_352;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_352) : "f"(zs_1[2]));
                    float mz_3;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_3) : "f"(_neg_350), "f"(_neg_351), "f"(_neg_352));
                    float _abs_354;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_354) : "f"(zs_1[0]));
                    float _neg_353;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_353) : "f"(_abs_354));
                    float _abs_356;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_356) : "f"(zs_1[1]));
                    float _neg_355;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_355) : "f"(_abs_356));
                    float _abs_358;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_358) : "f"(zs_1[2]));
                    float _neg_357;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_357) : "f"(_abs_358));
                    float ma_3;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_3) : "f"(_neg_353), "f"(_neg_355), "f"(_neg_357));
                    float _neg_360;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_360) : "f"(zs_1[3]));
                    float _neg_361;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_361) : "f"(zs_1[4]));
                    float _max3_359;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_359) : "f"(mz_3), "f"(_neg_360), "f"(_neg_361));
                    mz_3 = _max3_359;
                    float _abs_364;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_364) : "f"(zs_1[3]));
                    float _neg_363;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_363) : "f"(_abs_364));
                    float _abs_366;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_366) : "f"(zs_1[4]));
                    float _neg_365;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_365) : "f"(_abs_366));
                    float _max3_362;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_362) : "f"(ma_3), "f"(_neg_363), "f"(_neg_365));
                    ma_3 = _max3_362;
                    float _neg_368;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_368) : "f"(zs_1[5]));
                    float _neg_369;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_369) : "f"(zs_1[6]));
                    float _max3_367;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_367) : "f"(mz_3), "f"(_neg_368), "f"(_neg_369));
                    mz_3 = _max3_367;
                    float _abs_372;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_372) : "f"(zs_1[5]));
                    float _neg_371;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_371) : "f"(_abs_372));
                    float _abs_374;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_374) : "f"(zs_1[6]));
                    float _neg_373;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_373) : "f"(_abs_374));
                    float _max3_370;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_370) : "f"(ma_3), "f"(_neg_371), "f"(_neg_373));
                    ma_3 = _max3_370;
                    float _neg_376;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_376) : "f"(zs_1[7]));
                    float _max_375;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_375) : "f"(mz_3), "f"(_neg_376));
                    mz_3 = _max_375;
                    float _abs_379;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_379) : "f"(zs_1[7]));
                    float _neg_378;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_378) : "f"(_abs_379));
                    float _max_377;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_377) : "f"(ma_3), "f"(_neg_378));
                    ma_3 = _max_377;
                    float _abs_380;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_380) : "f"(zs_1[0]));
                    float _abs_381;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_381) : "f"(zs_1[1]));
                    float _abs_382;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_382) : "f"(zs_1[2]));
                    float mp_3;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_3) : "f"(_abs_380), "f"(_abs_381), "f"(_abs_382));
                    float _abs_384;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_384) : "f"(zs_1[3]));
                    float _abs_385;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_385) : "f"(zs_1[4]));
                    float _max3_383;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_383) : "f"(mp_3), "f"(_abs_384), "f"(_abs_385));
                    mp_3 = _max3_383;
                    float _abs_387;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_387) : "f"(zs_1[5]));
                    float _abs_388;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_388) : "f"(zs_1[6]));
                    float _max3_386;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_386) : "f"(mp_3), "f"(_abs_387), "f"(_abs_388));
                    mp_3 = _max3_386;
                    float _abs_390;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_390) : "f"(zs_1[7]));
                    float _max_389;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_389) : "f"(mp_3), "f"(_abs_390));
                    mp_3 = _max_389;
                    if (mz_3 <= 41.0f) {
                        if (ma_3 <= -1.1754944e-38f) {
                            if (mp_3 <= 3.4028235e+38f) {
                                fast_3 = 1;
                            }
                        }
                    }
                    if (fast_3 != 0) {
                        float2 rstd2_3 = make_float2(rstd_3, rstd_3);
                        float2 z2_12 = make_float2(zs_1[0], zs_1[1]);
                        float ta_12;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_12) : "f"(zs_1[0]), "f"(-0.00572498f), "f"(0.5f));
                        float tb_12;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_12) : "f"(zs_1[1]), "f"(-0.00572498f), "f"(0.5f));
                        float _fma_391;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_391) : "f"(ta_12), "f"(252.0f), "f"(1.2582913e+07f));
                        float _fma_392;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(tb_12), "f"(252.0f), "f"(1.2582913e+07f));
                        float2 t2_12 = make_float2(_fma_391, _fma_392);
                        float2 r2_12 = fma_f32x2_rn_noftz(z2_12, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_12));
                        r2_12 = fma_f32x2_rn_noftz(z2_12, make_float2(-1.925963e-08f, -1.925963e-08f), r2_12);
                        float2 nscale_12 = make_float2(__uint_as_float((__as_u32(t2_12.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_12.y) << 23) + 2147483648u));
                        float _exp2_393 = approx_exp2(r2_12.x);
                        float _exp2_394 = approx_exp2(r2_12.y);
                        float2 nden_12 = fma_f32x2_rn_noftz(nscale_12, make_float2(_exp2_393, _exp2_394), make_float2(-1.0f, -1.0f));
                        float _neg_396;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_396) : "f"(nden_12.x));
                        float _rcp_395 = approx_rcp(_neg_396);
                        float _neg_398;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_398) : "f"(nden_12.y));
                        float _rcp_397 = approx_rcp(_neg_398);
                        float2 r0_12 = make_float2(_rcp_395, _rcp_397);
                        float2 r1_12 = fma_f32x2_rn_noftz(r0_12, fma_f32x2_rn_noftz(nden_12, r0_12, make_float2(1.0f, 1.0f)), r0_12);
                        float2 sig2_12 = fma_f32x2_rn_noftz(r1_12, fma_f32x2_rn_noftz(nden_12, r1_12, make_float2(1.0f, 1.0f)), r1_12);
                        float2 q0_12 = fma_f32x2_rn_noftz(z2_12, r1_12, make_float2(0.0f, 0.0f));
                        float2 gate2_12 = fma_f32x2_rn_noftz(r1_12, fma_f32x2_rn_noftz(nden_12, q0_12, z2_12), q0_12);
                        float2 g2_12 = make_float2(gs_1[0], gs_1[1]);
                        float2 w2_12 = make_float2(wclaim[0], wclaim[1]);
                        float2 _f32x2_mul_a_399 = make_float2(xs_1[0], xs_1[1]);
                        float2 n2_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_399), "l"(*(const unsigned long long*)&rstd2_3));
                        float2 _f32x2_mul_a_400 = mul_f32x2_noftz(mul_f32x2_noftz(g2_12, mul_f32x2_noftz(n2_12, w2_12)), sig2_12);
                        float2 _f32x2_mul_b_401 = fma_f32x2_rn_noftz(z2_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_12), make_float2(1.0f, 1.0f));
                        float2 dz2_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_400), "l"(*(const unsigned long long*)&_f32x2_mul_b_401));
                        float2 a2_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_12) : "l"(*(const unsigned long long*)&g2_12), "l"(*(const unsigned long long*)&gate2_12));
                        float2 dn2_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_12) : "l"(*(const unsigned long long*)&a2_12), "l"(*(const unsigned long long*)&w2_12));
                        float2 acc2_12 = make_float2(acc[0], acc[1]);
                        acc2_12 = fma_f32x2_rn_noftz(a2_12, n2_12, acc2_12);
                        acc[0] = acc2_12.x;
                        acc[1] = acc2_12.y;
                        zs_1[0] = dz2_12.x;
                        zs_1[1] = dz2_12.y;
                        gs_1[0] = dn2_12.x;
                        gs_1[1] = dn2_12.y;
                        float2 z2_13 = make_float2(zs_1[2], zs_1[3]);
                        float ta_13;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_13) : "f"(zs_1[2]), "f"(-0.00572498f), "f"(0.5f));
                        float tb_13;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_13) : "f"(zs_1[3]), "f"(-0.00572498f), "f"(0.5f));
                        float _fma_402;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_402) : "f"(ta_13), "f"(252.0f), "f"(1.2582913e+07f));
                        float _fma_403;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(tb_13), "f"(252.0f), "f"(1.2582913e+07f));
                        float2 t2_13 = make_float2(_fma_402, _fma_403);
                        float2 r2_13 = fma_f32x2_rn_noftz(z2_13, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_13));
                        r2_13 = fma_f32x2_rn_noftz(z2_13, make_float2(-1.925963e-08f, -1.925963e-08f), r2_13);
                        float2 nscale_13 = make_float2(__uint_as_float((__as_u32(t2_13.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_13.y) << 23) + 2147483648u));
                        float _exp2_404 = approx_exp2(r2_13.x);
                        float _exp2_405 = approx_exp2(r2_13.y);
                        float2 nden_13 = fma_f32x2_rn_noftz(nscale_13, make_float2(_exp2_404, _exp2_405), make_float2(-1.0f, -1.0f));
                        float _neg_407;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_407) : "f"(nden_13.x));
                        float _rcp_406 = approx_rcp(_neg_407);
                        float _neg_409;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_409) : "f"(nden_13.y));
                        float _rcp_408 = approx_rcp(_neg_409);
                        float2 r0_13 = make_float2(_rcp_406, _rcp_408);
                        float2 r1_13 = fma_f32x2_rn_noftz(r0_13, fma_f32x2_rn_noftz(nden_13, r0_13, make_float2(1.0f, 1.0f)), r0_13);
                        float2 sig2_13 = fma_f32x2_rn_noftz(r1_13, fma_f32x2_rn_noftz(nden_13, r1_13, make_float2(1.0f, 1.0f)), r1_13);
                        float2 q0_13 = fma_f32x2_rn_noftz(z2_13, r1_13, make_float2(0.0f, 0.0f));
                        float2 gate2_13 = fma_f32x2_rn_noftz(r1_13, fma_f32x2_rn_noftz(nden_13, q0_13, z2_13), q0_13);
                        float2 g2_13 = make_float2(gs_1[2], gs_1[3]);
                        float2 w2_13 = make_float2(wclaim[2], wclaim[3]);
                        float2 _f32x2_mul_a_410 = make_float2(xs_1[2], xs_1[3]);
                        float2 n2_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_410), "l"(*(const unsigned long long*)&rstd2_3));
                        float2 _f32x2_mul_a_411 = mul_f32x2_noftz(mul_f32x2_noftz(g2_13, mul_f32x2_noftz(n2_13, w2_13)), sig2_13);
                        float2 _f32x2_mul_b_412 = fma_f32x2_rn_noftz(z2_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_13), make_float2(1.0f, 1.0f));
                        float2 dz2_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_411), "l"(*(const unsigned long long*)&_f32x2_mul_b_412));
                        float2 a2_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_13) : "l"(*(const unsigned long long*)&g2_13), "l"(*(const unsigned long long*)&gate2_13));
                        float2 dn2_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_13) : "l"(*(const unsigned long long*)&a2_13), "l"(*(const unsigned long long*)&w2_13));
                        float2 acc2_13 = make_float2(acc[2], acc[3]);
                        acc2_13 = fma_f32x2_rn_noftz(a2_13, n2_13, acc2_13);
                        acc[2] = acc2_13.x;
                        acc[3] = acc2_13.y;
                        zs_1[2] = dz2_13.x;
                        zs_1[3] = dz2_13.y;
                        gs_1[2] = dn2_13.x;
                        gs_1[3] = dn2_13.y;
                        float2 z2_14 = make_float2(zs_1[4], zs_1[5]);
                        float ta_14;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_14) : "f"(zs_1[4]), "f"(-0.00572498f), "f"(0.5f));
                        float tb_14;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_14) : "f"(zs_1[5]), "f"(-0.00572498f), "f"(0.5f));
                        float _fma_413;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_413) : "f"(ta_14), "f"(252.0f), "f"(1.2582913e+07f));
                        float _fma_414;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_414) : "f"(tb_14), "f"(252.0f), "f"(1.2582913e+07f));
                        float2 t2_14 = make_float2(_fma_413, _fma_414);
                        float2 r2_14 = fma_f32x2_rn_noftz(z2_14, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_14));
                        r2_14 = fma_f32x2_rn_noftz(z2_14, make_float2(-1.925963e-08f, -1.925963e-08f), r2_14);
                        float2 nscale_14 = make_float2(__uint_as_float((__as_u32(t2_14.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_14.y) << 23) + 2147483648u));
                        float _exp2_415 = approx_exp2(r2_14.x);
                        float _exp2_416 = approx_exp2(r2_14.y);
                        float2 nden_14 = fma_f32x2_rn_noftz(nscale_14, make_float2(_exp2_415, _exp2_416), make_float2(-1.0f, -1.0f));
                        float _neg_418;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_418) : "f"(nden_14.x));
                        float _rcp_417 = approx_rcp(_neg_418);
                        float _neg_420;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_420) : "f"(nden_14.y));
                        float _rcp_419 = approx_rcp(_neg_420);
                        float2 r0_14 = make_float2(_rcp_417, _rcp_419);
                        float2 r1_14 = fma_f32x2_rn_noftz(r0_14, fma_f32x2_rn_noftz(nden_14, r0_14, make_float2(1.0f, 1.0f)), r0_14);
                        float2 sig2_14 = fma_f32x2_rn_noftz(r1_14, fma_f32x2_rn_noftz(nden_14, r1_14, make_float2(1.0f, 1.0f)), r1_14);
                        float2 q0_14 = fma_f32x2_rn_noftz(z2_14, r1_14, make_float2(0.0f, 0.0f));
                        float2 gate2_14 = fma_f32x2_rn_noftz(r1_14, fma_f32x2_rn_noftz(nden_14, q0_14, z2_14), q0_14);
                        float2 g2_14 = make_float2(gs_1[4], gs_1[5]);
                        float2 w2_14 = make_float2(wclaim[4], wclaim[5]);
                        float2 _f32x2_mul_a_421 = make_float2(xs_1[4], xs_1[5]);
                        float2 n2_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_421), "l"(*(const unsigned long long*)&rstd2_3));
                        float2 _f32x2_mul_a_422 = mul_f32x2_noftz(mul_f32x2_noftz(g2_14, mul_f32x2_noftz(n2_14, w2_14)), sig2_14);
                        float2 _f32x2_mul_b_423 = fma_f32x2_rn_noftz(z2_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_14), make_float2(1.0f, 1.0f));
                        float2 dz2_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_422), "l"(*(const unsigned long long*)&_f32x2_mul_b_423));
                        float2 a2_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_14) : "l"(*(const unsigned long long*)&g2_14), "l"(*(const unsigned long long*)&gate2_14));
                        float2 dn2_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_14) : "l"(*(const unsigned long long*)&a2_14), "l"(*(const unsigned long long*)&w2_14));
                        float2 acc2_14 = make_float2(acc[4], acc[5]);
                        acc2_14 = fma_f32x2_rn_noftz(a2_14, n2_14, acc2_14);
                        acc[4] = acc2_14.x;
                        acc[5] = acc2_14.y;
                        zs_1[4] = dz2_14.x;
                        zs_1[5] = dz2_14.y;
                        gs_1[4] = dn2_14.x;
                        gs_1[5] = dn2_14.y;
                        float2 z2_15 = make_float2(zs_1[6], zs_1[7]);
                        float ta_15;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_15) : "f"(zs_1[6]), "f"(-0.00572498f), "f"(0.5f));
                        float tb_15;
                        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_15) : "f"(zs_1[7]), "f"(-0.00572498f), "f"(0.5f));
                        float _fma_424;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_424) : "f"(ta_15), "f"(252.0f), "f"(1.2582913e+07f));
                        float _fma_425;
                        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_425) : "f"(tb_15), "f"(252.0f), "f"(1.2582913e+07f));
                        float2 t2_15 = make_float2(_fma_424, _fma_425);
                        float2 r2_15 = fma_f32x2_rn_noftz(z2_15, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_15));
                        r2_15 = fma_f32x2_rn_noftz(z2_15, make_float2(-1.925963e-08f, -1.925963e-08f), r2_15);
                        float2 nscale_15 = make_float2(__uint_as_float((__as_u32(t2_15.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_15.y) << 23) + 2147483648u));
                        float _exp2_426 = approx_exp2(r2_15.x);
                        float _exp2_427 = approx_exp2(r2_15.y);
                        float2 nden_15 = fma_f32x2_rn_noftz(nscale_15, make_float2(_exp2_426, _exp2_427), make_float2(-1.0f, -1.0f));
                        float _neg_429;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_429) : "f"(nden_15.x));
                        float _rcp_428 = approx_rcp(_neg_429);
                        float _neg_431;
                        asm("neg.f32 %0, %1;" : "=f"(_neg_431) : "f"(nden_15.y));
                        float _rcp_430 = approx_rcp(_neg_431);
                        float2 r0_15 = make_float2(_rcp_428, _rcp_430);
                        float2 r1_15 = fma_f32x2_rn_noftz(r0_15, fma_f32x2_rn_noftz(nden_15, r0_15, make_float2(1.0f, 1.0f)), r0_15);
                        float2 sig2_15 = fma_f32x2_rn_noftz(r1_15, fma_f32x2_rn_noftz(nden_15, r1_15, make_float2(1.0f, 1.0f)), r1_15);
                        float2 q0_15 = fma_f32x2_rn_noftz(z2_15, r1_15, make_float2(0.0f, 0.0f));
                        float2 gate2_15 = fma_f32x2_rn_noftz(r1_15, fma_f32x2_rn_noftz(nden_15, q0_15, z2_15), q0_15);
                        float2 g2_15 = make_float2(gs_1[6], gs_1[7]);
                        float2 w2_15 = make_float2(wclaim[6], wclaim[7]);
                        float2 _f32x2_mul_a_432 = make_float2(xs_1[6], xs_1[7]);
                        float2 n2_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_432), "l"(*(const unsigned long long*)&rstd2_3));
                        float2 _f32x2_mul_a_433 = mul_f32x2_noftz(mul_f32x2_noftz(g2_15, mul_f32x2_noftz(n2_15, w2_15)), sig2_15);
                        float2 _f32x2_mul_b_434 = fma_f32x2_rn_noftz(z2_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_15), make_float2(1.0f, 1.0f));
                        float2 dz2_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_433), "l"(*(const unsigned long long*)&_f32x2_mul_b_434));
                        float2 a2_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_15) : "l"(*(const unsigned long long*)&g2_15), "l"(*(const unsigned long long*)&gate2_15));
                        float2 dn2_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_15) : "l"(*(const unsigned long long*)&a2_15), "l"(*(const unsigned long long*)&w2_15));
                        float2 acc2_15 = make_float2(acc[6], acc[7]);
                        acc2_15 = fma_f32x2_rn_noftz(a2_15, n2_15, acc2_15);
                        acc[6] = acc2_15.x;
                        acc[7] = acc2_15.y;
                        zs_1[6] = dz2_15.x;
                        zs_1[7] = dz2_15.y;
                        gs_1[6] = dn2_15.x;
                        gs_1[7] = dn2_15.y;
                        dot_3 = __fmaf_rn(gs_1[0], xs_1[0], dot_3);
                        dot_3 = __fmaf_rn(gs_1[1], xs_1[1], dot_3);
                        dot_3 = __fmaf_rn(gs_1[2], xs_1[2], dot_3);
                        dot_3 = __fmaf_rn(gs_1[3], xs_1[3], dot_3);
                        dot_3 = __fmaf_rn(gs_1[4], xs_1[4], dot_3);
                        dot_3 = __fmaf_rn(gs_1[5], xs_1[5], dot_3);
                        dot_3 = __fmaf_rn(gs_1[6], xs_1[6], dot_3);
                        dot_3 = __fmaf_rn(gs_1[7], xs_1[7], dot_3);
                    } else {
                        float _exp_435 = __nv_expf(zs_1[0] * -1.0f);
                        float den_24 = 1.0f + _exp_435;
                        float sig_24 = __fdiv_rn(1.0f, den_24);
                        float gate_24 = __fdiv_rn(zs_1[0], den_24);
                        float nj_24 = xs_1[0] * rstd_3;
                        float aj_24 = gs_1[0] * gate_24;
                        float dzj_24 = gs_1[0] * (nj_24 * wclaim[0]) * sig_24 * __fmaf_rn(zs_1[0], 1.0f - sig_24, 1.0f);
                        float dnj_24 = aj_24 * wclaim[0];
                        dot_3 = __fmaf_rn(dnj_24, xs_1[0], dot_3);
                        acc[0] = __fmaf_rn(aj_24, nj_24, acc[0]);
                        zs_1[0] = dzj_24;
                        gs_1[0] = dnj_24;
                        float _exp_436 = __nv_expf(zs_1[1] * -1.0f);
                        float den_25 = 1.0f + _exp_436;
                        float sig_25 = __fdiv_rn(1.0f, den_25);
                        float gate_25 = __fdiv_rn(zs_1[1], den_25);
                        float nj_25 = xs_1[1] * rstd_3;
                        float aj_25 = gs_1[1] * gate_25;
                        float dzj_25 = gs_1[1] * (nj_25 * wclaim[1]) * sig_25 * __fmaf_rn(zs_1[1], 1.0f - sig_25, 1.0f);
                        float dnj_25 = aj_25 * wclaim[1];
                        dot_3 = __fmaf_rn(dnj_25, xs_1[1], dot_3);
                        acc[1] = __fmaf_rn(aj_25, nj_25, acc[1]);
                        zs_1[1] = dzj_25;
                        gs_1[1] = dnj_25;
                        float _exp_437 = __nv_expf(zs_1[2] * -1.0f);
                        float den_26 = 1.0f + _exp_437;
                        float sig_26 = __fdiv_rn(1.0f, den_26);
                        float gate_26 = __fdiv_rn(zs_1[2], den_26);
                        float nj_26 = xs_1[2] * rstd_3;
                        float aj_26 = gs_1[2] * gate_26;
                        float dzj_26 = gs_1[2] * (nj_26 * wclaim[2]) * sig_26 * __fmaf_rn(zs_1[2], 1.0f - sig_26, 1.0f);
                        float dnj_26 = aj_26 * wclaim[2];
                        dot_3 = __fmaf_rn(dnj_26, xs_1[2], dot_3);
                        acc[2] = __fmaf_rn(aj_26, nj_26, acc[2]);
                        zs_1[2] = dzj_26;
                        gs_1[2] = dnj_26;
                        float _exp_438 = __nv_expf(zs_1[3] * -1.0f);
                        float den_27 = 1.0f + _exp_438;
                        float sig_27 = __fdiv_rn(1.0f, den_27);
                        float gate_27 = __fdiv_rn(zs_1[3], den_27);
                        float nj_27 = xs_1[3] * rstd_3;
                        float aj_27 = gs_1[3] * gate_27;
                        float dzj_27 = gs_1[3] * (nj_27 * wclaim[3]) * sig_27 * __fmaf_rn(zs_1[3], 1.0f - sig_27, 1.0f);
                        float dnj_27 = aj_27 * wclaim[3];
                        dot_3 = __fmaf_rn(dnj_27, xs_1[3], dot_3);
                        acc[3] = __fmaf_rn(aj_27, nj_27, acc[3]);
                        zs_1[3] = dzj_27;
                        gs_1[3] = dnj_27;
                        float _exp_439 = __nv_expf(zs_1[4] * -1.0f);
                        float den_28 = 1.0f + _exp_439;
                        float sig_28 = __fdiv_rn(1.0f, den_28);
                        float gate_28 = __fdiv_rn(zs_1[4], den_28);
                        float nj_28 = xs_1[4] * rstd_3;
                        float aj_28 = gs_1[4] * gate_28;
                        float dzj_28 = gs_1[4] * (nj_28 * wclaim[4]) * sig_28 * __fmaf_rn(zs_1[4], 1.0f - sig_28, 1.0f);
                        float dnj_28 = aj_28 * wclaim[4];
                        dot_3 = __fmaf_rn(dnj_28, xs_1[4], dot_3);
                        acc[4] = __fmaf_rn(aj_28, nj_28, acc[4]);
                        zs_1[4] = dzj_28;
                        gs_1[4] = dnj_28;
                        float _exp_440 = __nv_expf(zs_1[5] * -1.0f);
                        float den_29 = 1.0f + _exp_440;
                        float sig_29 = __fdiv_rn(1.0f, den_29);
                        float gate_29 = __fdiv_rn(zs_1[5], den_29);
                        float nj_29 = xs_1[5] * rstd_3;
                        float aj_29 = gs_1[5] * gate_29;
                        float dzj_29 = gs_1[5] * (nj_29 * wclaim[5]) * sig_29 * __fmaf_rn(zs_1[5], 1.0f - sig_29, 1.0f);
                        float dnj_29 = aj_29 * wclaim[5];
                        dot_3 = __fmaf_rn(dnj_29, xs_1[5], dot_3);
                        acc[5] = __fmaf_rn(aj_29, nj_29, acc[5]);
                        zs_1[5] = dzj_29;
                        gs_1[5] = dnj_29;
                        float _exp_441 = __nv_expf(zs_1[6] * -1.0f);
                        float den_30 = 1.0f + _exp_441;
                        float sig_30 = __fdiv_rn(1.0f, den_30);
                        float gate_30 = __fdiv_rn(zs_1[6], den_30);
                        float nj_30 = xs_1[6] * rstd_3;
                        float aj_30 = gs_1[6] * gate_30;
                        float dzj_30 = gs_1[6] * (nj_30 * wclaim[6]) * sig_30 * __fmaf_rn(zs_1[6], 1.0f - sig_30, 1.0f);
                        float dnj_30 = aj_30 * wclaim[6];
                        dot_3 = __fmaf_rn(dnj_30, xs_1[6], dot_3);
                        acc[6] = __fmaf_rn(aj_30, nj_30, acc[6]);
                        zs_1[6] = dzj_30;
                        gs_1[6] = dnj_30;
                        float _exp_442 = __nv_expf(zs_1[7] * -1.0f);
                        float den_31 = 1.0f + _exp_442;
                        float sig_31 = __fdiv_rn(1.0f, den_31);
                        float gate_31 = __fdiv_rn(zs_1[7], den_31);
                        float nj_31 = xs_1[7] * rstd_3;
                        float aj_31 = gs_1[7] * gate_31;
                        float dzj_31 = gs_1[7] * (nj_31 * wclaim[7]) * sig_31 * __fmaf_rn(zs_1[7], 1.0f - sig_31, 1.0f);
                        float dnj_31 = aj_31 * wclaim[7];
                        dot_3 = __fmaf_rn(dnj_31, xs_1[7], dot_3);
                        acc[7] = __fmaf_rn(aj_31, nj_31, acc[7]);
                        zs_1[7] = dzj_31;
                        gs_1[7] = dnj_31;
                    }
                    float _shfl_xor_sync_443 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                    dot_3 = dot_3 + _shfl_xor_sync_443;
                    float _shfl_xor_sync_444 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                    dot_3 = dot_3 + _shfl_xor_sync_444;
                    float _shfl_xor_sync_445 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                    dot_3 = dot_3 + _shfl_xor_sync_445;
                    float _shfl_xor_sync_446 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 8);
                    dot_3 = dot_3 + _shfl_xor_sync_446;
                    float coef_3 = rstd_3 * rstd_3 * rstd_3 * (dot_3 * -0.0078125f);
                    float2 coef2_3 = make_float2(coef_3, coef_3);
                    float2 rstd2b_3 = make_float2(rstd_3, rstd_3);
                    float2 dx2_12 = fma_f32x2_rn_noftz(make_float2(xs_1[0], xs_1[1]), coef2_3, mul_f32x2_noftz(make_float2(gs_1[0], gs_1[1]), rstd2b_3));
                    xs_1[0] = dx2_12.x;
                    xs_1[1] = dx2_12.y;
                    float2 dx2_13 = fma_f32x2_rn_noftz(make_float2(xs_1[2], xs_1[3]), coef2_3, mul_f32x2_noftz(make_float2(gs_1[2], gs_1[3]), rstd2b_3));
                    xs_1[2] = dx2_13.x;
                    xs_1[3] = dx2_13.y;
                    float2 dx2_14 = fma_f32x2_rn_noftz(make_float2(xs_1[4], xs_1[5]), coef2_3, mul_f32x2_noftz(make_float2(gs_1[4], gs_1[5]), rstd2b_3));
                    xs_1[4] = dx2_14.x;
                    xs_1[5] = dx2_14.y;
                    float2 dx2_15 = fma_f32x2_rn_noftz(make_float2(xs_1[6], xs_1[7]), coef2_3, mul_f32x2_noftz(make_float2(gs_1[6], gs_1[7]), rstd2b_3));
                    xs_1[6] = dx2_15.x;
                    xs_1[7] = dx2_15.y;
                    unsigned int packed_dx_3[4];
                    unsigned int packed_dz_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 0], xs_1[_lp*2+1 + 0]));
                        packed_dx_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 0], zs_1[_lp*2+1 + 0]));
                        packed_dz_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 2], xs_1[_lp*2+1 + 2]));
                        packed_dx_3[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 2], zs_1[_lp*2+1 + 2]));
                        packed_dz_3[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 4], xs_1[_lp*2+1 + 4]));
                        packed_dx_3[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 4], zs_1[_lp*2+1 + 4]));
                        packed_dz_3[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 6], xs_1[_lp*2+1 + 6]));
                        packed_dx_3[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 6], zs_1[_lp*2+1 + 6]));
                        packed_dz_3[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    if (row_1 < rows) {
                        long long orow_3 = (long long)row_1 * 128ll + (long long)col;
                        reinterpret_cast<int4*>(dx + orow_3)[0] = reinterpret_cast<int4*>(packed_dx_3)[0];
                        reinterpret_cast<int4*>(dz + orow_3)[0] = reinterpret_cast<int4*>(packed_dz_3)[0];
                    }
                }
                run = run + 1;
                if (run == 16) {
                    if (folded != 0) {
                        float s_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(fold[group * 128 + col]), "f"(acc[0]));
                        float bv_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7) : "f"(s_7), "f"(fold[group * 128 + col]));
                        float av_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7) : "f"(s_7), "f"(bv_7));
                        float _sub_447;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_447) : "f"(fold[group * 128 + col]), "f"(av_7));
                        float _sub_448;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_448) : "f"(acc[0]), "f"(bv_7));
                        float e_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7) : "f"(_sub_447), "f"(_sub_448));
                        fold[group * 128 + col] = s_7;
                        float _add_449;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_449) : "f"(fold[(16 + group) * 128 + col]), "f"(e_7));
                        fold[(16 + group) * 128 + col] = _add_449;
                        acc[0] = 0.0f;
                        float s_7_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7_1) : "f"(fold[group * 128 + col + 1]), "f"(acc[1]));
                        float bv_7_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7_1) : "f"(s_7_1), "f"(fold[group * 128 + col + 1]));
                        float av_7_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7_1) : "f"(s_7_1), "f"(bv_7_1));
                        float _sub_450;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_450) : "f"(fold[group * 128 + col + 1]), "f"(av_7_1));
                        float _sub_451;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_451) : "f"(acc[1]), "f"(bv_7_1));
                        float e_7_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7_1) : "f"(_sub_450), "f"(_sub_451));
                        fold[group * 128 + col + 1] = s_7_1;
                        float _add_452;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_452) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_7_1));
                        fold[(16 + group) * 128 + col + 1] = _add_452;
                        acc[1] = 0.0f;
                        float s_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_8) : "f"(fold[group * 128 + col + 2]), "f"(acc[2]));
                        float bv_8;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_8) : "f"(s_8), "f"(fold[group * 128 + col + 2]));
                        float av_8;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_8) : "f"(s_8), "f"(bv_8));
                        float _sub_453;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_453) : "f"(fold[group * 128 + col + 2]), "f"(av_8));
                        float _sub_454;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_454) : "f"(acc[2]), "f"(bv_8));
                        float e_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_8) : "f"(_sub_453), "f"(_sub_454));
                        fold[group * 128 + col + 2] = s_8;
                        float _add_455;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_455) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_8));
                        fold[(16 + group) * 128 + col + 2] = _add_455;
                        acc[2] = 0.0f;
                        float s_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(fold[group * 128 + col + 3]), "f"(acc[3]));
                        float bv_9;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_9) : "f"(s_9), "f"(fold[group * 128 + col + 3]));
                        float av_9;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_9) : "f"(s_9), "f"(bv_9));
                        float _sub_456;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_456) : "f"(fold[group * 128 + col + 3]), "f"(av_9));
                        float _sub_457;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_457) : "f"(acc[3]), "f"(bv_9));
                        float e_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_9) : "f"(_sub_456), "f"(_sub_457));
                        fold[group * 128 + col + 3] = s_9;
                        float _add_458;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_458) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_9));
                        fold[(16 + group) * 128 + col + 3] = _add_458;
                        acc[3] = 0.0f;
                        float s_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(fold[group * 128 + col + 4]), "f"(acc[4]));
                        float bv_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_10) : "f"(s_10), "f"(fold[group * 128 + col + 4]));
                        float av_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_10) : "f"(s_10), "f"(bv_10));
                        float _sub_459;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_459) : "f"(fold[group * 128 + col + 4]), "f"(av_10));
                        float _sub_460;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_460) : "f"(acc[4]), "f"(bv_10));
                        float e_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_10) : "f"(_sub_459), "f"(_sub_460));
                        fold[group * 128 + col + 4] = s_10;
                        float _add_461;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_461) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_10));
                        fold[(16 + group) * 128 + col + 4] = _add_461;
                        acc[4] = 0.0f;
                        float s_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(fold[group * 128 + col + 5]), "f"(acc[5]));
                        float bv_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_11) : "f"(s_11), "f"(fold[group * 128 + col + 5]));
                        float av_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_11) : "f"(s_11), "f"(bv_11));
                        float _sub_462;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_462) : "f"(fold[group * 128 + col + 5]), "f"(av_11));
                        float _sub_463;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_463) : "f"(acc[5]), "f"(bv_11));
                        float e_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_11) : "f"(_sub_462), "f"(_sub_463));
                        fold[group * 128 + col + 5] = s_11;
                        float _add_464;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_464) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_11));
                        fold[(16 + group) * 128 + col + 5] = _add_464;
                        acc[5] = 0.0f;
                        float s_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(fold[group * 128 + col + 6]), "f"(acc[6]));
                        float bv_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_12) : "f"(s_12), "f"(fold[group * 128 + col + 6]));
                        float av_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_12) : "f"(s_12), "f"(bv_12));
                        float _sub_465;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_465) : "f"(fold[group * 128 + col + 6]), "f"(av_12));
                        float _sub_466;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_466) : "f"(acc[6]), "f"(bv_12));
                        float e_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_12) : "f"(_sub_465), "f"(_sub_466));
                        fold[group * 128 + col + 6] = s_12;
                        float _add_467;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_467) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_12));
                        fold[(16 + group) * 128 + col + 6] = _add_467;
                        acc[6] = 0.0f;
                        float s_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(fold[group * 128 + col + 7]), "f"(acc[7]));
                        float bv_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_13) : "f"(s_13), "f"(fold[group * 128 + col + 7]));
                        float av_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_13) : "f"(s_13), "f"(bv_13));
                        float _sub_468;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_468) : "f"(fold[group * 128 + col + 7]), "f"(av_13));
                        float _sub_469;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_469) : "f"(acc[7]), "f"(bv_13));
                        float e_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_13) : "f"(_sub_468), "f"(_sub_469));
                        fold[group * 128 + col + 7] = s_13;
                        float _add_470;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_470) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_13));
                        fold[(16 + group) * 128 + col + 7] = _add_470;
                        acc[7] = 0.0f;
                    } else {
                        fold[group * 128 + col] = acc[0];
                        fold[(16 + group) * 128 + col] = 0.0f;
                        acc[0] = 0.0f;
                        fold[group * 128 + col + 1] = acc[1];
                        fold[(16 + group) * 128 + col + 1] = 0.0f;
                        acc[1] = 0.0f;
                        fold[group * 128 + col + 2] = acc[2];
                        fold[(16 + group) * 128 + col + 2] = 0.0f;
                        acc[2] = 0.0f;
                        fold[group * 128 + col + 3] = acc[3];
                        fold[(16 + group) * 128 + col + 3] = 0.0f;
                        acc[3] = 0.0f;
                        fold[group * 128 + col + 4] = acc[4];
                        fold[(16 + group) * 128 + col + 4] = 0.0f;
                        acc[4] = 0.0f;
                        fold[group * 128 + col + 5] = acc[5];
                        fold[(16 + group) * 128 + col + 5] = 0.0f;
                        acc[5] = 0.0f;
                        fold[group * 128 + col + 6] = acc[6];
                        fold[(16 + group) * 128 + col + 6] = 0.0f;
                        acc[6] = 0.0f;
                        fold[group * 128 + col + 7] = acc[7];
                        fold[(16 + group) * 128 + col + 7] = 0.0f;
                        acc[7] = 0.0f;
                    }
                    folded = 1;
                    run = 0;
                }
            }
        }
    }
    float part32[8];
    float p32_0 = acc[0];
    float _shfl_xor_sync_471 = __shfl_xor_sync(0xFFFFFFFF, p32_0, 16);
    p32_0 = p32_0 + _shfl_xor_sync_471;
    part32[0] = p32_0;
    float p32_1 = acc[1];
    float _shfl_xor_sync_472 = __shfl_xor_sync(0xFFFFFFFF, p32_1, 16);
    p32_1 = p32_1 + _shfl_xor_sync_472;
    part32[1] = p32_1;
    float p32_2 = acc[2];
    float _shfl_xor_sync_473 = __shfl_xor_sync(0xFFFFFFFF, p32_2, 16);
    p32_2 = p32_2 + _shfl_xor_sync_473;
    part32[2] = p32_2;
    float p32_3 = acc[3];
    float _shfl_xor_sync_474 = __shfl_xor_sync(0xFFFFFFFF, p32_3, 16);
    p32_3 = p32_3 + _shfl_xor_sync_474;
    part32[3] = p32_3;
    float p32_4 = acc[4];
    float _shfl_xor_sync_475 = __shfl_xor_sync(0xFFFFFFFF, p32_4, 16);
    p32_4 = p32_4 + _shfl_xor_sync_475;
    part32[4] = p32_4;
    float p32_5 = acc[5];
    float _shfl_xor_sync_476 = __shfl_xor_sync(0xFFFFFFFF, p32_5, 16);
    p32_5 = p32_5 + _shfl_xor_sync_476;
    part32[5] = p32_5;
    float p32_6 = acc[6];
    float _shfl_xor_sync_477 = __shfl_xor_sync(0xFFFFFFFF, p32_6, 16);
    p32_6 = p32_6 + _shfl_xor_sync_477;
    part32[6] = p32_6;
    float p32_7 = acc[7];
    float _shfl_xor_sync_478 = __shfl_xor_sync(0xFFFFFFFF, p32_7, 16);
    p32_7 = p32_7 + _shfl_xor_sync_478;
    part32[7] = p32_7;
    int short_ = 0;
    if (folded == 0) {
        if (run <= 12) {
            short_ = 1;
        }
    }
    if (short_ != 0) {
        if ((tid & 31) < 16) {
            fold32[(tid >> 5) * 128 + col] = part32[0];
            fold32[(tid >> 5) * 128 + col + 1] = part32[1];
            fold32[(tid >> 5) * 128 + col + 2] = part32[2];
            fold32[(tid >> 5) * 128 + col + 3] = part32[3];
            fold32[(tid >> 5) * 128 + col + 4] = part32[4];
            fold32[(tid >> 5) * 128 + col + 5] = part32[5];
            fold32[(tid >> 5) * 128 + col + 6] = part32[6];
            fold32[(tid >> 5) * 128 + col + 7] = part32[7];
        }
    } else if (folded != 0) {
        float s_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(fold[group * 128 + col]), "f"(acc[0]));
        float bv_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14) : "f"(s_14), "f"(fold[group * 128 + col]));
        float av_14;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14) : "f"(s_14), "f"(bv_14));
        float _sub_479;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_479) : "f"(fold[group * 128 + col]), "f"(av_14));
        float _sub_480;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_480) : "f"(acc[0]), "f"(bv_14));
        float e_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14) : "f"(_sub_479), "f"(_sub_480));
        fold[group * 128 + col] = s_14;
        float _add_481;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_481) : "f"(fold[(16 + group) * 128 + col]), "f"(e_14));
        fold[(16 + group) * 128 + col] = _add_481;
        acc[0] = 0.0f;
        float s_14_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14_1) : "f"(fold[group * 128 + col + 1]), "f"(acc[1]));
        float bv_14_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14_1) : "f"(s_14_1), "f"(fold[group * 128 + col + 1]));
        float av_14_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14_1) : "f"(s_14_1), "f"(bv_14_1));
        float _sub_482;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_482) : "f"(fold[group * 128 + col + 1]), "f"(av_14_1));
        float _sub_483;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_483) : "f"(acc[1]), "f"(bv_14_1));
        float e_14_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14_1) : "f"(_sub_482), "f"(_sub_483));
        fold[group * 128 + col + 1] = s_14_1;
        float _add_484;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_484) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_14_1));
        fold[(16 + group) * 128 + col + 1] = _add_484;
        acc[1] = 0.0f;
        float s_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(fold[group * 128 + col + 2]), "f"(acc[2]));
        float bv_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_15) : "f"(s_15), "f"(fold[group * 128 + col + 2]));
        float av_15;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_15) : "f"(s_15), "f"(bv_15));
        float _sub_485;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_485) : "f"(fold[group * 128 + col + 2]), "f"(av_15));
        float _sub_486;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_486) : "f"(acc[2]), "f"(bv_15));
        float e_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_15) : "f"(_sub_485), "f"(_sub_486));
        fold[group * 128 + col + 2] = s_15;
        float _add_487;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_487) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_15));
        fold[(16 + group) * 128 + col + 2] = _add_487;
        acc[2] = 0.0f;
        float s_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(fold[group * 128 + col + 3]), "f"(acc[3]));
        float bv_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_16) : "f"(s_16), "f"(fold[group * 128 + col + 3]));
        float av_16;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_16) : "f"(s_16), "f"(bv_16));
        float _sub_488;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_488) : "f"(fold[group * 128 + col + 3]), "f"(av_16));
        float _sub_489;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_489) : "f"(acc[3]), "f"(bv_16));
        float e_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_16) : "f"(_sub_488), "f"(_sub_489));
        fold[group * 128 + col + 3] = s_16;
        float _add_490;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_490) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_16));
        fold[(16 + group) * 128 + col + 3] = _add_490;
        acc[3] = 0.0f;
        float s_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(fold[group * 128 + col + 4]), "f"(acc[4]));
        float bv_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_17) : "f"(s_17), "f"(fold[group * 128 + col + 4]));
        float av_17;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_17) : "f"(s_17), "f"(bv_17));
        float _sub_491;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_491) : "f"(fold[group * 128 + col + 4]), "f"(av_17));
        float _sub_492;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_492) : "f"(acc[4]), "f"(bv_17));
        float e_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_17) : "f"(_sub_491), "f"(_sub_492));
        fold[group * 128 + col + 4] = s_17;
        float _add_493;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_493) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_17));
        fold[(16 + group) * 128 + col + 4] = _add_493;
        acc[4] = 0.0f;
        float s_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(fold[group * 128 + col + 5]), "f"(acc[5]));
        float bv_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_18) : "f"(s_18), "f"(fold[group * 128 + col + 5]));
        float av_18;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_18) : "f"(s_18), "f"(bv_18));
        float _sub_494;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_494) : "f"(fold[group * 128 + col + 5]), "f"(av_18));
        float _sub_495;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_495) : "f"(acc[5]), "f"(bv_18));
        float e_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_18) : "f"(_sub_494), "f"(_sub_495));
        fold[group * 128 + col + 5] = s_18;
        float _add_496;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_496) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_18));
        fold[(16 + group) * 128 + col + 5] = _add_496;
        acc[5] = 0.0f;
        float s_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(fold[group * 128 + col + 6]), "f"(acc[6]));
        float bv_19;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_19) : "f"(s_19), "f"(fold[group * 128 + col + 6]));
        float av_19;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_19) : "f"(s_19), "f"(bv_19));
        float _sub_497;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_497) : "f"(fold[group * 128 + col + 6]), "f"(av_19));
        float _sub_498;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_498) : "f"(acc[6]), "f"(bv_19));
        float e_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_19) : "f"(_sub_497), "f"(_sub_498));
        fold[group * 128 + col + 6] = s_19;
        float _add_499;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_499) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_19));
        fold[(16 + group) * 128 + col + 6] = _add_499;
        acc[6] = 0.0f;
        float s_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(fold[group * 128 + col + 7]), "f"(acc[7]));
        float bv_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_20) : "f"(s_20), "f"(fold[group * 128 + col + 7]));
        float av_20;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_20) : "f"(s_20), "f"(bv_20));
        float _sub_500;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_500) : "f"(fold[group * 128 + col + 7]), "f"(av_20));
        float _sub_501;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_501) : "f"(acc[7]), "f"(bv_20));
        float e_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_20) : "f"(_sub_500), "f"(_sub_501));
        fold[group * 128 + col + 7] = s_20;
        float _add_502;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_502) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_20));
        fold[(16 + group) * 128 + col + 7] = _add_502;
        acc[7] = 0.0f;
    } else {
        fold[group * 128 + col] = acc[0];
        fold[(16 + group) * 128 + col] = 0.0f;
        acc[0] = 0.0f;
        fold[group * 128 + col + 1] = acc[1];
        fold[(16 + group) * 128 + col + 1] = 0.0f;
        acc[1] = 0.0f;
        fold[group * 128 + col + 2] = acc[2];
        fold[(16 + group) * 128 + col + 2] = 0.0f;
        acc[2] = 0.0f;
        fold[group * 128 + col + 3] = acc[3];
        fold[(16 + group) * 128 + col + 3] = 0.0f;
        acc[3] = 0.0f;
        fold[group * 128 + col + 4] = acc[4];
        fold[(16 + group) * 128 + col + 4] = 0.0f;
        acc[4] = 0.0f;
        fold[group * 128 + col + 5] = acc[5];
        fold[(16 + group) * 128 + col + 5] = 0.0f;
        acc[5] = 0.0f;
        fold[group * 128 + col + 6] = acc[6];
        fold[(16 + group) * 128 + col + 6] = 0.0f;
        acc[6] = 0.0f;
        fold[group * 128 + col + 7] = acc[7];
        fold[(16 + group) * 128 + col + 7] = 0.0f;
        acc[7] = 0.0f;
    }
    __syncthreads();
    if (weight_grad != 0) {
        if (tid < 128) {
            double colsum = 0.0;
            if (short_ != 0) {
                float w32[8];
                w32[0] = fold32[tid];
                w32[1] = fold32[128 + tid];
                w32[2] = fold32[256 + tid];
                w32[3] = fold32[384 + tid];
                w32[4] = fold32[512 + tid];
                w32[5] = fold32[640 + tid];
                w32[6] = fold32[768 + tid];
                w32[7] = fold32[896 + tid];
                w32[0] = w32[0] + w32[1];
                w32[2] = w32[2] + w32[3];
                w32[4] = w32[4] + w32[5];
                w32[6] = w32[6] + w32[7];
                w32[0] = w32[0] + w32[2];
                w32[4] = w32[4] + w32[6];
                w32[0] = w32[0] + w32[4];
                colsum = (double)w32[0];
            } else {
                float hs = 0.0f;
                float ls = 0.0f;
                #pragma unroll
                for (int g = 0; g < 16; g++) {
                    float s_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(hs), "f"(fold[g * 128 + tid]));
                    float bv_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_21) : "f"(s_21), "f"(hs));
                    float av_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_21) : "f"(s_21), "f"(bv_21));
                    float _sub_503;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_503) : "f"(hs), "f"(av_21));
                    float _sub_504;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_504) : "f"(fold[g * 128 + tid]), "f"(bv_21));
                    float e_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_21) : "f"(_sub_503), "f"(_sub_504));
                    hs = s_21;
                    float _add_506;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_506) : "f"(e_21), "f"(fold[(16 + g) * 128 + tid]));
                    float _add_505;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_505) : "f"(ls), "f"(_add_506));
                    ls = _add_505;
                }
                float _abs_507;
                asm("abs.f32 %0, %1;" : "=f"(_abs_507) : "f"(hs));
                if (_abs_507 <= 3.4028235e+38f) {
                    colsum = (double)hs + (double)ls;
                } else {
                    colsum = (double)hs;
                }
            }
            if (colsum != 0.0) {
                atomicAdd(&dw_sums[(bid & 3) * 128 + tid], colsum);
            }
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
            if (tid < 128) {
                double dw_total = 0.0;
                double part_0 = atomicAdd(&dw_sums[tid], -0.0);
                dw_total = dw_total + part_0;
                dw_sums[tid] = 0.0;
                double part_1 = atomicAdd(&dw_sums[128 + tid], -0.0);
                dw_total = dw_total + part_1;
                dw_sums[128 + tid] = 0.0;
                double part_2 = atomicAdd(&dw_sums[256 + tid], -0.0);
                dw_total = dw_total + part_2;
                dw_sums[256 + tid] = 0.0;
                double part_3 = atomicAdd(&dw_sums[384 + tid], -0.0);
                dw_total = dw_total + part_3;
                dw_sums[384 + tid] = 0.0;
                if (accumulate != 0) {
                    __nv_bfloat16 prev = atomicAdd(&dw[tid], -0.0f);
                    dw_total = dw_total + (double)(float)prev;
                }
                dw[tid] = (__nv_bfloat16)(float)dw_total;
            }
        }
    }
}

} // extern "C"
