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
kernel_cake_rmsnorm_silu_gate_det_train_6b4739305d767297d565(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, unsigned long long* __restrict__ acc, unsigned int* __restrict__ flags, unsigned int* __restrict__ done, float* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(acc));
    __builtin_assume(__isGlobal(acc));
    asm volatile("" : "+l"(flags));
    __builtin_assume(__isGlobal(flags));

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
    unsigned long long* exa = reinterpret_cast<unsigned long long*>(smem_raw + 0);
    const int exa_addr = smem + 0;
    unsigned int* exf = reinterpret_cast<unsigned int*>(smem_raw + 9216);
    const int exf_addr = smem + 9216;
    int* cell = reinterpret_cast<int*>(smem_raw + 9728);
    const int cell_addr = smem + 9728;
    int* flag = reinterpret_cast<int*>(smem_raw + 9744);
    const int flag_addr = smem + 9744;
    int* tab = reinterpret_cast<int*>(smem_raw + 9760);
    const int tab_addr = smem + 9760;
    float* fold = reinterpret_cast<float*>(smem_raw + 9888);
    const int fold_addr = smem + 9888;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int nblocks = (rows + 15) / 16;
    if (tid < 128) {
        exa[tid] = 0ull;
        exa[128 + tid] = 0ull;
        exa[256 + tid] = 0ull;
        exa[384 + tid] = 0ull;
        exa[512 + tid] = 0ull;
        exa[640 + tid] = 0ull;
        exa[768 + tid] = 0ull;
        exa[896 + tid] = 0ull;
        exa[1024 + tid] = 0ull;
        exf[tid] = 0;
    }
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
                : "=r"(_atomic_fetch_inc_1) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
            ticket = (int)_atomic_fetch_inc_1;
        }
    }
    if (rest > 0) {
        int _min_2 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int sl0 = ((_min_2) > (1) ? (_min_2) : (1));
        if (bid < sl0) {
            int pblk = bid * (((rest + sl0 - 1) / sl0 + 1) / 2);
            int prow = ((pblk * 16 + group) < (rows - 1) ? (pblk * 16 + group) : (rows - 1));
            uint32_t pq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)prow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)prow);
            uint32_t ph = (uint32_t)((unsigned int)prow) - pq * (uint32_t)(heads.divisor);
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)pq * dy_tok + (long long)ph * dy_head + (long long)col))));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)pq * x_tok + (long long)ph * x_head + (long long)col))));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)pq * z_tok + (long long)ph * z_head + (long long)col))));
        }
    }
    float acc_r[8];
    acc_r[0] = 0.0f;
    acc_r[1] = 0.0f;
    acc_r[2] = 0.0f;
    acc_r[3] = 0.0f;
    acc_r[4] = 0.0f;
    acc_r[5] = 0.0f;
    acc_r[6] = 0.0f;
    acc_r[7] = 0.0f;
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
            int row = u * 16 + group;
            int src = ((row) < (rows - 1) ? (row) : (rows - 1));
            uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
            uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
            long long goff = (long long)tq * dy_tok + (long long)th * dy_head + (long long)col;
            long long xoff = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
            long long zoff = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
            float gr[8];
            float xr[8];
            float zr[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(dy + goff + 0);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                        (&gr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + xoff + 0);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                        (&xr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(z + zoff + 0);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                        (&zr[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
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
                    int nrow = (((u + 1) * 16 + group) < (rows - 1) ? ((u + 1) * 16 + group) : (rows - 1));
                    uint32_t nq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow);
                    uint32_t nh = (uint32_t)((unsigned int)nrow) - nq * (uint32_t)(heads.divisor);
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq * dy_tok + (long long)nh * dy_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq * x_tok + (long long)nh * x_head + (long long)col))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq * z_tok + (long long)nh * z_head + (long long)col))));
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
            float total = 0.0f;
            total = __fmaf_rn(xr[0], xr[0], total);
            total = __fmaf_rn(xr[1], xr[1], total);
            total = __fmaf_rn(xr[2], xr[2], total);
            total = __fmaf_rn(xr[3], xr[3], total);
            total = __fmaf_rn(xr[4], xr[4], total);
            total = __fmaf_rn(xr[5], xr[5], total);
            total = __fmaf_rn(xr[6], xr[6], total);
            total = __fmaf_rn(xr[7], xr[7], total);
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
            total = total + _shfl_xor_sync_8;
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_sync_9;
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
            total = total + _shfl_xor_sync_10;
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
            total = total + _shfl_xor_sync_11;
            float _sqrt_12;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_12) : "f"(total * 0.0078125f + eps));
            float rstd = __frcp_rn(_sqrt_12);
            float dot = 0.0f;
            int fast = 0;
            float _neg_13;
            asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(zr[0]));
            float _neg_14;
            asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(zr[1]));
            float _neg_15;
            asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(zr[2]));
            float mz;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz) : "f"(_neg_13), "f"(_neg_14), "f"(_neg_15));
            float _abs_17;
            asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(zr[0]));
            float _neg_16;
            asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(_abs_17));
            float _abs_19;
            asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(zr[1]));
            float _neg_18;
            asm("neg.f32 %0, %1;" : "=f"(_neg_18) : "f"(_abs_19));
            float _abs_21;
            asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(zr[2]));
            float _neg_20;
            asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(_abs_21));
            float ma;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma) : "f"(_neg_16), "f"(_neg_18), "f"(_neg_20));
            float _neg_23;
            asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(zr[3]));
            float _neg_24;
            asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(zr[4]));
            float _max3_22;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(mz), "f"(_neg_23), "f"(_neg_24));
            mz = _max3_22;
            float _abs_27;
            asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(zr[3]));
            float _neg_26;
            asm("neg.f32 %0, %1;" : "=f"(_neg_26) : "f"(_abs_27));
            float _abs_29;
            asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(zr[4]));
            float _neg_28;
            asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(_abs_29));
            float _max3_25;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(ma), "f"(_neg_26), "f"(_neg_28));
            ma = _max3_25;
            float _neg_31;
            asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(zr[5]));
            float _neg_32;
            asm("neg.f32 %0, %1;" : "=f"(_neg_32) : "f"(zr[6]));
            float _max3_30;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(mz), "f"(_neg_31), "f"(_neg_32));
            mz = _max3_30;
            float _abs_35;
            asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(zr[5]));
            float _neg_34;
            asm("neg.f32 %0, %1;" : "=f"(_neg_34) : "f"(_abs_35));
            float _abs_37;
            asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(zr[6]));
            float _neg_36;
            asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(_abs_37));
            float _max3_33;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_33) : "f"(ma), "f"(_neg_34), "f"(_neg_36));
            ma = _max3_33;
            float _neg_39;
            asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(zr[7]));
            float _max_38;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_38) : "f"(mz), "f"(_neg_39));
            mz = _max_38;
            float _abs_42;
            asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(zr[7]));
            float _neg_41;
            asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(_abs_42));
            float _max_40;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(ma), "f"(_neg_41));
            ma = _max_40;
            float _abs_43;
            asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(zr[0]));
            float _abs_44;
            asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(zr[1]));
            float _abs_45;
            asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(zr[2]));
            float mp;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp) : "f"(_abs_43), "f"(_abs_44), "f"(_abs_45));
            float _abs_47;
            asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(zr[3]));
            float _abs_48;
            asm("abs.f32 %0, %1;" : "=f"(_abs_48) : "f"(zr[4]));
            float _max3_46;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_46) : "f"(mp), "f"(_abs_47), "f"(_abs_48));
            mp = _max3_46;
            float _abs_50;
            asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(zr[5]));
            float _abs_51;
            asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(zr[6]));
            float _max3_49;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_49) : "f"(mp), "f"(_abs_50), "f"(_abs_51));
            mp = _max3_49;
            float _abs_53;
            asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(zr[7]));
            float _max_52;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(mp), "f"(_abs_53));
            mp = _max_52;
            if (mz <= 41.0f) {
                if (ma <= -1.1754944e-38f) {
                    if (mp <= 3.4028235e+38f) {
                        fast = 1;
                    }
                }
            }
            if (fast != 0) {
                float2 rstd2 = make_float2(rstd, rstd);
                float2 z2_0 = make_float2(zr[0], zr[1]);
                float ta_0;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zr[0]), "f"(-0.00572498f), "f"(0.5f));
                float tb_0;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zr[1]), "f"(-0.00572498f), "f"(0.5f));
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
                float2 g2_0 = make_float2(gr[0], gr[1]);
                float2 w2_0 = make_float2(wv[0], wv[1]);
                float2 _f32x2_mul_a_62 = make_float2(xr[0], xr[1]);
                float2 n2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_63 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
                float2 _f32x2_mul_b_64 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
                float2 dz2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
                float2 a2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
                float2 dn2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
                float2 acc2_0 = fma_f32x2_rn_noftz(a2_0, n2_0, make_float2(acc_r[0], acc_r[1]));
                acc_r[0] = ((row < rows) ? acc2_0.x : acc_r[0]);
                acc_r[1] = ((row < rows) ? acc2_0.y : acc_r[1]);
                zr[0] = dz2_0.x;
                zr[1] = dz2_0.y;
                gr[0] = dn2_0.x;
                gr[1] = dn2_0.y;
                float2 z2_1 = make_float2(zr[2], zr[3]);
                float ta_1;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zr[2]), "f"(-0.00572498f), "f"(0.5f));
                float tb_1;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zr[3]), "f"(-0.00572498f), "f"(0.5f));
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
                float2 g2_1 = make_float2(gr[2], gr[3]);
                float2 w2_1 = make_float2(wv[2], wv[3]);
                float2 _f32x2_mul_a_73 = make_float2(xr[2], xr[3]);
                float2 n2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_74 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
                float2 _f32x2_mul_b_75 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
                float2 dz2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                float2 a2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
                float2 dn2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
                float2 acc2_1 = fma_f32x2_rn_noftz(a2_1, n2_1, make_float2(acc_r[2], acc_r[3]));
                acc_r[2] = ((row < rows) ? acc2_1.x : acc_r[2]);
                acc_r[3] = ((row < rows) ? acc2_1.y : acc_r[3]);
                zr[2] = dz2_1.x;
                zr[3] = dz2_1.y;
                gr[2] = dn2_1.x;
                gr[3] = dn2_1.y;
                float2 z2_2 = make_float2(zr[4], zr[5]);
                float ta_2;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zr[4]), "f"(-0.00572498f), "f"(0.5f));
                float tb_2;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zr[5]), "f"(-0.00572498f), "f"(0.5f));
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
                float2 g2_2 = make_float2(gr[4], gr[5]);
                float2 w2_2 = make_float2(wv[4], wv[5]);
                float2 _f32x2_mul_a_84 = make_float2(xr[4], xr[5]);
                float2 n2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_85 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
                float2 _f32x2_mul_b_86 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
                float2 dz2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                float2 a2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
                float2 dn2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
                float2 acc2_2 = fma_f32x2_rn_noftz(a2_2, n2_2, make_float2(acc_r[4], acc_r[5]));
                acc_r[4] = ((row < rows) ? acc2_2.x : acc_r[4]);
                acc_r[5] = ((row < rows) ? acc2_2.y : acc_r[5]);
                zr[4] = dz2_2.x;
                zr[5] = dz2_2.y;
                gr[4] = dn2_2.x;
                gr[5] = dn2_2.y;
                float2 z2_3 = make_float2(zr[6], zr[7]);
                float ta_3;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zr[6]), "f"(-0.00572498f), "f"(0.5f));
                float tb_3;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zr[7]), "f"(-0.00572498f), "f"(0.5f));
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
                float2 g2_3 = make_float2(gr[6], gr[7]);
                float2 w2_3 = make_float2(wv[6], wv[7]);
                float2 _f32x2_mul_a_95 = make_float2(xr[6], xr[7]);
                float2 n2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_96 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
                float2 _f32x2_mul_b_97 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
                float2 dz2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_97));
                float2 a2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
                float2 dn2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
                float2 acc2_3 = fma_f32x2_rn_noftz(a2_3, n2_3, make_float2(acc_r[6], acc_r[7]));
                acc_r[6] = ((row < rows) ? acc2_3.x : acc_r[6]);
                acc_r[7] = ((row < rows) ? acc2_3.y : acc_r[7]);
                zr[6] = dz2_3.x;
                zr[7] = dz2_3.y;
                gr[6] = dn2_3.x;
                gr[7] = dn2_3.y;
                dot = __fmaf_rn(gr[0], xr[0], dot);
                dot = __fmaf_rn(gr[1], xr[1], dot);
                dot = __fmaf_rn(gr[2], xr[2], dot);
                dot = __fmaf_rn(gr[3], xr[3], dot);
                dot = __fmaf_rn(gr[4], xr[4], dot);
                dot = __fmaf_rn(gr[5], xr[5], dot);
                dot = __fmaf_rn(gr[6], xr[6], dot);
                dot = __fmaf_rn(gr[7], xr[7], dot);
            } else {
                float _exp_98 = __nv_expf(zr[0] * -1.0f);
                float den_0 = 1.0f + _exp_98;
                float sig_0 = __fdiv_rn(1.0f, den_0);
                float gate_0 = __fdiv_rn(zr[0], den_0);
                float nj_0 = xr[0] * rstd;
                float aj_0 = gr[0] * gate_0;
                float dzj_0 = gr[0] * (nj_0 * wv[0]) * sig_0 * __fmaf_rn(zr[0], 1.0f - sig_0, 1.0f);
                float dnj_0 = aj_0 * wv[0];
                dot = __fmaf_rn(dnj_0, xr[0], dot);
                acc_r[0] = ((row < rows) ? __fmaf_rn(aj_0, nj_0, acc_r[0]) : acc_r[0]);
                zr[0] = dzj_0;
                gr[0] = dnj_0;
                float _exp_99 = __nv_expf(zr[1] * -1.0f);
                float den_1 = 1.0f + _exp_99;
                float sig_1 = __fdiv_rn(1.0f, den_1);
                float gate_1 = __fdiv_rn(zr[1], den_1);
                float nj_1 = xr[1] * rstd;
                float aj_1 = gr[1] * gate_1;
                float dzj_1 = gr[1] * (nj_1 * wv[1]) * sig_1 * __fmaf_rn(zr[1], 1.0f - sig_1, 1.0f);
                float dnj_1 = aj_1 * wv[1];
                dot = __fmaf_rn(dnj_1, xr[1], dot);
                acc_r[1] = ((row < rows) ? __fmaf_rn(aj_1, nj_1, acc_r[1]) : acc_r[1]);
                zr[1] = dzj_1;
                gr[1] = dnj_1;
                float _exp_100 = __nv_expf(zr[2] * -1.0f);
                float den_2 = 1.0f + _exp_100;
                float sig_2 = __fdiv_rn(1.0f, den_2);
                float gate_2 = __fdiv_rn(zr[2], den_2);
                float nj_2 = xr[2] * rstd;
                float aj_2 = gr[2] * gate_2;
                float dzj_2 = gr[2] * (nj_2 * wv[2]) * sig_2 * __fmaf_rn(zr[2], 1.0f - sig_2, 1.0f);
                float dnj_2 = aj_2 * wv[2];
                dot = __fmaf_rn(dnj_2, xr[2], dot);
                acc_r[2] = ((row < rows) ? __fmaf_rn(aj_2, nj_2, acc_r[2]) : acc_r[2]);
                zr[2] = dzj_2;
                gr[2] = dnj_2;
                float _exp_101 = __nv_expf(zr[3] * -1.0f);
                float den_3 = 1.0f + _exp_101;
                float sig_3 = __fdiv_rn(1.0f, den_3);
                float gate_3 = __fdiv_rn(zr[3], den_3);
                float nj_3 = xr[3] * rstd;
                float aj_3 = gr[3] * gate_3;
                float dzj_3 = gr[3] * (nj_3 * wv[3]) * sig_3 * __fmaf_rn(zr[3], 1.0f - sig_3, 1.0f);
                float dnj_3 = aj_3 * wv[3];
                dot = __fmaf_rn(dnj_3, xr[3], dot);
                acc_r[3] = ((row < rows) ? __fmaf_rn(aj_3, nj_3, acc_r[3]) : acc_r[3]);
                zr[3] = dzj_3;
                gr[3] = dnj_3;
                float _exp_102 = __nv_expf(zr[4] * -1.0f);
                float den_4 = 1.0f + _exp_102;
                float sig_4 = __fdiv_rn(1.0f, den_4);
                float gate_4 = __fdiv_rn(zr[4], den_4);
                float nj_4 = xr[4] * rstd;
                float aj_4 = gr[4] * gate_4;
                float dzj_4 = gr[4] * (nj_4 * wv[4]) * sig_4 * __fmaf_rn(zr[4], 1.0f - sig_4, 1.0f);
                float dnj_4 = aj_4 * wv[4];
                dot = __fmaf_rn(dnj_4, xr[4], dot);
                acc_r[4] = ((row < rows) ? __fmaf_rn(aj_4, nj_4, acc_r[4]) : acc_r[4]);
                zr[4] = dzj_4;
                gr[4] = dnj_4;
                float _exp_103 = __nv_expf(zr[5] * -1.0f);
                float den_5 = 1.0f + _exp_103;
                float sig_5 = __fdiv_rn(1.0f, den_5);
                float gate_5 = __fdiv_rn(zr[5], den_5);
                float nj_5 = xr[5] * rstd;
                float aj_5 = gr[5] * gate_5;
                float dzj_5 = gr[5] * (nj_5 * wv[5]) * sig_5 * __fmaf_rn(zr[5], 1.0f - sig_5, 1.0f);
                float dnj_5 = aj_5 * wv[5];
                dot = __fmaf_rn(dnj_5, xr[5], dot);
                acc_r[5] = ((row < rows) ? __fmaf_rn(aj_5, nj_5, acc_r[5]) : acc_r[5]);
                zr[5] = dzj_5;
                gr[5] = dnj_5;
                float _exp_104 = __nv_expf(zr[6] * -1.0f);
                float den_6 = 1.0f + _exp_104;
                float sig_6 = __fdiv_rn(1.0f, den_6);
                float gate_6 = __fdiv_rn(zr[6], den_6);
                float nj_6 = xr[6] * rstd;
                float aj_6 = gr[6] * gate_6;
                float dzj_6 = gr[6] * (nj_6 * wv[6]) * sig_6 * __fmaf_rn(zr[6], 1.0f - sig_6, 1.0f);
                float dnj_6 = aj_6 * wv[6];
                dot = __fmaf_rn(dnj_6, xr[6], dot);
                acc_r[6] = ((row < rows) ? __fmaf_rn(aj_6, nj_6, acc_r[6]) : acc_r[6]);
                zr[6] = dzj_6;
                gr[6] = dnj_6;
                float _exp_105 = __nv_expf(zr[7] * -1.0f);
                float den_7 = 1.0f + _exp_105;
                float sig_7 = __fdiv_rn(1.0f, den_7);
                float gate_7 = __fdiv_rn(zr[7], den_7);
                float nj_7 = xr[7] * rstd;
                float aj_7 = gr[7] * gate_7;
                float dzj_7 = gr[7] * (nj_7 * wv[7]) * sig_7 * __fmaf_rn(zr[7], 1.0f - sig_7, 1.0f);
                float dnj_7 = aj_7 * wv[7];
                dot = __fmaf_rn(dnj_7, xr[7], dot);
                acc_r[7] = ((row < rows) ? __fmaf_rn(aj_7, nj_7, acc_r[7]) : acc_r[7]);
                zr[7] = dzj_7;
                gr[7] = dnj_7;
            }
            float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
            dot = dot + _shfl_xor_sync_106;
            float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
            dot = dot + _shfl_xor_sync_107;
            float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
            dot = dot + _shfl_xor_sync_108;
            float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
            dot = dot + _shfl_xor_sync_109;
            float coef = rstd * rstd * rstd * (dot * -0.0078125f);
            float2 coef2 = make_float2(coef, coef);
            float2 rstd2b = make_float2(rstd, rstd);
            float2 dx2_0 = fma_f32x2_rn_noftz(make_float2(xr[0], xr[1]), coef2, mul_f32x2_noftz(make_float2(gr[0], gr[1]), rstd2b));
            xr[0] = dx2_0.x;
            xr[1] = dx2_0.y;
            float2 dx2_1 = fma_f32x2_rn_noftz(make_float2(xr[2], xr[3]), coef2, mul_f32x2_noftz(make_float2(gr[2], gr[3]), rstd2b));
            xr[2] = dx2_1.x;
            xr[3] = dx2_1.y;
            float2 dx2_2 = fma_f32x2_rn_noftz(make_float2(xr[4], xr[5]), coef2, mul_f32x2_noftz(make_float2(gr[4], gr[5]), rstd2b));
            xr[4] = dx2_2.x;
            xr[5] = dx2_2.y;
            float2 dx2_3 = fma_f32x2_rn_noftz(make_float2(xr[6], xr[7]), coef2, mul_f32x2_noftz(make_float2(gr[6], gr[7]), rstd2b));
            xr[6] = dx2_3.x;
            xr[7] = dx2_3.y;
            unsigned int packed_dx[4];
            unsigned int packed_dz[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 0], xr[_lp*2+1 + 0]));
                packed_dx[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 0], zr[_lp*2+1 + 0]));
                packed_dz[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 2], xr[_lp*2+1 + 2]));
                packed_dx[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 2], zr[_lp*2+1 + 2]));
                packed_dz[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 4], xr[_lp*2+1 + 4]));
                packed_dx[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 4], zr[_lp*2+1 + 4]));
                packed_dz[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 6], xr[_lp*2+1 + 6]));
                packed_dx[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 6], zr[_lp*2+1 + 6]));
                packed_dz[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (row < rows) {
                long long orow = (long long)row * 128ll + (long long)col;
                reinterpret_cast<int4*>(dx + orow)[0] = reinterpret_cast<int4*>(packed_dx)[0];
                reinterpret_cast<int4*>(dz + orow)[0] = reinterpret_cast<int4*>(packed_dz)[0];
            }
            run = run + 1;
            if (run == 16) {
                if (folded != 0) {
                    float s;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                    float bv;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv) : "f"(s), "f"(fold[group * 128 + col]));
                    float av;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av) : "f"(s), "f"(bv));
                    float _sub_110;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_110) : "f"(fold[group * 128 + col]), "f"(av));
                    float _sub_111;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_111) : "f"(acc_r[0]), "f"(bv));
                    float e;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_110), "f"(_sub_111));
                    fold[group * 128 + col] = s;
                    float _add_112;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(fold[(16 + group) * 128 + col]), "f"(e));
                    fold[(16 + group) * 128 + col] = _add_112;
                    acc_r[0] = 0.0f;
                    float s_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_0) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                    float bv_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_0) : "f"(s_0), "f"(fold[group * 128 + col + 1]));
                    float av_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_0) : "f"(s_0), "f"(bv_0));
                    float _sub_113;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(fold[group * 128 + col + 1]), "f"(av_0));
                    float _sub_114;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_114) : "f"(acc_r[1]), "f"(bv_0));
                    float e_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_113), "f"(_sub_114));
                    fold[group * 128 + col + 1] = s_0;
                    float _add_115;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_0));
                    fold[(16 + group) * 128 + col + 1] = _add_115;
                    acc_r[1] = 0.0f;
                    float s_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                    float bv_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_1) : "f"(s_1), "f"(fold[group * 128 + col + 2]));
                    float av_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_1) : "f"(s_1), "f"(bv_1));
                    float _sub_116;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_116) : "f"(fold[group * 128 + col + 2]), "f"(av_1));
                    float _sub_117;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_117) : "f"(acc_r[2]), "f"(bv_1));
                    float e_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_116), "f"(_sub_117));
                    fold[group * 128 + col + 2] = s_1;
                    float _add_118;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_1));
                    fold[(16 + group) * 128 + col + 2] = _add_118;
                    acc_r[2] = 0.0f;
                    float s_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_2) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                    float bv_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_2) : "f"(s_2), "f"(fold[group * 128 + col + 3]));
                    float av_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_2) : "f"(s_2), "f"(bv_2));
                    float _sub_119;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_119) : "f"(fold[group * 128 + col + 3]), "f"(av_2));
                    float _sub_120;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(acc_r[3]), "f"(bv_2));
                    float e_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_119), "f"(_sub_120));
                    fold[group * 128 + col + 3] = s_2;
                    float _add_121;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_2));
                    fold[(16 + group) * 128 + col + 3] = _add_121;
                    acc_r[3] = 0.0f;
                    float s_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_3) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                    float bv_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_3) : "f"(s_3), "f"(fold[group * 128 + col + 4]));
                    float av_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_3) : "f"(s_3), "f"(bv_3));
                    float _sub_122;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_122) : "f"(fold[group * 128 + col + 4]), "f"(av_3));
                    float _sub_123;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_123) : "f"(acc_r[4]), "f"(bv_3));
                    float e_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_122), "f"(_sub_123));
                    fold[group * 128 + col + 4] = s_3;
                    float _add_124;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_3));
                    fold[(16 + group) * 128 + col + 4] = _add_124;
                    acc_r[4] = 0.0f;
                    float s_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_4) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                    float bv_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_4) : "f"(s_4), "f"(fold[group * 128 + col + 5]));
                    float av_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_4) : "f"(s_4), "f"(bv_4));
                    float _sub_125;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_125) : "f"(fold[group * 128 + col + 5]), "f"(av_4));
                    float _sub_126;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_126) : "f"(acc_r[5]), "f"(bv_4));
                    float e_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_4) : "f"(_sub_125), "f"(_sub_126));
                    fold[group * 128 + col + 5] = s_4;
                    float _add_127;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_4));
                    fold[(16 + group) * 128 + col + 5] = _add_127;
                    acc_r[5] = 0.0f;
                    float s_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_5) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                    float bv_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_5) : "f"(s_5), "f"(fold[group * 128 + col + 6]));
                    float av_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_5) : "f"(s_5), "f"(bv_5));
                    float _sub_128;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_128) : "f"(fold[group * 128 + col + 6]), "f"(av_5));
                    float _sub_129;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_129) : "f"(acc_r[6]), "f"(bv_5));
                    float e_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_5) : "f"(_sub_128), "f"(_sub_129));
                    fold[group * 128 + col + 6] = s_5;
                    float _add_130;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_5));
                    fold[(16 + group) * 128 + col + 6] = _add_130;
                    acc_r[6] = 0.0f;
                    float s_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_6) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                    float bv_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_6) : "f"(s_6), "f"(fold[group * 128 + col + 7]));
                    float av_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_6) : "f"(s_6), "f"(bv_6));
                    float _sub_131;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_131) : "f"(fold[group * 128 + col + 7]), "f"(av_6));
                    float _sub_132;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_132) : "f"(acc_r[7]), "f"(bv_6));
                    float e_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_6) : "f"(_sub_131), "f"(_sub_132));
                    fold[group * 128 + col + 7] = s_6;
                    float _add_133;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_6));
                    fold[(16 + group) * 128 + col + 7] = _add_133;
                    acc_r[7] = 0.0f;
                } else {
                    fold[group * 128 + col] = acc_r[0];
                    fold[(16 + group) * 128 + col] = 0.0f;
                    acc_r[0] = 0.0f;
                    fold[group * 128 + col + 1] = acc_r[1];
                    fold[(16 + group) * 128 + col + 1] = 0.0f;
                    acc_r[1] = 0.0f;
                    fold[group * 128 + col + 2] = acc_r[2];
                    fold[(16 + group) * 128 + col + 2] = 0.0f;
                    acc_r[2] = 0.0f;
                    fold[group * 128 + col + 3] = acc_r[3];
                    fold[(16 + group) * 128 + col + 3] = 0.0f;
                    acc_r[3] = 0.0f;
                    fold[group * 128 + col + 4] = acc_r[4];
                    fold[(16 + group) * 128 + col + 4] = 0.0f;
                    acc_r[4] = 0.0f;
                    fold[group * 128 + col + 5] = acc_r[5];
                    fold[(16 + group) * 128 + col + 5] = 0.0f;
                    acc_r[5] = 0.0f;
                    fold[group * 128 + col + 6] = acc_r[6];
                    fold[(16 + group) * 128 + col + 6] = 0.0f;
                    acc_r[6] = 0.0f;
                    fold[group * 128 + col + 7] = acc_r[7];
                    fold[(16 + group) * 128 + col + 7] = 0.0f;
                    acc_r[7] = 0.0f;
                }
                folded = 1;
                run = 0;
            }
        }
        if (run != 0) {
            if (folded != 0) {
                float s_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                float bv_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7) : "f"(s_7), "f"(fold[group * 128 + col]));
                float av_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7) : "f"(s_7), "f"(bv_7));
                float _sub_134;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_134) : "f"(fold[group * 128 + col]), "f"(av_7));
                float _sub_135;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_135) : "f"(acc_r[0]), "f"(bv_7));
                float e_7;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7) : "f"(_sub_134), "f"(_sub_135));
                fold[group * 128 + col] = s_7;
                float _add_136;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(fold[(16 + group) * 128 + col]), "f"(e_7));
                fold[(16 + group) * 128 + col] = _add_136;
                acc_r[0] = 0.0f;
                float s_7_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7_1) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                float bv_7_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7_1) : "f"(s_7_1), "f"(fold[group * 128 + col + 1]));
                float av_7_1;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7_1) : "f"(s_7_1), "f"(bv_7_1));
                float _sub_137;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_137) : "f"(fold[group * 128 + col + 1]), "f"(av_7_1));
                float _sub_138;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_138) : "f"(acc_r[1]), "f"(bv_7_1));
                float e_7_1;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7_1) : "f"(_sub_137), "f"(_sub_138));
                fold[group * 128 + col + 1] = s_7_1;
                float _add_139;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_7_1));
                fold[(16 + group) * 128 + col + 1] = _add_139;
                acc_r[1] = 0.0f;
                float s_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_8) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                float bv_8;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_8) : "f"(s_8), "f"(fold[group * 128 + col + 2]));
                float av_8;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_8) : "f"(s_8), "f"(bv_8));
                float _sub_140;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_140) : "f"(fold[group * 128 + col + 2]), "f"(av_8));
                float _sub_141;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_141) : "f"(acc_r[2]), "f"(bv_8));
                float e_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_8) : "f"(_sub_140), "f"(_sub_141));
                fold[group * 128 + col + 2] = s_8;
                float _add_142;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_8));
                fold[(16 + group) * 128 + col + 2] = _add_142;
                acc_r[2] = 0.0f;
                float s_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                float bv_9;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_9) : "f"(s_9), "f"(fold[group * 128 + col + 3]));
                float av_9;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_9) : "f"(s_9), "f"(bv_9));
                float _sub_143;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_143) : "f"(fold[group * 128 + col + 3]), "f"(av_9));
                float _sub_144;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_144) : "f"(acc_r[3]), "f"(bv_9));
                float e_9;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_9) : "f"(_sub_143), "f"(_sub_144));
                fold[group * 128 + col + 3] = s_9;
                float _add_145;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_9));
                fold[(16 + group) * 128 + col + 3] = _add_145;
                acc_r[3] = 0.0f;
                float s_10;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                float bv_10;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_10) : "f"(s_10), "f"(fold[group * 128 + col + 4]));
                float av_10;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_10) : "f"(s_10), "f"(bv_10));
                float _sub_146;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_146) : "f"(fold[group * 128 + col + 4]), "f"(av_10));
                float _sub_147;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_147) : "f"(acc_r[4]), "f"(bv_10));
                float e_10;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_10) : "f"(_sub_146), "f"(_sub_147));
                fold[group * 128 + col + 4] = s_10;
                float _add_148;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_10));
                fold[(16 + group) * 128 + col + 4] = _add_148;
                acc_r[4] = 0.0f;
                float s_11;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                float bv_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_11) : "f"(s_11), "f"(fold[group * 128 + col + 5]));
                float av_11;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_11) : "f"(s_11), "f"(bv_11));
                float _sub_149;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_149) : "f"(fold[group * 128 + col + 5]), "f"(av_11));
                float _sub_150;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_150) : "f"(acc_r[5]), "f"(bv_11));
                float e_11;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_11) : "f"(_sub_149), "f"(_sub_150));
                fold[group * 128 + col + 5] = s_11;
                float _add_151;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_11));
                fold[(16 + group) * 128 + col + 5] = _add_151;
                acc_r[5] = 0.0f;
                float s_12;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                float bv_12;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_12) : "f"(s_12), "f"(fold[group * 128 + col + 6]));
                float av_12;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_12) : "f"(s_12), "f"(bv_12));
                float _sub_152;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_152) : "f"(fold[group * 128 + col + 6]), "f"(av_12));
                float _sub_153;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_153) : "f"(acc_r[6]), "f"(bv_12));
                float e_12;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_12) : "f"(_sub_152), "f"(_sub_153));
                fold[group * 128 + col + 6] = s_12;
                float _add_154;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_12));
                fold[(16 + group) * 128 + col + 6] = _add_154;
                acc_r[6] = 0.0f;
                float s_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                float bv_13;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_13) : "f"(s_13), "f"(fold[group * 128 + col + 7]));
                float av_13;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_13) : "f"(s_13), "f"(bv_13));
                float _sub_155;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_155) : "f"(fold[group * 128 + col + 7]), "f"(av_13));
                float _sub_156;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_156) : "f"(acc_r[7]), "f"(bv_13));
                float e_13;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_13) : "f"(_sub_155), "f"(_sub_156));
                fold[group * 128 + col + 7] = s_13;
                float _add_157;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_13));
                fold[(16 + group) * 128 + col + 7] = _add_157;
                acc_r[7] = 0.0f;
            } else {
                fold[group * 128 + col] = acc_r[0];
                fold[(16 + group) * 128 + col] = 0.0f;
                acc_r[0] = 0.0f;
                fold[group * 128 + col + 1] = acc_r[1];
                fold[(16 + group) * 128 + col + 1] = 0.0f;
                acc_r[1] = 0.0f;
                fold[group * 128 + col + 2] = acc_r[2];
                fold[(16 + group) * 128 + col + 2] = 0.0f;
                acc_r[2] = 0.0f;
                fold[group * 128 + col + 3] = acc_r[3];
                fold[(16 + group) * 128 + col + 3] = 0.0f;
                acc_r[3] = 0.0f;
                fold[group * 128 + col + 4] = acc_r[4];
                fold[(16 + group) * 128 + col + 4] = 0.0f;
                acc_r[4] = 0.0f;
                fold[group * 128 + col + 5] = acc_r[5];
                fold[(16 + group) * 128 + col + 5] = 0.0f;
                acc_r[5] = 0.0f;
                fold[group * 128 + col + 6] = acc_r[6];
                fold[(16 + group) * 128 + col + 6] = 0.0f;
                acc_r[6] = 0.0f;
                fold[group * 128 + col + 7] = acc_r[7];
                fold[(16 + group) * 128 + col + 7] = 0.0f;
                acc_r[7] = 0.0f;
            }
            folded = 1;
            run = 0;
        }
        __syncthreads();
        if (tid < 128) {
            float hs = 0.0f;
            float ls = 0.0f;
            #pragma unroll
            for (int g = 0; g < 16; g++) {
                float s_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(hs), "f"(fold[g * 128 + tid]));
                float bv_14;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14) : "f"(s_14), "f"(hs));
                float av_14;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14) : "f"(s_14), "f"(bv_14));
                float _sub_158;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_158) : "f"(hs), "f"(av_14));
                float _sub_159;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_159) : "f"(fold[g * 128 + tid]), "f"(bv_14));
                float e_14;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14) : "f"(_sub_158), "f"(_sub_159));
                hs = s_14;
                float _add_161;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(e_14), "f"(fold[(16 + g) * 128 + tid]));
                float _add_160;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(ls), "f"(_add_161));
                ls = _add_160;
            }
            unsigned int bits = __as_u32(hs);
            unsigned int ex = bits >> 23 & 255;
            unsigned int man = bits & 8388607;
            if (ex == 255) {
                unsigned int fbit = 4;
                if (man == 0) {
                    fbit = 2;
                    if (bits >> 31 == 0) {
                        fbit = 1;
                    }
                }
                exf[tid] = exf[tid] | fbit;
            } else {
                if (ex != 0) {
                    man = man | 8388608;
                }
                if (man != 0) {
                    unsigned int _max_162 = ((ex) > (1) ? (ex) : (1));
                    unsigned int pos = _max_162 - 1;
                    unsigned long long wide = (unsigned long long)man << (unsigned long long)(pos & 31);
                    unsigned long long d0 = wide & 4294967295ull;
                    unsigned long long d1 = wide >> 32ull;
                    if (bits >> 31 != 0) {
                        d0 = -d0;
                        d1 = -d1;
                    }
                    int at = (int)(pos >> 5) * 128 + tid;
                    exa[at] = exa[at] + d0;
                    if (d1 != 0ull) {
                        exa[at + 128] = exa[at + 128] + d1;
                    }
                }
            }
            float _abs_163;
            asm("abs.f32 %0, %1;" : "=f"(_abs_163) : "f"(hs));
            if (_abs_163 <= 3.4028235e+38f) {
                unsigned int bits_0 = __as_u32(ls);
                unsigned int ex_0 = bits_0 >> 23 & 255;
                unsigned int man_0 = bits_0 & 8388607;
                if (ex_0 == 255) {
                    unsigned int fbit_1 = 4;
                    if (man_0 == 0) {
                        fbit_1 = 2;
                        if (bits_0 >> 31 == 0) {
                            fbit_1 = 1;
                        }
                    }
                    exf[tid] = exf[tid] | fbit_1;
                } else {
                    if (ex_0 != 0) {
                        man_0 = man_0 | 8388608;
                    }
                    if (man_0 != 0) {
                        unsigned int _max_164 = ((ex_0) > (1) ? (ex_0) : (1));
                        unsigned int pos_1 = _max_164 - 1;
                        unsigned long long wide_1 = (unsigned long long)man_0 << (unsigned long long)(pos_1 & 31);
                        unsigned long long d0_1 = wide_1 & 4294967295ull;
                        unsigned long long d1_1 = wide_1 >> 32ull;
                        if (bits_0 >> 31 != 0) {
                            d0_1 = -d0_1;
                            d1_1 = -d1_1;
                        }
                        int at_1 = (int)(pos_1 >> 5) * 128 + tid;
                        exa[at_1] = exa[at_1] + d0_1;
                        if (d1_1 != 0ull) {
                            exa[at_1 + 128] = exa[at_1 + 128] + d1_1;
                        }
                    }
                }
            }
        }
        folded = 0;
    }
    if (rest > 0) {
        {
            const uint4* _vptr_165 = reinterpret_cast<const uint4*>(w + col + 0);
            uint4 _vld_165[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_vld_165[_blk].x), "=r"(_vld_165[_blk].y), "=r"(_vld_165[_blk].z), "=r"(_vld_165[_blk].w) : "l"((const void*)(_vptr_165 + _blk)) : "memory");
                uint32_t* _vpairs_165 = reinterpret_cast<uint32_t*>(&_vld_165[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&wclaim[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_165[_pair]) << 16);
                    (&wclaim[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_165[_pair]) & 0xffff0000u);
                }
            }
        }
    }
    int trips = 0;
    if (rest > 0) {
        int _min_166 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_166) > (1) ? (_min_166) : (1));
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
                float gr_1[8];
                float xr_1[8];
                float zr_1[8];
                {
                    const uint4* _vptr_167 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                    uint4 _vld_167[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_167[_blk] = _vptr_167[_blk];
                        uint32_t* _vpairs_167 = reinterpret_cast<uint32_t*>(&_vld_167[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_167[_pair]) << 16);
                            (&gr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_167[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_168 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                    uint4 _vld_168[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_168[_blk] = _vptr_168[_blk];
                        uint32_t* _vpairs_168 = reinterpret_cast<uint32_t*>(&_vld_168[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_168[_pair]) << 16);
                            (&xr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_168[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_169 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                    uint4 _vld_169[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_169[_blk] = _vptr_169[_blk];
                        uint32_t* _vpairs_169 = reinterpret_cast<uint32_t*>(&_vld_169[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_169[_pair]) << 16);
                            (&zr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_169[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (i_1 + 1 < chunk_n) {
                    if (packed != 0) {
                        if (tid == 0) {
                            int nfirst_1 = (u_1 + 1) * 16;
                            int _min_170 = ((rows - nfirst_1) < (16) ? (rows - nfirst_1) : (16));
                            int nbytes_1 = _min_170 * 256;
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
                        uint32_t _atomic_fetch_inc_171;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_171) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        ticket = (int)_atomic_fetch_inc_171;
                    }
                }
                float total_1 = 0.0f;
                total_1 = __fmaf_rn(xr_1[0], xr_1[0], total_1);
                total_1 = __fmaf_rn(xr_1[1], xr_1[1], total_1);
                total_1 = __fmaf_rn(xr_1[2], xr_1[2], total_1);
                total_1 = __fmaf_rn(xr_1[3], xr_1[3], total_1);
                total_1 = __fmaf_rn(xr_1[4], xr_1[4], total_1);
                total_1 = __fmaf_rn(xr_1[5], xr_1[5], total_1);
                total_1 = __fmaf_rn(xr_1[6], xr_1[6], total_1);
                total_1 = __fmaf_rn(xr_1[7], xr_1[7], total_1);
                float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
                total_1 = total_1 + _shfl_xor_sync_172;
                float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
                total_1 = total_1 + _shfl_xor_sync_173;
                float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
                total_1 = total_1 + _shfl_xor_sync_174;
                float _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
                total_1 = total_1 + _shfl_xor_sync_175;
                float _sqrt_176;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_176) : "f"(total_1 * 0.0078125f + eps));
                float rstd_1 = __frcp_rn(_sqrt_176);
                float dot_1 = 0.0f;
                int fast_1 = 0;
                float _neg_177;
                asm("neg.f32 %0, %1;" : "=f"(_neg_177) : "f"(zr_1[0]));
                float _neg_178;
                asm("neg.f32 %0, %1;" : "=f"(_neg_178) : "f"(zr_1[1]));
                float _neg_179;
                asm("neg.f32 %0, %1;" : "=f"(_neg_179) : "f"(zr_1[2]));
                float mz_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_1) : "f"(_neg_177), "f"(_neg_178), "f"(_neg_179));
                float _abs_181;
                asm("abs.f32 %0, %1;" : "=f"(_abs_181) : "f"(zr_1[0]));
                float _neg_180;
                asm("neg.f32 %0, %1;" : "=f"(_neg_180) : "f"(_abs_181));
                float _abs_183;
                asm("abs.f32 %0, %1;" : "=f"(_abs_183) : "f"(zr_1[1]));
                float _neg_182;
                asm("neg.f32 %0, %1;" : "=f"(_neg_182) : "f"(_abs_183));
                float _abs_185;
                asm("abs.f32 %0, %1;" : "=f"(_abs_185) : "f"(zr_1[2]));
                float _neg_184;
                asm("neg.f32 %0, %1;" : "=f"(_neg_184) : "f"(_abs_185));
                float ma_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_1) : "f"(_neg_180), "f"(_neg_182), "f"(_neg_184));
                float _neg_187;
                asm("neg.f32 %0, %1;" : "=f"(_neg_187) : "f"(zr_1[3]));
                float _neg_188;
                asm("neg.f32 %0, %1;" : "=f"(_neg_188) : "f"(zr_1[4]));
                float _max3_186;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_186) : "f"(mz_1), "f"(_neg_187), "f"(_neg_188));
                mz_1 = _max3_186;
                float _abs_191;
                asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(zr_1[3]));
                float _neg_190;
                asm("neg.f32 %0, %1;" : "=f"(_neg_190) : "f"(_abs_191));
                float _abs_193;
                asm("abs.f32 %0, %1;" : "=f"(_abs_193) : "f"(zr_1[4]));
                float _neg_192;
                asm("neg.f32 %0, %1;" : "=f"(_neg_192) : "f"(_abs_193));
                float _max3_189;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_189) : "f"(ma_1), "f"(_neg_190), "f"(_neg_192));
                ma_1 = _max3_189;
                float _neg_195;
                asm("neg.f32 %0, %1;" : "=f"(_neg_195) : "f"(zr_1[5]));
                float _neg_196;
                asm("neg.f32 %0, %1;" : "=f"(_neg_196) : "f"(zr_1[6]));
                float _max3_194;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_194) : "f"(mz_1), "f"(_neg_195), "f"(_neg_196));
                mz_1 = _max3_194;
                float _abs_199;
                asm("abs.f32 %0, %1;" : "=f"(_abs_199) : "f"(zr_1[5]));
                float _neg_198;
                asm("neg.f32 %0, %1;" : "=f"(_neg_198) : "f"(_abs_199));
                float _abs_201;
                asm("abs.f32 %0, %1;" : "=f"(_abs_201) : "f"(zr_1[6]));
                float _neg_200;
                asm("neg.f32 %0, %1;" : "=f"(_neg_200) : "f"(_abs_201));
                float _max3_197;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_197) : "f"(ma_1), "f"(_neg_198), "f"(_neg_200));
                ma_1 = _max3_197;
                float _neg_203;
                asm("neg.f32 %0, %1;" : "=f"(_neg_203) : "f"(zr_1[7]));
                float _max_202;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_202) : "f"(mz_1), "f"(_neg_203));
                mz_1 = _max_202;
                float _abs_206;
                asm("abs.f32 %0, %1;" : "=f"(_abs_206) : "f"(zr_1[7]));
                float _neg_205;
                asm("neg.f32 %0, %1;" : "=f"(_neg_205) : "f"(_abs_206));
                float _max_204;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_204) : "f"(ma_1), "f"(_neg_205));
                ma_1 = _max_204;
                float _abs_207;
                asm("abs.f32 %0, %1;" : "=f"(_abs_207) : "f"(zr_1[0]));
                float _abs_208;
                asm("abs.f32 %0, %1;" : "=f"(_abs_208) : "f"(zr_1[1]));
                float _abs_209;
                asm("abs.f32 %0, %1;" : "=f"(_abs_209) : "f"(zr_1[2]));
                float mp_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_1) : "f"(_abs_207), "f"(_abs_208), "f"(_abs_209));
                float _abs_211;
                asm("abs.f32 %0, %1;" : "=f"(_abs_211) : "f"(zr_1[3]));
                float _abs_212;
                asm("abs.f32 %0, %1;" : "=f"(_abs_212) : "f"(zr_1[4]));
                float _max3_210;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_210) : "f"(mp_1), "f"(_abs_211), "f"(_abs_212));
                mp_1 = _max3_210;
                float _abs_214;
                asm("abs.f32 %0, %1;" : "=f"(_abs_214) : "f"(zr_1[5]));
                float _abs_215;
                asm("abs.f32 %0, %1;" : "=f"(_abs_215) : "f"(zr_1[6]));
                float _max3_213;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_213) : "f"(mp_1), "f"(_abs_214), "f"(_abs_215));
                mp_1 = _max3_213;
                float _abs_217;
                asm("abs.f32 %0, %1;" : "=f"(_abs_217) : "f"(zr_1[7]));
                float _max_216;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_216) : "f"(mp_1), "f"(_abs_217));
                mp_1 = _max_216;
                if (mz_1 <= 41.0f) {
                    if (ma_1 <= -1.1754944e-38f) {
                        if (mp_1 <= 3.4028235e+38f) {
                            fast_1 = 1;
                        }
                    }
                }
                if (fast_1 != 0) {
                    float2 rstd2_1 = make_float2(rstd_1, rstd_1);
                    float2 z2_4 = make_float2(zr_1[0], zr_1[1]);
                    float ta_4;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_4) : "f"(zr_1[0]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_4;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_4) : "f"(zr_1[1]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_218;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(ta_4), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_219;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(tb_4), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_4 = make_float2(_fma_218, _fma_219);
                    float2 r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_4));
                    r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.925963e-08f, -1.925963e-08f), r2_4);
                    float2 nscale_4 = make_float2(__uint_as_float((__as_u32(t2_4.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_4.y) << 23) + 2147483648u));
                    float _exp2_220 = approx_exp2(r2_4.x);
                    float _exp2_221 = approx_exp2(r2_4.y);
                    float2 nden_4 = fma_f32x2_rn_noftz(nscale_4, make_float2(_exp2_220, _exp2_221), make_float2(-1.0f, -1.0f));
                    float _neg_223;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_223) : "f"(nden_4.x));
                    float _rcp_222 = approx_rcp(_neg_223);
                    float _neg_225;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_225) : "f"(nden_4.y));
                    float _rcp_224 = approx_rcp(_neg_225);
                    float2 r0_4 = make_float2(_rcp_222, _rcp_224);
                    float2 r1_4 = fma_f32x2_rn_noftz(r0_4, fma_f32x2_rn_noftz(nden_4, r0_4, make_float2(1.0f, 1.0f)), r0_4);
                    float2 sig2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, r1_4, make_float2(1.0f, 1.0f)), r1_4);
                    float2 q0_4 = fma_f32x2_rn_noftz(z2_4, r1_4, make_float2(0.0f, 0.0f));
                    float2 gate2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, q0_4, z2_4), q0_4);
                    float2 g2_4 = make_float2(gr_1[0], gr_1[1]);
                    float2 w2_4 = make_float2(wclaim[0], wclaim[1]);
                    float2 _f32x2_mul_a_226 = make_float2(xr_1[0], xr_1[1]);
                    float2 n2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_227 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
                    float2 _f32x2_mul_b_228 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
                    float2 dz2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_227), "l"(*(const unsigned long long*)&_f32x2_mul_b_228));
                    float2 a2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&gate2_4));
                    float2 dn2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_4) : "l"(*(const unsigned long long*)&a2_4), "l"(*(const unsigned long long*)&w2_4));
                    float2 acc2_4 = fma_f32x2_rn_noftz(a2_4, n2_4, make_float2(acc_r[0], acc_r[1]));
                    acc_r[0] = ((row_1 < rows) ? acc2_4.x : acc_r[0]);
                    acc_r[1] = ((row_1 < rows) ? acc2_4.y : acc_r[1]);
                    zr_1[0] = dz2_4.x;
                    zr_1[1] = dz2_4.y;
                    gr_1[0] = dn2_4.x;
                    gr_1[1] = dn2_4.y;
                    float2 z2_5 = make_float2(zr_1[2], zr_1[3]);
                    float ta_5;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_5) : "f"(zr_1[2]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_5;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_5) : "f"(zr_1[3]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_229;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(ta_5), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_230;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(tb_5), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_5 = make_float2(_fma_229, _fma_230);
                    float2 r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_5));
                    r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.925963e-08f, -1.925963e-08f), r2_5);
                    float2 nscale_5 = make_float2(__uint_as_float((__as_u32(t2_5.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_5.y) << 23) + 2147483648u));
                    float _exp2_231 = approx_exp2(r2_5.x);
                    float _exp2_232 = approx_exp2(r2_5.y);
                    float2 nden_5 = fma_f32x2_rn_noftz(nscale_5, make_float2(_exp2_231, _exp2_232), make_float2(-1.0f, -1.0f));
                    float _neg_234;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_234) : "f"(nden_5.x));
                    float _rcp_233 = approx_rcp(_neg_234);
                    float _neg_236;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_236) : "f"(nden_5.y));
                    float _rcp_235 = approx_rcp(_neg_236);
                    float2 r0_5 = make_float2(_rcp_233, _rcp_235);
                    float2 r1_5 = fma_f32x2_rn_noftz(r0_5, fma_f32x2_rn_noftz(nden_5, r0_5, make_float2(1.0f, 1.0f)), r0_5);
                    float2 sig2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, r1_5, make_float2(1.0f, 1.0f)), r1_5);
                    float2 q0_5 = fma_f32x2_rn_noftz(z2_5, r1_5, make_float2(0.0f, 0.0f));
                    float2 gate2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, q0_5, z2_5), q0_5);
                    float2 g2_5 = make_float2(gr_1[2], gr_1[3]);
                    float2 w2_5 = make_float2(wclaim[2], wclaim[3]);
                    float2 _f32x2_mul_a_237 = make_float2(xr_1[2], xr_1[3]);
                    float2 n2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_237), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_238 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
                    float2 _f32x2_mul_b_239 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
                    float2 dz2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&_f32x2_mul_b_239));
                    float2 a2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&gate2_5));
                    float2 dn2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_5) : "l"(*(const unsigned long long*)&a2_5), "l"(*(const unsigned long long*)&w2_5));
                    float2 acc2_5 = fma_f32x2_rn_noftz(a2_5, n2_5, make_float2(acc_r[2], acc_r[3]));
                    acc_r[2] = ((row_1 < rows) ? acc2_5.x : acc_r[2]);
                    acc_r[3] = ((row_1 < rows) ? acc2_5.y : acc_r[3]);
                    zr_1[2] = dz2_5.x;
                    zr_1[3] = dz2_5.y;
                    gr_1[2] = dn2_5.x;
                    gr_1[3] = dn2_5.y;
                    float2 z2_6 = make_float2(zr_1[4], zr_1[5]);
                    float ta_6;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_6) : "f"(zr_1[4]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_6;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_6) : "f"(zr_1[5]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_240;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(ta_6), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_241;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(tb_6), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_6 = make_float2(_fma_240, _fma_241);
                    float2 r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_6));
                    r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.925963e-08f, -1.925963e-08f), r2_6);
                    float2 nscale_6 = make_float2(__uint_as_float((__as_u32(t2_6.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_6.y) << 23) + 2147483648u));
                    float _exp2_242 = approx_exp2(r2_6.x);
                    float _exp2_243 = approx_exp2(r2_6.y);
                    float2 nden_6 = fma_f32x2_rn_noftz(nscale_6, make_float2(_exp2_242, _exp2_243), make_float2(-1.0f, -1.0f));
                    float _neg_245;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_245) : "f"(nden_6.x));
                    float _rcp_244 = approx_rcp(_neg_245);
                    float _neg_247;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_247) : "f"(nden_6.y));
                    float _rcp_246 = approx_rcp(_neg_247);
                    float2 r0_6 = make_float2(_rcp_244, _rcp_246);
                    float2 r1_6 = fma_f32x2_rn_noftz(r0_6, fma_f32x2_rn_noftz(nden_6, r0_6, make_float2(1.0f, 1.0f)), r0_6);
                    float2 sig2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, r1_6, make_float2(1.0f, 1.0f)), r1_6);
                    float2 q0_6 = fma_f32x2_rn_noftz(z2_6, r1_6, make_float2(0.0f, 0.0f));
                    float2 gate2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, q0_6, z2_6), q0_6);
                    float2 g2_6 = make_float2(gr_1[4], gr_1[5]);
                    float2 w2_6 = make_float2(wclaim[4], wclaim[5]);
                    float2 _f32x2_mul_a_248 = make_float2(xr_1[4], xr_1[5]);
                    float2 n2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_248), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_249 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
                    float2 _f32x2_mul_b_250 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
                    float2 dz2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_249), "l"(*(const unsigned long long*)&_f32x2_mul_b_250));
                    float2 a2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&gate2_6));
                    float2 dn2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_6) : "l"(*(const unsigned long long*)&a2_6), "l"(*(const unsigned long long*)&w2_6));
                    float2 acc2_6 = fma_f32x2_rn_noftz(a2_6, n2_6, make_float2(acc_r[4], acc_r[5]));
                    acc_r[4] = ((row_1 < rows) ? acc2_6.x : acc_r[4]);
                    acc_r[5] = ((row_1 < rows) ? acc2_6.y : acc_r[5]);
                    zr_1[4] = dz2_6.x;
                    zr_1[5] = dz2_6.y;
                    gr_1[4] = dn2_6.x;
                    gr_1[5] = dn2_6.y;
                    float2 z2_7 = make_float2(zr_1[6], zr_1[7]);
                    float ta_7;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_7) : "f"(zr_1[6]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_7;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_7) : "f"(zr_1[7]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_251;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(ta_7), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_252;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(tb_7), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_7 = make_float2(_fma_251, _fma_252);
                    float2 r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_7));
                    r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.925963e-08f, -1.925963e-08f), r2_7);
                    float2 nscale_7 = make_float2(__uint_as_float((__as_u32(t2_7.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_7.y) << 23) + 2147483648u));
                    float _exp2_253 = approx_exp2(r2_7.x);
                    float _exp2_254 = approx_exp2(r2_7.y);
                    float2 nden_7 = fma_f32x2_rn_noftz(nscale_7, make_float2(_exp2_253, _exp2_254), make_float2(-1.0f, -1.0f));
                    float _neg_256;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_256) : "f"(nden_7.x));
                    float _rcp_255 = approx_rcp(_neg_256);
                    float _neg_258;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_258) : "f"(nden_7.y));
                    float _rcp_257 = approx_rcp(_neg_258);
                    float2 r0_7 = make_float2(_rcp_255, _rcp_257);
                    float2 r1_7 = fma_f32x2_rn_noftz(r0_7, fma_f32x2_rn_noftz(nden_7, r0_7, make_float2(1.0f, 1.0f)), r0_7);
                    float2 sig2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, r1_7, make_float2(1.0f, 1.0f)), r1_7);
                    float2 q0_7 = fma_f32x2_rn_noftz(z2_7, r1_7, make_float2(0.0f, 0.0f));
                    float2 gate2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, q0_7, z2_7), q0_7);
                    float2 g2_7 = make_float2(gr_1[6], gr_1[7]);
                    float2 w2_7 = make_float2(wclaim[6], wclaim[7]);
                    float2 _f32x2_mul_a_259 = make_float2(xr_1[6], xr_1[7]);
                    float2 n2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_259), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_260 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
                    float2 _f32x2_mul_b_261 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
                    float2 dz2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_260), "l"(*(const unsigned long long*)&_f32x2_mul_b_261));
                    float2 a2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&gate2_7));
                    float2 dn2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_7) : "l"(*(const unsigned long long*)&a2_7), "l"(*(const unsigned long long*)&w2_7));
                    float2 acc2_7 = fma_f32x2_rn_noftz(a2_7, n2_7, make_float2(acc_r[6], acc_r[7]));
                    acc_r[6] = ((row_1 < rows) ? acc2_7.x : acc_r[6]);
                    acc_r[7] = ((row_1 < rows) ? acc2_7.y : acc_r[7]);
                    zr_1[6] = dz2_7.x;
                    zr_1[7] = dz2_7.y;
                    gr_1[6] = dn2_7.x;
                    gr_1[7] = dn2_7.y;
                    dot_1 = __fmaf_rn(gr_1[0], xr_1[0], dot_1);
                    dot_1 = __fmaf_rn(gr_1[1], xr_1[1], dot_1);
                    dot_1 = __fmaf_rn(gr_1[2], xr_1[2], dot_1);
                    dot_1 = __fmaf_rn(gr_1[3], xr_1[3], dot_1);
                    dot_1 = __fmaf_rn(gr_1[4], xr_1[4], dot_1);
                    dot_1 = __fmaf_rn(gr_1[5], xr_1[5], dot_1);
                    dot_1 = __fmaf_rn(gr_1[6], xr_1[6], dot_1);
                    dot_1 = __fmaf_rn(gr_1[7], xr_1[7], dot_1);
                } else {
                    float _exp_262 = __nv_expf(zr_1[0] * -1.0f);
                    float den_8 = 1.0f + _exp_262;
                    float sig_8 = __fdiv_rn(1.0f, den_8);
                    float gate_8 = __fdiv_rn(zr_1[0], den_8);
                    float nj_8 = xr_1[0] * rstd_1;
                    float aj_8 = gr_1[0] * gate_8;
                    float dzj_8 = gr_1[0] * (nj_8 * wclaim[0]) * sig_8 * __fmaf_rn(zr_1[0], 1.0f - sig_8, 1.0f);
                    float dnj_8 = aj_8 * wclaim[0];
                    dot_1 = __fmaf_rn(dnj_8, xr_1[0], dot_1);
                    acc_r[0] = ((row_1 < rows) ? __fmaf_rn(aj_8, nj_8, acc_r[0]) : acc_r[0]);
                    zr_1[0] = dzj_8;
                    gr_1[0] = dnj_8;
                    float _exp_263 = __nv_expf(zr_1[1] * -1.0f);
                    float den_9 = 1.0f + _exp_263;
                    float sig_9 = __fdiv_rn(1.0f, den_9);
                    float gate_9 = __fdiv_rn(zr_1[1], den_9);
                    float nj_9 = xr_1[1] * rstd_1;
                    float aj_9 = gr_1[1] * gate_9;
                    float dzj_9 = gr_1[1] * (nj_9 * wclaim[1]) * sig_9 * __fmaf_rn(zr_1[1], 1.0f - sig_9, 1.0f);
                    float dnj_9 = aj_9 * wclaim[1];
                    dot_1 = __fmaf_rn(dnj_9, xr_1[1], dot_1);
                    acc_r[1] = ((row_1 < rows) ? __fmaf_rn(aj_9, nj_9, acc_r[1]) : acc_r[1]);
                    zr_1[1] = dzj_9;
                    gr_1[1] = dnj_9;
                    float _exp_264 = __nv_expf(zr_1[2] * -1.0f);
                    float den_10 = 1.0f + _exp_264;
                    float sig_10 = __fdiv_rn(1.0f, den_10);
                    float gate_10 = __fdiv_rn(zr_1[2], den_10);
                    float nj_10 = xr_1[2] * rstd_1;
                    float aj_10 = gr_1[2] * gate_10;
                    float dzj_10 = gr_1[2] * (nj_10 * wclaim[2]) * sig_10 * __fmaf_rn(zr_1[2], 1.0f - sig_10, 1.0f);
                    float dnj_10 = aj_10 * wclaim[2];
                    dot_1 = __fmaf_rn(dnj_10, xr_1[2], dot_1);
                    acc_r[2] = ((row_1 < rows) ? __fmaf_rn(aj_10, nj_10, acc_r[2]) : acc_r[2]);
                    zr_1[2] = dzj_10;
                    gr_1[2] = dnj_10;
                    float _exp_265 = __nv_expf(zr_1[3] * -1.0f);
                    float den_11 = 1.0f + _exp_265;
                    float sig_11 = __fdiv_rn(1.0f, den_11);
                    float gate_11 = __fdiv_rn(zr_1[3], den_11);
                    float nj_11 = xr_1[3] * rstd_1;
                    float aj_11 = gr_1[3] * gate_11;
                    float dzj_11 = gr_1[3] * (nj_11 * wclaim[3]) * sig_11 * __fmaf_rn(zr_1[3], 1.0f - sig_11, 1.0f);
                    float dnj_11 = aj_11 * wclaim[3];
                    dot_1 = __fmaf_rn(dnj_11, xr_1[3], dot_1);
                    acc_r[3] = ((row_1 < rows) ? __fmaf_rn(aj_11, nj_11, acc_r[3]) : acc_r[3]);
                    zr_1[3] = dzj_11;
                    gr_1[3] = dnj_11;
                    float _exp_266 = __nv_expf(zr_1[4] * -1.0f);
                    float den_12 = 1.0f + _exp_266;
                    float sig_12 = __fdiv_rn(1.0f, den_12);
                    float gate_12 = __fdiv_rn(zr_1[4], den_12);
                    float nj_12 = xr_1[4] * rstd_1;
                    float aj_12 = gr_1[4] * gate_12;
                    float dzj_12 = gr_1[4] * (nj_12 * wclaim[4]) * sig_12 * __fmaf_rn(zr_1[4], 1.0f - sig_12, 1.0f);
                    float dnj_12 = aj_12 * wclaim[4];
                    dot_1 = __fmaf_rn(dnj_12, xr_1[4], dot_1);
                    acc_r[4] = ((row_1 < rows) ? __fmaf_rn(aj_12, nj_12, acc_r[4]) : acc_r[4]);
                    zr_1[4] = dzj_12;
                    gr_1[4] = dnj_12;
                    float _exp_267 = __nv_expf(zr_1[5] * -1.0f);
                    float den_13 = 1.0f + _exp_267;
                    float sig_13 = __fdiv_rn(1.0f, den_13);
                    float gate_13 = __fdiv_rn(zr_1[5], den_13);
                    float nj_13 = xr_1[5] * rstd_1;
                    float aj_13 = gr_1[5] * gate_13;
                    float dzj_13 = gr_1[5] * (nj_13 * wclaim[5]) * sig_13 * __fmaf_rn(zr_1[5], 1.0f - sig_13, 1.0f);
                    float dnj_13 = aj_13 * wclaim[5];
                    dot_1 = __fmaf_rn(dnj_13, xr_1[5], dot_1);
                    acc_r[5] = ((row_1 < rows) ? __fmaf_rn(aj_13, nj_13, acc_r[5]) : acc_r[5]);
                    zr_1[5] = dzj_13;
                    gr_1[5] = dnj_13;
                    float _exp_268 = __nv_expf(zr_1[6] * -1.0f);
                    float den_14 = 1.0f + _exp_268;
                    float sig_14 = __fdiv_rn(1.0f, den_14);
                    float gate_14 = __fdiv_rn(zr_1[6], den_14);
                    float nj_14 = xr_1[6] * rstd_1;
                    float aj_14 = gr_1[6] * gate_14;
                    float dzj_14 = gr_1[6] * (nj_14 * wclaim[6]) * sig_14 * __fmaf_rn(zr_1[6], 1.0f - sig_14, 1.0f);
                    float dnj_14 = aj_14 * wclaim[6];
                    dot_1 = __fmaf_rn(dnj_14, xr_1[6], dot_1);
                    acc_r[6] = ((row_1 < rows) ? __fmaf_rn(aj_14, nj_14, acc_r[6]) : acc_r[6]);
                    zr_1[6] = dzj_14;
                    gr_1[6] = dnj_14;
                    float _exp_269 = __nv_expf(zr_1[7] * -1.0f);
                    float den_15 = 1.0f + _exp_269;
                    float sig_15 = __fdiv_rn(1.0f, den_15);
                    float gate_15 = __fdiv_rn(zr_1[7], den_15);
                    float nj_15 = xr_1[7] * rstd_1;
                    float aj_15 = gr_1[7] * gate_15;
                    float dzj_15 = gr_1[7] * (nj_15 * wclaim[7]) * sig_15 * __fmaf_rn(zr_1[7], 1.0f - sig_15, 1.0f);
                    float dnj_15 = aj_15 * wclaim[7];
                    dot_1 = __fmaf_rn(dnj_15, xr_1[7], dot_1);
                    acc_r[7] = ((row_1 < rows) ? __fmaf_rn(aj_15, nj_15, acc_r[7]) : acc_r[7]);
                    zr_1[7] = dzj_15;
                    gr_1[7] = dnj_15;
                }
                float _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_270;
                float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_271;
                float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_272;
                float _shfl_xor_sync_273 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
                dot_1 = dot_1 + _shfl_xor_sync_273;
                float coef_1 = rstd_1 * rstd_1 * rstd_1 * (dot_1 * -0.0078125f);
                float2 coef2_1 = make_float2(coef_1, coef_1);
                float2 rstd2b_1 = make_float2(rstd_1, rstd_1);
                float2 dx2_4 = fma_f32x2_rn_noftz(make_float2(xr_1[0], xr_1[1]), coef2_1, mul_f32x2_noftz(make_float2(gr_1[0], gr_1[1]), rstd2b_1));
                xr_1[0] = dx2_4.x;
                xr_1[1] = dx2_4.y;
                float2 dx2_5 = fma_f32x2_rn_noftz(make_float2(xr_1[2], xr_1[3]), coef2_1, mul_f32x2_noftz(make_float2(gr_1[2], gr_1[3]), rstd2b_1));
                xr_1[2] = dx2_5.x;
                xr_1[3] = dx2_5.y;
                float2 dx2_6 = fma_f32x2_rn_noftz(make_float2(xr_1[4], xr_1[5]), coef2_1, mul_f32x2_noftz(make_float2(gr_1[4], gr_1[5]), rstd2b_1));
                xr_1[4] = dx2_6.x;
                xr_1[5] = dx2_6.y;
                float2 dx2_7 = fma_f32x2_rn_noftz(make_float2(xr_1[6], xr_1[7]), coef2_1, mul_f32x2_noftz(make_float2(gr_1[6], gr_1[7]), rstd2b_1));
                xr_1[6] = dx2_7.x;
                xr_1[7] = dx2_7.y;
                unsigned int packed_dx_1[4];
                unsigned int packed_dz_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 0], xr_1[_lp*2+1 + 0]));
                    packed_dx_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 0], zr_1[_lp*2+1 + 0]));
                    packed_dz_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 2], xr_1[_lp*2+1 + 2]));
                    packed_dx_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 2], zr_1[_lp*2+1 + 2]));
                    packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 4], xr_1[_lp*2+1 + 4]));
                    packed_dx_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 4], zr_1[_lp*2+1 + 4]));
                    packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 6], xr_1[_lp*2+1 + 6]));
                    packed_dx_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 6], zr_1[_lp*2+1 + 6]));
                    packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row_1 < rows) {
                    long long orow_1 = (long long)row_1 * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                    reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
                }
                run = run + 1;
                if (run == 16) {
                    if (folded != 0) {
                        float s_15;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        float bv_15;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_15) : "f"(s_15), "f"(fold[group * 128 + col]));
                        float av_15;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_15) : "f"(s_15), "f"(bv_15));
                        float _sub_274;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_274) : "f"(fold[group * 128 + col]), "f"(av_15));
                        float _sub_275;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_275) : "f"(acc_r[0]), "f"(bv_15));
                        float e_15;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_15) : "f"(_sub_274), "f"(_sub_275));
                        fold[group * 128 + col] = s_15;
                        float _add_276;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(fold[(16 + group) * 128 + col]), "f"(e_15));
                        fold[(16 + group) * 128 + col] = _add_276;
                        acc_r[0] = 0.0f;
                        float s_14_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14_1) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        float bv_14_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14_1) : "f"(s_14_1), "f"(fold[group * 128 + col + 1]));
                        float av_14_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14_1) : "f"(s_14_1), "f"(bv_14_1));
                        float _sub_277;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_277) : "f"(fold[group * 128 + col + 1]), "f"(av_14_1));
                        float _sub_278;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_278) : "f"(acc_r[1]), "f"(bv_14_1));
                        float e_14_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14_1) : "f"(_sub_277), "f"(_sub_278));
                        fold[group * 128 + col + 1] = s_14_1;
                        float _add_279;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_279) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_14_1));
                        fold[(16 + group) * 128 + col + 1] = _add_279;
                        acc_r[1] = 0.0f;
                        float s_15_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15_1) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        float bv_15_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_15_1) : "f"(s_15_1), "f"(fold[group * 128 + col + 2]));
                        float av_15_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_15_1) : "f"(s_15_1), "f"(bv_15_1));
                        float _sub_280;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_280) : "f"(fold[group * 128 + col + 2]), "f"(av_15_1));
                        float _sub_281;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_281) : "f"(acc_r[2]), "f"(bv_15_1));
                        float e_15_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_15_1) : "f"(_sub_280), "f"(_sub_281));
                        fold[group * 128 + col + 2] = s_15_1;
                        float _add_282;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_15_1));
                        fold[(16 + group) * 128 + col + 2] = _add_282;
                        acc_r[2] = 0.0f;
                        float s_16;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        float bv_16;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_16) : "f"(s_16), "f"(fold[group * 128 + col + 3]));
                        float av_16;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_16) : "f"(s_16), "f"(bv_16));
                        float _sub_283;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_283) : "f"(fold[group * 128 + col + 3]), "f"(av_16));
                        float _sub_284;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_284) : "f"(acc_r[3]), "f"(bv_16));
                        float e_16;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_16) : "f"(_sub_283), "f"(_sub_284));
                        fold[group * 128 + col + 3] = s_16;
                        float _add_285;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_285) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_16));
                        fold[(16 + group) * 128 + col + 3] = _add_285;
                        acc_r[3] = 0.0f;
                        float s_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        float bv_17;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_17) : "f"(s_17), "f"(fold[group * 128 + col + 4]));
                        float av_17;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_17) : "f"(s_17), "f"(bv_17));
                        float _sub_286;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_286) : "f"(fold[group * 128 + col + 4]), "f"(av_17));
                        float _sub_287;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_287) : "f"(acc_r[4]), "f"(bv_17));
                        float e_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_17) : "f"(_sub_286), "f"(_sub_287));
                        fold[group * 128 + col + 4] = s_17;
                        float _add_288;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_17));
                        fold[(16 + group) * 128 + col + 4] = _add_288;
                        acc_r[4] = 0.0f;
                        float s_18;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        float bv_18;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_18) : "f"(s_18), "f"(fold[group * 128 + col + 5]));
                        float av_18;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_18) : "f"(s_18), "f"(bv_18));
                        float _sub_289;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_289) : "f"(fold[group * 128 + col + 5]), "f"(av_18));
                        float _sub_290;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_290) : "f"(acc_r[5]), "f"(bv_18));
                        float e_18;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_18) : "f"(_sub_289), "f"(_sub_290));
                        fold[group * 128 + col + 5] = s_18;
                        float _add_291;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_18));
                        fold[(16 + group) * 128 + col + 5] = _add_291;
                        acc_r[5] = 0.0f;
                        float s_19;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        float bv_19;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_19) : "f"(s_19), "f"(fold[group * 128 + col + 6]));
                        float av_19;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_19) : "f"(s_19), "f"(bv_19));
                        float _sub_292;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_292) : "f"(fold[group * 128 + col + 6]), "f"(av_19));
                        float _sub_293;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_293) : "f"(acc_r[6]), "f"(bv_19));
                        float e_19;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_19) : "f"(_sub_292), "f"(_sub_293));
                        fold[group * 128 + col + 6] = s_19;
                        float _add_294;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_19));
                        fold[(16 + group) * 128 + col + 6] = _add_294;
                        acc_r[6] = 0.0f;
                        float s_20;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        float bv_20;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_20) : "f"(s_20), "f"(fold[group * 128 + col + 7]));
                        float av_20;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_20) : "f"(s_20), "f"(bv_20));
                        float _sub_295;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_295) : "f"(fold[group * 128 + col + 7]), "f"(av_20));
                        float _sub_296;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_296) : "f"(acc_r[7]), "f"(bv_20));
                        float e_20;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_20) : "f"(_sub_295), "f"(_sub_296));
                        fold[group * 128 + col + 7] = s_20;
                        float _add_297;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_297) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_20));
                        fold[(16 + group) * 128 + col + 7] = _add_297;
                        acc_r[7] = 0.0f;
                    } else {
                        fold[group * 128 + col] = acc_r[0];
                        fold[(16 + group) * 128 + col] = 0.0f;
                        acc_r[0] = 0.0f;
                        fold[group * 128 + col + 1] = acc_r[1];
                        fold[(16 + group) * 128 + col + 1] = 0.0f;
                        acc_r[1] = 0.0f;
                        fold[group * 128 + col + 2] = acc_r[2];
                        fold[(16 + group) * 128 + col + 2] = 0.0f;
                        acc_r[2] = 0.0f;
                        fold[group * 128 + col + 3] = acc_r[3];
                        fold[(16 + group) * 128 + col + 3] = 0.0f;
                        acc_r[3] = 0.0f;
                        fold[group * 128 + col + 4] = acc_r[4];
                        fold[(16 + group) * 128 + col + 4] = 0.0f;
                        acc_r[4] = 0.0f;
                        fold[group * 128 + col + 5] = acc_r[5];
                        fold[(16 + group) * 128 + col + 5] = 0.0f;
                        acc_r[5] = 0.0f;
                        fold[group * 128 + col + 6] = acc_r[6];
                        fold[(16 + group) * 128 + col + 6] = 0.0f;
                        acc_r[6] = 0.0f;
                        fold[group * 128 + col + 7] = acc_r[7];
                        fold[(16 + group) * 128 + col + 7] = 0.0f;
                        acc_r[7] = 0.0f;
                    }
                    folded = 1;
                    run = 0;
                }
            }
            if (run != 0) {
                if (folded != 0) {
                    float s_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                    float bv_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_21) : "f"(s_21), "f"(fold[group * 128 + col]));
                    float av_21;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_21) : "f"(s_21), "f"(bv_21));
                    float _sub_298;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_298) : "f"(fold[group * 128 + col]), "f"(av_21));
                    float _sub_299;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_299) : "f"(acc_r[0]), "f"(bv_21));
                    float e_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_21) : "f"(_sub_298), "f"(_sub_299));
                    fold[group * 128 + col] = s_21;
                    float _add_300;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_300) : "f"(fold[(16 + group) * 128 + col]), "f"(e_21));
                    fold[(16 + group) * 128 + col] = _add_300;
                    acc_r[0] = 0.0f;
                    float s_21_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21_1) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                    float bv_21_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_21_1) : "f"(s_21_1), "f"(fold[group * 128 + col + 1]));
                    float av_21_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_21_1) : "f"(s_21_1), "f"(bv_21_1));
                    float _sub_301;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_301) : "f"(fold[group * 128 + col + 1]), "f"(av_21_1));
                    float _sub_302;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_302) : "f"(acc_r[1]), "f"(bv_21_1));
                    float e_21_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_21_1) : "f"(_sub_301), "f"(_sub_302));
                    fold[group * 128 + col + 1] = s_21_1;
                    float _add_303;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_303) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_21_1));
                    fold[(16 + group) * 128 + col + 1] = _add_303;
                    acc_r[1] = 0.0f;
                    float s_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_22) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                    float bv_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_22) : "f"(s_22), "f"(fold[group * 128 + col + 2]));
                    float av_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_22) : "f"(s_22), "f"(bv_22));
                    float _sub_304;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_304) : "f"(fold[group * 128 + col + 2]), "f"(av_22));
                    float _sub_305;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_305) : "f"(acc_r[2]), "f"(bv_22));
                    float e_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_22) : "f"(_sub_304), "f"(_sub_305));
                    fold[group * 128 + col + 2] = s_22;
                    float _add_306;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_306) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_22));
                    fold[(16 + group) * 128 + col + 2] = _add_306;
                    acc_r[2] = 0.0f;
                    float s_23;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_23) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                    float bv_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_23) : "f"(s_23), "f"(fold[group * 128 + col + 3]));
                    float av_23;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_23) : "f"(s_23), "f"(bv_23));
                    float _sub_307;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_307) : "f"(fold[group * 128 + col + 3]), "f"(av_23));
                    float _sub_308;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_308) : "f"(acc_r[3]), "f"(bv_23));
                    float e_23;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_23) : "f"(_sub_307), "f"(_sub_308));
                    fold[group * 128 + col + 3] = s_23;
                    float _add_309;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_309) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_23));
                    fold[(16 + group) * 128 + col + 3] = _add_309;
                    acc_r[3] = 0.0f;
                    float s_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_24) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                    float bv_24;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_24) : "f"(s_24), "f"(fold[group * 128 + col + 4]));
                    float av_24;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_24) : "f"(s_24), "f"(bv_24));
                    float _sub_310;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_310) : "f"(fold[group * 128 + col + 4]), "f"(av_24));
                    float _sub_311;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_311) : "f"(acc_r[4]), "f"(bv_24));
                    float e_24;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_24) : "f"(_sub_310), "f"(_sub_311));
                    fold[group * 128 + col + 4] = s_24;
                    float _add_312;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_312) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_24));
                    fold[(16 + group) * 128 + col + 4] = _add_312;
                    acc_r[4] = 0.0f;
                    float s_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_25) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                    float bv_25;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_25) : "f"(s_25), "f"(fold[group * 128 + col + 5]));
                    float av_25;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_25) : "f"(s_25), "f"(bv_25));
                    float _sub_313;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_313) : "f"(fold[group * 128 + col + 5]), "f"(av_25));
                    float _sub_314;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_314) : "f"(acc_r[5]), "f"(bv_25));
                    float e_25;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_25) : "f"(_sub_313), "f"(_sub_314));
                    fold[group * 128 + col + 5] = s_25;
                    float _add_315;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_315) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_25));
                    fold[(16 + group) * 128 + col + 5] = _add_315;
                    acc_r[5] = 0.0f;
                    float s_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_26) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                    float bv_26;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_26) : "f"(s_26), "f"(fold[group * 128 + col + 6]));
                    float av_26;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_26) : "f"(s_26), "f"(bv_26));
                    float _sub_316;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_316) : "f"(fold[group * 128 + col + 6]), "f"(av_26));
                    float _sub_317;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_317) : "f"(acc_r[6]), "f"(bv_26));
                    float e_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_26) : "f"(_sub_316), "f"(_sub_317));
                    fold[group * 128 + col + 6] = s_26;
                    float _add_318;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_318) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_26));
                    fold[(16 + group) * 128 + col + 6] = _add_318;
                    acc_r[6] = 0.0f;
                    float s_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_27) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                    float bv_27;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_27) : "f"(s_27), "f"(fold[group * 128 + col + 7]));
                    float av_27;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_27) : "f"(s_27), "f"(bv_27));
                    float _sub_319;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_319) : "f"(fold[group * 128 + col + 7]), "f"(av_27));
                    float _sub_320;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_320) : "f"(acc_r[7]), "f"(bv_27));
                    float e_27;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_27) : "f"(_sub_319), "f"(_sub_320));
                    fold[group * 128 + col + 7] = s_27;
                    float _add_321;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_321) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_27));
                    fold[(16 + group) * 128 + col + 7] = _add_321;
                    acc_r[7] = 0.0f;
                } else {
                    fold[group * 128 + col] = acc_r[0];
                    fold[(16 + group) * 128 + col] = 0.0f;
                    acc_r[0] = 0.0f;
                    fold[group * 128 + col + 1] = acc_r[1];
                    fold[(16 + group) * 128 + col + 1] = 0.0f;
                    acc_r[1] = 0.0f;
                    fold[group * 128 + col + 2] = acc_r[2];
                    fold[(16 + group) * 128 + col + 2] = 0.0f;
                    acc_r[2] = 0.0f;
                    fold[group * 128 + col + 3] = acc_r[3];
                    fold[(16 + group) * 128 + col + 3] = 0.0f;
                    acc_r[3] = 0.0f;
                    fold[group * 128 + col + 4] = acc_r[4];
                    fold[(16 + group) * 128 + col + 4] = 0.0f;
                    acc_r[4] = 0.0f;
                    fold[group * 128 + col + 5] = acc_r[5];
                    fold[(16 + group) * 128 + col + 5] = 0.0f;
                    acc_r[5] = 0.0f;
                    fold[group * 128 + col + 6] = acc_r[6];
                    fold[(16 + group) * 128 + col + 6] = 0.0f;
                    acc_r[6] = 0.0f;
                    fold[group * 128 + col + 7] = acc_r[7];
                    fold[(16 + group) * 128 + col + 7] = 0.0f;
                    acc_r[7] = 0.0f;
                }
                folded = 1;
                run = 0;
            }
        }
        __syncthreads();
        if (chunk_n > 0) {
            if (tid < 128) {
                float hs_1 = 0.0f;
                float ls_1 = 0.0f;
                #pragma unroll
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    float s_28;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_28) : "f"(hs_1), "f"(fold[g_1 * 128 + tid]));
                    float bv_28;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_28) : "f"(s_28), "f"(hs_1));
                    float av_28;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_28) : "f"(s_28), "f"(bv_28));
                    float _sub_322;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_322) : "f"(hs_1), "f"(av_28));
                    float _sub_323;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_323) : "f"(fold[g_1 * 128 + tid]), "f"(bv_28));
                    float e_28;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_28) : "f"(_sub_322), "f"(_sub_323));
                    hs_1 = s_28;
                    float _add_325;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_325) : "f"(e_28), "f"(fold[(16 + g_1) * 128 + tid]));
                    float _add_324;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_324) : "f"(ls_1), "f"(_add_325));
                    ls_1 = _add_324;
                }
                unsigned int bits_1 = __as_u32(hs_1);
                unsigned int ex_1 = bits_1 >> 23 & 255;
                unsigned int man_1 = bits_1 & 8388607;
                if (ex_1 == 255) {
                    unsigned int fbit_2 = 4;
                    if (man_1 == 0) {
                        fbit_2 = 2;
                        if (bits_1 >> 31 == 0) {
                            fbit_2 = 1;
                        }
                    }
                    exf[tid] = exf[tid] | fbit_2;
                } else {
                    if (ex_1 != 0) {
                        man_1 = man_1 | 8388608;
                    }
                    if (man_1 != 0) {
                        unsigned int _max_326 = ((ex_1) > (1) ? (ex_1) : (1));
                        unsigned int pos_2 = _max_326 - 1;
                        unsigned long long wide_2 = (unsigned long long)man_1 << (unsigned long long)(pos_2 & 31);
                        unsigned long long d0_2 = wide_2 & 4294967295ull;
                        unsigned long long d1_2 = wide_2 >> 32ull;
                        if (bits_1 >> 31 != 0) {
                            d0_2 = -d0_2;
                            d1_2 = -d1_2;
                        }
                        int at_2 = (int)(pos_2 >> 5) * 128 + tid;
                        exa[at_2] = exa[at_2] + d0_2;
                        if (d1_2 != 0ull) {
                            exa[at_2 + 128] = exa[at_2 + 128] + d1_2;
                        }
                    }
                }
                float _abs_327;
                asm("abs.f32 %0, %1;" : "=f"(_abs_327) : "f"(hs_1));
                if (_abs_327 <= 3.4028235e+38f) {
                    unsigned int bits_1_1 = __as_u32(ls_1);
                    unsigned int ex_1_1 = bits_1_1 >> 23 & 255;
                    unsigned int man_1_1 = bits_1_1 & 8388607;
                    if (ex_1_1 == 255) {
                        unsigned int fbit_3 = 4;
                        if (man_1_1 == 0) {
                            fbit_3 = 2;
                            if (bits_1_1 >> 31 == 0) {
                                fbit_3 = 1;
                            }
                        }
                        exf[tid] = exf[tid] | fbit_3;
                    } else {
                        if (ex_1_1 != 0) {
                            man_1_1 = man_1_1 | 8388608;
                        }
                        if (man_1_1 != 0) {
                            unsigned int _max_328 = ((ex_1_1) > (1) ? (ex_1_1) : (1));
                            unsigned int pos_3 = _max_328 - 1;
                            unsigned long long wide_3 = (unsigned long long)man_1_1 << (unsigned long long)(pos_3 & 31);
                            unsigned long long d0_3 = wide_3 & 4294967295ull;
                            unsigned long long d1_3 = wide_3 >> 32ull;
                            if (bits_1_1 >> 31 != 0) {
                                d0_3 = -d0_3;
                                d1_3 = -d1_3;
                            }
                            int at_3 = (int)(pos_3 >> 5) * 128 + tid;
                            exa[at_3] = exa[at_3] + d0_3;
                            if (d1_3 != 0ull) {
                                exa[at_3 + 128] = exa[at_3 + 128] + d1_3;
                            }
                        }
                    }
                }
            }
        }
        folded = 0;
    }
    unsigned long long dsum[9];
    __syncthreads();
    if (weight_grad != 0) {
        if (tid < 128) {
            int slot = bid & 1;
            unsigned long long dg_0 = exa[tid];
            if (dg_0 != 0ull) {
                atomicAdd(&acc[slot * 9 * 128 + tid], dg_0);
            }
            unsigned long long dg_1 = exa[128 + tid];
            if (dg_1 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 1) * 128 + tid], dg_1);
            }
            unsigned long long dg_2 = exa[256 + tid];
            if (dg_2 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 2) * 128 + tid], dg_2);
            }
            unsigned long long dg_3 = exa[384 + tid];
            if (dg_3 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 3) * 128 + tid], dg_3);
            }
            unsigned long long dg_4 = exa[512 + tid];
            if (dg_4 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 4) * 128 + tid], dg_4);
            }
            unsigned long long dg_5 = exa[640 + tid];
            if (dg_5 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 5) * 128 + tid], dg_5);
            }
            unsigned long long dg_6 = exa[768 + tid];
            if (dg_6 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 6) * 128 + tid], dg_6);
            }
            unsigned long long dg_7 = exa[896 + tid];
            if (dg_7 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 7) * 128 + tid], dg_7);
            }
            unsigned long long dg_8 = exa[1024 + tid];
            if (dg_8 != 0ull) {
                atomicAdd(&acc[(slot * 9 + 8) * 128 + tid], dg_8);
            }
            unsigned int fv = exf[tid];
            if (fv != 0) {
                atomicOr(&flags[tid], fv);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int is_last = 0;
        uint32_t arrival;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(arrival) : "l"(&done[0]), "r"(static_cast<uint32_t>((unsigned int)num_bids - 1)) : "memory");
        if (arrival == (unsigned int)num_bids - 1) {
            is_last = 1;
            done[1] = 0;
        }
        flag[0] = is_last;
    }
    __syncthreads();
    int flag_word[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&flag_word[0])) : "r"(flag_addr));
    if (flag_word[0] != 0) {
        if (weight_grad != 0) {
            if (tid < 128) {
                unsigned int flw[1];
                {
                    uint32_t _scalar_bits_329;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_329) : "l"((const void*)(flags + (tid))) : "memory");
                    flw[0] = (unsigned int)_scalar_bits_329;
                }
                unsigned long long dig[18];
                dig[0] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_330;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_330) : "l"((const void*)(acc + (tid))) : "memory");
                        dig[0] = (unsigned long long)_scalar_bits_330;
                    }
                }
                dig[1] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_331;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_331) : "l"((const void*)(acc + (1152 + tid))) : "memory");
                        dig[1] = (unsigned long long)_scalar_bits_331;
                    }
                }
                dig[2] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_332;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_332) : "l"((const void*)(acc + (128 + tid))) : "memory");
                        dig[2] = (unsigned long long)_scalar_bits_332;
                    }
                }
                dig[3] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_333;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_333) : "l"((const void*)(acc + (1280 + tid))) : "memory");
                        dig[3] = (unsigned long long)_scalar_bits_333;
                    }
                }
                dig[4] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_334;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_334) : "l"((const void*)(acc + (256 + tid))) : "memory");
                        dig[4] = (unsigned long long)_scalar_bits_334;
                    }
                }
                dig[5] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_335;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_335) : "l"((const void*)(acc + (1408 + tid))) : "memory");
                        dig[5] = (unsigned long long)_scalar_bits_335;
                    }
                }
                dig[6] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_336;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_336) : "l"((const void*)(acc + (384 + tid))) : "memory");
                        dig[6] = (unsigned long long)_scalar_bits_336;
                    }
                }
                dig[7] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_337;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_337) : "l"((const void*)(acc + (1536 + tid))) : "memory");
                        dig[7] = (unsigned long long)_scalar_bits_337;
                    }
                }
                dig[8] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_338;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_338) : "l"((const void*)(acc + (512 + tid))) : "memory");
                        dig[8] = (unsigned long long)_scalar_bits_338;
                    }
                }
                dig[9] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_339;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_339) : "l"((const void*)(acc + (1664 + tid))) : "memory");
                        dig[9] = (unsigned long long)_scalar_bits_339;
                    }
                }
                dig[10] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_340;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_340) : "l"((const void*)(acc + (640 + tid))) : "memory");
                        dig[10] = (unsigned long long)_scalar_bits_340;
                    }
                }
                dig[11] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_341;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_341) : "l"((const void*)(acc + (1792 + tid))) : "memory");
                        dig[11] = (unsigned long long)_scalar_bits_341;
                    }
                }
                dig[12] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_342;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_342) : "l"((const void*)(acc + (768 + tid))) : "memory");
                        dig[12] = (unsigned long long)_scalar_bits_342;
                    }
                }
                dig[13] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_343;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_343) : "l"((const void*)(acc + (1920 + tid))) : "memory");
                        dig[13] = (unsigned long long)_scalar_bits_343;
                    }
                }
                dig[14] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_344;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_344) : "l"((const void*)(acc + (896 + tid))) : "memory");
                        dig[14] = (unsigned long long)_scalar_bits_344;
                    }
                }
                dig[15] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_345;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_345) : "l"((const void*)(acc + (2048 + tid))) : "memory");
                        dig[15] = (unsigned long long)_scalar_bits_345;
                    }
                }
                dig[16] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_346;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_346) : "l"((const void*)(acc + (1024 + tid))) : "memory");
                        dig[16] = (unsigned long long)_scalar_bits_346;
                    }
                }
                dig[17] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_347;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_347) : "l"((const void*)(acc + (2176 + tid))) : "memory");
                        dig[17] = (unsigned long long)_scalar_bits_347;
                    }
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1152 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (128 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1280 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (256 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1408 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (384 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1536 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (512 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1664 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (640 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1792 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (768 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1920 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (896 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (2048 + tid)) = 0ull;
                }
                if (num_bids > 0) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (1024 + tid)) = 0ull;
                }
                if (num_bids > 1) {
                    *(reinterpret_cast<unsigned long long*>(acc) + (2176 + tid)) = 0ull;
                }
                unsigned long long digit_0 = dig[0];
                digit_0 = digit_0 + dig[1];
                dsum[0] = digit_0;
                unsigned long long digit_1 = dig[2];
                digit_1 = digit_1 + dig[3];
                dsum[1] = digit_1;
                unsigned long long digit_2 = dig[4];
                digit_2 = digit_2 + dig[5];
                dsum[2] = digit_2;
                unsigned long long digit_3 = dig[6];
                digit_3 = digit_3 + dig[7];
                dsum[3] = digit_3;
                unsigned long long digit_4 = dig[8];
                digit_4 = digit_4 + dig[9];
                dsum[4] = digit_4;
                unsigned long long digit_5 = dig[10];
                digit_5 = digit_5 + dig[11];
                dsum[5] = digit_5;
                unsigned long long digit_6 = dig[12];
                digit_6 = digit_6 + dig[13];
                dsum[6] = digit_6;
                unsigned long long digit_7 = dig[14];
                digit_7 = digit_7 + dig[15];
                dsum[7] = digit_7;
                unsigned long long digit_8 = dig[16];
                digit_8 = digit_8 + dig[17];
                dsum[8] = digit_8;
                *(reinterpret_cast<unsigned int*>(flags) + (tid)) = 0;
                unsigned int flv = flw[0];
                if (accumulate != 0) {
                    float pv = dw[tid];
                    unsigned int pbits = __as_u32(pv);
                    unsigned int pex = pbits >> 23 & 255;
                    unsigned int pman = pbits & 8388607;
                    if (pex == 255) {
                        unsigned int pf = 4;
                        if (pman == 0) {
                            pf = 2;
                            if (pbits >> 31 == 0) {
                                pf = 1;
                            }
                        }
                        flv = flv | pf;
                    } else {
                        if (pex != 0) {
                            pman = pman | 8388608;
                        }
                        if (pman != 0) {
                            unsigned int _max_348 = ((pex) > (1) ? (pex) : (1));
                            unsigned int ppos = _max_348 - 1;
                            unsigned long long pwide = (unsigned long long)pman << (unsigned long long)(ppos & 31);
                            unsigned long long p0 = pwide & 4294967295ull;
                            unsigned long long p1 = pwide >> 32ull;
                            if (pbits >> 31 != 0) {
                                p0 = -p0;
                                p1 = -p1;
                            }
                            int pk = (int)(ppos >> 5);
                            if (pk == 0) {
                                dsum[0] = dsum[0] + p0;
                            }
                            if (pk + 1 == 0) {
                                dsum[0] = dsum[0] + p1;
                            }
                            if (pk == 1) {
                                dsum[1] = dsum[1] + p0;
                            }
                            if (pk + 1 == 1) {
                                dsum[1] = dsum[1] + p1;
                            }
                            if (pk == 2) {
                                dsum[2] = dsum[2] + p0;
                            }
                            if (pk + 1 == 2) {
                                dsum[2] = dsum[2] + p1;
                            }
                            if (pk == 3) {
                                dsum[3] = dsum[3] + p0;
                            }
                            if (pk + 1 == 3) {
                                dsum[3] = dsum[3] + p1;
                            }
                            if (pk == 4) {
                                dsum[4] = dsum[4] + p0;
                            }
                            if (pk + 1 == 4) {
                                dsum[4] = dsum[4] + p1;
                            }
                            if (pk == 5) {
                                dsum[5] = dsum[5] + p0;
                            }
                            if (pk + 1 == 5) {
                                dsum[5] = dsum[5] + p1;
                            }
                            if (pk == 6) {
                                dsum[6] = dsum[6] + p0;
                            }
                            if (pk + 1 == 6) {
                                dsum[6] = dsum[6] + p1;
                            }
                            if (pk == 7) {
                                dsum[7] = dsum[7] + p0;
                            }
                            if (pk + 1 == 7) {
                                dsum[7] = dsum[7] + p1;
                            }
                            if (pk == 8) {
                                dsum[8] = dsum[8] + p0;
                            }
                            if (pk + 1 == 8) {
                                dsum[8] = dsum[8] + p1;
                            }
                        }
                    }
                }
                unsigned int words[10];
                long long carry = 0ll;
                unsigned long long t_0 = dsum[0] + (unsigned long long)carry;
                words[0] = (unsigned int)(t_0 & 4294967295ull);
                carry = (long long)t_0 >> 32ll;
                unsigned long long t_1 = dsum[1] + (unsigned long long)carry;
                words[1] = (unsigned int)(t_1 & 4294967295ull);
                carry = (long long)t_1 >> 32ll;
                unsigned long long t_2 = dsum[2] + (unsigned long long)carry;
                words[2] = (unsigned int)(t_2 & 4294967295ull);
                carry = (long long)t_2 >> 32ll;
                unsigned long long t_3 = dsum[3] + (unsigned long long)carry;
                words[3] = (unsigned int)(t_3 & 4294967295ull);
                carry = (long long)t_3 >> 32ll;
                unsigned long long t_4 = dsum[4] + (unsigned long long)carry;
                words[4] = (unsigned int)(t_4 & 4294967295ull);
                carry = (long long)t_4 >> 32ll;
                unsigned long long t_5 = dsum[5] + (unsigned long long)carry;
                words[5] = (unsigned int)(t_5 & 4294967295ull);
                carry = (long long)t_5 >> 32ll;
                unsigned long long t_6 = dsum[6] + (unsigned long long)carry;
                words[6] = (unsigned int)(t_6 & 4294967295ull);
                carry = (long long)t_6 >> 32ll;
                unsigned long long t_7 = dsum[7] + (unsigned long long)carry;
                words[7] = (unsigned int)(t_7 & 4294967295ull);
                carry = (long long)t_7 >> 32ll;
                long long top = (long long)dsum[8] + carry;
                int neg = 0;
                if (top < 0ll) {
                    neg = 1;
                    unsigned long long c1 = 1ull;
                    unsigned long long tn_0 = (unsigned long long)(words[0] ^ 4294967295u) + c1;
                    words[0] = (unsigned int)(tn_0 & 4294967295ull);
                    c1 = tn_0 >> 32ull;
                    unsigned long long tn_1 = (unsigned long long)(words[1] ^ 4294967295u) + c1;
                    words[1] = (unsigned int)(tn_1 & 4294967295ull);
                    c1 = tn_1 >> 32ull;
                    unsigned long long tn_2 = (unsigned long long)(words[2] ^ 4294967295u) + c1;
                    words[2] = (unsigned int)(tn_2 & 4294967295ull);
                    c1 = tn_2 >> 32ull;
                    unsigned long long tn_3 = (unsigned long long)(words[3] ^ 4294967295u) + c1;
                    words[3] = (unsigned int)(tn_3 & 4294967295ull);
                    c1 = tn_3 >> 32ull;
                    unsigned long long tn_4 = (unsigned long long)(words[4] ^ 4294967295u) + c1;
                    words[4] = (unsigned int)(tn_4 & 4294967295ull);
                    c1 = tn_4 >> 32ull;
                    unsigned long long tn_5 = (unsigned long long)(words[5] ^ 4294967295u) + c1;
                    words[5] = (unsigned int)(tn_5 & 4294967295ull);
                    c1 = tn_5 >> 32ull;
                    unsigned long long tn_6 = (unsigned long long)(words[6] ^ 4294967295u) + c1;
                    words[6] = (unsigned int)(tn_6 & 4294967295ull);
                    c1 = tn_6 >> 32ull;
                    unsigned long long tn_7 = (unsigned long long)(words[7] ^ 4294967295u) + c1;
                    words[7] = (unsigned int)(tn_7 & 4294967295ull);
                    c1 = tn_7 >> 32ull;
                    top = -1ll - top + (long long)c1;
                }
                words[8] = (unsigned int)((unsigned long long)top & 4294967295ull);
                words[9] = (unsigned int)((unsigned long long)top >> 32ull);
                int tw = -1;
                if (words[0] != 0) {
                    tw = 0;
                }
                if (words[1] != 0) {
                    tw = 1;
                }
                if (words[2] != 0) {
                    tw = 2;
                }
                if (words[3] != 0) {
                    tw = 3;
                }
                if (words[4] != 0) {
                    tw = 4;
                }
                if (words[5] != 0) {
                    tw = 5;
                }
                if (words[6] != 0) {
                    tw = 6;
                }
                if (words[7] != 0) {
                    tw = 7;
                }
                if (words[8] != 0) {
                    tw = 8;
                }
                if (words[9] != 0) {
                    tw = 9;
                }
                unsigned int wa = 0;
                unsigned int wb = 0;
                unsigned int wc = 0;
                unsigned int lowor = 0;
                if (tw == 0) {
                    wa = words[0];
                }
                if (tw == 1) {
                    wb = words[0];
                }
                if (tw == 2) {
                    wc = words[0];
                }
                if (tw > 2) {
                    lowor = lowor | words[0];
                }
                if (tw == 1) {
                    wa = words[1];
                }
                if (tw == 2) {
                    wb = words[1];
                }
                if (tw == 3) {
                    wc = words[1];
                }
                if (tw > 3) {
                    lowor = lowor | words[1];
                }
                if (tw == 2) {
                    wa = words[2];
                }
                if (tw == 3) {
                    wb = words[2];
                }
                if (tw == 4) {
                    wc = words[2];
                }
                if (tw > 4) {
                    lowor = lowor | words[2];
                }
                if (tw == 3) {
                    wa = words[3];
                }
                if (tw == 4) {
                    wb = words[3];
                }
                if (tw == 5) {
                    wc = words[3];
                }
                if (tw > 5) {
                    lowor = lowor | words[3];
                }
                if (tw == 4) {
                    wa = words[4];
                }
                if (tw == 5) {
                    wb = words[4];
                }
                if (tw == 6) {
                    wc = words[4];
                }
                if (tw > 6) {
                    lowor = lowor | words[4];
                }
                if (tw == 5) {
                    wa = words[5];
                }
                if (tw == 6) {
                    wb = words[5];
                }
                if (tw == 7) {
                    wc = words[5];
                }
                if (tw > 7) {
                    lowor = lowor | words[5];
                }
                if (tw == 6) {
                    wa = words[6];
                }
                if (tw == 7) {
                    wb = words[6];
                }
                if (tw == 8) {
                    wc = words[6];
                }
                if (tw > 8) {
                    lowor = lowor | words[6];
                }
                if (tw == 7) {
                    wa = words[7];
                }
                if (tw == 8) {
                    wb = words[7];
                }
                if (tw == 9) {
                    wc = words[7];
                }
                if (tw > 9) {
                    lowor = lowor | words[7];
                }
                if (tw == 8) {
                    wa = words[8];
                }
                if (tw == 9) {
                    wb = words[8];
                }
                if (tw == 10) {
                    wc = words[8];
                }
                if (tw > 10) {
                    lowor = lowor | words[8];
                }
                if (tw == 9) {
                    wa = words[9];
                }
                if (tw == 10) {
                    wb = words[9];
                }
                if (tw == 11) {
                    wc = words[9];
                }
                if (tw > 11) {
                    lowor = lowor | words[9];
                }
                unsigned int rbits = 0;
                if (tw >= 0) {
                    int lz = __clz((int)wa);
                    int bp = 32 * tw + 31 - lz;
                    if (bp <= 23) {
                        rbits = wa;
                    } else {
                        unsigned long long wide_c = (unsigned long long)wc << (unsigned long long)lz;
                        unsigned long long hi = ((unsigned long long)wa << 32ull | (unsigned long long)wb) << (unsigned long long)lz;
                        hi = hi | wide_c >> 32ull;
                        unsigned int mant = (unsigned int)(hi >> 40ull);
                        unsigned int sticky = 0;
                        if ((hi & 549755813887ull) != 0ull) {
                            sticky = 1;
                        }
                        if ((wide_c & 4294967295ull) != 0ull) {
                            sticky = 1;
                        }
                        if (lowor != 0) {
                            sticky = 1;
                        }
                        rbits = ((unsigned int)(bp - 23) << 23) + mant;
                        if ((hi >> 39ull & 1ull) != 0ull) {
                            if ((sticky | mant & 1) != 0) {
                                rbits = rbits + 1;
                            }
                        }
                        if (bp >= 277) {
                            rbits = 2139095040;
                        }
                    }
                    if (neg != 0) {
                        rbits = rbits | 2147483648u;
                    }
                }
                if (flv != 0) {
                    unsigned int special = 4286578688u;
                    if ((flv & 1) != 0) {
                        special = 2139095040;
                    }
                    if ((flv & 4) != 0) {
                        special = 2143289344;
                    }
                    if ((flv & 3) == 3) {
                        special = 2143289344;
                    }
                    rbits = special;
                }
                float res = __uint_as_float(rbits);
                dw[tid] = res;
            }
        }
    }
}

} // extern "C"
