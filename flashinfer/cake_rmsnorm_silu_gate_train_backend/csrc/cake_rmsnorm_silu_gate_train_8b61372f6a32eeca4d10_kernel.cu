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

#define CAKE_INF CUDART_INF_F
#define SMEM_FOLD_OFF 0
#define SMEM_FOLD_STAGE_BYTES 8192
#define SMEM_FOLD_STRIDE 8192
#define SMEM_CELL_OFF 8192
#define SMEM_CELL_STAGE_BYTES 16
#define SMEM_CELL_STRIDE 16
#define SMEM_FLAG_OFF 8208
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TAB_OFF 8224
#define SMEM_TAB_STAGE_BYTES 128
#define SMEM_TAB_STRIDE 128
#define SMEM_FOLD32_OFF 8352
#define SMEM_FOLD32_STAGE_BYTES 2048
#define SMEM_FOLD32_STRIDE 2048
#define SMEM_TOTAL 10496
#define THREADS 128
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
kernel_cake_rmsnorm_silu_gate_train_8b61372f6a32eeca4d10(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, double* __restrict__ dw_sums, unsigned int* __restrict__ dw_done, __nv_bfloat16* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
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

    // Kernel setup ops
    double* fold = reinterpret_cast<double*>(smem_raw + 0);
    const int fold_addr = smem + 0;
    int* cell = reinterpret_cast<int*>(smem_raw + 8192);
    const int cell_addr = smem + 8192;
    int* flag = reinterpret_cast<int*>(smem_raw + 8208);
    const int flag_addr = smem + 8208;
    int* tab = reinterpret_cast<int*>(smem_raw + 8224);
    const int tab_addr = smem + 8224;
    float* fold32 = reinterpret_cast<float*>(smem_raw + 8352);
    const int fold32_addr = smem + 8352;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int nblocks = (rows + 7) / 8;
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
            int prow_0 = ((pblk * 8 + group) < (rows - 1) ? (pblk * 8 + group) : (rows - 1));
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
            int row_0 = u * 8 + group;
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
            float total_0 = 0.0f;
            total_0 = __fmaf_rn(xs_0[0], xs_0[0], total_0);
            total_0 = __fmaf_rn(xs_0[1], xs_0[1], total_0);
            total_0 = __fmaf_rn(xs_0[2], xs_0[2], total_0);
            total_0 = __fmaf_rn(xs_0[3], xs_0[3], total_0);
            total_0 = __fmaf_rn(xs_0[4], xs_0[4], total_0);
            total_0 = __fmaf_rn(xs_0[5], xs_0[5], total_0);
            total_0 = __fmaf_rn(xs_0[6], xs_0[6], total_0);
            total_0 = __fmaf_rn(xs_0[7], xs_0[7], total_0);
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total_0, 1);
            total_0 = total_0 + _shfl_xor_sync_7;
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total_0, 2);
            total_0 = total_0 + _shfl_xor_sync_8;
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total_0, 4);
            total_0 = total_0 + _shfl_xor_sync_9;
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total_0, 8);
            total_0 = total_0 + _shfl_xor_sync_10;
            float _sqrt_11;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_11) : "f"(total_0 * 0.0078125f + eps));
            float rstd_0 = __frcp_rn(_sqrt_11);
            float live_0 = ((row_0 < rows) ? 1.0f : 0.0f);
            float dot_0 = 0.0f;
            int fast_0 = 0;
            float _neg_12;
            asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(zs_0[0]));
            float _neg_13;
            asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(zs_0[1]));
            float _neg_14;
            asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(zs_0[2]));
            float mz_0;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_0) : "f"(_neg_12), "f"(_neg_13), "f"(_neg_14));
            float _abs_16;
            asm("abs.f32 %0, %1;" : "=f"(_abs_16) : "f"(zs_0[0]));
            float _neg_15;
            asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(_abs_16));
            float _abs_18;
            asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(zs_0[1]));
            float _neg_17;
            asm("neg.f32 %0, %1;" : "=f"(_neg_17) : "f"(_abs_18));
            float _abs_20;
            asm("abs.f32 %0, %1;" : "=f"(_abs_20) : "f"(zs_0[2]));
            float _neg_19;
            asm("neg.f32 %0, %1;" : "=f"(_neg_19) : "f"(_abs_20));
            float ma_0;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_0) : "f"(_neg_15), "f"(_neg_17), "f"(_neg_19));
            float _neg_22;
            asm("neg.f32 %0, %1;" : "=f"(_neg_22) : "f"(zs_0[3]));
            float _neg_23;
            asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(zs_0[4]));
            float _max3_21;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(mz_0), "f"(_neg_22), "f"(_neg_23));
            mz_0 = _max3_21;
            float _abs_26;
            asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(zs_0[3]));
            float _neg_25;
            asm("neg.f32 %0, %1;" : "=f"(_neg_25) : "f"(_abs_26));
            float _abs_28;
            asm("abs.f32 %0, %1;" : "=f"(_abs_28) : "f"(zs_0[4]));
            float _neg_27;
            asm("neg.f32 %0, %1;" : "=f"(_neg_27) : "f"(_abs_28));
            float _max3_24;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(ma_0), "f"(_neg_25), "f"(_neg_27));
            ma_0 = _max3_24;
            float _neg_30;
            asm("neg.f32 %0, %1;" : "=f"(_neg_30) : "f"(zs_0[5]));
            float _neg_31;
            asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(zs_0[6]));
            float _max3_29;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(mz_0), "f"(_neg_30), "f"(_neg_31));
            mz_0 = _max3_29;
            float _abs_34;
            asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(zs_0[5]));
            float _neg_33;
            asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(_abs_34));
            float _abs_36;
            asm("abs.f32 %0, %1;" : "=f"(_abs_36) : "f"(zs_0[6]));
            float _neg_35;
            asm("neg.f32 %0, %1;" : "=f"(_neg_35) : "f"(_abs_36));
            float _max3_32;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_32) : "f"(ma_0), "f"(_neg_33), "f"(_neg_35));
            ma_0 = _max3_32;
            float _neg_38;
            asm("neg.f32 %0, %1;" : "=f"(_neg_38) : "f"(zs_0[7]));
            float _max_37;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(mz_0), "f"(_neg_38));
            mz_0 = _max_37;
            float _abs_41;
            asm("abs.f32 %0, %1;" : "=f"(_abs_41) : "f"(zs_0[7]));
            float _neg_40;
            asm("neg.f32 %0, %1;" : "=f"(_neg_40) : "f"(_abs_41));
            float _max_39;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_39) : "f"(ma_0), "f"(_neg_40));
            ma_0 = _max_39;
            float _abs_42;
            asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(zs_0[0]));
            float _abs_43;
            asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(zs_0[1]));
            float _abs_44;
            asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(zs_0[2]));
            float mp_0;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_0) : "f"(_abs_42), "f"(_abs_43), "f"(_abs_44));
            float _abs_46;
            asm("abs.f32 %0, %1;" : "=f"(_abs_46) : "f"(zs_0[3]));
            float _abs_47;
            asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(zs_0[4]));
            float _max3_45;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_45) : "f"(mp_0), "f"(_abs_46), "f"(_abs_47));
            mp_0 = _max3_45;
            float _abs_49;
            asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(zs_0[5]));
            float _abs_50;
            asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(zs_0[6]));
            float _max3_48;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
            #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
            #endif
            asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_48) : "f"(mp_0), "f"(_abs_49), "f"(_abs_50));
            mp_0 = _max3_48;
            float _abs_52;
            asm("abs.f32 %0, %1;" : "=f"(_abs_52) : "f"(zs_0[7]));
            float _max_51;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_51) : "f"(mp_0), "f"(_abs_52));
            mp_0 = _max_51;
            if (mz_0 <= 41.0f) {
                if (ma_0 <= -1.1754944e-38f) {
                    if (mp_0 <= 3.4028235e+38f) {
                        fast_0 = 1;
                    }
                }
            }
            if (fast_0 != 0) {
                float2 rstd2_0 = make_float2(rstd_0, rstd_0);
                float2 live2_0 = make_float2(live_0, live_0);
                float2 z2_0 = make_float2(zs_0[0], zs_0[1]);
                float ta_0;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs_0[0]), "f"(-0.00572498f), "f"(0.5f));
                float tb_0;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs_0[1]), "f"(-0.00572498f), "f"(0.5f));
                float _fma_53;
                asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(ta_0), "f"(252.0f), "f"(1.2582913e+07f));
                float _fma_54;
                asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(tb_0), "f"(252.0f), "f"(1.2582913e+07f));
                float2 t2_0 = make_float2(_fma_53, _fma_54);
                float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_0));
                r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
                float2 nscale_0 = make_float2(__uint_as_float((__as_u32(t2_0.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_0.y) << 23) + 2147483648u));
                float _exp2_55 = approx_exp2(r2_0.x);
                float _exp2_56 = approx_exp2(r2_0.y);
                float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_55, _exp2_56), make_float2(-1.0f, -1.0f));
                float _neg_58;
                asm("neg.f32 %0, %1;" : "=f"(_neg_58) : "f"(nden_0.x));
                float _rcp_57 = approx_rcp(_neg_58);
                float _neg_60;
                asm("neg.f32 %0, %1;" : "=f"(_neg_60) : "f"(nden_0.y));
                float _rcp_59 = approx_rcp(_neg_60);
                float2 r0_0 = make_float2(_rcp_57, _rcp_59);
                float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
                float2 sig2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, r1_0, make_float2(1.0f, 1.0f)), r1_0);
                float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
                float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
                float2 g2_0 = make_float2(gs_0[0], gs_0[1]);
                float2 w2_0 = make_float2(wv[0], wv[1]);
                float2 _f32x2_mul_a_61 = make_float2(xs_0[0], xs_0[1]);
                float2 n2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&rstd2_0));
                float2 _f32x2_mul_a_62 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
                float2 _f32x2_mul_b_63 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
                float2 dz2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
                float2 a2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
                float2 dn2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
                float2 acc2_0 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_0, live2_0), n2_0, make_float2(acc[0], acc[1]));
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
                float _fma_64;
                asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(ta_1), "f"(252.0f), "f"(1.2582913e+07f));
                float _fma_65;
                asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(tb_1), "f"(252.0f), "f"(1.2582913e+07f));
                float2 t2_1 = make_float2(_fma_64, _fma_65);
                float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_1));
                r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
                float2 nscale_1 = make_float2(__uint_as_float((__as_u32(t2_1.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_1.y) << 23) + 2147483648u));
                float _exp2_66 = approx_exp2(r2_1.x);
                float _exp2_67 = approx_exp2(r2_1.y);
                float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_66, _exp2_67), make_float2(-1.0f, -1.0f));
                float _neg_69;
                asm("neg.f32 %0, %1;" : "=f"(_neg_69) : "f"(nden_1.x));
                float _rcp_68 = approx_rcp(_neg_69);
                float _neg_71;
                asm("neg.f32 %0, %1;" : "=f"(_neg_71) : "f"(nden_1.y));
                float _rcp_70 = approx_rcp(_neg_71);
                float2 r0_1 = make_float2(_rcp_68, _rcp_70);
                float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
                float2 sig2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, r1_1, make_float2(1.0f, 1.0f)), r1_1);
                float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
                float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
                float2 g2_1 = make_float2(gs_0[2], gs_0[3]);
                float2 w2_1 = make_float2(wv[2], wv[3]);
                float2 _f32x2_mul_a_72 = make_float2(xs_0[2], xs_0[3]);
                float2 n2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&rstd2_0));
                float2 _f32x2_mul_a_73 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
                float2 _f32x2_mul_b_74 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
                float2 dz2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_74));
                float2 a2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
                float2 dn2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
                float2 acc2_1 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_1, live2_0), n2_1, make_float2(acc[2], acc[3]));
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
                float _fma_75;
                asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(ta_2), "f"(252.0f), "f"(1.2582913e+07f));
                float _fma_76;
                asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(tb_2), "f"(252.0f), "f"(1.2582913e+07f));
                float2 t2_2 = make_float2(_fma_75, _fma_76);
                float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_2));
                r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
                float2 nscale_2 = make_float2(__uint_as_float((__as_u32(t2_2.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_2.y) << 23) + 2147483648u));
                float _exp2_77 = approx_exp2(r2_2.x);
                float _exp2_78 = approx_exp2(r2_2.y);
                float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_77, _exp2_78), make_float2(-1.0f, -1.0f));
                float _neg_80;
                asm("neg.f32 %0, %1;" : "=f"(_neg_80) : "f"(nden_2.x));
                float _rcp_79 = approx_rcp(_neg_80);
                float _neg_82;
                asm("neg.f32 %0, %1;" : "=f"(_neg_82) : "f"(nden_2.y));
                float _rcp_81 = approx_rcp(_neg_82);
                float2 r0_2 = make_float2(_rcp_79, _rcp_81);
                float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
                float2 sig2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, r1_2, make_float2(1.0f, 1.0f)), r1_2);
                float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
                float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
                float2 g2_2 = make_float2(gs_0[4], gs_0[5]);
                float2 w2_2 = make_float2(wv[4], wv[5]);
                float2 _f32x2_mul_a_83 = make_float2(xs_0[4], xs_0[5]);
                float2 n2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&rstd2_0));
                float2 _f32x2_mul_a_84 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
                float2 _f32x2_mul_b_85 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
                float2 dz2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
                float2 a2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
                float2 dn2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
                float2 acc2_2 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_2, live2_0), n2_2, make_float2(acc[4], acc[5]));
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
                float _fma_86;
                asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(ta_3), "f"(252.0f), "f"(1.2582913e+07f));
                float _fma_87;
                asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(tb_3), "f"(252.0f), "f"(1.2582913e+07f));
                float2 t2_3 = make_float2(_fma_86, _fma_87);
                float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_3));
                r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
                float2 nscale_3 = make_float2(__uint_as_float((__as_u32(t2_3.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_3.y) << 23) + 2147483648u));
                float _exp2_88 = approx_exp2(r2_3.x);
                float _exp2_89 = approx_exp2(r2_3.y);
                float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_88, _exp2_89), make_float2(-1.0f, -1.0f));
                float _neg_91;
                asm("neg.f32 %0, %1;" : "=f"(_neg_91) : "f"(nden_3.x));
                float _rcp_90 = approx_rcp(_neg_91);
                float _neg_93;
                asm("neg.f32 %0, %1;" : "=f"(_neg_93) : "f"(nden_3.y));
                float _rcp_92 = approx_rcp(_neg_93);
                float2 r0_3 = make_float2(_rcp_90, _rcp_92);
                float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
                float2 sig2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, r1_3, make_float2(1.0f, 1.0f)), r1_3);
                float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
                float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
                float2 g2_3 = make_float2(gs_0[6], gs_0[7]);
                float2 w2_3 = make_float2(wv[6], wv[7]);
                float2 _f32x2_mul_a_94 = make_float2(xs_0[6], xs_0[7]);
                float2 n2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&rstd2_0));
                float2 _f32x2_mul_a_95 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
                float2 _f32x2_mul_b_96 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
                float2 dz2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_96));
                float2 a2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
                float2 dn2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
                float2 acc2_3 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_3, live2_0), n2_3, make_float2(acc[6], acc[7]));
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
                float _exp_97 = __nv_expf(zs_0[0] * -1.0f);
                float den_0 = 1.0f + _exp_97;
                float sig_0 = __fdiv_rn(1.0f, den_0);
                float gate_0 = __fdiv_rn(zs_0[0], den_0);
                float nj_0 = xs_0[0] * rstd_0;
                float aj_0 = gs_0[0] * gate_0;
                float dzj_0 = gs_0[0] * (nj_0 * wv[0]) * sig_0 * __fmaf_rn(zs_0[0], 1.0f - sig_0, 1.0f);
                float dnj_0 = aj_0 * wv[0];
                dot_0 = __fmaf_rn(dnj_0, xs_0[0], dot_0);
                acc[0] = __fmaf_rn(aj_0 * live_0, nj_0, acc[0]);
                zs_0[0] = dzj_0;
                gs_0[0] = dnj_0;
                float _exp_98 = __nv_expf(zs_0[1] * -1.0f);
                float den_1 = 1.0f + _exp_98;
                float sig_1 = __fdiv_rn(1.0f, den_1);
                float gate_1 = __fdiv_rn(zs_0[1], den_1);
                float nj_1 = xs_0[1] * rstd_0;
                float aj_1 = gs_0[1] * gate_1;
                float dzj_1 = gs_0[1] * (nj_1 * wv[1]) * sig_1 * __fmaf_rn(zs_0[1], 1.0f - sig_1, 1.0f);
                float dnj_1 = aj_1 * wv[1];
                dot_0 = __fmaf_rn(dnj_1, xs_0[1], dot_0);
                acc[1] = __fmaf_rn(aj_1 * live_0, nj_1, acc[1]);
                zs_0[1] = dzj_1;
                gs_0[1] = dnj_1;
                float _exp_99 = __nv_expf(zs_0[2] * -1.0f);
                float den_2 = 1.0f + _exp_99;
                float sig_2 = __fdiv_rn(1.0f, den_2);
                float gate_2 = __fdiv_rn(zs_0[2], den_2);
                float nj_2 = xs_0[2] * rstd_0;
                float aj_2 = gs_0[2] * gate_2;
                float dzj_2 = gs_0[2] * (nj_2 * wv[2]) * sig_2 * __fmaf_rn(zs_0[2], 1.0f - sig_2, 1.0f);
                float dnj_2 = aj_2 * wv[2];
                dot_0 = __fmaf_rn(dnj_2, xs_0[2], dot_0);
                acc[2] = __fmaf_rn(aj_2 * live_0, nj_2, acc[2]);
                zs_0[2] = dzj_2;
                gs_0[2] = dnj_2;
                float _exp_100 = __nv_expf(zs_0[3] * -1.0f);
                float den_3 = 1.0f + _exp_100;
                float sig_3 = __fdiv_rn(1.0f, den_3);
                float gate_3 = __fdiv_rn(zs_0[3], den_3);
                float nj_3 = xs_0[3] * rstd_0;
                float aj_3 = gs_0[3] * gate_3;
                float dzj_3 = gs_0[3] * (nj_3 * wv[3]) * sig_3 * __fmaf_rn(zs_0[3], 1.0f - sig_3, 1.0f);
                float dnj_3 = aj_3 * wv[3];
                dot_0 = __fmaf_rn(dnj_3, xs_0[3], dot_0);
                acc[3] = __fmaf_rn(aj_3 * live_0, nj_3, acc[3]);
                zs_0[3] = dzj_3;
                gs_0[3] = dnj_3;
                float _exp_101 = __nv_expf(zs_0[4] * -1.0f);
                float den_4 = 1.0f + _exp_101;
                float sig_4 = __fdiv_rn(1.0f, den_4);
                float gate_4 = __fdiv_rn(zs_0[4], den_4);
                float nj_4 = xs_0[4] * rstd_0;
                float aj_4 = gs_0[4] * gate_4;
                float dzj_4 = gs_0[4] * (nj_4 * wv[4]) * sig_4 * __fmaf_rn(zs_0[4], 1.0f - sig_4, 1.0f);
                float dnj_4 = aj_4 * wv[4];
                dot_0 = __fmaf_rn(dnj_4, xs_0[4], dot_0);
                acc[4] = __fmaf_rn(aj_4 * live_0, nj_4, acc[4]);
                zs_0[4] = dzj_4;
                gs_0[4] = dnj_4;
                float _exp_102 = __nv_expf(zs_0[5] * -1.0f);
                float den_5 = 1.0f + _exp_102;
                float sig_5 = __fdiv_rn(1.0f, den_5);
                float gate_5 = __fdiv_rn(zs_0[5], den_5);
                float nj_5 = xs_0[5] * rstd_0;
                float aj_5 = gs_0[5] * gate_5;
                float dzj_5 = gs_0[5] * (nj_5 * wv[5]) * sig_5 * __fmaf_rn(zs_0[5], 1.0f - sig_5, 1.0f);
                float dnj_5 = aj_5 * wv[5];
                dot_0 = __fmaf_rn(dnj_5, xs_0[5], dot_0);
                acc[5] = __fmaf_rn(aj_5 * live_0, nj_5, acc[5]);
                zs_0[5] = dzj_5;
                gs_0[5] = dnj_5;
                float _exp_103 = __nv_expf(zs_0[6] * -1.0f);
                float den_6 = 1.0f + _exp_103;
                float sig_6 = __fdiv_rn(1.0f, den_6);
                float gate_6 = __fdiv_rn(zs_0[6], den_6);
                float nj_6 = xs_0[6] * rstd_0;
                float aj_6 = gs_0[6] * gate_6;
                float dzj_6 = gs_0[6] * (nj_6 * wv[6]) * sig_6 * __fmaf_rn(zs_0[6], 1.0f - sig_6, 1.0f);
                float dnj_6 = aj_6 * wv[6];
                dot_0 = __fmaf_rn(dnj_6, xs_0[6], dot_0);
                acc[6] = __fmaf_rn(aj_6 * live_0, nj_6, acc[6]);
                zs_0[6] = dzj_6;
                gs_0[6] = dnj_6;
                float _exp_104 = __nv_expf(zs_0[7] * -1.0f);
                float den_7 = 1.0f + _exp_104;
                float sig_7 = __fdiv_rn(1.0f, den_7);
                float gate_7 = __fdiv_rn(zs_0[7], den_7);
                float nj_7 = xs_0[7] * rstd_0;
                float aj_7 = gs_0[7] * gate_7;
                float dzj_7 = gs_0[7] * (nj_7 * wv[7]) * sig_7 * __fmaf_rn(zs_0[7], 1.0f - sig_7, 1.0f);
                float dnj_7 = aj_7 * wv[7];
                dot_0 = __fmaf_rn(dnj_7, xs_0[7], dot_0);
                acc[7] = __fmaf_rn(aj_7 * live_0, nj_7, acc[7]);
                zs_0[7] = dzj_7;
                gs_0[7] = dnj_7;
            }
            float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
            dot_0 = dot_0 + _shfl_xor_sync_105;
            float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
            dot_0 = dot_0 + _shfl_xor_sync_106;
            float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
            dot_0 = dot_0 + _shfl_xor_sync_107;
            float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 8);
            dot_0 = dot_0 + _shfl_xor_sync_108;
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
                long long orow_0 = (long long)row_0 * 128 + (long long)col;
                reinterpret_cast<int4*>(dx + orow_0)[0] = reinterpret_cast<int4*>(packed_dx_0)[0];
                reinterpret_cast<int4*>(dz + orow_0)[0] = reinterpret_cast<int4*>(packed_dz_0)[0];
            }
            run = run + 1;
            if (run == 16) {
                if (folded != 0) {
                    fold[group * 128 + col] = fold[group * 128 + col] + (double)acc[0];
                    acc[0] = 0.0f;
                    fold[group * 128 + col + 1] = fold[group * 128 + col + 1] + (double)acc[1];
                    acc[1] = 0.0f;
                    fold[group * 128 + col + 2] = fold[group * 128 + col + 2] + (double)acc[2];
                    acc[2] = 0.0f;
                    fold[group * 128 + col + 3] = fold[group * 128 + col + 3] + (double)acc[3];
                    acc[3] = 0.0f;
                    fold[group * 128 + col + 4] = fold[group * 128 + col + 4] + (double)acc[4];
                    acc[4] = 0.0f;
                    fold[group * 128 + col + 5] = fold[group * 128 + col + 5] + (double)acc[5];
                    acc[5] = 0.0f;
                    fold[group * 128 + col + 6] = fold[group * 128 + col + 6] + (double)acc[6];
                    acc[6] = 0.0f;
                    fold[group * 128 + col + 7] = fold[group * 128 + col + 7] + (double)acc[7];
                    acc[7] = 0.0f;
                } else {
                    fold[group * 128 + col] = (double)acc[0];
                    acc[0] = 0.0f;
                    fold[group * 128 + col + 1] = (double)acc[1];
                    acc[1] = 0.0f;
                    fold[group * 128 + col + 2] = (double)acc[2];
                    acc[2] = 0.0f;
                    fold[group * 128 + col + 3] = (double)acc[3];
                    acc[3] = 0.0f;
                    fold[group * 128 + col + 4] = (double)acc[4];
                    acc[4] = 0.0f;
                    fold[group * 128 + col + 5] = (double)acc[5];
                    acc[5] = 0.0f;
                    fold[group * 128 + col + 6] = (double)acc[6];
                    acc[6] = 0.0f;
                    fold[group * 128 + col + 7] = (double)acc[7];
                    acc[7] = 0.0f;
                }
                folded = 1;
                run = 0;
            }
        }
    }
    if (rest > 0) {
        {
            const uint4* _vptr_109 = reinterpret_cast<const uint4*>(w + col + 0);
            uint4 _vld_109[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_vld_109[_blk].x), "=r"(_vld_109[_blk].y), "=r"(_vld_109[_blk].z), "=r"(_vld_109[_blk].w) : "l"((const void*)(_vptr_109 + _blk)) : "memory");
                uint32_t* _vpairs_109 = reinterpret_cast<uint32_t*>(&_vld_109[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&wclaim[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_109[_pair]) << 16);
                    (&wclaim[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_109[_pair]) & 0xffff0000u);
                }
            }
        }
    }
    int trips = 0;
    if (rest > 0) {
        int _min_110 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_110) > (1) ? (_min_110) : (1));
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
                int row_1 = u_1 * 8 + group;
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
                    const uint4* _vptr_111 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                    uint4 _vld_111[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_111[_blk] = _vptr_111[_blk];
                        uint32_t* _vpairs_111 = reinterpret_cast<uint32_t*>(&_vld_111[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) << 16);
                            (&gs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_112 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                    uint4 _vld_112[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_112[_blk] = _vptr_112[_blk];
                        uint32_t* _vpairs_112 = reinterpret_cast<uint32_t*>(&_vld_112[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_112[_pair]) << 16);
                            (&xs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_112[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_113 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                    uint4 _vld_113[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_113[_blk] = _vptr_113[_blk];
                        uint32_t* _vpairs_113 = reinterpret_cast<uint32_t*>(&_vld_113[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zs_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_113[_pair]) << 16);
                            (&zs_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_113[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (tid == 0) {
                    if (i_1 == claim_at) {
                        uint32_t _atomic_fetch_inc_114;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_114) : "l"(&dw_done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        ticket = (int)_atomic_fetch_inc_114;
                    }
                }
                float total_1 = 0.0f;
                total_1 = __fmaf_rn(xs_1[0], xs_1[0], total_1);
                total_1 = __fmaf_rn(xs_1[1], xs_1[1], total_1);
                total_1 = __fmaf_rn(xs_1[2], xs_1[2], total_1);
                total_1 = __fmaf_rn(xs_1[3], xs_1[3], total_1);
                total_1 = __fmaf_rn(xs_1[4], xs_1[4], total_1);
                total_1 = __fmaf_rn(xs_1[5], xs_1[5], total_1);
                total_1 = __fmaf_rn(xs_1[6], xs_1[6], total_1);
                total_1 = __fmaf_rn(xs_1[7], xs_1[7], total_1);
                float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
                total_1 = total_1 + _shfl_xor_sync_115;
                float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
                total_1 = total_1 + _shfl_xor_sync_116;
                float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
                total_1 = total_1 + _shfl_xor_sync_117;
                float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
                total_1 = total_1 + _shfl_xor_sync_118;
                float _sqrt_119;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_119) : "f"(total_1 * 0.0078125f + eps));
                float rstd_1 = __frcp_rn(_sqrt_119);
                float live_1 = ((row_1 < rows) ? 1.0f : 0.0f);
                float dot_1 = 0.0f;
                int fast_1 = 0;
                float _neg_120;
                asm("neg.f32 %0, %1;" : "=f"(_neg_120) : "f"(zs_1[0]));
                float _neg_121;
                asm("neg.f32 %0, %1;" : "=f"(_neg_121) : "f"(zs_1[1]));
                float _neg_122;
                asm("neg.f32 %0, %1;" : "=f"(_neg_122) : "f"(zs_1[2]));
                float mz_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_1) : "f"(_neg_120), "f"(_neg_121), "f"(_neg_122));
                float _abs_124;
                asm("abs.f32 %0, %1;" : "=f"(_abs_124) : "f"(zs_1[0]));
                float _neg_123;
                asm("neg.f32 %0, %1;" : "=f"(_neg_123) : "f"(_abs_124));
                float _abs_126;
                asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(zs_1[1]));
                float _neg_125;
                asm("neg.f32 %0, %1;" : "=f"(_neg_125) : "f"(_abs_126));
                float _abs_128;
                asm("abs.f32 %0, %1;" : "=f"(_abs_128) : "f"(zs_1[2]));
                float _neg_127;
                asm("neg.f32 %0, %1;" : "=f"(_neg_127) : "f"(_abs_128));
                float ma_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_1) : "f"(_neg_123), "f"(_neg_125), "f"(_neg_127));
                float _neg_130;
                asm("neg.f32 %0, %1;" : "=f"(_neg_130) : "f"(zs_1[3]));
                float _neg_131;
                asm("neg.f32 %0, %1;" : "=f"(_neg_131) : "f"(zs_1[4]));
                float _max3_129;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_129) : "f"(mz_1), "f"(_neg_130), "f"(_neg_131));
                mz_1 = _max3_129;
                float _abs_134;
                asm("abs.f32 %0, %1;" : "=f"(_abs_134) : "f"(zs_1[3]));
                float _neg_133;
                asm("neg.f32 %0, %1;" : "=f"(_neg_133) : "f"(_abs_134));
                float _abs_136;
                asm("abs.f32 %0, %1;" : "=f"(_abs_136) : "f"(zs_1[4]));
                float _neg_135;
                asm("neg.f32 %0, %1;" : "=f"(_neg_135) : "f"(_abs_136));
                float _max3_132;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_132) : "f"(ma_1), "f"(_neg_133), "f"(_neg_135));
                ma_1 = _max3_132;
                float _neg_138;
                asm("neg.f32 %0, %1;" : "=f"(_neg_138) : "f"(zs_1[5]));
                float _neg_139;
                asm("neg.f32 %0, %1;" : "=f"(_neg_139) : "f"(zs_1[6]));
                float _max3_137;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_137) : "f"(mz_1), "f"(_neg_138), "f"(_neg_139));
                mz_1 = _max3_137;
                float _abs_142;
                asm("abs.f32 %0, %1;" : "=f"(_abs_142) : "f"(zs_1[5]));
                float _neg_141;
                asm("neg.f32 %0, %1;" : "=f"(_neg_141) : "f"(_abs_142));
                float _abs_144;
                asm("abs.f32 %0, %1;" : "=f"(_abs_144) : "f"(zs_1[6]));
                float _neg_143;
                asm("neg.f32 %0, %1;" : "=f"(_neg_143) : "f"(_abs_144));
                float _max3_140;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_140) : "f"(ma_1), "f"(_neg_141), "f"(_neg_143));
                ma_1 = _max3_140;
                float _neg_146;
                asm("neg.f32 %0, %1;" : "=f"(_neg_146) : "f"(zs_1[7]));
                float _max_145;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_145) : "f"(mz_1), "f"(_neg_146));
                mz_1 = _max_145;
                float _abs_149;
                asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(zs_1[7]));
                float _neg_148;
                asm("neg.f32 %0, %1;" : "=f"(_neg_148) : "f"(_abs_149));
                float _max_147;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_147) : "f"(ma_1), "f"(_neg_148));
                ma_1 = _max_147;
                float _abs_150;
                asm("abs.f32 %0, %1;" : "=f"(_abs_150) : "f"(zs_1[0]));
                float _abs_151;
                asm("abs.f32 %0, %1;" : "=f"(_abs_151) : "f"(zs_1[1]));
                float _abs_152;
                asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(zs_1[2]));
                float mp_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_1) : "f"(_abs_150), "f"(_abs_151), "f"(_abs_152));
                float _abs_154;
                asm("abs.f32 %0, %1;" : "=f"(_abs_154) : "f"(zs_1[3]));
                float _abs_155;
                asm("abs.f32 %0, %1;" : "=f"(_abs_155) : "f"(zs_1[4]));
                float _max3_153;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_153) : "f"(mp_1), "f"(_abs_154), "f"(_abs_155));
                mp_1 = _max3_153;
                float _abs_157;
                asm("abs.f32 %0, %1;" : "=f"(_abs_157) : "f"(zs_1[5]));
                float _abs_158;
                asm("abs.f32 %0, %1;" : "=f"(_abs_158) : "f"(zs_1[6]));
                float _max3_156;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_156) : "f"(mp_1), "f"(_abs_157), "f"(_abs_158));
                mp_1 = _max3_156;
                float _abs_160;
                asm("abs.f32 %0, %1;" : "=f"(_abs_160) : "f"(zs_1[7]));
                float _max_159;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_159) : "f"(mp_1), "f"(_abs_160));
                mp_1 = _max_159;
                if (mz_1 <= 41.0f) {
                    if (ma_1 <= -1.1754944e-38f) {
                        if (mp_1 <= 3.4028235e+38f) {
                            fast_1 = 1;
                        }
                    }
                }
                if (fast_1 != 0) {
                    float2 rstd2_1 = make_float2(rstd_1, rstd_1);
                    float2 live2_1 = make_float2(live_1, live_1);
                    float2 z2_4 = make_float2(zs_1[0], zs_1[1]);
                    float ta_4;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_4) : "f"(zs_1[0]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_4;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_4) : "f"(zs_1[1]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_161;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(ta_4), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_162;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(tb_4), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_4 = make_float2(_fma_161, _fma_162);
                    float2 r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_4));
                    r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.925963e-08f, -1.925963e-08f), r2_4);
                    float2 nscale_4 = make_float2(__uint_as_float((__as_u32(t2_4.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_4.y) << 23) + 2147483648u));
                    float _exp2_163 = approx_exp2(r2_4.x);
                    float _exp2_164 = approx_exp2(r2_4.y);
                    float2 nden_4 = fma_f32x2_rn_noftz(nscale_4, make_float2(_exp2_163, _exp2_164), make_float2(-1.0f, -1.0f));
                    float _neg_166;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_166) : "f"(nden_4.x));
                    float _rcp_165 = approx_rcp(_neg_166);
                    float _neg_168;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_168) : "f"(nden_4.y));
                    float _rcp_167 = approx_rcp(_neg_168);
                    float2 r0_4 = make_float2(_rcp_165, _rcp_167);
                    float2 r1_4 = fma_f32x2_rn_noftz(r0_4, fma_f32x2_rn_noftz(nden_4, r0_4, make_float2(1.0f, 1.0f)), r0_4);
                    float2 sig2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, r1_4, make_float2(1.0f, 1.0f)), r1_4);
                    float2 q0_4 = fma_f32x2_rn_noftz(z2_4, r1_4, make_float2(0.0f, 0.0f));
                    float2 gate2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, q0_4, z2_4), q0_4);
                    float2 g2_4 = make_float2(gs_1[0], gs_1[1]);
                    float2 w2_4 = make_float2(wclaim[0], wclaim[1]);
                    float2 _f32x2_mul_a_169 = make_float2(xs_1[0], xs_1[1]);
                    float2 n2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_170 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
                    float2 _f32x2_mul_b_171 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
                    float2 dz2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&_f32x2_mul_b_171));
                    float2 a2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&gate2_4));
                    float2 dn2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_4) : "l"(*(const unsigned long long*)&a2_4), "l"(*(const unsigned long long*)&w2_4));
                    float2 acc2_4 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_4, live2_1), n2_4, make_float2(acc[0], acc[1]));
                    acc[0] = acc2_4.x;
                    acc[1] = acc2_4.y;
                    zs_1[0] = dz2_4.x;
                    zs_1[1] = dz2_4.y;
                    gs_1[0] = dn2_4.x;
                    gs_1[1] = dn2_4.y;
                    float2 z2_5 = make_float2(zs_1[2], zs_1[3]);
                    float ta_5;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_5) : "f"(zs_1[2]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_5;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_5) : "f"(zs_1[3]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_172;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(ta_5), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_173;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(tb_5), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_5 = make_float2(_fma_172, _fma_173);
                    float2 r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_5));
                    r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.925963e-08f, -1.925963e-08f), r2_5);
                    float2 nscale_5 = make_float2(__uint_as_float((__as_u32(t2_5.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_5.y) << 23) + 2147483648u));
                    float _exp2_174 = approx_exp2(r2_5.x);
                    float _exp2_175 = approx_exp2(r2_5.y);
                    float2 nden_5 = fma_f32x2_rn_noftz(nscale_5, make_float2(_exp2_174, _exp2_175), make_float2(-1.0f, -1.0f));
                    float _neg_177;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_177) : "f"(nden_5.x));
                    float _rcp_176 = approx_rcp(_neg_177);
                    float _neg_179;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_179) : "f"(nden_5.y));
                    float _rcp_178 = approx_rcp(_neg_179);
                    float2 r0_5 = make_float2(_rcp_176, _rcp_178);
                    float2 r1_5 = fma_f32x2_rn_noftz(r0_5, fma_f32x2_rn_noftz(nden_5, r0_5, make_float2(1.0f, 1.0f)), r0_5);
                    float2 sig2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, r1_5, make_float2(1.0f, 1.0f)), r1_5);
                    float2 q0_5 = fma_f32x2_rn_noftz(z2_5, r1_5, make_float2(0.0f, 0.0f));
                    float2 gate2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, q0_5, z2_5), q0_5);
                    float2 g2_5 = make_float2(gs_1[2], gs_1[3]);
                    float2 w2_5 = make_float2(wclaim[2], wclaim[3]);
                    float2 _f32x2_mul_a_180 = make_float2(xs_1[2], xs_1[3]);
                    float2 n2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_181 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
                    float2 _f32x2_mul_b_182 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
                    float2 dz2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&_f32x2_mul_b_182));
                    float2 a2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&gate2_5));
                    float2 dn2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_5) : "l"(*(const unsigned long long*)&a2_5), "l"(*(const unsigned long long*)&w2_5));
                    float2 acc2_5 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_5, live2_1), n2_5, make_float2(acc[2], acc[3]));
                    acc[2] = acc2_5.x;
                    acc[3] = acc2_5.y;
                    zs_1[2] = dz2_5.x;
                    zs_1[3] = dz2_5.y;
                    gs_1[2] = dn2_5.x;
                    gs_1[3] = dn2_5.y;
                    float2 z2_6 = make_float2(zs_1[4], zs_1[5]);
                    float ta_6;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_6) : "f"(zs_1[4]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_6;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_6) : "f"(zs_1[5]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_183;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(ta_6), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_184;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(tb_6), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_6 = make_float2(_fma_183, _fma_184);
                    float2 r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_6));
                    r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.925963e-08f, -1.925963e-08f), r2_6);
                    float2 nscale_6 = make_float2(__uint_as_float((__as_u32(t2_6.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_6.y) << 23) + 2147483648u));
                    float _exp2_185 = approx_exp2(r2_6.x);
                    float _exp2_186 = approx_exp2(r2_6.y);
                    float2 nden_6 = fma_f32x2_rn_noftz(nscale_6, make_float2(_exp2_185, _exp2_186), make_float2(-1.0f, -1.0f));
                    float _neg_188;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_188) : "f"(nden_6.x));
                    float _rcp_187 = approx_rcp(_neg_188);
                    float _neg_190;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_190) : "f"(nden_6.y));
                    float _rcp_189 = approx_rcp(_neg_190);
                    float2 r0_6 = make_float2(_rcp_187, _rcp_189);
                    float2 r1_6 = fma_f32x2_rn_noftz(r0_6, fma_f32x2_rn_noftz(nden_6, r0_6, make_float2(1.0f, 1.0f)), r0_6);
                    float2 sig2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, r1_6, make_float2(1.0f, 1.0f)), r1_6);
                    float2 q0_6 = fma_f32x2_rn_noftz(z2_6, r1_6, make_float2(0.0f, 0.0f));
                    float2 gate2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, q0_6, z2_6), q0_6);
                    float2 g2_6 = make_float2(gs_1[4], gs_1[5]);
                    float2 w2_6 = make_float2(wclaim[4], wclaim[5]);
                    float2 _f32x2_mul_a_191 = make_float2(xs_1[4], xs_1[5]);
                    float2 n2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_192 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
                    float2 _f32x2_mul_b_193 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
                    float2 dz2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_192), "l"(*(const unsigned long long*)&_f32x2_mul_b_193));
                    float2 a2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&gate2_6));
                    float2 dn2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_6) : "l"(*(const unsigned long long*)&a2_6), "l"(*(const unsigned long long*)&w2_6));
                    float2 acc2_6 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_6, live2_1), n2_6, make_float2(acc[4], acc[5]));
                    acc[4] = acc2_6.x;
                    acc[5] = acc2_6.y;
                    zs_1[4] = dz2_6.x;
                    zs_1[5] = dz2_6.y;
                    gs_1[4] = dn2_6.x;
                    gs_1[5] = dn2_6.y;
                    float2 z2_7 = make_float2(zs_1[6], zs_1[7]);
                    float ta_7;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_7) : "f"(zs_1[6]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_7;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_7) : "f"(zs_1[7]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_194;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(ta_7), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_195;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(tb_7), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_7 = make_float2(_fma_194, _fma_195);
                    float2 r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_7));
                    r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.925963e-08f, -1.925963e-08f), r2_7);
                    float2 nscale_7 = make_float2(__uint_as_float((__as_u32(t2_7.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_7.y) << 23) + 2147483648u));
                    float _exp2_196 = approx_exp2(r2_7.x);
                    float _exp2_197 = approx_exp2(r2_7.y);
                    float2 nden_7 = fma_f32x2_rn_noftz(nscale_7, make_float2(_exp2_196, _exp2_197), make_float2(-1.0f, -1.0f));
                    float _neg_199;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_199) : "f"(nden_7.x));
                    float _rcp_198 = approx_rcp(_neg_199);
                    float _neg_201;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_201) : "f"(nden_7.y));
                    float _rcp_200 = approx_rcp(_neg_201);
                    float2 r0_7 = make_float2(_rcp_198, _rcp_200);
                    float2 r1_7 = fma_f32x2_rn_noftz(r0_7, fma_f32x2_rn_noftz(nden_7, r0_7, make_float2(1.0f, 1.0f)), r0_7);
                    float2 sig2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, r1_7, make_float2(1.0f, 1.0f)), r1_7);
                    float2 q0_7 = fma_f32x2_rn_noftz(z2_7, r1_7, make_float2(0.0f, 0.0f));
                    float2 gate2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, q0_7, z2_7), q0_7);
                    float2 g2_7 = make_float2(gs_1[6], gs_1[7]);
                    float2 w2_7 = make_float2(wclaim[6], wclaim[7]);
                    float2 _f32x2_mul_a_202 = make_float2(xs_1[6], xs_1[7]);
                    float2 n2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_203 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
                    float2 _f32x2_mul_b_204 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
                    float2 dz2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_203), "l"(*(const unsigned long long*)&_f32x2_mul_b_204));
                    float2 a2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&gate2_7));
                    float2 dn2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_7) : "l"(*(const unsigned long long*)&a2_7), "l"(*(const unsigned long long*)&w2_7));
                    float2 acc2_7 = fma_f32x2_rn_noftz(mul_f32x2_noftz(a2_7, live2_1), n2_7, make_float2(acc[6], acc[7]));
                    acc[6] = acc2_7.x;
                    acc[7] = acc2_7.y;
                    zs_1[6] = dz2_7.x;
                    zs_1[7] = dz2_7.y;
                    gs_1[6] = dn2_7.x;
                    gs_1[7] = dn2_7.y;
                    dot_1 = __fmaf_rn(gs_1[0], xs_1[0], dot_1);
                    dot_1 = __fmaf_rn(gs_1[1], xs_1[1], dot_1);
                    dot_1 = __fmaf_rn(gs_1[2], xs_1[2], dot_1);
                    dot_1 = __fmaf_rn(gs_1[3], xs_1[3], dot_1);
                    dot_1 = __fmaf_rn(gs_1[4], xs_1[4], dot_1);
                    dot_1 = __fmaf_rn(gs_1[5], xs_1[5], dot_1);
                    dot_1 = __fmaf_rn(gs_1[6], xs_1[6], dot_1);
                    dot_1 = __fmaf_rn(gs_1[7], xs_1[7], dot_1);
                } else {
                    float _exp_205 = __nv_expf(zs_1[0] * -1.0f);
                    float den_8 = 1.0f + _exp_205;
                    float sig_8 = __fdiv_rn(1.0f, den_8);
                    float gate_8 = __fdiv_rn(zs_1[0], den_8);
                    float nj_8 = xs_1[0] * rstd_1;
                    float aj_8 = gs_1[0] * gate_8;
                    float dzj_8 = gs_1[0] * (nj_8 * wclaim[0]) * sig_8 * __fmaf_rn(zs_1[0], 1.0f - sig_8, 1.0f);
                    float dnj_8 = aj_8 * wclaim[0];
                    dot_1 = __fmaf_rn(dnj_8, xs_1[0], dot_1);
                    acc[0] = __fmaf_rn(aj_8 * live_1, nj_8, acc[0]);
                    zs_1[0] = dzj_8;
                    gs_1[0] = dnj_8;
                    float _exp_206 = __nv_expf(zs_1[1] * -1.0f);
                    float den_9 = 1.0f + _exp_206;
                    float sig_9 = __fdiv_rn(1.0f, den_9);
                    float gate_9 = __fdiv_rn(zs_1[1], den_9);
                    float nj_9 = xs_1[1] * rstd_1;
                    float aj_9 = gs_1[1] * gate_9;
                    float dzj_9 = gs_1[1] * (nj_9 * wclaim[1]) * sig_9 * __fmaf_rn(zs_1[1], 1.0f - sig_9, 1.0f);
                    float dnj_9 = aj_9 * wclaim[1];
                    dot_1 = __fmaf_rn(dnj_9, xs_1[1], dot_1);
                    acc[1] = __fmaf_rn(aj_9 * live_1, nj_9, acc[1]);
                    zs_1[1] = dzj_9;
                    gs_1[1] = dnj_9;
                    float _exp_207 = __nv_expf(zs_1[2] * -1.0f);
                    float den_10 = 1.0f + _exp_207;
                    float sig_10 = __fdiv_rn(1.0f, den_10);
                    float gate_10 = __fdiv_rn(zs_1[2], den_10);
                    float nj_10 = xs_1[2] * rstd_1;
                    float aj_10 = gs_1[2] * gate_10;
                    float dzj_10 = gs_1[2] * (nj_10 * wclaim[2]) * sig_10 * __fmaf_rn(zs_1[2], 1.0f - sig_10, 1.0f);
                    float dnj_10 = aj_10 * wclaim[2];
                    dot_1 = __fmaf_rn(dnj_10, xs_1[2], dot_1);
                    acc[2] = __fmaf_rn(aj_10 * live_1, nj_10, acc[2]);
                    zs_1[2] = dzj_10;
                    gs_1[2] = dnj_10;
                    float _exp_208 = __nv_expf(zs_1[3] * -1.0f);
                    float den_11 = 1.0f + _exp_208;
                    float sig_11 = __fdiv_rn(1.0f, den_11);
                    float gate_11 = __fdiv_rn(zs_1[3], den_11);
                    float nj_11 = xs_1[3] * rstd_1;
                    float aj_11 = gs_1[3] * gate_11;
                    float dzj_11 = gs_1[3] * (nj_11 * wclaim[3]) * sig_11 * __fmaf_rn(zs_1[3], 1.0f - sig_11, 1.0f);
                    float dnj_11 = aj_11 * wclaim[3];
                    dot_1 = __fmaf_rn(dnj_11, xs_1[3], dot_1);
                    acc[3] = __fmaf_rn(aj_11 * live_1, nj_11, acc[3]);
                    zs_1[3] = dzj_11;
                    gs_1[3] = dnj_11;
                    float _exp_209 = __nv_expf(zs_1[4] * -1.0f);
                    float den_12 = 1.0f + _exp_209;
                    float sig_12 = __fdiv_rn(1.0f, den_12);
                    float gate_12 = __fdiv_rn(zs_1[4], den_12);
                    float nj_12 = xs_1[4] * rstd_1;
                    float aj_12 = gs_1[4] * gate_12;
                    float dzj_12 = gs_1[4] * (nj_12 * wclaim[4]) * sig_12 * __fmaf_rn(zs_1[4], 1.0f - sig_12, 1.0f);
                    float dnj_12 = aj_12 * wclaim[4];
                    dot_1 = __fmaf_rn(dnj_12, xs_1[4], dot_1);
                    acc[4] = __fmaf_rn(aj_12 * live_1, nj_12, acc[4]);
                    zs_1[4] = dzj_12;
                    gs_1[4] = dnj_12;
                    float _exp_210 = __nv_expf(zs_1[5] * -1.0f);
                    float den_13 = 1.0f + _exp_210;
                    float sig_13 = __fdiv_rn(1.0f, den_13);
                    float gate_13 = __fdiv_rn(zs_1[5], den_13);
                    float nj_13 = xs_1[5] * rstd_1;
                    float aj_13 = gs_1[5] * gate_13;
                    float dzj_13 = gs_1[5] * (nj_13 * wclaim[5]) * sig_13 * __fmaf_rn(zs_1[5], 1.0f - sig_13, 1.0f);
                    float dnj_13 = aj_13 * wclaim[5];
                    dot_1 = __fmaf_rn(dnj_13, xs_1[5], dot_1);
                    acc[5] = __fmaf_rn(aj_13 * live_1, nj_13, acc[5]);
                    zs_1[5] = dzj_13;
                    gs_1[5] = dnj_13;
                    float _exp_211 = __nv_expf(zs_1[6] * -1.0f);
                    float den_14 = 1.0f + _exp_211;
                    float sig_14 = __fdiv_rn(1.0f, den_14);
                    float gate_14 = __fdiv_rn(zs_1[6], den_14);
                    float nj_14 = xs_1[6] * rstd_1;
                    float aj_14 = gs_1[6] * gate_14;
                    float dzj_14 = gs_1[6] * (nj_14 * wclaim[6]) * sig_14 * __fmaf_rn(zs_1[6], 1.0f - sig_14, 1.0f);
                    float dnj_14 = aj_14 * wclaim[6];
                    dot_1 = __fmaf_rn(dnj_14, xs_1[6], dot_1);
                    acc[6] = __fmaf_rn(aj_14 * live_1, nj_14, acc[6]);
                    zs_1[6] = dzj_14;
                    gs_1[6] = dnj_14;
                    float _exp_212 = __nv_expf(zs_1[7] * -1.0f);
                    float den_15 = 1.0f + _exp_212;
                    float sig_15 = __fdiv_rn(1.0f, den_15);
                    float gate_15 = __fdiv_rn(zs_1[7], den_15);
                    float nj_15 = xs_1[7] * rstd_1;
                    float aj_15 = gs_1[7] * gate_15;
                    float dzj_15 = gs_1[7] * (nj_15 * wclaim[7]) * sig_15 * __fmaf_rn(zs_1[7], 1.0f - sig_15, 1.0f);
                    float dnj_15 = aj_15 * wclaim[7];
                    dot_1 = __fmaf_rn(dnj_15, xs_1[7], dot_1);
                    acc[7] = __fmaf_rn(aj_15 * live_1, nj_15, acc[7]);
                    zs_1[7] = dzj_15;
                    gs_1[7] = dnj_15;
                }
                float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_213;
                float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_214;
                float _shfl_xor_sync_215 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_215;
                float _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
                dot_1 = dot_1 + _shfl_xor_sync_216;
                float coef_1 = rstd_1 * rstd_1 * rstd_1 * (dot_1 * -0.0078125f);
                float2 coef2_1 = make_float2(coef_1, coef_1);
                float2 rstd2b_1 = make_float2(rstd_1, rstd_1);
                float2 dx2_4 = fma_f32x2_rn_noftz(make_float2(xs_1[0], xs_1[1]), coef2_1, mul_f32x2_noftz(make_float2(gs_1[0], gs_1[1]), rstd2b_1));
                xs_1[0] = dx2_4.x;
                xs_1[1] = dx2_4.y;
                float2 dx2_5 = fma_f32x2_rn_noftz(make_float2(xs_1[2], xs_1[3]), coef2_1, mul_f32x2_noftz(make_float2(gs_1[2], gs_1[3]), rstd2b_1));
                xs_1[2] = dx2_5.x;
                xs_1[3] = dx2_5.y;
                float2 dx2_6 = fma_f32x2_rn_noftz(make_float2(xs_1[4], xs_1[5]), coef2_1, mul_f32x2_noftz(make_float2(gs_1[4], gs_1[5]), rstd2b_1));
                xs_1[4] = dx2_6.x;
                xs_1[5] = dx2_6.y;
                float2 dx2_7 = fma_f32x2_rn_noftz(make_float2(xs_1[6], xs_1[7]), coef2_1, mul_f32x2_noftz(make_float2(gs_1[6], gs_1[7]), rstd2b_1));
                xs_1[6] = dx2_7.x;
                xs_1[7] = dx2_7.y;
                unsigned int packed_dx_1[4];
                unsigned int packed_dz_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 0], xs_1[_lp*2+1 + 0]));
                    packed_dx_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 0], zs_1[_lp*2+1 + 0]));
                    packed_dz_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 2], xs_1[_lp*2+1 + 2]));
                    packed_dx_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 2], zs_1[_lp*2+1 + 2]));
                    packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 4], xs_1[_lp*2+1 + 4]));
                    packed_dx_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 4], zs_1[_lp*2+1 + 4]));
                    packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 6], xs_1[_lp*2+1 + 6]));
                    packed_dx_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs_1[_lp*2 + 6], zs_1[_lp*2+1 + 6]));
                    packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row_1 < rows) {
                    long long orow_1 = (long long)row_1 * 128 + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                    reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
                }
                run = run + 1;
                if (run == 16) {
                    if (folded != 0) {
                        fold[group * 128 + col] = fold[group * 128 + col] + (double)acc[0];
                        acc[0] = 0.0f;
                        fold[group * 128 + col + 1] = fold[group * 128 + col + 1] + (double)acc[1];
                        acc[1] = 0.0f;
                        fold[group * 128 + col + 2] = fold[group * 128 + col + 2] + (double)acc[2];
                        acc[2] = 0.0f;
                        fold[group * 128 + col + 3] = fold[group * 128 + col + 3] + (double)acc[3];
                        acc[3] = 0.0f;
                        fold[group * 128 + col + 4] = fold[group * 128 + col + 4] + (double)acc[4];
                        acc[4] = 0.0f;
                        fold[group * 128 + col + 5] = fold[group * 128 + col + 5] + (double)acc[5];
                        acc[5] = 0.0f;
                        fold[group * 128 + col + 6] = fold[group * 128 + col + 6] + (double)acc[6];
                        acc[6] = 0.0f;
                        fold[group * 128 + col + 7] = fold[group * 128 + col + 7] + (double)acc[7];
                        acc[7] = 0.0f;
                    } else {
                        fold[group * 128 + col] = (double)acc[0];
                        acc[0] = 0.0f;
                        fold[group * 128 + col + 1] = (double)acc[1];
                        acc[1] = 0.0f;
                        fold[group * 128 + col + 2] = (double)acc[2];
                        acc[2] = 0.0f;
                        fold[group * 128 + col + 3] = (double)acc[3];
                        acc[3] = 0.0f;
                        fold[group * 128 + col + 4] = (double)acc[4];
                        acc[4] = 0.0f;
                        fold[group * 128 + col + 5] = (double)acc[5];
                        acc[5] = 0.0f;
                        fold[group * 128 + col + 6] = (double)acc[6];
                        acc[6] = 0.0f;
                        fold[group * 128 + col + 7] = (double)acc[7];
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
    float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, p32_0, 16);
    p32_0 = p32_0 + _shfl_xor_sync_217;
    part32[0] = p32_0;
    float p32_1 = acc[1];
    float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, p32_1, 16);
    p32_1 = p32_1 + _shfl_xor_sync_218;
    part32[1] = p32_1;
    float p32_2 = acc[2];
    float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, p32_2, 16);
    p32_2 = p32_2 + _shfl_xor_sync_219;
    part32[2] = p32_2;
    float p32_3 = acc[3];
    float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, p32_3, 16);
    p32_3 = p32_3 + _shfl_xor_sync_220;
    part32[3] = p32_3;
    float p32_4 = acc[4];
    float _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, p32_4, 16);
    p32_4 = p32_4 + _shfl_xor_sync_221;
    part32[4] = p32_4;
    float p32_5 = acc[5];
    float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, p32_5, 16);
    p32_5 = p32_5 + _shfl_xor_sync_222;
    part32[5] = p32_5;
    float p32_6 = acc[6];
    float _shfl_xor_sync_223 = __shfl_xor_sync(0xFFFFFFFF, p32_6, 16);
    p32_6 = p32_6 + _shfl_xor_sync_223;
    part32[6] = p32_6;
    float p32_7 = acc[7];
    float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, p32_7, 16);
    p32_7 = p32_7 + _shfl_xor_sync_224;
    part32[7] = p32_7;
    int short_ = 0;
    if (folded == 0) {
        if (run <= 13) {
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
        fold[group * 128 + col] = fold[group * 128 + col] + (double)acc[0];
        acc[0] = 0.0f;
        fold[group * 128 + col + 1] = fold[group * 128 + col + 1] + (double)acc[1];
        acc[1] = 0.0f;
        fold[group * 128 + col + 2] = fold[group * 128 + col + 2] + (double)acc[2];
        acc[2] = 0.0f;
        fold[group * 128 + col + 3] = fold[group * 128 + col + 3] + (double)acc[3];
        acc[3] = 0.0f;
        fold[group * 128 + col + 4] = fold[group * 128 + col + 4] + (double)acc[4];
        acc[4] = 0.0f;
        fold[group * 128 + col + 5] = fold[group * 128 + col + 5] + (double)acc[5];
        acc[5] = 0.0f;
        fold[group * 128 + col + 6] = fold[group * 128 + col + 6] + (double)acc[6];
        acc[6] = 0.0f;
        fold[group * 128 + col + 7] = fold[group * 128 + col + 7] + (double)acc[7];
        acc[7] = 0.0f;
    } else {
        fold[group * 128 + col] = (double)acc[0];
        acc[0] = 0.0f;
        fold[group * 128 + col + 1] = (double)acc[1];
        acc[1] = 0.0f;
        fold[group * 128 + col + 2] = (double)acc[2];
        acc[2] = 0.0f;
        fold[group * 128 + col + 3] = (double)acc[3];
        acc[3] = 0.0f;
        fold[group * 128 + col + 4] = (double)acc[4];
        acc[4] = 0.0f;
        fold[group * 128 + col + 5] = (double)acc[5];
        acc[5] = 0.0f;
        fold[group * 128 + col + 6] = (double)acc[6];
        acc[6] = 0.0f;
        fold[group * 128 + col + 7] = (double)acc[7];
        acc[7] = 0.0f;
    }
    __syncthreads();
    if (weight_grad != 0) {
        if (tid < 128) {
            double colsum = 0.0;
            if (short_ != 0) {
                float w32[4];
                w32[0] = fold32[tid];
                w32[1] = fold32[128 + tid];
                w32[2] = fold32[256 + tid];
                w32[3] = fold32[384 + tid];
                w32[0] = w32[0] + w32[1];
                w32[2] = w32[2] + w32[3];
                w32[0] = w32[0] + w32[2];
                colsum = (double)w32[0];
            } else {
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    colsum = colsum + fold[g * 128 + tid];
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
