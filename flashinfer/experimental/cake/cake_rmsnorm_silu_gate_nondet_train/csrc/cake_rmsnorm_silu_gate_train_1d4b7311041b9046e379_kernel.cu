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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_1d4b7311041b9046e379(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, double* __restrict__ dw_sums, unsigned int* __restrict__ dw_done, __nv_bfloat16* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
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
            int pblk = bid;
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
    int nb = 0;
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
        flag[1] = ticket;
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
            float total = 0.0f;
            total = __fmaf_rn(xr[0], xr[0], total);
            total = __fmaf_rn(xr[1], xr[1], total);
            total = __fmaf_rn(xr[2], xr[2], total);
            total = __fmaf_rn(xr[3], xr[3], total);
            total = __fmaf_rn(xr[4], xr[4], total);
            total = __fmaf_rn(xr[5], xr[5], total);
            total = __fmaf_rn(xr[6], xr[6], total);
            total = __fmaf_rn(xr[7], xr[7], total);
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
            total = total + _shfl_xor_sync_7;
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_sync_8;
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
            total = total + _shfl_xor_sync_9;
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
            total = total + _shfl_xor_sync_10;
            float rstd;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
            float dot = 0.0f;
            int fast = 1;
            float wk[2];
            unsigned int packed_dz[4];
            if (fast != 0) {
                float2 rstd2 = make_float2(rstd, rstd);
                float2 z2_0 = make_float2(zr[0], zr[1]);
                float2 _f32x2_mul_b_11 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
                float _fma_12;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(zr[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_13;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(zr[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_0 = make_float2(_fma_12, _fma_13);
                float _exp2_14 = approx_exp2(ep2_0.x);
                float _exp2_15 = approx_exp2(ep2_0.y);
                float2 dq2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_14, _exp2_15), qr2_0, make_float2(0.25f, 0.25f));
                float _rcp_16 = approx_rcp(dq2_0.x);
                float _rcp_17 = approx_rcp(dq2_0.y);
                float2 _f32x2_mul_a_18 = make_float2(_rcp_16, _rcp_17);
                float2 _f32x2_mul_b_19 = make_float2(0.25f, 0.25f);
                float2 sig2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
                float2 gate2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&sig2_0));
                float2 g2_0 = make_float2(gr[0], gr[1]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                        : "r"(wbits[_pair]));
                }
                float2 w2_0 = make_float2(wk[0], wk[1]);
                float2 _f32x2_mul_a_20 = make_float2(xr[0], xr[1]);
                float2 n2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_21 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
                float2 _f32x2_mul_b_22 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
                float2 dz2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
                float2 a2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
                float2 dn2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
                float2 acc2_0 = fma_f32x2_rn_noftz(a2_0, n2_0, make_float2(acc_r[0], acc_r[1]));
                acc_r[0] = ((row < rows) ? acc2_0.x : acc_r[0]);
                acc_r[1] = ((row < rows) ? acc2_0.y : acc_r[1]);
                zr[0] = dz2_0.x;
                zr[1] = dz2_0.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 0], zr[_lp*2+1 + 0]));
                    packed_dz[_lp] = *(uint32_t*)&_bf2;
                }
                gr[0] = dn2_0.x;
                gr[1] = dn2_0.y;
                float2 z2_1 = make_float2(zr[2], zr[3]);
                float2 _f32x2_mul_b_23 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
                float _fma_24;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(zr[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_25;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(zr[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_1 = make_float2(_fma_24, _fma_25);
                float _exp2_26 = approx_exp2(ep2_1.x);
                float _exp2_27 = approx_exp2(ep2_1.y);
                float2 dq2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_26, _exp2_27), qr2_1, make_float2(0.25f, 0.25f));
                float _rcp_28 = approx_rcp(dq2_1.x);
                float _rcp_29 = approx_rcp(dq2_1.y);
                float2 _f32x2_mul_a_30 = make_float2(_rcp_28, _rcp_29);
                float2 _f32x2_mul_b_31 = make_float2(0.25f, 0.25f);
                float2 sig2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
                float2 gate2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&sig2_1));
                float2 g2_1 = make_float2(gr[2], gr[3]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                        : "r"(wbits[1 + _pair]));
                }
                float2 w2_1 = make_float2(wk[0], wk[1]);
                float2 _f32x2_mul_a_32 = make_float2(xr[2], xr[3]);
                float2 n2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_33 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
                float2 _f32x2_mul_b_34 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
                float2 dz2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
                float2 a2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
                float2 dn2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
                float2 acc2_1 = fma_f32x2_rn_noftz(a2_1, n2_1, make_float2(acc_r[2], acc_r[3]));
                acc_r[2] = ((row < rows) ? acc2_1.x : acc_r[2]);
                acc_r[3] = ((row < rows) ? acc2_1.y : acc_r[3]);
                zr[2] = dz2_1.x;
                zr[3] = dz2_1.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 2], zr[_lp*2+1 + 2]));
                    packed_dz[_lp + 1] = *(uint32_t*)&_bf2;
                }
                gr[2] = dn2_1.x;
                gr[3] = dn2_1.y;
                float2 z2_2 = make_float2(zr[4], zr[5]);
                float2 _f32x2_mul_b_35 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
                float _fma_36;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(zr[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_37;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(zr[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_2 = make_float2(_fma_36, _fma_37);
                float _exp2_38 = approx_exp2(ep2_2.x);
                float _exp2_39 = approx_exp2(ep2_2.y);
                float2 dq2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_38, _exp2_39), qr2_2, make_float2(0.25f, 0.25f));
                float _rcp_40 = approx_rcp(dq2_2.x);
                float _rcp_41 = approx_rcp(dq2_2.y);
                float2 _f32x2_mul_a_42 = make_float2(_rcp_40, _rcp_41);
                float2 _f32x2_mul_b_43 = make_float2(0.25f, 0.25f);
                float2 sig2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                float2 gate2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&sig2_2));
                float2 g2_2 = make_float2(gr[4], gr[5]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                        : "r"(wbits[2 + _pair]));
                }
                float2 w2_2 = make_float2(wk[0], wk[1]);
                float2 _f32x2_mul_a_44 = make_float2(xr[4], xr[5]);
                float2 n2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_45 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
                float2 _f32x2_mul_b_46 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
                float2 dz2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
                float2 a2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
                float2 dn2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
                float2 acc2_2 = fma_f32x2_rn_noftz(a2_2, n2_2, make_float2(acc_r[4], acc_r[5]));
                acc_r[4] = ((row < rows) ? acc2_2.x : acc_r[4]);
                acc_r[5] = ((row < rows) ? acc2_2.y : acc_r[5]);
                zr[4] = dz2_2.x;
                zr[5] = dz2_2.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 4], zr[_lp*2+1 + 4]));
                    packed_dz[_lp + 2] = *(uint32_t*)&_bf2;
                }
                gr[4] = dn2_2.x;
                gr[5] = dn2_2.y;
                float2 z2_3 = make_float2(zr[6], zr[7]);
                float2 _f32x2_mul_b_47 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
                float _fma_48;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(zr[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_49;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(zr[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_3 = make_float2(_fma_48, _fma_49);
                float _exp2_50 = approx_exp2(ep2_3.x);
                float _exp2_51 = approx_exp2(ep2_3.y);
                float2 dq2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_50, _exp2_51), qr2_3, make_float2(0.25f, 0.25f));
                float _rcp_52 = approx_rcp(dq2_3.x);
                float _rcp_53 = approx_rcp(dq2_3.y);
                float2 _f32x2_mul_a_54 = make_float2(_rcp_52, _rcp_53);
                float2 _f32x2_mul_b_55 = make_float2(0.25f, 0.25f);
                float2 sig2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
                float2 gate2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&sig2_3));
                float2 g2_3 = make_float2(gr[6], gr[7]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                        : "r"(wbits[3 + _pair]));
                }
                float2 w2_3 = make_float2(wk[0], wk[1]);
                float2 _f32x2_mul_a_56 = make_float2(xr[6], xr[7]);
                float2 n2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&rstd2));
                float2 _f32x2_mul_a_57 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
                float2 _f32x2_mul_b_58 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
                float2 dz2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
                float2 a2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
                float2 dn2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
                float2 acc2_3 = fma_f32x2_rn_noftz(a2_3, n2_3, make_float2(acc_r[6], acc_r[7]));
                acc_r[6] = ((row < rows) ? acc2_3.x : acc_r[6]);
                acc_r[7] = ((row < rows) ? acc2_3.y : acc_r[7]);
                zr[6] = dz2_3.x;
                zr[7] = dz2_3.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 6], zr[_lp*2+1 + 6]));
                    packed_dz[_lp + 3] = *(uint32_t*)&_bf2;
                }
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
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                        : "r"(wbits[_pair]));
                }
                float _exp_59 = __nv_expf(zr[0] * -1.0f);
                float den_0 = 1.0f + _exp_59;
                float sig_0 = __fdiv_rn(1.0f, den_0);
                float gate_0 = __fdiv_rn(zr[0], den_0);
                float nj_0 = xr[0] * rstd;
                float aj_0 = gr[0] * gate_0;
                float dzj_0 = gr[0] * (nj_0 * wk[0]) * sig_0 * __fmaf_rn(zr[0], 1.0f - sig_0, 1.0f);
                float dnj_0 = aj_0 * wk[0];
                dot = __fmaf_rn(dnj_0, xr[0], dot);
                acc_r[0] = ((row < rows) ? __fmaf_rn(aj_0, nj_0, acc_r[0]) : acc_r[0]);
                zr[0] = dzj_0;
                gr[0] = dnj_0;
                float _exp_60 = __nv_expf(zr[1] * -1.0f);
                float den_1 = 1.0f + _exp_60;
                float sig_1 = __fdiv_rn(1.0f, den_1);
                float gate_1 = __fdiv_rn(zr[1], den_1);
                float nj_1 = xr[1] * rstd;
                float aj_1 = gr[1] * gate_1;
                float dzj_1 = gr[1] * (nj_1 * wk[1]) * sig_1 * __fmaf_rn(zr[1], 1.0f - sig_1, 1.0f);
                float dnj_1 = aj_1 * wk[1];
                dot = __fmaf_rn(dnj_1, xr[1], dot);
                acc_r[1] = ((row < rows) ? __fmaf_rn(aj_1, nj_1, acc_r[1]) : acc_r[1]);
                zr[1] = dzj_1;
                gr[1] = dnj_1;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 0], zr[_lp*2+1 + 0]));
                    packed_dz[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                        : "r"(wbits[1 + _pair]));
                }
                float _exp_61 = __nv_expf(zr[2] * -1.0f);
                float den_2 = 1.0f + _exp_61;
                float sig_2 = __fdiv_rn(1.0f, den_2);
                float gate_2 = __fdiv_rn(zr[2], den_2);
                float nj_2 = xr[2] * rstd;
                float aj_2 = gr[2] * gate_2;
                float dzj_2 = gr[2] * (nj_2 * wk[0]) * sig_2 * __fmaf_rn(zr[2], 1.0f - sig_2, 1.0f);
                float dnj_2 = aj_2 * wk[0];
                dot = __fmaf_rn(dnj_2, xr[2], dot);
                acc_r[2] = ((row < rows) ? __fmaf_rn(aj_2, nj_2, acc_r[2]) : acc_r[2]);
                zr[2] = dzj_2;
                gr[2] = dnj_2;
                float _exp_62 = __nv_expf(zr[3] * -1.0f);
                float den_3 = 1.0f + _exp_62;
                float sig_3 = __fdiv_rn(1.0f, den_3);
                float gate_3 = __fdiv_rn(zr[3], den_3);
                float nj_3 = xr[3] * rstd;
                float aj_3 = gr[3] * gate_3;
                float dzj_3 = gr[3] * (nj_3 * wk[1]) * sig_3 * __fmaf_rn(zr[3], 1.0f - sig_3, 1.0f);
                float dnj_3 = aj_3 * wk[1];
                dot = __fmaf_rn(dnj_3, xr[3], dot);
                acc_r[3] = ((row < rows) ? __fmaf_rn(aj_3, nj_3, acc_r[3]) : acc_r[3]);
                zr[3] = dzj_3;
                gr[3] = dnj_3;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 2], zr[_lp*2+1 + 2]));
                    packed_dz[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                        : "r"(wbits[2 + _pair]));
                }
                float _exp_63 = __nv_expf(zr[4] * -1.0f);
                float den_4 = 1.0f + _exp_63;
                float sig_4 = __fdiv_rn(1.0f, den_4);
                float gate_4 = __fdiv_rn(zr[4], den_4);
                float nj_4 = xr[4] * rstd;
                float aj_4 = gr[4] * gate_4;
                float dzj_4 = gr[4] * (nj_4 * wk[0]) * sig_4 * __fmaf_rn(zr[4], 1.0f - sig_4, 1.0f);
                float dnj_4 = aj_4 * wk[0];
                dot = __fmaf_rn(dnj_4, xr[4], dot);
                acc_r[4] = ((row < rows) ? __fmaf_rn(aj_4, nj_4, acc_r[4]) : acc_r[4]);
                zr[4] = dzj_4;
                gr[4] = dnj_4;
                float _exp_64 = __nv_expf(zr[5] * -1.0f);
                float den_5 = 1.0f + _exp_64;
                float sig_5 = __fdiv_rn(1.0f, den_5);
                float gate_5 = __fdiv_rn(zr[5], den_5);
                float nj_5 = xr[5] * rstd;
                float aj_5 = gr[5] * gate_5;
                float dzj_5 = gr[5] * (nj_5 * wk[1]) * sig_5 * __fmaf_rn(zr[5], 1.0f - sig_5, 1.0f);
                float dnj_5 = aj_5 * wk[1];
                dot = __fmaf_rn(dnj_5, xr[5], dot);
                acc_r[5] = ((row < rows) ? __fmaf_rn(aj_5, nj_5, acc_r[5]) : acc_r[5]);
                zr[5] = dzj_5;
                gr[5] = dnj_5;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 4], zr[_lp*2+1 + 4]));
                    packed_dz[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk[_pair * 2])[0]), "=f"((&wk[_pair * 2])[1])
                        : "r"(wbits[3 + _pair]));
                }
                float _exp_65 = __nv_expf(zr[6] * -1.0f);
                float den_6 = 1.0f + _exp_65;
                float sig_6 = __fdiv_rn(1.0f, den_6);
                float gate_6 = __fdiv_rn(zr[6], den_6);
                float nj_6 = xr[6] * rstd;
                float aj_6 = gr[6] * gate_6;
                float dzj_6 = gr[6] * (nj_6 * wk[0]) * sig_6 * __fmaf_rn(zr[6], 1.0f - sig_6, 1.0f);
                float dnj_6 = aj_6 * wk[0];
                dot = __fmaf_rn(dnj_6, xr[6], dot);
                acc_r[6] = ((row < rows) ? __fmaf_rn(aj_6, nj_6, acc_r[6]) : acc_r[6]);
                zr[6] = dzj_6;
                gr[6] = dnj_6;
                float _exp_66 = __nv_expf(zr[7] * -1.0f);
                float den_7 = 1.0f + _exp_66;
                float sig_7 = __fdiv_rn(1.0f, den_7);
                float gate_7 = __fdiv_rn(zr[7], den_7);
                float nj_7 = xr[7] * rstd;
                float aj_7 = gr[7] * gate_7;
                float dzj_7 = gr[7] * (nj_7 * wk[1]) * sig_7 * __fmaf_rn(zr[7], 1.0f - sig_7, 1.0f);
                float dnj_7 = aj_7 * wk[1];
                dot = __fmaf_rn(dnj_7, xr[7], dot);
                acc_r[7] = ((row < rows) ? __fmaf_rn(aj_7, nj_7, acc_r[7]) : acc_r[7]);
                zr[7] = dzj_7;
                gr[7] = dnj_7;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr[_lp*2 + 6], zr[_lp*2+1 + 6]));
                    packed_dz[_lp + 3] = *(uint32_t*)&_bf2;
                }
            }
            float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
            dot = dot + _shfl_xor_sync_67;
            float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
            dot = dot + _shfl_xor_sync_68;
            float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
            dot = dot + _shfl_xor_sync_69;
            float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
            dot = dot + _shfl_xor_sync_70;
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
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 0], xr[_lp*2+1 + 0]));
                packed_dx[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 2], xr[_lp*2+1 + 2]));
                packed_dx[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 4], xr[_lp*2+1 + 4]));
                packed_dx[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr[_lp*2 + 6], xr[_lp*2+1 + 6]));
                packed_dx[_lp + 3] = *(uint32_t*)&_bf2;
            }
            if (row < rows) {
                long long orow = (long long)row * 128ll + (long long)col;
                reinterpret_cast<int4*>(dx + orow)[0] = reinterpret_cast<int4*>(packed_dx)[0];
                reinterpret_cast<int4*>(dz + orow)[0] = reinterpret_cast<int4*>(packed_dz)[0];
            }
            nb = nb + 1;
            if ((nb & 15) == 0) {
                if (nb - 1 >= 16) {
                    float s;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                    float bv;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv) : "f"(s), "f"(fold[group * 128 + col]));
                    float av;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av) : "f"(s), "f"(bv));
                    float _sub_71;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(fold[group * 128 + col]), "f"(av));
                    float _sub_72;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(acc_r[0]), "f"(bv));
                    float e;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_71), "f"(_sub_72));
                    fold[group * 128 + col] = s;
                    float _add_73;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(fold[(16 + group) * 128 + col]), "f"(e));
                    fold[(16 + group) * 128 + col] = _add_73;
                    acc_r[0] = 0.0f;
                    float s_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_0) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                    float bv_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_0) : "f"(s_0), "f"(fold[group * 128 + col + 1]));
                    float av_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_0) : "f"(s_0), "f"(bv_0));
                    float _sub_74;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(fold[group * 128 + col + 1]), "f"(av_0));
                    float _sub_75;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(acc_r[1]), "f"(bv_0));
                    float e_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_74), "f"(_sub_75));
                    fold[group * 128 + col + 1] = s_0;
                    float _add_76;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_0));
                    fold[(16 + group) * 128 + col + 1] = _add_76;
                    acc_r[1] = 0.0f;
                    float s_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                    float bv_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_1) : "f"(s_1), "f"(fold[group * 128 + col + 2]));
                    float av_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_1) : "f"(s_1), "f"(bv_1));
                    float _sub_77;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(fold[group * 128 + col + 2]), "f"(av_1));
                    float _sub_78;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(acc_r[2]), "f"(bv_1));
                    float e_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_77), "f"(_sub_78));
                    fold[group * 128 + col + 2] = s_1;
                    float _add_79;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_1));
                    fold[(16 + group) * 128 + col + 2] = _add_79;
                    acc_r[2] = 0.0f;
                    float s_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_2) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                    float bv_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_2) : "f"(s_2), "f"(fold[group * 128 + col + 3]));
                    float av_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_2) : "f"(s_2), "f"(bv_2));
                    float _sub_80;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(fold[group * 128 + col + 3]), "f"(av_2));
                    float _sub_81;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(acc_r[3]), "f"(bv_2));
                    float e_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_80), "f"(_sub_81));
                    fold[group * 128 + col + 3] = s_2;
                    float _add_82;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_2));
                    fold[(16 + group) * 128 + col + 3] = _add_82;
                    acc_r[3] = 0.0f;
                    float s_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_3) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                    float bv_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_3) : "f"(s_3), "f"(fold[group * 128 + col + 4]));
                    float av_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_3) : "f"(s_3), "f"(bv_3));
                    float _sub_83;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(fold[group * 128 + col + 4]), "f"(av_3));
                    float _sub_84;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(acc_r[4]), "f"(bv_3));
                    float e_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_83), "f"(_sub_84));
                    fold[group * 128 + col + 4] = s_3;
                    float _add_85;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_3));
                    fold[(16 + group) * 128 + col + 4] = _add_85;
                    acc_r[4] = 0.0f;
                    float s_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_4) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                    float bv_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_4) : "f"(s_4), "f"(fold[group * 128 + col + 5]));
                    float av_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_4) : "f"(s_4), "f"(bv_4));
                    float _sub_86;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(fold[group * 128 + col + 5]), "f"(av_4));
                    float _sub_87;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(acc_r[5]), "f"(bv_4));
                    float e_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_4) : "f"(_sub_86), "f"(_sub_87));
                    fold[group * 128 + col + 5] = s_4;
                    float _add_88;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_4));
                    fold[(16 + group) * 128 + col + 5] = _add_88;
                    acc_r[5] = 0.0f;
                    float s_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_5) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                    float bv_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_5) : "f"(s_5), "f"(fold[group * 128 + col + 6]));
                    float av_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_5) : "f"(s_5), "f"(bv_5));
                    float _sub_89;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_89) : "f"(fold[group * 128 + col + 6]), "f"(av_5));
                    float _sub_90;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(acc_r[6]), "f"(bv_5));
                    float e_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_5) : "f"(_sub_89), "f"(_sub_90));
                    fold[group * 128 + col + 6] = s_5;
                    float _add_91;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_5));
                    fold[(16 + group) * 128 + col + 6] = _add_91;
                    acc_r[6] = 0.0f;
                    float s_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_6) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                    float bv_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_6) : "f"(s_6), "f"(fold[group * 128 + col + 7]));
                    float av_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_6) : "f"(s_6), "f"(bv_6));
                    float _sub_92;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_92) : "f"(fold[group * 128 + col + 7]), "f"(av_6));
                    float _sub_93;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(acc_r[7]), "f"(bv_6));
                    float e_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_6) : "f"(_sub_92), "f"(_sub_93));
                    fold[group * 128 + col + 7] = s_6;
                    float _add_94;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_6));
                    fold[(16 + group) * 128 + col + 7] = _add_94;
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
                if (nb == 16384) {
                    if (weight_grad != 0) {
                        #pragma unroll 1
                        for (int j = 0; j < 8; j++) {
                            float fs = fold[group * 128 + col + j];
                            double part = (double)fs;
                            float _abs_95;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_95) : "f"(fs));
                            if (_abs_95 <= 3.4028235e+38f) {
                                part = part + (double)fold[(16 + group) * 128 + col + j];
                            }
                            if (part != 0.0) {
                                atomicAdd(&dw_sums[(bid & 3) * 128 + col + j], part);
                            }
                        }
                    }
                    nb = 0;
                }
            }
        }
    }
    int trips = 0;
    int ahead = 0;
    if (rest > 0) {
        int _min_96 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_96) > (1) ? (_min_96) : (1));
        int t_share = (rest + t_sl - 1) / t_sl;
#if __CUDA_ARCH__ == 1070
        if (t_share <= 64) {
#else
        if (t_share <= 96) {
#endif
            ahead = 1;
        }
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
    int _min_97 = ((num_bids) < (rest) ? (num_bids) : (rest));
    int stride = ((_min_97) > (1) ? (_min_97) : (1));
    #pragma unroll 1
    for (int it = 0; it < trips; it++) {
        int half = it % 2 * 2;
        if (tid == 0) {
            int tk = flag[1];
            flag[1] = 2147483647;
            int begin = 0;
            int count = 0;
            int e_first_0 = tab[0];
            int e_base_0 = tab[1];
            int e_size_0 = tab[2];
            begin = ((e_first_0 <= tk) ? e_base_0 + (tk - e_first_0) : begin);
            count = ((e_first_0 <= tk) ? e_size_0 : count);
            int e_first_1 = tab[4];
            int e_base_1 = tab[5];
            int e_size_1 = tab[6];
            begin = ((e_first_1 <= tk) ? e_base_1 + (tk - e_first_1) : begin);
            count = ((e_first_1 <= tk) ? e_size_1 : count);
            int e_first_2 = tab[8];
            int e_base_2 = tab[9];
            int e_size_2 = tab[10];
            begin = ((e_first_2 <= tk) ? e_base_2 + (tk - e_first_2) : begin);
            count = ((e_first_2 <= tk) ? e_size_2 : count);
            int e_first_3 = tab[12];
            int e_base_3 = tab[13];
            int e_size_3 = tab[14];
            begin = ((e_first_3 <= tk) ? e_base_3 + (tk - e_first_3) : begin);
            count = ((e_first_3 <= tk) ? e_size_3 : count);
            int e_first_4 = tab[16];
            int e_base_4 = tab[17];
            int e_size_4 = tab[18];
            begin = ((e_first_4 <= tk) ? e_base_4 + (tk - e_first_4) : begin);
            count = ((e_first_4 <= tk) ? e_size_4 : count);
            int e_first_5 = tab[20];
            int e_base_5 = tab[21];
            int e_size_5 = tab[22];
            begin = ((e_first_5 <= tk) ? e_base_5 + (tk - e_first_5) : begin);
            count = ((e_first_5 <= tk) ? e_size_5 : count);
            int e_first_6 = tab[24];
            int e_base_6 = tab[25];
            int e_size_6 = tab[26];
            begin = ((e_first_6 <= tk) ? e_base_6 + (tk - e_first_6) : begin);
            count = ((e_first_6 <= tk) ? e_size_6 : count);
            int e_first_7 = tab[28];
            int e_base_7 = tab[29];
            int e_size_7 = tab[30];
            begin = ((e_first_7 <= tk) ? e_base_7 + (tk - e_first_7) : begin);
            count = ((e_first_7 <= tk) ? e_size_7 : count);
            cell[half] = begin;
            cell[half + 1] = count;
        }
        __syncthreads();
        int chunk_lo = cell[half];
        int chunk_n = cell[half + 1];
        if (chunk_n > 0) {
            #pragma unroll 1
            for (int i_1 = 0; i_1 < chunk_n; i_1++) {
                int u_1 = bound + chunk_lo + i_1 * stride;
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
                    const uint4* _vptr_98 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                    uint4 _vld_98[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_98[_blk] = _vptr_98[_blk];
                        uint32_t* _vpairs_98 = reinterpret_cast<uint32_t*>(&_vld_98[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_98[_pair]) << 16);
                            (&gr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_98[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_99 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                    uint4 _vld_99[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_99[_blk] = _vptr_99[_blk];
                        uint32_t* _vpairs_99 = reinterpret_cast<uint32_t*>(&_vld_99[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_99[_pair]) << 16);
                            (&xr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_99[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_100 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                    uint4 _vld_100[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_100[_blk] = _vptr_100[_blk];
                        uint32_t* _vpairs_100 = reinterpret_cast<uint32_t*>(&_vld_100[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_100[_pair]) << 16);
                            (&zr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_100[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (ahead != 0) {
                    if (i_1 + 1 < chunk_n) {
                        if (packed != 0) {
                            if (tid == 0) {
                                int nfirst = (u_1 + stride) * 16;
                                int _min_101 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                                int nbytes = _min_101 * 256;
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                            }
                        } else {
                            int nrow = (((u_1 + stride) * 16 + group) < (rows - 1) ? ((u_1 + stride) * 16 + group) : (rows - 1));
                            uint32_t nq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow);
                            uint32_t nh = (uint32_t)((unsigned int)nrow) - nq * (uint32_t)(heads.divisor);
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq * dy_tok + (long long)nh * dy_head + (long long)col))));
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq * x_tok + (long long)nh * x_head + (long long)col))));
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq * z_tok + (long long)nh * z_head + (long long)col))));
                        }
                    }
                }
                if (tid == 0) {
                    if (trips - 1 - it > 0) {
                        if (i_1 + 1 == chunk_n) {
                            uint32_t _atomic_fetch_inc_102;
                            asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                                : "=r"(_atomic_fetch_inc_102) : "l"(&dw_done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                            flag[1] = (int)_atomic_fetch_inc_102;
                        }
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
                float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
                total_1 = total_1 + _shfl_xor_sync_103;
                float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
                total_1 = total_1 + _shfl_xor_sync_104;
                float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
                total_1 = total_1 + _shfl_xor_sync_105;
                float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
                total_1 = total_1 + _shfl_xor_sync_106;
                float rstd_1;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_1) : "f"(total_1 * 0.0078125f + eps));
                float dot_1 = 0.0f;
                int fast_1 = 1;
                float wk_1[2];
                unsigned int packed_dz_1[4];
                if (fast_1 != 0) {
                    float2 rstd2_1 = make_float2(rstd_1, rstd_1);
                    float2 z2_4 = make_float2(zr_1[0], zr_1[1]);
                    float2 _f32x2_mul_b_107 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_107));
                    float _fma_108;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(zr_1[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_109;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(zr_1[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_4 = make_float2(_fma_108, _fma_109);
                    float _exp2_110 = approx_exp2(ep2_4.x);
                    float _exp2_111 = approx_exp2(ep2_4.y);
                    float2 dq2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_110, _exp2_111), qr2_4, make_float2(0.25f, 0.25f));
                    float _rcp_112 = approx_rcp(dq2_4.x);
                    float _rcp_113 = approx_rcp(dq2_4.y);
                    float2 _f32x2_mul_a_114 = make_float2(_rcp_112, _rcp_113);
                    float2 _f32x2_mul_b_115 = make_float2(0.25f, 0.25f);
                    float2 sig2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&_f32x2_mul_b_115));
                    float2 gate2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&sig2_4));
                    float2 g2_4 = make_float2(gr_1[0], gr_1[1]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                            : "r"(wbits[_pair]));
                    }
                    float2 w2_4 = make_float2(wk_1[0], wk_1[1]);
                    float2 _f32x2_mul_a_116 = make_float2(xr_1[0], xr_1[1]);
                    float2 n2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_117 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
                    float2 _f32x2_mul_b_118 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
                    float2 dz2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
                    float2 a2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&gate2_4));
                    float2 dn2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_4) : "l"(*(const unsigned long long*)&a2_4), "l"(*(const unsigned long long*)&w2_4));
                    float2 acc2_4 = fma_f32x2_rn_noftz(a2_4, n2_4, make_float2(acc_r[0], acc_r[1]));
                    acc_r[0] = ((row_1 < rows) ? acc2_4.x : acc_r[0]);
                    acc_r[1] = ((row_1 < rows) ? acc2_4.y : acc_r[1]);
                    zr_1[0] = dz2_4.x;
                    zr_1[1] = dz2_4.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 0], zr_1[_lp*2+1 + 0]));
                        packed_dz_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    gr_1[0] = dn2_4.x;
                    gr_1[1] = dn2_4.y;
                    float2 z2_5 = make_float2(zr_1[2], zr_1[3]);
                    float2 _f32x2_mul_b_119 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_119));
                    float _fma_120;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(zr_1[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_121;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(zr_1[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_5 = make_float2(_fma_120, _fma_121);
                    float _exp2_122 = approx_exp2(ep2_5.x);
                    float _exp2_123 = approx_exp2(ep2_5.y);
                    float2 dq2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_122, _exp2_123), qr2_5, make_float2(0.25f, 0.25f));
                    float _rcp_124 = approx_rcp(dq2_5.x);
                    float _rcp_125 = approx_rcp(dq2_5.y);
                    float2 _f32x2_mul_a_126 = make_float2(_rcp_124, _rcp_125);
                    float2 _f32x2_mul_b_127 = make_float2(0.25f, 0.25f);
                    float2 sig2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&_f32x2_mul_b_127));
                    float2 gate2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&sig2_5));
                    float2 g2_5 = make_float2(gr_1[2], gr_1[3]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                            : "r"(wbits[1 + _pair]));
                    }
                    float2 w2_5 = make_float2(wk_1[0], wk_1[1]);
                    float2 _f32x2_mul_a_128 = make_float2(xr_1[2], xr_1[3]);
                    float2 n2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_129 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
                    float2 _f32x2_mul_b_130 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
                    float2 dz2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&_f32x2_mul_b_130));
                    float2 a2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&gate2_5));
                    float2 dn2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_5) : "l"(*(const unsigned long long*)&a2_5), "l"(*(const unsigned long long*)&w2_5));
                    float2 acc2_5 = fma_f32x2_rn_noftz(a2_5, n2_5, make_float2(acc_r[2], acc_r[3]));
                    acc_r[2] = ((row_1 < rows) ? acc2_5.x : acc_r[2]);
                    acc_r[3] = ((row_1 < rows) ? acc2_5.y : acc_r[3]);
                    zr_1[2] = dz2_5.x;
                    zr_1[3] = dz2_5.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 2], zr_1[_lp*2+1 + 2]));
                        packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    gr_1[2] = dn2_5.x;
                    gr_1[3] = dn2_5.y;
                    float2 z2_6 = make_float2(zr_1[4], zr_1[5]);
                    float2 _f32x2_mul_b_131 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
                    float _fma_132;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(zr_1[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_133;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(zr_1[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_6 = make_float2(_fma_132, _fma_133);
                    float _exp2_134 = approx_exp2(ep2_6.x);
                    float _exp2_135 = approx_exp2(ep2_6.y);
                    float2 dq2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_134, _exp2_135), qr2_6, make_float2(0.25f, 0.25f));
                    float _rcp_136 = approx_rcp(dq2_6.x);
                    float _rcp_137 = approx_rcp(dq2_6.y);
                    float2 _f32x2_mul_a_138 = make_float2(_rcp_136, _rcp_137);
                    float2 _f32x2_mul_b_139 = make_float2(0.25f, 0.25f);
                    float2 sig2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
                    float2 gate2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&sig2_6));
                    float2 g2_6 = make_float2(gr_1[4], gr_1[5]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                            : "r"(wbits[2 + _pair]));
                    }
                    float2 w2_6 = make_float2(wk_1[0], wk_1[1]);
                    float2 _f32x2_mul_a_140 = make_float2(xr_1[4], xr_1[5]);
                    float2 n2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_141 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
                    float2 _f32x2_mul_b_142 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
                    float2 dz2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&_f32x2_mul_b_142));
                    float2 a2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&gate2_6));
                    float2 dn2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_6) : "l"(*(const unsigned long long*)&a2_6), "l"(*(const unsigned long long*)&w2_6));
                    float2 acc2_6 = fma_f32x2_rn_noftz(a2_6, n2_6, make_float2(acc_r[4], acc_r[5]));
                    acc_r[4] = ((row_1 < rows) ? acc2_6.x : acc_r[4]);
                    acc_r[5] = ((row_1 < rows) ? acc2_6.y : acc_r[5]);
                    zr_1[4] = dz2_6.x;
                    zr_1[5] = dz2_6.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 4], zr_1[_lp*2+1 + 4]));
                        packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    gr_1[4] = dn2_6.x;
                    gr_1[5] = dn2_6.y;
                    float2 z2_7 = make_float2(zr_1[6], zr_1[7]);
                    float2 _f32x2_mul_b_143 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
                    float _fma_144;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(zr_1[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_145;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(zr_1[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_7 = make_float2(_fma_144, _fma_145);
                    float _exp2_146 = approx_exp2(ep2_7.x);
                    float _exp2_147 = approx_exp2(ep2_7.y);
                    float2 dq2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_146, _exp2_147), qr2_7, make_float2(0.25f, 0.25f));
                    float _rcp_148 = approx_rcp(dq2_7.x);
                    float _rcp_149 = approx_rcp(dq2_7.y);
                    float2 _f32x2_mul_a_150 = make_float2(_rcp_148, _rcp_149);
                    float2 _f32x2_mul_b_151 = make_float2(0.25f, 0.25f);
                    float2 sig2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
                    float2 gate2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&sig2_7));
                    float2 g2_7 = make_float2(gr_1[6], gr_1[7]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                            : "r"(wbits[3 + _pair]));
                    }
                    float2 w2_7 = make_float2(wk_1[0], wk_1[1]);
                    float2 _f32x2_mul_a_152 = make_float2(xr_1[6], xr_1[7]);
                    float2 n2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_153 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
                    float2 _f32x2_mul_b_154 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
                    float2 dz2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&_f32x2_mul_b_154));
                    float2 a2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&gate2_7));
                    float2 dn2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_7) : "l"(*(const unsigned long long*)&a2_7), "l"(*(const unsigned long long*)&w2_7));
                    float2 acc2_7 = fma_f32x2_rn_noftz(a2_7, n2_7, make_float2(acc_r[6], acc_r[7]));
                    acc_r[6] = ((row_1 < rows) ? acc2_7.x : acc_r[6]);
                    acc_r[7] = ((row_1 < rows) ? acc2_7.y : acc_r[7]);
                    zr_1[6] = dz2_7.x;
                    zr_1[7] = dz2_7.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 6], zr_1[_lp*2+1 + 6]));
                        packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
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
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                            : "r"(wbits[_pair]));
                    }
                    float _exp_155 = __nv_expf(zr_1[0] * -1.0f);
                    float den_8 = 1.0f + _exp_155;
                    float sig_8 = __fdiv_rn(1.0f, den_8);
                    float gate_8 = __fdiv_rn(zr_1[0], den_8);
                    float nj_8 = xr_1[0] * rstd_1;
                    float aj_8 = gr_1[0] * gate_8;
                    float dzj_8 = gr_1[0] * (nj_8 * wk_1[0]) * sig_8 * __fmaf_rn(zr_1[0], 1.0f - sig_8, 1.0f);
                    float dnj_8 = aj_8 * wk_1[0];
                    dot_1 = __fmaf_rn(dnj_8, xr_1[0], dot_1);
                    acc_r[0] = ((row_1 < rows) ? __fmaf_rn(aj_8, nj_8, acc_r[0]) : acc_r[0]);
                    zr_1[0] = dzj_8;
                    gr_1[0] = dnj_8;
                    float _exp_156 = __nv_expf(zr_1[1] * -1.0f);
                    float den_9 = 1.0f + _exp_156;
                    float sig_9 = __fdiv_rn(1.0f, den_9);
                    float gate_9 = __fdiv_rn(zr_1[1], den_9);
                    float nj_9 = xr_1[1] * rstd_1;
                    float aj_9 = gr_1[1] * gate_9;
                    float dzj_9 = gr_1[1] * (nj_9 * wk_1[1]) * sig_9 * __fmaf_rn(zr_1[1], 1.0f - sig_9, 1.0f);
                    float dnj_9 = aj_9 * wk_1[1];
                    dot_1 = __fmaf_rn(dnj_9, xr_1[1], dot_1);
                    acc_r[1] = ((row_1 < rows) ? __fmaf_rn(aj_9, nj_9, acc_r[1]) : acc_r[1]);
                    zr_1[1] = dzj_9;
                    gr_1[1] = dnj_9;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 0], zr_1[_lp*2+1 + 0]));
                        packed_dz_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                            : "r"(wbits[1 + _pair]));
                    }
                    float _exp_157 = __nv_expf(zr_1[2] * -1.0f);
                    float den_10 = 1.0f + _exp_157;
                    float sig_10 = __fdiv_rn(1.0f, den_10);
                    float gate_10 = __fdiv_rn(zr_1[2], den_10);
                    float nj_10 = xr_1[2] * rstd_1;
                    float aj_10 = gr_1[2] * gate_10;
                    float dzj_10 = gr_1[2] * (nj_10 * wk_1[0]) * sig_10 * __fmaf_rn(zr_1[2], 1.0f - sig_10, 1.0f);
                    float dnj_10 = aj_10 * wk_1[0];
                    dot_1 = __fmaf_rn(dnj_10, xr_1[2], dot_1);
                    acc_r[2] = ((row_1 < rows) ? __fmaf_rn(aj_10, nj_10, acc_r[2]) : acc_r[2]);
                    zr_1[2] = dzj_10;
                    gr_1[2] = dnj_10;
                    float _exp_158 = __nv_expf(zr_1[3] * -1.0f);
                    float den_11 = 1.0f + _exp_158;
                    float sig_11 = __fdiv_rn(1.0f, den_11);
                    float gate_11 = __fdiv_rn(zr_1[3], den_11);
                    float nj_11 = xr_1[3] * rstd_1;
                    float aj_11 = gr_1[3] * gate_11;
                    float dzj_11 = gr_1[3] * (nj_11 * wk_1[1]) * sig_11 * __fmaf_rn(zr_1[3], 1.0f - sig_11, 1.0f);
                    float dnj_11 = aj_11 * wk_1[1];
                    dot_1 = __fmaf_rn(dnj_11, xr_1[3], dot_1);
                    acc_r[3] = ((row_1 < rows) ? __fmaf_rn(aj_11, nj_11, acc_r[3]) : acc_r[3]);
                    zr_1[3] = dzj_11;
                    gr_1[3] = dnj_11;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 2], zr_1[_lp*2+1 + 2]));
                        packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                            : "r"(wbits[2 + _pair]));
                    }
                    float _exp_159 = __nv_expf(zr_1[4] * -1.0f);
                    float den_12 = 1.0f + _exp_159;
                    float sig_12 = __fdiv_rn(1.0f, den_12);
                    float gate_12 = __fdiv_rn(zr_1[4], den_12);
                    float nj_12 = xr_1[4] * rstd_1;
                    float aj_12 = gr_1[4] * gate_12;
                    float dzj_12 = gr_1[4] * (nj_12 * wk_1[0]) * sig_12 * __fmaf_rn(zr_1[4], 1.0f - sig_12, 1.0f);
                    float dnj_12 = aj_12 * wk_1[0];
                    dot_1 = __fmaf_rn(dnj_12, xr_1[4], dot_1);
                    acc_r[4] = ((row_1 < rows) ? __fmaf_rn(aj_12, nj_12, acc_r[4]) : acc_r[4]);
                    zr_1[4] = dzj_12;
                    gr_1[4] = dnj_12;
                    float _exp_160 = __nv_expf(zr_1[5] * -1.0f);
                    float den_13 = 1.0f + _exp_160;
                    float sig_13 = __fdiv_rn(1.0f, den_13);
                    float gate_13 = __fdiv_rn(zr_1[5], den_13);
                    float nj_13 = xr_1[5] * rstd_1;
                    float aj_13 = gr_1[5] * gate_13;
                    float dzj_13 = gr_1[5] * (nj_13 * wk_1[1]) * sig_13 * __fmaf_rn(zr_1[5], 1.0f - sig_13, 1.0f);
                    float dnj_13 = aj_13 * wk_1[1];
                    dot_1 = __fmaf_rn(dnj_13, xr_1[5], dot_1);
                    acc_r[5] = ((row_1 < rows) ? __fmaf_rn(aj_13, nj_13, acc_r[5]) : acc_r[5]);
                    zr_1[5] = dzj_13;
                    gr_1[5] = dnj_13;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 4], zr_1[_lp*2+1 + 4]));
                        packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_1[_pair * 2])[0]), "=f"((&wk_1[_pair * 2])[1])
                            : "r"(wbits[3 + _pair]));
                    }
                    float _exp_161 = __nv_expf(zr_1[6] * -1.0f);
                    float den_14 = 1.0f + _exp_161;
                    float sig_14 = __fdiv_rn(1.0f, den_14);
                    float gate_14 = __fdiv_rn(zr_1[6], den_14);
                    float nj_14 = xr_1[6] * rstd_1;
                    float aj_14 = gr_1[6] * gate_14;
                    float dzj_14 = gr_1[6] * (nj_14 * wk_1[0]) * sig_14 * __fmaf_rn(zr_1[6], 1.0f - sig_14, 1.0f);
                    float dnj_14 = aj_14 * wk_1[0];
                    dot_1 = __fmaf_rn(dnj_14, xr_1[6], dot_1);
                    acc_r[6] = ((row_1 < rows) ? __fmaf_rn(aj_14, nj_14, acc_r[6]) : acc_r[6]);
                    zr_1[6] = dzj_14;
                    gr_1[6] = dnj_14;
                    float _exp_162 = __nv_expf(zr_1[7] * -1.0f);
                    float den_15 = 1.0f + _exp_162;
                    float sig_15 = __fdiv_rn(1.0f, den_15);
                    float gate_15 = __fdiv_rn(zr_1[7], den_15);
                    float nj_15 = xr_1[7] * rstd_1;
                    float aj_15 = gr_1[7] * gate_15;
                    float dzj_15 = gr_1[7] * (nj_15 * wk_1[1]) * sig_15 * __fmaf_rn(zr_1[7], 1.0f - sig_15, 1.0f);
                    float dnj_15 = aj_15 * wk_1[1];
                    dot_1 = __fmaf_rn(dnj_15, xr_1[7], dot_1);
                    acc_r[7] = ((row_1 < rows) ? __fmaf_rn(aj_15, nj_15, acc_r[7]) : acc_r[7]);
                    zr_1[7] = dzj_15;
                    gr_1[7] = dnj_15;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_1[_lp*2 + 6], zr_1[_lp*2+1 + 6]));
                        packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                }
                float _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_163;
                float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_164;
                float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_165;
                float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
                dot_1 = dot_1 + _shfl_xor_sync_166;
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
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 0], xr_1[_lp*2+1 + 0]));
                    packed_dx_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 2], xr_1[_lp*2+1 + 2]));
                    packed_dx_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 4], xr_1[_lp*2+1 + 4]));
                    packed_dx_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_1[_lp*2 + 6], xr_1[_lp*2+1 + 6]));
                    packed_dx_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row_1 < rows) {
                    long long orow_1 = (long long)row_1 * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                    reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
                }
                nb = nb + 1;
                if ((nb & 15) == 0) {
                    if (nb - 1 >= 16) {
                        float s_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        float bv_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7) : "f"(s_7), "f"(fold[group * 128 + col]));
                        float av_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7) : "f"(s_7), "f"(bv_7));
                        float _sub_167;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_167) : "f"(fold[group * 128 + col]), "f"(av_7));
                        float _sub_168;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(acc_r[0]), "f"(bv_7));
                        float e_7;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7) : "f"(_sub_167), "f"(_sub_168));
                        fold[group * 128 + col] = s_7;
                        float _add_169;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(fold[(16 + group) * 128 + col]), "f"(e_7));
                        fold[(16 + group) * 128 + col] = _add_169;
                        acc_r[0] = 0.0f;
                        float s_7_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7_1) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        float bv_7_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7_1) : "f"(s_7_1), "f"(fold[group * 128 + col + 1]));
                        float av_7_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7_1) : "f"(s_7_1), "f"(bv_7_1));
                        float _sub_170;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_170) : "f"(fold[group * 128 + col + 1]), "f"(av_7_1));
                        float _sub_171;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_171) : "f"(acc_r[1]), "f"(bv_7_1));
                        float e_7_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7_1) : "f"(_sub_170), "f"(_sub_171));
                        fold[group * 128 + col + 1] = s_7_1;
                        float _add_172;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_7_1));
                        fold[(16 + group) * 128 + col + 1] = _add_172;
                        acc_r[1] = 0.0f;
                        float s_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_8) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        float bv_8;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_8) : "f"(s_8), "f"(fold[group * 128 + col + 2]));
                        float av_8;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_8) : "f"(s_8), "f"(bv_8));
                        float _sub_173;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_173) : "f"(fold[group * 128 + col + 2]), "f"(av_8));
                        float _sub_174;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_174) : "f"(acc_r[2]), "f"(bv_8));
                        float e_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_8) : "f"(_sub_173), "f"(_sub_174));
                        fold[group * 128 + col + 2] = s_8;
                        float _add_175;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_8));
                        fold[(16 + group) * 128 + col + 2] = _add_175;
                        acc_r[2] = 0.0f;
                        float s_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        float bv_9;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_9) : "f"(s_9), "f"(fold[group * 128 + col + 3]));
                        float av_9;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_9) : "f"(s_9), "f"(bv_9));
                        float _sub_176;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_176) : "f"(fold[group * 128 + col + 3]), "f"(av_9));
                        float _sub_177;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_177) : "f"(acc_r[3]), "f"(bv_9));
                        float e_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_9) : "f"(_sub_176), "f"(_sub_177));
                        fold[group * 128 + col + 3] = s_9;
                        float _add_178;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_9));
                        fold[(16 + group) * 128 + col + 3] = _add_178;
                        acc_r[3] = 0.0f;
                        float s_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        float bv_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_10) : "f"(s_10), "f"(fold[group * 128 + col + 4]));
                        float av_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_10) : "f"(s_10), "f"(bv_10));
                        float _sub_179;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_179) : "f"(fold[group * 128 + col + 4]), "f"(av_10));
                        float _sub_180;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_180) : "f"(acc_r[4]), "f"(bv_10));
                        float e_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_10) : "f"(_sub_179), "f"(_sub_180));
                        fold[group * 128 + col + 4] = s_10;
                        float _add_181;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_10));
                        fold[(16 + group) * 128 + col + 4] = _add_181;
                        acc_r[4] = 0.0f;
                        float s_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        float bv_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_11) : "f"(s_11), "f"(fold[group * 128 + col + 5]));
                        float av_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_11) : "f"(s_11), "f"(bv_11));
                        float _sub_182;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_182) : "f"(fold[group * 128 + col + 5]), "f"(av_11));
                        float _sub_183;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_183) : "f"(acc_r[5]), "f"(bv_11));
                        float e_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_11) : "f"(_sub_182), "f"(_sub_183));
                        fold[group * 128 + col + 5] = s_11;
                        float _add_184;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_11));
                        fold[(16 + group) * 128 + col + 5] = _add_184;
                        acc_r[5] = 0.0f;
                        float s_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        float bv_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_12) : "f"(s_12), "f"(fold[group * 128 + col + 6]));
                        float av_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_12) : "f"(s_12), "f"(bv_12));
                        float _sub_185;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_185) : "f"(fold[group * 128 + col + 6]), "f"(av_12));
                        float _sub_186;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_186) : "f"(acc_r[6]), "f"(bv_12));
                        float e_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_12) : "f"(_sub_185), "f"(_sub_186));
                        fold[group * 128 + col + 6] = s_12;
                        float _add_187;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_12));
                        fold[(16 + group) * 128 + col + 6] = _add_187;
                        acc_r[6] = 0.0f;
                        float s_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        float bv_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_13) : "f"(s_13), "f"(fold[group * 128 + col + 7]));
                        float av_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_13) : "f"(s_13), "f"(bv_13));
                        float _sub_188;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_188) : "f"(fold[group * 128 + col + 7]), "f"(av_13));
                        float _sub_189;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_189) : "f"(acc_r[7]), "f"(bv_13));
                        float e_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_13) : "f"(_sub_188), "f"(_sub_189));
                        fold[group * 128 + col + 7] = s_13;
                        float _add_190;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_13));
                        fold[(16 + group) * 128 + col + 7] = _add_190;
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
                    if (nb == 16384) {
                        if (weight_grad != 0) {
                            #pragma unroll 1
                            for (int j_1 = 0; j_1 < 8; j_1++) {
                                float fs_1 = fold[group * 128 + col + j_1];
                                double part_1 = (double)fs_1;
                                float _abs_191;
                                asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(fs_1));
                                if (_abs_191 <= 3.4028235e+38f) {
                                    part_1 = part_1 + (double)fold[(16 + group) * 128 + col + j_1];
                                }
                                if (part_1 != 0.0) {
                                    atomicAdd(&dw_sums[(bid & 3) * 128 + col + j_1], part_1);
                                }
                            }
                        }
                        nb = 0;
                    }
                }
            }
        }
    }
    float part32[8];
    float p32_0 = acc_r[0];
    float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, p32_0, 16);
    p32_0 = p32_0 + _shfl_xor_sync_192;
    part32[0] = p32_0;
    float p32_1 = acc_r[1];
    float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, p32_1, 16);
    p32_1 = p32_1 + _shfl_xor_sync_193;
    part32[1] = p32_1;
    float p32_2 = acc_r[2];
    float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, p32_2, 16);
    p32_2 = p32_2 + _shfl_xor_sync_194;
    part32[2] = p32_2;
    float p32_3 = acc_r[3];
    float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, p32_3, 16);
    p32_3 = p32_3 + _shfl_xor_sync_195;
    part32[3] = p32_3;
    float p32_4 = acc_r[4];
    float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, p32_4, 16);
    p32_4 = p32_4 + _shfl_xor_sync_196;
    part32[4] = p32_4;
    float p32_5 = acc_r[5];
    float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, p32_5, 16);
    p32_5 = p32_5 + _shfl_xor_sync_197;
    part32[5] = p32_5;
    float p32_6 = acc_r[6];
    float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, p32_6, 16);
    p32_6 = p32_6 + _shfl_xor_sync_198;
    part32[6] = p32_6;
    float p32_7 = acc_r[7];
    float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, p32_7, 16);
    p32_7 = p32_7 + _shfl_xor_sync_199;
    part32[7] = p32_7;
    int short_ = 0;
    if (nb <= 12) {
        short_ = 1;
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
    } else if ((nb & 15) != 0) {
        if (nb - 1 >= 16) {
            float s_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
            float bv_14;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14) : "f"(s_14), "f"(fold[group * 128 + col]));
            float av_14;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14) : "f"(s_14), "f"(bv_14));
            float _sub_200;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_200) : "f"(fold[group * 128 + col]), "f"(av_14));
            float _sub_201;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_201) : "f"(acc_r[0]), "f"(bv_14));
            float e_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14) : "f"(_sub_200), "f"(_sub_201));
            fold[group * 128 + col] = s_14;
            float _add_202;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(fold[(16 + group) * 128 + col]), "f"(e_14));
            fold[(16 + group) * 128 + col] = _add_202;
            acc_r[0] = 0.0f;
            float s_14_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14_1) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
            float bv_14_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14_1) : "f"(s_14_1), "f"(fold[group * 128 + col + 1]));
            float av_14_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14_1) : "f"(s_14_1), "f"(bv_14_1));
            float _sub_203;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_203) : "f"(fold[group * 128 + col + 1]), "f"(av_14_1));
            float _sub_204;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_204) : "f"(acc_r[1]), "f"(bv_14_1));
            float e_14_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14_1) : "f"(_sub_203), "f"(_sub_204));
            fold[group * 128 + col + 1] = s_14_1;
            float _add_205;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_14_1));
            fold[(16 + group) * 128 + col + 1] = _add_205;
            acc_r[1] = 0.0f;
            float s_15;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
            float bv_15;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_15) : "f"(s_15), "f"(fold[group * 128 + col + 2]));
            float av_15;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_15) : "f"(s_15), "f"(bv_15));
            float _sub_206;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_206) : "f"(fold[group * 128 + col + 2]), "f"(av_15));
            float _sub_207;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_207) : "f"(acc_r[2]), "f"(bv_15));
            float e_15;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_15) : "f"(_sub_206), "f"(_sub_207));
            fold[group * 128 + col + 2] = s_15;
            float _add_208;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_15));
            fold[(16 + group) * 128 + col + 2] = _add_208;
            acc_r[2] = 0.0f;
            float s_16;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
            float bv_16;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_16) : "f"(s_16), "f"(fold[group * 128 + col + 3]));
            float av_16;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_16) : "f"(s_16), "f"(bv_16));
            float _sub_209;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_209) : "f"(fold[group * 128 + col + 3]), "f"(av_16));
            float _sub_210;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_210) : "f"(acc_r[3]), "f"(bv_16));
            float e_16;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_16) : "f"(_sub_209), "f"(_sub_210));
            fold[group * 128 + col + 3] = s_16;
            float _add_211;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_16));
            fold[(16 + group) * 128 + col + 3] = _add_211;
            acc_r[3] = 0.0f;
            float s_17;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
            float bv_17;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_17) : "f"(s_17), "f"(fold[group * 128 + col + 4]));
            float av_17;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_17) : "f"(s_17), "f"(bv_17));
            float _sub_212;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_212) : "f"(fold[group * 128 + col + 4]), "f"(av_17));
            float _sub_213;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_213) : "f"(acc_r[4]), "f"(bv_17));
            float e_17;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_17) : "f"(_sub_212), "f"(_sub_213));
            fold[group * 128 + col + 4] = s_17;
            float _add_214;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_17));
            fold[(16 + group) * 128 + col + 4] = _add_214;
            acc_r[4] = 0.0f;
            float s_18;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
            float bv_18;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_18) : "f"(s_18), "f"(fold[group * 128 + col + 5]));
            float av_18;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_18) : "f"(s_18), "f"(bv_18));
            float _sub_215;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_215) : "f"(fold[group * 128 + col + 5]), "f"(av_18));
            float _sub_216;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_216) : "f"(acc_r[5]), "f"(bv_18));
            float e_18;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_18) : "f"(_sub_215), "f"(_sub_216));
            fold[group * 128 + col + 5] = s_18;
            float _add_217;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_18));
            fold[(16 + group) * 128 + col + 5] = _add_217;
            acc_r[5] = 0.0f;
            float s_19;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
            float bv_19;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_19) : "f"(s_19), "f"(fold[group * 128 + col + 6]));
            float av_19;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_19) : "f"(s_19), "f"(bv_19));
            float _sub_218;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_218) : "f"(fold[group * 128 + col + 6]), "f"(av_19));
            float _sub_219;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_219) : "f"(acc_r[6]), "f"(bv_19));
            float e_19;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_19) : "f"(_sub_218), "f"(_sub_219));
            fold[group * 128 + col + 6] = s_19;
            float _add_220;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_19));
            fold[(16 + group) * 128 + col + 6] = _add_220;
            acc_r[6] = 0.0f;
            float s_20;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
            float bv_20;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_20) : "f"(s_20), "f"(fold[group * 128 + col + 7]));
            float av_20;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_20) : "f"(s_20), "f"(bv_20));
            float _sub_221;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_221) : "f"(fold[group * 128 + col + 7]), "f"(av_20));
            float _sub_222;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_222) : "f"(acc_r[7]), "f"(bv_20));
            float e_20;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_20) : "f"(_sub_221), "f"(_sub_222));
            fold[group * 128 + col + 7] = s_20;
            float _add_223;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_20));
            fold[(16 + group) * 128 + col + 7] = _add_223;
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
                    float _sub_224;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_224) : "f"(hs), "f"(av_21));
                    float _sub_225;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_225) : "f"(fold[g * 128 + tid]), "f"(bv_21));
                    float e_21;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_21) : "f"(_sub_224), "f"(_sub_225));
                    hs = s_21;
                    float _add_227;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(e_21), "f"(fold[(16 + g) * 128 + tid]));
                    float _add_226;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(ls), "f"(_add_227));
                    ls = _add_226;
                }
                float _abs_228;
                asm("abs.f32 %0, %1;" : "=f"(_abs_228) : "f"(hs));
                if (_abs_228 <= 3.4028235e+38f) {
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
                double got_0 = atomicAdd(&dw_sums[tid], -0.0);
                dw_total = dw_total + got_0;
                dw_sums[tid] = 0.0;
                double got_1 = atomicAdd(&dw_sums[128 + tid], -0.0);
                dw_total = dw_total + got_1;
                dw_sums[128 + tid] = 0.0;
                double got_2 = atomicAdd(&dw_sums[256 + tid], -0.0);
                dw_total = dw_total + got_2;
                dw_sums[256 + tid] = 0.0;
                double got_3 = atomicAdd(&dw_sums[384 + tid], -0.0);
                dw_total = dw_total + got_3;
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
