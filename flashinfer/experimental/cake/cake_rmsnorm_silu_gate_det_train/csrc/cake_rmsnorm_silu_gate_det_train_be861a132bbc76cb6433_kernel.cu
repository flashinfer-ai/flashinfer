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
kernel_cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, unsigned long long* __restrict__ acc, unsigned int* __restrict__ flags, unsigned int* __restrict__ done, float* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad)
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
    unsigned int la_x[4];
    unsigned int la_g[4];
    unsigned int la_z[4];
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
        if (packed != 0) {
            int tail_i = 0;
            int u = bid + tail_i;
            int row = u * 16 + group;
            int src = ((row) < (rows - 1) ? (row) : (rows - 1));
            long long goff = (long long)src * 128ll + (long long)col;
            long long xoff = goff;
            long long zoff = goff;
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
            float wk[2];
            unsigned int packed_dz[4];
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
            float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
            dot = dot + _shfl_xor_sync_59;
            float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
            dot = dot + _shfl_xor_sync_60;
            float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
            dot = dot + _shfl_xor_sync_61;
            float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
            dot = dot + _shfl_xor_sync_62;
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
            long long orow = goff;
            if (row < rows) {
                reinterpret_cast<int4*>(dx + orow)[0] = reinterpret_cast<int4*>(packed_dx)[0];
                reinterpret_cast<int4*>(dz + orow)[0] = reinterpret_cast<int4*>(packed_dz)[0];
            }
            if ((tail_i & 15) == 15) {
                if (tail_i >= 16) {
                    float s;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                    float bv;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv) : "f"(s), "f"(fold[group * 128 + col]));
                    float av;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av) : "f"(s), "f"(bv));
                    float _sub_63;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(fold[group * 128 + col]), "f"(av));
                    float _sub_64;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(acc_r[0]), "f"(bv));
                    float e;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_63), "f"(_sub_64));
                    fold[group * 128 + col] = s;
                    float _add_65;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(fold[(16 + group) * 128 + col]), "f"(e));
                    fold[(16 + group) * 128 + col] = _add_65;
                    acc_r[0] = 0.0f;
                    float s_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_0) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                    float bv_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_0) : "f"(s_0), "f"(fold[group * 128 + col + 1]));
                    float av_0;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_0) : "f"(s_0), "f"(bv_0));
                    float _sub_66;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(fold[group * 128 + col + 1]), "f"(av_0));
                    float _sub_67;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(acc_r[1]), "f"(bv_0));
                    float e_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_66), "f"(_sub_67));
                    fold[group * 128 + col + 1] = s_0;
                    float _add_68;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_0));
                    fold[(16 + group) * 128 + col + 1] = _add_68;
                    acc_r[1] = 0.0f;
                    float s_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_1) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                    float bv_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_1) : "f"(s_1), "f"(fold[group * 128 + col + 2]));
                    float av_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_1) : "f"(s_1), "f"(bv_1));
                    float _sub_69;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(fold[group * 128 + col + 2]), "f"(av_1));
                    float _sub_70;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(acc_r[2]), "f"(bv_1));
                    float e_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_69), "f"(_sub_70));
                    fold[group * 128 + col + 2] = s_1;
                    float _add_71;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_1));
                    fold[(16 + group) * 128 + col + 2] = _add_71;
                    acc_r[2] = 0.0f;
                    float s_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_2) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                    float bv_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_2) : "f"(s_2), "f"(fold[group * 128 + col + 3]));
                    float av_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_2) : "f"(s_2), "f"(bv_2));
                    float _sub_72;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(fold[group * 128 + col + 3]), "f"(av_2));
                    float _sub_73;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(acc_r[3]), "f"(bv_2));
                    float e_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_72), "f"(_sub_73));
                    fold[group * 128 + col + 3] = s_2;
                    float _add_74;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_2));
                    fold[(16 + group) * 128 + col + 3] = _add_74;
                    acc_r[3] = 0.0f;
                    float s_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_3) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                    float bv_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_3) : "f"(s_3), "f"(fold[group * 128 + col + 4]));
                    float av_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_3) : "f"(s_3), "f"(bv_3));
                    float _sub_75;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(fold[group * 128 + col + 4]), "f"(av_3));
                    float _sub_76;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(acc_r[4]), "f"(bv_3));
                    float e_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_75), "f"(_sub_76));
                    fold[group * 128 + col + 4] = s_3;
                    float _add_77;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_3));
                    fold[(16 + group) * 128 + col + 4] = _add_77;
                    acc_r[4] = 0.0f;
                    float s_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_4) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                    float bv_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_4) : "f"(s_4), "f"(fold[group * 128 + col + 5]));
                    float av_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_4) : "f"(s_4), "f"(bv_4));
                    float _sub_78;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(fold[group * 128 + col + 5]), "f"(av_4));
                    float _sub_79;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(acc_r[5]), "f"(bv_4));
                    float e_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_4) : "f"(_sub_78), "f"(_sub_79));
                    fold[group * 128 + col + 5] = s_4;
                    float _add_80;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_4));
                    fold[(16 + group) * 128 + col + 5] = _add_80;
                    acc_r[5] = 0.0f;
                    float s_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_5) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                    float bv_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_5) : "f"(s_5), "f"(fold[group * 128 + col + 6]));
                    float av_5;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_5) : "f"(s_5), "f"(bv_5));
                    float _sub_81;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(fold[group * 128 + col + 6]), "f"(av_5));
                    float _sub_82;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(acc_r[6]), "f"(bv_5));
                    float e_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_5) : "f"(_sub_81), "f"(_sub_82));
                    fold[group * 128 + col + 6] = s_5;
                    float _add_83;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_5));
                    fold[(16 + group) * 128 + col + 6] = _add_83;
                    acc_r[6] = 0.0f;
                    float s_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_6) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                    float bv_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_6) : "f"(s_6), "f"(fold[group * 128 + col + 7]));
                    float av_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_6) : "f"(s_6), "f"(bv_6));
                    float _sub_84;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(fold[group * 128 + col + 7]), "f"(av_6));
                    float _sub_85;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(acc_r[7]), "f"(bv_6));
                    float e_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_6) : "f"(_sub_84), "f"(_sub_85));
                    fold[group * 128 + col + 7] = s_6;
                    float _add_86;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_6));
                    fold[(16 + group) * 128 + col + 7] = _add_86;
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
            {
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
        } else {
            int tail_i_1 = 0;
            int u_1 = bid + tail_i_1;
            int row_1 = u_1 * 16 + group;
            int src_1 = ((row_1) < (rows - 1) ? (row_1) : (rows - 1));
            uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_1), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_1);
            uint32_t th = (uint32_t)((unsigned int)src_1) - tq * (uint32_t)(heads.divisor);
            long long goff_1 = (long long)tq * dy_tok + (long long)th * dy_head + (long long)col;
            long long xoff_1 = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
            long long zoff_1 = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
            float gr_1[8];
            float xr_1[8];
            float zr_1[8];
            {
                const uint4* _vptr_87 = reinterpret_cast<const uint4*>(dy + goff_1 + 0);
                uint4 _vld_87[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_87[_blk] = _vptr_87[_blk];
                    uint32_t* _vpairs_87 = reinterpret_cast<uint32_t*>(&_vld_87[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_87[_pair]) << 16);
                        (&gr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_87[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_88 = reinterpret_cast<const uint4*>(x + xoff_1 + 0);
                uint4 _vld_88[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_88[_blk] = _vptr_88[_blk];
                    uint32_t* _vpairs_88 = reinterpret_cast<uint32_t*>(&_vld_88[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) << 16);
                        (&xr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) & 0xffff0000u);
                    }
                }
            }
            {
                const uint4* _vptr_89 = reinterpret_cast<const uint4*>(z + zoff_1 + 0);
                uint4 _vld_89[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_89[_blk] = _vptr_89[_blk];
                    uint32_t* _vpairs_89 = reinterpret_cast<uint32_t*>(&_vld_89[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) << 16);
                        (&zr_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) & 0xffff0000u);
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
            float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
            total_1 = total_1 + _shfl_xor_sync_90;
            float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
            total_1 = total_1 + _shfl_xor_sync_91;
            float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
            total_1 = total_1 + _shfl_xor_sync_92;
            float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
            total_1 = total_1 + _shfl_xor_sync_93;
            float rstd_1;
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_1) : "f"(total_1 * 0.0078125f + eps));
            float dot_1 = 0.0f;
            float wk_1[2];
            unsigned int packed_dz_1[4];
            float2 rstd2_1 = make_float2(rstd_1, rstd_1);
            float2 z2_4 = make_float2(zr_1[0], zr_1[1]);
            float2 _f32x2_mul_b_94 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
            float _fma_95;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(zr_1[0]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_96;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(zr_1[1]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_4 = make_float2(_fma_95, _fma_96);
            float _exp2_97 = approx_exp2(ep2_4.x);
            float _exp2_98 = approx_exp2(ep2_4.y);
            float2 dq2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_97, _exp2_98), qr2_4, make_float2(0.25f, 0.25f));
            float _rcp_99 = approx_rcp(dq2_4.x);
            float _rcp_100 = approx_rcp(dq2_4.y);
            float2 _f32x2_mul_a_101 = make_float2(_rcp_99, _rcp_100);
            float2 _f32x2_mul_b_102 = make_float2(0.25f, 0.25f);
            float2 sig2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_102));
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
            float2 _f32x2_mul_a_103 = make_float2(xr_1[0], xr_1[1]);
            float2 n2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_104 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
            float2 _f32x2_mul_b_105 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
            float2 dz2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
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
            float2 _f32x2_mul_b_106 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_106));
            float _fma_107;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(zr_1[2]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_108;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(zr_1[3]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_5 = make_float2(_fma_107, _fma_108);
            float _exp2_109 = approx_exp2(ep2_5.x);
            float _exp2_110 = approx_exp2(ep2_5.y);
            float2 dq2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_109, _exp2_110), qr2_5, make_float2(0.25f, 0.25f));
            float _rcp_111 = approx_rcp(dq2_5.x);
            float _rcp_112 = approx_rcp(dq2_5.y);
            float2 _f32x2_mul_a_113 = make_float2(_rcp_111, _rcp_112);
            float2 _f32x2_mul_b_114 = make_float2(0.25f, 0.25f);
            float2 sig2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
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
            float2 _f32x2_mul_a_115 = make_float2(xr_1[2], xr_1[3]);
            float2 n2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_116 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
            float2 _f32x2_mul_b_117 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
            float2 dz2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&_f32x2_mul_b_117));
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
            float2 _f32x2_mul_b_118 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
            float _fma_119;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(zr_1[4]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_120;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(zr_1[5]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_6 = make_float2(_fma_119, _fma_120);
            float _exp2_121 = approx_exp2(ep2_6.x);
            float _exp2_122 = approx_exp2(ep2_6.y);
            float2 dq2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_121, _exp2_122), qr2_6, make_float2(0.25f, 0.25f));
            float _rcp_123 = approx_rcp(dq2_6.x);
            float _rcp_124 = approx_rcp(dq2_6.y);
            float2 _f32x2_mul_a_125 = make_float2(_rcp_123, _rcp_124);
            float2 _f32x2_mul_b_126 = make_float2(0.25f, 0.25f);
            float2 sig2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&_f32x2_mul_b_126));
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
            float2 _f32x2_mul_a_127 = make_float2(xr_1[4], xr_1[5]);
            float2 n2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_128 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
            float2 _f32x2_mul_b_129 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
            float2 dz2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
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
            float2 _f32x2_mul_b_130 = make_float2(-1.442688f, -1.442688f);
            float2 ep2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_130));
            float _fma_131;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(zr_1[6]), "f"(-1.2221238e-06f), "f"(0.25f));
            float _fma_132;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(zr_1[7]), "f"(-1.2221238e-06f), "f"(0.25f));
            float2 qr2_7 = make_float2(_fma_131, _fma_132);
            float _exp2_133 = approx_exp2(ep2_7.x);
            float _exp2_134 = approx_exp2(ep2_7.y);
            float2 dq2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_133, _exp2_134), qr2_7, make_float2(0.25f, 0.25f));
            float _rcp_135 = approx_rcp(dq2_7.x);
            float _rcp_136 = approx_rcp(dq2_7.y);
            float2 _f32x2_mul_a_137 = make_float2(_rcp_135, _rcp_136);
            float2 _f32x2_mul_b_138 = make_float2(0.25f, 0.25f);
            float2 sig2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&_f32x2_mul_b_138));
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
            float2 _f32x2_mul_a_139 = make_float2(xr_1[6], xr_1[7]);
            float2 n2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&rstd2_1));
            float2 _f32x2_mul_a_140 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
            float2 _f32x2_mul_b_141 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
            float2 dz2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
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
            float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
            dot_1 = dot_1 + _shfl_xor_sync_142;
            float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
            dot_1 = dot_1 + _shfl_xor_sync_143;
            float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
            dot_1 = dot_1 + _shfl_xor_sync_144;
            float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
            dot_1 = dot_1 + _shfl_xor_sync_145;
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
            long long orow_1 = (long long)row_1 * 128ll + (long long)col;
            if (row_1 < rows) {
                reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
            }
            if ((tail_i_1 & 15) == 15) {
                if (tail_i_1 >= 16) {
                    float s_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_7) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                    float bv_7;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_7) : "f"(s_7), "f"(fold[group * 128 + col]));
                    float av_7;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_7) : "f"(s_7), "f"(bv_7));
                    float _sub_146;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_146) : "f"(fold[group * 128 + col]), "f"(av_7));
                    float _sub_147;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_147) : "f"(acc_r[0]), "f"(bv_7));
                    float e_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_7) : "f"(_sub_146), "f"(_sub_147));
                    fold[group * 128 + col] = s_7;
                    float _add_148;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(fold[(16 + group) * 128 + col]), "f"(e_7));
                    fold[(16 + group) * 128 + col] = _add_148;
                    acc_r[0] = 0.0f;
                    float s_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_14) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                    float bv_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_14) : "f"(s_14), "f"(fold[group * 128 + col + 1]));
                    float av_14;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_14) : "f"(s_14), "f"(bv_14));
                    float _sub_149;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_149) : "f"(fold[group * 128 + col + 1]), "f"(av_14));
                    float _sub_150;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_150) : "f"(acc_r[1]), "f"(bv_14));
                    float e_14;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_14) : "f"(_sub_149), "f"(_sub_150));
                    fold[group * 128 + col + 1] = s_14;
                    float _add_151;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_14));
                    fold[(16 + group) * 128 + col + 1] = _add_151;
                    acc_r[1] = 0.0f;
                    float s_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_15) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                    float bv_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_15) : "f"(s_15), "f"(fold[group * 128 + col + 2]));
                    float av_15;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_15) : "f"(s_15), "f"(bv_15));
                    float _sub_152;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_152) : "f"(fold[group * 128 + col + 2]), "f"(av_15));
                    float _sub_153;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_153) : "f"(acc_r[2]), "f"(bv_15));
                    float e_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_15) : "f"(_sub_152), "f"(_sub_153));
                    fold[group * 128 + col + 2] = s_15;
                    float _add_154;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_15));
                    fold[(16 + group) * 128 + col + 2] = _add_154;
                    acc_r[2] = 0.0f;
                    float s_16;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_16) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                    float bv_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_16) : "f"(s_16), "f"(fold[group * 128 + col + 3]));
                    float av_16;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_16) : "f"(s_16), "f"(bv_16));
                    float _sub_155;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_155) : "f"(fold[group * 128 + col + 3]), "f"(av_16));
                    float _sub_156;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_156) : "f"(acc_r[3]), "f"(bv_16));
                    float e_16;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_16) : "f"(_sub_155), "f"(_sub_156));
                    fold[group * 128 + col + 3] = s_16;
                    float _add_157;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_16));
                    fold[(16 + group) * 128 + col + 3] = _add_157;
                    acc_r[3] = 0.0f;
                    float s_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_17) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                    float bv_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_17) : "f"(s_17), "f"(fold[group * 128 + col + 4]));
                    float av_17;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_17) : "f"(s_17), "f"(bv_17));
                    float _sub_158;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_158) : "f"(fold[group * 128 + col + 4]), "f"(av_17));
                    float _sub_159;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_159) : "f"(acc_r[4]), "f"(bv_17));
                    float e_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_17) : "f"(_sub_158), "f"(_sub_159));
                    fold[group * 128 + col + 4] = s_17;
                    float _add_160;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_17));
                    fold[(16 + group) * 128 + col + 4] = _add_160;
                    acc_r[4] = 0.0f;
                    float s_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_18) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                    float bv_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_18) : "f"(s_18), "f"(fold[group * 128 + col + 5]));
                    float av_18;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_18) : "f"(s_18), "f"(bv_18));
                    float _sub_161;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_161) : "f"(fold[group * 128 + col + 5]), "f"(av_18));
                    float _sub_162;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_162) : "f"(acc_r[5]), "f"(bv_18));
                    float e_18;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_18) : "f"(_sub_161), "f"(_sub_162));
                    fold[group * 128 + col + 5] = s_18;
                    float _add_163;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_18));
                    fold[(16 + group) * 128 + col + 5] = _add_163;
                    acc_r[5] = 0.0f;
                    float s_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_19) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                    float bv_19;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_19) : "f"(s_19), "f"(fold[group * 128 + col + 6]));
                    float av_19;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_19) : "f"(s_19), "f"(bv_19));
                    float _sub_164;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_164) : "f"(fold[group * 128 + col + 6]), "f"(av_19));
                    float _sub_165;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_165) : "f"(acc_r[6]), "f"(bv_19));
                    float e_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_19) : "f"(_sub_164), "f"(_sub_165));
                    fold[group * 128 + col + 6] = s_19;
                    float _add_166;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_19));
                    fold[(16 + group) * 128 + col + 6] = _add_166;
                    acc_r[6] = 0.0f;
                    float s_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_20) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                    float bv_20;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_20) : "f"(s_20), "f"(fold[group * 128 + col + 7]));
                    float av_20;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_20) : "f"(s_20), "f"(bv_20));
                    float _sub_167;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_167) : "f"(fold[group * 128 + col + 7]), "f"(av_20));
                    float _sub_168;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_168) : "f"(acc_r[7]), "f"(bv_20));
                    float e_20;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_20) : "f"(_sub_167), "f"(_sub_168));
                    fold[group * 128 + col + 7] = s_20;
                    float _add_169;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_20));
                    fold[(16 + group) * 128 + col + 7] = _add_169;
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
            {
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
        if (tid < 128) {
            float hs = 0.0f;
            float ls = 0.0f;
            #pragma unroll
            for (int g = 0; g < 16; g++) {
                float s_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(s_8) : "f"(hs), "f"(fold[g * 128 + tid]));
                float bv_8;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_8) : "f"(s_8), "f"(hs));
                float av_8;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_8) : "f"(s_8), "f"(bv_8));
                float _sub_170;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_170) : "f"(hs), "f"(av_8));
                float _sub_171;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_171) : "f"(fold[g * 128 + tid]), "f"(bv_8));
                float e_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(e_8) : "f"(_sub_170), "f"(_sub_171));
                hs = s_8;
                float _add_173;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(e_8), "f"(fold[(16 + g) * 128 + tid]));
                float _add_172;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(ls), "f"(_add_173));
                ls = _add_172;
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
                    unsigned int _max_174 = ((ex) > (1) ? (ex) : (1));
                    unsigned int pos = _max_174 - 1;
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
            float _abs_175;
            asm("abs.f32 %0, %1;" : "=f"(_abs_175) : "f"(hs));
            if (_abs_175 <= 3.4028235e+38f) {
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
                        unsigned int _max_176 = ((ex_0) > (1) ? (ex_0) : (1));
                        unsigned int pos_1 = _max_176 - 1;
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
    }
    int trips = 0;
    int ahead = 0;
    if (rest > 0) {
        int _min_177 = ((num_bids) < (rest) ? (num_bids) : (rest));
        int t_sl = ((_min_177) > (1) ? (_min_177) : (1));
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
    if (tid == 0) {
        flag[1] = ticket;
    }
    int _min_178 = ((num_bids) < (rest) ? (num_bids) : (rest));
    int stride = ((_min_178) > (1) ? (_min_178) : (1));
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
            if (packed != 0) {
                if (chunk_n > 1) {
                    int vrow = (bound + chunk_lo) * 16 + group;
                    long long voff = (long long)vrow * 128ll + (long long)col;
                    {
                        uint4 _uv4_179 = *reinterpret_cast<const uint4*>(dy + voff + 0);
                        la_g[0 + 0] = _uv4_179.x;
                        la_g[0 + 1] = _uv4_179.y;
                        la_g[0 + 2] = _uv4_179.z;
                        la_g[0 + 3] = _uv4_179.w;
                    }
                    {
                        uint4 _uv4_180 = *reinterpret_cast<const uint4*>(x + voff + 0);
                        la_x[0 + 0] = _uv4_180.x;
                        la_x[0 + 1] = _uv4_180.y;
                        la_x[0 + 2] = _uv4_180.z;
                        la_x[0 + 3] = _uv4_180.w;
                    }
                    {
                        uint4 _uv4_181 = *reinterpret_cast<const uint4*>(z + voff + 0);
                        la_z[0 + 0] = _uv4_181.x;
                        la_z[0 + 1] = _uv4_181.y;
                        la_z[0 + 2] = _uv4_181.z;
                        la_z[0 + 3] = _uv4_181.w;
                    }
                }
                #pragma unroll 1
                for (int i = 0; i < chunk_n - 1; i++) {
                    int u_2 = bound + chunk_lo + i * stride;
                    int row_2 = u_2 * 16 + group;
                    long long goff_2 = (long long)row_2 * 128ll + (long long)col;
                    float gr_2[8];
                    float xr_2[8];
                    float zr_2[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_x[_pair]) << 16);
                        (&xr_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_x[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_g[_pair]) << 16);
                        (&gr_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_g[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_z[_pair]) << 16);
                        (&zr_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_z[_pair]) & 0xffff0000u);
                    }
                    if (i + 2 < chunk_n) {
                        int vrow_1 = (u_2 + stride) * 16 + group;
                        long long voff_1 = (long long)vrow_1 * 128ll + (long long)col;
                        {
                            uint4 _uv4_182 = *reinterpret_cast<const uint4*>(dy + voff_1 + 0);
                            la_g[0 + 0] = _uv4_182.x;
                            la_g[0 + 1] = _uv4_182.y;
                            la_g[0 + 2] = _uv4_182.z;
                            la_g[0 + 3] = _uv4_182.w;
                        }
                        {
                            uint4 _uv4_183 = *reinterpret_cast<const uint4*>(x + voff_1 + 0);
                            la_x[0 + 0] = _uv4_183.x;
                            la_x[0 + 1] = _uv4_183.y;
                            la_x[0 + 2] = _uv4_183.z;
                            la_x[0 + 3] = _uv4_183.w;
                        }
                        {
                            uint4 _uv4_184 = *reinterpret_cast<const uint4*>(z + voff_1 + 0);
                            la_z[0 + 0] = _uv4_184.x;
                            la_z[0 + 1] = _uv4_184.y;
                            la_z[0 + 2] = _uv4_184.z;
                            la_z[0 + 3] = _uv4_184.w;
                        }
                    }
                    if (ahead != 0) {
                        if (tid == 0) {
                            int nfirst = (u_2 + stride) * 16;
                            int _min_185 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                            int nbytes = _min_185 * 256;
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(dy + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                        }
                    }
                    float total_2 = 0.0f;
                    total_2 = __fmaf_rn(xr_2[0], xr_2[0], total_2);
                    total_2 = __fmaf_rn(xr_2[1], xr_2[1], total_2);
                    total_2 = __fmaf_rn(xr_2[2], xr_2[2], total_2);
                    total_2 = __fmaf_rn(xr_2[3], xr_2[3], total_2);
                    total_2 = __fmaf_rn(xr_2[4], xr_2[4], total_2);
                    total_2 = __fmaf_rn(xr_2[5], xr_2[5], total_2);
                    total_2 = __fmaf_rn(xr_2[6], xr_2[6], total_2);
                    total_2 = __fmaf_rn(xr_2[7], xr_2[7], total_2);
                    float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, total_2, 1);
                    total_2 = total_2 + _shfl_xor_sync_186;
                    float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, total_2, 2);
                    total_2 = total_2 + _shfl_xor_sync_187;
                    float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, total_2, 4);
                    total_2 = total_2 + _shfl_xor_sync_188;
                    float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, total_2, 8);
                    total_2 = total_2 + _shfl_xor_sync_189;
                    float rstd_2;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_2) : "f"(total_2 * 0.0078125f + eps));
                    float dot_2 = 0.0f;
                    float wk_2[2];
                    unsigned int packed_dz_2[4];
                    float2 rstd2_2 = make_float2(rstd_2, rstd_2);
                    float2 z2_8 = make_float2(zr_2[0], zr_2[1]);
                    float2 _f32x2_mul_b_190 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_8) : "l"(*(const unsigned long long*)&z2_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_190));
                    float _fma_191;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(zr_2[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_192;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(zr_2[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_8 = make_float2(_fma_191, _fma_192);
                    float _exp2_193 = approx_exp2(ep2_8.x);
                    float _exp2_194 = approx_exp2(ep2_8.y);
                    float2 dq2_8 = fma_f32x2_rn_noftz(make_float2(_exp2_193, _exp2_194), qr2_8, make_float2(0.25f, 0.25f));
                    float _rcp_195 = approx_rcp(dq2_8.x);
                    float _rcp_196 = approx_rcp(dq2_8.y);
                    float2 _f32x2_mul_a_197 = make_float2(_rcp_195, _rcp_196);
                    float2 _f32x2_mul_b_198 = make_float2(0.25f, 0.25f);
                    float2 sig2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&_f32x2_mul_b_198));
                    float2 gate2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_8) : "l"(*(const unsigned long long*)&z2_8), "l"(*(const unsigned long long*)&sig2_8));
                    float2 g2_8 = make_float2(gr_2[0], gr_2[1]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_2[_pair * 2])[0]), "=f"((&wk_2[_pair * 2])[1])
                            : "r"(wbits[_pair]));
                    }
                    float2 w2_8 = make_float2(wk_2[0], wk_2[1]);
                    float2 _f32x2_mul_a_199 = make_float2(xr_2[0], xr_2[1]);
                    float2 n2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_199), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_200 = mul_f32x2_noftz(mul_f32x2_noftz(g2_8, mul_f32x2_noftz(n2_8, w2_8)), sig2_8);
                    float2 _f32x2_mul_b_201 = fma_f32x2_rn_noftz(z2_8, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_8), make_float2(1.0f, 1.0f));
                    float2 dz2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
                    float2 a2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_8) : "l"(*(const unsigned long long*)&g2_8), "l"(*(const unsigned long long*)&gate2_8));
                    float2 dn2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_8) : "l"(*(const unsigned long long*)&a2_8), "l"(*(const unsigned long long*)&w2_8));
                    float2 acc2_8 = fma_f32x2_rn_noftz(a2_8, n2_8, make_float2(acc_r[0], acc_r[1]));
                    acc_r[0] = acc2_8.x;
                    acc_r[1] = acc2_8.y;
                    zr_2[0] = dz2_8.x;
                    zr_2[1] = dz2_8.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_2[_lp*2 + 0], zr_2[_lp*2+1 + 0]));
                        packed_dz_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    gr_2[0] = dn2_8.x;
                    gr_2[1] = dn2_8.y;
                    float2 z2_9 = make_float2(zr_2[2], zr_2[3]);
                    float2 _f32x2_mul_b_202 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_9) : "l"(*(const unsigned long long*)&z2_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_202));
                    float _fma_203;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(zr_2[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_204;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(zr_2[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_9 = make_float2(_fma_203, _fma_204);
                    float _exp2_205 = approx_exp2(ep2_9.x);
                    float _exp2_206 = approx_exp2(ep2_9.y);
                    float2 dq2_9 = fma_f32x2_rn_noftz(make_float2(_exp2_205, _exp2_206), qr2_9, make_float2(0.25f, 0.25f));
                    float _rcp_207 = approx_rcp(dq2_9.x);
                    float _rcp_208 = approx_rcp(dq2_9.y);
                    float2 _f32x2_mul_a_209 = make_float2(_rcp_207, _rcp_208);
                    float2 _f32x2_mul_b_210 = make_float2(0.25f, 0.25f);
                    float2 sig2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_209), "l"(*(const unsigned long long*)&_f32x2_mul_b_210));
                    float2 gate2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_9) : "l"(*(const unsigned long long*)&z2_9), "l"(*(const unsigned long long*)&sig2_9));
                    float2 g2_9 = make_float2(gr_2[2], gr_2[3]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_2[_pair * 2])[0]), "=f"((&wk_2[_pair * 2])[1])
                            : "r"(wbits[1 + _pair]));
                    }
                    float2 w2_9 = make_float2(wk_2[0], wk_2[1]);
                    float2 _f32x2_mul_a_211 = make_float2(xr_2[2], xr_2[3]);
                    float2 n2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_211), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_212 = mul_f32x2_noftz(mul_f32x2_noftz(g2_9, mul_f32x2_noftz(n2_9, w2_9)), sig2_9);
                    float2 _f32x2_mul_b_213 = fma_f32x2_rn_noftz(z2_9, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_9), make_float2(1.0f, 1.0f));
                    float2 dz2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_212), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
                    float2 a2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_9) : "l"(*(const unsigned long long*)&g2_9), "l"(*(const unsigned long long*)&gate2_9));
                    float2 dn2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_9) : "l"(*(const unsigned long long*)&a2_9), "l"(*(const unsigned long long*)&w2_9));
                    float2 acc2_9 = fma_f32x2_rn_noftz(a2_9, n2_9, make_float2(acc_r[2], acc_r[3]));
                    acc_r[2] = acc2_9.x;
                    acc_r[3] = acc2_9.y;
                    zr_2[2] = dz2_9.x;
                    zr_2[3] = dz2_9.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_2[_lp*2 + 2], zr_2[_lp*2+1 + 2]));
                        packed_dz_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    gr_2[2] = dn2_9.x;
                    gr_2[3] = dn2_9.y;
                    float2 z2_10 = make_float2(zr_2[4], zr_2[5]);
                    float2 _f32x2_mul_b_214 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_10) : "l"(*(const unsigned long long*)&z2_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_214));
                    float _fma_215;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(zr_2[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_216;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(zr_2[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_10 = make_float2(_fma_215, _fma_216);
                    float _exp2_217 = approx_exp2(ep2_10.x);
                    float _exp2_218 = approx_exp2(ep2_10.y);
                    float2 dq2_10 = fma_f32x2_rn_noftz(make_float2(_exp2_217, _exp2_218), qr2_10, make_float2(0.25f, 0.25f));
                    float _rcp_219 = approx_rcp(dq2_10.x);
                    float _rcp_220 = approx_rcp(dq2_10.y);
                    float2 _f32x2_mul_a_221 = make_float2(_rcp_219, _rcp_220);
                    float2 _f32x2_mul_b_222 = make_float2(0.25f, 0.25f);
                    float2 sig2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_221), "l"(*(const unsigned long long*)&_f32x2_mul_b_222));
                    float2 gate2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_10) : "l"(*(const unsigned long long*)&z2_10), "l"(*(const unsigned long long*)&sig2_10));
                    float2 g2_10 = make_float2(gr_2[4], gr_2[5]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_2[_pair * 2])[0]), "=f"((&wk_2[_pair * 2])[1])
                            : "r"(wbits[2 + _pair]));
                    }
                    float2 w2_10 = make_float2(wk_2[0], wk_2[1]);
                    float2 _f32x2_mul_a_223 = make_float2(xr_2[4], xr_2[5]);
                    float2 n2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_224 = mul_f32x2_noftz(mul_f32x2_noftz(g2_10, mul_f32x2_noftz(n2_10, w2_10)), sig2_10);
                    float2 _f32x2_mul_b_225 = fma_f32x2_rn_noftz(z2_10, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_10), make_float2(1.0f, 1.0f));
                    float2 dz2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_224), "l"(*(const unsigned long long*)&_f32x2_mul_b_225));
                    float2 a2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_10) : "l"(*(const unsigned long long*)&g2_10), "l"(*(const unsigned long long*)&gate2_10));
                    float2 dn2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_10) : "l"(*(const unsigned long long*)&a2_10), "l"(*(const unsigned long long*)&w2_10));
                    float2 acc2_10 = fma_f32x2_rn_noftz(a2_10, n2_10, make_float2(acc_r[4], acc_r[5]));
                    acc_r[4] = acc2_10.x;
                    acc_r[5] = acc2_10.y;
                    zr_2[4] = dz2_10.x;
                    zr_2[5] = dz2_10.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_2[_lp*2 + 4], zr_2[_lp*2+1 + 4]));
                        packed_dz_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    gr_2[4] = dn2_10.x;
                    gr_2[5] = dn2_10.y;
                    float2 z2_11 = make_float2(zr_2[6], zr_2[7]);
                    float2 _f32x2_mul_b_226 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_11) : "l"(*(const unsigned long long*)&z2_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_226));
                    float _fma_227;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(zr_2[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_228;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(zr_2[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_11 = make_float2(_fma_227, _fma_228);
                    float _exp2_229 = approx_exp2(ep2_11.x);
                    float _exp2_230 = approx_exp2(ep2_11.y);
                    float2 dq2_11 = fma_f32x2_rn_noftz(make_float2(_exp2_229, _exp2_230), qr2_11, make_float2(0.25f, 0.25f));
                    float _rcp_231 = approx_rcp(dq2_11.x);
                    float _rcp_232 = approx_rcp(dq2_11.y);
                    float2 _f32x2_mul_a_233 = make_float2(_rcp_231, _rcp_232);
                    float2 _f32x2_mul_b_234 = make_float2(0.25f, 0.25f);
                    float2 sig2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_233), "l"(*(const unsigned long long*)&_f32x2_mul_b_234));
                    float2 gate2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_11) : "l"(*(const unsigned long long*)&z2_11), "l"(*(const unsigned long long*)&sig2_11));
                    float2 g2_11 = make_float2(gr_2[6], gr_2[7]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_2[_pair * 2])[0]), "=f"((&wk_2[_pair * 2])[1])
                            : "r"(wbits[3 + _pair]));
                    }
                    float2 w2_11 = make_float2(wk_2[0], wk_2[1]);
                    float2 _f32x2_mul_a_235 = make_float2(xr_2[6], xr_2[7]);
                    float2 n2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_235), "l"(*(const unsigned long long*)&rstd2_2));
                    float2 _f32x2_mul_a_236 = mul_f32x2_noftz(mul_f32x2_noftz(g2_11, mul_f32x2_noftz(n2_11, w2_11)), sig2_11);
                    float2 _f32x2_mul_b_237 = fma_f32x2_rn_noftz(z2_11, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_11), make_float2(1.0f, 1.0f));
                    float2 dz2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_236), "l"(*(const unsigned long long*)&_f32x2_mul_b_237));
                    float2 a2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_11) : "l"(*(const unsigned long long*)&g2_11), "l"(*(const unsigned long long*)&gate2_11));
                    float2 dn2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_11) : "l"(*(const unsigned long long*)&a2_11), "l"(*(const unsigned long long*)&w2_11));
                    float2 acc2_11 = fma_f32x2_rn_noftz(a2_11, n2_11, make_float2(acc_r[6], acc_r[7]));
                    acc_r[6] = acc2_11.x;
                    acc_r[7] = acc2_11.y;
                    zr_2[6] = dz2_11.x;
                    zr_2[7] = dz2_11.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_2[_lp*2 + 6], zr_2[_lp*2+1 + 6]));
                        packed_dz_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    gr_2[6] = dn2_11.x;
                    gr_2[7] = dn2_11.y;
                    dot_2 = __fmaf_rn(gr_2[0], xr_2[0], dot_2);
                    dot_2 = __fmaf_rn(gr_2[1], xr_2[1], dot_2);
                    dot_2 = __fmaf_rn(gr_2[2], xr_2[2], dot_2);
                    dot_2 = __fmaf_rn(gr_2[3], xr_2[3], dot_2);
                    dot_2 = __fmaf_rn(gr_2[4], xr_2[4], dot_2);
                    dot_2 = __fmaf_rn(gr_2[5], xr_2[5], dot_2);
                    dot_2 = __fmaf_rn(gr_2[6], xr_2[6], dot_2);
                    dot_2 = __fmaf_rn(gr_2[7], xr_2[7], dot_2);
                    float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                    dot_2 = dot_2 + _shfl_xor_sync_238;
                    float _shfl_xor_sync_239 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                    dot_2 = dot_2 + _shfl_xor_sync_239;
                    float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                    dot_2 = dot_2 + _shfl_xor_sync_240;
                    float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 8);
                    dot_2 = dot_2 + _shfl_xor_sync_241;
                    float coef_2 = rstd_2 * rstd_2 * rstd_2 * (dot_2 * -0.0078125f);
                    float2 coef2_2 = make_float2(coef_2, coef_2);
                    float2 rstd2b_2 = make_float2(rstd_2, rstd_2);
                    float2 dx2_8 = fma_f32x2_rn_noftz(make_float2(xr_2[0], xr_2[1]), coef2_2, mul_f32x2_noftz(make_float2(gr_2[0], gr_2[1]), rstd2b_2));
                    xr_2[0] = dx2_8.x;
                    xr_2[1] = dx2_8.y;
                    float2 dx2_9 = fma_f32x2_rn_noftz(make_float2(xr_2[2], xr_2[3]), coef2_2, mul_f32x2_noftz(make_float2(gr_2[2], gr_2[3]), rstd2b_2));
                    xr_2[2] = dx2_9.x;
                    xr_2[3] = dx2_9.y;
                    float2 dx2_10 = fma_f32x2_rn_noftz(make_float2(xr_2[4], xr_2[5]), coef2_2, mul_f32x2_noftz(make_float2(gr_2[4], gr_2[5]), rstd2b_2));
                    xr_2[4] = dx2_10.x;
                    xr_2[5] = dx2_10.y;
                    float2 dx2_11 = fma_f32x2_rn_noftz(make_float2(xr_2[6], xr_2[7]), coef2_2, mul_f32x2_noftz(make_float2(gr_2[6], gr_2[7]), rstd2b_2));
                    xr_2[6] = dx2_11.x;
                    xr_2[7] = dx2_11.y;
                    unsigned int packed_dx_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_2[_lp*2 + 0], xr_2[_lp*2+1 + 0]));
                        packed_dx_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_2[_lp*2 + 2], xr_2[_lp*2+1 + 2]));
                        packed_dx_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_2[_lp*2 + 4], xr_2[_lp*2+1 + 4]));
                        packed_dx_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_2[_lp*2 + 6], xr_2[_lp*2+1 + 6]));
                        packed_dx_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    long long orow_2 = goff_2;
                    reinterpret_cast<int4*>(dx + orow_2)[0] = reinterpret_cast<int4*>(packed_dx_2)[0];
                    reinterpret_cast<int4*>(dz + orow_2)[0] = reinterpret_cast<int4*>(packed_dz_2)[0];
                    if ((i & 15) == 15) {
                        if (i >= 16) {
                            float s_9;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_9) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                            float bv_9;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_9) : "f"(s_9), "f"(fold[group * 128 + col]));
                            float av_9;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_9) : "f"(s_9), "f"(bv_9));
                            float _sub_242;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_242) : "f"(fold[group * 128 + col]), "f"(av_9));
                            float _sub_243;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_243) : "f"(acc_r[0]), "f"(bv_9));
                            float e_9;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_9) : "f"(_sub_242), "f"(_sub_243));
                            fold[group * 128 + col] = s_9;
                            float _add_244;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_244) : "f"(fold[(16 + group) * 128 + col]), "f"(e_9));
                            fold[(16 + group) * 128 + col] = _add_244;
                            acc_r[0] = 0.0f;
                            float s_28;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_28) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                            float bv_28;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_28) : "f"(s_28), "f"(fold[group * 128 + col + 1]));
                            float av_28;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_28) : "f"(s_28), "f"(bv_28));
                            float _sub_245;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_245) : "f"(fold[group * 128 + col + 1]), "f"(av_28));
                            float _sub_246;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_246) : "f"(acc_r[1]), "f"(bv_28));
                            float e_28;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_28) : "f"(_sub_245), "f"(_sub_246));
                            fold[group * 128 + col + 1] = s_28;
                            float _add_247;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_28));
                            fold[(16 + group) * 128 + col + 1] = _add_247;
                            acc_r[1] = 0.0f;
                            float s_29;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_29) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                            float bv_29;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_29) : "f"(s_29), "f"(fold[group * 128 + col + 2]));
                            float av_29;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_29) : "f"(s_29), "f"(bv_29));
                            float _sub_248;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_248) : "f"(fold[group * 128 + col + 2]), "f"(av_29));
                            float _sub_249;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_249) : "f"(acc_r[2]), "f"(bv_29));
                            float e_29;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_29) : "f"(_sub_248), "f"(_sub_249));
                            fold[group * 128 + col + 2] = s_29;
                            float _add_250;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_29));
                            fold[(16 + group) * 128 + col + 2] = _add_250;
                            acc_r[2] = 0.0f;
                            float s_30;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_30) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                            float bv_30;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_30) : "f"(s_30), "f"(fold[group * 128 + col + 3]));
                            float av_30;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_30) : "f"(s_30), "f"(bv_30));
                            float _sub_251;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_251) : "f"(fold[group * 128 + col + 3]), "f"(av_30));
                            float _sub_252;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_252) : "f"(acc_r[3]), "f"(bv_30));
                            float e_30;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_30) : "f"(_sub_251), "f"(_sub_252));
                            fold[group * 128 + col + 3] = s_30;
                            float _add_253;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_30));
                            fold[(16 + group) * 128 + col + 3] = _add_253;
                            acc_r[3] = 0.0f;
                            float s_31;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_31) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                            float bv_31;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_31) : "f"(s_31), "f"(fold[group * 128 + col + 4]));
                            float av_31;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_31) : "f"(s_31), "f"(bv_31));
                            float _sub_254;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_254) : "f"(fold[group * 128 + col + 4]), "f"(av_31));
                            float _sub_255;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_255) : "f"(acc_r[4]), "f"(bv_31));
                            float e_31;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_31) : "f"(_sub_254), "f"(_sub_255));
                            fold[group * 128 + col + 4] = s_31;
                            float _add_256;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_31));
                            fold[(16 + group) * 128 + col + 4] = _add_256;
                            acc_r[4] = 0.0f;
                            float s_32;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_32) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                            float bv_32;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_32) : "f"(s_32), "f"(fold[group * 128 + col + 5]));
                            float av_32;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_32) : "f"(s_32), "f"(bv_32));
                            float _sub_257;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_257) : "f"(fold[group * 128 + col + 5]), "f"(av_32));
                            float _sub_258;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_258) : "f"(acc_r[5]), "f"(bv_32));
                            float e_32;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_32) : "f"(_sub_257), "f"(_sub_258));
                            fold[group * 128 + col + 5] = s_32;
                            float _add_259;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_259) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_32));
                            fold[(16 + group) * 128 + col + 5] = _add_259;
                            acc_r[5] = 0.0f;
                            float s_33;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_33) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                            float bv_33;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_33) : "f"(s_33), "f"(fold[group * 128 + col + 6]));
                            float av_33;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_33) : "f"(s_33), "f"(bv_33));
                            float _sub_260;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_260) : "f"(fold[group * 128 + col + 6]), "f"(av_33));
                            float _sub_261;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_261) : "f"(acc_r[6]), "f"(bv_33));
                            float e_33;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_33) : "f"(_sub_260), "f"(_sub_261));
                            fold[group * 128 + col + 6] = s_33;
                            float _add_262;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_33));
                            fold[(16 + group) * 128 + col + 6] = _add_262;
                            acc_r[6] = 0.0f;
                            float s_34;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_34) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                            float bv_34;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_34) : "f"(s_34), "f"(fold[group * 128 + col + 7]));
                            float av_34;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_34) : "f"(s_34), "f"(bv_34));
                            float _sub_263;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_263) : "f"(fold[group * 128 + col + 7]), "f"(av_34));
                            float _sub_264;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_264) : "f"(acc_r[7]), "f"(bv_34));
                            float e_34;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_34) : "f"(_sub_263), "f"(_sub_264));
                            fold[group * 128 + col + 7] = s_34;
                            float _add_265;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_34));
                            fold[(16 + group) * 128 + col + 7] = _add_265;
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
                }
                int tail_i_2 = chunk_n - 1;
                int u_3 = bound + chunk_lo + tail_i_2 * stride;
                int row_3 = u_3 * 16 + group;
                int src_2 = ((row_3) < (rows - 1) ? (row_3) : (rows - 1));
                long long goff_3 = (long long)src_2 * 128ll + (long long)col;
                long long xoff_2 = goff_3;
                long long zoff_2 = goff_3;
                float gr_3[8];
                float xr_3[8];
                float zr_3[8];
                {
                    const uint4* _vptr_266 = reinterpret_cast<const uint4*>(dy + goff_3 + 0);
                    uint4 _vld_266[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_266[_blk] = _vptr_266[_blk];
                        uint32_t* _vpairs_266 = reinterpret_cast<uint32_t*>(&_vld_266[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_266[_pair]) << 16);
                            (&gr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_266[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_267 = reinterpret_cast<const uint4*>(x + xoff_2 + 0);
                    uint4 _vld_267[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_267[_blk] = _vptr_267[_blk];
                        uint32_t* _vpairs_267 = reinterpret_cast<uint32_t*>(&_vld_267[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_267[_pair]) << 16);
                            (&xr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_267[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_268 = reinterpret_cast<const uint4*>(z + zoff_2 + 0);
                    uint4 _vld_268[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_268[_blk] = _vptr_268[_blk];
                        uint32_t* _vpairs_268 = reinterpret_cast<uint32_t*>(&_vld_268[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_268[_pair]) << 16);
                            (&zr_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_268[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (tid == 0) {
                    if (trips - 1 - it > 0) {
                        uint32_t _atomic_fetch_inc_269;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_269) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        flag[1] = (int)_atomic_fetch_inc_269;
                    }
                }
                float total_3 = 0.0f;
                total_3 = __fmaf_rn(xr_3[0], xr_3[0], total_3);
                total_3 = __fmaf_rn(xr_3[1], xr_3[1], total_3);
                total_3 = __fmaf_rn(xr_3[2], xr_3[2], total_3);
                total_3 = __fmaf_rn(xr_3[3], xr_3[3], total_3);
                total_3 = __fmaf_rn(xr_3[4], xr_3[4], total_3);
                total_3 = __fmaf_rn(xr_3[5], xr_3[5], total_3);
                total_3 = __fmaf_rn(xr_3[6], xr_3[6], total_3);
                total_3 = __fmaf_rn(xr_3[7], xr_3[7], total_3);
                float _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, total_3, 1);
                total_3 = total_3 + _shfl_xor_sync_270;
                float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, total_3, 2);
                total_3 = total_3 + _shfl_xor_sync_271;
                float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, total_3, 4);
                total_3 = total_3 + _shfl_xor_sync_272;
                float _shfl_xor_sync_273 = __shfl_xor_sync(0xFFFFFFFF, total_3, 8);
                total_3 = total_3 + _shfl_xor_sync_273;
                float rstd_3;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_3) : "f"(total_3 * 0.0078125f + eps));
                float dot_3 = 0.0f;
                float wk_3[2];
                unsigned int packed_dz_3[4];
                float2 rstd2_3 = make_float2(rstd_3, rstd_3);
                float2 z2_12 = make_float2(zr_3[0], zr_3[1]);
                float2 _f32x2_mul_b_274 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_12) : "l"(*(const unsigned long long*)&z2_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_274));
                float _fma_275;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(zr_3[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_276;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(zr_3[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_12 = make_float2(_fma_275, _fma_276);
                float _exp2_277 = approx_exp2(ep2_12.x);
                float _exp2_278 = approx_exp2(ep2_12.y);
                float2 dq2_12 = fma_f32x2_rn_noftz(make_float2(_exp2_277, _exp2_278), qr2_12, make_float2(0.25f, 0.25f));
                float _rcp_279 = approx_rcp(dq2_12.x);
                float _rcp_280 = approx_rcp(dq2_12.y);
                float2 _f32x2_mul_a_281 = make_float2(_rcp_279, _rcp_280);
                float2 _f32x2_mul_b_282 = make_float2(0.25f, 0.25f);
                float2 sig2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_281), "l"(*(const unsigned long long*)&_f32x2_mul_b_282));
                float2 gate2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_12) : "l"(*(const unsigned long long*)&z2_12), "l"(*(const unsigned long long*)&sig2_12));
                float2 g2_12 = make_float2(gr_3[0], gr_3[1]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_3[_pair * 2])[0]), "=f"((&wk_3[_pair * 2])[1])
                        : "r"(wbits[_pair]));
                }
                float2 w2_12 = make_float2(wk_3[0], wk_3[1]);
                float2 _f32x2_mul_a_283 = make_float2(xr_3[0], xr_3[1]);
                float2 n2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_283), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_284 = mul_f32x2_noftz(mul_f32x2_noftz(g2_12, mul_f32x2_noftz(n2_12, w2_12)), sig2_12);
                float2 _f32x2_mul_b_285 = fma_f32x2_rn_noftz(z2_12, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_12), make_float2(1.0f, 1.0f));
                float2 dz2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_284), "l"(*(const unsigned long long*)&_f32x2_mul_b_285));
                float2 a2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_12) : "l"(*(const unsigned long long*)&g2_12), "l"(*(const unsigned long long*)&gate2_12));
                float2 dn2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_12) : "l"(*(const unsigned long long*)&a2_12), "l"(*(const unsigned long long*)&w2_12));
                float2 acc2_12 = fma_f32x2_rn_noftz(a2_12, n2_12, make_float2(acc_r[0], acc_r[1]));
                acc_r[0] = ((row_3 < rows) ? acc2_12.x : acc_r[0]);
                acc_r[1] = ((row_3 < rows) ? acc2_12.y : acc_r[1]);
                zr_3[0] = dz2_12.x;
                zr_3[1] = dz2_12.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_3[_lp*2 + 0], zr_3[_lp*2+1 + 0]));
                    packed_dz_3[_lp] = *(uint32_t*)&_bf2;
                }
                gr_3[0] = dn2_12.x;
                gr_3[1] = dn2_12.y;
                float2 z2_13 = make_float2(zr_3[2], zr_3[3]);
                float2 _f32x2_mul_b_286 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_13) : "l"(*(const unsigned long long*)&z2_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_286));
                float _fma_287;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(zr_3[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_288;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(zr_3[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_13 = make_float2(_fma_287, _fma_288);
                float _exp2_289 = approx_exp2(ep2_13.x);
                float _exp2_290 = approx_exp2(ep2_13.y);
                float2 dq2_13 = fma_f32x2_rn_noftz(make_float2(_exp2_289, _exp2_290), qr2_13, make_float2(0.25f, 0.25f));
                float _rcp_291 = approx_rcp(dq2_13.x);
                float _rcp_292 = approx_rcp(dq2_13.y);
                float2 _f32x2_mul_a_293 = make_float2(_rcp_291, _rcp_292);
                float2 _f32x2_mul_b_294 = make_float2(0.25f, 0.25f);
                float2 sig2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_293), "l"(*(const unsigned long long*)&_f32x2_mul_b_294));
                float2 gate2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_13) : "l"(*(const unsigned long long*)&z2_13), "l"(*(const unsigned long long*)&sig2_13));
                float2 g2_13 = make_float2(gr_3[2], gr_3[3]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_3[_pair * 2])[0]), "=f"((&wk_3[_pair * 2])[1])
                        : "r"(wbits[1 + _pair]));
                }
                float2 w2_13 = make_float2(wk_3[0], wk_3[1]);
                float2 _f32x2_mul_a_295 = make_float2(xr_3[2], xr_3[3]);
                float2 n2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_295), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_296 = mul_f32x2_noftz(mul_f32x2_noftz(g2_13, mul_f32x2_noftz(n2_13, w2_13)), sig2_13);
                float2 _f32x2_mul_b_297 = fma_f32x2_rn_noftz(z2_13, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_13), make_float2(1.0f, 1.0f));
                float2 dz2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_296), "l"(*(const unsigned long long*)&_f32x2_mul_b_297));
                float2 a2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_13) : "l"(*(const unsigned long long*)&g2_13), "l"(*(const unsigned long long*)&gate2_13));
                float2 dn2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_13) : "l"(*(const unsigned long long*)&a2_13), "l"(*(const unsigned long long*)&w2_13));
                float2 acc2_13 = fma_f32x2_rn_noftz(a2_13, n2_13, make_float2(acc_r[2], acc_r[3]));
                acc_r[2] = ((row_3 < rows) ? acc2_13.x : acc_r[2]);
                acc_r[3] = ((row_3 < rows) ? acc2_13.y : acc_r[3]);
                zr_3[2] = dz2_13.x;
                zr_3[3] = dz2_13.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_3[_lp*2 + 2], zr_3[_lp*2+1 + 2]));
                    packed_dz_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                gr_3[2] = dn2_13.x;
                gr_3[3] = dn2_13.y;
                float2 z2_14 = make_float2(zr_3[4], zr_3[5]);
                float2 _f32x2_mul_b_298 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_14) : "l"(*(const unsigned long long*)&z2_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_298));
                float _fma_299;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(zr_3[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_300;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(zr_3[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_14 = make_float2(_fma_299, _fma_300);
                float _exp2_301 = approx_exp2(ep2_14.x);
                float _exp2_302 = approx_exp2(ep2_14.y);
                float2 dq2_14 = fma_f32x2_rn_noftz(make_float2(_exp2_301, _exp2_302), qr2_14, make_float2(0.25f, 0.25f));
                float _rcp_303 = approx_rcp(dq2_14.x);
                float _rcp_304 = approx_rcp(dq2_14.y);
                float2 _f32x2_mul_a_305 = make_float2(_rcp_303, _rcp_304);
                float2 _f32x2_mul_b_306 = make_float2(0.25f, 0.25f);
                float2 sig2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_305), "l"(*(const unsigned long long*)&_f32x2_mul_b_306));
                float2 gate2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_14) : "l"(*(const unsigned long long*)&z2_14), "l"(*(const unsigned long long*)&sig2_14));
                float2 g2_14 = make_float2(gr_3[4], gr_3[5]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_3[_pair * 2])[0]), "=f"((&wk_3[_pair * 2])[1])
                        : "r"(wbits[2 + _pair]));
                }
                float2 w2_14 = make_float2(wk_3[0], wk_3[1]);
                float2 _f32x2_mul_a_307 = make_float2(xr_3[4], xr_3[5]);
                float2 n2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_307), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_308 = mul_f32x2_noftz(mul_f32x2_noftz(g2_14, mul_f32x2_noftz(n2_14, w2_14)), sig2_14);
                float2 _f32x2_mul_b_309 = fma_f32x2_rn_noftz(z2_14, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_14), make_float2(1.0f, 1.0f));
                float2 dz2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_308), "l"(*(const unsigned long long*)&_f32x2_mul_b_309));
                float2 a2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_14) : "l"(*(const unsigned long long*)&g2_14), "l"(*(const unsigned long long*)&gate2_14));
                float2 dn2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_14) : "l"(*(const unsigned long long*)&a2_14), "l"(*(const unsigned long long*)&w2_14));
                float2 acc2_14 = fma_f32x2_rn_noftz(a2_14, n2_14, make_float2(acc_r[4], acc_r[5]));
                acc_r[4] = ((row_3 < rows) ? acc2_14.x : acc_r[4]);
                acc_r[5] = ((row_3 < rows) ? acc2_14.y : acc_r[5]);
                zr_3[4] = dz2_14.x;
                zr_3[5] = dz2_14.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_3[_lp*2 + 4], zr_3[_lp*2+1 + 4]));
                    packed_dz_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                gr_3[4] = dn2_14.x;
                gr_3[5] = dn2_14.y;
                float2 z2_15 = make_float2(zr_3[6], zr_3[7]);
                float2 _f32x2_mul_b_310 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_15) : "l"(*(const unsigned long long*)&z2_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_310));
                float _fma_311;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(zr_3[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_312;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(zr_3[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_15 = make_float2(_fma_311, _fma_312);
                float _exp2_313 = approx_exp2(ep2_15.x);
                float _exp2_314 = approx_exp2(ep2_15.y);
                float2 dq2_15 = fma_f32x2_rn_noftz(make_float2(_exp2_313, _exp2_314), qr2_15, make_float2(0.25f, 0.25f));
                float _rcp_315 = approx_rcp(dq2_15.x);
                float _rcp_316 = approx_rcp(dq2_15.y);
                float2 _f32x2_mul_a_317 = make_float2(_rcp_315, _rcp_316);
                float2 _f32x2_mul_b_318 = make_float2(0.25f, 0.25f);
                float2 sig2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_317), "l"(*(const unsigned long long*)&_f32x2_mul_b_318));
                float2 gate2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_15) : "l"(*(const unsigned long long*)&z2_15), "l"(*(const unsigned long long*)&sig2_15));
                float2 g2_15 = make_float2(gr_3[6], gr_3[7]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_3[_pair * 2])[0]), "=f"((&wk_3[_pair * 2])[1])
                        : "r"(wbits[3 + _pair]));
                }
                float2 w2_15 = make_float2(wk_3[0], wk_3[1]);
                float2 _f32x2_mul_a_319 = make_float2(xr_3[6], xr_3[7]);
                float2 n2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_319), "l"(*(const unsigned long long*)&rstd2_3));
                float2 _f32x2_mul_a_320 = mul_f32x2_noftz(mul_f32x2_noftz(g2_15, mul_f32x2_noftz(n2_15, w2_15)), sig2_15);
                float2 _f32x2_mul_b_321 = fma_f32x2_rn_noftz(z2_15, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_15), make_float2(1.0f, 1.0f));
                float2 dz2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_320), "l"(*(const unsigned long long*)&_f32x2_mul_b_321));
                float2 a2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_15) : "l"(*(const unsigned long long*)&g2_15), "l"(*(const unsigned long long*)&gate2_15));
                float2 dn2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_15) : "l"(*(const unsigned long long*)&a2_15), "l"(*(const unsigned long long*)&w2_15));
                float2 acc2_15 = fma_f32x2_rn_noftz(a2_15, n2_15, make_float2(acc_r[6], acc_r[7]));
                acc_r[6] = ((row_3 < rows) ? acc2_15.x : acc_r[6]);
                acc_r[7] = ((row_3 < rows) ? acc2_15.y : acc_r[7]);
                zr_3[6] = dz2_15.x;
                zr_3[7] = dz2_15.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_3[_lp*2 + 6], zr_3[_lp*2+1 + 6]));
                    packed_dz_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                gr_3[6] = dn2_15.x;
                gr_3[7] = dn2_15.y;
                dot_3 = __fmaf_rn(gr_3[0], xr_3[0], dot_3);
                dot_3 = __fmaf_rn(gr_3[1], xr_3[1], dot_3);
                dot_3 = __fmaf_rn(gr_3[2], xr_3[2], dot_3);
                dot_3 = __fmaf_rn(gr_3[3], xr_3[3], dot_3);
                dot_3 = __fmaf_rn(gr_3[4], xr_3[4], dot_3);
                dot_3 = __fmaf_rn(gr_3[5], xr_3[5], dot_3);
                dot_3 = __fmaf_rn(gr_3[6], xr_3[6], dot_3);
                dot_3 = __fmaf_rn(gr_3[7], xr_3[7], dot_3);
                float _shfl_xor_sync_322 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_322;
                float _shfl_xor_sync_323 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_323;
                float _shfl_xor_sync_324 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_324;
                float _shfl_xor_sync_325 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 8);
                dot_3 = dot_3 + _shfl_xor_sync_325;
                float coef_3 = rstd_3 * rstd_3 * rstd_3 * (dot_3 * -0.0078125f);
                float2 coef2_3 = make_float2(coef_3, coef_3);
                float2 rstd2b_3 = make_float2(rstd_3, rstd_3);
                float2 dx2_12 = fma_f32x2_rn_noftz(make_float2(xr_3[0], xr_3[1]), coef2_3, mul_f32x2_noftz(make_float2(gr_3[0], gr_3[1]), rstd2b_3));
                xr_3[0] = dx2_12.x;
                xr_3[1] = dx2_12.y;
                float2 dx2_13 = fma_f32x2_rn_noftz(make_float2(xr_3[2], xr_3[3]), coef2_3, mul_f32x2_noftz(make_float2(gr_3[2], gr_3[3]), rstd2b_3));
                xr_3[2] = dx2_13.x;
                xr_3[3] = dx2_13.y;
                float2 dx2_14 = fma_f32x2_rn_noftz(make_float2(xr_3[4], xr_3[5]), coef2_3, mul_f32x2_noftz(make_float2(gr_3[4], gr_3[5]), rstd2b_3));
                xr_3[4] = dx2_14.x;
                xr_3[5] = dx2_14.y;
                float2 dx2_15 = fma_f32x2_rn_noftz(make_float2(xr_3[6], xr_3[7]), coef2_3, mul_f32x2_noftz(make_float2(gr_3[6], gr_3[7]), rstd2b_3));
                xr_3[6] = dx2_15.x;
                xr_3[7] = dx2_15.y;
                unsigned int packed_dx_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_3[_lp*2 + 0], xr_3[_lp*2+1 + 0]));
                    packed_dx_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_3[_lp*2 + 2], xr_3[_lp*2+1 + 2]));
                    packed_dx_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_3[_lp*2 + 4], xr_3[_lp*2+1 + 4]));
                    packed_dx_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_3[_lp*2 + 6], xr_3[_lp*2+1 + 6]));
                    packed_dx_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                long long orow_3 = goff_3;
                if (row_3 < rows) {
                    reinterpret_cast<int4*>(dx + orow_3)[0] = reinterpret_cast<int4*>(packed_dx_3)[0];
                    reinterpret_cast<int4*>(dz + orow_3)[0] = reinterpret_cast<int4*>(packed_dz_3)[0];
                }
                if ((tail_i_2 & 15) == 15) {
                    if (tail_i_2 >= 16) {
                        float s_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_10) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        float bv_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_10) : "f"(s_10), "f"(fold[group * 128 + col]));
                        float av_10;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_10) : "f"(s_10), "f"(bv_10));
                        float _sub_326;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_326) : "f"(fold[group * 128 + col]), "f"(av_10));
                        float _sub_327;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_327) : "f"(acc_r[0]), "f"(bv_10));
                        float e_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_10) : "f"(_sub_326), "f"(_sub_327));
                        fold[group * 128 + col] = s_10;
                        float _add_328;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_328) : "f"(fold[(16 + group) * 128 + col]), "f"(e_10));
                        fold[(16 + group) * 128 + col] = _add_328;
                        acc_r[0] = 0.0f;
                        float s_35;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_35) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        float bv_35;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_35) : "f"(s_35), "f"(fold[group * 128 + col + 1]));
                        float av_35;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_35) : "f"(s_35), "f"(bv_35));
                        float _sub_329;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_329) : "f"(fold[group * 128 + col + 1]), "f"(av_35));
                        float _sub_330;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_330) : "f"(acc_r[1]), "f"(bv_35));
                        float e_35;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_35) : "f"(_sub_329), "f"(_sub_330));
                        fold[group * 128 + col + 1] = s_35;
                        float _add_331;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_331) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_35));
                        fold[(16 + group) * 128 + col + 1] = _add_331;
                        acc_r[1] = 0.0f;
                        float s_36;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_36) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        float bv_36;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_36) : "f"(s_36), "f"(fold[group * 128 + col + 2]));
                        float av_36;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_36) : "f"(s_36), "f"(bv_36));
                        float _sub_332;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_332) : "f"(fold[group * 128 + col + 2]), "f"(av_36));
                        float _sub_333;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_333) : "f"(acc_r[2]), "f"(bv_36));
                        float e_36;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_36) : "f"(_sub_332), "f"(_sub_333));
                        fold[group * 128 + col + 2] = s_36;
                        float _add_334;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_334) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_36));
                        fold[(16 + group) * 128 + col + 2] = _add_334;
                        acc_r[2] = 0.0f;
                        float s_37;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_37) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        float bv_37;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_37) : "f"(s_37), "f"(fold[group * 128 + col + 3]));
                        float av_37;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_37) : "f"(s_37), "f"(bv_37));
                        float _sub_335;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_335) : "f"(fold[group * 128 + col + 3]), "f"(av_37));
                        float _sub_336;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_336) : "f"(acc_r[3]), "f"(bv_37));
                        float e_37;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_37) : "f"(_sub_335), "f"(_sub_336));
                        fold[group * 128 + col + 3] = s_37;
                        float _add_337;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_337) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_37));
                        fold[(16 + group) * 128 + col + 3] = _add_337;
                        acc_r[3] = 0.0f;
                        float s_38;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_38) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        float bv_38;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_38) : "f"(s_38), "f"(fold[group * 128 + col + 4]));
                        float av_38;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_38) : "f"(s_38), "f"(bv_38));
                        float _sub_338;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_338) : "f"(fold[group * 128 + col + 4]), "f"(av_38));
                        float _sub_339;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_339) : "f"(acc_r[4]), "f"(bv_38));
                        float e_38;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_38) : "f"(_sub_338), "f"(_sub_339));
                        fold[group * 128 + col + 4] = s_38;
                        float _add_340;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_340) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_38));
                        fold[(16 + group) * 128 + col + 4] = _add_340;
                        acc_r[4] = 0.0f;
                        float s_39;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_39) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        float bv_39;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_39) : "f"(s_39), "f"(fold[group * 128 + col + 5]));
                        float av_39;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_39) : "f"(s_39), "f"(bv_39));
                        float _sub_341;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_341) : "f"(fold[group * 128 + col + 5]), "f"(av_39));
                        float _sub_342;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_342) : "f"(acc_r[5]), "f"(bv_39));
                        float e_39;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_39) : "f"(_sub_341), "f"(_sub_342));
                        fold[group * 128 + col + 5] = s_39;
                        float _add_343;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_343) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_39));
                        fold[(16 + group) * 128 + col + 5] = _add_343;
                        acc_r[5] = 0.0f;
                        float s_40;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_40) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        float bv_40;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_40) : "f"(s_40), "f"(fold[group * 128 + col + 6]));
                        float av_40;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_40) : "f"(s_40), "f"(bv_40));
                        float _sub_344;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_344) : "f"(fold[group * 128 + col + 6]), "f"(av_40));
                        float _sub_345;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_345) : "f"(acc_r[6]), "f"(bv_40));
                        float e_40;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_40) : "f"(_sub_344), "f"(_sub_345));
                        fold[group * 128 + col + 6] = s_40;
                        float _add_346;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_346) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_40));
                        fold[(16 + group) * 128 + col + 6] = _add_346;
                        acc_r[6] = 0.0f;
                        float s_41;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_41) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        float bv_41;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_41) : "f"(s_41), "f"(fold[group * 128 + col + 7]));
                        float av_41;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_41) : "f"(s_41), "f"(bv_41));
                        float _sub_347;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_347) : "f"(fold[group * 128 + col + 7]), "f"(av_41));
                        float _sub_348;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_348) : "f"(acc_r[7]), "f"(bv_41));
                        float e_41;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_41) : "f"(_sub_347), "f"(_sub_348));
                        fold[group * 128 + col + 7] = s_41;
                        float _add_349;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_349) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_41));
                        fold[(16 + group) * 128 + col + 7] = _add_349;
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
                if ((chunk_n & 15) != 0) {
                    if (chunk_n - 1 >= 16) {
                        float s_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_11) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        float bv_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_11) : "f"(s_11), "f"(fold[group * 128 + col]));
                        float av_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_11) : "f"(s_11), "f"(bv_11));
                        float _sub_350;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_350) : "f"(fold[group * 128 + col]), "f"(av_11));
                        float _sub_351;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_351) : "f"(acc_r[0]), "f"(bv_11));
                        float e_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_11) : "f"(_sub_350), "f"(_sub_351));
                        fold[group * 128 + col] = s_11;
                        float _add_352;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_352) : "f"(fold[(16 + group) * 128 + col]), "f"(e_11));
                        fold[(16 + group) * 128 + col] = _add_352;
                        acc_r[0] = 0.0f;
                        float s_42;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_42) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        float bv_42;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_42) : "f"(s_42), "f"(fold[group * 128 + col + 1]));
                        float av_42;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_42) : "f"(s_42), "f"(bv_42));
                        float _sub_353;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_353) : "f"(fold[group * 128 + col + 1]), "f"(av_42));
                        float _sub_354;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_354) : "f"(acc_r[1]), "f"(bv_42));
                        float e_42;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_42) : "f"(_sub_353), "f"(_sub_354));
                        fold[group * 128 + col + 1] = s_42;
                        float _add_355;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_355) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_42));
                        fold[(16 + group) * 128 + col + 1] = _add_355;
                        acc_r[1] = 0.0f;
                        float s_43;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_43) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        float bv_43;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_43) : "f"(s_43), "f"(fold[group * 128 + col + 2]));
                        float av_43;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_43) : "f"(s_43), "f"(bv_43));
                        float _sub_356;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_356) : "f"(fold[group * 128 + col + 2]), "f"(av_43));
                        float _sub_357;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_357) : "f"(acc_r[2]), "f"(bv_43));
                        float e_43;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_43) : "f"(_sub_356), "f"(_sub_357));
                        fold[group * 128 + col + 2] = s_43;
                        float _add_358;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_358) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_43));
                        fold[(16 + group) * 128 + col + 2] = _add_358;
                        acc_r[2] = 0.0f;
                        float s_44;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_44) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        float bv_44;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_44) : "f"(s_44), "f"(fold[group * 128 + col + 3]));
                        float av_44;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_44) : "f"(s_44), "f"(bv_44));
                        float _sub_359;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_359) : "f"(fold[group * 128 + col + 3]), "f"(av_44));
                        float _sub_360;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_360) : "f"(acc_r[3]), "f"(bv_44));
                        float e_44;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_44) : "f"(_sub_359), "f"(_sub_360));
                        fold[group * 128 + col + 3] = s_44;
                        float _add_361;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_361) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_44));
                        fold[(16 + group) * 128 + col + 3] = _add_361;
                        acc_r[3] = 0.0f;
                        float s_45;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_45) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        float bv_45;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_45) : "f"(s_45), "f"(fold[group * 128 + col + 4]));
                        float av_45;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_45) : "f"(s_45), "f"(bv_45));
                        float _sub_362;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_362) : "f"(fold[group * 128 + col + 4]), "f"(av_45));
                        float _sub_363;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_363) : "f"(acc_r[4]), "f"(bv_45));
                        float e_45;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_45) : "f"(_sub_362), "f"(_sub_363));
                        fold[group * 128 + col + 4] = s_45;
                        float _add_364;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_364) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_45));
                        fold[(16 + group) * 128 + col + 4] = _add_364;
                        acc_r[4] = 0.0f;
                        float s_46;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_46) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        float bv_46;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_46) : "f"(s_46), "f"(fold[group * 128 + col + 5]));
                        float av_46;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_46) : "f"(s_46), "f"(bv_46));
                        float _sub_365;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_365) : "f"(fold[group * 128 + col + 5]), "f"(av_46));
                        float _sub_366;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_366) : "f"(acc_r[5]), "f"(bv_46));
                        float e_46;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_46) : "f"(_sub_365), "f"(_sub_366));
                        fold[group * 128 + col + 5] = s_46;
                        float _add_367;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_367) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_46));
                        fold[(16 + group) * 128 + col + 5] = _add_367;
                        acc_r[5] = 0.0f;
                        float s_47;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_47) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        float bv_47;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_47) : "f"(s_47), "f"(fold[group * 128 + col + 6]));
                        float av_47;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_47) : "f"(s_47), "f"(bv_47));
                        float _sub_368;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_368) : "f"(fold[group * 128 + col + 6]), "f"(av_47));
                        float _sub_369;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_369) : "f"(acc_r[6]), "f"(bv_47));
                        float e_47;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_47) : "f"(_sub_368), "f"(_sub_369));
                        fold[group * 128 + col + 6] = s_47;
                        float _add_370;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_370) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_47));
                        fold[(16 + group) * 128 + col + 6] = _add_370;
                        acc_r[6] = 0.0f;
                        float s_48;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_48) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        float bv_48;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_48) : "f"(s_48), "f"(fold[group * 128 + col + 7]));
                        float av_48;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_48) : "f"(s_48), "f"(bv_48));
                        float _sub_371;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_371) : "f"(fold[group * 128 + col + 7]), "f"(av_48));
                        float _sub_372;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_372) : "f"(acc_r[7]), "f"(bv_48));
                        float e_48;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_48) : "f"(_sub_371), "f"(_sub_372));
                        fold[group * 128 + col + 7] = s_48;
                        float _add_373;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_373) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_48));
                        fold[(16 + group) * 128 + col + 7] = _add_373;
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
            } else {
                if (chunk_n > 1) {
                    int vrow_2 = (bound + chunk_lo) * 16 + group;
                    uint32_t vq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)vrow_2), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)vrow_2);
                    uint32_t vh = (uint32_t)((unsigned int)vrow_2) - vq * (uint32_t)(heads.divisor);
                    long long vgoff = (long long)vq * dy_tok + (long long)vh * dy_head + (long long)col;
                    {
                        uint4 _uv4_374 = *reinterpret_cast<const uint4*>(dy + vgoff + 0);
                        la_g[0 + 0] = _uv4_374.x;
                        la_g[0 + 1] = _uv4_374.y;
                        la_g[0 + 2] = _uv4_374.z;
                        la_g[0 + 3] = _uv4_374.w;
                    }
                    long long vxoff = (long long)vq * x_tok + (long long)vh * x_head + (long long)col;
                    {
                        uint4 _uv4_375 = *reinterpret_cast<const uint4*>(x + vxoff + 0);
                        la_x[0 + 0] = _uv4_375.x;
                        la_x[0 + 1] = _uv4_375.y;
                        la_x[0 + 2] = _uv4_375.z;
                        la_x[0 + 3] = _uv4_375.w;
                    }
                    long long vzoff = (long long)vq * z_tok + (long long)vh * z_head + (long long)col;
                    {
                        uint4 _uv4_376 = *reinterpret_cast<const uint4*>(z + vzoff + 0);
                        la_z[0 + 0] = _uv4_376.x;
                        la_z[0 + 1] = _uv4_376.y;
                        la_z[0 + 2] = _uv4_376.z;
                        la_z[0 + 3] = _uv4_376.w;
                    }
                }
                #pragma unroll 1
                for (int i_1 = 0; i_1 < chunk_n - 1; i_1++) {
                    int u_4 = bound + chunk_lo + i_1 * stride;
                    int row_4 = u_4 * 16 + group;
                    float gr_4[8];
                    float xr_4[8];
                    float zr_4[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&xr_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_x[_pair]) << 16);
                        (&xr_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_x[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&gr_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_g[_pair]) << 16);
                        (&gr_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_g[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&zr_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(la_z[_pair]) << 16);
                        (&zr_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(la_z[_pair]) & 0xffff0000u);
                    }
                    if (i_1 + 2 < chunk_n) {
                        int vrow_3 = (u_4 + stride) * 16 + group;
                        uint32_t vq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)vrow_3), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)vrow_3);
                        uint32_t vh_1 = (uint32_t)((unsigned int)vrow_3) - vq_1 * (uint32_t)(heads.divisor);
                        long long vgoff_1 = (long long)vq_1 * dy_tok + (long long)vh_1 * dy_head + (long long)col;
                        {
                            uint4 _uv4_377 = *reinterpret_cast<const uint4*>(dy + vgoff_1 + 0);
                            la_g[0 + 0] = _uv4_377.x;
                            la_g[0 + 1] = _uv4_377.y;
                            la_g[0 + 2] = _uv4_377.z;
                            la_g[0 + 3] = _uv4_377.w;
                        }
                        long long vxoff_1 = (long long)vq_1 * x_tok + (long long)vh_1 * x_head + (long long)col;
                        {
                            uint4 _uv4_378 = *reinterpret_cast<const uint4*>(x + vxoff_1 + 0);
                            la_x[0 + 0] = _uv4_378.x;
                            la_x[0 + 1] = _uv4_378.y;
                            la_x[0 + 2] = _uv4_378.z;
                            la_x[0 + 3] = _uv4_378.w;
                        }
                        long long vzoff_1 = (long long)vq_1 * z_tok + (long long)vh_1 * z_head + (long long)col;
                        {
                            uint4 _uv4_379 = *reinterpret_cast<const uint4*>(z + vzoff_1 + 0);
                            la_z[0 + 0] = _uv4_379.x;
                            la_z[0 + 1] = _uv4_379.y;
                            la_z[0 + 2] = _uv4_379.z;
                            la_z[0 + 3] = _uv4_379.w;
                        }
                    }
                    if (ahead != 0) {
                        int nrow = (((u_4 + stride) * 16 + group) < (rows - 1) ? ((u_4 + stride) * 16 + group) : (rows - 1));
                        uint32_t nq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)nrow), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)nrow);
                        uint32_t nh = (uint32_t)((unsigned int)nrow) - nq * (uint32_t)(heads.divisor);
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dy + ((long long)nq * dy_tok + (long long)nh * dy_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + ((long long)nq * x_tok + (long long)nh * x_head + (long long)col))));
                        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(z + ((long long)nq * z_tok + (long long)nh * z_head + (long long)col))));
                    }
                    float total_4 = 0.0f;
                    total_4 = __fmaf_rn(xr_4[0], xr_4[0], total_4);
                    total_4 = __fmaf_rn(xr_4[1], xr_4[1], total_4);
                    total_4 = __fmaf_rn(xr_4[2], xr_4[2], total_4);
                    total_4 = __fmaf_rn(xr_4[3], xr_4[3], total_4);
                    total_4 = __fmaf_rn(xr_4[4], xr_4[4], total_4);
                    total_4 = __fmaf_rn(xr_4[5], xr_4[5], total_4);
                    total_4 = __fmaf_rn(xr_4[6], xr_4[6], total_4);
                    total_4 = __fmaf_rn(xr_4[7], xr_4[7], total_4);
                    float _shfl_xor_sync_380 = __shfl_xor_sync(0xFFFFFFFF, total_4, 1);
                    total_4 = total_4 + _shfl_xor_sync_380;
                    float _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, total_4, 2);
                    total_4 = total_4 + _shfl_xor_sync_381;
                    float _shfl_xor_sync_382 = __shfl_xor_sync(0xFFFFFFFF, total_4, 4);
                    total_4 = total_4 + _shfl_xor_sync_382;
                    float _shfl_xor_sync_383 = __shfl_xor_sync(0xFFFFFFFF, total_4, 8);
                    total_4 = total_4 + _shfl_xor_sync_383;
                    float rstd_4;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_4) : "f"(total_4 * 0.0078125f + eps));
                    float dot_4 = 0.0f;
                    float wk_4[2];
                    unsigned int packed_dz_4[4];
                    float2 rstd2_4 = make_float2(rstd_4, rstd_4);
                    float2 z2_16 = make_float2(zr_4[0], zr_4[1]);
                    float2 _f32x2_mul_b_384 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_16) : "l"(*(const unsigned long long*)&z2_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_384));
                    float _fma_385;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_385) : "f"(zr_4[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_386;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_386) : "f"(zr_4[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_16 = make_float2(_fma_385, _fma_386);
                    float _exp2_387 = approx_exp2(ep2_16.x);
                    float _exp2_388 = approx_exp2(ep2_16.y);
                    float2 dq2_16 = fma_f32x2_rn_noftz(make_float2(_exp2_387, _exp2_388), qr2_16, make_float2(0.25f, 0.25f));
                    float _rcp_389 = approx_rcp(dq2_16.x);
                    float _rcp_390 = approx_rcp(dq2_16.y);
                    float2 _f32x2_mul_a_391 = make_float2(_rcp_389, _rcp_390);
                    float2 _f32x2_mul_b_392 = make_float2(0.25f, 0.25f);
                    float2 sig2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_391), "l"(*(const unsigned long long*)&_f32x2_mul_b_392));
                    float2 gate2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_16) : "l"(*(const unsigned long long*)&z2_16), "l"(*(const unsigned long long*)&sig2_16));
                    float2 g2_16 = make_float2(gr_4[0], gr_4[1]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_4[_pair * 2])[0]), "=f"((&wk_4[_pair * 2])[1])
                            : "r"(wbits[_pair]));
                    }
                    float2 w2_16 = make_float2(wk_4[0], wk_4[1]);
                    float2 _f32x2_mul_a_393 = make_float2(xr_4[0], xr_4[1]);
                    float2 n2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_393), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_394 = mul_f32x2_noftz(mul_f32x2_noftz(g2_16, mul_f32x2_noftz(n2_16, w2_16)), sig2_16);
                    float2 _f32x2_mul_b_395 = fma_f32x2_rn_noftz(z2_16, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_16), make_float2(1.0f, 1.0f));
                    float2 dz2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_394), "l"(*(const unsigned long long*)&_f32x2_mul_b_395));
                    float2 a2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_16) : "l"(*(const unsigned long long*)&g2_16), "l"(*(const unsigned long long*)&gate2_16));
                    float2 dn2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_16) : "l"(*(const unsigned long long*)&a2_16), "l"(*(const unsigned long long*)&w2_16));
                    float2 acc2_16 = fma_f32x2_rn_noftz(a2_16, n2_16, make_float2(acc_r[0], acc_r[1]));
                    acc_r[0] = acc2_16.x;
                    acc_r[1] = acc2_16.y;
                    zr_4[0] = dz2_16.x;
                    zr_4[1] = dz2_16.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_4[_lp*2 + 0], zr_4[_lp*2+1 + 0]));
                        packed_dz_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    gr_4[0] = dn2_16.x;
                    gr_4[1] = dn2_16.y;
                    float2 z2_17 = make_float2(zr_4[2], zr_4[3]);
                    float2 _f32x2_mul_b_396 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_17) : "l"(*(const unsigned long long*)&z2_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_396));
                    float _fma_397;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(zr_4[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_398;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_398) : "f"(zr_4[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_17 = make_float2(_fma_397, _fma_398);
                    float _exp2_399 = approx_exp2(ep2_17.x);
                    float _exp2_400 = approx_exp2(ep2_17.y);
                    float2 dq2_17 = fma_f32x2_rn_noftz(make_float2(_exp2_399, _exp2_400), qr2_17, make_float2(0.25f, 0.25f));
                    float _rcp_401 = approx_rcp(dq2_17.x);
                    float _rcp_402 = approx_rcp(dq2_17.y);
                    float2 _f32x2_mul_a_403 = make_float2(_rcp_401, _rcp_402);
                    float2 _f32x2_mul_b_404 = make_float2(0.25f, 0.25f);
                    float2 sig2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_403), "l"(*(const unsigned long long*)&_f32x2_mul_b_404));
                    float2 gate2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_17) : "l"(*(const unsigned long long*)&z2_17), "l"(*(const unsigned long long*)&sig2_17));
                    float2 g2_17 = make_float2(gr_4[2], gr_4[3]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_4[_pair * 2])[0]), "=f"((&wk_4[_pair * 2])[1])
                            : "r"(wbits[1 + _pair]));
                    }
                    float2 w2_17 = make_float2(wk_4[0], wk_4[1]);
                    float2 _f32x2_mul_a_405 = make_float2(xr_4[2], xr_4[3]);
                    float2 n2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_405), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_406 = mul_f32x2_noftz(mul_f32x2_noftz(g2_17, mul_f32x2_noftz(n2_17, w2_17)), sig2_17);
                    float2 _f32x2_mul_b_407 = fma_f32x2_rn_noftz(z2_17, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_17), make_float2(1.0f, 1.0f));
                    float2 dz2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_406), "l"(*(const unsigned long long*)&_f32x2_mul_b_407));
                    float2 a2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_17) : "l"(*(const unsigned long long*)&g2_17), "l"(*(const unsigned long long*)&gate2_17));
                    float2 dn2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_17) : "l"(*(const unsigned long long*)&a2_17), "l"(*(const unsigned long long*)&w2_17));
                    float2 acc2_17 = fma_f32x2_rn_noftz(a2_17, n2_17, make_float2(acc_r[2], acc_r[3]));
                    acc_r[2] = acc2_17.x;
                    acc_r[3] = acc2_17.y;
                    zr_4[2] = dz2_17.x;
                    zr_4[3] = dz2_17.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_4[_lp*2 + 2], zr_4[_lp*2+1 + 2]));
                        packed_dz_4[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    gr_4[2] = dn2_17.x;
                    gr_4[3] = dn2_17.y;
                    float2 z2_18 = make_float2(zr_4[4], zr_4[5]);
                    float2 _f32x2_mul_b_408 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_18) : "l"(*(const unsigned long long*)&z2_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_408));
                    float _fma_409;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_409) : "f"(zr_4[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_410;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_410) : "f"(zr_4[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_18 = make_float2(_fma_409, _fma_410);
                    float _exp2_411 = approx_exp2(ep2_18.x);
                    float _exp2_412 = approx_exp2(ep2_18.y);
                    float2 dq2_18 = fma_f32x2_rn_noftz(make_float2(_exp2_411, _exp2_412), qr2_18, make_float2(0.25f, 0.25f));
                    float _rcp_413 = approx_rcp(dq2_18.x);
                    float _rcp_414 = approx_rcp(dq2_18.y);
                    float2 _f32x2_mul_a_415 = make_float2(_rcp_413, _rcp_414);
                    float2 _f32x2_mul_b_416 = make_float2(0.25f, 0.25f);
                    float2 sig2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_415), "l"(*(const unsigned long long*)&_f32x2_mul_b_416));
                    float2 gate2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_18) : "l"(*(const unsigned long long*)&z2_18), "l"(*(const unsigned long long*)&sig2_18));
                    float2 g2_18 = make_float2(gr_4[4], gr_4[5]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_4[_pair * 2])[0]), "=f"((&wk_4[_pair * 2])[1])
                            : "r"(wbits[2 + _pair]));
                    }
                    float2 w2_18 = make_float2(wk_4[0], wk_4[1]);
                    float2 _f32x2_mul_a_417 = make_float2(xr_4[4], xr_4[5]);
                    float2 n2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_417), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_418 = mul_f32x2_noftz(mul_f32x2_noftz(g2_18, mul_f32x2_noftz(n2_18, w2_18)), sig2_18);
                    float2 _f32x2_mul_b_419 = fma_f32x2_rn_noftz(z2_18, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_18), make_float2(1.0f, 1.0f));
                    float2 dz2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_418), "l"(*(const unsigned long long*)&_f32x2_mul_b_419));
                    float2 a2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_18) : "l"(*(const unsigned long long*)&g2_18), "l"(*(const unsigned long long*)&gate2_18));
                    float2 dn2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_18) : "l"(*(const unsigned long long*)&a2_18), "l"(*(const unsigned long long*)&w2_18));
                    float2 acc2_18 = fma_f32x2_rn_noftz(a2_18, n2_18, make_float2(acc_r[4], acc_r[5]));
                    acc_r[4] = acc2_18.x;
                    acc_r[5] = acc2_18.y;
                    zr_4[4] = dz2_18.x;
                    zr_4[5] = dz2_18.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_4[_lp*2 + 4], zr_4[_lp*2+1 + 4]));
                        packed_dz_4[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    gr_4[4] = dn2_18.x;
                    gr_4[5] = dn2_18.y;
                    float2 z2_19 = make_float2(zr_4[6], zr_4[7]);
                    float2 _f32x2_mul_b_420 = make_float2(-1.442688f, -1.442688f);
                    float2 ep2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_19) : "l"(*(const unsigned long long*)&z2_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_420));
                    float _fma_421;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_421) : "f"(zr_4[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float _fma_422;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_422) : "f"(zr_4[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                    float2 qr2_19 = make_float2(_fma_421, _fma_422);
                    float _exp2_423 = approx_exp2(ep2_19.x);
                    float _exp2_424 = approx_exp2(ep2_19.y);
                    float2 dq2_19 = fma_f32x2_rn_noftz(make_float2(_exp2_423, _exp2_424), qr2_19, make_float2(0.25f, 0.25f));
                    float _rcp_425 = approx_rcp(dq2_19.x);
                    float _rcp_426 = approx_rcp(dq2_19.y);
                    float2 _f32x2_mul_a_427 = make_float2(_rcp_425, _rcp_426);
                    float2 _f32x2_mul_b_428 = make_float2(0.25f, 0.25f);
                    float2 sig2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_427), "l"(*(const unsigned long long*)&_f32x2_mul_b_428));
                    float2 gate2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_19) : "l"(*(const unsigned long long*)&z2_19), "l"(*(const unsigned long long*)&sig2_19));
                    float2 g2_19 = make_float2(gr_4[6], gr_4[7]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&wk_4[_pair * 2])[0]), "=f"((&wk_4[_pair * 2])[1])
                            : "r"(wbits[3 + _pair]));
                    }
                    float2 w2_19 = make_float2(wk_4[0], wk_4[1]);
                    float2 _f32x2_mul_a_429 = make_float2(xr_4[6], xr_4[7]);
                    float2 n2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_429), "l"(*(const unsigned long long*)&rstd2_4));
                    float2 _f32x2_mul_a_430 = mul_f32x2_noftz(mul_f32x2_noftz(g2_19, mul_f32x2_noftz(n2_19, w2_19)), sig2_19);
                    float2 _f32x2_mul_b_431 = fma_f32x2_rn_noftz(z2_19, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_19), make_float2(1.0f, 1.0f));
                    float2 dz2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_430), "l"(*(const unsigned long long*)&_f32x2_mul_b_431));
                    float2 a2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_19) : "l"(*(const unsigned long long*)&g2_19), "l"(*(const unsigned long long*)&gate2_19));
                    float2 dn2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_19) : "l"(*(const unsigned long long*)&a2_19), "l"(*(const unsigned long long*)&w2_19));
                    float2 acc2_19 = fma_f32x2_rn_noftz(a2_19, n2_19, make_float2(acc_r[6], acc_r[7]));
                    acc_r[6] = acc2_19.x;
                    acc_r[7] = acc2_19.y;
                    zr_4[6] = dz2_19.x;
                    zr_4[7] = dz2_19.y;
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_4[_lp*2 + 6], zr_4[_lp*2+1 + 6]));
                        packed_dz_4[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    gr_4[6] = dn2_19.x;
                    gr_4[7] = dn2_19.y;
                    dot_4 = __fmaf_rn(gr_4[0], xr_4[0], dot_4);
                    dot_4 = __fmaf_rn(gr_4[1], xr_4[1], dot_4);
                    dot_4 = __fmaf_rn(gr_4[2], xr_4[2], dot_4);
                    dot_4 = __fmaf_rn(gr_4[3], xr_4[3], dot_4);
                    dot_4 = __fmaf_rn(gr_4[4], xr_4[4], dot_4);
                    dot_4 = __fmaf_rn(gr_4[5], xr_4[5], dot_4);
                    dot_4 = __fmaf_rn(gr_4[6], xr_4[6], dot_4);
                    dot_4 = __fmaf_rn(gr_4[7], xr_4[7], dot_4);
                    float _shfl_xor_sync_432 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                    dot_4 = dot_4 + _shfl_xor_sync_432;
                    float _shfl_xor_sync_433 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                    dot_4 = dot_4 + _shfl_xor_sync_433;
                    float _shfl_xor_sync_434 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                    dot_4 = dot_4 + _shfl_xor_sync_434;
                    float _shfl_xor_sync_435 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 8);
                    dot_4 = dot_4 + _shfl_xor_sync_435;
                    float coef_4 = rstd_4 * rstd_4 * rstd_4 * (dot_4 * -0.0078125f);
                    float2 coef2_4 = make_float2(coef_4, coef_4);
                    float2 rstd2b_4 = make_float2(rstd_4, rstd_4);
                    float2 dx2_16 = fma_f32x2_rn_noftz(make_float2(xr_4[0], xr_4[1]), coef2_4, mul_f32x2_noftz(make_float2(gr_4[0], gr_4[1]), rstd2b_4));
                    xr_4[0] = dx2_16.x;
                    xr_4[1] = dx2_16.y;
                    float2 dx2_17 = fma_f32x2_rn_noftz(make_float2(xr_4[2], xr_4[3]), coef2_4, mul_f32x2_noftz(make_float2(gr_4[2], gr_4[3]), rstd2b_4));
                    xr_4[2] = dx2_17.x;
                    xr_4[3] = dx2_17.y;
                    float2 dx2_18 = fma_f32x2_rn_noftz(make_float2(xr_4[4], xr_4[5]), coef2_4, mul_f32x2_noftz(make_float2(gr_4[4], gr_4[5]), rstd2b_4));
                    xr_4[4] = dx2_18.x;
                    xr_4[5] = dx2_18.y;
                    float2 dx2_19 = fma_f32x2_rn_noftz(make_float2(xr_4[6], xr_4[7]), coef2_4, mul_f32x2_noftz(make_float2(gr_4[6], gr_4[7]), rstd2b_4));
                    xr_4[6] = dx2_19.x;
                    xr_4[7] = dx2_19.y;
                    unsigned int packed_dx_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_4[_lp*2 + 0], xr_4[_lp*2+1 + 0]));
                        packed_dx_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_4[_lp*2 + 2], xr_4[_lp*2+1 + 2]));
                        packed_dx_4[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_4[_lp*2 + 4], xr_4[_lp*2+1 + 4]));
                        packed_dx_4[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_4[_lp*2 + 6], xr_4[_lp*2+1 + 6]));
                        packed_dx_4[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    long long orow_4 = (long long)row_4 * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_4)[0] = reinterpret_cast<int4*>(packed_dx_4)[0];
                    reinterpret_cast<int4*>(dz + orow_4)[0] = reinterpret_cast<int4*>(packed_dz_4)[0];
                    if ((i_1 & 15) == 15) {
                        if (i_1 >= 16) {
                            float s_12;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_12) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                            float bv_12;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_12) : "f"(s_12), "f"(fold[group * 128 + col]));
                            float av_12;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_12) : "f"(s_12), "f"(bv_12));
                            float _sub_436;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_436) : "f"(fold[group * 128 + col]), "f"(av_12));
                            float _sub_437;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_437) : "f"(acc_r[0]), "f"(bv_12));
                            float e_12;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_12) : "f"(_sub_436), "f"(_sub_437));
                            fold[group * 128 + col] = s_12;
                            float _add_438;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_438) : "f"(fold[(16 + group) * 128 + col]), "f"(e_12));
                            fold[(16 + group) * 128 + col] = _add_438;
                            acc_r[0] = 0.0f;
                            float s_49;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_49) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                            float bv_49;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_49) : "f"(s_49), "f"(fold[group * 128 + col + 1]));
                            float av_49;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_49) : "f"(s_49), "f"(bv_49));
                            float _sub_439;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_439) : "f"(fold[group * 128 + col + 1]), "f"(av_49));
                            float _sub_440;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_440) : "f"(acc_r[1]), "f"(bv_49));
                            float e_49;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_49) : "f"(_sub_439), "f"(_sub_440));
                            fold[group * 128 + col + 1] = s_49;
                            float _add_441;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_441) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_49));
                            fold[(16 + group) * 128 + col + 1] = _add_441;
                            acc_r[1] = 0.0f;
                            float s_50;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_50) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                            float bv_50;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_50) : "f"(s_50), "f"(fold[group * 128 + col + 2]));
                            float av_50;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_50) : "f"(s_50), "f"(bv_50));
                            float _sub_442;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_442) : "f"(fold[group * 128 + col + 2]), "f"(av_50));
                            float _sub_443;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_443) : "f"(acc_r[2]), "f"(bv_50));
                            float e_50;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_50) : "f"(_sub_442), "f"(_sub_443));
                            fold[group * 128 + col + 2] = s_50;
                            float _add_444;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_444) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_50));
                            fold[(16 + group) * 128 + col + 2] = _add_444;
                            acc_r[2] = 0.0f;
                            float s_51;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_51) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                            float bv_51;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_51) : "f"(s_51), "f"(fold[group * 128 + col + 3]));
                            float av_51;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_51) : "f"(s_51), "f"(bv_51));
                            float _sub_445;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_445) : "f"(fold[group * 128 + col + 3]), "f"(av_51));
                            float _sub_446;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_446) : "f"(acc_r[3]), "f"(bv_51));
                            float e_51;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_51) : "f"(_sub_445), "f"(_sub_446));
                            fold[group * 128 + col + 3] = s_51;
                            float _add_447;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_447) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_51));
                            fold[(16 + group) * 128 + col + 3] = _add_447;
                            acc_r[3] = 0.0f;
                            float s_52;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_52) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                            float bv_52;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_52) : "f"(s_52), "f"(fold[group * 128 + col + 4]));
                            float av_52;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_52) : "f"(s_52), "f"(bv_52));
                            float _sub_448;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_448) : "f"(fold[group * 128 + col + 4]), "f"(av_52));
                            float _sub_449;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_449) : "f"(acc_r[4]), "f"(bv_52));
                            float e_52;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_52) : "f"(_sub_448), "f"(_sub_449));
                            fold[group * 128 + col + 4] = s_52;
                            float _add_450;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_450) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_52));
                            fold[(16 + group) * 128 + col + 4] = _add_450;
                            acc_r[4] = 0.0f;
                            float s_53;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_53) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                            float bv_53;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_53) : "f"(s_53), "f"(fold[group * 128 + col + 5]));
                            float av_53;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_53) : "f"(s_53), "f"(bv_53));
                            float _sub_451;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_451) : "f"(fold[group * 128 + col + 5]), "f"(av_53));
                            float _sub_452;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_452) : "f"(acc_r[5]), "f"(bv_53));
                            float e_53;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_53) : "f"(_sub_451), "f"(_sub_452));
                            fold[group * 128 + col + 5] = s_53;
                            float _add_453;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_453) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_53));
                            fold[(16 + group) * 128 + col + 5] = _add_453;
                            acc_r[5] = 0.0f;
                            float s_54;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_54) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                            float bv_54;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_54) : "f"(s_54), "f"(fold[group * 128 + col + 6]));
                            float av_54;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_54) : "f"(s_54), "f"(bv_54));
                            float _sub_454;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_454) : "f"(fold[group * 128 + col + 6]), "f"(av_54));
                            float _sub_455;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_455) : "f"(acc_r[6]), "f"(bv_54));
                            float e_54;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_54) : "f"(_sub_454), "f"(_sub_455));
                            fold[group * 128 + col + 6] = s_54;
                            float _add_456;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_456) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_54));
                            fold[(16 + group) * 128 + col + 6] = _add_456;
                            acc_r[6] = 0.0f;
                            float s_55;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(s_55) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                            float bv_55;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_55) : "f"(s_55), "f"(fold[group * 128 + col + 7]));
                            float av_55;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_55) : "f"(s_55), "f"(bv_55));
                            float _sub_457;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_457) : "f"(fold[group * 128 + col + 7]), "f"(av_55));
                            float _sub_458;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_458) : "f"(acc_r[7]), "f"(bv_55));
                            float e_55;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_55) : "f"(_sub_457), "f"(_sub_458));
                            fold[group * 128 + col + 7] = s_55;
                            float _add_459;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_459) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_55));
                            fold[(16 + group) * 128 + col + 7] = _add_459;
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
                }
                int tail_i_3 = chunk_n - 1;
                int u_5 = bound + chunk_lo + tail_i_3 * stride;
                int row_5 = u_5 * 16 + group;
                int src_3 = ((row_5) < (rows - 1) ? (row_5) : (rows - 1));
                uint32_t tq_1 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_3), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_3);
                uint32_t th_1 = (uint32_t)((unsigned int)src_3) - tq_1 * (uint32_t)(heads.divisor);
                long long goff_4 = (long long)tq_1 * dy_tok + (long long)th_1 * dy_head + (long long)col;
                long long xoff_3 = (long long)tq_1 * x_tok + (long long)th_1 * x_head + (long long)col;
                long long zoff_3 = (long long)tq_1 * z_tok + (long long)th_1 * z_head + (long long)col;
                float gr_5[8];
                float xr_5[8];
                float zr_5[8];
                {
                    const uint4* _vptr_460 = reinterpret_cast<const uint4*>(dy + goff_4 + 0);
                    uint4 _vld_460[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_460[_blk] = _vptr_460[_blk];
                        uint32_t* _vpairs_460 = reinterpret_cast<uint32_t*>(&_vld_460[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_460[_pair]) << 16);
                            (&gr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_460[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_461 = reinterpret_cast<const uint4*>(x + xoff_3 + 0);
                    uint4 _vld_461[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_461[_blk] = _vptr_461[_blk];
                        uint32_t* _vpairs_461 = reinterpret_cast<uint32_t*>(&_vld_461[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_461[_pair]) << 16);
                            (&xr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_461[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_462 = reinterpret_cast<const uint4*>(z + zoff_3 + 0);
                    uint4 _vld_462[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_462[_blk] = _vptr_462[_blk];
                        uint32_t* _vpairs_462 = reinterpret_cast<uint32_t*>(&_vld_462[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zr_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_462[_pair]) << 16);
                            (&zr_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_462[_pair]) & 0xffff0000u);
                        }
                    }
                }
                if (tid == 0) {
                    if (trips - 1 - it > 0) {
                        uint32_t _atomic_fetch_inc_463;
                        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_463) : "l"(&done[1]), "r"(static_cast<uint32_t>(2147483647)) : "memory");
                        flag[1] = (int)_atomic_fetch_inc_463;
                    }
                }
                float total_5 = 0.0f;
                total_5 = __fmaf_rn(xr_5[0], xr_5[0], total_5);
                total_5 = __fmaf_rn(xr_5[1], xr_5[1], total_5);
                total_5 = __fmaf_rn(xr_5[2], xr_5[2], total_5);
                total_5 = __fmaf_rn(xr_5[3], xr_5[3], total_5);
                total_5 = __fmaf_rn(xr_5[4], xr_5[4], total_5);
                total_5 = __fmaf_rn(xr_5[5], xr_5[5], total_5);
                total_5 = __fmaf_rn(xr_5[6], xr_5[6], total_5);
                total_5 = __fmaf_rn(xr_5[7], xr_5[7], total_5);
                float _shfl_xor_sync_464 = __shfl_xor_sync(0xFFFFFFFF, total_5, 1);
                total_5 = total_5 + _shfl_xor_sync_464;
                float _shfl_xor_sync_465 = __shfl_xor_sync(0xFFFFFFFF, total_5, 2);
                total_5 = total_5 + _shfl_xor_sync_465;
                float _shfl_xor_sync_466 = __shfl_xor_sync(0xFFFFFFFF, total_5, 4);
                total_5 = total_5 + _shfl_xor_sync_466;
                float _shfl_xor_sync_467 = __shfl_xor_sync(0xFFFFFFFF, total_5, 8);
                total_5 = total_5 + _shfl_xor_sync_467;
                float rstd_5;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd_5) : "f"(total_5 * 0.0078125f + eps));
                float dot_5 = 0.0f;
                float wk_5[2];
                unsigned int packed_dz_5[4];
                float2 rstd2_5 = make_float2(rstd_5, rstd_5);
                float2 z2_20 = make_float2(zr_5[0], zr_5[1]);
                float2 _f32x2_mul_b_468 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_20) : "l"(*(const unsigned long long*)&z2_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_468));
                float _fma_469;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_469) : "f"(zr_5[0]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_470;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_470) : "f"(zr_5[1]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_20 = make_float2(_fma_469, _fma_470);
                float _exp2_471 = approx_exp2(ep2_20.x);
                float _exp2_472 = approx_exp2(ep2_20.y);
                float2 dq2_20 = fma_f32x2_rn_noftz(make_float2(_exp2_471, _exp2_472), qr2_20, make_float2(0.25f, 0.25f));
                float _rcp_473 = approx_rcp(dq2_20.x);
                float _rcp_474 = approx_rcp(dq2_20.y);
                float2 _f32x2_mul_a_475 = make_float2(_rcp_473, _rcp_474);
                float2 _f32x2_mul_b_476 = make_float2(0.25f, 0.25f);
                float2 sig2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_475), "l"(*(const unsigned long long*)&_f32x2_mul_b_476));
                float2 gate2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_20) : "l"(*(const unsigned long long*)&z2_20), "l"(*(const unsigned long long*)&sig2_20));
                float2 g2_20 = make_float2(gr_5[0], gr_5[1]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_5[_pair * 2])[0]), "=f"((&wk_5[_pair * 2])[1])
                        : "r"(wbits[_pair]));
                }
                float2 w2_20 = make_float2(wk_5[0], wk_5[1]);
                float2 _f32x2_mul_a_477 = make_float2(xr_5[0], xr_5[1]);
                float2 n2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_477), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_478 = mul_f32x2_noftz(mul_f32x2_noftz(g2_20, mul_f32x2_noftz(n2_20, w2_20)), sig2_20);
                float2 _f32x2_mul_b_479 = fma_f32x2_rn_noftz(z2_20, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_20), make_float2(1.0f, 1.0f));
                float2 dz2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_478), "l"(*(const unsigned long long*)&_f32x2_mul_b_479));
                float2 a2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_20) : "l"(*(const unsigned long long*)&g2_20), "l"(*(const unsigned long long*)&gate2_20));
                float2 dn2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_20) : "l"(*(const unsigned long long*)&a2_20), "l"(*(const unsigned long long*)&w2_20));
                float2 acc2_20 = fma_f32x2_rn_noftz(a2_20, n2_20, make_float2(acc_r[0], acc_r[1]));
                acc_r[0] = ((row_5 < rows) ? acc2_20.x : acc_r[0]);
                acc_r[1] = ((row_5 < rows) ? acc2_20.y : acc_r[1]);
                zr_5[0] = dz2_20.x;
                zr_5[1] = dz2_20.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_5[_lp*2 + 0], zr_5[_lp*2+1 + 0]));
                    packed_dz_5[_lp] = *(uint32_t*)&_bf2;
                }
                gr_5[0] = dn2_20.x;
                gr_5[1] = dn2_20.y;
                float2 z2_21 = make_float2(zr_5[2], zr_5[3]);
                float2 _f32x2_mul_b_480 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_21) : "l"(*(const unsigned long long*)&z2_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_480));
                float _fma_481;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_481) : "f"(zr_5[2]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_482;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_482) : "f"(zr_5[3]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_21 = make_float2(_fma_481, _fma_482);
                float _exp2_483 = approx_exp2(ep2_21.x);
                float _exp2_484 = approx_exp2(ep2_21.y);
                float2 dq2_21 = fma_f32x2_rn_noftz(make_float2(_exp2_483, _exp2_484), qr2_21, make_float2(0.25f, 0.25f));
                float _rcp_485 = approx_rcp(dq2_21.x);
                float _rcp_486 = approx_rcp(dq2_21.y);
                float2 _f32x2_mul_a_487 = make_float2(_rcp_485, _rcp_486);
                float2 _f32x2_mul_b_488 = make_float2(0.25f, 0.25f);
                float2 sig2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_487), "l"(*(const unsigned long long*)&_f32x2_mul_b_488));
                float2 gate2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_21) : "l"(*(const unsigned long long*)&z2_21), "l"(*(const unsigned long long*)&sig2_21));
                float2 g2_21 = make_float2(gr_5[2], gr_5[3]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_5[_pair * 2])[0]), "=f"((&wk_5[_pair * 2])[1])
                        : "r"(wbits[1 + _pair]));
                }
                float2 w2_21 = make_float2(wk_5[0], wk_5[1]);
                float2 _f32x2_mul_a_489 = make_float2(xr_5[2], xr_5[3]);
                float2 n2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_489), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_490 = mul_f32x2_noftz(mul_f32x2_noftz(g2_21, mul_f32x2_noftz(n2_21, w2_21)), sig2_21);
                float2 _f32x2_mul_b_491 = fma_f32x2_rn_noftz(z2_21, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_21), make_float2(1.0f, 1.0f));
                float2 dz2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_490), "l"(*(const unsigned long long*)&_f32x2_mul_b_491));
                float2 a2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_21) : "l"(*(const unsigned long long*)&g2_21), "l"(*(const unsigned long long*)&gate2_21));
                float2 dn2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_21) : "l"(*(const unsigned long long*)&a2_21), "l"(*(const unsigned long long*)&w2_21));
                float2 acc2_21 = fma_f32x2_rn_noftz(a2_21, n2_21, make_float2(acc_r[2], acc_r[3]));
                acc_r[2] = ((row_5 < rows) ? acc2_21.x : acc_r[2]);
                acc_r[3] = ((row_5 < rows) ? acc2_21.y : acc_r[3]);
                zr_5[2] = dz2_21.x;
                zr_5[3] = dz2_21.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_5[_lp*2 + 2], zr_5[_lp*2+1 + 2]));
                    packed_dz_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                gr_5[2] = dn2_21.x;
                gr_5[3] = dn2_21.y;
                float2 z2_22 = make_float2(zr_5[4], zr_5[5]);
                float2 _f32x2_mul_b_492 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_22) : "l"(*(const unsigned long long*)&z2_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_492));
                float _fma_493;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_493) : "f"(zr_5[4]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_494;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_494) : "f"(zr_5[5]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_22 = make_float2(_fma_493, _fma_494);
                float _exp2_495 = approx_exp2(ep2_22.x);
                float _exp2_496 = approx_exp2(ep2_22.y);
                float2 dq2_22 = fma_f32x2_rn_noftz(make_float2(_exp2_495, _exp2_496), qr2_22, make_float2(0.25f, 0.25f));
                float _rcp_497 = approx_rcp(dq2_22.x);
                float _rcp_498 = approx_rcp(dq2_22.y);
                float2 _f32x2_mul_a_499 = make_float2(_rcp_497, _rcp_498);
                float2 _f32x2_mul_b_500 = make_float2(0.25f, 0.25f);
                float2 sig2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_499), "l"(*(const unsigned long long*)&_f32x2_mul_b_500));
                float2 gate2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_22) : "l"(*(const unsigned long long*)&z2_22), "l"(*(const unsigned long long*)&sig2_22));
                float2 g2_22 = make_float2(gr_5[4], gr_5[5]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_5[_pair * 2])[0]), "=f"((&wk_5[_pair * 2])[1])
                        : "r"(wbits[2 + _pair]));
                }
                float2 w2_22 = make_float2(wk_5[0], wk_5[1]);
                float2 _f32x2_mul_a_501 = make_float2(xr_5[4], xr_5[5]);
                float2 n2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_501), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_502 = mul_f32x2_noftz(mul_f32x2_noftz(g2_22, mul_f32x2_noftz(n2_22, w2_22)), sig2_22);
                float2 _f32x2_mul_b_503 = fma_f32x2_rn_noftz(z2_22, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_22), make_float2(1.0f, 1.0f));
                float2 dz2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_502), "l"(*(const unsigned long long*)&_f32x2_mul_b_503));
                float2 a2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_22) : "l"(*(const unsigned long long*)&g2_22), "l"(*(const unsigned long long*)&gate2_22));
                float2 dn2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_22) : "l"(*(const unsigned long long*)&a2_22), "l"(*(const unsigned long long*)&w2_22));
                float2 acc2_22 = fma_f32x2_rn_noftz(a2_22, n2_22, make_float2(acc_r[4], acc_r[5]));
                acc_r[4] = ((row_5 < rows) ? acc2_22.x : acc_r[4]);
                acc_r[5] = ((row_5 < rows) ? acc2_22.y : acc_r[5]);
                zr_5[4] = dz2_22.x;
                zr_5[5] = dz2_22.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_5[_lp*2 + 4], zr_5[_lp*2+1 + 4]));
                    packed_dz_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                gr_5[4] = dn2_22.x;
                gr_5[5] = dn2_22.y;
                float2 z2_23 = make_float2(zr_5[6], zr_5[7]);
                float2 _f32x2_mul_b_504 = make_float2(-1.442688f, -1.442688f);
                float2 ep2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ep2_23) : "l"(*(const unsigned long long*)&z2_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_504));
                float _fma_505;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_505) : "f"(zr_5[6]), "f"(-1.2221238e-06f), "f"(0.25f));
                float _fma_506;
                asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(_fma_506) : "f"(zr_5[7]), "f"(-1.2221238e-06f), "f"(0.25f));
                float2 qr2_23 = make_float2(_fma_505, _fma_506);
                float _exp2_507 = approx_exp2(ep2_23.x);
                float _exp2_508 = approx_exp2(ep2_23.y);
                float2 dq2_23 = fma_f32x2_rn_noftz(make_float2(_exp2_507, _exp2_508), qr2_23, make_float2(0.25f, 0.25f));
                float _rcp_509 = approx_rcp(dq2_23.x);
                float _rcp_510 = approx_rcp(dq2_23.y);
                float2 _f32x2_mul_a_511 = make_float2(_rcp_509, _rcp_510);
                float2 _f32x2_mul_b_512 = make_float2(0.25f, 0.25f);
                float2 sig2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&sig2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_511), "l"(*(const unsigned long long*)&_f32x2_mul_b_512));
                float2 gate2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate2_23) : "l"(*(const unsigned long long*)&z2_23), "l"(*(const unsigned long long*)&sig2_23));
                float2 g2_23 = make_float2(gr_5[6], gr_5[7]);
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&wk_5[_pair * 2])[0]), "=f"((&wk_5[_pair * 2])[1])
                        : "r"(wbits[3 + _pair]));
                }
                float2 w2_23 = make_float2(wk_5[0], wk_5[1]);
                float2 _f32x2_mul_a_513 = make_float2(xr_5[6], xr_5[7]);
                float2 n2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_513), "l"(*(const unsigned long long*)&rstd2_5));
                float2 _f32x2_mul_a_514 = mul_f32x2_noftz(mul_f32x2_noftz(g2_23, mul_f32x2_noftz(n2_23, w2_23)), sig2_23);
                float2 _f32x2_mul_b_515 = fma_f32x2_rn_noftz(z2_23, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_23), make_float2(1.0f, 1.0f));
                float2 dz2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_514), "l"(*(const unsigned long long*)&_f32x2_mul_b_515));
                float2 a2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_23) : "l"(*(const unsigned long long*)&g2_23), "l"(*(const unsigned long long*)&gate2_23));
                float2 dn2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_23) : "l"(*(const unsigned long long*)&a2_23), "l"(*(const unsigned long long*)&w2_23));
                float2 acc2_23 = fma_f32x2_rn_noftz(a2_23, n2_23, make_float2(acc_r[6], acc_r[7]));
                acc_r[6] = ((row_5 < rows) ? acc2_23.x : acc_r[6]);
                acc_r[7] = ((row_5 < rows) ? acc2_23.y : acc_r[7]);
                zr_5[6] = dz2_23.x;
                zr_5[7] = dz2_23.y;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zr_5[_lp*2 + 6], zr_5[_lp*2+1 + 6]));
                    packed_dz_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                gr_5[6] = dn2_23.x;
                gr_5[7] = dn2_23.y;
                dot_5 = __fmaf_rn(gr_5[0], xr_5[0], dot_5);
                dot_5 = __fmaf_rn(gr_5[1], xr_5[1], dot_5);
                dot_5 = __fmaf_rn(gr_5[2], xr_5[2], dot_5);
                dot_5 = __fmaf_rn(gr_5[3], xr_5[3], dot_5);
                dot_5 = __fmaf_rn(gr_5[4], xr_5[4], dot_5);
                dot_5 = __fmaf_rn(gr_5[5], xr_5[5], dot_5);
                dot_5 = __fmaf_rn(gr_5[6], xr_5[6], dot_5);
                dot_5 = __fmaf_rn(gr_5[7], xr_5[7], dot_5);
                float _shfl_xor_sync_516 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_516;
                float _shfl_xor_sync_517 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_517;
                float _shfl_xor_sync_518 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_518;
                float _shfl_xor_sync_519 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 8);
                dot_5 = dot_5 + _shfl_xor_sync_519;
                float coef_5 = rstd_5 * rstd_5 * rstd_5 * (dot_5 * -0.0078125f);
                float2 coef2_5 = make_float2(coef_5, coef_5);
                float2 rstd2b_5 = make_float2(rstd_5, rstd_5);
                float2 dx2_20 = fma_f32x2_rn_noftz(make_float2(xr_5[0], xr_5[1]), coef2_5, mul_f32x2_noftz(make_float2(gr_5[0], gr_5[1]), rstd2b_5));
                xr_5[0] = dx2_20.x;
                xr_5[1] = dx2_20.y;
                float2 dx2_21 = fma_f32x2_rn_noftz(make_float2(xr_5[2], xr_5[3]), coef2_5, mul_f32x2_noftz(make_float2(gr_5[2], gr_5[3]), rstd2b_5));
                xr_5[2] = dx2_21.x;
                xr_5[3] = dx2_21.y;
                float2 dx2_22 = fma_f32x2_rn_noftz(make_float2(xr_5[4], xr_5[5]), coef2_5, mul_f32x2_noftz(make_float2(gr_5[4], gr_5[5]), rstd2b_5));
                xr_5[4] = dx2_22.x;
                xr_5[5] = dx2_22.y;
                float2 dx2_23 = fma_f32x2_rn_noftz(make_float2(xr_5[6], xr_5[7]), coef2_5, mul_f32x2_noftz(make_float2(gr_5[6], gr_5[7]), rstd2b_5));
                xr_5[6] = dx2_23.x;
                xr_5[7] = dx2_23.y;
                unsigned int packed_dx_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_5[_lp*2 + 0], xr_5[_lp*2+1 + 0]));
                    packed_dx_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_5[_lp*2 + 2], xr_5[_lp*2+1 + 2]));
                    packed_dx_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_5[_lp*2 + 4], xr_5[_lp*2+1 + 4]));
                    packed_dx_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xr_5[_lp*2 + 6], xr_5[_lp*2+1 + 6]));
                    packed_dx_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                long long orow_5 = (long long)row_5 * 128ll + (long long)col;
                if (row_5 < rows) {
                    reinterpret_cast<int4*>(dx + orow_5)[0] = reinterpret_cast<int4*>(packed_dx_5)[0];
                    reinterpret_cast<int4*>(dz + orow_5)[0] = reinterpret_cast<int4*>(packed_dz_5)[0];
                }
                if ((tail_i_3 & 15) == 15) {
                    if (tail_i_3 >= 16) {
                        float s_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_13) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        float bv_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_13) : "f"(s_13), "f"(fold[group * 128 + col]));
                        float av_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_13) : "f"(s_13), "f"(bv_13));
                        float _sub_520;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_520) : "f"(fold[group * 128 + col]), "f"(av_13));
                        float _sub_521;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_521) : "f"(acc_r[0]), "f"(bv_13));
                        float e_13;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_13) : "f"(_sub_520), "f"(_sub_521));
                        fold[group * 128 + col] = s_13;
                        float _add_522;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_522) : "f"(fold[(16 + group) * 128 + col]), "f"(e_13));
                        fold[(16 + group) * 128 + col] = _add_522;
                        acc_r[0] = 0.0f;
                        float s_56;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_56) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        float bv_56;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_56) : "f"(s_56), "f"(fold[group * 128 + col + 1]));
                        float av_56;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_56) : "f"(s_56), "f"(bv_56));
                        float _sub_523;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_523) : "f"(fold[group * 128 + col + 1]), "f"(av_56));
                        float _sub_524;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_524) : "f"(acc_r[1]), "f"(bv_56));
                        float e_56;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_56) : "f"(_sub_523), "f"(_sub_524));
                        fold[group * 128 + col + 1] = s_56;
                        float _add_525;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_525) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_56));
                        fold[(16 + group) * 128 + col + 1] = _add_525;
                        acc_r[1] = 0.0f;
                        float s_57;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_57) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        float bv_57;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_57) : "f"(s_57), "f"(fold[group * 128 + col + 2]));
                        float av_57;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_57) : "f"(s_57), "f"(bv_57));
                        float _sub_526;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_526) : "f"(fold[group * 128 + col + 2]), "f"(av_57));
                        float _sub_527;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_527) : "f"(acc_r[2]), "f"(bv_57));
                        float e_57;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_57) : "f"(_sub_526), "f"(_sub_527));
                        fold[group * 128 + col + 2] = s_57;
                        float _add_528;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_528) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_57));
                        fold[(16 + group) * 128 + col + 2] = _add_528;
                        acc_r[2] = 0.0f;
                        float s_58;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_58) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        float bv_58;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_58) : "f"(s_58), "f"(fold[group * 128 + col + 3]));
                        float av_58;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_58) : "f"(s_58), "f"(bv_58));
                        float _sub_529;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_529) : "f"(fold[group * 128 + col + 3]), "f"(av_58));
                        float _sub_530;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_530) : "f"(acc_r[3]), "f"(bv_58));
                        float e_58;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_58) : "f"(_sub_529), "f"(_sub_530));
                        fold[group * 128 + col + 3] = s_58;
                        float _add_531;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_531) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_58));
                        fold[(16 + group) * 128 + col + 3] = _add_531;
                        acc_r[3] = 0.0f;
                        float s_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_59) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        float bv_59;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_59) : "f"(s_59), "f"(fold[group * 128 + col + 4]));
                        float av_59;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_59) : "f"(s_59), "f"(bv_59));
                        float _sub_532;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_532) : "f"(fold[group * 128 + col + 4]), "f"(av_59));
                        float _sub_533;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_533) : "f"(acc_r[4]), "f"(bv_59));
                        float e_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_59) : "f"(_sub_532), "f"(_sub_533));
                        fold[group * 128 + col + 4] = s_59;
                        float _add_534;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_534) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_59));
                        fold[(16 + group) * 128 + col + 4] = _add_534;
                        acc_r[4] = 0.0f;
                        float s_60;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_60) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        float bv_60;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_60) : "f"(s_60), "f"(fold[group * 128 + col + 5]));
                        float av_60;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_60) : "f"(s_60), "f"(bv_60));
                        float _sub_535;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_535) : "f"(fold[group * 128 + col + 5]), "f"(av_60));
                        float _sub_536;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_536) : "f"(acc_r[5]), "f"(bv_60));
                        float e_60;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_60) : "f"(_sub_535), "f"(_sub_536));
                        fold[group * 128 + col + 5] = s_60;
                        float _add_537;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_537) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_60));
                        fold[(16 + group) * 128 + col + 5] = _add_537;
                        acc_r[5] = 0.0f;
                        float s_61;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_61) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        float bv_61;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_61) : "f"(s_61), "f"(fold[group * 128 + col + 6]));
                        float av_61;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_61) : "f"(s_61), "f"(bv_61));
                        float _sub_538;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_538) : "f"(fold[group * 128 + col + 6]), "f"(av_61));
                        float _sub_539;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_539) : "f"(acc_r[6]), "f"(bv_61));
                        float e_61;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_61) : "f"(_sub_538), "f"(_sub_539));
                        fold[group * 128 + col + 6] = s_61;
                        float _add_540;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_540) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_61));
                        fold[(16 + group) * 128 + col + 6] = _add_540;
                        acc_r[6] = 0.0f;
                        float s_62;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_62) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        float bv_62;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_62) : "f"(s_62), "f"(fold[group * 128 + col + 7]));
                        float av_62;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_62) : "f"(s_62), "f"(bv_62));
                        float _sub_541;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_541) : "f"(fold[group * 128 + col + 7]), "f"(av_62));
                        float _sub_542;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_542) : "f"(acc_r[7]), "f"(bv_62));
                        float e_62;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_62) : "f"(_sub_541), "f"(_sub_542));
                        fold[group * 128 + col + 7] = s_62;
                        float _add_543;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_543) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_62));
                        fold[(16 + group) * 128 + col + 7] = _add_543;
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
                if ((chunk_n & 15) != 0) {
                    if (chunk_n - 1 >= 16) {
                        float s_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_21) : "f"(fold[group * 128 + col]), "f"(acc_r[0]));
                        float bv_21;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_21) : "f"(s_21), "f"(fold[group * 128 + col]));
                        float av_21;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_21) : "f"(s_21), "f"(bv_21));
                        float _sub_544;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_544) : "f"(fold[group * 128 + col]), "f"(av_21));
                        float _sub_545;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_545) : "f"(acc_r[0]), "f"(bv_21));
                        float e_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_21) : "f"(_sub_544), "f"(_sub_545));
                        fold[group * 128 + col] = s_21;
                        float _add_546;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_546) : "f"(fold[(16 + group) * 128 + col]), "f"(e_21));
                        fold[(16 + group) * 128 + col] = _add_546;
                        acc_r[0] = 0.0f;
                        float s_63;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_63) : "f"(fold[group * 128 + col + 1]), "f"(acc_r[1]));
                        float bv_63;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_63) : "f"(s_63), "f"(fold[group * 128 + col + 1]));
                        float av_63;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_63) : "f"(s_63), "f"(bv_63));
                        float _sub_547;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_547) : "f"(fold[group * 128 + col + 1]), "f"(av_63));
                        float _sub_548;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_548) : "f"(acc_r[1]), "f"(bv_63));
                        float e_63;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_63) : "f"(_sub_547), "f"(_sub_548));
                        fold[group * 128 + col + 1] = s_63;
                        float _add_549;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_549) : "f"(fold[(16 + group) * 128 + col + 1]), "f"(e_63));
                        fold[(16 + group) * 128 + col + 1] = _add_549;
                        acc_r[1] = 0.0f;
                        float s_64;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_64) : "f"(fold[group * 128 + col + 2]), "f"(acc_r[2]));
                        float bv_64;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_64) : "f"(s_64), "f"(fold[group * 128 + col + 2]));
                        float av_64;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_64) : "f"(s_64), "f"(bv_64));
                        float _sub_550;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_550) : "f"(fold[group * 128 + col + 2]), "f"(av_64));
                        float _sub_551;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_551) : "f"(acc_r[2]), "f"(bv_64));
                        float e_64;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_64) : "f"(_sub_550), "f"(_sub_551));
                        fold[group * 128 + col + 2] = s_64;
                        float _add_552;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_552) : "f"(fold[(16 + group) * 128 + col + 2]), "f"(e_64));
                        fold[(16 + group) * 128 + col + 2] = _add_552;
                        acc_r[2] = 0.0f;
                        float s_65;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_65) : "f"(fold[group * 128 + col + 3]), "f"(acc_r[3]));
                        float bv_65;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_65) : "f"(s_65), "f"(fold[group * 128 + col + 3]));
                        float av_65;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_65) : "f"(s_65), "f"(bv_65));
                        float _sub_553;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_553) : "f"(fold[group * 128 + col + 3]), "f"(av_65));
                        float _sub_554;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_554) : "f"(acc_r[3]), "f"(bv_65));
                        float e_65;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_65) : "f"(_sub_553), "f"(_sub_554));
                        fold[group * 128 + col + 3] = s_65;
                        float _add_555;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_555) : "f"(fold[(16 + group) * 128 + col + 3]), "f"(e_65));
                        fold[(16 + group) * 128 + col + 3] = _add_555;
                        acc_r[3] = 0.0f;
                        float s_66;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_66) : "f"(fold[group * 128 + col + 4]), "f"(acc_r[4]));
                        float bv_66;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_66) : "f"(s_66), "f"(fold[group * 128 + col + 4]));
                        float av_66;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_66) : "f"(s_66), "f"(bv_66));
                        float _sub_556;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_556) : "f"(fold[group * 128 + col + 4]), "f"(av_66));
                        float _sub_557;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_557) : "f"(acc_r[4]), "f"(bv_66));
                        float e_66;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_66) : "f"(_sub_556), "f"(_sub_557));
                        fold[group * 128 + col + 4] = s_66;
                        float _add_558;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_558) : "f"(fold[(16 + group) * 128 + col + 4]), "f"(e_66));
                        fold[(16 + group) * 128 + col + 4] = _add_558;
                        acc_r[4] = 0.0f;
                        float s_67;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_67) : "f"(fold[group * 128 + col + 5]), "f"(acc_r[5]));
                        float bv_67;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_67) : "f"(s_67), "f"(fold[group * 128 + col + 5]));
                        float av_67;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_67) : "f"(s_67), "f"(bv_67));
                        float _sub_559;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_559) : "f"(fold[group * 128 + col + 5]), "f"(av_67));
                        float _sub_560;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_560) : "f"(acc_r[5]), "f"(bv_67));
                        float e_67;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_67) : "f"(_sub_559), "f"(_sub_560));
                        fold[group * 128 + col + 5] = s_67;
                        float _add_561;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_561) : "f"(fold[(16 + group) * 128 + col + 5]), "f"(e_67));
                        fold[(16 + group) * 128 + col + 5] = _add_561;
                        acc_r[5] = 0.0f;
                        float s_68;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_68) : "f"(fold[group * 128 + col + 6]), "f"(acc_r[6]));
                        float bv_68;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_68) : "f"(s_68), "f"(fold[group * 128 + col + 6]));
                        float av_68;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_68) : "f"(s_68), "f"(bv_68));
                        float _sub_562;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_562) : "f"(fold[group * 128 + col + 6]), "f"(av_68));
                        float _sub_563;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_563) : "f"(acc_r[6]), "f"(bv_68));
                        float e_68;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_68) : "f"(_sub_562), "f"(_sub_563));
                        fold[group * 128 + col + 6] = s_68;
                        float _add_564;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_564) : "f"(fold[(16 + group) * 128 + col + 6]), "f"(e_68));
                        fold[(16 + group) * 128 + col + 6] = _add_564;
                        acc_r[6] = 0.0f;
                        float s_69;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(s_69) : "f"(fold[group * 128 + col + 7]), "f"(acc_r[7]));
                        float bv_69;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_69) : "f"(s_69), "f"(fold[group * 128 + col + 7]));
                        float av_69;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_69) : "f"(s_69), "f"(bv_69));
                        float _sub_565;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_565) : "f"(fold[group * 128 + col + 7]), "f"(av_69));
                        float _sub_566;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_566) : "f"(acc_r[7]), "f"(bv_69));
                        float e_69;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(e_69) : "f"(_sub_565), "f"(_sub_566));
                        fold[group * 128 + col + 7] = s_69;
                        float _add_567;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_567) : "f"(fold[(16 + group) * 128 + col + 7]), "f"(e_69));
                        fold[(16 + group) * 128 + col + 7] = _add_567;
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
            }
        }
        __syncthreads();
        if (chunk_n > 0) {
            if (tid < 128) {
                float hs_1 = 0.0f;
                float ls_1 = 0.0f;
                #pragma unroll
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    float s_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(s_22) : "f"(hs_1), "f"(fold[g_1 * 128 + tid]));
                    float bv_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(bv_22) : "f"(s_22), "f"(hs_1));
                    float av_22;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(av_22) : "f"(s_22), "f"(bv_22));
                    float _sub_568;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_568) : "f"(hs_1), "f"(av_22));
                    float _sub_569;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_569) : "f"(fold[g_1 * 128 + tid]), "f"(bv_22));
                    float e_22;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(e_22) : "f"(_sub_568), "f"(_sub_569));
                    hs_1 = s_22;
                    float _add_571;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_571) : "f"(e_22), "f"(fold[(16 + g_1) * 128 + tid]));
                    float _add_570;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_570) : "f"(ls_1), "f"(_add_571));
                    ls_1 = _add_570;
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
                        unsigned int _max_572 = ((ex_1) > (1) ? (ex_1) : (1));
                        unsigned int pos_2 = _max_572 - 1;
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
                float _abs_573;
                asm("abs.f32 %0, %1;" : "=f"(_abs_573) : "f"(hs_1));
                if (_abs_573 <= 3.4028235e+38f) {
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
                            unsigned int _max_574 = ((ex_1_1) > (1) ? (ex_1_1) : (1));
                            unsigned int pos_3 = _max_574 - 1;
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
                    uint32_t _scalar_bits_575;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_575) : "l"((const void*)(flags + (tid))) : "memory");
                    flw[0] = (unsigned int)_scalar_bits_575;
                }
                unsigned long long dig[18];
                dig[0] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_576;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_576) : "l"((const void*)(acc + (tid))) : "memory");
                        dig[0] = (unsigned long long)_scalar_bits_576;
                    }
                }
                dig[1] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_577;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_577) : "l"((const void*)(acc + (1152 + tid))) : "memory");
                        dig[1] = (unsigned long long)_scalar_bits_577;
                    }
                }
                dig[2] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_578;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_578) : "l"((const void*)(acc + (128 + tid))) : "memory");
                        dig[2] = (unsigned long long)_scalar_bits_578;
                    }
                }
                dig[3] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_579;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_579) : "l"((const void*)(acc + (1280 + tid))) : "memory");
                        dig[3] = (unsigned long long)_scalar_bits_579;
                    }
                }
                dig[4] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_580;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_580) : "l"((const void*)(acc + (256 + tid))) : "memory");
                        dig[4] = (unsigned long long)_scalar_bits_580;
                    }
                }
                dig[5] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_581;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_581) : "l"((const void*)(acc + (1408 + tid))) : "memory");
                        dig[5] = (unsigned long long)_scalar_bits_581;
                    }
                }
                dig[6] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_582;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_582) : "l"((const void*)(acc + (384 + tid))) : "memory");
                        dig[6] = (unsigned long long)_scalar_bits_582;
                    }
                }
                dig[7] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_583;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_583) : "l"((const void*)(acc + (1536 + tid))) : "memory");
                        dig[7] = (unsigned long long)_scalar_bits_583;
                    }
                }
                dig[8] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_584;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_584) : "l"((const void*)(acc + (512 + tid))) : "memory");
                        dig[8] = (unsigned long long)_scalar_bits_584;
                    }
                }
                dig[9] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_585;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_585) : "l"((const void*)(acc + (1664 + tid))) : "memory");
                        dig[9] = (unsigned long long)_scalar_bits_585;
                    }
                }
                dig[10] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_586;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_586) : "l"((const void*)(acc + (640 + tid))) : "memory");
                        dig[10] = (unsigned long long)_scalar_bits_586;
                    }
                }
                dig[11] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_587;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_587) : "l"((const void*)(acc + (1792 + tid))) : "memory");
                        dig[11] = (unsigned long long)_scalar_bits_587;
                    }
                }
                dig[12] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_588;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_588) : "l"((const void*)(acc + (768 + tid))) : "memory");
                        dig[12] = (unsigned long long)_scalar_bits_588;
                    }
                }
                dig[13] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_589;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_589) : "l"((const void*)(acc + (1920 + tid))) : "memory");
                        dig[13] = (unsigned long long)_scalar_bits_589;
                    }
                }
                dig[14] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_590;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_590) : "l"((const void*)(acc + (896 + tid))) : "memory");
                        dig[14] = (unsigned long long)_scalar_bits_590;
                    }
                }
                dig[15] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_591;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_591) : "l"((const void*)(acc + (2048 + tid))) : "memory");
                        dig[15] = (unsigned long long)_scalar_bits_591;
                    }
                }
                dig[16] = 0ull;
                if (num_bids > 0) {
                    {
                        uint64_t _scalar_bits_592;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_592) : "l"((const void*)(acc + (1024 + tid))) : "memory");
                        dig[16] = (unsigned long long)_scalar_bits_592;
                    }
                }
                dig[17] = 0ull;
                if (num_bids > 1) {
                    {
                        uint64_t _scalar_bits_593;
                        asm volatile("ld.global.cg.b64 %0, [%1];"
                            : "=l"(_scalar_bits_593) : "l"((const void*)(acc + (2176 + tid))) : "memory");
                        dig[17] = (unsigned long long)_scalar_bits_593;
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
                            unsigned int _max_594 = ((pex) > (1) ? (pex) : (1));
                            unsigned int ppos = _max_594 - 1;
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
