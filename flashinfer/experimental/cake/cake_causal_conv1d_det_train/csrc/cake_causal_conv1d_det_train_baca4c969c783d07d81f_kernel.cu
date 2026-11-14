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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


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


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
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


#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030)
__device__ __forceinline__ float2 mul_nz_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}
#define CAKE_F32X2_MUL_NZ(a, b, nz) mul_nz_f32x2_rn_noftz(a, b)
#else
#define CAKE_F32X2_MUL_NZ(a, b, nz) fma_f32x2_rn_noftz(a, b, nz)
#endif



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_det_train_baca4c969c783d07d81f(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ partial, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap dy_map)
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

    const int mbar_base = smem;
    #define stage_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* red = reinterpret_cast<float*>(smem_raw + 1024);
    const int red_addr = smem + 1024;
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int stage_buf_addr = smem + 1024;

    // Mbarrier init (24 barriers at smem_raw[0..192))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stage_full: 24 barriers, init_count=32
            mbarrier_init(smem + 0, 32);
            mbarrier_init(smem + 8, 32);
            mbarrier_init(smem + 16, 32);
            mbarrier_init(smem + 24, 32);
            mbarrier_init(smem + 32, 32);
            mbarrier_init(smem + 40, 32);
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            mbarrier_init(smem + 64, 32);
            mbarrier_init(smem + 72, 32);
            mbarrier_init(smem + 80, 32);
            mbarrier_init(smem + 88, 32);
            mbarrier_init(smem + 96, 32);
            mbarrier_init(smem + 104, 32);
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    long long C64 = (long long)C;
    int ct = bid % n_ch_tiles;
    int rank = bid / n_ch_tiles;
    int g_ct = (num_bids - ct + n_ch_tiles - 1) / n_ch_tiles;
    int sub = tid / ct_vec;
    int v = tid - sub * ct_vec;
    int c0 = (ct * ct_vec + v) * 8;
    int streams = g_ct * n_sub;
    int L = (S + streams - 1) / streams;
    int t0 = (rank * n_sub + sub) * L;
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }
    bool grouped = (ct_vec & 31) == 0 && ct_vec > 32 && ct_vec * 8 == C;
    bool rowgrp = (ct_vec & 31) != 0 && ct_vec > 32 && ct_vec * 8 == C && n_sub > 1;
    bool ring_shared = grouped || rowgrp;
    if (grouped) {
        asm volatile("trap;" ::: "memory");
    } else if (rowgrp) {
        int RS = ct_vec * 16;
        int warp_1 = tid >> 5;
        long long c64 = (long long)c0;
        int t_end = ((t0 + L) < (S) ? (t0 + L) : (S));
        int t_stop = ((t_end + 3) < (S + 1) ? (t_end + 3) : (S + 1));
        int t_lim = ((t_stop) < (S) ? (t_stop) : (S));
        int active = 0;
        if (sub < n_sub && c0 < C && t0 < S) {
            active = 1;
        }
        int my_rows = 0;
        int my_steps = 0;
        if (active != 0) {
            my_rows = t_lim - t0;
            my_steps = t_stop - t0;
        }
        int t00 = rank * n_sub * L;
        int steps0 = 0;
        if (t00 < S) {
            int _min_1 = ((t00 + L) < (S) ? (t00 + L) : (S));
            int _min_0 = ((_min_1 + 3) < (S + 1) ? (_min_1 + 3) : (S + 1));
            steps0 = _min_0 - t00;
        }
        int n_stages = (steps0 + 8 - 1) / 8;
        int s_lo = ((warp_1 * 32 / ct_vec) < (n_sub - 1) ? (warp_1 * 32 / ct_vec) : (n_sub - 1));
        int s_hi = (((warp_1 * 32 + 31) / ct_vec) < (n_sub - 1) ? ((warp_1 * 32 + 31) / ct_vec) : (n_sub - 1));
        int _min_r = ((n_stages) < (1) ? (n_stages) : (1));
        #pragma unroll 1
        for (int k0 = 0; k0 < _min_r; k0++) {
            int slot0 = k0;
            int rows_k = 0;
            if (active != 0) {
                int _max_3 = ((my_rows - k0 * 8) > (0) ? (my_rows - k0 * 8) : (0));
                int _min_2 = ((_max_3) < (8) ? (_max_3) : (8));
                rows_k = _min_2;
            }
            if (sub < n_sub) {
                if (v == 0 && rows_k > 0) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (sub + 12 * (int)((unsigned int)k0 % 2)) * 8, rows_k * (2 * RS));
                    int sx = (int)stage_buf_addr + (sub * 2 + slot0) * (8 * RS);
                    long long src0 = (long long)(t0 + k0 * 8) * C64;
                    cp_async_bulk_gmem2smem(sx, reinterpret_cast<const uint8_t*>(x + src0), rows_k * RS, stage_full_addr + (sub + 12 * (int)((unsigned int)k0 % 2)) * 8);
                    cp_async_bulk_gmem2smem(sx + 98304, reinterpret_cast<const uint8_t*>(dy + src0), rows_k * RS, stage_full_addr + (sub + 12 * (int)((unsigned int)k0 % 2)) * 8);
                } else if (v < 32) {
                    mbarrier_arrive(stage_full_addr + (sub + 12 * (int)((unsigned int)k0 % 2)) * 8);
                }
            }
        }
        float wr[34];
        #if !(defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030))
        wr[32] = __uint_as_float((unsigned int)S >> 31 | 2147483648u);
        wr[33] = wr[32];
        #endif
        float win[32];
        float pend[32];
        #pragma unroll
        for (int i_1 = 0; i_1 < 32; i_1++) {
            win[i_1] = 0.0f;
            pend[i_1] = 0.0f;
        }
        int lo = 0;
        int hi = num_docs - 1;
        int doc_start = 0;
        int doc_end = 0;
        if (active != 0) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(w + (c64 + (long long)j) * 4ll);
                    wr[j * 4 + 0] = _v4.x;
                    wr[j * 4 + 1] = _v4.y;
                    wr[j * 4 + 2] = _v4.z;
                    wr[j * 4 + 3] = _v4.w;
                }
            }
            #pragma unroll 1
            for (int step = 0; step < search_steps; step++) {
                int mid = lo + hi + 1 >> 1;
                if (t0 >= cu[mid]) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            doc_start = cu[lo];
            doc_end = cu[lo + 1];
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                int tk = t0 - 3 + kh;
                if (tk >= doc_start) {
                    long long rowk = (long long)tk * C64 + c64;
                    {
                        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + rowk);
                        uint4 _vld_5[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_5[_blk] = _vptr_5[_blk];
                            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
        }
        int _min_s = ((n_stages) < (2) ? (n_stages) : (2));
        #pragma unroll 1
        for (int k1 = 1; k1 < _min_s; k1++) {
            int slot1 = k1;
            int rows_k_1 = 0;
            if (active != 0) {
                int _max_7 = ((my_rows - k1 * 8) > (0) ? (my_rows - k1 * 8) : (0));
                int _min_6 = ((_max_7) < (8) ? (_max_7) : (8));
                rows_k_1 = _min_6;
            }
            if (sub < n_sub) {
                if (v == 0 && rows_k_1 > 0) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (sub + 12 * (int)((unsigned int)k1 % 2)) * 8, rows_k_1 * (2 * RS));
                    int sx_1 = (int)stage_buf_addr + (sub * 2 + slot1) * (8 * RS);
                    long long src0_1 = (long long)(t0 + k1 * 8) * C64;
                    cp_async_bulk_gmem2smem(sx_1, reinterpret_cast<const uint8_t*>(x + src0_1), rows_k_1 * RS, stage_full_addr + (sub + 12 * (int)((unsigned int)k1 % 2)) * 8);
                    cp_async_bulk_gmem2smem(sx_1 + 98304, reinterpret_cast<const uint8_t*>(dy + src0_1), rows_k_1 * RS, stage_full_addr + (sub + 12 * (int)((unsigned int)k1 % 2)) * 8);
                } else if (v < 32) {
                    mbarrier_arrive(stage_full_addr + (sub + 12 * (int)((unsigned int)k1 % 2)) * 8);
                }
            }
        }
        unsigned int pkx[4];
        unsigned int pkd[4];
        float dyf[8];
        float dp[8];
        #pragma unroll 1
        for (int k = 0; k < n_stages; k++) {
            int slot = k % 2;
            int _stage_stage_full = s_lo + 12 * (int)((unsigned int)k % 2);
            int _parity_stage_full = (int)((unsigned int)k / 2 % 2);
            mbarrier_wait(stage_full_addr + (_stage_stage_full) * 8, _parity_stage_full);
            if (s_hi != s_lo) {
                int _stage_stage_full_1 = s_hi + 12 * (int)((unsigned int)k % 2);
                int _parity_stage_full_1 = (int)((unsigned int)k / 2 % 2);
                mbarrier_wait(stage_full_addr + (_stage_stage_full_1) * 8, _parity_stage_full_1);
            }
            if (active != 0) {
                #pragma unroll 1
                for (int hh = 0; hh < 2; hh++) {
                    int q = k * 2 + hh;
                    int tb = t0 + q * 4;
                    int _max_8 = ((my_rows - q * 4) > (0) ? (my_rows - q * 4) : (0));
                    int rows_k_2 = ((_max_8) < (4) ? (_max_8) : (4));
                    int _max_9 = ((my_steps - q * 4) > (0) ? (my_steps - q * 4) : (0));
                    int steps_k = ((_max_9) < (4) ? (_max_9) : (4));
                    int sx0 = (int)stage_buf_addr + (sub * 2 + slot) * (8 * RS) + v * 16 + hh * 4 * RS;
                    int interior = 0;
                    if (steps_k == 4 && rows_k_2 == 4 && t_end >= tb + 4 && doc_end >= tb + 4) {
                        if (q >= 1) {
                            if (doc_start <= tb - 3) {
                                interior = 2;
                            }
                        }
                    }
                    if (interior == 2) {
                        #pragma unroll
                        for (int u = 0; u < 4; u++) {
                            __int128_t _smem_raw_10;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(sx0 + u * RS));
                            pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                            pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                            pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                            pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                            __int128_t _smem_raw_11;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(sx0 + 98304 + u * RS));
                            pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                            pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                            pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                            pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
                            #pragma unroll
                            for (int kk = 0; kk < 4; kk++) {
                                win[u % 4 * 8 + 2 * kk] = __uint_as_float(pkx[kk] << 16);
                                win[u % 4 * 8 + 2 * kk + 1] = __uint_as_float(pkx[kk] & 4294901760u);
                                dyf[2 * kk] = __uint_as_float(pkd[kk] << 16);
                                dyf[2 * kk + 1] = __uint_as_float(pkd[kk] & 4294901760u);
                            }
                            #pragma unroll
                            for (int pv_h = 0; pv_h < 4; pv_h++) {
                                float2 pv_pre2 = add_f32x2_noftz(add_f32x2_noftz(add_f32x2_noftz(CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h], wr[8 * pv_h + 4]), make_float2(win[(u % 4 + 1) % 4 * 8 + 2 * pv_h], win[(u % 4 + 1) % 4 * 8 + 2 * pv_h + 1]), make_float2(wr[32], wr[33])), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h + 1], wr[8 * pv_h + 4 + 1]), make_float2(win[(u % 4 + 2) % 4 * 8 + 2 * pv_h], win[(u % 4 + 2) % 4 * 8 + 2 * pv_h + 1]), make_float2(wr[32], wr[33]))), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h + 2], wr[8 * pv_h + 4 + 2]), make_float2(win[(u % 4 + 3) % 4 * 8 + 2 * pv_h], win[(u % 4 + 3) % 4 * 8 + 2 * pv_h + 1]), make_float2(wr[32], wr[33]))), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h + 3], wr[8 * pv_h + 4 + 3]), make_float2(win[u % 4 * 8 + 2 * pv_h], win[u % 4 * 8 + 2 * pv_h + 1]), make_float2(wr[32], wr[33])));
                                float2 pv_p2 = make_float2(__uint_as_float(__as_u32(__float22bfloat162_rn(make_float2(0.0f, pv_pre2.x)))), __uint_as_float(__as_u32(__float22bfloat162_rn(make_float2(0.0f, pv_pre2.y)))));
                                float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
                                float2 pv_e2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2) : "l"(*(const unsigned long long*)&pv_p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
                                float _exp2_13 = approx_exp2(pv_e2.x);
                                float _exp2_14 = approx_exp2(pv_e2.y);
                                float2 pv_den2 = add_f32x2_noftz(make_float2(_exp2_13, _exp2_14), make_float2(1.0f, 1.0f));
                                float _rcp_15 = approx_rcp(pv_den2.x);
                                float _rcp_16 = approx_rcp(pv_den2.y);
                                float2 pv_sg2 = make_float2(_rcp_15, _rcp_16);
                                float2 _f32x2_mul_b_17 = fma_f32x2_rn_noftz(pv_p2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2), make_float2(1.0f, 1.0f));
                                float2 pv_g2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2) : "l"(*(const unsigned long long*)&pv_sg2), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030)
                                float2 _f32x2_mul_a_pv_d2 = make_float2(dyf[2 * pv_h], dyf[2 * pv_h + 1]);
                                float2 pv_d2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_d2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_pv_d2), "l"(*(const unsigned long long*)&pv_g2));
                                #else
                                float2 pv_d2 = CAKE_F32X2_MUL_NZ(make_float2(dyf[2 * pv_h], dyf[2 * pv_h + 1]), pv_g2, make_float2(wr[32], wr[33]));
                                #endif
                                dp[2 * pv_h] = pv_d2.x;
                                dp[2 * pv_h + 1] = pv_d2.y;
                                #pragma unroll
                                for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                                    float2 pv_a2 = fma_f32x2_rn_noftz(pv_d2, make_float2(win[(u % 4 + 1 + kk_1) % 4 * 8 + 2 * pv_h], win[(u % 4 + 1 + kk_1) % 4 * 8 + 2 * pv_h + 1]), make_float2(acc[8 * pv_h + kk_1], acc[8 * pv_h + 4 + kk_1]));
                                    acc[8 * pv_h + kk_1] = pv_a2.x;
                                    acc[8 * pv_h + 4 + kk_1] = pv_a2.y;
                                }
                                #pragma unroll
                                for (int pv_k = 0; pv_k < 3; pv_k++) {
                                    float2 pv_q2 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + pv_k], wr[8 * pv_h + 4 + pv_k]), pv_d2, make_float2(pend[(u % 4 + 1 + pv_k) % 4 * 8 + 2 * pv_h], pend[(u % 4 + 1 + pv_k) % 4 * 8 + 2 * pv_h + 1]));
                                    pend[(u % 4 + 1 + pv_k) % 4 * 8 + 2 * pv_h] = pv_q2.x;
                                    pend[(u % 4 + 1 + pv_k) % 4 * 8 + 2 * pv_h + 1] = pv_q2.y;
                                }
                                float2 _f32x2_mul_a_18 = make_float2(wr[8 * pv_h + 3], wr[8 * pv_h + 4 + 3]);
                                float2 pv_r2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&pv_d2));
                                pend[u % 4 * 8 + 2 * pv_h] = pv_r2.x;
                                pend[u % 4 * 8 + 2 * pv_h + 1] = pv_r2.y;
                            }
                            if (u >= 0) {
                                long long row3 = (long long)(tb + (u - 3)) * C64 + c64;
                                unsigned int packed[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend + (u % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                    packed[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend + (u % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                    packed[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend + (u % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                    packed[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend + (u % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                    packed[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(dx + (row3))[0] = reinterpret_cast<int4*>(packed)[0];
                            }
                        }
                    } else if (steps_k > 0) {
                        #pragma unroll
                        for (int u_1 = 0; u_1 < 4; u_1++) {
                            if (steps_k > u_1) {
                                int tcur = tb + u_1;
                                if (rows_k_2 > u_1) {
                                    __int128_t _smem_raw_19;
                                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_19) : "r"(sx0 + u_1 * RS));
                                    pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[0];
                                    pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[1];
                                    pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[2];
                                    pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[3];
                                    __int128_t _smem_raw_20;
                                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_20) : "r"(sx0 + 98304 + u_1 * RS));
                                    pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[0];
                                    pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[1];
                                    pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[2];
                                    pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[3];
                                    #pragma unroll
                                    for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                                        win[u_1 % 4 * 8 + 2 * kk_2] = __uint_as_float(pkx[kk_2] << 16);
                                        win[u_1 % 4 * 8 + 2 * kk_2 + 1] = __uint_as_float(pkx[kk_2] & 4294901760u);
                                        dyf[2 * kk_2] = __uint_as_float(pkd[kk_2] << 16);
                                        dyf[2 * kk_2 + 1] = __uint_as_float(pkd[kk_2] & 4294901760u);
                                    }
                                }
                                if (tcur >= doc_end) {
                                    #pragma unroll
                                    for (int m = 1; m < 4; m++) {
                                        int tm = tcur - m;
                                        if (tm >= t0 && tm < t_end && tm >= doc_start) {
                                            long long rowm = (long long)tm * C64 + c64;
                                            unsigned int _packed_r[4];
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2 + 0], (pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 0]));
                                                _packed_r[_lp] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2 + 2], (pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 2]));
                                                _packed_r[_lp + 1] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2 + 4], (pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 4]));
                                                _packed_r[_lp + 2] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2 + 6], (pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 6]));
                                                _packed_r[_lp + 3] = *(uint32_t*)&_bf2;
                                            }
                                            reinterpret_cast<int4*>(dx + (rowm))[0] = reinterpret_cast<int4*>(_packed_r)[0];
                                        }
                                    }
                                    if (tcur < S) {
                                        int lo2 = 0;
                                        int hi2 = num_docs - 1;
                                        #pragma unroll 1
                                        for (int step2 = 0; step2 < search_steps; step2++) {
                                            int mid2 = lo2 + hi2 + 1 >> 1;
                                            if (tcur >= cu[mid2]) {
                                                lo2 = mid2;
                                            } else {
                                                hi2 = mid2 - 1;
                                            }
                                        }
                                        doc_start = cu[lo2];
                                        doc_end = cu[lo2 + 1];
                                    } else {
                                        doc_start = tcur + 4;
                                        doc_end = tcur + 4;
                                    }
                                    #pragma unroll
                                    for (int m_1 = 1; m_1 < 4; m_1++) {
                                        #pragma unroll
                                        for (int j_1 = 0; j_1 < 8; j_1++) {
                                            win[(u_1 % 4 + 4 - m_1) % 4 * 8 + j_1] = 0.0f;
                                        }
                                    }
                                }
                                if (tcur < S) {
                                    if (tcur < t_end) {
                                        #pragma unroll
                                        for (int pv_h_1 = 0; pv_h_1 < 4; pv_h_1++) {
                                            float2 pv_pre2_1 = add_f32x2_noftz(add_f32x2_noftz(add_f32x2_noftz(CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h_1], wr[8 * pv_h_1 + 4]), make_float2(win[(u_1 % 4 + 1) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 1) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(wr[32], wr[33])), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h_1 + 1], wr[8 * pv_h_1 + 4 + 1]), make_float2(win[(u_1 % 4 + 2) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 2) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(wr[32], wr[33]))), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h_1 + 2], wr[8 * pv_h_1 + 4 + 2]), make_float2(win[(u_1 % 4 + 3) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 3) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(wr[32], wr[33]))), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h_1 + 3], wr[8 * pv_h_1 + 4 + 3]), make_float2(win[u_1 % 4 * 8 + 2 * pv_h_1], win[u_1 % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(wr[32], wr[33])));
                                            float2 pv_p2_1 = make_float2(__uint_as_float(__as_u32(__float22bfloat162_rn(make_float2(0.0f, pv_pre2_1.x)))), __uint_as_float(__as_u32(__float22bfloat162_rn(make_float2(0.0f, pv_pre2_1.y)))));
                                            float2 _f32x2_mul_b_21 = make_float2(-1.442695f, -1.442695f);
                                            float2 pv_e2_1;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_1) : "l"(*(const unsigned long long*)&pv_p2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
                                            float _exp2_22 = approx_exp2(pv_e2_1.x);
                                            float _exp2_23 = approx_exp2(pv_e2_1.y);
                                            float2 pv_den2_1 = add_f32x2_noftz(make_float2(_exp2_22, _exp2_23), make_float2(1.0f, 1.0f));
                                            float _rcp_24 = approx_rcp(pv_den2_1.x);
                                            float _rcp_25 = approx_rcp(pv_den2_1.y);
                                            float2 pv_sg2_1 = make_float2(_rcp_24, _rcp_25);
                                            float2 _f32x2_mul_b_26 = fma_f32x2_rn_noftz(pv_p2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_1), make_float2(1.0f, 1.0f));
                                            float2 pv_g2_1;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_1) : "l"(*(const unsigned long long*)&pv_sg2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
                                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030)
                                            float2 _f32x2_mul_a_pv_d2_1 = make_float2(dyf[2 * pv_h_1], dyf[2 * pv_h_1 + 1]);
                                            float2 pv_d2_1;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_d2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_pv_d2_1), "l"(*(const unsigned long long*)&pv_g2_1));
                                            #else
                                            float2 pv_d2_1 = CAKE_F32X2_MUL_NZ(make_float2(dyf[2 * pv_h_1], dyf[2 * pv_h_1 + 1]), pv_g2_1, make_float2(wr[32], wr[33]));
                                            #endif
                                            dp[2 * pv_h_1] = pv_d2_1.x;
                                            dp[2 * pv_h_1 + 1] = pv_d2_1.y;
                                            #pragma unroll
                                            for (int kk_3 = 0; kk_3 < 4; kk_3++) {
                                                float2 pv_a2_1 = fma_f32x2_rn_noftz(pv_d2_1, make_float2(win[(u_1 % 4 + 1 + kk_3) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 1 + kk_3) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(acc[8 * pv_h_1 + kk_3], acc[8 * pv_h_1 + 4 + kk_3]));
                                                acc[8 * pv_h_1 + kk_3] = pv_a2_1.x;
                                                acc[8 * pv_h_1 + 4 + kk_3] = pv_a2_1.y;
                                            }
                                            #pragma unroll
                                            for (int pv_k_1 = 0; pv_k_1 < 3; pv_k_1++) {
                                                float2 pv_q2_1 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + pv_k_1], wr[8 * pv_h_1 + 4 + pv_k_1]), pv_d2_1, make_float2(pend[(u_1 % 4 + 1 + pv_k_1) % 4 * 8 + 2 * pv_h_1], pend[(u_1 % 4 + 1 + pv_k_1) % 4 * 8 + 2 * pv_h_1 + 1]));
                                                pend[(u_1 % 4 + 1 + pv_k_1) % 4 * 8 + 2 * pv_h_1] = pv_q2_1.x;
                                                pend[(u_1 % 4 + 1 + pv_k_1) % 4 * 8 + 2 * pv_h_1 + 1] = pv_q2_1.y;
                                            }
                                            float2 _f32x2_mul_a_27 = make_float2(wr[8 * pv_h_1 + 3], wr[8 * pv_h_1 + 4 + 3]);
                                            float2 pv_r2_1;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&pv_d2_1));
                                            pend[u_1 % 4 * 8 + 2 * pv_h_1] = pv_r2_1.x;
                                            pend[u_1 % 4 * 8 + 2 * pv_h_1 + 1] = pv_r2_1.y;
                                        }
                                    } else {
                                        #pragma unroll
                                        for (int pv_h_2 = 0; pv_h_2 < 4; pv_h_2++) {
                                            float2 pv_pre2_2 = add_f32x2_noftz(add_f32x2_noftz(add_f32x2_noftz(CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h_2], wr[8 * pv_h_2 + 4]), make_float2(win[(u_1 % 4 + 1) % 4 * 8 + 2 * pv_h_2], win[(u_1 % 4 + 1) % 4 * 8 + 2 * pv_h_2 + 1]), make_float2(wr[32], wr[33])), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h_2 + 1], wr[8 * pv_h_2 + 4 + 1]), make_float2(win[(u_1 % 4 + 2) % 4 * 8 + 2 * pv_h_2], win[(u_1 % 4 + 2) % 4 * 8 + 2 * pv_h_2 + 1]), make_float2(wr[32], wr[33]))), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h_2 + 2], wr[8 * pv_h_2 + 4 + 2]), make_float2(win[(u_1 % 4 + 3) % 4 * 8 + 2 * pv_h_2], win[(u_1 % 4 + 3) % 4 * 8 + 2 * pv_h_2 + 1]), make_float2(wr[32], wr[33]))), CAKE_F32X2_MUL_NZ(make_float2(wr[8 * pv_h_2 + 3], wr[8 * pv_h_2 + 4 + 3]), make_float2(win[u_1 % 4 * 8 + 2 * pv_h_2], win[u_1 % 4 * 8 + 2 * pv_h_2 + 1]), make_float2(wr[32], wr[33])));
                                            float2 pv_p2_2 = make_float2(__uint_as_float(__as_u32(__float22bfloat162_rn(make_float2(0.0f, pv_pre2_2.x)))), __uint_as_float(__as_u32(__float22bfloat162_rn(make_float2(0.0f, pv_pre2_2.y)))));
                                            float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
                                            float2 pv_e2_2;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_2) : "l"(*(const unsigned long long*)&pv_p2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
                                            float _exp2_29 = approx_exp2(pv_e2_2.x);
                                            float _exp2_30 = approx_exp2(pv_e2_2.y);
                                            float2 pv_den2_2 = add_f32x2_noftz(make_float2(_exp2_29, _exp2_30), make_float2(1.0f, 1.0f));
                                            float _rcp_31 = approx_rcp(pv_den2_2.x);
                                            float _rcp_32 = approx_rcp(pv_den2_2.y);
                                            float2 pv_sg2_2 = make_float2(_rcp_31, _rcp_32);
                                            float2 _f32x2_mul_b_33 = fma_f32x2_rn_noftz(pv_p2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_2), make_float2(1.0f, 1.0f));
                                            float2 pv_g2_2;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_2) : "l"(*(const unsigned long long*)&pv_sg2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
                                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030)
                                            float2 _f32x2_mul_a_pv_d2_2 = make_float2(dyf[2 * pv_h_2], dyf[2 * pv_h_2 + 1]);
                                            float2 pv_d2_2;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_d2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_pv_d2_2), "l"(*(const unsigned long long*)&pv_g2_2));
                                            #else
                                            float2 pv_d2_2 = CAKE_F32X2_MUL_NZ(make_float2(dyf[2 * pv_h_2], dyf[2 * pv_h_2 + 1]), pv_g2_2, make_float2(wr[32], wr[33]));
                                            #endif
                                            dp[2 * pv_h_2] = pv_d2_2.x;
                                            dp[2 * pv_h_2 + 1] = pv_d2_2.y;
                                            #pragma unroll
                                            for (int pv_k_2 = 0; pv_k_2 < 3; pv_k_2++) {
                                                float2 pv_q2_2 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + pv_k_2], wr[8 * pv_h_2 + 4 + pv_k_2]), pv_d2_2, make_float2(pend[(u_1 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2], pend[(u_1 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2 + 1]));
                                                pend[(u_1 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2] = pv_q2_2.x;
                                                pend[(u_1 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2 + 1] = pv_q2_2.y;
                                            }
                                            float2 _f32x2_mul_a_34 = make_float2(wr[8 * pv_h_2 + 3], wr[8 * pv_h_2 + 4 + 3]);
                                            float2 pv_r2_2;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&pv_d2_2));
                                            pend[u_1 % 4 * 8 + 2 * pv_h_2] = pv_r2_2.x;
                                            pend[u_1 % 4 * 8 + 2 * pv_h_2 + 1] = pv_r2_2.y;
                                        }
                                    }
                                    int t3 = tcur - 3;
                                    if (t3 >= t0 && t3 >= doc_start) {
                                        long long row3_1 = (long long)t3 * C64 + c64;
                                        unsigned int _packed_s[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                            _packed_s[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                            _packed_s[_lp + 1] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                            _packed_s[_lp + 2] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                            _packed_s[_lp + 3] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(dx + (row3_1))[0] = reinterpret_cast<int4*>(_packed_s)[0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn = k + 2;
            if (kn < n_stages) {
                int rows_k_3 = 0;
                if (active != 0) {
                    int _max_36 = ((my_rows - (k + 2) * 8) > (0) ? (my_rows - (k + 2) * 8) : (0));
                    int _min_35 = ((_max_36) < (8) ? (_max_36) : (8));
                    rows_k_3 = _min_35;
                }
                if (sub < n_sub) {
                    if (v == 0 && rows_k_3 > 0) {
                        mbarrier_arrive_expect_tx(stage_full_addr + (sub + 12 * (int)((unsigned int)(k + 2) % 2)) * 8, rows_k_3 * (2 * RS));
                        int sx_2 = (int)stage_buf_addr + (sub * 2 + slot) * (8 * RS);
                        long long src0_2 = (long long)(t0 + (k + 2) * 8) * C64;
                        cp_async_bulk_gmem2smem(sx_2, reinterpret_cast<const uint8_t*>(x + src0_2), rows_k_3 * RS, stage_full_addr + (sub + 12 * (int)((unsigned int)(k + 2) % 2)) * 8);
                        cp_async_bulk_gmem2smem(sx_2 + 98304, reinterpret_cast<const uint8_t*>(dy + src0_2), rows_k_3 * RS, stage_full_addr + (sub + 12 * (int)((unsigned int)(k + 2) % 2)) * 8);
                    } else if (v < 32) {
                        mbarrier_arrive(stage_full_addr + (sub + 12 * (int)((unsigned int)(k + 2) % 2)) * 8);
                    }
                }
            }
        }
    } else {
        asm volatile("trap;" ::: "memory");
    }
    if (ring_shared) {
        __syncthreads();
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 32; i_2++) {
        red[(tid >> 5) * 4096 + i_2 * 32 + (tid & 31)] = acc[i_2];
    }
    __syncthreads();
    int c0_0 = (ct * ct_vec + v) * 8;
    if (sub == 0 && c0_0 < C) {
        float tot[32];
        #pragma unroll
        for (int i_3 = 0; i_3 < 32; i_3++) {
            tot[i_3] = red[(v >> 5) * 4096 + i_3 * 32 + (v & 31)];
        }
        #pragma unroll 1
        for (int s = 1; s < n_sub; s++) {
            int ts = s * ct_vec + v;
            #pragma unroll
            for (int i_4 = 0; i_4 < 32; i_4++) {
                tot[i_4] = tot[i_4] + red[(ts >> 5) * 4096 + i_4 * 32 + (ts & 31)];
            }
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            long long pidx = ((long long)rank * C64 + (long long)c0_0 + (long long)j_2) * 4ll;
            {
                float4 _v4 = make_float4(tot[j_2 * 4 + 0], tot[j_2 * 4 + 1], tot[j_2 * 4 + 2], tot[j_2 * 4 + 3]);
                *reinterpret_cast<float4*>(partial + pidx) = _v4;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Cleanup
    __syncthreads();
}

} // extern "C"
