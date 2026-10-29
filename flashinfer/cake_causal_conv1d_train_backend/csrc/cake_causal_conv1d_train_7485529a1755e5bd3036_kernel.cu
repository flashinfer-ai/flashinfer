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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_RED_OFF 1024
#define SMEM_RED_STAGE_BYTES 16384
#define SMEM_RED_STRIDE 16384
#define SMEM_STAGE_BUF_OFF 17408
#define SMEM_STAGE_BUF_STAGE_BYTES 65536
#define SMEM_STAGE_BUF_STRIDE 65536
#define SMEM_TOTAL 82944
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>
#include <cooperative_groups.h>

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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_train_7485529a1755e5bd3036(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, __nv_bfloat16* __restrict__ hist, int* __restrict__ hist_row_off, float* __restrict__ dpre_tail, int* __restrict__ tail_row_off, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap dy_map)
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

    // Kernel setup ops
    float* red = reinterpret_cast<float*>(smem_raw + 1024);
    const int red_addr = smem + 1024;
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int stage_buf_addr = smem + 17408;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stage_full: 8 barriers, init_count=32
            mbarrier_init(smem + 0, 32);
            mbarrier_init(smem + 8, 32);
            mbarrier_init(smem + 16, 32);
            mbarrier_init(smem + 24, 32);
            mbarrier_init(smem + 32, 32);
            mbarrier_init(smem + 40, 32);
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
    int grouped = 0;
    if ((ct_vec & 31) == 0 && ct_vec > 32 && ct_vec * 8 == C) {
        grouped = 1;
    }
    if (grouped != 0) {
        int RS = ct_vec * 16;
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
        int n_stages = (my_steps + 8 - 1) / 8;
        int _min_0 = ((n_stages) < (2) ? (n_stages) : (2));
        #pragma unroll 1
        for (int k0 = 0; k0 < _min_0; k0++) {
            int slot0 = k0;
            int GWn = ct_vec >> 5;
            int bslot = sub * 2 + slot0;
            int rows_k = 0;
            if (active != 0) {
                int _max_1 = ((my_rows - k0 * 8) > (0) ? (my_rows - k0 * 8) : (0));
                int _min_0_1 = ((_max_1) < (8) ? (_max_1) : (8));
                rows_k = _min_0_1;
            }
            if (v == 0 && rows_k > 0) {
                mbarrier_arrive_expect_tx(stage_full_addr + (bslot) * 8, rows_k * (2 * RS));
                int sx = (int)stage_buf_addr + (sub * (GWn * 2) + slot0 * GWn) * 4096;
                long long src0 = (long long)(t0 + k0 * 8) * C64 + c64;
                cp_async_bulk_gmem2smem(sx, reinterpret_cast<const uint8_t*>(x + src0), rows_k * RS, stage_full_addr + (bslot) * 8);
                cp_async_bulk_gmem2smem(sx + 32768, reinterpret_cast<const uint8_t*>(dy + src0), rows_k * RS, stage_full_addr + (bslot) * 8);
            } else if (v < 32) {
                mbarrier_arrive(stage_full_addr + (bslot) * 8);
            }
        }
        float wr[32];
        float w4[4];
        float win[32];
        float pend[32];
        float tailf[8];
        #pragma unroll
        for (int i_1 = 0; i_1 < 32; i_1++) {
            win[i_1] = 0.0f;
            pend[i_1] = 0.0f;
        }
        int lo = 0;
        int hi = num_docs - 1;
        int doc_start = 0;
        int doc_end = 0;
        int frag = 0;
        if (active != 0) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(w + (c64 + (long long)j) * 4);
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
            frag = lo;
            int tk = t0 - 1;
            if (tk >= doc_start) {
                long long rowk = (long long)tk * C64 + c64;
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(x + rowk);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&win[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow = hist_row_off[frag * 3 + 2 - (doc_start - t0)];
                if (hrow >= 0) {
                    long long rowh = (long long)hrow * C64 + c64;
                    {
                        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(hist + rowh);
                        uint4 _vld_4[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_4[_blk] = _vptr_4[_blk];
                            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                (&win[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            int tk_0 = t0 - 2;
            if (tk_0 >= doc_start) {
                long long rowk_1 = (long long)tk_0 * C64 + c64;
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + rowk_1);
                    uint4 _vld_5[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_5[_blk] = _vptr_5[_blk];
                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                            (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_1 = hist_row_off[frag * 3 + 1 - (doc_start - t0)];
                if (hrow_1 >= 0) {
                    long long rowh_1 = (long long)hrow_1 * C64 + c64;
                    {
                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(hist + rowh_1);
                        uint4 _vld_6[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_6[_blk] = _vptr_6[_blk];
                            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            int tk_1 = t0 - 3;
            if (tk_1 >= doc_start) {
                long long rowk_2 = (long long)tk_1 * C64 + c64;
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + rowk_2);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                            (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_2 = hist_row_off[frag * 3 - (doc_start - t0)];
                if (hrow_2 >= 0) {
                    long long rowh_2 = (long long)hrow_2 * C64 + c64;
                    {
                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(hist + rowh_2);
                        uint4 _vld_8[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_8[_blk] = _vptr_8[_blk];
                            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                            }
                        }
                    }
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
            int parity = k / 2 & 1;
            int bslot_1 = sub * 2 + slot;
            mbarrier_wait(stage_full_addr + (bslot_1) * 8, parity);
            if (active != 0) {
                int tb = t0 + k * 8;
                int _max_9 = ((my_rows - k * 8) > (0) ? (my_rows - k * 8) : (0));
                int rows_k_1 = ((_max_9) < (8) ? (_max_9) : (8));
                int _max_10 = ((my_steps - k * 8) > (0) ? (my_steps - k * 8) : (0));
                int steps_k = ((_max_10) < (8) ? (_max_10) : (8));
                int GWn_1 = ct_vec >> 5;
                int sx0 = (int)stage_buf_addr + (sub * (GWn_1 * 2) + slot * GWn_1) * 4096 + v * 16;
                int interior = 0;
                if (steps_k == 8 && rows_k_1 == 8 && t_end >= tb + 8 && doc_end >= tb + 8) {
                    if (k >= 1) {
                        if (doc_start <= tb - 3) {
                            interior = 1;
                        }
                    }
                }
                if (interior != 0) {
                    #pragma unroll
                    for (int u = 0; u < 8; u++) {
                        __int128_t _smem_raw_11;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(sx0 + u * RS));
                        pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                        pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                        pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                        pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
                        __int128_t _smem_raw_12;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_12) : "r"(sx0 + 32768 + u * RS));
                        pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[0];
                        pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[1];
                        pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[2];
                        pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[3];
                        #pragma unroll
                        for (int kk = 0; kk < 4; kk++) {
                            win[u % 4 * 8 + 2 * kk] = __uint_as_float(pkx[kk] << 16);
                            win[u % 4 * 8 + 2 * kk + 1] = __uint_as_float(pkx[kk] & 4294901760u);
                            dyf[2 * kk] = __uint_as_float(pkd[kk] << 16);
                            dyf[2 * kk + 1] = __uint_as_float(pkd[kk] & 4294901760u);
                        }
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 8; j_1++) {
                            #pragma unroll
                            for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                                w4[kk_1] = wr[j_1 * 4 + kk_1];
                            }
                            float _fma_14;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(w4[1]), "f"(win[(u % 4 + 2) % 4 * 8 + j_1]), "f"(mul_f32_rn_noftz(w4[0], win[(u % 4 + 1) % 4 * 8 + j_1])));
                            float _fma_13;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(w4[2]), "f"(win[(u % 4 + 3) % 4 * 8 + j_1]), "f"(_fma_14));
                            float _fma_0;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(w4[3]), "f"(win[u % 4 * 8 + j_1]), "f"(_fma_13));
                            float e = (-_fma_0) * 1.4426950408889634f;
                            float _exp2_15 = approx_exp2(e);
                            float den = 1.0f + _exp2_15;
                            float sg = approx_rcp(den);
                            float _fma_16;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(_fma_0), "f"(1.0f - sg), "f"(1.0f));
                            dp[j_1] = dyf[j_1] * (sg * _fma_16);
                            #pragma unroll
                            for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                                acc[j_1 * 4 + kk_2] = __fmaf_rn(dp[j_1], win[(u % 4 + 1 + kk_2) % 4 * 8 + j_1], acc[j_1 * 4 + kk_2]);
                            }
                            float _fma_17;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(w4[0]), "f"(dp[j_1]), "f"(pend[(u % 4 + 1) % 4 * 8 + j_1]));
                            pend[(u % 4 + 1) % 4 * 8 + j_1] = _fma_17;
                            float _fma_18;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(w4[1]), "f"(dp[j_1]), "f"(pend[(u % 4 + 2) % 4 * 8 + j_1]));
                            pend[(u % 4 + 2) % 4 * 8 + j_1] = _fma_18;
                            float _fma_19;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(w4[2]), "f"(dp[j_1]), "f"(pend[(u % 4 + 3) % 4 * 8 + j_1]));
                            pend[(u % 4 + 3) % 4 * 8 + j_1] = _fma_19;
                            pend[u % 4 * 8 + j_1] = w4[3] * dp[j_1];
                        }
                        long long row3 = (long long)(tb + (u - 3)) * C64 + c64;
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(pend[(u % 4 + 1) % 4 * 8 + 0], pend[(u % 4 + 1) % 4 * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(pend[(u % 4 + 1) % 4 * 8 + 2], pend[(u % 4 + 1) % 4 * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(pend[(u % 4 + 1) % 4 * 8 + 4], pend[(u % 4 + 1) % 4 * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(pend[(u % 4 + 1) % 4 * 8 + 6], pend[(u % 4 + 1) % 4 * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[row3 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int u_1 = 0; u_1 < 8; u_1++) {
                        if (steps_k > u_1) {
                            int tcur = tb + u_1;
                            if (rows_k_1 > u_1) {
                                __int128_t _smem_raw_20;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_20) : "r"(sx0 + u_1 * RS));
                                pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[0];
                                pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[1];
                                pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[2];
                                pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[3];
                                __int128_t _smem_raw_21;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_21) : "r"(sx0 + 32768 + u_1 * RS));
                                pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[0];
                                pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[1];
                                pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[2];
                                pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[3];
                                #pragma unroll
                                for (int kk_3 = 0; kk_3 < 4; kk_3++) {
                                    win[u_1 % 4 * 8 + 2 * kk_3] = __uint_as_float(pkx[kk_3] << 16);
                                    win[u_1 % 4 * 8 + 2 * kk_3 + 1] = __uint_as_float(pkx[kk_3] & 4294901760u);
                                    dyf[2 * kk_3] = __uint_as_float(pkd[kk_3] << 16);
                                    dyf[2 * kk_3 + 1] = __uint_as_float(pkd[kk_3] & 4294901760u);
                                }
                            }
                            if (tcur >= doc_end) {
                                int trow = tail_row_off[frag * 3];
                                if (trow >= 0) {
                                    long long rowr = (long long)trow * C64 + c64;
                                    {
                                        unsigned _ldv8_22_0;
                                        unsigned _ldv8_22_1;
                                        unsigned _ldv8_22_2;
                                        unsigned _ldv8_22_3;
                                        unsigned _ldv8_22_4;
                                        unsigned _ldv8_22_5;
                                        unsigned _ldv8_22_6;
                                        unsigned _ldv8_22_7;
                                        asm volatile(
                                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_ldv8_22_0), "=r"(_ldv8_22_1), "=r"(_ldv8_22_2), "=r"(_ldv8_22_3), "=r"(_ldv8_22_4), "=r"(_ldv8_22_5), "=r"(_ldv8_22_6), "=r"(_ldv8_22_7) : "l"((const void*)(dpre_tail + (rowr))) : "memory");
                                        tailf[0 + 0] = __uint_as_float(_ldv8_22_0);
                                        tailf[0 + 1] = __uint_as_float(_ldv8_22_1);
                                        tailf[0 + 2] = __uint_as_float(_ldv8_22_2);
                                        tailf[0 + 3] = __uint_as_float(_ldv8_22_3);
                                        tailf[0 + 4] = __uint_as_float(_ldv8_22_4);
                                        tailf[0 + 5] = __uint_as_float(_ldv8_22_5);
                                        tailf[0 + 6] = __uint_as_float(_ldv8_22_6);
                                        tailf[0 + 7] = __uint_as_float(_ldv8_22_7);
                                    }
                                    #pragma unroll
                                    for (int j_2 = 0; j_2 < 8; j_2++) {
                                        #pragma unroll
                                        for (int kk_4 = 0; kk_4 < 4; kk_4++) {
                                            w4[kk_4] = wr[j_2 * 4 + kk_4];
                                        }
                                        float _fma_23;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(w4[2]), "f"(tailf[j_2]), "f"(pend[(u_1 % 4 + 4 - 1) % 4 * 8 + j_2]));
                                        pend[(u_1 % 4 + 4 - 1) % 4 * 8 + j_2] = _fma_23;
                                        float _fma_24;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(w4[1]), "f"(tailf[j_2]), "f"(pend[(u_1 % 4 + 4 - 2) % 4 * 8 + j_2]));
                                        pend[(u_1 % 4 + 4 - 2) % 4 * 8 + j_2] = _fma_24;
                                        float _fma_25;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(w4[0]), "f"(tailf[j_2]), "f"(pend[(u_1 % 4 + 4 - 3) % 4 * 8 + j_2]));
                                        pend[(u_1 % 4 + 4 - 3) % 4 * 8 + j_2] = _fma_25;
                                    }
                                }
                                int trow_0 = tail_row_off[frag * 3 + 1];
                                if (trow_0 >= 0) {
                                    long long rowr_1 = (long long)trow_0 * C64 + c64;
                                    {
                                        unsigned _ldv8_26_0;
                                        unsigned _ldv8_26_1;
                                        unsigned _ldv8_26_2;
                                        unsigned _ldv8_26_3;
                                        unsigned _ldv8_26_4;
                                        unsigned _ldv8_26_5;
                                        unsigned _ldv8_26_6;
                                        unsigned _ldv8_26_7;
                                        asm volatile(
                                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_ldv8_26_0), "=r"(_ldv8_26_1), "=r"(_ldv8_26_2), "=r"(_ldv8_26_3), "=r"(_ldv8_26_4), "=r"(_ldv8_26_5), "=r"(_ldv8_26_6), "=r"(_ldv8_26_7) : "l"((const void*)(dpre_tail + (rowr_1))) : "memory");
                                        tailf[0 + 0] = __uint_as_float(_ldv8_26_0);
                                        tailf[0 + 1] = __uint_as_float(_ldv8_26_1);
                                        tailf[0 + 2] = __uint_as_float(_ldv8_26_2);
                                        tailf[0 + 3] = __uint_as_float(_ldv8_26_3);
                                        tailf[0 + 4] = __uint_as_float(_ldv8_26_4);
                                        tailf[0 + 5] = __uint_as_float(_ldv8_26_5);
                                        tailf[0 + 6] = __uint_as_float(_ldv8_26_6);
                                        tailf[0 + 7] = __uint_as_float(_ldv8_26_7);
                                    }
                                    #pragma unroll
                                    for (int j_3 = 0; j_3 < 8; j_3++) {
                                        #pragma unroll
                                        for (int kk_5 = 0; kk_5 < 4; kk_5++) {
                                            w4[kk_5] = wr[j_3 * 4 + kk_5];
                                        }
                                        float _fma_27;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(w4[1]), "f"(tailf[j_3]), "f"(pend[(u_1 % 4 + 4 + 1 - 2) % 4 * 8 + j_3]));
                                        pend[(u_1 % 4 + 4 + 1 - 2) % 4 * 8 + j_3] = _fma_27;
                                        float _fma_28;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(w4[0]), "f"(tailf[j_3]), "f"(pend[(u_1 % 4 + 4 + 1 - 3) % 4 * 8 + j_3]));
                                        pend[(u_1 % 4 + 4 + 1 - 3) % 4 * 8 + j_3] = _fma_28;
                                    }
                                }
                                int trow_1 = tail_row_off[frag * 3 + 2];
                                if (trow_1 >= 0) {
                                    long long rowr_2 = (long long)trow_1 * C64 + c64;
                                    {
                                        unsigned _ldv8_29_0;
                                        unsigned _ldv8_29_1;
                                        unsigned _ldv8_29_2;
                                        unsigned _ldv8_29_3;
                                        unsigned _ldv8_29_4;
                                        unsigned _ldv8_29_5;
                                        unsigned _ldv8_29_6;
                                        unsigned _ldv8_29_7;
                                        asm volatile(
                                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_ldv8_29_0), "=r"(_ldv8_29_1), "=r"(_ldv8_29_2), "=r"(_ldv8_29_3), "=r"(_ldv8_29_4), "=r"(_ldv8_29_5), "=r"(_ldv8_29_6), "=r"(_ldv8_29_7) : "l"((const void*)(dpre_tail + (rowr_2))) : "memory");
                                        tailf[0 + 0] = __uint_as_float(_ldv8_29_0);
                                        tailf[0 + 1] = __uint_as_float(_ldv8_29_1);
                                        tailf[0 + 2] = __uint_as_float(_ldv8_29_2);
                                        tailf[0 + 3] = __uint_as_float(_ldv8_29_3);
                                        tailf[0 + 4] = __uint_as_float(_ldv8_29_4);
                                        tailf[0 + 5] = __uint_as_float(_ldv8_29_5);
                                        tailf[0 + 6] = __uint_as_float(_ldv8_29_6);
                                        tailf[0 + 7] = __uint_as_float(_ldv8_29_7);
                                    }
                                    #pragma unroll
                                    for (int j_4 = 0; j_4 < 8; j_4++) {
                                        #pragma unroll
                                        for (int kk_6 = 0; kk_6 < 4; kk_6++) {
                                            w4[kk_6] = wr[j_4 * 4 + kk_6];
                                        }
                                        float _fma_30;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(w4[0]), "f"(tailf[j_4]), "f"(pend[(u_1 % 4 + 4 + 2 - 3) % 4 * 8 + j_4]));
                                        pend[(u_1 % 4 + 4 + 2 - 3) % 4 * 8 + j_4] = _fma_30;
                                    }
                                }
                                #pragma unroll
                                for (int m = 1; m < 4; m++) {
                                    int tm = tcur - m;
                                    if (tm >= t0 && tm < t_end && tm >= doc_start) {
                                        long long rowm = (long long)tm * C64 + c64;
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(pend[(u_1 % 4 + 4 - m) % 4 * 8 + 0], pend[(u_1 % 4 + 4 - m) % 4 * 8 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(pend[(u_1 % 4 + 4 - m) % 4 * 8 + 2], pend[(u_1 % 4 + 4 - m) % 4 * 8 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(pend[(u_1 % 4 + 4 - m) % 4 * 8 + 4], pend[(u_1 % 4 + 4 - m) % 4 * 8 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(pend[(u_1 % 4 + 4 - m) % 4 * 8 + 6], pend[(u_1 % 4 + 4 - m) % 4 * 8 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[rowm + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int m_1 = 1; m_1 < 4; m_1++) {
                                    #pragma unroll
                                    for (int j_5 = 0; j_5 < 8; j_5++) {
                                        win[(u_1 % 4 + 4 - m_1) % 4 * 8 + j_5] = 0.0f;
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
                                    frag = lo2;
                                    int hrow_3 = hist_row_off[frag * 3 + 2];
                                    if (hrow_3 >= 0) {
                                        long long rowh_3 = (long long)hrow_3 * C64 + c64;
                                        {
                                            const uint4* _vptr_31 = reinterpret_cast<const uint4*>(hist + rowh_3);
                                            uint4 _vld_31[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_31[_blk] = _vptr_31[_blk];
                                                uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win[(u_1 % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                                                    (&win[(u_1 % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                    int hrow_0 = hist_row_off[frag * 3 + 1];
                                    if (hrow_0 >= 0) {
                                        long long rowh_4 = (long long)hrow_0 * C64 + c64;
                                        {
                                            const uint4* _vptr_32 = reinterpret_cast<const uint4*>(hist + rowh_4);
                                            uint4 _vld_32[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_32[_blk] = _vptr_32[_blk];
                                                uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win[(u_1 % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                                                    (&win[(u_1 % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                    int hrow_1_1 = hist_row_off[frag * 3];
                                    if (hrow_1_1 >= 0) {
                                        long long rowh_5 = (long long)hrow_1_1 * C64 + c64;
                                        {
                                            const uint4* _vptr_33 = reinterpret_cast<const uint4*>(hist + rowh_5);
                                            uint4 _vld_33[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_33[_blk] = _vptr_33[_blk];
                                                uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win[(u_1 % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                                                    (&win[(u_1 % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    doc_start = tcur + 8;
                                    doc_end = tcur + 8;
                                }
                            }
                            if (tcur < S) {
                                if (tcur < t_end) {
                                    #pragma unroll
                                    for (int j_6 = 0; j_6 < 8; j_6++) {
                                        #pragma unroll
                                        for (int kk_7 = 0; kk_7 < 4; kk_7++) {
                                            w4[kk_7] = wr[j_6 * 4 + kk_7];
                                        }
                                        float _fma_35;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(w4[1]), "f"(win[(u_1 % 4 + 2) % 4 * 8 + j_6]), "f"(mul_f32_rn_noftz(w4[0], win[(u_1 % 4 + 1) % 4 * 8 + j_6])));
                                        float _fma_34;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(w4[2]), "f"(win[(u_1 % 4 + 3) % 4 * 8 + j_6]), "f"(_fma_35));
                                        float _fma_1;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(w4[3]), "f"(win[u_1 % 4 * 8 + j_6]), "f"(_fma_34));
                                        float e_1 = (-_fma_1) * 1.4426950408889634f;
                                        float _exp2_36 = approx_exp2(e_1);
                                        float den_1 = 1.0f + _exp2_36;
                                        float sg_1 = approx_rcp(den_1);
                                        float _fma_37;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(_fma_1), "f"(1.0f - sg_1), "f"(1.0f));
                                        dp[j_6] = dyf[j_6] * (sg_1 * _fma_37);
                                        #pragma unroll
                                        for (int kk_8 = 0; kk_8 < 4; kk_8++) {
                                            acc[j_6 * 4 + kk_8] = __fmaf_rn(dp[j_6], win[(u_1 % 4 + 1 + kk_8) % 4 * 8 + j_6], acc[j_6 * 4 + kk_8]);
                                        }
                                        float _fma_38;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(w4[0]), "f"(dp[j_6]), "f"(pend[(u_1 % 4 + 1) % 4 * 8 + j_6]));
                                        pend[(u_1 % 4 + 1) % 4 * 8 + j_6] = _fma_38;
                                        float _fma_39;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(w4[1]), "f"(dp[j_6]), "f"(pend[(u_1 % 4 + 2) % 4 * 8 + j_6]));
                                        pend[(u_1 % 4 + 2) % 4 * 8 + j_6] = _fma_39;
                                        float _fma_40;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(w4[2]), "f"(dp[j_6]), "f"(pend[(u_1 % 4 + 3) % 4 * 8 + j_6]));
                                        pend[(u_1 % 4 + 3) % 4 * 8 + j_6] = _fma_40;
                                        pend[u_1 % 4 * 8 + j_6] = w4[3] * dp[j_6];
                                    }
                                } else {
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 8; j_7++) {
                                        #pragma unroll
                                        for (int kk_9 = 0; kk_9 < 4; kk_9++) {
                                            w4[kk_9] = wr[j_7 * 4 + kk_9];
                                        }
                                        float _fma_42;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(w4[1]), "f"(win[(u_1 % 4 + 2) % 4 * 8 + j_7]), "f"(mul_f32_rn_noftz(w4[0], win[(u_1 % 4 + 1) % 4 * 8 + j_7])));
                                        float _fma_41;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(w4[2]), "f"(win[(u_1 % 4 + 3) % 4 * 8 + j_7]), "f"(_fma_42));
                                        float _fma_2;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(w4[3]), "f"(win[u_1 % 4 * 8 + j_7]), "f"(_fma_41));
                                        float e_2 = (-_fma_2) * 1.4426950408889634f;
                                        float _exp2_43 = approx_exp2(e_2);
                                        float den_2 = 1.0f + _exp2_43;
                                        float sg_2 = approx_rcp(den_2);
                                        float _fma_44;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(_fma_2), "f"(1.0f - sg_2), "f"(1.0f));
                                        dp[j_7] = dyf[j_7] * (sg_2 * _fma_44);
                                        float _fma_45;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(w4[0]), "f"(dp[j_7]), "f"(pend[(u_1 % 4 + 1) % 4 * 8 + j_7]));
                                        pend[(u_1 % 4 + 1) % 4 * 8 + j_7] = _fma_45;
                                        float _fma_46;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(w4[1]), "f"(dp[j_7]), "f"(pend[(u_1 % 4 + 2) % 4 * 8 + j_7]));
                                        pend[(u_1 % 4 + 2) % 4 * 8 + j_7] = _fma_46;
                                        float _fma_47;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(w4[2]), "f"(dp[j_7]), "f"(pend[(u_1 % 4 + 3) % 4 * 8 + j_7]));
                                        pend[(u_1 % 4 + 3) % 4 * 8 + j_7] = _fma_47;
                                        pend[u_1 % 4 * 8 + j_7] = w4[3] * dp[j_7];
                                    }
                                }
                                int t3 = tcur - 3;
                                if (t3 >= t0 && t3 >= doc_start) {
                                    long long row3_1 = (long long)t3 * C64 + c64;
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(pend[(u_1 % 4 + 1) % 4 * 8 + 0], pend[(u_1 % 4 + 1) % 4 * 8 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(pend[(u_1 % 4 + 1) % 4 * 8 + 2], pend[(u_1 % 4 + 1) % 4 * 8 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(pend[(u_1 % 4 + 1) % 4 * 8 + 4], pend[(u_1 % 4 + 1) % 4 * 8 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(pend[(u_1 % 4 + 1) % 4 * 8 + 6], pend[(u_1 % 4 + 1) % 4 * 8 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[row3_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int s = 0; s < 2; s++) {
                if (sub == s) {
                    asm volatile("barrier.sync %0, %1;" :: "r"(1 + s), "r"(ct_vec) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn = k + 2;
            if (kn < n_stages) {
                int GWn_2 = ct_vec >> 5;
                int bslot_0 = sub * 2 + slot;
                int rows_k_2 = 0;
                if (active != 0) {
                    int _max_49 = ((my_rows - kn * 8) > (0) ? (my_rows - kn * 8) : (0));
                    int _min_48 = ((_max_49) < (8) ? (_max_49) : (8));
                    rows_k_2 = _min_48;
                }
                if (v == 0 && rows_k_2 > 0) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (bslot_0) * 8, rows_k_2 * (2 * RS));
                    int sx_1 = (int)stage_buf_addr + (sub * (GWn_2 * 2) + slot * GWn_2) * 4096;
                    long long src0_1 = (long long)(t0 + kn * 8) * C64 + c64;
                    cp_async_bulk_gmem2smem(sx_1, reinterpret_cast<const uint8_t*>(x + src0_1), rows_k_2 * RS, stage_full_addr + (bslot_0) * 8);
                    cp_async_bulk_gmem2smem(sx_1 + 32768, reinterpret_cast<const uint8_t*>(dy + src0_1), rows_k_2 * RS, stage_full_addr + (bslot_0) * 8);
                } else if (v < 32) {
                    mbarrier_arrive(stage_full_addr + (bslot_0) * 8);
                }
            }
        }
    } else {
        int warp_1 = tid >> 5;
        int lane_1 = tid & 31;
        int li = ((lane_1) < (v) ? (lane_1) : (v));
        int seg_lanes = ((32 - (lane_1 - li)) < (ct_vec - (v - li)) ? (32 - (lane_1 - li)) : (ct_vec - (v - li)));
        int rstride = seg_lanes * 16;
        int seg_off = (lane_1 - li) * 128;
        int whole_row = 0;
        if (rstride == C * 2) {
            whole_row = 1;
        }
        long long c64_1 = (long long)c0;
        int t_end_1 = ((t0 + L) < (S) ? (t0 + L) : (S));
        int t_stop_1 = ((t_end_1 + 3) < (S + 1) ? (t_end_1 + 3) : (S + 1));
        int t_lim_1 = ((t_stop_1) < (S) ? (t_stop_1) : (S));
        int active_1 = 0;
        if (sub < n_sub && c0 < C && t0 < S) {
            active_1 = 1;
        }
        int my_rows_1 = 0;
        int my_steps_1 = 0;
        if (active_1 != 0) {
            my_rows_1 = t_lim_1 - t0;
            my_steps_1 = t_stop_1 - t0;
        }
        int sub0 = warp_1 * 32 / ct_vec;
        int t00 = (rank * n_sub + sub0) * L;
        int steps0 = 0;
        if (sub0 < n_sub && (ct * ct_vec + (warp_1 * 32 - sub0 * ct_vec)) * 8 < C && t00 < S) {
            int _min_51 = ((t00 + L) < (S) ? (t00 + L) : (S));
            int _min_50 = ((_min_51 + 3) < (S + 1) ? (_min_51 + 3) : (S + 1));
            steps0 = _min_50 - t00;
        }
        int n_stages_1 = (steps0 + 8 - 1) / 8;
        int _min_1 = ((n_stages_1) < (2) ? (n_stages_1) : (2));
        #pragma unroll 1
        for (int k0_1 = 0; k0_1 < _min_1; k0_1++) {
            int slot0_1 = k0_1;
            int bslot_2 = warp_1 * 2 + slot0_1;
            int rows_k_3 = 0;
            if (active_1 != 0) {
                int _max_53 = ((my_rows_1 - k0_1 * 8) > (0) ? (my_rows_1 - k0_1 * 8) : (0));
                int _min_52 = ((_max_53) < (8) ? (_max_53) : (8));
                rows_k_3 = _min_52;
            }
            if (li == 0 && rows_k_3 > 0) {
                int cb = ((rstride) < ((int)(C64 - c64_1) * 2) ? (rstride) : ((int)(C64 - c64_1) * 2));
                int sx_2 = (int)stage_buf_addr + bslot_2 * 8192 + seg_off;
                long long src0_2 = (long long)(t0 + k0_1 * 8) * C64 + c64_1;
                if (whole_row != 0) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (bslot_2) * 8, rows_k_3 * (2 * cb));
                    cp_async_bulk_gmem2smem(sx_2, reinterpret_cast<const uint8_t*>(x + src0_2), rows_k_3 * cb, stage_full_addr + (bslot_2) * 8);
                    cp_async_bulk_gmem2smem(sx_2 + 4096, reinterpret_cast<const uint8_t*>(dy + src0_2), rows_k_3 * cb, stage_full_addr + (bslot_2) * 8);
                } else if (rstride == 512) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (bslot_2) * 8, 8192);
                    int cc = (int)c64_1;
                    tma_2d_gmem2smem(sx_2, (&x_map), cc, t0 + k0_1 * 8, stage_full_addr + (bslot_2) * 8);
                    tma_2d_gmem2smem(sx_2 + 4096, (&dy_map), cc, t0 + k0_1 * 8, stage_full_addr + (bslot_2) * 8);
                } else {
                    mbarrier_arrive_expect_tx(stage_full_addr + (bslot_2) * 8, rows_k_3 * (2 * cb));
                    #pragma unroll 1
                    for (int i_2 = 0; i_2 < rows_k_3; i_2++) {
                        long long src = src0_2 + (long long)i_2 * C64;
                        cp_async_bulk_gmem2smem(sx_2 + i_2 * rstride, reinterpret_cast<const uint8_t*>(x + src), cb, stage_full_addr + (bslot_2) * 8);
                        cp_async_bulk_gmem2smem(sx_2 + 4096 + i_2 * rstride, reinterpret_cast<const uint8_t*>(dy + src), cb, stage_full_addr + (bslot_2) * 8);
                    }
                }
            } else {
                mbarrier_arrive(stage_full_addr + (bslot_2) * 8);
            }
        }
        float wr_1[32];
        float w4_1[4];
        float win_1[32];
        float pend_1[32];
        float tailf_1[8];
        #pragma unroll
        for (int i_3 = 0; i_3 < 32; i_3++) {
            win_1[i_3] = 0.0f;
            pend_1[i_3] = 0.0f;
        }
        int lo_1 = 0;
        int hi_1 = num_docs - 1;
        int doc_start_1 = 0;
        int doc_end_1 = 0;
        int frag_1 = 0;
        if (active_1 != 0) {
            #pragma unroll
            for (int j_8 = 0; j_8 < 8; j_8++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(w + (c64_1 + (long long)j_8) * 4);
                    wr_1[j_8 * 4 + 0] = _v4.x;
                    wr_1[j_8 * 4 + 1] = _v4.y;
                    wr_1[j_8 * 4 + 2] = _v4.z;
                    wr_1[j_8 * 4 + 3] = _v4.w;
                }
            }
            #pragma unroll 1
            for (int step_1 = 0; step_1 < search_steps; step_1++) {
                int mid_1 = lo_1 + hi_1 + 1 >> 1;
                if (t0 >= cu[mid_1]) {
                    lo_1 = mid_1;
                } else {
                    hi_1 = mid_1 - 1;
                }
            }
            doc_start_1 = cu[lo_1];
            doc_end_1 = cu[lo_1 + 1];
            frag_1 = lo_1;
            int tk_2 = t0 - 1;
            if (tk_2 >= doc_start_1) {
                long long rowk_3 = (long long)tk_2 * C64 + c64_1;
                {
                    const uint4* _vptr_55 = reinterpret_cast<const uint4*>(x + rowk_3);
                    uint4 _vld_55[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_55[_blk] = _vptr_55[_blk];
                        uint32_t* _vpairs_55 = reinterpret_cast<uint32_t*>(&_vld_55[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win_1[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) << 16);
                            (&win_1[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_4 = hist_row_off[frag_1 * 3 + 2 - (doc_start_1 - t0)];
                if (hrow_4 >= 0) {
                    long long rowh_6 = (long long)hrow_4 * C64 + c64_1;
                    {
                        const uint4* _vptr_56 = reinterpret_cast<const uint4*>(hist + rowh_6);
                        uint4 _vld_56[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_56[_blk] = _vptr_56[_blk];
                            uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
                                (&win_1[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            int tk_0_1 = t0 - 2;
            if (tk_0_1 >= doc_start_1) {
                long long rowk_4 = (long long)tk_0_1 * C64 + c64_1;
                {
                    const uint4* _vptr_57 = reinterpret_cast<const uint4*>(x + rowk_4);
                    uint4 _vld_57[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_57[_blk] = _vptr_57[_blk];
                        uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win_1[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
                            (&win_1[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_5 = hist_row_off[frag_1 * 3 + 1 - (doc_start_1 - t0)];
                if (hrow_5 >= 0) {
                    long long rowh_7 = (long long)hrow_5 * C64 + c64_1;
                    {
                        const uint4* _vptr_58 = reinterpret_cast<const uint4*>(hist + rowh_7);
                        uint4 _vld_58[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_58[_blk] = _vptr_58[_blk];
                            uint32_t* _vpairs_58 = reinterpret_cast<uint32_t*>(&_vld_58[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) << 16);
                                (&win_1[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            int tk_1_1 = t0 - 3;
            if (tk_1_1 >= doc_start_1) {
                long long rowk_5 = (long long)tk_1_1 * C64 + c64_1;
                {
                    const uint4* _vptr_59 = reinterpret_cast<const uint4*>(x + rowk_5);
                    uint4 _vld_59[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_59[_blk] = _vptr_59[_blk];
                        uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win_1[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
                            (&win_1[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_6 = hist_row_off[frag_1 * 3 - (doc_start_1 - t0)];
                if (hrow_6 >= 0) {
                    long long rowh_8 = (long long)hrow_6 * C64 + c64_1;
                    {
                        const uint4* _vptr_60 = reinterpret_cast<const uint4*>(hist + rowh_8);
                        uint4 _vld_60[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_60[_blk] = _vptr_60[_blk];
                            uint32_t* _vpairs_60 = reinterpret_cast<uint32_t*>(&_vld_60[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) << 16);
                                (&win_1[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
        }
        unsigned int pkx_1[4];
        unsigned int pkd_1[4];
        float dyf_1[8];
        float dp_1[8];
        #pragma unroll 1
        for (int k_1 = 0; k_1 < n_stages_1; k_1++) {
            int slot_1 = k_1 % 2;
            int parity_1 = k_1 / 2 & 1;
            int bslot_3 = warp_1 * 2 + slot_1;
            mbarrier_wait(stage_full_addr + (bslot_3) * 8, parity_1);
            if (active_1 != 0) {
                int tb_1 = t0 + k_1 * 8;
                int _max_61 = ((my_rows_1 - k_1 * 8) > (0) ? (my_rows_1 - k_1 * 8) : (0));
                int rows_k_4 = ((_max_61) < (8) ? (_max_61) : (8));
                int _max_62 = ((my_steps_1 - k_1 * 8) > (0) ? (my_steps_1 - k_1 * 8) : (0));
                int steps_k_1 = ((_max_62) < (8) ? (_max_62) : (8));
                int sx0_1 = (int)stage_buf_addr + bslot_3 * 8192 + seg_off + li * 16;
                int interior_1 = 0;
                if (steps_k_1 == 8 && rows_k_4 == 8 && t_end_1 >= tb_1 + 8 && doc_end_1 >= tb_1 + 8) {
                    if (k_1 >= 1) {
                        if (doc_start_1 <= tb_1 - 3) {
                            interior_1 = 1;
                        }
                    }
                }
                if (interior_1 != 0) {
                    #pragma unroll
                    for (int u_2 = 0; u_2 < 8; u_2++) {
                        __int128_t _smem_raw_63;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_63) : "r"(sx0_1 + u_2 * rstride));
                        pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[0];
                        pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[1];
                        pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[2];
                        pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[3];
                        __int128_t _smem_raw_64;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_64) : "r"(sx0_1 + 4096 + u_2 * rstride));
                        pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[0];
                        pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[1];
                        pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[2];
                        pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[3];
                        #pragma unroll
                        for (int kk_10 = 0; kk_10 < 4; kk_10++) {
                            win_1[u_2 % 4 * 8 + 2 * kk_10] = __uint_as_float(pkx_1[kk_10] << 16);
                            win_1[u_2 % 4 * 8 + 2 * kk_10 + 1] = __uint_as_float(pkx_1[kk_10] & 4294901760u);
                            dyf_1[2 * kk_10] = __uint_as_float(pkd_1[kk_10] << 16);
                            dyf_1[2 * kk_10 + 1] = __uint_as_float(pkd_1[kk_10] & 4294901760u);
                        }
                        #pragma unroll
                        for (int j_9 = 0; j_9 < 8; j_9++) {
                            #pragma unroll
                            for (int kk_11 = 0; kk_11 < 4; kk_11++) {
                                w4_1[kk_11] = wr_1[j_9 * 4 + kk_11];
                            }
                            float _fma_66;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(w4_1[1]), "f"(win_1[(u_2 % 4 + 2) % 4 * 8 + j_9]), "f"(mul_f32_rn_noftz(w4_1[0], win_1[(u_2 % 4 + 1) % 4 * 8 + j_9])));
                            float _fma_65;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(w4_1[2]), "f"(win_1[(u_2 % 4 + 3) % 4 * 8 + j_9]), "f"(_fma_66));
                            float _fma_3;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(w4_1[3]), "f"(win_1[u_2 % 4 * 8 + j_9]), "f"(_fma_65));
                            float e_3 = (-_fma_3) * 1.4426950408889634f;
                            float _exp2_67 = approx_exp2(e_3);
                            float den_3 = 1.0f + _exp2_67;
                            float sg_3 = approx_rcp(den_3);
                            float _fma_68;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(_fma_3), "f"(1.0f - sg_3), "f"(1.0f));
                            dp_1[j_9] = dyf_1[j_9] * (sg_3 * _fma_68);
                            #pragma unroll
                            for (int kk_12 = 0; kk_12 < 4; kk_12++) {
                                acc[j_9 * 4 + kk_12] = __fmaf_rn(dp_1[j_9], win_1[(u_2 % 4 + 1 + kk_12) % 4 * 8 + j_9], acc[j_9 * 4 + kk_12]);
                            }
                            float _fma_69;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(w4_1[0]), "f"(dp_1[j_9]), "f"(pend_1[(u_2 % 4 + 1) % 4 * 8 + j_9]));
                            pend_1[(u_2 % 4 + 1) % 4 * 8 + j_9] = _fma_69;
                            float _fma_70;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(w4_1[1]), "f"(dp_1[j_9]), "f"(pend_1[(u_2 % 4 + 2) % 4 * 8 + j_9]));
                            pend_1[(u_2 % 4 + 2) % 4 * 8 + j_9] = _fma_70;
                            float _fma_71;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(w4_1[2]), "f"(dp_1[j_9]), "f"(pend_1[(u_2 % 4 + 3) % 4 * 8 + j_9]));
                            pend_1[(u_2 % 4 + 3) % 4 * 8 + j_9] = _fma_71;
                            pend_1[u_2 % 4 * 8 + j_9] = w4_1[3] * dp_1[j_9];
                        }
                        long long row3_2 = (long long)(tb_1 + (u_2 - 3)) * C64 + c64_1;
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(pend_1[(u_2 % 4 + 1) % 4 * 8 + 0], pend_1[(u_2 % 4 + 1) % 4 * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(pend_1[(u_2 % 4 + 1) % 4 * 8 + 2], pend_1[(u_2 % 4 + 1) % 4 * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(pend_1[(u_2 % 4 + 1) % 4 * 8 + 4], pend_1[(u_2 % 4 + 1) % 4 * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(pend_1[(u_2 % 4 + 1) % 4 * 8 + 6], pend_1[(u_2 % 4 + 1) % 4 * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[row3_2 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int u_3 = 0; u_3 < 8; u_3++) {
                        if (steps_k_1 > u_3) {
                            int tcur_1 = tb_1 + u_3;
                            if (rows_k_4 > u_3) {
                                __int128_t _smem_raw_72;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_72) : "r"(sx0_1 + u_3 * rstride));
                                pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[0];
                                pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[1];
                                pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[2];
                                pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[3];
                                __int128_t _smem_raw_73;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_73) : "r"(sx0_1 + 4096 + u_3 * rstride));
                                pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_73)[0];
                                pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_73)[1];
                                pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_73)[2];
                                pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_73)[3];
                                #pragma unroll
                                for (int kk_13 = 0; kk_13 < 4; kk_13++) {
                                    win_1[u_3 % 4 * 8 + 2 * kk_13] = __uint_as_float(pkx_1[kk_13] << 16);
                                    win_1[u_3 % 4 * 8 + 2 * kk_13 + 1] = __uint_as_float(pkx_1[kk_13] & 4294901760u);
                                    dyf_1[2 * kk_13] = __uint_as_float(pkd_1[kk_13] << 16);
                                    dyf_1[2 * kk_13 + 1] = __uint_as_float(pkd_1[kk_13] & 4294901760u);
                                }
                            }
                            if (tcur_1 >= doc_end_1) {
                                int trow_2 = tail_row_off[frag_1 * 3];
                                if (trow_2 >= 0) {
                                    long long rowr_3 = (long long)trow_2 * C64 + c64_1;
                                    {
                                        unsigned _ldv8_74_0;
                                        unsigned _ldv8_74_1;
                                        unsigned _ldv8_74_2;
                                        unsigned _ldv8_74_3;
                                        unsigned _ldv8_74_4;
                                        unsigned _ldv8_74_5;
                                        unsigned _ldv8_74_6;
                                        unsigned _ldv8_74_7;
                                        asm volatile(
                                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_ldv8_74_0), "=r"(_ldv8_74_1), "=r"(_ldv8_74_2), "=r"(_ldv8_74_3), "=r"(_ldv8_74_4), "=r"(_ldv8_74_5), "=r"(_ldv8_74_6), "=r"(_ldv8_74_7) : "l"((const void*)(dpre_tail + (rowr_3))) : "memory");
                                        tailf_1[0 + 0] = __uint_as_float(_ldv8_74_0);
                                        tailf_1[0 + 1] = __uint_as_float(_ldv8_74_1);
                                        tailf_1[0 + 2] = __uint_as_float(_ldv8_74_2);
                                        tailf_1[0 + 3] = __uint_as_float(_ldv8_74_3);
                                        tailf_1[0 + 4] = __uint_as_float(_ldv8_74_4);
                                        tailf_1[0 + 5] = __uint_as_float(_ldv8_74_5);
                                        tailf_1[0 + 6] = __uint_as_float(_ldv8_74_6);
                                        tailf_1[0 + 7] = __uint_as_float(_ldv8_74_7);
                                    }
                                    #pragma unroll
                                    for (int j_10 = 0; j_10 < 8; j_10++) {
                                        #pragma unroll
                                        for (int kk_14 = 0; kk_14 < 4; kk_14++) {
                                            w4_1[kk_14] = wr_1[j_10 * 4 + kk_14];
                                        }
                                        float _fma_75;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(w4_1[2]), "f"(tailf_1[j_10]), "f"(pend_1[(u_3 % 4 + 4 - 1) % 4 * 8 + j_10]));
                                        pend_1[(u_3 % 4 + 4 - 1) % 4 * 8 + j_10] = _fma_75;
                                        float _fma_76;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(w4_1[1]), "f"(tailf_1[j_10]), "f"(pend_1[(u_3 % 4 + 4 - 2) % 4 * 8 + j_10]));
                                        pend_1[(u_3 % 4 + 4 - 2) % 4 * 8 + j_10] = _fma_76;
                                        float _fma_77;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(w4_1[0]), "f"(tailf_1[j_10]), "f"(pend_1[(u_3 % 4 + 4 - 3) % 4 * 8 + j_10]));
                                        pend_1[(u_3 % 4 + 4 - 3) % 4 * 8 + j_10] = _fma_77;
                                    }
                                }
                                int trow_0_1 = tail_row_off[frag_1 * 3 + 1];
                                if (trow_0_1 >= 0) {
                                    long long rowr_4 = (long long)trow_0_1 * C64 + c64_1;
                                    {
                                        unsigned _ldv8_78_0;
                                        unsigned _ldv8_78_1;
                                        unsigned _ldv8_78_2;
                                        unsigned _ldv8_78_3;
                                        unsigned _ldv8_78_4;
                                        unsigned _ldv8_78_5;
                                        unsigned _ldv8_78_6;
                                        unsigned _ldv8_78_7;
                                        asm volatile(
                                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_ldv8_78_0), "=r"(_ldv8_78_1), "=r"(_ldv8_78_2), "=r"(_ldv8_78_3), "=r"(_ldv8_78_4), "=r"(_ldv8_78_5), "=r"(_ldv8_78_6), "=r"(_ldv8_78_7) : "l"((const void*)(dpre_tail + (rowr_4))) : "memory");
                                        tailf_1[0 + 0] = __uint_as_float(_ldv8_78_0);
                                        tailf_1[0 + 1] = __uint_as_float(_ldv8_78_1);
                                        tailf_1[0 + 2] = __uint_as_float(_ldv8_78_2);
                                        tailf_1[0 + 3] = __uint_as_float(_ldv8_78_3);
                                        tailf_1[0 + 4] = __uint_as_float(_ldv8_78_4);
                                        tailf_1[0 + 5] = __uint_as_float(_ldv8_78_5);
                                        tailf_1[0 + 6] = __uint_as_float(_ldv8_78_6);
                                        tailf_1[0 + 7] = __uint_as_float(_ldv8_78_7);
                                    }
                                    #pragma unroll
                                    for (int j_11 = 0; j_11 < 8; j_11++) {
                                        #pragma unroll
                                        for (int kk_15 = 0; kk_15 < 4; kk_15++) {
                                            w4_1[kk_15] = wr_1[j_11 * 4 + kk_15];
                                        }
                                        float _fma_79;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(w4_1[1]), "f"(tailf_1[j_11]), "f"(pend_1[(u_3 % 4 + 4 + 1 - 2) % 4 * 8 + j_11]));
                                        pend_1[(u_3 % 4 + 4 + 1 - 2) % 4 * 8 + j_11] = _fma_79;
                                        float _fma_80;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(w4_1[0]), "f"(tailf_1[j_11]), "f"(pend_1[(u_3 % 4 + 4 + 1 - 3) % 4 * 8 + j_11]));
                                        pend_1[(u_3 % 4 + 4 + 1 - 3) % 4 * 8 + j_11] = _fma_80;
                                    }
                                }
                                int trow_1_1 = tail_row_off[frag_1 * 3 + 2];
                                if (trow_1_1 >= 0) {
                                    long long rowr_5 = (long long)trow_1_1 * C64 + c64_1;
                                    {
                                        unsigned _ldv8_81_0;
                                        unsigned _ldv8_81_1;
                                        unsigned _ldv8_81_2;
                                        unsigned _ldv8_81_3;
                                        unsigned _ldv8_81_4;
                                        unsigned _ldv8_81_5;
                                        unsigned _ldv8_81_6;
                                        unsigned _ldv8_81_7;
                                        asm volatile(
                                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_ldv8_81_0), "=r"(_ldv8_81_1), "=r"(_ldv8_81_2), "=r"(_ldv8_81_3), "=r"(_ldv8_81_4), "=r"(_ldv8_81_5), "=r"(_ldv8_81_6), "=r"(_ldv8_81_7) : "l"((const void*)(dpre_tail + (rowr_5))) : "memory");
                                        tailf_1[0 + 0] = __uint_as_float(_ldv8_81_0);
                                        tailf_1[0 + 1] = __uint_as_float(_ldv8_81_1);
                                        tailf_1[0 + 2] = __uint_as_float(_ldv8_81_2);
                                        tailf_1[0 + 3] = __uint_as_float(_ldv8_81_3);
                                        tailf_1[0 + 4] = __uint_as_float(_ldv8_81_4);
                                        tailf_1[0 + 5] = __uint_as_float(_ldv8_81_5);
                                        tailf_1[0 + 6] = __uint_as_float(_ldv8_81_6);
                                        tailf_1[0 + 7] = __uint_as_float(_ldv8_81_7);
                                    }
                                    #pragma unroll
                                    for (int j_12 = 0; j_12 < 8; j_12++) {
                                        #pragma unroll
                                        for (int kk_16 = 0; kk_16 < 4; kk_16++) {
                                            w4_1[kk_16] = wr_1[j_12 * 4 + kk_16];
                                        }
                                        float _fma_82;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(w4_1[0]), "f"(tailf_1[j_12]), "f"(pend_1[(u_3 % 4 + 4 + 2 - 3) % 4 * 8 + j_12]));
                                        pend_1[(u_3 % 4 + 4 + 2 - 3) % 4 * 8 + j_12] = _fma_82;
                                    }
                                }
                                #pragma unroll
                                for (int m_2 = 1; m_2 < 4; m_2++) {
                                    int tm_1 = tcur_1 - m_2;
                                    if (tm_1 >= t0 && tm_1 < t_end_1 && tm_1 >= doc_start_1) {
                                        long long rowm_1 = (long long)tm_1 * C64 + c64_1;
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(pend_1[(u_3 % 4 + 4 - m_2) % 4 * 8 + 0], pend_1[(u_3 % 4 + 4 - m_2) % 4 * 8 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(pend_1[(u_3 % 4 + 4 - m_2) % 4 * 8 + 2], pend_1[(u_3 % 4 + 4 - m_2) % 4 * 8 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(pend_1[(u_3 % 4 + 4 - m_2) % 4 * 8 + 4], pend_1[(u_3 % 4 + 4 - m_2) % 4 * 8 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(pend_1[(u_3 % 4 + 4 - m_2) % 4 * 8 + 6], pend_1[(u_3 % 4 + 4 - m_2) % 4 * 8 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[rowm_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int m_3 = 1; m_3 < 4; m_3++) {
                                    #pragma unroll
                                    for (int j_13 = 0; j_13 < 8; j_13++) {
                                        win_1[(u_3 % 4 + 4 - m_3) % 4 * 8 + j_13] = 0.0f;
                                    }
                                }
                                if (tcur_1 < S) {
                                    int lo2_1 = 0;
                                    int hi2_1 = num_docs - 1;
                                    #pragma unroll 1
                                    for (int step2_1 = 0; step2_1 < search_steps; step2_1++) {
                                        int mid2_1 = lo2_1 + hi2_1 + 1 >> 1;
                                        if (tcur_1 >= cu[mid2_1]) {
                                            lo2_1 = mid2_1;
                                        } else {
                                            hi2_1 = mid2_1 - 1;
                                        }
                                    }
                                    doc_start_1 = cu[lo2_1];
                                    doc_end_1 = cu[lo2_1 + 1];
                                    frag_1 = lo2_1;
                                    int hrow_7 = hist_row_off[frag_1 * 3 + 2];
                                    if (hrow_7 >= 0) {
                                        long long rowh_9 = (long long)hrow_7 * C64 + c64_1;
                                        {
                                            const uint4* _vptr_83 = reinterpret_cast<const uint4*>(hist + rowh_9);
                                            uint4 _vld_83[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_83[_blk] = _vptr_83[_blk];
                                                uint32_t* _vpairs_83 = reinterpret_cast<uint32_t*>(&_vld_83[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win_1[(u_3 % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) << 16);
                                                    (&win_1[(u_3 % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                    int hrow_0_1 = hist_row_off[frag_1 * 3 + 1];
                                    if (hrow_0_1 >= 0) {
                                        long long rowh_10 = (long long)hrow_0_1 * C64 + c64_1;
                                        {
                                            const uint4* _vptr_84 = reinterpret_cast<const uint4*>(hist + rowh_10);
                                            uint4 _vld_84[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_84[_blk] = _vptr_84[_blk];
                                                uint32_t* _vpairs_84 = reinterpret_cast<uint32_t*>(&_vld_84[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win_1[(u_3 % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_84[_pair]) << 16);
                                                    (&win_1[(u_3 % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_84[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                    int hrow_1_2 = hist_row_off[frag_1 * 3];
                                    if (hrow_1_2 >= 0) {
                                        long long rowh_11 = (long long)hrow_1_2 * C64 + c64_1;
                                        {
                                            const uint4* _vptr_85 = reinterpret_cast<const uint4*>(hist + rowh_11);
                                            uint4 _vld_85[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_85[_blk] = _vptr_85[_blk];
                                                uint32_t* _vpairs_85 = reinterpret_cast<uint32_t*>(&_vld_85[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win_1[(u_3 % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) << 16);
                                                    (&win_1[(u_3 % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    doc_start_1 = tcur_1 + 8;
                                    doc_end_1 = tcur_1 + 8;
                                }
                            }
                            if (tcur_1 < S) {
                                if (tcur_1 < t_end_1) {
                                    #pragma unroll
                                    for (int j_14 = 0; j_14 < 8; j_14++) {
                                        #pragma unroll
                                        for (int kk_17 = 0; kk_17 < 4; kk_17++) {
                                            w4_1[kk_17] = wr_1[j_14 * 4 + kk_17];
                                        }
                                        float _fma_87;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(w4_1[1]), "f"(win_1[(u_3 % 4 + 2) % 4 * 8 + j_14]), "f"(mul_f32_rn_noftz(w4_1[0], win_1[(u_3 % 4 + 1) % 4 * 8 + j_14])));
                                        float _fma_86;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(w4_1[2]), "f"(win_1[(u_3 % 4 + 3) % 4 * 8 + j_14]), "f"(_fma_87));
                                        float _fma_4;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(w4_1[3]), "f"(win_1[u_3 % 4 * 8 + j_14]), "f"(_fma_86));
                                        float e_4 = (-_fma_4) * 1.4426950408889634f;
                                        float _exp2_88 = approx_exp2(e_4);
                                        float den_4 = 1.0f + _exp2_88;
                                        float sg_4 = approx_rcp(den_4);
                                        float _fma_89;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(_fma_4), "f"(1.0f - sg_4), "f"(1.0f));
                                        dp_1[j_14] = dyf_1[j_14] * (sg_4 * _fma_89);
                                        #pragma unroll
                                        for (int kk_18 = 0; kk_18 < 4; kk_18++) {
                                            acc[j_14 * 4 + kk_18] = __fmaf_rn(dp_1[j_14], win_1[(u_3 % 4 + 1 + kk_18) % 4 * 8 + j_14], acc[j_14 * 4 + kk_18]);
                                        }
                                        float _fma_90;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(w4_1[0]), "f"(dp_1[j_14]), "f"(pend_1[(u_3 % 4 + 1) % 4 * 8 + j_14]));
                                        pend_1[(u_3 % 4 + 1) % 4 * 8 + j_14] = _fma_90;
                                        float _fma_91;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(w4_1[1]), "f"(dp_1[j_14]), "f"(pend_1[(u_3 % 4 + 2) % 4 * 8 + j_14]));
                                        pend_1[(u_3 % 4 + 2) % 4 * 8 + j_14] = _fma_91;
                                        float _fma_92;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(w4_1[2]), "f"(dp_1[j_14]), "f"(pend_1[(u_3 % 4 + 3) % 4 * 8 + j_14]));
                                        pend_1[(u_3 % 4 + 3) % 4 * 8 + j_14] = _fma_92;
                                        pend_1[u_3 % 4 * 8 + j_14] = w4_1[3] * dp_1[j_14];
                                    }
                                } else {
                                    #pragma unroll
                                    for (int j_15 = 0; j_15 < 8; j_15++) {
                                        #pragma unroll
                                        for (int kk_19 = 0; kk_19 < 4; kk_19++) {
                                            w4_1[kk_19] = wr_1[j_15 * 4 + kk_19];
                                        }
                                        float _fma_94;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(w4_1[1]), "f"(win_1[(u_3 % 4 + 2) % 4 * 8 + j_15]), "f"(mul_f32_rn_noftz(w4_1[0], win_1[(u_3 % 4 + 1) % 4 * 8 + j_15])));
                                        float _fma_93;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(w4_1[2]), "f"(win_1[(u_3 % 4 + 3) % 4 * 8 + j_15]), "f"(_fma_94));
                                        float _fma_5;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(w4_1[3]), "f"(win_1[u_3 % 4 * 8 + j_15]), "f"(_fma_93));
                                        float e_5 = (-_fma_5) * 1.4426950408889634f;
                                        float _exp2_95 = approx_exp2(e_5);
                                        float den_5 = 1.0f + _exp2_95;
                                        float sg_5 = approx_rcp(den_5);
                                        float _fma_96;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(_fma_5), "f"(1.0f - sg_5), "f"(1.0f));
                                        dp_1[j_15] = dyf_1[j_15] * (sg_5 * _fma_96);
                                        float _fma_97;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(w4_1[0]), "f"(dp_1[j_15]), "f"(pend_1[(u_3 % 4 + 1) % 4 * 8 + j_15]));
                                        pend_1[(u_3 % 4 + 1) % 4 * 8 + j_15] = _fma_97;
                                        float _fma_98;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(w4_1[1]), "f"(dp_1[j_15]), "f"(pend_1[(u_3 % 4 + 2) % 4 * 8 + j_15]));
                                        pend_1[(u_3 % 4 + 2) % 4 * 8 + j_15] = _fma_98;
                                        float _fma_99;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(w4_1[2]), "f"(dp_1[j_15]), "f"(pend_1[(u_3 % 4 + 3) % 4 * 8 + j_15]));
                                        pend_1[(u_3 % 4 + 3) % 4 * 8 + j_15] = _fma_99;
                                        pend_1[u_3 % 4 * 8 + j_15] = w4_1[3] * dp_1[j_15];
                                    }
                                }
                                int t3_1 = tcur_1 - 3;
                                if (t3_1 >= t0 && t3_1 >= doc_start_1) {
                                    long long row3_3 = (long long)t3_1 * C64 + c64_1;
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(pend_1[(u_3 % 4 + 1) % 4 * 8 + 0], pend_1[(u_3 % 4 + 1) % 4 * 8 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(pend_1[(u_3 % 4 + 1) % 4 * 8 + 2], pend_1[(u_3 % 4 + 1) % 4 * 8 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(pend_1[(u_3 % 4 + 1) % 4 * 8 + 4], pend_1[(u_3 % 4 + 1) % 4 * 8 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(pend_1[(u_3 % 4 + 1) % 4 * 8 + 6], pend_1[(u_3 % 4 + 1) % 4 * 8 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[row3_3 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn_1 = k_1 + 2;
            if (kn_1 < n_stages_1) {
                int bslot_0_1 = warp_1 * 2 + slot_1;
                int rows_k_5 = 0;
                if (active_1 != 0) {
                    int _max_101 = ((my_rows_1 - kn_1 * 8) > (0) ? (my_rows_1 - kn_1 * 8) : (0));
                    int _min_100 = ((_max_101) < (8) ? (_max_101) : (8));
                    rows_k_5 = _min_100;
                }
                if (li == 0 && rows_k_5 > 0) {
                    int cb_1 = ((rstride) < ((int)(C64 - c64_1) * 2) ? (rstride) : ((int)(C64 - c64_1) * 2));
                    int sx_3 = (int)stage_buf_addr + bslot_0_1 * 8192 + seg_off;
                    long long src0_3 = (long long)(t0 + kn_1 * 8) * C64 + c64_1;
                    if (whole_row != 0) {
                        mbarrier_arrive_expect_tx(stage_full_addr + (bslot_0_1) * 8, rows_k_5 * (2 * cb_1));
                        cp_async_bulk_gmem2smem(sx_3, reinterpret_cast<const uint8_t*>(x + src0_3), rows_k_5 * cb_1, stage_full_addr + (bslot_0_1) * 8);
                        cp_async_bulk_gmem2smem(sx_3 + 4096, reinterpret_cast<const uint8_t*>(dy + src0_3), rows_k_5 * cb_1, stage_full_addr + (bslot_0_1) * 8);
                    } else if (rstride == 512) {
                        mbarrier_arrive_expect_tx(stage_full_addr + (bslot_0_1) * 8, 8192);
                        int cc_1 = (int)c64_1;
                        tma_2d_gmem2smem(sx_3, (&x_map), cc_1, t0 + kn_1 * 8, stage_full_addr + (bslot_0_1) * 8);
                        tma_2d_gmem2smem(sx_3 + 4096, (&dy_map), cc_1, t0 + kn_1 * 8, stage_full_addr + (bslot_0_1) * 8);
                    } else {
                        mbarrier_arrive_expect_tx(stage_full_addr + (bslot_0_1) * 8, rows_k_5 * (2 * cb_1));
                        #pragma unroll 1
                        for (int i_4 = 0; i_4 < rows_k_5; i_4++) {
                            long long src_1 = src0_3 + (long long)i_4 * C64;
                            cp_async_bulk_gmem2smem(sx_3 + i_4 * rstride, reinterpret_cast<const uint8_t*>(x + src_1), cb_1, stage_full_addr + (bslot_0_1) * 8);
                            cp_async_bulk_gmem2smem(sx_3 + 4096 + i_4 * rstride, reinterpret_cast<const uint8_t*>(dy + src_1), cb_1, stage_full_addr + (bslot_0_1) * 8);
                        }
                    }
                } else {
                    mbarrier_arrive(stage_full_addr + (bslot_0_1) * 8);
                }
            }
        }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; i_5++) {
        red[tid * 32 + i_5] = acc[i_5];
    }
    __syncthreads();
    int c0_0 = (ct * ct_vec + v) * 8;
    if (sub == 0 && c0_0 < C) {
        float tot[32];
        #pragma unroll
        for (int i_6 = 0; i_6 < 32; i_6++) {
            tot[i_6] = red[v * 32 + i_6];
        }
        #pragma unroll 1
        for (int s_1 = 1; s_1 < n_sub; s_1++) {
            #pragma unroll
            for (int i_7 = 0; i_7 < 32; i_7++) {
                tot[i_7] = tot[i_7] + red[(s_1 * ct_vec + v) * 32 + i_7];
            }
        }
        #pragma unroll
        for (int j_16 = 0; j_16 < 8; j_16++) {
            long long pidx = ((long long)rank * C64 + (long long)c0_0 + (long long)j_16) * 4;
            {
                float4 _v4 = make_float4(tot[j_16 * 4 + 0], tot[j_16 * 4 + 1], tot[j_16 * 4 + 2], tot[j_16 * 4 + 3]);
                *reinterpret_cast<float4*>(partial + pidx) = _v4;
            }
        }
    }
    cooperative_groups::this_grid().sync();
    int lane_2 = tid & 31;
    int wch = tid >> 5;
    int n_groups = (C + 4 - 1) / 4;
    #pragma unroll 1
    for (int cg = bid; cg < n_groups; cg += num_bids) {
        int ch = cg * 4 + wch;
        if (ch < C) {
            int ctc = ch / 8 / ct_vec;
            int G = (num_bids - ctc + n_ch_tiles - 1) / n_ch_tiles;
            int n_trips = G + 31 >> 5;
            float a4[4];
            #pragma unroll
            for (int e_6 = 0; e_6 < 4; e_6++) {
                a4[e_6] = 0.0f;
            }
            #pragma unroll 4
            for (int k_2 = 0; k_2 < n_trips; k_2++) {
                int r = lane_2 + (k_2 << 5);
                if (r < G) {
                    float _vec_load_0[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial + ((long long)r * C64 + (long long)ch) * 4);
                        _vec_load_0[0 + 0] = _v4.x;
                        _vec_load_0[0 + 1] = _v4.y;
                        _vec_load_0[0 + 2] = _v4.z;
                        _vec_load_0[0 + 3] = _v4.w;
                    }
                    #pragma unroll
                    for (int e_7 = 0; e_7 < 4; e_7++) {
                        a4[e_7] = a4[e_7] + _vec_load_0[e_7];
                    }
                }
            }
            #pragma unroll
            for (int e_8 = 0; e_8 < 4; e_8++) {
                float other = __shfl_xor_sync(0xFFFFFFFF, a4[e_8], 16);
                a4[e_8] = a4[e_8] + other;
            }
            #pragma unroll
            for (int e_9 = 0; e_9 < 4; e_9++) {
                float other_1 = __shfl_xor_sync(0xFFFFFFFF, a4[e_9], 8);
                a4[e_9] = a4[e_9] + other_1;
            }
            #pragma unroll
            for (int e_10 = 0; e_10 < 4; e_10++) {
                float other_2 = __shfl_xor_sync(0xFFFFFFFF, a4[e_10], 4);
                a4[e_10] = a4[e_10] + other_2;
            }
            #pragma unroll
            for (int e_11 = 0; e_11 < 4; e_11++) {
                float other_3 = __shfl_xor_sync(0xFFFFFFFF, a4[e_11], 2);
                a4[e_11] = a4[e_11] + other_3;
            }
            #pragma unroll
            for (int e_12 = 0; e_12 < 4; e_12++) {
                float other_4 = __shfl_xor_sync(0xFFFFFFFF, a4[e_12], 1);
                a4[e_12] = a4[e_12] + other_4;
            }
            if (lane_2 == 0) {
                {
                    float4 _v4 = make_float4(a4[0 + 0], a4[0 + 1], a4[0 + 2], a4[0 + 3]);
                    *reinterpret_cast<float4*>(dw + (long long)ch * 4) = _v4;
                }
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
