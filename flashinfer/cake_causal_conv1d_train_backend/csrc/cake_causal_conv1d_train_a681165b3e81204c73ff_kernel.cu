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
#define SMEM_RED_STAGE_BYTES 20480
#define SMEM_RED_STRIDE 20480
#define SMEM_REDC_OFF 21504
#define SMEM_REDC_STAGE_BYTES 20480
#define SMEM_REDC_STRIDE 20480
#define SMEM_STAGE_BUF_OFF 41984
#define SMEM_STAGE_BUF_STAGE_BYTES 65536
#define SMEM_STAGE_BUF_STRIDE 65536
#define SMEM_TOTAL 107520
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 2

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
kernel_cake_causal_conv1d_train_a681165b3e81204c73ff(__nv_bfloat16* __restrict__ x, void* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ partial, void* __restrict__ b, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles, int flags, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap dy_map)
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
    float* redc = reinterpret_cast<float*>(smem_raw + 21504);
    const int redc_addr = smem + 21504;
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int stage_buf_addr = smem + 41984;

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
    float accb[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        accb[j] = 0.0f;
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 40; i_1++) {
        red[tid * 40 + i_1] = 0.0f;
        redc[tid * 40 + i_1] = 0.0f;
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
        float br[8];
        int wsm_addr = 0;
        float win[32];
        float pend[32];
        #pragma unroll
        for (int i_2 = 0; i_2 < 32; i_2++) {
            win[i_2] = 0.0f;
            pend[i_2] = 0.0f;
        }
        int lo = 0;
        int hi = num_docs - 1;
        int doc_start = 0;
        int doc_end = 0;
        if (active != 0) {
            if ((flags & 1) != 0) {
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    {
                        uint2 _vld_2;
                        _vld_2 = *reinterpret_cast<const uint2*>(reinterpret_cast<__nv_bfloat16*>(w) + (c64 + (long long)j_1) * 4);
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&wr[j_1 * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&wr[j_1 * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(w) + (c64 + (long long)j_2) * 4);
                        wr[j_2 * 4 + 0] = _v4.x;
                        wr[j_2 * 4 + 1] = _v4.y;
                        wr[j_2 * 4 + 2] = _v4.z;
                        wr[j_2 * 4 + 3] = _v4.w;
                    }
                }
            }
            if ((flags & 4) != 0) {
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(reinterpret_cast<__nv_bfloat16*>(b) + c64);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&br[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&br[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(b) + c64);
                    br[0 + 0] = _v4.x;
                    br[0 + 1] = _v4.y;
                    br[0 + 2] = _v4.z;
                    br[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(b) + c64 + 4);
                    br[4 + 0] = _v4.x;
                    br[4 + 1] = _v4.y;
                    br[4 + 2] = _v4.z;
                    br[4 + 3] = _v4.w;
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
                        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + rowk);
                        uint4 _vld_7[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_7[_blk] = _vptr_7[_blk];
                            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
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
                int _max_8 = ((my_rows - k * 8) > (0) ? (my_rows - k * 8) : (0));
                int rows_k_1 = ((_max_8) < (8) ? (_max_8) : (8));
                int _max_9 = ((my_steps - k * 8) > (0) ? (my_steps - k * 8) : (0));
                int steps_k = ((_max_9) < (8) ? (_max_9) : (8));
                int GWn_1 = ct_vec >> 5;
                int sx0 = (int)stage_buf_addr + (sub * (GWn_1 * 2) + slot * GWn_1) * 4096 + v * 16;
                int interior = 0;
                if (steps_k == 8 && rows_k_1 == 8 && t_end >= tb + 8 && doc_end >= tb + 8) {
                    if (k >= 1) {
                        if (doc_start <= tb - 3) {
                            interior = 2;
                        }
                    }
                }
                if (interior == 2) {
                    #pragma unroll
                    for (int u = 0; u < 8; u++) {
                        __int128_t _smem_raw_10;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(sx0 + u * RS));
                        pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                        pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                        pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                        pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                        __int128_t _smem_raw_11;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(sx0 + 32768 + u * RS));
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
                        for (int j_3 = 0; j_3 < 8; j_3++) {
                            #pragma unroll
                            for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                                w4[kk_1] = wr[j_3 * 4 + kk_1];
                            }
                            float _fma_14;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(w4[1]), "f"(win[(u % 4 + 2) % 4 * 8 + j_3]), "f"(mul_f32_rn_noftz(w4[0], win[(u % 4 + 1) % 4 * 8 + j_3])));
                            float _fma_13;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(w4[2]), "f"(win[(u % 4 + 3) % 4 * 8 + j_3]), "f"(_fma_14));
                            float _fma_12;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(w4[3]), "f"(win[u % 4 * 8 + j_3]), "f"(_fma_13));
                            float p;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(p) : "f"(_fma_12), "f"(br[j_3]));
                            float e = (-p) * 1.442695f;
                            float _exp2_15 = approx_exp2(e);
                            float den = 1.0f + _exp2_15;
                            float sg = approx_rcp(den);
                            float g = sg * __fmaf_rn(p, 1.0f - sg, 1.0f);
                            dp[j_3] = dyf[j_3] * g;
                            #pragma unroll
                            for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                                acc[j_3 * 4 + kk_2] = __fmaf_rn(dp[j_3], win[(u % 4 + 1 + kk_2) % 4 * 8 + j_3], acc[j_3 * 4 + kk_2]);
                            }
                            accb[j_3] = accb[j_3] + dp[j_3];
                            pend[(u % 4 + 1) % 4 * 8 + j_3] = __fmaf_rn(w4[0], dp[j_3], pend[(u % 4 + 1) % 4 * 8 + j_3]);
                            pend[(u % 4 + 2) % 4 * 8 + j_3] = __fmaf_rn(w4[1], dp[j_3], pend[(u % 4 + 2) % 4 * 8 + j_3]);
                            pend[(u % 4 + 3) % 4 * 8 + j_3] = __fmaf_rn(w4[2], dp[j_3], pend[(u % 4 + 3) % 4 * 8 + j_3]);
                            pend[u % 4 * 8 + j_3] = w4[3] * dp[j_3];
                        }
                        if (u >= 0) {
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
                    }
                } else if (interior == 1) {
                    #pragma unroll
                    for (int u_1 = 0; u_1 < 8; u_1++) {
                        __int128_t _smem_raw_16;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_16) : "r"(sx0 + u_1 * RS));
                        pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[0];
                        pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[1];
                        pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[2];
                        pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[3];
                        __int128_t _smem_raw_17;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_17) : "r"(sx0 + 32768 + u_1 * RS));
                        pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[0];
                        pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[1];
                        pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[2];
                        pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[3];
                        #pragma unroll
                        for (int kk_3 = 0; kk_3 < 4; kk_3++) {
                            win[u_1 % 4 * 8 + 2 * kk_3] = __uint_as_float(pkx[kk_3] << 16);
                            win[u_1 % 4 * 8 + 2 * kk_3 + 1] = __uint_as_float(pkx[kk_3] & 4294901760u);
                            dyf[2 * kk_3] = __uint_as_float(pkd[kk_3] << 16);
                            dyf[2 * kk_3 + 1] = __uint_as_float(pkd[kk_3] & 4294901760u);
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            #pragma unroll
                            for (int kk_4 = 0; kk_4 < 4; kk_4++) {
                                w4[kk_4] = wr[j_4 * 4 + kk_4];
                            }
                            float _fma_20;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(w4[1]), "f"(win[(u_1 % 4 + 2) % 4 * 8 + j_4]), "f"(mul_f32_rn_noftz(w4[0], win[(u_1 % 4 + 1) % 4 * 8 + j_4])));
                            float _fma_19;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(w4[2]), "f"(win[(u_1 % 4 + 3) % 4 * 8 + j_4]), "f"(_fma_20));
                            float _fma_18;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(w4[3]), "f"(win[u_1 % 4 * 8 + j_4]), "f"(_fma_19));
                            float p_1;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(p_1) : "f"(_fma_18), "f"(br[j_4]));
                            float e_1 = (-p_1) * 1.442695f;
                            float _exp2_21 = approx_exp2(e_1);
                            float den_1 = 1.0f + _exp2_21;
                            float sg_1 = approx_rcp(den_1);
                            float g_1 = sg_1 * __fmaf_rn(p_1, 1.0f - sg_1, 1.0f);
                            dp[j_4] = dyf[j_4] * g_1;
                            #pragma unroll
                            for (int kk_5 = 0; kk_5 < 4; kk_5++) {
                                acc[j_4 * 4 + kk_5] = __fmaf_rn(dp[j_4], win[(u_1 % 4 + 1 + kk_5) % 4 * 8 + j_4], acc[j_4 * 4 + kk_5]);
                            }
                            accb[j_4] = accb[j_4] + dp[j_4];
                            pend[(u_1 % 4 + 1) % 4 * 8 + j_4] = __fmaf_rn(w4[0], dp[j_4], pend[(u_1 % 4 + 1) % 4 * 8 + j_4]);
                            pend[(u_1 % 4 + 2) % 4 * 8 + j_4] = __fmaf_rn(w4[1], dp[j_4], pend[(u_1 % 4 + 2) % 4 * 8 + j_4]);
                            pend[(u_1 % 4 + 3) % 4 * 8 + j_4] = __fmaf_rn(w4[2], dp[j_4], pend[(u_1 % 4 + 3) % 4 * 8 + j_4]);
                            pend[u_1 % 4 * 8 + j_4] = w4[3] * dp[j_4];
                        }
                        if (u_1 >= 3) {
                            long long row3_1 = (long long)(tb + (u_1 - 3)) * C64 + c64;
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
                } else {
                    #pragma unroll
                    for (int u_2 = 0; u_2 < 8; u_2++) {
                        if (steps_k > u_2) {
                            int tcur = tb + u_2;
                            if (rows_k_1 > u_2) {
                                __int128_t _smem_raw_22;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_22) : "r"(sx0 + u_2 * RS));
                                pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[0];
                                pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[1];
                                pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[2];
                                pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[3];
                                __int128_t _smem_raw_23;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_23) : "r"(sx0 + 32768 + u_2 * RS));
                                pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[0];
                                pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[1];
                                pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[2];
                                pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_23)[3];
                                #pragma unroll
                                for (int kk_6 = 0; kk_6 < 4; kk_6++) {
                                    win[u_2 % 4 * 8 + 2 * kk_6] = __uint_as_float(pkx[kk_6] << 16);
                                    win[u_2 % 4 * 8 + 2 * kk_6 + 1] = __uint_as_float(pkx[kk_6] & 4294901760u);
                                    dyf[2 * kk_6] = __uint_as_float(pkd[kk_6] << 16);
                                    dyf[2 * kk_6 + 1] = __uint_as_float(pkd[kk_6] & 4294901760u);
                                }
                            }
                            if (tcur >= doc_end) {
                                #pragma unroll
                                for (int m = 1; m < 4; m++) {
                                    int tm = tcur - m;
                                    if (tm >= t0 && tm < t_end && tm >= doc_start) {
                                        long long rowm = (long long)tm * C64 + c64;
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(pend[(u_2 % 4 + 4 - m) % 4 * 8 + 0], pend[(u_2 % 4 + 4 - m) % 4 * 8 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(pend[(u_2 % 4 + 4 - m) % 4 * 8 + 2], pend[(u_2 % 4 + 4 - m) % 4 * 8 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(pend[(u_2 % 4 + 4 - m) % 4 * 8 + 4], pend[(u_2 % 4 + 4 - m) % 4 * 8 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(pend[(u_2 % 4 + 4 - m) % 4 * 8 + 6], pend[(u_2 % 4 + 4 - m) % 4 * 8 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[rowm + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
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
                                    doc_start = tcur + 8;
                                    doc_end = tcur + 8;
                                }
                                #pragma unroll
                                for (int m_1 = 1; m_1 < 4; m_1++) {
                                    #pragma unroll
                                    for (int j_5 = 0; j_5 < 8; j_5++) {
                                        win[(u_2 % 4 + 4 - m_1) % 4 * 8 + j_5] = 0.0f;
                                    }
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
                                        float _fma_26;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(w4[1]), "f"(win[(u_2 % 4 + 2) % 4 * 8 + j_6]), "f"(mul_f32_rn_noftz(w4[0], win[(u_2 % 4 + 1) % 4 * 8 + j_6])));
                                        float _fma_25;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(w4[2]), "f"(win[(u_2 % 4 + 3) % 4 * 8 + j_6]), "f"(_fma_26));
                                        float _fma_24;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(w4[3]), "f"(win[u_2 % 4 * 8 + j_6]), "f"(_fma_25));
                                        float p_2;
                                        asm("add.rn.f32 %0, %1, %2;" : "=f"(p_2) : "f"(_fma_24), "f"(br[j_6]));
                                        float e_2 = (-p_2) * 1.442695f;
                                        float _exp2_27 = approx_exp2(e_2);
                                        float den_2 = 1.0f + _exp2_27;
                                        float sg_2 = approx_rcp(den_2);
                                        float g_2 = sg_2 * __fmaf_rn(p_2, 1.0f - sg_2, 1.0f);
                                        dp[j_6] = dyf[j_6] * g_2;
                                        #pragma unroll
                                        for (int kk_8 = 0; kk_8 < 4; kk_8++) {
                                            acc[j_6 * 4 + kk_8] = __fmaf_rn(dp[j_6], win[(u_2 % 4 + 1 + kk_8) % 4 * 8 + j_6], acc[j_6 * 4 + kk_8]);
                                        }
                                        accb[j_6] = accb[j_6] + dp[j_6];
                                        pend[(u_2 % 4 + 1) % 4 * 8 + j_6] = __fmaf_rn(w4[0], dp[j_6], pend[(u_2 % 4 + 1) % 4 * 8 + j_6]);
                                        pend[(u_2 % 4 + 2) % 4 * 8 + j_6] = __fmaf_rn(w4[1], dp[j_6], pend[(u_2 % 4 + 2) % 4 * 8 + j_6]);
                                        pend[(u_2 % 4 + 3) % 4 * 8 + j_6] = __fmaf_rn(w4[2], dp[j_6], pend[(u_2 % 4 + 3) % 4 * 8 + j_6]);
                                        pend[u_2 % 4 * 8 + j_6] = w4[3] * dp[j_6];
                                    }
                                } else {
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 8; j_7++) {
                                        #pragma unroll
                                        for (int kk_9 = 0; kk_9 < 4; kk_9++) {
                                            w4[kk_9] = wr[j_7 * 4 + kk_9];
                                        }
                                        float _fma_30;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(w4[1]), "f"(win[(u_2 % 4 + 2) % 4 * 8 + j_7]), "f"(mul_f32_rn_noftz(w4[0], win[(u_2 % 4 + 1) % 4 * 8 + j_7])));
                                        float _fma_29;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(w4[2]), "f"(win[(u_2 % 4 + 3) % 4 * 8 + j_7]), "f"(_fma_30));
                                        float _fma_28;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(w4[3]), "f"(win[u_2 % 4 * 8 + j_7]), "f"(_fma_29));
                                        float p_3;
                                        asm("add.rn.f32 %0, %1, %2;" : "=f"(p_3) : "f"(_fma_28), "f"(br[j_7]));
                                        float e_3 = (-p_3) * 1.442695f;
                                        float _exp2_31 = approx_exp2(e_3);
                                        float den_3 = 1.0f + _exp2_31;
                                        float sg_3 = approx_rcp(den_3);
                                        float g_3 = sg_3 * __fmaf_rn(p_3, 1.0f - sg_3, 1.0f);
                                        dp[j_7] = dyf[j_7] * g_3;
                                        pend[(u_2 % 4 + 1) % 4 * 8 + j_7] = __fmaf_rn(w4[0], dp[j_7], pend[(u_2 % 4 + 1) % 4 * 8 + j_7]);
                                        pend[(u_2 % 4 + 2) % 4 * 8 + j_7] = __fmaf_rn(w4[1], dp[j_7], pend[(u_2 % 4 + 2) % 4 * 8 + j_7]);
                                        pend[(u_2 % 4 + 3) % 4 * 8 + j_7] = __fmaf_rn(w4[2], dp[j_7], pend[(u_2 % 4 + 3) % 4 * 8 + j_7]);
                                        pend[u_2 % 4 * 8 + j_7] = w4[3] * dp[j_7];
                                    }
                                }
                                int t3 = tcur - 3;
                                if (t3 >= t0 && t3 >= doc_start) {
                                    long long row3_2 = (long long)t3 * C64 + c64;
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(pend[(u_2 % 4 + 1) % 4 * 8 + 0], pend[(u_2 % 4 + 1) % 4 * 8 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(pend[(u_2 % 4 + 1) % 4 * 8 + 2], pend[(u_2 % 4 + 1) % 4 * 8 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(pend[(u_2 % 4 + 1) % 4 * 8 + 4], pend[(u_2 % 4 + 1) % 4 * 8 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(pend[(u_2 % 4 + 1) % 4 * 8 + 6], pend[(u_2 % 4 + 1) % 4 * 8 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[row3_2 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if ((k + 1) % 4 == 0) {
                int base = tid * 40;
                #pragma unroll
                for (int i_3 = 0; i_3 < 32; i_3++) {
                    float s = red[base + i_3];
                    float y;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(acc[i_3]), "f"(redc[base + i_3]));
                    float t;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t) : "f"(s), "f"(y));
                    float _sub_33;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(t), "f"(s));
                    float _sub_32;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(_sub_33), "f"(y));
                    redc[base + i_3] = _sub_32;
                    red[base + i_3] = t;
                    acc[i_3] = 0.0f;
                }
                #pragma unroll
                for (int j_8 = 0; j_8 < 8; j_8++) {
                    float s_1 = red[base + 32 + j_8];
                    float y_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_1) : "f"(accb[j_8]), "f"(redc[base + 32 + j_8]));
                    float t_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_1) : "f"(s_1), "f"(y_1));
                    float _sub_35;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(t_1), "f"(s_1));
                    float _sub_34;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(_sub_35), "f"(y_1));
                    redc[base + 32 + j_8] = _sub_34;
                    red[base + 32 + j_8] = t_1;
                    accb[j_8] = 0.0f;
                }
            }
            #pragma unroll
            for (int s_2 = 0; s_2 < 2; s_2++) {
                if (sub == s_2) {
                    asm volatile("barrier.sync %0, %1;" :: "r"(1 + s_2), "r"(ct_vec) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn = k + 2;
            if (kn < n_stages) {
                int GWn_2 = ct_vec >> 5;
                int bslot_0 = sub * 2 + slot;
                int rows_k_2 = 0;
                if (active != 0) {
                    int _max_37 = ((my_rows - kn * 8) > (0) ? (my_rows - kn * 8) : (0));
                    int _min_36 = ((_max_37) < (8) ? (_max_37) : (8));
                    rows_k_2 = _min_36;
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
        int full = 0;
        if (seg_lanes == 32) {
            full = 1;
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
            int _min_39 = ((t00 + L) < (S) ? (t00 + L) : (S));
            int _min_38 = ((_min_39 + 3) < (S + 1) ? (_min_39 + 3) : (S + 1));
            steps0 = _min_38 - t00;
        }
        int n_stages_1 = (steps0 + 8 - 1) / 8;
        int _min_1 = ((n_stages_1) < (2) ? (n_stages_1) : (2));
        #pragma unroll 1
        for (int k0_1 = 0; k0_1 < _min_1; k0_1++) {
            int slot0_1 = k0_1;
            int bslot_2 = warp_1 * 2 + slot0_1;
            int rows_k_3 = 0;
            if (active_1 != 0) {
                int _max_41 = ((my_rows_1 - k0_1 * 8) > (0) ? (my_rows_1 - k0_1 * 8) : (0));
                int _min_40 = ((_max_41) < (8) ? (_max_41) : (8));
                rows_k_3 = _min_40;
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
                    for (int i_4 = 0; i_4 < rows_k_3; i_4++) {
                        long long src = src0_2 + (long long)i_4 * C64;
                        cp_async_bulk_gmem2smem(sx_2 + i_4 * rstride, reinterpret_cast<const uint8_t*>(x + src), cb, stage_full_addr + (bslot_2) * 8);
                        cp_async_bulk_gmem2smem(sx_2 + 4096 + i_4 * rstride, reinterpret_cast<const uint8_t*>(dy + src), cb, stage_full_addr + (bslot_2) * 8);
                    }
                }
            } else {
                mbarrier_arrive(stage_full_addr + (bslot_2) * 8);
            }
        }
        float wr_1[32];
        float w4_1[4];
        float br_1[8];
        int wsm_addr_1 = 0;
        float win_1[32];
        float pend_1[32];
        #pragma unroll
        for (int i_5 = 0; i_5 < 32; i_5++) {
            win_1[i_5] = 0.0f;
            pend_1[i_5] = 0.0f;
        }
        int lo_1 = 0;
        int hi_1 = num_docs - 1;
        int doc_start_1 = 0;
        int doc_end_1 = 0;
        if (active_1 != 0) {
            if ((flags & 1) != 0) {
                #pragma unroll
                for (int j_9 = 0; j_9 < 8; j_9++) {
                    {
                        uint2 _vld_42;
                        _vld_42 = *reinterpret_cast<const uint2*>(reinterpret_cast<__nv_bfloat16*>(w) + (c64_1 + (long long)j_9) * 4);
                        uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&wr_1[j_9 * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                            (&wr_1[j_9 * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int j_10 = 0; j_10 < 8; j_10++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(w) + (c64_1 + (long long)j_10) * 4);
                        wr_1[j_10 * 4 + 0] = _v4.x;
                        wr_1[j_10 * 4 + 1] = _v4.y;
                        wr_1[j_10 * 4 + 2] = _v4.z;
                        wr_1[j_10 * 4 + 3] = _v4.w;
                    }
                }
            }
            if ((flags & 4) != 0) {
                {
                    const uint4* _vptr_44 = reinterpret_cast<const uint4*>(reinterpret_cast<__nv_bfloat16*>(b) + c64_1);
                    uint4 _vld_44[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_44[_blk] = _vptr_44[_blk];
                        uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&br_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
                            (&br_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(b) + c64_1);
                    br_1[0 + 0] = _v4.x;
                    br_1[0 + 1] = _v4.y;
                    br_1[0 + 2] = _v4.z;
                    br_1[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(b) + c64_1 + 4);
                    br_1[4 + 0] = _v4.x;
                    br_1[4 + 1] = _v4.y;
                    br_1[4 + 2] = _v4.z;
                    br_1[4 + 3] = _v4.w;
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
            #pragma unroll
            for (int kh_1 = 0; kh_1 < 3; kh_1++) {
                int tk_1 = t0 - 3 + kh_1;
                if (tk_1 >= doc_start_1) {
                    long long rowk_1 = (long long)tk_1 * C64 + c64_1;
                    {
                        const uint4* _vptr_47 = reinterpret_cast<const uint4*>(x + rowk_1);
                        uint4 _vld_47[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_47[_blk] = _vptr_47[_blk];
                            uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[(kh_1 + 1) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
                                (&win_1[(kh_1 + 1) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
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
                int _max_48 = ((my_rows_1 - k_1 * 8) > (0) ? (my_rows_1 - k_1 * 8) : (0));
                int rows_k_4 = ((_max_48) < (8) ? (_max_48) : (8));
                int _max_49 = ((my_steps_1 - k_1 * 8) > (0) ? (my_steps_1 - k_1 * 8) : (0));
                int steps_k_1 = ((_max_49) < (8) ? (_max_49) : (8));
                int sx0_1 = (int)stage_buf_addr + bslot_3 * 8192 + seg_off + li * 16;
                int interior_1 = 0;
                if (steps_k_1 == 8 && rows_k_4 == 8 && t_end_1 >= tb_1 + 8 && doc_end_1 >= tb_1 + 8) {
                    if (k_1 >= 1) {
                        if (doc_start_1 <= tb_1 - 3) {
                            interior_1 = 2;
                        }
                    }
                }
                if (interior_1 == 2) {
                    #pragma unroll
                    for (int u_3 = 0; u_3 < 8; u_3++) {
                        __int128_t _smem_raw_50;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_50) : "r"(sx0_1 + u_3 * rstride));
                        pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_50)[0];
                        pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_50)[1];
                        pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_50)[2];
                        pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_50)[3];
                        __int128_t _smem_raw_51;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_51) : "r"(sx0_1 + 4096 + u_3 * rstride));
                        pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_51)[0];
                        pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_51)[1];
                        pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_51)[2];
                        pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_51)[3];
                        #pragma unroll
                        for (int kk_10 = 0; kk_10 < 4; kk_10++) {
                            win_1[u_3 % 4 * 8 + 2 * kk_10] = __uint_as_float(pkx_1[kk_10] << 16);
                            win_1[u_3 % 4 * 8 + 2 * kk_10 + 1] = __uint_as_float(pkx_1[kk_10] & 4294901760u);
                            dyf_1[2 * kk_10] = __uint_as_float(pkd_1[kk_10] << 16);
                            dyf_1[2 * kk_10 + 1] = __uint_as_float(pkd_1[kk_10] & 4294901760u);
                        }
                        #pragma unroll
                        for (int j_11 = 0; j_11 < 8; j_11++) {
                            #pragma unroll
                            for (int kk_11 = 0; kk_11 < 4; kk_11++) {
                                w4_1[kk_11] = wr_1[j_11 * 4 + kk_11];
                            }
                            float _fma_54;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(w4_1[1]), "f"(win_1[(u_3 % 4 + 2) % 4 * 8 + j_11]), "f"(mul_f32_rn_noftz(w4_1[0], win_1[(u_3 % 4 + 1) % 4 * 8 + j_11])));
                            float _fma_53;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(w4_1[2]), "f"(win_1[(u_3 % 4 + 3) % 4 * 8 + j_11]), "f"(_fma_54));
                            float _fma_52;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(w4_1[3]), "f"(win_1[u_3 % 4 * 8 + j_11]), "f"(_fma_53));
                            float p_4;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(p_4) : "f"(_fma_52), "f"(br_1[j_11]));
                            float e_4 = (-p_4) * 1.442695f;
                            float _exp2_55 = approx_exp2(e_4);
                            float den_4 = 1.0f + _exp2_55;
                            float sg_4 = approx_rcp(den_4);
                            float g_4 = sg_4 * __fmaf_rn(p_4, 1.0f - sg_4, 1.0f);
                            dp_1[j_11] = dyf_1[j_11] * g_4;
                            #pragma unroll
                            for (int kk_12 = 0; kk_12 < 4; kk_12++) {
                                acc[j_11 * 4 + kk_12] = __fmaf_rn(dp_1[j_11], win_1[(u_3 % 4 + 1 + kk_12) % 4 * 8 + j_11], acc[j_11 * 4 + kk_12]);
                            }
                            accb[j_11] = accb[j_11] + dp_1[j_11];
                            pend_1[(u_3 % 4 + 1) % 4 * 8 + j_11] = __fmaf_rn(w4_1[0], dp_1[j_11], pend_1[(u_3 % 4 + 1) % 4 * 8 + j_11]);
                            pend_1[(u_3 % 4 + 2) % 4 * 8 + j_11] = __fmaf_rn(w4_1[1], dp_1[j_11], pend_1[(u_3 % 4 + 2) % 4 * 8 + j_11]);
                            pend_1[(u_3 % 4 + 3) % 4 * 8 + j_11] = __fmaf_rn(w4_1[2], dp_1[j_11], pend_1[(u_3 % 4 + 3) % 4 * 8 + j_11]);
                            pend_1[u_3 % 4 * 8 + j_11] = w4_1[3] * dp_1[j_11];
                        }
                        if (u_3 >= 0) {
                            long long row3_3 = (long long)(tb_1 + (u_3 - 3)) * C64 + c64_1;
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
                } else if (interior_1 == 1) {
                    #pragma unroll
                    for (int u_4 = 0; u_4 < 8; u_4++) {
                        __int128_t _smem_raw_56;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_56) : "r"(sx0_1 + u_4 * rstride));
                        pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_56)[0];
                        pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_56)[1];
                        pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_56)[2];
                        pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_56)[3];
                        __int128_t _smem_raw_57;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_57) : "r"(sx0_1 + 4096 + u_4 * rstride));
                        pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_57)[0];
                        pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_57)[1];
                        pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_57)[2];
                        pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_57)[3];
                        #pragma unroll
                        for (int kk_13 = 0; kk_13 < 4; kk_13++) {
                            win_1[u_4 % 4 * 8 + 2 * kk_13] = __uint_as_float(pkx_1[kk_13] << 16);
                            win_1[u_4 % 4 * 8 + 2 * kk_13 + 1] = __uint_as_float(pkx_1[kk_13] & 4294901760u);
                            dyf_1[2 * kk_13] = __uint_as_float(pkd_1[kk_13] << 16);
                            dyf_1[2 * kk_13 + 1] = __uint_as_float(pkd_1[kk_13] & 4294901760u);
                        }
                        #pragma unroll
                        for (int j_12 = 0; j_12 < 8; j_12++) {
                            #pragma unroll
                            for (int kk_14 = 0; kk_14 < 4; kk_14++) {
                                w4_1[kk_14] = wr_1[j_12 * 4 + kk_14];
                            }
                            float _fma_60;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(w4_1[1]), "f"(win_1[(u_4 % 4 + 2) % 4 * 8 + j_12]), "f"(mul_f32_rn_noftz(w4_1[0], win_1[(u_4 % 4 + 1) % 4 * 8 + j_12])));
                            float _fma_59;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(w4_1[2]), "f"(win_1[(u_4 % 4 + 3) % 4 * 8 + j_12]), "f"(_fma_60));
                            float _fma_58;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(w4_1[3]), "f"(win_1[u_4 % 4 * 8 + j_12]), "f"(_fma_59));
                            float p_5;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(p_5) : "f"(_fma_58), "f"(br_1[j_12]));
                            float e_5 = (-p_5) * 1.442695f;
                            float _exp2_61 = approx_exp2(e_5);
                            float den_5 = 1.0f + _exp2_61;
                            float sg_5 = approx_rcp(den_5);
                            float g_5 = sg_5 * __fmaf_rn(p_5, 1.0f - sg_5, 1.0f);
                            dp_1[j_12] = dyf_1[j_12] * g_5;
                            #pragma unroll
                            for (int kk_15 = 0; kk_15 < 4; kk_15++) {
                                acc[j_12 * 4 + kk_15] = __fmaf_rn(dp_1[j_12], win_1[(u_4 % 4 + 1 + kk_15) % 4 * 8 + j_12], acc[j_12 * 4 + kk_15]);
                            }
                            accb[j_12] = accb[j_12] + dp_1[j_12];
                            pend_1[(u_4 % 4 + 1) % 4 * 8 + j_12] = __fmaf_rn(w4_1[0], dp_1[j_12], pend_1[(u_4 % 4 + 1) % 4 * 8 + j_12]);
                            pend_1[(u_4 % 4 + 2) % 4 * 8 + j_12] = __fmaf_rn(w4_1[1], dp_1[j_12], pend_1[(u_4 % 4 + 2) % 4 * 8 + j_12]);
                            pend_1[(u_4 % 4 + 3) % 4 * 8 + j_12] = __fmaf_rn(w4_1[2], dp_1[j_12], pend_1[(u_4 % 4 + 3) % 4 * 8 + j_12]);
                            pend_1[u_4 % 4 * 8 + j_12] = w4_1[3] * dp_1[j_12];
                        }
                        if (u_4 >= 3) {
                            long long row3_4 = (long long)(tb_1 + (u_4 - 3)) * C64 + c64_1;
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(pend_1[(u_4 % 4 + 1) % 4 * 8 + 0], pend_1[(u_4 % 4 + 1) % 4 * 8 + 1]);
                                _pk[1] = __floats2bfloat162_rn(pend_1[(u_4 % 4 + 1) % 4 * 8 + 2], pend_1[(u_4 % 4 + 1) % 4 * 8 + 3]);
                                _pk[2] = __floats2bfloat162_rn(pend_1[(u_4 % 4 + 1) % 4 * 8 + 4], pend_1[(u_4 % 4 + 1) % 4 * 8 + 5]);
                                _pk[3] = __floats2bfloat162_rn(pend_1[(u_4 % 4 + 1) % 4 * 8 + 6], pend_1[(u_4 % 4 + 1) % 4 * 8 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[row3_4 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int u_5 = 0; u_5 < 8; u_5++) {
                        if (steps_k_1 > u_5) {
                            int tcur_1 = tb_1 + u_5;
                            if (rows_k_4 > u_5) {
                                __int128_t _smem_raw_62;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_62) : "r"(sx0_1 + u_5 * rstride));
                                pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[0];
                                pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[1];
                                pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[2];
                                pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[3];
                                __int128_t _smem_raw_63;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_63) : "r"(sx0_1 + 4096 + u_5 * rstride));
                                pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[0];
                                pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[1];
                                pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[2];
                                pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[3];
                                #pragma unroll
                                for (int kk_16 = 0; kk_16 < 4; kk_16++) {
                                    win_1[u_5 % 4 * 8 + 2 * kk_16] = __uint_as_float(pkx_1[kk_16] << 16);
                                    win_1[u_5 % 4 * 8 + 2 * kk_16 + 1] = __uint_as_float(pkx_1[kk_16] & 4294901760u);
                                    dyf_1[2 * kk_16] = __uint_as_float(pkd_1[kk_16] << 16);
                                    dyf_1[2 * kk_16 + 1] = __uint_as_float(pkd_1[kk_16] & 4294901760u);
                                }
                            }
                            if (tcur_1 >= doc_end_1) {
                                #pragma unroll
                                for (int m_2 = 1; m_2 < 4; m_2++) {
                                    int tm_1 = tcur_1 - m_2;
                                    if (tm_1 >= t0 && tm_1 < t_end_1 && tm_1 >= doc_start_1) {
                                        long long rowm_1 = (long long)tm_1 * C64 + c64_1;
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(pend_1[(u_5 % 4 + 4 - m_2) % 4 * 8 + 0], pend_1[(u_5 % 4 + 4 - m_2) % 4 * 8 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(pend_1[(u_5 % 4 + 4 - m_2) % 4 * 8 + 2], pend_1[(u_5 % 4 + 4 - m_2) % 4 * 8 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(pend_1[(u_5 % 4 + 4 - m_2) % 4 * 8 + 4], pend_1[(u_5 % 4 + 4 - m_2) % 4 * 8 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(pend_1[(u_5 % 4 + 4 - m_2) % 4 * 8 + 6], pend_1[(u_5 % 4 + 4 - m_2) % 4 * 8 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[rowm_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
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
                                } else {
                                    doc_start_1 = tcur_1 + 8;
                                    doc_end_1 = tcur_1 + 8;
                                }
                                #pragma unroll
                                for (int m_3 = 1; m_3 < 4; m_3++) {
                                    #pragma unroll
                                    for (int j_13 = 0; j_13 < 8; j_13++) {
                                        win_1[(u_5 % 4 + 4 - m_3) % 4 * 8 + j_13] = 0.0f;
                                    }
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
                                        float _fma_66;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(w4_1[1]), "f"(win_1[(u_5 % 4 + 2) % 4 * 8 + j_14]), "f"(mul_f32_rn_noftz(w4_1[0], win_1[(u_5 % 4 + 1) % 4 * 8 + j_14])));
                                        float _fma_65;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(w4_1[2]), "f"(win_1[(u_5 % 4 + 3) % 4 * 8 + j_14]), "f"(_fma_66));
                                        float _fma_64;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(w4_1[3]), "f"(win_1[u_5 % 4 * 8 + j_14]), "f"(_fma_65));
                                        float p_6;
                                        asm("add.rn.f32 %0, %1, %2;" : "=f"(p_6) : "f"(_fma_64), "f"(br_1[j_14]));
                                        float e_6 = (-p_6) * 1.442695f;
                                        float _exp2_67 = approx_exp2(e_6);
                                        float den_6 = 1.0f + _exp2_67;
                                        float sg_6 = approx_rcp(den_6);
                                        float g_6 = sg_6 * __fmaf_rn(p_6, 1.0f - sg_6, 1.0f);
                                        dp_1[j_14] = dyf_1[j_14] * g_6;
                                        #pragma unroll
                                        for (int kk_18 = 0; kk_18 < 4; kk_18++) {
                                            acc[j_14 * 4 + kk_18] = __fmaf_rn(dp_1[j_14], win_1[(u_5 % 4 + 1 + kk_18) % 4 * 8 + j_14], acc[j_14 * 4 + kk_18]);
                                        }
                                        accb[j_14] = accb[j_14] + dp_1[j_14];
                                        pend_1[(u_5 % 4 + 1) % 4 * 8 + j_14] = __fmaf_rn(w4_1[0], dp_1[j_14], pend_1[(u_5 % 4 + 1) % 4 * 8 + j_14]);
                                        pend_1[(u_5 % 4 + 2) % 4 * 8 + j_14] = __fmaf_rn(w4_1[1], dp_1[j_14], pend_1[(u_5 % 4 + 2) % 4 * 8 + j_14]);
                                        pend_1[(u_5 % 4 + 3) % 4 * 8 + j_14] = __fmaf_rn(w4_1[2], dp_1[j_14], pend_1[(u_5 % 4 + 3) % 4 * 8 + j_14]);
                                        pend_1[u_5 % 4 * 8 + j_14] = w4_1[3] * dp_1[j_14];
                                    }
                                } else {
                                    #pragma unroll
                                    for (int j_15 = 0; j_15 < 8; j_15++) {
                                        #pragma unroll
                                        for (int kk_19 = 0; kk_19 < 4; kk_19++) {
                                            w4_1[kk_19] = wr_1[j_15 * 4 + kk_19];
                                        }
                                        float _fma_70;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(w4_1[1]), "f"(win_1[(u_5 % 4 + 2) % 4 * 8 + j_15]), "f"(mul_f32_rn_noftz(w4_1[0], win_1[(u_5 % 4 + 1) % 4 * 8 + j_15])));
                                        float _fma_69;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(w4_1[2]), "f"(win_1[(u_5 % 4 + 3) % 4 * 8 + j_15]), "f"(_fma_70));
                                        float _fma_68;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(w4_1[3]), "f"(win_1[u_5 % 4 * 8 + j_15]), "f"(_fma_69));
                                        float p_7;
                                        asm("add.rn.f32 %0, %1, %2;" : "=f"(p_7) : "f"(_fma_68), "f"(br_1[j_15]));
                                        float e_7 = (-p_7) * 1.442695f;
                                        float _exp2_71 = approx_exp2(e_7);
                                        float den_7 = 1.0f + _exp2_71;
                                        float sg_7 = approx_rcp(den_7);
                                        float g_7 = sg_7 * __fmaf_rn(p_7, 1.0f - sg_7, 1.0f);
                                        dp_1[j_15] = dyf_1[j_15] * g_7;
                                        pend_1[(u_5 % 4 + 1) % 4 * 8 + j_15] = __fmaf_rn(w4_1[0], dp_1[j_15], pend_1[(u_5 % 4 + 1) % 4 * 8 + j_15]);
                                        pend_1[(u_5 % 4 + 2) % 4 * 8 + j_15] = __fmaf_rn(w4_1[1], dp_1[j_15], pend_1[(u_5 % 4 + 2) % 4 * 8 + j_15]);
                                        pend_1[(u_5 % 4 + 3) % 4 * 8 + j_15] = __fmaf_rn(w4_1[2], dp_1[j_15], pend_1[(u_5 % 4 + 3) % 4 * 8 + j_15]);
                                        pend_1[u_5 % 4 * 8 + j_15] = w4_1[3] * dp_1[j_15];
                                    }
                                }
                                int t3_1 = tcur_1 - 3;
                                if (t3_1 >= t0 && t3_1 >= doc_start_1) {
                                    long long row3_5 = (long long)t3_1 * C64 + c64_1;
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(pend_1[(u_5 % 4 + 1) % 4 * 8 + 0], pend_1[(u_5 % 4 + 1) % 4 * 8 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(pend_1[(u_5 % 4 + 1) % 4 * 8 + 2], pend_1[(u_5 % 4 + 1) % 4 * 8 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(pend_1[(u_5 % 4 + 1) % 4 * 8 + 4], pend_1[(u_5 % 4 + 1) % 4 * 8 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(pend_1[(u_5 % 4 + 1) % 4 * 8 + 6], pend_1[(u_5 % 4 + 1) % 4 * 8 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[row3_5 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if ((k_1 + 1) % 4 == 0) {
                int base_1 = tid * 40;
                #pragma unroll
                for (int i_6 = 0; i_6 < 32; i_6++) {
                    float s_3 = red[base_1 + i_6];
                    float y_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_2) : "f"(acc[i_6]), "f"(redc[base_1 + i_6]));
                    float t_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_2) : "f"(s_3), "f"(y_2));
                    float _sub_73;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(t_2), "f"(s_3));
                    float _sub_72;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(_sub_73), "f"(y_2));
                    redc[base_1 + i_6] = _sub_72;
                    red[base_1 + i_6] = t_2;
                    acc[i_6] = 0.0f;
                }
                #pragma unroll
                for (int j_16 = 0; j_16 < 8; j_16++) {
                    float s_4 = red[base_1 + 32 + j_16];
                    float y_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_3) : "f"(accb[j_16]), "f"(redc[base_1 + 32 + j_16]));
                    float t_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_3) : "f"(s_4), "f"(y_3));
                    float _sub_75;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(t_3), "f"(s_4));
                    float _sub_74;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(_sub_75), "f"(y_3));
                    redc[base_1 + 32 + j_16] = _sub_74;
                    red[base_1 + 32 + j_16] = t_3;
                    accb[j_16] = 0.0f;
                }
            }
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn_1 = k_1 + 2;
            if (kn_1 < n_stages_1) {
                int bslot_0_1 = warp_1 * 2 + slot_1;
                int rows_k_5 = 0;
                if (active_1 != 0) {
                    int _max_77 = ((my_rows_1 - kn_1 * 8) > (0) ? (my_rows_1 - kn_1 * 8) : (0));
                    int _min_76 = ((_max_77) < (8) ? (_max_77) : (8));
                    rows_k_5 = _min_76;
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
                        for (int i_7 = 0; i_7 < rows_k_5; i_7++) {
                            long long src_1 = src0_3 + (long long)i_7 * C64;
                            cp_async_bulk_gmem2smem(sx_3 + i_7 * rstride, reinterpret_cast<const uint8_t*>(x + src_1), cb_1, stage_full_addr + (bslot_0_1) * 8);
                            cp_async_bulk_gmem2smem(sx_3 + 4096 + i_7 * rstride, reinterpret_cast<const uint8_t*>(dy + src_1), cb_1, stage_full_addr + (bslot_0_1) * 8);
                        }
                    }
                } else {
                    mbarrier_arrive(stage_full_addr + (bslot_0_1) * 8);
                }
            }
        }
    }
    int base_2 = tid * 40;
    #pragma unroll
    for (int i_8 = 0; i_8 < 32; i_8++) {
        float s_5 = red[base_2 + i_8];
        float y_4;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_4) : "f"(acc[i_8]), "f"(redc[base_2 + i_8]));
        float t_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(t_4) : "f"(s_5), "f"(y_4));
        float _sub_79;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(t_4), "f"(s_5));
        float _sub_78;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(_sub_79), "f"(y_4));
        redc[base_2 + i_8] = _sub_78;
        red[base_2 + i_8] = t_4;
        acc[i_8] = 0.0f;
    }
    #pragma unroll
    for (int j_17 = 0; j_17 < 8; j_17++) {
        float s_6 = red[base_2 + 32 + j_17];
        float y_5;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_5) : "f"(accb[j_17]), "f"(redc[base_2 + 32 + j_17]));
        float t_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(t_5) : "f"(s_6), "f"(y_5));
        float _sub_81;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(t_5), "f"(s_6));
        float _sub_80;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(_sub_81), "f"(y_5));
        redc[base_2 + 32 + j_17] = _sub_80;
        red[base_2 + 32 + j_17] = t_5;
        accb[j_17] = 0.0f;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 40; i_9++) {
        float _sub_82;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(red[tid * 40 + i_9]), "f"(redc[tid * 40 + i_9]));
        red[tid * 40 + i_9] = _sub_82;
    }
    __syncthreads();
    int c0_0 = (ct * ct_vec + v) * 8;
    if (sub == 0 && c0_0 < C) {
        float tot[32];
        #pragma unroll
        for (int i_10 = 0; i_10 < 32; i_10++) {
            tot[i_10] = red[v * 40 + i_10];
        }
        #pragma unroll 1
        for (int s_7 = 1; s_7 < n_sub; s_7++) {
            #pragma unroll
            for (int i_11 = 0; i_11 < 32; i_11++) {
                tot[i_11] = tot[i_11] + red[(s_7 * ct_vec + v) * 40 + i_11];
            }
        }
        #pragma unroll
        for (int j_18 = 0; j_18 < 8; j_18++) {
            long long pidx = ((long long)rank * C64 + (long long)c0_0 + (long long)j_18) * 4;
            {
                float4 _v4 = make_float4(tot[j_18 * 4 + 0], tot[j_18 * 4 + 1], tot[j_18 * 4 + 2], tot[j_18 * 4 + 3]);
                *reinterpret_cast<float4*>(partial + pidx) = _v4;
            }
        }
        float totb[8];
        #pragma unroll
        for (int j_19 = 0; j_19 < 8; j_19++) {
            totb[j_19] = red[v * 40 + 32 + j_19];
        }
        #pragma unroll 1
        for (int s_8 = 1; s_8 < n_sub; s_8++) {
            #pragma unroll
            for (int j_20 = 0; j_20 < 8; j_20++) {
                totb[j_20] = totb[j_20] + red[(s_8 * ct_vec + v) * 40 + 32 + j_20];
            }
        }
        long long bidx = (long long)((num_bids + n_ch_tiles - 1) / n_ch_tiles) * C64 * 4 + (long long)rank * C64 + (long long)c0_0;
        {
            float4 _v4 = make_float4(totb[0 + 0], totb[0 + 1], totb[0 + 2], totb[0 + 3]);
            *reinterpret_cast<float4*>(partial + bidx) = _v4;
        }
        {
            float4 _v4 = make_float4(totb[4 + 0], totb[4 + 1], totb[4 + 2], totb[4 + 3]);
            *reinterpret_cast<float4*>(partial + bidx + 4) = _v4;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Cleanup
    __syncthreads();
}

} // extern "C"
