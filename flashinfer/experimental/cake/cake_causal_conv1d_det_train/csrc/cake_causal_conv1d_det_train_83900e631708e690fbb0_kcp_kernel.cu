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

#define LAUNCH_MIN_BLOCKS 3

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

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_det_train_83900e631708e690fbb0_kcp(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, __nv_bfloat16* __restrict__ hist, int* __restrict__ hist_row_off, float* __restrict__ dpre_tail, int* __restrict__ tail_row_off, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap dy_map)
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
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int stage_buf_addr = smem + 17408;

    // Mbarrier init (8 barriers at smem_raw[0..64))

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
    int grouped = 0;
    if ((ct_vec & 31) == 0 && ct_vec > 32 && ct_vec * 8 == C) {
        grouped = 1;
    }
    if (grouped != 0) {
        #pragma unroll 1
        for (int s = 0; s < 2; s++) {
            if (sub == s) {
                int RS = ct_vec * 16;
                long long c64 = (long long)c0;
                int t_end = ((t0 + L) < (S) ? (t0 + L) : (S));
                int t_stop = ((t_end + 3) < (S + 1) ? (t_end + 3) : (S + 1));
                int t_lim = ((t_stop) < (S) ? (t_stop) : (S));
                int active = 0;
                if (s < n_sub && c0 < C && t0 < S) {
                    active = 1;
                }
                int my_rows = 0;
                int my_steps = 0;
                if (active != 0) {
                    my_rows = t_lim - t0;
                    my_steps = t_stop - t0;
                }
                int n_stages = (my_steps + 4 - 1) / 4;
                int fill = s * 2;
                int _min_0 = ((n_stages) < (2) ? (n_stages) : (2));
                #pragma unroll 1
                for (int k0 = 0; k0 < _min_0; k0++) {
                    int slot0 = k0;
                    int GWn = ct_vec >> 5;
                    int bslot = s * 2 + slot0;
                    int rows_k = 0;
                    if (active != 0) {
                        int _max_1 = ((my_rows - k0 * 4) > (0) ? (my_rows - k0 * 4) : (0));
                        int _min_0_1 = ((_max_1) < (4) ? (_max_1) : (4));
                        rows_k = _min_0_1;
                    }
                    if (v == 0 && rows_k > 0) {
                        mbarrier_arrive_expect_tx(stage_full_addr + (fill) * 8, rows_k * (2 * RS));
                        int sx = (int)stage_buf_addr + (s * (GWn * 2) + slot0 * GWn) * 2048;
                        long long src0 = (long long)(t0 + k0 * 4) * C64 + c64;
                        cp_async_bulk_gmem2smem(sx, reinterpret_cast<const uint8_t*>(x + src0), rows_k * RS, stage_full_addr + (fill) * 8);
                        cp_async_bulk_gmem2smem(sx + 16384, reinterpret_cast<const uint8_t*>(dy + src0), rows_k * RS, stage_full_addr + (fill) * 8);
                    } else if (v < 32) {
                        mbarrier_arrive(stage_full_addr + (fill) * 8);
                    }
                    fill += 1;
                    if (fill == (s * 2 + 2)) { fill = s * 2; }
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
                    frag = lo;
                    int tk_2_0 = t0 - 1;
                    if (tk_2_0 >= doc_start) {
                        long long rowk_1_0 = (long long)tk_2_0 * C64 + c64;
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(x + rowk_1_0);
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
                        int hrow_1_0 = hist_row_off[frag * 3 + 2 - (doc_start - t0)];
                        if (hrow_1_0 >= 0) {
                            long long rowh_1_0 = (long long)hrow_1_0 * C64 + c64;
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(hist + rowh_1_0);
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
                    int tk_2_1 = t0 - 2;
                    if (tk_2_1 >= doc_start) {
                        long long rowk_1_1 = (long long)tk_2_1 * C64 + c64;
                        {
                            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + rowk_1_1);
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
                        int hrow_1_1 = hist_row_off[frag * 3 + 1 - (doc_start - t0)];
                        if (hrow_1_1 >= 0) {
                            long long rowh_1_1 = (long long)hrow_1_1 * C64 + c64;
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(hist + rowh_1_1);
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
                    int tk_2_2 = t0 - 3;
                    if (tk_2_2 >= doc_start) {
                        long long rowk_1_2 = (long long)tk_2_2 * C64 + c64;
                        {
                            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + rowk_1_2);
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
                        int hrow_1_2 = hist_row_off[frag * 3 - (doc_start - t0)];
                        if (hrow_1_2 >= 0) {
                            long long rowh_1_2 = (long long)hrow_1_2 * C64 + c64;
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(hist + rowh_1_2);
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
                int sbars_cursor = s * 2;
                unsigned int _phase_stage_full = 0;
                #pragma unroll 1
                for (int k = 0; k < n_stages; k++) {
                    int slot = k % 2;
                    int bslot_1 = s * 2 + slot;
                    mbarrier_wait(stage_full_addr + (sbars_cursor) * 8, _phase_stage_full);
                    sbars_cursor += 1;
                    if (sbars_cursor == (s * 2 + 2)) { sbars_cursor = s * 2; _phase_stage_full ^= 1; }
                    if (active != 0) {
                        int tb = t0 + k * 4;
                        int _max_9 = ((my_rows - k * 4) > (0) ? (my_rows - k * 4) : (0));
                        int rows_k_1 = ((_max_9) < (4) ? (_max_9) : (4));
                        int _max_10 = ((my_steps - k * 4) > (0) ? (my_steps - k * 4) : (0));
                        int steps_k = ((_max_10) < (4) ? (_max_10) : (4));
                        int GWn_1 = ct_vec >> 5;
                        int sx0 = (int)stage_buf_addr + (s * (GWn_1 * 2) + slot * GWn_1) * 2048 + v * 16;
                        int interior = 0;
                        if (steps_k == 4 && rows_k_1 == 4 && t_end >= tb + 4 && doc_end >= tb + 4) {
                            if (k >= 1) {
                                if (doc_start <= tb - 3) {
                                    interior = 1;
                                }
                            }
                        }
                        if (interior != 0) {
                            #pragma unroll
                            for (int u = 0; u < 4; u++) {
                                __int128_t _smem_raw_11;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(sx0 + u * RS));
                                pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                                pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                                pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                                pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
                                __int128_t _smem_raw_12;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_12) : "r"(sx0 + 16384 + u * RS));
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
                                for (int h = 0; h < 4; h++) {
                                    float2 p2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h + 3], wr[8 * h + 4 + 3]), make_float2(win[u % 4 * 8 + 2 * h], win[u % 4 * 8 + 2 * h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h + 2], wr[8 * h + 4 + 2]), make_float2(win[(u % 4 + 3) % 4 * 8 + 2 * h], win[(u % 4 + 3) % 4 * 8 + 2 * h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h + 1], wr[8 * h + 4 + 1]), make_float2(win[(u % 4 + 2) % 4 * 8 + 2 * h], win[(u % 4 + 2) % 4 * 8 + 2 * h + 1]), mul_f32x2_noftz(make_float2(wr[8 * h], wr[8 * h + 4]), make_float2(win[(u % 4 + 1) % 4 * 8 + 2 * h], win[(u % 4 + 1) % 4 * 8 + 2 * h + 1])))));
                                    float2 _f32x2_mul_b_13 = make_float2(-1.442695f, -1.442695f);
                                    float2 e2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2) : "l"(*(const unsigned long long*)&p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
                                    float _exp2_14 = approx_exp2(e2.x);
                                    float _exp2_15 = approx_exp2(e2.y);
                                    float2 den2 = add_f32x2_noftz(make_float2(_exp2_14, _exp2_15), make_float2(1.0f, 1.0f));
                                    float _rcp_16 = approx_rcp(den2.x);
                                    float _rcp_17 = approx_rcp(den2.y);
                                    float2 sg2 = make_float2(_rcp_16, _rcp_17);
                                    float2 _f32x2_mul_b_18 = fma_f32x2_rn_noftz(p2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2), make_float2(1.0f, 1.0f));
                                    float2 g2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2) : "l"(*(const unsigned long long*)&sg2), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
                                    float2 _f32x2_mul_a_19 = make_float2(dyf[2 * h], dyf[2 * h + 1]);
                                    float2 d2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&g2));
                                    dp[2 * h] = d2.x;
                                    dp[2 * h + 1] = d2.y;
                                    #pragma unroll
                                    for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                                        float2 a2 = fma_f32x2_rn_noftz(d2, make_float2(win[(u % 4 + 1 + kk_1) % 4 * 8 + 2 * h], win[(u % 4 + 1 + kk_1) % 4 * 8 + 2 * h + 1]), make_float2(acc[8 * h + kk_1], acc[8 * h + 4 + kk_1]));
                                        acc[8 * h + kk_1] = a2.x;
                                        acc[8 * h + 4 + kk_1] = a2.y;
                                    }
                                    #pragma unroll
                                    for (int k_0 = 0; k_0 < 3; k_0++) {
                                        float2 q2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h + k_0], wr[8 * h + 4 + k_0]), d2, make_float2(pend[(u % 4 + 1 + k_0) % 4 * 8 + 2 * h], pend[(u % 4 + 1 + k_0) % 4 * 8 + 2 * h + 1]));
                                        pend[(u % 4 + 1 + k_0) % 4 * 8 + 2 * h] = q2.x;
                                        pend[(u % 4 + 1 + k_0) % 4 * 8 + 2 * h + 1] = q2.y;
                                    }
                                    float2 _f32x2_mul_a_20 = make_float2(wr[8 * h + 3], wr[8 * h + 4 + 3]);
                                    float2 r2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&d2));
                                    pend[u % 4 * 8 + 2 * h] = r2.x;
                                    pend[u % 4 * 8 + 2 * h + 1] = r2.y;
                                }
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
                        } else {
                            #pragma unroll
                            for (int u_1 = 0; u_1 < 4; u_1++) {
                                if (steps_k > u_1) {
                                    int tcur = tb + u_1;
                                    if (rows_k_1 > u_1) {
                                        __int128_t _smem_raw_21;
                                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_21) : "r"(sx0 + u_1 * RS));
                                        pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[0];
                                        pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[1];
                                        pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[2];
                                        pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[3];
                                        __int128_t _smem_raw_22;
                                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_22) : "r"(sx0 + 16384 + u_1 * RS));
                                        pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[0];
                                        pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[1];
                                        pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[2];
                                        pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_22)[3];
                                        #pragma unroll
                                        for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                                            win[u_1 % 4 * 8 + 2 * kk_2] = __uint_as_float(pkx[kk_2] << 16);
                                            win[u_1 % 4 * 8 + 2 * kk_2 + 1] = __uint_as_float(pkx[kk_2] & 4294901760u);
                                            dyf[2 * kk_2] = __uint_as_float(pkd[kk_2] << 16);
                                            dyf[2 * kk_2 + 1] = __uint_as_float(pkd[kk_2] & 4294901760u);
                                        }
                                    }
                                    if (tcur >= doc_end) {
                                        int trow = tail_row_off[frag * 3];
                                        if (trow >= 0) {
                                            long long rowr = (long long)trow * C64 + c64;
                                            {
                                                unsigned _ldv8_23_0;
                                                unsigned _ldv8_23_1;
                                                unsigned _ldv8_23_2;
                                                unsigned _ldv8_23_3;
                                                unsigned _ldv8_23_4;
                                                unsigned _ldv8_23_5;
                                                unsigned _ldv8_23_6;
                                                unsigned _ldv8_23_7;
                                                asm volatile(
                                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                    : "=r"(_ldv8_23_0), "=r"(_ldv8_23_1), "=r"(_ldv8_23_2), "=r"(_ldv8_23_3), "=r"(_ldv8_23_4), "=r"(_ldv8_23_5), "=r"(_ldv8_23_6), "=r"(_ldv8_23_7) : "l"((const void*)(dpre_tail + (rowr))) : "memory");
                                                tailf[0 + 0] = __uint_as_float(_ldv8_23_0);
                                                tailf[0 + 1] = __uint_as_float(_ldv8_23_1);
                                                tailf[0 + 2] = __uint_as_float(_ldv8_23_2);
                                                tailf[0 + 3] = __uint_as_float(_ldv8_23_3);
                                                tailf[0 + 4] = __uint_as_float(_ldv8_23_4);
                                                tailf[0 + 5] = __uint_as_float(_ldv8_23_5);
                                                tailf[0 + 6] = __uint_as_float(_ldv8_23_6);
                                                tailf[0 + 7] = __uint_as_float(_ldv8_23_7);
                                            }
                                            #pragma unroll
                                            for (int j_1 = 0; j_1 < 8; j_1++) {
                                                #pragma unroll
                                                for (int kk_3 = 0; kk_3 < 4; kk_3++) {
                                                    w4[kk_3] = wr[j_1 * 4 + kk_3];
                                                }
                                                float _fma_24;
                                                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(w4[2]), "f"(tailf[j_1]), "f"(pend[(u_1 % 4 + 4 - 1) % 4 * 8 + j_1]));
                                                pend[(u_1 % 4 + 4 - 1) % 4 * 8 + j_1] = _fma_24;
                                                float _fma_25;
                                                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(w4[1]), "f"(tailf[j_1]), "f"(pend[(u_1 % 4 + 4 - 2) % 4 * 8 + j_1]));
                                                pend[(u_1 % 4 + 4 - 2) % 4 * 8 + j_1] = _fma_25;
                                                float _fma_26;
                                                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(w4[0]), "f"(tailf[j_1]), "f"(pend[(u_1 % 4 + 4 - 3) % 4 * 8 + j_1]));
                                                pend[(u_1 % 4 + 4 - 3) % 4 * 8 + j_1] = _fma_26;
                                            }
                                        }
                                        int trow_0 = tail_row_off[frag * 3 + 1];
                                        if (trow_0 >= 0) {
                                            long long rowr_1 = (long long)trow_0 * C64 + c64;
                                            {
                                                unsigned _ldv8_27_0;
                                                unsigned _ldv8_27_1;
                                                unsigned _ldv8_27_2;
                                                unsigned _ldv8_27_3;
                                                unsigned _ldv8_27_4;
                                                unsigned _ldv8_27_5;
                                                unsigned _ldv8_27_6;
                                                unsigned _ldv8_27_7;
                                                asm volatile(
                                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                    : "=r"(_ldv8_27_0), "=r"(_ldv8_27_1), "=r"(_ldv8_27_2), "=r"(_ldv8_27_3), "=r"(_ldv8_27_4), "=r"(_ldv8_27_5), "=r"(_ldv8_27_6), "=r"(_ldv8_27_7) : "l"((const void*)(dpre_tail + (rowr_1))) : "memory");
                                                tailf[0 + 0] = __uint_as_float(_ldv8_27_0);
                                                tailf[0 + 1] = __uint_as_float(_ldv8_27_1);
                                                tailf[0 + 2] = __uint_as_float(_ldv8_27_2);
                                                tailf[0 + 3] = __uint_as_float(_ldv8_27_3);
                                                tailf[0 + 4] = __uint_as_float(_ldv8_27_4);
                                                tailf[0 + 5] = __uint_as_float(_ldv8_27_5);
                                                tailf[0 + 6] = __uint_as_float(_ldv8_27_6);
                                                tailf[0 + 7] = __uint_as_float(_ldv8_27_7);
                                            }
                                            #pragma unroll
                                            for (int j_2 = 0; j_2 < 8; j_2++) {
                                                #pragma unroll
                                                for (int kk_4 = 0; kk_4 < 4; kk_4++) {
                                                    w4[kk_4] = wr[j_2 * 4 + kk_4];
                                                }
                                                float _fma_28;
                                                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(w4[1]), "f"(tailf[j_2]), "f"(pend[(u_1 % 4 + 4 + 1 - 2) % 4 * 8 + j_2]));
                                                pend[(u_1 % 4 + 4 + 1 - 2) % 4 * 8 + j_2] = _fma_28;
                                                float _fma_29;
                                                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(w4[0]), "f"(tailf[j_2]), "f"(pend[(u_1 % 4 + 4 + 1 - 3) % 4 * 8 + j_2]));
                                                pend[(u_1 % 4 + 4 + 1 - 3) % 4 * 8 + j_2] = _fma_29;
                                            }
                                        }
                                        int trow_1 = tail_row_off[frag * 3 + 2];
                                        if (trow_1 >= 0) {
                                            long long rowr_2 = (long long)trow_1 * C64 + c64;
                                            {
                                                unsigned _ldv8_30_0;
                                                unsigned _ldv8_30_1;
                                                unsigned _ldv8_30_2;
                                                unsigned _ldv8_30_3;
                                                unsigned _ldv8_30_4;
                                                unsigned _ldv8_30_5;
                                                unsigned _ldv8_30_6;
                                                unsigned _ldv8_30_7;
                                                asm volatile(
                                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                    : "=r"(_ldv8_30_0), "=r"(_ldv8_30_1), "=r"(_ldv8_30_2), "=r"(_ldv8_30_3), "=r"(_ldv8_30_4), "=r"(_ldv8_30_5), "=r"(_ldv8_30_6), "=r"(_ldv8_30_7) : "l"((const void*)(dpre_tail + (rowr_2))) : "memory");
                                                tailf[0 + 0] = __uint_as_float(_ldv8_30_0);
                                                tailf[0 + 1] = __uint_as_float(_ldv8_30_1);
                                                tailf[0 + 2] = __uint_as_float(_ldv8_30_2);
                                                tailf[0 + 3] = __uint_as_float(_ldv8_30_3);
                                                tailf[0 + 4] = __uint_as_float(_ldv8_30_4);
                                                tailf[0 + 5] = __uint_as_float(_ldv8_30_5);
                                                tailf[0 + 6] = __uint_as_float(_ldv8_30_6);
                                                tailf[0 + 7] = __uint_as_float(_ldv8_30_7);
                                            }
                                            #pragma unroll
                                            for (int j_3 = 0; j_3 < 8; j_3++) {
                                                #pragma unroll
                                                for (int kk_5 = 0; kk_5 < 4; kk_5++) {
                                                    w4[kk_5] = wr[j_3 * 4 + kk_5];
                                                }
                                                float _fma_31;
                                                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(w4[0]), "f"(tailf[j_3]), "f"(pend[(u_1 % 4 + 4 + 2 - 3) % 4 * 8 + j_3]));
                                                pend[(u_1 % 4 + 4 + 2 - 3) % 4 * 8 + j_3] = _fma_31;
                                            }
                                        }
                                        #pragma unroll
                                        for (int m = 1; m < 4; m++) {
                                            int tm = tcur - m;
                                            if (tm >= t0 && tm < t_end && tm >= doc_start) {
                                                long long rowm = (long long)tm * C64 + c64;
                                                unsigned int _packed_1[4];
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 1; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2 + 0], (pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 0]));
                                                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                                                }
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 1; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2 + 2], (pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 2]));
                                                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                                                }
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 1; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2 + 4], (pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 4]));
                                                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                                                }
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 1; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2 + 6], (pend + (u_1 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 6]));
                                                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                                                }
                                                reinterpret_cast<int4*>(dx + (rowm))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                                            }
                                        }
                                        #pragma unroll
                                        for (int m_1 = 1; m_1 < 4; m_1++) {
                                            #pragma unroll
                                            for (int j_4 = 0; j_4 < 8; j_4++) {
                                                win[(u_1 % 4 + 4 - m_1) % 4 * 8 + j_4] = 0.0f;
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
                                            int hrow_2_0 = hist_row_off[frag * 3 + 2];
                                            if (hrow_2_0 >= 0) {
                                                long long rowh_1_3 = (long long)hrow_2_0 * C64 + c64;
                                                {
                                                    const uint4* _vptr_32 = reinterpret_cast<const uint4*>(hist + rowh_1_3);
                                                    uint4 _vld_32[1];
                                                    #pragma unroll
                                                    for (int _blk = 0; _blk < 1; _blk++) {
                                                        _vld_32[_blk] = _vptr_32[_blk];
                                                        uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                                                        #pragma unroll
                                                        for (int _pair = 0; _pair < 4; _pair++) {
                                                            (&win[(u_1 % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                                                            (&win[(u_1 % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                                                        }
                                                    }
                                                }
                                            }
                                            int hrow_2_1 = hist_row_off[frag * 3 + 1];
                                            if (hrow_2_1 >= 0) {
                                                long long rowh_1_4 = (long long)hrow_2_1 * C64 + c64;
                                                {
                                                    const uint4* _vptr_33 = reinterpret_cast<const uint4*>(hist + rowh_1_4);
                                                    uint4 _vld_33[1];
                                                    #pragma unroll
                                                    for (int _blk = 0; _blk < 1; _blk++) {
                                                        _vld_33[_blk] = _vptr_33[_blk];
                                                        uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                                                        #pragma unroll
                                                        for (int _pair = 0; _pair < 4; _pair++) {
                                                            (&win[(u_1 % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                                                            (&win[(u_1 % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                                                        }
                                                    }
                                                }
                                            }
                                            int hrow_2_2 = hist_row_off[frag * 3];
                                            if (hrow_2_2 >= 0) {
                                                long long rowh_1_5 = (long long)hrow_2_2 * C64 + c64;
                                                {
                                                    const uint4* _vptr_34 = reinterpret_cast<const uint4*>(hist + rowh_1_5);
                                                    uint4 _vld_34[1];
                                                    #pragma unroll
                                                    for (int _blk = 0; _blk < 1; _blk++) {
                                                        _vld_34[_blk] = _vptr_34[_blk];
                                                        uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34[_blk]);
                                                        #pragma unroll
                                                        for (int _pair = 0; _pair < 4; _pair++) {
                                                            (&win[(u_1 % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                                                            (&win[(u_1 % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            doc_start = tcur + 4;
                                            doc_end = tcur + 4;
                                        }
                                    }
                                    if (tcur < S) {
                                        if (tcur < t_end) {
                                            #pragma unroll
                                            for (int h_1 = 0; h_1 < 4; h_1++) {
                                                float2 p2_1 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 3], wr[8 * h_1 + 4 + 3]), make_float2(win[u_1 % 4 * 8 + 2 * h_1], win[u_1 % 4 * 8 + 2 * h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 2], wr[8 * h_1 + 4 + 2]), make_float2(win[(u_1 % 4 + 3) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 3) % 4 * 8 + 2 * h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 1], wr[8 * h_1 + 4 + 1]), make_float2(win[(u_1 % 4 + 2) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 2) % 4 * 8 + 2 * h_1 + 1]), mul_f32x2_noftz(make_float2(wr[8 * h_1], wr[8 * h_1 + 4]), make_float2(win[(u_1 % 4 + 1) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 1) % 4 * 8 + 2 * h_1 + 1])))));
                                                float2 _f32x2_mul_b_35 = make_float2(-1.442695f, -1.442695f);
                                                float2 e2_1;
                                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_1) : "l"(*(const unsigned long long*)&p2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
                                                float _exp2_36 = approx_exp2(e2_1.x);
                                                float _exp2_37 = approx_exp2(e2_1.y);
                                                float2 den2_1 = add_f32x2_noftz(make_float2(_exp2_36, _exp2_37), make_float2(1.0f, 1.0f));
                                                float _rcp_38 = approx_rcp(den2_1.x);
                                                float _rcp_39 = approx_rcp(den2_1.y);
                                                float2 sg2_1 = make_float2(_rcp_38, _rcp_39);
                                                float2 _f32x2_mul_b_40 = fma_f32x2_rn_noftz(p2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2_1), make_float2(1.0f, 1.0f));
                                                float2 g2_1;
                                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_1) : "l"(*(const unsigned long long*)&sg2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
                                                float2 _f32x2_mul_a_41 = make_float2(dyf[2 * h_1], dyf[2 * h_1 + 1]);
                                                float2 d2_1;
                                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&g2_1));
                                                dp[2 * h_1] = d2_1.x;
                                                dp[2 * h_1 + 1] = d2_1.y;
                                                #pragma unroll
                                                for (int kk_6 = 0; kk_6 < 4; kk_6++) {
                                                    float2 a2_1 = fma_f32x2_rn_noftz(d2_1, make_float2(win[(u_1 % 4 + 1 + kk_6) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 1 + kk_6) % 4 * 8 + 2 * h_1 + 1]), make_float2(acc[8 * h_1 + kk_6], acc[8 * h_1 + 4 + kk_6]));
                                                    acc[8 * h_1 + kk_6] = a2_1.x;
                                                    acc[8 * h_1 + 4 + kk_6] = a2_1.y;
                                                }
                                                #pragma unroll
                                                for (int k_1 = 0; k_1 < 3; k_1++) {
                                                    float2 q2_1 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + k_1], wr[8 * h_1 + 4 + k_1]), d2_1, make_float2(pend[(u_1 % 4 + 1 + k_1) % 4 * 8 + 2 * h_1], pend[(u_1 % 4 + 1 + k_1) % 4 * 8 + 2 * h_1 + 1]));
                                                    pend[(u_1 % 4 + 1 + k_1) % 4 * 8 + 2 * h_1] = q2_1.x;
                                                    pend[(u_1 % 4 + 1 + k_1) % 4 * 8 + 2 * h_1 + 1] = q2_1.y;
                                                }
                                                float2 _f32x2_mul_a_42 = make_float2(wr[8 * h_1 + 3], wr[8 * h_1 + 4 + 3]);
                                                float2 r2_1;
                                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&d2_1));
                                                pend[u_1 % 4 * 8 + 2 * h_1] = r2_1.x;
                                                pend[u_1 % 4 * 8 + 2 * h_1 + 1] = r2_1.y;
                                            }
                                        } else {
                                            #pragma unroll
                                            for (int h_2 = 0; h_2 < 4; h_2++) {
                                                float2 p2_2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_2 + 3], wr[8 * h_2 + 4 + 3]), make_float2(win[u_1 % 4 * 8 + 2 * h_2], win[u_1 % 4 * 8 + 2 * h_2 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_2 + 2], wr[8 * h_2 + 4 + 2]), make_float2(win[(u_1 % 4 + 3) % 4 * 8 + 2 * h_2], win[(u_1 % 4 + 3) % 4 * 8 + 2 * h_2 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_2 + 1], wr[8 * h_2 + 4 + 1]), make_float2(win[(u_1 % 4 + 2) % 4 * 8 + 2 * h_2], win[(u_1 % 4 + 2) % 4 * 8 + 2 * h_2 + 1]), mul_f32x2_noftz(make_float2(wr[8 * h_2], wr[8 * h_2 + 4]), make_float2(win[(u_1 % 4 + 1) % 4 * 8 + 2 * h_2], win[(u_1 % 4 + 1) % 4 * 8 + 2 * h_2 + 1])))));
                                                float2 _f32x2_mul_b_43 = make_float2(-1.442695f, -1.442695f);
                                                float2 e2_2;
                                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_2) : "l"(*(const unsigned long long*)&p2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                                                float _exp2_44 = approx_exp2(e2_2.x);
                                                float _exp2_45 = approx_exp2(e2_2.y);
                                                float2 den2_2 = add_f32x2_noftz(make_float2(_exp2_44, _exp2_45), make_float2(1.0f, 1.0f));
                                                float _rcp_46 = approx_rcp(den2_2.x);
                                                float _rcp_47 = approx_rcp(den2_2.y);
                                                float2 sg2_2 = make_float2(_rcp_46, _rcp_47);
                                                float2 _f32x2_mul_b_48 = fma_f32x2_rn_noftz(p2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2_2), make_float2(1.0f, 1.0f));
                                                float2 g2_2;
                                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_2) : "l"(*(const unsigned long long*)&sg2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
                                                float2 _f32x2_mul_a_49 = make_float2(dyf[2 * h_2], dyf[2 * h_2 + 1]);
                                                float2 d2_2;
                                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&g2_2));
                                                dp[2 * h_2] = d2_2.x;
                                                dp[2 * h_2 + 1] = d2_2.y;
                                                #pragma unroll
                                                for (int k_2 = 0; k_2 < 3; k_2++) {
                                                    float2 q2_2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_2 + k_2], wr[8 * h_2 + 4 + k_2]), d2_2, make_float2(pend[(u_1 % 4 + 1 + k_2) % 4 * 8 + 2 * h_2], pend[(u_1 % 4 + 1 + k_2) % 4 * 8 + 2 * h_2 + 1]));
                                                    pend[(u_1 % 4 + 1 + k_2) % 4 * 8 + 2 * h_2] = q2_2.x;
                                                    pend[(u_1 % 4 + 1 + k_2) % 4 * 8 + 2 * h_2 + 1] = q2_2.y;
                                                }
                                                float2 _f32x2_mul_a_50 = make_float2(wr[8 * h_2 + 3], wr[8 * h_2 + 4 + 3]);
                                                float2 r2_2;
                                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&d2_2));
                                                pend[u_1 % 4 * 8 + 2 * h_2] = r2_2.x;
                                                pend[u_1 % 4 * 8 + 2 * h_2 + 1] = r2_2.y;
                                            }
                                        }
                                        int t3 = tcur - 3;
                                        if (t3 >= t0 && t3 >= doc_start) {
                                            long long row3_1 = (long long)t3 * C64 + c64;
                                            unsigned int _packed_2[4];
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                                _packed_2[_lp] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                                _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 1; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                                _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                                            }
                                            reinterpret_cast<int4*>(dx + (row3_1))[0] = reinterpret_cast<int4*>(_packed_2)[0];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int s_1 = 0; s_1 < 2; s_1++) {
                        if (s == s_1) {
                            asm volatile("barrier.sync %0, %1;" :: "r"(1 + s_1), "r"(ct_vec) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int kn = k + 2;
                    if (kn < n_stages) {
                        int GWn_2 = ct_vec >> 5;
                        int bslot_0 = s * 2 + slot;
                        int rows_k_2 = 0;
                        if (active != 0) {
                            int _max_52 = ((my_rows - kn * 4) > (0) ? (my_rows - kn * 4) : (0));
                            int _min_51 = ((_max_52) < (4) ? (_max_52) : (4));
                            rows_k_2 = _min_51;
                        }
                        if (v == 0 && rows_k_2 > 0) {
                            mbarrier_arrive_expect_tx(stage_full_addr + (((sbars_cursor + 1 < s * 2 + 2) ? sbars_cursor + 1 : s * 2)) * 8, rows_k_2 * (2 * RS));
                            int sx_1 = (int)stage_buf_addr + (s * (GWn_2 * 2) + slot * GWn_2) * 2048;
                            long long src0_1 = (long long)(t0 + kn * 4) * C64 + c64;
                            cp_async_bulk_gmem2smem(sx_1, reinterpret_cast<const uint8_t*>(x + src0_1), rows_k_2 * RS, stage_full_addr + (((sbars_cursor + 1 < s * 2 + 2) ? sbars_cursor + 1 : s * 2)) * 8);
                            cp_async_bulk_gmem2smem(sx_1 + 16384, reinterpret_cast<const uint8_t*>(dy + src0_1), rows_k_2 * RS, stage_full_addr + (((sbars_cursor + 1 < s * 2 + 2) ? sbars_cursor + 1 : s * 2)) * 8);
                        } else if (v < 32) {
                            mbarrier_arrive(stage_full_addr + (((sbars_cursor + 1 < s * 2 + 2) ? sbars_cursor + 1 : s * 2)) * 8);
                        }
                    }
                }
            }
        }
    } else {
        int warp_1 = tid >> 5;
        int lane_1 = tid & 31;
        int li = ((lane_1) < (v) ? (lane_1) : (v));
        int seg_lanes = ((32 - (lane_1 - li)) < (ct_vec - (v - li)) ? (32 - (lane_1 - li)) : (ct_vec - (v - li)));
        int rstride = seg_lanes * 16;
        int seg_off = (lane_1 - li) * 64;
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
            int _min_54 = ((t00 + L) < (S) ? (t00 + L) : (S));
            int _min_53 = ((_min_54 + 3) < (S + 1) ? (_min_54 + 3) : (S + 1));
            steps0 = _min_53 - t00;
        }
        int n_stages_1 = (steps0 + 4 - 1) / 4;
        int _min_1 = ((n_stages_1) < (2) ? (n_stages_1) : (2));
        #pragma unroll 1
        for (int k0_1 = 0; k0_1 < _min_1; k0_1++) {
            int slot0_1 = k0_1;
            int bslot_2 = warp_1 * 2 + slot0_1;
            int rows_k_3 = 0;
            if (active_1 != 0) {
                int _max_56 = ((my_rows_1 - k0_1 * 4) > (0) ? (my_rows_1 - k0_1 * 4) : (0));
                int _min_55 = ((_max_56) < (4) ? (_max_56) : (4));
                rows_k_3 = _min_55;
            }
            if (li == 0 && rows_k_3 > 0) {
                int cb = ((rstride) < ((int)(C64 - c64_1) * 2) ? (rstride) : ((int)(C64 - c64_1) * 2));
                int sx_2 = (int)stage_buf_addr + bslot_2 * 4096 + seg_off;
                long long src0_2 = (long long)(t0 + k0_1 * 4) * C64 + c64_1;
                if (whole_row != 0) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8, rows_k_3 * (2 * cb));
                    cp_async_bulk_gmem2smem(sx_2, reinterpret_cast<const uint8_t*>(x + src0_2), rows_k_3 * cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                    cp_async_bulk_gmem2smem(sx_2 + 2048, reinterpret_cast<const uint8_t*>(dy + src0_2), rows_k_3 * cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                } else if (rstride == 512) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8, 4096);
                    int cc = (int)c64_1;
                    tma_2d_gmem2smem(sx_2, (&x_map), cc, t0 + k0_1 * 4, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                    tma_2d_gmem2smem(sx_2 + 2048, (&dy_map), cc, t0 + k0_1 * 4, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                } else {
                    mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8, rows_k_3 * (2 * cb));
                    #pragma unroll 1
                    for (int i_2 = 0; i_2 < rows_k_3; i_2++) {
                        long long src = src0_2 + (long long)i_2 * C64;
                        cp_async_bulk_gmem2smem(sx_2 + i_2 * rstride, reinterpret_cast<const uint8_t*>(x + src), cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                        cp_async_bulk_gmem2smem(sx_2 + 2048 + i_2 * rstride, reinterpret_cast<const uint8_t*>(dy + src), cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                    }
                }
            } else {
                mbarrier_arrive(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
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
            for (int j_5 = 0; j_5 < 8; j_5++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(w + (c64_1 + (long long)j_5) * 4ll);
                    wr_1[j_5 * 4 + 0] = _v4.x;
                    wr_1[j_5 * 4 + 1] = _v4.y;
                    wr_1[j_5 * 4 + 2] = _v4.z;
                    wr_1[j_5 * 4 + 3] = _v4.w;
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
            int tk_2_3 = t0 - 1;
            if (tk_2_3 >= doc_start_1) {
                long long rowk_1_3 = (long long)tk_2_3 * C64 + c64_1;
                {
                    const uint4* _vptr_58 = reinterpret_cast<const uint4*>(x + rowk_1_3);
                    uint4 _vld_58[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_58[_blk] = _vptr_58[_blk];
                        uint32_t* _vpairs_58 = reinterpret_cast<uint32_t*>(&_vld_58[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win_1[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) << 16);
                            (&win_1[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_1_3 = hist_row_off[frag_1 * 3 + 2 - (doc_start_1 - t0)];
                if (hrow_1_3 >= 0) {
                    long long rowh_1_6 = (long long)hrow_1_3 * C64 + c64_1;
                    {
                        const uint4* _vptr_59 = reinterpret_cast<const uint4*>(hist + rowh_1_6);
                        uint4 _vld_59[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_59[_blk] = _vptr_59[_blk];
                            uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
                                (&win_1[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            int tk_2_4 = t0 - 2;
            if (tk_2_4 >= doc_start_1) {
                long long rowk_1_4 = (long long)tk_2_4 * C64 + c64_1;
                {
                    const uint4* _vptr_60 = reinterpret_cast<const uint4*>(x + rowk_1_4);
                    uint4 _vld_60[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_60[_blk] = _vptr_60[_blk];
                        uint32_t* _vpairs_60 = reinterpret_cast<uint32_t*>(&_vld_60[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win_1[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) << 16);
                            (&win_1[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_1_4 = hist_row_off[frag_1 * 3 + 1 - (doc_start_1 - t0)];
                if (hrow_1_4 >= 0) {
                    long long rowh_1_7 = (long long)hrow_1_4 * C64 + c64_1;
                    {
                        const uint4* _vptr_61 = reinterpret_cast<const uint4*>(hist + rowh_1_7);
                        uint4 _vld_61[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_61[_blk] = _vptr_61[_blk];
                            uint32_t* _vpairs_61 = reinterpret_cast<uint32_t*>(&_vld_61[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) << 16);
                                (&win_1[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            int tk_2_5 = t0 - 3;
            if (tk_2_5 >= doc_start_1) {
                long long rowk_1_5 = (long long)tk_2_5 * C64 + c64_1;
                {
                    const uint4* _vptr_62 = reinterpret_cast<const uint4*>(x + rowk_1_5);
                    uint4 _vld_62[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_62[_blk] = _vptr_62[_blk];
                        uint32_t* _vpairs_62 = reinterpret_cast<uint32_t*>(&_vld_62[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win_1[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_62[_pair]) << 16);
                            (&win_1[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_62[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_1_5 = hist_row_off[frag_1 * 3 - (doc_start_1 - t0)];
                if (hrow_1_5 >= 0) {
                    long long rowh_1_8 = (long long)hrow_1_5 * C64 + c64_1;
                    {
                        const uint4* _vptr_63 = reinterpret_cast<const uint4*>(hist + rowh_1_8);
                        uint4 _vld_63[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_63[_blk] = _vptr_63[_blk];
                            uint32_t* _vpairs_63 = reinterpret_cast<uint32_t*>(&_vld_63[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) << 16);
                                (&win_1[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) & 0xffff0000u);
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
        for (int k_3 = 0; k_3 < n_stages_1; k_3++) {
            int slot_1 = k_3 % 2;
            int bslot_3 = warp_1 * 2 + slot_1;
            int _stage_stage_full = warp_1 + 4 * (int)((unsigned int)k_3 % 2);
            int _parity_stage_full = (int)((unsigned int)k_3 / 2 % 2);
            mbarrier_wait(stage_full_addr + (_stage_stage_full) * 8, _parity_stage_full);
            if (active_1 != 0) {
                int tb_1 = t0 + k_3 * 4;
                int _max_64 = ((my_rows_1 - k_3 * 4) > (0) ? (my_rows_1 - k_3 * 4) : (0));
                int rows_k_4 = ((_max_64) < (4) ? (_max_64) : (4));
                int _max_65 = ((my_steps_1 - k_3 * 4) > (0) ? (my_steps_1 - k_3 * 4) : (0));
                int steps_k_1 = ((_max_65) < (4) ? (_max_65) : (4));
                int sx0_1 = (int)stage_buf_addr + bslot_3 * 4096 + seg_off + li * 16;
                int interior_1 = 0;
                if (steps_k_1 == 4 && rows_k_4 == 4 && t_end_1 >= tb_1 + 4 && doc_end_1 >= tb_1 + 4) {
                    if (k_3 >= 1) {
                        if (doc_start_1 <= tb_1 - 3) {
                            interior_1 = 1;
                        }
                    }
                }
                if (interior_1 != 0) {
                    #pragma unroll
                    for (int u_2 = 0; u_2 < 4; u_2++) {
                        __int128_t _smem_raw_66;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_66) : "r"(sx0_1 + u_2 * rstride));
                        pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_66)[0];
                        pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_66)[1];
                        pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_66)[2];
                        pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_66)[3];
                        __int128_t _smem_raw_67;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_67) : "r"(sx0_1 + 2048 + u_2 * rstride));
                        pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_67)[0];
                        pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_67)[1];
                        pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_67)[2];
                        pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_67)[3];
                        #pragma unroll
                        for (int kk_7 = 0; kk_7 < 4; kk_7++) {
                            win_1[u_2 % 4 * 8 + 2 * kk_7] = __uint_as_float(pkx_1[kk_7] << 16);
                            win_1[u_2 % 4 * 8 + 2 * kk_7 + 1] = __uint_as_float(pkx_1[kk_7] & 4294901760u);
                            dyf_1[2 * kk_7] = __uint_as_float(pkd_1[kk_7] << 16);
                            dyf_1[2 * kk_7 + 1] = __uint_as_float(pkd_1[kk_7] & 4294901760u);
                        }
                        #pragma unroll
                        for (int h_3 = 0; h_3 < 4; h_3++) {
                            float2 p2_3 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_3 + 3], wr_1[8 * h_3 + 4 + 3]), make_float2(win_1[u_2 % 4 * 8 + 2 * h_3], win_1[u_2 % 4 * 8 + 2 * h_3 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_3 + 2], wr_1[8 * h_3 + 4 + 2]), make_float2(win_1[(u_2 % 4 + 3) % 4 * 8 + 2 * h_3], win_1[(u_2 % 4 + 3) % 4 * 8 + 2 * h_3 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_3 + 1], wr_1[8 * h_3 + 4 + 1]), make_float2(win_1[(u_2 % 4 + 2) % 4 * 8 + 2 * h_3], win_1[(u_2 % 4 + 2) % 4 * 8 + 2 * h_3 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * h_3], wr_1[8 * h_3 + 4]), make_float2(win_1[(u_2 % 4 + 1) % 4 * 8 + 2 * h_3], win_1[(u_2 % 4 + 1) % 4 * 8 + 2 * h_3 + 1])))));
                            float2 _f32x2_mul_b_68 = make_float2(-1.442695f, -1.442695f);
                            float2 e2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_3) : "l"(*(const unsigned long long*)&p2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
                            float _exp2_69 = approx_exp2(e2_3.x);
                            float _exp2_70 = approx_exp2(e2_3.y);
                            float2 den2_3 = add_f32x2_noftz(make_float2(_exp2_69, _exp2_70), make_float2(1.0f, 1.0f));
                            float _rcp_71 = approx_rcp(den2_3.x);
                            float _rcp_72 = approx_rcp(den2_3.y);
                            float2 sg2_3 = make_float2(_rcp_71, _rcp_72);
                            float2 _f32x2_mul_b_73 = fma_f32x2_rn_noftz(p2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2_3), make_float2(1.0f, 1.0f));
                            float2 g2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_3) : "l"(*(const unsigned long long*)&sg2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
                            float2 _f32x2_mul_a_74 = make_float2(dyf_1[2 * h_3], dyf_1[2 * h_3 + 1]);
                            float2 d2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&g2_3));
                            dp_1[2 * h_3] = d2_3.x;
                            dp_1[2 * h_3 + 1] = d2_3.y;
                            #pragma unroll
                            for (int kk_8 = 0; kk_8 < 4; kk_8++) {
                                float2 a2_2 = fma_f32x2_rn_noftz(d2_3, make_float2(win_1[(u_2 % 4 + 1 + kk_8) % 4 * 8 + 2 * h_3], win_1[(u_2 % 4 + 1 + kk_8) % 4 * 8 + 2 * h_3 + 1]), make_float2(acc[8 * h_3 + kk_8], acc[8 * h_3 + 4 + kk_8]));
                                acc[8 * h_3 + kk_8] = a2_2.x;
                                acc[8 * h_3 + 4 + kk_8] = a2_2.y;
                            }
                            #pragma unroll
                            for (int k_3_1 = 0; k_3_1 < 3; k_3_1++) {
                                float2 q2_3 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_3 + k_3_1], wr_1[8 * h_3 + 4 + k_3_1]), d2_3, make_float2(pend_1[(u_2 % 4 + 1 + k_3_1) % 4 * 8 + 2 * h_3], pend_1[(u_2 % 4 + 1 + k_3_1) % 4 * 8 + 2 * h_3 + 1]));
                                pend_1[(u_2 % 4 + 1 + k_3_1) % 4 * 8 + 2 * h_3] = q2_3.x;
                                pend_1[(u_2 % 4 + 1 + k_3_1) % 4 * 8 + 2 * h_3 + 1] = q2_3.y;
                            }
                            float2 _f32x2_mul_a_75 = make_float2(wr_1[8 * h_3 + 3], wr_1[8 * h_3 + 4 + 3]);
                            float2 r2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&d2_3));
                            pend_1[u_2 % 4 * 8 + 2 * h_3] = r2_3.x;
                            pend_1[u_2 % 4 * 8 + 2 * h_3 + 1] = r2_3.y;
                        }
                        long long row3_2 = (long long)(tb_1 + (u_2 - 3)) * C64 + c64_1;
                        unsigned int packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_2 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend_1 + (u_2 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                            packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_2 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend_1 + (u_2 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                            packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_2 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend_1 + (u_2 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                            packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_2 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend_1 + (u_2 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                            packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + (row3_2))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    }
                } else {
                    #pragma unroll
                    for (int u_3 = 0; u_3 < 4; u_3++) {
                        if (steps_k_1 > u_3) {
                            int tcur_1 = tb_1 + u_3;
                            if (rows_k_4 > u_3) {
                                __int128_t _smem_raw_76;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_76) : "r"(sx0_1 + u_3 * rstride));
                                pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_76)[0];
                                pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_76)[1];
                                pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_76)[2];
                                pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_76)[3];
                                __int128_t _smem_raw_77;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_77) : "r"(sx0_1 + 2048 + u_3 * rstride));
                                pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_77)[0];
                                pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_77)[1];
                                pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_77)[2];
                                pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_77)[3];
                                #pragma unroll
                                for (int kk_9 = 0; kk_9 < 4; kk_9++) {
                                    win_1[u_3 % 4 * 8 + 2 * kk_9] = __uint_as_float(pkx_1[kk_9] << 16);
                                    win_1[u_3 % 4 * 8 + 2 * kk_9 + 1] = __uint_as_float(pkx_1[kk_9] & 4294901760u);
                                    dyf_1[2 * kk_9] = __uint_as_float(pkd_1[kk_9] << 16);
                                    dyf_1[2 * kk_9 + 1] = __uint_as_float(pkd_1[kk_9] & 4294901760u);
                                }
                            }
                            if (tcur_1 >= doc_end_1) {
                                int trow_2 = tail_row_off[frag_1 * 3];
                                if (trow_2 >= 0) {
                                    long long rowr_3 = (long long)trow_2 * C64 + c64_1;
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
                                            : "=r"(_ldv8_78_0), "=r"(_ldv8_78_1), "=r"(_ldv8_78_2), "=r"(_ldv8_78_3), "=r"(_ldv8_78_4), "=r"(_ldv8_78_5), "=r"(_ldv8_78_6), "=r"(_ldv8_78_7) : "l"((const void*)(dpre_tail + (rowr_3))) : "memory");
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
                                    for (int j_6 = 0; j_6 < 8; j_6++) {
                                        #pragma unroll
                                        for (int kk_10 = 0; kk_10 < 4; kk_10++) {
                                            w4_1[kk_10] = wr_1[j_6 * 4 + kk_10];
                                        }
                                        float _fma_79;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(w4_1[2]), "f"(tailf_1[j_6]), "f"(pend_1[(u_3 % 4 + 4 - 1) % 4 * 8 + j_6]));
                                        pend_1[(u_3 % 4 + 4 - 1) % 4 * 8 + j_6] = _fma_79;
                                        float _fma_80;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(w4_1[1]), "f"(tailf_1[j_6]), "f"(pend_1[(u_3 % 4 + 4 - 2) % 4 * 8 + j_6]));
                                        pend_1[(u_3 % 4 + 4 - 2) % 4 * 8 + j_6] = _fma_80;
                                        float _fma_81;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(w4_1[0]), "f"(tailf_1[j_6]), "f"(pend_1[(u_3 % 4 + 4 - 3) % 4 * 8 + j_6]));
                                        pend_1[(u_3 % 4 + 4 - 3) % 4 * 8 + j_6] = _fma_81;
                                    }
                                }
                                int trow_0_1 = tail_row_off[frag_1 * 3 + 1];
                                if (trow_0_1 >= 0) {
                                    long long rowr_4 = (long long)trow_0_1 * C64 + c64_1;
                                    {
                                        unsigned _ldv8_82_0;
                                        unsigned _ldv8_82_1;
                                        unsigned _ldv8_82_2;
                                        unsigned _ldv8_82_3;
                                        unsigned _ldv8_82_4;
                                        unsigned _ldv8_82_5;
                                        unsigned _ldv8_82_6;
                                        unsigned _ldv8_82_7;
                                        asm volatile(
                                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_ldv8_82_0), "=r"(_ldv8_82_1), "=r"(_ldv8_82_2), "=r"(_ldv8_82_3), "=r"(_ldv8_82_4), "=r"(_ldv8_82_5), "=r"(_ldv8_82_6), "=r"(_ldv8_82_7) : "l"((const void*)(dpre_tail + (rowr_4))) : "memory");
                                        tailf_1[0 + 0] = __uint_as_float(_ldv8_82_0);
                                        tailf_1[0 + 1] = __uint_as_float(_ldv8_82_1);
                                        tailf_1[0 + 2] = __uint_as_float(_ldv8_82_2);
                                        tailf_1[0 + 3] = __uint_as_float(_ldv8_82_3);
                                        tailf_1[0 + 4] = __uint_as_float(_ldv8_82_4);
                                        tailf_1[0 + 5] = __uint_as_float(_ldv8_82_5);
                                        tailf_1[0 + 6] = __uint_as_float(_ldv8_82_6);
                                        tailf_1[0 + 7] = __uint_as_float(_ldv8_82_7);
                                    }
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 8; j_7++) {
                                        #pragma unroll
                                        for (int kk_11 = 0; kk_11 < 4; kk_11++) {
                                            w4_1[kk_11] = wr_1[j_7 * 4 + kk_11];
                                        }
                                        float _fma_83;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(w4_1[1]), "f"(tailf_1[j_7]), "f"(pend_1[(u_3 % 4 + 4 + 1 - 2) % 4 * 8 + j_7]));
                                        pend_1[(u_3 % 4 + 4 + 1 - 2) % 4 * 8 + j_7] = _fma_83;
                                        float _fma_84;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(w4_1[0]), "f"(tailf_1[j_7]), "f"(pend_1[(u_3 % 4 + 4 + 1 - 3) % 4 * 8 + j_7]));
                                        pend_1[(u_3 % 4 + 4 + 1 - 3) % 4 * 8 + j_7] = _fma_84;
                                    }
                                }
                                int trow_1_1 = tail_row_off[frag_1 * 3 + 2];
                                if (trow_1_1 >= 0) {
                                    long long rowr_5 = (long long)trow_1_1 * C64 + c64_1;
                                    {
                                        unsigned _ldv8_85_0;
                                        unsigned _ldv8_85_1;
                                        unsigned _ldv8_85_2;
                                        unsigned _ldv8_85_3;
                                        unsigned _ldv8_85_4;
                                        unsigned _ldv8_85_5;
                                        unsigned _ldv8_85_6;
                                        unsigned _ldv8_85_7;
                                        asm volatile(
                                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_ldv8_85_0), "=r"(_ldv8_85_1), "=r"(_ldv8_85_2), "=r"(_ldv8_85_3), "=r"(_ldv8_85_4), "=r"(_ldv8_85_5), "=r"(_ldv8_85_6), "=r"(_ldv8_85_7) : "l"((const void*)(dpre_tail + (rowr_5))) : "memory");
                                        tailf_1[0 + 0] = __uint_as_float(_ldv8_85_0);
                                        tailf_1[0 + 1] = __uint_as_float(_ldv8_85_1);
                                        tailf_1[0 + 2] = __uint_as_float(_ldv8_85_2);
                                        tailf_1[0 + 3] = __uint_as_float(_ldv8_85_3);
                                        tailf_1[0 + 4] = __uint_as_float(_ldv8_85_4);
                                        tailf_1[0 + 5] = __uint_as_float(_ldv8_85_5);
                                        tailf_1[0 + 6] = __uint_as_float(_ldv8_85_6);
                                        tailf_1[0 + 7] = __uint_as_float(_ldv8_85_7);
                                    }
                                    #pragma unroll
                                    for (int j_8 = 0; j_8 < 8; j_8++) {
                                        #pragma unroll
                                        for (int kk_12 = 0; kk_12 < 4; kk_12++) {
                                            w4_1[kk_12] = wr_1[j_8 * 4 + kk_12];
                                        }
                                        float _fma_86;
                                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(w4_1[0]), "f"(tailf_1[j_8]), "f"(pend_1[(u_3 % 4 + 4 + 2 - 3) % 4 * 8 + j_8]));
                                        pend_1[(u_3 % 4 + 4 + 2 - 3) % 4 * 8 + j_8] = _fma_86;
                                    }
                                }
                                #pragma unroll
                                for (int m_2 = 1; m_2 < 4; m_2++) {
                                    int tm_1 = tcur_1 - m_2;
                                    if (tm_1 >= t0 && tm_1 < t_end_1 && tm_1 >= doc_start_1) {
                                        long long rowm_1 = (long long)tm_1 * C64 + c64_1;
                                        unsigned int _packed_4[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 4 - m_2) % 4 * 8)[_lp*2 + 0], (pend_1 + (u_3 % 4 + 4 - m_2) % 4 * 8)[_lp*2+1 + 0]));
                                            _packed_4[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 4 - m_2) % 4 * 8)[_lp*2 + 2], (pend_1 + (u_3 % 4 + 4 - m_2) % 4 * 8)[_lp*2+1 + 2]));
                                            _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 4 - m_2) % 4 * 8)[_lp*2 + 4], (pend_1 + (u_3 % 4 + 4 - m_2) % 4 * 8)[_lp*2+1 + 4]));
                                            _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 4 - m_2) % 4 * 8)[_lp*2 + 6], (pend_1 + (u_3 % 4 + 4 - m_2) % 4 * 8)[_lp*2+1 + 6]));
                                            _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(dx + (rowm_1))[0] = reinterpret_cast<int4*>(_packed_4)[0];
                                    }
                                }
                                #pragma unroll
                                for (int m_3 = 1; m_3 < 4; m_3++) {
                                    #pragma unroll
                                    for (int j_9 = 0; j_9 < 8; j_9++) {
                                        win_1[(u_3 % 4 + 4 - m_3) % 4 * 8 + j_9] = 0.0f;
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
                                    int hrow_2_3 = hist_row_off[frag_1 * 3 + 2];
                                    if (hrow_2_3 >= 0) {
                                        long long rowh_1_9 = (long long)hrow_2_3 * C64 + c64_1;
                                        {
                                            const uint4* _vptr_87 = reinterpret_cast<const uint4*>(hist + rowh_1_9);
                                            uint4 _vld_87[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_87[_blk] = _vptr_87[_blk];
                                                uint32_t* _vpairs_87 = reinterpret_cast<uint32_t*>(&_vld_87[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win_1[(u_3 % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_87[_pair]) << 16);
                                                    (&win_1[(u_3 % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_87[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                    int hrow_2_4 = hist_row_off[frag_1 * 3 + 1];
                                    if (hrow_2_4 >= 0) {
                                        long long rowh_1_10 = (long long)hrow_2_4 * C64 + c64_1;
                                        {
                                            const uint4* _vptr_88 = reinterpret_cast<const uint4*>(hist + rowh_1_10);
                                            uint4 _vld_88[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_88[_blk] = _vptr_88[_blk];
                                                uint32_t* _vpairs_88 = reinterpret_cast<uint32_t*>(&_vld_88[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win_1[(u_3 % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) << 16);
                                                    (&win_1[(u_3 % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                    int hrow_2_5 = hist_row_off[frag_1 * 3];
                                    if (hrow_2_5 >= 0) {
                                        long long rowh_1_11 = (long long)hrow_2_5 * C64 + c64_1;
                                        {
                                            const uint4* _vptr_89 = reinterpret_cast<const uint4*>(hist + rowh_1_11);
                                            uint4 _vld_89[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_89[_blk] = _vptr_89[_blk];
                                                uint32_t* _vpairs_89 = reinterpret_cast<uint32_t*>(&_vld_89[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&win_1[(u_3 % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) << 16);
                                                    (&win_1[(u_3 % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    doc_start_1 = tcur_1 + 4;
                                    doc_end_1 = tcur_1 + 4;
                                }
                            }
                            if (tcur_1 < S) {
                                if (tcur_1 < t_end_1) {
                                    #pragma unroll
                                    for (int h_4 = 0; h_4 < 4; h_4++) {
                                        float2 p2_4 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_4 + 3], wr_1[8 * h_4 + 4 + 3]), make_float2(win_1[u_3 % 4 * 8 + 2 * h_4], win_1[u_3 % 4 * 8 + 2 * h_4 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_4 + 2], wr_1[8 * h_4 + 4 + 2]), make_float2(win_1[(u_3 % 4 + 3) % 4 * 8 + 2 * h_4], win_1[(u_3 % 4 + 3) % 4 * 8 + 2 * h_4 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_4 + 1], wr_1[8 * h_4 + 4 + 1]), make_float2(win_1[(u_3 % 4 + 2) % 4 * 8 + 2 * h_4], win_1[(u_3 % 4 + 2) % 4 * 8 + 2 * h_4 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * h_4], wr_1[8 * h_4 + 4]), make_float2(win_1[(u_3 % 4 + 1) % 4 * 8 + 2 * h_4], win_1[(u_3 % 4 + 1) % 4 * 8 + 2 * h_4 + 1])))));
                                        float2 _f32x2_mul_b_90 = make_float2(-1.442695f, -1.442695f);
                                        float2 e2_4;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_4) : "l"(*(const unsigned long long*)&p2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_90));
                                        float _exp2_91 = approx_exp2(e2_4.x);
                                        float _exp2_92 = approx_exp2(e2_4.y);
                                        float2 den2_4 = add_f32x2_noftz(make_float2(_exp2_91, _exp2_92), make_float2(1.0f, 1.0f));
                                        float _rcp_93 = approx_rcp(den2_4.x);
                                        float _rcp_94 = approx_rcp(den2_4.y);
                                        float2 sg2_4 = make_float2(_rcp_93, _rcp_94);
                                        float2 _f32x2_mul_b_95 = fma_f32x2_rn_noftz(p2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2_4), make_float2(1.0f, 1.0f));
                                        float2 g2_4;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_4) : "l"(*(const unsigned long long*)&sg2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_95));
                                        float2 _f32x2_mul_a_96 = make_float2(dyf_1[2 * h_4], dyf_1[2 * h_4 + 1]);
                                        float2 d2_4;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&g2_4));
                                        dp_1[2 * h_4] = d2_4.x;
                                        dp_1[2 * h_4 + 1] = d2_4.y;
                                        #pragma unroll
                                        for (int kk_13 = 0; kk_13 < 4; kk_13++) {
                                            float2 a2_3 = fma_f32x2_rn_noftz(d2_4, make_float2(win_1[(u_3 % 4 + 1 + kk_13) % 4 * 8 + 2 * h_4], win_1[(u_3 % 4 + 1 + kk_13) % 4 * 8 + 2 * h_4 + 1]), make_float2(acc[8 * h_4 + kk_13], acc[8 * h_4 + 4 + kk_13]));
                                            acc[8 * h_4 + kk_13] = a2_3.x;
                                            acc[8 * h_4 + 4 + kk_13] = a2_3.y;
                                        }
                                        #pragma unroll
                                        for (int k_4 = 0; k_4 < 3; k_4++) {
                                            float2 q2_4 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_4 + k_4], wr_1[8 * h_4 + 4 + k_4]), d2_4, make_float2(pend_1[(u_3 % 4 + 1 + k_4) % 4 * 8 + 2 * h_4], pend_1[(u_3 % 4 + 1 + k_4) % 4 * 8 + 2 * h_4 + 1]));
                                            pend_1[(u_3 % 4 + 1 + k_4) % 4 * 8 + 2 * h_4] = q2_4.x;
                                            pend_1[(u_3 % 4 + 1 + k_4) % 4 * 8 + 2 * h_4 + 1] = q2_4.y;
                                        }
                                        float2 _f32x2_mul_a_97 = make_float2(wr_1[8 * h_4 + 3], wr_1[8 * h_4 + 4 + 3]);
                                        float2 r2_4;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&d2_4));
                                        pend_1[u_3 % 4 * 8 + 2 * h_4] = r2_4.x;
                                        pend_1[u_3 % 4 * 8 + 2 * h_4 + 1] = r2_4.y;
                                    }
                                } else {
                                    #pragma unroll
                                    for (int h_5 = 0; h_5 < 4; h_5++) {
                                        float2 p2_5 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_5 + 3], wr_1[8 * h_5 + 4 + 3]), make_float2(win_1[u_3 % 4 * 8 + 2 * h_5], win_1[u_3 % 4 * 8 + 2 * h_5 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_5 + 2], wr_1[8 * h_5 + 4 + 2]), make_float2(win_1[(u_3 % 4 + 3) % 4 * 8 + 2 * h_5], win_1[(u_3 % 4 + 3) % 4 * 8 + 2 * h_5 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_5 + 1], wr_1[8 * h_5 + 4 + 1]), make_float2(win_1[(u_3 % 4 + 2) % 4 * 8 + 2 * h_5], win_1[(u_3 % 4 + 2) % 4 * 8 + 2 * h_5 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * h_5], wr_1[8 * h_5 + 4]), make_float2(win_1[(u_3 % 4 + 1) % 4 * 8 + 2 * h_5], win_1[(u_3 % 4 + 1) % 4 * 8 + 2 * h_5 + 1])))));
                                        float2 _f32x2_mul_b_98 = make_float2(-1.442695f, -1.442695f);
                                        float2 e2_5;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_5) : "l"(*(const unsigned long long*)&p2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_98));
                                        float _exp2_99 = approx_exp2(e2_5.x);
                                        float _exp2_100 = approx_exp2(e2_5.y);
                                        float2 den2_5 = add_f32x2_noftz(make_float2(_exp2_99, _exp2_100), make_float2(1.0f, 1.0f));
                                        float _rcp_101 = approx_rcp(den2_5.x);
                                        float _rcp_102 = approx_rcp(den2_5.y);
                                        float2 sg2_5 = make_float2(_rcp_101, _rcp_102);
                                        float2 _f32x2_mul_b_103 = fma_f32x2_rn_noftz(p2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2_5), make_float2(1.0f, 1.0f));
                                        float2 g2_5;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_5) : "l"(*(const unsigned long long*)&sg2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_103));
                                        float2 _f32x2_mul_a_104 = make_float2(dyf_1[2 * h_5], dyf_1[2 * h_5 + 1]);
                                        float2 d2_5;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&g2_5));
                                        dp_1[2 * h_5] = d2_5.x;
                                        dp_1[2 * h_5 + 1] = d2_5.y;
                                        #pragma unroll
                                        for (int k_5 = 0; k_5 < 3; k_5++) {
                                            float2 q2_5 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * h_5 + k_5], wr_1[8 * h_5 + 4 + k_5]), d2_5, make_float2(pend_1[(u_3 % 4 + 1 + k_5) % 4 * 8 + 2 * h_5], pend_1[(u_3 % 4 + 1 + k_5) % 4 * 8 + 2 * h_5 + 1]));
                                            pend_1[(u_3 % 4 + 1 + k_5) % 4 * 8 + 2 * h_5] = q2_5.x;
                                            pend_1[(u_3 % 4 + 1 + k_5) % 4 * 8 + 2 * h_5 + 1] = q2_5.y;
                                        }
                                        float2 _f32x2_mul_a_105 = make_float2(wr_1[8 * h_5 + 3], wr_1[8 * h_5 + 4 + 3]);
                                        float2 r2_5;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&d2_5));
                                        pend_1[u_3 % 4 * 8 + 2 * h_5] = r2_5.x;
                                        pend_1[u_3 % 4 * 8 + 2 * h_5 + 1] = r2_5.y;
                                    }
                                }
                                int t3_1 = tcur_1 - 3;
                                if (t3_1 >= t0 && t3_1 >= doc_start_1) {
                                    long long row3_3 = (long long)t3_1 * C64 + c64_1;
                                    unsigned int _packed_5[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                        _packed_5[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                        _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                        _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                        _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<int4*>(dx + (row3_3))[0] = reinterpret_cast<int4*>(_packed_5)[0];
                                }
                            }
                        }
                    }
                }
            }
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn_1 = k_3 + 2;
            if (kn_1 < n_stages_1) {
                int bslot_1_1 = warp_1 * 2 + slot_1;
                int rows_k_5 = 0;
                if (active_1 != 0) {
                    int _max_107 = ((my_rows_1 - kn_1 * 4) > (0) ? (my_rows_1 - kn_1 * 4) : (0));
                    int _min_106 = ((_max_107) < (4) ? (_max_107) : (4));
                    rows_k_5 = _min_106;
                }
                if (li == 0 && rows_k_5 > 0) {
                    int cb_1 = ((rstride) < ((int)(C64 - c64_1) * 2) ? (rstride) : ((int)(C64 - c64_1) * 2));
                    int sx_3 = (int)stage_buf_addr + bslot_1_1 * 4096 + seg_off;
                    long long src0_3 = (long long)(t0 + kn_1 * 4) * C64 + c64_1;
                    if (whole_row != 0) {
                        mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8, rows_k_5 * (2 * cb_1));
                        cp_async_bulk_gmem2smem(sx_3, reinterpret_cast<const uint8_t*>(x + src0_3), rows_k_5 * cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8);
                        cp_async_bulk_gmem2smem(sx_3 + 2048, reinterpret_cast<const uint8_t*>(dy + src0_3), rows_k_5 * cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8);
                    } else if (rstride == 512) {
                        mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8, 4096);
                        int cc_1 = (int)c64_1;
                        tma_2d_gmem2smem(sx_3, (&x_map), cc_1, t0 + kn_1 * 4, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8);
                        tma_2d_gmem2smem(sx_3 + 2048, (&dy_map), cc_1, t0 + kn_1 * 4, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8);
                    } else {
                        mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8, rows_k_5 * (2 * cb_1));
                        #pragma unroll 1
                        for (int i_4 = 0; i_4 < rows_k_5; i_4++) {
                            long long src_1 = src0_3 + (long long)i_4 * C64;
                            cp_async_bulk_gmem2smem(sx_3 + i_4 * rstride, reinterpret_cast<const uint8_t*>(x + src_1), cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8);
                            cp_async_bulk_gmem2smem(sx_3 + 2048 + i_4 * rstride, reinterpret_cast<const uint8_t*>(dy + src_1), cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8);
                        }
                    }
                } else {
                    mbarrier_arrive(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_3 + 2) % 2)) * 8);
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
        for (int s_2 = 1; s_2 < n_sub; s_2++) {
            #pragma unroll
            for (int i_7 = 0; i_7 < 32; i_7++) {
                tot[i_7] = tot[i_7] + red[(s_2 * ct_vec + v) * 32 + i_7];
            }
        }
        #pragma unroll
        for (int j_10 = 0; j_10 < 8; j_10++) {
            long long pidx = ((long long)rank * C64 + (long long)c0_0 + (long long)j_10) * 4ll;
            {
                float4 _v4 = make_float4(tot[j_10 * 4 + 0], tot[j_10 * 4 + 1], tot[j_10 * 4 + 2], tot[j_10 * 4 + 3]);
                *reinterpret_cast<float4*>(partial + pidx) = _v4;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Cleanup
    __syncthreads();
}

} // extern "C"
