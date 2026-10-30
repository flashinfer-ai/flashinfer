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
#define SMEM_REDC_OFF 17408
#define SMEM_REDC_STAGE_BYTES 16384
#define SMEM_REDC_STRIDE 16384
#define SMEM_STAGE_BUF_OFF 33792
#define SMEM_STAGE_BUF_STAGE_BYTES 65536
#define SMEM_STAGE_BUF_STRIDE 65536
#define SMEM_TOTAL 99328
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
kernel_cake_causal_conv1d_train_170d89481a616ae977b3(__nv_bfloat16* __restrict__ x, void* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ partial, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles, int flags, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap dy_map)
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
    float* redc = reinterpret_cast<float*>(smem_raw + 17408);
    const int redc_addr = smem + 17408;
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int stage_buf_addr = smem + 33792;

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
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; i_1++) {
        red[i_1 * 128 + tid] = 0.0f;
        redc[i_1 * 128 + tid] = 0.0f;
    }
    bool grouped = (ct_vec & 31) == 0 && ct_vec > 32 && ct_vec * 8 == C;
    if (grouped) {
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
                mbarrier_arrive_expect_tx(stage_full_addr + (sub + 4 * (int)((unsigned int)k0 % 2)) * 8, rows_k * (2 * RS));
                int sx = (int)stage_buf_addr + (sub * (GWn * 2) + slot0 * GWn) * 4096;
                long long src0 = (long long)(t0 + k0 * 8) * C64 + c64;
                cp_async_bulk_gmem2smem(sx, reinterpret_cast<const uint8_t*>(x + src0), rows_k * RS, stage_full_addr + (sub + 4 * (int)((unsigned int)k0 % 2)) * 8);
                cp_async_bulk_gmem2smem(sx + 32768, reinterpret_cast<const uint8_t*>(dy + src0), rows_k * RS, stage_full_addr + (sub + 4 * (int)((unsigned int)k0 % 2)) * 8);
            } else if (v < 32) {
                mbarrier_arrive(stage_full_addr + (sub + 4 * (int)((unsigned int)k0 % 2)) * 8);
            }
        }
        float wr[34];
        wr[32] = __uint_as_float((unsigned int)S >> 31 | 2147483648u);
        wr[33] = wr[32];
        float w4[4];
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
                for (int j = 0; j < 8; j++) {
                    {
                        uint2 _vld_2;
                        _vld_2 = *reinterpret_cast<const uint2*>(reinterpret_cast<__nv_bfloat16*>(w) + (c64 + (long long)j) * 4);
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&wr[j * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&wr[j * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(w) + (c64 + (long long)j_1) * 4);
                        wr[j_1 * 4 + 0] = _v4.x;
                        wr[j_1 * 4 + 1] = _v4.y;
                        wr[j_1 * 4 + 2] = _v4.z;
                        wr[j_1 * 4 + 3] = _v4.w;
                    }
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
                        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + rowk);
                        uint4 _vld_4[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_4[_blk] = _vptr_4[_blk];
                            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
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
            int bslot_1 = sub * 2 + slot;
            int _stage_stage_full = sub + 4 * (int)((unsigned int)k % 2);
            int _parity_stage_full = (int)((unsigned int)k / 2 % 2);
            mbarrier_wait(stage_full_addr + (_stage_stage_full) * 8, _parity_stage_full);
            if (active != 0) {
                int tb = t0 + k * 8;
                int _max_5 = ((my_rows - k * 8) > (0) ? (my_rows - k * 8) : (0));
                int rows_k_1 = ((_max_5) < (8) ? (_max_5) : (8));
                int _max_6 = ((my_steps - k * 8) > (0) ? (my_steps - k * 8) : (0));
                int steps_k = ((_max_6) < (8) ? (_max_6) : (8));
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
                        __int128_t _smem_raw_7;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(sx0 + u * RS));
                        pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                        pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                        pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                        pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                        __int128_t _smem_raw_8;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(sx0 + 32768 + u * RS));
                        pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                        pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                        pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                        pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                        #pragma unroll
                        for (int kk = 0; kk < 4; kk++) {
                            win[u % 4 * 8 + 2 * kk] = __uint_as_float(pkx[kk] << 16);
                            win[u % 4 * 8 + 2 * kk + 1] = __uint_as_float(pkx[kk] & 4294901760u);
                            dyf[2 * kk] = __uint_as_float(pkd[kk] << 16);
                            dyf[2 * kk + 1] = __uint_as_float(pkd[kk] & 4294901760u);
                        }
                        #pragma unroll
                        for (int pv_h = 0; pv_h < 4; pv_h++) {
                            float2 pv_p2 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 3], wr[8 * pv_h + 4 + 3]), make_float2(win[u % 4 * 8 + 2 * pv_h], win[u % 4 * 8 + 2 * pv_h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 2], wr[8 * pv_h + 4 + 2]), make_float2(win[(u % 4 + 3) % 4 * 8 + 2 * pv_h], win[(u % 4 + 3) % 4 * 8 + 2 * pv_h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 1], wr[8 * pv_h + 4 + 1]), make_float2(win[(u % 4 + 2) % 4 * 8 + 2 * pv_h], win[(u % 4 + 2) % 4 * 8 + 2 * pv_h + 1]), mul_f32x2_noftz(make_float2(wr[8 * pv_h + 0], wr[8 * pv_h + 4 + 0]), make_float2(win[(u % 4 + 1) % 4 * 8 + 2 * pv_h], win[(u % 4 + 1) % 4 * 8 + 2 * pv_h + 1])))));
                            float2 _f32x2_mul_b_9 = make_float2(-1.442695f, -1.442695f);
                            float2 pv_e2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2) : "l"(*(const unsigned long long*)&pv_p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
                            float _exp2_10 = approx_exp2(pv_e2.x);
                            float _exp2_11 = approx_exp2(pv_e2.y);
                            float2 pv_den2 = add_f32x2_noftz(make_float2(_exp2_10, _exp2_11), make_float2(1.0f, 1.0f));
                            float _rcp_12 = approx_rcp(pv_den2.x);
                            float _rcp_13 = approx_rcp(pv_den2.y);
                            float2 pv_sg2 = make_float2(_rcp_12, _rcp_13);
                            float2 _f32x2_mul_b_14 = fma_f32x2_rn_noftz(pv_p2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2), make_float2(1.0f, 1.0f));
                            float2 pv_g2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2) : "l"(*(const unsigned long long*)&pv_sg2), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
                            float2 pv_d2 = fma_f32x2_rn_noftz(make_float2(dyf[0 + 2 * pv_h], dyf[0 + 2 * pv_h + 1]), pv_g2, make_float2(wr[32], wr[33]));
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
                            float2 _f32x2_mul_a_15 = make_float2(wr[8 * pv_h + 3], wr[8 * pv_h + 4 + 3]);
                            float2 pv_r2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&pv_d2));
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
                        for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                            win[u_1 % 4 * 8 + 2 * kk_2] = __uint_as_float(pkx[kk_2] << 16);
                            win[u_1 % 4 * 8 + 2 * kk_2 + 1] = __uint_as_float(pkx[kk_2] & 4294901760u);
                            dyf[2 * kk_2] = __uint_as_float(pkd[kk_2] << 16);
                            dyf[2 * kk_2 + 1] = __uint_as_float(pkd[kk_2] & 4294901760u);
                        }
                        #pragma unroll
                        for (int pv_h_1 = 0; pv_h_1 < 4; pv_h_1++) {
                            float2 pv_p2_1 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 3], wr[8 * pv_h_1 + 4 + 3]), make_float2(win[u_1 % 4 * 8 + 2 * pv_h_1], win[u_1 % 4 * 8 + 2 * pv_h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 2], wr[8 * pv_h_1 + 4 + 2]), make_float2(win[(u_1 % 4 + 3) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 3) % 4 * 8 + 2 * pv_h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 1], wr[8 * pv_h_1 + 4 + 1]), make_float2(win[(u_1 % 4 + 2) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 2) % 4 * 8 + 2 * pv_h_1 + 1]), mul_f32x2_noftz(make_float2(wr[8 * pv_h_1 + 0], wr[8 * pv_h_1 + 4 + 0]), make_float2(win[(u_1 % 4 + 1) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 1) % 4 * 8 + 2 * pv_h_1 + 1])))));
                            float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
                            float2 pv_e2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_1) : "l"(*(const unsigned long long*)&pv_p2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
                            float _exp2_19 = approx_exp2(pv_e2_1.x);
                            float _exp2_20 = approx_exp2(pv_e2_1.y);
                            float2 pv_den2_1 = add_f32x2_noftz(make_float2(_exp2_19, _exp2_20), make_float2(1.0f, 1.0f));
                            float _rcp_21 = approx_rcp(pv_den2_1.x);
                            float _rcp_22 = approx_rcp(pv_den2_1.y);
                            float2 pv_sg2_1 = make_float2(_rcp_21, _rcp_22);
                            float2 _f32x2_mul_b_23 = fma_f32x2_rn_noftz(pv_p2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_1), make_float2(1.0f, 1.0f));
                            float2 pv_g2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_1) : "l"(*(const unsigned long long*)&pv_sg2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
                            float2 pv_d2_1 = fma_f32x2_rn_noftz(make_float2(dyf[0 + 2 * pv_h_1], dyf[0 + 2 * pv_h_1 + 1]), pv_g2_1, make_float2(wr[32], wr[33]));
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
                            float2 _f32x2_mul_a_24 = make_float2(wr[8 * pv_h_1 + 3], wr[8 * pv_h_1 + 4 + 3]);
                            float2 pv_r2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&pv_d2_1));
                            pend[u_1 % 4 * 8 + 2 * pv_h_1] = pv_r2_1.x;
                            pend[u_1 % 4 * 8 + 2 * pv_h_1 + 1] = pv_r2_1.y;
                        }
                        if (u_1 >= 3) {
                            long long row3_1 = (long long)(tb + (u_1 - 3)) * C64 + c64;
                            unsigned int packed_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(dx + (row3_1))[0] = reinterpret_cast<int4*>(packed_1)[0];
                        }
                    }
                } else {
                    #pragma unroll
                    for (int u_2 = 0; u_2 < 8; u_2++) {
                        if (steps_k > u_2) {
                            int tcur = tb + u_2;
                            if (rows_k_1 > u_2) {
                                __int128_t _smem_raw_25;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_25) : "r"(sx0 + u_2 * RS));
                                pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[0];
                                pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[1];
                                pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[2];
                                pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[3];
                                __int128_t _smem_raw_26;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_26) : "r"(sx0 + 32768 + u_2 * RS));
                                pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[0];
                                pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[1];
                                pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[2];
                                pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[3];
                                #pragma unroll
                                for (int kk_4 = 0; kk_4 < 4; kk_4++) {
                                    win[u_2 % 4 * 8 + 2 * kk_4] = __uint_as_float(pkx[kk_4] << 16);
                                    win[u_2 % 4 * 8 + 2 * kk_4 + 1] = __uint_as_float(pkx[kk_4] & 4294901760u);
                                    dyf[2 * kk_4] = __uint_as_float(pkd[kk_4] << 16);
                                    dyf[2 * kk_4 + 1] = __uint_as_float(pkd[kk_4] & 4294901760u);
                                }
                            }
                            if (tcur >= doc_end) {
                                #pragma unroll
                                for (int m = 1; m < 4; m++) {
                                    int tm = tcur - m;
                                    if (tm >= t0 && tm < t_end && tm >= doc_start) {
                                        long long rowm = (long long)tm * C64 + c64;
                                        unsigned int _packed_2[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_2 % 4 + 4 - m) % 4 * 8)[_lp*2 + 0], (pend + (u_2 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 0]));
                                            _packed_2[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_2 % 4 + 4 - m) % 4 * 8)[_lp*2 + 2], (pend + (u_2 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 2]));
                                            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_2 % 4 + 4 - m) % 4 * 8)[_lp*2 + 4], (pend + (u_2 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 4]));
                                            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_2 % 4 + 4 - m) % 4 * 8)[_lp*2 + 6], (pend + (u_2 % 4 + 4 - m) % 4 * 8)[_lp*2+1 + 6]));
                                            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(dx + (rowm))[0] = reinterpret_cast<int4*>(_packed_2)[0];
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
                                    for (int j_2 = 0; j_2 < 8; j_2++) {
                                        win[(u_2 % 4 + 4 - m_1) % 4 * 8 + j_2] = 0.0f;
                                    }
                                }
                            }
                            if (tcur < S) {
                                if (tcur < t_end) {
                                    #pragma unroll
                                    for (int pv_h_2 = 0; pv_h_2 < 4; pv_h_2++) {
                                        float2 pv_p2_2 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + 3], wr[8 * pv_h_2 + 4 + 3]), make_float2(win[u_2 % 4 * 8 + 2 * pv_h_2], win[u_2 % 4 * 8 + 2 * pv_h_2 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + 2], wr[8 * pv_h_2 + 4 + 2]), make_float2(win[(u_2 % 4 + 3) % 4 * 8 + 2 * pv_h_2], win[(u_2 % 4 + 3) % 4 * 8 + 2 * pv_h_2 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + 1], wr[8 * pv_h_2 + 4 + 1]), make_float2(win[(u_2 % 4 + 2) % 4 * 8 + 2 * pv_h_2], win[(u_2 % 4 + 2) % 4 * 8 + 2 * pv_h_2 + 1]), mul_f32x2_noftz(make_float2(wr[8 * pv_h_2 + 0], wr[8 * pv_h_2 + 4 + 0]), make_float2(win[(u_2 % 4 + 1) % 4 * 8 + 2 * pv_h_2], win[(u_2 % 4 + 1) % 4 * 8 + 2 * pv_h_2 + 1])))));
                                        float2 _f32x2_mul_b_27 = make_float2(-1.442695f, -1.442695f);
                                        float2 pv_e2_2;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_2) : "l"(*(const unsigned long long*)&pv_p2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
                                        float _exp2_28 = approx_exp2(pv_e2_2.x);
                                        float _exp2_29 = approx_exp2(pv_e2_2.y);
                                        float2 pv_den2_2 = add_f32x2_noftz(make_float2(_exp2_28, _exp2_29), make_float2(1.0f, 1.0f));
                                        float _rcp_30 = approx_rcp(pv_den2_2.x);
                                        float _rcp_31 = approx_rcp(pv_den2_2.y);
                                        float2 pv_sg2_2 = make_float2(_rcp_30, _rcp_31);
                                        float2 _f32x2_mul_b_32 = fma_f32x2_rn_noftz(pv_p2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_2), make_float2(1.0f, 1.0f));
                                        float2 pv_g2_2;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_2) : "l"(*(const unsigned long long*)&pv_sg2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
                                        float2 pv_d2_2 = fma_f32x2_rn_noftz(make_float2(dyf[0 + 2 * pv_h_2], dyf[0 + 2 * pv_h_2 + 1]), pv_g2_2, make_float2(wr[32], wr[33]));
                                        dp[2 * pv_h_2] = pv_d2_2.x;
                                        dp[2 * pv_h_2 + 1] = pv_d2_2.y;
                                        #pragma unroll
                                        for (int kk_5 = 0; kk_5 < 4; kk_5++) {
                                            float2 pv_a2_2 = fma_f32x2_rn_noftz(pv_d2_2, make_float2(win[(u_2 % 4 + 1 + kk_5) % 4 * 8 + 2 * pv_h_2], win[(u_2 % 4 + 1 + kk_5) % 4 * 8 + 2 * pv_h_2 + 1]), make_float2(acc[8 * pv_h_2 + kk_5], acc[8 * pv_h_2 + 4 + kk_5]));
                                            acc[8 * pv_h_2 + kk_5] = pv_a2_2.x;
                                            acc[8 * pv_h_2 + 4 + kk_5] = pv_a2_2.y;
                                        }
                                        #pragma unroll
                                        for (int pv_k_2 = 0; pv_k_2 < 3; pv_k_2++) {
                                            float2 pv_q2_2 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + pv_k_2], wr[8 * pv_h_2 + 4 + pv_k_2]), pv_d2_2, make_float2(pend[(u_2 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2], pend[(u_2 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2 + 1]));
                                            pend[(u_2 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2] = pv_q2_2.x;
                                            pend[(u_2 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2 + 1] = pv_q2_2.y;
                                        }
                                        float2 _f32x2_mul_a_33 = make_float2(wr[8 * pv_h_2 + 3], wr[8 * pv_h_2 + 4 + 3]);
                                        float2 pv_r2_2;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&pv_d2_2));
                                        pend[u_2 % 4 * 8 + 2 * pv_h_2] = pv_r2_2.x;
                                        pend[u_2 % 4 * 8 + 2 * pv_h_2 + 1] = pv_r2_2.y;
                                    }
                                } else {
                                    #pragma unroll
                                    for (int pv_h_3 = 0; pv_h_3 < 4; pv_h_3++) {
                                        float2 pv_p2_3 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_3 + 3], wr[8 * pv_h_3 + 4 + 3]), make_float2(win[u_2 % 4 * 8 + 2 * pv_h_3], win[u_2 % 4 * 8 + 2 * pv_h_3 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_3 + 2], wr[8 * pv_h_3 + 4 + 2]), make_float2(win[(u_2 % 4 + 3) % 4 * 8 + 2 * pv_h_3], win[(u_2 % 4 + 3) % 4 * 8 + 2 * pv_h_3 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_3 + 1], wr[8 * pv_h_3 + 4 + 1]), make_float2(win[(u_2 % 4 + 2) % 4 * 8 + 2 * pv_h_3], win[(u_2 % 4 + 2) % 4 * 8 + 2 * pv_h_3 + 1]), mul_f32x2_noftz(make_float2(wr[8 * pv_h_3 + 0], wr[8 * pv_h_3 + 4 + 0]), make_float2(win[(u_2 % 4 + 1) % 4 * 8 + 2 * pv_h_3], win[(u_2 % 4 + 1) % 4 * 8 + 2 * pv_h_3 + 1])))));
                                        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
                                        float2 pv_e2_3;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_3) : "l"(*(const unsigned long long*)&pv_p2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
                                        float _exp2_35 = approx_exp2(pv_e2_3.x);
                                        float _exp2_36 = approx_exp2(pv_e2_3.y);
                                        float2 pv_den2_3 = add_f32x2_noftz(make_float2(_exp2_35, _exp2_36), make_float2(1.0f, 1.0f));
                                        float _rcp_37 = approx_rcp(pv_den2_3.x);
                                        float _rcp_38 = approx_rcp(pv_den2_3.y);
                                        float2 pv_sg2_3 = make_float2(_rcp_37, _rcp_38);
                                        float2 _f32x2_mul_b_39 = fma_f32x2_rn_noftz(pv_p2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_3), make_float2(1.0f, 1.0f));
                                        float2 pv_g2_3;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_3) : "l"(*(const unsigned long long*)&pv_sg2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
                                        float2 pv_d2_3 = fma_f32x2_rn_noftz(make_float2(dyf[0 + 2 * pv_h_3], dyf[0 + 2 * pv_h_3 + 1]), pv_g2_3, make_float2(wr[32], wr[33]));
                                        dp[2 * pv_h_3] = pv_d2_3.x;
                                        dp[2 * pv_h_3 + 1] = pv_d2_3.y;
                                        #pragma unroll
                                        for (int pv_k_3 = 0; pv_k_3 < 3; pv_k_3++) {
                                            float2 pv_q2_3 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_3 + pv_k_3], wr[8 * pv_h_3 + 4 + pv_k_3]), pv_d2_3, make_float2(pend[(u_2 % 4 + 1 + pv_k_3) % 4 * 8 + 2 * pv_h_3], pend[(u_2 % 4 + 1 + pv_k_3) % 4 * 8 + 2 * pv_h_3 + 1]));
                                            pend[(u_2 % 4 + 1 + pv_k_3) % 4 * 8 + 2 * pv_h_3] = pv_q2_3.x;
                                            pend[(u_2 % 4 + 1 + pv_k_3) % 4 * 8 + 2 * pv_h_3 + 1] = pv_q2_3.y;
                                        }
                                        float2 _f32x2_mul_a_40 = make_float2(wr[8 * pv_h_3 + 3], wr[8 * pv_h_3 + 4 + 3]);
                                        float2 pv_r2_3;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&pv_d2_3));
                                        pend[u_2 % 4 * 8 + 2 * pv_h_3] = pv_r2_3.x;
                                        pend[u_2 % 4 * 8 + 2 * pv_h_3 + 1] = pv_r2_3.y;
                                    }
                                }
                                int t3 = tcur - 3;
                                if (t3 >= t0 && t3 >= doc_start) {
                                    long long row3_2 = (long long)t3 * C64 + c64;
                                    unsigned int _packed_3[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_2 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend + (u_2 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                        _packed_3[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_2 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend + (u_2 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                        _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_2 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend + (u_2 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                        _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_2 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend + (u_2 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                        _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<int4*>(dx + (row3_2))[0] = reinterpret_cast<int4*>(_packed_3)[0];
                                }
                            }
                        }
                    }
                }
            }
            if ((k + 1) % 4 == 0) {
                #pragma unroll
                for (int i_3 = 0; i_3 < 32; i_3++) {
                    float s = red[i_3 * 128 + tid];
                    float y;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(acc[i_3]), "f"(redc[i_3 * 128 + tid]));
                    float t;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t) : "f"(s), "f"(y));
                    float _sub_42;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(t), "f"(s));
                    float _sub_41;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(_sub_42), "f"(y));
                    redc[i_3 * 128 + tid] = _sub_41;
                    red[i_3 * 128 + tid] = t;
                    acc[i_3] = 0.0f;
                }
            }
            #pragma unroll
            for (int s_1 = 0; s_1 < 2; s_1++) {
                if (sub == s_1) {
                    asm volatile("barrier.sync %0, %1;" :: "r"(1 + s_1), "r"(ct_vec) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn = k + 2;
            if (kn < n_stages) {
                int GWn_2 = ct_vec >> 5;
                int bslot_0 = sub * 2 + slot;
                int rows_k_2 = 0;
                if (active != 0) {
                    int _max_44 = ((my_rows - (k + 2) * 8) > (0) ? (my_rows - (k + 2) * 8) : (0));
                    int _min_43 = ((_max_44) < (8) ? (_max_44) : (8));
                    rows_k_2 = _min_43;
                }
                if (v == 0 && rows_k_2 > 0) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (sub + 4 * (int)((unsigned int)(k + 2) % 2)) * 8, rows_k_2 * (2 * RS));
                    int sx_1 = (int)stage_buf_addr + (sub * (GWn_2 * 2) + slot * GWn_2) * 4096;
                    long long src0_1 = (long long)(t0 + (k + 2) * 8) * C64 + c64;
                    cp_async_bulk_gmem2smem(sx_1, reinterpret_cast<const uint8_t*>(x + src0_1), rows_k_2 * RS, stage_full_addr + (sub + 4 * (int)((unsigned int)(k + 2) % 2)) * 8);
                    cp_async_bulk_gmem2smem(sx_1 + 32768, reinterpret_cast<const uint8_t*>(dy + src0_1), rows_k_2 * RS, stage_full_addr + (sub + 4 * (int)((unsigned int)(k + 2) % 2)) * 8);
                } else if (v < 32) {
                    mbarrier_arrive(stage_full_addr + (sub + 4 * (int)((unsigned int)(k + 2) % 2)) * 8);
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
            int _min_46 = ((t00 + L) < (S) ? (t00 + L) : (S));
            int _min_45 = ((_min_46 + 3) < (S + 1) ? (_min_46 + 3) : (S + 1));
            steps0 = _min_45 - t00;
        }
        int n_stages_1 = (steps0 + 8 - 1) / 8;
        int _min_1 = ((n_stages_1) < (2) ? (n_stages_1) : (2));
        #pragma unroll 1
        for (int k0_1 = 0; k0_1 < _min_1; k0_1++) {
            int slot0_1 = k0_1;
            int bslot_2 = warp_1 * 2 + slot0_1;
            int rows_k_3 = 0;
            if (active_1 != 0) {
                int _max_48 = ((my_rows_1 - k0_1 * 8) > (0) ? (my_rows_1 - k0_1 * 8) : (0));
                int _min_47 = ((_max_48) < (8) ? (_max_48) : (8));
                rows_k_3 = _min_47;
            }
            if (li == 0 && rows_k_3 > 0) {
                int cb = ((rstride) < ((int)(C64 - c64_1) * 2) ? (rstride) : ((int)(C64 - c64_1) * 2));
                int sx_2 = (int)stage_buf_addr + bslot_2 * 8192 + seg_off;
                long long src0_2 = (long long)(t0 + k0_1 * 8) * C64 + c64_1;
                if (whole_row != 0) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8, rows_k_3 * (2 * cb));
                    cp_async_bulk_gmem2smem(sx_2, reinterpret_cast<const uint8_t*>(x + src0_2), rows_k_3 * cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                    cp_async_bulk_gmem2smem(sx_2 + 4096, reinterpret_cast<const uint8_t*>(dy + src0_2), rows_k_3 * cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                } else if (rstride == 512) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8, 8192);
                    int cc = (int)c64_1;
                    tma_2d_gmem2smem(sx_2, (&x_map), cc, t0 + k0_1 * 8, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                    tma_2d_gmem2smem(sx_2 + 4096, (&dy_map), cc, t0 + k0_1 * 8, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                } else {
                    mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8, rows_k_3 * (2 * cb));
                    #pragma unroll 1
                    for (int i_4 = 0; i_4 < rows_k_3; i_4++) {
                        long long src = src0_2 + (long long)i_4 * C64;
                        cp_async_bulk_gmem2smem(sx_2 + i_4 * rstride, reinterpret_cast<const uint8_t*>(x + src), cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                        cp_async_bulk_gmem2smem(sx_2 + 4096 + i_4 * rstride, reinterpret_cast<const uint8_t*>(dy + src), cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                    }
                }
            } else {
                mbarrier_arrive(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
            }
        }
        float wr_1[34];
        wr_1[32] = __uint_as_float((unsigned int)S >> 31 | 2147483648u);
        wr_1[33] = wr_1[32];
        float w4_1[4];
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
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    {
                        uint2 _vld_49;
                        _vld_49 = *reinterpret_cast<const uint2*>(reinterpret_cast<__nv_bfloat16*>(w) + (c64_1 + (long long)j_3) * 4);
                        uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&wr_1[j_3 * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
                            (&wr_1[j_3 * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(w) + (c64_1 + (long long)j_4) * 4);
                        wr_1[j_4 * 4 + 0] = _v4.x;
                        wr_1[j_4 * 4 + 1] = _v4.y;
                        wr_1[j_4 * 4 + 2] = _v4.z;
                        wr_1[j_4 * 4 + 3] = _v4.w;
                    }
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
                        const uint4* _vptr_51 = reinterpret_cast<const uint4*>(x + rowk_1);
                        uint4 _vld_51[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_51[_blk] = _vptr_51[_blk];
                            uint32_t* _vpairs_51 = reinterpret_cast<uint32_t*>(&_vld_51[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[(kh_1 + 1) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) << 16);
                                (&win_1[(kh_1 + 1) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) & 0xffff0000u);
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
            int bslot_3 = warp_1 * 2 + slot_1;
            int _stage_stage_full_1 = warp_1 + 4 * (int)((unsigned int)k_1 % 2);
            int _parity_stage_full_1 = (int)((unsigned int)k_1 / 2 % 2);
            mbarrier_wait(stage_full_addr + (_stage_stage_full_1) * 8, _parity_stage_full_1);
            if (active_1 != 0) {
                int tb_1 = t0 + k_1 * 8;
                int _max_52 = ((my_rows_1 - k_1 * 8) > (0) ? (my_rows_1 - k_1 * 8) : (0));
                int rows_k_4 = ((_max_52) < (8) ? (_max_52) : (8));
                int _max_53 = ((my_steps_1 - k_1 * 8) > (0) ? (my_steps_1 - k_1 * 8) : (0));
                int steps_k_1 = ((_max_53) < (8) ? (_max_53) : (8));
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
                        __int128_t _smem_raw_54;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_54) : "r"(sx0_1 + u_3 * rstride));
                        pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_54)[0];
                        pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_54)[1];
                        pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_54)[2];
                        pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_54)[3];
                        __int128_t _smem_raw_55;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_55) : "r"(sx0_1 + 4096 + u_3 * rstride));
                        pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_55)[0];
                        pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_55)[1];
                        pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_55)[2];
                        pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_55)[3];
                        #pragma unroll
                        for (int kk_6 = 0; kk_6 < 4; kk_6++) {
                            win_1[u_3 % 4 * 8 + 2 * kk_6] = __uint_as_float(pkx_1[kk_6] << 16);
                            win_1[u_3 % 4 * 8 + 2 * kk_6 + 1] = __uint_as_float(pkx_1[kk_6] & 4294901760u);
                            dyf_1[2 * kk_6] = __uint_as_float(pkd_1[kk_6] << 16);
                            dyf_1[2 * kk_6 + 1] = __uint_as_float(pkd_1[kk_6] & 4294901760u);
                        }
                        #pragma unroll
                        for (int pv_h_4 = 0; pv_h_4 < 4; pv_h_4++) {
                            float2 pv_p2_4 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_4 + 3], wr_1[8 * pv_h_4 + 4 + 3]), make_float2(win_1[u_3 % 4 * 8 + 2 * pv_h_4], win_1[u_3 % 4 * 8 + 2 * pv_h_4 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_4 + 2], wr_1[8 * pv_h_4 + 4 + 2]), make_float2(win_1[(u_3 % 4 + 3) % 4 * 8 + 2 * pv_h_4], win_1[(u_3 % 4 + 3) % 4 * 8 + 2 * pv_h_4 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_4 + 1], wr_1[8 * pv_h_4 + 4 + 1]), make_float2(win_1[(u_3 % 4 + 2) % 4 * 8 + 2 * pv_h_4], win_1[(u_3 % 4 + 2) % 4 * 8 + 2 * pv_h_4 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * pv_h_4 + 0], wr_1[8 * pv_h_4 + 4 + 0]), make_float2(win_1[(u_3 % 4 + 1) % 4 * 8 + 2 * pv_h_4], win_1[(u_3 % 4 + 1) % 4 * 8 + 2 * pv_h_4 + 1])))));
                            float2 _f32x2_mul_b_56 = make_float2(-1.442695f, -1.442695f);
                            float2 pv_e2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_4) : "l"(*(const unsigned long long*)&pv_p2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
                            float _exp2_57 = approx_exp2(pv_e2_4.x);
                            float _exp2_58 = approx_exp2(pv_e2_4.y);
                            float2 pv_den2_4 = add_f32x2_noftz(make_float2(_exp2_57, _exp2_58), make_float2(1.0f, 1.0f));
                            float _rcp_59 = approx_rcp(pv_den2_4.x);
                            float _rcp_60 = approx_rcp(pv_den2_4.y);
                            float2 pv_sg2_4 = make_float2(_rcp_59, _rcp_60);
                            float2 _f32x2_mul_b_61 = fma_f32x2_rn_noftz(pv_p2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_4), make_float2(1.0f, 1.0f));
                            float2 pv_g2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_4) : "l"(*(const unsigned long long*)&pv_sg2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
                            float2 pv_d2_4 = fma_f32x2_rn_noftz(make_float2(dyf_1[0 + 2 * pv_h_4], dyf_1[0 + 2 * pv_h_4 + 1]), pv_g2_4, make_float2(wr_1[32], wr_1[33]));
                            dp_1[2 * pv_h_4] = pv_d2_4.x;
                            dp_1[2 * pv_h_4 + 1] = pv_d2_4.y;
                            #pragma unroll
                            for (int kk_7 = 0; kk_7 < 4; kk_7++) {
                                float2 pv_a2_3 = fma_f32x2_rn_noftz(pv_d2_4, make_float2(win_1[(u_3 % 4 + 1 + kk_7) % 4 * 8 + 2 * pv_h_4], win_1[(u_3 % 4 + 1 + kk_7) % 4 * 8 + 2 * pv_h_4 + 1]), make_float2(acc[8 * pv_h_4 + kk_7], acc[8 * pv_h_4 + 4 + kk_7]));
                                acc[8 * pv_h_4 + kk_7] = pv_a2_3.x;
                                acc[8 * pv_h_4 + 4 + kk_7] = pv_a2_3.y;
                            }
                            #pragma unroll
                            for (int pv_k_4 = 0; pv_k_4 < 3; pv_k_4++) {
                                float2 pv_q2_4 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_4 + pv_k_4], wr_1[8 * pv_h_4 + 4 + pv_k_4]), pv_d2_4, make_float2(pend_1[(u_3 % 4 + 1 + pv_k_4) % 4 * 8 + 2 * pv_h_4], pend_1[(u_3 % 4 + 1 + pv_k_4) % 4 * 8 + 2 * pv_h_4 + 1]));
                                pend_1[(u_3 % 4 + 1 + pv_k_4) % 4 * 8 + 2 * pv_h_4] = pv_q2_4.x;
                                pend_1[(u_3 % 4 + 1 + pv_k_4) % 4 * 8 + 2 * pv_h_4 + 1] = pv_q2_4.y;
                            }
                            float2 _f32x2_mul_a_62 = make_float2(wr_1[8 * pv_h_4 + 3], wr_1[8 * pv_h_4 + 4 + 3]);
                            float2 pv_r2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&pv_d2_4));
                            pend_1[u_3 % 4 * 8 + 2 * pv_h_4] = pv_r2_4.x;
                            pend_1[u_3 % 4 * 8 + 2 * pv_h_4 + 1] = pv_r2_4.y;
                        }
                        if (u_3 >= 0) {
                            long long row3_3 = (long long)(tb_1 + (u_3 - 3)) * C64 + c64_1;
                            unsigned int packed_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                packed_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend_1 + (u_3 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(dx + (row3_3))[0] = reinterpret_cast<int4*>(packed_2)[0];
                        }
                    }
                } else if (interior_1 == 1) {
                    #pragma unroll
                    for (int u_4 = 0; u_4 < 8; u_4++) {
                        __int128_t _smem_raw_63;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_63) : "r"(sx0_1 + u_4 * rstride));
                        pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[0];
                        pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[1];
                        pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[2];
                        pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[3];
                        __int128_t _smem_raw_64;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_64) : "r"(sx0_1 + 4096 + u_4 * rstride));
                        pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[0];
                        pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[1];
                        pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[2];
                        pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_64)[3];
                        #pragma unroll
                        for (int kk_8 = 0; kk_8 < 4; kk_8++) {
                            win_1[u_4 % 4 * 8 + 2 * kk_8] = __uint_as_float(pkx_1[kk_8] << 16);
                            win_1[u_4 % 4 * 8 + 2 * kk_8 + 1] = __uint_as_float(pkx_1[kk_8] & 4294901760u);
                            dyf_1[2 * kk_8] = __uint_as_float(pkd_1[kk_8] << 16);
                            dyf_1[2 * kk_8 + 1] = __uint_as_float(pkd_1[kk_8] & 4294901760u);
                        }
                        #pragma unroll
                        for (int pv_h_5 = 0; pv_h_5 < 4; pv_h_5++) {
                            float2 pv_p2_5 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_5 + 3], wr_1[8 * pv_h_5 + 4 + 3]), make_float2(win_1[u_4 % 4 * 8 + 2 * pv_h_5], win_1[u_4 % 4 * 8 + 2 * pv_h_5 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_5 + 2], wr_1[8 * pv_h_5 + 4 + 2]), make_float2(win_1[(u_4 % 4 + 3) % 4 * 8 + 2 * pv_h_5], win_1[(u_4 % 4 + 3) % 4 * 8 + 2 * pv_h_5 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_5 + 1], wr_1[8 * pv_h_5 + 4 + 1]), make_float2(win_1[(u_4 % 4 + 2) % 4 * 8 + 2 * pv_h_5], win_1[(u_4 % 4 + 2) % 4 * 8 + 2 * pv_h_5 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * pv_h_5 + 0], wr_1[8 * pv_h_5 + 4 + 0]), make_float2(win_1[(u_4 % 4 + 1) % 4 * 8 + 2 * pv_h_5], win_1[(u_4 % 4 + 1) % 4 * 8 + 2 * pv_h_5 + 1])))));
                            float2 _f32x2_mul_b_65 = make_float2(-1.442695f, -1.442695f);
                            float2 pv_e2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_5) : "l"(*(const unsigned long long*)&pv_p2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
                            float _exp2_66 = approx_exp2(pv_e2_5.x);
                            float _exp2_67 = approx_exp2(pv_e2_5.y);
                            float2 pv_den2_5 = add_f32x2_noftz(make_float2(_exp2_66, _exp2_67), make_float2(1.0f, 1.0f));
                            float _rcp_68 = approx_rcp(pv_den2_5.x);
                            float _rcp_69 = approx_rcp(pv_den2_5.y);
                            float2 pv_sg2_5 = make_float2(_rcp_68, _rcp_69);
                            float2 _f32x2_mul_b_70 = fma_f32x2_rn_noftz(pv_p2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_5), make_float2(1.0f, 1.0f));
                            float2 pv_g2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_5) : "l"(*(const unsigned long long*)&pv_sg2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
                            float2 pv_d2_5 = fma_f32x2_rn_noftz(make_float2(dyf_1[0 + 2 * pv_h_5], dyf_1[0 + 2 * pv_h_5 + 1]), pv_g2_5, make_float2(wr_1[32], wr_1[33]));
                            dp_1[2 * pv_h_5] = pv_d2_5.x;
                            dp_1[2 * pv_h_5 + 1] = pv_d2_5.y;
                            #pragma unroll
                            for (int kk_9 = 0; kk_9 < 4; kk_9++) {
                                float2 pv_a2_4 = fma_f32x2_rn_noftz(pv_d2_5, make_float2(win_1[(u_4 % 4 + 1 + kk_9) % 4 * 8 + 2 * pv_h_5], win_1[(u_4 % 4 + 1 + kk_9) % 4 * 8 + 2 * pv_h_5 + 1]), make_float2(acc[8 * pv_h_5 + kk_9], acc[8 * pv_h_5 + 4 + kk_9]));
                                acc[8 * pv_h_5 + kk_9] = pv_a2_4.x;
                                acc[8 * pv_h_5 + 4 + kk_9] = pv_a2_4.y;
                            }
                            #pragma unroll
                            for (int pv_k_5 = 0; pv_k_5 < 3; pv_k_5++) {
                                float2 pv_q2_5 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_5 + pv_k_5], wr_1[8 * pv_h_5 + 4 + pv_k_5]), pv_d2_5, make_float2(pend_1[(u_4 % 4 + 1 + pv_k_5) % 4 * 8 + 2 * pv_h_5], pend_1[(u_4 % 4 + 1 + pv_k_5) % 4 * 8 + 2 * pv_h_5 + 1]));
                                pend_1[(u_4 % 4 + 1 + pv_k_5) % 4 * 8 + 2 * pv_h_5] = pv_q2_5.x;
                                pend_1[(u_4 % 4 + 1 + pv_k_5) % 4 * 8 + 2 * pv_h_5 + 1] = pv_q2_5.y;
                            }
                            float2 _f32x2_mul_a_71 = make_float2(wr_1[8 * pv_h_5 + 3], wr_1[8 * pv_h_5 + 4 + 3]);
                            float2 pv_r2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&pv_d2_5));
                            pend_1[u_4 % 4 * 8 + 2 * pv_h_5] = pv_r2_5.x;
                            pend_1[u_4 % 4 * 8 + 2 * pv_h_5 + 1] = pv_r2_5.y;
                        }
                        if (u_4 >= 3) {
                            long long row3_4 = (long long)(tb_1 + (u_4 - 3)) * C64 + c64_1;
                            unsigned int packed_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_4 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend_1 + (u_4 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                packed_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_4 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend_1 + (u_4 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_4 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend_1 + (u_4 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_4 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend_1 + (u_4 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(dx + (row3_4))[0] = reinterpret_cast<int4*>(packed_3)[0];
                        }
                    }
                } else {
                    #pragma unroll
                    for (int u_5 = 0; u_5 < 8; u_5++) {
                        if (steps_k_1 > u_5) {
                            int tcur_1 = tb_1 + u_5;
                            if (rows_k_4 > u_5) {
                                __int128_t _smem_raw_72;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_72) : "r"(sx0_1 + u_5 * rstride));
                                pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[0];
                                pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[1];
                                pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[2];
                                pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[3];
                                __int128_t _smem_raw_73;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_73) : "r"(sx0_1 + 4096 + u_5 * rstride));
                                pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_73)[0];
                                pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_73)[1];
                                pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_73)[2];
                                pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_73)[3];
                                #pragma unroll
                                for (int kk_10 = 0; kk_10 < 4; kk_10++) {
                                    win_1[u_5 % 4 * 8 + 2 * kk_10] = __uint_as_float(pkx_1[kk_10] << 16);
                                    win_1[u_5 % 4 * 8 + 2 * kk_10 + 1] = __uint_as_float(pkx_1[kk_10] & 4294901760u);
                                    dyf_1[2 * kk_10] = __uint_as_float(pkd_1[kk_10] << 16);
                                    dyf_1[2 * kk_10 + 1] = __uint_as_float(pkd_1[kk_10] & 4294901760u);
                                }
                            }
                            if (tcur_1 >= doc_end_1) {
                                #pragma unroll
                                for (int m_2 = 1; m_2 < 4; m_2++) {
                                    int tm_1 = tcur_1 - m_2;
                                    if (tm_1 >= t0 && tm_1 < t_end_1 && tm_1 >= doc_start_1) {
                                        long long rowm_1 = (long long)tm_1 * C64 + c64_1;
                                        unsigned int _packed_6[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_5 % 4 + 4 - m_2) % 4 * 8)[_lp*2 + 0], (pend_1 + (u_5 % 4 + 4 - m_2) % 4 * 8)[_lp*2+1 + 0]));
                                            _packed_6[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_5 % 4 + 4 - m_2) % 4 * 8)[_lp*2 + 2], (pend_1 + (u_5 % 4 + 4 - m_2) % 4 * 8)[_lp*2+1 + 2]));
                                            _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_5 % 4 + 4 - m_2) % 4 * 8)[_lp*2 + 4], (pend_1 + (u_5 % 4 + 4 - m_2) % 4 * 8)[_lp*2+1 + 4]));
                                            _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                                        }
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_5 % 4 + 4 - m_2) % 4 * 8)[_lp*2 + 6], (pend_1 + (u_5 % 4 + 4 - m_2) % 4 * 8)[_lp*2+1 + 6]));
                                            _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(dx + (rowm_1))[0] = reinterpret_cast<int4*>(_packed_6)[0];
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
                                    for (int j_5 = 0; j_5 < 8; j_5++) {
                                        win_1[(u_5 % 4 + 4 - m_3) % 4 * 8 + j_5] = 0.0f;
                                    }
                                }
                            }
                            if (tcur_1 < S) {
                                if (tcur_1 < t_end_1) {
                                    #pragma unroll
                                    for (int pv_h_6 = 0; pv_h_6 < 4; pv_h_6++) {
                                        float2 pv_p2_6 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_6 + 3], wr_1[8 * pv_h_6 + 4 + 3]), make_float2(win_1[u_5 % 4 * 8 + 2 * pv_h_6], win_1[u_5 % 4 * 8 + 2 * pv_h_6 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_6 + 2], wr_1[8 * pv_h_6 + 4 + 2]), make_float2(win_1[(u_5 % 4 + 3) % 4 * 8 + 2 * pv_h_6], win_1[(u_5 % 4 + 3) % 4 * 8 + 2 * pv_h_6 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_6 + 1], wr_1[8 * pv_h_6 + 4 + 1]), make_float2(win_1[(u_5 % 4 + 2) % 4 * 8 + 2 * pv_h_6], win_1[(u_5 % 4 + 2) % 4 * 8 + 2 * pv_h_6 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * pv_h_6 + 0], wr_1[8 * pv_h_6 + 4 + 0]), make_float2(win_1[(u_5 % 4 + 1) % 4 * 8 + 2 * pv_h_6], win_1[(u_5 % 4 + 1) % 4 * 8 + 2 * pv_h_6 + 1])))));
                                        float2 _f32x2_mul_b_74 = make_float2(-1.442695f, -1.442695f);
                                        float2 pv_e2_6;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_6) : "l"(*(const unsigned long long*)&pv_p2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_74));
                                        float _exp2_75 = approx_exp2(pv_e2_6.x);
                                        float _exp2_76 = approx_exp2(pv_e2_6.y);
                                        float2 pv_den2_6 = add_f32x2_noftz(make_float2(_exp2_75, _exp2_76), make_float2(1.0f, 1.0f));
                                        float _rcp_77 = approx_rcp(pv_den2_6.x);
                                        float _rcp_78 = approx_rcp(pv_den2_6.y);
                                        float2 pv_sg2_6 = make_float2(_rcp_77, _rcp_78);
                                        float2 _f32x2_mul_b_79 = fma_f32x2_rn_noftz(pv_p2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_6), make_float2(1.0f, 1.0f));
                                        float2 pv_g2_6;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_6) : "l"(*(const unsigned long long*)&pv_sg2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
                                        float2 pv_d2_6 = fma_f32x2_rn_noftz(make_float2(dyf_1[0 + 2 * pv_h_6], dyf_1[0 + 2 * pv_h_6 + 1]), pv_g2_6, make_float2(wr_1[32], wr_1[33]));
                                        dp_1[2 * pv_h_6] = pv_d2_6.x;
                                        dp_1[2 * pv_h_6 + 1] = pv_d2_6.y;
                                        #pragma unroll
                                        for (int kk_11 = 0; kk_11 < 4; kk_11++) {
                                            float2 pv_a2_5 = fma_f32x2_rn_noftz(pv_d2_6, make_float2(win_1[(u_5 % 4 + 1 + kk_11) % 4 * 8 + 2 * pv_h_6], win_1[(u_5 % 4 + 1 + kk_11) % 4 * 8 + 2 * pv_h_6 + 1]), make_float2(acc[8 * pv_h_6 + kk_11], acc[8 * pv_h_6 + 4 + kk_11]));
                                            acc[8 * pv_h_6 + kk_11] = pv_a2_5.x;
                                            acc[8 * pv_h_6 + 4 + kk_11] = pv_a2_5.y;
                                        }
                                        #pragma unroll
                                        for (int pv_k_6 = 0; pv_k_6 < 3; pv_k_6++) {
                                            float2 pv_q2_6 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_6 + pv_k_6], wr_1[8 * pv_h_6 + 4 + pv_k_6]), pv_d2_6, make_float2(pend_1[(u_5 % 4 + 1 + pv_k_6) % 4 * 8 + 2 * pv_h_6], pend_1[(u_5 % 4 + 1 + pv_k_6) % 4 * 8 + 2 * pv_h_6 + 1]));
                                            pend_1[(u_5 % 4 + 1 + pv_k_6) % 4 * 8 + 2 * pv_h_6] = pv_q2_6.x;
                                            pend_1[(u_5 % 4 + 1 + pv_k_6) % 4 * 8 + 2 * pv_h_6 + 1] = pv_q2_6.y;
                                        }
                                        float2 _f32x2_mul_a_80 = make_float2(wr_1[8 * pv_h_6 + 3], wr_1[8 * pv_h_6 + 4 + 3]);
                                        float2 pv_r2_6;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&pv_d2_6));
                                        pend_1[u_5 % 4 * 8 + 2 * pv_h_6] = pv_r2_6.x;
                                        pend_1[u_5 % 4 * 8 + 2 * pv_h_6 + 1] = pv_r2_6.y;
                                    }
                                } else {
                                    #pragma unroll
                                    for (int pv_h_7 = 0; pv_h_7 < 4; pv_h_7++) {
                                        float2 pv_p2_7 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_7 + 3], wr_1[8 * pv_h_7 + 4 + 3]), make_float2(win_1[u_5 % 4 * 8 + 2 * pv_h_7], win_1[u_5 % 4 * 8 + 2 * pv_h_7 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_7 + 2], wr_1[8 * pv_h_7 + 4 + 2]), make_float2(win_1[(u_5 % 4 + 3) % 4 * 8 + 2 * pv_h_7], win_1[(u_5 % 4 + 3) % 4 * 8 + 2 * pv_h_7 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_7 + 1], wr_1[8 * pv_h_7 + 4 + 1]), make_float2(win_1[(u_5 % 4 + 2) % 4 * 8 + 2 * pv_h_7], win_1[(u_5 % 4 + 2) % 4 * 8 + 2 * pv_h_7 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * pv_h_7 + 0], wr_1[8 * pv_h_7 + 4 + 0]), make_float2(win_1[(u_5 % 4 + 1) % 4 * 8 + 2 * pv_h_7], win_1[(u_5 % 4 + 1) % 4 * 8 + 2 * pv_h_7 + 1])))));
                                        float2 _f32x2_mul_b_81 = make_float2(-1.442695f, -1.442695f);
                                        float2 pv_e2_7;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_7) : "l"(*(const unsigned long long*)&pv_p2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
                                        float _exp2_82 = approx_exp2(pv_e2_7.x);
                                        float _exp2_83 = approx_exp2(pv_e2_7.y);
                                        float2 pv_den2_7 = add_f32x2_noftz(make_float2(_exp2_82, _exp2_83), make_float2(1.0f, 1.0f));
                                        float _rcp_84 = approx_rcp(pv_den2_7.x);
                                        float _rcp_85 = approx_rcp(pv_den2_7.y);
                                        float2 pv_sg2_7 = make_float2(_rcp_84, _rcp_85);
                                        float2 _f32x2_mul_b_86 = fma_f32x2_rn_noftz(pv_p2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_7), make_float2(1.0f, 1.0f));
                                        float2 pv_g2_7;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_7) : "l"(*(const unsigned long long*)&pv_sg2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                                        float2 pv_d2_7 = fma_f32x2_rn_noftz(make_float2(dyf_1[0 + 2 * pv_h_7], dyf_1[0 + 2 * pv_h_7 + 1]), pv_g2_7, make_float2(wr_1[32], wr_1[33]));
                                        dp_1[2 * pv_h_7] = pv_d2_7.x;
                                        dp_1[2 * pv_h_7 + 1] = pv_d2_7.y;
                                        #pragma unroll
                                        for (int pv_k_7 = 0; pv_k_7 < 3; pv_k_7++) {
                                            float2 pv_q2_7 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_7 + pv_k_7], wr_1[8 * pv_h_7 + 4 + pv_k_7]), pv_d2_7, make_float2(pend_1[(u_5 % 4 + 1 + pv_k_7) % 4 * 8 + 2 * pv_h_7], pend_1[(u_5 % 4 + 1 + pv_k_7) % 4 * 8 + 2 * pv_h_7 + 1]));
                                            pend_1[(u_5 % 4 + 1 + pv_k_7) % 4 * 8 + 2 * pv_h_7] = pv_q2_7.x;
                                            pend_1[(u_5 % 4 + 1 + pv_k_7) % 4 * 8 + 2 * pv_h_7 + 1] = pv_q2_7.y;
                                        }
                                        float2 _f32x2_mul_a_87 = make_float2(wr_1[8 * pv_h_7 + 3], wr_1[8 * pv_h_7 + 4 + 3]);
                                        float2 pv_r2_7;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&pv_d2_7));
                                        pend_1[u_5 % 4 * 8 + 2 * pv_h_7] = pv_r2_7.x;
                                        pend_1[u_5 % 4 * 8 + 2 * pv_h_7 + 1] = pv_r2_7.y;
                                    }
                                }
                                int t3_1 = tcur_1 - 3;
                                if (t3_1 >= t0 && t3_1 >= doc_start_1) {
                                    long long row3_5 = (long long)t3_1 * C64 + c64_1;
                                    unsigned int _packed_7[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_5 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend_1 + (u_5 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                        _packed_7[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_5 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend_1 + (u_5 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                        _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_5 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend_1 + (u_5 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                        _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend_1 + (u_5 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend_1 + (u_5 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                        _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<int4*>(dx + (row3_5))[0] = reinterpret_cast<int4*>(_packed_7)[0];
                                }
                            }
                        }
                    }
                }
            }
            if ((k_1 + 1) % 4 == 0) {
                #pragma unroll
                for (int i_6 = 0; i_6 < 32; i_6++) {
                    float s_2 = red[i_6 * 128 + tid];
                    float y_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_1) : "f"(acc[i_6]), "f"(redc[i_6 * 128 + tid]));
                    float t_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_1) : "f"(s_2), "f"(y_1));
                    float _sub_89;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_89) : "f"(t_1), "f"(s_2));
                    float _sub_88;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(_sub_89), "f"(y_1));
                    redc[i_6 * 128 + tid] = _sub_88;
                    red[i_6 * 128 + tid] = t_1;
                    acc[i_6] = 0.0f;
                }
            }
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn_1 = k_1 + 2;
            if (kn_1 < n_stages_1) {
                int bslot_1_1 = warp_1 * 2 + slot_1;
                int rows_k_5 = 0;
                if (active_1 != 0) {
                    int _max_91 = ((my_rows_1 - (k_1 + 2) * 8) > (0) ? (my_rows_1 - (k_1 + 2) * 8) : (0));
                    int _min_90 = ((_max_91) < (8) ? (_max_91) : (8));
                    rows_k_5 = _min_90;
                }
                if (li == 0 && rows_k_5 > 0) {
                    int cb_1 = ((rstride) < ((int)(C64 - c64_1) * 2) ? (rstride) : ((int)(C64 - c64_1) * 2));
                    int sx_3 = (int)stage_buf_addr + bslot_1_1 * 8192 + seg_off;
                    long long src0_3 = (long long)(t0 + (k_1 + 2) * 8) * C64 + c64_1;
                    if (whole_row != 0) {
                        mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8, rows_k_5 * (2 * cb_1));
                        cp_async_bulk_gmem2smem(sx_3, reinterpret_cast<const uint8_t*>(x + src0_3), rows_k_5 * cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                        cp_async_bulk_gmem2smem(sx_3 + 4096, reinterpret_cast<const uint8_t*>(dy + src0_3), rows_k_5 * cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                    } else if (rstride == 512) {
                        mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8, 8192);
                        int cc_1 = (int)c64_1;
                        tma_2d_gmem2smem(sx_3, (&x_map), cc_1, t0 + (k_1 + 2) * 8, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                        tma_2d_gmem2smem(sx_3 + 4096, (&dy_map), cc_1, t0 + (k_1 + 2) * 8, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                    } else {
                        mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8, rows_k_5 * (2 * cb_1));
                        #pragma unroll 1
                        for (int i_7 = 0; i_7 < rows_k_5; i_7++) {
                            long long src_1 = src0_3 + (long long)i_7 * C64;
                            cp_async_bulk_gmem2smem(sx_3 + i_7 * rstride, reinterpret_cast<const uint8_t*>(x + src_1), cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                            cp_async_bulk_gmem2smem(sx_3 + 4096 + i_7 * rstride, reinterpret_cast<const uint8_t*>(dy + src_1), cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                        }
                    }
                } else {
                    mbarrier_arrive(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                }
            }
        }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 32; i_8++) {
        float s_3 = red[i_8 * 128 + tid];
        float y_2;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_2) : "f"(acc[i_8]), "f"(redc[i_8 * 128 + tid]));
        float t_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(t_2) : "f"(s_3), "f"(y_2));
        float _sub_93;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(t_2), "f"(s_3));
        float _sub_92;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_92) : "f"(_sub_93), "f"(y_2));
        redc[i_8 * 128 + tid] = _sub_92;
        red[i_8 * 128 + tid] = t_2;
        acc[i_8] = 0.0f;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; i_9++) {
        float _sub_94;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(red[i_9 * 128 + tid]), "f"(redc[i_9 * 128 + tid]));
        red[i_9 * 128 + tid] = _sub_94;
    }
    __syncthreads();
    int c0_0 = (ct * ct_vec + v) * 8;
    if (sub == 0 && c0_0 < C) {
        float tot[32];
        #pragma unroll
        for (int i_10 = 0; i_10 < 32; i_10++) {
            tot[i_10] = red[i_10 * 128 + v];
        }
        #pragma unroll 1
        for (int s_4 = 1; s_4 < n_sub; s_4++) {
            #pragma unroll
            for (int i_11 = 0; i_11 < 32; i_11++) {
                tot[i_11] = tot[i_11] + red[i_11 * 128 + (s_4 * ct_vec + v)];
            }
        }
        #pragma unroll
        for (int j_6 = 0; j_6 < 8; j_6++) {
            long long pidx = ((long long)rank * C64 + (long long)c0_0 + (long long)j_6) * 4;
            {
                float4 _v4 = make_float4(tot[j_6 * 4 + 0], tot[j_6 * 4 + 1], tot[j_6 * 4 + 2], tot[j_6 * 4 + 3]);
                *reinterpret_cast<float4*>(partial + pidx) = _v4;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Cleanup
    __syncthreads();
}

} // extern "C"
