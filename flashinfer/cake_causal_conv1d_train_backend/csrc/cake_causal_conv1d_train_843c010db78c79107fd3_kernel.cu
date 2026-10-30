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
#define SMEM_STAGE_BUF_STAGE_BYTES 32768
#define SMEM_STAGE_BUF_STRIDE 32768
#define SMEM_TOTAL 74752
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
kernel_cake_causal_conv1d_train_843c010db78c79107fd3(__nv_bfloat16* __restrict__ x, void* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ partial, void* __restrict__ b, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles, int flags)
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
        int n_stages = (my_steps + 4 - 1) / 4;
        int _min_0 = ((n_stages) < (2) ? (n_stages) : (2));
        #pragma unroll 1
        for (int k0 = 0; k0 < _min_0; k0++) {
            int slot0 = k0;
            int GWn = ct_vec >> 5;
            int bslot = sub * 2 + slot0;
            int rows_k = 0;
            if (active != 0) {
                int _max_1 = ((my_rows - k0 * 4) > (0) ? (my_rows - k0 * 4) : (0));
                int _min_0_1 = ((_max_1) < (4) ? (_max_1) : (4));
                rows_k = _min_0_1;
            }
            if (v == 0 && rows_k > 0) {
                mbarrier_arrive_expect_tx(stage_full_addr + (sub + 4 * (int)((unsigned int)k0 % 2)) * 8, rows_k * (2 * RS));
                int sx = (int)stage_buf_addr + (sub * (GWn * 2) + slot0 * GWn) * 2048;
                long long src0 = (long long)(t0 + k0 * 4) * C64 + c64;
                cp_async_bulk_gmem2smem(sx, reinterpret_cast<const uint8_t*>(x + src0), rows_k * RS, stage_full_addr + (sub + 4 * (int)((unsigned int)k0 % 2)) * 8);
                cp_async_bulk_gmem2smem(sx + 16384, reinterpret_cast<const uint8_t*>(dy + src0), rows_k * RS, stage_full_addr + (sub + 4 * (int)((unsigned int)k0 % 2)) * 8);
            } else if (v < 32) {
                mbarrier_arrive(stage_full_addr + (sub + 4 * (int)((unsigned int)k0 % 2)) * 8);
            }
        }
        float wr[34];
        wr[32] = __uint_as_float((unsigned int)S >> 31 | 2147483648u);
        wr[33] = wr[32];
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
            int bslot_1 = sub * 2 + slot;
            int _stage_stage_full = sub + 4 * (int)((unsigned int)k % 2);
            int _parity_stage_full = (int)((unsigned int)k / 2 % 2);
            mbarrier_wait(stage_full_addr + (_stage_stage_full) * 8, _parity_stage_full);
            if (active != 0) {
                int tb = t0 + k * 4;
                int _max_8 = ((my_rows - k * 4) > (0) ? (my_rows - k * 4) : (0));
                int rows_k_1 = ((_max_8) < (4) ? (_max_8) : (4));
                int _max_9 = ((my_steps - k * 4) > (0) ? (my_steps - k * 4) : (0));
                int steps_k = ((_max_9) < (4) ? (_max_9) : (4));
                int GWn_1 = ct_vec >> 5;
                int sx0 = (int)stage_buf_addr + (sub * (GWn_1 * 2) + slot * GWn_1) * 2048 + v * 16;
                int interior = 0;
                if (steps_k == 4 && rows_k_1 == 4 && t_end >= tb + 4 && doc_end >= tb + 4) {
                    if (k >= 1) {
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
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(sx0 + 16384 + u * RS));
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
                            float2 pv_p2 = add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 3], wr[8 * pv_h + 4 + 3]), make_float2(win[u % 4 * 8 + 2 * pv_h], win[u % 4 * 8 + 2 * pv_h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 2], wr[8 * pv_h + 4 + 2]), make_float2(win[(u % 4 + 3) % 4 * 8 + 2 * pv_h], win[(u % 4 + 3) % 4 * 8 + 2 * pv_h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h + 1], wr[8 * pv_h + 4 + 1]), make_float2(win[(u % 4 + 2) % 4 * 8 + 2 * pv_h], win[(u % 4 + 2) % 4 * 8 + 2 * pv_h + 1]), mul_f32x2_noftz(make_float2(wr[8 * pv_h + 0], wr[8 * pv_h + 4 + 0]), make_float2(win[(u % 4 + 1) % 4 * 8 + 2 * pv_h], win[(u % 4 + 1) % 4 * 8 + 2 * pv_h + 1]))))), make_float2(br[0 + 2 * pv_h], br[0 + 2 * pv_h + 1]));
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
                            float2 pv_d2 = fma_f32x2_rn_noftz(make_float2(dyf[0 + 2 * pv_h], dyf[0 + 2 * pv_h + 1]), pv_g2, make_float2(wr[32], wr[33]));
                            dp[2 * pv_h] = pv_d2.x;
                            dp[2 * pv_h + 1] = pv_d2.y;
                            #pragma unroll
                            for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                                float2 pv_a2 = fma_f32x2_rn_noftz(pv_d2, make_float2(win[(u % 4 + 1 + kk_1) % 4 * 8 + 2 * pv_h], win[(u % 4 + 1 + kk_1) % 4 * 8 + 2 * pv_h + 1]), make_float2(acc[8 * pv_h + kk_1], acc[8 * pv_h + 4 + kk_1]));
                                acc[8 * pv_h + kk_1] = pv_a2.x;
                                acc[8 * pv_h + 4 + kk_1] = pv_a2.y;
                            }
                            float2 pv_b2 = add_f32x2_noftz(make_float2(accb[0 + 2 * pv_h], accb[0 + 2 * pv_h + 1]), pv_d2);
                            accb[2 * pv_h] = pv_b2.x;
                            accb[2 * pv_h + 1] = pv_b2.y;
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
                } else if (interior == 1) {
                    #pragma unroll
                    for (int u_1 = 0; u_1 < 4; u_1++) {
                        __int128_t _smem_raw_19;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_19) : "r"(sx0 + u_1 * RS));
                        pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[0];
                        pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[1];
                        pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[2];
                        pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[3];
                        __int128_t _smem_raw_20;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_20) : "r"(sx0 + 16384 + u_1 * RS));
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
                        #pragma unroll
                        for (int pv_h_1 = 0; pv_h_1 < 4; pv_h_1++) {
                            float2 pv_p2_1 = add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 3], wr[8 * pv_h_1 + 4 + 3]), make_float2(win[u_1 % 4 * 8 + 2 * pv_h_1], win[u_1 % 4 * 8 + 2 * pv_h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 2], wr[8 * pv_h_1 + 4 + 2]), make_float2(win[(u_1 % 4 + 3) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 3) % 4 * 8 + 2 * pv_h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_1 + 1], wr[8 * pv_h_1 + 4 + 1]), make_float2(win[(u_1 % 4 + 2) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 2) % 4 * 8 + 2 * pv_h_1 + 1]), mul_f32x2_noftz(make_float2(wr[8 * pv_h_1 + 0], wr[8 * pv_h_1 + 4 + 0]), make_float2(win[(u_1 % 4 + 1) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 1) % 4 * 8 + 2 * pv_h_1 + 1]))))), make_float2(br[0 + 2 * pv_h_1], br[0 + 2 * pv_h_1 + 1]));
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
                            float2 pv_d2_1 = fma_f32x2_rn_noftz(make_float2(dyf[0 + 2 * pv_h_1], dyf[0 + 2 * pv_h_1 + 1]), pv_g2_1, make_float2(wr[32], wr[33]));
                            dp[2 * pv_h_1] = pv_d2_1.x;
                            dp[2 * pv_h_1 + 1] = pv_d2_1.y;
                            #pragma unroll
                            for (int kk_3 = 0; kk_3 < 4; kk_3++) {
                                float2 pv_a2_1 = fma_f32x2_rn_noftz(pv_d2_1, make_float2(win[(u_1 % 4 + 1 + kk_3) % 4 * 8 + 2 * pv_h_1], win[(u_1 % 4 + 1 + kk_3) % 4 * 8 + 2 * pv_h_1 + 1]), make_float2(acc[8 * pv_h_1 + kk_3], acc[8 * pv_h_1 + 4 + kk_3]));
                                acc[8 * pv_h_1 + kk_3] = pv_a2_1.x;
                                acc[8 * pv_h_1 + 4 + kk_3] = pv_a2_1.y;
                            }
                            float2 pv_b2_1 = add_f32x2_noftz(make_float2(accb[0 + 2 * pv_h_1], accb[0 + 2 * pv_h_1 + 1]), pv_d2_1);
                            accb[2 * pv_h_1] = pv_b2_1.x;
                            accb[2 * pv_h_1 + 1] = pv_b2_1.y;
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
                    for (int u_2 = 0; u_2 < 4; u_2++) {
                        if (steps_k > u_2) {
                            int tcur = tb + u_2;
                            if (rows_k_1 > u_2) {
                                __int128_t _smem_raw_28;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_28) : "r"(sx0 + u_2 * RS));
                                pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[0];
                                pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[1];
                                pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[2];
                                pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[3];
                                __int128_t _smem_raw_29;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_29) : "r"(sx0 + 16384 + u_2 * RS));
                                pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[0];
                                pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[1];
                                pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[2];
                                pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[3];
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
                                    doc_start = tcur + 4;
                                    doc_end = tcur + 4;
                                }
                                #pragma unroll
                                for (int m_1 = 1; m_1 < 4; m_1++) {
                                    #pragma unroll
                                    for (int j_3 = 0; j_3 < 8; j_3++) {
                                        win[(u_2 % 4 + 4 - m_1) % 4 * 8 + j_3] = 0.0f;
                                    }
                                }
                            }
                            if (tcur < S) {
                                if (tcur < t_end) {
                                    #pragma unroll
                                    for (int pv_h_2 = 0; pv_h_2 < 4; pv_h_2++) {
                                        float2 pv_p2_2 = add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + 3], wr[8 * pv_h_2 + 4 + 3]), make_float2(win[u_2 % 4 * 8 + 2 * pv_h_2], win[u_2 % 4 * 8 + 2 * pv_h_2 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + 2], wr[8 * pv_h_2 + 4 + 2]), make_float2(win[(u_2 % 4 + 3) % 4 * 8 + 2 * pv_h_2], win[(u_2 % 4 + 3) % 4 * 8 + 2 * pv_h_2 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + 1], wr[8 * pv_h_2 + 4 + 1]), make_float2(win[(u_2 % 4 + 2) % 4 * 8 + 2 * pv_h_2], win[(u_2 % 4 + 2) % 4 * 8 + 2 * pv_h_2 + 1]), mul_f32x2_noftz(make_float2(wr[8 * pv_h_2 + 0], wr[8 * pv_h_2 + 4 + 0]), make_float2(win[(u_2 % 4 + 1) % 4 * 8 + 2 * pv_h_2], win[(u_2 % 4 + 1) % 4 * 8 + 2 * pv_h_2 + 1]))))), make_float2(br[0 + 2 * pv_h_2], br[0 + 2 * pv_h_2 + 1]));
                                        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
                                        float2 pv_e2_2;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_2) : "l"(*(const unsigned long long*)&pv_p2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
                                        float _exp2_31 = approx_exp2(pv_e2_2.x);
                                        float _exp2_32 = approx_exp2(pv_e2_2.y);
                                        float2 pv_den2_2 = add_f32x2_noftz(make_float2(_exp2_31, _exp2_32), make_float2(1.0f, 1.0f));
                                        float _rcp_33 = approx_rcp(pv_den2_2.x);
                                        float _rcp_34 = approx_rcp(pv_den2_2.y);
                                        float2 pv_sg2_2 = make_float2(_rcp_33, _rcp_34);
                                        float2 _f32x2_mul_b_35 = fma_f32x2_rn_noftz(pv_p2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_2), make_float2(1.0f, 1.0f));
                                        float2 pv_g2_2;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_2) : "l"(*(const unsigned long long*)&pv_sg2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
                                        float2 pv_d2_2 = fma_f32x2_rn_noftz(make_float2(dyf[0 + 2 * pv_h_2], dyf[0 + 2 * pv_h_2 + 1]), pv_g2_2, make_float2(wr[32], wr[33]));
                                        dp[2 * pv_h_2] = pv_d2_2.x;
                                        dp[2 * pv_h_2 + 1] = pv_d2_2.y;
                                        #pragma unroll
                                        for (int kk_5 = 0; kk_5 < 4; kk_5++) {
                                            float2 pv_a2_2 = fma_f32x2_rn_noftz(pv_d2_2, make_float2(win[(u_2 % 4 + 1 + kk_5) % 4 * 8 + 2 * pv_h_2], win[(u_2 % 4 + 1 + kk_5) % 4 * 8 + 2 * pv_h_2 + 1]), make_float2(acc[8 * pv_h_2 + kk_5], acc[8 * pv_h_2 + 4 + kk_5]));
                                            acc[8 * pv_h_2 + kk_5] = pv_a2_2.x;
                                            acc[8 * pv_h_2 + 4 + kk_5] = pv_a2_2.y;
                                        }
                                        float2 pv_b2_2 = add_f32x2_noftz(make_float2(accb[0 + 2 * pv_h_2], accb[0 + 2 * pv_h_2 + 1]), pv_d2_2);
                                        accb[2 * pv_h_2] = pv_b2_2.x;
                                        accb[2 * pv_h_2 + 1] = pv_b2_2.y;
                                        #pragma unroll
                                        for (int pv_k_2 = 0; pv_k_2 < 3; pv_k_2++) {
                                            float2 pv_q2_2 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_2 + pv_k_2], wr[8 * pv_h_2 + 4 + pv_k_2]), pv_d2_2, make_float2(pend[(u_2 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2], pend[(u_2 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2 + 1]));
                                            pend[(u_2 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2] = pv_q2_2.x;
                                            pend[(u_2 % 4 + 1 + pv_k_2) % 4 * 8 + 2 * pv_h_2 + 1] = pv_q2_2.y;
                                        }
                                        float2 _f32x2_mul_a_36 = make_float2(wr[8 * pv_h_2 + 3], wr[8 * pv_h_2 + 4 + 3]);
                                        float2 pv_r2_2;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&pv_d2_2));
                                        pend[u_2 % 4 * 8 + 2 * pv_h_2] = pv_r2_2.x;
                                        pend[u_2 % 4 * 8 + 2 * pv_h_2 + 1] = pv_r2_2.y;
                                    }
                                } else {
                                    #pragma unroll
                                    for (int pv_h_3 = 0; pv_h_3 < 4; pv_h_3++) {
                                        float2 pv_p2_3 = add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_3 + 3], wr[8 * pv_h_3 + 4 + 3]), make_float2(win[u_2 % 4 * 8 + 2 * pv_h_3], win[u_2 % 4 * 8 + 2 * pv_h_3 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_3 + 2], wr[8 * pv_h_3 + 4 + 2]), make_float2(win[(u_2 % 4 + 3) % 4 * 8 + 2 * pv_h_3], win[(u_2 % 4 + 3) % 4 * 8 + 2 * pv_h_3 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_3 + 1], wr[8 * pv_h_3 + 4 + 1]), make_float2(win[(u_2 % 4 + 2) % 4 * 8 + 2 * pv_h_3], win[(u_2 % 4 + 2) % 4 * 8 + 2 * pv_h_3 + 1]), mul_f32x2_noftz(make_float2(wr[8 * pv_h_3 + 0], wr[8 * pv_h_3 + 4 + 0]), make_float2(win[(u_2 % 4 + 1) % 4 * 8 + 2 * pv_h_3], win[(u_2 % 4 + 1) % 4 * 8 + 2 * pv_h_3 + 1]))))), make_float2(br[0 + 2 * pv_h_3], br[0 + 2 * pv_h_3 + 1]));
                                        float2 _f32x2_mul_b_37 = make_float2(-1.442695f, -1.442695f);
                                        float2 pv_e2_3;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_3) : "l"(*(const unsigned long long*)&pv_p2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
                                        float _exp2_38 = approx_exp2(pv_e2_3.x);
                                        float _exp2_39 = approx_exp2(pv_e2_3.y);
                                        float2 pv_den2_3 = add_f32x2_noftz(make_float2(_exp2_38, _exp2_39), make_float2(1.0f, 1.0f));
                                        float _rcp_40 = approx_rcp(pv_den2_3.x);
                                        float _rcp_41 = approx_rcp(pv_den2_3.y);
                                        float2 pv_sg2_3 = make_float2(_rcp_40, _rcp_41);
                                        float2 _f32x2_mul_b_42 = fma_f32x2_rn_noftz(pv_p2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_3), make_float2(1.0f, 1.0f));
                                        float2 pv_g2_3;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_3) : "l"(*(const unsigned long long*)&pv_sg2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
                                        float2 pv_d2_3 = fma_f32x2_rn_noftz(make_float2(dyf[0 + 2 * pv_h_3], dyf[0 + 2 * pv_h_3 + 1]), pv_g2_3, make_float2(wr[32], wr[33]));
                                        dp[2 * pv_h_3] = pv_d2_3.x;
                                        dp[2 * pv_h_3 + 1] = pv_d2_3.y;
                                        #pragma unroll
                                        for (int pv_k_3 = 0; pv_k_3 < 3; pv_k_3++) {
                                            float2 pv_q2_3 = fma_f32x2_rn_noftz(make_float2(wr[8 * pv_h_3 + pv_k_3], wr[8 * pv_h_3 + 4 + pv_k_3]), pv_d2_3, make_float2(pend[(u_2 % 4 + 1 + pv_k_3) % 4 * 8 + 2 * pv_h_3], pend[(u_2 % 4 + 1 + pv_k_3) % 4 * 8 + 2 * pv_h_3 + 1]));
                                            pend[(u_2 % 4 + 1 + pv_k_3) % 4 * 8 + 2 * pv_h_3] = pv_q2_3.x;
                                            pend[(u_2 % 4 + 1 + pv_k_3) % 4 * 8 + 2 * pv_h_3 + 1] = pv_q2_3.y;
                                        }
                                        float2 _f32x2_mul_a_43 = make_float2(wr[8 * pv_h_3 + 3], wr[8 * pv_h_3 + 4 + 3]);
                                        float2 pv_r2_3;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&pv_d2_3));
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
                int base = tid * 40;
                #pragma unroll
                for (int i_3 = 0; i_3 < 32; i_3++) {
                    float s = red[base + i_3];
                    float y;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(acc[i_3]), "f"(redc[base + i_3]));
                    float t;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t) : "f"(s), "f"(y));
                    float _sub_45;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(t), "f"(s));
                    float _sub_44;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(_sub_45), "f"(y));
                    redc[base + i_3] = _sub_44;
                    red[base + i_3] = t;
                    acc[i_3] = 0.0f;
                }
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    float s_1 = red[base + 32 + j_4];
                    float y_1;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_1) : "f"(accb[j_4]), "f"(redc[base + 32 + j_4]));
                    float t_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_1) : "f"(s_1), "f"(y_1));
                    float _sub_47;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(t_1), "f"(s_1));
                    float _sub_46;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(_sub_47), "f"(y_1));
                    redc[base + 32 + j_4] = _sub_46;
                    red[base + 32 + j_4] = t_1;
                    accb[j_4] = 0.0f;
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
                    int _max_49 = ((my_rows - (k + 2) * 4) > (0) ? (my_rows - (k + 2) * 4) : (0));
                    int _min_48 = ((_max_49) < (4) ? (_max_49) : (4));
                    rows_k_2 = _min_48;
                }
                if (v == 0 && rows_k_2 > 0) {
                    mbarrier_arrive_expect_tx(stage_full_addr + (sub + 4 * (int)((unsigned int)(k + 2) % 2)) * 8, rows_k_2 * (2 * RS));
                    int sx_1 = (int)stage_buf_addr + (sub * (GWn_2 * 2) + slot * GWn_2) * 2048;
                    long long src0_1 = (long long)(t0 + (k + 2) * 4) * C64 + c64;
                    cp_async_bulk_gmem2smem(sx_1, reinterpret_cast<const uint8_t*>(x + src0_1), rows_k_2 * RS, stage_full_addr + (sub + 4 * (int)((unsigned int)(k + 2) % 2)) * 8);
                    cp_async_bulk_gmem2smem(sx_1 + 16384, reinterpret_cast<const uint8_t*>(dy + src0_1), rows_k_2 * RS, stage_full_addr + (sub + 4 * (int)((unsigned int)(k + 2) % 2)) * 8);
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
        int seg_off = (lane_1 - li) * 64;
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
            int _min_51 = ((t00 + L) < (S) ? (t00 + L) : (S));
            int _min_50 = ((_min_51 + 3) < (S + 1) ? (_min_51 + 3) : (S + 1));
            steps0 = _min_50 - t00;
        }
        int n_stages_1 = (steps0 + 4 - 1) / 4;
        int _min_1 = ((n_stages_1) < (2) ? (n_stages_1) : (2));
        #pragma unroll 1
        for (int k0_1 = 0; k0_1 < _min_1; k0_1++) {
            int slot0_1 = k0_1;
            int bslot_2 = warp_1 * 2 + slot0_1;
            int rows_k_3 = 0;
            if (active_1 != 0) {
                int _max_53 = ((my_rows_1 - k0_1 * 4) > (0) ? (my_rows_1 - k0_1 * 4) : (0));
                int _min_52 = ((_max_53) < (4) ? (_max_53) : (4));
                rows_k_3 = _min_52;
            }
            if (li == 0 && rows_k_3 > 0) {
                int cb = ((rstride) < ((int)(C64 - c64_1) * 2) ? (rstride) : ((int)(C64 - c64_1) * 2));
                mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8, rows_k_3 * (2 * cb));
                int sx_2 = (int)stage_buf_addr + bslot_2 * 4096 + seg_off;
                long long src0_2 = (long long)(t0 + k0_1 * 4) * C64 + c64_1;
                if (whole_row != 0) {
                    cp_async_bulk_gmem2smem(sx_2, reinterpret_cast<const uint8_t*>(x + src0_2), rows_k_3 * cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                    cp_async_bulk_gmem2smem(sx_2 + 2048, reinterpret_cast<const uint8_t*>(dy + src0_2), rows_k_3 * cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                } else {
                    #pragma unroll 1
                    for (int i_4 = 0; i_4 < rows_k_3; i_4++) {
                        long long src = src0_2 + (long long)i_4 * C64;
                        cp_async_bulk_gmem2smem(sx_2 + i_4 * rstride, reinterpret_cast<const uint8_t*>(x + src), cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
                        cp_async_bulk_gmem2smem(sx_2 + 2048 + i_4 * rstride, reinterpret_cast<const uint8_t*>(dy + src), cb, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)k0_1 % 2)) * 8);
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
                for (int j_5 = 0; j_5 < 8; j_5++) {
                    {
                        uint2 _vld_54;
                        _vld_54 = *reinterpret_cast<const uint2*>(reinterpret_cast<__nv_bfloat16*>(w) + (c64_1 + (long long)j_5) * 4);
                        uint32_t* _vpairs_54 = reinterpret_cast<uint32_t*>(&_vld_54);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            (&wr_1[j_5 * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) << 16);
                            (&wr_1[j_5 * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int j_6 = 0; j_6 < 8; j_6++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(w) + (c64_1 + (long long)j_6) * 4);
                        wr_1[j_6 * 4 + 0] = _v4.x;
                        wr_1[j_6 * 4 + 1] = _v4.y;
                        wr_1[j_6 * 4 + 2] = _v4.z;
                        wr_1[j_6 * 4 + 3] = _v4.w;
                    }
                }
            }
            if ((flags & 4) != 0) {
                {
                    const uint4* _vptr_56 = reinterpret_cast<const uint4*>(reinterpret_cast<__nv_bfloat16*>(b) + c64_1);
                    uint4 _vld_56[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_56[_blk] = _vptr_56[_blk];
                        uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&br_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
                            (&br_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
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
                        const uint4* _vptr_59 = reinterpret_cast<const uint4*>(x + rowk_1);
                        uint4 _vld_59[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_59[_blk] = _vptr_59[_blk];
                            uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win_1[(kh_1 + 1) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
                                (&win_1[(kh_1 + 1) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
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
                int tb_1 = t0 + k_1 * 4;
                int _max_60 = ((my_rows_1 - k_1 * 4) > (0) ? (my_rows_1 - k_1 * 4) : (0));
                int rows_k_4 = ((_max_60) < (4) ? (_max_60) : (4));
                int _max_61 = ((my_steps_1 - k_1 * 4) > (0) ? (my_steps_1 - k_1 * 4) : (0));
                int steps_k_1 = ((_max_61) < (4) ? (_max_61) : (4));
                int sx0_1 = (int)stage_buf_addr + bslot_3 * 4096 + seg_off + li * 16;
                int interior_1 = 0;
                if (steps_k_1 == 4 && rows_k_4 == 4 && t_end_1 >= tb_1 + 4 && doc_end_1 >= tb_1 + 4) {
                    if (k_1 >= 1) {
                        if (doc_start_1 <= tb_1 - 3) {
                            interior_1 = 2;
                        }
                    }
                }
                if (interior_1 == 2) {
                    #pragma unroll
                    for (int u_3 = 0; u_3 < 4; u_3++) {
                        __int128_t _smem_raw_62;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_62) : "r"(sx0_1 + u_3 * rstride));
                        pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[0];
                        pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[1];
                        pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[2];
                        pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_62)[3];
                        __int128_t _smem_raw_63;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_63) : "r"(sx0_1 + 2048 + u_3 * rstride));
                        pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[0];
                        pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[1];
                        pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[2];
                        pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_63)[3];
                        #pragma unroll
                        for (int kk_6 = 0; kk_6 < 4; kk_6++) {
                            win_1[u_3 % 4 * 8 + 2 * kk_6] = __uint_as_float(pkx_1[kk_6] << 16);
                            win_1[u_3 % 4 * 8 + 2 * kk_6 + 1] = __uint_as_float(pkx_1[kk_6] & 4294901760u);
                            dyf_1[2 * kk_6] = __uint_as_float(pkd_1[kk_6] << 16);
                            dyf_1[2 * kk_6 + 1] = __uint_as_float(pkd_1[kk_6] & 4294901760u);
                        }
                        #pragma unroll
                        for (int pv_h_4 = 0; pv_h_4 < 4; pv_h_4++) {
                            float2 pv_p2_4 = add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_4 + 3], wr_1[8 * pv_h_4 + 4 + 3]), make_float2(win_1[u_3 % 4 * 8 + 2 * pv_h_4], win_1[u_3 % 4 * 8 + 2 * pv_h_4 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_4 + 2], wr_1[8 * pv_h_4 + 4 + 2]), make_float2(win_1[(u_3 % 4 + 3) % 4 * 8 + 2 * pv_h_4], win_1[(u_3 % 4 + 3) % 4 * 8 + 2 * pv_h_4 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_4 + 1], wr_1[8 * pv_h_4 + 4 + 1]), make_float2(win_1[(u_3 % 4 + 2) % 4 * 8 + 2 * pv_h_4], win_1[(u_3 % 4 + 2) % 4 * 8 + 2 * pv_h_4 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * pv_h_4 + 0], wr_1[8 * pv_h_4 + 4 + 0]), make_float2(win_1[(u_3 % 4 + 1) % 4 * 8 + 2 * pv_h_4], win_1[(u_3 % 4 + 1) % 4 * 8 + 2 * pv_h_4 + 1]))))), make_float2(br_1[0 + 2 * pv_h_4], br_1[0 + 2 * pv_h_4 + 1]));
                            float2 _f32x2_mul_b_64 = make_float2(-1.442695f, -1.442695f);
                            float2 pv_e2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_4) : "l"(*(const unsigned long long*)&pv_p2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
                            float _exp2_65 = approx_exp2(pv_e2_4.x);
                            float _exp2_66 = approx_exp2(pv_e2_4.y);
                            float2 pv_den2_4 = add_f32x2_noftz(make_float2(_exp2_65, _exp2_66), make_float2(1.0f, 1.0f));
                            float _rcp_67 = approx_rcp(pv_den2_4.x);
                            float _rcp_68 = approx_rcp(pv_den2_4.y);
                            float2 pv_sg2_4 = make_float2(_rcp_67, _rcp_68);
                            float2 _f32x2_mul_b_69 = fma_f32x2_rn_noftz(pv_p2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_4), make_float2(1.0f, 1.0f));
                            float2 pv_g2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_4) : "l"(*(const unsigned long long*)&pv_sg2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
                            float2 pv_d2_4 = fma_f32x2_rn_noftz(make_float2(dyf_1[0 + 2 * pv_h_4], dyf_1[0 + 2 * pv_h_4 + 1]), pv_g2_4, make_float2(wr_1[32], wr_1[33]));
                            dp_1[2 * pv_h_4] = pv_d2_4.x;
                            dp_1[2 * pv_h_4 + 1] = pv_d2_4.y;
                            #pragma unroll
                            for (int kk_7 = 0; kk_7 < 4; kk_7++) {
                                float2 pv_a2_3 = fma_f32x2_rn_noftz(pv_d2_4, make_float2(win_1[(u_3 % 4 + 1 + kk_7) % 4 * 8 + 2 * pv_h_4], win_1[(u_3 % 4 + 1 + kk_7) % 4 * 8 + 2 * pv_h_4 + 1]), make_float2(acc[8 * pv_h_4 + kk_7], acc[8 * pv_h_4 + 4 + kk_7]));
                                acc[8 * pv_h_4 + kk_7] = pv_a2_3.x;
                                acc[8 * pv_h_4 + 4 + kk_7] = pv_a2_3.y;
                            }
                            float2 pv_b2_3 = add_f32x2_noftz(make_float2(accb[0 + 2 * pv_h_4], accb[0 + 2 * pv_h_4 + 1]), pv_d2_4);
                            accb[2 * pv_h_4] = pv_b2_3.x;
                            accb[2 * pv_h_4 + 1] = pv_b2_3.y;
                            #pragma unroll
                            for (int pv_k_4 = 0; pv_k_4 < 3; pv_k_4++) {
                                float2 pv_q2_4 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_4 + pv_k_4], wr_1[8 * pv_h_4 + 4 + pv_k_4]), pv_d2_4, make_float2(pend_1[(u_3 % 4 + 1 + pv_k_4) % 4 * 8 + 2 * pv_h_4], pend_1[(u_3 % 4 + 1 + pv_k_4) % 4 * 8 + 2 * pv_h_4 + 1]));
                                pend_1[(u_3 % 4 + 1 + pv_k_4) % 4 * 8 + 2 * pv_h_4] = pv_q2_4.x;
                                pend_1[(u_3 % 4 + 1 + pv_k_4) % 4 * 8 + 2 * pv_h_4 + 1] = pv_q2_4.y;
                            }
                            float2 _f32x2_mul_a_70 = make_float2(wr_1[8 * pv_h_4 + 3], wr_1[8 * pv_h_4 + 4 + 3]);
                            float2 pv_r2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&pv_d2_4));
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
                    for (int u_4 = 0; u_4 < 4; u_4++) {
                        __int128_t _smem_raw_71;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_71) : "r"(sx0_1 + u_4 * rstride));
                        pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_71)[0];
                        pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_71)[1];
                        pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_71)[2];
                        pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_71)[3];
                        __int128_t _smem_raw_72;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_72) : "r"(sx0_1 + 2048 + u_4 * rstride));
                        pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[0];
                        pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[1];
                        pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[2];
                        pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_72)[3];
                        #pragma unroll
                        for (int kk_8 = 0; kk_8 < 4; kk_8++) {
                            win_1[u_4 % 4 * 8 + 2 * kk_8] = __uint_as_float(pkx_1[kk_8] << 16);
                            win_1[u_4 % 4 * 8 + 2 * kk_8 + 1] = __uint_as_float(pkx_1[kk_8] & 4294901760u);
                            dyf_1[2 * kk_8] = __uint_as_float(pkd_1[kk_8] << 16);
                            dyf_1[2 * kk_8 + 1] = __uint_as_float(pkd_1[kk_8] & 4294901760u);
                        }
                        #pragma unroll
                        for (int pv_h_5 = 0; pv_h_5 < 4; pv_h_5++) {
                            float2 pv_p2_5 = add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_5 + 3], wr_1[8 * pv_h_5 + 4 + 3]), make_float2(win_1[u_4 % 4 * 8 + 2 * pv_h_5], win_1[u_4 % 4 * 8 + 2 * pv_h_5 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_5 + 2], wr_1[8 * pv_h_5 + 4 + 2]), make_float2(win_1[(u_4 % 4 + 3) % 4 * 8 + 2 * pv_h_5], win_1[(u_4 % 4 + 3) % 4 * 8 + 2 * pv_h_5 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_5 + 1], wr_1[8 * pv_h_5 + 4 + 1]), make_float2(win_1[(u_4 % 4 + 2) % 4 * 8 + 2 * pv_h_5], win_1[(u_4 % 4 + 2) % 4 * 8 + 2 * pv_h_5 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * pv_h_5 + 0], wr_1[8 * pv_h_5 + 4 + 0]), make_float2(win_1[(u_4 % 4 + 1) % 4 * 8 + 2 * pv_h_5], win_1[(u_4 % 4 + 1) % 4 * 8 + 2 * pv_h_5 + 1]))))), make_float2(br_1[0 + 2 * pv_h_5], br_1[0 + 2 * pv_h_5 + 1]));
                            float2 _f32x2_mul_b_73 = make_float2(-1.442695f, -1.442695f);
                            float2 pv_e2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_5) : "l"(*(const unsigned long long*)&pv_p2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
                            float _exp2_74 = approx_exp2(pv_e2_5.x);
                            float _exp2_75 = approx_exp2(pv_e2_5.y);
                            float2 pv_den2_5 = add_f32x2_noftz(make_float2(_exp2_74, _exp2_75), make_float2(1.0f, 1.0f));
                            float _rcp_76 = approx_rcp(pv_den2_5.x);
                            float _rcp_77 = approx_rcp(pv_den2_5.y);
                            float2 pv_sg2_5 = make_float2(_rcp_76, _rcp_77);
                            float2 _f32x2_mul_b_78 = fma_f32x2_rn_noftz(pv_p2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_5), make_float2(1.0f, 1.0f));
                            float2 pv_g2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_5) : "l"(*(const unsigned long long*)&pv_sg2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
                            float2 pv_d2_5 = fma_f32x2_rn_noftz(make_float2(dyf_1[0 + 2 * pv_h_5], dyf_1[0 + 2 * pv_h_5 + 1]), pv_g2_5, make_float2(wr_1[32], wr_1[33]));
                            dp_1[2 * pv_h_5] = pv_d2_5.x;
                            dp_1[2 * pv_h_5 + 1] = pv_d2_5.y;
                            #pragma unroll
                            for (int kk_9 = 0; kk_9 < 4; kk_9++) {
                                float2 pv_a2_4 = fma_f32x2_rn_noftz(pv_d2_5, make_float2(win_1[(u_4 % 4 + 1 + kk_9) % 4 * 8 + 2 * pv_h_5], win_1[(u_4 % 4 + 1 + kk_9) % 4 * 8 + 2 * pv_h_5 + 1]), make_float2(acc[8 * pv_h_5 + kk_9], acc[8 * pv_h_5 + 4 + kk_9]));
                                acc[8 * pv_h_5 + kk_9] = pv_a2_4.x;
                                acc[8 * pv_h_5 + 4 + kk_9] = pv_a2_4.y;
                            }
                            float2 pv_b2_4 = add_f32x2_noftz(make_float2(accb[0 + 2 * pv_h_5], accb[0 + 2 * pv_h_5 + 1]), pv_d2_5);
                            accb[2 * pv_h_5] = pv_b2_4.x;
                            accb[2 * pv_h_5 + 1] = pv_b2_4.y;
                            #pragma unroll
                            for (int pv_k_5 = 0; pv_k_5 < 3; pv_k_5++) {
                                float2 pv_q2_5 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_5 + pv_k_5], wr_1[8 * pv_h_5 + 4 + pv_k_5]), pv_d2_5, make_float2(pend_1[(u_4 % 4 + 1 + pv_k_5) % 4 * 8 + 2 * pv_h_5], pend_1[(u_4 % 4 + 1 + pv_k_5) % 4 * 8 + 2 * pv_h_5 + 1]));
                                pend_1[(u_4 % 4 + 1 + pv_k_5) % 4 * 8 + 2 * pv_h_5] = pv_q2_5.x;
                                pend_1[(u_4 % 4 + 1 + pv_k_5) % 4 * 8 + 2 * pv_h_5 + 1] = pv_q2_5.y;
                            }
                            float2 _f32x2_mul_a_79 = make_float2(wr_1[8 * pv_h_5 + 3], wr_1[8 * pv_h_5 + 4 + 3]);
                            float2 pv_r2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&pv_d2_5));
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
                    for (int u_5 = 0; u_5 < 4; u_5++) {
                        if (steps_k_1 > u_5) {
                            int tcur_1 = tb_1 + u_5;
                            if (rows_k_4 > u_5) {
                                __int128_t _smem_raw_80;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_80) : "r"(sx0_1 + u_5 * rstride));
                                pkx_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_80)[0];
                                pkx_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_80)[1];
                                pkx_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_80)[2];
                                pkx_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_80)[3];
                                __int128_t _smem_raw_81;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_81) : "r"(sx0_1 + 2048 + u_5 * rstride));
                                pkd_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_81)[0];
                                pkd_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_81)[1];
                                pkd_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_81)[2];
                                pkd_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_81)[3];
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
                                    doc_start_1 = tcur_1 + 4;
                                    doc_end_1 = tcur_1 + 4;
                                }
                                #pragma unroll
                                for (int m_3 = 1; m_3 < 4; m_3++) {
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 8; j_7++) {
                                        win_1[(u_5 % 4 + 4 - m_3) % 4 * 8 + j_7] = 0.0f;
                                    }
                                }
                            }
                            if (tcur_1 < S) {
                                if (tcur_1 < t_end_1) {
                                    #pragma unroll
                                    for (int pv_h_6 = 0; pv_h_6 < 4; pv_h_6++) {
                                        float2 pv_p2_6 = add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_6 + 3], wr_1[8 * pv_h_6 + 4 + 3]), make_float2(win_1[u_5 % 4 * 8 + 2 * pv_h_6], win_1[u_5 % 4 * 8 + 2 * pv_h_6 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_6 + 2], wr_1[8 * pv_h_6 + 4 + 2]), make_float2(win_1[(u_5 % 4 + 3) % 4 * 8 + 2 * pv_h_6], win_1[(u_5 % 4 + 3) % 4 * 8 + 2 * pv_h_6 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_6 + 1], wr_1[8 * pv_h_6 + 4 + 1]), make_float2(win_1[(u_5 % 4 + 2) % 4 * 8 + 2 * pv_h_6], win_1[(u_5 % 4 + 2) % 4 * 8 + 2 * pv_h_6 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * pv_h_6 + 0], wr_1[8 * pv_h_6 + 4 + 0]), make_float2(win_1[(u_5 % 4 + 1) % 4 * 8 + 2 * pv_h_6], win_1[(u_5 % 4 + 1) % 4 * 8 + 2 * pv_h_6 + 1]))))), make_float2(br_1[0 + 2 * pv_h_6], br_1[0 + 2 * pv_h_6 + 1]));
                                        float2 _f32x2_mul_b_82 = make_float2(-1.442695f, -1.442695f);
                                        float2 pv_e2_6;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_6) : "l"(*(const unsigned long long*)&pv_p2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
                                        float _exp2_83 = approx_exp2(pv_e2_6.x);
                                        float _exp2_84 = approx_exp2(pv_e2_6.y);
                                        float2 pv_den2_6 = add_f32x2_noftz(make_float2(_exp2_83, _exp2_84), make_float2(1.0f, 1.0f));
                                        float _rcp_85 = approx_rcp(pv_den2_6.x);
                                        float _rcp_86 = approx_rcp(pv_den2_6.y);
                                        float2 pv_sg2_6 = make_float2(_rcp_85, _rcp_86);
                                        float2 _f32x2_mul_b_87 = fma_f32x2_rn_noftz(pv_p2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_6), make_float2(1.0f, 1.0f));
                                        float2 pv_g2_6;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_6) : "l"(*(const unsigned long long*)&pv_sg2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
                                        float2 pv_d2_6 = fma_f32x2_rn_noftz(make_float2(dyf_1[0 + 2 * pv_h_6], dyf_1[0 + 2 * pv_h_6 + 1]), pv_g2_6, make_float2(wr_1[32], wr_1[33]));
                                        dp_1[2 * pv_h_6] = pv_d2_6.x;
                                        dp_1[2 * pv_h_6 + 1] = pv_d2_6.y;
                                        #pragma unroll
                                        for (int kk_11 = 0; kk_11 < 4; kk_11++) {
                                            float2 pv_a2_5 = fma_f32x2_rn_noftz(pv_d2_6, make_float2(win_1[(u_5 % 4 + 1 + kk_11) % 4 * 8 + 2 * pv_h_6], win_1[(u_5 % 4 + 1 + kk_11) % 4 * 8 + 2 * pv_h_6 + 1]), make_float2(acc[8 * pv_h_6 + kk_11], acc[8 * pv_h_6 + 4 + kk_11]));
                                            acc[8 * pv_h_6 + kk_11] = pv_a2_5.x;
                                            acc[8 * pv_h_6 + 4 + kk_11] = pv_a2_5.y;
                                        }
                                        float2 pv_b2_5 = add_f32x2_noftz(make_float2(accb[0 + 2 * pv_h_6], accb[0 + 2 * pv_h_6 + 1]), pv_d2_6);
                                        accb[2 * pv_h_6] = pv_b2_5.x;
                                        accb[2 * pv_h_6 + 1] = pv_b2_5.y;
                                        #pragma unroll
                                        for (int pv_k_6 = 0; pv_k_6 < 3; pv_k_6++) {
                                            float2 pv_q2_6 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_6 + pv_k_6], wr_1[8 * pv_h_6 + 4 + pv_k_6]), pv_d2_6, make_float2(pend_1[(u_5 % 4 + 1 + pv_k_6) % 4 * 8 + 2 * pv_h_6], pend_1[(u_5 % 4 + 1 + pv_k_6) % 4 * 8 + 2 * pv_h_6 + 1]));
                                            pend_1[(u_5 % 4 + 1 + pv_k_6) % 4 * 8 + 2 * pv_h_6] = pv_q2_6.x;
                                            pend_1[(u_5 % 4 + 1 + pv_k_6) % 4 * 8 + 2 * pv_h_6 + 1] = pv_q2_6.y;
                                        }
                                        float2 _f32x2_mul_a_88 = make_float2(wr_1[8 * pv_h_6 + 3], wr_1[8 * pv_h_6 + 4 + 3]);
                                        float2 pv_r2_6;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&pv_d2_6));
                                        pend_1[u_5 % 4 * 8 + 2 * pv_h_6] = pv_r2_6.x;
                                        pend_1[u_5 % 4 * 8 + 2 * pv_h_6 + 1] = pv_r2_6.y;
                                    }
                                } else {
                                    #pragma unroll
                                    for (int pv_h_7 = 0; pv_h_7 < 4; pv_h_7++) {
                                        float2 pv_p2_7 = add_f32x2_noftz(fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_7 + 3], wr_1[8 * pv_h_7 + 4 + 3]), make_float2(win_1[u_5 % 4 * 8 + 2 * pv_h_7], win_1[u_5 % 4 * 8 + 2 * pv_h_7 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_7 + 2], wr_1[8 * pv_h_7 + 4 + 2]), make_float2(win_1[(u_5 % 4 + 3) % 4 * 8 + 2 * pv_h_7], win_1[(u_5 % 4 + 3) % 4 * 8 + 2 * pv_h_7 + 1]), fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_7 + 1], wr_1[8 * pv_h_7 + 4 + 1]), make_float2(win_1[(u_5 % 4 + 2) % 4 * 8 + 2 * pv_h_7], win_1[(u_5 % 4 + 2) % 4 * 8 + 2 * pv_h_7 + 1]), mul_f32x2_noftz(make_float2(wr_1[8 * pv_h_7 + 0], wr_1[8 * pv_h_7 + 4 + 0]), make_float2(win_1[(u_5 % 4 + 1) % 4 * 8 + 2 * pv_h_7], win_1[(u_5 % 4 + 1) % 4 * 8 + 2 * pv_h_7 + 1]))))), make_float2(br_1[0 + 2 * pv_h_7], br_1[0 + 2 * pv_h_7 + 1]));
                                        float2 _f32x2_mul_b_89 = make_float2(-1.442695f, -1.442695f);
                                        float2 pv_e2_7;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_e2_7) : "l"(*(const unsigned long long*)&pv_p2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_89));
                                        float _exp2_90 = approx_exp2(pv_e2_7.x);
                                        float _exp2_91 = approx_exp2(pv_e2_7.y);
                                        float2 pv_den2_7 = add_f32x2_noftz(make_float2(_exp2_90, _exp2_91), make_float2(1.0f, 1.0f));
                                        float _rcp_92 = approx_rcp(pv_den2_7.x);
                                        float _rcp_93 = approx_rcp(pv_den2_7.y);
                                        float2 pv_sg2_7 = make_float2(_rcp_92, _rcp_93);
                                        float2 _f32x2_mul_b_94 = fma_f32x2_rn_noftz(pv_p2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), pv_sg2_7), make_float2(1.0f, 1.0f));
                                        float2 pv_g2_7;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_g2_7) : "l"(*(const unsigned long long*)&pv_sg2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_94));
                                        float2 pv_d2_7 = fma_f32x2_rn_noftz(make_float2(dyf_1[0 + 2 * pv_h_7], dyf_1[0 + 2 * pv_h_7 + 1]), pv_g2_7, make_float2(wr_1[32], wr_1[33]));
                                        dp_1[2 * pv_h_7] = pv_d2_7.x;
                                        dp_1[2 * pv_h_7 + 1] = pv_d2_7.y;
                                        #pragma unroll
                                        for (int pv_k_7 = 0; pv_k_7 < 3; pv_k_7++) {
                                            float2 pv_q2_7 = fma_f32x2_rn_noftz(make_float2(wr_1[8 * pv_h_7 + pv_k_7], wr_1[8 * pv_h_7 + 4 + pv_k_7]), pv_d2_7, make_float2(pend_1[(u_5 % 4 + 1 + pv_k_7) % 4 * 8 + 2 * pv_h_7], pend_1[(u_5 % 4 + 1 + pv_k_7) % 4 * 8 + 2 * pv_h_7 + 1]));
                                            pend_1[(u_5 % 4 + 1 + pv_k_7) % 4 * 8 + 2 * pv_h_7] = pv_q2_7.x;
                                            pend_1[(u_5 % 4 + 1 + pv_k_7) % 4 * 8 + 2 * pv_h_7 + 1] = pv_q2_7.y;
                                        }
                                        float2 _f32x2_mul_a_95 = make_float2(wr_1[8 * pv_h_7 + 3], wr_1[8 * pv_h_7 + 4 + 3]);
                                        float2 pv_r2_7;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pv_r2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&pv_d2_7));
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
                int base_1 = tid * 40;
                #pragma unroll
                for (int i_6 = 0; i_6 < 32; i_6++) {
                    float s_3 = red[base_1 + i_6];
                    float y_2;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_2) : "f"(acc[i_6]), "f"(redc[base_1 + i_6]));
                    float t_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_2) : "f"(s_3), "f"(y_2));
                    float _sub_97;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(t_2), "f"(s_3));
                    float _sub_96;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(_sub_97), "f"(y_2));
                    redc[base_1 + i_6] = _sub_96;
                    red[base_1 + i_6] = t_2;
                    acc[i_6] = 0.0f;
                }
                #pragma unroll
                for (int j_8 = 0; j_8 < 8; j_8++) {
                    float s_4 = red[base_1 + 32 + j_8];
                    float y_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_3) : "f"(accb[j_8]), "f"(redc[base_1 + 32 + j_8]));
                    float t_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(t_3) : "f"(s_4), "f"(y_3));
                    float _sub_99;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_99) : "f"(t_3), "f"(s_4));
                    float _sub_98;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_98) : "f"(_sub_99), "f"(y_3));
                    redc[base_1 + 32 + j_8] = _sub_98;
                    red[base_1 + 32 + j_8] = t_3;
                    accb[j_8] = 0.0f;
                }
            }
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int kn_1 = k_1 + 2;
            if (kn_1 < n_stages_1) {
                int bslot_1_1 = warp_1 * 2 + slot_1;
                int rows_k_5 = 0;
                if (active_1 != 0) {
                    int _max_101 = ((my_rows_1 - (k_1 + 2) * 4) > (0) ? (my_rows_1 - (k_1 + 2) * 4) : (0));
                    int _min_100 = ((_max_101) < (4) ? (_max_101) : (4));
                    rows_k_5 = _min_100;
                }
                if (li == 0 && rows_k_5 > 0) {
                    int cb_1 = ((rstride) < ((int)(C64 - c64_1) * 2) ? (rstride) : ((int)(C64 - c64_1) * 2));
                    mbarrier_arrive_expect_tx(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8, rows_k_5 * (2 * cb_1));
                    int sx_3 = (int)stage_buf_addr + bslot_1_1 * 4096 + seg_off;
                    long long src0_3 = (long long)(t0 + (k_1 + 2) * 4) * C64 + c64_1;
                    if (whole_row != 0) {
                        cp_async_bulk_gmem2smem(sx_3, reinterpret_cast<const uint8_t*>(x + src0_3), rows_k_5 * cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                        cp_async_bulk_gmem2smem(sx_3 + 2048, reinterpret_cast<const uint8_t*>(dy + src0_3), rows_k_5 * cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                    } else {
                        #pragma unroll 1
                        for (int i_7 = 0; i_7 < rows_k_5; i_7++) {
                            long long src_1 = src0_3 + (long long)i_7 * C64;
                            cp_async_bulk_gmem2smem(sx_3 + i_7 * rstride, reinterpret_cast<const uint8_t*>(x + src_1), cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                            cp_async_bulk_gmem2smem(sx_3 + 2048 + i_7 * rstride, reinterpret_cast<const uint8_t*>(dy + src_1), cb_1, stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
                        }
                    }
                } else {
                    mbarrier_arrive(stage_full_addr + (warp_1 + 4 * (int)((unsigned int)(k_1 + 2) % 2)) * 8);
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
        float _sub_103;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_103) : "f"(t_4), "f"(s_5));
        float _sub_102;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(_sub_103), "f"(y_4));
        redc[base_2 + i_8] = _sub_102;
        red[base_2 + i_8] = t_4;
        acc[i_8] = 0.0f;
    }
    #pragma unroll
    for (int j_9 = 0; j_9 < 8; j_9++) {
        float s_6 = red[base_2 + 32 + j_9];
        float y_5;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(y_5) : "f"(accb[j_9]), "f"(redc[base_2 + 32 + j_9]));
        float t_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(t_5) : "f"(s_6), "f"(y_5));
        float _sub_105;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(t_5), "f"(s_6));
        float _sub_104;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(_sub_105), "f"(y_5));
        redc[base_2 + 32 + j_9] = _sub_104;
        red[base_2 + 32 + j_9] = t_5;
        accb[j_9] = 0.0f;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 40; i_9++) {
        float _sub_106;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_106) : "f"(red[tid * 40 + i_9]), "f"(redc[tid * 40 + i_9]));
        red[tid * 40 + i_9] = _sub_106;
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
        for (int j_10 = 0; j_10 < 8; j_10++) {
            long long pidx = ((long long)rank * C64 + (long long)c0_0 + (long long)j_10) * 4;
            {
                float4 _v4 = make_float4(tot[j_10 * 4 + 0], tot[j_10 * 4 + 1], tot[j_10 * 4 + 2], tot[j_10 * 4 + 3]);
                *reinterpret_cast<float4*>(partial + pidx) = _v4;
            }
        }
        float totb[8];
        #pragma unroll
        for (int j_11 = 0; j_11 < 8; j_11++) {
            totb[j_11] = red[v * 40 + (32 + j_11)];
        }
        #pragma unroll 1
        for (int s_8 = 1; s_8 < n_sub; s_8++) {
            #pragma unroll
            for (int j_12 = 0; j_12 < 8; j_12++) {
                totb[j_12] = totb[j_12] + red[(s_8 * ct_vec + v) * 40 + (32 + j_12)];
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
