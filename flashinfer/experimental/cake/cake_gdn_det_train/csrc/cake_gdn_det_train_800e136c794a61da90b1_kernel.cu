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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_det_train_800e136c794a61da90b1(const __grid_constant__ CUtensorMap k_tma64, const __grid_constant__ CUtensorMap k_tma32, const __grid_constant__ CUtensorMap q_tma, float* __restrict__ glog, float* __restrict__ beta, unsigned int* __restrict__ a_out, float* __restrict__ rk_out, float* __restrict__ rq_out, float* __restrict__ gcs_out, unsigned int* __restrict__ qn_out, unsigned int* __restrict__ kn_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_chunks, int num_heads, int num_v_heads)
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

    const int mbar_base = smem + 225408;
    #define st_full_addr (mbar_base + 0)
    #define st_free_m_addr (mbar_base + 16)
    #define st_free_r_addr (mbar_base + 32)
    #define g_full_addr (mbar_base + 48)
    #define g_free_addr (mbar_base + 64)
    #define gates_ready_addr (mbar_base + 80)
    #define gates_free_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* s_ga0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_ga0_addr = smem + 0;
    __nv_bfloat16* s_ga1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_ga1_addr = smem + 16384;
    __nv_bfloat16* s_gb0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_gb0_addr = smem + 0;
    __nv_bfloat16* s_gb1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_gb1_addr = smem + 16384;
    __nv_bfloat16* s_q0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_q0_addr = smem + 32768;
    __nv_bfloat16* s_q1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 40960);
    const int s_q1_addr = smem + 40960;
    unsigned int* s_st = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int s_st_addr = smem + 0;
    float* s_gt = reinterpret_cast<float*>(smem_raw + 98304);
    const int s_gt_addr = smem + 98304;
    float* s_g = reinterpret_cast<float*>(smem_raw + 120960);
    const int s_g_addr = smem + 120960;

    // Mbarrier init (18 barriers at smem_raw[225408..225552))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // st_full: 2 barriers, init_count=1
            mbarrier_init(smem + 225408, 1);
            mbarrier_init(smem + 225416, 1);
            // st_free_m: 2 barriers, init_count=1
            mbarrier_init(smem + 225424, 1);
            mbarrier_init(smem + 225432, 1);
            // st_free_r: 2 barriers, init_count=448
            mbarrier_init(smem + 225440, 448);
            mbarrier_init(smem + 225448, 448);
            // g_full: 2 barriers, init_count=1
            mbarrier_init(smem + 225456, 1);
            mbarrier_init(smem + 225464, 1);
            // g_free: 2 barriers, init_count=384
            mbarrier_init(smem + 225472, 384);
            mbarrier_init(smem + 225480, 384);
            // --- pipeline 'gring' ---
            // gates_ready: 4 barriers, init_count=96
            mbarrier_init(smem + 225488, 96);
            mbarrier_init(smem + 225496, 96);
            mbarrier_init(smem + 225504, 96);
            mbarrier_init(smem + 225512, 96);
            // gates_free: 4 barriers, init_count=384
            mbarrier_init(smem + 225520, 384);
            mbarrier_init(smem + 225528, 384);
            mbarrier_init(smem + 225536, 384);
            mbarrier_init(smem + 225544, 384);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 225552);
    if (warp == 0) {
        int _tmem_hold = smem + 225552;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_g0 = taddr;
    const unsigned int tmem_g1 = taddr + 64;

    // ---- Role: cw ----
    if (warp <= 11) {
        { // cw_main
            int bid_1 = blockIdx.x;
            int grid = gridDim.x;
            int n_items = num_chunks * num_heads;
            int cnt = 0;
            if (bid_1 < n_items) {
                cnt = (n_items - bid_1 + grid - 1) / grid;
            }
            int lane_1 = lane;
            int wid = warp;
            int grp = wid / 2;
            int pw = wid % 2;
            int par = grp / 3;
            int hgrp = grp % 3;
            int gq = lane_1 / 4;
            int tq = lane_1 % 4;
            int gpk = num_v_heads / num_heads;
            long long ostride = (long long)(num_v_heads * 32);
            int gw = grp * 4352;
            unsigned int lb = s_g_addr + (unsigned int)(grp * 17408);
            int tl_base = 32 * (wid % 4) << 16;
            unsigned int gfl_c = 0;
            unsigned int gs_c = 0;
            unsigned int _phase_g_full = 0;
            unsigned int _phase_gates_ready = 0;
            #pragma unroll 1
            for (int it = 0; it < cnt; it++) {
                int item = bid_1 + it * grid;
                int ch = item / num_heads;
                int hq = item % num_heads;
                int tok0 = chunk_start[ch];
                int n = chunk_len[ch];
                int p_off = it % 4 * 1416;
                int hb = p_off + 72 + 448 * hgrp;
                if (it % 2 == par) {
                }
                mbarrier_wait(g_full_addr + (gfl_c) * 8, _phase_g_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (it % 2 == par) {
                }
                mbarrier_wait(gates_ready_addr + (gs_c) * 8, _phase_gates_ready);
                mbarrier_wait(st_full_addr + (gfl_c) * 8, _phase_g_full);
                if (it % 2 == par) {
                    unsigned int sb = s_st_addr + gfl_c * 49152;
                    if (pw == 0) {
                        int nc = lane_1 % 16;
                        int nkp = nc / 8;
                        int nch = nc % 8;
                        int nhl = lane_1 / 16;
                        unsigned int nsb = sb + 32768 + (unsigned int)(nkp * 8192 + nhl * 128);
                        long long nrow_stride = (long long)num_heads * 64ll;
                        long long ndst = ((long long)(tok0 + nhl) * (long long)num_heads + (long long)hq) * 64ll + (long long)(32 * nkp + 4 * nch);
                        #pragma unroll 1
                        for (int ni = hgrp; ni < 32; ni += 3) {
                            int nrow = 2 * ni + nhl;
                            float nrs = s_gt[p_off + 72 + 384 + nrow];
                            float2 nrs2 = make_float2(nrs, nrs);
                            unsigned int nw[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&nw[0])), "=r"(*reinterpret_cast<uint32_t*>(&nw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&nw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&nw[(0) + 3]))
                                : "r"(nsb + (unsigned int)(ni * 256 + ((nch ^ nrow % 8) << 4))));
                            float nf[8];
                            float2 _f32x2_mul_a_0 = make_float2(__uint_as_float(nw[0] << 16), __uint_as_float(nw[0] & 4294901760u));
                            float2 np2_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&np2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&nrs2));
                            nf[0] = np2_0.x;
                            nf[1] = np2_0.y;
                            float2 _f32x2_mul_a_1 = make_float2(__uint_as_float(nw[1] << 16), __uint_as_float(nw[1] & 4294901760u));
                            float2 np2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&np2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&nrs2));
                            nf[2] = np2_1.x;
                            nf[3] = np2_1.y;
                            float2 _f32x2_mul_a_2 = make_float2(__uint_as_float(nw[2] << 16), __uint_as_float(nw[2] & 4294901760u));
                            float2 np2_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&np2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&nrs2));
                            nf[4] = np2_2.x;
                            nf[5] = np2_2.y;
                            float2 _f32x2_mul_a_3 = make_float2(__uint_as_float(nw[3] << 16), __uint_as_float(nw[3] & 4294901760u));
                            float2 np2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&np2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_3), "l"(*(const unsigned long long*)&nrs2));
                            nf[6] = np2_3.x;
                            nf[7] = np2_3.y;
                            unsigned int npk[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nf[_lp*2 + 0], nf[_lp*2+1 + 0]));
                                npk[_lp] = *(uint32_t*)&_bf2;
                            }
                            if (nrow < n) {
                                reinterpret_cast<int4*>(qn_out + (ndst + (long long)(2 * ni) * nrow_stride))[0] = reinterpret_cast<int4*>(npk)[0];
                            }
                        }
                    } else {
                        int nc_1 = lane_1 % 16;
                        int nkp_1 = nc_1 / 8;
                        int nch_1 = nc_1 % 8;
                        int nhl_1 = lane_1 / 16;
                        unsigned int nsb_1 = sb + (unsigned int)(nkp_1 * 16384 + nhl_1 * 128);
                        long long nrow_stride_1 = (long long)num_heads * 64ll;
                        long long ndst_1 = ((long long)(tok0 + nhl_1) * (long long)num_heads + (long long)hq) * 64ll + (long long)(32 * nkp_1 + 4 * nch_1);
                        #pragma unroll 1
                        for (int ni_1 = hgrp; ni_1 < 32; ni_1 += 3) {
                            int nrow_1 = 2 * ni_1 + nhl_1;
                            float nrs_1 = s_gt[p_off + nrow_1];
                            float2 nrs2_1 = make_float2(nrs_1, nrs_1);
                            unsigned int nw_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&nw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&nw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&nw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&nw_1[(0) + 3]))
                                : "r"(nsb_1 + (unsigned int)(ni_1 * 256 + ((nch_1 ^ nrow_1 % 8) << 4))));
                            float nf_1[8];
                            float2 _f32x2_mul_a_4 = make_float2(__uint_as_float(nw_1[0] << 16), __uint_as_float(nw_1[0] & 4294901760u));
                            float2 np2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&np2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&nrs2_1));
                            nf_1[0] = np2_4.x;
                            nf_1[1] = np2_4.y;
                            float2 _f32x2_mul_a_5 = make_float2(__uint_as_float(nw_1[1] << 16), __uint_as_float(nw_1[1] & 4294901760u));
                            float2 np2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&np2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&nrs2_1));
                            nf_1[2] = np2_5.x;
                            nf_1[3] = np2_5.y;
                            float2 _f32x2_mul_a_6 = make_float2(__uint_as_float(nw_1[2] << 16), __uint_as_float(nw_1[2] & 4294901760u));
                            float2 np2_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&np2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&nrs2_1));
                            nf_1[4] = np2_6.x;
                            nf_1[5] = np2_6.y;
                            float2 _f32x2_mul_a_7 = make_float2(__uint_as_float(nw_1[3] << 16), __uint_as_float(nw_1[3] & 4294901760u));
                            float2 np2_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&np2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&nrs2_1));
                            nf_1[6] = np2_7.x;
                            nf_1[7] = np2_7.y;
                            unsigned int npk_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nf_1[_lp*2 + 0], nf_1[_lp*2+1 + 0]));
                                npk_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            if (nrow_1 < n) {
                                reinterpret_cast<int4*>(kn_out + (ndst_1 + (long long)(2 * ni_1) * nrow_stride_1))[0] = reinterpret_cast<int4*>(npk_1)[0];
                            }
                        }
                    }
                }
                mbarrier_arrive(st_free_r_addr + (gfl_c) * 8);
                if (it % 2 == par) {
                    int b_row = 32 * pw + lane_1;
                    float bd_gi = s_gt[hb + b_row];
                    float bd_br = s_gt[hb + 320 + b_row];
                    float bd_fl = s_gt[p_off + 64 + hgrp];
                    if (0 <= pw) {
                        float gv_0[32];
                        if (it % 2 == 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(gv_0[0]), "=f"(gv_0[1]), "=f"(gv_0[2]), "=f"(gv_0[3]), "=f"(gv_0[4]), "=f"(gv_0[5]), "=f"(gv_0[6]), "=f"(gv_0[7]), "=f"(gv_0[8]), "=f"(gv_0[9]), "=f"(gv_0[10]), "=f"(gv_0[11]), "=f"(gv_0[12]), "=f"(gv_0[13]), "=f"(gv_0[14]), "=f"(gv_0[15]), "=f"(gv_0[16]), "=f"(gv_0[17]), "=f"(gv_0[18]), "=f"(gv_0[19]), "=f"(gv_0[20]), "=f"(gv_0[21]), "=f"(gv_0[22]), "=f"(gv_0[23]), "=f"(gv_0[24]), "=f"(gv_0[25]), "=f"(gv_0[26]), "=f"(gv_0[27]), "=f"(gv_0[28]), "=f"(gv_0[29]), "=f"(gv_0[30]), "=f"(gv_0[31])
                                : "r"(tmem_g0 + tl_base + 0));
                        } else {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(gv_0[0]), "=f"(gv_0[1]), "=f"(gv_0[2]), "=f"(gv_0[3]), "=f"(gv_0[4]), "=f"(gv_0[5]), "=f"(gv_0[6]), "=f"(gv_0[7]), "=f"(gv_0[8]), "=f"(gv_0[9]), "=f"(gv_0[10]), "=f"(gv_0[11]), "=f"(gv_0[12]), "=f"(gv_0[13]), "=f"(gv_0[14]), "=f"(gv_0[15]), "=f"(gv_0[16]), "=f"(gv_0[17]), "=f"(gv_0[18]), "=f"(gv_0[19]), "=f"(gv_0[20]), "=f"(gv_0[21]), "=f"(gv_0[22]), "=f"(gv_0[23]), "=f"(gv_0[24]), "=f"(gv_0[25]), "=f"(gv_0[26]), "=f"(gv_0[27]), "=f"(gv_0[28]), "=f"(gv_0[29]), "=f"(gv_0[30]), "=f"(gv_0[31])
                                : "r"(tmem_g1 + tl_base + 0));
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (hgrp < gpk) {
                            if (bd_fl == 0.0f) {
                                float _exp2_8 = approx_exp2(bd_gi - s_gt[hb + 15]);
                                float rf0_0 = bd_br * _exp2_8;
                                float _exp2_9 = approx_exp2(bd_gi - s_gt[hb + 31]);
                                float rf1_0 = bd_br * _exp2_9;
                                float c4_0[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_0[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256) * 4)));
                                float l4_0[4];
                                l4_0[0] = rf0_0 * c4_0[0] * gv_0[0];
                                l4_0[1] = rf0_0 * c4_0[1] * gv_0[1];
                                l4_0[2] = rf0_0 * c4_0[2] * gv_0[2];
                                l4_0[3] = rf0_0 * c4_0[3] * gv_0[3];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)(b_row * 68 * 4)), "f"(l4_0[0]), "f"(l4_0[1]), "f"(l4_0[2]), "f"(l4_0[3]) : "memory");
                                float c4_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_1[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 4) * 4)));
                                float l4_1[4];
                                l4_1[0] = rf0_0 * c4_1[0] * gv_0[4];
                                l4_1[1] = rf0_0 * c4_1[1] * gv_0[5];
                                l4_1[2] = rf0_0 * c4_1[2] * gv_0[6];
                                l4_1[3] = rf0_0 * c4_1[3] * gv_0[7];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 4) * 4)), "f"(l4_1[0]), "f"(l4_1[1]), "f"(l4_1[2]), "f"(l4_1[3]) : "memory");
                                float c4_2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_2[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 8) * 4)));
                                float l4_2[4];
                                l4_2[0] = rf0_0 * c4_2[0] * gv_0[8];
                                l4_2[1] = rf0_0 * c4_2[1] * gv_0[9];
                                l4_2[2] = rf0_0 * c4_2[2] * gv_0[10];
                                l4_2[3] = rf0_0 * c4_2[3] * gv_0[11];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 8) * 4)), "f"(l4_2[0]), "f"(l4_2[1]), "f"(l4_2[2]), "f"(l4_2[3]) : "memory");
                                float c4_3[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_3[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 12) * 4)));
                                float l4_3[4];
                                l4_3[0] = rf0_0 * c4_3[0] * gv_0[12];
                                l4_3[1] = rf0_0 * c4_3[1] * gv_0[13];
                                l4_3[2] = rf0_0 * c4_3[2] * gv_0[14];
                                l4_3[3] = rf0_0 * c4_3[3] * gv_0[15];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 12) * 4)), "f"(l4_3[0]), "f"(l4_3[1]), "f"(l4_3[2]), "f"(l4_3[3]) : "memory");
                                float c4_4[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_4[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 16) * 4)));
                                float l4_4[4];
                                l4_4[0] = rf1_0 * c4_4[0] * gv_0[16];
                                l4_4[1] = rf1_0 * c4_4[1] * gv_0[17];
                                l4_4[2] = rf1_0 * c4_4[2] * gv_0[18];
                                l4_4[3] = rf1_0 * c4_4[3] * gv_0[19];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 16) * 4)), "f"(l4_4[0]), "f"(l4_4[1]), "f"(l4_4[2]), "f"(l4_4[3]) : "memory");
                                float c4_5[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_5[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 20) * 4)));
                                float l4_5[4];
                                l4_5[0] = rf1_0 * c4_5[0] * gv_0[20];
                                l4_5[1] = rf1_0 * c4_5[1] * gv_0[21];
                                l4_5[2] = rf1_0 * c4_5[2] * gv_0[22];
                                l4_5[3] = rf1_0 * c4_5[3] * gv_0[23];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 20) * 4)), "f"(l4_5[0]), "f"(l4_5[1]), "f"(l4_5[2]), "f"(l4_5[3]) : "memory");
                                float c4_6[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_6[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 24) * 4)));
                                float l4_6[4];
                                l4_6[0] = rf1_0 * c4_6[0] * gv_0[24];
                                l4_6[1] = rf1_0 * c4_6[1] * gv_0[25];
                                l4_6[2] = rf1_0 * c4_6[2] * gv_0[26];
                                l4_6[3] = rf1_0 * c4_6[3] * gv_0[27];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 24) * 4)), "f"(l4_6[0]), "f"(l4_6[1]), "f"(l4_6[2]), "f"(l4_6[3]) : "memory");
                                float c4_7[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_7[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 28) * 4)));
                                float l4_7[4];
                                l4_7[0] = rf1_0 * c4_7[0] * gv_0[28];
                                l4_7[1] = rf1_0 * c4_7[1] * gv_0[29];
                                l4_7[2] = rf1_0 * c4_7[2] * gv_0[30];
                                l4_7[3] = rf1_0 * c4_7[3] * gv_0[31];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 28) * 4)), "f"(l4_7[0]), "f"(l4_7[1]), "f"(l4_7[2]), "f"(l4_7[3]) : "memory");
                            } else {
                                float g4_0[4];
                                float r4_0[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_0[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)(hb * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_0[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)(p_off * 4)));
                                float l4_8[4];
                                float _exp2_10 = approx_exp2(bd_gi - g4_0[0]);
                                l4_8[0] = bd_br * r4_0[0] * _exp2_10 * gv_0[0];
                                float _exp2_11 = approx_exp2(bd_gi - g4_0[1]);
                                l4_8[1] = bd_br * r4_0[1] * _exp2_11 * gv_0[1];
                                float _exp2_12 = approx_exp2(bd_gi - g4_0[2]);
                                l4_8[2] = bd_br * r4_0[2] * _exp2_12 * gv_0[2];
                                float _exp2_13 = approx_exp2(bd_gi - g4_0[3]);
                                l4_8[3] = bd_br * r4_0[3] * _exp2_13 * gv_0[3];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)(b_row * 68 * 4)), "f"(l4_8[0]), "f"(l4_8[1]), "f"(l4_8[2]), "f"(l4_8[3]) : "memory");
                                float g4_1[4];
                                float r4_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_1[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 4) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_1[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 4) * 4)));
                                float l4_9[4];
                                float _exp2_14 = approx_exp2(bd_gi - g4_1[0]);
                                l4_9[0] = bd_br * r4_1[0] * _exp2_14 * gv_0[4];
                                float _exp2_15 = approx_exp2(bd_gi - g4_1[1]);
                                l4_9[1] = bd_br * r4_1[1] * _exp2_15 * gv_0[5];
                                float _exp2_16 = approx_exp2(bd_gi - g4_1[2]);
                                l4_9[2] = bd_br * r4_1[2] * _exp2_16 * gv_0[6];
                                float _exp2_17 = approx_exp2(bd_gi - g4_1[3]);
                                l4_9[3] = bd_br * r4_1[3] * _exp2_17 * gv_0[7];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 4) * 4)), "f"(l4_9[0]), "f"(l4_9[1]), "f"(l4_9[2]), "f"(l4_9[3]) : "memory");
                                float g4_2[4];
                                float r4_2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_2[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 8) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_2[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 8) * 4)));
                                float l4_10[4];
                                float _exp2_18 = approx_exp2(bd_gi - g4_2[0]);
                                l4_10[0] = bd_br * r4_2[0] * _exp2_18 * gv_0[8];
                                float _exp2_19 = approx_exp2(bd_gi - g4_2[1]);
                                l4_10[1] = bd_br * r4_2[1] * _exp2_19 * gv_0[9];
                                float _exp2_20 = approx_exp2(bd_gi - g4_2[2]);
                                l4_10[2] = bd_br * r4_2[2] * _exp2_20 * gv_0[10];
                                float _exp2_21 = approx_exp2(bd_gi - g4_2[3]);
                                l4_10[3] = bd_br * r4_2[3] * _exp2_21 * gv_0[11];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 8) * 4)), "f"(l4_10[0]), "f"(l4_10[1]), "f"(l4_10[2]), "f"(l4_10[3]) : "memory");
                                float g4_3[4];
                                float r4_3[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_3[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 12) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_3[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 12) * 4)));
                                float l4_11[4];
                                float _exp2_22 = approx_exp2(bd_gi - g4_3[0]);
                                l4_11[0] = bd_br * r4_3[0] * _exp2_22 * gv_0[12];
                                float _exp2_23 = approx_exp2(bd_gi - g4_3[1]);
                                l4_11[1] = bd_br * r4_3[1] * _exp2_23 * gv_0[13];
                                float _exp2_24 = approx_exp2(bd_gi - g4_3[2]);
                                l4_11[2] = bd_br * r4_3[2] * _exp2_24 * gv_0[14];
                                float _exp2_25 = approx_exp2(bd_gi - g4_3[3]);
                                l4_11[3] = bd_br * r4_3[3] * _exp2_25 * gv_0[15];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 12) * 4)), "f"(l4_11[0]), "f"(l4_11[1]), "f"(l4_11[2]), "f"(l4_11[3]) : "memory");
                                float g4_4[4];
                                float r4_4[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_4[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 16) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_4[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 16) * 4)));
                                float l4_12[4];
                                float _exp2_26 = approx_exp2(bd_gi - g4_4[0]);
                                l4_12[0] = bd_br * r4_4[0] * _exp2_26 * gv_0[16];
                                float _exp2_27 = approx_exp2(bd_gi - g4_4[1]);
                                l4_12[1] = bd_br * r4_4[1] * _exp2_27 * gv_0[17];
                                float _exp2_28 = approx_exp2(bd_gi - g4_4[2]);
                                l4_12[2] = bd_br * r4_4[2] * _exp2_28 * gv_0[18];
                                float _exp2_29 = approx_exp2(bd_gi - g4_4[3]);
                                l4_12[3] = bd_br * r4_4[3] * _exp2_29 * gv_0[19];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 16) * 4)), "f"(l4_12[0]), "f"(l4_12[1]), "f"(l4_12[2]), "f"(l4_12[3]) : "memory");
                                float g4_5[4];
                                float r4_5[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_5[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 20) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_5[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 20) * 4)));
                                float l4_13[4];
                                float _exp2_30 = approx_exp2(bd_gi - g4_5[0]);
                                l4_13[0] = bd_br * r4_5[0] * _exp2_30 * gv_0[20];
                                float _exp2_31 = approx_exp2(bd_gi - g4_5[1]);
                                l4_13[1] = bd_br * r4_5[1] * _exp2_31 * gv_0[21];
                                float _exp2_32 = approx_exp2(bd_gi - g4_5[2]);
                                l4_13[2] = bd_br * r4_5[2] * _exp2_32 * gv_0[22];
                                float _exp2_33 = approx_exp2(bd_gi - g4_5[3]);
                                l4_13[3] = bd_br * r4_5[3] * _exp2_33 * gv_0[23];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 20) * 4)), "f"(l4_13[0]), "f"(l4_13[1]), "f"(l4_13[2]), "f"(l4_13[3]) : "memory");
                                float g4_6[4];
                                float r4_6[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_6[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 24) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_6[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 24) * 4)));
                                float l4_14[4];
                                float _exp2_34 = approx_exp2(bd_gi - g4_6[0]);
                                l4_14[0] = bd_br * r4_6[0] * _exp2_34 * gv_0[24];
                                float _exp2_35 = approx_exp2(bd_gi - g4_6[1]);
                                l4_14[1] = bd_br * r4_6[1] * _exp2_35 * gv_0[25];
                                float _exp2_36 = approx_exp2(bd_gi - g4_6[2]);
                                l4_14[2] = bd_br * r4_6[2] * _exp2_36 * gv_0[26];
                                float _exp2_37 = approx_exp2(bd_gi - g4_6[3]);
                                l4_14[3] = bd_br * r4_6[3] * _exp2_37 * gv_0[27];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 24) * 4)), "f"(l4_14[0]), "f"(l4_14[1]), "f"(l4_14[2]), "f"(l4_14[3]) : "memory");
                                float g4_7[4];
                                float r4_7[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_7[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 28) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_7[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 28) * 4)));
                                float l4_15[4];
                                float _exp2_38 = approx_exp2(bd_gi - g4_7[0]);
                                l4_15[0] = bd_br * r4_7[0] * _exp2_38 * gv_0[28];
                                float _exp2_39 = approx_exp2(bd_gi - g4_7[1]);
                                l4_15[1] = bd_br * r4_7[1] * _exp2_39 * gv_0[29];
                                float _exp2_40 = approx_exp2(bd_gi - g4_7[2]);
                                l4_15[2] = bd_br * r4_7[2] * _exp2_40 * gv_0[30];
                                float _exp2_41 = approx_exp2(bd_gi - g4_7[3]);
                                l4_15[3] = bd_br * r4_7[3] * _exp2_41 * gv_0[31];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 28) * 4)), "f"(l4_15[0]), "f"(l4_15[1]), "f"(l4_15[2]), "f"(l4_15[3]) : "memory");
                            }
                        }
                    }
                    if (1 <= pw) {
                        float gv_1[32];
                        if (it % 2 == 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(gv_1[0]), "=f"(gv_1[1]), "=f"(gv_1[2]), "=f"(gv_1[3]), "=f"(gv_1[4]), "=f"(gv_1[5]), "=f"(gv_1[6]), "=f"(gv_1[7]), "=f"(gv_1[8]), "=f"(gv_1[9]), "=f"(gv_1[10]), "=f"(gv_1[11]), "=f"(gv_1[12]), "=f"(gv_1[13]), "=f"(gv_1[14]), "=f"(gv_1[15]), "=f"(gv_1[16]), "=f"(gv_1[17]), "=f"(gv_1[18]), "=f"(gv_1[19]), "=f"(gv_1[20]), "=f"(gv_1[21]), "=f"(gv_1[22]), "=f"(gv_1[23]), "=f"(gv_1[24]), "=f"(gv_1[25]), "=f"(gv_1[26]), "=f"(gv_1[27]), "=f"(gv_1[28]), "=f"(gv_1[29]), "=f"(gv_1[30]), "=f"(gv_1[31])
                                : "r"(tmem_g0 + tl_base + 32));
                        } else {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(gv_1[0]), "=f"(gv_1[1]), "=f"(gv_1[2]), "=f"(gv_1[3]), "=f"(gv_1[4]), "=f"(gv_1[5]), "=f"(gv_1[6]), "=f"(gv_1[7]), "=f"(gv_1[8]), "=f"(gv_1[9]), "=f"(gv_1[10]), "=f"(gv_1[11]), "=f"(gv_1[12]), "=f"(gv_1[13]), "=f"(gv_1[14]), "=f"(gv_1[15]), "=f"(gv_1[16]), "=f"(gv_1[17]), "=f"(gv_1[18]), "=f"(gv_1[19]), "=f"(gv_1[20]), "=f"(gv_1[21]), "=f"(gv_1[22]), "=f"(gv_1[23]), "=f"(gv_1[24]), "=f"(gv_1[25]), "=f"(gv_1[26]), "=f"(gv_1[27]), "=f"(gv_1[28]), "=f"(gv_1[29]), "=f"(gv_1[30]), "=f"(gv_1[31])
                                : "r"(tmem_g1 + tl_base + 32));
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (hgrp < gpk) {
                            if (bd_fl == 0.0f) {
                                float _exp2_42 = approx_exp2(bd_gi - s_gt[hb + 32 + 15]);
                                float rf0_1 = bd_br * _exp2_42;
                                float _exp2_43 = approx_exp2(bd_gi - s_gt[hb + 32 + 31]);
                                float rf1_1 = bd_br * _exp2_43;
                                float c4_8[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_8[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 32) * 4)));
                                float l4_16[4];
                                l4_16[0] = rf0_1 * c4_8[0] * gv_1[0];
                                l4_16[1] = rf0_1 * c4_8[1] * gv_1[1];
                                l4_16[2] = rf0_1 * c4_8[2] * gv_1[2];
                                l4_16[3] = rf0_1 * c4_8[3] * gv_1[3];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 32) * 4)), "f"(l4_16[0]), "f"(l4_16[1]), "f"(l4_16[2]), "f"(l4_16[3]) : "memory");
                                float c4_9[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_9[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 32 + 4) * 4)));
                                float l4_17[4];
                                l4_17[0] = rf0_1 * c4_9[0] * gv_1[4];
                                l4_17[1] = rf0_1 * c4_9[1] * gv_1[5];
                                l4_17[2] = rf0_1 * c4_9[2] * gv_1[6];
                                l4_17[3] = rf0_1 * c4_9[3] * gv_1[7];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 36) * 4)), "f"(l4_17[0]), "f"(l4_17[1]), "f"(l4_17[2]), "f"(l4_17[3]) : "memory");
                                float c4_10[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_10[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 32 + 8) * 4)));
                                float l4_18[4];
                                l4_18[0] = rf0_1 * c4_10[0] * gv_1[8];
                                l4_18[1] = rf0_1 * c4_10[1] * gv_1[9];
                                l4_18[2] = rf0_1 * c4_10[2] * gv_1[10];
                                l4_18[3] = rf0_1 * c4_10[3] * gv_1[11];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 40) * 4)), "f"(l4_18[0]), "f"(l4_18[1]), "f"(l4_18[2]), "f"(l4_18[3]) : "memory");
                                float c4_11[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_11[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 32 + 12) * 4)));
                                float l4_19[4];
                                l4_19[0] = rf0_1 * c4_11[0] * gv_1[12];
                                l4_19[1] = rf0_1 * c4_11[1] * gv_1[13];
                                l4_19[2] = rf0_1 * c4_11[2] * gv_1[14];
                                l4_19[3] = rf0_1 * c4_11[3] * gv_1[15];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 44) * 4)), "f"(l4_19[0]), "f"(l4_19[1]), "f"(l4_19[2]), "f"(l4_19[3]) : "memory");
                                float c4_12[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_12[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 32 + 16) * 4)));
                                float l4_20[4];
                                l4_20[0] = rf1_1 * c4_12[0] * gv_1[16];
                                l4_20[1] = rf1_1 * c4_12[1] * gv_1[17];
                                l4_20[2] = rf1_1 * c4_12[2] * gv_1[18];
                                l4_20[3] = rf1_1 * c4_12[3] * gv_1[19];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 48) * 4)), "f"(l4_20[0]), "f"(l4_20[1]), "f"(l4_20[2]), "f"(l4_20[3]) : "memory");
                                float c4_13[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_13[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 32 + 20) * 4)));
                                float l4_21[4];
                                l4_21[0] = rf1_1 * c4_13[0] * gv_1[20];
                                l4_21[1] = rf1_1 * c4_13[1] * gv_1[21];
                                l4_21[2] = rf1_1 * c4_13[2] * gv_1[22];
                                l4_21[3] = rf1_1 * c4_13[3] * gv_1[23];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 52) * 4)), "f"(l4_21[0]), "f"(l4_21[1]), "f"(l4_21[2]), "f"(l4_21[3]) : "memory");
                                float c4_14[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_14[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 32 + 24) * 4)));
                                float l4_22[4];
                                l4_22[0] = rf1_1 * c4_14[0] * gv_1[24];
                                l4_22[1] = rf1_1 * c4_14[1] * gv_1[25];
                                l4_22[2] = rf1_1 * c4_14[2] * gv_1[26];
                                l4_22[3] = rf1_1 * c4_14[3] * gv_1[27];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 56) * 4)), "f"(l4_22[0]), "f"(l4_22[1]), "f"(l4_22[2]), "f"(l4_22[3]) : "memory");
                                float c4_15[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&c4_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&c4_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c4_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c4_15[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 256 + 32 + 28) * 4)));
                                float l4_23[4];
                                l4_23[0] = rf1_1 * c4_15[0] * gv_1[28];
                                l4_23[1] = rf1_1 * c4_15[1] * gv_1[29];
                                l4_23[2] = rf1_1 * c4_15[2] * gv_1[30];
                                l4_23[3] = rf1_1 * c4_15[3] * gv_1[31];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 60) * 4)), "f"(l4_23[0]), "f"(l4_23[1]), "f"(l4_23[2]), "f"(l4_23[3]) : "memory");
                            } else {
                                float g4_8[4];
                                float r4_8[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_8[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 32) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_8[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 32) * 4)));
                                float l4_24[4];
                                float _exp2_44 = approx_exp2(bd_gi - g4_8[0]);
                                l4_24[0] = bd_br * r4_8[0] * _exp2_44 * gv_1[0];
                                float _exp2_45 = approx_exp2(bd_gi - g4_8[1]);
                                l4_24[1] = bd_br * r4_8[1] * _exp2_45 * gv_1[1];
                                float _exp2_46 = approx_exp2(bd_gi - g4_8[2]);
                                l4_24[2] = bd_br * r4_8[2] * _exp2_46 * gv_1[2];
                                float _exp2_47 = approx_exp2(bd_gi - g4_8[3]);
                                l4_24[3] = bd_br * r4_8[3] * _exp2_47 * gv_1[3];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 32) * 4)), "f"(l4_24[0]), "f"(l4_24[1]), "f"(l4_24[2]), "f"(l4_24[3]) : "memory");
                                float g4_9[4];
                                float r4_9[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_9[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 32 + 4) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_9[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 32 + 4) * 4)));
                                float l4_25[4];
                                float _exp2_48 = approx_exp2(bd_gi - g4_9[0]);
                                l4_25[0] = bd_br * r4_9[0] * _exp2_48 * gv_1[4];
                                float _exp2_49 = approx_exp2(bd_gi - g4_9[1]);
                                l4_25[1] = bd_br * r4_9[1] * _exp2_49 * gv_1[5];
                                float _exp2_50 = approx_exp2(bd_gi - g4_9[2]);
                                l4_25[2] = bd_br * r4_9[2] * _exp2_50 * gv_1[6];
                                float _exp2_51 = approx_exp2(bd_gi - g4_9[3]);
                                l4_25[3] = bd_br * r4_9[3] * _exp2_51 * gv_1[7];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 36) * 4)), "f"(l4_25[0]), "f"(l4_25[1]), "f"(l4_25[2]), "f"(l4_25[3]) : "memory");
                                float g4_10[4];
                                float r4_10[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_10[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 32 + 8) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_10[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 32 + 8) * 4)));
                                float l4_26[4];
                                float _exp2_52 = approx_exp2(bd_gi - g4_10[0]);
                                l4_26[0] = bd_br * r4_10[0] * _exp2_52 * gv_1[8];
                                float _exp2_53 = approx_exp2(bd_gi - g4_10[1]);
                                l4_26[1] = bd_br * r4_10[1] * _exp2_53 * gv_1[9];
                                float _exp2_54 = approx_exp2(bd_gi - g4_10[2]);
                                l4_26[2] = bd_br * r4_10[2] * _exp2_54 * gv_1[10];
                                float _exp2_55 = approx_exp2(bd_gi - g4_10[3]);
                                l4_26[3] = bd_br * r4_10[3] * _exp2_55 * gv_1[11];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 40) * 4)), "f"(l4_26[0]), "f"(l4_26[1]), "f"(l4_26[2]), "f"(l4_26[3]) : "memory");
                                float g4_11[4];
                                float r4_11[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_11[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 32 + 12) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_11[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 32 + 12) * 4)));
                                float l4_27[4];
                                float _exp2_56 = approx_exp2(bd_gi - g4_11[0]);
                                l4_27[0] = bd_br * r4_11[0] * _exp2_56 * gv_1[12];
                                float _exp2_57 = approx_exp2(bd_gi - g4_11[1]);
                                l4_27[1] = bd_br * r4_11[1] * _exp2_57 * gv_1[13];
                                float _exp2_58 = approx_exp2(bd_gi - g4_11[2]);
                                l4_27[2] = bd_br * r4_11[2] * _exp2_58 * gv_1[14];
                                float _exp2_59 = approx_exp2(bd_gi - g4_11[3]);
                                l4_27[3] = bd_br * r4_11[3] * _exp2_59 * gv_1[15];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 44) * 4)), "f"(l4_27[0]), "f"(l4_27[1]), "f"(l4_27[2]), "f"(l4_27[3]) : "memory");
                                float g4_12[4];
                                float r4_12[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_12[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 32 + 16) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_12[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 32 + 16) * 4)));
                                float l4_28[4];
                                float _exp2_60 = approx_exp2(bd_gi - g4_12[0]);
                                l4_28[0] = bd_br * r4_12[0] * _exp2_60 * gv_1[16];
                                float _exp2_61 = approx_exp2(bd_gi - g4_12[1]);
                                l4_28[1] = bd_br * r4_12[1] * _exp2_61 * gv_1[17];
                                float _exp2_62 = approx_exp2(bd_gi - g4_12[2]);
                                l4_28[2] = bd_br * r4_12[2] * _exp2_62 * gv_1[18];
                                float _exp2_63 = approx_exp2(bd_gi - g4_12[3]);
                                l4_28[3] = bd_br * r4_12[3] * _exp2_63 * gv_1[19];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 48) * 4)), "f"(l4_28[0]), "f"(l4_28[1]), "f"(l4_28[2]), "f"(l4_28[3]) : "memory");
                                float g4_13[4];
                                float r4_13[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_13[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 32 + 20) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_13[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 32 + 20) * 4)));
                                float l4_29[4];
                                float _exp2_64 = approx_exp2(bd_gi - g4_13[0]);
                                l4_29[0] = bd_br * r4_13[0] * _exp2_64 * gv_1[20];
                                float _exp2_65 = approx_exp2(bd_gi - g4_13[1]);
                                l4_29[1] = bd_br * r4_13[1] * _exp2_65 * gv_1[21];
                                float _exp2_66 = approx_exp2(bd_gi - g4_13[2]);
                                l4_29[2] = bd_br * r4_13[2] * _exp2_66 * gv_1[22];
                                float _exp2_67 = approx_exp2(bd_gi - g4_13[3]);
                                l4_29[3] = bd_br * r4_13[3] * _exp2_67 * gv_1[23];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 52) * 4)), "f"(l4_29[0]), "f"(l4_29[1]), "f"(l4_29[2]), "f"(l4_29[3]) : "memory");
                                float g4_14[4];
                                float r4_14[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_14[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 32 + 24) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_14[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 32 + 24) * 4)));
                                float l4_30[4];
                                float _exp2_68 = approx_exp2(bd_gi - g4_14[0]);
                                l4_30[0] = bd_br * r4_14[0] * _exp2_68 * gv_1[24];
                                float _exp2_69 = approx_exp2(bd_gi - g4_14[1]);
                                l4_30[1] = bd_br * r4_14[1] * _exp2_69 * gv_1[25];
                                float _exp2_70 = approx_exp2(bd_gi - g4_14[2]);
                                l4_30[2] = bd_br * r4_14[2] * _exp2_70 * gv_1[26];
                                float _exp2_71 = approx_exp2(bd_gi - g4_14[3]);
                                l4_30[3] = bd_br * r4_14[3] * _exp2_71 * gv_1[27];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 56) * 4)), "f"(l4_30[0]), "f"(l4_30[1]), "f"(l4_30[2]), "f"(l4_30[3]) : "memory");
                                float g4_15[4];
                                float r4_15[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g4_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&g4_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g4_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g4_15[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((hb + 32 + 28) * 4)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&r4_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&r4_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r4_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r4_15[(0) + 3]))
                                    : "r"(s_gt_addr + (unsigned int)((p_off + 32 + 28) * 4)));
                                float l4_31[4];
                                float _exp2_72 = approx_exp2(bd_gi - g4_15[0]);
                                l4_31[0] = bd_br * r4_15[0] * _exp2_72 * gv_1[28];
                                float _exp2_73 = approx_exp2(bd_gi - g4_15[1]);
                                l4_31[1] = bd_br * r4_15[1] * _exp2_73 * gv_1[29];
                                float _exp2_74 = approx_exp2(bd_gi - g4_15[2]);
                                l4_31[2] = bd_br * r4_15[2] * _exp2_74 * gv_1[30];
                                float _exp2_75 = approx_exp2(bd_gi - g4_15[3]);
                                l4_31[3] = bd_br * r4_15[3] * _exp2_75 * gv_1[31];
                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(lb + (unsigned int)((b_row * 68 + 60) * 4)), "f"(l4_31[0]), "f"(l4_31[1]), "f"(l4_31[2]), "f"(l4_31[3]) : "memory");
                            }
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(g_free_addr + (gfl_c) * 8);
                gfl_c += 1;
                if (gfl_c == 2) { gfl_c = 0; _phase_g_full ^= 1; }
                if (it % 2 == par) {
                    __syncwarp();
                    if (hgrp < gpk) {
                        int dg_b = 16 * (2 * pw + lane_1 / 16);
                        int dg_cc = lane_1 % 16;
                        float acc[16];
                        acc[0] = ((0 == dg_cc) ? 1.0f : 0.0f);
                        acc[1] = ((1 == dg_cc) ? 1.0f : 0.0f);
                        acc[2] = ((2 == dg_cc) ? 1.0f : 0.0f);
                        acc[3] = ((3 == dg_cc) ? 1.0f : 0.0f);
                        acc[4] = ((4 == dg_cc) ? 1.0f : 0.0f);
                        acc[5] = ((5 == dg_cc) ? 1.0f : 0.0f);
                        acc[6] = ((6 == dg_cc) ? 1.0f : 0.0f);
                        acc[7] = ((7 == dg_cc) ? 1.0f : 0.0f);
                        acc[8] = ((8 == dg_cc) ? 1.0f : 0.0f);
                        acc[9] = ((9 == dg_cc) ? 1.0f : 0.0f);
                        acc[10] = ((10 == dg_cc) ? 1.0f : 0.0f);
                        acc[11] = ((11 == dg_cc) ? 1.0f : 0.0f);
                        acc[12] = ((12 == dg_cc) ? 1.0f : 0.0f);
                        acc[13] = ((13 == dg_cc) ? 1.0f : 0.0f);
                        acc[14] = ((14 == dg_cc) ? 1.0f : 0.0f);
                        acc[15] = ((15 == dg_cc) ? 1.0f : 0.0f);
                        float lr_0[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_0[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 1) * 68 + dg_b) * 4)));
                        acc[1] = acc[1] - lr_0[0] * acc[0];
                        float lr_1[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_1[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 2) * 68 + dg_b) * 4)));
                        acc[2] = acc[2] - lr_1[0] * acc[0];
                        acc[2] = acc[2] - lr_1[1] * acc[1];
                        float lr_2[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_2[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 3) * 68 + dg_b) * 4)));
                        acc[3] = acc[3] - lr_2[0] * acc[0];
                        acc[3] = acc[3] - lr_2[1] * acc[1];
                        acc[3] = acc[3] - lr_2[2] * acc[2];
                        float lr_3[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_3[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 4) * 68 + dg_b) * 4)));
                        acc[4] = acc[4] - lr_3[0] * acc[0];
                        acc[4] = acc[4] - lr_3[1] * acc[1];
                        acc[4] = acc[4] - lr_3[2] * acc[2];
                        acc[4] = acc[4] - lr_3[3] * acc[3];
                        float lr_4[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_4[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 5) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_4[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 5) * 68 + (dg_b + 4)) * 4)));
                        acc[5] = acc[5] - lr_4[0] * acc[0];
                        acc[5] = acc[5] - lr_4[1] * acc[1];
                        acc[5] = acc[5] - lr_4[2] * acc[2];
                        acc[5] = acc[5] - lr_4[3] * acc[3];
                        acc[5] = acc[5] - lr_4[4] * acc[4];
                        float lr_5[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_5[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 6) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_5[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 6) * 68 + (dg_b + 4)) * 4)));
                        acc[6] = acc[6] - lr_5[0] * acc[0];
                        acc[6] = acc[6] - lr_5[1] * acc[1];
                        acc[6] = acc[6] - lr_5[2] * acc[2];
                        acc[6] = acc[6] - lr_5[3] * acc[3];
                        acc[6] = acc[6] - lr_5[4] * acc[4];
                        acc[6] = acc[6] - lr_5[5] * acc[5];
                        float lr_6[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_6[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 7) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_6[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 7) * 68 + (dg_b + 4)) * 4)));
                        acc[7] = acc[7] - lr_6[0] * acc[0];
                        acc[7] = acc[7] - lr_6[1] * acc[1];
                        acc[7] = acc[7] - lr_6[2] * acc[2];
                        acc[7] = acc[7] - lr_6[3] * acc[3];
                        acc[7] = acc[7] - lr_6[4] * acc[4];
                        acc[7] = acc[7] - lr_6[5] * acc[5];
                        acc[7] = acc[7] - lr_6[6] * acc[6];
                        float lr_7[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_7[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 8) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_7[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 8) * 68 + (dg_b + 4)) * 4)));
                        acc[8] = acc[8] - lr_7[0] * acc[0];
                        acc[8] = acc[8] - lr_7[1] * acc[1];
                        acc[8] = acc[8] - lr_7[2] * acc[2];
                        acc[8] = acc[8] - lr_7[3] * acc[3];
                        acc[8] = acc[8] - lr_7[4] * acc[4];
                        acc[8] = acc[8] - lr_7[5] * acc[5];
                        acc[8] = acc[8] - lr_7[6] * acc[6];
                        acc[8] = acc[8] - lr_7[7] * acc[7];
                        float lr_8[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 9) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 9) * 68 + (dg_b + 4)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_8[(8) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 9) * 68 + (dg_b + 8)) * 4)));
                        acc[9] = acc[9] - lr_8[0] * acc[0];
                        acc[9] = acc[9] - lr_8[1] * acc[1];
                        acc[9] = acc[9] - lr_8[2] * acc[2];
                        acc[9] = acc[9] - lr_8[3] * acc[3];
                        acc[9] = acc[9] - lr_8[4] * acc[4];
                        acc[9] = acc[9] - lr_8[5] * acc[5];
                        acc[9] = acc[9] - lr_8[6] * acc[6];
                        acc[9] = acc[9] - lr_8[7] * acc[7];
                        acc[9] = acc[9] - lr_8[8] * acc[8];
                        float lr_9[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 10) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 10) * 68 + (dg_b + 4)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_9[(8) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 10) * 68 + (dg_b + 8)) * 4)));
                        acc[10] = acc[10] - lr_9[0] * acc[0];
                        acc[10] = acc[10] - lr_9[1] * acc[1];
                        acc[10] = acc[10] - lr_9[2] * acc[2];
                        acc[10] = acc[10] - lr_9[3] * acc[3];
                        acc[10] = acc[10] - lr_9[4] * acc[4];
                        acc[10] = acc[10] - lr_9[5] * acc[5];
                        acc[10] = acc[10] - lr_9[6] * acc[6];
                        acc[10] = acc[10] - lr_9[7] * acc[7];
                        acc[10] = acc[10] - lr_9[8] * acc[8];
                        acc[10] = acc[10] - lr_9[9] * acc[9];
                        float lr_10[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 11) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 11) * 68 + (dg_b + 4)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_10[(8) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 11) * 68 + (dg_b + 8)) * 4)));
                        acc[11] = acc[11] - lr_10[0] * acc[0];
                        acc[11] = acc[11] - lr_10[1] * acc[1];
                        acc[11] = acc[11] - lr_10[2] * acc[2];
                        acc[11] = acc[11] - lr_10[3] * acc[3];
                        acc[11] = acc[11] - lr_10[4] * acc[4];
                        acc[11] = acc[11] - lr_10[5] * acc[5];
                        acc[11] = acc[11] - lr_10[6] * acc[6];
                        acc[11] = acc[11] - lr_10[7] * acc[7];
                        acc[11] = acc[11] - lr_10[8] * acc[8];
                        acc[11] = acc[11] - lr_10[9] * acc[9];
                        acc[11] = acc[11] - lr_10[10] * acc[10];
                        float lr_11[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 12) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 12) * 68 + (dg_b + 4)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_11[(8) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 12) * 68 + (dg_b + 8)) * 4)));
                        acc[12] = acc[12] - lr_11[0] * acc[0];
                        acc[12] = acc[12] - lr_11[1] * acc[1];
                        acc[12] = acc[12] - lr_11[2] * acc[2];
                        acc[12] = acc[12] - lr_11[3] * acc[3];
                        acc[12] = acc[12] - lr_11[4] * acc[4];
                        acc[12] = acc[12] - lr_11[5] * acc[5];
                        acc[12] = acc[12] - lr_11[6] * acc[6];
                        acc[12] = acc[12] - lr_11[7] * acc[7];
                        acc[12] = acc[12] - lr_11[8] * acc[8];
                        acc[12] = acc[12] - lr_11[9] * acc[9];
                        acc[12] = acc[12] - lr_11[10] * acc[10];
                        acc[12] = acc[12] - lr_11[11] * acc[11];
                        float lr_12[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 13) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 13) * 68 + (dg_b + 4)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(8) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 13) * 68 + (dg_b + 8)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_12[(12) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 13) * 68 + (dg_b + 12)) * 4)));
                        acc[13] = acc[13] - lr_12[0] * acc[0];
                        acc[13] = acc[13] - lr_12[1] * acc[1];
                        acc[13] = acc[13] - lr_12[2] * acc[2];
                        acc[13] = acc[13] - lr_12[3] * acc[3];
                        acc[13] = acc[13] - lr_12[4] * acc[4];
                        acc[13] = acc[13] - lr_12[5] * acc[5];
                        acc[13] = acc[13] - lr_12[6] * acc[6];
                        acc[13] = acc[13] - lr_12[7] * acc[7];
                        acc[13] = acc[13] - lr_12[8] * acc[8];
                        acc[13] = acc[13] - lr_12[9] * acc[9];
                        acc[13] = acc[13] - lr_12[10] * acc[10];
                        acc[13] = acc[13] - lr_12[11] * acc[11];
                        acc[13] = acc[13] - lr_12[12] * acc[12];
                        float lr_13[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 14) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 14) * 68 + (dg_b + 4)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(8) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 14) * 68 + (dg_b + 8)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_13[(12) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 14) * 68 + (dg_b + 12)) * 4)));
                        acc[14] = acc[14] - lr_13[0] * acc[0];
                        acc[14] = acc[14] - lr_13[1] * acc[1];
                        acc[14] = acc[14] - lr_13[2] * acc[2];
                        acc[14] = acc[14] - lr_13[3] * acc[3];
                        acc[14] = acc[14] - lr_13[4] * acc[4];
                        acc[14] = acc[14] - lr_13[5] * acc[5];
                        acc[14] = acc[14] - lr_13[6] * acc[6];
                        acc[14] = acc[14] - lr_13[7] * acc[7];
                        acc[14] = acc[14] - lr_13[8] * acc[8];
                        acc[14] = acc[14] - lr_13[9] * acc[9];
                        acc[14] = acc[14] - lr_13[10] * acc[10];
                        acc[14] = acc[14] - lr_13[11] * acc[11];
                        acc[14] = acc[14] - lr_13[12] * acc[12];
                        acc[14] = acc[14] - lr_13[13] * acc[13];
                        float lr_14[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(0) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 15) * 68 + dg_b) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(4) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 15) * 68 + (dg_b + 4)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(8) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 15) * 68 + (dg_b + 8)) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&lr_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lr_14[(12) + 3]))
                            : "r"(lb + (unsigned int)(((dg_b + 15) * 68 + (dg_b + 12)) * 4)));
                        acc[15] = acc[15] - lr_14[0] * acc[0];
                        acc[15] = acc[15] - lr_14[1] * acc[1];
                        acc[15] = acc[15] - lr_14[2] * acc[2];
                        acc[15] = acc[15] - lr_14[3] * acc[3];
                        acc[15] = acc[15] - lr_14[4] * acc[4];
                        acc[15] = acc[15] - lr_14[5] * acc[5];
                        acc[15] = acc[15] - lr_14[6] * acc[6];
                        acc[15] = acc[15] - lr_14[7] * acc[7];
                        acc[15] = acc[15] - lr_14[8] * acc[8];
                        acc[15] = acc[15] - lr_14[9] * acc[9];
                        acc[15] = acc[15] - lr_14[10] * acc[10];
                        acc[15] = acc[15] - lr_14[11] * acc[11];
                        acc[15] = acc[15] - lr_14[12] * acc[12];
                        acc[15] = acc[15] - lr_14[13] * acc[13];
                        acc[15] = acc[15] - lr_14[14] * acc[14];
                        __syncwarp();
                        {
                            uint32_t _addr_76 = static_cast<uint32_t>(lb + (unsigned int)((dg_b * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(acc[0]) : "memory");
                        }
                        {
                            uint32_t _addr_77 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 1) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(acc[1]) : "memory");
                        }
                        {
                            uint32_t _addr_78 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 2) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(acc[2]) : "memory");
                        }
                        {
                            uint32_t _addr_79 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 3) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_79), "f"(acc[3]) : "memory");
                        }
                        {
                            uint32_t _addr_80 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 4) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(acc[4]) : "memory");
                        }
                        {
                            uint32_t _addr_81 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 5) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_81), "f"(acc[5]) : "memory");
                        }
                        {
                            uint32_t _addr_82 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 6) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_82), "f"(acc[6]) : "memory");
                        }
                        {
                            uint32_t _addr_83 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 7) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_83), "f"(acc[7]) : "memory");
                        }
                        {
                            uint32_t _addr_84 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 8) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_84), "f"(acc[8]) : "memory");
                        }
                        {
                            uint32_t _addr_85 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 9) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_85), "f"(acc[9]) : "memory");
                        }
                        {
                            uint32_t _addr_86 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 10) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_86), "f"(acc[10]) : "memory");
                        }
                        {
                            uint32_t _addr_87 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 11) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_87), "f"(acc[11]) : "memory");
                        }
                        {
                            uint32_t _addr_88 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 12) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_88), "f"(acc[12]) : "memory");
                        }
                        {
                            uint32_t _addr_89 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 13) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_89), "f"(acc[13]) : "memory");
                        }
                        {
                            uint32_t _addr_90 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 14) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_90), "f"(acc[14]) : "memory");
                        }
                        {
                            uint32_t _addr_91 = static_cast<uint32_t>(lb + (unsigned int)(((dg_b + 15) * 68 + (dg_b + dg_cc)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_91), "f"(acc[15]) : "memory");
                        }
                        __syncwarp();
                    }
                    if (hgrp < gpk) {
                        int m1_i = 32 * pw + 16;
                        int m1_j = 32 * pw;
                        int m1_rt = 16 * pw;
                        float ra_b[4];
                        float ra_s[4];
                        float rc_b[4];
                        float rc_s[4];
                        float av_0[4];
                        av_0[0] = s_g[gw + ((m1_i + gq) * 68 + (m1_j + tq))];
                        av_0[1] = s_g[gw + ((m1_i + gq + 8) * 68 + (m1_j + tq))];
                        av_0[2] = s_g[gw + ((m1_i + gq) * 68 + (m1_j + tq + 4))];
                        av_0[3] = s_g[gw + ((m1_i + gq + 8) * 68 + (m1_j + tq + 4))];
                        float bva_0[2];
                        bva_0[0] = s_g[gw + ((m1_j + tq) * 68 + (m1_j + gq))];
                        bva_0[1] = s_g[gw + ((m1_j + tq + 4) * 68 + (m1_j + gq))];
                        unsigned int ah[4];
                        unsigned int al[4];
                        unsigned int bh[2];
                        unsigned int bl[2];
                        ah[0] = __as_u32(av_0[0]) + 4096 & 4294959104u;
                        al[0] = __as_u32(av_0[0] - __uint_as_float(ah[0]));
                        ah[1] = __as_u32(av_0[1]) + 4096 & 4294959104u;
                        al[1] = __as_u32(av_0[1] - __uint_as_float(ah[1]));
                        ah[2] = __as_u32(av_0[2]) + 4096 & 4294959104u;
                        al[2] = __as_u32(av_0[2] - __uint_as_float(ah[2]));
                        ah[3] = __as_u32(av_0[3]) + 4096 & 4294959104u;
                        al[3] = __as_u32(av_0[3] - __uint_as_float(ah[3]));
                        bh[0] = __as_u32(bva_0[0]) + 4096 & 4294959104u;
                        bl[0] = __as_u32(bva_0[0] - __uint_as_float(bh[0]));
                        bh[1] = __as_u32(bva_0[1]) + 4096 & 4294959104u;
                        bl[1] = __as_u32(bva_0[1] - __uint_as_float(bh[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(ra_s[0]), "=f"(ra_s[1]), "=f"(ra_s[2]), "=f"(ra_s[3])
                            : "r"(al[0]), "r"(al[1]), "r"(al[2]), "r"(al[3]), "r"(bh[0]), "r"(bh[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(ra_b[0]), "=f"(ra_b[1]), "=f"(ra_b[2]), "=f"(ra_b[3])
                            : "r"(ah[0]), "r"(ah[1]), "r"(ah[2]), "r"(ah[3]), "r"(bh[0]), "r"(bh[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(ra_s[0]), "+f"(ra_s[1]), "+f"(ra_s[2]), "+f"(ra_s[3])
                            : "r"(ah[0]), "r"(ah[1]), "r"(ah[2]), "r"(ah[3]), "r"(bl[0]), "r"(bl[1]));
                        float av_1[4];
                        av_1[0] = s_g[gw + ((m1_i + gq) * 68 + (m1_j + 8 + tq))];
                        av_1[1] = s_g[gw + ((m1_i + gq + 8) * 68 + (m1_j + 8 + tq))];
                        av_1[2] = s_g[gw + ((m1_i + gq) * 68 + (m1_j + 8 + tq + 4))];
                        av_1[3] = s_g[gw + ((m1_i + gq + 8) * 68 + (m1_j + 8 + tq + 4))];
                        float bva_1[2];
                        bva_1[0] = s_g[gw + ((m1_j + 8 + tq) * 68 + (m1_j + gq))];
                        bva_1[1] = s_g[gw + ((m1_j + 8 + tq + 4) * 68 + (m1_j + gq))];
                        unsigned int ah_0[4];
                        unsigned int al_0[4];
                        unsigned int bh_0[2];
                        unsigned int bl_0[2];
                        ah_0[0] = __as_u32(av_1[0]) + 4096 & 4294959104u;
                        al_0[0] = __as_u32(av_1[0] - __uint_as_float(ah_0[0]));
                        ah_0[1] = __as_u32(av_1[1]) + 4096 & 4294959104u;
                        al_0[1] = __as_u32(av_1[1] - __uint_as_float(ah_0[1]));
                        ah_0[2] = __as_u32(av_1[2]) + 4096 & 4294959104u;
                        al_0[2] = __as_u32(av_1[2] - __uint_as_float(ah_0[2]));
                        ah_0[3] = __as_u32(av_1[3]) + 4096 & 4294959104u;
                        al_0[3] = __as_u32(av_1[3] - __uint_as_float(ah_0[3]));
                        bh_0[0] = __as_u32(bva_1[0]) + 4096 & 4294959104u;
                        bl_0[0] = __as_u32(bva_1[0] - __uint_as_float(bh_0[0]));
                        bh_0[1] = __as_u32(bva_1[1]) + 4096 & 4294959104u;
                        bl_0[1] = __as_u32(bva_1[1] - __uint_as_float(bh_0[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(ra_s[0]), "+f"(ra_s[1]), "+f"(ra_s[2]), "+f"(ra_s[3])
                            : "r"(al_0[0]), "r"(al_0[1]), "r"(al_0[2]), "r"(al_0[3]), "r"(bh_0[0]), "r"(bh_0[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(ra_b[0]), "+f"(ra_b[1]), "+f"(ra_b[2]), "+f"(ra_b[3])
                            : "r"(ah_0[0]), "r"(ah_0[1]), "r"(ah_0[2]), "r"(ah_0[3]), "r"(bh_0[0]), "r"(bh_0[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(ra_s[0]), "+f"(ra_s[1]), "+f"(ra_s[2]), "+f"(ra_s[3])
                            : "r"(ah_0[0]), "r"(ah_0[1]), "r"(ah_0[2]), "r"(ah_0[3]), "r"(bl_0[0]), "r"(bl_0[1]));
                        float bvc_0[2];
                        bvc_0[0] = s_g[gw + ((m1_j + 8 + tq) * 68 + (m1_j + 8 + gq))];
                        bvc_0[1] = s_g[gw + ((m1_j + 8 + tq + 4) * 68 + (m1_j + 8 + gq))];
                        unsigned int ah_1[4];
                        unsigned int al_1[4];
                        unsigned int bh_1[2];
                        unsigned int bl_1[2];
                        ah_1[0] = __as_u32(av_1[0]) + 4096 & 4294959104u;
                        al_1[0] = __as_u32(av_1[0] - __uint_as_float(ah_1[0]));
                        ah_1[1] = __as_u32(av_1[1]) + 4096 & 4294959104u;
                        al_1[1] = __as_u32(av_1[1] - __uint_as_float(ah_1[1]));
                        ah_1[2] = __as_u32(av_1[2]) + 4096 & 4294959104u;
                        al_1[2] = __as_u32(av_1[2] - __uint_as_float(ah_1[2]));
                        ah_1[3] = __as_u32(av_1[3]) + 4096 & 4294959104u;
                        al_1[3] = __as_u32(av_1[3] - __uint_as_float(ah_1[3]));
                        bh_1[0] = __as_u32(bvc_0[0]) + 4096 & 4294959104u;
                        bl_1[0] = __as_u32(bvc_0[0] - __uint_as_float(bh_1[0]));
                        bh_1[1] = __as_u32(bvc_0[1]) + 4096 & 4294959104u;
                        bl_1[1] = __as_u32(bvc_0[1] - __uint_as_float(bh_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(rc_s[0]), "=f"(rc_s[1]), "=f"(rc_s[2]), "=f"(rc_s[3])
                            : "r"(al_1[0]), "r"(al_1[1]), "r"(al_1[2]), "r"(al_1[3]), "r"(bh_1[0]), "r"(bh_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(rc_b[0]), "=f"(rc_b[1]), "=f"(rc_b[2]), "=f"(rc_b[3])
                            : "r"(ah_1[0]), "r"(ah_1[1]), "r"(ah_1[2]), "r"(ah_1[3]), "r"(bh_1[0]), "r"(bh_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(rc_s[0]), "+f"(rc_s[1]), "+f"(rc_s[2]), "+f"(rc_s[3])
                            : "r"(ah_1[0]), "r"(ah_1[1]), "r"(ah_1[2]), "r"(ah_1[3]), "r"(bl_1[0]), "r"(bl_1[1]));
                        {
                            uint32_t _addr_92 = static_cast<uint32_t>(lb + (unsigned int)(((m1_rt + 2 * tq) * 68 + (32 + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_92), "f"(ra_b[0] + ra_s[0]) : "memory");
                        }
                        {
                            uint32_t _addr_93 = static_cast<uint32_t>(lb + (unsigned int)(((m1_rt + 8 + 2 * tq) * 68 + (32 + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_93), "f"(rc_b[0] + rc_s[0]) : "memory");
                        }
                        {
                            uint32_t _addr_94 = static_cast<uint32_t>(lb + (unsigned int)(((m1_rt + 2 * tq + 1) * 68 + (32 + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_94), "f"(ra_b[1] + ra_s[1]) : "memory");
                        }
                        {
                            uint32_t _addr_95 = static_cast<uint32_t>(lb + (unsigned int)(((m1_rt + 8 + 2 * tq + 1) * 68 + (32 + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_95), "f"(rc_b[1] + rc_s[1]) : "memory");
                        }
                        {
                            uint32_t _addr_96 = static_cast<uint32_t>(lb + (unsigned int)(((m1_rt + 2 * tq) * 68 + (32 + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_96), "f"(ra_b[2] + ra_s[2]) : "memory");
                        }
                        {
                            uint32_t _addr_97 = static_cast<uint32_t>(lb + (unsigned int)(((m1_rt + 8 + 2 * tq) * 68 + (32 + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_97), "f"(rc_b[2] + rc_s[2]) : "memory");
                        }
                        {
                            uint32_t _addr_98 = static_cast<uint32_t>(lb + (unsigned int)(((m1_rt + 2 * tq + 1) * 68 + (32 + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_98), "f"(ra_b[3] + ra_s[3]) : "memory");
                        }
                        {
                            uint32_t _addr_99 = static_cast<uint32_t>(lb + (unsigned int)(((m1_rt + 8 + 2 * tq + 1) * 68 + (32 + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_99), "f"(rc_b[3] + rc_s[3]) : "memory");
                        }
                        __syncwarp();
                        float xa_b[4];
                        float xa_s[4];
                        float xc_b[4];
                        float xc_s[4];
                        float av2_0[4];
                        av2_0[0] = s_g[gw + ((m1_i + gq) * 68 + (m1_i + tq))];
                        av2_0[1] = s_g[gw + ((m1_i + gq + 8) * 68 + (m1_i + tq))];
                        av2_0[2] = s_g[gw + ((m1_i + gq) * 68 + (m1_i + tq + 4))];
                        av2_0[3] = s_g[gw + ((m1_i + gq + 8) * 68 + (m1_i + tq + 4))];
                        float bw0_0[2];
                        float bw1_0[2];
                        bw0_0[0] = s_g[gw + ((m1_rt + gq) * 68 + (32 + tq))];
                        bw1_0[0] = s_g[gw + ((m1_rt + 8 + gq) * 68 + (32 + tq))];
                        bw0_0[1] = s_g[gw + ((m1_rt + gq) * 68 + (32 + tq + 4))];
                        bw1_0[1] = s_g[gw + ((m1_rt + 8 + gq) * 68 + (32 + tq + 4))];
                        unsigned int ah_2[4];
                        unsigned int al_2[4];
                        unsigned int bh_2[2];
                        unsigned int bl_2[2];
                        ah_2[0] = __as_u32(av2_0[0]) + 4096 & 4294959104u;
                        al_2[0] = __as_u32(av2_0[0] - __uint_as_float(ah_2[0]));
                        ah_2[1] = __as_u32(av2_0[1]) + 4096 & 4294959104u;
                        al_2[1] = __as_u32(av2_0[1] - __uint_as_float(ah_2[1]));
                        ah_2[2] = __as_u32(av2_0[2]) + 4096 & 4294959104u;
                        al_2[2] = __as_u32(av2_0[2] - __uint_as_float(ah_2[2]));
                        ah_2[3] = __as_u32(av2_0[3]) + 4096 & 4294959104u;
                        al_2[3] = __as_u32(av2_0[3] - __uint_as_float(ah_2[3]));
                        bh_2[0] = __as_u32(bw0_0[0]) + 4096 & 4294959104u;
                        bl_2[0] = __as_u32(bw0_0[0] - __uint_as_float(bh_2[0]));
                        bh_2[1] = __as_u32(bw0_0[1]) + 4096 & 4294959104u;
                        bl_2[1] = __as_u32(bw0_0[1] - __uint_as_float(bh_2[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(xa_s[0]), "=f"(xa_s[1]), "=f"(xa_s[2]), "=f"(xa_s[3])
                            : "r"(al_2[0]), "r"(al_2[1]), "r"(al_2[2]), "r"(al_2[3]), "r"(bh_2[0]), "r"(bh_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(xa_b[0]), "=f"(xa_b[1]), "=f"(xa_b[2]), "=f"(xa_b[3])
                            : "r"(ah_2[0]), "r"(ah_2[1]), "r"(ah_2[2]), "r"(ah_2[3]), "r"(bh_2[0]), "r"(bh_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(xa_s[0]), "+f"(xa_s[1]), "+f"(xa_s[2]), "+f"(xa_s[3])
                            : "r"(ah_2[0]), "r"(ah_2[1]), "r"(ah_2[2]), "r"(ah_2[3]), "r"(bl_2[0]), "r"(bl_2[1]));
                        unsigned int ah_3[4];
                        unsigned int al_3[4];
                        unsigned int bh_3[2];
                        unsigned int bl_3[2];
                        ah_3[0] = __as_u32(av2_0[0]) + 4096 & 4294959104u;
                        al_3[0] = __as_u32(av2_0[0] - __uint_as_float(ah_3[0]));
                        ah_3[1] = __as_u32(av2_0[1]) + 4096 & 4294959104u;
                        al_3[1] = __as_u32(av2_0[1] - __uint_as_float(ah_3[1]));
                        ah_3[2] = __as_u32(av2_0[2]) + 4096 & 4294959104u;
                        al_3[2] = __as_u32(av2_0[2] - __uint_as_float(ah_3[2]));
                        ah_3[3] = __as_u32(av2_0[3]) + 4096 & 4294959104u;
                        al_3[3] = __as_u32(av2_0[3] - __uint_as_float(ah_3[3]));
                        bh_3[0] = __as_u32(bw1_0[0]) + 4096 & 4294959104u;
                        bl_3[0] = __as_u32(bw1_0[0] - __uint_as_float(bh_3[0]));
                        bh_3[1] = __as_u32(bw1_0[1]) + 4096 & 4294959104u;
                        bl_3[1] = __as_u32(bw1_0[1] - __uint_as_float(bh_3[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(xc_s[0]), "=f"(xc_s[1]), "=f"(xc_s[2]), "=f"(xc_s[3])
                            : "r"(al_3[0]), "r"(al_3[1]), "r"(al_3[2]), "r"(al_3[3]), "r"(bh_3[0]), "r"(bh_3[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(xc_b[0]), "=f"(xc_b[1]), "=f"(xc_b[2]), "=f"(xc_b[3])
                            : "r"(ah_3[0]), "r"(ah_3[1]), "r"(ah_3[2]), "r"(ah_3[3]), "r"(bh_3[0]), "r"(bh_3[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(xc_s[0]), "+f"(xc_s[1]), "+f"(xc_s[2]), "+f"(xc_s[3])
                            : "r"(ah_3[0]), "r"(ah_3[1]), "r"(ah_3[2]), "r"(ah_3[3]), "r"(bl_3[0]), "r"(bl_3[1]));
                        float av2_1[4];
                        av2_1[0] = s_g[gw + ((m1_i + gq) * 68 + (m1_i + 8 + tq))];
                        av2_1[1] = s_g[gw + ((m1_i + gq + 8) * 68 + (m1_i + 8 + tq))];
                        av2_1[2] = s_g[gw + ((m1_i + gq) * 68 + (m1_i + 8 + tq + 4))];
                        av2_1[3] = s_g[gw + ((m1_i + gq + 8) * 68 + (m1_i + 8 + tq + 4))];
                        float bw0_1[2];
                        float bw1_1[2];
                        bw0_1[0] = s_g[gw + ((m1_rt + gq) * 68 + (40 + tq))];
                        bw1_1[0] = s_g[gw + ((m1_rt + 8 + gq) * 68 + (40 + tq))];
                        bw0_1[1] = s_g[gw + ((m1_rt + gq) * 68 + (40 + tq + 4))];
                        bw1_1[1] = s_g[gw + ((m1_rt + 8 + gq) * 68 + (40 + tq + 4))];
                        unsigned int ah_4[4];
                        unsigned int al_4[4];
                        unsigned int bh_4[2];
                        unsigned int bl_4[2];
                        ah_4[0] = __as_u32(av2_1[0]) + 4096 & 4294959104u;
                        al_4[0] = __as_u32(av2_1[0] - __uint_as_float(ah_4[0]));
                        ah_4[1] = __as_u32(av2_1[1]) + 4096 & 4294959104u;
                        al_4[1] = __as_u32(av2_1[1] - __uint_as_float(ah_4[1]));
                        ah_4[2] = __as_u32(av2_1[2]) + 4096 & 4294959104u;
                        al_4[2] = __as_u32(av2_1[2] - __uint_as_float(ah_4[2]));
                        ah_4[3] = __as_u32(av2_1[3]) + 4096 & 4294959104u;
                        al_4[3] = __as_u32(av2_1[3] - __uint_as_float(ah_4[3]));
                        bh_4[0] = __as_u32(bw0_1[0]) + 4096 & 4294959104u;
                        bl_4[0] = __as_u32(bw0_1[0] - __uint_as_float(bh_4[0]));
                        bh_4[1] = __as_u32(bw0_1[1]) + 4096 & 4294959104u;
                        bl_4[1] = __as_u32(bw0_1[1] - __uint_as_float(bh_4[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(xa_s[0]), "+f"(xa_s[1]), "+f"(xa_s[2]), "+f"(xa_s[3])
                            : "r"(al_4[0]), "r"(al_4[1]), "r"(al_4[2]), "r"(al_4[3]), "r"(bh_4[0]), "r"(bh_4[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(xa_b[0]), "+f"(xa_b[1]), "+f"(xa_b[2]), "+f"(xa_b[3])
                            : "r"(ah_4[0]), "r"(ah_4[1]), "r"(ah_4[2]), "r"(ah_4[3]), "r"(bh_4[0]), "r"(bh_4[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(xa_s[0]), "+f"(xa_s[1]), "+f"(xa_s[2]), "+f"(xa_s[3])
                            : "r"(ah_4[0]), "r"(ah_4[1]), "r"(ah_4[2]), "r"(ah_4[3]), "r"(bl_4[0]), "r"(bl_4[1]));
                        unsigned int ah_5[4];
                        unsigned int al_5[4];
                        unsigned int bh_5[2];
                        unsigned int bl_5[2];
                        ah_5[0] = __as_u32(av2_1[0]) + 4096 & 4294959104u;
                        al_5[0] = __as_u32(av2_1[0] - __uint_as_float(ah_5[0]));
                        ah_5[1] = __as_u32(av2_1[1]) + 4096 & 4294959104u;
                        al_5[1] = __as_u32(av2_1[1] - __uint_as_float(ah_5[1]));
                        ah_5[2] = __as_u32(av2_1[2]) + 4096 & 4294959104u;
                        al_5[2] = __as_u32(av2_1[2] - __uint_as_float(ah_5[2]));
                        ah_5[3] = __as_u32(av2_1[3]) + 4096 & 4294959104u;
                        al_5[3] = __as_u32(av2_1[3] - __uint_as_float(ah_5[3]));
                        bh_5[0] = __as_u32(bw1_1[0]) + 4096 & 4294959104u;
                        bl_5[0] = __as_u32(bw1_1[0] - __uint_as_float(bh_5[0]));
                        bh_5[1] = __as_u32(bw1_1[1]) + 4096 & 4294959104u;
                        bl_5[1] = __as_u32(bw1_1[1] - __uint_as_float(bh_5[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(xc_s[0]), "+f"(xc_s[1]), "+f"(xc_s[2]), "+f"(xc_s[3])
                            : "r"(al_5[0]), "r"(al_5[1]), "r"(al_5[2]), "r"(al_5[3]), "r"(bh_5[0]), "r"(bh_5[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(xc_b[0]), "+f"(xc_b[1]), "+f"(xc_b[2]), "+f"(xc_b[3])
                            : "r"(ah_5[0]), "r"(ah_5[1]), "r"(ah_5[2]), "r"(ah_5[3]), "r"(bh_5[0]), "r"(bh_5[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(xc_s[0]), "+f"(xc_s[1]), "+f"(xc_s[2]), "+f"(xc_s[3])
                            : "r"(ah_5[0]), "r"(ah_5[1]), "r"(ah_5[2]), "r"(ah_5[3]), "r"(bl_5[0]), "r"(bl_5[1]));
                        __syncwarp();
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((m1_i + gq) * 68 + (m1_j + 2 * tq)) * 4)), "f"(-(xa_b[0] + xa_s[0])), "f"(-(xa_b[1] + xa_s[1])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((m1_i + gq + 8) * 68 + (m1_j + 2 * tq)) * 4)), "f"(-(xa_b[2] + xa_s[2])), "f"(-(xa_b[3] + xa_s[3])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((m1_i + gq) * 68 + (m1_j + 8 + 2 * tq)) * 4)), "f"(-(xc_b[0] + xc_s[0])), "f"(-(xc_b[1] + xc_s[1])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((m1_i + gq + 8) * 68 + (m1_j + 8 + 2 * tq)) * 4)), "f"(-(xc_b[2] + xc_s[2])), "f"(-(xc_b[3] + xc_s[3])) : "memory");
                    }
                    if (grp == 0) {
                        asm volatile("barrier.sync 1, 64;" ::: "memory");
                    }
                    if (grp == 1) {
                        asm volatile("barrier.sync 2, 64;" ::: "memory");
                    }
                    if (grp == 2) {
                        asm volatile("barrier.sync 3, 64;" ::: "memory");
                    }
                    if (grp == 3) {
                        asm volatile("barrier.sync 5, 64;" ::: "memory");
                    }
                    if (grp == 4) {
                        asm volatile("barrier.sync 6, 64;" ::: "memory");
                    }
                    if (grp == 5) {
                        asm volatile("barrier.sync 7, 64;" ::: "memory");
                    }
                    if (hgrp < gpk) {
                        float p1_0b[4];
                        float p1_0s[4];
                        float p1_1b[4];
                        float p1_1s[4];
                        float p1_2b[4];
                        float p1_2s[4];
                        float p1_3b[4];
                        float p1_3s[4];
                        float av_2[4];
                        av_2[0] = s_g[gw + ((32 + 16 * pw + gq) * 68 + tq)];
                        av_2[1] = s_g[gw + ((32 + 16 * pw + gq + 8) * 68 + tq)];
                        av_2[2] = s_g[gw + ((32 + 16 * pw + gq) * 68 + (tq + 4))];
                        av_2[3] = s_g[gw + ((32 + 16 * pw + gq + 8) * 68 + (tq + 4))];
                        float bv_0[2];
                        bv_0[0] = s_g[gw + (tq * 68 + gq)];
                        bv_0[1] = s_g[gw + ((tq + 4) * 68 + gq)];
                        unsigned int ah_6[4];
                        unsigned int al_6[4];
                        unsigned int bh_6[2];
                        unsigned int bl_6[2];
                        ah_6[0] = __as_u32(av_2[0]) + 4096 & 4294959104u;
                        al_6[0] = __as_u32(av_2[0] - __uint_as_float(ah_6[0]));
                        ah_6[1] = __as_u32(av_2[1]) + 4096 & 4294959104u;
                        al_6[1] = __as_u32(av_2[1] - __uint_as_float(ah_6[1]));
                        ah_6[2] = __as_u32(av_2[2]) + 4096 & 4294959104u;
                        al_6[2] = __as_u32(av_2[2] - __uint_as_float(ah_6[2]));
                        ah_6[3] = __as_u32(av_2[3]) + 4096 & 4294959104u;
                        al_6[3] = __as_u32(av_2[3] - __uint_as_float(ah_6[3]));
                        bh_6[0] = __as_u32(bv_0[0]) + 4096 & 4294959104u;
                        bl_6[0] = __as_u32(bv_0[0] - __uint_as_float(bh_6[0]));
                        bh_6[1] = __as_u32(bv_0[1]) + 4096 & 4294959104u;
                        bl_6[1] = __as_u32(bv_0[1] - __uint_as_float(bh_6[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p1_0s[0]), "=f"(p1_0s[1]), "=f"(p1_0s[2]), "=f"(p1_0s[3])
                            : "r"(al_6[0]), "r"(al_6[1]), "r"(al_6[2]), "r"(al_6[3]), "r"(bh_6[0]), "r"(bh_6[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p1_0b[0]), "=f"(p1_0b[1]), "=f"(p1_0b[2]), "=f"(p1_0b[3])
                            : "r"(ah_6[0]), "r"(ah_6[1]), "r"(ah_6[2]), "r"(ah_6[3]), "r"(bh_6[0]), "r"(bh_6[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0s[0]), "+f"(p1_0s[1]), "+f"(p1_0s[2]), "+f"(p1_0s[3])
                            : "r"(ah_6[0]), "r"(ah_6[1]), "r"(ah_6[2]), "r"(ah_6[3]), "r"(bl_6[0]), "r"(bl_6[1]));
                        float bv_1[2];
                        bv_1[0] = s_g[gw + (tq * 68 + (8 + gq))];
                        bv_1[1] = s_g[gw + ((tq + 4) * 68 + (8 + gq))];
                        unsigned int ah_6_1[4];
                        unsigned int al_6_1[4];
                        unsigned int bh_6_1[2];
                        unsigned int bl_6_1[2];
                        ah_6_1[0] = __as_u32(av_2[0]) + 4096 & 4294959104u;
                        al_6_1[0] = __as_u32(av_2[0] - __uint_as_float(ah_6_1[0]));
                        ah_6_1[1] = __as_u32(av_2[1]) + 4096 & 4294959104u;
                        al_6_1[1] = __as_u32(av_2[1] - __uint_as_float(ah_6_1[1]));
                        ah_6_1[2] = __as_u32(av_2[2]) + 4096 & 4294959104u;
                        al_6_1[2] = __as_u32(av_2[2] - __uint_as_float(ah_6_1[2]));
                        ah_6_1[3] = __as_u32(av_2[3]) + 4096 & 4294959104u;
                        al_6_1[3] = __as_u32(av_2[3] - __uint_as_float(ah_6_1[3]));
                        bh_6_1[0] = __as_u32(bv_1[0]) + 4096 & 4294959104u;
                        bl_6_1[0] = __as_u32(bv_1[0] - __uint_as_float(bh_6_1[0]));
                        bh_6_1[1] = __as_u32(bv_1[1]) + 4096 & 4294959104u;
                        bl_6_1[1] = __as_u32(bv_1[1] - __uint_as_float(bh_6_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p1_1s[0]), "=f"(p1_1s[1]), "=f"(p1_1s[2]), "=f"(p1_1s[3])
                            : "r"(al_6_1[0]), "r"(al_6_1[1]), "r"(al_6_1[2]), "r"(al_6_1[3]), "r"(bh_6_1[0]), "r"(bh_6_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p1_1b[0]), "=f"(p1_1b[1]), "=f"(p1_1b[2]), "=f"(p1_1b[3])
                            : "r"(ah_6_1[0]), "r"(ah_6_1[1]), "r"(ah_6_1[2]), "r"(ah_6_1[3]), "r"(bh_6_1[0]), "r"(bh_6_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1s[0]), "+f"(p1_1s[1]), "+f"(p1_1s[2]), "+f"(p1_1s[3])
                            : "r"(ah_6_1[0]), "r"(ah_6_1[1]), "r"(ah_6_1[2]), "r"(ah_6_1[3]), "r"(bl_6_1[0]), "r"(bl_6_1[1]));
                        float av_3[4];
                        av_3[0] = s_g[gw + ((32 + 16 * pw + gq) * 68 + (8 + tq))];
                        av_3[1] = s_g[gw + ((32 + 16 * pw + gq + 8) * 68 + (8 + tq))];
                        av_3[2] = s_g[gw + ((32 + 16 * pw + gq) * 68 + (8 + tq + 4))];
                        av_3[3] = s_g[gw + ((32 + 16 * pw + gq + 8) * 68 + (8 + tq + 4))];
                        float bv_2[2];
                        bv_2[0] = s_g[gw + ((8 + tq) * 68 + gq)];
                        bv_2[1] = s_g[gw + ((8 + tq + 4) * 68 + gq)];
                        unsigned int ah_7[4];
                        unsigned int al_7[4];
                        unsigned int bh_7[2];
                        unsigned int bl_7[2];
                        ah_7[0] = __as_u32(av_3[0]) + 4096 & 4294959104u;
                        al_7[0] = __as_u32(av_3[0] - __uint_as_float(ah_7[0]));
                        ah_7[1] = __as_u32(av_3[1]) + 4096 & 4294959104u;
                        al_7[1] = __as_u32(av_3[1] - __uint_as_float(ah_7[1]));
                        ah_7[2] = __as_u32(av_3[2]) + 4096 & 4294959104u;
                        al_7[2] = __as_u32(av_3[2] - __uint_as_float(ah_7[2]));
                        ah_7[3] = __as_u32(av_3[3]) + 4096 & 4294959104u;
                        al_7[3] = __as_u32(av_3[3] - __uint_as_float(ah_7[3]));
                        bh_7[0] = __as_u32(bv_2[0]) + 4096 & 4294959104u;
                        bl_7[0] = __as_u32(bv_2[0] - __uint_as_float(bh_7[0]));
                        bh_7[1] = __as_u32(bv_2[1]) + 4096 & 4294959104u;
                        bl_7[1] = __as_u32(bv_2[1] - __uint_as_float(bh_7[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0s[0]), "+f"(p1_0s[1]), "+f"(p1_0s[2]), "+f"(p1_0s[3])
                            : "r"(al_7[0]), "r"(al_7[1]), "r"(al_7[2]), "r"(al_7[3]), "r"(bh_7[0]), "r"(bh_7[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0b[0]), "+f"(p1_0b[1]), "+f"(p1_0b[2]), "+f"(p1_0b[3])
                            : "r"(ah_7[0]), "r"(ah_7[1]), "r"(ah_7[2]), "r"(ah_7[3]), "r"(bh_7[0]), "r"(bh_7[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0s[0]), "+f"(p1_0s[1]), "+f"(p1_0s[2]), "+f"(p1_0s[3])
                            : "r"(ah_7[0]), "r"(ah_7[1]), "r"(ah_7[2]), "r"(ah_7[3]), "r"(bl_7[0]), "r"(bl_7[1]));
                        float bv_3[2];
                        bv_3[0] = s_g[gw + ((8 + tq) * 68 + (8 + gq))];
                        bv_3[1] = s_g[gw + ((8 + tq + 4) * 68 + (8 + gq))];
                        unsigned int ah_8[4];
                        unsigned int al_8[4];
                        unsigned int bh_8[2];
                        unsigned int bl_8[2];
                        ah_8[0] = __as_u32(av_3[0]) + 4096 & 4294959104u;
                        al_8[0] = __as_u32(av_3[0] - __uint_as_float(ah_8[0]));
                        ah_8[1] = __as_u32(av_3[1]) + 4096 & 4294959104u;
                        al_8[1] = __as_u32(av_3[1] - __uint_as_float(ah_8[1]));
                        ah_8[2] = __as_u32(av_3[2]) + 4096 & 4294959104u;
                        al_8[2] = __as_u32(av_3[2] - __uint_as_float(ah_8[2]));
                        ah_8[3] = __as_u32(av_3[3]) + 4096 & 4294959104u;
                        al_8[3] = __as_u32(av_3[3] - __uint_as_float(ah_8[3]));
                        bh_8[0] = __as_u32(bv_3[0]) + 4096 & 4294959104u;
                        bl_8[0] = __as_u32(bv_3[0] - __uint_as_float(bh_8[0]));
                        bh_8[1] = __as_u32(bv_3[1]) + 4096 & 4294959104u;
                        bl_8[1] = __as_u32(bv_3[1] - __uint_as_float(bh_8[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1s[0]), "+f"(p1_1s[1]), "+f"(p1_1s[2]), "+f"(p1_1s[3])
                            : "r"(al_8[0]), "r"(al_8[1]), "r"(al_8[2]), "r"(al_8[3]), "r"(bh_8[0]), "r"(bh_8[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1b[0]), "+f"(p1_1b[1]), "+f"(p1_1b[2]), "+f"(p1_1b[3])
                            : "r"(ah_8[0]), "r"(ah_8[1]), "r"(ah_8[2]), "r"(ah_8[3]), "r"(bh_8[0]), "r"(bh_8[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1s[0]), "+f"(p1_1s[1]), "+f"(p1_1s[2]), "+f"(p1_1s[3])
                            : "r"(ah_8[0]), "r"(ah_8[1]), "r"(ah_8[2]), "r"(ah_8[3]), "r"(bl_8[0]), "r"(bl_8[1]));
                        float av_4[4];
                        av_4[0] = s_g[gw + ((32 + 16 * pw + gq) * 68 + (16 + tq))];
                        av_4[1] = s_g[gw + ((32 + 16 * pw + gq + 8) * 68 + (16 + tq))];
                        av_4[2] = s_g[gw + ((32 + 16 * pw + gq) * 68 + (16 + tq + 4))];
                        av_4[3] = s_g[gw + ((32 + 16 * pw + gq + 8) * 68 + (16 + tq + 4))];
                        float bv_4[2];
                        bv_4[0] = s_g[gw + ((16 + tq) * 68 + gq)];
                        bv_4[1] = s_g[gw + ((16 + tq + 4) * 68 + gq)];
                        unsigned int ah_9[4];
                        unsigned int al_9[4];
                        unsigned int bh_9[2];
                        unsigned int bl_9[2];
                        ah_9[0] = __as_u32(av_4[0]) + 4096 & 4294959104u;
                        al_9[0] = __as_u32(av_4[0] - __uint_as_float(ah_9[0]));
                        ah_9[1] = __as_u32(av_4[1]) + 4096 & 4294959104u;
                        al_9[1] = __as_u32(av_4[1] - __uint_as_float(ah_9[1]));
                        ah_9[2] = __as_u32(av_4[2]) + 4096 & 4294959104u;
                        al_9[2] = __as_u32(av_4[2] - __uint_as_float(ah_9[2]));
                        ah_9[3] = __as_u32(av_4[3]) + 4096 & 4294959104u;
                        al_9[3] = __as_u32(av_4[3] - __uint_as_float(ah_9[3]));
                        bh_9[0] = __as_u32(bv_4[0]) + 4096 & 4294959104u;
                        bl_9[0] = __as_u32(bv_4[0] - __uint_as_float(bh_9[0]));
                        bh_9[1] = __as_u32(bv_4[1]) + 4096 & 4294959104u;
                        bl_9[1] = __as_u32(bv_4[1] - __uint_as_float(bh_9[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0s[0]), "+f"(p1_0s[1]), "+f"(p1_0s[2]), "+f"(p1_0s[3])
                            : "r"(al_9[0]), "r"(al_9[1]), "r"(al_9[2]), "r"(al_9[3]), "r"(bh_9[0]), "r"(bh_9[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0b[0]), "+f"(p1_0b[1]), "+f"(p1_0b[2]), "+f"(p1_0b[3])
                            : "r"(ah_9[0]), "r"(ah_9[1]), "r"(ah_9[2]), "r"(ah_9[3]), "r"(bh_9[0]), "r"(bh_9[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0s[0]), "+f"(p1_0s[1]), "+f"(p1_0s[2]), "+f"(p1_0s[3])
                            : "r"(ah_9[0]), "r"(ah_9[1]), "r"(ah_9[2]), "r"(ah_9[3]), "r"(bl_9[0]), "r"(bl_9[1]));
                        float bv_5[2];
                        bv_5[0] = s_g[gw + ((16 + tq) * 68 + (8 + gq))];
                        bv_5[1] = s_g[gw + ((16 + tq + 4) * 68 + (8 + gq))];
                        unsigned int ah_10[4];
                        unsigned int al_10[4];
                        unsigned int bh_10[2];
                        unsigned int bl_10[2];
                        ah_10[0] = __as_u32(av_4[0]) + 4096 & 4294959104u;
                        al_10[0] = __as_u32(av_4[0] - __uint_as_float(ah_10[0]));
                        ah_10[1] = __as_u32(av_4[1]) + 4096 & 4294959104u;
                        al_10[1] = __as_u32(av_4[1] - __uint_as_float(ah_10[1]));
                        ah_10[2] = __as_u32(av_4[2]) + 4096 & 4294959104u;
                        al_10[2] = __as_u32(av_4[2] - __uint_as_float(ah_10[2]));
                        ah_10[3] = __as_u32(av_4[3]) + 4096 & 4294959104u;
                        al_10[3] = __as_u32(av_4[3] - __uint_as_float(ah_10[3]));
                        bh_10[0] = __as_u32(bv_5[0]) + 4096 & 4294959104u;
                        bl_10[0] = __as_u32(bv_5[0] - __uint_as_float(bh_10[0]));
                        bh_10[1] = __as_u32(bv_5[1]) + 4096 & 4294959104u;
                        bl_10[1] = __as_u32(bv_5[1] - __uint_as_float(bh_10[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1s[0]), "+f"(p1_1s[1]), "+f"(p1_1s[2]), "+f"(p1_1s[3])
                            : "r"(al_10[0]), "r"(al_10[1]), "r"(al_10[2]), "r"(al_10[3]), "r"(bh_10[0]), "r"(bh_10[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1b[0]), "+f"(p1_1b[1]), "+f"(p1_1b[2]), "+f"(p1_1b[3])
                            : "r"(ah_10[0]), "r"(ah_10[1]), "r"(ah_10[2]), "r"(ah_10[3]), "r"(bh_10[0]), "r"(bh_10[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1s[0]), "+f"(p1_1s[1]), "+f"(p1_1s[2]), "+f"(p1_1s[3])
                            : "r"(ah_10[0]), "r"(ah_10[1]), "r"(ah_10[2]), "r"(ah_10[3]), "r"(bl_10[0]), "r"(bl_10[1]));
                        float bv_6[2];
                        bv_6[0] = s_g[gw + ((16 + tq) * 68 + (16 + gq))];
                        bv_6[1] = s_g[gw + ((16 + tq + 4) * 68 + (16 + gq))];
                        unsigned int ah_11[4];
                        unsigned int al_11[4];
                        unsigned int bh_11[2];
                        unsigned int bl_11[2];
                        ah_11[0] = __as_u32(av_4[0]) + 4096 & 4294959104u;
                        al_11[0] = __as_u32(av_4[0] - __uint_as_float(ah_11[0]));
                        ah_11[1] = __as_u32(av_4[1]) + 4096 & 4294959104u;
                        al_11[1] = __as_u32(av_4[1] - __uint_as_float(ah_11[1]));
                        ah_11[2] = __as_u32(av_4[2]) + 4096 & 4294959104u;
                        al_11[2] = __as_u32(av_4[2] - __uint_as_float(ah_11[2]));
                        ah_11[3] = __as_u32(av_4[3]) + 4096 & 4294959104u;
                        al_11[3] = __as_u32(av_4[3] - __uint_as_float(ah_11[3]));
                        bh_11[0] = __as_u32(bv_6[0]) + 4096 & 4294959104u;
                        bl_11[0] = __as_u32(bv_6[0] - __uint_as_float(bh_11[0]));
                        bh_11[1] = __as_u32(bv_6[1]) + 4096 & 4294959104u;
                        bl_11[1] = __as_u32(bv_6[1] - __uint_as_float(bh_11[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p1_2s[0]), "=f"(p1_2s[1]), "=f"(p1_2s[2]), "=f"(p1_2s[3])
                            : "r"(al_11[0]), "r"(al_11[1]), "r"(al_11[2]), "r"(al_11[3]), "r"(bh_11[0]), "r"(bh_11[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p1_2b[0]), "=f"(p1_2b[1]), "=f"(p1_2b[2]), "=f"(p1_2b[3])
                            : "r"(ah_11[0]), "r"(ah_11[1]), "r"(ah_11[2]), "r"(ah_11[3]), "r"(bh_11[0]), "r"(bh_11[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_2s[0]), "+f"(p1_2s[1]), "+f"(p1_2s[2]), "+f"(p1_2s[3])
                            : "r"(ah_11[0]), "r"(ah_11[1]), "r"(ah_11[2]), "r"(ah_11[3]), "r"(bl_11[0]), "r"(bl_11[1]));
                        float bv_7[2];
                        bv_7[0] = s_g[gw + ((16 + tq) * 68 + (24 + gq))];
                        bv_7[1] = s_g[gw + ((16 + tq + 4) * 68 + (24 + gq))];
                        unsigned int ah_12[4];
                        unsigned int al_12[4];
                        unsigned int bh_12[2];
                        unsigned int bl_12[2];
                        ah_12[0] = __as_u32(av_4[0]) + 4096 & 4294959104u;
                        al_12[0] = __as_u32(av_4[0] - __uint_as_float(ah_12[0]));
                        ah_12[1] = __as_u32(av_4[1]) + 4096 & 4294959104u;
                        al_12[1] = __as_u32(av_4[1] - __uint_as_float(ah_12[1]));
                        ah_12[2] = __as_u32(av_4[2]) + 4096 & 4294959104u;
                        al_12[2] = __as_u32(av_4[2] - __uint_as_float(ah_12[2]));
                        ah_12[3] = __as_u32(av_4[3]) + 4096 & 4294959104u;
                        al_12[3] = __as_u32(av_4[3] - __uint_as_float(ah_12[3]));
                        bh_12[0] = __as_u32(bv_7[0]) + 4096 & 4294959104u;
                        bl_12[0] = __as_u32(bv_7[0] - __uint_as_float(bh_12[0]));
                        bh_12[1] = __as_u32(bv_7[1]) + 4096 & 4294959104u;
                        bl_12[1] = __as_u32(bv_7[1] - __uint_as_float(bh_12[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p1_3s[0]), "=f"(p1_3s[1]), "=f"(p1_3s[2]), "=f"(p1_3s[3])
                            : "r"(al_12[0]), "r"(al_12[1]), "r"(al_12[2]), "r"(al_12[3]), "r"(bh_12[0]), "r"(bh_12[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p1_3b[0]), "=f"(p1_3b[1]), "=f"(p1_3b[2]), "=f"(p1_3b[3])
                            : "r"(ah_12[0]), "r"(ah_12[1]), "r"(ah_12[2]), "r"(ah_12[3]), "r"(bh_12[0]), "r"(bh_12[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_3s[0]), "+f"(p1_3s[1]), "+f"(p1_3s[2]), "+f"(p1_3s[3])
                            : "r"(ah_12[0]), "r"(ah_12[1]), "r"(ah_12[2]), "r"(ah_12[3]), "r"(bl_12[0]), "r"(bl_12[1]));
                        float av_5[4];
                        av_5[0] = s_g[gw + ((32 + 16 * pw + gq) * 68 + (24 + tq))];
                        av_5[1] = s_g[gw + ((32 + 16 * pw + gq + 8) * 68 + (24 + tq))];
                        av_5[2] = s_g[gw + ((32 + 16 * pw + gq) * 68 + (24 + tq + 4))];
                        av_5[3] = s_g[gw + ((32 + 16 * pw + gq + 8) * 68 + (24 + tq + 4))];
                        float bv_8[2];
                        bv_8[0] = s_g[gw + ((24 + tq) * 68 + gq)];
                        bv_8[1] = s_g[gw + ((24 + tq + 4) * 68 + gq)];
                        unsigned int ah_13[4];
                        unsigned int al_13[4];
                        unsigned int bh_13[2];
                        unsigned int bl_13[2];
                        ah_13[0] = __as_u32(av_5[0]) + 4096 & 4294959104u;
                        al_13[0] = __as_u32(av_5[0] - __uint_as_float(ah_13[0]));
                        ah_13[1] = __as_u32(av_5[1]) + 4096 & 4294959104u;
                        al_13[1] = __as_u32(av_5[1] - __uint_as_float(ah_13[1]));
                        ah_13[2] = __as_u32(av_5[2]) + 4096 & 4294959104u;
                        al_13[2] = __as_u32(av_5[2] - __uint_as_float(ah_13[2]));
                        ah_13[3] = __as_u32(av_5[3]) + 4096 & 4294959104u;
                        al_13[3] = __as_u32(av_5[3] - __uint_as_float(ah_13[3]));
                        bh_13[0] = __as_u32(bv_8[0]) + 4096 & 4294959104u;
                        bl_13[0] = __as_u32(bv_8[0] - __uint_as_float(bh_13[0]));
                        bh_13[1] = __as_u32(bv_8[1]) + 4096 & 4294959104u;
                        bl_13[1] = __as_u32(bv_8[1] - __uint_as_float(bh_13[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0s[0]), "+f"(p1_0s[1]), "+f"(p1_0s[2]), "+f"(p1_0s[3])
                            : "r"(al_13[0]), "r"(al_13[1]), "r"(al_13[2]), "r"(al_13[3]), "r"(bh_13[0]), "r"(bh_13[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0b[0]), "+f"(p1_0b[1]), "+f"(p1_0b[2]), "+f"(p1_0b[3])
                            : "r"(ah_13[0]), "r"(ah_13[1]), "r"(ah_13[2]), "r"(ah_13[3]), "r"(bh_13[0]), "r"(bh_13[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_0s[0]), "+f"(p1_0s[1]), "+f"(p1_0s[2]), "+f"(p1_0s[3])
                            : "r"(ah_13[0]), "r"(ah_13[1]), "r"(ah_13[2]), "r"(ah_13[3]), "r"(bl_13[0]), "r"(bl_13[1]));
                        float bv_9[2];
                        bv_9[0] = s_g[gw + ((24 + tq) * 68 + (8 + gq))];
                        bv_9[1] = s_g[gw + ((24 + tq + 4) * 68 + (8 + gq))];
                        unsigned int ah_14[4];
                        unsigned int al_14[4];
                        unsigned int bh_14[2];
                        unsigned int bl_14[2];
                        ah_14[0] = __as_u32(av_5[0]) + 4096 & 4294959104u;
                        al_14[0] = __as_u32(av_5[0] - __uint_as_float(ah_14[0]));
                        ah_14[1] = __as_u32(av_5[1]) + 4096 & 4294959104u;
                        al_14[1] = __as_u32(av_5[1] - __uint_as_float(ah_14[1]));
                        ah_14[2] = __as_u32(av_5[2]) + 4096 & 4294959104u;
                        al_14[2] = __as_u32(av_5[2] - __uint_as_float(ah_14[2]));
                        ah_14[3] = __as_u32(av_5[3]) + 4096 & 4294959104u;
                        al_14[3] = __as_u32(av_5[3] - __uint_as_float(ah_14[3]));
                        bh_14[0] = __as_u32(bv_9[0]) + 4096 & 4294959104u;
                        bl_14[0] = __as_u32(bv_9[0] - __uint_as_float(bh_14[0]));
                        bh_14[1] = __as_u32(bv_9[1]) + 4096 & 4294959104u;
                        bl_14[1] = __as_u32(bv_9[1] - __uint_as_float(bh_14[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1s[0]), "+f"(p1_1s[1]), "+f"(p1_1s[2]), "+f"(p1_1s[3])
                            : "r"(al_14[0]), "r"(al_14[1]), "r"(al_14[2]), "r"(al_14[3]), "r"(bh_14[0]), "r"(bh_14[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1b[0]), "+f"(p1_1b[1]), "+f"(p1_1b[2]), "+f"(p1_1b[3])
                            : "r"(ah_14[0]), "r"(ah_14[1]), "r"(ah_14[2]), "r"(ah_14[3]), "r"(bh_14[0]), "r"(bh_14[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_1s[0]), "+f"(p1_1s[1]), "+f"(p1_1s[2]), "+f"(p1_1s[3])
                            : "r"(ah_14[0]), "r"(ah_14[1]), "r"(ah_14[2]), "r"(ah_14[3]), "r"(bl_14[0]), "r"(bl_14[1]));
                        float bv_10[2];
                        bv_10[0] = s_g[gw + ((24 + tq) * 68 + (16 + gq))];
                        bv_10[1] = s_g[gw + ((24 + tq + 4) * 68 + (16 + gq))];
                        unsigned int ah_15[4];
                        unsigned int al_15[4];
                        unsigned int bh_15[2];
                        unsigned int bl_15[2];
                        ah_15[0] = __as_u32(av_5[0]) + 4096 & 4294959104u;
                        al_15[0] = __as_u32(av_5[0] - __uint_as_float(ah_15[0]));
                        ah_15[1] = __as_u32(av_5[1]) + 4096 & 4294959104u;
                        al_15[1] = __as_u32(av_5[1] - __uint_as_float(ah_15[1]));
                        ah_15[2] = __as_u32(av_5[2]) + 4096 & 4294959104u;
                        al_15[2] = __as_u32(av_5[2] - __uint_as_float(ah_15[2]));
                        ah_15[3] = __as_u32(av_5[3]) + 4096 & 4294959104u;
                        al_15[3] = __as_u32(av_5[3] - __uint_as_float(ah_15[3]));
                        bh_15[0] = __as_u32(bv_10[0]) + 4096 & 4294959104u;
                        bl_15[0] = __as_u32(bv_10[0] - __uint_as_float(bh_15[0]));
                        bh_15[1] = __as_u32(bv_10[1]) + 4096 & 4294959104u;
                        bl_15[1] = __as_u32(bv_10[1] - __uint_as_float(bh_15[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_2s[0]), "+f"(p1_2s[1]), "+f"(p1_2s[2]), "+f"(p1_2s[3])
                            : "r"(al_15[0]), "r"(al_15[1]), "r"(al_15[2]), "r"(al_15[3]), "r"(bh_15[0]), "r"(bh_15[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_2b[0]), "+f"(p1_2b[1]), "+f"(p1_2b[2]), "+f"(p1_2b[3])
                            : "r"(ah_15[0]), "r"(ah_15[1]), "r"(ah_15[2]), "r"(ah_15[3]), "r"(bh_15[0]), "r"(bh_15[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_2s[0]), "+f"(p1_2s[1]), "+f"(p1_2s[2]), "+f"(p1_2s[3])
                            : "r"(ah_15[0]), "r"(ah_15[1]), "r"(ah_15[2]), "r"(ah_15[3]), "r"(bl_15[0]), "r"(bl_15[1]));
                        float bv_11[2];
                        bv_11[0] = s_g[gw + ((24 + tq) * 68 + (24 + gq))];
                        bv_11[1] = s_g[gw + ((24 + tq + 4) * 68 + (24 + gq))];
                        unsigned int ah_16[4];
                        unsigned int al_16[4];
                        unsigned int bh_16[2];
                        unsigned int bl_16[2];
                        ah_16[0] = __as_u32(av_5[0]) + 4096 & 4294959104u;
                        al_16[0] = __as_u32(av_5[0] - __uint_as_float(ah_16[0]));
                        ah_16[1] = __as_u32(av_5[1]) + 4096 & 4294959104u;
                        al_16[1] = __as_u32(av_5[1] - __uint_as_float(ah_16[1]));
                        ah_16[2] = __as_u32(av_5[2]) + 4096 & 4294959104u;
                        al_16[2] = __as_u32(av_5[2] - __uint_as_float(ah_16[2]));
                        ah_16[3] = __as_u32(av_5[3]) + 4096 & 4294959104u;
                        al_16[3] = __as_u32(av_5[3] - __uint_as_float(ah_16[3]));
                        bh_16[0] = __as_u32(bv_11[0]) + 4096 & 4294959104u;
                        bl_16[0] = __as_u32(bv_11[0] - __uint_as_float(bh_16[0]));
                        bh_16[1] = __as_u32(bv_11[1]) + 4096 & 4294959104u;
                        bl_16[1] = __as_u32(bv_11[1] - __uint_as_float(bh_16[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_3s[0]), "+f"(p1_3s[1]), "+f"(p1_3s[2]), "+f"(p1_3s[3])
                            : "r"(al_16[0]), "r"(al_16[1]), "r"(al_16[2]), "r"(al_16[3]), "r"(bh_16[0]), "r"(bh_16[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_3b[0]), "+f"(p1_3b[1]), "+f"(p1_3b[2]), "+f"(p1_3b[3])
                            : "r"(ah_16[0]), "r"(ah_16[1]), "r"(ah_16[2]), "r"(ah_16[3]), "r"(bh_16[0]), "r"(bh_16[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p1_3s[0]), "+f"(p1_3s[1]), "+f"(p1_3s[2]), "+f"(p1_3s[3])
                            : "r"(ah_16[0]), "r"(ah_16[1]), "r"(ah_16[2]), "r"(ah_16[3]), "r"(bl_16[0]), "r"(bl_16[1]));
                        {
                            uint32_t _addr_100 = static_cast<uint32_t>(lb + (unsigned int)((2 * tq * 68 + (32 + 16 * pw + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_100), "f"(p1_0b[0] + p1_0s[0]) : "memory");
                        }
                        {
                            uint32_t _addr_101 = static_cast<uint32_t>(lb + (unsigned int)(((2 * tq + 1) * 68 + (32 + 16 * pw + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_101), "f"(p1_0b[1] + p1_0s[1]) : "memory");
                        }
                        {
                            uint32_t _addr_102 = static_cast<uint32_t>(lb + (unsigned int)((2 * tq * 68 + (32 + 16 * pw + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_102), "f"(p1_0b[2] + p1_0s[2]) : "memory");
                        }
                        {
                            uint32_t _addr_103 = static_cast<uint32_t>(lb + (unsigned int)(((2 * tq + 1) * 68 + (32 + 16 * pw + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_103), "f"(p1_0b[3] + p1_0s[3]) : "memory");
                        }
                        {
                            uint32_t _addr_104 = static_cast<uint32_t>(lb + (unsigned int)(((8 + 2 * tq) * 68 + (32 + 16 * pw + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_104), "f"(p1_1b[0] + p1_1s[0]) : "memory");
                        }
                        {
                            uint32_t _addr_105 = static_cast<uint32_t>(lb + (unsigned int)(((8 + 2 * tq + 1) * 68 + (32 + 16 * pw + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_105), "f"(p1_1b[1] + p1_1s[1]) : "memory");
                        }
                        {
                            uint32_t _addr_106 = static_cast<uint32_t>(lb + (unsigned int)(((8 + 2 * tq) * 68 + (32 + 16 * pw + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_106), "f"(p1_1b[2] + p1_1s[2]) : "memory");
                        }
                        {
                            uint32_t _addr_107 = static_cast<uint32_t>(lb + (unsigned int)(((8 + 2 * tq + 1) * 68 + (32 + 16 * pw + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_107), "f"(p1_1b[3] + p1_1s[3]) : "memory");
                        }
                        {
                            uint32_t _addr_108 = static_cast<uint32_t>(lb + (unsigned int)(((16 + 2 * tq) * 68 + (32 + 16 * pw + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_108), "f"(p1_2b[0] + p1_2s[0]) : "memory");
                        }
                        {
                            uint32_t _addr_109 = static_cast<uint32_t>(lb + (unsigned int)(((16 + 2 * tq + 1) * 68 + (32 + 16 * pw + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_109), "f"(p1_2b[1] + p1_2s[1]) : "memory");
                        }
                        {
                            uint32_t _addr_110 = static_cast<uint32_t>(lb + (unsigned int)(((16 + 2 * tq) * 68 + (32 + 16 * pw + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_110), "f"(p1_2b[2] + p1_2s[2]) : "memory");
                        }
                        {
                            uint32_t _addr_111 = static_cast<uint32_t>(lb + (unsigned int)(((16 + 2 * tq + 1) * 68 + (32 + 16 * pw + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_111), "f"(p1_2b[3] + p1_2s[3]) : "memory");
                        }
                        {
                            uint32_t _addr_112 = static_cast<uint32_t>(lb + (unsigned int)(((24 + 2 * tq) * 68 + (32 + 16 * pw + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_112), "f"(p1_3b[0] + p1_3s[0]) : "memory");
                        }
                        {
                            uint32_t _addr_113 = static_cast<uint32_t>(lb + (unsigned int)(((24 + 2 * tq + 1) * 68 + (32 + 16 * pw + gq)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_113), "f"(p1_3b[1] + p1_3s[1]) : "memory");
                        }
                        {
                            uint32_t _addr_114 = static_cast<uint32_t>(lb + (unsigned int)(((24 + 2 * tq) * 68 + (32 + 16 * pw + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_114), "f"(p1_3b[2] + p1_3s[2]) : "memory");
                        }
                        {
                            uint32_t _addr_115 = static_cast<uint32_t>(lb + (unsigned int)(((24 + 2 * tq + 1) * 68 + (32 + 16 * pw + gq + 8)) * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_115), "f"(p1_3b[3] + p1_3s[3]) : "memory");
                        }
                    }
                    if (grp == 0) {
                        asm volatile("barrier.sync 1, 64;" ::: "memory");
                    }
                    if (grp == 1) {
                        asm volatile("barrier.sync 2, 64;" ::: "memory");
                    }
                    if (grp == 2) {
                        asm volatile("barrier.sync 3, 64;" ::: "memory");
                    }
                    if (grp == 3) {
                        asm volatile("barrier.sync 5, 64;" ::: "memory");
                    }
                    if (grp == 4) {
                        asm volatile("barrier.sync 6, 64;" ::: "memory");
                    }
                    if (grp == 5) {
                        asm volatile("barrier.sync 7, 64;" ::: "memory");
                    }
                    if (hgrp < gpk) {
                        float p2_00b[4];
                        float p2_00s[4];
                        float p2_01b[4];
                        float p2_01s[4];
                        float p2_10b[4];
                        float p2_10s[4];
                        float p2_11b[4];
                        float p2_11s[4];
                        float bv2_0[2];
                        bv2_0[0] = s_g[gw + ((16 * pw + gq) * 68 + (32 + tq))];
                        bv2_0[1] = s_g[gw + ((16 * pw + gq) * 68 + (32 + tq + 4))];
                        float av3_0[4];
                        av3_0[0] = s_g[gw + ((32 + gq) * 68 + (32 + tq))];
                        av3_0[1] = s_g[gw + ((32 + gq + 8) * 68 + (32 + tq))];
                        av3_0[2] = s_g[gw + ((32 + gq) * 68 + (32 + tq + 4))];
                        av3_0[3] = s_g[gw + ((32 + gq + 8) * 68 + (32 + tq + 4))];
                        unsigned int ah_17[4];
                        unsigned int al_17[4];
                        unsigned int bh_17[2];
                        unsigned int bl_17[2];
                        ah_17[0] = __as_u32(av3_0[0]) + 4096 & 4294959104u;
                        al_17[0] = __as_u32(av3_0[0] - __uint_as_float(ah_17[0]));
                        ah_17[1] = __as_u32(av3_0[1]) + 4096 & 4294959104u;
                        al_17[1] = __as_u32(av3_0[1] - __uint_as_float(ah_17[1]));
                        ah_17[2] = __as_u32(av3_0[2]) + 4096 & 4294959104u;
                        al_17[2] = __as_u32(av3_0[2] - __uint_as_float(ah_17[2]));
                        ah_17[3] = __as_u32(av3_0[3]) + 4096 & 4294959104u;
                        al_17[3] = __as_u32(av3_0[3] - __uint_as_float(ah_17[3]));
                        bh_17[0] = __as_u32(bv2_0[0]) + 4096 & 4294959104u;
                        bl_17[0] = __as_u32(bv2_0[0] - __uint_as_float(bh_17[0]));
                        bh_17[1] = __as_u32(bv2_0[1]) + 4096 & 4294959104u;
                        bl_17[1] = __as_u32(bv2_0[1] - __uint_as_float(bh_17[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p2_00s[0]), "=f"(p2_00s[1]), "=f"(p2_00s[2]), "=f"(p2_00s[3])
                            : "r"(al_17[0]), "r"(al_17[1]), "r"(al_17[2]), "r"(al_17[3]), "r"(bh_17[0]), "r"(bh_17[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p2_00b[0]), "=f"(p2_00b[1]), "=f"(p2_00b[2]), "=f"(p2_00b[3])
                            : "r"(ah_17[0]), "r"(ah_17[1]), "r"(ah_17[2]), "r"(ah_17[3]), "r"(bh_17[0]), "r"(bh_17[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_00s[0]), "+f"(p2_00s[1]), "+f"(p2_00s[2]), "+f"(p2_00s[3])
                            : "r"(ah_17[0]), "r"(ah_17[1]), "r"(ah_17[2]), "r"(ah_17[3]), "r"(bl_17[0]), "r"(bl_17[1]));
                        float av3_1[4];
                        av3_1[0] = s_g[gw + ((48 + gq) * 68 + (32 + tq))];
                        av3_1[1] = s_g[gw + ((48 + gq + 8) * 68 + (32 + tq))];
                        av3_1[2] = s_g[gw + ((48 + gq) * 68 + (32 + tq + 4))];
                        av3_1[3] = s_g[gw + ((48 + gq + 8) * 68 + (32 + tq + 4))];
                        unsigned int ah_17_1[4];
                        unsigned int al_17_1[4];
                        unsigned int bh_17_1[2];
                        unsigned int bl_17_1[2];
                        ah_17_1[0] = __as_u32(av3_1[0]) + 4096 & 4294959104u;
                        al_17_1[0] = __as_u32(av3_1[0] - __uint_as_float(ah_17_1[0]));
                        ah_17_1[1] = __as_u32(av3_1[1]) + 4096 & 4294959104u;
                        al_17_1[1] = __as_u32(av3_1[1] - __uint_as_float(ah_17_1[1]));
                        ah_17_1[2] = __as_u32(av3_1[2]) + 4096 & 4294959104u;
                        al_17_1[2] = __as_u32(av3_1[2] - __uint_as_float(ah_17_1[2]));
                        ah_17_1[3] = __as_u32(av3_1[3]) + 4096 & 4294959104u;
                        al_17_1[3] = __as_u32(av3_1[3] - __uint_as_float(ah_17_1[3]));
                        bh_17_1[0] = __as_u32(bv2_0[0]) + 4096 & 4294959104u;
                        bl_17_1[0] = __as_u32(bv2_0[0] - __uint_as_float(bh_17_1[0]));
                        bh_17_1[1] = __as_u32(bv2_0[1]) + 4096 & 4294959104u;
                        bl_17_1[1] = __as_u32(bv2_0[1] - __uint_as_float(bh_17_1[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p2_01s[0]), "=f"(p2_01s[1]), "=f"(p2_01s[2]), "=f"(p2_01s[3])
                            : "r"(al_17_1[0]), "r"(al_17_1[1]), "r"(al_17_1[2]), "r"(al_17_1[3]), "r"(bh_17_1[0]), "r"(bh_17_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p2_01b[0]), "=f"(p2_01b[1]), "=f"(p2_01b[2]), "=f"(p2_01b[3])
                            : "r"(ah_17_1[0]), "r"(ah_17_1[1]), "r"(ah_17_1[2]), "r"(ah_17_1[3]), "r"(bh_17_1[0]), "r"(bh_17_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01s[0]), "+f"(p2_01s[1]), "+f"(p2_01s[2]), "+f"(p2_01s[3])
                            : "r"(ah_17_1[0]), "r"(ah_17_1[1]), "r"(ah_17_1[2]), "r"(ah_17_1[3]), "r"(bl_17_1[0]), "r"(bl_17_1[1]));
                        float bv2_1[2];
                        bv2_1[0] = s_g[gw + ((16 * pw + 8 + gq) * 68 + (32 + tq))];
                        bv2_1[1] = s_g[gw + ((16 * pw + 8 + gq) * 68 + (32 + tq + 4))];
                        float av3_2[4];
                        av3_2[0] = s_g[gw + ((32 + gq) * 68 + (32 + tq))];
                        av3_2[1] = s_g[gw + ((32 + gq + 8) * 68 + (32 + tq))];
                        av3_2[2] = s_g[gw + ((32 + gq) * 68 + (32 + tq + 4))];
                        av3_2[3] = s_g[gw + ((32 + gq + 8) * 68 + (32 + tq + 4))];
                        unsigned int ah_18[4];
                        unsigned int al_18[4];
                        unsigned int bh_18[2];
                        unsigned int bl_18[2];
                        ah_18[0] = __as_u32(av3_2[0]) + 4096 & 4294959104u;
                        al_18[0] = __as_u32(av3_2[0] - __uint_as_float(ah_18[0]));
                        ah_18[1] = __as_u32(av3_2[1]) + 4096 & 4294959104u;
                        al_18[1] = __as_u32(av3_2[1] - __uint_as_float(ah_18[1]));
                        ah_18[2] = __as_u32(av3_2[2]) + 4096 & 4294959104u;
                        al_18[2] = __as_u32(av3_2[2] - __uint_as_float(ah_18[2]));
                        ah_18[3] = __as_u32(av3_2[3]) + 4096 & 4294959104u;
                        al_18[3] = __as_u32(av3_2[3] - __uint_as_float(ah_18[3]));
                        bh_18[0] = __as_u32(bv2_1[0]) + 4096 & 4294959104u;
                        bl_18[0] = __as_u32(bv2_1[0] - __uint_as_float(bh_18[0]));
                        bh_18[1] = __as_u32(bv2_1[1]) + 4096 & 4294959104u;
                        bl_18[1] = __as_u32(bv2_1[1] - __uint_as_float(bh_18[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p2_10s[0]), "=f"(p2_10s[1]), "=f"(p2_10s[2]), "=f"(p2_10s[3])
                            : "r"(al_18[0]), "r"(al_18[1]), "r"(al_18[2]), "r"(al_18[3]), "r"(bh_18[0]), "r"(bh_18[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p2_10b[0]), "=f"(p2_10b[1]), "=f"(p2_10b[2]), "=f"(p2_10b[3])
                            : "r"(ah_18[0]), "r"(ah_18[1]), "r"(ah_18[2]), "r"(ah_18[3]), "r"(bh_18[0]), "r"(bh_18[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_10s[0]), "+f"(p2_10s[1]), "+f"(p2_10s[2]), "+f"(p2_10s[3])
                            : "r"(ah_18[0]), "r"(ah_18[1]), "r"(ah_18[2]), "r"(ah_18[3]), "r"(bl_18[0]), "r"(bl_18[1]));
                        float av3_3[4];
                        av3_3[0] = s_g[gw + ((48 + gq) * 68 + (32 + tq))];
                        av3_3[1] = s_g[gw + ((48 + gq + 8) * 68 + (32 + tq))];
                        av3_3[2] = s_g[gw + ((48 + gq) * 68 + (32 + tq + 4))];
                        av3_3[3] = s_g[gw + ((48 + gq + 8) * 68 + (32 + tq + 4))];
                        unsigned int ah_19[4];
                        unsigned int al_19[4];
                        unsigned int bh_19[2];
                        unsigned int bl_19[2];
                        ah_19[0] = __as_u32(av3_3[0]) + 4096 & 4294959104u;
                        al_19[0] = __as_u32(av3_3[0] - __uint_as_float(ah_19[0]));
                        ah_19[1] = __as_u32(av3_3[1]) + 4096 & 4294959104u;
                        al_19[1] = __as_u32(av3_3[1] - __uint_as_float(ah_19[1]));
                        ah_19[2] = __as_u32(av3_3[2]) + 4096 & 4294959104u;
                        al_19[2] = __as_u32(av3_3[2] - __uint_as_float(ah_19[2]));
                        ah_19[3] = __as_u32(av3_3[3]) + 4096 & 4294959104u;
                        al_19[3] = __as_u32(av3_3[3] - __uint_as_float(ah_19[3]));
                        bh_19[0] = __as_u32(bv2_1[0]) + 4096 & 4294959104u;
                        bl_19[0] = __as_u32(bv2_1[0] - __uint_as_float(bh_19[0]));
                        bh_19[1] = __as_u32(bv2_1[1]) + 4096 & 4294959104u;
                        bl_19[1] = __as_u32(bv2_1[1] - __uint_as_float(bh_19[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p2_11s[0]), "=f"(p2_11s[1]), "=f"(p2_11s[2]), "=f"(p2_11s[3])
                            : "r"(al_19[0]), "r"(al_19[1]), "r"(al_19[2]), "r"(al_19[3]), "r"(bh_19[0]), "r"(bh_19[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(p2_11b[0]), "=f"(p2_11b[1]), "=f"(p2_11b[2]), "=f"(p2_11b[3])
                            : "r"(ah_19[0]), "r"(ah_19[1]), "r"(ah_19[2]), "r"(ah_19[3]), "r"(bh_19[0]), "r"(bh_19[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11s[0]), "+f"(p2_11s[1]), "+f"(p2_11s[2]), "+f"(p2_11s[3])
                            : "r"(ah_19[0]), "r"(ah_19[1]), "r"(ah_19[2]), "r"(ah_19[3]), "r"(bl_19[0]), "r"(bl_19[1]));
                        float bv2_2[2];
                        bv2_2[0] = s_g[gw + ((16 * pw + gq) * 68 + (40 + tq))];
                        bv2_2[1] = s_g[gw + ((16 * pw + gq) * 68 + (40 + tq + 4))];
                        float av3_4[4];
                        av3_4[0] = s_g[gw + ((32 + gq) * 68 + (40 + tq))];
                        av3_4[1] = s_g[gw + ((32 + gq + 8) * 68 + (40 + tq))];
                        av3_4[2] = s_g[gw + ((32 + gq) * 68 + (40 + tq + 4))];
                        av3_4[3] = s_g[gw + ((32 + gq + 8) * 68 + (40 + tq + 4))];
                        unsigned int ah_20[4];
                        unsigned int al_20[4];
                        unsigned int bh_20[2];
                        unsigned int bl_20[2];
                        ah_20[0] = __as_u32(av3_4[0]) + 4096 & 4294959104u;
                        al_20[0] = __as_u32(av3_4[0] - __uint_as_float(ah_20[0]));
                        ah_20[1] = __as_u32(av3_4[1]) + 4096 & 4294959104u;
                        al_20[1] = __as_u32(av3_4[1] - __uint_as_float(ah_20[1]));
                        ah_20[2] = __as_u32(av3_4[2]) + 4096 & 4294959104u;
                        al_20[2] = __as_u32(av3_4[2] - __uint_as_float(ah_20[2]));
                        ah_20[3] = __as_u32(av3_4[3]) + 4096 & 4294959104u;
                        al_20[3] = __as_u32(av3_4[3] - __uint_as_float(ah_20[3]));
                        bh_20[0] = __as_u32(bv2_2[0]) + 4096 & 4294959104u;
                        bl_20[0] = __as_u32(bv2_2[0] - __uint_as_float(bh_20[0]));
                        bh_20[1] = __as_u32(bv2_2[1]) + 4096 & 4294959104u;
                        bl_20[1] = __as_u32(bv2_2[1] - __uint_as_float(bh_20[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_00s[0]), "+f"(p2_00s[1]), "+f"(p2_00s[2]), "+f"(p2_00s[3])
                            : "r"(al_20[0]), "r"(al_20[1]), "r"(al_20[2]), "r"(al_20[3]), "r"(bh_20[0]), "r"(bh_20[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_00b[0]), "+f"(p2_00b[1]), "+f"(p2_00b[2]), "+f"(p2_00b[3])
                            : "r"(ah_20[0]), "r"(ah_20[1]), "r"(ah_20[2]), "r"(ah_20[3]), "r"(bh_20[0]), "r"(bh_20[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_00s[0]), "+f"(p2_00s[1]), "+f"(p2_00s[2]), "+f"(p2_00s[3])
                            : "r"(ah_20[0]), "r"(ah_20[1]), "r"(ah_20[2]), "r"(ah_20[3]), "r"(bl_20[0]), "r"(bl_20[1]));
                        float av3_5[4];
                        av3_5[0] = s_g[gw + ((48 + gq) * 68 + (40 + tq))];
                        av3_5[1] = s_g[gw + ((48 + gq + 8) * 68 + (40 + tq))];
                        av3_5[2] = s_g[gw + ((48 + gq) * 68 + (40 + tq + 4))];
                        av3_5[3] = s_g[gw + ((48 + gq + 8) * 68 + (40 + tq + 4))];
                        unsigned int ah_21[4];
                        unsigned int al_21[4];
                        unsigned int bh_21[2];
                        unsigned int bl_21[2];
                        ah_21[0] = __as_u32(av3_5[0]) + 4096 & 4294959104u;
                        al_21[0] = __as_u32(av3_5[0] - __uint_as_float(ah_21[0]));
                        ah_21[1] = __as_u32(av3_5[1]) + 4096 & 4294959104u;
                        al_21[1] = __as_u32(av3_5[1] - __uint_as_float(ah_21[1]));
                        ah_21[2] = __as_u32(av3_5[2]) + 4096 & 4294959104u;
                        al_21[2] = __as_u32(av3_5[2] - __uint_as_float(ah_21[2]));
                        ah_21[3] = __as_u32(av3_5[3]) + 4096 & 4294959104u;
                        al_21[3] = __as_u32(av3_5[3] - __uint_as_float(ah_21[3]));
                        bh_21[0] = __as_u32(bv2_2[0]) + 4096 & 4294959104u;
                        bl_21[0] = __as_u32(bv2_2[0] - __uint_as_float(bh_21[0]));
                        bh_21[1] = __as_u32(bv2_2[1]) + 4096 & 4294959104u;
                        bl_21[1] = __as_u32(bv2_2[1] - __uint_as_float(bh_21[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01s[0]), "+f"(p2_01s[1]), "+f"(p2_01s[2]), "+f"(p2_01s[3])
                            : "r"(al_21[0]), "r"(al_21[1]), "r"(al_21[2]), "r"(al_21[3]), "r"(bh_21[0]), "r"(bh_21[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01b[0]), "+f"(p2_01b[1]), "+f"(p2_01b[2]), "+f"(p2_01b[3])
                            : "r"(ah_21[0]), "r"(ah_21[1]), "r"(ah_21[2]), "r"(ah_21[3]), "r"(bh_21[0]), "r"(bh_21[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01s[0]), "+f"(p2_01s[1]), "+f"(p2_01s[2]), "+f"(p2_01s[3])
                            : "r"(ah_21[0]), "r"(ah_21[1]), "r"(ah_21[2]), "r"(ah_21[3]), "r"(bl_21[0]), "r"(bl_21[1]));
                        float bv2_3[2];
                        bv2_3[0] = s_g[gw + ((16 * pw + 8 + gq) * 68 + (40 + tq))];
                        bv2_3[1] = s_g[gw + ((16 * pw + 8 + gq) * 68 + (40 + tq + 4))];
                        float av3_6[4];
                        av3_6[0] = s_g[gw + ((32 + gq) * 68 + (40 + tq))];
                        av3_6[1] = s_g[gw + ((32 + gq + 8) * 68 + (40 + tq))];
                        av3_6[2] = s_g[gw + ((32 + gq) * 68 + (40 + tq + 4))];
                        av3_6[3] = s_g[gw + ((32 + gq + 8) * 68 + (40 + tq + 4))];
                        unsigned int ah_22[4];
                        unsigned int al_22[4];
                        unsigned int bh_22[2];
                        unsigned int bl_22[2];
                        ah_22[0] = __as_u32(av3_6[0]) + 4096 & 4294959104u;
                        al_22[0] = __as_u32(av3_6[0] - __uint_as_float(ah_22[0]));
                        ah_22[1] = __as_u32(av3_6[1]) + 4096 & 4294959104u;
                        al_22[1] = __as_u32(av3_6[1] - __uint_as_float(ah_22[1]));
                        ah_22[2] = __as_u32(av3_6[2]) + 4096 & 4294959104u;
                        al_22[2] = __as_u32(av3_6[2] - __uint_as_float(ah_22[2]));
                        ah_22[3] = __as_u32(av3_6[3]) + 4096 & 4294959104u;
                        al_22[3] = __as_u32(av3_6[3] - __uint_as_float(ah_22[3]));
                        bh_22[0] = __as_u32(bv2_3[0]) + 4096 & 4294959104u;
                        bl_22[0] = __as_u32(bv2_3[0] - __uint_as_float(bh_22[0]));
                        bh_22[1] = __as_u32(bv2_3[1]) + 4096 & 4294959104u;
                        bl_22[1] = __as_u32(bv2_3[1] - __uint_as_float(bh_22[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_10s[0]), "+f"(p2_10s[1]), "+f"(p2_10s[2]), "+f"(p2_10s[3])
                            : "r"(al_22[0]), "r"(al_22[1]), "r"(al_22[2]), "r"(al_22[3]), "r"(bh_22[0]), "r"(bh_22[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_10b[0]), "+f"(p2_10b[1]), "+f"(p2_10b[2]), "+f"(p2_10b[3])
                            : "r"(ah_22[0]), "r"(ah_22[1]), "r"(ah_22[2]), "r"(ah_22[3]), "r"(bh_22[0]), "r"(bh_22[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_10s[0]), "+f"(p2_10s[1]), "+f"(p2_10s[2]), "+f"(p2_10s[3])
                            : "r"(ah_22[0]), "r"(ah_22[1]), "r"(ah_22[2]), "r"(ah_22[3]), "r"(bl_22[0]), "r"(bl_22[1]));
                        float av3_7[4];
                        av3_7[0] = s_g[gw + ((48 + gq) * 68 + (40 + tq))];
                        av3_7[1] = s_g[gw + ((48 + gq + 8) * 68 + (40 + tq))];
                        av3_7[2] = s_g[gw + ((48 + gq) * 68 + (40 + tq + 4))];
                        av3_7[3] = s_g[gw + ((48 + gq + 8) * 68 + (40 + tq + 4))];
                        unsigned int ah_23[4];
                        unsigned int al_23[4];
                        unsigned int bh_23[2];
                        unsigned int bl_23[2];
                        ah_23[0] = __as_u32(av3_7[0]) + 4096 & 4294959104u;
                        al_23[0] = __as_u32(av3_7[0] - __uint_as_float(ah_23[0]));
                        ah_23[1] = __as_u32(av3_7[1]) + 4096 & 4294959104u;
                        al_23[1] = __as_u32(av3_7[1] - __uint_as_float(ah_23[1]));
                        ah_23[2] = __as_u32(av3_7[2]) + 4096 & 4294959104u;
                        al_23[2] = __as_u32(av3_7[2] - __uint_as_float(ah_23[2]));
                        ah_23[3] = __as_u32(av3_7[3]) + 4096 & 4294959104u;
                        al_23[3] = __as_u32(av3_7[3] - __uint_as_float(ah_23[3]));
                        bh_23[0] = __as_u32(bv2_3[0]) + 4096 & 4294959104u;
                        bl_23[0] = __as_u32(bv2_3[0] - __uint_as_float(bh_23[0]));
                        bh_23[1] = __as_u32(bv2_3[1]) + 4096 & 4294959104u;
                        bl_23[1] = __as_u32(bv2_3[1] - __uint_as_float(bh_23[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11s[0]), "+f"(p2_11s[1]), "+f"(p2_11s[2]), "+f"(p2_11s[3])
                            : "r"(al_23[0]), "r"(al_23[1]), "r"(al_23[2]), "r"(al_23[3]), "r"(bh_23[0]), "r"(bh_23[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11b[0]), "+f"(p2_11b[1]), "+f"(p2_11b[2]), "+f"(p2_11b[3])
                            : "r"(ah_23[0]), "r"(ah_23[1]), "r"(ah_23[2]), "r"(ah_23[3]), "r"(bh_23[0]), "r"(bh_23[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11s[0]), "+f"(p2_11s[1]), "+f"(p2_11s[2]), "+f"(p2_11s[3])
                            : "r"(ah_23[0]), "r"(ah_23[1]), "r"(ah_23[2]), "r"(ah_23[3]), "r"(bl_23[0]), "r"(bl_23[1]));
                        float bv2_4[2];
                        bv2_4[0] = s_g[gw + ((16 * pw + gq) * 68 + (48 + tq))];
                        bv2_4[1] = s_g[gw + ((16 * pw + gq) * 68 + (48 + tq + 4))];
                        float av3_8[4];
                        av3_8[0] = s_g[gw + ((48 + gq) * 68 + (48 + tq))];
                        av3_8[1] = s_g[gw + ((48 + gq + 8) * 68 + (48 + tq))];
                        av3_8[2] = s_g[gw + ((48 + gq) * 68 + (48 + tq + 4))];
                        av3_8[3] = s_g[gw + ((48 + gq + 8) * 68 + (48 + tq + 4))];
                        unsigned int ah_24[4];
                        unsigned int al_24[4];
                        unsigned int bh_24[2];
                        unsigned int bl_24[2];
                        ah_24[0] = __as_u32(av3_8[0]) + 4096 & 4294959104u;
                        al_24[0] = __as_u32(av3_8[0] - __uint_as_float(ah_24[0]));
                        ah_24[1] = __as_u32(av3_8[1]) + 4096 & 4294959104u;
                        al_24[1] = __as_u32(av3_8[1] - __uint_as_float(ah_24[1]));
                        ah_24[2] = __as_u32(av3_8[2]) + 4096 & 4294959104u;
                        al_24[2] = __as_u32(av3_8[2] - __uint_as_float(ah_24[2]));
                        ah_24[3] = __as_u32(av3_8[3]) + 4096 & 4294959104u;
                        al_24[3] = __as_u32(av3_8[3] - __uint_as_float(ah_24[3]));
                        bh_24[0] = __as_u32(bv2_4[0]) + 4096 & 4294959104u;
                        bl_24[0] = __as_u32(bv2_4[0] - __uint_as_float(bh_24[0]));
                        bh_24[1] = __as_u32(bv2_4[1]) + 4096 & 4294959104u;
                        bl_24[1] = __as_u32(bv2_4[1] - __uint_as_float(bh_24[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01s[0]), "+f"(p2_01s[1]), "+f"(p2_01s[2]), "+f"(p2_01s[3])
                            : "r"(al_24[0]), "r"(al_24[1]), "r"(al_24[2]), "r"(al_24[3]), "r"(bh_24[0]), "r"(bh_24[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01b[0]), "+f"(p2_01b[1]), "+f"(p2_01b[2]), "+f"(p2_01b[3])
                            : "r"(ah_24[0]), "r"(ah_24[1]), "r"(ah_24[2]), "r"(ah_24[3]), "r"(bh_24[0]), "r"(bh_24[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01s[0]), "+f"(p2_01s[1]), "+f"(p2_01s[2]), "+f"(p2_01s[3])
                            : "r"(ah_24[0]), "r"(ah_24[1]), "r"(ah_24[2]), "r"(ah_24[3]), "r"(bl_24[0]), "r"(bl_24[1]));
                        float bv2_5[2];
                        bv2_5[0] = s_g[gw + ((16 * pw + 8 + gq) * 68 + (48 + tq))];
                        bv2_5[1] = s_g[gw + ((16 * pw + 8 + gq) * 68 + (48 + tq + 4))];
                        float av3_9[4];
                        av3_9[0] = s_g[gw + ((48 + gq) * 68 + (48 + tq))];
                        av3_9[1] = s_g[gw + ((48 + gq + 8) * 68 + (48 + tq))];
                        av3_9[2] = s_g[gw + ((48 + gq) * 68 + (48 + tq + 4))];
                        av3_9[3] = s_g[gw + ((48 + gq + 8) * 68 + (48 + tq + 4))];
                        unsigned int ah_25[4];
                        unsigned int al_25[4];
                        unsigned int bh_25[2];
                        unsigned int bl_25[2];
                        ah_25[0] = __as_u32(av3_9[0]) + 4096 & 4294959104u;
                        al_25[0] = __as_u32(av3_9[0] - __uint_as_float(ah_25[0]));
                        ah_25[1] = __as_u32(av3_9[1]) + 4096 & 4294959104u;
                        al_25[1] = __as_u32(av3_9[1] - __uint_as_float(ah_25[1]));
                        ah_25[2] = __as_u32(av3_9[2]) + 4096 & 4294959104u;
                        al_25[2] = __as_u32(av3_9[2] - __uint_as_float(ah_25[2]));
                        ah_25[3] = __as_u32(av3_9[3]) + 4096 & 4294959104u;
                        al_25[3] = __as_u32(av3_9[3] - __uint_as_float(ah_25[3]));
                        bh_25[0] = __as_u32(bv2_5[0]) + 4096 & 4294959104u;
                        bl_25[0] = __as_u32(bv2_5[0] - __uint_as_float(bh_25[0]));
                        bh_25[1] = __as_u32(bv2_5[1]) + 4096 & 4294959104u;
                        bl_25[1] = __as_u32(bv2_5[1] - __uint_as_float(bh_25[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11s[0]), "+f"(p2_11s[1]), "+f"(p2_11s[2]), "+f"(p2_11s[3])
                            : "r"(al_25[0]), "r"(al_25[1]), "r"(al_25[2]), "r"(al_25[3]), "r"(bh_25[0]), "r"(bh_25[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11b[0]), "+f"(p2_11b[1]), "+f"(p2_11b[2]), "+f"(p2_11b[3])
                            : "r"(ah_25[0]), "r"(ah_25[1]), "r"(ah_25[2]), "r"(ah_25[3]), "r"(bh_25[0]), "r"(bh_25[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11s[0]), "+f"(p2_11s[1]), "+f"(p2_11s[2]), "+f"(p2_11s[3])
                            : "r"(ah_25[0]), "r"(ah_25[1]), "r"(ah_25[2]), "r"(ah_25[3]), "r"(bl_25[0]), "r"(bl_25[1]));
                        float bv2_6[2];
                        bv2_6[0] = s_g[gw + ((16 * pw + gq) * 68 + (56 + tq))];
                        bv2_6[1] = s_g[gw + ((16 * pw + gq) * 68 + (56 + tq + 4))];
                        float av3_10[4];
                        av3_10[0] = s_g[gw + ((48 + gq) * 68 + (56 + tq))];
                        av3_10[1] = s_g[gw + ((48 + gq + 8) * 68 + (56 + tq))];
                        av3_10[2] = s_g[gw + ((48 + gq) * 68 + (56 + tq + 4))];
                        av3_10[3] = s_g[gw + ((48 + gq + 8) * 68 + (56 + tq + 4))];
                        unsigned int ah_26[4];
                        unsigned int al_26[4];
                        unsigned int bh_26[2];
                        unsigned int bl_26[2];
                        ah_26[0] = __as_u32(av3_10[0]) + 4096 & 4294959104u;
                        al_26[0] = __as_u32(av3_10[0] - __uint_as_float(ah_26[0]));
                        ah_26[1] = __as_u32(av3_10[1]) + 4096 & 4294959104u;
                        al_26[1] = __as_u32(av3_10[1] - __uint_as_float(ah_26[1]));
                        ah_26[2] = __as_u32(av3_10[2]) + 4096 & 4294959104u;
                        al_26[2] = __as_u32(av3_10[2] - __uint_as_float(ah_26[2]));
                        ah_26[3] = __as_u32(av3_10[3]) + 4096 & 4294959104u;
                        al_26[3] = __as_u32(av3_10[3] - __uint_as_float(ah_26[3]));
                        bh_26[0] = __as_u32(bv2_6[0]) + 4096 & 4294959104u;
                        bl_26[0] = __as_u32(bv2_6[0] - __uint_as_float(bh_26[0]));
                        bh_26[1] = __as_u32(bv2_6[1]) + 4096 & 4294959104u;
                        bl_26[1] = __as_u32(bv2_6[1] - __uint_as_float(bh_26[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01s[0]), "+f"(p2_01s[1]), "+f"(p2_01s[2]), "+f"(p2_01s[3])
                            : "r"(al_26[0]), "r"(al_26[1]), "r"(al_26[2]), "r"(al_26[3]), "r"(bh_26[0]), "r"(bh_26[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01b[0]), "+f"(p2_01b[1]), "+f"(p2_01b[2]), "+f"(p2_01b[3])
                            : "r"(ah_26[0]), "r"(ah_26[1]), "r"(ah_26[2]), "r"(ah_26[3]), "r"(bh_26[0]), "r"(bh_26[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_01s[0]), "+f"(p2_01s[1]), "+f"(p2_01s[2]), "+f"(p2_01s[3])
                            : "r"(ah_26[0]), "r"(ah_26[1]), "r"(ah_26[2]), "r"(ah_26[3]), "r"(bl_26[0]), "r"(bl_26[1]));
                        float bv2_7[2];
                        bv2_7[0] = s_g[gw + ((16 * pw + 8 + gq) * 68 + (56 + tq))];
                        bv2_7[1] = s_g[gw + ((16 * pw + 8 + gq) * 68 + (56 + tq + 4))];
                        float av3_11[4];
                        av3_11[0] = s_g[gw + ((48 + gq) * 68 + (56 + tq))];
                        av3_11[1] = s_g[gw + ((48 + gq + 8) * 68 + (56 + tq))];
                        av3_11[2] = s_g[gw + ((48 + gq) * 68 + (56 + tq + 4))];
                        av3_11[3] = s_g[gw + ((48 + gq + 8) * 68 + (56 + tq + 4))];
                        unsigned int ah_27[4];
                        unsigned int al_27[4];
                        unsigned int bh_27[2];
                        unsigned int bl_27[2];
                        ah_27[0] = __as_u32(av3_11[0]) + 4096 & 4294959104u;
                        al_27[0] = __as_u32(av3_11[0] - __uint_as_float(ah_27[0]));
                        ah_27[1] = __as_u32(av3_11[1]) + 4096 & 4294959104u;
                        al_27[1] = __as_u32(av3_11[1] - __uint_as_float(ah_27[1]));
                        ah_27[2] = __as_u32(av3_11[2]) + 4096 & 4294959104u;
                        al_27[2] = __as_u32(av3_11[2] - __uint_as_float(ah_27[2]));
                        ah_27[3] = __as_u32(av3_11[3]) + 4096 & 4294959104u;
                        al_27[3] = __as_u32(av3_11[3] - __uint_as_float(ah_27[3]));
                        bh_27[0] = __as_u32(bv2_7[0]) + 4096 & 4294959104u;
                        bl_27[0] = __as_u32(bv2_7[0] - __uint_as_float(bh_27[0]));
                        bh_27[1] = __as_u32(bv2_7[1]) + 4096 & 4294959104u;
                        bl_27[1] = __as_u32(bv2_7[1] - __uint_as_float(bh_27[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11s[0]), "+f"(p2_11s[1]), "+f"(p2_11s[2]), "+f"(p2_11s[3])
                            : "r"(al_27[0]), "r"(al_27[1]), "r"(al_27[2]), "r"(al_27[3]), "r"(bh_27[0]), "r"(bh_27[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11b[0]), "+f"(p2_11b[1]), "+f"(p2_11b[2]), "+f"(p2_11b[3])
                            : "r"(ah_27[0]), "r"(ah_27[1]), "r"(ah_27[2]), "r"(ah_27[3]), "r"(bh_27[0]), "r"(bh_27[1]));
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(p2_11s[0]), "+f"(p2_11s[1]), "+f"(p2_11s[2]), "+f"(p2_11s[3])
                            : "r"(ah_27[0]), "r"(ah_27[1]), "r"(ah_27[2]), "r"(ah_27[3]), "r"(bl_27[0]), "r"(bl_27[1]));
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((32 + gq) * 68 + (16 * pw + 2 * tq)) * 4)), "f"(-(p2_00b[0] + p2_00s[0])), "f"(-(p2_00b[1] + p2_00s[1])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((40 + gq) * 68 + (16 * pw + 2 * tq)) * 4)), "f"(-(p2_00b[2] + p2_00s[2])), "f"(-(p2_00b[3] + p2_00s[3])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((48 + gq) * 68 + (16 * pw + 2 * tq)) * 4)), "f"(-(p2_01b[0] + p2_01s[0])), "f"(-(p2_01b[1] + p2_01s[1])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((56 + gq) * 68 + (16 * pw + 2 * tq)) * 4)), "f"(-(p2_01b[2] + p2_01s[2])), "f"(-(p2_01b[3] + p2_01s[3])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((32 + gq) * 68 + (16 * pw + 8 + 2 * tq)) * 4)), "f"(-(p2_10b[0] + p2_10s[0])), "f"(-(p2_10b[1] + p2_10s[1])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((40 + gq) * 68 + (16 * pw + 8 + 2 * tq)) * 4)), "f"(-(p2_10b[2] + p2_10s[2])), "f"(-(p2_10b[3] + p2_10s[3])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((48 + gq) * 68 + (16 * pw + 8 + 2 * tq)) * 4)), "f"(-(p2_11b[0] + p2_11s[0])), "f"(-(p2_11b[1] + p2_11s[1])) : "memory");
                        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(lb + (unsigned int)(((56 + gq) * 68 + (16 * pw + 8 + 2 * tq)) * 4)), "f"(-(p2_11b[2] + p2_11s[2])), "f"(-(p2_11b[3] + p2_11s[3])) : "memory");
                    }
                    if (grp == 0) {
                        asm volatile("barrier.sync 1, 64;" ::: "memory");
                    }
                    if (grp == 1) {
                        asm volatile("barrier.sync 2, 64;" ::: "memory");
                    }
                    if (grp == 2) {
                        asm volatile("barrier.sync 3, 64;" ::: "memory");
                    }
                    if (grp == 3) {
                        asm volatile("barrier.sync 5, 64;" ::: "memory");
                    }
                    if (grp == 4) {
                        asm volatile("barrier.sync 6, 64;" ::: "memory");
                    }
                    if (grp == 5) {
                        asm volatile("barrier.sync 7, 64;" ::: "memory");
                    }
                    if (hgrp < gpk) {
                        int oc = lane_1 % 8;
                        int o_t = (32 * pw + lane_1) / 8;
                        long long obase = ((long long)tok0 * (long long)num_v_heads + (long long)(hq * gpk + hgrp)) * 32ll;
                        int orow_0 = o_t;
                        if (orow_0 < n) {
                            float xv_0[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_0[(0) + 3]))
                                : "r"(lb + (unsigned int)((orow_0 * 68 + 8 * oc) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_0[(4) + 3]))
                                : "r"(lb + (unsigned int)((orow_0 * 68 + (8 * oc + 4)) * 4)));
                            float ta_0[8];
                            ta_0[0] = ((8 * oc <= orow_0) ? xv_0[0] : 0.0f);
                            ta_0[1] = ((8 * oc + 1 <= orow_0) ? xv_0[1] : 0.0f);
                            ta_0[2] = ((8 * oc + 2 <= orow_0) ? xv_0[2] : 0.0f);
                            ta_0[3] = ((8 * oc + 3 <= orow_0) ? xv_0[3] : 0.0f);
                            ta_0[4] = ((8 * oc + 4 <= orow_0) ? xv_0[4] : 0.0f);
                            ta_0[5] = ((8 * oc + 5 <= orow_0) ? xv_0[5] : 0.0f);
                            ta_0[6] = ((8 * oc + 6 <= orow_0) ? xv_0[6] : 0.0f);
                            ta_0[7] = ((8 * oc + 7 <= orow_0) ? xv_0[7] : 0.0f);
                            unsigned int pa_0[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ta_0[_lp*2 + 0], ta_0[_lp*2+1 + 0]));
                                pa_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(a_out + (obase + (long long)orow_0 * ostride + (long long)(4 * oc)))[0] = reinterpret_cast<int4*>(pa_0)[0];
                        }
                        int orow_1 = o_t + 8;
                        if (orow_1 < n) {
                            float xv_1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_1[(0) + 3]))
                                : "r"(lb + (unsigned int)((orow_1 * 68 + 8 * oc) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_1[(4) + 3]))
                                : "r"(lb + (unsigned int)((orow_1 * 68 + (8 * oc + 4)) * 4)));
                            float ta_1[8];
                            ta_1[0] = ((8 * oc <= orow_1) ? xv_1[0] : 0.0f);
                            ta_1[1] = ((8 * oc + 1 <= orow_1) ? xv_1[1] : 0.0f);
                            ta_1[2] = ((8 * oc + 2 <= orow_1) ? xv_1[2] : 0.0f);
                            ta_1[3] = ((8 * oc + 3 <= orow_1) ? xv_1[3] : 0.0f);
                            ta_1[4] = ((8 * oc + 4 <= orow_1) ? xv_1[4] : 0.0f);
                            ta_1[5] = ((8 * oc + 5 <= orow_1) ? xv_1[5] : 0.0f);
                            ta_1[6] = ((8 * oc + 6 <= orow_1) ? xv_1[6] : 0.0f);
                            ta_1[7] = ((8 * oc + 7 <= orow_1) ? xv_1[7] : 0.0f);
                            unsigned int pa_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ta_1[_lp*2 + 0], ta_1[_lp*2+1 + 0]));
                                pa_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(a_out + (obase + (long long)orow_1 * ostride + (long long)(4 * oc)))[0] = reinterpret_cast<int4*>(pa_1)[0];
                        }
                        int orow_2 = o_t + 16;
                        if (orow_2 < n) {
                            float xv_2[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_2[(0) + 3]))
                                : "r"(lb + (unsigned int)((orow_2 * 68 + 8 * oc) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_2[(4) + 3]))
                                : "r"(lb + (unsigned int)((orow_2 * 68 + (8 * oc + 4)) * 4)));
                            float ta_2[8];
                            ta_2[0] = ((8 * oc <= orow_2) ? xv_2[0] : 0.0f);
                            ta_2[1] = ((8 * oc + 1 <= orow_2) ? xv_2[1] : 0.0f);
                            ta_2[2] = ((8 * oc + 2 <= orow_2) ? xv_2[2] : 0.0f);
                            ta_2[3] = ((8 * oc + 3 <= orow_2) ? xv_2[3] : 0.0f);
                            ta_2[4] = ((8 * oc + 4 <= orow_2) ? xv_2[4] : 0.0f);
                            ta_2[5] = ((8 * oc + 5 <= orow_2) ? xv_2[5] : 0.0f);
                            ta_2[6] = ((8 * oc + 6 <= orow_2) ? xv_2[6] : 0.0f);
                            ta_2[7] = ((8 * oc + 7 <= orow_2) ? xv_2[7] : 0.0f);
                            unsigned int pa_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ta_2[_lp*2 + 0], ta_2[_lp*2+1 + 0]));
                                pa_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(a_out + (obase + (long long)orow_2 * ostride + (long long)(4 * oc)))[0] = reinterpret_cast<int4*>(pa_2)[0];
                        }
                        int orow_3 = o_t + 24;
                        if (orow_3 < n) {
                            float xv_3[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_3[(0) + 3]))
                                : "r"(lb + (unsigned int)((orow_3 * 68 + 8 * oc) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_3[(4) + 3]))
                                : "r"(lb + (unsigned int)((orow_3 * 68 + (8 * oc + 4)) * 4)));
                            float ta_3[8];
                            ta_3[0] = ((8 * oc <= orow_3) ? xv_3[0] : 0.0f);
                            ta_3[1] = ((8 * oc + 1 <= orow_3) ? xv_3[1] : 0.0f);
                            ta_3[2] = ((8 * oc + 2 <= orow_3) ? xv_3[2] : 0.0f);
                            ta_3[3] = ((8 * oc + 3 <= orow_3) ? xv_3[3] : 0.0f);
                            ta_3[4] = ((8 * oc + 4 <= orow_3) ? xv_3[4] : 0.0f);
                            ta_3[5] = ((8 * oc + 5 <= orow_3) ? xv_3[5] : 0.0f);
                            ta_3[6] = ((8 * oc + 6 <= orow_3) ? xv_3[6] : 0.0f);
                            ta_3[7] = ((8 * oc + 7 <= orow_3) ? xv_3[7] : 0.0f);
                            unsigned int pa_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ta_3[_lp*2 + 0], ta_3[_lp*2+1 + 0]));
                                pa_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(a_out + (obase + (long long)orow_3 * ostride + (long long)(4 * oc)))[0] = reinterpret_cast<int4*>(pa_3)[0];
                        }
                        int orow_4 = o_t + 32;
                        if (orow_4 < n) {
                            float xv_4[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_4[(0) + 3]))
                                : "r"(lb + (unsigned int)((orow_4 * 68 + 8 * oc) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_4[(4) + 3]))
                                : "r"(lb + (unsigned int)((orow_4 * 68 + (8 * oc + 4)) * 4)));
                            float ta_4[8];
                            ta_4[0] = ((8 * oc <= orow_4) ? xv_4[0] : 0.0f);
                            ta_4[1] = ((8 * oc + 1 <= orow_4) ? xv_4[1] : 0.0f);
                            ta_4[2] = ((8 * oc + 2 <= orow_4) ? xv_4[2] : 0.0f);
                            ta_4[3] = ((8 * oc + 3 <= orow_4) ? xv_4[3] : 0.0f);
                            ta_4[4] = ((8 * oc + 4 <= orow_4) ? xv_4[4] : 0.0f);
                            ta_4[5] = ((8 * oc + 5 <= orow_4) ? xv_4[5] : 0.0f);
                            ta_4[6] = ((8 * oc + 6 <= orow_4) ? xv_4[6] : 0.0f);
                            ta_4[7] = ((8 * oc + 7 <= orow_4) ? xv_4[7] : 0.0f);
                            unsigned int pa_4[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ta_4[_lp*2 + 0], ta_4[_lp*2+1 + 0]));
                                pa_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(a_out + (obase + (long long)orow_4 * ostride + (long long)(4 * oc)))[0] = reinterpret_cast<int4*>(pa_4)[0];
                        }
                        int orow_5 = o_t + 40;
                        if (orow_5 < n) {
                            float xv_5[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_5[(0) + 3]))
                                : "r"(lb + (unsigned int)((orow_5 * 68 + 8 * oc) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_5[(4) + 3]))
                                : "r"(lb + (unsigned int)((orow_5 * 68 + (8 * oc + 4)) * 4)));
                            float ta_5[8];
                            ta_5[0] = ((8 * oc <= orow_5) ? xv_5[0] : 0.0f);
                            ta_5[1] = ((8 * oc + 1 <= orow_5) ? xv_5[1] : 0.0f);
                            ta_5[2] = ((8 * oc + 2 <= orow_5) ? xv_5[2] : 0.0f);
                            ta_5[3] = ((8 * oc + 3 <= orow_5) ? xv_5[3] : 0.0f);
                            ta_5[4] = ((8 * oc + 4 <= orow_5) ? xv_5[4] : 0.0f);
                            ta_5[5] = ((8 * oc + 5 <= orow_5) ? xv_5[5] : 0.0f);
                            ta_5[6] = ((8 * oc + 6 <= orow_5) ? xv_5[6] : 0.0f);
                            ta_5[7] = ((8 * oc + 7 <= orow_5) ? xv_5[7] : 0.0f);
                            unsigned int pa_5[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ta_5[_lp*2 + 0], ta_5[_lp*2+1 + 0]));
                                pa_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(a_out + (obase + (long long)orow_5 * ostride + (long long)(4 * oc)))[0] = reinterpret_cast<int4*>(pa_5)[0];
                        }
                        int orow_6 = o_t + 48;
                        if (orow_6 < n) {
                            float xv_6[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_6[(0) + 3]))
                                : "r"(lb + (unsigned int)((orow_6 * 68 + 8 * oc) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_6[(4) + 3]))
                                : "r"(lb + (unsigned int)((orow_6 * 68 + (8 * oc + 4)) * 4)));
                            float ta_6[8];
                            ta_6[0] = ((8 * oc <= orow_6) ? xv_6[0] : 0.0f);
                            ta_6[1] = ((8 * oc + 1 <= orow_6) ? xv_6[1] : 0.0f);
                            ta_6[2] = ((8 * oc + 2 <= orow_6) ? xv_6[2] : 0.0f);
                            ta_6[3] = ((8 * oc + 3 <= orow_6) ? xv_6[3] : 0.0f);
                            ta_6[4] = ((8 * oc + 4 <= orow_6) ? xv_6[4] : 0.0f);
                            ta_6[5] = ((8 * oc + 5 <= orow_6) ? xv_6[5] : 0.0f);
                            ta_6[6] = ((8 * oc + 6 <= orow_6) ? xv_6[6] : 0.0f);
                            ta_6[7] = ((8 * oc + 7 <= orow_6) ? xv_6[7] : 0.0f);
                            unsigned int pa_6[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ta_6[_lp*2 + 0], ta_6[_lp*2+1 + 0]));
                                pa_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(a_out + (obase + (long long)orow_6 * ostride + (long long)(4 * oc)))[0] = reinterpret_cast<int4*>(pa_6)[0];
                        }
                        int orow_7 = o_t + 56;
                        if (orow_7 < n) {
                            float xv_7[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_7[(0) + 3]))
                                : "r"(lb + (unsigned int)((orow_7 * 68 + 8 * oc) * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&xv_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv_7[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv_7[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv_7[(4) + 3]))
                                : "r"(lb + (unsigned int)((orow_7 * 68 + (8 * oc + 4)) * 4)));
                            float ta_7[8];
                            ta_7[0] = ((8 * oc <= orow_7) ? xv_7[0] : 0.0f);
                            ta_7[1] = ((8 * oc + 1 <= orow_7) ? xv_7[1] : 0.0f);
                            ta_7[2] = ((8 * oc + 2 <= orow_7) ? xv_7[2] : 0.0f);
                            ta_7[3] = ((8 * oc + 3 <= orow_7) ? xv_7[3] : 0.0f);
                            ta_7[4] = ((8 * oc + 4 <= orow_7) ? xv_7[4] : 0.0f);
                            ta_7[5] = ((8 * oc + 5 <= orow_7) ? xv_7[5] : 0.0f);
                            ta_7[6] = ((8 * oc + 6 <= orow_7) ? xv_7[6] : 0.0f);
                            ta_7[7] = ((8 * oc + 7 <= orow_7) ? xv_7[7] : 0.0f);
                            unsigned int pa_7[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ta_7[_lp*2 + 0], ta_7[_lp*2+1 + 0]));
                                pa_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(a_out + (obase + (long long)orow_7 * ostride + (long long)(4 * oc)))[0] = reinterpret_cast<int4*>(pa_7)[0];
                        }
                    }
                    if (grp == 0) {
                        asm volatile("barrier.sync 1, 64;" ::: "memory");
                    }
                    if (grp == 1) {
                        asm volatile("barrier.sync 2, 64;" ::: "memory");
                    }
                    if (grp == 2) {
                        asm volatile("barrier.sync 3, 64;" ::: "memory");
                    }
                    if (grp == 3) {
                        asm volatile("barrier.sync 5, 64;" ::: "memory");
                    }
                    if (grp == 4) {
                        asm volatile("barrier.sync 6, 64;" ::: "memory");
                    }
                    if (grp == 5) {
                        asm volatile("barrier.sync 7, 64;" ::: "memory");
                    }
                }
                mbarrier_arrive(gates_free_addr + (gs_c) * 8);
                gs_c += 1;
                if (gs_c == 4) { gs_c = 0; _phase_gates_ready ^= 1; }
            }
        }
    }
    // ---- Role: ld ----
    if (warp == 12) {
        { // ld_main
            int bid_2 = blockIdx.x;
            int grid_1 = gridDim.x;
            int n_items_1 = num_chunks * num_heads;
            int cnt_1 = 0;
            if (bid_2 < n_items_1) {
                cnt_1 = (n_items_1 - bid_2 + grid_1 - 1) / grid_1;
            }
            unsigned int ld_c = 0;
            unsigned int _phase_st_free_m = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int it_1 = 0; it_1 < cnt_1; it_1++) {
                    int item_1 = bid_2 + it_1 * grid_1;
                    int hq_1 = item_1 % num_heads;
                    int tok0_1 = chunk_start[item_1 / num_heads];
                    unsigned int so = ld_c * 49152;
                    mbarrier_wait(st_free_m_addr + (ld_c) * 8, _phase_st_free_m);
                    mbarrier_wait(st_free_r_addr + (ld_c) * 8, _phase_st_free_m);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(st_full_addr + (ld_c) * 8, 49152);
                    tma_3d_gmem2smem(s_ga0_addr + so, (&k_tma64), 0, hq_1, tok0_1, st_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_ga0_addr + so + 8192, (&k_tma64), 0, hq_1, tok0_1, st_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_ga1_addr + so, (&k_tma64), 64, hq_1, tok0_1, st_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_ga1_addr + so + 8192, (&k_tma64), 64, hq_1, tok0_1, st_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_q0_addr + so, (&q_tma), 0, hq_1, tok0_1, st_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_q1_addr + so, (&q_tma), 64, hq_1, tok0_1, st_full_addr + (ld_c) * 8);
                    ld_c += 1;
                    if (ld_c == 2) { ld_c = 0; _phase_st_free_m ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            int bid_3 = blockIdx.x;
            int grid_2 = gridDim.x;
            int n_items_2 = num_chunks * num_heads;
            int cnt_2 = 0;
            if (bid_3 < n_items_2) {
                cnt_2 = (n_items_2 - bid_3 + grid_2 - 1) / grid_2;
            }
            unsigned int mf_c = 0;
            unsigned int mg_c = 0;
            unsigned int mq_c = 0;
            int m_lane = lane;
            unsigned int _phase_st_full = 0;
            unsigned int _phase_g_free = 1;
            unsigned int _phase_gates_free = 1;
            #pragma unroll 1
            for (int it_2 = 0; it_2 < cnt_2; it_2++) {
                int m_item = bid_3 + it_2 * grid_2;
                int m_hq = m_item % num_heads;
                int m_tok0 = chunk_start[m_item / num_heads];
                int m_n = chunk_len[m_item / num_heads];
                int m_rq = it_2 % 4 * 1416 + 72 + 384;
                mbarrier_wait(st_full_addr + (mf_c) * 8, _phase_st_full);
                mbarrier_wait(g_free_addr + (mg_c) * 8, _phase_g_free);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    if (it_2 % 2 == 0) {
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_0 = (((s_ga0_addr) >> 4) & 0x7FFF) + (mf_c) * 3072;
                        int _smem_desc_lo_1 = (((s_gb0_addr) >> 4) & 0x7FFF) + (mf_c) * 3072;
#else
                        int _smem_desc_lo_0 = (((s_ga0_addr) >> 4) & 0x3FFF) + (mf_c) * 3072;
                        int _smem_desc_lo_1 = (((s_gb0_addr) >> 4) & 0x3FFF) + (mf_c) * 3072;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_g0), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_2 = (((s_ga1_addr) >> 4) & 0x7FFF) + (mf_c) * 3072;
                        int _smem_desc_lo_3 = (((s_gb1_addr) >> 4) & 0x7FFF) + (mf_c) * 3072;
#else
                        int _smem_desc_lo_2 = (((s_ga1_addr) >> 4) & 0x3FFF) + (mf_c) * 3072;
                        int _smem_desc_lo_3 = (((s_gb1_addr) >> 4) & 0x3FFF) + (mf_c) * 3072;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_g0), "r"(1) : "memory");
                    } else {
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_4 = (((s_ga0_addr) >> 4) & 0x7FFF) + (mf_c) * 3072;
                        int _smem_desc_lo_5 = (((s_gb0_addr) >> 4) & 0x7FFF) + (mf_c) * 3072;
#else
                        int _smem_desc_lo_4 = (((s_ga0_addr) >> 4) & 0x3FFF) + (mf_c) * 3072;
                        int _smem_desc_lo_5 = (((s_gb0_addr) >> 4) & 0x3FFF) + (mf_c) * 3072;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_g1), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = (((s_ga1_addr) >> 4) & 0x7FFF) + (mf_c) * 3072;
                        int _smem_desc_lo_7 = (((s_gb1_addr) >> 4) & 0x7FFF) + (mf_c) * 3072;
#else
                        int _smem_desc_lo_6 = (((s_ga1_addr) >> 4) & 0x3FFF) + (mf_c) * 3072;
                        int _smem_desc_lo_7 = (((s_gb1_addr) >> 4) & 0x3FFF) + (mf_c) * 3072;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_g1), "r"(1) : "memory");
                    }
                    tcgen05_commit(g_full_addr + (mg_c) * 8);
                    tcgen05_commit(st_free_m_addr + (mf_c) * 8);
                }
                mg_c += 1;
                if (mg_c == 2) { mg_c = 0; _phase_g_free ^= 1; }
                unsigned int mqb = s_st_addr + mf_c * 49152 + 32768;
                float mq0 = 0.0f;
                float mq1 = 0.0f;
                #pragma unroll 1
                for (int mr = 0; mr < 2; mr++) {
                    unsigned int rb1 = mqb + (unsigned int)((m_lane + 32 * mr) * 128) + 8192;
                    float aa[16];
                    unsigned int rsw_0 = (unsigned int)((0 ^ m_lane % 8) << 4);
                    unsigned int rw0_0[4];
                    unsigned int rw1_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_0[(0) + 3]))
                        : "r"(mqb + (unsigned int)((m_lane + 32 * mr) * 128) + rsw_0));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_0[(0) + 3]))
                        : "r"(rb1 + rsw_0));
                    float2 ac0_0 = make_float2(0.0f, 0.0f);
                    float2 ac1_0 = make_float2(0.0f, 0.0f);
                    float2 e0_0 = make_float2(__uint_as_float(rw0_0[0] << 16), __uint_as_float(rw0_0[2] << 16));
                    float2 e1_0 = make_float2(__uint_as_float(rw1_0[0] << 16), __uint_as_float(rw1_0[2] << 16));
                    ac0_0 = fma_f32x2_rn_ftz(e0_0, e0_0, ac0_0);
                    ac1_0 = fma_f32x2_rn_ftz(e1_0, e1_0, ac1_0);
                    float2 o0_0 = make_float2(__uint_as_float(rw0_0[0] & 4294901760u), __uint_as_float(rw0_0[2] & 4294901760u));
                    float2 o1_0 = make_float2(__uint_as_float(rw1_0[0] & 4294901760u), __uint_as_float(rw1_0[2] & 4294901760u));
                    ac0_0 = fma_f32x2_rn_ftz(o0_0, o0_0, ac0_0);
                    ac1_0 = fma_f32x2_rn_ftz(o1_0, o1_0, ac1_0);
                    float2 e0_1 = make_float2(__uint_as_float(rw0_0[1] << 16), __uint_as_float(rw0_0[3] << 16));
                    float2 e1_1 = make_float2(__uint_as_float(rw1_0[1] << 16), __uint_as_float(rw1_0[3] << 16));
                    ac0_0 = fma_f32x2_rn_ftz(e0_1, e0_1, ac0_0);
                    ac1_0 = fma_f32x2_rn_ftz(e1_1, e1_1, ac1_0);
                    float2 o0_1 = make_float2(__uint_as_float(rw0_0[1] & 4294901760u), __uint_as_float(rw0_0[3] & 4294901760u));
                    float2 o1_1 = make_float2(__uint_as_float(rw1_0[1] & 4294901760u), __uint_as_float(rw1_0[3] & 4294901760u));
                    ac0_0 = fma_f32x2_rn_ftz(o0_1, o0_1, ac0_0);
                    ac1_0 = fma_f32x2_rn_ftz(o1_1, o1_1, ac1_0);
                    aa[0] = ac0_0.x + ac1_0.x;
                    aa[1] = ac0_0.y + ac1_0.y;
                    unsigned int rsw_1 = (unsigned int)((1 ^ m_lane % 8) << 4);
                    unsigned int rw0_1[4];
                    unsigned int rw1_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_1[(0) + 3]))
                        : "r"(mqb + (unsigned int)((m_lane + 32 * mr) * 128) + rsw_1));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_1[(0) + 3]))
                        : "r"(rb1 + rsw_1));
                    float2 ac0_1 = make_float2(0.0f, 0.0f);
                    float2 ac1_1 = make_float2(0.0f, 0.0f);
                    float2 e0_2 = make_float2(__uint_as_float(rw0_1[0] << 16), __uint_as_float(rw0_1[2] << 16));
                    float2 e1_2 = make_float2(__uint_as_float(rw1_1[0] << 16), __uint_as_float(rw1_1[2] << 16));
                    ac0_1 = fma_f32x2_rn_ftz(e0_2, e0_2, ac0_1);
                    ac1_1 = fma_f32x2_rn_ftz(e1_2, e1_2, ac1_1);
                    float2 o0_2 = make_float2(__uint_as_float(rw0_1[0] & 4294901760u), __uint_as_float(rw0_1[2] & 4294901760u));
                    float2 o1_2 = make_float2(__uint_as_float(rw1_1[0] & 4294901760u), __uint_as_float(rw1_1[2] & 4294901760u));
                    ac0_1 = fma_f32x2_rn_ftz(o0_2, o0_2, ac0_1);
                    ac1_1 = fma_f32x2_rn_ftz(o1_2, o1_2, ac1_1);
                    float2 e0_3 = make_float2(__uint_as_float(rw0_1[1] << 16), __uint_as_float(rw0_1[3] << 16));
                    float2 e1_3 = make_float2(__uint_as_float(rw1_1[1] << 16), __uint_as_float(rw1_1[3] << 16));
                    ac0_1 = fma_f32x2_rn_ftz(e0_3, e0_3, ac0_1);
                    ac1_1 = fma_f32x2_rn_ftz(e1_3, e1_3, ac1_1);
                    float2 o0_3 = make_float2(__uint_as_float(rw0_1[1] & 4294901760u), __uint_as_float(rw0_1[3] & 4294901760u));
                    float2 o1_3 = make_float2(__uint_as_float(rw1_1[1] & 4294901760u), __uint_as_float(rw1_1[3] & 4294901760u));
                    ac0_1 = fma_f32x2_rn_ftz(o0_3, o0_3, ac0_1);
                    ac1_1 = fma_f32x2_rn_ftz(o1_3, o1_3, ac1_1);
                    aa[2] = ac0_1.x + ac1_1.x;
                    aa[3] = ac0_1.y + ac1_1.y;
                    unsigned int rsw_2 = (unsigned int)((2 ^ m_lane % 8) << 4);
                    unsigned int rw0_2[4];
                    unsigned int rw1_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_2[(0) + 3]))
                        : "r"(mqb + (unsigned int)((m_lane + 32 * mr) * 128) + rsw_2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_2[(0) + 3]))
                        : "r"(rb1 + rsw_2));
                    float2 ac0_2 = make_float2(0.0f, 0.0f);
                    float2 ac1_2 = make_float2(0.0f, 0.0f);
                    float2 e0_4 = make_float2(__uint_as_float(rw0_2[0] << 16), __uint_as_float(rw0_2[2] << 16));
                    float2 e1_4 = make_float2(__uint_as_float(rw1_2[0] << 16), __uint_as_float(rw1_2[2] << 16));
                    ac0_2 = fma_f32x2_rn_ftz(e0_4, e0_4, ac0_2);
                    ac1_2 = fma_f32x2_rn_ftz(e1_4, e1_4, ac1_2);
                    float2 o0_4 = make_float2(__uint_as_float(rw0_2[0] & 4294901760u), __uint_as_float(rw0_2[2] & 4294901760u));
                    float2 o1_4 = make_float2(__uint_as_float(rw1_2[0] & 4294901760u), __uint_as_float(rw1_2[2] & 4294901760u));
                    ac0_2 = fma_f32x2_rn_ftz(o0_4, o0_4, ac0_2);
                    ac1_2 = fma_f32x2_rn_ftz(o1_4, o1_4, ac1_2);
                    float2 e0_5 = make_float2(__uint_as_float(rw0_2[1] << 16), __uint_as_float(rw0_2[3] << 16));
                    float2 e1_5 = make_float2(__uint_as_float(rw1_2[1] << 16), __uint_as_float(rw1_2[3] << 16));
                    ac0_2 = fma_f32x2_rn_ftz(e0_5, e0_5, ac0_2);
                    ac1_2 = fma_f32x2_rn_ftz(e1_5, e1_5, ac1_2);
                    float2 o0_5 = make_float2(__uint_as_float(rw0_2[1] & 4294901760u), __uint_as_float(rw0_2[3] & 4294901760u));
                    float2 o1_5 = make_float2(__uint_as_float(rw1_2[1] & 4294901760u), __uint_as_float(rw1_2[3] & 4294901760u));
                    ac0_2 = fma_f32x2_rn_ftz(o0_5, o0_5, ac0_2);
                    ac1_2 = fma_f32x2_rn_ftz(o1_5, o1_5, ac1_2);
                    aa[4] = ac0_2.x + ac1_2.x;
                    aa[5] = ac0_2.y + ac1_2.y;
                    unsigned int rsw_3 = (unsigned int)((3 ^ m_lane % 8) << 4);
                    unsigned int rw0_3[4];
                    unsigned int rw1_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_3[(0) + 3]))
                        : "r"(mqb + (unsigned int)((m_lane + 32 * mr) * 128) + rsw_3));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_3[(0) + 3]))
                        : "r"(rb1 + rsw_3));
                    float2 ac0_3 = make_float2(0.0f, 0.0f);
                    float2 ac1_3 = make_float2(0.0f, 0.0f);
                    float2 e0_6 = make_float2(__uint_as_float(rw0_3[0] << 16), __uint_as_float(rw0_3[2] << 16));
                    float2 e1_6 = make_float2(__uint_as_float(rw1_3[0] << 16), __uint_as_float(rw1_3[2] << 16));
                    ac0_3 = fma_f32x2_rn_ftz(e0_6, e0_6, ac0_3);
                    ac1_3 = fma_f32x2_rn_ftz(e1_6, e1_6, ac1_3);
                    float2 o0_6 = make_float2(__uint_as_float(rw0_3[0] & 4294901760u), __uint_as_float(rw0_3[2] & 4294901760u));
                    float2 o1_6 = make_float2(__uint_as_float(rw1_3[0] & 4294901760u), __uint_as_float(rw1_3[2] & 4294901760u));
                    ac0_3 = fma_f32x2_rn_ftz(o0_6, o0_6, ac0_3);
                    ac1_3 = fma_f32x2_rn_ftz(o1_6, o1_6, ac1_3);
                    float2 e0_7 = make_float2(__uint_as_float(rw0_3[1] << 16), __uint_as_float(rw0_3[3] << 16));
                    float2 e1_7 = make_float2(__uint_as_float(rw1_3[1] << 16), __uint_as_float(rw1_3[3] << 16));
                    ac0_3 = fma_f32x2_rn_ftz(e0_7, e0_7, ac0_3);
                    ac1_3 = fma_f32x2_rn_ftz(e1_7, e1_7, ac1_3);
                    float2 o0_7 = make_float2(__uint_as_float(rw0_3[1] & 4294901760u), __uint_as_float(rw0_3[3] & 4294901760u));
                    float2 o1_7 = make_float2(__uint_as_float(rw1_3[1] & 4294901760u), __uint_as_float(rw1_3[3] & 4294901760u));
                    ac0_3 = fma_f32x2_rn_ftz(o0_7, o0_7, ac0_3);
                    ac1_3 = fma_f32x2_rn_ftz(o1_7, o1_7, ac1_3);
                    aa[6] = ac0_3.x + ac1_3.x;
                    aa[7] = ac0_3.y + ac1_3.y;
                    unsigned int rsw_4 = (unsigned int)((4 ^ m_lane % 8) << 4);
                    unsigned int rw0_4[4];
                    unsigned int rw1_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_4[(0) + 3]))
                        : "r"(mqb + (unsigned int)((m_lane + 32 * mr) * 128) + rsw_4));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_4[(0) + 3]))
                        : "r"(rb1 + rsw_4));
                    float2 ac0_4 = make_float2(0.0f, 0.0f);
                    float2 ac1_4 = make_float2(0.0f, 0.0f);
                    float2 e0_8 = make_float2(__uint_as_float(rw0_4[0] << 16), __uint_as_float(rw0_4[2] << 16));
                    float2 e1_8 = make_float2(__uint_as_float(rw1_4[0] << 16), __uint_as_float(rw1_4[2] << 16));
                    ac0_4 = fma_f32x2_rn_ftz(e0_8, e0_8, ac0_4);
                    ac1_4 = fma_f32x2_rn_ftz(e1_8, e1_8, ac1_4);
                    float2 o0_8 = make_float2(__uint_as_float(rw0_4[0] & 4294901760u), __uint_as_float(rw0_4[2] & 4294901760u));
                    float2 o1_8 = make_float2(__uint_as_float(rw1_4[0] & 4294901760u), __uint_as_float(rw1_4[2] & 4294901760u));
                    ac0_4 = fma_f32x2_rn_ftz(o0_8, o0_8, ac0_4);
                    ac1_4 = fma_f32x2_rn_ftz(o1_8, o1_8, ac1_4);
                    float2 e0_9 = make_float2(__uint_as_float(rw0_4[1] << 16), __uint_as_float(rw0_4[3] << 16));
                    float2 e1_9 = make_float2(__uint_as_float(rw1_4[1] << 16), __uint_as_float(rw1_4[3] << 16));
                    ac0_4 = fma_f32x2_rn_ftz(e0_9, e0_9, ac0_4);
                    ac1_4 = fma_f32x2_rn_ftz(e1_9, e1_9, ac1_4);
                    float2 o0_9 = make_float2(__uint_as_float(rw0_4[1] & 4294901760u), __uint_as_float(rw0_4[3] & 4294901760u));
                    float2 o1_9 = make_float2(__uint_as_float(rw1_4[1] & 4294901760u), __uint_as_float(rw1_4[3] & 4294901760u));
                    ac0_4 = fma_f32x2_rn_ftz(o0_9, o0_9, ac0_4);
                    ac1_4 = fma_f32x2_rn_ftz(o1_9, o1_9, ac1_4);
                    aa[8] = ac0_4.x + ac1_4.x;
                    aa[9] = ac0_4.y + ac1_4.y;
                    unsigned int rsw_5 = (unsigned int)((5 ^ m_lane % 8) << 4);
                    unsigned int rw0_5[4];
                    unsigned int rw1_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_5[(0) + 3]))
                        : "r"(mqb + (unsigned int)((m_lane + 32 * mr) * 128) + rsw_5));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_5[(0) + 3]))
                        : "r"(rb1 + rsw_5));
                    float2 ac0_5 = make_float2(0.0f, 0.0f);
                    float2 ac1_5 = make_float2(0.0f, 0.0f);
                    float2 e0_10 = make_float2(__uint_as_float(rw0_5[0] << 16), __uint_as_float(rw0_5[2] << 16));
                    float2 e1_10 = make_float2(__uint_as_float(rw1_5[0] << 16), __uint_as_float(rw1_5[2] << 16));
                    ac0_5 = fma_f32x2_rn_ftz(e0_10, e0_10, ac0_5);
                    ac1_5 = fma_f32x2_rn_ftz(e1_10, e1_10, ac1_5);
                    float2 o0_10 = make_float2(__uint_as_float(rw0_5[0] & 4294901760u), __uint_as_float(rw0_5[2] & 4294901760u));
                    float2 o1_10 = make_float2(__uint_as_float(rw1_5[0] & 4294901760u), __uint_as_float(rw1_5[2] & 4294901760u));
                    ac0_5 = fma_f32x2_rn_ftz(o0_10, o0_10, ac0_5);
                    ac1_5 = fma_f32x2_rn_ftz(o1_10, o1_10, ac1_5);
                    float2 e0_11 = make_float2(__uint_as_float(rw0_5[1] << 16), __uint_as_float(rw0_5[3] << 16));
                    float2 e1_11 = make_float2(__uint_as_float(rw1_5[1] << 16), __uint_as_float(rw1_5[3] << 16));
                    ac0_5 = fma_f32x2_rn_ftz(e0_11, e0_11, ac0_5);
                    ac1_5 = fma_f32x2_rn_ftz(e1_11, e1_11, ac1_5);
                    float2 o0_11 = make_float2(__uint_as_float(rw0_5[1] & 4294901760u), __uint_as_float(rw0_5[3] & 4294901760u));
                    float2 o1_11 = make_float2(__uint_as_float(rw1_5[1] & 4294901760u), __uint_as_float(rw1_5[3] & 4294901760u));
                    ac0_5 = fma_f32x2_rn_ftz(o0_11, o0_11, ac0_5);
                    ac1_5 = fma_f32x2_rn_ftz(o1_11, o1_11, ac1_5);
                    aa[10] = ac0_5.x + ac1_5.x;
                    aa[11] = ac0_5.y + ac1_5.y;
                    unsigned int rsw_6 = (unsigned int)((6 ^ m_lane % 8) << 4);
                    unsigned int rw0_6[4];
                    unsigned int rw1_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_6[(0) + 3]))
                        : "r"(mqb + (unsigned int)((m_lane + 32 * mr) * 128) + rsw_6));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_6[(0) + 3]))
                        : "r"(rb1 + rsw_6));
                    float2 ac0_6 = make_float2(0.0f, 0.0f);
                    float2 ac1_6 = make_float2(0.0f, 0.0f);
                    float2 e0_12 = make_float2(__uint_as_float(rw0_6[0] << 16), __uint_as_float(rw0_6[2] << 16));
                    float2 e1_12 = make_float2(__uint_as_float(rw1_6[0] << 16), __uint_as_float(rw1_6[2] << 16));
                    ac0_6 = fma_f32x2_rn_ftz(e0_12, e0_12, ac0_6);
                    ac1_6 = fma_f32x2_rn_ftz(e1_12, e1_12, ac1_6);
                    float2 o0_12 = make_float2(__uint_as_float(rw0_6[0] & 4294901760u), __uint_as_float(rw0_6[2] & 4294901760u));
                    float2 o1_12 = make_float2(__uint_as_float(rw1_6[0] & 4294901760u), __uint_as_float(rw1_6[2] & 4294901760u));
                    ac0_6 = fma_f32x2_rn_ftz(o0_12, o0_12, ac0_6);
                    ac1_6 = fma_f32x2_rn_ftz(o1_12, o1_12, ac1_6);
                    float2 e0_13 = make_float2(__uint_as_float(rw0_6[1] << 16), __uint_as_float(rw0_6[3] << 16));
                    float2 e1_13 = make_float2(__uint_as_float(rw1_6[1] << 16), __uint_as_float(rw1_6[3] << 16));
                    ac0_6 = fma_f32x2_rn_ftz(e0_13, e0_13, ac0_6);
                    ac1_6 = fma_f32x2_rn_ftz(e1_13, e1_13, ac1_6);
                    float2 o0_13 = make_float2(__uint_as_float(rw0_6[1] & 4294901760u), __uint_as_float(rw0_6[3] & 4294901760u));
                    float2 o1_13 = make_float2(__uint_as_float(rw1_6[1] & 4294901760u), __uint_as_float(rw1_6[3] & 4294901760u));
                    ac0_6 = fma_f32x2_rn_ftz(o0_13, o0_13, ac0_6);
                    ac1_6 = fma_f32x2_rn_ftz(o1_13, o1_13, ac1_6);
                    aa[12] = ac0_6.x + ac1_6.x;
                    aa[13] = ac0_6.y + ac1_6.y;
                    unsigned int rsw_7 = (unsigned int)((7 ^ m_lane % 8) << 4);
                    unsigned int rw0_7[4];
                    unsigned int rw1_7[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_7[(0) + 3]))
                        : "r"(mqb + (unsigned int)((m_lane + 32 * mr) * 128) + rsw_7));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_7[(0) + 3]))
                        : "r"(rb1 + rsw_7));
                    float2 ac0_7 = make_float2(0.0f, 0.0f);
                    float2 ac1_7 = make_float2(0.0f, 0.0f);
                    float2 e0_14 = make_float2(__uint_as_float(rw0_7[0] << 16), __uint_as_float(rw0_7[2] << 16));
                    float2 e1_14 = make_float2(__uint_as_float(rw1_7[0] << 16), __uint_as_float(rw1_7[2] << 16));
                    ac0_7 = fma_f32x2_rn_ftz(e0_14, e0_14, ac0_7);
                    ac1_7 = fma_f32x2_rn_ftz(e1_14, e1_14, ac1_7);
                    float2 o0_14 = make_float2(__uint_as_float(rw0_7[0] & 4294901760u), __uint_as_float(rw0_7[2] & 4294901760u));
                    float2 o1_14 = make_float2(__uint_as_float(rw1_7[0] & 4294901760u), __uint_as_float(rw1_7[2] & 4294901760u));
                    ac0_7 = fma_f32x2_rn_ftz(o0_14, o0_14, ac0_7);
                    ac1_7 = fma_f32x2_rn_ftz(o1_14, o1_14, ac1_7);
                    float2 e0_15 = make_float2(__uint_as_float(rw0_7[1] << 16), __uint_as_float(rw0_7[3] << 16));
                    float2 e1_15 = make_float2(__uint_as_float(rw1_7[1] << 16), __uint_as_float(rw1_7[3] << 16));
                    ac0_7 = fma_f32x2_rn_ftz(e0_15, e0_15, ac0_7);
                    ac1_7 = fma_f32x2_rn_ftz(e1_15, e1_15, ac1_7);
                    float2 o0_15 = make_float2(__uint_as_float(rw0_7[1] & 4294901760u), __uint_as_float(rw0_7[3] & 4294901760u));
                    float2 o1_15 = make_float2(__uint_as_float(rw1_7[1] & 4294901760u), __uint_as_float(rw1_7[3] & 4294901760u));
                    ac0_7 = fma_f32x2_rn_ftz(o0_15, o0_15, ac0_7);
                    ac1_7 = fma_f32x2_rn_ftz(o1_15, o1_15, ac1_7);
                    aa[14] = ac0_7.x + ac1_7.x;
                    aa[15] = ac0_7.y + ac1_7.y;
                    aa[0] = aa[0] + aa[8];
                    aa[1] = aa[1] + aa[9];
                    aa[2] = aa[2] + aa[10];
                    aa[3] = aa[3] + aa[11];
                    aa[4] = aa[4] + aa[12];
                    aa[5] = aa[5] + aa[13];
                    aa[6] = aa[6] + aa[14];
                    aa[7] = aa[7] + aa[15];
                    aa[0] = aa[0] + aa[4];
                    aa[1] = aa[1] + aa[5];
                    aa[2] = aa[2] + aa[6];
                    aa[3] = aa[3] + aa[7];
                    aa[0] = aa[0] + aa[2];
                    aa[1] = aa[1] + aa[3];
                    float rs = rsqrtf(aa[0] + aa[1] + 1e-06f);
                    float mr_s = rs;
                    if (mr == 0) {
                        mq0 = mr_s;
                    }
                    if (mr == 1) {
                        mq1 = mr_s;
                    }
                }
                mbarrier_arrive(st_free_r_addr + (mf_c) * 8);
                mf_c += 1;
                if (mf_c == 2) { mf_c = 0; _phase_st_full ^= 1; }
                mbarrier_wait(gates_free_addr + (mq_c) * 8, _phase_gates_free);
                s_gt[m_rq + m_lane] = mq0;
                s_gt[m_rq + 32 + m_lane] = mq1;
                mbarrier_arrive(gates_ready_addr + (mq_c) * 8);
                mq_c += 1;
                if (mq_c == 4) { mq_c = 0; _phase_gates_free ^= 1; }
                if (m_lane < m_n) {
                    rq_out[(long long)(m_tok0 + m_lane) * (long long)num_heads + (long long)m_hq] = mq0;
                }
                if (m_lane + 32 < m_n) {
                    rq_out[(long long)(m_tok0 + m_lane + 32) * (long long)num_heads + (long long)m_hq] = mq1;
                }
            }
        }
    }
    // ---- Role: gtr ----
    if (warp == 14) {
        { // gtr_main
            int bid_4 = blockIdx.x;
            int grid_3 = gridDim.x;
            int n_items_3 = num_chunks * num_heads;
            int cnt_3 = 0;
            if (bid_4 < n_items_3) {
                cnt_3 = (n_items_3 - bid_4 + grid_3 - 1) / grid_3;
            }
            int lane_2 = lane;
            int r_gpk = num_v_heads / num_heads;
            unsigned int rf_c = 0;
            unsigned int rg_c = 0;
            unsigned int _phase_st_full_1 = 0;
            unsigned int _phase_gates_free_1 = 1;
            #pragma unroll 1
            for (int it_3 = 0; it_3 < cnt_3; it_3++) {
                int item_2 = bid_4 + it_3 * grid_3;
                int ch_1 = item_2 / num_heads;
                int hq_2 = item_2 % num_heads;
                int tok0_2 = chunk_start[ch_1];
                int n_1 = chunk_len[ch_1];
                int p_off_1 = it_3 % 4 * 1416;
                unsigned int sbase = rf_c * 49152;
                unsigned int kb = s_st_addr + sbase;
                mbarrier_wait(st_full_addr + (rf_c) * 8, _phase_st_full_1);
                float rk0 = 0.0f;
                float rk1 = 0.0f;
                #pragma unroll 1
                for (int rr = 0; rr < 2; rr++) {
                    unsigned int rb1_1 = kb + (unsigned int)((lane_2 + 32 * rr) * 128) + 16384;
                    float aa_1[16];
                    unsigned int rsw_8 = (unsigned int)((0 ^ lane_2 % 8) << 4);
                    unsigned int rw0_8[4];
                    unsigned int rw1_8[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_8[(0) + 3]))
                        : "r"(kb + (unsigned int)((lane_2 + 32 * rr) * 128) + rsw_8));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_8[(0) + 3]))
                        : "r"(rb1_1 + rsw_8));
                    float2 ac0_8 = make_float2(0.0f, 0.0f);
                    float2 ac1_8 = make_float2(0.0f, 0.0f);
                    float2 e0_16 = make_float2(__uint_as_float(rw0_8[0] << 16), __uint_as_float(rw0_8[2] << 16));
                    float2 e1_16 = make_float2(__uint_as_float(rw1_8[0] << 16), __uint_as_float(rw1_8[2] << 16));
                    ac0_8 = fma_f32x2_rn_ftz(e0_16, e0_16, ac0_8);
                    ac1_8 = fma_f32x2_rn_ftz(e1_16, e1_16, ac1_8);
                    float2 o0_16 = make_float2(__uint_as_float(rw0_8[0] & 4294901760u), __uint_as_float(rw0_8[2] & 4294901760u));
                    float2 o1_16 = make_float2(__uint_as_float(rw1_8[0] & 4294901760u), __uint_as_float(rw1_8[2] & 4294901760u));
                    ac0_8 = fma_f32x2_rn_ftz(o0_16, o0_16, ac0_8);
                    ac1_8 = fma_f32x2_rn_ftz(o1_16, o1_16, ac1_8);
                    float2 e0_17 = make_float2(__uint_as_float(rw0_8[1] << 16), __uint_as_float(rw0_8[3] << 16));
                    float2 e1_17 = make_float2(__uint_as_float(rw1_8[1] << 16), __uint_as_float(rw1_8[3] << 16));
                    ac0_8 = fma_f32x2_rn_ftz(e0_17, e0_17, ac0_8);
                    ac1_8 = fma_f32x2_rn_ftz(e1_17, e1_17, ac1_8);
                    float2 o0_17 = make_float2(__uint_as_float(rw0_8[1] & 4294901760u), __uint_as_float(rw0_8[3] & 4294901760u));
                    float2 o1_17 = make_float2(__uint_as_float(rw1_8[1] & 4294901760u), __uint_as_float(rw1_8[3] & 4294901760u));
                    ac0_8 = fma_f32x2_rn_ftz(o0_17, o0_17, ac0_8);
                    ac1_8 = fma_f32x2_rn_ftz(o1_17, o1_17, ac1_8);
                    aa_1[0] = ac0_8.x + ac1_8.x;
                    aa_1[1] = ac0_8.y + ac1_8.y;
                    unsigned int rsw_9 = (unsigned int)((1 ^ lane_2 % 8) << 4);
                    unsigned int rw0_9[4];
                    unsigned int rw1_9[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_9[(0) + 3]))
                        : "r"(kb + (unsigned int)((lane_2 + 32 * rr) * 128) + rsw_9));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_9[(0) + 3]))
                        : "r"(rb1_1 + rsw_9));
                    float2 ac0_9 = make_float2(0.0f, 0.0f);
                    float2 ac1_9 = make_float2(0.0f, 0.0f);
                    float2 e0_18 = make_float2(__uint_as_float(rw0_9[0] << 16), __uint_as_float(rw0_9[2] << 16));
                    float2 e1_18 = make_float2(__uint_as_float(rw1_9[0] << 16), __uint_as_float(rw1_9[2] << 16));
                    ac0_9 = fma_f32x2_rn_ftz(e0_18, e0_18, ac0_9);
                    ac1_9 = fma_f32x2_rn_ftz(e1_18, e1_18, ac1_9);
                    float2 o0_18 = make_float2(__uint_as_float(rw0_9[0] & 4294901760u), __uint_as_float(rw0_9[2] & 4294901760u));
                    float2 o1_18 = make_float2(__uint_as_float(rw1_9[0] & 4294901760u), __uint_as_float(rw1_9[2] & 4294901760u));
                    ac0_9 = fma_f32x2_rn_ftz(o0_18, o0_18, ac0_9);
                    ac1_9 = fma_f32x2_rn_ftz(o1_18, o1_18, ac1_9);
                    float2 e0_19 = make_float2(__uint_as_float(rw0_9[1] << 16), __uint_as_float(rw0_9[3] << 16));
                    float2 e1_19 = make_float2(__uint_as_float(rw1_9[1] << 16), __uint_as_float(rw1_9[3] << 16));
                    ac0_9 = fma_f32x2_rn_ftz(e0_19, e0_19, ac0_9);
                    ac1_9 = fma_f32x2_rn_ftz(e1_19, e1_19, ac1_9);
                    float2 o0_19 = make_float2(__uint_as_float(rw0_9[1] & 4294901760u), __uint_as_float(rw0_9[3] & 4294901760u));
                    float2 o1_19 = make_float2(__uint_as_float(rw1_9[1] & 4294901760u), __uint_as_float(rw1_9[3] & 4294901760u));
                    ac0_9 = fma_f32x2_rn_ftz(o0_19, o0_19, ac0_9);
                    ac1_9 = fma_f32x2_rn_ftz(o1_19, o1_19, ac1_9);
                    aa_1[2] = ac0_9.x + ac1_9.x;
                    aa_1[3] = ac0_9.y + ac1_9.y;
                    unsigned int rsw_10 = (unsigned int)((2 ^ lane_2 % 8) << 4);
                    unsigned int rw0_10[4];
                    unsigned int rw1_10[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_10[(0) + 3]))
                        : "r"(kb + (unsigned int)((lane_2 + 32 * rr) * 128) + rsw_10));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_10[(0) + 3]))
                        : "r"(rb1_1 + rsw_10));
                    float2 ac0_10 = make_float2(0.0f, 0.0f);
                    float2 ac1_10 = make_float2(0.0f, 0.0f);
                    float2 e0_20 = make_float2(__uint_as_float(rw0_10[0] << 16), __uint_as_float(rw0_10[2] << 16));
                    float2 e1_20 = make_float2(__uint_as_float(rw1_10[0] << 16), __uint_as_float(rw1_10[2] << 16));
                    ac0_10 = fma_f32x2_rn_ftz(e0_20, e0_20, ac0_10);
                    ac1_10 = fma_f32x2_rn_ftz(e1_20, e1_20, ac1_10);
                    float2 o0_20 = make_float2(__uint_as_float(rw0_10[0] & 4294901760u), __uint_as_float(rw0_10[2] & 4294901760u));
                    float2 o1_20 = make_float2(__uint_as_float(rw1_10[0] & 4294901760u), __uint_as_float(rw1_10[2] & 4294901760u));
                    ac0_10 = fma_f32x2_rn_ftz(o0_20, o0_20, ac0_10);
                    ac1_10 = fma_f32x2_rn_ftz(o1_20, o1_20, ac1_10);
                    float2 e0_21 = make_float2(__uint_as_float(rw0_10[1] << 16), __uint_as_float(rw0_10[3] << 16));
                    float2 e1_21 = make_float2(__uint_as_float(rw1_10[1] << 16), __uint_as_float(rw1_10[3] << 16));
                    ac0_10 = fma_f32x2_rn_ftz(e0_21, e0_21, ac0_10);
                    ac1_10 = fma_f32x2_rn_ftz(e1_21, e1_21, ac1_10);
                    float2 o0_21 = make_float2(__uint_as_float(rw0_10[1] & 4294901760u), __uint_as_float(rw0_10[3] & 4294901760u));
                    float2 o1_21 = make_float2(__uint_as_float(rw1_10[1] & 4294901760u), __uint_as_float(rw1_10[3] & 4294901760u));
                    ac0_10 = fma_f32x2_rn_ftz(o0_21, o0_21, ac0_10);
                    ac1_10 = fma_f32x2_rn_ftz(o1_21, o1_21, ac1_10);
                    aa_1[4] = ac0_10.x + ac1_10.x;
                    aa_1[5] = ac0_10.y + ac1_10.y;
                    unsigned int rsw_11 = (unsigned int)((3 ^ lane_2 % 8) << 4);
                    unsigned int rw0_11[4];
                    unsigned int rw1_11[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_11[(0) + 3]))
                        : "r"(kb + (unsigned int)((lane_2 + 32 * rr) * 128) + rsw_11));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_11[(0) + 3]))
                        : "r"(rb1_1 + rsw_11));
                    float2 ac0_11 = make_float2(0.0f, 0.0f);
                    float2 ac1_11 = make_float2(0.0f, 0.0f);
                    float2 e0_22 = make_float2(__uint_as_float(rw0_11[0] << 16), __uint_as_float(rw0_11[2] << 16));
                    float2 e1_22 = make_float2(__uint_as_float(rw1_11[0] << 16), __uint_as_float(rw1_11[2] << 16));
                    ac0_11 = fma_f32x2_rn_ftz(e0_22, e0_22, ac0_11);
                    ac1_11 = fma_f32x2_rn_ftz(e1_22, e1_22, ac1_11);
                    float2 o0_22 = make_float2(__uint_as_float(rw0_11[0] & 4294901760u), __uint_as_float(rw0_11[2] & 4294901760u));
                    float2 o1_22 = make_float2(__uint_as_float(rw1_11[0] & 4294901760u), __uint_as_float(rw1_11[2] & 4294901760u));
                    ac0_11 = fma_f32x2_rn_ftz(o0_22, o0_22, ac0_11);
                    ac1_11 = fma_f32x2_rn_ftz(o1_22, o1_22, ac1_11);
                    float2 e0_23 = make_float2(__uint_as_float(rw0_11[1] << 16), __uint_as_float(rw0_11[3] << 16));
                    float2 e1_23 = make_float2(__uint_as_float(rw1_11[1] << 16), __uint_as_float(rw1_11[3] << 16));
                    ac0_11 = fma_f32x2_rn_ftz(e0_23, e0_23, ac0_11);
                    ac1_11 = fma_f32x2_rn_ftz(e1_23, e1_23, ac1_11);
                    float2 o0_23 = make_float2(__uint_as_float(rw0_11[1] & 4294901760u), __uint_as_float(rw0_11[3] & 4294901760u));
                    float2 o1_23 = make_float2(__uint_as_float(rw1_11[1] & 4294901760u), __uint_as_float(rw1_11[3] & 4294901760u));
                    ac0_11 = fma_f32x2_rn_ftz(o0_23, o0_23, ac0_11);
                    ac1_11 = fma_f32x2_rn_ftz(o1_23, o1_23, ac1_11);
                    aa_1[6] = ac0_11.x + ac1_11.x;
                    aa_1[7] = ac0_11.y + ac1_11.y;
                    unsigned int rsw_12 = (unsigned int)((4 ^ lane_2 % 8) << 4);
                    unsigned int rw0_12[4];
                    unsigned int rw1_12[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_12[(0) + 3]))
                        : "r"(kb + (unsigned int)((lane_2 + 32 * rr) * 128) + rsw_12));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_12[(0) + 3]))
                        : "r"(rb1_1 + rsw_12));
                    float2 ac0_12 = make_float2(0.0f, 0.0f);
                    float2 ac1_12 = make_float2(0.0f, 0.0f);
                    float2 e0_24 = make_float2(__uint_as_float(rw0_12[0] << 16), __uint_as_float(rw0_12[2] << 16));
                    float2 e1_24 = make_float2(__uint_as_float(rw1_12[0] << 16), __uint_as_float(rw1_12[2] << 16));
                    ac0_12 = fma_f32x2_rn_ftz(e0_24, e0_24, ac0_12);
                    ac1_12 = fma_f32x2_rn_ftz(e1_24, e1_24, ac1_12);
                    float2 o0_24 = make_float2(__uint_as_float(rw0_12[0] & 4294901760u), __uint_as_float(rw0_12[2] & 4294901760u));
                    float2 o1_24 = make_float2(__uint_as_float(rw1_12[0] & 4294901760u), __uint_as_float(rw1_12[2] & 4294901760u));
                    ac0_12 = fma_f32x2_rn_ftz(o0_24, o0_24, ac0_12);
                    ac1_12 = fma_f32x2_rn_ftz(o1_24, o1_24, ac1_12);
                    float2 e0_25 = make_float2(__uint_as_float(rw0_12[1] << 16), __uint_as_float(rw0_12[3] << 16));
                    float2 e1_25 = make_float2(__uint_as_float(rw1_12[1] << 16), __uint_as_float(rw1_12[3] << 16));
                    ac0_12 = fma_f32x2_rn_ftz(e0_25, e0_25, ac0_12);
                    ac1_12 = fma_f32x2_rn_ftz(e1_25, e1_25, ac1_12);
                    float2 o0_25 = make_float2(__uint_as_float(rw0_12[1] & 4294901760u), __uint_as_float(rw0_12[3] & 4294901760u));
                    float2 o1_25 = make_float2(__uint_as_float(rw1_12[1] & 4294901760u), __uint_as_float(rw1_12[3] & 4294901760u));
                    ac0_12 = fma_f32x2_rn_ftz(o0_25, o0_25, ac0_12);
                    ac1_12 = fma_f32x2_rn_ftz(o1_25, o1_25, ac1_12);
                    aa_1[8] = ac0_12.x + ac1_12.x;
                    aa_1[9] = ac0_12.y + ac1_12.y;
                    unsigned int rsw_13 = (unsigned int)((5 ^ lane_2 % 8) << 4);
                    unsigned int rw0_13[4];
                    unsigned int rw1_13[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_13[(0) + 3]))
                        : "r"(kb + (unsigned int)((lane_2 + 32 * rr) * 128) + rsw_13));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_13[(0) + 3]))
                        : "r"(rb1_1 + rsw_13));
                    float2 ac0_13 = make_float2(0.0f, 0.0f);
                    float2 ac1_13 = make_float2(0.0f, 0.0f);
                    float2 e0_26 = make_float2(__uint_as_float(rw0_13[0] << 16), __uint_as_float(rw0_13[2] << 16));
                    float2 e1_26 = make_float2(__uint_as_float(rw1_13[0] << 16), __uint_as_float(rw1_13[2] << 16));
                    ac0_13 = fma_f32x2_rn_ftz(e0_26, e0_26, ac0_13);
                    ac1_13 = fma_f32x2_rn_ftz(e1_26, e1_26, ac1_13);
                    float2 o0_26 = make_float2(__uint_as_float(rw0_13[0] & 4294901760u), __uint_as_float(rw0_13[2] & 4294901760u));
                    float2 o1_26 = make_float2(__uint_as_float(rw1_13[0] & 4294901760u), __uint_as_float(rw1_13[2] & 4294901760u));
                    ac0_13 = fma_f32x2_rn_ftz(o0_26, o0_26, ac0_13);
                    ac1_13 = fma_f32x2_rn_ftz(o1_26, o1_26, ac1_13);
                    float2 e0_27 = make_float2(__uint_as_float(rw0_13[1] << 16), __uint_as_float(rw0_13[3] << 16));
                    float2 e1_27 = make_float2(__uint_as_float(rw1_13[1] << 16), __uint_as_float(rw1_13[3] << 16));
                    ac0_13 = fma_f32x2_rn_ftz(e0_27, e0_27, ac0_13);
                    ac1_13 = fma_f32x2_rn_ftz(e1_27, e1_27, ac1_13);
                    float2 o0_27 = make_float2(__uint_as_float(rw0_13[1] & 4294901760u), __uint_as_float(rw0_13[3] & 4294901760u));
                    float2 o1_27 = make_float2(__uint_as_float(rw1_13[1] & 4294901760u), __uint_as_float(rw1_13[3] & 4294901760u));
                    ac0_13 = fma_f32x2_rn_ftz(o0_27, o0_27, ac0_13);
                    ac1_13 = fma_f32x2_rn_ftz(o1_27, o1_27, ac1_13);
                    aa_1[10] = ac0_13.x + ac1_13.x;
                    aa_1[11] = ac0_13.y + ac1_13.y;
                    unsigned int rsw_14 = (unsigned int)((6 ^ lane_2 % 8) << 4);
                    unsigned int rw0_14[4];
                    unsigned int rw1_14[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_14[(0) + 3]))
                        : "r"(kb + (unsigned int)((lane_2 + 32 * rr) * 128) + rsw_14));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_14[(0) + 3]))
                        : "r"(rb1_1 + rsw_14));
                    float2 ac0_14 = make_float2(0.0f, 0.0f);
                    float2 ac1_14 = make_float2(0.0f, 0.0f);
                    float2 e0_28 = make_float2(__uint_as_float(rw0_14[0] << 16), __uint_as_float(rw0_14[2] << 16));
                    float2 e1_28 = make_float2(__uint_as_float(rw1_14[0] << 16), __uint_as_float(rw1_14[2] << 16));
                    ac0_14 = fma_f32x2_rn_ftz(e0_28, e0_28, ac0_14);
                    ac1_14 = fma_f32x2_rn_ftz(e1_28, e1_28, ac1_14);
                    float2 o0_28 = make_float2(__uint_as_float(rw0_14[0] & 4294901760u), __uint_as_float(rw0_14[2] & 4294901760u));
                    float2 o1_28 = make_float2(__uint_as_float(rw1_14[0] & 4294901760u), __uint_as_float(rw1_14[2] & 4294901760u));
                    ac0_14 = fma_f32x2_rn_ftz(o0_28, o0_28, ac0_14);
                    ac1_14 = fma_f32x2_rn_ftz(o1_28, o1_28, ac1_14);
                    float2 e0_29 = make_float2(__uint_as_float(rw0_14[1] << 16), __uint_as_float(rw0_14[3] << 16));
                    float2 e1_29 = make_float2(__uint_as_float(rw1_14[1] << 16), __uint_as_float(rw1_14[3] << 16));
                    ac0_14 = fma_f32x2_rn_ftz(e0_29, e0_29, ac0_14);
                    ac1_14 = fma_f32x2_rn_ftz(e1_29, e1_29, ac1_14);
                    float2 o0_29 = make_float2(__uint_as_float(rw0_14[1] & 4294901760u), __uint_as_float(rw0_14[3] & 4294901760u));
                    float2 o1_29 = make_float2(__uint_as_float(rw1_14[1] & 4294901760u), __uint_as_float(rw1_14[3] & 4294901760u));
                    ac0_14 = fma_f32x2_rn_ftz(o0_29, o0_29, ac0_14);
                    ac1_14 = fma_f32x2_rn_ftz(o1_29, o1_29, ac1_14);
                    aa_1[12] = ac0_14.x + ac1_14.x;
                    aa_1[13] = ac0_14.y + ac1_14.y;
                    unsigned int rsw_15 = (unsigned int)((7 ^ lane_2 % 8) << 4);
                    unsigned int rw0_15[4];
                    unsigned int rw1_15[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw0_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw0_15[(0) + 3]))
                        : "r"(kb + (unsigned int)((lane_2 + 32 * rr) * 128) + rsw_15));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rw1_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rw1_15[(0) + 3]))
                        : "r"(rb1_1 + rsw_15));
                    float2 ac0_15 = make_float2(0.0f, 0.0f);
                    float2 ac1_15 = make_float2(0.0f, 0.0f);
                    float2 e0_30 = make_float2(__uint_as_float(rw0_15[0] << 16), __uint_as_float(rw0_15[2] << 16));
                    float2 e1_30 = make_float2(__uint_as_float(rw1_15[0] << 16), __uint_as_float(rw1_15[2] << 16));
                    ac0_15 = fma_f32x2_rn_ftz(e0_30, e0_30, ac0_15);
                    ac1_15 = fma_f32x2_rn_ftz(e1_30, e1_30, ac1_15);
                    float2 o0_30 = make_float2(__uint_as_float(rw0_15[0] & 4294901760u), __uint_as_float(rw0_15[2] & 4294901760u));
                    float2 o1_30 = make_float2(__uint_as_float(rw1_15[0] & 4294901760u), __uint_as_float(rw1_15[2] & 4294901760u));
                    ac0_15 = fma_f32x2_rn_ftz(o0_30, o0_30, ac0_15);
                    ac1_15 = fma_f32x2_rn_ftz(o1_30, o1_30, ac1_15);
                    float2 e0_31 = make_float2(__uint_as_float(rw0_15[1] << 16), __uint_as_float(rw0_15[3] << 16));
                    float2 e1_31 = make_float2(__uint_as_float(rw1_15[1] << 16), __uint_as_float(rw1_15[3] << 16));
                    ac0_15 = fma_f32x2_rn_ftz(e0_31, e0_31, ac0_15);
                    ac1_15 = fma_f32x2_rn_ftz(e1_31, e1_31, ac1_15);
                    float2 o0_31 = make_float2(__uint_as_float(rw0_15[1] & 4294901760u), __uint_as_float(rw0_15[3] & 4294901760u));
                    float2 o1_31 = make_float2(__uint_as_float(rw1_15[1] & 4294901760u), __uint_as_float(rw1_15[3] & 4294901760u));
                    ac0_15 = fma_f32x2_rn_ftz(o0_31, o0_31, ac0_15);
                    ac1_15 = fma_f32x2_rn_ftz(o1_31, o1_31, ac1_15);
                    aa_1[14] = ac0_15.x + ac1_15.x;
                    aa_1[15] = ac0_15.y + ac1_15.y;
                    aa_1[0] = aa_1[0] + aa_1[8];
                    aa_1[1] = aa_1[1] + aa_1[9];
                    aa_1[2] = aa_1[2] + aa_1[10];
                    aa_1[3] = aa_1[3] + aa_1[11];
                    aa_1[4] = aa_1[4] + aa_1[12];
                    aa_1[5] = aa_1[5] + aa_1[13];
                    aa_1[6] = aa_1[6] + aa_1[14];
                    aa_1[7] = aa_1[7] + aa_1[15];
                    aa_1[0] = aa_1[0] + aa_1[4];
                    aa_1[1] = aa_1[1] + aa_1[5];
                    aa_1[2] = aa_1[2] + aa_1[6];
                    aa_1[3] = aa_1[3] + aa_1[7];
                    aa_1[0] = aa_1[0] + aa_1[2];
                    aa_1[1] = aa_1[1] + aa_1[3];
                    float rs_1 = rsqrtf(aa_1[0] + aa_1[1] + 1e-06f);
                    float rr_s = rs_1;
                    if (rr == 0) {
                        rk0 = rr_s;
                    }
                    if (rr == 1) {
                        rk1 = rr_s;
                    }
                }
                mbarrier_arrive(st_free_r_addr + (rf_c) * 8);
                rf_c += 1;
                if (rf_c == 2) { rf_c = 0; _phase_st_full_1 ^= 1; }
                if (lane_2 < n_1) {
                    rk_out[(long long)(tok0_2 + lane_2) * (long long)num_heads + (long long)hq_2] = rk0;
                }
                if (lane_2 + 32 < n_1) {
                    rk_out[(long long)(tok0_2 + lane_2 + 32) * (long long)num_heads + (long long)hq_2] = rk1;
                }
                float t0 = ((lane_2 < n_1) ? rk0 : 0.0f);
                float t1 = ((lane_2 + 32 < n_1) ? rk1 : 0.0f);
                mbarrier_wait(gates_free_addr + (rg_c) * 8, _phase_gates_free_1);
                s_gt[p_off_1 + lane_2] = t0;
                s_gt[p_off_1 + 32 + lane_2] = t1;
                asm volatile("barrier.sync 4, 64;" ::: "memory");
                if (0 < r_gpk) {
                    int r_p_0 = p_off_1 + 72 + lane_2;
                    float r_rk_0 = ((1) ? t0 : t1);
                    float r_bb_0 = s_gt[r_p_0 + 64];
                    s_gt[r_p_0 + 256] = r_rk_0 * s_gt[r_p_0 + 192];
                    s_gt[r_p_0 + 320] = r_bb_0 * r_rk_0;
                    int r_p_1 = p_off_1 + 72 + lane_2 + 32;
                    float r_rk_1 = ((0) ? t0 : t1);
                    float r_bb_1 = s_gt[r_p_1 + 64];
                    s_gt[r_p_1 + 256] = r_rk_1 * s_gt[r_p_1 + 192];
                    s_gt[r_p_1 + 320] = r_bb_1 * r_rk_1;
                }
                if (1 < r_gpk) {
                    int r_p_2 = p_off_1 + 72 + 448 + lane_2;
                    float r_rk_2 = ((1) ? t0 : t1);
                    float r_bb_2 = s_gt[r_p_2 + 64];
                    s_gt[r_p_2 + 256] = r_rk_2 * s_gt[r_p_2 + 192];
                    s_gt[r_p_2 + 320] = r_bb_2 * r_rk_2;
                    int r_p_3 = p_off_1 + 72 + 448 + lane_2 + 32;
                    float r_rk_3 = ((0) ? t0 : t1);
                    float r_bb_3 = s_gt[r_p_3 + 64];
                    s_gt[r_p_3 + 256] = r_rk_3 * s_gt[r_p_3 + 192];
                    s_gt[r_p_3 + 320] = r_bb_3 * r_rk_3;
                }
                if (2 < r_gpk) {
                    int r_p_4 = p_off_1 + 72 + 896 + lane_2;
                    float r_rk_4 = ((1) ? t0 : t1);
                    float r_bb_4 = s_gt[r_p_4 + 64];
                    s_gt[r_p_4 + 256] = r_rk_4 * s_gt[r_p_4 + 192];
                    s_gt[r_p_4 + 320] = r_bb_4 * r_rk_4;
                    int r_p_5 = p_off_1 + 72 + 896 + lane_2 + 32;
                    float r_rk_5 = ((0) ? t0 : t1);
                    float r_bb_5 = s_gt[r_p_5 + 64];
                    s_gt[r_p_5 + 256] = r_rk_5 * s_gt[r_p_5 + 192];
                    s_gt[r_p_5 + 320] = r_bb_5 * r_rk_5;
                }
                mbarrier_arrive(gates_ready_addr + (rg_c) * 8);
                rg_c += 1;
                if (rg_c == 4) { rg_c = 0; _phase_gates_free_1 ^= 1; }
            }
        }
    }
    // ---- Role: gtv ----
    if (warp == 15) {
        { // gtv_main
            int bid_5 = blockIdx.x;
            int grid_4 = gridDim.x;
            int n_items_4 = num_chunks * num_heads;
            int cnt_4 = 0;
            if (bid_5 < n_items_4) {
                cnt_4 = (n_items_4 - bid_5 + grid_4 - 1) / grid_4;
            }
            int lane_3 = lane;
            int gpk_1 = num_v_heads / num_heads;
            unsigned int vg_c = 0;
            int ct_tok = 0;
            int ct_n = 0;
            if (lane_3 < cnt_4) {
                ct_tok = chunk_start[(bid_5 + lane_3 * grid_4) / num_heads];
                ct_n = chunk_len[(bid_5 + lane_3 * grid_4) / num_heads];
            }
            float na[6];
            float nb[6];
            na[0] = 0.0f;
            nb[0] = 0.0f;
            na[1] = 0.0f;
            nb[1] = 0.0f;
            na[2] = 0.0f;
            nb[2] = 0.0f;
            na[3] = 0.0f;
            nb[3] = 0.0f;
            na[4] = 0.0f;
            nb[4] = 0.0f;
            na[5] = 0.0f;
            nb[5] = 0.0f;
            int nx_tok = __shfl_sync(0xFFFFFFFF, ct_tok, 0);
            int nx_n = __shfl_sync(0xFFFFFFFF, ct_n, 0);
            if (cnt_4 > 0) {
                if (0 < gpk_1) {
                    long long p_r0_0 = (long long)(nx_tok + lane_3) * (long long)num_v_heads + (long long)(bid_5 % num_heads * gpk_1);
                    long long p_r1_0 = (long long)(nx_tok + lane_3 + 32) * (long long)num_v_heads + (long long)(bid_5 % num_heads * gpk_1);
                    if (lane_3 < nx_n) {
                        na[0] = glog[p_r0_0];
                        nb[0] = beta[p_r0_0];
                    }
                    if (lane_3 + 32 < nx_n) {
                        na[1] = glog[p_r1_0];
                        nb[1] = beta[p_r1_0];
                    }
                }
                if (1 < gpk_1) {
                    long long p_r0_1 = (long long)(nx_tok + lane_3) * (long long)num_v_heads + (long long)(bid_5 % num_heads * gpk_1 + 1);
                    long long p_r1_1 = (long long)(nx_tok + lane_3 + 32) * (long long)num_v_heads + (long long)(bid_5 % num_heads * gpk_1 + 1);
                    if (lane_3 < nx_n) {
                        na[2] = glog[p_r0_1];
                        nb[2] = beta[p_r0_1];
                    }
                    if (lane_3 + 32 < nx_n) {
                        na[3] = glog[p_r1_1];
                        nb[3] = beta[p_r1_1];
                    }
                }
                if (2 < gpk_1) {
                    long long p_r0_2 = (long long)(nx_tok + lane_3) * (long long)num_v_heads + (long long)(bid_5 % num_heads * gpk_1 + 2);
                    long long p_r1_2 = (long long)(nx_tok + lane_3 + 32) * (long long)num_v_heads + (long long)(bid_5 % num_heads * gpk_1 + 2);
                    if (lane_3 < nx_n) {
                        na[4] = glog[p_r0_2];
                        nb[4] = beta[p_r0_2];
                    }
                    if (lane_3 + 32 < nx_n) {
                        na[5] = glog[p_r1_2];
                        nb[5] = beta[p_r1_2];
                    }
                }
            }
            float va[6];
            float vb[6];
            unsigned int _phase_gates_free_2 = 1;
            #pragma unroll 1
            for (int it_4 = 0; it_4 < cnt_4; it_4++) {
                int item_3 = bid_5 + it_4 * grid_4;
                int hq_3 = item_3 % num_heads;
                int tok0_3 = nx_tok;
                int n_2 = nx_n;
                int p_off_2 = it_4 % 4 * 1416;
                va[0] = na[0];
                vb[0] = nb[0];
                na[0] = 0.0f;
                nb[0] = 0.0f;
                va[1] = na[1];
                vb[1] = nb[1];
                na[1] = 0.0f;
                nb[1] = 0.0f;
                va[2] = na[2];
                vb[2] = nb[2];
                na[2] = 0.0f;
                nb[2] = 0.0f;
                va[3] = na[3];
                vb[3] = nb[3];
                na[3] = 0.0f;
                nb[3] = 0.0f;
                va[4] = na[4];
                vb[4] = nb[4];
                na[4] = 0.0f;
                nb[4] = 0.0f;
                va[5] = na[5];
                vb[5] = nb[5];
                na[5] = 0.0f;
                nb[5] = 0.0f;
                int j = it_4 + 1;
                if (j < cnt_4) {
                    if (j % 32 == 0) {
                        ct_tok = 0;
                        ct_n = 0;
                        if (j + lane_3 < cnt_4) {
                            ct_tok = chunk_start[(bid_5 + (j + lane_3) * grid_4) / num_heads];
                            ct_n = chunk_len[(bid_5 + (j + lane_3) * grid_4) / num_heads];
                        }
                    }
                    int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, ct_tok, j % 32);
                    nx_tok = _shfl_sync_0;
                    int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, ct_n, j % 32);
                    nx_n = _shfl_sync_1;
                    if (0 < gpk_1) {
                        long long p_r0_3 = (long long)(nx_tok + lane_3) * (long long)num_v_heads + (long long)((bid_5 + j * grid_4) % num_heads * gpk_1);
                        long long p_r1_3 = (long long)(nx_tok + lane_3 + 32) * (long long)num_v_heads + (long long)((bid_5 + j * grid_4) % num_heads * gpk_1);
                        if (lane_3 < nx_n) {
                            na[0] = glog[p_r0_3];
                            nb[0] = beta[p_r0_3];
                        }
                        if (lane_3 + 32 < nx_n) {
                            na[1] = glog[p_r1_3];
                            nb[1] = beta[p_r1_3];
                        }
                    }
                    if (1 < gpk_1) {
                        long long p_r0_4 = (long long)(nx_tok + lane_3) * (long long)num_v_heads + (long long)((bid_5 + j * grid_4) % num_heads * gpk_1 + 1);
                        long long p_r1_4 = (long long)(nx_tok + lane_3 + 32) * (long long)num_v_heads + (long long)((bid_5 + j * grid_4) % num_heads * gpk_1 + 1);
                        if (lane_3 < nx_n) {
                            na[2] = glog[p_r0_4];
                            nb[2] = beta[p_r0_4];
                        }
                        if (lane_3 + 32 < nx_n) {
                            na[3] = glog[p_r1_4];
                            nb[3] = beta[p_r1_4];
                        }
                    }
                    if (2 < gpk_1) {
                        long long p_r0_5 = (long long)(nx_tok + lane_3) * (long long)num_v_heads + (long long)((bid_5 + j * grid_4) % num_heads * gpk_1 + 2);
                        long long p_r1_5 = (long long)(nx_tok + lane_3 + 32) * (long long)num_v_heads + (long long)((bid_5 + j * grid_4) % num_heads * gpk_1 + 2);
                        if (lane_3 < nx_n) {
                            na[4] = glog[p_r0_5];
                            nb[4] = beta[p_r0_5];
                        }
                        if (lane_3 + 32 < nx_n) {
                            na[5] = glog[p_r1_5];
                            nb[5] = beta[p_r1_5];
                        }
                    }
                }
                mbarrier_wait(gates_free_addr + (vg_c) * 8, _phase_gates_free_2);
                if (0 < gpk_1) {
                    int hv_0 = hq_3 * gpk_1;
                    int hb_0 = p_off_2 + 72;
                    float a0_0 = va[0];
                    float a1_0 = va[1];
                    float u0_0 = __shfl_up_sync(0xFFFFFFFF, a0_0, 1, 32);
                    float u1_0 = __shfl_up_sync(0xFFFFFFFF, a1_0, 1, 32);
                    if (lane_3 >= 1) {
                        a0_0 = a0_0 + u0_0;
                        a1_0 = a1_0 + u1_0;
                    }
                    float u0_1 = __shfl_up_sync(0xFFFFFFFF, a0_0, 2, 32);
                    float u1_1 = __shfl_up_sync(0xFFFFFFFF, a1_0, 2, 32);
                    if (lane_3 >= 2) {
                        a0_0 = a0_0 + u0_1;
                        a1_0 = a1_0 + u1_1;
                    }
                    float u0_2 = __shfl_up_sync(0xFFFFFFFF, a0_0, 4, 32);
                    float u1_2 = __shfl_up_sync(0xFFFFFFFF, a1_0, 4, 32);
                    if (lane_3 >= 4) {
                        a0_0 = a0_0 + u0_2;
                        a1_0 = a1_0 + u1_2;
                    }
                    float u0_3 = __shfl_up_sync(0xFFFFFFFF, a0_0, 8, 32);
                    float u1_3 = __shfl_up_sync(0xFFFFFFFF, a1_0, 8, 32);
                    if (lane_3 >= 8) {
                        a0_0 = a0_0 + u0_3;
                        a1_0 = a1_0 + u1_3;
                    }
                    float u0_4 = __shfl_up_sync(0xFFFFFFFF, a0_0, 16, 32);
                    float u1_4 = __shfl_up_sync(0xFFFFFFFF, a1_0, 16, 32);
                    if (lane_3 >= 16) {
                        a0_0 = a0_0 + u0_4;
                        a1_0 = a1_0 + u1_4;
                    }
                    float tot0_0 = __shfl_sync(0xFFFFFFFF, a0_0, 31);
                    a1_0 = a1_0 + tot0_0;
                    a0_0 = a0_0 * 1.442695f;
                    a1_0 = a1_0 * 1.442695f;
                    if (lane_3 < n_2) {
                        gcs_out[(long long)(tok0_3 + lane_3) * (long long)num_v_heads + (long long)hv_0] = a0_0;
                    }
                    if (lane_3 + 32 < n_2) {
                        gcs_out[(long long)(tok0_3 + lane_3 + 32) * (long long)num_v_heads + (long long)hv_0] = a1_0;
                    }
                    float f0_0 = __shfl_sync(0xFFFFFFFF, a0_0, 15);
                    float f1_0 = __shfl_sync(0xFFFFFFFF, a0_0, 31);
                    float f2_0 = __shfl_sync(0xFFFFFFFF, a1_0, 15);
                    float f3_0 = __shfl_sync(0xFFFFFFFF, a1_0, 31);
                    float z0_0 = __shfl_sync(0xFFFFFFFF, a0_0, 0);
                    float z1_0 = __shfl_sync(0xFFFFFFFF, a0_0, 16);
                    float z2_0 = __shfl_sync(0xFFFFFFFF, a1_0, 0);
                    float z3_0 = __shfl_sync(0xFFFFFFFF, a1_0, 16);
                    float rf_lo_0 = ((lane_3 < 16) ? f0_0 : f1_0);
                    float rf_hi_0 = ((lane_3 < 16) ? f2_0 : f3_0);
                    s_gt[hb_0 + lane_3] = a0_0;
                    s_gt[hb_0 + 32 + lane_3] = a1_0;
                    s_gt[hb_0 + 64 + lane_3] = vb[0];
                    s_gt[hb_0 + 64 + 32 + lane_3] = vb[1];
                    float _exp2_2 = approx_exp2(a0_0);
                    s_gt[hb_0 + 128 + lane_3] = _exp2_2;
                    float _exp2_3 = approx_exp2(a1_0);
                    s_gt[hb_0 + 128 + 32 + lane_3] = _exp2_3;
                    float _exp2_4 = approx_exp2(rf_lo_0 - a0_0);
                    s_gt[hb_0 + 192 + lane_3] = _exp2_4;
                    float _exp2_5 = approx_exp2(rf_hi_0 - a1_0);
                    s_gt[hb_0 + 192 + 32 + lane_3] = _exp2_5;
                    float fl_0 = 0.0f;
                    if (z0_0 - f0_0 > 64.0f) {
                        fl_0 = 1.0f;
                    }
                    if (z1_0 - f1_0 > 64.0f) {
                        fl_0 = 1.0f;
                    }
                    if (z2_0 - f2_0 > 64.0f) {
                        fl_0 = 1.0f;
                    }
                    if (z3_0 - f3_0 > 64.0f) {
                        fl_0 = 1.0f;
                    }
                    if (lane_3 == 0) {
                        s_gt[p_off_2 + 64] = fl_0;
                    }
                }
                if (1 < gpk_1) {
                    int hv_1 = hq_3 * gpk_1 + 1;
                    int hb_1 = p_off_2 + 72 + 448;
                    float a0_1 = va[2];
                    float a1_1 = va[3];
                    float u0_5 = __shfl_up_sync(0xFFFFFFFF, a0_1, 1, 32);
                    float u1_5 = __shfl_up_sync(0xFFFFFFFF, a1_1, 1, 32);
                    if (lane_3 >= 1) {
                        a0_1 = a0_1 + u0_5;
                        a1_1 = a1_1 + u1_5;
                    }
                    float u0_6 = __shfl_up_sync(0xFFFFFFFF, a0_1, 2, 32);
                    float u1_6 = __shfl_up_sync(0xFFFFFFFF, a1_1, 2, 32);
                    if (lane_3 >= 2) {
                        a0_1 = a0_1 + u0_6;
                        a1_1 = a1_1 + u1_6;
                    }
                    float u0_7 = __shfl_up_sync(0xFFFFFFFF, a0_1, 4, 32);
                    float u1_7 = __shfl_up_sync(0xFFFFFFFF, a1_1, 4, 32);
                    if (lane_3 >= 4) {
                        a0_1 = a0_1 + u0_7;
                        a1_1 = a1_1 + u1_7;
                    }
                    float u0_8 = __shfl_up_sync(0xFFFFFFFF, a0_1, 8, 32);
                    float u1_8 = __shfl_up_sync(0xFFFFFFFF, a1_1, 8, 32);
                    if (lane_3 >= 8) {
                        a0_1 = a0_1 + u0_8;
                        a1_1 = a1_1 + u1_8;
                    }
                    float u0_9 = __shfl_up_sync(0xFFFFFFFF, a0_1, 16, 32);
                    float u1_9 = __shfl_up_sync(0xFFFFFFFF, a1_1, 16, 32);
                    if (lane_3 >= 16) {
                        a0_1 = a0_1 + u0_9;
                        a1_1 = a1_1 + u1_9;
                    }
                    float tot0_1 = __shfl_sync(0xFFFFFFFF, a0_1, 31);
                    a1_1 = a1_1 + tot0_1;
                    a0_1 = a0_1 * 1.442695f;
                    a1_1 = a1_1 * 1.442695f;
                    if (lane_3 < n_2) {
                        gcs_out[(long long)(tok0_3 + lane_3) * (long long)num_v_heads + (long long)hv_1] = a0_1;
                    }
                    if (lane_3 + 32 < n_2) {
                        gcs_out[(long long)(tok0_3 + lane_3 + 32) * (long long)num_v_heads + (long long)hv_1] = a1_1;
                    }
                    float f0_1 = __shfl_sync(0xFFFFFFFF, a0_1, 15);
                    float f1_1 = __shfl_sync(0xFFFFFFFF, a0_1, 31);
                    float f2_1 = __shfl_sync(0xFFFFFFFF, a1_1, 15);
                    float f3_1 = __shfl_sync(0xFFFFFFFF, a1_1, 31);
                    float z0_1 = __shfl_sync(0xFFFFFFFF, a0_1, 0);
                    float z1_1 = __shfl_sync(0xFFFFFFFF, a0_1, 16);
                    float z2_1 = __shfl_sync(0xFFFFFFFF, a1_1, 0);
                    float z3_1 = __shfl_sync(0xFFFFFFFF, a1_1, 16);
                    float rf_lo_1 = ((lane_3 < 16) ? f0_1 : f1_1);
                    float rf_hi_1 = ((lane_3 < 16) ? f2_1 : f3_1);
                    s_gt[hb_1 + lane_3] = a0_1;
                    s_gt[hb_1 + 32 + lane_3] = a1_1;
                    s_gt[hb_1 + 64 + lane_3] = vb[2];
                    s_gt[hb_1 + 64 + 32 + lane_3] = vb[3];
                    float _exp2_6 = approx_exp2(a0_1);
                    s_gt[hb_1 + 128 + lane_3] = _exp2_6;
                    float _exp2_7 = approx_exp2(a1_1);
                    s_gt[hb_1 + 128 + 32 + lane_3] = _exp2_7;
                    float _exp2_8_1 = approx_exp2(rf_lo_1 - a0_1);
                    s_gt[hb_1 + 192 + lane_3] = _exp2_8_1;
                    float _exp2_9_1 = approx_exp2(rf_hi_1 - a1_1);
                    s_gt[hb_1 + 192 + 32 + lane_3] = _exp2_9_1;
                    float fl_1 = 0.0f;
                    if (z0_1 - f0_1 > 64.0f) {
                        fl_1 = 1.0f;
                    }
                    if (z1_1 - f1_1 > 64.0f) {
                        fl_1 = 1.0f;
                    }
                    if (z2_1 - f2_1 > 64.0f) {
                        fl_1 = 1.0f;
                    }
                    if (z3_1 - f3_1 > 64.0f) {
                        fl_1 = 1.0f;
                    }
                    if (lane_3 == 0) {
                        s_gt[p_off_2 + 64 + 1] = fl_1;
                    }
                }
                if (2 < gpk_1) {
                    int hv_2 = hq_3 * gpk_1 + 2;
                    int hb_2 = p_off_2 + 72 + 896;
                    float a0_2 = va[4];
                    float a1_2 = va[5];
                    float u0_10 = __shfl_up_sync(0xFFFFFFFF, a0_2, 1, 32);
                    float u1_10 = __shfl_up_sync(0xFFFFFFFF, a1_2, 1, 32);
                    if (lane_3 >= 1) {
                        a0_2 = a0_2 + u0_10;
                        a1_2 = a1_2 + u1_10;
                    }
                    float u0_11 = __shfl_up_sync(0xFFFFFFFF, a0_2, 2, 32);
                    float u1_11 = __shfl_up_sync(0xFFFFFFFF, a1_2, 2, 32);
                    if (lane_3 >= 2) {
                        a0_2 = a0_2 + u0_11;
                        a1_2 = a1_2 + u1_11;
                    }
                    float u0_12 = __shfl_up_sync(0xFFFFFFFF, a0_2, 4, 32);
                    float u1_12 = __shfl_up_sync(0xFFFFFFFF, a1_2, 4, 32);
                    if (lane_3 >= 4) {
                        a0_2 = a0_2 + u0_12;
                        a1_2 = a1_2 + u1_12;
                    }
                    float u0_13 = __shfl_up_sync(0xFFFFFFFF, a0_2, 8, 32);
                    float u1_13 = __shfl_up_sync(0xFFFFFFFF, a1_2, 8, 32);
                    if (lane_3 >= 8) {
                        a0_2 = a0_2 + u0_13;
                        a1_2 = a1_2 + u1_13;
                    }
                    float u0_14 = __shfl_up_sync(0xFFFFFFFF, a0_2, 16, 32);
                    float u1_14 = __shfl_up_sync(0xFFFFFFFF, a1_2, 16, 32);
                    if (lane_3 >= 16) {
                        a0_2 = a0_2 + u0_14;
                        a1_2 = a1_2 + u1_14;
                    }
                    float tot0_2 = __shfl_sync(0xFFFFFFFF, a0_2, 31);
                    a1_2 = a1_2 + tot0_2;
                    a0_2 = a0_2 * 1.442695f;
                    a1_2 = a1_2 * 1.442695f;
                    if (lane_3 < n_2) {
                        gcs_out[(long long)(tok0_3 + lane_3) * (long long)num_v_heads + (long long)hv_2] = a0_2;
                    }
                    if (lane_3 + 32 < n_2) {
                        gcs_out[(long long)(tok0_3 + lane_3 + 32) * (long long)num_v_heads + (long long)hv_2] = a1_2;
                    }
                    float f0_2 = __shfl_sync(0xFFFFFFFF, a0_2, 15);
                    float f1_2 = __shfl_sync(0xFFFFFFFF, a0_2, 31);
                    float f2_2 = __shfl_sync(0xFFFFFFFF, a1_2, 15);
                    float f3_2 = __shfl_sync(0xFFFFFFFF, a1_2, 31);
                    float z0_2 = __shfl_sync(0xFFFFFFFF, a0_2, 0);
                    float z1_2 = __shfl_sync(0xFFFFFFFF, a0_2, 16);
                    float z2_2 = __shfl_sync(0xFFFFFFFF, a1_2, 0);
                    float z3_2 = __shfl_sync(0xFFFFFFFF, a1_2, 16);
                    float rf_lo_2 = ((lane_3 < 16) ? f0_2 : f1_2);
                    float rf_hi_2 = ((lane_3 < 16) ? f2_2 : f3_2);
                    s_gt[hb_2 + lane_3] = a0_2;
                    s_gt[hb_2 + 32 + lane_3] = a1_2;
                    s_gt[hb_2 + 64 + lane_3] = vb[4];
                    s_gt[hb_2 + 64 + 32 + lane_3] = vb[5];
                    float _exp2_10_1 = approx_exp2(a0_2);
                    s_gt[hb_2 + 128 + lane_3] = _exp2_10_1;
                    float _exp2_11_1 = approx_exp2(a1_2);
                    s_gt[hb_2 + 128 + 32 + lane_3] = _exp2_11_1;
                    float _exp2_12_1 = approx_exp2(rf_lo_2 - a0_2);
                    s_gt[hb_2 + 192 + lane_3] = _exp2_12_1;
                    float _exp2_13_1 = approx_exp2(rf_hi_2 - a1_2);
                    s_gt[hb_2 + 192 + 32 + lane_3] = _exp2_13_1;
                    float fl_2 = 0.0f;
                    if (z0_2 - f0_2 > 64.0f) {
                        fl_2 = 1.0f;
                    }
                    if (z1_2 - f1_2 > 64.0f) {
                        fl_2 = 1.0f;
                    }
                    if (z2_2 - f2_2 > 64.0f) {
                        fl_2 = 1.0f;
                    }
                    if (z3_2 - f3_2 > 64.0f) {
                        fl_2 = 1.0f;
                    }
                    if (lane_3 == 0) {
                        s_gt[p_off_2 + 64 + 2] = fl_2;
                    }
                }
                asm volatile("barrier.sync 4, 64;" ::: "memory");
                mbarrier_arrive(gates_ready_addr + (vg_c) * 8);
                vg_c += 1;
                if (vg_c == 4) { vg_c = 0; _phase_gates_free_2 ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
