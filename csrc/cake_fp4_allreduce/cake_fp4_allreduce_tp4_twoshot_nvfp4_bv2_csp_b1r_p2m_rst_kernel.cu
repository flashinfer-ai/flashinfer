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
#define SMEM_PAT_OFF 1024
#define SMEM_PAT_STAGE_BYTES 12288
#define SMEM_PAT_STRIDE 12288
#define SMEM_CTL_OFF 13312
#define SMEM_CTL_STAGE_BYTES 16
#define SMEM_CTL_STRIDE 16
#define SMEM_S_AM_OFF 226176
#define SMEM_S_AM_STAGE_BYTES 80
#define SMEM_S_AM_STRIDE 80
#define SMEM_TAB_OFF 226304
#define SMEM_TAB_STAGE_BYTES 64
#define SMEM_TAB_STRIDE 64
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 212800
#define SMEM_WSM_STRIDE 212800
#define SMEM_TOTAL 226432
#define THREADS 352
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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(352, LAUNCH_MIN_BLOCKS) void
cake_fp4_allreduce_tp4_twoshot_nvfp4_bv2_csp_b1r_p2m_rst(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ ts_wire, uint8_t* __restrict__ ts_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
    #define xb_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* pat = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int pat_addr = smem + 1024;
    unsigned int* ctl = reinterpret_cast<unsigned int*>(smem_raw + 13312);
    const int ctl_addr = smem + 13312;
    float* s_am = reinterpret_cast<float*>(smem_raw + 226176);
    const int s_am_addr = smem + 226176;
    long long* tab = reinterpret_cast<long long*>(smem_raw + 226304);
    const int tab_addr = smem + 226304;
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + 13376);
    const int wsm_addr = smem + 13376;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 20 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: workers ----
    if (warp <= 9) {
        { // workers_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            long long bid64 = (long long)bid;
            long long g64 = (long long)gridDim.x;
            long long quads64 = (long long)(rows / 4);
            int r0 = (int)(4 * (bid64 * quads64 / g64));
            int r1 = (int)(4 * ((bid64 + 1) * quads64 / g64) + (bid64 + 1) / g64 * (long long)(rows % 4));
            int nrows = r1 - r0;
            int warp_0 = (int)warp;
            int lane_1 = (int)lane;
            unsigned int xc = 2 * warp_0;
            if (tid == 0) {
                int base = bid * 4;
                ctl[0] = ctrl[base];
                ctl[1] = ctrl[base + 2];
                ctl[2] = ctrl[base + 3];
            }
            if (tid < 8) {
                tab[tid] = peer_ptrs[tid];
            }
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int f = lane_1 >> 1 & 3;
            int xb_off = warp_0 * 21280;
            int own_off = xb_off + 12288;
            int stg_off = own_off + 6944;
            long long rank64 = (long long)rank;
            long long capo64 = (long long)((cap + 4 - 1) / 4);
            long long buf64 = (long long)buf;
            float g_s_0 = gs[0];
            float g_s_1 = gs[1];
            float g_s_2 = gs[2];
            float g_s_3 = gs[3];
            long long sa_w = 4 * capo64 * 3456;
            long long sa_r = 4 * capo64 * 16;
            long long rs_w = buf64 * (2 * sa_w);
            long long ag_w = rs_w + sa_w;
            long long rs_r = buf64 * (2 * sa_r);
            long long ag_r = rs_r + sa_r;
            int n = 0;
            if (nrows > warp_0) {
                n = (nrows - warp_0 + 10 - 1) / 10;
            }
            if (n > 0 && lane_1 == 0) {
                long long first_row = (long long)(r0 + warp_0);
                int slot = 0;
                mbarrier_arrive_expect_tx(xb_full_addr + (xc) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(xb_off + slot * 3472), xq + (first_row * 3456), 3456, xb_full_addr + (xc) * 8);
            }
            unsigned int _phase_xb_full = 0;
            #pragma unroll 1
            for (int k = 0; k < n; k++) {
                int ob_off = xb_off + k % 3 * 3472;
                if (k >= 2) {
                    if (lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                }
                if (lane_1 == 0 && n > k + 1) {
                    long long next_row = (long long)(r0 + warp_0 + 10 * (k + 1));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int slot_1 = (k + 1) % 3;
                    mbarrier_arrive_expect_tx(xb_full_addr + (((xc + 1 < (unsigned int)(2 * warp_0) + 2) ? xc + 1 : (unsigned int)(2 * warp_0))) * 8, 3456);
                    cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(xb_off + slot_1 * 3472), xq + (next_row * 3456), 3456, xb_full_addr + (((xc + 1 < (unsigned int)(2 * warp_0) + 2) ? xc + 1 : (unsigned int)(2 * warp_0))) * 8);
                }
                mbarrier_wait(xb_full_addr + (xc) * 8, _phase_xb_full);
                unsigned int va_0 = wsm_addr + (unsigned int)(ob_off + 16 * lane_1);
                unsigned int w4_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 3]))
                    : "r"(va_0));
                unsigned int c4_0[4];
                unsigned int carry = (w4_0[0] & 2004318071) + 2004318071;
                c4_0[0] = w4_0[0] & (carry | 2004318071);
                unsigned int carry_0 = (w4_0[1] & 2004318071) + 2004318071;
                c4_0[1] = w4_0[1] & (carry_0 | 2004318071);
                unsigned int carry_1 = (w4_0[2] & 2004318071) + 2004318071;
                c4_0[2] = w4_0[2] & (carry_1 | 2004318071);
                unsigned int carry_2 = (w4_0[3] & 2004318071) + 2004318071;
                c4_0[3] = w4_0[3] & (carry_2 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_0), "r"(c4_0[0]), "r"(c4_0[1]), "r"(c4_0[2]), "r"(c4_0[3]) : "memory");
                unsigned int va_1 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_1 + 32));
                unsigned int w4_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                    : "r"(va_1));
                unsigned int c4_1[4];
                unsigned int carry_3 = (w4_1[0] & 2004318071) + 2004318071;
                c4_1[0] = w4_1[0] & (carry_3 | 2004318071);
                unsigned int carry_4 = (w4_1[1] & 2004318071) + 2004318071;
                c4_1[1] = w4_1[1] & (carry_4 | 2004318071);
                unsigned int carry_5 = (w4_1[2] & 2004318071) + 2004318071;
                c4_1[2] = w4_1[2] & (carry_5 | 2004318071);
                unsigned int carry_6 = (w4_1[3] & 2004318071) + 2004318071;
                c4_1[3] = w4_1[3] & (carry_6 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_1), "r"(c4_1[0]), "r"(c4_1[1]), "r"(c4_1[2]), "r"(c4_1[3]) : "memory");
                unsigned int va_2 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_1 + 64));
                unsigned int w4_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 3]))
                    : "r"(va_2));
                unsigned int c4_2[4];
                unsigned int carry_7 = (w4_2[0] & 2004318071) + 2004318071;
                c4_2[0] = w4_2[0] & (carry_7 | 2004318071);
                unsigned int carry_8 = (w4_2[1] & 2004318071) + 2004318071;
                c4_2[1] = w4_2[1] & (carry_8 | 2004318071);
                unsigned int carry_9 = (w4_2[2] & 2004318071) + 2004318071;
                c4_2[2] = w4_2[2] & (carry_9 | 2004318071);
                unsigned int carry_10 = (w4_2[3] & 2004318071) + 2004318071;
                c4_2[3] = w4_2[3] & (carry_10 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_2), "r"(c4_2[0]), "r"(c4_2[1]), "r"(c4_2[2]), "r"(c4_2[3]) : "memory");
                unsigned int va_3 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_1 + 96));
                unsigned int w4_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 3]))
                    : "r"(va_3));
                unsigned int c4_3[4];
                unsigned int carry_11 = (w4_3[0] & 2004318071) + 2004318071;
                c4_3[0] = w4_3[0] & (carry_11 | 2004318071);
                unsigned int carry_12 = (w4_3[1] & 2004318071) + 2004318071;
                c4_3[1] = w4_3[1] & (carry_12 | 2004318071);
                unsigned int carry_13 = (w4_3[2] & 2004318071) + 2004318071;
                c4_3[2] = w4_3[2] & (carry_13 | 2004318071);
                unsigned int carry_14 = (w4_3[3] & 2004318071) + 2004318071;
                c4_3[3] = w4_3[3] & (carry_14 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_3), "r"(c4_3[0]), "r"(c4_3[1]), "r"(c4_3[2]), "r"(c4_3[3]) : "memory");
                unsigned int va_4 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_1 + 128));
                unsigned int w4_4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                    : "r"(va_4));
                unsigned int c4_4[4];
                unsigned int carry_15 = (w4_4[0] & 2004318071) + 2004318071;
                c4_4[0] = w4_4[0] & (carry_15 | 2004318071);
                unsigned int carry_16 = (w4_4[1] & 2004318071) + 2004318071;
                c4_4[1] = w4_4[1] & (carry_16 | 2004318071);
                unsigned int carry_17 = (w4_4[2] & 2004318071) + 2004318071;
                c4_4[2] = w4_4[2] & (carry_17 | 2004318071);
                unsigned int carry_18 = (w4_4[3] & 2004318071) + 2004318071;
                c4_4[3] = w4_4[3] & (carry_18 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_4), "r"(c4_4[0]), "r"(c4_4[1]), "r"(c4_4[2]), "r"(c4_4[3]) : "memory");
                unsigned int va_5 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_1 + 160));
                unsigned int w4_5[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 3]))
                    : "r"(va_5));
                unsigned int c4_5[4];
                unsigned int carry_19 = (w4_5[0] & 2004318071) + 2004318071;
                c4_5[0] = w4_5[0] & (carry_19 | 2004318071);
                unsigned int carry_20 = (w4_5[1] & 2004318071) + 2004318071;
                c4_5[1] = w4_5[1] & (carry_20 | 2004318071);
                unsigned int carry_21 = (w4_5[2] & 2004318071) + 2004318071;
                c4_5[2] = w4_5[2] & (carry_21 | 2004318071);
                unsigned int carry_22 = (w4_5[3] & 2004318071) + 2004318071;
                c4_5[3] = w4_5[3] & (carry_22 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_5), "r"(c4_5[0]), "r"(c4_5[1]), "r"(c4_5[2]), "r"(c4_5[3]) : "memory");
                if (lane_1 < 24) {
                    unsigned int sa = wsm_addr + (unsigned int)(ob_off + 3072 + 16 * lane_1);
                    unsigned int s4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s4[0])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 3]))
                        : "r"(sa));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sa), "r"(s4[0] & 2139062143), "r"(s4[1] & 2139062143), "r"(s4[2] & 2139062143), "r"(s4[3] & 2139062143) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    long long row64 = (long long)(r0 + warp_0 + 10 * k);
                    int owner = (int)(row64 % 4);
                    int ix = (int)(row64 / 4);
                    int rs_wo = (int)rs_w;
                    int rs_ro = (int)rs_r;
                    int capo = (int)capo64;
                    int rk = (int)rank64;
                    long long base_w = tab[owner];
                    long long base_r = tab[4 + owner];
                    uint8_t* dst_w = reinterpret_cast<uint8_t*>(base_w);
                    int v_off = rs_wo + (rk * capo + ix) * 3072;
                    int sc_off = rs_wo + 4 * capo * 3072 + (rk * capo + ix) * 384;
                    {
                        void* _cpbulk_dst_0 = reinterpret_cast<void*>(dst_w + v_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_0), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_1 = reinterpret_cast<void*>(dst_w + sc_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_1), "r"(wsm_addr + (unsigned int)(ob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                xc += 1;
                if (xc == (2 * warp_0 + 2)) { xc = 2 * warp_0; _phase_xb_full ^= 1; }
            }
            int rb0 = r0 + (rank - r0 % 4 + 4) % 4;
            int nB = 0;
            if (rb0 < r0 + nrows) {
                nB = (r0 + nrows - 1 - rb0) / 4 + 1;
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            __syncwarp();
            int nC = nrows - nB;
            int wB = (warp_0 - nrows % 10 + 10) % 10;
            int wC = (warp_0 - (nrows + nB) % 10 + 10) % 10;
            int kb = 0;
            int q0 = (r0 + 4 - 1) / 4;
            int q1 = (r0 + nrows) / 4;
            #pragma unroll 1
            for (int qd = q0 + wB; qd < q1; qd += 10) {
                long long qix64 = (long long)qd;
                long long qrow64 = qix64 * 4 + rank64;
                int qob_off = xb_off + (kb & 1) * 3472;
                if (kb >= 2) {
                    if (lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                }
                int rs_wo_1 = (int)rs_w;
                int rs_ro_1 = (int)rs_r;
                int capo_1 = (int)capo64;
                int ix_1 = (int)qix64;
                int rk_1 = (int)rank64;
                int v = rs_wo_1 + ix_1 * 3072;
                int sc = rs_wo_1 + 4 * capo_1 * 3072 + ix_1 * 384;
                int v_1 = rs_wo_1 + (capo_1 + ix_1) * 3072;
                int sc_1 = rs_wo_1 + 4 * capo_1 * 3072 + (capo_1 + ix_1) * 384;
                int v_2 = rs_wo_1 + (2 * capo_1 + ix_1) * 3072;
                int sc_2 = rs_wo_1 + 4 * capo_1 * 3072 + (2 * capo_1 + ix_1) * 384;
                int v_3 = rs_wo_1 + (3 * capo_1 + ix_1) * 3072;
                int sc_3 = rs_wo_1 + 4 * capo_1 * 3072 + (3 * capo_1 + ix_1) * 384;
                float am = 0.0f;
                int ret_lane = xb_off + 6944 + 16 * lane_1;
                #pragma unroll 1
                for (int jj = 0; jj < 3; jj++) {
                    int pp0 = lane_1 + 64 * jj;
                    int pp1 = pp0 + 32;
                    uint32_t fp4_wire_values[32];
                    uint32_t fp4_wire_scales[8];
                    uint16_t _wire_h16_2_0;
                    uint16_t _wire_h16_2_1;
                    uint16_t _wire_h16_2_2;
                    uint16_t _wire_h16_2_3;
                    uint16_t _wire_h16_2_4;
                    uint16_t _wire_h16_2_5;
                    uint16_t _wire_h16_2_6;
                    uint16_t _wire_h16_2_7;
                    uint32_t _wire_vpend_2 = 255u;
                    uint32_t _wire_spend_2 = 255u;
                    do {
                        if (_wire_vpend_2 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[0]), "=r"(fp4_wire_values[1]), "=r"(fp4_wire_values[2]), "=r"(fp4_wire_values[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_2 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[4]), "=r"(fp4_wire_values[5]), "=r"(fp4_wire_values[6]), "=r"(fp4_wire_values[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_2 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[8]), "=r"(fp4_wire_values[9]), "=r"(fp4_wire_values[10]), "=r"(fp4_wire_values[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_2 & 8u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[12]), "=r"(fp4_wire_values[13]), "=r"(fp4_wire_values[14]), "=r"(fp4_wire_values[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_2 & 16u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[16]), "=r"(fp4_wire_values[17]), "=r"(fp4_wire_values[18]), "=r"(fp4_wire_values[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_2 & 32u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[20]), "=r"(fp4_wire_values[21]), "=r"(fp4_wire_values[22]), "=r"(fp4_wire_values[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_2 & 64u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[24]), "=r"(fp4_wire_values[25]), "=r"(fp4_wire_values[26]), "=r"(fp4_wire_values[27]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_2 & 128u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[28]), "=r"(fp4_wire_values[29]), "=r"(fp4_wire_values[30]), "=r"(fp4_wire_values[31]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_spend_2 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp0)))) : "memory");
                            fp4_wire_scales[0] = (uint32_t)_wire_h16_2_0;
                        }
                        if (_wire_spend_2 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp0)))) : "memory");
                            fp4_wire_scales[1] = (uint32_t)_wire_h16_2_1;
                        }
                        if (_wire_spend_2 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * pp0)))) : "memory");
                            fp4_wire_scales[2] = (uint32_t)_wire_h16_2_2;
                        }
                        if (_wire_spend_2 & 8u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp0)))) : "memory");
                            fp4_wire_scales[3] = (uint32_t)_wire_h16_2_3;
                        }
                        if (_wire_spend_2 & 16u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp1)))) : "memory");
                            fp4_wire_scales[4] = (uint32_t)_wire_h16_2_4;
                        }
                        if (_wire_spend_2 & 32u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp1)))) : "memory");
                            fp4_wire_scales[5] = (uint32_t)_wire_h16_2_5;
                        }
                        if (_wire_spend_2 & 64u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_6) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * pp1)))) : "memory");
                            fp4_wire_scales[6] = (uint32_t)_wire_h16_2_6;
                        }
                        if (_wire_spend_2 & 128u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_7) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp1)))) : "memory");
                            fp4_wire_scales[7] = (uint32_t)_wire_h16_2_7;
                        }
                        _wire_vpend_2 = ((uint32_t)((((fp4_wire_values[0] & ~((fp4_wire_values[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[1] & ~((fp4_wire_values[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[2] & ~((fp4_wire_values[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[3] & ~((fp4_wire_values[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((fp4_wire_values[4] & ~((fp4_wire_values[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[5] & ~((fp4_wire_values[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[6] & ~((fp4_wire_values[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[7] & ~((fp4_wire_values[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((fp4_wire_values[8] & ~((fp4_wire_values[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[9] & ~((fp4_wire_values[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[10] & ~((fp4_wire_values[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[11] & ~((fp4_wire_values[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((fp4_wire_values[12] & ~((fp4_wire_values[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[13] & ~((fp4_wire_values[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[14] & ~((fp4_wire_values[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[15] & ~((fp4_wire_values[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((fp4_wire_values[16] & ~((fp4_wire_values[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[17] & ~((fp4_wire_values[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[18] & ~((fp4_wire_values[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[19] & ~((fp4_wire_values[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((fp4_wire_values[20] & ~((fp4_wire_values[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[21] & ~((fp4_wire_values[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[22] & ~((fp4_wire_values[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[23] & ~((fp4_wire_values[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5) | ((uint32_t)((((fp4_wire_values[24] & ~((fp4_wire_values[24] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[25] & ~((fp4_wire_values[25] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[26] & ~((fp4_wire_values[26] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[27] & ~((fp4_wire_values[27] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 6) | ((uint32_t)((((fp4_wire_values[28] & ~((fp4_wire_values[28] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[29] & ~((fp4_wire_values[29] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[30] & ~((fp4_wire_values[30] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[31] & ~((fp4_wire_values[31] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 7);
                        _wire_spend_2 = ((uint32_t)(((uint32_t)((fp4_wire_scales[0] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((fp4_wire_scales[1] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((fp4_wire_scales[2] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((fp4_wire_scales[3] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((fp4_wire_scales[4] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((fp4_wire_scales[5] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5) | ((uint32_t)(((uint32_t)((fp4_wire_scales[6] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[6] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 6) | ((uint32_t)(((uint32_t)((fp4_wire_scales[7] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[7] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 7);
                    } while ((_wire_vpend_2 | _wire_spend_2) != 0u);
                    int wbase = ret_lane + 4096 * jj;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)wbase), "r"(fp4_wire_values[0]), "r"(fp4_wire_values[1]), "r"(fp4_wire_values[2]), "r"(fp4_wire_values[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase + 512)), "r"(fp4_wire_values[4]), "r"(fp4_wire_values[5]), "r"(fp4_wire_values[6]), "r"(fp4_wire_values[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase + 1024)), "r"(fp4_wire_values[8]), "r"(fp4_wire_values[9]), "r"(fp4_wire_values[10]), "r"(fp4_wire_values[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase + 1536)), "r"(fp4_wire_values[12]), "r"(fp4_wire_values[13]), "r"(fp4_wire_values[14]), "r"(fp4_wire_values[15]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase + 2048)), "r"(fp4_wire_values[16]), "r"(fp4_wire_values[17]), "r"(fp4_wire_values[18]), "r"(fp4_wire_values[19]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase + 2560)), "r"(fp4_wire_values[20]), "r"(fp4_wire_values[21]), "r"(fp4_wire_values[22]), "r"(fp4_wire_values[23]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase + 3072)), "r"(fp4_wire_values[24]), "r"(fp4_wire_values[25]), "r"(fp4_wire_values[26]), "r"(fp4_wire_values[27]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase + 3584)), "r"(fp4_wire_values[28]), "r"(fp4_wire_values[29]), "r"(fp4_wire_values[30]), "r"(fp4_wire_values[31]) : "memory");
                    float acc_0[32] = {0};
                    uint32_t e4m3x2_to_f16x2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2) : "h"((uint16_t)(fp4_wire_scales[0])));
                    float cvt_f32_f16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16) : "h"((uint16_t)(e4m3x2_to_f16x2)));
                    float cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1) : "h"((uint16_t)(e4m3x2_to_f16x2 >> 16)));
                    float s0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(cvt_f32_f16), "f"(g_s_0));
                    float s1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1) : "f"(cvt_f32_f16_1), "f"(g_s_0));
                    float2 scale0 = make_float2(s0, s0);
                    float2 scale1 = make_float2(s1, s1);
                    float tv_0[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[0]));
                        (&tv_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_0[0]), "=f"(tv_0[1]), "=f"(tv_0[2]), "=f"(tv_0[3]), "=f"(tv_0[4]), "=f"(tv_0[5]), "=f"(tv_0[6]), "=f"(tv_0[7]) : "r"(fp4_wire_values[0]));
                        #endif
                    }
                    float2 a_0 = make_float2(acc_0[0], acc_0[1]);
                    float2 q_0 = make_float2(tv_0[0], tv_0[1]);
                    float2 r_0 = fma_f32x2_rn_noftz(q_0, scale0, a_0);
                    acc_0[0] = r_0.x;
                    acc_0[1] = r_0.y;
                    float2 a_1 = make_float2(acc_0[2], acc_0[3]);
                    float2 q_1 = make_float2(tv_0[2], tv_0[3]);
                    float2 r_1 = fma_f32x2_rn_noftz(q_1, scale0, a_1);
                    acc_0[2] = r_1.x;
                    acc_0[3] = r_1.y;
                    float2 a_2 = make_float2(acc_0[4], acc_0[5]);
                    float2 q_2 = make_float2(tv_0[4], tv_0[5]);
                    float2 r_2 = fma_f32x2_rn_noftz(q_2, scale0, a_2);
                    acc_0[4] = r_2.x;
                    acc_0[5] = r_2.y;
                    float2 a_3 = make_float2(acc_0[6], acc_0[7]);
                    float2 q_3 = make_float2(tv_0[6], tv_0[7]);
                    float2 r_3 = fma_f32x2_rn_noftz(q_3, scale0, a_3);
                    acc_0[6] = r_3.x;
                    acc_0[7] = r_3.y;
                    float tv_1[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[1 + 0]));
                        (&tv_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_1[0]), "=f"(tv_1[1]), "=f"(tv_1[2]), "=f"(tv_1[3]), "=f"(tv_1[4]), "=f"(tv_1[5]), "=f"(tv_1[6]), "=f"(tv_1[7]) : "r"(fp4_wire_values[1 + 0]));
                        #endif
                    }
                    float2 a_4 = make_float2(acc_0[8], acc_0[9]);
                    float2 q_4 = make_float2(tv_1[0], tv_1[1]);
                    float2 r_4 = fma_f32x2_rn_noftz(q_4, scale0, a_4);
                    acc_0[8] = r_4.x;
                    acc_0[9] = r_4.y;
                    float2 a_5 = make_float2(acc_0[10], acc_0[11]);
                    float2 q_5 = make_float2(tv_1[2], tv_1[3]);
                    float2 r_5 = fma_f32x2_rn_noftz(q_5, scale0, a_5);
                    acc_0[10] = r_5.x;
                    acc_0[11] = r_5.y;
                    float2 a_6 = make_float2(acc_0[12], acc_0[13]);
                    float2 q_6 = make_float2(tv_1[4], tv_1[5]);
                    float2 r_6 = fma_f32x2_rn_noftz(q_6, scale0, a_6);
                    acc_0[12] = r_6.x;
                    acc_0[13] = r_6.y;
                    float2 a_7 = make_float2(acc_0[14], acc_0[15]);
                    float2 q_7 = make_float2(tv_1[6], tv_1[7]);
                    float2 r_7 = fma_f32x2_rn_noftz(q_7, scale0, a_7);
                    acc_0[14] = r_7.x;
                    acc_0[15] = r_7.y;
                    float tv_2[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[2 + 0]));
                        (&tv_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_2[0]), "=f"(tv_2[1]), "=f"(tv_2[2]), "=f"(tv_2[3]), "=f"(tv_2[4]), "=f"(tv_2[5]), "=f"(tv_2[6]), "=f"(tv_2[7]) : "r"(fp4_wire_values[2 + 0]));
                        #endif
                    }
                    float2 a_8 = make_float2(acc_0[16], acc_0[17]);
                    float2 q_8 = make_float2(tv_2[0], tv_2[1]);
                    float2 r_8 = fma_f32x2_rn_noftz(q_8, scale1, a_8);
                    acc_0[16] = r_8.x;
                    acc_0[17] = r_8.y;
                    float2 a_9 = make_float2(acc_0[18], acc_0[19]);
                    float2 q_9 = make_float2(tv_2[2], tv_2[3]);
                    float2 r_9 = fma_f32x2_rn_noftz(q_9, scale1, a_9);
                    acc_0[18] = r_9.x;
                    acc_0[19] = r_9.y;
                    float2 a_10 = make_float2(acc_0[20], acc_0[21]);
                    float2 q_10 = make_float2(tv_2[4], tv_2[5]);
                    float2 r_10 = fma_f32x2_rn_noftz(q_10, scale1, a_10);
                    acc_0[20] = r_10.x;
                    acc_0[21] = r_10.y;
                    float2 a_11 = make_float2(acc_0[22], acc_0[23]);
                    float2 q_11 = make_float2(tv_2[6], tv_2[7]);
                    float2 r_11 = fma_f32x2_rn_noftz(q_11, scale1, a_11);
                    acc_0[22] = r_11.x;
                    acc_0[23] = r_11.y;
                    float tv_3[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[3 + 0]));
                        (&tv_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_3[0]), "=f"(tv_3[1]), "=f"(tv_3[2]), "=f"(tv_3[3]), "=f"(tv_3[4]), "=f"(tv_3[5]), "=f"(tv_3[6]), "=f"(tv_3[7]) : "r"(fp4_wire_values[3 + 0]));
                        #endif
                    }
                    float2 a_12 = make_float2(acc_0[24], acc_0[25]);
                    float2 q_12 = make_float2(tv_3[0], tv_3[1]);
                    float2 r_12 = fma_f32x2_rn_noftz(q_12, scale1, a_12);
                    acc_0[24] = r_12.x;
                    acc_0[25] = r_12.y;
                    float2 a_13 = make_float2(acc_0[26], acc_0[27]);
                    float2 q_13 = make_float2(tv_3[2], tv_3[3]);
                    float2 r_13 = fma_f32x2_rn_noftz(q_13, scale1, a_13);
                    acc_0[26] = r_13.x;
                    acc_0[27] = r_13.y;
                    float2 a_14 = make_float2(acc_0[28], acc_0[29]);
                    float2 q_14 = make_float2(tv_3[4], tv_3[5]);
                    float2 r_14 = fma_f32x2_rn_noftz(q_14, scale1, a_14);
                    acc_0[28] = r_14.x;
                    acc_0[29] = r_14.y;
                    float2 a_15 = make_float2(acc_0[30], acc_0[31]);
                    float2 q_15 = make_float2(tv_3[6], tv_3[7]);
                    float2 r_15 = fma_f32x2_rn_noftz(q_15, scale1, a_15);
                    acc_0[30] = r_15.x;
                    acc_0[31] = r_15.y;
                    uint32_t e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_0) : "h"((uint16_t)(fp4_wire_scales[1])));
                    float cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0)));
                    float cvt_f32_f16_1_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0 >> 16)));
                    float s0_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(cvt_f32_f16_0), "f"(g_s_1));
                    float s1_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_0) : "f"(cvt_f32_f16_1_0), "f"(g_s_1));
                    float2 scale0_0 = make_float2(s0_0, s0_0);
                    float2 scale1_0 = make_float2(s1_0, s1_0);
                    float tv_4[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[4 + 0]));
                        (&tv_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_4[0]), "=f"(tv_4[1]), "=f"(tv_4[2]), "=f"(tv_4[3]), "=f"(tv_4[4]), "=f"(tv_4[5]), "=f"(tv_4[6]), "=f"(tv_4[7]) : "r"(fp4_wire_values[4 + 0]));
                        #endif
                    }
                    float2 a_16 = make_float2(acc_0[0], acc_0[1]);
                    float2 q_16 = make_float2(tv_4[0], tv_4[1]);
                    float2 r_16 = fma_f32x2_rn_noftz(q_16, scale0_0, a_16);
                    acc_0[0] = r_16.x;
                    acc_0[1] = r_16.y;
                    float2 a_17 = make_float2(acc_0[2], acc_0[3]);
                    float2 q_17 = make_float2(tv_4[2], tv_4[3]);
                    float2 r_17 = fma_f32x2_rn_noftz(q_17, scale0_0, a_17);
                    acc_0[2] = r_17.x;
                    acc_0[3] = r_17.y;
                    float2 a_18 = make_float2(acc_0[4], acc_0[5]);
                    float2 q_18 = make_float2(tv_4[4], tv_4[5]);
                    float2 r_18 = fma_f32x2_rn_noftz(q_18, scale0_0, a_18);
                    acc_0[4] = r_18.x;
                    acc_0[5] = r_18.y;
                    float2 a_19 = make_float2(acc_0[6], acc_0[7]);
                    float2 q_19 = make_float2(tv_4[6], tv_4[7]);
                    float2 r_19 = fma_f32x2_rn_noftz(q_19, scale0_0, a_19);
                    acc_0[6] = r_19.x;
                    acc_0[7] = r_19.y;
                    float tv_5[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[5 + 0]));
                        (&tv_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_5[0]), "=f"(tv_5[1]), "=f"(tv_5[2]), "=f"(tv_5[3]), "=f"(tv_5[4]), "=f"(tv_5[5]), "=f"(tv_5[6]), "=f"(tv_5[7]) : "r"(fp4_wire_values[5 + 0]));
                        #endif
                    }
                    float2 a_20 = make_float2(acc_0[8], acc_0[9]);
                    float2 q_20 = make_float2(tv_5[0], tv_5[1]);
                    float2 r_20 = fma_f32x2_rn_noftz(q_20, scale0_0, a_20);
                    acc_0[8] = r_20.x;
                    acc_0[9] = r_20.y;
                    float2 a_21 = make_float2(acc_0[10], acc_0[11]);
                    float2 q_21 = make_float2(tv_5[2], tv_5[3]);
                    float2 r_21 = fma_f32x2_rn_noftz(q_21, scale0_0, a_21);
                    acc_0[10] = r_21.x;
                    acc_0[11] = r_21.y;
                    float2 a_22 = make_float2(acc_0[12], acc_0[13]);
                    float2 q_22 = make_float2(tv_5[4], tv_5[5]);
                    float2 r_22 = fma_f32x2_rn_noftz(q_22, scale0_0, a_22);
                    acc_0[12] = r_22.x;
                    acc_0[13] = r_22.y;
                    float2 a_23 = make_float2(acc_0[14], acc_0[15]);
                    float2 q_23 = make_float2(tv_5[6], tv_5[7]);
                    float2 r_23 = fma_f32x2_rn_noftz(q_23, scale0_0, a_23);
                    acc_0[14] = r_23.x;
                    acc_0[15] = r_23.y;
                    float tv_6[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[6 + 0]));
                        (&tv_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_6[0]), "=f"(tv_6[1]), "=f"(tv_6[2]), "=f"(tv_6[3]), "=f"(tv_6[4]), "=f"(tv_6[5]), "=f"(tv_6[6]), "=f"(tv_6[7]) : "r"(fp4_wire_values[6 + 0]));
                        #endif
                    }
                    float2 a_24 = make_float2(acc_0[16], acc_0[17]);
                    float2 q_24 = make_float2(tv_6[0], tv_6[1]);
                    float2 r_24 = fma_f32x2_rn_noftz(q_24, scale1_0, a_24);
                    acc_0[16] = r_24.x;
                    acc_0[17] = r_24.y;
                    float2 a_25 = make_float2(acc_0[18], acc_0[19]);
                    float2 q_25 = make_float2(tv_6[2], tv_6[3]);
                    float2 r_25 = fma_f32x2_rn_noftz(q_25, scale1_0, a_25);
                    acc_0[18] = r_25.x;
                    acc_0[19] = r_25.y;
                    float2 a_26 = make_float2(acc_0[20], acc_0[21]);
                    float2 q_26 = make_float2(tv_6[4], tv_6[5]);
                    float2 r_26 = fma_f32x2_rn_noftz(q_26, scale1_0, a_26);
                    acc_0[20] = r_26.x;
                    acc_0[21] = r_26.y;
                    float2 a_27 = make_float2(acc_0[22], acc_0[23]);
                    float2 q_27 = make_float2(tv_6[6], tv_6[7]);
                    float2 r_27 = fma_f32x2_rn_noftz(q_27, scale1_0, a_27);
                    acc_0[22] = r_27.x;
                    acc_0[23] = r_27.y;
                    float tv_7[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[7 + 0]));
                        (&tv_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_7[0]), "=f"(tv_7[1]), "=f"(tv_7[2]), "=f"(tv_7[3]), "=f"(tv_7[4]), "=f"(tv_7[5]), "=f"(tv_7[6]), "=f"(tv_7[7]) : "r"(fp4_wire_values[7 + 0]));
                        #endif
                    }
                    float2 a_28 = make_float2(acc_0[24], acc_0[25]);
                    float2 q_28 = make_float2(tv_7[0], tv_7[1]);
                    float2 r_28 = fma_f32x2_rn_noftz(q_28, scale1_0, a_28);
                    acc_0[24] = r_28.x;
                    acc_0[25] = r_28.y;
                    float2 a_29 = make_float2(acc_0[26], acc_0[27]);
                    float2 q_29 = make_float2(tv_7[2], tv_7[3]);
                    float2 r_29 = fma_f32x2_rn_noftz(q_29, scale1_0, a_29);
                    acc_0[26] = r_29.x;
                    acc_0[27] = r_29.y;
                    float2 a_30 = make_float2(acc_0[28], acc_0[29]);
                    float2 q_30 = make_float2(tv_7[4], tv_7[5]);
                    float2 r_30 = fma_f32x2_rn_noftz(q_30, scale1_0, a_30);
                    acc_0[28] = r_30.x;
                    acc_0[29] = r_30.y;
                    float2 a_31 = make_float2(acc_0[30], acc_0[31]);
                    float2 q_31 = make_float2(tv_7[6], tv_7[7]);
                    float2 r_31 = fma_f32x2_rn_noftz(q_31, scale1_0, a_31);
                    acc_0[30] = r_31.x;
                    acc_0[31] = r_31.y;
                    uint32_t e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_1) : "h"((uint16_t)(fp4_wire_scales[2])));
                    float cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_2) : "h"((uint16_t)(e4m3x2_to_f16x2_1)));
                    float cvt_f32_f16_1_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_1) : "h"((uint16_t)(e4m3x2_to_f16x2_1 >> 16)));
                    float s0_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(cvt_f32_f16_2), "f"(g_s_2));
                    float s1_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_1) : "f"(cvt_f32_f16_1_1), "f"(g_s_2));
                    float2 scale0_1 = make_float2(s0_1, s0_1);
                    float2 scale1_1 = make_float2(s1_1, s1_1);
                    float tv_8[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[8 + 0]));
                        (&tv_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_8[0]), "=f"(tv_8[1]), "=f"(tv_8[2]), "=f"(tv_8[3]), "=f"(tv_8[4]), "=f"(tv_8[5]), "=f"(tv_8[6]), "=f"(tv_8[7]) : "r"(fp4_wire_values[8 + 0]));
                        #endif
                    }
                    float2 a_32 = make_float2(acc_0[0], acc_0[1]);
                    float2 q_32 = make_float2(tv_8[0], tv_8[1]);
                    float2 r_32 = fma_f32x2_rn_noftz(q_32, scale0_1, a_32);
                    acc_0[0] = r_32.x;
                    acc_0[1] = r_32.y;
                    float2 a_33 = make_float2(acc_0[2], acc_0[3]);
                    float2 q_33 = make_float2(tv_8[2], tv_8[3]);
                    float2 r_33 = fma_f32x2_rn_noftz(q_33, scale0_1, a_33);
                    acc_0[2] = r_33.x;
                    acc_0[3] = r_33.y;
                    float2 a_34 = make_float2(acc_0[4], acc_0[5]);
                    float2 q_34 = make_float2(tv_8[4], tv_8[5]);
                    float2 r_34 = fma_f32x2_rn_noftz(q_34, scale0_1, a_34);
                    acc_0[4] = r_34.x;
                    acc_0[5] = r_34.y;
                    float2 a_35 = make_float2(acc_0[6], acc_0[7]);
                    float2 q_35 = make_float2(tv_8[6], tv_8[7]);
                    float2 r_35 = fma_f32x2_rn_noftz(q_35, scale0_1, a_35);
                    acc_0[6] = r_35.x;
                    acc_0[7] = r_35.y;
                    float tv_9[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[9 + 0]));
                        (&tv_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_9[0]), "=f"(tv_9[1]), "=f"(tv_9[2]), "=f"(tv_9[3]), "=f"(tv_9[4]), "=f"(tv_9[5]), "=f"(tv_9[6]), "=f"(tv_9[7]) : "r"(fp4_wire_values[9 + 0]));
                        #endif
                    }
                    float2 a_36 = make_float2(acc_0[8], acc_0[9]);
                    float2 q_36 = make_float2(tv_9[0], tv_9[1]);
                    float2 r_36 = fma_f32x2_rn_noftz(q_36, scale0_1, a_36);
                    acc_0[8] = r_36.x;
                    acc_0[9] = r_36.y;
                    float2 a_37 = make_float2(acc_0[10], acc_0[11]);
                    float2 q_37 = make_float2(tv_9[2], tv_9[3]);
                    float2 r_37 = fma_f32x2_rn_noftz(q_37, scale0_1, a_37);
                    acc_0[10] = r_37.x;
                    acc_0[11] = r_37.y;
                    float2 a_38 = make_float2(acc_0[12], acc_0[13]);
                    float2 q_38 = make_float2(tv_9[4], tv_9[5]);
                    float2 r_38 = fma_f32x2_rn_noftz(q_38, scale0_1, a_38);
                    acc_0[12] = r_38.x;
                    acc_0[13] = r_38.y;
                    float2 a_39 = make_float2(acc_0[14], acc_0[15]);
                    float2 q_39 = make_float2(tv_9[6], tv_9[7]);
                    float2 r_39 = fma_f32x2_rn_noftz(q_39, scale0_1, a_39);
                    acc_0[14] = r_39.x;
                    acc_0[15] = r_39.y;
                    float tv_10[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[10 + 0]));
                        (&tv_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_10[0]), "=f"(tv_10[1]), "=f"(tv_10[2]), "=f"(tv_10[3]), "=f"(tv_10[4]), "=f"(tv_10[5]), "=f"(tv_10[6]), "=f"(tv_10[7]) : "r"(fp4_wire_values[10 + 0]));
                        #endif
                    }
                    float2 a_40 = make_float2(acc_0[16], acc_0[17]);
                    float2 q_40 = make_float2(tv_10[0], tv_10[1]);
                    float2 r_40 = fma_f32x2_rn_noftz(q_40, scale1_1, a_40);
                    acc_0[16] = r_40.x;
                    acc_0[17] = r_40.y;
                    float2 a_41 = make_float2(acc_0[18], acc_0[19]);
                    float2 q_41 = make_float2(tv_10[2], tv_10[3]);
                    float2 r_41 = fma_f32x2_rn_noftz(q_41, scale1_1, a_41);
                    acc_0[18] = r_41.x;
                    acc_0[19] = r_41.y;
                    float2 a_42 = make_float2(acc_0[20], acc_0[21]);
                    float2 q_42 = make_float2(tv_10[4], tv_10[5]);
                    float2 r_42 = fma_f32x2_rn_noftz(q_42, scale1_1, a_42);
                    acc_0[20] = r_42.x;
                    acc_0[21] = r_42.y;
                    float2 a_43 = make_float2(acc_0[22], acc_0[23]);
                    float2 q_43 = make_float2(tv_10[6], tv_10[7]);
                    float2 r_43 = fma_f32x2_rn_noftz(q_43, scale1_1, a_43);
                    acc_0[22] = r_43.x;
                    acc_0[23] = r_43.y;
                    float tv_11[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[11 + 0]));
                        (&tv_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_11[0]), "=f"(tv_11[1]), "=f"(tv_11[2]), "=f"(tv_11[3]), "=f"(tv_11[4]), "=f"(tv_11[5]), "=f"(tv_11[6]), "=f"(tv_11[7]) : "r"(fp4_wire_values[11 + 0]));
                        #endif
                    }
                    float2 a_44 = make_float2(acc_0[24], acc_0[25]);
                    float2 q_44 = make_float2(tv_11[0], tv_11[1]);
                    float2 r_44 = fma_f32x2_rn_noftz(q_44, scale1_1, a_44);
                    acc_0[24] = r_44.x;
                    acc_0[25] = r_44.y;
                    float2 a_45 = make_float2(acc_0[26], acc_0[27]);
                    float2 q_45 = make_float2(tv_11[2], tv_11[3]);
                    float2 r_45 = fma_f32x2_rn_noftz(q_45, scale1_1, a_45);
                    acc_0[26] = r_45.x;
                    acc_0[27] = r_45.y;
                    float2 a_46 = make_float2(acc_0[28], acc_0[29]);
                    float2 q_46 = make_float2(tv_11[4], tv_11[5]);
                    float2 r_46 = fma_f32x2_rn_noftz(q_46, scale1_1, a_46);
                    acc_0[28] = r_46.x;
                    acc_0[29] = r_46.y;
                    float2 a_47 = make_float2(acc_0[30], acc_0[31]);
                    float2 q_47 = make_float2(tv_11[6], tv_11[7]);
                    float2 r_47 = fma_f32x2_rn_noftz(q_47, scale1_1, a_47);
                    acc_0[30] = r_47.x;
                    acc_0[31] = r_47.y;
                    uint32_t e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_2) : "h"((uint16_t)(fp4_wire_scales[3])));
                    float cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_3) : "h"((uint16_t)(e4m3x2_to_f16x2_2)));
                    float cvt_f32_f16_1_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_2) : "h"((uint16_t)(e4m3x2_to_f16x2_2 >> 16)));
                    float s0_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(cvt_f32_f16_3), "f"(g_s_3));
                    float s1_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_2) : "f"(cvt_f32_f16_1_2), "f"(g_s_3));
                    float2 scale0_2 = make_float2(s0_2, s0_2);
                    float2 scale1_2 = make_float2(s1_2, s1_2);
                    float tv_12[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[12 + 0]));
                        (&tv_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_12[0]), "=f"(tv_12[1]), "=f"(tv_12[2]), "=f"(tv_12[3]), "=f"(tv_12[4]), "=f"(tv_12[5]), "=f"(tv_12[6]), "=f"(tv_12[7]) : "r"(fp4_wire_values[12 + 0]));
                        #endif
                    }
                    float2 a_48 = make_float2(acc_0[0], acc_0[1]);
                    float2 q_48 = make_float2(tv_12[0], tv_12[1]);
                    float2 r_48 = fma_f32x2_rn_noftz(q_48, scale0_2, a_48);
                    acc_0[0] = r_48.x;
                    acc_0[1] = r_48.y;
                    float2 a_49 = make_float2(acc_0[2], acc_0[3]);
                    float2 q_49 = make_float2(tv_12[2], tv_12[3]);
                    float2 r_49 = fma_f32x2_rn_noftz(q_49, scale0_2, a_49);
                    acc_0[2] = r_49.x;
                    acc_0[3] = r_49.y;
                    float2 a_50 = make_float2(acc_0[4], acc_0[5]);
                    float2 q_50 = make_float2(tv_12[4], tv_12[5]);
                    float2 r_50 = fma_f32x2_rn_noftz(q_50, scale0_2, a_50);
                    acc_0[4] = r_50.x;
                    acc_0[5] = r_50.y;
                    float2 a_51 = make_float2(acc_0[6], acc_0[7]);
                    float2 q_51 = make_float2(tv_12[6], tv_12[7]);
                    float2 r_51 = fma_f32x2_rn_noftz(q_51, scale0_2, a_51);
                    acc_0[6] = r_51.x;
                    acc_0[7] = r_51.y;
                    float tv_13[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[13 + 0]));
                        (&tv_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_13[0]), "=f"(tv_13[1]), "=f"(tv_13[2]), "=f"(tv_13[3]), "=f"(tv_13[4]), "=f"(tv_13[5]), "=f"(tv_13[6]), "=f"(tv_13[7]) : "r"(fp4_wire_values[13 + 0]));
                        #endif
                    }
                    float2 a_52 = make_float2(acc_0[8], acc_0[9]);
                    float2 q_52 = make_float2(tv_13[0], tv_13[1]);
                    float2 r_52 = fma_f32x2_rn_noftz(q_52, scale0_2, a_52);
                    acc_0[8] = r_52.x;
                    acc_0[9] = r_52.y;
                    float2 a_53 = make_float2(acc_0[10], acc_0[11]);
                    float2 q_53 = make_float2(tv_13[2], tv_13[3]);
                    float2 r_53 = fma_f32x2_rn_noftz(q_53, scale0_2, a_53);
                    acc_0[10] = r_53.x;
                    acc_0[11] = r_53.y;
                    float2 a_54 = make_float2(acc_0[12], acc_0[13]);
                    float2 q_54 = make_float2(tv_13[4], tv_13[5]);
                    float2 r_54 = fma_f32x2_rn_noftz(q_54, scale0_2, a_54);
                    acc_0[12] = r_54.x;
                    acc_0[13] = r_54.y;
                    float2 a_55 = make_float2(acc_0[14], acc_0[15]);
                    float2 q_55 = make_float2(tv_13[6], tv_13[7]);
                    float2 r_55 = fma_f32x2_rn_noftz(q_55, scale0_2, a_55);
                    acc_0[14] = r_55.x;
                    acc_0[15] = r_55.y;
                    float tv_14[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[14 + 0]));
                        (&tv_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_14[0]), "=f"(tv_14[1]), "=f"(tv_14[2]), "=f"(tv_14[3]), "=f"(tv_14[4]), "=f"(tv_14[5]), "=f"(tv_14[6]), "=f"(tv_14[7]) : "r"(fp4_wire_values[14 + 0]));
                        #endif
                    }
                    float2 a_56 = make_float2(acc_0[16], acc_0[17]);
                    float2 q_56 = make_float2(tv_14[0], tv_14[1]);
                    float2 r_56 = fma_f32x2_rn_noftz(q_56, scale1_2, a_56);
                    acc_0[16] = r_56.x;
                    acc_0[17] = r_56.y;
                    float2 a_57 = make_float2(acc_0[18], acc_0[19]);
                    float2 q_57 = make_float2(tv_14[2], tv_14[3]);
                    float2 r_57 = fma_f32x2_rn_noftz(q_57, scale1_2, a_57);
                    acc_0[18] = r_57.x;
                    acc_0[19] = r_57.y;
                    float2 a_58 = make_float2(acc_0[20], acc_0[21]);
                    float2 q_58 = make_float2(tv_14[4], tv_14[5]);
                    float2 r_58 = fma_f32x2_rn_noftz(q_58, scale1_2, a_58);
                    acc_0[20] = r_58.x;
                    acc_0[21] = r_58.y;
                    float2 a_59 = make_float2(acc_0[22], acc_0[23]);
                    float2 q_59 = make_float2(tv_14[6], tv_14[7]);
                    float2 r_59 = fma_f32x2_rn_noftz(q_59, scale1_2, a_59);
                    acc_0[22] = r_59.x;
                    acc_0[23] = r_59.y;
                    float tv_15[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[15 + 0]));
                        (&tv_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_15[0]), "=f"(tv_15[1]), "=f"(tv_15[2]), "=f"(tv_15[3]), "=f"(tv_15[4]), "=f"(tv_15[5]), "=f"(tv_15[6]), "=f"(tv_15[7]) : "r"(fp4_wire_values[15 + 0]));
                        #endif
                    }
                    float2 a_60 = make_float2(acc_0[24], acc_0[25]);
                    float2 q_60 = make_float2(tv_15[0], tv_15[1]);
                    float2 r_60 = fma_f32x2_rn_noftz(q_60, scale1_2, a_60);
                    acc_0[24] = r_60.x;
                    acc_0[25] = r_60.y;
                    float2 a_61 = make_float2(acc_0[26], acc_0[27]);
                    float2 q_61 = make_float2(tv_15[2], tv_15[3]);
                    float2 r_61 = fma_f32x2_rn_noftz(q_61, scale1_2, a_61);
                    acc_0[26] = r_61.x;
                    acc_0[27] = r_61.y;
                    float2 a_62 = make_float2(acc_0[28], acc_0[29]);
                    float2 q_62 = make_float2(tv_15[4], tv_15[5]);
                    float2 r_62 = fma_f32x2_rn_noftz(q_62, scale1_2, a_62);
                    acc_0[28] = r_62.x;
                    acc_0[29] = r_62.y;
                    float2 a_63 = make_float2(acc_0[30], acc_0[31]);
                    float2 q_63 = make_float2(tv_15[6], tv_15[7]);
                    float2 r_63 = fma_f32x2_rn_noftz(q_63, scale1_2, a_63);
                    acc_0[30] = r_63.x;
                    acc_0[31] = r_63.y;
                    float cur = am;
                    cur = fmaxf(cur, fabsf(acc_0[0]));
                    cur = fmaxf(cur, fabsf(acc_0[1]));
                    cur = fmaxf(cur, fabsf(acc_0[2]));
                    cur = fmaxf(cur, fabsf(acc_0[3]));
                    cur = fmaxf(cur, fabsf(acc_0[4]));
                    cur = fmaxf(cur, fabsf(acc_0[5]));
                    cur = fmaxf(cur, fabsf(acc_0[6]));
                    cur = fmaxf(cur, fabsf(acc_0[7]));
                    cur = fmaxf(cur, fabsf(acc_0[8]));
                    cur = fmaxf(cur, fabsf(acc_0[9]));
                    cur = fmaxf(cur, fabsf(acc_0[10]));
                    cur = fmaxf(cur, fabsf(acc_0[11]));
                    cur = fmaxf(cur, fabsf(acc_0[12]));
                    cur = fmaxf(cur, fabsf(acc_0[13]));
                    cur = fmaxf(cur, fabsf(acc_0[14]));
                    cur = fmaxf(cur, fabsf(acc_0[15]));
                    cur = fmaxf(cur, fabsf(acc_0[16]));
                    cur = fmaxf(cur, fabsf(acc_0[17]));
                    cur = fmaxf(cur, fabsf(acc_0[18]));
                    cur = fmaxf(cur, fabsf(acc_0[19]));
                    cur = fmaxf(cur, fabsf(acc_0[20]));
                    cur = fmaxf(cur, fabsf(acc_0[21]));
                    cur = fmaxf(cur, fabsf(acc_0[22]));
                    cur = fmaxf(cur, fabsf(acc_0[23]));
                    cur = fmaxf(cur, fabsf(acc_0[24]));
                    cur = fmaxf(cur, fabsf(acc_0[25]));
                    cur = fmaxf(cur, fabsf(acc_0[26]));
                    cur = fmaxf(cur, fabsf(acc_0[27]));
                    cur = fmaxf(cur, fabsf(acc_0[28]));
                    cur = fmaxf(cur, fabsf(acc_0[29]));
                    cur = fmaxf(cur, fabsf(acc_0[30]));
                    cur = fmaxf(cur, fabsf(acc_0[31]));
                    am = cur;
                    float acc_1[32] = {0};
                    uint32_t e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_3) : "h"((uint16_t)(fp4_wire_scales[4])));
                    float cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_4) : "h"((uint16_t)(e4m3x2_to_f16x2_3)));
                    float cvt_f32_f16_1_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_3) : "h"((uint16_t)(e4m3x2_to_f16x2_3 >> 16)));
                    float s0_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_3) : "f"(cvt_f32_f16_4), "f"(g_s_0));
                    float s1_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_3) : "f"(cvt_f32_f16_1_3), "f"(g_s_0));
                    float2 scale0_3 = make_float2(s0_3, s0_3);
                    float2 scale1_3 = make_float2(s1_3, s1_3);
                    float tv_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[16 + 0]));
                        (&tv_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_16[0]), "=f"(tv_16[1]), "=f"(tv_16[2]), "=f"(tv_16[3]), "=f"(tv_16[4]), "=f"(tv_16[5]), "=f"(tv_16[6]), "=f"(tv_16[7]) : "r"(fp4_wire_values[16 + 0]));
                        #endif
                    }
                    float2 a_64 = make_float2(acc_1[0], acc_1[1]);
                    float2 q_64 = make_float2(tv_16[0], tv_16[1]);
                    float2 r_64 = fma_f32x2_rn_noftz(q_64, scale0_3, a_64);
                    acc_1[0] = r_64.x;
                    acc_1[1] = r_64.y;
                    float2 a_65 = make_float2(acc_1[2], acc_1[3]);
                    float2 q_65 = make_float2(tv_16[2], tv_16[3]);
                    float2 r_65 = fma_f32x2_rn_noftz(q_65, scale0_3, a_65);
                    acc_1[2] = r_65.x;
                    acc_1[3] = r_65.y;
                    float2 a_66 = make_float2(acc_1[4], acc_1[5]);
                    float2 q_66 = make_float2(tv_16[4], tv_16[5]);
                    float2 r_66 = fma_f32x2_rn_noftz(q_66, scale0_3, a_66);
                    acc_1[4] = r_66.x;
                    acc_1[5] = r_66.y;
                    float2 a_67 = make_float2(acc_1[6], acc_1[7]);
                    float2 q_67 = make_float2(tv_16[6], tv_16[7]);
                    float2 r_67 = fma_f32x2_rn_noftz(q_67, scale0_3, a_67);
                    acc_1[6] = r_67.x;
                    acc_1[7] = r_67.y;
                    float tv_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[17 + 0]));
                        (&tv_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_17[0]), "=f"(tv_17[1]), "=f"(tv_17[2]), "=f"(tv_17[3]), "=f"(tv_17[4]), "=f"(tv_17[5]), "=f"(tv_17[6]), "=f"(tv_17[7]) : "r"(fp4_wire_values[17 + 0]));
                        #endif
                    }
                    float2 a_68 = make_float2(acc_1[8], acc_1[9]);
                    float2 q_68 = make_float2(tv_17[0], tv_17[1]);
                    float2 r_68 = fma_f32x2_rn_noftz(q_68, scale0_3, a_68);
                    acc_1[8] = r_68.x;
                    acc_1[9] = r_68.y;
                    float2 a_69 = make_float2(acc_1[10], acc_1[11]);
                    float2 q_69 = make_float2(tv_17[2], tv_17[3]);
                    float2 r_69 = fma_f32x2_rn_noftz(q_69, scale0_3, a_69);
                    acc_1[10] = r_69.x;
                    acc_1[11] = r_69.y;
                    float2 a_70 = make_float2(acc_1[12], acc_1[13]);
                    float2 q_70 = make_float2(tv_17[4], tv_17[5]);
                    float2 r_70 = fma_f32x2_rn_noftz(q_70, scale0_3, a_70);
                    acc_1[12] = r_70.x;
                    acc_1[13] = r_70.y;
                    float2 a_71 = make_float2(acc_1[14], acc_1[15]);
                    float2 q_71 = make_float2(tv_17[6], tv_17[7]);
                    float2 r_71 = fma_f32x2_rn_noftz(q_71, scale0_3, a_71);
                    acc_1[14] = r_71.x;
                    acc_1[15] = r_71.y;
                    float tv_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[18 + 0]));
                        (&tv_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_18[0]), "=f"(tv_18[1]), "=f"(tv_18[2]), "=f"(tv_18[3]), "=f"(tv_18[4]), "=f"(tv_18[5]), "=f"(tv_18[6]), "=f"(tv_18[7]) : "r"(fp4_wire_values[18 + 0]));
                        #endif
                    }
                    float2 a_72 = make_float2(acc_1[16], acc_1[17]);
                    float2 q_72 = make_float2(tv_18[0], tv_18[1]);
                    float2 r_72 = fma_f32x2_rn_noftz(q_72, scale1_3, a_72);
                    acc_1[16] = r_72.x;
                    acc_1[17] = r_72.y;
                    float2 a_73 = make_float2(acc_1[18], acc_1[19]);
                    float2 q_73 = make_float2(tv_18[2], tv_18[3]);
                    float2 r_73 = fma_f32x2_rn_noftz(q_73, scale1_3, a_73);
                    acc_1[18] = r_73.x;
                    acc_1[19] = r_73.y;
                    float2 a_74 = make_float2(acc_1[20], acc_1[21]);
                    float2 q_74 = make_float2(tv_18[4], tv_18[5]);
                    float2 r_74 = fma_f32x2_rn_noftz(q_74, scale1_3, a_74);
                    acc_1[20] = r_74.x;
                    acc_1[21] = r_74.y;
                    float2 a_75 = make_float2(acc_1[22], acc_1[23]);
                    float2 q_75 = make_float2(tv_18[6], tv_18[7]);
                    float2 r_75 = fma_f32x2_rn_noftz(q_75, scale1_3, a_75);
                    acc_1[22] = r_75.x;
                    acc_1[23] = r_75.y;
                    float tv_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[19 + 0]));
                        (&tv_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_19[0]), "=f"(tv_19[1]), "=f"(tv_19[2]), "=f"(tv_19[3]), "=f"(tv_19[4]), "=f"(tv_19[5]), "=f"(tv_19[6]), "=f"(tv_19[7]) : "r"(fp4_wire_values[19 + 0]));
                        #endif
                    }
                    float2 a_76 = make_float2(acc_1[24], acc_1[25]);
                    float2 q_76 = make_float2(tv_19[0], tv_19[1]);
                    float2 r_76 = fma_f32x2_rn_noftz(q_76, scale1_3, a_76);
                    acc_1[24] = r_76.x;
                    acc_1[25] = r_76.y;
                    float2 a_77 = make_float2(acc_1[26], acc_1[27]);
                    float2 q_77 = make_float2(tv_19[2], tv_19[3]);
                    float2 r_77 = fma_f32x2_rn_noftz(q_77, scale1_3, a_77);
                    acc_1[26] = r_77.x;
                    acc_1[27] = r_77.y;
                    float2 a_78 = make_float2(acc_1[28], acc_1[29]);
                    float2 q_78 = make_float2(tv_19[4], tv_19[5]);
                    float2 r_78 = fma_f32x2_rn_noftz(q_78, scale1_3, a_78);
                    acc_1[28] = r_78.x;
                    acc_1[29] = r_78.y;
                    float2 a_79 = make_float2(acc_1[30], acc_1[31]);
                    float2 q_79 = make_float2(tv_19[6], tv_19[7]);
                    float2 r_79 = fma_f32x2_rn_noftz(q_79, scale1_3, a_79);
                    acc_1[30] = r_79.x;
                    acc_1[31] = r_79.y;
                    uint32_t e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_4) : "h"((uint16_t)(fp4_wire_scales[5])));
                    float cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_5) : "h"((uint16_t)(e4m3x2_to_f16x2_4)));
                    float cvt_f32_f16_1_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_4) : "h"((uint16_t)(e4m3x2_to_f16x2_4 >> 16)));
                    float s0_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_4) : "f"(cvt_f32_f16_5), "f"(g_s_1));
                    float s1_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_4) : "f"(cvt_f32_f16_1_4), "f"(g_s_1));
                    float2 scale0_4 = make_float2(s0_4, s0_4);
                    float2 scale1_4 = make_float2(s1_4, s1_4);
                    float tv_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[20 + 0]));
                        (&tv_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_20[0]), "=f"(tv_20[1]), "=f"(tv_20[2]), "=f"(tv_20[3]), "=f"(tv_20[4]), "=f"(tv_20[5]), "=f"(tv_20[6]), "=f"(tv_20[7]) : "r"(fp4_wire_values[20 + 0]));
                        #endif
                    }
                    float2 a_80 = make_float2(acc_1[0], acc_1[1]);
                    float2 q_80 = make_float2(tv_20[0], tv_20[1]);
                    float2 r_80 = fma_f32x2_rn_noftz(q_80, scale0_4, a_80);
                    acc_1[0] = r_80.x;
                    acc_1[1] = r_80.y;
                    float2 a_81 = make_float2(acc_1[2], acc_1[3]);
                    float2 q_81 = make_float2(tv_20[2], tv_20[3]);
                    float2 r_81 = fma_f32x2_rn_noftz(q_81, scale0_4, a_81);
                    acc_1[2] = r_81.x;
                    acc_1[3] = r_81.y;
                    float2 a_82 = make_float2(acc_1[4], acc_1[5]);
                    float2 q_82 = make_float2(tv_20[4], tv_20[5]);
                    float2 r_82 = fma_f32x2_rn_noftz(q_82, scale0_4, a_82);
                    acc_1[4] = r_82.x;
                    acc_1[5] = r_82.y;
                    float2 a_83 = make_float2(acc_1[6], acc_1[7]);
                    float2 q_83 = make_float2(tv_20[6], tv_20[7]);
                    float2 r_83 = fma_f32x2_rn_noftz(q_83, scale0_4, a_83);
                    acc_1[6] = r_83.x;
                    acc_1[7] = r_83.y;
                    float tv_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[21 + 0]));
                        (&tv_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_21[0]), "=f"(tv_21[1]), "=f"(tv_21[2]), "=f"(tv_21[3]), "=f"(tv_21[4]), "=f"(tv_21[5]), "=f"(tv_21[6]), "=f"(tv_21[7]) : "r"(fp4_wire_values[21 + 0]));
                        #endif
                    }
                    float2 a_84 = make_float2(acc_1[8], acc_1[9]);
                    float2 q_84 = make_float2(tv_21[0], tv_21[1]);
                    float2 r_84 = fma_f32x2_rn_noftz(q_84, scale0_4, a_84);
                    acc_1[8] = r_84.x;
                    acc_1[9] = r_84.y;
                    float2 a_85 = make_float2(acc_1[10], acc_1[11]);
                    float2 q_85 = make_float2(tv_21[2], tv_21[3]);
                    float2 r_85 = fma_f32x2_rn_noftz(q_85, scale0_4, a_85);
                    acc_1[10] = r_85.x;
                    acc_1[11] = r_85.y;
                    float2 a_86 = make_float2(acc_1[12], acc_1[13]);
                    float2 q_86 = make_float2(tv_21[4], tv_21[5]);
                    float2 r_86 = fma_f32x2_rn_noftz(q_86, scale0_4, a_86);
                    acc_1[12] = r_86.x;
                    acc_1[13] = r_86.y;
                    float2 a_87 = make_float2(acc_1[14], acc_1[15]);
                    float2 q_87 = make_float2(tv_21[6], tv_21[7]);
                    float2 r_87 = fma_f32x2_rn_noftz(q_87, scale0_4, a_87);
                    acc_1[14] = r_87.x;
                    acc_1[15] = r_87.y;
                    float tv_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[22 + 0]));
                        (&tv_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_22[0]), "=f"(tv_22[1]), "=f"(tv_22[2]), "=f"(tv_22[3]), "=f"(tv_22[4]), "=f"(tv_22[5]), "=f"(tv_22[6]), "=f"(tv_22[7]) : "r"(fp4_wire_values[22 + 0]));
                        #endif
                    }
                    float2 a_88 = make_float2(acc_1[16], acc_1[17]);
                    float2 q_88 = make_float2(tv_22[0], tv_22[1]);
                    float2 r_88 = fma_f32x2_rn_noftz(q_88, scale1_4, a_88);
                    acc_1[16] = r_88.x;
                    acc_1[17] = r_88.y;
                    float2 a_89 = make_float2(acc_1[18], acc_1[19]);
                    float2 q_89 = make_float2(tv_22[2], tv_22[3]);
                    float2 r_89 = fma_f32x2_rn_noftz(q_89, scale1_4, a_89);
                    acc_1[18] = r_89.x;
                    acc_1[19] = r_89.y;
                    float2 a_90 = make_float2(acc_1[20], acc_1[21]);
                    float2 q_90 = make_float2(tv_22[4], tv_22[5]);
                    float2 r_90 = fma_f32x2_rn_noftz(q_90, scale1_4, a_90);
                    acc_1[20] = r_90.x;
                    acc_1[21] = r_90.y;
                    float2 a_91 = make_float2(acc_1[22], acc_1[23]);
                    float2 q_91 = make_float2(tv_22[6], tv_22[7]);
                    float2 r_91 = fma_f32x2_rn_noftz(q_91, scale1_4, a_91);
                    acc_1[22] = r_91.x;
                    acc_1[23] = r_91.y;
                    float tv_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[23 + 0]));
                        (&tv_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_23[0]), "=f"(tv_23[1]), "=f"(tv_23[2]), "=f"(tv_23[3]), "=f"(tv_23[4]), "=f"(tv_23[5]), "=f"(tv_23[6]), "=f"(tv_23[7]) : "r"(fp4_wire_values[23 + 0]));
                        #endif
                    }
                    float2 a_92 = make_float2(acc_1[24], acc_1[25]);
                    float2 q_92 = make_float2(tv_23[0], tv_23[1]);
                    float2 r_92 = fma_f32x2_rn_noftz(q_92, scale1_4, a_92);
                    acc_1[24] = r_92.x;
                    acc_1[25] = r_92.y;
                    float2 a_93 = make_float2(acc_1[26], acc_1[27]);
                    float2 q_93 = make_float2(tv_23[2], tv_23[3]);
                    float2 r_93 = fma_f32x2_rn_noftz(q_93, scale1_4, a_93);
                    acc_1[26] = r_93.x;
                    acc_1[27] = r_93.y;
                    float2 a_94 = make_float2(acc_1[28], acc_1[29]);
                    float2 q_94 = make_float2(tv_23[4], tv_23[5]);
                    float2 r_94 = fma_f32x2_rn_noftz(q_94, scale1_4, a_94);
                    acc_1[28] = r_94.x;
                    acc_1[29] = r_94.y;
                    float2 a_95 = make_float2(acc_1[30], acc_1[31]);
                    float2 q_95 = make_float2(tv_23[6], tv_23[7]);
                    float2 r_95 = fma_f32x2_rn_noftz(q_95, scale1_4, a_95);
                    acc_1[30] = r_95.x;
                    acc_1[31] = r_95.y;
                    uint32_t e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_5) : "h"((uint16_t)(fp4_wire_scales[6])));
                    float cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_6) : "h"((uint16_t)(e4m3x2_to_f16x2_5)));
                    float cvt_f32_f16_1_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_5) : "h"((uint16_t)(e4m3x2_to_f16x2_5 >> 16)));
                    float s0_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_5) : "f"(cvt_f32_f16_6), "f"(g_s_2));
                    float s1_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_5) : "f"(cvt_f32_f16_1_5), "f"(g_s_2));
                    float2 scale0_5 = make_float2(s0_5, s0_5);
                    float2 scale1_5 = make_float2(s1_5, s1_5);
                    float tv_24[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[24 + 0]));
                        (&tv_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_24[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_24[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_24[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_24[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_24[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_24[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_24[0]), "=f"(tv_24[1]), "=f"(tv_24[2]), "=f"(tv_24[3]), "=f"(tv_24[4]), "=f"(tv_24[5]), "=f"(tv_24[6]), "=f"(tv_24[7]) : "r"(fp4_wire_values[24 + 0]));
                        #endif
                    }
                    float2 a_96 = make_float2(acc_1[0], acc_1[1]);
                    float2 q_96 = make_float2(tv_24[0], tv_24[1]);
                    float2 r_96 = fma_f32x2_rn_noftz(q_96, scale0_5, a_96);
                    acc_1[0] = r_96.x;
                    acc_1[1] = r_96.y;
                    float2 a_97 = make_float2(acc_1[2], acc_1[3]);
                    float2 q_97 = make_float2(tv_24[2], tv_24[3]);
                    float2 r_97 = fma_f32x2_rn_noftz(q_97, scale0_5, a_97);
                    acc_1[2] = r_97.x;
                    acc_1[3] = r_97.y;
                    float2 a_98 = make_float2(acc_1[4], acc_1[5]);
                    float2 q_98 = make_float2(tv_24[4], tv_24[5]);
                    float2 r_98 = fma_f32x2_rn_noftz(q_98, scale0_5, a_98);
                    acc_1[4] = r_98.x;
                    acc_1[5] = r_98.y;
                    float2 a_99 = make_float2(acc_1[6], acc_1[7]);
                    float2 q_99 = make_float2(tv_24[6], tv_24[7]);
                    float2 r_99 = fma_f32x2_rn_noftz(q_99, scale0_5, a_99);
                    acc_1[6] = r_99.x;
                    acc_1[7] = r_99.y;
                    float tv_25[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[25 + 0]));
                        (&tv_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_25[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_25[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_25[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_25[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_25[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_25[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_25[0]), "=f"(tv_25[1]), "=f"(tv_25[2]), "=f"(tv_25[3]), "=f"(tv_25[4]), "=f"(tv_25[5]), "=f"(tv_25[6]), "=f"(tv_25[7]) : "r"(fp4_wire_values[25 + 0]));
                        #endif
                    }
                    float2 a_100 = make_float2(acc_1[8], acc_1[9]);
                    float2 q_100 = make_float2(tv_25[0], tv_25[1]);
                    float2 r_100 = fma_f32x2_rn_noftz(q_100, scale0_5, a_100);
                    acc_1[8] = r_100.x;
                    acc_1[9] = r_100.y;
                    float2 a_101 = make_float2(acc_1[10], acc_1[11]);
                    float2 q_101 = make_float2(tv_25[2], tv_25[3]);
                    float2 r_101 = fma_f32x2_rn_noftz(q_101, scale0_5, a_101);
                    acc_1[10] = r_101.x;
                    acc_1[11] = r_101.y;
                    float2 a_102 = make_float2(acc_1[12], acc_1[13]);
                    float2 q_102 = make_float2(tv_25[4], tv_25[5]);
                    float2 r_102 = fma_f32x2_rn_noftz(q_102, scale0_5, a_102);
                    acc_1[12] = r_102.x;
                    acc_1[13] = r_102.y;
                    float2 a_103 = make_float2(acc_1[14], acc_1[15]);
                    float2 q_103 = make_float2(tv_25[6], tv_25[7]);
                    float2 r_103 = fma_f32x2_rn_noftz(q_103, scale0_5, a_103);
                    acc_1[14] = r_103.x;
                    acc_1[15] = r_103.y;
                    float tv_26[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[26 + 0]));
                        (&tv_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_26[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_26[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_26[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_26[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_26[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_26[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_26[0]), "=f"(tv_26[1]), "=f"(tv_26[2]), "=f"(tv_26[3]), "=f"(tv_26[4]), "=f"(tv_26[5]), "=f"(tv_26[6]), "=f"(tv_26[7]) : "r"(fp4_wire_values[26 + 0]));
                        #endif
                    }
                    float2 a_104 = make_float2(acc_1[16], acc_1[17]);
                    float2 q_104 = make_float2(tv_26[0], tv_26[1]);
                    float2 r_104 = fma_f32x2_rn_noftz(q_104, scale1_5, a_104);
                    acc_1[16] = r_104.x;
                    acc_1[17] = r_104.y;
                    float2 a_105 = make_float2(acc_1[18], acc_1[19]);
                    float2 q_105 = make_float2(tv_26[2], tv_26[3]);
                    float2 r_105 = fma_f32x2_rn_noftz(q_105, scale1_5, a_105);
                    acc_1[18] = r_105.x;
                    acc_1[19] = r_105.y;
                    float2 a_106 = make_float2(acc_1[20], acc_1[21]);
                    float2 q_106 = make_float2(tv_26[4], tv_26[5]);
                    float2 r_106 = fma_f32x2_rn_noftz(q_106, scale1_5, a_106);
                    acc_1[20] = r_106.x;
                    acc_1[21] = r_106.y;
                    float2 a_107 = make_float2(acc_1[22], acc_1[23]);
                    float2 q_107 = make_float2(tv_26[6], tv_26[7]);
                    float2 r_107 = fma_f32x2_rn_noftz(q_107, scale1_5, a_107);
                    acc_1[22] = r_107.x;
                    acc_1[23] = r_107.y;
                    float tv_27[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[27 + 0]));
                        (&tv_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_27[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_27[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_27[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_27[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_27[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_27[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_27[0]), "=f"(tv_27[1]), "=f"(tv_27[2]), "=f"(tv_27[3]), "=f"(tv_27[4]), "=f"(tv_27[5]), "=f"(tv_27[6]), "=f"(tv_27[7]) : "r"(fp4_wire_values[27 + 0]));
                        #endif
                    }
                    float2 a_108 = make_float2(acc_1[24], acc_1[25]);
                    float2 q_108 = make_float2(tv_27[0], tv_27[1]);
                    float2 r_108 = fma_f32x2_rn_noftz(q_108, scale1_5, a_108);
                    acc_1[24] = r_108.x;
                    acc_1[25] = r_108.y;
                    float2 a_109 = make_float2(acc_1[26], acc_1[27]);
                    float2 q_109 = make_float2(tv_27[2], tv_27[3]);
                    float2 r_109 = fma_f32x2_rn_noftz(q_109, scale1_5, a_109);
                    acc_1[26] = r_109.x;
                    acc_1[27] = r_109.y;
                    float2 a_110 = make_float2(acc_1[28], acc_1[29]);
                    float2 q_110 = make_float2(tv_27[4], tv_27[5]);
                    float2 r_110 = fma_f32x2_rn_noftz(q_110, scale1_5, a_110);
                    acc_1[28] = r_110.x;
                    acc_1[29] = r_110.y;
                    float2 a_111 = make_float2(acc_1[30], acc_1[31]);
                    float2 q_111 = make_float2(tv_27[6], tv_27[7]);
                    float2 r_111 = fma_f32x2_rn_noftz(q_111, scale1_5, a_111);
                    acc_1[30] = r_111.x;
                    acc_1[31] = r_111.y;
                    uint32_t e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_6) : "h"((uint16_t)(fp4_wire_scales[7])));
                    float cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_7) : "h"((uint16_t)(e4m3x2_to_f16x2_6)));
                    float cvt_f32_f16_1_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_6) : "h"((uint16_t)(e4m3x2_to_f16x2_6 >> 16)));
                    float s0_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_6) : "f"(cvt_f32_f16_7), "f"(g_s_3));
                    float s1_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_6) : "f"(cvt_f32_f16_1_6), "f"(g_s_3));
                    float2 scale0_6 = make_float2(s0_6, s0_6);
                    float2 scale1_6 = make_float2(s1_6, s1_6);
                    float tv_28[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[28 + 0]));
                        (&tv_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_28[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_28[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_28[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_28[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_28[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_28[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_28[0]), "=f"(tv_28[1]), "=f"(tv_28[2]), "=f"(tv_28[3]), "=f"(tv_28[4]), "=f"(tv_28[5]), "=f"(tv_28[6]), "=f"(tv_28[7]) : "r"(fp4_wire_values[28 + 0]));
                        #endif
                    }
                    float2 a_112 = make_float2(acc_1[0], acc_1[1]);
                    float2 q_112 = make_float2(tv_28[0], tv_28[1]);
                    float2 r_112 = fma_f32x2_rn_noftz(q_112, scale0_6, a_112);
                    acc_1[0] = r_112.x;
                    acc_1[1] = r_112.y;
                    float2 a_113 = make_float2(acc_1[2], acc_1[3]);
                    float2 q_113 = make_float2(tv_28[2], tv_28[3]);
                    float2 r_113 = fma_f32x2_rn_noftz(q_113, scale0_6, a_113);
                    acc_1[2] = r_113.x;
                    acc_1[3] = r_113.y;
                    float2 a_114 = make_float2(acc_1[4], acc_1[5]);
                    float2 q_114 = make_float2(tv_28[4], tv_28[5]);
                    float2 r_114 = fma_f32x2_rn_noftz(q_114, scale0_6, a_114);
                    acc_1[4] = r_114.x;
                    acc_1[5] = r_114.y;
                    float2 a_115 = make_float2(acc_1[6], acc_1[7]);
                    float2 q_115 = make_float2(tv_28[6], tv_28[7]);
                    float2 r_115 = fma_f32x2_rn_noftz(q_115, scale0_6, a_115);
                    acc_1[6] = r_115.x;
                    acc_1[7] = r_115.y;
                    float tv_29[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[29 + 0]));
                        (&tv_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_29[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_29[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_29[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_29[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_29[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_29[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_29[0]), "=f"(tv_29[1]), "=f"(tv_29[2]), "=f"(tv_29[3]), "=f"(tv_29[4]), "=f"(tv_29[5]), "=f"(tv_29[6]), "=f"(tv_29[7]) : "r"(fp4_wire_values[29 + 0]));
                        #endif
                    }
                    float2 a_116 = make_float2(acc_1[8], acc_1[9]);
                    float2 q_116 = make_float2(tv_29[0], tv_29[1]);
                    float2 r_116 = fma_f32x2_rn_noftz(q_116, scale0_6, a_116);
                    acc_1[8] = r_116.x;
                    acc_1[9] = r_116.y;
                    float2 a_117 = make_float2(acc_1[10], acc_1[11]);
                    float2 q_117 = make_float2(tv_29[2], tv_29[3]);
                    float2 r_117 = fma_f32x2_rn_noftz(q_117, scale0_6, a_117);
                    acc_1[10] = r_117.x;
                    acc_1[11] = r_117.y;
                    float2 a_118 = make_float2(acc_1[12], acc_1[13]);
                    float2 q_118 = make_float2(tv_29[4], tv_29[5]);
                    float2 r_118 = fma_f32x2_rn_noftz(q_118, scale0_6, a_118);
                    acc_1[12] = r_118.x;
                    acc_1[13] = r_118.y;
                    float2 a_119 = make_float2(acc_1[14], acc_1[15]);
                    float2 q_119 = make_float2(tv_29[6], tv_29[7]);
                    float2 r_119 = fma_f32x2_rn_noftz(q_119, scale0_6, a_119);
                    acc_1[14] = r_119.x;
                    acc_1[15] = r_119.y;
                    float tv_30[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[30 + 0]));
                        (&tv_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_30[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_30[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_30[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_30[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_30[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_30[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_30[0]), "=f"(tv_30[1]), "=f"(tv_30[2]), "=f"(tv_30[3]), "=f"(tv_30[4]), "=f"(tv_30[5]), "=f"(tv_30[6]), "=f"(tv_30[7]) : "r"(fp4_wire_values[30 + 0]));
                        #endif
                    }
                    float2 a_120 = make_float2(acc_1[16], acc_1[17]);
                    float2 q_120 = make_float2(tv_30[0], tv_30[1]);
                    float2 r_120 = fma_f32x2_rn_noftz(q_120, scale1_6, a_120);
                    acc_1[16] = r_120.x;
                    acc_1[17] = r_120.y;
                    float2 a_121 = make_float2(acc_1[18], acc_1[19]);
                    float2 q_121 = make_float2(tv_30[2], tv_30[3]);
                    float2 r_121 = fma_f32x2_rn_noftz(q_121, scale1_6, a_121);
                    acc_1[18] = r_121.x;
                    acc_1[19] = r_121.y;
                    float2 a_122 = make_float2(acc_1[20], acc_1[21]);
                    float2 q_122 = make_float2(tv_30[4], tv_30[5]);
                    float2 r_122 = fma_f32x2_rn_noftz(q_122, scale1_6, a_122);
                    acc_1[20] = r_122.x;
                    acc_1[21] = r_122.y;
                    float2 a_123 = make_float2(acc_1[22], acc_1[23]);
                    float2 q_123 = make_float2(tv_30[6], tv_30[7]);
                    float2 r_123 = fma_f32x2_rn_noftz(q_123, scale1_6, a_123);
                    acc_1[22] = r_123.x;
                    acc_1[23] = r_123.y;
                    float tv_31[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[31 + 0]));
                        (&tv_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_31[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_31[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_31[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_31[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_31[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_31[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_31[0]), "=f"(tv_31[1]), "=f"(tv_31[2]), "=f"(tv_31[3]), "=f"(tv_31[4]), "=f"(tv_31[5]), "=f"(tv_31[6]), "=f"(tv_31[7]) : "r"(fp4_wire_values[31 + 0]));
                        #endif
                    }
                    float2 a_124 = make_float2(acc_1[24], acc_1[25]);
                    float2 q_124 = make_float2(tv_31[0], tv_31[1]);
                    float2 r_124 = fma_f32x2_rn_noftz(q_124, scale1_6, a_124);
                    acc_1[24] = r_124.x;
                    acc_1[25] = r_124.y;
                    float2 a_125 = make_float2(acc_1[26], acc_1[27]);
                    float2 q_125 = make_float2(tv_31[2], tv_31[3]);
                    float2 r_125 = fma_f32x2_rn_noftz(q_125, scale1_6, a_125);
                    acc_1[26] = r_125.x;
                    acc_1[27] = r_125.y;
                    float2 a_126 = make_float2(acc_1[28], acc_1[29]);
                    float2 q_126 = make_float2(tv_31[4], tv_31[5]);
                    float2 r_126 = fma_f32x2_rn_noftz(q_126, scale1_6, a_126);
                    acc_1[28] = r_126.x;
                    acc_1[29] = r_126.y;
                    float2 a_127 = make_float2(acc_1[30], acc_1[31]);
                    float2 q_127 = make_float2(tv_31[6], tv_31[7]);
                    float2 r_127 = fma_f32x2_rn_noftz(q_127, scale1_6, a_127);
                    acc_1[30] = r_127.x;
                    acc_1[31] = r_127.y;
                    float cur_0 = am;
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[0]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[1]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[2]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[3]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[4]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[5]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[6]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[7]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[8]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[9]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[10]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[11]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[12]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[13]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[14]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[15]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[16]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[17]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[18]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[19]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[20]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[21]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[22]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[23]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[24]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[25]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[26]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[27]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[28]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[29]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[30]));
                    cur_0 = fmaxf(cur_0, fabsf(acc_1[31]));
                    am = cur_0;
                }
                float rs2 = 1.0f;
                unsigned int sw[4];
                unsigned int vec_load[1];
                {
                    uint32_t _scalar_bits_3;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc + (int)(4 * (lane_1 >> 1)))) + (0))) : "memory");
                    vec_load[0] = (unsigned int)_scalar_bits_3;
                }
                sw[0] = vec_load[0];
                unsigned int vec_load_0[1];
                {
                    uint32_t _scalar_bits_4;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_4) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_1 + (int)(4 * (lane_1 >> 1)))) + (0))) : "memory");
                    vec_load_0[0] = (unsigned int)_scalar_bits_4;
                }
                sw[1] = vec_load_0[0];
                unsigned int vec_load_1[1];
                {
                    uint32_t _scalar_bits_5;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_5) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_2 + (int)(4 * (lane_1 >> 1)))) + (0))) : "memory");
                    vec_load_1[0] = (unsigned int)_scalar_bits_5;
                }
                sw[2] = vec_load_1[0];
                unsigned int vec_load_2[1];
                {
                    uint32_t _scalar_bits_6;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_6) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_3 + (int)(4 * (lane_1 >> 1)))) + (0))) : "memory");
                    vec_load_2[0] = (unsigned int)_scalar_bits_6;
                }
                sw[3] = vec_load_2[0];
                float _warp_reduce_7 = am;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_7 = max_noftz(_warp_reduce_7, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset));
                am = _warp_reduce_7;
                float fdiv_rn = __fdiv_rn(2688.0f, am);
                float inv = 0.0f;
                if (am > 0.0f) {
                    inv = fdiv_rn;
                }
                float rs = __fdiv_rn(am, 2688.0f);
                rs2 = rs;
                #pragma unroll 1
                for (int j_0 = 0; j_0 < 6; j_0++) {
                    int pp_0 = lane_1 + 32 * j_0;
                    unsigned int c_0 = sw[0] >> (unsigned int)(16 * (pp_0 & 1)) & 65535;
                    unsigned int c_1 = sw[1] >> (unsigned int)(16 * (pp_0 & 1)) & 65535;
                    unsigned int c_2 = sw[2] >> (unsigned int)(16 * (pp_0 & 1)) & 65535;
                    unsigned int c_3 = sw[3] >> (unsigned int)(16 * (pp_0 & 1)) & 65535;
                    if (j_0 + 1 < 6) {
                        unsigned int vec_load_3[1];
                        {
                            uint32_t _scalar_bits_8;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_8) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc + (int)(4 * (pp_0 + 32 >> 1)))) + (0))) : "memory");
                            vec_load_3[0] = (unsigned int)_scalar_bits_8;
                        }
                        sw[0] = vec_load_3[0];
                        unsigned int vec_load_4[1];
                        {
                            uint32_t _scalar_bits_9;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_9) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_1 + (int)(4 * (pp_0 + 32 >> 1)))) + (0))) : "memory");
                            vec_load_4[0] = (unsigned int)_scalar_bits_9;
                        }
                        sw[1] = vec_load_4[0];
                        unsigned int vec_load_5[1];
                        {
                            uint32_t _scalar_bits_10;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_10) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_2 + (int)(4 * (pp_0 + 32 >> 1)))) + (0))) : "memory");
                            vec_load_5[0] = (unsigned int)_scalar_bits_10;
                        }
                        sw[2] = vec_load_5[0];
                        unsigned int vec_load_6[1];
                        {
                            uint32_t _scalar_bits_11;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_11) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_3 + (int)(4 * (pp_0 + 32 >> 1)))) + (0))) : "memory");
                            vec_load_6[0] = (unsigned int)_scalar_bits_11;
                        }
                        sw[3] = vec_load_6[0];
                    }
                    int rbase_0 = ret_lane + 2048 * j_0;
                    unsigned int w4_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)rbase_0));
                    unsigned int w4_7[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rbase_0 + 512)));
                    unsigned int w4_8[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rbase_0 + 1024)));
                    unsigned int w4_9[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rbase_0 + 1536)));
                    float acc2_0[32] = {0};
                    uint32_t e4m3x2_to_f16x2_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_7) : "h"((uint16_t)(c_0)));
                    float cvt_f32_f16_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_8) : "h"((uint16_t)(e4m3x2_to_f16x2_7)));
                    float cvt_f32_f16_1_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_7) : "h"((uint16_t)(e4m3x2_to_f16x2_7 >> 16)));
                    float s0_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_7) : "f"(cvt_f32_f16_8), "f"(g_s_0));
                    float s1_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_7) : "f"(cvt_f32_f16_1_7), "f"(g_s_0));
                    float2 scale0_7 = make_float2(s0_7, s0_7);
                    float2 scale1_7 = make_float2(s1_7, s1_7);
                    float tv_32[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_6[0]));
                        (&tv_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_32[0]), "=f"(tv_32[1]), "=f"(tv_32[2]), "=f"(tv_32[3]), "=f"(tv_32[4]), "=f"(tv_32[5]), "=f"(tv_32[6]), "=f"(tv_32[7]) : "r"(w4_6[0]));
                        #endif
                    }
                    float2 a_128 = make_float2(acc2_0[0], acc2_0[1]);
                    float2 q_128 = make_float2(tv_32[0], tv_32[1]);
                    float2 r_128 = fma_f32x2_rn_noftz(q_128, scale0_7, a_128);
                    acc2_0[0] = r_128.x;
                    acc2_0[1] = r_128.y;
                    float2 a_129 = make_float2(acc2_0[2], acc2_0[3]);
                    float2 q_129 = make_float2(tv_32[2], tv_32[3]);
                    float2 r_129 = fma_f32x2_rn_noftz(q_129, scale0_7, a_129);
                    acc2_0[2] = r_129.x;
                    acc2_0[3] = r_129.y;
                    float2 a_130 = make_float2(acc2_0[4], acc2_0[5]);
                    float2 q_130 = make_float2(tv_32[4], tv_32[5]);
                    float2 r_130 = fma_f32x2_rn_noftz(q_130, scale0_7, a_130);
                    acc2_0[4] = r_130.x;
                    acc2_0[5] = r_130.y;
                    float2 a_131 = make_float2(acc2_0[6], acc2_0[7]);
                    float2 q_131 = make_float2(tv_32[6], tv_32[7]);
                    float2 r_131 = fma_f32x2_rn_noftz(q_131, scale0_7, a_131);
                    acc2_0[6] = r_131.x;
                    acc2_0[7] = r_131.y;
                    float tv_33[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_6[1 + 0]));
                        (&tv_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_33[0]), "=f"(tv_33[1]), "=f"(tv_33[2]), "=f"(tv_33[3]), "=f"(tv_33[4]), "=f"(tv_33[5]), "=f"(tv_33[6]), "=f"(tv_33[7]) : "r"(w4_6[1 + 0]));
                        #endif
                    }
                    float2 a_132 = make_float2(acc2_0[8], acc2_0[9]);
                    float2 q_132 = make_float2(tv_33[0], tv_33[1]);
                    float2 r_132 = fma_f32x2_rn_noftz(q_132, scale0_7, a_132);
                    acc2_0[8] = r_132.x;
                    acc2_0[9] = r_132.y;
                    float2 a_133 = make_float2(acc2_0[10], acc2_0[11]);
                    float2 q_133 = make_float2(tv_33[2], tv_33[3]);
                    float2 r_133 = fma_f32x2_rn_noftz(q_133, scale0_7, a_133);
                    acc2_0[10] = r_133.x;
                    acc2_0[11] = r_133.y;
                    float2 a_134 = make_float2(acc2_0[12], acc2_0[13]);
                    float2 q_134 = make_float2(tv_33[4], tv_33[5]);
                    float2 r_134 = fma_f32x2_rn_noftz(q_134, scale0_7, a_134);
                    acc2_0[12] = r_134.x;
                    acc2_0[13] = r_134.y;
                    float2 a_135 = make_float2(acc2_0[14], acc2_0[15]);
                    float2 q_135 = make_float2(tv_33[6], tv_33[7]);
                    float2 r_135 = fma_f32x2_rn_noftz(q_135, scale0_7, a_135);
                    acc2_0[14] = r_135.x;
                    acc2_0[15] = r_135.y;
                    float tv_34[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_6[2 + 0]));
                        (&tv_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_34[0]), "=f"(tv_34[1]), "=f"(tv_34[2]), "=f"(tv_34[3]), "=f"(tv_34[4]), "=f"(tv_34[5]), "=f"(tv_34[6]), "=f"(tv_34[7]) : "r"(w4_6[2 + 0]));
                        #endif
                    }
                    float2 a_136 = make_float2(acc2_0[16], acc2_0[17]);
                    float2 q_136 = make_float2(tv_34[0], tv_34[1]);
                    float2 r_136 = fma_f32x2_rn_noftz(q_136, scale1_7, a_136);
                    acc2_0[16] = r_136.x;
                    acc2_0[17] = r_136.y;
                    float2 a_137 = make_float2(acc2_0[18], acc2_0[19]);
                    float2 q_137 = make_float2(tv_34[2], tv_34[3]);
                    float2 r_137 = fma_f32x2_rn_noftz(q_137, scale1_7, a_137);
                    acc2_0[18] = r_137.x;
                    acc2_0[19] = r_137.y;
                    float2 a_138 = make_float2(acc2_0[20], acc2_0[21]);
                    float2 q_138 = make_float2(tv_34[4], tv_34[5]);
                    float2 r_138 = fma_f32x2_rn_noftz(q_138, scale1_7, a_138);
                    acc2_0[20] = r_138.x;
                    acc2_0[21] = r_138.y;
                    float2 a_139 = make_float2(acc2_0[22], acc2_0[23]);
                    float2 q_139 = make_float2(tv_34[6], tv_34[7]);
                    float2 r_139 = fma_f32x2_rn_noftz(q_139, scale1_7, a_139);
                    acc2_0[22] = r_139.x;
                    acc2_0[23] = r_139.y;
                    float tv_35[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_6[3 + 0]));
                        (&tv_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_35[0]), "=f"(tv_35[1]), "=f"(tv_35[2]), "=f"(tv_35[3]), "=f"(tv_35[4]), "=f"(tv_35[5]), "=f"(tv_35[6]), "=f"(tv_35[7]) : "r"(w4_6[3 + 0]));
                        #endif
                    }
                    float2 a_140 = make_float2(acc2_0[24], acc2_0[25]);
                    float2 q_140 = make_float2(tv_35[0], tv_35[1]);
                    float2 r_140 = fma_f32x2_rn_noftz(q_140, scale1_7, a_140);
                    acc2_0[24] = r_140.x;
                    acc2_0[25] = r_140.y;
                    float2 a_141 = make_float2(acc2_0[26], acc2_0[27]);
                    float2 q_141 = make_float2(tv_35[2], tv_35[3]);
                    float2 r_141 = fma_f32x2_rn_noftz(q_141, scale1_7, a_141);
                    acc2_0[26] = r_141.x;
                    acc2_0[27] = r_141.y;
                    float2 a_142 = make_float2(acc2_0[28], acc2_0[29]);
                    float2 q_142 = make_float2(tv_35[4], tv_35[5]);
                    float2 r_142 = fma_f32x2_rn_noftz(q_142, scale1_7, a_142);
                    acc2_0[28] = r_142.x;
                    acc2_0[29] = r_142.y;
                    float2 a_143 = make_float2(acc2_0[30], acc2_0[31]);
                    float2 q_143 = make_float2(tv_35[6], tv_35[7]);
                    float2 r_143 = fma_f32x2_rn_noftz(q_143, scale1_7, a_143);
                    acc2_0[30] = r_143.x;
                    acc2_0[31] = r_143.y;
                    uint32_t e4m3x2_to_f16x2_7_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_7_1) : "h"((uint16_t)(c_1)));
                    float cvt_f32_f16_8_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_8_1) : "h"((uint16_t)(e4m3x2_to_f16x2_7_1)));
                    float cvt_f32_f16_1_7_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_7_1) : "h"((uint16_t)(e4m3x2_to_f16x2_7_1 >> 16)));
                    float s0_7_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_7_1) : "f"(cvt_f32_f16_8_1), "f"(g_s_1));
                    float s1_7_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_7_1) : "f"(cvt_f32_f16_1_7_1), "f"(g_s_1));
                    float2 scale0_7_1 = make_float2(s0_7_1, s0_7_1);
                    float2 scale1_7_1 = make_float2(s1_7_1, s1_7_1);
                    float tv_36[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_7[0]));
                        (&tv_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_36[0]), "=f"(tv_36[1]), "=f"(tv_36[2]), "=f"(tv_36[3]), "=f"(tv_36[4]), "=f"(tv_36[5]), "=f"(tv_36[6]), "=f"(tv_36[7]) : "r"(w4_7[0]));
                        #endif
                    }
                    float2 a_144 = make_float2(acc2_0[0], acc2_0[1]);
                    float2 q_144 = make_float2(tv_36[0], tv_36[1]);
                    float2 r_144 = fma_f32x2_rn_noftz(q_144, scale0_7_1, a_144);
                    acc2_0[0] = r_144.x;
                    acc2_0[1] = r_144.y;
                    float2 a_145 = make_float2(acc2_0[2], acc2_0[3]);
                    float2 q_145 = make_float2(tv_36[2], tv_36[3]);
                    float2 r_145 = fma_f32x2_rn_noftz(q_145, scale0_7_1, a_145);
                    acc2_0[2] = r_145.x;
                    acc2_0[3] = r_145.y;
                    float2 a_146 = make_float2(acc2_0[4], acc2_0[5]);
                    float2 q_146 = make_float2(tv_36[4], tv_36[5]);
                    float2 r_146 = fma_f32x2_rn_noftz(q_146, scale0_7_1, a_146);
                    acc2_0[4] = r_146.x;
                    acc2_0[5] = r_146.y;
                    float2 a_147 = make_float2(acc2_0[6], acc2_0[7]);
                    float2 q_147 = make_float2(tv_36[6], tv_36[7]);
                    float2 r_147 = fma_f32x2_rn_noftz(q_147, scale0_7_1, a_147);
                    acc2_0[6] = r_147.x;
                    acc2_0[7] = r_147.y;
                    float tv_37[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_7[1 + 0]));
                        (&tv_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_37[0]), "=f"(tv_37[1]), "=f"(tv_37[2]), "=f"(tv_37[3]), "=f"(tv_37[4]), "=f"(tv_37[5]), "=f"(tv_37[6]), "=f"(tv_37[7]) : "r"(w4_7[1 + 0]));
                        #endif
                    }
                    float2 a_148 = make_float2(acc2_0[8], acc2_0[9]);
                    float2 q_148 = make_float2(tv_37[0], tv_37[1]);
                    float2 r_148 = fma_f32x2_rn_noftz(q_148, scale0_7_1, a_148);
                    acc2_0[8] = r_148.x;
                    acc2_0[9] = r_148.y;
                    float2 a_149 = make_float2(acc2_0[10], acc2_0[11]);
                    float2 q_149 = make_float2(tv_37[2], tv_37[3]);
                    float2 r_149 = fma_f32x2_rn_noftz(q_149, scale0_7_1, a_149);
                    acc2_0[10] = r_149.x;
                    acc2_0[11] = r_149.y;
                    float2 a_150 = make_float2(acc2_0[12], acc2_0[13]);
                    float2 q_150 = make_float2(tv_37[4], tv_37[5]);
                    float2 r_150 = fma_f32x2_rn_noftz(q_150, scale0_7_1, a_150);
                    acc2_0[12] = r_150.x;
                    acc2_0[13] = r_150.y;
                    float2 a_151 = make_float2(acc2_0[14], acc2_0[15]);
                    float2 q_151 = make_float2(tv_37[6], tv_37[7]);
                    float2 r_151 = fma_f32x2_rn_noftz(q_151, scale0_7_1, a_151);
                    acc2_0[14] = r_151.x;
                    acc2_0[15] = r_151.y;
                    float tv_38[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_7[2 + 0]));
                        (&tv_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_38[0]), "=f"(tv_38[1]), "=f"(tv_38[2]), "=f"(tv_38[3]), "=f"(tv_38[4]), "=f"(tv_38[5]), "=f"(tv_38[6]), "=f"(tv_38[7]) : "r"(w4_7[2 + 0]));
                        #endif
                    }
                    float2 a_152 = make_float2(acc2_0[16], acc2_0[17]);
                    float2 q_152 = make_float2(tv_38[0], tv_38[1]);
                    float2 r_152 = fma_f32x2_rn_noftz(q_152, scale1_7_1, a_152);
                    acc2_0[16] = r_152.x;
                    acc2_0[17] = r_152.y;
                    float2 a_153 = make_float2(acc2_0[18], acc2_0[19]);
                    float2 q_153 = make_float2(tv_38[2], tv_38[3]);
                    float2 r_153 = fma_f32x2_rn_noftz(q_153, scale1_7_1, a_153);
                    acc2_0[18] = r_153.x;
                    acc2_0[19] = r_153.y;
                    float2 a_154 = make_float2(acc2_0[20], acc2_0[21]);
                    float2 q_154 = make_float2(tv_38[4], tv_38[5]);
                    float2 r_154 = fma_f32x2_rn_noftz(q_154, scale1_7_1, a_154);
                    acc2_0[20] = r_154.x;
                    acc2_0[21] = r_154.y;
                    float2 a_155 = make_float2(acc2_0[22], acc2_0[23]);
                    float2 q_155 = make_float2(tv_38[6], tv_38[7]);
                    float2 r_155 = fma_f32x2_rn_noftz(q_155, scale1_7_1, a_155);
                    acc2_0[22] = r_155.x;
                    acc2_0[23] = r_155.y;
                    float tv_39[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_7[3 + 0]));
                        (&tv_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_39[0]), "=f"(tv_39[1]), "=f"(tv_39[2]), "=f"(tv_39[3]), "=f"(tv_39[4]), "=f"(tv_39[5]), "=f"(tv_39[6]), "=f"(tv_39[7]) : "r"(w4_7[3 + 0]));
                        #endif
                    }
                    float2 a_156 = make_float2(acc2_0[24], acc2_0[25]);
                    float2 q_156 = make_float2(tv_39[0], tv_39[1]);
                    float2 r_156 = fma_f32x2_rn_noftz(q_156, scale1_7_1, a_156);
                    acc2_0[24] = r_156.x;
                    acc2_0[25] = r_156.y;
                    float2 a_157 = make_float2(acc2_0[26], acc2_0[27]);
                    float2 q_157 = make_float2(tv_39[2], tv_39[3]);
                    float2 r_157 = fma_f32x2_rn_noftz(q_157, scale1_7_1, a_157);
                    acc2_0[26] = r_157.x;
                    acc2_0[27] = r_157.y;
                    float2 a_158 = make_float2(acc2_0[28], acc2_0[29]);
                    float2 q_158 = make_float2(tv_39[4], tv_39[5]);
                    float2 r_158 = fma_f32x2_rn_noftz(q_158, scale1_7_1, a_158);
                    acc2_0[28] = r_158.x;
                    acc2_0[29] = r_158.y;
                    float2 a_159 = make_float2(acc2_0[30], acc2_0[31]);
                    float2 q_159 = make_float2(tv_39[6], tv_39[7]);
                    float2 r_159 = fma_f32x2_rn_noftz(q_159, scale1_7_1, a_159);
                    acc2_0[30] = r_159.x;
                    acc2_0[31] = r_159.y;
                    uint32_t e4m3x2_to_f16x2_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_8) : "h"((uint16_t)(c_2)));
                    float cvt_f32_f16_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_9) : "h"((uint16_t)(e4m3x2_to_f16x2_8)));
                    float cvt_f32_f16_1_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_8) : "h"((uint16_t)(e4m3x2_to_f16x2_8 >> 16)));
                    float s0_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_8) : "f"(cvt_f32_f16_9), "f"(g_s_2));
                    float s1_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_8) : "f"(cvt_f32_f16_1_8), "f"(g_s_2));
                    float2 scale0_8 = make_float2(s0_8, s0_8);
                    float2 scale1_8 = make_float2(s1_8, s1_8);
                    float tv_40[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_8[0]));
                        (&tv_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_40[0]), "=f"(tv_40[1]), "=f"(tv_40[2]), "=f"(tv_40[3]), "=f"(tv_40[4]), "=f"(tv_40[5]), "=f"(tv_40[6]), "=f"(tv_40[7]) : "r"(w4_8[0]));
                        #endif
                    }
                    float2 a_160 = make_float2(acc2_0[0], acc2_0[1]);
                    float2 q_160 = make_float2(tv_40[0], tv_40[1]);
                    float2 r_160 = fma_f32x2_rn_noftz(q_160, scale0_8, a_160);
                    acc2_0[0] = r_160.x;
                    acc2_0[1] = r_160.y;
                    float2 a_161 = make_float2(acc2_0[2], acc2_0[3]);
                    float2 q_161 = make_float2(tv_40[2], tv_40[3]);
                    float2 r_161 = fma_f32x2_rn_noftz(q_161, scale0_8, a_161);
                    acc2_0[2] = r_161.x;
                    acc2_0[3] = r_161.y;
                    float2 a_162 = make_float2(acc2_0[4], acc2_0[5]);
                    float2 q_162 = make_float2(tv_40[4], tv_40[5]);
                    float2 r_162 = fma_f32x2_rn_noftz(q_162, scale0_8, a_162);
                    acc2_0[4] = r_162.x;
                    acc2_0[5] = r_162.y;
                    float2 a_163 = make_float2(acc2_0[6], acc2_0[7]);
                    float2 q_163 = make_float2(tv_40[6], tv_40[7]);
                    float2 r_163 = fma_f32x2_rn_noftz(q_163, scale0_8, a_163);
                    acc2_0[6] = r_163.x;
                    acc2_0[7] = r_163.y;
                    float tv_41[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_8[1 + 0]));
                        (&tv_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_41[0]), "=f"(tv_41[1]), "=f"(tv_41[2]), "=f"(tv_41[3]), "=f"(tv_41[4]), "=f"(tv_41[5]), "=f"(tv_41[6]), "=f"(tv_41[7]) : "r"(w4_8[1 + 0]));
                        #endif
                    }
                    float2 a_164 = make_float2(acc2_0[8], acc2_0[9]);
                    float2 q_164 = make_float2(tv_41[0], tv_41[1]);
                    float2 r_164 = fma_f32x2_rn_noftz(q_164, scale0_8, a_164);
                    acc2_0[8] = r_164.x;
                    acc2_0[9] = r_164.y;
                    float2 a_165 = make_float2(acc2_0[10], acc2_0[11]);
                    float2 q_165 = make_float2(tv_41[2], tv_41[3]);
                    float2 r_165 = fma_f32x2_rn_noftz(q_165, scale0_8, a_165);
                    acc2_0[10] = r_165.x;
                    acc2_0[11] = r_165.y;
                    float2 a_166 = make_float2(acc2_0[12], acc2_0[13]);
                    float2 q_166 = make_float2(tv_41[4], tv_41[5]);
                    float2 r_166 = fma_f32x2_rn_noftz(q_166, scale0_8, a_166);
                    acc2_0[12] = r_166.x;
                    acc2_0[13] = r_166.y;
                    float2 a_167 = make_float2(acc2_0[14], acc2_0[15]);
                    float2 q_167 = make_float2(tv_41[6], tv_41[7]);
                    float2 r_167 = fma_f32x2_rn_noftz(q_167, scale0_8, a_167);
                    acc2_0[14] = r_167.x;
                    acc2_0[15] = r_167.y;
                    float tv_42[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_8[2 + 0]));
                        (&tv_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_42[0]), "=f"(tv_42[1]), "=f"(tv_42[2]), "=f"(tv_42[3]), "=f"(tv_42[4]), "=f"(tv_42[5]), "=f"(tv_42[6]), "=f"(tv_42[7]) : "r"(w4_8[2 + 0]));
                        #endif
                    }
                    float2 a_168 = make_float2(acc2_0[16], acc2_0[17]);
                    float2 q_168 = make_float2(tv_42[0], tv_42[1]);
                    float2 r_168 = fma_f32x2_rn_noftz(q_168, scale1_8, a_168);
                    acc2_0[16] = r_168.x;
                    acc2_0[17] = r_168.y;
                    float2 a_169 = make_float2(acc2_0[18], acc2_0[19]);
                    float2 q_169 = make_float2(tv_42[2], tv_42[3]);
                    float2 r_169 = fma_f32x2_rn_noftz(q_169, scale1_8, a_169);
                    acc2_0[18] = r_169.x;
                    acc2_0[19] = r_169.y;
                    float2 a_170 = make_float2(acc2_0[20], acc2_0[21]);
                    float2 q_170 = make_float2(tv_42[4], tv_42[5]);
                    float2 r_170 = fma_f32x2_rn_noftz(q_170, scale1_8, a_170);
                    acc2_0[20] = r_170.x;
                    acc2_0[21] = r_170.y;
                    float2 a_171 = make_float2(acc2_0[22], acc2_0[23]);
                    float2 q_171 = make_float2(tv_42[6], tv_42[7]);
                    float2 r_171 = fma_f32x2_rn_noftz(q_171, scale1_8, a_171);
                    acc2_0[22] = r_171.x;
                    acc2_0[23] = r_171.y;
                    float tv_43[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_8[3 + 0]));
                        (&tv_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_43[0]), "=f"(tv_43[1]), "=f"(tv_43[2]), "=f"(tv_43[3]), "=f"(tv_43[4]), "=f"(tv_43[5]), "=f"(tv_43[6]), "=f"(tv_43[7]) : "r"(w4_8[3 + 0]));
                        #endif
                    }
                    float2 a_172 = make_float2(acc2_0[24], acc2_0[25]);
                    float2 q_172 = make_float2(tv_43[0], tv_43[1]);
                    float2 r_172 = fma_f32x2_rn_noftz(q_172, scale1_8, a_172);
                    acc2_0[24] = r_172.x;
                    acc2_0[25] = r_172.y;
                    float2 a_173 = make_float2(acc2_0[26], acc2_0[27]);
                    float2 q_173 = make_float2(tv_43[2], tv_43[3]);
                    float2 r_173 = fma_f32x2_rn_noftz(q_173, scale1_8, a_173);
                    acc2_0[26] = r_173.x;
                    acc2_0[27] = r_173.y;
                    float2 a_174 = make_float2(acc2_0[28], acc2_0[29]);
                    float2 q_174 = make_float2(tv_43[4], tv_43[5]);
                    float2 r_174 = fma_f32x2_rn_noftz(q_174, scale1_8, a_174);
                    acc2_0[28] = r_174.x;
                    acc2_0[29] = r_174.y;
                    float2 a_175 = make_float2(acc2_0[30], acc2_0[31]);
                    float2 q_175 = make_float2(tv_43[6], tv_43[7]);
                    float2 r_175 = fma_f32x2_rn_noftz(q_175, scale1_8, a_175);
                    acc2_0[30] = r_175.x;
                    acc2_0[31] = r_175.y;
                    uint32_t e4m3x2_to_f16x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_9) : "h"((uint16_t)(c_3)));
                    float cvt_f32_f16_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_10) : "h"((uint16_t)(e4m3x2_to_f16x2_9)));
                    float cvt_f32_f16_1_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_9) : "h"((uint16_t)(e4m3x2_to_f16x2_9 >> 16)));
                    float s0_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_9) : "f"(cvt_f32_f16_10), "f"(g_s_3));
                    float s1_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_9) : "f"(cvt_f32_f16_1_9), "f"(g_s_3));
                    float2 scale0_9 = make_float2(s0_9, s0_9);
                    float2 scale1_9 = make_float2(s1_9, s1_9);
                    float tv_44[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_9[0]));
                        (&tv_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_44[0]), "=f"(tv_44[1]), "=f"(tv_44[2]), "=f"(tv_44[3]), "=f"(tv_44[4]), "=f"(tv_44[5]), "=f"(tv_44[6]), "=f"(tv_44[7]) : "r"(w4_9[0]));
                        #endif
                    }
                    float2 a_176 = make_float2(acc2_0[0], acc2_0[1]);
                    float2 q_176 = make_float2(tv_44[0], tv_44[1]);
                    float2 r_176 = fma_f32x2_rn_noftz(q_176, scale0_9, a_176);
                    acc2_0[0] = r_176.x;
                    acc2_0[1] = r_176.y;
                    float2 a_177 = make_float2(acc2_0[2], acc2_0[3]);
                    float2 q_177 = make_float2(tv_44[2], tv_44[3]);
                    float2 r_177 = fma_f32x2_rn_noftz(q_177, scale0_9, a_177);
                    acc2_0[2] = r_177.x;
                    acc2_0[3] = r_177.y;
                    float2 a_178 = make_float2(acc2_0[4], acc2_0[5]);
                    float2 q_178 = make_float2(tv_44[4], tv_44[5]);
                    float2 r_178 = fma_f32x2_rn_noftz(q_178, scale0_9, a_178);
                    acc2_0[4] = r_178.x;
                    acc2_0[5] = r_178.y;
                    float2 a_179 = make_float2(acc2_0[6], acc2_0[7]);
                    float2 q_179 = make_float2(tv_44[6], tv_44[7]);
                    float2 r_179 = fma_f32x2_rn_noftz(q_179, scale0_9, a_179);
                    acc2_0[6] = r_179.x;
                    acc2_0[7] = r_179.y;
                    float tv_45[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_9[1 + 0]));
                        (&tv_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_45[0]), "=f"(tv_45[1]), "=f"(tv_45[2]), "=f"(tv_45[3]), "=f"(tv_45[4]), "=f"(tv_45[5]), "=f"(tv_45[6]), "=f"(tv_45[7]) : "r"(w4_9[1 + 0]));
                        #endif
                    }
                    float2 a_180 = make_float2(acc2_0[8], acc2_0[9]);
                    float2 q_180 = make_float2(tv_45[0], tv_45[1]);
                    float2 r_180 = fma_f32x2_rn_noftz(q_180, scale0_9, a_180);
                    acc2_0[8] = r_180.x;
                    acc2_0[9] = r_180.y;
                    float2 a_181 = make_float2(acc2_0[10], acc2_0[11]);
                    float2 q_181 = make_float2(tv_45[2], tv_45[3]);
                    float2 r_181 = fma_f32x2_rn_noftz(q_181, scale0_9, a_181);
                    acc2_0[10] = r_181.x;
                    acc2_0[11] = r_181.y;
                    float2 a_182 = make_float2(acc2_0[12], acc2_0[13]);
                    float2 q_182 = make_float2(tv_45[4], tv_45[5]);
                    float2 r_182 = fma_f32x2_rn_noftz(q_182, scale0_9, a_182);
                    acc2_0[12] = r_182.x;
                    acc2_0[13] = r_182.y;
                    float2 a_183 = make_float2(acc2_0[14], acc2_0[15]);
                    float2 q_183 = make_float2(tv_45[6], tv_45[7]);
                    float2 r_183 = fma_f32x2_rn_noftz(q_183, scale0_9, a_183);
                    acc2_0[14] = r_183.x;
                    acc2_0[15] = r_183.y;
                    float tv_46[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_9[2 + 0]));
                        (&tv_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_46[0]), "=f"(tv_46[1]), "=f"(tv_46[2]), "=f"(tv_46[3]), "=f"(tv_46[4]), "=f"(tv_46[5]), "=f"(tv_46[6]), "=f"(tv_46[7]) : "r"(w4_9[2 + 0]));
                        #endif
                    }
                    float2 a_184 = make_float2(acc2_0[16], acc2_0[17]);
                    float2 q_184 = make_float2(tv_46[0], tv_46[1]);
                    float2 r_184 = fma_f32x2_rn_noftz(q_184, scale1_9, a_184);
                    acc2_0[16] = r_184.x;
                    acc2_0[17] = r_184.y;
                    float2 a_185 = make_float2(acc2_0[18], acc2_0[19]);
                    float2 q_185 = make_float2(tv_46[2], tv_46[3]);
                    float2 r_185 = fma_f32x2_rn_noftz(q_185, scale1_9, a_185);
                    acc2_0[18] = r_185.x;
                    acc2_0[19] = r_185.y;
                    float2 a_186 = make_float2(acc2_0[20], acc2_0[21]);
                    float2 q_186 = make_float2(tv_46[4], tv_46[5]);
                    float2 r_186 = fma_f32x2_rn_noftz(q_186, scale1_9, a_186);
                    acc2_0[20] = r_186.x;
                    acc2_0[21] = r_186.y;
                    float2 a_187 = make_float2(acc2_0[22], acc2_0[23]);
                    float2 q_187 = make_float2(tv_46[6], tv_46[7]);
                    float2 r_187 = fma_f32x2_rn_noftz(q_187, scale1_9, a_187);
                    acc2_0[22] = r_187.x;
                    acc2_0[23] = r_187.y;
                    float tv_47[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_9[3 + 0]));
                        (&tv_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_47[0]), "=f"(tv_47[1]), "=f"(tv_47[2]), "=f"(tv_47[3]), "=f"(tv_47[4]), "=f"(tv_47[5]), "=f"(tv_47[6]), "=f"(tv_47[7]) : "r"(w4_9[3 + 0]));
                        #endif
                    }
                    float2 a_188 = make_float2(acc2_0[24], acc2_0[25]);
                    float2 q_188 = make_float2(tv_47[0], tv_47[1]);
                    float2 r_188 = fma_f32x2_rn_noftz(q_188, scale1_9, a_188);
                    acc2_0[24] = r_188.x;
                    acc2_0[25] = r_188.y;
                    float2 a_189 = make_float2(acc2_0[26], acc2_0[27]);
                    float2 q_189 = make_float2(tv_47[2], tv_47[3]);
                    float2 r_189 = fma_f32x2_rn_noftz(q_189, scale1_9, a_189);
                    acc2_0[26] = r_189.x;
                    acc2_0[27] = r_189.y;
                    float2 a_190 = make_float2(acc2_0[28], acc2_0[29]);
                    float2 q_190 = make_float2(tv_47[4], tv_47[5]);
                    float2 r_190 = fma_f32x2_rn_noftz(q_190, scale1_9, a_190);
                    acc2_0[28] = r_190.x;
                    acc2_0[29] = r_190.y;
                    float2 a_191 = make_float2(acc2_0[30], acc2_0[31]);
                    float2 q_191 = make_float2(tv_47[6], tv_47[7]);
                    float2 r_191 = fma_f32x2_rn_noftz(q_191, scale1_9, a_191);
                    acc2_0[30] = r_191.x;
                    acc2_0[31] = r_191.y;
                    unsigned int q[4];
                    float bm0 = 0.0f;
                    float bm1 = 0.0f;
                    bm0 = fmaxf(bm0, fabsf(acc2_0[0]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[16]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[1]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[17]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[2]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[18]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[3]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[19]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[4]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[20]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[5]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[21]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[6]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[22]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[7]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[23]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[8]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[24]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[9]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[25]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[10]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[26]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[11]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[27]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[12]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[28]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[13]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[29]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[14]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[30]));
                    bm0 = fmaxf(bm0, fabsf(acc2_0[15]));
                    bm1 = fmaxf(bm1, fabsf(acc2_0[31]));
                    float _fdiv_12 = __fdiv_rn(mul_f32_rn_noftz(bm1, inv), 6.0f);
                    float _fdiv_13 = __fdiv_rn(mul_f32_rn_noftz(bm0, inv), 6.0f);
                    uint16_t e4m3x2_f32;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32) : "f"(_fdiv_12), "f"(_fdiv_13));
                    uint32_t e4m3x2_to_f16x2_10;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_10) : "h"((uint16_t)(e4m3x2_f32)));
                    float cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_11) : "h"((uint16_t)(e4m3x2_to_f16x2_10)));
                    float cvt_f32_f16_1_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_10) : "h"((uint16_t)(e4m3x2_to_f16x2_10 >> 16)));
                    float fdiv_rn_0 = __fdiv_rn(inv, cvt_f32_f16_11);
                    float fdiv_rn_1 = __fdiv_rn(inv, cvt_f32_f16_1_10);
                    float m0 = 0.0f;
                    float m1 = 0.0f;
                    if (cvt_f32_f16_11 > 0.0f) {
                        m0 = fdiv_rn_0;
                    }
                    if (cvt_f32_f16_1_10 > 0.0f) {
                        m1 = fdiv_rn_1;
                    }
                    float2 mm_0 = make_float2(m0, m0);
                    float2 pair_0 = make_float2(acc2_0[0], acc2_0[1]);
                    float2 g_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_0) : "l"(*(const unsigned long long*)&pair_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_0;
                    uint16_t _e2m1_pair_h_14;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(g_0.x), "f"(g_0.y));
                    fp4_0 = (uint32_t)_e2m1_pair_h_14;
                    float2 pair_1 = make_float2(acc2_0[2], acc2_0[3]);
                    float2 g_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1) : "l"(*(const unsigned long long*)&pair_1), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_1;
                    uint16_t _e2m1_pair_h_15;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_1.x), "f"(g_1.y));
                    fp4_1 = (uint32_t)_e2m1_pair_h_15;
                    float2 pair_2 = make_float2(acc2_0[4], acc2_0[5]);
                    float2 g_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2) : "l"(*(const unsigned long long*)&pair_2), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_2;
                    uint16_t _e2m1_pair_h_16;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_2.x), "f"(g_2.y));
                    fp4_2 = (uint32_t)_e2m1_pair_h_16;
                    float2 pair_3 = make_float2(acc2_0[6], acc2_0[7]);
                    float2 g_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3) : "l"(*(const unsigned long long*)&pair_3), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t _fp4_e2m1_pack_pair_17;
                    uint16_t _e2m1_pair_h_18;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(g_3.x), "f"(g_3.y));
                    _fp4_e2m1_pack_pair_17 = (uint32_t)_e2m1_pair_h_18;
                    unsigned int word_0 = fp4_0 | fp4_1 << 8 | fp4_2 << 16 | _fp4_e2m1_pack_pair_17 << 24;
                    unsigned int carry_23 = (word_0 & 2004318071) + 2004318071;
                    q[0] = word_0 & (carry_23 | 2004318071);
                    float2 mm_1 = make_float2(m0, m0);
                    float2 pair_4 = make_float2(acc2_0[8], acc2_0[9]);
                    float2 g_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_4) : "l"(*(const unsigned long long*)&pair_4), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_3;
                    uint16_t _e2m1_pair_h_19;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_19) : "f"(g_4.x), "f"(g_4.y));
                    fp4_3 = (uint32_t)_e2m1_pair_h_19;
                    float2 pair_5 = make_float2(acc2_0[10], acc2_0[11]);
                    float2 g_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_5) : "l"(*(const unsigned long long*)&pair_5), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_4;
                    uint16_t _e2m1_pair_h_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_5.x), "f"(g_5.y));
                    fp4_4 = (uint32_t)_e2m1_pair_h_20;
                    float2 pair_6 = make_float2(acc2_0[12], acc2_0[13]);
                    float2 g_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_6) : "l"(*(const unsigned long long*)&pair_6), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_5;
                    uint16_t _e2m1_pair_h_21;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_6.x), "f"(g_6.y));
                    fp4_5 = (uint32_t)_e2m1_pair_h_21;
                    float2 pair_7 = make_float2(acc2_0[14], acc2_0[15]);
                    float2 g_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_7) : "l"(*(const unsigned long long*)&pair_7), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t _fp4_e2m1_pack_pair_22;
                    uint16_t _e2m1_pair_h_23;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(g_7.x), "f"(g_7.y));
                    _fp4_e2m1_pack_pair_22 = (uint32_t)_e2m1_pair_h_23;
                    unsigned int word_1 = fp4_3 | fp4_4 << 8 | fp4_5 << 16 | _fp4_e2m1_pack_pair_22 << 24;
                    unsigned int carry_23_1 = (word_1 & 2004318071) + 2004318071;
                    q[1] = word_1 & (carry_23_1 | 2004318071);
                    float2 mm_2 = make_float2(m1, m1);
                    float2 pair_8 = make_float2(acc2_0[16], acc2_0[17]);
                    float2 g_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_8) : "l"(*(const unsigned long long*)&pair_8), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_6;
                    uint16_t _e2m1_pair_h_24;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_24) : "f"(g_8.x), "f"(g_8.y));
                    fp4_6 = (uint32_t)_e2m1_pair_h_24;
                    float2 pair_9 = make_float2(acc2_0[18], acc2_0[19]);
                    float2 g_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_9) : "l"(*(const unsigned long long*)&pair_9), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_7;
                    uint16_t _e2m1_pair_h_25;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_9.x), "f"(g_9.y));
                    fp4_7 = (uint32_t)_e2m1_pair_h_25;
                    float2 pair_10 = make_float2(acc2_0[20], acc2_0[21]);
                    float2 g_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_10) : "l"(*(const unsigned long long*)&pair_10), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_8;
                    uint16_t _e2m1_pair_h_26;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(g_10.x), "f"(g_10.y));
                    fp4_8 = (uint32_t)_e2m1_pair_h_26;
                    float2 pair_11 = make_float2(acc2_0[22], acc2_0[23]);
                    float2 g_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_11) : "l"(*(const unsigned long long*)&pair_11), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t _fp4_e2m1_pack_pair_27;
                    uint16_t _e2m1_pair_h_28;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_28) : "f"(g_11.x), "f"(g_11.y));
                    _fp4_e2m1_pack_pair_27 = (uint32_t)_e2m1_pair_h_28;
                    unsigned int word_2 = fp4_6 | fp4_7 << 8 | fp4_8 << 16 | _fp4_e2m1_pack_pair_27 << 24;
                    unsigned int carry_24 = (word_2 & 2004318071) + 2004318071;
                    q[2] = word_2 & (carry_24 | 2004318071);
                    float2 mm_3 = make_float2(m1, m1);
                    float2 pair_12 = make_float2(acc2_0[24], acc2_0[25]);
                    float2 g_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_12) : "l"(*(const unsigned long long*)&pair_12), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_9;
                    uint16_t _e2m1_pair_h_29;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_29) : "f"(g_12.x), "f"(g_12.y));
                    fp4_9 = (uint32_t)_e2m1_pair_h_29;
                    float2 pair_13 = make_float2(acc2_0[26], acc2_0[27]);
                    float2 g_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_13) : "l"(*(const unsigned long long*)&pair_13), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_10;
                    uint16_t _e2m1_pair_h_30;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_30) : "f"(g_13.x), "f"(g_13.y));
                    fp4_10 = (uint32_t)_e2m1_pair_h_30;
                    float2 pair_14 = make_float2(acc2_0[28], acc2_0[29]);
                    float2 g_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_14) : "l"(*(const unsigned long long*)&pair_14), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_11;
                    uint16_t _e2m1_pair_h_31;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_31) : "f"(g_14.x), "f"(g_14.y));
                    fp4_11 = (uint32_t)_e2m1_pair_h_31;
                    float2 pair_15 = make_float2(acc2_0[30], acc2_0[31]);
                    float2 g_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_15) : "l"(*(const unsigned long long*)&pair_15), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t _fp4_e2m1_pack_pair_32;
                    uint16_t _e2m1_pair_h_33;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(g_15.x), "f"(g_15.y));
                    _fp4_e2m1_pack_pair_32 = (uint32_t)_e2m1_pair_h_33;
                    unsigned int word_3 = fp4_9 | fp4_10 << 8 | fp4_11 << 16 | _fp4_e2m1_pack_pair_32 << 24;
                    unsigned int carry_25 = (word_3 & 2004318071) + 2004318071;
                    q[3] = word_3 & (carry_25 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(qob_off + 16 * pp_0)), "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3]) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(qob_off + 3072 + 2 * pp_0)), "h"((uint16_t)(e4m3x2_f32 & 32639)) : "memory");
                    float d[32] = {0};
                    uint32_t e4m3x2_to_f16x2_11;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_11) : "h"((uint16_t)(e4m3x2_f32 & 32639)));
                    float cvt_f32_f16_12;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_12) : "h"((uint16_t)(e4m3x2_to_f16x2_11)));
                    float cvt_f32_f16_1_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_11) : "h"((uint16_t)(e4m3x2_to_f16x2_11 >> 16)));
                    float s0_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_10) : "f"(cvt_f32_f16_12), "f"(rs2));
                    float s1_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_10) : "f"(cvt_f32_f16_1_11), "f"(rs2));
                    float2 scale0_10 = make_float2(s0_10, s0_10);
                    float2 scale1_10 = make_float2(s1_10, s1_10);
                    float tv_48[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q[0]));
                        (&tv_48[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_48[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_48[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_48[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_48[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_48[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_48[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_48[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_48[0]), "=f"(tv_48[1]), "=f"(tv_48[2]), "=f"(tv_48[3]), "=f"(tv_48[4]), "=f"(tv_48[5]), "=f"(tv_48[6]), "=f"(tv_48[7]) : "r"(q[0]));
                        #endif
                    }
                    float2 a_192 = make_float2(d[0], d[1]);
                    float2 q_192 = make_float2(tv_48[0], tv_48[1]);
                    float2 r_192 = fma_f32x2_rn_noftz(q_192, scale0_10, a_192);
                    d[0] = r_192.x;
                    d[1] = r_192.y;
                    float2 a_193 = make_float2(d[2], d[3]);
                    float2 q_193 = make_float2(tv_48[2], tv_48[3]);
                    float2 r_193 = fma_f32x2_rn_noftz(q_193, scale0_10, a_193);
                    d[2] = r_193.x;
                    d[3] = r_193.y;
                    float2 a_194 = make_float2(d[4], d[5]);
                    float2 q_194 = make_float2(tv_48[4], tv_48[5]);
                    float2 r_194 = fma_f32x2_rn_noftz(q_194, scale0_10, a_194);
                    d[4] = r_194.x;
                    d[5] = r_194.y;
                    float2 a_195 = make_float2(d[6], d[7]);
                    float2 q_195 = make_float2(tv_48[6], tv_48[7]);
                    float2 r_195 = fma_f32x2_rn_noftz(q_195, scale0_10, a_195);
                    d[6] = r_195.x;
                    d[7] = r_195.y;
                    float tv_49[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q[1 + 0]));
                        (&tv_49[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_49[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_49[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_49[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_49[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_49[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_49[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_49[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_49[0]), "=f"(tv_49[1]), "=f"(tv_49[2]), "=f"(tv_49[3]), "=f"(tv_49[4]), "=f"(tv_49[5]), "=f"(tv_49[6]), "=f"(tv_49[7]) : "r"(q[1 + 0]));
                        #endif
                    }
                    float2 a_196 = make_float2(d[8], d[9]);
                    float2 q_196 = make_float2(tv_49[0], tv_49[1]);
                    float2 r_196 = fma_f32x2_rn_noftz(q_196, scale0_10, a_196);
                    d[8] = r_196.x;
                    d[9] = r_196.y;
                    float2 a_197 = make_float2(d[10], d[11]);
                    float2 q_197 = make_float2(tv_49[2], tv_49[3]);
                    float2 r_197 = fma_f32x2_rn_noftz(q_197, scale0_10, a_197);
                    d[10] = r_197.x;
                    d[11] = r_197.y;
                    float2 a_198 = make_float2(d[12], d[13]);
                    float2 q_198 = make_float2(tv_49[4], tv_49[5]);
                    float2 r_198 = fma_f32x2_rn_noftz(q_198, scale0_10, a_198);
                    d[12] = r_198.x;
                    d[13] = r_198.y;
                    float2 a_199 = make_float2(d[14], d[15]);
                    float2 q_199 = make_float2(tv_49[6], tv_49[7]);
                    float2 r_199 = fma_f32x2_rn_noftz(q_199, scale0_10, a_199);
                    d[14] = r_199.x;
                    d[15] = r_199.y;
                    float tv_50[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q[2 + 0]));
                        (&tv_50[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_50[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_50[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_50[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_50[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_50[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_50[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_50[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_50[0]), "=f"(tv_50[1]), "=f"(tv_50[2]), "=f"(tv_50[3]), "=f"(tv_50[4]), "=f"(tv_50[5]), "=f"(tv_50[6]), "=f"(tv_50[7]) : "r"(q[2 + 0]));
                        #endif
                    }
                    float2 a_200 = make_float2(d[16], d[17]);
                    float2 q_200 = make_float2(tv_50[0], tv_50[1]);
                    float2 r_200 = fma_f32x2_rn_noftz(q_200, scale1_10, a_200);
                    d[16] = r_200.x;
                    d[17] = r_200.y;
                    float2 a_201 = make_float2(d[18], d[19]);
                    float2 q_201 = make_float2(tv_50[2], tv_50[3]);
                    float2 r_201 = fma_f32x2_rn_noftz(q_201, scale1_10, a_201);
                    d[18] = r_201.x;
                    d[19] = r_201.y;
                    float2 a_202 = make_float2(d[20], d[21]);
                    float2 q_202 = make_float2(tv_50[4], tv_50[5]);
                    float2 r_202 = fma_f32x2_rn_noftz(q_202, scale1_10, a_202);
                    d[20] = r_202.x;
                    d[21] = r_202.y;
                    float2 a_203 = make_float2(d[22], d[23]);
                    float2 q_203 = make_float2(tv_50[6], tv_50[7]);
                    float2 r_203 = fma_f32x2_rn_noftz(q_203, scale1_10, a_203);
                    d[22] = r_203.x;
                    d[23] = r_203.y;
                    float tv_51[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q[3 + 0]));
                        (&tv_51[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_51[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_51[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_51[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_51[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_51[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_51[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_51[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_51[0]), "=f"(tv_51[1]), "=f"(tv_51[2]), "=f"(tv_51[3]), "=f"(tv_51[4]), "=f"(tv_51[5]), "=f"(tv_51[6]), "=f"(tv_51[7]) : "r"(q[3 + 0]));
                        #endif
                    }
                    float2 a_204 = make_float2(d[24], d[25]);
                    float2 q_204 = make_float2(tv_51[0], tv_51[1]);
                    float2 r_204 = fma_f32x2_rn_noftz(q_204, scale1_10, a_204);
                    d[24] = r_204.x;
                    d[25] = r_204.y;
                    float2 a_205 = make_float2(d[26], d[27]);
                    float2 q_205 = make_float2(tv_51[2], tv_51[3]);
                    float2 r_205 = fma_f32x2_rn_noftz(q_205, scale1_10, a_205);
                    d[26] = r_205.x;
                    d[27] = r_205.y;
                    float2 a_206 = make_float2(d[28], d[29]);
                    float2 q_206 = make_float2(tv_51[4], tv_51[5]);
                    float2 r_206 = fma_f32x2_rn_noftz(q_206, scale1_10, a_206);
                    d[28] = r_206.x;
                    d[29] = r_206.y;
                    float2 a_207 = make_float2(d[30], d[31]);
                    float2 q_207 = make_float2(tv_51[6], tv_51[7]);
                    float2 r_207 = fma_f32x2_rn_noftz(q_207, scale1_10, a_207);
                    d[30] = r_207.x;
                    d[31] = r_207.y;
                    unsigned int d_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d[_lp*2 + 0], d[_lp*2+1 + 0]));
                        d_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_bf16[0]), "r"(d_bf16[1]), "r"(d_bf16[2]), "r"(d_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_bf16[4]), "r"(d_bf16[5]), "r"(d_bf16[6]), "r"(d_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_bf16[8]), "r"(d_bf16[9]), "r"(d_bf16[10]), "r"(d_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_bf16[12]), "r"(d_bf16[13]), "r"(d_bf16[14]), "r"(d_bf16[15]) : "memory");
                    __syncwarp();
                    int c_4 = lane_1;
                    int l_0 = c_4 >> 2;
                    unsigned int w4_10[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_0 + (c_4 & 3 ^ l_0 >> 1 & 3)))));
                    long long idx_0 = qrow64 * 3072 + (long long)(4 * (128 * j_0 + c_4));
                    {
                        int4 _iv4 = make_int4(w4_10[0 + 0], w4_10[0 + 1], w4_10[0 + 2], w4_10[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_0) = _iv4;
                    }
                    int c_5 = lane_1 + 32;
                    int l_1 = c_5 >> 2;
                    unsigned int w4_11[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1 + (c_5 & 3 ^ l_1 >> 1 & 3)))));
                    long long idx_1 = qrow64 * 3072 + (long long)(4 * (128 * j_0 + c_5));
                    {
                        int4 _iv4 = make_int4(w4_11[0 + 0], w4_11[0 + 1], w4_11[0 + 2], w4_11[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1) = _iv4;
                    }
                    int c_6 = lane_1 + 64;
                    int l_2 = c_6 >> 2;
                    unsigned int w4_12[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_2 + (c_6 & 3 ^ l_2 >> 1 & 3)))));
                    long long idx_2 = qrow64 * 3072 + (long long)(4 * (128 * j_0 + c_6));
                    {
                        int4 _iv4 = make_int4(w4_12[0 + 0], w4_12[0 + 1], w4_12[0 + 2], w4_12[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_2) = _iv4;
                    }
                    int c_7 = lane_1 + 96;
                    int l_3 = c_7 >> 2;
                    unsigned int w4_13[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_3 + (c_7 & 3 ^ l_3 >> 1 & 3)))));
                    long long idx_3 = qrow64 * 3072 + (long long)(4 * (128 * j_0 + c_7));
                    {
                        int4 _iv4 = make_int4(w4_13[0 + 0], w4_13[0 + 1], w4_13[0 + 2], w4_13[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_3) = _iv4;
                    }
                    __syncwarp();
                }
                if (lane_1 < 4) {
                    {
                        uint32_t _addr_34 = static_cast<uint32_t>(wsm_addr + (unsigned int)(qob_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(rs2) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    int ag_wo = (int)ag_w;
                    int ag_ro = (int)ag_r;
                    int capo_0 = (int)capo64;
                    int ix_1_1 = (int)qix64;
                    int rk_2 = (int)rank64;
                    int v_off_1 = ag_wo + (rk_2 * capo_0 + ix_1_1) * 3072;
                    int sc_off_1 = ag_wo + 4 * capo_0 * 3072 + (rk_2 * capo_0 + ix_1_1) * 384;
                    int m_off = ag_ro + (rk_2 * capo_0 + ix_1_1) * 16;
                    int peer_1_0 = (rank + 1) % 4;
                    uint8_t* pw_1_0 = reinterpret_cast<uint8_t*>(tab[peer_1_0]);
                    uint8_t* pr_1_0 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_0]);
                    {
                        void* _cpbulk_dst_35 = reinterpret_cast<void*>(pw_1_0 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_35), "r"(wsm_addr + (unsigned int)qob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_36 = reinterpret_cast<void*>(pw_1_0 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_36), "r"(wsm_addr + (unsigned int)(qob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_37 = reinterpret_cast<void*>(pr_1_0 + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_37), "r"(wsm_addr + (unsigned int)(qob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_1_1 = (rank + 2) % 4;
                    uint8_t* pw_1_1 = reinterpret_cast<uint8_t*>(tab[peer_1_1]);
                    uint8_t* pr_1_1 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_1]);
                    {
                        void* _cpbulk_dst_38 = reinterpret_cast<void*>(pw_1_1 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_38), "r"(wsm_addr + (unsigned int)qob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_39 = reinterpret_cast<void*>(pw_1_1 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_39), "r"(wsm_addr + (unsigned int)(qob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_40 = reinterpret_cast<void*>(pr_1_1 + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_40), "r"(wsm_addr + (unsigned int)(qob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_1_2 = (rank + 3) % 4;
                    uint8_t* pw_1_2 = reinterpret_cast<uint8_t*>(tab[peer_1_2]);
                    uint8_t* pr_1_2 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_2]);
                    {
                        void* _cpbulk_dst_41 = reinterpret_cast<void*>(pw_1_2 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_41), "r"(wsm_addr + (unsigned int)qob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_42 = reinterpret_cast<void*>(pw_1_2 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_42), "r"(wsm_addr + (unsigned int)(qob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_43 = reinterpret_cast<void*>(pr_1_2 + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_43), "r"(wsm_addr + (unsigned int)(qob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                #pragma unroll 1
                for (int dq = 1; dq < 4; dq++) {
                    int po = (rank + dq) % 4;
                    long long qcrow64 = qix64 * 4 + (long long)po;
                    long long qsrc64 = (long long)po;
                    int ag_wo_1 = (int)ag_w;
                    int ag_ro_1 = (int)ag_r;
                    int capo_0_1 = (int)capo64;
                    int src = (int)qsrc64;
                    int ix_0 = (int)qix64;
                    int m_off_1 = ag_ro_1 + (src * capo_0_1 + ix_0) * 16;
                    float rs_0 = 0.0f;
                    uint32_t polled[1];
                    do {
                        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(polled[0]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + m_off_1)) : "memory");
                    } while (
                        (polled[0] == 0x80000000u)
                    );
                    rs_0 = __uint_as_float(polled[0]);
                    int v_0 = ag_wo_1 + (src * capo_0_1 + ix_0) * 3072;
                    int sc_0 = ag_wo_1 + 4 * capo_0_1 * 3072 + (src * capo_0_1 + ix_0) * 384;
                    #pragma unroll 1
                    for (int jj_1 = 0; jj_1 < 2; jj_1++) {
                        int pp0_1 = lane_1 + 96 * jj_1;
                        int pp1_1 = pp0_1 + 32;
                        int pp2 = pp0_1 + 64;
                        uint32_t fp4_wire_values_1[12];
                        uint32_t fp4_wire_scales_1[3];
                        uint16_t _wire_h16_44_0;
                        uint16_t _wire_h16_44_1;
                        uint16_t _wire_h16_44_2;
                        uint32_t _wire_vpend_44 = 7u;
                        uint32_t _wire_spend_44 = 7u;
                        do {
                            if (_wire_vpend_44 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_1[0]), "=r"(fp4_wire_values_1[1]), "=r"(fp4_wire_values_1[2]), "=r"(fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp0_1)))) : "memory");
                            }
                            if (_wire_vpend_44 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_1[4]), "=r"(fp4_wire_values_1[5]), "=r"(fp4_wire_values_1[6]), "=r"(fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp1_1)))) : "memory");
                            }
                            if (_wire_vpend_44 & 4u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_1[8]), "=r"(fp4_wire_values_1[9]), "=r"(fp4_wire_values_1[10]), "=r"(fp4_wire_values_1[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp2)))) : "memory");
                            }
                            if (_wire_spend_44 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_44_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_0 + (int)(2 * pp0_1)))) : "memory");
                                fp4_wire_scales_1[0] = (uint32_t)_wire_h16_44_0;
                            }
                            if (_wire_spend_44 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_44_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_0 + (int)(2 * pp1_1)))) : "memory");
                                fp4_wire_scales_1[1] = (uint32_t)_wire_h16_44_1;
                            }
                            if (_wire_spend_44 & 4u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_44_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_0 + (int)(2 * pp2)))) : "memory");
                                fp4_wire_scales_1[2] = (uint32_t)_wire_h16_44_2;
                            }
                            _wire_vpend_44 = ((uint32_t)((((fp4_wire_values_1[0] & ~((fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[1] & ~((fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[2] & ~((fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[3] & ~((fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((fp4_wire_values_1[4] & ~((fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[5] & ~((fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[6] & ~((fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[7] & ~((fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((fp4_wire_values_1[8] & ~((fp4_wire_values_1[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[9] & ~((fp4_wire_values_1[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[10] & ~((fp4_wire_values_1[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[11] & ~((fp4_wire_values_1[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2);
                            _wire_spend_44 = ((uint32_t)(((uint32_t)((fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_1[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_1[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((fp4_wire_scales_1[2] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_1[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2);
                        } while ((_wire_vpend_44 | _wire_spend_44) != 0u);
                        float acc[32] = {0};
                        uint32_t e4m3x2_to_f16x2_12;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_12) : "h"((uint16_t)(fp4_wire_scales_1[0])));
                        float cvt_f32_f16_13;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_13) : "h"((uint16_t)(e4m3x2_to_f16x2_12)));
                        float cvt_f32_f16_1_12;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_12) : "h"((uint16_t)(e4m3x2_to_f16x2_12 >> 16)));
                        float s0_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_11) : "f"(cvt_f32_f16_13), "f"(rs_0));
                        float s1_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_11) : "f"(cvt_f32_f16_1_12), "f"(rs_0));
                        float2 scale0_11 = make_float2(s0_11, s0_11);
                        float2 scale1_11 = make_float2(s1_11, s1_11);
                        float tv_52[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[0]));
                            (&tv_52[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_52[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_52[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_52[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_52[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_52[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_52[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_52[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_52[0]), "=f"(tv_52[1]), "=f"(tv_52[2]), "=f"(tv_52[3]), "=f"(tv_52[4]), "=f"(tv_52[5]), "=f"(tv_52[6]), "=f"(tv_52[7]) : "r"(fp4_wire_values_1[0]));
                            #endif
                        }
                        float2 a_208 = make_float2(acc[0], acc[1]);
                        float2 q_208 = make_float2(tv_52[0], tv_52[1]);
                        float2 r_208 = fma_f32x2_rn_noftz(q_208, scale0_11, a_208);
                        acc[0] = r_208.x;
                        acc[1] = r_208.y;
                        float2 a_209 = make_float2(acc[2], acc[3]);
                        float2 q_209 = make_float2(tv_52[2], tv_52[3]);
                        float2 r_209 = fma_f32x2_rn_noftz(q_209, scale0_11, a_209);
                        acc[2] = r_209.x;
                        acc[3] = r_209.y;
                        float2 a_210 = make_float2(acc[4], acc[5]);
                        float2 q_210 = make_float2(tv_52[4], tv_52[5]);
                        float2 r_210 = fma_f32x2_rn_noftz(q_210, scale0_11, a_210);
                        acc[4] = r_210.x;
                        acc[5] = r_210.y;
                        float2 a_211 = make_float2(acc[6], acc[7]);
                        float2 q_211 = make_float2(tv_52[6], tv_52[7]);
                        float2 r_211 = fma_f32x2_rn_noftz(q_211, scale0_11, a_211);
                        acc[6] = r_211.x;
                        acc[7] = r_211.y;
                        float tv_53[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[1 + 0]));
                            (&tv_53[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_53[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_53[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_53[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_53[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_53[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_53[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_53[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_53[0]), "=f"(tv_53[1]), "=f"(tv_53[2]), "=f"(tv_53[3]), "=f"(tv_53[4]), "=f"(tv_53[5]), "=f"(tv_53[6]), "=f"(tv_53[7]) : "r"(fp4_wire_values_1[1 + 0]));
                            #endif
                        }
                        float2 a_212 = make_float2(acc[8], acc[9]);
                        float2 q_212 = make_float2(tv_53[0], tv_53[1]);
                        float2 r_212 = fma_f32x2_rn_noftz(q_212, scale0_11, a_212);
                        acc[8] = r_212.x;
                        acc[9] = r_212.y;
                        float2 a_213 = make_float2(acc[10], acc[11]);
                        float2 q_213 = make_float2(tv_53[2], tv_53[3]);
                        float2 r_213 = fma_f32x2_rn_noftz(q_213, scale0_11, a_213);
                        acc[10] = r_213.x;
                        acc[11] = r_213.y;
                        float2 a_214 = make_float2(acc[12], acc[13]);
                        float2 q_214 = make_float2(tv_53[4], tv_53[5]);
                        float2 r_214 = fma_f32x2_rn_noftz(q_214, scale0_11, a_214);
                        acc[12] = r_214.x;
                        acc[13] = r_214.y;
                        float2 a_215 = make_float2(acc[14], acc[15]);
                        float2 q_215 = make_float2(tv_53[6], tv_53[7]);
                        float2 r_215 = fma_f32x2_rn_noftz(q_215, scale0_11, a_215);
                        acc[14] = r_215.x;
                        acc[15] = r_215.y;
                        float tv_54[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[2 + 0]));
                            (&tv_54[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_54[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_54[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_54[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_54[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_54[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_54[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_54[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_54[0]), "=f"(tv_54[1]), "=f"(tv_54[2]), "=f"(tv_54[3]), "=f"(tv_54[4]), "=f"(tv_54[5]), "=f"(tv_54[6]), "=f"(tv_54[7]) : "r"(fp4_wire_values_1[2 + 0]));
                            #endif
                        }
                        float2 a_216 = make_float2(acc[16], acc[17]);
                        float2 q_216 = make_float2(tv_54[0], tv_54[1]);
                        float2 r_216 = fma_f32x2_rn_noftz(q_216, scale1_11, a_216);
                        acc[16] = r_216.x;
                        acc[17] = r_216.y;
                        float2 a_217 = make_float2(acc[18], acc[19]);
                        float2 q_217 = make_float2(tv_54[2], tv_54[3]);
                        float2 r_217 = fma_f32x2_rn_noftz(q_217, scale1_11, a_217);
                        acc[18] = r_217.x;
                        acc[19] = r_217.y;
                        float2 a_218 = make_float2(acc[20], acc[21]);
                        float2 q_218 = make_float2(tv_54[4], tv_54[5]);
                        float2 r_218 = fma_f32x2_rn_noftz(q_218, scale1_11, a_218);
                        acc[20] = r_218.x;
                        acc[21] = r_218.y;
                        float2 a_219 = make_float2(acc[22], acc[23]);
                        float2 q_219 = make_float2(tv_54[6], tv_54[7]);
                        float2 r_219 = fma_f32x2_rn_noftz(q_219, scale1_11, a_219);
                        acc[22] = r_219.x;
                        acc[23] = r_219.y;
                        float tv_55[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[3 + 0]));
                            (&tv_55[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_55[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_55[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_55[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_55[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_55[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_55[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_55[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_55[0]), "=f"(tv_55[1]), "=f"(tv_55[2]), "=f"(tv_55[3]), "=f"(tv_55[4]), "=f"(tv_55[5]), "=f"(tv_55[6]), "=f"(tv_55[7]) : "r"(fp4_wire_values_1[3 + 0]));
                            #endif
                        }
                        float2 a_220 = make_float2(acc[24], acc[25]);
                        float2 q_220 = make_float2(tv_55[0], tv_55[1]);
                        float2 r_220 = fma_f32x2_rn_noftz(q_220, scale1_11, a_220);
                        acc[24] = r_220.x;
                        acc[25] = r_220.y;
                        float2 a_221 = make_float2(acc[26], acc[27]);
                        float2 q_221 = make_float2(tv_55[2], tv_55[3]);
                        float2 r_221 = fma_f32x2_rn_noftz(q_221, scale1_11, a_221);
                        acc[26] = r_221.x;
                        acc[27] = r_221.y;
                        float2 a_222 = make_float2(acc[28], acc[29]);
                        float2 q_222 = make_float2(tv_55[4], tv_55[5]);
                        float2 r_222 = fma_f32x2_rn_noftz(q_222, scale1_11, a_222);
                        acc[28] = r_222.x;
                        acc[29] = r_222.y;
                        float2 a_223 = make_float2(acc[30], acc[31]);
                        float2 q_223 = make_float2(tv_55[6], tv_55[7]);
                        float2 r_223 = fma_f32x2_rn_noftz(q_223, scale1_11, a_223);
                        acc[30] = r_223.x;
                        acc[31] = r_223.y;
                        unsigned int acc_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                            acc_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                        __syncwarp();
                        int c_8 = lane_1;
                        int l_4 = c_8 >> 2;
                        unsigned int w4_14[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_4 + (c_8 & 3 ^ l_4 >> 1 & 3)))));
                        long long idx_4 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1) + c_8));
                        {
                            int4 _iv4 = make_int4(w4_14[0 + 0], w4_14[0 + 1], w4_14[0 + 2], w4_14[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_4) = _iv4;
                        }
                        int c_9 = lane_1 + 32;
                        int l_5 = c_9 >> 2;
                        unsigned int w4_15[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_5 + (c_9 & 3 ^ l_5 >> 1 & 3)))));
                        long long idx_5 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1) + c_9));
                        {
                            int4 _iv4 = make_int4(w4_15[0 + 0], w4_15[0 + 1], w4_15[0 + 2], w4_15[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_5) = _iv4;
                        }
                        int c_10 = lane_1 + 64;
                        int l_6 = c_10 >> 2;
                        unsigned int w4_16[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_6 + (c_10 & 3 ^ l_6 >> 1 & 3)))));
                        long long idx_6 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1) + c_10));
                        {
                            int4 _iv4 = make_int4(w4_16[0 + 0], w4_16[0 + 1], w4_16[0 + 2], w4_16[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_6) = _iv4;
                        }
                        int c_11 = lane_1 + 96;
                        int l_7 = c_11 >> 2;
                        unsigned int w4_17[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_7 + (c_11 & 3 ^ l_7 >> 1 & 3)))));
                        long long idx_7 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1) + c_11));
                        {
                            int4 _iv4 = make_int4(w4_17[0 + 0], w4_17[0 + 1], w4_17[0 + 2], w4_17[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_7) = _iv4;
                        }
                        __syncwarp();
                        float acc_2[32] = {0};
                        uint32_t e4m3x2_to_f16x2_12_1;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_12_1) : "h"((uint16_t)(fp4_wire_scales_1[1])));
                        float cvt_f32_f16_13_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_13_1) : "h"((uint16_t)(e4m3x2_to_f16x2_12_1)));
                        float cvt_f32_f16_1_12_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_12_1) : "h"((uint16_t)(e4m3x2_to_f16x2_12_1 >> 16)));
                        float s0_11_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_11_1) : "f"(cvt_f32_f16_13_1), "f"(rs_0));
                        float s1_11_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_11_1) : "f"(cvt_f32_f16_1_12_1), "f"(rs_0));
                        float2 scale0_11_1 = make_float2(s0_11_1, s0_11_1);
                        float2 scale1_11_1 = make_float2(s1_11_1, s1_11_1);
                        float tv_56[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[4 + 0]));
                            (&tv_56[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_56[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_56[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_56[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_56[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_56[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_56[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_56[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_56[0]), "=f"(tv_56[1]), "=f"(tv_56[2]), "=f"(tv_56[3]), "=f"(tv_56[4]), "=f"(tv_56[5]), "=f"(tv_56[6]), "=f"(tv_56[7]) : "r"(fp4_wire_values_1[4 + 0]));
                            #endif
                        }
                        float2 a_224 = make_float2(acc_2[0], acc_2[1]);
                        float2 q_224 = make_float2(tv_56[0], tv_56[1]);
                        float2 r_224 = fma_f32x2_rn_noftz(q_224, scale0_11_1, a_224);
                        acc_2[0] = r_224.x;
                        acc_2[1] = r_224.y;
                        float2 a_225 = make_float2(acc_2[2], acc_2[3]);
                        float2 q_225 = make_float2(tv_56[2], tv_56[3]);
                        float2 r_225 = fma_f32x2_rn_noftz(q_225, scale0_11_1, a_225);
                        acc_2[2] = r_225.x;
                        acc_2[3] = r_225.y;
                        float2 a_226 = make_float2(acc_2[4], acc_2[5]);
                        float2 q_226 = make_float2(tv_56[4], tv_56[5]);
                        float2 r_226 = fma_f32x2_rn_noftz(q_226, scale0_11_1, a_226);
                        acc_2[4] = r_226.x;
                        acc_2[5] = r_226.y;
                        float2 a_227 = make_float2(acc_2[6], acc_2[7]);
                        float2 q_227 = make_float2(tv_56[6], tv_56[7]);
                        float2 r_227 = fma_f32x2_rn_noftz(q_227, scale0_11_1, a_227);
                        acc_2[6] = r_227.x;
                        acc_2[7] = r_227.y;
                        float tv_57[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[5 + 0]));
                            (&tv_57[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_57[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_57[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_57[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_57[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_57[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_57[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_57[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_57[0]), "=f"(tv_57[1]), "=f"(tv_57[2]), "=f"(tv_57[3]), "=f"(tv_57[4]), "=f"(tv_57[5]), "=f"(tv_57[6]), "=f"(tv_57[7]) : "r"(fp4_wire_values_1[5 + 0]));
                            #endif
                        }
                        float2 a_228 = make_float2(acc_2[8], acc_2[9]);
                        float2 q_228 = make_float2(tv_57[0], tv_57[1]);
                        float2 r_228 = fma_f32x2_rn_noftz(q_228, scale0_11_1, a_228);
                        acc_2[8] = r_228.x;
                        acc_2[9] = r_228.y;
                        float2 a_229 = make_float2(acc_2[10], acc_2[11]);
                        float2 q_229 = make_float2(tv_57[2], tv_57[3]);
                        float2 r_229 = fma_f32x2_rn_noftz(q_229, scale0_11_1, a_229);
                        acc_2[10] = r_229.x;
                        acc_2[11] = r_229.y;
                        float2 a_230 = make_float2(acc_2[12], acc_2[13]);
                        float2 q_230 = make_float2(tv_57[4], tv_57[5]);
                        float2 r_230 = fma_f32x2_rn_noftz(q_230, scale0_11_1, a_230);
                        acc_2[12] = r_230.x;
                        acc_2[13] = r_230.y;
                        float2 a_231 = make_float2(acc_2[14], acc_2[15]);
                        float2 q_231 = make_float2(tv_57[6], tv_57[7]);
                        float2 r_231 = fma_f32x2_rn_noftz(q_231, scale0_11_1, a_231);
                        acc_2[14] = r_231.x;
                        acc_2[15] = r_231.y;
                        float tv_58[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[6 + 0]));
                            (&tv_58[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_58[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_58[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_58[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_58[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_58[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_58[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_58[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_58[0]), "=f"(tv_58[1]), "=f"(tv_58[2]), "=f"(tv_58[3]), "=f"(tv_58[4]), "=f"(tv_58[5]), "=f"(tv_58[6]), "=f"(tv_58[7]) : "r"(fp4_wire_values_1[6 + 0]));
                            #endif
                        }
                        float2 a_232 = make_float2(acc_2[16], acc_2[17]);
                        float2 q_232 = make_float2(tv_58[0], tv_58[1]);
                        float2 r_232 = fma_f32x2_rn_noftz(q_232, scale1_11_1, a_232);
                        acc_2[16] = r_232.x;
                        acc_2[17] = r_232.y;
                        float2 a_233 = make_float2(acc_2[18], acc_2[19]);
                        float2 q_233 = make_float2(tv_58[2], tv_58[3]);
                        float2 r_233 = fma_f32x2_rn_noftz(q_233, scale1_11_1, a_233);
                        acc_2[18] = r_233.x;
                        acc_2[19] = r_233.y;
                        float2 a_234 = make_float2(acc_2[20], acc_2[21]);
                        float2 q_234 = make_float2(tv_58[4], tv_58[5]);
                        float2 r_234 = fma_f32x2_rn_noftz(q_234, scale1_11_1, a_234);
                        acc_2[20] = r_234.x;
                        acc_2[21] = r_234.y;
                        float2 a_235 = make_float2(acc_2[22], acc_2[23]);
                        float2 q_235 = make_float2(tv_58[6], tv_58[7]);
                        float2 r_235 = fma_f32x2_rn_noftz(q_235, scale1_11_1, a_235);
                        acc_2[22] = r_235.x;
                        acc_2[23] = r_235.y;
                        float tv_59[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[7 + 0]));
                            (&tv_59[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_59[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_59[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_59[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_59[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_59[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_59[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_59[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_59[0]), "=f"(tv_59[1]), "=f"(tv_59[2]), "=f"(tv_59[3]), "=f"(tv_59[4]), "=f"(tv_59[5]), "=f"(tv_59[6]), "=f"(tv_59[7]) : "r"(fp4_wire_values_1[7 + 0]));
                            #endif
                        }
                        float2 a_236 = make_float2(acc_2[24], acc_2[25]);
                        float2 q_236 = make_float2(tv_59[0], tv_59[1]);
                        float2 r_236 = fma_f32x2_rn_noftz(q_236, scale1_11_1, a_236);
                        acc_2[24] = r_236.x;
                        acc_2[25] = r_236.y;
                        float2 a_237 = make_float2(acc_2[26], acc_2[27]);
                        float2 q_237 = make_float2(tv_59[2], tv_59[3]);
                        float2 r_237 = fma_f32x2_rn_noftz(q_237, scale1_11_1, a_237);
                        acc_2[26] = r_237.x;
                        acc_2[27] = r_237.y;
                        float2 a_238 = make_float2(acc_2[28], acc_2[29]);
                        float2 q_238 = make_float2(tv_59[4], tv_59[5]);
                        float2 r_238 = fma_f32x2_rn_noftz(q_238, scale1_11_1, a_238);
                        acc_2[28] = r_238.x;
                        acc_2[29] = r_238.y;
                        float2 a_239 = make_float2(acc_2[30], acc_2[31]);
                        float2 q_239 = make_float2(tv_59[6], tv_59[7]);
                        float2 r_239 = fma_f32x2_rn_noftz(q_239, scale1_11_1, a_239);
                        acc_2[30] = r_239.x;
                        acc_2[31] = r_239.y;
                        unsigned int acc_bf16_0[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_2[_lp*2 + 0], acc_2[_lp*2+1 + 0]));
                            acc_bf16_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16_0[0]), "r"(acc_bf16_0[1]), "r"(acc_bf16_0[2]), "r"(acc_bf16_0[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16_0[4]), "r"(acc_bf16_0[5]), "r"(acc_bf16_0[6]), "r"(acc_bf16_0[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16_0[8]), "r"(acc_bf16_0[9]), "r"(acc_bf16_0[10]), "r"(acc_bf16_0[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16_0[12]), "r"(acc_bf16_0[13]), "r"(acc_bf16_0[14]), "r"(acc_bf16_0[15]) : "memory");
                        __syncwarp();
                        int c_12 = lane_1;
                        int l_8 = c_12 >> 2;
                        unsigned int w4_18[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_8 + (c_12 & 3 ^ l_8 >> 1 & 3)))));
                        long long idx_8 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 1) + c_12));
                        {
                            int4 _iv4 = make_int4(w4_18[0 + 0], w4_18[0 + 1], w4_18[0 + 2], w4_18[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_8) = _iv4;
                        }
                        int c_13 = lane_1 + 32;
                        int l_9 = c_13 >> 2;
                        unsigned int w4_19[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_9 + (c_13 & 3 ^ l_9 >> 1 & 3)))));
                        long long idx_9 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 1) + c_13));
                        {
                            int4 _iv4 = make_int4(w4_19[0 + 0], w4_19[0 + 1], w4_19[0 + 2], w4_19[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_9) = _iv4;
                        }
                        int c_14 = lane_1 + 64;
                        int l_10 = c_14 >> 2;
                        unsigned int w4_20[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_10 + (c_14 & 3 ^ l_10 >> 1 & 3)))));
                        long long idx_10 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 1) + c_14));
                        {
                            int4 _iv4 = make_int4(w4_20[0 + 0], w4_20[0 + 1], w4_20[0 + 2], w4_20[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_10) = _iv4;
                        }
                        int c_15 = lane_1 + 96;
                        int l_11 = c_15 >> 2;
                        unsigned int w4_21[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_11 + (c_15 & 3 ^ l_11 >> 1 & 3)))));
                        long long idx_11 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 1) + c_15));
                        {
                            int4 _iv4 = make_int4(w4_21[0 + 0], w4_21[0 + 1], w4_21[0 + 2], w4_21[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_11) = _iv4;
                        }
                        __syncwarp();
                        float acc_3[32] = {0};
                        uint32_t e4m3x2_to_f16x2_13;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_13) : "h"((uint16_t)(fp4_wire_scales_1[2])));
                        float cvt_f32_f16_14;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_14) : "h"((uint16_t)(e4m3x2_to_f16x2_13)));
                        float cvt_f32_f16_1_13;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_13) : "h"((uint16_t)(e4m3x2_to_f16x2_13 >> 16)));
                        float s0_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_12) : "f"(cvt_f32_f16_14), "f"(rs_0));
                        float s1_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_12) : "f"(cvt_f32_f16_1_13), "f"(rs_0));
                        float2 scale0_12 = make_float2(s0_12, s0_12);
                        float2 scale1_12 = make_float2(s1_12, s1_12);
                        float tv_60[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[8 + 0]));
                            (&tv_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_60[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_60[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_60[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_60[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_60[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_60[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_60[0]), "=f"(tv_60[1]), "=f"(tv_60[2]), "=f"(tv_60[3]), "=f"(tv_60[4]), "=f"(tv_60[5]), "=f"(tv_60[6]), "=f"(tv_60[7]) : "r"(fp4_wire_values_1[8 + 0]));
                            #endif
                        }
                        float2 a_240 = make_float2(acc_3[0], acc_3[1]);
                        float2 q_240 = make_float2(tv_60[0], tv_60[1]);
                        float2 r_240 = fma_f32x2_rn_noftz(q_240, scale0_12, a_240);
                        acc_3[0] = r_240.x;
                        acc_3[1] = r_240.y;
                        float2 a_241 = make_float2(acc_3[2], acc_3[3]);
                        float2 q_241 = make_float2(tv_60[2], tv_60[3]);
                        float2 r_241 = fma_f32x2_rn_noftz(q_241, scale0_12, a_241);
                        acc_3[2] = r_241.x;
                        acc_3[3] = r_241.y;
                        float2 a_242 = make_float2(acc_3[4], acc_3[5]);
                        float2 q_242 = make_float2(tv_60[4], tv_60[5]);
                        float2 r_242 = fma_f32x2_rn_noftz(q_242, scale0_12, a_242);
                        acc_3[4] = r_242.x;
                        acc_3[5] = r_242.y;
                        float2 a_243 = make_float2(acc_3[6], acc_3[7]);
                        float2 q_243 = make_float2(tv_60[6], tv_60[7]);
                        float2 r_243 = fma_f32x2_rn_noftz(q_243, scale0_12, a_243);
                        acc_3[6] = r_243.x;
                        acc_3[7] = r_243.y;
                        float tv_61[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[9 + 0]));
                            (&tv_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_61[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_61[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_61[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_61[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_61[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_61[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_61[0]), "=f"(tv_61[1]), "=f"(tv_61[2]), "=f"(tv_61[3]), "=f"(tv_61[4]), "=f"(tv_61[5]), "=f"(tv_61[6]), "=f"(tv_61[7]) : "r"(fp4_wire_values_1[9 + 0]));
                            #endif
                        }
                        float2 a_244 = make_float2(acc_3[8], acc_3[9]);
                        float2 q_244 = make_float2(tv_61[0], tv_61[1]);
                        float2 r_244 = fma_f32x2_rn_noftz(q_244, scale0_12, a_244);
                        acc_3[8] = r_244.x;
                        acc_3[9] = r_244.y;
                        float2 a_245 = make_float2(acc_3[10], acc_3[11]);
                        float2 q_245 = make_float2(tv_61[2], tv_61[3]);
                        float2 r_245 = fma_f32x2_rn_noftz(q_245, scale0_12, a_245);
                        acc_3[10] = r_245.x;
                        acc_3[11] = r_245.y;
                        float2 a_246 = make_float2(acc_3[12], acc_3[13]);
                        float2 q_246 = make_float2(tv_61[4], tv_61[5]);
                        float2 r_246 = fma_f32x2_rn_noftz(q_246, scale0_12, a_246);
                        acc_3[12] = r_246.x;
                        acc_3[13] = r_246.y;
                        float2 a_247 = make_float2(acc_3[14], acc_3[15]);
                        float2 q_247 = make_float2(tv_61[6], tv_61[7]);
                        float2 r_247 = fma_f32x2_rn_noftz(q_247, scale0_12, a_247);
                        acc_3[14] = r_247.x;
                        acc_3[15] = r_247.y;
                        float tv_62[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[10 + 0]));
                            (&tv_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_62[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_62[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_62[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_62[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_62[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_62[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_62[0]), "=f"(tv_62[1]), "=f"(tv_62[2]), "=f"(tv_62[3]), "=f"(tv_62[4]), "=f"(tv_62[5]), "=f"(tv_62[6]), "=f"(tv_62[7]) : "r"(fp4_wire_values_1[10 + 0]));
                            #endif
                        }
                        float2 a_248 = make_float2(acc_3[16], acc_3[17]);
                        float2 q_248 = make_float2(tv_62[0], tv_62[1]);
                        float2 r_248 = fma_f32x2_rn_noftz(q_248, scale1_12, a_248);
                        acc_3[16] = r_248.x;
                        acc_3[17] = r_248.y;
                        float2 a_249 = make_float2(acc_3[18], acc_3[19]);
                        float2 q_249 = make_float2(tv_62[2], tv_62[3]);
                        float2 r_249 = fma_f32x2_rn_noftz(q_249, scale1_12, a_249);
                        acc_3[18] = r_249.x;
                        acc_3[19] = r_249.y;
                        float2 a_250 = make_float2(acc_3[20], acc_3[21]);
                        float2 q_250 = make_float2(tv_62[4], tv_62[5]);
                        float2 r_250 = fma_f32x2_rn_noftz(q_250, scale1_12, a_250);
                        acc_3[20] = r_250.x;
                        acc_3[21] = r_250.y;
                        float2 a_251 = make_float2(acc_3[22], acc_3[23]);
                        float2 q_251 = make_float2(tv_62[6], tv_62[7]);
                        float2 r_251 = fma_f32x2_rn_noftz(q_251, scale1_12, a_251);
                        acc_3[22] = r_251.x;
                        acc_3[23] = r_251.y;
                        float tv_63[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[11 + 0]));
                            (&tv_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_63[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_63[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_63[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_63[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_63[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_63[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_63[0]), "=f"(tv_63[1]), "=f"(tv_63[2]), "=f"(tv_63[3]), "=f"(tv_63[4]), "=f"(tv_63[5]), "=f"(tv_63[6]), "=f"(tv_63[7]) : "r"(fp4_wire_values_1[11 + 0]));
                            #endif
                        }
                        float2 a_252 = make_float2(acc_3[24], acc_3[25]);
                        float2 q_252 = make_float2(tv_63[0], tv_63[1]);
                        float2 r_252 = fma_f32x2_rn_noftz(q_252, scale1_12, a_252);
                        acc_3[24] = r_252.x;
                        acc_3[25] = r_252.y;
                        float2 a_253 = make_float2(acc_3[26], acc_3[27]);
                        float2 q_253 = make_float2(tv_63[2], tv_63[3]);
                        float2 r_253 = fma_f32x2_rn_noftz(q_253, scale1_12, a_253);
                        acc_3[26] = r_253.x;
                        acc_3[27] = r_253.y;
                        float2 a_254 = make_float2(acc_3[28], acc_3[29]);
                        float2 q_254 = make_float2(tv_63[4], tv_63[5]);
                        float2 r_254 = fma_f32x2_rn_noftz(q_254, scale1_12, a_254);
                        acc_3[28] = r_254.x;
                        acc_3[29] = r_254.y;
                        float2 a_255 = make_float2(acc_3[30], acc_3[31]);
                        float2 q_255 = make_float2(tv_63[6], tv_63[7]);
                        float2 r_255 = fma_f32x2_rn_noftz(q_255, scale1_12, a_255);
                        acc_3[30] = r_255.x;
                        acc_3[31] = r_255.y;
                        unsigned int acc_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_3[_lp*2 + 0], acc_3[_lp*2+1 + 0]));
                            acc_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16_1[0]), "r"(acc_bf16_1[1]), "r"(acc_bf16_1[2]), "r"(acc_bf16_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16_1[4]), "r"(acc_bf16_1[5]), "r"(acc_bf16_1[6]), "r"(acc_bf16_1[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16_1[8]), "r"(acc_bf16_1[9]), "r"(acc_bf16_1[10]), "r"(acc_bf16_1[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16_1[12]), "r"(acc_bf16_1[13]), "r"(acc_bf16_1[14]), "r"(acc_bf16_1[15]) : "memory");
                        __syncwarp();
                        int c_16 = lane_1;
                        int l_12 = c_16 >> 2;
                        unsigned int w4_22[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_12 + (c_16 & 3 ^ l_12 >> 1 & 3)))));
                        long long idx_12 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 2) + c_16));
                        {
                            int4 _iv4 = make_int4(w4_22[0 + 0], w4_22[0 + 1], w4_22[0 + 2], w4_22[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_12) = _iv4;
                        }
                        int c_17 = lane_1 + 32;
                        int l_13 = c_17 >> 2;
                        unsigned int w4_23[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_13 + (c_17 & 3 ^ l_13 >> 1 & 3)))));
                        long long idx_13 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 2) + c_17));
                        {
                            int4 _iv4 = make_int4(w4_23[0 + 0], w4_23[0 + 1], w4_23[0 + 2], w4_23[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_13) = _iv4;
                        }
                        int c_18 = lane_1 + 64;
                        int l_14 = c_18 >> 2;
                        unsigned int w4_24[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_14 + (c_18 & 3 ^ l_14 >> 1 & 3)))));
                        long long idx_14 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 2) + c_18));
                        {
                            int4 _iv4 = make_int4(w4_24[0 + 0], w4_24[0 + 1], w4_24[0 + 2], w4_24[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_14) = _iv4;
                        }
                        int c_19 = lane_1 + 96;
                        int l_15 = c_19 >> 2;
                        unsigned int w4_25[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_25[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_15 + (c_19 & 3 ^ l_15 >> 1 & 3)))));
                        long long idx_15 = qcrow64 * 3072 + (long long)(4 * (128 * (3 * jj_1 + 2) + c_19));
                        {
                            int4 _iv4 = make_int4(w4_25[0 + 0], w4_25[0 + 1], w4_25[0 + 2], w4_25[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_15) = _iv4;
                        }
                        __syncwarp();
                    }
                }
                kb = kb + 1;
            }
            int tq = r0 + nrows - 4 * q1;
            int nbt = 0;
            if (rank < tq) {
                nbt = 1;
            }
            long long tix64 = (long long)q1;
            #pragma unroll 1
            for (int i = warp_0; i < nbt; i += 10) {
                if (lane_1 == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                __syncwarp();
                if (kb >= 2) {
                    if (lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                }
                int rs_wo_2 = (int)rs_w;
                int rs_ro_2 = (int)rs_r;
                int capo_2 = (int)capo64;
                int ix_2 = (int)tix64;
                int rk_3 = (int)rank64;
                int v_4 = rs_wo_2 + ix_2 * 3072;
                int sc_4 = rs_wo_2 + 4 * capo_2 * 3072 + ix_2 * 384;
                int v_1_1 = rs_wo_2 + (capo_2 + ix_2) * 3072;
                int sc_1_1 = rs_wo_2 + 4 * capo_2 * 3072 + (capo_2 + ix_2) * 384;
                int v_2_1 = rs_wo_2 + (2 * capo_2 + ix_2) * 3072;
                int sc_2_1 = rs_wo_2 + 4 * capo_2 * 3072 + (2 * capo_2 + ix_2) * 384;
                int v_3_1 = rs_wo_2 + (3 * capo_2 + ix_2) * 3072;
                int sc_3_1 = rs_wo_2 + 4 * capo_2 * 3072 + (3 * capo_2 + ix_2) * 384;
                float am_1 = 0.0f;
                int ret_lane_1 = xb_off + 6944 + 16 * lane_1;
                #pragma unroll 1
                for (int jj_2 = 0; jj_2 < 3; jj_2++) {
                    int pp0_2 = lane_1 + 64 * jj_2;
                    int pp1_2 = pp0_2 + 32;
                    uint32_t fp4_wire_values_2[32];
                    uint32_t fp4_wire_scales_2[8];
                    uint16_t _wire_h16_45_0;
                    uint16_t _wire_h16_45_1;
                    uint16_t _wire_h16_45_2;
                    uint16_t _wire_h16_45_3;
                    uint16_t _wire_h16_45_4;
                    uint16_t _wire_h16_45_5;
                    uint16_t _wire_h16_45_6;
                    uint16_t _wire_h16_45_7;
                    uint32_t _wire_vpend_45 = 255u;
                    uint32_t _wire_spend_45 = 255u;
                    do {
                        if (_wire_vpend_45 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_2[0]), "=r"(fp4_wire_values_2[1]), "=r"(fp4_wire_values_2[2]), "=r"(fp4_wire_values_2[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp0_2)))) : "memory");
                        }
                        if (_wire_vpend_45 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_2[4]), "=r"(fp4_wire_values_2[5]), "=r"(fp4_wire_values_2[6]), "=r"(fp4_wire_values_2[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1_1 + (int)(16 * pp0_2)))) : "memory");
                        }
                        if (_wire_vpend_45 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_2[8]), "=r"(fp4_wire_values_2[9]), "=r"(fp4_wire_values_2[10]), "=r"(fp4_wire_values_2[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2_1 + (int)(16 * pp0_2)))) : "memory");
                        }
                        if (_wire_vpend_45 & 8u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_2[12]), "=r"(fp4_wire_values_2[13]), "=r"(fp4_wire_values_2[14]), "=r"(fp4_wire_values_2[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3_1 + (int)(16 * pp0_2)))) : "memory");
                        }
                        if (_wire_vpend_45 & 16u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_2[16]), "=r"(fp4_wire_values_2[17]), "=r"(fp4_wire_values_2[18]), "=r"(fp4_wire_values_2[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp1_2)))) : "memory");
                        }
                        if (_wire_vpend_45 & 32u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_2[20]), "=r"(fp4_wire_values_2[21]), "=r"(fp4_wire_values_2[22]), "=r"(fp4_wire_values_2[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1_1 + (int)(16 * pp1_2)))) : "memory");
                        }
                        if (_wire_vpend_45 & 64u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_2[24]), "=r"(fp4_wire_values_2[25]), "=r"(fp4_wire_values_2[26]), "=r"(fp4_wire_values_2[27]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2_1 + (int)(16 * pp1_2)))) : "memory");
                        }
                        if (_wire_vpend_45 & 128u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_2[28]), "=r"(fp4_wire_values_2[29]), "=r"(fp4_wire_values_2[30]), "=r"(fp4_wire_values_2[31]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3_1 + (int)(16 * pp1_2)))) : "memory");
                        }
                        if (_wire_spend_45 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_45_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp0_2)))) : "memory");
                            fp4_wire_scales_2[0] = (uint32_t)_wire_h16_45_0;
                        }
                        if (_wire_spend_45 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_45_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1_1 + (int)(2 * pp0_2)))) : "memory");
                            fp4_wire_scales_2[1] = (uint32_t)_wire_h16_45_1;
                        }
                        if (_wire_spend_45 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_45_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2_1 + (int)(2 * pp0_2)))) : "memory");
                            fp4_wire_scales_2[2] = (uint32_t)_wire_h16_45_2;
                        }
                        if (_wire_spend_45 & 8u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_45_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3_1 + (int)(2 * pp0_2)))) : "memory");
                            fp4_wire_scales_2[3] = (uint32_t)_wire_h16_45_3;
                        }
                        if (_wire_spend_45 & 16u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_45_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp1_2)))) : "memory");
                            fp4_wire_scales_2[4] = (uint32_t)_wire_h16_45_4;
                        }
                        if (_wire_spend_45 & 32u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_45_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1_1 + (int)(2 * pp1_2)))) : "memory");
                            fp4_wire_scales_2[5] = (uint32_t)_wire_h16_45_5;
                        }
                        if (_wire_spend_45 & 64u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_45_6) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2_1 + (int)(2 * pp1_2)))) : "memory");
                            fp4_wire_scales_2[6] = (uint32_t)_wire_h16_45_6;
                        }
                        if (_wire_spend_45 & 128u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_45_7) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3_1 + (int)(2 * pp1_2)))) : "memory");
                            fp4_wire_scales_2[7] = (uint32_t)_wire_h16_45_7;
                        }
                        _wire_vpend_45 = ((uint32_t)((((fp4_wire_values_2[0] & ~((fp4_wire_values_2[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[1] & ~((fp4_wire_values_2[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[2] & ~((fp4_wire_values_2[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[3] & ~((fp4_wire_values_2[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((fp4_wire_values_2[4] & ~((fp4_wire_values_2[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[5] & ~((fp4_wire_values_2[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[6] & ~((fp4_wire_values_2[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[7] & ~((fp4_wire_values_2[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((fp4_wire_values_2[8] & ~((fp4_wire_values_2[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[9] & ~((fp4_wire_values_2[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[10] & ~((fp4_wire_values_2[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[11] & ~((fp4_wire_values_2[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((fp4_wire_values_2[12] & ~((fp4_wire_values_2[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[13] & ~((fp4_wire_values_2[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[14] & ~((fp4_wire_values_2[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[15] & ~((fp4_wire_values_2[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((fp4_wire_values_2[16] & ~((fp4_wire_values_2[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[17] & ~((fp4_wire_values_2[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[18] & ~((fp4_wire_values_2[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[19] & ~((fp4_wire_values_2[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((fp4_wire_values_2[20] & ~((fp4_wire_values_2[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[21] & ~((fp4_wire_values_2[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[22] & ~((fp4_wire_values_2[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[23] & ~((fp4_wire_values_2[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5) | ((uint32_t)((((fp4_wire_values_2[24] & ~((fp4_wire_values_2[24] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[25] & ~((fp4_wire_values_2[25] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[26] & ~((fp4_wire_values_2[26] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[27] & ~((fp4_wire_values_2[27] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 6) | ((uint32_t)((((fp4_wire_values_2[28] & ~((fp4_wire_values_2[28] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[29] & ~((fp4_wire_values_2[29] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[30] & ~((fp4_wire_values_2[30] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_2[31] & ~((fp4_wire_values_2[31] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 7);
                        _wire_spend_45 = ((uint32_t)(((uint32_t)((fp4_wire_scales_2[0] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_2[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((fp4_wire_scales_2[1] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_2[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((fp4_wire_scales_2[2] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_2[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((fp4_wire_scales_2[3] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_2[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((fp4_wire_scales_2[4] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_2[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((fp4_wire_scales_2[5] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_2[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5) | ((uint32_t)(((uint32_t)((fp4_wire_scales_2[6] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_2[6] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 6) | ((uint32_t)(((uint32_t)((fp4_wire_scales_2[7] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_2[7] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 7);
                    } while ((_wire_vpend_45 | _wire_spend_45) != 0u);
                    int wbase_1 = ret_lane_1 + 4096 * jj_2;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)wbase_1), "r"(fp4_wire_values_2[0]), "r"(fp4_wire_values_2[1]), "r"(fp4_wire_values_2[2]), "r"(fp4_wire_values_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase_1 + 512)), "r"(fp4_wire_values_2[4]), "r"(fp4_wire_values_2[5]), "r"(fp4_wire_values_2[6]), "r"(fp4_wire_values_2[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase_1 + 1024)), "r"(fp4_wire_values_2[8]), "r"(fp4_wire_values_2[9]), "r"(fp4_wire_values_2[10]), "r"(fp4_wire_values_2[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase_1 + 1536)), "r"(fp4_wire_values_2[12]), "r"(fp4_wire_values_2[13]), "r"(fp4_wire_values_2[14]), "r"(fp4_wire_values_2[15]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase_1 + 2048)), "r"(fp4_wire_values_2[16]), "r"(fp4_wire_values_2[17]), "r"(fp4_wire_values_2[18]), "r"(fp4_wire_values_2[19]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase_1 + 2560)), "r"(fp4_wire_values_2[20]), "r"(fp4_wire_values_2[21]), "r"(fp4_wire_values_2[22]), "r"(fp4_wire_values_2[23]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase_1 + 3072)), "r"(fp4_wire_values_2[24]), "r"(fp4_wire_values_2[25]), "r"(fp4_wire_values_2[26]), "r"(fp4_wire_values_2[27]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(wbase_1 + 3584)), "r"(fp4_wire_values_2[28]), "r"(fp4_wire_values_2[29]), "r"(fp4_wire_values_2[30]), "r"(fp4_wire_values_2[31]) : "memory");
                    float acc_4[32] = {0};
                    uint32_t e4m3x2_to_f16x2_14;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_14) : "h"((uint16_t)(fp4_wire_scales_2[0])));
                    float cvt_f32_f16_15;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_15) : "h"((uint16_t)(e4m3x2_to_f16x2_14)));
                    float cvt_f32_f16_1_14;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_14) : "h"((uint16_t)(e4m3x2_to_f16x2_14 >> 16)));
                    float s0_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_13) : "f"(cvt_f32_f16_15), "f"(g_s_0));
                    float s1_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_13) : "f"(cvt_f32_f16_1_14), "f"(g_s_0));
                    float2 scale0_13 = make_float2(s0_13, s0_13);
                    float2 scale1_13 = make_float2(s1_13, s1_13);
                    float tv_64[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[0]));
                        (&tv_64[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_64[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_64[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_64[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_64[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_64[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_64[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_64[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_64[0]), "=f"(tv_64[1]), "=f"(tv_64[2]), "=f"(tv_64[3]), "=f"(tv_64[4]), "=f"(tv_64[5]), "=f"(tv_64[6]), "=f"(tv_64[7]) : "r"(fp4_wire_values_2[0]));
                        #endif
                    }
                    float2 a_256 = make_float2(acc_4[0], acc_4[1]);
                    float2 q_256 = make_float2(tv_64[0], tv_64[1]);
                    float2 r_256 = fma_f32x2_rn_noftz(q_256, scale0_13, a_256);
                    acc_4[0] = r_256.x;
                    acc_4[1] = r_256.y;
                    float2 a_257 = make_float2(acc_4[2], acc_4[3]);
                    float2 q_257 = make_float2(tv_64[2], tv_64[3]);
                    float2 r_257 = fma_f32x2_rn_noftz(q_257, scale0_13, a_257);
                    acc_4[2] = r_257.x;
                    acc_4[3] = r_257.y;
                    float2 a_258 = make_float2(acc_4[4], acc_4[5]);
                    float2 q_258 = make_float2(tv_64[4], tv_64[5]);
                    float2 r_258 = fma_f32x2_rn_noftz(q_258, scale0_13, a_258);
                    acc_4[4] = r_258.x;
                    acc_4[5] = r_258.y;
                    float2 a_259 = make_float2(acc_4[6], acc_4[7]);
                    float2 q_259 = make_float2(tv_64[6], tv_64[7]);
                    float2 r_259 = fma_f32x2_rn_noftz(q_259, scale0_13, a_259);
                    acc_4[6] = r_259.x;
                    acc_4[7] = r_259.y;
                    float tv_65[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[1 + 0]));
                        (&tv_65[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_65[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_65[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_65[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_65[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_65[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_65[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_65[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_65[0]), "=f"(tv_65[1]), "=f"(tv_65[2]), "=f"(tv_65[3]), "=f"(tv_65[4]), "=f"(tv_65[5]), "=f"(tv_65[6]), "=f"(tv_65[7]) : "r"(fp4_wire_values_2[1 + 0]));
                        #endif
                    }
                    float2 a_260 = make_float2(acc_4[8], acc_4[9]);
                    float2 q_260 = make_float2(tv_65[0], tv_65[1]);
                    float2 r_260 = fma_f32x2_rn_noftz(q_260, scale0_13, a_260);
                    acc_4[8] = r_260.x;
                    acc_4[9] = r_260.y;
                    float2 a_261 = make_float2(acc_4[10], acc_4[11]);
                    float2 q_261 = make_float2(tv_65[2], tv_65[3]);
                    float2 r_261 = fma_f32x2_rn_noftz(q_261, scale0_13, a_261);
                    acc_4[10] = r_261.x;
                    acc_4[11] = r_261.y;
                    float2 a_262 = make_float2(acc_4[12], acc_4[13]);
                    float2 q_262 = make_float2(tv_65[4], tv_65[5]);
                    float2 r_262 = fma_f32x2_rn_noftz(q_262, scale0_13, a_262);
                    acc_4[12] = r_262.x;
                    acc_4[13] = r_262.y;
                    float2 a_263 = make_float2(acc_4[14], acc_4[15]);
                    float2 q_263 = make_float2(tv_65[6], tv_65[7]);
                    float2 r_263 = fma_f32x2_rn_noftz(q_263, scale0_13, a_263);
                    acc_4[14] = r_263.x;
                    acc_4[15] = r_263.y;
                    float tv_66[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[2 + 0]));
                        (&tv_66[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_66[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_66[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_66[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_66[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_66[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_66[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_66[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_66[0]), "=f"(tv_66[1]), "=f"(tv_66[2]), "=f"(tv_66[3]), "=f"(tv_66[4]), "=f"(tv_66[5]), "=f"(tv_66[6]), "=f"(tv_66[7]) : "r"(fp4_wire_values_2[2 + 0]));
                        #endif
                    }
                    float2 a_264 = make_float2(acc_4[16], acc_4[17]);
                    float2 q_264 = make_float2(tv_66[0], tv_66[1]);
                    float2 r_264 = fma_f32x2_rn_noftz(q_264, scale1_13, a_264);
                    acc_4[16] = r_264.x;
                    acc_4[17] = r_264.y;
                    float2 a_265 = make_float2(acc_4[18], acc_4[19]);
                    float2 q_265 = make_float2(tv_66[2], tv_66[3]);
                    float2 r_265 = fma_f32x2_rn_noftz(q_265, scale1_13, a_265);
                    acc_4[18] = r_265.x;
                    acc_4[19] = r_265.y;
                    float2 a_266 = make_float2(acc_4[20], acc_4[21]);
                    float2 q_266 = make_float2(tv_66[4], tv_66[5]);
                    float2 r_266 = fma_f32x2_rn_noftz(q_266, scale1_13, a_266);
                    acc_4[20] = r_266.x;
                    acc_4[21] = r_266.y;
                    float2 a_267 = make_float2(acc_4[22], acc_4[23]);
                    float2 q_267 = make_float2(tv_66[6], tv_66[7]);
                    float2 r_267 = fma_f32x2_rn_noftz(q_267, scale1_13, a_267);
                    acc_4[22] = r_267.x;
                    acc_4[23] = r_267.y;
                    float tv_67[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[3 + 0]));
                        (&tv_67[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_67[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_67[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_67[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_67[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_67[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_67[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_67[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_67[0]), "=f"(tv_67[1]), "=f"(tv_67[2]), "=f"(tv_67[3]), "=f"(tv_67[4]), "=f"(tv_67[5]), "=f"(tv_67[6]), "=f"(tv_67[7]) : "r"(fp4_wire_values_2[3 + 0]));
                        #endif
                    }
                    float2 a_268 = make_float2(acc_4[24], acc_4[25]);
                    float2 q_268 = make_float2(tv_67[0], tv_67[1]);
                    float2 r_268 = fma_f32x2_rn_noftz(q_268, scale1_13, a_268);
                    acc_4[24] = r_268.x;
                    acc_4[25] = r_268.y;
                    float2 a_269 = make_float2(acc_4[26], acc_4[27]);
                    float2 q_269 = make_float2(tv_67[2], tv_67[3]);
                    float2 r_269 = fma_f32x2_rn_noftz(q_269, scale1_13, a_269);
                    acc_4[26] = r_269.x;
                    acc_4[27] = r_269.y;
                    float2 a_270 = make_float2(acc_4[28], acc_4[29]);
                    float2 q_270 = make_float2(tv_67[4], tv_67[5]);
                    float2 r_270 = fma_f32x2_rn_noftz(q_270, scale1_13, a_270);
                    acc_4[28] = r_270.x;
                    acc_4[29] = r_270.y;
                    float2 a_271 = make_float2(acc_4[30], acc_4[31]);
                    float2 q_271 = make_float2(tv_67[6], tv_67[7]);
                    float2 r_271 = fma_f32x2_rn_noftz(q_271, scale1_13, a_271);
                    acc_4[30] = r_271.x;
                    acc_4[31] = r_271.y;
                    uint32_t e4m3x2_to_f16x2_14_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_14_1) : "h"((uint16_t)(fp4_wire_scales_2[1])));
                    float cvt_f32_f16_15_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_15_1) : "h"((uint16_t)(e4m3x2_to_f16x2_14_1)));
                    float cvt_f32_f16_1_14_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_14_1) : "h"((uint16_t)(e4m3x2_to_f16x2_14_1 >> 16)));
                    float s0_13_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_13_1) : "f"(cvt_f32_f16_15_1), "f"(g_s_1));
                    float s1_13_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_13_1) : "f"(cvt_f32_f16_1_14_1), "f"(g_s_1));
                    float2 scale0_13_1 = make_float2(s0_13_1, s0_13_1);
                    float2 scale1_13_1 = make_float2(s1_13_1, s1_13_1);
                    float tv_68[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[4 + 0]));
                        (&tv_68[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_68[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_68[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_68[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_68[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_68[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_68[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_68[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_68[0]), "=f"(tv_68[1]), "=f"(tv_68[2]), "=f"(tv_68[3]), "=f"(tv_68[4]), "=f"(tv_68[5]), "=f"(tv_68[6]), "=f"(tv_68[7]) : "r"(fp4_wire_values_2[4 + 0]));
                        #endif
                    }
                    float2 a_272 = make_float2(acc_4[0], acc_4[1]);
                    float2 q_272 = make_float2(tv_68[0], tv_68[1]);
                    float2 r_272 = fma_f32x2_rn_noftz(q_272, scale0_13_1, a_272);
                    acc_4[0] = r_272.x;
                    acc_4[1] = r_272.y;
                    float2 a_273 = make_float2(acc_4[2], acc_4[3]);
                    float2 q_273 = make_float2(tv_68[2], tv_68[3]);
                    float2 r_273 = fma_f32x2_rn_noftz(q_273, scale0_13_1, a_273);
                    acc_4[2] = r_273.x;
                    acc_4[3] = r_273.y;
                    float2 a_274 = make_float2(acc_4[4], acc_4[5]);
                    float2 q_274 = make_float2(tv_68[4], tv_68[5]);
                    float2 r_274 = fma_f32x2_rn_noftz(q_274, scale0_13_1, a_274);
                    acc_4[4] = r_274.x;
                    acc_4[5] = r_274.y;
                    float2 a_275 = make_float2(acc_4[6], acc_4[7]);
                    float2 q_275 = make_float2(tv_68[6], tv_68[7]);
                    float2 r_275 = fma_f32x2_rn_noftz(q_275, scale0_13_1, a_275);
                    acc_4[6] = r_275.x;
                    acc_4[7] = r_275.y;
                    float tv_69[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[5 + 0]));
                        (&tv_69[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_69[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_69[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_69[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_69[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_69[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_69[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_69[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_69[0]), "=f"(tv_69[1]), "=f"(tv_69[2]), "=f"(tv_69[3]), "=f"(tv_69[4]), "=f"(tv_69[5]), "=f"(tv_69[6]), "=f"(tv_69[7]) : "r"(fp4_wire_values_2[5 + 0]));
                        #endif
                    }
                    float2 a_276 = make_float2(acc_4[8], acc_4[9]);
                    float2 q_276 = make_float2(tv_69[0], tv_69[1]);
                    float2 r_276 = fma_f32x2_rn_noftz(q_276, scale0_13_1, a_276);
                    acc_4[8] = r_276.x;
                    acc_4[9] = r_276.y;
                    float2 a_277 = make_float2(acc_4[10], acc_4[11]);
                    float2 q_277 = make_float2(tv_69[2], tv_69[3]);
                    float2 r_277 = fma_f32x2_rn_noftz(q_277, scale0_13_1, a_277);
                    acc_4[10] = r_277.x;
                    acc_4[11] = r_277.y;
                    float2 a_278 = make_float2(acc_4[12], acc_4[13]);
                    float2 q_278 = make_float2(tv_69[4], tv_69[5]);
                    float2 r_278 = fma_f32x2_rn_noftz(q_278, scale0_13_1, a_278);
                    acc_4[12] = r_278.x;
                    acc_4[13] = r_278.y;
                    float2 a_279 = make_float2(acc_4[14], acc_4[15]);
                    float2 q_279 = make_float2(tv_69[6], tv_69[7]);
                    float2 r_279 = fma_f32x2_rn_noftz(q_279, scale0_13_1, a_279);
                    acc_4[14] = r_279.x;
                    acc_4[15] = r_279.y;
                    float tv_70[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[6 + 0]));
                        (&tv_70[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_70[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_70[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_70[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_70[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_70[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_70[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_70[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_70[0]), "=f"(tv_70[1]), "=f"(tv_70[2]), "=f"(tv_70[3]), "=f"(tv_70[4]), "=f"(tv_70[5]), "=f"(tv_70[6]), "=f"(tv_70[7]) : "r"(fp4_wire_values_2[6 + 0]));
                        #endif
                    }
                    float2 a_280 = make_float2(acc_4[16], acc_4[17]);
                    float2 q_280 = make_float2(tv_70[0], tv_70[1]);
                    float2 r_280 = fma_f32x2_rn_noftz(q_280, scale1_13_1, a_280);
                    acc_4[16] = r_280.x;
                    acc_4[17] = r_280.y;
                    float2 a_281 = make_float2(acc_4[18], acc_4[19]);
                    float2 q_281 = make_float2(tv_70[2], tv_70[3]);
                    float2 r_281 = fma_f32x2_rn_noftz(q_281, scale1_13_1, a_281);
                    acc_4[18] = r_281.x;
                    acc_4[19] = r_281.y;
                    float2 a_282 = make_float2(acc_4[20], acc_4[21]);
                    float2 q_282 = make_float2(tv_70[4], tv_70[5]);
                    float2 r_282 = fma_f32x2_rn_noftz(q_282, scale1_13_1, a_282);
                    acc_4[20] = r_282.x;
                    acc_4[21] = r_282.y;
                    float2 a_283 = make_float2(acc_4[22], acc_4[23]);
                    float2 q_283 = make_float2(tv_70[6], tv_70[7]);
                    float2 r_283 = fma_f32x2_rn_noftz(q_283, scale1_13_1, a_283);
                    acc_4[22] = r_283.x;
                    acc_4[23] = r_283.y;
                    float tv_71[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[7 + 0]));
                        (&tv_71[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_71[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_71[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_71[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_71[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_71[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_71[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_71[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_71[0]), "=f"(tv_71[1]), "=f"(tv_71[2]), "=f"(tv_71[3]), "=f"(tv_71[4]), "=f"(tv_71[5]), "=f"(tv_71[6]), "=f"(tv_71[7]) : "r"(fp4_wire_values_2[7 + 0]));
                        #endif
                    }
                    float2 a_284 = make_float2(acc_4[24], acc_4[25]);
                    float2 q_284 = make_float2(tv_71[0], tv_71[1]);
                    float2 r_284 = fma_f32x2_rn_noftz(q_284, scale1_13_1, a_284);
                    acc_4[24] = r_284.x;
                    acc_4[25] = r_284.y;
                    float2 a_285 = make_float2(acc_4[26], acc_4[27]);
                    float2 q_285 = make_float2(tv_71[2], tv_71[3]);
                    float2 r_285 = fma_f32x2_rn_noftz(q_285, scale1_13_1, a_285);
                    acc_4[26] = r_285.x;
                    acc_4[27] = r_285.y;
                    float2 a_286 = make_float2(acc_4[28], acc_4[29]);
                    float2 q_286 = make_float2(tv_71[4], tv_71[5]);
                    float2 r_286 = fma_f32x2_rn_noftz(q_286, scale1_13_1, a_286);
                    acc_4[28] = r_286.x;
                    acc_4[29] = r_286.y;
                    float2 a_287 = make_float2(acc_4[30], acc_4[31]);
                    float2 q_287 = make_float2(tv_71[6], tv_71[7]);
                    float2 r_287 = fma_f32x2_rn_noftz(q_287, scale1_13_1, a_287);
                    acc_4[30] = r_287.x;
                    acc_4[31] = r_287.y;
                    uint32_t e4m3x2_to_f16x2_15;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_15) : "h"((uint16_t)(fp4_wire_scales_2[2])));
                    float cvt_f32_f16_16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_16) : "h"((uint16_t)(e4m3x2_to_f16x2_15)));
                    float cvt_f32_f16_1_15;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_15) : "h"((uint16_t)(e4m3x2_to_f16x2_15 >> 16)));
                    float s0_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_14) : "f"(cvt_f32_f16_16), "f"(g_s_2));
                    float s1_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_14) : "f"(cvt_f32_f16_1_15), "f"(g_s_2));
                    float2 scale0_14 = make_float2(s0_14, s0_14);
                    float2 scale1_14 = make_float2(s1_14, s1_14);
                    float tv_72[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[8 + 0]));
                        (&tv_72[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_72[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_72[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_72[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_72[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_72[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_72[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_72[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_72[0]), "=f"(tv_72[1]), "=f"(tv_72[2]), "=f"(tv_72[3]), "=f"(tv_72[4]), "=f"(tv_72[5]), "=f"(tv_72[6]), "=f"(tv_72[7]) : "r"(fp4_wire_values_2[8 + 0]));
                        #endif
                    }
                    float2 a_288 = make_float2(acc_4[0], acc_4[1]);
                    float2 q_288 = make_float2(tv_72[0], tv_72[1]);
                    float2 r_288 = fma_f32x2_rn_noftz(q_288, scale0_14, a_288);
                    acc_4[0] = r_288.x;
                    acc_4[1] = r_288.y;
                    float2 a_289 = make_float2(acc_4[2], acc_4[3]);
                    float2 q_289 = make_float2(tv_72[2], tv_72[3]);
                    float2 r_289 = fma_f32x2_rn_noftz(q_289, scale0_14, a_289);
                    acc_4[2] = r_289.x;
                    acc_4[3] = r_289.y;
                    float2 a_290 = make_float2(acc_4[4], acc_4[5]);
                    float2 q_290 = make_float2(tv_72[4], tv_72[5]);
                    float2 r_290 = fma_f32x2_rn_noftz(q_290, scale0_14, a_290);
                    acc_4[4] = r_290.x;
                    acc_4[5] = r_290.y;
                    float2 a_291 = make_float2(acc_4[6], acc_4[7]);
                    float2 q_291 = make_float2(tv_72[6], tv_72[7]);
                    float2 r_291 = fma_f32x2_rn_noftz(q_291, scale0_14, a_291);
                    acc_4[6] = r_291.x;
                    acc_4[7] = r_291.y;
                    float tv_73[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[9 + 0]));
                        (&tv_73[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_73[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_73[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_73[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_73[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_73[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_73[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_73[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_73[0]), "=f"(tv_73[1]), "=f"(tv_73[2]), "=f"(tv_73[3]), "=f"(tv_73[4]), "=f"(tv_73[5]), "=f"(tv_73[6]), "=f"(tv_73[7]) : "r"(fp4_wire_values_2[9 + 0]));
                        #endif
                    }
                    float2 a_292 = make_float2(acc_4[8], acc_4[9]);
                    float2 q_292 = make_float2(tv_73[0], tv_73[1]);
                    float2 r_292 = fma_f32x2_rn_noftz(q_292, scale0_14, a_292);
                    acc_4[8] = r_292.x;
                    acc_4[9] = r_292.y;
                    float2 a_293 = make_float2(acc_4[10], acc_4[11]);
                    float2 q_293 = make_float2(tv_73[2], tv_73[3]);
                    float2 r_293 = fma_f32x2_rn_noftz(q_293, scale0_14, a_293);
                    acc_4[10] = r_293.x;
                    acc_4[11] = r_293.y;
                    float2 a_294 = make_float2(acc_4[12], acc_4[13]);
                    float2 q_294 = make_float2(tv_73[4], tv_73[5]);
                    float2 r_294 = fma_f32x2_rn_noftz(q_294, scale0_14, a_294);
                    acc_4[12] = r_294.x;
                    acc_4[13] = r_294.y;
                    float2 a_295 = make_float2(acc_4[14], acc_4[15]);
                    float2 q_295 = make_float2(tv_73[6], tv_73[7]);
                    float2 r_295 = fma_f32x2_rn_noftz(q_295, scale0_14, a_295);
                    acc_4[14] = r_295.x;
                    acc_4[15] = r_295.y;
                    float tv_74[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[10 + 0]));
                        (&tv_74[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_74[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_74[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_74[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_74[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_74[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_74[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_74[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_74[0]), "=f"(tv_74[1]), "=f"(tv_74[2]), "=f"(tv_74[3]), "=f"(tv_74[4]), "=f"(tv_74[5]), "=f"(tv_74[6]), "=f"(tv_74[7]) : "r"(fp4_wire_values_2[10 + 0]));
                        #endif
                    }
                    float2 a_296 = make_float2(acc_4[16], acc_4[17]);
                    float2 q_296 = make_float2(tv_74[0], tv_74[1]);
                    float2 r_296 = fma_f32x2_rn_noftz(q_296, scale1_14, a_296);
                    acc_4[16] = r_296.x;
                    acc_4[17] = r_296.y;
                    float2 a_297 = make_float2(acc_4[18], acc_4[19]);
                    float2 q_297 = make_float2(tv_74[2], tv_74[3]);
                    float2 r_297 = fma_f32x2_rn_noftz(q_297, scale1_14, a_297);
                    acc_4[18] = r_297.x;
                    acc_4[19] = r_297.y;
                    float2 a_298 = make_float2(acc_4[20], acc_4[21]);
                    float2 q_298 = make_float2(tv_74[4], tv_74[5]);
                    float2 r_298 = fma_f32x2_rn_noftz(q_298, scale1_14, a_298);
                    acc_4[20] = r_298.x;
                    acc_4[21] = r_298.y;
                    float2 a_299 = make_float2(acc_4[22], acc_4[23]);
                    float2 q_299 = make_float2(tv_74[6], tv_74[7]);
                    float2 r_299 = fma_f32x2_rn_noftz(q_299, scale1_14, a_299);
                    acc_4[22] = r_299.x;
                    acc_4[23] = r_299.y;
                    float tv_75[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[11 + 0]));
                        (&tv_75[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_75[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_75[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_75[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_75[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_75[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_75[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_75[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_75[0]), "=f"(tv_75[1]), "=f"(tv_75[2]), "=f"(tv_75[3]), "=f"(tv_75[4]), "=f"(tv_75[5]), "=f"(tv_75[6]), "=f"(tv_75[7]) : "r"(fp4_wire_values_2[11 + 0]));
                        #endif
                    }
                    float2 a_300 = make_float2(acc_4[24], acc_4[25]);
                    float2 q_300 = make_float2(tv_75[0], tv_75[1]);
                    float2 r_300 = fma_f32x2_rn_noftz(q_300, scale1_14, a_300);
                    acc_4[24] = r_300.x;
                    acc_4[25] = r_300.y;
                    float2 a_301 = make_float2(acc_4[26], acc_4[27]);
                    float2 q_301 = make_float2(tv_75[2], tv_75[3]);
                    float2 r_301 = fma_f32x2_rn_noftz(q_301, scale1_14, a_301);
                    acc_4[26] = r_301.x;
                    acc_4[27] = r_301.y;
                    float2 a_302 = make_float2(acc_4[28], acc_4[29]);
                    float2 q_302 = make_float2(tv_75[4], tv_75[5]);
                    float2 r_302 = fma_f32x2_rn_noftz(q_302, scale1_14, a_302);
                    acc_4[28] = r_302.x;
                    acc_4[29] = r_302.y;
                    float2 a_303 = make_float2(acc_4[30], acc_4[31]);
                    float2 q_303 = make_float2(tv_75[6], tv_75[7]);
                    float2 r_303 = fma_f32x2_rn_noftz(q_303, scale1_14, a_303);
                    acc_4[30] = r_303.x;
                    acc_4[31] = r_303.y;
                    uint32_t e4m3x2_to_f16x2_16;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_16) : "h"((uint16_t)(fp4_wire_scales_2[3])));
                    float cvt_f32_f16_17;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_17) : "h"((uint16_t)(e4m3x2_to_f16x2_16)));
                    float cvt_f32_f16_1_16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_16) : "h"((uint16_t)(e4m3x2_to_f16x2_16 >> 16)));
                    float s0_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_15) : "f"(cvt_f32_f16_17), "f"(g_s_3));
                    float s1_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_15) : "f"(cvt_f32_f16_1_16), "f"(g_s_3));
                    float2 scale0_15 = make_float2(s0_15, s0_15);
                    float2 scale1_15 = make_float2(s1_15, s1_15);
                    float tv_76[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[12 + 0]));
                        (&tv_76[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_76[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_76[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_76[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_76[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_76[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_76[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_76[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_76[0]), "=f"(tv_76[1]), "=f"(tv_76[2]), "=f"(tv_76[3]), "=f"(tv_76[4]), "=f"(tv_76[5]), "=f"(tv_76[6]), "=f"(tv_76[7]) : "r"(fp4_wire_values_2[12 + 0]));
                        #endif
                    }
                    float2 a_304 = make_float2(acc_4[0], acc_4[1]);
                    float2 q_304 = make_float2(tv_76[0], tv_76[1]);
                    float2 r_304 = fma_f32x2_rn_noftz(q_304, scale0_15, a_304);
                    acc_4[0] = r_304.x;
                    acc_4[1] = r_304.y;
                    float2 a_305 = make_float2(acc_4[2], acc_4[3]);
                    float2 q_305 = make_float2(tv_76[2], tv_76[3]);
                    float2 r_305 = fma_f32x2_rn_noftz(q_305, scale0_15, a_305);
                    acc_4[2] = r_305.x;
                    acc_4[3] = r_305.y;
                    float2 a_306 = make_float2(acc_4[4], acc_4[5]);
                    float2 q_306 = make_float2(tv_76[4], tv_76[5]);
                    float2 r_306 = fma_f32x2_rn_noftz(q_306, scale0_15, a_306);
                    acc_4[4] = r_306.x;
                    acc_4[5] = r_306.y;
                    float2 a_307 = make_float2(acc_4[6], acc_4[7]);
                    float2 q_307 = make_float2(tv_76[6], tv_76[7]);
                    float2 r_307 = fma_f32x2_rn_noftz(q_307, scale0_15, a_307);
                    acc_4[6] = r_307.x;
                    acc_4[7] = r_307.y;
                    float tv_77[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[13 + 0]));
                        (&tv_77[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_77[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_77[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_77[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_77[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_77[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_77[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_77[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_77[0]), "=f"(tv_77[1]), "=f"(tv_77[2]), "=f"(tv_77[3]), "=f"(tv_77[4]), "=f"(tv_77[5]), "=f"(tv_77[6]), "=f"(tv_77[7]) : "r"(fp4_wire_values_2[13 + 0]));
                        #endif
                    }
                    float2 a_308 = make_float2(acc_4[8], acc_4[9]);
                    float2 q_308 = make_float2(tv_77[0], tv_77[1]);
                    float2 r_308 = fma_f32x2_rn_noftz(q_308, scale0_15, a_308);
                    acc_4[8] = r_308.x;
                    acc_4[9] = r_308.y;
                    float2 a_309 = make_float2(acc_4[10], acc_4[11]);
                    float2 q_309 = make_float2(tv_77[2], tv_77[3]);
                    float2 r_309 = fma_f32x2_rn_noftz(q_309, scale0_15, a_309);
                    acc_4[10] = r_309.x;
                    acc_4[11] = r_309.y;
                    float2 a_310 = make_float2(acc_4[12], acc_4[13]);
                    float2 q_310 = make_float2(tv_77[4], tv_77[5]);
                    float2 r_310 = fma_f32x2_rn_noftz(q_310, scale0_15, a_310);
                    acc_4[12] = r_310.x;
                    acc_4[13] = r_310.y;
                    float2 a_311 = make_float2(acc_4[14], acc_4[15]);
                    float2 q_311 = make_float2(tv_77[6], tv_77[7]);
                    float2 r_311 = fma_f32x2_rn_noftz(q_311, scale0_15, a_311);
                    acc_4[14] = r_311.x;
                    acc_4[15] = r_311.y;
                    float tv_78[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[14 + 0]));
                        (&tv_78[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_78[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_78[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_78[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_78[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_78[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_78[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_78[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_78[0]), "=f"(tv_78[1]), "=f"(tv_78[2]), "=f"(tv_78[3]), "=f"(tv_78[4]), "=f"(tv_78[5]), "=f"(tv_78[6]), "=f"(tv_78[7]) : "r"(fp4_wire_values_2[14 + 0]));
                        #endif
                    }
                    float2 a_312 = make_float2(acc_4[16], acc_4[17]);
                    float2 q_312 = make_float2(tv_78[0], tv_78[1]);
                    float2 r_312 = fma_f32x2_rn_noftz(q_312, scale1_15, a_312);
                    acc_4[16] = r_312.x;
                    acc_4[17] = r_312.y;
                    float2 a_313 = make_float2(acc_4[18], acc_4[19]);
                    float2 q_313 = make_float2(tv_78[2], tv_78[3]);
                    float2 r_313 = fma_f32x2_rn_noftz(q_313, scale1_15, a_313);
                    acc_4[18] = r_313.x;
                    acc_4[19] = r_313.y;
                    float2 a_314 = make_float2(acc_4[20], acc_4[21]);
                    float2 q_314 = make_float2(tv_78[4], tv_78[5]);
                    float2 r_314 = fma_f32x2_rn_noftz(q_314, scale1_15, a_314);
                    acc_4[20] = r_314.x;
                    acc_4[21] = r_314.y;
                    float2 a_315 = make_float2(acc_4[22], acc_4[23]);
                    float2 q_315 = make_float2(tv_78[6], tv_78[7]);
                    float2 r_315 = fma_f32x2_rn_noftz(q_315, scale1_15, a_315);
                    acc_4[22] = r_315.x;
                    acc_4[23] = r_315.y;
                    float tv_79[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[15 + 0]));
                        (&tv_79[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_79[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_79[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_79[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_79[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_79[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_79[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_79[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_79[0]), "=f"(tv_79[1]), "=f"(tv_79[2]), "=f"(tv_79[3]), "=f"(tv_79[4]), "=f"(tv_79[5]), "=f"(tv_79[6]), "=f"(tv_79[7]) : "r"(fp4_wire_values_2[15 + 0]));
                        #endif
                    }
                    float2 a_316 = make_float2(acc_4[24], acc_4[25]);
                    float2 q_316 = make_float2(tv_79[0], tv_79[1]);
                    float2 r_316 = fma_f32x2_rn_noftz(q_316, scale1_15, a_316);
                    acc_4[24] = r_316.x;
                    acc_4[25] = r_316.y;
                    float2 a_317 = make_float2(acc_4[26], acc_4[27]);
                    float2 q_317 = make_float2(tv_79[2], tv_79[3]);
                    float2 r_317 = fma_f32x2_rn_noftz(q_317, scale1_15, a_317);
                    acc_4[26] = r_317.x;
                    acc_4[27] = r_317.y;
                    float2 a_318 = make_float2(acc_4[28], acc_4[29]);
                    float2 q_318 = make_float2(tv_79[4], tv_79[5]);
                    float2 r_318 = fma_f32x2_rn_noftz(q_318, scale1_15, a_318);
                    acc_4[28] = r_318.x;
                    acc_4[29] = r_318.y;
                    float2 a_319 = make_float2(acc_4[30], acc_4[31]);
                    float2 q_319 = make_float2(tv_79[6], tv_79[7]);
                    float2 r_319 = fma_f32x2_rn_noftz(q_319, scale1_15, a_319);
                    acc_4[30] = r_319.x;
                    acc_4[31] = r_319.y;
                    float cur_1 = am_1;
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[0]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[1]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[2]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[3]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[4]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[5]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[6]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[7]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[8]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[9]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[10]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[11]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[12]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[13]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[14]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[15]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[16]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[17]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[18]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[19]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[20]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[21]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[22]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[23]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[24]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[25]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[26]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[27]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[28]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[29]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[30]));
                    cur_1 = fmaxf(cur_1, fabsf(acc_4[31]));
                    am_1 = cur_1;
                    float acc_5[32] = {0};
                    uint32_t e4m3x2_to_f16x2_17;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_17) : "h"((uint16_t)(fp4_wire_scales_2[4])));
                    float cvt_f32_f16_18;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_18) : "h"((uint16_t)(e4m3x2_to_f16x2_17)));
                    float cvt_f32_f16_1_17;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_17) : "h"((uint16_t)(e4m3x2_to_f16x2_17 >> 16)));
                    float s0_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_16) : "f"(cvt_f32_f16_18), "f"(g_s_0));
                    float s1_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_16) : "f"(cvt_f32_f16_1_17), "f"(g_s_0));
                    float2 scale0_16 = make_float2(s0_16, s0_16);
                    float2 scale1_16 = make_float2(s1_16, s1_16);
                    float tv_80[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[16 + 0]));
                        (&tv_80[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_80[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_80[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_80[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_80[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_80[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_80[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_80[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_80[0]), "=f"(tv_80[1]), "=f"(tv_80[2]), "=f"(tv_80[3]), "=f"(tv_80[4]), "=f"(tv_80[5]), "=f"(tv_80[6]), "=f"(tv_80[7]) : "r"(fp4_wire_values_2[16 + 0]));
                        #endif
                    }
                    float2 a_320 = make_float2(acc_5[0], acc_5[1]);
                    float2 q_320 = make_float2(tv_80[0], tv_80[1]);
                    float2 r_320 = fma_f32x2_rn_noftz(q_320, scale0_16, a_320);
                    acc_5[0] = r_320.x;
                    acc_5[1] = r_320.y;
                    float2 a_321 = make_float2(acc_5[2], acc_5[3]);
                    float2 q_321 = make_float2(tv_80[2], tv_80[3]);
                    float2 r_321 = fma_f32x2_rn_noftz(q_321, scale0_16, a_321);
                    acc_5[2] = r_321.x;
                    acc_5[3] = r_321.y;
                    float2 a_322 = make_float2(acc_5[4], acc_5[5]);
                    float2 q_322 = make_float2(tv_80[4], tv_80[5]);
                    float2 r_322 = fma_f32x2_rn_noftz(q_322, scale0_16, a_322);
                    acc_5[4] = r_322.x;
                    acc_5[5] = r_322.y;
                    float2 a_323 = make_float2(acc_5[6], acc_5[7]);
                    float2 q_323 = make_float2(tv_80[6], tv_80[7]);
                    float2 r_323 = fma_f32x2_rn_noftz(q_323, scale0_16, a_323);
                    acc_5[6] = r_323.x;
                    acc_5[7] = r_323.y;
                    float tv_81[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[17 + 0]));
                        (&tv_81[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_81[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_81[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_81[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_81[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_81[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_81[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_81[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_81[0]), "=f"(tv_81[1]), "=f"(tv_81[2]), "=f"(tv_81[3]), "=f"(tv_81[4]), "=f"(tv_81[5]), "=f"(tv_81[6]), "=f"(tv_81[7]) : "r"(fp4_wire_values_2[17 + 0]));
                        #endif
                    }
                    float2 a_324 = make_float2(acc_5[8], acc_5[9]);
                    float2 q_324 = make_float2(tv_81[0], tv_81[1]);
                    float2 r_324 = fma_f32x2_rn_noftz(q_324, scale0_16, a_324);
                    acc_5[8] = r_324.x;
                    acc_5[9] = r_324.y;
                    float2 a_325 = make_float2(acc_5[10], acc_5[11]);
                    float2 q_325 = make_float2(tv_81[2], tv_81[3]);
                    float2 r_325 = fma_f32x2_rn_noftz(q_325, scale0_16, a_325);
                    acc_5[10] = r_325.x;
                    acc_5[11] = r_325.y;
                    float2 a_326 = make_float2(acc_5[12], acc_5[13]);
                    float2 q_326 = make_float2(tv_81[4], tv_81[5]);
                    float2 r_326 = fma_f32x2_rn_noftz(q_326, scale0_16, a_326);
                    acc_5[12] = r_326.x;
                    acc_5[13] = r_326.y;
                    float2 a_327 = make_float2(acc_5[14], acc_5[15]);
                    float2 q_327 = make_float2(tv_81[6], tv_81[7]);
                    float2 r_327 = fma_f32x2_rn_noftz(q_327, scale0_16, a_327);
                    acc_5[14] = r_327.x;
                    acc_5[15] = r_327.y;
                    float tv_82[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[18 + 0]));
                        (&tv_82[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_82[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_82[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_82[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_82[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_82[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_82[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_82[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_82[0]), "=f"(tv_82[1]), "=f"(tv_82[2]), "=f"(tv_82[3]), "=f"(tv_82[4]), "=f"(tv_82[5]), "=f"(tv_82[6]), "=f"(tv_82[7]) : "r"(fp4_wire_values_2[18 + 0]));
                        #endif
                    }
                    float2 a_328 = make_float2(acc_5[16], acc_5[17]);
                    float2 q_328 = make_float2(tv_82[0], tv_82[1]);
                    float2 r_328 = fma_f32x2_rn_noftz(q_328, scale1_16, a_328);
                    acc_5[16] = r_328.x;
                    acc_5[17] = r_328.y;
                    float2 a_329 = make_float2(acc_5[18], acc_5[19]);
                    float2 q_329 = make_float2(tv_82[2], tv_82[3]);
                    float2 r_329 = fma_f32x2_rn_noftz(q_329, scale1_16, a_329);
                    acc_5[18] = r_329.x;
                    acc_5[19] = r_329.y;
                    float2 a_330 = make_float2(acc_5[20], acc_5[21]);
                    float2 q_330 = make_float2(tv_82[4], tv_82[5]);
                    float2 r_330 = fma_f32x2_rn_noftz(q_330, scale1_16, a_330);
                    acc_5[20] = r_330.x;
                    acc_5[21] = r_330.y;
                    float2 a_331 = make_float2(acc_5[22], acc_5[23]);
                    float2 q_331 = make_float2(tv_82[6], tv_82[7]);
                    float2 r_331 = fma_f32x2_rn_noftz(q_331, scale1_16, a_331);
                    acc_5[22] = r_331.x;
                    acc_5[23] = r_331.y;
                    float tv_83[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[19 + 0]));
                        (&tv_83[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_83[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_83[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_83[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_83[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_83[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_83[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_83[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_83[0]), "=f"(tv_83[1]), "=f"(tv_83[2]), "=f"(tv_83[3]), "=f"(tv_83[4]), "=f"(tv_83[5]), "=f"(tv_83[6]), "=f"(tv_83[7]) : "r"(fp4_wire_values_2[19 + 0]));
                        #endif
                    }
                    float2 a_332 = make_float2(acc_5[24], acc_5[25]);
                    float2 q_332 = make_float2(tv_83[0], tv_83[1]);
                    float2 r_332 = fma_f32x2_rn_noftz(q_332, scale1_16, a_332);
                    acc_5[24] = r_332.x;
                    acc_5[25] = r_332.y;
                    float2 a_333 = make_float2(acc_5[26], acc_5[27]);
                    float2 q_333 = make_float2(tv_83[2], tv_83[3]);
                    float2 r_333 = fma_f32x2_rn_noftz(q_333, scale1_16, a_333);
                    acc_5[26] = r_333.x;
                    acc_5[27] = r_333.y;
                    float2 a_334 = make_float2(acc_5[28], acc_5[29]);
                    float2 q_334 = make_float2(tv_83[4], tv_83[5]);
                    float2 r_334 = fma_f32x2_rn_noftz(q_334, scale1_16, a_334);
                    acc_5[28] = r_334.x;
                    acc_5[29] = r_334.y;
                    float2 a_335 = make_float2(acc_5[30], acc_5[31]);
                    float2 q_335 = make_float2(tv_83[6], tv_83[7]);
                    float2 r_335 = fma_f32x2_rn_noftz(q_335, scale1_16, a_335);
                    acc_5[30] = r_335.x;
                    acc_5[31] = r_335.y;
                    uint32_t e4m3x2_to_f16x2_18;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_18) : "h"((uint16_t)(fp4_wire_scales_2[5])));
                    float cvt_f32_f16_19;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_19) : "h"((uint16_t)(e4m3x2_to_f16x2_18)));
                    float cvt_f32_f16_1_18;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_18) : "h"((uint16_t)(e4m3x2_to_f16x2_18 >> 16)));
                    float s0_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_17) : "f"(cvt_f32_f16_19), "f"(g_s_1));
                    float s1_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_17) : "f"(cvt_f32_f16_1_18), "f"(g_s_1));
                    float2 scale0_17 = make_float2(s0_17, s0_17);
                    float2 scale1_17 = make_float2(s1_17, s1_17);
                    float tv_84[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[20 + 0]));
                        (&tv_84[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_84[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_84[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_84[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_84[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_84[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_84[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_84[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_84[0]), "=f"(tv_84[1]), "=f"(tv_84[2]), "=f"(tv_84[3]), "=f"(tv_84[4]), "=f"(tv_84[5]), "=f"(tv_84[6]), "=f"(tv_84[7]) : "r"(fp4_wire_values_2[20 + 0]));
                        #endif
                    }
                    float2 a_336 = make_float2(acc_5[0], acc_5[1]);
                    float2 q_336 = make_float2(tv_84[0], tv_84[1]);
                    float2 r_336 = fma_f32x2_rn_noftz(q_336, scale0_17, a_336);
                    acc_5[0] = r_336.x;
                    acc_5[1] = r_336.y;
                    float2 a_337 = make_float2(acc_5[2], acc_5[3]);
                    float2 q_337 = make_float2(tv_84[2], tv_84[3]);
                    float2 r_337 = fma_f32x2_rn_noftz(q_337, scale0_17, a_337);
                    acc_5[2] = r_337.x;
                    acc_5[3] = r_337.y;
                    float2 a_338 = make_float2(acc_5[4], acc_5[5]);
                    float2 q_338 = make_float2(tv_84[4], tv_84[5]);
                    float2 r_338 = fma_f32x2_rn_noftz(q_338, scale0_17, a_338);
                    acc_5[4] = r_338.x;
                    acc_5[5] = r_338.y;
                    float2 a_339 = make_float2(acc_5[6], acc_5[7]);
                    float2 q_339 = make_float2(tv_84[6], tv_84[7]);
                    float2 r_339 = fma_f32x2_rn_noftz(q_339, scale0_17, a_339);
                    acc_5[6] = r_339.x;
                    acc_5[7] = r_339.y;
                    float tv_85[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[21 + 0]));
                        (&tv_85[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_85[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_85[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_85[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_85[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_85[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_85[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_85[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_85[0]), "=f"(tv_85[1]), "=f"(tv_85[2]), "=f"(tv_85[3]), "=f"(tv_85[4]), "=f"(tv_85[5]), "=f"(tv_85[6]), "=f"(tv_85[7]) : "r"(fp4_wire_values_2[21 + 0]));
                        #endif
                    }
                    float2 a_340 = make_float2(acc_5[8], acc_5[9]);
                    float2 q_340 = make_float2(tv_85[0], tv_85[1]);
                    float2 r_340 = fma_f32x2_rn_noftz(q_340, scale0_17, a_340);
                    acc_5[8] = r_340.x;
                    acc_5[9] = r_340.y;
                    float2 a_341 = make_float2(acc_5[10], acc_5[11]);
                    float2 q_341 = make_float2(tv_85[2], tv_85[3]);
                    float2 r_341 = fma_f32x2_rn_noftz(q_341, scale0_17, a_341);
                    acc_5[10] = r_341.x;
                    acc_5[11] = r_341.y;
                    float2 a_342 = make_float2(acc_5[12], acc_5[13]);
                    float2 q_342 = make_float2(tv_85[4], tv_85[5]);
                    float2 r_342 = fma_f32x2_rn_noftz(q_342, scale0_17, a_342);
                    acc_5[12] = r_342.x;
                    acc_5[13] = r_342.y;
                    float2 a_343 = make_float2(acc_5[14], acc_5[15]);
                    float2 q_343 = make_float2(tv_85[6], tv_85[7]);
                    float2 r_343 = fma_f32x2_rn_noftz(q_343, scale0_17, a_343);
                    acc_5[14] = r_343.x;
                    acc_5[15] = r_343.y;
                    float tv_86[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[22 + 0]));
                        (&tv_86[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_86[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_86[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_86[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_86[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_86[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_86[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_86[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_86[0]), "=f"(tv_86[1]), "=f"(tv_86[2]), "=f"(tv_86[3]), "=f"(tv_86[4]), "=f"(tv_86[5]), "=f"(tv_86[6]), "=f"(tv_86[7]) : "r"(fp4_wire_values_2[22 + 0]));
                        #endif
                    }
                    float2 a_344 = make_float2(acc_5[16], acc_5[17]);
                    float2 q_344 = make_float2(tv_86[0], tv_86[1]);
                    float2 r_344 = fma_f32x2_rn_noftz(q_344, scale1_17, a_344);
                    acc_5[16] = r_344.x;
                    acc_5[17] = r_344.y;
                    float2 a_345 = make_float2(acc_5[18], acc_5[19]);
                    float2 q_345 = make_float2(tv_86[2], tv_86[3]);
                    float2 r_345 = fma_f32x2_rn_noftz(q_345, scale1_17, a_345);
                    acc_5[18] = r_345.x;
                    acc_5[19] = r_345.y;
                    float2 a_346 = make_float2(acc_5[20], acc_5[21]);
                    float2 q_346 = make_float2(tv_86[4], tv_86[5]);
                    float2 r_346 = fma_f32x2_rn_noftz(q_346, scale1_17, a_346);
                    acc_5[20] = r_346.x;
                    acc_5[21] = r_346.y;
                    float2 a_347 = make_float2(acc_5[22], acc_5[23]);
                    float2 q_347 = make_float2(tv_86[6], tv_86[7]);
                    float2 r_347 = fma_f32x2_rn_noftz(q_347, scale1_17, a_347);
                    acc_5[22] = r_347.x;
                    acc_5[23] = r_347.y;
                    float tv_87[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[23 + 0]));
                        (&tv_87[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_87[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_87[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_87[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_87[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_87[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_87[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_87[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_87[0]), "=f"(tv_87[1]), "=f"(tv_87[2]), "=f"(tv_87[3]), "=f"(tv_87[4]), "=f"(tv_87[5]), "=f"(tv_87[6]), "=f"(tv_87[7]) : "r"(fp4_wire_values_2[23 + 0]));
                        #endif
                    }
                    float2 a_348 = make_float2(acc_5[24], acc_5[25]);
                    float2 q_348 = make_float2(tv_87[0], tv_87[1]);
                    float2 r_348 = fma_f32x2_rn_noftz(q_348, scale1_17, a_348);
                    acc_5[24] = r_348.x;
                    acc_5[25] = r_348.y;
                    float2 a_349 = make_float2(acc_5[26], acc_5[27]);
                    float2 q_349 = make_float2(tv_87[2], tv_87[3]);
                    float2 r_349 = fma_f32x2_rn_noftz(q_349, scale1_17, a_349);
                    acc_5[26] = r_349.x;
                    acc_5[27] = r_349.y;
                    float2 a_350 = make_float2(acc_5[28], acc_5[29]);
                    float2 q_350 = make_float2(tv_87[4], tv_87[5]);
                    float2 r_350 = fma_f32x2_rn_noftz(q_350, scale1_17, a_350);
                    acc_5[28] = r_350.x;
                    acc_5[29] = r_350.y;
                    float2 a_351 = make_float2(acc_5[30], acc_5[31]);
                    float2 q_351 = make_float2(tv_87[6], tv_87[7]);
                    float2 r_351 = fma_f32x2_rn_noftz(q_351, scale1_17, a_351);
                    acc_5[30] = r_351.x;
                    acc_5[31] = r_351.y;
                    uint32_t e4m3x2_to_f16x2_19;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_19) : "h"((uint16_t)(fp4_wire_scales_2[6])));
                    float cvt_f32_f16_20;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_20) : "h"((uint16_t)(e4m3x2_to_f16x2_19)));
                    float cvt_f32_f16_1_19;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_19) : "h"((uint16_t)(e4m3x2_to_f16x2_19 >> 16)));
                    float s0_18;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_18) : "f"(cvt_f32_f16_20), "f"(g_s_2));
                    float s1_18;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_18) : "f"(cvt_f32_f16_1_19), "f"(g_s_2));
                    float2 scale0_18 = make_float2(s0_18, s0_18);
                    float2 scale1_18 = make_float2(s1_18, s1_18);
                    float tv_88[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[24 + 0]));
                        (&tv_88[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_88[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_88[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_88[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_88[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_88[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_88[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_88[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_88[0]), "=f"(tv_88[1]), "=f"(tv_88[2]), "=f"(tv_88[3]), "=f"(tv_88[4]), "=f"(tv_88[5]), "=f"(tv_88[6]), "=f"(tv_88[7]) : "r"(fp4_wire_values_2[24 + 0]));
                        #endif
                    }
                    float2 a_352 = make_float2(acc_5[0], acc_5[1]);
                    float2 q_352 = make_float2(tv_88[0], tv_88[1]);
                    float2 r_352 = fma_f32x2_rn_noftz(q_352, scale0_18, a_352);
                    acc_5[0] = r_352.x;
                    acc_5[1] = r_352.y;
                    float2 a_353 = make_float2(acc_5[2], acc_5[3]);
                    float2 q_353 = make_float2(tv_88[2], tv_88[3]);
                    float2 r_353 = fma_f32x2_rn_noftz(q_353, scale0_18, a_353);
                    acc_5[2] = r_353.x;
                    acc_5[3] = r_353.y;
                    float2 a_354 = make_float2(acc_5[4], acc_5[5]);
                    float2 q_354 = make_float2(tv_88[4], tv_88[5]);
                    float2 r_354 = fma_f32x2_rn_noftz(q_354, scale0_18, a_354);
                    acc_5[4] = r_354.x;
                    acc_5[5] = r_354.y;
                    float2 a_355 = make_float2(acc_5[6], acc_5[7]);
                    float2 q_355 = make_float2(tv_88[6], tv_88[7]);
                    float2 r_355 = fma_f32x2_rn_noftz(q_355, scale0_18, a_355);
                    acc_5[6] = r_355.x;
                    acc_5[7] = r_355.y;
                    float tv_89[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[25 + 0]));
                        (&tv_89[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_89[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_89[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_89[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_89[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_89[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_89[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_89[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_89[0]), "=f"(tv_89[1]), "=f"(tv_89[2]), "=f"(tv_89[3]), "=f"(tv_89[4]), "=f"(tv_89[5]), "=f"(tv_89[6]), "=f"(tv_89[7]) : "r"(fp4_wire_values_2[25 + 0]));
                        #endif
                    }
                    float2 a_356 = make_float2(acc_5[8], acc_5[9]);
                    float2 q_356 = make_float2(tv_89[0], tv_89[1]);
                    float2 r_356 = fma_f32x2_rn_noftz(q_356, scale0_18, a_356);
                    acc_5[8] = r_356.x;
                    acc_5[9] = r_356.y;
                    float2 a_357 = make_float2(acc_5[10], acc_5[11]);
                    float2 q_357 = make_float2(tv_89[2], tv_89[3]);
                    float2 r_357 = fma_f32x2_rn_noftz(q_357, scale0_18, a_357);
                    acc_5[10] = r_357.x;
                    acc_5[11] = r_357.y;
                    float2 a_358 = make_float2(acc_5[12], acc_5[13]);
                    float2 q_358 = make_float2(tv_89[4], tv_89[5]);
                    float2 r_358 = fma_f32x2_rn_noftz(q_358, scale0_18, a_358);
                    acc_5[12] = r_358.x;
                    acc_5[13] = r_358.y;
                    float2 a_359 = make_float2(acc_5[14], acc_5[15]);
                    float2 q_359 = make_float2(tv_89[6], tv_89[7]);
                    float2 r_359 = fma_f32x2_rn_noftz(q_359, scale0_18, a_359);
                    acc_5[14] = r_359.x;
                    acc_5[15] = r_359.y;
                    float tv_90[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[26 + 0]));
                        (&tv_90[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_90[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_90[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_90[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_90[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_90[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_90[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_90[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_90[0]), "=f"(tv_90[1]), "=f"(tv_90[2]), "=f"(tv_90[3]), "=f"(tv_90[4]), "=f"(tv_90[5]), "=f"(tv_90[6]), "=f"(tv_90[7]) : "r"(fp4_wire_values_2[26 + 0]));
                        #endif
                    }
                    float2 a_360 = make_float2(acc_5[16], acc_5[17]);
                    float2 q_360 = make_float2(tv_90[0], tv_90[1]);
                    float2 r_360 = fma_f32x2_rn_noftz(q_360, scale1_18, a_360);
                    acc_5[16] = r_360.x;
                    acc_5[17] = r_360.y;
                    float2 a_361 = make_float2(acc_5[18], acc_5[19]);
                    float2 q_361 = make_float2(tv_90[2], tv_90[3]);
                    float2 r_361 = fma_f32x2_rn_noftz(q_361, scale1_18, a_361);
                    acc_5[18] = r_361.x;
                    acc_5[19] = r_361.y;
                    float2 a_362 = make_float2(acc_5[20], acc_5[21]);
                    float2 q_362 = make_float2(tv_90[4], tv_90[5]);
                    float2 r_362 = fma_f32x2_rn_noftz(q_362, scale1_18, a_362);
                    acc_5[20] = r_362.x;
                    acc_5[21] = r_362.y;
                    float2 a_363 = make_float2(acc_5[22], acc_5[23]);
                    float2 q_363 = make_float2(tv_90[6], tv_90[7]);
                    float2 r_363 = fma_f32x2_rn_noftz(q_363, scale1_18, a_363);
                    acc_5[22] = r_363.x;
                    acc_5[23] = r_363.y;
                    float tv_91[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[27 + 0]));
                        (&tv_91[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_91[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_91[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_91[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_91[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_91[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_91[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_91[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_91[0]), "=f"(tv_91[1]), "=f"(tv_91[2]), "=f"(tv_91[3]), "=f"(tv_91[4]), "=f"(tv_91[5]), "=f"(tv_91[6]), "=f"(tv_91[7]) : "r"(fp4_wire_values_2[27 + 0]));
                        #endif
                    }
                    float2 a_364 = make_float2(acc_5[24], acc_5[25]);
                    float2 q_364 = make_float2(tv_91[0], tv_91[1]);
                    float2 r_364 = fma_f32x2_rn_noftz(q_364, scale1_18, a_364);
                    acc_5[24] = r_364.x;
                    acc_5[25] = r_364.y;
                    float2 a_365 = make_float2(acc_5[26], acc_5[27]);
                    float2 q_365 = make_float2(tv_91[2], tv_91[3]);
                    float2 r_365 = fma_f32x2_rn_noftz(q_365, scale1_18, a_365);
                    acc_5[26] = r_365.x;
                    acc_5[27] = r_365.y;
                    float2 a_366 = make_float2(acc_5[28], acc_5[29]);
                    float2 q_366 = make_float2(tv_91[4], tv_91[5]);
                    float2 r_366 = fma_f32x2_rn_noftz(q_366, scale1_18, a_366);
                    acc_5[28] = r_366.x;
                    acc_5[29] = r_366.y;
                    float2 a_367 = make_float2(acc_5[30], acc_5[31]);
                    float2 q_367 = make_float2(tv_91[6], tv_91[7]);
                    float2 r_367 = fma_f32x2_rn_noftz(q_367, scale1_18, a_367);
                    acc_5[30] = r_367.x;
                    acc_5[31] = r_367.y;
                    uint32_t e4m3x2_to_f16x2_20;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_20) : "h"((uint16_t)(fp4_wire_scales_2[7])));
                    float cvt_f32_f16_21;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_21) : "h"((uint16_t)(e4m3x2_to_f16x2_20)));
                    float cvt_f32_f16_1_20;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_20) : "h"((uint16_t)(e4m3x2_to_f16x2_20 >> 16)));
                    float s0_19;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_19) : "f"(cvt_f32_f16_21), "f"(g_s_3));
                    float s1_19;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_19) : "f"(cvt_f32_f16_1_20), "f"(g_s_3));
                    float2 scale0_19 = make_float2(s0_19, s0_19);
                    float2 scale1_19 = make_float2(s1_19, s1_19);
                    float tv_92[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[28 + 0]));
                        (&tv_92[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_92[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_92[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_92[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_92[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_92[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_92[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_92[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_92[0]), "=f"(tv_92[1]), "=f"(tv_92[2]), "=f"(tv_92[3]), "=f"(tv_92[4]), "=f"(tv_92[5]), "=f"(tv_92[6]), "=f"(tv_92[7]) : "r"(fp4_wire_values_2[28 + 0]));
                        #endif
                    }
                    float2 a_368 = make_float2(acc_5[0], acc_5[1]);
                    float2 q_368 = make_float2(tv_92[0], tv_92[1]);
                    float2 r_368 = fma_f32x2_rn_noftz(q_368, scale0_19, a_368);
                    acc_5[0] = r_368.x;
                    acc_5[1] = r_368.y;
                    float2 a_369 = make_float2(acc_5[2], acc_5[3]);
                    float2 q_369 = make_float2(tv_92[2], tv_92[3]);
                    float2 r_369 = fma_f32x2_rn_noftz(q_369, scale0_19, a_369);
                    acc_5[2] = r_369.x;
                    acc_5[3] = r_369.y;
                    float2 a_370 = make_float2(acc_5[4], acc_5[5]);
                    float2 q_370 = make_float2(tv_92[4], tv_92[5]);
                    float2 r_370 = fma_f32x2_rn_noftz(q_370, scale0_19, a_370);
                    acc_5[4] = r_370.x;
                    acc_5[5] = r_370.y;
                    float2 a_371 = make_float2(acc_5[6], acc_5[7]);
                    float2 q_371 = make_float2(tv_92[6], tv_92[7]);
                    float2 r_371 = fma_f32x2_rn_noftz(q_371, scale0_19, a_371);
                    acc_5[6] = r_371.x;
                    acc_5[7] = r_371.y;
                    float tv_93[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[29 + 0]));
                        (&tv_93[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_93[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_93[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_93[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_93[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_93[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_93[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_93[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_93[0]), "=f"(tv_93[1]), "=f"(tv_93[2]), "=f"(tv_93[3]), "=f"(tv_93[4]), "=f"(tv_93[5]), "=f"(tv_93[6]), "=f"(tv_93[7]) : "r"(fp4_wire_values_2[29 + 0]));
                        #endif
                    }
                    float2 a_372 = make_float2(acc_5[8], acc_5[9]);
                    float2 q_372 = make_float2(tv_93[0], tv_93[1]);
                    float2 r_372 = fma_f32x2_rn_noftz(q_372, scale0_19, a_372);
                    acc_5[8] = r_372.x;
                    acc_5[9] = r_372.y;
                    float2 a_373 = make_float2(acc_5[10], acc_5[11]);
                    float2 q_373 = make_float2(tv_93[2], tv_93[3]);
                    float2 r_373 = fma_f32x2_rn_noftz(q_373, scale0_19, a_373);
                    acc_5[10] = r_373.x;
                    acc_5[11] = r_373.y;
                    float2 a_374 = make_float2(acc_5[12], acc_5[13]);
                    float2 q_374 = make_float2(tv_93[4], tv_93[5]);
                    float2 r_374 = fma_f32x2_rn_noftz(q_374, scale0_19, a_374);
                    acc_5[12] = r_374.x;
                    acc_5[13] = r_374.y;
                    float2 a_375 = make_float2(acc_5[14], acc_5[15]);
                    float2 q_375 = make_float2(tv_93[6], tv_93[7]);
                    float2 r_375 = fma_f32x2_rn_noftz(q_375, scale0_19, a_375);
                    acc_5[14] = r_375.x;
                    acc_5[15] = r_375.y;
                    float tv_94[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[30 + 0]));
                        (&tv_94[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_94[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_94[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_94[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_94[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_94[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_94[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_94[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_94[0]), "=f"(tv_94[1]), "=f"(tv_94[2]), "=f"(tv_94[3]), "=f"(tv_94[4]), "=f"(tv_94[5]), "=f"(tv_94[6]), "=f"(tv_94[7]) : "r"(fp4_wire_values_2[30 + 0]));
                        #endif
                    }
                    float2 a_376 = make_float2(acc_5[16], acc_5[17]);
                    float2 q_376 = make_float2(tv_94[0], tv_94[1]);
                    float2 r_376 = fma_f32x2_rn_noftz(q_376, scale1_19, a_376);
                    acc_5[16] = r_376.x;
                    acc_5[17] = r_376.y;
                    float2 a_377 = make_float2(acc_5[18], acc_5[19]);
                    float2 q_377 = make_float2(tv_94[2], tv_94[3]);
                    float2 r_377 = fma_f32x2_rn_noftz(q_377, scale1_19, a_377);
                    acc_5[18] = r_377.x;
                    acc_5[19] = r_377.y;
                    float2 a_378 = make_float2(acc_5[20], acc_5[21]);
                    float2 q_378 = make_float2(tv_94[4], tv_94[5]);
                    float2 r_378 = fma_f32x2_rn_noftz(q_378, scale1_19, a_378);
                    acc_5[20] = r_378.x;
                    acc_5[21] = r_378.y;
                    float2 a_379 = make_float2(acc_5[22], acc_5[23]);
                    float2 q_379 = make_float2(tv_94[6], tv_94[7]);
                    float2 r_379 = fma_f32x2_rn_noftz(q_379, scale1_19, a_379);
                    acc_5[22] = r_379.x;
                    acc_5[23] = r_379.y;
                    float tv_95[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_2[31 + 0]));
                        (&tv_95[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_95[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_95[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_95[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_95[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_95[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_95[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_95[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_95[0]), "=f"(tv_95[1]), "=f"(tv_95[2]), "=f"(tv_95[3]), "=f"(tv_95[4]), "=f"(tv_95[5]), "=f"(tv_95[6]), "=f"(tv_95[7]) : "r"(fp4_wire_values_2[31 + 0]));
                        #endif
                    }
                    float2 a_380 = make_float2(acc_5[24], acc_5[25]);
                    float2 q_380 = make_float2(tv_95[0], tv_95[1]);
                    float2 r_380 = fma_f32x2_rn_noftz(q_380, scale1_19, a_380);
                    acc_5[24] = r_380.x;
                    acc_5[25] = r_380.y;
                    float2 a_381 = make_float2(acc_5[26], acc_5[27]);
                    float2 q_381 = make_float2(tv_95[2], tv_95[3]);
                    float2 r_381 = fma_f32x2_rn_noftz(q_381, scale1_19, a_381);
                    acc_5[26] = r_381.x;
                    acc_5[27] = r_381.y;
                    float2 a_382 = make_float2(acc_5[28], acc_5[29]);
                    float2 q_382 = make_float2(tv_95[4], tv_95[5]);
                    float2 r_382 = fma_f32x2_rn_noftz(q_382, scale1_19, a_382);
                    acc_5[28] = r_382.x;
                    acc_5[29] = r_382.y;
                    float2 a_383 = make_float2(acc_5[30], acc_5[31]);
                    float2 q_383 = make_float2(tv_95[6], tv_95[7]);
                    float2 r_383 = fma_f32x2_rn_noftz(q_383, scale1_19, a_383);
                    acc_5[30] = r_383.x;
                    acc_5[31] = r_383.y;
                    float cur_1_1 = am_1;
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[0]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[1]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[2]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[3]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[4]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[5]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[6]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[7]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[8]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[9]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[10]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[11]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[12]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[13]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[14]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[15]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[16]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[17]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[18]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[19]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[20]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[21]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[22]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[23]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[24]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[25]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[26]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[27]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[28]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[29]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[30]));
                    cur_1_1 = fmaxf(cur_1_1, fabsf(acc_5[31]));
                    am_1 = cur_1_1;
                }
                float rs2_1 = 1.0f;
                unsigned int sw_1[4];
                unsigned int vec_load_7[1];
                {
                    uint32_t _scalar_bits_46;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_46) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_4 + (int)(4 * (lane_1 >> 1)))) + (0))) : "memory");
                    vec_load_7[0] = (unsigned int)_scalar_bits_46;
                }
                sw_1[0] = vec_load_7[0];
                unsigned int vec_load_7_1[1];
                {
                    uint32_t _scalar_bits_47;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_47) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_1_1 + (int)(4 * (lane_1 >> 1)))) + (0))) : "memory");
                    vec_load_7_1[0] = (unsigned int)_scalar_bits_47;
                }
                sw_1[1] = vec_load_7_1[0];
                unsigned int vec_load_8[1];
                {
                    uint32_t _scalar_bits_48;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_48) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_2_1 + (int)(4 * (lane_1 >> 1)))) + (0))) : "memory");
                    vec_load_8[0] = (unsigned int)_scalar_bits_48;
                }
                sw_1[2] = vec_load_8[0];
                unsigned int vec_load_9[1];
                {
                    uint32_t _scalar_bits_49;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_49) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_3_1 + (int)(4 * (lane_1 >> 1)))) + (0))) : "memory");
                    vec_load_9[0] = (unsigned int)_scalar_bits_49;
                }
                sw_1[3] = vec_load_9[0];
                float _warp_reduce_50 = am_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_50 = max_noftz(_warp_reduce_50, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_50, offset));
                am_1 = _warp_reduce_50;
                float fdiv_rn_2 = __fdiv_rn(2688.0f, am_1);
                float inv_1 = 0.0f;
                if (am_1 > 0.0f) {
                    inv_1 = fdiv_rn_2;
                }
                float rs_1 = __fdiv_rn(am_1, 2688.0f);
                rs2_1 = rs_1;
                #pragma unroll 1
                for (int j_1 = 0; j_1 < 6; j_1++) {
                    int pp_1 = lane_1 + 32 * j_1;
                    unsigned int c_20 = sw_1[0] >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    unsigned int c_21 = sw_1[1] >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    unsigned int c_22 = sw_1[2] >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    unsigned int c_23 = sw_1[3] >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    if (j_1 + 1 < 6) {
                        unsigned int vec_load_10[1];
                        {
                            uint32_t _scalar_bits_51;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_51) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_4 + (int)(4 * (pp_1 + 32 >> 1)))) + (0))) : "memory");
                            vec_load_10[0] = (unsigned int)_scalar_bits_51;
                        }
                        sw_1[0] = vec_load_10[0];
                        unsigned int vec_load_11[1];
                        {
                            uint32_t _scalar_bits_52;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_52) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_1_1 + (int)(4 * (pp_1 + 32 >> 1)))) + (0))) : "memory");
                            vec_load_11[0] = (unsigned int)_scalar_bits_52;
                        }
                        sw_1[1] = vec_load_11[0];
                        unsigned int vec_load_12[1];
                        {
                            uint32_t _scalar_bits_53;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_53) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_2_1 + (int)(4 * (pp_1 + 32 >> 1)))) + (0))) : "memory");
                            vec_load_12[0] = (unsigned int)_scalar_bits_53;
                        }
                        sw_1[2] = vec_load_12[0];
                        unsigned int vec_load_13[1];
                        {
                            uint32_t _scalar_bits_54;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_54) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_3_1 + (int)(4 * (pp_1 + 32 >> 1)))) + (0))) : "memory");
                            vec_load_13[0] = (unsigned int)_scalar_bits_54;
                        }
                        sw_1[3] = vec_load_13[0];
                    }
                    int rbase_1 = ret_lane_1 + 2048 * j_1;
                    unsigned int w4_26[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_26[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)rbase_1));
                    unsigned int w4_27[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_27[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rbase_1 + 512)));
                    unsigned int w4_28[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_28[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rbase_1 + 1024)));
                    unsigned int w4_29[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_29[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rbase_1 + 1536)));
                    float acc2_1[32] = {0};
                    uint32_t e4m3x2_to_f16x2_21;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_21) : "h"((uint16_t)(c_20)));
                    float cvt_f32_f16_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_22) : "h"((uint16_t)(e4m3x2_to_f16x2_21)));
                    float cvt_f32_f16_1_21;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_21) : "h"((uint16_t)(e4m3x2_to_f16x2_21 >> 16)));
                    float s0_20;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_20) : "f"(cvt_f32_f16_22), "f"(g_s_0));
                    float s1_20;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_20) : "f"(cvt_f32_f16_1_21), "f"(g_s_0));
                    float2 scale0_20 = make_float2(s0_20, s0_20);
                    float2 scale1_20 = make_float2(s1_20, s1_20);
                    float tv_96[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_26[0]));
                        (&tv_96[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_96[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_96[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_96[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_96[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_96[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_96[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_96[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_96[0]), "=f"(tv_96[1]), "=f"(tv_96[2]), "=f"(tv_96[3]), "=f"(tv_96[4]), "=f"(tv_96[5]), "=f"(tv_96[6]), "=f"(tv_96[7]) : "r"(w4_26[0]));
                        #endif
                    }
                    float2 a_384 = make_float2(acc2_1[0], acc2_1[1]);
                    float2 q_384 = make_float2(tv_96[0], tv_96[1]);
                    float2 r_384 = fma_f32x2_rn_noftz(q_384, scale0_20, a_384);
                    acc2_1[0] = r_384.x;
                    acc2_1[1] = r_384.y;
                    float2 a_385 = make_float2(acc2_1[2], acc2_1[3]);
                    float2 q_385 = make_float2(tv_96[2], tv_96[3]);
                    float2 r_385 = fma_f32x2_rn_noftz(q_385, scale0_20, a_385);
                    acc2_1[2] = r_385.x;
                    acc2_1[3] = r_385.y;
                    float2 a_386 = make_float2(acc2_1[4], acc2_1[5]);
                    float2 q_386 = make_float2(tv_96[4], tv_96[5]);
                    float2 r_386 = fma_f32x2_rn_noftz(q_386, scale0_20, a_386);
                    acc2_1[4] = r_386.x;
                    acc2_1[5] = r_386.y;
                    float2 a_387 = make_float2(acc2_1[6], acc2_1[7]);
                    float2 q_387 = make_float2(tv_96[6], tv_96[7]);
                    float2 r_387 = fma_f32x2_rn_noftz(q_387, scale0_20, a_387);
                    acc2_1[6] = r_387.x;
                    acc2_1[7] = r_387.y;
                    float tv_97[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_26[1 + 0]));
                        (&tv_97[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_97[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_97[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_97[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_97[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_97[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_97[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_97[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_97[0]), "=f"(tv_97[1]), "=f"(tv_97[2]), "=f"(tv_97[3]), "=f"(tv_97[4]), "=f"(tv_97[5]), "=f"(tv_97[6]), "=f"(tv_97[7]) : "r"(w4_26[1 + 0]));
                        #endif
                    }
                    float2 a_388 = make_float2(acc2_1[8], acc2_1[9]);
                    float2 q_388 = make_float2(tv_97[0], tv_97[1]);
                    float2 r_388 = fma_f32x2_rn_noftz(q_388, scale0_20, a_388);
                    acc2_1[8] = r_388.x;
                    acc2_1[9] = r_388.y;
                    float2 a_389 = make_float2(acc2_1[10], acc2_1[11]);
                    float2 q_389 = make_float2(tv_97[2], tv_97[3]);
                    float2 r_389 = fma_f32x2_rn_noftz(q_389, scale0_20, a_389);
                    acc2_1[10] = r_389.x;
                    acc2_1[11] = r_389.y;
                    float2 a_390 = make_float2(acc2_1[12], acc2_1[13]);
                    float2 q_390 = make_float2(tv_97[4], tv_97[5]);
                    float2 r_390 = fma_f32x2_rn_noftz(q_390, scale0_20, a_390);
                    acc2_1[12] = r_390.x;
                    acc2_1[13] = r_390.y;
                    float2 a_391 = make_float2(acc2_1[14], acc2_1[15]);
                    float2 q_391 = make_float2(tv_97[6], tv_97[7]);
                    float2 r_391 = fma_f32x2_rn_noftz(q_391, scale0_20, a_391);
                    acc2_1[14] = r_391.x;
                    acc2_1[15] = r_391.y;
                    float tv_98[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_26[2 + 0]));
                        (&tv_98[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_98[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_98[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_98[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_98[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_98[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_98[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_98[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_98[0]), "=f"(tv_98[1]), "=f"(tv_98[2]), "=f"(tv_98[3]), "=f"(tv_98[4]), "=f"(tv_98[5]), "=f"(tv_98[6]), "=f"(tv_98[7]) : "r"(w4_26[2 + 0]));
                        #endif
                    }
                    float2 a_392 = make_float2(acc2_1[16], acc2_1[17]);
                    float2 q_392 = make_float2(tv_98[0], tv_98[1]);
                    float2 r_392 = fma_f32x2_rn_noftz(q_392, scale1_20, a_392);
                    acc2_1[16] = r_392.x;
                    acc2_1[17] = r_392.y;
                    float2 a_393 = make_float2(acc2_1[18], acc2_1[19]);
                    float2 q_393 = make_float2(tv_98[2], tv_98[3]);
                    float2 r_393 = fma_f32x2_rn_noftz(q_393, scale1_20, a_393);
                    acc2_1[18] = r_393.x;
                    acc2_1[19] = r_393.y;
                    float2 a_394 = make_float2(acc2_1[20], acc2_1[21]);
                    float2 q_394 = make_float2(tv_98[4], tv_98[5]);
                    float2 r_394 = fma_f32x2_rn_noftz(q_394, scale1_20, a_394);
                    acc2_1[20] = r_394.x;
                    acc2_1[21] = r_394.y;
                    float2 a_395 = make_float2(acc2_1[22], acc2_1[23]);
                    float2 q_395 = make_float2(tv_98[6], tv_98[7]);
                    float2 r_395 = fma_f32x2_rn_noftz(q_395, scale1_20, a_395);
                    acc2_1[22] = r_395.x;
                    acc2_1[23] = r_395.y;
                    float tv_99[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_26[3 + 0]));
                        (&tv_99[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_99[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_99[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_99[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_99[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_99[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_99[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_99[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_99[0]), "=f"(tv_99[1]), "=f"(tv_99[2]), "=f"(tv_99[3]), "=f"(tv_99[4]), "=f"(tv_99[5]), "=f"(tv_99[6]), "=f"(tv_99[7]) : "r"(w4_26[3 + 0]));
                        #endif
                    }
                    float2 a_396 = make_float2(acc2_1[24], acc2_1[25]);
                    float2 q_396 = make_float2(tv_99[0], tv_99[1]);
                    float2 r_396 = fma_f32x2_rn_noftz(q_396, scale1_20, a_396);
                    acc2_1[24] = r_396.x;
                    acc2_1[25] = r_396.y;
                    float2 a_397 = make_float2(acc2_1[26], acc2_1[27]);
                    float2 q_397 = make_float2(tv_99[2], tv_99[3]);
                    float2 r_397 = fma_f32x2_rn_noftz(q_397, scale1_20, a_397);
                    acc2_1[26] = r_397.x;
                    acc2_1[27] = r_397.y;
                    float2 a_398 = make_float2(acc2_1[28], acc2_1[29]);
                    float2 q_398 = make_float2(tv_99[4], tv_99[5]);
                    float2 r_398 = fma_f32x2_rn_noftz(q_398, scale1_20, a_398);
                    acc2_1[28] = r_398.x;
                    acc2_1[29] = r_398.y;
                    float2 a_399 = make_float2(acc2_1[30], acc2_1[31]);
                    float2 q_399 = make_float2(tv_99[6], tv_99[7]);
                    float2 r_399 = fma_f32x2_rn_noftz(q_399, scale1_20, a_399);
                    acc2_1[30] = r_399.x;
                    acc2_1[31] = r_399.y;
                    uint32_t e4m3x2_to_f16x2_21_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_21_1) : "h"((uint16_t)(c_21)));
                    float cvt_f32_f16_22_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_22_1) : "h"((uint16_t)(e4m3x2_to_f16x2_21_1)));
                    float cvt_f32_f16_1_21_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_21_1) : "h"((uint16_t)(e4m3x2_to_f16x2_21_1 >> 16)));
                    float s0_20_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_20_1) : "f"(cvt_f32_f16_22_1), "f"(g_s_1));
                    float s1_20_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_20_1) : "f"(cvt_f32_f16_1_21_1), "f"(g_s_1));
                    float2 scale0_20_1 = make_float2(s0_20_1, s0_20_1);
                    float2 scale1_20_1 = make_float2(s1_20_1, s1_20_1);
                    float tv_100[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_27[0]));
                        (&tv_100[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_100[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_100[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_100[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_100[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_100[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_100[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_100[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_100[0]), "=f"(tv_100[1]), "=f"(tv_100[2]), "=f"(tv_100[3]), "=f"(tv_100[4]), "=f"(tv_100[5]), "=f"(tv_100[6]), "=f"(tv_100[7]) : "r"(w4_27[0]));
                        #endif
                    }
                    float2 a_400 = make_float2(acc2_1[0], acc2_1[1]);
                    float2 q_400 = make_float2(tv_100[0], tv_100[1]);
                    float2 r_400 = fma_f32x2_rn_noftz(q_400, scale0_20_1, a_400);
                    acc2_1[0] = r_400.x;
                    acc2_1[1] = r_400.y;
                    float2 a_401 = make_float2(acc2_1[2], acc2_1[3]);
                    float2 q_401 = make_float2(tv_100[2], tv_100[3]);
                    float2 r_401 = fma_f32x2_rn_noftz(q_401, scale0_20_1, a_401);
                    acc2_1[2] = r_401.x;
                    acc2_1[3] = r_401.y;
                    float2 a_402 = make_float2(acc2_1[4], acc2_1[5]);
                    float2 q_402 = make_float2(tv_100[4], tv_100[5]);
                    float2 r_402 = fma_f32x2_rn_noftz(q_402, scale0_20_1, a_402);
                    acc2_1[4] = r_402.x;
                    acc2_1[5] = r_402.y;
                    float2 a_403 = make_float2(acc2_1[6], acc2_1[7]);
                    float2 q_403 = make_float2(tv_100[6], tv_100[7]);
                    float2 r_403 = fma_f32x2_rn_noftz(q_403, scale0_20_1, a_403);
                    acc2_1[6] = r_403.x;
                    acc2_1[7] = r_403.y;
                    float tv_101[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_27[1 + 0]));
                        (&tv_101[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_101[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_101[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_101[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_101[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_101[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_101[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_101[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_101[0]), "=f"(tv_101[1]), "=f"(tv_101[2]), "=f"(tv_101[3]), "=f"(tv_101[4]), "=f"(tv_101[5]), "=f"(tv_101[6]), "=f"(tv_101[7]) : "r"(w4_27[1 + 0]));
                        #endif
                    }
                    float2 a_404 = make_float2(acc2_1[8], acc2_1[9]);
                    float2 q_404 = make_float2(tv_101[0], tv_101[1]);
                    float2 r_404 = fma_f32x2_rn_noftz(q_404, scale0_20_1, a_404);
                    acc2_1[8] = r_404.x;
                    acc2_1[9] = r_404.y;
                    float2 a_405 = make_float2(acc2_1[10], acc2_1[11]);
                    float2 q_405 = make_float2(tv_101[2], tv_101[3]);
                    float2 r_405 = fma_f32x2_rn_noftz(q_405, scale0_20_1, a_405);
                    acc2_1[10] = r_405.x;
                    acc2_1[11] = r_405.y;
                    float2 a_406 = make_float2(acc2_1[12], acc2_1[13]);
                    float2 q_406 = make_float2(tv_101[4], tv_101[5]);
                    float2 r_406 = fma_f32x2_rn_noftz(q_406, scale0_20_1, a_406);
                    acc2_1[12] = r_406.x;
                    acc2_1[13] = r_406.y;
                    float2 a_407 = make_float2(acc2_1[14], acc2_1[15]);
                    float2 q_407 = make_float2(tv_101[6], tv_101[7]);
                    float2 r_407 = fma_f32x2_rn_noftz(q_407, scale0_20_1, a_407);
                    acc2_1[14] = r_407.x;
                    acc2_1[15] = r_407.y;
                    float tv_102[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_27[2 + 0]));
                        (&tv_102[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_102[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_102[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_102[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_102[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_102[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_102[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_102[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_102[0]), "=f"(tv_102[1]), "=f"(tv_102[2]), "=f"(tv_102[3]), "=f"(tv_102[4]), "=f"(tv_102[5]), "=f"(tv_102[6]), "=f"(tv_102[7]) : "r"(w4_27[2 + 0]));
                        #endif
                    }
                    float2 a_408 = make_float2(acc2_1[16], acc2_1[17]);
                    float2 q_408 = make_float2(tv_102[0], tv_102[1]);
                    float2 r_408 = fma_f32x2_rn_noftz(q_408, scale1_20_1, a_408);
                    acc2_1[16] = r_408.x;
                    acc2_1[17] = r_408.y;
                    float2 a_409 = make_float2(acc2_1[18], acc2_1[19]);
                    float2 q_409 = make_float2(tv_102[2], tv_102[3]);
                    float2 r_409 = fma_f32x2_rn_noftz(q_409, scale1_20_1, a_409);
                    acc2_1[18] = r_409.x;
                    acc2_1[19] = r_409.y;
                    float2 a_410 = make_float2(acc2_1[20], acc2_1[21]);
                    float2 q_410 = make_float2(tv_102[4], tv_102[5]);
                    float2 r_410 = fma_f32x2_rn_noftz(q_410, scale1_20_1, a_410);
                    acc2_1[20] = r_410.x;
                    acc2_1[21] = r_410.y;
                    float2 a_411 = make_float2(acc2_1[22], acc2_1[23]);
                    float2 q_411 = make_float2(tv_102[6], tv_102[7]);
                    float2 r_411 = fma_f32x2_rn_noftz(q_411, scale1_20_1, a_411);
                    acc2_1[22] = r_411.x;
                    acc2_1[23] = r_411.y;
                    float tv_103[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_27[3 + 0]));
                        (&tv_103[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_103[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_103[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_103[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_103[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_103[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_103[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_103[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_103[0]), "=f"(tv_103[1]), "=f"(tv_103[2]), "=f"(tv_103[3]), "=f"(tv_103[4]), "=f"(tv_103[5]), "=f"(tv_103[6]), "=f"(tv_103[7]) : "r"(w4_27[3 + 0]));
                        #endif
                    }
                    float2 a_412 = make_float2(acc2_1[24], acc2_1[25]);
                    float2 q_412 = make_float2(tv_103[0], tv_103[1]);
                    float2 r_412 = fma_f32x2_rn_noftz(q_412, scale1_20_1, a_412);
                    acc2_1[24] = r_412.x;
                    acc2_1[25] = r_412.y;
                    float2 a_413 = make_float2(acc2_1[26], acc2_1[27]);
                    float2 q_413 = make_float2(tv_103[2], tv_103[3]);
                    float2 r_413 = fma_f32x2_rn_noftz(q_413, scale1_20_1, a_413);
                    acc2_1[26] = r_413.x;
                    acc2_1[27] = r_413.y;
                    float2 a_414 = make_float2(acc2_1[28], acc2_1[29]);
                    float2 q_414 = make_float2(tv_103[4], tv_103[5]);
                    float2 r_414 = fma_f32x2_rn_noftz(q_414, scale1_20_1, a_414);
                    acc2_1[28] = r_414.x;
                    acc2_1[29] = r_414.y;
                    float2 a_415 = make_float2(acc2_1[30], acc2_1[31]);
                    float2 q_415 = make_float2(tv_103[6], tv_103[7]);
                    float2 r_415 = fma_f32x2_rn_noftz(q_415, scale1_20_1, a_415);
                    acc2_1[30] = r_415.x;
                    acc2_1[31] = r_415.y;
                    uint32_t e4m3x2_to_f16x2_22;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_22) : "h"((uint16_t)(c_22)));
                    float cvt_f32_f16_23;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_23) : "h"((uint16_t)(e4m3x2_to_f16x2_22)));
                    float cvt_f32_f16_1_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_22) : "h"((uint16_t)(e4m3x2_to_f16x2_22 >> 16)));
                    float s0_21;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_21) : "f"(cvt_f32_f16_23), "f"(g_s_2));
                    float s1_21;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_21) : "f"(cvt_f32_f16_1_22), "f"(g_s_2));
                    float2 scale0_21 = make_float2(s0_21, s0_21);
                    float2 scale1_21 = make_float2(s1_21, s1_21);
                    float tv_104[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_28[0]));
                        (&tv_104[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_104[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_104[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_104[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_104[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_104[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_104[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_104[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_104[0]), "=f"(tv_104[1]), "=f"(tv_104[2]), "=f"(tv_104[3]), "=f"(tv_104[4]), "=f"(tv_104[5]), "=f"(tv_104[6]), "=f"(tv_104[7]) : "r"(w4_28[0]));
                        #endif
                    }
                    float2 a_416 = make_float2(acc2_1[0], acc2_1[1]);
                    float2 q_416 = make_float2(tv_104[0], tv_104[1]);
                    float2 r_416 = fma_f32x2_rn_noftz(q_416, scale0_21, a_416);
                    acc2_1[0] = r_416.x;
                    acc2_1[1] = r_416.y;
                    float2 a_417 = make_float2(acc2_1[2], acc2_1[3]);
                    float2 q_417 = make_float2(tv_104[2], tv_104[3]);
                    float2 r_417 = fma_f32x2_rn_noftz(q_417, scale0_21, a_417);
                    acc2_1[2] = r_417.x;
                    acc2_1[3] = r_417.y;
                    float2 a_418 = make_float2(acc2_1[4], acc2_1[5]);
                    float2 q_418 = make_float2(tv_104[4], tv_104[5]);
                    float2 r_418 = fma_f32x2_rn_noftz(q_418, scale0_21, a_418);
                    acc2_1[4] = r_418.x;
                    acc2_1[5] = r_418.y;
                    float2 a_419 = make_float2(acc2_1[6], acc2_1[7]);
                    float2 q_419 = make_float2(tv_104[6], tv_104[7]);
                    float2 r_419 = fma_f32x2_rn_noftz(q_419, scale0_21, a_419);
                    acc2_1[6] = r_419.x;
                    acc2_1[7] = r_419.y;
                    float tv_105[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_28[1 + 0]));
                        (&tv_105[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_105[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_105[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_105[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_105[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_105[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_105[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_105[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_105[0]), "=f"(tv_105[1]), "=f"(tv_105[2]), "=f"(tv_105[3]), "=f"(tv_105[4]), "=f"(tv_105[5]), "=f"(tv_105[6]), "=f"(tv_105[7]) : "r"(w4_28[1 + 0]));
                        #endif
                    }
                    float2 a_420 = make_float2(acc2_1[8], acc2_1[9]);
                    float2 q_420 = make_float2(tv_105[0], tv_105[1]);
                    float2 r_420 = fma_f32x2_rn_noftz(q_420, scale0_21, a_420);
                    acc2_1[8] = r_420.x;
                    acc2_1[9] = r_420.y;
                    float2 a_421 = make_float2(acc2_1[10], acc2_1[11]);
                    float2 q_421 = make_float2(tv_105[2], tv_105[3]);
                    float2 r_421 = fma_f32x2_rn_noftz(q_421, scale0_21, a_421);
                    acc2_1[10] = r_421.x;
                    acc2_1[11] = r_421.y;
                    float2 a_422 = make_float2(acc2_1[12], acc2_1[13]);
                    float2 q_422 = make_float2(tv_105[4], tv_105[5]);
                    float2 r_422 = fma_f32x2_rn_noftz(q_422, scale0_21, a_422);
                    acc2_1[12] = r_422.x;
                    acc2_1[13] = r_422.y;
                    float2 a_423 = make_float2(acc2_1[14], acc2_1[15]);
                    float2 q_423 = make_float2(tv_105[6], tv_105[7]);
                    float2 r_423 = fma_f32x2_rn_noftz(q_423, scale0_21, a_423);
                    acc2_1[14] = r_423.x;
                    acc2_1[15] = r_423.y;
                    float tv_106[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_28[2 + 0]));
                        (&tv_106[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_106[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_106[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_106[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_106[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_106[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_106[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_106[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_106[0]), "=f"(tv_106[1]), "=f"(tv_106[2]), "=f"(tv_106[3]), "=f"(tv_106[4]), "=f"(tv_106[5]), "=f"(tv_106[6]), "=f"(tv_106[7]) : "r"(w4_28[2 + 0]));
                        #endif
                    }
                    float2 a_424 = make_float2(acc2_1[16], acc2_1[17]);
                    float2 q_424 = make_float2(tv_106[0], tv_106[1]);
                    float2 r_424 = fma_f32x2_rn_noftz(q_424, scale1_21, a_424);
                    acc2_1[16] = r_424.x;
                    acc2_1[17] = r_424.y;
                    float2 a_425 = make_float2(acc2_1[18], acc2_1[19]);
                    float2 q_425 = make_float2(tv_106[2], tv_106[3]);
                    float2 r_425 = fma_f32x2_rn_noftz(q_425, scale1_21, a_425);
                    acc2_1[18] = r_425.x;
                    acc2_1[19] = r_425.y;
                    float2 a_426 = make_float2(acc2_1[20], acc2_1[21]);
                    float2 q_426 = make_float2(tv_106[4], tv_106[5]);
                    float2 r_426 = fma_f32x2_rn_noftz(q_426, scale1_21, a_426);
                    acc2_1[20] = r_426.x;
                    acc2_1[21] = r_426.y;
                    float2 a_427 = make_float2(acc2_1[22], acc2_1[23]);
                    float2 q_427 = make_float2(tv_106[6], tv_106[7]);
                    float2 r_427 = fma_f32x2_rn_noftz(q_427, scale1_21, a_427);
                    acc2_1[22] = r_427.x;
                    acc2_1[23] = r_427.y;
                    float tv_107[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_28[3 + 0]));
                        (&tv_107[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_107[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_107[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_107[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_107[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_107[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_107[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_107[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_107[0]), "=f"(tv_107[1]), "=f"(tv_107[2]), "=f"(tv_107[3]), "=f"(tv_107[4]), "=f"(tv_107[5]), "=f"(tv_107[6]), "=f"(tv_107[7]) : "r"(w4_28[3 + 0]));
                        #endif
                    }
                    float2 a_428 = make_float2(acc2_1[24], acc2_1[25]);
                    float2 q_428 = make_float2(tv_107[0], tv_107[1]);
                    float2 r_428 = fma_f32x2_rn_noftz(q_428, scale1_21, a_428);
                    acc2_1[24] = r_428.x;
                    acc2_1[25] = r_428.y;
                    float2 a_429 = make_float2(acc2_1[26], acc2_1[27]);
                    float2 q_429 = make_float2(tv_107[2], tv_107[3]);
                    float2 r_429 = fma_f32x2_rn_noftz(q_429, scale1_21, a_429);
                    acc2_1[26] = r_429.x;
                    acc2_1[27] = r_429.y;
                    float2 a_430 = make_float2(acc2_1[28], acc2_1[29]);
                    float2 q_430 = make_float2(tv_107[4], tv_107[5]);
                    float2 r_430 = fma_f32x2_rn_noftz(q_430, scale1_21, a_430);
                    acc2_1[28] = r_430.x;
                    acc2_1[29] = r_430.y;
                    float2 a_431 = make_float2(acc2_1[30], acc2_1[31]);
                    float2 q_431 = make_float2(tv_107[6], tv_107[7]);
                    float2 r_431 = fma_f32x2_rn_noftz(q_431, scale1_21, a_431);
                    acc2_1[30] = r_431.x;
                    acc2_1[31] = r_431.y;
                    uint32_t e4m3x2_to_f16x2_23;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_23) : "h"((uint16_t)(c_23)));
                    float cvt_f32_f16_24;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_24) : "h"((uint16_t)(e4m3x2_to_f16x2_23)));
                    float cvt_f32_f16_1_23;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_23) : "h"((uint16_t)(e4m3x2_to_f16x2_23 >> 16)));
                    float s0_22;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_22) : "f"(cvt_f32_f16_24), "f"(g_s_3));
                    float s1_22;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_22) : "f"(cvt_f32_f16_1_23), "f"(g_s_3));
                    float2 scale0_22 = make_float2(s0_22, s0_22);
                    float2 scale1_22 = make_float2(s1_22, s1_22);
                    float tv_108[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_29[0]));
                        (&tv_108[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_108[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_108[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_108[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_108[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_108[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_108[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_108[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_108[0]), "=f"(tv_108[1]), "=f"(tv_108[2]), "=f"(tv_108[3]), "=f"(tv_108[4]), "=f"(tv_108[5]), "=f"(tv_108[6]), "=f"(tv_108[7]) : "r"(w4_29[0]));
                        #endif
                    }
                    float2 a_432 = make_float2(acc2_1[0], acc2_1[1]);
                    float2 q_432 = make_float2(tv_108[0], tv_108[1]);
                    float2 r_432 = fma_f32x2_rn_noftz(q_432, scale0_22, a_432);
                    acc2_1[0] = r_432.x;
                    acc2_1[1] = r_432.y;
                    float2 a_433 = make_float2(acc2_1[2], acc2_1[3]);
                    float2 q_433 = make_float2(tv_108[2], tv_108[3]);
                    float2 r_433 = fma_f32x2_rn_noftz(q_433, scale0_22, a_433);
                    acc2_1[2] = r_433.x;
                    acc2_1[3] = r_433.y;
                    float2 a_434 = make_float2(acc2_1[4], acc2_1[5]);
                    float2 q_434 = make_float2(tv_108[4], tv_108[5]);
                    float2 r_434 = fma_f32x2_rn_noftz(q_434, scale0_22, a_434);
                    acc2_1[4] = r_434.x;
                    acc2_1[5] = r_434.y;
                    float2 a_435 = make_float2(acc2_1[6], acc2_1[7]);
                    float2 q_435 = make_float2(tv_108[6], tv_108[7]);
                    float2 r_435 = fma_f32x2_rn_noftz(q_435, scale0_22, a_435);
                    acc2_1[6] = r_435.x;
                    acc2_1[7] = r_435.y;
                    float tv_109[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_29[1 + 0]));
                        (&tv_109[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_109[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_109[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_109[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_109[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_109[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_109[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_109[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_109[0]), "=f"(tv_109[1]), "=f"(tv_109[2]), "=f"(tv_109[3]), "=f"(tv_109[4]), "=f"(tv_109[5]), "=f"(tv_109[6]), "=f"(tv_109[7]) : "r"(w4_29[1 + 0]));
                        #endif
                    }
                    float2 a_436 = make_float2(acc2_1[8], acc2_1[9]);
                    float2 q_436 = make_float2(tv_109[0], tv_109[1]);
                    float2 r_436 = fma_f32x2_rn_noftz(q_436, scale0_22, a_436);
                    acc2_1[8] = r_436.x;
                    acc2_1[9] = r_436.y;
                    float2 a_437 = make_float2(acc2_1[10], acc2_1[11]);
                    float2 q_437 = make_float2(tv_109[2], tv_109[3]);
                    float2 r_437 = fma_f32x2_rn_noftz(q_437, scale0_22, a_437);
                    acc2_1[10] = r_437.x;
                    acc2_1[11] = r_437.y;
                    float2 a_438 = make_float2(acc2_1[12], acc2_1[13]);
                    float2 q_438 = make_float2(tv_109[4], tv_109[5]);
                    float2 r_438 = fma_f32x2_rn_noftz(q_438, scale0_22, a_438);
                    acc2_1[12] = r_438.x;
                    acc2_1[13] = r_438.y;
                    float2 a_439 = make_float2(acc2_1[14], acc2_1[15]);
                    float2 q_439 = make_float2(tv_109[6], tv_109[7]);
                    float2 r_439 = fma_f32x2_rn_noftz(q_439, scale0_22, a_439);
                    acc2_1[14] = r_439.x;
                    acc2_1[15] = r_439.y;
                    float tv_110[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_29[2 + 0]));
                        (&tv_110[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_110[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_110[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_110[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_110[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_110[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_110[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_110[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_110[0]), "=f"(tv_110[1]), "=f"(tv_110[2]), "=f"(tv_110[3]), "=f"(tv_110[4]), "=f"(tv_110[5]), "=f"(tv_110[6]), "=f"(tv_110[7]) : "r"(w4_29[2 + 0]));
                        #endif
                    }
                    float2 a_440 = make_float2(acc2_1[16], acc2_1[17]);
                    float2 q_440 = make_float2(tv_110[0], tv_110[1]);
                    float2 r_440 = fma_f32x2_rn_noftz(q_440, scale1_22, a_440);
                    acc2_1[16] = r_440.x;
                    acc2_1[17] = r_440.y;
                    float2 a_441 = make_float2(acc2_1[18], acc2_1[19]);
                    float2 q_441 = make_float2(tv_110[2], tv_110[3]);
                    float2 r_441 = fma_f32x2_rn_noftz(q_441, scale1_22, a_441);
                    acc2_1[18] = r_441.x;
                    acc2_1[19] = r_441.y;
                    float2 a_442 = make_float2(acc2_1[20], acc2_1[21]);
                    float2 q_442 = make_float2(tv_110[4], tv_110[5]);
                    float2 r_442 = fma_f32x2_rn_noftz(q_442, scale1_22, a_442);
                    acc2_1[20] = r_442.x;
                    acc2_1[21] = r_442.y;
                    float2 a_443 = make_float2(acc2_1[22], acc2_1[23]);
                    float2 q_443 = make_float2(tv_110[6], tv_110[7]);
                    float2 r_443 = fma_f32x2_rn_noftz(q_443, scale1_22, a_443);
                    acc2_1[22] = r_443.x;
                    acc2_1[23] = r_443.y;
                    float tv_111[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(w4_29[3 + 0]));
                        (&tv_111[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_111[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_111[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_111[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_111[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_111[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_111[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_111[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_111[0]), "=f"(tv_111[1]), "=f"(tv_111[2]), "=f"(tv_111[3]), "=f"(tv_111[4]), "=f"(tv_111[5]), "=f"(tv_111[6]), "=f"(tv_111[7]) : "r"(w4_29[3 + 0]));
                        #endif
                    }
                    float2 a_444 = make_float2(acc2_1[24], acc2_1[25]);
                    float2 q_444 = make_float2(tv_111[0], tv_111[1]);
                    float2 r_444 = fma_f32x2_rn_noftz(q_444, scale1_22, a_444);
                    acc2_1[24] = r_444.x;
                    acc2_1[25] = r_444.y;
                    float2 a_445 = make_float2(acc2_1[26], acc2_1[27]);
                    float2 q_445 = make_float2(tv_111[2], tv_111[3]);
                    float2 r_445 = fma_f32x2_rn_noftz(q_445, scale1_22, a_445);
                    acc2_1[26] = r_445.x;
                    acc2_1[27] = r_445.y;
                    float2 a_446 = make_float2(acc2_1[28], acc2_1[29]);
                    float2 q_446 = make_float2(tv_111[4], tv_111[5]);
                    float2 r_446 = fma_f32x2_rn_noftz(q_446, scale1_22, a_446);
                    acc2_1[28] = r_446.x;
                    acc2_1[29] = r_446.y;
                    float2 a_447 = make_float2(acc2_1[30], acc2_1[31]);
                    float2 q_447 = make_float2(tv_111[6], tv_111[7]);
                    float2 r_447 = fma_f32x2_rn_noftz(q_447, scale1_22, a_447);
                    acc2_1[30] = r_447.x;
                    acc2_1[31] = r_447.y;
                    unsigned int q_448[4];
                    float bm0_1 = 0.0f;
                    float bm1_1 = 0.0f;
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[0]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[16]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[1]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[17]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[2]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[18]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[3]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[19]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[4]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[20]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[5]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[21]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[6]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[22]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[7]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[23]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[8]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[24]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[9]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[25]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[10]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[26]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[11]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[27]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[12]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[28]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[13]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[29]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[14]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[30]));
                    bm0_1 = fmaxf(bm0_1, fabsf(acc2_1[15]));
                    bm1_1 = fmaxf(bm1_1, fabsf(acc2_1[31]));
                    float _fdiv_55 = __fdiv_rn(mul_f32_rn_noftz(bm1_1, inv_1), 6.0f);
                    float _fdiv_56 = __fdiv_rn(mul_f32_rn_noftz(bm0_1, inv_1), 6.0f);
                    uint16_t e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_1) : "f"(_fdiv_55), "f"(_fdiv_56));
                    uint32_t e4m3x2_to_f16x2_24;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_24) : "h"((uint16_t)(e4m3x2_f32_1)));
                    float cvt_f32_f16_25;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_25) : "h"((uint16_t)(e4m3x2_to_f16x2_24)));
                    float cvt_f32_f16_1_24;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_24) : "h"((uint16_t)(e4m3x2_to_f16x2_24 >> 16)));
                    float fdiv_rn_1_1 = __fdiv_rn(inv_1, cvt_f32_f16_25);
                    float fdiv_rn_1_0 = __fdiv_rn(inv_1, cvt_f32_f16_1_24);
                    float m0_1 = 0.0f;
                    float m1_1 = 0.0f;
                    if (cvt_f32_f16_25 > 0.0f) {
                        m0_1 = fdiv_rn_1_1;
                    }
                    if (cvt_f32_f16_1_24 > 0.0f) {
                        m1_1 = fdiv_rn_1_0;
                    }
                    float2 mm_4 = make_float2(m0_1, m0_1);
                    float2 pair_16 = make_float2(acc2_1[0], acc2_1[1]);
                    float2 g_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_16) : "l"(*(const unsigned long long*)&pair_16), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_12;
                    uint16_t _e2m1_pair_h_57;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(g_16.x), "f"(g_16.y));
                    fp4_12 = (uint32_t)_e2m1_pair_h_57;
                    float2 pair_17 = make_float2(acc2_1[2], acc2_1[3]);
                    float2 g_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_17) : "l"(*(const unsigned long long*)&pair_17), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_13;
                    uint16_t _e2m1_pair_h_58;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_58) : "f"(g_17.x), "f"(g_17.y));
                    fp4_13 = (uint32_t)_e2m1_pair_h_58;
                    float2 pair_18 = make_float2(acc2_1[4], acc2_1[5]);
                    float2 g_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_18) : "l"(*(const unsigned long long*)&pair_18), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_14;
                    uint16_t _e2m1_pair_h_59;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(g_18.x), "f"(g_18.y));
                    fp4_14 = (uint32_t)_e2m1_pair_h_59;
                    float2 pair_19 = make_float2(acc2_1[6], acc2_1[7]);
                    float2 g_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_19) : "l"(*(const unsigned long long*)&pair_19), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t _fp4_e2m1_pack_pair_60;
                    uint16_t _e2m1_pair_h_61;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_61) : "f"(g_19.x), "f"(g_19.y));
                    _fp4_e2m1_pack_pair_60 = (uint32_t)_e2m1_pair_h_61;
                    unsigned int word_4 = fp4_12 | fp4_13 << 8 | fp4_14 << 16 | _fp4_e2m1_pack_pair_60 << 24;
                    unsigned int carry_26 = (word_4 & 2004318071) + 2004318071;
                    q_448[0] = word_4 & (carry_26 | 2004318071);
                    float2 mm_5 = make_float2(m0_1, m0_1);
                    float2 pair_20 = make_float2(acc2_1[8], acc2_1[9]);
                    float2 g_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_20) : "l"(*(const unsigned long long*)&pair_20), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_15;
                    uint16_t _e2m1_pair_h_62;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_62) : "f"(g_20.x), "f"(g_20.y));
                    fp4_15 = (uint32_t)_e2m1_pair_h_62;
                    float2 pair_21 = make_float2(acc2_1[10], acc2_1[11]);
                    float2 g_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_21) : "l"(*(const unsigned long long*)&pair_21), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_16;
                    uint16_t _e2m1_pair_h_63;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_63) : "f"(g_21.x), "f"(g_21.y));
                    fp4_16 = (uint32_t)_e2m1_pair_h_63;
                    float2 pair_22 = make_float2(acc2_1[12], acc2_1[13]);
                    float2 g_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_22) : "l"(*(const unsigned long long*)&pair_22), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_17;
                    uint16_t _e2m1_pair_h_64;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_64) : "f"(g_22.x), "f"(g_22.y));
                    fp4_17 = (uint32_t)_e2m1_pair_h_64;
                    float2 pair_23 = make_float2(acc2_1[14], acc2_1[15]);
                    float2 g_23;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_23) : "l"(*(const unsigned long long*)&pair_23), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t _fp4_e2m1_pack_pair_65;
                    uint16_t _e2m1_pair_h_66;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_66) : "f"(g_23.x), "f"(g_23.y));
                    _fp4_e2m1_pack_pair_65 = (uint32_t)_e2m1_pair_h_66;
                    unsigned int word_5 = fp4_15 | fp4_16 << 8 | fp4_17 << 16 | _fp4_e2m1_pack_pair_65 << 24;
                    unsigned int carry_26_1 = (word_5 & 2004318071) + 2004318071;
                    q_448[1] = word_5 & (carry_26_1 | 2004318071);
                    float2 mm_6 = make_float2(m1_1, m1_1);
                    float2 pair_24 = make_float2(acc2_1[16], acc2_1[17]);
                    float2 g_24;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_24) : "l"(*(const unsigned long long*)&pair_24), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_18;
                    uint16_t _e2m1_pair_h_67;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_67) : "f"(g_24.x), "f"(g_24.y));
                    fp4_18 = (uint32_t)_e2m1_pair_h_67;
                    float2 pair_25 = make_float2(acc2_1[18], acc2_1[19]);
                    float2 g_25;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_25) : "l"(*(const unsigned long long*)&pair_25), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_19;
                    uint16_t _e2m1_pair_h_68;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_68) : "f"(g_25.x), "f"(g_25.y));
                    fp4_19 = (uint32_t)_e2m1_pair_h_68;
                    float2 pair_26 = make_float2(acc2_1[20], acc2_1[21]);
                    float2 g_26;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_26) : "l"(*(const unsigned long long*)&pair_26), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_20;
                    uint16_t _e2m1_pair_h_69;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_69) : "f"(g_26.x), "f"(g_26.y));
                    fp4_20 = (uint32_t)_e2m1_pair_h_69;
                    float2 pair_27 = make_float2(acc2_1[22], acc2_1[23]);
                    float2 g_27;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_27) : "l"(*(const unsigned long long*)&pair_27), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t _fp4_e2m1_pack_pair_70;
                    uint16_t _e2m1_pair_h_71;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_71) : "f"(g_27.x), "f"(g_27.y));
                    _fp4_e2m1_pack_pair_70 = (uint32_t)_e2m1_pair_h_71;
                    unsigned int word_6 = fp4_18 | fp4_19 << 8 | fp4_20 << 16 | _fp4_e2m1_pack_pair_70 << 24;
                    unsigned int carry_27 = (word_6 & 2004318071) + 2004318071;
                    q_448[2] = word_6 & (carry_27 | 2004318071);
                    float2 mm_7 = make_float2(m1_1, m1_1);
                    float2 pair_28 = make_float2(acc2_1[24], acc2_1[25]);
                    float2 g_28;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_28) : "l"(*(const unsigned long long*)&pair_28), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_21;
                    uint16_t _e2m1_pair_h_72;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_72) : "f"(g_28.x), "f"(g_28.y));
                    fp4_21 = (uint32_t)_e2m1_pair_h_72;
                    float2 pair_29 = make_float2(acc2_1[26], acc2_1[27]);
                    float2 g_29;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_29) : "l"(*(const unsigned long long*)&pair_29), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_22;
                    uint16_t _e2m1_pair_h_73;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_73) : "f"(g_29.x), "f"(g_29.y));
                    fp4_22 = (uint32_t)_e2m1_pair_h_73;
                    float2 pair_30 = make_float2(acc2_1[28], acc2_1[29]);
                    float2 g_30;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_30) : "l"(*(const unsigned long long*)&pair_30), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_23;
                    uint16_t _e2m1_pair_h_74;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_74) : "f"(g_30.x), "f"(g_30.y));
                    fp4_23 = (uint32_t)_e2m1_pair_h_74;
                    float2 pair_31 = make_float2(acc2_1[30], acc2_1[31]);
                    float2 g_31;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_31) : "l"(*(const unsigned long long*)&pair_31), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t _fp4_e2m1_pack_pair_75;
                    uint16_t _e2m1_pair_h_76;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_76) : "f"(g_31.x), "f"(g_31.y));
                    _fp4_e2m1_pack_pair_75 = (uint32_t)_e2m1_pair_h_76;
                    unsigned int word_7 = fp4_21 | fp4_22 << 8 | fp4_23 << 16 | _fp4_e2m1_pack_pair_75 << 24;
                    unsigned int carry_28 = (word_7 & 2004318071) + 2004318071;
                    q_448[3] = word_7 & (carry_28 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(xb_off + 16 * pp_1)), "r"(q_448[0]), "r"(q_448[1]), "r"(q_448[2]), "r"(q_448[3]) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(xb_off + 3072 + 2 * pp_1)), "h"((uint16_t)(e4m3x2_f32_1 & 32639)) : "memory");
                    float d_1[32] = {0};
                    uint32_t e4m3x2_to_f16x2_25;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_25) : "h"((uint16_t)(e4m3x2_f32_1 & 32639)));
                    float cvt_f32_f16_26;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_26) : "h"((uint16_t)(e4m3x2_to_f16x2_25)));
                    float cvt_f32_f16_1_25;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_25) : "h"((uint16_t)(e4m3x2_to_f16x2_25 >> 16)));
                    float s0_23;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_23) : "f"(cvt_f32_f16_26), "f"(rs2_1));
                    float s1_23;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_23) : "f"(cvt_f32_f16_1_25), "f"(rs2_1));
                    float2 scale0_23 = make_float2(s0_23, s0_23);
                    float2 scale1_23 = make_float2(s1_23, s1_23);
                    float tv_112[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_448[0]));
                        (&tv_112[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_112[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_112[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_112[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_112[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_112[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_112[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_112[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_112[0]), "=f"(tv_112[1]), "=f"(tv_112[2]), "=f"(tv_112[3]), "=f"(tv_112[4]), "=f"(tv_112[5]), "=f"(tv_112[6]), "=f"(tv_112[7]) : "r"(q_448[0]));
                        #endif
                    }
                    float2 a_448 = make_float2(d_1[0], d_1[1]);
                    float2 q_448_1 = make_float2(tv_112[0], tv_112[1]);
                    float2 r_448 = fma_f32x2_rn_noftz(q_448_1, scale0_23, a_448);
                    d_1[0] = r_448.x;
                    d_1[1] = r_448.y;
                    float2 a_449 = make_float2(d_1[2], d_1[3]);
                    float2 q_449 = make_float2(tv_112[2], tv_112[3]);
                    float2 r_449 = fma_f32x2_rn_noftz(q_449, scale0_23, a_449);
                    d_1[2] = r_449.x;
                    d_1[3] = r_449.y;
                    float2 a_450 = make_float2(d_1[4], d_1[5]);
                    float2 q_450 = make_float2(tv_112[4], tv_112[5]);
                    float2 r_450 = fma_f32x2_rn_noftz(q_450, scale0_23, a_450);
                    d_1[4] = r_450.x;
                    d_1[5] = r_450.y;
                    float2 a_451 = make_float2(d_1[6], d_1[7]);
                    float2 q_451 = make_float2(tv_112[6], tv_112[7]);
                    float2 r_451 = fma_f32x2_rn_noftz(q_451, scale0_23, a_451);
                    d_1[6] = r_451.x;
                    d_1[7] = r_451.y;
                    float tv_113[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_448[1 + 0]));
                        (&tv_113[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_113[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_113[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_113[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_113[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_113[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_113[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_113[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_113[0]), "=f"(tv_113[1]), "=f"(tv_113[2]), "=f"(tv_113[3]), "=f"(tv_113[4]), "=f"(tv_113[5]), "=f"(tv_113[6]), "=f"(tv_113[7]) : "r"(q_448[1 + 0]));
                        #endif
                    }
                    float2 a_452 = make_float2(d_1[8], d_1[9]);
                    float2 q_452 = make_float2(tv_113[0], tv_113[1]);
                    float2 r_452 = fma_f32x2_rn_noftz(q_452, scale0_23, a_452);
                    d_1[8] = r_452.x;
                    d_1[9] = r_452.y;
                    float2 a_453 = make_float2(d_1[10], d_1[11]);
                    float2 q_453 = make_float2(tv_113[2], tv_113[3]);
                    float2 r_453 = fma_f32x2_rn_noftz(q_453, scale0_23, a_453);
                    d_1[10] = r_453.x;
                    d_1[11] = r_453.y;
                    float2 a_454 = make_float2(d_1[12], d_1[13]);
                    float2 q_454 = make_float2(tv_113[4], tv_113[5]);
                    float2 r_454 = fma_f32x2_rn_noftz(q_454, scale0_23, a_454);
                    d_1[12] = r_454.x;
                    d_1[13] = r_454.y;
                    float2 a_455 = make_float2(d_1[14], d_1[15]);
                    float2 q_455 = make_float2(tv_113[6], tv_113[7]);
                    float2 r_455 = fma_f32x2_rn_noftz(q_455, scale0_23, a_455);
                    d_1[14] = r_455.x;
                    d_1[15] = r_455.y;
                    float tv_114[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_448[2 + 0]));
                        (&tv_114[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_114[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_114[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_114[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_114[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_114[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_114[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_114[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_114[0]), "=f"(tv_114[1]), "=f"(tv_114[2]), "=f"(tv_114[3]), "=f"(tv_114[4]), "=f"(tv_114[5]), "=f"(tv_114[6]), "=f"(tv_114[7]) : "r"(q_448[2 + 0]));
                        #endif
                    }
                    float2 a_456 = make_float2(d_1[16], d_1[17]);
                    float2 q_456 = make_float2(tv_114[0], tv_114[1]);
                    float2 r_456 = fma_f32x2_rn_noftz(q_456, scale1_23, a_456);
                    d_1[16] = r_456.x;
                    d_1[17] = r_456.y;
                    float2 a_457 = make_float2(d_1[18], d_1[19]);
                    float2 q_457 = make_float2(tv_114[2], tv_114[3]);
                    float2 r_457 = fma_f32x2_rn_noftz(q_457, scale1_23, a_457);
                    d_1[18] = r_457.x;
                    d_1[19] = r_457.y;
                    float2 a_458 = make_float2(d_1[20], d_1[21]);
                    float2 q_458 = make_float2(tv_114[4], tv_114[5]);
                    float2 r_458 = fma_f32x2_rn_noftz(q_458, scale1_23, a_458);
                    d_1[20] = r_458.x;
                    d_1[21] = r_458.y;
                    float2 a_459 = make_float2(d_1[22], d_1[23]);
                    float2 q_459 = make_float2(tv_114[6], tv_114[7]);
                    float2 r_459 = fma_f32x2_rn_noftz(q_459, scale1_23, a_459);
                    d_1[22] = r_459.x;
                    d_1[23] = r_459.y;
                    float tv_115[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_448[3 + 0]));
                        (&tv_115[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_115[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_115[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_115[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_115[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_115[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_115[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_115[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_115[0]), "=f"(tv_115[1]), "=f"(tv_115[2]), "=f"(tv_115[3]), "=f"(tv_115[4]), "=f"(tv_115[5]), "=f"(tv_115[6]), "=f"(tv_115[7]) : "r"(q_448[3 + 0]));
                        #endif
                    }
                    float2 a_460 = make_float2(d_1[24], d_1[25]);
                    float2 q_460 = make_float2(tv_115[0], tv_115[1]);
                    float2 r_460 = fma_f32x2_rn_noftz(q_460, scale1_23, a_460);
                    d_1[24] = r_460.x;
                    d_1[25] = r_460.y;
                    float2 a_461 = make_float2(d_1[26], d_1[27]);
                    float2 q_461 = make_float2(tv_115[2], tv_115[3]);
                    float2 r_461 = fma_f32x2_rn_noftz(q_461, scale1_23, a_461);
                    d_1[26] = r_461.x;
                    d_1[27] = r_461.y;
                    float2 a_462 = make_float2(d_1[28], d_1[29]);
                    float2 q_462 = make_float2(tv_115[4], tv_115[5]);
                    float2 r_462 = fma_f32x2_rn_noftz(q_462, scale1_23, a_462);
                    d_1[28] = r_462.x;
                    d_1[29] = r_462.y;
                    float2 a_463 = make_float2(d_1[30], d_1[31]);
                    float2 q_463 = make_float2(tv_115[6], tv_115[7]);
                    float2 r_463 = fma_f32x2_rn_noftz(q_463, scale1_23, a_463);
                    d_1[30] = r_463.x;
                    d_1[31] = r_463.y;
                    unsigned int d_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_1[_lp*2 + 0], d_1[_lp*2+1 + 0]));
                        d_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_bf16_1[0]), "r"(d_bf16_1[1]), "r"(d_bf16_1[2]), "r"(d_bf16_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_bf16_1[4]), "r"(d_bf16_1[5]), "r"(d_bf16_1[6]), "r"(d_bf16_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_bf16_1[8]), "r"(d_bf16_1[9]), "r"(d_bf16_1[10]), "r"(d_bf16_1[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_bf16_1[12]), "r"(d_bf16_1[13]), "r"(d_bf16_1[14]), "r"(d_bf16_1[15]) : "memory");
                    __syncwarp();
                    int c_24 = lane_1;
                    int l_16 = c_24 >> 2;
                    unsigned int w4_30[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_30[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_16 + (c_24 & 3 ^ l_16 >> 1 & 3)))));
                    long long idx_16 = (tix64 * 4 + rank64) * 3072 + (long long)(4 * (128 * j_1 + c_24));
                    {
                        int4 _iv4 = make_int4(w4_30[0 + 0], w4_30[0 + 1], w4_30[0 + 2], w4_30[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_16) = _iv4;
                    }
                    int c_25 = lane_1 + 32;
                    int l_17 = c_25 >> 2;
                    unsigned int w4_31[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_31[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_17 + (c_25 & 3 ^ l_17 >> 1 & 3)))));
                    long long idx_17 = (tix64 * 4 + rank64) * 3072 + (long long)(4 * (128 * j_1 + c_25));
                    {
                        int4 _iv4 = make_int4(w4_31[0 + 0], w4_31[0 + 1], w4_31[0 + 2], w4_31[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_17) = _iv4;
                    }
                    int c_26 = lane_1 + 64;
                    int l_18 = c_26 >> 2;
                    unsigned int w4_32[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_32[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_18 + (c_26 & 3 ^ l_18 >> 1 & 3)))));
                    long long idx_18 = (tix64 * 4 + rank64) * 3072 + (long long)(4 * (128 * j_1 + c_26));
                    {
                        int4 _iv4 = make_int4(w4_32[0 + 0], w4_32[0 + 1], w4_32[0 + 2], w4_32[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_18) = _iv4;
                    }
                    int c_27 = lane_1 + 96;
                    int l_19 = c_27 >> 2;
                    unsigned int w4_33[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_33[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_19 + (c_27 & 3 ^ l_19 >> 1 & 3)))));
                    long long idx_19 = (tix64 * 4 + rank64) * 3072 + (long long)(4 * (128 * j_1 + c_27));
                    {
                        int4 _iv4 = make_int4(w4_33[0 + 0], w4_33[0 + 1], w4_33[0 + 2], w4_33[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_19) = _iv4;
                    }
                    __syncwarp();
                }
                if (lane_1 < 4) {
                    {
                        uint32_t _addr_77 = static_cast<uint32_t>(wsm_addr + (unsigned int)(xb_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(rs2_1) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    int ag_wo_2 = (int)ag_w;
                    int ag_ro_2 = (int)ag_r;
                    int capo_0_2 = (int)capo64;
                    int ix_1_2 = (int)tix64;
                    int rk_2_1 = (int)rank64;
                    int v_off_2 = ag_wo_2 + (rk_2_1 * capo_0_2 + ix_1_2) * 3072;
                    int sc_off_2 = ag_wo_2 + 4 * capo_0_2 * 3072 + (rk_2_1 * capo_0_2 + ix_1_2) * 384;
                    int m_off_2 = ag_ro_2 + (rk_2_1 * capo_0_2 + ix_1_2) * 16;
                    int peer_1_3 = (rank + 1) % 4;
                    uint8_t* pw_1_3 = reinterpret_cast<uint8_t*>(tab[peer_1_3]);
                    uint8_t* pr_1_3 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_3]);
                    {
                        void* _cpbulk_dst_78 = reinterpret_cast<void*>(pw_1_3 + v_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_78), "r"(wsm_addr + (unsigned int)xb_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_79 = reinterpret_cast<void*>(pw_1_3 + sc_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_79), "r"(wsm_addr + (unsigned int)(xb_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_80 = reinterpret_cast<void*>(pr_1_3 + m_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_80), "r"(wsm_addr + (unsigned int)(xb_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_1_4 = (rank + 2) % 4;
                    uint8_t* pw_1_4 = reinterpret_cast<uint8_t*>(tab[peer_1_4]);
                    uint8_t* pr_1_4 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_4]);
                    {
                        void* _cpbulk_dst_81 = reinterpret_cast<void*>(pw_1_4 + v_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_81), "r"(wsm_addr + (unsigned int)xb_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_82 = reinterpret_cast<void*>(pw_1_4 + sc_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_82), "r"(wsm_addr + (unsigned int)(xb_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_83 = reinterpret_cast<void*>(pr_1_4 + m_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_83), "r"(wsm_addr + (unsigned int)(xb_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_1_5 = (rank + 3) % 4;
                    uint8_t* pw_1_5 = reinterpret_cast<uint8_t*>(tab[peer_1_5]);
                    uint8_t* pr_1_5 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_5]);
                    {
                        void* _cpbulk_dst_84 = reinterpret_cast<void*>(pw_1_5 + v_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_84), "r"(wsm_addr + (unsigned int)xb_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_85 = reinterpret_cast<void*>(pw_1_5 + sc_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_85), "r"(wsm_addr + (unsigned int)(xb_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_86 = reinterpret_cast<void*>(pr_1_5 + m_off_2);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_86), "r"(wsm_addr + (unsigned int)(xb_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            #pragma unroll 1
            for (int c = (warp_0 + 9) % 10; c < tq - nbt; c += 10) {
                int tbump = 0;
                if (c >= rank) {
                    tbump = 1;
                }
                int tcrow = 4 * q1 + c + tbump;
                long long tcsrc64 = (long long)(tcrow % 4);
                int ag_wo_3 = (int)ag_w;
                int ag_ro_3 = (int)ag_r;
                int capo_3 = (int)capo64;
                int src_1 = (int)tcsrc64;
                int ix_3 = (int)tix64;
                int m_off_3 = ag_ro_3 + (src_1 * capo_3 + ix_3) * 16;
                float rs_2 = 0.0f;
                uint32_t polled_1[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(polled_1[0]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + m_off_3)) : "memory");
                } while (
                    (polled_1[0] == 0x80000000u)
                );
                rs_2 = __uint_as_float(polled_1[0]);
                int v_5 = ag_wo_3 + (src_1 * capo_3 + ix_3) * 3072;
                int sc_5 = ag_wo_3 + 4 * capo_3 * 3072 + (src_1 * capo_3 + ix_3) * 384;
                #pragma unroll 1
                for (int jj_3 = 0; jj_3 < 2; jj_3++) {
                    int pp0_3 = lane_1 + 96 * jj_3;
                    int pp1_3 = pp0_3 + 32;
                    int pp2_1 = pp0_3 + 64;
                    uint32_t fp4_wire_values_3[12];
                    uint32_t fp4_wire_scales_3[3];
                    uint16_t _wire_h16_87_0;
                    uint16_t _wire_h16_87_1;
                    uint16_t _wire_h16_87_2;
                    uint32_t _wire_vpend_87 = 7u;
                    uint32_t _wire_spend_87 = 7u;
                    do {
                        if (_wire_vpend_87 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_3[0]), "=r"(fp4_wire_values_3[1]), "=r"(fp4_wire_values_3[2]), "=r"(fp4_wire_values_3[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp0_3)))) : "memory");
                        }
                        if (_wire_vpend_87 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_3[4]), "=r"(fp4_wire_values_3[5]), "=r"(fp4_wire_values_3[6]), "=r"(fp4_wire_values_3[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp1_3)))) : "memory");
                        }
                        if (_wire_vpend_87 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_3[8]), "=r"(fp4_wire_values_3[9]), "=r"(fp4_wire_values_3[10]), "=r"(fp4_wire_values_3[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp2_1)))) : "memory");
                        }
                        if (_wire_spend_87 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_87_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_5 + (int)(2 * pp0_3)))) : "memory");
                            fp4_wire_scales_3[0] = (uint32_t)_wire_h16_87_0;
                        }
                        if (_wire_spend_87 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_87_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_5 + (int)(2 * pp1_3)))) : "memory");
                            fp4_wire_scales_3[1] = (uint32_t)_wire_h16_87_1;
                        }
                        if (_wire_spend_87 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_87_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_5 + (int)(2 * pp2_1)))) : "memory");
                            fp4_wire_scales_3[2] = (uint32_t)_wire_h16_87_2;
                        }
                        _wire_vpend_87 = ((uint32_t)((((fp4_wire_values_3[0] & ~((fp4_wire_values_3[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[1] & ~((fp4_wire_values_3[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[2] & ~((fp4_wire_values_3[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[3] & ~((fp4_wire_values_3[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((fp4_wire_values_3[4] & ~((fp4_wire_values_3[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[5] & ~((fp4_wire_values_3[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[6] & ~((fp4_wire_values_3[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[7] & ~((fp4_wire_values_3[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((fp4_wire_values_3[8] & ~((fp4_wire_values_3[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[9] & ~((fp4_wire_values_3[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[10] & ~((fp4_wire_values_3[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_3[11] & ~((fp4_wire_values_3[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2);
                        _wire_spend_87 = ((uint32_t)(((uint32_t)((fp4_wire_scales_3[0] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_3[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((fp4_wire_scales_3[1] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_3[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((fp4_wire_scales_3[2] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_3[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2);
                    } while ((_wire_vpend_87 | _wire_spend_87) != 0u);
                    float acc_6[32] = {0};
                    uint32_t e4m3x2_to_f16x2_26;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_26) : "h"((uint16_t)(fp4_wire_scales_3[0])));
                    float cvt_f32_f16_27;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_27) : "h"((uint16_t)(e4m3x2_to_f16x2_26)));
                    float cvt_f32_f16_1_26;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_26) : "h"((uint16_t)(e4m3x2_to_f16x2_26 >> 16)));
                    float s0_24;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_24) : "f"(cvt_f32_f16_27), "f"(rs_2));
                    float s1_24;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_24) : "f"(cvt_f32_f16_1_26), "f"(rs_2));
                    float2 scale0_24 = make_float2(s0_24, s0_24);
                    float2 scale1_24 = make_float2(s1_24, s1_24);
                    float tv_116[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[0]));
                        (&tv_116[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_116[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_116[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_116[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_116[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_116[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_116[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_116[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_116[0]), "=f"(tv_116[1]), "=f"(tv_116[2]), "=f"(tv_116[3]), "=f"(tv_116[4]), "=f"(tv_116[5]), "=f"(tv_116[6]), "=f"(tv_116[7]) : "r"(fp4_wire_values_3[0]));
                        #endif
                    }
                    float2 a_464 = make_float2(acc_6[0], acc_6[1]);
                    float2 q_464 = make_float2(tv_116[0], tv_116[1]);
                    float2 r_464 = fma_f32x2_rn_noftz(q_464, scale0_24, a_464);
                    acc_6[0] = r_464.x;
                    acc_6[1] = r_464.y;
                    float2 a_465 = make_float2(acc_6[2], acc_6[3]);
                    float2 q_465 = make_float2(tv_116[2], tv_116[3]);
                    float2 r_465 = fma_f32x2_rn_noftz(q_465, scale0_24, a_465);
                    acc_6[2] = r_465.x;
                    acc_6[3] = r_465.y;
                    float2 a_466 = make_float2(acc_6[4], acc_6[5]);
                    float2 q_466 = make_float2(tv_116[4], tv_116[5]);
                    float2 r_466 = fma_f32x2_rn_noftz(q_466, scale0_24, a_466);
                    acc_6[4] = r_466.x;
                    acc_6[5] = r_466.y;
                    float2 a_467 = make_float2(acc_6[6], acc_6[7]);
                    float2 q_467 = make_float2(tv_116[6], tv_116[7]);
                    float2 r_467 = fma_f32x2_rn_noftz(q_467, scale0_24, a_467);
                    acc_6[6] = r_467.x;
                    acc_6[7] = r_467.y;
                    float tv_117[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[1 + 0]));
                        (&tv_117[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_117[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_117[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_117[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_117[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_117[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_117[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_117[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_117[0]), "=f"(tv_117[1]), "=f"(tv_117[2]), "=f"(tv_117[3]), "=f"(tv_117[4]), "=f"(tv_117[5]), "=f"(tv_117[6]), "=f"(tv_117[7]) : "r"(fp4_wire_values_3[1 + 0]));
                        #endif
                    }
                    float2 a_468 = make_float2(acc_6[8], acc_6[9]);
                    float2 q_468 = make_float2(tv_117[0], tv_117[1]);
                    float2 r_468 = fma_f32x2_rn_noftz(q_468, scale0_24, a_468);
                    acc_6[8] = r_468.x;
                    acc_6[9] = r_468.y;
                    float2 a_469 = make_float2(acc_6[10], acc_6[11]);
                    float2 q_469 = make_float2(tv_117[2], tv_117[3]);
                    float2 r_469 = fma_f32x2_rn_noftz(q_469, scale0_24, a_469);
                    acc_6[10] = r_469.x;
                    acc_6[11] = r_469.y;
                    float2 a_470 = make_float2(acc_6[12], acc_6[13]);
                    float2 q_470 = make_float2(tv_117[4], tv_117[5]);
                    float2 r_470 = fma_f32x2_rn_noftz(q_470, scale0_24, a_470);
                    acc_6[12] = r_470.x;
                    acc_6[13] = r_470.y;
                    float2 a_471 = make_float2(acc_6[14], acc_6[15]);
                    float2 q_471 = make_float2(tv_117[6], tv_117[7]);
                    float2 r_471 = fma_f32x2_rn_noftz(q_471, scale0_24, a_471);
                    acc_6[14] = r_471.x;
                    acc_6[15] = r_471.y;
                    float tv_118[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[2 + 0]));
                        (&tv_118[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_118[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_118[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_118[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_118[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_118[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_118[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_118[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_118[0]), "=f"(tv_118[1]), "=f"(tv_118[2]), "=f"(tv_118[3]), "=f"(tv_118[4]), "=f"(tv_118[5]), "=f"(tv_118[6]), "=f"(tv_118[7]) : "r"(fp4_wire_values_3[2 + 0]));
                        #endif
                    }
                    float2 a_472 = make_float2(acc_6[16], acc_6[17]);
                    float2 q_472 = make_float2(tv_118[0], tv_118[1]);
                    float2 r_472 = fma_f32x2_rn_noftz(q_472, scale1_24, a_472);
                    acc_6[16] = r_472.x;
                    acc_6[17] = r_472.y;
                    float2 a_473 = make_float2(acc_6[18], acc_6[19]);
                    float2 q_473 = make_float2(tv_118[2], tv_118[3]);
                    float2 r_473 = fma_f32x2_rn_noftz(q_473, scale1_24, a_473);
                    acc_6[18] = r_473.x;
                    acc_6[19] = r_473.y;
                    float2 a_474 = make_float2(acc_6[20], acc_6[21]);
                    float2 q_474 = make_float2(tv_118[4], tv_118[5]);
                    float2 r_474 = fma_f32x2_rn_noftz(q_474, scale1_24, a_474);
                    acc_6[20] = r_474.x;
                    acc_6[21] = r_474.y;
                    float2 a_475 = make_float2(acc_6[22], acc_6[23]);
                    float2 q_475 = make_float2(tv_118[6], tv_118[7]);
                    float2 r_475 = fma_f32x2_rn_noftz(q_475, scale1_24, a_475);
                    acc_6[22] = r_475.x;
                    acc_6[23] = r_475.y;
                    float tv_119[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[3 + 0]));
                        (&tv_119[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_119[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_119[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_119[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_119[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_119[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_119[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_119[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_119[0]), "=f"(tv_119[1]), "=f"(tv_119[2]), "=f"(tv_119[3]), "=f"(tv_119[4]), "=f"(tv_119[5]), "=f"(tv_119[6]), "=f"(tv_119[7]) : "r"(fp4_wire_values_3[3 + 0]));
                        #endif
                    }
                    float2 a_476 = make_float2(acc_6[24], acc_6[25]);
                    float2 q_476 = make_float2(tv_119[0], tv_119[1]);
                    float2 r_476 = fma_f32x2_rn_noftz(q_476, scale1_24, a_476);
                    acc_6[24] = r_476.x;
                    acc_6[25] = r_476.y;
                    float2 a_477 = make_float2(acc_6[26], acc_6[27]);
                    float2 q_477 = make_float2(tv_119[2], tv_119[3]);
                    float2 r_477 = fma_f32x2_rn_noftz(q_477, scale1_24, a_477);
                    acc_6[26] = r_477.x;
                    acc_6[27] = r_477.y;
                    float2 a_478 = make_float2(acc_6[28], acc_6[29]);
                    float2 q_478 = make_float2(tv_119[4], tv_119[5]);
                    float2 r_478 = fma_f32x2_rn_noftz(q_478, scale1_24, a_478);
                    acc_6[28] = r_478.x;
                    acc_6[29] = r_478.y;
                    float2 a_479 = make_float2(acc_6[30], acc_6[31]);
                    float2 q_479 = make_float2(tv_119[6], tv_119[7]);
                    float2 r_479 = fma_f32x2_rn_noftz(q_479, scale1_24, a_479);
                    acc_6[30] = r_479.x;
                    acc_6[31] = r_479.y;
                    unsigned int acc_bf16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_6[_lp*2 + 0], acc_6[_lp*2+1 + 0]));
                        acc_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16_2[0]), "r"(acc_bf16_2[1]), "r"(acc_bf16_2[2]), "r"(acc_bf16_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16_2[4]), "r"(acc_bf16_2[5]), "r"(acc_bf16_2[6]), "r"(acc_bf16_2[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16_2[8]), "r"(acc_bf16_2[9]), "r"(acc_bf16_2[10]), "r"(acc_bf16_2[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16_2[12]), "r"(acc_bf16_2[13]), "r"(acc_bf16_2[14]), "r"(acc_bf16_2[15]) : "memory");
                    __syncwarp();
                    int c_28 = lane_1;
                    int l_20 = c_28 >> 2;
                    unsigned int w4_34[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_34[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_20 + (c_28 & 3 ^ l_20 >> 1 & 3)))));
                    long long idx_20 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3) + c_28));
                    {
                        int4 _iv4 = make_int4(w4_34[0 + 0], w4_34[0 + 1], w4_34[0 + 2], w4_34[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_20) = _iv4;
                    }
                    int c_29 = lane_1 + 32;
                    int l_21 = c_29 >> 2;
                    unsigned int w4_35[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_35[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_21 + (c_29 & 3 ^ l_21 >> 1 & 3)))));
                    long long idx_21 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3) + c_29));
                    {
                        int4 _iv4 = make_int4(w4_35[0 + 0], w4_35[0 + 1], w4_35[0 + 2], w4_35[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_21) = _iv4;
                    }
                    int c_30 = lane_1 + 64;
                    int l_22 = c_30 >> 2;
                    unsigned int w4_36[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_36[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_22 + (c_30 & 3 ^ l_22 >> 1 & 3)))));
                    long long idx_22 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3) + c_30));
                    {
                        int4 _iv4 = make_int4(w4_36[0 + 0], w4_36[0 + 1], w4_36[0 + 2], w4_36[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_22) = _iv4;
                    }
                    int c_31 = lane_1 + 96;
                    int l_23 = c_31 >> 2;
                    unsigned int w4_37[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_37[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_23 + (c_31 & 3 ^ l_23 >> 1 & 3)))));
                    long long idx_23 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3) + c_31));
                    {
                        int4 _iv4 = make_int4(w4_37[0 + 0], w4_37[0 + 1], w4_37[0 + 2], w4_37[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_23) = _iv4;
                    }
                    __syncwarp();
                    float acc_6_1[32] = {0};
                    uint32_t e4m3x2_to_f16x2_26_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_26_1) : "h"((uint16_t)(fp4_wire_scales_3[1])));
                    float cvt_f32_f16_27_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_27_1) : "h"((uint16_t)(e4m3x2_to_f16x2_26_1)));
                    float cvt_f32_f16_1_26_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_26_1) : "h"((uint16_t)(e4m3x2_to_f16x2_26_1 >> 16)));
                    float s0_24_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_24_1) : "f"(cvt_f32_f16_27_1), "f"(rs_2));
                    float s1_24_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_24_1) : "f"(cvt_f32_f16_1_26_1), "f"(rs_2));
                    float2 scale0_24_1 = make_float2(s0_24_1, s0_24_1);
                    float2 scale1_24_1 = make_float2(s1_24_1, s1_24_1);
                    float tv_120[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[4 + 0]));
                        (&tv_120[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_120[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_120[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_120[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_120[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_120[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_120[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_120[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_120[0]), "=f"(tv_120[1]), "=f"(tv_120[2]), "=f"(tv_120[3]), "=f"(tv_120[4]), "=f"(tv_120[5]), "=f"(tv_120[6]), "=f"(tv_120[7]) : "r"(fp4_wire_values_3[4 + 0]));
                        #endif
                    }
                    float2 a_480 = make_float2(acc_6_1[0], acc_6_1[1]);
                    float2 q_480 = make_float2(tv_120[0], tv_120[1]);
                    float2 r_480 = fma_f32x2_rn_noftz(q_480, scale0_24_1, a_480);
                    acc_6_1[0] = r_480.x;
                    acc_6_1[1] = r_480.y;
                    float2 a_481 = make_float2(acc_6_1[2], acc_6_1[3]);
                    float2 q_481 = make_float2(tv_120[2], tv_120[3]);
                    float2 r_481 = fma_f32x2_rn_noftz(q_481, scale0_24_1, a_481);
                    acc_6_1[2] = r_481.x;
                    acc_6_1[3] = r_481.y;
                    float2 a_482 = make_float2(acc_6_1[4], acc_6_1[5]);
                    float2 q_482 = make_float2(tv_120[4], tv_120[5]);
                    float2 r_482 = fma_f32x2_rn_noftz(q_482, scale0_24_1, a_482);
                    acc_6_1[4] = r_482.x;
                    acc_6_1[5] = r_482.y;
                    float2 a_483 = make_float2(acc_6_1[6], acc_6_1[7]);
                    float2 q_483 = make_float2(tv_120[6], tv_120[7]);
                    float2 r_483 = fma_f32x2_rn_noftz(q_483, scale0_24_1, a_483);
                    acc_6_1[6] = r_483.x;
                    acc_6_1[7] = r_483.y;
                    float tv_121[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[5 + 0]));
                        (&tv_121[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_121[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_121[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_121[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_121[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_121[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_121[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_121[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_121[0]), "=f"(tv_121[1]), "=f"(tv_121[2]), "=f"(tv_121[3]), "=f"(tv_121[4]), "=f"(tv_121[5]), "=f"(tv_121[6]), "=f"(tv_121[7]) : "r"(fp4_wire_values_3[5 + 0]));
                        #endif
                    }
                    float2 a_484 = make_float2(acc_6_1[8], acc_6_1[9]);
                    float2 q_484 = make_float2(tv_121[0], tv_121[1]);
                    float2 r_484 = fma_f32x2_rn_noftz(q_484, scale0_24_1, a_484);
                    acc_6_1[8] = r_484.x;
                    acc_6_1[9] = r_484.y;
                    float2 a_485 = make_float2(acc_6_1[10], acc_6_1[11]);
                    float2 q_485 = make_float2(tv_121[2], tv_121[3]);
                    float2 r_485 = fma_f32x2_rn_noftz(q_485, scale0_24_1, a_485);
                    acc_6_1[10] = r_485.x;
                    acc_6_1[11] = r_485.y;
                    float2 a_486 = make_float2(acc_6_1[12], acc_6_1[13]);
                    float2 q_486 = make_float2(tv_121[4], tv_121[5]);
                    float2 r_486 = fma_f32x2_rn_noftz(q_486, scale0_24_1, a_486);
                    acc_6_1[12] = r_486.x;
                    acc_6_1[13] = r_486.y;
                    float2 a_487 = make_float2(acc_6_1[14], acc_6_1[15]);
                    float2 q_487 = make_float2(tv_121[6], tv_121[7]);
                    float2 r_487 = fma_f32x2_rn_noftz(q_487, scale0_24_1, a_487);
                    acc_6_1[14] = r_487.x;
                    acc_6_1[15] = r_487.y;
                    float tv_122[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[6 + 0]));
                        (&tv_122[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_122[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_122[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_122[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_122[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_122[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_122[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_122[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_122[0]), "=f"(tv_122[1]), "=f"(tv_122[2]), "=f"(tv_122[3]), "=f"(tv_122[4]), "=f"(tv_122[5]), "=f"(tv_122[6]), "=f"(tv_122[7]) : "r"(fp4_wire_values_3[6 + 0]));
                        #endif
                    }
                    float2 a_488 = make_float2(acc_6_1[16], acc_6_1[17]);
                    float2 q_488 = make_float2(tv_122[0], tv_122[1]);
                    float2 r_488 = fma_f32x2_rn_noftz(q_488, scale1_24_1, a_488);
                    acc_6_1[16] = r_488.x;
                    acc_6_1[17] = r_488.y;
                    float2 a_489 = make_float2(acc_6_1[18], acc_6_1[19]);
                    float2 q_489 = make_float2(tv_122[2], tv_122[3]);
                    float2 r_489 = fma_f32x2_rn_noftz(q_489, scale1_24_1, a_489);
                    acc_6_1[18] = r_489.x;
                    acc_6_1[19] = r_489.y;
                    float2 a_490 = make_float2(acc_6_1[20], acc_6_1[21]);
                    float2 q_490 = make_float2(tv_122[4], tv_122[5]);
                    float2 r_490 = fma_f32x2_rn_noftz(q_490, scale1_24_1, a_490);
                    acc_6_1[20] = r_490.x;
                    acc_6_1[21] = r_490.y;
                    float2 a_491 = make_float2(acc_6_1[22], acc_6_1[23]);
                    float2 q_491 = make_float2(tv_122[6], tv_122[7]);
                    float2 r_491 = fma_f32x2_rn_noftz(q_491, scale1_24_1, a_491);
                    acc_6_1[22] = r_491.x;
                    acc_6_1[23] = r_491.y;
                    float tv_123[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[7 + 0]));
                        (&tv_123[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_123[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_123[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_123[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_123[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_123[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_123[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_123[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_123[0]), "=f"(tv_123[1]), "=f"(tv_123[2]), "=f"(tv_123[3]), "=f"(tv_123[4]), "=f"(tv_123[5]), "=f"(tv_123[6]), "=f"(tv_123[7]) : "r"(fp4_wire_values_3[7 + 0]));
                        #endif
                    }
                    float2 a_492 = make_float2(acc_6_1[24], acc_6_1[25]);
                    float2 q_492 = make_float2(tv_123[0], tv_123[1]);
                    float2 r_492 = fma_f32x2_rn_noftz(q_492, scale1_24_1, a_492);
                    acc_6_1[24] = r_492.x;
                    acc_6_1[25] = r_492.y;
                    float2 a_493 = make_float2(acc_6_1[26], acc_6_1[27]);
                    float2 q_493 = make_float2(tv_123[2], tv_123[3]);
                    float2 r_493 = fma_f32x2_rn_noftz(q_493, scale1_24_1, a_493);
                    acc_6_1[26] = r_493.x;
                    acc_6_1[27] = r_493.y;
                    float2 a_494 = make_float2(acc_6_1[28], acc_6_1[29]);
                    float2 q_494 = make_float2(tv_123[4], tv_123[5]);
                    float2 r_494 = fma_f32x2_rn_noftz(q_494, scale1_24_1, a_494);
                    acc_6_1[28] = r_494.x;
                    acc_6_1[29] = r_494.y;
                    float2 a_495 = make_float2(acc_6_1[30], acc_6_1[31]);
                    float2 q_495 = make_float2(tv_123[6], tv_123[7]);
                    float2 r_495 = fma_f32x2_rn_noftz(q_495, scale1_24_1, a_495);
                    acc_6_1[30] = r_495.x;
                    acc_6_1[31] = r_495.y;
                    unsigned int acc_bf16_2_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_6_1[_lp*2 + 0], acc_6_1[_lp*2+1 + 0]));
                        acc_bf16_2_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16_2_1[0]), "r"(acc_bf16_2_1[1]), "r"(acc_bf16_2_1[2]), "r"(acc_bf16_2_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16_2_1[4]), "r"(acc_bf16_2_1[5]), "r"(acc_bf16_2_1[6]), "r"(acc_bf16_2_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16_2_1[8]), "r"(acc_bf16_2_1[9]), "r"(acc_bf16_2_1[10]), "r"(acc_bf16_2_1[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16_2_1[12]), "r"(acc_bf16_2_1[13]), "r"(acc_bf16_2_1[14]), "r"(acc_bf16_2_1[15]) : "memory");
                    __syncwarp();
                    int c_32 = lane_1;
                    int l_24 = c_32 >> 2;
                    unsigned int w4_38[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_38[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_38[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_38[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_24 + (c_32 & 3 ^ l_24 >> 1 & 3)))));
                    long long idx_24 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3 + 1) + c_32));
                    {
                        int4 _iv4 = make_int4(w4_38[0 + 0], w4_38[0 + 1], w4_38[0 + 2], w4_38[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_24) = _iv4;
                    }
                    int c_33 = lane_1 + 32;
                    int l_25 = c_33 >> 2;
                    unsigned int w4_39[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_39[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_25 + (c_33 & 3 ^ l_25 >> 1 & 3)))));
                    long long idx_25 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3 + 1) + c_33));
                    {
                        int4 _iv4 = make_int4(w4_39[0 + 0], w4_39[0 + 1], w4_39[0 + 2], w4_39[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_25) = _iv4;
                    }
                    int c_34 = lane_1 + 64;
                    int l_26 = c_34 >> 2;
                    unsigned int w4_40[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_40[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_40[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_26 + (c_34 & 3 ^ l_26 >> 1 & 3)))));
                    long long idx_26 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3 + 1) + c_34));
                    {
                        int4 _iv4 = make_int4(w4_40[0 + 0], w4_40[0 + 1], w4_40[0 + 2], w4_40[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_26) = _iv4;
                    }
                    int c_35 = lane_1 + 96;
                    int l_27 = c_35 >> 2;
                    unsigned int w4_41[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_41[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_41[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_41[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_41[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_27 + (c_35 & 3 ^ l_27 >> 1 & 3)))));
                    long long idx_27 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3 + 1) + c_35));
                    {
                        int4 _iv4 = make_int4(w4_41[0 + 0], w4_41[0 + 1], w4_41[0 + 2], w4_41[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_27) = _iv4;
                    }
                    __syncwarp();
                    float acc_7[32] = {0};
                    uint32_t e4m3x2_to_f16x2_27;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_27) : "h"((uint16_t)(fp4_wire_scales_3[2])));
                    float cvt_f32_f16_28;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_28) : "h"((uint16_t)(e4m3x2_to_f16x2_27)));
                    float cvt_f32_f16_1_27;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_27) : "h"((uint16_t)(e4m3x2_to_f16x2_27 >> 16)));
                    float s0_25;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_25) : "f"(cvt_f32_f16_28), "f"(rs_2));
                    float s1_25;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_25) : "f"(cvt_f32_f16_1_27), "f"(rs_2));
                    float2 scale0_25 = make_float2(s0_25, s0_25);
                    float2 scale1_25 = make_float2(s1_25, s1_25);
                    float tv_124[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[8 + 0]));
                        (&tv_124[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_124[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_124[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_124[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_124[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_124[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_124[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_124[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_124[0]), "=f"(tv_124[1]), "=f"(tv_124[2]), "=f"(tv_124[3]), "=f"(tv_124[4]), "=f"(tv_124[5]), "=f"(tv_124[6]), "=f"(tv_124[7]) : "r"(fp4_wire_values_3[8 + 0]));
                        #endif
                    }
                    float2 a_496 = make_float2(acc_7[0], acc_7[1]);
                    float2 q_496 = make_float2(tv_124[0], tv_124[1]);
                    float2 r_496 = fma_f32x2_rn_noftz(q_496, scale0_25, a_496);
                    acc_7[0] = r_496.x;
                    acc_7[1] = r_496.y;
                    float2 a_497 = make_float2(acc_7[2], acc_7[3]);
                    float2 q_497 = make_float2(tv_124[2], tv_124[3]);
                    float2 r_497 = fma_f32x2_rn_noftz(q_497, scale0_25, a_497);
                    acc_7[2] = r_497.x;
                    acc_7[3] = r_497.y;
                    float2 a_498 = make_float2(acc_7[4], acc_7[5]);
                    float2 q_498 = make_float2(tv_124[4], tv_124[5]);
                    float2 r_498 = fma_f32x2_rn_noftz(q_498, scale0_25, a_498);
                    acc_7[4] = r_498.x;
                    acc_7[5] = r_498.y;
                    float2 a_499 = make_float2(acc_7[6], acc_7[7]);
                    float2 q_499 = make_float2(tv_124[6], tv_124[7]);
                    float2 r_499 = fma_f32x2_rn_noftz(q_499, scale0_25, a_499);
                    acc_7[6] = r_499.x;
                    acc_7[7] = r_499.y;
                    float tv_125[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[9 + 0]));
                        (&tv_125[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_125[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_125[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_125[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_125[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_125[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_125[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_125[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_125[0]), "=f"(tv_125[1]), "=f"(tv_125[2]), "=f"(tv_125[3]), "=f"(tv_125[4]), "=f"(tv_125[5]), "=f"(tv_125[6]), "=f"(tv_125[7]) : "r"(fp4_wire_values_3[9 + 0]));
                        #endif
                    }
                    float2 a_500 = make_float2(acc_7[8], acc_7[9]);
                    float2 q_500 = make_float2(tv_125[0], tv_125[1]);
                    float2 r_500 = fma_f32x2_rn_noftz(q_500, scale0_25, a_500);
                    acc_7[8] = r_500.x;
                    acc_7[9] = r_500.y;
                    float2 a_501 = make_float2(acc_7[10], acc_7[11]);
                    float2 q_501 = make_float2(tv_125[2], tv_125[3]);
                    float2 r_501 = fma_f32x2_rn_noftz(q_501, scale0_25, a_501);
                    acc_7[10] = r_501.x;
                    acc_7[11] = r_501.y;
                    float2 a_502 = make_float2(acc_7[12], acc_7[13]);
                    float2 q_502 = make_float2(tv_125[4], tv_125[5]);
                    float2 r_502 = fma_f32x2_rn_noftz(q_502, scale0_25, a_502);
                    acc_7[12] = r_502.x;
                    acc_7[13] = r_502.y;
                    float2 a_503 = make_float2(acc_7[14], acc_7[15]);
                    float2 q_503 = make_float2(tv_125[6], tv_125[7]);
                    float2 r_503 = fma_f32x2_rn_noftz(q_503, scale0_25, a_503);
                    acc_7[14] = r_503.x;
                    acc_7[15] = r_503.y;
                    float tv_126[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[10 + 0]));
                        (&tv_126[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_126[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_126[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_126[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_126[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_126[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_126[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_126[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_126[0]), "=f"(tv_126[1]), "=f"(tv_126[2]), "=f"(tv_126[3]), "=f"(tv_126[4]), "=f"(tv_126[5]), "=f"(tv_126[6]), "=f"(tv_126[7]) : "r"(fp4_wire_values_3[10 + 0]));
                        #endif
                    }
                    float2 a_504 = make_float2(acc_7[16], acc_7[17]);
                    float2 q_504 = make_float2(tv_126[0], tv_126[1]);
                    float2 r_504 = fma_f32x2_rn_noftz(q_504, scale1_25, a_504);
                    acc_7[16] = r_504.x;
                    acc_7[17] = r_504.y;
                    float2 a_505 = make_float2(acc_7[18], acc_7[19]);
                    float2 q_505 = make_float2(tv_126[2], tv_126[3]);
                    float2 r_505 = fma_f32x2_rn_noftz(q_505, scale1_25, a_505);
                    acc_7[18] = r_505.x;
                    acc_7[19] = r_505.y;
                    float2 a_506 = make_float2(acc_7[20], acc_7[21]);
                    float2 q_506 = make_float2(tv_126[4], tv_126[5]);
                    float2 r_506 = fma_f32x2_rn_noftz(q_506, scale1_25, a_506);
                    acc_7[20] = r_506.x;
                    acc_7[21] = r_506.y;
                    float2 a_507 = make_float2(acc_7[22], acc_7[23]);
                    float2 q_507 = make_float2(tv_126[6], tv_126[7]);
                    float2 r_507 = fma_f32x2_rn_noftz(q_507, scale1_25, a_507);
                    acc_7[22] = r_507.x;
                    acc_7[23] = r_507.y;
                    float tv_127[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_3[11 + 0]));
                        (&tv_127[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_127[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_127[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_127[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_127[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_127[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_127[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_127[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_127[0]), "=f"(tv_127[1]), "=f"(tv_127[2]), "=f"(tv_127[3]), "=f"(tv_127[4]), "=f"(tv_127[5]), "=f"(tv_127[6]), "=f"(tv_127[7]) : "r"(fp4_wire_values_3[11 + 0]));
                        #endif
                    }
                    float2 a_508 = make_float2(acc_7[24], acc_7[25]);
                    float2 q_508 = make_float2(tv_127[0], tv_127[1]);
                    float2 r_508 = fma_f32x2_rn_noftz(q_508, scale1_25, a_508);
                    acc_7[24] = r_508.x;
                    acc_7[25] = r_508.y;
                    float2 a_509 = make_float2(acc_7[26], acc_7[27]);
                    float2 q_509 = make_float2(tv_127[2], tv_127[3]);
                    float2 r_509 = fma_f32x2_rn_noftz(q_509, scale1_25, a_509);
                    acc_7[26] = r_509.x;
                    acc_7[27] = r_509.y;
                    float2 a_510 = make_float2(acc_7[28], acc_7[29]);
                    float2 q_510 = make_float2(tv_127[4], tv_127[5]);
                    float2 r_510 = fma_f32x2_rn_noftz(q_510, scale1_25, a_510);
                    acc_7[28] = r_510.x;
                    acc_7[29] = r_510.y;
                    float2 a_511 = make_float2(acc_7[30], acc_7[31]);
                    float2 q_511 = make_float2(tv_127[6], tv_127[7]);
                    float2 r_511 = fma_f32x2_rn_noftz(q_511, scale1_25, a_511);
                    acc_7[30] = r_511.x;
                    acc_7[31] = r_511.y;
                    unsigned int acc_bf16_3[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_7[_lp*2 + 0], acc_7[_lp*2+1 + 0]));
                        acc_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16_3[4]), "r"(acc_bf16_3[5]), "r"(acc_bf16_3[6]), "r"(acc_bf16_3[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16_3[8]), "r"(acc_bf16_3[9]), "r"(acc_bf16_3[10]), "r"(acc_bf16_3[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16_3[12]), "r"(acc_bf16_3[13]), "r"(acc_bf16_3[14]), "r"(acc_bf16_3[15]) : "memory");
                    __syncwarp();
                    int c_36 = lane_1;
                    int l_28 = c_36 >> 2;
                    unsigned int w4_42[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_42[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_42[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_42[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_42[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_28 + (c_36 & 3 ^ l_28 >> 1 & 3)))));
                    long long idx_28 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3 + 2) + c_36));
                    {
                        int4 _iv4 = make_int4(w4_42[0 + 0], w4_42[0 + 1], w4_42[0 + 2], w4_42[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_28) = _iv4;
                    }
                    int c_37 = lane_1 + 32;
                    int l_29 = c_37 >> 2;
                    unsigned int w4_43[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_43[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_43[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_43[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_43[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_29 + (c_37 & 3 ^ l_29 >> 1 & 3)))));
                    long long idx_29 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3 + 2) + c_37));
                    {
                        int4 _iv4 = make_int4(w4_43[0 + 0], w4_43[0 + 1], w4_43[0 + 2], w4_43[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_29) = _iv4;
                    }
                    int c_38 = lane_1 + 64;
                    int l_30 = c_38 >> 2;
                    unsigned int w4_44[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_44[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_44[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_30 + (c_38 & 3 ^ l_30 >> 1 & 3)))));
                    long long idx_30 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3 + 2) + c_38));
                    {
                        int4 _iv4 = make_int4(w4_44[0 + 0], w4_44[0 + 1], w4_44[0 + 2], w4_44[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_30) = _iv4;
                    }
                    int c_39 = lane_1 + 96;
                    int l_31 = c_39 >> 2;
                    unsigned int w4_45[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_45[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_45[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_45[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_31 + (c_39 & 3 ^ l_31 >> 1 & 3)))));
                    long long idx_31 = (long long)tcrow * 3072 + (long long)(4 * (128 * (3 * jj_3 + 2) + c_39));
                    {
                        int4 _iv4 = make_int4(w4_45[0 + 0], w4_45[0 + 1], w4_45[0 + 2], w4_45[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_31) = _iv4;
                    }
                    __syncwarp();
                }
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tid == 0) {
                int base_1 = bid * 4;
                ctrl[base_1 + 2] = (unsigned int)rows;
                ctrl[base_1 + 3] = 2;
                ctrl[base_1] = (call + 1) % 3;
            }
        }
    }
    // ---- Role: reset_warp ----
    if (warp == 10) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            unsigned int sv = 2290649224;
            unsigned int ss = 4294967295;
            unsigned int sr = 2147483648;
            #pragma unroll 1
            for (int i_1 = lane_0; i_1 < 1024; i_1 += 32) {
                pat[i_1] = sv;
                pat[1024 + i_1] = ss;
                pat[2048 + i_1] = sr;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call_1 = ctl[0];
            unsigned int prev_rows = ctl[1];
            unsigned int prev_algo = ctl[2];
            unsigned int pbuf = (call_1 + 2) % 3;
            if (prev_algo != 0) {
                long long bid64_1 = (long long)bid;
                long long g64_1 = (long long)gridDim.x;
                long long lane64 = (long long)lane_0;
                long long pr64 = (long long)prev_rows;
                long long pro64 = (pr64 + 3) / 4;
                long long pb64 = (long long)pbuf;
                long long capo64_1 = (long long)((cap + 4 - 1) / 4);
                long long sa_w_1 = 4 * capo64_1 * 3456;
                long long sa_r_1 = 4 * capo64_1 * 16;
                long long rs_w_1 = pb64 * (2 * sa_w_1);
                long long rs_r_1 = pb64 * (2 * sa_r_1);
                long long v_base_1_0 = rs_w_1;
                long long s_base_1_0 = rs_w_1 + 4 * capo64_1 * 3072;
                long long m_base_1_0 = rs_r_1;
                unsigned int sv4[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long u = pro64 * 3072 / 16;
                long long lo = u * bid64_1 / g64_1;
                long long hi = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo + lane64;
                #pragma unroll 1
                for (long long g = g0; g < hi; g += 32) {
                    long long widx = v_base_1_0 + 16 * g >> 2;
                    {
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx) = _iv4;
                    }
                }
                long long u_0 = pro64 * 384 / 16;
                long long lo_0 = u_0 * bid64_1 / g64_1;
                long long hi_0 = u_0 * (bid64_1 + 1) / g64_1;
                long long g0_0 = lo_0 + lane64;
                #pragma unroll 1
                for (long long g_32 = g0_0; g_32 < hi_0; g_32 += 32) {
                    long long widx_1 = s_base_1_0 + 16 * g_32 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_1) = _iv4;
                    }
                }
                long long u_1 = pro64 * 16 / 16;
                long long lo_1 = u_1 * bid64_1 / g64_1;
                long long hi_1 = u_1 * (bid64_1 + 1) / g64_1;
                long long g0_1 = lo_1 + lane64;
                #pragma unroll 1
                for (long long g_33 = g0_1; g_33 < hi_1; g_33 += 32) {
                    long long widx_2 = m_base_1_0 + 16 * g_33 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_2) = _iv4;
                    }
                }
                long long v_base_1_1 = rs_w_1 + capo64_1 * 3072;
                long long s_base_1_1 = rs_w_1 + 4 * capo64_1 * 3072 + capo64_1 * 384;
                long long m_base_1_1 = rs_r_1 + capo64_1 * 16;
                unsigned int sv4_0[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4_0[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4_0[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long u_2 = pro64 * 3072 / 16;
                long long lo_2 = u_2 * bid64_1 / g64_1;
                long long hi_2 = u_2 * (bid64_1 + 1) / g64_1;
                long long g0_2 = lo_2 + lane64;
                #pragma unroll 1
                for (long long g_34 = g0_2; g_34 < hi_2; g_34 += 32) {
                    long long widx_3 = v_base_1_1 + 16 * g_34 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4_0[0 + 0], sv4_0[0 + 1], sv4_0[0 + 2], sv4_0[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_3) = _iv4;
                    }
                }
                long long u_3 = pro64 * 384 / 16;
                long long lo_3 = u_3 * bid64_1 / g64_1;
                long long hi_3 = u_3 * (bid64_1 + 1) / g64_1;
                long long g0_3 = lo_3 + lane64;
                #pragma unroll 1
                for (long long g_35 = g0_3; g_35 < hi_3; g_35 += 32) {
                    long long widx_4 = s_base_1_1 + 16 * g_35 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4_0[0 + 0], ss4_0[0 + 1], ss4_0[0 + 2], ss4_0[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_4) = _iv4;
                    }
                }
                long long u_4 = pro64 * 16 / 16;
                long long lo_4 = u_4 * bid64_1 / g64_1;
                long long hi_4 = u_4 * (bid64_1 + 1) / g64_1;
                long long g0_4 = lo_4 + lane64;
                #pragma unroll 1
                for (long long g_36 = g0_4; g_36 < hi_4; g_36 += 32) {
                    long long widx_5 = m_base_1_1 + 16 * g_36 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4_0[0 + 0], sr4_0[0 + 1], sr4_0[0 + 2], sr4_0[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_5) = _iv4;
                    }
                }
                long long v_base_1_2 = rs_w_1 + 2 * capo64_1 * 3072;
                long long s_base_1_2 = rs_w_1 + 4 * capo64_1 * 3072 + 2 * capo64_1 * 384;
                long long m_base_1_2 = rs_r_1 + 2 * capo64_1 * 16;
                unsigned int sv4_1[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4_1[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4_1[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long u_5 = pro64 * 3072 / 16;
                long long lo_5 = u_5 * bid64_1 / g64_1;
                long long hi_5 = u_5 * (bid64_1 + 1) / g64_1;
                long long g0_5 = lo_5 + lane64;
                #pragma unroll 1
                for (long long g_37 = g0_5; g_37 < hi_5; g_37 += 32) {
                    long long widx_6 = v_base_1_2 + 16 * g_37 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4_1[0 + 0], sv4_1[0 + 1], sv4_1[0 + 2], sv4_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_6) = _iv4;
                    }
                }
                long long u_6 = pro64 * 384 / 16;
                long long lo_6 = u_6 * bid64_1 / g64_1;
                long long hi_6 = u_6 * (bid64_1 + 1) / g64_1;
                long long g0_6 = lo_6 + lane64;
                #pragma unroll 1
                for (long long g_38 = g0_6; g_38 < hi_6; g_38 += 32) {
                    long long widx_7 = s_base_1_2 + 16 * g_38 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4_1[0 + 0], ss4_1[0 + 1], ss4_1[0 + 2], ss4_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_7) = _iv4;
                    }
                }
                long long u_7 = pro64 * 16 / 16;
                long long lo_7 = u_7 * bid64_1 / g64_1;
                long long hi_7 = u_7 * (bid64_1 + 1) / g64_1;
                long long g0_7 = lo_7 + lane64;
                #pragma unroll 1
                for (long long g_39 = g0_7; g_39 < hi_7; g_39 += 32) {
                    long long widx_8 = m_base_1_2 + 16 * g_39 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4_1[0 + 0], sr4_1[0 + 1], sr4_1[0 + 2], sr4_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_8) = _iv4;
                    }
                }
                long long v_base_1_3 = rs_w_1 + 3 * capo64_1 * 3072;
                long long s_base_1_3 = rs_w_1 + 4 * capo64_1 * 3072 + 3 * capo64_1 * 384;
                long long m_base_1_3 = rs_r_1 + 3 * capo64_1 * 16;
                unsigned int sv4_2[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4_2[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4_2[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long u_8 = pro64 * 3072 / 16;
                long long lo_8 = u_8 * bid64_1 / g64_1;
                long long hi_8 = u_8 * (bid64_1 + 1) / g64_1;
                long long g0_8 = lo_8 + lane64;
                #pragma unroll 1
                for (long long g_40 = g0_8; g_40 < hi_8; g_40 += 32) {
                    long long widx_9 = v_base_1_3 + 16 * g_40 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4_2[0 + 0], sv4_2[0 + 1], sv4_2[0 + 2], sv4_2[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_9) = _iv4;
                    }
                }
                long long u_9 = pro64 * 384 / 16;
                long long lo_9 = u_9 * bid64_1 / g64_1;
                long long hi_9 = u_9 * (bid64_1 + 1) / g64_1;
                long long g0_9 = lo_9 + lane64;
                #pragma unroll 1
                for (long long g_41 = g0_9; g_41 < hi_9; g_41 += 32) {
                    long long widx_10 = s_base_1_3 + 16 * g_41 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4_2[0 + 0], ss4_2[0 + 1], ss4_2[0 + 2], ss4_2[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_10) = _iv4;
                    }
                }
                long long u_10 = pro64 * 16 / 16;
                long long lo_10 = u_10 * bid64_1 / g64_1;
                long long hi_10 = u_10 * (bid64_1 + 1) / g64_1;
                long long g0_10 = lo_10 + lane64;
                #pragma unroll 1
                for (long long g_42 = g0_10; g_42 < hi_10; g_42 += 32) {
                    long long widx_11 = m_base_1_3 + 16 * g_42 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4_2[0 + 0], sr4_2[0 + 1], sr4_2[0 + 2], sr4_2[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_11) = _iv4;
                    }
                }
                if (0 != rank) {
                    long long v_base2_1_0 = rs_w_1 + sa_w_1;
                    long long s_base2_1_0 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072;
                    long long m_base2_1_0 = rs_r_1 + sa_r_1;
                    unsigned int sv4_3[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                    unsigned int ss4_3[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                    unsigned int sr4_3[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                    long long u_11 = pro64 * 3072 / 16;
                    long long lo_11 = u_11 * bid64_1 / g64_1;
                    long long hi_11 = u_11 * (bid64_1 + 1) / g64_1;
                    long long g0_11 = lo_11 + lane64;
                    #pragma unroll 1
                    for (long long g_43 = g0_11; g_43 < hi_11; g_43 += 32) {
                        long long widx_12 = v_base2_1_0 + 16 * g_43 >> 2;
                        {
                            int4 _iv4 = make_int4(sv4_3[0 + 0], sv4_3[0 + 1], sv4_3[0 + 2], sv4_3[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_12) = _iv4;
                        }
                    }
                    long long u_12 = pro64 * 384 / 16;
                    long long lo_12 = u_12 * bid64_1 / g64_1;
                    long long hi_12 = u_12 * (bid64_1 + 1) / g64_1;
                    long long g0_12 = lo_12 + lane64;
                    #pragma unroll 1
                    for (long long g_44 = g0_12; g_44 < hi_12; g_44 += 32) {
                        long long widx_13 = s_base2_1_0 + 16 * g_44 >> 2;
                        {
                            int4 _iv4 = make_int4(ss4_3[0 + 0], ss4_3[0 + 1], ss4_3[0 + 2], ss4_3[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_13) = _iv4;
                        }
                    }
                    long long u_13 = pro64 * 16 / 16;
                    long long lo_13 = u_13 * bid64_1 / g64_1;
                    long long hi_13 = u_13 * (bid64_1 + 1) / g64_1;
                    long long g0_13 = lo_13 + lane64;
                    #pragma unroll 1
                    for (long long g_45 = g0_13; g_45 < hi_13; g_45 += 32) {
                        long long widx_14 = m_base2_1_0 + 16 * g_45 >> 2;
                        {
                            int4 _iv4 = make_int4(sr4_3[0 + 0], sr4_3[0 + 1], sr4_3[0 + 2], sr4_3[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_14) = _iv4;
                        }
                    }
                }
                if (1 != rank) {
                    long long v_base2_1_1 = rs_w_1 + sa_w_1 + capo64_1 * 3072;
                    long long s_base2_1_1 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + capo64_1 * 384;
                    long long m_base2_1_1 = rs_r_1 + sa_r_1 + capo64_1 * 16;
                    unsigned int sv4_4[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                    unsigned int ss4_4[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                    unsigned int sr4_4[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                    long long u_14 = pro64 * 3072 / 16;
                    long long lo_14 = u_14 * bid64_1 / g64_1;
                    long long hi_14 = u_14 * (bid64_1 + 1) / g64_1;
                    long long g0_14 = lo_14 + lane64;
                    #pragma unroll 1
                    for (long long g_46 = g0_14; g_46 < hi_14; g_46 += 32) {
                        long long widx_15 = v_base2_1_1 + 16 * g_46 >> 2;
                        {
                            int4 _iv4 = make_int4(sv4_4[0 + 0], sv4_4[0 + 1], sv4_4[0 + 2], sv4_4[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_15) = _iv4;
                        }
                    }
                    long long u_15 = pro64 * 384 / 16;
                    long long lo_15 = u_15 * bid64_1 / g64_1;
                    long long hi_15 = u_15 * (bid64_1 + 1) / g64_1;
                    long long g0_15 = lo_15 + lane64;
                    #pragma unroll 1
                    for (long long g_47 = g0_15; g_47 < hi_15; g_47 += 32) {
                        long long widx_16 = s_base2_1_1 + 16 * g_47 >> 2;
                        {
                            int4 _iv4 = make_int4(ss4_4[0 + 0], ss4_4[0 + 1], ss4_4[0 + 2], ss4_4[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_16) = _iv4;
                        }
                    }
                    long long u_16 = pro64 * 16 / 16;
                    long long lo_16 = u_16 * bid64_1 / g64_1;
                    long long hi_16 = u_16 * (bid64_1 + 1) / g64_1;
                    long long g0_16 = lo_16 + lane64;
                    #pragma unroll 1
                    for (long long g_48 = g0_16; g_48 < hi_16; g_48 += 32) {
                        long long widx_17 = m_base2_1_1 + 16 * g_48 >> 2;
                        {
                            int4 _iv4 = make_int4(sr4_4[0 + 0], sr4_4[0 + 1], sr4_4[0 + 2], sr4_4[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_17) = _iv4;
                        }
                    }
                }
                if (2 != rank) {
                    long long v_base2_1_2 = rs_w_1 + sa_w_1 + 2 * capo64_1 * 3072;
                    long long s_base2_1_2 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + 2 * capo64_1 * 384;
                    long long m_base2_1_2 = rs_r_1 + sa_r_1 + 2 * capo64_1 * 16;
                    unsigned int sv4_5[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                    unsigned int ss4_5[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                    unsigned int sr4_5[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                    long long u_17 = pro64 * 3072 / 16;
                    long long lo_17 = u_17 * bid64_1 / g64_1;
                    long long hi_17 = u_17 * (bid64_1 + 1) / g64_1;
                    long long g0_17 = lo_17 + lane64;
                    #pragma unroll 1
                    for (long long g_49 = g0_17; g_49 < hi_17; g_49 += 32) {
                        long long widx_18 = v_base2_1_2 + 16 * g_49 >> 2;
                        {
                            int4 _iv4 = make_int4(sv4_5[0 + 0], sv4_5[0 + 1], sv4_5[0 + 2], sv4_5[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_18) = _iv4;
                        }
                    }
                    long long u_18 = pro64 * 384 / 16;
                    long long lo_18 = u_18 * bid64_1 / g64_1;
                    long long hi_18 = u_18 * (bid64_1 + 1) / g64_1;
                    long long g0_18 = lo_18 + lane64;
                    #pragma unroll 1
                    for (long long g_50 = g0_18; g_50 < hi_18; g_50 += 32) {
                        long long widx_19 = s_base2_1_2 + 16 * g_50 >> 2;
                        {
                            int4 _iv4 = make_int4(ss4_5[0 + 0], ss4_5[0 + 1], ss4_5[0 + 2], ss4_5[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_19) = _iv4;
                        }
                    }
                    long long u_19 = pro64 * 16 / 16;
                    long long lo_19 = u_19 * bid64_1 / g64_1;
                    long long hi_19 = u_19 * (bid64_1 + 1) / g64_1;
                    long long g0_19 = lo_19 + lane64;
                    #pragma unroll 1
                    for (long long g_51 = g0_19; g_51 < hi_19; g_51 += 32) {
                        long long widx_20 = m_base2_1_2 + 16 * g_51 >> 2;
                        {
                            int4 _iv4 = make_int4(sr4_5[0 + 0], sr4_5[0 + 1], sr4_5[0 + 2], sr4_5[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_20) = _iv4;
                        }
                    }
                }
                if (3 != rank) {
                    long long v_base2_1_3 = rs_w_1 + sa_w_1 + 3 * capo64_1 * 3072;
                    long long s_base2_1_3 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + 3 * capo64_1 * 384;
                    long long m_base2_1_3 = rs_r_1 + sa_r_1 + 3 * capo64_1 * 16;
                    unsigned int sv4_6[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                    unsigned int ss4_6[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                    unsigned int sr4_6[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                    long long u_20 = pro64 * 3072 / 16;
                    long long lo_20 = u_20 * bid64_1 / g64_1;
                    long long hi_20 = u_20 * (bid64_1 + 1) / g64_1;
                    long long g0_20 = lo_20 + lane64;
                    #pragma unroll 1
                    for (long long g_52 = g0_20; g_52 < hi_20; g_52 += 32) {
                        long long widx_21 = v_base2_1_3 + 16 * g_52 >> 2;
                        {
                            int4 _iv4 = make_int4(sv4_6[0 + 0], sv4_6[0 + 1], sv4_6[0 + 2], sv4_6[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_21) = _iv4;
                        }
                    }
                    long long u_21 = pro64 * 384 / 16;
                    long long lo_21 = u_21 * bid64_1 / g64_1;
                    long long hi_21 = u_21 * (bid64_1 + 1) / g64_1;
                    long long g0_21 = lo_21 + lane64;
                    #pragma unroll 1
                    for (long long g_53 = g0_21; g_53 < hi_21; g_53 += 32) {
                        long long widx_22 = s_base2_1_3 + 16 * g_53 >> 2;
                        {
                            int4 _iv4 = make_int4(ss4_6[0 + 0], ss4_6[0 + 1], ss4_6[0 + 2], ss4_6[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_wire) + widx_22) = _iv4;
                        }
                    }
                    long long u_22 = pro64 * 16 / 16;
                    long long lo_22 = u_22 * bid64_1 / g64_1;
                    long long hi_22 = u_22 * (bid64_1 + 1) / g64_1;
                    long long g0_22 = lo_22 + lane64;
                    #pragma unroll 1
                    for (long long g_54 = g0_22; g_54 < hi_22; g_54 += 32) {
                        long long widx_23 = m_base2_1_3 + 16 * g_54 >> 2;
                        {
                            int4 _iv4 = make_int4(sr4_6[0 + 0], sr4_6[0 + 1], sr4_6[0 + 2], sr4_6[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(ts_rs) + widx_23) = _iv4;
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
