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
cake_fp4_allreduce_tp4_twoshot_bf16_bv1_csp_b1r_rst(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ ts_wire, uint8_t* __restrict__ ts_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
                mbarrier_arrive_expect_tx(xb_full_addr + (xc) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, x + (first_row * 6144), 12288, xb_full_addr + (xc) * 8);
            }
            unsigned int _phase_xb_full = 0;
            #pragma unroll 1
            for (int k = 0; k < n; k++) {
                int ob_off = own_off + (k & 1) * 3472;
                if (k >= 2) {
                    if (lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                }
                float rs = 1.0f;
                mbarrier_wait(xb_full_addr + (xc) * 8, _phase_xb_full);
                unsigned int bmp[6];
                float am = 0.0f;
                int pp = lane_1;
                int base_1 = xb_off + 64 * pp;
                unsigned int a = 0;
                unsigned int b = 0;
                unsigned int c0[4];
                unsigned int c1[4];
                unsigned int c2[4];
                unsigned int c3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_1 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_1 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_1 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_1 + 16 * (f ^ 3))));
                uint32_t _bf16x2_abs_0;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(c0[0]));
                uint32_t _bf16x2_max_0;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(a), "r"(_bf16x2_abs_0));
                a = _bf16x2_max_0;
                uint32_t _bf16x2_abs_1;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(c0[1]));
                uint32_t _bf16x2_max_1;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(a), "r"(_bf16x2_abs_1));
                a = _bf16x2_max_1;
                uint32_t _bf16x2_abs_2;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(c0[2]));
                uint32_t _bf16x2_max_2;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(a), "r"(_bf16x2_abs_2));
                a = _bf16x2_max_2;
                uint32_t _bf16x2_abs_3;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(c0[3]));
                uint32_t _bf16x2_max_3;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(a), "r"(_bf16x2_abs_3));
                a = _bf16x2_max_3;
                uint32_t _bf16x2_abs_4;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(c1[0]));
                uint32_t _bf16x2_max_4;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(a), "r"(_bf16x2_abs_4));
                a = _bf16x2_max_4;
                uint32_t _bf16x2_abs_5;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(c1[1]));
                uint32_t _bf16x2_max_5;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(a), "r"(_bf16x2_abs_5));
                a = _bf16x2_max_5;
                uint32_t _bf16x2_abs_6;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(c1[2]));
                uint32_t _bf16x2_max_6;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(a), "r"(_bf16x2_abs_6));
                a = _bf16x2_max_6;
                uint32_t _bf16x2_abs_7;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(c1[3]));
                uint32_t _bf16x2_max_7;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_7) : "r"(a), "r"(_bf16x2_abs_7));
                a = _bf16x2_max_7;
                uint32_t _bf16x2_abs_8;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_8) : "r"(c2[0]));
                uint32_t _bf16x2_max_8;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_8) : "r"(b), "r"(_bf16x2_abs_8));
                b = _bf16x2_max_8;
                uint32_t _bf16x2_abs_9;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_9) : "r"(c2[1]));
                uint32_t _bf16x2_max_9;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_9) : "r"(b), "r"(_bf16x2_abs_9));
                b = _bf16x2_max_9;
                uint32_t _bf16x2_abs_10;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_10) : "r"(c2[2]));
                uint32_t _bf16x2_max_10;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_10) : "r"(b), "r"(_bf16x2_abs_10));
                b = _bf16x2_max_10;
                uint32_t _bf16x2_abs_11;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_11) : "r"(c2[3]));
                uint32_t _bf16x2_max_11;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_11) : "r"(b), "r"(_bf16x2_abs_11));
                b = _bf16x2_max_11;
                uint32_t _bf16x2_abs_12;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_12) : "r"(c3[0]));
                uint32_t _bf16x2_max_12;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_12) : "r"(b), "r"(_bf16x2_abs_12));
                b = _bf16x2_max_12;
                uint32_t _bf16x2_abs_13;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_13) : "r"(c3[1]));
                uint32_t _bf16x2_max_13;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_13) : "r"(b), "r"(_bf16x2_abs_13));
                b = _bf16x2_max_13;
                uint32_t _bf16x2_abs_14;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_14) : "r"(c3[2]));
                uint32_t _bf16x2_max_14;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_14) : "r"(b), "r"(_bf16x2_abs_14));
                b = _bf16x2_max_14;
                uint32_t _bf16x2_abs_15;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_15) : "r"(c3[3]));
                uint32_t _bf16x2_max_15;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_15) : "r"(b), "r"(_bf16x2_abs_15));
                b = _bf16x2_max_15;
                unsigned int xa = ((a & 65535) > (a >> 16) ? (a & 65535) : (a >> 16));
                unsigned int xb = ((b & 65535) > (b >> 16) ? (b & 65535) : (b >> 16));
                if ((f & 2) != 0) {
                    bmp[0] = xb | xa << 16;
                } else {
                    bmp[0] = xa | xb << 16;
                }
                unsigned int _max_0 = ((xa) > (xb) ? (xa) : (xb));
                am = fmaxf(am, __uint_as_float(_max_0 << 16));
                int pp_0 = lane_1 + 32;
                int base_1_1 = xb_off + 64 * pp_0;
                unsigned int a_2 = 0;
                unsigned int b_3 = 0;
                unsigned int c0_4[4];
                unsigned int c1_5[4];
                unsigned int c2_6[4];
                unsigned int c3_7[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_4[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_1_1 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_5[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_1_1 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_6[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_1_1 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_7[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_1_1 + 16 * (f ^ 3))));
                uint32_t _bf16x2_abs_16;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16) : "r"(c0_4[0]));
                uint32_t _bf16x2_max_16;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_16) : "r"(a_2), "r"(_bf16x2_abs_16));
                a_2 = _bf16x2_max_16;
                uint32_t _bf16x2_abs_17;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_17) : "r"(c0_4[1]));
                uint32_t _bf16x2_max_17;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_17) : "r"(a_2), "r"(_bf16x2_abs_17));
                a_2 = _bf16x2_max_17;
                uint32_t _bf16x2_abs_18;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_18) : "r"(c0_4[2]));
                uint32_t _bf16x2_max_18;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_18) : "r"(a_2), "r"(_bf16x2_abs_18));
                a_2 = _bf16x2_max_18;
                uint32_t _bf16x2_abs_19;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_19) : "r"(c0_4[3]));
                uint32_t _bf16x2_max_19;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_19) : "r"(a_2), "r"(_bf16x2_abs_19));
                a_2 = _bf16x2_max_19;
                uint32_t _bf16x2_abs_20;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_20) : "r"(c1_5[0]));
                uint32_t _bf16x2_max_20;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_20) : "r"(a_2), "r"(_bf16x2_abs_20));
                a_2 = _bf16x2_max_20;
                uint32_t _bf16x2_abs_21;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_21) : "r"(c1_5[1]));
                uint32_t _bf16x2_max_21;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_21) : "r"(a_2), "r"(_bf16x2_abs_21));
                a_2 = _bf16x2_max_21;
                uint32_t _bf16x2_abs_22;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_22) : "r"(c1_5[2]));
                uint32_t _bf16x2_max_22;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_22) : "r"(a_2), "r"(_bf16x2_abs_22));
                a_2 = _bf16x2_max_22;
                uint32_t _bf16x2_abs_23;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_23) : "r"(c1_5[3]));
                uint32_t _bf16x2_max_23;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_23) : "r"(a_2), "r"(_bf16x2_abs_23));
                a_2 = _bf16x2_max_23;
                uint32_t _bf16x2_abs_24;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_24) : "r"(c2_6[0]));
                uint32_t _bf16x2_max_24;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_24) : "r"(b_3), "r"(_bf16x2_abs_24));
                b_3 = _bf16x2_max_24;
                uint32_t _bf16x2_abs_25;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_25) : "r"(c2_6[1]));
                uint32_t _bf16x2_max_25;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_25) : "r"(b_3), "r"(_bf16x2_abs_25));
                b_3 = _bf16x2_max_25;
                uint32_t _bf16x2_abs_26;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_26) : "r"(c2_6[2]));
                uint32_t _bf16x2_max_26;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_26) : "r"(b_3), "r"(_bf16x2_abs_26));
                b_3 = _bf16x2_max_26;
                uint32_t _bf16x2_abs_27;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_27) : "r"(c2_6[3]));
                uint32_t _bf16x2_max_27;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_27) : "r"(b_3), "r"(_bf16x2_abs_27));
                b_3 = _bf16x2_max_27;
                uint32_t _bf16x2_abs_28;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_28) : "r"(c3_7[0]));
                uint32_t _bf16x2_max_28;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_28) : "r"(b_3), "r"(_bf16x2_abs_28));
                b_3 = _bf16x2_max_28;
                uint32_t _bf16x2_abs_29;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_29) : "r"(c3_7[1]));
                uint32_t _bf16x2_max_29;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_29) : "r"(b_3), "r"(_bf16x2_abs_29));
                b_3 = _bf16x2_max_29;
                uint32_t _bf16x2_abs_30;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_30) : "r"(c3_7[2]));
                uint32_t _bf16x2_max_30;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_30) : "r"(b_3), "r"(_bf16x2_abs_30));
                b_3 = _bf16x2_max_30;
                uint32_t _bf16x2_abs_31;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_31) : "r"(c3_7[3]));
                uint32_t _bf16x2_max_31;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_31) : "r"(b_3), "r"(_bf16x2_abs_31));
                b_3 = _bf16x2_max_31;
                unsigned int xa_8 = ((a_2 & 65535) > (a_2 >> 16) ? (a_2 & 65535) : (a_2 >> 16));
                unsigned int xb_9 = ((b_3 & 65535) > (b_3 >> 16) ? (b_3 & 65535) : (b_3 >> 16));
                if ((f & 2) != 0) {
                    bmp[1] = xb_9 | xa_8 << 16;
                } else {
                    bmp[1] = xa_8 | xb_9 << 16;
                }
                unsigned int _max_1 = ((xa_8) > (xb_9) ? (xa_8) : (xb_9));
                am = fmaxf(am, __uint_as_float(_max_1 << 16));
                int pp_10 = lane_1 + 64;
                int base_11 = xb_off + 64 * pp_10;
                unsigned int a_12 = 0;
                unsigned int b_13 = 0;
                unsigned int c0_14[4];
                unsigned int c1_15[4];
                unsigned int c2_16[4];
                unsigned int c3_17[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_14[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_11 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_15[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_11 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_16[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_11 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_17[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_11 + 16 * (f ^ 3))));
                uint32_t _bf16x2_abs_32;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_32) : "r"(c0_14[0]));
                uint32_t _bf16x2_max_32;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_32) : "r"(a_12), "r"(_bf16x2_abs_32));
                a_12 = _bf16x2_max_32;
                uint32_t _bf16x2_abs_33;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_33) : "r"(c0_14[1]));
                uint32_t _bf16x2_max_33;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_33) : "r"(a_12), "r"(_bf16x2_abs_33));
                a_12 = _bf16x2_max_33;
                uint32_t _bf16x2_abs_34;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_34) : "r"(c0_14[2]));
                uint32_t _bf16x2_max_34;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_34) : "r"(a_12), "r"(_bf16x2_abs_34));
                a_12 = _bf16x2_max_34;
                uint32_t _bf16x2_abs_35;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_35) : "r"(c0_14[3]));
                uint32_t _bf16x2_max_35;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_35) : "r"(a_12), "r"(_bf16x2_abs_35));
                a_12 = _bf16x2_max_35;
                uint32_t _bf16x2_abs_36;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_36) : "r"(c1_15[0]));
                uint32_t _bf16x2_max_36;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_36) : "r"(a_12), "r"(_bf16x2_abs_36));
                a_12 = _bf16x2_max_36;
                uint32_t _bf16x2_abs_37;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_37) : "r"(c1_15[1]));
                uint32_t _bf16x2_max_37;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_37) : "r"(a_12), "r"(_bf16x2_abs_37));
                a_12 = _bf16x2_max_37;
                uint32_t _bf16x2_abs_38;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_38) : "r"(c1_15[2]));
                uint32_t _bf16x2_max_38;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_38) : "r"(a_12), "r"(_bf16x2_abs_38));
                a_12 = _bf16x2_max_38;
                uint32_t _bf16x2_abs_39;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_39) : "r"(c1_15[3]));
                uint32_t _bf16x2_max_39;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_39) : "r"(a_12), "r"(_bf16x2_abs_39));
                a_12 = _bf16x2_max_39;
                uint32_t _bf16x2_abs_40;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_40) : "r"(c2_16[0]));
                uint32_t _bf16x2_max_40;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_40) : "r"(b_13), "r"(_bf16x2_abs_40));
                b_13 = _bf16x2_max_40;
                uint32_t _bf16x2_abs_41;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_41) : "r"(c2_16[1]));
                uint32_t _bf16x2_max_41;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_41) : "r"(b_13), "r"(_bf16x2_abs_41));
                b_13 = _bf16x2_max_41;
                uint32_t _bf16x2_abs_42;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_42) : "r"(c2_16[2]));
                uint32_t _bf16x2_max_42;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_42) : "r"(b_13), "r"(_bf16x2_abs_42));
                b_13 = _bf16x2_max_42;
                uint32_t _bf16x2_abs_43;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_43) : "r"(c2_16[3]));
                uint32_t _bf16x2_max_43;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_43) : "r"(b_13), "r"(_bf16x2_abs_43));
                b_13 = _bf16x2_max_43;
                uint32_t _bf16x2_abs_44;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_44) : "r"(c3_17[0]));
                uint32_t _bf16x2_max_44;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_44) : "r"(b_13), "r"(_bf16x2_abs_44));
                b_13 = _bf16x2_max_44;
                uint32_t _bf16x2_abs_45;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_45) : "r"(c3_17[1]));
                uint32_t _bf16x2_max_45;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_45) : "r"(b_13), "r"(_bf16x2_abs_45));
                b_13 = _bf16x2_max_45;
                uint32_t _bf16x2_abs_46;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_46) : "r"(c3_17[2]));
                uint32_t _bf16x2_max_46;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_46) : "r"(b_13), "r"(_bf16x2_abs_46));
                b_13 = _bf16x2_max_46;
                uint32_t _bf16x2_abs_47;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_47) : "r"(c3_17[3]));
                uint32_t _bf16x2_max_47;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_47) : "r"(b_13), "r"(_bf16x2_abs_47));
                b_13 = _bf16x2_max_47;
                unsigned int xa_18 = ((a_12 & 65535) > (a_12 >> 16) ? (a_12 & 65535) : (a_12 >> 16));
                unsigned int xb_19 = ((b_13 & 65535) > (b_13 >> 16) ? (b_13 & 65535) : (b_13 >> 16));
                if ((f & 2) != 0) {
                    bmp[2] = xb_19 | xa_18 << 16;
                } else {
                    bmp[2] = xa_18 | xb_19 << 16;
                }
                unsigned int _max_2 = ((xa_18) > (xb_19) ? (xa_18) : (xb_19));
                am = fmaxf(am, __uint_as_float(_max_2 << 16));
                int pp_20 = lane_1 + 96;
                int base_21 = xb_off + 64 * pp_20;
                unsigned int a_22 = 0;
                unsigned int b_23 = 0;
                unsigned int c0_24[4];
                unsigned int c1_25[4];
                unsigned int c2_26[4];
                unsigned int c3_27[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_24[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_21 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_25[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_21 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_26[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_21 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_27[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_21 + 16 * (f ^ 3))));
                uint32_t _bf16x2_abs_48;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_48) : "r"(c0_24[0]));
                uint32_t _bf16x2_max_48;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_48) : "r"(a_22), "r"(_bf16x2_abs_48));
                a_22 = _bf16x2_max_48;
                uint32_t _bf16x2_abs_49;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_49) : "r"(c0_24[1]));
                uint32_t _bf16x2_max_49;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_49) : "r"(a_22), "r"(_bf16x2_abs_49));
                a_22 = _bf16x2_max_49;
                uint32_t _bf16x2_abs_50;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_50) : "r"(c0_24[2]));
                uint32_t _bf16x2_max_50;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_50) : "r"(a_22), "r"(_bf16x2_abs_50));
                a_22 = _bf16x2_max_50;
                uint32_t _bf16x2_abs_51;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_51) : "r"(c0_24[3]));
                uint32_t _bf16x2_max_51;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_51) : "r"(a_22), "r"(_bf16x2_abs_51));
                a_22 = _bf16x2_max_51;
                uint32_t _bf16x2_abs_52;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_52) : "r"(c1_25[0]));
                uint32_t _bf16x2_max_52;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_52) : "r"(a_22), "r"(_bf16x2_abs_52));
                a_22 = _bf16x2_max_52;
                uint32_t _bf16x2_abs_53;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_53) : "r"(c1_25[1]));
                uint32_t _bf16x2_max_53;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_53) : "r"(a_22), "r"(_bf16x2_abs_53));
                a_22 = _bf16x2_max_53;
                uint32_t _bf16x2_abs_54;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_54) : "r"(c1_25[2]));
                uint32_t _bf16x2_max_54;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_54) : "r"(a_22), "r"(_bf16x2_abs_54));
                a_22 = _bf16x2_max_54;
                uint32_t _bf16x2_abs_55;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_55) : "r"(c1_25[3]));
                uint32_t _bf16x2_max_55;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_55) : "r"(a_22), "r"(_bf16x2_abs_55));
                a_22 = _bf16x2_max_55;
                uint32_t _bf16x2_abs_56;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_56) : "r"(c2_26[0]));
                uint32_t _bf16x2_max_56;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_56) : "r"(b_23), "r"(_bf16x2_abs_56));
                b_23 = _bf16x2_max_56;
                uint32_t _bf16x2_abs_57;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_57) : "r"(c2_26[1]));
                uint32_t _bf16x2_max_57;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_57) : "r"(b_23), "r"(_bf16x2_abs_57));
                b_23 = _bf16x2_max_57;
                uint32_t _bf16x2_abs_58;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_58) : "r"(c2_26[2]));
                uint32_t _bf16x2_max_58;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_58) : "r"(b_23), "r"(_bf16x2_abs_58));
                b_23 = _bf16x2_max_58;
                uint32_t _bf16x2_abs_59;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_59) : "r"(c2_26[3]));
                uint32_t _bf16x2_max_59;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_59) : "r"(b_23), "r"(_bf16x2_abs_59));
                b_23 = _bf16x2_max_59;
                uint32_t _bf16x2_abs_60;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_60) : "r"(c3_27[0]));
                uint32_t _bf16x2_max_60;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_60) : "r"(b_23), "r"(_bf16x2_abs_60));
                b_23 = _bf16x2_max_60;
                uint32_t _bf16x2_abs_61;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_61) : "r"(c3_27[1]));
                uint32_t _bf16x2_max_61;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_61) : "r"(b_23), "r"(_bf16x2_abs_61));
                b_23 = _bf16x2_max_61;
                uint32_t _bf16x2_abs_62;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_62) : "r"(c3_27[2]));
                uint32_t _bf16x2_max_62;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_62) : "r"(b_23), "r"(_bf16x2_abs_62));
                b_23 = _bf16x2_max_62;
                uint32_t _bf16x2_abs_63;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_63) : "r"(c3_27[3]));
                uint32_t _bf16x2_max_63;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_63) : "r"(b_23), "r"(_bf16x2_abs_63));
                b_23 = _bf16x2_max_63;
                unsigned int xa_28 = ((a_22 & 65535) > (a_22 >> 16) ? (a_22 & 65535) : (a_22 >> 16));
                unsigned int xb_29 = ((b_23 & 65535) > (b_23 >> 16) ? (b_23 & 65535) : (b_23 >> 16));
                if ((f & 2) != 0) {
                    bmp[3] = xb_29 | xa_28 << 16;
                } else {
                    bmp[3] = xa_28 | xb_29 << 16;
                }
                unsigned int _max_3 = ((xa_28) > (xb_29) ? (xa_28) : (xb_29));
                am = fmaxf(am, __uint_as_float(_max_3 << 16));
                int pp_30 = lane_1 + 128;
                int base_31 = xb_off + 64 * pp_30;
                unsigned int a_32 = 0;
                unsigned int b_33 = 0;
                unsigned int c0_34[4];
                unsigned int c1_35[4];
                unsigned int c2_36[4];
                unsigned int c3_37[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_34[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_31 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_35[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_31 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_36[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_31 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_37[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_31 + 16 * (f ^ 3))));
                uint32_t _bf16x2_abs_64;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_64) : "r"(c0_34[0]));
                uint32_t _bf16x2_max_64;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_64) : "r"(a_32), "r"(_bf16x2_abs_64));
                a_32 = _bf16x2_max_64;
                uint32_t _bf16x2_abs_65;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_65) : "r"(c0_34[1]));
                uint32_t _bf16x2_max_65;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_65) : "r"(a_32), "r"(_bf16x2_abs_65));
                a_32 = _bf16x2_max_65;
                uint32_t _bf16x2_abs_66;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_66) : "r"(c0_34[2]));
                uint32_t _bf16x2_max_66;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_66) : "r"(a_32), "r"(_bf16x2_abs_66));
                a_32 = _bf16x2_max_66;
                uint32_t _bf16x2_abs_67;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_67) : "r"(c0_34[3]));
                uint32_t _bf16x2_max_67;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_67) : "r"(a_32), "r"(_bf16x2_abs_67));
                a_32 = _bf16x2_max_67;
                uint32_t _bf16x2_abs_68;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_68) : "r"(c1_35[0]));
                uint32_t _bf16x2_max_68;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_68) : "r"(a_32), "r"(_bf16x2_abs_68));
                a_32 = _bf16x2_max_68;
                uint32_t _bf16x2_abs_69;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_69) : "r"(c1_35[1]));
                uint32_t _bf16x2_max_69;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_69) : "r"(a_32), "r"(_bf16x2_abs_69));
                a_32 = _bf16x2_max_69;
                uint32_t _bf16x2_abs_70;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_70) : "r"(c1_35[2]));
                uint32_t _bf16x2_max_70;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_70) : "r"(a_32), "r"(_bf16x2_abs_70));
                a_32 = _bf16x2_max_70;
                uint32_t _bf16x2_abs_71;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_71) : "r"(c1_35[3]));
                uint32_t _bf16x2_max_71;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_71) : "r"(a_32), "r"(_bf16x2_abs_71));
                a_32 = _bf16x2_max_71;
                uint32_t _bf16x2_abs_72;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_72) : "r"(c2_36[0]));
                uint32_t _bf16x2_max_72;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_72) : "r"(b_33), "r"(_bf16x2_abs_72));
                b_33 = _bf16x2_max_72;
                uint32_t _bf16x2_abs_73;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_73) : "r"(c2_36[1]));
                uint32_t _bf16x2_max_73;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_73) : "r"(b_33), "r"(_bf16x2_abs_73));
                b_33 = _bf16x2_max_73;
                uint32_t _bf16x2_abs_74;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_74) : "r"(c2_36[2]));
                uint32_t _bf16x2_max_74;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_74) : "r"(b_33), "r"(_bf16x2_abs_74));
                b_33 = _bf16x2_max_74;
                uint32_t _bf16x2_abs_75;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_75) : "r"(c2_36[3]));
                uint32_t _bf16x2_max_75;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_75) : "r"(b_33), "r"(_bf16x2_abs_75));
                b_33 = _bf16x2_max_75;
                uint32_t _bf16x2_abs_76;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_76) : "r"(c3_37[0]));
                uint32_t _bf16x2_max_76;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_76) : "r"(b_33), "r"(_bf16x2_abs_76));
                b_33 = _bf16x2_max_76;
                uint32_t _bf16x2_abs_77;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_77) : "r"(c3_37[1]));
                uint32_t _bf16x2_max_77;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_77) : "r"(b_33), "r"(_bf16x2_abs_77));
                b_33 = _bf16x2_max_77;
                uint32_t _bf16x2_abs_78;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_78) : "r"(c3_37[2]));
                uint32_t _bf16x2_max_78;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_78) : "r"(b_33), "r"(_bf16x2_abs_78));
                b_33 = _bf16x2_max_78;
                uint32_t _bf16x2_abs_79;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_79) : "r"(c3_37[3]));
                uint32_t _bf16x2_max_79;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_79) : "r"(b_33), "r"(_bf16x2_abs_79));
                b_33 = _bf16x2_max_79;
                unsigned int xa_38 = ((a_32 & 65535) > (a_32 >> 16) ? (a_32 & 65535) : (a_32 >> 16));
                unsigned int xb_39 = ((b_33 & 65535) > (b_33 >> 16) ? (b_33 & 65535) : (b_33 >> 16));
                if ((f & 2) != 0) {
                    bmp[4] = xb_39 | xa_38 << 16;
                } else {
                    bmp[4] = xa_38 | xb_39 << 16;
                }
                unsigned int _max_4 = ((xa_38) > (xb_39) ? (xa_38) : (xb_39));
                am = fmaxf(am, __uint_as_float(_max_4 << 16));
                int pp_40 = lane_1 + 160;
                int base_41 = xb_off + 64 * pp_40;
                unsigned int a_42 = 0;
                unsigned int b_43 = 0;
                unsigned int c0_44[4];
                unsigned int c1_45[4];
                unsigned int c2_46[4];
                unsigned int c3_47[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c0_44[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_44[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_44[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_44[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_41 + 16 * f)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c1_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_45[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_41 + 16 * (f ^ 1))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c2_46[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_46[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_41 + 16 * (f ^ 2))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&c3_47[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_47[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base_41 + 16 * (f ^ 3))));
                uint32_t _bf16x2_abs_80;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_80) : "r"(c0_44[0]));
                uint32_t _bf16x2_max_80;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_80) : "r"(a_42), "r"(_bf16x2_abs_80));
                a_42 = _bf16x2_max_80;
                uint32_t _bf16x2_abs_81;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_81) : "r"(c0_44[1]));
                uint32_t _bf16x2_max_81;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_81) : "r"(a_42), "r"(_bf16x2_abs_81));
                a_42 = _bf16x2_max_81;
                uint32_t _bf16x2_abs_82;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_82) : "r"(c0_44[2]));
                uint32_t _bf16x2_max_82;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_82) : "r"(a_42), "r"(_bf16x2_abs_82));
                a_42 = _bf16x2_max_82;
                uint32_t _bf16x2_abs_83;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_83) : "r"(c0_44[3]));
                uint32_t _bf16x2_max_83;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_83) : "r"(a_42), "r"(_bf16x2_abs_83));
                a_42 = _bf16x2_max_83;
                uint32_t _bf16x2_abs_84;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_84) : "r"(c1_45[0]));
                uint32_t _bf16x2_max_84;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_84) : "r"(a_42), "r"(_bf16x2_abs_84));
                a_42 = _bf16x2_max_84;
                uint32_t _bf16x2_abs_85;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_85) : "r"(c1_45[1]));
                uint32_t _bf16x2_max_85;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_85) : "r"(a_42), "r"(_bf16x2_abs_85));
                a_42 = _bf16x2_max_85;
                uint32_t _bf16x2_abs_86;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_86) : "r"(c1_45[2]));
                uint32_t _bf16x2_max_86;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_86) : "r"(a_42), "r"(_bf16x2_abs_86));
                a_42 = _bf16x2_max_86;
                uint32_t _bf16x2_abs_87;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_87) : "r"(c1_45[3]));
                uint32_t _bf16x2_max_87;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_87) : "r"(a_42), "r"(_bf16x2_abs_87));
                a_42 = _bf16x2_max_87;
                uint32_t _bf16x2_abs_88;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_88) : "r"(c2_46[0]));
                uint32_t _bf16x2_max_88;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_88) : "r"(b_43), "r"(_bf16x2_abs_88));
                b_43 = _bf16x2_max_88;
                uint32_t _bf16x2_abs_89;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_89) : "r"(c2_46[1]));
                uint32_t _bf16x2_max_89;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_89) : "r"(b_43), "r"(_bf16x2_abs_89));
                b_43 = _bf16x2_max_89;
                uint32_t _bf16x2_abs_90;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_90) : "r"(c2_46[2]));
                uint32_t _bf16x2_max_90;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_90) : "r"(b_43), "r"(_bf16x2_abs_90));
                b_43 = _bf16x2_max_90;
                uint32_t _bf16x2_abs_91;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_91) : "r"(c2_46[3]));
                uint32_t _bf16x2_max_91;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_91) : "r"(b_43), "r"(_bf16x2_abs_91));
                b_43 = _bf16x2_max_91;
                uint32_t _bf16x2_abs_92;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_92) : "r"(c3_47[0]));
                uint32_t _bf16x2_max_92;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_92) : "r"(b_43), "r"(_bf16x2_abs_92));
                b_43 = _bf16x2_max_92;
                uint32_t _bf16x2_abs_93;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_93) : "r"(c3_47[1]));
                uint32_t _bf16x2_max_93;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_93) : "r"(b_43), "r"(_bf16x2_abs_93));
                b_43 = _bf16x2_max_93;
                uint32_t _bf16x2_abs_94;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_94) : "r"(c3_47[2]));
                uint32_t _bf16x2_max_94;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_94) : "r"(b_43), "r"(_bf16x2_abs_94));
                b_43 = _bf16x2_max_94;
                uint32_t _bf16x2_abs_95;
                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_95) : "r"(c3_47[3]));
                uint32_t _bf16x2_max_95;
                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_95) : "r"(b_43), "r"(_bf16x2_abs_95));
                b_43 = _bf16x2_max_95;
                unsigned int xa_48 = ((a_42 & 65535) > (a_42 >> 16) ? (a_42 & 65535) : (a_42 >> 16));
                unsigned int xb_49 = ((b_43 & 65535) > (b_43 >> 16) ? (b_43 & 65535) : (b_43 >> 16));
                if ((f & 2) != 0) {
                    bmp[5] = xb_49 | xa_48 << 16;
                } else {
                    bmp[5] = xa_48 | xb_49 << 16;
                }
                unsigned int _max_5 = ((xa_48) > (xb_49) ? (xa_48) : (xb_49));
                am = fmaxf(am, __uint_as_float(_max_5 << 16));
                float _warp_reduce_6 = am;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_6 = max_noftz(_warp_reduce_6, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset));
                am = _warp_reduce_6;
                float fdiv_rn = __fdiv_rn(2688.0f, am);
                float inv = 0.0f;
                if (am > 0.0f) {
                    inv = fdiv_rn;
                }
                float rs_0 = __fdiv_rn(am, 2688.0f);
                int pp2_1_0 = lane_1;
                int base2_1_0 = xb_off + 64 * pp2_1_0;
                float bm0_1_0 = __uint_as_float(bmp[0] << 16);
                float bm1_1_0 = __uint_as_float(bmp[0] & 4294901760u);
                float _fdiv_7 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_0, inv), 6.0f);
                float _fdiv_8 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_0, inv), 6.0f);
                uint16_t e4m3x2_f32;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32) : "f"(_fdiv_7), "f"(_fdiv_8));
                uint32_t e4m3x2_to_f16x2;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2) : "h"((uint16_t)(e4m3x2_f32)));
                float cvt_f32_f16;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16) : "h"((uint16_t)(e4m3x2_to_f16x2)));
                float cvt_f32_f16_1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1) : "h"((uint16_t)(e4m3x2_to_f16x2 >> 16)));
                float fdiv_rn_0 = __fdiv_rn(inv, cvt_f32_f16);
                float fdiv_rn_1 = __fdiv_rn(inv, cvt_f32_f16_1);
                float m0 = 0.0f;
                float m1 = 0.0f;
                if (cvt_f32_f16 > 0.0f) {
                    m0 = fdiv_rn_0;
                }
                if (cvt_f32_f16_1 > 0.0f) {
                    m1 = fdiv_rn_1;
                }
                unsigned int w_1_0[4];
                int cf_1_0 = f ^ 0;
                unsigned int v4_1_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_0)));
                float m_1_0 = m0;
                if ((cf_1_0 & 2) != 0) {
                    m_1_0 = m1;
                }
                float2 mm = make_float2(m_1_0, m_1_0);
                float2 pair = make_float2(__uint_as_float(v4_1_0[0] << 16), __uint_as_float(v4_1_0[0] & 4294901760u));
                float2 g;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&pair), "l"(*(const unsigned long long*)&mm));
                uint32_t fp4_pair;
                uint16_t _e2m1_pair_h_9;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_9) : "f"(g.x), "f"(g.y));
                fp4_pair = (uint32_t)_e2m1_pair_h_9;
                float2 pair_1 = make_float2(__uint_as_float(v4_1_0[1] << 16), __uint_as_float(v4_1_0[1] & 4294901760u));
                float2 g_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1) : "l"(*(const unsigned long long*)&pair_1), "l"(*(const unsigned long long*)&mm));
                uint32_t fp4_pair_1;
                uint16_t _e2m1_pair_h_10;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(g_1.x), "f"(g_1.y));
                fp4_pair_1 = (uint32_t)_e2m1_pair_h_10;
                float2 pair_2 = make_float2(__uint_as_float(v4_1_0[2] << 16), __uint_as_float(v4_1_0[2] & 4294901760u));
                float2 g_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2) : "l"(*(const unsigned long long*)&pair_2), "l"(*(const unsigned long long*)&mm));
                uint32_t fp4_pair_2;
                uint16_t _e2m1_pair_h_11;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(g_2.x), "f"(g_2.y));
                fp4_pair_2 = (uint32_t)_e2m1_pair_h_11;
                float2 pair_3 = make_float2(__uint_as_float(v4_1_0[3] << 16), __uint_as_float(v4_1_0[3] & 4294901760u));
                float2 g_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3) : "l"(*(const unsigned long long*)&pair_3), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_e2m1_pack_pair_12;
                uint16_t _e2m1_pair_h_13;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(g_3.x), "f"(g_3.y));
                _fp4_e2m1_pack_pair_12 = (uint32_t)_e2m1_pair_h_13;
                unsigned int word = fp4_pair | fp4_pair_1 << 8 | fp4_pair_2 << 16 | _fp4_e2m1_pack_pair_12 << 24;
                unsigned int carry = (word & 2004318071) + 2004318071;
                w_1_0[0] = word & (carry | 2004318071);
                int cf_1_1 = f ^ 1;
                unsigned int v4_1_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_1)));
                float m_1_1 = m0;
                if ((cf_1_1 & 2) != 0) {
                    m_1_1 = m1;
                }
                float2 mm_0 = make_float2(m_1_1, m_1_1);
                float2 pair_0 = make_float2(__uint_as_float(v4_1_1[0] << 16), __uint_as_float(v4_1_1[0] & 4294901760u));
                float2 g_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_0) : "l"(*(const unsigned long long*)&pair_0), "l"(*(const unsigned long long*)&mm_0));
                uint32_t fp4_pair_0;
                uint16_t _e2m1_pair_h_14;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(g_0.x), "f"(g_0.y));
                fp4_pair_0 = (uint32_t)_e2m1_pair_h_14;
                float2 pair_1_0 = make_float2(__uint_as_float(v4_1_1[1] << 16), __uint_as_float(v4_1_1[1] & 4294901760u));
                float2 g_1_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_0) : "l"(*(const unsigned long long*)&pair_1_0), "l"(*(const unsigned long long*)&mm_0));
                uint32_t fp4_pair_1_0;
                uint16_t _e2m1_pair_h_15;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_1_0.x), "f"(g_1_0.y));
                fp4_pair_1_0 = (uint32_t)_e2m1_pair_h_15;
                float2 pair_2_0 = make_float2(__uint_as_float(v4_1_1[2] << 16), __uint_as_float(v4_1_1[2] & 4294901760u));
                float2 g_2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_0) : "l"(*(const unsigned long long*)&pair_2_0), "l"(*(const unsigned long long*)&mm_0));
                uint32_t fp4_pair_2_0;
                uint16_t _e2m1_pair_h_16;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_2_0.x), "f"(g_2_0.y));
                fp4_pair_2_0 = (uint32_t)_e2m1_pair_h_16;
                float2 pair_3_0 = make_float2(__uint_as_float(v4_1_1[3] << 16), __uint_as_float(v4_1_1[3] & 4294901760u));
                float2 g_3_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_0) : "l"(*(const unsigned long long*)&pair_3_0), "l"(*(const unsigned long long*)&mm_0));
                uint32_t _fp4_e2m1_pack_pair_17;
                uint16_t _e2m1_pair_h_18;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(g_3_0.x), "f"(g_3_0.y));
                _fp4_e2m1_pack_pair_17 = (uint32_t)_e2m1_pair_h_18;
                unsigned int word_0 = fp4_pair_0 | fp4_pair_1_0 << 8 | fp4_pair_2_0 << 16 | _fp4_e2m1_pack_pair_17 << 24;
                unsigned int carry_0 = (word_0 & 2004318071) + 2004318071;
                w_1_0[1] = word_0 & (carry_0 | 2004318071);
                int cf_1_2 = f ^ 2;
                unsigned int v4_1_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_2)));
                float m_1_2 = m0;
                if ((cf_1_2 & 2) != 0) {
                    m_1_2 = m1;
                }
                float2 mm_1 = make_float2(m_1_2, m_1_2);
                float2 pair_4 = make_float2(__uint_as_float(v4_1_2[0] << 16), __uint_as_float(v4_1_2[0] & 4294901760u));
                float2 g_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_4) : "l"(*(const unsigned long long*)&pair_4), "l"(*(const unsigned long long*)&mm_1));
                uint32_t fp4_pair_3;
                uint16_t _e2m1_pair_h_19;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_19) : "f"(g_4.x), "f"(g_4.y));
                fp4_pair_3 = (uint32_t)_e2m1_pair_h_19;
                float2 pair_1_1 = make_float2(__uint_as_float(v4_1_2[1] << 16), __uint_as_float(v4_1_2[1] & 4294901760u));
                float2 g_1_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_1) : "l"(*(const unsigned long long*)&pair_1_1), "l"(*(const unsigned long long*)&mm_1));
                uint32_t fp4_pair_1_1;
                uint16_t _e2m1_pair_h_20;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_1_1.x), "f"(g_1_1.y));
                fp4_pair_1_1 = (uint32_t)_e2m1_pair_h_20;
                float2 pair_2_1 = make_float2(__uint_as_float(v4_1_2[2] << 16), __uint_as_float(v4_1_2[2] & 4294901760u));
                float2 g_2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_1) : "l"(*(const unsigned long long*)&pair_2_1), "l"(*(const unsigned long long*)&mm_1));
                uint32_t fp4_pair_2_1;
                uint16_t _e2m1_pair_h_21;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_2_1.x), "f"(g_2_1.y));
                fp4_pair_2_1 = (uint32_t)_e2m1_pair_h_21;
                float2 pair_3_1 = make_float2(__uint_as_float(v4_1_2[3] << 16), __uint_as_float(v4_1_2[3] & 4294901760u));
                float2 g_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_1) : "l"(*(const unsigned long long*)&pair_3_1), "l"(*(const unsigned long long*)&mm_1));
                uint32_t _fp4_e2m1_pack_pair_22;
                uint16_t _e2m1_pair_h_23;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(g_3_1.x), "f"(g_3_1.y));
                _fp4_e2m1_pack_pair_22 = (uint32_t)_e2m1_pair_h_23;
                unsigned int word_1 = fp4_pair_3 | fp4_pair_1_1 << 8 | fp4_pair_2_1 << 16 | _fp4_e2m1_pack_pair_22 << 24;
                unsigned int carry_1 = (word_1 & 2004318071) + 2004318071;
                w_1_0[2] = word_1 & (carry_1 | 2004318071);
                int cf_1_3 = f ^ 3;
                unsigned int v4_1_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_3)));
                float m_1_3 = m0;
                if ((cf_1_3 & 2) != 0) {
                    m_1_3 = m1;
                }
                float2 mm_2 = make_float2(m_1_3, m_1_3);
                float2 pair_5 = make_float2(__uint_as_float(v4_1_3[0] << 16), __uint_as_float(v4_1_3[0] & 4294901760u));
                float2 g_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_5) : "l"(*(const unsigned long long*)&pair_5), "l"(*(const unsigned long long*)&mm_2));
                uint32_t fp4_pair_4;
                uint16_t _e2m1_pair_h_24;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_24) : "f"(g_5.x), "f"(g_5.y));
                fp4_pair_4 = (uint32_t)_e2m1_pair_h_24;
                float2 pair_1_2 = make_float2(__uint_as_float(v4_1_3[1] << 16), __uint_as_float(v4_1_3[1] & 4294901760u));
                float2 g_1_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_2) : "l"(*(const unsigned long long*)&pair_1_2), "l"(*(const unsigned long long*)&mm_2));
                uint32_t fp4_pair_1_2;
                uint16_t _e2m1_pair_h_25;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_1_2.x), "f"(g_1_2.y));
                fp4_pair_1_2 = (uint32_t)_e2m1_pair_h_25;
                float2 pair_2_2 = make_float2(__uint_as_float(v4_1_3[2] << 16), __uint_as_float(v4_1_3[2] & 4294901760u));
                float2 g_2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_2) : "l"(*(const unsigned long long*)&pair_2_2), "l"(*(const unsigned long long*)&mm_2));
                uint32_t fp4_pair_2_2;
                uint16_t _e2m1_pair_h_26;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(g_2_2.x), "f"(g_2_2.y));
                fp4_pair_2_2 = (uint32_t)_e2m1_pair_h_26;
                float2 pair_3_2 = make_float2(__uint_as_float(v4_1_3[3] << 16), __uint_as_float(v4_1_3[3] & 4294901760u));
                float2 g_3_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_2) : "l"(*(const unsigned long long*)&pair_3_2), "l"(*(const unsigned long long*)&mm_2));
                uint32_t _fp4_e2m1_pack_pair_27;
                uint16_t _e2m1_pair_h_28;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_28) : "f"(g_3_2.x), "f"(g_3_2.y));
                _fp4_e2m1_pack_pair_27 = (uint32_t)_e2m1_pair_h_28;
                unsigned int word_2 = fp4_pair_4 | fp4_pair_1_2 << 8 | fp4_pair_2_2 << 16 | _fp4_e2m1_pack_pair_27 << 24;
                unsigned int carry_2 = (word_2 & 2004318071) + 2004318071;
                w_1_0[3] = word_2 & (carry_2 | 2004318071);
                bool fodd_1_0 = (f & 1) != 0;
                bool fhi_1_0 = (f & 2) != 0;
                unsigned int t0_1_0 = w_1_0[0];
                unsigned int t1_1_0 = w_1_0[1];
                unsigned int t2_1_0 = w_1_0[2];
                unsigned int t3_1_0 = w_1_0[3];
                if (fodd_1_0) {
                    t0_1_0 = w_1_0[1];
                    t1_1_0 = w_1_0[0];
                    t2_1_0 = w_1_0[3];
                    t3_1_0 = w_1_0[2];
                }
                unsigned int q0_1_0 = t0_1_0;
                unsigned int q1_1_0 = t1_1_0;
                unsigned int q2_1_0 = t2_1_0;
                unsigned int q3_1_0 = t3_1_0;
                if (fhi_1_0) {
                    q0_1_0 = t2_1_0;
                    q1_1_0 = t3_1_0;
                    q2_1_0 = t0_1_0;
                    q3_1_0 = t1_1_0;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_0)), "r"(q0_1_0), "r"(q1_1_0), "r"(q2_1_0), "r"(q3_1_0) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_0)), "h"((uint16_t)(e4m3x2_f32 & 32639)) : "memory");
                int pp2_1_1 = lane_1 + 32;
                int base2_1_1 = xb_off + 64 * pp2_1_1;
                float bm0_1_1 = __uint_as_float(bmp[1] << 16);
                float bm1_1_1 = __uint_as_float(bmp[1] & 4294901760u);
                float _fdiv_29 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_1, inv), 6.0f);
                float _fdiv_30 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_1, inv), 6.0f);
                uint16_t e4m3x2_f32_0;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_0) : "f"(_fdiv_29), "f"(_fdiv_30));
                uint32_t e4m3x2_to_f16x2_0;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_0) : "h"((uint16_t)(e4m3x2_f32_0)));
                float cvt_f32_f16_0;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0)));
                float cvt_f32_f16_1_0;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0 >> 16)));
                float fdiv_rn_2 = __fdiv_rn(inv, cvt_f32_f16_0);
                float fdiv_rn_1_0 = __fdiv_rn(inv, cvt_f32_f16_1_0);
                float m0_0 = 0.0f;
                float m1_0 = 0.0f;
                if (cvt_f32_f16_0 > 0.0f) {
                    m0_0 = fdiv_rn_2;
                }
                if (cvt_f32_f16_1_0 > 0.0f) {
                    m1_0 = fdiv_rn_1_0;
                }
                unsigned int w_1_1[4];
                int cf_1_4 = f ^ 0;
                unsigned int v4_1_4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_4)));
                float m_1_4 = m0_0;
                if ((cf_1_4 & 2) != 0) {
                    m_1_4 = m1_0;
                }
                float2 mm_3 = make_float2(m_1_4, m_1_4);
                float2 pair_6 = make_float2(__uint_as_float(v4_1_4[0] << 16), __uint_as_float(v4_1_4[0] & 4294901760u));
                float2 g_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_6) : "l"(*(const unsigned long long*)&pair_6), "l"(*(const unsigned long long*)&mm_3));
                uint32_t fp4_pair_5;
                uint16_t _e2m1_pair_h_31;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_31) : "f"(g_6.x), "f"(g_6.y));
                fp4_pair_5 = (uint32_t)_e2m1_pair_h_31;
                float2 pair_1_3 = make_float2(__uint_as_float(v4_1_4[1] << 16), __uint_as_float(v4_1_4[1] & 4294901760u));
                float2 g_1_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_3) : "l"(*(const unsigned long long*)&pair_1_3), "l"(*(const unsigned long long*)&mm_3));
                uint32_t fp4_pair_1_3;
                uint16_t _e2m1_pair_h_32;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(g_1_3.x), "f"(g_1_3.y));
                fp4_pair_1_3 = (uint32_t)_e2m1_pair_h_32;
                float2 pair_2_3 = make_float2(__uint_as_float(v4_1_4[2] << 16), __uint_as_float(v4_1_4[2] & 4294901760u));
                float2 g_2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_3) : "l"(*(const unsigned long long*)&pair_2_3), "l"(*(const unsigned long long*)&mm_3));
                uint32_t fp4_pair_2_3;
                uint16_t _e2m1_pair_h_33;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(g_2_3.x), "f"(g_2_3.y));
                fp4_pair_2_3 = (uint32_t)_e2m1_pair_h_33;
                float2 pair_3_3 = make_float2(__uint_as_float(v4_1_4[3] << 16), __uint_as_float(v4_1_4[3] & 4294901760u));
                float2 g_3_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_3) : "l"(*(const unsigned long long*)&pair_3_3), "l"(*(const unsigned long long*)&mm_3));
                uint32_t _fp4_e2m1_pack_pair_34;
                uint16_t _e2m1_pair_h_35;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_35) : "f"(g_3_3.x), "f"(g_3_3.y));
                _fp4_e2m1_pack_pair_34 = (uint32_t)_e2m1_pair_h_35;
                unsigned int word_3 = fp4_pair_5 | fp4_pair_1_3 << 8 | fp4_pair_2_3 << 16 | _fp4_e2m1_pack_pair_34 << 24;
                unsigned int carry_3 = (word_3 & 2004318071) + 2004318071;
                w_1_1[0] = word_3 & (carry_3 | 2004318071);
                int cf_1_5 = f ^ 1;
                unsigned int v4_1_5[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_5)));
                float m_1_5 = m0_0;
                if ((cf_1_5 & 2) != 0) {
                    m_1_5 = m1_0;
                }
                float2 mm_4 = make_float2(m_1_5, m_1_5);
                float2 pair_7 = make_float2(__uint_as_float(v4_1_5[0] << 16), __uint_as_float(v4_1_5[0] & 4294901760u));
                float2 g_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_7) : "l"(*(const unsigned long long*)&pair_7), "l"(*(const unsigned long long*)&mm_4));
                uint32_t fp4_pair_6;
                uint16_t _e2m1_pair_h_36;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_36) : "f"(g_7.x), "f"(g_7.y));
                fp4_pair_6 = (uint32_t)_e2m1_pair_h_36;
                float2 pair_1_4 = make_float2(__uint_as_float(v4_1_5[1] << 16), __uint_as_float(v4_1_5[1] & 4294901760u));
                float2 g_1_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_4) : "l"(*(const unsigned long long*)&pair_1_4), "l"(*(const unsigned long long*)&mm_4));
                uint32_t fp4_pair_1_4;
                uint16_t _e2m1_pair_h_37;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(g_1_4.x), "f"(g_1_4.y));
                fp4_pair_1_4 = (uint32_t)_e2m1_pair_h_37;
                float2 pair_2_4 = make_float2(__uint_as_float(v4_1_5[2] << 16), __uint_as_float(v4_1_5[2] & 4294901760u));
                float2 g_2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_4) : "l"(*(const unsigned long long*)&pair_2_4), "l"(*(const unsigned long long*)&mm_4));
                uint32_t fp4_pair_2_4;
                uint16_t _e2m1_pair_h_38;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(g_2_4.x), "f"(g_2_4.y));
                fp4_pair_2_4 = (uint32_t)_e2m1_pair_h_38;
                float2 pair_3_4 = make_float2(__uint_as_float(v4_1_5[3] << 16), __uint_as_float(v4_1_5[3] & 4294901760u));
                float2 g_3_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_4) : "l"(*(const unsigned long long*)&pair_3_4), "l"(*(const unsigned long long*)&mm_4));
                uint32_t _fp4_e2m1_pack_pair_39;
                uint16_t _e2m1_pair_h_40;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_40) : "f"(g_3_4.x), "f"(g_3_4.y));
                _fp4_e2m1_pack_pair_39 = (uint32_t)_e2m1_pair_h_40;
                unsigned int word_4 = fp4_pair_6 | fp4_pair_1_4 << 8 | fp4_pair_2_4 << 16 | _fp4_e2m1_pack_pair_39 << 24;
                unsigned int carry_4 = (word_4 & 2004318071) + 2004318071;
                w_1_1[1] = word_4 & (carry_4 | 2004318071);
                int cf_1_6 = f ^ 2;
                unsigned int v4_1_6[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_6)));
                float m_1_6 = m0_0;
                if ((cf_1_6 & 2) != 0) {
                    m_1_6 = m1_0;
                }
                float2 mm_5 = make_float2(m_1_6, m_1_6);
                float2 pair_8 = make_float2(__uint_as_float(v4_1_6[0] << 16), __uint_as_float(v4_1_6[0] & 4294901760u));
                float2 g_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_8) : "l"(*(const unsigned long long*)&pair_8), "l"(*(const unsigned long long*)&mm_5));
                uint32_t fp4_pair_7;
                uint16_t _e2m1_pair_h_41;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(g_8.x), "f"(g_8.y));
                fp4_pair_7 = (uint32_t)_e2m1_pair_h_41;
                float2 pair_1_5 = make_float2(__uint_as_float(v4_1_6[1] << 16), __uint_as_float(v4_1_6[1] & 4294901760u));
                float2 g_1_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_5) : "l"(*(const unsigned long long*)&pair_1_5), "l"(*(const unsigned long long*)&mm_5));
                uint32_t fp4_pair_1_5;
                uint16_t _e2m1_pair_h_42;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(g_1_5.x), "f"(g_1_5.y));
                fp4_pair_1_5 = (uint32_t)_e2m1_pair_h_42;
                float2 pair_2_5 = make_float2(__uint_as_float(v4_1_6[2] << 16), __uint_as_float(v4_1_6[2] & 4294901760u));
                float2 g_2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_5) : "l"(*(const unsigned long long*)&pair_2_5), "l"(*(const unsigned long long*)&mm_5));
                uint32_t fp4_pair_2_5;
                uint16_t _e2m1_pair_h_43;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(g_2_5.x), "f"(g_2_5.y));
                fp4_pair_2_5 = (uint32_t)_e2m1_pair_h_43;
                float2 pair_3_5 = make_float2(__uint_as_float(v4_1_6[3] << 16), __uint_as_float(v4_1_6[3] & 4294901760u));
                float2 g_3_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_5) : "l"(*(const unsigned long long*)&pair_3_5), "l"(*(const unsigned long long*)&mm_5));
                uint32_t _fp4_e2m1_pack_pair_44;
                uint16_t _e2m1_pair_h_45;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(g_3_5.x), "f"(g_3_5.y));
                _fp4_e2m1_pack_pair_44 = (uint32_t)_e2m1_pair_h_45;
                unsigned int word_5 = fp4_pair_7 | fp4_pair_1_5 << 8 | fp4_pair_2_5 << 16 | _fp4_e2m1_pack_pair_44 << 24;
                unsigned int carry_5 = (word_5 & 2004318071) + 2004318071;
                w_1_1[2] = word_5 & (carry_5 | 2004318071);
                int cf_1_7 = f ^ 3;
                unsigned int v4_1_7[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_7)));
                float m_1_7 = m0_0;
                if ((cf_1_7 & 2) != 0) {
                    m_1_7 = m1_0;
                }
                float2 mm_6 = make_float2(m_1_7, m_1_7);
                float2 pair_9 = make_float2(__uint_as_float(v4_1_7[0] << 16), __uint_as_float(v4_1_7[0] & 4294901760u));
                float2 g_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_9) : "l"(*(const unsigned long long*)&pair_9), "l"(*(const unsigned long long*)&mm_6));
                uint32_t fp4_pair_8;
                uint16_t _e2m1_pair_h_46;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_46) : "f"(g_9.x), "f"(g_9.y));
                fp4_pair_8 = (uint32_t)_e2m1_pair_h_46;
                float2 pair_1_6 = make_float2(__uint_as_float(v4_1_7[1] << 16), __uint_as_float(v4_1_7[1] & 4294901760u));
                float2 g_1_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_6) : "l"(*(const unsigned long long*)&pair_1_6), "l"(*(const unsigned long long*)&mm_6));
                uint32_t fp4_pair_1_6;
                uint16_t _e2m1_pair_h_47;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(g_1_6.x), "f"(g_1_6.y));
                fp4_pair_1_6 = (uint32_t)_e2m1_pair_h_47;
                float2 pair_2_6 = make_float2(__uint_as_float(v4_1_7[2] << 16), __uint_as_float(v4_1_7[2] & 4294901760u));
                float2 g_2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_6) : "l"(*(const unsigned long long*)&pair_2_6), "l"(*(const unsigned long long*)&mm_6));
                uint32_t fp4_pair_2_6;
                uint16_t _e2m1_pair_h_48;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_48) : "f"(g_2_6.x), "f"(g_2_6.y));
                fp4_pair_2_6 = (uint32_t)_e2m1_pair_h_48;
                float2 pair_3_6 = make_float2(__uint_as_float(v4_1_7[3] << 16), __uint_as_float(v4_1_7[3] & 4294901760u));
                float2 g_3_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_6) : "l"(*(const unsigned long long*)&pair_3_6), "l"(*(const unsigned long long*)&mm_6));
                uint32_t _fp4_e2m1_pack_pair_49;
                uint16_t _e2m1_pair_h_50;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_50) : "f"(g_3_6.x), "f"(g_3_6.y));
                _fp4_e2m1_pack_pair_49 = (uint32_t)_e2m1_pair_h_50;
                unsigned int word_6 = fp4_pair_8 | fp4_pair_1_6 << 8 | fp4_pair_2_6 << 16 | _fp4_e2m1_pack_pair_49 << 24;
                unsigned int carry_6 = (word_6 & 2004318071) + 2004318071;
                w_1_1[3] = word_6 & (carry_6 | 2004318071);
                bool fodd_1_1 = (f & 1) != 0;
                bool fhi_1_1 = (f & 2) != 0;
                unsigned int t0_1_1 = w_1_1[0];
                unsigned int t1_1_1 = w_1_1[1];
                unsigned int t2_1_1 = w_1_1[2];
                unsigned int t3_1_1 = w_1_1[3];
                if (fodd_1_1) {
                    t0_1_1 = w_1_1[1];
                    t1_1_1 = w_1_1[0];
                    t2_1_1 = w_1_1[3];
                    t3_1_1 = w_1_1[2];
                }
                unsigned int q0_1_1 = t0_1_1;
                unsigned int q1_1_1 = t1_1_1;
                unsigned int q2_1_1 = t2_1_1;
                unsigned int q3_1_1 = t3_1_1;
                if (fhi_1_1) {
                    q0_1_1 = t2_1_1;
                    q1_1_1 = t3_1_1;
                    q2_1_1 = t0_1_1;
                    q3_1_1 = t1_1_1;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_1)), "r"(q0_1_1), "r"(q1_1_1), "r"(q2_1_1), "r"(q3_1_1) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_1)), "h"((uint16_t)(e4m3x2_f32_0 & 32639)) : "memory");
                int pp2_1_2 = lane_1 + 64;
                int base2_1_2 = xb_off + 64 * pp2_1_2;
                float bm0_1_2 = __uint_as_float(bmp[2] << 16);
                float bm1_1_2 = __uint_as_float(bmp[2] & 4294901760u);
                float _fdiv_51 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_2, inv), 6.0f);
                float _fdiv_52 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_2, inv), 6.0f);
                uint16_t e4m3x2_f32_1;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_1) : "f"(_fdiv_51), "f"(_fdiv_52));
                uint32_t e4m3x2_to_f16x2_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_1) : "h"((uint16_t)(e4m3x2_f32_1)));
                float cvt_f32_f16_2;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_2) : "h"((uint16_t)(e4m3x2_to_f16x2_1)));
                float cvt_f32_f16_1_1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_1) : "h"((uint16_t)(e4m3x2_to_f16x2_1 >> 16)));
                float fdiv_rn_3 = __fdiv_rn(inv, cvt_f32_f16_2);
                float fdiv_rn_1_1 = __fdiv_rn(inv, cvt_f32_f16_1_1);
                float m0_1 = 0.0f;
                float m1_1 = 0.0f;
                if (cvt_f32_f16_2 > 0.0f) {
                    m0_1 = fdiv_rn_3;
                }
                if (cvt_f32_f16_1_1 > 0.0f) {
                    m1_1 = fdiv_rn_1_1;
                }
                unsigned int w_1_2[4];
                int cf_1_8 = f ^ 0;
                unsigned int v4_1_8[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_8)));
                float m_1_8 = m0_1;
                if ((cf_1_8 & 2) != 0) {
                    m_1_8 = m1_1;
                }
                float2 mm_7 = make_float2(m_1_8, m_1_8);
                float2 pair_10 = make_float2(__uint_as_float(v4_1_8[0] << 16), __uint_as_float(v4_1_8[0] & 4294901760u));
                float2 g_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_10) : "l"(*(const unsigned long long*)&pair_10), "l"(*(const unsigned long long*)&mm_7));
                uint32_t fp4_pair_9;
                uint16_t _e2m1_pair_h_53;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_53) : "f"(g_10.x), "f"(g_10.y));
                fp4_pair_9 = (uint32_t)_e2m1_pair_h_53;
                float2 pair_1_7 = make_float2(__uint_as_float(v4_1_8[1] << 16), __uint_as_float(v4_1_8[1] & 4294901760u));
                float2 g_1_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_7) : "l"(*(const unsigned long long*)&pair_1_7), "l"(*(const unsigned long long*)&mm_7));
                uint32_t fp4_pair_1_7;
                uint16_t _e2m1_pair_h_54;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_54) : "f"(g_1_7.x), "f"(g_1_7.y));
                fp4_pair_1_7 = (uint32_t)_e2m1_pair_h_54;
                float2 pair_2_7 = make_float2(__uint_as_float(v4_1_8[2] << 16), __uint_as_float(v4_1_8[2] & 4294901760u));
                float2 g_2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_7) : "l"(*(const unsigned long long*)&pair_2_7), "l"(*(const unsigned long long*)&mm_7));
                uint32_t fp4_pair_2_7;
                uint16_t _e2m1_pair_h_55;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(g_2_7.x), "f"(g_2_7.y));
                fp4_pair_2_7 = (uint32_t)_e2m1_pair_h_55;
                float2 pair_3_7 = make_float2(__uint_as_float(v4_1_8[3] << 16), __uint_as_float(v4_1_8[3] & 4294901760u));
                float2 g_3_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_7) : "l"(*(const unsigned long long*)&pair_3_7), "l"(*(const unsigned long long*)&mm_7));
                uint32_t _fp4_e2m1_pack_pair_56;
                uint16_t _e2m1_pair_h_57;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(g_3_7.x), "f"(g_3_7.y));
                _fp4_e2m1_pack_pair_56 = (uint32_t)_e2m1_pair_h_57;
                unsigned int word_7 = fp4_pair_9 | fp4_pair_1_7 << 8 | fp4_pair_2_7 << 16 | _fp4_e2m1_pack_pair_56 << 24;
                unsigned int carry_7 = (word_7 & 2004318071) + 2004318071;
                w_1_2[0] = word_7 & (carry_7 | 2004318071);
                int cf_1_9 = f ^ 1;
                unsigned int v4_1_9[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_9)));
                float m_1_9 = m0_1;
                if ((cf_1_9 & 2) != 0) {
                    m_1_9 = m1_1;
                }
                float2 mm_8 = make_float2(m_1_9, m_1_9);
                float2 pair_11 = make_float2(__uint_as_float(v4_1_9[0] << 16), __uint_as_float(v4_1_9[0] & 4294901760u));
                float2 g_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_11) : "l"(*(const unsigned long long*)&pair_11), "l"(*(const unsigned long long*)&mm_8));
                uint32_t fp4_pair_10;
                uint16_t _e2m1_pair_h_58;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_58) : "f"(g_11.x), "f"(g_11.y));
                fp4_pair_10 = (uint32_t)_e2m1_pair_h_58;
                float2 pair_1_8 = make_float2(__uint_as_float(v4_1_9[1] << 16), __uint_as_float(v4_1_9[1] & 4294901760u));
                float2 g_1_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_8) : "l"(*(const unsigned long long*)&pair_1_8), "l"(*(const unsigned long long*)&mm_8));
                uint32_t fp4_pair_1_8;
                uint16_t _e2m1_pair_h_59;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(g_1_8.x), "f"(g_1_8.y));
                fp4_pair_1_8 = (uint32_t)_e2m1_pair_h_59;
                float2 pair_2_8 = make_float2(__uint_as_float(v4_1_9[2] << 16), __uint_as_float(v4_1_9[2] & 4294901760u));
                float2 g_2_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_8) : "l"(*(const unsigned long long*)&pair_2_8), "l"(*(const unsigned long long*)&mm_8));
                uint32_t fp4_pair_2_8;
                uint16_t _e2m1_pair_h_60;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_60) : "f"(g_2_8.x), "f"(g_2_8.y));
                fp4_pair_2_8 = (uint32_t)_e2m1_pair_h_60;
                float2 pair_3_8 = make_float2(__uint_as_float(v4_1_9[3] << 16), __uint_as_float(v4_1_9[3] & 4294901760u));
                float2 g_3_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_8) : "l"(*(const unsigned long long*)&pair_3_8), "l"(*(const unsigned long long*)&mm_8));
                uint32_t _fp4_e2m1_pack_pair_61;
                uint16_t _e2m1_pair_h_62;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_62) : "f"(g_3_8.x), "f"(g_3_8.y));
                _fp4_e2m1_pack_pair_61 = (uint32_t)_e2m1_pair_h_62;
                unsigned int word_8 = fp4_pair_10 | fp4_pair_1_8 << 8 | fp4_pair_2_8 << 16 | _fp4_e2m1_pack_pair_61 << 24;
                unsigned int carry_8 = (word_8 & 2004318071) + 2004318071;
                w_1_2[1] = word_8 & (carry_8 | 2004318071);
                int cf_1_10 = f ^ 2;
                unsigned int v4_1_10[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_10)));
                float m_1_10 = m0_1;
                if ((cf_1_10 & 2) != 0) {
                    m_1_10 = m1_1;
                }
                float2 mm_9 = make_float2(m_1_10, m_1_10);
                float2 pair_12 = make_float2(__uint_as_float(v4_1_10[0] << 16), __uint_as_float(v4_1_10[0] & 4294901760u));
                float2 g_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_12) : "l"(*(const unsigned long long*)&pair_12), "l"(*(const unsigned long long*)&mm_9));
                uint32_t fp4_pair_11;
                uint16_t _e2m1_pair_h_63;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_63) : "f"(g_12.x), "f"(g_12.y));
                fp4_pair_11 = (uint32_t)_e2m1_pair_h_63;
                float2 pair_1_9 = make_float2(__uint_as_float(v4_1_10[1] << 16), __uint_as_float(v4_1_10[1] & 4294901760u));
                float2 g_1_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_9) : "l"(*(const unsigned long long*)&pair_1_9), "l"(*(const unsigned long long*)&mm_9));
                uint32_t fp4_pair_1_9;
                uint16_t _e2m1_pair_h_64;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_64) : "f"(g_1_9.x), "f"(g_1_9.y));
                fp4_pair_1_9 = (uint32_t)_e2m1_pair_h_64;
                float2 pair_2_9 = make_float2(__uint_as_float(v4_1_10[2] << 16), __uint_as_float(v4_1_10[2] & 4294901760u));
                float2 g_2_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_9) : "l"(*(const unsigned long long*)&pair_2_9), "l"(*(const unsigned long long*)&mm_9));
                uint32_t fp4_pair_2_9;
                uint16_t _e2m1_pair_h_65;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_65) : "f"(g_2_9.x), "f"(g_2_9.y));
                fp4_pair_2_9 = (uint32_t)_e2m1_pair_h_65;
                float2 pair_3_9 = make_float2(__uint_as_float(v4_1_10[3] << 16), __uint_as_float(v4_1_10[3] & 4294901760u));
                float2 g_3_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_9) : "l"(*(const unsigned long long*)&pair_3_9), "l"(*(const unsigned long long*)&mm_9));
                uint32_t _fp4_e2m1_pack_pair_66;
                uint16_t _e2m1_pair_h_67;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_67) : "f"(g_3_9.x), "f"(g_3_9.y));
                _fp4_e2m1_pack_pair_66 = (uint32_t)_e2m1_pair_h_67;
                unsigned int word_9 = fp4_pair_11 | fp4_pair_1_9 << 8 | fp4_pair_2_9 << 16 | _fp4_e2m1_pack_pair_66 << 24;
                unsigned int carry_9 = (word_9 & 2004318071) + 2004318071;
                w_1_2[2] = word_9 & (carry_9 | 2004318071);
                int cf_1_11 = f ^ 3;
                unsigned int v4_1_11[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_11)));
                float m_1_11 = m0_1;
                if ((cf_1_11 & 2) != 0) {
                    m_1_11 = m1_1;
                }
                float2 mm_10 = make_float2(m_1_11, m_1_11);
                float2 pair_13 = make_float2(__uint_as_float(v4_1_11[0] << 16), __uint_as_float(v4_1_11[0] & 4294901760u));
                float2 g_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_13) : "l"(*(const unsigned long long*)&pair_13), "l"(*(const unsigned long long*)&mm_10));
                uint32_t fp4_pair_12;
                uint16_t _e2m1_pair_h_68;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_68) : "f"(g_13.x), "f"(g_13.y));
                fp4_pair_12 = (uint32_t)_e2m1_pair_h_68;
                float2 pair_1_10 = make_float2(__uint_as_float(v4_1_11[1] << 16), __uint_as_float(v4_1_11[1] & 4294901760u));
                float2 g_1_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_10) : "l"(*(const unsigned long long*)&pair_1_10), "l"(*(const unsigned long long*)&mm_10));
                uint32_t fp4_pair_1_10;
                uint16_t _e2m1_pair_h_69;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_69) : "f"(g_1_10.x), "f"(g_1_10.y));
                fp4_pair_1_10 = (uint32_t)_e2m1_pair_h_69;
                float2 pair_2_10 = make_float2(__uint_as_float(v4_1_11[2] << 16), __uint_as_float(v4_1_11[2] & 4294901760u));
                float2 g_2_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_10) : "l"(*(const unsigned long long*)&pair_2_10), "l"(*(const unsigned long long*)&mm_10));
                uint32_t fp4_pair_2_10;
                uint16_t _e2m1_pair_h_70;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_70) : "f"(g_2_10.x), "f"(g_2_10.y));
                fp4_pair_2_10 = (uint32_t)_e2m1_pair_h_70;
                float2 pair_3_10 = make_float2(__uint_as_float(v4_1_11[3] << 16), __uint_as_float(v4_1_11[3] & 4294901760u));
                float2 g_3_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_10) : "l"(*(const unsigned long long*)&pair_3_10), "l"(*(const unsigned long long*)&mm_10));
                uint32_t _fp4_e2m1_pack_pair_71;
                uint16_t _e2m1_pair_h_72;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_72) : "f"(g_3_10.x), "f"(g_3_10.y));
                _fp4_e2m1_pack_pair_71 = (uint32_t)_e2m1_pair_h_72;
                unsigned int word_10 = fp4_pair_12 | fp4_pair_1_10 << 8 | fp4_pair_2_10 << 16 | _fp4_e2m1_pack_pair_71 << 24;
                unsigned int carry_10 = (word_10 & 2004318071) + 2004318071;
                w_1_2[3] = word_10 & (carry_10 | 2004318071);
                bool fodd_1_2 = (f & 1) != 0;
                bool fhi_1_2 = (f & 2) != 0;
                unsigned int t0_1_2 = w_1_2[0];
                unsigned int t1_1_2 = w_1_2[1];
                unsigned int t2_1_2 = w_1_2[2];
                unsigned int t3_1_2 = w_1_2[3];
                if (fodd_1_2) {
                    t0_1_2 = w_1_2[1];
                    t1_1_2 = w_1_2[0];
                    t2_1_2 = w_1_2[3];
                    t3_1_2 = w_1_2[2];
                }
                unsigned int q0_1_2 = t0_1_2;
                unsigned int q1_1_2 = t1_1_2;
                unsigned int q2_1_2 = t2_1_2;
                unsigned int q3_1_2 = t3_1_2;
                if (fhi_1_2) {
                    q0_1_2 = t2_1_2;
                    q1_1_2 = t3_1_2;
                    q2_1_2 = t0_1_2;
                    q3_1_2 = t1_1_2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_2)), "r"(q0_1_2), "r"(q1_1_2), "r"(q2_1_2), "r"(q3_1_2) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_2)), "h"((uint16_t)(e4m3x2_f32_1 & 32639)) : "memory");
                int pp2_1_3 = lane_1 + 96;
                int base2_1_3 = xb_off + 64 * pp2_1_3;
                float bm0_1_3 = __uint_as_float(bmp[3] << 16);
                float bm1_1_3 = __uint_as_float(bmp[3] & 4294901760u);
                float _fdiv_73 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_3, inv), 6.0f);
                float _fdiv_74 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_3, inv), 6.0f);
                uint16_t e4m3x2_f32_2;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_2) : "f"(_fdiv_73), "f"(_fdiv_74));
                uint32_t e4m3x2_to_f16x2_2;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_2) : "h"((uint16_t)(e4m3x2_f32_2)));
                float cvt_f32_f16_3;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_3) : "h"((uint16_t)(e4m3x2_to_f16x2_2)));
                float cvt_f32_f16_1_2;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_2) : "h"((uint16_t)(e4m3x2_to_f16x2_2 >> 16)));
                float fdiv_rn_4 = __fdiv_rn(inv, cvt_f32_f16_3);
                float fdiv_rn_1_2 = __fdiv_rn(inv, cvt_f32_f16_1_2);
                float m0_2 = 0.0f;
                float m1_2 = 0.0f;
                if (cvt_f32_f16_3 > 0.0f) {
                    m0_2 = fdiv_rn_4;
                }
                if (cvt_f32_f16_1_2 > 0.0f) {
                    m1_2 = fdiv_rn_1_2;
                }
                unsigned int w_1_3[4];
                int cf_1_12 = f ^ 0;
                unsigned int v4_1_12[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_12[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_3 + 16 * cf_1_12)));
                float m_1_12 = m0_2;
                if ((cf_1_12 & 2) != 0) {
                    m_1_12 = m1_2;
                }
                float2 mm_11 = make_float2(m_1_12, m_1_12);
                float2 pair_14 = make_float2(__uint_as_float(v4_1_12[0] << 16), __uint_as_float(v4_1_12[0] & 4294901760u));
                float2 g_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_14) : "l"(*(const unsigned long long*)&pair_14), "l"(*(const unsigned long long*)&mm_11));
                uint32_t fp4_pair_13;
                uint16_t _e2m1_pair_h_75;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_75) : "f"(g_14.x), "f"(g_14.y));
                fp4_pair_13 = (uint32_t)_e2m1_pair_h_75;
                float2 pair_1_11 = make_float2(__uint_as_float(v4_1_12[1] << 16), __uint_as_float(v4_1_12[1] & 4294901760u));
                float2 g_1_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_11) : "l"(*(const unsigned long long*)&pair_1_11), "l"(*(const unsigned long long*)&mm_11));
                uint32_t fp4_pair_1_11;
                uint16_t _e2m1_pair_h_76;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_76) : "f"(g_1_11.x), "f"(g_1_11.y));
                fp4_pair_1_11 = (uint32_t)_e2m1_pair_h_76;
                float2 pair_2_11 = make_float2(__uint_as_float(v4_1_12[2] << 16), __uint_as_float(v4_1_12[2] & 4294901760u));
                float2 g_2_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_11) : "l"(*(const unsigned long long*)&pair_2_11), "l"(*(const unsigned long long*)&mm_11));
                uint32_t fp4_pair_2_11;
                uint16_t _e2m1_pair_h_77;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_77) : "f"(g_2_11.x), "f"(g_2_11.y));
                fp4_pair_2_11 = (uint32_t)_e2m1_pair_h_77;
                float2 pair_3_11 = make_float2(__uint_as_float(v4_1_12[3] << 16), __uint_as_float(v4_1_12[3] & 4294901760u));
                float2 g_3_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_11) : "l"(*(const unsigned long long*)&pair_3_11), "l"(*(const unsigned long long*)&mm_11));
                uint32_t _fp4_e2m1_pack_pair_78;
                uint16_t _e2m1_pair_h_79;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_79) : "f"(g_3_11.x), "f"(g_3_11.y));
                _fp4_e2m1_pack_pair_78 = (uint32_t)_e2m1_pair_h_79;
                unsigned int word_11 = fp4_pair_13 | fp4_pair_1_11 << 8 | fp4_pair_2_11 << 16 | _fp4_e2m1_pack_pair_78 << 24;
                unsigned int carry_11 = (word_11 & 2004318071) + 2004318071;
                w_1_3[0] = word_11 & (carry_11 | 2004318071);
                int cf_1_13 = f ^ 1;
                unsigned int v4_1_13[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_13[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_3 + 16 * cf_1_13)));
                float m_1_13 = m0_2;
                if ((cf_1_13 & 2) != 0) {
                    m_1_13 = m1_2;
                }
                float2 mm_12 = make_float2(m_1_13, m_1_13);
                float2 pair_15 = make_float2(__uint_as_float(v4_1_13[0] << 16), __uint_as_float(v4_1_13[0] & 4294901760u));
                float2 g_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_15) : "l"(*(const unsigned long long*)&pair_15), "l"(*(const unsigned long long*)&mm_12));
                uint32_t fp4_pair_14;
                uint16_t _e2m1_pair_h_80;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_80) : "f"(g_15.x), "f"(g_15.y));
                fp4_pair_14 = (uint32_t)_e2m1_pair_h_80;
                float2 pair_1_12 = make_float2(__uint_as_float(v4_1_13[1] << 16), __uint_as_float(v4_1_13[1] & 4294901760u));
                float2 g_1_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_12) : "l"(*(const unsigned long long*)&pair_1_12), "l"(*(const unsigned long long*)&mm_12));
                uint32_t fp4_pair_1_12;
                uint16_t _e2m1_pair_h_81;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_81) : "f"(g_1_12.x), "f"(g_1_12.y));
                fp4_pair_1_12 = (uint32_t)_e2m1_pair_h_81;
                float2 pair_2_12 = make_float2(__uint_as_float(v4_1_13[2] << 16), __uint_as_float(v4_1_13[2] & 4294901760u));
                float2 g_2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_12) : "l"(*(const unsigned long long*)&pair_2_12), "l"(*(const unsigned long long*)&mm_12));
                uint32_t fp4_pair_2_12;
                uint16_t _e2m1_pair_h_82;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_82) : "f"(g_2_12.x), "f"(g_2_12.y));
                fp4_pair_2_12 = (uint32_t)_e2m1_pair_h_82;
                float2 pair_3_12 = make_float2(__uint_as_float(v4_1_13[3] << 16), __uint_as_float(v4_1_13[3] & 4294901760u));
                float2 g_3_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_12) : "l"(*(const unsigned long long*)&pair_3_12), "l"(*(const unsigned long long*)&mm_12));
                uint32_t _fp4_e2m1_pack_pair_83;
                uint16_t _e2m1_pair_h_84;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_84) : "f"(g_3_12.x), "f"(g_3_12.y));
                _fp4_e2m1_pack_pair_83 = (uint32_t)_e2m1_pair_h_84;
                unsigned int word_12 = fp4_pair_14 | fp4_pair_1_12 << 8 | fp4_pair_2_12 << 16 | _fp4_e2m1_pack_pair_83 << 24;
                unsigned int carry_12 = (word_12 & 2004318071) + 2004318071;
                w_1_3[1] = word_12 & (carry_12 | 2004318071);
                int cf_1_14 = f ^ 2;
                unsigned int v4_1_14[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_14[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_3 + 16 * cf_1_14)));
                float m_1_14 = m0_2;
                if ((cf_1_14 & 2) != 0) {
                    m_1_14 = m1_2;
                }
                float2 mm_13 = make_float2(m_1_14, m_1_14);
                float2 pair_16 = make_float2(__uint_as_float(v4_1_14[0] << 16), __uint_as_float(v4_1_14[0] & 4294901760u));
                float2 g_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_16) : "l"(*(const unsigned long long*)&pair_16), "l"(*(const unsigned long long*)&mm_13));
                uint32_t fp4_pair_15;
                uint16_t _e2m1_pair_h_85;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_85) : "f"(g_16.x), "f"(g_16.y));
                fp4_pair_15 = (uint32_t)_e2m1_pair_h_85;
                float2 pair_1_13 = make_float2(__uint_as_float(v4_1_14[1] << 16), __uint_as_float(v4_1_14[1] & 4294901760u));
                float2 g_1_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_13) : "l"(*(const unsigned long long*)&pair_1_13), "l"(*(const unsigned long long*)&mm_13));
                uint32_t fp4_pair_1_13;
                uint16_t _e2m1_pair_h_86;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_86) : "f"(g_1_13.x), "f"(g_1_13.y));
                fp4_pair_1_13 = (uint32_t)_e2m1_pair_h_86;
                float2 pair_2_13 = make_float2(__uint_as_float(v4_1_14[2] << 16), __uint_as_float(v4_1_14[2] & 4294901760u));
                float2 g_2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_13) : "l"(*(const unsigned long long*)&pair_2_13), "l"(*(const unsigned long long*)&mm_13));
                uint32_t fp4_pair_2_13;
                uint16_t _e2m1_pair_h_87;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_87) : "f"(g_2_13.x), "f"(g_2_13.y));
                fp4_pair_2_13 = (uint32_t)_e2m1_pair_h_87;
                float2 pair_3_13 = make_float2(__uint_as_float(v4_1_14[3] << 16), __uint_as_float(v4_1_14[3] & 4294901760u));
                float2 g_3_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_13) : "l"(*(const unsigned long long*)&pair_3_13), "l"(*(const unsigned long long*)&mm_13));
                uint32_t _fp4_e2m1_pack_pair_88;
                uint16_t _e2m1_pair_h_89;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_89) : "f"(g_3_13.x), "f"(g_3_13.y));
                _fp4_e2m1_pack_pair_88 = (uint32_t)_e2m1_pair_h_89;
                unsigned int word_13 = fp4_pair_15 | fp4_pair_1_13 << 8 | fp4_pair_2_13 << 16 | _fp4_e2m1_pack_pair_88 << 24;
                unsigned int carry_13 = (word_13 & 2004318071) + 2004318071;
                w_1_3[2] = word_13 & (carry_13 | 2004318071);
                int cf_1_15 = f ^ 3;
                unsigned int v4_1_15[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_15[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_3 + 16 * cf_1_15)));
                float m_1_15 = m0_2;
                if ((cf_1_15 & 2) != 0) {
                    m_1_15 = m1_2;
                }
                float2 mm_14 = make_float2(m_1_15, m_1_15);
                float2 pair_17 = make_float2(__uint_as_float(v4_1_15[0] << 16), __uint_as_float(v4_1_15[0] & 4294901760u));
                float2 g_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_17) : "l"(*(const unsigned long long*)&pair_17), "l"(*(const unsigned long long*)&mm_14));
                uint32_t fp4_pair_16;
                uint16_t _e2m1_pair_h_90;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_90) : "f"(g_17.x), "f"(g_17.y));
                fp4_pair_16 = (uint32_t)_e2m1_pair_h_90;
                float2 pair_1_14 = make_float2(__uint_as_float(v4_1_15[1] << 16), __uint_as_float(v4_1_15[1] & 4294901760u));
                float2 g_1_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_14) : "l"(*(const unsigned long long*)&pair_1_14), "l"(*(const unsigned long long*)&mm_14));
                uint32_t fp4_pair_1_14;
                uint16_t _e2m1_pair_h_91;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_91) : "f"(g_1_14.x), "f"(g_1_14.y));
                fp4_pair_1_14 = (uint32_t)_e2m1_pair_h_91;
                float2 pair_2_14 = make_float2(__uint_as_float(v4_1_15[2] << 16), __uint_as_float(v4_1_15[2] & 4294901760u));
                float2 g_2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_14) : "l"(*(const unsigned long long*)&pair_2_14), "l"(*(const unsigned long long*)&mm_14));
                uint32_t fp4_pair_2_14;
                uint16_t _e2m1_pair_h_92;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_92) : "f"(g_2_14.x), "f"(g_2_14.y));
                fp4_pair_2_14 = (uint32_t)_e2m1_pair_h_92;
                float2 pair_3_14 = make_float2(__uint_as_float(v4_1_15[3] << 16), __uint_as_float(v4_1_15[3] & 4294901760u));
                float2 g_3_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_14) : "l"(*(const unsigned long long*)&pair_3_14), "l"(*(const unsigned long long*)&mm_14));
                uint32_t _fp4_e2m1_pack_pair_93;
                uint16_t _e2m1_pair_h_94;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_94) : "f"(g_3_14.x), "f"(g_3_14.y));
                _fp4_e2m1_pack_pair_93 = (uint32_t)_e2m1_pair_h_94;
                unsigned int word_14 = fp4_pair_16 | fp4_pair_1_14 << 8 | fp4_pair_2_14 << 16 | _fp4_e2m1_pack_pair_93 << 24;
                unsigned int carry_14 = (word_14 & 2004318071) + 2004318071;
                w_1_3[3] = word_14 & (carry_14 | 2004318071);
                bool fodd_1_3 = (f & 1) != 0;
                bool fhi_1_3 = (f & 2) != 0;
                unsigned int t0_1_3 = w_1_3[0];
                unsigned int t1_1_3 = w_1_3[1];
                unsigned int t2_1_3 = w_1_3[2];
                unsigned int t3_1_3 = w_1_3[3];
                if (fodd_1_3) {
                    t0_1_3 = w_1_3[1];
                    t1_1_3 = w_1_3[0];
                    t2_1_3 = w_1_3[3];
                    t3_1_3 = w_1_3[2];
                }
                unsigned int q0_1_3 = t0_1_3;
                unsigned int q1_1_3 = t1_1_3;
                unsigned int q2_1_3 = t2_1_3;
                unsigned int q3_1_3 = t3_1_3;
                if (fhi_1_3) {
                    q0_1_3 = t2_1_3;
                    q1_1_3 = t3_1_3;
                    q2_1_3 = t0_1_3;
                    q3_1_3 = t1_1_3;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_3)), "r"(q0_1_3), "r"(q1_1_3), "r"(q2_1_3), "r"(q3_1_3) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_3)), "h"((uint16_t)(e4m3x2_f32_2 & 32639)) : "memory");
                int pp2_1_4 = lane_1 + 128;
                int base2_1_4 = xb_off + 64 * pp2_1_4;
                float bm0_1_4 = __uint_as_float(bmp[4] << 16);
                float bm1_1_4 = __uint_as_float(bmp[4] & 4294901760u);
                float _fdiv_95 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_4, inv), 6.0f);
                float _fdiv_96 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_4, inv), 6.0f);
                uint16_t e4m3x2_f32_3;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_3) : "f"(_fdiv_95), "f"(_fdiv_96));
                uint32_t e4m3x2_to_f16x2_3;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_3) : "h"((uint16_t)(e4m3x2_f32_3)));
                float cvt_f32_f16_4;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_4) : "h"((uint16_t)(e4m3x2_to_f16x2_3)));
                float cvt_f32_f16_1_3;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_3) : "h"((uint16_t)(e4m3x2_to_f16x2_3 >> 16)));
                float fdiv_rn_5 = __fdiv_rn(inv, cvt_f32_f16_4);
                float fdiv_rn_1_3 = __fdiv_rn(inv, cvt_f32_f16_1_3);
                float m0_3 = 0.0f;
                float m1_3 = 0.0f;
                if (cvt_f32_f16_4 > 0.0f) {
                    m0_3 = fdiv_rn_5;
                }
                if (cvt_f32_f16_1_3 > 0.0f) {
                    m1_3 = fdiv_rn_1_3;
                }
                unsigned int w_1_4[4];
                int cf_1_16 = f ^ 0;
                unsigned int v4_1_16[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_16[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_4 + 16 * cf_1_16)));
                float m_1_16 = m0_3;
                if ((cf_1_16 & 2) != 0) {
                    m_1_16 = m1_3;
                }
                float2 mm_15 = make_float2(m_1_16, m_1_16);
                float2 pair_18 = make_float2(__uint_as_float(v4_1_16[0] << 16), __uint_as_float(v4_1_16[0] & 4294901760u));
                float2 g_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_18) : "l"(*(const unsigned long long*)&pair_18), "l"(*(const unsigned long long*)&mm_15));
                uint32_t fp4_pair_17;
                uint16_t _e2m1_pair_h_97;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_97) : "f"(g_18.x), "f"(g_18.y));
                fp4_pair_17 = (uint32_t)_e2m1_pair_h_97;
                float2 pair_1_15 = make_float2(__uint_as_float(v4_1_16[1] << 16), __uint_as_float(v4_1_16[1] & 4294901760u));
                float2 g_1_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_15) : "l"(*(const unsigned long long*)&pair_1_15), "l"(*(const unsigned long long*)&mm_15));
                uint32_t fp4_pair_1_15;
                uint16_t _e2m1_pair_h_98;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_98) : "f"(g_1_15.x), "f"(g_1_15.y));
                fp4_pair_1_15 = (uint32_t)_e2m1_pair_h_98;
                float2 pair_2_15 = make_float2(__uint_as_float(v4_1_16[2] << 16), __uint_as_float(v4_1_16[2] & 4294901760u));
                float2 g_2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_15) : "l"(*(const unsigned long long*)&pair_2_15), "l"(*(const unsigned long long*)&mm_15));
                uint32_t fp4_pair_2_15;
                uint16_t _e2m1_pair_h_99;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_99) : "f"(g_2_15.x), "f"(g_2_15.y));
                fp4_pair_2_15 = (uint32_t)_e2m1_pair_h_99;
                float2 pair_3_15 = make_float2(__uint_as_float(v4_1_16[3] << 16), __uint_as_float(v4_1_16[3] & 4294901760u));
                float2 g_3_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_15) : "l"(*(const unsigned long long*)&pair_3_15), "l"(*(const unsigned long long*)&mm_15));
                uint32_t _fp4_e2m1_pack_pair_100;
                uint16_t _e2m1_pair_h_101;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_101) : "f"(g_3_15.x), "f"(g_3_15.y));
                _fp4_e2m1_pack_pair_100 = (uint32_t)_e2m1_pair_h_101;
                unsigned int word_15 = fp4_pair_17 | fp4_pair_1_15 << 8 | fp4_pair_2_15 << 16 | _fp4_e2m1_pack_pair_100 << 24;
                unsigned int carry_15 = (word_15 & 2004318071) + 2004318071;
                w_1_4[0] = word_15 & (carry_15 | 2004318071);
                int cf_1_17 = f ^ 1;
                unsigned int v4_1_17[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_17[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_4 + 16 * cf_1_17)));
                float m_1_17 = m0_3;
                if ((cf_1_17 & 2) != 0) {
                    m_1_17 = m1_3;
                }
                float2 mm_16 = make_float2(m_1_17, m_1_17);
                float2 pair_19 = make_float2(__uint_as_float(v4_1_17[0] << 16), __uint_as_float(v4_1_17[0] & 4294901760u));
                float2 g_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_19) : "l"(*(const unsigned long long*)&pair_19), "l"(*(const unsigned long long*)&mm_16));
                uint32_t fp4_pair_18;
                uint16_t _e2m1_pair_h_102;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_102) : "f"(g_19.x), "f"(g_19.y));
                fp4_pair_18 = (uint32_t)_e2m1_pair_h_102;
                float2 pair_1_16 = make_float2(__uint_as_float(v4_1_17[1] << 16), __uint_as_float(v4_1_17[1] & 4294901760u));
                float2 g_1_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_16) : "l"(*(const unsigned long long*)&pair_1_16), "l"(*(const unsigned long long*)&mm_16));
                uint32_t fp4_pair_1_16;
                uint16_t _e2m1_pair_h_103;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_103) : "f"(g_1_16.x), "f"(g_1_16.y));
                fp4_pair_1_16 = (uint32_t)_e2m1_pair_h_103;
                float2 pair_2_16 = make_float2(__uint_as_float(v4_1_17[2] << 16), __uint_as_float(v4_1_17[2] & 4294901760u));
                float2 g_2_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_16) : "l"(*(const unsigned long long*)&pair_2_16), "l"(*(const unsigned long long*)&mm_16));
                uint32_t fp4_pair_2_16;
                uint16_t _e2m1_pair_h_104;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_104) : "f"(g_2_16.x), "f"(g_2_16.y));
                fp4_pair_2_16 = (uint32_t)_e2m1_pair_h_104;
                float2 pair_3_16 = make_float2(__uint_as_float(v4_1_17[3] << 16), __uint_as_float(v4_1_17[3] & 4294901760u));
                float2 g_3_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_16) : "l"(*(const unsigned long long*)&pair_3_16), "l"(*(const unsigned long long*)&mm_16));
                uint32_t _fp4_e2m1_pack_pair_105;
                uint16_t _e2m1_pair_h_106;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_106) : "f"(g_3_16.x), "f"(g_3_16.y));
                _fp4_e2m1_pack_pair_105 = (uint32_t)_e2m1_pair_h_106;
                unsigned int word_16 = fp4_pair_18 | fp4_pair_1_16 << 8 | fp4_pair_2_16 << 16 | _fp4_e2m1_pack_pair_105 << 24;
                unsigned int carry_16 = (word_16 & 2004318071) + 2004318071;
                w_1_4[1] = word_16 & (carry_16 | 2004318071);
                int cf_1_18 = f ^ 2;
                unsigned int v4_1_18[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_18[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_4 + 16 * cf_1_18)));
                float m_1_18 = m0_3;
                if ((cf_1_18 & 2) != 0) {
                    m_1_18 = m1_3;
                }
                float2 mm_17 = make_float2(m_1_18, m_1_18);
                float2 pair_20 = make_float2(__uint_as_float(v4_1_18[0] << 16), __uint_as_float(v4_1_18[0] & 4294901760u));
                float2 g_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_20) : "l"(*(const unsigned long long*)&pair_20), "l"(*(const unsigned long long*)&mm_17));
                uint32_t fp4_pair_19;
                uint16_t _e2m1_pair_h_107;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_107) : "f"(g_20.x), "f"(g_20.y));
                fp4_pair_19 = (uint32_t)_e2m1_pair_h_107;
                float2 pair_1_17 = make_float2(__uint_as_float(v4_1_18[1] << 16), __uint_as_float(v4_1_18[1] & 4294901760u));
                float2 g_1_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_17) : "l"(*(const unsigned long long*)&pair_1_17), "l"(*(const unsigned long long*)&mm_17));
                uint32_t fp4_pair_1_17;
                uint16_t _e2m1_pair_h_108;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_108) : "f"(g_1_17.x), "f"(g_1_17.y));
                fp4_pair_1_17 = (uint32_t)_e2m1_pair_h_108;
                float2 pair_2_17 = make_float2(__uint_as_float(v4_1_18[2] << 16), __uint_as_float(v4_1_18[2] & 4294901760u));
                float2 g_2_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_17) : "l"(*(const unsigned long long*)&pair_2_17), "l"(*(const unsigned long long*)&mm_17));
                uint32_t fp4_pair_2_17;
                uint16_t _e2m1_pair_h_109;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_109) : "f"(g_2_17.x), "f"(g_2_17.y));
                fp4_pair_2_17 = (uint32_t)_e2m1_pair_h_109;
                float2 pair_3_17 = make_float2(__uint_as_float(v4_1_18[3] << 16), __uint_as_float(v4_1_18[3] & 4294901760u));
                float2 g_3_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_17) : "l"(*(const unsigned long long*)&pair_3_17), "l"(*(const unsigned long long*)&mm_17));
                uint32_t _fp4_e2m1_pack_pair_110;
                uint16_t _e2m1_pair_h_111;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_111) : "f"(g_3_17.x), "f"(g_3_17.y));
                _fp4_e2m1_pack_pair_110 = (uint32_t)_e2m1_pair_h_111;
                unsigned int word_17 = fp4_pair_19 | fp4_pair_1_17 << 8 | fp4_pair_2_17 << 16 | _fp4_e2m1_pack_pair_110 << 24;
                unsigned int carry_17 = (word_17 & 2004318071) + 2004318071;
                w_1_4[2] = word_17 & (carry_17 | 2004318071);
                int cf_1_19 = f ^ 3;
                unsigned int v4_1_19[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_19[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_4 + 16 * cf_1_19)));
                float m_1_19 = m0_3;
                if ((cf_1_19 & 2) != 0) {
                    m_1_19 = m1_3;
                }
                float2 mm_18 = make_float2(m_1_19, m_1_19);
                float2 pair_21 = make_float2(__uint_as_float(v4_1_19[0] << 16), __uint_as_float(v4_1_19[0] & 4294901760u));
                float2 g_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_21) : "l"(*(const unsigned long long*)&pair_21), "l"(*(const unsigned long long*)&mm_18));
                uint32_t fp4_pair_20;
                uint16_t _e2m1_pair_h_112;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_112) : "f"(g_21.x), "f"(g_21.y));
                fp4_pair_20 = (uint32_t)_e2m1_pair_h_112;
                float2 pair_1_18 = make_float2(__uint_as_float(v4_1_19[1] << 16), __uint_as_float(v4_1_19[1] & 4294901760u));
                float2 g_1_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_18) : "l"(*(const unsigned long long*)&pair_1_18), "l"(*(const unsigned long long*)&mm_18));
                uint32_t fp4_pair_1_18;
                uint16_t _e2m1_pair_h_113;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_113) : "f"(g_1_18.x), "f"(g_1_18.y));
                fp4_pair_1_18 = (uint32_t)_e2m1_pair_h_113;
                float2 pair_2_18 = make_float2(__uint_as_float(v4_1_19[2] << 16), __uint_as_float(v4_1_19[2] & 4294901760u));
                float2 g_2_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_18) : "l"(*(const unsigned long long*)&pair_2_18), "l"(*(const unsigned long long*)&mm_18));
                uint32_t fp4_pair_2_18;
                uint16_t _e2m1_pair_h_114;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_114) : "f"(g_2_18.x), "f"(g_2_18.y));
                fp4_pair_2_18 = (uint32_t)_e2m1_pair_h_114;
                float2 pair_3_18 = make_float2(__uint_as_float(v4_1_19[3] << 16), __uint_as_float(v4_1_19[3] & 4294901760u));
                float2 g_3_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_18) : "l"(*(const unsigned long long*)&pair_3_18), "l"(*(const unsigned long long*)&mm_18));
                uint32_t _fp4_e2m1_pack_pair_115;
                uint16_t _e2m1_pair_h_116;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_116) : "f"(g_3_18.x), "f"(g_3_18.y));
                _fp4_e2m1_pack_pair_115 = (uint32_t)_e2m1_pair_h_116;
                unsigned int word_18 = fp4_pair_20 | fp4_pair_1_18 << 8 | fp4_pair_2_18 << 16 | _fp4_e2m1_pack_pair_115 << 24;
                unsigned int carry_18 = (word_18 & 2004318071) + 2004318071;
                w_1_4[3] = word_18 & (carry_18 | 2004318071);
                bool fodd_1_4 = (f & 1) != 0;
                bool fhi_1_4 = (f & 2) != 0;
                unsigned int t0_1_4 = w_1_4[0];
                unsigned int t1_1_4 = w_1_4[1];
                unsigned int t2_1_4 = w_1_4[2];
                unsigned int t3_1_4 = w_1_4[3];
                if (fodd_1_4) {
                    t0_1_4 = w_1_4[1];
                    t1_1_4 = w_1_4[0];
                    t2_1_4 = w_1_4[3];
                    t3_1_4 = w_1_4[2];
                }
                unsigned int q0_1_4 = t0_1_4;
                unsigned int q1_1_4 = t1_1_4;
                unsigned int q2_1_4 = t2_1_4;
                unsigned int q3_1_4 = t3_1_4;
                if (fhi_1_4) {
                    q0_1_4 = t2_1_4;
                    q1_1_4 = t3_1_4;
                    q2_1_4 = t0_1_4;
                    q3_1_4 = t1_1_4;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_4)), "r"(q0_1_4), "r"(q1_1_4), "r"(q2_1_4), "r"(q3_1_4) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_4)), "h"((uint16_t)(e4m3x2_f32_3 & 32639)) : "memory");
                int pp2_1_5 = lane_1 + 160;
                int base2_1_5 = xb_off + 64 * pp2_1_5;
                float bm0_1_5 = __uint_as_float(bmp[5] << 16);
                float bm1_1_5 = __uint_as_float(bmp[5] & 4294901760u);
                float _fdiv_117 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_5, inv), 6.0f);
                float _fdiv_118 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_5, inv), 6.0f);
                uint16_t e4m3x2_f32_4;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_4) : "f"(_fdiv_117), "f"(_fdiv_118));
                uint32_t e4m3x2_to_f16x2_4;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_4) : "h"((uint16_t)(e4m3x2_f32_4)));
                float cvt_f32_f16_5;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_5) : "h"((uint16_t)(e4m3x2_to_f16x2_4)));
                float cvt_f32_f16_1_4;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_4) : "h"((uint16_t)(e4m3x2_to_f16x2_4 >> 16)));
                float fdiv_rn_6 = __fdiv_rn(inv, cvt_f32_f16_5);
                float fdiv_rn_1_4 = __fdiv_rn(inv, cvt_f32_f16_1_4);
                float m0_4 = 0.0f;
                float m1_4 = 0.0f;
                if (cvt_f32_f16_5 > 0.0f) {
                    m0_4 = fdiv_rn_6;
                }
                if (cvt_f32_f16_1_4 > 0.0f) {
                    m1_4 = fdiv_rn_1_4;
                }
                unsigned int w_1_5[4];
                int cf_1_20 = f ^ 0;
                unsigned int v4_1_20[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_20[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_5 + 16 * cf_1_20)));
                float m_1_20 = m0_4;
                if ((cf_1_20 & 2) != 0) {
                    m_1_20 = m1_4;
                }
                float2 mm_19 = make_float2(m_1_20, m_1_20);
                float2 pair_22 = make_float2(__uint_as_float(v4_1_20[0] << 16), __uint_as_float(v4_1_20[0] & 4294901760u));
                float2 g_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_22) : "l"(*(const unsigned long long*)&pair_22), "l"(*(const unsigned long long*)&mm_19));
                uint32_t fp4_pair_21;
                uint16_t _e2m1_pair_h_119;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_119) : "f"(g_22.x), "f"(g_22.y));
                fp4_pair_21 = (uint32_t)_e2m1_pair_h_119;
                float2 pair_1_19 = make_float2(__uint_as_float(v4_1_20[1] << 16), __uint_as_float(v4_1_20[1] & 4294901760u));
                float2 g_1_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_19) : "l"(*(const unsigned long long*)&pair_1_19), "l"(*(const unsigned long long*)&mm_19));
                uint32_t fp4_pair_1_19;
                uint16_t _e2m1_pair_h_120;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_120) : "f"(g_1_19.x), "f"(g_1_19.y));
                fp4_pair_1_19 = (uint32_t)_e2m1_pair_h_120;
                float2 pair_2_19 = make_float2(__uint_as_float(v4_1_20[2] << 16), __uint_as_float(v4_1_20[2] & 4294901760u));
                float2 g_2_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_19) : "l"(*(const unsigned long long*)&pair_2_19), "l"(*(const unsigned long long*)&mm_19));
                uint32_t fp4_pair_2_19;
                uint16_t _e2m1_pair_h_121;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_121) : "f"(g_2_19.x), "f"(g_2_19.y));
                fp4_pair_2_19 = (uint32_t)_e2m1_pair_h_121;
                float2 pair_3_19 = make_float2(__uint_as_float(v4_1_20[3] << 16), __uint_as_float(v4_1_20[3] & 4294901760u));
                float2 g_3_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_19) : "l"(*(const unsigned long long*)&pair_3_19), "l"(*(const unsigned long long*)&mm_19));
                uint32_t _fp4_e2m1_pack_pair_122;
                uint16_t _e2m1_pair_h_123;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_123) : "f"(g_3_19.x), "f"(g_3_19.y));
                _fp4_e2m1_pack_pair_122 = (uint32_t)_e2m1_pair_h_123;
                unsigned int word_19 = fp4_pair_21 | fp4_pair_1_19 << 8 | fp4_pair_2_19 << 16 | _fp4_e2m1_pack_pair_122 << 24;
                unsigned int carry_19 = (word_19 & 2004318071) + 2004318071;
                w_1_5[0] = word_19 & (carry_19 | 2004318071);
                int cf_1_21 = f ^ 1;
                unsigned int v4_1_21[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_21[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_5 + 16 * cf_1_21)));
                float m_1_21 = m0_4;
                if ((cf_1_21 & 2) != 0) {
                    m_1_21 = m1_4;
                }
                float2 mm_20 = make_float2(m_1_21, m_1_21);
                float2 pair_23 = make_float2(__uint_as_float(v4_1_21[0] << 16), __uint_as_float(v4_1_21[0] & 4294901760u));
                float2 g_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_23) : "l"(*(const unsigned long long*)&pair_23), "l"(*(const unsigned long long*)&mm_20));
                uint32_t fp4_pair_22;
                uint16_t _e2m1_pair_h_124;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_124) : "f"(g_23.x), "f"(g_23.y));
                fp4_pair_22 = (uint32_t)_e2m1_pair_h_124;
                float2 pair_1_20 = make_float2(__uint_as_float(v4_1_21[1] << 16), __uint_as_float(v4_1_21[1] & 4294901760u));
                float2 g_1_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_20) : "l"(*(const unsigned long long*)&pair_1_20), "l"(*(const unsigned long long*)&mm_20));
                uint32_t fp4_pair_1_20;
                uint16_t _e2m1_pair_h_125;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_125) : "f"(g_1_20.x), "f"(g_1_20.y));
                fp4_pair_1_20 = (uint32_t)_e2m1_pair_h_125;
                float2 pair_2_20 = make_float2(__uint_as_float(v4_1_21[2] << 16), __uint_as_float(v4_1_21[2] & 4294901760u));
                float2 g_2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_20) : "l"(*(const unsigned long long*)&pair_2_20), "l"(*(const unsigned long long*)&mm_20));
                uint32_t fp4_pair_2_20;
                uint16_t _e2m1_pair_h_126;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_126) : "f"(g_2_20.x), "f"(g_2_20.y));
                fp4_pair_2_20 = (uint32_t)_e2m1_pair_h_126;
                float2 pair_3_20 = make_float2(__uint_as_float(v4_1_21[3] << 16), __uint_as_float(v4_1_21[3] & 4294901760u));
                float2 g_3_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_20) : "l"(*(const unsigned long long*)&pair_3_20), "l"(*(const unsigned long long*)&mm_20));
                uint32_t _fp4_e2m1_pack_pair_127;
                uint16_t _e2m1_pair_h_128;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_128) : "f"(g_3_20.x), "f"(g_3_20.y));
                _fp4_e2m1_pack_pair_127 = (uint32_t)_e2m1_pair_h_128;
                unsigned int word_20 = fp4_pair_22 | fp4_pair_1_20 << 8 | fp4_pair_2_20 << 16 | _fp4_e2m1_pack_pair_127 << 24;
                unsigned int carry_20 = (word_20 & 2004318071) + 2004318071;
                w_1_5[1] = word_20 & (carry_20 | 2004318071);
                int cf_1_22 = f ^ 2;
                unsigned int v4_1_22[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_22[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_5 + 16 * cf_1_22)));
                float m_1_22 = m0_4;
                if ((cf_1_22 & 2) != 0) {
                    m_1_22 = m1_4;
                }
                float2 mm_21 = make_float2(m_1_22, m_1_22);
                float2 pair_24 = make_float2(__uint_as_float(v4_1_22[0] << 16), __uint_as_float(v4_1_22[0] & 4294901760u));
                float2 g_24;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_24) : "l"(*(const unsigned long long*)&pair_24), "l"(*(const unsigned long long*)&mm_21));
                uint32_t fp4_pair_23;
                uint16_t _e2m1_pair_h_129;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_129) : "f"(g_24.x), "f"(g_24.y));
                fp4_pair_23 = (uint32_t)_e2m1_pair_h_129;
                float2 pair_1_21 = make_float2(__uint_as_float(v4_1_22[1] << 16), __uint_as_float(v4_1_22[1] & 4294901760u));
                float2 g_1_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_21) : "l"(*(const unsigned long long*)&pair_1_21), "l"(*(const unsigned long long*)&mm_21));
                uint32_t fp4_pair_1_21;
                uint16_t _e2m1_pair_h_130;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_130) : "f"(g_1_21.x), "f"(g_1_21.y));
                fp4_pair_1_21 = (uint32_t)_e2m1_pair_h_130;
                float2 pair_2_21 = make_float2(__uint_as_float(v4_1_22[2] << 16), __uint_as_float(v4_1_22[2] & 4294901760u));
                float2 g_2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_21) : "l"(*(const unsigned long long*)&pair_2_21), "l"(*(const unsigned long long*)&mm_21));
                uint32_t fp4_pair_2_21;
                uint16_t _e2m1_pair_h_131;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_131) : "f"(g_2_21.x), "f"(g_2_21.y));
                fp4_pair_2_21 = (uint32_t)_e2m1_pair_h_131;
                float2 pair_3_21 = make_float2(__uint_as_float(v4_1_22[3] << 16), __uint_as_float(v4_1_22[3] & 4294901760u));
                float2 g_3_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_21) : "l"(*(const unsigned long long*)&pair_3_21), "l"(*(const unsigned long long*)&mm_21));
                uint32_t _fp4_e2m1_pack_pair_132;
                uint16_t _e2m1_pair_h_133;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_133) : "f"(g_3_21.x), "f"(g_3_21.y));
                _fp4_e2m1_pack_pair_132 = (uint32_t)_e2m1_pair_h_133;
                unsigned int word_21 = fp4_pair_23 | fp4_pair_1_21 << 8 | fp4_pair_2_21 << 16 | _fp4_e2m1_pack_pair_132 << 24;
                unsigned int carry_21 = (word_21 & 2004318071) + 2004318071;
                w_1_5[2] = word_21 & (carry_21 | 2004318071);
                int cf_1_23 = f ^ 3;
                unsigned int v4_1_23[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_23[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(base2_1_5 + 16 * cf_1_23)));
                float m_1_23 = m0_4;
                if ((cf_1_23 & 2) != 0) {
                    m_1_23 = m1_4;
                }
                float2 mm_22 = make_float2(m_1_23, m_1_23);
                float2 pair_25 = make_float2(__uint_as_float(v4_1_23[0] << 16), __uint_as_float(v4_1_23[0] & 4294901760u));
                float2 g_25;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_25) : "l"(*(const unsigned long long*)&pair_25), "l"(*(const unsigned long long*)&mm_22));
                uint32_t fp4_pair_24;
                uint16_t _e2m1_pair_h_134;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_134) : "f"(g_25.x), "f"(g_25.y));
                fp4_pair_24 = (uint32_t)_e2m1_pair_h_134;
                float2 pair_1_22 = make_float2(__uint_as_float(v4_1_23[1] << 16), __uint_as_float(v4_1_23[1] & 4294901760u));
                float2 g_1_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_22) : "l"(*(const unsigned long long*)&pair_1_22), "l"(*(const unsigned long long*)&mm_22));
                uint32_t fp4_pair_1_22;
                uint16_t _e2m1_pair_h_135;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_135) : "f"(g_1_22.x), "f"(g_1_22.y));
                fp4_pair_1_22 = (uint32_t)_e2m1_pair_h_135;
                float2 pair_2_22 = make_float2(__uint_as_float(v4_1_23[2] << 16), __uint_as_float(v4_1_23[2] & 4294901760u));
                float2 g_2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_22) : "l"(*(const unsigned long long*)&pair_2_22), "l"(*(const unsigned long long*)&mm_22));
                uint32_t fp4_pair_2_22;
                uint16_t _e2m1_pair_h_136;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_136) : "f"(g_2_22.x), "f"(g_2_22.y));
                fp4_pair_2_22 = (uint32_t)_e2m1_pair_h_136;
                float2 pair_3_22 = make_float2(__uint_as_float(v4_1_23[3] << 16), __uint_as_float(v4_1_23[3] & 4294901760u));
                float2 g_3_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_22) : "l"(*(const unsigned long long*)&pair_3_22), "l"(*(const unsigned long long*)&mm_22));
                uint32_t _fp4_e2m1_pack_pair_137;
                uint16_t _e2m1_pair_h_138;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_138) : "f"(g_3_22.x), "f"(g_3_22.y));
                _fp4_e2m1_pack_pair_137 = (uint32_t)_e2m1_pair_h_138;
                unsigned int word_22 = fp4_pair_24 | fp4_pair_1_22 << 8 | fp4_pair_2_22 << 16 | _fp4_e2m1_pack_pair_137 << 24;
                unsigned int carry_22 = (word_22 & 2004318071) + 2004318071;
                w_1_5[3] = word_22 & (carry_22 | 2004318071);
                bool fodd_1_5 = (f & 1) != 0;
                bool fhi_1_5 = (f & 2) != 0;
                unsigned int t0_1_5 = w_1_5[0];
                unsigned int t1_1_5 = w_1_5[1];
                unsigned int t2_1_5 = w_1_5[2];
                unsigned int t3_1_5 = w_1_5[3];
                if (fodd_1_5) {
                    t0_1_5 = w_1_5[1];
                    t1_1_5 = w_1_5[0];
                    t2_1_5 = w_1_5[3];
                    t3_1_5 = w_1_5[2];
                }
                unsigned int q0_1_5 = t0_1_5;
                unsigned int q1_1_5 = t1_1_5;
                unsigned int q2_1_5 = t2_1_5;
                unsigned int q3_1_5 = t3_1_5;
                if (fhi_1_5) {
                    q0_1_5 = t2_1_5;
                    q1_1_5 = t3_1_5;
                    q2_1_5 = t0_1_5;
                    q3_1_5 = t1_1_5;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_5)), "r"(q0_1_5), "r"(q1_1_5), "r"(q2_1_5), "r"(q3_1_5) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_5)), "h"((uint16_t)(e4m3x2_f32_4 & 32639)) : "memory");
                rs = rs_0;
                __syncwarp();
                if (n > k + 1 && lane_1 == 0) {
                    long long next_row2 = (long long)(r0 + warp_0 + 10 * (k + 1));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(xb_full_addr + (((xc + 1 < (unsigned int)(2 * warp_0) + 2) ? xc + 1 : (unsigned int)(2 * warp_0))) * 8, 12288);
                    cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, x + (next_row2 * 6144), 12288, xb_full_addr + (((xc + 1 < (unsigned int)(2 * warp_0) + 2) ? xc + 1 : (unsigned int)(2 * warp_0))) * 8);
                }
                if (lane_1 < 4) {
                    {
                        uint32_t _addr_139 = static_cast<uint32_t>(wsm_addr + (unsigned int)(ob_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_139), "f"(rs) : "memory");
                    }
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
                        void* _cpbulk_dst_140 = reinterpret_cast<void*>(dst_w + v_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_140), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_141 = reinterpret_cast<void*>(dst_w + sc_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_141), "r"(wsm_addr + (unsigned int)(ob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    uint8_t* dst_r = reinterpret_cast<uint8_t*>(base_r);
                    int m_off = rs_ro + (rk * capo + ix) * 16;
                    {
                        void* _cpbulk_dst_142 = reinterpret_cast<void*>(dst_r + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_142), "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 384)), "r"((uint32_t)(16))
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
            int g32 = (int)gridDim.x;
            if (rows / g32 <= 8) {
                asm volatile("barrier.arrive 7, 352;" ::: "memory");
            }
            int nC = nrows - nB;
            int wB = (warp_0 - nrows % 10 + 10) % 10;
            int wC = (warp_0 - (nrows + nB) % 10 + 10) % 10;
            int kb = 0;
            int t = warp_0 >> 1;
            int h = warp_0 & 1;
            int lxb_off = 2 * t * 21280;
            int i0 = (t - (nrows % 10 >> 1) + 5) % 5;
            #pragma unroll 1
            for (int i = i0; i < nB; i += 5) {
                int trow = rb0 + 4 * i;
                long long tix64 = (long long)(trow / 4);
                long long trow64 = (long long)trow;
                int kbp = kb & 1;
                int tob_off = lxb_off + kbp * 3472;
                if (h == 0 && kb >= 2 && lane_1 == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                int rs_wo_1 = (int)rs_w;
                int rs_ro_1 = (int)rs_r;
                int capo_1 = (int)capo64;
                int ix_1 = (int)tix64;
                int rk_1 = (int)rank64;
                uint32_t _sysv_poll32_group_0[4];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + (rs_ro_1 + ix_1 * 16))) : "memory");
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[1]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + (rs_ro_1 + (capo_1 + ix_1) * 16))) : "memory");
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[2]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + (rs_ro_1 + (2 * capo_1 + ix_1) * 16))) : "memory");
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[3]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + (rs_ro_1 + (3 * capo_1 + ix_1) * 16))) : "memory");
                } while (
                    (_sysv_poll32_group_0[0] == 0x80000000u) || (_sysv_poll32_group_0[1] == 0x80000000u) || (_sysv_poll32_group_0[2] == 0x80000000u) || (_sysv_poll32_group_0[3] == 0x80000000u)
                );
                float rs_s = __uint_as_float(_sysv_poll32_group_0[0]);
                float rs_s_0 = __uint_as_float(_sysv_poll32_group_0[1]);
                float rs_s_1 = __uint_as_float(_sysv_poll32_group_0[2]);
                float rs_s_2 = __uint_as_float(_sysv_poll32_group_0[3]);
                int v = rs_wo_1 + ix_1 * 3072;
                int sc = rs_wo_1 + 4 * capo_1 * 3072 + ix_1 * 384;
                int v_1 = rs_wo_1 + (capo_1 + ix_1) * 3072;
                int sc_1 = rs_wo_1 + 4 * capo_1 * 3072 + (capo_1 + ix_1) * 384;
                int v_2 = rs_wo_1 + (2 * capo_1 + ix_1) * 3072;
                int sc_2 = rs_wo_1 + 4 * capo_1 * 3072 + (2 * capo_1 + ix_1) * 384;
                int v_3 = rs_wo_1 + (3 * capo_1 + ix_1) * 3072;
                int sc_3 = rs_wo_1 + 4 * capo_1 * 3072 + (3 * capo_1 + ix_1) * 384;
                int pp_0_1 = lane_1 + 32 * (3 * h);
                int pp_1 = lane_1 + 32 * (3 * h + 1);
                int pp_2 = lane_1 + 32 * (3 * h + 2);
                uint32_t fp4_wire_values[48];
                uint32_t fp4_wire_scales[12];
                uint16_t _wire_h16_143_0;
                uint16_t _wire_h16_143_1;
                uint16_t _wire_h16_143_2;
                uint16_t _wire_h16_143_3;
                uint16_t _wire_h16_143_4;
                uint16_t _wire_h16_143_5;
                uint16_t _wire_h16_143_6;
                uint16_t _wire_h16_143_7;
                uint16_t _wire_h16_143_8;
                uint16_t _wire_h16_143_9;
                uint16_t _wire_h16_143_10;
                uint16_t _wire_h16_143_11;
                uint32_t _wire_vpend_143 = 4095u;
                uint32_t _wire_spend_143 = 4095u;
                do {
                    if (_wire_vpend_143 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[0]), "=r"(fp4_wire_values[1]), "=r"(fp4_wire_values[2]), "=r"(fp4_wire_values[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_0_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 2u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[4]), "=r"(fp4_wire_values[5]), "=r"(fp4_wire_values[6]), "=r"(fp4_wire_values[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp_0_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 4u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[8]), "=r"(fp4_wire_values[9]), "=r"(fp4_wire_values[10]), "=r"(fp4_wire_values[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp_0_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 8u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[12]), "=r"(fp4_wire_values[13]), "=r"(fp4_wire_values[14]), "=r"(fp4_wire_values[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp_0_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 16u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[16]), "=r"(fp4_wire_values[17]), "=r"(fp4_wire_values[18]), "=r"(fp4_wire_values[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 32u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[20]), "=r"(fp4_wire_values[21]), "=r"(fp4_wire_values[22]), "=r"(fp4_wire_values[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 64u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[24]), "=r"(fp4_wire_values[25]), "=r"(fp4_wire_values[26]), "=r"(fp4_wire_values[27]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 128u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[28]), "=r"(fp4_wire_values[29]), "=r"(fp4_wire_values[30]), "=r"(fp4_wire_values[31]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 256u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[32]), "=r"(fp4_wire_values[33]), "=r"(fp4_wire_values[34]), "=r"(fp4_wire_values[35]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_2)))) : "memory");
                    }
                    if (_wire_vpend_143 & 512u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[36]), "=r"(fp4_wire_values[37]), "=r"(fp4_wire_values[38]), "=r"(fp4_wire_values[39]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp_2)))) : "memory");
                    }
                    if (_wire_vpend_143 & 1024u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[40]), "=r"(fp4_wire_values[41]), "=r"(fp4_wire_values[42]), "=r"(fp4_wire_values[43]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp_2)))) : "memory");
                    }
                    if (_wire_vpend_143 & 2048u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[44]), "=r"(fp4_wire_values[45]), "=r"(fp4_wire_values[46]), "=r"(fp4_wire_values[47]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp_2)))) : "memory");
                    }
                    if (_wire_spend_143 & 1u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp_0_1)))) : "memory");
                        fp4_wire_scales[0] = (uint32_t)_wire_h16_143_0;
                    }
                    if (_wire_spend_143 & 2u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp_0_1)))) : "memory");
                        fp4_wire_scales[1] = (uint32_t)_wire_h16_143_1;
                    }
                    if (_wire_spend_143 & 4u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * pp_0_1)))) : "memory");
                        fp4_wire_scales[2] = (uint32_t)_wire_h16_143_2;
                    }
                    if (_wire_spend_143 & 8u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp_0_1)))) : "memory");
                        fp4_wire_scales[3] = (uint32_t)_wire_h16_143_3;
                    }
                    if (_wire_spend_143 & 16u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp_1)))) : "memory");
                        fp4_wire_scales[4] = (uint32_t)_wire_h16_143_4;
                    }
                    if (_wire_spend_143 & 32u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp_1)))) : "memory");
                        fp4_wire_scales[5] = (uint32_t)_wire_h16_143_5;
                    }
                    if (_wire_spend_143 & 64u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_6) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * pp_1)))) : "memory");
                        fp4_wire_scales[6] = (uint32_t)_wire_h16_143_6;
                    }
                    if (_wire_spend_143 & 128u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_7) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp_1)))) : "memory");
                        fp4_wire_scales[7] = (uint32_t)_wire_h16_143_7;
                    }
                    if (_wire_spend_143 & 256u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_8) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp_2)))) : "memory");
                        fp4_wire_scales[8] = (uint32_t)_wire_h16_143_8;
                    }
                    if (_wire_spend_143 & 512u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_9) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp_2)))) : "memory");
                        fp4_wire_scales[9] = (uint32_t)_wire_h16_143_9;
                    }
                    if (_wire_spend_143 & 1024u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_10) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * pp_2)))) : "memory");
                        fp4_wire_scales[10] = (uint32_t)_wire_h16_143_10;
                    }
                    if (_wire_spend_143 & 2048u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_11) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp_2)))) : "memory");
                        fp4_wire_scales[11] = (uint32_t)_wire_h16_143_11;
                    }
                    _wire_vpend_143 = ((uint32_t)((((fp4_wire_values[0] & ~((fp4_wire_values[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[1] & ~((fp4_wire_values[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[2] & ~((fp4_wire_values[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[3] & ~((fp4_wire_values[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((fp4_wire_values[4] & ~((fp4_wire_values[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[5] & ~((fp4_wire_values[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[6] & ~((fp4_wire_values[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[7] & ~((fp4_wire_values[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((fp4_wire_values[8] & ~((fp4_wire_values[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[9] & ~((fp4_wire_values[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[10] & ~((fp4_wire_values[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[11] & ~((fp4_wire_values[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((fp4_wire_values[12] & ~((fp4_wire_values[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[13] & ~((fp4_wire_values[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[14] & ~((fp4_wire_values[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[15] & ~((fp4_wire_values[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((fp4_wire_values[16] & ~((fp4_wire_values[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[17] & ~((fp4_wire_values[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[18] & ~((fp4_wire_values[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[19] & ~((fp4_wire_values[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((fp4_wire_values[20] & ~((fp4_wire_values[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[21] & ~((fp4_wire_values[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[22] & ~((fp4_wire_values[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[23] & ~((fp4_wire_values[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5) | ((uint32_t)((((fp4_wire_values[24] & ~((fp4_wire_values[24] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[25] & ~((fp4_wire_values[25] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[26] & ~((fp4_wire_values[26] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[27] & ~((fp4_wire_values[27] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 6) | ((uint32_t)((((fp4_wire_values[28] & ~((fp4_wire_values[28] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[29] & ~((fp4_wire_values[29] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[30] & ~((fp4_wire_values[30] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[31] & ~((fp4_wire_values[31] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 7) | ((uint32_t)((((fp4_wire_values[32] & ~((fp4_wire_values[32] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[33] & ~((fp4_wire_values[33] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[34] & ~((fp4_wire_values[34] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[35] & ~((fp4_wire_values[35] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 8) | ((uint32_t)((((fp4_wire_values[36] & ~((fp4_wire_values[36] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[37] & ~((fp4_wire_values[37] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[38] & ~((fp4_wire_values[38] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[39] & ~((fp4_wire_values[39] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 9) | ((uint32_t)((((fp4_wire_values[40] & ~((fp4_wire_values[40] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[41] & ~((fp4_wire_values[41] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[42] & ~((fp4_wire_values[42] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[43] & ~((fp4_wire_values[43] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 10) | ((uint32_t)((((fp4_wire_values[44] & ~((fp4_wire_values[44] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[45] & ~((fp4_wire_values[45] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[46] & ~((fp4_wire_values[46] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[47] & ~((fp4_wire_values[47] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 11);
                    _wire_spend_143 = ((uint32_t)(((uint32_t)((fp4_wire_scales[0] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((fp4_wire_scales[1] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((fp4_wire_scales[2] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((fp4_wire_scales[3] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((fp4_wire_scales[4] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((fp4_wire_scales[5] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5) | ((uint32_t)(((uint32_t)((fp4_wire_scales[6] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[6] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 6) | ((uint32_t)(((uint32_t)((fp4_wire_scales[7] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[7] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 7) | ((uint32_t)(((uint32_t)((fp4_wire_scales[8] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[8] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 8) | ((uint32_t)(((uint32_t)((fp4_wire_scales[9] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[9] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 9) | ((uint32_t)(((uint32_t)((fp4_wire_scales[10] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[10] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 10) | ((uint32_t)(((uint32_t)((fp4_wire_scales[11] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[11] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 11);
                } while ((_wire_vpend_143 | _wire_spend_143) != 0u);
                float acc_0[32] = {0};
                uint32_t e4m3x2_to_f16x2_5;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_5) : "h"((uint16_t)(fp4_wire_scales[0])));
                float cvt_f32_f16_6;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_6) : "h"((uint16_t)(e4m3x2_to_f16x2_5)));
                float cvt_f32_f16_1_5;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_5) : "h"((uint16_t)(e4m3x2_to_f16x2_5 >> 16)));
                float s0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(cvt_f32_f16_6), "f"(rs_s));
                float s1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1) : "f"(cvt_f32_f16_1_5), "f"(rs_s));
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
                float2 a_2_1 = make_float2(acc_0[4], acc_0[5]);
                float2 q_2 = make_float2(tv_0[4], tv_0[5]);
                float2 r_2 = fma_f32x2_rn_noftz(q_2, scale0, a_2_1);
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
                float2 a_12_1 = make_float2(acc_0[24], acc_0[25]);
                float2 q_12 = make_float2(tv_3[0], tv_3[1]);
                float2 r_12 = fma_f32x2_rn_noftz(q_12, scale1, a_12_1);
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
                uint32_t e4m3x2_to_f16x2_5_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_5_1) : "h"((uint16_t)(fp4_wire_scales[1])));
                float cvt_f32_f16_6_1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_6_1) : "h"((uint16_t)(e4m3x2_to_f16x2_5_1)));
                float cvt_f32_f16_1_5_1;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_5_1) : "h"((uint16_t)(e4m3x2_to_f16x2_5_1 >> 16)));
                float s0_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(cvt_f32_f16_6_1), "f"(rs_s_0));
                float s1_0;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_0) : "f"(cvt_f32_f16_1_5_1), "f"(rs_s_0));
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
                float2 a_22_1 = make_float2(acc_0[12], acc_0[13]);
                float2 q_22 = make_float2(tv_5[4], tv_5[5]);
                float2 r_22 = fma_f32x2_rn_noftz(q_22, scale0_0, a_22_1);
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
                uint32_t e4m3x2_to_f16x2_6;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_6) : "h"((uint16_t)(fp4_wire_scales[2])));
                float cvt_f32_f16_7;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_7) : "h"((uint16_t)(e4m3x2_to_f16x2_6)));
                float cvt_f32_f16_1_6;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_6) : "h"((uint16_t)(e4m3x2_to_f16x2_6 >> 16)));
                float s0_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(cvt_f32_f16_7), "f"(rs_s_1));
                float s1_1;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_1) : "f"(cvt_f32_f16_1_6), "f"(rs_s_1));
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
                float2 a_32_1 = make_float2(acc_0[0], acc_0[1]);
                float2 q_32 = make_float2(tv_8[0], tv_8[1]);
                float2 r_32 = fma_f32x2_rn_noftz(q_32, scale0_1, a_32_1);
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
                float2 a_42_1 = make_float2(acc_0[20], acc_0[21]);
                float2 q_42 = make_float2(tv_10[4], tv_10[5]);
                float2 r_42 = fma_f32x2_rn_noftz(q_42, scale1_1, a_42_1);
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
                uint32_t e4m3x2_to_f16x2_7;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_7) : "h"((uint16_t)(fp4_wire_scales[3])));
                float cvt_f32_f16_8;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_8) : "h"((uint16_t)(e4m3x2_to_f16x2_7)));
                float cvt_f32_f16_1_7;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_7) : "h"((uint16_t)(e4m3x2_to_f16x2_7 >> 16)));
                float s0_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(cvt_f32_f16_8), "f"(rs_s_2));
                float s1_2;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_2) : "f"(cvt_f32_f16_1_7), "f"(rs_s_2));
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
                float acc_1[32] = {0};
                uint32_t e4m3x2_to_f16x2_8;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_8) : "h"((uint16_t)(fp4_wire_scales[4])));
                float cvt_f32_f16_9;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_9) : "h"((uint16_t)(e4m3x2_to_f16x2_8)));
                float cvt_f32_f16_1_8;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_8) : "h"((uint16_t)(e4m3x2_to_f16x2_8 >> 16)));
                float s0_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_3) : "f"(cvt_f32_f16_9), "f"(rs_s));
                float s1_3;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_3) : "f"(cvt_f32_f16_1_8), "f"(rs_s));
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
                uint32_t e4m3x2_to_f16x2_9;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_9) : "h"((uint16_t)(fp4_wire_scales[5])));
                float cvt_f32_f16_10;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_10) : "h"((uint16_t)(e4m3x2_to_f16x2_9)));
                float cvt_f32_f16_1_9;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_9) : "h"((uint16_t)(e4m3x2_to_f16x2_9 >> 16)));
                float s0_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_4) : "f"(cvt_f32_f16_10), "f"(rs_s_0));
                float s1_4;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_4) : "f"(cvt_f32_f16_1_9), "f"(rs_s_0));
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
                uint32_t e4m3x2_to_f16x2_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_10) : "h"((uint16_t)(fp4_wire_scales[6])));
                float cvt_f32_f16_11;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_11) : "h"((uint16_t)(e4m3x2_to_f16x2_10)));
                float cvt_f32_f16_1_10;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_10) : "h"((uint16_t)(e4m3x2_to_f16x2_10 >> 16)));
                float s0_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_5) : "f"(cvt_f32_f16_11), "f"(rs_s_1));
                float s1_5;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_5) : "f"(cvt_f32_f16_1_10), "f"(rs_s_1));
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
                uint32_t e4m3x2_to_f16x2_11;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_11) : "h"((uint16_t)(fp4_wire_scales[7])));
                float cvt_f32_f16_12;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_12) : "h"((uint16_t)(e4m3x2_to_f16x2_11)));
                float cvt_f32_f16_1_11;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_11) : "h"((uint16_t)(e4m3x2_to_f16x2_11 >> 16)));
                float s0_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_6) : "f"(cvt_f32_f16_12), "f"(rs_s_2));
                float s1_6;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_6) : "f"(cvt_f32_f16_1_11), "f"(rs_s_2));
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
                float acc_2[32] = {0};
                uint32_t e4m3x2_to_f16x2_12;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_12) : "h"((uint16_t)(fp4_wire_scales[8])));
                float cvt_f32_f16_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_13) : "h"((uint16_t)(e4m3x2_to_f16x2_12)));
                float cvt_f32_f16_1_12;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_12) : "h"((uint16_t)(e4m3x2_to_f16x2_12 >> 16)));
                float s0_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_7) : "f"(cvt_f32_f16_13), "f"(rs_s));
                float s1_7;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_7) : "f"(cvt_f32_f16_1_12), "f"(rs_s));
                float2 scale0_7 = make_float2(s0_7, s0_7);
                float2 scale1_7 = make_float2(s1_7, s1_7);
                float tv_32[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[32 + 0]));
                    (&tv_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_32[0]), "=f"(tv_32[1]), "=f"(tv_32[2]), "=f"(tv_32[3]), "=f"(tv_32[4]), "=f"(tv_32[5]), "=f"(tv_32[6]), "=f"(tv_32[7]) : "r"(fp4_wire_values[32 + 0]));
                    #endif
                }
                float2 a_128 = make_float2(acc_2[0], acc_2[1]);
                float2 q_128 = make_float2(tv_32[0], tv_32[1]);
                float2 r_128 = fma_f32x2_rn_noftz(q_128, scale0_7, a_128);
                acc_2[0] = r_128.x;
                acc_2[1] = r_128.y;
                float2 a_129 = make_float2(acc_2[2], acc_2[3]);
                float2 q_129 = make_float2(tv_32[2], tv_32[3]);
                float2 r_129 = fma_f32x2_rn_noftz(q_129, scale0_7, a_129);
                acc_2[2] = r_129.x;
                acc_2[3] = r_129.y;
                float2 a_130 = make_float2(acc_2[4], acc_2[5]);
                float2 q_130 = make_float2(tv_32[4], tv_32[5]);
                float2 r_130 = fma_f32x2_rn_noftz(q_130, scale0_7, a_130);
                acc_2[4] = r_130.x;
                acc_2[5] = r_130.y;
                float2 a_131 = make_float2(acc_2[6], acc_2[7]);
                float2 q_131 = make_float2(tv_32[6], tv_32[7]);
                float2 r_131 = fma_f32x2_rn_noftz(q_131, scale0_7, a_131);
                acc_2[6] = r_131.x;
                acc_2[7] = r_131.y;
                float tv_33[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[33 + 0]));
                    (&tv_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_33[0]), "=f"(tv_33[1]), "=f"(tv_33[2]), "=f"(tv_33[3]), "=f"(tv_33[4]), "=f"(tv_33[5]), "=f"(tv_33[6]), "=f"(tv_33[7]) : "r"(fp4_wire_values[33 + 0]));
                    #endif
                }
                float2 a_132 = make_float2(acc_2[8], acc_2[9]);
                float2 q_132 = make_float2(tv_33[0], tv_33[1]);
                float2 r_132 = fma_f32x2_rn_noftz(q_132, scale0_7, a_132);
                acc_2[8] = r_132.x;
                acc_2[9] = r_132.y;
                float2 a_133 = make_float2(acc_2[10], acc_2[11]);
                float2 q_133 = make_float2(tv_33[2], tv_33[3]);
                float2 r_133 = fma_f32x2_rn_noftz(q_133, scale0_7, a_133);
                acc_2[10] = r_133.x;
                acc_2[11] = r_133.y;
                float2 a_134 = make_float2(acc_2[12], acc_2[13]);
                float2 q_134 = make_float2(tv_33[4], tv_33[5]);
                float2 r_134 = fma_f32x2_rn_noftz(q_134, scale0_7, a_134);
                acc_2[12] = r_134.x;
                acc_2[13] = r_134.y;
                float2 a_135 = make_float2(acc_2[14], acc_2[15]);
                float2 q_135 = make_float2(tv_33[6], tv_33[7]);
                float2 r_135 = fma_f32x2_rn_noftz(q_135, scale0_7, a_135);
                acc_2[14] = r_135.x;
                acc_2[15] = r_135.y;
                float tv_34[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[34 + 0]));
                    (&tv_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_34[0]), "=f"(tv_34[1]), "=f"(tv_34[2]), "=f"(tv_34[3]), "=f"(tv_34[4]), "=f"(tv_34[5]), "=f"(tv_34[6]), "=f"(tv_34[7]) : "r"(fp4_wire_values[34 + 0]));
                    #endif
                }
                float2 a_136 = make_float2(acc_2[16], acc_2[17]);
                float2 q_136 = make_float2(tv_34[0], tv_34[1]);
                float2 r_136 = fma_f32x2_rn_noftz(q_136, scale1_7, a_136);
                acc_2[16] = r_136.x;
                acc_2[17] = r_136.y;
                float2 a_137 = make_float2(acc_2[18], acc_2[19]);
                float2 q_137 = make_float2(tv_34[2], tv_34[3]);
                float2 r_137 = fma_f32x2_rn_noftz(q_137, scale1_7, a_137);
                acc_2[18] = r_137.x;
                acc_2[19] = r_137.y;
                float2 a_138 = make_float2(acc_2[20], acc_2[21]);
                float2 q_138 = make_float2(tv_34[4], tv_34[5]);
                float2 r_138 = fma_f32x2_rn_noftz(q_138, scale1_7, a_138);
                acc_2[20] = r_138.x;
                acc_2[21] = r_138.y;
                float2 a_139 = make_float2(acc_2[22], acc_2[23]);
                float2 q_139 = make_float2(tv_34[6], tv_34[7]);
                float2 r_139 = fma_f32x2_rn_noftz(q_139, scale1_7, a_139);
                acc_2[22] = r_139.x;
                acc_2[23] = r_139.y;
                float tv_35[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[35 + 0]));
                    (&tv_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_35[0]), "=f"(tv_35[1]), "=f"(tv_35[2]), "=f"(tv_35[3]), "=f"(tv_35[4]), "=f"(tv_35[5]), "=f"(tv_35[6]), "=f"(tv_35[7]) : "r"(fp4_wire_values[35 + 0]));
                    #endif
                }
                float2 a_140 = make_float2(acc_2[24], acc_2[25]);
                float2 q_140 = make_float2(tv_35[0], tv_35[1]);
                float2 r_140 = fma_f32x2_rn_noftz(q_140, scale1_7, a_140);
                acc_2[24] = r_140.x;
                acc_2[25] = r_140.y;
                float2 a_141 = make_float2(acc_2[26], acc_2[27]);
                float2 q_141 = make_float2(tv_35[2], tv_35[3]);
                float2 r_141 = fma_f32x2_rn_noftz(q_141, scale1_7, a_141);
                acc_2[26] = r_141.x;
                acc_2[27] = r_141.y;
                float2 a_142 = make_float2(acc_2[28], acc_2[29]);
                float2 q_142 = make_float2(tv_35[4], tv_35[5]);
                float2 r_142 = fma_f32x2_rn_noftz(q_142, scale1_7, a_142);
                acc_2[28] = r_142.x;
                acc_2[29] = r_142.y;
                float2 a_143 = make_float2(acc_2[30], acc_2[31]);
                float2 q_143 = make_float2(tv_35[6], tv_35[7]);
                float2 r_143 = fma_f32x2_rn_noftz(q_143, scale1_7, a_143);
                acc_2[30] = r_143.x;
                acc_2[31] = r_143.y;
                uint32_t e4m3x2_to_f16x2_13;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_13) : "h"((uint16_t)(fp4_wire_scales[9])));
                float cvt_f32_f16_14;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_14) : "h"((uint16_t)(e4m3x2_to_f16x2_13)));
                float cvt_f32_f16_1_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_13) : "h"((uint16_t)(e4m3x2_to_f16x2_13 >> 16)));
                float s0_8;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_8) : "f"(cvt_f32_f16_14), "f"(rs_s_0));
                float s1_8;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_8) : "f"(cvt_f32_f16_1_13), "f"(rs_s_0));
                float2 scale0_8 = make_float2(s0_8, s0_8);
                float2 scale1_8 = make_float2(s1_8, s1_8);
                float tv_36[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[36 + 0]));
                    (&tv_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_36[0]), "=f"(tv_36[1]), "=f"(tv_36[2]), "=f"(tv_36[3]), "=f"(tv_36[4]), "=f"(tv_36[5]), "=f"(tv_36[6]), "=f"(tv_36[7]) : "r"(fp4_wire_values[36 + 0]));
                    #endif
                }
                float2 a_144 = make_float2(acc_2[0], acc_2[1]);
                float2 q_144 = make_float2(tv_36[0], tv_36[1]);
                float2 r_144 = fma_f32x2_rn_noftz(q_144, scale0_8, a_144);
                acc_2[0] = r_144.x;
                acc_2[1] = r_144.y;
                float2 a_145 = make_float2(acc_2[2], acc_2[3]);
                float2 q_145 = make_float2(tv_36[2], tv_36[3]);
                float2 r_145 = fma_f32x2_rn_noftz(q_145, scale0_8, a_145);
                acc_2[2] = r_145.x;
                acc_2[3] = r_145.y;
                float2 a_146 = make_float2(acc_2[4], acc_2[5]);
                float2 q_146 = make_float2(tv_36[4], tv_36[5]);
                float2 r_146 = fma_f32x2_rn_noftz(q_146, scale0_8, a_146);
                acc_2[4] = r_146.x;
                acc_2[5] = r_146.y;
                float2 a_147 = make_float2(acc_2[6], acc_2[7]);
                float2 q_147 = make_float2(tv_36[6], tv_36[7]);
                float2 r_147 = fma_f32x2_rn_noftz(q_147, scale0_8, a_147);
                acc_2[6] = r_147.x;
                acc_2[7] = r_147.y;
                float tv_37[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[37 + 0]));
                    (&tv_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_37[0]), "=f"(tv_37[1]), "=f"(tv_37[2]), "=f"(tv_37[3]), "=f"(tv_37[4]), "=f"(tv_37[5]), "=f"(tv_37[6]), "=f"(tv_37[7]) : "r"(fp4_wire_values[37 + 0]));
                    #endif
                }
                float2 a_148 = make_float2(acc_2[8], acc_2[9]);
                float2 q_148 = make_float2(tv_37[0], tv_37[1]);
                float2 r_148 = fma_f32x2_rn_noftz(q_148, scale0_8, a_148);
                acc_2[8] = r_148.x;
                acc_2[9] = r_148.y;
                float2 a_149 = make_float2(acc_2[10], acc_2[11]);
                float2 q_149 = make_float2(tv_37[2], tv_37[3]);
                float2 r_149 = fma_f32x2_rn_noftz(q_149, scale0_8, a_149);
                acc_2[10] = r_149.x;
                acc_2[11] = r_149.y;
                float2 a_150 = make_float2(acc_2[12], acc_2[13]);
                float2 q_150 = make_float2(tv_37[4], tv_37[5]);
                float2 r_150 = fma_f32x2_rn_noftz(q_150, scale0_8, a_150);
                acc_2[12] = r_150.x;
                acc_2[13] = r_150.y;
                float2 a_151 = make_float2(acc_2[14], acc_2[15]);
                float2 q_151 = make_float2(tv_37[6], tv_37[7]);
                float2 r_151 = fma_f32x2_rn_noftz(q_151, scale0_8, a_151);
                acc_2[14] = r_151.x;
                acc_2[15] = r_151.y;
                float tv_38[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[38 + 0]));
                    (&tv_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_38[0]), "=f"(tv_38[1]), "=f"(tv_38[2]), "=f"(tv_38[3]), "=f"(tv_38[4]), "=f"(tv_38[5]), "=f"(tv_38[6]), "=f"(tv_38[7]) : "r"(fp4_wire_values[38 + 0]));
                    #endif
                }
                float2 a_152 = make_float2(acc_2[16], acc_2[17]);
                float2 q_152 = make_float2(tv_38[0], tv_38[1]);
                float2 r_152 = fma_f32x2_rn_noftz(q_152, scale1_8, a_152);
                acc_2[16] = r_152.x;
                acc_2[17] = r_152.y;
                float2 a_153 = make_float2(acc_2[18], acc_2[19]);
                float2 q_153 = make_float2(tv_38[2], tv_38[3]);
                float2 r_153 = fma_f32x2_rn_noftz(q_153, scale1_8, a_153);
                acc_2[18] = r_153.x;
                acc_2[19] = r_153.y;
                float2 a_154 = make_float2(acc_2[20], acc_2[21]);
                float2 q_154 = make_float2(tv_38[4], tv_38[5]);
                float2 r_154 = fma_f32x2_rn_noftz(q_154, scale1_8, a_154);
                acc_2[20] = r_154.x;
                acc_2[21] = r_154.y;
                float2 a_155 = make_float2(acc_2[22], acc_2[23]);
                float2 q_155 = make_float2(tv_38[6], tv_38[7]);
                float2 r_155 = fma_f32x2_rn_noftz(q_155, scale1_8, a_155);
                acc_2[22] = r_155.x;
                acc_2[23] = r_155.y;
                float tv_39[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[39 + 0]));
                    (&tv_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_39[0]), "=f"(tv_39[1]), "=f"(tv_39[2]), "=f"(tv_39[3]), "=f"(tv_39[4]), "=f"(tv_39[5]), "=f"(tv_39[6]), "=f"(tv_39[7]) : "r"(fp4_wire_values[39 + 0]));
                    #endif
                }
                float2 a_156 = make_float2(acc_2[24], acc_2[25]);
                float2 q_156 = make_float2(tv_39[0], tv_39[1]);
                float2 r_156 = fma_f32x2_rn_noftz(q_156, scale1_8, a_156);
                acc_2[24] = r_156.x;
                acc_2[25] = r_156.y;
                float2 a_157 = make_float2(acc_2[26], acc_2[27]);
                float2 q_157 = make_float2(tv_39[2], tv_39[3]);
                float2 r_157 = fma_f32x2_rn_noftz(q_157, scale1_8, a_157);
                acc_2[26] = r_157.x;
                acc_2[27] = r_157.y;
                float2 a_158 = make_float2(acc_2[28], acc_2[29]);
                float2 q_158 = make_float2(tv_39[4], tv_39[5]);
                float2 r_158 = fma_f32x2_rn_noftz(q_158, scale1_8, a_158);
                acc_2[28] = r_158.x;
                acc_2[29] = r_158.y;
                float2 a_159 = make_float2(acc_2[30], acc_2[31]);
                float2 q_159 = make_float2(tv_39[6], tv_39[7]);
                float2 r_159 = fma_f32x2_rn_noftz(q_159, scale1_8, a_159);
                acc_2[30] = r_159.x;
                acc_2[31] = r_159.y;
                uint32_t e4m3x2_to_f16x2_14;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_14) : "h"((uint16_t)(fp4_wire_scales[10])));
                float cvt_f32_f16_15;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_15) : "h"((uint16_t)(e4m3x2_to_f16x2_14)));
                float cvt_f32_f16_1_14;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_14) : "h"((uint16_t)(e4m3x2_to_f16x2_14 >> 16)));
                float s0_9;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_9) : "f"(cvt_f32_f16_15), "f"(rs_s_1));
                float s1_9;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_9) : "f"(cvt_f32_f16_1_14), "f"(rs_s_1));
                float2 scale0_9 = make_float2(s0_9, s0_9);
                float2 scale1_9 = make_float2(s1_9, s1_9);
                float tv_40[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[40 + 0]));
                    (&tv_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_40[0]), "=f"(tv_40[1]), "=f"(tv_40[2]), "=f"(tv_40[3]), "=f"(tv_40[4]), "=f"(tv_40[5]), "=f"(tv_40[6]), "=f"(tv_40[7]) : "r"(fp4_wire_values[40 + 0]));
                    #endif
                }
                float2 a_160 = make_float2(acc_2[0], acc_2[1]);
                float2 q_160 = make_float2(tv_40[0], tv_40[1]);
                float2 r_160 = fma_f32x2_rn_noftz(q_160, scale0_9, a_160);
                acc_2[0] = r_160.x;
                acc_2[1] = r_160.y;
                float2 a_161 = make_float2(acc_2[2], acc_2[3]);
                float2 q_161 = make_float2(tv_40[2], tv_40[3]);
                float2 r_161 = fma_f32x2_rn_noftz(q_161, scale0_9, a_161);
                acc_2[2] = r_161.x;
                acc_2[3] = r_161.y;
                float2 a_162 = make_float2(acc_2[4], acc_2[5]);
                float2 q_162 = make_float2(tv_40[4], tv_40[5]);
                float2 r_162 = fma_f32x2_rn_noftz(q_162, scale0_9, a_162);
                acc_2[4] = r_162.x;
                acc_2[5] = r_162.y;
                float2 a_163 = make_float2(acc_2[6], acc_2[7]);
                float2 q_163 = make_float2(tv_40[6], tv_40[7]);
                float2 r_163 = fma_f32x2_rn_noftz(q_163, scale0_9, a_163);
                acc_2[6] = r_163.x;
                acc_2[7] = r_163.y;
                float tv_41[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[41 + 0]));
                    (&tv_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_41[0]), "=f"(tv_41[1]), "=f"(tv_41[2]), "=f"(tv_41[3]), "=f"(tv_41[4]), "=f"(tv_41[5]), "=f"(tv_41[6]), "=f"(tv_41[7]) : "r"(fp4_wire_values[41 + 0]));
                    #endif
                }
                float2 a_164 = make_float2(acc_2[8], acc_2[9]);
                float2 q_164 = make_float2(tv_41[0], tv_41[1]);
                float2 r_164 = fma_f32x2_rn_noftz(q_164, scale0_9, a_164);
                acc_2[8] = r_164.x;
                acc_2[9] = r_164.y;
                float2 a_165 = make_float2(acc_2[10], acc_2[11]);
                float2 q_165 = make_float2(tv_41[2], tv_41[3]);
                float2 r_165 = fma_f32x2_rn_noftz(q_165, scale0_9, a_165);
                acc_2[10] = r_165.x;
                acc_2[11] = r_165.y;
                float2 a_166 = make_float2(acc_2[12], acc_2[13]);
                float2 q_166 = make_float2(tv_41[4], tv_41[5]);
                float2 r_166 = fma_f32x2_rn_noftz(q_166, scale0_9, a_166);
                acc_2[12] = r_166.x;
                acc_2[13] = r_166.y;
                float2 a_167 = make_float2(acc_2[14], acc_2[15]);
                float2 q_167 = make_float2(tv_41[6], tv_41[7]);
                float2 r_167 = fma_f32x2_rn_noftz(q_167, scale0_9, a_167);
                acc_2[14] = r_167.x;
                acc_2[15] = r_167.y;
                float tv_42[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[42 + 0]));
                    (&tv_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_42[0]), "=f"(tv_42[1]), "=f"(tv_42[2]), "=f"(tv_42[3]), "=f"(tv_42[4]), "=f"(tv_42[5]), "=f"(tv_42[6]), "=f"(tv_42[7]) : "r"(fp4_wire_values[42 + 0]));
                    #endif
                }
                float2 a_168 = make_float2(acc_2[16], acc_2[17]);
                float2 q_168 = make_float2(tv_42[0], tv_42[1]);
                float2 r_168 = fma_f32x2_rn_noftz(q_168, scale1_9, a_168);
                acc_2[16] = r_168.x;
                acc_2[17] = r_168.y;
                float2 a_169 = make_float2(acc_2[18], acc_2[19]);
                float2 q_169 = make_float2(tv_42[2], tv_42[3]);
                float2 r_169 = fma_f32x2_rn_noftz(q_169, scale1_9, a_169);
                acc_2[18] = r_169.x;
                acc_2[19] = r_169.y;
                float2 a_170 = make_float2(acc_2[20], acc_2[21]);
                float2 q_170 = make_float2(tv_42[4], tv_42[5]);
                float2 r_170 = fma_f32x2_rn_noftz(q_170, scale1_9, a_170);
                acc_2[20] = r_170.x;
                acc_2[21] = r_170.y;
                float2 a_171 = make_float2(acc_2[22], acc_2[23]);
                float2 q_171 = make_float2(tv_42[6], tv_42[7]);
                float2 r_171 = fma_f32x2_rn_noftz(q_171, scale1_9, a_171);
                acc_2[22] = r_171.x;
                acc_2[23] = r_171.y;
                float tv_43[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[43 + 0]));
                    (&tv_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_43[0]), "=f"(tv_43[1]), "=f"(tv_43[2]), "=f"(tv_43[3]), "=f"(tv_43[4]), "=f"(tv_43[5]), "=f"(tv_43[6]), "=f"(tv_43[7]) : "r"(fp4_wire_values[43 + 0]));
                    #endif
                }
                float2 a_172 = make_float2(acc_2[24], acc_2[25]);
                float2 q_172 = make_float2(tv_43[0], tv_43[1]);
                float2 r_172 = fma_f32x2_rn_noftz(q_172, scale1_9, a_172);
                acc_2[24] = r_172.x;
                acc_2[25] = r_172.y;
                float2 a_173 = make_float2(acc_2[26], acc_2[27]);
                float2 q_173 = make_float2(tv_43[2], tv_43[3]);
                float2 r_173 = fma_f32x2_rn_noftz(q_173, scale1_9, a_173);
                acc_2[26] = r_173.x;
                acc_2[27] = r_173.y;
                float2 a_174 = make_float2(acc_2[28], acc_2[29]);
                float2 q_174 = make_float2(tv_43[4], tv_43[5]);
                float2 r_174 = fma_f32x2_rn_noftz(q_174, scale1_9, a_174);
                acc_2[28] = r_174.x;
                acc_2[29] = r_174.y;
                float2 a_175 = make_float2(acc_2[30], acc_2[31]);
                float2 q_175 = make_float2(tv_43[6], tv_43[7]);
                float2 r_175 = fma_f32x2_rn_noftz(q_175, scale1_9, a_175);
                acc_2[30] = r_175.x;
                acc_2[31] = r_175.y;
                uint32_t e4m3x2_to_f16x2_15;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_15) : "h"((uint16_t)(fp4_wire_scales[11])));
                float cvt_f32_f16_16;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_16) : "h"((uint16_t)(e4m3x2_to_f16x2_15)));
                float cvt_f32_f16_1_15;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_15) : "h"((uint16_t)(e4m3x2_to_f16x2_15 >> 16)));
                float s0_10;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_10) : "f"(cvt_f32_f16_16), "f"(rs_s_2));
                float s1_10;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_10) : "f"(cvt_f32_f16_1_15), "f"(rs_s_2));
                float2 scale0_10 = make_float2(s0_10, s0_10);
                float2 scale1_10 = make_float2(s1_10, s1_10);
                float tv_44[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[44 + 0]));
                    (&tv_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_44[0]), "=f"(tv_44[1]), "=f"(tv_44[2]), "=f"(tv_44[3]), "=f"(tv_44[4]), "=f"(tv_44[5]), "=f"(tv_44[6]), "=f"(tv_44[7]) : "r"(fp4_wire_values[44 + 0]));
                    #endif
                }
                float2 a_176 = make_float2(acc_2[0], acc_2[1]);
                float2 q_176 = make_float2(tv_44[0], tv_44[1]);
                float2 r_176 = fma_f32x2_rn_noftz(q_176, scale0_10, a_176);
                acc_2[0] = r_176.x;
                acc_2[1] = r_176.y;
                float2 a_177 = make_float2(acc_2[2], acc_2[3]);
                float2 q_177 = make_float2(tv_44[2], tv_44[3]);
                float2 r_177 = fma_f32x2_rn_noftz(q_177, scale0_10, a_177);
                acc_2[2] = r_177.x;
                acc_2[3] = r_177.y;
                float2 a_178 = make_float2(acc_2[4], acc_2[5]);
                float2 q_178 = make_float2(tv_44[4], tv_44[5]);
                float2 r_178 = fma_f32x2_rn_noftz(q_178, scale0_10, a_178);
                acc_2[4] = r_178.x;
                acc_2[5] = r_178.y;
                float2 a_179 = make_float2(acc_2[6], acc_2[7]);
                float2 q_179 = make_float2(tv_44[6], tv_44[7]);
                float2 r_179 = fma_f32x2_rn_noftz(q_179, scale0_10, a_179);
                acc_2[6] = r_179.x;
                acc_2[7] = r_179.y;
                float tv_45[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[45 + 0]));
                    (&tv_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_45[0]), "=f"(tv_45[1]), "=f"(tv_45[2]), "=f"(tv_45[3]), "=f"(tv_45[4]), "=f"(tv_45[5]), "=f"(tv_45[6]), "=f"(tv_45[7]) : "r"(fp4_wire_values[45 + 0]));
                    #endif
                }
                float2 a_180 = make_float2(acc_2[8], acc_2[9]);
                float2 q_180 = make_float2(tv_45[0], tv_45[1]);
                float2 r_180 = fma_f32x2_rn_noftz(q_180, scale0_10, a_180);
                acc_2[8] = r_180.x;
                acc_2[9] = r_180.y;
                float2 a_181 = make_float2(acc_2[10], acc_2[11]);
                float2 q_181 = make_float2(tv_45[2], tv_45[3]);
                float2 r_181 = fma_f32x2_rn_noftz(q_181, scale0_10, a_181);
                acc_2[10] = r_181.x;
                acc_2[11] = r_181.y;
                float2 a_182 = make_float2(acc_2[12], acc_2[13]);
                float2 q_182 = make_float2(tv_45[4], tv_45[5]);
                float2 r_182 = fma_f32x2_rn_noftz(q_182, scale0_10, a_182);
                acc_2[12] = r_182.x;
                acc_2[13] = r_182.y;
                float2 a_183 = make_float2(acc_2[14], acc_2[15]);
                float2 q_183 = make_float2(tv_45[6], tv_45[7]);
                float2 r_183 = fma_f32x2_rn_noftz(q_183, scale0_10, a_183);
                acc_2[14] = r_183.x;
                acc_2[15] = r_183.y;
                float tv_46[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[46 + 0]));
                    (&tv_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_46[0]), "=f"(tv_46[1]), "=f"(tv_46[2]), "=f"(tv_46[3]), "=f"(tv_46[4]), "=f"(tv_46[5]), "=f"(tv_46[6]), "=f"(tv_46[7]) : "r"(fp4_wire_values[46 + 0]));
                    #endif
                }
                float2 a_184 = make_float2(acc_2[16], acc_2[17]);
                float2 q_184 = make_float2(tv_46[0], tv_46[1]);
                float2 r_184 = fma_f32x2_rn_noftz(q_184, scale1_10, a_184);
                acc_2[16] = r_184.x;
                acc_2[17] = r_184.y;
                float2 a_185 = make_float2(acc_2[18], acc_2[19]);
                float2 q_185 = make_float2(tv_46[2], tv_46[3]);
                float2 r_185 = fma_f32x2_rn_noftz(q_185, scale1_10, a_185);
                acc_2[18] = r_185.x;
                acc_2[19] = r_185.y;
                float2 a_186 = make_float2(acc_2[20], acc_2[21]);
                float2 q_186 = make_float2(tv_46[4], tv_46[5]);
                float2 r_186 = fma_f32x2_rn_noftz(q_186, scale1_10, a_186);
                acc_2[20] = r_186.x;
                acc_2[21] = r_186.y;
                float2 a_187 = make_float2(acc_2[22], acc_2[23]);
                float2 q_187 = make_float2(tv_46[6], tv_46[7]);
                float2 r_187 = fma_f32x2_rn_noftz(q_187, scale1_10, a_187);
                acc_2[22] = r_187.x;
                acc_2[23] = r_187.y;
                float tv_47[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values[47 + 0]));
                    (&tv_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_47[0]), "=f"(tv_47[1]), "=f"(tv_47[2]), "=f"(tv_47[3]), "=f"(tv_47[4]), "=f"(tv_47[5]), "=f"(tv_47[6]), "=f"(tv_47[7]) : "r"(fp4_wire_values[47 + 0]));
                    #endif
                }
                float2 a_188 = make_float2(acc_2[24], acc_2[25]);
                float2 q_188 = make_float2(tv_47[0], tv_47[1]);
                float2 r_188 = fma_f32x2_rn_noftz(q_188, scale1_10, a_188);
                acc_2[24] = r_188.x;
                acc_2[25] = r_188.y;
                float2 a_189 = make_float2(acc_2[26], acc_2[27]);
                float2 q_189 = make_float2(tv_47[2], tv_47[3]);
                float2 r_189 = fma_f32x2_rn_noftz(q_189, scale1_10, a_189);
                acc_2[26] = r_189.x;
                acc_2[27] = r_189.y;
                float2 a_190 = make_float2(acc_2[28], acc_2[29]);
                float2 q_190 = make_float2(tv_47[4], tv_47[5]);
                float2 r_190 = fma_f32x2_rn_noftz(q_190, scale1_10, a_190);
                acc_2[28] = r_190.x;
                acc_2[29] = r_190.y;
                float2 a_191 = make_float2(acc_2[30], acc_2[31]);
                float2 q_191 = make_float2(tv_47[6], tv_47[7]);
                float2 r_191 = fma_f32x2_rn_noftz(q_191, scale1_10, a_191);
                acc_2[30] = r_191.x;
                acc_2[31] = r_191.y;
                float am_1 = 0.0f;
                float cur = am_1;
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
                am_1 = cur;
                float cur_0 = am_1;
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
                am_1 = cur_0;
                float cur_1 = am_1;
                cur_1 = fmaxf(cur_1, fabsf(acc_2[0]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[1]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[2]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[3]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[4]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[5]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[6]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[7]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[8]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[9]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[10]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[11]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[12]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[13]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[14]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[15]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[16]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[17]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[18]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[19]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[20]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[21]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[22]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[23]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[24]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[25]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[26]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[27]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[28]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[29]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[30]));
                cur_1 = fmaxf(cur_1, fabsf(acc_2[31]));
                am_1 = cur_1;
                float _warp_reduce_144 = am_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_144 = max_noftz(_warp_reduce_144, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_144, offset));
                am_1 = _warp_reduce_144;
                int sam_base = 4 * t + 2 * kbp;
                if (lane_1 == 0) {
                    s_am[sam_base + h] = am_1;
                }
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                float am2 = fmaxf(s_am[sam_base], s_am[sam_base + 1]);
                float rs2 = 1.0f;
                float fdiv_rn_7 = __fdiv_rn(2688.0f, am2);
                float inv_1 = 0.0f;
                if (am2 > 0.0f) {
                    inv_1 = fdiv_rn_7;
                }
                float rs_1 = __fdiv_rn(am2, 2688.0f);
                rs2 = rs_1;
                int pp2_0 = lane_1 + 32 * (3 * h);
                unsigned int q[4];
                float bm0 = 0.0f;
                float bm1 = 0.0f;
                bm0 = fmaxf(bm0, fabsf(acc_0[0]));
                bm1 = fmaxf(bm1, fabsf(acc_0[16]));
                bm0 = fmaxf(bm0, fabsf(acc_0[1]));
                bm1 = fmaxf(bm1, fabsf(acc_0[17]));
                bm0 = fmaxf(bm0, fabsf(acc_0[2]));
                bm1 = fmaxf(bm1, fabsf(acc_0[18]));
                bm0 = fmaxf(bm0, fabsf(acc_0[3]));
                bm1 = fmaxf(bm1, fabsf(acc_0[19]));
                bm0 = fmaxf(bm0, fabsf(acc_0[4]));
                bm1 = fmaxf(bm1, fabsf(acc_0[20]));
                bm0 = fmaxf(bm0, fabsf(acc_0[5]));
                bm1 = fmaxf(bm1, fabsf(acc_0[21]));
                bm0 = fmaxf(bm0, fabsf(acc_0[6]));
                bm1 = fmaxf(bm1, fabsf(acc_0[22]));
                bm0 = fmaxf(bm0, fabsf(acc_0[7]));
                bm1 = fmaxf(bm1, fabsf(acc_0[23]));
                bm0 = fmaxf(bm0, fabsf(acc_0[8]));
                bm1 = fmaxf(bm1, fabsf(acc_0[24]));
                bm0 = fmaxf(bm0, fabsf(acc_0[9]));
                bm1 = fmaxf(bm1, fabsf(acc_0[25]));
                bm0 = fmaxf(bm0, fabsf(acc_0[10]));
                bm1 = fmaxf(bm1, fabsf(acc_0[26]));
                bm0 = fmaxf(bm0, fabsf(acc_0[11]));
                bm1 = fmaxf(bm1, fabsf(acc_0[27]));
                bm0 = fmaxf(bm0, fabsf(acc_0[12]));
                bm1 = fmaxf(bm1, fabsf(acc_0[28]));
                bm0 = fmaxf(bm0, fabsf(acc_0[13]));
                bm1 = fmaxf(bm1, fabsf(acc_0[29]));
                bm0 = fmaxf(bm0, fabsf(acc_0[14]));
                bm1 = fmaxf(bm1, fabsf(acc_0[30]));
                bm0 = fmaxf(bm0, fabsf(acc_0[15]));
                bm1 = fmaxf(bm1, fabsf(acc_0[31]));
                float _fdiv_145 = __fdiv_rn(mul_f32_rn_noftz(bm1, inv_1), 6.0f);
                float _fdiv_146 = __fdiv_rn(mul_f32_rn_noftz(bm0, inv_1), 6.0f);
                uint16_t e4m3x2_f32_5;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_5) : "f"(_fdiv_145), "f"(_fdiv_146));
                uint32_t e4m3x2_to_f16x2_16;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_16) : "h"((uint16_t)(e4m3x2_f32_5)));
                float cvt_f32_f16_17;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_17) : "h"((uint16_t)(e4m3x2_to_f16x2_16)));
                float cvt_f32_f16_1_16;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_16) : "h"((uint16_t)(e4m3x2_to_f16x2_16 >> 16)));
                float fdiv_rn_7_1 = __fdiv_rn(inv_1, cvt_f32_f16_17);
                float fdiv_rn_1_5 = __fdiv_rn(inv_1, cvt_f32_f16_1_16);
                float m0_5 = 0.0f;
                float m1_5 = 0.0f;
                if (cvt_f32_f16_17 > 0.0f) {
                    m0_5 = fdiv_rn_7_1;
                }
                if (cvt_f32_f16_1_16 > 0.0f) {
                    m1_5 = fdiv_rn_1_5;
                }
                float2 mm_23 = make_float2(m0_5, m0_5);
                float2 pair_26 = make_float2(acc_0[0], acc_0[1]);
                float2 g_26;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_26) : "l"(*(const unsigned long long*)&pair_26), "l"(*(const unsigned long long*)&mm_23));
                uint32_t fp4_0;
                uint16_t _e2m1_pair_h_147;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_147) : "f"(g_26.x), "f"(g_26.y));
                fp4_0 = (uint32_t)_e2m1_pair_h_147;
                float2 pair_27 = make_float2(acc_0[2], acc_0[3]);
                float2 g_27;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_27) : "l"(*(const unsigned long long*)&pair_27), "l"(*(const unsigned long long*)&mm_23));
                uint32_t fp4_1;
                uint16_t _e2m1_pair_h_148;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_148) : "f"(g_27.x), "f"(g_27.y));
                fp4_1 = (uint32_t)_e2m1_pair_h_148;
                float2 pair_28 = make_float2(acc_0[4], acc_0[5]);
                float2 g_28;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_28) : "l"(*(const unsigned long long*)&pair_28), "l"(*(const unsigned long long*)&mm_23));
                uint32_t fp4_2;
                uint16_t _e2m1_pair_h_149;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_149) : "f"(g_28.x), "f"(g_28.y));
                fp4_2 = (uint32_t)_e2m1_pair_h_149;
                float2 pair_29 = make_float2(acc_0[6], acc_0[7]);
                float2 g_29;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_29) : "l"(*(const unsigned long long*)&pair_29), "l"(*(const unsigned long long*)&mm_23));
                uint32_t _fp4_e2m1_pack_pair_150;
                uint16_t _e2m1_pair_h_151;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_151) : "f"(g_29.x), "f"(g_29.y));
                _fp4_e2m1_pack_pair_150 = (uint32_t)_e2m1_pair_h_151;
                unsigned int word_23 = fp4_0 | fp4_1 << 8 | fp4_2 << 16 | _fp4_e2m1_pack_pair_150 << 24;
                unsigned int carry_23 = (word_23 & 2004318071) + 2004318071;
                q[0] = word_23 & (carry_23 | 2004318071);
                float2 mm_24 = make_float2(m0_5, m0_5);
                float2 pair_30 = make_float2(acc_0[8], acc_0[9]);
                float2 g_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_30) : "l"(*(const unsigned long long*)&pair_30), "l"(*(const unsigned long long*)&mm_24));
                uint32_t fp4_3;
                uint16_t _e2m1_pair_h_152;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_152) : "f"(g_30.x), "f"(g_30.y));
                fp4_3 = (uint32_t)_e2m1_pair_h_152;
                float2 pair_31 = make_float2(acc_0[10], acc_0[11]);
                float2 g_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_31) : "l"(*(const unsigned long long*)&pair_31), "l"(*(const unsigned long long*)&mm_24));
                uint32_t fp4_4;
                uint16_t _e2m1_pair_h_153;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_153) : "f"(g_31.x), "f"(g_31.y));
                fp4_4 = (uint32_t)_e2m1_pair_h_153;
                float2 pair_32 = make_float2(acc_0[12], acc_0[13]);
                float2 g_32;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_32) : "l"(*(const unsigned long long*)&pair_32), "l"(*(const unsigned long long*)&mm_24));
                uint32_t fp4_5;
                uint16_t _e2m1_pair_h_154;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_154) : "f"(g_32.x), "f"(g_32.y));
                fp4_5 = (uint32_t)_e2m1_pair_h_154;
                float2 pair_33 = make_float2(acc_0[14], acc_0[15]);
                float2 g_33;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_33) : "l"(*(const unsigned long long*)&pair_33), "l"(*(const unsigned long long*)&mm_24));
                uint32_t _fp4_e2m1_pack_pair_155;
                uint16_t _e2m1_pair_h_156;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_156) : "f"(g_33.x), "f"(g_33.y));
                _fp4_e2m1_pack_pair_155 = (uint32_t)_e2m1_pair_h_156;
                unsigned int word_24 = fp4_3 | fp4_4 << 8 | fp4_5 << 16 | _fp4_e2m1_pack_pair_155 << 24;
                unsigned int carry_23_1 = (word_24 & 2004318071) + 2004318071;
                q[1] = word_24 & (carry_23_1 | 2004318071);
                float2 mm_25 = make_float2(m1_5, m1_5);
                float2 pair_34 = make_float2(acc_0[16], acc_0[17]);
                float2 g_34;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_34) : "l"(*(const unsigned long long*)&pair_34), "l"(*(const unsigned long long*)&mm_25));
                uint32_t fp4_6;
                uint16_t _e2m1_pair_h_157;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_157) : "f"(g_34.x), "f"(g_34.y));
                fp4_6 = (uint32_t)_e2m1_pair_h_157;
                float2 pair_35 = make_float2(acc_0[18], acc_0[19]);
                float2 g_35;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_35) : "l"(*(const unsigned long long*)&pair_35), "l"(*(const unsigned long long*)&mm_25));
                uint32_t fp4_7;
                uint16_t _e2m1_pair_h_158;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_158) : "f"(g_35.x), "f"(g_35.y));
                fp4_7 = (uint32_t)_e2m1_pair_h_158;
                float2 pair_36 = make_float2(acc_0[20], acc_0[21]);
                float2 g_36;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_36) : "l"(*(const unsigned long long*)&pair_36), "l"(*(const unsigned long long*)&mm_25));
                uint32_t fp4_8;
                uint16_t _e2m1_pair_h_159;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_159) : "f"(g_36.x), "f"(g_36.y));
                fp4_8 = (uint32_t)_e2m1_pair_h_159;
                float2 pair_37 = make_float2(acc_0[22], acc_0[23]);
                float2 g_37;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_37) : "l"(*(const unsigned long long*)&pair_37), "l"(*(const unsigned long long*)&mm_25));
                uint32_t _fp4_e2m1_pack_pair_160;
                uint16_t _e2m1_pair_h_161;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_161) : "f"(g_37.x), "f"(g_37.y));
                _fp4_e2m1_pack_pair_160 = (uint32_t)_e2m1_pair_h_161;
                unsigned int word_25 = fp4_6 | fp4_7 << 8 | fp4_8 << 16 | _fp4_e2m1_pack_pair_160 << 24;
                unsigned int carry_24 = (word_25 & 2004318071) + 2004318071;
                q[2] = word_25 & (carry_24 | 2004318071);
                float2 mm_26 = make_float2(m1_5, m1_5);
                float2 pair_38 = make_float2(acc_0[24], acc_0[25]);
                float2 g_38;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_38) : "l"(*(const unsigned long long*)&pair_38), "l"(*(const unsigned long long*)&mm_26));
                uint32_t fp4_9;
                uint16_t _e2m1_pair_h_162;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_162) : "f"(g_38.x), "f"(g_38.y));
                fp4_9 = (uint32_t)_e2m1_pair_h_162;
                float2 pair_39 = make_float2(acc_0[26], acc_0[27]);
                float2 g_39;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_39) : "l"(*(const unsigned long long*)&pair_39), "l"(*(const unsigned long long*)&mm_26));
                uint32_t fp4_10;
                uint16_t _e2m1_pair_h_163;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_163) : "f"(g_39.x), "f"(g_39.y));
                fp4_10 = (uint32_t)_e2m1_pair_h_163;
                float2 pair_40 = make_float2(acc_0[28], acc_0[29]);
                float2 g_40;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_40) : "l"(*(const unsigned long long*)&pair_40), "l"(*(const unsigned long long*)&mm_26));
                uint32_t fp4_11;
                uint16_t _e2m1_pair_h_164;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_164) : "f"(g_40.x), "f"(g_40.y));
                fp4_11 = (uint32_t)_e2m1_pair_h_164;
                float2 pair_41 = make_float2(acc_0[30], acc_0[31]);
                float2 g_41;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_41) : "l"(*(const unsigned long long*)&pair_41), "l"(*(const unsigned long long*)&mm_26));
                uint32_t _fp4_e2m1_pack_pair_165;
                uint16_t _e2m1_pair_h_166;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_166) : "f"(g_41.x), "f"(g_41.y));
                _fp4_e2m1_pack_pair_165 = (uint32_t)_e2m1_pair_h_166;
                unsigned int word_26 = fp4_9 | fp4_10 << 8 | fp4_11 << 16 | _fp4_e2m1_pack_pair_165 << 24;
                unsigned int carry_25 = (word_26 & 2004318071) + 2004318071;
                q[3] = word_26 & (carry_25 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_0)), "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2_0)), "h"((uint16_t)(e4m3x2_f32_5 & 32639)) : "memory");
                float d[32] = {0};
                uint32_t e4m3x2_to_f16x2_17;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_17) : "h"((uint16_t)(e4m3x2_f32_5 & 32639)));
                float cvt_f32_f16_18;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_18) : "h"((uint16_t)(e4m3x2_to_f16x2_17)));
                float cvt_f32_f16_1_17;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_17) : "h"((uint16_t)(e4m3x2_to_f16x2_17 >> 16)));
                float s0_11;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_11) : "f"(cvt_f32_f16_18), "f"(rs2));
                float s1_11;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_11) : "f"(cvt_f32_f16_1_17), "f"(rs2));
                float2 scale0_11 = make_float2(s0_11, s0_11);
                float2 scale1_11 = make_float2(s1_11, s1_11);
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
                float2 r_192 = fma_f32x2_rn_noftz(q_192, scale0_11, a_192);
                d[0] = r_192.x;
                d[1] = r_192.y;
                float2 a_193 = make_float2(d[2], d[3]);
                float2 q_193 = make_float2(tv_48[2], tv_48[3]);
                float2 r_193 = fma_f32x2_rn_noftz(q_193, scale0_11, a_193);
                d[2] = r_193.x;
                d[3] = r_193.y;
                float2 a_194 = make_float2(d[4], d[5]);
                float2 q_194 = make_float2(tv_48[4], tv_48[5]);
                float2 r_194 = fma_f32x2_rn_noftz(q_194, scale0_11, a_194);
                d[4] = r_194.x;
                d[5] = r_194.y;
                float2 a_195 = make_float2(d[6], d[7]);
                float2 q_195 = make_float2(tv_48[6], tv_48[7]);
                float2 r_195 = fma_f32x2_rn_noftz(q_195, scale0_11, a_195);
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
                float2 r_196 = fma_f32x2_rn_noftz(q_196, scale0_11, a_196);
                d[8] = r_196.x;
                d[9] = r_196.y;
                float2 a_197 = make_float2(d[10], d[11]);
                float2 q_197 = make_float2(tv_49[2], tv_49[3]);
                float2 r_197 = fma_f32x2_rn_noftz(q_197, scale0_11, a_197);
                d[10] = r_197.x;
                d[11] = r_197.y;
                float2 a_198 = make_float2(d[12], d[13]);
                float2 q_198 = make_float2(tv_49[4], tv_49[5]);
                float2 r_198 = fma_f32x2_rn_noftz(q_198, scale0_11, a_198);
                d[12] = r_198.x;
                d[13] = r_198.y;
                float2 a_199 = make_float2(d[14], d[15]);
                float2 q_199 = make_float2(tv_49[6], tv_49[7]);
                float2 r_199 = fma_f32x2_rn_noftz(q_199, scale0_11, a_199);
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
                float2 r_200 = fma_f32x2_rn_noftz(q_200, scale1_11, a_200);
                d[16] = r_200.x;
                d[17] = r_200.y;
                float2 a_201 = make_float2(d[18], d[19]);
                float2 q_201 = make_float2(tv_50[2], tv_50[3]);
                float2 r_201 = fma_f32x2_rn_noftz(q_201, scale1_11, a_201);
                d[18] = r_201.x;
                d[19] = r_201.y;
                float2 a_202 = make_float2(d[20], d[21]);
                float2 q_202 = make_float2(tv_50[4], tv_50[5]);
                float2 r_202 = fma_f32x2_rn_noftz(q_202, scale1_11, a_202);
                d[20] = r_202.x;
                d[21] = r_202.y;
                float2 a_203 = make_float2(d[22], d[23]);
                float2 q_203 = make_float2(tv_50[6], tv_50[7]);
                float2 r_203 = fma_f32x2_rn_noftz(q_203, scale1_11, a_203);
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
                float2 r_204 = fma_f32x2_rn_noftz(q_204, scale1_11, a_204);
                d[24] = r_204.x;
                d[25] = r_204.y;
                float2 a_205 = make_float2(d[26], d[27]);
                float2 q_205 = make_float2(tv_51[2], tv_51[3]);
                float2 r_205 = fma_f32x2_rn_noftz(q_205, scale1_11, a_205);
                d[26] = r_205.x;
                d[27] = r_205.y;
                float2 a_206 = make_float2(d[28], d[29]);
                float2 q_206 = make_float2(tv_51[4], tv_51[5]);
                float2 r_206 = fma_f32x2_rn_noftz(q_206, scale1_11, a_206);
                d[28] = r_206.x;
                d[29] = r_206.y;
                float2 a_207 = make_float2(d[30], d[31]);
                float2 q_207 = make_float2(tv_51[6], tv_51[7]);
                float2 r_207 = fma_f32x2_rn_noftz(q_207, scale1_11, a_207);
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
                int c_0 = lane_1;
                int l_0 = c_0 >> 2;
                unsigned int w4_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_0 + (c_0 & 3 ^ l_0 >> 1 & 3)))));
                long long idx_0 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_0));
                {
                    int4 _iv4 = make_int4(w4_0[0 + 0], w4_0[0 + 1], w4_0[0 + 2], w4_0[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_0) = _iv4;
                }
                int c_1 = lane_1 + 32;
                int l_1 = c_1 >> 2;
                unsigned int w4_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1 + (c_1 & 3 ^ l_1 >> 1 & 3)))));
                long long idx_1 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_1));
                {
                    int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1) = _iv4;
                }
                int c_2 = lane_1 + 64;
                int l_2 = c_2 >> 2;
                unsigned int w4_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_2 + (c_2 & 3 ^ l_2 >> 1 & 3)))));
                long long idx_2 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_2));
                {
                    int4 _iv4 = make_int4(w4_2[0 + 0], w4_2[0 + 1], w4_2[0 + 2], w4_2[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_2) = _iv4;
                }
                int c_3 = lane_1 + 96;
                int l_3 = c_3 >> 2;
                unsigned int w4_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_3 + (c_3 & 3 ^ l_3 >> 1 & 3)))));
                long long idx_3 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_3));
                {
                    int4 _iv4 = make_int4(w4_3[0 + 0], w4_3[0 + 1], w4_3[0 + 2], w4_3[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_3) = _iv4;
                }
                __syncwarp();
                int pp2_1 = lane_1 + 32 * (3 * h + 1);
                unsigned int q_208[4];
                float bm0_0 = 0.0f;
                float bm1_0 = 0.0f;
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[0]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[16]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[1]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[17]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[2]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[18]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[3]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[19]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[4]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[20]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[5]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[21]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[6]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[22]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[7]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[23]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[8]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[24]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[9]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[25]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[10]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[26]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[11]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[27]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[12]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[28]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[13]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[29]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[14]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[30]));
                bm0_0 = fmaxf(bm0_0, fabsf(acc_1[15]));
                bm1_0 = fmaxf(bm1_0, fabsf(acc_1[31]));
                float _fdiv_167 = __fdiv_rn(mul_f32_rn_noftz(bm1_0, inv_1), 6.0f);
                float _fdiv_168 = __fdiv_rn(mul_f32_rn_noftz(bm0_0, inv_1), 6.0f);
                uint16_t e4m3x2_f32_5_1;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_5_1) : "f"(_fdiv_167), "f"(_fdiv_168));
                uint32_t e4m3x2_to_f16x2_18;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_18) : "h"((uint16_t)(e4m3x2_f32_5_1)));
                float cvt_f32_f16_19;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_19) : "h"((uint16_t)(e4m3x2_to_f16x2_18)));
                float cvt_f32_f16_1_18;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_18) : "h"((uint16_t)(e4m3x2_to_f16x2_18 >> 16)));
                float fdiv_rn_8 = __fdiv_rn(inv_1, cvt_f32_f16_19);
                float fdiv_rn_1_5_1 = __fdiv_rn(inv_1, cvt_f32_f16_1_18);
                float m0_5_1 = 0.0f;
                float m1_5_1 = 0.0f;
                if (cvt_f32_f16_19 > 0.0f) {
                    m0_5_1 = fdiv_rn_8;
                }
                if (cvt_f32_f16_1_18 > 0.0f) {
                    m1_5_1 = fdiv_rn_1_5_1;
                }
                float2 mm_27 = make_float2(m0_5_1, m0_5_1);
                float2 pair_42 = make_float2(acc_1[0], acc_1[1]);
                float2 g_42;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_42) : "l"(*(const unsigned long long*)&pair_42), "l"(*(const unsigned long long*)&mm_27));
                uint32_t fp4_12;
                uint16_t _e2m1_pair_h_169;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_169) : "f"(g_42.x), "f"(g_42.y));
                fp4_12 = (uint32_t)_e2m1_pair_h_169;
                float2 pair_43 = make_float2(acc_1[2], acc_1[3]);
                float2 g_43;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_43) : "l"(*(const unsigned long long*)&pair_43), "l"(*(const unsigned long long*)&mm_27));
                uint32_t fp4_13;
                uint16_t _e2m1_pair_h_170;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_170) : "f"(g_43.x), "f"(g_43.y));
                fp4_13 = (uint32_t)_e2m1_pair_h_170;
                float2 pair_44 = make_float2(acc_1[4], acc_1[5]);
                float2 g_44;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_44) : "l"(*(const unsigned long long*)&pair_44), "l"(*(const unsigned long long*)&mm_27));
                uint32_t fp4_14;
                uint16_t _e2m1_pair_h_171;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_171) : "f"(g_44.x), "f"(g_44.y));
                fp4_14 = (uint32_t)_e2m1_pair_h_171;
                float2 pair_45 = make_float2(acc_1[6], acc_1[7]);
                float2 g_45;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_45) : "l"(*(const unsigned long long*)&pair_45), "l"(*(const unsigned long long*)&mm_27));
                uint32_t _fp4_e2m1_pack_pair_172;
                uint16_t _e2m1_pair_h_173;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_173) : "f"(g_45.x), "f"(g_45.y));
                _fp4_e2m1_pack_pair_172 = (uint32_t)_e2m1_pair_h_173;
                unsigned int word_27 = fp4_12 | fp4_13 << 8 | fp4_14 << 16 | _fp4_e2m1_pack_pair_172 << 24;
                unsigned int carry_26 = (word_27 & 2004318071) + 2004318071;
                q_208[0] = word_27 & (carry_26 | 2004318071);
                float2 mm_28 = make_float2(m0_5_1, m0_5_1);
                float2 pair_46 = make_float2(acc_1[8], acc_1[9]);
                float2 g_46;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_46) : "l"(*(const unsigned long long*)&pair_46), "l"(*(const unsigned long long*)&mm_28));
                uint32_t fp4_15;
                uint16_t _e2m1_pair_h_174;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_174) : "f"(g_46.x), "f"(g_46.y));
                fp4_15 = (uint32_t)_e2m1_pair_h_174;
                float2 pair_47 = make_float2(acc_1[10], acc_1[11]);
                float2 g_47;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_47) : "l"(*(const unsigned long long*)&pair_47), "l"(*(const unsigned long long*)&mm_28));
                uint32_t fp4_16;
                uint16_t _e2m1_pair_h_175;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_175) : "f"(g_47.x), "f"(g_47.y));
                fp4_16 = (uint32_t)_e2m1_pair_h_175;
                float2 pair_48 = make_float2(acc_1[12], acc_1[13]);
                float2 g_48;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_48) : "l"(*(const unsigned long long*)&pair_48), "l"(*(const unsigned long long*)&mm_28));
                uint32_t fp4_17;
                uint16_t _e2m1_pair_h_176;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_176) : "f"(g_48.x), "f"(g_48.y));
                fp4_17 = (uint32_t)_e2m1_pair_h_176;
                float2 pair_49 = make_float2(acc_1[14], acc_1[15]);
                float2 g_49;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_49) : "l"(*(const unsigned long long*)&pair_49), "l"(*(const unsigned long long*)&mm_28));
                uint32_t _fp4_e2m1_pack_pair_177;
                uint16_t _e2m1_pair_h_178;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_178) : "f"(g_49.x), "f"(g_49.y));
                _fp4_e2m1_pack_pair_177 = (uint32_t)_e2m1_pair_h_178;
                unsigned int word_28 = fp4_15 | fp4_16 << 8 | fp4_17 << 16 | _fp4_e2m1_pack_pair_177 << 24;
                unsigned int carry_27 = (word_28 & 2004318071) + 2004318071;
                q_208[1] = word_28 & (carry_27 | 2004318071);
                float2 mm_29 = make_float2(m1_5_1, m1_5_1);
                float2 pair_50 = make_float2(acc_1[16], acc_1[17]);
                float2 g_50;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_50) : "l"(*(const unsigned long long*)&pair_50), "l"(*(const unsigned long long*)&mm_29));
                uint32_t fp4_18;
                uint16_t _e2m1_pair_h_179;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_179) : "f"(g_50.x), "f"(g_50.y));
                fp4_18 = (uint32_t)_e2m1_pair_h_179;
                float2 pair_51 = make_float2(acc_1[18], acc_1[19]);
                float2 g_51;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_51) : "l"(*(const unsigned long long*)&pair_51), "l"(*(const unsigned long long*)&mm_29));
                uint32_t fp4_19;
                uint16_t _e2m1_pair_h_180;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_180) : "f"(g_51.x), "f"(g_51.y));
                fp4_19 = (uint32_t)_e2m1_pair_h_180;
                float2 pair_52 = make_float2(acc_1[20], acc_1[21]);
                float2 g_52;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_52) : "l"(*(const unsigned long long*)&pair_52), "l"(*(const unsigned long long*)&mm_29));
                uint32_t fp4_20;
                uint16_t _e2m1_pair_h_181;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_181) : "f"(g_52.x), "f"(g_52.y));
                fp4_20 = (uint32_t)_e2m1_pair_h_181;
                float2 pair_53 = make_float2(acc_1[22], acc_1[23]);
                float2 g_53;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_53) : "l"(*(const unsigned long long*)&pair_53), "l"(*(const unsigned long long*)&mm_29));
                uint32_t _fp4_e2m1_pack_pair_182;
                uint16_t _e2m1_pair_h_183;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_183) : "f"(g_53.x), "f"(g_53.y));
                _fp4_e2m1_pack_pair_182 = (uint32_t)_e2m1_pair_h_183;
                unsigned int word_29 = fp4_18 | fp4_19 << 8 | fp4_20 << 16 | _fp4_e2m1_pack_pair_182 << 24;
                unsigned int carry_28 = (word_29 & 2004318071) + 2004318071;
                q_208[2] = word_29 & (carry_28 | 2004318071);
                float2 mm_30 = make_float2(m1_5_1, m1_5_1);
                float2 pair_54 = make_float2(acc_1[24], acc_1[25]);
                float2 g_54;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_54) : "l"(*(const unsigned long long*)&pair_54), "l"(*(const unsigned long long*)&mm_30));
                uint32_t fp4_21;
                uint16_t _e2m1_pair_h_184;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_184) : "f"(g_54.x), "f"(g_54.y));
                fp4_21 = (uint32_t)_e2m1_pair_h_184;
                float2 pair_55 = make_float2(acc_1[26], acc_1[27]);
                float2 g_55;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_55) : "l"(*(const unsigned long long*)&pair_55), "l"(*(const unsigned long long*)&mm_30));
                uint32_t fp4_22;
                uint16_t _e2m1_pair_h_185;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_185) : "f"(g_55.x), "f"(g_55.y));
                fp4_22 = (uint32_t)_e2m1_pair_h_185;
                float2 pair_56 = make_float2(acc_1[28], acc_1[29]);
                float2 g_56;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_56) : "l"(*(const unsigned long long*)&pair_56), "l"(*(const unsigned long long*)&mm_30));
                uint32_t fp4_23;
                uint16_t _e2m1_pair_h_186;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_186) : "f"(g_56.x), "f"(g_56.y));
                fp4_23 = (uint32_t)_e2m1_pair_h_186;
                float2 pair_57 = make_float2(acc_1[30], acc_1[31]);
                float2 g_57;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_57) : "l"(*(const unsigned long long*)&pair_57), "l"(*(const unsigned long long*)&mm_30));
                uint32_t _fp4_e2m1_pack_pair_187;
                uint16_t _e2m1_pair_h_188;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_188) : "f"(g_57.x), "f"(g_57.y));
                _fp4_e2m1_pack_pair_187 = (uint32_t)_e2m1_pair_h_188;
                unsigned int word_30 = fp4_21 | fp4_22 << 8 | fp4_23 << 16 | _fp4_e2m1_pack_pair_187 << 24;
                unsigned int carry_29 = (word_30 & 2004318071) + 2004318071;
                q_208[3] = word_30 & (carry_29 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_1)), "r"(q_208[0]), "r"(q_208[1]), "r"(q_208[2]), "r"(q_208[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2_1)), "h"((uint16_t)(e4m3x2_f32_5_1 & 32639)) : "memory");
                float d_0[32] = {0};
                uint32_t e4m3x2_to_f16x2_19;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_19) : "h"((uint16_t)(e4m3x2_f32_5_1 & 32639)));
                float cvt_f32_f16_20;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_20) : "h"((uint16_t)(e4m3x2_to_f16x2_19)));
                float cvt_f32_f16_1_19;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_19) : "h"((uint16_t)(e4m3x2_to_f16x2_19 >> 16)));
                float s0_12;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_12) : "f"(cvt_f32_f16_20), "f"(rs2));
                float s1_12;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_12) : "f"(cvt_f32_f16_1_19), "f"(rs2));
                float2 scale0_12 = make_float2(s0_12, s0_12);
                float2 scale1_12 = make_float2(s1_12, s1_12);
                float tv_52[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_208[0]));
                    (&tv_52[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_52[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_52[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_52[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_52[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_52[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_52[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_52[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_52[0]), "=f"(tv_52[1]), "=f"(tv_52[2]), "=f"(tv_52[3]), "=f"(tv_52[4]), "=f"(tv_52[5]), "=f"(tv_52[6]), "=f"(tv_52[7]) : "r"(q_208[0]));
                    #endif
                }
                float2 a_208 = make_float2(d_0[0], d_0[1]);
                float2 q_209 = make_float2(tv_52[0], tv_52[1]);
                float2 r_208 = fma_f32x2_rn_noftz(q_209, scale0_12, a_208);
                d_0[0] = r_208.x;
                d_0[1] = r_208.y;
                float2 a_209 = make_float2(d_0[2], d_0[3]);
                float2 q_210 = make_float2(tv_52[2], tv_52[3]);
                float2 r_209 = fma_f32x2_rn_noftz(q_210, scale0_12, a_209);
                d_0[2] = r_209.x;
                d_0[3] = r_209.y;
                float2 a_210 = make_float2(d_0[4], d_0[5]);
                float2 q_211 = make_float2(tv_52[4], tv_52[5]);
                float2 r_210 = fma_f32x2_rn_noftz(q_211, scale0_12, a_210);
                d_0[4] = r_210.x;
                d_0[5] = r_210.y;
                float2 a_211 = make_float2(d_0[6], d_0[7]);
                float2 q_212 = make_float2(tv_52[6], tv_52[7]);
                float2 r_211 = fma_f32x2_rn_noftz(q_212, scale0_12, a_211);
                d_0[6] = r_211.x;
                d_0[7] = r_211.y;
                float tv_53[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_208[1 + 0]));
                    (&tv_53[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_53[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_53[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_53[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_53[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_53[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_53[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_53[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_53[0]), "=f"(tv_53[1]), "=f"(tv_53[2]), "=f"(tv_53[3]), "=f"(tv_53[4]), "=f"(tv_53[5]), "=f"(tv_53[6]), "=f"(tv_53[7]) : "r"(q_208[1 + 0]));
                    #endif
                }
                float2 a_212 = make_float2(d_0[8], d_0[9]);
                float2 q_213 = make_float2(tv_53[0], tv_53[1]);
                float2 r_212 = fma_f32x2_rn_noftz(q_213, scale0_12, a_212);
                d_0[8] = r_212.x;
                d_0[9] = r_212.y;
                float2 a_213 = make_float2(d_0[10], d_0[11]);
                float2 q_214 = make_float2(tv_53[2], tv_53[3]);
                float2 r_213 = fma_f32x2_rn_noftz(q_214, scale0_12, a_213);
                d_0[10] = r_213.x;
                d_0[11] = r_213.y;
                float2 a_214 = make_float2(d_0[12], d_0[13]);
                float2 q_215 = make_float2(tv_53[4], tv_53[5]);
                float2 r_214 = fma_f32x2_rn_noftz(q_215, scale0_12, a_214);
                d_0[12] = r_214.x;
                d_0[13] = r_214.y;
                float2 a_215 = make_float2(d_0[14], d_0[15]);
                float2 q_216 = make_float2(tv_53[6], tv_53[7]);
                float2 r_215 = fma_f32x2_rn_noftz(q_216, scale0_12, a_215);
                d_0[14] = r_215.x;
                d_0[15] = r_215.y;
                float tv_54[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_208[2 + 0]));
                    (&tv_54[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_54[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_54[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_54[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_54[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_54[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_54[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_54[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_54[0]), "=f"(tv_54[1]), "=f"(tv_54[2]), "=f"(tv_54[3]), "=f"(tv_54[4]), "=f"(tv_54[5]), "=f"(tv_54[6]), "=f"(tv_54[7]) : "r"(q_208[2 + 0]));
                    #endif
                }
                float2 a_216 = make_float2(d_0[16], d_0[17]);
                float2 q_217 = make_float2(tv_54[0], tv_54[1]);
                float2 r_216 = fma_f32x2_rn_noftz(q_217, scale1_12, a_216);
                d_0[16] = r_216.x;
                d_0[17] = r_216.y;
                float2 a_217 = make_float2(d_0[18], d_0[19]);
                float2 q_218 = make_float2(tv_54[2], tv_54[3]);
                float2 r_217 = fma_f32x2_rn_noftz(q_218, scale1_12, a_217);
                d_0[18] = r_217.x;
                d_0[19] = r_217.y;
                float2 a_218 = make_float2(d_0[20], d_0[21]);
                float2 q_219 = make_float2(tv_54[4], tv_54[5]);
                float2 r_218 = fma_f32x2_rn_noftz(q_219, scale1_12, a_218);
                d_0[20] = r_218.x;
                d_0[21] = r_218.y;
                float2 a_219 = make_float2(d_0[22], d_0[23]);
                float2 q_220 = make_float2(tv_54[6], tv_54[7]);
                float2 r_219 = fma_f32x2_rn_noftz(q_220, scale1_12, a_219);
                d_0[22] = r_219.x;
                d_0[23] = r_219.y;
                float tv_55[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_208[3 + 0]));
                    (&tv_55[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_55[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_55[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_55[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_55[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_55[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_55[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_55[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_55[0]), "=f"(tv_55[1]), "=f"(tv_55[2]), "=f"(tv_55[3]), "=f"(tv_55[4]), "=f"(tv_55[5]), "=f"(tv_55[6]), "=f"(tv_55[7]) : "r"(q_208[3 + 0]));
                    #endif
                }
                float2 a_220 = make_float2(d_0[24], d_0[25]);
                float2 q_221 = make_float2(tv_55[0], tv_55[1]);
                float2 r_220 = fma_f32x2_rn_noftz(q_221, scale1_12, a_220);
                d_0[24] = r_220.x;
                d_0[25] = r_220.y;
                float2 a_221 = make_float2(d_0[26], d_0[27]);
                float2 q_222 = make_float2(tv_55[2], tv_55[3]);
                float2 r_221 = fma_f32x2_rn_noftz(q_222, scale1_12, a_221);
                d_0[26] = r_221.x;
                d_0[27] = r_221.y;
                float2 a_222 = make_float2(d_0[28], d_0[29]);
                float2 q_223 = make_float2(tv_55[4], tv_55[5]);
                float2 r_222 = fma_f32x2_rn_noftz(q_223, scale1_12, a_222);
                d_0[28] = r_222.x;
                d_0[29] = r_222.y;
                float2 a_223 = make_float2(d_0[30], d_0[31]);
                float2 q_224 = make_float2(tv_55[6], tv_55[7]);
                float2 r_223 = fma_f32x2_rn_noftz(q_224, scale1_12, a_223);
                d_0[30] = r_223.x;
                d_0[31] = r_223.y;
                unsigned int d_bf16_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_0[_lp*2 + 0], d_0[_lp*2+1 + 0]));
                    d_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_bf16_0[0]), "r"(d_bf16_0[1]), "r"(d_bf16_0[2]), "r"(d_bf16_0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_bf16_0[4]), "r"(d_bf16_0[5]), "r"(d_bf16_0[6]), "r"(d_bf16_0[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_bf16_0[8]), "r"(d_bf16_0[9]), "r"(d_bf16_0[10]), "r"(d_bf16_0[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_bf16_0[12]), "r"(d_bf16_0[13]), "r"(d_bf16_0[14]), "r"(d_bf16_0[15]) : "memory");
                __syncwarp();
                int c_4 = lane_1;
                int l_4 = c_4 >> 2;
                unsigned int w4_4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_4 + (c_4 & 3 ^ l_4 >> 1 & 3)))));
                long long idx_4 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_4));
                {
                    int4 _iv4 = make_int4(w4_4[0 + 0], w4_4[0 + 1], w4_4[0 + 2], w4_4[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_4) = _iv4;
                }
                int c_5 = lane_1 + 32;
                int l_5 = c_5 >> 2;
                unsigned int w4_5[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_5 + (c_5 & 3 ^ l_5 >> 1 & 3)))));
                long long idx_5 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_5));
                {
                    int4 _iv4 = make_int4(w4_5[0 + 0], w4_5[0 + 1], w4_5[0 + 2], w4_5[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_5) = _iv4;
                }
                int c_6 = lane_1 + 64;
                int l_6 = c_6 >> 2;
                unsigned int w4_6[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_6 + (c_6 & 3 ^ l_6 >> 1 & 3)))));
                long long idx_6 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_6));
                {
                    int4 _iv4 = make_int4(w4_6[0 + 0], w4_6[0 + 1], w4_6[0 + 2], w4_6[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_6) = _iv4;
                }
                int c_7 = lane_1 + 96;
                int l_7 = c_7 >> 2;
                unsigned int w4_7[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_7 + (c_7 & 3 ^ l_7 >> 1 & 3)))));
                long long idx_7 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_7));
                {
                    int4 _iv4 = make_int4(w4_7[0 + 0], w4_7[0 + 1], w4_7[0 + 2], w4_7[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_7) = _iv4;
                }
                __syncwarp();
                int pp2_2 = lane_1 + 32 * (3 * h + 2);
                unsigned int q_225[4];
                float bm0_1 = 0.0f;
                float bm1_1 = 0.0f;
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[0]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[16]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[1]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[17]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[2]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[18]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[3]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[19]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[4]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[20]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[5]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[21]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[6]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[22]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[7]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[23]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[8]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[24]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[9]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[25]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[10]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[26]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[11]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[27]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[12]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[28]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[13]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[29]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[14]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[30]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc_2[15]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc_2[31]));
                float _fdiv_189 = __fdiv_rn(mul_f32_rn_noftz(bm1_1, inv_1), 6.0f);
                float _fdiv_190 = __fdiv_rn(mul_f32_rn_noftz(bm0_1, inv_1), 6.0f);
                uint16_t e4m3x2_f32_6;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_6) : "f"(_fdiv_189), "f"(_fdiv_190));
                uint32_t e4m3x2_to_f16x2_20;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_20) : "h"((uint16_t)(e4m3x2_f32_6)));
                float cvt_f32_f16_21;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_21) : "h"((uint16_t)(e4m3x2_to_f16x2_20)));
                float cvt_f32_f16_1_20;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_20) : "h"((uint16_t)(e4m3x2_to_f16x2_20 >> 16)));
                float fdiv_rn_9 = __fdiv_rn(inv_1, cvt_f32_f16_21);
                float fdiv_rn_1_6 = __fdiv_rn(inv_1, cvt_f32_f16_1_20);
                float m0_6 = 0.0f;
                float m1_6 = 0.0f;
                if (cvt_f32_f16_21 > 0.0f) {
                    m0_6 = fdiv_rn_9;
                }
                if (cvt_f32_f16_1_20 > 0.0f) {
                    m1_6 = fdiv_rn_1_6;
                }
                float2 mm_31 = make_float2(m0_6, m0_6);
                float2 pair_58 = make_float2(acc_2[0], acc_2[1]);
                float2 g_58;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_58) : "l"(*(const unsigned long long*)&pair_58), "l"(*(const unsigned long long*)&mm_31));
                uint32_t fp4_24;
                uint16_t _e2m1_pair_h_191;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_191) : "f"(g_58.x), "f"(g_58.y));
                fp4_24 = (uint32_t)_e2m1_pair_h_191;
                float2 pair_59 = make_float2(acc_2[2], acc_2[3]);
                float2 g_59;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_59) : "l"(*(const unsigned long long*)&pair_59), "l"(*(const unsigned long long*)&mm_31));
                uint32_t fp4_25;
                uint16_t _e2m1_pair_h_192;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_192) : "f"(g_59.x), "f"(g_59.y));
                fp4_25 = (uint32_t)_e2m1_pair_h_192;
                float2 pair_60 = make_float2(acc_2[4], acc_2[5]);
                float2 g_60;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_60) : "l"(*(const unsigned long long*)&pair_60), "l"(*(const unsigned long long*)&mm_31));
                uint32_t fp4_26;
                uint16_t _e2m1_pair_h_193;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_193) : "f"(g_60.x), "f"(g_60.y));
                fp4_26 = (uint32_t)_e2m1_pair_h_193;
                float2 pair_61 = make_float2(acc_2[6], acc_2[7]);
                float2 g_61;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_61) : "l"(*(const unsigned long long*)&pair_61), "l"(*(const unsigned long long*)&mm_31));
                uint32_t _fp4_e2m1_pack_pair_194;
                uint16_t _e2m1_pair_h_195;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_195) : "f"(g_61.x), "f"(g_61.y));
                _fp4_e2m1_pack_pair_194 = (uint32_t)_e2m1_pair_h_195;
                unsigned int word_31 = fp4_24 | fp4_25 << 8 | fp4_26 << 16 | _fp4_e2m1_pack_pair_194 << 24;
                unsigned int carry_30 = (word_31 & 2004318071) + 2004318071;
                q_225[0] = word_31 & (carry_30 | 2004318071);
                float2 mm_32 = make_float2(m0_6, m0_6);
                float2 pair_62 = make_float2(acc_2[8], acc_2[9]);
                float2 g_62;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_62) : "l"(*(const unsigned long long*)&pair_62), "l"(*(const unsigned long long*)&mm_32));
                uint32_t fp4_27;
                uint16_t _e2m1_pair_h_196;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_196) : "f"(g_62.x), "f"(g_62.y));
                fp4_27 = (uint32_t)_e2m1_pair_h_196;
                float2 pair_63 = make_float2(acc_2[10], acc_2[11]);
                float2 g_63;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_63) : "l"(*(const unsigned long long*)&pair_63), "l"(*(const unsigned long long*)&mm_32));
                uint32_t fp4_28;
                uint16_t _e2m1_pair_h_197;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_197) : "f"(g_63.x), "f"(g_63.y));
                fp4_28 = (uint32_t)_e2m1_pair_h_197;
                float2 pair_64 = make_float2(acc_2[12], acc_2[13]);
                float2 g_64;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_64) : "l"(*(const unsigned long long*)&pair_64), "l"(*(const unsigned long long*)&mm_32));
                uint32_t fp4_29;
                uint16_t _e2m1_pair_h_198;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_198) : "f"(g_64.x), "f"(g_64.y));
                fp4_29 = (uint32_t)_e2m1_pair_h_198;
                float2 pair_65 = make_float2(acc_2[14], acc_2[15]);
                float2 g_65;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_65) : "l"(*(const unsigned long long*)&pair_65), "l"(*(const unsigned long long*)&mm_32));
                uint32_t _fp4_e2m1_pack_pair_199;
                uint16_t _e2m1_pair_h_200;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_200) : "f"(g_65.x), "f"(g_65.y));
                _fp4_e2m1_pack_pair_199 = (uint32_t)_e2m1_pair_h_200;
                unsigned int word_32 = fp4_27 | fp4_28 << 8 | fp4_29 << 16 | _fp4_e2m1_pack_pair_199 << 24;
                unsigned int carry_31 = (word_32 & 2004318071) + 2004318071;
                q_225[1] = word_32 & (carry_31 | 2004318071);
                float2 mm_33 = make_float2(m1_6, m1_6);
                float2 pair_66 = make_float2(acc_2[16], acc_2[17]);
                float2 g_66;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_66) : "l"(*(const unsigned long long*)&pair_66), "l"(*(const unsigned long long*)&mm_33));
                uint32_t fp4_30;
                uint16_t _e2m1_pair_h_201;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_201) : "f"(g_66.x), "f"(g_66.y));
                fp4_30 = (uint32_t)_e2m1_pair_h_201;
                float2 pair_67 = make_float2(acc_2[18], acc_2[19]);
                float2 g_67;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_67) : "l"(*(const unsigned long long*)&pair_67), "l"(*(const unsigned long long*)&mm_33));
                uint32_t fp4_31;
                uint16_t _e2m1_pair_h_202;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_202) : "f"(g_67.x), "f"(g_67.y));
                fp4_31 = (uint32_t)_e2m1_pair_h_202;
                float2 pair_68 = make_float2(acc_2[20], acc_2[21]);
                float2 g_68;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_68) : "l"(*(const unsigned long long*)&pair_68), "l"(*(const unsigned long long*)&mm_33));
                uint32_t fp4_32;
                uint16_t _e2m1_pair_h_203;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_203) : "f"(g_68.x), "f"(g_68.y));
                fp4_32 = (uint32_t)_e2m1_pair_h_203;
                float2 pair_69 = make_float2(acc_2[22], acc_2[23]);
                float2 g_69;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_69) : "l"(*(const unsigned long long*)&pair_69), "l"(*(const unsigned long long*)&mm_33));
                uint32_t _fp4_e2m1_pack_pair_204;
                uint16_t _e2m1_pair_h_205;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_205) : "f"(g_69.x), "f"(g_69.y));
                _fp4_e2m1_pack_pair_204 = (uint32_t)_e2m1_pair_h_205;
                unsigned int word_33 = fp4_30 | fp4_31 << 8 | fp4_32 << 16 | _fp4_e2m1_pack_pair_204 << 24;
                unsigned int carry_32 = (word_33 & 2004318071) + 2004318071;
                q_225[2] = word_33 & (carry_32 | 2004318071);
                float2 mm_34 = make_float2(m1_6, m1_6);
                float2 pair_70 = make_float2(acc_2[24], acc_2[25]);
                float2 g_70;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_70) : "l"(*(const unsigned long long*)&pair_70), "l"(*(const unsigned long long*)&mm_34));
                uint32_t fp4_33;
                uint16_t _e2m1_pair_h_206;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_206) : "f"(g_70.x), "f"(g_70.y));
                fp4_33 = (uint32_t)_e2m1_pair_h_206;
                float2 pair_71 = make_float2(acc_2[26], acc_2[27]);
                float2 g_71;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_71) : "l"(*(const unsigned long long*)&pair_71), "l"(*(const unsigned long long*)&mm_34));
                uint32_t fp4_34;
                uint16_t _e2m1_pair_h_207;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_207) : "f"(g_71.x), "f"(g_71.y));
                fp4_34 = (uint32_t)_e2m1_pair_h_207;
                float2 pair_72 = make_float2(acc_2[28], acc_2[29]);
                float2 g_72;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_72) : "l"(*(const unsigned long long*)&pair_72), "l"(*(const unsigned long long*)&mm_34));
                uint32_t fp4_35;
                uint16_t _e2m1_pair_h_208;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_208) : "f"(g_72.x), "f"(g_72.y));
                fp4_35 = (uint32_t)_e2m1_pair_h_208;
                float2 pair_73 = make_float2(acc_2[30], acc_2[31]);
                float2 g_73;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_73) : "l"(*(const unsigned long long*)&pair_73), "l"(*(const unsigned long long*)&mm_34));
                uint32_t _fp4_e2m1_pack_pair_209;
                uint16_t _e2m1_pair_h_210;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_210) : "f"(g_73.x), "f"(g_73.y));
                _fp4_e2m1_pack_pair_209 = (uint32_t)_e2m1_pair_h_210;
                unsigned int word_34 = fp4_33 | fp4_34 << 8 | fp4_35 << 16 | _fp4_e2m1_pack_pair_209 << 24;
                unsigned int carry_33 = (word_34 & 2004318071) + 2004318071;
                q_225[3] = word_34 & (carry_33 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_2)), "r"(q_225[0]), "r"(q_225[1]), "r"(q_225[2]), "r"(q_225[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2_2)), "h"((uint16_t)(e4m3x2_f32_6 & 32639)) : "memory");
                float d_1[32] = {0};
                uint32_t e4m3x2_to_f16x2_21;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_21) : "h"((uint16_t)(e4m3x2_f32_6 & 32639)));
                float cvt_f32_f16_22;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_22) : "h"((uint16_t)(e4m3x2_to_f16x2_21)));
                float cvt_f32_f16_1_21;
                asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_21) : "h"((uint16_t)(e4m3x2_to_f16x2_21 >> 16)));
                float s0_13;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_13) : "f"(cvt_f32_f16_22), "f"(rs2));
                float s1_13;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_13) : "f"(cvt_f32_f16_1_21), "f"(rs2));
                float2 scale0_13 = make_float2(s0_13, s0_13);
                float2 scale1_13 = make_float2(s1_13, s1_13);
                float tv_56[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_225[0]));
                    (&tv_56[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_56[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_56[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_56[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_56[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_56[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_56[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_56[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_56[0]), "=f"(tv_56[1]), "=f"(tv_56[2]), "=f"(tv_56[3]), "=f"(tv_56[4]), "=f"(tv_56[5]), "=f"(tv_56[6]), "=f"(tv_56[7]) : "r"(q_225[0]));
                    #endif
                }
                float2 a_224 = make_float2(d_1[0], d_1[1]);
                float2 q_226 = make_float2(tv_56[0], tv_56[1]);
                float2 r_224 = fma_f32x2_rn_noftz(q_226, scale0_13, a_224);
                d_1[0] = r_224.x;
                d_1[1] = r_224.y;
                float2 a_225 = make_float2(d_1[2], d_1[3]);
                float2 q_227 = make_float2(tv_56[2], tv_56[3]);
                float2 r_225 = fma_f32x2_rn_noftz(q_227, scale0_13, a_225);
                d_1[2] = r_225.x;
                d_1[3] = r_225.y;
                float2 a_226 = make_float2(d_1[4], d_1[5]);
                float2 q_228 = make_float2(tv_56[4], tv_56[5]);
                float2 r_226 = fma_f32x2_rn_noftz(q_228, scale0_13, a_226);
                d_1[4] = r_226.x;
                d_1[5] = r_226.y;
                float2 a_227 = make_float2(d_1[6], d_1[7]);
                float2 q_229 = make_float2(tv_56[6], tv_56[7]);
                float2 r_227 = fma_f32x2_rn_noftz(q_229, scale0_13, a_227);
                d_1[6] = r_227.x;
                d_1[7] = r_227.y;
                float tv_57[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_225[1 + 0]));
                    (&tv_57[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_57[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_57[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_57[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_57[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_57[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_57[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_57[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_57[0]), "=f"(tv_57[1]), "=f"(tv_57[2]), "=f"(tv_57[3]), "=f"(tv_57[4]), "=f"(tv_57[5]), "=f"(tv_57[6]), "=f"(tv_57[7]) : "r"(q_225[1 + 0]));
                    #endif
                }
                float2 a_228 = make_float2(d_1[8], d_1[9]);
                float2 q_230 = make_float2(tv_57[0], tv_57[1]);
                float2 r_228 = fma_f32x2_rn_noftz(q_230, scale0_13, a_228);
                d_1[8] = r_228.x;
                d_1[9] = r_228.y;
                float2 a_229 = make_float2(d_1[10], d_1[11]);
                float2 q_231 = make_float2(tv_57[2], tv_57[3]);
                float2 r_229 = fma_f32x2_rn_noftz(q_231, scale0_13, a_229);
                d_1[10] = r_229.x;
                d_1[11] = r_229.y;
                float2 a_230 = make_float2(d_1[12], d_1[13]);
                float2 q_232 = make_float2(tv_57[4], tv_57[5]);
                float2 r_230 = fma_f32x2_rn_noftz(q_232, scale0_13, a_230);
                d_1[12] = r_230.x;
                d_1[13] = r_230.y;
                float2 a_231 = make_float2(d_1[14], d_1[15]);
                float2 q_233 = make_float2(tv_57[6], tv_57[7]);
                float2 r_231 = fma_f32x2_rn_noftz(q_233, scale0_13, a_231);
                d_1[14] = r_231.x;
                d_1[15] = r_231.y;
                float tv_58[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_225[2 + 0]));
                    (&tv_58[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_58[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_58[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_58[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_58[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_58[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_58[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_58[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_58[0]), "=f"(tv_58[1]), "=f"(tv_58[2]), "=f"(tv_58[3]), "=f"(tv_58[4]), "=f"(tv_58[5]), "=f"(tv_58[6]), "=f"(tv_58[7]) : "r"(q_225[2 + 0]));
                    #endif
                }
                float2 a_232 = make_float2(d_1[16], d_1[17]);
                float2 q_234 = make_float2(tv_58[0], tv_58[1]);
                float2 r_232 = fma_f32x2_rn_noftz(q_234, scale1_13, a_232);
                d_1[16] = r_232.x;
                d_1[17] = r_232.y;
                float2 a_233 = make_float2(d_1[18], d_1[19]);
                float2 q_235 = make_float2(tv_58[2], tv_58[3]);
                float2 r_233 = fma_f32x2_rn_noftz(q_235, scale1_13, a_233);
                d_1[18] = r_233.x;
                d_1[19] = r_233.y;
                float2 a_234 = make_float2(d_1[20], d_1[21]);
                float2 q_236 = make_float2(tv_58[4], tv_58[5]);
                float2 r_234 = fma_f32x2_rn_noftz(q_236, scale1_13, a_234);
                d_1[20] = r_234.x;
                d_1[21] = r_234.y;
                float2 a_235 = make_float2(d_1[22], d_1[23]);
                float2 q_237 = make_float2(tv_58[6], tv_58[7]);
                float2 r_235 = fma_f32x2_rn_noftz(q_237, scale1_13, a_235);
                d_1[22] = r_235.x;
                d_1[23] = r_235.y;
                float tv_59[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_225[3 + 0]));
                    (&tv_59[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&tv_59[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&tv_59[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&tv_59[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&tv_59[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&tv_59[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&tv_59[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&tv_59[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_59[0]), "=f"(tv_59[1]), "=f"(tv_59[2]), "=f"(tv_59[3]), "=f"(tv_59[4]), "=f"(tv_59[5]), "=f"(tv_59[6]), "=f"(tv_59[7]) : "r"(q_225[3 + 0]));
                    #endif
                }
                float2 a_236 = make_float2(d_1[24], d_1[25]);
                float2 q_238 = make_float2(tv_59[0], tv_59[1]);
                float2 r_236 = fma_f32x2_rn_noftz(q_238, scale1_13, a_236);
                d_1[24] = r_236.x;
                d_1[25] = r_236.y;
                float2 a_237 = make_float2(d_1[26], d_1[27]);
                float2 q_239 = make_float2(tv_59[2], tv_59[3]);
                float2 r_237 = fma_f32x2_rn_noftz(q_239, scale1_13, a_237);
                d_1[26] = r_237.x;
                d_1[27] = r_237.y;
                float2 a_238 = make_float2(d_1[28], d_1[29]);
                float2 q_240 = make_float2(tv_59[4], tv_59[5]);
                float2 r_238 = fma_f32x2_rn_noftz(q_240, scale1_13, a_238);
                d_1[28] = r_238.x;
                d_1[29] = r_238.y;
                float2 a_239 = make_float2(d_1[30], d_1[31]);
                float2 q_241 = make_float2(tv_59[6], tv_59[7]);
                float2 r_239 = fma_f32x2_rn_noftz(q_241, scale1_13, a_239);
                d_1[30] = r_239.x;
                d_1[31] = r_239.y;
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
                int c_8 = lane_1;
                int l_8 = c_8 >> 2;
                unsigned int w4_8[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_8 + (c_8 & 3 ^ l_8 >> 1 & 3)))));
                long long idx_8 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_8));
                {
                    int4 _iv4 = make_int4(w4_8[0 + 0], w4_8[0 + 1], w4_8[0 + 2], w4_8[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_8) = _iv4;
                }
                int c_9 = lane_1 + 32;
                int l_9 = c_9 >> 2;
                unsigned int w4_9[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_9 + (c_9 & 3 ^ l_9 >> 1 & 3)))));
                long long idx_9 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_9));
                {
                    int4 _iv4 = make_int4(w4_9[0 + 0], w4_9[0 + 1], w4_9[0 + 2], w4_9[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_9) = _iv4;
                }
                int c_10 = lane_1 + 64;
                int l_10 = c_10 >> 2;
                unsigned int w4_10[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_10 + (c_10 & 3 ^ l_10 >> 1 & 3)))));
                long long idx_10 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_10));
                {
                    int4 _iv4 = make_int4(w4_10[0 + 0], w4_10[0 + 1], w4_10[0 + 2], w4_10[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_10) = _iv4;
                }
                int c_11 = lane_1 + 96;
                int l_11 = c_11 >> 2;
                unsigned int w4_11[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_11[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_11 + (c_11 & 3 ^ l_11 >> 1 & 3)))));
                long long idx_11 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_11));
                {
                    int4 _iv4 = make_int4(w4_11[0 + 0], w4_11[0 + 1], w4_11[0 + 2], w4_11[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_11) = _iv4;
                }
                __syncwarp();
                if (h == 0 && lane_1 < 4) {
                    {
                        uint32_t _addr_211 = static_cast<uint32_t>(wsm_addr + (unsigned int)(tob_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_211), "f"(rs2) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                if (h == 0 && lane_1 == 0) {
                    int ag_wo = (int)ag_w;
                    int ag_ro = (int)ag_r;
                    int capo_0 = (int)capo64;
                    int ix_1_1 = (int)tix64;
                    int rk_2 = (int)rank64;
                    int v_off_1 = ag_wo + (rk_2 * capo_0 + ix_1_1) * 3072;
                    int sc_off_1 = ag_wo + 4 * capo_0 * 3072 + (rk_2 * capo_0 + ix_1_1) * 384;
                    int m_off_1 = ag_ro + (rk_2 * capo_0 + ix_1_1) * 16;
                    int peer_1_0 = (rank + 1) % 4;
                    uint8_t* pw_1_0 = reinterpret_cast<uint8_t*>(tab[peer_1_0]);
                    uint8_t* pr_1_0 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_0]);
                    {
                        void* _cpbulk_dst_212 = reinterpret_cast<void*>(pw_1_0 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_212), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_213 = reinterpret_cast<void*>(pw_1_0 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_213), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_214 = reinterpret_cast<void*>(pr_1_0 + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_214), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_1_1 = (rank + 2) % 4;
                    uint8_t* pw_1_1 = reinterpret_cast<uint8_t*>(tab[peer_1_1]);
                    uint8_t* pr_1_1 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_1]);
                    {
                        void* _cpbulk_dst_215 = reinterpret_cast<void*>(pw_1_1 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_215), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_216 = reinterpret_cast<void*>(pw_1_1 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_216), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_217 = reinterpret_cast<void*>(pr_1_1 + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_217), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_1_2 = (rank + 3) % 4;
                    uint8_t* pw_1_2 = reinterpret_cast<uint8_t*>(tab[peer_1_2]);
                    uint8_t* pr_1_2 = reinterpret_cast<uint8_t*>(tab[4 + peer_1_2]);
                    {
                        void* _cpbulk_dst_218 = reinterpret_cast<void*>(pw_1_2 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_218), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_219 = reinterpret_cast<void*>(pw_1_2 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_219), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_220 = reinterpret_cast<void*>(pr_1_2 + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_220), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                kb = kb + 1;
            }
            int fme = rb0 - r0;
            #pragma unroll 1
            for (int c = wC; c < nC; c += 10) {
                int blk = c / 3;
                int w = c % 3;
                int bump = 0;
                if (w >= fme) {
                    bump = 1;
                }
                int crow = r0 + blk * 4 + w + bump;
                long long cix64 = (long long)(crow / 4);
                long long crow64 = (long long)crow;
                long long csrc64 = (long long)(crow % 4);
                int ag_wo_1 = (int)ag_w;
                int ag_ro_1 = (int)ag_r;
                int capo_2 = (int)capo64;
                int src = (int)csrc64;
                int ix_2 = (int)cix64;
                int m_off_2 = ag_ro_1 + (src * capo_2 + ix_2) * 16;
                float rs_2 = 0.0f;
                uint32_t polled[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(polled[0]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + m_off_2)) : "memory");
                } while (
                    (polled[0] == 0x80000000u)
                );
                rs_2 = __uint_as_float(polled[0]);
                int v_4 = ag_wo_1 + (src * capo_2 + ix_2) * 3072;
                int sc_4 = ag_wo_1 + 4 * capo_2 * 3072 + (src * capo_2 + ix_2) * 384;
                #pragma unroll 1
                for (int jj = 0; jj < 2; jj++) {
                    int pp0 = lane_1 + 96 * jj;
                    int pp1 = pp0 + 32;
                    int pp2 = pp0 + 64;
                    uint32_t fp4_wire_values_1[12];
                    uint32_t fp4_wire_scales_1[3];
                    uint16_t _wire_h16_221_0;
                    uint16_t _wire_h16_221_1;
                    uint16_t _wire_h16_221_2;
                    uint32_t _wire_vpend_221 = 7u;
                    uint32_t _wire_spend_221 = 7u;
                    do {
                        if (_wire_vpend_221 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_1[0]), "=r"(fp4_wire_values_1[1]), "=r"(fp4_wire_values_1[2]), "=r"(fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_221 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_1[4]), "=r"(fp4_wire_values_1[5]), "=r"(fp4_wire_values_1[6]), "=r"(fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_221 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_1[8]), "=r"(fp4_wire_values_1[9]), "=r"(fp4_wire_values_1[10]), "=r"(fp4_wire_values_1[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp2)))) : "memory");
                        }
                        if (_wire_spend_221 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_221_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp0)))) : "memory");
                            fp4_wire_scales_1[0] = (uint32_t)_wire_h16_221_0;
                        }
                        if (_wire_spend_221 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_221_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp1)))) : "memory");
                            fp4_wire_scales_1[1] = (uint32_t)_wire_h16_221_1;
                        }
                        if (_wire_spend_221 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_221_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp2)))) : "memory");
                            fp4_wire_scales_1[2] = (uint32_t)_wire_h16_221_2;
                        }
                        _wire_vpend_221 = ((uint32_t)((((fp4_wire_values_1[0] & ~((fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[1] & ~((fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[2] & ~((fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[3] & ~((fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((fp4_wire_values_1[4] & ~((fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[5] & ~((fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[6] & ~((fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[7] & ~((fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((fp4_wire_values_1[8] & ~((fp4_wire_values_1[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[9] & ~((fp4_wire_values_1[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[10] & ~((fp4_wire_values_1[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[11] & ~((fp4_wire_values_1[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2);
                        _wire_spend_221 = ((uint32_t)(((uint32_t)((fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_1[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_1[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((fp4_wire_scales_1[2] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_1[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2);
                    } while ((_wire_vpend_221 | _wire_spend_221) != 0u);
                    float acc[32] = {0};
                    uint32_t e4m3x2_to_f16x2_22;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_22) : "h"((uint16_t)(fp4_wire_scales_1[0])));
                    float cvt_f32_f16_23;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_23) : "h"((uint16_t)(e4m3x2_to_f16x2_22)));
                    float cvt_f32_f16_1_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_22) : "h"((uint16_t)(e4m3x2_to_f16x2_22 >> 16)));
                    float s0_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_14) : "f"(cvt_f32_f16_23), "f"(rs_2));
                    float s1_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_14) : "f"(cvt_f32_f16_1_22), "f"(rs_2));
                    float2 scale0_14 = make_float2(s0_14, s0_14);
                    float2 scale1_14 = make_float2(s1_14, s1_14);
                    float tv_60[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[0]));
                        (&tv_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_60[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_60[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_60[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_60[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_60[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_60[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_60[0]), "=f"(tv_60[1]), "=f"(tv_60[2]), "=f"(tv_60[3]), "=f"(tv_60[4]), "=f"(tv_60[5]), "=f"(tv_60[6]), "=f"(tv_60[7]) : "r"(fp4_wire_values_1[0]));
                        #endif
                    }
                    float2 a_240 = make_float2(acc[0], acc[1]);
                    float2 q_242 = make_float2(tv_60[0], tv_60[1]);
                    float2 r_240 = fma_f32x2_rn_noftz(q_242, scale0_14, a_240);
                    acc[0] = r_240.x;
                    acc[1] = r_240.y;
                    float2 a_241 = make_float2(acc[2], acc[3]);
                    float2 q_243 = make_float2(tv_60[2], tv_60[3]);
                    float2 r_241 = fma_f32x2_rn_noftz(q_243, scale0_14, a_241);
                    acc[2] = r_241.x;
                    acc[3] = r_241.y;
                    float2 a_242 = make_float2(acc[4], acc[5]);
                    float2 q_244 = make_float2(tv_60[4], tv_60[5]);
                    float2 r_242 = fma_f32x2_rn_noftz(q_244, scale0_14, a_242);
                    acc[4] = r_242.x;
                    acc[5] = r_242.y;
                    float2 a_243 = make_float2(acc[6], acc[7]);
                    float2 q_245 = make_float2(tv_60[6], tv_60[7]);
                    float2 r_243 = fma_f32x2_rn_noftz(q_245, scale0_14, a_243);
                    acc[6] = r_243.x;
                    acc[7] = r_243.y;
                    float tv_61[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[1 + 0]));
                        (&tv_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_61[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_61[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_61[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_61[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_61[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_61[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_61[0]), "=f"(tv_61[1]), "=f"(tv_61[2]), "=f"(tv_61[3]), "=f"(tv_61[4]), "=f"(tv_61[5]), "=f"(tv_61[6]), "=f"(tv_61[7]) : "r"(fp4_wire_values_1[1 + 0]));
                        #endif
                    }
                    float2 a_244 = make_float2(acc[8], acc[9]);
                    float2 q_246 = make_float2(tv_61[0], tv_61[1]);
                    float2 r_244 = fma_f32x2_rn_noftz(q_246, scale0_14, a_244);
                    acc[8] = r_244.x;
                    acc[9] = r_244.y;
                    float2 a_245 = make_float2(acc[10], acc[11]);
                    float2 q_247 = make_float2(tv_61[2], tv_61[3]);
                    float2 r_245 = fma_f32x2_rn_noftz(q_247, scale0_14, a_245);
                    acc[10] = r_245.x;
                    acc[11] = r_245.y;
                    float2 a_246 = make_float2(acc[12], acc[13]);
                    float2 q_248 = make_float2(tv_61[4], tv_61[5]);
                    float2 r_246 = fma_f32x2_rn_noftz(q_248, scale0_14, a_246);
                    acc[12] = r_246.x;
                    acc[13] = r_246.y;
                    float2 a_247 = make_float2(acc[14], acc[15]);
                    float2 q_249 = make_float2(tv_61[6], tv_61[7]);
                    float2 r_247 = fma_f32x2_rn_noftz(q_249, scale0_14, a_247);
                    acc[14] = r_247.x;
                    acc[15] = r_247.y;
                    float tv_62[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[2 + 0]));
                        (&tv_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_62[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_62[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_62[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_62[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_62[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_62[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_62[0]), "=f"(tv_62[1]), "=f"(tv_62[2]), "=f"(tv_62[3]), "=f"(tv_62[4]), "=f"(tv_62[5]), "=f"(tv_62[6]), "=f"(tv_62[7]) : "r"(fp4_wire_values_1[2 + 0]));
                        #endif
                    }
                    float2 a_248 = make_float2(acc[16], acc[17]);
                    float2 q_250 = make_float2(tv_62[0], tv_62[1]);
                    float2 r_248 = fma_f32x2_rn_noftz(q_250, scale1_14, a_248);
                    acc[16] = r_248.x;
                    acc[17] = r_248.y;
                    float2 a_249 = make_float2(acc[18], acc[19]);
                    float2 q_251 = make_float2(tv_62[2], tv_62[3]);
                    float2 r_249 = fma_f32x2_rn_noftz(q_251, scale1_14, a_249);
                    acc[18] = r_249.x;
                    acc[19] = r_249.y;
                    float2 a_250 = make_float2(acc[20], acc[21]);
                    float2 q_252 = make_float2(tv_62[4], tv_62[5]);
                    float2 r_250 = fma_f32x2_rn_noftz(q_252, scale1_14, a_250);
                    acc[20] = r_250.x;
                    acc[21] = r_250.y;
                    float2 a_251 = make_float2(acc[22], acc[23]);
                    float2 q_253 = make_float2(tv_62[6], tv_62[7]);
                    float2 r_251 = fma_f32x2_rn_noftz(q_253, scale1_14, a_251);
                    acc[22] = r_251.x;
                    acc[23] = r_251.y;
                    float tv_63[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[3 + 0]));
                        (&tv_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_63[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_63[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_63[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_63[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_63[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_63[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_63[0]), "=f"(tv_63[1]), "=f"(tv_63[2]), "=f"(tv_63[3]), "=f"(tv_63[4]), "=f"(tv_63[5]), "=f"(tv_63[6]), "=f"(tv_63[7]) : "r"(fp4_wire_values_1[3 + 0]));
                        #endif
                    }
                    float2 a_252 = make_float2(acc[24], acc[25]);
                    float2 q_254 = make_float2(tv_63[0], tv_63[1]);
                    float2 r_252 = fma_f32x2_rn_noftz(q_254, scale1_14, a_252);
                    acc[24] = r_252.x;
                    acc[25] = r_252.y;
                    float2 a_253 = make_float2(acc[26], acc[27]);
                    float2 q_255 = make_float2(tv_63[2], tv_63[3]);
                    float2 r_253 = fma_f32x2_rn_noftz(q_255, scale1_14, a_253);
                    acc[26] = r_253.x;
                    acc[27] = r_253.y;
                    float2 a_254 = make_float2(acc[28], acc[29]);
                    float2 q_256 = make_float2(tv_63[4], tv_63[5]);
                    float2 r_254 = fma_f32x2_rn_noftz(q_256, scale1_14, a_254);
                    acc[28] = r_254.x;
                    acc[29] = r_254.y;
                    float2 a_255 = make_float2(acc[30], acc[31]);
                    float2 q_257 = make_float2(tv_63[6], tv_63[7]);
                    float2 r_255 = fma_f32x2_rn_noftz(q_257, scale1_14, a_255);
                    acc[30] = r_255.x;
                    acc[31] = r_255.y;
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
                    int c_12 = lane_1;
                    int l_12 = c_12 >> 2;
                    unsigned int w4_12[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_12 + (c_12 & 3 ^ l_12 >> 1 & 3)))));
                    long long idx_12 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_12));
                    {
                        int4 _iv4 = make_int4(w4_12[0 + 0], w4_12[0 + 1], w4_12[0 + 2], w4_12[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_12) = _iv4;
                    }
                    int c_13 = lane_1 + 32;
                    int l_13 = c_13 >> 2;
                    unsigned int w4_13[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_13[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_13 + (c_13 & 3 ^ l_13 >> 1 & 3)))));
                    long long idx_13 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_13));
                    {
                        int4 _iv4 = make_int4(w4_13[0 + 0], w4_13[0 + 1], w4_13[0 + 2], w4_13[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_13) = _iv4;
                    }
                    int c_14 = lane_1 + 64;
                    int l_14 = c_14 >> 2;
                    unsigned int w4_14[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_14[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_14 + (c_14 & 3 ^ l_14 >> 1 & 3)))));
                    long long idx_14 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_14));
                    {
                        int4 _iv4 = make_int4(w4_14[0 + 0], w4_14[0 + 1], w4_14[0 + 2], w4_14[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_14) = _iv4;
                    }
                    int c_15 = lane_1 + 96;
                    int l_15 = c_15 >> 2;
                    unsigned int w4_15[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_15[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_15 + (c_15 & 3 ^ l_15 >> 1 & 3)))));
                    long long idx_15 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_15));
                    {
                        int4 _iv4 = make_int4(w4_15[0 + 0], w4_15[0 + 1], w4_15[0 + 2], w4_15[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_15) = _iv4;
                    }
                    __syncwarp();
                    float acc_3[32] = {0};
                    uint32_t e4m3x2_to_f16x2_22_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_22_1) : "h"((uint16_t)(fp4_wire_scales_1[1])));
                    float cvt_f32_f16_23_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_23_1) : "h"((uint16_t)(e4m3x2_to_f16x2_22_1)));
                    float cvt_f32_f16_1_22_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_22_1) : "h"((uint16_t)(e4m3x2_to_f16x2_22_1 >> 16)));
                    float s0_14_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_14_1) : "f"(cvt_f32_f16_23_1), "f"(rs_2));
                    float s1_14_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_14_1) : "f"(cvt_f32_f16_1_22_1), "f"(rs_2));
                    float2 scale0_14_1 = make_float2(s0_14_1, s0_14_1);
                    float2 scale1_14_1 = make_float2(s1_14_1, s1_14_1);
                    float tv_64[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[4 + 0]));
                        (&tv_64[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_64[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_64[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_64[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_64[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_64[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_64[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_64[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_64[0]), "=f"(tv_64[1]), "=f"(tv_64[2]), "=f"(tv_64[3]), "=f"(tv_64[4]), "=f"(tv_64[5]), "=f"(tv_64[6]), "=f"(tv_64[7]) : "r"(fp4_wire_values_1[4 + 0]));
                        #endif
                    }
                    float2 a_256 = make_float2(acc_3[0], acc_3[1]);
                    float2 q_258 = make_float2(tv_64[0], tv_64[1]);
                    float2 r_256 = fma_f32x2_rn_noftz(q_258, scale0_14_1, a_256);
                    acc_3[0] = r_256.x;
                    acc_3[1] = r_256.y;
                    float2 a_257 = make_float2(acc_3[2], acc_3[3]);
                    float2 q_259 = make_float2(tv_64[2], tv_64[3]);
                    float2 r_257 = fma_f32x2_rn_noftz(q_259, scale0_14_1, a_257);
                    acc_3[2] = r_257.x;
                    acc_3[3] = r_257.y;
                    float2 a_258 = make_float2(acc_3[4], acc_3[5]);
                    float2 q_260 = make_float2(tv_64[4], tv_64[5]);
                    float2 r_258 = fma_f32x2_rn_noftz(q_260, scale0_14_1, a_258);
                    acc_3[4] = r_258.x;
                    acc_3[5] = r_258.y;
                    float2 a_259 = make_float2(acc_3[6], acc_3[7]);
                    float2 q_261 = make_float2(tv_64[6], tv_64[7]);
                    float2 r_259 = fma_f32x2_rn_noftz(q_261, scale0_14_1, a_259);
                    acc_3[6] = r_259.x;
                    acc_3[7] = r_259.y;
                    float tv_65[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[5 + 0]));
                        (&tv_65[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_65[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_65[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_65[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_65[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_65[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_65[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_65[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_65[0]), "=f"(tv_65[1]), "=f"(tv_65[2]), "=f"(tv_65[3]), "=f"(tv_65[4]), "=f"(tv_65[5]), "=f"(tv_65[6]), "=f"(tv_65[7]) : "r"(fp4_wire_values_1[5 + 0]));
                        #endif
                    }
                    float2 a_260 = make_float2(acc_3[8], acc_3[9]);
                    float2 q_262 = make_float2(tv_65[0], tv_65[1]);
                    float2 r_260 = fma_f32x2_rn_noftz(q_262, scale0_14_1, a_260);
                    acc_3[8] = r_260.x;
                    acc_3[9] = r_260.y;
                    float2 a_261 = make_float2(acc_3[10], acc_3[11]);
                    float2 q_263 = make_float2(tv_65[2], tv_65[3]);
                    float2 r_261 = fma_f32x2_rn_noftz(q_263, scale0_14_1, a_261);
                    acc_3[10] = r_261.x;
                    acc_3[11] = r_261.y;
                    float2 a_262 = make_float2(acc_3[12], acc_3[13]);
                    float2 q_264 = make_float2(tv_65[4], tv_65[5]);
                    float2 r_262 = fma_f32x2_rn_noftz(q_264, scale0_14_1, a_262);
                    acc_3[12] = r_262.x;
                    acc_3[13] = r_262.y;
                    float2 a_263 = make_float2(acc_3[14], acc_3[15]);
                    float2 q_265 = make_float2(tv_65[6], tv_65[7]);
                    float2 r_263 = fma_f32x2_rn_noftz(q_265, scale0_14_1, a_263);
                    acc_3[14] = r_263.x;
                    acc_3[15] = r_263.y;
                    float tv_66[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[6 + 0]));
                        (&tv_66[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_66[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_66[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_66[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_66[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_66[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_66[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_66[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_66[0]), "=f"(tv_66[1]), "=f"(tv_66[2]), "=f"(tv_66[3]), "=f"(tv_66[4]), "=f"(tv_66[5]), "=f"(tv_66[6]), "=f"(tv_66[7]) : "r"(fp4_wire_values_1[6 + 0]));
                        #endif
                    }
                    float2 a_264 = make_float2(acc_3[16], acc_3[17]);
                    float2 q_266 = make_float2(tv_66[0], tv_66[1]);
                    float2 r_264 = fma_f32x2_rn_noftz(q_266, scale1_14_1, a_264);
                    acc_3[16] = r_264.x;
                    acc_3[17] = r_264.y;
                    float2 a_265 = make_float2(acc_3[18], acc_3[19]);
                    float2 q_267 = make_float2(tv_66[2], tv_66[3]);
                    float2 r_265 = fma_f32x2_rn_noftz(q_267, scale1_14_1, a_265);
                    acc_3[18] = r_265.x;
                    acc_3[19] = r_265.y;
                    float2 a_266 = make_float2(acc_3[20], acc_3[21]);
                    float2 q_268 = make_float2(tv_66[4], tv_66[5]);
                    float2 r_266 = fma_f32x2_rn_noftz(q_268, scale1_14_1, a_266);
                    acc_3[20] = r_266.x;
                    acc_3[21] = r_266.y;
                    float2 a_267 = make_float2(acc_3[22], acc_3[23]);
                    float2 q_269 = make_float2(tv_66[6], tv_66[7]);
                    float2 r_267 = fma_f32x2_rn_noftz(q_269, scale1_14_1, a_267);
                    acc_3[22] = r_267.x;
                    acc_3[23] = r_267.y;
                    float tv_67[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[7 + 0]));
                        (&tv_67[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_67[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_67[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_67[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_67[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_67[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_67[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_67[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_67[0]), "=f"(tv_67[1]), "=f"(tv_67[2]), "=f"(tv_67[3]), "=f"(tv_67[4]), "=f"(tv_67[5]), "=f"(tv_67[6]), "=f"(tv_67[7]) : "r"(fp4_wire_values_1[7 + 0]));
                        #endif
                    }
                    float2 a_268 = make_float2(acc_3[24], acc_3[25]);
                    float2 q_270 = make_float2(tv_67[0], tv_67[1]);
                    float2 r_268 = fma_f32x2_rn_noftz(q_270, scale1_14_1, a_268);
                    acc_3[24] = r_268.x;
                    acc_3[25] = r_268.y;
                    float2 a_269 = make_float2(acc_3[26], acc_3[27]);
                    float2 q_271 = make_float2(tv_67[2], tv_67[3]);
                    float2 r_269 = fma_f32x2_rn_noftz(q_271, scale1_14_1, a_269);
                    acc_3[26] = r_269.x;
                    acc_3[27] = r_269.y;
                    float2 a_270 = make_float2(acc_3[28], acc_3[29]);
                    float2 q_272 = make_float2(tv_67[4], tv_67[5]);
                    float2 r_270 = fma_f32x2_rn_noftz(q_272, scale1_14_1, a_270);
                    acc_3[28] = r_270.x;
                    acc_3[29] = r_270.y;
                    float2 a_271 = make_float2(acc_3[30], acc_3[31]);
                    float2 q_273 = make_float2(tv_67[6], tv_67[7]);
                    float2 r_271 = fma_f32x2_rn_noftz(q_273, scale1_14_1, a_271);
                    acc_3[30] = r_271.x;
                    acc_3[31] = r_271.y;
                    unsigned int acc_bf16_0[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_3[_lp*2 + 0], acc_3[_lp*2+1 + 0]));
                        acc_bf16_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16_0[0]), "r"(acc_bf16_0[1]), "r"(acc_bf16_0[2]), "r"(acc_bf16_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16_0[4]), "r"(acc_bf16_0[5]), "r"(acc_bf16_0[6]), "r"(acc_bf16_0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16_0[8]), "r"(acc_bf16_0[9]), "r"(acc_bf16_0[10]), "r"(acc_bf16_0[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16_0[12]), "r"(acc_bf16_0[13]), "r"(acc_bf16_0[14]), "r"(acc_bf16_0[15]) : "memory");
                    __syncwarp();
                    int c_16 = lane_1;
                    int l_16 = c_16 >> 2;
                    unsigned int w4_16[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_16 + (c_16 & 3 ^ l_16 >> 1 & 3)))));
                    long long idx_16 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_16));
                    {
                        int4 _iv4 = make_int4(w4_16[0 + 0], w4_16[0 + 1], w4_16[0 + 2], w4_16[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_16) = _iv4;
                    }
                    int c_17 = lane_1 + 32;
                    int l_17 = c_17 >> 2;
                    unsigned int w4_17[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_17[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_17 + (c_17 & 3 ^ l_17 >> 1 & 3)))));
                    long long idx_17 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_17));
                    {
                        int4 _iv4 = make_int4(w4_17[0 + 0], w4_17[0 + 1], w4_17[0 + 2], w4_17[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_17) = _iv4;
                    }
                    int c_18 = lane_1 + 64;
                    int l_18 = c_18 >> 2;
                    unsigned int w4_18[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_18[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_18 + (c_18 & 3 ^ l_18 >> 1 & 3)))));
                    long long idx_18 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_18));
                    {
                        int4 _iv4 = make_int4(w4_18[0 + 0], w4_18[0 + 1], w4_18[0 + 2], w4_18[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_18) = _iv4;
                    }
                    int c_19 = lane_1 + 96;
                    int l_19 = c_19 >> 2;
                    unsigned int w4_19[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_19[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_19 + (c_19 & 3 ^ l_19 >> 1 & 3)))));
                    long long idx_19 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_19));
                    {
                        int4 _iv4 = make_int4(w4_19[0 + 0], w4_19[0 + 1], w4_19[0 + 2], w4_19[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_19) = _iv4;
                    }
                    __syncwarp();
                    float acc_4[32] = {0};
                    uint32_t e4m3x2_to_f16x2_23;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_23) : "h"((uint16_t)(fp4_wire_scales_1[2])));
                    float cvt_f32_f16_24;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_24) : "h"((uint16_t)(e4m3x2_to_f16x2_23)));
                    float cvt_f32_f16_1_23;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_23) : "h"((uint16_t)(e4m3x2_to_f16x2_23 >> 16)));
                    float s0_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_15) : "f"(cvt_f32_f16_24), "f"(rs_2));
                    float s1_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_15) : "f"(cvt_f32_f16_1_23), "f"(rs_2));
                    float2 scale0_15 = make_float2(s0_15, s0_15);
                    float2 scale1_15 = make_float2(s1_15, s1_15);
                    float tv_68[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[8 + 0]));
                        (&tv_68[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_68[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_68[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_68[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_68[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_68[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_68[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_68[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_68[0]), "=f"(tv_68[1]), "=f"(tv_68[2]), "=f"(tv_68[3]), "=f"(tv_68[4]), "=f"(tv_68[5]), "=f"(tv_68[6]), "=f"(tv_68[7]) : "r"(fp4_wire_values_1[8 + 0]));
                        #endif
                    }
                    float2 a_272 = make_float2(acc_4[0], acc_4[1]);
                    float2 q_274 = make_float2(tv_68[0], tv_68[1]);
                    float2 r_272 = fma_f32x2_rn_noftz(q_274, scale0_15, a_272);
                    acc_4[0] = r_272.x;
                    acc_4[1] = r_272.y;
                    float2 a_273 = make_float2(acc_4[2], acc_4[3]);
                    float2 q_275 = make_float2(tv_68[2], tv_68[3]);
                    float2 r_273 = fma_f32x2_rn_noftz(q_275, scale0_15, a_273);
                    acc_4[2] = r_273.x;
                    acc_4[3] = r_273.y;
                    float2 a_274 = make_float2(acc_4[4], acc_4[5]);
                    float2 q_276 = make_float2(tv_68[4], tv_68[5]);
                    float2 r_274 = fma_f32x2_rn_noftz(q_276, scale0_15, a_274);
                    acc_4[4] = r_274.x;
                    acc_4[5] = r_274.y;
                    float2 a_275 = make_float2(acc_4[6], acc_4[7]);
                    float2 q_277 = make_float2(tv_68[6], tv_68[7]);
                    float2 r_275 = fma_f32x2_rn_noftz(q_277, scale0_15, a_275);
                    acc_4[6] = r_275.x;
                    acc_4[7] = r_275.y;
                    float tv_69[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[9 + 0]));
                        (&tv_69[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_69[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_69[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_69[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_69[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_69[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_69[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_69[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_69[0]), "=f"(tv_69[1]), "=f"(tv_69[2]), "=f"(tv_69[3]), "=f"(tv_69[4]), "=f"(tv_69[5]), "=f"(tv_69[6]), "=f"(tv_69[7]) : "r"(fp4_wire_values_1[9 + 0]));
                        #endif
                    }
                    float2 a_276 = make_float2(acc_4[8], acc_4[9]);
                    float2 q_278 = make_float2(tv_69[0], tv_69[1]);
                    float2 r_276 = fma_f32x2_rn_noftz(q_278, scale0_15, a_276);
                    acc_4[8] = r_276.x;
                    acc_4[9] = r_276.y;
                    float2 a_277 = make_float2(acc_4[10], acc_4[11]);
                    float2 q_279 = make_float2(tv_69[2], tv_69[3]);
                    float2 r_277 = fma_f32x2_rn_noftz(q_279, scale0_15, a_277);
                    acc_4[10] = r_277.x;
                    acc_4[11] = r_277.y;
                    float2 a_278 = make_float2(acc_4[12], acc_4[13]);
                    float2 q_280 = make_float2(tv_69[4], tv_69[5]);
                    float2 r_278 = fma_f32x2_rn_noftz(q_280, scale0_15, a_278);
                    acc_4[12] = r_278.x;
                    acc_4[13] = r_278.y;
                    float2 a_279 = make_float2(acc_4[14], acc_4[15]);
                    float2 q_281 = make_float2(tv_69[6], tv_69[7]);
                    float2 r_279 = fma_f32x2_rn_noftz(q_281, scale0_15, a_279);
                    acc_4[14] = r_279.x;
                    acc_4[15] = r_279.y;
                    float tv_70[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[10 + 0]));
                        (&tv_70[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_70[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_70[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_70[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_70[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_70[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_70[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_70[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_70[0]), "=f"(tv_70[1]), "=f"(tv_70[2]), "=f"(tv_70[3]), "=f"(tv_70[4]), "=f"(tv_70[5]), "=f"(tv_70[6]), "=f"(tv_70[7]) : "r"(fp4_wire_values_1[10 + 0]));
                        #endif
                    }
                    float2 a_280 = make_float2(acc_4[16], acc_4[17]);
                    float2 q_282 = make_float2(tv_70[0], tv_70[1]);
                    float2 r_280 = fma_f32x2_rn_noftz(q_282, scale1_15, a_280);
                    acc_4[16] = r_280.x;
                    acc_4[17] = r_280.y;
                    float2 a_281 = make_float2(acc_4[18], acc_4[19]);
                    float2 q_283 = make_float2(tv_70[2], tv_70[3]);
                    float2 r_281 = fma_f32x2_rn_noftz(q_283, scale1_15, a_281);
                    acc_4[18] = r_281.x;
                    acc_4[19] = r_281.y;
                    float2 a_282 = make_float2(acc_4[20], acc_4[21]);
                    float2 q_284 = make_float2(tv_70[4], tv_70[5]);
                    float2 r_282 = fma_f32x2_rn_noftz(q_284, scale1_15, a_282);
                    acc_4[20] = r_282.x;
                    acc_4[21] = r_282.y;
                    float2 a_283 = make_float2(acc_4[22], acc_4[23]);
                    float2 q_285 = make_float2(tv_70[6], tv_70[7]);
                    float2 r_283 = fma_f32x2_rn_noftz(q_285, scale1_15, a_283);
                    acc_4[22] = r_283.x;
                    acc_4[23] = r_283.y;
                    float tv_71[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(fp4_wire_values_1[11 + 0]));
                        (&tv_71[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_71[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_71[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_71[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_71[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_71[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_71[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_71[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_71[0]), "=f"(tv_71[1]), "=f"(tv_71[2]), "=f"(tv_71[3]), "=f"(tv_71[4]), "=f"(tv_71[5]), "=f"(tv_71[6]), "=f"(tv_71[7]) : "r"(fp4_wire_values_1[11 + 0]));
                        #endif
                    }
                    float2 a_284 = make_float2(acc_4[24], acc_4[25]);
                    float2 q_286 = make_float2(tv_71[0], tv_71[1]);
                    float2 r_284 = fma_f32x2_rn_noftz(q_286, scale1_15, a_284);
                    acc_4[24] = r_284.x;
                    acc_4[25] = r_284.y;
                    float2 a_285 = make_float2(acc_4[26], acc_4[27]);
                    float2 q_287 = make_float2(tv_71[2], tv_71[3]);
                    float2 r_285 = fma_f32x2_rn_noftz(q_287, scale1_15, a_285);
                    acc_4[26] = r_285.x;
                    acc_4[27] = r_285.y;
                    float2 a_286 = make_float2(acc_4[28], acc_4[29]);
                    float2 q_288 = make_float2(tv_71[4], tv_71[5]);
                    float2 r_286 = fma_f32x2_rn_noftz(q_288, scale1_15, a_286);
                    acc_4[28] = r_286.x;
                    acc_4[29] = r_286.y;
                    float2 a_287 = make_float2(acc_4[30], acc_4[31]);
                    float2 q_289 = make_float2(tv_71[6], tv_71[7]);
                    float2 r_287 = fma_f32x2_rn_noftz(q_289, scale1_15, a_287);
                    acc_4[30] = r_287.x;
                    acc_4[31] = r_287.y;
                    unsigned int acc_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_4[_lp*2 + 0], acc_4[_lp*2+1 + 0]));
                        acc_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16_1[0]), "r"(acc_bf16_1[1]), "r"(acc_bf16_1[2]), "r"(acc_bf16_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16_1[4]), "r"(acc_bf16_1[5]), "r"(acc_bf16_1[6]), "r"(acc_bf16_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16_1[8]), "r"(acc_bf16_1[9]), "r"(acc_bf16_1[10]), "r"(acc_bf16_1[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16_1[12]), "r"(acc_bf16_1[13]), "r"(acc_bf16_1[14]), "r"(acc_bf16_1[15]) : "memory");
                    __syncwarp();
                    int c_20 = lane_1;
                    int l_20 = c_20 >> 2;
                    unsigned int w4_20[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_20 + (c_20 & 3 ^ l_20 >> 1 & 3)))));
                    long long idx_20 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_20));
                    {
                        int4 _iv4 = make_int4(w4_20[0 + 0], w4_20[0 + 1], w4_20[0 + 2], w4_20[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_20) = _iv4;
                    }
                    int c_21 = lane_1 + 32;
                    int l_21 = c_21 >> 2;
                    unsigned int w4_21[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_21[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_21 + (c_21 & 3 ^ l_21 >> 1 & 3)))));
                    long long idx_21 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_21));
                    {
                        int4 _iv4 = make_int4(w4_21[0 + 0], w4_21[0 + 1], w4_21[0 + 2], w4_21[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_21) = _iv4;
                    }
                    int c_22 = lane_1 + 64;
                    int l_22 = c_22 >> 2;
                    unsigned int w4_22[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_22[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_22 + (c_22 & 3 ^ l_22 >> 1 & 3)))));
                    long long idx_22 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_22));
                    {
                        int4 _iv4 = make_int4(w4_22[0 + 0], w4_22[0 + 1], w4_22[0 + 2], w4_22[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_22) = _iv4;
                    }
                    int c_23 = lane_1 + 96;
                    int l_23 = c_23 >> 2;
                    unsigned int w4_23[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_23[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_23 + (c_23 & 3 ^ l_23 >> 1 & 3)))));
                    long long idx_23 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_23));
                    {
                        int4 _iv4 = make_int4(w4_23[0 + 0], w4_23[0 + 1], w4_23[0 + 2], w4_23[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_23) = _iv4;
                    }
                    __syncwarp();
                }
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tid == 0) {
                int base_2 = bid * 4;
                ctrl[base_2 + 2] = (unsigned int)rows;
                ctrl[base_2 + 3] = 2;
                ctrl[base_2] = (call + 1) % 3;
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
            int g32_1 = (int)gridDim.x;
            if (rows / g32_1 <= 8) {
                asm volatile("barrier.sync 7, 352;" ::: "memory");
            }
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
                for (long long g_74 = g0; g_74 < hi; g_74 += 32) {
                    long long widx = v_base_1_0 + 16 * g_74 >> 2;
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
                for (long long g_75 = g0_0; g_75 < hi_0; g_75 += 32) {
                    long long widx_1 = s_base_1_0 + 16 * g_75 >> 2;
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
                for (long long g_76 = g0_1; g_76 < hi_1; g_76 += 32) {
                    long long widx_2 = m_base_1_0 + 16 * g_76 >> 2;
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
                for (long long g_77 = g0_2; g_77 < hi_2; g_77 += 32) {
                    long long widx_3 = v_base_1_1 + 16 * g_77 >> 2;
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
                for (long long g_78 = g0_3; g_78 < hi_3; g_78 += 32) {
                    long long widx_4 = s_base_1_1 + 16 * g_78 >> 2;
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
                for (long long g_79 = g0_4; g_79 < hi_4; g_79 += 32) {
                    long long widx_5 = m_base_1_1 + 16 * g_79 >> 2;
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
                for (long long g_80 = g0_5; g_80 < hi_5; g_80 += 32) {
                    long long widx_6 = v_base_1_2 + 16 * g_80 >> 2;
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
                for (long long g_81 = g0_6; g_81 < hi_6; g_81 += 32) {
                    long long widx_7 = s_base_1_2 + 16 * g_81 >> 2;
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
                for (long long g_82 = g0_7; g_82 < hi_7; g_82 += 32) {
                    long long widx_8 = m_base_1_2 + 16 * g_82 >> 2;
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
                for (long long g_83 = g0_8; g_83 < hi_8; g_83 += 32) {
                    long long widx_9 = v_base_1_3 + 16 * g_83 >> 2;
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
                for (long long g_84 = g0_9; g_84 < hi_9; g_84 += 32) {
                    long long widx_10 = s_base_1_3 + 16 * g_84 >> 2;
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
                for (long long g_85 = g0_10; g_85 < hi_10; g_85 += 32) {
                    long long widx_11 = m_base_1_3 + 16 * g_85 >> 2;
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
                    for (long long g_86 = g0_11; g_86 < hi_11; g_86 += 32) {
                        long long widx_12 = v_base2_1_0 + 16 * g_86 >> 2;
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
                    for (long long g_87 = g0_12; g_87 < hi_12; g_87 += 32) {
                        long long widx_13 = s_base2_1_0 + 16 * g_87 >> 2;
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
                    for (long long g_88 = g0_13; g_88 < hi_13; g_88 += 32) {
                        long long widx_14 = m_base2_1_0 + 16 * g_88 >> 2;
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
                    for (long long g_89 = g0_14; g_89 < hi_14; g_89 += 32) {
                        long long widx_15 = v_base2_1_1 + 16 * g_89 >> 2;
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
                    for (long long g_90 = g0_15; g_90 < hi_15; g_90 += 32) {
                        long long widx_16 = s_base2_1_1 + 16 * g_90 >> 2;
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
                    for (long long g_91 = g0_16; g_91 < hi_16; g_91 += 32) {
                        long long widx_17 = m_base2_1_1 + 16 * g_91 >> 2;
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
                    for (long long g_92 = g0_17; g_92 < hi_17; g_92 += 32) {
                        long long widx_18 = v_base2_1_2 + 16 * g_92 >> 2;
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
                    for (long long g_93 = g0_18; g_93 < hi_18; g_93 += 32) {
                        long long widx_19 = s_base2_1_2 + 16 * g_93 >> 2;
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
                    for (long long g_94 = g0_19; g_94 < hi_19; g_94 += 32) {
                        long long widx_20 = m_base2_1_2 + 16 * g_94 >> 2;
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
                    for (long long g_95 = g0_20; g_95 < hi_20; g_95 += 32) {
                        long long widx_21 = v_base2_1_3 + 16 * g_95 >> 2;
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
                    for (long long g_96 = g0_21; g_96 < hi_21; g_96 += 32) {
                        long long widx_22 = s_base2_1_3 + 16 * g_96 >> 2;
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
                    for (long long g_97 = g0_22; g_97 < hi_22; g_97 += 32) {
                        long long widx_23 = m_base2_1_3 + 16 * g_97 >> 2;
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
