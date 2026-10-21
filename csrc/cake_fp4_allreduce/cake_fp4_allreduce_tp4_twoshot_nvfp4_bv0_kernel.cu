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

extern "C" {

__global__ __launch_bounds__(352, LAUNCH_MIN_BLOCKS) void
cake_fp4_allreduce_tp4_twoshot_nvfp4_bv0(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ ts_wire, uint8_t* __restrict__ ts_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
            long long bid64 = (long long)bid;
            long long g64 = (long long)gridDim.x;
            long long quads64 = (long long)(rows / 4);
            int r0 = (int)(4 * (bid64 * quads64 / g64));
            int r1 = (int)(4 * ((bid64 + 1) * quads64 / g64));
            int nrows = r1 - r0;
            int warp_0 = (int)warp;
            int lane_1 = (int)lane;
            if (tid == 0) {
                ctl[0] = ctrl[0];
                ctl[1] = ctrl[2];
                ctl[2] = ctrl[3];
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
            float g_s = gs[0];
            float g_s_2 = gs[1];
            float g_s_3 = gs[2];
            float g_s_4 = gs[3];
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
                int stage = 2 * warp_0;
                int slot = 0;
                mbarrier_arrive_expect_tx(xb_full_addr + (stage) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(xb_off + slot * 3472), xq + (first_row * 3456), 3456, xb_full_addr + (stage) * 8);
            }
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
                    int stage_1 = 2 * warp_0 + (k + 1 & 1);
                    int slot_1 = (k + 1) % 3;
                    mbarrier_arrive_expect_tx(xb_full_addr + (stage_1) * 8, 3456);
                    cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(xb_off + slot_1 * 3472), xq + (next_row * 3456), 3456, xb_full_addr + (stage_1) * 8);
                }
                mbarrier_wait(xb_full_addr + (2 * warp_0 + (k & 1)) * 8, k >> 1 & 1);
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
            #pragma unroll 1
            for (int i = wB; i < nB; i += 10) {
                int brow = rb0 + 4 * i;
                long long bix64 = (long long)(brow / 4);
                long long brow64 = (long long)brow;
                int bob_off = xb_off + (kb & 1) * 3472;
                if (kb >= 2) {
                    if (lane_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                }
                int rs_wo_1 = (int)rs_w;
                int rs_ro_1 = (int)rs_r;
                int capo_1 = (int)capo64;
                int ix_1 = (int)bix64;
                int rk_1 = (int)rank64;
                int v = rs_wo_1 + ix_1 * 3072;
                int sc = rs_wo_1 + 4 * capo_1 * 3072 + ix_1 * 384;
                int v_0 = rs_wo_1 + (capo_1 + ix_1) * 3072;
                int sc_1 = rs_wo_1 + 4 * capo_1 * 3072 + (capo_1 + ix_1) * 384;
                int v_2 = rs_wo_1 + (2 * capo_1 + ix_1) * 3072;
                int sc_3 = rs_wo_1 + 4 * capo_1 * 3072 + (2 * capo_1 + ix_1) * 384;
                int v_4 = rs_wo_1 + (3 * capo_1 + ix_1) * 3072;
                int sc_5 = rs_wo_1 + 4 * capo_1 * 3072 + (3 * capo_1 + ix_1) * 384;
                float am = 0.0f;
                #pragma unroll 1
                for (int jj = 0; jj < 3; jj++) {
                    int pp0 = lane_1 + 64 * jj;
                    int pp1 = pp0 + 32;
                    uint32_t _fp4_wire_values_0[32];
                    uint32_t _fp4_wire_scales_0[8];
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
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_2 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_2 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_2 & 8u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[12]), "=r"(_fp4_wire_values_0[13]), "=r"(_fp4_wire_values_0[14]), "=r"(_fp4_wire_values_0[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_2 & 16u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[16]), "=r"(_fp4_wire_values_0[17]), "=r"(_fp4_wire_values_0[18]), "=r"(_fp4_wire_values_0[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_2 & 32u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[20]), "=r"(_fp4_wire_values_0[21]), "=r"(_fp4_wire_values_0[22]), "=r"(_fp4_wire_values_0[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_2 & 64u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[24]), "=r"(_fp4_wire_values_0[25]), "=r"(_fp4_wire_values_0[26]), "=r"(_fp4_wire_values_0[27]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_2 & 128u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[28]), "=r"(_fp4_wire_values_0[29]), "=r"(_fp4_wire_values_0[30]), "=r"(_fp4_wire_values_0[31]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_spend_2 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_2_0;
                        }
                        if (_wire_spend_2 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_2_1;
                        }
                        if (_wire_spend_2 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_2_2;
                        }
                        if (_wire_spend_2 & 8u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_5 + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_0[3] = (uint32_t)_wire_h16_2_3;
                        }
                        if (_wire_spend_2 & 16u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[4] = (uint32_t)_wire_h16_2_4;
                        }
                        if (_wire_spend_2 & 32u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[5] = (uint32_t)_wire_h16_2_5;
                        }
                        if (_wire_spend_2 & 64u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_6) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[6] = (uint32_t)_wire_h16_2_6;
                        }
                        if (_wire_spend_2 & 128u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_7) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_5 + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_0[7] = (uint32_t)_wire_h16_2_7;
                        }
                        _wire_vpend_2 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((_fp4_wire_values_0[12] & ~((_fp4_wire_values_0[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[13] & ~((_fp4_wire_values_0[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[14] & ~((_fp4_wire_values_0[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[15] & ~((_fp4_wire_values_0[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((_fp4_wire_values_0[16] & ~((_fp4_wire_values_0[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[17] & ~((_fp4_wire_values_0[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[18] & ~((_fp4_wire_values_0[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[19] & ~((_fp4_wire_values_0[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((_fp4_wire_values_0[20] & ~((_fp4_wire_values_0[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[21] & ~((_fp4_wire_values_0[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[22] & ~((_fp4_wire_values_0[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[23] & ~((_fp4_wire_values_0[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5) | ((uint32_t)((((_fp4_wire_values_0[24] & ~((_fp4_wire_values_0[24] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[25] & ~((_fp4_wire_values_0[25] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[26] & ~((_fp4_wire_values_0[26] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[27] & ~((_fp4_wire_values_0[27] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 6) | ((uint32_t)((((_fp4_wire_values_0[28] & ~((_fp4_wire_values_0[28] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[29] & ~((_fp4_wire_values_0[29] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[30] & ~((_fp4_wire_values_0[30] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[31] & ~((_fp4_wire_values_0[31] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 7);
                        _wire_spend_2 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[3] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[4] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[5] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[6] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[6] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 6) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[7] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[7] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 7);
                    } while ((_wire_vpend_2 | _wire_spend_2) != 0u);
                    float acc[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(_fp4_wire_scales_0[0])));
                    float _cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_e4m3x2_to_f16x2_0)));
                    float _cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(_e4m3x2_to_f16x2_0 >> 16)));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_cvt_f32_f16_0), "f"(g_s));
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_cvt_f32_f16_1), "f"(g_s));
                    float2 _f2_0 = make_float2(_mul_0, _mul_0);
                    float2 scale0 = _f2_0;
                    float2 _f2_1 = make_float2(_mul_1, _mul_1);
                    float2 scale1 = _f2_1;
                    float _unpack_fp4_0[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[0]));
                        (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(_fp4_wire_values_0[0]));
                        #endif
                    }
                    float2 _f2_2 = make_float2(acc[0], acc[1]);
                    float2 a = _f2_2;
                    float2 _f2_3 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 q = _f2_3;
                    float2 r = fma_f32x2_rn_noftz(q, scale0, a);
                    acc[0] = r.x;
                    acc[1] = r.y;
                    float2 _f2_4 = make_float2(acc[2], acc[3]);
                    float2 a_0 = _f2_4;
                    float2 _f2_5 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 q_1 = _f2_5;
                    float2 r_2 = fma_f32x2_rn_noftz(q_1, scale0, a_0);
                    acc[2] = r_2.x;
                    acc[3] = r_2.y;
                    float2 _f2_6 = make_float2(acc[4], acc[5]);
                    float2 a_3 = _f2_6;
                    float2 _f2_7 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 q_4 = _f2_7;
                    float2 r_5 = fma_f32x2_rn_noftz(q_4, scale0, a_3);
                    acc[4] = r_5.x;
                    acc[5] = r_5.y;
                    float2 _f2_8 = make_float2(acc[6], acc[7]);
                    float2 a_6 = _f2_8;
                    float2 _f2_9 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 q_7 = _f2_9;
                    float2 r_8 = fma_f32x2_rn_noftz(q_7, scale0, a_6);
                    acc[6] = r_8.x;
                    acc[7] = r_8.y;
                    float _unpack_fp4_1[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[1 + 0]));
                        (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(_fp4_wire_values_0[1 + 0]));
                        #endif
                    }
                    float2 _f2_10 = make_float2(acc[8], acc[9]);
                    float2 a_9 = _f2_10;
                    float2 _f2_11 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 q_10 = _f2_11;
                    float2 r_11 = fma_f32x2_rn_noftz(q_10, scale0, a_9);
                    acc[8] = r_11.x;
                    acc[9] = r_11.y;
                    float2 _f2_12 = make_float2(acc[10], acc[11]);
                    float2 a_12 = _f2_12;
                    float2 _f2_13 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 q_13 = _f2_13;
                    float2 r_14 = fma_f32x2_rn_noftz(q_13, scale0, a_12);
                    acc[10] = r_14.x;
                    acc[11] = r_14.y;
                    float2 _f2_14 = make_float2(acc[12], acc[13]);
                    float2 a_15 = _f2_14;
                    float2 _f2_15 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 q_16 = _f2_15;
                    float2 r_17 = fma_f32x2_rn_noftz(q_16, scale0, a_15);
                    acc[12] = r_17.x;
                    acc[13] = r_17.y;
                    float2 _f2_16 = make_float2(acc[14], acc[15]);
                    float2 a_18 = _f2_16;
                    float2 _f2_17 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 q_19 = _f2_17;
                    float2 r_20 = fma_f32x2_rn_noftz(q_19, scale0, a_18);
                    acc[14] = r_20.x;
                    acc[15] = r_20.y;
                    float _unpack_fp4_2[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[2 + 0]));
                        (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(_fp4_wire_values_0[2 + 0]));
                        #endif
                    }
                    float2 _f2_18 = make_float2(acc[16], acc[17]);
                    float2 a_21 = _f2_18;
                    float2 _f2_19 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 q_22 = _f2_19;
                    float2 r_23 = fma_f32x2_rn_noftz(q_22, scale1, a_21);
                    acc[16] = r_23.x;
                    acc[17] = r_23.y;
                    float2 _f2_20 = make_float2(acc[18], acc[19]);
                    float2 a_24 = _f2_20;
                    float2 _f2_21 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 q_25 = _f2_21;
                    float2 r_26 = fma_f32x2_rn_noftz(q_25, scale1, a_24);
                    acc[18] = r_26.x;
                    acc[19] = r_26.y;
                    float2 _f2_22 = make_float2(acc[20], acc[21]);
                    float2 a_27 = _f2_22;
                    float2 _f2_23 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 q_28 = _f2_23;
                    float2 r_29 = fma_f32x2_rn_noftz(q_28, scale1, a_27);
                    acc[20] = r_29.x;
                    acc[21] = r_29.y;
                    float2 _f2_24 = make_float2(acc[22], acc[23]);
                    float2 a_30 = _f2_24;
                    float2 _f2_25 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 q_31 = _f2_25;
                    float2 r_32 = fma_f32x2_rn_noftz(q_31, scale1, a_30);
                    acc[22] = r_32.x;
                    acc[23] = r_32.y;
                    float _unpack_fp4_3[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[3 + 0]));
                        (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(_fp4_wire_values_0[3 + 0]));
                        #endif
                    }
                    float2 _f2_26 = make_float2(acc[24], acc[25]);
                    float2 a_33 = _f2_26;
                    float2 _f2_27 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 q_34 = _f2_27;
                    float2 r_35 = fma_f32x2_rn_noftz(q_34, scale1, a_33);
                    acc[24] = r_35.x;
                    acc[25] = r_35.y;
                    float2 _f2_28 = make_float2(acc[26], acc[27]);
                    float2 a_36 = _f2_28;
                    float2 _f2_29 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 q_37 = _f2_29;
                    float2 r_38 = fma_f32x2_rn_noftz(q_37, scale1, a_36);
                    acc[26] = r_38.x;
                    acc[27] = r_38.y;
                    float2 _f2_30 = make_float2(acc[28], acc[29]);
                    float2 a_39 = _f2_30;
                    float2 _f2_31 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 q_40 = _f2_31;
                    float2 r_41 = fma_f32x2_rn_noftz(q_40, scale1, a_39);
                    acc[28] = r_41.x;
                    acc[29] = r_41.y;
                    float2 _f2_32 = make_float2(acc[30], acc[31]);
                    float2 a_42 = _f2_32;
                    float2 _f2_33 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 q_43 = _f2_33;
                    float2 r_44 = fma_f32x2_rn_noftz(q_43, scale1, a_42);
                    acc[30] = r_44.x;
                    acc[31] = r_44.y;
                    uint32_t _e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(_fp4_wire_scales_0[1])));
                    float _cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(_e4m3x2_to_f16x2_1)));
                    float _cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(_e4m3x2_to_f16x2_1 >> 16)));
                    float _mul_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_cvt_f32_f16_2), "f"(g_s_2));
                    float _mul_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_cvt_f32_f16_3), "f"(g_s_2));
                    float2 _f2_34 = make_float2(_mul_2, _mul_2);
                    float2 scale0_45 = _f2_34;
                    float2 _f2_35 = make_float2(_mul_3, _mul_3);
                    float2 scale1_46 = _f2_35;
                    float _unpack_fp4_4[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[4 + 0]));
                        (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(_fp4_wire_values_0[4 + 0]));
                        #endif
                    }
                    float2 _f2_36 = make_float2(acc[0], acc[1]);
                    float2 a_47 = _f2_36;
                    float2 _f2_37 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 q_48 = _f2_37;
                    float2 r_49 = fma_f32x2_rn_noftz(q_48, scale0_45, a_47);
                    acc[0] = r_49.x;
                    acc[1] = r_49.y;
                    float2 _f2_38 = make_float2(acc[2], acc[3]);
                    float2 a_50 = _f2_38;
                    float2 _f2_39 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 q_51 = _f2_39;
                    float2 r_52 = fma_f32x2_rn_noftz(q_51, scale0_45, a_50);
                    acc[2] = r_52.x;
                    acc[3] = r_52.y;
                    float2 _f2_40 = make_float2(acc[4], acc[5]);
                    float2 a_53 = _f2_40;
                    float2 _f2_41 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 q_54 = _f2_41;
                    float2 r_55 = fma_f32x2_rn_noftz(q_54, scale0_45, a_53);
                    acc[4] = r_55.x;
                    acc[5] = r_55.y;
                    float2 _f2_42 = make_float2(acc[6], acc[7]);
                    float2 a_56 = _f2_42;
                    float2 _f2_43 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 q_57 = _f2_43;
                    float2 r_58 = fma_f32x2_rn_noftz(q_57, scale0_45, a_56);
                    acc[6] = r_58.x;
                    acc[7] = r_58.y;
                    float _unpack_fp4_5[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[5 + 0]));
                        (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(_fp4_wire_values_0[5 + 0]));
                        #endif
                    }
                    float2 _f2_44 = make_float2(acc[8], acc[9]);
                    float2 a_59 = _f2_44;
                    float2 _f2_45 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 q_60 = _f2_45;
                    float2 r_61 = fma_f32x2_rn_noftz(q_60, scale0_45, a_59);
                    acc[8] = r_61.x;
                    acc[9] = r_61.y;
                    float2 _f2_46 = make_float2(acc[10], acc[11]);
                    float2 a_62 = _f2_46;
                    float2 _f2_47 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 q_63 = _f2_47;
                    float2 r_64 = fma_f32x2_rn_noftz(q_63, scale0_45, a_62);
                    acc[10] = r_64.x;
                    acc[11] = r_64.y;
                    float2 _f2_48 = make_float2(acc[12], acc[13]);
                    float2 a_65 = _f2_48;
                    float2 _f2_49 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 q_66 = _f2_49;
                    float2 r_67 = fma_f32x2_rn_noftz(q_66, scale0_45, a_65);
                    acc[12] = r_67.x;
                    acc[13] = r_67.y;
                    float2 _f2_50 = make_float2(acc[14], acc[15]);
                    float2 a_68 = _f2_50;
                    float2 _f2_51 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 q_69 = _f2_51;
                    float2 r_70 = fma_f32x2_rn_noftz(q_69, scale0_45, a_68);
                    acc[14] = r_70.x;
                    acc[15] = r_70.y;
                    float _unpack_fp4_6[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[6 + 0]));
                        (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(_fp4_wire_values_0[6 + 0]));
                        #endif
                    }
                    float2 _f2_52 = make_float2(acc[16], acc[17]);
                    float2 a_71 = _f2_52;
                    float2 _f2_53 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 q_72 = _f2_53;
                    float2 r_73 = fma_f32x2_rn_noftz(q_72, scale1_46, a_71);
                    acc[16] = r_73.x;
                    acc[17] = r_73.y;
                    float2 _f2_54 = make_float2(acc[18], acc[19]);
                    float2 a_74 = _f2_54;
                    float2 _f2_55 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 q_75 = _f2_55;
                    float2 r_76 = fma_f32x2_rn_noftz(q_75, scale1_46, a_74);
                    acc[18] = r_76.x;
                    acc[19] = r_76.y;
                    float2 _f2_56 = make_float2(acc[20], acc[21]);
                    float2 a_77 = _f2_56;
                    float2 _f2_57 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 q_78 = _f2_57;
                    float2 r_79 = fma_f32x2_rn_noftz(q_78, scale1_46, a_77);
                    acc[20] = r_79.x;
                    acc[21] = r_79.y;
                    float2 _f2_58 = make_float2(acc[22], acc[23]);
                    float2 a_80 = _f2_58;
                    float2 _f2_59 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 q_81 = _f2_59;
                    float2 r_82 = fma_f32x2_rn_noftz(q_81, scale1_46, a_80);
                    acc[22] = r_82.x;
                    acc[23] = r_82.y;
                    float _unpack_fp4_7[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[7 + 0]));
                        (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(_fp4_wire_values_0[7 + 0]));
                        #endif
                    }
                    float2 _f2_60 = make_float2(acc[24], acc[25]);
                    float2 a_83 = _f2_60;
                    float2 _f2_61 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 q_84 = _f2_61;
                    float2 r_85 = fma_f32x2_rn_noftz(q_84, scale1_46, a_83);
                    acc[24] = r_85.x;
                    acc[25] = r_85.y;
                    float2 _f2_62 = make_float2(acc[26], acc[27]);
                    float2 a_86 = _f2_62;
                    float2 _f2_63 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 q_87 = _f2_63;
                    float2 r_88 = fma_f32x2_rn_noftz(q_87, scale1_46, a_86);
                    acc[26] = r_88.x;
                    acc[27] = r_88.y;
                    float2 _f2_64 = make_float2(acc[28], acc[29]);
                    float2 a_89 = _f2_64;
                    float2 _f2_65 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 q_90 = _f2_65;
                    float2 r_91 = fma_f32x2_rn_noftz(q_90, scale1_46, a_89);
                    acc[28] = r_91.x;
                    acc[29] = r_91.y;
                    float2 _f2_66 = make_float2(acc[30], acc[31]);
                    float2 a_92 = _f2_66;
                    float2 _f2_67 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 q_93 = _f2_67;
                    float2 r_94 = fma_f32x2_rn_noftz(q_93, scale1_46, a_92);
                    acc[30] = r_94.x;
                    acc[31] = r_94.y;
                    uint32_t _e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(_fp4_wire_scales_0[2])));
                    float _cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(_e4m3x2_to_f16x2_2)));
                    float _cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(_e4m3x2_to_f16x2_2 >> 16)));
                    float _mul_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_cvt_f32_f16_4), "f"(g_s_3));
                    float _mul_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_cvt_f32_f16_5), "f"(g_s_3));
                    float2 _f2_68 = make_float2(_mul_4, _mul_4);
                    float2 scale0_95 = _f2_68;
                    float2 _f2_69 = make_float2(_mul_5, _mul_5);
                    float2 scale1_96 = _f2_69;
                    float _unpack_fp4_8[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[8 + 0]));
                        (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(_fp4_wire_values_0[8 + 0]));
                        #endif
                    }
                    float2 _f2_70 = make_float2(acc[0], acc[1]);
                    float2 a_97 = _f2_70;
                    float2 _f2_71 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 q_98 = _f2_71;
                    float2 r_99 = fma_f32x2_rn_noftz(q_98, scale0_95, a_97);
                    acc[0] = r_99.x;
                    acc[1] = r_99.y;
                    float2 _f2_72 = make_float2(acc[2], acc[3]);
                    float2 a_100 = _f2_72;
                    float2 _f2_73 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 q_101 = _f2_73;
                    float2 r_102 = fma_f32x2_rn_noftz(q_101, scale0_95, a_100);
                    acc[2] = r_102.x;
                    acc[3] = r_102.y;
                    float2 _f2_74 = make_float2(acc[4], acc[5]);
                    float2 a_103 = _f2_74;
                    float2 _f2_75 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 q_104 = _f2_75;
                    float2 r_105 = fma_f32x2_rn_noftz(q_104, scale0_95, a_103);
                    acc[4] = r_105.x;
                    acc[5] = r_105.y;
                    float2 _f2_76 = make_float2(acc[6], acc[7]);
                    float2 a_106 = _f2_76;
                    float2 _f2_77 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 q_107 = _f2_77;
                    float2 r_108 = fma_f32x2_rn_noftz(q_107, scale0_95, a_106);
                    acc[6] = r_108.x;
                    acc[7] = r_108.y;
                    float _unpack_fp4_9[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[9 + 0]));
                        (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(_fp4_wire_values_0[9 + 0]));
                        #endif
                    }
                    float2 _f2_78 = make_float2(acc[8], acc[9]);
                    float2 a_109 = _f2_78;
                    float2 _f2_79 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 q_110 = _f2_79;
                    float2 r_111 = fma_f32x2_rn_noftz(q_110, scale0_95, a_109);
                    acc[8] = r_111.x;
                    acc[9] = r_111.y;
                    float2 _f2_80 = make_float2(acc[10], acc[11]);
                    float2 a_112 = _f2_80;
                    float2 _f2_81 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 q_113 = _f2_81;
                    float2 r_114 = fma_f32x2_rn_noftz(q_113, scale0_95, a_112);
                    acc[10] = r_114.x;
                    acc[11] = r_114.y;
                    float2 _f2_82 = make_float2(acc[12], acc[13]);
                    float2 a_115 = _f2_82;
                    float2 _f2_83 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 q_116 = _f2_83;
                    float2 r_117 = fma_f32x2_rn_noftz(q_116, scale0_95, a_115);
                    acc[12] = r_117.x;
                    acc[13] = r_117.y;
                    float2 _f2_84 = make_float2(acc[14], acc[15]);
                    float2 a_118 = _f2_84;
                    float2 _f2_85 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 q_119 = _f2_85;
                    float2 r_120 = fma_f32x2_rn_noftz(q_119, scale0_95, a_118);
                    acc[14] = r_120.x;
                    acc[15] = r_120.y;
                    float _unpack_fp4_10[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[10 + 0]));
                        (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(_fp4_wire_values_0[10 + 0]));
                        #endif
                    }
                    float2 _f2_86 = make_float2(acc[16], acc[17]);
                    float2 a_121 = _f2_86;
                    float2 _f2_87 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 q_122 = _f2_87;
                    float2 r_123 = fma_f32x2_rn_noftz(q_122, scale1_96, a_121);
                    acc[16] = r_123.x;
                    acc[17] = r_123.y;
                    float2 _f2_88 = make_float2(acc[18], acc[19]);
                    float2 a_124 = _f2_88;
                    float2 _f2_89 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 q_125 = _f2_89;
                    float2 r_126 = fma_f32x2_rn_noftz(q_125, scale1_96, a_124);
                    acc[18] = r_126.x;
                    acc[19] = r_126.y;
                    float2 _f2_90 = make_float2(acc[20], acc[21]);
                    float2 a_127 = _f2_90;
                    float2 _f2_91 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 q_128 = _f2_91;
                    float2 r_129 = fma_f32x2_rn_noftz(q_128, scale1_96, a_127);
                    acc[20] = r_129.x;
                    acc[21] = r_129.y;
                    float2 _f2_92 = make_float2(acc[22], acc[23]);
                    float2 a_130 = _f2_92;
                    float2 _f2_93 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 q_131 = _f2_93;
                    float2 r_132 = fma_f32x2_rn_noftz(q_131, scale1_96, a_130);
                    acc[22] = r_132.x;
                    acc[23] = r_132.y;
                    float _unpack_fp4_11[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[11 + 0]));
                        (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(_fp4_wire_values_0[11 + 0]));
                        #endif
                    }
                    float2 _f2_94 = make_float2(acc[24], acc[25]);
                    float2 a_133 = _f2_94;
                    float2 _f2_95 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 q_134 = _f2_95;
                    float2 r_135 = fma_f32x2_rn_noftz(q_134, scale1_96, a_133);
                    acc[24] = r_135.x;
                    acc[25] = r_135.y;
                    float2 _f2_96 = make_float2(acc[26], acc[27]);
                    float2 a_136 = _f2_96;
                    float2 _f2_97 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 q_137 = _f2_97;
                    float2 r_138 = fma_f32x2_rn_noftz(q_137, scale1_96, a_136);
                    acc[26] = r_138.x;
                    acc[27] = r_138.y;
                    float2 _f2_98 = make_float2(acc[28], acc[29]);
                    float2 a_139 = _f2_98;
                    float2 _f2_99 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 q_140 = _f2_99;
                    float2 r_141 = fma_f32x2_rn_noftz(q_140, scale1_96, a_139);
                    acc[28] = r_141.x;
                    acc[29] = r_141.y;
                    float2 _f2_100 = make_float2(acc[30], acc[31]);
                    float2 a_142 = _f2_100;
                    float2 _f2_101 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 q_143 = _f2_101;
                    float2 r_144 = fma_f32x2_rn_noftz(q_143, scale1_96, a_142);
                    acc[30] = r_144.x;
                    acc[31] = r_144.y;
                    uint32_t _e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(_fp4_wire_scales_0[3])));
                    float _cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(_e4m3x2_to_f16x2_3)));
                    float _cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(_e4m3x2_to_f16x2_3 >> 16)));
                    float _mul_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_cvt_f32_f16_6), "f"(g_s_4));
                    float _mul_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_cvt_f32_f16_7), "f"(g_s_4));
                    float2 _f2_102 = make_float2(_mul_6, _mul_6);
                    float2 scale0_145 = _f2_102;
                    float2 _f2_103 = make_float2(_mul_7, _mul_7);
                    float2 scale1_146 = _f2_103;
                    float _unpack_fp4_12[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[12 + 0]));
                        (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(_fp4_wire_values_0[12 + 0]));
                        #endif
                    }
                    float2 _f2_104 = make_float2(acc[0], acc[1]);
                    float2 a_147 = _f2_104;
                    float2 _f2_105 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 q_148 = _f2_105;
                    float2 r_149 = fma_f32x2_rn_noftz(q_148, scale0_145, a_147);
                    acc[0] = r_149.x;
                    acc[1] = r_149.y;
                    float2 _f2_106 = make_float2(acc[2], acc[3]);
                    float2 a_150 = _f2_106;
                    float2 _f2_107 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 q_151 = _f2_107;
                    float2 r_152 = fma_f32x2_rn_noftz(q_151, scale0_145, a_150);
                    acc[2] = r_152.x;
                    acc[3] = r_152.y;
                    float2 _f2_108 = make_float2(acc[4], acc[5]);
                    float2 a_153 = _f2_108;
                    float2 _f2_109 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 q_154 = _f2_109;
                    float2 r_155 = fma_f32x2_rn_noftz(q_154, scale0_145, a_153);
                    acc[4] = r_155.x;
                    acc[5] = r_155.y;
                    float2 _f2_110 = make_float2(acc[6], acc[7]);
                    float2 a_156 = _f2_110;
                    float2 _f2_111 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 q_157 = _f2_111;
                    float2 r_158 = fma_f32x2_rn_noftz(q_157, scale0_145, a_156);
                    acc[6] = r_158.x;
                    acc[7] = r_158.y;
                    float _unpack_fp4_13[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[13 + 0]));
                        (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(_fp4_wire_values_0[13 + 0]));
                        #endif
                    }
                    float2 _f2_112 = make_float2(acc[8], acc[9]);
                    float2 a_159 = _f2_112;
                    float2 _f2_113 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 q_160 = _f2_113;
                    float2 r_161 = fma_f32x2_rn_noftz(q_160, scale0_145, a_159);
                    acc[8] = r_161.x;
                    acc[9] = r_161.y;
                    float2 _f2_114 = make_float2(acc[10], acc[11]);
                    float2 a_162 = _f2_114;
                    float2 _f2_115 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 q_163 = _f2_115;
                    float2 r_164 = fma_f32x2_rn_noftz(q_163, scale0_145, a_162);
                    acc[10] = r_164.x;
                    acc[11] = r_164.y;
                    float2 _f2_116 = make_float2(acc[12], acc[13]);
                    float2 a_165 = _f2_116;
                    float2 _f2_117 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 q_166 = _f2_117;
                    float2 r_167 = fma_f32x2_rn_noftz(q_166, scale0_145, a_165);
                    acc[12] = r_167.x;
                    acc[13] = r_167.y;
                    float2 _f2_118 = make_float2(acc[14], acc[15]);
                    float2 a_168 = _f2_118;
                    float2 _f2_119 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 q_169 = _f2_119;
                    float2 r_170 = fma_f32x2_rn_noftz(q_169, scale0_145, a_168);
                    acc[14] = r_170.x;
                    acc[15] = r_170.y;
                    float _unpack_fp4_14[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[14 + 0]));
                        (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(_fp4_wire_values_0[14 + 0]));
                        #endif
                    }
                    float2 _f2_120 = make_float2(acc[16], acc[17]);
                    float2 a_171 = _f2_120;
                    float2 _f2_121 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 q_172 = _f2_121;
                    float2 r_173 = fma_f32x2_rn_noftz(q_172, scale1_146, a_171);
                    acc[16] = r_173.x;
                    acc[17] = r_173.y;
                    float2 _f2_122 = make_float2(acc[18], acc[19]);
                    float2 a_174 = _f2_122;
                    float2 _f2_123 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 q_175 = _f2_123;
                    float2 r_176 = fma_f32x2_rn_noftz(q_175, scale1_146, a_174);
                    acc[18] = r_176.x;
                    acc[19] = r_176.y;
                    float2 _f2_124 = make_float2(acc[20], acc[21]);
                    float2 a_177 = _f2_124;
                    float2 _f2_125 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 q_178 = _f2_125;
                    float2 r_179 = fma_f32x2_rn_noftz(q_178, scale1_146, a_177);
                    acc[20] = r_179.x;
                    acc[21] = r_179.y;
                    float2 _f2_126 = make_float2(acc[22], acc[23]);
                    float2 a_180 = _f2_126;
                    float2 _f2_127 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 q_181 = _f2_127;
                    float2 r_182 = fma_f32x2_rn_noftz(q_181, scale1_146, a_180);
                    acc[22] = r_182.x;
                    acc[23] = r_182.y;
                    float _unpack_fp4_15[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[15 + 0]));
                        (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(_fp4_wire_values_0[15 + 0]));
                        #endif
                    }
                    float2 _f2_128 = make_float2(acc[24], acc[25]);
                    float2 a_183 = _f2_128;
                    float2 _f2_129 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 q_184 = _f2_129;
                    float2 r_185 = fma_f32x2_rn_noftz(q_184, scale1_146, a_183);
                    acc[24] = r_185.x;
                    acc[25] = r_185.y;
                    float2 _f2_130 = make_float2(acc[26], acc[27]);
                    float2 a_186 = _f2_130;
                    float2 _f2_131 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 q_187 = _f2_131;
                    float2 r_188 = fma_f32x2_rn_noftz(q_187, scale1_146, a_186);
                    acc[26] = r_188.x;
                    acc[27] = r_188.y;
                    float2 _f2_132 = make_float2(acc[28], acc[29]);
                    float2 a_189 = _f2_132;
                    float2 _f2_133 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 q_190 = _f2_133;
                    float2 r_191 = fma_f32x2_rn_noftz(q_190, scale1_146, a_189);
                    acc[28] = r_191.x;
                    acc[29] = r_191.y;
                    float2 _f2_134 = make_float2(acc[30], acc[31]);
                    float2 a_192 = _f2_134;
                    float2 _f2_135 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 q_193 = _f2_135;
                    float2 r_194 = fma_f32x2_rn_noftz(q_193, scale1_146, a_192);
                    acc[30] = r_194.x;
                    acc[31] = r_194.y;
                    float cur = am;
                    float _fabs_0;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_0) : "f"(acc[0]));
                    float _max_0 = max_noftz(cur, _fabs_0);
                    cur = _max_0;
                    float _fabs_1;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_1) : "f"(acc[1]));
                    float _max_1 = max_noftz(cur, _fabs_1);
                    cur = _max_1;
                    float _fabs_2;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_2) : "f"(acc[2]));
                    float _max_2 = max_noftz(cur, _fabs_2);
                    cur = _max_2;
                    float _fabs_3;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_3) : "f"(acc[3]));
                    float _max_3 = max_noftz(cur, _fabs_3);
                    cur = _max_3;
                    float _fabs_4;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_4) : "f"(acc[4]));
                    float _max_4 = max_noftz(cur, _fabs_4);
                    cur = _max_4;
                    float _fabs_5;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_5) : "f"(acc[5]));
                    float _max_5 = max_noftz(cur, _fabs_5);
                    cur = _max_5;
                    float _fabs_6;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_6) : "f"(acc[6]));
                    float _max_6 = max_noftz(cur, _fabs_6);
                    cur = _max_6;
                    float _fabs_7;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_7) : "f"(acc[7]));
                    float _max_7 = max_noftz(cur, _fabs_7);
                    cur = _max_7;
                    float _fabs_8;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_8) : "f"(acc[8]));
                    float _max_8 = max_noftz(cur, _fabs_8);
                    cur = _max_8;
                    float _fabs_9;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_9) : "f"(acc[9]));
                    float _max_9 = max_noftz(cur, _fabs_9);
                    cur = _max_9;
                    float _fabs_10;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_10) : "f"(acc[10]));
                    float _max_10 = max_noftz(cur, _fabs_10);
                    cur = _max_10;
                    float _fabs_11;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_11) : "f"(acc[11]));
                    float _max_11 = max_noftz(cur, _fabs_11);
                    cur = _max_11;
                    float _fabs_12;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_12) : "f"(acc[12]));
                    float _max_12 = max_noftz(cur, _fabs_12);
                    cur = _max_12;
                    float _fabs_13;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_13) : "f"(acc[13]));
                    float _max_13 = max_noftz(cur, _fabs_13);
                    cur = _max_13;
                    float _fabs_14;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_14) : "f"(acc[14]));
                    float _max_14 = max_noftz(cur, _fabs_14);
                    cur = _max_14;
                    float _fabs_15;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_15) : "f"(acc[15]));
                    float _max_15 = max_noftz(cur, _fabs_15);
                    cur = _max_15;
                    float _fabs_16;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_16) : "f"(acc[16]));
                    float _max_16 = max_noftz(cur, _fabs_16);
                    cur = _max_16;
                    float _fabs_17;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_17) : "f"(acc[17]));
                    float _max_17 = max_noftz(cur, _fabs_17);
                    cur = _max_17;
                    float _fabs_18;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_18) : "f"(acc[18]));
                    float _max_18 = max_noftz(cur, _fabs_18);
                    cur = _max_18;
                    float _fabs_19;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_19) : "f"(acc[19]));
                    float _max_19 = max_noftz(cur, _fabs_19);
                    cur = _max_19;
                    float _fabs_20;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_20) : "f"(acc[20]));
                    float _max_20 = max_noftz(cur, _fabs_20);
                    cur = _max_20;
                    float _fabs_21;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_21) : "f"(acc[21]));
                    float _max_21 = max_noftz(cur, _fabs_21);
                    cur = _max_21;
                    float _fabs_22;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_22) : "f"(acc[22]));
                    float _max_22 = max_noftz(cur, _fabs_22);
                    cur = _max_22;
                    float _fabs_23;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_23) : "f"(acc[23]));
                    float _max_23 = max_noftz(cur, _fabs_23);
                    cur = _max_23;
                    float _fabs_24;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_24) : "f"(acc[24]));
                    float _max_24 = max_noftz(cur, _fabs_24);
                    cur = _max_24;
                    float _fabs_25;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_25) : "f"(acc[25]));
                    float _max_25 = max_noftz(cur, _fabs_25);
                    cur = _max_25;
                    float _fabs_26;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_26) : "f"(acc[26]));
                    float _max_26 = max_noftz(cur, _fabs_26);
                    cur = _max_26;
                    float _fabs_27;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_27) : "f"(acc[27]));
                    float _max_27 = max_noftz(cur, _fabs_27);
                    cur = _max_27;
                    float _fabs_28;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_28) : "f"(acc[28]));
                    float _max_28 = max_noftz(cur, _fabs_28);
                    cur = _max_28;
                    float _fabs_29;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_29) : "f"(acc[29]));
                    float _max_29 = max_noftz(cur, _fabs_29);
                    cur = _max_29;
                    float _fabs_30;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_30) : "f"(acc[30]));
                    float _max_30 = max_noftz(cur, _fabs_30);
                    cur = _max_30;
                    float _fabs_31;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_31) : "f"(acc[31]));
                    float _max_31 = max_noftz(cur, _fabs_31);
                    cur = _max_31;
                    am = cur;
                    float acc_195[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(_fp4_wire_scales_0[4])));
                    float _cvt_f32_f16_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(_e4m3x2_to_f16x2_4)));
                    float _cvt_f32_f16_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(_e4m3x2_to_f16x2_4 >> 16)));
                    float _mul_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_cvt_f32_f16_8), "f"(g_s));
                    float _mul_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_cvt_f32_f16_9), "f"(g_s));
                    float2 _f2_136 = make_float2(_mul_8, _mul_8);
                    float2 scale0_196 = _f2_136;
                    float2 _f2_137 = make_float2(_mul_9, _mul_9);
                    float2 scale1_197 = _f2_137;
                    float _unpack_fp4_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[16 + 0]));
                        (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(_fp4_wire_values_0[16 + 0]));
                        #endif
                    }
                    float2 _f2_138 = make_float2(acc_195[0], acc_195[1]);
                    float2 a_198 = _f2_138;
                    float2 _f2_139 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 q_199 = _f2_139;
                    float2 r_200 = fma_f32x2_rn_noftz(q_199, scale0_196, a_198);
                    acc_195[0] = r_200.x;
                    acc_195[1] = r_200.y;
                    float2 _f2_140 = make_float2(acc_195[2], acc_195[3]);
                    float2 a_201 = _f2_140;
                    float2 _f2_141 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 q_202 = _f2_141;
                    float2 r_203 = fma_f32x2_rn_noftz(q_202, scale0_196, a_201);
                    acc_195[2] = r_203.x;
                    acc_195[3] = r_203.y;
                    float2 _f2_142 = make_float2(acc_195[4], acc_195[5]);
                    float2 a_204 = _f2_142;
                    float2 _f2_143 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 q_205 = _f2_143;
                    float2 r_206 = fma_f32x2_rn_noftz(q_205, scale0_196, a_204);
                    acc_195[4] = r_206.x;
                    acc_195[5] = r_206.y;
                    float2 _f2_144 = make_float2(acc_195[6], acc_195[7]);
                    float2 a_207 = _f2_144;
                    float2 _f2_145 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 q_208 = _f2_145;
                    float2 r_209 = fma_f32x2_rn_noftz(q_208, scale0_196, a_207);
                    acc_195[6] = r_209.x;
                    acc_195[7] = r_209.y;
                    float _unpack_fp4_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[17 + 0]));
                        (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(_fp4_wire_values_0[17 + 0]));
                        #endif
                    }
                    float2 _f2_146 = make_float2(acc_195[8], acc_195[9]);
                    float2 a_210 = _f2_146;
                    float2 _f2_147 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 q_211 = _f2_147;
                    float2 r_212 = fma_f32x2_rn_noftz(q_211, scale0_196, a_210);
                    acc_195[8] = r_212.x;
                    acc_195[9] = r_212.y;
                    float2 _f2_148 = make_float2(acc_195[10], acc_195[11]);
                    float2 a_213 = _f2_148;
                    float2 _f2_149 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 q_214 = _f2_149;
                    float2 r_215 = fma_f32x2_rn_noftz(q_214, scale0_196, a_213);
                    acc_195[10] = r_215.x;
                    acc_195[11] = r_215.y;
                    float2 _f2_150 = make_float2(acc_195[12], acc_195[13]);
                    float2 a_216 = _f2_150;
                    float2 _f2_151 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 q_217 = _f2_151;
                    float2 r_218 = fma_f32x2_rn_noftz(q_217, scale0_196, a_216);
                    acc_195[12] = r_218.x;
                    acc_195[13] = r_218.y;
                    float2 _f2_152 = make_float2(acc_195[14], acc_195[15]);
                    float2 a_219 = _f2_152;
                    float2 _f2_153 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 q_220 = _f2_153;
                    float2 r_221 = fma_f32x2_rn_noftz(q_220, scale0_196, a_219);
                    acc_195[14] = r_221.x;
                    acc_195[15] = r_221.y;
                    float _unpack_fp4_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[18 + 0]));
                        (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(_fp4_wire_values_0[18 + 0]));
                        #endif
                    }
                    float2 _f2_154 = make_float2(acc_195[16], acc_195[17]);
                    float2 a_222 = _f2_154;
                    float2 _f2_155 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 q_223 = _f2_155;
                    float2 r_224 = fma_f32x2_rn_noftz(q_223, scale1_197, a_222);
                    acc_195[16] = r_224.x;
                    acc_195[17] = r_224.y;
                    float2 _f2_156 = make_float2(acc_195[18], acc_195[19]);
                    float2 a_225 = _f2_156;
                    float2 _f2_157 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 q_226 = _f2_157;
                    float2 r_227 = fma_f32x2_rn_noftz(q_226, scale1_197, a_225);
                    acc_195[18] = r_227.x;
                    acc_195[19] = r_227.y;
                    float2 _f2_158 = make_float2(acc_195[20], acc_195[21]);
                    float2 a_228 = _f2_158;
                    float2 _f2_159 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 q_229 = _f2_159;
                    float2 r_230 = fma_f32x2_rn_noftz(q_229, scale1_197, a_228);
                    acc_195[20] = r_230.x;
                    acc_195[21] = r_230.y;
                    float2 _f2_160 = make_float2(acc_195[22], acc_195[23]);
                    float2 a_231 = _f2_160;
                    float2 _f2_161 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 q_232 = _f2_161;
                    float2 r_233 = fma_f32x2_rn_noftz(q_232, scale1_197, a_231);
                    acc_195[22] = r_233.x;
                    acc_195[23] = r_233.y;
                    float _unpack_fp4_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[19 + 0]));
                        (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(_fp4_wire_values_0[19 + 0]));
                        #endif
                    }
                    float2 _f2_162 = make_float2(acc_195[24], acc_195[25]);
                    float2 a_234 = _f2_162;
                    float2 _f2_163 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 q_235 = _f2_163;
                    float2 r_236 = fma_f32x2_rn_noftz(q_235, scale1_197, a_234);
                    acc_195[24] = r_236.x;
                    acc_195[25] = r_236.y;
                    float2 _f2_164 = make_float2(acc_195[26], acc_195[27]);
                    float2 a_237 = _f2_164;
                    float2 _f2_165 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 q_238 = _f2_165;
                    float2 r_239 = fma_f32x2_rn_noftz(q_238, scale1_197, a_237);
                    acc_195[26] = r_239.x;
                    acc_195[27] = r_239.y;
                    float2 _f2_166 = make_float2(acc_195[28], acc_195[29]);
                    float2 a_240 = _f2_166;
                    float2 _f2_167 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 q_241 = _f2_167;
                    float2 r_242 = fma_f32x2_rn_noftz(q_241, scale1_197, a_240);
                    acc_195[28] = r_242.x;
                    acc_195[29] = r_242.y;
                    float2 _f2_168 = make_float2(acc_195[30], acc_195[31]);
                    float2 a_243 = _f2_168;
                    float2 _f2_169 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 q_244 = _f2_169;
                    float2 r_245 = fma_f32x2_rn_noftz(q_244, scale1_197, a_243);
                    acc_195[30] = r_245.x;
                    acc_195[31] = r_245.y;
                    uint32_t _e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(_fp4_wire_scales_0[5])));
                    float _cvt_f32_f16_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(_e4m3x2_to_f16x2_5)));
                    float _cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(_e4m3x2_to_f16x2_5 >> 16)));
                    float _mul_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_cvt_f32_f16_10), "f"(g_s_2));
                    float _mul_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_cvt_f32_f16_11), "f"(g_s_2));
                    float2 _f2_170 = make_float2(_mul_10, _mul_10);
                    float2 scale0_246 = _f2_170;
                    float2 _f2_171 = make_float2(_mul_11, _mul_11);
                    float2 scale1_247 = _f2_171;
                    float _unpack_fp4_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[20 + 0]));
                        (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(_fp4_wire_values_0[20 + 0]));
                        #endif
                    }
                    float2 _f2_172 = make_float2(acc_195[0], acc_195[1]);
                    float2 a_248 = _f2_172;
                    float2 _f2_173 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 q_249 = _f2_173;
                    float2 r_250 = fma_f32x2_rn_noftz(q_249, scale0_246, a_248);
                    acc_195[0] = r_250.x;
                    acc_195[1] = r_250.y;
                    float2 _f2_174 = make_float2(acc_195[2], acc_195[3]);
                    float2 a_251 = _f2_174;
                    float2 _f2_175 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 q_252 = _f2_175;
                    float2 r_253 = fma_f32x2_rn_noftz(q_252, scale0_246, a_251);
                    acc_195[2] = r_253.x;
                    acc_195[3] = r_253.y;
                    float2 _f2_176 = make_float2(acc_195[4], acc_195[5]);
                    float2 a_254 = _f2_176;
                    float2 _f2_177 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 q_255 = _f2_177;
                    float2 r_256 = fma_f32x2_rn_noftz(q_255, scale0_246, a_254);
                    acc_195[4] = r_256.x;
                    acc_195[5] = r_256.y;
                    float2 _f2_178 = make_float2(acc_195[6], acc_195[7]);
                    float2 a_257 = _f2_178;
                    float2 _f2_179 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 q_258 = _f2_179;
                    float2 r_259 = fma_f32x2_rn_noftz(q_258, scale0_246, a_257);
                    acc_195[6] = r_259.x;
                    acc_195[7] = r_259.y;
                    float _unpack_fp4_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[21 + 0]));
                        (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(_fp4_wire_values_0[21 + 0]));
                        #endif
                    }
                    float2 _f2_180 = make_float2(acc_195[8], acc_195[9]);
                    float2 a_260 = _f2_180;
                    float2 _f2_181 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 q_261 = _f2_181;
                    float2 r_262 = fma_f32x2_rn_noftz(q_261, scale0_246, a_260);
                    acc_195[8] = r_262.x;
                    acc_195[9] = r_262.y;
                    float2 _f2_182 = make_float2(acc_195[10], acc_195[11]);
                    float2 a_263 = _f2_182;
                    float2 _f2_183 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 q_264 = _f2_183;
                    float2 r_265 = fma_f32x2_rn_noftz(q_264, scale0_246, a_263);
                    acc_195[10] = r_265.x;
                    acc_195[11] = r_265.y;
                    float2 _f2_184 = make_float2(acc_195[12], acc_195[13]);
                    float2 a_266 = _f2_184;
                    float2 _f2_185 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 q_267 = _f2_185;
                    float2 r_268 = fma_f32x2_rn_noftz(q_267, scale0_246, a_266);
                    acc_195[12] = r_268.x;
                    acc_195[13] = r_268.y;
                    float2 _f2_186 = make_float2(acc_195[14], acc_195[15]);
                    float2 a_269 = _f2_186;
                    float2 _f2_187 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 q_270 = _f2_187;
                    float2 r_271 = fma_f32x2_rn_noftz(q_270, scale0_246, a_269);
                    acc_195[14] = r_271.x;
                    acc_195[15] = r_271.y;
                    float _unpack_fp4_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[22 + 0]));
                        (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(_fp4_wire_values_0[22 + 0]));
                        #endif
                    }
                    float2 _f2_188 = make_float2(acc_195[16], acc_195[17]);
                    float2 a_272 = _f2_188;
                    float2 _f2_189 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 q_273 = _f2_189;
                    float2 r_274 = fma_f32x2_rn_noftz(q_273, scale1_247, a_272);
                    acc_195[16] = r_274.x;
                    acc_195[17] = r_274.y;
                    float2 _f2_190 = make_float2(acc_195[18], acc_195[19]);
                    float2 a_275 = _f2_190;
                    float2 _f2_191 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 q_276 = _f2_191;
                    float2 r_277 = fma_f32x2_rn_noftz(q_276, scale1_247, a_275);
                    acc_195[18] = r_277.x;
                    acc_195[19] = r_277.y;
                    float2 _f2_192 = make_float2(acc_195[20], acc_195[21]);
                    float2 a_278 = _f2_192;
                    float2 _f2_193 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 q_279 = _f2_193;
                    float2 r_280 = fma_f32x2_rn_noftz(q_279, scale1_247, a_278);
                    acc_195[20] = r_280.x;
                    acc_195[21] = r_280.y;
                    float2 _f2_194 = make_float2(acc_195[22], acc_195[23]);
                    float2 a_281 = _f2_194;
                    float2 _f2_195 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 q_282 = _f2_195;
                    float2 r_283 = fma_f32x2_rn_noftz(q_282, scale1_247, a_281);
                    acc_195[22] = r_283.x;
                    acc_195[23] = r_283.y;
                    float _unpack_fp4_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[23 + 0]));
                        (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(_fp4_wire_values_0[23 + 0]));
                        #endif
                    }
                    float2 _f2_196 = make_float2(acc_195[24], acc_195[25]);
                    float2 a_284 = _f2_196;
                    float2 _f2_197 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 q_285 = _f2_197;
                    float2 r_286 = fma_f32x2_rn_noftz(q_285, scale1_247, a_284);
                    acc_195[24] = r_286.x;
                    acc_195[25] = r_286.y;
                    float2 _f2_198 = make_float2(acc_195[26], acc_195[27]);
                    float2 a_287 = _f2_198;
                    float2 _f2_199 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 q_288 = _f2_199;
                    float2 r_289 = fma_f32x2_rn_noftz(q_288, scale1_247, a_287);
                    acc_195[26] = r_289.x;
                    acc_195[27] = r_289.y;
                    float2 _f2_200 = make_float2(acc_195[28], acc_195[29]);
                    float2 a_290 = _f2_200;
                    float2 _f2_201 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 q_291 = _f2_201;
                    float2 r_292 = fma_f32x2_rn_noftz(q_291, scale1_247, a_290);
                    acc_195[28] = r_292.x;
                    acc_195[29] = r_292.y;
                    float2 _f2_202 = make_float2(acc_195[30], acc_195[31]);
                    float2 a_293 = _f2_202;
                    float2 _f2_203 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 q_294 = _f2_203;
                    float2 r_295 = fma_f32x2_rn_noftz(q_294, scale1_247, a_293);
                    acc_195[30] = r_295.x;
                    acc_195[31] = r_295.y;
                    uint32_t _e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)(_fp4_wire_scales_0[6])));
                    float _cvt_f32_f16_12;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(_e4m3x2_to_f16x2_6)));
                    float _cvt_f32_f16_13;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(_e4m3x2_to_f16x2_6 >> 16)));
                    float _mul_12;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_cvt_f32_f16_12), "f"(g_s_3));
                    float _mul_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_cvt_f32_f16_13), "f"(g_s_3));
                    float2 _f2_204 = make_float2(_mul_12, _mul_12);
                    float2 scale0_296 = _f2_204;
                    float2 _f2_205 = make_float2(_mul_13, _mul_13);
                    float2 scale1_297 = _f2_205;
                    float _unpack_fp4_24[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[24 + 0]));
                        (&_unpack_fp4_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_24[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_24[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_24[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_24[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_24[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_24[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_24[0]), "=f"(_unpack_fp4_24[1]), "=f"(_unpack_fp4_24[2]), "=f"(_unpack_fp4_24[3]), "=f"(_unpack_fp4_24[4]), "=f"(_unpack_fp4_24[5]), "=f"(_unpack_fp4_24[6]), "=f"(_unpack_fp4_24[7]) : "r"(_fp4_wire_values_0[24 + 0]));
                        #endif
                    }
                    float2 _f2_206 = make_float2(acc_195[0], acc_195[1]);
                    float2 a_298 = _f2_206;
                    float2 _f2_207 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                    float2 q_299 = _f2_207;
                    float2 r_300 = fma_f32x2_rn_noftz(q_299, scale0_296, a_298);
                    acc_195[0] = r_300.x;
                    acc_195[1] = r_300.y;
                    float2 _f2_208 = make_float2(acc_195[2], acc_195[3]);
                    float2 a_301 = _f2_208;
                    float2 _f2_209 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                    float2 q_302 = _f2_209;
                    float2 r_303 = fma_f32x2_rn_noftz(q_302, scale0_296, a_301);
                    acc_195[2] = r_303.x;
                    acc_195[3] = r_303.y;
                    float2 _f2_210 = make_float2(acc_195[4], acc_195[5]);
                    float2 a_304 = _f2_210;
                    float2 _f2_211 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                    float2 q_305 = _f2_211;
                    float2 r_306 = fma_f32x2_rn_noftz(q_305, scale0_296, a_304);
                    acc_195[4] = r_306.x;
                    acc_195[5] = r_306.y;
                    float2 _f2_212 = make_float2(acc_195[6], acc_195[7]);
                    float2 a_307 = _f2_212;
                    float2 _f2_213 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                    float2 q_308 = _f2_213;
                    float2 r_309 = fma_f32x2_rn_noftz(q_308, scale0_296, a_307);
                    acc_195[6] = r_309.x;
                    acc_195[7] = r_309.y;
                    float _unpack_fp4_25[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[25 + 0]));
                        (&_unpack_fp4_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_25[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_25[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_25[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_25[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_25[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_25[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_25[0]), "=f"(_unpack_fp4_25[1]), "=f"(_unpack_fp4_25[2]), "=f"(_unpack_fp4_25[3]), "=f"(_unpack_fp4_25[4]), "=f"(_unpack_fp4_25[5]), "=f"(_unpack_fp4_25[6]), "=f"(_unpack_fp4_25[7]) : "r"(_fp4_wire_values_0[25 + 0]));
                        #endif
                    }
                    float2 _f2_214 = make_float2(acc_195[8], acc_195[9]);
                    float2 a_310 = _f2_214;
                    float2 _f2_215 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                    float2 q_311 = _f2_215;
                    float2 r_312 = fma_f32x2_rn_noftz(q_311, scale0_296, a_310);
                    acc_195[8] = r_312.x;
                    acc_195[9] = r_312.y;
                    float2 _f2_216 = make_float2(acc_195[10], acc_195[11]);
                    float2 a_313 = _f2_216;
                    float2 _f2_217 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                    float2 q_314 = _f2_217;
                    float2 r_315 = fma_f32x2_rn_noftz(q_314, scale0_296, a_313);
                    acc_195[10] = r_315.x;
                    acc_195[11] = r_315.y;
                    float2 _f2_218 = make_float2(acc_195[12], acc_195[13]);
                    float2 a_316 = _f2_218;
                    float2 _f2_219 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                    float2 q_317 = _f2_219;
                    float2 r_318 = fma_f32x2_rn_noftz(q_317, scale0_296, a_316);
                    acc_195[12] = r_318.x;
                    acc_195[13] = r_318.y;
                    float2 _f2_220 = make_float2(acc_195[14], acc_195[15]);
                    float2 a_319 = _f2_220;
                    float2 _f2_221 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                    float2 q_320 = _f2_221;
                    float2 r_321 = fma_f32x2_rn_noftz(q_320, scale0_296, a_319);
                    acc_195[14] = r_321.x;
                    acc_195[15] = r_321.y;
                    float _unpack_fp4_26[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[26 + 0]));
                        (&_unpack_fp4_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_26[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_26[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_26[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_26[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_26[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_26[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_26[0]), "=f"(_unpack_fp4_26[1]), "=f"(_unpack_fp4_26[2]), "=f"(_unpack_fp4_26[3]), "=f"(_unpack_fp4_26[4]), "=f"(_unpack_fp4_26[5]), "=f"(_unpack_fp4_26[6]), "=f"(_unpack_fp4_26[7]) : "r"(_fp4_wire_values_0[26 + 0]));
                        #endif
                    }
                    float2 _f2_222 = make_float2(acc_195[16], acc_195[17]);
                    float2 a_322 = _f2_222;
                    float2 _f2_223 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                    float2 q_323 = _f2_223;
                    float2 r_324 = fma_f32x2_rn_noftz(q_323, scale1_297, a_322);
                    acc_195[16] = r_324.x;
                    acc_195[17] = r_324.y;
                    float2 _f2_224 = make_float2(acc_195[18], acc_195[19]);
                    float2 a_325 = _f2_224;
                    float2 _f2_225 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                    float2 q_326 = _f2_225;
                    float2 r_327 = fma_f32x2_rn_noftz(q_326, scale1_297, a_325);
                    acc_195[18] = r_327.x;
                    acc_195[19] = r_327.y;
                    float2 _f2_226 = make_float2(acc_195[20], acc_195[21]);
                    float2 a_328 = _f2_226;
                    float2 _f2_227 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                    float2 q_329 = _f2_227;
                    float2 r_330 = fma_f32x2_rn_noftz(q_329, scale1_297, a_328);
                    acc_195[20] = r_330.x;
                    acc_195[21] = r_330.y;
                    float2 _f2_228 = make_float2(acc_195[22], acc_195[23]);
                    float2 a_331 = _f2_228;
                    float2 _f2_229 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                    float2 q_332 = _f2_229;
                    float2 r_333 = fma_f32x2_rn_noftz(q_332, scale1_297, a_331);
                    acc_195[22] = r_333.x;
                    acc_195[23] = r_333.y;
                    float _unpack_fp4_27[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[27 + 0]));
                        (&_unpack_fp4_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_27[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_27[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_27[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_27[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_27[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_27[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_27[0]), "=f"(_unpack_fp4_27[1]), "=f"(_unpack_fp4_27[2]), "=f"(_unpack_fp4_27[3]), "=f"(_unpack_fp4_27[4]), "=f"(_unpack_fp4_27[5]), "=f"(_unpack_fp4_27[6]), "=f"(_unpack_fp4_27[7]) : "r"(_fp4_wire_values_0[27 + 0]));
                        #endif
                    }
                    float2 _f2_230 = make_float2(acc_195[24], acc_195[25]);
                    float2 a_334 = _f2_230;
                    float2 _f2_231 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                    float2 q_335 = _f2_231;
                    float2 r_336 = fma_f32x2_rn_noftz(q_335, scale1_297, a_334);
                    acc_195[24] = r_336.x;
                    acc_195[25] = r_336.y;
                    float2 _f2_232 = make_float2(acc_195[26], acc_195[27]);
                    float2 a_337 = _f2_232;
                    float2 _f2_233 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                    float2 q_338 = _f2_233;
                    float2 r_339 = fma_f32x2_rn_noftz(q_338, scale1_297, a_337);
                    acc_195[26] = r_339.x;
                    acc_195[27] = r_339.y;
                    float2 _f2_234 = make_float2(acc_195[28], acc_195[29]);
                    float2 a_340 = _f2_234;
                    float2 _f2_235 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                    float2 q_341 = _f2_235;
                    float2 r_342 = fma_f32x2_rn_noftz(q_341, scale1_297, a_340);
                    acc_195[28] = r_342.x;
                    acc_195[29] = r_342.y;
                    float2 _f2_236 = make_float2(acc_195[30], acc_195[31]);
                    float2 a_343 = _f2_236;
                    float2 _f2_237 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                    float2 q_344 = _f2_237;
                    float2 r_345 = fma_f32x2_rn_noftz(q_344, scale1_297, a_343);
                    acc_195[30] = r_345.x;
                    acc_195[31] = r_345.y;
                    uint32_t _e4m3x2_to_f16x2_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)(_fp4_wire_scales_0[7])));
                    float _cvt_f32_f16_14;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(_e4m3x2_to_f16x2_7)));
                    float _cvt_f32_f16_15;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(_e4m3x2_to_f16x2_7 >> 16)));
                    float _mul_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_cvt_f32_f16_14), "f"(g_s_4));
                    float _mul_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_cvt_f32_f16_15), "f"(g_s_4));
                    float2 _f2_238 = make_float2(_mul_14, _mul_14);
                    float2 scale0_346 = _f2_238;
                    float2 _f2_239 = make_float2(_mul_15, _mul_15);
                    float2 scale1_347 = _f2_239;
                    float _unpack_fp4_28[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[28 + 0]));
                        (&_unpack_fp4_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_28[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_28[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_28[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_28[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_28[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_28[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_28[0]), "=f"(_unpack_fp4_28[1]), "=f"(_unpack_fp4_28[2]), "=f"(_unpack_fp4_28[3]), "=f"(_unpack_fp4_28[4]), "=f"(_unpack_fp4_28[5]), "=f"(_unpack_fp4_28[6]), "=f"(_unpack_fp4_28[7]) : "r"(_fp4_wire_values_0[28 + 0]));
                        #endif
                    }
                    float2 _f2_240 = make_float2(acc_195[0], acc_195[1]);
                    float2 a_348 = _f2_240;
                    float2 _f2_241 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                    float2 q_349 = _f2_241;
                    float2 r_350 = fma_f32x2_rn_noftz(q_349, scale0_346, a_348);
                    acc_195[0] = r_350.x;
                    acc_195[1] = r_350.y;
                    float2 _f2_242 = make_float2(acc_195[2], acc_195[3]);
                    float2 a_351 = _f2_242;
                    float2 _f2_243 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                    float2 q_352 = _f2_243;
                    float2 r_353 = fma_f32x2_rn_noftz(q_352, scale0_346, a_351);
                    acc_195[2] = r_353.x;
                    acc_195[3] = r_353.y;
                    float2 _f2_244 = make_float2(acc_195[4], acc_195[5]);
                    float2 a_354 = _f2_244;
                    float2 _f2_245 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                    float2 q_355 = _f2_245;
                    float2 r_356 = fma_f32x2_rn_noftz(q_355, scale0_346, a_354);
                    acc_195[4] = r_356.x;
                    acc_195[5] = r_356.y;
                    float2 _f2_246 = make_float2(acc_195[6], acc_195[7]);
                    float2 a_357 = _f2_246;
                    float2 _f2_247 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                    float2 q_358 = _f2_247;
                    float2 r_359 = fma_f32x2_rn_noftz(q_358, scale0_346, a_357);
                    acc_195[6] = r_359.x;
                    acc_195[7] = r_359.y;
                    float _unpack_fp4_29[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[29 + 0]));
                        (&_unpack_fp4_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_29[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_29[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_29[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_29[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_29[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_29[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_29[0]), "=f"(_unpack_fp4_29[1]), "=f"(_unpack_fp4_29[2]), "=f"(_unpack_fp4_29[3]), "=f"(_unpack_fp4_29[4]), "=f"(_unpack_fp4_29[5]), "=f"(_unpack_fp4_29[6]), "=f"(_unpack_fp4_29[7]) : "r"(_fp4_wire_values_0[29 + 0]));
                        #endif
                    }
                    float2 _f2_248 = make_float2(acc_195[8], acc_195[9]);
                    float2 a_360 = _f2_248;
                    float2 _f2_249 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                    float2 q_361 = _f2_249;
                    float2 r_362 = fma_f32x2_rn_noftz(q_361, scale0_346, a_360);
                    acc_195[8] = r_362.x;
                    acc_195[9] = r_362.y;
                    float2 _f2_250 = make_float2(acc_195[10], acc_195[11]);
                    float2 a_363 = _f2_250;
                    float2 _f2_251 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                    float2 q_364 = _f2_251;
                    float2 r_365 = fma_f32x2_rn_noftz(q_364, scale0_346, a_363);
                    acc_195[10] = r_365.x;
                    acc_195[11] = r_365.y;
                    float2 _f2_252 = make_float2(acc_195[12], acc_195[13]);
                    float2 a_366 = _f2_252;
                    float2 _f2_253 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                    float2 q_367 = _f2_253;
                    float2 r_368 = fma_f32x2_rn_noftz(q_367, scale0_346, a_366);
                    acc_195[12] = r_368.x;
                    acc_195[13] = r_368.y;
                    float2 _f2_254 = make_float2(acc_195[14], acc_195[15]);
                    float2 a_369 = _f2_254;
                    float2 _f2_255 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                    float2 q_370 = _f2_255;
                    float2 r_371 = fma_f32x2_rn_noftz(q_370, scale0_346, a_369);
                    acc_195[14] = r_371.x;
                    acc_195[15] = r_371.y;
                    float _unpack_fp4_30[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[30 + 0]));
                        (&_unpack_fp4_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_30[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_30[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_30[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_30[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_30[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_30[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_30[0]), "=f"(_unpack_fp4_30[1]), "=f"(_unpack_fp4_30[2]), "=f"(_unpack_fp4_30[3]), "=f"(_unpack_fp4_30[4]), "=f"(_unpack_fp4_30[5]), "=f"(_unpack_fp4_30[6]), "=f"(_unpack_fp4_30[7]) : "r"(_fp4_wire_values_0[30 + 0]));
                        #endif
                    }
                    float2 _f2_256 = make_float2(acc_195[16], acc_195[17]);
                    float2 a_372 = _f2_256;
                    float2 _f2_257 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                    float2 q_373 = _f2_257;
                    float2 r_374 = fma_f32x2_rn_noftz(q_373, scale1_347, a_372);
                    acc_195[16] = r_374.x;
                    acc_195[17] = r_374.y;
                    float2 _f2_258 = make_float2(acc_195[18], acc_195[19]);
                    float2 a_375 = _f2_258;
                    float2 _f2_259 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                    float2 q_376 = _f2_259;
                    float2 r_377 = fma_f32x2_rn_noftz(q_376, scale1_347, a_375);
                    acc_195[18] = r_377.x;
                    acc_195[19] = r_377.y;
                    float2 _f2_260 = make_float2(acc_195[20], acc_195[21]);
                    float2 a_378 = _f2_260;
                    float2 _f2_261 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                    float2 q_379 = _f2_261;
                    float2 r_380 = fma_f32x2_rn_noftz(q_379, scale1_347, a_378);
                    acc_195[20] = r_380.x;
                    acc_195[21] = r_380.y;
                    float2 _f2_262 = make_float2(acc_195[22], acc_195[23]);
                    float2 a_381 = _f2_262;
                    float2 _f2_263 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                    float2 q_382 = _f2_263;
                    float2 r_383 = fma_f32x2_rn_noftz(q_382, scale1_347, a_381);
                    acc_195[22] = r_383.x;
                    acc_195[23] = r_383.y;
                    float _unpack_fp4_31[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[31 + 0]));
                        (&_unpack_fp4_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_31[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_31[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_31[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_31[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_31[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_31[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_31[0]), "=f"(_unpack_fp4_31[1]), "=f"(_unpack_fp4_31[2]), "=f"(_unpack_fp4_31[3]), "=f"(_unpack_fp4_31[4]), "=f"(_unpack_fp4_31[5]), "=f"(_unpack_fp4_31[6]), "=f"(_unpack_fp4_31[7]) : "r"(_fp4_wire_values_0[31 + 0]));
                        #endif
                    }
                    float2 _f2_264 = make_float2(acc_195[24], acc_195[25]);
                    float2 a_384 = _f2_264;
                    float2 _f2_265 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                    float2 q_385 = _f2_265;
                    float2 r_386 = fma_f32x2_rn_noftz(q_385, scale1_347, a_384);
                    acc_195[24] = r_386.x;
                    acc_195[25] = r_386.y;
                    float2 _f2_266 = make_float2(acc_195[26], acc_195[27]);
                    float2 a_387 = _f2_266;
                    float2 _f2_267 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                    float2 q_388 = _f2_267;
                    float2 r_389 = fma_f32x2_rn_noftz(q_388, scale1_347, a_387);
                    acc_195[26] = r_389.x;
                    acc_195[27] = r_389.y;
                    float2 _f2_268 = make_float2(acc_195[28], acc_195[29]);
                    float2 a_390 = _f2_268;
                    float2 _f2_269 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                    float2 q_391 = _f2_269;
                    float2 r_392 = fma_f32x2_rn_noftz(q_391, scale1_347, a_390);
                    acc_195[28] = r_392.x;
                    acc_195[29] = r_392.y;
                    float2 _f2_270 = make_float2(acc_195[30], acc_195[31]);
                    float2 a_393 = _f2_270;
                    float2 _f2_271 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                    float2 q_394 = _f2_271;
                    float2 r_395 = fma_f32x2_rn_noftz(q_394, scale1_347, a_393);
                    acc_195[30] = r_395.x;
                    acc_195[31] = r_395.y;
                    float cur_396 = am;
                    float _fabs_32;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_32) : "f"(acc_195[0]));
                    float _max_32 = max_noftz(cur_396, _fabs_32);
                    cur_396 = _max_32;
                    float _fabs_33;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_33) : "f"(acc_195[1]));
                    float _max_33 = max_noftz(cur_396, _fabs_33);
                    cur_396 = _max_33;
                    float _fabs_34;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_34) : "f"(acc_195[2]));
                    float _max_34 = max_noftz(cur_396, _fabs_34);
                    cur_396 = _max_34;
                    float _fabs_35;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_35) : "f"(acc_195[3]));
                    float _max_35 = max_noftz(cur_396, _fabs_35);
                    cur_396 = _max_35;
                    float _fabs_36;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_36) : "f"(acc_195[4]));
                    float _max_36 = max_noftz(cur_396, _fabs_36);
                    cur_396 = _max_36;
                    float _fabs_37;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_37) : "f"(acc_195[5]));
                    float _max_37 = max_noftz(cur_396, _fabs_37);
                    cur_396 = _max_37;
                    float _fabs_38;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_38) : "f"(acc_195[6]));
                    float _max_38 = max_noftz(cur_396, _fabs_38);
                    cur_396 = _max_38;
                    float _fabs_39;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_39) : "f"(acc_195[7]));
                    float _max_39 = max_noftz(cur_396, _fabs_39);
                    cur_396 = _max_39;
                    float _fabs_40;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_40) : "f"(acc_195[8]));
                    float _max_40 = max_noftz(cur_396, _fabs_40);
                    cur_396 = _max_40;
                    float _fabs_41;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_41) : "f"(acc_195[9]));
                    float _max_41 = max_noftz(cur_396, _fabs_41);
                    cur_396 = _max_41;
                    float _fabs_42;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_42) : "f"(acc_195[10]));
                    float _max_42 = max_noftz(cur_396, _fabs_42);
                    cur_396 = _max_42;
                    float _fabs_43;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_43) : "f"(acc_195[11]));
                    float _max_43 = max_noftz(cur_396, _fabs_43);
                    cur_396 = _max_43;
                    float _fabs_44;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_44) : "f"(acc_195[12]));
                    float _max_44 = max_noftz(cur_396, _fabs_44);
                    cur_396 = _max_44;
                    float _fabs_45;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_45) : "f"(acc_195[13]));
                    float _max_45 = max_noftz(cur_396, _fabs_45);
                    cur_396 = _max_45;
                    float _fabs_46;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_46) : "f"(acc_195[14]));
                    float _max_46 = max_noftz(cur_396, _fabs_46);
                    cur_396 = _max_46;
                    float _fabs_47;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_47) : "f"(acc_195[15]));
                    float _max_47 = max_noftz(cur_396, _fabs_47);
                    cur_396 = _max_47;
                    float _fabs_48;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_48) : "f"(acc_195[16]));
                    float _max_48 = max_noftz(cur_396, _fabs_48);
                    cur_396 = _max_48;
                    float _fabs_49;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_49) : "f"(acc_195[17]));
                    float _max_49 = max_noftz(cur_396, _fabs_49);
                    cur_396 = _max_49;
                    float _fabs_50;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_50) : "f"(acc_195[18]));
                    float _max_50 = max_noftz(cur_396, _fabs_50);
                    cur_396 = _max_50;
                    float _fabs_51;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_51) : "f"(acc_195[19]));
                    float _max_51 = max_noftz(cur_396, _fabs_51);
                    cur_396 = _max_51;
                    float _fabs_52;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_52) : "f"(acc_195[20]));
                    float _max_52 = max_noftz(cur_396, _fabs_52);
                    cur_396 = _max_52;
                    float _fabs_53;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_53) : "f"(acc_195[21]));
                    float _max_53 = max_noftz(cur_396, _fabs_53);
                    cur_396 = _max_53;
                    float _fabs_54;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_54) : "f"(acc_195[22]));
                    float _max_54 = max_noftz(cur_396, _fabs_54);
                    cur_396 = _max_54;
                    float _fabs_55;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_55) : "f"(acc_195[23]));
                    float _max_55 = max_noftz(cur_396, _fabs_55);
                    cur_396 = _max_55;
                    float _fabs_56;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_56) : "f"(acc_195[24]));
                    float _max_56 = max_noftz(cur_396, _fabs_56);
                    cur_396 = _max_56;
                    float _fabs_57;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_57) : "f"(acc_195[25]));
                    float _max_57 = max_noftz(cur_396, _fabs_57);
                    cur_396 = _max_57;
                    float _fabs_58;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_58) : "f"(acc_195[26]));
                    float _max_58 = max_noftz(cur_396, _fabs_58);
                    cur_396 = _max_58;
                    float _fabs_59;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_59) : "f"(acc_195[27]));
                    float _max_59 = max_noftz(cur_396, _fabs_59);
                    cur_396 = _max_59;
                    float _fabs_60;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_60) : "f"(acc_195[28]));
                    float _max_60 = max_noftz(cur_396, _fabs_60);
                    cur_396 = _max_60;
                    float _fabs_61;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_61) : "f"(acc_195[29]));
                    float _max_61 = max_noftz(cur_396, _fabs_61);
                    cur_396 = _max_61;
                    float _fabs_62;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_62) : "f"(acc_195[30]));
                    float _max_62 = max_noftz(cur_396, _fabs_62);
                    cur_396 = _max_62;
                    float _fabs_63;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_63) : "f"(acc_195[31]));
                    float _max_63 = max_noftz(cur_396, _fabs_63);
                    cur_396 = _max_63;
                    am = cur_396;
                }
                float rs2 = 1.0f;
                float _warp_reduce_0 = am;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                am = _warp_reduce_0;
                float _fdiv_rn_0 = __fdiv_rn(2688.0f, am);
                float inv = 0.0f;
                if (am > 0.0f) {
                    inv = _fdiv_rn_0;
                }
                float _fdiv_rn_1 = __fdiv_rn(am, 2688.0f);
                float rs = _fdiv_rn_1;
                rs2 = rs;
                #pragma unroll 1
                for (int j = 0; j < 6; j++) {
                    int pp = lane_1 + 32 * j;
                    unsigned int _vec_load_0[4];
                    {
                        uint32_t _uv4_3_0;
                        uint32_t _uv4_3_1;
                        uint32_t _uv4_3_2;
                        uint32_t _uv4_3_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_3_0), "=r"(_uv4_3_1), "=r"(_uv4_3_2), "=r"(_uv4_3_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp))) + (0))) : "memory");
                        _vec_load_0[0 + 0] = _uv4_3_0;
                        _vec_load_0[0 + 1] = _uv4_3_1;
                        _vec_load_0[0 + 2] = _uv4_3_2;
                        _vec_load_0[0 + 3] = _uv4_3_3;
                    }
                    unsigned int _vec_load_1[1];
                    {
                        uint32_t _scalar_bits_4;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_4) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc + (int)(4 * (pp >> 1)))) + (0))) : "memory");
                        _vec_load_1[0] = (unsigned int)_scalar_bits_4;
                    }
                    unsigned int c = _vec_load_1[0] >> (unsigned int)(16 * (pp & 1)) & 65535;
                    unsigned int _vec_load_2[4];
                    {
                        uint32_t _uv4_5_0;
                        uint32_t _uv4_5_1;
                        uint32_t _uv4_5_2;
                        uint32_t _uv4_5_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_5_0), "=r"(_uv4_5_1), "=r"(_uv4_5_2), "=r"(_uv4_5_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp))) + (0))) : "memory");
                        _vec_load_2[0 + 0] = _uv4_5_0;
                        _vec_load_2[0 + 1] = _uv4_5_1;
                        _vec_load_2[0 + 2] = _uv4_5_2;
                        _vec_load_2[0 + 3] = _uv4_5_3;
                    }
                    unsigned int _vec_load_3[1];
                    {
                        uint32_t _scalar_bits_6;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_6) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_1 + (int)(4 * (pp >> 1)))) + (0))) : "memory");
                        _vec_load_3[0] = (unsigned int)_scalar_bits_6;
                    }
                    unsigned int c_0 = _vec_load_3[0] >> (unsigned int)(16 * (pp & 1)) & 65535;
                    unsigned int _vec_load_4[4];
                    {
                        uint32_t _uv4_7_0;
                        uint32_t _uv4_7_1;
                        uint32_t _uv4_7_2;
                        uint32_t _uv4_7_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_7_0), "=r"(_uv4_7_1), "=r"(_uv4_7_2), "=r"(_uv4_7_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp))) + (0))) : "memory");
                        _vec_load_4[0 + 0] = _uv4_7_0;
                        _vec_load_4[0 + 1] = _uv4_7_1;
                        _vec_load_4[0 + 2] = _uv4_7_2;
                        _vec_load_4[0 + 3] = _uv4_7_3;
                    }
                    unsigned int _vec_load_5[1];
                    {
                        uint32_t _scalar_bits_8;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_8) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_3 + (int)(4 * (pp >> 1)))) + (0))) : "memory");
                        _vec_load_5[0] = (unsigned int)_scalar_bits_8;
                    }
                    unsigned int c_1 = _vec_load_5[0] >> (unsigned int)(16 * (pp & 1)) & 65535;
                    unsigned int _vec_load_6[4];
                    {
                        uint32_t _uv4_9_0;
                        uint32_t _uv4_9_1;
                        uint32_t _uv4_9_2;
                        uint32_t _uv4_9_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_uv4_9_0), "=r"(_uv4_9_1), "=r"(_uv4_9_2), "=r"(_uv4_9_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp))) + (0))) : "memory");
                        _vec_load_6[0 + 0] = _uv4_9_0;
                        _vec_load_6[0 + 1] = _uv4_9_1;
                        _vec_load_6[0 + 2] = _uv4_9_2;
                        _vec_load_6[0 + 3] = _uv4_9_3;
                    }
                    unsigned int _vec_load_7[1];
                    {
                        uint32_t _scalar_bits_10;
                        asm volatile("ld.global.cg.b32 %0, [%1];"
                            : "=r"(_scalar_bits_10) : "l"((const void*)(reinterpret_cast<unsigned int*>(ts_wire + (sc_5 + (int)(4 * (pp >> 1)))) + (0))) : "memory");
                        _vec_load_7[0] = (unsigned int)_scalar_bits_10;
                    }
                    unsigned int c_2 = _vec_load_7[0] >> (unsigned int)(16 * (pp & 1)) & 65535;
                    float acc2[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)(c)));
                    float _cvt_f32_f16_16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(_e4m3x2_to_f16x2_8)));
                    float _cvt_f32_f16_17;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(_e4m3x2_to_f16x2_8 >> 16)));
                    float _mul_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_cvt_f32_f16_16), "f"(g_s));
                    float _mul_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_cvt_f32_f16_17), "f"(g_s));
                    float2 _f2_272 = make_float2(_mul_16, _mul_16);
                    float2 scale0_1 = _f2_272;
                    float2 _f2_273 = make_float2(_mul_17, _mul_17);
                    float2 scale1_1 = _f2_273;
                    float _unpack_fp4_32[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_0[0]));
                        (&_unpack_fp4_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_32[0]), "=f"(_unpack_fp4_32[1]), "=f"(_unpack_fp4_32[2]), "=f"(_unpack_fp4_32[3]), "=f"(_unpack_fp4_32[4]), "=f"(_unpack_fp4_32[5]), "=f"(_unpack_fp4_32[6]), "=f"(_unpack_fp4_32[7]) : "r"(_vec_load_0[0]));
                        #endif
                    }
                    float2 _f2_274 = make_float2(acc2[0], acc2[1]);
                    float2 a_1 = _f2_274;
                    float2 _f2_275 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                    float2 q_2 = _f2_275;
                    float2 r_1 = fma_f32x2_rn_noftz(q_2, scale0_1, a_1);
                    acc2[0] = r_1.x;
                    acc2[1] = r_1.y;
                    float2 _f2_276 = make_float2(acc2[2], acc2[3]);
                    float2 a_3_1 = _f2_276;
                    float2 _f2_277 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                    float2 q_4_1 = _f2_277;
                    float2 r_5_1 = fma_f32x2_rn_noftz(q_4_1, scale0_1, a_3_1);
                    acc2[2] = r_5_1.x;
                    acc2[3] = r_5_1.y;
                    float2 _f2_278 = make_float2(acc2[4], acc2[5]);
                    float2 a_6_1 = _f2_278;
                    float2 _f2_279 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                    float2 q_7_1 = _f2_279;
                    float2 r_8_1 = fma_f32x2_rn_noftz(q_7_1, scale0_1, a_6_1);
                    acc2[4] = r_8_1.x;
                    acc2[5] = r_8_1.y;
                    float2 _f2_280 = make_float2(acc2[6], acc2[7]);
                    float2 a_9_1 = _f2_280;
                    float2 _f2_281 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                    float2 q_10_1 = _f2_281;
                    float2 r_11_1 = fma_f32x2_rn_noftz(q_10_1, scale0_1, a_9_1);
                    acc2[6] = r_11_1.x;
                    acc2[7] = r_11_1.y;
                    float _unpack_fp4_33[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_0[1 + 0]));
                        (&_unpack_fp4_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_33[0]), "=f"(_unpack_fp4_33[1]), "=f"(_unpack_fp4_33[2]), "=f"(_unpack_fp4_33[3]), "=f"(_unpack_fp4_33[4]), "=f"(_unpack_fp4_33[5]), "=f"(_unpack_fp4_33[6]), "=f"(_unpack_fp4_33[7]) : "r"(_vec_load_0[1 + 0]));
                        #endif
                    }
                    float2 _f2_282 = make_float2(acc2[8], acc2[9]);
                    float2 a_12_1 = _f2_282;
                    float2 _f2_283 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                    float2 q_13_1 = _f2_283;
                    float2 r_14_1 = fma_f32x2_rn_noftz(q_13_1, scale0_1, a_12_1);
                    acc2[8] = r_14_1.x;
                    acc2[9] = r_14_1.y;
                    float2 _f2_284 = make_float2(acc2[10], acc2[11]);
                    float2 a_15_1 = _f2_284;
                    float2 _f2_285 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                    float2 q_16_1 = _f2_285;
                    float2 r_17_1 = fma_f32x2_rn_noftz(q_16_1, scale0_1, a_15_1);
                    acc2[10] = r_17_1.x;
                    acc2[11] = r_17_1.y;
                    float2 _f2_286 = make_float2(acc2[12], acc2[13]);
                    float2 a_18_1 = _f2_286;
                    float2 _f2_287 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                    float2 q_19_1 = _f2_287;
                    float2 r_20_1 = fma_f32x2_rn_noftz(q_19_1, scale0_1, a_18_1);
                    acc2[12] = r_20_1.x;
                    acc2[13] = r_20_1.y;
                    float2 _f2_288 = make_float2(acc2[14], acc2[15]);
                    float2 a_21_1 = _f2_288;
                    float2 _f2_289 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                    float2 q_22_1 = _f2_289;
                    float2 r_23_1 = fma_f32x2_rn_noftz(q_22_1, scale0_1, a_21_1);
                    acc2[14] = r_23_1.x;
                    acc2[15] = r_23_1.y;
                    float _unpack_fp4_34[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_0[2 + 0]));
                        (&_unpack_fp4_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_34[0]), "=f"(_unpack_fp4_34[1]), "=f"(_unpack_fp4_34[2]), "=f"(_unpack_fp4_34[3]), "=f"(_unpack_fp4_34[4]), "=f"(_unpack_fp4_34[5]), "=f"(_unpack_fp4_34[6]), "=f"(_unpack_fp4_34[7]) : "r"(_vec_load_0[2 + 0]));
                        #endif
                    }
                    float2 _f2_290 = make_float2(acc2[16], acc2[17]);
                    float2 a_24_1 = _f2_290;
                    float2 _f2_291 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                    float2 q_25_1 = _f2_291;
                    float2 r_26_1 = fma_f32x2_rn_noftz(q_25_1, scale1_1, a_24_1);
                    acc2[16] = r_26_1.x;
                    acc2[17] = r_26_1.y;
                    float2 _f2_292 = make_float2(acc2[18], acc2[19]);
                    float2 a_27_1 = _f2_292;
                    float2 _f2_293 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                    float2 q_28_1 = _f2_293;
                    float2 r_29_1 = fma_f32x2_rn_noftz(q_28_1, scale1_1, a_27_1);
                    acc2[18] = r_29_1.x;
                    acc2[19] = r_29_1.y;
                    float2 _f2_294 = make_float2(acc2[20], acc2[21]);
                    float2 a_30_1 = _f2_294;
                    float2 _f2_295 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                    float2 q_31_1 = _f2_295;
                    float2 r_32_1 = fma_f32x2_rn_noftz(q_31_1, scale1_1, a_30_1);
                    acc2[20] = r_32_1.x;
                    acc2[21] = r_32_1.y;
                    float2 _f2_296 = make_float2(acc2[22], acc2[23]);
                    float2 a_33_1 = _f2_296;
                    float2 _f2_297 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                    float2 q_34_1 = _f2_297;
                    float2 r_35_1 = fma_f32x2_rn_noftz(q_34_1, scale1_1, a_33_1);
                    acc2[22] = r_35_1.x;
                    acc2[23] = r_35_1.y;
                    float _unpack_fp4_35[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_0[3 + 0]));
                        (&_unpack_fp4_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_35[0]), "=f"(_unpack_fp4_35[1]), "=f"(_unpack_fp4_35[2]), "=f"(_unpack_fp4_35[3]), "=f"(_unpack_fp4_35[4]), "=f"(_unpack_fp4_35[5]), "=f"(_unpack_fp4_35[6]), "=f"(_unpack_fp4_35[7]) : "r"(_vec_load_0[3 + 0]));
                        #endif
                    }
                    float2 _f2_298 = make_float2(acc2[24], acc2[25]);
                    float2 a_36_1 = _f2_298;
                    float2 _f2_299 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                    float2 q_37_1 = _f2_299;
                    float2 r_38_1 = fma_f32x2_rn_noftz(q_37_1, scale1_1, a_36_1);
                    acc2[24] = r_38_1.x;
                    acc2[25] = r_38_1.y;
                    float2 _f2_300 = make_float2(acc2[26], acc2[27]);
                    float2 a_39_1 = _f2_300;
                    float2 _f2_301 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                    float2 q_40_1 = _f2_301;
                    float2 r_41_1 = fma_f32x2_rn_noftz(q_40_1, scale1_1, a_39_1);
                    acc2[26] = r_41_1.x;
                    acc2[27] = r_41_1.y;
                    float2 _f2_302 = make_float2(acc2[28], acc2[29]);
                    float2 a_42_1 = _f2_302;
                    float2 _f2_303 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                    float2 q_43_1 = _f2_303;
                    float2 r_44_1 = fma_f32x2_rn_noftz(q_43_1, scale1_1, a_42_1);
                    acc2[28] = r_44_1.x;
                    acc2[29] = r_44_1.y;
                    float2 _f2_304 = make_float2(acc2[30], acc2[31]);
                    float2 a_45 = _f2_304;
                    float2 _f2_305 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                    float2 q_46 = _f2_305;
                    float2 r_47 = fma_f32x2_rn_noftz(q_46, scale1_1, a_45);
                    acc2[30] = r_47.x;
                    acc2[31] = r_47.y;
                    uint32_t _e4m3x2_to_f16x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)(c_0)));
                    float _cvt_f32_f16_18;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(_e4m3x2_to_f16x2_9)));
                    float _cvt_f32_f16_19;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(_e4m3x2_to_f16x2_9 >> 16)));
                    float _mul_18;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_cvt_f32_f16_18), "f"(g_s_2));
                    float _mul_19;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_cvt_f32_f16_19), "f"(g_s_2));
                    float2 _f2_306 = make_float2(_mul_18, _mul_18);
                    float2 scale0_48 = _f2_306;
                    float2 _f2_307 = make_float2(_mul_19, _mul_19);
                    float2 scale1_49 = _f2_307;
                    float _unpack_fp4_36[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_2[0]));
                        (&_unpack_fp4_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_36[0]), "=f"(_unpack_fp4_36[1]), "=f"(_unpack_fp4_36[2]), "=f"(_unpack_fp4_36[3]), "=f"(_unpack_fp4_36[4]), "=f"(_unpack_fp4_36[5]), "=f"(_unpack_fp4_36[6]), "=f"(_unpack_fp4_36[7]) : "r"(_vec_load_2[0]));
                        #endif
                    }
                    float2 _f2_308 = make_float2(acc2[0], acc2[1]);
                    float2 a_50_1 = _f2_308;
                    float2 _f2_309 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                    float2 q_51_1 = _f2_309;
                    float2 r_52_1 = fma_f32x2_rn_noftz(q_51_1, scale0_48, a_50_1);
                    acc2[0] = r_52_1.x;
                    acc2[1] = r_52_1.y;
                    float2 _f2_310 = make_float2(acc2[2], acc2[3]);
                    float2 a_53_1 = _f2_310;
                    float2 _f2_311 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                    float2 q_54_1 = _f2_311;
                    float2 r_55_1 = fma_f32x2_rn_noftz(q_54_1, scale0_48, a_53_1);
                    acc2[2] = r_55_1.x;
                    acc2[3] = r_55_1.y;
                    float2 _f2_312 = make_float2(acc2[4], acc2[5]);
                    float2 a_56_1 = _f2_312;
                    float2 _f2_313 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                    float2 q_57_1 = _f2_313;
                    float2 r_58_1 = fma_f32x2_rn_noftz(q_57_1, scale0_48, a_56_1);
                    acc2[4] = r_58_1.x;
                    acc2[5] = r_58_1.y;
                    float2 _f2_314 = make_float2(acc2[6], acc2[7]);
                    float2 a_59_1 = _f2_314;
                    float2 _f2_315 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                    float2 q_60_1 = _f2_315;
                    float2 r_61_1 = fma_f32x2_rn_noftz(q_60_1, scale0_48, a_59_1);
                    acc2[6] = r_61_1.x;
                    acc2[7] = r_61_1.y;
                    float _unpack_fp4_37[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_2[1 + 0]));
                        (&_unpack_fp4_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_37[0]), "=f"(_unpack_fp4_37[1]), "=f"(_unpack_fp4_37[2]), "=f"(_unpack_fp4_37[3]), "=f"(_unpack_fp4_37[4]), "=f"(_unpack_fp4_37[5]), "=f"(_unpack_fp4_37[6]), "=f"(_unpack_fp4_37[7]) : "r"(_vec_load_2[1 + 0]));
                        #endif
                    }
                    float2 _f2_316 = make_float2(acc2[8], acc2[9]);
                    float2 a_62_1 = _f2_316;
                    float2 _f2_317 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                    float2 q_63_1 = _f2_317;
                    float2 r_64_1 = fma_f32x2_rn_noftz(q_63_1, scale0_48, a_62_1);
                    acc2[8] = r_64_1.x;
                    acc2[9] = r_64_1.y;
                    float2 _f2_318 = make_float2(acc2[10], acc2[11]);
                    float2 a_65_1 = _f2_318;
                    float2 _f2_319 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                    float2 q_66_1 = _f2_319;
                    float2 r_67_1 = fma_f32x2_rn_noftz(q_66_1, scale0_48, a_65_1);
                    acc2[10] = r_67_1.x;
                    acc2[11] = r_67_1.y;
                    float2 _f2_320 = make_float2(acc2[12], acc2[13]);
                    float2 a_68_1 = _f2_320;
                    float2 _f2_321 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                    float2 q_69_1 = _f2_321;
                    float2 r_70_1 = fma_f32x2_rn_noftz(q_69_1, scale0_48, a_68_1);
                    acc2[12] = r_70_1.x;
                    acc2[13] = r_70_1.y;
                    float2 _f2_322 = make_float2(acc2[14], acc2[15]);
                    float2 a_71_1 = _f2_322;
                    float2 _f2_323 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                    float2 q_72_1 = _f2_323;
                    float2 r_73_1 = fma_f32x2_rn_noftz(q_72_1, scale0_48, a_71_1);
                    acc2[14] = r_73_1.x;
                    acc2[15] = r_73_1.y;
                    float _unpack_fp4_38[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_2[2 + 0]));
                        (&_unpack_fp4_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_38[0]), "=f"(_unpack_fp4_38[1]), "=f"(_unpack_fp4_38[2]), "=f"(_unpack_fp4_38[3]), "=f"(_unpack_fp4_38[4]), "=f"(_unpack_fp4_38[5]), "=f"(_unpack_fp4_38[6]), "=f"(_unpack_fp4_38[7]) : "r"(_vec_load_2[2 + 0]));
                        #endif
                    }
                    float2 _f2_324 = make_float2(acc2[16], acc2[17]);
                    float2 a_74_1 = _f2_324;
                    float2 _f2_325 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                    float2 q_75_1 = _f2_325;
                    float2 r_76_1 = fma_f32x2_rn_noftz(q_75_1, scale1_49, a_74_1);
                    acc2[16] = r_76_1.x;
                    acc2[17] = r_76_1.y;
                    float2 _f2_326 = make_float2(acc2[18], acc2[19]);
                    float2 a_77_1 = _f2_326;
                    float2 _f2_327 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                    float2 q_78_1 = _f2_327;
                    float2 r_79_1 = fma_f32x2_rn_noftz(q_78_1, scale1_49, a_77_1);
                    acc2[18] = r_79_1.x;
                    acc2[19] = r_79_1.y;
                    float2 _f2_328 = make_float2(acc2[20], acc2[21]);
                    float2 a_80_1 = _f2_328;
                    float2 _f2_329 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                    float2 q_81_1 = _f2_329;
                    float2 r_82_1 = fma_f32x2_rn_noftz(q_81_1, scale1_49, a_80_1);
                    acc2[20] = r_82_1.x;
                    acc2[21] = r_82_1.y;
                    float2 _f2_330 = make_float2(acc2[22], acc2[23]);
                    float2 a_83_1 = _f2_330;
                    float2 _f2_331 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                    float2 q_84_1 = _f2_331;
                    float2 r_85_1 = fma_f32x2_rn_noftz(q_84_1, scale1_49, a_83_1);
                    acc2[22] = r_85_1.x;
                    acc2[23] = r_85_1.y;
                    float _unpack_fp4_39[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_2[3 + 0]));
                        (&_unpack_fp4_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_39[0]), "=f"(_unpack_fp4_39[1]), "=f"(_unpack_fp4_39[2]), "=f"(_unpack_fp4_39[3]), "=f"(_unpack_fp4_39[4]), "=f"(_unpack_fp4_39[5]), "=f"(_unpack_fp4_39[6]), "=f"(_unpack_fp4_39[7]) : "r"(_vec_load_2[3 + 0]));
                        #endif
                    }
                    float2 _f2_332 = make_float2(acc2[24], acc2[25]);
                    float2 a_86_1 = _f2_332;
                    float2 _f2_333 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                    float2 q_87_1 = _f2_333;
                    float2 r_88_1 = fma_f32x2_rn_noftz(q_87_1, scale1_49, a_86_1);
                    acc2[24] = r_88_1.x;
                    acc2[25] = r_88_1.y;
                    float2 _f2_334 = make_float2(acc2[26], acc2[27]);
                    float2 a_89_1 = _f2_334;
                    float2 _f2_335 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                    float2 q_90_1 = _f2_335;
                    float2 r_91_1 = fma_f32x2_rn_noftz(q_90_1, scale1_49, a_89_1);
                    acc2[26] = r_91_1.x;
                    acc2[27] = r_91_1.y;
                    float2 _f2_336 = make_float2(acc2[28], acc2[29]);
                    float2 a_92_1 = _f2_336;
                    float2 _f2_337 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                    float2 q_93_1 = _f2_337;
                    float2 r_94_1 = fma_f32x2_rn_noftz(q_93_1, scale1_49, a_92_1);
                    acc2[28] = r_94_1.x;
                    acc2[29] = r_94_1.y;
                    float2 _f2_338 = make_float2(acc2[30], acc2[31]);
                    float2 a_95 = _f2_338;
                    float2 _f2_339 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                    float2 q_96 = _f2_339;
                    float2 r_97 = fma_f32x2_rn_noftz(q_96, scale1_49, a_95);
                    acc2[30] = r_97.x;
                    acc2[31] = r_97.y;
                    uint32_t _e4m3x2_to_f16x2_10;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)(c_1)));
                    float _cvt_f32_f16_20;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(_e4m3x2_to_f16x2_10)));
                    float _cvt_f32_f16_21;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(_e4m3x2_to_f16x2_10 >> 16)));
                    float _mul_20;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_cvt_f32_f16_20), "f"(g_s_3));
                    float _mul_21;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_cvt_f32_f16_21), "f"(g_s_3));
                    float2 _f2_340 = make_float2(_mul_20, _mul_20);
                    float2 scale0_98 = _f2_340;
                    float2 _f2_341 = make_float2(_mul_21, _mul_21);
                    float2 scale1_99 = _f2_341;
                    float _unpack_fp4_40[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_4[0]));
                        (&_unpack_fp4_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_40[0]), "=f"(_unpack_fp4_40[1]), "=f"(_unpack_fp4_40[2]), "=f"(_unpack_fp4_40[3]), "=f"(_unpack_fp4_40[4]), "=f"(_unpack_fp4_40[5]), "=f"(_unpack_fp4_40[6]), "=f"(_unpack_fp4_40[7]) : "r"(_vec_load_4[0]));
                        #endif
                    }
                    float2 _f2_342 = make_float2(acc2[0], acc2[1]);
                    float2 a_100_1 = _f2_342;
                    float2 _f2_343 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                    float2 q_101_1 = _f2_343;
                    float2 r_102_1 = fma_f32x2_rn_noftz(q_101_1, scale0_98, a_100_1);
                    acc2[0] = r_102_1.x;
                    acc2[1] = r_102_1.y;
                    float2 _f2_344 = make_float2(acc2[2], acc2[3]);
                    float2 a_103_1 = _f2_344;
                    float2 _f2_345 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                    float2 q_104_1 = _f2_345;
                    float2 r_105_1 = fma_f32x2_rn_noftz(q_104_1, scale0_98, a_103_1);
                    acc2[2] = r_105_1.x;
                    acc2[3] = r_105_1.y;
                    float2 _f2_346 = make_float2(acc2[4], acc2[5]);
                    float2 a_106_1 = _f2_346;
                    float2 _f2_347 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                    float2 q_107_1 = _f2_347;
                    float2 r_108_1 = fma_f32x2_rn_noftz(q_107_1, scale0_98, a_106_1);
                    acc2[4] = r_108_1.x;
                    acc2[5] = r_108_1.y;
                    float2 _f2_348 = make_float2(acc2[6], acc2[7]);
                    float2 a_109_1 = _f2_348;
                    float2 _f2_349 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                    float2 q_110_1 = _f2_349;
                    float2 r_111_1 = fma_f32x2_rn_noftz(q_110_1, scale0_98, a_109_1);
                    acc2[6] = r_111_1.x;
                    acc2[7] = r_111_1.y;
                    float _unpack_fp4_41[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_4[1 + 0]));
                        (&_unpack_fp4_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_41[0]), "=f"(_unpack_fp4_41[1]), "=f"(_unpack_fp4_41[2]), "=f"(_unpack_fp4_41[3]), "=f"(_unpack_fp4_41[4]), "=f"(_unpack_fp4_41[5]), "=f"(_unpack_fp4_41[6]), "=f"(_unpack_fp4_41[7]) : "r"(_vec_load_4[1 + 0]));
                        #endif
                    }
                    float2 _f2_350 = make_float2(acc2[8], acc2[9]);
                    float2 a_112_1 = _f2_350;
                    float2 _f2_351 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                    float2 q_113_1 = _f2_351;
                    float2 r_114_1 = fma_f32x2_rn_noftz(q_113_1, scale0_98, a_112_1);
                    acc2[8] = r_114_1.x;
                    acc2[9] = r_114_1.y;
                    float2 _f2_352 = make_float2(acc2[10], acc2[11]);
                    float2 a_115_1 = _f2_352;
                    float2 _f2_353 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                    float2 q_116_1 = _f2_353;
                    float2 r_117_1 = fma_f32x2_rn_noftz(q_116_1, scale0_98, a_115_1);
                    acc2[10] = r_117_1.x;
                    acc2[11] = r_117_1.y;
                    float2 _f2_354 = make_float2(acc2[12], acc2[13]);
                    float2 a_118_1 = _f2_354;
                    float2 _f2_355 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                    float2 q_119_1 = _f2_355;
                    float2 r_120_1 = fma_f32x2_rn_noftz(q_119_1, scale0_98, a_118_1);
                    acc2[12] = r_120_1.x;
                    acc2[13] = r_120_1.y;
                    float2 _f2_356 = make_float2(acc2[14], acc2[15]);
                    float2 a_121_1 = _f2_356;
                    float2 _f2_357 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                    float2 q_122_1 = _f2_357;
                    float2 r_123_1 = fma_f32x2_rn_noftz(q_122_1, scale0_98, a_121_1);
                    acc2[14] = r_123_1.x;
                    acc2[15] = r_123_1.y;
                    float _unpack_fp4_42[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_4[2 + 0]));
                        (&_unpack_fp4_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_42[0]), "=f"(_unpack_fp4_42[1]), "=f"(_unpack_fp4_42[2]), "=f"(_unpack_fp4_42[3]), "=f"(_unpack_fp4_42[4]), "=f"(_unpack_fp4_42[5]), "=f"(_unpack_fp4_42[6]), "=f"(_unpack_fp4_42[7]) : "r"(_vec_load_4[2 + 0]));
                        #endif
                    }
                    float2 _f2_358 = make_float2(acc2[16], acc2[17]);
                    float2 a_124_1 = _f2_358;
                    float2 _f2_359 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                    float2 q_125_1 = _f2_359;
                    float2 r_126_1 = fma_f32x2_rn_noftz(q_125_1, scale1_99, a_124_1);
                    acc2[16] = r_126_1.x;
                    acc2[17] = r_126_1.y;
                    float2 _f2_360 = make_float2(acc2[18], acc2[19]);
                    float2 a_127_1 = _f2_360;
                    float2 _f2_361 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                    float2 q_128_1 = _f2_361;
                    float2 r_129_1 = fma_f32x2_rn_noftz(q_128_1, scale1_99, a_127_1);
                    acc2[18] = r_129_1.x;
                    acc2[19] = r_129_1.y;
                    float2 _f2_362 = make_float2(acc2[20], acc2[21]);
                    float2 a_130_1 = _f2_362;
                    float2 _f2_363 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                    float2 q_131_1 = _f2_363;
                    float2 r_132_1 = fma_f32x2_rn_noftz(q_131_1, scale1_99, a_130_1);
                    acc2[20] = r_132_1.x;
                    acc2[21] = r_132_1.y;
                    float2 _f2_364 = make_float2(acc2[22], acc2[23]);
                    float2 a_133_1 = _f2_364;
                    float2 _f2_365 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                    float2 q_134_1 = _f2_365;
                    float2 r_135_1 = fma_f32x2_rn_noftz(q_134_1, scale1_99, a_133_1);
                    acc2[22] = r_135_1.x;
                    acc2[23] = r_135_1.y;
                    float _unpack_fp4_43[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_4[3 + 0]));
                        (&_unpack_fp4_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_43[0]), "=f"(_unpack_fp4_43[1]), "=f"(_unpack_fp4_43[2]), "=f"(_unpack_fp4_43[3]), "=f"(_unpack_fp4_43[4]), "=f"(_unpack_fp4_43[5]), "=f"(_unpack_fp4_43[6]), "=f"(_unpack_fp4_43[7]) : "r"(_vec_load_4[3 + 0]));
                        #endif
                    }
                    float2 _f2_366 = make_float2(acc2[24], acc2[25]);
                    float2 a_136_1 = _f2_366;
                    float2 _f2_367 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                    float2 q_137_1 = _f2_367;
                    float2 r_138_1 = fma_f32x2_rn_noftz(q_137_1, scale1_99, a_136_1);
                    acc2[24] = r_138_1.x;
                    acc2[25] = r_138_1.y;
                    float2 _f2_368 = make_float2(acc2[26], acc2[27]);
                    float2 a_139_1 = _f2_368;
                    float2 _f2_369 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                    float2 q_140_1 = _f2_369;
                    float2 r_141_1 = fma_f32x2_rn_noftz(q_140_1, scale1_99, a_139_1);
                    acc2[26] = r_141_1.x;
                    acc2[27] = r_141_1.y;
                    float2 _f2_370 = make_float2(acc2[28], acc2[29]);
                    float2 a_142_1 = _f2_370;
                    float2 _f2_371 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                    float2 q_143_1 = _f2_371;
                    float2 r_144_1 = fma_f32x2_rn_noftz(q_143_1, scale1_99, a_142_1);
                    acc2[28] = r_144_1.x;
                    acc2[29] = r_144_1.y;
                    float2 _f2_372 = make_float2(acc2[30], acc2[31]);
                    float2 a_145 = _f2_372;
                    float2 _f2_373 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                    float2 q_146 = _f2_373;
                    float2 r_147 = fma_f32x2_rn_noftz(q_146, scale1_99, a_145);
                    acc2[30] = r_147.x;
                    acc2[31] = r_147.y;
                    uint32_t _e4m3x2_to_f16x2_11;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)(c_2)));
                    float _cvt_f32_f16_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(_e4m3x2_to_f16x2_11)));
                    float _cvt_f32_f16_23;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(_e4m3x2_to_f16x2_11 >> 16)));
                    float _mul_22;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_cvt_f32_f16_22), "f"(g_s_4));
                    float _mul_23;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_cvt_f32_f16_23), "f"(g_s_4));
                    float2 _f2_374 = make_float2(_mul_22, _mul_22);
                    float2 scale0_148 = _f2_374;
                    float2 _f2_375 = make_float2(_mul_23, _mul_23);
                    float2 scale1_149 = _f2_375;
                    float _unpack_fp4_44[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_6[0]));
                        (&_unpack_fp4_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_44[0]), "=f"(_unpack_fp4_44[1]), "=f"(_unpack_fp4_44[2]), "=f"(_unpack_fp4_44[3]), "=f"(_unpack_fp4_44[4]), "=f"(_unpack_fp4_44[5]), "=f"(_unpack_fp4_44[6]), "=f"(_unpack_fp4_44[7]) : "r"(_vec_load_6[0]));
                        #endif
                    }
                    float2 _f2_376 = make_float2(acc2[0], acc2[1]);
                    float2 a_150_1 = _f2_376;
                    float2 _f2_377 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                    float2 q_151_1 = _f2_377;
                    float2 r_152_1 = fma_f32x2_rn_noftz(q_151_1, scale0_148, a_150_1);
                    acc2[0] = r_152_1.x;
                    acc2[1] = r_152_1.y;
                    float2 _f2_378 = make_float2(acc2[2], acc2[3]);
                    float2 a_153_1 = _f2_378;
                    float2 _f2_379 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                    float2 q_154_1 = _f2_379;
                    float2 r_155_1 = fma_f32x2_rn_noftz(q_154_1, scale0_148, a_153_1);
                    acc2[2] = r_155_1.x;
                    acc2[3] = r_155_1.y;
                    float2 _f2_380 = make_float2(acc2[4], acc2[5]);
                    float2 a_156_1 = _f2_380;
                    float2 _f2_381 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                    float2 q_157_1 = _f2_381;
                    float2 r_158_1 = fma_f32x2_rn_noftz(q_157_1, scale0_148, a_156_1);
                    acc2[4] = r_158_1.x;
                    acc2[5] = r_158_1.y;
                    float2 _f2_382 = make_float2(acc2[6], acc2[7]);
                    float2 a_159_1 = _f2_382;
                    float2 _f2_383 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                    float2 q_160_1 = _f2_383;
                    float2 r_161_1 = fma_f32x2_rn_noftz(q_160_1, scale0_148, a_159_1);
                    acc2[6] = r_161_1.x;
                    acc2[7] = r_161_1.y;
                    float _unpack_fp4_45[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_6[1 + 0]));
                        (&_unpack_fp4_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_45[0]), "=f"(_unpack_fp4_45[1]), "=f"(_unpack_fp4_45[2]), "=f"(_unpack_fp4_45[3]), "=f"(_unpack_fp4_45[4]), "=f"(_unpack_fp4_45[5]), "=f"(_unpack_fp4_45[6]), "=f"(_unpack_fp4_45[7]) : "r"(_vec_load_6[1 + 0]));
                        #endif
                    }
                    float2 _f2_384 = make_float2(acc2[8], acc2[9]);
                    float2 a_162_1 = _f2_384;
                    float2 _f2_385 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                    float2 q_163_1 = _f2_385;
                    float2 r_164_1 = fma_f32x2_rn_noftz(q_163_1, scale0_148, a_162_1);
                    acc2[8] = r_164_1.x;
                    acc2[9] = r_164_1.y;
                    float2 _f2_386 = make_float2(acc2[10], acc2[11]);
                    float2 a_165_1 = _f2_386;
                    float2 _f2_387 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                    float2 q_166_1 = _f2_387;
                    float2 r_167_1 = fma_f32x2_rn_noftz(q_166_1, scale0_148, a_165_1);
                    acc2[10] = r_167_1.x;
                    acc2[11] = r_167_1.y;
                    float2 _f2_388 = make_float2(acc2[12], acc2[13]);
                    float2 a_168_1 = _f2_388;
                    float2 _f2_389 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                    float2 q_169_1 = _f2_389;
                    float2 r_170_1 = fma_f32x2_rn_noftz(q_169_1, scale0_148, a_168_1);
                    acc2[12] = r_170_1.x;
                    acc2[13] = r_170_1.y;
                    float2 _f2_390 = make_float2(acc2[14], acc2[15]);
                    float2 a_171_1 = _f2_390;
                    float2 _f2_391 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                    float2 q_172_1 = _f2_391;
                    float2 r_173_1 = fma_f32x2_rn_noftz(q_172_1, scale0_148, a_171_1);
                    acc2[14] = r_173_1.x;
                    acc2[15] = r_173_1.y;
                    float _unpack_fp4_46[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_6[2 + 0]));
                        (&_unpack_fp4_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_46[0]), "=f"(_unpack_fp4_46[1]), "=f"(_unpack_fp4_46[2]), "=f"(_unpack_fp4_46[3]), "=f"(_unpack_fp4_46[4]), "=f"(_unpack_fp4_46[5]), "=f"(_unpack_fp4_46[6]), "=f"(_unpack_fp4_46[7]) : "r"(_vec_load_6[2 + 0]));
                        #endif
                    }
                    float2 _f2_392 = make_float2(acc2[16], acc2[17]);
                    float2 a_174_1 = _f2_392;
                    float2 _f2_393 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                    float2 q_175_1 = _f2_393;
                    float2 r_176_1 = fma_f32x2_rn_noftz(q_175_1, scale1_149, a_174_1);
                    acc2[16] = r_176_1.x;
                    acc2[17] = r_176_1.y;
                    float2 _f2_394 = make_float2(acc2[18], acc2[19]);
                    float2 a_177_1 = _f2_394;
                    float2 _f2_395 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                    float2 q_178_1 = _f2_395;
                    float2 r_179_1 = fma_f32x2_rn_noftz(q_178_1, scale1_149, a_177_1);
                    acc2[18] = r_179_1.x;
                    acc2[19] = r_179_1.y;
                    float2 _f2_396 = make_float2(acc2[20], acc2[21]);
                    float2 a_180_1 = _f2_396;
                    float2 _f2_397 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                    float2 q_181_1 = _f2_397;
                    float2 r_182_1 = fma_f32x2_rn_noftz(q_181_1, scale1_149, a_180_1);
                    acc2[20] = r_182_1.x;
                    acc2[21] = r_182_1.y;
                    float2 _f2_398 = make_float2(acc2[22], acc2[23]);
                    float2 a_183_1 = _f2_398;
                    float2 _f2_399 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                    float2 q_184_1 = _f2_399;
                    float2 r_185_1 = fma_f32x2_rn_noftz(q_184_1, scale1_149, a_183_1);
                    acc2[22] = r_185_1.x;
                    acc2[23] = r_185_1.y;
                    float _unpack_fp4_47[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_vec_load_6[3 + 0]));
                        (&_unpack_fp4_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_47[0]), "=f"(_unpack_fp4_47[1]), "=f"(_unpack_fp4_47[2]), "=f"(_unpack_fp4_47[3]), "=f"(_unpack_fp4_47[4]), "=f"(_unpack_fp4_47[5]), "=f"(_unpack_fp4_47[6]), "=f"(_unpack_fp4_47[7]) : "r"(_vec_load_6[3 + 0]));
                        #endif
                    }
                    float2 _f2_400 = make_float2(acc2[24], acc2[25]);
                    float2 a_186_1 = _f2_400;
                    float2 _f2_401 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                    float2 q_187_1 = _f2_401;
                    float2 r_188_1 = fma_f32x2_rn_noftz(q_187_1, scale1_149, a_186_1);
                    acc2[24] = r_188_1.x;
                    acc2[25] = r_188_1.y;
                    float2 _f2_402 = make_float2(acc2[26], acc2[27]);
                    float2 a_189_1 = _f2_402;
                    float2 _f2_403 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                    float2 q_190_1 = _f2_403;
                    float2 r_191_1 = fma_f32x2_rn_noftz(q_190_1, scale1_149, a_189_1);
                    acc2[26] = r_191_1.x;
                    acc2[27] = r_191_1.y;
                    float2 _f2_404 = make_float2(acc2[28], acc2[29]);
                    float2 a_192_1 = _f2_404;
                    float2 _f2_405 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                    float2 q_193_1 = _f2_405;
                    float2 r_194_1 = fma_f32x2_rn_noftz(q_193_1, scale1_149, a_192_1);
                    acc2[28] = r_194_1.x;
                    acc2[29] = r_194_1.y;
                    float2 _f2_406 = make_float2(acc2[30], acc2[31]);
                    float2 a_195 = _f2_406;
                    float2 _f2_407 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                    float2 q_196 = _f2_407;
                    float2 r_197 = fma_f32x2_rn_noftz(q_196, scale1_149, a_195);
                    acc2[30] = r_197.x;
                    acc2[31] = r_197.y;
                    unsigned int q_198[4];
                    float bm0 = 0.0f;
                    float bm1 = 0.0f;
                    float _fabs_64;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_64) : "f"(acc2[0]));
                    float _max_64 = max_noftz(bm0, _fabs_64);
                    bm0 = _max_64;
                    float _fabs_65;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_65) : "f"(acc2[16]));
                    float _max_65 = max_noftz(bm1, _fabs_65);
                    bm1 = _max_65;
                    float _fabs_66;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_66) : "f"(acc2[1]));
                    float _max_66 = max_noftz(bm0, _fabs_66);
                    bm0 = _max_66;
                    float _fabs_67;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_67) : "f"(acc2[17]));
                    float _max_67 = max_noftz(bm1, _fabs_67);
                    bm1 = _max_67;
                    float _fabs_68;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_68) : "f"(acc2[2]));
                    float _max_68 = max_noftz(bm0, _fabs_68);
                    bm0 = _max_68;
                    float _fabs_69;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_69) : "f"(acc2[18]));
                    float _max_69 = max_noftz(bm1, _fabs_69);
                    bm1 = _max_69;
                    float _fabs_70;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_70) : "f"(acc2[3]));
                    float _max_70 = max_noftz(bm0, _fabs_70);
                    bm0 = _max_70;
                    float _fabs_71;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_71) : "f"(acc2[19]));
                    float _max_71 = max_noftz(bm1, _fabs_71);
                    bm1 = _max_71;
                    float _fabs_72;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_72) : "f"(acc2[4]));
                    float _max_72 = max_noftz(bm0, _fabs_72);
                    bm0 = _max_72;
                    float _fabs_73;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_73) : "f"(acc2[20]));
                    float _max_73 = max_noftz(bm1, _fabs_73);
                    bm1 = _max_73;
                    float _fabs_74;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_74) : "f"(acc2[5]));
                    float _max_74 = max_noftz(bm0, _fabs_74);
                    bm0 = _max_74;
                    float _fabs_75;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_75) : "f"(acc2[21]));
                    float _max_75 = max_noftz(bm1, _fabs_75);
                    bm1 = _max_75;
                    float _fabs_76;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_76) : "f"(acc2[6]));
                    float _max_76 = max_noftz(bm0, _fabs_76);
                    bm0 = _max_76;
                    float _fabs_77;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_77) : "f"(acc2[22]));
                    float _max_77 = max_noftz(bm1, _fabs_77);
                    bm1 = _max_77;
                    float _fabs_78;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_78) : "f"(acc2[7]));
                    float _max_78 = max_noftz(bm0, _fabs_78);
                    bm0 = _max_78;
                    float _fabs_79;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_79) : "f"(acc2[23]));
                    float _max_79 = max_noftz(bm1, _fabs_79);
                    bm1 = _max_79;
                    float _fabs_80;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_80) : "f"(acc2[8]));
                    float _max_80 = max_noftz(bm0, _fabs_80);
                    bm0 = _max_80;
                    float _fabs_81;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_81) : "f"(acc2[24]));
                    float _max_81 = max_noftz(bm1, _fabs_81);
                    bm1 = _max_81;
                    float _fabs_82;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_82) : "f"(acc2[9]));
                    float _max_82 = max_noftz(bm0, _fabs_82);
                    bm0 = _max_82;
                    float _fabs_83;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_83) : "f"(acc2[25]));
                    float _max_83 = max_noftz(bm1, _fabs_83);
                    bm1 = _max_83;
                    float _fabs_84;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_84) : "f"(acc2[10]));
                    float _max_84 = max_noftz(bm0, _fabs_84);
                    bm0 = _max_84;
                    float _fabs_85;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_85) : "f"(acc2[26]));
                    float _max_85 = max_noftz(bm1, _fabs_85);
                    bm1 = _max_85;
                    float _fabs_86;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_86) : "f"(acc2[11]));
                    float _max_86 = max_noftz(bm0, _fabs_86);
                    bm0 = _max_86;
                    float _fabs_87;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_87) : "f"(acc2[27]));
                    float _max_87 = max_noftz(bm1, _fabs_87);
                    bm1 = _max_87;
                    float _fabs_88;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_88) : "f"(acc2[12]));
                    float _max_88 = max_noftz(bm0, _fabs_88);
                    bm0 = _max_88;
                    float _fabs_89;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_89) : "f"(acc2[28]));
                    float _max_89 = max_noftz(bm1, _fabs_89);
                    bm1 = _max_89;
                    float _fabs_90;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_90) : "f"(acc2[13]));
                    float _max_90 = max_noftz(bm0, _fabs_90);
                    bm0 = _max_90;
                    float _fabs_91;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_91) : "f"(acc2[29]));
                    float _max_91 = max_noftz(bm1, _fabs_91);
                    bm1 = _max_91;
                    float _fabs_92;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_92) : "f"(acc2[14]));
                    float _max_92 = max_noftz(bm0, _fabs_92);
                    bm0 = _max_92;
                    float _fabs_93;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_93) : "f"(acc2[30]));
                    float _max_93 = max_noftz(bm1, _fabs_93);
                    bm1 = _max_93;
                    float _fabs_94;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_94) : "f"(acc2[15]));
                    float _max_94 = max_noftz(bm0, _fabs_94);
                    bm0 = _max_94;
                    float _fabs_95;
                    asm("abs.f32 %0, %1;" : "=f"(_fabs_95) : "f"(acc2[31]));
                    float _max_95 = max_noftz(bm1, _fabs_95);
                    bm1 = _max_95;
                    float _mul_24;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(bm0), "f"(inv));
                    float _fdiv_rn_2 = __fdiv_rn(_mul_24, 6.0f);
                    float _mul_25;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(bm1), "f"(inv));
                    float _fdiv_rn_3 = __fdiv_rn(_mul_25, 6.0f);
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_rn_3), "f"(_fdiv_rn_2));
                    uint32_t _e4m3x2_to_f16x2_12;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_12) : "h"((uint16_t)(_e4m3x2_f32_0)));
                    float _cvt_f32_f16_24;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_24) : "h"((uint16_t)(_e4m3x2_to_f16x2_12)));
                    float _cvt_f32_f16_25;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_25) : "h"((uint16_t)(_e4m3x2_to_f16x2_12 >> 16)));
                    float _fdiv_rn_4 = __fdiv_rn(inv, _cvt_f32_f16_24);
                    float _fdiv_rn_5 = __fdiv_rn(inv, _cvt_f32_f16_25);
                    float m0 = 0.0f;
                    float m1 = 0.0f;
                    if (_cvt_f32_f16_24 > 0.0f) {
                        m0 = _fdiv_rn_4;
                    }
                    if (_cvt_f32_f16_25 > 0.0f) {
                        m1 = _fdiv_rn_5;
                    }
                    float2 _f2_408 = make_float2(m0, m0);
                    float2 mm = _f2_408;
                    float2 _f2_409 = make_float2(acc2[0], acc2[1]);
                    float2 pair = _f2_409;
                    float2 g;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&pair), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_0;
                    uint16_t _e2m1_pair_h_11;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(g.x), "f"(g.y));
                    _fp4_pair_0 = (uint32_t)_e2m1_pair_h_11;
                    float2 _f2_410 = make_float2(acc2[2], acc2[3]);
                    float2 pair_199 = _f2_410;
                    float2 g_200;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_200) : "l"(*(const unsigned long long*)&pair_199), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_1;
                    uint16_t _e2m1_pair_h_12;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(g_200.x), "f"(g_200.y));
                    _fp4_pair_1 = (uint32_t)_e2m1_pair_h_12;
                    float2 _f2_411 = make_float2(acc2[4], acc2[5]);
                    float2 pair_201 = _f2_411;
                    float2 g_202;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_202) : "l"(*(const unsigned long long*)&pair_201), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_2;
                    uint16_t _e2m1_pair_h_13;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(g_202.x), "f"(g_202.y));
                    _fp4_pair_2 = (uint32_t)_e2m1_pair_h_13;
                    float2 _f2_412 = make_float2(acc2[6], acc2[7]);
                    float2 pair_203 = _f2_412;
                    float2 g_204;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_204) : "l"(*(const unsigned long long*)&pair_203), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_3;
                    uint16_t _e2m1_pair_h_14;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(g_204.x), "f"(g_204.y));
                    _fp4_pair_3 = (uint32_t)_e2m1_pair_h_14;
                    unsigned int word = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_pair_3 << 24;
                    unsigned int carry = (word & 2004318071) + 2004318071;
                    q_198[0] = word & (carry | 2004318071);
                    float2 _f2_413 = make_float2(m0, m0);
                    float2 mm_205 = _f2_413;
                    float2 _f2_414 = make_float2(acc2[8], acc2[9]);
                    float2 pair_206 = _f2_414;
                    float2 g_207;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_207) : "l"(*(const unsigned long long*)&pair_206), "l"(*(const unsigned long long*)&mm_205));
                    uint32_t _fp4_pair_4;
                    uint16_t _e2m1_pair_h_15;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_207.x), "f"(g_207.y));
                    _fp4_pair_4 = (uint32_t)_e2m1_pair_h_15;
                    float2 _f2_415 = make_float2(acc2[10], acc2[11]);
                    float2 pair_208 = _f2_415;
                    float2 g_209;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_209) : "l"(*(const unsigned long long*)&pair_208), "l"(*(const unsigned long long*)&mm_205));
                    uint32_t _fp4_pair_5;
                    uint16_t _e2m1_pair_h_16;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_209.x), "f"(g_209.y));
                    _fp4_pair_5 = (uint32_t)_e2m1_pair_h_16;
                    float2 _f2_416 = make_float2(acc2[12], acc2[13]);
                    float2 pair_210 = _f2_416;
                    float2 g_211;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_211) : "l"(*(const unsigned long long*)&pair_210), "l"(*(const unsigned long long*)&mm_205));
                    uint32_t _fp4_pair_6;
                    uint16_t _e2m1_pair_h_17;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(g_211.x), "f"(g_211.y));
                    _fp4_pair_6 = (uint32_t)_e2m1_pair_h_17;
                    float2 _f2_417 = make_float2(acc2[14], acc2[15]);
                    float2 pair_212 = _f2_417;
                    float2 g_213;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_213) : "l"(*(const unsigned long long*)&pair_212), "l"(*(const unsigned long long*)&mm_205));
                    uint32_t _fp4_pair_7;
                    uint16_t _e2m1_pair_h_18;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(g_213.x), "f"(g_213.y));
                    _fp4_pair_7 = (uint32_t)_e2m1_pair_h_18;
                    unsigned int word_214 = _fp4_pair_4 | _fp4_pair_5 << 8 | _fp4_pair_6 << 16 | _fp4_pair_7 << 24;
                    unsigned int carry_215 = (word_214 & 2004318071) + 2004318071;
                    q_198[1] = word_214 & (carry_215 | 2004318071);
                    float2 _f2_418 = make_float2(m1, m1);
                    float2 mm_216 = _f2_418;
                    float2 _f2_419 = make_float2(acc2[16], acc2[17]);
                    float2 pair_217 = _f2_419;
                    float2 g_218;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_218) : "l"(*(const unsigned long long*)&pair_217), "l"(*(const unsigned long long*)&mm_216));
                    uint32_t _fp4_pair_8;
                    uint16_t _e2m1_pair_h_19;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_19) : "f"(g_218.x), "f"(g_218.y));
                    _fp4_pair_8 = (uint32_t)_e2m1_pair_h_19;
                    float2 _f2_420 = make_float2(acc2[18], acc2[19]);
                    float2 pair_219 = _f2_420;
                    float2 g_220;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_220) : "l"(*(const unsigned long long*)&pair_219), "l"(*(const unsigned long long*)&mm_216));
                    uint32_t _fp4_pair_9;
                    uint16_t _e2m1_pair_h_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_220.x), "f"(g_220.y));
                    _fp4_pair_9 = (uint32_t)_e2m1_pair_h_20;
                    float2 _f2_421 = make_float2(acc2[20], acc2[21]);
                    float2 pair_221 = _f2_421;
                    float2 g_222;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_222) : "l"(*(const unsigned long long*)&pair_221), "l"(*(const unsigned long long*)&mm_216));
                    uint32_t _fp4_pair_10;
                    uint16_t _e2m1_pair_h_21;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_222.x), "f"(g_222.y));
                    _fp4_pair_10 = (uint32_t)_e2m1_pair_h_21;
                    float2 _f2_422 = make_float2(acc2[22], acc2[23]);
                    float2 pair_223 = _f2_422;
                    float2 g_224;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_224) : "l"(*(const unsigned long long*)&pair_223), "l"(*(const unsigned long long*)&mm_216));
                    uint32_t _fp4_pair_11;
                    uint16_t _e2m1_pair_h_22;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(g_224.x), "f"(g_224.y));
                    _fp4_pair_11 = (uint32_t)_e2m1_pair_h_22;
                    unsigned int word_225 = _fp4_pair_8 | _fp4_pair_9 << 8 | _fp4_pair_10 << 16 | _fp4_pair_11 << 24;
                    unsigned int carry_226 = (word_225 & 2004318071) + 2004318071;
                    q_198[2] = word_225 & (carry_226 | 2004318071);
                    float2 _f2_423 = make_float2(m1, m1);
                    float2 mm_227 = _f2_423;
                    float2 _f2_424 = make_float2(acc2[24], acc2[25]);
                    float2 pair_228 = _f2_424;
                    float2 g_229;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_229) : "l"(*(const unsigned long long*)&pair_228), "l"(*(const unsigned long long*)&mm_227));
                    uint32_t _fp4_pair_12;
                    uint16_t _e2m1_pair_h_23;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(g_229.x), "f"(g_229.y));
                    _fp4_pair_12 = (uint32_t)_e2m1_pair_h_23;
                    float2 _f2_425 = make_float2(acc2[26], acc2[27]);
                    float2 pair_230 = _f2_425;
                    float2 g_231;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_231) : "l"(*(const unsigned long long*)&pair_230), "l"(*(const unsigned long long*)&mm_227));
                    uint32_t _fp4_pair_13;
                    uint16_t _e2m1_pair_h_24;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_24) : "f"(g_231.x), "f"(g_231.y));
                    _fp4_pair_13 = (uint32_t)_e2m1_pair_h_24;
                    float2 _f2_426 = make_float2(acc2[28], acc2[29]);
                    float2 pair_232 = _f2_426;
                    float2 g_233;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_233) : "l"(*(const unsigned long long*)&pair_232), "l"(*(const unsigned long long*)&mm_227));
                    uint32_t _fp4_pair_14;
                    uint16_t _e2m1_pair_h_25;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_233.x), "f"(g_233.y));
                    _fp4_pair_14 = (uint32_t)_e2m1_pair_h_25;
                    float2 _f2_427 = make_float2(acc2[30], acc2[31]);
                    float2 pair_234 = _f2_427;
                    float2 g_235;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_235) : "l"(*(const unsigned long long*)&pair_234), "l"(*(const unsigned long long*)&mm_227));
                    uint32_t _fp4_pair_15;
                    uint16_t _e2m1_pair_h_26;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(g_235.x), "f"(g_235.y));
                    _fp4_pair_15 = (uint32_t)_e2m1_pair_h_26;
                    unsigned int word_236 = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_pair_15 << 24;
                    unsigned int carry_237 = (word_236 & 2004318071) + 2004318071;
                    q_198[3] = word_236 & (carry_237 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(bob_off + 16 * pp)), "r"(q_198[0]), "r"(q_198[1]), "r"(q_198[2]), "r"(q_198[3]) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(bob_off + 3072 + 2 * pp)), "h"((uint16_t)(_e4m3x2_f32_0)) : "memory");
                    float d[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_13;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_13) : "h"((uint16_t)(_e4m3x2_f32_0)));
                    float _cvt_f32_f16_26;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_26) : "h"((uint16_t)(_e4m3x2_to_f16x2_13)));
                    float _cvt_f32_f16_27;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_27) : "h"((uint16_t)(_e4m3x2_to_f16x2_13 >> 16)));
                    float _mul_26;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_cvt_f32_f16_26), "f"(rs2));
                    float _mul_27;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_cvt_f32_f16_27), "f"(rs2));
                    float2 _f2_428 = make_float2(_mul_26, _mul_26);
                    float2 scale0_238 = _f2_428;
                    float2 _f2_429 = make_float2(_mul_27, _mul_27);
                    float2 scale1_239 = _f2_429;
                    float _unpack_fp4_48[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_198[0]));
                        (&_unpack_fp4_48[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_48[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_48[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_48[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_48[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_48[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_48[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_48[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_48[0]), "=f"(_unpack_fp4_48[1]), "=f"(_unpack_fp4_48[2]), "=f"(_unpack_fp4_48[3]), "=f"(_unpack_fp4_48[4]), "=f"(_unpack_fp4_48[5]), "=f"(_unpack_fp4_48[6]), "=f"(_unpack_fp4_48[7]) : "r"(q_198[0]));
                        #endif
                    }
                    float2 _f2_430 = make_float2(d[0], d[1]);
                    float2 a_240_1 = _f2_430;
                    float2 _f2_431 = make_float2(_unpack_fp4_48[0], _unpack_fp4_48[1]);
                    float2 q_241_1 = _f2_431;
                    float2 r_242_1 = fma_f32x2_rn_noftz(q_241_1, scale0_238, a_240_1);
                    d[0] = r_242_1.x;
                    d[1] = r_242_1.y;
                    float2 _f2_432 = make_float2(d[2], d[3]);
                    float2 a_243_1 = _f2_432;
                    float2 _f2_433 = make_float2(_unpack_fp4_48[2], _unpack_fp4_48[3]);
                    float2 q_244_1 = _f2_433;
                    float2 r_245_1 = fma_f32x2_rn_noftz(q_244_1, scale0_238, a_243_1);
                    d[2] = r_245_1.x;
                    d[3] = r_245_1.y;
                    float2 _f2_434 = make_float2(d[4], d[5]);
                    float2 a_246 = _f2_434;
                    float2 _f2_435 = make_float2(_unpack_fp4_48[4], _unpack_fp4_48[5]);
                    float2 q_247 = _f2_435;
                    float2 r_248 = fma_f32x2_rn_noftz(q_247, scale0_238, a_246);
                    d[4] = r_248.x;
                    d[5] = r_248.y;
                    float2 _f2_436 = make_float2(d[6], d[7]);
                    float2 a_249 = _f2_436;
                    float2 _f2_437 = make_float2(_unpack_fp4_48[6], _unpack_fp4_48[7]);
                    float2 q_250 = _f2_437;
                    float2 r_251 = fma_f32x2_rn_noftz(q_250, scale0_238, a_249);
                    d[6] = r_251.x;
                    d[7] = r_251.y;
                    float _unpack_fp4_49[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_198[1 + 0]));
                        (&_unpack_fp4_49[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_49[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_49[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_49[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_49[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_49[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_49[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_49[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_49[0]), "=f"(_unpack_fp4_49[1]), "=f"(_unpack_fp4_49[2]), "=f"(_unpack_fp4_49[3]), "=f"(_unpack_fp4_49[4]), "=f"(_unpack_fp4_49[5]), "=f"(_unpack_fp4_49[6]), "=f"(_unpack_fp4_49[7]) : "r"(q_198[1 + 0]));
                        #endif
                    }
                    float2 _f2_438 = make_float2(d[8], d[9]);
                    float2 a_252 = _f2_438;
                    float2 _f2_439 = make_float2(_unpack_fp4_49[0], _unpack_fp4_49[1]);
                    float2 q_253 = _f2_439;
                    float2 r_254 = fma_f32x2_rn_noftz(q_253, scale0_238, a_252);
                    d[8] = r_254.x;
                    d[9] = r_254.y;
                    float2 _f2_440 = make_float2(d[10], d[11]);
                    float2 a_255 = _f2_440;
                    float2 _f2_441 = make_float2(_unpack_fp4_49[2], _unpack_fp4_49[3]);
                    float2 q_256 = _f2_441;
                    float2 r_257 = fma_f32x2_rn_noftz(q_256, scale0_238, a_255);
                    d[10] = r_257.x;
                    d[11] = r_257.y;
                    float2 _f2_442 = make_float2(d[12], d[13]);
                    float2 a_258 = _f2_442;
                    float2 _f2_443 = make_float2(_unpack_fp4_49[4], _unpack_fp4_49[5]);
                    float2 q_259 = _f2_443;
                    float2 r_260 = fma_f32x2_rn_noftz(q_259, scale0_238, a_258);
                    d[12] = r_260.x;
                    d[13] = r_260.y;
                    float2 _f2_444 = make_float2(d[14], d[15]);
                    float2 a_261 = _f2_444;
                    float2 _f2_445 = make_float2(_unpack_fp4_49[6], _unpack_fp4_49[7]);
                    float2 q_262 = _f2_445;
                    float2 r_263 = fma_f32x2_rn_noftz(q_262, scale0_238, a_261);
                    d[14] = r_263.x;
                    d[15] = r_263.y;
                    float _unpack_fp4_50[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_198[2 + 0]));
                        (&_unpack_fp4_50[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_50[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_50[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_50[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_50[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_50[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_50[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_50[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_50[0]), "=f"(_unpack_fp4_50[1]), "=f"(_unpack_fp4_50[2]), "=f"(_unpack_fp4_50[3]), "=f"(_unpack_fp4_50[4]), "=f"(_unpack_fp4_50[5]), "=f"(_unpack_fp4_50[6]), "=f"(_unpack_fp4_50[7]) : "r"(q_198[2 + 0]));
                        #endif
                    }
                    float2 _f2_446 = make_float2(d[16], d[17]);
                    float2 a_264 = _f2_446;
                    float2 _f2_447 = make_float2(_unpack_fp4_50[0], _unpack_fp4_50[1]);
                    float2 q_265 = _f2_447;
                    float2 r_266 = fma_f32x2_rn_noftz(q_265, scale1_239, a_264);
                    d[16] = r_266.x;
                    d[17] = r_266.y;
                    float2 _f2_448 = make_float2(d[18], d[19]);
                    float2 a_267 = _f2_448;
                    float2 _f2_449 = make_float2(_unpack_fp4_50[2], _unpack_fp4_50[3]);
                    float2 q_268 = _f2_449;
                    float2 r_269 = fma_f32x2_rn_noftz(q_268, scale1_239, a_267);
                    d[18] = r_269.x;
                    d[19] = r_269.y;
                    float2 _f2_450 = make_float2(d[20], d[21]);
                    float2 a_270 = _f2_450;
                    float2 _f2_451 = make_float2(_unpack_fp4_50[4], _unpack_fp4_50[5]);
                    float2 q_271 = _f2_451;
                    float2 r_272 = fma_f32x2_rn_noftz(q_271, scale1_239, a_270);
                    d[20] = r_272.x;
                    d[21] = r_272.y;
                    float2 _f2_452 = make_float2(d[22], d[23]);
                    float2 a_273 = _f2_452;
                    float2 _f2_453 = make_float2(_unpack_fp4_50[6], _unpack_fp4_50[7]);
                    float2 q_274 = _f2_453;
                    float2 r_275 = fma_f32x2_rn_noftz(q_274, scale1_239, a_273);
                    d[22] = r_275.x;
                    d[23] = r_275.y;
                    float _unpack_fp4_51[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_198[3 + 0]));
                        (&_unpack_fp4_51[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_51[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_51[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_51[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_51[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_51[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_51[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_51[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_51[0]), "=f"(_unpack_fp4_51[1]), "=f"(_unpack_fp4_51[2]), "=f"(_unpack_fp4_51[3]), "=f"(_unpack_fp4_51[4]), "=f"(_unpack_fp4_51[5]), "=f"(_unpack_fp4_51[6]), "=f"(_unpack_fp4_51[7]) : "r"(q_198[3 + 0]));
                        #endif
                    }
                    float2 _f2_454 = make_float2(d[24], d[25]);
                    float2 a_276 = _f2_454;
                    float2 _f2_455 = make_float2(_unpack_fp4_51[0], _unpack_fp4_51[1]);
                    float2 q_277 = _f2_455;
                    float2 r_278 = fma_f32x2_rn_noftz(q_277, scale1_239, a_276);
                    d[24] = r_278.x;
                    d[25] = r_278.y;
                    float2 _f2_456 = make_float2(d[26], d[27]);
                    float2 a_279 = _f2_456;
                    float2 _f2_457 = make_float2(_unpack_fp4_51[2], _unpack_fp4_51[3]);
                    float2 q_280 = _f2_457;
                    float2 r_281 = fma_f32x2_rn_noftz(q_280, scale1_239, a_279);
                    d[26] = r_281.x;
                    d[27] = r_281.y;
                    float2 _f2_458 = make_float2(d[28], d[29]);
                    float2 a_282 = _f2_458;
                    float2 _f2_459 = make_float2(_unpack_fp4_51[4], _unpack_fp4_51[5]);
                    float2 q_283 = _f2_459;
                    float2 r_284 = fma_f32x2_rn_noftz(q_283, scale1_239, a_282);
                    d[28] = r_284.x;
                    d[29] = r_284.y;
                    float2 _f2_460 = make_float2(d[30], d[31]);
                    float2 a_285 = _f2_460;
                    float2 _f2_461 = make_float2(_unpack_fp4_51[6], _unpack_fp4_51[7]);
                    float2 q_286 = _f2_461;
                    float2 r_287 = fma_f32x2_rn_noftz(q_286, scale1_239, a_285);
                    d[30] = r_287.x;
                    d[31] = r_287.y;
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
                    int c_288 = lane_1;
                    int l = c_288 >> 2;
                    unsigned int w4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l + (c_288 & 3 ^ l >> 1 & 3)))));
                    long long idx = brow64 * 3072 + (long long)(4 * (128 * j + c_288));
                    {
                        int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx) = _iv4;
                    }
                    int c_289 = lane_1 + 32;
                    int l_290 = c_289 >> 2;
                    unsigned int w4_291[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_291[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_291[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_291[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_291[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_290 + (c_289 & 3 ^ l_290 >> 1 & 3)))));
                    long long idx_292 = brow64 * 3072 + (long long)(4 * (128 * j + c_289));
                    {
                        int4 _iv4 = make_int4(w4_291[0 + 0], w4_291[0 + 1], w4_291[0 + 2], w4_291[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_292) = _iv4;
                    }
                    int c_293 = lane_1 + 64;
                    int l_294 = c_293 >> 2;
                    unsigned int w4_295[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_295[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_295[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_295[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_295[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_294 + (c_293 & 3 ^ l_294 >> 1 & 3)))));
                    long long idx_296 = brow64 * 3072 + (long long)(4 * (128 * j + c_293));
                    {
                        int4 _iv4 = make_int4(w4_295[0 + 0], w4_295[0 + 1], w4_295[0 + 2], w4_295[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_296) = _iv4;
                    }
                    int c_297 = lane_1 + 96;
                    int l_298 = c_297 >> 2;
                    unsigned int w4_299[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_299[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_299[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_299[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_299[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_298 + (c_297 & 3 ^ l_298 >> 1 & 3)))));
                    long long idx_300 = brow64 * 3072 + (long long)(4 * (128 * j + c_297));
                    {
                        int4 _iv4 = make_int4(w4_299[0 + 0], w4_299[0 + 1], w4_299[0 + 2], w4_299[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_300) = _iv4;
                    }
                    __syncwarp();
                }
                if (lane_1 < 4) {
                    {
                        uint32_t _addr_27 = static_cast<uint32_t>(wsm_addr + (unsigned int)(bob_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(rs2) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    int ag_wo = (int)ag_w;
                    int ag_ro = (int)ag_r;
                    int capo_0 = (int)capo64;
                    int ix_1_1 = (int)bix64;
                    int rk_2 = (int)rank64;
                    int v_off_1 = ag_wo + (rk_2 * capo_0 + ix_1_1) * 3072;
                    int sc_off_1 = ag_wo + 4 * capo_0 * 3072 + (rk_2 * capo_0 + ix_1_1) * 384;
                    int m_off = ag_ro + (rk_2 * capo_0 + ix_1_1) * 16;
                    int peer = (rank + 1) % 4;
                    uint8_t* pw = reinterpret_cast<uint8_t*>(tab[peer]);
                    uint8_t* pr = reinterpret_cast<uint8_t*>(tab[4 + peer]);
                    {
                        void* _cpbulk_dst_28 = reinterpret_cast<void*>(pw + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_28), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_29 = reinterpret_cast<void*>(pw + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_29), "r"(wsm_addr + (unsigned int)(bob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_30 = reinterpret_cast<void*>(pr + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_30), "r"(wsm_addr + (unsigned int)(bob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_3 = (rank + 2) % 4;
                    uint8_t* pw_4 = reinterpret_cast<uint8_t*>(tab[peer_3]);
                    uint8_t* pr_5 = reinterpret_cast<uint8_t*>(tab[4 + peer_3]);
                    {
                        void* _cpbulk_dst_31 = reinterpret_cast<void*>(pw_4 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_31), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_32 = reinterpret_cast<void*>(pw_4 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_32), "r"(wsm_addr + (unsigned int)(bob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_33 = reinterpret_cast<void*>(pr_5 + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_33), "r"(wsm_addr + (unsigned int)(bob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_6 = (rank + 3) % 4;
                    uint8_t* pw_7 = reinterpret_cast<uint8_t*>(tab[peer_6]);
                    uint8_t* pr_8 = reinterpret_cast<uint8_t*>(tab[4 + peer_6]);
                    {
                        void* _cpbulk_dst_34 = reinterpret_cast<void*>(pw_7 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_34), "r"(wsm_addr + (unsigned int)bob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_35 = reinterpret_cast<void*>(pw_7 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_35), "r"(wsm_addr + (unsigned int)(bob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_36 = reinterpret_cast<void*>(pr_8 + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_36), "r"(wsm_addr + (unsigned int)(bob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                kb = kb + 1;
            }
            int fme = rb0 - r0;
            #pragma unroll 1
            for (int c_3 = wC; c_3 < nC; c_3 += 10) {
                int blk = c_3 / 3;
                int w = c_3 % 3;
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
                int m_off_1 = ag_ro_1 + (src * capo_2 + ix_2) * 16;
                float rs_1 = 0.0f;
                uint32_t _sysv_poll32_group_0[1];
                do {
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(ts_rs + m_off_1)) : "memory");
                } while (
                    (_sysv_poll32_group_0[0] == 0x80000000u)
                );
                rs_1 = __uint_as_float(_sysv_poll32_group_0[0]);
                int v_1 = ag_wo_1 + (src * capo_2 + ix_2) * 3072;
                int sc_2 = ag_wo_1 + 4 * capo_2 * 3072 + (src * capo_2 + ix_2) * 384;
                int ppj = lane_1;
                int ppj_0 = lane_1 + 32;
                int ppj_1 = lane_1 + 64;
                int ppj_2 = lane_1 + 96;
                int ppj_3 = lane_1 + 128;
                int ppj_4 = lane_1 + 160;
                uint32_t _fp4_wire_values_1[24];
                uint32_t _fp4_wire_scales_1[6];
                uint16_t _wire_h16_37_0;
                uint16_t _wire_h16_37_1;
                uint16_t _wire_h16_37_2;
                uint16_t _wire_h16_37_3;
                uint16_t _wire_h16_37_4;
                uint16_t _wire_h16_37_5;
                uint32_t _wire_vpend_37 = 63u;
                uint32_t _wire_spend_37 = 63u;
                do {
                    if (_wire_vpend_37 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[0]), "=r"(_fp4_wire_values_1[1]), "=r"(_fp4_wire_values_1[2]), "=r"(_fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj)))) : "memory");
                    }
                    if (_wire_vpend_37 & 2u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[4]), "=r"(_fp4_wire_values_1[5]), "=r"(_fp4_wire_values_1[6]), "=r"(_fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_0)))) : "memory");
                    }
                    if (_wire_vpend_37 & 4u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[8]), "=r"(_fp4_wire_values_1[9]), "=r"(_fp4_wire_values_1[10]), "=r"(_fp4_wire_values_1[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_1)))) : "memory");
                    }
                    if (_wire_vpend_37 & 8u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[12]), "=r"(_fp4_wire_values_1[13]), "=r"(_fp4_wire_values_1[14]), "=r"(_fp4_wire_values_1[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_2)))) : "memory");
                    }
                    if (_wire_vpend_37 & 16u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[16]), "=r"(_fp4_wire_values_1[17]), "=r"(_fp4_wire_values_1[18]), "=r"(_fp4_wire_values_1[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_3)))) : "memory");
                    }
                    if (_wire_vpend_37 & 32u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[20]), "=r"(_fp4_wire_values_1[21]), "=r"(_fp4_wire_values_1[22]), "=r"(_fp4_wire_values_1[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * ppj_4)))) : "memory");
                    }
                    if (_wire_spend_37 & 1u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_37_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * ppj)))) : "memory");
                        _fp4_wire_scales_1[0] = (uint32_t)_wire_h16_37_0;
                    }
                    if (_wire_spend_37 & 2u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_37_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * ppj_0)))) : "memory");
                        _fp4_wire_scales_1[1] = (uint32_t)_wire_h16_37_1;
                    }
                    if (_wire_spend_37 & 4u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_37_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * ppj_1)))) : "memory");
                        _fp4_wire_scales_1[2] = (uint32_t)_wire_h16_37_2;
                    }
                    if (_wire_spend_37 & 8u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_37_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * ppj_2)))) : "memory");
                        _fp4_wire_scales_1[3] = (uint32_t)_wire_h16_37_3;
                    }
                    if (_wire_spend_37 & 16u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_37_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * ppj_3)))) : "memory");
                        _fp4_wire_scales_1[4] = (uint32_t)_wire_h16_37_4;
                    }
                    if (_wire_spend_37 & 32u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_37_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * ppj_4)))) : "memory");
                        _fp4_wire_scales_1[5] = (uint32_t)_wire_h16_37_5;
                    }
                    _wire_vpend_37 = ((uint32_t)((((_fp4_wire_values_1[0] & ~((_fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[1] & ~((_fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[2] & ~((_fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[3] & ~((_fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_1[4] & ~((_fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[5] & ~((_fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[6] & ~((_fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[7] & ~((_fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_1[8] & ~((_fp4_wire_values_1[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[9] & ~((_fp4_wire_values_1[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[10] & ~((_fp4_wire_values_1[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[11] & ~((_fp4_wire_values_1[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((_fp4_wire_values_1[12] & ~((_fp4_wire_values_1[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[13] & ~((_fp4_wire_values_1[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[14] & ~((_fp4_wire_values_1[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[15] & ~((_fp4_wire_values_1[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((_fp4_wire_values_1[16] & ~((_fp4_wire_values_1[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[17] & ~((_fp4_wire_values_1[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[18] & ~((_fp4_wire_values_1[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[19] & ~((_fp4_wire_values_1[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((_fp4_wire_values_1[20] & ~((_fp4_wire_values_1[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[21] & ~((_fp4_wire_values_1[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[22] & ~((_fp4_wire_values_1[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[23] & ~((_fp4_wire_values_1[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5);
                    _wire_spend_37 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[3] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[4] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[5] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5);
                } while ((_wire_vpend_37 | _wire_spend_37) != 0u);
                float acc_1[32] = {0};
                uint32_t _e4m3x2_to_f16x2_14;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_14) : "h"((uint16_t)(_fp4_wire_scales_1[0])));
                float _cvt_f32_f16_28;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_28) : "h"((uint16_t)(_e4m3x2_to_f16x2_14)));
                float _cvt_f32_f16_29;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_29) : "h"((uint16_t)(_e4m3x2_to_f16x2_14 >> 16)));
                float _mul_28;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_cvt_f32_f16_28), "f"(rs_1));
                float _mul_29;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_cvt_f32_f16_29), "f"(rs_1));
                float2 _f2_462 = make_float2(_mul_28, _mul_28);
                float2 scale0_2 = _f2_462;
                float2 _f2_463 = make_float2(_mul_29, _mul_29);
                float2 scale1_2 = _f2_463;
                float _unpack_fp4_52[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[0]));
                    (&_unpack_fp4_52[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_52[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_52[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_52[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_52[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_52[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_52[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_52[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_52[0]), "=f"(_unpack_fp4_52[1]), "=f"(_unpack_fp4_52[2]), "=f"(_unpack_fp4_52[3]), "=f"(_unpack_fp4_52[4]), "=f"(_unpack_fp4_52[5]), "=f"(_unpack_fp4_52[6]), "=f"(_unpack_fp4_52[7]) : "r"(_fp4_wire_values_1[0]));
                    #endif
                }
                float2 _f2_464 = make_float2(acc_1[0], acc_1[1]);
                float2 a_2 = _f2_464;
                float2 _f2_465 = make_float2(_unpack_fp4_52[0], _unpack_fp4_52[1]);
                float2 q_3 = _f2_465;
                float2 r_3 = fma_f32x2_rn_noftz(q_3, scale0_2, a_2);
                acc_1[0] = r_3.x;
                acc_1[1] = r_3.y;
                float2 _f2_466 = make_float2(acc_1[2], acc_1[3]);
                float2 a_5 = _f2_466;
                float2 _f2_467 = make_float2(_unpack_fp4_52[2], _unpack_fp4_52[3]);
                float2 q_6 = _f2_467;
                float2 r_7 = fma_f32x2_rn_noftz(q_6, scale0_2, a_5);
                acc_1[2] = r_7.x;
                acc_1[3] = r_7.y;
                float2 _f2_468 = make_float2(acc_1[4], acc_1[5]);
                float2 a_8 = _f2_468;
                float2 _f2_469 = make_float2(_unpack_fp4_52[4], _unpack_fp4_52[5]);
                float2 q_9 = _f2_469;
                float2 r_10 = fma_f32x2_rn_noftz(q_9, scale0_2, a_8);
                acc_1[4] = r_10.x;
                acc_1[5] = r_10.y;
                float2 _f2_470 = make_float2(acc_1[6], acc_1[7]);
                float2 a_11 = _f2_470;
                float2 _f2_471 = make_float2(_unpack_fp4_52[6], _unpack_fp4_52[7]);
                float2 q_12 = _f2_471;
                float2 r_13 = fma_f32x2_rn_noftz(q_12, scale0_2, a_11);
                acc_1[6] = r_13.x;
                acc_1[7] = r_13.y;
                float _unpack_fp4_53[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[1 + 0]));
                    (&_unpack_fp4_53[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_53[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_53[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_53[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_53[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_53[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_53[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_53[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_53[0]), "=f"(_unpack_fp4_53[1]), "=f"(_unpack_fp4_53[2]), "=f"(_unpack_fp4_53[3]), "=f"(_unpack_fp4_53[4]), "=f"(_unpack_fp4_53[5]), "=f"(_unpack_fp4_53[6]), "=f"(_unpack_fp4_53[7]) : "r"(_fp4_wire_values_1[1 + 0]));
                    #endif
                }
                float2 _f2_472 = make_float2(acc_1[8], acc_1[9]);
                float2 a_14 = _f2_472;
                float2 _f2_473 = make_float2(_unpack_fp4_53[0], _unpack_fp4_53[1]);
                float2 q_15 = _f2_473;
                float2 r_16 = fma_f32x2_rn_noftz(q_15, scale0_2, a_14);
                acc_1[8] = r_16.x;
                acc_1[9] = r_16.y;
                float2 _f2_474 = make_float2(acc_1[10], acc_1[11]);
                float2 a_17 = _f2_474;
                float2 _f2_475 = make_float2(_unpack_fp4_53[2], _unpack_fp4_53[3]);
                float2 q_18 = _f2_475;
                float2 r_19 = fma_f32x2_rn_noftz(q_18, scale0_2, a_17);
                acc_1[10] = r_19.x;
                acc_1[11] = r_19.y;
                float2 _f2_476 = make_float2(acc_1[12], acc_1[13]);
                float2 a_20 = _f2_476;
                float2 _f2_477 = make_float2(_unpack_fp4_53[4], _unpack_fp4_53[5]);
                float2 q_21 = _f2_477;
                float2 r_22 = fma_f32x2_rn_noftz(q_21, scale0_2, a_20);
                acc_1[12] = r_22.x;
                acc_1[13] = r_22.y;
                float2 _f2_478 = make_float2(acc_1[14], acc_1[15]);
                float2 a_23 = _f2_478;
                float2 _f2_479 = make_float2(_unpack_fp4_53[6], _unpack_fp4_53[7]);
                float2 q_24 = _f2_479;
                float2 r_25 = fma_f32x2_rn_noftz(q_24, scale0_2, a_23);
                acc_1[14] = r_25.x;
                acc_1[15] = r_25.y;
                float _unpack_fp4_54[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[2 + 0]));
                    (&_unpack_fp4_54[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_54[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_54[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_54[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_54[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_54[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_54[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_54[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_54[0]), "=f"(_unpack_fp4_54[1]), "=f"(_unpack_fp4_54[2]), "=f"(_unpack_fp4_54[3]), "=f"(_unpack_fp4_54[4]), "=f"(_unpack_fp4_54[5]), "=f"(_unpack_fp4_54[6]), "=f"(_unpack_fp4_54[7]) : "r"(_fp4_wire_values_1[2 + 0]));
                    #endif
                }
                float2 _f2_480 = make_float2(acc_1[16], acc_1[17]);
                float2 a_26 = _f2_480;
                float2 _f2_481 = make_float2(_unpack_fp4_54[0], _unpack_fp4_54[1]);
                float2 q_27 = _f2_481;
                float2 r_28 = fma_f32x2_rn_noftz(q_27, scale1_2, a_26);
                acc_1[16] = r_28.x;
                acc_1[17] = r_28.y;
                float2 _f2_482 = make_float2(acc_1[18], acc_1[19]);
                float2 a_29 = _f2_482;
                float2 _f2_483 = make_float2(_unpack_fp4_54[2], _unpack_fp4_54[3]);
                float2 q_30 = _f2_483;
                float2 r_31 = fma_f32x2_rn_noftz(q_30, scale1_2, a_29);
                acc_1[18] = r_31.x;
                acc_1[19] = r_31.y;
                float2 _f2_484 = make_float2(acc_1[20], acc_1[21]);
                float2 a_32 = _f2_484;
                float2 _f2_485 = make_float2(_unpack_fp4_54[4], _unpack_fp4_54[5]);
                float2 q_33 = _f2_485;
                float2 r_34 = fma_f32x2_rn_noftz(q_33, scale1_2, a_32);
                acc_1[20] = r_34.x;
                acc_1[21] = r_34.y;
                float2 _f2_486 = make_float2(acc_1[22], acc_1[23]);
                float2 a_35 = _f2_486;
                float2 _f2_487 = make_float2(_unpack_fp4_54[6], _unpack_fp4_54[7]);
                float2 q_36 = _f2_487;
                float2 r_37 = fma_f32x2_rn_noftz(q_36, scale1_2, a_35);
                acc_1[22] = r_37.x;
                acc_1[23] = r_37.y;
                float _unpack_fp4_55[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[3 + 0]));
                    (&_unpack_fp4_55[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_55[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_55[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_55[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_55[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_55[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_55[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_55[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_55[0]), "=f"(_unpack_fp4_55[1]), "=f"(_unpack_fp4_55[2]), "=f"(_unpack_fp4_55[3]), "=f"(_unpack_fp4_55[4]), "=f"(_unpack_fp4_55[5]), "=f"(_unpack_fp4_55[6]), "=f"(_unpack_fp4_55[7]) : "r"(_fp4_wire_values_1[3 + 0]));
                    #endif
                }
                float2 _f2_488 = make_float2(acc_1[24], acc_1[25]);
                float2 a_38 = _f2_488;
                float2 _f2_489 = make_float2(_unpack_fp4_55[0], _unpack_fp4_55[1]);
                float2 q_39 = _f2_489;
                float2 r_40 = fma_f32x2_rn_noftz(q_39, scale1_2, a_38);
                acc_1[24] = r_40.x;
                acc_1[25] = r_40.y;
                float2 _f2_490 = make_float2(acc_1[26], acc_1[27]);
                float2 a_41 = _f2_490;
                float2 _f2_491 = make_float2(_unpack_fp4_55[2], _unpack_fp4_55[3]);
                float2 q_42 = _f2_491;
                float2 r_43 = fma_f32x2_rn_noftz(q_42, scale1_2, a_41);
                acc_1[26] = r_43.x;
                acc_1[27] = r_43.y;
                float2 _f2_492 = make_float2(acc_1[28], acc_1[29]);
                float2 a_44 = _f2_492;
                float2 _f2_493 = make_float2(_unpack_fp4_55[4], _unpack_fp4_55[5]);
                float2 q_45 = _f2_493;
                float2 r_46 = fma_f32x2_rn_noftz(q_45, scale1_2, a_44);
                acc_1[28] = r_46.x;
                acc_1[29] = r_46.y;
                float2 _f2_494 = make_float2(acc_1[30], acc_1[31]);
                float2 a_47_1 = _f2_494;
                float2 _f2_495 = make_float2(_unpack_fp4_55[6], _unpack_fp4_55[7]);
                float2 q_48_1 = _f2_495;
                float2 r_49_1 = fma_f32x2_rn_noftz(q_48_1, scale1_2, a_47_1);
                acc_1[30] = r_49_1.x;
                acc_1[31] = r_49_1.y;
                unsigned int acc_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_1[_lp*2 + 0], acc_1[_lp*2+1 + 0]));
                    acc_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                __syncwarp();
                int c_50 = lane_1;
                int l_1 = c_50 >> 2;
                unsigned int w4_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1 + (c_50 & 3 ^ l_1 >> 1 & 3)))));
                long long idx_1 = crow64 * 3072 + (long long)(4 * c_50);
                {
                    int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1) = _iv4;
                }
                int c_51 = lane_1 + 32;
                int l_52 = c_51 >> 2;
                unsigned int w4_53[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_53[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_52 + (c_51 & 3 ^ l_52 >> 1 & 3)))));
                long long idx_54 = crow64 * 3072 + (long long)(4 * c_51);
                {
                    int4 _iv4 = make_int4(w4_53[0 + 0], w4_53[0 + 1], w4_53[0 + 2], w4_53[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_54) = _iv4;
                }
                int c_55 = lane_1 + 64;
                int l_56 = c_55 >> 2;
                unsigned int w4_57[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_57[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_56 + (c_55 & 3 ^ l_56 >> 1 & 3)))));
                long long idx_58 = crow64 * 3072 + (long long)(4 * c_55);
                {
                    int4 _iv4 = make_int4(w4_57[0 + 0], w4_57[0 + 1], w4_57[0 + 2], w4_57[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_58) = _iv4;
                }
                int c_59 = lane_1 + 96;
                int l_60 = c_59 >> 2;
                unsigned int w4_61[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_61[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_60 + (c_59 & 3 ^ l_60 >> 1 & 3)))));
                long long idx_62 = crow64 * 3072 + (long long)(4 * c_59);
                {
                    int4 _iv4 = make_int4(w4_61[0 + 0], w4_61[0 + 1], w4_61[0 + 2], w4_61[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_62) = _iv4;
                }
                __syncwarp();
                float acc_63[32] = {0};
                uint32_t _e4m3x2_to_f16x2_15;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_15) : "h"((uint16_t)(_fp4_wire_scales_1[1])));
                float _cvt_f32_f16_30;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_30) : "h"((uint16_t)(_e4m3x2_to_f16x2_15)));
                float _cvt_f32_f16_31;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_31) : "h"((uint16_t)(_e4m3x2_to_f16x2_15 >> 16)));
                float _mul_30;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_cvt_f32_f16_30), "f"(rs_1));
                float _mul_31;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_cvt_f32_f16_31), "f"(rs_1));
                float2 _f2_496 = make_float2(_mul_30, _mul_30);
                float2 scale0_64 = _f2_496;
                float2 _f2_497 = make_float2(_mul_31, _mul_31);
                float2 scale1_65 = _f2_497;
                float _unpack_fp4_56[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[4 + 0]));
                    (&_unpack_fp4_56[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_56[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_56[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_56[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_56[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_56[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_56[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_56[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_56[0]), "=f"(_unpack_fp4_56[1]), "=f"(_unpack_fp4_56[2]), "=f"(_unpack_fp4_56[3]), "=f"(_unpack_fp4_56[4]), "=f"(_unpack_fp4_56[5]), "=f"(_unpack_fp4_56[6]), "=f"(_unpack_fp4_56[7]) : "r"(_fp4_wire_values_1[4 + 0]));
                    #endif
                }
                float2 _f2_498 = make_float2(acc_63[0], acc_63[1]);
                float2 a_66 = _f2_498;
                float2 _f2_499 = make_float2(_unpack_fp4_56[0], _unpack_fp4_56[1]);
                float2 q_67 = _f2_499;
                float2 r_68 = fma_f32x2_rn_noftz(q_67, scale0_64, a_66);
                acc_63[0] = r_68.x;
                acc_63[1] = r_68.y;
                float2 _f2_500 = make_float2(acc_63[2], acc_63[3]);
                float2 a_69 = _f2_500;
                float2 _f2_501 = make_float2(_unpack_fp4_56[2], _unpack_fp4_56[3]);
                float2 q_70 = _f2_501;
                float2 r_71 = fma_f32x2_rn_noftz(q_70, scale0_64, a_69);
                acc_63[2] = r_71.x;
                acc_63[3] = r_71.y;
                float2 _f2_502 = make_float2(acc_63[4], acc_63[5]);
                float2 a_72 = _f2_502;
                float2 _f2_503 = make_float2(_unpack_fp4_56[4], _unpack_fp4_56[5]);
                float2 q_73 = _f2_503;
                float2 r_74 = fma_f32x2_rn_noftz(q_73, scale0_64, a_72);
                acc_63[4] = r_74.x;
                acc_63[5] = r_74.y;
                float2 _f2_504 = make_float2(acc_63[6], acc_63[7]);
                float2 a_75 = _f2_504;
                float2 _f2_505 = make_float2(_unpack_fp4_56[6], _unpack_fp4_56[7]);
                float2 q_76 = _f2_505;
                float2 r_77 = fma_f32x2_rn_noftz(q_76, scale0_64, a_75);
                acc_63[6] = r_77.x;
                acc_63[7] = r_77.y;
                float _unpack_fp4_57[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[5 + 0]));
                    (&_unpack_fp4_57[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_57[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_57[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_57[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_57[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_57[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_57[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_57[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_57[0]), "=f"(_unpack_fp4_57[1]), "=f"(_unpack_fp4_57[2]), "=f"(_unpack_fp4_57[3]), "=f"(_unpack_fp4_57[4]), "=f"(_unpack_fp4_57[5]), "=f"(_unpack_fp4_57[6]), "=f"(_unpack_fp4_57[7]) : "r"(_fp4_wire_values_1[5 + 0]));
                    #endif
                }
                float2 _f2_506 = make_float2(acc_63[8], acc_63[9]);
                float2 a_78 = _f2_506;
                float2 _f2_507 = make_float2(_unpack_fp4_57[0], _unpack_fp4_57[1]);
                float2 q_79 = _f2_507;
                float2 r_80 = fma_f32x2_rn_noftz(q_79, scale0_64, a_78);
                acc_63[8] = r_80.x;
                acc_63[9] = r_80.y;
                float2 _f2_508 = make_float2(acc_63[10], acc_63[11]);
                float2 a_81 = _f2_508;
                float2 _f2_509 = make_float2(_unpack_fp4_57[2], _unpack_fp4_57[3]);
                float2 q_82 = _f2_509;
                float2 r_83 = fma_f32x2_rn_noftz(q_82, scale0_64, a_81);
                acc_63[10] = r_83.x;
                acc_63[11] = r_83.y;
                float2 _f2_510 = make_float2(acc_63[12], acc_63[13]);
                float2 a_84 = _f2_510;
                float2 _f2_511 = make_float2(_unpack_fp4_57[4], _unpack_fp4_57[5]);
                float2 q_85 = _f2_511;
                float2 r_86 = fma_f32x2_rn_noftz(q_85, scale0_64, a_84);
                acc_63[12] = r_86.x;
                acc_63[13] = r_86.y;
                float2 _f2_512 = make_float2(acc_63[14], acc_63[15]);
                float2 a_87 = _f2_512;
                float2 _f2_513 = make_float2(_unpack_fp4_57[6], _unpack_fp4_57[7]);
                float2 q_88 = _f2_513;
                float2 r_89 = fma_f32x2_rn_noftz(q_88, scale0_64, a_87);
                acc_63[14] = r_89.x;
                acc_63[15] = r_89.y;
                float _unpack_fp4_58[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[6 + 0]));
                    (&_unpack_fp4_58[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_58[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_58[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_58[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_58[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_58[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_58[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_58[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_58[0]), "=f"(_unpack_fp4_58[1]), "=f"(_unpack_fp4_58[2]), "=f"(_unpack_fp4_58[3]), "=f"(_unpack_fp4_58[4]), "=f"(_unpack_fp4_58[5]), "=f"(_unpack_fp4_58[6]), "=f"(_unpack_fp4_58[7]) : "r"(_fp4_wire_values_1[6 + 0]));
                    #endif
                }
                float2 _f2_514 = make_float2(acc_63[16], acc_63[17]);
                float2 a_90 = _f2_514;
                float2 _f2_515 = make_float2(_unpack_fp4_58[0], _unpack_fp4_58[1]);
                float2 q_91 = _f2_515;
                float2 r_92 = fma_f32x2_rn_noftz(q_91, scale1_65, a_90);
                acc_63[16] = r_92.x;
                acc_63[17] = r_92.y;
                float2 _f2_516 = make_float2(acc_63[18], acc_63[19]);
                float2 a_93 = _f2_516;
                float2 _f2_517 = make_float2(_unpack_fp4_58[2], _unpack_fp4_58[3]);
                float2 q_94 = _f2_517;
                float2 r_95 = fma_f32x2_rn_noftz(q_94, scale1_65, a_93);
                acc_63[18] = r_95.x;
                acc_63[19] = r_95.y;
                float2 _f2_518 = make_float2(acc_63[20], acc_63[21]);
                float2 a_96 = _f2_518;
                float2 _f2_519 = make_float2(_unpack_fp4_58[4], _unpack_fp4_58[5]);
                float2 q_97 = _f2_519;
                float2 r_98 = fma_f32x2_rn_noftz(q_97, scale1_65, a_96);
                acc_63[20] = r_98.x;
                acc_63[21] = r_98.y;
                float2 _f2_520 = make_float2(acc_63[22], acc_63[23]);
                float2 a_99 = _f2_520;
                float2 _f2_521 = make_float2(_unpack_fp4_58[6], _unpack_fp4_58[7]);
                float2 q_100 = _f2_521;
                float2 r_101 = fma_f32x2_rn_noftz(q_100, scale1_65, a_99);
                acc_63[22] = r_101.x;
                acc_63[23] = r_101.y;
                float _unpack_fp4_59[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[7 + 0]));
                    (&_unpack_fp4_59[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_59[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_59[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_59[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_59[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_59[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_59[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_59[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_59[0]), "=f"(_unpack_fp4_59[1]), "=f"(_unpack_fp4_59[2]), "=f"(_unpack_fp4_59[3]), "=f"(_unpack_fp4_59[4]), "=f"(_unpack_fp4_59[5]), "=f"(_unpack_fp4_59[6]), "=f"(_unpack_fp4_59[7]) : "r"(_fp4_wire_values_1[7 + 0]));
                    #endif
                }
                float2 _f2_522 = make_float2(acc_63[24], acc_63[25]);
                float2 a_102 = _f2_522;
                float2 _f2_523 = make_float2(_unpack_fp4_59[0], _unpack_fp4_59[1]);
                float2 q_103 = _f2_523;
                float2 r_104 = fma_f32x2_rn_noftz(q_103, scale1_65, a_102);
                acc_63[24] = r_104.x;
                acc_63[25] = r_104.y;
                float2 _f2_524 = make_float2(acc_63[26], acc_63[27]);
                float2 a_105 = _f2_524;
                float2 _f2_525 = make_float2(_unpack_fp4_59[2], _unpack_fp4_59[3]);
                float2 q_106 = _f2_525;
                float2 r_107 = fma_f32x2_rn_noftz(q_106, scale1_65, a_105);
                acc_63[26] = r_107.x;
                acc_63[27] = r_107.y;
                float2 _f2_526 = make_float2(acc_63[28], acc_63[29]);
                float2 a_108 = _f2_526;
                float2 _f2_527 = make_float2(_unpack_fp4_59[4], _unpack_fp4_59[5]);
                float2 q_109 = _f2_527;
                float2 r_110 = fma_f32x2_rn_noftz(q_109, scale1_65, a_108);
                acc_63[28] = r_110.x;
                acc_63[29] = r_110.y;
                float2 _f2_528 = make_float2(acc_63[30], acc_63[31]);
                float2 a_111 = _f2_528;
                float2 _f2_529 = make_float2(_unpack_fp4_59[6], _unpack_fp4_59[7]);
                float2 q_112 = _f2_529;
                float2 r_113 = fma_f32x2_rn_noftz(q_112, scale1_65, a_111);
                acc_63[30] = r_113.x;
                acc_63[31] = r_113.y;
                unsigned int acc_63_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_63[_lp*2 + 0], acc_63[_lp*2+1 + 0]));
                    acc_63_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_63_bf16[0]), "r"(acc_63_bf16[1]), "r"(acc_63_bf16[2]), "r"(acc_63_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_63_bf16[4]), "r"(acc_63_bf16[5]), "r"(acc_63_bf16[6]), "r"(acc_63_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_63_bf16[8]), "r"(acc_63_bf16[9]), "r"(acc_63_bf16[10]), "r"(acc_63_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_63_bf16[12]), "r"(acc_63_bf16[13]), "r"(acc_63_bf16[14]), "r"(acc_63_bf16[15]) : "memory");
                __syncwarp();
                int c_114 = lane_1;
                int l_115 = c_114 >> 2;
                unsigned int w4_116[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_116[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_116[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_116[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_116[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_115 + (c_114 & 3 ^ l_115 >> 1 & 3)))));
                long long idx_117 = crow64 * 3072 + (long long)(4 * (128 + c_114));
                {
                    int4 _iv4 = make_int4(w4_116[0 + 0], w4_116[0 + 1], w4_116[0 + 2], w4_116[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_117) = _iv4;
                }
                int c_118 = lane_1 + 32;
                int l_119 = c_118 >> 2;
                unsigned int w4_120[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_120[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_120[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_120[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_120[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_119 + (c_118 & 3 ^ l_119 >> 1 & 3)))));
                long long idx_121 = crow64 * 3072 + (long long)(4 * (128 + c_118));
                {
                    int4 _iv4 = make_int4(w4_120[0 + 0], w4_120[0 + 1], w4_120[0 + 2], w4_120[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_121) = _iv4;
                }
                int c_122 = lane_1 + 64;
                int l_123 = c_122 >> 2;
                unsigned int w4_124[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_124[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_124[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_124[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_124[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_123 + (c_122 & 3 ^ l_123 >> 1 & 3)))));
                long long idx_125 = crow64 * 3072 + (long long)(4 * (128 + c_122));
                {
                    int4 _iv4 = make_int4(w4_124[0 + 0], w4_124[0 + 1], w4_124[0 + 2], w4_124[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_125) = _iv4;
                }
                int c_126 = lane_1 + 96;
                int l_127 = c_126 >> 2;
                unsigned int w4_128[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_128[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_128[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_128[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_128[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_127 + (c_126 & 3 ^ l_127 >> 1 & 3)))));
                long long idx_129 = crow64 * 3072 + (long long)(4 * (128 + c_126));
                {
                    int4 _iv4 = make_int4(w4_128[0 + 0], w4_128[0 + 1], w4_128[0 + 2], w4_128[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_129) = _iv4;
                }
                __syncwarp();
                float acc_130[32] = {0};
                uint32_t _e4m3x2_to_f16x2_16;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_16) : "h"((uint16_t)(_fp4_wire_scales_1[2])));
                float _cvt_f32_f16_32;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_32) : "h"((uint16_t)(_e4m3x2_to_f16x2_16)));
                float _cvt_f32_f16_33;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_33) : "h"((uint16_t)(_e4m3x2_to_f16x2_16 >> 16)));
                float _mul_32;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(_cvt_f32_f16_32), "f"(rs_1));
                float _mul_33;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_cvt_f32_f16_33), "f"(rs_1));
                float2 _f2_530 = make_float2(_mul_32, _mul_32);
                float2 scale0_131 = _f2_530;
                float2 _f2_531 = make_float2(_mul_33, _mul_33);
                float2 scale1_132 = _f2_531;
                float _unpack_fp4_60[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[8 + 0]));
                    (&_unpack_fp4_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_60[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_60[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_60[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_60[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_60[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_60[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_60[0]), "=f"(_unpack_fp4_60[1]), "=f"(_unpack_fp4_60[2]), "=f"(_unpack_fp4_60[3]), "=f"(_unpack_fp4_60[4]), "=f"(_unpack_fp4_60[5]), "=f"(_unpack_fp4_60[6]), "=f"(_unpack_fp4_60[7]) : "r"(_fp4_wire_values_1[8 + 0]));
                    #endif
                }
                float2 _f2_532 = make_float2(acc_130[0], acc_130[1]);
                float2 a_133_2 = _f2_532;
                float2 _f2_533 = make_float2(_unpack_fp4_60[0], _unpack_fp4_60[1]);
                float2 q_134_2 = _f2_533;
                float2 r_135_2 = fma_f32x2_rn_noftz(q_134_2, scale0_131, a_133_2);
                acc_130[0] = r_135_2.x;
                acc_130[1] = r_135_2.y;
                float2 _f2_534 = make_float2(acc_130[2], acc_130[3]);
                float2 a_136_2 = _f2_534;
                float2 _f2_535 = make_float2(_unpack_fp4_60[2], _unpack_fp4_60[3]);
                float2 q_137_2 = _f2_535;
                float2 r_138_2 = fma_f32x2_rn_noftz(q_137_2, scale0_131, a_136_2);
                acc_130[2] = r_138_2.x;
                acc_130[3] = r_138_2.y;
                float2 _f2_536 = make_float2(acc_130[4], acc_130[5]);
                float2 a_139_2 = _f2_536;
                float2 _f2_537 = make_float2(_unpack_fp4_60[4], _unpack_fp4_60[5]);
                float2 q_140_2 = _f2_537;
                float2 r_141_2 = fma_f32x2_rn_noftz(q_140_2, scale0_131, a_139_2);
                acc_130[4] = r_141_2.x;
                acc_130[5] = r_141_2.y;
                float2 _f2_538 = make_float2(acc_130[6], acc_130[7]);
                float2 a_142_2 = _f2_538;
                float2 _f2_539 = make_float2(_unpack_fp4_60[6], _unpack_fp4_60[7]);
                float2 q_143_2 = _f2_539;
                float2 r_144_2 = fma_f32x2_rn_noftz(q_143_2, scale0_131, a_142_2);
                acc_130[6] = r_144_2.x;
                acc_130[7] = r_144_2.y;
                float _unpack_fp4_61[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[9 + 0]));
                    (&_unpack_fp4_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_61[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_61[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_61[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_61[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_61[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_61[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_61[0]), "=f"(_unpack_fp4_61[1]), "=f"(_unpack_fp4_61[2]), "=f"(_unpack_fp4_61[3]), "=f"(_unpack_fp4_61[4]), "=f"(_unpack_fp4_61[5]), "=f"(_unpack_fp4_61[6]), "=f"(_unpack_fp4_61[7]) : "r"(_fp4_wire_values_1[9 + 0]));
                    #endif
                }
                float2 _f2_540 = make_float2(acc_130[8], acc_130[9]);
                float2 a_145_1 = _f2_540;
                float2 _f2_541 = make_float2(_unpack_fp4_61[0], _unpack_fp4_61[1]);
                float2 q_146_1 = _f2_541;
                float2 r_147_1 = fma_f32x2_rn_noftz(q_146_1, scale0_131, a_145_1);
                acc_130[8] = r_147_1.x;
                acc_130[9] = r_147_1.y;
                float2 _f2_542 = make_float2(acc_130[10], acc_130[11]);
                float2 a_148 = _f2_542;
                float2 _f2_543 = make_float2(_unpack_fp4_61[2], _unpack_fp4_61[3]);
                float2 q_149 = _f2_543;
                float2 r_150 = fma_f32x2_rn_noftz(q_149, scale0_131, a_148);
                acc_130[10] = r_150.x;
                acc_130[11] = r_150.y;
                float2 _f2_544 = make_float2(acc_130[12], acc_130[13]);
                float2 a_151 = _f2_544;
                float2 _f2_545 = make_float2(_unpack_fp4_61[4], _unpack_fp4_61[5]);
                float2 q_152 = _f2_545;
                float2 r_153 = fma_f32x2_rn_noftz(q_152, scale0_131, a_151);
                acc_130[12] = r_153.x;
                acc_130[13] = r_153.y;
                float2 _f2_546 = make_float2(acc_130[14], acc_130[15]);
                float2 a_154 = _f2_546;
                float2 _f2_547 = make_float2(_unpack_fp4_61[6], _unpack_fp4_61[7]);
                float2 q_155 = _f2_547;
                float2 r_156 = fma_f32x2_rn_noftz(q_155, scale0_131, a_154);
                acc_130[14] = r_156.x;
                acc_130[15] = r_156.y;
                float _unpack_fp4_62[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[10 + 0]));
                    (&_unpack_fp4_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_62[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_62[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_62[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_62[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_62[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_62[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_62[0]), "=f"(_unpack_fp4_62[1]), "=f"(_unpack_fp4_62[2]), "=f"(_unpack_fp4_62[3]), "=f"(_unpack_fp4_62[4]), "=f"(_unpack_fp4_62[5]), "=f"(_unpack_fp4_62[6]), "=f"(_unpack_fp4_62[7]) : "r"(_fp4_wire_values_1[10 + 0]));
                    #endif
                }
                float2 _f2_548 = make_float2(acc_130[16], acc_130[17]);
                float2 a_157 = _f2_548;
                float2 _f2_549 = make_float2(_unpack_fp4_62[0], _unpack_fp4_62[1]);
                float2 q_158 = _f2_549;
                float2 r_159 = fma_f32x2_rn_noftz(q_158, scale1_132, a_157);
                acc_130[16] = r_159.x;
                acc_130[17] = r_159.y;
                float2 _f2_550 = make_float2(acc_130[18], acc_130[19]);
                float2 a_160 = _f2_550;
                float2 _f2_551 = make_float2(_unpack_fp4_62[2], _unpack_fp4_62[3]);
                float2 q_161 = _f2_551;
                float2 r_162 = fma_f32x2_rn_noftz(q_161, scale1_132, a_160);
                acc_130[18] = r_162.x;
                acc_130[19] = r_162.y;
                float2 _f2_552 = make_float2(acc_130[20], acc_130[21]);
                float2 a_163 = _f2_552;
                float2 _f2_553 = make_float2(_unpack_fp4_62[4], _unpack_fp4_62[5]);
                float2 q_164 = _f2_553;
                float2 r_165 = fma_f32x2_rn_noftz(q_164, scale1_132, a_163);
                acc_130[20] = r_165.x;
                acc_130[21] = r_165.y;
                float2 _f2_554 = make_float2(acc_130[22], acc_130[23]);
                float2 a_166 = _f2_554;
                float2 _f2_555 = make_float2(_unpack_fp4_62[6], _unpack_fp4_62[7]);
                float2 q_167 = _f2_555;
                float2 r_168 = fma_f32x2_rn_noftz(q_167, scale1_132, a_166);
                acc_130[22] = r_168.x;
                acc_130[23] = r_168.y;
                float _unpack_fp4_63[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[11 + 0]));
                    (&_unpack_fp4_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_63[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_63[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_63[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_63[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_63[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_63[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_63[0]), "=f"(_unpack_fp4_63[1]), "=f"(_unpack_fp4_63[2]), "=f"(_unpack_fp4_63[3]), "=f"(_unpack_fp4_63[4]), "=f"(_unpack_fp4_63[5]), "=f"(_unpack_fp4_63[6]), "=f"(_unpack_fp4_63[7]) : "r"(_fp4_wire_values_1[11 + 0]));
                    #endif
                }
                float2 _f2_556 = make_float2(acc_130[24], acc_130[25]);
                float2 a_169 = _f2_556;
                float2 _f2_557 = make_float2(_unpack_fp4_63[0], _unpack_fp4_63[1]);
                float2 q_170 = _f2_557;
                float2 r_171 = fma_f32x2_rn_noftz(q_170, scale1_132, a_169);
                acc_130[24] = r_171.x;
                acc_130[25] = r_171.y;
                float2 _f2_558 = make_float2(acc_130[26], acc_130[27]);
                float2 a_172 = _f2_558;
                float2 _f2_559 = make_float2(_unpack_fp4_63[2], _unpack_fp4_63[3]);
                float2 q_173 = _f2_559;
                float2 r_174 = fma_f32x2_rn_noftz(q_173, scale1_132, a_172);
                acc_130[26] = r_174.x;
                acc_130[27] = r_174.y;
                float2 _f2_560 = make_float2(acc_130[28], acc_130[29]);
                float2 a_175 = _f2_560;
                float2 _f2_561 = make_float2(_unpack_fp4_63[4], _unpack_fp4_63[5]);
                float2 q_176 = _f2_561;
                float2 r_177 = fma_f32x2_rn_noftz(q_176, scale1_132, a_175);
                acc_130[28] = r_177.x;
                acc_130[29] = r_177.y;
                float2 _f2_562 = make_float2(acc_130[30], acc_130[31]);
                float2 a_178 = _f2_562;
                float2 _f2_563 = make_float2(_unpack_fp4_63[6], _unpack_fp4_63[7]);
                float2 q_179 = _f2_563;
                float2 r_180 = fma_f32x2_rn_noftz(q_179, scale1_132, a_178);
                acc_130[30] = r_180.x;
                acc_130[31] = r_180.y;
                unsigned int acc_130_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_130[_lp*2 + 0], acc_130[_lp*2+1 + 0]));
                    acc_130_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_130_bf16[0]), "r"(acc_130_bf16[1]), "r"(acc_130_bf16[2]), "r"(acc_130_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_130_bf16[4]), "r"(acc_130_bf16[5]), "r"(acc_130_bf16[6]), "r"(acc_130_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_130_bf16[8]), "r"(acc_130_bf16[9]), "r"(acc_130_bf16[10]), "r"(acc_130_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_130_bf16[12]), "r"(acc_130_bf16[13]), "r"(acc_130_bf16[14]), "r"(acc_130_bf16[15]) : "memory");
                __syncwarp();
                int c_181 = lane_1;
                int l_182 = c_181 >> 2;
                unsigned int w4_183[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_183[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_183[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_183[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_183[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_182 + (c_181 & 3 ^ l_182 >> 1 & 3)))));
                long long idx_184 = crow64 * 3072 + (long long)(4 * (256 + c_181));
                {
                    int4 _iv4 = make_int4(w4_183[0 + 0], w4_183[0 + 1], w4_183[0 + 2], w4_183[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_184) = _iv4;
                }
                int c_185 = lane_1 + 32;
                int l_186 = c_185 >> 2;
                unsigned int w4_187[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_187[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_187[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_187[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_187[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_186 + (c_185 & 3 ^ l_186 >> 1 & 3)))));
                long long idx_188 = crow64 * 3072 + (long long)(4 * (256 + c_185));
                {
                    int4 _iv4 = make_int4(w4_187[0 + 0], w4_187[0 + 1], w4_187[0 + 2], w4_187[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_188) = _iv4;
                }
                int c_189 = lane_1 + 64;
                int l_190 = c_189 >> 2;
                unsigned int w4_191[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_191[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_191[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_191[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_191[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_190 + (c_189 & 3 ^ l_190 >> 1 & 3)))));
                long long idx_192 = crow64 * 3072 + (long long)(4 * (256 + c_189));
                {
                    int4 _iv4 = make_int4(w4_191[0 + 0], w4_191[0 + 1], w4_191[0 + 2], w4_191[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_192) = _iv4;
                }
                int c_193 = lane_1 + 96;
                int l_194 = c_193 >> 2;
                unsigned int w4_195[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_195[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_195[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_195[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_195[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_194 + (c_193 & 3 ^ l_194 >> 1 & 3)))));
                long long idx_196 = crow64 * 3072 + (long long)(4 * (256 + c_193));
                {
                    int4 _iv4 = make_int4(w4_195[0 + 0], w4_195[0 + 1], w4_195[0 + 2], w4_195[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_196) = _iv4;
                }
                __syncwarp();
                float acc_197[32] = {0};
                uint32_t _e4m3x2_to_f16x2_17;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_17) : "h"((uint16_t)(_fp4_wire_scales_1[3])));
                float _cvt_f32_f16_34;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_34) : "h"((uint16_t)(_e4m3x2_to_f16x2_17)));
                float _cvt_f32_f16_35;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_35) : "h"((uint16_t)(_e4m3x2_to_f16x2_17 >> 16)));
                float _mul_34;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_cvt_f32_f16_34), "f"(rs_1));
                float _mul_35;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_cvt_f32_f16_35), "f"(rs_1));
                float2 _f2_564 = make_float2(_mul_34, _mul_34);
                float2 scale0_198 = _f2_564;
                float2 _f2_565 = make_float2(_mul_35, _mul_35);
                float2 scale1_199 = _f2_565;
                float _unpack_fp4_64[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[12 + 0]));
                    (&_unpack_fp4_64[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_64[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_64[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_64[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_64[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_64[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_64[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_64[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_64[0]), "=f"(_unpack_fp4_64[1]), "=f"(_unpack_fp4_64[2]), "=f"(_unpack_fp4_64[3]), "=f"(_unpack_fp4_64[4]), "=f"(_unpack_fp4_64[5]), "=f"(_unpack_fp4_64[6]), "=f"(_unpack_fp4_64[7]) : "r"(_fp4_wire_values_1[12 + 0]));
                    #endif
                }
                float2 _f2_566 = make_float2(acc_197[0], acc_197[1]);
                float2 a_200 = _f2_566;
                float2 _f2_567 = make_float2(_unpack_fp4_64[0], _unpack_fp4_64[1]);
                float2 q_201 = _f2_567;
                float2 r_202 = fma_f32x2_rn_noftz(q_201, scale0_198, a_200);
                acc_197[0] = r_202.x;
                acc_197[1] = r_202.y;
                float2 _f2_568 = make_float2(acc_197[2], acc_197[3]);
                float2 a_203 = _f2_568;
                float2 _f2_569 = make_float2(_unpack_fp4_64[2], _unpack_fp4_64[3]);
                float2 q_204 = _f2_569;
                float2 r_205 = fma_f32x2_rn_noftz(q_204, scale0_198, a_203);
                acc_197[2] = r_205.x;
                acc_197[3] = r_205.y;
                float2 _f2_570 = make_float2(acc_197[4], acc_197[5]);
                float2 a_206 = _f2_570;
                float2 _f2_571 = make_float2(_unpack_fp4_64[4], _unpack_fp4_64[5]);
                float2 q_207 = _f2_571;
                float2 r_208 = fma_f32x2_rn_noftz(q_207, scale0_198, a_206);
                acc_197[4] = r_208.x;
                acc_197[5] = r_208.y;
                float2 _f2_572 = make_float2(acc_197[6], acc_197[7]);
                float2 a_209 = _f2_572;
                float2 _f2_573 = make_float2(_unpack_fp4_64[6], _unpack_fp4_64[7]);
                float2 q_210 = _f2_573;
                float2 r_211 = fma_f32x2_rn_noftz(q_210, scale0_198, a_209);
                acc_197[6] = r_211.x;
                acc_197[7] = r_211.y;
                float _unpack_fp4_65[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[13 + 0]));
                    (&_unpack_fp4_65[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_65[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_65[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_65[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_65[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_65[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_65[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_65[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_65[0]), "=f"(_unpack_fp4_65[1]), "=f"(_unpack_fp4_65[2]), "=f"(_unpack_fp4_65[3]), "=f"(_unpack_fp4_65[4]), "=f"(_unpack_fp4_65[5]), "=f"(_unpack_fp4_65[6]), "=f"(_unpack_fp4_65[7]) : "r"(_fp4_wire_values_1[13 + 0]));
                    #endif
                }
                float2 _f2_574 = make_float2(acc_197[8], acc_197[9]);
                float2 a_212 = _f2_574;
                float2 _f2_575 = make_float2(_unpack_fp4_65[0], _unpack_fp4_65[1]);
                float2 q_213 = _f2_575;
                float2 r_214 = fma_f32x2_rn_noftz(q_213, scale0_198, a_212);
                acc_197[8] = r_214.x;
                acc_197[9] = r_214.y;
                float2 _f2_576 = make_float2(acc_197[10], acc_197[11]);
                float2 a_215 = _f2_576;
                float2 _f2_577 = make_float2(_unpack_fp4_65[2], _unpack_fp4_65[3]);
                float2 q_216 = _f2_577;
                float2 r_217 = fma_f32x2_rn_noftz(q_216, scale0_198, a_215);
                acc_197[10] = r_217.x;
                acc_197[11] = r_217.y;
                float2 _f2_578 = make_float2(acc_197[12], acc_197[13]);
                float2 a_218 = _f2_578;
                float2 _f2_579 = make_float2(_unpack_fp4_65[4], _unpack_fp4_65[5]);
                float2 q_219 = _f2_579;
                float2 r_220 = fma_f32x2_rn_noftz(q_219, scale0_198, a_218);
                acc_197[12] = r_220.x;
                acc_197[13] = r_220.y;
                float2 _f2_580 = make_float2(acc_197[14], acc_197[15]);
                float2 a_221 = _f2_580;
                float2 _f2_581 = make_float2(_unpack_fp4_65[6], _unpack_fp4_65[7]);
                float2 q_222 = _f2_581;
                float2 r_223 = fma_f32x2_rn_noftz(q_222, scale0_198, a_221);
                acc_197[14] = r_223.x;
                acc_197[15] = r_223.y;
                float _unpack_fp4_66[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[14 + 0]));
                    (&_unpack_fp4_66[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_66[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_66[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_66[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_66[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_66[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_66[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_66[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_66[0]), "=f"(_unpack_fp4_66[1]), "=f"(_unpack_fp4_66[2]), "=f"(_unpack_fp4_66[3]), "=f"(_unpack_fp4_66[4]), "=f"(_unpack_fp4_66[5]), "=f"(_unpack_fp4_66[6]), "=f"(_unpack_fp4_66[7]) : "r"(_fp4_wire_values_1[14 + 0]));
                    #endif
                }
                float2 _f2_582 = make_float2(acc_197[16], acc_197[17]);
                float2 a_224 = _f2_582;
                float2 _f2_583 = make_float2(_unpack_fp4_66[0], _unpack_fp4_66[1]);
                float2 q_225 = _f2_583;
                float2 r_226 = fma_f32x2_rn_noftz(q_225, scale1_199, a_224);
                acc_197[16] = r_226.x;
                acc_197[17] = r_226.y;
                float2 _f2_584 = make_float2(acc_197[18], acc_197[19]);
                float2 a_227 = _f2_584;
                float2 _f2_585 = make_float2(_unpack_fp4_66[2], _unpack_fp4_66[3]);
                float2 q_228 = _f2_585;
                float2 r_229 = fma_f32x2_rn_noftz(q_228, scale1_199, a_227);
                acc_197[18] = r_229.x;
                acc_197[19] = r_229.y;
                float2 _f2_586 = make_float2(acc_197[20], acc_197[21]);
                float2 a_230 = _f2_586;
                float2 _f2_587 = make_float2(_unpack_fp4_66[4], _unpack_fp4_66[5]);
                float2 q_231 = _f2_587;
                float2 r_232 = fma_f32x2_rn_noftz(q_231, scale1_199, a_230);
                acc_197[20] = r_232.x;
                acc_197[21] = r_232.y;
                float2 _f2_588 = make_float2(acc_197[22], acc_197[23]);
                float2 a_233 = _f2_588;
                float2 _f2_589 = make_float2(_unpack_fp4_66[6], _unpack_fp4_66[7]);
                float2 q_234 = _f2_589;
                float2 r_235 = fma_f32x2_rn_noftz(q_234, scale1_199, a_233);
                acc_197[22] = r_235.x;
                acc_197[23] = r_235.y;
                float _unpack_fp4_67[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[15 + 0]));
                    (&_unpack_fp4_67[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_67[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_67[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_67[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_67[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_67[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_67[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_67[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_67[0]), "=f"(_unpack_fp4_67[1]), "=f"(_unpack_fp4_67[2]), "=f"(_unpack_fp4_67[3]), "=f"(_unpack_fp4_67[4]), "=f"(_unpack_fp4_67[5]), "=f"(_unpack_fp4_67[6]), "=f"(_unpack_fp4_67[7]) : "r"(_fp4_wire_values_1[15 + 0]));
                    #endif
                }
                float2 _f2_590 = make_float2(acc_197[24], acc_197[25]);
                float2 a_236 = _f2_590;
                float2 _f2_591 = make_float2(_unpack_fp4_67[0], _unpack_fp4_67[1]);
                float2 q_237 = _f2_591;
                float2 r_238 = fma_f32x2_rn_noftz(q_237, scale1_199, a_236);
                acc_197[24] = r_238.x;
                acc_197[25] = r_238.y;
                float2 _f2_592 = make_float2(acc_197[26], acc_197[27]);
                float2 a_239 = _f2_592;
                float2 _f2_593 = make_float2(_unpack_fp4_67[2], _unpack_fp4_67[3]);
                float2 q_240 = _f2_593;
                float2 r_241 = fma_f32x2_rn_noftz(q_240, scale1_199, a_239);
                acc_197[26] = r_241.x;
                acc_197[27] = r_241.y;
                float2 _f2_594 = make_float2(acc_197[28], acc_197[29]);
                float2 a_242 = _f2_594;
                float2 _f2_595 = make_float2(_unpack_fp4_67[4], _unpack_fp4_67[5]);
                float2 q_243 = _f2_595;
                float2 r_244 = fma_f32x2_rn_noftz(q_243, scale1_199, a_242);
                acc_197[28] = r_244.x;
                acc_197[29] = r_244.y;
                float2 _f2_596 = make_float2(acc_197[30], acc_197[31]);
                float2 a_245 = _f2_596;
                float2 _f2_597 = make_float2(_unpack_fp4_67[6], _unpack_fp4_67[7]);
                float2 q_246 = _f2_597;
                float2 r_247 = fma_f32x2_rn_noftz(q_246, scale1_199, a_245);
                acc_197[30] = r_247.x;
                acc_197[31] = r_247.y;
                unsigned int acc_197_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_197[_lp*2 + 0], acc_197[_lp*2+1 + 0]));
                    acc_197_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_197_bf16[0]), "r"(acc_197_bf16[1]), "r"(acc_197_bf16[2]), "r"(acc_197_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_197_bf16[4]), "r"(acc_197_bf16[5]), "r"(acc_197_bf16[6]), "r"(acc_197_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_197_bf16[8]), "r"(acc_197_bf16[9]), "r"(acc_197_bf16[10]), "r"(acc_197_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_197_bf16[12]), "r"(acc_197_bf16[13]), "r"(acc_197_bf16[14]), "r"(acc_197_bf16[15]) : "memory");
                __syncwarp();
                int c_248 = lane_1;
                int l_249 = c_248 >> 2;
                unsigned int w4_250[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_250[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_250[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_249 + (c_248 & 3 ^ l_249 >> 1 & 3)))));
                long long idx_251 = crow64 * 3072 + (long long)(4 * (384 + c_248));
                {
                    int4 _iv4 = make_int4(w4_250[0 + 0], w4_250[0 + 1], w4_250[0 + 2], w4_250[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_251) = _iv4;
                }
                int c_252 = lane_1 + 32;
                int l_253 = c_252 >> 2;
                unsigned int w4_254[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_254[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_254[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_253 + (c_252 & 3 ^ l_253 >> 1 & 3)))));
                long long idx_255 = crow64 * 3072 + (long long)(4 * (384 + c_252));
                {
                    int4 _iv4 = make_int4(w4_254[0 + 0], w4_254[0 + 1], w4_254[0 + 2], w4_254[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_255) = _iv4;
                }
                int c_256 = lane_1 + 64;
                int l_257 = c_256 >> 2;
                unsigned int w4_258[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_258[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_258[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_258[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_258[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_257 + (c_256 & 3 ^ l_257 >> 1 & 3)))));
                long long idx_259 = crow64 * 3072 + (long long)(4 * (384 + c_256));
                {
                    int4 _iv4 = make_int4(w4_258[0 + 0], w4_258[0 + 1], w4_258[0 + 2], w4_258[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_259) = _iv4;
                }
                int c_260 = lane_1 + 96;
                int l_261 = c_260 >> 2;
                unsigned int w4_262[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_262[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_262[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_262[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_262[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_261 + (c_260 & 3 ^ l_261 >> 1 & 3)))));
                long long idx_263 = crow64 * 3072 + (long long)(4 * (384 + c_260));
                {
                    int4 _iv4 = make_int4(w4_262[0 + 0], w4_262[0 + 1], w4_262[0 + 2], w4_262[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_263) = _iv4;
                }
                __syncwarp();
                float acc_264[32] = {0};
                uint32_t _e4m3x2_to_f16x2_18;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_18) : "h"((uint16_t)(_fp4_wire_scales_1[4])));
                float _cvt_f32_f16_36;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_36) : "h"((uint16_t)(_e4m3x2_to_f16x2_18)));
                float _cvt_f32_f16_37;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_37) : "h"((uint16_t)(_e4m3x2_to_f16x2_18 >> 16)));
                float _mul_36;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(_cvt_f32_f16_36), "f"(rs_1));
                float _mul_37;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(_cvt_f32_f16_37), "f"(rs_1));
                float2 _f2_598 = make_float2(_mul_36, _mul_36);
                float2 scale0_265 = _f2_598;
                float2 _f2_599 = make_float2(_mul_37, _mul_37);
                float2 scale1_266 = _f2_599;
                float _unpack_fp4_68[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[16 + 0]));
                    (&_unpack_fp4_68[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_68[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_68[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_68[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_68[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_68[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_68[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_68[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_68[0]), "=f"(_unpack_fp4_68[1]), "=f"(_unpack_fp4_68[2]), "=f"(_unpack_fp4_68[3]), "=f"(_unpack_fp4_68[4]), "=f"(_unpack_fp4_68[5]), "=f"(_unpack_fp4_68[6]), "=f"(_unpack_fp4_68[7]) : "r"(_fp4_wire_values_1[16 + 0]));
                    #endif
                }
                float2 _f2_600 = make_float2(acc_264[0], acc_264[1]);
                float2 a_267_1 = _f2_600;
                float2 _f2_601 = make_float2(_unpack_fp4_68[0], _unpack_fp4_68[1]);
                float2 q_268_1 = _f2_601;
                float2 r_269_1 = fma_f32x2_rn_noftz(q_268_1, scale0_265, a_267_1);
                acc_264[0] = r_269_1.x;
                acc_264[1] = r_269_1.y;
                float2 _f2_602 = make_float2(acc_264[2], acc_264[3]);
                float2 a_270_1 = _f2_602;
                float2 _f2_603 = make_float2(_unpack_fp4_68[2], _unpack_fp4_68[3]);
                float2 q_271_1 = _f2_603;
                float2 r_272_1 = fma_f32x2_rn_noftz(q_271_1, scale0_265, a_270_1);
                acc_264[2] = r_272_1.x;
                acc_264[3] = r_272_1.y;
                float2 _f2_604 = make_float2(acc_264[4], acc_264[5]);
                float2 a_273_1 = _f2_604;
                float2 _f2_605 = make_float2(_unpack_fp4_68[4], _unpack_fp4_68[5]);
                float2 q_274_1 = _f2_605;
                float2 r_275_1 = fma_f32x2_rn_noftz(q_274_1, scale0_265, a_273_1);
                acc_264[4] = r_275_1.x;
                acc_264[5] = r_275_1.y;
                float2 _f2_606 = make_float2(acc_264[6], acc_264[7]);
                float2 a_276_1 = _f2_606;
                float2 _f2_607 = make_float2(_unpack_fp4_68[6], _unpack_fp4_68[7]);
                float2 q_277_1 = _f2_607;
                float2 r_278_1 = fma_f32x2_rn_noftz(q_277_1, scale0_265, a_276_1);
                acc_264[6] = r_278_1.x;
                acc_264[7] = r_278_1.y;
                float _unpack_fp4_69[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[17 + 0]));
                    (&_unpack_fp4_69[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_69[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_69[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_69[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_69[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_69[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_69[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_69[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_69[0]), "=f"(_unpack_fp4_69[1]), "=f"(_unpack_fp4_69[2]), "=f"(_unpack_fp4_69[3]), "=f"(_unpack_fp4_69[4]), "=f"(_unpack_fp4_69[5]), "=f"(_unpack_fp4_69[6]), "=f"(_unpack_fp4_69[7]) : "r"(_fp4_wire_values_1[17 + 0]));
                    #endif
                }
                float2 _f2_608 = make_float2(acc_264[8], acc_264[9]);
                float2 a_279_1 = _f2_608;
                float2 _f2_609 = make_float2(_unpack_fp4_69[0], _unpack_fp4_69[1]);
                float2 q_280_1 = _f2_609;
                float2 r_281_1 = fma_f32x2_rn_noftz(q_280_1, scale0_265, a_279_1);
                acc_264[8] = r_281_1.x;
                acc_264[9] = r_281_1.y;
                float2 _f2_610 = make_float2(acc_264[10], acc_264[11]);
                float2 a_282_1 = _f2_610;
                float2 _f2_611 = make_float2(_unpack_fp4_69[2], _unpack_fp4_69[3]);
                float2 q_283_1 = _f2_611;
                float2 r_284_1 = fma_f32x2_rn_noftz(q_283_1, scale0_265, a_282_1);
                acc_264[10] = r_284_1.x;
                acc_264[11] = r_284_1.y;
                float2 _f2_612 = make_float2(acc_264[12], acc_264[13]);
                float2 a_285_1 = _f2_612;
                float2 _f2_613 = make_float2(_unpack_fp4_69[4], _unpack_fp4_69[5]);
                float2 q_286_1 = _f2_613;
                float2 r_287_1 = fma_f32x2_rn_noftz(q_286_1, scale0_265, a_285_1);
                acc_264[12] = r_287_1.x;
                acc_264[13] = r_287_1.y;
                float2 _f2_614 = make_float2(acc_264[14], acc_264[15]);
                float2 a_288 = _f2_614;
                float2 _f2_615 = make_float2(_unpack_fp4_69[6], _unpack_fp4_69[7]);
                float2 q_289 = _f2_615;
                float2 r_290 = fma_f32x2_rn_noftz(q_289, scale0_265, a_288);
                acc_264[14] = r_290.x;
                acc_264[15] = r_290.y;
                float _unpack_fp4_70[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[18 + 0]));
                    (&_unpack_fp4_70[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_70[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_70[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_70[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_70[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_70[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_70[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_70[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_70[0]), "=f"(_unpack_fp4_70[1]), "=f"(_unpack_fp4_70[2]), "=f"(_unpack_fp4_70[3]), "=f"(_unpack_fp4_70[4]), "=f"(_unpack_fp4_70[5]), "=f"(_unpack_fp4_70[6]), "=f"(_unpack_fp4_70[7]) : "r"(_fp4_wire_values_1[18 + 0]));
                    #endif
                }
                float2 _f2_616 = make_float2(acc_264[16], acc_264[17]);
                float2 a_291 = _f2_616;
                float2 _f2_617 = make_float2(_unpack_fp4_70[0], _unpack_fp4_70[1]);
                float2 q_292 = _f2_617;
                float2 r_293 = fma_f32x2_rn_noftz(q_292, scale1_266, a_291);
                acc_264[16] = r_293.x;
                acc_264[17] = r_293.y;
                float2 _f2_618 = make_float2(acc_264[18], acc_264[19]);
                float2 a_294 = _f2_618;
                float2 _f2_619 = make_float2(_unpack_fp4_70[2], _unpack_fp4_70[3]);
                float2 q_295 = _f2_619;
                float2 r_296 = fma_f32x2_rn_noftz(q_295, scale1_266, a_294);
                acc_264[18] = r_296.x;
                acc_264[19] = r_296.y;
                float2 _f2_620 = make_float2(acc_264[20], acc_264[21]);
                float2 a_297 = _f2_620;
                float2 _f2_621 = make_float2(_unpack_fp4_70[4], _unpack_fp4_70[5]);
                float2 q_298 = _f2_621;
                float2 r_299 = fma_f32x2_rn_noftz(q_298, scale1_266, a_297);
                acc_264[20] = r_299.x;
                acc_264[21] = r_299.y;
                float2 _f2_622 = make_float2(acc_264[22], acc_264[23]);
                float2 a_300 = _f2_622;
                float2 _f2_623 = make_float2(_unpack_fp4_70[6], _unpack_fp4_70[7]);
                float2 q_301 = _f2_623;
                float2 r_302 = fma_f32x2_rn_noftz(q_301, scale1_266, a_300);
                acc_264[22] = r_302.x;
                acc_264[23] = r_302.y;
                float _unpack_fp4_71[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[19 + 0]));
                    (&_unpack_fp4_71[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_71[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_71[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_71[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_71[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_71[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_71[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_71[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_71[0]), "=f"(_unpack_fp4_71[1]), "=f"(_unpack_fp4_71[2]), "=f"(_unpack_fp4_71[3]), "=f"(_unpack_fp4_71[4]), "=f"(_unpack_fp4_71[5]), "=f"(_unpack_fp4_71[6]), "=f"(_unpack_fp4_71[7]) : "r"(_fp4_wire_values_1[19 + 0]));
                    #endif
                }
                float2 _f2_624 = make_float2(acc_264[24], acc_264[25]);
                float2 a_303 = _f2_624;
                float2 _f2_625 = make_float2(_unpack_fp4_71[0], _unpack_fp4_71[1]);
                float2 q_304 = _f2_625;
                float2 r_305 = fma_f32x2_rn_noftz(q_304, scale1_266, a_303);
                acc_264[24] = r_305.x;
                acc_264[25] = r_305.y;
                float2 _f2_626 = make_float2(acc_264[26], acc_264[27]);
                float2 a_306 = _f2_626;
                float2 _f2_627 = make_float2(_unpack_fp4_71[2], _unpack_fp4_71[3]);
                float2 q_307 = _f2_627;
                float2 r_308 = fma_f32x2_rn_noftz(q_307, scale1_266, a_306);
                acc_264[26] = r_308.x;
                acc_264[27] = r_308.y;
                float2 _f2_628 = make_float2(acc_264[28], acc_264[29]);
                float2 a_309 = _f2_628;
                float2 _f2_629 = make_float2(_unpack_fp4_71[4], _unpack_fp4_71[5]);
                float2 q_310 = _f2_629;
                float2 r_311 = fma_f32x2_rn_noftz(q_310, scale1_266, a_309);
                acc_264[28] = r_311.x;
                acc_264[29] = r_311.y;
                float2 _f2_630 = make_float2(acc_264[30], acc_264[31]);
                float2 a_312 = _f2_630;
                float2 _f2_631 = make_float2(_unpack_fp4_71[6], _unpack_fp4_71[7]);
                float2 q_313 = _f2_631;
                float2 r_314 = fma_f32x2_rn_noftz(q_313, scale1_266, a_312);
                acc_264[30] = r_314.x;
                acc_264[31] = r_314.y;
                unsigned int acc_264_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_264[_lp*2 + 0], acc_264[_lp*2+1 + 0]));
                    acc_264_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_264_bf16[0]), "r"(acc_264_bf16[1]), "r"(acc_264_bf16[2]), "r"(acc_264_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_264_bf16[4]), "r"(acc_264_bf16[5]), "r"(acc_264_bf16[6]), "r"(acc_264_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_264_bf16[8]), "r"(acc_264_bf16[9]), "r"(acc_264_bf16[10]), "r"(acc_264_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_264_bf16[12]), "r"(acc_264_bf16[13]), "r"(acc_264_bf16[14]), "r"(acc_264_bf16[15]) : "memory");
                __syncwarp();
                int c_315 = lane_1;
                int l_316 = c_315 >> 2;
                unsigned int w4_317[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_317[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_317[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_317[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_317[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_316 + (c_315 & 3 ^ l_316 >> 1 & 3)))));
                long long idx_318 = crow64 * 3072 + (long long)(4 * (512 + c_315));
                {
                    int4 _iv4 = make_int4(w4_317[0 + 0], w4_317[0 + 1], w4_317[0 + 2], w4_317[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_318) = _iv4;
                }
                int c_319 = lane_1 + 32;
                int l_320 = c_319 >> 2;
                unsigned int w4_321[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_321[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_321[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_321[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_321[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_320 + (c_319 & 3 ^ l_320 >> 1 & 3)))));
                long long idx_322 = crow64 * 3072 + (long long)(4 * (512 + c_319));
                {
                    int4 _iv4 = make_int4(w4_321[0 + 0], w4_321[0 + 1], w4_321[0 + 2], w4_321[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_322) = _iv4;
                }
                int c_323 = lane_1 + 64;
                int l_324 = c_323 >> 2;
                unsigned int w4_325[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_325[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_325[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_325[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_325[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_324 + (c_323 & 3 ^ l_324 >> 1 & 3)))));
                long long idx_326 = crow64 * 3072 + (long long)(4 * (512 + c_323));
                {
                    int4 _iv4 = make_int4(w4_325[0 + 0], w4_325[0 + 1], w4_325[0 + 2], w4_325[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_326) = _iv4;
                }
                int c_327 = lane_1 + 96;
                int l_328 = c_327 >> 2;
                unsigned int w4_329[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_329[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_329[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_329[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_329[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_328 + (c_327 & 3 ^ l_328 >> 1 & 3)))));
                long long idx_330 = crow64 * 3072 + (long long)(4 * (512 + c_327));
                {
                    int4 _iv4 = make_int4(w4_329[0 + 0], w4_329[0 + 1], w4_329[0 + 2], w4_329[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_330) = _iv4;
                }
                __syncwarp();
                float acc_331[32] = {0};
                uint32_t _e4m3x2_to_f16x2_19;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_19) : "h"((uint16_t)(_fp4_wire_scales_1[5])));
                float _cvt_f32_f16_38;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_38) : "h"((uint16_t)(_e4m3x2_to_f16x2_19)));
                float _cvt_f32_f16_39;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_39) : "h"((uint16_t)(_e4m3x2_to_f16x2_19 >> 16)));
                float _mul_38;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(_cvt_f32_f16_38), "f"(rs_1));
                float _mul_39;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(_cvt_f32_f16_39), "f"(rs_1));
                float2 _f2_632 = make_float2(_mul_38, _mul_38);
                float2 scale0_332 = _f2_632;
                float2 _f2_633 = make_float2(_mul_39, _mul_39);
                float2 scale1_333 = _f2_633;
                float _unpack_fp4_72[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[20 + 0]));
                    (&_unpack_fp4_72[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_72[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_72[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_72[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_72[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_72[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_72[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_72[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_72[0]), "=f"(_unpack_fp4_72[1]), "=f"(_unpack_fp4_72[2]), "=f"(_unpack_fp4_72[3]), "=f"(_unpack_fp4_72[4]), "=f"(_unpack_fp4_72[5]), "=f"(_unpack_fp4_72[6]), "=f"(_unpack_fp4_72[7]) : "r"(_fp4_wire_values_1[20 + 0]));
                    #endif
                }
                float2 _f2_634 = make_float2(acc_331[0], acc_331[1]);
                float2 a_334_1 = _f2_634;
                float2 _f2_635 = make_float2(_unpack_fp4_72[0], _unpack_fp4_72[1]);
                float2 q_335_1 = _f2_635;
                float2 r_336_1 = fma_f32x2_rn_noftz(q_335_1, scale0_332, a_334_1);
                acc_331[0] = r_336_1.x;
                acc_331[1] = r_336_1.y;
                float2 _f2_636 = make_float2(acc_331[2], acc_331[3]);
                float2 a_337_1 = _f2_636;
                float2 _f2_637 = make_float2(_unpack_fp4_72[2], _unpack_fp4_72[3]);
                float2 q_338_1 = _f2_637;
                float2 r_339_1 = fma_f32x2_rn_noftz(q_338_1, scale0_332, a_337_1);
                acc_331[2] = r_339_1.x;
                acc_331[3] = r_339_1.y;
                float2 _f2_638 = make_float2(acc_331[4], acc_331[5]);
                float2 a_340_1 = _f2_638;
                float2 _f2_639 = make_float2(_unpack_fp4_72[4], _unpack_fp4_72[5]);
                float2 q_341_1 = _f2_639;
                float2 r_342_1 = fma_f32x2_rn_noftz(q_341_1, scale0_332, a_340_1);
                acc_331[4] = r_342_1.x;
                acc_331[5] = r_342_1.y;
                float2 _f2_640 = make_float2(acc_331[6], acc_331[7]);
                float2 a_343_1 = _f2_640;
                float2 _f2_641 = make_float2(_unpack_fp4_72[6], _unpack_fp4_72[7]);
                float2 q_344_1 = _f2_641;
                float2 r_345_1 = fma_f32x2_rn_noftz(q_344_1, scale0_332, a_343_1);
                acc_331[6] = r_345_1.x;
                acc_331[7] = r_345_1.y;
                float _unpack_fp4_73[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[21 + 0]));
                    (&_unpack_fp4_73[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_73[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_73[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_73[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_73[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_73[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_73[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_73[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_73[0]), "=f"(_unpack_fp4_73[1]), "=f"(_unpack_fp4_73[2]), "=f"(_unpack_fp4_73[3]), "=f"(_unpack_fp4_73[4]), "=f"(_unpack_fp4_73[5]), "=f"(_unpack_fp4_73[6]), "=f"(_unpack_fp4_73[7]) : "r"(_fp4_wire_values_1[21 + 0]));
                    #endif
                }
                float2 _f2_642 = make_float2(acc_331[8], acc_331[9]);
                float2 a_346 = _f2_642;
                float2 _f2_643 = make_float2(_unpack_fp4_73[0], _unpack_fp4_73[1]);
                float2 q_347 = _f2_643;
                float2 r_348 = fma_f32x2_rn_noftz(q_347, scale0_332, a_346);
                acc_331[8] = r_348.x;
                acc_331[9] = r_348.y;
                float2 _f2_644 = make_float2(acc_331[10], acc_331[11]);
                float2 a_349 = _f2_644;
                float2 _f2_645 = make_float2(_unpack_fp4_73[2], _unpack_fp4_73[3]);
                float2 q_350 = _f2_645;
                float2 r_351 = fma_f32x2_rn_noftz(q_350, scale0_332, a_349);
                acc_331[10] = r_351.x;
                acc_331[11] = r_351.y;
                float2 _f2_646 = make_float2(acc_331[12], acc_331[13]);
                float2 a_352 = _f2_646;
                float2 _f2_647 = make_float2(_unpack_fp4_73[4], _unpack_fp4_73[5]);
                float2 q_353 = _f2_647;
                float2 r_354 = fma_f32x2_rn_noftz(q_353, scale0_332, a_352);
                acc_331[12] = r_354.x;
                acc_331[13] = r_354.y;
                float2 _f2_648 = make_float2(acc_331[14], acc_331[15]);
                float2 a_355 = _f2_648;
                float2 _f2_649 = make_float2(_unpack_fp4_73[6], _unpack_fp4_73[7]);
                float2 q_356 = _f2_649;
                float2 r_357 = fma_f32x2_rn_noftz(q_356, scale0_332, a_355);
                acc_331[14] = r_357.x;
                acc_331[15] = r_357.y;
                float _unpack_fp4_74[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[22 + 0]));
                    (&_unpack_fp4_74[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_74[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_74[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_74[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_74[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_74[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_74[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_74[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_74[0]), "=f"(_unpack_fp4_74[1]), "=f"(_unpack_fp4_74[2]), "=f"(_unpack_fp4_74[3]), "=f"(_unpack_fp4_74[4]), "=f"(_unpack_fp4_74[5]), "=f"(_unpack_fp4_74[6]), "=f"(_unpack_fp4_74[7]) : "r"(_fp4_wire_values_1[22 + 0]));
                    #endif
                }
                float2 _f2_650 = make_float2(acc_331[16], acc_331[17]);
                float2 a_358 = _f2_650;
                float2 _f2_651 = make_float2(_unpack_fp4_74[0], _unpack_fp4_74[1]);
                float2 q_359 = _f2_651;
                float2 r_360 = fma_f32x2_rn_noftz(q_359, scale1_333, a_358);
                acc_331[16] = r_360.x;
                acc_331[17] = r_360.y;
                float2 _f2_652 = make_float2(acc_331[18], acc_331[19]);
                float2 a_361 = _f2_652;
                float2 _f2_653 = make_float2(_unpack_fp4_74[2], _unpack_fp4_74[3]);
                float2 q_362 = _f2_653;
                float2 r_363 = fma_f32x2_rn_noftz(q_362, scale1_333, a_361);
                acc_331[18] = r_363.x;
                acc_331[19] = r_363.y;
                float2 _f2_654 = make_float2(acc_331[20], acc_331[21]);
                float2 a_364 = _f2_654;
                float2 _f2_655 = make_float2(_unpack_fp4_74[4], _unpack_fp4_74[5]);
                float2 q_365 = _f2_655;
                float2 r_366 = fma_f32x2_rn_noftz(q_365, scale1_333, a_364);
                acc_331[20] = r_366.x;
                acc_331[21] = r_366.y;
                float2 _f2_656 = make_float2(acc_331[22], acc_331[23]);
                float2 a_367 = _f2_656;
                float2 _f2_657 = make_float2(_unpack_fp4_74[6], _unpack_fp4_74[7]);
                float2 q_368 = _f2_657;
                float2 r_369 = fma_f32x2_rn_noftz(q_368, scale1_333, a_367);
                acc_331[22] = r_369.x;
                acc_331[23] = r_369.y;
                float _unpack_fp4_75[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[23 + 0]));
                    (&_unpack_fp4_75[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_75[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_75[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_75[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_75[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_75[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_75[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_75[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_75[0]), "=f"(_unpack_fp4_75[1]), "=f"(_unpack_fp4_75[2]), "=f"(_unpack_fp4_75[3]), "=f"(_unpack_fp4_75[4]), "=f"(_unpack_fp4_75[5]), "=f"(_unpack_fp4_75[6]), "=f"(_unpack_fp4_75[7]) : "r"(_fp4_wire_values_1[23 + 0]));
                    #endif
                }
                float2 _f2_658 = make_float2(acc_331[24], acc_331[25]);
                float2 a_370 = _f2_658;
                float2 _f2_659 = make_float2(_unpack_fp4_75[0], _unpack_fp4_75[1]);
                float2 q_371 = _f2_659;
                float2 r_372 = fma_f32x2_rn_noftz(q_371, scale1_333, a_370);
                acc_331[24] = r_372.x;
                acc_331[25] = r_372.y;
                float2 _f2_660 = make_float2(acc_331[26], acc_331[27]);
                float2 a_373 = _f2_660;
                float2 _f2_661 = make_float2(_unpack_fp4_75[2], _unpack_fp4_75[3]);
                float2 q_374 = _f2_661;
                float2 r_375 = fma_f32x2_rn_noftz(q_374, scale1_333, a_373);
                acc_331[26] = r_375.x;
                acc_331[27] = r_375.y;
                float2 _f2_662 = make_float2(acc_331[28], acc_331[29]);
                float2 a_376 = _f2_662;
                float2 _f2_663 = make_float2(_unpack_fp4_75[4], _unpack_fp4_75[5]);
                float2 q_377 = _f2_663;
                float2 r_378 = fma_f32x2_rn_noftz(q_377, scale1_333, a_376);
                acc_331[28] = r_378.x;
                acc_331[29] = r_378.y;
                float2 _f2_664 = make_float2(acc_331[30], acc_331[31]);
                float2 a_379 = _f2_664;
                float2 _f2_665 = make_float2(_unpack_fp4_75[6], _unpack_fp4_75[7]);
                float2 q_380 = _f2_665;
                float2 r_381 = fma_f32x2_rn_noftz(q_380, scale1_333, a_379);
                acc_331[30] = r_381.x;
                acc_331[31] = r_381.y;
                unsigned int acc_331_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_331[_lp*2 + 0], acc_331[_lp*2+1 + 0]));
                    acc_331_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_331_bf16[0]), "r"(acc_331_bf16[1]), "r"(acc_331_bf16[2]), "r"(acc_331_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_331_bf16[4]), "r"(acc_331_bf16[5]), "r"(acc_331_bf16[6]), "r"(acc_331_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_331_bf16[8]), "r"(acc_331_bf16[9]), "r"(acc_331_bf16[10]), "r"(acc_331_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_331_bf16[12]), "r"(acc_331_bf16[13]), "r"(acc_331_bf16[14]), "r"(acc_331_bf16[15]) : "memory");
                __syncwarp();
                int c_382 = lane_1;
                int l_383 = c_382 >> 2;
                unsigned int w4_384[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_384[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_384[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_384[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_384[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_383 + (c_382 & 3 ^ l_383 >> 1 & 3)))));
                long long idx_385 = crow64 * 3072 + (long long)(4 * (640 + c_382));
                {
                    int4 _iv4 = make_int4(w4_384[0 + 0], w4_384[0 + 1], w4_384[0 + 2], w4_384[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_385) = _iv4;
                }
                int c_386 = lane_1 + 32;
                int l_387 = c_386 >> 2;
                unsigned int w4_388[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_388[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_388[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_388[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_388[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_387 + (c_386 & 3 ^ l_387 >> 1 & 3)))));
                long long idx_389 = crow64 * 3072 + (long long)(4 * (640 + c_386));
                {
                    int4 _iv4 = make_int4(w4_388[0 + 0], w4_388[0 + 1], w4_388[0 + 2], w4_388[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_389) = _iv4;
                }
                int c_390 = lane_1 + 64;
                int l_391 = c_390 >> 2;
                unsigned int w4_392[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_392[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_392[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_392[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_392[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_391 + (c_390 & 3 ^ l_391 >> 1 & 3)))));
                long long idx_393 = crow64 * 3072 + (long long)(4 * (640 + c_390));
                {
                    int4 _iv4 = make_int4(w4_392[0 + 0], w4_392[0 + 1], w4_392[0 + 2], w4_392[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_393) = _iv4;
                }
                int c_394 = lane_1 + 96;
                int l_395 = c_394 >> 2;
                unsigned int w4_396[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_396[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_396[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_396[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_396[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_395 + (c_394 & 3 ^ l_395 >> 1 & 3)))));
                long long idx_397 = crow64 * 3072 + (long long)(4 * (640 + c_394));
                {
                    int4 _iv4 = make_int4(w4_396[0 + 0], w4_396[0 + 1], w4_396[0 + 2], w4_396[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_397) = _iv4;
                }
                __syncwarp();
            }
            if (lane_1 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 6, 352;" ::: "memory");
            if (tid == 0) {
                __threadfence();
                unsigned int _atomic_old_0 = atomicAdd(&ctrl[1], 1);
                unsigned int last = (unsigned int)gridDim.x - 1;
                if (_atomic_old_0 == last) {
                    ctrl[1] = 0;
                    ctrl[2] = (unsigned int)rows;
                    ctrl[3] = 2;
                    __threadfence();
                    ctrl[0] = call + 1;
                    __threadfence();
                }
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
                long long v_base = rs_w_1;
                long long s_base = rs_w_1 + 4 * capo64_1 * 3072;
                long long m_base = rs_r_1;
                long long u = pro64 * 3072 / 16;
                long long lo = u * bid64_1 / g64_1 * 16;
                long long hi = u * (bid64_1 + 1) / g64_1 * 16;
                int npages = (int)((hi - lo + 4095) / 4096);
                #pragma unroll 1
                for (int i_2 = 0; i_2 < npages; i_2++) {
                    if ((i_2 & 31) == lane_0) {
                        long long o = lo + (long long)i_2 * 4096;
                        long long rem = hi - o;
                        unsigned int nb = 4096;
                        if (rem < 4096) {
                            nb = (unsigned int)rem;
                        }
                        {
                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(ts_wire + (v_base + o));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_0), "r"(pat_addr), "r"((uint32_t)(nb))
                                : "memory");
                        }
                    }
                }
                long long u_0 = pro64 * 384 / 16;
                long long lo_1 = u_0 * bid64_1 / g64_1 * 16;
                long long hi_2 = u_0 * (bid64_1 + 1) / g64_1 * 16;
                int npages_3 = (int)((hi_2 - lo_1 + 4095) / 4096);
                #pragma unroll 1
                for (int i_3 = 0; i_3 < npages_3; i_3++) {
                    if ((i_3 & 31) == lane_0) {
                        long long o_1 = lo_1 + (long long)i_3 * 4096;
                        long long rem_1 = hi_2 - o_1;
                        unsigned int nb_1 = 4096;
                        if (rem_1 < 4096) {
                            nb_1 = (unsigned int)rem_1;
                        }
                        {
                            void* _cpbulk_dst_1 = reinterpret_cast<void*>(ts_wire + (s_base + o_1));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_1), "r"(pat_addr + 4096), "r"((uint32_t)(nb_1))
                                : "memory");
                        }
                    }
                }
                long long u_4 = pro64 * 16 / 16;
                long long lo_5 = u_4 * bid64_1 / g64_1 * 16;
                long long hi_6 = u_4 * (bid64_1 + 1) / g64_1 * 16;
                int npages_7 = (int)((hi_6 - lo_5 + 4095) / 4096);
                #pragma unroll 1
                for (int i_4 = 0; i_4 < npages_7; i_4++) {
                    if ((i_4 & 31) == lane_0) {
                        long long o_2 = lo_5 + (long long)i_4 * 4096;
                        long long rem_2 = hi_6 - o_2;
                        unsigned int nb_2 = 4096;
                        if (rem_2 < 4096) {
                            nb_2 = (unsigned int)rem_2;
                        }
                        {
                            void* _cpbulk_dst_2 = reinterpret_cast<void*>(ts_rs + (m_base + o_2));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_2), "r"(pat_addr + 8192), "r"((uint32_t)(nb_2))
                                : "memory");
                        }
                    }
                }
                long long v_base_8 = rs_w_1 + capo64_1 * 3072;
                long long s_base_9 = rs_w_1 + 4 * capo64_1 * 3072 + capo64_1 * 384;
                long long m_base_10 = rs_r_1 + capo64_1 * 16;
                long long u_11 = pro64 * 3072 / 16;
                long long lo_12 = u_11 * bid64_1 / g64_1 * 16;
                long long hi_13 = u_11 * (bid64_1 + 1) / g64_1 * 16;
                int npages_14 = (int)((hi_13 - lo_12 + 4095) / 4096);
                #pragma unroll 1
                for (int i_5 = 0; i_5 < npages_14; i_5++) {
                    if ((i_5 & 31) == lane_0) {
                        long long o_3 = lo_12 + (long long)i_5 * 4096;
                        long long rem_3 = hi_13 - o_3;
                        unsigned int nb_3 = 4096;
                        if (rem_3 < 4096) {
                            nb_3 = (unsigned int)rem_3;
                        }
                        {
                            void* _cpbulk_dst_3 = reinterpret_cast<void*>(ts_wire + (v_base_8 + o_3));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_3), "r"(pat_addr), "r"((uint32_t)(nb_3))
                                : "memory");
                        }
                    }
                }
                long long u_15 = pro64 * 384 / 16;
                long long lo_16 = u_15 * bid64_1 / g64_1 * 16;
                long long hi_17 = u_15 * (bid64_1 + 1) / g64_1 * 16;
                int npages_18 = (int)((hi_17 - lo_16 + 4095) / 4096);
                #pragma unroll 1
                for (int i_6 = 0; i_6 < npages_18; i_6++) {
                    if ((i_6 & 31) == lane_0) {
                        long long o_4 = lo_16 + (long long)i_6 * 4096;
                        long long rem_4 = hi_17 - o_4;
                        unsigned int nb_4 = 4096;
                        if (rem_4 < 4096) {
                            nb_4 = (unsigned int)rem_4;
                        }
                        {
                            void* _cpbulk_dst_4 = reinterpret_cast<void*>(ts_wire + (s_base_9 + o_4));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_4), "r"(pat_addr + 4096), "r"((uint32_t)(nb_4))
                                : "memory");
                        }
                    }
                }
                long long u_19 = pro64 * 16 / 16;
                long long lo_20 = u_19 * bid64_1 / g64_1 * 16;
                long long hi_21 = u_19 * (bid64_1 + 1) / g64_1 * 16;
                int npages_22 = (int)((hi_21 - lo_20 + 4095) / 4096);
                #pragma unroll 1
                for (int i_7 = 0; i_7 < npages_22; i_7++) {
                    if ((i_7 & 31) == lane_0) {
                        long long o_5 = lo_20 + (long long)i_7 * 4096;
                        long long rem_5 = hi_21 - o_5;
                        unsigned int nb_5 = 4096;
                        if (rem_5 < 4096) {
                            nb_5 = (unsigned int)rem_5;
                        }
                        {
                            void* _cpbulk_dst_5 = reinterpret_cast<void*>(ts_rs + (m_base_10 + o_5));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_5), "r"(pat_addr + 8192), "r"((uint32_t)(nb_5))
                                : "memory");
                        }
                    }
                }
                long long v_base_23 = rs_w_1 + 2 * capo64_1 * 3072;
                long long s_base_24 = rs_w_1 + 4 * capo64_1 * 3072 + 2 * capo64_1 * 384;
                long long m_base_25 = rs_r_1 + 2 * capo64_1 * 16;
                long long u_26 = pro64 * 3072 / 16;
                long long lo_27 = u_26 * bid64_1 / g64_1 * 16;
                long long hi_28 = u_26 * (bid64_1 + 1) / g64_1 * 16;
                int npages_29 = (int)((hi_28 - lo_27 + 4095) / 4096);
                #pragma unroll 1
                for (int i_8 = 0; i_8 < npages_29; i_8++) {
                    if ((i_8 & 31) == lane_0) {
                        long long o_6 = lo_27 + (long long)i_8 * 4096;
                        long long rem_6 = hi_28 - o_6;
                        unsigned int nb_6 = 4096;
                        if (rem_6 < 4096) {
                            nb_6 = (unsigned int)rem_6;
                        }
                        {
                            void* _cpbulk_dst_6 = reinterpret_cast<void*>(ts_wire + (v_base_23 + o_6));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_6), "r"(pat_addr), "r"((uint32_t)(nb_6))
                                : "memory");
                        }
                    }
                }
                long long u_30 = pro64 * 384 / 16;
                long long lo_31 = u_30 * bid64_1 / g64_1 * 16;
                long long hi_32 = u_30 * (bid64_1 + 1) / g64_1 * 16;
                int npages_33 = (int)((hi_32 - lo_31 + 4095) / 4096);
                #pragma unroll 1
                for (int i_9 = 0; i_9 < npages_33; i_9++) {
                    if ((i_9 & 31) == lane_0) {
                        long long o_7 = lo_31 + (long long)i_9 * 4096;
                        long long rem_7 = hi_32 - o_7;
                        unsigned int nb_7 = 4096;
                        if (rem_7 < 4096) {
                            nb_7 = (unsigned int)rem_7;
                        }
                        {
                            void* _cpbulk_dst_7 = reinterpret_cast<void*>(ts_wire + (s_base_24 + o_7));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_7), "r"(pat_addr + 4096), "r"((uint32_t)(nb_7))
                                : "memory");
                        }
                    }
                }
                long long u_34 = pro64 * 16 / 16;
                long long lo_35 = u_34 * bid64_1 / g64_1 * 16;
                long long hi_36 = u_34 * (bid64_1 + 1) / g64_1 * 16;
                int npages_37 = (int)((hi_36 - lo_35 + 4095) / 4096);
                #pragma unroll 1
                for (int i_10 = 0; i_10 < npages_37; i_10++) {
                    if ((i_10 & 31) == lane_0) {
                        long long o_8 = lo_35 + (long long)i_10 * 4096;
                        long long rem_8 = hi_36 - o_8;
                        unsigned int nb_8 = 4096;
                        if (rem_8 < 4096) {
                            nb_8 = (unsigned int)rem_8;
                        }
                        {
                            void* _cpbulk_dst_8 = reinterpret_cast<void*>(ts_rs + (m_base_25 + o_8));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_8), "r"(pat_addr + 8192), "r"((uint32_t)(nb_8))
                                : "memory");
                        }
                    }
                }
                long long v_base_38 = rs_w_1 + 3 * capo64_1 * 3072;
                long long s_base_39 = rs_w_1 + 4 * capo64_1 * 3072 + 3 * capo64_1 * 384;
                long long m_base_40 = rs_r_1 + 3 * capo64_1 * 16;
                long long u_41 = pro64 * 3072 / 16;
                long long lo_42 = u_41 * bid64_1 / g64_1 * 16;
                long long hi_43 = u_41 * (bid64_1 + 1) / g64_1 * 16;
                int npages_44 = (int)((hi_43 - lo_42 + 4095) / 4096);
                #pragma unroll 1
                for (int i_11 = 0; i_11 < npages_44; i_11++) {
                    if ((i_11 & 31) == lane_0) {
                        long long o_9 = lo_42 + (long long)i_11 * 4096;
                        long long rem_9 = hi_43 - o_9;
                        unsigned int nb_9 = 4096;
                        if (rem_9 < 4096) {
                            nb_9 = (unsigned int)rem_9;
                        }
                        {
                            void* _cpbulk_dst_9 = reinterpret_cast<void*>(ts_wire + (v_base_38 + o_9));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_9), "r"(pat_addr), "r"((uint32_t)(nb_9))
                                : "memory");
                        }
                    }
                }
                long long u_45 = pro64 * 384 / 16;
                long long lo_46 = u_45 * bid64_1 / g64_1 * 16;
                long long hi_47 = u_45 * (bid64_1 + 1) / g64_1 * 16;
                int npages_48 = (int)((hi_47 - lo_46 + 4095) / 4096);
                #pragma unroll 1
                for (int i_12 = 0; i_12 < npages_48; i_12++) {
                    if ((i_12 & 31) == lane_0) {
                        long long o_10 = lo_46 + (long long)i_12 * 4096;
                        long long rem_10 = hi_47 - o_10;
                        unsigned int nb_10 = 4096;
                        if (rem_10 < 4096) {
                            nb_10 = (unsigned int)rem_10;
                        }
                        {
                            void* _cpbulk_dst_10 = reinterpret_cast<void*>(ts_wire + (s_base_39 + o_10));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_10), "r"(pat_addr + 4096), "r"((uint32_t)(nb_10))
                                : "memory");
                        }
                    }
                }
                long long u_49 = pro64 * 16 / 16;
                long long lo_50 = u_49 * bid64_1 / g64_1 * 16;
                long long hi_51 = u_49 * (bid64_1 + 1) / g64_1 * 16;
                int npages_52 = (int)((hi_51 - lo_50 + 4095) / 4096);
                #pragma unroll 1
                for (int i_13 = 0; i_13 < npages_52; i_13++) {
                    if ((i_13 & 31) == lane_0) {
                        long long o_11 = lo_50 + (long long)i_13 * 4096;
                        long long rem_11 = hi_51 - o_11;
                        unsigned int nb_11 = 4096;
                        if (rem_11 < 4096) {
                            nb_11 = (unsigned int)rem_11;
                        }
                        {
                            void* _cpbulk_dst_11 = reinterpret_cast<void*>(ts_rs + (m_base_40 + o_11));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_11), "r"(pat_addr + 8192), "r"((uint32_t)(nb_11))
                                : "memory");
                        }
                    }
                }
                if (0 != rank) {
                    long long v_base2 = rs_w_1 + sa_w_1;
                    long long s_base2 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072;
                    long long m_base2 = rs_r_1 + sa_r_1;
                    long long u_1 = pro64 * 3072 / 16;
                    long long lo_2 = u_1 * bid64_1 / g64_1 * 16;
                    long long hi_3 = u_1 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_4 = (int)((hi_3 - lo_2 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_14 = 0; i_14 < npages_4; i_14++) {
                        if ((i_14 & 31) == lane_0) {
                            long long o_12 = lo_2 + (long long)i_14 * 4096;
                            long long rem_12 = hi_3 - o_12;
                            unsigned int nb_12 = 4096;
                            if (rem_12 < 4096) {
                                nb_12 = (unsigned int)rem_12;
                            }
                            {
                                void* _cpbulk_dst_12 = reinterpret_cast<void*>(ts_wire + (v_base2 + o_12));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_12), "r"(pat_addr), "r"((uint32_t)(nb_12))
                                    : "memory");
                            }
                        }
                    }
                    long long u_5 = pro64 * 384 / 16;
                    long long lo_6 = u_5 * bid64_1 / g64_1 * 16;
                    long long hi_7 = u_5 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_8 = (int)((hi_7 - lo_6 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_15 = 0; i_15 < npages_8; i_15++) {
                        if ((i_15 & 31) == lane_0) {
                            long long o_13 = lo_6 + (long long)i_15 * 4096;
                            long long rem_13 = hi_7 - o_13;
                            unsigned int nb_13 = 4096;
                            if (rem_13 < 4096) {
                                nb_13 = (unsigned int)rem_13;
                            }
                            {
                                void* _cpbulk_dst_13 = reinterpret_cast<void*>(ts_wire + (s_base2 + o_13));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_13), "r"(pat_addr + 4096), "r"((uint32_t)(nb_13))
                                    : "memory");
                            }
                        }
                    }
                    long long u_9 = pro64 * 16 / 16;
                    long long lo_10 = u_9 * bid64_1 / g64_1 * 16;
                    long long hi_11 = u_9 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_12 = (int)((hi_11 - lo_10 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_16 = 0; i_16 < npages_12; i_16++) {
                        if ((i_16 & 31) == lane_0) {
                            long long o_14 = lo_10 + (long long)i_16 * 4096;
                            long long rem_14 = hi_11 - o_14;
                            unsigned int nb_14 = 4096;
                            if (rem_14 < 4096) {
                                nb_14 = (unsigned int)rem_14;
                            }
                            {
                                void* _cpbulk_dst_14 = reinterpret_cast<void*>(ts_rs + (m_base2 + o_14));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_14), "r"(pat_addr + 8192), "r"((uint32_t)(nb_14))
                                    : "memory");
                            }
                        }
                    }
                }
                if (1 != rank) {
                    long long v_base2_1 = rs_w_1 + sa_w_1 + capo64_1 * 3072;
                    long long s_base2_1 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + capo64_1 * 384;
                    long long m_base2_1 = rs_r_1 + sa_r_1 + capo64_1 * 16;
                    long long u_1_1 = pro64 * 3072 / 16;
                    long long lo_2_1 = u_1_1 * bid64_1 / g64_1 * 16;
                    long long hi_3_1 = u_1_1 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_4_1 = (int)((hi_3_1 - lo_2_1 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_17 = 0; i_17 < npages_4_1; i_17++) {
                        if ((i_17 & 31) == lane_0) {
                            long long o_15 = lo_2_1 + (long long)i_17 * 4096;
                            long long rem_15 = hi_3_1 - o_15;
                            unsigned int nb_15 = 4096;
                            if (rem_15 < 4096) {
                                nb_15 = (unsigned int)rem_15;
                            }
                            {
                                void* _cpbulk_dst_15 = reinterpret_cast<void*>(ts_wire + (v_base2_1 + o_15));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_15), "r"(pat_addr), "r"((uint32_t)(nb_15))
                                    : "memory");
                            }
                        }
                    }
                    long long u_5_1 = pro64 * 384 / 16;
                    long long lo_6_1 = u_5_1 * bid64_1 / g64_1 * 16;
                    long long hi_7_1 = u_5_1 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_8_1 = (int)((hi_7_1 - lo_6_1 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_18 = 0; i_18 < npages_8_1; i_18++) {
                        if ((i_18 & 31) == lane_0) {
                            long long o_16 = lo_6_1 + (long long)i_18 * 4096;
                            long long rem_16 = hi_7_1 - o_16;
                            unsigned int nb_16 = 4096;
                            if (rem_16 < 4096) {
                                nb_16 = (unsigned int)rem_16;
                            }
                            {
                                void* _cpbulk_dst_16 = reinterpret_cast<void*>(ts_wire + (s_base2_1 + o_16));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_16), "r"(pat_addr + 4096), "r"((uint32_t)(nb_16))
                                    : "memory");
                            }
                        }
                    }
                    long long u_9_1 = pro64 * 16 / 16;
                    long long lo_10_1 = u_9_1 * bid64_1 / g64_1 * 16;
                    long long hi_11_1 = u_9_1 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_12_1 = (int)((hi_11_1 - lo_10_1 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_19 = 0; i_19 < npages_12_1; i_19++) {
                        if ((i_19 & 31) == lane_0) {
                            long long o_17 = lo_10_1 + (long long)i_19 * 4096;
                            long long rem_17 = hi_11_1 - o_17;
                            unsigned int nb_17 = 4096;
                            if (rem_17 < 4096) {
                                nb_17 = (unsigned int)rem_17;
                            }
                            {
                                void* _cpbulk_dst_17 = reinterpret_cast<void*>(ts_rs + (m_base2_1 + o_17));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_17), "r"(pat_addr + 8192), "r"((uint32_t)(nb_17))
                                    : "memory");
                            }
                        }
                    }
                }
                if (2 != rank) {
                    long long v_base2_2 = rs_w_1 + sa_w_1 + 2 * capo64_1 * 3072;
                    long long s_base2_2 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + 2 * capo64_1 * 384;
                    long long m_base2_2 = rs_r_1 + sa_r_1 + 2 * capo64_1 * 16;
                    long long u_1_2 = pro64 * 3072 / 16;
                    long long lo_2_2 = u_1_2 * bid64_1 / g64_1 * 16;
                    long long hi_3_2 = u_1_2 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_4_2 = (int)((hi_3_2 - lo_2_2 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_20 = 0; i_20 < npages_4_2; i_20++) {
                        if ((i_20 & 31) == lane_0) {
                            long long o_18 = lo_2_2 + (long long)i_20 * 4096;
                            long long rem_18 = hi_3_2 - o_18;
                            unsigned int nb_18 = 4096;
                            if (rem_18 < 4096) {
                                nb_18 = (unsigned int)rem_18;
                            }
                            {
                                void* _cpbulk_dst_18 = reinterpret_cast<void*>(ts_wire + (v_base2_2 + o_18));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_18), "r"(pat_addr), "r"((uint32_t)(nb_18))
                                    : "memory");
                            }
                        }
                    }
                    long long u_5_2 = pro64 * 384 / 16;
                    long long lo_6_2 = u_5_2 * bid64_1 / g64_1 * 16;
                    long long hi_7_2 = u_5_2 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_8_2 = (int)((hi_7_2 - lo_6_2 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_21 = 0; i_21 < npages_8_2; i_21++) {
                        if ((i_21 & 31) == lane_0) {
                            long long o_19 = lo_6_2 + (long long)i_21 * 4096;
                            long long rem_19 = hi_7_2 - o_19;
                            unsigned int nb_19 = 4096;
                            if (rem_19 < 4096) {
                                nb_19 = (unsigned int)rem_19;
                            }
                            {
                                void* _cpbulk_dst_19 = reinterpret_cast<void*>(ts_wire + (s_base2_2 + o_19));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_19), "r"(pat_addr + 4096), "r"((uint32_t)(nb_19))
                                    : "memory");
                            }
                        }
                    }
                    long long u_9_2 = pro64 * 16 / 16;
                    long long lo_10_2 = u_9_2 * bid64_1 / g64_1 * 16;
                    long long hi_11_2 = u_9_2 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_12_2 = (int)((hi_11_2 - lo_10_2 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_22 = 0; i_22 < npages_12_2; i_22++) {
                        if ((i_22 & 31) == lane_0) {
                            long long o_20 = lo_10_2 + (long long)i_22 * 4096;
                            long long rem_20 = hi_11_2 - o_20;
                            unsigned int nb_20 = 4096;
                            if (rem_20 < 4096) {
                                nb_20 = (unsigned int)rem_20;
                            }
                            {
                                void* _cpbulk_dst_20 = reinterpret_cast<void*>(ts_rs + (m_base2_2 + o_20));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_20), "r"(pat_addr + 8192), "r"((uint32_t)(nb_20))
                                    : "memory");
                            }
                        }
                    }
                }
                if (3 != rank) {
                    long long v_base2_3 = rs_w_1 + sa_w_1 + 3 * capo64_1 * 3072;
                    long long s_base2_3 = rs_w_1 + sa_w_1 + 4 * capo64_1 * 3072 + 3 * capo64_1 * 384;
                    long long m_base2_3 = rs_r_1 + sa_r_1 + 3 * capo64_1 * 16;
                    long long u_1_3 = pro64 * 3072 / 16;
                    long long lo_2_3 = u_1_3 * bid64_1 / g64_1 * 16;
                    long long hi_3_3 = u_1_3 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_4_3 = (int)((hi_3_3 - lo_2_3 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_23 = 0; i_23 < npages_4_3; i_23++) {
                        if ((i_23 & 31) == lane_0) {
                            long long o_21 = lo_2_3 + (long long)i_23 * 4096;
                            long long rem_21 = hi_3_3 - o_21;
                            unsigned int nb_21 = 4096;
                            if (rem_21 < 4096) {
                                nb_21 = (unsigned int)rem_21;
                            }
                            {
                                void* _cpbulk_dst_21 = reinterpret_cast<void*>(ts_wire + (v_base2_3 + o_21));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_21), "r"(pat_addr), "r"((uint32_t)(nb_21))
                                    : "memory");
                            }
                        }
                    }
                    long long u_5_3 = pro64 * 384 / 16;
                    long long lo_6_3 = u_5_3 * bid64_1 / g64_1 * 16;
                    long long hi_7_3 = u_5_3 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_8_3 = (int)((hi_7_3 - lo_6_3 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_24 = 0; i_24 < npages_8_3; i_24++) {
                        if ((i_24 & 31) == lane_0) {
                            long long o_22 = lo_6_3 + (long long)i_24 * 4096;
                            long long rem_22 = hi_7_3 - o_22;
                            unsigned int nb_22 = 4096;
                            if (rem_22 < 4096) {
                                nb_22 = (unsigned int)rem_22;
                            }
                            {
                                void* _cpbulk_dst_22 = reinterpret_cast<void*>(ts_wire + (s_base2_3 + o_22));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_22), "r"(pat_addr + 4096), "r"((uint32_t)(nb_22))
                                    : "memory");
                            }
                        }
                    }
                    long long u_9_3 = pro64 * 16 / 16;
                    long long lo_10_3 = u_9_3 * bid64_1 / g64_1 * 16;
                    long long hi_11_3 = u_9_3 * (bid64_1 + 1) / g64_1 * 16;
                    int npages_12_3 = (int)((hi_11_3 - lo_10_3 + 4095) / 4096);
                    #pragma unroll 1
                    for (int i_25 = 0; i_25 < npages_12_3; i_25++) {
                        if ((i_25 & 31) == lane_0) {
                            long long o_23 = lo_10_3 + (long long)i_25 * 4096;
                            long long rem_23 = hi_11_3 - o_23;
                            unsigned int nb_23 = 4096;
                            if (rem_23 < 4096) {
                                nb_23 = (unsigned int)rem_23;
                            }
                            {
                                void* _cpbulk_dst_23 = reinterpret_cast<void*>(ts_rs + (m_base2_3 + o_23));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_23), "r"(pat_addr + 8192), "r"((uint32_t)(nb_23))
                                    : "memory");
                            }
                        }
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 6, 352;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
