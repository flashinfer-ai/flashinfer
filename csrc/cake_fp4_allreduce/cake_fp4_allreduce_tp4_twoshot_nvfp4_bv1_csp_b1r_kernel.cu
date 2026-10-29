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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(352, LAUNCH_MIN_BLOCKS) void
cake_fp4_allreduce_tp4_twoshot_nvfp4_bv1_csp_b1r(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ ts_wire, uint8_t* __restrict__ ts_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
                int v = rs_wo_1 + ix_1 * 3072;
                int sc = rs_wo_1 + 4 * capo_1 * 3072 + ix_1 * 384;
                int v_0 = rs_wo_1 + (capo_1 + ix_1) * 3072;
                int sc_1 = rs_wo_1 + 4 * capo_1 * 3072 + (capo_1 + ix_1) * 384;
                int v_2 = rs_wo_1 + (2 * capo_1 + ix_1) * 3072;
                int sc_3 = rs_wo_1 + 4 * capo_1 * 3072 + (2 * capo_1 + ix_1) * 384;
                int v_4 = rs_wo_1 + (3 * capo_1 + ix_1) * 3072;
                int sc_5 = rs_wo_1 + 4 * capo_1 * 3072 + (3 * capo_1 + ix_1) * 384;
                int pp = lane_1 + 32 * (3 * h);
                int pp_6 = lane_1 + 32 * (3 * h + 1);
                int pp_7 = lane_1 + 32 * (3 * h + 2);
                uint32_t _fp4_wire_values_0[48];
                uint32_t _fp4_wire_scales_0[12];
                uint16_t _wire_h16_2_0;
                uint16_t _wire_h16_2_1;
                uint16_t _wire_h16_2_2;
                uint16_t _wire_h16_2_3;
                uint16_t _wire_h16_2_4;
                uint16_t _wire_h16_2_5;
                uint16_t _wire_h16_2_6;
                uint16_t _wire_h16_2_7;
                uint16_t _wire_h16_2_8;
                uint16_t _wire_h16_2_9;
                uint16_t _wire_h16_2_10;
                uint16_t _wire_h16_2_11;
                uint32_t _wire_vpend_2 = 4095u;
                uint32_t _wire_spend_2 = 4095u;
                do {
                    if (_wire_vpend_2 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp)))) : "memory");
                    }
                    if (_wire_vpend_2 & 2u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp)))) : "memory");
                    }
                    if (_wire_vpend_2 & 4u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp)))) : "memory");
                    }
                    if (_wire_vpend_2 & 8u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[12]), "=r"(_fp4_wire_values_0[13]), "=r"(_fp4_wire_values_0[14]), "=r"(_fp4_wire_values_0[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp)))) : "memory");
                    }
                    if (_wire_vpend_2 & 16u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[16]), "=r"(_fp4_wire_values_0[17]), "=r"(_fp4_wire_values_0[18]), "=r"(_fp4_wire_values_0[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_6)))) : "memory");
                    }
                    if (_wire_vpend_2 & 32u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[20]), "=r"(_fp4_wire_values_0[21]), "=r"(_fp4_wire_values_0[22]), "=r"(_fp4_wire_values_0[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp_6)))) : "memory");
                    }
                    if (_wire_vpend_2 & 64u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[24]), "=r"(_fp4_wire_values_0[25]), "=r"(_fp4_wire_values_0[26]), "=r"(_fp4_wire_values_0[27]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp_6)))) : "memory");
                    }
                    if (_wire_vpend_2 & 128u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[28]), "=r"(_fp4_wire_values_0[29]), "=r"(_fp4_wire_values_0[30]), "=r"(_fp4_wire_values_0[31]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp_6)))) : "memory");
                    }
                    if (_wire_vpend_2 & 256u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[32]), "=r"(_fp4_wire_values_0[33]), "=r"(_fp4_wire_values_0[34]), "=r"(_fp4_wire_values_0[35]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_7)))) : "memory");
                    }
                    if (_wire_vpend_2 & 512u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[36]), "=r"(_fp4_wire_values_0[37]), "=r"(_fp4_wire_values_0[38]), "=r"(_fp4_wire_values_0[39]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_0 + (int)(16 * pp_7)))) : "memory");
                    }
                    if (_wire_vpend_2 & 1024u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[40]), "=r"(_fp4_wire_values_0[41]), "=r"(_fp4_wire_values_0[42]), "=r"(_fp4_wire_values_0[43]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_2 + (int)(16 * pp_7)))) : "memory");
                    }
                    if (_wire_vpend_2 & 2048u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[44]), "=r"(_fp4_wire_values_0[45]), "=r"(_fp4_wire_values_0[46]), "=r"(_fp4_wire_values_0[47]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_4 + (int)(16 * pp_7)))) : "memory");
                    }
                    if (_wire_spend_2 & 1u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp)))) : "memory");
                        _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_2_0;
                    }
                    if (_wire_spend_2 & 2u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp)))) : "memory");
                        _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_2_1;
                    }
                    if (_wire_spend_2 & 4u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp)))) : "memory");
                        _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_2_2;
                    }
                    if (_wire_spend_2 & 8u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_5 + (int)(2 * pp)))) : "memory");
                        _fp4_wire_scales_0[3] = (uint32_t)_wire_h16_2_3;
                    }
                    if (_wire_spend_2 & 16u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp_6)))) : "memory");
                        _fp4_wire_scales_0[4] = (uint32_t)_wire_h16_2_4;
                    }
                    if (_wire_spend_2 & 32u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp_6)))) : "memory");
                        _fp4_wire_scales_0[5] = (uint32_t)_wire_h16_2_5;
                    }
                    if (_wire_spend_2 & 64u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_6) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp_6)))) : "memory");
                        _fp4_wire_scales_0[6] = (uint32_t)_wire_h16_2_6;
                    }
                    if (_wire_spend_2 & 128u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_7) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_5 + (int)(2 * pp_6)))) : "memory");
                        _fp4_wire_scales_0[7] = (uint32_t)_wire_h16_2_7;
                    }
                    if (_wire_spend_2 & 256u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_8) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp_7)))) : "memory");
                        _fp4_wire_scales_0[8] = (uint32_t)_wire_h16_2_8;
                    }
                    if (_wire_spend_2 & 512u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_9) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp_7)))) : "memory");
                        _fp4_wire_scales_0[9] = (uint32_t)_wire_h16_2_9;
                    }
                    if (_wire_spend_2 & 1024u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_10) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_3 + (int)(2 * pp_7)))) : "memory");
                        _fp4_wire_scales_0[10] = (uint32_t)_wire_h16_2_10;
                    }
                    if (_wire_spend_2 & 2048u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_2_11) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_5 + (int)(2 * pp_7)))) : "memory");
                        _fp4_wire_scales_0[11] = (uint32_t)_wire_h16_2_11;
                    }
                    _wire_vpend_2 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((_fp4_wire_values_0[12] & ~((_fp4_wire_values_0[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[13] & ~((_fp4_wire_values_0[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[14] & ~((_fp4_wire_values_0[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[15] & ~((_fp4_wire_values_0[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((_fp4_wire_values_0[16] & ~((_fp4_wire_values_0[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[17] & ~((_fp4_wire_values_0[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[18] & ~((_fp4_wire_values_0[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[19] & ~((_fp4_wire_values_0[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((_fp4_wire_values_0[20] & ~((_fp4_wire_values_0[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[21] & ~((_fp4_wire_values_0[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[22] & ~((_fp4_wire_values_0[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[23] & ~((_fp4_wire_values_0[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5) | ((uint32_t)((((_fp4_wire_values_0[24] & ~((_fp4_wire_values_0[24] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[25] & ~((_fp4_wire_values_0[25] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[26] & ~((_fp4_wire_values_0[26] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[27] & ~((_fp4_wire_values_0[27] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 6) | ((uint32_t)((((_fp4_wire_values_0[28] & ~((_fp4_wire_values_0[28] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[29] & ~((_fp4_wire_values_0[29] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[30] & ~((_fp4_wire_values_0[30] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[31] & ~((_fp4_wire_values_0[31] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 7) | ((uint32_t)((((_fp4_wire_values_0[32] & ~((_fp4_wire_values_0[32] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[33] & ~((_fp4_wire_values_0[33] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[34] & ~((_fp4_wire_values_0[34] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[35] & ~((_fp4_wire_values_0[35] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 8) | ((uint32_t)((((_fp4_wire_values_0[36] & ~((_fp4_wire_values_0[36] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[37] & ~((_fp4_wire_values_0[37] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[38] & ~((_fp4_wire_values_0[38] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[39] & ~((_fp4_wire_values_0[39] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 9) | ((uint32_t)((((_fp4_wire_values_0[40] & ~((_fp4_wire_values_0[40] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[41] & ~((_fp4_wire_values_0[41] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[42] & ~((_fp4_wire_values_0[42] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[43] & ~((_fp4_wire_values_0[43] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 10) | ((uint32_t)((((_fp4_wire_values_0[44] & ~((_fp4_wire_values_0[44] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[45] & ~((_fp4_wire_values_0[45] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[46] & ~((_fp4_wire_values_0[46] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[47] & ~((_fp4_wire_values_0[47] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 11);
                    _wire_spend_2 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[3] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[4] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[5] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[6] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[6] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 6) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[7] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[7] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 7) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[8] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[8] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 8) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[9] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[9] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 9) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[10] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[10] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 10) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[11] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[11] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 11);
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
                float2 scale0 = make_float2(_mul_0, _mul_0);
                float2 scale1 = make_float2(_mul_1, _mul_1);
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
                float2 a = make_float2(acc[0], acc[1]);
                float2 q = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                float2 r = fma_f32x2_rn_noftz(q, scale0, a);
                acc[0] = r.x;
                acc[1] = r.y;
                float2 a_8 = make_float2(acc[2], acc[3]);
                float2 q_9 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                float2 r_10 = fma_f32x2_rn_noftz(q_9, scale0, a_8);
                acc[2] = r_10.x;
                acc[3] = r_10.y;
                float2 a_11 = make_float2(acc[4], acc[5]);
                float2 q_12 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                float2 r_13 = fma_f32x2_rn_noftz(q_12, scale0, a_11);
                acc[4] = r_13.x;
                acc[5] = r_13.y;
                float2 a_14 = make_float2(acc[6], acc[7]);
                float2 q_15 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                float2 r_16 = fma_f32x2_rn_noftz(q_15, scale0, a_14);
                acc[6] = r_16.x;
                acc[7] = r_16.y;
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
                float2 a_17 = make_float2(acc[8], acc[9]);
                float2 q_18 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                float2 r_19 = fma_f32x2_rn_noftz(q_18, scale0, a_17);
                acc[8] = r_19.x;
                acc[9] = r_19.y;
                float2 a_20 = make_float2(acc[10], acc[11]);
                float2 q_21 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                float2 r_22 = fma_f32x2_rn_noftz(q_21, scale0, a_20);
                acc[10] = r_22.x;
                acc[11] = r_22.y;
                float2 a_23 = make_float2(acc[12], acc[13]);
                float2 q_24 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                float2 r_25 = fma_f32x2_rn_noftz(q_24, scale0, a_23);
                acc[12] = r_25.x;
                acc[13] = r_25.y;
                float2 a_26 = make_float2(acc[14], acc[15]);
                float2 q_27 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                float2 r_28 = fma_f32x2_rn_noftz(q_27, scale0, a_26);
                acc[14] = r_28.x;
                acc[15] = r_28.y;
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
                float2 a_29 = make_float2(acc[16], acc[17]);
                float2 q_30 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                float2 r_31 = fma_f32x2_rn_noftz(q_30, scale1, a_29);
                acc[16] = r_31.x;
                acc[17] = r_31.y;
                float2 a_32 = make_float2(acc[18], acc[19]);
                float2 q_33 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                float2 r_34 = fma_f32x2_rn_noftz(q_33, scale1, a_32);
                acc[18] = r_34.x;
                acc[19] = r_34.y;
                float2 a_35 = make_float2(acc[20], acc[21]);
                float2 q_36 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                float2 r_37 = fma_f32x2_rn_noftz(q_36, scale1, a_35);
                acc[20] = r_37.x;
                acc[21] = r_37.y;
                float2 a_38 = make_float2(acc[22], acc[23]);
                float2 q_39 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                float2 r_40 = fma_f32x2_rn_noftz(q_39, scale1, a_38);
                acc[22] = r_40.x;
                acc[23] = r_40.y;
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
                float2 a_41 = make_float2(acc[24], acc[25]);
                float2 q_42 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                float2 r_43 = fma_f32x2_rn_noftz(q_42, scale1, a_41);
                acc[24] = r_43.x;
                acc[25] = r_43.y;
                float2 a_44 = make_float2(acc[26], acc[27]);
                float2 q_45 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                float2 r_46 = fma_f32x2_rn_noftz(q_45, scale1, a_44);
                acc[26] = r_46.x;
                acc[27] = r_46.y;
                float2 a_47 = make_float2(acc[28], acc[29]);
                float2 q_48 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                float2 r_49 = fma_f32x2_rn_noftz(q_48, scale1, a_47);
                acc[28] = r_49.x;
                acc[29] = r_49.y;
                float2 a_50 = make_float2(acc[30], acc[31]);
                float2 q_51 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                float2 r_52 = fma_f32x2_rn_noftz(q_51, scale1, a_50);
                acc[30] = r_52.x;
                acc[31] = r_52.y;
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
                float2 scale0_53 = make_float2(_mul_2, _mul_2);
                float2 scale1_54 = make_float2(_mul_3, _mul_3);
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
                float2 a_55 = make_float2(acc[0], acc[1]);
                float2 q_56 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                float2 r_57 = fma_f32x2_rn_noftz(q_56, scale0_53, a_55);
                acc[0] = r_57.x;
                acc[1] = r_57.y;
                float2 a_58 = make_float2(acc[2], acc[3]);
                float2 q_59 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                float2 r_60 = fma_f32x2_rn_noftz(q_59, scale0_53, a_58);
                acc[2] = r_60.x;
                acc[3] = r_60.y;
                float2 a_61 = make_float2(acc[4], acc[5]);
                float2 q_62 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                float2 r_63 = fma_f32x2_rn_noftz(q_62, scale0_53, a_61);
                acc[4] = r_63.x;
                acc[5] = r_63.y;
                float2 a_64 = make_float2(acc[6], acc[7]);
                float2 q_65 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                float2 r_66 = fma_f32x2_rn_noftz(q_65, scale0_53, a_64);
                acc[6] = r_66.x;
                acc[7] = r_66.y;
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
                float2 a_67 = make_float2(acc[8], acc[9]);
                float2 q_68 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                float2 r_69 = fma_f32x2_rn_noftz(q_68, scale0_53, a_67);
                acc[8] = r_69.x;
                acc[9] = r_69.y;
                float2 a_70 = make_float2(acc[10], acc[11]);
                float2 q_71 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                float2 r_72 = fma_f32x2_rn_noftz(q_71, scale0_53, a_70);
                acc[10] = r_72.x;
                acc[11] = r_72.y;
                float2 a_73 = make_float2(acc[12], acc[13]);
                float2 q_74 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                float2 r_75 = fma_f32x2_rn_noftz(q_74, scale0_53, a_73);
                acc[12] = r_75.x;
                acc[13] = r_75.y;
                float2 a_76 = make_float2(acc[14], acc[15]);
                float2 q_77 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                float2 r_78 = fma_f32x2_rn_noftz(q_77, scale0_53, a_76);
                acc[14] = r_78.x;
                acc[15] = r_78.y;
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
                float2 a_79 = make_float2(acc[16], acc[17]);
                float2 q_80 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                float2 r_81 = fma_f32x2_rn_noftz(q_80, scale1_54, a_79);
                acc[16] = r_81.x;
                acc[17] = r_81.y;
                float2 a_82 = make_float2(acc[18], acc[19]);
                float2 q_83 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                float2 r_84 = fma_f32x2_rn_noftz(q_83, scale1_54, a_82);
                acc[18] = r_84.x;
                acc[19] = r_84.y;
                float2 a_85 = make_float2(acc[20], acc[21]);
                float2 q_86 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                float2 r_87 = fma_f32x2_rn_noftz(q_86, scale1_54, a_85);
                acc[20] = r_87.x;
                acc[21] = r_87.y;
                float2 a_88 = make_float2(acc[22], acc[23]);
                float2 q_89 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                float2 r_90 = fma_f32x2_rn_noftz(q_89, scale1_54, a_88);
                acc[22] = r_90.x;
                acc[23] = r_90.y;
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
                float2 a_91 = make_float2(acc[24], acc[25]);
                float2 q_92 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                float2 r_93 = fma_f32x2_rn_noftz(q_92, scale1_54, a_91);
                acc[24] = r_93.x;
                acc[25] = r_93.y;
                float2 a_94 = make_float2(acc[26], acc[27]);
                float2 q_95 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                float2 r_96 = fma_f32x2_rn_noftz(q_95, scale1_54, a_94);
                acc[26] = r_96.x;
                acc[27] = r_96.y;
                float2 a_97 = make_float2(acc[28], acc[29]);
                float2 q_98 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                float2 r_99 = fma_f32x2_rn_noftz(q_98, scale1_54, a_97);
                acc[28] = r_99.x;
                acc[29] = r_99.y;
                float2 a_100 = make_float2(acc[30], acc[31]);
                float2 q_101 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                float2 r_102 = fma_f32x2_rn_noftz(q_101, scale1_54, a_100);
                acc[30] = r_102.x;
                acc[31] = r_102.y;
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
                float2 scale0_103 = make_float2(_mul_4, _mul_4);
                float2 scale1_104 = make_float2(_mul_5, _mul_5);
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
                float2 a_105 = make_float2(acc[0], acc[1]);
                float2 q_106 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                float2 r_107 = fma_f32x2_rn_noftz(q_106, scale0_103, a_105);
                acc[0] = r_107.x;
                acc[1] = r_107.y;
                float2 a_108 = make_float2(acc[2], acc[3]);
                float2 q_109 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                float2 r_110 = fma_f32x2_rn_noftz(q_109, scale0_103, a_108);
                acc[2] = r_110.x;
                acc[3] = r_110.y;
                float2 a_111 = make_float2(acc[4], acc[5]);
                float2 q_112 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                float2 r_113 = fma_f32x2_rn_noftz(q_112, scale0_103, a_111);
                acc[4] = r_113.x;
                acc[5] = r_113.y;
                float2 a_114 = make_float2(acc[6], acc[7]);
                float2 q_115 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                float2 r_116 = fma_f32x2_rn_noftz(q_115, scale0_103, a_114);
                acc[6] = r_116.x;
                acc[7] = r_116.y;
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
                float2 a_117 = make_float2(acc[8], acc[9]);
                float2 q_118 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                float2 r_119 = fma_f32x2_rn_noftz(q_118, scale0_103, a_117);
                acc[8] = r_119.x;
                acc[9] = r_119.y;
                float2 a_120 = make_float2(acc[10], acc[11]);
                float2 q_121 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                float2 r_122 = fma_f32x2_rn_noftz(q_121, scale0_103, a_120);
                acc[10] = r_122.x;
                acc[11] = r_122.y;
                float2 a_123 = make_float2(acc[12], acc[13]);
                float2 q_124 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                float2 r_125 = fma_f32x2_rn_noftz(q_124, scale0_103, a_123);
                acc[12] = r_125.x;
                acc[13] = r_125.y;
                float2 a_126 = make_float2(acc[14], acc[15]);
                float2 q_127 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                float2 r_128 = fma_f32x2_rn_noftz(q_127, scale0_103, a_126);
                acc[14] = r_128.x;
                acc[15] = r_128.y;
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
                float2 a_129 = make_float2(acc[16], acc[17]);
                float2 q_130 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                float2 r_131 = fma_f32x2_rn_noftz(q_130, scale1_104, a_129);
                acc[16] = r_131.x;
                acc[17] = r_131.y;
                float2 a_132 = make_float2(acc[18], acc[19]);
                float2 q_133 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                float2 r_134 = fma_f32x2_rn_noftz(q_133, scale1_104, a_132);
                acc[18] = r_134.x;
                acc[19] = r_134.y;
                float2 a_135 = make_float2(acc[20], acc[21]);
                float2 q_136 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                float2 r_137 = fma_f32x2_rn_noftz(q_136, scale1_104, a_135);
                acc[20] = r_137.x;
                acc[21] = r_137.y;
                float2 a_138 = make_float2(acc[22], acc[23]);
                float2 q_139 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                float2 r_140 = fma_f32x2_rn_noftz(q_139, scale1_104, a_138);
                acc[22] = r_140.x;
                acc[23] = r_140.y;
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
                float2 a_141 = make_float2(acc[24], acc[25]);
                float2 q_142 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                float2 r_143 = fma_f32x2_rn_noftz(q_142, scale1_104, a_141);
                acc[24] = r_143.x;
                acc[25] = r_143.y;
                float2 a_144 = make_float2(acc[26], acc[27]);
                float2 q_145 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                float2 r_146 = fma_f32x2_rn_noftz(q_145, scale1_104, a_144);
                acc[26] = r_146.x;
                acc[27] = r_146.y;
                float2 a_147 = make_float2(acc[28], acc[29]);
                float2 q_148 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                float2 r_149 = fma_f32x2_rn_noftz(q_148, scale1_104, a_147);
                acc[28] = r_149.x;
                acc[29] = r_149.y;
                float2 a_150 = make_float2(acc[30], acc[31]);
                float2 q_151 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                float2 r_152 = fma_f32x2_rn_noftz(q_151, scale1_104, a_150);
                acc[30] = r_152.x;
                acc[31] = r_152.y;
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
                float2 scale0_153 = make_float2(_mul_6, _mul_6);
                float2 scale1_154 = make_float2(_mul_7, _mul_7);
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
                float2 a_155 = make_float2(acc[0], acc[1]);
                float2 q_156 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                float2 r_157 = fma_f32x2_rn_noftz(q_156, scale0_153, a_155);
                acc[0] = r_157.x;
                acc[1] = r_157.y;
                float2 a_158 = make_float2(acc[2], acc[3]);
                float2 q_159 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                float2 r_160 = fma_f32x2_rn_noftz(q_159, scale0_153, a_158);
                acc[2] = r_160.x;
                acc[3] = r_160.y;
                float2 a_161 = make_float2(acc[4], acc[5]);
                float2 q_162 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                float2 r_163 = fma_f32x2_rn_noftz(q_162, scale0_153, a_161);
                acc[4] = r_163.x;
                acc[5] = r_163.y;
                float2 a_164 = make_float2(acc[6], acc[7]);
                float2 q_165 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                float2 r_166 = fma_f32x2_rn_noftz(q_165, scale0_153, a_164);
                acc[6] = r_166.x;
                acc[7] = r_166.y;
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
                float2 a_167 = make_float2(acc[8], acc[9]);
                float2 q_168 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                float2 r_169 = fma_f32x2_rn_noftz(q_168, scale0_153, a_167);
                acc[8] = r_169.x;
                acc[9] = r_169.y;
                float2 a_170 = make_float2(acc[10], acc[11]);
                float2 q_171 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                float2 r_172 = fma_f32x2_rn_noftz(q_171, scale0_153, a_170);
                acc[10] = r_172.x;
                acc[11] = r_172.y;
                float2 a_173 = make_float2(acc[12], acc[13]);
                float2 q_174 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                float2 r_175 = fma_f32x2_rn_noftz(q_174, scale0_153, a_173);
                acc[12] = r_175.x;
                acc[13] = r_175.y;
                float2 a_176 = make_float2(acc[14], acc[15]);
                float2 q_177 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                float2 r_178 = fma_f32x2_rn_noftz(q_177, scale0_153, a_176);
                acc[14] = r_178.x;
                acc[15] = r_178.y;
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
                float2 a_179 = make_float2(acc[16], acc[17]);
                float2 q_180 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                float2 r_181 = fma_f32x2_rn_noftz(q_180, scale1_154, a_179);
                acc[16] = r_181.x;
                acc[17] = r_181.y;
                float2 a_182 = make_float2(acc[18], acc[19]);
                float2 q_183 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                float2 r_184 = fma_f32x2_rn_noftz(q_183, scale1_154, a_182);
                acc[18] = r_184.x;
                acc[19] = r_184.y;
                float2 a_185 = make_float2(acc[20], acc[21]);
                float2 q_186 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                float2 r_187 = fma_f32x2_rn_noftz(q_186, scale1_154, a_185);
                acc[20] = r_187.x;
                acc[21] = r_187.y;
                float2 a_188 = make_float2(acc[22], acc[23]);
                float2 q_189 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                float2 r_190 = fma_f32x2_rn_noftz(q_189, scale1_154, a_188);
                acc[22] = r_190.x;
                acc[23] = r_190.y;
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
                float2 a_191 = make_float2(acc[24], acc[25]);
                float2 q_192 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                float2 r_193 = fma_f32x2_rn_noftz(q_192, scale1_154, a_191);
                acc[24] = r_193.x;
                acc[25] = r_193.y;
                float2 a_194 = make_float2(acc[26], acc[27]);
                float2 q_195 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                float2 r_196 = fma_f32x2_rn_noftz(q_195, scale1_154, a_194);
                acc[26] = r_196.x;
                acc[27] = r_196.y;
                float2 a_197 = make_float2(acc[28], acc[29]);
                float2 q_198 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                float2 r_199 = fma_f32x2_rn_noftz(q_198, scale1_154, a_197);
                acc[28] = r_199.x;
                acc[29] = r_199.y;
                float2 a_200 = make_float2(acc[30], acc[31]);
                float2 q_201 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                float2 r_202 = fma_f32x2_rn_noftz(q_201, scale1_154, a_200);
                acc[30] = r_202.x;
                acc[31] = r_202.y;
                float acc_203[32] = {0};
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
                float2 scale0_204 = make_float2(_mul_8, _mul_8);
                float2 scale1_205 = make_float2(_mul_9, _mul_9);
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
                float2 a_206 = make_float2(acc_203[0], acc_203[1]);
                float2 q_207 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                float2 r_208 = fma_f32x2_rn_noftz(q_207, scale0_204, a_206);
                acc_203[0] = r_208.x;
                acc_203[1] = r_208.y;
                float2 a_209 = make_float2(acc_203[2], acc_203[3]);
                float2 q_210 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                float2 r_211 = fma_f32x2_rn_noftz(q_210, scale0_204, a_209);
                acc_203[2] = r_211.x;
                acc_203[3] = r_211.y;
                float2 a_212 = make_float2(acc_203[4], acc_203[5]);
                float2 q_213 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                float2 r_214 = fma_f32x2_rn_noftz(q_213, scale0_204, a_212);
                acc_203[4] = r_214.x;
                acc_203[5] = r_214.y;
                float2 a_215 = make_float2(acc_203[6], acc_203[7]);
                float2 q_216 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                float2 r_217 = fma_f32x2_rn_noftz(q_216, scale0_204, a_215);
                acc_203[6] = r_217.x;
                acc_203[7] = r_217.y;
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
                float2 a_218 = make_float2(acc_203[8], acc_203[9]);
                float2 q_219 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                float2 r_220 = fma_f32x2_rn_noftz(q_219, scale0_204, a_218);
                acc_203[8] = r_220.x;
                acc_203[9] = r_220.y;
                float2 a_221 = make_float2(acc_203[10], acc_203[11]);
                float2 q_222 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                float2 r_223 = fma_f32x2_rn_noftz(q_222, scale0_204, a_221);
                acc_203[10] = r_223.x;
                acc_203[11] = r_223.y;
                float2 a_224 = make_float2(acc_203[12], acc_203[13]);
                float2 q_225 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                float2 r_226 = fma_f32x2_rn_noftz(q_225, scale0_204, a_224);
                acc_203[12] = r_226.x;
                acc_203[13] = r_226.y;
                float2 a_227 = make_float2(acc_203[14], acc_203[15]);
                float2 q_228 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                float2 r_229 = fma_f32x2_rn_noftz(q_228, scale0_204, a_227);
                acc_203[14] = r_229.x;
                acc_203[15] = r_229.y;
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
                float2 a_230 = make_float2(acc_203[16], acc_203[17]);
                float2 q_231 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                float2 r_232 = fma_f32x2_rn_noftz(q_231, scale1_205, a_230);
                acc_203[16] = r_232.x;
                acc_203[17] = r_232.y;
                float2 a_233 = make_float2(acc_203[18], acc_203[19]);
                float2 q_234 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                float2 r_235 = fma_f32x2_rn_noftz(q_234, scale1_205, a_233);
                acc_203[18] = r_235.x;
                acc_203[19] = r_235.y;
                float2 a_236 = make_float2(acc_203[20], acc_203[21]);
                float2 q_237 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                float2 r_238 = fma_f32x2_rn_noftz(q_237, scale1_205, a_236);
                acc_203[20] = r_238.x;
                acc_203[21] = r_238.y;
                float2 a_239 = make_float2(acc_203[22], acc_203[23]);
                float2 q_240 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                float2 r_241 = fma_f32x2_rn_noftz(q_240, scale1_205, a_239);
                acc_203[22] = r_241.x;
                acc_203[23] = r_241.y;
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
                float2 a_242 = make_float2(acc_203[24], acc_203[25]);
                float2 q_243 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                float2 r_244 = fma_f32x2_rn_noftz(q_243, scale1_205, a_242);
                acc_203[24] = r_244.x;
                acc_203[25] = r_244.y;
                float2 a_245 = make_float2(acc_203[26], acc_203[27]);
                float2 q_246 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                float2 r_247 = fma_f32x2_rn_noftz(q_246, scale1_205, a_245);
                acc_203[26] = r_247.x;
                acc_203[27] = r_247.y;
                float2 a_248 = make_float2(acc_203[28], acc_203[29]);
                float2 q_249 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                float2 r_250 = fma_f32x2_rn_noftz(q_249, scale1_205, a_248);
                acc_203[28] = r_250.x;
                acc_203[29] = r_250.y;
                float2 a_251 = make_float2(acc_203[30], acc_203[31]);
                float2 q_252 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                float2 r_253 = fma_f32x2_rn_noftz(q_252, scale1_205, a_251);
                acc_203[30] = r_253.x;
                acc_203[31] = r_253.y;
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
                float2 scale0_254 = make_float2(_mul_10, _mul_10);
                float2 scale1_255 = make_float2(_mul_11, _mul_11);
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
                float2 a_256 = make_float2(acc_203[0], acc_203[1]);
                float2 q_257 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                float2 r_258 = fma_f32x2_rn_noftz(q_257, scale0_254, a_256);
                acc_203[0] = r_258.x;
                acc_203[1] = r_258.y;
                float2 a_259 = make_float2(acc_203[2], acc_203[3]);
                float2 q_260 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                float2 r_261 = fma_f32x2_rn_noftz(q_260, scale0_254, a_259);
                acc_203[2] = r_261.x;
                acc_203[3] = r_261.y;
                float2 a_262 = make_float2(acc_203[4], acc_203[5]);
                float2 q_263 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                float2 r_264 = fma_f32x2_rn_noftz(q_263, scale0_254, a_262);
                acc_203[4] = r_264.x;
                acc_203[5] = r_264.y;
                float2 a_265 = make_float2(acc_203[6], acc_203[7]);
                float2 q_266 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                float2 r_267 = fma_f32x2_rn_noftz(q_266, scale0_254, a_265);
                acc_203[6] = r_267.x;
                acc_203[7] = r_267.y;
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
                float2 a_268 = make_float2(acc_203[8], acc_203[9]);
                float2 q_269 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                float2 r_270 = fma_f32x2_rn_noftz(q_269, scale0_254, a_268);
                acc_203[8] = r_270.x;
                acc_203[9] = r_270.y;
                float2 a_271 = make_float2(acc_203[10], acc_203[11]);
                float2 q_272 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                float2 r_273 = fma_f32x2_rn_noftz(q_272, scale0_254, a_271);
                acc_203[10] = r_273.x;
                acc_203[11] = r_273.y;
                float2 a_274 = make_float2(acc_203[12], acc_203[13]);
                float2 q_275 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                float2 r_276 = fma_f32x2_rn_noftz(q_275, scale0_254, a_274);
                acc_203[12] = r_276.x;
                acc_203[13] = r_276.y;
                float2 a_277 = make_float2(acc_203[14], acc_203[15]);
                float2 q_278 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                float2 r_279 = fma_f32x2_rn_noftz(q_278, scale0_254, a_277);
                acc_203[14] = r_279.x;
                acc_203[15] = r_279.y;
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
                float2 a_280 = make_float2(acc_203[16], acc_203[17]);
                float2 q_281 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                float2 r_282 = fma_f32x2_rn_noftz(q_281, scale1_255, a_280);
                acc_203[16] = r_282.x;
                acc_203[17] = r_282.y;
                float2 a_283 = make_float2(acc_203[18], acc_203[19]);
                float2 q_284 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                float2 r_285 = fma_f32x2_rn_noftz(q_284, scale1_255, a_283);
                acc_203[18] = r_285.x;
                acc_203[19] = r_285.y;
                float2 a_286 = make_float2(acc_203[20], acc_203[21]);
                float2 q_287 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                float2 r_288 = fma_f32x2_rn_noftz(q_287, scale1_255, a_286);
                acc_203[20] = r_288.x;
                acc_203[21] = r_288.y;
                float2 a_289 = make_float2(acc_203[22], acc_203[23]);
                float2 q_290 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                float2 r_291 = fma_f32x2_rn_noftz(q_290, scale1_255, a_289);
                acc_203[22] = r_291.x;
                acc_203[23] = r_291.y;
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
                float2 a_292 = make_float2(acc_203[24], acc_203[25]);
                float2 q_293 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                float2 r_294 = fma_f32x2_rn_noftz(q_293, scale1_255, a_292);
                acc_203[24] = r_294.x;
                acc_203[25] = r_294.y;
                float2 a_295 = make_float2(acc_203[26], acc_203[27]);
                float2 q_296 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                float2 r_297 = fma_f32x2_rn_noftz(q_296, scale1_255, a_295);
                acc_203[26] = r_297.x;
                acc_203[27] = r_297.y;
                float2 a_298 = make_float2(acc_203[28], acc_203[29]);
                float2 q_299 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                float2 r_300 = fma_f32x2_rn_noftz(q_299, scale1_255, a_298);
                acc_203[28] = r_300.x;
                acc_203[29] = r_300.y;
                float2 a_301 = make_float2(acc_203[30], acc_203[31]);
                float2 q_302 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                float2 r_303 = fma_f32x2_rn_noftz(q_302, scale1_255, a_301);
                acc_203[30] = r_303.x;
                acc_203[31] = r_303.y;
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
                float2 scale0_304 = make_float2(_mul_12, _mul_12);
                float2 scale1_305 = make_float2(_mul_13, _mul_13);
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
                float2 a_306 = make_float2(acc_203[0], acc_203[1]);
                float2 q_307 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                float2 r_308 = fma_f32x2_rn_noftz(q_307, scale0_304, a_306);
                acc_203[0] = r_308.x;
                acc_203[1] = r_308.y;
                float2 a_309 = make_float2(acc_203[2], acc_203[3]);
                float2 q_310 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                float2 r_311 = fma_f32x2_rn_noftz(q_310, scale0_304, a_309);
                acc_203[2] = r_311.x;
                acc_203[3] = r_311.y;
                float2 a_312 = make_float2(acc_203[4], acc_203[5]);
                float2 q_313 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                float2 r_314 = fma_f32x2_rn_noftz(q_313, scale0_304, a_312);
                acc_203[4] = r_314.x;
                acc_203[5] = r_314.y;
                float2 a_315 = make_float2(acc_203[6], acc_203[7]);
                float2 q_316 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                float2 r_317 = fma_f32x2_rn_noftz(q_316, scale0_304, a_315);
                acc_203[6] = r_317.x;
                acc_203[7] = r_317.y;
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
                float2 a_318 = make_float2(acc_203[8], acc_203[9]);
                float2 q_319 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                float2 r_320 = fma_f32x2_rn_noftz(q_319, scale0_304, a_318);
                acc_203[8] = r_320.x;
                acc_203[9] = r_320.y;
                float2 a_321 = make_float2(acc_203[10], acc_203[11]);
                float2 q_322 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                float2 r_323 = fma_f32x2_rn_noftz(q_322, scale0_304, a_321);
                acc_203[10] = r_323.x;
                acc_203[11] = r_323.y;
                float2 a_324 = make_float2(acc_203[12], acc_203[13]);
                float2 q_325 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                float2 r_326 = fma_f32x2_rn_noftz(q_325, scale0_304, a_324);
                acc_203[12] = r_326.x;
                acc_203[13] = r_326.y;
                float2 a_327 = make_float2(acc_203[14], acc_203[15]);
                float2 q_328 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                float2 r_329 = fma_f32x2_rn_noftz(q_328, scale0_304, a_327);
                acc_203[14] = r_329.x;
                acc_203[15] = r_329.y;
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
                float2 a_330 = make_float2(acc_203[16], acc_203[17]);
                float2 q_331 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                float2 r_332 = fma_f32x2_rn_noftz(q_331, scale1_305, a_330);
                acc_203[16] = r_332.x;
                acc_203[17] = r_332.y;
                float2 a_333 = make_float2(acc_203[18], acc_203[19]);
                float2 q_334 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                float2 r_335 = fma_f32x2_rn_noftz(q_334, scale1_305, a_333);
                acc_203[18] = r_335.x;
                acc_203[19] = r_335.y;
                float2 a_336 = make_float2(acc_203[20], acc_203[21]);
                float2 q_337 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                float2 r_338 = fma_f32x2_rn_noftz(q_337, scale1_305, a_336);
                acc_203[20] = r_338.x;
                acc_203[21] = r_338.y;
                float2 a_339 = make_float2(acc_203[22], acc_203[23]);
                float2 q_340 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                float2 r_341 = fma_f32x2_rn_noftz(q_340, scale1_305, a_339);
                acc_203[22] = r_341.x;
                acc_203[23] = r_341.y;
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
                float2 a_342 = make_float2(acc_203[24], acc_203[25]);
                float2 q_343 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                float2 r_344 = fma_f32x2_rn_noftz(q_343, scale1_305, a_342);
                acc_203[24] = r_344.x;
                acc_203[25] = r_344.y;
                float2 a_345 = make_float2(acc_203[26], acc_203[27]);
                float2 q_346 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                float2 r_347 = fma_f32x2_rn_noftz(q_346, scale1_305, a_345);
                acc_203[26] = r_347.x;
                acc_203[27] = r_347.y;
                float2 a_348 = make_float2(acc_203[28], acc_203[29]);
                float2 q_349 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                float2 r_350 = fma_f32x2_rn_noftz(q_349, scale1_305, a_348);
                acc_203[28] = r_350.x;
                acc_203[29] = r_350.y;
                float2 a_351 = make_float2(acc_203[30], acc_203[31]);
                float2 q_352 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                float2 r_353 = fma_f32x2_rn_noftz(q_352, scale1_305, a_351);
                acc_203[30] = r_353.x;
                acc_203[31] = r_353.y;
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
                float2 scale0_354 = make_float2(_mul_14, _mul_14);
                float2 scale1_355 = make_float2(_mul_15, _mul_15);
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
                float2 a_356 = make_float2(acc_203[0], acc_203[1]);
                float2 q_357 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                float2 r_358 = fma_f32x2_rn_noftz(q_357, scale0_354, a_356);
                acc_203[0] = r_358.x;
                acc_203[1] = r_358.y;
                float2 a_359 = make_float2(acc_203[2], acc_203[3]);
                float2 q_360 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                float2 r_361 = fma_f32x2_rn_noftz(q_360, scale0_354, a_359);
                acc_203[2] = r_361.x;
                acc_203[3] = r_361.y;
                float2 a_362 = make_float2(acc_203[4], acc_203[5]);
                float2 q_363 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                float2 r_364 = fma_f32x2_rn_noftz(q_363, scale0_354, a_362);
                acc_203[4] = r_364.x;
                acc_203[5] = r_364.y;
                float2 a_365 = make_float2(acc_203[6], acc_203[7]);
                float2 q_366 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                float2 r_367 = fma_f32x2_rn_noftz(q_366, scale0_354, a_365);
                acc_203[6] = r_367.x;
                acc_203[7] = r_367.y;
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
                float2 a_368 = make_float2(acc_203[8], acc_203[9]);
                float2 q_369 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                float2 r_370 = fma_f32x2_rn_noftz(q_369, scale0_354, a_368);
                acc_203[8] = r_370.x;
                acc_203[9] = r_370.y;
                float2 a_371 = make_float2(acc_203[10], acc_203[11]);
                float2 q_372 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                float2 r_373 = fma_f32x2_rn_noftz(q_372, scale0_354, a_371);
                acc_203[10] = r_373.x;
                acc_203[11] = r_373.y;
                float2 a_374 = make_float2(acc_203[12], acc_203[13]);
                float2 q_375 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                float2 r_376 = fma_f32x2_rn_noftz(q_375, scale0_354, a_374);
                acc_203[12] = r_376.x;
                acc_203[13] = r_376.y;
                float2 a_377 = make_float2(acc_203[14], acc_203[15]);
                float2 q_378 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                float2 r_379 = fma_f32x2_rn_noftz(q_378, scale0_354, a_377);
                acc_203[14] = r_379.x;
                acc_203[15] = r_379.y;
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
                float2 a_380 = make_float2(acc_203[16], acc_203[17]);
                float2 q_381 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                float2 r_382 = fma_f32x2_rn_noftz(q_381, scale1_355, a_380);
                acc_203[16] = r_382.x;
                acc_203[17] = r_382.y;
                float2 a_383 = make_float2(acc_203[18], acc_203[19]);
                float2 q_384 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                float2 r_385 = fma_f32x2_rn_noftz(q_384, scale1_355, a_383);
                acc_203[18] = r_385.x;
                acc_203[19] = r_385.y;
                float2 a_386 = make_float2(acc_203[20], acc_203[21]);
                float2 q_387 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                float2 r_388 = fma_f32x2_rn_noftz(q_387, scale1_355, a_386);
                acc_203[20] = r_388.x;
                acc_203[21] = r_388.y;
                float2 a_389 = make_float2(acc_203[22], acc_203[23]);
                float2 q_390 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                float2 r_391 = fma_f32x2_rn_noftz(q_390, scale1_355, a_389);
                acc_203[22] = r_391.x;
                acc_203[23] = r_391.y;
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
                float2 a_392 = make_float2(acc_203[24], acc_203[25]);
                float2 q_393 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                float2 r_394 = fma_f32x2_rn_noftz(q_393, scale1_355, a_392);
                acc_203[24] = r_394.x;
                acc_203[25] = r_394.y;
                float2 a_395 = make_float2(acc_203[26], acc_203[27]);
                float2 q_396 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                float2 r_397 = fma_f32x2_rn_noftz(q_396, scale1_355, a_395);
                acc_203[26] = r_397.x;
                acc_203[27] = r_397.y;
                float2 a_398 = make_float2(acc_203[28], acc_203[29]);
                float2 q_399 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                float2 r_400 = fma_f32x2_rn_noftz(q_399, scale1_355, a_398);
                acc_203[28] = r_400.x;
                acc_203[29] = r_400.y;
                float2 a_401 = make_float2(acc_203[30], acc_203[31]);
                float2 q_402 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                float2 r_403 = fma_f32x2_rn_noftz(q_402, scale1_355, a_401);
                acc_203[30] = r_403.x;
                acc_203[31] = r_403.y;
                float acc_404[32] = {0};
                uint32_t _e4m3x2_to_f16x2_8;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)(_fp4_wire_scales_0[8])));
                float _cvt_f32_f16_16;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(_e4m3x2_to_f16x2_8)));
                float _cvt_f32_f16_17;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(_e4m3x2_to_f16x2_8 >> 16)));
                float _mul_16;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_cvt_f32_f16_16), "f"(g_s));
                float _mul_17;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_cvt_f32_f16_17), "f"(g_s));
                float2 scale0_405 = make_float2(_mul_16, _mul_16);
                float2 scale1_406 = make_float2(_mul_17, _mul_17);
                float _unpack_fp4_32[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[32 + 0]));
                    (&_unpack_fp4_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_32[0]), "=f"(_unpack_fp4_32[1]), "=f"(_unpack_fp4_32[2]), "=f"(_unpack_fp4_32[3]), "=f"(_unpack_fp4_32[4]), "=f"(_unpack_fp4_32[5]), "=f"(_unpack_fp4_32[6]), "=f"(_unpack_fp4_32[7]) : "r"(_fp4_wire_values_0[32 + 0]));
                    #endif
                }
                float2 a_407 = make_float2(acc_404[0], acc_404[1]);
                float2 q_408 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                float2 r_409 = fma_f32x2_rn_noftz(q_408, scale0_405, a_407);
                acc_404[0] = r_409.x;
                acc_404[1] = r_409.y;
                float2 a_410 = make_float2(acc_404[2], acc_404[3]);
                float2 q_411 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                float2 r_412 = fma_f32x2_rn_noftz(q_411, scale0_405, a_410);
                acc_404[2] = r_412.x;
                acc_404[3] = r_412.y;
                float2 a_413 = make_float2(acc_404[4], acc_404[5]);
                float2 q_414 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                float2 r_415 = fma_f32x2_rn_noftz(q_414, scale0_405, a_413);
                acc_404[4] = r_415.x;
                acc_404[5] = r_415.y;
                float2 a_416 = make_float2(acc_404[6], acc_404[7]);
                float2 q_417 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                float2 r_418 = fma_f32x2_rn_noftz(q_417, scale0_405, a_416);
                acc_404[6] = r_418.x;
                acc_404[7] = r_418.y;
                float _unpack_fp4_33[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[33 + 0]));
                    (&_unpack_fp4_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_33[0]), "=f"(_unpack_fp4_33[1]), "=f"(_unpack_fp4_33[2]), "=f"(_unpack_fp4_33[3]), "=f"(_unpack_fp4_33[4]), "=f"(_unpack_fp4_33[5]), "=f"(_unpack_fp4_33[6]), "=f"(_unpack_fp4_33[7]) : "r"(_fp4_wire_values_0[33 + 0]));
                    #endif
                }
                float2 a_419 = make_float2(acc_404[8], acc_404[9]);
                float2 q_420 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                float2 r_421 = fma_f32x2_rn_noftz(q_420, scale0_405, a_419);
                acc_404[8] = r_421.x;
                acc_404[9] = r_421.y;
                float2 a_422 = make_float2(acc_404[10], acc_404[11]);
                float2 q_423 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                float2 r_424 = fma_f32x2_rn_noftz(q_423, scale0_405, a_422);
                acc_404[10] = r_424.x;
                acc_404[11] = r_424.y;
                float2 a_425 = make_float2(acc_404[12], acc_404[13]);
                float2 q_426 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                float2 r_427 = fma_f32x2_rn_noftz(q_426, scale0_405, a_425);
                acc_404[12] = r_427.x;
                acc_404[13] = r_427.y;
                float2 a_428 = make_float2(acc_404[14], acc_404[15]);
                float2 q_429 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                float2 r_430 = fma_f32x2_rn_noftz(q_429, scale0_405, a_428);
                acc_404[14] = r_430.x;
                acc_404[15] = r_430.y;
                float _unpack_fp4_34[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[34 + 0]));
                    (&_unpack_fp4_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_34[0]), "=f"(_unpack_fp4_34[1]), "=f"(_unpack_fp4_34[2]), "=f"(_unpack_fp4_34[3]), "=f"(_unpack_fp4_34[4]), "=f"(_unpack_fp4_34[5]), "=f"(_unpack_fp4_34[6]), "=f"(_unpack_fp4_34[7]) : "r"(_fp4_wire_values_0[34 + 0]));
                    #endif
                }
                float2 a_431 = make_float2(acc_404[16], acc_404[17]);
                float2 q_432 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                float2 r_433 = fma_f32x2_rn_noftz(q_432, scale1_406, a_431);
                acc_404[16] = r_433.x;
                acc_404[17] = r_433.y;
                float2 a_434 = make_float2(acc_404[18], acc_404[19]);
                float2 q_435 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                float2 r_436 = fma_f32x2_rn_noftz(q_435, scale1_406, a_434);
                acc_404[18] = r_436.x;
                acc_404[19] = r_436.y;
                float2 a_437 = make_float2(acc_404[20], acc_404[21]);
                float2 q_438 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                float2 r_439 = fma_f32x2_rn_noftz(q_438, scale1_406, a_437);
                acc_404[20] = r_439.x;
                acc_404[21] = r_439.y;
                float2 a_440 = make_float2(acc_404[22], acc_404[23]);
                float2 q_441 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                float2 r_442 = fma_f32x2_rn_noftz(q_441, scale1_406, a_440);
                acc_404[22] = r_442.x;
                acc_404[23] = r_442.y;
                float _unpack_fp4_35[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[35 + 0]));
                    (&_unpack_fp4_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_35[0]), "=f"(_unpack_fp4_35[1]), "=f"(_unpack_fp4_35[2]), "=f"(_unpack_fp4_35[3]), "=f"(_unpack_fp4_35[4]), "=f"(_unpack_fp4_35[5]), "=f"(_unpack_fp4_35[6]), "=f"(_unpack_fp4_35[7]) : "r"(_fp4_wire_values_0[35 + 0]));
                    #endif
                }
                float2 a_443 = make_float2(acc_404[24], acc_404[25]);
                float2 q_444 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                float2 r_445 = fma_f32x2_rn_noftz(q_444, scale1_406, a_443);
                acc_404[24] = r_445.x;
                acc_404[25] = r_445.y;
                float2 a_446 = make_float2(acc_404[26], acc_404[27]);
                float2 q_447 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                float2 r_448 = fma_f32x2_rn_noftz(q_447, scale1_406, a_446);
                acc_404[26] = r_448.x;
                acc_404[27] = r_448.y;
                float2 a_449 = make_float2(acc_404[28], acc_404[29]);
                float2 q_450 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                float2 r_451 = fma_f32x2_rn_noftz(q_450, scale1_406, a_449);
                acc_404[28] = r_451.x;
                acc_404[29] = r_451.y;
                float2 a_452 = make_float2(acc_404[30], acc_404[31]);
                float2 q_453 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                float2 r_454 = fma_f32x2_rn_noftz(q_453, scale1_406, a_452);
                acc_404[30] = r_454.x;
                acc_404[31] = r_454.y;
                uint32_t _e4m3x2_to_f16x2_9;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)(_fp4_wire_scales_0[9])));
                float _cvt_f32_f16_18;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(_e4m3x2_to_f16x2_9)));
                float _cvt_f32_f16_19;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(_e4m3x2_to_f16x2_9 >> 16)));
                float _mul_18;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_cvt_f32_f16_18), "f"(g_s_2));
                float _mul_19;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_cvt_f32_f16_19), "f"(g_s_2));
                float2 scale0_455 = make_float2(_mul_18, _mul_18);
                float2 scale1_456 = make_float2(_mul_19, _mul_19);
                float _unpack_fp4_36[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[36 + 0]));
                    (&_unpack_fp4_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_36[0]), "=f"(_unpack_fp4_36[1]), "=f"(_unpack_fp4_36[2]), "=f"(_unpack_fp4_36[3]), "=f"(_unpack_fp4_36[4]), "=f"(_unpack_fp4_36[5]), "=f"(_unpack_fp4_36[6]), "=f"(_unpack_fp4_36[7]) : "r"(_fp4_wire_values_0[36 + 0]));
                    #endif
                }
                float2 a_457 = make_float2(acc_404[0], acc_404[1]);
                float2 q_458 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                float2 r_459 = fma_f32x2_rn_noftz(q_458, scale0_455, a_457);
                acc_404[0] = r_459.x;
                acc_404[1] = r_459.y;
                float2 a_460 = make_float2(acc_404[2], acc_404[3]);
                float2 q_461 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                float2 r_462 = fma_f32x2_rn_noftz(q_461, scale0_455, a_460);
                acc_404[2] = r_462.x;
                acc_404[3] = r_462.y;
                float2 a_463 = make_float2(acc_404[4], acc_404[5]);
                float2 q_464 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                float2 r_465 = fma_f32x2_rn_noftz(q_464, scale0_455, a_463);
                acc_404[4] = r_465.x;
                acc_404[5] = r_465.y;
                float2 a_466 = make_float2(acc_404[6], acc_404[7]);
                float2 q_467 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                float2 r_468 = fma_f32x2_rn_noftz(q_467, scale0_455, a_466);
                acc_404[6] = r_468.x;
                acc_404[7] = r_468.y;
                float _unpack_fp4_37[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[37 + 0]));
                    (&_unpack_fp4_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_37[0]), "=f"(_unpack_fp4_37[1]), "=f"(_unpack_fp4_37[2]), "=f"(_unpack_fp4_37[3]), "=f"(_unpack_fp4_37[4]), "=f"(_unpack_fp4_37[5]), "=f"(_unpack_fp4_37[6]), "=f"(_unpack_fp4_37[7]) : "r"(_fp4_wire_values_0[37 + 0]));
                    #endif
                }
                float2 a_469 = make_float2(acc_404[8], acc_404[9]);
                float2 q_470 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                float2 r_471 = fma_f32x2_rn_noftz(q_470, scale0_455, a_469);
                acc_404[8] = r_471.x;
                acc_404[9] = r_471.y;
                float2 a_472 = make_float2(acc_404[10], acc_404[11]);
                float2 q_473 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                float2 r_474 = fma_f32x2_rn_noftz(q_473, scale0_455, a_472);
                acc_404[10] = r_474.x;
                acc_404[11] = r_474.y;
                float2 a_475 = make_float2(acc_404[12], acc_404[13]);
                float2 q_476 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                float2 r_477 = fma_f32x2_rn_noftz(q_476, scale0_455, a_475);
                acc_404[12] = r_477.x;
                acc_404[13] = r_477.y;
                float2 a_478 = make_float2(acc_404[14], acc_404[15]);
                float2 q_479 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                float2 r_480 = fma_f32x2_rn_noftz(q_479, scale0_455, a_478);
                acc_404[14] = r_480.x;
                acc_404[15] = r_480.y;
                float _unpack_fp4_38[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[38 + 0]));
                    (&_unpack_fp4_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_38[0]), "=f"(_unpack_fp4_38[1]), "=f"(_unpack_fp4_38[2]), "=f"(_unpack_fp4_38[3]), "=f"(_unpack_fp4_38[4]), "=f"(_unpack_fp4_38[5]), "=f"(_unpack_fp4_38[6]), "=f"(_unpack_fp4_38[7]) : "r"(_fp4_wire_values_0[38 + 0]));
                    #endif
                }
                float2 a_481 = make_float2(acc_404[16], acc_404[17]);
                float2 q_482 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                float2 r_483 = fma_f32x2_rn_noftz(q_482, scale1_456, a_481);
                acc_404[16] = r_483.x;
                acc_404[17] = r_483.y;
                float2 a_484 = make_float2(acc_404[18], acc_404[19]);
                float2 q_485 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                float2 r_486 = fma_f32x2_rn_noftz(q_485, scale1_456, a_484);
                acc_404[18] = r_486.x;
                acc_404[19] = r_486.y;
                float2 a_487 = make_float2(acc_404[20], acc_404[21]);
                float2 q_488 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                float2 r_489 = fma_f32x2_rn_noftz(q_488, scale1_456, a_487);
                acc_404[20] = r_489.x;
                acc_404[21] = r_489.y;
                float2 a_490 = make_float2(acc_404[22], acc_404[23]);
                float2 q_491 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                float2 r_492 = fma_f32x2_rn_noftz(q_491, scale1_456, a_490);
                acc_404[22] = r_492.x;
                acc_404[23] = r_492.y;
                float _unpack_fp4_39[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[39 + 0]));
                    (&_unpack_fp4_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_39[0]), "=f"(_unpack_fp4_39[1]), "=f"(_unpack_fp4_39[2]), "=f"(_unpack_fp4_39[3]), "=f"(_unpack_fp4_39[4]), "=f"(_unpack_fp4_39[5]), "=f"(_unpack_fp4_39[6]), "=f"(_unpack_fp4_39[7]) : "r"(_fp4_wire_values_0[39 + 0]));
                    #endif
                }
                float2 a_493 = make_float2(acc_404[24], acc_404[25]);
                float2 q_494 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                float2 r_495 = fma_f32x2_rn_noftz(q_494, scale1_456, a_493);
                acc_404[24] = r_495.x;
                acc_404[25] = r_495.y;
                float2 a_496 = make_float2(acc_404[26], acc_404[27]);
                float2 q_497 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                float2 r_498 = fma_f32x2_rn_noftz(q_497, scale1_456, a_496);
                acc_404[26] = r_498.x;
                acc_404[27] = r_498.y;
                float2 a_499 = make_float2(acc_404[28], acc_404[29]);
                float2 q_500 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                float2 r_501 = fma_f32x2_rn_noftz(q_500, scale1_456, a_499);
                acc_404[28] = r_501.x;
                acc_404[29] = r_501.y;
                float2 a_502 = make_float2(acc_404[30], acc_404[31]);
                float2 q_503 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                float2 r_504 = fma_f32x2_rn_noftz(q_503, scale1_456, a_502);
                acc_404[30] = r_504.x;
                acc_404[31] = r_504.y;
                uint32_t _e4m3x2_to_f16x2_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)(_fp4_wire_scales_0[10])));
                float _cvt_f32_f16_20;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(_e4m3x2_to_f16x2_10)));
                float _cvt_f32_f16_21;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(_e4m3x2_to_f16x2_10 >> 16)));
                float _mul_20;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_cvt_f32_f16_20), "f"(g_s_3));
                float _mul_21;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_cvt_f32_f16_21), "f"(g_s_3));
                float2 scale0_505 = make_float2(_mul_20, _mul_20);
                float2 scale1_506 = make_float2(_mul_21, _mul_21);
                float _unpack_fp4_40[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[40 + 0]));
                    (&_unpack_fp4_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_40[0]), "=f"(_unpack_fp4_40[1]), "=f"(_unpack_fp4_40[2]), "=f"(_unpack_fp4_40[3]), "=f"(_unpack_fp4_40[4]), "=f"(_unpack_fp4_40[5]), "=f"(_unpack_fp4_40[6]), "=f"(_unpack_fp4_40[7]) : "r"(_fp4_wire_values_0[40 + 0]));
                    #endif
                }
                float2 a_507 = make_float2(acc_404[0], acc_404[1]);
                float2 q_508 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                float2 r_509 = fma_f32x2_rn_noftz(q_508, scale0_505, a_507);
                acc_404[0] = r_509.x;
                acc_404[1] = r_509.y;
                float2 a_510 = make_float2(acc_404[2], acc_404[3]);
                float2 q_511 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                float2 r_512 = fma_f32x2_rn_noftz(q_511, scale0_505, a_510);
                acc_404[2] = r_512.x;
                acc_404[3] = r_512.y;
                float2 a_513 = make_float2(acc_404[4], acc_404[5]);
                float2 q_514 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                float2 r_515 = fma_f32x2_rn_noftz(q_514, scale0_505, a_513);
                acc_404[4] = r_515.x;
                acc_404[5] = r_515.y;
                float2 a_516 = make_float2(acc_404[6], acc_404[7]);
                float2 q_517 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                float2 r_518 = fma_f32x2_rn_noftz(q_517, scale0_505, a_516);
                acc_404[6] = r_518.x;
                acc_404[7] = r_518.y;
                float _unpack_fp4_41[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[41 + 0]));
                    (&_unpack_fp4_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_41[0]), "=f"(_unpack_fp4_41[1]), "=f"(_unpack_fp4_41[2]), "=f"(_unpack_fp4_41[3]), "=f"(_unpack_fp4_41[4]), "=f"(_unpack_fp4_41[5]), "=f"(_unpack_fp4_41[6]), "=f"(_unpack_fp4_41[7]) : "r"(_fp4_wire_values_0[41 + 0]));
                    #endif
                }
                float2 a_519 = make_float2(acc_404[8], acc_404[9]);
                float2 q_520 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                float2 r_521 = fma_f32x2_rn_noftz(q_520, scale0_505, a_519);
                acc_404[8] = r_521.x;
                acc_404[9] = r_521.y;
                float2 a_522 = make_float2(acc_404[10], acc_404[11]);
                float2 q_523 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                float2 r_524 = fma_f32x2_rn_noftz(q_523, scale0_505, a_522);
                acc_404[10] = r_524.x;
                acc_404[11] = r_524.y;
                float2 a_525 = make_float2(acc_404[12], acc_404[13]);
                float2 q_526 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                float2 r_527 = fma_f32x2_rn_noftz(q_526, scale0_505, a_525);
                acc_404[12] = r_527.x;
                acc_404[13] = r_527.y;
                float2 a_528 = make_float2(acc_404[14], acc_404[15]);
                float2 q_529 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                float2 r_530 = fma_f32x2_rn_noftz(q_529, scale0_505, a_528);
                acc_404[14] = r_530.x;
                acc_404[15] = r_530.y;
                float _unpack_fp4_42[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[42 + 0]));
                    (&_unpack_fp4_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_42[0]), "=f"(_unpack_fp4_42[1]), "=f"(_unpack_fp4_42[2]), "=f"(_unpack_fp4_42[3]), "=f"(_unpack_fp4_42[4]), "=f"(_unpack_fp4_42[5]), "=f"(_unpack_fp4_42[6]), "=f"(_unpack_fp4_42[7]) : "r"(_fp4_wire_values_0[42 + 0]));
                    #endif
                }
                float2 a_531 = make_float2(acc_404[16], acc_404[17]);
                float2 q_532 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                float2 r_533 = fma_f32x2_rn_noftz(q_532, scale1_506, a_531);
                acc_404[16] = r_533.x;
                acc_404[17] = r_533.y;
                float2 a_534 = make_float2(acc_404[18], acc_404[19]);
                float2 q_535 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                float2 r_536 = fma_f32x2_rn_noftz(q_535, scale1_506, a_534);
                acc_404[18] = r_536.x;
                acc_404[19] = r_536.y;
                float2 a_537 = make_float2(acc_404[20], acc_404[21]);
                float2 q_538 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                float2 r_539 = fma_f32x2_rn_noftz(q_538, scale1_506, a_537);
                acc_404[20] = r_539.x;
                acc_404[21] = r_539.y;
                float2 a_540 = make_float2(acc_404[22], acc_404[23]);
                float2 q_541 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                float2 r_542 = fma_f32x2_rn_noftz(q_541, scale1_506, a_540);
                acc_404[22] = r_542.x;
                acc_404[23] = r_542.y;
                float _unpack_fp4_43[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[43 + 0]));
                    (&_unpack_fp4_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_43[0]), "=f"(_unpack_fp4_43[1]), "=f"(_unpack_fp4_43[2]), "=f"(_unpack_fp4_43[3]), "=f"(_unpack_fp4_43[4]), "=f"(_unpack_fp4_43[5]), "=f"(_unpack_fp4_43[6]), "=f"(_unpack_fp4_43[7]) : "r"(_fp4_wire_values_0[43 + 0]));
                    #endif
                }
                float2 a_543 = make_float2(acc_404[24], acc_404[25]);
                float2 q_544 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                float2 r_545 = fma_f32x2_rn_noftz(q_544, scale1_506, a_543);
                acc_404[24] = r_545.x;
                acc_404[25] = r_545.y;
                float2 a_546 = make_float2(acc_404[26], acc_404[27]);
                float2 q_547 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                float2 r_548 = fma_f32x2_rn_noftz(q_547, scale1_506, a_546);
                acc_404[26] = r_548.x;
                acc_404[27] = r_548.y;
                float2 a_549 = make_float2(acc_404[28], acc_404[29]);
                float2 q_550 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                float2 r_551 = fma_f32x2_rn_noftz(q_550, scale1_506, a_549);
                acc_404[28] = r_551.x;
                acc_404[29] = r_551.y;
                float2 a_552 = make_float2(acc_404[30], acc_404[31]);
                float2 q_553 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                float2 r_554 = fma_f32x2_rn_noftz(q_553, scale1_506, a_552);
                acc_404[30] = r_554.x;
                acc_404[31] = r_554.y;
                uint32_t _e4m3x2_to_f16x2_11;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)(_fp4_wire_scales_0[11])));
                float _cvt_f32_f16_22;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(_e4m3x2_to_f16x2_11)));
                float _cvt_f32_f16_23;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(_e4m3x2_to_f16x2_11 >> 16)));
                float _mul_22;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_cvt_f32_f16_22), "f"(g_s_4));
                float _mul_23;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_cvt_f32_f16_23), "f"(g_s_4));
                float2 scale0_555 = make_float2(_mul_22, _mul_22);
                float2 scale1_556 = make_float2(_mul_23, _mul_23);
                float _unpack_fp4_44[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[44 + 0]));
                    (&_unpack_fp4_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_44[0]), "=f"(_unpack_fp4_44[1]), "=f"(_unpack_fp4_44[2]), "=f"(_unpack_fp4_44[3]), "=f"(_unpack_fp4_44[4]), "=f"(_unpack_fp4_44[5]), "=f"(_unpack_fp4_44[6]), "=f"(_unpack_fp4_44[7]) : "r"(_fp4_wire_values_0[44 + 0]));
                    #endif
                }
                float2 a_557 = make_float2(acc_404[0], acc_404[1]);
                float2 q_558 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                float2 r_559 = fma_f32x2_rn_noftz(q_558, scale0_555, a_557);
                acc_404[0] = r_559.x;
                acc_404[1] = r_559.y;
                float2 a_560 = make_float2(acc_404[2], acc_404[3]);
                float2 q_561 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                float2 r_562 = fma_f32x2_rn_noftz(q_561, scale0_555, a_560);
                acc_404[2] = r_562.x;
                acc_404[3] = r_562.y;
                float2 a_563 = make_float2(acc_404[4], acc_404[5]);
                float2 q_564 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                float2 r_565 = fma_f32x2_rn_noftz(q_564, scale0_555, a_563);
                acc_404[4] = r_565.x;
                acc_404[5] = r_565.y;
                float2 a_566 = make_float2(acc_404[6], acc_404[7]);
                float2 q_567 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                float2 r_568 = fma_f32x2_rn_noftz(q_567, scale0_555, a_566);
                acc_404[6] = r_568.x;
                acc_404[7] = r_568.y;
                float _unpack_fp4_45[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[45 + 0]));
                    (&_unpack_fp4_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_45[0]), "=f"(_unpack_fp4_45[1]), "=f"(_unpack_fp4_45[2]), "=f"(_unpack_fp4_45[3]), "=f"(_unpack_fp4_45[4]), "=f"(_unpack_fp4_45[5]), "=f"(_unpack_fp4_45[6]), "=f"(_unpack_fp4_45[7]) : "r"(_fp4_wire_values_0[45 + 0]));
                    #endif
                }
                float2 a_569 = make_float2(acc_404[8], acc_404[9]);
                float2 q_570 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                float2 r_571 = fma_f32x2_rn_noftz(q_570, scale0_555, a_569);
                acc_404[8] = r_571.x;
                acc_404[9] = r_571.y;
                float2 a_572 = make_float2(acc_404[10], acc_404[11]);
                float2 q_573 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                float2 r_574 = fma_f32x2_rn_noftz(q_573, scale0_555, a_572);
                acc_404[10] = r_574.x;
                acc_404[11] = r_574.y;
                float2 a_575 = make_float2(acc_404[12], acc_404[13]);
                float2 q_576 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                float2 r_577 = fma_f32x2_rn_noftz(q_576, scale0_555, a_575);
                acc_404[12] = r_577.x;
                acc_404[13] = r_577.y;
                float2 a_578 = make_float2(acc_404[14], acc_404[15]);
                float2 q_579 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                float2 r_580 = fma_f32x2_rn_noftz(q_579, scale0_555, a_578);
                acc_404[14] = r_580.x;
                acc_404[15] = r_580.y;
                float _unpack_fp4_46[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[46 + 0]));
                    (&_unpack_fp4_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_46[0]), "=f"(_unpack_fp4_46[1]), "=f"(_unpack_fp4_46[2]), "=f"(_unpack_fp4_46[3]), "=f"(_unpack_fp4_46[4]), "=f"(_unpack_fp4_46[5]), "=f"(_unpack_fp4_46[6]), "=f"(_unpack_fp4_46[7]) : "r"(_fp4_wire_values_0[46 + 0]));
                    #endif
                }
                float2 a_581 = make_float2(acc_404[16], acc_404[17]);
                float2 q_582 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                float2 r_583 = fma_f32x2_rn_noftz(q_582, scale1_556, a_581);
                acc_404[16] = r_583.x;
                acc_404[17] = r_583.y;
                float2 a_584 = make_float2(acc_404[18], acc_404[19]);
                float2 q_585 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                float2 r_586 = fma_f32x2_rn_noftz(q_585, scale1_556, a_584);
                acc_404[18] = r_586.x;
                acc_404[19] = r_586.y;
                float2 a_587 = make_float2(acc_404[20], acc_404[21]);
                float2 q_588 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                float2 r_589 = fma_f32x2_rn_noftz(q_588, scale1_556, a_587);
                acc_404[20] = r_589.x;
                acc_404[21] = r_589.y;
                float2 a_590 = make_float2(acc_404[22], acc_404[23]);
                float2 q_591 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                float2 r_592 = fma_f32x2_rn_noftz(q_591, scale1_556, a_590);
                acc_404[22] = r_592.x;
                acc_404[23] = r_592.y;
                float _unpack_fp4_47[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[47 + 0]));
                    (&_unpack_fp4_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_47[0]), "=f"(_unpack_fp4_47[1]), "=f"(_unpack_fp4_47[2]), "=f"(_unpack_fp4_47[3]), "=f"(_unpack_fp4_47[4]), "=f"(_unpack_fp4_47[5]), "=f"(_unpack_fp4_47[6]), "=f"(_unpack_fp4_47[7]) : "r"(_fp4_wire_values_0[47 + 0]));
                    #endif
                }
                float2 a_593 = make_float2(acc_404[24], acc_404[25]);
                float2 q_594 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                float2 r_595 = fma_f32x2_rn_noftz(q_594, scale1_556, a_593);
                acc_404[24] = r_595.x;
                acc_404[25] = r_595.y;
                float2 a_596 = make_float2(acc_404[26], acc_404[27]);
                float2 q_597 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                float2 r_598 = fma_f32x2_rn_noftz(q_597, scale1_556, a_596);
                acc_404[26] = r_598.x;
                acc_404[27] = r_598.y;
                float2 a_599 = make_float2(acc_404[28], acc_404[29]);
                float2 q_600 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                float2 r_601 = fma_f32x2_rn_noftz(q_600, scale1_556, a_599);
                acc_404[28] = r_601.x;
                acc_404[29] = r_601.y;
                float2 a_602 = make_float2(acc_404[30], acc_404[31]);
                float2 q_603 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                float2 r_604 = fma_f32x2_rn_noftz(q_603, scale1_556, a_602);
                acc_404[30] = r_604.x;
                acc_404[31] = r_604.y;
                float am = 0.0f;
                float cur = am;
                cur = fmaxf(cur, fabsf(acc[0]));
                cur = fmaxf(cur, fabsf(acc[1]));
                cur = fmaxf(cur, fabsf(acc[2]));
                cur = fmaxf(cur, fabsf(acc[3]));
                cur = fmaxf(cur, fabsf(acc[4]));
                cur = fmaxf(cur, fabsf(acc[5]));
                cur = fmaxf(cur, fabsf(acc[6]));
                cur = fmaxf(cur, fabsf(acc[7]));
                cur = fmaxf(cur, fabsf(acc[8]));
                cur = fmaxf(cur, fabsf(acc[9]));
                cur = fmaxf(cur, fabsf(acc[10]));
                cur = fmaxf(cur, fabsf(acc[11]));
                cur = fmaxf(cur, fabsf(acc[12]));
                cur = fmaxf(cur, fabsf(acc[13]));
                cur = fmaxf(cur, fabsf(acc[14]));
                cur = fmaxf(cur, fabsf(acc[15]));
                cur = fmaxf(cur, fabsf(acc[16]));
                cur = fmaxf(cur, fabsf(acc[17]));
                cur = fmaxf(cur, fabsf(acc[18]));
                cur = fmaxf(cur, fabsf(acc[19]));
                cur = fmaxf(cur, fabsf(acc[20]));
                cur = fmaxf(cur, fabsf(acc[21]));
                cur = fmaxf(cur, fabsf(acc[22]));
                cur = fmaxf(cur, fabsf(acc[23]));
                cur = fmaxf(cur, fabsf(acc[24]));
                cur = fmaxf(cur, fabsf(acc[25]));
                cur = fmaxf(cur, fabsf(acc[26]));
                cur = fmaxf(cur, fabsf(acc[27]));
                cur = fmaxf(cur, fabsf(acc[28]));
                cur = fmaxf(cur, fabsf(acc[29]));
                cur = fmaxf(cur, fabsf(acc[30]));
                cur = fmaxf(cur, fabsf(acc[31]));
                am = cur;
                float cur_605 = am;
                cur_605 = fmaxf(cur_605, fabsf(acc_203[0]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[1]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[2]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[3]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[4]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[5]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[6]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[7]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[8]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[9]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[10]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[11]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[12]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[13]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[14]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[15]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[16]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[17]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[18]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[19]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[20]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[21]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[22]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[23]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[24]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[25]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[26]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[27]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[28]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[29]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[30]));
                cur_605 = fmaxf(cur_605, fabsf(acc_203[31]));
                am = cur_605;
                float cur_606 = am;
                cur_606 = fmaxf(cur_606, fabsf(acc_404[0]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[1]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[2]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[3]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[4]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[5]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[6]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[7]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[8]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[9]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[10]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[11]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[12]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[13]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[14]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[15]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[16]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[17]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[18]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[19]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[20]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[21]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[22]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[23]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[24]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[25]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[26]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[27]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[28]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[29]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[30]));
                cur_606 = fmaxf(cur_606, fabsf(acc_404[31]));
                am = cur_606;
                float _warp_reduce_3 = am;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 = max_noftz(_warp_reduce_3, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset));
                am = _warp_reduce_3;
                int sam_base = 4 * t + 2 * kbp;
                if (lane_1 == 0) {
                    s_am[sam_base + h] = am;
                }
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                float am2 = fmaxf(s_am[sam_base], s_am[sam_base + 1]);
                float rs2 = 1.0f;
                float _fdiv_rn_0 = __fdiv_rn(2688.0f, am2);
                float inv = 0.0f;
                if (am2 > 0.0f) {
                    inv = _fdiv_rn_0;
                }
                float rs = __fdiv_rn(am2, 2688.0f);
                rs2 = rs;
                int pp2 = lane_1 + 32 * (3 * h);
                unsigned int q_607[4];
                float bm0 = 0.0f;
                float bm1 = 0.0f;
                bm0 = fmaxf(bm0, fabsf(acc[0]));
                bm1 = fmaxf(bm1, fabsf(acc[16]));
                bm0 = fmaxf(bm0, fabsf(acc[1]));
                bm1 = fmaxf(bm1, fabsf(acc[17]));
                bm0 = fmaxf(bm0, fabsf(acc[2]));
                bm1 = fmaxf(bm1, fabsf(acc[18]));
                bm0 = fmaxf(bm0, fabsf(acc[3]));
                bm1 = fmaxf(bm1, fabsf(acc[19]));
                bm0 = fmaxf(bm0, fabsf(acc[4]));
                bm1 = fmaxf(bm1, fabsf(acc[20]));
                bm0 = fmaxf(bm0, fabsf(acc[5]));
                bm1 = fmaxf(bm1, fabsf(acc[21]));
                bm0 = fmaxf(bm0, fabsf(acc[6]));
                bm1 = fmaxf(bm1, fabsf(acc[22]));
                bm0 = fmaxf(bm0, fabsf(acc[7]));
                bm1 = fmaxf(bm1, fabsf(acc[23]));
                bm0 = fmaxf(bm0, fabsf(acc[8]));
                bm1 = fmaxf(bm1, fabsf(acc[24]));
                bm0 = fmaxf(bm0, fabsf(acc[9]));
                bm1 = fmaxf(bm1, fabsf(acc[25]));
                bm0 = fmaxf(bm0, fabsf(acc[10]));
                bm1 = fmaxf(bm1, fabsf(acc[26]));
                bm0 = fmaxf(bm0, fabsf(acc[11]));
                bm1 = fmaxf(bm1, fabsf(acc[27]));
                bm0 = fmaxf(bm0, fabsf(acc[12]));
                bm1 = fmaxf(bm1, fabsf(acc[28]));
                bm0 = fmaxf(bm0, fabsf(acc[13]));
                bm1 = fmaxf(bm1, fabsf(acc[29]));
                bm0 = fmaxf(bm0, fabsf(acc[14]));
                bm1 = fmaxf(bm1, fabsf(acc[30]));
                bm0 = fmaxf(bm0, fabsf(acc[15]));
                bm1 = fmaxf(bm1, fabsf(acc[31]));
                float _fdiv_4 = __fdiv_rn(mul_f32_rn_noftz(bm1, inv), 6.0f);
                float _fdiv_5 = __fdiv_rn(mul_f32_rn_noftz(bm0, inv), 6.0f);
                uint16_t _e4m3x2_f32_0;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_4), "f"(_fdiv_5));
                uint32_t _e4m3x2_to_f16x2_12;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_12) : "h"((uint16_t)(_e4m3x2_f32_0)));
                float _cvt_f32_f16_24;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_24) : "h"((uint16_t)(_e4m3x2_to_f16x2_12)));
                float _cvt_f32_f16_25;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_25) : "h"((uint16_t)(_e4m3x2_to_f16x2_12 >> 16)));
                float _fdiv_rn_1 = __fdiv_rn(inv, _cvt_f32_f16_24);
                float _fdiv_rn_2 = __fdiv_rn(inv, _cvt_f32_f16_25);
                float m0 = 0.0f;
                float m1 = 0.0f;
                if (_cvt_f32_f16_24 > 0.0f) {
                    m0 = _fdiv_rn_1;
                }
                if (_cvt_f32_f16_25 > 0.0f) {
                    m1 = _fdiv_rn_2;
                }
                float2 mm = make_float2(m0, m0);
                float2 pair = make_float2(acc[0], acc[1]);
                float2 g;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&pair), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_pair_0;
                uint16_t _e2m1_pair_h_6;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_6) : "f"(g.x), "f"(g.y));
                _fp4_pair_0 = (uint32_t)_e2m1_pair_h_6;
                float2 pair_608 = make_float2(acc[2], acc[3]);
                float2 g_609;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_609) : "l"(*(const unsigned long long*)&pair_608), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_pair_1;
                uint16_t _e2m1_pair_h_7;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_7) : "f"(g_609.x), "f"(g_609.y));
                _fp4_pair_1 = (uint32_t)_e2m1_pair_h_7;
                float2 pair_610 = make_float2(acc[4], acc[5]);
                float2 g_611;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_611) : "l"(*(const unsigned long long*)&pair_610), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_pair_2;
                uint16_t _e2m1_pair_h_8;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_8) : "f"(g_611.x), "f"(g_611.y));
                _fp4_pair_2 = (uint32_t)_e2m1_pair_h_8;
                float2 pair_612 = make_float2(acc[6], acc[7]);
                float2 g_613;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_613) : "l"(*(const unsigned long long*)&pair_612), "l"(*(const unsigned long long*)&mm));
                uint32_t _fp4_e2m1_pack_pair_9;
                uint16_t _e2m1_pair_h_10;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(g_613.x), "f"(g_613.y));
                _fp4_e2m1_pack_pair_9 = (uint32_t)_e2m1_pair_h_10;
                unsigned int word = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_e2m1_pack_pair_9 << 24;
                unsigned int carry = (word & 2004318071) + 2004318071;
                q_607[0] = word & (carry | 2004318071);
                float2 mm_614 = make_float2(m0, m0);
                float2 pair_615 = make_float2(acc[8], acc[9]);
                float2 g_616;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_616) : "l"(*(const unsigned long long*)&pair_615), "l"(*(const unsigned long long*)&mm_614));
                uint32_t _fp4_pair_3;
                uint16_t _e2m1_pair_h_11;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(g_616.x), "f"(g_616.y));
                _fp4_pair_3 = (uint32_t)_e2m1_pair_h_11;
                float2 pair_617 = make_float2(acc[10], acc[11]);
                float2 g_618;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_618) : "l"(*(const unsigned long long*)&pair_617), "l"(*(const unsigned long long*)&mm_614));
                uint32_t _fp4_pair_4;
                uint16_t _e2m1_pair_h_12;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(g_618.x), "f"(g_618.y));
                _fp4_pair_4 = (uint32_t)_e2m1_pair_h_12;
                float2 pair_619 = make_float2(acc[12], acc[13]);
                float2 g_620;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_620) : "l"(*(const unsigned long long*)&pair_619), "l"(*(const unsigned long long*)&mm_614));
                uint32_t _fp4_pair_5;
                uint16_t _e2m1_pair_h_13;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(g_620.x), "f"(g_620.y));
                _fp4_pair_5 = (uint32_t)_e2m1_pair_h_13;
                float2 pair_621 = make_float2(acc[14], acc[15]);
                float2 g_622;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_622) : "l"(*(const unsigned long long*)&pair_621), "l"(*(const unsigned long long*)&mm_614));
                uint32_t _fp4_e2m1_pack_pair_14;
                uint16_t _e2m1_pair_h_15;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_622.x), "f"(g_622.y));
                _fp4_e2m1_pack_pair_14 = (uint32_t)_e2m1_pair_h_15;
                unsigned int word_623 = _fp4_pair_3 | _fp4_pair_4 << 8 | _fp4_pair_5 << 16 | _fp4_e2m1_pack_pair_14 << 24;
                unsigned int carry_624 = (word_623 & 2004318071) + 2004318071;
                q_607[1] = word_623 & (carry_624 | 2004318071);
                float2 mm_625 = make_float2(m1, m1);
                float2 pair_626 = make_float2(acc[16], acc[17]);
                float2 g_627;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_627) : "l"(*(const unsigned long long*)&pair_626), "l"(*(const unsigned long long*)&mm_625));
                uint32_t _fp4_pair_6;
                uint16_t _e2m1_pair_h_16;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_627.x), "f"(g_627.y));
                _fp4_pair_6 = (uint32_t)_e2m1_pair_h_16;
                float2 pair_628 = make_float2(acc[18], acc[19]);
                float2 g_629;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_629) : "l"(*(const unsigned long long*)&pair_628), "l"(*(const unsigned long long*)&mm_625));
                uint32_t _fp4_pair_7;
                uint16_t _e2m1_pair_h_17;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(g_629.x), "f"(g_629.y));
                _fp4_pair_7 = (uint32_t)_e2m1_pair_h_17;
                float2 pair_630 = make_float2(acc[20], acc[21]);
                float2 g_631;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_631) : "l"(*(const unsigned long long*)&pair_630), "l"(*(const unsigned long long*)&mm_625));
                uint32_t _fp4_pair_8;
                uint16_t _e2m1_pair_h_18;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(g_631.x), "f"(g_631.y));
                _fp4_pair_8 = (uint32_t)_e2m1_pair_h_18;
                float2 pair_632 = make_float2(acc[22], acc[23]);
                float2 g_633;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_633) : "l"(*(const unsigned long long*)&pair_632), "l"(*(const unsigned long long*)&mm_625));
                uint32_t _fp4_e2m1_pack_pair_19;
                uint16_t _e2m1_pair_h_20;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_633.x), "f"(g_633.y));
                _fp4_e2m1_pack_pair_19 = (uint32_t)_e2m1_pair_h_20;
                unsigned int word_634 = _fp4_pair_6 | _fp4_pair_7 << 8 | _fp4_pair_8 << 16 | _fp4_e2m1_pack_pair_19 << 24;
                unsigned int carry_635 = (word_634 & 2004318071) + 2004318071;
                q_607[2] = word_634 & (carry_635 | 2004318071);
                float2 mm_636 = make_float2(m1, m1);
                float2 pair_637 = make_float2(acc[24], acc[25]);
                float2 g_638;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_638) : "l"(*(const unsigned long long*)&pair_637), "l"(*(const unsigned long long*)&mm_636));
                uint32_t _fp4_pair_9;
                uint16_t _e2m1_pair_h_21;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_638.x), "f"(g_638.y));
                _fp4_pair_9 = (uint32_t)_e2m1_pair_h_21;
                float2 pair_639 = make_float2(acc[26], acc[27]);
                float2 g_640;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_640) : "l"(*(const unsigned long long*)&pair_639), "l"(*(const unsigned long long*)&mm_636));
                uint32_t _fp4_pair_10;
                uint16_t _e2m1_pair_h_22;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(g_640.x), "f"(g_640.y));
                _fp4_pair_10 = (uint32_t)_e2m1_pair_h_22;
                float2 pair_641 = make_float2(acc[28], acc[29]);
                float2 g_642;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_642) : "l"(*(const unsigned long long*)&pair_641), "l"(*(const unsigned long long*)&mm_636));
                uint32_t _fp4_pair_11;
                uint16_t _e2m1_pair_h_23;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(g_642.x), "f"(g_642.y));
                _fp4_pair_11 = (uint32_t)_e2m1_pair_h_23;
                float2 pair_643 = make_float2(acc[30], acc[31]);
                float2 g_644;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_644) : "l"(*(const unsigned long long*)&pair_643), "l"(*(const unsigned long long*)&mm_636));
                uint32_t _fp4_e2m1_pack_pair_24;
                uint16_t _e2m1_pair_h_25;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_644.x), "f"(g_644.y));
                _fp4_e2m1_pack_pair_24 = (uint32_t)_e2m1_pair_h_25;
                unsigned int word_645 = _fp4_pair_9 | _fp4_pair_10 << 8 | _fp4_pair_11 << 16 | _fp4_e2m1_pack_pair_24 << 24;
                unsigned int carry_646 = (word_645 & 2004318071) + 2004318071;
                q_607[3] = word_645 & (carry_646 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2)), "r"(q_607[0]), "r"(q_607[1]), "r"(q_607[2]), "r"(q_607[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2)), "h"((uint16_t)(_e4m3x2_f32_0)) : "memory");
                float d[32] = {0};
                uint32_t _e4m3x2_to_f16x2_13;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_13) : "h"((uint16_t)(_e4m3x2_f32_0)));
                float _cvt_f32_f16_26;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_26) : "h"((uint16_t)(_e4m3x2_to_f16x2_13)));
                float _cvt_f32_f16_27;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_27) : "h"((uint16_t)(_e4m3x2_to_f16x2_13 >> 16)));
                float _mul_24;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(_cvt_f32_f16_26), "f"(rs2));
                float _mul_25;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(_cvt_f32_f16_27), "f"(rs2));
                float2 scale0_647 = make_float2(_mul_24, _mul_24);
                float2 scale1_648 = make_float2(_mul_25, _mul_25);
                float _unpack_fp4_48[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_607[0]));
                    (&_unpack_fp4_48[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_48[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_48[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_48[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_48[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_48[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_48[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_48[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_48[0]), "=f"(_unpack_fp4_48[1]), "=f"(_unpack_fp4_48[2]), "=f"(_unpack_fp4_48[3]), "=f"(_unpack_fp4_48[4]), "=f"(_unpack_fp4_48[5]), "=f"(_unpack_fp4_48[6]), "=f"(_unpack_fp4_48[7]) : "r"(q_607[0]));
                    #endif
                }
                float2 a_649 = make_float2(d[0], d[1]);
                float2 q_650 = make_float2(_unpack_fp4_48[0], _unpack_fp4_48[1]);
                float2 r_651 = fma_f32x2_rn_noftz(q_650, scale0_647, a_649);
                d[0] = r_651.x;
                d[1] = r_651.y;
                float2 a_652 = make_float2(d[2], d[3]);
                float2 q_653 = make_float2(_unpack_fp4_48[2], _unpack_fp4_48[3]);
                float2 r_654 = fma_f32x2_rn_noftz(q_653, scale0_647, a_652);
                d[2] = r_654.x;
                d[3] = r_654.y;
                float2 a_655 = make_float2(d[4], d[5]);
                float2 q_656 = make_float2(_unpack_fp4_48[4], _unpack_fp4_48[5]);
                float2 r_657 = fma_f32x2_rn_noftz(q_656, scale0_647, a_655);
                d[4] = r_657.x;
                d[5] = r_657.y;
                float2 a_658 = make_float2(d[6], d[7]);
                float2 q_659 = make_float2(_unpack_fp4_48[6], _unpack_fp4_48[7]);
                float2 r_660 = fma_f32x2_rn_noftz(q_659, scale0_647, a_658);
                d[6] = r_660.x;
                d[7] = r_660.y;
                float _unpack_fp4_49[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_607[1 + 0]));
                    (&_unpack_fp4_49[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_49[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_49[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_49[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_49[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_49[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_49[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_49[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_49[0]), "=f"(_unpack_fp4_49[1]), "=f"(_unpack_fp4_49[2]), "=f"(_unpack_fp4_49[3]), "=f"(_unpack_fp4_49[4]), "=f"(_unpack_fp4_49[5]), "=f"(_unpack_fp4_49[6]), "=f"(_unpack_fp4_49[7]) : "r"(q_607[1 + 0]));
                    #endif
                }
                float2 a_661 = make_float2(d[8], d[9]);
                float2 q_662 = make_float2(_unpack_fp4_49[0], _unpack_fp4_49[1]);
                float2 r_663 = fma_f32x2_rn_noftz(q_662, scale0_647, a_661);
                d[8] = r_663.x;
                d[9] = r_663.y;
                float2 a_664 = make_float2(d[10], d[11]);
                float2 q_665 = make_float2(_unpack_fp4_49[2], _unpack_fp4_49[3]);
                float2 r_666 = fma_f32x2_rn_noftz(q_665, scale0_647, a_664);
                d[10] = r_666.x;
                d[11] = r_666.y;
                float2 a_667 = make_float2(d[12], d[13]);
                float2 q_668 = make_float2(_unpack_fp4_49[4], _unpack_fp4_49[5]);
                float2 r_669 = fma_f32x2_rn_noftz(q_668, scale0_647, a_667);
                d[12] = r_669.x;
                d[13] = r_669.y;
                float2 a_670 = make_float2(d[14], d[15]);
                float2 q_671 = make_float2(_unpack_fp4_49[6], _unpack_fp4_49[7]);
                float2 r_672 = fma_f32x2_rn_noftz(q_671, scale0_647, a_670);
                d[14] = r_672.x;
                d[15] = r_672.y;
                float _unpack_fp4_50[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_607[2 + 0]));
                    (&_unpack_fp4_50[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_50[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_50[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_50[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_50[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_50[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_50[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_50[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_50[0]), "=f"(_unpack_fp4_50[1]), "=f"(_unpack_fp4_50[2]), "=f"(_unpack_fp4_50[3]), "=f"(_unpack_fp4_50[4]), "=f"(_unpack_fp4_50[5]), "=f"(_unpack_fp4_50[6]), "=f"(_unpack_fp4_50[7]) : "r"(q_607[2 + 0]));
                    #endif
                }
                float2 a_673 = make_float2(d[16], d[17]);
                float2 q_674 = make_float2(_unpack_fp4_50[0], _unpack_fp4_50[1]);
                float2 r_675 = fma_f32x2_rn_noftz(q_674, scale1_648, a_673);
                d[16] = r_675.x;
                d[17] = r_675.y;
                float2 a_676 = make_float2(d[18], d[19]);
                float2 q_677 = make_float2(_unpack_fp4_50[2], _unpack_fp4_50[3]);
                float2 r_678 = fma_f32x2_rn_noftz(q_677, scale1_648, a_676);
                d[18] = r_678.x;
                d[19] = r_678.y;
                float2 a_679 = make_float2(d[20], d[21]);
                float2 q_680 = make_float2(_unpack_fp4_50[4], _unpack_fp4_50[5]);
                float2 r_681 = fma_f32x2_rn_noftz(q_680, scale1_648, a_679);
                d[20] = r_681.x;
                d[21] = r_681.y;
                float2 a_682 = make_float2(d[22], d[23]);
                float2 q_683 = make_float2(_unpack_fp4_50[6], _unpack_fp4_50[7]);
                float2 r_684 = fma_f32x2_rn_noftz(q_683, scale1_648, a_682);
                d[22] = r_684.x;
                d[23] = r_684.y;
                float _unpack_fp4_51[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_607[3 + 0]));
                    (&_unpack_fp4_51[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_51[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_51[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_51[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_51[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_51[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_51[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_51[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_51[0]), "=f"(_unpack_fp4_51[1]), "=f"(_unpack_fp4_51[2]), "=f"(_unpack_fp4_51[3]), "=f"(_unpack_fp4_51[4]), "=f"(_unpack_fp4_51[5]), "=f"(_unpack_fp4_51[6]), "=f"(_unpack_fp4_51[7]) : "r"(q_607[3 + 0]));
                    #endif
                }
                float2 a_685 = make_float2(d[24], d[25]);
                float2 q_686 = make_float2(_unpack_fp4_51[0], _unpack_fp4_51[1]);
                float2 r_687 = fma_f32x2_rn_noftz(q_686, scale1_648, a_685);
                d[24] = r_687.x;
                d[25] = r_687.y;
                float2 a_688 = make_float2(d[26], d[27]);
                float2 q_689 = make_float2(_unpack_fp4_51[2], _unpack_fp4_51[3]);
                float2 r_690 = fma_f32x2_rn_noftz(q_689, scale1_648, a_688);
                d[26] = r_690.x;
                d[27] = r_690.y;
                float2 a_691 = make_float2(d[28], d[29]);
                float2 q_692 = make_float2(_unpack_fp4_51[4], _unpack_fp4_51[5]);
                float2 r_693 = fma_f32x2_rn_noftz(q_692, scale1_648, a_691);
                d[28] = r_693.x;
                d[29] = r_693.y;
                float2 a_694 = make_float2(d[30], d[31]);
                float2 q_695 = make_float2(_unpack_fp4_51[6], _unpack_fp4_51[7]);
                float2 r_696 = fma_f32x2_rn_noftz(q_695, scale1_648, a_694);
                d[30] = r_696.x;
                d[31] = r_696.y;
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
                int c = lane_1;
                int l = c >> 2;
                unsigned int w4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l + (c & 3 ^ l >> 1 & 3)))));
                long long idx = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c));
                {
                    int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx) = _iv4;
                }
                int c_697 = lane_1 + 32;
                int l_698 = c_697 >> 2;
                unsigned int w4_699[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_699[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_699[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_699[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_699[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_698 + (c_697 & 3 ^ l_698 >> 1 & 3)))));
                long long idx_700 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_697));
                {
                    int4 _iv4 = make_int4(w4_699[0 + 0], w4_699[0 + 1], w4_699[0 + 2], w4_699[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_700) = _iv4;
                }
                int c_701 = lane_1 + 64;
                int l_702 = c_701 >> 2;
                unsigned int w4_703[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_703[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_703[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_703[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_703[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_702 + (c_701 & 3 ^ l_702 >> 1 & 3)))));
                long long idx_704 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_701));
                {
                    int4 _iv4 = make_int4(w4_703[0 + 0], w4_703[0 + 1], w4_703[0 + 2], w4_703[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_704) = _iv4;
                }
                int c_705 = lane_1 + 96;
                int l_706 = c_705 >> 2;
                unsigned int w4_707[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_707[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_707[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_707[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_707[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_706 + (c_705 & 3 ^ l_706 >> 1 & 3)))));
                long long idx_708 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_705));
                {
                    int4 _iv4 = make_int4(w4_707[0 + 0], w4_707[0 + 1], w4_707[0 + 2], w4_707[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_708) = _iv4;
                }
                __syncwarp();
                int pp2_709 = lane_1 + 32 * (3 * h + 1);
                unsigned int q_710[4];
                float bm0_711 = 0.0f;
                float bm1_712 = 0.0f;
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[0]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[16]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[1]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[17]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[2]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[18]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[3]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[19]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[4]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[20]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[5]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[21]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[6]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[22]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[7]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[23]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[8]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[24]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[9]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[25]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[10]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[26]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[11]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[27]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[12]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[28]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[13]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[29]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[14]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[30]));
                bm0_711 = fmaxf(bm0_711, fabsf(acc_203[15]));
                bm1_712 = fmaxf(bm1_712, fabsf(acc_203[31]));
                float _fdiv_26 = __fdiv_rn(mul_f32_rn_noftz(bm1_712, inv), 6.0f);
                float _fdiv_27 = __fdiv_rn(mul_f32_rn_noftz(bm0_711, inv), 6.0f);
                uint16_t _e4m3x2_f32_1;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fdiv_26), "f"(_fdiv_27));
                uint32_t _e4m3x2_to_f16x2_14;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_14) : "h"((uint16_t)(_e4m3x2_f32_1)));
                float _cvt_f32_f16_28;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_28) : "h"((uint16_t)(_e4m3x2_to_f16x2_14)));
                float _cvt_f32_f16_29;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_29) : "h"((uint16_t)(_e4m3x2_to_f16x2_14 >> 16)));
                float _fdiv_rn_3 = __fdiv_rn(inv, _cvt_f32_f16_28);
                float _fdiv_rn_4 = __fdiv_rn(inv, _cvt_f32_f16_29);
                float m0_713 = 0.0f;
                float m1_714 = 0.0f;
                if (_cvt_f32_f16_28 > 0.0f) {
                    m0_713 = _fdiv_rn_3;
                }
                if (_cvt_f32_f16_29 > 0.0f) {
                    m1_714 = _fdiv_rn_4;
                }
                float2 mm_715 = make_float2(m0_713, m0_713);
                float2 pair_716 = make_float2(acc_203[0], acc_203[1]);
                float2 g_717;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_717) : "l"(*(const unsigned long long*)&pair_716), "l"(*(const unsigned long long*)&mm_715));
                uint32_t _fp4_pair_12;
                uint16_t _e2m1_pair_h_28;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_28) : "f"(g_717.x), "f"(g_717.y));
                _fp4_pair_12 = (uint32_t)_e2m1_pair_h_28;
                float2 pair_718 = make_float2(acc_203[2], acc_203[3]);
                float2 g_719;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_719) : "l"(*(const unsigned long long*)&pair_718), "l"(*(const unsigned long long*)&mm_715));
                uint32_t _fp4_pair_13;
                uint16_t _e2m1_pair_h_29;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_29) : "f"(g_719.x), "f"(g_719.y));
                _fp4_pair_13 = (uint32_t)_e2m1_pair_h_29;
                float2 pair_720 = make_float2(acc_203[4], acc_203[5]);
                float2 g_721;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_721) : "l"(*(const unsigned long long*)&pair_720), "l"(*(const unsigned long long*)&mm_715));
                uint32_t _fp4_pair_14;
                uint16_t _e2m1_pair_h_30;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_30) : "f"(g_721.x), "f"(g_721.y));
                _fp4_pair_14 = (uint32_t)_e2m1_pair_h_30;
                float2 pair_722 = make_float2(acc_203[6], acc_203[7]);
                float2 g_723;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_723) : "l"(*(const unsigned long long*)&pair_722), "l"(*(const unsigned long long*)&mm_715));
                uint32_t _fp4_e2m1_pack_pair_31;
                uint16_t _e2m1_pair_h_32;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(g_723.x), "f"(g_723.y));
                _fp4_e2m1_pack_pair_31 = (uint32_t)_e2m1_pair_h_32;
                unsigned int word_724 = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_e2m1_pack_pair_31 << 24;
                unsigned int carry_725 = (word_724 & 2004318071) + 2004318071;
                q_710[0] = word_724 & (carry_725 | 2004318071);
                float2 mm_726 = make_float2(m0_713, m0_713);
                float2 pair_727 = make_float2(acc_203[8], acc_203[9]);
                float2 g_728;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_728) : "l"(*(const unsigned long long*)&pair_727), "l"(*(const unsigned long long*)&mm_726));
                uint32_t _fp4_pair_15;
                uint16_t _e2m1_pair_h_33;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(g_728.x), "f"(g_728.y));
                _fp4_pair_15 = (uint32_t)_e2m1_pair_h_33;
                float2 pair_729 = make_float2(acc_203[10], acc_203[11]);
                float2 g_730;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_730) : "l"(*(const unsigned long long*)&pair_729), "l"(*(const unsigned long long*)&mm_726));
                uint32_t _fp4_pair_16;
                uint16_t _e2m1_pair_h_34;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_34) : "f"(g_730.x), "f"(g_730.y));
                _fp4_pair_16 = (uint32_t)_e2m1_pair_h_34;
                float2 pair_731 = make_float2(acc_203[12], acc_203[13]);
                float2 g_732;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_732) : "l"(*(const unsigned long long*)&pair_731), "l"(*(const unsigned long long*)&mm_726));
                uint32_t _fp4_pair_17;
                uint16_t _e2m1_pair_h_35;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_35) : "f"(g_732.x), "f"(g_732.y));
                _fp4_pair_17 = (uint32_t)_e2m1_pair_h_35;
                float2 pair_733 = make_float2(acc_203[14], acc_203[15]);
                float2 g_734;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_734) : "l"(*(const unsigned long long*)&pair_733), "l"(*(const unsigned long long*)&mm_726));
                uint32_t _fp4_e2m1_pack_pair_36;
                uint16_t _e2m1_pair_h_37;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(g_734.x), "f"(g_734.y));
                _fp4_e2m1_pack_pair_36 = (uint32_t)_e2m1_pair_h_37;
                unsigned int word_735 = _fp4_pair_15 | _fp4_pair_16 << 8 | _fp4_pair_17 << 16 | _fp4_e2m1_pack_pair_36 << 24;
                unsigned int carry_736 = (word_735 & 2004318071) + 2004318071;
                q_710[1] = word_735 & (carry_736 | 2004318071);
                float2 mm_737 = make_float2(m1_714, m1_714);
                float2 pair_738 = make_float2(acc_203[16], acc_203[17]);
                float2 g_739;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_739) : "l"(*(const unsigned long long*)&pair_738), "l"(*(const unsigned long long*)&mm_737));
                uint32_t _fp4_pair_18;
                uint16_t _e2m1_pair_h_38;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(g_739.x), "f"(g_739.y));
                _fp4_pair_18 = (uint32_t)_e2m1_pair_h_38;
                float2 pair_740 = make_float2(acc_203[18], acc_203[19]);
                float2 g_741;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_741) : "l"(*(const unsigned long long*)&pair_740), "l"(*(const unsigned long long*)&mm_737));
                uint32_t _fp4_pair_19;
                uint16_t _e2m1_pair_h_39;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_39) : "f"(g_741.x), "f"(g_741.y));
                _fp4_pair_19 = (uint32_t)_e2m1_pair_h_39;
                float2 pair_742 = make_float2(acc_203[20], acc_203[21]);
                float2 g_743;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_743) : "l"(*(const unsigned long long*)&pair_742), "l"(*(const unsigned long long*)&mm_737));
                uint32_t _fp4_pair_20;
                uint16_t _e2m1_pair_h_40;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_40) : "f"(g_743.x), "f"(g_743.y));
                _fp4_pair_20 = (uint32_t)_e2m1_pair_h_40;
                float2 pair_744 = make_float2(acc_203[22], acc_203[23]);
                float2 g_745;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_745) : "l"(*(const unsigned long long*)&pair_744), "l"(*(const unsigned long long*)&mm_737));
                uint32_t _fp4_e2m1_pack_pair_41;
                uint16_t _e2m1_pair_h_42;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(g_745.x), "f"(g_745.y));
                _fp4_e2m1_pack_pair_41 = (uint32_t)_e2m1_pair_h_42;
                unsigned int word_746 = _fp4_pair_18 | _fp4_pair_19 << 8 | _fp4_pair_20 << 16 | _fp4_e2m1_pack_pair_41 << 24;
                unsigned int carry_747 = (word_746 & 2004318071) + 2004318071;
                q_710[2] = word_746 & (carry_747 | 2004318071);
                float2 mm_748 = make_float2(m1_714, m1_714);
                float2 pair_749 = make_float2(acc_203[24], acc_203[25]);
                float2 g_750;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_750) : "l"(*(const unsigned long long*)&pair_749), "l"(*(const unsigned long long*)&mm_748));
                uint32_t _fp4_pair_21;
                uint16_t _e2m1_pair_h_43;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(g_750.x), "f"(g_750.y));
                _fp4_pair_21 = (uint32_t)_e2m1_pair_h_43;
                float2 pair_751 = make_float2(acc_203[26], acc_203[27]);
                float2 g_752;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_752) : "l"(*(const unsigned long long*)&pair_751), "l"(*(const unsigned long long*)&mm_748));
                uint32_t _fp4_pair_22;
                uint16_t _e2m1_pair_h_44;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_44) : "f"(g_752.x), "f"(g_752.y));
                _fp4_pair_22 = (uint32_t)_e2m1_pair_h_44;
                float2 pair_753 = make_float2(acc_203[28], acc_203[29]);
                float2 g_754;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_754) : "l"(*(const unsigned long long*)&pair_753), "l"(*(const unsigned long long*)&mm_748));
                uint32_t _fp4_pair_23;
                uint16_t _e2m1_pair_h_45;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(g_754.x), "f"(g_754.y));
                _fp4_pair_23 = (uint32_t)_e2m1_pair_h_45;
                float2 pair_755 = make_float2(acc_203[30], acc_203[31]);
                float2 g_756;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_756) : "l"(*(const unsigned long long*)&pair_755), "l"(*(const unsigned long long*)&mm_748));
                uint32_t _fp4_e2m1_pack_pair_46;
                uint16_t _e2m1_pair_h_47;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(g_756.x), "f"(g_756.y));
                _fp4_e2m1_pack_pair_46 = (uint32_t)_e2m1_pair_h_47;
                unsigned int word_757 = _fp4_pair_21 | _fp4_pair_22 << 8 | _fp4_pair_23 << 16 | _fp4_e2m1_pack_pair_46 << 24;
                unsigned int carry_758 = (word_757 & 2004318071) + 2004318071;
                q_710[3] = word_757 & (carry_758 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_709)), "r"(q_710[0]), "r"(q_710[1]), "r"(q_710[2]), "r"(q_710[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2_709)), "h"((uint16_t)(_e4m3x2_f32_1)) : "memory");
                float d_759[32] = {0};
                uint32_t _e4m3x2_to_f16x2_15;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_15) : "h"((uint16_t)(_e4m3x2_f32_1)));
                float _cvt_f32_f16_30;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_30) : "h"((uint16_t)(_e4m3x2_to_f16x2_15)));
                float _cvt_f32_f16_31;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_31) : "h"((uint16_t)(_e4m3x2_to_f16x2_15 >> 16)));
                float _mul_26;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_cvt_f32_f16_30), "f"(rs2));
                float _mul_27;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_cvt_f32_f16_31), "f"(rs2));
                float2 scale0_760 = make_float2(_mul_26, _mul_26);
                float2 scale1_761 = make_float2(_mul_27, _mul_27);
                float _unpack_fp4_52[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_710[0]));
                    (&_unpack_fp4_52[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_52[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_52[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_52[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_52[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_52[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_52[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_52[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_52[0]), "=f"(_unpack_fp4_52[1]), "=f"(_unpack_fp4_52[2]), "=f"(_unpack_fp4_52[3]), "=f"(_unpack_fp4_52[4]), "=f"(_unpack_fp4_52[5]), "=f"(_unpack_fp4_52[6]), "=f"(_unpack_fp4_52[7]) : "r"(q_710[0]));
                    #endif
                }
                float2 a_762 = make_float2(d_759[0], d_759[1]);
                float2 q_763 = make_float2(_unpack_fp4_52[0], _unpack_fp4_52[1]);
                float2 r_764 = fma_f32x2_rn_noftz(q_763, scale0_760, a_762);
                d_759[0] = r_764.x;
                d_759[1] = r_764.y;
                float2 a_765 = make_float2(d_759[2], d_759[3]);
                float2 q_766 = make_float2(_unpack_fp4_52[2], _unpack_fp4_52[3]);
                float2 r_767 = fma_f32x2_rn_noftz(q_766, scale0_760, a_765);
                d_759[2] = r_767.x;
                d_759[3] = r_767.y;
                float2 a_768 = make_float2(d_759[4], d_759[5]);
                float2 q_769 = make_float2(_unpack_fp4_52[4], _unpack_fp4_52[5]);
                float2 r_770 = fma_f32x2_rn_noftz(q_769, scale0_760, a_768);
                d_759[4] = r_770.x;
                d_759[5] = r_770.y;
                float2 a_771 = make_float2(d_759[6], d_759[7]);
                float2 q_772 = make_float2(_unpack_fp4_52[6], _unpack_fp4_52[7]);
                float2 r_773 = fma_f32x2_rn_noftz(q_772, scale0_760, a_771);
                d_759[6] = r_773.x;
                d_759[7] = r_773.y;
                float _unpack_fp4_53[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_710[1 + 0]));
                    (&_unpack_fp4_53[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_53[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_53[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_53[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_53[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_53[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_53[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_53[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_53[0]), "=f"(_unpack_fp4_53[1]), "=f"(_unpack_fp4_53[2]), "=f"(_unpack_fp4_53[3]), "=f"(_unpack_fp4_53[4]), "=f"(_unpack_fp4_53[5]), "=f"(_unpack_fp4_53[6]), "=f"(_unpack_fp4_53[7]) : "r"(q_710[1 + 0]));
                    #endif
                }
                float2 a_774 = make_float2(d_759[8], d_759[9]);
                float2 q_775 = make_float2(_unpack_fp4_53[0], _unpack_fp4_53[1]);
                float2 r_776 = fma_f32x2_rn_noftz(q_775, scale0_760, a_774);
                d_759[8] = r_776.x;
                d_759[9] = r_776.y;
                float2 a_777 = make_float2(d_759[10], d_759[11]);
                float2 q_778 = make_float2(_unpack_fp4_53[2], _unpack_fp4_53[3]);
                float2 r_779 = fma_f32x2_rn_noftz(q_778, scale0_760, a_777);
                d_759[10] = r_779.x;
                d_759[11] = r_779.y;
                float2 a_780 = make_float2(d_759[12], d_759[13]);
                float2 q_781 = make_float2(_unpack_fp4_53[4], _unpack_fp4_53[5]);
                float2 r_782 = fma_f32x2_rn_noftz(q_781, scale0_760, a_780);
                d_759[12] = r_782.x;
                d_759[13] = r_782.y;
                float2 a_783 = make_float2(d_759[14], d_759[15]);
                float2 q_784 = make_float2(_unpack_fp4_53[6], _unpack_fp4_53[7]);
                float2 r_785 = fma_f32x2_rn_noftz(q_784, scale0_760, a_783);
                d_759[14] = r_785.x;
                d_759[15] = r_785.y;
                float _unpack_fp4_54[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_710[2 + 0]));
                    (&_unpack_fp4_54[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_54[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_54[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_54[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_54[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_54[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_54[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_54[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_54[0]), "=f"(_unpack_fp4_54[1]), "=f"(_unpack_fp4_54[2]), "=f"(_unpack_fp4_54[3]), "=f"(_unpack_fp4_54[4]), "=f"(_unpack_fp4_54[5]), "=f"(_unpack_fp4_54[6]), "=f"(_unpack_fp4_54[7]) : "r"(q_710[2 + 0]));
                    #endif
                }
                float2 a_786 = make_float2(d_759[16], d_759[17]);
                float2 q_787 = make_float2(_unpack_fp4_54[0], _unpack_fp4_54[1]);
                float2 r_788 = fma_f32x2_rn_noftz(q_787, scale1_761, a_786);
                d_759[16] = r_788.x;
                d_759[17] = r_788.y;
                float2 a_789 = make_float2(d_759[18], d_759[19]);
                float2 q_790 = make_float2(_unpack_fp4_54[2], _unpack_fp4_54[3]);
                float2 r_791 = fma_f32x2_rn_noftz(q_790, scale1_761, a_789);
                d_759[18] = r_791.x;
                d_759[19] = r_791.y;
                float2 a_792 = make_float2(d_759[20], d_759[21]);
                float2 q_793 = make_float2(_unpack_fp4_54[4], _unpack_fp4_54[5]);
                float2 r_794 = fma_f32x2_rn_noftz(q_793, scale1_761, a_792);
                d_759[20] = r_794.x;
                d_759[21] = r_794.y;
                float2 a_795 = make_float2(d_759[22], d_759[23]);
                float2 q_796 = make_float2(_unpack_fp4_54[6], _unpack_fp4_54[7]);
                float2 r_797 = fma_f32x2_rn_noftz(q_796, scale1_761, a_795);
                d_759[22] = r_797.x;
                d_759[23] = r_797.y;
                float _unpack_fp4_55[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_710[3 + 0]));
                    (&_unpack_fp4_55[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_55[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_55[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_55[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_55[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_55[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_55[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_55[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_55[0]), "=f"(_unpack_fp4_55[1]), "=f"(_unpack_fp4_55[2]), "=f"(_unpack_fp4_55[3]), "=f"(_unpack_fp4_55[4]), "=f"(_unpack_fp4_55[5]), "=f"(_unpack_fp4_55[6]), "=f"(_unpack_fp4_55[7]) : "r"(q_710[3 + 0]));
                    #endif
                }
                float2 a_798 = make_float2(d_759[24], d_759[25]);
                float2 q_799 = make_float2(_unpack_fp4_55[0], _unpack_fp4_55[1]);
                float2 r_800 = fma_f32x2_rn_noftz(q_799, scale1_761, a_798);
                d_759[24] = r_800.x;
                d_759[25] = r_800.y;
                float2 a_801 = make_float2(d_759[26], d_759[27]);
                float2 q_802 = make_float2(_unpack_fp4_55[2], _unpack_fp4_55[3]);
                float2 r_803 = fma_f32x2_rn_noftz(q_802, scale1_761, a_801);
                d_759[26] = r_803.x;
                d_759[27] = r_803.y;
                float2 a_804 = make_float2(d_759[28], d_759[29]);
                float2 q_805 = make_float2(_unpack_fp4_55[4], _unpack_fp4_55[5]);
                float2 r_806 = fma_f32x2_rn_noftz(q_805, scale1_761, a_804);
                d_759[28] = r_806.x;
                d_759[29] = r_806.y;
                float2 a_807 = make_float2(d_759[30], d_759[31]);
                float2 q_808 = make_float2(_unpack_fp4_55[6], _unpack_fp4_55[7]);
                float2 r_809 = fma_f32x2_rn_noftz(q_808, scale1_761, a_807);
                d_759[30] = r_809.x;
                d_759[31] = r_809.y;
                unsigned int d_759_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_759[_lp*2 + 0], d_759[_lp*2+1 + 0]));
                    d_759_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_759_bf16[0]), "r"(d_759_bf16[1]), "r"(d_759_bf16[2]), "r"(d_759_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_759_bf16[4]), "r"(d_759_bf16[5]), "r"(d_759_bf16[6]), "r"(d_759_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_759_bf16[8]), "r"(d_759_bf16[9]), "r"(d_759_bf16[10]), "r"(d_759_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_759_bf16[12]), "r"(d_759_bf16[13]), "r"(d_759_bf16[14]), "r"(d_759_bf16[15]) : "memory");
                __syncwarp();
                int c_810 = lane_1;
                int l_811 = c_810 >> 2;
                unsigned int w4_812[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_812[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_812[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_812[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_812[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_811 + (c_810 & 3 ^ l_811 >> 1 & 3)))));
                long long idx_813 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_810));
                {
                    int4 _iv4 = make_int4(w4_812[0 + 0], w4_812[0 + 1], w4_812[0 + 2], w4_812[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_813) = _iv4;
                }
                int c_814 = lane_1 + 32;
                int l_815 = c_814 >> 2;
                unsigned int w4_816[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_816[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_816[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_816[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_816[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_815 + (c_814 & 3 ^ l_815 >> 1 & 3)))));
                long long idx_817 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_814));
                {
                    int4 _iv4 = make_int4(w4_816[0 + 0], w4_816[0 + 1], w4_816[0 + 2], w4_816[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_817) = _iv4;
                }
                int c_818 = lane_1 + 64;
                int l_819 = c_818 >> 2;
                unsigned int w4_820[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_820[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_820[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_820[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_820[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_819 + (c_818 & 3 ^ l_819 >> 1 & 3)))));
                long long idx_821 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_818));
                {
                    int4 _iv4 = make_int4(w4_820[0 + 0], w4_820[0 + 1], w4_820[0 + 2], w4_820[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_821) = _iv4;
                }
                int c_822 = lane_1 + 96;
                int l_823 = c_822 >> 2;
                unsigned int w4_824[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_824[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_824[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_824[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_824[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_823 + (c_822 & 3 ^ l_823 >> 1 & 3)))));
                long long idx_825 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_822));
                {
                    int4 _iv4 = make_int4(w4_824[0 + 0], w4_824[0 + 1], w4_824[0 + 2], w4_824[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_825) = _iv4;
                }
                __syncwarp();
                int pp2_826 = lane_1 + 32 * (3 * h + 2);
                unsigned int q_827[4];
                float bm0_828 = 0.0f;
                float bm1_829 = 0.0f;
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[0]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[16]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[1]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[17]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[2]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[18]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[3]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[19]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[4]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[20]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[5]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[21]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[6]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[22]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[7]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[23]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[8]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[24]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[9]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[25]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[10]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[26]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[11]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[27]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[12]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[28]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[13]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[29]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[14]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[30]));
                bm0_828 = fmaxf(bm0_828, fabsf(acc_404[15]));
                bm1_829 = fmaxf(bm1_829, fabsf(acc_404[31]));
                float _fdiv_48 = __fdiv_rn(mul_f32_rn_noftz(bm1_829, inv), 6.0f);
                float _fdiv_49 = __fdiv_rn(mul_f32_rn_noftz(bm0_828, inv), 6.0f);
                uint16_t _e4m3x2_f32_2;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fdiv_48), "f"(_fdiv_49));
                uint32_t _e4m3x2_to_f16x2_16;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_16) : "h"((uint16_t)(_e4m3x2_f32_2)));
                float _cvt_f32_f16_32;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_32) : "h"((uint16_t)(_e4m3x2_to_f16x2_16)));
                float _cvt_f32_f16_33;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_33) : "h"((uint16_t)(_e4m3x2_to_f16x2_16 >> 16)));
                float _fdiv_rn_5 = __fdiv_rn(inv, _cvt_f32_f16_32);
                float _fdiv_rn_6 = __fdiv_rn(inv, _cvt_f32_f16_33);
                float m0_830 = 0.0f;
                float m1_831 = 0.0f;
                if (_cvt_f32_f16_32 > 0.0f) {
                    m0_830 = _fdiv_rn_5;
                }
                if (_cvt_f32_f16_33 > 0.0f) {
                    m1_831 = _fdiv_rn_6;
                }
                float2 mm_832 = make_float2(m0_830, m0_830);
                float2 pair_833 = make_float2(acc_404[0], acc_404[1]);
                float2 g_834;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_834) : "l"(*(const unsigned long long*)&pair_833), "l"(*(const unsigned long long*)&mm_832));
                uint32_t _fp4_pair_24;
                uint16_t _e2m1_pair_h_50;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_50) : "f"(g_834.x), "f"(g_834.y));
                _fp4_pair_24 = (uint32_t)_e2m1_pair_h_50;
                float2 pair_835 = make_float2(acc_404[2], acc_404[3]);
                float2 g_836;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_836) : "l"(*(const unsigned long long*)&pair_835), "l"(*(const unsigned long long*)&mm_832));
                uint32_t _fp4_pair_25;
                uint16_t _e2m1_pair_h_51;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_51) : "f"(g_836.x), "f"(g_836.y));
                _fp4_pair_25 = (uint32_t)_e2m1_pair_h_51;
                float2 pair_837 = make_float2(acc_404[4], acc_404[5]);
                float2 g_838;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_838) : "l"(*(const unsigned long long*)&pair_837), "l"(*(const unsigned long long*)&mm_832));
                uint32_t _fp4_pair_26;
                uint16_t _e2m1_pair_h_52;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_52) : "f"(g_838.x), "f"(g_838.y));
                _fp4_pair_26 = (uint32_t)_e2m1_pair_h_52;
                float2 pair_839 = make_float2(acc_404[6], acc_404[7]);
                float2 g_840;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_840) : "l"(*(const unsigned long long*)&pair_839), "l"(*(const unsigned long long*)&mm_832));
                uint32_t _fp4_e2m1_pack_pair_53;
                uint16_t _e2m1_pair_h_54;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_54) : "f"(g_840.x), "f"(g_840.y));
                _fp4_e2m1_pack_pair_53 = (uint32_t)_e2m1_pair_h_54;
                unsigned int word_841 = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_e2m1_pack_pair_53 << 24;
                unsigned int carry_842 = (word_841 & 2004318071) + 2004318071;
                q_827[0] = word_841 & (carry_842 | 2004318071);
                float2 mm_843 = make_float2(m0_830, m0_830);
                float2 pair_844 = make_float2(acc_404[8], acc_404[9]);
                float2 g_845;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_845) : "l"(*(const unsigned long long*)&pair_844), "l"(*(const unsigned long long*)&mm_843));
                uint32_t _fp4_pair_27;
                uint16_t _e2m1_pair_h_55;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(g_845.x), "f"(g_845.y));
                _fp4_pair_27 = (uint32_t)_e2m1_pair_h_55;
                float2 pair_846 = make_float2(acc_404[10], acc_404[11]);
                float2 g_847;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_847) : "l"(*(const unsigned long long*)&pair_846), "l"(*(const unsigned long long*)&mm_843));
                uint32_t _fp4_pair_28;
                uint16_t _e2m1_pair_h_56;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_56) : "f"(g_847.x), "f"(g_847.y));
                _fp4_pair_28 = (uint32_t)_e2m1_pair_h_56;
                float2 pair_848 = make_float2(acc_404[12], acc_404[13]);
                float2 g_849;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_849) : "l"(*(const unsigned long long*)&pair_848), "l"(*(const unsigned long long*)&mm_843));
                uint32_t _fp4_pair_29;
                uint16_t _e2m1_pair_h_57;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(g_849.x), "f"(g_849.y));
                _fp4_pair_29 = (uint32_t)_e2m1_pair_h_57;
                float2 pair_850 = make_float2(acc_404[14], acc_404[15]);
                float2 g_851;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_851) : "l"(*(const unsigned long long*)&pair_850), "l"(*(const unsigned long long*)&mm_843));
                uint32_t _fp4_e2m1_pack_pair_58;
                uint16_t _e2m1_pair_h_59;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(g_851.x), "f"(g_851.y));
                _fp4_e2m1_pack_pair_58 = (uint32_t)_e2m1_pair_h_59;
                unsigned int word_852 = _fp4_pair_27 | _fp4_pair_28 << 8 | _fp4_pair_29 << 16 | _fp4_e2m1_pack_pair_58 << 24;
                unsigned int carry_853 = (word_852 & 2004318071) + 2004318071;
                q_827[1] = word_852 & (carry_853 | 2004318071);
                float2 mm_854 = make_float2(m1_831, m1_831);
                float2 pair_855 = make_float2(acc_404[16], acc_404[17]);
                float2 g_856;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_856) : "l"(*(const unsigned long long*)&pair_855), "l"(*(const unsigned long long*)&mm_854));
                uint32_t _fp4_pair_30;
                uint16_t _e2m1_pair_h_60;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_60) : "f"(g_856.x), "f"(g_856.y));
                _fp4_pair_30 = (uint32_t)_e2m1_pair_h_60;
                float2 pair_857 = make_float2(acc_404[18], acc_404[19]);
                float2 g_858;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_858) : "l"(*(const unsigned long long*)&pair_857), "l"(*(const unsigned long long*)&mm_854));
                uint32_t _fp4_pair_31;
                uint16_t _e2m1_pair_h_61;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_61) : "f"(g_858.x), "f"(g_858.y));
                _fp4_pair_31 = (uint32_t)_e2m1_pair_h_61;
                float2 pair_859 = make_float2(acc_404[20], acc_404[21]);
                float2 g_860;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_860) : "l"(*(const unsigned long long*)&pair_859), "l"(*(const unsigned long long*)&mm_854));
                uint32_t _fp4_pair_32;
                uint16_t _e2m1_pair_h_62;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_62) : "f"(g_860.x), "f"(g_860.y));
                _fp4_pair_32 = (uint32_t)_e2m1_pair_h_62;
                float2 pair_861 = make_float2(acc_404[22], acc_404[23]);
                float2 g_862;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_862) : "l"(*(const unsigned long long*)&pair_861), "l"(*(const unsigned long long*)&mm_854));
                uint32_t _fp4_e2m1_pack_pair_63;
                uint16_t _e2m1_pair_h_64;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_64) : "f"(g_862.x), "f"(g_862.y));
                _fp4_e2m1_pack_pair_63 = (uint32_t)_e2m1_pair_h_64;
                unsigned int word_863 = _fp4_pair_30 | _fp4_pair_31 << 8 | _fp4_pair_32 << 16 | _fp4_e2m1_pack_pair_63 << 24;
                unsigned int carry_864 = (word_863 & 2004318071) + 2004318071;
                q_827[2] = word_863 & (carry_864 | 2004318071);
                float2 mm_865 = make_float2(m1_831, m1_831);
                float2 pair_866 = make_float2(acc_404[24], acc_404[25]);
                float2 g_867;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_867) : "l"(*(const unsigned long long*)&pair_866), "l"(*(const unsigned long long*)&mm_865));
                uint32_t _fp4_pair_33;
                uint16_t _e2m1_pair_h_65;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_65) : "f"(g_867.x), "f"(g_867.y));
                _fp4_pair_33 = (uint32_t)_e2m1_pair_h_65;
                float2 pair_868 = make_float2(acc_404[26], acc_404[27]);
                float2 g_869;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_869) : "l"(*(const unsigned long long*)&pair_868), "l"(*(const unsigned long long*)&mm_865));
                uint32_t _fp4_pair_34;
                uint16_t _e2m1_pair_h_66;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_66) : "f"(g_869.x), "f"(g_869.y));
                _fp4_pair_34 = (uint32_t)_e2m1_pair_h_66;
                float2 pair_870 = make_float2(acc_404[28], acc_404[29]);
                float2 g_871;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_871) : "l"(*(const unsigned long long*)&pair_870), "l"(*(const unsigned long long*)&mm_865));
                uint32_t _fp4_pair_35;
                uint16_t _e2m1_pair_h_67;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_67) : "f"(g_871.x), "f"(g_871.y));
                _fp4_pair_35 = (uint32_t)_e2m1_pair_h_67;
                float2 pair_872 = make_float2(acc_404[30], acc_404[31]);
                float2 g_873;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_873) : "l"(*(const unsigned long long*)&pair_872), "l"(*(const unsigned long long*)&mm_865));
                uint32_t _fp4_e2m1_pack_pair_68;
                uint16_t _e2m1_pair_h_69;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_69) : "f"(g_873.x), "f"(g_873.y));
                _fp4_e2m1_pack_pair_68 = (uint32_t)_e2m1_pair_h_69;
                unsigned int word_874 = _fp4_pair_33 | _fp4_pair_34 << 8 | _fp4_pair_35 << 16 | _fp4_e2m1_pack_pair_68 << 24;
                unsigned int carry_875 = (word_874 & 2004318071) + 2004318071;
                q_827[3] = word_874 & (carry_875 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_826)), "r"(q_827[0]), "r"(q_827[1]), "r"(q_827[2]), "r"(q_827[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2_826)), "h"((uint16_t)(_e4m3x2_f32_2)) : "memory");
                float d_876[32] = {0};
                uint32_t _e4m3x2_to_f16x2_17;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_17) : "h"((uint16_t)(_e4m3x2_f32_2)));
                float _cvt_f32_f16_34;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_34) : "h"((uint16_t)(_e4m3x2_to_f16x2_17)));
                float _cvt_f32_f16_35;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_35) : "h"((uint16_t)(_e4m3x2_to_f16x2_17 >> 16)));
                float _mul_28;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_cvt_f32_f16_34), "f"(rs2));
                float _mul_29;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_cvt_f32_f16_35), "f"(rs2));
                float2 scale0_877 = make_float2(_mul_28, _mul_28);
                float2 scale1_878 = make_float2(_mul_29, _mul_29);
                float _unpack_fp4_56[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_827[0]));
                    (&_unpack_fp4_56[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_56[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_56[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_56[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_56[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_56[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_56[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_56[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_56[0]), "=f"(_unpack_fp4_56[1]), "=f"(_unpack_fp4_56[2]), "=f"(_unpack_fp4_56[3]), "=f"(_unpack_fp4_56[4]), "=f"(_unpack_fp4_56[5]), "=f"(_unpack_fp4_56[6]), "=f"(_unpack_fp4_56[7]) : "r"(q_827[0]));
                    #endif
                }
                float2 a_879 = make_float2(d_876[0], d_876[1]);
                float2 q_880 = make_float2(_unpack_fp4_56[0], _unpack_fp4_56[1]);
                float2 r_881 = fma_f32x2_rn_noftz(q_880, scale0_877, a_879);
                d_876[0] = r_881.x;
                d_876[1] = r_881.y;
                float2 a_882 = make_float2(d_876[2], d_876[3]);
                float2 q_883 = make_float2(_unpack_fp4_56[2], _unpack_fp4_56[3]);
                float2 r_884 = fma_f32x2_rn_noftz(q_883, scale0_877, a_882);
                d_876[2] = r_884.x;
                d_876[3] = r_884.y;
                float2 a_885 = make_float2(d_876[4], d_876[5]);
                float2 q_886 = make_float2(_unpack_fp4_56[4], _unpack_fp4_56[5]);
                float2 r_887 = fma_f32x2_rn_noftz(q_886, scale0_877, a_885);
                d_876[4] = r_887.x;
                d_876[5] = r_887.y;
                float2 a_888 = make_float2(d_876[6], d_876[7]);
                float2 q_889 = make_float2(_unpack_fp4_56[6], _unpack_fp4_56[7]);
                float2 r_890 = fma_f32x2_rn_noftz(q_889, scale0_877, a_888);
                d_876[6] = r_890.x;
                d_876[7] = r_890.y;
                float _unpack_fp4_57[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_827[1 + 0]));
                    (&_unpack_fp4_57[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_57[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_57[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_57[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_57[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_57[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_57[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_57[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_57[0]), "=f"(_unpack_fp4_57[1]), "=f"(_unpack_fp4_57[2]), "=f"(_unpack_fp4_57[3]), "=f"(_unpack_fp4_57[4]), "=f"(_unpack_fp4_57[5]), "=f"(_unpack_fp4_57[6]), "=f"(_unpack_fp4_57[7]) : "r"(q_827[1 + 0]));
                    #endif
                }
                float2 a_891 = make_float2(d_876[8], d_876[9]);
                float2 q_892 = make_float2(_unpack_fp4_57[0], _unpack_fp4_57[1]);
                float2 r_893 = fma_f32x2_rn_noftz(q_892, scale0_877, a_891);
                d_876[8] = r_893.x;
                d_876[9] = r_893.y;
                float2 a_894 = make_float2(d_876[10], d_876[11]);
                float2 q_895 = make_float2(_unpack_fp4_57[2], _unpack_fp4_57[3]);
                float2 r_896 = fma_f32x2_rn_noftz(q_895, scale0_877, a_894);
                d_876[10] = r_896.x;
                d_876[11] = r_896.y;
                float2 a_897 = make_float2(d_876[12], d_876[13]);
                float2 q_898 = make_float2(_unpack_fp4_57[4], _unpack_fp4_57[5]);
                float2 r_899 = fma_f32x2_rn_noftz(q_898, scale0_877, a_897);
                d_876[12] = r_899.x;
                d_876[13] = r_899.y;
                float2 a_900 = make_float2(d_876[14], d_876[15]);
                float2 q_901 = make_float2(_unpack_fp4_57[6], _unpack_fp4_57[7]);
                float2 r_902 = fma_f32x2_rn_noftz(q_901, scale0_877, a_900);
                d_876[14] = r_902.x;
                d_876[15] = r_902.y;
                float _unpack_fp4_58[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_827[2 + 0]));
                    (&_unpack_fp4_58[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_58[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_58[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_58[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_58[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_58[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_58[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_58[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_58[0]), "=f"(_unpack_fp4_58[1]), "=f"(_unpack_fp4_58[2]), "=f"(_unpack_fp4_58[3]), "=f"(_unpack_fp4_58[4]), "=f"(_unpack_fp4_58[5]), "=f"(_unpack_fp4_58[6]), "=f"(_unpack_fp4_58[7]) : "r"(q_827[2 + 0]));
                    #endif
                }
                float2 a_903 = make_float2(d_876[16], d_876[17]);
                float2 q_904 = make_float2(_unpack_fp4_58[0], _unpack_fp4_58[1]);
                float2 r_905 = fma_f32x2_rn_noftz(q_904, scale1_878, a_903);
                d_876[16] = r_905.x;
                d_876[17] = r_905.y;
                float2 a_906 = make_float2(d_876[18], d_876[19]);
                float2 q_907 = make_float2(_unpack_fp4_58[2], _unpack_fp4_58[3]);
                float2 r_908 = fma_f32x2_rn_noftz(q_907, scale1_878, a_906);
                d_876[18] = r_908.x;
                d_876[19] = r_908.y;
                float2 a_909 = make_float2(d_876[20], d_876[21]);
                float2 q_910 = make_float2(_unpack_fp4_58[4], _unpack_fp4_58[5]);
                float2 r_911 = fma_f32x2_rn_noftz(q_910, scale1_878, a_909);
                d_876[20] = r_911.x;
                d_876[21] = r_911.y;
                float2 a_912 = make_float2(d_876[22], d_876[23]);
                float2 q_913 = make_float2(_unpack_fp4_58[6], _unpack_fp4_58[7]);
                float2 r_914 = fma_f32x2_rn_noftz(q_913, scale1_878, a_912);
                d_876[22] = r_914.x;
                d_876[23] = r_914.y;
                float _unpack_fp4_59[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_827[3 + 0]));
                    (&_unpack_fp4_59[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_59[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_59[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_59[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_59[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_59[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_59[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_59[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_59[0]), "=f"(_unpack_fp4_59[1]), "=f"(_unpack_fp4_59[2]), "=f"(_unpack_fp4_59[3]), "=f"(_unpack_fp4_59[4]), "=f"(_unpack_fp4_59[5]), "=f"(_unpack_fp4_59[6]), "=f"(_unpack_fp4_59[7]) : "r"(q_827[3 + 0]));
                    #endif
                }
                float2 a_915 = make_float2(d_876[24], d_876[25]);
                float2 q_916 = make_float2(_unpack_fp4_59[0], _unpack_fp4_59[1]);
                float2 r_917 = fma_f32x2_rn_noftz(q_916, scale1_878, a_915);
                d_876[24] = r_917.x;
                d_876[25] = r_917.y;
                float2 a_918 = make_float2(d_876[26], d_876[27]);
                float2 q_919 = make_float2(_unpack_fp4_59[2], _unpack_fp4_59[3]);
                float2 r_920 = fma_f32x2_rn_noftz(q_919, scale1_878, a_918);
                d_876[26] = r_920.x;
                d_876[27] = r_920.y;
                float2 a_921 = make_float2(d_876[28], d_876[29]);
                float2 q_922 = make_float2(_unpack_fp4_59[4], _unpack_fp4_59[5]);
                float2 r_923 = fma_f32x2_rn_noftz(q_922, scale1_878, a_921);
                d_876[28] = r_923.x;
                d_876[29] = r_923.y;
                float2 a_924 = make_float2(d_876[30], d_876[31]);
                float2 q_925 = make_float2(_unpack_fp4_59[6], _unpack_fp4_59[7]);
                float2 r_926 = fma_f32x2_rn_noftz(q_925, scale1_878, a_924);
                d_876[30] = r_926.x;
                d_876[31] = r_926.y;
                unsigned int d_876_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_876[_lp*2 + 0], d_876[_lp*2+1 + 0]));
                    d_876_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_876_bf16[0]), "r"(d_876_bf16[1]), "r"(d_876_bf16[2]), "r"(d_876_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_876_bf16[4]), "r"(d_876_bf16[5]), "r"(d_876_bf16[6]), "r"(d_876_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_876_bf16[8]), "r"(d_876_bf16[9]), "r"(d_876_bf16[10]), "r"(d_876_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_876_bf16[12]), "r"(d_876_bf16[13]), "r"(d_876_bf16[14]), "r"(d_876_bf16[15]) : "memory");
                __syncwarp();
                int c_927 = lane_1;
                int l_928 = c_927 >> 2;
                unsigned int w4_929[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_929[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_929[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_929[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_929[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_928 + (c_927 & 3 ^ l_928 >> 1 & 3)))));
                long long idx_930 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_927));
                {
                    int4 _iv4 = make_int4(w4_929[0 + 0], w4_929[0 + 1], w4_929[0 + 2], w4_929[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_930) = _iv4;
                }
                int c_931 = lane_1 + 32;
                int l_932 = c_931 >> 2;
                unsigned int w4_933[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_933[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_933[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_933[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_933[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_932 + (c_931 & 3 ^ l_932 >> 1 & 3)))));
                long long idx_934 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_931));
                {
                    int4 _iv4 = make_int4(w4_933[0 + 0], w4_933[0 + 1], w4_933[0 + 2], w4_933[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_934) = _iv4;
                }
                int c_935 = lane_1 + 64;
                int l_936 = c_935 >> 2;
                unsigned int w4_937[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_937[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_937[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_937[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_937[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_936 + (c_935 & 3 ^ l_936 >> 1 & 3)))));
                long long idx_938 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_935));
                {
                    int4 _iv4 = make_int4(w4_937[0 + 0], w4_937[0 + 1], w4_937[0 + 2], w4_937[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_938) = _iv4;
                }
                int c_939 = lane_1 + 96;
                int l_940 = c_939 >> 2;
                unsigned int w4_941[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_941[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_941[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_941[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_941[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_940 + (c_939 & 3 ^ l_940 >> 1 & 3)))));
                long long idx_942 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_939));
                {
                    int4 _iv4 = make_int4(w4_941[0 + 0], w4_941[0 + 1], w4_941[0 + 2], w4_941[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_942) = _iv4;
                }
                __syncwarp();
                if (h == 0 && lane_1 < 4) {
                    {
                        uint32_t _addr_70 = static_cast<uint32_t>(wsm_addr + (unsigned int)(tob_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(rs2) : "memory");
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
                    int m_off = ag_ro + (rk_2 * capo_0 + ix_1_1) * 16;
                    int peer = (rank + 1) % 4;
                    uint8_t* pw = reinterpret_cast<uint8_t*>(tab[peer]);
                    uint8_t* pr = reinterpret_cast<uint8_t*>(tab[4 + peer]);
                    {
                        void* _cpbulk_dst_71 = reinterpret_cast<void*>(pw + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_71), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_72 = reinterpret_cast<void*>(pw + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_72), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_73 = reinterpret_cast<void*>(pr + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_73), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_3 = (rank + 2) % 4;
                    uint8_t* pw_4 = reinterpret_cast<uint8_t*>(tab[peer_3]);
                    uint8_t* pr_5 = reinterpret_cast<uint8_t*>(tab[4 + peer_3]);
                    {
                        void* _cpbulk_dst_74 = reinterpret_cast<void*>(pw_4 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_74), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_75 = reinterpret_cast<void*>(pw_4 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_75), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_76 = reinterpret_cast<void*>(pr_5 + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_76), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_6 = (rank + 3) % 4;
                    uint8_t* pw_7 = reinterpret_cast<uint8_t*>(tab[peer_6]);
                    uint8_t* pr_8 = reinterpret_cast<uint8_t*>(tab[4 + peer_6]);
                    {
                        void* _cpbulk_dst_77 = reinterpret_cast<void*>(pw_7 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_77), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_78 = reinterpret_cast<void*>(pw_7 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_78), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_79 = reinterpret_cast<void*>(pr_8 + m_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_79), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                kb = kb + 1;
            }
            int fme = rb0 - r0;
            #pragma unroll 1
            for (int c_1 = wC; c_1 < nC; c_1 += 10) {
                int blk = c_1 / 3;
                int w = c_1 % 3;
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
                #pragma unroll 1
                for (int jj = 0; jj < 2; jj++) {
                    int pp0 = lane_1 + 96 * jj;
                    int pp1 = pp0 + 32;
                    int pp2_1 = pp0 + 64;
                    uint32_t _fp4_wire_values_1[12];
                    uint32_t _fp4_wire_scales_1[3];
                    uint16_t _wire_h16_80_0;
                    uint16_t _wire_h16_80_1;
                    uint16_t _wire_h16_80_2;
                    uint32_t _wire_vpend_80 = 7u;
                    uint32_t _wire_spend_80 = 7u;
                    do {
                        if (_wire_vpend_80 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[0]), "=r"(_fp4_wire_values_1[1]), "=r"(_fp4_wire_values_1[2]), "=r"(_fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_80 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[4]), "=r"(_fp4_wire_values_1[5]), "=r"(_fp4_wire_values_1[6]), "=r"(_fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_80 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[8]), "=r"(_fp4_wire_values_1[9]), "=r"(_fp4_wire_values_1[10]), "=r"(_fp4_wire_values_1[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp2_1)))) : "memory");
                        }
                        if (_wire_spend_80 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_80_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_1[0] = (uint32_t)_wire_h16_80_0;
                        }
                        if (_wire_spend_80 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_80_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_1[1] = (uint32_t)_wire_h16_80_1;
                        }
                        if (_wire_spend_80 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_80_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_2 + (int)(2 * pp2_1)))) : "memory");
                            _fp4_wire_scales_1[2] = (uint32_t)_wire_h16_80_2;
                        }
                        _wire_vpend_80 = ((uint32_t)((((_fp4_wire_values_1[0] & ~((_fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[1] & ~((_fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[2] & ~((_fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[3] & ~((_fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_1[4] & ~((_fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[5] & ~((_fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[6] & ~((_fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[7] & ~((_fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_1[8] & ~((_fp4_wire_values_1[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[9] & ~((_fp4_wire_values_1[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[10] & ~((_fp4_wire_values_1[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[11] & ~((_fp4_wire_values_1[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2);
                        _wire_spend_80 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2);
                    } while ((_wire_vpend_80 | _wire_spend_80) != 0u);
                    float acc_1[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_18;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_18) : "h"((uint16_t)(_fp4_wire_scales_1[0])));
                    float _cvt_f32_f16_36;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_36) : "h"((uint16_t)(_e4m3x2_to_f16x2_18)));
                    float _cvt_f32_f16_37;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_37) : "h"((uint16_t)(_e4m3x2_to_f16x2_18 >> 16)));
                    float _mul_30;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_cvt_f32_f16_36), "f"(rs_1));
                    float _mul_31;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_cvt_f32_f16_37), "f"(rs_1));
                    float2 scale0_1 = make_float2(_mul_30, _mul_30);
                    float2 scale1_1 = make_float2(_mul_31, _mul_31);
                    float _unpack_fp4_60[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[0]));
                        (&_unpack_fp4_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_60[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_60[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_60[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_60[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_60[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_60[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_60[0]), "=f"(_unpack_fp4_60[1]), "=f"(_unpack_fp4_60[2]), "=f"(_unpack_fp4_60[3]), "=f"(_unpack_fp4_60[4]), "=f"(_unpack_fp4_60[5]), "=f"(_unpack_fp4_60[6]), "=f"(_unpack_fp4_60[7]) : "r"(_fp4_wire_values_1[0]));
                        #endif
                    }
                    float2 a_1 = make_float2(acc_1[0], acc_1[1]);
                    float2 q_1 = make_float2(_unpack_fp4_60[0], _unpack_fp4_60[1]);
                    float2 r_1 = fma_f32x2_rn_noftz(q_1, scale0_1, a_1);
                    acc_1[0] = r_1.x;
                    acc_1[1] = r_1.y;
                    float2 a_0 = make_float2(acc_1[2], acc_1[3]);
                    float2 q_1_1 = make_float2(_unpack_fp4_60[2], _unpack_fp4_60[3]);
                    float2 r_2 = fma_f32x2_rn_noftz(q_1_1, scale0_1, a_0);
                    acc_1[2] = r_2.x;
                    acc_1[3] = r_2.y;
                    float2 a_3 = make_float2(acc_1[4], acc_1[5]);
                    float2 q_4 = make_float2(_unpack_fp4_60[4], _unpack_fp4_60[5]);
                    float2 r_5 = fma_f32x2_rn_noftz(q_4, scale0_1, a_3);
                    acc_1[4] = r_5.x;
                    acc_1[5] = r_5.y;
                    float2 a_6 = make_float2(acc_1[6], acc_1[7]);
                    float2 q_7 = make_float2(_unpack_fp4_60[6], _unpack_fp4_60[7]);
                    float2 r_8 = fma_f32x2_rn_noftz(q_7, scale0_1, a_6);
                    acc_1[6] = r_8.x;
                    acc_1[7] = r_8.y;
                    float _unpack_fp4_61[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[1 + 0]));
                        (&_unpack_fp4_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_61[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_61[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_61[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_61[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_61[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_61[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_61[0]), "=f"(_unpack_fp4_61[1]), "=f"(_unpack_fp4_61[2]), "=f"(_unpack_fp4_61[3]), "=f"(_unpack_fp4_61[4]), "=f"(_unpack_fp4_61[5]), "=f"(_unpack_fp4_61[6]), "=f"(_unpack_fp4_61[7]) : "r"(_fp4_wire_values_1[1 + 0]));
                        #endif
                    }
                    float2 a_9 = make_float2(acc_1[8], acc_1[9]);
                    float2 q_10 = make_float2(_unpack_fp4_61[0], _unpack_fp4_61[1]);
                    float2 r_11 = fma_f32x2_rn_noftz(q_10, scale0_1, a_9);
                    acc_1[8] = r_11.x;
                    acc_1[9] = r_11.y;
                    float2 a_12 = make_float2(acc_1[10], acc_1[11]);
                    float2 q_13 = make_float2(_unpack_fp4_61[2], _unpack_fp4_61[3]);
                    float2 r_14 = fma_f32x2_rn_noftz(q_13, scale0_1, a_12);
                    acc_1[10] = r_14.x;
                    acc_1[11] = r_14.y;
                    float2 a_15 = make_float2(acc_1[12], acc_1[13]);
                    float2 q_16 = make_float2(_unpack_fp4_61[4], _unpack_fp4_61[5]);
                    float2 r_17 = fma_f32x2_rn_noftz(q_16, scale0_1, a_15);
                    acc_1[12] = r_17.x;
                    acc_1[13] = r_17.y;
                    float2 a_18 = make_float2(acc_1[14], acc_1[15]);
                    float2 q_19 = make_float2(_unpack_fp4_61[6], _unpack_fp4_61[7]);
                    float2 r_20 = fma_f32x2_rn_noftz(q_19, scale0_1, a_18);
                    acc_1[14] = r_20.x;
                    acc_1[15] = r_20.y;
                    float _unpack_fp4_62[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[2 + 0]));
                        (&_unpack_fp4_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_62[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_62[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_62[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_62[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_62[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_62[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_62[0]), "=f"(_unpack_fp4_62[1]), "=f"(_unpack_fp4_62[2]), "=f"(_unpack_fp4_62[3]), "=f"(_unpack_fp4_62[4]), "=f"(_unpack_fp4_62[5]), "=f"(_unpack_fp4_62[6]), "=f"(_unpack_fp4_62[7]) : "r"(_fp4_wire_values_1[2 + 0]));
                        #endif
                    }
                    float2 a_21 = make_float2(acc_1[16], acc_1[17]);
                    float2 q_22 = make_float2(_unpack_fp4_62[0], _unpack_fp4_62[1]);
                    float2 r_23 = fma_f32x2_rn_noftz(q_22, scale1_1, a_21);
                    acc_1[16] = r_23.x;
                    acc_1[17] = r_23.y;
                    float2 a_24 = make_float2(acc_1[18], acc_1[19]);
                    float2 q_25 = make_float2(_unpack_fp4_62[2], _unpack_fp4_62[3]);
                    float2 r_26 = fma_f32x2_rn_noftz(q_25, scale1_1, a_24);
                    acc_1[18] = r_26.x;
                    acc_1[19] = r_26.y;
                    float2 a_27 = make_float2(acc_1[20], acc_1[21]);
                    float2 q_28 = make_float2(_unpack_fp4_62[4], _unpack_fp4_62[5]);
                    float2 r_29 = fma_f32x2_rn_noftz(q_28, scale1_1, a_27);
                    acc_1[20] = r_29.x;
                    acc_1[21] = r_29.y;
                    float2 a_30 = make_float2(acc_1[22], acc_1[23]);
                    float2 q_31 = make_float2(_unpack_fp4_62[6], _unpack_fp4_62[7]);
                    float2 r_32 = fma_f32x2_rn_noftz(q_31, scale1_1, a_30);
                    acc_1[22] = r_32.x;
                    acc_1[23] = r_32.y;
                    float _unpack_fp4_63[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[3 + 0]));
                        (&_unpack_fp4_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_63[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_63[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_63[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_63[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_63[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_63[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_63[0]), "=f"(_unpack_fp4_63[1]), "=f"(_unpack_fp4_63[2]), "=f"(_unpack_fp4_63[3]), "=f"(_unpack_fp4_63[4]), "=f"(_unpack_fp4_63[5]), "=f"(_unpack_fp4_63[6]), "=f"(_unpack_fp4_63[7]) : "r"(_fp4_wire_values_1[3 + 0]));
                        #endif
                    }
                    float2 a_33 = make_float2(acc_1[24], acc_1[25]);
                    float2 q_34 = make_float2(_unpack_fp4_63[0], _unpack_fp4_63[1]);
                    float2 r_35 = fma_f32x2_rn_noftz(q_34, scale1_1, a_33);
                    acc_1[24] = r_35.x;
                    acc_1[25] = r_35.y;
                    float2 a_36 = make_float2(acc_1[26], acc_1[27]);
                    float2 q_37 = make_float2(_unpack_fp4_63[2], _unpack_fp4_63[3]);
                    float2 r_38 = fma_f32x2_rn_noftz(q_37, scale1_1, a_36);
                    acc_1[26] = r_38.x;
                    acc_1[27] = r_38.y;
                    float2 a_39 = make_float2(acc_1[28], acc_1[29]);
                    float2 q_40 = make_float2(_unpack_fp4_63[4], _unpack_fp4_63[5]);
                    float2 r_41 = fma_f32x2_rn_noftz(q_40, scale1_1, a_39);
                    acc_1[28] = r_41.x;
                    acc_1[29] = r_41.y;
                    float2 a_42 = make_float2(acc_1[30], acc_1[31]);
                    float2 q_43 = make_float2(_unpack_fp4_63[6], _unpack_fp4_63[7]);
                    float2 r_44 = fma_f32x2_rn_noftz(q_43, scale1_1, a_42);
                    acc_1[30] = r_44.x;
                    acc_1[31] = r_44.y;
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
                    int c_45 = lane_1;
                    int l_1 = c_45 >> 2;
                    unsigned int w4_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1 + (c_45 & 3 ^ l_1 >> 1 & 3)))));
                    long long idx_1 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_45));
                    {
                        int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1) = _iv4;
                    }
                    int c_46 = lane_1 + 32;
                    int l_47 = c_46 >> 2;
                    unsigned int w4_48[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_47 + (c_46 & 3 ^ l_47 >> 1 & 3)))));
                    long long idx_49 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_46));
                    {
                        int4 _iv4 = make_int4(w4_48[0 + 0], w4_48[0 + 1], w4_48[0 + 2], w4_48[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_49) = _iv4;
                    }
                    int c_50 = lane_1 + 64;
                    int l_51 = c_50 >> 2;
                    unsigned int w4_52[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_52[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_51 + (c_50 & 3 ^ l_51 >> 1 & 3)))));
                    long long idx_53 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_50));
                    {
                        int4 _iv4 = make_int4(w4_52[0 + 0], w4_52[0 + 1], w4_52[0 + 2], w4_52[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_53) = _iv4;
                    }
                    int c_54 = lane_1 + 96;
                    int l_55 = c_54 >> 2;
                    unsigned int w4_56[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_55 + (c_54 & 3 ^ l_55 >> 1 & 3)))));
                    long long idx_57 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_54));
                    {
                        int4 _iv4 = make_int4(w4_56[0 + 0], w4_56[0 + 1], w4_56[0 + 2], w4_56[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_57) = _iv4;
                    }
                    __syncwarp();
                    float acc_58[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_19;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_19) : "h"((uint16_t)(_fp4_wire_scales_1[1])));
                    float _cvt_f32_f16_38;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_38) : "h"((uint16_t)(_e4m3x2_to_f16x2_19)));
                    float _cvt_f32_f16_39;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_39) : "h"((uint16_t)(_e4m3x2_to_f16x2_19 >> 16)));
                    float _mul_32;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(_cvt_f32_f16_38), "f"(rs_1));
                    float _mul_33;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_cvt_f32_f16_39), "f"(rs_1));
                    float2 scale0_59 = make_float2(_mul_32, _mul_32);
                    float2 scale1_60 = make_float2(_mul_33, _mul_33);
                    float _unpack_fp4_64[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[4 + 0]));
                        (&_unpack_fp4_64[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_64[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_64[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_64[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_64[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_64[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_64[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_64[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_64[0]), "=f"(_unpack_fp4_64[1]), "=f"(_unpack_fp4_64[2]), "=f"(_unpack_fp4_64[3]), "=f"(_unpack_fp4_64[4]), "=f"(_unpack_fp4_64[5]), "=f"(_unpack_fp4_64[6]), "=f"(_unpack_fp4_64[7]) : "r"(_fp4_wire_values_1[4 + 0]));
                        #endif
                    }
                    float2 a_61_1 = make_float2(acc_58[0], acc_58[1]);
                    float2 q_62_1 = make_float2(_unpack_fp4_64[0], _unpack_fp4_64[1]);
                    float2 r_63_1 = fma_f32x2_rn_noftz(q_62_1, scale0_59, a_61_1);
                    acc_58[0] = r_63_1.x;
                    acc_58[1] = r_63_1.y;
                    float2 a_64_1 = make_float2(acc_58[2], acc_58[3]);
                    float2 q_65_1 = make_float2(_unpack_fp4_64[2], _unpack_fp4_64[3]);
                    float2 r_66_1 = fma_f32x2_rn_noftz(q_65_1, scale0_59, a_64_1);
                    acc_58[2] = r_66_1.x;
                    acc_58[3] = r_66_1.y;
                    float2 a_67_1 = make_float2(acc_58[4], acc_58[5]);
                    float2 q_68_1 = make_float2(_unpack_fp4_64[4], _unpack_fp4_64[5]);
                    float2 r_69_1 = fma_f32x2_rn_noftz(q_68_1, scale0_59, a_67_1);
                    acc_58[4] = r_69_1.x;
                    acc_58[5] = r_69_1.y;
                    float2 a_70_1 = make_float2(acc_58[6], acc_58[7]);
                    float2 q_71_1 = make_float2(_unpack_fp4_64[6], _unpack_fp4_64[7]);
                    float2 r_72_1 = fma_f32x2_rn_noftz(q_71_1, scale0_59, a_70_1);
                    acc_58[6] = r_72_1.x;
                    acc_58[7] = r_72_1.y;
                    float _unpack_fp4_65[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[5 + 0]));
                        (&_unpack_fp4_65[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_65[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_65[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_65[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_65[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_65[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_65[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_65[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_65[0]), "=f"(_unpack_fp4_65[1]), "=f"(_unpack_fp4_65[2]), "=f"(_unpack_fp4_65[3]), "=f"(_unpack_fp4_65[4]), "=f"(_unpack_fp4_65[5]), "=f"(_unpack_fp4_65[6]), "=f"(_unpack_fp4_65[7]) : "r"(_fp4_wire_values_1[5 + 0]));
                        #endif
                    }
                    float2 a_73_1 = make_float2(acc_58[8], acc_58[9]);
                    float2 q_74_1 = make_float2(_unpack_fp4_65[0], _unpack_fp4_65[1]);
                    float2 r_75_1 = fma_f32x2_rn_noftz(q_74_1, scale0_59, a_73_1);
                    acc_58[8] = r_75_1.x;
                    acc_58[9] = r_75_1.y;
                    float2 a_76_1 = make_float2(acc_58[10], acc_58[11]);
                    float2 q_77_1 = make_float2(_unpack_fp4_65[2], _unpack_fp4_65[3]);
                    float2 r_78_1 = fma_f32x2_rn_noftz(q_77_1, scale0_59, a_76_1);
                    acc_58[10] = r_78_1.x;
                    acc_58[11] = r_78_1.y;
                    float2 a_79_1 = make_float2(acc_58[12], acc_58[13]);
                    float2 q_80_1 = make_float2(_unpack_fp4_65[4], _unpack_fp4_65[5]);
                    float2 r_81_1 = fma_f32x2_rn_noftz(q_80_1, scale0_59, a_79_1);
                    acc_58[12] = r_81_1.x;
                    acc_58[13] = r_81_1.y;
                    float2 a_82_1 = make_float2(acc_58[14], acc_58[15]);
                    float2 q_83_1 = make_float2(_unpack_fp4_65[6], _unpack_fp4_65[7]);
                    float2 r_84_1 = fma_f32x2_rn_noftz(q_83_1, scale0_59, a_82_1);
                    acc_58[14] = r_84_1.x;
                    acc_58[15] = r_84_1.y;
                    float _unpack_fp4_66[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[6 + 0]));
                        (&_unpack_fp4_66[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_66[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_66[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_66[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_66[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_66[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_66[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_66[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_66[0]), "=f"(_unpack_fp4_66[1]), "=f"(_unpack_fp4_66[2]), "=f"(_unpack_fp4_66[3]), "=f"(_unpack_fp4_66[4]), "=f"(_unpack_fp4_66[5]), "=f"(_unpack_fp4_66[6]), "=f"(_unpack_fp4_66[7]) : "r"(_fp4_wire_values_1[6 + 0]));
                        #endif
                    }
                    float2 a_85_1 = make_float2(acc_58[16], acc_58[17]);
                    float2 q_86_1 = make_float2(_unpack_fp4_66[0], _unpack_fp4_66[1]);
                    float2 r_87_1 = fma_f32x2_rn_noftz(q_86_1, scale1_60, a_85_1);
                    acc_58[16] = r_87_1.x;
                    acc_58[17] = r_87_1.y;
                    float2 a_88_1 = make_float2(acc_58[18], acc_58[19]);
                    float2 q_89_1 = make_float2(_unpack_fp4_66[2], _unpack_fp4_66[3]);
                    float2 r_90_1 = fma_f32x2_rn_noftz(q_89_1, scale1_60, a_88_1);
                    acc_58[18] = r_90_1.x;
                    acc_58[19] = r_90_1.y;
                    float2 a_91_1 = make_float2(acc_58[20], acc_58[21]);
                    float2 q_92_1 = make_float2(_unpack_fp4_66[4], _unpack_fp4_66[5]);
                    float2 r_93_1 = fma_f32x2_rn_noftz(q_92_1, scale1_60, a_91_1);
                    acc_58[20] = r_93_1.x;
                    acc_58[21] = r_93_1.y;
                    float2 a_94_1 = make_float2(acc_58[22], acc_58[23]);
                    float2 q_95_1 = make_float2(_unpack_fp4_66[6], _unpack_fp4_66[7]);
                    float2 r_96_1 = fma_f32x2_rn_noftz(q_95_1, scale1_60, a_94_1);
                    acc_58[22] = r_96_1.x;
                    acc_58[23] = r_96_1.y;
                    float _unpack_fp4_67[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[7 + 0]));
                        (&_unpack_fp4_67[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_67[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_67[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_67[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_67[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_67[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_67[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_67[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_67[0]), "=f"(_unpack_fp4_67[1]), "=f"(_unpack_fp4_67[2]), "=f"(_unpack_fp4_67[3]), "=f"(_unpack_fp4_67[4]), "=f"(_unpack_fp4_67[5]), "=f"(_unpack_fp4_67[6]), "=f"(_unpack_fp4_67[7]) : "r"(_fp4_wire_values_1[7 + 0]));
                        #endif
                    }
                    float2 a_97_1 = make_float2(acc_58[24], acc_58[25]);
                    float2 q_98_1 = make_float2(_unpack_fp4_67[0], _unpack_fp4_67[1]);
                    float2 r_99_1 = fma_f32x2_rn_noftz(q_98_1, scale1_60, a_97_1);
                    acc_58[24] = r_99_1.x;
                    acc_58[25] = r_99_1.y;
                    float2 a_100_1 = make_float2(acc_58[26], acc_58[27]);
                    float2 q_101_1 = make_float2(_unpack_fp4_67[2], _unpack_fp4_67[3]);
                    float2 r_102_1 = fma_f32x2_rn_noftz(q_101_1, scale1_60, a_100_1);
                    acc_58[26] = r_102_1.x;
                    acc_58[27] = r_102_1.y;
                    float2 a_103 = make_float2(acc_58[28], acc_58[29]);
                    float2 q_104 = make_float2(_unpack_fp4_67[4], _unpack_fp4_67[5]);
                    float2 r_105 = fma_f32x2_rn_noftz(q_104, scale1_60, a_103);
                    acc_58[28] = r_105.x;
                    acc_58[29] = r_105.y;
                    float2 a_106 = make_float2(acc_58[30], acc_58[31]);
                    float2 q_107 = make_float2(_unpack_fp4_67[6], _unpack_fp4_67[7]);
                    float2 r_108 = fma_f32x2_rn_noftz(q_107, scale1_60, a_106);
                    acc_58[30] = r_108.x;
                    acc_58[31] = r_108.y;
                    unsigned int acc_58_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_58[_lp*2 + 0], acc_58[_lp*2+1 + 0]));
                        acc_58_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_58_bf16[0]), "r"(acc_58_bf16[1]), "r"(acc_58_bf16[2]), "r"(acc_58_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_58_bf16[4]), "r"(acc_58_bf16[5]), "r"(acc_58_bf16[6]), "r"(acc_58_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_58_bf16[8]), "r"(acc_58_bf16[9]), "r"(acc_58_bf16[10]), "r"(acc_58_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_58_bf16[12]), "r"(acc_58_bf16[13]), "r"(acc_58_bf16[14]), "r"(acc_58_bf16[15]) : "memory");
                    __syncwarp();
                    int c_109 = lane_1;
                    int l_110 = c_109 >> 2;
                    unsigned int w4_111[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_111[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_111[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_111[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_111[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_110 + (c_109 & 3 ^ l_110 >> 1 & 3)))));
                    long long idx_112 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_109));
                    {
                        int4 _iv4 = make_int4(w4_111[0 + 0], w4_111[0 + 1], w4_111[0 + 2], w4_111[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_112) = _iv4;
                    }
                    int c_113 = lane_1 + 32;
                    int l_114 = c_113 >> 2;
                    unsigned int w4_115[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_115[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_115[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_115[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_115[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_114 + (c_113 & 3 ^ l_114 >> 1 & 3)))));
                    long long idx_116 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_113));
                    {
                        int4 _iv4 = make_int4(w4_115[0 + 0], w4_115[0 + 1], w4_115[0 + 2], w4_115[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_116) = _iv4;
                    }
                    int c_117 = lane_1 + 64;
                    int l_118 = c_117 >> 2;
                    unsigned int w4_119[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_119[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_119[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_119[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_119[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_118 + (c_117 & 3 ^ l_118 >> 1 & 3)))));
                    long long idx_120 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_117));
                    {
                        int4 _iv4 = make_int4(w4_119[0 + 0], w4_119[0 + 1], w4_119[0 + 2], w4_119[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_120) = _iv4;
                    }
                    int c_121 = lane_1 + 96;
                    int l_122 = c_121 >> 2;
                    unsigned int w4_123[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_123[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_123[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_123[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_123[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_122 + (c_121 & 3 ^ l_122 >> 1 & 3)))));
                    long long idx_124 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_121));
                    {
                        int4 _iv4 = make_int4(w4_123[0 + 0], w4_123[0 + 1], w4_123[0 + 2], w4_123[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_124) = _iv4;
                    }
                    __syncwarp();
                    float acc_125[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_20;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_20) : "h"((uint16_t)(_fp4_wire_scales_1[2])));
                    float _cvt_f32_f16_40;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_40) : "h"((uint16_t)(_e4m3x2_to_f16x2_20)));
                    float _cvt_f32_f16_41;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_41) : "h"((uint16_t)(_e4m3x2_to_f16x2_20 >> 16)));
                    float _mul_34;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_cvt_f32_f16_40), "f"(rs_1));
                    float _mul_35;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_cvt_f32_f16_41), "f"(rs_1));
                    float2 scale0_126 = make_float2(_mul_34, _mul_34);
                    float2 scale1_127 = make_float2(_mul_35, _mul_35);
                    float _unpack_fp4_68[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[8 + 0]));
                        (&_unpack_fp4_68[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_68[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_68[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_68[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_68[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_68[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_68[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_68[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_68[0]), "=f"(_unpack_fp4_68[1]), "=f"(_unpack_fp4_68[2]), "=f"(_unpack_fp4_68[3]), "=f"(_unpack_fp4_68[4]), "=f"(_unpack_fp4_68[5]), "=f"(_unpack_fp4_68[6]), "=f"(_unpack_fp4_68[7]) : "r"(_fp4_wire_values_1[8 + 0]));
                        #endif
                    }
                    float2 a_128 = make_float2(acc_125[0], acc_125[1]);
                    float2 q_129 = make_float2(_unpack_fp4_68[0], _unpack_fp4_68[1]);
                    float2 r_130 = fma_f32x2_rn_noftz(q_129, scale0_126, a_128);
                    acc_125[0] = r_130.x;
                    acc_125[1] = r_130.y;
                    float2 a_131 = make_float2(acc_125[2], acc_125[3]);
                    float2 q_132 = make_float2(_unpack_fp4_68[2], _unpack_fp4_68[3]);
                    float2 r_133 = fma_f32x2_rn_noftz(q_132, scale0_126, a_131);
                    acc_125[2] = r_133.x;
                    acc_125[3] = r_133.y;
                    float2 a_134 = make_float2(acc_125[4], acc_125[5]);
                    float2 q_135 = make_float2(_unpack_fp4_68[4], _unpack_fp4_68[5]);
                    float2 r_136 = fma_f32x2_rn_noftz(q_135, scale0_126, a_134);
                    acc_125[4] = r_136.x;
                    acc_125[5] = r_136.y;
                    float2 a_137 = make_float2(acc_125[6], acc_125[7]);
                    float2 q_138 = make_float2(_unpack_fp4_68[6], _unpack_fp4_68[7]);
                    float2 r_139 = fma_f32x2_rn_noftz(q_138, scale0_126, a_137);
                    acc_125[6] = r_139.x;
                    acc_125[7] = r_139.y;
                    float _unpack_fp4_69[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[9 + 0]));
                        (&_unpack_fp4_69[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_69[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_69[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_69[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_69[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_69[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_69[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_69[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_69[0]), "=f"(_unpack_fp4_69[1]), "=f"(_unpack_fp4_69[2]), "=f"(_unpack_fp4_69[3]), "=f"(_unpack_fp4_69[4]), "=f"(_unpack_fp4_69[5]), "=f"(_unpack_fp4_69[6]), "=f"(_unpack_fp4_69[7]) : "r"(_fp4_wire_values_1[9 + 0]));
                        #endif
                    }
                    float2 a_140 = make_float2(acc_125[8], acc_125[9]);
                    float2 q_141 = make_float2(_unpack_fp4_69[0], _unpack_fp4_69[1]);
                    float2 r_142 = fma_f32x2_rn_noftz(q_141, scale0_126, a_140);
                    acc_125[8] = r_142.x;
                    acc_125[9] = r_142.y;
                    float2 a_143 = make_float2(acc_125[10], acc_125[11]);
                    float2 q_144 = make_float2(_unpack_fp4_69[2], _unpack_fp4_69[3]);
                    float2 r_145 = fma_f32x2_rn_noftz(q_144, scale0_126, a_143);
                    acc_125[10] = r_145.x;
                    acc_125[11] = r_145.y;
                    float2 a_146 = make_float2(acc_125[12], acc_125[13]);
                    float2 q_147 = make_float2(_unpack_fp4_69[4], _unpack_fp4_69[5]);
                    float2 r_148 = fma_f32x2_rn_noftz(q_147, scale0_126, a_146);
                    acc_125[12] = r_148.x;
                    acc_125[13] = r_148.y;
                    float2 a_149 = make_float2(acc_125[14], acc_125[15]);
                    float2 q_150 = make_float2(_unpack_fp4_69[6], _unpack_fp4_69[7]);
                    float2 r_151 = fma_f32x2_rn_noftz(q_150, scale0_126, a_149);
                    acc_125[14] = r_151.x;
                    acc_125[15] = r_151.y;
                    float _unpack_fp4_70[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[10 + 0]));
                        (&_unpack_fp4_70[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_70[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_70[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_70[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_70[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_70[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_70[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_70[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_70[0]), "=f"(_unpack_fp4_70[1]), "=f"(_unpack_fp4_70[2]), "=f"(_unpack_fp4_70[3]), "=f"(_unpack_fp4_70[4]), "=f"(_unpack_fp4_70[5]), "=f"(_unpack_fp4_70[6]), "=f"(_unpack_fp4_70[7]) : "r"(_fp4_wire_values_1[10 + 0]));
                        #endif
                    }
                    float2 a_152 = make_float2(acc_125[16], acc_125[17]);
                    float2 q_153 = make_float2(_unpack_fp4_70[0], _unpack_fp4_70[1]);
                    float2 r_154 = fma_f32x2_rn_noftz(q_153, scale1_127, a_152);
                    acc_125[16] = r_154.x;
                    acc_125[17] = r_154.y;
                    float2 a_155_1 = make_float2(acc_125[18], acc_125[19]);
                    float2 q_156_1 = make_float2(_unpack_fp4_70[2], _unpack_fp4_70[3]);
                    float2 r_157_1 = fma_f32x2_rn_noftz(q_156_1, scale1_127, a_155_1);
                    acc_125[18] = r_157_1.x;
                    acc_125[19] = r_157_1.y;
                    float2 a_158_1 = make_float2(acc_125[20], acc_125[21]);
                    float2 q_159_1 = make_float2(_unpack_fp4_70[4], _unpack_fp4_70[5]);
                    float2 r_160_1 = fma_f32x2_rn_noftz(q_159_1, scale1_127, a_158_1);
                    acc_125[20] = r_160_1.x;
                    acc_125[21] = r_160_1.y;
                    float2 a_161_1 = make_float2(acc_125[22], acc_125[23]);
                    float2 q_162_1 = make_float2(_unpack_fp4_70[6], _unpack_fp4_70[7]);
                    float2 r_163_1 = fma_f32x2_rn_noftz(q_162_1, scale1_127, a_161_1);
                    acc_125[22] = r_163_1.x;
                    acc_125[23] = r_163_1.y;
                    float _unpack_fp4_71[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[11 + 0]));
                        (&_unpack_fp4_71[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_71[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_71[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_71[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_71[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_71[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_71[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_71[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_71[0]), "=f"(_unpack_fp4_71[1]), "=f"(_unpack_fp4_71[2]), "=f"(_unpack_fp4_71[3]), "=f"(_unpack_fp4_71[4]), "=f"(_unpack_fp4_71[5]), "=f"(_unpack_fp4_71[6]), "=f"(_unpack_fp4_71[7]) : "r"(_fp4_wire_values_1[11 + 0]));
                        #endif
                    }
                    float2 a_164_1 = make_float2(acc_125[24], acc_125[25]);
                    float2 q_165_1 = make_float2(_unpack_fp4_71[0], _unpack_fp4_71[1]);
                    float2 r_166_1 = fma_f32x2_rn_noftz(q_165_1, scale1_127, a_164_1);
                    acc_125[24] = r_166_1.x;
                    acc_125[25] = r_166_1.y;
                    float2 a_167_1 = make_float2(acc_125[26], acc_125[27]);
                    float2 q_168_1 = make_float2(_unpack_fp4_71[2], _unpack_fp4_71[3]);
                    float2 r_169_1 = fma_f32x2_rn_noftz(q_168_1, scale1_127, a_167_1);
                    acc_125[26] = r_169_1.x;
                    acc_125[27] = r_169_1.y;
                    float2 a_170_1 = make_float2(acc_125[28], acc_125[29]);
                    float2 q_171_1 = make_float2(_unpack_fp4_71[4], _unpack_fp4_71[5]);
                    float2 r_172_1 = fma_f32x2_rn_noftz(q_171_1, scale1_127, a_170_1);
                    acc_125[28] = r_172_1.x;
                    acc_125[29] = r_172_1.y;
                    float2 a_173_1 = make_float2(acc_125[30], acc_125[31]);
                    float2 q_174_1 = make_float2(_unpack_fp4_71[6], _unpack_fp4_71[7]);
                    float2 r_175_1 = fma_f32x2_rn_noftz(q_174_1, scale1_127, a_173_1);
                    acc_125[30] = r_175_1.x;
                    acc_125[31] = r_175_1.y;
                    unsigned int acc_125_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_125[_lp*2 + 0], acc_125[_lp*2+1 + 0]));
                        acc_125_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_125_bf16[0]), "r"(acc_125_bf16[1]), "r"(acc_125_bf16[2]), "r"(acc_125_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_125_bf16[4]), "r"(acc_125_bf16[5]), "r"(acc_125_bf16[6]), "r"(acc_125_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_125_bf16[8]), "r"(acc_125_bf16[9]), "r"(acc_125_bf16[10]), "r"(acc_125_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_125_bf16[12]), "r"(acc_125_bf16[13]), "r"(acc_125_bf16[14]), "r"(acc_125_bf16[15]) : "memory");
                    __syncwarp();
                    int c_176 = lane_1;
                    int l_177 = c_176 >> 2;
                    unsigned int w4_178[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_178[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_178[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_178[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_178[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_177 + (c_176 & 3 ^ l_177 >> 1 & 3)))));
                    long long idx_179 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_176));
                    {
                        int4 _iv4 = make_int4(w4_178[0 + 0], w4_178[0 + 1], w4_178[0 + 2], w4_178[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_179) = _iv4;
                    }
                    int c_180 = lane_1 + 32;
                    int l_181 = c_180 >> 2;
                    unsigned int w4_182[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_182[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_182[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_182[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_182[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_181 + (c_180 & 3 ^ l_181 >> 1 & 3)))));
                    long long idx_183 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_180));
                    {
                        int4 _iv4 = make_int4(w4_182[0 + 0], w4_182[0 + 1], w4_182[0 + 2], w4_182[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_183) = _iv4;
                    }
                    int c_184 = lane_1 + 64;
                    int l_185 = c_184 >> 2;
                    unsigned int w4_186[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_186[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_186[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_186[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_186[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_185 + (c_184 & 3 ^ l_185 >> 1 & 3)))));
                    long long idx_187 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_184));
                    {
                        int4 _iv4 = make_int4(w4_186[0 + 0], w4_186[0 + 1], w4_186[0 + 2], w4_186[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_187) = _iv4;
                    }
                    int c_188 = lane_1 + 96;
                    int l_189 = c_188 >> 2;
                    unsigned int w4_190[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_190[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_190[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_190[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_190[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_189 + (c_188 & 3 ^ l_189 >> 1 & 3)))));
                    long long idx_191 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_188));
                    {
                        int4 _iv4 = make_int4(w4_190[0 + 0], w4_190[0 + 1], w4_190[0 + 2], w4_190[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_191) = _iv4;
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
                ctrl[base_1] = call + 1;
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
        }
    }

    // Cleanup
}

} // extern "C"
