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
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 212800
#define SMEM_WSM_STRIDE 212800
#define SMEM_TOTAL 226176
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
cake_fp4_allreduce_tp2_oneshot_nvfp4(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ q_wire, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + 13376);
    const int wsm_addr = smem + 13376;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 10 barriers, init_count=1
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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: workers ----
    if (warp <= 9) {
        { // workers_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            float gs_own = gs[rank];
            float gs_peer = gs[1 - rank];
            long long bid64 = (long long)bid;
            long long g64 = (long long)gridDim.x;
            long long rows64 = (long long)rows;
            int r0 = (int)(bid64 * rows64 / g64);
            int r1 = (int)((bid64 + 1) * rows64 / g64);
            int nrows = r1 - r0;
            int warp_0 = (int)warp;
            int lane_1 = (int)lane;
            int t = warp_0 >> 1;
            int h = warp_0 & 1;
            int leader = t * 2;
            if (lane_1 == 0 && h == 0 && nrows > t) {
                long long first_row = (long long)(r0 + t);
                mbarrier_arrive_expect_tx(xb_full_addr + (leader) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(leader * 21280), xq + (first_row * 3456), 3456, xb_full_addr + (leader) * 8);
            }
            if (tid == 0) {
                int base = bid * 4;
                ctl[0] = ctrl[base];
                ctl[1] = ctrl[base + 2];
                ctl[2] = ctrl[base + 3];
            }
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int n = 0;
            if (nrows > t) {
                n = (nrows - t + 5 - 1) / 5;
            }
            int f = lane_1 >> 1 & 3;
            int ring_off = leader * 21280;
            int stg_off = warp_0 * 21280 + 12288 + 6944;
            int peer = 1 - rank;
            long long cap64 = (long long)cap;
            long long rank64 = (long long)rank;
            long long peer64 = (long long)peer;
            long long buf64 = (long long)buf;
            long long q_base = buf64 * (2 * cap64 * 3456);
            uint8_t* peer_wire = reinterpret_cast<uint8_t*>(peer_ptrs[peer]);
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    int slot = k % 3;
                    int ob_off = ring_off + slot * 3472;
                    mbarrier_wait(xb_full_addr + (leader) * 8, k & 1);
                    if (h == 0 && lane_1 == 0) {
                        long long row64 = (long long)(r0 + t + 5 * k);
                        {
                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(peer_wire + (q_base + (rank64 * cap64 + row64) * 3456));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_0), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3456))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                    if (h == 0 && lane_1 == 0 && n > k + 1) {
                        asm volatile("cp.async.bulk.wait_group.read 2;");
                        long long next_row = (long long)(r0 + t + 5 * (k + 1));
                        int next_slot = (k + 1) % 3;
                        mbarrier_arrive_expect_tx(xb_full_addr + (leader) * 8, 3456);
                        cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(ring_off + next_slot * 3472), xq + (next_row * 3456), 3456, xb_full_addr + (leader) * 8);
                    }
                }
                int kd = k - 1;
                if (kd >= 0 && kd < n) {
                    long long drow = (long long)(r0 + t + 5 * kd);
                    int dslot = kd % 3;
                    int dob_off = ring_off + dslot * 3472;
                    long long src64 = peer64;
                    long long rec = q_base + (src64 * cap64 + drow) * 3456;
                    unsigned int own[12];
                    unsigned int own_sc[3];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * (lane_1 + 32 * (3 * h)))));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * (lane_1 + 32 * (3 * h)))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[4])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * (lane_1 + 32 * (3 * h + 1)))));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[1])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * (lane_1 + 32 * (3 * h + 1)))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[8])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * (lane_1 + 32 * (3 * h + 2)))));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[2])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * (lane_1 + 32 * (3 * h + 2)))));
                    uint32_t _fp4_wire_values_0[12];
                    uint32_t _fp4_wire_scales_0[3];
                    uint16_t _wire_h16_1_0;
                    uint16_t _wire_h16_1_1;
                    uint16_t _wire_h16_1_2;
                    uint32_t _wire_vpend_1 = 7u;
                    uint32_t _wire_spend_1 = 7u;
                    do {
                        if (_wire_vpend_1 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(q_wire + (rec + (long long)(16 * (lane_1 + 32 * (3 * h)))))) : "memory");
                        }
                        if (_wire_vpend_1 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(q_wire + (rec + (long long)(16 * (lane_1 + 32 * (3 * h + 1)))))) : "memory");
                        }
                        if (_wire_vpend_1 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(q_wire + (rec + (long long)(16 * (lane_1 + 32 * (3 * h + 2)))))) : "memory");
                        }
                        if (_wire_spend_1 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_1_0) : "l"(reinterpret_cast<uint16_t*>(q_wire + (rec + (long long)(3072 + 2 * (lane_1 + 32 * (3 * h)))))) : "memory");
                            _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_1_0;
                        }
                        if (_wire_spend_1 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_1_1) : "l"(reinterpret_cast<uint16_t*>(q_wire + (rec + (long long)(3072 + 2 * (lane_1 + 32 * (3 * h + 1)))))) : "memory");
                            _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_1_1;
                        }
                        if (_wire_spend_1 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_1_2) : "l"(reinterpret_cast<uint16_t*>(q_wire + (rec + (long long)(3072 + 2 * (lane_1 + 32 * (3 * h + 2)))))) : "memory");
                            _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_1_2;
                        }
                        _wire_vpend_1 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2);
                        _wire_spend_1 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2);
                    } while ((_wire_vpend_1 | _wire_spend_1) != 0u);
                    bool first = rank == 0;
                    unsigned int a[4];
                    unsigned int b[4];
                    a[0] = ((first) ? own[0] : _fp4_wire_values_0[0]);
                    b[0] = ((first) ? _fp4_wire_values_0[0] : own[0]);
                    a[1] = ((first) ? own[1] : _fp4_wire_values_0[1]);
                    b[1] = ((first) ? _fp4_wire_values_0[1] : own[1]);
                    a[2] = ((first) ? own[2] : _fp4_wire_values_0[2]);
                    b[2] = ((first) ? _fp4_wire_values_0[2] : own[2]);
                    a[3] = ((first) ? own[3] : _fp4_wire_values_0[3]);
                    b[3] = ((first) ? _fp4_wire_values_0[3] : own[3]);
                    unsigned int a_sc = ((first) ? own_sc[0] : _fp4_wire_scales_0[0]);
                    unsigned int b_sc = ((first) ? _fp4_wire_scales_0[0] : own_sc[0]);
                    float a_rs = ((first) ? gs_own : gs_peer);
                    float b_rs = ((first) ? gs_peer : gs_own);
                    float acc[32];
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    acc[16] = 0.0f;
                    acc[17] = 0.0f;
                    acc[18] = 0.0f;
                    acc[19] = 0.0f;
                    acc[20] = 0.0f;
                    acc[21] = 0.0f;
                    acc[22] = 0.0f;
                    acc[23] = 0.0f;
                    acc[24] = 0.0f;
                    acc[25] = 0.0f;
                    acc[26] = 0.0f;
                    acc[27] = 0.0f;
                    acc[28] = 0.0f;
                    acc[29] = 0.0f;
                    acc[30] = 0.0f;
                    acc[31] = 0.0f;
                    uint32_t _e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(a_sc)));
                    float _cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_e4m3x2_to_f16x2_0)));
                    float _cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(_e4m3x2_to_f16x2_0 >> 16)));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_cvt_f32_f16_0), "f"(a_rs));
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_cvt_f32_f16_1), "f"(a_rs));
                    float2 scale0 = make_float2(_mul_0, _mul_0);
                    float2 scale1 = make_float2(_mul_1, _mul_1);
                    float _unpack_fp4_0[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a[0]));
                        (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(a[0]));
                        #endif
                    }
                    float2 a_0 = make_float2(acc[0], acc[1]);
                    float2 q = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 r = fma_f32x2_rn_noftz(q, scale0, a_0);
                    acc[0] = r.x;
                    acc[1] = r.y;
                    float2 a_1 = make_float2(acc[2], acc[3]);
                    float2 q_2 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 r_3 = fma_f32x2_rn_noftz(q_2, scale0, a_1);
                    acc[2] = r_3.x;
                    acc[3] = r_3.y;
                    float2 a_4 = make_float2(acc[4], acc[5]);
                    float2 q_5 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 r_6 = fma_f32x2_rn_noftz(q_5, scale0, a_4);
                    acc[4] = r_6.x;
                    acc[5] = r_6.y;
                    float2 a_7 = make_float2(acc[6], acc[7]);
                    float2 q_8 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 r_9 = fma_f32x2_rn_noftz(q_8, scale0, a_7);
                    acc[6] = r_9.x;
                    acc[7] = r_9.y;
                    float _unpack_fp4_1[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a[1 + 0]));
                        (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(a[1 + 0]));
                        #endif
                    }
                    float2 a_10 = make_float2(acc[8], acc[9]);
                    float2 q_11 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 r_12 = fma_f32x2_rn_noftz(q_11, scale0, a_10);
                    acc[8] = r_12.x;
                    acc[9] = r_12.y;
                    float2 a_13 = make_float2(acc[10], acc[11]);
                    float2 q_14 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 r_15 = fma_f32x2_rn_noftz(q_14, scale0, a_13);
                    acc[10] = r_15.x;
                    acc[11] = r_15.y;
                    float2 a_16 = make_float2(acc[12], acc[13]);
                    float2 q_17 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 r_18 = fma_f32x2_rn_noftz(q_17, scale0, a_16);
                    acc[12] = r_18.x;
                    acc[13] = r_18.y;
                    float2 a_19 = make_float2(acc[14], acc[15]);
                    float2 q_20 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 r_21 = fma_f32x2_rn_noftz(q_20, scale0, a_19);
                    acc[14] = r_21.x;
                    acc[15] = r_21.y;
                    float _unpack_fp4_2[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a[2 + 0]));
                        (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(a[2 + 0]));
                        #endif
                    }
                    float2 a_22 = make_float2(acc[16], acc[17]);
                    float2 q_23 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 r_24 = fma_f32x2_rn_noftz(q_23, scale1, a_22);
                    acc[16] = r_24.x;
                    acc[17] = r_24.y;
                    float2 a_25 = make_float2(acc[18], acc[19]);
                    float2 q_26 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 r_27 = fma_f32x2_rn_noftz(q_26, scale1, a_25);
                    acc[18] = r_27.x;
                    acc[19] = r_27.y;
                    float2 a_28 = make_float2(acc[20], acc[21]);
                    float2 q_29 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 r_30 = fma_f32x2_rn_noftz(q_29, scale1, a_28);
                    acc[20] = r_30.x;
                    acc[21] = r_30.y;
                    float2 a_31 = make_float2(acc[22], acc[23]);
                    float2 q_32 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 r_33 = fma_f32x2_rn_noftz(q_32, scale1, a_31);
                    acc[22] = r_33.x;
                    acc[23] = r_33.y;
                    float _unpack_fp4_3[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a[3 + 0]));
                        (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(a[3 + 0]));
                        #endif
                    }
                    float2 a_34 = make_float2(acc[24], acc[25]);
                    float2 q_35 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 r_36 = fma_f32x2_rn_noftz(q_35, scale1, a_34);
                    acc[24] = r_36.x;
                    acc[25] = r_36.y;
                    float2 a_37 = make_float2(acc[26], acc[27]);
                    float2 q_38 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 r_39 = fma_f32x2_rn_noftz(q_38, scale1, a_37);
                    acc[26] = r_39.x;
                    acc[27] = r_39.y;
                    float2 a_40 = make_float2(acc[28], acc[29]);
                    float2 q_41 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 r_42 = fma_f32x2_rn_noftz(q_41, scale1, a_40);
                    acc[28] = r_42.x;
                    acc[29] = r_42.y;
                    float2 a_43 = make_float2(acc[30], acc[31]);
                    float2 q_44 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 r_45 = fma_f32x2_rn_noftz(q_44, scale1, a_43);
                    acc[30] = r_45.x;
                    acc[31] = r_45.y;
                    uint32_t _e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(b_sc)));
                    float _cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(_e4m3x2_to_f16x2_1)));
                    float _cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(_e4m3x2_to_f16x2_1 >> 16)));
                    float _mul_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_cvt_f32_f16_2), "f"(b_rs));
                    float _mul_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_cvt_f32_f16_3), "f"(b_rs));
                    float2 scale0_46 = make_float2(_mul_2, _mul_2);
                    float2 scale1_47 = make_float2(_mul_3, _mul_3);
                    float _unpack_fp4_4[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b[0]));
                        (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(b[0]));
                        #endif
                    }
                    float2 a_48 = make_float2(acc[0], acc[1]);
                    float2 q_49 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 r_50 = fma_f32x2_rn_noftz(q_49, scale0_46, a_48);
                    acc[0] = r_50.x;
                    acc[1] = r_50.y;
                    float2 a_51 = make_float2(acc[2], acc[3]);
                    float2 q_52 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 r_53 = fma_f32x2_rn_noftz(q_52, scale0_46, a_51);
                    acc[2] = r_53.x;
                    acc[3] = r_53.y;
                    float2 a_54 = make_float2(acc[4], acc[5]);
                    float2 q_55 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 r_56 = fma_f32x2_rn_noftz(q_55, scale0_46, a_54);
                    acc[4] = r_56.x;
                    acc[5] = r_56.y;
                    float2 a_57 = make_float2(acc[6], acc[7]);
                    float2 q_58 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 r_59 = fma_f32x2_rn_noftz(q_58, scale0_46, a_57);
                    acc[6] = r_59.x;
                    acc[7] = r_59.y;
                    float _unpack_fp4_5[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b[1 + 0]));
                        (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(b[1 + 0]));
                        #endif
                    }
                    float2 a_60 = make_float2(acc[8], acc[9]);
                    float2 q_61 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 r_62 = fma_f32x2_rn_noftz(q_61, scale0_46, a_60);
                    acc[8] = r_62.x;
                    acc[9] = r_62.y;
                    float2 a_63 = make_float2(acc[10], acc[11]);
                    float2 q_64 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 r_65 = fma_f32x2_rn_noftz(q_64, scale0_46, a_63);
                    acc[10] = r_65.x;
                    acc[11] = r_65.y;
                    float2 a_66 = make_float2(acc[12], acc[13]);
                    float2 q_67 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 r_68 = fma_f32x2_rn_noftz(q_67, scale0_46, a_66);
                    acc[12] = r_68.x;
                    acc[13] = r_68.y;
                    float2 a_69 = make_float2(acc[14], acc[15]);
                    float2 q_70 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 r_71 = fma_f32x2_rn_noftz(q_70, scale0_46, a_69);
                    acc[14] = r_71.x;
                    acc[15] = r_71.y;
                    float _unpack_fp4_6[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b[2 + 0]));
                        (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(b[2 + 0]));
                        #endif
                    }
                    float2 a_72 = make_float2(acc[16], acc[17]);
                    float2 q_73 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 r_74 = fma_f32x2_rn_noftz(q_73, scale1_47, a_72);
                    acc[16] = r_74.x;
                    acc[17] = r_74.y;
                    float2 a_75 = make_float2(acc[18], acc[19]);
                    float2 q_76 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 r_77 = fma_f32x2_rn_noftz(q_76, scale1_47, a_75);
                    acc[18] = r_77.x;
                    acc[19] = r_77.y;
                    float2 a_78 = make_float2(acc[20], acc[21]);
                    float2 q_79 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 r_80 = fma_f32x2_rn_noftz(q_79, scale1_47, a_78);
                    acc[20] = r_80.x;
                    acc[21] = r_80.y;
                    float2 a_81 = make_float2(acc[22], acc[23]);
                    float2 q_82 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 r_83 = fma_f32x2_rn_noftz(q_82, scale1_47, a_81);
                    acc[22] = r_83.x;
                    acc[23] = r_83.y;
                    float _unpack_fp4_7[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b[3 + 0]));
                        (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(b[3 + 0]));
                        #endif
                    }
                    float2 a_84 = make_float2(acc[24], acc[25]);
                    float2 q_85 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 r_86 = fma_f32x2_rn_noftz(q_85, scale1_47, a_84);
                    acc[24] = r_86.x;
                    acc[25] = r_86.y;
                    float2 a_87 = make_float2(acc[26], acc[27]);
                    float2 q_88 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 r_89 = fma_f32x2_rn_noftz(q_88, scale1_47, a_87);
                    acc[26] = r_89.x;
                    acc[27] = r_89.y;
                    float2 a_90 = make_float2(acc[28], acc[29]);
                    float2 q_91 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 r_92 = fma_f32x2_rn_noftz(q_91, scale1_47, a_90);
                    acc[28] = r_92.x;
                    acc[29] = r_92.y;
                    float2 a_93 = make_float2(acc[30], acc[31]);
                    float2 q_94 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 r_95 = fma_f32x2_rn_noftz(q_94, scale1_47, a_93);
                    acc[30] = r_95.x;
                    acc[31] = r_95.y;
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
                    int c = lane_1;
                    int l = c >> 2;
                    unsigned int w4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l + (c & 3 ^ l >> 1 & 3)))));
                    long long idx = drow * 3072 + (long long)(4 * (128 * (3 * h) + c));
                    {
                        int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx) = _iv4;
                    }
                    int c_96 = lane_1 + 32;
                    int l_97 = c_96 >> 2;
                    unsigned int w4_98[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_98[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_98[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_98[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_98[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_97 + (c_96 & 3 ^ l_97 >> 1 & 3)))));
                    long long idx_99 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_96));
                    {
                        int4 _iv4 = make_int4(w4_98[0 + 0], w4_98[0 + 1], w4_98[0 + 2], w4_98[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_99) = _iv4;
                    }
                    int c_100 = lane_1 + 64;
                    int l_101 = c_100 >> 2;
                    unsigned int w4_102[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_102[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_102[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_102[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_102[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_101 + (c_100 & 3 ^ l_101 >> 1 & 3)))));
                    long long idx_103 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_100));
                    {
                        int4 _iv4 = make_int4(w4_102[0 + 0], w4_102[0 + 1], w4_102[0 + 2], w4_102[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_103) = _iv4;
                    }
                    int c_104 = lane_1 + 96;
                    int l_105 = c_104 >> 2;
                    unsigned int w4_106[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_106[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_106[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_106[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_106[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_105 + (c_104 & 3 ^ l_105 >> 1 & 3)))));
                    long long idx_107 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_104));
                    {
                        int4 _iv4 = make_int4(w4_106[0 + 0], w4_106[0 + 1], w4_106[0 + 2], w4_106[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_107) = _iv4;
                    }
                    __syncwarp();
                    unsigned int a_108[4];
                    unsigned int b_109[4];
                    a_108[0] = ((first) ? own[4] : _fp4_wire_values_0[4]);
                    b_109[0] = ((first) ? _fp4_wire_values_0[4] : own[4]);
                    a_108[1] = ((first) ? own[5] : _fp4_wire_values_0[5]);
                    b_109[1] = ((first) ? _fp4_wire_values_0[5] : own[5]);
                    a_108[2] = ((first) ? own[6] : _fp4_wire_values_0[6]);
                    b_109[2] = ((first) ? _fp4_wire_values_0[6] : own[6]);
                    a_108[3] = ((first) ? own[7] : _fp4_wire_values_0[7]);
                    b_109[3] = ((first) ? _fp4_wire_values_0[7] : own[7]);
                    unsigned int a_sc_110 = ((first) ? own_sc[1] : _fp4_wire_scales_0[1]);
                    unsigned int b_sc_111 = ((first) ? _fp4_wire_scales_0[1] : own_sc[1]);
                    float a_rs_112 = ((first) ? gs_own : gs_peer);
                    float b_rs_113 = ((first) ? gs_peer : gs_own);
                    float acc_114[32];
                    acc_114[0] = 0.0f;
                    acc_114[1] = 0.0f;
                    acc_114[2] = 0.0f;
                    acc_114[3] = 0.0f;
                    acc_114[4] = 0.0f;
                    acc_114[5] = 0.0f;
                    acc_114[6] = 0.0f;
                    acc_114[7] = 0.0f;
                    acc_114[8] = 0.0f;
                    acc_114[9] = 0.0f;
                    acc_114[10] = 0.0f;
                    acc_114[11] = 0.0f;
                    acc_114[12] = 0.0f;
                    acc_114[13] = 0.0f;
                    acc_114[14] = 0.0f;
                    acc_114[15] = 0.0f;
                    acc_114[16] = 0.0f;
                    acc_114[17] = 0.0f;
                    acc_114[18] = 0.0f;
                    acc_114[19] = 0.0f;
                    acc_114[20] = 0.0f;
                    acc_114[21] = 0.0f;
                    acc_114[22] = 0.0f;
                    acc_114[23] = 0.0f;
                    acc_114[24] = 0.0f;
                    acc_114[25] = 0.0f;
                    acc_114[26] = 0.0f;
                    acc_114[27] = 0.0f;
                    acc_114[28] = 0.0f;
                    acc_114[29] = 0.0f;
                    acc_114[30] = 0.0f;
                    acc_114[31] = 0.0f;
                    uint32_t _e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(a_sc_110)));
                    float _cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(_e4m3x2_to_f16x2_2)));
                    float _cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(_e4m3x2_to_f16x2_2 >> 16)));
                    float _mul_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_cvt_f32_f16_4), "f"(a_rs_112));
                    float _mul_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_cvt_f32_f16_5), "f"(a_rs_112));
                    float2 scale0_115 = make_float2(_mul_4, _mul_4);
                    float2 scale1_116 = make_float2(_mul_5, _mul_5);
                    float _unpack_fp4_8[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_108[0]));
                        (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(a_108[0]));
                        #endif
                    }
                    float2 a_117 = make_float2(acc_114[0], acc_114[1]);
                    float2 q_118 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 r_119 = fma_f32x2_rn_noftz(q_118, scale0_115, a_117);
                    acc_114[0] = r_119.x;
                    acc_114[1] = r_119.y;
                    float2 a_120 = make_float2(acc_114[2], acc_114[3]);
                    float2 q_121 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 r_122 = fma_f32x2_rn_noftz(q_121, scale0_115, a_120);
                    acc_114[2] = r_122.x;
                    acc_114[3] = r_122.y;
                    float2 a_123 = make_float2(acc_114[4], acc_114[5]);
                    float2 q_124 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 r_125 = fma_f32x2_rn_noftz(q_124, scale0_115, a_123);
                    acc_114[4] = r_125.x;
                    acc_114[5] = r_125.y;
                    float2 a_126 = make_float2(acc_114[6], acc_114[7]);
                    float2 q_127 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 r_128 = fma_f32x2_rn_noftz(q_127, scale0_115, a_126);
                    acc_114[6] = r_128.x;
                    acc_114[7] = r_128.y;
                    float _unpack_fp4_9[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_108[1 + 0]));
                        (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(a_108[1 + 0]));
                        #endif
                    }
                    float2 a_129 = make_float2(acc_114[8], acc_114[9]);
                    float2 q_130 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 r_131 = fma_f32x2_rn_noftz(q_130, scale0_115, a_129);
                    acc_114[8] = r_131.x;
                    acc_114[9] = r_131.y;
                    float2 a_132 = make_float2(acc_114[10], acc_114[11]);
                    float2 q_133 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 r_134 = fma_f32x2_rn_noftz(q_133, scale0_115, a_132);
                    acc_114[10] = r_134.x;
                    acc_114[11] = r_134.y;
                    float2 a_135 = make_float2(acc_114[12], acc_114[13]);
                    float2 q_136 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 r_137 = fma_f32x2_rn_noftz(q_136, scale0_115, a_135);
                    acc_114[12] = r_137.x;
                    acc_114[13] = r_137.y;
                    float2 a_138 = make_float2(acc_114[14], acc_114[15]);
                    float2 q_139 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 r_140 = fma_f32x2_rn_noftz(q_139, scale0_115, a_138);
                    acc_114[14] = r_140.x;
                    acc_114[15] = r_140.y;
                    float _unpack_fp4_10[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_108[2 + 0]));
                        (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(a_108[2 + 0]));
                        #endif
                    }
                    float2 a_141 = make_float2(acc_114[16], acc_114[17]);
                    float2 q_142 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 r_143 = fma_f32x2_rn_noftz(q_142, scale1_116, a_141);
                    acc_114[16] = r_143.x;
                    acc_114[17] = r_143.y;
                    float2 a_144 = make_float2(acc_114[18], acc_114[19]);
                    float2 q_145 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 r_146 = fma_f32x2_rn_noftz(q_145, scale1_116, a_144);
                    acc_114[18] = r_146.x;
                    acc_114[19] = r_146.y;
                    float2 a_147 = make_float2(acc_114[20], acc_114[21]);
                    float2 q_148 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 r_149 = fma_f32x2_rn_noftz(q_148, scale1_116, a_147);
                    acc_114[20] = r_149.x;
                    acc_114[21] = r_149.y;
                    float2 a_150 = make_float2(acc_114[22], acc_114[23]);
                    float2 q_151 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 r_152 = fma_f32x2_rn_noftz(q_151, scale1_116, a_150);
                    acc_114[22] = r_152.x;
                    acc_114[23] = r_152.y;
                    float _unpack_fp4_11[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_108[3 + 0]));
                        (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(a_108[3 + 0]));
                        #endif
                    }
                    float2 a_153 = make_float2(acc_114[24], acc_114[25]);
                    float2 q_154 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 r_155 = fma_f32x2_rn_noftz(q_154, scale1_116, a_153);
                    acc_114[24] = r_155.x;
                    acc_114[25] = r_155.y;
                    float2 a_156 = make_float2(acc_114[26], acc_114[27]);
                    float2 q_157 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 r_158 = fma_f32x2_rn_noftz(q_157, scale1_116, a_156);
                    acc_114[26] = r_158.x;
                    acc_114[27] = r_158.y;
                    float2 a_159 = make_float2(acc_114[28], acc_114[29]);
                    float2 q_160 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 r_161 = fma_f32x2_rn_noftz(q_160, scale1_116, a_159);
                    acc_114[28] = r_161.x;
                    acc_114[29] = r_161.y;
                    float2 a_162 = make_float2(acc_114[30], acc_114[31]);
                    float2 q_163 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 r_164 = fma_f32x2_rn_noftz(q_163, scale1_116, a_162);
                    acc_114[30] = r_164.x;
                    acc_114[31] = r_164.y;
                    uint32_t _e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(b_sc_111)));
                    float _cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(_e4m3x2_to_f16x2_3)));
                    float _cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(_e4m3x2_to_f16x2_3 >> 16)));
                    float _mul_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_cvt_f32_f16_6), "f"(b_rs_113));
                    float _mul_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_cvt_f32_f16_7), "f"(b_rs_113));
                    float2 scale0_165 = make_float2(_mul_6, _mul_6);
                    float2 scale1_166 = make_float2(_mul_7, _mul_7);
                    float _unpack_fp4_12[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_109[0]));
                        (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(b_109[0]));
                        #endif
                    }
                    float2 a_167 = make_float2(acc_114[0], acc_114[1]);
                    float2 q_168 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 r_169 = fma_f32x2_rn_noftz(q_168, scale0_165, a_167);
                    acc_114[0] = r_169.x;
                    acc_114[1] = r_169.y;
                    float2 a_170 = make_float2(acc_114[2], acc_114[3]);
                    float2 q_171 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 r_172 = fma_f32x2_rn_noftz(q_171, scale0_165, a_170);
                    acc_114[2] = r_172.x;
                    acc_114[3] = r_172.y;
                    float2 a_173 = make_float2(acc_114[4], acc_114[5]);
                    float2 q_174 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 r_175 = fma_f32x2_rn_noftz(q_174, scale0_165, a_173);
                    acc_114[4] = r_175.x;
                    acc_114[5] = r_175.y;
                    float2 a_176 = make_float2(acc_114[6], acc_114[7]);
                    float2 q_177 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 r_178 = fma_f32x2_rn_noftz(q_177, scale0_165, a_176);
                    acc_114[6] = r_178.x;
                    acc_114[7] = r_178.y;
                    float _unpack_fp4_13[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_109[1 + 0]));
                        (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(b_109[1 + 0]));
                        #endif
                    }
                    float2 a_179 = make_float2(acc_114[8], acc_114[9]);
                    float2 q_180 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 r_181 = fma_f32x2_rn_noftz(q_180, scale0_165, a_179);
                    acc_114[8] = r_181.x;
                    acc_114[9] = r_181.y;
                    float2 a_182 = make_float2(acc_114[10], acc_114[11]);
                    float2 q_183 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 r_184 = fma_f32x2_rn_noftz(q_183, scale0_165, a_182);
                    acc_114[10] = r_184.x;
                    acc_114[11] = r_184.y;
                    float2 a_185 = make_float2(acc_114[12], acc_114[13]);
                    float2 q_186 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 r_187 = fma_f32x2_rn_noftz(q_186, scale0_165, a_185);
                    acc_114[12] = r_187.x;
                    acc_114[13] = r_187.y;
                    float2 a_188 = make_float2(acc_114[14], acc_114[15]);
                    float2 q_189 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 r_190 = fma_f32x2_rn_noftz(q_189, scale0_165, a_188);
                    acc_114[14] = r_190.x;
                    acc_114[15] = r_190.y;
                    float _unpack_fp4_14[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_109[2 + 0]));
                        (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(b_109[2 + 0]));
                        #endif
                    }
                    float2 a_191 = make_float2(acc_114[16], acc_114[17]);
                    float2 q_192 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 r_193 = fma_f32x2_rn_noftz(q_192, scale1_166, a_191);
                    acc_114[16] = r_193.x;
                    acc_114[17] = r_193.y;
                    float2 a_194 = make_float2(acc_114[18], acc_114[19]);
                    float2 q_195 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 r_196 = fma_f32x2_rn_noftz(q_195, scale1_166, a_194);
                    acc_114[18] = r_196.x;
                    acc_114[19] = r_196.y;
                    float2 a_197 = make_float2(acc_114[20], acc_114[21]);
                    float2 q_198 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 r_199 = fma_f32x2_rn_noftz(q_198, scale1_166, a_197);
                    acc_114[20] = r_199.x;
                    acc_114[21] = r_199.y;
                    float2 a_200 = make_float2(acc_114[22], acc_114[23]);
                    float2 q_201 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 r_202 = fma_f32x2_rn_noftz(q_201, scale1_166, a_200);
                    acc_114[22] = r_202.x;
                    acc_114[23] = r_202.y;
                    float _unpack_fp4_15[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_109[3 + 0]));
                        (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(b_109[3 + 0]));
                        #endif
                    }
                    float2 a_203 = make_float2(acc_114[24], acc_114[25]);
                    float2 q_204 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 r_205 = fma_f32x2_rn_noftz(q_204, scale1_166, a_203);
                    acc_114[24] = r_205.x;
                    acc_114[25] = r_205.y;
                    float2 a_206 = make_float2(acc_114[26], acc_114[27]);
                    float2 q_207 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 r_208 = fma_f32x2_rn_noftz(q_207, scale1_166, a_206);
                    acc_114[26] = r_208.x;
                    acc_114[27] = r_208.y;
                    float2 a_209 = make_float2(acc_114[28], acc_114[29]);
                    float2 q_210 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 r_211 = fma_f32x2_rn_noftz(q_210, scale1_166, a_209);
                    acc_114[28] = r_211.x;
                    acc_114[29] = r_211.y;
                    float2 a_212 = make_float2(acc_114[30], acc_114[31]);
                    float2 q_213 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 r_214 = fma_f32x2_rn_noftz(q_213, scale1_166, a_212);
                    acc_114[30] = r_214.x;
                    acc_114[31] = r_214.y;
                    unsigned int acc_114_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_114[_lp*2 + 0], acc_114[_lp*2+1 + 0]));
                        acc_114_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_114_bf16[0]), "r"(acc_114_bf16[1]), "r"(acc_114_bf16[2]), "r"(acc_114_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_114_bf16[4]), "r"(acc_114_bf16[5]), "r"(acc_114_bf16[6]), "r"(acc_114_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_114_bf16[8]), "r"(acc_114_bf16[9]), "r"(acc_114_bf16[10]), "r"(acc_114_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_114_bf16[12]), "r"(acc_114_bf16[13]), "r"(acc_114_bf16[14]), "r"(acc_114_bf16[15]) : "memory");
                    __syncwarp();
                    int c_215 = lane_1;
                    int l_216 = c_215 >> 2;
                    unsigned int w4_217[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_217[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_217[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_217[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_217[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_216 + (c_215 & 3 ^ l_216 >> 1 & 3)))));
                    long long idx_218 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_215));
                    {
                        int4 _iv4 = make_int4(w4_217[0 + 0], w4_217[0 + 1], w4_217[0 + 2], w4_217[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_218) = _iv4;
                    }
                    int c_219 = lane_1 + 32;
                    int l_220 = c_219 >> 2;
                    unsigned int w4_221[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_221[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_221[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_221[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_221[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_220 + (c_219 & 3 ^ l_220 >> 1 & 3)))));
                    long long idx_222 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_219));
                    {
                        int4 _iv4 = make_int4(w4_221[0 + 0], w4_221[0 + 1], w4_221[0 + 2], w4_221[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_222) = _iv4;
                    }
                    int c_223 = lane_1 + 64;
                    int l_224 = c_223 >> 2;
                    unsigned int w4_225[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_225[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_225[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_225[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_225[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_224 + (c_223 & 3 ^ l_224 >> 1 & 3)))));
                    long long idx_226 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_223));
                    {
                        int4 _iv4 = make_int4(w4_225[0 + 0], w4_225[0 + 1], w4_225[0 + 2], w4_225[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_226) = _iv4;
                    }
                    int c_227 = lane_1 + 96;
                    int l_228 = c_227 >> 2;
                    unsigned int w4_229[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_229[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_229[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_229[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_229[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_228 + (c_227 & 3 ^ l_228 >> 1 & 3)))));
                    long long idx_230 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_227));
                    {
                        int4 _iv4 = make_int4(w4_229[0 + 0], w4_229[0 + 1], w4_229[0 + 2], w4_229[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_230) = _iv4;
                    }
                    __syncwarp();
                    unsigned int a_231[4];
                    unsigned int b_232[4];
                    a_231[0] = ((first) ? own[8] : _fp4_wire_values_0[8]);
                    b_232[0] = ((first) ? _fp4_wire_values_0[8] : own[8]);
                    a_231[1] = ((first) ? own[9] : _fp4_wire_values_0[9]);
                    b_232[1] = ((first) ? _fp4_wire_values_0[9] : own[9]);
                    a_231[2] = ((first) ? own[10] : _fp4_wire_values_0[10]);
                    b_232[2] = ((first) ? _fp4_wire_values_0[10] : own[10]);
                    a_231[3] = ((first) ? own[11] : _fp4_wire_values_0[11]);
                    b_232[3] = ((first) ? _fp4_wire_values_0[11] : own[11]);
                    unsigned int a_sc_233 = ((first) ? own_sc[2] : _fp4_wire_scales_0[2]);
                    unsigned int b_sc_234 = ((first) ? _fp4_wire_scales_0[2] : own_sc[2]);
                    float a_rs_235 = ((first) ? gs_own : gs_peer);
                    float b_rs_236 = ((first) ? gs_peer : gs_own);
                    float acc_237[32];
                    acc_237[0] = 0.0f;
                    acc_237[1] = 0.0f;
                    acc_237[2] = 0.0f;
                    acc_237[3] = 0.0f;
                    acc_237[4] = 0.0f;
                    acc_237[5] = 0.0f;
                    acc_237[6] = 0.0f;
                    acc_237[7] = 0.0f;
                    acc_237[8] = 0.0f;
                    acc_237[9] = 0.0f;
                    acc_237[10] = 0.0f;
                    acc_237[11] = 0.0f;
                    acc_237[12] = 0.0f;
                    acc_237[13] = 0.0f;
                    acc_237[14] = 0.0f;
                    acc_237[15] = 0.0f;
                    acc_237[16] = 0.0f;
                    acc_237[17] = 0.0f;
                    acc_237[18] = 0.0f;
                    acc_237[19] = 0.0f;
                    acc_237[20] = 0.0f;
                    acc_237[21] = 0.0f;
                    acc_237[22] = 0.0f;
                    acc_237[23] = 0.0f;
                    acc_237[24] = 0.0f;
                    acc_237[25] = 0.0f;
                    acc_237[26] = 0.0f;
                    acc_237[27] = 0.0f;
                    acc_237[28] = 0.0f;
                    acc_237[29] = 0.0f;
                    acc_237[30] = 0.0f;
                    acc_237[31] = 0.0f;
                    uint32_t _e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(a_sc_233)));
                    float _cvt_f32_f16_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(_e4m3x2_to_f16x2_4)));
                    float _cvt_f32_f16_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(_e4m3x2_to_f16x2_4 >> 16)));
                    float _mul_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_cvt_f32_f16_8), "f"(a_rs_235));
                    float _mul_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_cvt_f32_f16_9), "f"(a_rs_235));
                    float2 scale0_238 = make_float2(_mul_8, _mul_8);
                    float2 scale1_239 = make_float2(_mul_9, _mul_9);
                    float _unpack_fp4_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_231[0]));
                        (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(a_231[0]));
                        #endif
                    }
                    float2 a_240 = make_float2(acc_237[0], acc_237[1]);
                    float2 q_241 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 r_242 = fma_f32x2_rn_noftz(q_241, scale0_238, a_240);
                    acc_237[0] = r_242.x;
                    acc_237[1] = r_242.y;
                    float2 a_243 = make_float2(acc_237[2], acc_237[3]);
                    float2 q_244 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 r_245 = fma_f32x2_rn_noftz(q_244, scale0_238, a_243);
                    acc_237[2] = r_245.x;
                    acc_237[3] = r_245.y;
                    float2 a_246 = make_float2(acc_237[4], acc_237[5]);
                    float2 q_247 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 r_248 = fma_f32x2_rn_noftz(q_247, scale0_238, a_246);
                    acc_237[4] = r_248.x;
                    acc_237[5] = r_248.y;
                    float2 a_249 = make_float2(acc_237[6], acc_237[7]);
                    float2 q_250 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 r_251 = fma_f32x2_rn_noftz(q_250, scale0_238, a_249);
                    acc_237[6] = r_251.x;
                    acc_237[7] = r_251.y;
                    float _unpack_fp4_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_231[1 + 0]));
                        (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(a_231[1 + 0]));
                        #endif
                    }
                    float2 a_252 = make_float2(acc_237[8], acc_237[9]);
                    float2 q_253 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 r_254 = fma_f32x2_rn_noftz(q_253, scale0_238, a_252);
                    acc_237[8] = r_254.x;
                    acc_237[9] = r_254.y;
                    float2 a_255 = make_float2(acc_237[10], acc_237[11]);
                    float2 q_256 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 r_257 = fma_f32x2_rn_noftz(q_256, scale0_238, a_255);
                    acc_237[10] = r_257.x;
                    acc_237[11] = r_257.y;
                    float2 a_258 = make_float2(acc_237[12], acc_237[13]);
                    float2 q_259 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 r_260 = fma_f32x2_rn_noftz(q_259, scale0_238, a_258);
                    acc_237[12] = r_260.x;
                    acc_237[13] = r_260.y;
                    float2 a_261 = make_float2(acc_237[14], acc_237[15]);
                    float2 q_262 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 r_263 = fma_f32x2_rn_noftz(q_262, scale0_238, a_261);
                    acc_237[14] = r_263.x;
                    acc_237[15] = r_263.y;
                    float _unpack_fp4_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_231[2 + 0]));
                        (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(a_231[2 + 0]));
                        #endif
                    }
                    float2 a_264 = make_float2(acc_237[16], acc_237[17]);
                    float2 q_265 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 r_266 = fma_f32x2_rn_noftz(q_265, scale1_239, a_264);
                    acc_237[16] = r_266.x;
                    acc_237[17] = r_266.y;
                    float2 a_267 = make_float2(acc_237[18], acc_237[19]);
                    float2 q_268 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 r_269 = fma_f32x2_rn_noftz(q_268, scale1_239, a_267);
                    acc_237[18] = r_269.x;
                    acc_237[19] = r_269.y;
                    float2 a_270 = make_float2(acc_237[20], acc_237[21]);
                    float2 q_271 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 r_272 = fma_f32x2_rn_noftz(q_271, scale1_239, a_270);
                    acc_237[20] = r_272.x;
                    acc_237[21] = r_272.y;
                    float2 a_273 = make_float2(acc_237[22], acc_237[23]);
                    float2 q_274 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 r_275 = fma_f32x2_rn_noftz(q_274, scale1_239, a_273);
                    acc_237[22] = r_275.x;
                    acc_237[23] = r_275.y;
                    float _unpack_fp4_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_231[3 + 0]));
                        (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(a_231[3 + 0]));
                        #endif
                    }
                    float2 a_276 = make_float2(acc_237[24], acc_237[25]);
                    float2 q_277 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 r_278 = fma_f32x2_rn_noftz(q_277, scale1_239, a_276);
                    acc_237[24] = r_278.x;
                    acc_237[25] = r_278.y;
                    float2 a_279 = make_float2(acc_237[26], acc_237[27]);
                    float2 q_280 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 r_281 = fma_f32x2_rn_noftz(q_280, scale1_239, a_279);
                    acc_237[26] = r_281.x;
                    acc_237[27] = r_281.y;
                    float2 a_282 = make_float2(acc_237[28], acc_237[29]);
                    float2 q_283 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 r_284 = fma_f32x2_rn_noftz(q_283, scale1_239, a_282);
                    acc_237[28] = r_284.x;
                    acc_237[29] = r_284.y;
                    float2 a_285 = make_float2(acc_237[30], acc_237[31]);
                    float2 q_286 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 r_287 = fma_f32x2_rn_noftz(q_286, scale1_239, a_285);
                    acc_237[30] = r_287.x;
                    acc_237[31] = r_287.y;
                    uint32_t _e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(b_sc_234)));
                    float _cvt_f32_f16_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(_e4m3x2_to_f16x2_5)));
                    float _cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(_e4m3x2_to_f16x2_5 >> 16)));
                    float _mul_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_cvt_f32_f16_10), "f"(b_rs_236));
                    float _mul_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_cvt_f32_f16_11), "f"(b_rs_236));
                    float2 scale0_288 = make_float2(_mul_10, _mul_10);
                    float2 scale1_289 = make_float2(_mul_11, _mul_11);
                    float _unpack_fp4_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_232[0]));
                        (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(b_232[0]));
                        #endif
                    }
                    float2 a_290 = make_float2(acc_237[0], acc_237[1]);
                    float2 q_291 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 r_292 = fma_f32x2_rn_noftz(q_291, scale0_288, a_290);
                    acc_237[0] = r_292.x;
                    acc_237[1] = r_292.y;
                    float2 a_293 = make_float2(acc_237[2], acc_237[3]);
                    float2 q_294 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 r_295 = fma_f32x2_rn_noftz(q_294, scale0_288, a_293);
                    acc_237[2] = r_295.x;
                    acc_237[3] = r_295.y;
                    float2 a_296 = make_float2(acc_237[4], acc_237[5]);
                    float2 q_297 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 r_298 = fma_f32x2_rn_noftz(q_297, scale0_288, a_296);
                    acc_237[4] = r_298.x;
                    acc_237[5] = r_298.y;
                    float2 a_299 = make_float2(acc_237[6], acc_237[7]);
                    float2 q_300 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 r_301 = fma_f32x2_rn_noftz(q_300, scale0_288, a_299);
                    acc_237[6] = r_301.x;
                    acc_237[7] = r_301.y;
                    float _unpack_fp4_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_232[1 + 0]));
                        (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(b_232[1 + 0]));
                        #endif
                    }
                    float2 a_302 = make_float2(acc_237[8], acc_237[9]);
                    float2 q_303 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 r_304 = fma_f32x2_rn_noftz(q_303, scale0_288, a_302);
                    acc_237[8] = r_304.x;
                    acc_237[9] = r_304.y;
                    float2 a_305 = make_float2(acc_237[10], acc_237[11]);
                    float2 q_306 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 r_307 = fma_f32x2_rn_noftz(q_306, scale0_288, a_305);
                    acc_237[10] = r_307.x;
                    acc_237[11] = r_307.y;
                    float2 a_308 = make_float2(acc_237[12], acc_237[13]);
                    float2 q_309 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 r_310 = fma_f32x2_rn_noftz(q_309, scale0_288, a_308);
                    acc_237[12] = r_310.x;
                    acc_237[13] = r_310.y;
                    float2 a_311 = make_float2(acc_237[14], acc_237[15]);
                    float2 q_312 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 r_313 = fma_f32x2_rn_noftz(q_312, scale0_288, a_311);
                    acc_237[14] = r_313.x;
                    acc_237[15] = r_313.y;
                    float _unpack_fp4_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_232[2 + 0]));
                        (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(b_232[2 + 0]));
                        #endif
                    }
                    float2 a_314 = make_float2(acc_237[16], acc_237[17]);
                    float2 q_315 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 r_316 = fma_f32x2_rn_noftz(q_315, scale1_289, a_314);
                    acc_237[16] = r_316.x;
                    acc_237[17] = r_316.y;
                    float2 a_317 = make_float2(acc_237[18], acc_237[19]);
                    float2 q_318 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 r_319 = fma_f32x2_rn_noftz(q_318, scale1_289, a_317);
                    acc_237[18] = r_319.x;
                    acc_237[19] = r_319.y;
                    float2 a_320 = make_float2(acc_237[20], acc_237[21]);
                    float2 q_321 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 r_322 = fma_f32x2_rn_noftz(q_321, scale1_289, a_320);
                    acc_237[20] = r_322.x;
                    acc_237[21] = r_322.y;
                    float2 a_323 = make_float2(acc_237[22], acc_237[23]);
                    float2 q_324 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 r_325 = fma_f32x2_rn_noftz(q_324, scale1_289, a_323);
                    acc_237[22] = r_325.x;
                    acc_237[23] = r_325.y;
                    float _unpack_fp4_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_232[3 + 0]));
                        (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(b_232[3 + 0]));
                        #endif
                    }
                    float2 a_326 = make_float2(acc_237[24], acc_237[25]);
                    float2 q_327 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 r_328 = fma_f32x2_rn_noftz(q_327, scale1_289, a_326);
                    acc_237[24] = r_328.x;
                    acc_237[25] = r_328.y;
                    float2 a_329 = make_float2(acc_237[26], acc_237[27]);
                    float2 q_330 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 r_331 = fma_f32x2_rn_noftz(q_330, scale1_289, a_329);
                    acc_237[26] = r_331.x;
                    acc_237[27] = r_331.y;
                    float2 a_332 = make_float2(acc_237[28], acc_237[29]);
                    float2 q_333 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 r_334 = fma_f32x2_rn_noftz(q_333, scale1_289, a_332);
                    acc_237[28] = r_334.x;
                    acc_237[29] = r_334.y;
                    float2 a_335 = make_float2(acc_237[30], acc_237[31]);
                    float2 q_336 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 r_337 = fma_f32x2_rn_noftz(q_336, scale1_289, a_335);
                    acc_237[30] = r_337.x;
                    acc_237[31] = r_337.y;
                    unsigned int acc_237_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_237[_lp*2 + 0], acc_237[_lp*2+1 + 0]));
                        acc_237_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_237_bf16[0]), "r"(acc_237_bf16[1]), "r"(acc_237_bf16[2]), "r"(acc_237_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_237_bf16[4]), "r"(acc_237_bf16[5]), "r"(acc_237_bf16[6]), "r"(acc_237_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_237_bf16[8]), "r"(acc_237_bf16[9]), "r"(acc_237_bf16[10]), "r"(acc_237_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_237_bf16[12]), "r"(acc_237_bf16[13]), "r"(acc_237_bf16[14]), "r"(acc_237_bf16[15]) : "memory");
                    __syncwarp();
                    int c_338 = lane_1;
                    int l_339 = c_338 >> 2;
                    unsigned int w4_340[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_340[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_340[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_339 + (c_338 & 3 ^ l_339 >> 1 & 3)))));
                    long long idx_341 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_338));
                    {
                        int4 _iv4 = make_int4(w4_340[0 + 0], w4_340[0 + 1], w4_340[0 + 2], w4_340[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_341) = _iv4;
                    }
                    int c_342 = lane_1 + 32;
                    int l_343 = c_342 >> 2;
                    unsigned int w4_344[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_344[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_344[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_343 + (c_342 & 3 ^ l_343 >> 1 & 3)))));
                    long long idx_345 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_342));
                    {
                        int4 _iv4 = make_int4(w4_344[0 + 0], w4_344[0 + 1], w4_344[0 + 2], w4_344[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_345) = _iv4;
                    }
                    int c_346 = lane_1 + 64;
                    int l_347 = c_346 >> 2;
                    unsigned int w4_348[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_348[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_348[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_347 + (c_346 & 3 ^ l_347 >> 1 & 3)))));
                    long long idx_349 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_346));
                    {
                        int4 _iv4 = make_int4(w4_348[0 + 0], w4_348[0 + 1], w4_348[0 + 2], w4_348[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_349) = _iv4;
                    }
                    int c_350 = lane_1 + 96;
                    int l_351 = c_350 >> 2;
                    unsigned int w4_352[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_352[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_352[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_352[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_352[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_351 + (c_350 & 3 ^ l_351 >> 1 & 3)))));
                    long long idx_353 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_350));
                    {
                        int4 _iv4 = make_int4(w4_352[0 + 0], w4_352[0 + 1], w4_352[0 + 2], w4_352[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_353) = _iv4;
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
                ctrl[base_1 + 3] = 3;
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
            for (int i = lane_0; i < 1024; i += 32) {
                pat[i] = sv;
                pat[1024 + i] = ss;
                pat[2048 + i] = sr;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 0, 352;" ::: "memory");
            unsigned int prev_rows = ctl[1];
            unsigned int prev_algo = ctl[2];
            unsigned int pbuf = (ctl[0] + 2) % 3;
            if (prev_algo != 0) {
                long long bid64_1 = (long long)bid;
                long long g64_1 = (long long)gridDim.x;
                long long pr64 = (long long)prev_rows;
                long long pb64 = (long long)pbuf;
                long long cap64_1 = (long long)cap;
                long long peer64_1 = (long long)(1 - rank);
                long long lane64 = (long long)lane_0;
                unsigned int sv4[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                long long base_2 = pb64 * (2 * cap64_1 * 3456) + peer64_1 * cap64_1 * 3456;
                long long u = pr64 * 3456 / 16;
                long long lo = u * bid64_1 / g64_1;
                long long hi = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo + lane64;
                int gi = (int)(g0 % 216);
                #pragma unroll 1
                for (long long g = g0; g < hi; g += 32) {
                    unsigned int w4_1[4];
                    w4_1[0] = ss4[0];
                    w4_1[1] = ss4[1];
                    w4_1[2] = ss4[2];
                    w4_1[3] = ss4[3];
                    if (gi < 192) {
                        w4_1[0] = sv4[0];
                        w4_1[1] = sv4[1];
                        w4_1[2] = sv4[2];
                        w4_1[3] = sv4[3];
                    }
                    long long widx = base_2 + 16 * g >> 2;
                    {
                        int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(q_wire) + widx) = _iv4;
                    }
                    int gn = gi + 32;
                    gi = gn;
                    if (gn >= 216) {
                        gi = gn - 216;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
