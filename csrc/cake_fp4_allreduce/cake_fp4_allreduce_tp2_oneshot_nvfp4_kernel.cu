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
            asm volatile("fence.proxy.async.global;" ::: "memory");
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
            unsigned int _phase_xb_full = 0;
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    int slot = k % 3;
                    int ob_off = ring_off + slot * 3472;
                    mbarrier_wait(xb_full_addr + (leader) * 8, _phase_xb_full);
                    unsigned int va_0 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_1 + 32 * h));
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
                    unsigned int va_1 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_1 + 32 * h + 64));
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
                    unsigned int va_2 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_1 + 32 * h + 128));
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
                    if (lane_1 + 32 * h < 24) {
                        unsigned int sa = wsm_addr + (unsigned int)(ob_off + 3072 + 16 * (lane_1 + 32 * h));
                        unsigned int s4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s4[0])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 3]))
                            : "r"(sa));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sa), "r"(s4[0] & 2139062143), "r"(s4[1] & 2139062143), "r"(s4[2] & 2139062143), "r"(s4[3] & 2139062143) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
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
                    if (h == 0 && lane_1 == 0 && n > k + 1) {
                        asm volatile("cp.async.bulk.wait_group.read 2;");
                        long long next_row = (long long)(r0 + t + 5 * (k + 1));
                        int next_slot = (k + 1) % 3;
                        mbarrier_arrive_expect_tx(xb_full_addr + (leader) * 8, 3456);
                        cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(ring_off + next_slot * 3472), xq + (next_row * 3456), 3456, xb_full_addr + (leader) * 8);
                    }
                }
                _phase_xb_full ^= 1;
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
                    uint32_t e4m3x2_to_f16x2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2) : "h"((uint16_t)(a_sc)));
                    float cvt_f32_f16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16) : "h"((uint16_t)(e4m3x2_to_f16x2)));
                    float cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1) : "h"((uint16_t)(e4m3x2_to_f16x2 >> 16)));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(cvt_f32_f16), "f"(a_rs));
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(cvt_f32_f16_1), "f"(a_rs));
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
                    float2 q_0 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 r_0 = fma_f32x2_rn_noftz(q_0, scale0, a_0);
                    acc[0] = r_0.x;
                    acc[1] = r_0.y;
                    float2 a_1 = make_float2(acc[2], acc[3]);
                    float2 q_1 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 r_1 = fma_f32x2_rn_noftz(q_1, scale0, a_1);
                    acc[2] = r_1.x;
                    acc[3] = r_1.y;
                    float2 a_2 = make_float2(acc[4], acc[5]);
                    float2 q_2 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 r_2 = fma_f32x2_rn_noftz(q_2, scale0, a_2);
                    acc[4] = r_2.x;
                    acc[5] = r_2.y;
                    float2 a_3 = make_float2(acc[6], acc[7]);
                    float2 q_3 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 r_3 = fma_f32x2_rn_noftz(q_3, scale0, a_3);
                    acc[6] = r_3.x;
                    acc[7] = r_3.y;
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
                    float2 a_4 = make_float2(acc[8], acc[9]);
                    float2 q_4 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 r_4 = fma_f32x2_rn_noftz(q_4, scale0, a_4);
                    acc[8] = r_4.x;
                    acc[9] = r_4.y;
                    float2 a_5 = make_float2(acc[10], acc[11]);
                    float2 q_5 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 r_5 = fma_f32x2_rn_noftz(q_5, scale0, a_5);
                    acc[10] = r_5.x;
                    acc[11] = r_5.y;
                    float2 a_6 = make_float2(acc[12], acc[13]);
                    float2 q_6 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 r_6 = fma_f32x2_rn_noftz(q_6, scale0, a_6);
                    acc[12] = r_6.x;
                    acc[13] = r_6.y;
                    float2 a_7 = make_float2(acc[14], acc[15]);
                    float2 q_7 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 r_7 = fma_f32x2_rn_noftz(q_7, scale0, a_7);
                    acc[14] = r_7.x;
                    acc[15] = r_7.y;
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
                    float2 a_8 = make_float2(acc[16], acc[17]);
                    float2 q_8 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 r_8 = fma_f32x2_rn_noftz(q_8, scale1, a_8);
                    acc[16] = r_8.x;
                    acc[17] = r_8.y;
                    float2 a_9 = make_float2(acc[18], acc[19]);
                    float2 q_9 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 r_9 = fma_f32x2_rn_noftz(q_9, scale1, a_9);
                    acc[18] = r_9.x;
                    acc[19] = r_9.y;
                    float2 a_10 = make_float2(acc[20], acc[21]);
                    float2 q_10 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 r_10 = fma_f32x2_rn_noftz(q_10, scale1, a_10);
                    acc[20] = r_10.x;
                    acc[21] = r_10.y;
                    float2 a_11 = make_float2(acc[22], acc[23]);
                    float2 q_11 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 r_11 = fma_f32x2_rn_noftz(q_11, scale1, a_11);
                    acc[22] = r_11.x;
                    acc[23] = r_11.y;
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
                    float2 a_12 = make_float2(acc[24], acc[25]);
                    float2 q_12 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 r_12 = fma_f32x2_rn_noftz(q_12, scale1, a_12);
                    acc[24] = r_12.x;
                    acc[25] = r_12.y;
                    float2 a_13 = make_float2(acc[26], acc[27]);
                    float2 q_13 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 r_13 = fma_f32x2_rn_noftz(q_13, scale1, a_13);
                    acc[26] = r_13.x;
                    acc[27] = r_13.y;
                    float2 a_14 = make_float2(acc[28], acc[29]);
                    float2 q_14 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 r_14 = fma_f32x2_rn_noftz(q_14, scale1, a_14);
                    acc[28] = r_14.x;
                    acc[29] = r_14.y;
                    float2 a_15 = make_float2(acc[30], acc[31]);
                    float2 q_15 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 r_15 = fma_f32x2_rn_noftz(q_15, scale1, a_15);
                    acc[30] = r_15.x;
                    acc[31] = r_15.y;
                    uint32_t e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_0) : "h"((uint16_t)(b_sc)));
                    float cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0)));
                    float cvt_f32_f16_1_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0 >> 16)));
                    float _mul_0_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_0) : "f"(cvt_f32_f16_0), "f"(b_rs));
                    float _mul_1_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_0) : "f"(cvt_f32_f16_1_0), "f"(b_rs));
                    float2 scale0_0 = make_float2(_mul_0_0, _mul_0_0);
                    float2 scale1_0 = make_float2(_mul_1_0, _mul_1_0);
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
                    float2 a_16 = make_float2(acc[0], acc[1]);
                    float2 q_16 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 r_16 = fma_f32x2_rn_noftz(q_16, scale0_0, a_16);
                    acc[0] = r_16.x;
                    acc[1] = r_16.y;
                    float2 a_17 = make_float2(acc[2], acc[3]);
                    float2 q_17 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 r_17 = fma_f32x2_rn_noftz(q_17, scale0_0, a_17);
                    acc[2] = r_17.x;
                    acc[3] = r_17.y;
                    float2 a_18 = make_float2(acc[4], acc[5]);
                    float2 q_18 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 r_18 = fma_f32x2_rn_noftz(q_18, scale0_0, a_18);
                    acc[4] = r_18.x;
                    acc[5] = r_18.y;
                    float2 a_19 = make_float2(acc[6], acc[7]);
                    float2 q_19 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 r_19 = fma_f32x2_rn_noftz(q_19, scale0_0, a_19);
                    acc[6] = r_19.x;
                    acc[7] = r_19.y;
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
                    float2 a_20 = make_float2(acc[8], acc[9]);
                    float2 q_20 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 r_20 = fma_f32x2_rn_noftz(q_20, scale0_0, a_20);
                    acc[8] = r_20.x;
                    acc[9] = r_20.y;
                    float2 a_21 = make_float2(acc[10], acc[11]);
                    float2 q_21 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 r_21 = fma_f32x2_rn_noftz(q_21, scale0_0, a_21);
                    acc[10] = r_21.x;
                    acc[11] = r_21.y;
                    float2 a_22 = make_float2(acc[12], acc[13]);
                    float2 q_22 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 r_22 = fma_f32x2_rn_noftz(q_22, scale0_0, a_22);
                    acc[12] = r_22.x;
                    acc[13] = r_22.y;
                    float2 a_23 = make_float2(acc[14], acc[15]);
                    float2 q_23 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 r_23 = fma_f32x2_rn_noftz(q_23, scale0_0, a_23);
                    acc[14] = r_23.x;
                    acc[15] = r_23.y;
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
                    float2 a_24 = make_float2(acc[16], acc[17]);
                    float2 q_24 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 r_24 = fma_f32x2_rn_noftz(q_24, scale1_0, a_24);
                    acc[16] = r_24.x;
                    acc[17] = r_24.y;
                    float2 a_25 = make_float2(acc[18], acc[19]);
                    float2 q_25 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 r_25 = fma_f32x2_rn_noftz(q_25, scale1_0, a_25);
                    acc[18] = r_25.x;
                    acc[19] = r_25.y;
                    float2 a_26 = make_float2(acc[20], acc[21]);
                    float2 q_26 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 r_26 = fma_f32x2_rn_noftz(q_26, scale1_0, a_26);
                    acc[20] = r_26.x;
                    acc[21] = r_26.y;
                    float2 a_27 = make_float2(acc[22], acc[23]);
                    float2 q_27 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 r_27 = fma_f32x2_rn_noftz(q_27, scale1_0, a_27);
                    acc[22] = r_27.x;
                    acc[23] = r_27.y;
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
                    float2 a_28 = make_float2(acc[24], acc[25]);
                    float2 q_28 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 r_28 = fma_f32x2_rn_noftz(q_28, scale1_0, a_28);
                    acc[24] = r_28.x;
                    acc[25] = r_28.y;
                    float2 a_29 = make_float2(acc[26], acc[27]);
                    float2 q_29 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 r_29 = fma_f32x2_rn_noftz(q_29, scale1_0, a_29);
                    acc[26] = r_29.x;
                    acc[27] = r_29.y;
                    float2 a_30 = make_float2(acc[28], acc[29]);
                    float2 q_30 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 r_30 = fma_f32x2_rn_noftz(q_30, scale1_0, a_30);
                    acc[28] = r_30.x;
                    acc[29] = r_30.y;
                    float2 a_31 = make_float2(acc[30], acc[31]);
                    float2 q_31 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 r_31 = fma_f32x2_rn_noftz(q_31, scale1_0, a_31);
                    acc[30] = r_31.x;
                    acc[31] = r_31.y;
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
                    int c_1_0 = lane_1;
                    int l_1_0 = c_1_0 >> 2;
                    unsigned int w4_1_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1_0 + (c_1_0 & 3 ^ l_1_0 >> 1 & 3)))));
                    long long idx_1_0 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_1_0));
                    {
                        int4 _iv4 = make_int4(w4_1_0[0 + 0], w4_1_0[0 + 1], w4_1_0[0 + 2], w4_1_0[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1_0) = _iv4;
                    }
                    int c_1_1 = lane_1 + 32;
                    int l_1_1 = c_1_1 >> 2;
                    unsigned int w4_1_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1_1 + (c_1_1 & 3 ^ l_1_1 >> 1 & 3)))));
                    long long idx_1_1 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_1_1));
                    {
                        int4 _iv4 = make_int4(w4_1_1[0 + 0], w4_1_1[0 + 1], w4_1_1[0 + 2], w4_1_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1_1) = _iv4;
                    }
                    int c_1_2 = lane_1 + 64;
                    int l_1_2 = c_1_2 >> 2;
                    unsigned int w4_1_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1_2 + (c_1_2 & 3 ^ l_1_2 >> 1 & 3)))));
                    long long idx_1_2 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_1_2));
                    {
                        int4 _iv4 = make_int4(w4_1_2[0 + 0], w4_1_2[0 + 1], w4_1_2[0 + 2], w4_1_2[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1_2) = _iv4;
                    }
                    int c_1_3 = lane_1 + 96;
                    int l_1_3 = c_1_3 >> 2;
                    unsigned int w4_1_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1_3 + (c_1_3 & 3 ^ l_1_3 >> 1 & 3)))));
                    long long idx_1_3 = drow * 3072 + (long long)(4 * (128 * (3 * h) + c_1_3));
                    {
                        int4 _iv4 = make_int4(w4_1_3[0 + 0], w4_1_3[0 + 1], w4_1_3[0 + 2], w4_1_3[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1_3) = _iv4;
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
                    uint32_t e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_1) : "h"((uint16_t)(a_sc_110)));
                    float cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_2) : "h"((uint16_t)(e4m3x2_to_f16x2_1)));
                    float cvt_f32_f16_1_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_1) : "h"((uint16_t)(e4m3x2_to_f16x2_1 >> 16)));
                    float _mul_0_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_1) : "f"(cvt_f32_f16_2), "f"(a_rs_112));
                    float _mul_1_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_1) : "f"(cvt_f32_f16_1_1), "f"(a_rs_112));
                    float2 scale0_1 = make_float2(_mul_0_1, _mul_0_1);
                    float2 scale1_1 = make_float2(_mul_1_1, _mul_1_1);
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
                    float2 a_32 = make_float2(acc_114[0], acc_114[1]);
                    float2 q_32 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 r_32 = fma_f32x2_rn_noftz(q_32, scale0_1, a_32);
                    acc_114[0] = r_32.x;
                    acc_114[1] = r_32.y;
                    float2 a_33 = make_float2(acc_114[2], acc_114[3]);
                    float2 q_33 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 r_33 = fma_f32x2_rn_noftz(q_33, scale0_1, a_33);
                    acc_114[2] = r_33.x;
                    acc_114[3] = r_33.y;
                    float2 a_34 = make_float2(acc_114[4], acc_114[5]);
                    float2 q_34 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 r_34 = fma_f32x2_rn_noftz(q_34, scale0_1, a_34);
                    acc_114[4] = r_34.x;
                    acc_114[5] = r_34.y;
                    float2 a_35 = make_float2(acc_114[6], acc_114[7]);
                    float2 q_35 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 r_35 = fma_f32x2_rn_noftz(q_35, scale0_1, a_35);
                    acc_114[6] = r_35.x;
                    acc_114[7] = r_35.y;
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
                    float2 a_36 = make_float2(acc_114[8], acc_114[9]);
                    float2 q_36 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 r_36 = fma_f32x2_rn_noftz(q_36, scale0_1, a_36);
                    acc_114[8] = r_36.x;
                    acc_114[9] = r_36.y;
                    float2 a_37 = make_float2(acc_114[10], acc_114[11]);
                    float2 q_37 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 r_37 = fma_f32x2_rn_noftz(q_37, scale0_1, a_37);
                    acc_114[10] = r_37.x;
                    acc_114[11] = r_37.y;
                    float2 a_38 = make_float2(acc_114[12], acc_114[13]);
                    float2 q_38 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 r_38 = fma_f32x2_rn_noftz(q_38, scale0_1, a_38);
                    acc_114[12] = r_38.x;
                    acc_114[13] = r_38.y;
                    float2 a_39 = make_float2(acc_114[14], acc_114[15]);
                    float2 q_39 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 r_39 = fma_f32x2_rn_noftz(q_39, scale0_1, a_39);
                    acc_114[14] = r_39.x;
                    acc_114[15] = r_39.y;
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
                    float2 a_40 = make_float2(acc_114[16], acc_114[17]);
                    float2 q_40 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 r_40 = fma_f32x2_rn_noftz(q_40, scale1_1, a_40);
                    acc_114[16] = r_40.x;
                    acc_114[17] = r_40.y;
                    float2 a_41 = make_float2(acc_114[18], acc_114[19]);
                    float2 q_41 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 r_41 = fma_f32x2_rn_noftz(q_41, scale1_1, a_41);
                    acc_114[18] = r_41.x;
                    acc_114[19] = r_41.y;
                    float2 a_42 = make_float2(acc_114[20], acc_114[21]);
                    float2 q_42 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 r_42 = fma_f32x2_rn_noftz(q_42, scale1_1, a_42);
                    acc_114[20] = r_42.x;
                    acc_114[21] = r_42.y;
                    float2 a_43 = make_float2(acc_114[22], acc_114[23]);
                    float2 q_43 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 r_43 = fma_f32x2_rn_noftz(q_43, scale1_1, a_43);
                    acc_114[22] = r_43.x;
                    acc_114[23] = r_43.y;
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
                    float2 a_44 = make_float2(acc_114[24], acc_114[25]);
                    float2 q_44 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 r_44 = fma_f32x2_rn_noftz(q_44, scale1_1, a_44);
                    acc_114[24] = r_44.x;
                    acc_114[25] = r_44.y;
                    float2 a_45 = make_float2(acc_114[26], acc_114[27]);
                    float2 q_45 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 r_45 = fma_f32x2_rn_noftz(q_45, scale1_1, a_45);
                    acc_114[26] = r_45.x;
                    acc_114[27] = r_45.y;
                    float2 a_46 = make_float2(acc_114[28], acc_114[29]);
                    float2 q_46 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 r_46 = fma_f32x2_rn_noftz(q_46, scale1_1, a_46);
                    acc_114[28] = r_46.x;
                    acc_114[29] = r_46.y;
                    float2 a_47 = make_float2(acc_114[30], acc_114[31]);
                    float2 q_47 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 r_47 = fma_f32x2_rn_noftz(q_47, scale1_1, a_47);
                    acc_114[30] = r_47.x;
                    acc_114[31] = r_47.y;
                    uint32_t e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_2) : "h"((uint16_t)(b_sc_111)));
                    float cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_3) : "h"((uint16_t)(e4m3x2_to_f16x2_2)));
                    float cvt_f32_f16_1_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_2) : "h"((uint16_t)(e4m3x2_to_f16x2_2 >> 16)));
                    float _mul_0_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_2) : "f"(cvt_f32_f16_3), "f"(b_rs_113));
                    float _mul_1_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_2) : "f"(cvt_f32_f16_1_2), "f"(b_rs_113));
                    float2 scale0_2 = make_float2(_mul_0_2, _mul_0_2);
                    float2 scale1_2 = make_float2(_mul_1_2, _mul_1_2);
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
                    float2 a_48 = make_float2(acc_114[0], acc_114[1]);
                    float2 q_48 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 r_48 = fma_f32x2_rn_noftz(q_48, scale0_2, a_48);
                    acc_114[0] = r_48.x;
                    acc_114[1] = r_48.y;
                    float2 a_49 = make_float2(acc_114[2], acc_114[3]);
                    float2 q_49 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 r_49 = fma_f32x2_rn_noftz(q_49, scale0_2, a_49);
                    acc_114[2] = r_49.x;
                    acc_114[3] = r_49.y;
                    float2 a_50 = make_float2(acc_114[4], acc_114[5]);
                    float2 q_50 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 r_50 = fma_f32x2_rn_noftz(q_50, scale0_2, a_50);
                    acc_114[4] = r_50.x;
                    acc_114[5] = r_50.y;
                    float2 a_51 = make_float2(acc_114[6], acc_114[7]);
                    float2 q_51 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 r_51 = fma_f32x2_rn_noftz(q_51, scale0_2, a_51);
                    acc_114[6] = r_51.x;
                    acc_114[7] = r_51.y;
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
                    float2 a_52 = make_float2(acc_114[8], acc_114[9]);
                    float2 q_52 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 r_52 = fma_f32x2_rn_noftz(q_52, scale0_2, a_52);
                    acc_114[8] = r_52.x;
                    acc_114[9] = r_52.y;
                    float2 a_53 = make_float2(acc_114[10], acc_114[11]);
                    float2 q_53 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 r_53 = fma_f32x2_rn_noftz(q_53, scale0_2, a_53);
                    acc_114[10] = r_53.x;
                    acc_114[11] = r_53.y;
                    float2 a_54 = make_float2(acc_114[12], acc_114[13]);
                    float2 q_54 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 r_54 = fma_f32x2_rn_noftz(q_54, scale0_2, a_54);
                    acc_114[12] = r_54.x;
                    acc_114[13] = r_54.y;
                    float2 a_55 = make_float2(acc_114[14], acc_114[15]);
                    float2 q_55 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 r_55 = fma_f32x2_rn_noftz(q_55, scale0_2, a_55);
                    acc_114[14] = r_55.x;
                    acc_114[15] = r_55.y;
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
                    float2 a_56 = make_float2(acc_114[16], acc_114[17]);
                    float2 q_56 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 r_56 = fma_f32x2_rn_noftz(q_56, scale1_2, a_56);
                    acc_114[16] = r_56.x;
                    acc_114[17] = r_56.y;
                    float2 a_57 = make_float2(acc_114[18], acc_114[19]);
                    float2 q_57 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 r_57 = fma_f32x2_rn_noftz(q_57, scale1_2, a_57);
                    acc_114[18] = r_57.x;
                    acc_114[19] = r_57.y;
                    float2 a_58 = make_float2(acc_114[20], acc_114[21]);
                    float2 q_58 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 r_58 = fma_f32x2_rn_noftz(q_58, scale1_2, a_58);
                    acc_114[20] = r_58.x;
                    acc_114[21] = r_58.y;
                    float2 a_59 = make_float2(acc_114[22], acc_114[23]);
                    float2 q_59 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 r_59 = fma_f32x2_rn_noftz(q_59, scale1_2, a_59);
                    acc_114[22] = r_59.x;
                    acc_114[23] = r_59.y;
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
                    float2 a_60 = make_float2(acc_114[24], acc_114[25]);
                    float2 q_60 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 r_60 = fma_f32x2_rn_noftz(q_60, scale1_2, a_60);
                    acc_114[24] = r_60.x;
                    acc_114[25] = r_60.y;
                    float2 a_61 = make_float2(acc_114[26], acc_114[27]);
                    float2 q_61 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 r_61 = fma_f32x2_rn_noftz(q_61, scale1_2, a_61);
                    acc_114[26] = r_61.x;
                    acc_114[27] = r_61.y;
                    float2 a_62 = make_float2(acc_114[28], acc_114[29]);
                    float2 q_62 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 r_62 = fma_f32x2_rn_noftz(q_62, scale1_2, a_62);
                    acc_114[28] = r_62.x;
                    acc_114[29] = r_62.y;
                    float2 a_63 = make_float2(acc_114[30], acc_114[31]);
                    float2 q_63 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 r_63 = fma_f32x2_rn_noftz(q_63, scale1_2, a_63);
                    acc_114[30] = r_63.x;
                    acc_114[31] = r_63.y;
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
                    int c_0 = lane_1;
                    int l_0 = c_0 >> 2;
                    unsigned int w4_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_0 + (c_0 & 3 ^ l_0 >> 1 & 3)))));
                    long long idx_0 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_0));
                    {
                        int4 _iv4 = make_int4(w4_3[0 + 0], w4_3[0 + 1], w4_3[0 + 2], w4_3[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_0) = _iv4;
                    }
                    int c_1 = lane_1 + 32;
                    int l_1 = c_1 >> 2;
                    unsigned int w4_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1 + (c_1 & 3 ^ l_1 >> 1 & 3)))));
                    long long idx_1 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_1));
                    {
                        int4 _iv4 = make_int4(w4_4[0 + 0], w4_4[0 + 1], w4_4[0 + 2], w4_4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1) = _iv4;
                    }
                    int c_2 = lane_1 + 64;
                    int l_2 = c_2 >> 2;
                    unsigned int w4_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_2 + (c_2 & 3 ^ l_2 >> 1 & 3)))));
                    long long idx_2 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_2));
                    {
                        int4 _iv4 = make_int4(w4_5[0 + 0], w4_5[0 + 1], w4_5[0 + 2], w4_5[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_2) = _iv4;
                    }
                    int c_3 = lane_1 + 96;
                    int l_3 = c_3 >> 2;
                    unsigned int w4_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_6[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_3 + (c_3 & 3 ^ l_3 >> 1 & 3)))));
                    long long idx_3 = drow * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_3));
                    {
                        int4 _iv4 = make_int4(w4_6[0 + 0], w4_6[0 + 1], w4_6[0 + 2], w4_6[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_3) = _iv4;
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
                    uint32_t e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_3) : "h"((uint16_t)(a_sc_233)));
                    float cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_4) : "h"((uint16_t)(e4m3x2_to_f16x2_3)));
                    float cvt_f32_f16_1_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_3) : "h"((uint16_t)(e4m3x2_to_f16x2_3 >> 16)));
                    float _mul_0_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_3) : "f"(cvt_f32_f16_4), "f"(a_rs_235));
                    float _mul_1_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_3) : "f"(cvt_f32_f16_1_3), "f"(a_rs_235));
                    float2 scale0_3 = make_float2(_mul_0_3, _mul_0_3);
                    float2 scale1_3 = make_float2(_mul_1_3, _mul_1_3);
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
                    float2 a_64 = make_float2(acc_237[0], acc_237[1]);
                    float2 q_64 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 r_64 = fma_f32x2_rn_noftz(q_64, scale0_3, a_64);
                    acc_237[0] = r_64.x;
                    acc_237[1] = r_64.y;
                    float2 a_65 = make_float2(acc_237[2], acc_237[3]);
                    float2 q_65 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 r_65 = fma_f32x2_rn_noftz(q_65, scale0_3, a_65);
                    acc_237[2] = r_65.x;
                    acc_237[3] = r_65.y;
                    float2 a_66 = make_float2(acc_237[4], acc_237[5]);
                    float2 q_66 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 r_66 = fma_f32x2_rn_noftz(q_66, scale0_3, a_66);
                    acc_237[4] = r_66.x;
                    acc_237[5] = r_66.y;
                    float2 a_67 = make_float2(acc_237[6], acc_237[7]);
                    float2 q_67 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 r_67 = fma_f32x2_rn_noftz(q_67, scale0_3, a_67);
                    acc_237[6] = r_67.x;
                    acc_237[7] = r_67.y;
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
                    float2 a_68 = make_float2(acc_237[8], acc_237[9]);
                    float2 q_68 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 r_68 = fma_f32x2_rn_noftz(q_68, scale0_3, a_68);
                    acc_237[8] = r_68.x;
                    acc_237[9] = r_68.y;
                    float2 a_69 = make_float2(acc_237[10], acc_237[11]);
                    float2 q_69 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 r_69 = fma_f32x2_rn_noftz(q_69, scale0_3, a_69);
                    acc_237[10] = r_69.x;
                    acc_237[11] = r_69.y;
                    float2 a_70 = make_float2(acc_237[12], acc_237[13]);
                    float2 q_70 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 r_70 = fma_f32x2_rn_noftz(q_70, scale0_3, a_70);
                    acc_237[12] = r_70.x;
                    acc_237[13] = r_70.y;
                    float2 a_71 = make_float2(acc_237[14], acc_237[15]);
                    float2 q_71 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 r_71 = fma_f32x2_rn_noftz(q_71, scale0_3, a_71);
                    acc_237[14] = r_71.x;
                    acc_237[15] = r_71.y;
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
                    float2 a_72 = make_float2(acc_237[16], acc_237[17]);
                    float2 q_72 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 r_72 = fma_f32x2_rn_noftz(q_72, scale1_3, a_72);
                    acc_237[16] = r_72.x;
                    acc_237[17] = r_72.y;
                    float2 a_73 = make_float2(acc_237[18], acc_237[19]);
                    float2 q_73 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 r_73 = fma_f32x2_rn_noftz(q_73, scale1_3, a_73);
                    acc_237[18] = r_73.x;
                    acc_237[19] = r_73.y;
                    float2 a_74 = make_float2(acc_237[20], acc_237[21]);
                    float2 q_74 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 r_74 = fma_f32x2_rn_noftz(q_74, scale1_3, a_74);
                    acc_237[20] = r_74.x;
                    acc_237[21] = r_74.y;
                    float2 a_75 = make_float2(acc_237[22], acc_237[23]);
                    float2 q_75 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 r_75 = fma_f32x2_rn_noftz(q_75, scale1_3, a_75);
                    acc_237[22] = r_75.x;
                    acc_237[23] = r_75.y;
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
                    float2 a_76 = make_float2(acc_237[24], acc_237[25]);
                    float2 q_76 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 r_76 = fma_f32x2_rn_noftz(q_76, scale1_3, a_76);
                    acc_237[24] = r_76.x;
                    acc_237[25] = r_76.y;
                    float2 a_77 = make_float2(acc_237[26], acc_237[27]);
                    float2 q_77 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 r_77 = fma_f32x2_rn_noftz(q_77, scale1_3, a_77);
                    acc_237[26] = r_77.x;
                    acc_237[27] = r_77.y;
                    float2 a_78 = make_float2(acc_237[28], acc_237[29]);
                    float2 q_78 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 r_78 = fma_f32x2_rn_noftz(q_78, scale1_3, a_78);
                    acc_237[28] = r_78.x;
                    acc_237[29] = r_78.y;
                    float2 a_79 = make_float2(acc_237[30], acc_237[31]);
                    float2 q_79 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 r_79 = fma_f32x2_rn_noftz(q_79, scale1_3, a_79);
                    acc_237[30] = r_79.x;
                    acc_237[31] = r_79.y;
                    uint32_t e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_4) : "h"((uint16_t)(b_sc_234)));
                    float cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_5) : "h"((uint16_t)(e4m3x2_to_f16x2_4)));
                    float cvt_f32_f16_1_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_4) : "h"((uint16_t)(e4m3x2_to_f16x2_4 >> 16)));
                    float _mul_0_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0_4) : "f"(cvt_f32_f16_5), "f"(b_rs_236));
                    float _mul_1_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1_4) : "f"(cvt_f32_f16_1_4), "f"(b_rs_236));
                    float2 scale0_4 = make_float2(_mul_0_4, _mul_0_4);
                    float2 scale1_4 = make_float2(_mul_1_4, _mul_1_4);
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
                    float2 a_80 = make_float2(acc_237[0], acc_237[1]);
                    float2 q_80 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 r_80 = fma_f32x2_rn_noftz(q_80, scale0_4, a_80);
                    acc_237[0] = r_80.x;
                    acc_237[1] = r_80.y;
                    float2 a_81 = make_float2(acc_237[2], acc_237[3]);
                    float2 q_81 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 r_81 = fma_f32x2_rn_noftz(q_81, scale0_4, a_81);
                    acc_237[2] = r_81.x;
                    acc_237[3] = r_81.y;
                    float2 a_82 = make_float2(acc_237[4], acc_237[5]);
                    float2 q_82 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 r_82 = fma_f32x2_rn_noftz(q_82, scale0_4, a_82);
                    acc_237[4] = r_82.x;
                    acc_237[5] = r_82.y;
                    float2 a_83 = make_float2(acc_237[6], acc_237[7]);
                    float2 q_83 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 r_83 = fma_f32x2_rn_noftz(q_83, scale0_4, a_83);
                    acc_237[6] = r_83.x;
                    acc_237[7] = r_83.y;
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
                    float2 a_84 = make_float2(acc_237[8], acc_237[9]);
                    float2 q_84 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 r_84 = fma_f32x2_rn_noftz(q_84, scale0_4, a_84);
                    acc_237[8] = r_84.x;
                    acc_237[9] = r_84.y;
                    float2 a_85 = make_float2(acc_237[10], acc_237[11]);
                    float2 q_85 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 r_85 = fma_f32x2_rn_noftz(q_85, scale0_4, a_85);
                    acc_237[10] = r_85.x;
                    acc_237[11] = r_85.y;
                    float2 a_86 = make_float2(acc_237[12], acc_237[13]);
                    float2 q_86 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 r_86 = fma_f32x2_rn_noftz(q_86, scale0_4, a_86);
                    acc_237[12] = r_86.x;
                    acc_237[13] = r_86.y;
                    float2 a_87 = make_float2(acc_237[14], acc_237[15]);
                    float2 q_87 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 r_87 = fma_f32x2_rn_noftz(q_87, scale0_4, a_87);
                    acc_237[14] = r_87.x;
                    acc_237[15] = r_87.y;
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
                    float2 a_88 = make_float2(acc_237[16], acc_237[17]);
                    float2 q_88 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 r_88 = fma_f32x2_rn_noftz(q_88, scale1_4, a_88);
                    acc_237[16] = r_88.x;
                    acc_237[17] = r_88.y;
                    float2 a_89 = make_float2(acc_237[18], acc_237[19]);
                    float2 q_89 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 r_89 = fma_f32x2_rn_noftz(q_89, scale1_4, a_89);
                    acc_237[18] = r_89.x;
                    acc_237[19] = r_89.y;
                    float2 a_90 = make_float2(acc_237[20], acc_237[21]);
                    float2 q_90 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 r_90 = fma_f32x2_rn_noftz(q_90, scale1_4, a_90);
                    acc_237[20] = r_90.x;
                    acc_237[21] = r_90.y;
                    float2 a_91 = make_float2(acc_237[22], acc_237[23]);
                    float2 q_91 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 r_91 = fma_f32x2_rn_noftz(q_91, scale1_4, a_91);
                    acc_237[22] = r_91.x;
                    acc_237[23] = r_91.y;
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
                    float2 a_92 = make_float2(acc_237[24], acc_237[25]);
                    float2 q_92 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 r_92 = fma_f32x2_rn_noftz(q_92, scale1_4, a_92);
                    acc_237[24] = r_92.x;
                    acc_237[25] = r_92.y;
                    float2 a_93 = make_float2(acc_237[26], acc_237[27]);
                    float2 q_93 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 r_93 = fma_f32x2_rn_noftz(q_93, scale1_4, a_93);
                    acc_237[26] = r_93.x;
                    acc_237[27] = r_93.y;
                    float2 a_94 = make_float2(acc_237[28], acc_237[29]);
                    float2 q_94 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 r_94 = fma_f32x2_rn_noftz(q_94, scale1_4, a_94);
                    acc_237[28] = r_94.x;
                    acc_237[29] = r_94.y;
                    float2 a_95 = make_float2(acc_237[30], acc_237[31]);
                    float2 q_95 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 r_95 = fma_f32x2_rn_noftz(q_95, scale1_4, a_95);
                    acc_237[30] = r_95.x;
                    acc_237[31] = r_95.y;
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
                    int c_4 = lane_1;
                    int l_4 = c_4 >> 2;
                    unsigned int w4_7[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_7[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_4 + (c_4 & 3 ^ l_4 >> 1 & 3)))));
                    long long idx_4 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_4));
                    {
                        int4 _iv4 = make_int4(w4_7[0 + 0], w4_7[0 + 1], w4_7[0 + 2], w4_7[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_4) = _iv4;
                    }
                    int c_5 = lane_1 + 32;
                    int l_5 = c_5 >> 2;
                    unsigned int w4_8[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_5 + (c_5 & 3 ^ l_5 >> 1 & 3)))));
                    long long idx_5 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_5));
                    {
                        int4 _iv4 = make_int4(w4_8[0 + 0], w4_8[0 + 1], w4_8[0 + 2], w4_8[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_5) = _iv4;
                    }
                    int c_6 = lane_1 + 64;
                    int l_6 = c_6 >> 2;
                    unsigned int w4_9[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_9[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_6 + (c_6 & 3 ^ l_6 >> 1 & 3)))));
                    long long idx_6 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_6));
                    {
                        int4 _iv4 = make_int4(w4_9[0 + 0], w4_9[0 + 1], w4_9[0 + 2], w4_9[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_6) = _iv4;
                    }
                    int c_7 = lane_1 + 96;
                    int l_7 = c_7 >> 2;
                    unsigned int w4_10[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_10[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_7 + (c_7 & 3 ^ l_7 >> 1 & 3)))));
                    long long idx_7 = drow * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_7));
                    {
                        int4 _iv4 = make_int4(w4_10[0 + 0], w4_10[0 + 1], w4_10[0 + 2], w4_10[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_7) = _iv4;
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
            for (int i = lane_0; i < 1024; i += 32) {
                pat[i] = sv;
                pat[1024 + i] = ss;
                pat[2048 + i] = sr;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
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
                    unsigned int w4[4];
                    w4[0] = ss4[0];
                    w4[1] = ss4[1];
                    w4[2] = ss4[2];
                    w4[3] = ss4[3];
                    if (gi < 192) {
                        w4[0] = sv4[0];
                        w4[1] = sv4[1];
                        w4[2] = sv4[2];
                        w4[3] = sv4[3];
                    }
                    long long widx = base_2 + 16 * g >> 2;
                    {
                        int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
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
