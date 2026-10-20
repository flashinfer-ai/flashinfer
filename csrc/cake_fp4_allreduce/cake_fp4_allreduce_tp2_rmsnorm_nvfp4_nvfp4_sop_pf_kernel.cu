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
#define SMEM_WROW_OFF 1024
#define SMEM_WROW_STAGE_BYTES 12288
#define SMEM_WROW_STRIDE 12288
#define SMEM_CTL_OFF 13312
#define SMEM_CTL_STAGE_BYTES 16
#define SMEM_CTL_STRIDE 16
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 198016
#define SMEM_WSM_STRIDE 198016
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 198016
#define SMEM_WSM32_STRIDE 198016
#define SMEM_TOTAL 211456
#define THREADS 288
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

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
cake_fp4_allreduce_tp2_rmsnorm_nvfp4_nvfp4_sop_pf(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ res_in, __nv_bfloat16* __restrict__ w, uint8_t* __restrict__ outq, uint8_t* __restrict__ out_sf, float* __restrict__ gs_out, __nv_bfloat16* __restrict__ res_out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap, float eps, float wbias)
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
    #define rb_full_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* wrow = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int wrow_addr = smem + 1024;
    unsigned int* ctl = reinterpret_cast<unsigned int*>(smem_raw + 13312);
    const int ctl_addr = smem + 13312;
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + 13376);
    const int wsm_addr = smem + 13376;
    float* wsm32 = reinterpret_cast<float*>(smem_raw + 13376);
    const int wsm32_addr = smem + 13376;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // rb_full: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: workers ----
    if (warp <= 7) {
        { // workers_main
            int tid_0 = (int)tid;
            #pragma unroll 1
            for (int i = tid_0; i < 768; i += 256) {
                unsigned int _vec_load_0[4];
                {
                    uint32_t _uv4_0_0;
                    uint32_t _uv4_0_1;
                    uint32_t _uv4_0_2;
                    uint32_t _uv4_0_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(w) + (16 * i)) + (0))) : "memory");
                    _vec_load_0[0 + 0] = _uv4_0_0;
                    _vec_load_0[0 + 1] = _uv4_0_1;
                    _vec_load_0[0 + 2] = _uv4_0_2;
                    _vec_load_0[0 + 3] = _uv4_0_3;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow_addr + (unsigned int)(16 * i)), "r"(_vec_load_0[0]), "r"(_vec_load_0[1]), "r"(_vec_load_0[2]), "r"(_vec_load_0[3]) : "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            long long bid64 = (long long)bid;
            long long g64 = (long long)gridDim.x;
            long long rows64 = (long long)rows;
            int r0 = (int)(bid64 * rows64 / g64);
            int r1 = (int)((bid64 + 1) * rows64 / g64);
            int nrows = r1 - r0;
            int warp_1 = (int)warp;
            int lane_2 = (int)lane;
            int n = 0;
            if (nrows > warp_1) {
                n = (nrows - warp_1 + 8 - 1) / 8;
            }
            if (n > 0 && lane_2 == 0) {
                long long first_row = (long long)(r0 + warp_1);
                mbarrier_arrive_expect_tx(rb_full_addr + (warp_1) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 24752 + 12464), res_in + (first_row * 6144), 12288, rb_full_addr + (warp_1) * 8);
                mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 24752), xq + (first_row * 3456), 3456, xb_full_addr + (warp_1) * 8);
            }
            if (tid == 0) {
                ctl[0] = ctrl[0];
                ctl[1] = ctrl[2];
                ctl[2] = ctrl[3];
            }
            asm volatile("barrier.sync 0, 288;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int f = lane_2 >> 1 & 3;
            int xb_off = warp_1 * 24752;
            int own_off = xb_off + 12288;
            int stg_off = xb_off + 10416;
            int rb_off = stg_off + 2048;
            int peer = 1 - rank;
            long long cap64 = (long long)cap;
            long long rank64 = (long long)rank;
            long long peer64 = (long long)peer;
            long long buf64 = (long long)buf;
            long long v_base = buf64 * (2 * cap64 * 3456);
            long long s_base = v_base + 2 * cap64 * 3072;
            long long m_base = buf64 * (2 * cap64 * 16);
            uint8_t* peer_wire = reinterpret_cast<uint8_t*>(peer_ptrs[peer]);
            uint8_t* peer_rs = reinterpret_cast<uint8_t*>(peer_ptrs[2 + peer]);
            float gso = gs_out[0];
            float gs_me = gs[rank];
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    long long row64 = (long long)(r0 + warp_1 + 8 * k);
                    int ob_off = xb_off + k % 3 * 3472;
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * lane_2)))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * (lane_2 + 32))))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * (lane_2 + 64))))));
                    float rs = 1.0f;
                    mbarrier_wait(xb_full_addr + (warp_1) * 8, k & 1);
                    rs = gs_me;
                    if (n > k + 1 && lane_2 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        long long next_row = (long long)(r0 + warp_1 + 8 * (k + 1));
                        mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 3456);
                        cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(xb_off + (k + 1) % 3 * 3472), xq + (next_row * 3456), 3456, xb_full_addr + (warp_1) * 8);
                    }
                    if (lane_2 < 4) {
                        {
                            uint32_t _addr_1 = static_cast<uint32_t>(wsm_addr + (unsigned int)(ob_off + 3072 + 384 + 4 * lane_2));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(rs) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        {
                            void* _cpbulk_dst_2 = reinterpret_cast<void*>(peer_wire + (v_base + (rank64 * cap64 + row64) * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_2), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_3 = reinterpret_cast<void*>(peer_wire + (s_base + (rank64 * cap64 + row64) * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_3), "r"(wsm_addr + (unsigned int)(ob_off + 3072)), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_4 = reinterpret_cast<void*>(peer_rs + (m_base + (rank64 * cap64 + row64) * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_4), "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 384)), "r"((uint32_t)(16))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                int kd = k - 1;
                if (kd >= 0 && kd < n) {
                    long long drow = (long long)(r0 + warp_1 + 8 * kd);
                    int dob_off = xb_off + kd % 3 * 3472;
                    float own_rs = wsm32[dob_off + 3072 + 384 >> 2];
                    bool first = rank == 0;
                    long long src64 = peer64;
                    long long v_row = v_base + (src64 * cap64 + drow) * 3072;
                    long long s_row = s_base + (src64 * cap64 + drow) * 384;
                    long long m_row = m_base + (src64 * cap64 + drow) * 16;
                    uint32_t _sysv_poll32_group_0[1];
                    do {
                        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                    } while (
                        (_sysv_poll32_group_0[0] == 0x80000000u)
                    );
                    float rs_peer = __uint_as_float(_sysv_poll32_group_0[0]);
                    float rs_a = ((first) ? own_rs : rs_peer);
                    float rs_b = ((first) ? rs_peer : own_rs);
                    float ssx = 0.0f;
                    float ssy = 0.0f;
                    #pragma unroll 1
                    for (int j0 = 0; j0 < 6; j0 += 2) {
                        int pp0 = lane_2 + 32 * j0;
                        int pp1 = pp0 + 32;
                        uint32_t _fp4_wire_values_0[8];
                        uint32_t _fp4_wire_scales_0[2];
                        uint16_t _wire_h16_5_0;
                        uint16_t _wire_h16_5_1;
                        uint32_t _wire_vpend_5 = 3u;
                        uint32_t _wire_spend_5 = 3u;
                        do {
                            if (_wire_vpend_5 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0)))) : "memory");
                            }
                            if (_wire_vpend_5 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1)))) : "memory");
                            }
                            if (_wire_spend_5 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_5_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0)))) : "memory");
                                _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_5_0;
                            }
                            if (_wire_spend_5 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_5_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1)))) : "memory");
                                _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_5_1;
                            }
                            _wire_vpend_5 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_5 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
                        } while ((_wire_vpend_5 | _wire_spend_5) != 0u);
                        if (j0 == 0) {
                            mbarrier_wait(rb_full_addr + (warp_1) * 8, kd & 1);
                        }
                        unsigned int own[4];
                        unsigned int own_sc[1];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp0)));
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp0)));
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
                        float acc[32] = {0};
                        bool fhi = (f & 2) != 0;
                        bool fodd = (f & 1) != 0;
                        bool fhi_0 = (f & 2) != 0;
                        unsigned int s[4];
                        s[0] = a[0];
                        s[1] = a[1];
                        s[2] = a[2];
                        s[3] = a[3];
                        if (fodd) {
                            s[0] = a[1];
                            s[1] = a[0];
                            s[2] = a[3];
                            s[3] = a[2];
                        }
                        unsigned int r[4];
                        r[0] = s[0];
                        r[1] = s[1];
                        r[2] = s[2];
                        r[3] = s[3];
                        if (fhi_0) {
                            r[0] = s[2];
                            r[1] = s[3];
                            r[2] = s[0];
                            r[3] = s[1];
                        }
                        bool fodd_1 = (f & 1) != 0;
                        bool fhi_2 = (f & 2) != 0;
                        unsigned int s_3[4];
                        s_3[0] = b[0];
                        s_3[1] = b[1];
                        s_3[2] = b[2];
                        s_3[3] = b[3];
                        if (fodd_1) {
                            s_3[0] = b[1];
                            s_3[1] = b[0];
                            s_3[2] = b[3];
                            s_3[3] = b[2];
                        }
                        unsigned int r_4[4];
                        r_4[0] = s_3[0];
                        r_4[1] = s_3[1];
                        r_4[2] = s_3[2];
                        r_4[3] = s_3[3];
                        if (fhi_2) {
                            r_4[0] = s_3[2];
                            r_4[1] = s_3[3];
                            r_4[2] = s_3[0];
                            r_4[3] = s_3[1];
                        }
                        uint32_t _e4m3x2_to_f16x2_0;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(a_sc)));
                        float _cvt_f32_f16_0;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_e4m3x2_to_f16x2_0)));
                        float _cvt_f32_f16_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(_e4m3x2_to_f16x2_0 >> 16)));
                        float _mul_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_cvt_f32_f16_0), "f"(rs_a));
                        float _mul_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_cvt_f32_f16_1), "f"(rs_a));
                        float lo = _mul_0;
                        float hi = _mul_1;
                        if (fhi) {
                            lo = _mul_1;
                            hi = _mul_0;
                        }
                        float2 scale0 = make_float2(lo, lo);
                        float2 scale1 = make_float2(hi, hi);
                        float _unpack_fp4_0[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[0]));
                            (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(r[0]));
                            #endif
                        }
                        float2 a_5 = make_float2(acc[0], acc[1]);
                        float2 q = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                        float2 r_6 = fma_f32x2_rn_noftz(q, scale0, a_5);
                        acc[0] = r_6.x;
                        acc[1] = r_6.y;
                        float2 a_7 = make_float2(acc[2], acc[3]);
                        float2 q_8 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                        float2 r_9 = fma_f32x2_rn_noftz(q_8, scale0, a_7);
                        acc[2] = r_9.x;
                        acc[3] = r_9.y;
                        float2 a_10 = make_float2(acc[4], acc[5]);
                        float2 q_11 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                        float2 r_12 = fma_f32x2_rn_noftz(q_11, scale0, a_10);
                        acc[4] = r_12.x;
                        acc[5] = r_12.y;
                        float2 a_13 = make_float2(acc[6], acc[7]);
                        float2 q_14 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                        float2 r_15 = fma_f32x2_rn_noftz(q_14, scale0, a_13);
                        acc[6] = r_15.x;
                        acc[7] = r_15.y;
                        float _unpack_fp4_1[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[1 + 0]));
                            (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(r[1 + 0]));
                            #endif
                        }
                        float2 a_16 = make_float2(acc[8], acc[9]);
                        float2 q_17 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                        float2 r_18 = fma_f32x2_rn_noftz(q_17, scale0, a_16);
                        acc[8] = r_18.x;
                        acc[9] = r_18.y;
                        float2 a_19 = make_float2(acc[10], acc[11]);
                        float2 q_20 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                        float2 r_21 = fma_f32x2_rn_noftz(q_20, scale0, a_19);
                        acc[10] = r_21.x;
                        acc[11] = r_21.y;
                        float2 a_22 = make_float2(acc[12], acc[13]);
                        float2 q_23 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                        float2 r_24 = fma_f32x2_rn_noftz(q_23, scale0, a_22);
                        acc[12] = r_24.x;
                        acc[13] = r_24.y;
                        float2 a_25 = make_float2(acc[14], acc[15]);
                        float2 q_26 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                        float2 r_27 = fma_f32x2_rn_noftz(q_26, scale0, a_25);
                        acc[14] = r_27.x;
                        acc[15] = r_27.y;
                        float _unpack_fp4_2[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[2 + 0]));
                            (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(r[2 + 0]));
                            #endif
                        }
                        float2 a_28 = make_float2(acc[16], acc[17]);
                        float2 q_29 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                        float2 r_30 = fma_f32x2_rn_noftz(q_29, scale1, a_28);
                        acc[16] = r_30.x;
                        acc[17] = r_30.y;
                        float2 a_31 = make_float2(acc[18], acc[19]);
                        float2 q_32 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                        float2 r_33 = fma_f32x2_rn_noftz(q_32, scale1, a_31);
                        acc[18] = r_33.x;
                        acc[19] = r_33.y;
                        float2 a_34 = make_float2(acc[20], acc[21]);
                        float2 q_35 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                        float2 r_36 = fma_f32x2_rn_noftz(q_35, scale1, a_34);
                        acc[20] = r_36.x;
                        acc[21] = r_36.y;
                        float2 a_37 = make_float2(acc[22], acc[23]);
                        float2 q_38 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                        float2 r_39 = fma_f32x2_rn_noftz(q_38, scale1, a_37);
                        acc[22] = r_39.x;
                        acc[23] = r_39.y;
                        float _unpack_fp4_3[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[3 + 0]));
                            (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(r[3 + 0]));
                            #endif
                        }
                        float2 a_40 = make_float2(acc[24], acc[25]);
                        float2 q_41 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                        float2 r_42 = fma_f32x2_rn_noftz(q_41, scale1, a_40);
                        acc[24] = r_42.x;
                        acc[25] = r_42.y;
                        float2 a_43 = make_float2(acc[26], acc[27]);
                        float2 q_44 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                        float2 r_45 = fma_f32x2_rn_noftz(q_44, scale1, a_43);
                        acc[26] = r_45.x;
                        acc[27] = r_45.y;
                        float2 a_46 = make_float2(acc[28], acc[29]);
                        float2 q_47 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                        float2 r_48 = fma_f32x2_rn_noftz(q_47, scale1, a_46);
                        acc[28] = r_48.x;
                        acc[29] = r_48.y;
                        float2 a_49 = make_float2(acc[30], acc[31]);
                        float2 q_50 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                        float2 r_51 = fma_f32x2_rn_noftz(q_50, scale1, a_49);
                        acc[30] = r_51.x;
                        acc[31] = r_51.y;
                        uint32_t _e4m3x2_to_f16x2_1;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(b_sc)));
                        float _cvt_f32_f16_2;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(_e4m3x2_to_f16x2_1)));
                        float _cvt_f32_f16_3;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(_e4m3x2_to_f16x2_1 >> 16)));
                        float _mul_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_cvt_f32_f16_2), "f"(rs_b));
                        float _mul_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_cvt_f32_f16_3), "f"(rs_b));
                        float lo_52 = _mul_2;
                        float hi_53 = _mul_3;
                        if (fhi) {
                            lo_52 = _mul_3;
                            hi_53 = _mul_2;
                        }
                        float2 scale0_54 = make_float2(lo_52, lo_52);
                        float2 scale1_55 = make_float2(hi_53, hi_53);
                        float _unpack_fp4_4[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_4[0]));
                            (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(r_4[0]));
                            #endif
                        }
                        float2 a_56 = make_float2(acc[0], acc[1]);
                        float2 q_57 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                        float2 r_58 = fma_f32x2_rn_noftz(q_57, scale0_54, a_56);
                        acc[0] = r_58.x;
                        acc[1] = r_58.y;
                        float2 a_59 = make_float2(acc[2], acc[3]);
                        float2 q_60 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                        float2 r_61 = fma_f32x2_rn_noftz(q_60, scale0_54, a_59);
                        acc[2] = r_61.x;
                        acc[3] = r_61.y;
                        float2 a_62 = make_float2(acc[4], acc[5]);
                        float2 q_63 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                        float2 r_64 = fma_f32x2_rn_noftz(q_63, scale0_54, a_62);
                        acc[4] = r_64.x;
                        acc[5] = r_64.y;
                        float2 a_65 = make_float2(acc[6], acc[7]);
                        float2 q_66 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                        float2 r_67 = fma_f32x2_rn_noftz(q_66, scale0_54, a_65);
                        acc[6] = r_67.x;
                        acc[7] = r_67.y;
                        float _unpack_fp4_5[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_4[1 + 0]));
                            (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(r_4[1 + 0]));
                            #endif
                        }
                        float2 a_68 = make_float2(acc[8], acc[9]);
                        float2 q_69 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                        float2 r_70 = fma_f32x2_rn_noftz(q_69, scale0_54, a_68);
                        acc[8] = r_70.x;
                        acc[9] = r_70.y;
                        float2 a_71 = make_float2(acc[10], acc[11]);
                        float2 q_72 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                        float2 r_73 = fma_f32x2_rn_noftz(q_72, scale0_54, a_71);
                        acc[10] = r_73.x;
                        acc[11] = r_73.y;
                        float2 a_74 = make_float2(acc[12], acc[13]);
                        float2 q_75 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                        float2 r_76 = fma_f32x2_rn_noftz(q_75, scale0_54, a_74);
                        acc[12] = r_76.x;
                        acc[13] = r_76.y;
                        float2 a_77 = make_float2(acc[14], acc[15]);
                        float2 q_78 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                        float2 r_79 = fma_f32x2_rn_noftz(q_78, scale0_54, a_77);
                        acc[14] = r_79.x;
                        acc[15] = r_79.y;
                        float _unpack_fp4_6[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_4[2 + 0]));
                            (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(r_4[2 + 0]));
                            #endif
                        }
                        float2 a_80 = make_float2(acc[16], acc[17]);
                        float2 q_81 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                        float2 r_82 = fma_f32x2_rn_noftz(q_81, scale1_55, a_80);
                        acc[16] = r_82.x;
                        acc[17] = r_82.y;
                        float2 a_83 = make_float2(acc[18], acc[19]);
                        float2 q_84 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                        float2 r_85 = fma_f32x2_rn_noftz(q_84, scale1_55, a_83);
                        acc[18] = r_85.x;
                        acc[19] = r_85.y;
                        float2 a_86 = make_float2(acc[20], acc[21]);
                        float2 q_87 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                        float2 r_88 = fma_f32x2_rn_noftz(q_87, scale1_55, a_86);
                        acc[20] = r_88.x;
                        acc[21] = r_88.y;
                        float2 a_89 = make_float2(acc[22], acc[23]);
                        float2 q_90 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                        float2 r_91 = fma_f32x2_rn_noftz(q_90, scale1_55, a_89);
                        acc[22] = r_91.x;
                        acc[23] = r_91.y;
                        float _unpack_fp4_7[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_4[3 + 0]));
                            (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(r_4[3 + 0]));
                            #endif
                        }
                        float2 a_92 = make_float2(acc[24], acc[25]);
                        float2 q_93 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                        float2 r_94 = fma_f32x2_rn_noftz(q_93, scale1_55, a_92);
                        acc[24] = r_94.x;
                        acc[25] = r_94.y;
                        float2 a_95 = make_float2(acc[26], acc[27]);
                        float2 q_96 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                        float2 r_97 = fma_f32x2_rn_noftz(q_96, scale1_55, a_95);
                        acc[26] = r_97.x;
                        acc[27] = r_97.y;
                        float2 a_98 = make_float2(acc[28], acc[29]);
                        float2 q_99 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                        float2 r_100 = fma_f32x2_rn_noftz(q_99, scale1_55, a_98);
                        acc[28] = r_100.x;
                        acc[29] = r_100.y;
                        float2 a_101 = make_float2(acc[30], acc[31]);
                        float2 q_102 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                        float2 r_103 = fma_f32x2_rn_noftz(q_102, scale1_55, a_101);
                        acc[30] = r_103.x;
                        acc[31] = r_103.y;
                        unsigned int t[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[0])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[4])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[8])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[12])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 3))));
                        float2 a2 = make_float2(acc[0], acc[1]);
                        float2 z = add_f32x2_noftz(a2, make_float2(__uint_as_float(t[0] << 16), __uint_as_float(t[0] & 4294901760u)));
                        acc[0] = z.x;
                        acc[1] = z.y;
                        float2 ss2 = fma_f32x2_rn_noftz(z, z, make_float2(ssx, ssy));
                        ssx = ss2.x;
                        ssy = ss2.y;
                        float2 a2_104 = make_float2(acc[2], acc[3]);
                        float2 z_105 = add_f32x2_noftz(a2_104, make_float2(__uint_as_float(t[1] << 16), __uint_as_float(t[1] & 4294901760u)));
                        acc[2] = z_105.x;
                        acc[3] = z_105.y;
                        float2 ss2_106 = fma_f32x2_rn_noftz(z_105, z_105, make_float2(ssx, ssy));
                        ssx = ss2_106.x;
                        ssy = ss2_106.y;
                        float2 a2_107 = make_float2(acc[4], acc[5]);
                        float2 z_108 = add_f32x2_noftz(a2_107, make_float2(__uint_as_float(t[2] << 16), __uint_as_float(t[2] & 4294901760u)));
                        acc[4] = z_108.x;
                        acc[5] = z_108.y;
                        float2 ss2_109 = fma_f32x2_rn_noftz(z_108, z_108, make_float2(ssx, ssy));
                        ssx = ss2_109.x;
                        ssy = ss2_109.y;
                        float2 a2_110 = make_float2(acc[6], acc[7]);
                        float2 z_111 = add_f32x2_noftz(a2_110, make_float2(__uint_as_float(t[3] << 16), __uint_as_float(t[3] & 4294901760u)));
                        acc[6] = z_111.x;
                        acc[7] = z_111.y;
                        float2 ss2_112 = fma_f32x2_rn_noftz(z_111, z_111, make_float2(ssx, ssy));
                        ssx = ss2_112.x;
                        ssy = ss2_112.y;
                        float2 a2_113 = make_float2(acc[8], acc[9]);
                        float2 z_114 = add_f32x2_noftz(a2_113, make_float2(__uint_as_float(t[4] << 16), __uint_as_float(t[4] & 4294901760u)));
                        acc[8] = z_114.x;
                        acc[9] = z_114.y;
                        float2 ss2_115 = fma_f32x2_rn_noftz(z_114, z_114, make_float2(ssx, ssy));
                        ssx = ss2_115.x;
                        ssy = ss2_115.y;
                        float2 a2_116 = make_float2(acc[10], acc[11]);
                        float2 z_117 = add_f32x2_noftz(a2_116, make_float2(__uint_as_float(t[5] << 16), __uint_as_float(t[5] & 4294901760u)));
                        acc[10] = z_117.x;
                        acc[11] = z_117.y;
                        float2 ss2_118 = fma_f32x2_rn_noftz(z_117, z_117, make_float2(ssx, ssy));
                        ssx = ss2_118.x;
                        ssy = ss2_118.y;
                        float2 a2_119 = make_float2(acc[12], acc[13]);
                        float2 z_120 = add_f32x2_noftz(a2_119, make_float2(__uint_as_float(t[6] << 16), __uint_as_float(t[6] & 4294901760u)));
                        acc[12] = z_120.x;
                        acc[13] = z_120.y;
                        float2 ss2_121 = fma_f32x2_rn_noftz(z_120, z_120, make_float2(ssx, ssy));
                        ssx = ss2_121.x;
                        ssy = ss2_121.y;
                        float2 a2_122 = make_float2(acc[14], acc[15]);
                        float2 z_123 = add_f32x2_noftz(a2_122, make_float2(__uint_as_float(t[7] << 16), __uint_as_float(t[7] & 4294901760u)));
                        acc[14] = z_123.x;
                        acc[15] = z_123.y;
                        float2 ss2_124 = fma_f32x2_rn_noftz(z_123, z_123, make_float2(ssx, ssy));
                        ssx = ss2_124.x;
                        ssy = ss2_124.y;
                        float2 a2_125 = make_float2(acc[16], acc[17]);
                        float2 z_126 = add_f32x2_noftz(a2_125, make_float2(__uint_as_float(t[8] << 16), __uint_as_float(t[8] & 4294901760u)));
                        acc[16] = z_126.x;
                        acc[17] = z_126.y;
                        float2 ss2_127 = fma_f32x2_rn_noftz(z_126, z_126, make_float2(ssx, ssy));
                        ssx = ss2_127.x;
                        ssy = ss2_127.y;
                        float2 a2_128 = make_float2(acc[18], acc[19]);
                        float2 z_129 = add_f32x2_noftz(a2_128, make_float2(__uint_as_float(t[9] << 16), __uint_as_float(t[9] & 4294901760u)));
                        acc[18] = z_129.x;
                        acc[19] = z_129.y;
                        float2 ss2_130 = fma_f32x2_rn_noftz(z_129, z_129, make_float2(ssx, ssy));
                        ssx = ss2_130.x;
                        ssy = ss2_130.y;
                        float2 a2_131 = make_float2(acc[20], acc[21]);
                        float2 z_132 = add_f32x2_noftz(a2_131, make_float2(__uint_as_float(t[10] << 16), __uint_as_float(t[10] & 4294901760u)));
                        acc[20] = z_132.x;
                        acc[21] = z_132.y;
                        float2 ss2_133 = fma_f32x2_rn_noftz(z_132, z_132, make_float2(ssx, ssy));
                        ssx = ss2_133.x;
                        ssy = ss2_133.y;
                        float2 a2_134 = make_float2(acc[22], acc[23]);
                        float2 z_135 = add_f32x2_noftz(a2_134, make_float2(__uint_as_float(t[11] << 16), __uint_as_float(t[11] & 4294901760u)));
                        acc[22] = z_135.x;
                        acc[23] = z_135.y;
                        float2 ss2_136 = fma_f32x2_rn_noftz(z_135, z_135, make_float2(ssx, ssy));
                        ssx = ss2_136.x;
                        ssy = ss2_136.y;
                        float2 a2_137 = make_float2(acc[24], acc[25]);
                        float2 z_138 = add_f32x2_noftz(a2_137, make_float2(__uint_as_float(t[12] << 16), __uint_as_float(t[12] & 4294901760u)));
                        acc[24] = z_138.x;
                        acc[25] = z_138.y;
                        float2 ss2_139 = fma_f32x2_rn_noftz(z_138, z_138, make_float2(ssx, ssy));
                        ssx = ss2_139.x;
                        ssy = ss2_139.y;
                        float2 a2_140 = make_float2(acc[26], acc[27]);
                        float2 z_141 = add_f32x2_noftz(a2_140, make_float2(__uint_as_float(t[13] << 16), __uint_as_float(t[13] & 4294901760u)));
                        acc[26] = z_141.x;
                        acc[27] = z_141.y;
                        float2 ss2_142 = fma_f32x2_rn_noftz(z_141, z_141, make_float2(ssx, ssy));
                        ssx = ss2_142.x;
                        ssy = ss2_142.y;
                        float2 a2_143 = make_float2(acc[28], acc[29]);
                        float2 z_144 = add_f32x2_noftz(a2_143, make_float2(__uint_as_float(t[14] << 16), __uint_as_float(t[14] & 4294901760u)));
                        acc[28] = z_144.x;
                        acc[29] = z_144.y;
                        float2 ss2_145 = fma_f32x2_rn_noftz(z_144, z_144, make_float2(ssx, ssy));
                        ssx = ss2_145.x;
                        ssy = ss2_145.y;
                        float2 a2_146 = make_float2(acc[30], acc[31]);
                        float2 z_147 = add_f32x2_noftz(a2_146, make_float2(__uint_as_float(t[15] << 16), __uint_as_float(t[15] & 4294901760u)));
                        acc[30] = z_147.x;
                        acc[31] = z_147.y;
                        float2 ss2_148 = fma_f32x2_rn_noftz(z_147, z_147, make_float2(ssx, ssy));
                        ssx = ss2_148.x;
                        ssy = ss2_148.y;
                        unsigned int acc_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                            acc_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 0))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 1))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 2))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 3))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                        unsigned int own_149[4];
                        unsigned int own_sc_150[1];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own_149[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_149[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_149[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_149[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp1)));
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_150[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp1)));
                        unsigned int a_151[4];
                        unsigned int b_152[4];
                        a_151[0] = ((first) ? own_149[0] : _fp4_wire_values_0[4]);
                        b_152[0] = ((first) ? _fp4_wire_values_0[4] : own_149[0]);
                        a_151[1] = ((first) ? own_149[1] : _fp4_wire_values_0[5]);
                        b_152[1] = ((first) ? _fp4_wire_values_0[5] : own_149[1]);
                        a_151[2] = ((first) ? own_149[2] : _fp4_wire_values_0[6]);
                        b_152[2] = ((first) ? _fp4_wire_values_0[6] : own_149[2]);
                        a_151[3] = ((first) ? own_149[3] : _fp4_wire_values_0[7]);
                        b_152[3] = ((first) ? _fp4_wire_values_0[7] : own_149[3]);
                        unsigned int a_sc_153 = ((first) ? own_sc_150[0] : _fp4_wire_scales_0[1]);
                        unsigned int b_sc_154 = ((first) ? _fp4_wire_scales_0[1] : own_sc_150[0]);
                        float acc_155[32] = {0};
                        bool fhi_156 = (f & 2) != 0;
                        bool fodd_157 = (f & 1) != 0;
                        bool fhi_158 = (f & 2) != 0;
                        unsigned int s_159[4];
                        s_159[0] = a_151[0];
                        s_159[1] = a_151[1];
                        s_159[2] = a_151[2];
                        s_159[3] = a_151[3];
                        if (fodd_157) {
                            s_159[0] = a_151[1];
                            s_159[1] = a_151[0];
                            s_159[2] = a_151[3];
                            s_159[3] = a_151[2];
                        }
                        unsigned int r_160[4];
                        r_160[0] = s_159[0];
                        r_160[1] = s_159[1];
                        r_160[2] = s_159[2];
                        r_160[3] = s_159[3];
                        if (fhi_158) {
                            r_160[0] = s_159[2];
                            r_160[1] = s_159[3];
                            r_160[2] = s_159[0];
                            r_160[3] = s_159[1];
                        }
                        bool fodd_161 = (f & 1) != 0;
                        bool fhi_162 = (f & 2) != 0;
                        unsigned int s_163[4];
                        s_163[0] = b_152[0];
                        s_163[1] = b_152[1];
                        s_163[2] = b_152[2];
                        s_163[3] = b_152[3];
                        if (fodd_161) {
                            s_163[0] = b_152[1];
                            s_163[1] = b_152[0];
                            s_163[2] = b_152[3];
                            s_163[3] = b_152[2];
                        }
                        unsigned int r_164[4];
                        r_164[0] = s_163[0];
                        r_164[1] = s_163[1];
                        r_164[2] = s_163[2];
                        r_164[3] = s_163[3];
                        if (fhi_162) {
                            r_164[0] = s_163[2];
                            r_164[1] = s_163[3];
                            r_164[2] = s_163[0];
                            r_164[3] = s_163[1];
                        }
                        uint32_t _e4m3x2_to_f16x2_2;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(a_sc_153)));
                        float _cvt_f32_f16_4;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(_e4m3x2_to_f16x2_2)));
                        float _cvt_f32_f16_5;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(_e4m3x2_to_f16x2_2 >> 16)));
                        float _mul_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_cvt_f32_f16_4), "f"(rs_a));
                        float _mul_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_cvt_f32_f16_5), "f"(rs_a));
                        float lo_165 = _mul_4;
                        float hi_166 = _mul_5;
                        if (fhi_156) {
                            lo_165 = _mul_5;
                            hi_166 = _mul_4;
                        }
                        float2 scale0_167 = make_float2(lo_165, lo_165);
                        float2 scale1_168 = make_float2(hi_166, hi_166);
                        float _unpack_fp4_8[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_160[0]));
                            (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(r_160[0]));
                            #endif
                        }
                        float2 a_169 = make_float2(acc_155[0], acc_155[1]);
                        float2 q_170 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                        float2 r_171 = fma_f32x2_rn_noftz(q_170, scale0_167, a_169);
                        acc_155[0] = r_171.x;
                        acc_155[1] = r_171.y;
                        float2 a_172 = make_float2(acc_155[2], acc_155[3]);
                        float2 q_173 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                        float2 r_174 = fma_f32x2_rn_noftz(q_173, scale0_167, a_172);
                        acc_155[2] = r_174.x;
                        acc_155[3] = r_174.y;
                        float2 a_175 = make_float2(acc_155[4], acc_155[5]);
                        float2 q_176 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                        float2 r_177 = fma_f32x2_rn_noftz(q_176, scale0_167, a_175);
                        acc_155[4] = r_177.x;
                        acc_155[5] = r_177.y;
                        float2 a_178 = make_float2(acc_155[6], acc_155[7]);
                        float2 q_179 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                        float2 r_180 = fma_f32x2_rn_noftz(q_179, scale0_167, a_178);
                        acc_155[6] = r_180.x;
                        acc_155[7] = r_180.y;
                        float _unpack_fp4_9[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_160[1 + 0]));
                            (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(r_160[1 + 0]));
                            #endif
                        }
                        float2 a_181 = make_float2(acc_155[8], acc_155[9]);
                        float2 q_182 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                        float2 r_183 = fma_f32x2_rn_noftz(q_182, scale0_167, a_181);
                        acc_155[8] = r_183.x;
                        acc_155[9] = r_183.y;
                        float2 a_184 = make_float2(acc_155[10], acc_155[11]);
                        float2 q_185 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                        float2 r_186 = fma_f32x2_rn_noftz(q_185, scale0_167, a_184);
                        acc_155[10] = r_186.x;
                        acc_155[11] = r_186.y;
                        float2 a_187 = make_float2(acc_155[12], acc_155[13]);
                        float2 q_188 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                        float2 r_189 = fma_f32x2_rn_noftz(q_188, scale0_167, a_187);
                        acc_155[12] = r_189.x;
                        acc_155[13] = r_189.y;
                        float2 a_190 = make_float2(acc_155[14], acc_155[15]);
                        float2 q_191 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                        float2 r_192 = fma_f32x2_rn_noftz(q_191, scale0_167, a_190);
                        acc_155[14] = r_192.x;
                        acc_155[15] = r_192.y;
                        float _unpack_fp4_10[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_160[2 + 0]));
                            (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(r_160[2 + 0]));
                            #endif
                        }
                        float2 a_193 = make_float2(acc_155[16], acc_155[17]);
                        float2 q_194 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                        float2 r_195 = fma_f32x2_rn_noftz(q_194, scale1_168, a_193);
                        acc_155[16] = r_195.x;
                        acc_155[17] = r_195.y;
                        float2 a_196 = make_float2(acc_155[18], acc_155[19]);
                        float2 q_197 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                        float2 r_198 = fma_f32x2_rn_noftz(q_197, scale1_168, a_196);
                        acc_155[18] = r_198.x;
                        acc_155[19] = r_198.y;
                        float2 a_199 = make_float2(acc_155[20], acc_155[21]);
                        float2 q_200 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                        float2 r_201 = fma_f32x2_rn_noftz(q_200, scale1_168, a_199);
                        acc_155[20] = r_201.x;
                        acc_155[21] = r_201.y;
                        float2 a_202 = make_float2(acc_155[22], acc_155[23]);
                        float2 q_203 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                        float2 r_204 = fma_f32x2_rn_noftz(q_203, scale1_168, a_202);
                        acc_155[22] = r_204.x;
                        acc_155[23] = r_204.y;
                        float _unpack_fp4_11[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_160[3 + 0]));
                            (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(r_160[3 + 0]));
                            #endif
                        }
                        float2 a_205 = make_float2(acc_155[24], acc_155[25]);
                        float2 q_206 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                        float2 r_207 = fma_f32x2_rn_noftz(q_206, scale1_168, a_205);
                        acc_155[24] = r_207.x;
                        acc_155[25] = r_207.y;
                        float2 a_208 = make_float2(acc_155[26], acc_155[27]);
                        float2 q_209 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                        float2 r_210 = fma_f32x2_rn_noftz(q_209, scale1_168, a_208);
                        acc_155[26] = r_210.x;
                        acc_155[27] = r_210.y;
                        float2 a_211 = make_float2(acc_155[28], acc_155[29]);
                        float2 q_212 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                        float2 r_213 = fma_f32x2_rn_noftz(q_212, scale1_168, a_211);
                        acc_155[28] = r_213.x;
                        acc_155[29] = r_213.y;
                        float2 a_214 = make_float2(acc_155[30], acc_155[31]);
                        float2 q_215 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                        float2 r_216 = fma_f32x2_rn_noftz(q_215, scale1_168, a_214);
                        acc_155[30] = r_216.x;
                        acc_155[31] = r_216.y;
                        uint32_t _e4m3x2_to_f16x2_3;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(b_sc_154)));
                        float _cvt_f32_f16_6;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(_e4m3x2_to_f16x2_3)));
                        float _cvt_f32_f16_7;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(_e4m3x2_to_f16x2_3 >> 16)));
                        float _mul_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_cvt_f32_f16_6), "f"(rs_b));
                        float _mul_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_cvt_f32_f16_7), "f"(rs_b));
                        float lo_217 = _mul_6;
                        float hi_218 = _mul_7;
                        if (fhi_156) {
                            lo_217 = _mul_7;
                            hi_218 = _mul_6;
                        }
                        float2 scale0_219 = make_float2(lo_217, lo_217);
                        float2 scale1_220 = make_float2(hi_218, hi_218);
                        float _unpack_fp4_12[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_164[0]));
                            (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(r_164[0]));
                            #endif
                        }
                        float2 a_221 = make_float2(acc_155[0], acc_155[1]);
                        float2 q_222 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                        float2 r_223 = fma_f32x2_rn_noftz(q_222, scale0_219, a_221);
                        acc_155[0] = r_223.x;
                        acc_155[1] = r_223.y;
                        float2 a_224 = make_float2(acc_155[2], acc_155[3]);
                        float2 q_225 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                        float2 r_226 = fma_f32x2_rn_noftz(q_225, scale0_219, a_224);
                        acc_155[2] = r_226.x;
                        acc_155[3] = r_226.y;
                        float2 a_227 = make_float2(acc_155[4], acc_155[5]);
                        float2 q_228 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                        float2 r_229 = fma_f32x2_rn_noftz(q_228, scale0_219, a_227);
                        acc_155[4] = r_229.x;
                        acc_155[5] = r_229.y;
                        float2 a_230 = make_float2(acc_155[6], acc_155[7]);
                        float2 q_231 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                        float2 r_232 = fma_f32x2_rn_noftz(q_231, scale0_219, a_230);
                        acc_155[6] = r_232.x;
                        acc_155[7] = r_232.y;
                        float _unpack_fp4_13[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_164[1 + 0]));
                            (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(r_164[1 + 0]));
                            #endif
                        }
                        float2 a_233 = make_float2(acc_155[8], acc_155[9]);
                        float2 q_234 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                        float2 r_235 = fma_f32x2_rn_noftz(q_234, scale0_219, a_233);
                        acc_155[8] = r_235.x;
                        acc_155[9] = r_235.y;
                        float2 a_236 = make_float2(acc_155[10], acc_155[11]);
                        float2 q_237 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                        float2 r_238 = fma_f32x2_rn_noftz(q_237, scale0_219, a_236);
                        acc_155[10] = r_238.x;
                        acc_155[11] = r_238.y;
                        float2 a_239 = make_float2(acc_155[12], acc_155[13]);
                        float2 q_240 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                        float2 r_241 = fma_f32x2_rn_noftz(q_240, scale0_219, a_239);
                        acc_155[12] = r_241.x;
                        acc_155[13] = r_241.y;
                        float2 a_242 = make_float2(acc_155[14], acc_155[15]);
                        float2 q_243 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                        float2 r_244 = fma_f32x2_rn_noftz(q_243, scale0_219, a_242);
                        acc_155[14] = r_244.x;
                        acc_155[15] = r_244.y;
                        float _unpack_fp4_14[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_164[2 + 0]));
                            (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(r_164[2 + 0]));
                            #endif
                        }
                        float2 a_245 = make_float2(acc_155[16], acc_155[17]);
                        float2 q_246 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                        float2 r_247 = fma_f32x2_rn_noftz(q_246, scale1_220, a_245);
                        acc_155[16] = r_247.x;
                        acc_155[17] = r_247.y;
                        float2 a_248 = make_float2(acc_155[18], acc_155[19]);
                        float2 q_249 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                        float2 r_250 = fma_f32x2_rn_noftz(q_249, scale1_220, a_248);
                        acc_155[18] = r_250.x;
                        acc_155[19] = r_250.y;
                        float2 a_251 = make_float2(acc_155[20], acc_155[21]);
                        float2 q_252 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                        float2 r_253 = fma_f32x2_rn_noftz(q_252, scale1_220, a_251);
                        acc_155[20] = r_253.x;
                        acc_155[21] = r_253.y;
                        float2 a_254 = make_float2(acc_155[22], acc_155[23]);
                        float2 q_255 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                        float2 r_256 = fma_f32x2_rn_noftz(q_255, scale1_220, a_254);
                        acc_155[22] = r_256.x;
                        acc_155[23] = r_256.y;
                        float _unpack_fp4_15[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_164[3 + 0]));
                            (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(r_164[3 + 0]));
                            #endif
                        }
                        float2 a_257 = make_float2(acc_155[24], acc_155[25]);
                        float2 q_258 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                        float2 r_259 = fma_f32x2_rn_noftz(q_258, scale1_220, a_257);
                        acc_155[24] = r_259.x;
                        acc_155[25] = r_259.y;
                        float2 a_260 = make_float2(acc_155[26], acc_155[27]);
                        float2 q_261 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                        float2 r_262 = fma_f32x2_rn_noftz(q_261, scale1_220, a_260);
                        acc_155[26] = r_262.x;
                        acc_155[27] = r_262.y;
                        float2 a_263 = make_float2(acc_155[28], acc_155[29]);
                        float2 q_264 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                        float2 r_265 = fma_f32x2_rn_noftz(q_264, scale1_220, a_263);
                        acc_155[28] = r_265.x;
                        acc_155[29] = r_265.y;
                        float2 a_266 = make_float2(acc_155[30], acc_155[31]);
                        float2 q_267 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                        float2 r_268 = fma_f32x2_rn_noftz(q_267, scale1_220, a_266);
                        acc_155[30] = r_268.x;
                        acc_155[31] = r_268.y;
                        unsigned int t_269[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_269[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_269[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_269[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_269[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 3))));
                        float2 a2_270 = make_float2(acc_155[0], acc_155[1]);
                        float2 z_271 = add_f32x2_noftz(a2_270, make_float2(__uint_as_float(t_269[0] << 16), __uint_as_float(t_269[0] & 4294901760u)));
                        acc_155[0] = z_271.x;
                        acc_155[1] = z_271.y;
                        float2 ss2_272 = fma_f32x2_rn_noftz(z_271, z_271, make_float2(ssx, ssy));
                        ssx = ss2_272.x;
                        ssy = ss2_272.y;
                        float2 a2_273 = make_float2(acc_155[2], acc_155[3]);
                        float2 z_274 = add_f32x2_noftz(a2_273, make_float2(__uint_as_float(t_269[1] << 16), __uint_as_float(t_269[1] & 4294901760u)));
                        acc_155[2] = z_274.x;
                        acc_155[3] = z_274.y;
                        float2 ss2_275 = fma_f32x2_rn_noftz(z_274, z_274, make_float2(ssx, ssy));
                        ssx = ss2_275.x;
                        ssy = ss2_275.y;
                        float2 a2_276 = make_float2(acc_155[4], acc_155[5]);
                        float2 z_277 = add_f32x2_noftz(a2_276, make_float2(__uint_as_float(t_269[2] << 16), __uint_as_float(t_269[2] & 4294901760u)));
                        acc_155[4] = z_277.x;
                        acc_155[5] = z_277.y;
                        float2 ss2_278 = fma_f32x2_rn_noftz(z_277, z_277, make_float2(ssx, ssy));
                        ssx = ss2_278.x;
                        ssy = ss2_278.y;
                        float2 a2_279 = make_float2(acc_155[6], acc_155[7]);
                        float2 z_280 = add_f32x2_noftz(a2_279, make_float2(__uint_as_float(t_269[3] << 16), __uint_as_float(t_269[3] & 4294901760u)));
                        acc_155[6] = z_280.x;
                        acc_155[7] = z_280.y;
                        float2 ss2_281 = fma_f32x2_rn_noftz(z_280, z_280, make_float2(ssx, ssy));
                        ssx = ss2_281.x;
                        ssy = ss2_281.y;
                        float2 a2_282 = make_float2(acc_155[8], acc_155[9]);
                        float2 z_283 = add_f32x2_noftz(a2_282, make_float2(__uint_as_float(t_269[4] << 16), __uint_as_float(t_269[4] & 4294901760u)));
                        acc_155[8] = z_283.x;
                        acc_155[9] = z_283.y;
                        float2 ss2_284 = fma_f32x2_rn_noftz(z_283, z_283, make_float2(ssx, ssy));
                        ssx = ss2_284.x;
                        ssy = ss2_284.y;
                        float2 a2_285 = make_float2(acc_155[10], acc_155[11]);
                        float2 z_286 = add_f32x2_noftz(a2_285, make_float2(__uint_as_float(t_269[5] << 16), __uint_as_float(t_269[5] & 4294901760u)));
                        acc_155[10] = z_286.x;
                        acc_155[11] = z_286.y;
                        float2 ss2_287 = fma_f32x2_rn_noftz(z_286, z_286, make_float2(ssx, ssy));
                        ssx = ss2_287.x;
                        ssy = ss2_287.y;
                        float2 a2_288 = make_float2(acc_155[12], acc_155[13]);
                        float2 z_289 = add_f32x2_noftz(a2_288, make_float2(__uint_as_float(t_269[6] << 16), __uint_as_float(t_269[6] & 4294901760u)));
                        acc_155[12] = z_289.x;
                        acc_155[13] = z_289.y;
                        float2 ss2_290 = fma_f32x2_rn_noftz(z_289, z_289, make_float2(ssx, ssy));
                        ssx = ss2_290.x;
                        ssy = ss2_290.y;
                        float2 a2_291 = make_float2(acc_155[14], acc_155[15]);
                        float2 z_292 = add_f32x2_noftz(a2_291, make_float2(__uint_as_float(t_269[7] << 16), __uint_as_float(t_269[7] & 4294901760u)));
                        acc_155[14] = z_292.x;
                        acc_155[15] = z_292.y;
                        float2 ss2_293 = fma_f32x2_rn_noftz(z_292, z_292, make_float2(ssx, ssy));
                        ssx = ss2_293.x;
                        ssy = ss2_293.y;
                        float2 a2_294 = make_float2(acc_155[16], acc_155[17]);
                        float2 z_295 = add_f32x2_noftz(a2_294, make_float2(__uint_as_float(t_269[8] << 16), __uint_as_float(t_269[8] & 4294901760u)));
                        acc_155[16] = z_295.x;
                        acc_155[17] = z_295.y;
                        float2 ss2_296 = fma_f32x2_rn_noftz(z_295, z_295, make_float2(ssx, ssy));
                        ssx = ss2_296.x;
                        ssy = ss2_296.y;
                        float2 a2_297 = make_float2(acc_155[18], acc_155[19]);
                        float2 z_298 = add_f32x2_noftz(a2_297, make_float2(__uint_as_float(t_269[9] << 16), __uint_as_float(t_269[9] & 4294901760u)));
                        acc_155[18] = z_298.x;
                        acc_155[19] = z_298.y;
                        float2 ss2_299 = fma_f32x2_rn_noftz(z_298, z_298, make_float2(ssx, ssy));
                        ssx = ss2_299.x;
                        ssy = ss2_299.y;
                        float2 a2_300 = make_float2(acc_155[20], acc_155[21]);
                        float2 z_301 = add_f32x2_noftz(a2_300, make_float2(__uint_as_float(t_269[10] << 16), __uint_as_float(t_269[10] & 4294901760u)));
                        acc_155[20] = z_301.x;
                        acc_155[21] = z_301.y;
                        float2 ss2_302 = fma_f32x2_rn_noftz(z_301, z_301, make_float2(ssx, ssy));
                        ssx = ss2_302.x;
                        ssy = ss2_302.y;
                        float2 a2_303 = make_float2(acc_155[22], acc_155[23]);
                        float2 z_304 = add_f32x2_noftz(a2_303, make_float2(__uint_as_float(t_269[11] << 16), __uint_as_float(t_269[11] & 4294901760u)));
                        acc_155[22] = z_304.x;
                        acc_155[23] = z_304.y;
                        float2 ss2_305 = fma_f32x2_rn_noftz(z_304, z_304, make_float2(ssx, ssy));
                        ssx = ss2_305.x;
                        ssy = ss2_305.y;
                        float2 a2_306 = make_float2(acc_155[24], acc_155[25]);
                        float2 z_307 = add_f32x2_noftz(a2_306, make_float2(__uint_as_float(t_269[12] << 16), __uint_as_float(t_269[12] & 4294901760u)));
                        acc_155[24] = z_307.x;
                        acc_155[25] = z_307.y;
                        float2 ss2_308 = fma_f32x2_rn_noftz(z_307, z_307, make_float2(ssx, ssy));
                        ssx = ss2_308.x;
                        ssy = ss2_308.y;
                        float2 a2_309 = make_float2(acc_155[26], acc_155[27]);
                        float2 z_310 = add_f32x2_noftz(a2_309, make_float2(__uint_as_float(t_269[13] << 16), __uint_as_float(t_269[13] & 4294901760u)));
                        acc_155[26] = z_310.x;
                        acc_155[27] = z_310.y;
                        float2 ss2_311 = fma_f32x2_rn_noftz(z_310, z_310, make_float2(ssx, ssy));
                        ssx = ss2_311.x;
                        ssy = ss2_311.y;
                        float2 a2_312 = make_float2(acc_155[28], acc_155[29]);
                        float2 z_313 = add_f32x2_noftz(a2_312, make_float2(__uint_as_float(t_269[14] << 16), __uint_as_float(t_269[14] & 4294901760u)));
                        acc_155[28] = z_313.x;
                        acc_155[29] = z_313.y;
                        float2 ss2_314 = fma_f32x2_rn_noftz(z_313, z_313, make_float2(ssx, ssy));
                        ssx = ss2_314.x;
                        ssy = ss2_314.y;
                        float2 a2_315 = make_float2(acc_155[30], acc_155[31]);
                        float2 z_316 = add_f32x2_noftz(a2_315, make_float2(__uint_as_float(t_269[15] << 16), __uint_as_float(t_269[15] & 4294901760u)));
                        acc_155[30] = z_316.x;
                        acc_155[31] = z_316.y;
                        float2 ss2_317 = fma_f32x2_rn_noftz(z_316, z_316, make_float2(ssx, ssy));
                        ssx = ss2_317.x;
                        ssy = ss2_317.y;
                        unsigned int acc_155_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_155[_lp*2 + 0], acc_155[_lp*2+1 + 0]));
                            acc_155_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 0))), "r"(acc_155_bf16[0]), "r"(acc_155_bf16[1]), "r"(acc_155_bf16[2]), "r"(acc_155_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 1))), "r"(acc_155_bf16[4]), "r"(acc_155_bf16[5]), "r"(acc_155_bf16[6]), "r"(acc_155_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 2))), "r"(acc_155_bf16[8]), "r"(acc_155_bf16[9]), "r"(acc_155_bf16[10]), "r"(acc_155_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 3))), "r"(acc_155_bf16[12]), "r"(acc_155_bf16[13]), "r"(acc_155_bf16[14]), "r"(acc_155_bf16[15]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        {
                            void* _cpbulk_dst_6 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(res_out) + (drow * 12288));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_6), "r"(wsm_addr + (unsigned int)rb_off), "r"((uint32_t)(12288))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    float ss;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ss) : "f"(ssx), "f"(ssy));
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, ss, 16);
                    float _add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(ss), "f"(_shfl_xor_sync_8));
                    ss = _add_7;
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, ss, 8);
                    float _add_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(ss), "f"(_shfl_xor_sync_10));
                    ss = _add_9;
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, ss, 4);
                    float _add_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(ss), "f"(_shfl_xor_sync_12));
                    ss = _add_11;
                    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, ss, 2);
                    float _add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(ss), "f"(_shfl_xor_sync_14));
                    ss = _add_13;
                    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, ss, 1);
                    float _add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(ss), "f"(_shfl_xor_sync_16));
                    ss = _add_15;
                    float _fdiv_17 = __fdiv_rn(ss, 6144.0f);
                    float arg;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(arg) : "f"(_fdiv_17), "f"(eps));
                    float rinv;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rinv) : "f"(arg));
                    float2 r2 = make_float2(rinv, rinv);
                    float2 b2 = make_float2(wbias, wbias);
                    float six = 6.0f;
                    float e4m3_max = 448.0f;
                    float rcp6 = approx_rcp(six);
                    float rcp_gs = approx_rcp(gso);
                    float rcp448 = approx_rcp(e4m3_max);
                    #pragma unroll 1
                    for (int j = 0; j < 6; j++) {
                        int pp = lane_2 + 32 * j;
                        unsigned int tx[16];
                        unsigned int tw[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[0])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[4])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[4])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[8])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[8])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[12])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp + 16 * (f ^ 3))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[12])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp + 16 * (f ^ 3))));
                        float y[32];
                        float2 _f32x2_mul_a_18 = make_float2(__uint_as_float(tx[0] << 16), __uint_as_float(tx[0] & 4294901760u));
                        float2 tz;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&r2));
                        float2 tw_0 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[0] << 16), __uint_as_float(tw[0] & 4294901760u)));
                        float2 t_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&tz), "l"(*(const unsigned long long*)&tw_0));
                        y[0] = t_1.x;
                        y[1] = t_1.y;
                        float2 _f32x2_mul_a_19 = make_float2(__uint_as_float(tx[1] << 16), __uint_as_float(tx[1] & 4294901760u));
                        float2 tz_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&r2));
                        float2 tw_2 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[1] << 16), __uint_as_float(tw[1] & 4294901760u)));
                        float2 t_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&tz_1), "l"(*(const unsigned long long*)&tw_2));
                        y[2] = t_3.x;
                        y[3] = t_3.y;
                        float2 _f32x2_mul_a_20 = make_float2(__uint_as_float(tx[2] << 16), __uint_as_float(tx[2] & 4294901760u));
                        float2 tz_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&r2));
                        float2 tw_5 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[2] << 16), __uint_as_float(tw[2] & 4294901760u)));
                        float2 t_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&tz_4), "l"(*(const unsigned long long*)&tw_5));
                        y[4] = t_6.x;
                        y[5] = t_6.y;
                        float2 _f32x2_mul_a_21 = make_float2(__uint_as_float(tx[3] << 16), __uint_as_float(tx[3] & 4294901760u));
                        float2 tz_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&r2));
                        float2 tw_8 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[3] << 16), __uint_as_float(tw[3] & 4294901760u)));
                        float2 t_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&tz_7), "l"(*(const unsigned long long*)&tw_8));
                        y[6] = t_9.x;
                        y[7] = t_9.y;
                        float2 _f32x2_mul_a_22 = make_float2(__uint_as_float(tx[4] << 16), __uint_as_float(tx[4] & 4294901760u));
                        float2 tz_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&r2));
                        float2 tw_11 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[4] << 16), __uint_as_float(tw[4] & 4294901760u)));
                        float2 t_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&tz_10), "l"(*(const unsigned long long*)&tw_11));
                        y[8] = t_12.x;
                        y[9] = t_12.y;
                        float2 _f32x2_mul_a_23 = make_float2(__uint_as_float(tx[5] << 16), __uint_as_float(tx[5] & 4294901760u));
                        float2 tz_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&r2));
                        float2 tw_14 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[5] << 16), __uint_as_float(tw[5] & 4294901760u)));
                        float2 t_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&tz_13), "l"(*(const unsigned long long*)&tw_14));
                        y[10] = t_15.x;
                        y[11] = t_15.y;
                        float2 _f32x2_mul_a_24 = make_float2(__uint_as_float(tx[6] << 16), __uint_as_float(tx[6] & 4294901760u));
                        float2 tz_16;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&r2));
                        float2 tw_17 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[6] << 16), __uint_as_float(tw[6] & 4294901760u)));
                        float2 t_18;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&tz_16), "l"(*(const unsigned long long*)&tw_17));
                        y[12] = t_18.x;
                        y[13] = t_18.y;
                        float2 _f32x2_mul_a_25 = make_float2(__uint_as_float(tx[7] << 16), __uint_as_float(tx[7] & 4294901760u));
                        float2 tz_19;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&r2));
                        float2 tw_20 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[7] << 16), __uint_as_float(tw[7] & 4294901760u)));
                        float2 t_21;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&tz_19), "l"(*(const unsigned long long*)&tw_20));
                        y[14] = t_21.x;
                        y[15] = t_21.y;
                        float2 _f32x2_mul_a_26 = make_float2(__uint_as_float(tx[8] << 16), __uint_as_float(tx[8] & 4294901760u));
                        float2 tz_22;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&r2));
                        float2 tw_23 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[8] << 16), __uint_as_float(tw[8] & 4294901760u)));
                        float2 t_24;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_24) : "l"(*(const unsigned long long*)&tz_22), "l"(*(const unsigned long long*)&tw_23));
                        y[16] = t_24.x;
                        y[17] = t_24.y;
                        float2 _f32x2_mul_a_27 = make_float2(__uint_as_float(tx[9] << 16), __uint_as_float(tx[9] & 4294901760u));
                        float2 tz_25;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&r2));
                        float2 tw_26 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[9] << 16), __uint_as_float(tw[9] & 4294901760u)));
                        float2 t_27;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&tz_25), "l"(*(const unsigned long long*)&tw_26));
                        y[18] = t_27.x;
                        y[19] = t_27.y;
                        float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(tx[10] << 16), __uint_as_float(tx[10] & 4294901760u));
                        float2 tz_28;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&r2));
                        float2 tw_29 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[10] << 16), __uint_as_float(tw[10] & 4294901760u)));
                        float2 t_30;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_30) : "l"(*(const unsigned long long*)&tz_28), "l"(*(const unsigned long long*)&tw_29));
                        y[20] = t_30.x;
                        y[21] = t_30.y;
                        float2 _f32x2_mul_a_29 = make_float2(__uint_as_float(tx[11] << 16), __uint_as_float(tx[11] & 4294901760u));
                        float2 tz_31;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&r2));
                        float2 tw_32 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[11] << 16), __uint_as_float(tw[11] & 4294901760u)));
                        float2 t_33;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_33) : "l"(*(const unsigned long long*)&tz_31), "l"(*(const unsigned long long*)&tw_32));
                        y[22] = t_33.x;
                        y[23] = t_33.y;
                        float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(tx[12] << 16), __uint_as_float(tx[12] & 4294901760u));
                        float2 tz_34;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&r2));
                        float2 tw_35 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[12] << 16), __uint_as_float(tw[12] & 4294901760u)));
                        float2 t_36;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_36) : "l"(*(const unsigned long long*)&tz_34), "l"(*(const unsigned long long*)&tw_35));
                        y[24] = t_36.x;
                        y[25] = t_36.y;
                        float2 _f32x2_mul_a_31 = make_float2(__uint_as_float(tx[13] << 16), __uint_as_float(tx[13] & 4294901760u));
                        float2 tz_37;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&r2));
                        float2 tw_38 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[13] << 16), __uint_as_float(tw[13] & 4294901760u)));
                        float2 t_39;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_39) : "l"(*(const unsigned long long*)&tz_37), "l"(*(const unsigned long long*)&tw_38));
                        y[26] = t_39.x;
                        y[27] = t_39.y;
                        float2 _f32x2_mul_a_32 = make_float2(__uint_as_float(tx[14] << 16), __uint_as_float(tx[14] & 4294901760u));
                        float2 tz_40;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&r2));
                        float2 tw_41 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[14] << 16), __uint_as_float(tw[14] & 4294901760u)));
                        float2 t_42;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_42) : "l"(*(const unsigned long long*)&tz_40), "l"(*(const unsigned long long*)&tw_41));
                        y[28] = t_42.x;
                        y[29] = t_42.y;
                        float2 _f32x2_mul_a_33 = make_float2(__uint_as_float(tx[15] << 16), __uint_as_float(tx[15] & 4294901760u));
                        float2 tz_43;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&r2));
                        float2 tw_44 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[15] << 16), __uint_as_float(tw[15] & 4294901760u)));
                        float2 t_45;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_45) : "l"(*(const unsigned long long*)&tz_43), "l"(*(const unsigned long long*)&tw_44));
                        y[30] = t_45.x;
                        y[31] = t_45.y;
                        unsigned int y_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
                            y_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float yb[32];
                        yb[0] = __uint_as_float(y_bf16[0] << 16);
                        yb[1] = __uint_as_float(y_bf16[0] & 4294901760u);
                        yb[2] = __uint_as_float(y_bf16[1] << 16);
                        yb[3] = __uint_as_float(y_bf16[1] & 4294901760u);
                        yb[4] = __uint_as_float(y_bf16[2] << 16);
                        yb[5] = __uint_as_float(y_bf16[2] & 4294901760u);
                        yb[6] = __uint_as_float(y_bf16[3] << 16);
                        yb[7] = __uint_as_float(y_bf16[3] & 4294901760u);
                        yb[8] = __uint_as_float(y_bf16[4] << 16);
                        yb[9] = __uint_as_float(y_bf16[4] & 4294901760u);
                        yb[10] = __uint_as_float(y_bf16[5] << 16);
                        yb[11] = __uint_as_float(y_bf16[5] & 4294901760u);
                        yb[12] = __uint_as_float(y_bf16[6] << 16);
                        yb[13] = __uint_as_float(y_bf16[6] & 4294901760u);
                        yb[14] = __uint_as_float(y_bf16[7] << 16);
                        yb[15] = __uint_as_float(y_bf16[7] & 4294901760u);
                        yb[16] = __uint_as_float(y_bf16[8] << 16);
                        yb[17] = __uint_as_float(y_bf16[8] & 4294901760u);
                        yb[18] = __uint_as_float(y_bf16[9] << 16);
                        yb[19] = __uint_as_float(y_bf16[9] & 4294901760u);
                        yb[20] = __uint_as_float(y_bf16[10] << 16);
                        yb[21] = __uint_as_float(y_bf16[10] & 4294901760u);
                        yb[22] = __uint_as_float(y_bf16[11] << 16);
                        yb[23] = __uint_as_float(y_bf16[11] & 4294901760u);
                        yb[24] = __uint_as_float(y_bf16[12] << 16);
                        yb[25] = __uint_as_float(y_bf16[12] & 4294901760u);
                        yb[26] = __uint_as_float(y_bf16[13] << 16);
                        yb[27] = __uint_as_float(y_bf16[13] & 4294901760u);
                        yb[28] = __uint_as_float(y_bf16[14] << 16);
                        yb[29] = __uint_as_float(y_bf16[14] & 4294901760u);
                        yb[30] = __uint_as_float(y_bf16[15] << 16);
                        yb[31] = __uint_as_float(y_bf16[15] & 4294901760u);
                        float zero = 0.0f;
                        bool fodd_2 = (f & 1) != 0;
                        bool fhi_1 = (f & 2) != 0;
                        float amag0 = fabsf(yb[0]);
                        float amag1 = fabsf(yb[1]);
                        float amag2 = fabsf(yb[2]);
                        float amag3 = fabsf(yb[3]);
                        float amag4 = fabsf(yb[4]);
                        float amag5 = fabsf(yb[5]);
                        float amag6 = fabsf(yb[6]);
                        float amag7 = fabsf(yb[7]);
                        float amag8 = fabsf(yb[8]);
                        float amag9 = fabsf(yb[9]);
                        float amag10 = fabsf(yb[10]);
                        float amag11 = fabsf(yb[11]);
                        float amag12 = fabsf(yb[12]);
                        float amag13 = fabsf(yb[13]);
                        float amag14 = fabsf(yb[14]);
                        float amag15 = fabsf(yb[15]);
                        float amax0_0;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_0) : "f"(zero), "f"(amag0), "f"(amag1));
                        float amax0_1;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_1) : "f"(amag2), "f"(amag3), "f"(amag4));
                        float amax0_2;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_2) : "f"(amag5), "f"(amag6), "f"(amag7));
                        float amax0_3;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_3) : "f"(amag8), "f"(amag9), "f"(amag10));
                        float amax0_4;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_4) : "f"(amag11), "f"(amag12), "f"(amag13));
                        float amax0_5;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_5) : "f"(amag14), "f"(amag15), "f"(zero));
                        float amax1_0;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax1_0) : "f"(amax0_0), "f"(amax0_1), "f"(amax0_2));
                        float amax1_1;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax1_1) : "f"(amax0_3), "f"(amax0_4), "f"(amax0_5));
                        float amax2_0;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax2_0) : "f"(amax1_0), "f"(amax1_1), "f"(zero));
                        float amag0_46 = fabsf(yb[16]);
                        float amag1_47 = fabsf(yb[17]);
                        float amag2_48 = fabsf(yb[18]);
                        float amag3_49 = fabsf(yb[19]);
                        float amag4_50 = fabsf(yb[20]);
                        float amag5_51 = fabsf(yb[21]);
                        float amag6_52 = fabsf(yb[22]);
                        float amag7_53 = fabsf(yb[23]);
                        float amag8_54 = fabsf(yb[24]);
                        float amag9_55 = fabsf(yb[25]);
                        float amag10_56 = fabsf(yb[26]);
                        float amag11_57 = fabsf(yb[27]);
                        float amag12_58 = fabsf(yb[28]);
                        float amag13_59 = fabsf(yb[29]);
                        float amag14_60 = fabsf(yb[30]);
                        float amag15_61 = fabsf(yb[31]);
                        float amax0_0_62;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_0_62) : "f"(zero), "f"(amag0_46), "f"(amag1_47));
                        float amax0_1_63;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_1_63) : "f"(amag2_48), "f"(amag3_49), "f"(amag4_50));
                        float amax0_2_64;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_2_64) : "f"(amag5_51), "f"(amag6_52), "f"(amag7_53));
                        float amax0_3_65;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_3_65) : "f"(amag8_54), "f"(amag9_55), "f"(amag10_56));
                        float amax0_4_66;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_4_66) : "f"(amag11_57), "f"(amag12_58), "f"(amag13_59));
                        float amax0_5_67;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax0_5_67) : "f"(amag14_60), "f"(amag15_61), "f"(zero));
                        float amax1_0_68;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax1_0_68) : "f"(amax0_0_62), "f"(amax0_1_63), "f"(amax0_2_64));
                        float amax1_1_69;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax1_1_69) : "f"(amax0_3_65), "f"(amax0_4_66), "f"(amax0_5_67));
                        float amax2_0_70;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(amax2_0_70) : "f"(amax1_0_68), "f"(amax1_1_69), "f"(zero));
                        float vm0 = amax2_0;
                        float vm1 = amax2_0_70;
                        if (fhi_1) {
                            vm0 = amax2_0_70;
                            vm1 = amax2_0;
                        }
                        unsigned int sfb[2];
                        float sfv;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfv) : "f"(gso), "f"(mul_f32_rn_noftz(vm0, rcp6)));
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(zero), "f"(sfv));
                        sfb[0] = (unsigned int)_e4m3x2_f32_0 & 255;
                        uint32_t _e4m3x2_to_f16x2_4;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(_e4m3x2_f32_0)));
                        float _cvt_f32_f16_8;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(_e4m3x2_to_f16x2_4)));
                        float _cvt_f32_f16_9;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(_e4m3x2_to_f16x2_4 >> 16)));
                        float _rcp_0 = approx_rcp(mul_f32_rn_noftz(_cvt_f32_f16_8, rcp_gs));
                        float oscale = 0.0f;
                        if (vm0 != 0.0f) {
                            oscale = _rcp_0;
                        }
                        float sfv_71;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfv_71) : "f"(gso), "f"(mul_f32_rn_noftz(vm1, rcp6)));
                        uint16_t _e4m3x2_f32_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(zero), "f"(sfv_71));
                        sfb[1] = (unsigned int)_e4m3x2_f32_1 & 255;
                        uint32_t _e4m3x2_to_f16x2_5;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(_e4m3x2_f32_1)));
                        float _cvt_f32_f16_10;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(_e4m3x2_to_f16x2_5)));
                        float _cvt_f32_f16_11;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(_e4m3x2_to_f16x2_5 >> 16)));
                        float _rcp_1 = approx_rcp(mul_f32_rn_noftz(_cvt_f32_f16_10, rcp_gs));
                        float oscale_72 = 0.0f;
                        if (vm1 != 0.0f) {
                            oscale_72 = _rcp_1;
                        }
                        float os_p0 = oscale;
                        float os_p1 = oscale_72;
                        if (fhi_1) {
                            os_p0 = oscale_72;
                            os_p1 = oscale;
                        }
                        unsigned int w_73[4];
                        uint32_t _fp4_e2m1_pack_pair_34;
                        uint16_t _e2m1_pair_h_35;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_35) : "f"(mul_f32_rn_noftz(yb[0], os_p0)), "f"(mul_f32_rn_noftz(yb[1], os_p0)));
                        _fp4_e2m1_pack_pair_34 = (uint32_t)_e2m1_pair_h_35;
                        uint32_t _fp4_e2m1_pack_pair_36;
                        uint16_t _e2m1_pair_h_37;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(mul_f32_rn_noftz(yb[2], os_p0)), "f"(mul_f32_rn_noftz(yb[3], os_p0)));
                        _fp4_e2m1_pack_pair_36 = (uint32_t)_e2m1_pair_h_37;
                        uint32_t _fp4_e2m1_pack_pair_38;
                        uint16_t _e2m1_pair_h_39;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_39) : "f"(mul_f32_rn_noftz(yb[4], os_p0)), "f"(mul_f32_rn_noftz(yb[5], os_p0)));
                        _fp4_e2m1_pack_pair_38 = (uint32_t)_e2m1_pair_h_39;
                        uint32_t _fp4_e2m1_pack_pair_40;
                        uint16_t _e2m1_pair_h_41;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(mul_f32_rn_noftz(yb[6], os_p0)), "f"(mul_f32_rn_noftz(yb[7], os_p0)));
                        _fp4_e2m1_pack_pair_40 = (uint32_t)_e2m1_pair_h_41;
                        w_73[0] = _fp4_e2m1_pack_pair_34 | _fp4_e2m1_pack_pair_36 << 8 | _fp4_e2m1_pack_pair_38 << 16 | _fp4_e2m1_pack_pair_40 << 24;
                        uint32_t _fp4_e2m1_pack_pair_42;
                        uint16_t _e2m1_pair_h_43;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(mul_f32_rn_noftz(yb[8], os_p0)), "f"(mul_f32_rn_noftz(yb[9], os_p0)));
                        _fp4_e2m1_pack_pair_42 = (uint32_t)_e2m1_pair_h_43;
                        uint32_t _fp4_e2m1_pack_pair_44;
                        uint16_t _e2m1_pair_h_45;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_45) : "f"(mul_f32_rn_noftz(yb[10], os_p0)), "f"(mul_f32_rn_noftz(yb[11], os_p0)));
                        _fp4_e2m1_pack_pair_44 = (uint32_t)_e2m1_pair_h_45;
                        uint32_t _fp4_e2m1_pack_pair_46;
                        uint16_t _e2m1_pair_h_47;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(mul_f32_rn_noftz(yb[12], os_p0)), "f"(mul_f32_rn_noftz(yb[13], os_p0)));
                        _fp4_e2m1_pack_pair_46 = (uint32_t)_e2m1_pair_h_47;
                        uint32_t _fp4_e2m1_pack_pair_48;
                        uint16_t _e2m1_pair_h_49;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_49) : "f"(mul_f32_rn_noftz(yb[14], os_p0)), "f"(mul_f32_rn_noftz(yb[15], os_p0)));
                        _fp4_e2m1_pack_pair_48 = (uint32_t)_e2m1_pair_h_49;
                        w_73[1] = _fp4_e2m1_pack_pair_42 | _fp4_e2m1_pack_pair_44 << 8 | _fp4_e2m1_pack_pair_46 << 16 | _fp4_e2m1_pack_pair_48 << 24;
                        uint32_t _fp4_e2m1_pack_pair_50;
                        uint16_t _e2m1_pair_h_51;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_51) : "f"(mul_f32_rn_noftz(yb[16], os_p1)), "f"(mul_f32_rn_noftz(yb[17], os_p1)));
                        _fp4_e2m1_pack_pair_50 = (uint32_t)_e2m1_pair_h_51;
                        uint32_t _fp4_e2m1_pack_pair_52;
                        uint16_t _e2m1_pair_h_53;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_53) : "f"(mul_f32_rn_noftz(yb[18], os_p1)), "f"(mul_f32_rn_noftz(yb[19], os_p1)));
                        _fp4_e2m1_pack_pair_52 = (uint32_t)_e2m1_pair_h_53;
                        uint32_t _fp4_e2m1_pack_pair_54;
                        uint16_t _e2m1_pair_h_55;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(mul_f32_rn_noftz(yb[20], os_p1)), "f"(mul_f32_rn_noftz(yb[21], os_p1)));
                        _fp4_e2m1_pack_pair_54 = (uint32_t)_e2m1_pair_h_55;
                        uint32_t _fp4_e2m1_pack_pair_56;
                        uint16_t _e2m1_pair_h_57;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(mul_f32_rn_noftz(yb[22], os_p1)), "f"(mul_f32_rn_noftz(yb[23], os_p1)));
                        _fp4_e2m1_pack_pair_56 = (uint32_t)_e2m1_pair_h_57;
                        w_73[2] = _fp4_e2m1_pack_pair_50 | _fp4_e2m1_pack_pair_52 << 8 | _fp4_e2m1_pack_pair_54 << 16 | _fp4_e2m1_pack_pair_56 << 24;
                        uint32_t _fp4_e2m1_pack_pair_58;
                        uint16_t _e2m1_pair_h_59;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(mul_f32_rn_noftz(yb[24], os_p1)), "f"(mul_f32_rn_noftz(yb[25], os_p1)));
                        _fp4_e2m1_pack_pair_58 = (uint32_t)_e2m1_pair_h_59;
                        uint32_t _fp4_e2m1_pack_pair_60;
                        uint16_t _e2m1_pair_h_61;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_61) : "f"(mul_f32_rn_noftz(yb[26], os_p1)), "f"(mul_f32_rn_noftz(yb[27], os_p1)));
                        _fp4_e2m1_pack_pair_60 = (uint32_t)_e2m1_pair_h_61;
                        uint32_t _fp4_e2m1_pack_pair_62;
                        uint16_t _e2m1_pair_h_63;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_63) : "f"(mul_f32_rn_noftz(yb[28], os_p1)), "f"(mul_f32_rn_noftz(yb[29], os_p1)));
                        _fp4_e2m1_pack_pair_62 = (uint32_t)_e2m1_pair_h_63;
                        uint32_t _fp4_e2m1_pack_pair_64;
                        uint16_t _e2m1_pair_h_65;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_65) : "f"(mul_f32_rn_noftz(yb[30], os_p1)), "f"(mul_f32_rn_noftz(yb[31], os_p1)));
                        _fp4_e2m1_pack_pair_64 = (uint32_t)_e2m1_pair_h_65;
                        w_73[3] = _fp4_e2m1_pack_pair_58 | _fp4_e2m1_pack_pair_60 << 8 | _fp4_e2m1_pack_pair_62 << 16 | _fp4_e2m1_pack_pair_64 << 24;
                        unsigned int sw[4];
                        sw[0] = w_73[0];
                        sw[1] = w_73[1];
                        sw[2] = w_73[2];
                        sw[3] = w_73[3];
                        if (fodd_2) {
                            sw[0] = w_73[1];
                            sw[1] = w_73[0];
                            sw[2] = w_73[3];
                            sw[3] = w_73[2];
                        }
                        unsigned int words[4];
                        words[0] = sw[0];
                        words[1] = sw[1];
                        words[2] = sw[2];
                        words[3] = sw[3];
                        if (fhi_1) {
                            words[0] = sw[2];
                            words[1] = sw[3];
                            words[2] = sw[0];
                            words[3] = sw[1];
                        }
                        long long vidx = drow * 3072 + (long long)(16 * pp) >> 2;
                        {
                            int4 _iv4 = make_int4(words[0 + 0], words[0 + 1], words[0 + 2], words[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(outq) + vidx) = _iv4;
                        }
                        unsigned int half = sfb[0] | sfb[1] << 8;
                        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, half, 1);
                        unsigned int oddmask = -(unsigned int)(lane_2 & 1);
                        unsigned int lo_1 = half & ~oddmask | _shfl_xor_0 & oddmask;
                        unsigned int hi_1 = _shfl_xor_0 & ~oddmask | half & oddmask;
                        unsigned int word = lo_1 | hi_1 << 16;
                        long long rowpart = drow / 128 * 49152 + drow % 32 * 16 + drow % 128 / 32 * 4;
                        int bpart = (4 * (pp >> 1) >> 2) * 512 + (4 * (pp >> 1) & 3);
                        long long off = rowpart + (long long)bpart;
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(out_sf + off)) + (0)) = word;
                    }
                    if (n > kd + 1) {
                        __syncwarp();
                        if (lane_2 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            long long nrow = (long long)(r0 + warp_1 + 8 * (kd + 1));
                            mbarrier_arrive_expect_tx(rb_full_addr + (warp_1) * 8, 12288);
                            cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)rb_off, res_in + (nrow * 6144), 12288, rb_full_addr + (warp_1) * 8);
                        }
                    }
                }
            }
            if (lane_2 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 6, 288;" ::: "memory");
            if (tid == 0) {
                __threadfence();
                unsigned int _atomic_old_0 = atomicAdd(&ctrl[1], 1);
                unsigned int last = (unsigned int)gridDim.x - 1;
                if (_atomic_old_0 == last) {
                    ctrl[1] = 0;
                    ctrl[2] = (unsigned int)rows;
                    ctrl[3] = 1;
                    __threadfence();
                    ctrl[0] = call + 1;
                    __threadfence();
                }
            }
        }
    }
    // ---- Role: reset_warp ----
    if (warp == 8) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 0, 288;" ::: "memory");
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
                long long pb64 = (long long)pbuf;
                long long cap64_1 = (long long)cap;
                long long peer64_1 = (long long)(1 - rank);
                unsigned int sv4[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long v_base_1 = pb64 * (2 * cap64_1 * 3456) + peer64_1 * cap64_1 * 3072;
                long long s_base_1 = pb64 * (2 * cap64_1 * 3456) + 2 * cap64_1 * 3072 + peer64_1 * cap64_1 * 384;
                long long m_base_1 = pb64 * (2 * cap64_1 * 16) + peer64_1 * cap64_1 * 16;
                long long u = pr64 * 3072 / 16;
                long long lo_2 = u * bid64_1 / g64_1;
                long long hi_2 = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo_2 + lane64;
                #pragma unroll 1
                for (long long g = g0; g < hi_2; g += 32) {
                    long long widx = v_base_1 + 16 * g >> 2;
                    {
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx) = _iv4;
                    }
                }
                long long u_0 = pr64 * 384 / 16;
                long long lo_1_1 = u_0 * bid64_1 / g64_1;
                long long hi_2_1 = u_0 * (bid64_1 + 1) / g64_1;
                long long g0_3 = lo_1_1 + lane64;
                #pragma unroll 1
                for (long long g_1 = g0_3; g_1 < hi_2_1; g_1 += 32) {
                    long long widx_1 = s_base_1 + 16 * g_1 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx_1) = _iv4;
                    }
                }
                long long u_4 = pr64 * 16 / 16;
                long long lo_5 = u_4 * bid64_1 / g64_1;
                long long hi_6 = u_4 * (bid64_1 + 1) / g64_1;
                long long g0_7 = lo_5 + lane64;
                #pragma unroll 1
                for (long long g_2 = g0_7; g_2 < hi_6; g_2 += 32) {
                    long long widx_2 = m_base_1 + 16 * g_2 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_rs) + widx_2) = _iv4;
                    }
                }
            }
            asm volatile("barrier.sync 6, 288;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
