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

#define CAKE_INF CUDART_INF_F
#define NUM_TMA_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 49152
#define SMEM_SMEM_B_OFF 16384
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 49152
#define SMEM_TOTAL 196704
#define THREADS 384
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



// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}






__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_sm90_bf16_megamoe_fc1_gated_clamp(unsigned int num_experts, unsigned int shape_n, unsigned int shape_k, float clamp_limit, const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap W, long long* __restrict__ offsets, __nv_bfloat16* __restrict__ D)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 196608;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 32)
    #define empty_local_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int smem_b_addr = smem + 16384;
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory");
        }
    }
    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[196608..196704)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // Stage-major initialization across barrier declarations.
            mbarrier_init(smem + 196608, 1);
            mbarrier_init(smem + 196640, 512);
            mbarrier_init(smem + 196672, 256);
            mbarrier_init(smem + 196616, 1);
            mbarrier_init(smem + 196648, 512);
            mbarrier_init(smem + 196680, 256);
            mbarrier_init(smem + 196624, 1);
            mbarrier_init(smem + 196656, 512);
            mbarrier_init(smem + 196688, 256);
            mbarrier_init(smem + 196632, 1);
            mbarrier_init(smem + 196664, 512);
            mbarrier_init(smem + 196696, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Kernel post-init ops
    unsigned int cid = (unsigned int)cluster_id;
    unsigned int ncl = (unsigned int)num_clusters;
    unsigned int rank = (unsigned int)cta_rank;
    float clamp_lo = -clamp_limit;
    unsigned int n_tiles = shape_n / 256;
    unsigned int n_pairs = (n_tiles + 1) / 2;
    unsigned int num_k_stages = shape_k / 64;
    unsigned int n_out = shape_n / 2;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (static_cast<uint32_t>(warp) >= 8u) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        if (warp == 8) {
            if (elect_sync()) {
                unsigned int it_p = 0;
                unsigned int tile_base_p = 0;
                for (unsigned int e_p = 0; e_p < num_experts; e_p++) {
                    unsigned int row0_p = (unsigned int)offsets[e_p];
                    unsigned int row1_p = (unsigned int)offsets[e_p + 1];
                    unsigned int cnt_p = row1_p - row0_p;
                    unsigned int m_tiles_p = (cnt_p + 128 - 1) / 128;
                    unsigned int p_pairs_p = m_tiles_p / 2;
                    unsigned int p_count_p = p_pairs_p * n_tiles;
                    unsigned int tail_p = m_tiles_p - p_pairs_p * 2;
                    unsigned int n_tiles_e_p = p_count_p + tail_p * n_pairs;
                    unsigned int rem_p = (cid + ncl - tile_base_p % ncl) % ncl;
                    unsigned int _max_0 = ((n_tiles_e_p) > (rem_p) ? (n_tiles_e_p) : (rem_p));
                    unsigned int my_tiles_p = (_max_0 - rem_p + ncl - 1) / ncl;
                    for (unsigned int j_p = 0; j_p < my_tiles_p; j_p++) {
                        unsigned int t_p = rem_p + j_p * ncl;
                        unsigned int u_p = t_p - p_count_p;
                        unsigned int m0_lead_p = row0_p + p_pairs_p * 256;
                        unsigned int n0_lead_p = u_p * 2 * 256;
                        unsigned int n0_peer_p = n0_lead_p + 256;
                        if (t_p < p_count_p) {
                            unsigned int mp_p = t_p % p_pairs_p;
                            unsigned int nt_p = t_p / p_pairs_p;
                            m0_lead_p = row0_p + mp_p * 256;
                            n0_lead_p = nt_p * 256;
                            n0_peer_p = nt_p * 256;
                        }
                        for (unsigned int k_p = 0; k_p < num_k_stages; k_p++) {
                            unsigned int stage_p = it_p % 4;
                            mbarrier_wait_relaxed_hint(empty_local_addr + (stage_p) * 8, it_p / 4 + 1 & 1, 10000000);
                            if (cta_rank == 0) {
                                mbarrier_wait_cluster_hint(empty_addr + (stage_p) * 8, it_p / 4 + 1 & 1, 10000000);
                                if (t_p < p_count_p) {
                                    {
                                        const uint64_t _tma_operand_pack_0_desc = (uint64_t)((&A));
                                        const uint32_t _tma_operand_pack_0_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                        const uint32_t _tma_operand_pack_0_dst = (uint32_t)(smem_a_addr + stage_p * 49152);
                                        const int32_t _tma_operand_pack_0_coord_0 = (int32_t)(k_p * 64);
                                        const int32_t _tma_operand_pack_0_coord_1 = (int32_t)(m0_lead_p);
                                        const uint64_t _tma_operand_pack_0_cache = (uint64_t)(0x1000000000000000ULL);
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1, {%3, %4}], [%2], %5;"
                                            :: "r"(_tma_operand_pack_0_dst), "l"(_tma_operand_pack_0_desc), "r"(_tma_operand_pack_0_mbar),
                                               "r"(_tma_operand_pack_0_coord_0), "r"(_tma_operand_pack_0_coord_1), "l"(_tma_operand_pack_0_cache) : "memory");
                                    }
                                    {
                                        const uint64_t _tma_operand_pack_1_desc = (uint64_t)((&A));
                                        const uint32_t _tma_operand_pack_1_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                        const uint32_t _tma_operand_pack_1_dst = (uint32_t)(smem_a_addr + stage_p * 49152);
                                        const int32_t _tma_operand_pack_1_coord_0 = (int32_t)(k_p * 64);
                                        const int32_t _tma_operand_pack_1_coord_1 = (int32_t)(m0_lead_p + 128);
                                        const uint16_t _tma_operand_pack_1_multicast = (uint16_t)(2);
                                        const uint64_t _tma_operand_pack_1_cache = (uint64_t)(0x1000000000000000ULL);
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                            " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                                            :: "r"(_tma_operand_pack_1_dst), "l"(_tma_operand_pack_1_desc), "r"(_tma_operand_pack_1_coord_0), "r"(_tma_operand_pack_1_coord_1),
                                               "r"(_tma_operand_pack_1_mbar), "h"((uint16_t)(_tma_operand_pack_1_multicast)), "l"(_tma_operand_pack_1_cache) : "memory");
                                    }
                                    {
                                        const uint64_t _tma_operand_pack_2_desc = (uint64_t)((&W));
                                        const uint32_t _tma_operand_pack_2_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                        const uint32_t _tma_operand_pack_2_dst = (uint32_t)(smem_b_addr + stage_p * 49152);
                                        const int32_t _tma_operand_pack_2_coord_0 = (int32_t)(k_p * 64);
                                        const int32_t _tma_operand_pack_2_coord_1 = (int32_t)(e_p * shape_n + n0_lead_p);
                                        const uint16_t _tma_operand_pack_2_multicast = (uint16_t)(3);
                                        const uint64_t _tma_operand_pack_2_cache = (uint64_t)(0x1000000000000000ULL);
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                            " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                                            :: "r"(_tma_operand_pack_2_dst), "l"(_tma_operand_pack_2_desc), "r"(_tma_operand_pack_2_coord_0), "r"(_tma_operand_pack_2_coord_1),
                                               "r"(_tma_operand_pack_2_mbar), "h"((uint16_t)(_tma_operand_pack_2_multicast)), "l"(_tma_operand_pack_2_cache) : "memory");
                                    }
                                } else {
                                    {
                                        const uint64_t _tma_operand_pack_3_desc = (uint64_t)((&A));
                                        const uint32_t _tma_operand_pack_3_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                        const uint32_t _tma_operand_pack_3_dst = (uint32_t)(smem_a_addr + stage_p * 49152);
                                        const int32_t _tma_operand_pack_3_coord_0 = (int32_t)(k_p * 64);
                                        const int32_t _tma_operand_pack_3_coord_1 = (int32_t)(m0_lead_p);
                                        const uint16_t _tma_operand_pack_3_multicast = (uint16_t)(3);
                                        const uint64_t _tma_operand_pack_3_cache = (uint64_t)(0x1000000000000000ULL);
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                            " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                                            :: "r"(_tma_operand_pack_3_dst), "l"(_tma_operand_pack_3_desc), "r"(_tma_operand_pack_3_coord_0), "r"(_tma_operand_pack_3_coord_1),
                                               "r"(_tma_operand_pack_3_mbar), "h"((uint16_t)(_tma_operand_pack_3_multicast)), "l"(_tma_operand_pack_3_cache) : "memory");
                                    }
                                    {
                                        const uint64_t _tma_operand_pack_4_desc = (uint64_t)((&W));
                                        const uint32_t _tma_operand_pack_4_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                        const uint32_t _tma_operand_pack_4_dst = (uint32_t)(smem_b_addr + stage_p * 49152);
                                        const int32_t _tma_operand_pack_4_coord_0 = (int32_t)(k_p * 64);
                                        const int32_t _tma_operand_pack_4_coord_1 = (int32_t)(e_p * shape_n + n0_lead_p);
                                        const uint64_t _tma_operand_pack_4_cache = (uint64_t)(0x1000000000000000ULL);
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1, {%3, %4}], [%2], %5;"
                                            :: "r"(_tma_operand_pack_4_dst), "l"(_tma_operand_pack_4_desc), "r"(_tma_operand_pack_4_mbar),
                                               "r"(_tma_operand_pack_4_coord_0), "r"(_tma_operand_pack_4_coord_1), "l"(_tma_operand_pack_4_cache) : "memory");
                                    }
                                }
                            }
                            if (rank == 1) {
                                if (t_p >= p_count_p) {
                                    {
                                        const uint64_t _tma_operand_pack_5_desc = (uint64_t)((&W));
                                        const uint32_t _tma_operand_pack_5_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                        const uint32_t _tma_operand_pack_5_dst = (uint32_t)(smem_b_addr + stage_p * 49152);
                                        const int32_t _tma_operand_pack_5_coord_0 = (int32_t)(k_p * 64);
                                        const int32_t _tma_operand_pack_5_coord_1 = (int32_t)(e_p * shape_n + n0_peer_p);
                                        const uint64_t _tma_operand_pack_5_cache = (uint64_t)(0x1000000000000000ULL);
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1, {%3, %4}], [%2], %5;"
                                            :: "r"(_tma_operand_pack_5_dst), "l"(_tma_operand_pack_5_desc), "r"(_tma_operand_pack_5_mbar),
                                               "r"(_tma_operand_pack_5_coord_0), "r"(_tma_operand_pack_5_coord_1), "l"(_tma_operand_pack_5_cache) : "memory");
                                    }
                                }
                            }
                            mbarrier_arrive_expect_tx(full_addr + (stage_p) * 8, 49152);
                            it_p = it_p + 1;
                        }
                    }
                    tile_base_p = tile_base_p + n_tiles_e_p;
                }
            }
        }
    }
    else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        uint32_t warp_group_idx = static_cast<uint32_t>(tid) / 128u;
        unsigned int math_wg_idx = make_warp_uniform(warp_group_idx);
        float accum[128] = {0};
        unsigned int pk[4] = {0};
        unsigned int it_c = 0;
        unsigned int tile_base_c = 0;
        unsigned int warp_in_wg = (unsigned int)warp % 4;
        unsigned int lane_row = lane / 4;
        unsigned int lane_col = lane % 4 * 2;
        unsigned int q_lane = lane % 4;
        unsigned int m_odd = -(q_lane & 1);
        unsigned int m_even = m_odd ^ 4294967295u;
        unsigned int m_hi = -(q_lane >> 1 & 1);
        unsigned int m_lo = m_hi ^ 4294967295u;
        for (unsigned int e_c = 0; e_c < num_experts; e_c++) {
            unsigned int row0_c = (unsigned int)offsets[e_c];
            unsigned int row1_c = (unsigned int)offsets[e_c + 1];
            unsigned int cnt_c = row1_c - row0_c;
            unsigned int m_tiles_c = (cnt_c + 128 - 1) / 128;
            unsigned int p_pairs_c = m_tiles_c / 2;
            unsigned int p_count_c = p_pairs_c * n_tiles;
            unsigned int tail_c = m_tiles_c - p_pairs_c * 2;
            unsigned int n_tiles_e_c = p_count_c + tail_c * n_pairs;
            unsigned int rem_c = (cid + ncl - tile_base_c % ncl) % ncl;
            unsigned int _max_1 = ((n_tiles_e_c) > (rem_c) ? (n_tiles_e_c) : (rem_c));
            unsigned int my_tiles_c = (_max_1 - rem_c + ncl - 1) / ncl;
            for (unsigned int j_c = 0; j_c < my_tiles_c; j_c++) {
                unsigned int t_c = rem_c + j_c * ncl;
                unsigned int u_c = t_c - p_count_c;
                unsigned int n_sel_c = u_c * 2 + rank;
                unsigned int m0_c = row0_c + p_pairs_c * 256;
                unsigned int n0_c = n_sel_c * 256;
                unsigned int zero_c = 0;
                unsigned int row_lim_c = ((n_sel_c < n_tiles) ? row1_c : zero_c);
                if (t_c < p_count_c) {
                    unsigned int mp_c = t_c % p_pairs_c;
                    unsigned int nt_c = t_c / p_pairs_c;
                    m0_c = row0_c + (mp_c * 2 + rank) * 128;
                    n0_c = nt_c * 256;
                    row_lim_c = row1_c;
                }
                accum[0] = 0.0f;
                accum[1] = 0.0f;
                accum[2] = 0.0f;
                accum[3] = 0.0f;
                accum[4] = 0.0f;
                accum[5] = 0.0f;
                accum[6] = 0.0f;
                accum[7] = 0.0f;
                accum[8] = 0.0f;
                accum[9] = 0.0f;
                accum[10] = 0.0f;
                accum[11] = 0.0f;
                accum[12] = 0.0f;
                accum[13] = 0.0f;
                accum[14] = 0.0f;
                accum[15] = 0.0f;
                accum[16] = 0.0f;
                accum[17] = 0.0f;
                accum[18] = 0.0f;
                accum[19] = 0.0f;
                accum[20] = 0.0f;
                accum[21] = 0.0f;
                accum[22] = 0.0f;
                accum[23] = 0.0f;
                accum[24] = 0.0f;
                accum[25] = 0.0f;
                accum[26] = 0.0f;
                accum[27] = 0.0f;
                accum[28] = 0.0f;
                accum[29] = 0.0f;
                accum[30] = 0.0f;
                accum[31] = 0.0f;
                accum[32] = 0.0f;
                accum[33] = 0.0f;
                accum[34] = 0.0f;
                accum[35] = 0.0f;
                accum[36] = 0.0f;
                accum[37] = 0.0f;
                accum[38] = 0.0f;
                accum[39] = 0.0f;
                accum[40] = 0.0f;
                accum[41] = 0.0f;
                accum[42] = 0.0f;
                accum[43] = 0.0f;
                accum[44] = 0.0f;
                accum[45] = 0.0f;
                accum[46] = 0.0f;
                accum[47] = 0.0f;
                accum[48] = 0.0f;
                accum[49] = 0.0f;
                accum[50] = 0.0f;
                accum[51] = 0.0f;
                accum[52] = 0.0f;
                accum[53] = 0.0f;
                accum[54] = 0.0f;
                accum[55] = 0.0f;
                accum[56] = 0.0f;
                accum[57] = 0.0f;
                accum[58] = 0.0f;
                accum[59] = 0.0f;
                accum[60] = 0.0f;
                accum[61] = 0.0f;
                accum[62] = 0.0f;
                accum[63] = 0.0f;
                accum[64] = 0.0f;
                accum[65] = 0.0f;
                accum[66] = 0.0f;
                accum[67] = 0.0f;
                accum[68] = 0.0f;
                accum[69] = 0.0f;
                accum[70] = 0.0f;
                accum[71] = 0.0f;
                accum[72] = 0.0f;
                accum[73] = 0.0f;
                accum[74] = 0.0f;
                accum[75] = 0.0f;
                accum[76] = 0.0f;
                accum[77] = 0.0f;
                accum[78] = 0.0f;
                accum[79] = 0.0f;
                accum[80] = 0.0f;
                accum[81] = 0.0f;
                accum[82] = 0.0f;
                accum[83] = 0.0f;
                accum[84] = 0.0f;
                accum[85] = 0.0f;
                accum[86] = 0.0f;
                accum[87] = 0.0f;
                accum[88] = 0.0f;
                accum[89] = 0.0f;
                accum[90] = 0.0f;
                accum[91] = 0.0f;
                accum[92] = 0.0f;
                accum[93] = 0.0f;
                accum[94] = 0.0f;
                accum[95] = 0.0f;
                accum[96] = 0.0f;
                accum[97] = 0.0f;
                accum[98] = 0.0f;
                accum[99] = 0.0f;
                accum[100] = 0.0f;
                accum[101] = 0.0f;
                accum[102] = 0.0f;
                accum[103] = 0.0f;
                accum[104] = 0.0f;
                accum[105] = 0.0f;
                accum[106] = 0.0f;
                accum[107] = 0.0f;
                accum[108] = 0.0f;
                accum[109] = 0.0f;
                accum[110] = 0.0f;
                accum[111] = 0.0f;
                accum[112] = 0.0f;
                accum[113] = 0.0f;
                accum[114] = 0.0f;
                accum[115] = 0.0f;
                accum[116] = 0.0f;
                accum[117] = 0.0f;
                accum[118] = 0.0f;
                accum[119] = 0.0f;
                accum[120] = 0.0f;
                accum[121] = 0.0f;
                accum[122] = 0.0f;
                accum[123] = 0.0f;
                accum[124] = 0.0f;
                accum[125] = 0.0f;
                accum[126] = 0.0f;
                accum[127] = 0.0f;
                for (unsigned int k_c = 0; k_c < num_k_stages; k_c++) {
                    unsigned int stage_c = it_c % 4;
                    mbarrier_wait_relaxed_hint(full_addr + (stage_c) * 8, it_c / 4 & 1, 10000000);
                    asm volatile("" : "+f"(accum[0]) :: "memory");
                    asm volatile("" : "+f"(accum[1]) :: "memory");
                    asm volatile("" : "+f"(accum[2]) :: "memory");
                    asm volatile("" : "+f"(accum[3]) :: "memory");
                    asm volatile("" : "+f"(accum[4]) :: "memory");
                    asm volatile("" : "+f"(accum[5]) :: "memory");
                    asm volatile("" : "+f"(accum[6]) :: "memory");
                    asm volatile("" : "+f"(accum[7]) :: "memory");
                    asm volatile("" : "+f"(accum[8]) :: "memory");
                    asm volatile("" : "+f"(accum[9]) :: "memory");
                    asm volatile("" : "+f"(accum[10]) :: "memory");
                    asm volatile("" : "+f"(accum[11]) :: "memory");
                    asm volatile("" : "+f"(accum[12]) :: "memory");
                    asm volatile("" : "+f"(accum[13]) :: "memory");
                    asm volatile("" : "+f"(accum[14]) :: "memory");
                    asm volatile("" : "+f"(accum[15]) :: "memory");
                    asm volatile("" : "+f"(accum[16]) :: "memory");
                    asm volatile("" : "+f"(accum[17]) :: "memory");
                    asm volatile("" : "+f"(accum[18]) :: "memory");
                    asm volatile("" : "+f"(accum[19]) :: "memory");
                    asm volatile("" : "+f"(accum[20]) :: "memory");
                    asm volatile("" : "+f"(accum[21]) :: "memory");
                    asm volatile("" : "+f"(accum[22]) :: "memory");
                    asm volatile("" : "+f"(accum[23]) :: "memory");
                    asm volatile("" : "+f"(accum[24]) :: "memory");
                    asm volatile("" : "+f"(accum[25]) :: "memory");
                    asm volatile("" : "+f"(accum[26]) :: "memory");
                    asm volatile("" : "+f"(accum[27]) :: "memory");
                    asm volatile("" : "+f"(accum[28]) :: "memory");
                    asm volatile("" : "+f"(accum[29]) :: "memory");
                    asm volatile("" : "+f"(accum[30]) :: "memory");
                    asm volatile("" : "+f"(accum[31]) :: "memory");
                    asm volatile("" : "+f"(accum[32]) :: "memory");
                    asm volatile("" : "+f"(accum[33]) :: "memory");
                    asm volatile("" : "+f"(accum[34]) :: "memory");
                    asm volatile("" : "+f"(accum[35]) :: "memory");
                    asm volatile("" : "+f"(accum[36]) :: "memory");
                    asm volatile("" : "+f"(accum[37]) :: "memory");
                    asm volatile("" : "+f"(accum[38]) :: "memory");
                    asm volatile("" : "+f"(accum[39]) :: "memory");
                    asm volatile("" : "+f"(accum[40]) :: "memory");
                    asm volatile("" : "+f"(accum[41]) :: "memory");
                    asm volatile("" : "+f"(accum[42]) :: "memory");
                    asm volatile("" : "+f"(accum[43]) :: "memory");
                    asm volatile("" : "+f"(accum[44]) :: "memory");
                    asm volatile("" : "+f"(accum[45]) :: "memory");
                    asm volatile("" : "+f"(accum[46]) :: "memory");
                    asm volatile("" : "+f"(accum[47]) :: "memory");
                    asm volatile("" : "+f"(accum[48]) :: "memory");
                    asm volatile("" : "+f"(accum[49]) :: "memory");
                    asm volatile("" : "+f"(accum[50]) :: "memory");
                    asm volatile("" : "+f"(accum[51]) :: "memory");
                    asm volatile("" : "+f"(accum[52]) :: "memory");
                    asm volatile("" : "+f"(accum[53]) :: "memory");
                    asm volatile("" : "+f"(accum[54]) :: "memory");
                    asm volatile("" : "+f"(accum[55]) :: "memory");
                    asm volatile("" : "+f"(accum[56]) :: "memory");
                    asm volatile("" : "+f"(accum[57]) :: "memory");
                    asm volatile("" : "+f"(accum[58]) :: "memory");
                    asm volatile("" : "+f"(accum[59]) :: "memory");
                    asm volatile("" : "+f"(accum[60]) :: "memory");
                    asm volatile("" : "+f"(accum[61]) :: "memory");
                    asm volatile("" : "+f"(accum[62]) :: "memory");
                    asm volatile("" : "+f"(accum[63]) :: "memory");
                    asm volatile("" : "+f"(accum[64]) :: "memory");
                    asm volatile("" : "+f"(accum[65]) :: "memory");
                    asm volatile("" : "+f"(accum[66]) :: "memory");
                    asm volatile("" : "+f"(accum[67]) :: "memory");
                    asm volatile("" : "+f"(accum[68]) :: "memory");
                    asm volatile("" : "+f"(accum[69]) :: "memory");
                    asm volatile("" : "+f"(accum[70]) :: "memory");
                    asm volatile("" : "+f"(accum[71]) :: "memory");
                    asm volatile("" : "+f"(accum[72]) :: "memory");
                    asm volatile("" : "+f"(accum[73]) :: "memory");
                    asm volatile("" : "+f"(accum[74]) :: "memory");
                    asm volatile("" : "+f"(accum[75]) :: "memory");
                    asm volatile("" : "+f"(accum[76]) :: "memory");
                    asm volatile("" : "+f"(accum[77]) :: "memory");
                    asm volatile("" : "+f"(accum[78]) :: "memory");
                    asm volatile("" : "+f"(accum[79]) :: "memory");
                    asm volatile("" : "+f"(accum[80]) :: "memory");
                    asm volatile("" : "+f"(accum[81]) :: "memory");
                    asm volatile("" : "+f"(accum[82]) :: "memory");
                    asm volatile("" : "+f"(accum[83]) :: "memory");
                    asm volatile("" : "+f"(accum[84]) :: "memory");
                    asm volatile("" : "+f"(accum[85]) :: "memory");
                    asm volatile("" : "+f"(accum[86]) :: "memory");
                    asm volatile("" : "+f"(accum[87]) :: "memory");
                    asm volatile("" : "+f"(accum[88]) :: "memory");
                    asm volatile("" : "+f"(accum[89]) :: "memory");
                    asm volatile("" : "+f"(accum[90]) :: "memory");
                    asm volatile("" : "+f"(accum[91]) :: "memory");
                    asm volatile("" : "+f"(accum[92]) :: "memory");
                    asm volatile("" : "+f"(accum[93]) :: "memory");
                    asm volatile("" : "+f"(accum[94]) :: "memory");
                    asm volatile("" : "+f"(accum[95]) :: "memory");
                    asm volatile("" : "+f"(accum[96]) :: "memory");
                    asm volatile("" : "+f"(accum[97]) :: "memory");
                    asm volatile("" : "+f"(accum[98]) :: "memory");
                    asm volatile("" : "+f"(accum[99]) :: "memory");
                    asm volatile("" : "+f"(accum[100]) :: "memory");
                    asm volatile("" : "+f"(accum[101]) :: "memory");
                    asm volatile("" : "+f"(accum[102]) :: "memory");
                    asm volatile("" : "+f"(accum[103]) :: "memory");
                    asm volatile("" : "+f"(accum[104]) :: "memory");
                    asm volatile("" : "+f"(accum[105]) :: "memory");
                    asm volatile("" : "+f"(accum[106]) :: "memory");
                    asm volatile("" : "+f"(accum[107]) :: "memory");
                    asm volatile("" : "+f"(accum[108]) :: "memory");
                    asm volatile("" : "+f"(accum[109]) :: "memory");
                    asm volatile("" : "+f"(accum[110]) :: "memory");
                    asm volatile("" : "+f"(accum[111]) :: "memory");
                    asm volatile("" : "+f"(accum[112]) :: "memory");
                    asm volatile("" : "+f"(accum[113]) :: "memory");
                    asm volatile("" : "+f"(accum[114]) :: "memory");
                    asm volatile("" : "+f"(accum[115]) :: "memory");
                    asm volatile("" : "+f"(accum[116]) :: "memory");
                    asm volatile("" : "+f"(accum[117]) :: "memory");
                    asm volatile("" : "+f"(accum[118]) :: "memory");
                    asm volatile("" : "+f"(accum[119]) :: "memory");
                    asm volatile("" : "+f"(accum[120]) :: "memory");
                    asm volatile("" : "+f"(accum[121]) :: "memory");
                    asm volatile("" : "+f"(accum[122]) :: "memory");
                    asm volatile("" : "+f"(accum[123]) :: "memory");
                    asm volatile("" : "+f"(accum[124]) :: "memory");
                    asm volatile("" : "+f"(accum[125]) :: "memory");
                    asm volatile("" : "+f"(accum[126]) :: "memory");
                    asm volatile("" : "+f"(accum[127]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint32_t _wgmma_stage_base_0 = (unsigned int)smem + (unsigned int)stage_c * 49152;
                    {
                        uint64_t _wgmma_a_0_0 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(0) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_1 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(0) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95]), "+f"(accum[96]), "+f"(accum[97]), "+f"(accum[98]), "+f"(accum[99]), "+f"(accum[100]), "+f"(accum[101]), "+f"(accum[102]), "+f"(accum[103]), "+f"(accum[104]), "+f"(accum[105]), "+f"(accum[106]), "+f"(accum[107]), "+f"(accum[108]), "+f"(accum[109]), "+f"(accum[110]), "+f"(accum[111]), "+f"(accum[112]), "+f"(accum[113]), "+f"(accum[114]), "+f"(accum[115]), "+f"(accum[116]), "+f"(accum[117]), "+f"(accum[118]), "+f"(accum[119]), "+f"(accum[120]), "+f"(accum[121]), "+f"(accum[122]), "+f"(accum[123]), "+f"(accum[124]), "+f"(accum[125]), "+f"(accum[126]), "+f"(accum[127])
                            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
                            : "memory");
                    }
                    {
                        uint64_t _wgmma_a_0_2 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(16) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_3 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(16) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95]), "+f"(accum[96]), "+f"(accum[97]), "+f"(accum[98]), "+f"(accum[99]), "+f"(accum[100]), "+f"(accum[101]), "+f"(accum[102]), "+f"(accum[103]), "+f"(accum[104]), "+f"(accum[105]), "+f"(accum[106]), "+f"(accum[107]), "+f"(accum[108]), "+f"(accum[109]), "+f"(accum[110]), "+f"(accum[111]), "+f"(accum[112]), "+f"(accum[113]), "+f"(accum[114]), "+f"(accum[115]), "+f"(accum[116]), "+f"(accum[117]), "+f"(accum[118]), "+f"(accum[119]), "+f"(accum[120]), "+f"(accum[121]), "+f"(accum[122]), "+f"(accum[123]), "+f"(accum[124]), "+f"(accum[125]), "+f"(accum[126]), "+f"(accum[127])
                            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_3)
                            : "memory");
                    }
                    {
                        uint64_t _wgmma_a_0_4 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(32) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_5 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(32) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95]), "+f"(accum[96]), "+f"(accum[97]), "+f"(accum[98]), "+f"(accum[99]), "+f"(accum[100]), "+f"(accum[101]), "+f"(accum[102]), "+f"(accum[103]), "+f"(accum[104]), "+f"(accum[105]), "+f"(accum[106]), "+f"(accum[107]), "+f"(accum[108]), "+f"(accum[109]), "+f"(accum[110]), "+f"(accum[111]), "+f"(accum[112]), "+f"(accum[113]), "+f"(accum[114]), "+f"(accum[115]), "+f"(accum[116]), "+f"(accum[117]), "+f"(accum[118]), "+f"(accum[119]), "+f"(accum[120]), "+f"(accum[121]), "+f"(accum[122]), "+f"(accum[123]), "+f"(accum[124]), "+f"(accum[125]), "+f"(accum[126]), "+f"(accum[127])
                            : "l"(_wgmma_a_0_4), "l"(_wgmma_b_0_5)
                            : "memory");
                    }
                    {
                        uint64_t _wgmma_a_0_6 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(48) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_7 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(48) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95]), "+f"(accum[96]), "+f"(accum[97]), "+f"(accum[98]), "+f"(accum[99]), "+f"(accum[100]), "+f"(accum[101]), "+f"(accum[102]), "+f"(accum[103]), "+f"(accum[104]), "+f"(accum[105]), "+f"(accum[106]), "+f"(accum[107]), "+f"(accum[108]), "+f"(accum[109]), "+f"(accum[110]), "+f"(accum[111]), "+f"(accum[112]), "+f"(accum[113]), "+f"(accum[114]), "+f"(accum[115]), "+f"(accum[116]), "+f"(accum[117]), "+f"(accum[118]), "+f"(accum[119]), "+f"(accum[120]), "+f"(accum[121]), "+f"(accum[122]), "+f"(accum[123]), "+f"(accum[124]), "+f"(accum[125]), "+f"(accum[126]), "+f"(accum[127])
                            : "l"(_wgmma_a_0_6), "l"(_wgmma_b_0_7)
                            : "memory");
                    }
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("" : "+f"(accum[0]) :: "memory");
                    asm volatile("" : "+f"(accum[1]) :: "memory");
                    asm volatile("" : "+f"(accum[2]) :: "memory");
                    asm volatile("" : "+f"(accum[3]) :: "memory");
                    asm volatile("" : "+f"(accum[4]) :: "memory");
                    asm volatile("" : "+f"(accum[5]) :: "memory");
                    asm volatile("" : "+f"(accum[6]) :: "memory");
                    asm volatile("" : "+f"(accum[7]) :: "memory");
                    asm volatile("" : "+f"(accum[8]) :: "memory");
                    asm volatile("" : "+f"(accum[9]) :: "memory");
                    asm volatile("" : "+f"(accum[10]) :: "memory");
                    asm volatile("" : "+f"(accum[11]) :: "memory");
                    asm volatile("" : "+f"(accum[12]) :: "memory");
                    asm volatile("" : "+f"(accum[13]) :: "memory");
                    asm volatile("" : "+f"(accum[14]) :: "memory");
                    asm volatile("" : "+f"(accum[15]) :: "memory");
                    asm volatile("" : "+f"(accum[16]) :: "memory");
                    asm volatile("" : "+f"(accum[17]) :: "memory");
                    asm volatile("" : "+f"(accum[18]) :: "memory");
                    asm volatile("" : "+f"(accum[19]) :: "memory");
                    asm volatile("" : "+f"(accum[20]) :: "memory");
                    asm volatile("" : "+f"(accum[21]) :: "memory");
                    asm volatile("" : "+f"(accum[22]) :: "memory");
                    asm volatile("" : "+f"(accum[23]) :: "memory");
                    asm volatile("" : "+f"(accum[24]) :: "memory");
                    asm volatile("" : "+f"(accum[25]) :: "memory");
                    asm volatile("" : "+f"(accum[26]) :: "memory");
                    asm volatile("" : "+f"(accum[27]) :: "memory");
                    asm volatile("" : "+f"(accum[28]) :: "memory");
                    asm volatile("" : "+f"(accum[29]) :: "memory");
                    asm volatile("" : "+f"(accum[30]) :: "memory");
                    asm volatile("" : "+f"(accum[31]) :: "memory");
                    asm volatile("" : "+f"(accum[32]) :: "memory");
                    asm volatile("" : "+f"(accum[33]) :: "memory");
                    asm volatile("" : "+f"(accum[34]) :: "memory");
                    asm volatile("" : "+f"(accum[35]) :: "memory");
                    asm volatile("" : "+f"(accum[36]) :: "memory");
                    asm volatile("" : "+f"(accum[37]) :: "memory");
                    asm volatile("" : "+f"(accum[38]) :: "memory");
                    asm volatile("" : "+f"(accum[39]) :: "memory");
                    asm volatile("" : "+f"(accum[40]) :: "memory");
                    asm volatile("" : "+f"(accum[41]) :: "memory");
                    asm volatile("" : "+f"(accum[42]) :: "memory");
                    asm volatile("" : "+f"(accum[43]) :: "memory");
                    asm volatile("" : "+f"(accum[44]) :: "memory");
                    asm volatile("" : "+f"(accum[45]) :: "memory");
                    asm volatile("" : "+f"(accum[46]) :: "memory");
                    asm volatile("" : "+f"(accum[47]) :: "memory");
                    asm volatile("" : "+f"(accum[48]) :: "memory");
                    asm volatile("" : "+f"(accum[49]) :: "memory");
                    asm volatile("" : "+f"(accum[50]) :: "memory");
                    asm volatile("" : "+f"(accum[51]) :: "memory");
                    asm volatile("" : "+f"(accum[52]) :: "memory");
                    asm volatile("" : "+f"(accum[53]) :: "memory");
                    asm volatile("" : "+f"(accum[54]) :: "memory");
                    asm volatile("" : "+f"(accum[55]) :: "memory");
                    asm volatile("" : "+f"(accum[56]) :: "memory");
                    asm volatile("" : "+f"(accum[57]) :: "memory");
                    asm volatile("" : "+f"(accum[58]) :: "memory");
                    asm volatile("" : "+f"(accum[59]) :: "memory");
                    asm volatile("" : "+f"(accum[60]) :: "memory");
                    asm volatile("" : "+f"(accum[61]) :: "memory");
                    asm volatile("" : "+f"(accum[62]) :: "memory");
                    asm volatile("" : "+f"(accum[63]) :: "memory");
                    asm volatile("" : "+f"(accum[64]) :: "memory");
                    asm volatile("" : "+f"(accum[65]) :: "memory");
                    asm volatile("" : "+f"(accum[66]) :: "memory");
                    asm volatile("" : "+f"(accum[67]) :: "memory");
                    asm volatile("" : "+f"(accum[68]) :: "memory");
                    asm volatile("" : "+f"(accum[69]) :: "memory");
                    asm volatile("" : "+f"(accum[70]) :: "memory");
                    asm volatile("" : "+f"(accum[71]) :: "memory");
                    asm volatile("" : "+f"(accum[72]) :: "memory");
                    asm volatile("" : "+f"(accum[73]) :: "memory");
                    asm volatile("" : "+f"(accum[74]) :: "memory");
                    asm volatile("" : "+f"(accum[75]) :: "memory");
                    asm volatile("" : "+f"(accum[76]) :: "memory");
                    asm volatile("" : "+f"(accum[77]) :: "memory");
                    asm volatile("" : "+f"(accum[78]) :: "memory");
                    asm volatile("" : "+f"(accum[79]) :: "memory");
                    asm volatile("" : "+f"(accum[80]) :: "memory");
                    asm volatile("" : "+f"(accum[81]) :: "memory");
                    asm volatile("" : "+f"(accum[82]) :: "memory");
                    asm volatile("" : "+f"(accum[83]) :: "memory");
                    asm volatile("" : "+f"(accum[84]) :: "memory");
                    asm volatile("" : "+f"(accum[85]) :: "memory");
                    asm volatile("" : "+f"(accum[86]) :: "memory");
                    asm volatile("" : "+f"(accum[87]) :: "memory");
                    asm volatile("" : "+f"(accum[88]) :: "memory");
                    asm volatile("" : "+f"(accum[89]) :: "memory");
                    asm volatile("" : "+f"(accum[90]) :: "memory");
                    asm volatile("" : "+f"(accum[91]) :: "memory");
                    asm volatile("" : "+f"(accum[92]) :: "memory");
                    asm volatile("" : "+f"(accum[93]) :: "memory");
                    asm volatile("" : "+f"(accum[94]) :: "memory");
                    asm volatile("" : "+f"(accum[95]) :: "memory");
                    asm volatile("" : "+f"(accum[96]) :: "memory");
                    asm volatile("" : "+f"(accum[97]) :: "memory");
                    asm volatile("" : "+f"(accum[98]) :: "memory");
                    asm volatile("" : "+f"(accum[99]) :: "memory");
                    asm volatile("" : "+f"(accum[100]) :: "memory");
                    asm volatile("" : "+f"(accum[101]) :: "memory");
                    asm volatile("" : "+f"(accum[102]) :: "memory");
                    asm volatile("" : "+f"(accum[103]) :: "memory");
                    asm volatile("" : "+f"(accum[104]) :: "memory");
                    asm volatile("" : "+f"(accum[105]) :: "memory");
                    asm volatile("" : "+f"(accum[106]) :: "memory");
                    asm volatile("" : "+f"(accum[107]) :: "memory");
                    asm volatile("" : "+f"(accum[108]) :: "memory");
                    asm volatile("" : "+f"(accum[109]) :: "memory");
                    asm volatile("" : "+f"(accum[110]) :: "memory");
                    asm volatile("" : "+f"(accum[111]) :: "memory");
                    asm volatile("" : "+f"(accum[112]) :: "memory");
                    asm volatile("" : "+f"(accum[113]) :: "memory");
                    asm volatile("" : "+f"(accum[114]) :: "memory");
                    asm volatile("" : "+f"(accum[115]) :: "memory");
                    asm volatile("" : "+f"(accum[116]) :: "memory");
                    asm volatile("" : "+f"(accum[117]) :: "memory");
                    asm volatile("" : "+f"(accum[118]) :: "memory");
                    asm volatile("" : "+f"(accum[119]) :: "memory");
                    asm volatile("" : "+f"(accum[120]) :: "memory");
                    asm volatile("" : "+f"(accum[121]) :: "memory");
                    asm volatile("" : "+f"(accum[122]) :: "memory");
                    asm volatile("" : "+f"(accum[123]) :: "memory");
                    asm volatile("" : "+f"(accum[124]) :: "memory");
                    asm volatile("" : "+f"(accum[125]) :: "memory");
                    asm volatile("" : "+f"(accum[126]) :: "memory");
                    asm volatile("" : "+f"(accum[127]) :: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    mbarrier_arrive(empty_local_addr + (stage_c) * 8);
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((empty_addr + (stage_c) * 8) & 0xFEFFFFFF) : "memory");
                    it_c = it_c + 1;
                }
                unsigned int row_lo = m0_c + math_wg_idx * 64 + warp_in_wg * 16 + lane_row;
                unsigned int row_hi = row_lo + 8;
                unsigned int col_v = n0_c / 2 + q_lane * 8;
                float _max_2 = max_noftz(accum[0], clamp_lo);
                float _min_0 = fminf(_max_2, clamp_limit);
                float _max_3 = max_noftz(accum[1], clamp_lo);
                float _min_1 = fminf(_max_3, clamp_limit);
                float _max_4 = max_noftz(accum[16], clamp_lo);
                float _min_2 = fminf(_max_4, clamp_limit);
                float _max_5 = max_noftz(accum[17], clamp_lo);
                float _min_3 = fminf(_max_5, clamp_limit);
                float _exp2_0 = approx_exp2(_min_0 * -1.4426950408889634f);
                float _rcp_0 = approx_rcp(_exp2_0 + 1.0f);
                float _exp2_1 = approx_exp2(_min_1 * -1.4426950408889634f);
                float _rcp_1 = approx_rcp(_exp2_1 + 1.0f);
                __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_min_0 * _rcp_0 * _min_2, _min_1 * _rcp_1 * _min_3));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_0)[0];
                float _max_6 = max_noftz(accum[4], clamp_lo);
                float _min_4 = fminf(_max_6, clamp_limit);
                float _max_7 = max_noftz(accum[5], clamp_lo);
                float _min_5 = fminf(_max_7, clamp_limit);
                float _max_8 = max_noftz(accum[20], clamp_lo);
                float _min_6 = fminf(_max_8, clamp_limit);
                float _max_9 = max_noftz(accum[21], clamp_lo);
                float _min_7 = fminf(_max_9, clamp_limit);
                float _exp2_2 = approx_exp2(_min_4 * -1.4426950408889634f);
                float _rcp_2 = approx_rcp(_exp2_2 + 1.0f);
                float _exp2_3 = approx_exp2(_min_5 * -1.4426950408889634f);
                float _rcp_3 = approx_rcp(_exp2_3 + 1.0f);
                __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_min_4 * _rcp_2 * _min_6, _min_5 * _rcp_3 * _min_7));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_1)[0];
                float _max_10 = max_noftz(accum[8], clamp_lo);
                float _min_8 = fminf(_max_10, clamp_limit);
                float _max_11 = max_noftz(accum[9], clamp_lo);
                float _min_9 = fminf(_max_11, clamp_limit);
                float _max_12 = max_noftz(accum[24], clamp_lo);
                float _min_10 = fminf(_max_12, clamp_limit);
                float _max_13 = max_noftz(accum[25], clamp_lo);
                float _min_11 = fminf(_max_13, clamp_limit);
                float _exp2_4 = approx_exp2(_min_8 * -1.4426950408889634f);
                float _rcp_4 = approx_rcp(_exp2_4 + 1.0f);
                float _exp2_5 = approx_exp2(_min_9 * -1.4426950408889634f);
                float _rcp_5 = approx_rcp(_exp2_5 + 1.0f);
                __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_min_8 * _rcp_4 * _min_10, _min_9 * _rcp_5 * _min_11));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_2)[0];
                float _max_14 = max_noftz(accum[12], clamp_lo);
                float _min_12 = fminf(_max_14, clamp_limit);
                float _max_15 = max_noftz(accum[13], clamp_lo);
                float _min_13 = fminf(_max_15, clamp_limit);
                float _max_16 = max_noftz(accum[28], clamp_lo);
                float _min_14 = fminf(_max_16, clamp_limit);
                float _max_17 = max_noftz(accum[29], clamp_lo);
                float _min_15 = fminf(_max_17, clamp_limit);
                float _exp2_6 = approx_exp2(_min_12 * -1.4426950408889634f);
                float _rcp_6 = approx_rcp(_exp2_6 + 1.0f);
                float _exp2_7 = approx_exp2(_min_13 * -1.4426950408889634f);
                float _rcp_7 = approx_rcp(_exp2_7 + 1.0f);
                __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_min_12 * _rcp_6 * _min_14, _min_13 * _rcp_7 * _min_15));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_3)[0];
                unsigned int x0 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, x0, 1);
                unsigned int y0 = _shfl_xor_0;
                unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, x1, 1);
                unsigned int y1 = _shfl_xor_1;
                pk[0] = y0 & m_odd | pk[0] & m_even;
                pk[1] = y0 & m_even | pk[1] & m_odd;
                pk[2] = y1 & m_odd | pk[2] & m_even;
                pk[3] = y1 & m_even | pk[3] & m_odd;
                unsigned int x2 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, x2, 2);
                unsigned int y2 = _shfl_xor_2;
                unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, x3, 2);
                unsigned int y3 = _shfl_xor_3;
                pk[0] = y2 & m_hi | pk[0] & m_lo;
                pk[2] = y2 & m_lo | pk[2] & m_hi;
                pk[1] = y3 & m_hi | pk[1] & m_lo;
                pk[3] = y3 & m_lo | pk[3] & m_hi;
                if (row_lo < row_lim_c) {
                    reinterpret_cast<int4*>(D + (row_lo * n_out + col_v))[0] = reinterpret_cast<int4*>(pk)[0];
                }
                float _max_18 = max_noftz(accum[2], clamp_lo);
                float _min_16 = fminf(_max_18, clamp_limit);
                float _max_19 = max_noftz(accum[3], clamp_lo);
                float _min_17 = fminf(_max_19, clamp_limit);
                float _max_20 = max_noftz(accum[18], clamp_lo);
                float _min_18 = fminf(_max_20, clamp_limit);
                float _max_21 = max_noftz(accum[19], clamp_lo);
                float _min_19 = fminf(_max_21, clamp_limit);
                float _exp2_8 = approx_exp2(_min_16 * -1.4426950408889634f);
                float _rcp_8 = approx_rcp(_exp2_8 + 1.0f);
                float _exp2_9 = approx_exp2(_min_17 * -1.4426950408889634f);
                float _rcp_9 = approx_rcp(_exp2_9 + 1.0f);
                __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_min_16 * _rcp_8 * _min_18, _min_17 * _rcp_9 * _min_19));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_4)[0];
                float _max_22 = max_noftz(accum[6], clamp_lo);
                float _min_20 = fminf(_max_22, clamp_limit);
                float _max_23 = max_noftz(accum[7], clamp_lo);
                float _min_21 = fminf(_max_23, clamp_limit);
                float _max_24 = max_noftz(accum[22], clamp_lo);
                float _min_22 = fminf(_max_24, clamp_limit);
                float _max_25 = max_noftz(accum[23], clamp_lo);
                float _min_23 = fminf(_max_25, clamp_limit);
                float _exp2_10 = approx_exp2(_min_20 * -1.4426950408889634f);
                float _rcp_10 = approx_rcp(_exp2_10 + 1.0f);
                float _exp2_11 = approx_exp2(_min_21 * -1.4426950408889634f);
                float _rcp_11 = approx_rcp(_exp2_11 + 1.0f);
                __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_min_20 * _rcp_10 * _min_22, _min_21 * _rcp_11 * _min_23));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_5)[0];
                float _max_26 = max_noftz(accum[10], clamp_lo);
                float _min_24 = fminf(_max_26, clamp_limit);
                float _max_27 = max_noftz(accum[11], clamp_lo);
                float _min_25 = fminf(_max_27, clamp_limit);
                float _max_28 = max_noftz(accum[26], clamp_lo);
                float _min_26 = fminf(_max_28, clamp_limit);
                float _max_29 = max_noftz(accum[27], clamp_lo);
                float _min_27 = fminf(_max_29, clamp_limit);
                float _exp2_12 = approx_exp2(_min_24 * -1.4426950408889634f);
                float _rcp_12 = approx_rcp(_exp2_12 + 1.0f);
                float _exp2_13 = approx_exp2(_min_25 * -1.4426950408889634f);
                float _rcp_13 = approx_rcp(_exp2_13 + 1.0f);
                __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_min_24 * _rcp_12 * _min_26, _min_25 * _rcp_13 * _min_27));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_6)[0];
                float _max_30 = max_noftz(accum[14], clamp_lo);
                float _min_28 = fminf(_max_30, clamp_limit);
                float _max_31 = max_noftz(accum[15], clamp_lo);
                float _min_29 = fminf(_max_31, clamp_limit);
                float _max_32 = max_noftz(accum[30], clamp_lo);
                float _min_30 = fminf(_max_32, clamp_limit);
                float _max_33 = max_noftz(accum[31], clamp_lo);
                float _min_31 = fminf(_max_33, clamp_limit);
                float _exp2_14 = approx_exp2(_min_28 * -1.4426950408889634f);
                float _rcp_14 = approx_rcp(_exp2_14 + 1.0f);
                float _exp2_15 = approx_exp2(_min_29 * -1.4426950408889634f);
                float _rcp_15 = approx_rcp(_exp2_15 + 1.0f);
                __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_min_28 * _rcp_14 * _min_30, _min_29 * _rcp_15 * _min_31));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_7)[0];
                unsigned int x0_0 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_1 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, x0_0, 1);
                unsigned int y0_2 = _shfl_xor_4;
                unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, x1_1, 1);
                unsigned int y1_3 = _shfl_xor_5;
                pk[0] = y0_2 & m_odd | pk[0] & m_even;
                pk[1] = y0_2 & m_even | pk[1] & m_odd;
                pk[2] = y1_3 & m_odd | pk[2] & m_even;
                pk[3] = y1_3 & m_even | pk[3] & m_odd;
                unsigned int x2_4 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_5 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, x2_4, 2);
                unsigned int y2_6 = _shfl_xor_6;
                unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, x3_5, 2);
                unsigned int y3_7 = _shfl_xor_7;
                pk[0] = y2_6 & m_hi | pk[0] & m_lo;
                pk[2] = y2_6 & m_lo | pk[2] & m_hi;
                pk[1] = y3_7 & m_hi | pk[1] & m_lo;
                pk[3] = y3_7 & m_lo | pk[3] & m_hi;
                if (row_hi < row_lim_c) {
                    reinterpret_cast<int4*>(D + (row_hi * n_out + col_v))[0] = reinterpret_cast<int4*>(pk)[0];
                }
                unsigned int col_v_8 = n0_c / 2 + (4 + q_lane) * 8;
                float _max_34 = max_noftz(accum[32], clamp_lo);
                float _min_32 = fminf(_max_34, clamp_limit);
                float _max_35 = max_noftz(accum[33], clamp_lo);
                float _min_33 = fminf(_max_35, clamp_limit);
                float _max_36 = max_noftz(accum[48], clamp_lo);
                float _min_34 = fminf(_max_36, clamp_limit);
                float _max_37 = max_noftz(accum[49], clamp_lo);
                float _min_35 = fminf(_max_37, clamp_limit);
                float _exp2_16 = approx_exp2(_min_32 * -1.4426950408889634f);
                float _rcp_16 = approx_rcp(_exp2_16 + 1.0f);
                float _exp2_17 = approx_exp2(_min_33 * -1.4426950408889634f);
                float _rcp_17 = approx_rcp(_exp2_17 + 1.0f);
                __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_min_32 * _rcp_16 * _min_34, _min_33 * _rcp_17 * _min_35));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_8)[0];
                float _max_38 = max_noftz(accum[36], clamp_lo);
                float _min_36 = fminf(_max_38, clamp_limit);
                float _max_39 = max_noftz(accum[37], clamp_lo);
                float _min_37 = fminf(_max_39, clamp_limit);
                float _max_40 = max_noftz(accum[52], clamp_lo);
                float _min_38 = fminf(_max_40, clamp_limit);
                float _max_41 = max_noftz(accum[53], clamp_lo);
                float _min_39 = fminf(_max_41, clamp_limit);
                float _exp2_18 = approx_exp2(_min_36 * -1.4426950408889634f);
                float _rcp_18 = approx_rcp(_exp2_18 + 1.0f);
                float _exp2_19 = approx_exp2(_min_37 * -1.4426950408889634f);
                float _rcp_19 = approx_rcp(_exp2_19 + 1.0f);
                __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_min_36 * _rcp_18 * _min_38, _min_37 * _rcp_19 * _min_39));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_9)[0];
                float _max_42 = max_noftz(accum[40], clamp_lo);
                float _min_40 = fminf(_max_42, clamp_limit);
                float _max_43 = max_noftz(accum[41], clamp_lo);
                float _min_41 = fminf(_max_43, clamp_limit);
                float _max_44 = max_noftz(accum[56], clamp_lo);
                float _min_42 = fminf(_max_44, clamp_limit);
                float _max_45 = max_noftz(accum[57], clamp_lo);
                float _min_43 = fminf(_max_45, clamp_limit);
                float _exp2_20 = approx_exp2(_min_40 * -1.4426950408889634f);
                float _rcp_20 = approx_rcp(_exp2_20 + 1.0f);
                float _exp2_21 = approx_exp2(_min_41 * -1.4426950408889634f);
                float _rcp_21 = approx_rcp(_exp2_21 + 1.0f);
                __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_min_40 * _rcp_20 * _min_42, _min_41 * _rcp_21 * _min_43));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_10)[0];
                float _max_46 = max_noftz(accum[44], clamp_lo);
                float _min_44 = fminf(_max_46, clamp_limit);
                float _max_47 = max_noftz(accum[45], clamp_lo);
                float _min_45 = fminf(_max_47, clamp_limit);
                float _max_48 = max_noftz(accum[60], clamp_lo);
                float _min_46 = fminf(_max_48, clamp_limit);
                float _max_49 = max_noftz(accum[61], clamp_lo);
                float _min_47 = fminf(_max_49, clamp_limit);
                float _exp2_22 = approx_exp2(_min_44 * -1.4426950408889634f);
                float _rcp_22 = approx_rcp(_exp2_22 + 1.0f);
                float _exp2_23 = approx_exp2(_min_45 * -1.4426950408889634f);
                float _rcp_23 = approx_rcp(_exp2_23 + 1.0f);
                __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_min_44 * _rcp_22 * _min_46, _min_45 * _rcp_23 * _min_47));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_11)[0];
                unsigned int x0_9 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_10 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, x0_9, 1);
                unsigned int y0_11 = _shfl_xor_8;
                unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, x1_10, 1);
                unsigned int y1_12 = _shfl_xor_9;
                pk[0] = y0_11 & m_odd | pk[0] & m_even;
                pk[1] = y0_11 & m_even | pk[1] & m_odd;
                pk[2] = y1_12 & m_odd | pk[2] & m_even;
                pk[3] = y1_12 & m_even | pk[3] & m_odd;
                unsigned int x2_13 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_14 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, x2_13, 2);
                unsigned int y2_15 = _shfl_xor_10;
                unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, x3_14, 2);
                unsigned int y3_16 = _shfl_xor_11;
                pk[0] = y2_15 & m_hi | pk[0] & m_lo;
                pk[2] = y2_15 & m_lo | pk[2] & m_hi;
                pk[1] = y3_16 & m_hi | pk[1] & m_lo;
                pk[3] = y3_16 & m_lo | pk[3] & m_hi;
                if (row_lo < row_lim_c) {
                    reinterpret_cast<int4*>(D + (row_lo * n_out + col_v_8))[0] = reinterpret_cast<int4*>(pk)[0];
                }
                float _max_50 = max_noftz(accum[34], clamp_lo);
                float _min_48 = fminf(_max_50, clamp_limit);
                float _max_51 = max_noftz(accum[35], clamp_lo);
                float _min_49 = fminf(_max_51, clamp_limit);
                float _max_52 = max_noftz(accum[50], clamp_lo);
                float _min_50 = fminf(_max_52, clamp_limit);
                float _max_53 = max_noftz(accum[51], clamp_lo);
                float _min_51 = fminf(_max_53, clamp_limit);
                float _exp2_24 = approx_exp2(_min_48 * -1.4426950408889634f);
                float _rcp_24 = approx_rcp(_exp2_24 + 1.0f);
                float _exp2_25 = approx_exp2(_min_49 * -1.4426950408889634f);
                float _rcp_25 = approx_rcp(_exp2_25 + 1.0f);
                __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_min_48 * _rcp_24 * _min_50, _min_49 * _rcp_25 * _min_51));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_12)[0];
                float _max_54 = max_noftz(accum[38], clamp_lo);
                float _min_52 = fminf(_max_54, clamp_limit);
                float _max_55 = max_noftz(accum[39], clamp_lo);
                float _min_53 = fminf(_max_55, clamp_limit);
                float _max_56 = max_noftz(accum[54], clamp_lo);
                float _min_54 = fminf(_max_56, clamp_limit);
                float _max_57 = max_noftz(accum[55], clamp_lo);
                float _min_55 = fminf(_max_57, clamp_limit);
                float _exp2_26 = approx_exp2(_min_52 * -1.4426950408889634f);
                float _rcp_26 = approx_rcp(_exp2_26 + 1.0f);
                float _exp2_27 = approx_exp2(_min_53 * -1.4426950408889634f);
                float _rcp_27 = approx_rcp(_exp2_27 + 1.0f);
                __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_min_52 * _rcp_26 * _min_54, _min_53 * _rcp_27 * _min_55));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_13)[0];
                float _max_58 = max_noftz(accum[42], clamp_lo);
                float _min_56 = fminf(_max_58, clamp_limit);
                float _max_59 = max_noftz(accum[43], clamp_lo);
                float _min_57 = fminf(_max_59, clamp_limit);
                float _max_60 = max_noftz(accum[58], clamp_lo);
                float _min_58 = fminf(_max_60, clamp_limit);
                float _max_61 = max_noftz(accum[59], clamp_lo);
                float _min_59 = fminf(_max_61, clamp_limit);
                float _exp2_28 = approx_exp2(_min_56 * -1.4426950408889634f);
                float _rcp_28 = approx_rcp(_exp2_28 + 1.0f);
                float _exp2_29 = approx_exp2(_min_57 * -1.4426950408889634f);
                float _rcp_29 = approx_rcp(_exp2_29 + 1.0f);
                __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_min_56 * _rcp_28 * _min_58, _min_57 * _rcp_29 * _min_59));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_14)[0];
                float _max_62 = max_noftz(accum[46], clamp_lo);
                float _min_60 = fminf(_max_62, clamp_limit);
                float _max_63 = max_noftz(accum[47], clamp_lo);
                float _min_61 = fminf(_max_63, clamp_limit);
                float _max_64 = max_noftz(accum[62], clamp_lo);
                float _min_62 = fminf(_max_64, clamp_limit);
                float _max_65 = max_noftz(accum[63], clamp_lo);
                float _min_63 = fminf(_max_65, clamp_limit);
                float _exp2_30 = approx_exp2(_min_60 * -1.4426950408889634f);
                float _rcp_30 = approx_rcp(_exp2_30 + 1.0f);
                float _exp2_31 = approx_exp2(_min_61 * -1.4426950408889634f);
                float _rcp_31 = approx_rcp(_exp2_31 + 1.0f);
                __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_min_60 * _rcp_30 * _min_62, _min_61 * _rcp_31 * _min_63));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_15)[0];
                unsigned int x0_17 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_18 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, x0_17, 1);
                unsigned int y0_19 = _shfl_xor_12;
                unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, x1_18, 1);
                unsigned int y1_20 = _shfl_xor_13;
                pk[0] = y0_19 & m_odd | pk[0] & m_even;
                pk[1] = y0_19 & m_even | pk[1] & m_odd;
                pk[2] = y1_20 & m_odd | pk[2] & m_even;
                pk[3] = y1_20 & m_even | pk[3] & m_odd;
                unsigned int x2_21 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_22 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, x2_21, 2);
                unsigned int y2_23 = _shfl_xor_14;
                unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, x3_22, 2);
                unsigned int y3_24 = _shfl_xor_15;
                pk[0] = y2_23 & m_hi | pk[0] & m_lo;
                pk[2] = y2_23 & m_lo | pk[2] & m_hi;
                pk[1] = y3_24 & m_hi | pk[1] & m_lo;
                pk[3] = y3_24 & m_lo | pk[3] & m_hi;
                if (row_hi < row_lim_c) {
                    reinterpret_cast<int4*>(D + (row_hi * n_out + col_v_8))[0] = reinterpret_cast<int4*>(pk)[0];
                }
                unsigned int col_v_25 = n0_c / 2 + (8 + q_lane) * 8;
                float _max_66 = max_noftz(accum[64], clamp_lo);
                float _min_64 = fminf(_max_66, clamp_limit);
                float _max_67 = max_noftz(accum[65], clamp_lo);
                float _min_65 = fminf(_max_67, clamp_limit);
                float _max_68 = max_noftz(accum[80], clamp_lo);
                float _min_66 = fminf(_max_68, clamp_limit);
                float _max_69 = max_noftz(accum[81], clamp_lo);
                float _min_67 = fminf(_max_69, clamp_limit);
                float _exp2_32 = approx_exp2(_min_64 * -1.4426950408889634f);
                float _rcp_32 = approx_rcp(_exp2_32 + 1.0f);
                float _exp2_33 = approx_exp2(_min_65 * -1.4426950408889634f);
                float _rcp_33 = approx_rcp(_exp2_33 + 1.0f);
                __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_min_64 * _rcp_32 * _min_66, _min_65 * _rcp_33 * _min_67));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_16)[0];
                float _max_70 = max_noftz(accum[68], clamp_lo);
                float _min_68 = fminf(_max_70, clamp_limit);
                float _max_71 = max_noftz(accum[69], clamp_lo);
                float _min_69 = fminf(_max_71, clamp_limit);
                float _max_72 = max_noftz(accum[84], clamp_lo);
                float _min_70 = fminf(_max_72, clamp_limit);
                float _max_73 = max_noftz(accum[85], clamp_lo);
                float _min_71 = fminf(_max_73, clamp_limit);
                float _exp2_34 = approx_exp2(_min_68 * -1.4426950408889634f);
                float _rcp_34 = approx_rcp(_exp2_34 + 1.0f);
                float _exp2_35 = approx_exp2(_min_69 * -1.4426950408889634f);
                float _rcp_35 = approx_rcp(_exp2_35 + 1.0f);
                __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_min_68 * _rcp_34 * _min_70, _min_69 * _rcp_35 * _min_71));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_17)[0];
                float _max_74 = max_noftz(accum[72], clamp_lo);
                float _min_72 = fminf(_max_74, clamp_limit);
                float _max_75 = max_noftz(accum[73], clamp_lo);
                float _min_73 = fminf(_max_75, clamp_limit);
                float _max_76 = max_noftz(accum[88], clamp_lo);
                float _min_74 = fminf(_max_76, clamp_limit);
                float _max_77 = max_noftz(accum[89], clamp_lo);
                float _min_75 = fminf(_max_77, clamp_limit);
                float _exp2_36 = approx_exp2(_min_72 * -1.4426950408889634f);
                float _rcp_36 = approx_rcp(_exp2_36 + 1.0f);
                float _exp2_37 = approx_exp2(_min_73 * -1.4426950408889634f);
                float _rcp_37 = approx_rcp(_exp2_37 + 1.0f);
                __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_min_72 * _rcp_36 * _min_74, _min_73 * _rcp_37 * _min_75));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_18)[0];
                float _max_78 = max_noftz(accum[76], clamp_lo);
                float _min_76 = fminf(_max_78, clamp_limit);
                float _max_79 = max_noftz(accum[77], clamp_lo);
                float _min_77 = fminf(_max_79, clamp_limit);
                float _max_80 = max_noftz(accum[92], clamp_lo);
                float _min_78 = fminf(_max_80, clamp_limit);
                float _max_81 = max_noftz(accum[93], clamp_lo);
                float _min_79 = fminf(_max_81, clamp_limit);
                float _exp2_38 = approx_exp2(_min_76 * -1.4426950408889634f);
                float _rcp_38 = approx_rcp(_exp2_38 + 1.0f);
                float _exp2_39 = approx_exp2(_min_77 * -1.4426950408889634f);
                float _rcp_39 = approx_rcp(_exp2_39 + 1.0f);
                __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_min_76 * _rcp_38 * _min_78, _min_77 * _rcp_39 * _min_79));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_19)[0];
                unsigned int x0_26 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_27 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, x0_26, 1);
                unsigned int y0_28 = _shfl_xor_16;
                unsigned int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, x1_27, 1);
                unsigned int y1_29 = _shfl_xor_17;
                pk[0] = y0_28 & m_odd | pk[0] & m_even;
                pk[1] = y0_28 & m_even | pk[1] & m_odd;
                pk[2] = y1_29 & m_odd | pk[2] & m_even;
                pk[3] = y1_29 & m_even | pk[3] & m_odd;
                unsigned int x2_30 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_31 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, x2_30, 2);
                unsigned int y2_32 = _shfl_xor_18;
                unsigned int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, x3_31, 2);
                unsigned int y3_33 = _shfl_xor_19;
                pk[0] = y2_32 & m_hi | pk[0] & m_lo;
                pk[2] = y2_32 & m_lo | pk[2] & m_hi;
                pk[1] = y3_33 & m_hi | pk[1] & m_lo;
                pk[3] = y3_33 & m_lo | pk[3] & m_hi;
                if (row_lo < row_lim_c) {
                    reinterpret_cast<int4*>(D + (row_lo * n_out + col_v_25))[0] = reinterpret_cast<int4*>(pk)[0];
                }
                float _max_82 = max_noftz(accum[66], clamp_lo);
                float _min_80 = fminf(_max_82, clamp_limit);
                float _max_83 = max_noftz(accum[67], clamp_lo);
                float _min_81 = fminf(_max_83, clamp_limit);
                float _max_84 = max_noftz(accum[82], clamp_lo);
                float _min_82 = fminf(_max_84, clamp_limit);
                float _max_85 = max_noftz(accum[83], clamp_lo);
                float _min_83 = fminf(_max_85, clamp_limit);
                float _exp2_40 = approx_exp2(_min_80 * -1.4426950408889634f);
                float _rcp_40 = approx_rcp(_exp2_40 + 1.0f);
                float _exp2_41 = approx_exp2(_min_81 * -1.4426950408889634f);
                float _rcp_41 = approx_rcp(_exp2_41 + 1.0f);
                __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_min_80 * _rcp_40 * _min_82, _min_81 * _rcp_41 * _min_83));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_20)[0];
                float _max_86 = max_noftz(accum[70], clamp_lo);
                float _min_84 = fminf(_max_86, clamp_limit);
                float _max_87 = max_noftz(accum[71], clamp_lo);
                float _min_85 = fminf(_max_87, clamp_limit);
                float _max_88 = max_noftz(accum[86], clamp_lo);
                float _min_86 = fminf(_max_88, clamp_limit);
                float _max_89 = max_noftz(accum[87], clamp_lo);
                float _min_87 = fminf(_max_89, clamp_limit);
                float _exp2_42 = approx_exp2(_min_84 * -1.4426950408889634f);
                float _rcp_42 = approx_rcp(_exp2_42 + 1.0f);
                float _exp2_43 = approx_exp2(_min_85 * -1.4426950408889634f);
                float _rcp_43 = approx_rcp(_exp2_43 + 1.0f);
                __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_min_84 * _rcp_42 * _min_86, _min_85 * _rcp_43 * _min_87));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_21)[0];
                float _max_90 = max_noftz(accum[74], clamp_lo);
                float _min_88 = fminf(_max_90, clamp_limit);
                float _max_91 = max_noftz(accum[75], clamp_lo);
                float _min_89 = fminf(_max_91, clamp_limit);
                float _max_92 = max_noftz(accum[90], clamp_lo);
                float _min_90 = fminf(_max_92, clamp_limit);
                float _max_93 = max_noftz(accum[91], clamp_lo);
                float _min_91 = fminf(_max_93, clamp_limit);
                float _exp2_44 = approx_exp2(_min_88 * -1.4426950408889634f);
                float _rcp_44 = approx_rcp(_exp2_44 + 1.0f);
                float _exp2_45 = approx_exp2(_min_89 * -1.4426950408889634f);
                float _rcp_45 = approx_rcp(_exp2_45 + 1.0f);
                __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_min_88 * _rcp_44 * _min_90, _min_89 * _rcp_45 * _min_91));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_22)[0];
                float _max_94 = max_noftz(accum[78], clamp_lo);
                float _min_92 = fminf(_max_94, clamp_limit);
                float _max_95 = max_noftz(accum[79], clamp_lo);
                float _min_93 = fminf(_max_95, clamp_limit);
                float _max_96 = max_noftz(accum[94], clamp_lo);
                float _min_94 = fminf(_max_96, clamp_limit);
                float _max_97 = max_noftz(accum[95], clamp_lo);
                float _min_95 = fminf(_max_97, clamp_limit);
                float _exp2_46 = approx_exp2(_min_92 * -1.4426950408889634f);
                float _rcp_46 = approx_rcp(_exp2_46 + 1.0f);
                float _exp2_47 = approx_exp2(_min_93 * -1.4426950408889634f);
                float _rcp_47 = approx_rcp(_exp2_47 + 1.0f);
                __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_min_92 * _rcp_46 * _min_94, _min_93 * _rcp_47 * _min_95));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_23)[0];
                unsigned int x0_34 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_35 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, x0_34, 1);
                unsigned int y0_36 = _shfl_xor_20;
                unsigned int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, x1_35, 1);
                unsigned int y1_37 = _shfl_xor_21;
                pk[0] = y0_36 & m_odd | pk[0] & m_even;
                pk[1] = y0_36 & m_even | pk[1] & m_odd;
                pk[2] = y1_37 & m_odd | pk[2] & m_even;
                pk[3] = y1_37 & m_even | pk[3] & m_odd;
                unsigned int x2_38 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_39 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, x2_38, 2);
                unsigned int y2_40 = _shfl_xor_22;
                unsigned int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, x3_39, 2);
                unsigned int y3_41 = _shfl_xor_23;
                pk[0] = y2_40 & m_hi | pk[0] & m_lo;
                pk[2] = y2_40 & m_lo | pk[2] & m_hi;
                pk[1] = y3_41 & m_hi | pk[1] & m_lo;
                pk[3] = y3_41 & m_lo | pk[3] & m_hi;
                if (row_hi < row_lim_c) {
                    reinterpret_cast<int4*>(D + (row_hi * n_out + col_v_25))[0] = reinterpret_cast<int4*>(pk)[0];
                }
                unsigned int col_v_42 = n0_c / 2 + (12 + q_lane) * 8;
                float _max_98 = max_noftz(accum[96], clamp_lo);
                float _min_96 = fminf(_max_98, clamp_limit);
                float _max_99 = max_noftz(accum[97], clamp_lo);
                float _min_97 = fminf(_max_99, clamp_limit);
                float _max_100 = max_noftz(accum[112], clamp_lo);
                float _min_98 = fminf(_max_100, clamp_limit);
                float _max_101 = max_noftz(accum[113], clamp_lo);
                float _min_99 = fminf(_max_101, clamp_limit);
                float _exp2_48 = approx_exp2(_min_96 * -1.4426950408889634f);
                float _rcp_48 = approx_rcp(_exp2_48 + 1.0f);
                float _exp2_49 = approx_exp2(_min_97 * -1.4426950408889634f);
                float _rcp_49 = approx_rcp(_exp2_49 + 1.0f);
                __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_min_96 * _rcp_48 * _min_98, _min_97 * _rcp_49 * _min_99));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_24)[0];
                float _max_102 = max_noftz(accum[100], clamp_lo);
                float _min_100 = fminf(_max_102, clamp_limit);
                float _max_103 = max_noftz(accum[101], clamp_lo);
                float _min_101 = fminf(_max_103, clamp_limit);
                float _max_104 = max_noftz(accum[116], clamp_lo);
                float _min_102 = fminf(_max_104, clamp_limit);
                float _max_105 = max_noftz(accum[117], clamp_lo);
                float _min_103 = fminf(_max_105, clamp_limit);
                float _exp2_50 = approx_exp2(_min_100 * -1.4426950408889634f);
                float _rcp_50 = approx_rcp(_exp2_50 + 1.0f);
                float _exp2_51 = approx_exp2(_min_101 * -1.4426950408889634f);
                float _rcp_51 = approx_rcp(_exp2_51 + 1.0f);
                __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_min_100 * _rcp_50 * _min_102, _min_101 * _rcp_51 * _min_103));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_25)[0];
                float _max_106 = max_noftz(accum[104], clamp_lo);
                float _min_104 = fminf(_max_106, clamp_limit);
                float _max_107 = max_noftz(accum[105], clamp_lo);
                float _min_105 = fminf(_max_107, clamp_limit);
                float _max_108 = max_noftz(accum[120], clamp_lo);
                float _min_106 = fminf(_max_108, clamp_limit);
                float _max_109 = max_noftz(accum[121], clamp_lo);
                float _min_107 = fminf(_max_109, clamp_limit);
                float _exp2_52 = approx_exp2(_min_104 * -1.4426950408889634f);
                float _rcp_52 = approx_rcp(_exp2_52 + 1.0f);
                float _exp2_53 = approx_exp2(_min_105 * -1.4426950408889634f);
                float _rcp_53 = approx_rcp(_exp2_53 + 1.0f);
                __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_min_104 * _rcp_52 * _min_106, _min_105 * _rcp_53 * _min_107));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_26)[0];
                float _max_110 = max_noftz(accum[108], clamp_lo);
                float _min_108 = fminf(_max_110, clamp_limit);
                float _max_111 = max_noftz(accum[109], clamp_lo);
                float _min_109 = fminf(_max_111, clamp_limit);
                float _max_112 = max_noftz(accum[124], clamp_lo);
                float _min_110 = fminf(_max_112, clamp_limit);
                float _max_113 = max_noftz(accum[125], clamp_lo);
                float _min_111 = fminf(_max_113, clamp_limit);
                float _exp2_54 = approx_exp2(_min_108 * -1.4426950408889634f);
                float _rcp_54 = approx_rcp(_exp2_54 + 1.0f);
                float _exp2_55 = approx_exp2(_min_109 * -1.4426950408889634f);
                float _rcp_55 = approx_rcp(_exp2_55 + 1.0f);
                __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_min_108 * _rcp_54 * _min_110, _min_109 * _rcp_55 * _min_111));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_27)[0];
                unsigned int x0_43 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_44 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, x0_43, 1);
                unsigned int y0_45 = _shfl_xor_24;
                unsigned int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, x1_44, 1);
                unsigned int y1_46 = _shfl_xor_25;
                pk[0] = y0_45 & m_odd | pk[0] & m_even;
                pk[1] = y0_45 & m_even | pk[1] & m_odd;
                pk[2] = y1_46 & m_odd | pk[2] & m_even;
                pk[3] = y1_46 & m_even | pk[3] & m_odd;
                unsigned int x2_47 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_48 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, x2_47, 2);
                unsigned int y2_49 = _shfl_xor_26;
                unsigned int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, x3_48, 2);
                unsigned int y3_50 = _shfl_xor_27;
                pk[0] = y2_49 & m_hi | pk[0] & m_lo;
                pk[2] = y2_49 & m_lo | pk[2] & m_hi;
                pk[1] = y3_50 & m_hi | pk[1] & m_lo;
                pk[3] = y3_50 & m_lo | pk[3] & m_hi;
                if (row_lo < row_lim_c) {
                    reinterpret_cast<int4*>(D + (row_lo * n_out + col_v_42))[0] = reinterpret_cast<int4*>(pk)[0];
                }
                float _max_114 = max_noftz(accum[98], clamp_lo);
                float _min_112 = fminf(_max_114, clamp_limit);
                float _max_115 = max_noftz(accum[99], clamp_lo);
                float _min_113 = fminf(_max_115, clamp_limit);
                float _max_116 = max_noftz(accum[114], clamp_lo);
                float _min_114 = fminf(_max_116, clamp_limit);
                float _max_117 = max_noftz(accum[115], clamp_lo);
                float _min_115 = fminf(_max_117, clamp_limit);
                float _exp2_56 = approx_exp2(_min_112 * -1.4426950408889634f);
                float _rcp_56 = approx_rcp(_exp2_56 + 1.0f);
                float _exp2_57 = approx_exp2(_min_113 * -1.4426950408889634f);
                float _rcp_57 = approx_rcp(_exp2_57 + 1.0f);
                __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(_min_112 * _rcp_56 * _min_114, _min_113 * _rcp_57 * _min_115));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_28)[0];
                float _max_118 = max_noftz(accum[102], clamp_lo);
                float _min_116 = fminf(_max_118, clamp_limit);
                float _max_119 = max_noftz(accum[103], clamp_lo);
                float _min_117 = fminf(_max_119, clamp_limit);
                float _max_120 = max_noftz(accum[118], clamp_lo);
                float _min_118 = fminf(_max_120, clamp_limit);
                float _max_121 = max_noftz(accum[119], clamp_lo);
                float _min_119 = fminf(_max_121, clamp_limit);
                float _exp2_58 = approx_exp2(_min_116 * -1.4426950408889634f);
                float _rcp_58 = approx_rcp(_exp2_58 + 1.0f);
                float _exp2_59 = approx_exp2(_min_117 * -1.4426950408889634f);
                float _rcp_59 = approx_rcp(_exp2_59 + 1.0f);
                __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(_min_116 * _rcp_58 * _min_118, _min_117 * _rcp_59 * _min_119));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_29)[0];
                float _max_122 = max_noftz(accum[106], clamp_lo);
                float _min_120 = fminf(_max_122, clamp_limit);
                float _max_123 = max_noftz(accum[107], clamp_lo);
                float _min_121 = fminf(_max_123, clamp_limit);
                float _max_124 = max_noftz(accum[122], clamp_lo);
                float _min_122 = fminf(_max_124, clamp_limit);
                float _max_125 = max_noftz(accum[123], clamp_lo);
                float _min_123 = fminf(_max_125, clamp_limit);
                float _exp2_60 = approx_exp2(_min_120 * -1.4426950408889634f);
                float _rcp_60 = approx_rcp(_exp2_60 + 1.0f);
                float _exp2_61 = approx_exp2(_min_121 * -1.4426950408889634f);
                float _rcp_61 = approx_rcp(_exp2_61 + 1.0f);
                __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(_min_120 * _rcp_60 * _min_122, _min_121 * _rcp_61 * _min_123));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_30)[0];
                float _max_126 = max_noftz(accum[110], clamp_lo);
                float _min_124 = fminf(_max_126, clamp_limit);
                float _max_127 = max_noftz(accum[111], clamp_lo);
                float _min_125 = fminf(_max_127, clamp_limit);
                float _max_128 = max_noftz(accum[126], clamp_lo);
                float _min_126 = fminf(_max_128, clamp_limit);
                float _max_129 = max_noftz(accum[127], clamp_lo);
                float _min_127 = fminf(_max_129, clamp_limit);
                float _exp2_62 = approx_exp2(_min_124 * -1.4426950408889634f);
                float _rcp_62 = approx_rcp(_exp2_62 + 1.0f);
                float _exp2_63 = approx_exp2(_min_125 * -1.4426950408889634f);
                float _rcp_63 = approx_rcp(_exp2_63 + 1.0f);
                __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(_min_124 * _rcp_62 * _min_126, _min_125 * _rcp_63 * _min_127));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_31)[0];
                unsigned int x0_51 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_52 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, x0_51, 1);
                unsigned int y0_53 = _shfl_xor_28;
                unsigned int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, x1_52, 1);
                unsigned int y1_54 = _shfl_xor_29;
                pk[0] = y0_53 & m_odd | pk[0] & m_even;
                pk[1] = y0_53 & m_even | pk[1] & m_odd;
                pk[2] = y1_54 & m_odd | pk[2] & m_even;
                pk[3] = y1_54 & m_even | pk[3] & m_odd;
                unsigned int x2_55 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_56 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, x2_55, 2);
                unsigned int y2_57 = _shfl_xor_30;
                unsigned int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, x3_56, 2);
                unsigned int y3_58 = _shfl_xor_31;
                pk[0] = y2_57 & m_hi | pk[0] & m_lo;
                pk[2] = y2_57 & m_lo | pk[2] & m_hi;
                pk[1] = y3_58 & m_hi | pk[1] & m_lo;
                pk[3] = y3_58 & m_lo | pk[3] & m_hi;
                if (row_hi < row_lim_c) {
                    reinterpret_cast<int4*>(D + (row_hi * n_out + col_v_42))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            tile_base_c = tile_base_c + n_tiles_e_c;
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Cleanup
}

} // extern "C"
