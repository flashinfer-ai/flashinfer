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






__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_cake_sm90_bf16_megamoe_fc2(unsigned int num_experts, unsigned int shape_n, unsigned int shape_k, float clamp_limit, const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap W, long long* __restrict__ offsets, __nv_bfloat16* __restrict__ D)
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
    unsigned int n_tiles = shape_n / 256;
    unsigned int n_pairs = (n_tiles + 1) / 2;
    unsigned int num_k_stages = shape_k / 64;
    unsigned int n_out = shape_n;
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
        float outv[4] = {0};
        unsigned int it_c = 0;
        unsigned int tile_base_c = 0;
        unsigned int warp_in_wg = (unsigned int)warp % 4;
        unsigned int lane_row = lane / 4;
        unsigned int lane_col = lane % 4 * 2;
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
                unsigned int col_d = n0_c + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[2 + 0], accum[2 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d)))[0]) = _pk;
                    }
                }
                unsigned int col_d_0 = n0_c + 8 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[4 + 0], accum[4 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_0)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[6 + 0], accum[6 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_0)))[0]) = _pk;
                    }
                }
                unsigned int col_d_1 = n0_c + 16 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[8 + 0], accum[8 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_1)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[10 + 0], accum[10 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_1)))[0]) = _pk;
                    }
                }
                unsigned int col_d_2 = n0_c + 24 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[12 + 0], accum[12 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_2)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[14 + 0], accum[14 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_2)))[0]) = _pk;
                    }
                }
                unsigned int col_d_3 = n0_c + 32 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[16 + 0], accum[16 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_3)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[18 + 0], accum[18 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_3)))[0]) = _pk;
                    }
                }
                unsigned int col_d_4 = n0_c + 40 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[20 + 0], accum[20 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_4)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[22 + 0], accum[22 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_4)))[0]) = _pk;
                    }
                }
                unsigned int col_d_5 = n0_c + 48 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[24 + 0], accum[24 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_5)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[26 + 0], accum[26 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_5)))[0]) = _pk;
                    }
                }
                unsigned int col_d_6 = n0_c + 56 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[28 + 0], accum[28 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_6)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[30 + 0], accum[30 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_6)))[0]) = _pk;
                    }
                }
                unsigned int col_d_7 = n0_c + 64 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[32 + 0], accum[32 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_7)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[34 + 0], accum[34 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_7)))[0]) = _pk;
                    }
                }
                unsigned int col_d_8 = n0_c + 72 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[36 + 0], accum[36 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_8)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[38 + 0], accum[38 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_8)))[0]) = _pk;
                    }
                }
                unsigned int col_d_9 = n0_c + 80 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[40 + 0], accum[40 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_9)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[42 + 0], accum[42 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_9)))[0]) = _pk;
                    }
                }
                unsigned int col_d_10 = n0_c + 88 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[44 + 0], accum[44 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_10)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[46 + 0], accum[46 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_10)))[0]) = _pk;
                    }
                }
                unsigned int col_d_11 = n0_c + 96 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[48 + 0], accum[48 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_11)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[50 + 0], accum[50 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_11)))[0]) = _pk;
                    }
                }
                unsigned int col_d_12 = n0_c + 104 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[52 + 0], accum[52 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_12)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[54 + 0], accum[54 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_12)))[0]) = _pk;
                    }
                }
                unsigned int col_d_13 = n0_c + 112 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[56 + 0], accum[56 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_13)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[58 + 0], accum[58 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_13)))[0]) = _pk;
                    }
                }
                unsigned int col_d_14 = n0_c + 120 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[60 + 0], accum[60 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_14)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[62 + 0], accum[62 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_14)))[0]) = _pk;
                    }
                }
                unsigned int col_d_15 = n0_c + 128 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[64 + 0], accum[64 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_15)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[66 + 0], accum[66 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_15)))[0]) = _pk;
                    }
                }
                unsigned int col_d_16 = n0_c + 136 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[68 + 0], accum[68 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_16)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[70 + 0], accum[70 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_16)))[0]) = _pk;
                    }
                }
                unsigned int col_d_17 = n0_c + 144 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[72 + 0], accum[72 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_17)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[74 + 0], accum[74 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_17)))[0]) = _pk;
                    }
                }
                unsigned int col_d_18 = n0_c + 152 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[76 + 0], accum[76 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_18)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[78 + 0], accum[78 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_18)))[0]) = _pk;
                    }
                }
                unsigned int col_d_19 = n0_c + 160 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[80 + 0], accum[80 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_19)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[82 + 0], accum[82 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_19)))[0]) = _pk;
                    }
                }
                unsigned int col_d_20 = n0_c + 168 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[84 + 0], accum[84 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_20)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[86 + 0], accum[86 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_20)))[0]) = _pk;
                    }
                }
                unsigned int col_d_21 = n0_c + 176 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[88 + 0], accum[88 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_21)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[90 + 0], accum[90 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_21)))[0]) = _pk;
                    }
                }
                unsigned int col_d_22 = n0_c + 184 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[92 + 0], accum[92 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_22)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[94 + 0], accum[94 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_22)))[0]) = _pk;
                    }
                }
                unsigned int col_d_23 = n0_c + 192 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[96 + 0], accum[96 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_23)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[98 + 0], accum[98 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_23)))[0]) = _pk;
                    }
                }
                unsigned int col_d_24 = n0_c + 200 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[100 + 0], accum[100 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_24)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[102 + 0], accum[102 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_24)))[0]) = _pk;
                    }
                }
                unsigned int col_d_25 = n0_c + 208 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[104 + 0], accum[104 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_25)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[106 + 0], accum[106 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_25)))[0]) = _pk;
                    }
                }
                unsigned int col_d_26 = n0_c + 216 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[108 + 0], accum[108 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_26)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[110 + 0], accum[110 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_26)))[0]) = _pk;
                    }
                }
                unsigned int col_d_27 = n0_c + 224 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[112 + 0], accum[112 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_27)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[114 + 0], accum[114 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_27)))[0]) = _pk;
                    }
                }
                unsigned int col_d_28 = n0_c + 232 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[116 + 0], accum[116 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_28)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[118 + 0], accum[118 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_28)))[0]) = _pk;
                    }
                }
                unsigned int col_d_29 = n0_c + 240 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[120 + 0], accum[120 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_29)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[122 + 0], accum[122 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_29)))[0]) = _pk;
                    }
                }
                unsigned int col_d_30 = n0_c + 248 + lane_col;
                if (row_lo < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[124 + 0], accum[124 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_lo * n_out + col_d_30)))[0]) = _pk;
                    }
                }
                if (row_hi < row_lim_c) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(accum[126 + 0], accum[126 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(D + (row_hi * n_out + col_d_30)))[0]) = _pk;
                    }
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
