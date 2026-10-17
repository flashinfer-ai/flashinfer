/*
 * Copyright (c) 2023 by FlashInfer team.
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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_quant_lora_gemm_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_TMA_PIPE_STAGES 5
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 40960
#define SMEM_SMEM_B_OFF 16384
#define SMEM_SMEM_B_STAGE_BYTES 24576
#define SMEM_SMEM_B_STRIDE 40960
#define SMEM_SMEM_D_OFF 0
#define SMEM_SMEM_D_STAGE_BYTES 16384
#define SMEM_SMEM_D_STRIDE 40960
#define SMEM_SMEM_BL_OFF 16384
#define SMEM_SMEM_BL_STAGE_BYTES 24576
#define SMEM_SMEM_BL_STRIDE 40960
#define SMEM_TOTAL 204928
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_quant_lora_gemm_b24e291cc9bbcb6adad1(unsigned int M, unsigned int N, unsigned int K, unsigned int rank, unsigned int group_m, int has_adapter, int has_bias, const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap D, const __grid_constant__ CUtensorMap BL, float* __restrict__ a_scale, float* __restrict__ b_scale, float* __restrict__ lora_scale, __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 204800;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 40)
    #define empty_local_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_D_OFF);
    const int smem_d_addr = smem + SMEM_SMEM_D_OFF;
    __nv_bfloat16* smem_bl = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_BL_OFF);
    const int smem_bl_addr = smem + SMEM_SMEM_BL_OFF;
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&BL))) : "memory");
        }
    }
    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 15 barriers)
    // Mbarriers at smem_raw[204800..204920)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // Stage-major initialization across barrier declarations.
            mbarrier_init(smem + 204800, 1);
            mbarrier_init(smem + 204840, 512);
            mbarrier_init(smem + 204880, 256);
            mbarrier_init(smem + 204808, 1);
            mbarrier_init(smem + 204848, 512);
            mbarrier_init(smem + 204888, 256);
            mbarrier_init(smem + 204816, 1);
            mbarrier_init(smem + 204856, 512);
            mbarrier_init(smem + 204896, 256);
            mbarrier_init(smem + 204824, 1);
            mbarrier_init(smem + 204864, 512);
            mbarrier_init(smem + 204904, 256);
            mbarrier_init(smem + 204832, 1);
            mbarrier_init(smem + 204872, 512);
            mbarrier_init(smem + 204912, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Kernel post-init ops
    unsigned int cid = (unsigned int)cluster_id;
    unsigned int ncl = (unsigned int)num_clusters;
    unsigned int rank_cta = (unsigned int)cta_rank;
    unsigned int m_tiles = (M + 127) / 128;
    unsigned int n_tiles = (N + 191) / 192;
    unsigned int pairs = m_tiles / 2;
    unsigned int tail = m_tiles - pairs * 2;
    unsigned int p_count = pairs * n_tiles;
    unsigned int n_pairs = (n_tiles + 1) / 2;
    unsigned int total_tiles = p_count + tail * n_pairs;
    unsigned int tiles_per_group = group_m * n_tiles;
    unsigned int num_k_stages = (K + 127) / 128;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float alpha_v = a_scale[0] * b_scale[0];
    int use_lora = has_adapter;
    float s_v = 0.0f;
    if (use_lora != 0) {
        s_v = lora_scale[0];
        if (s_v == 0.0f) {
            use_lora = 0;
        }
    }
    unsigned int lora_stages = 0;
    if (use_lora != 0) {
        lora_stages = (rank + 63) / 64;
    }
    if (static_cast<uint32_t>(warp) >= 8u) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        if (warp == 8) {
            if (elect_sync()) {
                unsigned int it_p = 0;
                for (unsigned int t_p = cid; t_p < total_tiles; t_p += ncl) {
                    unsigned int u_p = t_p - p_count;
                    unsigned int m0_lead_p = pairs * 256;
                    unsigned int n0_lead_p = u_p * 2 * 192;
                    unsigned int n0_peer_p = n0_lead_p + 192;
                    if (p_count > t_p) {
                        unsigned int g_p = t_p / tiles_per_group;
                        unsigned int l_p = t_p - g_p * tiles_per_group;
                        unsigned int _min_0 = ((group_m) < (pairs - g_p * group_m) ? (group_m) : (pairs - g_p * group_m));
                        unsigned int gsz_p = _min_0;
                        unsigned int mp_p = g_p * group_m + l_p % gsz_p;
                        unsigned int nt_p = l_p / gsz_p;
                        m0_lead_p = mp_p * 256;
                        n0_lead_p = nt_p * 192;
                        n0_peer_p = nt_p * 192;
                    }
                    for (unsigned int k_p = 0; k_p < num_k_stages; k_p++) {
                        unsigned int stage_p = it_p % 5;
                        mbarrier_wait_relaxed_hint(empty_local_addr + (stage_p) * 8, it_p / 5 + 1 & 1, 10000000);
                        if (cta_rank == 0) {
                            mbarrier_wait_cluster_hint(empty_addr + (stage_p) * 8, it_p / 5 + 1 & 1, 10000000);
                            if (p_count > t_p) {
                                {
                                    const uint64_t _tma_operand_pack_0_desc = (uint64_t)((&A));
                                    const uint32_t _tma_operand_pack_0_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                    const uint32_t _tma_operand_pack_0_dst = (uint32_t)(smem_a_addr + stage_p * 40960);
                                    const int32_t _tma_operand_pack_0_coord_0 = (int32_t)(k_p * 128);
                                    const int32_t _tma_operand_pack_0_coord_1 = (int32_t)(m0_lead_p);
                                    const uint64_t _tma_operand_pack_0_cache = (uint64_t)(0x14F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%3, %4}], [%2], %5;"
                                        :: "r"(_tma_operand_pack_0_dst), "l"(_tma_operand_pack_0_desc), "r"(_tma_operand_pack_0_mbar),
                                           "r"(_tma_operand_pack_0_coord_0), "r"(_tma_operand_pack_0_coord_1), "l"(_tma_operand_pack_0_cache) : "memory");
                                }
                                {
                                    const uint64_t _tma_operand_pack_1_desc = (uint64_t)((&A));
                                    const uint32_t _tma_operand_pack_1_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                    const uint32_t _tma_operand_pack_1_dst = (uint32_t)(smem_a_addr + stage_p * 40960);
                                    const int32_t _tma_operand_pack_1_coord_0 = (int32_t)(k_p * 128);
                                    const int32_t _tma_operand_pack_1_coord_1 = (int32_t)(m0_lead_p + 128);
                                    const uint16_t _tma_operand_pack_1_multicast = (uint16_t)(2);
                                    const uint64_t _tma_operand_pack_1_cache = (uint64_t)(0x14F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                        " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                                        :: "r"(_tma_operand_pack_1_dst), "l"(_tma_operand_pack_1_desc), "r"(_tma_operand_pack_1_coord_0), "r"(_tma_operand_pack_1_coord_1),
                                           "r"(_tma_operand_pack_1_mbar), "h"((uint16_t)(_tma_operand_pack_1_multicast)), "l"(_tma_operand_pack_1_cache) : "memory");
                                }
                                {
                                    const uint64_t _tma_operand_pack_2_desc = (uint64_t)((&W));
                                    const uint32_t _tma_operand_pack_2_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                    const uint32_t _tma_operand_pack_2_dst = (uint32_t)(smem_b_addr + stage_p * 40960);
                                    const int32_t _tma_operand_pack_2_coord_0 = (int32_t)(k_p * 128);
                                    const int32_t _tma_operand_pack_2_coord_1 = (int32_t)(n0_lead_p);
                                    const uint16_t _tma_operand_pack_2_multicast = (uint16_t)(3);
                                    const uint64_t _tma_operand_pack_2_cache = (uint64_t)(0x12F0000000000000ULL);
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
                                    const uint32_t _tma_operand_pack_3_dst = (uint32_t)(smem_a_addr + stage_p * 40960);
                                    const int32_t _tma_operand_pack_3_coord_0 = (int32_t)(k_p * 128);
                                    const int32_t _tma_operand_pack_3_coord_1 = (int32_t)(m0_lead_p);
                                    const uint16_t _tma_operand_pack_3_multicast = (uint16_t)(3);
                                    const uint64_t _tma_operand_pack_3_cache = (uint64_t)(0x14F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                        " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                                        :: "r"(_tma_operand_pack_3_dst), "l"(_tma_operand_pack_3_desc), "r"(_tma_operand_pack_3_coord_0), "r"(_tma_operand_pack_3_coord_1),
                                           "r"(_tma_operand_pack_3_mbar), "h"((uint16_t)(_tma_operand_pack_3_multicast)), "l"(_tma_operand_pack_3_cache) : "memory");
                                }
                                {
                                    const uint64_t _tma_operand_pack_4_desc = (uint64_t)((&W));
                                    const uint32_t _tma_operand_pack_4_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                    const uint32_t _tma_operand_pack_4_dst = (uint32_t)(smem_b_addr + stage_p * 40960);
                                    const int32_t _tma_operand_pack_4_coord_0 = (int32_t)(k_p * 128);
                                    const int32_t _tma_operand_pack_4_coord_1 = (int32_t)(n0_lead_p);
                                    const uint64_t _tma_operand_pack_4_cache = (uint64_t)(0x12F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%3, %4}], [%2], %5;"
                                        :: "r"(_tma_operand_pack_4_dst), "l"(_tma_operand_pack_4_desc), "r"(_tma_operand_pack_4_mbar),
                                           "r"(_tma_operand_pack_4_coord_0), "r"(_tma_operand_pack_4_coord_1), "l"(_tma_operand_pack_4_cache) : "memory");
                                }
                            }
                        }
                        if (rank_cta == 1) {
                            if (p_count <= t_p) {
                                {
                                    const uint64_t _tma_operand_pack_5_desc = (uint64_t)((&W));
                                    const uint32_t _tma_operand_pack_5_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                    const uint32_t _tma_operand_pack_5_dst = (uint32_t)(smem_b_addr + stage_p * 40960);
                                    const int32_t _tma_operand_pack_5_coord_0 = (int32_t)(k_p * 128);
                                    const int32_t _tma_operand_pack_5_coord_1 = (int32_t)(n0_peer_p);
                                    const uint64_t _tma_operand_pack_5_cache = (uint64_t)(0x12F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%3, %4}], [%2], %5;"
                                        :: "r"(_tma_operand_pack_5_dst), "l"(_tma_operand_pack_5_desc), "r"(_tma_operand_pack_5_mbar),
                                           "r"(_tma_operand_pack_5_coord_0), "r"(_tma_operand_pack_5_coord_1), "l"(_tma_operand_pack_5_cache) : "memory");
                                }
                            }
                        }
                        mbarrier_arrive_expect_tx(full_addr + (stage_p) * 8, 40960);
                        it_p = it_p + 1;
                    }
                    for (unsigned int s_p = 0; s_p < lora_stages; s_p++) {
                        unsigned int stage_pl = it_p % 5;
                        mbarrier_wait_relaxed_hint(empty_local_addr + (stage_pl) * 8, it_p / 5 + 1 & 1, 10000000);
                        if (cta_rank == 0) {
                            mbarrier_wait_cluster_hint(empty_addr + (stage_pl) * 8, it_p / 5 + 1 & 1, 10000000);
                            if (p_count > t_p) {
                                {
                                    const uint64_t _tma_operand_pack_6_desc = (uint64_t)((&D));
                                    const uint32_t _tma_operand_pack_6_mbar = (uint32_t)(full_addr + (stage_pl) * 8);
                                    const uint32_t _tma_operand_pack_6_dst = (uint32_t)(smem_d_addr + stage_pl * 40960);
                                    const int32_t _tma_operand_pack_6_coord_0 = (int32_t)(s_p * 64);
                                    const int32_t _tma_operand_pack_6_coord_1 = (int32_t)(m0_lead_p);
                                    const uint64_t _tma_operand_pack_6_cache = (uint64_t)(0x14F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%3, %4}], [%2], %5;"
                                        :: "r"(_tma_operand_pack_6_dst), "l"(_tma_operand_pack_6_desc), "r"(_tma_operand_pack_6_mbar),
                                           "r"(_tma_operand_pack_6_coord_0), "r"(_tma_operand_pack_6_coord_1), "l"(_tma_operand_pack_6_cache) : "memory");
                                }
                                {
                                    const uint64_t _tma_operand_pack_7_desc = (uint64_t)((&D));
                                    const uint32_t _tma_operand_pack_7_mbar = (uint32_t)(full_addr + (stage_pl) * 8);
                                    const uint32_t _tma_operand_pack_7_dst = (uint32_t)(smem_d_addr + stage_pl * 40960);
                                    const int32_t _tma_operand_pack_7_coord_0 = (int32_t)(s_p * 64);
                                    const int32_t _tma_operand_pack_7_coord_1 = (int32_t)(m0_lead_p + 128);
                                    const uint16_t _tma_operand_pack_7_multicast = (uint16_t)(2);
                                    const uint64_t _tma_operand_pack_7_cache = (uint64_t)(0x14F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                        " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                                        :: "r"(_tma_operand_pack_7_dst), "l"(_tma_operand_pack_7_desc), "r"(_tma_operand_pack_7_coord_0), "r"(_tma_operand_pack_7_coord_1),
                                           "r"(_tma_operand_pack_7_mbar), "h"((uint16_t)(_tma_operand_pack_7_multicast)), "l"(_tma_operand_pack_7_cache) : "memory");
                                }
                                {
                                    const uint64_t _tma_operand_pack_8_desc = (uint64_t)((&BL));
                                    const uint32_t _tma_operand_pack_8_mbar = (uint32_t)(full_addr + (stage_pl) * 8);
                                    const uint32_t _tma_operand_pack_8_dst = (uint32_t)(smem_bl_addr + stage_pl * 40960);
                                    const int32_t _tma_operand_pack_8_coord_0 = (int32_t)(s_p * 64);
                                    const int32_t _tma_operand_pack_8_coord_1 = (int32_t)(n0_lead_p);
                                    const uint16_t _tma_operand_pack_8_multicast = (uint16_t)(3);
                                    const uint64_t _tma_operand_pack_8_cache = (uint64_t)(0x12F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                        " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                                        :: "r"(_tma_operand_pack_8_dst), "l"(_tma_operand_pack_8_desc), "r"(_tma_operand_pack_8_coord_0), "r"(_tma_operand_pack_8_coord_1),
                                           "r"(_tma_operand_pack_8_mbar), "h"((uint16_t)(_tma_operand_pack_8_multicast)), "l"(_tma_operand_pack_8_cache) : "memory");
                                }
                            } else {
                                {
                                    const uint64_t _tma_operand_pack_9_desc = (uint64_t)((&D));
                                    const uint32_t _tma_operand_pack_9_mbar = (uint32_t)(full_addr + (stage_pl) * 8);
                                    const uint32_t _tma_operand_pack_9_dst = (uint32_t)(smem_d_addr + stage_pl * 40960);
                                    const int32_t _tma_operand_pack_9_coord_0 = (int32_t)(s_p * 64);
                                    const int32_t _tma_operand_pack_9_coord_1 = (int32_t)(m0_lead_p);
                                    const uint16_t _tma_operand_pack_9_multicast = (uint16_t)(3);
                                    const uint64_t _tma_operand_pack_9_cache = (uint64_t)(0x14F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                                        " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                                        :: "r"(_tma_operand_pack_9_dst), "l"(_tma_operand_pack_9_desc), "r"(_tma_operand_pack_9_coord_0), "r"(_tma_operand_pack_9_coord_1),
                                           "r"(_tma_operand_pack_9_mbar), "h"((uint16_t)(_tma_operand_pack_9_multicast)), "l"(_tma_operand_pack_9_cache) : "memory");
                                }
                                {
                                    const uint64_t _tma_operand_pack_10_desc = (uint64_t)((&BL));
                                    const uint32_t _tma_operand_pack_10_mbar = (uint32_t)(full_addr + (stage_pl) * 8);
                                    const uint32_t _tma_operand_pack_10_dst = (uint32_t)(smem_bl_addr + stage_pl * 40960);
                                    const int32_t _tma_operand_pack_10_coord_0 = (int32_t)(s_p * 64);
                                    const int32_t _tma_operand_pack_10_coord_1 = (int32_t)(n0_lead_p);
                                    const uint64_t _tma_operand_pack_10_cache = (uint64_t)(0x12F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%3, %4}], [%2], %5;"
                                        :: "r"(_tma_operand_pack_10_dst), "l"(_tma_operand_pack_10_desc), "r"(_tma_operand_pack_10_mbar),
                                           "r"(_tma_operand_pack_10_coord_0), "r"(_tma_operand_pack_10_coord_1), "l"(_tma_operand_pack_10_cache) : "memory");
                                }
                            }
                        }
                        if (rank_cta == 1) {
                            if (p_count <= t_p) {
                                {
                                    const uint64_t _tma_operand_pack_11_desc = (uint64_t)((&BL));
                                    const uint32_t _tma_operand_pack_11_mbar = (uint32_t)(full_addr + (stage_pl) * 8);
                                    const uint32_t _tma_operand_pack_11_dst = (uint32_t)(smem_bl_addr + stage_pl * 40960);
                                    const int32_t _tma_operand_pack_11_coord_0 = (int32_t)(s_p * 64);
                                    const int32_t _tma_operand_pack_11_coord_1 = (int32_t)(n0_peer_p);
                                    const uint64_t _tma_operand_pack_11_cache = (uint64_t)(0x12F0000000000000ULL);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%3, %4}], [%2], %5;"
                                        :: "r"(_tma_operand_pack_11_dst), "l"(_tma_operand_pack_11_desc), "r"(_tma_operand_pack_11_mbar),
                                           "r"(_tma_operand_pack_11_coord_0), "r"(_tma_operand_pack_11_coord_1), "l"(_tma_operand_pack_11_cache) : "memory");
                                }
                            }
                        }
                        mbarrier_arrive_expect_tx(full_addr + (stage_pl) * 8, 40960);
                        it_p = it_p + 1;
                    }
                }
            }
        }
    }
    else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        uint32_t warp_group_idx = static_cast<uint32_t>(tid) / 128u;
        unsigned int math_wg_idx = make_warp_uniform(warp_group_idx);
        float accum[96] = {0};
        float acc_fin[96] = {0};
        unsigned int pk[4] = {0};
        float blo[4] = {0};
        float bhi[4] = {0};
        unsigned int it_c = 0;
        unsigned int warp_in_wg = (unsigned int)warp % 4;
        unsigned int lane_row = lane / 4;
        unsigned int lane_col = lane % 4 * 2;
        unsigned int q_lane = lane % 4;
        unsigned int m_odd = -(q_lane & 1);
        unsigned int m_even = m_odd ^ 4294967295u;
        unsigned int m_hi = -(q_lane >> 1 & 1);
        unsigned int m_lo = m_hi ^ 4294967295u;
        for (unsigned int t_c = cid; t_c < total_tiles; t_c += ncl) {
            unsigned int u_c = t_c - p_count;
            unsigned int n_sel_c = u_c * 2 + rank_cta;
            unsigned int m0_c = pairs * 256;
            unsigned int n0_c = n_sel_c * 192;
            unsigned int zero_c = 0;
            unsigned int row_lim_c = ((n_sel_c < n_tiles) ? M : zero_c);
            if (p_count > t_c) {
                unsigned int g_c = t_c / tiles_per_group;
                unsigned int l_c = t_c - g_c * tiles_per_group;
                unsigned int _min_1 = ((group_m) < (pairs - g_c * group_m) ? (group_m) : (pairs - g_c * group_m));
                unsigned int gsz_c = _min_1;
                unsigned int mp_c = g_c * group_m + l_c % gsz_c;
                unsigned int nt_c = l_c / gsz_c;
                m0_c = (mp_c * 2 + rank_cta) * 128;
                n0_c = nt_c * 192;
                row_lim_c = M;
            }
            acc_fin[0] = 0.0f;
            acc_fin[1] = 0.0f;
            acc_fin[2] = 0.0f;
            acc_fin[3] = 0.0f;
            acc_fin[4] = 0.0f;
            acc_fin[5] = 0.0f;
            acc_fin[6] = 0.0f;
            acc_fin[7] = 0.0f;
            acc_fin[8] = 0.0f;
            acc_fin[9] = 0.0f;
            acc_fin[10] = 0.0f;
            acc_fin[11] = 0.0f;
            acc_fin[12] = 0.0f;
            acc_fin[13] = 0.0f;
            acc_fin[14] = 0.0f;
            acc_fin[15] = 0.0f;
            acc_fin[16] = 0.0f;
            acc_fin[17] = 0.0f;
            acc_fin[18] = 0.0f;
            acc_fin[19] = 0.0f;
            acc_fin[20] = 0.0f;
            acc_fin[21] = 0.0f;
            acc_fin[22] = 0.0f;
            acc_fin[23] = 0.0f;
            acc_fin[24] = 0.0f;
            acc_fin[25] = 0.0f;
            acc_fin[26] = 0.0f;
            acc_fin[27] = 0.0f;
            acc_fin[28] = 0.0f;
            acc_fin[29] = 0.0f;
            acc_fin[30] = 0.0f;
            acc_fin[31] = 0.0f;
            acc_fin[32] = 0.0f;
            acc_fin[33] = 0.0f;
            acc_fin[34] = 0.0f;
            acc_fin[35] = 0.0f;
            acc_fin[36] = 0.0f;
            acc_fin[37] = 0.0f;
            acc_fin[38] = 0.0f;
            acc_fin[39] = 0.0f;
            acc_fin[40] = 0.0f;
            acc_fin[41] = 0.0f;
            acc_fin[42] = 0.0f;
            acc_fin[43] = 0.0f;
            acc_fin[44] = 0.0f;
            acc_fin[45] = 0.0f;
            acc_fin[46] = 0.0f;
            acc_fin[47] = 0.0f;
            acc_fin[48] = 0.0f;
            acc_fin[49] = 0.0f;
            acc_fin[50] = 0.0f;
            acc_fin[51] = 0.0f;
            acc_fin[52] = 0.0f;
            acc_fin[53] = 0.0f;
            acc_fin[54] = 0.0f;
            acc_fin[55] = 0.0f;
            acc_fin[56] = 0.0f;
            acc_fin[57] = 0.0f;
            acc_fin[58] = 0.0f;
            acc_fin[59] = 0.0f;
            acc_fin[60] = 0.0f;
            acc_fin[61] = 0.0f;
            acc_fin[62] = 0.0f;
            acc_fin[63] = 0.0f;
            acc_fin[64] = 0.0f;
            acc_fin[65] = 0.0f;
            acc_fin[66] = 0.0f;
            acc_fin[67] = 0.0f;
            acc_fin[68] = 0.0f;
            acc_fin[69] = 0.0f;
            acc_fin[70] = 0.0f;
            acc_fin[71] = 0.0f;
            acc_fin[72] = 0.0f;
            acc_fin[73] = 0.0f;
            acc_fin[74] = 0.0f;
            acc_fin[75] = 0.0f;
            acc_fin[76] = 0.0f;
            acc_fin[77] = 0.0f;
            acc_fin[78] = 0.0f;
            acc_fin[79] = 0.0f;
            acc_fin[80] = 0.0f;
            acc_fin[81] = 0.0f;
            acc_fin[82] = 0.0f;
            acc_fin[83] = 0.0f;
            acc_fin[84] = 0.0f;
            acc_fin[85] = 0.0f;
            acc_fin[86] = 0.0f;
            acc_fin[87] = 0.0f;
            acc_fin[88] = 0.0f;
            acc_fin[89] = 0.0f;
            acc_fin[90] = 0.0f;
            acc_fin[91] = 0.0f;
            acc_fin[92] = 0.0f;
            acc_fin[93] = 0.0f;
            acc_fin[94] = 0.0f;
            acc_fin[95] = 0.0f;
            for (unsigned int k_c = 0; k_c < num_k_stages; k_c++) {
                unsigned int stage_c = it_c % 5;
                unsigned int blk_pos = k_c % 8;
                unsigned int one_c = 1;
                unsigned int _min_2 = ((blk_pos) < (one_c) ? (blk_pos) : (one_c));
                unsigned int first_in_period = one_c - _min_2;
                unsigned int blocks_left = 7 - blk_pos;
                unsigned int _min_3 = ((blocks_left) < (one_c) ? (blocks_left) : (one_c));
                unsigned int promote_now = one_c - _min_3;
                unsigned int last_k = num_k_stages - 1;
                if (k_c == last_k) {
                    promote_now = one_c;
                }
                mbarrier_wait_relaxed_hint(full_addr + (stage_c) * 8, it_c / 5 & 1, 10000000);
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
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                uint32_t _wgmma_stage_base_0 = (unsigned int)smem + (unsigned int)stage_c * 40960;
                {
                    uint64_t _wgmma_a_0_0 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(0) * 1u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_1 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(0) * 1u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %98, 0;\nwgmma.mma_async.sync.aligned.m64n192k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, p, 1, 1;\n}\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95])
                        : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1), "r"(1 - first_in_period)
                        : "memory");
                }
                {
                    uint64_t _wgmma_a_0_2 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(32) * 1u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_3 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(32) * 1u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n192k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 1, 1, 1;\n}\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95])
                        : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_3)
                        : "memory");
                }
                {
                    uint64_t _wgmma_a_0_4 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(64) * 1u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_5 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(64) * 1u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n192k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 1, 1, 1;\n}\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95])
                        : "l"(_wgmma_a_0_4), "l"(_wgmma_b_0_5)
                        : "memory");
                }
                {
                    uint64_t _wgmma_a_0_6 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(96) * 1u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_7 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(96) * 1u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n192k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 1, 1, 1;\n}\n"
                        : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95])
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
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                mbarrier_arrive(empty_local_addr + (stage_c) * 8);
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((empty_addr + (stage_c) * 8) & 0xFEFFFFFF) : "memory");
                if (promote_now != 0) {
                    acc_fin[0] = acc_fin[0] + accum[0];
                    acc_fin[1] = acc_fin[1] + accum[1];
                    acc_fin[2] = acc_fin[2] + accum[2];
                    acc_fin[3] = acc_fin[3] + accum[3];
                    acc_fin[4] = acc_fin[4] + accum[4];
                    acc_fin[5] = acc_fin[5] + accum[5];
                    acc_fin[6] = acc_fin[6] + accum[6];
                    acc_fin[7] = acc_fin[7] + accum[7];
                    acc_fin[8] = acc_fin[8] + accum[8];
                    acc_fin[9] = acc_fin[9] + accum[9];
                    acc_fin[10] = acc_fin[10] + accum[10];
                    acc_fin[11] = acc_fin[11] + accum[11];
                    acc_fin[12] = acc_fin[12] + accum[12];
                    acc_fin[13] = acc_fin[13] + accum[13];
                    acc_fin[14] = acc_fin[14] + accum[14];
                    acc_fin[15] = acc_fin[15] + accum[15];
                    acc_fin[16] = acc_fin[16] + accum[16];
                    acc_fin[17] = acc_fin[17] + accum[17];
                    acc_fin[18] = acc_fin[18] + accum[18];
                    acc_fin[19] = acc_fin[19] + accum[19];
                    acc_fin[20] = acc_fin[20] + accum[20];
                    acc_fin[21] = acc_fin[21] + accum[21];
                    acc_fin[22] = acc_fin[22] + accum[22];
                    acc_fin[23] = acc_fin[23] + accum[23];
                    acc_fin[24] = acc_fin[24] + accum[24];
                    acc_fin[25] = acc_fin[25] + accum[25];
                    acc_fin[26] = acc_fin[26] + accum[26];
                    acc_fin[27] = acc_fin[27] + accum[27];
                    acc_fin[28] = acc_fin[28] + accum[28];
                    acc_fin[29] = acc_fin[29] + accum[29];
                    acc_fin[30] = acc_fin[30] + accum[30];
                    acc_fin[31] = acc_fin[31] + accum[31];
                    acc_fin[32] = acc_fin[32] + accum[32];
                    acc_fin[33] = acc_fin[33] + accum[33];
                    acc_fin[34] = acc_fin[34] + accum[34];
                    acc_fin[35] = acc_fin[35] + accum[35];
                    acc_fin[36] = acc_fin[36] + accum[36];
                    acc_fin[37] = acc_fin[37] + accum[37];
                    acc_fin[38] = acc_fin[38] + accum[38];
                    acc_fin[39] = acc_fin[39] + accum[39];
                    acc_fin[40] = acc_fin[40] + accum[40];
                    acc_fin[41] = acc_fin[41] + accum[41];
                    acc_fin[42] = acc_fin[42] + accum[42];
                    acc_fin[43] = acc_fin[43] + accum[43];
                    acc_fin[44] = acc_fin[44] + accum[44];
                    acc_fin[45] = acc_fin[45] + accum[45];
                    acc_fin[46] = acc_fin[46] + accum[46];
                    acc_fin[47] = acc_fin[47] + accum[47];
                    acc_fin[48] = acc_fin[48] + accum[48];
                    acc_fin[49] = acc_fin[49] + accum[49];
                    acc_fin[50] = acc_fin[50] + accum[50];
                    acc_fin[51] = acc_fin[51] + accum[51];
                    acc_fin[52] = acc_fin[52] + accum[52];
                    acc_fin[53] = acc_fin[53] + accum[53];
                    acc_fin[54] = acc_fin[54] + accum[54];
                    acc_fin[55] = acc_fin[55] + accum[55];
                    acc_fin[56] = acc_fin[56] + accum[56];
                    acc_fin[57] = acc_fin[57] + accum[57];
                    acc_fin[58] = acc_fin[58] + accum[58];
                    acc_fin[59] = acc_fin[59] + accum[59];
                    acc_fin[60] = acc_fin[60] + accum[60];
                    acc_fin[61] = acc_fin[61] + accum[61];
                    acc_fin[62] = acc_fin[62] + accum[62];
                    acc_fin[63] = acc_fin[63] + accum[63];
                    acc_fin[64] = acc_fin[64] + accum[64];
                    acc_fin[65] = acc_fin[65] + accum[65];
                    acc_fin[66] = acc_fin[66] + accum[66];
                    acc_fin[67] = acc_fin[67] + accum[67];
                    acc_fin[68] = acc_fin[68] + accum[68];
                    acc_fin[69] = acc_fin[69] + accum[69];
                    acc_fin[70] = acc_fin[70] + accum[70];
                    acc_fin[71] = acc_fin[71] + accum[71];
                    acc_fin[72] = acc_fin[72] + accum[72];
                    acc_fin[73] = acc_fin[73] + accum[73];
                    acc_fin[74] = acc_fin[74] + accum[74];
                    acc_fin[75] = acc_fin[75] + accum[75];
                    acc_fin[76] = acc_fin[76] + accum[76];
                    acc_fin[77] = acc_fin[77] + accum[77];
                    acc_fin[78] = acc_fin[78] + accum[78];
                    acc_fin[79] = acc_fin[79] + accum[79];
                    acc_fin[80] = acc_fin[80] + accum[80];
                    acc_fin[81] = acc_fin[81] + accum[81];
                    acc_fin[82] = acc_fin[82] + accum[82];
                    acc_fin[83] = acc_fin[83] + accum[83];
                    acc_fin[84] = acc_fin[84] + accum[84];
                    acc_fin[85] = acc_fin[85] + accum[85];
                    acc_fin[86] = acc_fin[86] + accum[86];
                    acc_fin[87] = acc_fin[87] + accum[87];
                    acc_fin[88] = acc_fin[88] + accum[88];
                    acc_fin[89] = acc_fin[89] + accum[89];
                    acc_fin[90] = acc_fin[90] + accum[90];
                    acc_fin[91] = acc_fin[91] + accum[91];
                    acc_fin[92] = acc_fin[92] + accum[92];
                    acc_fin[93] = acc_fin[93] + accum[93];
                    acc_fin[94] = acc_fin[94] + accum[94];
                    acc_fin[95] = acc_fin[95] + accum[95];
                }
                it_c = it_c + 1;
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
            for (unsigned int s_c = 0; s_c < lora_stages; s_c++) {
                unsigned int stage_cl = it_c % 5;
                mbarrier_wait_relaxed_hint(full_addr + (stage_cl) * 8, it_c / 5 & 1, 10000000);
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
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                uint32_t _wgmma_stage_base_1 = (unsigned int)smem + (unsigned int)stage_cl * 40960;
                unsigned int slab_col0 = s_c * 64;
                if (slab_col0 < rank) {
                    {
                        uint64_t _wgmma_a_0_8 = (((uint64_t)(((_wgmma_stage_base_1 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(0) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_9 = (((uint64_t)(((_wgmma_stage_base_1 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(0) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95])
                            : "l"(_wgmma_a_0_8), "l"(_wgmma_b_0_9)
                            : "memory");
                    }
                }
                if (slab_col0 + 16 < rank) {
                    {
                        uint64_t _wgmma_a_0_10 = (((uint64_t)(((_wgmma_stage_base_1 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(16) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_11 = (((uint64_t)(((_wgmma_stage_base_1 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(16) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95])
                            : "l"(_wgmma_a_0_10), "l"(_wgmma_b_0_11)
                            : "memory");
                    }
                }
                if (slab_col0 + 32 < rank) {
                    {
                        uint64_t _wgmma_a_0_12 = (((uint64_t)(((_wgmma_stage_base_1 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(32) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_13 = (((uint64_t)(((_wgmma_stage_base_1 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(32) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95])
                            : "l"(_wgmma_a_0_12), "l"(_wgmma_b_0_13)
                            : "memory");
                    }
                }
                if (slab_col0 + 48 < rank) {
                    {
                        uint64_t _wgmma_a_0_14 = (((uint64_t)(((_wgmma_stage_base_1 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(48) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_15 = (((uint64_t)(((_wgmma_stage_base_1 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(48) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63]), "+f"(accum[64]), "+f"(accum[65]), "+f"(accum[66]), "+f"(accum[67]), "+f"(accum[68]), "+f"(accum[69]), "+f"(accum[70]), "+f"(accum[71]), "+f"(accum[72]), "+f"(accum[73]), "+f"(accum[74]), "+f"(accum[75]), "+f"(accum[76]), "+f"(accum[77]), "+f"(accum[78]), "+f"(accum[79]), "+f"(accum[80]), "+f"(accum[81]), "+f"(accum[82]), "+f"(accum[83]), "+f"(accum[84]), "+f"(accum[85]), "+f"(accum[86]), "+f"(accum[87]), "+f"(accum[88]), "+f"(accum[89]), "+f"(accum[90]), "+f"(accum[91]), "+f"(accum[92]), "+f"(accum[93]), "+f"(accum[94]), "+f"(accum[95])
                            : "l"(_wgmma_a_0_14), "l"(_wgmma_b_0_15)
                            : "memory");
                    }
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
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                mbarrier_arrive(empty_local_addr + (stage_cl) * 8);
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((empty_addr + (stage_cl) * 8) & 0xFEFFFFFF) : "memory");
                it_c = it_c + 1;
            }
            unsigned int row_lo = m0_c + math_wg_idx * 64 + warp_in_wg * 16 + lane_row;
            unsigned int row_hi = row_lo + 8;
            unsigned int bcol0 = 0;
            unsigned int bias_word = 0;
            unsigned int bw_lo = 0;
            unsigned int bw_hi = 0;
            float b_lo_f = 0.0f;
            float b_hi_f = 0.0f;
            blo[0] = 0.0f;
            bhi[0] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c;
                if (bcol0 < N) {
                    unsigned int _vec_load_0[2];
                    {
                        uint32_t _bf16x2_bits_12;
                        _bf16x2_bits_12 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_0[0] = _bf16x2_bits_12;
                    }
                    bias_word = _vec_load_0[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[0] = b_lo_f;
                    bhi[0] = b_hi_f;
                }
            }
            blo[1] = 0.0f;
            bhi[1] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 8;
                if (bcol0 < N) {
                    unsigned int _vec_load_1[2];
                    {
                        uint32_t _bf16x2_bits_13;
                        _bf16x2_bits_13 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_1[0] = _bf16x2_bits_13;
                    }
                    bias_word = _vec_load_1[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[1] = b_lo_f;
                    bhi[1] = b_hi_f;
                }
            }
            blo[2] = 0.0f;
            bhi[2] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 16;
                if (bcol0 < N) {
                    unsigned int _vec_load_2[2];
                    {
                        uint32_t _bf16x2_bits_14;
                        _bf16x2_bits_14 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_2[0] = _bf16x2_bits_14;
                    }
                    bias_word = _vec_load_2[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[2] = b_lo_f;
                    bhi[2] = b_hi_f;
                }
            }
            blo[3] = 0.0f;
            bhi[3] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 24;
                if (bcol0 < N) {
                    unsigned int _vec_load_3[2];
                    {
                        uint32_t _bf16x2_bits_15;
                        _bf16x2_bits_15 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_3[0] = _bf16x2_bits_15;
                    }
                    bias_word = _vec_load_3[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[3] = b_lo_f;
                    bhi[3] = b_hi_f;
                }
            }
            unsigned int col_v = n0_c + q_lane * 8;
            float _fma_0 = __fmaf_rn(accum[0], s_v, blo[0]);
            float _fma_1 = __fmaf_rn(accum[1], s_v, bhi[0]);
            float _fma_2 = __fmaf_rn(acc_fin[0], alpha_v, _fma_0);
            float _fma_3 = __fmaf_rn(acc_fin[1], alpha_v, _fma_1);
            __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_fma_2, _fma_3));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_0)[0];
            float _fma_4 = __fmaf_rn(accum[4], s_v, blo[1]);
            float _fma_5 = __fmaf_rn(accum[5], s_v, bhi[1]);
            float _fma_6 = __fmaf_rn(acc_fin[4], alpha_v, _fma_4);
            float _fma_7 = __fmaf_rn(acc_fin[5], alpha_v, _fma_5);
            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_fma_6, _fma_7));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_1)[0];
            float _fma_8 = __fmaf_rn(accum[8], s_v, blo[2]);
            float _fma_9 = __fmaf_rn(accum[9], s_v, bhi[2]);
            float _fma_10 = __fmaf_rn(acc_fin[8], alpha_v, _fma_8);
            float _fma_11 = __fmaf_rn(acc_fin[9], alpha_v, _fma_9);
            __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_fma_10, _fma_11));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_2)[0];
            float _fma_12 = __fmaf_rn(accum[12], s_v, blo[3]);
            float _fma_13 = __fmaf_rn(accum[13], s_v, bhi[3]);
            float _fma_14 = __fmaf_rn(acc_fin[12], alpha_v, _fma_12);
            float _fma_15 = __fmaf_rn(acc_fin[13], alpha_v, _fma_13);
            __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_fma_14, _fma_15));
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
                if (col_v < N) {
                    reinterpret_cast<int4*>(out + (row_lo * N + col_v))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            float _fma_16 = __fmaf_rn(accum[2], s_v, blo[0]);
            float _fma_17 = __fmaf_rn(accum[3], s_v, bhi[0]);
            float _fma_18 = __fmaf_rn(acc_fin[2], alpha_v, _fma_16);
            float _fma_19 = __fmaf_rn(acc_fin[3], alpha_v, _fma_17);
            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_fma_18, _fma_19));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_4)[0];
            float _fma_20 = __fmaf_rn(accum[6], s_v, blo[1]);
            float _fma_21 = __fmaf_rn(accum[7], s_v, bhi[1]);
            float _fma_22 = __fmaf_rn(acc_fin[6], alpha_v, _fma_20);
            float _fma_23 = __fmaf_rn(acc_fin[7], alpha_v, _fma_21);
            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_fma_22, _fma_23));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_5)[0];
            float _fma_24 = __fmaf_rn(accum[10], s_v, blo[2]);
            float _fma_25 = __fmaf_rn(accum[11], s_v, bhi[2]);
            float _fma_26 = __fmaf_rn(acc_fin[10], alpha_v, _fma_24);
            float _fma_27 = __fmaf_rn(acc_fin[11], alpha_v, _fma_25);
            __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_fma_26, _fma_27));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_6)[0];
            float _fma_28 = __fmaf_rn(accum[14], s_v, blo[3]);
            float _fma_29 = __fmaf_rn(accum[15], s_v, bhi[3]);
            float _fma_30 = __fmaf_rn(acc_fin[14], alpha_v, _fma_28);
            float _fma_31 = __fmaf_rn(acc_fin[15], alpha_v, _fma_29);
            __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_fma_30, _fma_31));
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
                if (col_v < N) {
                    reinterpret_cast<int4*>(out + (row_hi * N + col_v))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            blo[0] = 0.0f;
            bhi[0] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 32;
                if (bcol0 < N) {
                    unsigned int _vec_load_4[2];
                    {
                        uint32_t _bf16x2_bits_16;
                        _bf16x2_bits_16 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_4[0] = _bf16x2_bits_16;
                    }
                    bias_word = _vec_load_4[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[0] = b_lo_f;
                    bhi[0] = b_hi_f;
                }
            }
            blo[1] = 0.0f;
            bhi[1] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 40;
                if (bcol0 < N) {
                    unsigned int _vec_load_5[2];
                    {
                        uint32_t _bf16x2_bits_17;
                        _bf16x2_bits_17 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_5[0] = _bf16x2_bits_17;
                    }
                    bias_word = _vec_load_5[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[1] = b_lo_f;
                    bhi[1] = b_hi_f;
                }
            }
            blo[2] = 0.0f;
            bhi[2] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 48;
                if (bcol0 < N) {
                    unsigned int _vec_load_6[2];
                    {
                        uint32_t _bf16x2_bits_18;
                        _bf16x2_bits_18 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_6[0] = _bf16x2_bits_18;
                    }
                    bias_word = _vec_load_6[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[2] = b_lo_f;
                    bhi[2] = b_hi_f;
                }
            }
            blo[3] = 0.0f;
            bhi[3] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 56;
                if (bcol0 < N) {
                    unsigned int _vec_load_7[2];
                    {
                        uint32_t _bf16x2_bits_19;
                        _bf16x2_bits_19 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_7[0] = _bf16x2_bits_19;
                    }
                    bias_word = _vec_load_7[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[3] = b_lo_f;
                    bhi[3] = b_hi_f;
                }
            }
            unsigned int col_v_8 = n0_c + (4 + q_lane) * 8;
            float _fma_32 = __fmaf_rn(accum[16], s_v, blo[0]);
            float _fma_33 = __fmaf_rn(accum[17], s_v, bhi[0]);
            float _fma_34 = __fmaf_rn(acc_fin[16], alpha_v, _fma_32);
            float _fma_35 = __fmaf_rn(acc_fin[17], alpha_v, _fma_33);
            __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_fma_34, _fma_35));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_8)[0];
            float _fma_36 = __fmaf_rn(accum[20], s_v, blo[1]);
            float _fma_37 = __fmaf_rn(accum[21], s_v, bhi[1]);
            float _fma_38 = __fmaf_rn(acc_fin[20], alpha_v, _fma_36);
            float _fma_39 = __fmaf_rn(acc_fin[21], alpha_v, _fma_37);
            __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_fma_38, _fma_39));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_9)[0];
            float _fma_40 = __fmaf_rn(accum[24], s_v, blo[2]);
            float _fma_41 = __fmaf_rn(accum[25], s_v, bhi[2]);
            float _fma_42 = __fmaf_rn(acc_fin[24], alpha_v, _fma_40);
            float _fma_43 = __fmaf_rn(acc_fin[25], alpha_v, _fma_41);
            __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_fma_42, _fma_43));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_10)[0];
            float _fma_44 = __fmaf_rn(accum[28], s_v, blo[3]);
            float _fma_45 = __fmaf_rn(accum[29], s_v, bhi[3]);
            float _fma_46 = __fmaf_rn(acc_fin[28], alpha_v, _fma_44);
            float _fma_47 = __fmaf_rn(acc_fin[29], alpha_v, _fma_45);
            __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_fma_46, _fma_47));
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
                if (col_v_8 < N) {
                    reinterpret_cast<int4*>(out + (row_lo * N + col_v_8))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            float _fma_48 = __fmaf_rn(accum[18], s_v, blo[0]);
            float _fma_49 = __fmaf_rn(accum[19], s_v, bhi[0]);
            float _fma_50 = __fmaf_rn(acc_fin[18], alpha_v, _fma_48);
            float _fma_51 = __fmaf_rn(acc_fin[19], alpha_v, _fma_49);
            __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_fma_50, _fma_51));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_12)[0];
            float _fma_52 = __fmaf_rn(accum[22], s_v, blo[1]);
            float _fma_53 = __fmaf_rn(accum[23], s_v, bhi[1]);
            float _fma_54 = __fmaf_rn(acc_fin[22], alpha_v, _fma_52);
            float _fma_55 = __fmaf_rn(acc_fin[23], alpha_v, _fma_53);
            __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_fma_54, _fma_55));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_13)[0];
            float _fma_56 = __fmaf_rn(accum[26], s_v, blo[2]);
            float _fma_57 = __fmaf_rn(accum[27], s_v, bhi[2]);
            float _fma_58 = __fmaf_rn(acc_fin[26], alpha_v, _fma_56);
            float _fma_59 = __fmaf_rn(acc_fin[27], alpha_v, _fma_57);
            __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_fma_58, _fma_59));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_14)[0];
            float _fma_60 = __fmaf_rn(accum[30], s_v, blo[3]);
            float _fma_61 = __fmaf_rn(accum[31], s_v, bhi[3]);
            float _fma_62 = __fmaf_rn(acc_fin[30], alpha_v, _fma_60);
            float _fma_63 = __fmaf_rn(acc_fin[31], alpha_v, _fma_61);
            __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_fma_62, _fma_63));
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
                if (col_v_8 < N) {
                    reinterpret_cast<int4*>(out + (row_hi * N + col_v_8))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            blo[0] = 0.0f;
            bhi[0] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 64;
                if (bcol0 < N) {
                    unsigned int _vec_load_8[2];
                    {
                        uint32_t _bf16x2_bits_20;
                        _bf16x2_bits_20 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_8[0] = _bf16x2_bits_20;
                    }
                    bias_word = _vec_load_8[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[0] = b_lo_f;
                    bhi[0] = b_hi_f;
                }
            }
            blo[1] = 0.0f;
            bhi[1] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 72;
                if (bcol0 < N) {
                    unsigned int _vec_load_9[2];
                    {
                        uint32_t _bf16x2_bits_21;
                        _bf16x2_bits_21 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_9[0] = _bf16x2_bits_21;
                    }
                    bias_word = _vec_load_9[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[1] = b_lo_f;
                    bhi[1] = b_hi_f;
                }
            }
            blo[2] = 0.0f;
            bhi[2] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 80;
                if (bcol0 < N) {
                    unsigned int _vec_load_10[2];
                    {
                        uint32_t _bf16x2_bits_22;
                        _bf16x2_bits_22 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_10[0] = _bf16x2_bits_22;
                    }
                    bias_word = _vec_load_10[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[2] = b_lo_f;
                    bhi[2] = b_hi_f;
                }
            }
            blo[3] = 0.0f;
            bhi[3] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 88;
                if (bcol0 < N) {
                    unsigned int _vec_load_11[2];
                    {
                        uint32_t _bf16x2_bits_23;
                        _bf16x2_bits_23 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_11[0] = _bf16x2_bits_23;
                    }
                    bias_word = _vec_load_11[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[3] = b_lo_f;
                    bhi[3] = b_hi_f;
                }
            }
            unsigned int col_v_25 = n0_c + (8 + q_lane) * 8;
            float _fma_64 = __fmaf_rn(accum[32], s_v, blo[0]);
            float _fma_65 = __fmaf_rn(accum[33], s_v, bhi[0]);
            float _fma_66 = __fmaf_rn(acc_fin[32], alpha_v, _fma_64);
            float _fma_67 = __fmaf_rn(acc_fin[33], alpha_v, _fma_65);
            __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_fma_66, _fma_67));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_16)[0];
            float _fma_68 = __fmaf_rn(accum[36], s_v, blo[1]);
            float _fma_69 = __fmaf_rn(accum[37], s_v, bhi[1]);
            float _fma_70 = __fmaf_rn(acc_fin[36], alpha_v, _fma_68);
            float _fma_71 = __fmaf_rn(acc_fin[37], alpha_v, _fma_69);
            __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_fma_70, _fma_71));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_17)[0];
            float _fma_72 = __fmaf_rn(accum[40], s_v, blo[2]);
            float _fma_73 = __fmaf_rn(accum[41], s_v, bhi[2]);
            float _fma_74 = __fmaf_rn(acc_fin[40], alpha_v, _fma_72);
            float _fma_75 = __fmaf_rn(acc_fin[41], alpha_v, _fma_73);
            __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_fma_74, _fma_75));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_18)[0];
            float _fma_76 = __fmaf_rn(accum[44], s_v, blo[3]);
            float _fma_77 = __fmaf_rn(accum[45], s_v, bhi[3]);
            float _fma_78 = __fmaf_rn(acc_fin[44], alpha_v, _fma_76);
            float _fma_79 = __fmaf_rn(acc_fin[45], alpha_v, _fma_77);
            __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_fma_78, _fma_79));
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
                if (col_v_25 < N) {
                    reinterpret_cast<int4*>(out + (row_lo * N + col_v_25))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            float _fma_80 = __fmaf_rn(accum[34], s_v, blo[0]);
            float _fma_81 = __fmaf_rn(accum[35], s_v, bhi[0]);
            float _fma_82 = __fmaf_rn(acc_fin[34], alpha_v, _fma_80);
            float _fma_83 = __fmaf_rn(acc_fin[35], alpha_v, _fma_81);
            __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_fma_82, _fma_83));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_20)[0];
            float _fma_84 = __fmaf_rn(accum[38], s_v, blo[1]);
            float _fma_85 = __fmaf_rn(accum[39], s_v, bhi[1]);
            float _fma_86 = __fmaf_rn(acc_fin[38], alpha_v, _fma_84);
            float _fma_87 = __fmaf_rn(acc_fin[39], alpha_v, _fma_85);
            __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_fma_86, _fma_87));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_21)[0];
            float _fma_88 = __fmaf_rn(accum[42], s_v, blo[2]);
            float _fma_89 = __fmaf_rn(accum[43], s_v, bhi[2]);
            float _fma_90 = __fmaf_rn(acc_fin[42], alpha_v, _fma_88);
            float _fma_91 = __fmaf_rn(acc_fin[43], alpha_v, _fma_89);
            __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_fma_90, _fma_91));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_22)[0];
            float _fma_92 = __fmaf_rn(accum[46], s_v, blo[3]);
            float _fma_93 = __fmaf_rn(accum[47], s_v, bhi[3]);
            float _fma_94 = __fmaf_rn(acc_fin[46], alpha_v, _fma_92);
            float _fma_95 = __fmaf_rn(acc_fin[47], alpha_v, _fma_93);
            __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_fma_94, _fma_95));
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
                if (col_v_25 < N) {
                    reinterpret_cast<int4*>(out + (row_hi * N + col_v_25))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            blo[0] = 0.0f;
            bhi[0] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 96;
                if (bcol0 < N) {
                    unsigned int _vec_load_12[2];
                    {
                        uint32_t _bf16x2_bits_24;
                        _bf16x2_bits_24 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_12[0] = _bf16x2_bits_24;
                    }
                    bias_word = _vec_load_12[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[0] = b_lo_f;
                    bhi[0] = b_hi_f;
                }
            }
            blo[1] = 0.0f;
            bhi[1] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 104;
                if (bcol0 < N) {
                    unsigned int _vec_load_13[2];
                    {
                        uint32_t _bf16x2_bits_25;
                        _bf16x2_bits_25 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_13[0] = _bf16x2_bits_25;
                    }
                    bias_word = _vec_load_13[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[1] = b_lo_f;
                    bhi[1] = b_hi_f;
                }
            }
            blo[2] = 0.0f;
            bhi[2] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 112;
                if (bcol0 < N) {
                    unsigned int _vec_load_14[2];
                    {
                        uint32_t _bf16x2_bits_26;
                        _bf16x2_bits_26 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_14[0] = _bf16x2_bits_26;
                    }
                    bias_word = _vec_load_14[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[2] = b_lo_f;
                    bhi[2] = b_hi_f;
                }
            }
            blo[3] = 0.0f;
            bhi[3] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 120;
                if (bcol0 < N) {
                    unsigned int _vec_load_15[2];
                    {
                        uint32_t _bf16x2_bits_27;
                        _bf16x2_bits_27 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_15[0] = _bf16x2_bits_27;
                    }
                    bias_word = _vec_load_15[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[3] = b_lo_f;
                    bhi[3] = b_hi_f;
                }
            }
            unsigned int col_v_42 = n0_c + (12 + q_lane) * 8;
            float _fma_96 = __fmaf_rn(accum[48], s_v, blo[0]);
            float _fma_97 = __fmaf_rn(accum[49], s_v, bhi[0]);
            float _fma_98 = __fmaf_rn(acc_fin[48], alpha_v, _fma_96);
            float _fma_99 = __fmaf_rn(acc_fin[49], alpha_v, _fma_97);
            __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_fma_98, _fma_99));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_24)[0];
            float _fma_100 = __fmaf_rn(accum[52], s_v, blo[1]);
            float _fma_101 = __fmaf_rn(accum[53], s_v, bhi[1]);
            float _fma_102 = __fmaf_rn(acc_fin[52], alpha_v, _fma_100);
            float _fma_103 = __fmaf_rn(acc_fin[53], alpha_v, _fma_101);
            __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_fma_102, _fma_103));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_25)[0];
            float _fma_104 = __fmaf_rn(accum[56], s_v, blo[2]);
            float _fma_105 = __fmaf_rn(accum[57], s_v, bhi[2]);
            float _fma_106 = __fmaf_rn(acc_fin[56], alpha_v, _fma_104);
            float _fma_107 = __fmaf_rn(acc_fin[57], alpha_v, _fma_105);
            __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_fma_106, _fma_107));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_26)[0];
            float _fma_108 = __fmaf_rn(accum[60], s_v, blo[3]);
            float _fma_109 = __fmaf_rn(accum[61], s_v, bhi[3]);
            float _fma_110 = __fmaf_rn(acc_fin[60], alpha_v, _fma_108);
            float _fma_111 = __fmaf_rn(acc_fin[61], alpha_v, _fma_109);
            __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_fma_110, _fma_111));
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
                if (col_v_42 < N) {
                    reinterpret_cast<int4*>(out + (row_lo * N + col_v_42))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            float _fma_112 = __fmaf_rn(accum[50], s_v, blo[0]);
            float _fma_113 = __fmaf_rn(accum[51], s_v, bhi[0]);
            float _fma_114 = __fmaf_rn(acc_fin[50], alpha_v, _fma_112);
            float _fma_115 = __fmaf_rn(acc_fin[51], alpha_v, _fma_113);
            __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(_fma_114, _fma_115));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_28)[0];
            float _fma_116 = __fmaf_rn(accum[54], s_v, blo[1]);
            float _fma_117 = __fmaf_rn(accum[55], s_v, bhi[1]);
            float _fma_118 = __fmaf_rn(acc_fin[54], alpha_v, _fma_116);
            float _fma_119 = __fmaf_rn(acc_fin[55], alpha_v, _fma_117);
            __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(_fma_118, _fma_119));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_29)[0];
            float _fma_120 = __fmaf_rn(accum[58], s_v, blo[2]);
            float _fma_121 = __fmaf_rn(accum[59], s_v, bhi[2]);
            float _fma_122 = __fmaf_rn(acc_fin[58], alpha_v, _fma_120);
            float _fma_123 = __fmaf_rn(acc_fin[59], alpha_v, _fma_121);
            __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(_fma_122, _fma_123));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_30)[0];
            float _fma_124 = __fmaf_rn(accum[62], s_v, blo[3]);
            float _fma_125 = __fmaf_rn(accum[63], s_v, bhi[3]);
            float _fma_126 = __fmaf_rn(acc_fin[62], alpha_v, _fma_124);
            float _fma_127 = __fmaf_rn(acc_fin[63], alpha_v, _fma_125);
            __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(_fma_126, _fma_127));
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
                if (col_v_42 < N) {
                    reinterpret_cast<int4*>(out + (row_hi * N + col_v_42))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            blo[0] = 0.0f;
            bhi[0] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 128;
                if (bcol0 < N) {
                    unsigned int _vec_load_16[2];
                    {
                        uint32_t _bf16x2_bits_28;
                        _bf16x2_bits_28 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_16[0] = _bf16x2_bits_28;
                    }
                    bias_word = _vec_load_16[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[0] = b_lo_f;
                    bhi[0] = b_hi_f;
                }
            }
            blo[1] = 0.0f;
            bhi[1] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 136;
                if (bcol0 < N) {
                    unsigned int _vec_load_17[2];
                    {
                        uint32_t _bf16x2_bits_29;
                        _bf16x2_bits_29 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_17[0] = _bf16x2_bits_29;
                    }
                    bias_word = _vec_load_17[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[1] = b_lo_f;
                    bhi[1] = b_hi_f;
                }
            }
            blo[2] = 0.0f;
            bhi[2] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 144;
                if (bcol0 < N) {
                    unsigned int _vec_load_18[2];
                    {
                        uint32_t _bf16x2_bits_30;
                        _bf16x2_bits_30 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_18[0] = _bf16x2_bits_30;
                    }
                    bias_word = _vec_load_18[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[2] = b_lo_f;
                    bhi[2] = b_hi_f;
                }
            }
            blo[3] = 0.0f;
            bhi[3] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 152;
                if (bcol0 < N) {
                    unsigned int _vec_load_19[2];
                    {
                        uint32_t _bf16x2_bits_31;
                        _bf16x2_bits_31 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_19[0] = _bf16x2_bits_31;
                    }
                    bias_word = _vec_load_19[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[3] = b_lo_f;
                    bhi[3] = b_hi_f;
                }
            }
            unsigned int col_v_59 = n0_c + (16 + q_lane) * 8;
            float _fma_128 = __fmaf_rn(accum[64], s_v, blo[0]);
            float _fma_129 = __fmaf_rn(accum[65], s_v, bhi[0]);
            float _fma_130 = __fmaf_rn(acc_fin[64], alpha_v, _fma_128);
            float _fma_131 = __fmaf_rn(acc_fin[65], alpha_v, _fma_129);
            __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(_fma_130, _fma_131));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_32)[0];
            float _fma_132 = __fmaf_rn(accum[68], s_v, blo[1]);
            float _fma_133 = __fmaf_rn(accum[69], s_v, bhi[1]);
            float _fma_134 = __fmaf_rn(acc_fin[68], alpha_v, _fma_132);
            float _fma_135 = __fmaf_rn(acc_fin[69], alpha_v, _fma_133);
            __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(_fma_134, _fma_135));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_33)[0];
            float _fma_136 = __fmaf_rn(accum[72], s_v, blo[2]);
            float _fma_137 = __fmaf_rn(accum[73], s_v, bhi[2]);
            float _fma_138 = __fmaf_rn(acc_fin[72], alpha_v, _fma_136);
            float _fma_139 = __fmaf_rn(acc_fin[73], alpha_v, _fma_137);
            __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(_fma_138, _fma_139));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_34)[0];
            float _fma_140 = __fmaf_rn(accum[76], s_v, blo[3]);
            float _fma_141 = __fmaf_rn(accum[77], s_v, bhi[3]);
            float _fma_142 = __fmaf_rn(acc_fin[76], alpha_v, _fma_140);
            float _fma_143 = __fmaf_rn(acc_fin[77], alpha_v, _fma_141);
            __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(_fma_142, _fma_143));
            pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_35)[0];
            unsigned int x0_60 = pk[1] & m_even | pk[0] & m_odd;
            unsigned int x1_61 = pk[3] & m_even | pk[2] & m_odd;
            unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, x0_60, 1);
            unsigned int y0_62 = _shfl_xor_32;
            unsigned int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, x1_61, 1);
            unsigned int y1_63 = _shfl_xor_33;
            pk[0] = y0_62 & m_odd | pk[0] & m_even;
            pk[1] = y0_62 & m_even | pk[1] & m_odd;
            pk[2] = y1_63 & m_odd | pk[2] & m_even;
            pk[3] = y1_63 & m_even | pk[3] & m_odd;
            unsigned int x2_64 = pk[2] & m_lo | pk[0] & m_hi;
            unsigned int x3_65 = pk[3] & m_lo | pk[1] & m_hi;
            unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, x2_64, 2);
            unsigned int y2_66 = _shfl_xor_34;
            unsigned int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, x3_65, 2);
            unsigned int y3_67 = _shfl_xor_35;
            pk[0] = y2_66 & m_hi | pk[0] & m_lo;
            pk[2] = y2_66 & m_lo | pk[2] & m_hi;
            pk[1] = y3_67 & m_hi | pk[1] & m_lo;
            pk[3] = y3_67 & m_lo | pk[3] & m_hi;
            if (row_lo < row_lim_c) {
                if (col_v_59 < N) {
                    reinterpret_cast<int4*>(out + (row_lo * N + col_v_59))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            float _fma_144 = __fmaf_rn(accum[66], s_v, blo[0]);
            float _fma_145 = __fmaf_rn(accum[67], s_v, bhi[0]);
            float _fma_146 = __fmaf_rn(acc_fin[66], alpha_v, _fma_144);
            float _fma_147 = __fmaf_rn(acc_fin[67], alpha_v, _fma_145);
            __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(_fma_146, _fma_147));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_36)[0];
            float _fma_148 = __fmaf_rn(accum[70], s_v, blo[1]);
            float _fma_149 = __fmaf_rn(accum[71], s_v, bhi[1]);
            float _fma_150 = __fmaf_rn(acc_fin[70], alpha_v, _fma_148);
            float _fma_151 = __fmaf_rn(acc_fin[71], alpha_v, _fma_149);
            __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(_fma_150, _fma_151));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_37)[0];
            float _fma_152 = __fmaf_rn(accum[74], s_v, blo[2]);
            float _fma_153 = __fmaf_rn(accum[75], s_v, bhi[2]);
            float _fma_154 = __fmaf_rn(acc_fin[74], alpha_v, _fma_152);
            float _fma_155 = __fmaf_rn(acc_fin[75], alpha_v, _fma_153);
            __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(_fma_154, _fma_155));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_38)[0];
            float _fma_156 = __fmaf_rn(accum[78], s_v, blo[3]);
            float _fma_157 = __fmaf_rn(accum[79], s_v, bhi[3]);
            float _fma_158 = __fmaf_rn(acc_fin[78], alpha_v, _fma_156);
            float _fma_159 = __fmaf_rn(acc_fin[79], alpha_v, _fma_157);
            __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(_fma_158, _fma_159));
            pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_39)[0];
            unsigned int x0_68 = pk[1] & m_even | pk[0] & m_odd;
            unsigned int x1_69 = pk[3] & m_even | pk[2] & m_odd;
            unsigned int _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, x0_68, 1);
            unsigned int y0_70 = _shfl_xor_36;
            unsigned int _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, x1_69, 1);
            unsigned int y1_71 = _shfl_xor_37;
            pk[0] = y0_70 & m_odd | pk[0] & m_even;
            pk[1] = y0_70 & m_even | pk[1] & m_odd;
            pk[2] = y1_71 & m_odd | pk[2] & m_even;
            pk[3] = y1_71 & m_even | pk[3] & m_odd;
            unsigned int x2_72 = pk[2] & m_lo | pk[0] & m_hi;
            unsigned int x3_73 = pk[3] & m_lo | pk[1] & m_hi;
            unsigned int _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, x2_72, 2);
            unsigned int y2_74 = _shfl_xor_38;
            unsigned int _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, x3_73, 2);
            unsigned int y3_75 = _shfl_xor_39;
            pk[0] = y2_74 & m_hi | pk[0] & m_lo;
            pk[2] = y2_74 & m_lo | pk[2] & m_hi;
            pk[1] = y3_75 & m_hi | pk[1] & m_lo;
            pk[3] = y3_75 & m_lo | pk[3] & m_hi;
            if (row_hi < row_lim_c) {
                if (col_v_59 < N) {
                    reinterpret_cast<int4*>(out + (row_hi * N + col_v_59))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            blo[0] = 0.0f;
            bhi[0] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 160;
                if (bcol0 < N) {
                    unsigned int _vec_load_20[2];
                    {
                        uint32_t _bf16x2_bits_32;
                        _bf16x2_bits_32 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_20[0] = _bf16x2_bits_32;
                    }
                    bias_word = _vec_load_20[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[0] = b_lo_f;
                    bhi[0] = b_hi_f;
                }
            }
            blo[1] = 0.0f;
            bhi[1] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 168;
                if (bcol0 < N) {
                    unsigned int _vec_load_21[2];
                    {
                        uint32_t _bf16x2_bits_33;
                        _bf16x2_bits_33 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_21[0] = _bf16x2_bits_33;
                    }
                    bias_word = _vec_load_21[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[1] = b_lo_f;
                    bhi[1] = b_hi_f;
                }
            }
            blo[2] = 0.0f;
            bhi[2] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 176;
                if (bcol0 < N) {
                    unsigned int _vec_load_22[2];
                    {
                        uint32_t _bf16x2_bits_34;
                        _bf16x2_bits_34 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_22[0] = _bf16x2_bits_34;
                    }
                    bias_word = _vec_load_22[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[2] = b_lo_f;
                    bhi[2] = b_hi_f;
                }
            }
            blo[3] = 0.0f;
            bhi[3] = 0.0f;
            if (has_bias != 0) {
                bcol0 = n0_c + 184;
                if (bcol0 < N) {
                    unsigned int _vec_load_23[2];
                    {
                        uint32_t _bf16x2_bits_35;
                        _bf16x2_bits_35 = *reinterpret_cast<const uint32_t*>(bias + bcol0 + lane_col);
                        _vec_load_23[0] = _bf16x2_bits_35;
                    }
                    bias_word = _vec_load_23[0];
                    bw_lo = bias_word << 16;
                    bw_hi = bias_word & 4294901760u;
                    b_lo_f = reinterpret_cast<float*>(&bw_lo)[0];
                    b_hi_f = reinterpret_cast<float*>(&bw_hi)[0];
                    blo[3] = b_lo_f;
                    bhi[3] = b_hi_f;
                }
            }
            unsigned int col_v_76 = n0_c + (20 + q_lane) * 8;
            float _fma_160 = __fmaf_rn(accum[80], s_v, blo[0]);
            float _fma_161 = __fmaf_rn(accum[81], s_v, bhi[0]);
            float _fma_162 = __fmaf_rn(acc_fin[80], alpha_v, _fma_160);
            float _fma_163 = __fmaf_rn(acc_fin[81], alpha_v, _fma_161);
            __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(_fma_162, _fma_163));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_40)[0];
            float _fma_164 = __fmaf_rn(accum[84], s_v, blo[1]);
            float _fma_165 = __fmaf_rn(accum[85], s_v, bhi[1]);
            float _fma_166 = __fmaf_rn(acc_fin[84], alpha_v, _fma_164);
            float _fma_167 = __fmaf_rn(acc_fin[85], alpha_v, _fma_165);
            __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(_fma_166, _fma_167));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_41)[0];
            float _fma_168 = __fmaf_rn(accum[88], s_v, blo[2]);
            float _fma_169 = __fmaf_rn(accum[89], s_v, bhi[2]);
            float _fma_170 = __fmaf_rn(acc_fin[88], alpha_v, _fma_168);
            float _fma_171 = __fmaf_rn(acc_fin[89], alpha_v, _fma_169);
            __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(_fma_170, _fma_171));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_42)[0];
            float _fma_172 = __fmaf_rn(accum[92], s_v, blo[3]);
            float _fma_173 = __fmaf_rn(accum[93], s_v, bhi[3]);
            float _fma_174 = __fmaf_rn(acc_fin[92], alpha_v, _fma_172);
            float _fma_175 = __fmaf_rn(acc_fin[93], alpha_v, _fma_173);
            __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(_fma_174, _fma_175));
            pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_43)[0];
            unsigned int x0_77 = pk[1] & m_even | pk[0] & m_odd;
            unsigned int x1_78 = pk[3] & m_even | pk[2] & m_odd;
            unsigned int _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, x0_77, 1);
            unsigned int y0_79 = _shfl_xor_40;
            unsigned int _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, x1_78, 1);
            unsigned int y1_80 = _shfl_xor_41;
            pk[0] = y0_79 & m_odd | pk[0] & m_even;
            pk[1] = y0_79 & m_even | pk[1] & m_odd;
            pk[2] = y1_80 & m_odd | pk[2] & m_even;
            pk[3] = y1_80 & m_even | pk[3] & m_odd;
            unsigned int x2_81 = pk[2] & m_lo | pk[0] & m_hi;
            unsigned int x3_82 = pk[3] & m_lo | pk[1] & m_hi;
            unsigned int _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, x2_81, 2);
            unsigned int y2_83 = _shfl_xor_42;
            unsigned int _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, x3_82, 2);
            unsigned int y3_84 = _shfl_xor_43;
            pk[0] = y2_83 & m_hi | pk[0] & m_lo;
            pk[2] = y2_83 & m_lo | pk[2] & m_hi;
            pk[1] = y3_84 & m_hi | pk[1] & m_lo;
            pk[3] = y3_84 & m_lo | pk[3] & m_hi;
            if (row_lo < row_lim_c) {
                if (col_v_76 < N) {
                    reinterpret_cast<int4*>(out + (row_lo * N + col_v_76))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
            float _fma_176 = __fmaf_rn(accum[82], s_v, blo[0]);
            float _fma_177 = __fmaf_rn(accum[83], s_v, bhi[0]);
            float _fma_178 = __fmaf_rn(acc_fin[82], alpha_v, _fma_176);
            float _fma_179 = __fmaf_rn(acc_fin[83], alpha_v, _fma_177);
            __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(_fma_178, _fma_179));
            pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_44)[0];
            float _fma_180 = __fmaf_rn(accum[86], s_v, blo[1]);
            float _fma_181 = __fmaf_rn(accum[87], s_v, bhi[1]);
            float _fma_182 = __fmaf_rn(acc_fin[86], alpha_v, _fma_180);
            float _fma_183 = __fmaf_rn(acc_fin[87], alpha_v, _fma_181);
            __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(_fma_182, _fma_183));
            pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_45)[0];
            float _fma_184 = __fmaf_rn(accum[90], s_v, blo[2]);
            float _fma_185 = __fmaf_rn(accum[91], s_v, bhi[2]);
            float _fma_186 = __fmaf_rn(acc_fin[90], alpha_v, _fma_184);
            float _fma_187 = __fmaf_rn(acc_fin[91], alpha_v, _fma_185);
            __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(_fma_186, _fma_187));
            pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_46)[0];
            float _fma_188 = __fmaf_rn(accum[94], s_v, blo[3]);
            float _fma_189 = __fmaf_rn(accum[95], s_v, bhi[3]);
            float _fma_190 = __fmaf_rn(acc_fin[94], alpha_v, _fma_188);
            float _fma_191 = __fmaf_rn(acc_fin[95], alpha_v, _fma_189);
            __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(_fma_190, _fma_191));
            pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_47)[0];
            unsigned int x0_85 = pk[1] & m_even | pk[0] & m_odd;
            unsigned int x1_86 = pk[3] & m_even | pk[2] & m_odd;
            unsigned int _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, x0_85, 1);
            unsigned int y0_87 = _shfl_xor_44;
            unsigned int _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, x1_86, 1);
            unsigned int y1_88 = _shfl_xor_45;
            pk[0] = y0_87 & m_odd | pk[0] & m_even;
            pk[1] = y0_87 & m_even | pk[1] & m_odd;
            pk[2] = y1_88 & m_odd | pk[2] & m_even;
            pk[3] = y1_88 & m_even | pk[3] & m_odd;
            unsigned int x2_89 = pk[2] & m_lo | pk[0] & m_hi;
            unsigned int x3_90 = pk[3] & m_lo | pk[1] & m_hi;
            unsigned int _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, x2_89, 2);
            unsigned int y2_91 = _shfl_xor_46;
            unsigned int _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, x3_90, 2);
            unsigned int y3_92 = _shfl_xor_47;
            pk[0] = y2_91 & m_hi | pk[0] & m_lo;
            pk[2] = y2_91 & m_lo | pk[2] & m_hi;
            pk[1] = y3_92 & m_hi | pk[1] & m_lo;
            pk[3] = y3_92 & m_lo | pk[3] & m_hi;
            if (row_hi < row_lim_c) {
                if (col_v_76 < N) {
                    reinterpret_cast<int4*>(out + (row_hi * N + col_v_76))[0] = reinterpret_cast<int4*>(pk)[0];
                }
            }
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Cleanup
}

} // extern "C"
