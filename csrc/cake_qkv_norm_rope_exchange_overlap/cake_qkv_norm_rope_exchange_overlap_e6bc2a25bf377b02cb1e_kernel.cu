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
#define SMEM_TOTAL 196672
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



__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_qkv_norm_rope_exchange_overlap_e6bc2a25bf377b02cb1e(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ cos_sin, __nv_bfloat16* __restrict__ q_scale, __nv_bfloat16* __restrict__ k_scale, long long* __restrict__ tiles, long long* __restrict__ dst_ptrs, long long* __restrict__ ready_ptrs, unsigned int* __restrict__ counters, int num_tiles, int M, int K, int S_local, int head_stride, long long batch_stride, long long slot_elem_offset, int pad_base, int num_words, int rank, int world, unsigned int ready_target)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
        }
    }
    __syncwarp();

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[196608..196672)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // Stage-major initialization across barrier declarations.
            mbarrier_init(smem + 196608, 1);
            mbarrier_init(smem + 196640, 256);
            mbarrier_init(smem + 196616, 1);
            mbarrier_init(smem + 196648, 256);
            mbarrier_init(smem + 196624, 1);
            mbarrier_init(smem + 196656, 256);
            mbarrier_init(smem + 196632, 1);
            mbarrier_init(smem + 196664, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }
    __syncthreads();
    // Inc phase consumes the registers released above.
    if (warp >= 0 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
    }
    __syncthreads();

    // ---- Role: producer ----
    if (warp >= 8 && warp <= 11) {
        { // producer_main
            unsigned int total_tiles_p = (unsigned int)num_tiles;
            unsigned int num_k_stages_p = (unsigned int)K / 64;
            if (warp == 8) {
                if (elect_sync()) {
                    unsigned int it_p = 0;
                    #pragma unroll 1
                    for (unsigned int tile_p = bid; tile_p < total_tiles_p; tile_p += num_bids) {
                        unsigned int mt_p = (unsigned int)tiles[tile_p * 8];
                        unsigned int n0_p = (unsigned int)tiles[tile_p * 8 + 1];
                        for (unsigned int k_p = 0; k_p < num_k_stages_p; k_p++) {
                            unsigned int stage_p = it_p % 4;
                            mbarrier_wait_hint(empty_addr + (stage_p) * 8, it_p / 4 + 1 & 1, 10000000);
                            {
                                const uint64_t _tma_operand_pack_0_desc = (uint64_t)((&A));
                                const uint32_t _tma_operand_pack_0_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                const uint32_t _tma_operand_pack_0_dst = (uint32_t)(smem_a_addr + stage_p * 49152);
                                const int32_t _tma_operand_pack_0_coord_0 = (int32_t)(k_p * 64);
                                const int32_t _tma_operand_pack_0_coord_1 = (int32_t)(mt_p * 128);
                                const uint64_t _tma_operand_pack_0_cache = (uint64_t)(0x1000000000000000ULL);
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%3, %4}], [%2], %5;"
                                    :: "r"(_tma_operand_pack_0_dst), "l"(_tma_operand_pack_0_desc), "r"(_tma_operand_pack_0_mbar),
                                       "r"(_tma_operand_pack_0_coord_0), "r"(_tma_operand_pack_0_coord_1), "l"(_tma_operand_pack_0_cache) : "memory");
                            }
                            {
                                const uint64_t _tma_operand_pack_1_desc = (uint64_t)((&B));
                                const uint32_t _tma_operand_pack_1_mbar = (uint32_t)(full_addr + (stage_p) * 8);
                                const uint32_t _tma_operand_pack_1_dst = (uint32_t)(smem_b_addr + stage_p * 49152);
                                const int32_t _tma_operand_pack_1_coord_0 = (int32_t)(k_p * 64);
                                const int32_t _tma_operand_pack_1_coord_1 = (int32_t)(n0_p);
                                const uint64_t _tma_operand_pack_1_cache = (uint64_t)(0x1000000000000000ULL);
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%3, %4}], [%2], %5;"
                                    :: "r"(_tma_operand_pack_1_dst), "l"(_tma_operand_pack_1_desc), "r"(_tma_operand_pack_1_mbar),
                                       "r"(_tma_operand_pack_1_coord_0), "r"(_tma_operand_pack_1_coord_1), "l"(_tma_operand_pack_1_cache) : "memory");
                            }
                            mbarrier_arrive_expect_tx(full_addr + (stage_p) * 8, 49152);
                            it_p = it_p + 1;
                        }
                    }
                }
            }
        }
    // ---- Role: math ----
    } else if (warp <= 7) {
        { // math_main
            unsigned int total_tiles_c = (unsigned int)num_tiles;
            unsigned int num_k_stages_c = (unsigned int)K / 64;
            unsigned int rows_m = (unsigned int)M;
            unsigned int s_local = (unsigned int)S_local;
            long long head_stride_e = (long long)head_stride;
            unsigned int words_per_dest = (unsigned int)num_words;
            unsigned int pad_base_u = (unsigned int)pad_base;
            uint32_t warp_group_idx = static_cast<uint32_t>(tid) / 128u;
            unsigned int math_wg_idx = make_warp_uniform(warp_group_idx);
            float accum[128] = {0};
            unsigned int pk[4] = {0};
            unsigned int it_c = 0;
            unsigned int warp_in_wg = (unsigned int)warp % 4;
            unsigned int lane_row = lane / 4;
            unsigned int q_lane = lane % 4;
            unsigned int m_odd = -(q_lane & 1);
            unsigned int m_even = m_odd ^ 4294967295u;
            unsigned int m_hi = -(q_lane >> 1 & 1);
            unsigned int m_lo = m_hi ^ 4294967295u;
            #pragma unroll 1
            for (unsigned int tile_c = bid; tile_c < total_tiles_c; tile_c += num_bids) {
                unsigned int desc_c = tile_c * 8;
                unsigned int mt_c = (unsigned int)tiles[desc_c];
                unsigned int dest_c = (unsigned int)tiles[desc_c + 2];
                unsigned int norm_kind = (unsigned int)tiles[desc_c + 3];
                long long base_elem = tiles[desc_c + 4];
                long long row_stride = tiles[desc_c + 5];
                unsigned int word_c = (unsigned int)tiles[desc_c + 6];
                unsigned int last_ticket = (unsigned int)tiles[desc_c + 7];
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
                for (unsigned int k_c = 0; k_c < num_k_stages_c; k_c++) {
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
                    mbarrier_arrive(empty_addr + (stage_c) * 8);
                    it_c = it_c + 1;
                }
                unsigned int local_lo = mt_c * 128 + math_wg_idx * 64 + warp_in_wg * 16 + lane_row;
                unsigned int local_hi = local_lo + 8;
                unsigned int safe_lo = ((local_lo < rows_m) ? local_lo : rows_m - 1);
                unsigned int safe_hi = ((local_hi < rows_m) ? local_hi : rows_m - 1);
                unsigned int b_lo = safe_lo / s_local;
                unsigned int b_hi = safe_hi / s_local;
                unsigned int s_lo = safe_lo - b_lo * s_local;
                unsigned int s_hi = safe_hi - b_hi * s_local;
                long long row_off_lo = slot_elem_offset + base_elem + (long long)b_lo * batch_stride + (long long)s_lo * row_stride;
                long long row_off_hi = slot_elem_offset + base_elem + (long long)b_hi * batch_stride + (long long)s_hi * row_stride;
                unsigned int cs_lo = safe_lo * 128;
                unsigned int cs_hi = safe_hi * 128;
                if (norm_kind != 0) {
                    float sumsq = 0.0f;
                    sumsq += accum[0] * accum[0];
                    sumsq += accum[1] * accum[1];
                    sumsq += accum[4] * accum[4];
                    sumsq += accum[5] * accum[5];
                    sumsq += accum[8] * accum[8];
                    sumsq += accum[9] * accum[9];
                    sumsq += accum[12] * accum[12];
                    sumsq += accum[13] * accum[13];
                    sumsq += accum[16] * accum[16];
                    sumsq += accum[17] * accum[17];
                    sumsq += accum[20] * accum[20];
                    sumsq += accum[21] * accum[21];
                    sumsq += accum[24] * accum[24];
                    sumsq += accum[25] * accum[25];
                    sumsq += accum[28] * accum[28];
                    sumsq += accum[29] * accum[29];
                    sumsq += accum[32] * accum[32];
                    sumsq += accum[33] * accum[33];
                    sumsq += accum[36] * accum[36];
                    sumsq += accum[37] * accum[37];
                    sumsq += accum[40] * accum[40];
                    sumsq += accum[41] * accum[41];
                    sumsq += accum[44] * accum[44];
                    sumsq += accum[45] * accum[45];
                    sumsq += accum[48] * accum[48];
                    sumsq += accum[49] * accum[49];
                    sumsq += accum[52] * accum[52];
                    sumsq += accum[53] * accum[53];
                    sumsq += accum[56] * accum[56];
                    sumsq += accum[57] * accum[57];
                    sumsq += accum[60] * accum[60];
                    sumsq += accum[61] * accum[61];
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, sumsq, 1);
                    float sq1 = _shfl_xor_0;
                    sumsq += sq1;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, sumsq, 2);
                    float sq2 = _shfl_xor_1;
                    sumsq += sq2;
                    float _rsqrt_0 = rsqrtf(sumsq * 0.0078125f + 1e-06f);
                    float rrms = _rsqrt_0;
                    unsigned int d_col = 2 * q_lane;
                    unsigned int pair_k = q_lane;
                    float _vec_load_0[2];
                    {
                        uint32_t _bf16x2_bits_0;
                        _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col : k_scale + d_col) + 0);
                        (&_vec_load_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_0) << 16);
                        (&_vec_load_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_0) & 0xffff0000u);
                    }
                    float cos_v = cos_sin[cs_lo + pair_k];
                    float sin_v = cos_sin[cs_lo + 64 + pair_k];
                    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(accum[0] * rrms);
                    float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                    float xn0 = _cvt_f32_0;
                    __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(accum[1] * rrms);
                    float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                    float xn1 = _cvt_f32_1;
                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(xn0 * _vec_load_0[0]);
                    float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                    float xs0 = _cvt_f32_2;
                    __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(xn1 * _vec_load_0[1]);
                    float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                    float xs1 = _cvt_f32_3;
                    accum[0] = xs0 * cos_v - xs1 * sin_v;
                    accum[1] = xs1 * cos_v + xs0 * sin_v;
                    unsigned int d_col_0 = 8 + 2 * q_lane;
                    unsigned int pair_k_1 = 4 + q_lane;
                    float _vec_load_1[2];
                    {
                        uint32_t _bf16x2_bits_1;
                        _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_0 : k_scale + d_col_0) + 0);
                        (&_vec_load_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) << 16);
                        (&_vec_load_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) & 0xffff0000u);
                    }
                    float cos_v_2 = cos_sin[cs_lo + pair_k_1];
                    float sin_v_3 = cos_sin[cs_lo + 64 + pair_k_1];
                    __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(accum[4] * rrms);
                    float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                    float xn0_4 = _cvt_f32_4;
                    __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(accum[5] * rrms);
                    float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                    float xn1_5 = _cvt_f32_5;
                    __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(xn0_4 * _vec_load_1[0]);
                    float _cvt_f32_6 = __bfloat162float(_cvt_bf16_6);
                    float xs0_6 = _cvt_f32_6;
                    __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(xn1_5 * _vec_load_1[1]);
                    float _cvt_f32_7 = __bfloat162float(_cvt_bf16_7);
                    float xs1_7 = _cvt_f32_7;
                    accum[4] = xs0_6 * cos_v_2 - xs1_7 * sin_v_3;
                    accum[5] = xs1_7 * cos_v_2 + xs0_6 * sin_v_3;
                    unsigned int d_col_8 = 16 + 2 * q_lane;
                    unsigned int pair_k_9 = 8 + q_lane;
                    float _vec_load_2[2];
                    {
                        uint32_t _bf16x2_bits_2;
                        _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_8 : k_scale + d_col_8) + 0);
                        (&_vec_load_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) << 16);
                        (&_vec_load_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) & 0xffff0000u);
                    }
                    float cos_v_10 = cos_sin[cs_lo + pair_k_9];
                    float sin_v_11 = cos_sin[cs_lo + 64 + pair_k_9];
                    __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(accum[8] * rrms);
                    float _cvt_f32_8 = __bfloat162float(_cvt_bf16_8);
                    float xn0_12 = _cvt_f32_8;
                    __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(accum[9] * rrms);
                    float _cvt_f32_9 = __bfloat162float(_cvt_bf16_9);
                    float xn1_13 = _cvt_f32_9;
                    __nv_bfloat16 _cvt_bf16_10 = __float2bfloat16(xn0_12 * _vec_load_2[0]);
                    float _cvt_f32_10 = __bfloat162float(_cvt_bf16_10);
                    float xs0_14 = _cvt_f32_10;
                    __nv_bfloat16 _cvt_bf16_11 = __float2bfloat16(xn1_13 * _vec_load_2[1]);
                    float _cvt_f32_11 = __bfloat162float(_cvt_bf16_11);
                    float xs1_15 = _cvt_f32_11;
                    accum[8] = xs0_14 * cos_v_10 - xs1_15 * sin_v_11;
                    accum[9] = xs1_15 * cos_v_10 + xs0_14 * sin_v_11;
                    unsigned int d_col_16 = 24 + 2 * q_lane;
                    unsigned int pair_k_17 = 12 + q_lane;
                    float _vec_load_3[2];
                    {
                        uint32_t _bf16x2_bits_3;
                        _bf16x2_bits_3 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_16 : k_scale + d_col_16) + 0);
                        (&_vec_load_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_3) << 16);
                        (&_vec_load_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_3) & 0xffff0000u);
                    }
                    float cos_v_18 = cos_sin[cs_lo + pair_k_17];
                    float sin_v_19 = cos_sin[cs_lo + 64 + pair_k_17];
                    __nv_bfloat16 _cvt_bf16_12 = __float2bfloat16(accum[12] * rrms);
                    float _cvt_f32_12 = __bfloat162float(_cvt_bf16_12);
                    float xn0_20 = _cvt_f32_12;
                    __nv_bfloat16 _cvt_bf16_13 = __float2bfloat16(accum[13] * rrms);
                    float _cvt_f32_13 = __bfloat162float(_cvt_bf16_13);
                    float xn1_21 = _cvt_f32_13;
                    __nv_bfloat16 _cvt_bf16_14 = __float2bfloat16(xn0_20 * _vec_load_3[0]);
                    float _cvt_f32_14 = __bfloat162float(_cvt_bf16_14);
                    float xs0_22 = _cvt_f32_14;
                    __nv_bfloat16 _cvt_bf16_15 = __float2bfloat16(xn1_21 * _vec_load_3[1]);
                    float _cvt_f32_15 = __bfloat162float(_cvt_bf16_15);
                    float xs1_23 = _cvt_f32_15;
                    accum[12] = xs0_22 * cos_v_18 - xs1_23 * sin_v_19;
                    accum[13] = xs1_23 * cos_v_18 + xs0_22 * sin_v_19;
                    unsigned int d_col_24 = 32 + 2 * q_lane;
                    unsigned int pair_k_25 = 16 + q_lane;
                    float _vec_load_4[2];
                    {
                        uint32_t _bf16x2_bits_4;
                        _bf16x2_bits_4 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_24 : k_scale + d_col_24) + 0);
                        (&_vec_load_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_4) << 16);
                        (&_vec_load_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_4) & 0xffff0000u);
                    }
                    float cos_v_26 = cos_sin[cs_lo + pair_k_25];
                    float sin_v_27 = cos_sin[cs_lo + 64 + pair_k_25];
                    __nv_bfloat16 _cvt_bf16_16 = __float2bfloat16(accum[16] * rrms);
                    float _cvt_f32_16 = __bfloat162float(_cvt_bf16_16);
                    float xn0_28 = _cvt_f32_16;
                    __nv_bfloat16 _cvt_bf16_17 = __float2bfloat16(accum[17] * rrms);
                    float _cvt_f32_17 = __bfloat162float(_cvt_bf16_17);
                    float xn1_29 = _cvt_f32_17;
                    __nv_bfloat16 _cvt_bf16_18 = __float2bfloat16(xn0_28 * _vec_load_4[0]);
                    float _cvt_f32_18 = __bfloat162float(_cvt_bf16_18);
                    float xs0_30 = _cvt_f32_18;
                    __nv_bfloat16 _cvt_bf16_19 = __float2bfloat16(xn1_29 * _vec_load_4[1]);
                    float _cvt_f32_19 = __bfloat162float(_cvt_bf16_19);
                    float xs1_31 = _cvt_f32_19;
                    accum[16] = xs0_30 * cos_v_26 - xs1_31 * sin_v_27;
                    accum[17] = xs1_31 * cos_v_26 + xs0_30 * sin_v_27;
                    unsigned int d_col_32 = 40 + 2 * q_lane;
                    unsigned int pair_k_33 = 20 + q_lane;
                    float _vec_load_5[2];
                    {
                        uint32_t _bf16x2_bits_5;
                        _bf16x2_bits_5 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_32 : k_scale + d_col_32) + 0);
                        (&_vec_load_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_5) << 16);
                        (&_vec_load_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_5) & 0xffff0000u);
                    }
                    float cos_v_34 = cos_sin[cs_lo + pair_k_33];
                    float sin_v_35 = cos_sin[cs_lo + 64 + pair_k_33];
                    __nv_bfloat16 _cvt_bf16_20 = __float2bfloat16(accum[20] * rrms);
                    float _cvt_f32_20 = __bfloat162float(_cvt_bf16_20);
                    float xn0_36 = _cvt_f32_20;
                    __nv_bfloat16 _cvt_bf16_21 = __float2bfloat16(accum[21] * rrms);
                    float _cvt_f32_21 = __bfloat162float(_cvt_bf16_21);
                    float xn1_37 = _cvt_f32_21;
                    __nv_bfloat16 _cvt_bf16_22 = __float2bfloat16(xn0_36 * _vec_load_5[0]);
                    float _cvt_f32_22 = __bfloat162float(_cvt_bf16_22);
                    float xs0_38 = _cvt_f32_22;
                    __nv_bfloat16 _cvt_bf16_23 = __float2bfloat16(xn1_37 * _vec_load_5[1]);
                    float _cvt_f32_23 = __bfloat162float(_cvt_bf16_23);
                    float xs1_39 = _cvt_f32_23;
                    accum[20] = xs0_38 * cos_v_34 - xs1_39 * sin_v_35;
                    accum[21] = xs1_39 * cos_v_34 + xs0_38 * sin_v_35;
                    unsigned int d_col_40 = 48 + 2 * q_lane;
                    unsigned int pair_k_41 = 24 + q_lane;
                    float _vec_load_6[2];
                    {
                        uint32_t _bf16x2_bits_6;
                        _bf16x2_bits_6 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_40 : k_scale + d_col_40) + 0);
                        (&_vec_load_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_6) << 16);
                        (&_vec_load_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_6) & 0xffff0000u);
                    }
                    float cos_v_42 = cos_sin[cs_lo + pair_k_41];
                    float sin_v_43 = cos_sin[cs_lo + 64 + pair_k_41];
                    __nv_bfloat16 _cvt_bf16_24 = __float2bfloat16(accum[24] * rrms);
                    float _cvt_f32_24 = __bfloat162float(_cvt_bf16_24);
                    float xn0_44 = _cvt_f32_24;
                    __nv_bfloat16 _cvt_bf16_25 = __float2bfloat16(accum[25] * rrms);
                    float _cvt_f32_25 = __bfloat162float(_cvt_bf16_25);
                    float xn1_45 = _cvt_f32_25;
                    __nv_bfloat16 _cvt_bf16_26 = __float2bfloat16(xn0_44 * _vec_load_6[0]);
                    float _cvt_f32_26 = __bfloat162float(_cvt_bf16_26);
                    float xs0_46 = _cvt_f32_26;
                    __nv_bfloat16 _cvt_bf16_27 = __float2bfloat16(xn1_45 * _vec_load_6[1]);
                    float _cvt_f32_27 = __bfloat162float(_cvt_bf16_27);
                    float xs1_47 = _cvt_f32_27;
                    accum[24] = xs0_46 * cos_v_42 - xs1_47 * sin_v_43;
                    accum[25] = xs1_47 * cos_v_42 + xs0_46 * sin_v_43;
                    unsigned int d_col_48 = 56 + 2 * q_lane;
                    unsigned int pair_k_49 = 28 + q_lane;
                    float _vec_load_7[2];
                    {
                        uint32_t _bf16x2_bits_7;
                        _bf16x2_bits_7 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_48 : k_scale + d_col_48) + 0);
                        (&_vec_load_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_7) << 16);
                        (&_vec_load_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_7) & 0xffff0000u);
                    }
                    float cos_v_50 = cos_sin[cs_lo + pair_k_49];
                    float sin_v_51 = cos_sin[cs_lo + 64 + pair_k_49];
                    __nv_bfloat16 _cvt_bf16_28 = __float2bfloat16(accum[28] * rrms);
                    float _cvt_f32_28 = __bfloat162float(_cvt_bf16_28);
                    float xn0_52 = _cvt_f32_28;
                    __nv_bfloat16 _cvt_bf16_29 = __float2bfloat16(accum[29] * rrms);
                    float _cvt_f32_29 = __bfloat162float(_cvt_bf16_29);
                    float xn1_53 = _cvt_f32_29;
                    __nv_bfloat16 _cvt_bf16_30 = __float2bfloat16(xn0_52 * _vec_load_7[0]);
                    float _cvt_f32_30 = __bfloat162float(_cvt_bf16_30);
                    float xs0_54 = _cvt_f32_30;
                    __nv_bfloat16 _cvt_bf16_31 = __float2bfloat16(xn1_53 * _vec_load_7[1]);
                    float _cvt_f32_31 = __bfloat162float(_cvt_bf16_31);
                    float xs1_55 = _cvt_f32_31;
                    accum[28] = xs0_54 * cos_v_50 - xs1_55 * sin_v_51;
                    accum[29] = xs1_55 * cos_v_50 + xs0_54 * sin_v_51;
                    unsigned int d_col_56 = 64 + 2 * q_lane;
                    unsigned int pair_k_57 = 32 + q_lane;
                    float _vec_load_8[2];
                    {
                        uint32_t _bf16x2_bits_8;
                        _bf16x2_bits_8 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_56 : k_scale + d_col_56) + 0);
                        (&_vec_load_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_8) << 16);
                        (&_vec_load_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_8) & 0xffff0000u);
                    }
                    float cos_v_58 = cos_sin[cs_lo + pair_k_57];
                    float sin_v_59 = cos_sin[cs_lo + 64 + pair_k_57];
                    __nv_bfloat16 _cvt_bf16_32 = __float2bfloat16(accum[32] * rrms);
                    float _cvt_f32_32 = __bfloat162float(_cvt_bf16_32);
                    float xn0_60 = _cvt_f32_32;
                    __nv_bfloat16 _cvt_bf16_33 = __float2bfloat16(accum[33] * rrms);
                    float _cvt_f32_33 = __bfloat162float(_cvt_bf16_33);
                    float xn1_61 = _cvt_f32_33;
                    __nv_bfloat16 _cvt_bf16_34 = __float2bfloat16(xn0_60 * _vec_load_8[0]);
                    float _cvt_f32_34 = __bfloat162float(_cvt_bf16_34);
                    float xs0_62 = _cvt_f32_34;
                    __nv_bfloat16 _cvt_bf16_35 = __float2bfloat16(xn1_61 * _vec_load_8[1]);
                    float _cvt_f32_35 = __bfloat162float(_cvt_bf16_35);
                    float xs1_63 = _cvt_f32_35;
                    accum[32] = xs0_62 * cos_v_58 - xs1_63 * sin_v_59;
                    accum[33] = xs1_63 * cos_v_58 + xs0_62 * sin_v_59;
                    unsigned int d_col_64 = 72 + 2 * q_lane;
                    unsigned int pair_k_65 = 36 + q_lane;
                    float _vec_load_9[2];
                    {
                        uint32_t _bf16x2_bits_9;
                        _bf16x2_bits_9 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_64 : k_scale + d_col_64) + 0);
                        (&_vec_load_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_9) << 16);
                        (&_vec_load_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_9) & 0xffff0000u);
                    }
                    float cos_v_66 = cos_sin[cs_lo + pair_k_65];
                    float sin_v_67 = cos_sin[cs_lo + 64 + pair_k_65];
                    __nv_bfloat16 _cvt_bf16_36 = __float2bfloat16(accum[36] * rrms);
                    float _cvt_f32_36 = __bfloat162float(_cvt_bf16_36);
                    float xn0_68 = _cvt_f32_36;
                    __nv_bfloat16 _cvt_bf16_37 = __float2bfloat16(accum[37] * rrms);
                    float _cvt_f32_37 = __bfloat162float(_cvt_bf16_37);
                    float xn1_69 = _cvt_f32_37;
                    __nv_bfloat16 _cvt_bf16_38 = __float2bfloat16(xn0_68 * _vec_load_9[0]);
                    float _cvt_f32_38 = __bfloat162float(_cvt_bf16_38);
                    float xs0_70 = _cvt_f32_38;
                    __nv_bfloat16 _cvt_bf16_39 = __float2bfloat16(xn1_69 * _vec_load_9[1]);
                    float _cvt_f32_39 = __bfloat162float(_cvt_bf16_39);
                    float xs1_71 = _cvt_f32_39;
                    accum[36] = xs0_70 * cos_v_66 - xs1_71 * sin_v_67;
                    accum[37] = xs1_71 * cos_v_66 + xs0_70 * sin_v_67;
                    unsigned int d_col_72 = 80 + 2 * q_lane;
                    unsigned int pair_k_73 = 40 + q_lane;
                    float _vec_load_10[2];
                    {
                        uint32_t _bf16x2_bits_10;
                        _bf16x2_bits_10 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_72 : k_scale + d_col_72) + 0);
                        (&_vec_load_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_10) << 16);
                        (&_vec_load_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_10) & 0xffff0000u);
                    }
                    float cos_v_74 = cos_sin[cs_lo + pair_k_73];
                    float sin_v_75 = cos_sin[cs_lo + 64 + pair_k_73];
                    __nv_bfloat16 _cvt_bf16_40 = __float2bfloat16(accum[40] * rrms);
                    float _cvt_f32_40 = __bfloat162float(_cvt_bf16_40);
                    float xn0_76 = _cvt_f32_40;
                    __nv_bfloat16 _cvt_bf16_41 = __float2bfloat16(accum[41] * rrms);
                    float _cvt_f32_41 = __bfloat162float(_cvt_bf16_41);
                    float xn1_77 = _cvt_f32_41;
                    __nv_bfloat16 _cvt_bf16_42 = __float2bfloat16(xn0_76 * _vec_load_10[0]);
                    float _cvt_f32_42 = __bfloat162float(_cvt_bf16_42);
                    float xs0_78 = _cvt_f32_42;
                    __nv_bfloat16 _cvt_bf16_43 = __float2bfloat16(xn1_77 * _vec_load_10[1]);
                    float _cvt_f32_43 = __bfloat162float(_cvt_bf16_43);
                    float xs1_79 = _cvt_f32_43;
                    accum[40] = xs0_78 * cos_v_74 - xs1_79 * sin_v_75;
                    accum[41] = xs1_79 * cos_v_74 + xs0_78 * sin_v_75;
                    unsigned int d_col_80 = 88 + 2 * q_lane;
                    unsigned int pair_k_81 = 44 + q_lane;
                    float _vec_load_11[2];
                    {
                        uint32_t _bf16x2_bits_11;
                        _bf16x2_bits_11 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_80 : k_scale + d_col_80) + 0);
                        (&_vec_load_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_11) << 16);
                        (&_vec_load_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_11) & 0xffff0000u);
                    }
                    float cos_v_82 = cos_sin[cs_lo + pair_k_81];
                    float sin_v_83 = cos_sin[cs_lo + 64 + pair_k_81];
                    __nv_bfloat16 _cvt_bf16_44 = __float2bfloat16(accum[44] * rrms);
                    float _cvt_f32_44 = __bfloat162float(_cvt_bf16_44);
                    float xn0_84 = _cvt_f32_44;
                    __nv_bfloat16 _cvt_bf16_45 = __float2bfloat16(accum[45] * rrms);
                    float _cvt_f32_45 = __bfloat162float(_cvt_bf16_45);
                    float xn1_85 = _cvt_f32_45;
                    __nv_bfloat16 _cvt_bf16_46 = __float2bfloat16(xn0_84 * _vec_load_11[0]);
                    float _cvt_f32_46 = __bfloat162float(_cvt_bf16_46);
                    float xs0_86 = _cvt_f32_46;
                    __nv_bfloat16 _cvt_bf16_47 = __float2bfloat16(xn1_85 * _vec_load_11[1]);
                    float _cvt_f32_47 = __bfloat162float(_cvt_bf16_47);
                    float xs1_87 = _cvt_f32_47;
                    accum[44] = xs0_86 * cos_v_82 - xs1_87 * sin_v_83;
                    accum[45] = xs1_87 * cos_v_82 + xs0_86 * sin_v_83;
                    unsigned int d_col_88 = 96 + 2 * q_lane;
                    unsigned int pair_k_89 = 48 + q_lane;
                    float _vec_load_12[2];
                    {
                        uint32_t _bf16x2_bits_12;
                        _bf16x2_bits_12 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_88 : k_scale + d_col_88) + 0);
                        (&_vec_load_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_12) << 16);
                        (&_vec_load_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_12) & 0xffff0000u);
                    }
                    float cos_v_90 = cos_sin[cs_lo + pair_k_89];
                    float sin_v_91 = cos_sin[cs_lo + 64 + pair_k_89];
                    __nv_bfloat16 _cvt_bf16_48 = __float2bfloat16(accum[48] * rrms);
                    float _cvt_f32_48 = __bfloat162float(_cvt_bf16_48);
                    float xn0_92 = _cvt_f32_48;
                    __nv_bfloat16 _cvt_bf16_49 = __float2bfloat16(accum[49] * rrms);
                    float _cvt_f32_49 = __bfloat162float(_cvt_bf16_49);
                    float xn1_93 = _cvt_f32_49;
                    __nv_bfloat16 _cvt_bf16_50 = __float2bfloat16(xn0_92 * _vec_load_12[0]);
                    float _cvt_f32_50 = __bfloat162float(_cvt_bf16_50);
                    float xs0_94 = _cvt_f32_50;
                    __nv_bfloat16 _cvt_bf16_51 = __float2bfloat16(xn1_93 * _vec_load_12[1]);
                    float _cvt_f32_51 = __bfloat162float(_cvt_bf16_51);
                    float xs1_95 = _cvt_f32_51;
                    accum[48] = xs0_94 * cos_v_90 - xs1_95 * sin_v_91;
                    accum[49] = xs1_95 * cos_v_90 + xs0_94 * sin_v_91;
                    unsigned int d_col_96 = 104 + 2 * q_lane;
                    unsigned int pair_k_97 = 52 + q_lane;
                    float _vec_load_13[2];
                    {
                        uint32_t _bf16x2_bits_13;
                        _bf16x2_bits_13 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_96 : k_scale + d_col_96) + 0);
                        (&_vec_load_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_13) << 16);
                        (&_vec_load_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_13) & 0xffff0000u);
                    }
                    float cos_v_98 = cos_sin[cs_lo + pair_k_97];
                    float sin_v_99 = cos_sin[cs_lo + 64 + pair_k_97];
                    __nv_bfloat16 _cvt_bf16_52 = __float2bfloat16(accum[52] * rrms);
                    float _cvt_f32_52 = __bfloat162float(_cvt_bf16_52);
                    float xn0_100 = _cvt_f32_52;
                    __nv_bfloat16 _cvt_bf16_53 = __float2bfloat16(accum[53] * rrms);
                    float _cvt_f32_53 = __bfloat162float(_cvt_bf16_53);
                    float xn1_101 = _cvt_f32_53;
                    __nv_bfloat16 _cvt_bf16_54 = __float2bfloat16(xn0_100 * _vec_load_13[0]);
                    float _cvt_f32_54 = __bfloat162float(_cvt_bf16_54);
                    float xs0_102 = _cvt_f32_54;
                    __nv_bfloat16 _cvt_bf16_55 = __float2bfloat16(xn1_101 * _vec_load_13[1]);
                    float _cvt_f32_55 = __bfloat162float(_cvt_bf16_55);
                    float xs1_103 = _cvt_f32_55;
                    accum[52] = xs0_102 * cos_v_98 - xs1_103 * sin_v_99;
                    accum[53] = xs1_103 * cos_v_98 + xs0_102 * sin_v_99;
                    unsigned int d_col_104 = 112 + 2 * q_lane;
                    unsigned int pair_k_105 = 56 + q_lane;
                    float _vec_load_14[2];
                    {
                        uint32_t _bf16x2_bits_14;
                        _bf16x2_bits_14 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_104 : k_scale + d_col_104) + 0);
                        (&_vec_load_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_14) << 16);
                        (&_vec_load_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_14) & 0xffff0000u);
                    }
                    float cos_v_106 = cos_sin[cs_lo + pair_k_105];
                    float sin_v_107 = cos_sin[cs_lo + 64 + pair_k_105];
                    __nv_bfloat16 _cvt_bf16_56 = __float2bfloat16(accum[56] * rrms);
                    float _cvt_f32_56 = __bfloat162float(_cvt_bf16_56);
                    float xn0_108 = _cvt_f32_56;
                    __nv_bfloat16 _cvt_bf16_57 = __float2bfloat16(accum[57] * rrms);
                    float _cvt_f32_57 = __bfloat162float(_cvt_bf16_57);
                    float xn1_109 = _cvt_f32_57;
                    __nv_bfloat16 _cvt_bf16_58 = __float2bfloat16(xn0_108 * _vec_load_14[0]);
                    float _cvt_f32_58 = __bfloat162float(_cvt_bf16_58);
                    float xs0_110 = _cvt_f32_58;
                    __nv_bfloat16 _cvt_bf16_59 = __float2bfloat16(xn1_109 * _vec_load_14[1]);
                    float _cvt_f32_59 = __bfloat162float(_cvt_bf16_59);
                    float xs1_111 = _cvt_f32_59;
                    accum[56] = xs0_110 * cos_v_106 - xs1_111 * sin_v_107;
                    accum[57] = xs1_111 * cos_v_106 + xs0_110 * sin_v_107;
                    unsigned int d_col_112 = 120 + 2 * q_lane;
                    unsigned int pair_k_113 = 60 + q_lane;
                    float _vec_load_15[2];
                    {
                        uint32_t _bf16x2_bits_15;
                        _bf16x2_bits_15 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_112 : k_scale + d_col_112) + 0);
                        (&_vec_load_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_15) << 16);
                        (&_vec_load_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_15) & 0xffff0000u);
                    }
                    float cos_v_114 = cos_sin[cs_lo + pair_k_113];
                    float sin_v_115 = cos_sin[cs_lo + 64 + pair_k_113];
                    __nv_bfloat16 _cvt_bf16_60 = __float2bfloat16(accum[60] * rrms);
                    float _cvt_f32_60 = __bfloat162float(_cvt_bf16_60);
                    float xn0_116 = _cvt_f32_60;
                    __nv_bfloat16 _cvt_bf16_61 = __float2bfloat16(accum[61] * rrms);
                    float _cvt_f32_61 = __bfloat162float(_cvt_bf16_61);
                    float xn1_117 = _cvt_f32_61;
                    __nv_bfloat16 _cvt_bf16_62 = __float2bfloat16(xn0_116 * _vec_load_15[0]);
                    float _cvt_f32_62 = __bfloat162float(_cvt_bf16_62);
                    float xs0_118 = _cvt_f32_62;
                    __nv_bfloat16 _cvt_bf16_63 = __float2bfloat16(xn1_117 * _vec_load_15[1]);
                    float _cvt_f32_63 = __bfloat162float(_cvt_bf16_63);
                    float xs1_119 = _cvt_f32_63;
                    accum[60] = xs0_118 * cos_v_114 - xs1_119 * sin_v_115;
                    accum[61] = xs1_119 * cos_v_114 + xs0_118 * sin_v_115;
                    float sumsq_120 = 0.0f;
                    sumsq_120 += accum[2] * accum[2];
                    sumsq_120 += accum[3] * accum[3];
                    sumsq_120 += accum[6] * accum[6];
                    sumsq_120 += accum[7] * accum[7];
                    sumsq_120 += accum[10] * accum[10];
                    sumsq_120 += accum[11] * accum[11];
                    sumsq_120 += accum[14] * accum[14];
                    sumsq_120 += accum[15] * accum[15];
                    sumsq_120 += accum[18] * accum[18];
                    sumsq_120 += accum[19] * accum[19];
                    sumsq_120 += accum[22] * accum[22];
                    sumsq_120 += accum[23] * accum[23];
                    sumsq_120 += accum[26] * accum[26];
                    sumsq_120 += accum[27] * accum[27];
                    sumsq_120 += accum[30] * accum[30];
                    sumsq_120 += accum[31] * accum[31];
                    sumsq_120 += accum[34] * accum[34];
                    sumsq_120 += accum[35] * accum[35];
                    sumsq_120 += accum[38] * accum[38];
                    sumsq_120 += accum[39] * accum[39];
                    sumsq_120 += accum[42] * accum[42];
                    sumsq_120 += accum[43] * accum[43];
                    sumsq_120 += accum[46] * accum[46];
                    sumsq_120 += accum[47] * accum[47];
                    sumsq_120 += accum[50] * accum[50];
                    sumsq_120 += accum[51] * accum[51];
                    sumsq_120 += accum[54] * accum[54];
                    sumsq_120 += accum[55] * accum[55];
                    sumsq_120 += accum[58] * accum[58];
                    sumsq_120 += accum[59] * accum[59];
                    sumsq_120 += accum[62] * accum[62];
                    sumsq_120 += accum[63] * accum[63];
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, sumsq_120, 1);
                    float sq1_121 = _shfl_xor_2;
                    sumsq_120 += sq1_121;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, sumsq_120, 2);
                    float sq2_122 = _shfl_xor_3;
                    sumsq_120 += sq2_122;
                    float _rsqrt_1 = rsqrtf(sumsq_120 * 0.0078125f + 1e-06f);
                    float rrms_123 = _rsqrt_1;
                    unsigned int d_col_124 = 2 * q_lane;
                    unsigned int pair_k_125 = q_lane;
                    float _vec_load_16[2];
                    {
                        uint32_t _bf16x2_bits_16;
                        _bf16x2_bits_16 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_124 : k_scale + d_col_124) + 0);
                        (&_vec_load_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_16) << 16);
                        (&_vec_load_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_16) & 0xffff0000u);
                    }
                    float cos_v_126 = cos_sin[cs_hi + pair_k_125];
                    float sin_v_127 = cos_sin[cs_hi + 64 + pair_k_125];
                    __nv_bfloat16 _cvt_bf16_64 = __float2bfloat16(accum[2] * rrms_123);
                    float _cvt_f32_64 = __bfloat162float(_cvt_bf16_64);
                    float xn0_128 = _cvt_f32_64;
                    __nv_bfloat16 _cvt_bf16_65 = __float2bfloat16(accum[3] * rrms_123);
                    float _cvt_f32_65 = __bfloat162float(_cvt_bf16_65);
                    float xn1_129 = _cvt_f32_65;
                    __nv_bfloat16 _cvt_bf16_66 = __float2bfloat16(xn0_128 * _vec_load_16[0]);
                    float _cvt_f32_66 = __bfloat162float(_cvt_bf16_66);
                    float xs0_130 = _cvt_f32_66;
                    __nv_bfloat16 _cvt_bf16_67 = __float2bfloat16(xn1_129 * _vec_load_16[1]);
                    float _cvt_f32_67 = __bfloat162float(_cvt_bf16_67);
                    float xs1_131 = _cvt_f32_67;
                    accum[2] = xs0_130 * cos_v_126 - xs1_131 * sin_v_127;
                    accum[3] = xs1_131 * cos_v_126 + xs0_130 * sin_v_127;
                    unsigned int d_col_132 = 8 + 2 * q_lane;
                    unsigned int pair_k_133 = 4 + q_lane;
                    float _vec_load_17[2];
                    {
                        uint32_t _bf16x2_bits_17;
                        _bf16x2_bits_17 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_132 : k_scale + d_col_132) + 0);
                        (&_vec_load_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_17) << 16);
                        (&_vec_load_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_17) & 0xffff0000u);
                    }
                    float cos_v_134 = cos_sin[cs_hi + pair_k_133];
                    float sin_v_135 = cos_sin[cs_hi + 64 + pair_k_133];
                    __nv_bfloat16 _cvt_bf16_68 = __float2bfloat16(accum[6] * rrms_123);
                    float _cvt_f32_68 = __bfloat162float(_cvt_bf16_68);
                    float xn0_136 = _cvt_f32_68;
                    __nv_bfloat16 _cvt_bf16_69 = __float2bfloat16(accum[7] * rrms_123);
                    float _cvt_f32_69 = __bfloat162float(_cvt_bf16_69);
                    float xn1_137 = _cvt_f32_69;
                    __nv_bfloat16 _cvt_bf16_70 = __float2bfloat16(xn0_136 * _vec_load_17[0]);
                    float _cvt_f32_70 = __bfloat162float(_cvt_bf16_70);
                    float xs0_138 = _cvt_f32_70;
                    __nv_bfloat16 _cvt_bf16_71 = __float2bfloat16(xn1_137 * _vec_load_17[1]);
                    float _cvt_f32_71 = __bfloat162float(_cvt_bf16_71);
                    float xs1_139 = _cvt_f32_71;
                    accum[6] = xs0_138 * cos_v_134 - xs1_139 * sin_v_135;
                    accum[7] = xs1_139 * cos_v_134 + xs0_138 * sin_v_135;
                    unsigned int d_col_140 = 16 + 2 * q_lane;
                    unsigned int pair_k_141 = 8 + q_lane;
                    float _vec_load_18[2];
                    {
                        uint32_t _bf16x2_bits_18;
                        _bf16x2_bits_18 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_140 : k_scale + d_col_140) + 0);
                        (&_vec_load_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_18) << 16);
                        (&_vec_load_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_18) & 0xffff0000u);
                    }
                    float cos_v_142 = cos_sin[cs_hi + pair_k_141];
                    float sin_v_143 = cos_sin[cs_hi + 64 + pair_k_141];
                    __nv_bfloat16 _cvt_bf16_72 = __float2bfloat16(accum[10] * rrms_123);
                    float _cvt_f32_72 = __bfloat162float(_cvt_bf16_72);
                    float xn0_144 = _cvt_f32_72;
                    __nv_bfloat16 _cvt_bf16_73 = __float2bfloat16(accum[11] * rrms_123);
                    float _cvt_f32_73 = __bfloat162float(_cvt_bf16_73);
                    float xn1_145 = _cvt_f32_73;
                    __nv_bfloat16 _cvt_bf16_74 = __float2bfloat16(xn0_144 * _vec_load_18[0]);
                    float _cvt_f32_74 = __bfloat162float(_cvt_bf16_74);
                    float xs0_146 = _cvt_f32_74;
                    __nv_bfloat16 _cvt_bf16_75 = __float2bfloat16(xn1_145 * _vec_load_18[1]);
                    float _cvt_f32_75 = __bfloat162float(_cvt_bf16_75);
                    float xs1_147 = _cvt_f32_75;
                    accum[10] = xs0_146 * cos_v_142 - xs1_147 * sin_v_143;
                    accum[11] = xs1_147 * cos_v_142 + xs0_146 * sin_v_143;
                    unsigned int d_col_148 = 24 + 2 * q_lane;
                    unsigned int pair_k_149 = 12 + q_lane;
                    float _vec_load_19[2];
                    {
                        uint32_t _bf16x2_bits_19;
                        _bf16x2_bits_19 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_148 : k_scale + d_col_148) + 0);
                        (&_vec_load_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_19) << 16);
                        (&_vec_load_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_19) & 0xffff0000u);
                    }
                    float cos_v_150 = cos_sin[cs_hi + pair_k_149];
                    float sin_v_151 = cos_sin[cs_hi + 64 + pair_k_149];
                    __nv_bfloat16 _cvt_bf16_76 = __float2bfloat16(accum[14] * rrms_123);
                    float _cvt_f32_76 = __bfloat162float(_cvt_bf16_76);
                    float xn0_152 = _cvt_f32_76;
                    __nv_bfloat16 _cvt_bf16_77 = __float2bfloat16(accum[15] * rrms_123);
                    float _cvt_f32_77 = __bfloat162float(_cvt_bf16_77);
                    float xn1_153 = _cvt_f32_77;
                    __nv_bfloat16 _cvt_bf16_78 = __float2bfloat16(xn0_152 * _vec_load_19[0]);
                    float _cvt_f32_78 = __bfloat162float(_cvt_bf16_78);
                    float xs0_154 = _cvt_f32_78;
                    __nv_bfloat16 _cvt_bf16_79 = __float2bfloat16(xn1_153 * _vec_load_19[1]);
                    float _cvt_f32_79 = __bfloat162float(_cvt_bf16_79);
                    float xs1_155 = _cvt_f32_79;
                    accum[14] = xs0_154 * cos_v_150 - xs1_155 * sin_v_151;
                    accum[15] = xs1_155 * cos_v_150 + xs0_154 * sin_v_151;
                    unsigned int d_col_156 = 32 + 2 * q_lane;
                    unsigned int pair_k_157 = 16 + q_lane;
                    float _vec_load_20[2];
                    {
                        uint32_t _bf16x2_bits_20;
                        _bf16x2_bits_20 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_156 : k_scale + d_col_156) + 0);
                        (&_vec_load_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_20) << 16);
                        (&_vec_load_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_20) & 0xffff0000u);
                    }
                    float cos_v_158 = cos_sin[cs_hi + pair_k_157];
                    float sin_v_159 = cos_sin[cs_hi + 64 + pair_k_157];
                    __nv_bfloat16 _cvt_bf16_80 = __float2bfloat16(accum[18] * rrms_123);
                    float _cvt_f32_80 = __bfloat162float(_cvt_bf16_80);
                    float xn0_160 = _cvt_f32_80;
                    __nv_bfloat16 _cvt_bf16_81 = __float2bfloat16(accum[19] * rrms_123);
                    float _cvt_f32_81 = __bfloat162float(_cvt_bf16_81);
                    float xn1_161 = _cvt_f32_81;
                    __nv_bfloat16 _cvt_bf16_82 = __float2bfloat16(xn0_160 * _vec_load_20[0]);
                    float _cvt_f32_82 = __bfloat162float(_cvt_bf16_82);
                    float xs0_162 = _cvt_f32_82;
                    __nv_bfloat16 _cvt_bf16_83 = __float2bfloat16(xn1_161 * _vec_load_20[1]);
                    float _cvt_f32_83 = __bfloat162float(_cvt_bf16_83);
                    float xs1_163 = _cvt_f32_83;
                    accum[18] = xs0_162 * cos_v_158 - xs1_163 * sin_v_159;
                    accum[19] = xs1_163 * cos_v_158 + xs0_162 * sin_v_159;
                    unsigned int d_col_164 = 40 + 2 * q_lane;
                    unsigned int pair_k_165 = 20 + q_lane;
                    float _vec_load_21[2];
                    {
                        uint32_t _bf16x2_bits_21;
                        _bf16x2_bits_21 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_164 : k_scale + d_col_164) + 0);
                        (&_vec_load_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_21) << 16);
                        (&_vec_load_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_21) & 0xffff0000u);
                    }
                    float cos_v_166 = cos_sin[cs_hi + pair_k_165];
                    float sin_v_167 = cos_sin[cs_hi + 64 + pair_k_165];
                    __nv_bfloat16 _cvt_bf16_84 = __float2bfloat16(accum[22] * rrms_123);
                    float _cvt_f32_84 = __bfloat162float(_cvt_bf16_84);
                    float xn0_168 = _cvt_f32_84;
                    __nv_bfloat16 _cvt_bf16_85 = __float2bfloat16(accum[23] * rrms_123);
                    float _cvt_f32_85 = __bfloat162float(_cvt_bf16_85);
                    float xn1_169 = _cvt_f32_85;
                    __nv_bfloat16 _cvt_bf16_86 = __float2bfloat16(xn0_168 * _vec_load_21[0]);
                    float _cvt_f32_86 = __bfloat162float(_cvt_bf16_86);
                    float xs0_170 = _cvt_f32_86;
                    __nv_bfloat16 _cvt_bf16_87 = __float2bfloat16(xn1_169 * _vec_load_21[1]);
                    float _cvt_f32_87 = __bfloat162float(_cvt_bf16_87);
                    float xs1_171 = _cvt_f32_87;
                    accum[22] = xs0_170 * cos_v_166 - xs1_171 * sin_v_167;
                    accum[23] = xs1_171 * cos_v_166 + xs0_170 * sin_v_167;
                    unsigned int d_col_172 = 48 + 2 * q_lane;
                    unsigned int pair_k_173 = 24 + q_lane;
                    float _vec_load_22[2];
                    {
                        uint32_t _bf16x2_bits_22;
                        _bf16x2_bits_22 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_172 : k_scale + d_col_172) + 0);
                        (&_vec_load_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) << 16);
                        (&_vec_load_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) & 0xffff0000u);
                    }
                    float cos_v_174 = cos_sin[cs_hi + pair_k_173];
                    float sin_v_175 = cos_sin[cs_hi + 64 + pair_k_173];
                    __nv_bfloat16 _cvt_bf16_88 = __float2bfloat16(accum[26] * rrms_123);
                    float _cvt_f32_88 = __bfloat162float(_cvt_bf16_88);
                    float xn0_176 = _cvt_f32_88;
                    __nv_bfloat16 _cvt_bf16_89 = __float2bfloat16(accum[27] * rrms_123);
                    float _cvt_f32_89 = __bfloat162float(_cvt_bf16_89);
                    float xn1_177 = _cvt_f32_89;
                    __nv_bfloat16 _cvt_bf16_90 = __float2bfloat16(xn0_176 * _vec_load_22[0]);
                    float _cvt_f32_90 = __bfloat162float(_cvt_bf16_90);
                    float xs0_178 = _cvt_f32_90;
                    __nv_bfloat16 _cvt_bf16_91 = __float2bfloat16(xn1_177 * _vec_load_22[1]);
                    float _cvt_f32_91 = __bfloat162float(_cvt_bf16_91);
                    float xs1_179 = _cvt_f32_91;
                    accum[26] = xs0_178 * cos_v_174 - xs1_179 * sin_v_175;
                    accum[27] = xs1_179 * cos_v_174 + xs0_178 * sin_v_175;
                    unsigned int d_col_180 = 56 + 2 * q_lane;
                    unsigned int pair_k_181 = 28 + q_lane;
                    float _vec_load_23[2];
                    {
                        uint32_t _bf16x2_bits_23;
                        _bf16x2_bits_23 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_180 : k_scale + d_col_180) + 0);
                        (&_vec_load_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) << 16);
                        (&_vec_load_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) & 0xffff0000u);
                    }
                    float cos_v_182 = cos_sin[cs_hi + pair_k_181];
                    float sin_v_183 = cos_sin[cs_hi + 64 + pair_k_181];
                    __nv_bfloat16 _cvt_bf16_92 = __float2bfloat16(accum[30] * rrms_123);
                    float _cvt_f32_92 = __bfloat162float(_cvt_bf16_92);
                    float xn0_184 = _cvt_f32_92;
                    __nv_bfloat16 _cvt_bf16_93 = __float2bfloat16(accum[31] * rrms_123);
                    float _cvt_f32_93 = __bfloat162float(_cvt_bf16_93);
                    float xn1_185 = _cvt_f32_93;
                    __nv_bfloat16 _cvt_bf16_94 = __float2bfloat16(xn0_184 * _vec_load_23[0]);
                    float _cvt_f32_94 = __bfloat162float(_cvt_bf16_94);
                    float xs0_186 = _cvt_f32_94;
                    __nv_bfloat16 _cvt_bf16_95 = __float2bfloat16(xn1_185 * _vec_load_23[1]);
                    float _cvt_f32_95 = __bfloat162float(_cvt_bf16_95);
                    float xs1_187 = _cvt_f32_95;
                    accum[30] = xs0_186 * cos_v_182 - xs1_187 * sin_v_183;
                    accum[31] = xs1_187 * cos_v_182 + xs0_186 * sin_v_183;
                    unsigned int d_col_188 = 64 + 2 * q_lane;
                    unsigned int pair_k_189 = 32 + q_lane;
                    float _vec_load_24[2];
                    {
                        uint32_t _bf16x2_bits_24;
                        _bf16x2_bits_24 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_188 : k_scale + d_col_188) + 0);
                        (&_vec_load_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) << 16);
                        (&_vec_load_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) & 0xffff0000u);
                    }
                    float cos_v_190 = cos_sin[cs_hi + pair_k_189];
                    float sin_v_191 = cos_sin[cs_hi + 64 + pair_k_189];
                    __nv_bfloat16 _cvt_bf16_96 = __float2bfloat16(accum[34] * rrms_123);
                    float _cvt_f32_96 = __bfloat162float(_cvt_bf16_96);
                    float xn0_192 = _cvt_f32_96;
                    __nv_bfloat16 _cvt_bf16_97 = __float2bfloat16(accum[35] * rrms_123);
                    float _cvt_f32_97 = __bfloat162float(_cvt_bf16_97);
                    float xn1_193 = _cvt_f32_97;
                    __nv_bfloat16 _cvt_bf16_98 = __float2bfloat16(xn0_192 * _vec_load_24[0]);
                    float _cvt_f32_98 = __bfloat162float(_cvt_bf16_98);
                    float xs0_194 = _cvt_f32_98;
                    __nv_bfloat16 _cvt_bf16_99 = __float2bfloat16(xn1_193 * _vec_load_24[1]);
                    float _cvt_f32_99 = __bfloat162float(_cvt_bf16_99);
                    float xs1_195 = _cvt_f32_99;
                    accum[34] = xs0_194 * cos_v_190 - xs1_195 * sin_v_191;
                    accum[35] = xs1_195 * cos_v_190 + xs0_194 * sin_v_191;
                    unsigned int d_col_196 = 72 + 2 * q_lane;
                    unsigned int pair_k_197 = 36 + q_lane;
                    float _vec_load_25[2];
                    {
                        uint32_t _bf16x2_bits_25;
                        _bf16x2_bits_25 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_196 : k_scale + d_col_196) + 0);
                        (&_vec_load_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) << 16);
                        (&_vec_load_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) & 0xffff0000u);
                    }
                    float cos_v_198 = cos_sin[cs_hi + pair_k_197];
                    float sin_v_199 = cos_sin[cs_hi + 64 + pair_k_197];
                    __nv_bfloat16 _cvt_bf16_100 = __float2bfloat16(accum[38] * rrms_123);
                    float _cvt_f32_100 = __bfloat162float(_cvt_bf16_100);
                    float xn0_200 = _cvt_f32_100;
                    __nv_bfloat16 _cvt_bf16_101 = __float2bfloat16(accum[39] * rrms_123);
                    float _cvt_f32_101 = __bfloat162float(_cvt_bf16_101);
                    float xn1_201 = _cvt_f32_101;
                    __nv_bfloat16 _cvt_bf16_102 = __float2bfloat16(xn0_200 * _vec_load_25[0]);
                    float _cvt_f32_102 = __bfloat162float(_cvt_bf16_102);
                    float xs0_202 = _cvt_f32_102;
                    __nv_bfloat16 _cvt_bf16_103 = __float2bfloat16(xn1_201 * _vec_load_25[1]);
                    float _cvt_f32_103 = __bfloat162float(_cvt_bf16_103);
                    float xs1_203 = _cvt_f32_103;
                    accum[38] = xs0_202 * cos_v_198 - xs1_203 * sin_v_199;
                    accum[39] = xs1_203 * cos_v_198 + xs0_202 * sin_v_199;
                    unsigned int d_col_204 = 80 + 2 * q_lane;
                    unsigned int pair_k_205 = 40 + q_lane;
                    float _vec_load_26[2];
                    {
                        uint32_t _bf16x2_bits_26;
                        _bf16x2_bits_26 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_204 : k_scale + d_col_204) + 0);
                        (&_vec_load_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) << 16);
                        (&_vec_load_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) & 0xffff0000u);
                    }
                    float cos_v_206 = cos_sin[cs_hi + pair_k_205];
                    float sin_v_207 = cos_sin[cs_hi + 64 + pair_k_205];
                    __nv_bfloat16 _cvt_bf16_104 = __float2bfloat16(accum[42] * rrms_123);
                    float _cvt_f32_104 = __bfloat162float(_cvt_bf16_104);
                    float xn0_208 = _cvt_f32_104;
                    __nv_bfloat16 _cvt_bf16_105 = __float2bfloat16(accum[43] * rrms_123);
                    float _cvt_f32_105 = __bfloat162float(_cvt_bf16_105);
                    float xn1_209 = _cvt_f32_105;
                    __nv_bfloat16 _cvt_bf16_106 = __float2bfloat16(xn0_208 * _vec_load_26[0]);
                    float _cvt_f32_106 = __bfloat162float(_cvt_bf16_106);
                    float xs0_210 = _cvt_f32_106;
                    __nv_bfloat16 _cvt_bf16_107 = __float2bfloat16(xn1_209 * _vec_load_26[1]);
                    float _cvt_f32_107 = __bfloat162float(_cvt_bf16_107);
                    float xs1_211 = _cvt_f32_107;
                    accum[42] = xs0_210 * cos_v_206 - xs1_211 * sin_v_207;
                    accum[43] = xs1_211 * cos_v_206 + xs0_210 * sin_v_207;
                    unsigned int d_col_212 = 88 + 2 * q_lane;
                    unsigned int pair_k_213 = 44 + q_lane;
                    float _vec_load_27[2];
                    {
                        uint32_t _bf16x2_bits_27;
                        _bf16x2_bits_27 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_212 : k_scale + d_col_212) + 0);
                        (&_vec_load_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) << 16);
                        (&_vec_load_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) & 0xffff0000u);
                    }
                    float cos_v_214 = cos_sin[cs_hi + pair_k_213];
                    float sin_v_215 = cos_sin[cs_hi + 64 + pair_k_213];
                    __nv_bfloat16 _cvt_bf16_108 = __float2bfloat16(accum[46] * rrms_123);
                    float _cvt_f32_108 = __bfloat162float(_cvt_bf16_108);
                    float xn0_216 = _cvt_f32_108;
                    __nv_bfloat16 _cvt_bf16_109 = __float2bfloat16(accum[47] * rrms_123);
                    float _cvt_f32_109 = __bfloat162float(_cvt_bf16_109);
                    float xn1_217 = _cvt_f32_109;
                    __nv_bfloat16 _cvt_bf16_110 = __float2bfloat16(xn0_216 * _vec_load_27[0]);
                    float _cvt_f32_110 = __bfloat162float(_cvt_bf16_110);
                    float xs0_218 = _cvt_f32_110;
                    __nv_bfloat16 _cvt_bf16_111 = __float2bfloat16(xn1_217 * _vec_load_27[1]);
                    float _cvt_f32_111 = __bfloat162float(_cvt_bf16_111);
                    float xs1_219 = _cvt_f32_111;
                    accum[46] = xs0_218 * cos_v_214 - xs1_219 * sin_v_215;
                    accum[47] = xs1_219 * cos_v_214 + xs0_218 * sin_v_215;
                    unsigned int d_col_220 = 96 + 2 * q_lane;
                    unsigned int pair_k_221 = 48 + q_lane;
                    float _vec_load_28[2];
                    {
                        uint32_t _bf16x2_bits_28;
                        _bf16x2_bits_28 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_220 : k_scale + d_col_220) + 0);
                        (&_vec_load_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_28) << 16);
                        (&_vec_load_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_28) & 0xffff0000u);
                    }
                    float cos_v_222 = cos_sin[cs_hi + pair_k_221];
                    float sin_v_223 = cos_sin[cs_hi + 64 + pair_k_221];
                    __nv_bfloat16 _cvt_bf16_112 = __float2bfloat16(accum[50] * rrms_123);
                    float _cvt_f32_112 = __bfloat162float(_cvt_bf16_112);
                    float xn0_224 = _cvt_f32_112;
                    __nv_bfloat16 _cvt_bf16_113 = __float2bfloat16(accum[51] * rrms_123);
                    float _cvt_f32_113 = __bfloat162float(_cvt_bf16_113);
                    float xn1_225 = _cvt_f32_113;
                    __nv_bfloat16 _cvt_bf16_114 = __float2bfloat16(xn0_224 * _vec_load_28[0]);
                    float _cvt_f32_114 = __bfloat162float(_cvt_bf16_114);
                    float xs0_226 = _cvt_f32_114;
                    __nv_bfloat16 _cvt_bf16_115 = __float2bfloat16(xn1_225 * _vec_load_28[1]);
                    float _cvt_f32_115 = __bfloat162float(_cvt_bf16_115);
                    float xs1_227 = _cvt_f32_115;
                    accum[50] = xs0_226 * cos_v_222 - xs1_227 * sin_v_223;
                    accum[51] = xs1_227 * cos_v_222 + xs0_226 * sin_v_223;
                    unsigned int d_col_228 = 104 + 2 * q_lane;
                    unsigned int pair_k_229 = 52 + q_lane;
                    float _vec_load_29[2];
                    {
                        uint32_t _bf16x2_bits_29;
                        _bf16x2_bits_29 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_228 : k_scale + d_col_228) + 0);
                        (&_vec_load_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_29) << 16);
                        (&_vec_load_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_29) & 0xffff0000u);
                    }
                    float cos_v_230 = cos_sin[cs_hi + pair_k_229];
                    float sin_v_231 = cos_sin[cs_hi + 64 + pair_k_229];
                    __nv_bfloat16 _cvt_bf16_116 = __float2bfloat16(accum[54] * rrms_123);
                    float _cvt_f32_116 = __bfloat162float(_cvt_bf16_116);
                    float xn0_232 = _cvt_f32_116;
                    __nv_bfloat16 _cvt_bf16_117 = __float2bfloat16(accum[55] * rrms_123);
                    float _cvt_f32_117 = __bfloat162float(_cvt_bf16_117);
                    float xn1_233 = _cvt_f32_117;
                    __nv_bfloat16 _cvt_bf16_118 = __float2bfloat16(xn0_232 * _vec_load_29[0]);
                    float _cvt_f32_118 = __bfloat162float(_cvt_bf16_118);
                    float xs0_234 = _cvt_f32_118;
                    __nv_bfloat16 _cvt_bf16_119 = __float2bfloat16(xn1_233 * _vec_load_29[1]);
                    float _cvt_f32_119 = __bfloat162float(_cvt_bf16_119);
                    float xs1_235 = _cvt_f32_119;
                    accum[54] = xs0_234 * cos_v_230 - xs1_235 * sin_v_231;
                    accum[55] = xs1_235 * cos_v_230 + xs0_234 * sin_v_231;
                    unsigned int d_col_236 = 112 + 2 * q_lane;
                    unsigned int pair_k_237 = 56 + q_lane;
                    float _vec_load_30[2];
                    {
                        uint32_t _bf16x2_bits_30;
                        _bf16x2_bits_30 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_236 : k_scale + d_col_236) + 0);
                        (&_vec_load_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_30) << 16);
                        (&_vec_load_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_30) & 0xffff0000u);
                    }
                    float cos_v_238 = cos_sin[cs_hi + pair_k_237];
                    float sin_v_239 = cos_sin[cs_hi + 64 + pair_k_237];
                    __nv_bfloat16 _cvt_bf16_120 = __float2bfloat16(accum[58] * rrms_123);
                    float _cvt_f32_120 = __bfloat162float(_cvt_bf16_120);
                    float xn0_240 = _cvt_f32_120;
                    __nv_bfloat16 _cvt_bf16_121 = __float2bfloat16(accum[59] * rrms_123);
                    float _cvt_f32_121 = __bfloat162float(_cvt_bf16_121);
                    float xn1_241 = _cvt_f32_121;
                    __nv_bfloat16 _cvt_bf16_122 = __float2bfloat16(xn0_240 * _vec_load_30[0]);
                    float _cvt_f32_122 = __bfloat162float(_cvt_bf16_122);
                    float xs0_242 = _cvt_f32_122;
                    __nv_bfloat16 _cvt_bf16_123 = __float2bfloat16(xn1_241 * _vec_load_30[1]);
                    float _cvt_f32_123 = __bfloat162float(_cvt_bf16_123);
                    float xs1_243 = _cvt_f32_123;
                    accum[58] = xs0_242 * cos_v_238 - xs1_243 * sin_v_239;
                    accum[59] = xs1_243 * cos_v_238 + xs0_242 * sin_v_239;
                    unsigned int d_col_244 = 120 + 2 * q_lane;
                    unsigned int pair_k_245 = 60 + q_lane;
                    float _vec_load_31[2];
                    {
                        uint32_t _bf16x2_bits_31;
                        _bf16x2_bits_31 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_244 : k_scale + d_col_244) + 0);
                        (&_vec_load_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_31) << 16);
                        (&_vec_load_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_31) & 0xffff0000u);
                    }
                    float cos_v_246 = cos_sin[cs_hi + pair_k_245];
                    float sin_v_247 = cos_sin[cs_hi + 64 + pair_k_245];
                    __nv_bfloat16 _cvt_bf16_124 = __float2bfloat16(accum[62] * rrms_123);
                    float _cvt_f32_124 = __bfloat162float(_cvt_bf16_124);
                    float xn0_248 = _cvt_f32_124;
                    __nv_bfloat16 _cvt_bf16_125 = __float2bfloat16(accum[63] * rrms_123);
                    float _cvt_f32_125 = __bfloat162float(_cvt_bf16_125);
                    float xn1_249 = _cvt_f32_125;
                    __nv_bfloat16 _cvt_bf16_126 = __float2bfloat16(xn0_248 * _vec_load_31[0]);
                    float _cvt_f32_126 = __bfloat162float(_cvt_bf16_126);
                    float xs0_250 = _cvt_f32_126;
                    __nv_bfloat16 _cvt_bf16_127 = __float2bfloat16(xn1_249 * _vec_load_31[1]);
                    float _cvt_f32_127 = __bfloat162float(_cvt_bf16_127);
                    float xs1_251 = _cvt_f32_127;
                    accum[62] = xs0_250 * cos_v_246 - xs1_251 * sin_v_247;
                    accum[63] = xs1_251 * cos_v_246 + xs0_250 * sin_v_247;
                    float sumsq_252 = 0.0f;
                    sumsq_252 += accum[64] * accum[64];
                    sumsq_252 += accum[65] * accum[65];
                    sumsq_252 += accum[68] * accum[68];
                    sumsq_252 += accum[69] * accum[69];
                    sumsq_252 += accum[72] * accum[72];
                    sumsq_252 += accum[73] * accum[73];
                    sumsq_252 += accum[76] * accum[76];
                    sumsq_252 += accum[77] * accum[77];
                    sumsq_252 += accum[80] * accum[80];
                    sumsq_252 += accum[81] * accum[81];
                    sumsq_252 += accum[84] * accum[84];
                    sumsq_252 += accum[85] * accum[85];
                    sumsq_252 += accum[88] * accum[88];
                    sumsq_252 += accum[89] * accum[89];
                    sumsq_252 += accum[92] * accum[92];
                    sumsq_252 += accum[93] * accum[93];
                    sumsq_252 += accum[96] * accum[96];
                    sumsq_252 += accum[97] * accum[97];
                    sumsq_252 += accum[100] * accum[100];
                    sumsq_252 += accum[101] * accum[101];
                    sumsq_252 += accum[104] * accum[104];
                    sumsq_252 += accum[105] * accum[105];
                    sumsq_252 += accum[108] * accum[108];
                    sumsq_252 += accum[109] * accum[109];
                    sumsq_252 += accum[112] * accum[112];
                    sumsq_252 += accum[113] * accum[113];
                    sumsq_252 += accum[116] * accum[116];
                    sumsq_252 += accum[117] * accum[117];
                    sumsq_252 += accum[120] * accum[120];
                    sumsq_252 += accum[121] * accum[121];
                    sumsq_252 += accum[124] * accum[124];
                    sumsq_252 += accum[125] * accum[125];
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sumsq_252, 1);
                    float sq1_253 = _shfl_xor_4;
                    sumsq_252 += sq1_253;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, sumsq_252, 2);
                    float sq2_254 = _shfl_xor_5;
                    sumsq_252 += sq2_254;
                    float _rsqrt_2 = rsqrtf(sumsq_252 * 0.0078125f + 1e-06f);
                    float rrms_255 = _rsqrt_2;
                    unsigned int d_col_256 = 2 * q_lane;
                    unsigned int pair_k_257 = q_lane;
                    float _vec_load_32[2];
                    {
                        uint32_t _bf16x2_bits_32;
                        _bf16x2_bits_32 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_256 : k_scale + d_col_256) + 0);
                        (&_vec_load_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_32) << 16);
                        (&_vec_load_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_32) & 0xffff0000u);
                    }
                    float cos_v_258 = cos_sin[cs_lo + pair_k_257];
                    float sin_v_259 = cos_sin[cs_lo + 64 + pair_k_257];
                    __nv_bfloat16 _cvt_bf16_128 = __float2bfloat16(accum[64] * rrms_255);
                    float _cvt_f32_128 = __bfloat162float(_cvt_bf16_128);
                    float xn0_260 = _cvt_f32_128;
                    __nv_bfloat16 _cvt_bf16_129 = __float2bfloat16(accum[65] * rrms_255);
                    float _cvt_f32_129 = __bfloat162float(_cvt_bf16_129);
                    float xn1_261 = _cvt_f32_129;
                    __nv_bfloat16 _cvt_bf16_130 = __float2bfloat16(xn0_260 * _vec_load_32[0]);
                    float _cvt_f32_130 = __bfloat162float(_cvt_bf16_130);
                    float xs0_262 = _cvt_f32_130;
                    __nv_bfloat16 _cvt_bf16_131 = __float2bfloat16(xn1_261 * _vec_load_32[1]);
                    float _cvt_f32_131 = __bfloat162float(_cvt_bf16_131);
                    float xs1_263 = _cvt_f32_131;
                    accum[64] = xs0_262 * cos_v_258 - xs1_263 * sin_v_259;
                    accum[65] = xs1_263 * cos_v_258 + xs0_262 * sin_v_259;
                    unsigned int d_col_264 = 8 + 2 * q_lane;
                    unsigned int pair_k_265 = 4 + q_lane;
                    float _vec_load_33[2];
                    {
                        uint32_t _bf16x2_bits_33;
                        _bf16x2_bits_33 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_264 : k_scale + d_col_264) + 0);
                        (&_vec_load_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_33) << 16);
                        (&_vec_load_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_33) & 0xffff0000u);
                    }
                    float cos_v_266 = cos_sin[cs_lo + pair_k_265];
                    float sin_v_267 = cos_sin[cs_lo + 64 + pair_k_265];
                    __nv_bfloat16 _cvt_bf16_132 = __float2bfloat16(accum[68] * rrms_255);
                    float _cvt_f32_132 = __bfloat162float(_cvt_bf16_132);
                    float xn0_268 = _cvt_f32_132;
                    __nv_bfloat16 _cvt_bf16_133 = __float2bfloat16(accum[69] * rrms_255);
                    float _cvt_f32_133 = __bfloat162float(_cvt_bf16_133);
                    float xn1_269 = _cvt_f32_133;
                    __nv_bfloat16 _cvt_bf16_134 = __float2bfloat16(xn0_268 * _vec_load_33[0]);
                    float _cvt_f32_134 = __bfloat162float(_cvt_bf16_134);
                    float xs0_270 = _cvt_f32_134;
                    __nv_bfloat16 _cvt_bf16_135 = __float2bfloat16(xn1_269 * _vec_load_33[1]);
                    float _cvt_f32_135 = __bfloat162float(_cvt_bf16_135);
                    float xs1_271 = _cvt_f32_135;
                    accum[68] = xs0_270 * cos_v_266 - xs1_271 * sin_v_267;
                    accum[69] = xs1_271 * cos_v_266 + xs0_270 * sin_v_267;
                    unsigned int d_col_272 = 16 + 2 * q_lane;
                    unsigned int pair_k_273 = 8 + q_lane;
                    float _vec_load_34[2];
                    {
                        uint32_t _bf16x2_bits_34;
                        _bf16x2_bits_34 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_272 : k_scale + d_col_272) + 0);
                        (&_vec_load_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_34) << 16);
                        (&_vec_load_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_34) & 0xffff0000u);
                    }
                    float cos_v_274 = cos_sin[cs_lo + pair_k_273];
                    float sin_v_275 = cos_sin[cs_lo + 64 + pair_k_273];
                    __nv_bfloat16 _cvt_bf16_136 = __float2bfloat16(accum[72] * rrms_255);
                    float _cvt_f32_136 = __bfloat162float(_cvt_bf16_136);
                    float xn0_276 = _cvt_f32_136;
                    __nv_bfloat16 _cvt_bf16_137 = __float2bfloat16(accum[73] * rrms_255);
                    float _cvt_f32_137 = __bfloat162float(_cvt_bf16_137);
                    float xn1_277 = _cvt_f32_137;
                    __nv_bfloat16 _cvt_bf16_138 = __float2bfloat16(xn0_276 * _vec_load_34[0]);
                    float _cvt_f32_138 = __bfloat162float(_cvt_bf16_138);
                    float xs0_278 = _cvt_f32_138;
                    __nv_bfloat16 _cvt_bf16_139 = __float2bfloat16(xn1_277 * _vec_load_34[1]);
                    float _cvt_f32_139 = __bfloat162float(_cvt_bf16_139);
                    float xs1_279 = _cvt_f32_139;
                    accum[72] = xs0_278 * cos_v_274 - xs1_279 * sin_v_275;
                    accum[73] = xs1_279 * cos_v_274 + xs0_278 * sin_v_275;
                    unsigned int d_col_280 = 24 + 2 * q_lane;
                    unsigned int pair_k_281 = 12 + q_lane;
                    float _vec_load_35[2];
                    {
                        uint32_t _bf16x2_bits_35;
                        _bf16x2_bits_35 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_280 : k_scale + d_col_280) + 0);
                        (&_vec_load_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_35) << 16);
                        (&_vec_load_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_35) & 0xffff0000u);
                    }
                    float cos_v_282 = cos_sin[cs_lo + pair_k_281];
                    float sin_v_283 = cos_sin[cs_lo + 64 + pair_k_281];
                    __nv_bfloat16 _cvt_bf16_140 = __float2bfloat16(accum[76] * rrms_255);
                    float _cvt_f32_140 = __bfloat162float(_cvt_bf16_140);
                    float xn0_284 = _cvt_f32_140;
                    __nv_bfloat16 _cvt_bf16_141 = __float2bfloat16(accum[77] * rrms_255);
                    float _cvt_f32_141 = __bfloat162float(_cvt_bf16_141);
                    float xn1_285 = _cvt_f32_141;
                    __nv_bfloat16 _cvt_bf16_142 = __float2bfloat16(xn0_284 * _vec_load_35[0]);
                    float _cvt_f32_142 = __bfloat162float(_cvt_bf16_142);
                    float xs0_286 = _cvt_f32_142;
                    __nv_bfloat16 _cvt_bf16_143 = __float2bfloat16(xn1_285 * _vec_load_35[1]);
                    float _cvt_f32_143 = __bfloat162float(_cvt_bf16_143);
                    float xs1_287 = _cvt_f32_143;
                    accum[76] = xs0_286 * cos_v_282 - xs1_287 * sin_v_283;
                    accum[77] = xs1_287 * cos_v_282 + xs0_286 * sin_v_283;
                    unsigned int d_col_288 = 32 + 2 * q_lane;
                    unsigned int pair_k_289 = 16 + q_lane;
                    float _vec_load_36[2];
                    {
                        uint32_t _bf16x2_bits_36;
                        _bf16x2_bits_36 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_288 : k_scale + d_col_288) + 0);
                        (&_vec_load_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_36) << 16);
                        (&_vec_load_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_36) & 0xffff0000u);
                    }
                    float cos_v_290 = cos_sin[cs_lo + pair_k_289];
                    float sin_v_291 = cos_sin[cs_lo + 64 + pair_k_289];
                    __nv_bfloat16 _cvt_bf16_144 = __float2bfloat16(accum[80] * rrms_255);
                    float _cvt_f32_144 = __bfloat162float(_cvt_bf16_144);
                    float xn0_292 = _cvt_f32_144;
                    __nv_bfloat16 _cvt_bf16_145 = __float2bfloat16(accum[81] * rrms_255);
                    float _cvt_f32_145 = __bfloat162float(_cvt_bf16_145);
                    float xn1_293 = _cvt_f32_145;
                    __nv_bfloat16 _cvt_bf16_146 = __float2bfloat16(xn0_292 * _vec_load_36[0]);
                    float _cvt_f32_146 = __bfloat162float(_cvt_bf16_146);
                    float xs0_294 = _cvt_f32_146;
                    __nv_bfloat16 _cvt_bf16_147 = __float2bfloat16(xn1_293 * _vec_load_36[1]);
                    float _cvt_f32_147 = __bfloat162float(_cvt_bf16_147);
                    float xs1_295 = _cvt_f32_147;
                    accum[80] = xs0_294 * cos_v_290 - xs1_295 * sin_v_291;
                    accum[81] = xs1_295 * cos_v_290 + xs0_294 * sin_v_291;
                    unsigned int d_col_296 = 40 + 2 * q_lane;
                    unsigned int pair_k_297 = 20 + q_lane;
                    float _vec_load_37[2];
                    {
                        uint32_t _bf16x2_bits_37;
                        _bf16x2_bits_37 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_296 : k_scale + d_col_296) + 0);
                        (&_vec_load_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_37) << 16);
                        (&_vec_load_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_37) & 0xffff0000u);
                    }
                    float cos_v_298 = cos_sin[cs_lo + pair_k_297];
                    float sin_v_299 = cos_sin[cs_lo + 64 + pair_k_297];
                    __nv_bfloat16 _cvt_bf16_148 = __float2bfloat16(accum[84] * rrms_255);
                    float _cvt_f32_148 = __bfloat162float(_cvt_bf16_148);
                    float xn0_300 = _cvt_f32_148;
                    __nv_bfloat16 _cvt_bf16_149 = __float2bfloat16(accum[85] * rrms_255);
                    float _cvt_f32_149 = __bfloat162float(_cvt_bf16_149);
                    float xn1_301 = _cvt_f32_149;
                    __nv_bfloat16 _cvt_bf16_150 = __float2bfloat16(xn0_300 * _vec_load_37[0]);
                    float _cvt_f32_150 = __bfloat162float(_cvt_bf16_150);
                    float xs0_302 = _cvt_f32_150;
                    __nv_bfloat16 _cvt_bf16_151 = __float2bfloat16(xn1_301 * _vec_load_37[1]);
                    float _cvt_f32_151 = __bfloat162float(_cvt_bf16_151);
                    float xs1_303 = _cvt_f32_151;
                    accum[84] = xs0_302 * cos_v_298 - xs1_303 * sin_v_299;
                    accum[85] = xs1_303 * cos_v_298 + xs0_302 * sin_v_299;
                    unsigned int d_col_304 = 48 + 2 * q_lane;
                    unsigned int pair_k_305 = 24 + q_lane;
                    float _vec_load_38[2];
                    {
                        uint32_t _bf16x2_bits_38;
                        _bf16x2_bits_38 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_304 : k_scale + d_col_304) + 0);
                        (&_vec_load_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_38) << 16);
                        (&_vec_load_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_38) & 0xffff0000u);
                    }
                    float cos_v_306 = cos_sin[cs_lo + pair_k_305];
                    float sin_v_307 = cos_sin[cs_lo + 64 + pair_k_305];
                    __nv_bfloat16 _cvt_bf16_152 = __float2bfloat16(accum[88] * rrms_255);
                    float _cvt_f32_152 = __bfloat162float(_cvt_bf16_152);
                    float xn0_308 = _cvt_f32_152;
                    __nv_bfloat16 _cvt_bf16_153 = __float2bfloat16(accum[89] * rrms_255);
                    float _cvt_f32_153 = __bfloat162float(_cvt_bf16_153);
                    float xn1_309 = _cvt_f32_153;
                    __nv_bfloat16 _cvt_bf16_154 = __float2bfloat16(xn0_308 * _vec_load_38[0]);
                    float _cvt_f32_154 = __bfloat162float(_cvt_bf16_154);
                    float xs0_310 = _cvt_f32_154;
                    __nv_bfloat16 _cvt_bf16_155 = __float2bfloat16(xn1_309 * _vec_load_38[1]);
                    float _cvt_f32_155 = __bfloat162float(_cvt_bf16_155);
                    float xs1_311 = _cvt_f32_155;
                    accum[88] = xs0_310 * cos_v_306 - xs1_311 * sin_v_307;
                    accum[89] = xs1_311 * cos_v_306 + xs0_310 * sin_v_307;
                    unsigned int d_col_312 = 56 + 2 * q_lane;
                    unsigned int pair_k_313 = 28 + q_lane;
                    float _vec_load_39[2];
                    {
                        uint32_t _bf16x2_bits_39;
                        _bf16x2_bits_39 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_312 : k_scale + d_col_312) + 0);
                        (&_vec_load_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_39) << 16);
                        (&_vec_load_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_39) & 0xffff0000u);
                    }
                    float cos_v_314 = cos_sin[cs_lo + pair_k_313];
                    float sin_v_315 = cos_sin[cs_lo + 64 + pair_k_313];
                    __nv_bfloat16 _cvt_bf16_156 = __float2bfloat16(accum[92] * rrms_255);
                    float _cvt_f32_156 = __bfloat162float(_cvt_bf16_156);
                    float xn0_316 = _cvt_f32_156;
                    __nv_bfloat16 _cvt_bf16_157 = __float2bfloat16(accum[93] * rrms_255);
                    float _cvt_f32_157 = __bfloat162float(_cvt_bf16_157);
                    float xn1_317 = _cvt_f32_157;
                    __nv_bfloat16 _cvt_bf16_158 = __float2bfloat16(xn0_316 * _vec_load_39[0]);
                    float _cvt_f32_158 = __bfloat162float(_cvt_bf16_158);
                    float xs0_318 = _cvt_f32_158;
                    __nv_bfloat16 _cvt_bf16_159 = __float2bfloat16(xn1_317 * _vec_load_39[1]);
                    float _cvt_f32_159 = __bfloat162float(_cvt_bf16_159);
                    float xs1_319 = _cvt_f32_159;
                    accum[92] = xs0_318 * cos_v_314 - xs1_319 * sin_v_315;
                    accum[93] = xs1_319 * cos_v_314 + xs0_318 * sin_v_315;
                    unsigned int d_col_320 = 64 + 2 * q_lane;
                    unsigned int pair_k_321 = 32 + q_lane;
                    float _vec_load_40[2];
                    {
                        uint32_t _bf16x2_bits_40;
                        _bf16x2_bits_40 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_320 : k_scale + d_col_320) + 0);
                        (&_vec_load_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_40) << 16);
                        (&_vec_load_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_40) & 0xffff0000u);
                    }
                    float cos_v_322 = cos_sin[cs_lo + pair_k_321];
                    float sin_v_323 = cos_sin[cs_lo + 64 + pair_k_321];
                    __nv_bfloat16 _cvt_bf16_160 = __float2bfloat16(accum[96] * rrms_255);
                    float _cvt_f32_160 = __bfloat162float(_cvt_bf16_160);
                    float xn0_324 = _cvt_f32_160;
                    __nv_bfloat16 _cvt_bf16_161 = __float2bfloat16(accum[97] * rrms_255);
                    float _cvt_f32_161 = __bfloat162float(_cvt_bf16_161);
                    float xn1_325 = _cvt_f32_161;
                    __nv_bfloat16 _cvt_bf16_162 = __float2bfloat16(xn0_324 * _vec_load_40[0]);
                    float _cvt_f32_162 = __bfloat162float(_cvt_bf16_162);
                    float xs0_326 = _cvt_f32_162;
                    __nv_bfloat16 _cvt_bf16_163 = __float2bfloat16(xn1_325 * _vec_load_40[1]);
                    float _cvt_f32_163 = __bfloat162float(_cvt_bf16_163);
                    float xs1_327 = _cvt_f32_163;
                    accum[96] = xs0_326 * cos_v_322 - xs1_327 * sin_v_323;
                    accum[97] = xs1_327 * cos_v_322 + xs0_326 * sin_v_323;
                    unsigned int d_col_328 = 72 + 2 * q_lane;
                    unsigned int pair_k_329 = 36 + q_lane;
                    float _vec_load_41[2];
                    {
                        uint32_t _bf16x2_bits_41;
                        _bf16x2_bits_41 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_328 : k_scale + d_col_328) + 0);
                        (&_vec_load_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_41) << 16);
                        (&_vec_load_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_41) & 0xffff0000u);
                    }
                    float cos_v_330 = cos_sin[cs_lo + pair_k_329];
                    float sin_v_331 = cos_sin[cs_lo + 64 + pair_k_329];
                    __nv_bfloat16 _cvt_bf16_164 = __float2bfloat16(accum[100] * rrms_255);
                    float _cvt_f32_164 = __bfloat162float(_cvt_bf16_164);
                    float xn0_332 = _cvt_f32_164;
                    __nv_bfloat16 _cvt_bf16_165 = __float2bfloat16(accum[101] * rrms_255);
                    float _cvt_f32_165 = __bfloat162float(_cvt_bf16_165);
                    float xn1_333 = _cvt_f32_165;
                    __nv_bfloat16 _cvt_bf16_166 = __float2bfloat16(xn0_332 * _vec_load_41[0]);
                    float _cvt_f32_166 = __bfloat162float(_cvt_bf16_166);
                    float xs0_334 = _cvt_f32_166;
                    __nv_bfloat16 _cvt_bf16_167 = __float2bfloat16(xn1_333 * _vec_load_41[1]);
                    float _cvt_f32_167 = __bfloat162float(_cvt_bf16_167);
                    float xs1_335 = _cvt_f32_167;
                    accum[100] = xs0_334 * cos_v_330 - xs1_335 * sin_v_331;
                    accum[101] = xs1_335 * cos_v_330 + xs0_334 * sin_v_331;
                    unsigned int d_col_336 = 80 + 2 * q_lane;
                    unsigned int pair_k_337 = 40 + q_lane;
                    float _vec_load_42[2];
                    {
                        uint32_t _bf16x2_bits_42;
                        _bf16x2_bits_42 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_336 : k_scale + d_col_336) + 0);
                        (&_vec_load_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_42) << 16);
                        (&_vec_load_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_42) & 0xffff0000u);
                    }
                    float cos_v_338 = cos_sin[cs_lo + pair_k_337];
                    float sin_v_339 = cos_sin[cs_lo + 64 + pair_k_337];
                    __nv_bfloat16 _cvt_bf16_168 = __float2bfloat16(accum[104] * rrms_255);
                    float _cvt_f32_168 = __bfloat162float(_cvt_bf16_168);
                    float xn0_340 = _cvt_f32_168;
                    __nv_bfloat16 _cvt_bf16_169 = __float2bfloat16(accum[105] * rrms_255);
                    float _cvt_f32_169 = __bfloat162float(_cvt_bf16_169);
                    float xn1_341 = _cvt_f32_169;
                    __nv_bfloat16 _cvt_bf16_170 = __float2bfloat16(xn0_340 * _vec_load_42[0]);
                    float _cvt_f32_170 = __bfloat162float(_cvt_bf16_170);
                    float xs0_342 = _cvt_f32_170;
                    __nv_bfloat16 _cvt_bf16_171 = __float2bfloat16(xn1_341 * _vec_load_42[1]);
                    float _cvt_f32_171 = __bfloat162float(_cvt_bf16_171);
                    float xs1_343 = _cvt_f32_171;
                    accum[104] = xs0_342 * cos_v_338 - xs1_343 * sin_v_339;
                    accum[105] = xs1_343 * cos_v_338 + xs0_342 * sin_v_339;
                    unsigned int d_col_344 = 88 + 2 * q_lane;
                    unsigned int pair_k_345 = 44 + q_lane;
                    float _vec_load_43[2];
                    {
                        uint32_t _bf16x2_bits_43;
                        _bf16x2_bits_43 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_344 : k_scale + d_col_344) + 0);
                        (&_vec_load_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_43) << 16);
                        (&_vec_load_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_43) & 0xffff0000u);
                    }
                    float cos_v_346 = cos_sin[cs_lo + pair_k_345];
                    float sin_v_347 = cos_sin[cs_lo + 64 + pair_k_345];
                    __nv_bfloat16 _cvt_bf16_172 = __float2bfloat16(accum[108] * rrms_255);
                    float _cvt_f32_172 = __bfloat162float(_cvt_bf16_172);
                    float xn0_348 = _cvt_f32_172;
                    __nv_bfloat16 _cvt_bf16_173 = __float2bfloat16(accum[109] * rrms_255);
                    float _cvt_f32_173 = __bfloat162float(_cvt_bf16_173);
                    float xn1_349 = _cvt_f32_173;
                    __nv_bfloat16 _cvt_bf16_174 = __float2bfloat16(xn0_348 * _vec_load_43[0]);
                    float _cvt_f32_174 = __bfloat162float(_cvt_bf16_174);
                    float xs0_350 = _cvt_f32_174;
                    __nv_bfloat16 _cvt_bf16_175 = __float2bfloat16(xn1_349 * _vec_load_43[1]);
                    float _cvt_f32_175 = __bfloat162float(_cvt_bf16_175);
                    float xs1_351 = _cvt_f32_175;
                    accum[108] = xs0_350 * cos_v_346 - xs1_351 * sin_v_347;
                    accum[109] = xs1_351 * cos_v_346 + xs0_350 * sin_v_347;
                    unsigned int d_col_352 = 96 + 2 * q_lane;
                    unsigned int pair_k_353 = 48 + q_lane;
                    float _vec_load_44[2];
                    {
                        uint32_t _bf16x2_bits_44;
                        _bf16x2_bits_44 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_352 : k_scale + d_col_352) + 0);
                        (&_vec_load_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_44) << 16);
                        (&_vec_load_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_44) & 0xffff0000u);
                    }
                    float cos_v_354 = cos_sin[cs_lo + pair_k_353];
                    float sin_v_355 = cos_sin[cs_lo + 64 + pair_k_353];
                    __nv_bfloat16 _cvt_bf16_176 = __float2bfloat16(accum[112] * rrms_255);
                    float _cvt_f32_176 = __bfloat162float(_cvt_bf16_176);
                    float xn0_356 = _cvt_f32_176;
                    __nv_bfloat16 _cvt_bf16_177 = __float2bfloat16(accum[113] * rrms_255);
                    float _cvt_f32_177 = __bfloat162float(_cvt_bf16_177);
                    float xn1_357 = _cvt_f32_177;
                    __nv_bfloat16 _cvt_bf16_178 = __float2bfloat16(xn0_356 * _vec_load_44[0]);
                    float _cvt_f32_178 = __bfloat162float(_cvt_bf16_178);
                    float xs0_358 = _cvt_f32_178;
                    __nv_bfloat16 _cvt_bf16_179 = __float2bfloat16(xn1_357 * _vec_load_44[1]);
                    float _cvt_f32_179 = __bfloat162float(_cvt_bf16_179);
                    float xs1_359 = _cvt_f32_179;
                    accum[112] = xs0_358 * cos_v_354 - xs1_359 * sin_v_355;
                    accum[113] = xs1_359 * cos_v_354 + xs0_358 * sin_v_355;
                    unsigned int d_col_360 = 104 + 2 * q_lane;
                    unsigned int pair_k_361 = 52 + q_lane;
                    float _vec_load_45[2];
                    {
                        uint32_t _bf16x2_bits_45;
                        _bf16x2_bits_45 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_360 : k_scale + d_col_360) + 0);
                        (&_vec_load_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_45) << 16);
                        (&_vec_load_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_45) & 0xffff0000u);
                    }
                    float cos_v_362 = cos_sin[cs_lo + pair_k_361];
                    float sin_v_363 = cos_sin[cs_lo + 64 + pair_k_361];
                    __nv_bfloat16 _cvt_bf16_180 = __float2bfloat16(accum[116] * rrms_255);
                    float _cvt_f32_180 = __bfloat162float(_cvt_bf16_180);
                    float xn0_364 = _cvt_f32_180;
                    __nv_bfloat16 _cvt_bf16_181 = __float2bfloat16(accum[117] * rrms_255);
                    float _cvt_f32_181 = __bfloat162float(_cvt_bf16_181);
                    float xn1_365 = _cvt_f32_181;
                    __nv_bfloat16 _cvt_bf16_182 = __float2bfloat16(xn0_364 * _vec_load_45[0]);
                    float _cvt_f32_182 = __bfloat162float(_cvt_bf16_182);
                    float xs0_366 = _cvt_f32_182;
                    __nv_bfloat16 _cvt_bf16_183 = __float2bfloat16(xn1_365 * _vec_load_45[1]);
                    float _cvt_f32_183 = __bfloat162float(_cvt_bf16_183);
                    float xs1_367 = _cvt_f32_183;
                    accum[116] = xs0_366 * cos_v_362 - xs1_367 * sin_v_363;
                    accum[117] = xs1_367 * cos_v_362 + xs0_366 * sin_v_363;
                    unsigned int d_col_368 = 112 + 2 * q_lane;
                    unsigned int pair_k_369 = 56 + q_lane;
                    float _vec_load_46[2];
                    {
                        uint32_t _bf16x2_bits_46;
                        _bf16x2_bits_46 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_368 : k_scale + d_col_368) + 0);
                        (&_vec_load_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_46) << 16);
                        (&_vec_load_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_46) & 0xffff0000u);
                    }
                    float cos_v_370 = cos_sin[cs_lo + pair_k_369];
                    float sin_v_371 = cos_sin[cs_lo + 64 + pair_k_369];
                    __nv_bfloat16 _cvt_bf16_184 = __float2bfloat16(accum[120] * rrms_255);
                    float _cvt_f32_184 = __bfloat162float(_cvt_bf16_184);
                    float xn0_372 = _cvt_f32_184;
                    __nv_bfloat16 _cvt_bf16_185 = __float2bfloat16(accum[121] * rrms_255);
                    float _cvt_f32_185 = __bfloat162float(_cvt_bf16_185);
                    float xn1_373 = _cvt_f32_185;
                    __nv_bfloat16 _cvt_bf16_186 = __float2bfloat16(xn0_372 * _vec_load_46[0]);
                    float _cvt_f32_186 = __bfloat162float(_cvt_bf16_186);
                    float xs0_374 = _cvt_f32_186;
                    __nv_bfloat16 _cvt_bf16_187 = __float2bfloat16(xn1_373 * _vec_load_46[1]);
                    float _cvt_f32_187 = __bfloat162float(_cvt_bf16_187);
                    float xs1_375 = _cvt_f32_187;
                    accum[120] = xs0_374 * cos_v_370 - xs1_375 * sin_v_371;
                    accum[121] = xs1_375 * cos_v_370 + xs0_374 * sin_v_371;
                    unsigned int d_col_376 = 120 + 2 * q_lane;
                    unsigned int pair_k_377 = 60 + q_lane;
                    float _vec_load_47[2];
                    {
                        uint32_t _bf16x2_bits_47;
                        _bf16x2_bits_47 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_376 : k_scale + d_col_376) + 0);
                        (&_vec_load_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_47) << 16);
                        (&_vec_load_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_47) & 0xffff0000u);
                    }
                    float cos_v_378 = cos_sin[cs_lo + pair_k_377];
                    float sin_v_379 = cos_sin[cs_lo + 64 + pair_k_377];
                    __nv_bfloat16 _cvt_bf16_188 = __float2bfloat16(accum[124] * rrms_255);
                    float _cvt_f32_188 = __bfloat162float(_cvt_bf16_188);
                    float xn0_380 = _cvt_f32_188;
                    __nv_bfloat16 _cvt_bf16_189 = __float2bfloat16(accum[125] * rrms_255);
                    float _cvt_f32_189 = __bfloat162float(_cvt_bf16_189);
                    float xn1_381 = _cvt_f32_189;
                    __nv_bfloat16 _cvt_bf16_190 = __float2bfloat16(xn0_380 * _vec_load_47[0]);
                    float _cvt_f32_190 = __bfloat162float(_cvt_bf16_190);
                    float xs0_382 = _cvt_f32_190;
                    __nv_bfloat16 _cvt_bf16_191 = __float2bfloat16(xn1_381 * _vec_load_47[1]);
                    float _cvt_f32_191 = __bfloat162float(_cvt_bf16_191);
                    float xs1_383 = _cvt_f32_191;
                    accum[124] = xs0_382 * cos_v_378 - xs1_383 * sin_v_379;
                    accum[125] = xs1_383 * cos_v_378 + xs0_382 * sin_v_379;
                    float sumsq_384 = 0.0f;
                    sumsq_384 += accum[66] * accum[66];
                    sumsq_384 += accum[67] * accum[67];
                    sumsq_384 += accum[70] * accum[70];
                    sumsq_384 += accum[71] * accum[71];
                    sumsq_384 += accum[74] * accum[74];
                    sumsq_384 += accum[75] * accum[75];
                    sumsq_384 += accum[78] * accum[78];
                    sumsq_384 += accum[79] * accum[79];
                    sumsq_384 += accum[82] * accum[82];
                    sumsq_384 += accum[83] * accum[83];
                    sumsq_384 += accum[86] * accum[86];
                    sumsq_384 += accum[87] * accum[87];
                    sumsq_384 += accum[90] * accum[90];
                    sumsq_384 += accum[91] * accum[91];
                    sumsq_384 += accum[94] * accum[94];
                    sumsq_384 += accum[95] * accum[95];
                    sumsq_384 += accum[98] * accum[98];
                    sumsq_384 += accum[99] * accum[99];
                    sumsq_384 += accum[102] * accum[102];
                    sumsq_384 += accum[103] * accum[103];
                    sumsq_384 += accum[106] * accum[106];
                    sumsq_384 += accum[107] * accum[107];
                    sumsq_384 += accum[110] * accum[110];
                    sumsq_384 += accum[111] * accum[111];
                    sumsq_384 += accum[114] * accum[114];
                    sumsq_384 += accum[115] * accum[115];
                    sumsq_384 += accum[118] * accum[118];
                    sumsq_384 += accum[119] * accum[119];
                    sumsq_384 += accum[122] * accum[122];
                    sumsq_384 += accum[123] * accum[123];
                    sumsq_384 += accum[126] * accum[126];
                    sumsq_384 += accum[127] * accum[127];
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, sumsq_384, 1);
                    float sq1_385 = _shfl_xor_6;
                    sumsq_384 += sq1_385;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, sumsq_384, 2);
                    float sq2_386 = _shfl_xor_7;
                    sumsq_384 += sq2_386;
                    float _rsqrt_3 = rsqrtf(sumsq_384 * 0.0078125f + 1e-06f);
                    float rrms_387 = _rsqrt_3;
                    unsigned int d_col_388 = 2 * q_lane;
                    unsigned int pair_k_389 = q_lane;
                    float _vec_load_48[2];
                    {
                        uint32_t _bf16x2_bits_48;
                        _bf16x2_bits_48 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_388 : k_scale + d_col_388) + 0);
                        (&_vec_load_48[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_48) << 16);
                        (&_vec_load_48[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_48) & 0xffff0000u);
                    }
                    float cos_v_390 = cos_sin[cs_hi + pair_k_389];
                    float sin_v_391 = cos_sin[cs_hi + 64 + pair_k_389];
                    __nv_bfloat16 _cvt_bf16_192 = __float2bfloat16(accum[66] * rrms_387);
                    float _cvt_f32_192 = __bfloat162float(_cvt_bf16_192);
                    float xn0_392 = _cvt_f32_192;
                    __nv_bfloat16 _cvt_bf16_193 = __float2bfloat16(accum[67] * rrms_387);
                    float _cvt_f32_193 = __bfloat162float(_cvt_bf16_193);
                    float xn1_393 = _cvt_f32_193;
                    __nv_bfloat16 _cvt_bf16_194 = __float2bfloat16(xn0_392 * _vec_load_48[0]);
                    float _cvt_f32_194 = __bfloat162float(_cvt_bf16_194);
                    float xs0_394 = _cvt_f32_194;
                    __nv_bfloat16 _cvt_bf16_195 = __float2bfloat16(xn1_393 * _vec_load_48[1]);
                    float _cvt_f32_195 = __bfloat162float(_cvt_bf16_195);
                    float xs1_395 = _cvt_f32_195;
                    accum[66] = xs0_394 * cos_v_390 - xs1_395 * sin_v_391;
                    accum[67] = xs1_395 * cos_v_390 + xs0_394 * sin_v_391;
                    unsigned int d_col_396 = 8 + 2 * q_lane;
                    unsigned int pair_k_397 = 4 + q_lane;
                    float _vec_load_49[2];
                    {
                        uint32_t _bf16x2_bits_49;
                        _bf16x2_bits_49 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_396 : k_scale + d_col_396) + 0);
                        (&_vec_load_49[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_49) << 16);
                        (&_vec_load_49[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_49) & 0xffff0000u);
                    }
                    float cos_v_398 = cos_sin[cs_hi + pair_k_397];
                    float sin_v_399 = cos_sin[cs_hi + 64 + pair_k_397];
                    __nv_bfloat16 _cvt_bf16_196 = __float2bfloat16(accum[70] * rrms_387);
                    float _cvt_f32_196 = __bfloat162float(_cvt_bf16_196);
                    float xn0_400 = _cvt_f32_196;
                    __nv_bfloat16 _cvt_bf16_197 = __float2bfloat16(accum[71] * rrms_387);
                    float _cvt_f32_197 = __bfloat162float(_cvt_bf16_197);
                    float xn1_401 = _cvt_f32_197;
                    __nv_bfloat16 _cvt_bf16_198 = __float2bfloat16(xn0_400 * _vec_load_49[0]);
                    float _cvt_f32_198 = __bfloat162float(_cvt_bf16_198);
                    float xs0_402 = _cvt_f32_198;
                    __nv_bfloat16 _cvt_bf16_199 = __float2bfloat16(xn1_401 * _vec_load_49[1]);
                    float _cvt_f32_199 = __bfloat162float(_cvt_bf16_199);
                    float xs1_403 = _cvt_f32_199;
                    accum[70] = xs0_402 * cos_v_398 - xs1_403 * sin_v_399;
                    accum[71] = xs1_403 * cos_v_398 + xs0_402 * sin_v_399;
                    unsigned int d_col_404 = 16 + 2 * q_lane;
                    unsigned int pair_k_405 = 8 + q_lane;
                    float _vec_load_50[2];
                    {
                        uint32_t _bf16x2_bits_50;
                        _bf16x2_bits_50 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_404 : k_scale + d_col_404) + 0);
                        (&_vec_load_50[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_50) << 16);
                        (&_vec_load_50[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_50) & 0xffff0000u);
                    }
                    float cos_v_406 = cos_sin[cs_hi + pair_k_405];
                    float sin_v_407 = cos_sin[cs_hi + 64 + pair_k_405];
                    __nv_bfloat16 _cvt_bf16_200 = __float2bfloat16(accum[74] * rrms_387);
                    float _cvt_f32_200 = __bfloat162float(_cvt_bf16_200);
                    float xn0_408 = _cvt_f32_200;
                    __nv_bfloat16 _cvt_bf16_201 = __float2bfloat16(accum[75] * rrms_387);
                    float _cvt_f32_201 = __bfloat162float(_cvt_bf16_201);
                    float xn1_409 = _cvt_f32_201;
                    __nv_bfloat16 _cvt_bf16_202 = __float2bfloat16(xn0_408 * _vec_load_50[0]);
                    float _cvt_f32_202 = __bfloat162float(_cvt_bf16_202);
                    float xs0_410 = _cvt_f32_202;
                    __nv_bfloat16 _cvt_bf16_203 = __float2bfloat16(xn1_409 * _vec_load_50[1]);
                    float _cvt_f32_203 = __bfloat162float(_cvt_bf16_203);
                    float xs1_411 = _cvt_f32_203;
                    accum[74] = xs0_410 * cos_v_406 - xs1_411 * sin_v_407;
                    accum[75] = xs1_411 * cos_v_406 + xs0_410 * sin_v_407;
                    unsigned int d_col_412 = 24 + 2 * q_lane;
                    unsigned int pair_k_413 = 12 + q_lane;
                    float _vec_load_51[2];
                    {
                        uint32_t _bf16x2_bits_51;
                        _bf16x2_bits_51 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_412 : k_scale + d_col_412) + 0);
                        (&_vec_load_51[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_51) << 16);
                        (&_vec_load_51[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_51) & 0xffff0000u);
                    }
                    float cos_v_414 = cos_sin[cs_hi + pair_k_413];
                    float sin_v_415 = cos_sin[cs_hi + 64 + pair_k_413];
                    __nv_bfloat16 _cvt_bf16_204 = __float2bfloat16(accum[78] * rrms_387);
                    float _cvt_f32_204 = __bfloat162float(_cvt_bf16_204);
                    float xn0_416 = _cvt_f32_204;
                    __nv_bfloat16 _cvt_bf16_205 = __float2bfloat16(accum[79] * rrms_387);
                    float _cvt_f32_205 = __bfloat162float(_cvt_bf16_205);
                    float xn1_417 = _cvt_f32_205;
                    __nv_bfloat16 _cvt_bf16_206 = __float2bfloat16(xn0_416 * _vec_load_51[0]);
                    float _cvt_f32_206 = __bfloat162float(_cvt_bf16_206);
                    float xs0_418 = _cvt_f32_206;
                    __nv_bfloat16 _cvt_bf16_207 = __float2bfloat16(xn1_417 * _vec_load_51[1]);
                    float _cvt_f32_207 = __bfloat162float(_cvt_bf16_207);
                    float xs1_419 = _cvt_f32_207;
                    accum[78] = xs0_418 * cos_v_414 - xs1_419 * sin_v_415;
                    accum[79] = xs1_419 * cos_v_414 + xs0_418 * sin_v_415;
                    unsigned int d_col_420 = 32 + 2 * q_lane;
                    unsigned int pair_k_421 = 16 + q_lane;
                    float _vec_load_52[2];
                    {
                        uint32_t _bf16x2_bits_52;
                        _bf16x2_bits_52 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_420 : k_scale + d_col_420) + 0);
                        (&_vec_load_52[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_52) << 16);
                        (&_vec_load_52[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_52) & 0xffff0000u);
                    }
                    float cos_v_422 = cos_sin[cs_hi + pair_k_421];
                    float sin_v_423 = cos_sin[cs_hi + 64 + pair_k_421];
                    __nv_bfloat16 _cvt_bf16_208 = __float2bfloat16(accum[82] * rrms_387);
                    float _cvt_f32_208 = __bfloat162float(_cvt_bf16_208);
                    float xn0_424 = _cvt_f32_208;
                    __nv_bfloat16 _cvt_bf16_209 = __float2bfloat16(accum[83] * rrms_387);
                    float _cvt_f32_209 = __bfloat162float(_cvt_bf16_209);
                    float xn1_425 = _cvt_f32_209;
                    __nv_bfloat16 _cvt_bf16_210 = __float2bfloat16(xn0_424 * _vec_load_52[0]);
                    float _cvt_f32_210 = __bfloat162float(_cvt_bf16_210);
                    float xs0_426 = _cvt_f32_210;
                    __nv_bfloat16 _cvt_bf16_211 = __float2bfloat16(xn1_425 * _vec_load_52[1]);
                    float _cvt_f32_211 = __bfloat162float(_cvt_bf16_211);
                    float xs1_427 = _cvt_f32_211;
                    accum[82] = xs0_426 * cos_v_422 - xs1_427 * sin_v_423;
                    accum[83] = xs1_427 * cos_v_422 + xs0_426 * sin_v_423;
                    unsigned int d_col_428 = 40 + 2 * q_lane;
                    unsigned int pair_k_429 = 20 + q_lane;
                    float _vec_load_53[2];
                    {
                        uint32_t _bf16x2_bits_53;
                        _bf16x2_bits_53 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_428 : k_scale + d_col_428) + 0);
                        (&_vec_load_53[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_53) << 16);
                        (&_vec_load_53[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_53) & 0xffff0000u);
                    }
                    float cos_v_430 = cos_sin[cs_hi + pair_k_429];
                    float sin_v_431 = cos_sin[cs_hi + 64 + pair_k_429];
                    __nv_bfloat16 _cvt_bf16_212 = __float2bfloat16(accum[86] * rrms_387);
                    float _cvt_f32_212 = __bfloat162float(_cvt_bf16_212);
                    float xn0_432 = _cvt_f32_212;
                    __nv_bfloat16 _cvt_bf16_213 = __float2bfloat16(accum[87] * rrms_387);
                    float _cvt_f32_213 = __bfloat162float(_cvt_bf16_213);
                    float xn1_433 = _cvt_f32_213;
                    __nv_bfloat16 _cvt_bf16_214 = __float2bfloat16(xn0_432 * _vec_load_53[0]);
                    float _cvt_f32_214 = __bfloat162float(_cvt_bf16_214);
                    float xs0_434 = _cvt_f32_214;
                    __nv_bfloat16 _cvt_bf16_215 = __float2bfloat16(xn1_433 * _vec_load_53[1]);
                    float _cvt_f32_215 = __bfloat162float(_cvt_bf16_215);
                    float xs1_435 = _cvt_f32_215;
                    accum[86] = xs0_434 * cos_v_430 - xs1_435 * sin_v_431;
                    accum[87] = xs1_435 * cos_v_430 + xs0_434 * sin_v_431;
                    unsigned int d_col_436 = 48 + 2 * q_lane;
                    unsigned int pair_k_437 = 24 + q_lane;
                    float _vec_load_54[2];
                    {
                        uint32_t _bf16x2_bits_54;
                        _bf16x2_bits_54 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_436 : k_scale + d_col_436) + 0);
                        (&_vec_load_54[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_54) << 16);
                        (&_vec_load_54[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_54) & 0xffff0000u);
                    }
                    float cos_v_438 = cos_sin[cs_hi + pair_k_437];
                    float sin_v_439 = cos_sin[cs_hi + 64 + pair_k_437];
                    __nv_bfloat16 _cvt_bf16_216 = __float2bfloat16(accum[90] * rrms_387);
                    float _cvt_f32_216 = __bfloat162float(_cvt_bf16_216);
                    float xn0_440 = _cvt_f32_216;
                    __nv_bfloat16 _cvt_bf16_217 = __float2bfloat16(accum[91] * rrms_387);
                    float _cvt_f32_217 = __bfloat162float(_cvt_bf16_217);
                    float xn1_441 = _cvt_f32_217;
                    __nv_bfloat16 _cvt_bf16_218 = __float2bfloat16(xn0_440 * _vec_load_54[0]);
                    float _cvt_f32_218 = __bfloat162float(_cvt_bf16_218);
                    float xs0_442 = _cvt_f32_218;
                    __nv_bfloat16 _cvt_bf16_219 = __float2bfloat16(xn1_441 * _vec_load_54[1]);
                    float _cvt_f32_219 = __bfloat162float(_cvt_bf16_219);
                    float xs1_443 = _cvt_f32_219;
                    accum[90] = xs0_442 * cos_v_438 - xs1_443 * sin_v_439;
                    accum[91] = xs1_443 * cos_v_438 + xs0_442 * sin_v_439;
                    unsigned int d_col_444 = 56 + 2 * q_lane;
                    unsigned int pair_k_445 = 28 + q_lane;
                    float _vec_load_55[2];
                    {
                        uint32_t _bf16x2_bits_55;
                        _bf16x2_bits_55 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_444 : k_scale + d_col_444) + 0);
                        (&_vec_load_55[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_55) << 16);
                        (&_vec_load_55[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_55) & 0xffff0000u);
                    }
                    float cos_v_446 = cos_sin[cs_hi + pair_k_445];
                    float sin_v_447 = cos_sin[cs_hi + 64 + pair_k_445];
                    __nv_bfloat16 _cvt_bf16_220 = __float2bfloat16(accum[94] * rrms_387);
                    float _cvt_f32_220 = __bfloat162float(_cvt_bf16_220);
                    float xn0_448 = _cvt_f32_220;
                    __nv_bfloat16 _cvt_bf16_221 = __float2bfloat16(accum[95] * rrms_387);
                    float _cvt_f32_221 = __bfloat162float(_cvt_bf16_221);
                    float xn1_449 = _cvt_f32_221;
                    __nv_bfloat16 _cvt_bf16_222 = __float2bfloat16(xn0_448 * _vec_load_55[0]);
                    float _cvt_f32_222 = __bfloat162float(_cvt_bf16_222);
                    float xs0_450 = _cvt_f32_222;
                    __nv_bfloat16 _cvt_bf16_223 = __float2bfloat16(xn1_449 * _vec_load_55[1]);
                    float _cvt_f32_223 = __bfloat162float(_cvt_bf16_223);
                    float xs1_451 = _cvt_f32_223;
                    accum[94] = xs0_450 * cos_v_446 - xs1_451 * sin_v_447;
                    accum[95] = xs1_451 * cos_v_446 + xs0_450 * sin_v_447;
                    unsigned int d_col_452 = 64 + 2 * q_lane;
                    unsigned int pair_k_453 = 32 + q_lane;
                    float _vec_load_56[2];
                    {
                        uint32_t _bf16x2_bits_56;
                        _bf16x2_bits_56 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_452 : k_scale + d_col_452) + 0);
                        (&_vec_load_56[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_56) << 16);
                        (&_vec_load_56[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_56) & 0xffff0000u);
                    }
                    float cos_v_454 = cos_sin[cs_hi + pair_k_453];
                    float sin_v_455 = cos_sin[cs_hi + 64 + pair_k_453];
                    __nv_bfloat16 _cvt_bf16_224 = __float2bfloat16(accum[98] * rrms_387);
                    float _cvt_f32_224 = __bfloat162float(_cvt_bf16_224);
                    float xn0_456 = _cvt_f32_224;
                    __nv_bfloat16 _cvt_bf16_225 = __float2bfloat16(accum[99] * rrms_387);
                    float _cvt_f32_225 = __bfloat162float(_cvt_bf16_225);
                    float xn1_457 = _cvt_f32_225;
                    __nv_bfloat16 _cvt_bf16_226 = __float2bfloat16(xn0_456 * _vec_load_56[0]);
                    float _cvt_f32_226 = __bfloat162float(_cvt_bf16_226);
                    float xs0_458 = _cvt_f32_226;
                    __nv_bfloat16 _cvt_bf16_227 = __float2bfloat16(xn1_457 * _vec_load_56[1]);
                    float _cvt_f32_227 = __bfloat162float(_cvt_bf16_227);
                    float xs1_459 = _cvt_f32_227;
                    accum[98] = xs0_458 * cos_v_454 - xs1_459 * sin_v_455;
                    accum[99] = xs1_459 * cos_v_454 + xs0_458 * sin_v_455;
                    unsigned int d_col_460 = 72 + 2 * q_lane;
                    unsigned int pair_k_461 = 36 + q_lane;
                    float _vec_load_57[2];
                    {
                        uint32_t _bf16x2_bits_57;
                        _bf16x2_bits_57 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_460 : k_scale + d_col_460) + 0);
                        (&_vec_load_57[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_57) << 16);
                        (&_vec_load_57[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_57) & 0xffff0000u);
                    }
                    float cos_v_462 = cos_sin[cs_hi + pair_k_461];
                    float sin_v_463 = cos_sin[cs_hi + 64 + pair_k_461];
                    __nv_bfloat16 _cvt_bf16_228 = __float2bfloat16(accum[102] * rrms_387);
                    float _cvt_f32_228 = __bfloat162float(_cvt_bf16_228);
                    float xn0_464 = _cvt_f32_228;
                    __nv_bfloat16 _cvt_bf16_229 = __float2bfloat16(accum[103] * rrms_387);
                    float _cvt_f32_229 = __bfloat162float(_cvt_bf16_229);
                    float xn1_465 = _cvt_f32_229;
                    __nv_bfloat16 _cvt_bf16_230 = __float2bfloat16(xn0_464 * _vec_load_57[0]);
                    float _cvt_f32_230 = __bfloat162float(_cvt_bf16_230);
                    float xs0_466 = _cvt_f32_230;
                    __nv_bfloat16 _cvt_bf16_231 = __float2bfloat16(xn1_465 * _vec_load_57[1]);
                    float _cvt_f32_231 = __bfloat162float(_cvt_bf16_231);
                    float xs1_467 = _cvt_f32_231;
                    accum[102] = xs0_466 * cos_v_462 - xs1_467 * sin_v_463;
                    accum[103] = xs1_467 * cos_v_462 + xs0_466 * sin_v_463;
                    unsigned int d_col_468 = 80 + 2 * q_lane;
                    unsigned int pair_k_469 = 40 + q_lane;
                    float _vec_load_58[2];
                    {
                        uint32_t _bf16x2_bits_58;
                        _bf16x2_bits_58 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_468 : k_scale + d_col_468) + 0);
                        (&_vec_load_58[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_58) << 16);
                        (&_vec_load_58[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_58) & 0xffff0000u);
                    }
                    float cos_v_470 = cos_sin[cs_hi + pair_k_469];
                    float sin_v_471 = cos_sin[cs_hi + 64 + pair_k_469];
                    __nv_bfloat16 _cvt_bf16_232 = __float2bfloat16(accum[106] * rrms_387);
                    float _cvt_f32_232 = __bfloat162float(_cvt_bf16_232);
                    float xn0_472 = _cvt_f32_232;
                    __nv_bfloat16 _cvt_bf16_233 = __float2bfloat16(accum[107] * rrms_387);
                    float _cvt_f32_233 = __bfloat162float(_cvt_bf16_233);
                    float xn1_473 = _cvt_f32_233;
                    __nv_bfloat16 _cvt_bf16_234 = __float2bfloat16(xn0_472 * _vec_load_58[0]);
                    float _cvt_f32_234 = __bfloat162float(_cvt_bf16_234);
                    float xs0_474 = _cvt_f32_234;
                    __nv_bfloat16 _cvt_bf16_235 = __float2bfloat16(xn1_473 * _vec_load_58[1]);
                    float _cvt_f32_235 = __bfloat162float(_cvt_bf16_235);
                    float xs1_475 = _cvt_f32_235;
                    accum[106] = xs0_474 * cos_v_470 - xs1_475 * sin_v_471;
                    accum[107] = xs1_475 * cos_v_470 + xs0_474 * sin_v_471;
                    unsigned int d_col_476 = 88 + 2 * q_lane;
                    unsigned int pair_k_477 = 44 + q_lane;
                    float _vec_load_59[2];
                    {
                        uint32_t _bf16x2_bits_59;
                        _bf16x2_bits_59 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_476 : k_scale + d_col_476) + 0);
                        (&_vec_load_59[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_59) << 16);
                        (&_vec_load_59[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_59) & 0xffff0000u);
                    }
                    float cos_v_478 = cos_sin[cs_hi + pair_k_477];
                    float sin_v_479 = cos_sin[cs_hi + 64 + pair_k_477];
                    __nv_bfloat16 _cvt_bf16_236 = __float2bfloat16(accum[110] * rrms_387);
                    float _cvt_f32_236 = __bfloat162float(_cvt_bf16_236);
                    float xn0_480 = _cvt_f32_236;
                    __nv_bfloat16 _cvt_bf16_237 = __float2bfloat16(accum[111] * rrms_387);
                    float _cvt_f32_237 = __bfloat162float(_cvt_bf16_237);
                    float xn1_481 = _cvt_f32_237;
                    __nv_bfloat16 _cvt_bf16_238 = __float2bfloat16(xn0_480 * _vec_load_59[0]);
                    float _cvt_f32_238 = __bfloat162float(_cvt_bf16_238);
                    float xs0_482 = _cvt_f32_238;
                    __nv_bfloat16 _cvt_bf16_239 = __float2bfloat16(xn1_481 * _vec_load_59[1]);
                    float _cvt_f32_239 = __bfloat162float(_cvt_bf16_239);
                    float xs1_483 = _cvt_f32_239;
                    accum[110] = xs0_482 * cos_v_478 - xs1_483 * sin_v_479;
                    accum[111] = xs1_483 * cos_v_478 + xs0_482 * sin_v_479;
                    unsigned int d_col_484 = 96 + 2 * q_lane;
                    unsigned int pair_k_485 = 48 + q_lane;
                    float _vec_load_60[2];
                    {
                        uint32_t _bf16x2_bits_60;
                        _bf16x2_bits_60 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_484 : k_scale + d_col_484) + 0);
                        (&_vec_load_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_60) << 16);
                        (&_vec_load_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_60) & 0xffff0000u);
                    }
                    float cos_v_486 = cos_sin[cs_hi + pair_k_485];
                    float sin_v_487 = cos_sin[cs_hi + 64 + pair_k_485];
                    __nv_bfloat16 _cvt_bf16_240 = __float2bfloat16(accum[114] * rrms_387);
                    float _cvt_f32_240 = __bfloat162float(_cvt_bf16_240);
                    float xn0_488 = _cvt_f32_240;
                    __nv_bfloat16 _cvt_bf16_241 = __float2bfloat16(accum[115] * rrms_387);
                    float _cvt_f32_241 = __bfloat162float(_cvt_bf16_241);
                    float xn1_489 = _cvt_f32_241;
                    __nv_bfloat16 _cvt_bf16_242 = __float2bfloat16(xn0_488 * _vec_load_60[0]);
                    float _cvt_f32_242 = __bfloat162float(_cvt_bf16_242);
                    float xs0_490 = _cvt_f32_242;
                    __nv_bfloat16 _cvt_bf16_243 = __float2bfloat16(xn1_489 * _vec_load_60[1]);
                    float _cvt_f32_243 = __bfloat162float(_cvt_bf16_243);
                    float xs1_491 = _cvt_f32_243;
                    accum[114] = xs0_490 * cos_v_486 - xs1_491 * sin_v_487;
                    accum[115] = xs1_491 * cos_v_486 + xs0_490 * sin_v_487;
                    unsigned int d_col_492 = 104 + 2 * q_lane;
                    unsigned int pair_k_493 = 52 + q_lane;
                    float _vec_load_61[2];
                    {
                        uint32_t _bf16x2_bits_61;
                        _bf16x2_bits_61 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_492 : k_scale + d_col_492) + 0);
                        (&_vec_load_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_61) << 16);
                        (&_vec_load_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_61) & 0xffff0000u);
                    }
                    float cos_v_494 = cos_sin[cs_hi + pair_k_493];
                    float sin_v_495 = cos_sin[cs_hi + 64 + pair_k_493];
                    __nv_bfloat16 _cvt_bf16_244 = __float2bfloat16(accum[118] * rrms_387);
                    float _cvt_f32_244 = __bfloat162float(_cvt_bf16_244);
                    float xn0_496 = _cvt_f32_244;
                    __nv_bfloat16 _cvt_bf16_245 = __float2bfloat16(accum[119] * rrms_387);
                    float _cvt_f32_245 = __bfloat162float(_cvt_bf16_245);
                    float xn1_497 = _cvt_f32_245;
                    __nv_bfloat16 _cvt_bf16_246 = __float2bfloat16(xn0_496 * _vec_load_61[0]);
                    float _cvt_f32_246 = __bfloat162float(_cvt_bf16_246);
                    float xs0_498 = _cvt_f32_246;
                    __nv_bfloat16 _cvt_bf16_247 = __float2bfloat16(xn1_497 * _vec_load_61[1]);
                    float _cvt_f32_247 = __bfloat162float(_cvt_bf16_247);
                    float xs1_499 = _cvt_f32_247;
                    accum[118] = xs0_498 * cos_v_494 - xs1_499 * sin_v_495;
                    accum[119] = xs1_499 * cos_v_494 + xs0_498 * sin_v_495;
                    unsigned int d_col_500 = 112 + 2 * q_lane;
                    unsigned int pair_k_501 = 56 + q_lane;
                    float _vec_load_62[2];
                    {
                        uint32_t _bf16x2_bits_62;
                        _bf16x2_bits_62 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_500 : k_scale + d_col_500) + 0);
                        (&_vec_load_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_62) << 16);
                        (&_vec_load_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_62) & 0xffff0000u);
                    }
                    float cos_v_502 = cos_sin[cs_hi + pair_k_501];
                    float sin_v_503 = cos_sin[cs_hi + 64 + pair_k_501];
                    __nv_bfloat16 _cvt_bf16_248 = __float2bfloat16(accum[122] * rrms_387);
                    float _cvt_f32_248 = __bfloat162float(_cvt_bf16_248);
                    float xn0_504 = _cvt_f32_248;
                    __nv_bfloat16 _cvt_bf16_249 = __float2bfloat16(accum[123] * rrms_387);
                    float _cvt_f32_249 = __bfloat162float(_cvt_bf16_249);
                    float xn1_505 = _cvt_f32_249;
                    __nv_bfloat16 _cvt_bf16_250 = __float2bfloat16(xn0_504 * _vec_load_62[0]);
                    float _cvt_f32_250 = __bfloat162float(_cvt_bf16_250);
                    float xs0_506 = _cvt_f32_250;
                    __nv_bfloat16 _cvt_bf16_251 = __float2bfloat16(xn1_505 * _vec_load_62[1]);
                    float _cvt_f32_251 = __bfloat162float(_cvt_bf16_251);
                    float xs1_507 = _cvt_f32_251;
                    accum[122] = xs0_506 * cos_v_502 - xs1_507 * sin_v_503;
                    accum[123] = xs1_507 * cos_v_502 + xs0_506 * sin_v_503;
                    unsigned int d_col_508 = 120 + 2 * q_lane;
                    unsigned int pair_k_509 = 60 + q_lane;
                    float _vec_load_63[2];
                    {
                        uint32_t _bf16x2_bits_63;
                        _bf16x2_bits_63 = *reinterpret_cast<const uint32_t*>(((norm_kind == 1) ? q_scale + d_col_508 : k_scale + d_col_508) + 0);
                        (&_vec_load_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_63) << 16);
                        (&_vec_load_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_63) & 0xffff0000u);
                    }
                    float cos_v_510 = cos_sin[cs_hi + pair_k_509];
                    float sin_v_511 = cos_sin[cs_hi + 64 + pair_k_509];
                    __nv_bfloat16 _cvt_bf16_252 = __float2bfloat16(accum[126] * rrms_387);
                    float _cvt_f32_252 = __bfloat162float(_cvt_bf16_252);
                    float xn0_512 = _cvt_f32_252;
                    __nv_bfloat16 _cvt_bf16_253 = __float2bfloat16(accum[127] * rrms_387);
                    float _cvt_f32_253 = __bfloat162float(_cvt_bf16_253);
                    float xn1_513 = _cvt_f32_253;
                    __nv_bfloat16 _cvt_bf16_254 = __float2bfloat16(xn0_512 * _vec_load_63[0]);
                    float _cvt_f32_254 = __bfloat162float(_cvt_bf16_254);
                    float xs0_514 = _cvt_f32_254;
                    __nv_bfloat16 _cvt_bf16_255 = __float2bfloat16(xn1_513 * _vec_load_63[1]);
                    float _cvt_f32_255 = __bfloat162float(_cvt_bf16_255);
                    float xs1_515 = _cvt_f32_255;
                    accum[126] = xs0_514 * cos_v_510 - xs1_515 * sin_v_511;
                    accum[127] = xs1_515 * cos_v_510 + xs0_514 * sin_v_511;
                }
                __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(dst_ptrs[dest_c]);
                unsigned int d0 = q_lane * 8;
                long long col_off = (long long)d0;
                __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(accum[0], accum[1]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_0)[0];
                __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(accum[4], accum[5]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_1)[0];
                __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(accum[8], accum[9]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_2)[0];
                __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(accum[12], accum[13]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_3)[0];
                unsigned int x0 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, x0, 1);
                unsigned int y0 = _shfl_xor_8;
                unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, x1, 1);
                unsigned int y1 = _shfl_xor_9;
                pk[0] = y0 & m_odd | pk[0] & m_even;
                pk[1] = y0 & m_even | pk[1] & m_odd;
                pk[2] = y1 & m_odd | pk[2] & m_even;
                pk[3] = y1 & m_even | pk[3] & m_odd;
                unsigned int x2 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, x2, 2);
                unsigned int y2 = _shfl_xor_10;
                unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, x3, 2);
                unsigned int y3 = _shfl_xor_11;
                pk[0] = y2 & m_hi | pk[0] & m_lo;
                pk[2] = y2 & m_lo | pk[2] & m_hi;
                pk[1] = y3 & m_hi | pk[1] & m_lo;
                pk[3] = y3 & m_lo | pk[3] & m_hi;
                if (local_lo < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_lo + col_off)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(accum[2], accum[3]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_4)[0];
                __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(accum[6], accum[7]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_5)[0];
                __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(accum[10], accum[11]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_6)[0];
                __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(accum[14], accum[15]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_7)[0];
                unsigned int x0_0 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_1 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, x0_0, 1);
                unsigned int y0_2 = _shfl_xor_12;
                unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, x1_1, 1);
                unsigned int y1_3 = _shfl_xor_13;
                pk[0] = y0_2 & m_odd | pk[0] & m_even;
                pk[1] = y0_2 & m_even | pk[1] & m_odd;
                pk[2] = y1_3 & m_odd | pk[2] & m_even;
                pk[3] = y1_3 & m_even | pk[3] & m_odd;
                unsigned int x2_4 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_5 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, x2_4, 2);
                unsigned int y2_6 = _shfl_xor_14;
                unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, x3_5, 2);
                unsigned int y3_7 = _shfl_xor_15;
                pk[0] = y2_6 & m_hi | pk[0] & m_lo;
                pk[2] = y2_6 & m_lo | pk[2] & m_hi;
                pk[1] = y3_7 & m_hi | pk[1] & m_lo;
                pk[3] = y3_7 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                unsigned int d0_8 = 32 + q_lane * 8;
                long long col_off_9 = (long long)d0_8;
                __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(accum[16], accum[17]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_8)[0];
                __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(accum[20], accum[21]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_9)[0];
                __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(accum[24], accum[25]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_10)[0];
                __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(accum[28], accum[29]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_11)[0];
                unsigned int x0_10 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_11 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, x0_10, 1);
                unsigned int y0_12 = _shfl_xor_16;
                unsigned int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, x1_11, 1);
                unsigned int y1_13 = _shfl_xor_17;
                pk[0] = y0_12 & m_odd | pk[0] & m_even;
                pk[1] = y0_12 & m_even | pk[1] & m_odd;
                pk[2] = y1_13 & m_odd | pk[2] & m_even;
                pk[3] = y1_13 & m_even | pk[3] & m_odd;
                unsigned int x2_14 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_15 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, x2_14, 2);
                unsigned int y2_16 = _shfl_xor_18;
                unsigned int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, x3_15, 2);
                unsigned int y3_17 = _shfl_xor_19;
                pk[0] = y2_16 & m_hi | pk[0] & m_lo;
                pk[2] = y2_16 & m_lo | pk[2] & m_hi;
                pk[1] = y3_17 & m_hi | pk[1] & m_lo;
                pk[3] = y3_17 & m_lo | pk[3] & m_hi;
                if (local_lo < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_lo + col_off_9)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(accum[18], accum[19]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_12)[0];
                __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(accum[22], accum[23]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_13)[0];
                __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(accum[26], accum[27]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_14)[0];
                __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(accum[30], accum[31]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_15)[0];
                unsigned int x0_18 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_19 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, x0_18, 1);
                unsigned int y0_20 = _shfl_xor_20;
                unsigned int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, x1_19, 1);
                unsigned int y1_21 = _shfl_xor_21;
                pk[0] = y0_20 & m_odd | pk[0] & m_even;
                pk[1] = y0_20 & m_even | pk[1] & m_odd;
                pk[2] = y1_21 & m_odd | pk[2] & m_even;
                pk[3] = y1_21 & m_even | pk[3] & m_odd;
                unsigned int x2_22 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_23 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, x2_22, 2);
                unsigned int y2_24 = _shfl_xor_22;
                unsigned int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, x3_23, 2);
                unsigned int y3_25 = _shfl_xor_23;
                pk[0] = y2_24 & m_hi | pk[0] & m_lo;
                pk[2] = y2_24 & m_lo | pk[2] & m_hi;
                pk[1] = y3_25 & m_hi | pk[1] & m_lo;
                pk[3] = y3_25 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off_9)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                unsigned int d0_26 = 64 + q_lane * 8;
                long long col_off_27 = (long long)d0_26;
                __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(accum[32], accum[33]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_16)[0];
                __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(accum[36], accum[37]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_17)[0];
                __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(accum[40], accum[41]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_18)[0];
                __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(accum[44], accum[45]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_19)[0];
                unsigned int x0_28 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_29 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, x0_28, 1);
                unsigned int y0_30 = _shfl_xor_24;
                unsigned int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, x1_29, 1);
                unsigned int y1_31 = _shfl_xor_25;
                pk[0] = y0_30 & m_odd | pk[0] & m_even;
                pk[1] = y0_30 & m_even | pk[1] & m_odd;
                pk[2] = y1_31 & m_odd | pk[2] & m_even;
                pk[3] = y1_31 & m_even | pk[3] & m_odd;
                unsigned int x2_32 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_33 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, x2_32, 2);
                unsigned int y2_34 = _shfl_xor_26;
                unsigned int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, x3_33, 2);
                unsigned int y3_35 = _shfl_xor_27;
                pk[0] = y2_34 & m_hi | pk[0] & m_lo;
                pk[2] = y2_34 & m_lo | pk[2] & m_hi;
                pk[1] = y3_35 & m_hi | pk[1] & m_lo;
                pk[3] = y3_35 & m_lo | pk[3] & m_hi;
                if (local_lo < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_lo + col_off_27)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(accum[34], accum[35]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_20)[0];
                __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(accum[38], accum[39]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_21)[0];
                __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(accum[42], accum[43]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_22)[0];
                __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(accum[46], accum[47]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_23)[0];
                unsigned int x0_36 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_37 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, x0_36, 1);
                unsigned int y0_38 = _shfl_xor_28;
                unsigned int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, x1_37, 1);
                unsigned int y1_39 = _shfl_xor_29;
                pk[0] = y0_38 & m_odd | pk[0] & m_even;
                pk[1] = y0_38 & m_even | pk[1] & m_odd;
                pk[2] = y1_39 & m_odd | pk[2] & m_even;
                pk[3] = y1_39 & m_even | pk[3] & m_odd;
                unsigned int x2_40 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_41 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, x2_40, 2);
                unsigned int y2_42 = _shfl_xor_30;
                unsigned int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, x3_41, 2);
                unsigned int y3_43 = _shfl_xor_31;
                pk[0] = y2_42 & m_hi | pk[0] & m_lo;
                pk[2] = y2_42 & m_lo | pk[2] & m_hi;
                pk[1] = y3_43 & m_hi | pk[1] & m_lo;
                pk[3] = y3_43 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off_27)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                unsigned int d0_44 = 96 + q_lane * 8;
                long long col_off_45 = (long long)d0_44;
                __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(accum[48], accum[49]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_24)[0];
                __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(accum[52], accum[53]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_25)[0];
                __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(accum[56], accum[57]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_26)[0];
                __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(accum[60], accum[61]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_27)[0];
                unsigned int x0_46 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_47 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, x0_46, 1);
                unsigned int y0_48 = _shfl_xor_32;
                unsigned int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, x1_47, 1);
                unsigned int y1_49 = _shfl_xor_33;
                pk[0] = y0_48 & m_odd | pk[0] & m_even;
                pk[1] = y0_48 & m_even | pk[1] & m_odd;
                pk[2] = y1_49 & m_odd | pk[2] & m_even;
                pk[3] = y1_49 & m_even | pk[3] & m_odd;
                unsigned int x2_50 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_51 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, x2_50, 2);
                unsigned int y2_52 = _shfl_xor_34;
                unsigned int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, x3_51, 2);
                unsigned int y3_53 = _shfl_xor_35;
                pk[0] = y2_52 & m_hi | pk[0] & m_lo;
                pk[2] = y2_52 & m_lo | pk[2] & m_hi;
                pk[1] = y3_53 & m_hi | pk[1] & m_lo;
                pk[3] = y3_53 & m_lo | pk[3] & m_hi;
                if (local_lo < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_lo + col_off_45)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(accum[50], accum[51]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_28)[0];
                __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(accum[54], accum[55]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_29)[0];
                __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(accum[58], accum[59]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_30)[0];
                __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(accum[62], accum[63]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_31)[0];
                unsigned int x0_54 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_55 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, x0_54, 1);
                unsigned int y0_56 = _shfl_xor_36;
                unsigned int _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, x1_55, 1);
                unsigned int y1_57 = _shfl_xor_37;
                pk[0] = y0_56 & m_odd | pk[0] & m_even;
                pk[1] = y0_56 & m_even | pk[1] & m_odd;
                pk[2] = y1_57 & m_odd | pk[2] & m_even;
                pk[3] = y1_57 & m_even | pk[3] & m_odd;
                unsigned int x2_58 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_59 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, x2_58, 2);
                unsigned int y2_60 = _shfl_xor_38;
                unsigned int _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, x3_59, 2);
                unsigned int y3_61 = _shfl_xor_39;
                pk[0] = y2_60 & m_hi | pk[0] & m_lo;
                pk[2] = y2_60 & m_lo | pk[2] & m_hi;
                pk[1] = y3_61 & m_hi | pk[1] & m_lo;
                pk[3] = y3_61 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off_45)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                unsigned int d0_62 = q_lane * 8;
                long long col_off_63 = (long long)d0_62 + head_stride_e;
                __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(accum[64], accum[65]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_32)[0];
                __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(accum[68], accum[69]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_33)[0];
                __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(accum[72], accum[73]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_34)[0];
                __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(accum[76], accum[77]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_35)[0];
                unsigned int x0_64 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_65 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, x0_64, 1);
                unsigned int y0_66 = _shfl_xor_40;
                unsigned int _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, x1_65, 1);
                unsigned int y1_67 = _shfl_xor_41;
                pk[0] = y0_66 & m_odd | pk[0] & m_even;
                pk[1] = y0_66 & m_even | pk[1] & m_odd;
                pk[2] = y1_67 & m_odd | pk[2] & m_even;
                pk[3] = y1_67 & m_even | pk[3] & m_odd;
                unsigned int x2_68 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_69 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, x2_68, 2);
                unsigned int y2_70 = _shfl_xor_42;
                unsigned int _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, x3_69, 2);
                unsigned int y3_71 = _shfl_xor_43;
                pk[0] = y2_70 & m_hi | pk[0] & m_lo;
                pk[2] = y2_70 & m_lo | pk[2] & m_hi;
                pk[1] = y3_71 & m_hi | pk[1] & m_lo;
                pk[3] = y3_71 & m_lo | pk[3] & m_hi;
                if (local_lo < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_lo + col_off_63)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(accum[66], accum[67]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_36)[0];
                __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(accum[70], accum[71]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_37)[0];
                __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(accum[74], accum[75]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_38)[0];
                __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(accum[78], accum[79]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_39)[0];
                unsigned int x0_72 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_73 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, x0_72, 1);
                unsigned int y0_74 = _shfl_xor_44;
                unsigned int _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, x1_73, 1);
                unsigned int y1_75 = _shfl_xor_45;
                pk[0] = y0_74 & m_odd | pk[0] & m_even;
                pk[1] = y0_74 & m_even | pk[1] & m_odd;
                pk[2] = y1_75 & m_odd | pk[2] & m_even;
                pk[3] = y1_75 & m_even | pk[3] & m_odd;
                unsigned int x2_76 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_77 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, x2_76, 2);
                unsigned int y2_78 = _shfl_xor_46;
                unsigned int _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, x3_77, 2);
                unsigned int y3_79 = _shfl_xor_47;
                pk[0] = y2_78 & m_hi | pk[0] & m_lo;
                pk[2] = y2_78 & m_lo | pk[2] & m_hi;
                pk[1] = y3_79 & m_hi | pk[1] & m_lo;
                pk[3] = y3_79 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off_63)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                unsigned int d0_80 = 32 + q_lane * 8;
                long long col_off_81 = (long long)d0_80 + head_stride_e;
                __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(accum[80], accum[81]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_40)[0];
                __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(accum[84], accum[85]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_41)[0];
                __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(accum[88], accum[89]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_42)[0];
                __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(accum[92], accum[93]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_43)[0];
                unsigned int x0_82 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_83 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, x0_82, 1);
                unsigned int y0_84 = _shfl_xor_48;
                unsigned int _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, x1_83, 1);
                unsigned int y1_85 = _shfl_xor_49;
                pk[0] = y0_84 & m_odd | pk[0] & m_even;
                pk[1] = y0_84 & m_even | pk[1] & m_odd;
                pk[2] = y1_85 & m_odd | pk[2] & m_even;
                pk[3] = y1_85 & m_even | pk[3] & m_odd;
                unsigned int x2_86 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_87 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, x2_86, 2);
                unsigned int y2_88 = _shfl_xor_50;
                unsigned int _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, x3_87, 2);
                unsigned int y3_89 = _shfl_xor_51;
                pk[0] = y2_88 & m_hi | pk[0] & m_lo;
                pk[2] = y2_88 & m_lo | pk[2] & m_hi;
                pk[1] = y3_89 & m_hi | pk[1] & m_lo;
                pk[3] = y3_89 & m_lo | pk[3] & m_hi;
                if (local_lo < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_lo + col_off_81)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(accum[82], accum[83]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_44)[0];
                __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(accum[86], accum[87]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_45)[0];
                __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(accum[90], accum[91]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_46)[0];
                __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(accum[94], accum[95]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_47)[0];
                unsigned int x0_90 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_91 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, x0_90, 1);
                unsigned int y0_92 = _shfl_xor_52;
                unsigned int _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, x1_91, 1);
                unsigned int y1_93 = _shfl_xor_53;
                pk[0] = y0_92 & m_odd | pk[0] & m_even;
                pk[1] = y0_92 & m_even | pk[1] & m_odd;
                pk[2] = y1_93 & m_odd | pk[2] & m_even;
                pk[3] = y1_93 & m_even | pk[3] & m_odd;
                unsigned int x2_94 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_95 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, x2_94, 2);
                unsigned int y2_96 = _shfl_xor_54;
                unsigned int _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, x3_95, 2);
                unsigned int y3_97 = _shfl_xor_55;
                pk[0] = y2_96 & m_hi | pk[0] & m_lo;
                pk[2] = y2_96 & m_lo | pk[2] & m_hi;
                pk[1] = y3_97 & m_hi | pk[1] & m_lo;
                pk[3] = y3_97 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off_81)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                unsigned int d0_98 = 64 + q_lane * 8;
                long long col_off_99 = (long long)d0_98 + head_stride_e;
                __nv_bfloat162 _bf16x2_48 = __float22bfloat162_rn(make_float2(accum[96], accum[97]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_48)[0];
                __nv_bfloat162 _bf16x2_49 = __float22bfloat162_rn(make_float2(accum[100], accum[101]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_49)[0];
                __nv_bfloat162 _bf16x2_50 = __float22bfloat162_rn(make_float2(accum[104], accum[105]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_50)[0];
                __nv_bfloat162 _bf16x2_51 = __float22bfloat162_rn(make_float2(accum[108], accum[109]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_51)[0];
                unsigned int x0_100 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_101 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, x0_100, 1);
                unsigned int y0_102 = _shfl_xor_56;
                unsigned int _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, x1_101, 1);
                unsigned int y1_103 = _shfl_xor_57;
                pk[0] = y0_102 & m_odd | pk[0] & m_even;
                pk[1] = y0_102 & m_even | pk[1] & m_odd;
                pk[2] = y1_103 & m_odd | pk[2] & m_even;
                pk[3] = y1_103 & m_even | pk[3] & m_odd;
                unsigned int x2_104 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_105 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, x2_104, 2);
                unsigned int y2_106 = _shfl_xor_58;
                unsigned int _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, x3_105, 2);
                unsigned int y3_107 = _shfl_xor_59;
                pk[0] = y2_106 & m_hi | pk[0] & m_lo;
                pk[2] = y2_106 & m_lo | pk[2] & m_hi;
                pk[1] = y3_107 & m_hi | pk[1] & m_lo;
                pk[3] = y3_107 & m_lo | pk[3] & m_hi;
                if (local_lo < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_lo + col_off_99)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                __nv_bfloat162 _bf16x2_52 = __float22bfloat162_rn(make_float2(accum[98], accum[99]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_52)[0];
                __nv_bfloat162 _bf16x2_53 = __float22bfloat162_rn(make_float2(accum[102], accum[103]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_53)[0];
                __nv_bfloat162 _bf16x2_54 = __float22bfloat162_rn(make_float2(accum[106], accum[107]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_54)[0];
                __nv_bfloat162 _bf16x2_55 = __float22bfloat162_rn(make_float2(accum[110], accum[111]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_55)[0];
                unsigned int x0_108 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_109 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, x0_108, 1);
                unsigned int y0_110 = _shfl_xor_60;
                unsigned int _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, x1_109, 1);
                unsigned int y1_111 = _shfl_xor_61;
                pk[0] = y0_110 & m_odd | pk[0] & m_even;
                pk[1] = y0_110 & m_even | pk[1] & m_odd;
                pk[2] = y1_111 & m_odd | pk[2] & m_even;
                pk[3] = y1_111 & m_even | pk[3] & m_odd;
                unsigned int x2_112 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_113 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, x2_112, 2);
                unsigned int y2_114 = _shfl_xor_62;
                unsigned int _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, x3_113, 2);
                unsigned int y3_115 = _shfl_xor_63;
                pk[0] = y2_114 & m_hi | pk[0] & m_lo;
                pk[2] = y2_114 & m_lo | pk[2] & m_hi;
                pk[1] = y3_115 & m_hi | pk[1] & m_lo;
                pk[3] = y3_115 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off_99)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                unsigned int d0_116 = 96 + q_lane * 8;
                long long col_off_117 = (long long)d0_116 + head_stride_e;
                __nv_bfloat162 _bf16x2_56 = __float22bfloat162_rn(make_float2(accum[112], accum[113]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_56)[0];
                __nv_bfloat162 _bf16x2_57 = __float22bfloat162_rn(make_float2(accum[116], accum[117]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_57)[0];
                __nv_bfloat162 _bf16x2_58 = __float22bfloat162_rn(make_float2(accum[120], accum[121]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_58)[0];
                __nv_bfloat162 _bf16x2_59 = __float22bfloat162_rn(make_float2(accum[124], accum[125]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_59)[0];
                unsigned int x0_118 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_119 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, x0_118, 1);
                unsigned int y0_120 = _shfl_xor_64;
                unsigned int _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, x1_119, 1);
                unsigned int y1_121 = _shfl_xor_65;
                pk[0] = y0_120 & m_odd | pk[0] & m_even;
                pk[1] = y0_120 & m_even | pk[1] & m_odd;
                pk[2] = y1_121 & m_odd | pk[2] & m_even;
                pk[3] = y1_121 & m_even | pk[3] & m_odd;
                unsigned int x2_122 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_123 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, x2_122, 2);
                unsigned int y2_124 = _shfl_xor_66;
                unsigned int _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, x3_123, 2);
                unsigned int y3_125 = _shfl_xor_67;
                pk[0] = y2_124 & m_hi | pk[0] & m_lo;
                pk[2] = y2_124 & m_lo | pk[2] & m_hi;
                pk[1] = y3_125 & m_hi | pk[1] & m_lo;
                pk[3] = y3_125 & m_lo | pk[3] & m_hi;
                if (local_lo < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_lo + col_off_117)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                __nv_bfloat162 _bf16x2_60 = __float22bfloat162_rn(make_float2(accum[114], accum[115]));
                pk[0] = reinterpret_cast<unsigned int*>(&_bf16x2_60)[0];
                __nv_bfloat162 _bf16x2_61 = __float22bfloat162_rn(make_float2(accum[118], accum[119]));
                pk[1] = reinterpret_cast<unsigned int*>(&_bf16x2_61)[0];
                __nv_bfloat162 _bf16x2_62 = __float22bfloat162_rn(make_float2(accum[122], accum[123]));
                pk[2] = reinterpret_cast<unsigned int*>(&_bf16x2_62)[0];
                __nv_bfloat162 _bf16x2_63 = __float22bfloat162_rn(make_float2(accum[126], accum[127]));
                pk[3] = reinterpret_cast<unsigned int*>(&_bf16x2_63)[0];
                unsigned int x0_126 = pk[1] & m_even | pk[0] & m_odd;
                unsigned int x1_127 = pk[3] & m_even | pk[2] & m_odd;
                unsigned int _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, x0_126, 1);
                unsigned int y0_128 = _shfl_xor_68;
                unsigned int _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, x1_127, 1);
                unsigned int y1_129 = _shfl_xor_69;
                pk[0] = y0_128 & m_odd | pk[0] & m_even;
                pk[1] = y0_128 & m_even | pk[1] & m_odd;
                pk[2] = y1_129 & m_odd | pk[2] & m_even;
                pk[3] = y1_129 & m_even | pk[3] & m_odd;
                unsigned int x2_130 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_131 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, x2_130, 2);
                unsigned int y2_132 = _shfl_xor_70;
                unsigned int _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, x3_131, 2);
                unsigned int y3_133 = _shfl_xor_71;
                pk[0] = y2_132 & m_hi | pk[0] & m_lo;
                pk[2] = y2_132 & m_lo | pk[2] & m_hi;
                pk[1] = y3_133 & m_hi | pk[1] & m_lo;
                pk[3] = y3_133 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off_117)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        unsigned int* ready_dst = reinterpret_cast<unsigned int*>(ready_ptrs[dest_c]);
                        uint32_t _atomic_inc_old_0;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_inc_old_0) : "l"(&counters[dest_c * words_per_dest + word_c]), "r"(static_cast<uint32_t>(last_ticket)) : "memory");
                        if (_atomic_inc_old_0 == last_ticket) {
                            __threadfence_system();
                            asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(ready_dst) + (pad_base_u + word_c))), "r"(static_cast<unsigned int>(ready_target)) : "memory");
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
