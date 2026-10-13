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
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 16384
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_TOTAL 131136
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
kernel_cake_qkv_norm_rope_exchange_overlap_849dd94cf08ca290ee35(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ cos_sin, __nv_bfloat16* __restrict__ q_scale, __nv_bfloat16* __restrict__ k_scale, long long* __restrict__ tiles, long long* __restrict__ dst_ptrs, long long* __restrict__ ready_ptrs, unsigned int* __restrict__ counters, int num_tiles, int M, int K, int S_local, int head_stride, long long batch_stride, long long slot_elem_offset, int pad_base, int num_words, int rank, int world, unsigned int ready_target)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 131072;
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
    // Mbarriers at smem_raw[131072..131136)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // Stage-major initialization across barrier declarations.
            mbarrier_init(smem + 131072, 1);
            mbarrier_init(smem + 131104, 256);
            mbarrier_init(smem + 131080, 1);
            mbarrier_init(smem + 131112, 256);
            mbarrier_init(smem + 131088, 1);
            mbarrier_init(smem + 131120, 256);
            mbarrier_init(smem + 131096, 1);
            mbarrier_init(smem + 131128, 256);
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
                                const uint32_t _tma_operand_pack_0_dst = (uint32_t)(smem_a_addr + stage_p * 32768);
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
                                const uint32_t _tma_operand_pack_1_dst = (uint32_t)(smem_b_addr + stage_p * 32768);
                                const int32_t _tma_operand_pack_1_coord_0 = (int32_t)(k_p * 64);
                                const int32_t _tma_operand_pack_1_coord_1 = (int32_t)(n0_p);
                                const uint64_t _tma_operand_pack_1_cache = (uint64_t)(0x1000000000000000ULL);
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%3, %4}], [%2], %5;"
                                    :: "r"(_tma_operand_pack_1_dst), "l"(_tma_operand_pack_1_desc), "r"(_tma_operand_pack_1_mbar),
                                       "r"(_tma_operand_pack_1_coord_0), "r"(_tma_operand_pack_1_coord_1), "l"(_tma_operand_pack_1_cache) : "memory");
                            }
                            mbarrier_arrive_expect_tx(full_addr + (stage_p) * 8, 32768);
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
            float accum[64] = {0};
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
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint32_t _wgmma_stage_base_0 = (unsigned int)smem + (unsigned int)stage_c * 32768;
                    {
                        uint64_t _wgmma_a_0_0 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(0) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_1 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(0) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
                            : "memory");
                    }
                    {
                        uint64_t _wgmma_a_0_2 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(16) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_3 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(16) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_3)
                            : "memory");
                    }
                    {
                        uint64_t _wgmma_a_0_4 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(32) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_5 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(32) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                            : "l"(_wgmma_a_0_4), "l"(_wgmma_b_0_5)
                            : "memory");
                    }
                    {
                        uint64_t _wgmma_a_0_6 = (((uint64_t)(((_wgmma_stage_base_0 + (0u + static_cast<uint32_t>(math_wg_idx * 64) * 128u + static_cast<uint32_t>(48) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_7 = (((uint64_t)(((_wgmma_stage_base_0 + (16384u + static_cast<uint32_t>(0) * 128u + static_cast<uint32_t>(48) * 2u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
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
                unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, x0, 1);
                unsigned int y0 = _shfl_xor_4;
                unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, x1, 1);
                unsigned int y1 = _shfl_xor_5;
                pk[0] = y0 & m_odd | pk[0] & m_even;
                pk[1] = y0 & m_even | pk[1] & m_odd;
                pk[2] = y1 & m_odd | pk[2] & m_even;
                pk[3] = y1 & m_even | pk[3] & m_odd;
                unsigned int x2 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, x2, 2);
                unsigned int y2 = _shfl_xor_6;
                unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, x3, 2);
                unsigned int y3 = _shfl_xor_7;
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
                unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, x0_0, 1);
                unsigned int y0_2 = _shfl_xor_8;
                unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, x1_1, 1);
                unsigned int y1_3 = _shfl_xor_9;
                pk[0] = y0_2 & m_odd | pk[0] & m_even;
                pk[1] = y0_2 & m_even | pk[1] & m_odd;
                pk[2] = y1_3 & m_odd | pk[2] & m_even;
                pk[3] = y1_3 & m_even | pk[3] & m_odd;
                unsigned int x2_4 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_5 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, x2_4, 2);
                unsigned int y2_6 = _shfl_xor_10;
                unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, x3_5, 2);
                unsigned int y3_7 = _shfl_xor_11;
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
                unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, x0_10, 1);
                unsigned int y0_12 = _shfl_xor_12;
                unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, x1_11, 1);
                unsigned int y1_13 = _shfl_xor_13;
                pk[0] = y0_12 & m_odd | pk[0] & m_even;
                pk[1] = y0_12 & m_even | pk[1] & m_odd;
                pk[2] = y1_13 & m_odd | pk[2] & m_even;
                pk[3] = y1_13 & m_even | pk[3] & m_odd;
                unsigned int x2_14 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_15 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, x2_14, 2);
                unsigned int y2_16 = _shfl_xor_14;
                unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, x3_15, 2);
                unsigned int y3_17 = _shfl_xor_15;
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
                unsigned int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, x0_18, 1);
                unsigned int y0_20 = _shfl_xor_16;
                unsigned int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, x1_19, 1);
                unsigned int y1_21 = _shfl_xor_17;
                pk[0] = y0_20 & m_odd | pk[0] & m_even;
                pk[1] = y0_20 & m_even | pk[1] & m_odd;
                pk[2] = y1_21 & m_odd | pk[2] & m_even;
                pk[3] = y1_21 & m_even | pk[3] & m_odd;
                unsigned int x2_22 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_23 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, x2_22, 2);
                unsigned int y2_24 = _shfl_xor_18;
                unsigned int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, x3_23, 2);
                unsigned int y3_25 = _shfl_xor_19;
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
                unsigned int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, x0_28, 1);
                unsigned int y0_30 = _shfl_xor_20;
                unsigned int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, x1_29, 1);
                unsigned int y1_31 = _shfl_xor_21;
                pk[0] = y0_30 & m_odd | pk[0] & m_even;
                pk[1] = y0_30 & m_even | pk[1] & m_odd;
                pk[2] = y1_31 & m_odd | pk[2] & m_even;
                pk[3] = y1_31 & m_even | pk[3] & m_odd;
                unsigned int x2_32 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_33 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, x2_32, 2);
                unsigned int y2_34 = _shfl_xor_22;
                unsigned int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, x3_33, 2);
                unsigned int y3_35 = _shfl_xor_23;
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
                unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, x0_36, 1);
                unsigned int y0_38 = _shfl_xor_24;
                unsigned int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, x1_37, 1);
                unsigned int y1_39 = _shfl_xor_25;
                pk[0] = y0_38 & m_odd | pk[0] & m_even;
                pk[1] = y0_38 & m_even | pk[1] & m_odd;
                pk[2] = y1_39 & m_odd | pk[2] & m_even;
                pk[3] = y1_39 & m_even | pk[3] & m_odd;
                unsigned int x2_40 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_41 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, x2_40, 2);
                unsigned int y2_42 = _shfl_xor_26;
                unsigned int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, x3_41, 2);
                unsigned int y3_43 = _shfl_xor_27;
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
                unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, x0_46, 1);
                unsigned int y0_48 = _shfl_xor_28;
                unsigned int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, x1_47, 1);
                unsigned int y1_49 = _shfl_xor_29;
                pk[0] = y0_48 & m_odd | pk[0] & m_even;
                pk[1] = y0_48 & m_even | pk[1] & m_odd;
                pk[2] = y1_49 & m_odd | pk[2] & m_even;
                pk[3] = y1_49 & m_even | pk[3] & m_odd;
                unsigned int x2_50 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_51 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, x2_50, 2);
                unsigned int y2_52 = _shfl_xor_30;
                unsigned int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, x3_51, 2);
                unsigned int y3_53 = _shfl_xor_31;
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
                unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, x0_54, 1);
                unsigned int y0_56 = _shfl_xor_32;
                unsigned int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, x1_55, 1);
                unsigned int y1_57 = _shfl_xor_33;
                pk[0] = y0_56 & m_odd | pk[0] & m_even;
                pk[1] = y0_56 & m_even | pk[1] & m_odd;
                pk[2] = y1_57 & m_odd | pk[2] & m_even;
                pk[3] = y1_57 & m_even | pk[3] & m_odd;
                unsigned int x2_58 = pk[2] & m_lo | pk[0] & m_hi;
                unsigned int x3_59 = pk[3] & m_lo | pk[1] & m_hi;
                unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, x2_58, 2);
                unsigned int y2_60 = _shfl_xor_34;
                unsigned int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, x3_59, 2);
                unsigned int y3_61 = _shfl_xor_35;
                pk[0] = y2_60 & m_hi | pk[0] & m_lo;
                pk[2] = y2_60 & m_lo | pk[2] & m_hi;
                pk[1] = y3_61 & m_hi | pk[1] & m_lo;
                pk[3] = y3_61 & m_lo | pk[3] & m_hi;
                if (local_hi < rows_m) {
                    asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + (row_off_hi + col_off_45)), "r"((pk)[0]), "r"((pk)[1]), "r"((pk)[2]), "r"((pk)[3]) : "memory");
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
