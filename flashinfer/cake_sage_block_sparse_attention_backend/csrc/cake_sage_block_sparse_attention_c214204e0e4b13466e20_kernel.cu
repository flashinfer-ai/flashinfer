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

// The exporter's render of each of the twenty SM120 specializations of this kernel, folded into one
// text over five specialization macros.  Each library compiles this file with
// -DCAKE_SAGE_SM120_KERNEL=<its kernel symbol> and the flag macros CAKE_SAGE_HAS_BLOCK_NUMS,
// CAKE_SAGE_BLOCK_SIZES_MODE (0-3), CAKE_SAGE_FULL_K64_TILES, CAKE_SAGE_UNIFORM_NONEMPTY and
// CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES; each #if arm holds the rendered lines of the specializations it
// names, and the #ifndef defaults below select the (0, 0, 0, 0, 0) specialization.
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
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 8192
#define SMEM_Q_SMEM_STRIDE 8192
#define SMEM_K_SMEM_OFF 9216
#define SMEM_K_SMEM_STAGE_BYTES 8192
#define SMEM_K_SMEM_STRIDE 8192
#define SMEM_V_TMA_SMEM_OFF 17408
#define SMEM_V_TMA_SMEM_STAGE_BYTES 8192
#define SMEM_V_TMA_SMEM_STRIDE 8192
#define SMEM_V_SMEM_OFF 17408
#define SMEM_V_SMEM_STAGE_BYTES 8192
#define SMEM_V_SMEM_STRIDE 8192
#define SMEM_O_SMEM_OFF 1024
#define SMEM_O_SMEM_STAGE_BYTES 16384
#define SMEM_O_SMEM_STRIDE 16384
#define SMEM_TOTAL 25600
#define THREADS 128
#ifndef CAKE_SAGE_HAS_BLOCK_NUMS
#define CAKE_SAGE_HAS_BLOCK_NUMS 0
#endif
#ifndef CAKE_SAGE_BLOCK_SIZES_MODE
#define CAKE_SAGE_BLOCK_SIZES_MODE 0
#endif
#ifndef CAKE_SAGE_FULL_K64_TILES
#define CAKE_SAGE_FULL_K64_TILES 0
#endif
#ifndef CAKE_SAGE_UNIFORM_NONEMPTY
#define CAKE_SAGE_UNIFORM_NONEMPTY 0
#endif
#ifndef CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
#define CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES 0
#endif
#ifndef CAKE_SAGE_SM120_KERNEL
#define CAKE_SAGE_SM120_KERNEL kernel_cake_sage_block_sparse_attention_c214204e0e4b13466e20
#endif
#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>
#include <cuda_awbarrier_primitives.h>

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


__device__ __forceinline__ void mbarrier_init_owner_lane(
    void* mbar_ptr, uint32_t count) {
    __mbarrier_init(reinterpret_cast<__mbarrier_t*>(mbar_ptr), count);
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



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)







__device__ __forceinline__ void tma_store_5d(
    const void *tmap, int x, int y, int z, int w, int v, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4, %5}], [%6];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
CAKE_SAGE_SM120_KERNEL(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap V_map, const __grid_constant__ CUtensorMap O_map, float* __restrict__ q_scale, float* __restrict__ k_scale, float* __restrict__ v_scale, int* __restrict__ q2k_block_index, int* __restrict__ q2k_block_nums, int* __restrict__ block_sizes, int seqlen_q, int seqlen_k, int num_heads, int q2k_capacity, int num_k_blocks, int block_sparse_num, float softmax_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define k_full_addr (mbar_base + 8)
    #define v_full_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int8_t* q_smem = reinterpret_cast<int8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    int8_t* k_smem = reinterpret_cast<int8_t*>(smem_raw + 9216);
    const int k_smem_addr = smem + 9216;
    uint8_t* v_tma_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int v_tma_smem_addr = smem + 17408;
    uint8_t* v_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
    __nv_bfloat16* o_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int o_smem_addr = smem + 1024;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (tid == 0) {
        // q_full: 1 barriers, init_count=1
        mbarrier_init_owner_lane(smem_raw + 0, 1);
        // k_full: 1 barriers, init_count=1
        mbarrier_init_owner_lane(smem_raw + 8, 1);
        // v_full: 1 barriers, init_count=1
        mbarrier_init_owner_lane(smem_raw + 16, 1);
    }

    // CUTLASS owner-lane publication sequence
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

    __syncthreads();

    // === Task calls (declaration order) ===
    int q_block = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int q_base = q_block * 64;
    int metadata_row = (batch * num_heads + head) * ((seqlen_q + 64 - 1) / 64) + q_block;
    int selected_count = block_sparse_num;
#if CAKE_SAGE_HAS_BLOCK_NUMS
    {
        selected_count = q2k_block_nums[metadata_row];
    }
#endif  // CAKE_SAGE_HAS_BLOCK_NUMS
    int index_base = metadata_row * q2k_capacity;
    if (warp == 0 && lane == 0 && q_block == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q_map))) : "memory"); }
    if (warp == 0 && lane == 0 && q_block == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K_map))) : "memory"); }
    if (warp == 0 && lane == 0 && q_block == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V_map))) : "memory"); }
    if (warp == 0 && lane == 0 && q_block == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&O_map))) : "memory"); }
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_full_addr, 8192);
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                :: "r"(q_smem_addr), "l"((&Q_map)), "r"(0), "r"(q_base), "r"(head), "r"(batch),
                   "r"(q_full_addr), "l"(0x14F0000000000000ULL) : "memory");
        }
    }
    int first_physical = 0;
#if CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    {
        {
            if (lane == 0) {
                first_physical = q2k_block_index[index_base + selected_count - 1];
            }
            int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, first_physical, 0);
            first_physical = _shfl_sync_0;
        }
#elif CAKE_SAGE_UNIFORM_NONEMPTY
    {
        {
            first_physical = q2k_block_index[index_base + selected_count - 1];
        }
#else
    if (selected_count > 0) {
        first_physical = q2k_block_index[index_base + selected_count - 1];
#endif  // CAKE_SAGE_UNIFORM_NONEMPTY, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr, 8192);
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr), "l"((&K_map)), "r"(0), "r"(first_physical * 64), "r"(head), "r"(batch),
                       "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                mbarrier_arrive_expect_tx(v_full_addr, 8192);
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_tma_smem_addr), "l"((&V_map)), "r"(first_physical * 64), "r"(0), "r"(head), "r"(batch),
                       "r"(v_full_addr), "l"(0x14F0000000000000ULL) : "memory");
            }
        }
    }
    unsigned int q_frag[4];
    unsigned int q_prefetch[4];
    unsigned int k_frag[4];
    unsigned int v_frag[4];
    int scores_i32[32];
    float scores[32];
    unsigned int probabilities[8];
    float output[64];
    float row_max[2];
    float row_shift[2];
    float row_sum[2];
    float row_scale[2];
    float local_sum[2];
    float pack_values[4];
    float p_half0[16];
    for (int row = 0; row < 2; row++) {
        row_max[row] = -CUDART_INF_F;
        row_shift[row] = 0.0f;
        row_sum[row] = 0.0f;
    }
    for (int feature = 0; feature < 8; feature++) {
        for (int elem = 0; elem < 8; elem++) {
            output[feature * 8 + elem] = 0.0f;
        }
    }
    int q_scale_index = q_base / 128 * 4 + (q_base % 128 + warp * 16) / 32;
    int q_scale_base = (batch * num_heads + head) * ((seqlen_q + 127) / 128 * 4);
    float q_scale_lane = 0.0f;
    if (lane == 0) {
        q_scale_lane = q_scale[q_scale_base + q_scale_index];
    }
    float q_scale_value = __shfl_sync(0xFFFFFFFF, q_scale_lane, 0);
    float q_softmax_log2 = q_scale_value * softmax_scale * 1.442695f;
    int k_scale_base = (batch * num_heads + head) * num_k_blocks;
    int v_scale_base = (batch * num_heads + head) * 128;
    unsigned int _phase_q_full_0 = 0;
    mbarrier_wait(q_full_addr, _phase_q_full_0);
    _phase_q_full_0 ^= 1;
    int current_physical = first_physical;
    int _min_0 = ((selected_count) < (1) ? (selected_count) : (1));
    unsigned int _phase_k_full_0 = 0;
    unsigned int _phase_v_full_0 = 0;
    #pragma unroll 1
    for (int load_count = 0; load_count < _min_0; load_count++) {
        mbarrier_wait(k_full_addr, _phase_k_full_0);
        _phase_k_full_0 ^= 1;
        unsigned int q_row_base = warp * 16 + lane % 16;
        unsigned int q_col_base = lane / 16;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
            : "r"(q_smem_addr + (q_row_base * 8 + (unsigned int)((q_col_base * 16 ^ (q_row_base & 7) << 4) / 16)) * 16)
            : "memory");
        unsigned int q_row_base_0 = warp * 16 + lane % 16;
        unsigned int q_col_base_1 = lane / 16;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_prefetch[0]), "=r"(q_prefetch[1]), "=r"(q_prefetch[2]), "=r"(q_prefetch[3])
            : "r"(q_smem_addr + (q_row_base_0 * 8 + (unsigned int)(((q_col_base_1 + 2) * 16 ^ (q_row_base_0 & 7) << 4) / 16)) * 16)
            : "memory");
        unsigned int k_row_base = lane % 8 + lane / 16 * 8;
        unsigned int k_col_base = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + (k_row_base * 8 + (unsigned int)((k_col_base * 16 ^ (k_row_base & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[0]), "=r"(scores_i32[1]), "=r"(scores_i32[2]), "=r"(scores_i32[3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[4]), "=r"(scores_i32[(4) + 1]), "=r"(scores_i32[(4) + 2]), "=r"(scores_i32[(4) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base + 16) * 8 + (unsigned int)((k_col_base * 16 ^ (k_row_base + 16 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[8]), "=r"(scores_i32[(8) + 1]), "=r"(scores_i32[(8) + 2]), "=r"(scores_i32[(8) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[12]), "=r"(scores_i32[(12) + 1]), "=r"(scores_i32[(12) + 2]), "=r"(scores_i32[(12) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base + 32) * 8 + (unsigned int)((k_col_base * 16 ^ (k_row_base + 32 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[16]), "=r"(scores_i32[(16) + 1]), "=r"(scores_i32[(16) + 2]), "=r"(scores_i32[(16) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[20]), "=r"(scores_i32[(20) + 1]), "=r"(scores_i32[(20) + 2]), "=r"(scores_i32[(20) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base + 48) * 8 + (unsigned int)((k_col_base * 16 ^ (k_row_base + 48 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[24]), "=r"(scores_i32[(24) + 1]), "=r"(scores_i32[(24) + 2]), "=r"(scores_i32[(24) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[28]), "=r"(scores_i32[(28) + 1]), "=r"(scores_i32[(28) + 2]), "=r"(scores_i32[(28) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        unsigned int q_row_base_2 = warp * 16 + lane % 16;
        unsigned int q_col_base_3 = lane / 16;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
            : "r"(q_smem_addr + (q_row_base_2 * 8 + (unsigned int)(((q_col_base_3 + 4) * 16 ^ (q_row_base_2 & 7) << 4) / 16)) * 16)
            : "memory");
        unsigned int k_row_base_4 = lane % 8 + lane / 16 * 8;
        unsigned int k_col_base_5 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + (k_row_base_4 * 8 + (unsigned int)(((k_col_base_5 + 2) * 16 ^ (k_row_base_4 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]), "+r"(scores_i32[(4) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_4 + 16) * 8 + (unsigned int)(((k_col_base_5 + 2) * 16 ^ (k_row_base_4 + 16 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]), "+r"(scores_i32[(8) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]), "+r"(scores_i32[(12) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_4 + 32) * 8 + (unsigned int)(((k_col_base_5 + 2) * 16 ^ (k_row_base_4 + 32 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]), "+r"(scores_i32[(16) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]), "+r"(scores_i32[(20) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_4 + 48) * 8 + (unsigned int)(((k_col_base_5 + 2) * 16 ^ (k_row_base_4 + 48 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]), "+r"(scores_i32[(24) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]), "+r"(scores_i32[(28) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        unsigned int q_row_base_6 = warp * 16 + lane % 16;
        unsigned int q_col_base_7 = lane / 16;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_prefetch[0]), "=r"(q_prefetch[1]), "=r"(q_prefetch[2]), "=r"(q_prefetch[3])
            : "r"(q_smem_addr + (q_row_base_6 * 8 + (unsigned int)(((q_col_base_7 + 6) * 16 ^ (q_row_base_6 & 7) << 4) / 16)) * 16)
            : "memory");
        unsigned int k_row_base_8 = lane % 8 + lane / 16 * 8;
        unsigned int k_col_base_9 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + (k_row_base_8 * 8 + (unsigned int)(((k_col_base_9 + 4) * 16 ^ (k_row_base_8 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]), "+r"(scores_i32[(4) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_8 + 16) * 8 + (unsigned int)(((k_col_base_9 + 4) * 16 ^ (k_row_base_8 + 16 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]), "+r"(scores_i32[(8) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]), "+r"(scores_i32[(12) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_8 + 32) * 8 + (unsigned int)(((k_col_base_9 + 4) * 16 ^ (k_row_base_8 + 32 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]), "+r"(scores_i32[(16) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]), "+r"(scores_i32[(20) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_8 + 48) * 8 + (unsigned int)(((k_col_base_9 + 4) * 16 ^ (k_row_base_8 + 48 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]), "+r"(scores_i32[(24) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]), "+r"(scores_i32[(28) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        unsigned int k_row_base_10 = lane % 8 + lane / 16 * 8;
        unsigned int k_col_base_11 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + (k_row_base_10 * 8 + (unsigned int)(((k_col_base_11 + 6) * 16 ^ (k_row_base_10 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]), "+r"(scores_i32[(4) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_10 + 16) * 8 + (unsigned int)(((k_col_base_11 + 6) * 16 ^ (k_row_base_10 + 16 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]), "+r"(scores_i32[(8) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]), "+r"(scores_i32[(12) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_10 + 32) * 8 + (unsigned int)(((k_col_base_11 + 6) * 16 ^ (k_row_base_10 + 32 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]), "+r"(scores_i32[(16) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]), "+r"(scores_i32[(20) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_10 + 48) * 8 + (unsigned int)(((k_col_base_11 + 6) * 16 ^ (k_row_base_10 + 48 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]), "+r"(scores_i32[(24) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]), "+r"(scores_i32[(28) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        float k_scale_lane = 0.0f;
        if (lane == 0) {
            k_scale_lane = k_scale[k_scale_base + current_physical];
        }
        float k_scale_value = __shfl_sync(0xFFFFFFFF, k_scale_lane, 0);
        float score_scale_log2 = q_softmax_log2 * k_scale_value;
        int valid_tokens = 64;
#if CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_FULL_K64_TILES && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens = seqlen_k - current_physical * 64;
            if (valid_tokens > 64) {
                valid_tokens = 64;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
            {
                int _min_1 = ((valid_tokens) < (block_sizes[(batch * num_heads + head) * num_k_blocks + current_physical]) ? (valid_tokens) : (block_sizes[(batch * num_heads + head) * num_k_blocks + current_physical]));
                valid_tokens = _min_1;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_FULL_K64_TILES && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens = seqlen_k - current_physical * 64;
            if (valid_tokens > 64) {
                valid_tokens = 64;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
            {
                int _min_0_1 = ((valid_tokens) < (block_sizes[(batch * num_heads + head) * num_k_blocks + current_physical]) ? (valid_tokens) : (block_sizes[(batch * num_heads + head) * num_k_blocks + current_physical]));
                valid_tokens = _min_0_1;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_FULL_K64_TILES && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens = seqlen_k - current_physical * 64;
            if (valid_tokens > 64) {
                valid_tokens = 64;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
            {
                int _min_1 = ((valid_tokens) < (block_sizes[batch * num_k_blocks + current_physical]) ? (valid_tokens) : (block_sizes[batch * num_k_blocks + current_physical]));
                valid_tokens = _min_1;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_FULL_K64_TILES && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens = seqlen_k - current_physical * 64;
            if (valid_tokens > 64) {
                valid_tokens = 64;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
            {
                int _min_0_1 = ((valid_tokens) < (block_sizes[batch * num_k_blocks + current_physical]) ? (valid_tokens) : (block_sizes[batch * num_k_blocks + current_physical]));
                valid_tokens = _min_0_1;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_FULL_K64_TILES && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens = seqlen_k - current_physical * 64;
            if (valid_tokens > 64) {
                valid_tokens = 64;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
            {
                int _min_1 = ((valid_tokens) < (block_sizes[current_physical]) ? (valid_tokens) : (block_sizes[current_physical]));
                valid_tokens = _min_1;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_FULL_K64_TILES && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens = seqlen_k - current_physical * 64;
            if (valid_tokens > 64) {
                valid_tokens = 64;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
            {
                int _min_0_1 = ((valid_tokens) < (block_sizes[current_physical]) ? (valid_tokens) : (block_sizes[current_physical]));
                valid_tokens = _min_0_1;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
        }
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && !CAKE_SAGE_FULL_K64_TILES && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && !CAKE_SAGE_FULL_K64_TILES && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        {
            valid_tokens = seqlen_k - current_physical * 64;
            if (valid_tokens > 64) {
                valid_tokens = 64;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
            if (valid_tokens < 0) {
                valid_tokens = 0;
            }
        }
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_FULL_K64_TILES, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        int preload_count = load_count + 1;
        int next_physical = 0;
        if (preload_count < selected_count) {
            {
#if CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                next_physical = current_physical - 1;
                if (next_physical < 0) {
                    next_physical = next_physical + num_k_blocks;
                }
#else
                next_physical = q2k_block_index[index_base + selected_count - 1 - preload_count];
#endif  // CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
            }
            __syncthreads();
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr, 8192);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(k_smem_addr), "l"((&K_map)), "r"(0), "r"(next_physical * 64), "r"(head), "r"(batch),
                           "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        scores[0] = (float)scores_i32[0];
        scores[1] = (float)scores_i32[1];
        scores[2] = (float)scores_i32[2];
        scores[3] = (float)scores_i32[3];
        scores[4] = (float)scores_i32[4];
        scores[5] = (float)scores_i32[5];
        scores[6] = (float)scores_i32[6];
        scores[7] = (float)scores_i32[7];
        scores[8] = (float)scores_i32[8];
        scores[9] = (float)scores_i32[9];
        scores[10] = (float)scores_i32[10];
        scores[11] = (float)scores_i32[11];
        scores[12] = (float)scores_i32[12];
        scores[13] = (float)scores_i32[13];
        scores[14] = (float)scores_i32[14];
        scores[15] = (float)scores_i32[15];
        scores[16] = (float)scores_i32[16];
        scores[17] = (float)scores_i32[17];
        scores[18] = (float)scores_i32[18];
        scores[19] = (float)scores_i32[19];
        scores[20] = (float)scores_i32[20];
        scores[21] = (float)scores_i32[21];
        scores[22] = (float)scores_i32[22];
        scores[23] = (float)scores_i32[23];
        scores[24] = (float)scores_i32[24];
        scores[25] = (float)scores_i32[25];
        scores[26] = (float)scores_i32[26];
        scores[27] = (float)scores_i32[27];
        scores[28] = (float)scores_i32[28];
        scores[29] = (float)scores_i32[29];
        scores[30] = (float)scores_i32[30];
        scores[31] = (float)scores_i32[31];
        if (valid_tokens < 64) {
            int lane_col_base = 2 * (lane % 4);
            int token_in_tile = lane_col_base;
            if (token_in_tile >= valid_tokens) {
                scores[0] = -CUDART_INF_F;
            }
            int token_in_tile_0 = lane_col_base + 1;
            if (token_in_tile_0 >= valid_tokens) {
                scores[1] = -CUDART_INF_F;
            }
            int token_in_tile_1 = lane_col_base;
            if (token_in_tile_1 >= valid_tokens) {
                scores[2] = -CUDART_INF_F;
            }
            int token_in_tile_2 = lane_col_base + 1;
            if (token_in_tile_2 >= valid_tokens) {
                scores[3] = -CUDART_INF_F;
            }
            int token_in_tile_3 = lane_col_base + 8;
            if (token_in_tile_3 >= valid_tokens) {
                scores[4] = -CUDART_INF_F;
            }
            int token_in_tile_4 = lane_col_base + 8 + 1;
            if (token_in_tile_4 >= valid_tokens) {
                scores[5] = -CUDART_INF_F;
            }
            int token_in_tile_5 = lane_col_base + 8;
            if (token_in_tile_5 >= valid_tokens) {
                scores[6] = -CUDART_INF_F;
            }
            int token_in_tile_6 = lane_col_base + 8 + 1;
            if (token_in_tile_6 >= valid_tokens) {
                scores[7] = -CUDART_INF_F;
            }
            int token_in_tile_7 = 16 + lane_col_base;
            if (token_in_tile_7 >= valid_tokens) {
                scores[8] = -CUDART_INF_F;
            }
            int token_in_tile_8 = 16 + lane_col_base + 1;
            if (token_in_tile_8 >= valid_tokens) {
                scores[9] = -CUDART_INF_F;
            }
            int token_in_tile_9 = 16 + lane_col_base;
            if (token_in_tile_9 >= valid_tokens) {
                scores[10] = -CUDART_INF_F;
            }
            int token_in_tile_10 = 16 + lane_col_base + 1;
            if (token_in_tile_10 >= valid_tokens) {
                scores[11] = -CUDART_INF_F;
            }
            int token_in_tile_11 = 16 + lane_col_base + 8;
            if (token_in_tile_11 >= valid_tokens) {
                scores[12] = -CUDART_INF_F;
            }
            int token_in_tile_12 = 16 + lane_col_base + 8 + 1;
            if (token_in_tile_12 >= valid_tokens) {
                scores[13] = -CUDART_INF_F;
            }
            int token_in_tile_13 = 16 + lane_col_base + 8;
            if (token_in_tile_13 >= valid_tokens) {
                scores[14] = -CUDART_INF_F;
            }
            int token_in_tile_14 = 16 + lane_col_base + 8 + 1;
            if (token_in_tile_14 >= valid_tokens) {
                scores[15] = -CUDART_INF_F;
            }
            int token_in_tile_15 = 32 + lane_col_base;
            if (token_in_tile_15 >= valid_tokens) {
                scores[16] = -CUDART_INF_F;
            }
            int token_in_tile_16 = 32 + lane_col_base + 1;
            if (token_in_tile_16 >= valid_tokens) {
                scores[17] = -CUDART_INF_F;
            }
            int token_in_tile_17 = 32 + lane_col_base;
            if (token_in_tile_17 >= valid_tokens) {
                scores[18] = -CUDART_INF_F;
            }
            int token_in_tile_18 = 32 + lane_col_base + 1;
            if (token_in_tile_18 >= valid_tokens) {
                scores[19] = -CUDART_INF_F;
            }
            int token_in_tile_19 = 32 + lane_col_base + 8;
            if (token_in_tile_19 >= valid_tokens) {
                scores[20] = -CUDART_INF_F;
            }
            int token_in_tile_20 = 32 + lane_col_base + 8 + 1;
            if (token_in_tile_20 >= valid_tokens) {
                scores[21] = -CUDART_INF_F;
            }
            int token_in_tile_21 = 32 + lane_col_base + 8;
            if (token_in_tile_21 >= valid_tokens) {
                scores[22] = -CUDART_INF_F;
            }
            int token_in_tile_22 = 32 + lane_col_base + 8 + 1;
            if (token_in_tile_22 >= valid_tokens) {
                scores[23] = -CUDART_INF_F;
            }
            int token_in_tile_23 = 48 + lane_col_base;
            if (token_in_tile_23 >= valid_tokens) {
                scores[24] = -CUDART_INF_F;
            }
            int token_in_tile_24 = 48 + lane_col_base + 1;
            if (token_in_tile_24 >= valid_tokens) {
                scores[25] = -CUDART_INF_F;
            }
            int token_in_tile_25 = 48 + lane_col_base;
            if (token_in_tile_25 >= valid_tokens) {
                scores[26] = -CUDART_INF_F;
            }
            int token_in_tile_26 = 48 + lane_col_base + 1;
            if (token_in_tile_26 >= valid_tokens) {
                scores[27] = -CUDART_INF_F;
            }
            int token_in_tile_27 = 48 + lane_col_base + 8;
            if (token_in_tile_27 >= valid_tokens) {
                scores[28] = -CUDART_INF_F;
            }
            int token_in_tile_28 = 48 + lane_col_base + 8 + 1;
            if (token_in_tile_28 >= valid_tokens) {
                scores[29] = -CUDART_INF_F;
            }
            int token_in_tile_29 = 48 + lane_col_base + 8;
            if (token_in_tile_29 >= valid_tokens) {
                scores[30] = -CUDART_INF_F;
            }
            int token_in_tile_30 = 48 + lane_col_base + 8 + 1;
            if (token_in_tile_30 >= valid_tokens) {
                scores[31] = -CUDART_INF_F;
            }
        }
        float carry_current[2];
        float carry_safe[2];
        float carry_delta[2];
        float previous_max = row_max[0];
        float group_max = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[4], scores[5]));
        group_max = fmaxf(group_max, fmaxf(fmaxf(scores[8], scores[9]), fmaxf(scores[12], scores[13])));
        group_max = fmaxf(group_max, fmaxf(fmaxf(scores[16], scores[17]), fmaxf(scores[20], scores[21])));
        group_max = fmaxf(group_max, fmaxf(fmaxf(scores[24], scores[25]), fmaxf(scores[28], scores[29])));
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
        group_max = fmaxf(group_max, _shfl_xor_sync_2);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
        group_max = fmaxf(group_max, _shfl_xor_sync_3);
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
        group_max = fmaxf(group_max, _shfl_xor_sync_1);
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
        group_max = fmaxf(group_max, _shfl_xor_sync_2);
#else
        float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, group_max, 1);
        group_max = fmaxf(group_max, _shfl_xor_sync_0);
        float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, group_max, 2);
        group_max = fmaxf(group_max, _shfl_xor_sync_1);
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float current_max = fmaxf(previous_max, group_max * score_scale_log2);
#if CAKE_SAGE_FULL_K64_TILES
        float safe_max = current_max;
#else
        float safe_max = ((current_max == -CUDART_INF_F) ? 0.0f : current_max);
#endif  // CAKE_SAGE_FULL_K64_TILES
        carry_current[0] = current_max;
        carry_safe[0] = safe_max;
        carry_delta[0] = previous_max - safe_max;
        row_scale[0] = 1.0f;
        float previous_max_12 = row_max[1];
        float group_max_13 = fmaxf(fmaxf(scores[2], scores[3]), fmaxf(scores[6], scores[7]));
        group_max_13 = fmaxf(group_max_13, fmaxf(fmaxf(scores[10], scores[11]), fmaxf(scores[14], scores[15])));
        group_max_13 = fmaxf(group_max_13, fmaxf(fmaxf(scores[18], scores[19]), fmaxf(scores[22], scores[23])));
        group_max_13 = fmaxf(group_max_13, fmaxf(fmaxf(scores[26], scores[27]), fmaxf(scores[30], scores[31])));
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, group_max_13, 1);
        group_max_13 = fmaxf(group_max_13, _shfl_xor_sync_4);
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, group_max_13, 2);
        group_max_13 = fmaxf(group_max_13, _shfl_xor_sync_5);
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, group_max_13, 1);
        group_max_13 = fmaxf(group_max_13, _shfl_xor_sync_3);
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, group_max_13, 2);
        group_max_13 = fmaxf(group_max_13, _shfl_xor_sync_4);
#else
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, group_max_13, 1);
        group_max_13 = fmaxf(group_max_13, _shfl_xor_sync_2);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, group_max_13, 2);
        group_max_13 = fmaxf(group_max_13, _shfl_xor_sync_3);
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float current_max_14 = fmaxf(previous_max_12, group_max_13 * score_scale_log2);
#if CAKE_SAGE_FULL_K64_TILES
        float safe_max_15 = current_max_14;
#else
        float safe_max_15 = ((current_max_14 == -CUDART_INF_F) ? 0.0f : current_max_14);
#endif  // CAKE_SAGE_FULL_K64_TILES
        carry_current[1] = current_max_14;
        carry_safe[1] = safe_max_15;
        carry_delta[1] = previous_max_12 - safe_max_15;
        row_scale[1] = 1.0f;
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        int _warp_vote_6 = __any_sync(0xFFFFFFFF, carry_delta[0] < -0.8073549f || carry_delta[1] < -0.8073549f);
        if (_warp_vote_6 != 0) {
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        int _warp_vote_5 = __any_sync(0xFFFFFFFF, carry_delta[0] < -0.8073549f || carry_delta[1] < -0.8073549f);
        if (_warp_vote_5 != 0) {
#else
        int _warp_vote_4 = __any_sync(0xFFFFFFFF, carry_delta[0] < -0.8073549f || carry_delta[1] < -0.8073549f);
        if (_warp_vote_4 != 0) {
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
            if (carry_delta[0] < -0.8073549f) {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float _exp2_7 = approx_exp2(carry_delta[0]);
                row_scale[0] = _exp2_7;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float _exp2_6 = approx_exp2(carry_delta[0]);
                row_scale[0] = _exp2_6;
#else
                float _exp2_5 = approx_exp2(carry_delta[0]);
                row_scale[0] = _exp2_5;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                row_max[0] = carry_current[0];
                row_shift[0] = carry_safe[0] - 8.0f;
            }
            if (carry_delta[1] < -0.8073549f) {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float _exp2_8 = approx_exp2(carry_delta[1]);
                row_scale[1] = _exp2_8;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float _exp2_7 = approx_exp2(carry_delta[1]);
                row_scale[1] = _exp2_7;
#else
                float _exp2_6 = approx_exp2(carry_delta[1]);
                row_scale[1] = _exp2_6;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                row_max[1] = carry_current[1];
                row_shift[1] = carry_safe[1] - 8.0f;
            }
            {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float2 _pair_scale_even2_9 = make_float2(row_scale[0], row_scale[0]);
                float2 _pair_scale_odd2_9 = make_float2(row_scale[1], row_scale[1]);
                float2* _pair_scale_src2_9 = reinterpret_cast<float2*>(&output[0]);
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_9));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_9[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_9));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float2 _pair_scale_even2_8 = make_float2(row_scale[0], row_scale[0]);
                float2 _pair_scale_odd2_8 = make_float2(row_scale[1], row_scale[1]);
                float2* _pair_scale_src2_8 = reinterpret_cast<float2*>(&output[0]);
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_8));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_8[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_8));
#else
                float2 _pair_scale_even2_7 = make_float2(row_scale[0], row_scale[0]);
                float2 _pair_scale_odd2_7 = make_float2(row_scale[1], row_scale[1]);
                float2* _pair_scale_src2_7 = reinterpret_cast<float2*>(&output[0]);
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_7));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_7[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_7));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
            }
        }
        float first0 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2, -row_shift[0]));
        scores[0] = first0;
        local_sum[0] = first0;
        float first1 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2, -row_shift[1]));
        scores[2] = first1;
        local_sum[1] = first1;
        float value0 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2, -row_shift[0]));
        scores[1] = value0;
        local_sum[0] = local_sum[0] + value0;
        float value1 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2, -row_shift[1]));
        scores[3] = value1;
        local_sum[1] = local_sum[1] + value1;
        float value0_16 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2, -row_shift[0]));
        scores[4] = value0_16;
        local_sum[0] = local_sum[0] + value0_16;
        float value1_17 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2, -row_shift[1]));
        scores[6] = value1_17;
        local_sum[1] = local_sum[1] + value1_17;
        float value0_18 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2, -row_shift[0]));
        scores[5] = value0_18;
        local_sum[0] = local_sum[0] + value0_18;
        float value1_19 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2, -row_shift[1]));
        scores[7] = value1_19;
        local_sum[1] = local_sum[1] + value1_19;
        float value0_20 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2, -row_shift[0]));
        scores[8] = value0_20;
        local_sum[0] = local_sum[0] + value0_20;
        float value1_21 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2, -row_shift[1]));
        scores[10] = value1_21;
        local_sum[1] = local_sum[1] + value1_21;
        float value0_22 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2, -row_shift[0]));
        scores[9] = value0_22;
        local_sum[0] = local_sum[0] + value0_22;
        float value1_23 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2, -row_shift[1]));
        scores[11] = value1_23;
        local_sum[1] = local_sum[1] + value1_23;
        float value0_24 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2, -row_shift[0]));
        scores[12] = value0_24;
        local_sum[0] = local_sum[0] + value0_24;
        float value1_25 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2, -row_shift[1]));
        scores[14] = value1_25;
        local_sum[1] = local_sum[1] + value1_25;
        float value0_26 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2, -row_shift[0]));
        scores[13] = value0_26;
        local_sum[0] = local_sum[0] + value0_26;
        float value1_27 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2, -row_shift[1]));
        scores[15] = value1_27;
        local_sum[1] = local_sum[1] + value1_27;
        pack_values[0] = scores[0];
        pack_values[1] = scores[1];
        pack_values[2] = scores[4];
        pack_values[3] = scores[5];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[0] = _packed;
        }
        pack_values[0] = scores[2];
        pack_values[1] = scores[3];
        pack_values[2] = scores[6];
        pack_values[3] = scores[7];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[1] = _packed;
        }
        pack_values[0] = scores[8];
        pack_values[1] = scores[9];
        pack_values[2] = scores[12];
        pack_values[3] = scores[13];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[2] = _packed;
        }
        pack_values[0] = scores[10];
        pack_values[1] = scores[11];
        pack_values[2] = scores[14];
        pack_values[3] = scores[15];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[3] = _packed;
        }
        mbarrier_wait(v_full_addr, _phase_v_full_0);
        _phase_v_full_0 ^= 1;
        unsigned int v_row_base = lane % 8 + lane / 16 * 8;
        unsigned int v_col_base = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + (v_row_base * 4 + (unsigned int)((v_col_base * 16 ^ (v_row_base >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[0]), "+f"(output[1]), "+f"(output[2]), "+f"(output[3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[4]), "+f"(output[(4) + 1]), "+f"(output[(4) + 2]), "+f"(output[(4) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base + 16) * 4 + (unsigned int)((v_col_base * 16 ^ (v_row_base + 16 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[8]), "+f"(output[(8) + 1]), "+f"(output[(8) + 2]), "+f"(output[(8) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[12]), "+f"(output[(12) + 1]), "+f"(output[(12) + 2]), "+f"(output[(12) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base + 32) * 4 + (unsigned int)((v_col_base * 16 ^ (v_row_base + 32 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[16]), "+f"(output[(16) + 1]), "+f"(output[(16) + 2]), "+f"(output[(16) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[20]), "+f"(output[(20) + 1]), "+f"(output[(20) + 2]), "+f"(output[(20) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base + 48) * 4 + (unsigned int)((v_col_base * 16 ^ (v_row_base + 48 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[24]), "+f"(output[(24) + 1]), "+f"(output[(24) + 2]), "+f"(output[(24) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[28]), "+f"(output[(28) + 1]), "+f"(output[(28) + 2]), "+f"(output[(28) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base + 64) * 4 + (unsigned int)((v_col_base * 16 ^ (v_row_base + 64 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[32]), "+f"(output[(32) + 1]), "+f"(output[(32) + 2]), "+f"(output[(32) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[36]), "+f"(output[(36) + 1]), "+f"(output[(36) + 2]), "+f"(output[(36) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base + 80) * 4 + (unsigned int)((v_col_base * 16 ^ (v_row_base + 80 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[40]), "+f"(output[(40) + 1]), "+f"(output[(40) + 2]), "+f"(output[(40) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[44]), "+f"(output[(44) + 1]), "+f"(output[(44) + 2]), "+f"(output[(44) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base + 96) * 4 + (unsigned int)((v_col_base * 16 ^ (v_row_base + 96 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[48]), "+f"(output[(48) + 1]), "+f"(output[(48) + 2]), "+f"(output[(48) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[52]), "+f"(output[(52) + 1]), "+f"(output[(52) + 2]), "+f"(output[(52) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base + 112) * 4 + (unsigned int)((v_col_base * 16 ^ (v_row_base + 112 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[56]), "+f"(output[(56) + 1]), "+f"(output[(56) + 2]), "+f"(output[(56) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[60]), "+f"(output[(60) + 1]), "+f"(output[(60) + 2]), "+f"(output[(60) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        float value0_28 = approx_exp2(__fmaf_rn(scores[16], score_scale_log2, -row_shift[0]));
        scores[16] = value0_28;
        local_sum[0] = local_sum[0] + value0_28;
        float value1_29 = approx_exp2(__fmaf_rn(scores[18], score_scale_log2, -row_shift[1]));
        scores[18] = value1_29;
        local_sum[1] = local_sum[1] + value1_29;
        float value0_30 = approx_exp2(__fmaf_rn(scores[17], score_scale_log2, -row_shift[0]));
        scores[17] = value0_30;
        local_sum[0] = local_sum[0] + value0_30;
        float value1_31 = approx_exp2(__fmaf_rn(scores[19], score_scale_log2, -row_shift[1]));
        scores[19] = value1_31;
        local_sum[1] = local_sum[1] + value1_31;
        float value0_32 = approx_exp2(__fmaf_rn(scores[20], score_scale_log2, -row_shift[0]));
        scores[20] = value0_32;
        local_sum[0] = local_sum[0] + value0_32;
        float value1_33 = approx_exp2(__fmaf_rn(scores[22], score_scale_log2, -row_shift[1]));
        scores[22] = value1_33;
        local_sum[1] = local_sum[1] + value1_33;
        float value0_34 = approx_exp2(__fmaf_rn(scores[21], score_scale_log2, -row_shift[0]));
        scores[21] = value0_34;
        local_sum[0] = local_sum[0] + value0_34;
        float value1_35 = approx_exp2(__fmaf_rn(scores[23], score_scale_log2, -row_shift[1]));
        scores[23] = value1_35;
        local_sum[1] = local_sum[1] + value1_35;
        float value0_36 = approx_exp2(__fmaf_rn(scores[24], score_scale_log2, -row_shift[0]));
        scores[24] = value0_36;
        local_sum[0] = local_sum[0] + value0_36;
        float value1_37 = approx_exp2(__fmaf_rn(scores[26], score_scale_log2, -row_shift[1]));
        scores[26] = value1_37;
        local_sum[1] = local_sum[1] + value1_37;
        float value0_38 = approx_exp2(__fmaf_rn(scores[25], score_scale_log2, -row_shift[0]));
        scores[25] = value0_38;
        local_sum[0] = local_sum[0] + value0_38;
        float value1_39 = approx_exp2(__fmaf_rn(scores[27], score_scale_log2, -row_shift[1]));
        scores[27] = value1_39;
        local_sum[1] = local_sum[1] + value1_39;
        float value0_40 = approx_exp2(__fmaf_rn(scores[28], score_scale_log2, -row_shift[0]));
        scores[28] = value0_40;
        local_sum[0] = local_sum[0] + value0_40;
        float value1_41 = approx_exp2(__fmaf_rn(scores[30], score_scale_log2, -row_shift[1]));
        scores[30] = value1_41;
        local_sum[1] = local_sum[1] + value1_41;
        float value0_42 = approx_exp2(__fmaf_rn(scores[29], score_scale_log2, -row_shift[0]));
        scores[29] = value0_42;
        local_sum[0] = local_sum[0] + value0_42;
        float value1_43 = approx_exp2(__fmaf_rn(scores[31], score_scale_log2, -row_shift[1]));
        scores[31] = value1_43;
        local_sum[1] = local_sum[1] + value1_43;
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 1);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_10;
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 2);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_11;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 1);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_9;
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 2);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_10;
#else
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 1);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_8;
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 2);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_9;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        row_sum[0] = __fmaf_rn(row_sum[0], row_scale[0], local_sum[0]);
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 1);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_12;
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 2);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_13;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 1);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_11;
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 2);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_12;
#else
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 1);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_10;
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 2);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_11;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        row_sum[1] = __fmaf_rn(row_sum[1], row_scale[1], local_sum[1]);
        pack_values[0] = scores[16];
        pack_values[1] = scores[17];
        pack_values[2] = scores[20];
        pack_values[3] = scores[21];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[4] = _packed;
        }
        pack_values[0] = scores[18];
        pack_values[1] = scores[19];
        pack_values[2] = scores[22];
        pack_values[3] = scores[23];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[5] = _packed;
        }
        pack_values[0] = scores[24];
        pack_values[1] = scores[25];
        pack_values[2] = scores[28];
        pack_values[3] = scores[29];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[6] = _packed;
        }
        pack_values[0] = scores[26];
        pack_values[1] = scores[27];
        pack_values[2] = scores[30];
        pack_values[3] = scores[31];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[7] = _packed;
        }
        unsigned int v_row_base_44 = lane % 8 + lane / 16 * 8;
        unsigned int v_col_base_45 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + (v_row_base_44 * 4 + (unsigned int)(((v_col_base_45 + 2) * 16 ^ (v_row_base_44 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[0]), "+f"(output[1]), "+f"(output[2]), "+f"(output[3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[4]), "+f"(output[(4) + 1]), "+f"(output[(4) + 2]), "+f"(output[(4) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_44 + 16) * 4 + (unsigned int)(((v_col_base_45 + 2) * 16 ^ (v_row_base_44 + 16 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[8]), "+f"(output[(8) + 1]), "+f"(output[(8) + 2]), "+f"(output[(8) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[12]), "+f"(output[(12) + 1]), "+f"(output[(12) + 2]), "+f"(output[(12) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_44 + 32) * 4 + (unsigned int)(((v_col_base_45 + 2) * 16 ^ (v_row_base_44 + 32 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[16]), "+f"(output[(16) + 1]), "+f"(output[(16) + 2]), "+f"(output[(16) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[20]), "+f"(output[(20) + 1]), "+f"(output[(20) + 2]), "+f"(output[(20) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_44 + 48) * 4 + (unsigned int)(((v_col_base_45 + 2) * 16 ^ (v_row_base_44 + 48 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[24]), "+f"(output[(24) + 1]), "+f"(output[(24) + 2]), "+f"(output[(24) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[28]), "+f"(output[(28) + 1]), "+f"(output[(28) + 2]), "+f"(output[(28) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_44 + 64) * 4 + (unsigned int)(((v_col_base_45 + 2) * 16 ^ (v_row_base_44 + 64 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[32]), "+f"(output[(32) + 1]), "+f"(output[(32) + 2]), "+f"(output[(32) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[36]), "+f"(output[(36) + 1]), "+f"(output[(36) + 2]), "+f"(output[(36) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_44 + 80) * 4 + (unsigned int)(((v_col_base_45 + 2) * 16 ^ (v_row_base_44 + 80 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[40]), "+f"(output[(40) + 1]), "+f"(output[(40) + 2]), "+f"(output[(40) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[44]), "+f"(output[(44) + 1]), "+f"(output[(44) + 2]), "+f"(output[(44) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_44 + 96) * 4 + (unsigned int)(((v_col_base_45 + 2) * 16 ^ (v_row_base_44 + 96 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[48]), "+f"(output[(48) + 1]), "+f"(output[(48) + 2]), "+f"(output[(48) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[52]), "+f"(output[(52) + 1]), "+f"(output[(52) + 2]), "+f"(output[(52) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_44 + 112) * 4 + (unsigned int)(((v_col_base_45 + 2) * 16 ^ (v_row_base_44 + 112 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[56]), "+f"(output[(56) + 1]), "+f"(output[(56) + 2]), "+f"(output[(56) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[60]), "+f"(output[(60) + 1]), "+f"(output[(60) + 2]), "+f"(output[(60) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        if (preload_count < selected_count) {
            __syncthreads();
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(v_full_addr, 8192);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(v_tma_smem_addr), "l"((&V_map)), "r"(next_physical * 64), "r"(0), "r"(head), "r"(batch),
                           "r"(v_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        current_physical = next_physical;
    }
    #pragma unroll 1
    for (int load_count_1 = 1; load_count_1 < selected_count; load_count_1++) {
        mbarrier_wait(k_full_addr, _phase_k_full_0);
        _phase_k_full_0 ^= 1;
        unsigned int q_row_base_1 = warp * 16 + lane % 16;
        unsigned int q_col_base_2 = lane / 16;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
            : "r"(q_smem_addr + (q_row_base_1 * 8 + (unsigned int)((q_col_base_2 * 16 ^ (q_row_base_1 & 7) << 4) / 16)) * 16)
            : "memory");
        unsigned int q_row_base_0_1 = warp * 16 + lane % 16;
        unsigned int q_col_base_1_1 = lane / 16;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_prefetch[0]), "=r"(q_prefetch[1]), "=r"(q_prefetch[2]), "=r"(q_prefetch[3])
            : "r"(q_smem_addr + (q_row_base_0_1 * 8 + (unsigned int)(((q_col_base_1_1 + 2) * 16 ^ (q_row_base_0_1 & 7) << 4) / 16)) * 16)
            : "memory");
        unsigned int k_row_base_1 = lane % 8 + lane / 16 * 8;
        unsigned int k_col_base_1 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + (k_row_base_1 * 8 + (unsigned int)((k_col_base_1 * 16 ^ (k_row_base_1 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[0]), "=r"(scores_i32[1]), "=r"(scores_i32[2]), "=r"(scores_i32[3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[4]), "=r"(scores_i32[(4) + 1]), "=r"(scores_i32[(4) + 2]), "=r"(scores_i32[(4) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_1 + 16) * 8 + (unsigned int)((k_col_base_1 * 16 ^ (k_row_base_1 + 16 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[8]), "=r"(scores_i32[(8) + 1]), "=r"(scores_i32[(8) + 2]), "=r"(scores_i32[(8) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[12]), "=r"(scores_i32[(12) + 1]), "=r"(scores_i32[(12) + 2]), "=r"(scores_i32[(12) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_1 + 32) * 8 + (unsigned int)((k_col_base_1 * 16 ^ (k_row_base_1 + 32 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[16]), "=r"(scores_i32[(16) + 1]), "=r"(scores_i32[(16) + 2]), "=r"(scores_i32[(16) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[20]), "=r"(scores_i32[(20) + 1]), "=r"(scores_i32[(20) + 2]), "=r"(scores_i32[(20) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_1 + 48) * 8 + (unsigned int)((k_col_base_1 * 16 ^ (k_row_base_1 + 48 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[24]), "=r"(scores_i32[(24) + 1]), "=r"(scores_i32[(24) + 2]), "=r"(scores_i32[(24) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(scores_i32[28]), "=r"(scores_i32[(28) + 1]), "=r"(scores_i32[(28) + 2]), "=r"(scores_i32[(28) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        unsigned int q_row_base_2_1 = warp * 16 + lane % 16;
        unsigned int q_col_base_3_1 = lane / 16;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
            : "r"(q_smem_addr + (q_row_base_2_1 * 8 + (unsigned int)(((q_col_base_3_1 + 4) * 16 ^ (q_row_base_2_1 & 7) << 4) / 16)) * 16)
            : "memory");
        unsigned int k_row_base_4_1 = lane % 8 + lane / 16 * 8;
        unsigned int k_col_base_5_1 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + (k_row_base_4_1 * 8 + (unsigned int)(((k_col_base_5_1 + 2) * 16 ^ (k_row_base_4_1 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]), "+r"(scores_i32[(4) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_4_1 + 16) * 8 + (unsigned int)(((k_col_base_5_1 + 2) * 16 ^ (k_row_base_4_1 + 16 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]), "+r"(scores_i32[(8) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]), "+r"(scores_i32[(12) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_4_1 + 32) * 8 + (unsigned int)(((k_col_base_5_1 + 2) * 16 ^ (k_row_base_4_1 + 32 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]), "+r"(scores_i32[(16) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]), "+r"(scores_i32[(20) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_4_1 + 48) * 8 + (unsigned int)(((k_col_base_5_1 + 2) * 16 ^ (k_row_base_4_1 + 48 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]), "+r"(scores_i32[(24) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]), "+r"(scores_i32[(28) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        unsigned int q_row_base_6_1 = warp * 16 + lane % 16;
        unsigned int q_col_base_7_1 = lane / 16;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_prefetch[0]), "=r"(q_prefetch[1]), "=r"(q_prefetch[2]), "=r"(q_prefetch[3])
            : "r"(q_smem_addr + (q_row_base_6_1 * 8 + (unsigned int)(((q_col_base_7_1 + 6) * 16 ^ (q_row_base_6_1 & 7) << 4) / 16)) * 16)
            : "memory");
        unsigned int k_row_base_8_1 = lane % 8 + lane / 16 * 8;
        unsigned int k_col_base_9_1 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + (k_row_base_8_1 * 8 + (unsigned int)(((k_col_base_9_1 + 4) * 16 ^ (k_row_base_8_1 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]), "+r"(scores_i32[(4) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_8_1 + 16) * 8 + (unsigned int)(((k_col_base_9_1 + 4) * 16 ^ (k_row_base_8_1 + 16 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]), "+r"(scores_i32[(8) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]), "+r"(scores_i32[(12) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_8_1 + 32) * 8 + (unsigned int)(((k_col_base_9_1 + 4) * 16 ^ (k_row_base_8_1 + 32 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]), "+r"(scores_i32[(16) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]), "+r"(scores_i32[(20) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_8_1 + 48) * 8 + (unsigned int)(((k_col_base_9_1 + 4) * 16 ^ (k_row_base_8_1 + 48 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]), "+r"(scores_i32[(24) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]), "+r"(scores_i32[(28) + 3])
            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        unsigned int k_row_base_10_1 = lane % 8 + lane / 16 * 8;
        unsigned int k_col_base_11_1 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + (k_row_base_10_1 * 8 + (unsigned int)(((k_col_base_11_1 + 6) * 16 ^ (k_row_base_10_1 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[0]), "+r"(scores_i32[1]), "+r"(scores_i32[2]), "+r"(scores_i32[3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[4]), "+r"(scores_i32[(4) + 1]), "+r"(scores_i32[(4) + 2]), "+r"(scores_i32[(4) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_10_1 + 16) * 8 + (unsigned int)(((k_col_base_11_1 + 6) * 16 ^ (k_row_base_10_1 + 16 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[8]), "+r"(scores_i32[(8) + 1]), "+r"(scores_i32[(8) + 2]), "+r"(scores_i32[(8) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[12]), "+r"(scores_i32[(12) + 1]), "+r"(scores_i32[(12) + 2]), "+r"(scores_i32[(12) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_10_1 + 32) * 8 + (unsigned int)(((k_col_base_11_1 + 6) * 16 ^ (k_row_base_10_1 + 32 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[16]), "+r"(scores_i32[(16) + 1]), "+r"(scores_i32[(16) + 2]), "+r"(scores_i32[(16) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[20]), "+r"(scores_i32[(20) + 1]), "+r"(scores_i32[(20) + 2]), "+r"(scores_i32[(20) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(k_smem_addr + ((k_row_base_10_1 + 48) * 8 + (unsigned int)(((k_col_base_11_1 + 6) * 16 ^ (k_row_base_10_1 + 48 & 7) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[24]), "+r"(scores_i32[(24) + 1]), "+r"(scores_i32[(24) + 2]), "+r"(scores_i32[(24) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[0]), "r"(k_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(scores_i32[28]), "+r"(scores_i32[(28) + 1]), "+r"(scores_i32[(28) + 2]), "+r"(scores_i32[(28) + 3])
            : "r"(q_prefetch[0]), "r"(q_prefetch[1]), "r"(q_prefetch[2]), "r"(q_prefetch[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
        float k_scale_lane_1 = 0.0f;
        if (lane == 0) {
            k_scale_lane_1 = k_scale[k_scale_base + current_physical];
        }
        float k_scale_value_1 = __shfl_sync(0xFFFFFFFF, k_scale_lane_1, 0);
        float score_scale_log2_1 = q_softmax_log2 * k_scale_value_1;
        int valid_tokens_1 = 64;
#if CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_FULL_K64_TILES && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens_1 = seqlen_k - current_physical * 64;
            if (valid_tokens_1 > 64) {
                valid_tokens_1 = 64;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
            {
                int _min_14 = ((valid_tokens_1) < (block_sizes[(batch * num_heads + head) * num_k_blocks + current_physical]) ? (valid_tokens_1) : (block_sizes[(batch * num_heads + head) * num_k_blocks + current_physical]));
                valid_tokens_1 = _min_14;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_FULL_K64_TILES && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens_1 = seqlen_k - current_physical * 64;
            if (valid_tokens_1 > 64) {
                valid_tokens_1 = 64;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
            {
                int _min_13 = ((valid_tokens_1) < (block_sizes[(batch * num_heads + head) * num_k_blocks + current_physical]) ? (valid_tokens_1) : (block_sizes[(batch * num_heads + head) * num_k_blocks + current_physical]));
                valid_tokens_1 = _min_13;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_FULL_K64_TILES && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens_1 = seqlen_k - current_physical * 64;
            if (valid_tokens_1 > 64) {
                valid_tokens_1 = 64;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
            {
                int _min_14 = ((valid_tokens_1) < (block_sizes[batch * num_k_blocks + current_physical]) ? (valid_tokens_1) : (block_sizes[batch * num_k_blocks + current_physical]));
                valid_tokens_1 = _min_14;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_FULL_K64_TILES && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens_1 = seqlen_k - current_physical * 64;
            if (valid_tokens_1 > 64) {
                valid_tokens_1 = 64;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
            {
                int _min_13 = ((valid_tokens_1) < (block_sizes[batch * num_k_blocks + current_physical]) ? (valid_tokens_1) : (block_sizes[batch * num_k_blocks + current_physical]));
                valid_tokens_1 = _min_13;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_FULL_K64_TILES && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens_1 = seqlen_k - current_physical * 64;
            if (valid_tokens_1 > 64) {
                valid_tokens_1 = 64;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
            {
                int _min_14 = ((valid_tokens_1) < (block_sizes[current_physical]) ? (valid_tokens_1) : (block_sizes[current_physical]));
                valid_tokens_1 = _min_14;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
        }
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_FULL_K64_TILES && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        {
            valid_tokens_1 = seqlen_k - current_physical * 64;
            if (valid_tokens_1 > 64) {
                valid_tokens_1 = 64;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
            {
                int _min_13 = ((valid_tokens_1) < (block_sizes[current_physical]) ? (valid_tokens_1) : (block_sizes[current_physical]));
                valid_tokens_1 = _min_13;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
        }
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && !CAKE_SAGE_FULL_K64_TILES && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 0 && !CAKE_SAGE_FULL_K64_TILES && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        {
            valid_tokens_1 = seqlen_k - current_physical * 64;
            if (valid_tokens_1 > 64) {
                valid_tokens_1 = 64;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
            if (valid_tokens_1 < 0) {
                valid_tokens_1 = 0;
            }
        }
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_FULL_K64_TILES, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        int preload_count_1 = load_count_1 + 1;
        int next_physical_1 = 0;
        if (preload_count_1 < selected_count) {
            {
#if CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                next_physical_1 = current_physical - 1;
                if (next_physical_1 < 0) {
                    next_physical_1 = next_physical_1 + num_k_blocks;
                }
#else
                next_physical_1 = q2k_block_index[index_base + selected_count - 1 - preload_count_1];
#endif  // CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
            }
            __syncthreads();
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr, 8192);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(k_smem_addr), "l"((&K_map)), "r"(0), "r"(next_physical_1 * 64), "r"(head), "r"(batch),
                           "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        scores[0] = (float)scores_i32[0];
        scores[1] = (float)scores_i32[1];
        scores[2] = (float)scores_i32[2];
        scores[3] = (float)scores_i32[3];
        scores[4] = (float)scores_i32[4];
        scores[5] = (float)scores_i32[5];
        scores[6] = (float)scores_i32[6];
        scores[7] = (float)scores_i32[7];
        scores[8] = (float)scores_i32[8];
        scores[9] = (float)scores_i32[9];
        scores[10] = (float)scores_i32[10];
        scores[11] = (float)scores_i32[11];
        scores[12] = (float)scores_i32[12];
        scores[13] = (float)scores_i32[13];
        scores[14] = (float)scores_i32[14];
        scores[15] = (float)scores_i32[15];
        scores[16] = (float)scores_i32[16];
        scores[17] = (float)scores_i32[17];
        scores[18] = (float)scores_i32[18];
        scores[19] = (float)scores_i32[19];
        scores[20] = (float)scores_i32[20];
        scores[21] = (float)scores_i32[21];
        scores[22] = (float)scores_i32[22];
        scores[23] = (float)scores_i32[23];
        scores[24] = (float)scores_i32[24];
        scores[25] = (float)scores_i32[25];
        scores[26] = (float)scores_i32[26];
        scores[27] = (float)scores_i32[27];
        scores[28] = (float)scores_i32[28];
        scores[29] = (float)scores_i32[29];
        scores[30] = (float)scores_i32[30];
        scores[31] = (float)scores_i32[31];
        if (valid_tokens_1 < 64) {
            int lane_col_base_1 = 2 * (lane % 4);
            int token_in_tile_31 = lane_col_base_1;
            if (token_in_tile_31 >= valid_tokens_1) {
                scores[0] = -CUDART_INF_F;
            }
            int token_in_tile_0_1 = lane_col_base_1 + 1;
            if (token_in_tile_0_1 >= valid_tokens_1) {
                scores[1] = -CUDART_INF_F;
            }
            int token_in_tile_1_1 = lane_col_base_1;
            if (token_in_tile_1_1 >= valid_tokens_1) {
                scores[2] = -CUDART_INF_F;
            }
            int token_in_tile_2_1 = lane_col_base_1 + 1;
            if (token_in_tile_2_1 >= valid_tokens_1) {
                scores[3] = -CUDART_INF_F;
            }
            int token_in_tile_3_1 = lane_col_base_1 + 8;
            if (token_in_tile_3_1 >= valid_tokens_1) {
                scores[4] = -CUDART_INF_F;
            }
            int token_in_tile_4_1 = lane_col_base_1 + 8 + 1;
            if (token_in_tile_4_1 >= valid_tokens_1) {
                scores[5] = -CUDART_INF_F;
            }
            int token_in_tile_5_1 = lane_col_base_1 + 8;
            if (token_in_tile_5_1 >= valid_tokens_1) {
                scores[6] = -CUDART_INF_F;
            }
            int token_in_tile_6_1 = lane_col_base_1 + 8 + 1;
            if (token_in_tile_6_1 >= valid_tokens_1) {
                scores[7] = -CUDART_INF_F;
            }
            int token_in_tile_7_1 = 16 + lane_col_base_1;
            if (token_in_tile_7_1 >= valid_tokens_1) {
                scores[8] = -CUDART_INF_F;
            }
            int token_in_tile_8_1 = 16 + lane_col_base_1 + 1;
            if (token_in_tile_8_1 >= valid_tokens_1) {
                scores[9] = -CUDART_INF_F;
            }
            int token_in_tile_9_1 = 16 + lane_col_base_1;
            if (token_in_tile_9_1 >= valid_tokens_1) {
                scores[10] = -CUDART_INF_F;
            }
            int token_in_tile_10_1 = 16 + lane_col_base_1 + 1;
            if (token_in_tile_10_1 >= valid_tokens_1) {
                scores[11] = -CUDART_INF_F;
            }
            int token_in_tile_11_1 = 16 + lane_col_base_1 + 8;
            if (token_in_tile_11_1 >= valid_tokens_1) {
                scores[12] = -CUDART_INF_F;
            }
            int token_in_tile_12_1 = 16 + lane_col_base_1 + 8 + 1;
            if (token_in_tile_12_1 >= valid_tokens_1) {
                scores[13] = -CUDART_INF_F;
            }
            int token_in_tile_13_1 = 16 + lane_col_base_1 + 8;
            if (token_in_tile_13_1 >= valid_tokens_1) {
                scores[14] = -CUDART_INF_F;
            }
            int token_in_tile_14_1 = 16 + lane_col_base_1 + 8 + 1;
            if (token_in_tile_14_1 >= valid_tokens_1) {
                scores[15] = -CUDART_INF_F;
            }
            int token_in_tile_15_1 = 32 + lane_col_base_1;
            if (token_in_tile_15_1 >= valid_tokens_1) {
                scores[16] = -CUDART_INF_F;
            }
            int token_in_tile_16_1 = 32 + lane_col_base_1 + 1;
            if (token_in_tile_16_1 >= valid_tokens_1) {
                scores[17] = -CUDART_INF_F;
            }
            int token_in_tile_17_1 = 32 + lane_col_base_1;
            if (token_in_tile_17_1 >= valid_tokens_1) {
                scores[18] = -CUDART_INF_F;
            }
            int token_in_tile_18_1 = 32 + lane_col_base_1 + 1;
            if (token_in_tile_18_1 >= valid_tokens_1) {
                scores[19] = -CUDART_INF_F;
            }
            int token_in_tile_19_1 = 32 + lane_col_base_1 + 8;
            if (token_in_tile_19_1 >= valid_tokens_1) {
                scores[20] = -CUDART_INF_F;
            }
            int token_in_tile_20_1 = 32 + lane_col_base_1 + 8 + 1;
            if (token_in_tile_20_1 >= valid_tokens_1) {
                scores[21] = -CUDART_INF_F;
            }
            int token_in_tile_21_1 = 32 + lane_col_base_1 + 8;
            if (token_in_tile_21_1 >= valid_tokens_1) {
                scores[22] = -CUDART_INF_F;
            }
            int token_in_tile_22_1 = 32 + lane_col_base_1 + 8 + 1;
            if (token_in_tile_22_1 >= valid_tokens_1) {
                scores[23] = -CUDART_INF_F;
            }
            int token_in_tile_23_1 = 48 + lane_col_base_1;
            if (token_in_tile_23_1 >= valid_tokens_1) {
                scores[24] = -CUDART_INF_F;
            }
            int token_in_tile_24_1 = 48 + lane_col_base_1 + 1;
            if (token_in_tile_24_1 >= valid_tokens_1) {
                scores[25] = -CUDART_INF_F;
            }
            int token_in_tile_25_1 = 48 + lane_col_base_1;
            if (token_in_tile_25_1 >= valid_tokens_1) {
                scores[26] = -CUDART_INF_F;
            }
            int token_in_tile_26_1 = 48 + lane_col_base_1 + 1;
            if (token_in_tile_26_1 >= valid_tokens_1) {
                scores[27] = -CUDART_INF_F;
            }
            int token_in_tile_27_1 = 48 + lane_col_base_1 + 8;
            if (token_in_tile_27_1 >= valid_tokens_1) {
                scores[28] = -CUDART_INF_F;
            }
            int token_in_tile_28_1 = 48 + lane_col_base_1 + 8 + 1;
            if (token_in_tile_28_1 >= valid_tokens_1) {
                scores[29] = -CUDART_INF_F;
            }
            int token_in_tile_29_1 = 48 + lane_col_base_1 + 8;
            if (token_in_tile_29_1 >= valid_tokens_1) {
                scores[30] = -CUDART_INF_F;
            }
            int token_in_tile_30_1 = 48 + lane_col_base_1 + 8 + 1;
            if (token_in_tile_30_1 >= valid_tokens_1) {
                scores[31] = -CUDART_INF_F;
            }
        }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _exp2_15 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2_1, -row_shift[0]));
        p_half0[0] = _exp2_15;
        float _exp2_16 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2_1, -row_shift[1]));
        p_half0[2] = _exp2_16;
        float _exp2_17 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2_1, -row_shift[0]));
        p_half0[1] = _exp2_17;
        float _exp2_18 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2_1, -row_shift[1]));
        p_half0[3] = _exp2_18;
        float _exp2_19 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2_1, -row_shift[0]));
        p_half0[4] = _exp2_19;
        float _exp2_20 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2_1, -row_shift[1]));
        p_half0[6] = _exp2_20;
        float _exp2_21 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2_1, -row_shift[0]));
        p_half0[5] = _exp2_21;
        float _exp2_22 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2_1, -row_shift[1]));
        p_half0[7] = _exp2_22;
        float _exp2_23 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2_1, -row_shift[0]));
        p_half0[8] = _exp2_23;
        float _exp2_24 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2_1, -row_shift[1]));
        p_half0[10] = _exp2_24;
        float _exp2_25 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2_1, -row_shift[0]));
        p_half0[9] = _exp2_25;
        float _exp2_26 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2_1, -row_shift[1]));
        p_half0[11] = _exp2_26;
        float _exp2_27 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2_1, -row_shift[0]));
        p_half0[12] = _exp2_27;
        float _exp2_28 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2_1, -row_shift[1]));
        p_half0[14] = _exp2_28;
        float _exp2_29 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2_1, -row_shift[0]));
        p_half0[13] = _exp2_29;
        float _exp2_30 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2_1, -row_shift[1]));
        p_half0[15] = _exp2_30;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _exp2_14 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2_1, -row_shift[0]));
        p_half0[0] = _exp2_14;
        float _exp2_15 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2_1, -row_shift[1]));
        p_half0[2] = _exp2_15;
        float _exp2_16 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2_1, -row_shift[0]));
        p_half0[1] = _exp2_16;
        float _exp2_17 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2_1, -row_shift[1]));
        p_half0[3] = _exp2_17;
        float _exp2_18 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2_1, -row_shift[0]));
        p_half0[4] = _exp2_18;
        float _exp2_19 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2_1, -row_shift[1]));
        p_half0[6] = _exp2_19;
        float _exp2_20 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2_1, -row_shift[0]));
        p_half0[5] = _exp2_20;
        float _exp2_21 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2_1, -row_shift[1]));
        p_half0[7] = _exp2_21;
        float _exp2_22 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2_1, -row_shift[0]));
        p_half0[8] = _exp2_22;
        float _exp2_23 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2_1, -row_shift[1]));
        p_half0[10] = _exp2_23;
        float _exp2_24 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2_1, -row_shift[0]));
        p_half0[9] = _exp2_24;
        float _exp2_25 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2_1, -row_shift[1]));
        p_half0[11] = _exp2_25;
        float _exp2_26 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2_1, -row_shift[0]));
        p_half0[12] = _exp2_26;
        float _exp2_27 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2_1, -row_shift[1]));
        p_half0[14] = _exp2_27;
        float _exp2_28 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2_1, -row_shift[0]));
        p_half0[13] = _exp2_28;
        float _exp2_29 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2_1, -row_shift[1]));
        p_half0[15] = _exp2_29;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float _exp2_13 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2_1, -row_shift[0]));
        p_half0[0] = _exp2_13;
        float _exp2_14 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2_1, -row_shift[1]));
        p_half0[2] = _exp2_14;
        float _exp2_15 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2_1, -row_shift[0]));
        p_half0[1] = _exp2_15;
        float _exp2_16 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2_1, -row_shift[1]));
        p_half0[3] = _exp2_16;
        float _exp2_17 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2_1, -row_shift[0]));
        p_half0[4] = _exp2_17;
        float _exp2_18 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2_1, -row_shift[1]));
        p_half0[6] = _exp2_18;
        float _exp2_19 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2_1, -row_shift[0]));
        p_half0[5] = _exp2_19;
        float _exp2_20 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2_1, -row_shift[1]));
        p_half0[7] = _exp2_20;
        float _exp2_21 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2_1, -row_shift[0]));
        p_half0[8] = _exp2_21;
        float _exp2_22 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2_1, -row_shift[1]));
        p_half0[10] = _exp2_22;
        float _exp2_23 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2_1, -row_shift[0]));
        p_half0[9] = _exp2_23;
        float _exp2_24 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2_1, -row_shift[1]));
        p_half0[11] = _exp2_24;
        float _exp2_25 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2_1, -row_shift[0]));
        p_half0[12] = _exp2_25;
        float _exp2_26 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2_1, -row_shift[1]));
        p_half0[14] = _exp2_26;
        float _exp2_27 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2_1, -row_shift[0]));
        p_half0[13] = _exp2_27;
        float _exp2_28 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2_1, -row_shift[1]));
        p_half0[15] = _exp2_28;
#else
        float _exp2_12 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2_1, -row_shift[0]));
        p_half0[0] = _exp2_12;
        float _exp2_13 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2_1, -row_shift[1]));
        p_half0[2] = _exp2_13;
        float _exp2_14 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2_1, -row_shift[0]));
        p_half0[1] = _exp2_14;
        float _exp2_15 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2_1, -row_shift[1]));
        p_half0[3] = _exp2_15;
        float _exp2_16 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2_1, -row_shift[0]));
        p_half0[4] = _exp2_16;
        float _exp2_17 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2_1, -row_shift[1]));
        p_half0[6] = _exp2_17;
        float _exp2_18 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2_1, -row_shift[0]));
        p_half0[5] = _exp2_18;
        float _exp2_19 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2_1, -row_shift[1]));
        p_half0[7] = _exp2_19;
        float _exp2_20 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2_1, -row_shift[0]));
        p_half0[8] = _exp2_20;
        float _exp2_21 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2_1, -row_shift[1]));
        p_half0[10] = _exp2_21;
        float _exp2_22 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2_1, -row_shift[0]));
        p_half0[9] = _exp2_22;
        float _exp2_23 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2_1, -row_shift[1]));
        p_half0[11] = _exp2_23;
        float _exp2_24 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2_1, -row_shift[0]));
        p_half0[12] = _exp2_24;
        float _exp2_25 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2_1, -row_shift[1]));
        p_half0[14] = _exp2_25;
        float _exp2_26 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2_1, -row_shift[0]));
        p_half0[13] = _exp2_26;
        float _exp2_27 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2_1, -row_shift[1]));
        p_half0[15] = _exp2_27;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float carry_current_1[2];
        float carry_safe_1[2];
        float carry_delta_1[2];
        float previous_max_1 = row_max[0];
        float group_max_1 = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[4], scores[5]));
        group_max_1 = fmaxf(group_max_1, fmaxf(fmaxf(scores[8], scores[9]), fmaxf(scores[12], scores[13])));
        group_max_1 = fmaxf(group_max_1, fmaxf(fmaxf(scores[16], scores[17]), fmaxf(scores[20], scores[21])));
        group_max_1 = fmaxf(group_max_1, fmaxf(fmaxf(scores[24], scores[25]), fmaxf(scores[28], scores[29])));
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 1);
        group_max_1 = fmaxf(group_max_1, _shfl_xor_sync_31);
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 2);
        group_max_1 = fmaxf(group_max_1, _shfl_xor_sync_32);
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 1);
        group_max_1 = fmaxf(group_max_1, _shfl_xor_sync_30);
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 2);
        group_max_1 = fmaxf(group_max_1, _shfl_xor_sync_31);
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 1);
        group_max_1 = fmaxf(group_max_1, _shfl_xor_sync_29);
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 2);
        group_max_1 = fmaxf(group_max_1, _shfl_xor_sync_30);
#else
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 1);
        group_max_1 = fmaxf(group_max_1, _shfl_xor_sync_28);
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, group_max_1, 2);
        group_max_1 = fmaxf(group_max_1, _shfl_xor_sync_29);
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float current_max_1 = fmaxf(previous_max_1, group_max_1 * score_scale_log2_1);
#if CAKE_SAGE_FULL_K64_TILES
        float safe_max_1 = current_max_1;
#else
        float safe_max_1 = ((current_max_1 == -CUDART_INF_F) ? 0.0f : current_max_1);
#endif  // CAKE_SAGE_FULL_K64_TILES
        carry_current_1[0] = current_max_1;
        carry_safe_1[0] = safe_max_1;
        carry_delta_1[0] = previous_max_1 - safe_max_1;
        row_scale[0] = 1.0f;
        float previous_max_12_1 = row_max[1];
        float group_max_13_1 = fmaxf(fmaxf(scores[2], scores[3]), fmaxf(scores[6], scores[7]));
        group_max_13_1 = fmaxf(group_max_13_1, fmaxf(fmaxf(scores[10], scores[11]), fmaxf(scores[14], scores[15])));
        group_max_13_1 = fmaxf(group_max_13_1, fmaxf(fmaxf(scores[18], scores[19]), fmaxf(scores[22], scores[23])));
        group_max_13_1 = fmaxf(group_max_13_1, fmaxf(fmaxf(scores[26], scores[27]), fmaxf(scores[30], scores[31])));
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, group_max_13_1, 1);
        group_max_13_1 = fmaxf(group_max_13_1, _shfl_xor_sync_33);
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, group_max_13_1, 2);
        group_max_13_1 = fmaxf(group_max_13_1, _shfl_xor_sync_34);
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, group_max_13_1, 1);
        group_max_13_1 = fmaxf(group_max_13_1, _shfl_xor_sync_32);
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, group_max_13_1, 2);
        group_max_13_1 = fmaxf(group_max_13_1, _shfl_xor_sync_33);
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, group_max_13_1, 1);
        group_max_13_1 = fmaxf(group_max_13_1, _shfl_xor_sync_31);
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, group_max_13_1, 2);
        group_max_13_1 = fmaxf(group_max_13_1, _shfl_xor_sync_32);
#else
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, group_max_13_1, 1);
        group_max_13_1 = fmaxf(group_max_13_1, _shfl_xor_sync_30);
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, group_max_13_1, 2);
        group_max_13_1 = fmaxf(group_max_13_1, _shfl_xor_sync_31);
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float current_max_14_1 = fmaxf(previous_max_12_1, group_max_13_1 * score_scale_log2_1);
#if CAKE_SAGE_FULL_K64_TILES
        float safe_max_15_1 = current_max_14_1;
#else
        float safe_max_15_1 = ((current_max_14_1 == -CUDART_INF_F) ? 0.0f : current_max_14_1);
#endif  // CAKE_SAGE_FULL_K64_TILES
        carry_current_1[1] = current_max_14_1;
        carry_safe_1[1] = safe_max_15_1;
        carry_delta_1[1] = previous_max_12_1 - safe_max_15_1;
        row_scale[1] = 1.0f;
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        int _warp_vote_35 = __any_sync(0xFFFFFFFF, carry_delta_1[0] < -0.8073549f || carry_delta_1[1] < -0.8073549f);
        if (_warp_vote_35 != 0) {
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        int _warp_vote_34 = __any_sync(0xFFFFFFFF, carry_delta_1[0] < -0.8073549f || carry_delta_1[1] < -0.8073549f);
        if (_warp_vote_34 != 0) {
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        int _warp_vote_33 = __any_sync(0xFFFFFFFF, carry_delta_1[0] < -0.8073549f || carry_delta_1[1] < -0.8073549f);
        if (_warp_vote_33 != 0) {
#else
        int _warp_vote_32 = __any_sync(0xFFFFFFFF, carry_delta_1[0] < -0.8073549f || carry_delta_1[1] < -0.8073549f);
        if (_warp_vote_32 != 0) {
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
            if (carry_delta_1[0] < -0.8073549f) {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float _exp2_36 = approx_exp2(carry_delta_1[0]);
                row_scale[0] = _exp2_36;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float _exp2_35 = approx_exp2(carry_delta_1[0]);
                row_scale[0] = _exp2_35;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                float _exp2_34 = approx_exp2(carry_delta_1[0]);
                row_scale[0] = _exp2_34;
#else
                float _exp2_33 = approx_exp2(carry_delta_1[0]);
                row_scale[0] = _exp2_33;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                row_max[0] = carry_current_1[0];
                row_shift[0] = carry_safe_1[0] - 8.0f;
            }
            if (carry_delta_1[1] < -0.8073549f) {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float _exp2_37 = approx_exp2(carry_delta_1[1]);
                row_scale[1] = _exp2_37;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float _exp2_36 = approx_exp2(carry_delta_1[1]);
                row_scale[1] = _exp2_36;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                float _exp2_35 = approx_exp2(carry_delta_1[1]);
                row_scale[1] = _exp2_35;
#else
                float _exp2_34 = approx_exp2(carry_delta_1[1]);
                row_scale[1] = _exp2_34;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                row_max[1] = carry_current_1[1];
                row_shift[1] = carry_safe_1[1] - 8.0f;
            }
            {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float2 _pair_scale_even2_38 = make_float2(row_scale[0], row_scale[0]);
                float2 _pair_scale_odd2_38 = make_float2(row_scale[1], row_scale[1]);
                float2* _pair_scale_src2_38 = reinterpret_cast<float2*>(&output[0]);
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_38));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_38[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_38));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
                float2 _pair_scale_even2_37 = make_float2(row_scale[0], row_scale[0]);
                float2 _pair_scale_odd2_37 = make_float2(row_scale[1], row_scale[1]);
                float2* _pair_scale_src2_37 = reinterpret_cast<float2*>(&output[0]);
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_37));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_37[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_37));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
                float2 _pair_scale_even2_36 = make_float2(row_scale[0], row_scale[0]);
                float2 _pair_scale_odd2_36 = make_float2(row_scale[1], row_scale[1]);
                float2* _pair_scale_src2_36 = reinterpret_cast<float2*>(&output[0]);
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_36));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_36[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_36));
#else
                float2 _pair_scale_even2_35 = make_float2(row_scale[0], row_scale[0]);
                float2 _pair_scale_odd2_35 = make_float2(row_scale[1], row_scale[1]);
                float2* _pair_scale_src2_35 = reinterpret_cast<float2*>(&output[0]);
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_35));
                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_35[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_35));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
            }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
            float _exp2_39 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2_1, -row_shift[0]));
            p_half0[0] = _exp2_39;
            float _exp2_40 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2_1, -row_shift[1]));
            p_half0[2] = _exp2_40;
            float _exp2_41 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2_1, -row_shift[0]));
            p_half0[1] = _exp2_41;
            float _exp2_42 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2_1, -row_shift[1]));
            p_half0[3] = _exp2_42;
            float _exp2_43 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2_1, -row_shift[0]));
            p_half0[4] = _exp2_43;
            float _exp2_44 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2_1, -row_shift[1]));
            p_half0[6] = _exp2_44;
            float _exp2_45 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2_1, -row_shift[0]));
            p_half0[5] = _exp2_45;
            float _exp2_46 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2_1, -row_shift[1]));
            p_half0[7] = _exp2_46;
            float _exp2_47 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2_1, -row_shift[0]));
            p_half0[8] = _exp2_47;
            float _exp2_48 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2_1, -row_shift[1]));
            p_half0[10] = _exp2_48;
            float _exp2_49 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2_1, -row_shift[0]));
            p_half0[9] = _exp2_49;
            float _exp2_50 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2_1, -row_shift[1]));
            p_half0[11] = _exp2_50;
            float _exp2_51 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2_1, -row_shift[0]));
            p_half0[12] = _exp2_51;
            float _exp2_52 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2_1, -row_shift[1]));
            p_half0[14] = _exp2_52;
            float _exp2_53 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2_1, -row_shift[0]));
            p_half0[13] = _exp2_53;
            float _exp2_54 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2_1, -row_shift[1]));
            p_half0[15] = _exp2_54;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
            float _exp2_38 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2_1, -row_shift[0]));
            p_half0[0] = _exp2_38;
            float _exp2_39 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2_1, -row_shift[1]));
            p_half0[2] = _exp2_39;
            float _exp2_40 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2_1, -row_shift[0]));
            p_half0[1] = _exp2_40;
            float _exp2_41 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2_1, -row_shift[1]));
            p_half0[3] = _exp2_41;
            float _exp2_42 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2_1, -row_shift[0]));
            p_half0[4] = _exp2_42;
            float _exp2_43 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2_1, -row_shift[1]));
            p_half0[6] = _exp2_43;
            float _exp2_44 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2_1, -row_shift[0]));
            p_half0[5] = _exp2_44;
            float _exp2_45 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2_1, -row_shift[1]));
            p_half0[7] = _exp2_45;
            float _exp2_46 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2_1, -row_shift[0]));
            p_half0[8] = _exp2_46;
            float _exp2_47 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2_1, -row_shift[1]));
            p_half0[10] = _exp2_47;
            float _exp2_48 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2_1, -row_shift[0]));
            p_half0[9] = _exp2_48;
            float _exp2_49 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2_1, -row_shift[1]));
            p_half0[11] = _exp2_49;
            float _exp2_50 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2_1, -row_shift[0]));
            p_half0[12] = _exp2_50;
            float _exp2_51 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2_1, -row_shift[1]));
            p_half0[14] = _exp2_51;
            float _exp2_52 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2_1, -row_shift[0]));
            p_half0[13] = _exp2_52;
            float _exp2_53 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2_1, -row_shift[1]));
            p_half0[15] = _exp2_53;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
            float _exp2_37 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2_1, -row_shift[0]));
            p_half0[0] = _exp2_37;
            float _exp2_38 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2_1, -row_shift[1]));
            p_half0[2] = _exp2_38;
            float _exp2_39 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2_1, -row_shift[0]));
            p_half0[1] = _exp2_39;
            float _exp2_40 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2_1, -row_shift[1]));
            p_half0[3] = _exp2_40;
            float _exp2_41 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2_1, -row_shift[0]));
            p_half0[4] = _exp2_41;
            float _exp2_42 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2_1, -row_shift[1]));
            p_half0[6] = _exp2_42;
            float _exp2_43 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2_1, -row_shift[0]));
            p_half0[5] = _exp2_43;
            float _exp2_44 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2_1, -row_shift[1]));
            p_half0[7] = _exp2_44;
            float _exp2_45 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2_1, -row_shift[0]));
            p_half0[8] = _exp2_45;
            float _exp2_46 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2_1, -row_shift[1]));
            p_half0[10] = _exp2_46;
            float _exp2_47 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2_1, -row_shift[0]));
            p_half0[9] = _exp2_47;
            float _exp2_48 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2_1, -row_shift[1]));
            p_half0[11] = _exp2_48;
            float _exp2_49 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2_1, -row_shift[0]));
            p_half0[12] = _exp2_49;
            float _exp2_50 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2_1, -row_shift[1]));
            p_half0[14] = _exp2_50;
            float _exp2_51 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2_1, -row_shift[0]));
            p_half0[13] = _exp2_51;
            float _exp2_52 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2_1, -row_shift[1]));
            p_half0[15] = _exp2_52;
#else
            float _exp2_36 = approx_exp2(__fmaf_rn(scores[0], score_scale_log2_1, -row_shift[0]));
            p_half0[0] = _exp2_36;
            float _exp2_37 = approx_exp2(__fmaf_rn(scores[2], score_scale_log2_1, -row_shift[1]));
            p_half0[2] = _exp2_37;
            float _exp2_38 = approx_exp2(__fmaf_rn(scores[1], score_scale_log2_1, -row_shift[0]));
            p_half0[1] = _exp2_38;
            float _exp2_39 = approx_exp2(__fmaf_rn(scores[3], score_scale_log2_1, -row_shift[1]));
            p_half0[3] = _exp2_39;
            float _exp2_40 = approx_exp2(__fmaf_rn(scores[4], score_scale_log2_1, -row_shift[0]));
            p_half0[4] = _exp2_40;
            float _exp2_41 = approx_exp2(__fmaf_rn(scores[6], score_scale_log2_1, -row_shift[1]));
            p_half0[6] = _exp2_41;
            float _exp2_42 = approx_exp2(__fmaf_rn(scores[5], score_scale_log2_1, -row_shift[0]));
            p_half0[5] = _exp2_42;
            float _exp2_43 = approx_exp2(__fmaf_rn(scores[7], score_scale_log2_1, -row_shift[1]));
            p_half0[7] = _exp2_43;
            float _exp2_44 = approx_exp2(__fmaf_rn(scores[8], score_scale_log2_1, -row_shift[0]));
            p_half0[8] = _exp2_44;
            float _exp2_45 = approx_exp2(__fmaf_rn(scores[10], score_scale_log2_1, -row_shift[1]));
            p_half0[10] = _exp2_45;
            float _exp2_46 = approx_exp2(__fmaf_rn(scores[9], score_scale_log2_1, -row_shift[0]));
            p_half0[9] = _exp2_46;
            float _exp2_47 = approx_exp2(__fmaf_rn(scores[11], score_scale_log2_1, -row_shift[1]));
            p_half0[11] = _exp2_47;
            float _exp2_48 = approx_exp2(__fmaf_rn(scores[12], score_scale_log2_1, -row_shift[0]));
            p_half0[12] = _exp2_48;
            float _exp2_49 = approx_exp2(__fmaf_rn(scores[14], score_scale_log2_1, -row_shift[1]));
            p_half0[14] = _exp2_49;
            float _exp2_50 = approx_exp2(__fmaf_rn(scores[13], score_scale_log2_1, -row_shift[0]));
            p_half0[13] = _exp2_50;
            float _exp2_51 = approx_exp2(__fmaf_rn(scores[15], score_scale_log2_1, -row_shift[1]));
            p_half0[15] = _exp2_51;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        }
        local_sum[0] = p_half0[0];
        local_sum[1] = p_half0[2];
        local_sum[0] = local_sum[0] + p_half0[1];
        local_sum[1] = local_sum[1] + p_half0[3];
        local_sum[0] = local_sum[0] + p_half0[4];
        local_sum[1] = local_sum[1] + p_half0[6];
        local_sum[0] = local_sum[0] + p_half0[5];
        local_sum[1] = local_sum[1] + p_half0[7];
        local_sum[0] = local_sum[0] + p_half0[8];
        local_sum[1] = local_sum[1] + p_half0[10];
        local_sum[0] = local_sum[0] + p_half0[9];
        local_sum[1] = local_sum[1] + p_half0[11];
        local_sum[0] = local_sum[0] + p_half0[12];
        local_sum[1] = local_sum[1] + p_half0[14];
        local_sum[0] = local_sum[0] + p_half0[13];
        local_sum[1] = local_sum[1] + p_half0[15];
        pack_values[0] = p_half0[0];
        pack_values[1] = p_half0[1];
        pack_values[2] = p_half0[4];
        pack_values[3] = p_half0[5];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[0] = _packed;
        }
        pack_values[0] = p_half0[2];
        pack_values[1] = p_half0[3];
        pack_values[2] = p_half0[6];
        pack_values[3] = p_half0[7];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[1] = _packed;
        }
        pack_values[0] = p_half0[8];
        pack_values[1] = p_half0[9];
        pack_values[2] = p_half0[12];
        pack_values[3] = p_half0[13];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[2] = _packed;
        }
        pack_values[0] = p_half0[10];
        pack_values[1] = p_half0[11];
        pack_values[2] = p_half0[14];
        pack_values[3] = p_half0[15];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[3] = _packed;
        }
        mbarrier_wait(v_full_addr, _phase_v_full_0);
        _phase_v_full_0 ^= 1;
        unsigned int v_row_base_1 = lane % 8 + lane / 16 * 8;
        unsigned int v_col_base_1 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + (v_row_base_1 * 4 + (unsigned int)((v_col_base_1 * 16 ^ (v_row_base_1 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[0]), "+f"(output[1]), "+f"(output[2]), "+f"(output[3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[4]), "+f"(output[(4) + 1]), "+f"(output[(4) + 2]), "+f"(output[(4) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_1 + 16) * 4 + (unsigned int)((v_col_base_1 * 16 ^ (v_row_base_1 + 16 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[8]), "+f"(output[(8) + 1]), "+f"(output[(8) + 2]), "+f"(output[(8) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[12]), "+f"(output[(12) + 1]), "+f"(output[(12) + 2]), "+f"(output[(12) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_1 + 32) * 4 + (unsigned int)((v_col_base_1 * 16 ^ (v_row_base_1 + 32 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[16]), "+f"(output[(16) + 1]), "+f"(output[(16) + 2]), "+f"(output[(16) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[20]), "+f"(output[(20) + 1]), "+f"(output[(20) + 2]), "+f"(output[(20) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_1 + 48) * 4 + (unsigned int)((v_col_base_1 * 16 ^ (v_row_base_1 + 48 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[24]), "+f"(output[(24) + 1]), "+f"(output[(24) + 2]), "+f"(output[(24) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[28]), "+f"(output[(28) + 1]), "+f"(output[(28) + 2]), "+f"(output[(28) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_1 + 64) * 4 + (unsigned int)((v_col_base_1 * 16 ^ (v_row_base_1 + 64 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[32]), "+f"(output[(32) + 1]), "+f"(output[(32) + 2]), "+f"(output[(32) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[36]), "+f"(output[(36) + 1]), "+f"(output[(36) + 2]), "+f"(output[(36) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_1 + 80) * 4 + (unsigned int)((v_col_base_1 * 16 ^ (v_row_base_1 + 80 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[40]), "+f"(output[(40) + 1]), "+f"(output[(40) + 2]), "+f"(output[(40) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[44]), "+f"(output[(44) + 1]), "+f"(output[(44) + 2]), "+f"(output[(44) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_1 + 96) * 4 + (unsigned int)((v_col_base_1 * 16 ^ (v_row_base_1 + 96 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[48]), "+f"(output[(48) + 1]), "+f"(output[(48) + 2]), "+f"(output[(48) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[52]), "+f"(output[(52) + 1]), "+f"(output[(52) + 2]), "+f"(output[(52) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_1 + 112) * 4 + (unsigned int)((v_col_base_1 * 16 ^ (v_row_base_1 + 112 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[56]), "+f"(output[(56) + 1]), "+f"(output[(56) + 2]), "+f"(output[(56) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[60]), "+f"(output[(60) + 1]), "+f"(output[(60) + 2]), "+f"(output[(60) + 3])
            : "r"(probabilities[0]), "r"(probabilities[1]), "r"(probabilities[2]), "r"(probabilities[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        float value0_1 = approx_exp2(__fmaf_rn(scores[16], score_scale_log2_1, -row_shift[0]));
        scores[16] = value0_1;
        local_sum[0] = local_sum[0] + value0_1;
        float value1_1 = approx_exp2(__fmaf_rn(scores[18], score_scale_log2_1, -row_shift[1]));
        scores[18] = value1_1;
        local_sum[1] = local_sum[1] + value1_1;
        float value0_16_1 = approx_exp2(__fmaf_rn(scores[17], score_scale_log2_1, -row_shift[0]));
        scores[17] = value0_16_1;
        local_sum[0] = local_sum[0] + value0_16_1;
        float value1_17_1 = approx_exp2(__fmaf_rn(scores[19], score_scale_log2_1, -row_shift[1]));
        scores[19] = value1_17_1;
        local_sum[1] = local_sum[1] + value1_17_1;
        float value0_18_1 = approx_exp2(__fmaf_rn(scores[20], score_scale_log2_1, -row_shift[0]));
        scores[20] = value0_18_1;
        local_sum[0] = local_sum[0] + value0_18_1;
        float value1_19_1 = approx_exp2(__fmaf_rn(scores[22], score_scale_log2_1, -row_shift[1]));
        scores[22] = value1_19_1;
        local_sum[1] = local_sum[1] + value1_19_1;
        float value0_20_1 = approx_exp2(__fmaf_rn(scores[21], score_scale_log2_1, -row_shift[0]));
        scores[21] = value0_20_1;
        local_sum[0] = local_sum[0] + value0_20_1;
        float value1_21_1 = approx_exp2(__fmaf_rn(scores[23], score_scale_log2_1, -row_shift[1]));
        scores[23] = value1_21_1;
        local_sum[1] = local_sum[1] + value1_21_1;
        float value0_22_1 = approx_exp2(__fmaf_rn(scores[24], score_scale_log2_1, -row_shift[0]));
        scores[24] = value0_22_1;
        local_sum[0] = local_sum[0] + value0_22_1;
        float value1_23_1 = approx_exp2(__fmaf_rn(scores[26], score_scale_log2_1, -row_shift[1]));
        scores[26] = value1_23_1;
        local_sum[1] = local_sum[1] + value1_23_1;
        float value0_24_1 = approx_exp2(__fmaf_rn(scores[25], score_scale_log2_1, -row_shift[0]));
        scores[25] = value0_24_1;
        local_sum[0] = local_sum[0] + value0_24_1;
        float value1_25_1 = approx_exp2(__fmaf_rn(scores[27], score_scale_log2_1, -row_shift[1]));
        scores[27] = value1_25_1;
        local_sum[1] = local_sum[1] + value1_25_1;
        float value0_26_1 = approx_exp2(__fmaf_rn(scores[28], score_scale_log2_1, -row_shift[0]));
        scores[28] = value0_26_1;
        local_sum[0] = local_sum[0] + value0_26_1;
        float value1_27_1 = approx_exp2(__fmaf_rn(scores[30], score_scale_log2_1, -row_shift[1]));
        scores[30] = value1_27_1;
        local_sum[1] = local_sum[1] + value1_27_1;
        float value0_28_1 = approx_exp2(__fmaf_rn(scores[29], score_scale_log2_1, -row_shift[0]));
        scores[29] = value0_28_1;
        local_sum[0] = local_sum[0] + value0_28_1;
        float value1_29_1 = approx_exp2(__fmaf_rn(scores[31], score_scale_log2_1, -row_shift[1]));
        scores[31] = value1_29_1;
        local_sum[1] = local_sum[1] + value1_29_1;
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 1);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 2);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_56;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 1);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_54;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 2);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_55;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 1);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 2);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_54;
#else
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 1);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, local_sum[0], 2);
        local_sum[0] = local_sum[0] + _shfl_xor_sync_53;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        row_sum[0] = __fmaf_rn(row_sum[0], row_scale[0], local_sum[0]);
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 1);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_57;
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 2);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_58;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 1);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_56;
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 2);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_57;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 1);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 2);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_56;
#else
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 1);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_54;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, local_sum[1], 2);
        local_sum[1] = local_sum[1] + _shfl_xor_sync_55;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        row_sum[1] = __fmaf_rn(row_sum[1], row_scale[1], local_sum[1]);
        pack_values[0] = scores[16];
        pack_values[1] = scores[17];
        pack_values[2] = scores[20];
        pack_values[3] = scores[21];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[4] = _packed;
        }
        pack_values[0] = scores[18];
        pack_values[1] = scores[19];
        pack_values[2] = scores[22];
        pack_values[3] = scores[23];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[5] = _packed;
        }
        pack_values[0] = scores[24];
        pack_values[1] = scores[25];
        pack_values[2] = scores[28];
        pack_values[3] = scores[29];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[6] = _packed;
        }
        pack_values[0] = scores[26];
        pack_values[1] = scores[27];
        pack_values[2] = scores[30];
        pack_values[3] = scores[31];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(pack_values[0]), "f"(pack_values[1]),
                                   "f"(pack_values[2]), "f"(pack_values[3]));
            probabilities[7] = _packed;
        }
        unsigned int v_row_base_30 = lane % 8 + lane / 16 * 8;
        unsigned int v_col_base_31 = lane / 8 % 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + (v_row_base_30 * 4 + (unsigned int)(((v_col_base_31 + 2) * 16 ^ (v_row_base_30 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[0]), "+f"(output[1]), "+f"(output[2]), "+f"(output[3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[4]), "+f"(output[(4) + 1]), "+f"(output[(4) + 2]), "+f"(output[(4) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_30 + 16) * 4 + (unsigned int)(((v_col_base_31 + 2) * 16 ^ (v_row_base_30 + 16 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[8]), "+f"(output[(8) + 1]), "+f"(output[(8) + 2]), "+f"(output[(8) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[12]), "+f"(output[(12) + 1]), "+f"(output[(12) + 2]), "+f"(output[(12) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_30 + 32) * 4 + (unsigned int)(((v_col_base_31 + 2) * 16 ^ (v_row_base_30 + 32 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[16]), "+f"(output[(16) + 1]), "+f"(output[(16) + 2]), "+f"(output[(16) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[20]), "+f"(output[(20) + 1]), "+f"(output[(20) + 2]), "+f"(output[(20) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_30 + 48) * 4 + (unsigned int)(((v_col_base_31 + 2) * 16 ^ (v_row_base_30 + 48 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[24]), "+f"(output[(24) + 1]), "+f"(output[(24) + 2]), "+f"(output[(24) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[28]), "+f"(output[(28) + 1]), "+f"(output[(28) + 2]), "+f"(output[(28) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_30 + 64) * 4 + (unsigned int)(((v_col_base_31 + 2) * 16 ^ (v_row_base_30 + 64 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[32]), "+f"(output[(32) + 1]), "+f"(output[(32) + 2]), "+f"(output[(32) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[36]), "+f"(output[(36) + 1]), "+f"(output[(36) + 2]), "+f"(output[(36) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_30 + 80) * 4 + (unsigned int)(((v_col_base_31 + 2) * 16 ^ (v_row_base_30 + 80 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[40]), "+f"(output[(40) + 1]), "+f"(output[(40) + 2]), "+f"(output[(40) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[44]), "+f"(output[(44) + 1]), "+f"(output[(44) + 2]), "+f"(output[(44) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_30 + 96) * 4 + (unsigned int)(((v_col_base_31 + 2) * 16 ^ (v_row_base_30 + 96 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[48]), "+f"(output[(48) + 1]), "+f"(output[(48) + 2]), "+f"(output[(48) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[52]), "+f"(output[(52) + 1]), "+f"(output[(52) + 2]), "+f"(output[(52) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
            : "r"(v_smem_addr + ((v_row_base_30 + 112) * 4 + (unsigned int)(((v_col_base_31 + 2) * 16 ^ (v_row_base_30 + 112 >> 1 & 3) << 4) / 16)) * 16)
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[56]), "+f"(output[(56) + 1]), "+f"(output[(56) + 2]), "+f"(output[(56) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(output[60]), "+f"(output[(60) + 1]), "+f"(output[(60) + 2]), "+f"(output[(60) + 3])
            : "r"(probabilities[4]), "r"(probabilities[(4) + 1]), "r"(probabilities[(4) + 2]), "r"(probabilities[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
        if (preload_count_1 < selected_count) {
            __syncthreads();
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(v_full_addr, 8192);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(v_tma_smem_addr), "l"((&V_map)), "r"(next_physical_1 * 64), "r"(0), "r"(head), "r"(batch),
                           "r"(v_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        current_physical = next_physical_1;
    }
    __syncthreads();
    float reciprocal[2];
    float total = row_sum[0];
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    float _rcp_59 = approx_rcp(((total != 0.0f && total == total) ? total : 1.0f));
    reciprocal[0] = _rcp_59;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    float _rcp_58 = approx_rcp(((total != 0.0f && total == total) ? total : 1.0f));
    reciprocal[0] = _rcp_58;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    float _rcp_57 = approx_rcp(((total != 0.0f && total == total) ? total : 1.0f));
    reciprocal[0] = _rcp_57;
#else
    float _rcp_56 = approx_rcp(((total != 0.0f && total == total) ? total : 1.0f));
    reciprocal[0] = _rcp_56;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    float total_0 = row_sum[1];
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    float _rcp_60 = approx_rcp(((total_0 != 0.0f && total_0 == total_0) ? total_0 : 1.0f));
    reciprocal[1] = _rcp_60;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    float _rcp_59 = approx_rcp(((total_0 != 0.0f && total_0 == total_0) ? total_0 : 1.0f));
    reciprocal[1] = _rcp_59;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    float _rcp_58 = approx_rcp(((total_0 != 0.0f && total_0 == total_0) ? total_0 : 1.0f));
    reciprocal[1] = _rcp_58;
#else
    float _rcp_57 = approx_rcp(((total_0 != 0.0f && total_0 == total_0) ? total_0 : 1.0f));
    reciprocal[1] = _rcp_57;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    unsigned int scale_base = v_scale_base + lane % 4 * 2;
    float _vec_load_0[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_61 = *reinterpret_cast<const float2*>(v_scale + scale_base);
        _vec_load_0[0] = _v2_61.x;
        _vec_load_0[0 + 1] = _v2_61.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_60 = *reinterpret_cast<const float2*>(v_scale + scale_base);
        _vec_load_0[0] = _v2_60.x;
        _vec_load_0[0 + 1] = _v2_60.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_59 = *reinterpret_cast<const float2*>(v_scale + scale_base);
        _vec_load_0[0] = _v2_59.x;
        _vec_load_0[0 + 1] = _v2_59.y;
#else
        float2 _v2_58 = *reinterpret_cast<const float2*>(v_scale + scale_base);
        _vec_load_0[0] = _v2_58.x;
        _vec_load_0[0 + 1] = _v2_58.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float _vec_load_1[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_62 = *reinterpret_cast<const float2*>(v_scale + scale_base + 8);
        _vec_load_1[0] = _v2_62.x;
        _vec_load_1[0 + 1] = _v2_62.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_61 = *reinterpret_cast<const float2*>(v_scale + scale_base + 8);
        _vec_load_1[0] = _v2_61.x;
        _vec_load_1[0 + 1] = _v2_61.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_60 = *reinterpret_cast<const float2*>(v_scale + scale_base + 8);
        _vec_load_1[0] = _v2_60.x;
        _vec_load_1[0 + 1] = _v2_60.y;
#else
        float2 _v2_59 = *reinterpret_cast<const float2*>(v_scale + scale_base + 8);
        _vec_load_1[0] = _v2_59.x;
        _vec_load_1[0 + 1] = _v2_59.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float scale0 = _vec_load_0[0];
    float scale1 = _vec_load_0[1];
    float scale2 = _vec_load_1[0];
    float scale3 = _vec_load_1[1];
    output[0] = output[0] * reciprocal[0] * scale0;
    output[1] = output[1] * reciprocal[0] * scale1;
    output[2] = output[2] * reciprocal[1] * scale0;
    output[3] = output[3] * reciprocal[1] * scale1;
    output[4] = output[4] * reciprocal[0] * scale2;
    output[5] = output[5] * reciprocal[0] * scale3;
    output[6] = output[6] * reciprocal[1] * scale2;
    output[7] = output[7] * reciprocal[1] * scale3;
    unsigned int packed[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output[_lp*2 + 0], output[_lp*2+1 + 0]));
        packed[_lp] = *(uint32_t*)&_bf2;
    }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_63 = static_cast<uint32_t>(o_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_62 = static_cast<uint32_t>(o_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    uint32_t _stmatrix_addr_61 = static_cast<uint32_t>(o_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
    uint32_t _stmatrix_addr_60 = static_cast<uint32_t>(o_smem_addr + (unsigned int)((lane / 16 / 8 * 512 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_63), "r"(*reinterpret_cast<const uint32_t*>(&packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[3]))
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_62), "r"(*reinterpret_cast<const uint32_t*>(&packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[3]))
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        :: "r"(_stmatrix_addr_61), "r"(*reinterpret_cast<const uint32_t*>(&packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[3]))
#else
        :: "r"(_stmatrix_addr_60), "r"(*reinterpret_cast<const uint32_t*>(&packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[3]))
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        : "memory");
    unsigned int scale_base_1 = v_scale_base + 16 + lane % 4 * 2;
    float _vec_load_2[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_64 = *reinterpret_cast<const float2*>(v_scale + scale_base_1);
        _vec_load_2[0] = _v2_64.x;
        _vec_load_2[0 + 1] = _v2_64.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_63 = *reinterpret_cast<const float2*>(v_scale + scale_base_1);
        _vec_load_2[0] = _v2_63.x;
        _vec_load_2[0 + 1] = _v2_63.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_62 = *reinterpret_cast<const float2*>(v_scale + scale_base_1);
        _vec_load_2[0] = _v2_62.x;
        _vec_load_2[0 + 1] = _v2_62.y;
#else
        float2 _v2_61 = *reinterpret_cast<const float2*>(v_scale + scale_base_1);
        _vec_load_2[0] = _v2_61.x;
        _vec_load_2[0 + 1] = _v2_61.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float _vec_load_3[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_65 = *reinterpret_cast<const float2*>(v_scale + scale_base_1 + 8);
        _vec_load_3[0] = _v2_65.x;
        _vec_load_3[0 + 1] = _v2_65.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_64 = *reinterpret_cast<const float2*>(v_scale + scale_base_1 + 8);
        _vec_load_3[0] = _v2_64.x;
        _vec_load_3[0 + 1] = _v2_64.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_63 = *reinterpret_cast<const float2*>(v_scale + scale_base_1 + 8);
        _vec_load_3[0] = _v2_63.x;
        _vec_load_3[0 + 1] = _v2_63.y;
#else
        float2 _v2_62 = *reinterpret_cast<const float2*>(v_scale + scale_base_1 + 8);
        _vec_load_3[0] = _v2_62.x;
        _vec_load_3[0 + 1] = _v2_62.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float scale0_2 = _vec_load_2[0];
    float scale1_3 = _vec_load_2[1];
    float scale2_4 = _vec_load_3[0];
    float scale3_5 = _vec_load_3[1];
    output[8] = output[8] * reciprocal[0] * scale0_2;
    output[9] = output[9] * reciprocal[0] * scale1_3;
    output[10] = output[10] * reciprocal[1] * scale0_2;
    output[11] = output[11] * reciprocal[1] * scale1_3;
    output[12] = output[12] * reciprocal[0] * scale2_4;
    output[13] = output[13] * reciprocal[0] * scale3_5;
    output[14] = output[14] * reciprocal[1] * scale2_4;
    output[15] = output[15] * reciprocal[1] * scale3_5;
    unsigned int packed_6[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output[_lp*2 + 8], output[_lp*2+1 + 8]));
        packed_6[_lp] = *(uint32_t*)&_bf2;
    }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_66 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((2 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((2 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_65 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((2 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((2 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    uint32_t _stmatrix_addr_64 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((2 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((2 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
    uint32_t _stmatrix_addr_63 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((2 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((2 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_66), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[3]))
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_65), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[3]))
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        :: "r"(_stmatrix_addr_64), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[3]))
#else
        :: "r"(_stmatrix_addr_63), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[3]))
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        : "memory");
    unsigned int scale_base_7 = v_scale_base + 32 + lane % 4 * 2;
    float _vec_load_4[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_67 = *reinterpret_cast<const float2*>(v_scale + scale_base_7);
        _vec_load_4[0] = _v2_67.x;
        _vec_load_4[0 + 1] = _v2_67.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_66 = *reinterpret_cast<const float2*>(v_scale + scale_base_7);
        _vec_load_4[0] = _v2_66.x;
        _vec_load_4[0 + 1] = _v2_66.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_65 = *reinterpret_cast<const float2*>(v_scale + scale_base_7);
        _vec_load_4[0] = _v2_65.x;
        _vec_load_4[0 + 1] = _v2_65.y;
#else
        float2 _v2_64 = *reinterpret_cast<const float2*>(v_scale + scale_base_7);
        _vec_load_4[0] = _v2_64.x;
        _vec_load_4[0 + 1] = _v2_64.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float _vec_load_5[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_68 = *reinterpret_cast<const float2*>(v_scale + scale_base_7 + 8);
        _vec_load_5[0] = _v2_68.x;
        _vec_load_5[0 + 1] = _v2_68.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_67 = *reinterpret_cast<const float2*>(v_scale + scale_base_7 + 8);
        _vec_load_5[0] = _v2_67.x;
        _vec_load_5[0 + 1] = _v2_67.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_66 = *reinterpret_cast<const float2*>(v_scale + scale_base_7 + 8);
        _vec_load_5[0] = _v2_66.x;
        _vec_load_5[0 + 1] = _v2_66.y;
#else
        float2 _v2_65 = *reinterpret_cast<const float2*>(v_scale + scale_base_7 + 8);
        _vec_load_5[0] = _v2_65.x;
        _vec_load_5[0 + 1] = _v2_65.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float scale0_8 = _vec_load_4[0];
    float scale1_9 = _vec_load_4[1];
    float scale2_10 = _vec_load_5[0];
    float scale3_11 = _vec_load_5[1];
    output[16] = output[16] * reciprocal[0] * scale0_8;
    output[17] = output[17] * reciprocal[0] * scale1_9;
    output[18] = output[18] * reciprocal[1] * scale0_8;
    output[19] = output[19] * reciprocal[1] * scale1_9;
    output[20] = output[20] * reciprocal[0] * scale2_10;
    output[21] = output[21] * reciprocal[0] * scale3_11;
    output[22] = output[22] * reciprocal[1] * scale2_10;
    output[23] = output[23] * reciprocal[1] * scale3_11;
    unsigned int packed_12[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output[_lp*2 + 16], output[_lp*2+1 + 16]));
        packed_12[_lp] = *(uint32_t*)&_bf2;
    }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_69 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((4 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((4 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_68 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((4 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((4 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    uint32_t _stmatrix_addr_67 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((4 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((4 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
    uint32_t _stmatrix_addr_66 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((4 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((4 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_69), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[3]))
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_68), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[3]))
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        :: "r"(_stmatrix_addr_67), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[3]))
#else
        :: "r"(_stmatrix_addr_66), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[3]))
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        : "memory");
    unsigned int scale_base_13 = v_scale_base + 48 + lane % 4 * 2;
    float _vec_load_6[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_70 = *reinterpret_cast<const float2*>(v_scale + scale_base_13);
        _vec_load_6[0] = _v2_70.x;
        _vec_load_6[0 + 1] = _v2_70.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_69 = *reinterpret_cast<const float2*>(v_scale + scale_base_13);
        _vec_load_6[0] = _v2_69.x;
        _vec_load_6[0 + 1] = _v2_69.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_68 = *reinterpret_cast<const float2*>(v_scale + scale_base_13);
        _vec_load_6[0] = _v2_68.x;
        _vec_load_6[0 + 1] = _v2_68.y;
#else
        float2 _v2_67 = *reinterpret_cast<const float2*>(v_scale + scale_base_13);
        _vec_load_6[0] = _v2_67.x;
        _vec_load_6[0 + 1] = _v2_67.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float _vec_load_7[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_71 = *reinterpret_cast<const float2*>(v_scale + scale_base_13 + 8);
        _vec_load_7[0] = _v2_71.x;
        _vec_load_7[0 + 1] = _v2_71.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_70 = *reinterpret_cast<const float2*>(v_scale + scale_base_13 + 8);
        _vec_load_7[0] = _v2_70.x;
        _vec_load_7[0 + 1] = _v2_70.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_69 = *reinterpret_cast<const float2*>(v_scale + scale_base_13 + 8);
        _vec_load_7[0] = _v2_69.x;
        _vec_load_7[0 + 1] = _v2_69.y;
#else
        float2 _v2_68 = *reinterpret_cast<const float2*>(v_scale + scale_base_13 + 8);
        _vec_load_7[0] = _v2_68.x;
        _vec_load_7[0 + 1] = _v2_68.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float scale0_14 = _vec_load_6[0];
    float scale1_15 = _vec_load_6[1];
    float scale2_16 = _vec_load_7[0];
    float scale3_17 = _vec_load_7[1];
    output[24] = output[24] * reciprocal[0] * scale0_14;
    output[25] = output[25] * reciprocal[0] * scale1_15;
    output[26] = output[26] * reciprocal[1] * scale0_14;
    output[27] = output[27] * reciprocal[1] * scale1_15;
    output[28] = output[28] * reciprocal[0] * scale2_16;
    output[29] = output[29] * reciprocal[0] * scale3_17;
    output[30] = output[30] * reciprocal[1] * scale2_16;
    output[31] = output[31] * reciprocal[1] * scale3_17;
    unsigned int packed_18[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output[_lp*2 + 24], output[_lp*2+1 + 24]));
        packed_18[_lp] = *(uint32_t*)&_bf2;
    }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_72 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((6 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((6 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((6 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((6 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    uint32_t _stmatrix_addr_70 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((6 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((6 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
    uint32_t _stmatrix_addr_69 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((6 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((6 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_72), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[3]))
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[3]))
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        :: "r"(_stmatrix_addr_70), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[3]))
#else
        :: "r"(_stmatrix_addr_69), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[3]))
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        : "memory");
    unsigned int scale_base_19 = v_scale_base + 64 + lane % 4 * 2;
    float _vec_load_8[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_73 = *reinterpret_cast<const float2*>(v_scale + scale_base_19);
        _vec_load_8[0] = _v2_73.x;
        _vec_load_8[0 + 1] = _v2_73.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_72 = *reinterpret_cast<const float2*>(v_scale + scale_base_19);
        _vec_load_8[0] = _v2_72.x;
        _vec_load_8[0 + 1] = _v2_72.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_71 = *reinterpret_cast<const float2*>(v_scale + scale_base_19);
        _vec_load_8[0] = _v2_71.x;
        _vec_load_8[0 + 1] = _v2_71.y;
#else
        float2 _v2_70 = *reinterpret_cast<const float2*>(v_scale + scale_base_19);
        _vec_load_8[0] = _v2_70.x;
        _vec_load_8[0 + 1] = _v2_70.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float _vec_load_9[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_74 = *reinterpret_cast<const float2*>(v_scale + scale_base_19 + 8);
        _vec_load_9[0] = _v2_74.x;
        _vec_load_9[0 + 1] = _v2_74.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_73 = *reinterpret_cast<const float2*>(v_scale + scale_base_19 + 8);
        _vec_load_9[0] = _v2_73.x;
        _vec_load_9[0 + 1] = _v2_73.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_72 = *reinterpret_cast<const float2*>(v_scale + scale_base_19 + 8);
        _vec_load_9[0] = _v2_72.x;
        _vec_load_9[0 + 1] = _v2_72.y;
#else
        float2 _v2_71 = *reinterpret_cast<const float2*>(v_scale + scale_base_19 + 8);
        _vec_load_9[0] = _v2_71.x;
        _vec_load_9[0 + 1] = _v2_71.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float scale0_20 = _vec_load_8[0];
    float scale1_21 = _vec_load_8[1];
    float scale2_22 = _vec_load_9[0];
    float scale3_23 = _vec_load_9[1];
    output[32] = output[32] * reciprocal[0] * scale0_20;
    output[33] = output[33] * reciprocal[0] * scale1_21;
    output[34] = output[34] * reciprocal[1] * scale0_20;
    output[35] = output[35] * reciprocal[1] * scale1_21;
    output[36] = output[36] * reciprocal[0] * scale2_22;
    output[37] = output[37] * reciprocal[0] * scale3_23;
    output[38] = output[38] * reciprocal[1] * scale2_22;
    output[39] = output[39] * reciprocal[1] * scale3_23;
    unsigned int packed_24[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output[_lp*2 + 32], output[_lp*2+1 + 32]));
        packed_24[_lp] = *(uint32_t*)&_bf2;
    }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_75 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((8 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((8 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_74 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((8 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((8 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    uint32_t _stmatrix_addr_73 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((8 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((8 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
    uint32_t _stmatrix_addr_72 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((8 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((8 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[3]))
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_74), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[3]))
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        :: "r"(_stmatrix_addr_73), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[3]))
#else
        :: "r"(_stmatrix_addr_72), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_24[3]))
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        : "memory");
    unsigned int scale_base_25 = v_scale_base + 80 + lane % 4 * 2;
    float _vec_load_10[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_76 = *reinterpret_cast<const float2*>(v_scale + scale_base_25);
        _vec_load_10[0] = _v2_76.x;
        _vec_load_10[0 + 1] = _v2_76.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_75 = *reinterpret_cast<const float2*>(v_scale + scale_base_25);
        _vec_load_10[0] = _v2_75.x;
        _vec_load_10[0 + 1] = _v2_75.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_74 = *reinterpret_cast<const float2*>(v_scale + scale_base_25);
        _vec_load_10[0] = _v2_74.x;
        _vec_load_10[0 + 1] = _v2_74.y;
#else
        float2 _v2_73 = *reinterpret_cast<const float2*>(v_scale + scale_base_25);
        _vec_load_10[0] = _v2_73.x;
        _vec_load_10[0 + 1] = _v2_73.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float _vec_load_11[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_77 = *reinterpret_cast<const float2*>(v_scale + scale_base_25 + 8);
        _vec_load_11[0] = _v2_77.x;
        _vec_load_11[0 + 1] = _v2_77.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_76 = *reinterpret_cast<const float2*>(v_scale + scale_base_25 + 8);
        _vec_load_11[0] = _v2_76.x;
        _vec_load_11[0 + 1] = _v2_76.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_75 = *reinterpret_cast<const float2*>(v_scale + scale_base_25 + 8);
        _vec_load_11[0] = _v2_75.x;
        _vec_load_11[0 + 1] = _v2_75.y;
#else
        float2 _v2_74 = *reinterpret_cast<const float2*>(v_scale + scale_base_25 + 8);
        _vec_load_11[0] = _v2_74.x;
        _vec_load_11[0 + 1] = _v2_74.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float scale0_26 = _vec_load_10[0];
    float scale1_27 = _vec_load_10[1];
    float scale2_28 = _vec_load_11[0];
    float scale3_29 = _vec_load_11[1];
    output[40] = output[40] * reciprocal[0] * scale0_26;
    output[41] = output[41] * reciprocal[0] * scale1_27;
    output[42] = output[42] * reciprocal[1] * scale0_26;
    output[43] = output[43] * reciprocal[1] * scale1_27;
    output[44] = output[44] * reciprocal[0] * scale2_28;
    output[45] = output[45] * reciprocal[0] * scale3_29;
    output[46] = output[46] * reciprocal[1] * scale2_28;
    output[47] = output[47] * reciprocal[1] * scale3_29;
    unsigned int packed_30[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output[_lp*2 + 40], output[_lp*2+1 + 40]));
        packed_30[_lp] = *(uint32_t*)&_bf2;
    }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_78 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((10 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((10 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_77 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((10 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((10 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    uint32_t _stmatrix_addr_76 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((10 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((10 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
    uint32_t _stmatrix_addr_75 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((10 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((10 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_78), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[3]))
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_77), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[3]))
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        :: "r"(_stmatrix_addr_76), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[3]))
#else
        :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_30[3]))
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        : "memory");
    unsigned int scale_base_31 = v_scale_base + 96 + lane % 4 * 2;
    float _vec_load_12[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_79 = *reinterpret_cast<const float2*>(v_scale + scale_base_31);
        _vec_load_12[0] = _v2_79.x;
        _vec_load_12[0 + 1] = _v2_79.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_78 = *reinterpret_cast<const float2*>(v_scale + scale_base_31);
        _vec_load_12[0] = _v2_78.x;
        _vec_load_12[0 + 1] = _v2_78.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_77 = *reinterpret_cast<const float2*>(v_scale + scale_base_31);
        _vec_load_12[0] = _v2_77.x;
        _vec_load_12[0 + 1] = _v2_77.y;
#else
        float2 _v2_76 = *reinterpret_cast<const float2*>(v_scale + scale_base_31);
        _vec_load_12[0] = _v2_76.x;
        _vec_load_12[0 + 1] = _v2_76.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float _vec_load_13[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_80 = *reinterpret_cast<const float2*>(v_scale + scale_base_31 + 8);
        _vec_load_13[0] = _v2_80.x;
        _vec_load_13[0 + 1] = _v2_80.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_79 = *reinterpret_cast<const float2*>(v_scale + scale_base_31 + 8);
        _vec_load_13[0] = _v2_79.x;
        _vec_load_13[0 + 1] = _v2_79.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_78 = *reinterpret_cast<const float2*>(v_scale + scale_base_31 + 8);
        _vec_load_13[0] = _v2_78.x;
        _vec_load_13[0 + 1] = _v2_78.y;
#else
        float2 _v2_77 = *reinterpret_cast<const float2*>(v_scale + scale_base_31 + 8);
        _vec_load_13[0] = _v2_77.x;
        _vec_load_13[0 + 1] = _v2_77.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float scale0_32 = _vec_load_12[0];
    float scale1_33 = _vec_load_12[1];
    float scale2_34 = _vec_load_13[0];
    float scale3_35 = _vec_load_13[1];
    output[48] = output[48] * reciprocal[0] * scale0_32;
    output[49] = output[49] * reciprocal[0] * scale1_33;
    output[50] = output[50] * reciprocal[1] * scale0_32;
    output[51] = output[51] * reciprocal[1] * scale1_33;
    output[52] = output[52] * reciprocal[0] * scale2_34;
    output[53] = output[53] * reciprocal[0] * scale3_35;
    output[54] = output[54] * reciprocal[1] * scale2_34;
    output[55] = output[55] * reciprocal[1] * scale3_35;
    unsigned int packed_36[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output[_lp*2 + 48], output[_lp*2+1 + 48]));
        packed_36[_lp] = *(uint32_t*)&_bf2;
    }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_81 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((12 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((12 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_80 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((12 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((12 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    uint32_t _stmatrix_addr_79 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((12 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((12 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
    uint32_t _stmatrix_addr_78 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((12 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((12 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_81), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[3]))
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_80), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[3]))
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        :: "r"(_stmatrix_addr_79), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[3]))
#else
        :: "r"(_stmatrix_addr_78), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_36[3]))
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        : "memory");
    unsigned int scale_base_37 = v_scale_base + 112 + lane % 4 * 2;
    float _vec_load_14[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_82 = *reinterpret_cast<const float2*>(v_scale + scale_base_37);
        _vec_load_14[0] = _v2_82.x;
        _vec_load_14[0 + 1] = _v2_82.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_81 = *reinterpret_cast<const float2*>(v_scale + scale_base_37);
        _vec_load_14[0] = _v2_81.x;
        _vec_load_14[0 + 1] = _v2_81.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_80 = *reinterpret_cast<const float2*>(v_scale + scale_base_37);
        _vec_load_14[0] = _v2_80.x;
        _vec_load_14[0 + 1] = _v2_80.y;
#else
        float2 _v2_79 = *reinterpret_cast<const float2*>(v_scale + scale_base_37);
        _vec_load_14[0] = _v2_79.x;
        _vec_load_14[0 + 1] = _v2_79.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float _vec_load_15[2];
    {
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_83 = *reinterpret_cast<const float2*>(v_scale + scale_base_37 + 8);
        _vec_load_15[0] = _v2_83.x;
        _vec_load_15[0 + 1] = _v2_83.y;
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        float2 _v2_82 = *reinterpret_cast<const float2*>(v_scale + scale_base_37 + 8);
        _vec_load_15[0] = _v2_82.x;
        _vec_load_15[0 + 1] = _v2_82.y;
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        float2 _v2_81 = *reinterpret_cast<const float2*>(v_scale + scale_base_37 + 8);
        _vec_load_15[0] = _v2_81.x;
        _vec_load_15[0 + 1] = _v2_81.y;
#else
        float2 _v2_80 = *reinterpret_cast<const float2*>(v_scale + scale_base_37 + 8);
        _vec_load_15[0] = _v2_80.x;
        _vec_load_15[0 + 1] = _v2_80.y;
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    }
    float scale0_38 = _vec_load_14[0];
    float scale1_39 = _vec_load_14[1];
    float scale2_40 = _vec_load_15[0];
    float scale3_41 = _vec_load_15[1];
    output[56] = output[56] * reciprocal[0] * scale0_38;
    output[57] = output[57] * reciprocal[0] * scale1_39;
    output[58] = output[58] * reciprocal[1] * scale0_38;
    output[59] = output[59] * reciprocal[1] * scale1_39;
    output[60] = output[60] * reciprocal[0] * scale2_40;
    output[61] = output[61] * reciprocal[0] * scale3_41;
    output[62] = output[62] * reciprocal[1] * scale2_40;
    output[63] = output[63] * reciprocal[1] * scale3_41;
    unsigned int packed_42[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output[_lp*2 + 56], output[_lp*2+1 + 56]));
        packed_42[_lp] = *(uint32_t*)&_bf2;
    }
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_84 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((14 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((14 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
    uint32_t _stmatrix_addr_83 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((14 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((14 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    uint32_t _stmatrix_addr_82 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((14 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((14 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
    uint32_t _stmatrix_addr_81 = static_cast<uint32_t>(o_smem_addr + (unsigned int)(((14 + lane / 16) / 8 * 512 + (warp * 16 + lane % 16) * 8 + ((14 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_84), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[3]))
#elif (CAKE_SAGE_BLOCK_SIZES_MODE == 1 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 2 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES) || (CAKE_SAGE_BLOCK_SIZES_MODE == 3 && !CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES)
        :: "r"(_stmatrix_addr_83), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[3]))
#elif CAKE_SAGE_BLOCK_SIZES_MODE == 0 && CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        :: "r"(_stmatrix_addr_82), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[3]))
#else
        :: "r"(_stmatrix_addr_81), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_42[3]))
#endif  // CAKE_SAGE_BLOCK_SIZES_MODE, CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES
        : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        if (elect_sync()) {
            tma_store_5d((&O_map), 0, q_base, head, batch, 0, o_smem_addr);
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
