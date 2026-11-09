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



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)





extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_vsa_sm120_aeffa6c83df0348168ae(const __grid_constant__ CUtensorMap q_map, const __grid_constant__ CUtensorMap k_map, const __grid_constant__ CUtensorMap v_map, __half* __restrict__ out, float* __restrict__ lse, float* __restrict__ partial_ws, int* __restrict__ split_counters, int* __restrict__ q2k_index, int* __restrict__ work, int num_heads, int group_size, unsigned int total_work, int write_lse, float softmax_scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define full_addr (mbar_base + 16)
    #define empty_addr (mbar_base + 48)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __half* q_smem = reinterpret_cast<__half*>(smem_raw + 1024);
    const int q_addr = smem + 1024;
    __half* kv_smem = reinterpret_cast<__half*>(smem_raw + 33792);
    const int kv_addr = smem + 33792;

    // Mbarrier init (10 barriers at smem_raw[0..80))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'ring' ---
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // empty: 4 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            mbarrier_init(smem + 56, 128);
            mbarrier_init(smem + 64, 128);
            mbarrier_init(smem + 72, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int instance_id = (warp - 0) / 4;
    int g = instance_id;
    int gtid = (int)tid % 128;
    unsigned int pos = 0;
    unsigned int q_tile = q_addr + (unsigned int)g * 16384;
    unsigned int _phase_q_full = 0;
    unsigned int _phase_full = 0;
    #pragma unroll 1
    for (unsigned int pair = bid; pair < total_work; pair += 2 * num_bids) {
        int pair_item = (int)pair * 8;
        int other_item = ((int)pair + num_bids) * 8;
        int first0 = work[pair_item + 2];
        int end0 = work[pair_item + 3];
        int split0 = work[pair_item + 4];
        int first1 = 0;
        int end1 = 0;
        int split1 = -1;
        if (pair + (unsigned int)num_bids < total_work) {
            first1 = work[other_item + 2];
            end1 = work[other_item + 3];
            split1 = work[other_item + 4];
        }
        unsigned int work_id = pair + (unsigned int)(g * num_bids);
        int live = ((work_id < total_work) ? 1 : 0);
        int item = (int)((live != 0) ? work_id : pair) * 8;
        int head = work[item];
        int q_block = work[item + 1];
        int first = ((g == 0) ? first0 : first1);
        int count = ((g == 0) ? end0 : end1) - first;
        int split_group = ((g == 0) ? split0 : split1);
        int split_parts = work[item + 5];
        int split_part = work[item + 6];
        int split_slot = work[item + 7];
        int kv_head = head / group_size;
        int q_base = q_block * 64;
        if (warp % 4 == 0) {
            if (count < 0) {
                asm volatile("trap;" ::: "memory");
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr + (g) * 8, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(q_tile), "l"((&q_map)), "r"(0), "r"(q_base), "r"(head), "r"(0),
                       "r"(q_full_addr + (g) * 8), "l"(0x12F0000000000000ULL) : "memory");
            }
            if (count > 0) {
                int block0 = q2k_index[first];
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(full_addr + ((unsigned int)g + pos) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(kv_addr + ((unsigned int)g + pos) * 16384), "l"((&k_map)), "r"(0), "r"(block0 * 64), "r"(kv_head), "r"(0),
                           "r"(full_addr + ((unsigned int)g + pos) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(full_addr + ((unsigned int)g + ((pos + 2 < 4) ? pos + 2 : 0)) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(kv_addr + ((unsigned int)g + ((pos + 2 < 4) ? pos + 2 : 0)) * 16384), "l"((&v_map)), "r"(0), "r"(block0 * 64), "r"(kv_head), "r"(0),
                           "r"(full_addr + ((unsigned int)g + ((pos + 2 < 4) ? pos + 2 : 0)) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        unsigned int q_frags[32];
        unsigned int k_frag[4];
        unsigned int v_frag[4];
        float qk_acc[32];
        unsigned int p_frag[16];
        float out_acc[64];
        float m_state[2];
        float d_state[2];
        out_acc[0] = 0.0f;
        out_acc[1] = 0.0f;
        out_acc[2] = 0.0f;
        out_acc[3] = 0.0f;
        out_acc[4] = 0.0f;
        out_acc[5] = 0.0f;
        out_acc[6] = 0.0f;
        out_acc[7] = 0.0f;
        out_acc[8] = 0.0f;
        out_acc[9] = 0.0f;
        out_acc[10] = 0.0f;
        out_acc[11] = 0.0f;
        out_acc[12] = 0.0f;
        out_acc[13] = 0.0f;
        out_acc[14] = 0.0f;
        out_acc[15] = 0.0f;
        out_acc[16] = 0.0f;
        out_acc[17] = 0.0f;
        out_acc[18] = 0.0f;
        out_acc[19] = 0.0f;
        out_acc[20] = 0.0f;
        out_acc[21] = 0.0f;
        out_acc[22] = 0.0f;
        out_acc[23] = 0.0f;
        out_acc[24] = 0.0f;
        out_acc[25] = 0.0f;
        out_acc[26] = 0.0f;
        out_acc[27] = 0.0f;
        out_acc[28] = 0.0f;
        out_acc[29] = 0.0f;
        out_acc[30] = 0.0f;
        out_acc[31] = 0.0f;
        out_acc[32] = 0.0f;
        out_acc[33] = 0.0f;
        out_acc[34] = 0.0f;
        out_acc[35] = 0.0f;
        out_acc[36] = 0.0f;
        out_acc[37] = 0.0f;
        out_acc[38] = 0.0f;
        out_acc[39] = 0.0f;
        out_acc[40] = 0.0f;
        out_acc[41] = 0.0f;
        out_acc[42] = 0.0f;
        out_acc[43] = 0.0f;
        out_acc[44] = 0.0f;
        out_acc[45] = 0.0f;
        out_acc[46] = 0.0f;
        out_acc[47] = 0.0f;
        out_acc[48] = 0.0f;
        out_acc[49] = 0.0f;
        out_acc[50] = 0.0f;
        out_acc[51] = 0.0f;
        out_acc[52] = 0.0f;
        out_acc[53] = 0.0f;
        out_acc[54] = 0.0f;
        out_acc[55] = 0.0f;
        out_acc[56] = 0.0f;
        out_acc[57] = 0.0f;
        out_acc[58] = 0.0f;
        out_acc[59] = 0.0f;
        out_acc[60] = 0.0f;
        out_acc[61] = 0.0f;
        out_acc[62] = 0.0f;
        out_acc[63] = 0.0f;
        m_state[0] = -CUDART_INF_F;
        d_state[0] = 1.0f;
        m_state[1] = -CUDART_INF_F;
        d_state[1] = 1.0f;
        mbarrier_wait(q_full_addr + (g) * 8, _phase_q_full);
        _phase_q_full ^= 1;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[0]), "=r"(q_frags[1]), "=r"(q_frags[2]), "=r"(q_frags[3])
            : "r"(q_tile + (unsigned int)((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(q_tile + (unsigned int)((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[8]), "=r"(q_frags[9]), "=r"(q_frags[10]), "=r"(q_frags[11])
            : "r"(q_tile + (unsigned int)((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[12]), "=r"(q_frags[13]), "=r"(q_frags[14]), "=r"(q_frags[15])
            : "r"(q_tile + (unsigned int)((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[16]), "=r"(q_frags[17]), "=r"(q_frags[18]), "=r"(q_frags[19])
            : "r"(q_tile + (unsigned int)(((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 512) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[20]), "=r"(q_frags[21]), "=r"(q_frags[22]), "=r"(q_frags[23])
            : "r"(q_tile + (unsigned int)(((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 512 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[24]), "=r"(q_frags[25]), "=r"(q_frags[26]), "=r"(q_frags[27])
            : "r"(q_tile + (unsigned int)(((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 512 ^ 2 ^ 6) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[28]), "=r"(q_frags[29]), "=r"(q_frags[30]), "=r"(q_frags[31])
            : "r"(q_tile + (unsigned int)(((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 512 ^ 2 ^ 6 ^ 2) * 16))
            : "memory");
        int next_block = 0;
        #pragma unroll 1
        for (int j = 0; j < count; j++) {
            if (j + 1 < count) {
                next_block = q2k_index[first + j + 1];
            }
            mbarrier_wait(full_addr + ((unsigned int)g + pos) * 8, _phase_full);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[0]), "=f"(qk_acc[1]), "=f"(qk_acc[2]), "=f"(qk_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[4]), "=f"(qk_acc[(4) + 1]), "=f"(qk_acc[(4) + 2]), "=f"(qk_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[8]), "=f"(qk_acc[(8) + 1]), "=f"(qk_acc[(8) + 2]), "=f"(qk_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[12]), "=f"(qk_acc[(12) + 1]), "=f"(qk_acc[(12) + 2]), "=f"(qk_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[16]), "=f"(qk_acc[(16) + 1]), "=f"(qk_acc[(16) + 2]), "=f"(qk_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[20]), "=f"(qk_acc[(20) + 1]), "=f"(qk_acc[(20) + 2]), "=f"(qk_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 384) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[24]), "=f"(qk_acc[(24) + 1]), "=f"(qk_acc[(24) + 2]), "=f"(qk_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[28]), "=f"(qk_acc[(28) + 1]), "=f"(qk_acc[(28) + 2]), "=f"(qk_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 384) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 384) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 384) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 384) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 384) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 384) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 256) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 384) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            mbarrier_arrive(empty_addr + ((unsigned int)g + pos) * 8);
            if (warp % 4 == 0) {
                mbarrier_wait(empty_addr + ((unsigned int)g + pos) * 8, _phase_full);
                if (j + 1 < count) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(full_addr + ((unsigned int)g + ((pos + 4 >= 4) ? pos + 4 - 4 : pos + 4)) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(kv_addr + ((unsigned int)g + pos) * 16384), "l"((&k_map)), "r"(0), "r"(next_block * 64), "r"(kv_head), "r"(0),
                               "r"(full_addr + ((unsigned int)g + ((pos + 4 >= 4) ? pos + 4 - 4 : pos + 4)) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            pos += 2;
            if (pos >= 4) { pos -= 4; _phase_full ^= 1; }
            float o_scale[2];
            float m_prev[2];
            m_prev[0] = m_state[0];
            m_state[0] = fmaxf(m_state[0], fmaxf(fmaxf(qk_acc[0], qk_acc[1]), fmaxf(qk_acc[4], qk_acc[5])));
            m_state[0] = fmaxf(m_state[0], fmaxf(fmaxf(qk_acc[8], qk_acc[9]), fmaxf(qk_acc[12], qk_acc[13])));
            m_state[0] = fmaxf(m_state[0], fmaxf(fmaxf(qk_acc[16], qk_acc[17]), fmaxf(qk_acc[20], qk_acc[21])));
            m_state[0] = fmaxf(m_state[0], fmaxf(fmaxf(qk_acc[24], qk_acc[25]), fmaxf(qk_acc[28], qk_acc[29])));
            float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, m_state[0], 2);
            m_state[0] = fmaxf(m_state[0], _shfl_xor_sync_0);
            float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, m_state[0], 1);
            m_state[0] = fmaxf(m_state[0], _shfl_xor_sync_1);
            o_scale[0] = 1.0f;
            m_prev[1] = m_state[1];
            m_state[1] = fmaxf(m_state[1], fmaxf(fmaxf(qk_acc[2], qk_acc[3]), fmaxf(qk_acc[6], qk_acc[7])));
            m_state[1] = fmaxf(m_state[1], fmaxf(fmaxf(qk_acc[10], qk_acc[11]), fmaxf(qk_acc[14], qk_acc[15])));
            m_state[1] = fmaxf(m_state[1], fmaxf(fmaxf(qk_acc[18], qk_acc[19]), fmaxf(qk_acc[22], qk_acc[23])));
            m_state[1] = fmaxf(m_state[1], fmaxf(fmaxf(qk_acc[26], qk_acc[27]), fmaxf(qk_acc[30], qk_acc[31])));
            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, m_state[1], 2);
            m_state[1] = fmaxf(m_state[1], _shfl_xor_sync_2);
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, m_state[1], 1);
            m_state[1] = fmaxf(m_state[1], _shfl_xor_sync_3);
            o_scale[1] = 1.0f;
            int moved = __any_sync(0xFFFFFFFF, m_state[0] != m_prev[0] || m_state[1] != m_prev[1]);
            if (moved != 0) {
                if (m_state[0] != m_prev[0]) {
                    float _exp2_4 = approx_exp2(__fmaf_rn(m_prev[0], softmax_scale_log2, -(m_state[0] * softmax_scale_log2)));
                    o_scale[0] = _exp2_4;
                }
                if (m_state[1] != m_prev[1]) {
                    float _exp2_5 = approx_exp2(__fmaf_rn(m_prev[1], softmax_scale_log2, -(m_state[1] * softmax_scale_log2)));
                    o_scale[1] = _exp2_5;
                }
                {
                    float2 _pair_scale_even2_6 = make_float2(o_scale[0], o_scale[0]);
                    float2 _pair_scale_odd2_6 = make_float2(o_scale[1], o_scale[1]);
                    float2* _pair_scale_src2_6 = reinterpret_cast<float2*>(&out_acc[0]);
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_6));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_6[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_6));
                }
            }
            float row0_shift = m_state[0] * softmax_scale_log2;
            float row1_shift = m_state[1] * softmax_scale_log2;
            float e0_0 = approx_exp2(__fmaf_rn(qk_acc[0], softmax_scale_log2, -row0_shift));
            float e1_0 = approx_exp2(__fmaf_rn(qk_acc[1], softmax_scale_log2, -row0_shift));
            qk_acc[0] = e0_0;
            qk_acc[1] = e1_0;
            float row_sum_0 = e0_0 + e1_0;
            float e0_1 = approx_exp2(__fmaf_rn(qk_acc[2], softmax_scale_log2, -row1_shift));
            float e1_1 = approx_exp2(__fmaf_rn(qk_acc[3], softmax_scale_log2, -row1_shift));
            qk_acc[2] = e0_1;
            qk_acc[3] = e1_1;
            float row_sum_1 = e0_1 + e1_1;
            float e0_2 = approx_exp2(__fmaf_rn(qk_acc[4], softmax_scale_log2, -row0_shift));
            float e1_2 = approx_exp2(__fmaf_rn(qk_acc[5], softmax_scale_log2, -row0_shift));
            qk_acc[4] = e0_2;
            qk_acc[5] = e1_2;
            float e0_3 = approx_exp2(__fmaf_rn(qk_acc[6], softmax_scale_log2, -row1_shift));
            float e1_3 = approx_exp2(__fmaf_rn(qk_acc[7], softmax_scale_log2, -row1_shift));
            qk_acc[6] = e0_3;
            qk_acc[7] = e1_3;
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(qk_acc[_lp*2 + 0], qk_acc[_lp*2+1 + 0]));
                p_frag[_lp] = *(uint32_t*)&_h2;
            }
            d_state[0] = __fmaf_rn(d_state[0], o_scale[0], row_sum_0 + e0_2 + e1_2);
            d_state[1] = __fmaf_rn(d_state[1], o_scale[1], row_sum_1 + e0_3 + e1_3);
            float e0_4 = approx_exp2(__fmaf_rn(qk_acc[8], softmax_scale_log2, -row0_shift));
            float e1_4 = approx_exp2(__fmaf_rn(qk_acc[9], softmax_scale_log2, -row0_shift));
            qk_acc[8] = e0_4;
            qk_acc[9] = e1_4;
            float row_sum_2 = e0_4 + e1_4;
            float e0_5 = approx_exp2(__fmaf_rn(qk_acc[10], softmax_scale_log2, -row1_shift));
            float e1_5 = approx_exp2(__fmaf_rn(qk_acc[11], softmax_scale_log2, -row1_shift));
            qk_acc[10] = e0_5;
            qk_acc[11] = e1_5;
            float row_sum_3 = e0_5 + e1_5;
            float e0_6 = approx_exp2(__fmaf_rn(qk_acc[12], softmax_scale_log2, -row0_shift));
            float e1_6 = approx_exp2(__fmaf_rn(qk_acc[13], softmax_scale_log2, -row0_shift));
            qk_acc[12] = e0_6;
            qk_acc[13] = e1_6;
            float e0_7 = approx_exp2(__fmaf_rn(qk_acc[14], softmax_scale_log2, -row1_shift));
            float e1_7 = approx_exp2(__fmaf_rn(qk_acc[15], softmax_scale_log2, -row1_shift));
            qk_acc[14] = e0_7;
            qk_acc[15] = e1_7;
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(qk_acc[_lp*2 + 8], qk_acc[_lp*2+1 + 8]));
                p_frag[_lp + 4] = *(uint32_t*)&_h2;
            }
            d_state[0] = d_state[0] + (row_sum_2 + e0_6 + e1_6);
            d_state[1] = d_state[1] + (row_sum_3 + e0_7 + e1_7);
            float e0_8 = approx_exp2(__fmaf_rn(qk_acc[16], softmax_scale_log2, -row0_shift));
            float e1_8 = approx_exp2(__fmaf_rn(qk_acc[17], softmax_scale_log2, -row0_shift));
            qk_acc[16] = e0_8;
            qk_acc[17] = e1_8;
            float row_sum_4 = e0_8 + e1_8;
            float e0_9 = approx_exp2(__fmaf_rn(qk_acc[18], softmax_scale_log2, -row1_shift));
            float e1_9 = approx_exp2(__fmaf_rn(qk_acc[19], softmax_scale_log2, -row1_shift));
            qk_acc[18] = e0_9;
            qk_acc[19] = e1_9;
            float row_sum_5 = e0_9 + e1_9;
            float e0_10 = approx_exp2(__fmaf_rn(qk_acc[20], softmax_scale_log2, -row0_shift));
            float e1_10 = approx_exp2(__fmaf_rn(qk_acc[21], softmax_scale_log2, -row0_shift));
            qk_acc[20] = e0_10;
            qk_acc[21] = e1_10;
            float e0_11 = approx_exp2(__fmaf_rn(qk_acc[22], softmax_scale_log2, -row1_shift));
            float e1_11 = approx_exp2(__fmaf_rn(qk_acc[23], softmax_scale_log2, -row1_shift));
            qk_acc[22] = e0_11;
            qk_acc[23] = e1_11;
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(qk_acc[_lp*2 + 16], qk_acc[_lp*2+1 + 16]));
                p_frag[_lp + 8] = *(uint32_t*)&_h2;
            }
            d_state[0] = d_state[0] + (row_sum_4 + e0_10 + e1_10);
            d_state[1] = d_state[1] + (row_sum_5 + e0_11 + e1_11);
            float e0_12 = approx_exp2(__fmaf_rn(qk_acc[24], softmax_scale_log2, -row0_shift));
            float e1_12 = approx_exp2(__fmaf_rn(qk_acc[25], softmax_scale_log2, -row0_shift));
            qk_acc[24] = e0_12;
            qk_acc[25] = e1_12;
            float row_sum_6 = e0_12 + e1_12;
            float e0_13 = approx_exp2(__fmaf_rn(qk_acc[26], softmax_scale_log2, -row1_shift));
            float e1_13 = approx_exp2(__fmaf_rn(qk_acc[27], softmax_scale_log2, -row1_shift));
            qk_acc[26] = e0_13;
            qk_acc[27] = e1_13;
            float row_sum_7 = e0_13 + e1_13;
            float e0_14 = approx_exp2(__fmaf_rn(qk_acc[28], softmax_scale_log2, -row0_shift));
            float e1_14 = approx_exp2(__fmaf_rn(qk_acc[29], softmax_scale_log2, -row0_shift));
            qk_acc[28] = e0_14;
            qk_acc[29] = e1_14;
            float e0_15 = approx_exp2(__fmaf_rn(qk_acc[30], softmax_scale_log2, -row1_shift));
            float e1_15 = approx_exp2(__fmaf_rn(qk_acc[31], softmax_scale_log2, -row1_shift));
            qk_acc[30] = e0_15;
            qk_acc[31] = e1_15;
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(qk_acc[_lp*2 + 24], qk_acc[_lp*2+1 + 24]));
                p_frag[_lp + 12] = *(uint32_t*)&_h2;
            }
            d_state[0] = d_state[0] + (row_sum_6 + e0_14 + e1_14);
            d_state[1] = d_state[1] + (row_sum_7 + e0_15 + e1_15);
            mbarrier_wait(full_addr + ((unsigned int)g + pos) * 8, _phase_full);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((lane / 16 / 8 * 512 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 2) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 2) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 4) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 4) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 6) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 6) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 8) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 8) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 10) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 10) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 12) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 12) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 14) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 14) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((lane / 16 / 8 * 512 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((lane / 16 / 8 * 512 + (32 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)((lane / 16 / 8 * 512 + (48 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(kv_addr + ((unsigned int)g + pos) * 16384 + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            mbarrier_arrive(empty_addr + ((unsigned int)g + pos) * 8);
            if (warp % 4 == 0) {
                mbarrier_wait(empty_addr + ((unsigned int)g + pos) * 8, _phase_full);
                if (j + 1 < count) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(full_addr + ((unsigned int)g + ((pos + 4 >= 4) ? pos + 4 - 4 : pos + 4)) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(kv_addr + ((unsigned int)g + pos) * 16384), "l"((&v_map)), "r"(0), "r"(next_block * 64), "r"(kv_head), "r"(0),
                               "r"(full_addr + ((unsigned int)g + ((pos + 4 >= 4) ? pos + 4 - 4 : pos + 4)) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            pos += 2;
            if (pos >= 4) { pos -= 4; _phase_full ^= 1; }
        }
        int store = live;
        if (split_group >= 0) {
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)gtid)) + (0)) = out_acc[0];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(128 + gtid))) + (0)) = out_acc[1];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(256 + gtid))) + (0)) = out_acc[2];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(384 + gtid))) + (0)) = out_acc[3];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(512 + gtid))) + (0)) = out_acc[4];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(640 + gtid))) + (0)) = out_acc[5];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(768 + gtid))) + (0)) = out_acc[6];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(896 + gtid))) + (0)) = out_acc[7];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(1024 + gtid))) + (0)) = out_acc[8];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(1152 + gtid))) + (0)) = out_acc[9];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(1280 + gtid))) + (0)) = out_acc[10];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(1408 + gtid))) + (0)) = out_acc[11];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(1536 + gtid))) + (0)) = out_acc[12];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(1664 + gtid))) + (0)) = out_acc[13];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(1792 + gtid))) + (0)) = out_acc[14];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(1920 + gtid))) + (0)) = out_acc[15];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(2048 + gtid))) + (0)) = out_acc[16];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(2176 + gtid))) + (0)) = out_acc[17];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(2304 + gtid))) + (0)) = out_acc[18];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(2432 + gtid))) + (0)) = out_acc[19];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(2560 + gtid))) + (0)) = out_acc[20];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(2688 + gtid))) + (0)) = out_acc[21];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(2816 + gtid))) + (0)) = out_acc[22];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(2944 + gtid))) + (0)) = out_acc[23];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(3072 + gtid))) + (0)) = out_acc[24];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(3200 + gtid))) + (0)) = out_acc[25];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(3328 + gtid))) + (0)) = out_acc[26];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(3456 + gtid))) + (0)) = out_acc[27];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(3584 + gtid))) + (0)) = out_acc[28];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(3712 + gtid))) + (0)) = out_acc[29];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(3840 + gtid))) + (0)) = out_acc[30];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(3968 + gtid))) + (0)) = out_acc[31];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(4096 + gtid))) + (0)) = out_acc[32];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(4224 + gtid))) + (0)) = out_acc[33];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(4352 + gtid))) + (0)) = out_acc[34];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(4480 + gtid))) + (0)) = out_acc[35];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(4608 + gtid))) + (0)) = out_acc[36];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(4736 + gtid))) + (0)) = out_acc[37];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(4864 + gtid))) + (0)) = out_acc[38];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(4992 + gtid))) + (0)) = out_acc[39];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(5120 + gtid))) + (0)) = out_acc[40];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(5248 + gtid))) + (0)) = out_acc[41];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(5376 + gtid))) + (0)) = out_acc[42];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(5504 + gtid))) + (0)) = out_acc[43];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(5632 + gtid))) + (0)) = out_acc[44];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(5760 + gtid))) + (0)) = out_acc[45];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(5888 + gtid))) + (0)) = out_acc[46];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(6016 + gtid))) + (0)) = out_acc[47];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(6144 + gtid))) + (0)) = out_acc[48];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(6272 + gtid))) + (0)) = out_acc[49];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(6400 + gtid))) + (0)) = out_acc[50];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(6528 + gtid))) + (0)) = out_acc[51];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(6656 + gtid))) + (0)) = out_acc[52];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(6784 + gtid))) + (0)) = out_acc[53];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(6912 + gtid))) + (0)) = out_acc[54];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(7040 + gtid))) + (0)) = out_acc[55];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(7168 + gtid))) + (0)) = out_acc[56];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(7296 + gtid))) + (0)) = out_acc[57];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(7424 + gtid))) + (0)) = out_acc[58];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(7552 + gtid))) + (0)) = out_acc[59];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(7680 + gtid))) + (0)) = out_acc[60];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(7808 + gtid))) + (0)) = out_acc[61];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(7936 + gtid))) + (0)) = out_acc[62];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(8064 + gtid))) + (0)) = out_acc[63];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(8192 + gtid))) + (0)) = m_state[0];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(8448 + gtid))) + (0)) = d_state[0];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(8320 + gtid))) + (0)) = m_state[1];
            *(reinterpret_cast<float*>(partial_ws + ((long long)split_slot * 8704 + (long long)(8576 + gtid))) + (0)) = d_state[1];
            __threadfence();
            asm volatile("barrier.sync %0, 128;" :: "r"(1 + g) : "memory");
            int last = 0;
            if (gtid == 0) {
                int arrived = atomicAdd(&split_counters[split_group], 1);
                if (arrived + 1 == split_parts) {
                    last = 1;
                    int _rewound = atomicAdd(&split_counters[split_group], -split_parts);
                }
            }
            uint32_t _barrier_vote_7;
            asm volatile("{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, %1, 0;\n\tbarrier.red.or.pred q, %2, 128, p;\n\tselp.u32 %0, 1, 0, q;\n\t}" : "=r"(_barrier_vote_7) : "r"((uint32_t)(last != 0)), "r"(1 + g) : "memory");
            store = (int)_barrier_vote_7;
            if (store != 0) {
                __threadfence();
                for (int other = 0; other < split_parts; other++) {
                    long long base = (long long)(split_slot - split_part + other) * 8704 + (long long)gtid;
                    float m_o0 = partial_ws[base + 8192];
                    float m_o1 = partial_ws[base + 8192 + 128];
                    float d_o0 = partial_ws[base + 8448];
                    float d_o1 = partial_ws[base + 8448 + 128];
                    if (other == 0) {
                        m_state[0] = m_o0;
                        m_state[1] = m_o1;
                        d_state[0] = d_o0;
                        d_state[1] = d_o1;
                        out_acc[0] = partial_ws[base];
                        out_acc[1] = partial_ws[base + 128];
                        out_acc[2] = partial_ws[base + 256];
                        out_acc[3] = partial_ws[base + 384];
                        out_acc[4] = partial_ws[base + 512];
                        out_acc[5] = partial_ws[base + 640];
                        out_acc[6] = partial_ws[base + 768];
                        out_acc[7] = partial_ws[base + 896];
                        out_acc[8] = partial_ws[base + 1024];
                        out_acc[9] = partial_ws[base + 1152];
                        out_acc[10] = partial_ws[base + 1280];
                        out_acc[11] = partial_ws[base + 1408];
                        out_acc[12] = partial_ws[base + 1536];
                        out_acc[13] = partial_ws[base + 1664];
                        out_acc[14] = partial_ws[base + 1792];
                        out_acc[15] = partial_ws[base + 1920];
                        out_acc[16] = partial_ws[base + 2048];
                        out_acc[17] = partial_ws[base + 2176];
                        out_acc[18] = partial_ws[base + 2304];
                        out_acc[19] = partial_ws[base + 2432];
                        out_acc[20] = partial_ws[base + 2560];
                        out_acc[21] = partial_ws[base + 2688];
                        out_acc[22] = partial_ws[base + 2816];
                        out_acc[23] = partial_ws[base + 2944];
                        out_acc[24] = partial_ws[base + 3072];
                        out_acc[25] = partial_ws[base + 3200];
                        out_acc[26] = partial_ws[base + 3328];
                        out_acc[27] = partial_ws[base + 3456];
                        out_acc[28] = partial_ws[base + 3584];
                        out_acc[29] = partial_ws[base + 3712];
                        out_acc[30] = partial_ws[base + 3840];
                        out_acc[31] = partial_ws[base + 3968];
                        out_acc[32] = partial_ws[base + 4096];
                        out_acc[33] = partial_ws[base + 4224];
                        out_acc[34] = partial_ws[base + 4352];
                        out_acc[35] = partial_ws[base + 4480];
                        out_acc[36] = partial_ws[base + 4608];
                        out_acc[37] = partial_ws[base + 4736];
                        out_acc[38] = partial_ws[base + 4864];
                        out_acc[39] = partial_ws[base + 4992];
                        out_acc[40] = partial_ws[base + 5120];
                        out_acc[41] = partial_ws[base + 5248];
                        out_acc[42] = partial_ws[base + 5376];
                        out_acc[43] = partial_ws[base + 5504];
                        out_acc[44] = partial_ws[base + 5632];
                        out_acc[45] = partial_ws[base + 5760];
                        out_acc[46] = partial_ws[base + 5888];
                        out_acc[47] = partial_ws[base + 6016];
                        out_acc[48] = partial_ws[base + 6144];
                        out_acc[49] = partial_ws[base + 6272];
                        out_acc[50] = partial_ws[base + 6400];
                        out_acc[51] = partial_ws[base + 6528];
                        out_acc[52] = partial_ws[base + 6656];
                        out_acc[53] = partial_ws[base + 6784];
                        out_acc[54] = partial_ws[base + 6912];
                        out_acc[55] = partial_ws[base + 7040];
                        out_acc[56] = partial_ws[base + 7168];
                        out_acc[57] = partial_ws[base + 7296];
                        out_acc[58] = partial_ws[base + 7424];
                        out_acc[59] = partial_ws[base + 7552];
                        out_acc[60] = partial_ws[base + 7680];
                        out_acc[61] = partial_ws[base + 7808];
                        out_acc[62] = partial_ws[base + 7936];
                        out_acc[63] = partial_ws[base + 8064];
                    } else {
                        float m_new0 = fmaxf(m_state[0], m_o0);
                        float m_new1 = fmaxf(m_state[1], m_o1);
                        float a0 = approx_exp2(__fmaf_rn(m_state[0], softmax_scale_log2, -(m_new0 * softmax_scale_log2)));
                        float b0 = approx_exp2(__fmaf_rn(m_o0, softmax_scale_log2, -(m_new0 * softmax_scale_log2)));
                        float a1 = approx_exp2(__fmaf_rn(m_state[1], softmax_scale_log2, -(m_new1 * softmax_scale_log2)));
                        float b1 = approx_exp2(__fmaf_rn(m_o1, softmax_scale_log2, -(m_new1 * softmax_scale_log2)));
                        m_state[0] = m_new0;
                        m_state[1] = m_new1;
                        d_state[0] = __fmaf_rn(d_state[0], a0, d_o0 * b0);
                        d_state[1] = __fmaf_rn(d_state[1], a1, d_o1 * b1);
                        float o_other_0 = partial_ws[base];
                        out_acc[0] = __fmaf_rn(out_acc[0], a0, o_other_0 * b0);
                        float o_other_1 = partial_ws[base + 128];
                        out_acc[1] = __fmaf_rn(out_acc[1], a0, o_other_1 * b0);
                        float o_other_2 = partial_ws[base + 256];
                        out_acc[2] = __fmaf_rn(out_acc[2], a1, o_other_2 * b1);
                        float o_other_3 = partial_ws[base + 384];
                        out_acc[3] = __fmaf_rn(out_acc[3], a1, o_other_3 * b1);
                        float o_other_4 = partial_ws[base + 512];
                        out_acc[4] = __fmaf_rn(out_acc[4], a0, o_other_4 * b0);
                        float o_other_5 = partial_ws[base + 640];
                        out_acc[5] = __fmaf_rn(out_acc[5], a0, o_other_5 * b0);
                        float o_other_6 = partial_ws[base + 768];
                        out_acc[6] = __fmaf_rn(out_acc[6], a1, o_other_6 * b1);
                        float o_other_7 = partial_ws[base + 896];
                        out_acc[7] = __fmaf_rn(out_acc[7], a1, o_other_7 * b1);
                        float o_other_8 = partial_ws[base + 1024];
                        out_acc[8] = __fmaf_rn(out_acc[8], a0, o_other_8 * b0);
                        float o_other_9 = partial_ws[base + 1152];
                        out_acc[9] = __fmaf_rn(out_acc[9], a0, o_other_9 * b0);
                        float o_other_10 = partial_ws[base + 1280];
                        out_acc[10] = __fmaf_rn(out_acc[10], a1, o_other_10 * b1);
                        float o_other_11 = partial_ws[base + 1408];
                        out_acc[11] = __fmaf_rn(out_acc[11], a1, o_other_11 * b1);
                        float o_other_12 = partial_ws[base + 1536];
                        out_acc[12] = __fmaf_rn(out_acc[12], a0, o_other_12 * b0);
                        float o_other_13 = partial_ws[base + 1664];
                        out_acc[13] = __fmaf_rn(out_acc[13], a0, o_other_13 * b0);
                        float o_other_14 = partial_ws[base + 1792];
                        out_acc[14] = __fmaf_rn(out_acc[14], a1, o_other_14 * b1);
                        float o_other_15 = partial_ws[base + 1920];
                        out_acc[15] = __fmaf_rn(out_acc[15], a1, o_other_15 * b1);
                        float o_other_16 = partial_ws[base + 2048];
                        out_acc[16] = __fmaf_rn(out_acc[16], a0, o_other_16 * b0);
                        float o_other_17 = partial_ws[base + 2176];
                        out_acc[17] = __fmaf_rn(out_acc[17], a0, o_other_17 * b0);
                        float o_other_18 = partial_ws[base + 2304];
                        out_acc[18] = __fmaf_rn(out_acc[18], a1, o_other_18 * b1);
                        float o_other_19 = partial_ws[base + 2432];
                        out_acc[19] = __fmaf_rn(out_acc[19], a1, o_other_19 * b1);
                        float o_other_20 = partial_ws[base + 2560];
                        out_acc[20] = __fmaf_rn(out_acc[20], a0, o_other_20 * b0);
                        float o_other_21 = partial_ws[base + 2688];
                        out_acc[21] = __fmaf_rn(out_acc[21], a0, o_other_21 * b0);
                        float o_other_22 = partial_ws[base + 2816];
                        out_acc[22] = __fmaf_rn(out_acc[22], a1, o_other_22 * b1);
                        float o_other_23 = partial_ws[base + 2944];
                        out_acc[23] = __fmaf_rn(out_acc[23], a1, o_other_23 * b1);
                        float o_other_24 = partial_ws[base + 3072];
                        out_acc[24] = __fmaf_rn(out_acc[24], a0, o_other_24 * b0);
                        float o_other_25 = partial_ws[base + 3200];
                        out_acc[25] = __fmaf_rn(out_acc[25], a0, o_other_25 * b0);
                        float o_other_26 = partial_ws[base + 3328];
                        out_acc[26] = __fmaf_rn(out_acc[26], a1, o_other_26 * b1);
                        float o_other_27 = partial_ws[base + 3456];
                        out_acc[27] = __fmaf_rn(out_acc[27], a1, o_other_27 * b1);
                        float o_other_28 = partial_ws[base + 3584];
                        out_acc[28] = __fmaf_rn(out_acc[28], a0, o_other_28 * b0);
                        float o_other_29 = partial_ws[base + 3712];
                        out_acc[29] = __fmaf_rn(out_acc[29], a0, o_other_29 * b0);
                        float o_other_30 = partial_ws[base + 3840];
                        out_acc[30] = __fmaf_rn(out_acc[30], a1, o_other_30 * b1);
                        float o_other_31 = partial_ws[base + 3968];
                        out_acc[31] = __fmaf_rn(out_acc[31], a1, o_other_31 * b1);
                        float o_other_32 = partial_ws[base + 4096];
                        out_acc[32] = __fmaf_rn(out_acc[32], a0, o_other_32 * b0);
                        float o_other_33 = partial_ws[base + 4224];
                        out_acc[33] = __fmaf_rn(out_acc[33], a0, o_other_33 * b0);
                        float o_other_34 = partial_ws[base + 4352];
                        out_acc[34] = __fmaf_rn(out_acc[34], a1, o_other_34 * b1);
                        float o_other_35 = partial_ws[base + 4480];
                        out_acc[35] = __fmaf_rn(out_acc[35], a1, o_other_35 * b1);
                        float o_other_36 = partial_ws[base + 4608];
                        out_acc[36] = __fmaf_rn(out_acc[36], a0, o_other_36 * b0);
                        float o_other_37 = partial_ws[base + 4736];
                        out_acc[37] = __fmaf_rn(out_acc[37], a0, o_other_37 * b0);
                        float o_other_38 = partial_ws[base + 4864];
                        out_acc[38] = __fmaf_rn(out_acc[38], a1, o_other_38 * b1);
                        float o_other_39 = partial_ws[base + 4992];
                        out_acc[39] = __fmaf_rn(out_acc[39], a1, o_other_39 * b1);
                        float o_other_40 = partial_ws[base + 5120];
                        out_acc[40] = __fmaf_rn(out_acc[40], a0, o_other_40 * b0);
                        float o_other_41 = partial_ws[base + 5248];
                        out_acc[41] = __fmaf_rn(out_acc[41], a0, o_other_41 * b0);
                        float o_other_42 = partial_ws[base + 5376];
                        out_acc[42] = __fmaf_rn(out_acc[42], a1, o_other_42 * b1);
                        float o_other_43 = partial_ws[base + 5504];
                        out_acc[43] = __fmaf_rn(out_acc[43], a1, o_other_43 * b1);
                        float o_other_44 = partial_ws[base + 5632];
                        out_acc[44] = __fmaf_rn(out_acc[44], a0, o_other_44 * b0);
                        float o_other_45 = partial_ws[base + 5760];
                        out_acc[45] = __fmaf_rn(out_acc[45], a0, o_other_45 * b0);
                        float o_other_46 = partial_ws[base + 5888];
                        out_acc[46] = __fmaf_rn(out_acc[46], a1, o_other_46 * b1);
                        float o_other_47 = partial_ws[base + 6016];
                        out_acc[47] = __fmaf_rn(out_acc[47], a1, o_other_47 * b1);
                        float o_other_48 = partial_ws[base + 6144];
                        out_acc[48] = __fmaf_rn(out_acc[48], a0, o_other_48 * b0);
                        float o_other_49 = partial_ws[base + 6272];
                        out_acc[49] = __fmaf_rn(out_acc[49], a0, o_other_49 * b0);
                        float o_other_50 = partial_ws[base + 6400];
                        out_acc[50] = __fmaf_rn(out_acc[50], a1, o_other_50 * b1);
                        float o_other_51 = partial_ws[base + 6528];
                        out_acc[51] = __fmaf_rn(out_acc[51], a1, o_other_51 * b1);
                        float o_other_52 = partial_ws[base + 6656];
                        out_acc[52] = __fmaf_rn(out_acc[52], a0, o_other_52 * b0);
                        float o_other_53 = partial_ws[base + 6784];
                        out_acc[53] = __fmaf_rn(out_acc[53], a0, o_other_53 * b0);
                        float o_other_54 = partial_ws[base + 6912];
                        out_acc[54] = __fmaf_rn(out_acc[54], a1, o_other_54 * b1);
                        float o_other_55 = partial_ws[base + 7040];
                        out_acc[55] = __fmaf_rn(out_acc[55], a1, o_other_55 * b1);
                        float o_other_56 = partial_ws[base + 7168];
                        out_acc[56] = __fmaf_rn(out_acc[56], a0, o_other_56 * b0);
                        float o_other_57 = partial_ws[base + 7296];
                        out_acc[57] = __fmaf_rn(out_acc[57], a0, o_other_57 * b0);
                        float o_other_58 = partial_ws[base + 7424];
                        out_acc[58] = __fmaf_rn(out_acc[58], a1, o_other_58 * b1);
                        float o_other_59 = partial_ws[base + 7552];
                        out_acc[59] = __fmaf_rn(out_acc[59], a1, o_other_59 * b1);
                        float o_other_60 = partial_ws[base + 7680];
                        out_acc[60] = __fmaf_rn(out_acc[60], a0, o_other_60 * b0);
                        float o_other_61 = partial_ws[base + 7808];
                        out_acc[61] = __fmaf_rn(out_acc[61], a0, o_other_61 * b0);
                        float o_other_62 = partial_ws[base + 7936];
                        out_acc[62] = __fmaf_rn(out_acc[62], a1, o_other_62 * b1);
                        float o_other_63 = partial_ws[base + 8064];
                        out_acc[63] = __fmaf_rn(out_acc[63], a1, o_other_63 * b1);
                    }
                }
            }
        }
        if (store != 0) {
            float peer_d_0 = __shfl_xor_sync(0xFFFFFFFF, d_state[0], 2);
            d_state[0] = d_state[0] + peer_d_0;
            float peer_d2_0 = __shfl_xor_sync(0xFFFFFFFF, d_state[0], 1);
            d_state[0] = d_state[0] + peer_d2_0;
            float peer_d_1 = __shfl_xor_sync(0xFFFFFFFF, d_state[1], 2);
            d_state[1] = d_state[1] + peer_d_1;
            float peer_d2_1 = __shfl_xor_sync(0xFFFFFFFF, d_state[1], 1);
            d_state[1] = d_state[1] + peer_d2_1;
            float rcp0 = approx_rcp(d_state[0]);
            float rcp1 = approx_rcp(d_state[1]);
            {
                float2 _pair_scale_even2_8 = make_float2(rcp0, rcp0);
                float2 _pair_scale_odd2_8 = make_float2(rcp1, rcp1);
                float2* _pair_scale_src2_8 = reinterpret_cast<float2*>(&out_acc[0]);
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
            }
            unsigned int o_pack_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 0], out_acc[_lp*2+1 + 0]));
                o_pack_0[_lp] = *(uint32_t*)&_h2;
            }
            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(q_tile + (unsigned int)((lane / 16 / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_0[3]))
                : "memory");
            unsigned int o_pack_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 8], out_acc[_lp*2+1 + 8]));
                o_pack_1[_lp] = *(uint32_t*)&_h2;
            }
            uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(q_tile + (unsigned int)(((2 + lane / 16) / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + ((2 + lane / 16) % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_1[3]))
                : "memory");
            unsigned int o_pack_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 16], out_acc[_lp*2+1 + 16]));
                o_pack_2[_lp] = *(uint32_t*)&_h2;
            }
            uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(q_tile + (unsigned int)(((4 + lane / 16) / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + ((4 + lane / 16) % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[3]))
                : "memory");
            unsigned int o_pack_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 24], out_acc[_lp*2+1 + 24]));
                o_pack_3[_lp] = *(uint32_t*)&_h2;
            }
            uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(q_tile + (unsigned int)(((6 + lane / 16) / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + ((6 + lane / 16) % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[3]))
                : "memory");
            unsigned int o_pack_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 32], out_acc[_lp*2+1 + 32]));
                o_pack_4[_lp] = *(uint32_t*)&_h2;
            }
            uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(q_tile + (unsigned int)(((8 + lane / 16) / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + ((8 + lane / 16) % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[3]))
                : "memory");
            unsigned int o_pack_5[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 40], out_acc[_lp*2+1 + 40]));
                o_pack_5[_lp] = *(uint32_t*)&_h2;
            }
            uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(q_tile + (unsigned int)(((10 + lane / 16) / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + ((10 + lane / 16) % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[3]))
                : "memory");
            unsigned int o_pack_6[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 48], out_acc[_lp*2+1 + 48]));
                o_pack_6[_lp] = *(uint32_t*)&_h2;
            }
            uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(q_tile + (unsigned int)(((12 + lane / 16) / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + ((12 + lane / 16) % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[3]))
                : "memory");
            unsigned int o_pack_7[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 56], out_acc[_lp*2+1 + 56]));
                o_pack_7[_lp] = *(uint32_t*)&_h2;
            }
            uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(q_tile + (unsigned int)(((14 + lane / 16) / 8 * 512 + (warp % 4 * 16 + lane % 16) * 8 + ((14 + lane / 16) % 8 * 16 ^ (warp % 4 * 16 + lane % 16 & 7) << 4) / 16) * 16));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[3]))
                : "memory");
            __syncwarp();
            unsigned int o_vec[4];
            int row_out_0 = (q_base + (warp % 4 * 16 + lane / 8)) * num_heads + head;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_tile + (unsigned int)((lane % 8 / 8 * 512 + (warp % 4 * 16 + lane / 8) * 8 + (lane % 8 % 8 * 16 ^ (warp % 4 * 16 + lane / 8 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(out + ((long long)row_out_0 * 128 + (long long)(lane % 8 * 8)))[0] = reinterpret_cast<int4*>(o_vec)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_tile + (unsigned int)(((lane % 8 + 8) / 8 * 512 + (warp % 4 * 16 + lane / 8) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp % 4 * 16 + lane / 8 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(out + ((long long)row_out_0 * 128 + (long long)(lane % 8 * 8 + 64)))[0] = reinterpret_cast<int4*>(o_vec)[0];
            int row_out_1 = (q_base + (warp % 4 * 16 + lane / 8 + 4)) * num_heads + head;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_tile + (unsigned int)((lane % 8 / 8 * 512 + (warp % 4 * 16 + lane / 8 + 4) * 8 + (lane % 8 % 8 * 16 ^ (warp % 4 * 16 + lane / 8 + 4 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(out + ((long long)row_out_1 * 128 + (long long)(lane % 8 * 8)))[0] = reinterpret_cast<int4*>(o_vec)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_tile + (unsigned int)(((lane % 8 + 8) / 8 * 512 + (warp % 4 * 16 + lane / 8 + 4) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp % 4 * 16 + lane / 8 + 4 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(out + ((long long)row_out_1 * 128 + (long long)(lane % 8 * 8 + 64)))[0] = reinterpret_cast<int4*>(o_vec)[0];
            int row_out_2 = (q_base + (warp % 4 * 16 + lane / 8 + 8)) * num_heads + head;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_tile + (unsigned int)((lane % 8 / 8 * 512 + (warp % 4 * 16 + lane / 8 + 8) * 8 + (lane % 8 % 8 * 16 ^ (warp % 4 * 16 + lane / 8 + 8 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(out + ((long long)row_out_2 * 128 + (long long)(lane % 8 * 8)))[0] = reinterpret_cast<int4*>(o_vec)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_tile + (unsigned int)(((lane % 8 + 8) / 8 * 512 + (warp % 4 * 16 + lane / 8 + 8) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp % 4 * 16 + lane / 8 + 8 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(out + ((long long)row_out_2 * 128 + (long long)(lane % 8 * 8 + 64)))[0] = reinterpret_cast<int4*>(o_vec)[0];
            int row_out_3 = (q_base + (warp % 4 * 16 + lane / 8 + 12)) * num_heads + head;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_tile + (unsigned int)((lane % 8 / 8 * 512 + (warp % 4 * 16 + lane / 8 + 12) * 8 + (lane % 8 % 8 * 16 ^ (warp % 4 * 16 + lane / 8 + 12 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(out + ((long long)row_out_3 * 128 + (long long)(lane % 8 * 8)))[0] = reinterpret_cast<int4*>(o_vec)[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                : "r"(q_tile + (unsigned int)(((lane % 8 + 8) / 8 * 512 + (warp % 4 * 16 + lane / 8 + 12) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp % 4 * 16 + lane / 8 + 12 & 7) << 4) / 16) * 16)));
            reinterpret_cast<int4*>(out + ((long long)row_out_3 * 128 + (long long)(lane % 8 * 8 + 64)))[0] = reinterpret_cast<int4*>(o_vec)[0];
            if (write_lse != 0) {
                if (lane % 4 == 0) {
                    int lse_row_0 = q_base + warp % 4 * 16 + lane / 4;
                    float _log2_17;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_17) : "f"(d_state[0]));
                    float lse_value_0 = (m_state[0] * softmax_scale_log2 + _log2_17) * 0.6931472f;
                    *(reinterpret_cast<float*>(lse + (lse_row_0 * num_heads + head)) + (0)) = lse_value_0;
                    int lse_row_1 = q_base + warp % 4 * 16 + lane / 4 + 8;
                    float _log2_18;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_18) : "f"(d_state[1]));
                    float lse_value_1 = (m_state[1] * softmax_scale_log2 + _log2_18) * 0.6931472f;
                    *(reinterpret_cast<float*>(lse + (lse_row_1 * num_heads + head)) + (0)) = lse_value_1;
                }
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync %0, 128;" :: "r"(1 + g) : "memory");
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
