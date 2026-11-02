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
#define NUM_KV_STAGES 2
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 32768
#define SMEM_Q_SMEM_STRIDE 32768
#define SMEM_K_SMEM_OFF 33792
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_V_SMEM_OFF 66560
#define SMEM_V_SMEM_STAGE_BYTES 16384
#define SMEM_V_SMEM_STRIDE 16384
#define SMEM_SPLIT_FLAG_OFF 99328
#define SMEM_SPLIT_FLAG_STAGE_BYTES 4
#define SMEM_SPLIT_FLAG_STRIDE 4
#define SMEM_TOTAL 99456
#define THREADS 256
#define NUM_HEADS_CONST 56
#define HEAD_MAJOR 1
#define KV_UNROLL 1
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



#if __CUDA_ARCH__ == 900
#else
// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)
#endif



#if __CUDA_ARCH__ == 900
__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
#else

#endif

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_dense_attention_1ec8efce8d0b0be5f54a(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap V_map, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_ws, int* __restrict__ split_counters, int tokens, int num_q_tiles, int num_kv_tiles, unsigned int total_work, unsigned int split_base, int split_k, int split_tiles, float softmax_scale_log2, unsigned int q_scale_bf16x2)
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
    #define v_full_addr (mbar_base + 24)
    #define k_empty_addr (mbar_base + 40)
    #define v_empty_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_smem_addr = smem + 33792;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int v_smem_addr = smem + 66560;
    int* split_flag = reinterpret_cast<int*>(smem_raw + 99328);
    const int split_flag_addr = smem + 99328;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv' ---
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 40, 256);
            mbarrier_init(smem + 48, 256);
            // v_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 56, 256);
            mbarrier_init(smem + 64, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int last_kv_tile = num_kv_tiles - 1;
    int tail_rows = tokens - last_kv_tile * 64;
    unsigned int kv_stage = 0;
    unsigned int kv_phase = 0;
    if (warp >= 4) {
        asm volatile("barrier.arrive 1, 256;" ::: "memory");
    }
    unsigned int _phase_q_full_0 = 0;
    #pragma unroll 1
    for (unsigned int work_id = bid; work_id < total_work; work_id += num_bids) {
        unsigned int item_id = work_id;
        int split_part = 0;
        int split_slot = -1;
        int kv_begin = 0;
        int item_last_tile = last_kv_tile;
        if (work_id >= split_base) {
            unsigned int tail_id = work_id - split_base;
            item_id = split_base + tail_id / (unsigned int)split_k;
            split_part = tail_id % (unsigned int)split_k;
            split_slot = tail_id;
            kv_begin = split_part * split_tiles;
            item_last_tile = kv_begin + split_tiles - 1;
            if (item_last_tile > last_kv_tile) {
                item_last_tile = last_kv_tile;
            }
        }
        int head_idx = 0;
        int q_tile_idx = 0;
        {
            head_idx = item_id / (unsigned int)num_q_tiles;
            q_tile_idx = item_id % (unsigned int)num_q_tiles;
        }
        int q_tile_base = q_tile_idx * 128;
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 32768);
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(q_smem_addr), "l"((&Q_map)), "r"(0), "r"(q_tile_base), "r"(head_idx), "r"(0),
                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        if (item_last_tile >= kv_begin) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + (kv_stage % 2) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(k_smem_addr + kv_stage % 2 * 16384), "l"((&K_map)), "r"(0), "r"(kv_begin * 64), "r"(head_idx), "r"(0),
                           "r"(k_full_addr + (kv_stage % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(v_full_addr + (kv_stage % 2) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(v_smem_addr + kv_stage % 2 * 16384), "l"((&V_map)), "r"(0), "r"(kv_begin * 64), "r"(head_idx), "r"(0),
                           "r"(v_full_addr + (kv_stage % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        if (item_last_tile >= kv_begin + 1) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(k_smem_addr + (kv_stage + 1) % 2 * 16384), "l"((&K_map)), "r"(0), "r"((kv_begin + 1) * 64), "r"(head_idx), "r"(0),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(v_smem_addr + (kv_stage + 1) % 2 * 16384), "l"((&V_map)), "r"(0), "r"((kv_begin + 1) * 64), "r"(head_idx), "r"(0),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
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
        m_state[1] = -CUDART_INF_F;
        d_state[0] = 1.0f;
        d_state[1] = 1.0f;
        mbarrier_wait(q_full_addr, _phase_q_full_0);
        _phase_q_full_0 ^= 1;
        unsigned int vec[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16)));
        uint32_t _bf16x2_mul_0;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(vec[0]), "r"(q_scale_bf16x2));
        vec[0] = _bf16x2_mul_0;
        uint32_t _bf16x2_mul_1;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(vec[1]), "r"(q_scale_bf16x2));
        vec[1] = _bf16x2_mul_1;
        uint32_t _bf16x2_mul_2;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(vec[2]), "r"(q_scale_bf16x2));
        vec[2] = _bf16x2_mul_2;
        uint32_t _bf16x2_mul_3;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(vec[3]), "r"(q_scale_bf16x2));
        vec[3] = _bf16x2_mul_3;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16)), "r"(vec[0]), "r"(vec[1]), "r"(vec[2]), "r"(vec[3]));
        unsigned int vec_0[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_0[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 4096)));
        uint32_t _bf16x2_mul_4;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(vec_0[0]), "r"(q_scale_bf16x2));
        vec_0[0] = _bf16x2_mul_4;
        uint32_t _bf16x2_mul_5;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(vec_0[1]), "r"(q_scale_bf16x2));
        vec_0[1] = _bf16x2_mul_5;
        uint32_t _bf16x2_mul_6;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(vec_0[2]), "r"(q_scale_bf16x2));
        vec_0[2] = _bf16x2_mul_6;
        uint32_t _bf16x2_mul_7;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(vec_0[3]), "r"(q_scale_bf16x2));
        vec_0[3] = _bf16x2_mul_7;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 4096)), "r"(vec_0[0]), "r"(vec_0[1]), "r"(vec_0[2]), "r"(vec_0[3]));
        unsigned int vec_1[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_1[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 8192)));
        uint32_t _bf16x2_mul_8;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(vec_1[0]), "r"(q_scale_bf16x2));
        vec_1[0] = _bf16x2_mul_8;
        uint32_t _bf16x2_mul_9;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(vec_1[1]), "r"(q_scale_bf16x2));
        vec_1[1] = _bf16x2_mul_9;
        uint32_t _bf16x2_mul_10;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(vec_1[2]), "r"(q_scale_bf16x2));
        vec_1[2] = _bf16x2_mul_10;
        uint32_t _bf16x2_mul_11;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(vec_1[3]), "r"(q_scale_bf16x2));
        vec_1[3] = _bf16x2_mul_11;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 8192)), "r"(vec_1[0]), "r"(vec_1[1]), "r"(vec_1[2]), "r"(vec_1[3]));
        unsigned int vec_2[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_2[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 12288)));
        uint32_t _bf16x2_mul_12;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(vec_2[0]), "r"(q_scale_bf16x2));
        vec_2[0] = _bf16x2_mul_12;
        uint32_t _bf16x2_mul_13;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(vec_2[1]), "r"(q_scale_bf16x2));
        vec_2[1] = _bf16x2_mul_13;
        uint32_t _bf16x2_mul_14;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(vec_2[2]), "r"(q_scale_bf16x2));
        vec_2[2] = _bf16x2_mul_14;
        uint32_t _bf16x2_mul_15;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(vec_2[3]), "r"(q_scale_bf16x2));
        vec_2[3] = _bf16x2_mul_15;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 12288)), "r"(vec_2[0]), "r"(vec_2[1]), "r"(vec_2[2]), "r"(vec_2[3]));
        unsigned int vec_3[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_3[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 16384)));
        uint32_t _bf16x2_mul_16;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(vec_3[0]), "r"(q_scale_bf16x2));
        vec_3[0] = _bf16x2_mul_16;
        uint32_t _bf16x2_mul_17;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(vec_3[1]), "r"(q_scale_bf16x2));
        vec_3[1] = _bf16x2_mul_17;
        uint32_t _bf16x2_mul_18;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(vec_3[2]), "r"(q_scale_bf16x2));
        vec_3[2] = _bf16x2_mul_18;
        uint32_t _bf16x2_mul_19;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(vec_3[3]), "r"(q_scale_bf16x2));
        vec_3[3] = _bf16x2_mul_19;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 16384)), "r"(vec_3[0]), "r"(vec_3[1]), "r"(vec_3[2]), "r"(vec_3[3]));
        unsigned int vec_4[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_4[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 20480)));
        uint32_t _bf16x2_mul_20;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(vec_4[0]), "r"(q_scale_bf16x2));
        vec_4[0] = _bf16x2_mul_20;
        uint32_t _bf16x2_mul_21;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(vec_4[1]), "r"(q_scale_bf16x2));
        vec_4[1] = _bf16x2_mul_21;
        uint32_t _bf16x2_mul_22;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(vec_4[2]), "r"(q_scale_bf16x2));
        vec_4[2] = _bf16x2_mul_22;
        uint32_t _bf16x2_mul_23;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(vec_4[3]), "r"(q_scale_bf16x2));
        vec_4[3] = _bf16x2_mul_23;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 20480)), "r"(vec_4[0]), "r"(vec_4[1]), "r"(vec_4[2]), "r"(vec_4[3]));
        unsigned int vec_5[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_5[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 24576)));
        uint32_t _bf16x2_mul_24;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(vec_5[0]), "r"(q_scale_bf16x2));
        vec_5[0] = _bf16x2_mul_24;
        uint32_t _bf16x2_mul_25;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(vec_5[1]), "r"(q_scale_bf16x2));
        vec_5[1] = _bf16x2_mul_25;
        uint32_t _bf16x2_mul_26;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(vec_5[2]), "r"(q_scale_bf16x2));
        vec_5[2] = _bf16x2_mul_26;
        uint32_t _bf16x2_mul_27;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(vec_5[3]), "r"(q_scale_bf16x2));
        vec_5[3] = _bf16x2_mul_27;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 24576)), "r"(vec_5[0]), "r"(vec_5[1]), "r"(vec_5[2]), "r"(vec_5[3]));
        unsigned int vec_6[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vec_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vec_6[(0) + 3]))
            : "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 28672)));
        uint32_t _bf16x2_mul_28;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(vec_6[0]), "r"(q_scale_bf16x2));
        vec_6[0] = _bf16x2_mul_28;
        uint32_t _bf16x2_mul_29;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(vec_6[1]), "r"(q_scale_bf16x2));
        vec_6[1] = _bf16x2_mul_29;
        uint32_t _bf16x2_mul_30;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(vec_6[2]), "r"(q_scale_bf16x2));
        vec_6[2] = _bf16x2_mul_30;
        uint32_t _bf16x2_mul_31;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(vec_6[3]), "r"(q_scale_bf16x2));
        vec_6[3] = _bf16x2_mul_31;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_smem_addr + (unsigned int)((warp * 32 + lane) * 16 + 28672)), "r"(vec_6[0]), "r"(vec_6[1]), "r"(vec_6[2]), "r"(vec_6[3]));
        __syncthreads();
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[0]), "=r"(q_frags[1]), "=r"(q_frags[2]), "=r"(q_frags[3])
            : "r"(q_smem_addr + (unsigned int)((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(q_smem_addr + (unsigned int)((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[8]), "=r"(q_frags[9]), "=r"(q_frags[10]), "=r"(q_frags[11])
            : "r"(q_smem_addr + (unsigned int)((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[12]), "=r"(q_frags[13]), "=r"(q_frags[14]), "=r"(q_frags[15])
            : "r"(q_smem_addr + (unsigned int)((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[16]), "=r"(q_frags[17]), "=r"(q_frags[18]), "=r"(q_frags[19])
            : "r"(q_smem_addr + (unsigned int)(((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 1024) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[20]), "=r"(q_frags[21]), "=r"(q_frags[22]), "=r"(q_frags[23])
            : "r"(q_smem_addr + (unsigned int)(((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 1024 ^ 2) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[24]), "=r"(q_frags[25]), "=r"(q_frags[26]), "=r"(q_frags[27])
            : "r"(q_smem_addr + (unsigned int)(((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 1024 ^ 2 ^ 6) * 16))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[28]), "=r"(q_frags[29]), "=r"(q_frags[30]), "=r"(q_frags[31])
            : "r"(q_smem_addr + (unsigned int)(((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 1024 ^ 2 ^ 6 ^ 2) * 16))
            : "memory");
        constexpr int kv_tile_idx_unroll = KV_UNROLL;
        #pragma unroll kv_tile_idx_unroll
        for (int kv_tile_idx = kv_begin; kv_tile_idx < item_last_tile + 1; kv_tile_idx++) {
            mbarrier_wait(k_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[0]), "=f"(qk_acc[1]), "=f"(qk_acc[2]), "=f"(qk_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[4]), "=f"(qk_acc[(4) + 1]), "=f"(qk_acc[(4) + 2]), "=f"(qk_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[8]), "=f"(qk_acc[(8) + 1]), "=f"(qk_acc[(8) + 2]), "=f"(qk_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[12]), "=f"(qk_acc[(12) + 1]), "=f"(qk_acc[(12) + 2]), "=f"(qk_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[16]), "=f"(qk_acc[(16) + 1]), "=f"(qk_acc[(16) + 2]), "=f"(qk_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[20]), "=f"(qk_acc[(20) + 1]), "=f"(qk_acc[(20) + 2]), "=f"(qk_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[24]), "=f"(qk_acc[(24) + 1]), "=f"(qk_acc[(24) + 2]), "=f"(qk_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(qk_acc[28]), "=f"(qk_acc[(28) + 1]), "=f"(qk_acc[(28) + 2]), "=f"(qk_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[16]), "r"(q_frags[(16) + 1]), "r"(q_frags[(16) + 2]), "r"(q_frags[(16) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[20]), "r"(q_frags[(20) + 1]), "r"(q_frags[(20) + 2]), "r"(q_frags[(20) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[24]), "r"(q_frags[(24) + 1]), "r"(q_frags[(24) + 2]), "r"(q_frags[(24) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[0]), "+f"(qk_acc[1]), "+f"(qk_acc[2]), "+f"(qk_acc[3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[4]), "+f"(qk_acc[(4) + 1]), "+f"(qk_acc[(4) + 2]), "+f"(qk_acc[(4) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[8]), "+f"(qk_acc[(8) + 1]), "+f"(qk_acc[(8) + 2]), "+f"(qk_acc[(8) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[12]), "+f"(qk_acc[(12) + 1]), "+f"(qk_acc[(12) + 2]), "+f"(qk_acc[(12) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[16]), "+f"(qk_acc[(16) + 1]), "+f"(qk_acc[(16) + 2]), "+f"(qk_acc[(16) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[20]), "+f"(qk_acc[(20) + 1]), "+f"(qk_acc[(20) + 2]), "+f"(qk_acc[(20) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(k_smem_addr + kv_stage * 16384 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 512 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) + 512 - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128 + 128 ^ 6) - 512 + 128 + 128 + 128 + 128 ^ 2) - 512 + 128 + 128 + 128) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[24]), "+f"(qk_acc[(24) + 1]), "+f"(qk_acc[(24) + 2]), "+f"(qk_acc[(24) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[0]), "r"(k_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(qk_acc[28]), "+f"(qk_acc[(28) + 1]), "+f"(qk_acc[(28) + 2]), "+f"(qk_acc[(28) + 3])
                : "r"(q_frags[28]), "r"(q_frags[(28) + 1]), "r"(q_frags[(28) + 2]), "r"(q_frags[(28) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
            if (last_kv_tile == kv_tile_idx && tail_rows < 64) {
                if (kv_tile_idx * 64 + 2 * (lane % 4) >= tokens) {
                    qk_acc[0] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[1] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 2 * (lane % 4) >= tokens) {
                    qk_acc[2] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[3] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[4] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[5] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[6] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[7] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 16 + 2 * (lane % 4) >= tokens) {
                    qk_acc[8] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 16 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[9] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 16 + 2 * (lane % 4) >= tokens) {
                    qk_acc[10] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 16 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[11] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 16 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[12] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 16 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[13] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 16 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[14] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 16 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[15] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 32 + 2 * (lane % 4) >= tokens) {
                    qk_acc[16] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 32 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[17] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 32 + 2 * (lane % 4) >= tokens) {
                    qk_acc[18] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 32 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[19] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 32 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[20] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 32 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[21] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 32 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[22] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 32 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[23] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 48 + 2 * (lane % 4) >= tokens) {
                    qk_acc[24] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 48 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[25] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 48 + 2 * (lane % 4) >= tokens) {
                    qk_acc[26] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 48 + 2 * (lane % 4) + 1 >= tokens) {
                    qk_acc[27] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 48 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[28] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 48 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[29] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 48 + 2 * (lane % 4) + 8 >= tokens) {
                    qk_acc[30] = -CUDART_INF_F;
                }
                if (kv_tile_idx * 64 + 48 + 2 * (lane % 4) + 8 + 1 >= tokens) {
                    qk_acc[31] = -CUDART_INF_F;
                }
            }
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
#if __CUDA_ARCH__ == 900
                float _mul_operand_0 = o_scale[0];
                out_acc[0] = mul_f32_rn_noftz(out_acc[0], _mul_operand_0);
                out_acc[1] = mul_f32_rn_noftz(out_acc[1], _mul_operand_0);
                float _mul_operand_1 = o_scale[1];
                out_acc[2] = mul_f32_rn_noftz(out_acc[2], _mul_operand_1);
                out_acc[3] = mul_f32_rn_noftz(out_acc[3], _mul_operand_1);
                float _mul_operand_2 = o_scale[0];
                out_acc[4] = mul_f32_rn_noftz(out_acc[4], _mul_operand_2);
                out_acc[5] = mul_f32_rn_noftz(out_acc[5], _mul_operand_2);
                float _mul_operand_3 = o_scale[1];
                out_acc[6] = mul_f32_rn_noftz(out_acc[6], _mul_operand_3);
                out_acc[7] = mul_f32_rn_noftz(out_acc[7], _mul_operand_3);
                float _mul_operand_4 = o_scale[0];
                out_acc[8] = mul_f32_rn_noftz(out_acc[8], _mul_operand_4);
                out_acc[9] = mul_f32_rn_noftz(out_acc[9], _mul_operand_4);
                float _mul_operand_5 = o_scale[1];
                out_acc[10] = mul_f32_rn_noftz(out_acc[10], _mul_operand_5);
                out_acc[11] = mul_f32_rn_noftz(out_acc[11], _mul_operand_5);
                float _mul_operand_6 = o_scale[0];
                out_acc[12] = mul_f32_rn_noftz(out_acc[12], _mul_operand_6);
                out_acc[13] = mul_f32_rn_noftz(out_acc[13], _mul_operand_6);
                float _mul_operand_7 = o_scale[1];
                out_acc[14] = mul_f32_rn_noftz(out_acc[14], _mul_operand_7);
                out_acc[15] = mul_f32_rn_noftz(out_acc[15], _mul_operand_7);
                float _mul_operand_8 = o_scale[0];
                out_acc[16] = mul_f32_rn_noftz(out_acc[16], _mul_operand_8);
                out_acc[17] = mul_f32_rn_noftz(out_acc[17], _mul_operand_8);
                float _mul_operand_9 = o_scale[1];
                out_acc[18] = mul_f32_rn_noftz(out_acc[18], _mul_operand_9);
                out_acc[19] = mul_f32_rn_noftz(out_acc[19], _mul_operand_9);
                float _mul_operand_10 = o_scale[0];
                out_acc[20] = mul_f32_rn_noftz(out_acc[20], _mul_operand_10);
                out_acc[21] = mul_f32_rn_noftz(out_acc[21], _mul_operand_10);
                float _mul_operand_11 = o_scale[1];
                out_acc[22] = mul_f32_rn_noftz(out_acc[22], _mul_operand_11);
                out_acc[23] = mul_f32_rn_noftz(out_acc[23], _mul_operand_11);
                float _mul_operand_12 = o_scale[0];
                out_acc[24] = mul_f32_rn_noftz(out_acc[24], _mul_operand_12);
                out_acc[25] = mul_f32_rn_noftz(out_acc[25], _mul_operand_12);
                float _mul_operand_13 = o_scale[1];
                out_acc[26] = mul_f32_rn_noftz(out_acc[26], _mul_operand_13);
                out_acc[27] = mul_f32_rn_noftz(out_acc[27], _mul_operand_13);
                float _mul_operand_14 = o_scale[0];
                out_acc[28] = mul_f32_rn_noftz(out_acc[28], _mul_operand_14);
                out_acc[29] = mul_f32_rn_noftz(out_acc[29], _mul_operand_14);
                float _mul_operand_15 = o_scale[1];
                out_acc[30] = mul_f32_rn_noftz(out_acc[30], _mul_operand_15);
                out_acc[31] = mul_f32_rn_noftz(out_acc[31], _mul_operand_15);
                float _mul_operand_16 = o_scale[0];
                out_acc[32] = mul_f32_rn_noftz(out_acc[32], _mul_operand_16);
                out_acc[33] = mul_f32_rn_noftz(out_acc[33], _mul_operand_16);
                float _mul_operand_17 = o_scale[1];
                out_acc[34] = mul_f32_rn_noftz(out_acc[34], _mul_operand_17);
                out_acc[35] = mul_f32_rn_noftz(out_acc[35], _mul_operand_17);
                float _mul_operand_18 = o_scale[0];
                out_acc[36] = mul_f32_rn_noftz(out_acc[36], _mul_operand_18);
                out_acc[37] = mul_f32_rn_noftz(out_acc[37], _mul_operand_18);
                float _mul_operand_19 = o_scale[1];
                out_acc[38] = mul_f32_rn_noftz(out_acc[38], _mul_operand_19);
                out_acc[39] = mul_f32_rn_noftz(out_acc[39], _mul_operand_19);
                float _mul_operand_20 = o_scale[0];
                out_acc[40] = mul_f32_rn_noftz(out_acc[40], _mul_operand_20);
                out_acc[41] = mul_f32_rn_noftz(out_acc[41], _mul_operand_20);
                float _mul_operand_21 = o_scale[1];
                out_acc[42] = mul_f32_rn_noftz(out_acc[42], _mul_operand_21);
                out_acc[43] = mul_f32_rn_noftz(out_acc[43], _mul_operand_21);
                float _mul_operand_22 = o_scale[0];
                out_acc[44] = mul_f32_rn_noftz(out_acc[44], _mul_operand_22);
                out_acc[45] = mul_f32_rn_noftz(out_acc[45], _mul_operand_22);
                float _mul_operand_23 = o_scale[1];
                out_acc[46] = mul_f32_rn_noftz(out_acc[46], _mul_operand_23);
                out_acc[47] = mul_f32_rn_noftz(out_acc[47], _mul_operand_23);
                float _mul_operand_24 = o_scale[0];
                out_acc[48] = mul_f32_rn_noftz(out_acc[48], _mul_operand_24);
                out_acc[49] = mul_f32_rn_noftz(out_acc[49], _mul_operand_24);
                float _mul_operand_25 = o_scale[1];
                out_acc[50] = mul_f32_rn_noftz(out_acc[50], _mul_operand_25);
                out_acc[51] = mul_f32_rn_noftz(out_acc[51], _mul_operand_25);
                float _mul_operand_26 = o_scale[0];
                out_acc[52] = mul_f32_rn_noftz(out_acc[52], _mul_operand_26);
                out_acc[53] = mul_f32_rn_noftz(out_acc[53], _mul_operand_26);
                float _mul_operand_27 = o_scale[1];
                out_acc[54] = mul_f32_rn_noftz(out_acc[54], _mul_operand_27);
                out_acc[55] = mul_f32_rn_noftz(out_acc[55], _mul_operand_27);
                float _mul_operand_28 = o_scale[0];
                out_acc[56] = mul_f32_rn_noftz(out_acc[56], _mul_operand_28);
                out_acc[57] = mul_f32_rn_noftz(out_acc[57], _mul_operand_28);
                float _mul_operand_29 = o_scale[1];
                out_acc[58] = mul_f32_rn_noftz(out_acc[58], _mul_operand_29);
                out_acc[59] = mul_f32_rn_noftz(out_acc[59], _mul_operand_29);
                float _mul_operand_30 = o_scale[0];
                out_acc[60] = mul_f32_rn_noftz(out_acc[60], _mul_operand_30);
                out_acc[61] = mul_f32_rn_noftz(out_acc[61], _mul_operand_30);
                float _mul_operand_31 = o_scale[1];
                out_acc[62] = mul_f32_rn_noftz(out_acc[62], _mul_operand_31);
                out_acc[63] = mul_f32_rn_noftz(out_acc[63], _mul_operand_31);
#else
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
#endif
            }
            float row0_shift = m_state[0] * softmax_scale_log2;
            float row1_shift = m_state[1] * softmax_scale_log2;
            float e0 = approx_exp2(__fmaf_rn(qk_acc[0], softmax_scale_log2, -row0_shift));
            float e1 = approx_exp2(__fmaf_rn(qk_acc[1], softmax_scale_log2, -row0_shift));
            qk_acc[0] = e0;
            qk_acc[1] = e1;
            float row_sum = e0 + e1;
            float e0_0 = approx_exp2(__fmaf_rn(qk_acc[2], softmax_scale_log2, -row1_shift));
            float e1_1 = approx_exp2(__fmaf_rn(qk_acc[3], softmax_scale_log2, -row1_shift));
            qk_acc[2] = e0_0;
            qk_acc[3] = e1_1;
            float row_sum_2 = e0_0 + e1_1;
            float e0_3 = approx_exp2(__fmaf_rn(qk_acc[4], softmax_scale_log2, -row0_shift));
            float e1_4 = approx_exp2(__fmaf_rn(qk_acc[5], softmax_scale_log2, -row0_shift));
            qk_acc[4] = e0_3;
            qk_acc[5] = e1_4;
            float e0_5 = approx_exp2(__fmaf_rn(qk_acc[6], softmax_scale_log2, -row1_shift));
            float e1_6 = approx_exp2(__fmaf_rn(qk_acc[7], softmax_scale_log2, -row1_shift));
            qk_acc[6] = e0_5;
            qk_acc[7] = e1_6;
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_acc[_lp*2 + 0], qk_acc[_lp*2+1 + 0]));
                p_frag[_lp] = *(uint32_t*)&_bf2;
            }
            d_state[0] = __fmaf_rn(d_state[0], o_scale[0], row_sum + e0_3 + e1_4);
            d_state[1] = __fmaf_rn(d_state[1], o_scale[1], row_sum_2 + e0_5 + e1_6);
            float e0_7 = approx_exp2(__fmaf_rn(qk_acc[8], softmax_scale_log2, -row0_shift));
            float e1_8 = approx_exp2(__fmaf_rn(qk_acc[9], softmax_scale_log2, -row0_shift));
            qk_acc[8] = e0_7;
            qk_acc[9] = e1_8;
            float row_sum_9 = e0_7 + e1_8;
            float e0_10 = approx_exp2(__fmaf_rn(qk_acc[10], softmax_scale_log2, -row1_shift));
            float e1_11 = approx_exp2(__fmaf_rn(qk_acc[11], softmax_scale_log2, -row1_shift));
            qk_acc[10] = e0_10;
            qk_acc[11] = e1_11;
            float row_sum_12 = e0_10 + e1_11;
            float e0_13 = approx_exp2(__fmaf_rn(qk_acc[12], softmax_scale_log2, -row0_shift));
            float e1_14 = approx_exp2(__fmaf_rn(qk_acc[13], softmax_scale_log2, -row0_shift));
            qk_acc[12] = e0_13;
            qk_acc[13] = e1_14;
            float e0_15 = approx_exp2(__fmaf_rn(qk_acc[14], softmax_scale_log2, -row1_shift));
            float e1_16 = approx_exp2(__fmaf_rn(qk_acc[15], softmax_scale_log2, -row1_shift));
            qk_acc[14] = e0_15;
            qk_acc[15] = e1_16;
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_acc[_lp*2 + 8], qk_acc[_lp*2+1 + 8]));
                p_frag[_lp + 4] = *(uint32_t*)&_bf2;
            }
            d_state[0] = d_state[0] + (row_sum_9 + e0_13 + e1_14);
            d_state[1] = d_state[1] + (row_sum_12 + e0_15 + e1_16);
            float e0_17 = approx_exp2(__fmaf_rn(qk_acc[16], softmax_scale_log2, -row0_shift));
            float e1_18 = approx_exp2(__fmaf_rn(qk_acc[17], softmax_scale_log2, -row0_shift));
            qk_acc[16] = e0_17;
            qk_acc[17] = e1_18;
            float row_sum_19 = e0_17 + e1_18;
            float e0_20 = approx_exp2(__fmaf_rn(qk_acc[18], softmax_scale_log2, -row1_shift));
            float e1_21 = approx_exp2(__fmaf_rn(qk_acc[19], softmax_scale_log2, -row1_shift));
            qk_acc[18] = e0_20;
            qk_acc[19] = e1_21;
            float row_sum_22 = e0_20 + e1_21;
            float e0_23 = approx_exp2(__fmaf_rn(qk_acc[20], softmax_scale_log2, -row0_shift));
            float e1_24 = approx_exp2(__fmaf_rn(qk_acc[21], softmax_scale_log2, -row0_shift));
            qk_acc[20] = e0_23;
            qk_acc[21] = e1_24;
            float e0_25 = approx_exp2(__fmaf_rn(qk_acc[22], softmax_scale_log2, -row1_shift));
            float e1_26 = approx_exp2(__fmaf_rn(qk_acc[23], softmax_scale_log2, -row1_shift));
            qk_acc[22] = e0_25;
            qk_acc[23] = e1_26;
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_acc[_lp*2 + 16], qk_acc[_lp*2+1 + 16]));
                p_frag[_lp + 8] = *(uint32_t*)&_bf2;
            }
            d_state[0] = d_state[0] + (row_sum_19 + e0_23 + e1_24);
            d_state[1] = d_state[1] + (row_sum_22 + e0_25 + e1_26);
            float e0_27 = approx_exp2(__fmaf_rn(qk_acc[24], softmax_scale_log2, -row0_shift));
            float e1_28 = approx_exp2(__fmaf_rn(qk_acc[25], softmax_scale_log2, -row0_shift));
            qk_acc[24] = e0_27;
            qk_acc[25] = e1_28;
            float row_sum_29 = e0_27 + e1_28;
            float e0_30 = approx_exp2(__fmaf_rn(qk_acc[26], softmax_scale_log2, -row1_shift));
            float e1_31 = approx_exp2(__fmaf_rn(qk_acc[27], softmax_scale_log2, -row1_shift));
            qk_acc[26] = e0_30;
            qk_acc[27] = e1_31;
            float row_sum_32 = e0_30 + e1_31;
            float e0_33 = approx_exp2(__fmaf_rn(qk_acc[28], softmax_scale_log2, -row0_shift));
            float e1_34 = approx_exp2(__fmaf_rn(qk_acc[29], softmax_scale_log2, -row0_shift));
            qk_acc[28] = e0_33;
            qk_acc[29] = e1_34;
            float e0_35 = approx_exp2(__fmaf_rn(qk_acc[30], softmax_scale_log2, -row1_shift));
            float e1_36 = approx_exp2(__fmaf_rn(qk_acc[31], softmax_scale_log2, -row1_shift));
            qk_acc[30] = e0_35;
            qk_acc[31] = e1_36;
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_acc[_lp*2 + 24], qk_acc[_lp*2+1 + 24]));
                p_frag[_lp + 12] = *(uint32_t*)&_bf2;
            }
            d_state[0] = d_state[0] + (row_sum_29 + e0_33 + e1_34);
            d_state[1] = d_state[1] + (row_sum_32 + e0_35 + e1_36);
            mbarrier_wait(v_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)((lane / 16 / 8 * 512 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 2) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 2) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 4) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 4) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 6) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 6) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 8) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 8) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 10) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 10) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 12) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 12) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 14) / 8 * 512 + lane % 16 * 8 + ((lane / 16 + 14) % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)((lane / 16 / 8 * 512 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (16 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)((lane / 16 / 8 * 512 + (32 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (32 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (32 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)((lane / 16 / 8 * 512 + (48 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[0]), "+f"(out_acc[1]), "+f"(out_acc[2]), "+f"(out_acc[3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[4]), "+f"(out_acc[(4) + 1]), "+f"(out_acc[(4) + 2]), "+f"(out_acc[(4) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 2) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 2) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[8]), "+f"(out_acc[(8) + 1]), "+f"(out_acc[(8) + 2]), "+f"(out_acc[(8) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[12]), "+f"(out_acc[(12) + 1]), "+f"(out_acc[(12) + 2]), "+f"(out_acc[(12) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 4) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 4) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[16]), "+f"(out_acc[(16) + 1]), "+f"(out_acc[(16) + 2]), "+f"(out_acc[(16) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[20]), "+f"(out_acc[(20) + 1]), "+f"(out_acc[(20) + 2]), "+f"(out_acc[(20) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 6) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 6) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[24]), "+f"(out_acc[(24) + 1]), "+f"(out_acc[(24) + 2]), "+f"(out_acc[(24) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[28]), "+f"(out_acc[(28) + 1]), "+f"(out_acc[(28) + 2]), "+f"(out_acc[(28) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 8) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 8) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[32]), "+f"(out_acc[(32) + 1]), "+f"(out_acc[(32) + 2]), "+f"(out_acc[(32) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[36]), "+f"(out_acc[(36) + 1]), "+f"(out_acc[(36) + 2]), "+f"(out_acc[(36) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 10) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 10) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[40]), "+f"(out_acc[(40) + 1]), "+f"(out_acc[(40) + 2]), "+f"(out_acc[(40) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[44]), "+f"(out_acc[(44) + 1]), "+f"(out_acc[(44) + 2]), "+f"(out_acc[(44) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 12) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 12) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[48]), "+f"(out_acc[(48) + 1]), "+f"(out_acc[(48) + 2]), "+f"(out_acc[(48) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[52]), "+f"(out_acc[(52) + 1]), "+f"(out_acc[(52) + 2]), "+f"(out_acc[(52) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(v_smem_addr + kv_stage * 16384 + (unsigned int)(((lane / 16 + 14) / 8 * 512 + (48 + lane % 16) * 8 + ((lane / 16 + 14) % 8 * 16 ^ (48 + lane % 16 & 7) << 4) / 16) * 16))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[56]), "+f"(out_acc[(56) + 1]), "+f"(out_acc[(56) + 2]), "+f"(out_acc[(56) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(out_acc[60]), "+f"(out_acc[(60) + 1]), "+f"(out_acc[(60) + 2]), "+f"(out_acc[(60) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            mbarrier_arrive(v_empty_addr + (kv_stage) * 8);
            if (warp == 4) {
                mbarrier_wait(k_empty_addr + (kv_stage) * 8, kv_phase);
                mbarrier_wait(v_empty_addr + (kv_stage) * 8, kv_phase);
            }
            if (item_last_tile >= kv_tile_idx + 2) {
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(k_smem_addr + kv_stage * 16384), "l"((&K_map)), "r"(0), "r"(kv_tile_idx * 64 + 64 + 64), "r"(head_idx), "r"(0),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(v_smem_addr + kv_stage * 16384), "l"((&V_map)), "r"(0), "r"(kv_tile_idx * 64 + 64 + 64), "r"(head_idx), "r"(0),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
        }
        int store_rows = 1;
        if (split_slot >= 0) {
            int slot_base = split_slot * 17408;
            *(reinterpret_cast<float*>(partial_ws + (slot_base + tid)) + (0)) = out_acc[0];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 256 + tid)) + (0)) = out_acc[1];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 512 + tid)) + (0)) = out_acc[2];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 768 + tid)) + (0)) = out_acc[3];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 1024 + tid)) + (0)) = out_acc[4];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 1280 + tid)) + (0)) = out_acc[5];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 1536 + tid)) + (0)) = out_acc[6];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 1792 + tid)) + (0)) = out_acc[7];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 2048 + tid)) + (0)) = out_acc[8];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 2304 + tid)) + (0)) = out_acc[9];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 2560 + tid)) + (0)) = out_acc[10];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 2816 + tid)) + (0)) = out_acc[11];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 3072 + tid)) + (0)) = out_acc[12];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 3328 + tid)) + (0)) = out_acc[13];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 3584 + tid)) + (0)) = out_acc[14];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 3840 + tid)) + (0)) = out_acc[15];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 4096 + tid)) + (0)) = out_acc[16];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 4352 + tid)) + (0)) = out_acc[17];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 4608 + tid)) + (0)) = out_acc[18];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 4864 + tid)) + (0)) = out_acc[19];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 5120 + tid)) + (0)) = out_acc[20];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 5376 + tid)) + (0)) = out_acc[21];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 5632 + tid)) + (0)) = out_acc[22];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 5888 + tid)) + (0)) = out_acc[23];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 6144 + tid)) + (0)) = out_acc[24];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 6400 + tid)) + (0)) = out_acc[25];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 6656 + tid)) + (0)) = out_acc[26];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 6912 + tid)) + (0)) = out_acc[27];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 7168 + tid)) + (0)) = out_acc[28];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 7424 + tid)) + (0)) = out_acc[29];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 7680 + tid)) + (0)) = out_acc[30];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 7936 + tid)) + (0)) = out_acc[31];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 8192 + tid)) + (0)) = out_acc[32];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 8448 + tid)) + (0)) = out_acc[33];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 8704 + tid)) + (0)) = out_acc[34];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 8960 + tid)) + (0)) = out_acc[35];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 9216 + tid)) + (0)) = out_acc[36];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 9472 + tid)) + (0)) = out_acc[37];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 9728 + tid)) + (0)) = out_acc[38];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 9984 + tid)) + (0)) = out_acc[39];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 10240 + tid)) + (0)) = out_acc[40];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 10496 + tid)) + (0)) = out_acc[41];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 10752 + tid)) + (0)) = out_acc[42];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 11008 + tid)) + (0)) = out_acc[43];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 11264 + tid)) + (0)) = out_acc[44];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 11520 + tid)) + (0)) = out_acc[45];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 11776 + tid)) + (0)) = out_acc[46];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 12032 + tid)) + (0)) = out_acc[47];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 12288 + tid)) + (0)) = out_acc[48];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 12544 + tid)) + (0)) = out_acc[49];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 12800 + tid)) + (0)) = out_acc[50];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 13056 + tid)) + (0)) = out_acc[51];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 13312 + tid)) + (0)) = out_acc[52];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 13568 + tid)) + (0)) = out_acc[53];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 13824 + tid)) + (0)) = out_acc[54];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 14080 + tid)) + (0)) = out_acc[55];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 14336 + tid)) + (0)) = out_acc[56];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 14592 + tid)) + (0)) = out_acc[57];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 14848 + tid)) + (0)) = out_acc[58];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 15104 + tid)) + (0)) = out_acc[59];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 15360 + tid)) + (0)) = out_acc[60];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 15616 + tid)) + (0)) = out_acc[61];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 15872 + tid)) + (0)) = out_acc[62];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 16128 + tid)) + (0)) = out_acc[63];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 16384 + tid)) + (0)) = m_state[0];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 16896 + tid)) + (0)) = d_state[0];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 16640 + tid)) + (0)) = m_state[1];
            *(reinterpret_cast<float*>(partial_ws + (slot_base + 17152 + tid)) + (0)) = d_state[1];
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                int tail_item = split_slot / split_k;
                int arrived = atomicAdd(&split_counters[tail_item], 1);
                int merge_flag = 0;
                if (arrived + 1 == split_k) {
                    merge_flag = 1;
                    int _atomic_old_0 = atomicAdd(&split_counters[tail_item], -split_k);
                }
                split_flag[0] = merge_flag;
            }
            __syncthreads();
            store_rows = split_flag[0];
            if (store_rows != 0) {
                __threadfence();
                for (int other = 0; other < split_k; other++) {
                    int base = (split_slot - split_part) * 17408 + other * 17408 + tid;
                    float m_o0 = partial_ws[base + 16384];
                    float m_o1 = partial_ws[base + 16384 + 256];
                    float d_o0 = partial_ws[base + 16384 + 512];
                    float d_o1 = partial_ws[base + 16384 + 768];
                    if (other == 0) {
                        m_state[0] = m_o0;
                        m_state[1] = m_o1;
                        d_state[0] = d_o0;
                        d_state[1] = d_o1;
                        out_acc[0] = partial_ws[base];
                        out_acc[1] = partial_ws[base + 256];
                        out_acc[2] = partial_ws[base + 512];
                        out_acc[3] = partial_ws[base + 768];
                        out_acc[4] = partial_ws[base + 1024];
                        out_acc[5] = partial_ws[base + 1280];
                        out_acc[6] = partial_ws[base + 1536];
                        out_acc[7] = partial_ws[base + 1792];
                        out_acc[8] = partial_ws[base + 2048];
                        out_acc[9] = partial_ws[base + 2304];
                        out_acc[10] = partial_ws[base + 2560];
                        out_acc[11] = partial_ws[base + 2816];
                        out_acc[12] = partial_ws[base + 3072];
                        out_acc[13] = partial_ws[base + 3328];
                        out_acc[14] = partial_ws[base + 3584];
                        out_acc[15] = partial_ws[base + 3840];
                        out_acc[16] = partial_ws[base + 4096];
                        out_acc[17] = partial_ws[base + 4352];
                        out_acc[18] = partial_ws[base + 4608];
                        out_acc[19] = partial_ws[base + 4864];
                        out_acc[20] = partial_ws[base + 5120];
                        out_acc[21] = partial_ws[base + 5376];
                        out_acc[22] = partial_ws[base + 5632];
                        out_acc[23] = partial_ws[base + 5888];
                        out_acc[24] = partial_ws[base + 6144];
                        out_acc[25] = partial_ws[base + 6400];
                        out_acc[26] = partial_ws[base + 6656];
                        out_acc[27] = partial_ws[base + 6912];
                        out_acc[28] = partial_ws[base + 7168];
                        out_acc[29] = partial_ws[base + 7424];
                        out_acc[30] = partial_ws[base + 7680];
                        out_acc[31] = partial_ws[base + 7936];
                        out_acc[32] = partial_ws[base + 8192];
                        out_acc[33] = partial_ws[base + 8448];
                        out_acc[34] = partial_ws[base + 8704];
                        out_acc[35] = partial_ws[base + 8960];
                        out_acc[36] = partial_ws[base + 9216];
                        out_acc[37] = partial_ws[base + 9472];
                        out_acc[38] = partial_ws[base + 9728];
                        out_acc[39] = partial_ws[base + 9984];
                        out_acc[40] = partial_ws[base + 10240];
                        out_acc[41] = partial_ws[base + 10496];
                        out_acc[42] = partial_ws[base + 10752];
                        out_acc[43] = partial_ws[base + 11008];
                        out_acc[44] = partial_ws[base + 11264];
                        out_acc[45] = partial_ws[base + 11520];
                        out_acc[46] = partial_ws[base + 11776];
                        out_acc[47] = partial_ws[base + 12032];
                        out_acc[48] = partial_ws[base + 12288];
                        out_acc[49] = partial_ws[base + 12544];
                        out_acc[50] = partial_ws[base + 12800];
                        out_acc[51] = partial_ws[base + 13056];
                        out_acc[52] = partial_ws[base + 13312];
                        out_acc[53] = partial_ws[base + 13568];
                        out_acc[54] = partial_ws[base + 13824];
                        out_acc[55] = partial_ws[base + 14080];
                        out_acc[56] = partial_ws[base + 14336];
                        out_acc[57] = partial_ws[base + 14592];
                        out_acc[58] = partial_ws[base + 14848];
                        out_acc[59] = partial_ws[base + 15104];
                        out_acc[60] = partial_ws[base + 15360];
                        out_acc[61] = partial_ws[base + 15616];
                        out_acc[62] = partial_ws[base + 15872];
                        out_acc[63] = partial_ws[base + 16128];
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
                        float o_other = partial_ws[base];
                        out_acc[0] = __fmaf_rn(out_acc[0], a0, o_other * b0);
                        float o_other_0 = partial_ws[base + 256];
                        out_acc[1] = __fmaf_rn(out_acc[1], a0, o_other_0 * b0);
                        float o_other_1 = partial_ws[base + 512];
                        out_acc[2] = __fmaf_rn(out_acc[2], a1, o_other_1 * b1);
                        float o_other_2 = partial_ws[base + 768];
                        out_acc[3] = __fmaf_rn(out_acc[3], a1, o_other_2 * b1);
                        float o_other_3 = partial_ws[base + 1024];
                        out_acc[4] = __fmaf_rn(out_acc[4], a0, o_other_3 * b0);
                        float o_other_4 = partial_ws[base + 1280];
                        out_acc[5] = __fmaf_rn(out_acc[5], a0, o_other_4 * b0);
                        float o_other_5 = partial_ws[base + 1536];
                        out_acc[6] = __fmaf_rn(out_acc[6], a1, o_other_5 * b1);
                        float o_other_6 = partial_ws[base + 1792];
                        out_acc[7] = __fmaf_rn(out_acc[7], a1, o_other_6 * b1);
                        float o_other_7 = partial_ws[base + 2048];
                        out_acc[8] = __fmaf_rn(out_acc[8], a0, o_other_7 * b0);
                        float o_other_8 = partial_ws[base + 2304];
                        out_acc[9] = __fmaf_rn(out_acc[9], a0, o_other_8 * b0);
                        float o_other_9 = partial_ws[base + 2560];
                        out_acc[10] = __fmaf_rn(out_acc[10], a1, o_other_9 * b1);
                        float o_other_10 = partial_ws[base + 2816];
                        out_acc[11] = __fmaf_rn(out_acc[11], a1, o_other_10 * b1);
                        float o_other_11 = partial_ws[base + 3072];
                        out_acc[12] = __fmaf_rn(out_acc[12], a0, o_other_11 * b0);
                        float o_other_12 = partial_ws[base + 3328];
                        out_acc[13] = __fmaf_rn(out_acc[13], a0, o_other_12 * b0);
                        float o_other_13 = partial_ws[base + 3584];
                        out_acc[14] = __fmaf_rn(out_acc[14], a1, o_other_13 * b1);
                        float o_other_14 = partial_ws[base + 3840];
                        out_acc[15] = __fmaf_rn(out_acc[15], a1, o_other_14 * b1);
                        float o_other_15 = partial_ws[base + 4096];
                        out_acc[16] = __fmaf_rn(out_acc[16], a0, o_other_15 * b0);
                        float o_other_16 = partial_ws[base + 4352];
                        out_acc[17] = __fmaf_rn(out_acc[17], a0, o_other_16 * b0);
                        float o_other_17 = partial_ws[base + 4608];
                        out_acc[18] = __fmaf_rn(out_acc[18], a1, o_other_17 * b1);
                        float o_other_18 = partial_ws[base + 4864];
                        out_acc[19] = __fmaf_rn(out_acc[19], a1, o_other_18 * b1);
                        float o_other_19 = partial_ws[base + 5120];
                        out_acc[20] = __fmaf_rn(out_acc[20], a0, o_other_19 * b0);
                        float o_other_20 = partial_ws[base + 5376];
                        out_acc[21] = __fmaf_rn(out_acc[21], a0, o_other_20 * b0);
                        float o_other_21 = partial_ws[base + 5632];
                        out_acc[22] = __fmaf_rn(out_acc[22], a1, o_other_21 * b1);
                        float o_other_22 = partial_ws[base + 5888];
                        out_acc[23] = __fmaf_rn(out_acc[23], a1, o_other_22 * b1);
                        float o_other_23 = partial_ws[base + 6144];
                        out_acc[24] = __fmaf_rn(out_acc[24], a0, o_other_23 * b0);
                        float o_other_24 = partial_ws[base + 6400];
                        out_acc[25] = __fmaf_rn(out_acc[25], a0, o_other_24 * b0);
                        float o_other_25 = partial_ws[base + 6656];
                        out_acc[26] = __fmaf_rn(out_acc[26], a1, o_other_25 * b1);
                        float o_other_26 = partial_ws[base + 6912];
                        out_acc[27] = __fmaf_rn(out_acc[27], a1, o_other_26 * b1);
                        float o_other_27 = partial_ws[base + 7168];
                        out_acc[28] = __fmaf_rn(out_acc[28], a0, o_other_27 * b0);
                        float o_other_28 = partial_ws[base + 7424];
                        out_acc[29] = __fmaf_rn(out_acc[29], a0, o_other_28 * b0);
                        float o_other_29 = partial_ws[base + 7680];
                        out_acc[30] = __fmaf_rn(out_acc[30], a1, o_other_29 * b1);
                        float o_other_30 = partial_ws[base + 7936];
                        out_acc[31] = __fmaf_rn(out_acc[31], a1, o_other_30 * b1);
                        float o_other_31 = partial_ws[base + 8192];
                        out_acc[32] = __fmaf_rn(out_acc[32], a0, o_other_31 * b0);
                        float o_other_32 = partial_ws[base + 8448];
                        out_acc[33] = __fmaf_rn(out_acc[33], a0, o_other_32 * b0);
                        float o_other_33 = partial_ws[base + 8704];
                        out_acc[34] = __fmaf_rn(out_acc[34], a1, o_other_33 * b1);
                        float o_other_34 = partial_ws[base + 8960];
                        out_acc[35] = __fmaf_rn(out_acc[35], a1, o_other_34 * b1);
                        float o_other_35 = partial_ws[base + 9216];
                        out_acc[36] = __fmaf_rn(out_acc[36], a0, o_other_35 * b0);
                        float o_other_36 = partial_ws[base + 9472];
                        out_acc[37] = __fmaf_rn(out_acc[37], a0, o_other_36 * b0);
                        float o_other_37 = partial_ws[base + 9728];
                        out_acc[38] = __fmaf_rn(out_acc[38], a1, o_other_37 * b1);
                        float o_other_38 = partial_ws[base + 9984];
                        out_acc[39] = __fmaf_rn(out_acc[39], a1, o_other_38 * b1);
                        float o_other_39 = partial_ws[base + 10240];
                        out_acc[40] = __fmaf_rn(out_acc[40], a0, o_other_39 * b0);
                        float o_other_40 = partial_ws[base + 10496];
                        out_acc[41] = __fmaf_rn(out_acc[41], a0, o_other_40 * b0);
                        float o_other_41 = partial_ws[base + 10752];
                        out_acc[42] = __fmaf_rn(out_acc[42], a1, o_other_41 * b1);
                        float o_other_42 = partial_ws[base + 11008];
                        out_acc[43] = __fmaf_rn(out_acc[43], a1, o_other_42 * b1);
                        float o_other_43 = partial_ws[base + 11264];
                        out_acc[44] = __fmaf_rn(out_acc[44], a0, o_other_43 * b0);
                        float o_other_44 = partial_ws[base + 11520];
                        out_acc[45] = __fmaf_rn(out_acc[45], a0, o_other_44 * b0);
                        float o_other_45 = partial_ws[base + 11776];
                        out_acc[46] = __fmaf_rn(out_acc[46], a1, o_other_45 * b1);
                        float o_other_46 = partial_ws[base + 12032];
                        out_acc[47] = __fmaf_rn(out_acc[47], a1, o_other_46 * b1);
                        float o_other_47 = partial_ws[base + 12288];
                        out_acc[48] = __fmaf_rn(out_acc[48], a0, o_other_47 * b0);
                        float o_other_48 = partial_ws[base + 12544];
                        out_acc[49] = __fmaf_rn(out_acc[49], a0, o_other_48 * b0);
                        float o_other_49 = partial_ws[base + 12800];
                        out_acc[50] = __fmaf_rn(out_acc[50], a1, o_other_49 * b1);
                        float o_other_50 = partial_ws[base + 13056];
                        out_acc[51] = __fmaf_rn(out_acc[51], a1, o_other_50 * b1);
                        float o_other_51 = partial_ws[base + 13312];
                        out_acc[52] = __fmaf_rn(out_acc[52], a0, o_other_51 * b0);
                        float o_other_52 = partial_ws[base + 13568];
                        out_acc[53] = __fmaf_rn(out_acc[53], a0, o_other_52 * b0);
                        float o_other_53 = partial_ws[base + 13824];
                        out_acc[54] = __fmaf_rn(out_acc[54], a1, o_other_53 * b1);
                        float o_other_54 = partial_ws[base + 14080];
                        out_acc[55] = __fmaf_rn(out_acc[55], a1, o_other_54 * b1);
                        float o_other_55 = partial_ws[base + 14336];
                        out_acc[56] = __fmaf_rn(out_acc[56], a0, o_other_55 * b0);
                        float o_other_56 = partial_ws[base + 14592];
                        out_acc[57] = __fmaf_rn(out_acc[57], a0, o_other_56 * b0);
                        float o_other_57 = partial_ws[base + 14848];
                        out_acc[58] = __fmaf_rn(out_acc[58], a1, o_other_57 * b1);
                        float o_other_58 = partial_ws[base + 15104];
                        out_acc[59] = __fmaf_rn(out_acc[59], a1, o_other_58 * b1);
                        float o_other_59 = partial_ws[base + 15360];
                        out_acc[60] = __fmaf_rn(out_acc[60], a0, o_other_59 * b0);
                        float o_other_60 = partial_ws[base + 15616];
                        out_acc[61] = __fmaf_rn(out_acc[61], a0, o_other_60 * b0);
                        float o_other_61 = partial_ws[base + 15872];
                        out_acc[62] = __fmaf_rn(out_acc[62], a1, o_other_61 * b1);
                        float o_other_62 = partial_ws[base + 16128];
                        out_acc[63] = __fmaf_rn(out_acc[63], a1, o_other_62 * b1);
                    }
                }
            }
        }
        if (store_rows != 0) {
            float peer_d = __shfl_xor_sync(0xFFFFFFFF, d_state[0], 2);
            d_state[0] = d_state[0] + peer_d;
            float peer_d2 = __shfl_xor_sync(0xFFFFFFFF, d_state[0], 1);
            d_state[0] = d_state[0] + peer_d2;
            float peer_d_0 = __shfl_xor_sync(0xFFFFFFFF, d_state[1], 2);
            d_state[1] = d_state[1] + peer_d_0;
            float peer_d2_1 = __shfl_xor_sync(0xFFFFFFFF, d_state[1], 1);
            d_state[1] = d_state[1] + peer_d2_1;
            float _rcp_0 = approx_rcp(d_state[0]);
            float _rcp_1 = approx_rcp(d_state[1]);
#if __CUDA_ARCH__ == 900
            out_acc[0] = mul_f32_rn_noftz(out_acc[0], _rcp_0);
            out_acc[1] = mul_f32_rn_noftz(out_acc[1], _rcp_0);
            out_acc[2] = mul_f32_rn_noftz(out_acc[2], _rcp_1);
            out_acc[3] = mul_f32_rn_noftz(out_acc[3], _rcp_1);
            out_acc[4] = mul_f32_rn_noftz(out_acc[4], _rcp_0);
            out_acc[5] = mul_f32_rn_noftz(out_acc[5], _rcp_0);
            out_acc[6] = mul_f32_rn_noftz(out_acc[6], _rcp_1);
            out_acc[7] = mul_f32_rn_noftz(out_acc[7], _rcp_1);
            out_acc[8] = mul_f32_rn_noftz(out_acc[8], _rcp_0);
            out_acc[9] = mul_f32_rn_noftz(out_acc[9], _rcp_0);
            out_acc[10] = mul_f32_rn_noftz(out_acc[10], _rcp_1);
            out_acc[11] = mul_f32_rn_noftz(out_acc[11], _rcp_1);
            out_acc[12] = mul_f32_rn_noftz(out_acc[12], _rcp_0);
            out_acc[13] = mul_f32_rn_noftz(out_acc[13], _rcp_0);
            out_acc[14] = mul_f32_rn_noftz(out_acc[14], _rcp_1);
            out_acc[15] = mul_f32_rn_noftz(out_acc[15], _rcp_1);
            out_acc[16] = mul_f32_rn_noftz(out_acc[16], _rcp_0);
            out_acc[17] = mul_f32_rn_noftz(out_acc[17], _rcp_0);
            out_acc[18] = mul_f32_rn_noftz(out_acc[18], _rcp_1);
            out_acc[19] = mul_f32_rn_noftz(out_acc[19], _rcp_1);
            out_acc[20] = mul_f32_rn_noftz(out_acc[20], _rcp_0);
            out_acc[21] = mul_f32_rn_noftz(out_acc[21], _rcp_0);
            out_acc[22] = mul_f32_rn_noftz(out_acc[22], _rcp_1);
            out_acc[23] = mul_f32_rn_noftz(out_acc[23], _rcp_1);
            out_acc[24] = mul_f32_rn_noftz(out_acc[24], _rcp_0);
            out_acc[25] = mul_f32_rn_noftz(out_acc[25], _rcp_0);
            out_acc[26] = mul_f32_rn_noftz(out_acc[26], _rcp_1);
            out_acc[27] = mul_f32_rn_noftz(out_acc[27], _rcp_1);
            out_acc[28] = mul_f32_rn_noftz(out_acc[28], _rcp_0);
            out_acc[29] = mul_f32_rn_noftz(out_acc[29], _rcp_0);
            out_acc[30] = mul_f32_rn_noftz(out_acc[30], _rcp_1);
            out_acc[31] = mul_f32_rn_noftz(out_acc[31], _rcp_1);
            out_acc[32] = mul_f32_rn_noftz(out_acc[32], _rcp_0);
            out_acc[33] = mul_f32_rn_noftz(out_acc[33], _rcp_0);
            out_acc[34] = mul_f32_rn_noftz(out_acc[34], _rcp_1);
            out_acc[35] = mul_f32_rn_noftz(out_acc[35], _rcp_1);
            out_acc[36] = mul_f32_rn_noftz(out_acc[36], _rcp_0);
            out_acc[37] = mul_f32_rn_noftz(out_acc[37], _rcp_0);
            out_acc[38] = mul_f32_rn_noftz(out_acc[38], _rcp_1);
            out_acc[39] = mul_f32_rn_noftz(out_acc[39], _rcp_1);
            out_acc[40] = mul_f32_rn_noftz(out_acc[40], _rcp_0);
            out_acc[41] = mul_f32_rn_noftz(out_acc[41], _rcp_0);
            out_acc[42] = mul_f32_rn_noftz(out_acc[42], _rcp_1);
            out_acc[43] = mul_f32_rn_noftz(out_acc[43], _rcp_1);
            out_acc[44] = mul_f32_rn_noftz(out_acc[44], _rcp_0);
            out_acc[45] = mul_f32_rn_noftz(out_acc[45], _rcp_0);
            out_acc[46] = mul_f32_rn_noftz(out_acc[46], _rcp_1);
            out_acc[47] = mul_f32_rn_noftz(out_acc[47], _rcp_1);
            out_acc[48] = mul_f32_rn_noftz(out_acc[48], _rcp_0);
            out_acc[49] = mul_f32_rn_noftz(out_acc[49], _rcp_0);
            out_acc[50] = mul_f32_rn_noftz(out_acc[50], _rcp_1);
            out_acc[51] = mul_f32_rn_noftz(out_acc[51], _rcp_1);
            out_acc[52] = mul_f32_rn_noftz(out_acc[52], _rcp_0);
            out_acc[53] = mul_f32_rn_noftz(out_acc[53], _rcp_0);
            out_acc[54] = mul_f32_rn_noftz(out_acc[54], _rcp_1);
            out_acc[55] = mul_f32_rn_noftz(out_acc[55], _rcp_1);
            out_acc[56] = mul_f32_rn_noftz(out_acc[56], _rcp_0);
            out_acc[57] = mul_f32_rn_noftz(out_acc[57], _rcp_0);
            out_acc[58] = mul_f32_rn_noftz(out_acc[58], _rcp_1);
            out_acc[59] = mul_f32_rn_noftz(out_acc[59], _rcp_1);
            out_acc[60] = mul_f32_rn_noftz(out_acc[60], _rcp_0);
            out_acc[61] = mul_f32_rn_noftz(out_acc[61], _rcp_0);
            out_acc[62] = mul_f32_rn_noftz(out_acc[62], _rcp_1);
            out_acc[63] = mul_f32_rn_noftz(out_acc[63], _rcp_1);
#else
            {
                float2 _pair_scale_even2_7 = make_float2(_rcp_0, _rcp_0);
                float2 _pair_scale_odd2_7 = make_float2(_rcp_1, _rcp_1);
                float2* _pair_scale_src2_7 = reinterpret_cast<float2*>(&out_acc[0]);
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
            }
#endif
            unsigned int o_pack[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 0], out_acc[_lp*2+1 + 0]));
                o_pack[_lp] = *(uint32_t*)&_bf2;
            }
#if __CUDA_ARCH__ == 900
            uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(q_smem_addr + (unsigned int)((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
            uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(q_smem_addr + (unsigned int)((lane / 16 / 8 * 1024 + (warp * 16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if __CUDA_ARCH__ == 900
                :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[3]))
#else
                :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack[3]))
#endif
                : "memory");
            unsigned int o_pack_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 8], out_acc[_lp*2+1 + 8]));
                o_pack_2[_lp] = *(uint32_t*)&_bf2;
            }
#if __CUDA_ARCH__ == 900
            uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((2 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((2 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((2 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((2 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if __CUDA_ARCH__ == 900
                :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[3]))
#else
                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_2[3]))
#endif
                : "memory");
            unsigned int o_pack_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 16], out_acc[_lp*2+1 + 16]));
                o_pack_3[_lp] = *(uint32_t*)&_bf2;
            }
#if __CUDA_ARCH__ == 900
            uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((4 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((4 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
            uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((4 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((4 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if __CUDA_ARCH__ == 900
                :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[3]))
#else
                :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_3[3]))
#endif
                : "memory");
            unsigned int o_pack_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 24], out_acc[_lp*2+1 + 24]));
                o_pack_4[_lp] = *(uint32_t*)&_bf2;
            }
#if __CUDA_ARCH__ == 900
            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((6 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((6 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
            uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((6 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((6 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if __CUDA_ARCH__ == 900
                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[3]))
#else
                :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_4[3]))
#endif
                : "memory");
            unsigned int o_pack_5[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 32], out_acc[_lp*2+1 + 32]));
                o_pack_5[_lp] = *(uint32_t*)&_bf2;
            }
#if __CUDA_ARCH__ == 900
            uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((8 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((8 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
            uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((8 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((8 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if __CUDA_ARCH__ == 900
                :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[3]))
#else
                :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_5[3]))
#endif
                : "memory");
            unsigned int o_pack_6[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 40], out_acc[_lp*2+1 + 40]));
                o_pack_6[_lp] = *(uint32_t*)&_bf2;
            }
#if __CUDA_ARCH__ == 900
            uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((10 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((10 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
            uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((10 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((10 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if __CUDA_ARCH__ == 900
                :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[3]))
#else
                :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_6[3]))
#endif
                : "memory");
            unsigned int o_pack_7[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 48], out_acc[_lp*2+1 + 48]));
                o_pack_7[_lp] = *(uint32_t*)&_bf2;
            }
#if __CUDA_ARCH__ == 900
            uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((12 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((12 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
            uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((12 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((12 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if __CUDA_ARCH__ == 900
                :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[3]))
#else
                :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_7[3]))
#endif
                : "memory");
            unsigned int o_pack_8[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_acc[_lp*2 + 56], out_acc[_lp*2+1 + 56]));
                o_pack_8[_lp] = *(uint32_t*)&_bf2;
            }
#if __CUDA_ARCH__ == 900
            uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((14 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((14 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#else
            uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(q_smem_addr + (unsigned int)(((14 + lane / 16) / 8 * 1024 + (warp * 16 + lane % 16) * 8 + ((14 + lane / 16) % 8 * 16 ^ (warp * 16 + lane % 16 & 7) << 4) / 16) * 16));
#endif
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
#if __CUDA_ARCH__ == 900
                :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_8[3]))
#else
                :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_pack_8[3]))
#endif
                : "memory");
            __syncwarp();
            unsigned int o_vec[4];
            int q_pos = q_tile_base + (warp * 16 + lane / 8);
            if (q_pos < tokens) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                    : "r"(q_smem_addr + (unsigned int)((lane % 8 / 8 * 1024 + (warp * 16 + lane / 8) * 8 + (lane % 8 % 8 * 16 ^ (warp * 16 + lane / 8 & 7) << 4) / 16) * 16)));
                reinterpret_cast<int4*>(O + ((q_pos * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8))[0] = reinterpret_cast<int4*>(o_vec)[0];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                    : "r"(q_smem_addr + (unsigned int)(((lane % 8 + 8) / 8 * 1024 + (warp * 16 + lane / 8) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp * 16 + lane / 8 & 7) << 4) / 16) * 16)));
                reinterpret_cast<int4*>(O + ((q_pos * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8 + 64))[0] = reinterpret_cast<int4*>(o_vec)[0];
            }
            int q_pos_9 = q_tile_base + (warp * 16 + lane / 8 + 4);
            if (q_pos_9 < tokens) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                    : "r"(q_smem_addr + (unsigned int)((lane % 8 / 8 * 1024 + (warp * 16 + lane / 8 + 4) * 8 + (lane % 8 % 8 * 16 ^ (warp * 16 + lane / 8 + 4 & 7) << 4) / 16) * 16)));
                reinterpret_cast<int4*>(O + ((q_pos_9 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8))[0] = reinterpret_cast<int4*>(o_vec)[0];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                    : "r"(q_smem_addr + (unsigned int)(((lane % 8 + 8) / 8 * 1024 + (warp * 16 + lane / 8 + 4) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp * 16 + lane / 8 + 4 & 7) << 4) / 16) * 16)));
                reinterpret_cast<int4*>(O + ((q_pos_9 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8 + 64))[0] = reinterpret_cast<int4*>(o_vec)[0];
            }
            int q_pos_10 = q_tile_base + (warp * 16 + lane / 8 + 8);
            if (q_pos_10 < tokens) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                    : "r"(q_smem_addr + (unsigned int)((lane % 8 / 8 * 1024 + (warp * 16 + lane / 8 + 8) * 8 + (lane % 8 % 8 * 16 ^ (warp * 16 + lane / 8 + 8 & 7) << 4) / 16) * 16)));
                reinterpret_cast<int4*>(O + ((q_pos_10 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8))[0] = reinterpret_cast<int4*>(o_vec)[0];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                    : "r"(q_smem_addr + (unsigned int)(((lane % 8 + 8) / 8 * 1024 + (warp * 16 + lane / 8 + 8) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp * 16 + lane / 8 + 8 & 7) << 4) / 16) * 16)));
                reinterpret_cast<int4*>(O + ((q_pos_10 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8 + 64))[0] = reinterpret_cast<int4*>(o_vec)[0];
            }
            int q_pos_11 = q_tile_base + (warp * 16 + lane / 8 + 12);
            if (q_pos_11 < tokens) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                    : "r"(q_smem_addr + (unsigned int)((lane % 8 / 8 * 1024 + (warp * 16 + lane / 8 + 12) * 8 + (lane % 8 % 8 * 16 ^ (warp * 16 + lane / 8 + 12 & 7) << 4) / 16) * 16)));
                reinterpret_cast<int4*>(O + ((q_pos_11 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8))[0] = reinterpret_cast<int4*>(o_vec)[0];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                    : "r"(q_smem_addr + (unsigned int)(((lane % 8 + 8) / 8 * 1024 + (warp * 16 + lane / 8 + 12) * 8 + ((lane % 8 + 8) % 8 * 16 ^ (warp * 16 + lane / 8 + 12 & 7) << 4) / 16) * 16)));
                reinterpret_cast<int4*>(O + ((q_pos_11 * NUM_HEADS_CONST + head_idx) * 128 + lane % 8 * 8 + 64))[0] = reinterpret_cast<int4*>(o_vec)[0];
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
    }
    if (warp < 4) {
        asm volatile("barrier.sync 1, 256;" ::: "memory");
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
