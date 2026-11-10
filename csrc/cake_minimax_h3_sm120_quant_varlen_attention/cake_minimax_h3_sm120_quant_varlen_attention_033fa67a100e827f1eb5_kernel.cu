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
#define SMEM_Q_OFF 1024
#define SMEM_Q_STAGE_BYTES 16384
#define SMEM_Q_STRIDE 16384
#define SMEM_K_STAGE_OFF 17408
#define SMEM_K_STAGE_STAGE_BYTES 16384
#define SMEM_K_STAGE_STRIDE 16384
#define SMEM_V_STAGE_OFF 50176
#define SMEM_V_STAGE_STAGE_BYTES 16384
#define SMEM_V_STAGE_STRIDE 16384
#define SMEM_SLOT_CELL_OFF 0
#define SMEM_SLOT_CELL_STAGE_BYTES 16
#define SMEM_SLOT_CELL_STRIDE 16
#define SMEM_TOTAL 82944
#define THREADS 256
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
kernel_cake_minimax_h3_sm120_quant_varlen_attention_033fa67a100e827f1eb5(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap V_map, __nv_bfloat16* O, float* q_scale, float* k_scale, float* v_scale, int* __restrict__ seg_begin, int* __restrict__ seg_len, int* __restrict__ unit_table, int total_units, int num_heads, int num_kblocks, float softmax_scale_log2, unsigned int* hctr, int lead_ctas, int tail_ctas)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[16];
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
    uint8_t* Q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int Q_addr = smem + 1024;
    uint8_t* K_stage = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int K_stage_addr = smem + 17408;
    uint8_t* V_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int V_stage_addr = smem + 50176;
    int* slot_cell = reinterpret_cast<int*>(smem_static_raw + 0);
    const int slot_cell_addr = (int)(unsigned long long)__cvta_generic_to_shared(slot_cell);

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
            // k_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            // v_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    unsigned int kv_stage = 0;
    unsigned int kv_phase = 0;
    if (warp >= 4) {
        asm volatile("barrier.arrive 1, 256;" ::: "memory");
    }
    unsigned int fp8_unit_word_main[1];
    fp8_unit_word_main[0] = 2139062143;
    uint8_t fp8_unit_sfa_main[1];
    uint8_t fp8_unit_sfb_main[1];
    fp8_unit_sfa_main[0] = fp8_unit_word_main[0];
    fp8_unit_sfb_main[0] = fp8_unit_word_main[0];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned int nn_r[2];
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&hctr[4 * num_heads], 1);
        slot_cell[0] = (int)_atomic_fetch_add_0;
        unsigned int _atomic_fetch_add_1 = atomicAdd(&hctr[4 * num_heads], 1);
        slot_cell[1] = (int)_atomic_fetch_add_1;
    }
    __syncthreads();
    int cur = slot_cell[0];
    int acq_head = -1;
    unsigned int _phase_q_full_0 = 0;
    unsigned int _phase_k_full = 0;
    #pragma unroll 1
    for (int it = 0; it < total_units; it++) {
        if (cur >= total_units) {
            break;
        }
        int seg = unit_table[2 * cur];
        int packed = unit_table[2 * cur + 1];
        int head = packed >> 16;
        int q_tile = packed & 65535;
        int seg_begin_v = seg_begin[seg];
        int seg_end_v = seg_begin_v + seg_len[seg];
        int q_row0 = seg_begin_v + q_tile * 128;
        int kb_first = seg_begin_v / 128;
        int kb_last = (seg_end_v - 1) / 128;
        if (tid == 0) {
            unsigned int _atomic_fetch_add_2 = atomicAdd(&hctr[4 * num_heads], 1);
            nn_r[0] = _atomic_fetch_add_2;
        }
        if (head != acq_head) {
            if (elect_sync()) {
                {
                unsigned int _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + head))) : "memory");
                } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(1));
                }
            }
            __syncwarp();
            asm volatile("fence.proxy.async.global;" ::: "memory");
            acq_head = head;
        }
        if (it == 0) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(q_row0), "r"(head),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            if (kb_first <= kb_last) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + kv_stage * 16384), "l"((&K_map)), "r"(0), "r"(kb_first * 128), "r"(head),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"(kb_first * 128), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (kb_last >= kb_first + 1) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&K_map)), "r"(0), "r"((kb_first + 1) * 128), "r"(head),
                               "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(V_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&V_map)), "r"((kb_first + 1) * 128), "r"(0), "r"(head),
                               "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
        }
        int row0 = q_row0 + warp * 16 + (lane >> 2);
        int row1 = row0 + 8;
        int row0_c = ((row0 < seg_end_v) ? row0 : seg_end_v - 1);
        int row1_c = ((row1 < seg_end_v) ? row1 : seg_end_v - 1);
        float qs0 = q_scale[row0_c * num_heads + head];
        float qs1 = q_scale[row1_c * num_heads + head];
        float c_row0 = qs0 * softmax_scale_log2;
        float c_row1 = qs1 * softmax_scale_log2;
        unsigned int q_frags[16];
        unsigned int k_frag[4];
        unsigned int v_frag[4];
        float s_acc[64];
        unsigned int p_frag[16];
        float p_tmp[8];
        float o_acc[64];
        float o_tmp[4];
        float m_state[2];
        float l_state[2];
        o_acc[0] = 0.0f;
        o_acc[1] = 0.0f;
        o_acc[2] = 0.0f;
        o_acc[3] = 0.0f;
        o_acc[4] = 0.0f;
        o_acc[5] = 0.0f;
        o_acc[6] = 0.0f;
        o_acc[7] = 0.0f;
        o_acc[8] = 0.0f;
        o_acc[9] = 0.0f;
        o_acc[10] = 0.0f;
        o_acc[11] = 0.0f;
        o_acc[12] = 0.0f;
        o_acc[13] = 0.0f;
        o_acc[14] = 0.0f;
        o_acc[15] = 0.0f;
        o_acc[16] = 0.0f;
        o_acc[17] = 0.0f;
        o_acc[18] = 0.0f;
        o_acc[19] = 0.0f;
        o_acc[20] = 0.0f;
        o_acc[21] = 0.0f;
        o_acc[22] = 0.0f;
        o_acc[23] = 0.0f;
        o_acc[24] = 0.0f;
        o_acc[25] = 0.0f;
        o_acc[26] = 0.0f;
        o_acc[27] = 0.0f;
        o_acc[28] = 0.0f;
        o_acc[29] = 0.0f;
        o_acc[30] = 0.0f;
        o_acc[31] = 0.0f;
        o_acc[32] = 0.0f;
        o_acc[33] = 0.0f;
        o_acc[34] = 0.0f;
        o_acc[35] = 0.0f;
        o_acc[36] = 0.0f;
        o_acc[37] = 0.0f;
        o_acc[38] = 0.0f;
        o_acc[39] = 0.0f;
        o_acc[40] = 0.0f;
        o_acc[41] = 0.0f;
        o_acc[42] = 0.0f;
        o_acc[43] = 0.0f;
        o_acc[44] = 0.0f;
        o_acc[45] = 0.0f;
        o_acc[46] = 0.0f;
        o_acc[47] = 0.0f;
        o_acc[48] = 0.0f;
        o_acc[49] = 0.0f;
        o_acc[50] = 0.0f;
        o_acc[51] = 0.0f;
        o_acc[52] = 0.0f;
        o_acc[53] = 0.0f;
        o_acc[54] = 0.0f;
        o_acc[55] = 0.0f;
        o_acc[56] = 0.0f;
        o_acc[57] = 0.0f;
        o_acc[58] = 0.0f;
        o_acc[59] = 0.0f;
        o_acc[60] = 0.0f;
        o_acc[61] = 0.0f;
        o_acc[62] = 0.0f;
        o_acc[63] = 0.0f;
        m_state[0] = -CUDART_INF_F;
        m_state[1] = -CUDART_INF_F;
        l_state[0] = 0.0f;
        l_state[1] = 0.0f;
        mbarrier_wait(q_full_addr, _phase_q_full_0);
        _phase_q_full_0 ^= 1;
        int nslot = slot_cell[it + 1 & 1];
        int has_next = ((nslot < total_units) ? 1 : 0);
        int nslot_c = ((nslot < total_units) ? nslot : cur);
        int nseg = unit_table[2 * nslot_c];
        int npacked = unit_table[2 * nslot_c + 1];
        int next_head = npacked >> 16;
        int nseg_begin = seg_begin[nseg];
        int next_q_row0 = nseg_begin + (npacked & 65535) * 128;
        int next_kb_first = nseg_begin / 128;
        int next_kb_last = (nseg_begin + seg_len[nseg] - 1) / 128;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[0]), "=r"(q_frags[1]), "=r"(q_frags[2]), "=r"(q_frags[3])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[8]), "=r"(q_frags[9]), "=r"(q_frags[10]), "=r"(q_frags[11])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[12]), "=r"(q_frags[13]), "=r"(q_frags[14]), "=r"(q_frags[15])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        #pragma unroll 1
        for (int kb = kb_first; kb < kb_last + 1; kb++) {
            mbarrier_wait(k_full_addr + (kv_stage) * 8, _phase_k_full);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[0]), "=f"(s_acc[1]), "=f"(s_acc[2]), "=f"(s_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[4]), "=f"(s_acc[(4) + 1]), "=f"(s_acc[(4) + 2]), "=f"(s_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[8]), "=f"(s_acc[(8) + 1]), "=f"(s_acc[(8) + 2]), "=f"(s_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[12]), "=f"(s_acc[(12) + 1]), "=f"(s_acc[(12) + 2]), "=f"(s_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[16]), "=f"(s_acc[(16) + 1]), "=f"(s_acc[(16) + 2]), "=f"(s_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[20]), "=f"(s_acc[(20) + 1]), "=f"(s_acc[(20) + 2]), "=f"(s_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[24]), "=f"(s_acc[(24) + 1]), "=f"(s_acc[(24) + 2]), "=f"(s_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[28]), "=f"(s_acc[(28) + 1]), "=f"(s_acc[(28) + 2]), "=f"(s_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[32]), "=f"(s_acc[(32) + 1]), "=f"(s_acc[(32) + 2]), "=f"(s_acc[(32) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[36]), "=f"(s_acc[(36) + 1]), "=f"(s_acc[(36) + 2]), "=f"(s_acc[(36) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[40]), "=f"(s_acc[(40) + 1]), "=f"(s_acc[(40) + 2]), "=f"(s_acc[(40) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[44]), "=f"(s_acc[(44) + 1]), "=f"(s_acc[(44) + 2]), "=f"(s_acc[(44) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[48]), "=f"(s_acc[(48) + 1]), "=f"(s_acc[(48) + 2]), "=f"(s_acc[(48) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[52]), "=f"(s_acc[(52) + 1]), "=f"(s_acc[(52) + 2]), "=f"(s_acc[(52) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[56]), "=f"(s_acc[(56) + 1]), "=f"(s_acc[(56) + 2]), "=f"(s_acc[(56) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[60]), "=f"(s_acc[(60) + 1]), "=f"(s_acc[(60) + 2]), "=f"(s_acc[(60) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
            }
            if (kb_first == kb || kb_last == kb) {
                s_acc[0] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[0]);
                s_acc[1] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[1]);
                s_acc[2] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[2]);
                s_acc[3] = ((seg_begin_v > kb * 128 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[3]);
                s_acc[4] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[4]);
                s_acc[5] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[5]);
                s_acc[6] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[6]);
                s_acc[7] = ((seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[7]);
                s_acc[8] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[8]);
                s_acc[9] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[9]);
                s_acc[10] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[10]);
                s_acc[11] = ((seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[11]);
                s_acc[12] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[12]);
                s_acc[13] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[13]);
                s_acc[14] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[14]);
                s_acc[15] = ((seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[15]);
                s_acc[16] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[16]);
                s_acc[17] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[17]);
                s_acc[18] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[18]);
                s_acc[19] = ((seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[19]);
                s_acc[20] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[20]);
                s_acc[21] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[21]);
                s_acc[22] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[22]);
                s_acc[23] = ((seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[23]);
                s_acc[24] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[24]);
                s_acc[25] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[25]);
                s_acc[26] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[26]);
                s_acc[27] = ((seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[27]);
                s_acc[28] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[28]);
                s_acc[29] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[29]);
                s_acc[30] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[30]);
                s_acc[31] = ((seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[31]);
                s_acc[32] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[32]);
                s_acc[33] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[33]);
                s_acc[34] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[34]);
                s_acc[35] = ((seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[35]);
                s_acc[36] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[36]);
                s_acc[37] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[37]);
                s_acc[38] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[38]);
                s_acc[39] = ((seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[39]);
                s_acc[40] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[40]);
                s_acc[41] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[41]);
                s_acc[42] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[42]);
                s_acc[43] = ((seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[43]);
                s_acc[44] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[44]);
                s_acc[45] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[45]);
                s_acc[46] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[46]);
                s_acc[47] = ((seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[47]);
                s_acc[48] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[48]);
                s_acc[49] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[49]);
                s_acc[50] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[50]);
                s_acc[51] = ((seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[51]);
                s_acc[52] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[52]);
                s_acc[53] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[53]);
                s_acc[54] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[54]);
                s_acc[55] = ((seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[55]);
                s_acc[56] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[56]);
                s_acc[57] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[57]);
                s_acc[58] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[58]);
                s_acc[59] = ((seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[59]);
                s_acc[60] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[60]);
                s_acc[61] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[61]);
                s_acc[62] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -CUDART_INF_F : s_acc[62]);
                s_acc[63] = ((seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1 || seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_acc[63]);
            }
            float _max_0 = fmaxf(s_acc[0], s_acc[1]);
            float _max_1 = fmaxf(s_acc[2], s_acc[3]);
            float _max_2 = fmaxf(_max_0, fmaxf(s_acc[4], s_acc[5]));
            float _max_3 = fmaxf(_max_1, fmaxf(s_acc[6], s_acc[7]));
            float _max_4 = fmaxf(_max_2, fmaxf(s_acc[8], s_acc[9]));
            float _max_5 = fmaxf(_max_3, fmaxf(s_acc[10], s_acc[11]));
            float _max_6 = fmaxf(_max_4, fmaxf(s_acc[12], s_acc[13]));
            float _max_7 = fmaxf(_max_5, fmaxf(s_acc[14], s_acc[15]));
            float _max_8 = fmaxf(_max_6, fmaxf(s_acc[16], s_acc[17]));
            float _max_9 = fmaxf(_max_7, fmaxf(s_acc[18], s_acc[19]));
            float _max_10 = fmaxf(_max_8, fmaxf(s_acc[20], s_acc[21]));
            float _max_11 = fmaxf(_max_9, fmaxf(s_acc[22], s_acc[23]));
            float _max_12 = fmaxf(_max_10, fmaxf(s_acc[24], s_acc[25]));
            float _max_13 = fmaxf(_max_11, fmaxf(s_acc[26], s_acc[27]));
            float _max_14 = fmaxf(_max_12, fmaxf(s_acc[28], s_acc[29]));
            float _max_15 = fmaxf(_max_13, fmaxf(s_acc[30], s_acc[31]));
            float _max_16 = fmaxf(_max_14, fmaxf(s_acc[32], s_acc[33]));
            float _max_17 = fmaxf(_max_15, fmaxf(s_acc[34], s_acc[35]));
            float _max_18 = fmaxf(_max_16, fmaxf(s_acc[36], s_acc[37]));
            float _max_19 = fmaxf(_max_17, fmaxf(s_acc[38], s_acc[39]));
            float _max_20 = fmaxf(_max_18, fmaxf(s_acc[40], s_acc[41]));
            float _max_21 = fmaxf(_max_19, fmaxf(s_acc[42], s_acc[43]));
            float _max_22 = fmaxf(_max_20, fmaxf(s_acc[44], s_acc[45]));
            float _max_23 = fmaxf(_max_21, fmaxf(s_acc[46], s_acc[47]));
            float _max_24 = fmaxf(_max_22, fmaxf(s_acc[48], s_acc[49]));
            float _max_25 = fmaxf(_max_23, fmaxf(s_acc[50], s_acc[51]));
            float _max_26 = fmaxf(_max_24, fmaxf(s_acc[52], s_acc[53]));
            float _max_27 = fmaxf(_max_25, fmaxf(s_acc[54], s_acc[55]));
            float _max_28 = fmaxf(_max_26, fmaxf(s_acc[56], s_acc[57]));
            float _max_29 = fmaxf(_max_27, fmaxf(s_acc[58], s_acc[59]));
            float _max_30 = fmaxf(_max_28, fmaxf(s_acc[60], s_acc[61]));
            float _max_31 = fmaxf(_max_29, fmaxf(s_acc[62], s_acc[63]));
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, _max_30, 1);
            float _max_32 = fmaxf(_max_30, _shfl_xor_sync_3);
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, _max_32, 2);
            float _max_33 = fmaxf(_max_32, _shfl_xor_sync_4);
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, _max_31, 1);
            float _max_34 = fmaxf(_max_31, _shfl_xor_sync_5);
            float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, _max_34, 2);
            float _max_35 = fmaxf(_max_34, _shfl_xor_sync_6);
            float _max_36 = fmaxf(m_state[0], _max_33 * (c_row0 * k_scale[head * num_kblocks + kb]));
            float _max_37 = fmaxf(m_state[1], _max_35 * (c_row1 * k_scale[head * num_kblocks + kb]));
            float _exp2_0 = approx_exp2(m_state[0] - _max_36);
            float _exp2_1 = approx_exp2(m_state[1] - _max_37);
            m_state[0] = _max_36;
            m_state[1] = _max_37;
            {
                float2 _pair_scale_even2_7 = make_float2(_exp2_0, _exp2_0);
                float2 _pair_scale_odd2_7 = make_float2(_exp2_1, _exp2_1);
                float2* _pair_scale_src2_7 = reinterpret_cast<float2*>(&o_acc[0]);
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
            float _exp2_2 = approx_exp2(__fmaf_rn(s_acc[0], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[0] = _exp2_2;
            float _exp2_3 = approx_exp2(__fmaf_rn(s_acc[1], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[1] = _exp2_3;
            float _exp2_4 = approx_exp2(__fmaf_rn(s_acc[2], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[2] = _exp2_4;
            float _exp2_5 = approx_exp2(__fmaf_rn(s_acc[3], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[3] = _exp2_5;
            float _exp2_6 = approx_exp2(__fmaf_rn(s_acc[4], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[4] = _exp2_6;
            float _exp2_7 = approx_exp2(__fmaf_rn(s_acc[5], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[5] = _exp2_7;
            float _exp2_8 = approx_exp2(__fmaf_rn(s_acc[6], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[6] = _exp2_8;
            float _exp2_9 = approx_exp2(__fmaf_rn(s_acc[7], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[7] = _exp2_9;
            float _exp2_10 = approx_exp2(__fmaf_rn(s_acc[8], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[8] = _exp2_10;
            float _exp2_11 = approx_exp2(__fmaf_rn(s_acc[9], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[9] = _exp2_11;
            float _exp2_12 = approx_exp2(__fmaf_rn(s_acc[10], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[10] = _exp2_12;
            float _exp2_13 = approx_exp2(__fmaf_rn(s_acc[11], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[11] = _exp2_13;
            float _exp2_14 = approx_exp2(__fmaf_rn(s_acc[12], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[12] = _exp2_14;
            float _exp2_15 = approx_exp2(__fmaf_rn(s_acc[13], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[13] = _exp2_15;
            float _exp2_16 = approx_exp2(__fmaf_rn(s_acc[14], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[14] = _exp2_16;
            float _exp2_17 = approx_exp2(__fmaf_rn(s_acc[15], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[15] = _exp2_17;
            float _exp2_18 = approx_exp2(__fmaf_rn(s_acc[16], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[16] = _exp2_18;
            float _exp2_19 = approx_exp2(__fmaf_rn(s_acc[17], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[17] = _exp2_19;
            float _exp2_20 = approx_exp2(__fmaf_rn(s_acc[18], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[18] = _exp2_20;
            float _exp2_21 = approx_exp2(__fmaf_rn(s_acc[19], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[19] = _exp2_21;
            float _exp2_22 = approx_exp2(__fmaf_rn(s_acc[20], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[20] = _exp2_22;
            float _exp2_23 = approx_exp2(__fmaf_rn(s_acc[21], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[21] = _exp2_23;
            float _exp2_24 = approx_exp2(__fmaf_rn(s_acc[22], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[22] = _exp2_24;
            float _exp2_25 = approx_exp2(__fmaf_rn(s_acc[23], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[23] = _exp2_25;
            float _exp2_26 = approx_exp2(__fmaf_rn(s_acc[24], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[24] = _exp2_26;
            float _exp2_27 = approx_exp2(__fmaf_rn(s_acc[25], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[25] = _exp2_27;
            float _exp2_28 = approx_exp2(__fmaf_rn(s_acc[26], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[26] = _exp2_28;
            float _exp2_29 = approx_exp2(__fmaf_rn(s_acc[27], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[27] = _exp2_29;
            float _exp2_30 = approx_exp2(__fmaf_rn(s_acc[28], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[28] = _exp2_30;
            float _exp2_31 = approx_exp2(__fmaf_rn(s_acc[29], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[29] = _exp2_31;
            float _exp2_32 = approx_exp2(__fmaf_rn(s_acc[30], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[30] = _exp2_32;
            float _exp2_33 = approx_exp2(__fmaf_rn(s_acc[31], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[31] = _exp2_33;
            float _exp2_34 = approx_exp2(__fmaf_rn(s_acc[32], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[32] = _exp2_34;
            float _exp2_35 = approx_exp2(__fmaf_rn(s_acc[33], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[33] = _exp2_35;
            float _exp2_36 = approx_exp2(__fmaf_rn(s_acc[34], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[34] = _exp2_36;
            float _exp2_37 = approx_exp2(__fmaf_rn(s_acc[35], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[35] = _exp2_37;
            float _exp2_38 = approx_exp2(__fmaf_rn(s_acc[36], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[36] = _exp2_38;
            float _exp2_39 = approx_exp2(__fmaf_rn(s_acc[37], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[37] = _exp2_39;
            float _exp2_40 = approx_exp2(__fmaf_rn(s_acc[38], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[38] = _exp2_40;
            float _exp2_41 = approx_exp2(__fmaf_rn(s_acc[39], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[39] = _exp2_41;
            float _exp2_42 = approx_exp2(__fmaf_rn(s_acc[40], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[40] = _exp2_42;
            float _exp2_43 = approx_exp2(__fmaf_rn(s_acc[41], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[41] = _exp2_43;
            float _exp2_44 = approx_exp2(__fmaf_rn(s_acc[42], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[42] = _exp2_44;
            float _exp2_45 = approx_exp2(__fmaf_rn(s_acc[43], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[43] = _exp2_45;
            float _exp2_46 = approx_exp2(__fmaf_rn(s_acc[44], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[44] = _exp2_46;
            float _exp2_47 = approx_exp2(__fmaf_rn(s_acc[45], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[45] = _exp2_47;
            float _exp2_48 = approx_exp2(__fmaf_rn(s_acc[46], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[46] = _exp2_48;
            float _exp2_49 = approx_exp2(__fmaf_rn(s_acc[47], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[47] = _exp2_49;
            float _exp2_50 = approx_exp2(__fmaf_rn(s_acc[48], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[48] = _exp2_50;
            float _exp2_51 = approx_exp2(__fmaf_rn(s_acc[49], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[49] = _exp2_51;
            float _exp2_52 = approx_exp2(__fmaf_rn(s_acc[50], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[50] = _exp2_52;
            float _exp2_53 = approx_exp2(__fmaf_rn(s_acc[51], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[51] = _exp2_53;
            float _exp2_54 = approx_exp2(__fmaf_rn(s_acc[52], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[52] = _exp2_54;
            float _exp2_55 = approx_exp2(__fmaf_rn(s_acc[53], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[53] = _exp2_55;
            float _exp2_56 = approx_exp2(__fmaf_rn(s_acc[54], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[54] = _exp2_56;
            float _exp2_57 = approx_exp2(__fmaf_rn(s_acc[55], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[55] = _exp2_57;
            float _exp2_58 = approx_exp2(__fmaf_rn(s_acc[56], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[56] = _exp2_58;
            float _exp2_59 = approx_exp2(__fmaf_rn(s_acc[57], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[57] = _exp2_59;
            float _exp2_60 = approx_exp2(__fmaf_rn(s_acc[58], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[58] = _exp2_60;
            float _exp2_61 = approx_exp2(__fmaf_rn(s_acc[59], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[59] = _exp2_61;
            float _exp2_62 = approx_exp2(__fmaf_rn(s_acc[60], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[60] = _exp2_62;
            float _exp2_63 = approx_exp2(__fmaf_rn(s_acc[61], c_row0 * k_scale[head * num_kblocks + kb], 8.0f - _max_36));
            s_acc[61] = _exp2_63;
            float _exp2_64 = approx_exp2(__fmaf_rn(s_acc[62], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[62] = _exp2_64;
            float _exp2_65 = approx_exp2(__fmaf_rn(s_acc[63], c_row1 * k_scale[head * num_kblocks + kb], 8.0f - _max_37));
            s_acc[63] = _exp2_65;
            l_state[0] = __fmaf_rn(l_state[0], _exp2_0, _exp2_2 + _exp2_3 + _exp2_6 + _exp2_7 + _exp2_10 + _exp2_11 + _exp2_14 + _exp2_15 + _exp2_18 + _exp2_19 + _exp2_22 + _exp2_23 + _exp2_26 + _exp2_27 + _exp2_30 + _exp2_31 + _exp2_34 + _exp2_35 + _exp2_38 + _exp2_39 + _exp2_42 + _exp2_43 + _exp2_46 + _exp2_47 + _exp2_50 + _exp2_51 + _exp2_54 + _exp2_55 + _exp2_58 + _exp2_59 + _exp2_62 + _exp2_63);
            l_state[1] = __fmaf_rn(l_state[1], _exp2_1, _exp2_4 + _exp2_5 + _exp2_8 + _exp2_9 + _exp2_12 + _exp2_13 + _exp2_16 + _exp2_17 + _exp2_20 + _exp2_21 + _exp2_24 + _exp2_25 + _exp2_28 + _exp2_29 + _exp2_32 + _exp2_33 + _exp2_36 + _exp2_37 + _exp2_40 + _exp2_41 + _exp2_44 + _exp2_45 + _exp2_48 + _exp2_49 + _exp2_52 + _exp2_53 + _exp2_56 + _exp2_57 + _exp2_60 + _exp2_61 + _exp2_64 + _exp2_65);
            p_tmp[0] = s_acc[0];
            p_tmp[1] = s_acc[1];
            p_tmp[2] = s_acc[4];
            p_tmp[3] = s_acc[5];
            p_tmp[4] = s_acc[2];
            p_tmp[5] = s_acc[3];
            p_tmp[6] = s_acc[6];
            p_tmp[7] = s_acc[7];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[0] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[1] = _packed;
            }
            p_tmp[0] = s_acc[8];
            p_tmp[1] = s_acc[9];
            p_tmp[2] = s_acc[12];
            p_tmp[3] = s_acc[13];
            p_tmp[4] = s_acc[10];
            p_tmp[5] = s_acc[11];
            p_tmp[6] = s_acc[14];
            p_tmp[7] = s_acc[15];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[2] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[3] = _packed;
            }
            p_tmp[0] = s_acc[16];
            p_tmp[1] = s_acc[17];
            p_tmp[2] = s_acc[20];
            p_tmp[3] = s_acc[21];
            p_tmp[4] = s_acc[18];
            p_tmp[5] = s_acc[19];
            p_tmp[6] = s_acc[22];
            p_tmp[7] = s_acc[23];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[4] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[5] = _packed;
            }
            p_tmp[0] = s_acc[24];
            p_tmp[1] = s_acc[25];
            p_tmp[2] = s_acc[28];
            p_tmp[3] = s_acc[29];
            p_tmp[4] = s_acc[26];
            p_tmp[5] = s_acc[27];
            p_tmp[6] = s_acc[30];
            p_tmp[7] = s_acc[31];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[6] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[7] = _packed;
            }
            p_tmp[0] = s_acc[32];
            p_tmp[1] = s_acc[33];
            p_tmp[2] = s_acc[36];
            p_tmp[3] = s_acc[37];
            p_tmp[4] = s_acc[34];
            p_tmp[5] = s_acc[35];
            p_tmp[6] = s_acc[38];
            p_tmp[7] = s_acc[39];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[8] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[9] = _packed;
            }
            p_tmp[0] = s_acc[40];
            p_tmp[1] = s_acc[41];
            p_tmp[2] = s_acc[44];
            p_tmp[3] = s_acc[45];
            p_tmp[4] = s_acc[42];
            p_tmp[5] = s_acc[43];
            p_tmp[6] = s_acc[46];
            p_tmp[7] = s_acc[47];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[10] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[11] = _packed;
            }
            p_tmp[0] = s_acc[48];
            p_tmp[1] = s_acc[49];
            p_tmp[2] = s_acc[52];
            p_tmp[3] = s_acc[53];
            p_tmp[4] = s_acc[50];
            p_tmp[5] = s_acc[51];
            p_tmp[6] = s_acc[54];
            p_tmp[7] = s_acc[55];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[12] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[13] = _packed;
            }
            p_tmp[0] = s_acc[56];
            p_tmp[1] = s_acc[57];
            p_tmp[2] = s_acc[60];
            p_tmp[3] = s_acc[61];
            p_tmp[4] = s_acc[58];
            p_tmp[5] = s_acc[59];
            p_tmp[6] = s_acc[62];
            p_tmp[7] = s_acc[63];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[0]), "f"(p_tmp[1]),
                                       "f"(p_tmp[2]), "f"(p_tmp[3]));
                p_frag[14] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(p_tmp[4]), "f"(p_tmp[5]),
                                       "f"(p_tmp[6]), "f"(p_tmp[7]));
                p_frag[15] = _packed;
            }
            mbarrier_wait(v_full_addr + (kv_stage) * 8, _phase_k_full);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[8]), "r"(p_frag[(8) + 1]), "r"(p_frag[(8) + 2]), "r"(p_frag[(8) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(V_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[12]), "r"(p_frag[(12) + 1]), "r"(p_frag[(12) + 2]), "r"(p_frag[(12) + 3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(fp8_unit_sfa_main[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_main[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(v_empty_addr + (kv_stage) * 8);
            }
            if (warp == 0) {
                mbarrier_wait(k_empty_addr + (kv_stage) * 8, _phase_k_full);
            }
            if (kb_last >= kb + 2) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + kv_stage * 16384), "l"((&K_map)), "r"(0), "r"((kb + 2) * 128), "r"(head),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (kb == kb_last && has_next == 1) {
                if (warp == 0) {
                    if (next_head != head) {
                        if (elect_sync()) {
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + next_head))) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(1));
                            }
                        }
                        __syncwarp();
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                }
                if (warp == 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                if (tid == 0) {
                    slot_cell[it & 1] = (int)nn_r[0];
                }
                if (warp == 0) {
                    __syncwarp();
                }
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(next_q_row0), "r"(next_head),
                               "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&K_map)), "r"(0), "r"(next_kb_first * 128), "r"(next_head),
                               "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                if (next_kb_last >= next_kb_first + 1) {
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 16384);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(K_stage_addr + kv_stage * 16384), "l"((&K_map)), "r"(0), "r"((next_kb_first + 1) * 128), "r"(next_head),
                                   "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
            }
            if (warp == 4) {
                mbarrier_wait(v_empty_addr + (kv_stage) * 8, _phase_k_full);
            }
            if (kb_last >= kb + 2) {
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"((kb + 2) * 128), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (kb == kb_last && has_next == 1) {
                if (warp == 4) {
                    if (next_head != head) {
                        if (elect_sync()) {
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + next_head))) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(1));
                            }
                        }
                        __syncwarp();
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                }
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(V_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&V_map)), "r"(next_kb_first * 128), "r"(0), "r"(next_head),
                               "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                if (next_kb_last >= next_kb_first + 1) {
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 16384);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(V_stage_addr + kv_stage * 16384), "l"((&V_map)), "r"((next_kb_first + 1) * 128), "r"(0), "r"(next_head),
                                   "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 2) { kv_stage = 0; _phase_k_full ^= 1; }
        }
        float l0 = l_state[0];
        float l1 = l_state[1];
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, l0, 1);
        l0 = l0 + _shfl_xor_sync_8;
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, l0, 2);
        l0 = l0 + _shfl_xor_sync_9;
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, l1, 1);
        l1 = l1 + _shfl_xor_sync_10;
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, l1, 2);
        l1 = l1 + _shfl_xor_sync_11;
        float inv0 = approx_rcp(l0);
        float inv1 = approx_rcp(l1);
        float _vec_load_0[2];
        {
            float2 _v2_12 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + (lane & 3) * 2) + 0);
            _vec_load_0[0] = _v2_12.x;
            _vec_load_0[0 + 1] = _v2_12.y;
        }
        o_tmp[0] = o_acc[0] * inv0 * _vec_load_0[0];
        o_tmp[1] = o_acc[1] * inv0 * _vec_load_0[1];
        o_tmp[2] = o_acc[2] * inv1 * _vec_load_0[0];
        o_tmp[3] = o_acc[3] * inv1 * _vec_load_0[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_0[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2))[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_0[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_0[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2))[0] = reinterpret_cast<unsigned*>(_packed_32_0)[0];
        }
        float _vec_load_1[2];
        {
            float2 _v2_13 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 8 + (lane & 3) * 2) + 0);
            _vec_load_1[0] = _v2_13.x;
            _vec_load_1[0 + 1] = _v2_13.y;
        }
        o_tmp[0] = o_acc[4] * inv0 * _vec_load_1[0];
        o_tmp[1] = o_acc[5] * inv0 * _vec_load_1[1];
        o_tmp[2] = o_acc[6] * inv1 * _vec_load_1[0];
        o_tmp[3] = o_acc[7] * inv1 * _vec_load_1[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 8))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 8))[0] = reinterpret_cast<unsigned*>(_packed_32_1)[0];
        }
        float _vec_load_2[2];
        {
            float2 _v2_14 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 16 + (lane & 3) * 2) + 0);
            _vec_load_2[0] = _v2_14.x;
            _vec_load_2[0 + 1] = _v2_14.y;
        }
        o_tmp[0] = o_acc[8] * inv0 * _vec_load_2[0];
        o_tmp[1] = o_acc[9] * inv0 * _vec_load_2[1];
        o_tmp[2] = o_acc[10] * inv1 * _vec_load_2[0];
        o_tmp[3] = o_acc[11] * inv1 * _vec_load_2[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_2[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 16))[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_2[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 16))[0] = reinterpret_cast<unsigned*>(_packed_32_2)[0];
        }
        float _vec_load_3[2];
        {
            float2 _v2_15 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 24 + (lane & 3) * 2) + 0);
            _vec_load_3[0] = _v2_15.x;
            _vec_load_3[0 + 1] = _v2_15.y;
        }
        o_tmp[0] = o_acc[12] * inv0 * _vec_load_3[0];
        o_tmp[1] = o_acc[13] * inv0 * _vec_load_3[1];
        o_tmp[2] = o_acc[14] * inv1 * _vec_load_3[0];
        o_tmp[3] = o_acc[15] * inv1 * _vec_load_3[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_3[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 24))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_3[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 24))[0] = reinterpret_cast<unsigned*>(_packed_32_3)[0];
        }
        float _vec_load_4[2];
        {
            float2 _v2_16 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 32 + (lane & 3) * 2) + 0);
            _vec_load_4[0] = _v2_16.x;
            _vec_load_4[0 + 1] = _v2_16.y;
        }
        o_tmp[0] = o_acc[16] * inv0 * _vec_load_4[0];
        o_tmp[1] = o_acc[17] * inv0 * _vec_load_4[1];
        o_tmp[2] = o_acc[18] * inv1 * _vec_load_4[0];
        o_tmp[3] = o_acc[19] * inv1 * _vec_load_4[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_4[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_4[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_4[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_32_4)[0];
        }
        float _vec_load_5[2];
        {
            float2 _v2_17 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 40 + (lane & 3) * 2) + 0);
            _vec_load_5[0] = _v2_17.x;
            _vec_load_5[0 + 1] = _v2_17.y;
        }
        o_tmp[0] = o_acc[20] * inv0 * _vec_load_5[0];
        o_tmp[1] = o_acc[21] * inv0 * _vec_load_5[1];
        o_tmp[2] = o_acc[22] * inv1 * _vec_load_5[0];
        o_tmp[3] = o_acc[23] * inv1 * _vec_load_5[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_5[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_5[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 40))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_5[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_5[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 40))[0] = reinterpret_cast<unsigned*>(_packed_32_5)[0];
        }
        float _vec_load_6[2];
        {
            float2 _v2_18 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 48 + (lane & 3) * 2) + 0);
            _vec_load_6[0] = _v2_18.x;
            _vec_load_6[0 + 1] = _v2_18.y;
        }
        o_tmp[0] = o_acc[24] * inv0 * _vec_load_6[0];
        o_tmp[1] = o_acc[25] * inv0 * _vec_load_6[1];
        o_tmp[2] = o_acc[26] * inv1 * _vec_load_6[0];
        o_tmp[3] = o_acc[27] * inv1 * _vec_load_6[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_6[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_6[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_6)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_6[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_6[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_32_6)[0];
        }
        float _vec_load_7[2];
        {
            float2 _v2_19 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 56 + (lane & 3) * 2) + 0);
            _vec_load_7[0] = _v2_19.x;
            _vec_load_7[0 + 1] = _v2_19.y;
        }
        o_tmp[0] = o_acc[28] * inv0 * _vec_load_7[0];
        o_tmp[1] = o_acc[29] * inv0 * _vec_load_7[1];
        o_tmp[2] = o_acc[30] * inv1 * _vec_load_7[0];
        o_tmp[3] = o_acc[31] * inv1 * _vec_load_7[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_7[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_7[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 56))[0] = reinterpret_cast<unsigned*>(_packed_7)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_7[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_7[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 56))[0] = reinterpret_cast<unsigned*>(_packed_32_7)[0];
        }
        float _vec_load_8[2];
        {
            float2 _v2_20 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 64 + (lane & 3) * 2) + 0);
            _vec_load_8[0] = _v2_20.x;
            _vec_load_8[0 + 1] = _v2_20.y;
        }
        o_tmp[0] = o_acc[32] * inv0 * _vec_load_8[0];
        o_tmp[1] = o_acc[33] * inv0 * _vec_load_8[1];
        o_tmp[2] = o_acc[34] * inv1 * _vec_load_8[0];
        o_tmp[3] = o_acc[35] * inv1 * _vec_load_8[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_8[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_8[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 64))[0] = reinterpret_cast<unsigned*>(_packed_8)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_8[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_8[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 64))[0] = reinterpret_cast<unsigned*>(_packed_32_8)[0];
        }
        float _vec_load_9[2];
        {
            float2 _v2_21 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 72 + (lane & 3) * 2) + 0);
            _vec_load_9[0] = _v2_21.x;
            _vec_load_9[0 + 1] = _v2_21.y;
        }
        o_tmp[0] = o_acc[36] * inv0 * _vec_load_9[0];
        o_tmp[1] = o_acc[37] * inv0 * _vec_load_9[1];
        o_tmp[2] = o_acc[38] * inv1 * _vec_load_9[0];
        o_tmp[3] = o_acc[39] * inv1 * _vec_load_9[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_9[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_9[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 72))[0] = reinterpret_cast<unsigned*>(_packed_9)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_9[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_9[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 72))[0] = reinterpret_cast<unsigned*>(_packed_32_9)[0];
        }
        float _vec_load_10[2];
        {
            float2 _v2_22 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 80 + (lane & 3) * 2) + 0);
            _vec_load_10[0] = _v2_22.x;
            _vec_load_10[0 + 1] = _v2_22.y;
        }
        o_tmp[0] = o_acc[40] * inv0 * _vec_load_10[0];
        o_tmp[1] = o_acc[41] * inv0 * _vec_load_10[1];
        o_tmp[2] = o_acc[42] * inv1 * _vec_load_10[0];
        o_tmp[3] = o_acc[43] * inv1 * _vec_load_10[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_10[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_10[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 80))[0] = reinterpret_cast<unsigned*>(_packed_10)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_10[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_10[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 80))[0] = reinterpret_cast<unsigned*>(_packed_32_10)[0];
        }
        float _vec_load_11[2];
        {
            float2 _v2_23 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 88 + (lane & 3) * 2) + 0);
            _vec_load_11[0] = _v2_23.x;
            _vec_load_11[0 + 1] = _v2_23.y;
        }
        o_tmp[0] = o_acc[44] * inv0 * _vec_load_11[0];
        o_tmp[1] = o_acc[45] * inv0 * _vec_load_11[1];
        o_tmp[2] = o_acc[46] * inv1 * _vec_load_11[0];
        o_tmp[3] = o_acc[47] * inv1 * _vec_load_11[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_11[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_11[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 88))[0] = reinterpret_cast<unsigned*>(_packed_11)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_11[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_11[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 88))[0] = reinterpret_cast<unsigned*>(_packed_32_11)[0];
        }
        float _vec_load_12[2];
        {
            float2 _v2_24 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 96 + (lane & 3) * 2) + 0);
            _vec_load_12[0] = _v2_24.x;
            _vec_load_12[0 + 1] = _v2_24.y;
        }
        o_tmp[0] = o_acc[48] * inv0 * _vec_load_12[0];
        o_tmp[1] = o_acc[49] * inv0 * _vec_load_12[1];
        o_tmp[2] = o_acc[50] * inv1 * _vec_load_12[0];
        o_tmp[3] = o_acc[51] * inv1 * _vec_load_12[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_12[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_12[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 96))[0] = reinterpret_cast<unsigned*>(_packed_12)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_12[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_12[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 96))[0] = reinterpret_cast<unsigned*>(_packed_32_12)[0];
        }
        float _vec_load_13[2];
        {
            float2 _v2_25 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 104 + (lane & 3) * 2) + 0);
            _vec_load_13[0] = _v2_25.x;
            _vec_load_13[0 + 1] = _v2_25.y;
        }
        o_tmp[0] = o_acc[52] * inv0 * _vec_load_13[0];
        o_tmp[1] = o_acc[53] * inv0 * _vec_load_13[1];
        o_tmp[2] = o_acc[54] * inv1 * _vec_load_13[0];
        o_tmp[3] = o_acc[55] * inv1 * _vec_load_13[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_13[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_13[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 104))[0] = reinterpret_cast<unsigned*>(_packed_13)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_13[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_13[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 104))[0] = reinterpret_cast<unsigned*>(_packed_32_13)[0];
        }
        float _vec_load_14[2];
        {
            float2 _v2_26 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 112 + (lane & 3) * 2) + 0);
            _vec_load_14[0] = _v2_26.x;
            _vec_load_14[0 + 1] = _v2_26.y;
        }
        o_tmp[0] = o_acc[56] * inv0 * _vec_load_14[0];
        o_tmp[1] = o_acc[57] * inv0 * _vec_load_14[1];
        o_tmp[2] = o_acc[58] * inv1 * _vec_load_14[0];
        o_tmp[3] = o_acc[59] * inv1 * _vec_load_14[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_14[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_14[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 112))[0] = reinterpret_cast<unsigned*>(_packed_14)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_14[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_14[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 112))[0] = reinterpret_cast<unsigned*>(_packed_32_14)[0];
        }
        float _vec_load_15[2];
        {
            float2 _v2_27 = *reinterpret_cast<const float2*>(v_scale + ((seg * num_heads + head) * 128 + 120 + (lane & 3) * 2) + 0);
            _vec_load_15[0] = _v2_27.x;
            _vec_load_15[0 + 1] = _v2_27.y;
        }
        o_tmp[0] = o_acc[60] * inv0 * _vec_load_15[0];
        o_tmp[1] = o_acc[61] * inv0 * _vec_load_15[1];
        o_tmp[2] = o_acc[62] * inv1 * _vec_load_15[0];
        o_tmp[3] = o_acc[63] * inv1 * _vec_load_15[1];
        if (row0 < seg_end_v) {
            unsigned int _packed_15[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_15[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row0 * num_heads + head) * 128 + (lane & 3) * 2 + 120))[0] = reinterpret_cast<unsigned*>(_packed_15)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_32_15[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_32_15[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + ((row1 * num_heads + head) * 128 + (lane & 3) * 2 + 120))[0] = reinterpret_cast<unsigned*>(_packed_32_15)[0];
        }
        cur = nslot;
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int eold;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(eold) : "l"(&hctr[4 * num_heads + 1]), "r"(static_cast<uint32_t>(1)) : "memory");
        if ((int)eold == lead_ctas + tail_ctas - 1) {
            *(reinterpret_cast<unsigned int*>(hctr + (4 * num_heads)) + (0)) = (unsigned int)0;
            *(reinterpret_cast<unsigned int*>(hctr + (4 * num_heads + 1)) + (0)) = (unsigned int)0;
            #pragma unroll 1
            for (int hh = 0; hh < num_heads; hh++) {
                *(reinterpret_cast<unsigned int*>(hctr + (3 * num_heads + hh)) + (0)) = (unsigned int)0;
            }
        }
    }
    if (tid == 0) {
        {
        unsigned int _acquire_observed;
        do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (4 * num_heads + 2))) : "memory");
        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(lead_ctas + 1)) >= static_cast<unsigned int>(1));
        }
        unsigned int pold;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(pold) : "l"(&hctr[4 * num_heads + 3]), "r"(static_cast<uint32_t>(1)) : "memory");
        if ((int)pold == tail_ctas - 1) {
            *(reinterpret_cast<unsigned int*>(hctr + (4 * num_heads + 2)) + (0)) = (unsigned int)0;
            *(reinterpret_cast<unsigned int*>(hctr + (4 * num_heads + 3)) + (0)) = (unsigned int)0;
        }
    }
    if (warp < 4) {
        asm volatile("barrier.sync 1, 256;" ::: "memory");
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
