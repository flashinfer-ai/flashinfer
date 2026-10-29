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
#define NUM_DELTA_STAGES 2
#define SMEM_Q_OFF 1024
#define SMEM_Q_STAGE_BYTES 8192
#define SMEM_Q_STRIDE 8192
#define SMEM_K_STAGE_OFF 9216
#define SMEM_K_STAGE_STAGE_BYTES 8192
#define SMEM_K_STAGE_STRIDE 8192
#define SMEM_KSF_STAGE_OFF 25600
#define SMEM_KSF_STAGE_STAGE_BYTES 1024
#define SMEM_KSF_STAGE_STRIDE 1024
#define SMEM_K_STAGE_0_OFF 27648
#define SMEM_K_STAGE_0_STAGE_BYTES 8192
#define SMEM_K_STAGE_0_STRIDE 8192
#define SMEM_KSF_STAGE_0_OFF 44032
#define SMEM_KSF_STAGE_0_STAGE_BYTES 1024
#define SMEM_KSF_STAGE_0_STRIDE 1024
#define SMEM_DELTA_OFF 46080
#define SMEM_DELTA_STAGE_BYTES 2048
#define SMEM_DELTA_STRIDE 2048
#define SMEM_SLOT_CELL_OFF 0
#define SMEM_SLOT_CELL_STAGE_BYTES 16
#define SMEM_SLOT_CELL_STRIDE 16
#define SMEM_TOTAL 48128
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
kernel_cake_minimax_h3_sm120_nvfp4_varlen_attention_2ec2cbea6d8f3ba7e22c(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap KSF_map, const __grid_constant__ CUtensorMap VT_map, const __grid_constant__ CUtensorMap VSF_map, __nv_bfloat16* O, unsigned int* q_sf, unsigned int* qm4, unsigned int* qm_sf, int* __restrict__ seg_tile_begin, int* __restrict__ unit_table, int total_units, int num_heads, int num_kblocks, float softmax_scale_log2, unsigned int* hctr, int units_per_head, int lead_ctas, int tail_ctas)
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
    #define delta_ready_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* Q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int Q_addr = smem + 1024;
    uint8_t* K_stage = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int K_stage_addr = smem + 9216;
    unsigned int* KSF_stage = reinterpret_cast<unsigned int*>(smem_raw + 25600);
    const int KSF_stage_addr = smem + 25600;
    uint8_t* K_stage_0 = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int K_stage_addr_0 = smem + 27648;
    unsigned int* KSF_stage_0 = reinterpret_cast<unsigned int*>(smem_raw + 44032);
    const int KSF_stage_addr_0 = smem + 44032;
    float* DELTA = reinterpret_cast<float*>(smem_raw + 46080);
    const int DELTA_addr = smem + 46080;
    int* slot_cell = reinterpret_cast<int*>(smem_static_raw + 0);
    const int slot_cell_addr = (int)(unsigned long long)__cvta_generic_to_shared(slot_cell);

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[0..88)

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
            // v_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 56, 256);
            mbarrier_init(smem + 64, 256);
            // --- pipeline 'delta' ---
            // delta_ready: 2 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    unsigned int kv_stage = 0;
    unsigned int kv_phase = 0;
    unsigned int d_stage = 0;
    unsigned int d_phase = 0;
    if (warp >= 4) {
        asm volatile("barrier.arrive 1, 256;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&hctr[5 * num_heads], 1);
        slot_cell[0] = (int)_atomic_fetch_add_0;
    }
    __syncthreads();
    unsigned int _phase_q_full_0 = 0;
    unsigned int _phase_k_full = 0;
    #pragma unroll 1
    for (int it = 0; it < total_units; it++) {
        int slot = slot_cell[it & 1];
        if (slot >= total_units) {
            break;
        }
        int _vec_load_0[4];
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(unit_table + (4 * slot) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            _vec_load_0[0 + 0] = _ivld_1.x;
            _vec_load_0[0 + 1] = _ivld_1.y;
            _vec_load_0[0 + 2] = _ivld_1.z;
            _vec_load_0[0 + 3] = _ivld_1.w;
        }
        int seg = _vec_load_0[0];
        int packed = _vec_load_0[1];
        int head = packed >> 16;
        int q_tile = packed & 65535;
        int seg_begin_v = _vec_load_0[2];
        int seg_end_v = _vec_load_0[3];
        int q_row0 = seg_begin_v + q_tile * 128;
        int kb_first = seg_begin_v / 128;
        int kb_last = (seg_end_v - 1) / 128;
        if (tid == 0) {
            unsigned int nxt = atomicAdd(&hctr[5 * num_heads], 1);
            {
            unsigned int _acquire_observed;
            do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(hctr) + (3 * num_heads + head))) : "memory");
            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(1));
            }
            unsigned int uold;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(uold) : "l"(&hctr[4 * num_heads + head]), "r"(static_cast<uint32_t>(1)) : "memory");
            if ((int)uold == units_per_head - 1) {
                *(reinterpret_cast<unsigned int*>(hctr + (3 * num_heads + head)) + (0)) = (unsigned int)0;
                *(reinterpret_cast<unsigned int*>(hctr + (4 * num_heads + head)) + (0)) = (unsigned int)0;
            }
            slot_cell[it + 1 & 1] = (int)nxt;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        asm volatile("fence.proxy.async.global;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 8192);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(q_row0), "r"(head),
                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        if (warp == 0) {
            if (elect_sync()) {
                if (kb_first <= kb_last) {
                    mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr + kv_stage * 8192), "l"((&K_map)), "r"(0), "r"(kb_first * 128), "r"(head),
                           "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr + kv_stage * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + kb_first) * 64),
                           "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr_0 + kv_stage * 8192), "l"((&VT_map)), "r"(kb_first * 64), "r"(0), "r"(head),
                           "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr_0 + kv_stage * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + kb_first) * 64),
                           "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        if (kb_last >= kb_first + 1) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 1) % 2) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr + (kv_stage + 1) % 2 * 8192), "l"((&K_map)), "r"(0), "r"((kb_first + 1) * 128), "r"(head),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr + (kv_stage + 1) % 2 * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 1)) * 64),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr_0 + (kv_stage + 1) % 2 * 8192), "l"((&VT_map)), "r"((kb_first + 1) * 64), "r"(0), "r"(head),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr_0 + (kv_stage + 1) % 2 * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 1)) * 64),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        int row0 = q_row0 + warp * 16 + (lane >> 2);
        int row1 = row0 + 8;
        int row0_c = ((row0 < seg_end_v) ? row0 : seg_end_v - 1);
        int row1_c = ((row1 < seg_end_v) ? row1 : seg_end_v - 1);
        int sf_row = (((lane & 1) == 1) ? row1_c : row0_c);
        unsigned int qsf[2];
        for (int u = 0; u < 2; u++) {
            qsf[u] = q_sf[(sf_row * num_heads + head) * 2 + u];
        }
        int qmrow = (seg_tile_begin[seg] + q_tile) * num_heads + head;
        unsigned int qm_frag0[4];
        unsigned int qm_frag1[4];
        unsigned int qmsf[2];
        unsigned int qmb0[2];
        unsigned int qmb1[2];
        qm_frag0[0] = ((lane >> 2 == 0) ? qm4[qmrow * 16 + (lane & 3)] : (unsigned int)0);
        qm_frag0[1] = (unsigned int)0;
        qm_frag0[2] = ((lane >> 2 == 0) ? qm4[qmrow * 16 + 4 + (lane & 3)] : (unsigned int)0);
        qm_frag0[3] = (unsigned int)0;
        qmsf[0] = qm_sf[qmrow * 2];
        qm_frag1[0] = ((lane >> 2 == 0) ? qm4[qmrow * 16 + 8 + (lane & 3)] : (unsigned int)0);
        qm_frag1[1] = (unsigned int)0;
        qm_frag1[2] = ((lane >> 2 == 0) ? qm4[qmrow * 16 + 8 + 4 + (lane & 3)] : (unsigned int)0);
        qm_frag1[3] = (unsigned int)0;
        qmsf[1] = qm_sf[qmrow * 2 + 1];
        unsigned int ka0[4];
        unsigned int ka1[4];
        unsigned int q_frags[8];
        unsigned int k_frag[8];
        unsigned int v_frag[8];
        float s_acc[64];
        float d_tmp[8];
        unsigned int p_frag[8];
        float p_tmp[32];
        unsigned int psf[2];
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
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[0]), "=r"(q_frags[1]), "=r"(q_frags[2]), "=r"(q_frags[3])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(Q_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
            : "memory");
        __syncthreads();
        #pragma unroll 1
        for (int kb = kb_first; kb < kb_last + 1; kb++) {
            mbarrier_wait(k_full_addr + (kv_stage) * 8, _phase_k_full);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs[2];
            #pragma unroll
            for (int _lr = 0; _lr < 2; _lr++) {
                smem_regs[_lr] = KSF_stage[kv_stage * 256 + (unsigned int)((0 + 8 * (lane & 3) + (lane >> 2)) * 2) + (unsigned int)_lr];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[0]), "=f"(s_acc[1]), "=f"(s_acc[2]), "=f"(s_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[4]), "=f"(s_acc[(4) + 1]), "=f"(s_acc[(4) + 2]), "=f"(s_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[1])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[8]), "=f"(s_acc[(8) + 1]), "=f"(s_acc[(8) + 2]), "=f"(s_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[1])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[12]), "=f"(s_acc[(12) + 1]), "=f"(s_acc[(12) + 2]), "=f"(s_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[1])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_0[2];
            #pragma unroll
            for (int _lr_1 = 0; _lr_1 < 2; _lr_1++) {
                smem_regs_0[_lr_1] = KSF_stage[kv_stage * 256 + (unsigned int)((32 + 8 * (lane & 3) + (lane >> 2)) * 2) + (unsigned int)_lr_1];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[16]), "=f"(s_acc[(16) + 1]), "=f"(s_acc[(16) + 2]), "=f"(s_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[20]), "=f"(s_acc[(20) + 1]), "=f"(s_acc[(20) + 2]), "=f"(s_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[1])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[24]), "=f"(s_acc[(24) + 1]), "=f"(s_acc[(24) + 2]), "=f"(s_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[1])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[28]), "=f"(s_acc[(28) + 1]), "=f"(s_acc[(28) + 2]), "=f"(s_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[1])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_1[2];
            #pragma unroll
            for (int _lr_2 = 0; _lr_2 < 2; _lr_2++) {
                smem_regs_1[_lr_2] = KSF_stage[kv_stage * 256 + (unsigned int)((64 + 8 * (lane & 3) + (lane >> 2)) * 2) + (unsigned int)_lr_2];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[32]), "=f"(s_acc[(32) + 1]), "=f"(s_acc[(32) + 2]), "=f"(s_acc[(32) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[36]), "=f"(s_acc[(36) + 1]), "=f"(s_acc[(36) + 2]), "=f"(s_acc[(36) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[1])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[40]), "=f"(s_acc[(40) + 1]), "=f"(s_acc[(40) + 2]), "=f"(s_acc[(40) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[1])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[44]), "=f"(s_acc[(44) + 1]), "=f"(s_acc[(44) + 2]), "=f"(s_acc[(44) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[1])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_2[2];
            #pragma unroll
            for (int _lr_3 = 0; _lr_3 < 2; _lr_3++) {
                smem_regs_2[_lr_3] = KSF_stage[kv_stage * 256 + (unsigned int)((96 + 8 * (lane & 3) + (lane >> 2)) * 2) + (unsigned int)_lr_3];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[48]), "=f"(s_acc[(48) + 1]), "=f"(s_acc[(48) + 2]), "=f"(s_acc[(48) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[52]), "=f"(s_acc[(52) + 1]), "=f"(s_acc[(52) + 2]), "=f"(s_acc[(52) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[1])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[56]), "=f"(s_acc[(56) + 1]), "=f"(s_acc[(56) + 2]), "=f"(s_acc[(56) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[1])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(s_acc[60]), "=f"(s_acc[(60) + 1]), "=f"(s_acc[(60) + 2]), "=f"(s_acc[(60) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[1])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
            }
            if (kb_first == kb || kb_last == kb) {
                if (seg_begin_v > kb * 128 + 2 * (lane & 3)) {
                    s_acc[0] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 2 * (lane & 3)) {
                    s_acc[0] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 2 * (lane & 3) + 1) {
                    s_acc[1] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) {
                    s_acc[1] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 2 * (lane & 3)) {
                    s_acc[2] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 2 * (lane & 3)) {
                    s_acc[2] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 2 * (lane & 3) + 1) {
                    s_acc[3] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) {
                    s_acc[3] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 8 + 2 * (lane & 3)) {
                    s_acc[4] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) {
                    s_acc[4] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1) {
                    s_acc[5] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) {
                    s_acc[5] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 8 + 2 * (lane & 3)) {
                    s_acc[6] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) {
                    s_acc[6] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 8 + 2 * (lane & 3) + 1) {
                    s_acc[7] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) {
                    s_acc[7] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 16 + 2 * (lane & 3)) {
                    s_acc[8] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) {
                    s_acc[8] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1) {
                    s_acc[9] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) {
                    s_acc[9] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 16 + 2 * (lane & 3)) {
                    s_acc[10] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) {
                    s_acc[10] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 16 + 2 * (lane & 3) + 1) {
                    s_acc[11] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) {
                    s_acc[11] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 24 + 2 * (lane & 3)) {
                    s_acc[12] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) {
                    s_acc[12] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1) {
                    s_acc[13] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) {
                    s_acc[13] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 24 + 2 * (lane & 3)) {
                    s_acc[14] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) {
                    s_acc[14] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 24 + 2 * (lane & 3) + 1) {
                    s_acc[15] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) {
                    s_acc[15] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 32 + 2 * (lane & 3)) {
                    s_acc[16] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) {
                    s_acc[16] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1) {
                    s_acc[17] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) {
                    s_acc[17] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 32 + 2 * (lane & 3)) {
                    s_acc[18] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) {
                    s_acc[18] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 32 + 2 * (lane & 3) + 1) {
                    s_acc[19] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) {
                    s_acc[19] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 40 + 2 * (lane & 3)) {
                    s_acc[20] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) {
                    s_acc[20] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1) {
                    s_acc[21] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) {
                    s_acc[21] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 40 + 2 * (lane & 3)) {
                    s_acc[22] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) {
                    s_acc[22] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 40 + 2 * (lane & 3) + 1) {
                    s_acc[23] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) {
                    s_acc[23] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 48 + 2 * (lane & 3)) {
                    s_acc[24] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) {
                    s_acc[24] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1) {
                    s_acc[25] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) {
                    s_acc[25] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 48 + 2 * (lane & 3)) {
                    s_acc[26] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) {
                    s_acc[26] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 48 + 2 * (lane & 3) + 1) {
                    s_acc[27] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) {
                    s_acc[27] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 56 + 2 * (lane & 3)) {
                    s_acc[28] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) {
                    s_acc[28] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1) {
                    s_acc[29] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) {
                    s_acc[29] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 56 + 2 * (lane & 3)) {
                    s_acc[30] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) {
                    s_acc[30] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 56 + 2 * (lane & 3) + 1) {
                    s_acc[31] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) {
                    s_acc[31] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 64 + 2 * (lane & 3)) {
                    s_acc[32] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) {
                    s_acc[32] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1) {
                    s_acc[33] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) {
                    s_acc[33] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 64 + 2 * (lane & 3)) {
                    s_acc[34] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) {
                    s_acc[34] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 64 + 2 * (lane & 3) + 1) {
                    s_acc[35] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) {
                    s_acc[35] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 72 + 2 * (lane & 3)) {
                    s_acc[36] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) {
                    s_acc[36] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1) {
                    s_acc[37] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) {
                    s_acc[37] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 72 + 2 * (lane & 3)) {
                    s_acc[38] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) {
                    s_acc[38] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 72 + 2 * (lane & 3) + 1) {
                    s_acc[39] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) {
                    s_acc[39] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 80 + 2 * (lane & 3)) {
                    s_acc[40] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) {
                    s_acc[40] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1) {
                    s_acc[41] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) {
                    s_acc[41] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 80 + 2 * (lane & 3)) {
                    s_acc[42] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) {
                    s_acc[42] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 80 + 2 * (lane & 3) + 1) {
                    s_acc[43] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) {
                    s_acc[43] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 88 + 2 * (lane & 3)) {
                    s_acc[44] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) {
                    s_acc[44] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1) {
                    s_acc[45] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) {
                    s_acc[45] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 88 + 2 * (lane & 3)) {
                    s_acc[46] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) {
                    s_acc[46] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 88 + 2 * (lane & 3) + 1) {
                    s_acc[47] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) {
                    s_acc[47] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 96 + 2 * (lane & 3)) {
                    s_acc[48] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) {
                    s_acc[48] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1) {
                    s_acc[49] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) {
                    s_acc[49] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 96 + 2 * (lane & 3)) {
                    s_acc[50] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) {
                    s_acc[50] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 96 + 2 * (lane & 3) + 1) {
                    s_acc[51] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) {
                    s_acc[51] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 104 + 2 * (lane & 3)) {
                    s_acc[52] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) {
                    s_acc[52] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1) {
                    s_acc[53] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) {
                    s_acc[53] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 104 + 2 * (lane & 3)) {
                    s_acc[54] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) {
                    s_acc[54] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 104 + 2 * (lane & 3) + 1) {
                    s_acc[55] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) {
                    s_acc[55] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 112 + 2 * (lane & 3)) {
                    s_acc[56] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) {
                    s_acc[56] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1) {
                    s_acc[57] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) {
                    s_acc[57] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 112 + 2 * (lane & 3)) {
                    s_acc[58] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) {
                    s_acc[58] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 112 + 2 * (lane & 3) + 1) {
                    s_acc[59] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) {
                    s_acc[59] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 120 + 2 * (lane & 3)) {
                    s_acc[60] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) {
                    s_acc[60] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1) {
                    s_acc[61] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) {
                    s_acc[61] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 120 + 2 * (lane & 3)) {
                    s_acc[62] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) {
                    s_acc[62] = -CUDART_INF_F;
                }
                if (seg_begin_v > kb * 128 + 120 + 2 * (lane & 3) + 1) {
                    s_acc[63] = -CUDART_INF_F;
                } else if (seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) {
                    s_acc[63] = -CUDART_INF_F;
                }
            }
            float _max_0 = fmaxf(s_acc[0], s_acc[1]);
            float _max_1 = fmaxf(s_acc[2], s_acc[3]);
            float _max_2 = fmaxf(_max_0, fmaxf(s_acc[4], s_acc[5]));
            float _max_3 = fmaxf(_max_1, fmaxf(s_acc[6], s_acc[7]));
            float _max_4 = fmaxf(_max_2, fmaxf(s_acc[8], s_acc[9]));
            float _max_5 = fmaxf(_max_3, fmaxf(s_acc[10], s_acc[11]));
            float _max_6 = fmaxf(_max_4, fmaxf(s_acc[12], s_acc[13]));
            float _max_7 = fmaxf(_max_5, fmaxf(s_acc[14], s_acc[15]));
            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, _max_6, 1);
            float _max_8 = fmaxf(_max_6, _shfl_xor_sync_2);
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, _max_7, 1);
            float _max_9 = fmaxf(_max_7, _shfl_xor_sync_3);
            float _fma_0 = __fmaf_rn(_max_8, -softmax_scale_log2, 2.5849626f);
            float _fma_1 = __fmaf_rn(_max_9, -softmax_scale_log2, 2.5849626f);
            float _max_10 = fmaxf(s_acc[16], s_acc[17]);
            float _max_11 = fmaxf(s_acc[18], s_acc[19]);
            float _max_12 = fmaxf(_max_10, fmaxf(s_acc[20], s_acc[21]));
            float _max_13 = fmaxf(_max_11, fmaxf(s_acc[22], s_acc[23]));
            float _max_14 = fmaxf(_max_12, fmaxf(s_acc[24], s_acc[25]));
            float _max_15 = fmaxf(_max_13, fmaxf(s_acc[26], s_acc[27]));
            float _max_16 = fmaxf(_max_14, fmaxf(s_acc[28], s_acc[29]));
            float _max_17 = fmaxf(_max_15, fmaxf(s_acc[30], s_acc[31]));
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, _max_16, 1);
            float _max_18 = fmaxf(_max_16, _shfl_xor_sync_4);
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, _max_17, 1);
            float _max_19 = fmaxf(_max_17, _shfl_xor_sync_5);
            float _fma_2 = __fmaf_rn(_max_18, -softmax_scale_log2, 2.5849626f);
            float _fma_3 = __fmaf_rn(_max_19, -softmax_scale_log2, 2.5849626f);
            float _exp2_4 = approx_exp2(__fmaf_rn(s_acc[0], softmax_scale_log2, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_acc[0] = _exp2_4;
            float _exp2_5 = approx_exp2(__fmaf_rn(s_acc[1], softmax_scale_log2, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_acc[1] = _exp2_5;
            float _exp2_6 = approx_exp2(__fmaf_rn(s_acc[2], softmax_scale_log2, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_acc[2] = _exp2_6;
            float _exp2_7 = approx_exp2(__fmaf_rn(s_acc[3], softmax_scale_log2, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_acc[3] = _exp2_7;
            float _exp2_8 = approx_exp2(__fmaf_rn(s_acc[4], softmax_scale_log2, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_acc[4] = _exp2_8;
            float _exp2_9 = approx_exp2(__fmaf_rn(s_acc[5], softmax_scale_log2, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_acc[5] = _exp2_9;
            float _exp2_10 = approx_exp2(__fmaf_rn(s_acc[6], softmax_scale_log2, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_acc[6] = _exp2_10;
            float _exp2_11 = approx_exp2(__fmaf_rn(s_acc[7], softmax_scale_log2, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_acc[7] = _exp2_11;
            float _exp2_12 = approx_exp2(__fmaf_rn(s_acc[8], softmax_scale_log2, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_acc[8] = _exp2_12;
            float _exp2_13 = approx_exp2(__fmaf_rn(s_acc[9], softmax_scale_log2, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_acc[9] = _exp2_13;
            float _exp2_14 = approx_exp2(__fmaf_rn(s_acc[10], softmax_scale_log2, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_acc[10] = _exp2_14;
            float _exp2_15 = approx_exp2(__fmaf_rn(s_acc[11], softmax_scale_log2, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_acc[11] = _exp2_15;
            float _exp2_16 = approx_exp2(__fmaf_rn(s_acc[12], softmax_scale_log2, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_acc[12] = _exp2_16;
            float _exp2_17 = approx_exp2(__fmaf_rn(s_acc[13], softmax_scale_log2, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_acc[13] = _exp2_17;
            float _exp2_18 = approx_exp2(__fmaf_rn(s_acc[14], softmax_scale_log2, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_acc[14] = _exp2_18;
            float _exp2_19 = approx_exp2(__fmaf_rn(s_acc[15], softmax_scale_log2, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_acc[15] = _exp2_19;
            float _psum_a4 = _exp2_4 + _exp2_5 + _exp2_8 + _exp2_9 + _exp2_12 + _exp2_13 + _exp2_16 + _exp2_17;
            float _psum_a5 = _exp2_6 + _exp2_7 + _exp2_10 + _exp2_11 + _exp2_14 + _exp2_15 + _exp2_18 + _exp2_19;
            float _exp2_22 = approx_exp2(__fmaf_rn(s_acc[16], softmax_scale_log2, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_acc[16] = _exp2_22;
            float _exp2_23 = approx_exp2(__fmaf_rn(s_acc[17], softmax_scale_log2, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_acc[17] = _exp2_23;
            float _exp2_24 = approx_exp2(__fmaf_rn(s_acc[18], softmax_scale_log2, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_acc[18] = _exp2_24;
            float _exp2_25 = approx_exp2(__fmaf_rn(s_acc[19], softmax_scale_log2, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_acc[19] = _exp2_25;
            float _exp2_26 = approx_exp2(__fmaf_rn(s_acc[20], softmax_scale_log2, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_acc[20] = _exp2_26;
            float _exp2_27 = approx_exp2(__fmaf_rn(s_acc[21], softmax_scale_log2, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_acc[21] = _exp2_27;
            float _exp2_28 = approx_exp2(__fmaf_rn(s_acc[22], softmax_scale_log2, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_acc[22] = _exp2_28;
            float _exp2_29 = approx_exp2(__fmaf_rn(s_acc[23], softmax_scale_log2, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_acc[23] = _exp2_29;
            float _exp2_30 = approx_exp2(__fmaf_rn(s_acc[24], softmax_scale_log2, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_acc[24] = _exp2_30;
            float _exp2_31 = approx_exp2(__fmaf_rn(s_acc[25], softmax_scale_log2, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_acc[25] = _exp2_31;
            float _exp2_32 = approx_exp2(__fmaf_rn(s_acc[26], softmax_scale_log2, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_acc[26] = _exp2_32;
            float _exp2_33 = approx_exp2(__fmaf_rn(s_acc[27], softmax_scale_log2, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_acc[27] = _exp2_33;
            float _exp2_34 = approx_exp2(__fmaf_rn(s_acc[28], softmax_scale_log2, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_acc[28] = _exp2_34;
            float _exp2_35 = approx_exp2(__fmaf_rn(s_acc[29], softmax_scale_log2, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_acc[29] = _exp2_35;
            float _exp2_36 = approx_exp2(__fmaf_rn(s_acc[30], softmax_scale_log2, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_acc[30] = _exp2_36;
            float _exp2_37 = approx_exp2(__fmaf_rn(s_acc[31], softmax_scale_log2, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_acc[31] = _exp2_37;
            float _psum_b4 = _exp2_22 + _exp2_23 + _exp2_26 + _exp2_27 + _exp2_30 + _exp2_31 + _exp2_34 + _exp2_35;
            float _psum_b5 = _exp2_24 + _exp2_25 + _exp2_28 + _exp2_29 + _exp2_32 + _exp2_33 + _exp2_36 + _exp2_37;
            p_tmp[0] = s_acc[0];
            p_tmp[1] = s_acc[1];
            p_tmp[2] = s_acc[4];
            p_tmp[3] = s_acc[5];
            p_tmp[4] = s_acc[8];
            p_tmp[5] = s_acc[9];
            p_tmp[6] = s_acc[12];
            p_tmp[7] = s_acc[13];
            p_tmp[8] = s_acc[2];
            p_tmp[9] = s_acc[3];
            p_tmp[10] = s_acc[6];
            p_tmp[11] = s_acc[7];
            p_tmp[12] = s_acc[10];
            p_tmp[13] = s_acc[11];
            p_tmp[14] = s_acc[14];
            p_tmp[15] = s_acc[15];
            p_tmp[16] = s_acc[16];
            p_tmp[17] = s_acc[17];
            p_tmp[18] = s_acc[20];
            p_tmp[19] = s_acc[21];
            p_tmp[20] = s_acc[24];
            p_tmp[21] = s_acc[25];
            p_tmp[22] = s_acc[28];
            p_tmp[23] = s_acc[29];
            p_tmp[24] = s_acc[18];
            p_tmp[25] = s_acc[19];
            p_tmp[26] = s_acc[22];
            p_tmp[27] = s_acc[23];
            p_tmp[28] = s_acc[26];
            p_tmp[29] = s_acc[27];
            p_tmp[30] = s_acc[30];
            p_tmp[31] = s_acc[31];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[0]) : "f"(p_tmp[0]), "f"(p_tmp[1]), "f"(p_tmp[2]), "f"(p_tmp[3]), "f"(p_tmp[4]), "f"(p_tmp[5]), "f"(p_tmp[6]), "f"(p_tmp[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[1]) : "f"(p_tmp[8]), "f"(p_tmp[9]), "f"(p_tmp[10]), "f"(p_tmp[11]), "f"(p_tmp[12]), "f"(p_tmp[13]), "f"(p_tmp[14]), "f"(p_tmp[15]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[2]) : "f"(p_tmp[16]), "f"(p_tmp[17]), "f"(p_tmp[18]), "f"(p_tmp[19]), "f"(p_tmp[20]), "f"(p_tmp[21]), "f"(p_tmp[22]), "f"(p_tmp[23]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[3]) : "f"(p_tmp[24]), "f"(p_tmp[25]), "f"(p_tmp[26]), "f"(p_tmp[27]), "f"(p_tmp[28]), "f"(p_tmp[29]), "f"(p_tmp[30]), "f"(p_tmp[31]));
            float _max_20 = fmaxf(s_acc[32], s_acc[33]);
            float _max_21 = fmaxf(s_acc[34], s_acc[35]);
            float _max_22 = fmaxf(_max_20, fmaxf(s_acc[36], s_acc[37]));
            float _max_23 = fmaxf(_max_21, fmaxf(s_acc[38], s_acc[39]));
            float _max_24 = fmaxf(_max_22, fmaxf(s_acc[40], s_acc[41]));
            float _max_25 = fmaxf(_max_23, fmaxf(s_acc[42], s_acc[43]));
            float _max_26 = fmaxf(_max_24, fmaxf(s_acc[44], s_acc[45]));
            float _max_27 = fmaxf(_max_25, fmaxf(s_acc[46], s_acc[47]));
            float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, _max_26, 1);
            float _max_28 = fmaxf(_max_26, _shfl_xor_sync_6);
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, _max_27, 1);
            float _max_29 = fmaxf(_max_27, _shfl_xor_sync_7);
            float _max_30 = fmaxf(s_acc[48], s_acc[49]);
            float _max_31 = fmaxf(s_acc[50], s_acc[51]);
            float _max_32 = fmaxf(_max_30, fmaxf(s_acc[52], s_acc[53]));
            float _max_33 = fmaxf(_max_31, fmaxf(s_acc[54], s_acc[55]));
            float _max_34 = fmaxf(_max_32, fmaxf(s_acc[56], s_acc[57]));
            float _max_35 = fmaxf(_max_33, fmaxf(s_acc[58], s_acc[59]));
            float _max_36 = fmaxf(_max_34, fmaxf(s_acc[60], s_acc[61]));
            float _max_37 = fmaxf(_max_35, fmaxf(s_acc[62], s_acc[63]));
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, _max_36, 1);
            float _max_38 = fmaxf(_max_36, _shfl_xor_sync_8);
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, _max_37, 1);
            float _max_39 = fmaxf(_max_37, _shfl_xor_sync_9);
            float max_1 = fmaxf(fmaxf(_max_8, _max_18), fmaxf(_max_28, _max_38));
            float max_1_0 = fmaxf(fmaxf(_max_9, _max_19), fmaxf(_max_29, _max_39));
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, max_1, 2);
            float _max_42 = fmaxf(max_1, _shfl_xor_sync_10);
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, max_1_0, 2);
            float _max_43 = fmaxf(max_1_0, _shfl_xor_sync_11);
            float _max_44 = fmaxf(m_state[0], _max_42 * softmax_scale_log2);
            float _max_45 = fmaxf(m_state[1], _max_43 * softmax_scale_log2);
            float _exp2_0 = approx_exp2(m_state[0] - _max_44);
            float _exp2_1 = approx_exp2(m_state[1] - _max_45);
            int _vote_0 = __any_sync(0xFFFFFFFF, _max_44 > m_state[0] || _max_45 > m_state[1]);
            m_state[0] = _max_44;
            m_state[1] = _max_45;
            if (_vote_0 != 0) {
                {
                    float2 _pair_scale_even2_12 = make_float2(_exp2_0, _exp2_0);
                    float2 _pair_scale_odd2_12 = make_float2(_exp2_1, _exp2_1);
                    float2* _pair_scale_src2_12 = reinterpret_cast<float2*>(&o_acc[0]);
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_12));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_12[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_12));
                }
            }
            float _exp2_2 = approx_exp2(__fmaf_rn(_max_8, softmax_scale_log2, 11.392318f - _max_44));
            float _exp2_3 = approx_exp2(__fmaf_rn(_max_9, softmax_scale_log2, 11.392318f - _max_45));
            uint16_t _e4m3x2_f32_0;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_exp2_3 * 0.16666667f), "f"(_exp2_2 * 0.16666667f));
            float _exp2_20 = approx_exp2(__fmaf_rn(_max_18, softmax_scale_log2, 11.392318f - _max_44));
            float _exp2_21 = approx_exp2(__fmaf_rn(_max_19, softmax_scale_log2, 11.392318f - _max_45));
            uint16_t _e4m3x2_f32_1;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_exp2_21 * 0.16666667f), "f"(_exp2_20 * 0.16666667f));
            float _fma_4 = __fmaf_rn(_psum_b4, _exp2_20 * 0.16666667f, _psum_a4 * (_exp2_2 * 0.16666667f));
            float _fma_5 = __fmaf_rn(_psum_b5, _exp2_21 * 0.16666667f, _psum_a5 * (_exp2_3 * 0.16666667f));
            unsigned int _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 16), 2);
            psf[0] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 16) | _shfl_xor_sync_13;
            float _exp2_38 = approx_exp2(__fmaf_rn(_max_28, softmax_scale_log2, 11.392318f - _max_44));
            float _exp2_39 = approx_exp2(__fmaf_rn(_max_29, softmax_scale_log2, 11.392318f - _max_45));
            uint16_t _e4m3x2_f32_2;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_exp2_39 * 0.16666667f), "f"(_exp2_38 * 0.16666667f));
            float _fma_6 = __fmaf_rn(_max_28, -softmax_scale_log2, 2.5849626f);
            float _fma_7 = __fmaf_rn(_max_29, -softmax_scale_log2, 2.5849626f);
            float _exp2_40 = approx_exp2(__fmaf_rn(s_acc[32], softmax_scale_log2, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_acc[32] = _exp2_40;
            float _exp2_41 = approx_exp2(__fmaf_rn(s_acc[33], softmax_scale_log2, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_acc[33] = _exp2_41;
            float _exp2_42 = approx_exp2(__fmaf_rn(s_acc[34], softmax_scale_log2, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_acc[34] = _exp2_42;
            float _exp2_43 = approx_exp2(__fmaf_rn(s_acc[35], softmax_scale_log2, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_acc[35] = _exp2_43;
            float _exp2_44 = approx_exp2(__fmaf_rn(s_acc[36], softmax_scale_log2, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_acc[36] = _exp2_44;
            float _exp2_45 = approx_exp2(__fmaf_rn(s_acc[37], softmax_scale_log2, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_acc[37] = _exp2_45;
            float _exp2_46 = approx_exp2(__fmaf_rn(s_acc[38], softmax_scale_log2, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_acc[38] = _exp2_46;
            float _exp2_47 = approx_exp2(__fmaf_rn(s_acc[39], softmax_scale_log2, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_acc[39] = _exp2_47;
            float _exp2_48 = approx_exp2(__fmaf_rn(s_acc[40], softmax_scale_log2, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_acc[40] = _exp2_48;
            float _exp2_49 = approx_exp2(__fmaf_rn(s_acc[41], softmax_scale_log2, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_acc[41] = _exp2_49;
            float _exp2_50 = approx_exp2(__fmaf_rn(s_acc[42], softmax_scale_log2, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_acc[42] = _exp2_50;
            float _exp2_51 = approx_exp2(__fmaf_rn(s_acc[43], softmax_scale_log2, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_acc[43] = _exp2_51;
            float _exp2_52 = approx_exp2(__fmaf_rn(s_acc[44], softmax_scale_log2, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_acc[44] = _exp2_52;
            float _exp2_53 = approx_exp2(__fmaf_rn(s_acc[45], softmax_scale_log2, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_acc[45] = _exp2_53;
            float _exp2_54 = approx_exp2(__fmaf_rn(s_acc[46], softmax_scale_log2, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_acc[46] = _exp2_54;
            float _exp2_55 = approx_exp2(__fmaf_rn(s_acc[47], softmax_scale_log2, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_acc[47] = _exp2_55;
            float _fma_8 = __fmaf_rn(_exp2_40 + _exp2_41 + _exp2_44 + _exp2_45 + _exp2_48 + _exp2_49 + _exp2_52 + _exp2_53, _exp2_38 * 0.16666667f, _fma_4);
            float _fma_9 = __fmaf_rn(_exp2_42 + _exp2_43 + _exp2_46 + _exp2_47 + _exp2_50 + _exp2_51 + _exp2_54 + _exp2_55, _exp2_39 * 0.16666667f, _fma_5);
            float _exp2_56 = approx_exp2(__fmaf_rn(_max_38, softmax_scale_log2, 11.392318f - _max_44));
            float _exp2_57 = approx_exp2(__fmaf_rn(_max_39, softmax_scale_log2, 11.392318f - _max_45));
            uint16_t _e4m3x2_f32_3;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(_exp2_57 * 0.16666667f), "f"(_exp2_56 * 0.16666667f));
            float _fma_10 = __fmaf_rn(_max_38, -softmax_scale_log2, 2.5849626f);
            float _fma_11 = __fmaf_rn(_max_39, -softmax_scale_log2, 2.5849626f);
            float _exp2_58 = approx_exp2(__fmaf_rn(s_acc[48], softmax_scale_log2, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_acc[48] = _exp2_58;
            float _exp2_59 = approx_exp2(__fmaf_rn(s_acc[49], softmax_scale_log2, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_acc[49] = _exp2_59;
            float _exp2_60 = approx_exp2(__fmaf_rn(s_acc[50], softmax_scale_log2, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_acc[50] = _exp2_60;
            float _exp2_61 = approx_exp2(__fmaf_rn(s_acc[51], softmax_scale_log2, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_acc[51] = _exp2_61;
            float _exp2_62 = approx_exp2(__fmaf_rn(s_acc[52], softmax_scale_log2, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_acc[52] = _exp2_62;
            float _exp2_63 = approx_exp2(__fmaf_rn(s_acc[53], softmax_scale_log2, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_acc[53] = _exp2_63;
            float _exp2_64 = approx_exp2(__fmaf_rn(s_acc[54], softmax_scale_log2, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_acc[54] = _exp2_64;
            float _exp2_65 = approx_exp2(__fmaf_rn(s_acc[55], softmax_scale_log2, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_acc[55] = _exp2_65;
            float _exp2_66 = approx_exp2(__fmaf_rn(s_acc[56], softmax_scale_log2, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_acc[56] = _exp2_66;
            float _exp2_67 = approx_exp2(__fmaf_rn(s_acc[57], softmax_scale_log2, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_acc[57] = _exp2_67;
            float _exp2_68 = approx_exp2(__fmaf_rn(s_acc[58], softmax_scale_log2, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_acc[58] = _exp2_68;
            float _exp2_69 = approx_exp2(__fmaf_rn(s_acc[59], softmax_scale_log2, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_acc[59] = _exp2_69;
            float _exp2_70 = approx_exp2(__fmaf_rn(s_acc[60], softmax_scale_log2, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_acc[60] = _exp2_70;
            float _exp2_71 = approx_exp2(__fmaf_rn(s_acc[61], softmax_scale_log2, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_acc[61] = _exp2_71;
            float _exp2_72 = approx_exp2(__fmaf_rn(s_acc[62], softmax_scale_log2, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_acc[62] = _exp2_72;
            float _exp2_73 = approx_exp2(__fmaf_rn(s_acc[63], softmax_scale_log2, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_acc[63] = _exp2_73;
            float _fma_12 = __fmaf_rn(_exp2_58 + _exp2_59 + _exp2_62 + _exp2_63 + _exp2_66 + _exp2_67 + _exp2_70 + _exp2_71, _exp2_56 * 0.16666667f, _fma_8);
            float _fma_13 = __fmaf_rn(_exp2_60 + _exp2_61 + _exp2_64 + _exp2_65 + _exp2_68 + _exp2_69 + _exp2_72 + _exp2_73, _exp2_57 * 0.16666667f, _fma_9);
            unsigned int _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 16), 2);
            psf[1] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 16) | _shfl_xor_sync_14;
            p_tmp[0] = s_acc[32];
            p_tmp[1] = s_acc[33];
            p_tmp[2] = s_acc[36];
            p_tmp[3] = s_acc[37];
            p_tmp[4] = s_acc[40];
            p_tmp[5] = s_acc[41];
            p_tmp[6] = s_acc[44];
            p_tmp[7] = s_acc[45];
            p_tmp[8] = s_acc[34];
            p_tmp[9] = s_acc[35];
            p_tmp[10] = s_acc[38];
            p_tmp[11] = s_acc[39];
            p_tmp[12] = s_acc[42];
            p_tmp[13] = s_acc[43];
            p_tmp[14] = s_acc[46];
            p_tmp[15] = s_acc[47];
            p_tmp[16] = s_acc[48];
            p_tmp[17] = s_acc[49];
            p_tmp[18] = s_acc[52];
            p_tmp[19] = s_acc[53];
            p_tmp[20] = s_acc[56];
            p_tmp[21] = s_acc[57];
            p_tmp[22] = s_acc[60];
            p_tmp[23] = s_acc[61];
            p_tmp[24] = s_acc[50];
            p_tmp[25] = s_acc[51];
            p_tmp[26] = s_acc[54];
            p_tmp[27] = s_acc[55];
            p_tmp[28] = s_acc[58];
            p_tmp[29] = s_acc[59];
            p_tmp[30] = s_acc[62];
            p_tmp[31] = s_acc[63];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[4]) : "f"(p_tmp[0]), "f"(p_tmp[1]), "f"(p_tmp[2]), "f"(p_tmp[3]), "f"(p_tmp[4]), "f"(p_tmp[5]), "f"(p_tmp[6]), "f"(p_tmp[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[5]) : "f"(p_tmp[8]), "f"(p_tmp[9]), "f"(p_tmp[10]), "f"(p_tmp[11]), "f"(p_tmp[12]), "f"(p_tmp[13]), "f"(p_tmp[14]), "f"(p_tmp[15]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[6]) : "f"(p_tmp[16]), "f"(p_tmp[17]), "f"(p_tmp[18]), "f"(p_tmp[19]), "f"(p_tmp[20]), "f"(p_tmp[21]), "f"(p_tmp[22]), "f"(p_tmp[23]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[7]) : "f"(p_tmp[24]), "f"(p_tmp[25]), "f"(p_tmp[26]), "f"(p_tmp[27]), "f"(p_tmp[28]), "f"(p_tmp[29]), "f"(p_tmp[30]), "f"(p_tmp[31]));
            l_state[0] = __fmaf_rn(l_state[0], _exp2_0, _fma_12);
            l_state[1] = __fmaf_rn(l_state[1], _exp2_1, _fma_13);
            mbarrier_wait(v_full_addr + (kv_stage) * 8, _phase_k_full);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_3[2];
            #pragma unroll
            for (int _lr_4 = 0; _lr_4 < 2; _lr_4++) {
                smem_regs_3[_lr_4] = KSF_stage_0[kv_stage * 256 + (unsigned int)((0 + 8 * (lane & 3) + (lane >> 2)) * 2) + (unsigned int)_lr_4];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[1])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[1])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[1])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_4[2];
            #pragma unroll
            for (int _lr_5 = 0; _lr_5 < 2; _lr_5++) {
                smem_regs_4[_lr_5] = KSF_stage_0[kv_stage * 256 + (unsigned int)((32 + 8 * (lane & 3) + (lane >> 2)) * 2) + (unsigned int)_lr_5];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[1])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[1])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[1])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_5[2];
            #pragma unroll
            for (int _lr_6 = 0; _lr_6 < 2; _lr_6++) {
                smem_regs_5[_lr_6] = KSF_stage_0[kv_stage * 256 + (unsigned int)((64 + 8 * (lane & 3) + (lane >> 2)) * 2) + (unsigned int)_lr_6];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[1])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[1])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[1])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_6[2];
            #pragma unroll
            for (int _lr_7 = 0; _lr_7 < 2; _lr_7++) {
                smem_regs_6[_lr_7] = KSF_stage_0[kv_stage * 256 + (unsigned int)((96 + 8 * (lane & 3) + (lane >> 2)) * 2) + (unsigned int)_lr_7];
            }
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[1])), "h"((uint16_t)0), "h"((uint16_t)1));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[1])), "h"((uint16_t)0), "h"((uint16_t)2));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[1])), "h"((uint16_t)0), "h"((uint16_t)3));
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(v_empty_addr + (kv_stage) * 8);
            if (warp == 0) {
                mbarrier_wait(k_empty_addr + (kv_stage) * 8, _phase_k_full);
            }
            if (kb_last >= kb + 2) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 9216);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + kv_stage * 8192), "l"((&K_map)), "r"(0), "r"((kb + 2) * 128), "r"(head),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(KSF_stage_addr + kv_stage * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb + 2)) * 64),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (warp == 4) {
                mbarrier_wait(v_empty_addr + (kv_stage) * 8, _phase_k_full);
            }
            if (kb_last >= kb + 2) {
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 9216);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr_0 + kv_stage * 8192), "l"((&VT_map)), "r"((kb + 2) * 64), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(KSF_stage_addr_0 + kv_stage * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb + 2)) * 64),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 2) { kv_stage = 0; _phase_k_full ^= 1; }
        }
        float l0 = l_state[0];
        float l1 = l_state[1];
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, l0, 1);
        l0 = l0 + _shfl_xor_sync_15;
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, l0, 2);
        l0 = l0 + _shfl_xor_sync_16;
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, l1, 1);
        l1 = l1 + _shfl_xor_sync_17;
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, l1, 2);
        l1 = l1 + _shfl_xor_sync_18;
        float inv0 = approx_rcp(l0);
        float inv1 = approx_rcp(l1);
        o_tmp[0] = o_acc[0] * inv0;
        o_tmp[1] = o_acc[1] * inv0;
        o_tmp[2] = o_acc[2] * inv1;
        o_tmp[3] = o_acc[3] * inv1;
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
        o_tmp[0] = o_acc[4] * inv0;
        o_tmp[1] = o_acc[5] * inv0;
        o_tmp[2] = o_acc[6] * inv1;
        o_tmp[3] = o_acc[7] * inv1;
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
        o_tmp[0] = o_acc[8] * inv0;
        o_tmp[1] = o_acc[9] * inv0;
        o_tmp[2] = o_acc[10] * inv1;
        o_tmp[3] = o_acc[11] * inv1;
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
        o_tmp[0] = o_acc[12] * inv0;
        o_tmp[1] = o_acc[13] * inv0;
        o_tmp[2] = o_acc[14] * inv1;
        o_tmp[3] = o_acc[15] * inv1;
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
        o_tmp[0] = o_acc[16] * inv0;
        o_tmp[1] = o_acc[17] * inv0;
        o_tmp[2] = o_acc[18] * inv1;
        o_tmp[3] = o_acc[19] * inv1;
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
        o_tmp[0] = o_acc[20] * inv0;
        o_tmp[1] = o_acc[21] * inv0;
        o_tmp[2] = o_acc[22] * inv1;
        o_tmp[3] = o_acc[23] * inv1;
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
        o_tmp[0] = o_acc[24] * inv0;
        o_tmp[1] = o_acc[25] * inv0;
        o_tmp[2] = o_acc[26] * inv1;
        o_tmp[3] = o_acc[27] * inv1;
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
        o_tmp[0] = o_acc[28] * inv0;
        o_tmp[1] = o_acc[29] * inv0;
        o_tmp[2] = o_acc[30] * inv1;
        o_tmp[3] = o_acc[31] * inv1;
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
        o_tmp[0] = o_acc[32] * inv0;
        o_tmp[1] = o_acc[33] * inv0;
        o_tmp[2] = o_acc[34] * inv1;
        o_tmp[3] = o_acc[35] * inv1;
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
        o_tmp[0] = o_acc[36] * inv0;
        o_tmp[1] = o_acc[37] * inv0;
        o_tmp[2] = o_acc[38] * inv1;
        o_tmp[3] = o_acc[39] * inv1;
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
        o_tmp[0] = o_acc[40] * inv0;
        o_tmp[1] = o_acc[41] * inv0;
        o_tmp[2] = o_acc[42] * inv1;
        o_tmp[3] = o_acc[43] * inv1;
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
        o_tmp[0] = o_acc[44] * inv0;
        o_tmp[1] = o_acc[45] * inv0;
        o_tmp[2] = o_acc[46] * inv1;
        o_tmp[3] = o_acc[47] * inv1;
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
        o_tmp[0] = o_acc[48] * inv0;
        o_tmp[1] = o_acc[49] * inv0;
        o_tmp[2] = o_acc[50] * inv1;
        o_tmp[3] = o_acc[51] * inv1;
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
        o_tmp[0] = o_acc[52] * inv0;
        o_tmp[1] = o_acc[53] * inv0;
        o_tmp[2] = o_acc[54] * inv1;
        o_tmp[3] = o_acc[55] * inv1;
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
        o_tmp[0] = o_acc[56] * inv0;
        o_tmp[1] = o_acc[57] * inv0;
        o_tmp[2] = o_acc[58] * inv1;
        o_tmp[3] = o_acc[59] * inv1;
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
        o_tmp[0] = o_acc[60] * inv0;
        o_tmp[1] = o_acc[61] * inv0;
        o_tmp[2] = o_acc[62] * inv1;
        o_tmp[3] = o_acc[63] * inv1;
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
        __syncthreads();
    }
    if (tid == 0) {
        unsigned int eold;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(eold) : "l"(&hctr[5 * num_heads + 1]), "r"(static_cast<uint32_t>(1)) : "memory");
        if ((int)eold == lead_ctas + tail_ctas - 1) {
            *(reinterpret_cast<unsigned int*>(hctr + (5 * num_heads)) + (0)) = (unsigned int)0;
            *(reinterpret_cast<unsigned int*>(hctr + (5 * num_heads + 1)) + (0)) = (unsigned int)0;
        }
    }
    if (tid == 0) {
        if (tail_ctas > 0) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(hctr) + (5 * num_heads + 2))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
    }
    if (warp < 4) {
        asm volatile("barrier.sync 1, 256;" ::: "memory");
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
