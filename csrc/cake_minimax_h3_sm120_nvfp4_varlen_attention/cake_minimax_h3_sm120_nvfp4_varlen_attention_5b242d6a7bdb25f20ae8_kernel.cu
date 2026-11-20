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
#define NUM_KV_STAGES 3
#define NUM_DELTA_STAGES 2
#define SMEM_Q_OFF 1024
#define SMEM_Q_STAGE_BYTES 8192
#define SMEM_Q_STRIDE 8192
#define SMEM_K_STAGE_OFF 9216
#define SMEM_K_STAGE_STAGE_BYTES 8192
#define SMEM_K_STAGE_STRIDE 8192
#define SMEM_KSF_STAGE_OFF 33792
#define SMEM_KSF_STAGE_STAGE_BYTES 1024
#define SMEM_KSF_STAGE_STRIDE 1024
#define SMEM_K_STAGE_0_OFF 36864
#define SMEM_K_STAGE_0_STAGE_BYTES 8192
#define SMEM_K_STAGE_0_STRIDE 8192
#define SMEM_KSF_STAGE_0_OFF 61440
#define SMEM_KSF_STAGE_0_STAGE_BYTES 1024
#define SMEM_KSF_STAGE_0_STRIDE 1024
#define SMEM_DELTA_OFF 64512
#define SMEM_DELTA_STAGE_BYTES 2048
#define SMEM_DELTA_STRIDE 2048
#define SMEM_TOTAL 66560
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
kernel_cake_minimax_h3_sm120_nvfp4_varlen_attention_5b242d6a7bdb25f20ae8(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap KSF_map, const __grid_constant__ CUtensorMap VT_map, const __grid_constant__ CUtensorMap VSF_map, __nv_bfloat16* __restrict__ O, unsigned int* __restrict__ q_sf, unsigned int* __restrict__ qm4, unsigned int* __restrict__ qm_sf, int* __restrict__ seg_tile_begin, int* __restrict__ unit_table, int total_units, int num_heads, int num_kblocks, float softmax_scale_log2)
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
    #define v_full_addr (mbar_base + 32)
    #define k_empty_addr (mbar_base + 56)
    #define v_empty_addr (mbar_base + 80)
    #define delta_ready_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* Q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int Q_addr = smem + 1024;
    uint8_t* K_stage = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int K_stage_addr = smem + 9216;
    unsigned int* KSF_stage = reinterpret_cast<unsigned int*>(smem_raw + 33792);
    const int KSF_stage_addr = smem + 33792;
    uint8_t* K_stage_0 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int K_stage_addr_0 = smem + 36864;
    unsigned int* KSF_stage_0 = reinterpret_cast<unsigned int*>(smem_raw + 61440);
    const int KSF_stage_addr_0 = smem + 61440;
    float* DELTA = reinterpret_cast<float*>(smem_raw + 64512);
    const int DELTA_addr = smem + 64512;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 15 barriers)
    // Mbarriers at smem_raw[0..120)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // k_empty: 3 barriers, init_count=256
            mbarrier_init(smem + 56, 256);
            mbarrier_init(smem + 64, 256);
            mbarrier_init(smem + 72, 256);
            // v_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            mbarrier_init(smem + 96, 8);
            // --- pipeline 'delta' ---
            // delta_ready: 2 barriers, init_count=8
            mbarrier_init(smem + 104, 8);
            mbarrier_init(smem + 112, 8);
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int _vec_load_0[4];
    {
        const int4* _ivptr_0 = reinterpret_cast<const int4*>(unit_table + (4 * bid) + 0);
        int4 _ivld_0;
        _ivld_0 = *_ivptr_0;
        _vec_load_0[0 + 0] = _ivld_0.x;
        _vec_load_0[0 + 1] = _ivld_0.y;
        _vec_load_0[0 + 2] = _ivld_0.z;
        _vec_load_0[0 + 3] = _ivld_0.w;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_full_addr, 8192);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(_vec_load_0[2] + (_vec_load_0[1] & 65535) * 128), "r"(_vec_load_0[1] >> 16),
                   "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    unsigned int _phase_q_full_0 = 0;
    unsigned int _phase_k_full = 0;
    unsigned int _phase_delta_ready = 0;
    #pragma unroll 1
    for (int slot = bid; slot < total_units; slot += num_bids) {
        int _vec_load_1[4];
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(unit_table + (4 * slot) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            _vec_load_1[0 + 0] = _ivld_1.x;
            _vec_load_1[0 + 1] = _ivld_1.y;
            _vec_load_1[0 + 2] = _ivld_1.z;
            _vec_load_1[0 + 3] = _ivld_1.w;
        }
        int seg = _vec_load_1[0];
        int packed = _vec_load_1[1];
        int head = packed >> 16;
        int q_tile = packed & 65535;
        int seg_begin_v = _vec_load_1[2];
        int seg_end_v = _vec_load_1[3];
        int q_row0 = seg_begin_v + q_tile * 128;
        int kb_first = seg_begin_v / 128;
        int kb_last = (seg_end_v - 1) / 128;
        int next_slot = ((slot + num_bids < total_units) ? slot + num_bids : slot);
        int _vec_load_2[4];
        {
            const int4* _ivptr_2 = reinterpret_cast<const int4*>(unit_table + (4 * next_slot) + 0);
            int4 _ivld_2;
            _ivld_2 = *_ivptr_2;
            _vec_load_2[0 + 0] = _ivld_2.x;
            _vec_load_2[0 + 1] = _ivld_2.y;
            _vec_load_2[0 + 2] = _ivld_2.z;
            _vec_load_2[0 + 3] = _ivld_2.w;
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
                    mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 1) % 3) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr + (kv_stage + 1) % 3 * 8192), "l"((&K_map)), "r"(0), "r"((kb_first + 1) * 128), "r"(head),
                           "r"(k_full_addr + ((kv_stage + 1) % 3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr + (kv_stage + 1) % 3 * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 1)) * 64),
                           "r"(k_full_addr + ((kv_stage + 1) % 3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 3) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr_0 + (kv_stage + 1) % 3 * 8192), "l"((&VT_map)), "r"((kb_first + 1) * 64), "r"(0), "r"(head),
                           "r"(v_full_addr + ((kv_stage + 1) % 3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr_0 + (kv_stage + 1) % 3 * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 1)) * 64),
                           "r"(v_full_addr + ((kv_stage + 1) % 3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        if (kb_last >= kb_first + 2) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + ((kv_stage + 2) % 3) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr + (kv_stage + 2) % 3 * 8192), "l"((&K_map)), "r"(0), "r"((kb_first + 2) * 128), "r"(head),
                           "r"(k_full_addr + ((kv_stage + 2) % 3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr + (kv_stage + 2) % 3 * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 2)) * 64),
                           "r"(k_full_addr + ((kv_stage + 2) % 3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 2) % 3) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K_stage_addr_0 + (kv_stage + 2) % 3 * 8192), "l"((&VT_map)), "r"((kb_first + 2) * 64), "r"(0), "r"(head),
                           "r"(v_full_addr + ((kv_stage + 2) % 3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(KSF_stage_addr_0 + (kv_stage + 2) % 3 * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 2)) * 64),
                           "r"(v_full_addr + ((kv_stage + 2) % 3) * 8), "l"(0x14F0000000000000ULL) : "memory");
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
        mbarrier_wait(k_full_addr + (kv_stage) * 8, _phase_k_full);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
            : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
            : "memory");
        unsigned int smem_regs[2];
        #pragma unroll
        for (int _lr = 0; _lr < 2; _lr++) {
            smem_regs[_lr] = KSF_stage[kv_stage * 256 + (unsigned int)((2 * warp * 8 + (lane >> 2)) * 2) + (unsigned int)_lr];
        }
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
            : "=f"(d_tmp[0]), "=f"(d_tmp[1]), "=f"(d_tmp[2]), "=f"(d_tmp[3])
            : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qmsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0));
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
            : "+f"(d_tmp[0]), "+f"(d_tmp[1]), "+f"(d_tmp[2]), "+f"(d_tmp[3])
            : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qmsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[1])), "h"((uint16_t)0), "h"((uint16_t)0));
        unsigned int smem_regs_0[2];
        #pragma unroll
        for (int _lr_1 = 0; _lr_1 < 2; _lr_1++) {
            smem_regs_0[_lr_1] = KSF_stage[kv_stage * 256 + (unsigned int)(((2 * warp + 1) * 8 + (lane >> 2)) * 2) + (unsigned int)_lr_1];
        }
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
            : "=f"(d_tmp[4]), "=f"(d_tmp[(4) + 1]), "=f"(d_tmp[(4) + 2]), "=f"(d_tmp[(4) + 3])
            : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qmsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
            : "+f"(d_tmp[4]), "+f"(d_tmp[(4) + 1]), "+f"(d_tmp[(4) + 2]), "+f"(d_tmp[(4) + 3])
            : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qmsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[1])), "h"((uint16_t)0), "h"((uint16_t)0));
        if (lane < 4) {
            DELTA[d_stage * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2)] = d_tmp[0];
            DELTA[d_stage * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2) + 1] = d_tmp[1];
            DELTA[d_stage * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2)] = d_tmp[4];
            DELTA[d_stage * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2) + 1] = d_tmp[5];
        }
        __syncwarp();
        if (elect_sync()) {
            mbarrier_arrive(delta_ready_addr + (d_stage) * 8);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 8192);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(Q_addr), "l"((&Q_map)), "r"(0), "r"(_vec_load_2[2] + (_vec_load_2[1] & 65535) * 128), "r"(_vec_load_2[1] >> 16),
                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        #pragma unroll 1
        for (int kb = kb_first; kb < kb_last + 1; kb++) {
            mbarrier_wait(k_full_addr + (kv_stage) * 8, _phase_k_full);
            mbarrier_wait(delta_ready_addr + (d_stage) * 8, _phase_delta_ready);
            float smem_regs_1[2];
            #pragma unroll
            for (int _lr_2 = 0; _lr_2 < 2; _lr_2++) {
                smem_regs_1[_lr_2] = DELTA[d_stage * 128 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_2];
            }
            s_acc[0] = smem_regs_1[0];
            s_acc[1] = smem_regs_1[1];
            s_acc[2] = smem_regs_1[0];
            s_acc[3] = smem_regs_1[1];
            float smem_regs_2[2];
            #pragma unroll
            for (int _lr_3 = 0; _lr_3 < 2; _lr_3++) {
                smem_regs_2[_lr_3] = DELTA[d_stage * 128 + 8 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_3];
            }
            s_acc[4] = smem_regs_2[0];
            s_acc[5] = smem_regs_2[1];
            s_acc[6] = smem_regs_2[0];
            s_acc[7] = smem_regs_2[1];
            float smem_regs_3[2];
            #pragma unroll
            for (int _lr_4 = 0; _lr_4 < 2; _lr_4++) {
                smem_regs_3[_lr_4] = DELTA[d_stage * 128 + 16 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_4];
            }
            s_acc[8] = smem_regs_3[0];
            s_acc[9] = smem_regs_3[1];
            s_acc[10] = smem_regs_3[0];
            s_acc[11] = smem_regs_3[1];
            float smem_regs_4[2];
            #pragma unroll
            for (int _lr_5 = 0; _lr_5 < 2; _lr_5++) {
                smem_regs_4[_lr_5] = DELTA[d_stage * 128 + 24 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_5];
            }
            s_acc[12] = smem_regs_4[0];
            s_acc[13] = smem_regs_4[1];
            s_acc[14] = smem_regs_4[0];
            s_acc[15] = smem_regs_4[1];
            float smem_regs_5[2];
            #pragma unroll
            for (int _lr_6 = 0; _lr_6 < 2; _lr_6++) {
                smem_regs_5[_lr_6] = DELTA[d_stage * 128 + 32 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_6];
            }
            s_acc[16] = smem_regs_5[0];
            s_acc[17] = smem_regs_5[1];
            s_acc[18] = smem_regs_5[0];
            s_acc[19] = smem_regs_5[1];
            float smem_regs_6[2];
            #pragma unroll
            for (int _lr_7 = 0; _lr_7 < 2; _lr_7++) {
                smem_regs_6[_lr_7] = DELTA[d_stage * 128 + 40 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_7];
            }
            s_acc[20] = smem_regs_6[0];
            s_acc[21] = smem_regs_6[1];
            s_acc[22] = smem_regs_6[0];
            s_acc[23] = smem_regs_6[1];
            float smem_regs_7[2];
            #pragma unroll
            for (int _lr_8 = 0; _lr_8 < 2; _lr_8++) {
                smem_regs_7[_lr_8] = DELTA[d_stage * 128 + 48 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_8];
            }
            s_acc[24] = smem_regs_7[0];
            s_acc[25] = smem_regs_7[1];
            s_acc[26] = smem_regs_7[0];
            s_acc[27] = smem_regs_7[1];
            float smem_regs_8[2];
            #pragma unroll
            for (int _lr_9 = 0; _lr_9 < 2; _lr_9++) {
                smem_regs_8[_lr_9] = DELTA[d_stage * 128 + 56 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_9];
            }
            s_acc[28] = smem_regs_8[0];
            s_acc[29] = smem_regs_8[1];
            s_acc[30] = smem_regs_8[0];
            s_acc[31] = smem_regs_8[1];
            float smem_regs_9[2];
            #pragma unroll
            for (int _lr_10 = 0; _lr_10 < 2; _lr_10++) {
                smem_regs_9[_lr_10] = DELTA[d_stage * 128 + 64 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_10];
            }
            s_acc[32] = smem_regs_9[0];
            s_acc[33] = smem_regs_9[1];
            s_acc[34] = smem_regs_9[0];
            s_acc[35] = smem_regs_9[1];
            float smem_regs_10[2];
            #pragma unroll
            for (int _lr_11 = 0; _lr_11 < 2; _lr_11++) {
                smem_regs_10[_lr_11] = DELTA[d_stage * 128 + 72 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_11];
            }
            s_acc[36] = smem_regs_10[0];
            s_acc[37] = smem_regs_10[1];
            s_acc[38] = smem_regs_10[0];
            s_acc[39] = smem_regs_10[1];
            float smem_regs_11[2];
            #pragma unroll
            for (int _lr_12 = 0; _lr_12 < 2; _lr_12++) {
                smem_regs_11[_lr_12] = DELTA[d_stage * 128 + 80 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_12];
            }
            s_acc[40] = smem_regs_11[0];
            s_acc[41] = smem_regs_11[1];
            s_acc[42] = smem_regs_11[0];
            s_acc[43] = smem_regs_11[1];
            float smem_regs_12[2];
            #pragma unroll
            for (int _lr_13 = 0; _lr_13 < 2; _lr_13++) {
                smem_regs_12[_lr_13] = DELTA[d_stage * 128 + 88 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_13];
            }
            s_acc[44] = smem_regs_12[0];
            s_acc[45] = smem_regs_12[1];
            s_acc[46] = smem_regs_12[0];
            s_acc[47] = smem_regs_12[1];
            float smem_regs_13[2];
            #pragma unroll
            for (int _lr_14 = 0; _lr_14 < 2; _lr_14++) {
                smem_regs_13[_lr_14] = DELTA[d_stage * 128 + 96 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_14];
            }
            s_acc[48] = smem_regs_13[0];
            s_acc[49] = smem_regs_13[1];
            s_acc[50] = smem_regs_13[0];
            s_acc[51] = smem_regs_13[1];
            float smem_regs_14[2];
            #pragma unroll
            for (int _lr_15 = 0; _lr_15 < 2; _lr_15++) {
                smem_regs_14[_lr_15] = DELTA[d_stage * 128 + 104 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_15];
            }
            s_acc[52] = smem_regs_14[0];
            s_acc[53] = smem_regs_14[1];
            s_acc[54] = smem_regs_14[0];
            s_acc[55] = smem_regs_14[1];
            float smem_regs_15[2];
            #pragma unroll
            for (int _lr_16 = 0; _lr_16 < 2; _lr_16++) {
                smem_regs_15[_lr_16] = DELTA[d_stage * 128 + 112 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_16];
            }
            s_acc[56] = smem_regs_15[0];
            s_acc[57] = smem_regs_15[1];
            s_acc[58] = smem_regs_15[0];
            s_acc[59] = smem_regs_15[1];
            float smem_regs_16[2];
            #pragma unroll
            for (int _lr_17 = 0; _lr_17 < 2; _lr_17++) {
                smem_regs_16[_lr_17] = DELTA[d_stage * 128 + 120 + (unsigned int)((lane & 3) * 2) + (unsigned int)_lr_17];
            }
            s_acc[60] = smem_regs_16[0];
            s_acc[61] = smem_regs_16[1];
            s_acc[62] = smem_regs_16[0];
            s_acc[63] = smem_regs_16[1];
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_17[2];
            #pragma unroll
            for (int _lr_18 = 0; _lr_18 < 2; _lr_18++) {
                smem_regs_17[_lr_18] = KSF_stage[kv_stage * 256 + (unsigned int)((lane >> 2) * 2) + (unsigned int)_lr_18];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_18[2];
            #pragma unroll
            for (int _lr_19 = 0; _lr_19 < 2; _lr_19++) {
                smem_regs_18[_lr_19] = KSF_stage[kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2) + (unsigned int)_lr_19];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_19[2];
            #pragma unroll
            for (int _lr_20 = 0; _lr_20 < 2; _lr_20++) {
                smem_regs_19[_lr_20] = KSF_stage[kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2) + (unsigned int)_lr_20];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_20[2];
            #pragma unroll
            for (int _lr_21 = 0; _lr_21 < 2; _lr_21++) {
                smem_regs_20[_lr_21] = KSF_stage[kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2) + (unsigned int)_lr_21];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_21[2];
            #pragma unroll
            for (int _lr_22 = 0; _lr_22 < 2; _lr_22++) {
                smem_regs_21[_lr_22] = KSF_stage[kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2) + (unsigned int)_lr_22];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_22[2];
            #pragma unroll
            for (int _lr_23 = 0; _lr_23 < 2; _lr_23++) {
                smem_regs_22[_lr_23] = KSF_stage[kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2) + (unsigned int)_lr_23];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_23[2];
            #pragma unroll
            for (int _lr_24 = 0; _lr_24 < 2; _lr_24++) {
                smem_regs_23[_lr_24] = KSF_stage[kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2) + (unsigned int)_lr_24];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_23[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_24[2];
            #pragma unroll
            for (int _lr_25 = 0; _lr_25 < 2; _lr_25++) {
                smem_regs_24[_lr_25] = KSF_stage[kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2) + (unsigned int)_lr_25];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_24[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_25[2];
            #pragma unroll
            for (int _lr_26 = 0; _lr_26 < 2; _lr_26++) {
                smem_regs_25[_lr_26] = KSF_stage[kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2) + (unsigned int)_lr_26];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_25[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_26[2];
            #pragma unroll
            for (int _lr_27 = 0; _lr_27 < 2; _lr_27++) {
                smem_regs_26[_lr_27] = KSF_stage[kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2) + (unsigned int)_lr_27];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_26[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_27[2];
            #pragma unroll
            for (int _lr_28 = 0; _lr_28 < 2; _lr_28++) {
                smem_regs_27[_lr_28] = KSF_stage[kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2) + (unsigned int)_lr_28];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_28[2];
            #pragma unroll
            for (int _lr_29 = 0; _lr_29 < 2; _lr_29++) {
                smem_regs_28[_lr_29] = KSF_stage[kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2) + (unsigned int)_lr_29];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_29[2];
            #pragma unroll
            for (int _lr_30 = 0; _lr_30 < 2; _lr_30++) {
                smem_regs_29[_lr_30] = KSF_stage[kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2) + (unsigned int)_lr_30];
            }
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_30[2];
            #pragma unroll
            for (int _lr_31 = 0; _lr_31 < 2; _lr_31++) {
                smem_regs_30[_lr_31] = KSF_stage[kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2) + (unsigned int)_lr_31];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_31[2];
            #pragma unroll
            for (int _lr_32 = 0; _lr_32 < 2; _lr_32++) {
                smem_regs_31[_lr_32] = KSF_stage[kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2) + (unsigned int)_lr_32];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_31[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_31[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_32[2];
            #pragma unroll
            for (int _lr_33 = 0; _lr_33 < 2; _lr_33++) {
                smem_regs_32[_lr_33] = KSF_stage[kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2) + (unsigned int)_lr_33];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_32[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_32[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            if (kb_last >= kb + 1) {
                mbarrier_wait(k_full_addr + (((kv_stage + 1 < 3) ? kv_stage + 1 : (unsigned int)0)) * 8, ((kv_stage + 1 < 3) ? _phase_k_full : _phase_k_full ^ 1));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                    : "r"(K_stage_addr + ((kv_stage + 1 < 3) ? kv_stage + 1 : (unsigned int)0) * 8192 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                    : "r"(K_stage_addr + ((kv_stage + 1 < 3) ? kv_stage + 1 : (unsigned int)0) * 8192 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                    : "memory");
                unsigned int smem_regs_33[2];
                #pragma unroll
                for (int _lr_34 = 0; _lr_34 < 2; _lr_34++) {
                    smem_regs_33[_lr_34] = KSF_stage[((kv_stage + 1 < 3) ? kv_stage + 1 : (unsigned int)0) * 256 + (unsigned int)((2 * warp * 8 + (lane >> 2)) * 2) + (unsigned int)_lr_34];
                }
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                    : "=f"(d_tmp[0]), "=f"(d_tmp[1]), "=f"(d_tmp[2]), "=f"(d_tmp[3])
                    : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qmsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_33[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                    : "+f"(d_tmp[0]), "+f"(d_tmp[1]), "+f"(d_tmp[2]), "+f"(d_tmp[3])
                    : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qmsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_33[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                unsigned int smem_regs_34[2];
                #pragma unroll
                for (int _lr_35 = 0; _lr_35 < 2; _lr_35++) {
                    smem_regs_34[_lr_35] = KSF_stage[((kv_stage + 1 < 3) ? kv_stage + 1 : (unsigned int)0) * 256 + (unsigned int)(((2 * warp + 1) * 8 + (lane >> 2)) * 2) + (unsigned int)_lr_35];
                }
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                    : "=f"(d_tmp[4]), "=f"(d_tmp[(4) + 1]), "=f"(d_tmp[(4) + 2]), "=f"(d_tmp[(4) + 3])
                    : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qmsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_34[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                    : "+f"(d_tmp[4]), "+f"(d_tmp[(4) + 1]), "+f"(d_tmp[(4) + 2]), "+f"(d_tmp[(4) + 3])
                    : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qmsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_34[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                if (lane < 4) {
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2)] = d_tmp[0];
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2) + 1] = d_tmp[1];
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2)] = d_tmp[4];
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2) + 1] = d_tmp[5];
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(delta_ready_addr + (((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0)) * 8);
                }
            }
            d_stage += 1;
            if (d_stage == 2) { d_stage = 0; _phase_delta_ready ^= 1; }
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
            float a0_0 = fmaxf(s_acc[0], s_acc[1]);
            float a1_0 = fmaxf(s_acc[2], s_acc[3]);
            float a0_1 = fmaxf(a0_0, fmaxf(s_acc[4], s_acc[5]));
            float a1_1 = fmaxf(a1_0, fmaxf(s_acc[6], s_acc[7]));
            float a0_2 = fmaxf(a0_1, fmaxf(s_acc[8], s_acc[9]));
            float a1_2 = fmaxf(a1_1, fmaxf(s_acc[10], s_acc[11]));
            float a0_3 = fmaxf(a0_2, fmaxf(s_acc[12], s_acc[13]));
            float a1_3 = fmaxf(a1_2, fmaxf(s_acc[14], s_acc[15]));
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, a0_3, 1);
            float m0 = fmaxf(a0_3, _shfl_xor_sync_3);
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, a1_3, 1);
            float m1 = fmaxf(a1_3, _shfl_xor_sync_4);
            float a0_4 = fmaxf(s_acc[16], s_acc[17]);
            float a1_4 = fmaxf(s_acc[18], s_acc[19]);
            float a0_5 = fmaxf(a0_4, fmaxf(s_acc[20], s_acc[21]));
            float a1_5 = fmaxf(a1_4, fmaxf(s_acc[22], s_acc[23]));
            float a0_6 = fmaxf(a0_5, fmaxf(s_acc[24], s_acc[25]));
            float a1_6 = fmaxf(a1_5, fmaxf(s_acc[26], s_acc[27]));
            float a0_7 = fmaxf(a0_6, fmaxf(s_acc[28], s_acc[29]));
            float a1_7 = fmaxf(a1_6, fmaxf(s_acc[30], s_acc[31]));
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, a0_7, 1);
            float m0_0 = fmaxf(a0_7, _shfl_xor_sync_5);
            float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, a1_7, 1);
            float m1_0 = fmaxf(a1_7, _shfl_xor_sync_6);
            float a0_8 = fmaxf(s_acc[32], s_acc[33]);
            float a1_8 = fmaxf(s_acc[34], s_acc[35]);
            float a0_9 = fmaxf(a0_8, fmaxf(s_acc[36], s_acc[37]));
            float a1_9 = fmaxf(a1_8, fmaxf(s_acc[38], s_acc[39]));
            float a0_10 = fmaxf(a0_9, fmaxf(s_acc[40], s_acc[41]));
            float a1_10 = fmaxf(a1_9, fmaxf(s_acc[42], s_acc[43]));
            float a0_11 = fmaxf(a0_10, fmaxf(s_acc[44], s_acc[45]));
            float a1_11 = fmaxf(a1_10, fmaxf(s_acc[46], s_acc[47]));
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, a0_11, 1);
            float m0_1 = fmaxf(a0_11, _shfl_xor_sync_7);
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, a1_11, 1);
            float m1_1 = fmaxf(a1_11, _shfl_xor_sync_8);
            float a0_12 = fmaxf(s_acc[48], s_acc[49]);
            float a1_12 = fmaxf(s_acc[50], s_acc[51]);
            float a0_13 = fmaxf(a0_12, fmaxf(s_acc[52], s_acc[53]));
            float a1_13 = fmaxf(a1_12, fmaxf(s_acc[54], s_acc[55]));
            float a0_14 = fmaxf(a0_13, fmaxf(s_acc[56], s_acc[57]));
            float a1_14 = fmaxf(a1_13, fmaxf(s_acc[58], s_acc[59]));
            float a0_15 = fmaxf(a0_14, fmaxf(s_acc[60], s_acc[61]));
            float a1_15 = fmaxf(a1_14, fmaxf(s_acc[62], s_acc[63]));
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, a0_15, 1);
            float m0_2 = fmaxf(a0_15, _shfl_xor_sync_9);
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, a1_15, 1);
            float m1_2 = fmaxf(a1_15, _shfl_xor_sync_10);
            float max_1 = fmaxf(fmaxf(m0, m0_0), fmaxf(m0_1, m0_2));
            float max_1_0 = fmaxf(fmaxf(m1, m1_0), fmaxf(m1_1, m1_2));
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, max_1, 2);
            float r0 = fmaxf(max_1, _shfl_xor_sync_11);
            float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, max_1_0, 2);
            float r1 = fmaxf(max_1_0, _shfl_xor_sync_12);
            float mn0 = fmaxf(m_state[0], r0 * softmax_scale_log2);
            float mn1 = fmaxf(m_state[1], r1 * softmax_scale_log2);
            float sc0 = approx_exp2(m_state[0] - mn0);
            float sc1 = approx_exp2(m_state[1] - mn1);
            int vote = __any_sync(0xFFFFFFFF, mn0 > m_state[0] || mn1 > m_state[1]);
            m_state[0] = mn0;
            m_state[1] = mn1;
            if (vote != 0) {
                {
                    float2 _pair_scale_even2_13 = make_float2(sc0, sc0);
                    float2 _pair_scale_odd2_13 = make_float2(sc1, sc1);
                    float2* _pair_scale_src2_13 = reinterpret_cast<float2*>(&o_acc[0]);
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_13));
                    asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_13[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_13));
                }
            }
            float eA_0 = approx_exp2(__fmaf_rn(m0, softmax_scale_log2, 11.392318f - mn0));
            float eB_0 = approx_exp2(__fmaf_rn(m1, softmax_scale_log2, 11.392318f - mn1));
            uint16_t cv_0;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(cv_0) : "f"(eB_0 * 0.16666667f), "f"(eA_0 * 0.16666667f));
            float f0_0 = __fmaf_rn(m0, -softmax_scale_log2, 2.5849626f);
            float f1_0 = __fmaf_rn(m1, -softmax_scale_log2, 2.5849626f);
            float x_0 = approx_exp2(__fmaf_rn(s_acc[0], softmax_scale_log2, ((m0 > -CUDART_INF_F) ? f0_0 : 0.0f)));
            s_acc[0] = x_0;
            float x_1 = approx_exp2(__fmaf_rn(s_acc[1], softmax_scale_log2, ((m0 > -CUDART_INF_F) ? f0_0 : 0.0f)));
            s_acc[1] = x_1;
            float x_2 = approx_exp2(__fmaf_rn(s_acc[2], softmax_scale_log2, ((m1 > -CUDART_INF_F) ? f1_0 : 0.0f)));
            s_acc[2] = x_2;
            float x_3 = approx_exp2(__fmaf_rn(s_acc[3], softmax_scale_log2, ((m1 > -CUDART_INF_F) ? f1_0 : 0.0f)));
            s_acc[3] = x_3;
            float x_4 = approx_exp2(__fmaf_rn(s_acc[4], softmax_scale_log2, ((m0 > -CUDART_INF_F) ? f0_0 : 0.0f)));
            s_acc[4] = x_4;
            float x_5 = approx_exp2(__fmaf_rn(s_acc[5], softmax_scale_log2, ((m0 > -CUDART_INF_F) ? f0_0 : 0.0f)));
            s_acc[5] = x_5;
            float x_6 = approx_exp2(__fmaf_rn(s_acc[6], softmax_scale_log2, ((m1 > -CUDART_INF_F) ? f1_0 : 0.0f)));
            s_acc[6] = x_6;
            float x_7 = approx_exp2(__fmaf_rn(s_acc[7], softmax_scale_log2, ((m1 > -CUDART_INF_F) ? f1_0 : 0.0f)));
            s_acc[7] = x_7;
            float x_8 = approx_exp2(__fmaf_rn(s_acc[8], softmax_scale_log2, ((m0 > -CUDART_INF_F) ? f0_0 : 0.0f)));
            s_acc[8] = x_8;
            float x_9 = approx_exp2(__fmaf_rn(s_acc[9], softmax_scale_log2, ((m0 > -CUDART_INF_F) ? f0_0 : 0.0f)));
            s_acc[9] = x_9;
            float x_10 = approx_exp2(__fmaf_rn(s_acc[10], softmax_scale_log2, ((m1 > -CUDART_INF_F) ? f1_0 : 0.0f)));
            s_acc[10] = x_10;
            float x_11 = approx_exp2(__fmaf_rn(s_acc[11], softmax_scale_log2, ((m1 > -CUDART_INF_F) ? f1_0 : 0.0f)));
            s_acc[11] = x_11;
            float x_12 = approx_exp2(__fmaf_rn(s_acc[12], softmax_scale_log2, ((m0 > -CUDART_INF_F) ? f0_0 : 0.0f)));
            s_acc[12] = x_12;
            float x_13 = approx_exp2(__fmaf_rn(s_acc[13], softmax_scale_log2, ((m0 > -CUDART_INF_F) ? f0_0 : 0.0f)));
            s_acc[13] = x_13;
            float x_14 = approx_exp2(__fmaf_rn(s_acc[14], softmax_scale_log2, ((m1 > -CUDART_INF_F) ? f1_0 : 0.0f)));
            s_acc[14] = x_14;
            float x_15 = approx_exp2(__fmaf_rn(s_acc[15], softmax_scale_log2, ((m1 > -CUDART_INF_F) ? f1_0 : 0.0f)));
            s_acc[15] = x_15;
            float eA_1 = approx_exp2(__fmaf_rn(m0_0, softmax_scale_log2, 11.392318f - mn0));
            float eB_1 = approx_exp2(__fmaf_rn(m1_0, softmax_scale_log2, 11.392318f - mn1));
            uint16_t cv_1;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(cv_1) : "f"(eB_1 * 0.16666667f), "f"(eA_1 * 0.16666667f));
            float f0_1 = __fmaf_rn(m0_0, -softmax_scale_log2, 2.5849626f);
            float f1_1 = __fmaf_rn(m1_0, -softmax_scale_log2, 2.5849626f);
            float x_16 = approx_exp2(__fmaf_rn(s_acc[16], softmax_scale_log2, ((m0_0 > -CUDART_INF_F) ? f0_1 : 0.0f)));
            s_acc[16] = x_16;
            float x_17 = approx_exp2(__fmaf_rn(s_acc[17], softmax_scale_log2, ((m0_0 > -CUDART_INF_F) ? f0_1 : 0.0f)));
            s_acc[17] = x_17;
            float x_18 = approx_exp2(__fmaf_rn(s_acc[18], softmax_scale_log2, ((m1_0 > -CUDART_INF_F) ? f1_1 : 0.0f)));
            s_acc[18] = x_18;
            float x_19 = approx_exp2(__fmaf_rn(s_acc[19], softmax_scale_log2, ((m1_0 > -CUDART_INF_F) ? f1_1 : 0.0f)));
            s_acc[19] = x_19;
            float x_20 = approx_exp2(__fmaf_rn(s_acc[20], softmax_scale_log2, ((m0_0 > -CUDART_INF_F) ? f0_1 : 0.0f)));
            s_acc[20] = x_20;
            float x_21 = approx_exp2(__fmaf_rn(s_acc[21], softmax_scale_log2, ((m0_0 > -CUDART_INF_F) ? f0_1 : 0.0f)));
            s_acc[21] = x_21;
            float x_22 = approx_exp2(__fmaf_rn(s_acc[22], softmax_scale_log2, ((m1_0 > -CUDART_INF_F) ? f1_1 : 0.0f)));
            s_acc[22] = x_22;
            float x_23 = approx_exp2(__fmaf_rn(s_acc[23], softmax_scale_log2, ((m1_0 > -CUDART_INF_F) ? f1_1 : 0.0f)));
            s_acc[23] = x_23;
            float x_24 = approx_exp2(__fmaf_rn(s_acc[24], softmax_scale_log2, ((m0_0 > -CUDART_INF_F) ? f0_1 : 0.0f)));
            s_acc[24] = x_24;
            float x_25 = approx_exp2(__fmaf_rn(s_acc[25], softmax_scale_log2, ((m0_0 > -CUDART_INF_F) ? f0_1 : 0.0f)));
            s_acc[25] = x_25;
            float x_26 = approx_exp2(__fmaf_rn(s_acc[26], softmax_scale_log2, ((m1_0 > -CUDART_INF_F) ? f1_1 : 0.0f)));
            s_acc[26] = x_26;
            float x_27 = approx_exp2(__fmaf_rn(s_acc[27], softmax_scale_log2, ((m1_0 > -CUDART_INF_F) ? f1_1 : 0.0f)));
            s_acc[27] = x_27;
            float x_28 = approx_exp2(__fmaf_rn(s_acc[28], softmax_scale_log2, ((m0_0 > -CUDART_INF_F) ? f0_1 : 0.0f)));
            s_acc[28] = x_28;
            float x_29 = approx_exp2(__fmaf_rn(s_acc[29], softmax_scale_log2, ((m0_0 > -CUDART_INF_F) ? f0_1 : 0.0f)));
            s_acc[29] = x_29;
            float x_30 = approx_exp2(__fmaf_rn(s_acc[30], softmax_scale_log2, ((m1_0 > -CUDART_INF_F) ? f1_1 : 0.0f)));
            s_acc[30] = x_30;
            float x_31 = approx_exp2(__fmaf_rn(s_acc[31], softmax_scale_log2, ((m1_0 > -CUDART_INF_F) ? f1_1 : 0.0f)));
            s_acc[31] = x_31;
            float s0_0 = __fmaf_rn(x_16 + x_17 + x_20 + x_21 + x_24 + x_25 + x_28 + x_29, eA_1 * 0.16666667f, (x_0 + x_1 + x_4 + x_5 + x_8 + x_9 + x_12 + x_13) * (eA_0 * 0.16666667f));
            float s1_0 = __fmaf_rn(x_18 + x_19 + x_22 + x_23 + x_26 + x_27 + x_30 + x_31, eB_1 * 0.16666667f, (x_2 + x_3 + x_6 + x_7 + x_10 + x_11 + x_14 + x_15) * (eB_0 * 0.16666667f));
            unsigned int _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)cv_0 >> 8 : (unsigned int)cv_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)cv_0 >> 8 : (unsigned int)cv_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)cv_1 >> 8 : (unsigned int)cv_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)cv_1 >> 8 : (unsigned int)cv_1) & 255) << 16), 2);
            psf[0] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)cv_0 >> 8 : (unsigned int)cv_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)cv_0 >> 8 : (unsigned int)cv_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)cv_1 >> 8 : (unsigned int)cv_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)cv_1 >> 8 : (unsigned int)cv_1) & 255) << 16) | _shfl_xor_sync_14;
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
            float eA_2 = approx_exp2(__fmaf_rn(m0_1, softmax_scale_log2, 11.392318f - mn0));
            float eB_2 = approx_exp2(__fmaf_rn(m1_1, softmax_scale_log2, 11.392318f - mn1));
            uint16_t cv_2;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(cv_2) : "f"(eB_2 * 0.16666667f), "f"(eA_2 * 0.16666667f));
            float f0_2 = __fmaf_rn(m0_1, -softmax_scale_log2, 2.5849626f);
            float f1_2 = __fmaf_rn(m1_1, -softmax_scale_log2, 2.5849626f);
            float x_32 = approx_exp2(__fmaf_rn(s_acc[32], softmax_scale_log2, ((m0_1 > -CUDART_INF_F) ? f0_2 : 0.0f)));
            s_acc[32] = x_32;
            float x_33 = approx_exp2(__fmaf_rn(s_acc[33], softmax_scale_log2, ((m0_1 > -CUDART_INF_F) ? f0_2 : 0.0f)));
            s_acc[33] = x_33;
            float x_34 = approx_exp2(__fmaf_rn(s_acc[34], softmax_scale_log2, ((m1_1 > -CUDART_INF_F) ? f1_2 : 0.0f)));
            s_acc[34] = x_34;
            float x_35 = approx_exp2(__fmaf_rn(s_acc[35], softmax_scale_log2, ((m1_1 > -CUDART_INF_F) ? f1_2 : 0.0f)));
            s_acc[35] = x_35;
            float x_36 = approx_exp2(__fmaf_rn(s_acc[36], softmax_scale_log2, ((m0_1 > -CUDART_INF_F) ? f0_2 : 0.0f)));
            s_acc[36] = x_36;
            float x_37 = approx_exp2(__fmaf_rn(s_acc[37], softmax_scale_log2, ((m0_1 > -CUDART_INF_F) ? f0_2 : 0.0f)));
            s_acc[37] = x_37;
            float x_38 = approx_exp2(__fmaf_rn(s_acc[38], softmax_scale_log2, ((m1_1 > -CUDART_INF_F) ? f1_2 : 0.0f)));
            s_acc[38] = x_38;
            float x_39 = approx_exp2(__fmaf_rn(s_acc[39], softmax_scale_log2, ((m1_1 > -CUDART_INF_F) ? f1_2 : 0.0f)));
            s_acc[39] = x_39;
            float x_40 = approx_exp2(__fmaf_rn(s_acc[40], softmax_scale_log2, ((m0_1 > -CUDART_INF_F) ? f0_2 : 0.0f)));
            s_acc[40] = x_40;
            float x_41 = approx_exp2(__fmaf_rn(s_acc[41], softmax_scale_log2, ((m0_1 > -CUDART_INF_F) ? f0_2 : 0.0f)));
            s_acc[41] = x_41;
            float x_42 = approx_exp2(__fmaf_rn(s_acc[42], softmax_scale_log2, ((m1_1 > -CUDART_INF_F) ? f1_2 : 0.0f)));
            s_acc[42] = x_42;
            float x_43 = approx_exp2(__fmaf_rn(s_acc[43], softmax_scale_log2, ((m1_1 > -CUDART_INF_F) ? f1_2 : 0.0f)));
            s_acc[43] = x_43;
            float x_44 = approx_exp2(__fmaf_rn(s_acc[44], softmax_scale_log2, ((m0_1 > -CUDART_INF_F) ? f0_2 : 0.0f)));
            s_acc[44] = x_44;
            float x_45 = approx_exp2(__fmaf_rn(s_acc[45], softmax_scale_log2, ((m0_1 > -CUDART_INF_F) ? f0_2 : 0.0f)));
            s_acc[45] = x_45;
            float x_46 = approx_exp2(__fmaf_rn(s_acc[46], softmax_scale_log2, ((m1_1 > -CUDART_INF_F) ? f1_2 : 0.0f)));
            s_acc[46] = x_46;
            float x_47 = approx_exp2(__fmaf_rn(s_acc[47], softmax_scale_log2, ((m1_1 > -CUDART_INF_F) ? f1_2 : 0.0f)));
            s_acc[47] = x_47;
            float s0_1 = __fmaf_rn(x_32 + x_33 + x_36 + x_37 + x_40 + x_41 + x_44 + x_45, eA_2 * 0.16666667f, s0_0);
            float s1_1 = __fmaf_rn(x_34 + x_35 + x_38 + x_39 + x_42 + x_43 + x_46 + x_47, eB_2 * 0.16666667f, s1_0);
            float eA_3 = approx_exp2(__fmaf_rn(m0_2, softmax_scale_log2, 11.392318f - mn0));
            float eB_3 = approx_exp2(__fmaf_rn(m1_2, softmax_scale_log2, 11.392318f - mn1));
            uint16_t cv_3;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(cv_3) : "f"(eB_3 * 0.16666667f), "f"(eA_3 * 0.16666667f));
            float f0_3 = __fmaf_rn(m0_2, -softmax_scale_log2, 2.5849626f);
            float f1_3 = __fmaf_rn(m1_2, -softmax_scale_log2, 2.5849626f);
            float x_48 = approx_exp2(__fmaf_rn(s_acc[48], softmax_scale_log2, ((m0_2 > -CUDART_INF_F) ? f0_3 : 0.0f)));
            s_acc[48] = x_48;
            float x_49 = approx_exp2(__fmaf_rn(s_acc[49], softmax_scale_log2, ((m0_2 > -CUDART_INF_F) ? f0_3 : 0.0f)));
            s_acc[49] = x_49;
            float x_50 = approx_exp2(__fmaf_rn(s_acc[50], softmax_scale_log2, ((m1_2 > -CUDART_INF_F) ? f1_3 : 0.0f)));
            s_acc[50] = x_50;
            float x_51 = approx_exp2(__fmaf_rn(s_acc[51], softmax_scale_log2, ((m1_2 > -CUDART_INF_F) ? f1_3 : 0.0f)));
            s_acc[51] = x_51;
            float x_52 = approx_exp2(__fmaf_rn(s_acc[52], softmax_scale_log2, ((m0_2 > -CUDART_INF_F) ? f0_3 : 0.0f)));
            s_acc[52] = x_52;
            float x_53 = approx_exp2(__fmaf_rn(s_acc[53], softmax_scale_log2, ((m0_2 > -CUDART_INF_F) ? f0_3 : 0.0f)));
            s_acc[53] = x_53;
            float x_54 = approx_exp2(__fmaf_rn(s_acc[54], softmax_scale_log2, ((m1_2 > -CUDART_INF_F) ? f1_3 : 0.0f)));
            s_acc[54] = x_54;
            float x_55 = approx_exp2(__fmaf_rn(s_acc[55], softmax_scale_log2, ((m1_2 > -CUDART_INF_F) ? f1_3 : 0.0f)));
            s_acc[55] = x_55;
            float x_56 = approx_exp2(__fmaf_rn(s_acc[56], softmax_scale_log2, ((m0_2 > -CUDART_INF_F) ? f0_3 : 0.0f)));
            s_acc[56] = x_56;
            float x_57 = approx_exp2(__fmaf_rn(s_acc[57], softmax_scale_log2, ((m0_2 > -CUDART_INF_F) ? f0_3 : 0.0f)));
            s_acc[57] = x_57;
            float x_58 = approx_exp2(__fmaf_rn(s_acc[58], softmax_scale_log2, ((m1_2 > -CUDART_INF_F) ? f1_3 : 0.0f)));
            s_acc[58] = x_58;
            float x_59 = approx_exp2(__fmaf_rn(s_acc[59], softmax_scale_log2, ((m1_2 > -CUDART_INF_F) ? f1_3 : 0.0f)));
            s_acc[59] = x_59;
            float x_60 = approx_exp2(__fmaf_rn(s_acc[60], softmax_scale_log2, ((m0_2 > -CUDART_INF_F) ? f0_3 : 0.0f)));
            s_acc[60] = x_60;
            float x_61 = approx_exp2(__fmaf_rn(s_acc[61], softmax_scale_log2, ((m0_2 > -CUDART_INF_F) ? f0_3 : 0.0f)));
            s_acc[61] = x_61;
            float x_62 = approx_exp2(__fmaf_rn(s_acc[62], softmax_scale_log2, ((m1_2 > -CUDART_INF_F) ? f1_3 : 0.0f)));
            s_acc[62] = x_62;
            float x_63 = approx_exp2(__fmaf_rn(s_acc[63], softmax_scale_log2, ((m1_2 > -CUDART_INF_F) ? f1_3 : 0.0f)));
            s_acc[63] = x_63;
            float s0_2 = __fmaf_rn(x_48 + x_49 + x_52 + x_53 + x_56 + x_57 + x_60 + x_61, eA_3 * 0.16666667f, s0_1);
            float s1_2 = __fmaf_rn(x_50 + x_51 + x_54 + x_55 + x_58 + x_59 + x_62 + x_63, eB_3 * 0.16666667f, s1_1);
            unsigned int _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)cv_2 >> 8 : (unsigned int)cv_2) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)cv_2 >> 8 : (unsigned int)cv_2) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)cv_3 >> 8 : (unsigned int)cv_3) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)cv_3 >> 8 : (unsigned int)cv_3) & 255) << 16), 2);
            psf[1] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)cv_2 >> 8 : (unsigned int)cv_2) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)cv_2 >> 8 : (unsigned int)cv_2) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)cv_3 >> 8 : (unsigned int)cv_3) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)cv_3 >> 8 : (unsigned int)cv_3) & 255) << 16) | _shfl_xor_sync_15;
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
            l_state[0] = __fmaf_rn(l_state[0], sc0, s0_2);
            l_state[1] = __fmaf_rn(l_state[1], sc1, s1_2);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
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
            unsigned int smem_regs_35[2];
            #pragma unroll
            for (int _lr_36 = 0; _lr_36 < 2; _lr_36++) {
                smem_regs_35[_lr_36] = KSF_stage_0[kv_stage * 256 + (unsigned int)((lane >> 2) * 2) + (unsigned int)_lr_36];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_35[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_35[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_36[2];
            #pragma unroll
            for (int _lr_37 = 0; _lr_37 < 2; _lr_37++) {
                smem_regs_36[_lr_37] = KSF_stage_0[kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2) + (unsigned int)_lr_37];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_36[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_36[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_37[2];
            #pragma unroll
            for (int _lr_38 = 0; _lr_38 < 2; _lr_38++) {
                smem_regs_37[_lr_38] = KSF_stage_0[kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2) + (unsigned int)_lr_38];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_37[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_37[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_38[2];
            #pragma unroll
            for (int _lr_39 = 0; _lr_39 < 2; _lr_39++) {
                smem_regs_38[_lr_39] = KSF_stage_0[kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2) + (unsigned int)_lr_39];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_38[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_38[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_39[2];
            #pragma unroll
            for (int _lr_40 = 0; _lr_40 < 2; _lr_40++) {
                smem_regs_39[_lr_40] = KSF_stage_0[kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2) + (unsigned int)_lr_40];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_39[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_39[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_40[2];
            #pragma unroll
            for (int _lr_41 = 0; _lr_41 < 2; _lr_41++) {
                smem_regs_40[_lr_41] = KSF_stage_0[kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2) + (unsigned int)_lr_41];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_40[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_40[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_41[2];
            #pragma unroll
            for (int _lr_42 = 0; _lr_42 < 2; _lr_42++) {
                smem_regs_41[_lr_42] = KSF_stage_0[kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2) + (unsigned int)_lr_42];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_41[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_41[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_42[2];
            #pragma unroll
            for (int _lr_43 = 0; _lr_43 < 2; _lr_43++) {
                smem_regs_42[_lr_43] = KSF_stage_0[kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2) + (unsigned int)_lr_43];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_42[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_42[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_43[2];
            #pragma unroll
            for (int _lr_44 = 0; _lr_44 < 2; _lr_44++) {
                smem_regs_43[_lr_44] = KSF_stage_0[kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2) + (unsigned int)_lr_44];
            }
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_43[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_43[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_44[2];
            #pragma unroll
            for (int _lr_45 = 0; _lr_45 < 2; _lr_45++) {
                smem_regs_44[_lr_45] = KSF_stage_0[kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2) + (unsigned int)_lr_45];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_44[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_44[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_45[2];
            #pragma unroll
            for (int _lr_46 = 0; _lr_46 < 2; _lr_46++) {
                smem_regs_45[_lr_46] = KSF_stage_0[kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2) + (unsigned int)_lr_46];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_45[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_45[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_46[2];
            #pragma unroll
            for (int _lr_47 = 0; _lr_47 < 2; _lr_47++) {
                smem_regs_46[_lr_47] = KSF_stage_0[kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2) + (unsigned int)_lr_47];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_46[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_46[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_47[2];
            #pragma unroll
            for (int _lr_48 = 0; _lr_48 < 2; _lr_48++) {
                smem_regs_47[_lr_48] = KSF_stage_0[kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2) + (unsigned int)_lr_48];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_47[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_47[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_48[2];
            #pragma unroll
            for (int _lr_49 = 0; _lr_49 < 2; _lr_49++) {
                smem_regs_48[_lr_49] = KSF_stage_0[kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2) + (unsigned int)_lr_49];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_48[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_48[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(K_stage_addr_0 + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_49[2];
            #pragma unroll
            for (int _lr_50 = 0; _lr_50 < 2; _lr_50++) {
                smem_regs_49[_lr_50] = KSF_stage_0[kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2) + (unsigned int)_lr_50];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_49[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_49[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_50[2];
            #pragma unroll
            for (int _lr_51 = 0; _lr_51 < 2; _lr_51++) {
                smem_regs_50[_lr_51] = KSF_stage_0[kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2) + (unsigned int)_lr_51];
            }
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_50[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_50[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(v_empty_addr + (kv_stage) * 8);
            }
            if (warp == 0) {
                mbarrier_wait(k_empty_addr + (kv_stage) * 8, _phase_k_full);
            }
            if (kb_last >= kb + 3) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 9216);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr + kv_stage * 8192), "l"((&K_map)), "r"(0), "r"((kb + 3) * 128), "r"(head),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(KSF_stage_addr + kv_stage * 1024), "l"((&KSF_map)), "r"(0), "r"((head * num_kblocks + (kb + 3)) * 64),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (warp == 4) {
                mbarrier_wait(v_empty_addr + (kv_stage) * 8, _phase_k_full);
            }
            if (kb_last >= kb + 3) {
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 9216);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K_stage_addr_0 + kv_stage * 8192), "l"((&VT_map)), "r"((kb + 3) * 64), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(KSF_stage_addr_0 + kv_stage * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb + 3)) * 64),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 3) { kv_stage = 0; _phase_k_full ^= 1; }
        }
        float l0 = l_state[0];
        float l1 = l_state[1];
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, l0, 1);
        l0 = l0 + _shfl_xor_sync_16;
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, l0, 2);
        l0 = l0 + _shfl_xor_sync_17;
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, l1, 1);
        l1 = l1 + _shfl_xor_sync_18;
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, l1, 2);
        l1 = l1 + _shfl_xor_sync_19;
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
    mbarrier_wait(q_full_addr, _phase_q_full_0);
    _phase_q_full_0 ^= 1;
    if (warp < 4) {
        asm volatile("barrier.sync 1, 256;" ::: "memory");
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
