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
#define SMEM_Q8_OFF 1024
#define SMEM_Q8_STAGE_BYTES 16384
#define SMEM_Q8_STRIDE 16384
#define SMEM_K8_STAGE_OFF 17408
#define SMEM_K8_STAGE_STAGE_BYTES 16384
#define SMEM_K8_STAGE_STRIDE 16384
#define SMEM_VT_STAGE_OFF 50176
#define SMEM_VT_STAGE_STAGE_BYTES 8192
#define SMEM_VT_STAGE_STRIDE 8192
#define SMEM_VSF_STAGE_OFF 66560
#define SMEM_VSF_STAGE_STAGE_BYTES 1024
#define SMEM_VSF_STAGE_STRIDE 1024
#define SMEM_DELTA_OFF 68608
#define SMEM_DELTA_STAGE_BYTES 1024
#define SMEM_DELTA_STRIDE 1024
#define SMEM_TOTAL 69632
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
kernel_cake_nvfp4_attention_94fb7e2f8ee5c56bd085(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap VT_map, const __grid_constant__ CUtensorMap VSF_map, __nv_bfloat16* __restrict__ O, unsigned int* __restrict__ qm8, float* __restrict__ q_amax, float* __restrict__ qm_amax, float* __restrict__ k_amax, int* __restrict__ unit_table, int total_units, int num_heads, int num_qblocks, int num_kblocks, long long o_stride_b, long long o_stride_h, long long o_stride_s, float softmax_scale_log2)
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
    #define delta_ready_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* Q8 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int Q8_addr = smem + 1024;
    uint8_t* K8_stage = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int K8_stage_addr = smem + 17408;
    uint8_t* VT_stage = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int VT_stage_addr = smem + 50176;
    unsigned int* VSF_stage = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int VSF_stage_addr = smem + 66560;
    float* DELTA = reinterpret_cast<float*>(smem_raw + 68608);
    const int DELTA_addr = smem + 68608;

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
            // v_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
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
    unsigned long long osb = (unsigned long long)o_stride_b;
    unsigned long long osh = (unsigned long long)o_stride_h;
    unsigned long long oss = (unsigned long long)o_stride_s;
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
            mbarrier_arrive_expect_tx(q_full_addr, 16384);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(Q8_addr), "l"((&Q_map)), "r"(0), "r"(_vec_load_0[2] + (_vec_load_0[1] & 65535) * 128), "r"(_vec_load_0[1] >> 16),
                   "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    unsigned int _phase_q_full_0 = 0;
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
        int batch = _vec_load_1[0];
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
        if (kb_first <= kb_last) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(K8_stage_addr + kv_stage * 16384), "l"((&K_map)), "r"(0), "r"(kb_first * 128), "r"(head),
                           "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(VT_stage_addr + kv_stage * 8192), "l"((&VT_map)), "r"(kb_first * 64), "r"(0), "r"(head),
                           "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(VSF_stage_addr + kv_stage * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + kb_first) * 64),
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
                        :: "r"(K8_stage_addr + (kv_stage + 1) % 2 * 16384), "l"((&K_map)), "r"(0), "r"((kb_first + 1) * 128), "r"(head),
                           "r"(k_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage + 1) % 2) * 8, 9216);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(VT_stage_addr + (kv_stage + 1) % 2 * 8192), "l"((&VT_map)), "r"((kb_first + 1) * 64), "r"(0), "r"(head),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(VSF_stage_addr + (kv_stage + 1) % 2 * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb_first + 1)) * 64),
                           "r"(v_full_addr + ((kv_stage + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        int row0 = q_row0 + warp * 16 + (lane >> 2);
        int row1 = row0 + 8;
        int qmrow = (batch * num_qblocks + q_tile) * num_heads + head;
        float qa = q_amax[qmrow];
        float qma = qm_amax[qmrow];
        float amax_s = fmaxf(qa, (qa + qma) * 0.5f);
        float sq = __fdiv_rn(amax_s, 127.0f);
        unsigned int qm_frag0[4];
        unsigned int qm_frag1[4];
        unsigned int qm_frag2[4];
        unsigned int qm_frag3[4];
        qm_frag0[0] = ((lane >> 2 == 0) ? qm8[qmrow * 32 + (lane & 3)] : (unsigned int)0);
        qm_frag0[1] = (unsigned int)0;
        qm_frag0[2] = ((lane >> 2 == 0) ? qm8[qmrow * 32 + 4 + (lane & 3)] : (unsigned int)0);
        qm_frag0[3] = (unsigned int)0;
        qm_frag1[0] = ((lane >> 2 == 0) ? qm8[qmrow * 32 + 8 + (lane & 3)] : (unsigned int)0);
        qm_frag1[1] = (unsigned int)0;
        qm_frag1[2] = ((lane >> 2 == 0) ? qm8[qmrow * 32 + 8 + 4 + (lane & 3)] : (unsigned int)0);
        qm_frag1[3] = (unsigned int)0;
        qm_frag2[0] = ((lane >> 2 == 0) ? qm8[qmrow * 32 + 16 + (lane & 3)] : (unsigned int)0);
        qm_frag2[1] = (unsigned int)0;
        qm_frag2[2] = ((lane >> 2 == 0) ? qm8[qmrow * 32 + 16 + 4 + (lane & 3)] : (unsigned int)0);
        qm_frag2[3] = (unsigned int)0;
        qm_frag3[0] = ((lane >> 2 == 0) ? qm8[qmrow * 32 + 24 + (lane & 3)] : (unsigned int)0);
        qm_frag3[1] = (unsigned int)0;
        qm_frag3[2] = ((lane >> 2 == 0) ? qm8[qmrow * 32 + 24 + 4 + (lane & 3)] : (unsigned int)0);
        qm_frag3[3] = (unsigned int)0;
        unsigned long long obase = (unsigned long long)batch * osb + (unsigned long long)head * osh;
        unsigned int q_frags[16];
        unsigned int k_frag[16];
        unsigned int v_frag[8];
        int s_acc[64];
        float s_f[64];
        int d_tmp[8];
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
            : "r"(Q8_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[4]), "=r"(q_frags[5]), "=r"(q_frags[6]), "=r"(q_frags[7])
            : "r"(Q8_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[8]), "=r"(q_frags[9]), "=r"(q_frags[10]), "=r"(q_frags[11])
            : "r"(Q8_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(q_frags[12]), "=r"(q_frags[13]), "=r"(q_frags[14]), "=r"(q_frags[15])
            : "r"(Q8_addr + (unsigned int)((warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 >> 1) * 16 ^ (warp * 16 + (lane >> 3 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        mbarrier_wait(k_full_addr + (kv_stage) * 8, kv_phase);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
            : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
            : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
            : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
            : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(d_tmp[0]), "=r"(d_tmp[1]), "=r"(d_tmp[2]), "=r"(d_tmp[3])
            : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d_tmp[0]), "+r"(d_tmp[1]), "+r"(d_tmp[2]), "+r"(d_tmp[3])
            : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d_tmp[0]), "+r"(d_tmp[1]), "+r"(d_tmp[2]), "+r"(d_tmp[3])
            : "r"(qm_frag2[0]), "r"(qm_frag2[1]), "r"(qm_frag2[2]), "r"(qm_frag2[3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d_tmp[0]), "+r"(d_tmp[1]), "+r"(d_tmp[2]), "+r"(d_tmp[3])
            : "r"(qm_frag3[0]), "r"(qm_frag3[1]), "r"(qm_frag3[2]), "r"(qm_frag3[3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(d_tmp[4]), "=r"(d_tmp[(4) + 1]), "=r"(d_tmp[(4) + 2]), "=r"(d_tmp[(4) + 3])
            : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d_tmp[4]), "+r"(d_tmp[(4) + 1]), "+r"(d_tmp[(4) + 2]), "+r"(d_tmp[(4) + 3])
            : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d_tmp[4]), "+r"(d_tmp[(4) + 1]), "+r"(d_tmp[(4) + 2]), "+r"(d_tmp[(4) + 3])
            : "r"(qm_frag2[0]), "r"(qm_frag2[1]), "r"(qm_frag2[2]), "r"(qm_frag2[3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d_tmp[4]), "+r"(d_tmp[(4) + 1]), "+r"(d_tmp[(4) + 2]), "+r"(d_tmp[(4) + 3])
            : "r"(qm_frag3[0]), "r"(qm_frag3[1]), "r"(qm_frag3[2]), "r"(qm_frag3[3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
        if (lane < 4) {
            DELTA[d_stage * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2)] = __uint_as_float((unsigned int)(d_tmp[0] + 1262485504)) - 2.5165824e+07f;
            DELTA[d_stage * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2 + 1)] = __uint_as_float((unsigned int)(d_tmp[1] + 1262485504)) - 2.5165824e+07f;
            DELTA[d_stage * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2)] = __uint_as_float((unsigned int)(d_tmp[4] + 1262485504)) - 2.5165824e+07f;
            DELTA[d_stage * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2 + 1)] = __uint_as_float((unsigned int)(d_tmp[5] + 1262485504)) - 2.5165824e+07f;
        }
        __syncwarp();
        if (elect_sync()) {
            mbarrier_arrive(delta_ready_addr + (d_stage) * 8);
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 16384);
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                    :: "r"(Q8_addr), "l"((&Q_map)), "r"(0), "r"(_vec_load_2[2] + (_vec_load_2[1] & 65535) * 128), "r"(_vec_load_2[1] >> 16),
                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
        #pragma unroll 1
        for (int kb = kb_first; kb < kb_last + 1; kb++) {
            float sk = __fdiv_rn(k_amax[head * num_kblocks + kb], 127.0f);
            float c_t = softmax_scale_log2 * (sq * sk);
            mbarrier_wait(k_full_addr + (kv_stage) * 8, kv_phase);
            mbarrier_wait(delta_ready_addr + (d_stage) * 8, d_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[0]), "=r"(s_acc[1]), "=r"(s_acc[2]), "=r"(s_acc[3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[0]), "+r"(s_acc[1]), "+r"(s_acc[2]), "+r"(s_acc[3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[0]), "+r"(s_acc[1]), "+r"(s_acc[2]), "+r"(s_acc[3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[0]), "+r"(s_acc[1]), "+r"(s_acc[2]), "+r"(s_acc[3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[4]), "=r"(s_acc[(4) + 1]), "=r"(s_acc[(4) + 2]), "=r"(s_acc[(4) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[4]), "+r"(s_acc[(4) + 1]), "+r"(s_acc[(4) + 2]), "+r"(s_acc[(4) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[4]), "+r"(s_acc[(4) + 1]), "+r"(s_acc[(4) + 2]), "+r"(s_acc[(4) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[4]), "+r"(s_acc[(4) + 1]), "+r"(s_acc[(4) + 2]), "+r"(s_acc[(4) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[8]), "=r"(s_acc[(8) + 1]), "=r"(s_acc[(8) + 2]), "=r"(s_acc[(8) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[8]), "+r"(s_acc[(8) + 1]), "+r"(s_acc[(8) + 2]), "+r"(s_acc[(8) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[8]), "+r"(s_acc[(8) + 1]), "+r"(s_acc[(8) + 2]), "+r"(s_acc[(8) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[8]), "+r"(s_acc[(8) + 1]), "+r"(s_acc[(8) + 2]), "+r"(s_acc[(8) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[12]), "=r"(s_acc[(12) + 1]), "=r"(s_acc[(12) + 2]), "=r"(s_acc[(12) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[12]), "+r"(s_acc[(12) + 1]), "+r"(s_acc[(12) + 2]), "+r"(s_acc[(12) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[12]), "+r"(s_acc[(12) + 1]), "+r"(s_acc[(12) + 2]), "+r"(s_acc[(12) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[12]), "+r"(s_acc[(12) + 1]), "+r"(s_acc[(12) + 2]), "+r"(s_acc[(12) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[16]), "=r"(s_acc[(16) + 1]), "=r"(s_acc[(16) + 2]), "=r"(s_acc[(16) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[16]), "+r"(s_acc[(16) + 1]), "+r"(s_acc[(16) + 2]), "+r"(s_acc[(16) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[16]), "+r"(s_acc[(16) + 1]), "+r"(s_acc[(16) + 2]), "+r"(s_acc[(16) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[16]), "+r"(s_acc[(16) + 1]), "+r"(s_acc[(16) + 2]), "+r"(s_acc[(16) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[20]), "=r"(s_acc[(20) + 1]), "=r"(s_acc[(20) + 2]), "=r"(s_acc[(20) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[20]), "+r"(s_acc[(20) + 1]), "+r"(s_acc[(20) + 2]), "+r"(s_acc[(20) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[20]), "+r"(s_acc[(20) + 1]), "+r"(s_acc[(20) + 2]), "+r"(s_acc[(20) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[20]), "+r"(s_acc[(20) + 1]), "+r"(s_acc[(20) + 2]), "+r"(s_acc[(20) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[24]), "=r"(s_acc[(24) + 1]), "=r"(s_acc[(24) + 2]), "=r"(s_acc[(24) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[24]), "+r"(s_acc[(24) + 1]), "+r"(s_acc[(24) + 2]), "+r"(s_acc[(24) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[24]), "+r"(s_acc[(24) + 1]), "+r"(s_acc[(24) + 2]), "+r"(s_acc[(24) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[24]), "+r"(s_acc[(24) + 1]), "+r"(s_acc[(24) + 2]), "+r"(s_acc[(24) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[28]), "=r"(s_acc[(28) + 1]), "=r"(s_acc[(28) + 2]), "=r"(s_acc[(28) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[28]), "+r"(s_acc[(28) + 1]), "+r"(s_acc[(28) + 2]), "+r"(s_acc[(28) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[28]), "+r"(s_acc[(28) + 1]), "+r"(s_acc[(28) + 2]), "+r"(s_acc[(28) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[28]), "+r"(s_acc[(28) + 1]), "+r"(s_acc[(28) + 2]), "+r"(s_acc[(28) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[32]), "=r"(s_acc[(32) + 1]), "=r"(s_acc[(32) + 2]), "=r"(s_acc[(32) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[32]), "+r"(s_acc[(32) + 1]), "+r"(s_acc[(32) + 2]), "+r"(s_acc[(32) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[32]), "+r"(s_acc[(32) + 1]), "+r"(s_acc[(32) + 2]), "+r"(s_acc[(32) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[32]), "+r"(s_acc[(32) + 1]), "+r"(s_acc[(32) + 2]), "+r"(s_acc[(32) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[36]), "=r"(s_acc[(36) + 1]), "=r"(s_acc[(36) + 2]), "=r"(s_acc[(36) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[36]), "+r"(s_acc[(36) + 1]), "+r"(s_acc[(36) + 2]), "+r"(s_acc[(36) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[36]), "+r"(s_acc[(36) + 1]), "+r"(s_acc[(36) + 2]), "+r"(s_acc[(36) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[36]), "+r"(s_acc[(36) + 1]), "+r"(s_acc[(36) + 2]), "+r"(s_acc[(36) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[40]), "=r"(s_acc[(40) + 1]), "=r"(s_acc[(40) + 2]), "=r"(s_acc[(40) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[40]), "+r"(s_acc[(40) + 1]), "+r"(s_acc[(40) + 2]), "+r"(s_acc[(40) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[40]), "+r"(s_acc[(40) + 1]), "+r"(s_acc[(40) + 2]), "+r"(s_acc[(40) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[40]), "+r"(s_acc[(40) + 1]), "+r"(s_acc[(40) + 2]), "+r"(s_acc[(40) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[44]), "=r"(s_acc[(44) + 1]), "=r"(s_acc[(44) + 2]), "=r"(s_acc[(44) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[44]), "+r"(s_acc[(44) + 1]), "+r"(s_acc[(44) + 2]), "+r"(s_acc[(44) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[44]), "+r"(s_acc[(44) + 1]), "+r"(s_acc[(44) + 2]), "+r"(s_acc[(44) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[44]), "+r"(s_acc[(44) + 1]), "+r"(s_acc[(44) + 2]), "+r"(s_acc[(44) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[48]), "=r"(s_acc[(48) + 1]), "=r"(s_acc[(48) + 2]), "=r"(s_acc[(48) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[48]), "+r"(s_acc[(48) + 1]), "+r"(s_acc[(48) + 2]), "+r"(s_acc[(48) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[48]), "+r"(s_acc[(48) + 1]), "+r"(s_acc[(48) + 2]), "+r"(s_acc[(48) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[48]), "+r"(s_acc[(48) + 1]), "+r"(s_acc[(48) + 2]), "+r"(s_acc[(48) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[52]), "=r"(s_acc[(52) + 1]), "=r"(s_acc[(52) + 2]), "=r"(s_acc[(52) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[52]), "+r"(s_acc[(52) + 1]), "+r"(s_acc[(52) + 2]), "+r"(s_acc[(52) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[52]), "+r"(s_acc[(52) + 1]), "+r"(s_acc[(52) + 2]), "+r"(s_acc[(52) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[52]), "+r"(s_acc[(52) + 1]), "+r"(s_acc[(52) + 2]), "+r"(s_acc[(52) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                : "r"(K8_stage_addr + kv_stage * 16384 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[56]), "=r"(s_acc[(56) + 1]), "=r"(s_acc[(56) + 2]), "=r"(s_acc[(56) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[56]), "+r"(s_acc[(56) + 1]), "+r"(s_acc[(56) + 2]), "+r"(s_acc[(56) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[56]), "+r"(s_acc[(56) + 1]), "+r"(s_acc[(56) + 2]), "+r"(s_acc[(56) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[56]), "+r"(s_acc[(56) + 1]), "+r"(s_acc[(56) + 2]), "+r"(s_acc[(56) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(s_acc[60]), "=r"(s_acc[(60) + 1]), "=r"(s_acc[(60) + 2]), "=r"(s_acc[(60) + 3])
                : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[60]), "+r"(s_acc[(60) + 1]), "+r"(s_acc[(60) + 2]), "+r"(s_acc[(60) + 3])
                : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[60]), "+r"(s_acc[(60) + 1]), "+r"(s_acc[(60) + 2]), "+r"(s_acc[(60) + 3])
                : "r"(q_frags[8]), "r"(q_frags[(8) + 1]), "r"(q_frags[(8) + 2]), "r"(q_frags[(8) + 3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(s_acc[60]), "+r"(s_acc[(60) + 1]), "+r"(s_acc[(60) + 2]), "+r"(s_acc[(60) + 3])
                : "r"(q_frags[12]), "r"(q_frags[(12) + 1]), "r"(q_frags[(12) + 2]), "r"(q_frags[(12) + 3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            if (kb_last >= kb + 1) {
                mbarrier_wait(k_full_addr + (((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0)) * 8, ((kv_stage + 1 < 2) ? kv_phase : kv_phase ^ 1));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                    : "r"(K8_stage_addr + ((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0) * 16384 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)((lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                    : "r"(K8_stage_addr + ((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0) * 16384 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k_frag[8]), "=r"(k_frag[9]), "=r"(k_frag[10]), "=r"(k_frag[11])
                    : "r"(K8_stage_addr + ((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0) * 16384 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(64 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k_frag[12]), "=r"(k_frag[13]), "=r"(k_frag[14]), "=r"(k_frag[15])
                    : "r"(K8_stage_addr + ((kv_stage + 1 < 2) ? kv_stage + 1 : (unsigned int)0) * 16384 + (unsigned int)((warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 128) + (unsigned int)(96 + (lane >> 3 & 1) * 16 ^ (warp * 16 + (lane >> 4 & 1) * 8 + (lane & 7) & 7) << 4))
                    : "memory");
            }
            float smem_regs[2];
            smem_regs[0] = DELTA[d_stage * 128 + (unsigned int)((lane & 3) * 2)];
            smem_regs[1] = DELTA[d_stage * 128 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[0] = __uint_as_float((unsigned int)(s_acc[0] + 1262485504)) + smem_regs[0];
            s_f[1] = __uint_as_float((unsigned int)(s_acc[1] + 1262485504)) + smem_regs[1];
            s_f[2] = __uint_as_float((unsigned int)(s_acc[2] + 1262485504)) + smem_regs[0];
            s_f[3] = __uint_as_float((unsigned int)(s_acc[3] + 1262485504)) + smem_regs[1];
            float smem_regs_0[2];
            smem_regs_0[0] = DELTA[d_stage * 128 + 8 + (unsigned int)((lane & 3) * 2)];
            smem_regs_0[1] = DELTA[d_stage * 128 + 8 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[4] = __uint_as_float((unsigned int)(s_acc[4] + 1262485504)) + smem_regs_0[0];
            s_f[5] = __uint_as_float((unsigned int)(s_acc[5] + 1262485504)) + smem_regs_0[1];
            s_f[6] = __uint_as_float((unsigned int)(s_acc[6] + 1262485504)) + smem_regs_0[0];
            s_f[7] = __uint_as_float((unsigned int)(s_acc[7] + 1262485504)) + smem_regs_0[1];
            float smem_regs_1[2];
            smem_regs_1[0] = DELTA[d_stage * 128 + 16 + (unsigned int)((lane & 3) * 2)];
            smem_regs_1[1] = DELTA[d_stage * 128 + 16 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[8] = __uint_as_float((unsigned int)(s_acc[8] + 1262485504)) + smem_regs_1[0];
            s_f[9] = __uint_as_float((unsigned int)(s_acc[9] + 1262485504)) + smem_regs_1[1];
            s_f[10] = __uint_as_float((unsigned int)(s_acc[10] + 1262485504)) + smem_regs_1[0];
            s_f[11] = __uint_as_float((unsigned int)(s_acc[11] + 1262485504)) + smem_regs_1[1];
            float smem_regs_2[2];
            smem_regs_2[0] = DELTA[d_stage * 128 + 24 + (unsigned int)((lane & 3) * 2)];
            smem_regs_2[1] = DELTA[d_stage * 128 + 24 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[12] = __uint_as_float((unsigned int)(s_acc[12] + 1262485504)) + smem_regs_2[0];
            s_f[13] = __uint_as_float((unsigned int)(s_acc[13] + 1262485504)) + smem_regs_2[1];
            s_f[14] = __uint_as_float((unsigned int)(s_acc[14] + 1262485504)) + smem_regs_2[0];
            s_f[15] = __uint_as_float((unsigned int)(s_acc[15] + 1262485504)) + smem_regs_2[1];
            float smem_regs_3[2];
            smem_regs_3[0] = DELTA[d_stage * 128 + 32 + (unsigned int)((lane & 3) * 2)];
            smem_regs_3[1] = DELTA[d_stage * 128 + 32 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[16] = __uint_as_float((unsigned int)(s_acc[16] + 1262485504)) + smem_regs_3[0];
            s_f[17] = __uint_as_float((unsigned int)(s_acc[17] + 1262485504)) + smem_regs_3[1];
            s_f[18] = __uint_as_float((unsigned int)(s_acc[18] + 1262485504)) + smem_regs_3[0];
            s_f[19] = __uint_as_float((unsigned int)(s_acc[19] + 1262485504)) + smem_regs_3[1];
            float smem_regs_4[2];
            smem_regs_4[0] = DELTA[d_stage * 128 + 40 + (unsigned int)((lane & 3) * 2)];
            smem_regs_4[1] = DELTA[d_stage * 128 + 40 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[20] = __uint_as_float((unsigned int)(s_acc[20] + 1262485504)) + smem_regs_4[0];
            s_f[21] = __uint_as_float((unsigned int)(s_acc[21] + 1262485504)) + smem_regs_4[1];
            s_f[22] = __uint_as_float((unsigned int)(s_acc[22] + 1262485504)) + smem_regs_4[0];
            s_f[23] = __uint_as_float((unsigned int)(s_acc[23] + 1262485504)) + smem_regs_4[1];
            float smem_regs_5[2];
            smem_regs_5[0] = DELTA[d_stage * 128 + 48 + (unsigned int)((lane & 3) * 2)];
            smem_regs_5[1] = DELTA[d_stage * 128 + 48 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[24] = __uint_as_float((unsigned int)(s_acc[24] + 1262485504)) + smem_regs_5[0];
            s_f[25] = __uint_as_float((unsigned int)(s_acc[25] + 1262485504)) + smem_regs_5[1];
            s_f[26] = __uint_as_float((unsigned int)(s_acc[26] + 1262485504)) + smem_regs_5[0];
            s_f[27] = __uint_as_float((unsigned int)(s_acc[27] + 1262485504)) + smem_regs_5[1];
            float smem_regs_6[2];
            smem_regs_6[0] = DELTA[d_stage * 128 + 56 + (unsigned int)((lane & 3) * 2)];
            smem_regs_6[1] = DELTA[d_stage * 128 + 56 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[28] = __uint_as_float((unsigned int)(s_acc[28] + 1262485504)) + smem_regs_6[0];
            s_f[29] = __uint_as_float((unsigned int)(s_acc[29] + 1262485504)) + smem_regs_6[1];
            s_f[30] = __uint_as_float((unsigned int)(s_acc[30] + 1262485504)) + smem_regs_6[0];
            s_f[31] = __uint_as_float((unsigned int)(s_acc[31] + 1262485504)) + smem_regs_6[1];
            float smem_regs_7[2];
            smem_regs_7[0] = DELTA[d_stage * 128 + 64 + (unsigned int)((lane & 3) * 2)];
            smem_regs_7[1] = DELTA[d_stage * 128 + 64 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[32] = __uint_as_float((unsigned int)(s_acc[32] + 1262485504)) + smem_regs_7[0];
            s_f[33] = __uint_as_float((unsigned int)(s_acc[33] + 1262485504)) + smem_regs_7[1];
            s_f[34] = __uint_as_float((unsigned int)(s_acc[34] + 1262485504)) + smem_regs_7[0];
            s_f[35] = __uint_as_float((unsigned int)(s_acc[35] + 1262485504)) + smem_regs_7[1];
            float smem_regs_8[2];
            smem_regs_8[0] = DELTA[d_stage * 128 + 72 + (unsigned int)((lane & 3) * 2)];
            smem_regs_8[1] = DELTA[d_stage * 128 + 72 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[36] = __uint_as_float((unsigned int)(s_acc[36] + 1262485504)) + smem_regs_8[0];
            s_f[37] = __uint_as_float((unsigned int)(s_acc[37] + 1262485504)) + smem_regs_8[1];
            s_f[38] = __uint_as_float((unsigned int)(s_acc[38] + 1262485504)) + smem_regs_8[0];
            s_f[39] = __uint_as_float((unsigned int)(s_acc[39] + 1262485504)) + smem_regs_8[1];
            float smem_regs_9[2];
            smem_regs_9[0] = DELTA[d_stage * 128 + 80 + (unsigned int)((lane & 3) * 2)];
            smem_regs_9[1] = DELTA[d_stage * 128 + 80 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[40] = __uint_as_float((unsigned int)(s_acc[40] + 1262485504)) + smem_regs_9[0];
            s_f[41] = __uint_as_float((unsigned int)(s_acc[41] + 1262485504)) + smem_regs_9[1];
            s_f[42] = __uint_as_float((unsigned int)(s_acc[42] + 1262485504)) + smem_regs_9[0];
            s_f[43] = __uint_as_float((unsigned int)(s_acc[43] + 1262485504)) + smem_regs_9[1];
            float smem_regs_10[2];
            smem_regs_10[0] = DELTA[d_stage * 128 + 88 + (unsigned int)((lane & 3) * 2)];
            smem_regs_10[1] = DELTA[d_stage * 128 + 88 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[44] = __uint_as_float((unsigned int)(s_acc[44] + 1262485504)) + smem_regs_10[0];
            s_f[45] = __uint_as_float((unsigned int)(s_acc[45] + 1262485504)) + smem_regs_10[1];
            s_f[46] = __uint_as_float((unsigned int)(s_acc[46] + 1262485504)) + smem_regs_10[0];
            s_f[47] = __uint_as_float((unsigned int)(s_acc[47] + 1262485504)) + smem_regs_10[1];
            float smem_regs_11[2];
            smem_regs_11[0] = DELTA[d_stage * 128 + 96 + (unsigned int)((lane & 3) * 2)];
            smem_regs_11[1] = DELTA[d_stage * 128 + 96 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[48] = __uint_as_float((unsigned int)(s_acc[48] + 1262485504)) + smem_regs_11[0];
            s_f[49] = __uint_as_float((unsigned int)(s_acc[49] + 1262485504)) + smem_regs_11[1];
            s_f[50] = __uint_as_float((unsigned int)(s_acc[50] + 1262485504)) + smem_regs_11[0];
            s_f[51] = __uint_as_float((unsigned int)(s_acc[51] + 1262485504)) + smem_regs_11[1];
            float smem_regs_12[2];
            smem_regs_12[0] = DELTA[d_stage * 128 + 104 + (unsigned int)((lane & 3) * 2)];
            smem_regs_12[1] = DELTA[d_stage * 128 + 104 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[52] = __uint_as_float((unsigned int)(s_acc[52] + 1262485504)) + smem_regs_12[0];
            s_f[53] = __uint_as_float((unsigned int)(s_acc[53] + 1262485504)) + smem_regs_12[1];
            s_f[54] = __uint_as_float((unsigned int)(s_acc[54] + 1262485504)) + smem_regs_12[0];
            s_f[55] = __uint_as_float((unsigned int)(s_acc[55] + 1262485504)) + smem_regs_12[1];
            float smem_regs_13[2];
            smem_regs_13[0] = DELTA[d_stage * 128 + 112 + (unsigned int)((lane & 3) * 2)];
            smem_regs_13[1] = DELTA[d_stage * 128 + 112 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[56] = __uint_as_float((unsigned int)(s_acc[56] + 1262485504)) + smem_regs_13[0];
            s_f[57] = __uint_as_float((unsigned int)(s_acc[57] + 1262485504)) + smem_regs_13[1];
            s_f[58] = __uint_as_float((unsigned int)(s_acc[58] + 1262485504)) + smem_regs_13[0];
            s_f[59] = __uint_as_float((unsigned int)(s_acc[59] + 1262485504)) + smem_regs_13[1];
            float smem_regs_14[2];
            smem_regs_14[0] = DELTA[d_stage * 128 + 120 + (unsigned int)((lane & 3) * 2)];
            smem_regs_14[1] = DELTA[d_stage * 128 + 120 + (unsigned int)((lane & 3) * 2) + 1];
            s_f[60] = __uint_as_float((unsigned int)(s_acc[60] + 1262485504)) + smem_regs_14[0];
            s_f[61] = __uint_as_float((unsigned int)(s_acc[61] + 1262485504)) + smem_regs_14[1];
            s_f[62] = __uint_as_float((unsigned int)(s_acc[62] + 1262485504)) + smem_regs_14[0];
            s_f[63] = __uint_as_float((unsigned int)(s_acc[63] + 1262485504)) + smem_regs_14[1];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
            }
            if (kb_last >= kb + 1) {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=r"(d_tmp[0]), "=r"(d_tmp[1]), "=r"(d_tmp[2]), "=r"(d_tmp[3])
                    : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[0]), "r"(k_frag[1]), "r"(0), "r"(0), "r"(0), "r"(0));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(d_tmp[0]), "+r"(d_tmp[1]), "+r"(d_tmp[2]), "+r"(d_tmp[3])
                    : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(d_tmp[0]), "+r"(d_tmp[1]), "+r"(d_tmp[2]), "+r"(d_tmp[3])
                    : "r"(qm_frag2[0]), "r"(qm_frag2[1]), "r"(qm_frag2[2]), "r"(qm_frag2[3]), "r"(k_frag[8]), "r"(k_frag[(8) + 1]));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(d_tmp[0]), "+r"(d_tmp[1]), "+r"(d_tmp[2]), "+r"(d_tmp[3])
                    : "r"(qm_frag3[0]), "r"(qm_frag3[1]), "r"(qm_frag3[2]), "r"(qm_frag3[3]), "r"(k_frag[12]), "r"(k_frag[(12) + 1]));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=r"(d_tmp[4]), "=r"(d_tmp[(4) + 1]), "=r"(d_tmp[(4) + 2]), "=r"(d_tmp[(4) + 3])
                    : "r"(qm_frag0[0]), "r"(qm_frag0[1]), "r"(qm_frag0[2]), "r"(qm_frag0[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(d_tmp[4]), "+r"(d_tmp[(4) + 1]), "+r"(d_tmp[(4) + 2]), "+r"(d_tmp[(4) + 3])
                    : "r"(qm_frag1[0]), "r"(qm_frag1[1]), "r"(qm_frag1[2]), "r"(qm_frag1[3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(d_tmp[4]), "+r"(d_tmp[(4) + 1]), "+r"(d_tmp[(4) + 2]), "+r"(d_tmp[(4) + 3])
                    : "r"(qm_frag2[0]), "r"(qm_frag2[1]), "r"(qm_frag2[2]), "r"(qm_frag2[3]), "r"(k_frag[10]), "r"(k_frag[(10) + 1]));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(d_tmp[4]), "+r"(d_tmp[(4) + 1]), "+r"(d_tmp[(4) + 2]), "+r"(d_tmp[(4) + 3])
                    : "r"(qm_frag3[0]), "r"(qm_frag3[1]), "r"(qm_frag3[2]), "r"(qm_frag3[3]), "r"(k_frag[14]), "r"(k_frag[(14) + 1]));
                if (lane < 4) {
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2)] = __uint_as_float((unsigned int)(d_tmp[0] + 1262485504)) - 2.5165824e+07f;
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)(2 * warp * 8) + (unsigned int)((lane & 3) * 2 + 1)] = __uint_as_float((unsigned int)(d_tmp[1] + 1262485504)) - 2.5165824e+07f;
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2)] = __uint_as_float((unsigned int)(d_tmp[4] + 1262485504)) - 2.5165824e+07f;
                    DELTA[((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0) * 128 + (unsigned int)((2 * warp + 1) * 8) + (unsigned int)((lane & 3) * 2 + 1)] = __uint_as_float((unsigned int)(d_tmp[5] + 1262485504)) - 2.5165824e+07f;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(delta_ready_addr + (((d_stage + 1 < 2) ? d_stage + 1 : (unsigned int)0)) * 8);
                }
            }
            d_stage += 1;
            if (d_stage == 2) { d_stage = 0; d_phase ^= 1; }
            if (kb_last == kb) {
                s_f[0] = ((seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[0]);
                s_f[1] = ((seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[1]);
                s_f[2] = ((seg_end_v <= kb * 128 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[2]);
                s_f[3] = ((seg_end_v <= kb * 128 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[3]);
                s_f[4] = ((seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[4]);
                s_f[5] = ((seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[5]);
                s_f[6] = ((seg_end_v <= kb * 128 + 8 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[6]);
                s_f[7] = ((seg_end_v <= kb * 128 + 8 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[7]);
                s_f[8] = ((seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[8]);
                s_f[9] = ((seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[9]);
                s_f[10] = ((seg_end_v <= kb * 128 + 16 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[10]);
                s_f[11] = ((seg_end_v <= kb * 128 + 16 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[11]);
                s_f[12] = ((seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[12]);
                s_f[13] = ((seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[13]);
                s_f[14] = ((seg_end_v <= kb * 128 + 24 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[14]);
                s_f[15] = ((seg_end_v <= kb * 128 + 24 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[15]);
                s_f[16] = ((seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[16]);
                s_f[17] = ((seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[17]);
                s_f[18] = ((seg_end_v <= kb * 128 + 32 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[18]);
                s_f[19] = ((seg_end_v <= kb * 128 + 32 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[19]);
                s_f[20] = ((seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[20]);
                s_f[21] = ((seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[21]);
                s_f[22] = ((seg_end_v <= kb * 128 + 40 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[22]);
                s_f[23] = ((seg_end_v <= kb * 128 + 40 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[23]);
                s_f[24] = ((seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[24]);
                s_f[25] = ((seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[25]);
                s_f[26] = ((seg_end_v <= kb * 128 + 48 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[26]);
                s_f[27] = ((seg_end_v <= kb * 128 + 48 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[27]);
                s_f[28] = ((seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[28]);
                s_f[29] = ((seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[29]);
                s_f[30] = ((seg_end_v <= kb * 128 + 56 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[30]);
                s_f[31] = ((seg_end_v <= kb * 128 + 56 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[31]);
                s_f[32] = ((seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[32]);
                s_f[33] = ((seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[33]);
                s_f[34] = ((seg_end_v <= kb * 128 + 64 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[34]);
                s_f[35] = ((seg_end_v <= kb * 128 + 64 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[35]);
                s_f[36] = ((seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[36]);
                s_f[37] = ((seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[37]);
                s_f[38] = ((seg_end_v <= kb * 128 + 72 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[38]);
                s_f[39] = ((seg_end_v <= kb * 128 + 72 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[39]);
                s_f[40] = ((seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[40]);
                s_f[41] = ((seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[41]);
                s_f[42] = ((seg_end_v <= kb * 128 + 80 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[42]);
                s_f[43] = ((seg_end_v <= kb * 128 + 80 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[43]);
                s_f[44] = ((seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[44]);
                s_f[45] = ((seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[45]);
                s_f[46] = ((seg_end_v <= kb * 128 + 88 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[46]);
                s_f[47] = ((seg_end_v <= kb * 128 + 88 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[47]);
                s_f[48] = ((seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[48]);
                s_f[49] = ((seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[49]);
                s_f[50] = ((seg_end_v <= kb * 128 + 96 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[50]);
                s_f[51] = ((seg_end_v <= kb * 128 + 96 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[51]);
                s_f[52] = ((seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[52]);
                s_f[53] = ((seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[53]);
                s_f[54] = ((seg_end_v <= kb * 128 + 104 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[54]);
                s_f[55] = ((seg_end_v <= kb * 128 + 104 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[55]);
                s_f[56] = ((seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[56]);
                s_f[57] = ((seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[57]);
                s_f[58] = ((seg_end_v <= kb * 128 + 112 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[58]);
                s_f[59] = ((seg_end_v <= kb * 128 + 112 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[59]);
                s_f[60] = ((seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[60]);
                s_f[61] = ((seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[61]);
                s_f[62] = ((seg_end_v <= kb * 128 + 120 + 2 * (lane & 3)) ? -CUDART_INF_F : s_f[62]);
                s_f[63] = ((seg_end_v <= kb * 128 + 120 + 2 * (lane & 3) + 1) ? -CUDART_INF_F : s_f[63]);
            }
            float _max_0 = fmaxf(s_f[0], s_f[1]);
            float _max_1 = fmaxf(s_f[2], s_f[3]);
            float _max_2 = fmaxf(_max_0, fmaxf(s_f[4], s_f[5]));
            float _max_3 = fmaxf(_max_1, fmaxf(s_f[6], s_f[7]));
            float _max_4 = fmaxf(_max_2, fmaxf(s_f[8], s_f[9]));
            float _max_5 = fmaxf(_max_3, fmaxf(s_f[10], s_f[11]));
            float _max_6 = fmaxf(_max_4, fmaxf(s_f[12], s_f[13]));
            float _max_7 = fmaxf(_max_5, fmaxf(s_f[14], s_f[15]));
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, _max_6, 1);
            float _max_8 = fmaxf(_max_6, _shfl_xor_sync_3);
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, _max_7, 1);
            float _max_9 = fmaxf(_max_7, _shfl_xor_sync_4);
            float _max_10 = fmaxf(s_f[16], s_f[17]);
            float _max_11 = fmaxf(s_f[18], s_f[19]);
            float _max_12 = fmaxf(_max_10, fmaxf(s_f[20], s_f[21]));
            float _max_13 = fmaxf(_max_11, fmaxf(s_f[22], s_f[23]));
            float _max_14 = fmaxf(_max_12, fmaxf(s_f[24], s_f[25]));
            float _max_15 = fmaxf(_max_13, fmaxf(s_f[26], s_f[27]));
            float _max_16 = fmaxf(_max_14, fmaxf(s_f[28], s_f[29]));
            float _max_17 = fmaxf(_max_15, fmaxf(s_f[30], s_f[31]));
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, _max_16, 1);
            float _max_18 = fmaxf(_max_16, _shfl_xor_sync_5);
            float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, _max_17, 1);
            float _max_19 = fmaxf(_max_17, _shfl_xor_sync_6);
            float _max_20 = fmaxf(s_f[32], s_f[33]);
            float _max_21 = fmaxf(s_f[34], s_f[35]);
            float _max_22 = fmaxf(_max_20, fmaxf(s_f[36], s_f[37]));
            float _max_23 = fmaxf(_max_21, fmaxf(s_f[38], s_f[39]));
            float _max_24 = fmaxf(_max_22, fmaxf(s_f[40], s_f[41]));
            float _max_25 = fmaxf(_max_23, fmaxf(s_f[42], s_f[43]));
            float _max_26 = fmaxf(_max_24, fmaxf(s_f[44], s_f[45]));
            float _max_27 = fmaxf(_max_25, fmaxf(s_f[46], s_f[47]));
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, _max_26, 1);
            float _max_28 = fmaxf(_max_26, _shfl_xor_sync_7);
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, _max_27, 1);
            float _max_29 = fmaxf(_max_27, _shfl_xor_sync_8);
            float _max_30 = fmaxf(s_f[48], s_f[49]);
            float _max_31 = fmaxf(s_f[50], s_f[51]);
            float _max_32 = fmaxf(_max_30, fmaxf(s_f[52], s_f[53]));
            float _max_33 = fmaxf(_max_31, fmaxf(s_f[54], s_f[55]));
            float _max_34 = fmaxf(_max_32, fmaxf(s_f[56], s_f[57]));
            float _max_35 = fmaxf(_max_33, fmaxf(s_f[58], s_f[59]));
            float _max_36 = fmaxf(_max_34, fmaxf(s_f[60], s_f[61]));
            float _max_37 = fmaxf(_max_35, fmaxf(s_f[62], s_f[63]));
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, _max_36, 1);
            float _max_38 = fmaxf(_max_36, _shfl_xor_sync_9);
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, _max_37, 1);
            float _max_39 = fmaxf(_max_37, _shfl_xor_sync_10);
            float _max_40 = fmaxf(fmaxf(_max_8, _max_18), fmaxf(_max_28, _max_38));
            float _max_41 = fmaxf(fmaxf(_max_9, _max_19), fmaxf(_max_29, _max_39));
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, _max_40, 2);
            float _max_42 = fmaxf(_max_40, _shfl_xor_sync_11);
            float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, _max_41, 2);
            float _max_43 = fmaxf(_max_41, _shfl_xor_sync_12);
            float _max_44 = fmaxf(m_state[0], _max_42 * c_t);
            float _max_45 = fmaxf(m_state[1], _max_43 * c_t);
            float _exp2_0 = approx_exp2(m_state[0] - _max_44);
            float _exp2_1 = approx_exp2(m_state[1] - _max_45);
            int _vote_0 = __any_sync(0xFFFFFFFF, _max_44 > m_state[0] || _max_45 > m_state[1]);
            m_state[0] = _max_44;
            m_state[1] = _max_45;
            if (_vote_0 != 0) {
                {
                    float2 _pair_scale_even2_13 = make_float2(_exp2_0, _exp2_0);
                    float2 _pair_scale_odd2_13 = make_float2(_exp2_1, _exp2_1);
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
            float _exp2_2 = approx_exp2(__fmaf_rn(_max_8, c_t, 11.392318f - _max_44));
            float _exp2_3 = approx_exp2(__fmaf_rn(_max_9, c_t, 11.392318f - _max_45));
            uint16_t _e4m3x2_f32_0;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_exp2_3 * 0.16666667f), "f"(_exp2_2 * 0.16666667f));
            float _fma_0 = __fmaf_rn(_max_8, -c_t, 2.5849626f);
            float _fma_1 = __fmaf_rn(_max_9, -c_t, 2.5849626f);
            float _exp2_4 = approx_exp2(__fmaf_rn(s_f[0], c_t, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_f[0] = _exp2_4;
            float _exp2_5 = approx_exp2(__fmaf_rn(s_f[1], c_t, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_f[1] = _exp2_5;
            float _exp2_6 = approx_exp2(__fmaf_rn(s_f[2], c_t, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_f[2] = _exp2_6;
            float _exp2_7 = approx_exp2(__fmaf_rn(s_f[3], c_t, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_f[3] = _exp2_7;
            float _exp2_8 = approx_exp2(__fmaf_rn(s_f[4], c_t, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_f[4] = _exp2_8;
            float _exp2_9 = approx_exp2(__fmaf_rn(s_f[5], c_t, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_f[5] = _exp2_9;
            float _exp2_10 = approx_exp2(__fmaf_rn(s_f[6], c_t, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_f[6] = _exp2_10;
            float _exp2_11 = approx_exp2(__fmaf_rn(s_f[7], c_t, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_f[7] = _exp2_11;
            float _exp2_12 = approx_exp2(__fmaf_rn(s_f[8], c_t, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_f[8] = _exp2_12;
            float _exp2_13 = approx_exp2(__fmaf_rn(s_f[9], c_t, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_f[9] = _exp2_13;
            float _exp2_14 = approx_exp2(__fmaf_rn(s_f[10], c_t, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_f[10] = _exp2_14;
            float _exp2_15 = approx_exp2(__fmaf_rn(s_f[11], c_t, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_f[11] = _exp2_15;
            float _exp2_16 = approx_exp2(__fmaf_rn(s_f[12], c_t, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_f[12] = _exp2_16;
            float _exp2_17 = approx_exp2(__fmaf_rn(s_f[13], c_t, ((_max_8 > -CUDART_INF_F) ? _fma_0 : 0.0f)));
            s_f[13] = _exp2_17;
            float _exp2_18 = approx_exp2(__fmaf_rn(s_f[14], c_t, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_f[14] = _exp2_18;
            float _exp2_19 = approx_exp2(__fmaf_rn(s_f[15], c_t, ((_max_9 > -CUDART_INF_F) ? _fma_1 : 0.0f)));
            s_f[15] = _exp2_19;
            float _exp2_20 = approx_exp2(__fmaf_rn(_max_18, c_t, 11.392318f - _max_44));
            float _exp2_21 = approx_exp2(__fmaf_rn(_max_19, c_t, 11.392318f - _max_45));
            uint16_t _e4m3x2_f32_1;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_exp2_21 * 0.16666667f), "f"(_exp2_20 * 0.16666667f));
            float _fma_2 = __fmaf_rn(_max_18, -c_t, 2.5849626f);
            float _fma_3 = __fmaf_rn(_max_19, -c_t, 2.5849626f);
            float _exp2_22 = approx_exp2(__fmaf_rn(s_f[16], c_t, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_f[16] = _exp2_22;
            float _exp2_23 = approx_exp2(__fmaf_rn(s_f[17], c_t, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_f[17] = _exp2_23;
            float _exp2_24 = approx_exp2(__fmaf_rn(s_f[18], c_t, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_f[18] = _exp2_24;
            float _exp2_25 = approx_exp2(__fmaf_rn(s_f[19], c_t, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_f[19] = _exp2_25;
            float _exp2_26 = approx_exp2(__fmaf_rn(s_f[20], c_t, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_f[20] = _exp2_26;
            float _exp2_27 = approx_exp2(__fmaf_rn(s_f[21], c_t, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_f[21] = _exp2_27;
            float _exp2_28 = approx_exp2(__fmaf_rn(s_f[22], c_t, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_f[22] = _exp2_28;
            float _exp2_29 = approx_exp2(__fmaf_rn(s_f[23], c_t, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_f[23] = _exp2_29;
            float _exp2_30 = approx_exp2(__fmaf_rn(s_f[24], c_t, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_f[24] = _exp2_30;
            float _exp2_31 = approx_exp2(__fmaf_rn(s_f[25], c_t, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_f[25] = _exp2_31;
            float _exp2_32 = approx_exp2(__fmaf_rn(s_f[26], c_t, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_f[26] = _exp2_32;
            float _exp2_33 = approx_exp2(__fmaf_rn(s_f[27], c_t, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_f[27] = _exp2_33;
            float _exp2_34 = approx_exp2(__fmaf_rn(s_f[28], c_t, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_f[28] = _exp2_34;
            float _exp2_35 = approx_exp2(__fmaf_rn(s_f[29], c_t, ((_max_18 > -CUDART_INF_F) ? _fma_2 : 0.0f)));
            s_f[29] = _exp2_35;
            float _exp2_36 = approx_exp2(__fmaf_rn(s_f[30], c_t, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_f[30] = _exp2_36;
            float _exp2_37 = approx_exp2(__fmaf_rn(s_f[31], c_t, ((_max_19 > -CUDART_INF_F) ? _fma_3 : 0.0f)));
            s_f[31] = _exp2_37;
            float _fma_4 = __fmaf_rn(_exp2_22 + _exp2_23 + _exp2_26 + _exp2_27 + _exp2_30 + _exp2_31 + _exp2_34 + _exp2_35, _exp2_20 * 0.16666667f, (_exp2_4 + _exp2_5 + _exp2_8 + _exp2_9 + _exp2_12 + _exp2_13 + _exp2_16 + _exp2_17) * (_exp2_2 * 0.16666667f));
            float _fma_5 = __fmaf_rn(_exp2_24 + _exp2_25 + _exp2_28 + _exp2_29 + _exp2_32 + _exp2_33 + _exp2_36 + _exp2_37, _exp2_21 * 0.16666667f, (_exp2_6 + _exp2_7 + _exp2_10 + _exp2_11 + _exp2_14 + _exp2_15 + _exp2_18 + _exp2_19) * (_exp2_3 * 0.16666667f));
            unsigned int _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 16), 2);
            psf[0] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 16) | _shfl_xor_sync_14;
            p_tmp[0] = s_f[0];
            p_tmp[1] = s_f[1];
            p_tmp[2] = s_f[4];
            p_tmp[3] = s_f[5];
            p_tmp[4] = s_f[8];
            p_tmp[5] = s_f[9];
            p_tmp[6] = s_f[12];
            p_tmp[7] = s_f[13];
            p_tmp[8] = s_f[2];
            p_tmp[9] = s_f[3];
            p_tmp[10] = s_f[6];
            p_tmp[11] = s_f[7];
            p_tmp[12] = s_f[10];
            p_tmp[13] = s_f[11];
            p_tmp[14] = s_f[14];
            p_tmp[15] = s_f[15];
            p_tmp[16] = s_f[16];
            p_tmp[17] = s_f[17];
            p_tmp[18] = s_f[20];
            p_tmp[19] = s_f[21];
            p_tmp[20] = s_f[24];
            p_tmp[21] = s_f[25];
            p_tmp[22] = s_f[28];
            p_tmp[23] = s_f[29];
            p_tmp[24] = s_f[18];
            p_tmp[25] = s_f[19];
            p_tmp[26] = s_f[22];
            p_tmp[27] = s_f[23];
            p_tmp[28] = s_f[26];
            p_tmp[29] = s_f[27];
            p_tmp[30] = s_f[30];
            p_tmp[31] = s_f[31];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[0]) : "f"(p_tmp[0]), "f"(p_tmp[1]), "f"(p_tmp[2]), "f"(p_tmp[3]), "f"(p_tmp[4]), "f"(p_tmp[5]), "f"(p_tmp[6]), "f"(p_tmp[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[1]) : "f"(p_tmp[8]), "f"(p_tmp[9]), "f"(p_tmp[10]), "f"(p_tmp[11]), "f"(p_tmp[12]), "f"(p_tmp[13]), "f"(p_tmp[14]), "f"(p_tmp[15]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[2]) : "f"(p_tmp[16]), "f"(p_tmp[17]), "f"(p_tmp[18]), "f"(p_tmp[19]), "f"(p_tmp[20]), "f"(p_tmp[21]), "f"(p_tmp[22]), "f"(p_tmp[23]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[3]) : "f"(p_tmp[24]), "f"(p_tmp[25]), "f"(p_tmp[26]), "f"(p_tmp[27]), "f"(p_tmp[28]), "f"(p_tmp[29]), "f"(p_tmp[30]), "f"(p_tmp[31]));
            float _exp2_38 = approx_exp2(__fmaf_rn(_max_28, c_t, 11.392318f - _max_44));
            float _exp2_39 = approx_exp2(__fmaf_rn(_max_29, c_t, 11.392318f - _max_45));
            uint16_t _e4m3x2_f32_2;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_exp2_39 * 0.16666667f), "f"(_exp2_38 * 0.16666667f));
            float _fma_6 = __fmaf_rn(_max_28, -c_t, 2.5849626f);
            float _fma_7 = __fmaf_rn(_max_29, -c_t, 2.5849626f);
            float _exp2_40 = approx_exp2(__fmaf_rn(s_f[32], c_t, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_f[32] = _exp2_40;
            float _exp2_41 = approx_exp2(__fmaf_rn(s_f[33], c_t, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_f[33] = _exp2_41;
            float _exp2_42 = approx_exp2(__fmaf_rn(s_f[34], c_t, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_f[34] = _exp2_42;
            float _exp2_43 = approx_exp2(__fmaf_rn(s_f[35], c_t, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_f[35] = _exp2_43;
            float _exp2_44 = approx_exp2(__fmaf_rn(s_f[36], c_t, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_f[36] = _exp2_44;
            float _exp2_45 = approx_exp2(__fmaf_rn(s_f[37], c_t, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_f[37] = _exp2_45;
            float _exp2_46 = approx_exp2(__fmaf_rn(s_f[38], c_t, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_f[38] = _exp2_46;
            float _exp2_47 = approx_exp2(__fmaf_rn(s_f[39], c_t, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_f[39] = _exp2_47;
            float _exp2_48 = approx_exp2(__fmaf_rn(s_f[40], c_t, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_f[40] = _exp2_48;
            float _exp2_49 = approx_exp2(__fmaf_rn(s_f[41], c_t, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_f[41] = _exp2_49;
            float _exp2_50 = approx_exp2(__fmaf_rn(s_f[42], c_t, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_f[42] = _exp2_50;
            float _exp2_51 = approx_exp2(__fmaf_rn(s_f[43], c_t, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_f[43] = _exp2_51;
            float _exp2_52 = approx_exp2(__fmaf_rn(s_f[44], c_t, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_f[44] = _exp2_52;
            float _exp2_53 = approx_exp2(__fmaf_rn(s_f[45], c_t, ((_max_28 > -CUDART_INF_F) ? _fma_6 : 0.0f)));
            s_f[45] = _exp2_53;
            float _exp2_54 = approx_exp2(__fmaf_rn(s_f[46], c_t, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_f[46] = _exp2_54;
            float _exp2_55 = approx_exp2(__fmaf_rn(s_f[47], c_t, ((_max_29 > -CUDART_INF_F) ? _fma_7 : 0.0f)));
            s_f[47] = _exp2_55;
            float _fma_8 = __fmaf_rn(_exp2_40 + _exp2_41 + _exp2_44 + _exp2_45 + _exp2_48 + _exp2_49 + _exp2_52 + _exp2_53, _exp2_38 * 0.16666667f, _fma_4);
            float _fma_9 = __fmaf_rn(_exp2_42 + _exp2_43 + _exp2_46 + _exp2_47 + _exp2_50 + _exp2_51 + _exp2_54 + _exp2_55, _exp2_39 * 0.16666667f, _fma_5);
            float _exp2_56 = approx_exp2(__fmaf_rn(_max_38, c_t, 11.392318f - _max_44));
            float _exp2_57 = approx_exp2(__fmaf_rn(_max_39, c_t, 11.392318f - _max_45));
            uint16_t _e4m3x2_f32_3;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(_exp2_57 * 0.16666667f), "f"(_exp2_56 * 0.16666667f));
            float _fma_10 = __fmaf_rn(_max_38, -c_t, 2.5849626f);
            float _fma_11 = __fmaf_rn(_max_39, -c_t, 2.5849626f);
            float _exp2_58 = approx_exp2(__fmaf_rn(s_f[48], c_t, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_f[48] = _exp2_58;
            float _exp2_59 = approx_exp2(__fmaf_rn(s_f[49], c_t, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_f[49] = _exp2_59;
            float _exp2_60 = approx_exp2(__fmaf_rn(s_f[50], c_t, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_f[50] = _exp2_60;
            float _exp2_61 = approx_exp2(__fmaf_rn(s_f[51], c_t, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_f[51] = _exp2_61;
            float _exp2_62 = approx_exp2(__fmaf_rn(s_f[52], c_t, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_f[52] = _exp2_62;
            float _exp2_63 = approx_exp2(__fmaf_rn(s_f[53], c_t, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_f[53] = _exp2_63;
            float _exp2_64 = approx_exp2(__fmaf_rn(s_f[54], c_t, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_f[54] = _exp2_64;
            float _exp2_65 = approx_exp2(__fmaf_rn(s_f[55], c_t, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_f[55] = _exp2_65;
            float _exp2_66 = approx_exp2(__fmaf_rn(s_f[56], c_t, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_f[56] = _exp2_66;
            float _exp2_67 = approx_exp2(__fmaf_rn(s_f[57], c_t, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_f[57] = _exp2_67;
            float _exp2_68 = approx_exp2(__fmaf_rn(s_f[58], c_t, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_f[58] = _exp2_68;
            float _exp2_69 = approx_exp2(__fmaf_rn(s_f[59], c_t, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_f[59] = _exp2_69;
            float _exp2_70 = approx_exp2(__fmaf_rn(s_f[60], c_t, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_f[60] = _exp2_70;
            float _exp2_71 = approx_exp2(__fmaf_rn(s_f[61], c_t, ((_max_38 > -CUDART_INF_F) ? _fma_10 : 0.0f)));
            s_f[61] = _exp2_71;
            float _exp2_72 = approx_exp2(__fmaf_rn(s_f[62], c_t, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_f[62] = _exp2_72;
            float _exp2_73 = approx_exp2(__fmaf_rn(s_f[63], c_t, ((_max_39 > -CUDART_INF_F) ? _fma_11 : 0.0f)));
            s_f[63] = _exp2_73;
            float _fma_12 = __fmaf_rn(_exp2_58 + _exp2_59 + _exp2_62 + _exp2_63 + _exp2_66 + _exp2_67 + _exp2_70 + _exp2_71, _exp2_56 * 0.16666667f, _fma_8);
            float _fma_13 = __fmaf_rn(_exp2_60 + _exp2_61 + _exp2_64 + _exp2_65 + _exp2_68 + _exp2_69 + _exp2_72 + _exp2_73, _exp2_57 * 0.16666667f, _fma_9);
            unsigned int _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 16), 2);
            psf[1] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_2 >> 8 : (unsigned int)_e4m3x2_f32_2) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_3 >> 8 : (unsigned int)_e4m3x2_f32_3) & 255) << 16) | _shfl_xor_sync_15;
            p_tmp[0] = s_f[32];
            p_tmp[1] = s_f[33];
            p_tmp[2] = s_f[36];
            p_tmp[3] = s_f[37];
            p_tmp[4] = s_f[40];
            p_tmp[5] = s_f[41];
            p_tmp[6] = s_f[44];
            p_tmp[7] = s_f[45];
            p_tmp[8] = s_f[34];
            p_tmp[9] = s_f[35];
            p_tmp[10] = s_f[38];
            p_tmp[11] = s_f[39];
            p_tmp[12] = s_f[42];
            p_tmp[13] = s_f[43];
            p_tmp[14] = s_f[46];
            p_tmp[15] = s_f[47];
            p_tmp[16] = s_f[48];
            p_tmp[17] = s_f[49];
            p_tmp[18] = s_f[52];
            p_tmp[19] = s_f[53];
            p_tmp[20] = s_f[56];
            p_tmp[21] = s_f[57];
            p_tmp[22] = s_f[60];
            p_tmp[23] = s_f[61];
            p_tmp[24] = s_f[50];
            p_tmp[25] = s_f[51];
            p_tmp[26] = s_f[54];
            p_tmp[27] = s_f[55];
            p_tmp[28] = s_f[58];
            p_tmp[29] = s_f[59];
            p_tmp[30] = s_f[62];
            p_tmp[31] = s_f[63];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[4]) : "f"(p_tmp[0]), "f"(p_tmp[1]), "f"(p_tmp[2]), "f"(p_tmp[3]), "f"(p_tmp[4]), "f"(p_tmp[5]), "f"(p_tmp[6]), "f"(p_tmp[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[5]) : "f"(p_tmp[8]), "f"(p_tmp[9]), "f"(p_tmp[10]), "f"(p_tmp[11]), "f"(p_tmp[12]), "f"(p_tmp[13]), "f"(p_tmp[14]), "f"(p_tmp[15]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[6]) : "f"(p_tmp[16]), "f"(p_tmp[17]), "f"(p_tmp[18]), "f"(p_tmp[19]), "f"(p_tmp[20]), "f"(p_tmp[21]), "f"(p_tmp[22]), "f"(p_tmp[23]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(p_frag[7]) : "f"(p_tmp[24]), "f"(p_tmp[25]), "f"(p_tmp[26]), "f"(p_tmp[27]), "f"(p_tmp[28]), "f"(p_tmp[29]), "f"(p_tmp[30]), "f"(p_tmp[31]));
            l_state[0] = __fmaf_rn(l_state[0], _exp2_0, _fma_12);
            l_state[1] = __fmaf_rn(l_state[1], _exp2_1, _fma_13);
            mbarrier_wait(v_full_addr + (kv_stage) * 8, kv_phase);
            asm volatile("barrier.sync %0, 256;" :: "r"(1 + warp / 4) : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_15[2];
            smem_regs_15[0] = VSF_stage[kv_stage * 256 + (unsigned int)((lane >> 2) * 2)];
            smem_regs_15[1] = VSF_stage[kv_stage * 256 + (unsigned int)((lane >> 2) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[0]), "+f"(o_acc[1]), "+f"(o_acc[2]), "+f"(o_acc[3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_16[2];
            smem_regs_16[0] = VSF_stage[kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2)];
            smem_regs_16[1] = VSF_stage[kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[4]), "+f"(o_acc[(4) + 1]), "+f"(o_acc[(4) + 2]), "+f"(o_acc[(4) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_16[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_17[2];
            smem_regs_17[0] = VSF_stage[kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2)];
            smem_regs_17[1] = VSF_stage[kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[8]), "+f"(o_acc[(8) + 1]), "+f"(o_acc[(8) + 2]), "+f"(o_acc[(8) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_17[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_18[2];
            smem_regs_18[0] = VSF_stage[kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2)];
            smem_regs_18[1] = VSF_stage[kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[12]), "+f"(o_acc[(12) + 1]), "+f"(o_acc[(12) + 2]), "+f"(o_acc[(12) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_18[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_19[2];
            smem_regs_19[0] = VSF_stage[kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2)];
            smem_regs_19[1] = VSF_stage[kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[16]), "+f"(o_acc[(16) + 1]), "+f"(o_acc[(16) + 2]), "+f"(o_acc[(16) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_19[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_20[2];
            smem_regs_20[0] = VSF_stage[kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2)];
            smem_regs_20[1] = VSF_stage[kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[20]), "+f"(o_acc[(20) + 1]), "+f"(o_acc[(20) + 2]), "+f"(o_acc[(20) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_20[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_21[2];
            smem_regs_21[0] = VSF_stage[kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2)];
            smem_regs_21[1] = VSF_stage[kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[24]), "+f"(o_acc[(24) + 1]), "+f"(o_acc[(24) + 2]), "+f"(o_acc[(24) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_21[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_22[2];
            smem_regs_22[0] = VSF_stage[kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2)];
            smem_regs_22[1] = VSF_stage[kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[28]), "+f"(o_acc[(28) + 1]), "+f"(o_acc[(28) + 2]), "+f"(o_acc[(28) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_22[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_23[2];
            smem_regs_23[0] = VSF_stage[kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2)];
            smem_regs_23[1] = VSF_stage[kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_23[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[32]), "+f"(o_acc[(32) + 1]), "+f"(o_acc[(32) + 2]), "+f"(o_acc[(32) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_23[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_24[2];
            smem_regs_24[0] = VSF_stage[kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2)];
            smem_regs_24[1] = VSF_stage[kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_24[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[36]), "+f"(o_acc[(36) + 1]), "+f"(o_acc[(36) + 2]), "+f"(o_acc[(36) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_24[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_25[2];
            smem_regs_25[0] = VSF_stage[kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2)];
            smem_regs_25[1] = VSF_stage[kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_25[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[40]), "+f"(o_acc[(40) + 1]), "+f"(o_acc[(40) + 2]), "+f"(o_acc[(40) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_25[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_26[2];
            smem_regs_26[0] = VSF_stage[kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2)];
            smem_regs_26[1] = VSF_stage[kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_26[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[44]), "+f"(o_acc[(44) + 1]), "+f"(o_acc[(44) + 2]), "+f"(o_acc[(44) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_26[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_27[2];
            smem_regs_27[0] = VSF_stage[kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2)];
            smem_regs_27[1] = VSF_stage[kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[48]), "+f"(o_acc[(48) + 1]), "+f"(o_acc[(48) + 2]), "+f"(o_acc[(48) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_27[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_28[2];
            smem_regs_28[0] = VSF_stage[kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2)];
            smem_regs_28[1] = VSF_stage[kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[52]), "+f"(o_acc[(52) + 1]), "+f"(o_acc[(52) + 2]), "+f"(o_acc[(52) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_28[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                : "r"(VT_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                : "memory");
            unsigned int smem_regs_29[2];
            smem_regs_29[0] = VSF_stage[kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2)];
            smem_regs_29[1] = VSF_stage[kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[56]), "+f"(o_acc[(56) + 1]), "+f"(o_acc[(56) + 2]), "+f"(o_acc[(56) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_29[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            unsigned int smem_regs_30[2];
            smem_regs_30[0] = VSF_stage[kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2)];
            smem_regs_30[1] = VSF_stage[kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2) + 1];
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[0]), "r"(p_frag[1]), "r"(p_frag[2]), "r"(p_frag[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[0])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                : "+f"(o_acc[60]), "+f"(o_acc[(60) + 1]), "+f"(o_acc[(60) + 2]), "+f"(o_acc[(60) + 3])
                : "r"(p_frag[4]), "r"(p_frag[(4) + 1]), "r"(p_frag[(4) + 2]), "r"(p_frag[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_30[1])), "h"((uint16_t)0), "h"((uint16_t)0));
            asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(v_empty_addr + (kv_stage) * 8);
            }
            if (warp == 0) {
                mbarrier_wait(k_empty_addr + (kv_stage) * 8, kv_phase);
            }
            if (kb_last >= kb + 2) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (kv_stage) * 8, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(K8_stage_addr + kv_stage * 16384), "l"((&K_map)), "r"(0), "r"((kb + 2) * 128), "r"(head),
                               "r"(k_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (warp == 4) {
                mbarrier_wait(v_empty_addr + (kv_stage) * 8, kv_phase);
            }
            if (kb_last >= kb + 2) {
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(v_full_addr + (kv_stage) * 8, 9216);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(VT_stage_addr + kv_stage * 8192), "l"((&VT_map)), "r"((kb + 2) * 64), "r"(0), "r"(head),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(VSF_stage_addr + kv_stage * 1024), "l"((&VSF_map)), "r"(0), "r"((head * num_kblocks + (kb + 2)) * 64),
                               "r"(v_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            kv_stage += 1;
            if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
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
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2)))[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2)))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
        }
        o_tmp[0] = o_acc[4] * inv0;
        o_tmp[1] = o_acc[5] * inv0;
        o_tmp[2] = o_acc[6] * inv1;
        o_tmp[3] = o_acc[7] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_2[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 8))[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_3[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 8))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
        }
        o_tmp[0] = o_acc[8] * inv0;
        o_tmp[1] = o_acc[9] * inv0;
        o_tmp[2] = o_acc[10] * inv1;
        o_tmp[3] = o_acc[11] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_4[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 16))[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_5[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_5[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 16))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
        }
        o_tmp[0] = o_acc[12] * inv0;
        o_tmp[1] = o_acc[13] * inv0;
        o_tmp[2] = o_acc[14] * inv1;
        o_tmp[3] = o_acc[15] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_6[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_6[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 24))[0] = reinterpret_cast<unsigned*>(_packed_6)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_7[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_7[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 24))[0] = reinterpret_cast<unsigned*>(_packed_7)[0];
        }
        o_tmp[0] = o_acc[16] * inv0;
        o_tmp[1] = o_acc[17] * inv0;
        o_tmp[2] = o_acc[18] * inv1;
        o_tmp[3] = o_acc[19] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_8[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_8[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 32))[0] = reinterpret_cast<unsigned*>(_packed_8)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_9[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_9[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 32))[0] = reinterpret_cast<unsigned*>(_packed_9)[0];
        }
        o_tmp[0] = o_acc[20] * inv0;
        o_tmp[1] = o_acc[21] * inv0;
        o_tmp[2] = o_acc[22] * inv1;
        o_tmp[3] = o_acc[23] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_10[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_10[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 40))[0] = reinterpret_cast<unsigned*>(_packed_10)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_11[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_11[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 40))[0] = reinterpret_cast<unsigned*>(_packed_11)[0];
        }
        o_tmp[0] = o_acc[24] * inv0;
        o_tmp[1] = o_acc[25] * inv0;
        o_tmp[2] = o_acc[26] * inv1;
        o_tmp[3] = o_acc[27] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_12[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_12[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 48))[0] = reinterpret_cast<unsigned*>(_packed_12)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_13[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_13[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 48))[0] = reinterpret_cast<unsigned*>(_packed_13)[0];
        }
        o_tmp[0] = o_acc[28] * inv0;
        o_tmp[1] = o_acc[29] * inv0;
        o_tmp[2] = o_acc[30] * inv1;
        o_tmp[3] = o_acc[31] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_14[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_14[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 56))[0] = reinterpret_cast<unsigned*>(_packed_14)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_15[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_15[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 56))[0] = reinterpret_cast<unsigned*>(_packed_15)[0];
        }
        o_tmp[0] = o_acc[32] * inv0;
        o_tmp[1] = o_acc[33] * inv0;
        o_tmp[2] = o_acc[34] * inv1;
        o_tmp[3] = o_acc[35] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_16[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_16[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 64))[0] = reinterpret_cast<unsigned*>(_packed_16)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_17[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_17[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 64))[0] = reinterpret_cast<unsigned*>(_packed_17)[0];
        }
        o_tmp[0] = o_acc[36] * inv0;
        o_tmp[1] = o_acc[37] * inv0;
        o_tmp[2] = o_acc[38] * inv1;
        o_tmp[3] = o_acc[39] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_18[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_18[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 72))[0] = reinterpret_cast<unsigned*>(_packed_18)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_19[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_19[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 72))[0] = reinterpret_cast<unsigned*>(_packed_19)[0];
        }
        o_tmp[0] = o_acc[40] * inv0;
        o_tmp[1] = o_acc[41] * inv0;
        o_tmp[2] = o_acc[42] * inv1;
        o_tmp[3] = o_acc[43] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_20[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_20[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 80))[0] = reinterpret_cast<unsigned*>(_packed_20)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_21[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_21[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 80))[0] = reinterpret_cast<unsigned*>(_packed_21)[0];
        }
        o_tmp[0] = o_acc[44] * inv0;
        o_tmp[1] = o_acc[45] * inv0;
        o_tmp[2] = o_acc[46] * inv1;
        o_tmp[3] = o_acc[47] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_22[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_22[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 88))[0] = reinterpret_cast<unsigned*>(_packed_22)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_23[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_23[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 88))[0] = reinterpret_cast<unsigned*>(_packed_23)[0];
        }
        o_tmp[0] = o_acc[48] * inv0;
        o_tmp[1] = o_acc[49] * inv0;
        o_tmp[2] = o_acc[50] * inv1;
        o_tmp[3] = o_acc[51] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_24[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_24[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 96))[0] = reinterpret_cast<unsigned*>(_packed_24)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_25[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_25[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 96))[0] = reinterpret_cast<unsigned*>(_packed_25)[0];
        }
        o_tmp[0] = o_acc[52] * inv0;
        o_tmp[1] = o_acc[53] * inv0;
        o_tmp[2] = o_acc[54] * inv1;
        o_tmp[3] = o_acc[55] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_26[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_26[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 104))[0] = reinterpret_cast<unsigned*>(_packed_26)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_27[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_27[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 104))[0] = reinterpret_cast<unsigned*>(_packed_27)[0];
        }
        o_tmp[0] = o_acc[56] * inv0;
        o_tmp[1] = o_acc[57] * inv0;
        o_tmp[2] = o_acc[58] * inv1;
        o_tmp[3] = o_acc[59] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_28[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_28[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 112))[0] = reinterpret_cast<unsigned*>(_packed_28)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_29[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_29[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 112))[0] = reinterpret_cast<unsigned*>(_packed_29)[0];
        }
        o_tmp[0] = o_acc[60] * inv0;
        o_tmp[1] = o_acc[61] * inv0;
        o_tmp[2] = o_acc[62] * inv1;
        o_tmp[3] = o_acc[63] * inv1;
        if (row0 < seg_end_v) {
            unsigned int _packed_30[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                _packed_30[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row0 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 120))[0] = reinterpret_cast<unsigned*>(_packed_30)[0];
        }
        if (row1 < seg_end_v) {
            unsigned int _packed_31[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                _packed_31[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(O + (obase + (unsigned long long)(row1 - seg_begin_v) * oss + (unsigned long long)((lane & 3) * 2) + 120))[0] = reinterpret_cast<unsigned*>(_packed_31)[0];
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
