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
#define NUM_PREC_STAGES 2
#define SMEM_Q_OFF 1024
#define SMEM_Q_STAGE_BYTES 8192
#define SMEM_Q_STRIDE 8192
#define SMEM_K_STAGE_OFF 9216
#define SMEM_K_STAGE_STAGE_BYTES 8192
#define SMEM_K_STAGE_STRIDE 8192
#define SMEM_KSF_STAGE_OFF 25600
#define SMEM_KSF_STAGE_STAGE_BYTES 1024
#define SMEM_KSF_STAGE_STRIDE 1024
#define SMEM_VT_STAGE_OFF 27648
#define SMEM_VT_STAGE_STAGE_BYTES 8192
#define SMEM_VT_STAGE_STRIDE 8192
#define SMEM_VSF_STAGE_OFF 44032
#define SMEM_VSF_STAGE_STAGE_BYTES 1024
#define SMEM_VSF_STAGE_STRIDE 1024
#define SMEM_DELTA_OFF 46080
#define SMEM_DELTA_STAGE_BYTES 2048
#define SMEM_DELTA_STRIDE 2048
#define SMEM_PREC_OFF 48128
#define SMEM_PREC_STAGE_BYTES 32768
#define SMEM_PREC_STRIDE 32768
#define SMEM_TOTAL 80896
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







__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_nvfp4_varlen_attention_ac4dc40c1d6424bf7950(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap KSF_map, const __grid_constant__ CUtensorMap VT_map, const __grid_constant__ CUtensorMap VSF_map, __nv_bfloat16* __restrict__ O, unsigned int* __restrict__ q_sf, unsigned int* __restrict__ qm4, unsigned int* __restrict__ qm_sf, int* __restrict__ seg_tile_begin, int* __restrict__ unit_table, int total_units, int num_heads, int num_kblocks, float softmax_scale_log2)
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
    #define p_full_addr (mbar_base + 88)
    #define p_empty_addr (mbar_base + 216)

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
    uint8_t* VT_stage = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int VT_stage_addr = smem + 27648;
    unsigned int* VSF_stage = reinterpret_cast<unsigned int*>(smem_raw + 44032);
    const int VSF_stage_addr = smem + 44032;
    float* DELTA = reinterpret_cast<float*>(smem_raw + 46080);
    const int DELTA_addr = smem + 46080;
    unsigned int* PREC = reinterpret_cast<unsigned int*>(smem_raw + 48128);
    const int PREC_addr = smem + 48128;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 43 barriers)
    // Mbarriers at smem_raw[0..344)

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
            // v_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 56, 4);
            mbarrier_init(smem + 64, 4);
            // --- pipeline 'delta' ---
            // delta_ready: 2 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            // p_full: 16 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // p_empty: 16 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: qks ----
    if (warp <= 7) {
        { // qks_main
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            unsigned int d_stage = 0;
            unsigned int d_phase = 0;
            unsigned int p_slot = 0;
            unsigned int p_phase = 0;
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
                if (kb_first <= kb_last) {
                    if (warp == 0) {
                        if (elect_sync()) {
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
                unsigned int q_frags[8];
                unsigned int k_frag[8];
                float s_acc[64];
                float d_tmp[8];
                unsigned int p_frag[8];
                float p_tmp[32];
                unsigned int psf[2];
                float m_state[2];
                float l_state[2];
                float sc_out[2];
                unsigned int mv_out[1];
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
                asm volatile("barrier.sync 5, 256;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
                    mbarrier_wait(k_full_addr + (kv_stage) * 8, kv_phase);
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
                        smem_regs[_lr] = KSF_stage[kv_stage * 256 + (unsigned int)((lane >> 2) * 2) + (unsigned int)_lr];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[0]), "=f"(s_acc[1]), "=f"(s_acc[2]), "=f"(s_acc[3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[0]), "+f"(s_acc[1]), "+f"(s_acc[2]), "+f"(s_acc[3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_0[2];
                    #pragma unroll
                    for (int _lr_1 = 0; _lr_1 < 2; _lr_1++) {
                        smem_regs_0[_lr_1] = KSF_stage[kv_stage * 256 + (unsigned int)((8 + (lane >> 2)) * 2) + (unsigned int)_lr_1];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[4]), "=f"(s_acc[(4) + 1]), "=f"(s_acc[(4) + 2]), "=f"(s_acc[(4) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[4]), "+f"(s_acc[(4) + 1]), "+f"(s_acc[(4) + 2]), "+f"(s_acc[(4) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_1[2];
                    #pragma unroll
                    for (int _lr_2 = 0; _lr_2 < 2; _lr_2++) {
                        smem_regs_1[_lr_2] = KSF_stage[kv_stage * 256 + (unsigned int)((16 + (lane >> 2)) * 2) + (unsigned int)_lr_2];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[8]), "=f"(s_acc[(8) + 1]), "=f"(s_acc[(8) + 2]), "=f"(s_acc[(8) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[8]), "+f"(s_acc[(8) + 1]), "+f"(s_acc[(8) + 2]), "+f"(s_acc[(8) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_2[2];
                    #pragma unroll
                    for (int _lr_3 = 0; _lr_3 < 2; _lr_3++) {
                        smem_regs_2[_lr_3] = KSF_stage[kv_stage * 256 + (unsigned int)((24 + (lane >> 2)) * 2) + (unsigned int)_lr_3];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[12]), "=f"(s_acc[(12) + 1]), "=f"(s_acc[(12) + 2]), "=f"(s_acc[(12) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[12]), "+f"(s_acc[(12) + 1]), "+f"(s_acc[(12) + 2]), "+f"(s_acc[(12) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_3[2];
                    #pragma unroll
                    for (int _lr_4 = 0; _lr_4 < 2; _lr_4++) {
                        smem_regs_3[_lr_4] = KSF_stage[kv_stage * 256 + (unsigned int)((32 + (lane >> 2)) * 2) + (unsigned int)_lr_4];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[16]), "=f"(s_acc[(16) + 1]), "=f"(s_acc[(16) + 2]), "=f"(s_acc[(16) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[16]), "+f"(s_acc[(16) + 1]), "+f"(s_acc[(16) + 2]), "+f"(s_acc[(16) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_4[2];
                    #pragma unroll
                    for (int _lr_5 = 0; _lr_5 < 2; _lr_5++) {
                        smem_regs_4[_lr_5] = KSF_stage[kv_stage * 256 + (unsigned int)((40 + (lane >> 2)) * 2) + (unsigned int)_lr_5];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[20]), "=f"(s_acc[(20) + 1]), "=f"(s_acc[(20) + 2]), "=f"(s_acc[(20) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[20]), "+f"(s_acc[(20) + 1]), "+f"(s_acc[(20) + 2]), "+f"(s_acc[(20) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_5[2];
                    #pragma unroll
                    for (int _lr_6 = 0; _lr_6 < 2; _lr_6++) {
                        smem_regs_5[_lr_6] = KSF_stage[kv_stage * 256 + (unsigned int)((48 + (lane >> 2)) * 2) + (unsigned int)_lr_6];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[24]), "=f"(s_acc[(24) + 1]), "=f"(s_acc[(24) + 2]), "=f"(s_acc[(24) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[24]), "+f"(s_acc[(24) + 1]), "+f"(s_acc[(24) + 2]), "+f"(s_acc[(24) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_6[2];
                    #pragma unroll
                    for (int _lr_7 = 0; _lr_7 < 2; _lr_7++) {
                        smem_regs_6[_lr_7] = KSF_stage[kv_stage * 256 + (unsigned int)((56 + (lane >> 2)) * 2) + (unsigned int)_lr_7];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[28]), "=f"(s_acc[(28) + 1]), "=f"(s_acc[(28) + 2]), "=f"(s_acc[(28) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[28]), "+f"(s_acc[(28) + 1]), "+f"(s_acc[(28) + 2]), "+f"(s_acc[(28) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_7[2];
                    #pragma unroll
                    for (int _lr_8 = 0; _lr_8 < 2; _lr_8++) {
                        smem_regs_7[_lr_8] = KSF_stage[kv_stage * 256 + (unsigned int)((64 + (lane >> 2)) * 2) + (unsigned int)_lr_8];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[32]), "=f"(s_acc[(32) + 1]), "=f"(s_acc[(32) + 2]), "=f"(s_acc[(32) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[32]), "+f"(s_acc[(32) + 1]), "+f"(s_acc[(32) + 2]), "+f"(s_acc[(32) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_8[2];
                    #pragma unroll
                    for (int _lr_9 = 0; _lr_9 < 2; _lr_9++) {
                        smem_regs_8[_lr_9] = KSF_stage[kv_stage * 256 + (unsigned int)((72 + (lane >> 2)) * 2) + (unsigned int)_lr_9];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[36]), "=f"(s_acc[(36) + 1]), "=f"(s_acc[(36) + 2]), "=f"(s_acc[(36) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[36]), "+f"(s_acc[(36) + 1]), "+f"(s_acc[(36) + 2]), "+f"(s_acc[(36) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_9[2];
                    #pragma unroll
                    for (int _lr_10 = 0; _lr_10 < 2; _lr_10++) {
                        smem_regs_9[_lr_10] = KSF_stage[kv_stage * 256 + (unsigned int)((80 + (lane >> 2)) * 2) + (unsigned int)_lr_10];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[40]), "=f"(s_acc[(40) + 1]), "=f"(s_acc[(40) + 2]), "=f"(s_acc[(40) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[40]), "+f"(s_acc[(40) + 1]), "+f"(s_acc[(40) + 2]), "+f"(s_acc[(40) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_10[2];
                    #pragma unroll
                    for (int _lr_11 = 0; _lr_11 < 2; _lr_11++) {
                        smem_regs_10[_lr_11] = KSF_stage[kv_stage * 256 + (unsigned int)((88 + (lane >> 2)) * 2) + (unsigned int)_lr_11];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[44]), "=f"(s_acc[(44) + 1]), "=f"(s_acc[(44) + 2]), "=f"(s_acc[(44) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[44]), "+f"(s_acc[(44) + 1]), "+f"(s_acc[(44) + 2]), "+f"(s_acc[(44) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_11[2];
                    #pragma unroll
                    for (int _lr_12 = 0; _lr_12 < 2; _lr_12++) {
                        smem_regs_11[_lr_12] = KSF_stage[kv_stage * 256 + (unsigned int)((96 + (lane >> 2)) * 2) + (unsigned int)_lr_12];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[48]), "=f"(s_acc[(48) + 1]), "=f"(s_acc[(48) + 2]), "=f"(s_acc[(48) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[48]), "+f"(s_acc[(48) + 1]), "+f"(s_acc[(48) + 2]), "+f"(s_acc[(48) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_12[2];
                    #pragma unroll
                    for (int _lr_13 = 0; _lr_13 < 2; _lr_13++) {
                        smem_regs_12[_lr_13] = KSF_stage[kv_stage * 256 + (unsigned int)((104 + (lane >> 2)) * 2) + (unsigned int)_lr_13];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[52]), "=f"(s_acc[(52) + 1]), "=f"(s_acc[(52) + 2]), "=f"(s_acc[(52) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[52]), "+f"(s_acc[(52) + 1]), "+f"(s_acc[(52) + 2]), "+f"(s_acc[(52) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[0]), "=r"(k_frag[1]), "=r"(k_frag[2]), "=r"(k_frag[3])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(k_frag[4]), "=r"(k_frag[5]), "=r"(k_frag[6]), "=r"(k_frag[7])
                        : "r"(K_stage_addr + kv_stage * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_13[2];
                    #pragma unroll
                    for (int _lr_14 = 0; _lr_14 < 2; _lr_14++) {
                        smem_regs_13[_lr_14] = KSF_stage[kv_stage * 256 + (unsigned int)((112 + (lane >> 2)) * 2) + (unsigned int)_lr_14];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[56]), "=f"(s_acc[(56) + 1]), "=f"(s_acc[(56) + 2]), "=f"(s_acc[(56) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[0]), "r"(k_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[56]), "+f"(s_acc[(56) + 1]), "+f"(s_acc[(56) + 2]), "+f"(s_acc[(56) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[4]), "r"(k_frag[(4) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_14[2];
                    #pragma unroll
                    for (int _lr_15 = 0; _lr_15 < 2; _lr_15++) {
                        smem_regs_14[_lr_15] = KSF_stage[kv_stage * 256 + (unsigned int)((120 + (lane >> 2)) * 2) + (unsigned int)_lr_15];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                        : "=f"(s_acc[60]), "=f"(s_acc[(60) + 1]), "=f"(s_acc[(60) + 2]), "=f"(s_acc[(60) + 3])
                        : "r"(q_frags[0]), "r"(q_frags[1]), "r"(q_frags[2]), "r"(q_frags[3]), "r"(k_frag[2]), "r"(k_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(qsf[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(s_acc[60]), "+f"(s_acc[(60) + 1]), "+f"(s_acc[(60) + 2]), "+f"(s_acc[(60) + 3])
                        : "r"(q_frags[4]), "r"(q_frags[(4) + 1]), "r"(q_frags[(4) + 2]), "r"(q_frags[(4) + 3]), "r"(k_frag[6]), "r"(k_frag[(6) + 1]), "r"((uint32_t)(qsf[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("barrier.arrive %0, 256;" :: "r"(2 - warp / 4) : "memory");
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
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, _max_6, 1);
                    float _max_8 = fmaxf(_max_6, _shfl_xor_sync_3);
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, _max_7, 1);
                    float _max_9 = fmaxf(_max_7, _shfl_xor_sync_4);
                    float _max_10 = fmaxf(s_acc[16], s_acc[17]);
                    float _max_11 = fmaxf(s_acc[18], s_acc[19]);
                    float _max_12 = fmaxf(_max_10, fmaxf(s_acc[20], s_acc[21]));
                    float _max_13 = fmaxf(_max_11, fmaxf(s_acc[22], s_acc[23]));
                    float _max_14 = fmaxf(_max_12, fmaxf(s_acc[24], s_acc[25]));
                    float _max_15 = fmaxf(_max_13, fmaxf(s_acc[26], s_acc[27]));
                    float _max_16 = fmaxf(_max_14, fmaxf(s_acc[28], s_acc[29]));
                    float _max_17 = fmaxf(_max_15, fmaxf(s_acc[30], s_acc[31]));
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, _max_16, 1);
                    float _max_18 = fmaxf(_max_16, _shfl_xor_sync_5);
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, _max_17, 1);
                    float _max_19 = fmaxf(_max_17, _shfl_xor_sync_6);
                    float _max_20 = fmaxf(s_acc[32], s_acc[33]);
                    float _max_21 = fmaxf(s_acc[34], s_acc[35]);
                    float _max_22 = fmaxf(_max_20, fmaxf(s_acc[36], s_acc[37]));
                    float _max_23 = fmaxf(_max_21, fmaxf(s_acc[38], s_acc[39]));
                    float _max_24 = fmaxf(_max_22, fmaxf(s_acc[40], s_acc[41]));
                    float _max_25 = fmaxf(_max_23, fmaxf(s_acc[42], s_acc[43]));
                    float _max_26 = fmaxf(_max_24, fmaxf(s_acc[44], s_acc[45]));
                    float _max_27 = fmaxf(_max_25, fmaxf(s_acc[46], s_acc[47]));
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, _max_26, 1);
                    float _max_28 = fmaxf(_max_26, _shfl_xor_sync_7);
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, _max_27, 1);
                    float _max_29 = fmaxf(_max_27, _shfl_xor_sync_8);
                    float _max_30 = fmaxf(s_acc[48], s_acc[49]);
                    float _max_31 = fmaxf(s_acc[50], s_acc[51]);
                    float _max_32 = fmaxf(_max_30, fmaxf(s_acc[52], s_acc[53]));
                    float _max_33 = fmaxf(_max_31, fmaxf(s_acc[54], s_acc[55]));
                    float _max_34 = fmaxf(_max_32, fmaxf(s_acc[56], s_acc[57]));
                    float _max_35 = fmaxf(_max_33, fmaxf(s_acc[58], s_acc[59]));
                    float _max_36 = fmaxf(_max_34, fmaxf(s_acc[60], s_acc[61]));
                    float _max_37 = fmaxf(_max_35, fmaxf(s_acc[62], s_acc[63]));
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
                    float _max_44 = fmaxf(m_state[0], _max_42 * softmax_scale_log2);
                    float _max_45 = fmaxf(m_state[1], _max_43 * softmax_scale_log2);
                    float _exp2_0 = approx_exp2(m_state[0] - _max_44);
                    float _exp2_1 = approx_exp2(m_state[1] - _max_45);
                    int _vote_0 = __any_sync(0xFFFFFFFF, _max_44 > m_state[0] || _max_45 > m_state[1]);
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(k_empty_addr + (kv_stage) * 8);
                    }
                    m_state[0] = _max_44;
                    m_state[1] = _max_45;
                    sc_out[0] = _exp2_0;
                    sc_out[1] = _exp2_1;
                    mv_out[0] = _vote_0;
                    float _exp2_2 = approx_exp2(__fmaf_rn(_max_8, softmax_scale_log2, 11.392318f - _max_44));
                    float _exp2_3 = approx_exp2(__fmaf_rn(_max_9, softmax_scale_log2, 11.392318f - _max_45));
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_exp2_3 * 0.16666667f), "f"(_exp2_2 * 0.16666667f));
                    float _fma_0 = __fmaf_rn(_max_8, -softmax_scale_log2, 2.5849626f);
                    float _fma_1 = __fmaf_rn(_max_9, -softmax_scale_log2, 2.5849626f);
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
                    float _exp2_20 = approx_exp2(__fmaf_rn(_max_18, softmax_scale_log2, 11.392318f - _max_44));
                    float _exp2_21 = approx_exp2(__fmaf_rn(_max_19, softmax_scale_log2, 11.392318f - _max_45));
                    uint16_t _e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_exp2_21 * 0.16666667f), "f"(_exp2_20 * 0.16666667f));
                    float _fma_2 = __fmaf_rn(_max_18, -softmax_scale_log2, 2.5849626f);
                    float _fma_3 = __fmaf_rn(_max_19, -softmax_scale_log2, 2.5849626f);
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
                    float _fma_4 = __fmaf_rn(_exp2_22 + _exp2_23 + _exp2_26 + _exp2_27 + _exp2_30 + _exp2_31 + _exp2_34 + _exp2_35, _exp2_20 * 0.16666667f, (_exp2_4 + _exp2_5 + _exp2_8 + _exp2_9 + _exp2_12 + _exp2_13 + _exp2_16 + _exp2_17) * (_exp2_2 * 0.16666667f));
                    float _fma_5 = __fmaf_rn(_exp2_24 + _exp2_25 + _exp2_28 + _exp2_29 + _exp2_32 + _exp2_33 + _exp2_36 + _exp2_37, _exp2_21 * 0.16666667f, (_exp2_6 + _exp2_7 + _exp2_10 + _exp2_11 + _exp2_14 + _exp2_15 + _exp2_18 + _exp2_19) * (_exp2_3 * 0.16666667f));
                    unsigned int _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 16), 2);
                    psf[0] = (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) << 8 : (((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_0 >> 8 : (unsigned int)_e4m3x2_f32_0) & 255) | (((lane & 2) == 2) ? ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 24 : ((((lane & 1) == 1) ? (unsigned int)_e4m3x2_f32_1 >> 8 : (unsigned int)_e4m3x2_f32_1) & 255) << 16) | _shfl_xor_sync_13;
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
                    mbarrier_wait(p_empty_addr + (p_slot * 8 + (unsigned int)warp) * 8, p_phase ^ 1);
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane] = p_frag[0];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 32] = p_frag[1];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 64] = p_frag[2];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 96] = p_frag[3];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 128] = p_frag[4];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 160] = p_frag[5];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 192] = p_frag[6];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 224] = p_frag[7];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 256] = psf[0];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 288] = psf[1];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 320] = __as_u32(sc_out[0]);
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 352] = __as_u32(sc_out[1]);
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 384] = mv_out[0];
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 416] = __as_u32(l_state[0]);
                    PREC[(p_slot * 8 + (unsigned int)warp) * 512 + (unsigned int)lane + 448] = __as_u32(l_state[1]);
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (p_slot * 8 + (unsigned int)warp) * 8);
                    }
                    if (warp == 0) {
                        mbarrier_wait(k_empty_addr + (kv_stage) * 8, kv_phase);
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
                    kv_stage += 1;
                    if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
                    p_slot += 1;
                    if (p_slot == 2) { p_slot = 0; p_phase ^= 1; }
                }
                asm volatile("barrier.sync 5, 256;" ::: "memory");
            }
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
        }
    }
    // ---- Role: pv ----
    if (warp >= 8 && warp <= 11) {
        { // pv_main
            unsigned int kv_stage_1 = 0;
            unsigned int kv_phase_1 = 0;
            unsigned int p_slot_1 = 0;
            unsigned int p_phase_1 = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            int pvw = warp - 8;
            #pragma unroll 1
            for (int slot_1 = bid; slot_1 < total_units; slot_1 += num_bids) {
                int _vec_load_3[4];
                {
                    const int4* _ivptr_0 = reinterpret_cast<const int4*>(unit_table + (4 * slot_1) + 0);
                    int4 _ivld_0;
                    _ivld_0 = *_ivptr_0;
                    _vec_load_3[0 + 0] = _ivld_0.x;
                    _vec_load_3[0 + 1] = _ivld_0.y;
                    _vec_load_3[0 + 2] = _ivld_0.z;
                    _vec_load_3[0 + 3] = _ivld_0.w;
                }
                int packed_1 = _vec_load_3[1];
                int head_1 = packed_1 >> 16;
                int q_tile_1 = packed_1 & 65535;
                int seg_begin_v_1 = _vec_load_3[2];
                int seg_end_v_1 = _vec_load_3[3];
                int q_row0_1 = seg_begin_v_1 + q_tile_1 * 128;
                int kb_first_1 = seg_begin_v_1 / 128;
                int kb_last_1 = (seg_end_v_1 - 1) / 128;
                if (kb_first_1 <= kb_last_1) {
                    if (warp == 8) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(v_full_addr + (kv_stage_1) * 8, 9216);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(VT_stage_addr + kv_stage_1 * 8192), "l"((&VT_map)), "r"(kb_first_1 * 64), "r"(0), "r"(head_1),
                                   "r"(v_full_addr + (kv_stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(VSF_stage_addr + kv_stage_1 * 1024), "l"((&VSF_map)), "r"(0), "r"((head_1 * num_kblocks + kb_first_1) * 64),
                                   "r"(v_full_addr + (kv_stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                if (kb_last_1 >= kb_first_1 + 1) {
                    if (warp == 8) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(v_full_addr + ((kv_stage_1 + 1) % 2) * 8, 9216);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(VT_stage_addr + (kv_stage_1 + 1) % 2 * 8192), "l"((&VT_map)), "r"((kb_first_1 + 1) * 64), "r"(0), "r"(head_1),
                                   "r"(v_full_addr + ((kv_stage_1 + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(VSF_stage_addr + (kv_stage_1 + 1) % 2 * 1024), "l"((&VSF_map)), "r"(0), "r"((head_1 * num_kblocks + (kb_first_1 + 1)) * 64),
                                   "r"(v_full_addr + ((kv_stage_1 + 1) % 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                float o_acc0[64];
                float o_acc1[64];
                unsigned int pf0[8];
                unsigned int pf1[8];
                unsigned int psf0[2];
                unsigned int psf1[2];
                float sc0[2];
                float sc1[2];
                float lsum0[2];
                float lsum1[2];
                unsigned int v_frag[8];
                float o_tmp[4];
                o_acc0[0] = 0.0f;
                o_acc0[1] = 0.0f;
                o_acc0[2] = 0.0f;
                o_acc0[3] = 0.0f;
                o_acc0[4] = 0.0f;
                o_acc0[5] = 0.0f;
                o_acc0[6] = 0.0f;
                o_acc0[7] = 0.0f;
                o_acc0[8] = 0.0f;
                o_acc0[9] = 0.0f;
                o_acc0[10] = 0.0f;
                o_acc0[11] = 0.0f;
                o_acc0[12] = 0.0f;
                o_acc0[13] = 0.0f;
                o_acc0[14] = 0.0f;
                o_acc0[15] = 0.0f;
                o_acc0[16] = 0.0f;
                o_acc0[17] = 0.0f;
                o_acc0[18] = 0.0f;
                o_acc0[19] = 0.0f;
                o_acc0[20] = 0.0f;
                o_acc0[21] = 0.0f;
                o_acc0[22] = 0.0f;
                o_acc0[23] = 0.0f;
                o_acc0[24] = 0.0f;
                o_acc0[25] = 0.0f;
                o_acc0[26] = 0.0f;
                o_acc0[27] = 0.0f;
                o_acc0[28] = 0.0f;
                o_acc0[29] = 0.0f;
                o_acc0[30] = 0.0f;
                o_acc0[31] = 0.0f;
                o_acc0[32] = 0.0f;
                o_acc0[33] = 0.0f;
                o_acc0[34] = 0.0f;
                o_acc0[35] = 0.0f;
                o_acc0[36] = 0.0f;
                o_acc0[37] = 0.0f;
                o_acc0[38] = 0.0f;
                o_acc0[39] = 0.0f;
                o_acc0[40] = 0.0f;
                o_acc0[41] = 0.0f;
                o_acc0[42] = 0.0f;
                o_acc0[43] = 0.0f;
                o_acc0[44] = 0.0f;
                o_acc0[45] = 0.0f;
                o_acc0[46] = 0.0f;
                o_acc0[47] = 0.0f;
                o_acc0[48] = 0.0f;
                o_acc0[49] = 0.0f;
                o_acc0[50] = 0.0f;
                o_acc0[51] = 0.0f;
                o_acc0[52] = 0.0f;
                o_acc0[53] = 0.0f;
                o_acc0[54] = 0.0f;
                o_acc0[55] = 0.0f;
                o_acc0[56] = 0.0f;
                o_acc0[57] = 0.0f;
                o_acc0[58] = 0.0f;
                o_acc0[59] = 0.0f;
                o_acc0[60] = 0.0f;
                o_acc0[61] = 0.0f;
                o_acc0[62] = 0.0f;
                o_acc0[63] = 0.0f;
                o_acc1[0] = 0.0f;
                o_acc1[1] = 0.0f;
                o_acc1[2] = 0.0f;
                o_acc1[3] = 0.0f;
                o_acc1[4] = 0.0f;
                o_acc1[5] = 0.0f;
                o_acc1[6] = 0.0f;
                o_acc1[7] = 0.0f;
                o_acc1[8] = 0.0f;
                o_acc1[9] = 0.0f;
                o_acc1[10] = 0.0f;
                o_acc1[11] = 0.0f;
                o_acc1[12] = 0.0f;
                o_acc1[13] = 0.0f;
                o_acc1[14] = 0.0f;
                o_acc1[15] = 0.0f;
                o_acc1[16] = 0.0f;
                o_acc1[17] = 0.0f;
                o_acc1[18] = 0.0f;
                o_acc1[19] = 0.0f;
                o_acc1[20] = 0.0f;
                o_acc1[21] = 0.0f;
                o_acc1[22] = 0.0f;
                o_acc1[23] = 0.0f;
                o_acc1[24] = 0.0f;
                o_acc1[25] = 0.0f;
                o_acc1[26] = 0.0f;
                o_acc1[27] = 0.0f;
                o_acc1[28] = 0.0f;
                o_acc1[29] = 0.0f;
                o_acc1[30] = 0.0f;
                o_acc1[31] = 0.0f;
                o_acc1[32] = 0.0f;
                o_acc1[33] = 0.0f;
                o_acc1[34] = 0.0f;
                o_acc1[35] = 0.0f;
                o_acc1[36] = 0.0f;
                o_acc1[37] = 0.0f;
                o_acc1[38] = 0.0f;
                o_acc1[39] = 0.0f;
                o_acc1[40] = 0.0f;
                o_acc1[41] = 0.0f;
                o_acc1[42] = 0.0f;
                o_acc1[43] = 0.0f;
                o_acc1[44] = 0.0f;
                o_acc1[45] = 0.0f;
                o_acc1[46] = 0.0f;
                o_acc1[47] = 0.0f;
                o_acc1[48] = 0.0f;
                o_acc1[49] = 0.0f;
                o_acc1[50] = 0.0f;
                o_acc1[51] = 0.0f;
                o_acc1[52] = 0.0f;
                o_acc1[53] = 0.0f;
                o_acc1[54] = 0.0f;
                o_acc1[55] = 0.0f;
                o_acc1[56] = 0.0f;
                o_acc1[57] = 0.0f;
                o_acc1[58] = 0.0f;
                o_acc1[59] = 0.0f;
                o_acc1[60] = 0.0f;
                o_acc1[61] = 0.0f;
                o_acc1[62] = 0.0f;
                o_acc1[63] = 0.0f;
                asm volatile("barrier.sync 6, 128;" ::: "memory");
                #pragma unroll 1
                for (int kb_1 = kb_first_1; kb_1 < kb_last_1 + 1; kb_1++) {
                    mbarrier_wait(v_full_addr + (kv_stage_1) * 8, kv_phase_1);
                    mbarrier_wait(p_full_addr + (p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 8, p_phase_1);
                    pf0[0] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane];
                    pf0[1] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 32];
                    pf0[2] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 64];
                    pf0[3] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 96];
                    pf0[4] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 128];
                    pf0[5] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 160];
                    pf0[6] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 192];
                    pf0[7] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 224];
                    psf0[0] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 256];
                    psf0[1] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 288];
                    sc0[0] = __uint_as_float(PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 320]);
                    sc0[1] = __uint_as_float(PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 352]);
                    lsum0[0] = __uint_as_float(PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 416]);
                    lsum0[1] = __uint_as_float(PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 448]);
                    if (PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 512 + (unsigned int)lane + 384] != 0) {
                        {
                            float2 _pair_scale_even2_1 = make_float2(sc0[0], sc0[0]);
                            float2 _pair_scale_odd2_1 = make_float2(sc0[1], sc0[1]);
                            float2* _pair_scale_src2_1 = reinterpret_cast<float2*>(&o_acc0[0]);
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                        }
                    }
                    mbarrier_wait(p_full_addr + (p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 8, p_phase_1);
                    pf1[0] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane];
                    pf1[1] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 32];
                    pf1[2] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 64];
                    pf1[3] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 96];
                    pf1[4] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 128];
                    pf1[5] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 160];
                    pf1[6] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 192];
                    pf1[7] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 224];
                    psf1[0] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 256];
                    psf1[1] = PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 288];
                    sc1[0] = __uint_as_float(PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 320]);
                    sc1[1] = __uint_as_float(PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 352]);
                    lsum1[0] = __uint_as_float(PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 416]);
                    lsum1[1] = __uint_as_float(PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 448]);
                    if (PREC[(p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 512 + (unsigned int)lane + 384] != 0) {
                        {
                            float2 _pair_scale_even2_2 = make_float2(sc1[0], sc1[0]);
                            float2 _pair_scale_odd2_2 = make_float2(sc1[1], sc1[1]);
                            float2* _pair_scale_src2_2 = reinterpret_cast<float2*>(&o_acc1[0]);
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[16]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[17]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[18]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[19]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[20]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[21]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[22]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[23]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[24]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[25]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[26]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[27]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[28]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[29]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[30]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[31]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_empty_addr + (p_slot_1 * 8 + (unsigned int)(pvw * 2)) * 8);
                        mbarrier_arrive(p_empty_addr + (p_slot_1 * 8 + (unsigned int)(pvw * 2 + 1)) * 8);
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)(((lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ ((lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_15[2];
                    #pragma unroll
                    for (int _lr_16 = 0; _lr_16 < 2; _lr_16++) {
                        smem_regs_15[_lr_16] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((lane >> 2) * 2) + (unsigned int)_lr_16];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[0]), "+f"(o_acc0[1]), "+f"(o_acc0[2]), "+f"(o_acc0[3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[0]), "+f"(o_acc0[1]), "+f"(o_acc0[2]), "+f"(o_acc0[3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[0]), "+f"(o_acc1[1]), "+f"(o_acc1[2]), "+f"(o_acc1[3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[0]), "+f"(o_acc1[1]), "+f"(o_acc1[2]), "+f"(o_acc1[3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_15[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_0_1[2];
                    #pragma unroll
                    for (int _lr_17 = 0; _lr_17 < 2; _lr_17++) {
                        smem_regs_0_1[_lr_17] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((8 + (lane >> 2)) * 2) + (unsigned int)_lr_17];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[4]), "+f"(o_acc0[(4) + 1]), "+f"(o_acc0[(4) + 2]), "+f"(o_acc0[(4) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[4]), "+f"(o_acc0[(4) + 1]), "+f"(o_acc0[(4) + 2]), "+f"(o_acc0[(4) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[4]), "+f"(o_acc1[(4) + 1]), "+f"(o_acc1[(4) + 2]), "+f"(o_acc1[(4) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[4]), "+f"(o_acc1[(4) + 1]), "+f"(o_acc1[(4) + 2]), "+f"(o_acc1[(4) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_0_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((16 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (16 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_1_1[2];
                    #pragma unroll
                    for (int _lr_18 = 0; _lr_18 < 2; _lr_18++) {
                        smem_regs_1_1[_lr_18] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((16 + (lane >> 2)) * 2) + (unsigned int)_lr_18];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[8]), "+f"(o_acc0[(8) + 1]), "+f"(o_acc0[(8) + 2]), "+f"(o_acc0[(8) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[8]), "+f"(o_acc0[(8) + 1]), "+f"(o_acc0[(8) + 2]), "+f"(o_acc0[(8) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[8]), "+f"(o_acc1[(8) + 1]), "+f"(o_acc1[(8) + 2]), "+f"(o_acc1[(8) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[8]), "+f"(o_acc1[(8) + 1]), "+f"(o_acc1[(8) + 2]), "+f"(o_acc1[(8) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_1_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_2_1[2];
                    #pragma unroll
                    for (int _lr_19 = 0; _lr_19 < 2; _lr_19++) {
                        smem_regs_2_1[_lr_19] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((24 + (lane >> 2)) * 2) + (unsigned int)_lr_19];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[12]), "+f"(o_acc0[(12) + 1]), "+f"(o_acc0[(12) + 2]), "+f"(o_acc0[(12) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[12]), "+f"(o_acc0[(12) + 1]), "+f"(o_acc0[(12) + 2]), "+f"(o_acc0[(12) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[12]), "+f"(o_acc1[(12) + 1]), "+f"(o_acc1[(12) + 2]), "+f"(o_acc1[(12) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[12]), "+f"(o_acc1[(12) + 1]), "+f"(o_acc1[(12) + 2]), "+f"(o_acc1[(12) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_2_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((32 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (32 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_3_1[2];
                    #pragma unroll
                    for (int _lr_20 = 0; _lr_20 < 2; _lr_20++) {
                        smem_regs_3_1[_lr_20] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((32 + (lane >> 2)) * 2) + (unsigned int)_lr_20];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[16]), "+f"(o_acc0[(16) + 1]), "+f"(o_acc0[(16) + 2]), "+f"(o_acc0[(16) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[16]), "+f"(o_acc0[(16) + 1]), "+f"(o_acc0[(16) + 2]), "+f"(o_acc0[(16) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[16]), "+f"(o_acc1[(16) + 1]), "+f"(o_acc1[(16) + 2]), "+f"(o_acc1[(16) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[16]), "+f"(o_acc1[(16) + 1]), "+f"(o_acc1[(16) + 2]), "+f"(o_acc1[(16) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_3_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_4_1[2];
                    #pragma unroll
                    for (int _lr_21 = 0; _lr_21 < 2; _lr_21++) {
                        smem_regs_4_1[_lr_21] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((40 + (lane >> 2)) * 2) + (unsigned int)_lr_21];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[20]), "+f"(o_acc0[(20) + 1]), "+f"(o_acc0[(20) + 2]), "+f"(o_acc0[(20) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[20]), "+f"(o_acc0[(20) + 1]), "+f"(o_acc0[(20) + 2]), "+f"(o_acc0[(20) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[20]), "+f"(o_acc1[(20) + 1]), "+f"(o_acc1[(20) + 2]), "+f"(o_acc1[(20) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[20]), "+f"(o_acc1[(20) + 1]), "+f"(o_acc1[(20) + 2]), "+f"(o_acc1[(20) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_4_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((48 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (48 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_5_1[2];
                    #pragma unroll
                    for (int _lr_22 = 0; _lr_22 < 2; _lr_22++) {
                        smem_regs_5_1[_lr_22] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((48 + (lane >> 2)) * 2) + (unsigned int)_lr_22];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[24]), "+f"(o_acc0[(24) + 1]), "+f"(o_acc0[(24) + 2]), "+f"(o_acc0[(24) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[24]), "+f"(o_acc0[(24) + 1]), "+f"(o_acc0[(24) + 2]), "+f"(o_acc0[(24) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[24]), "+f"(o_acc1[(24) + 1]), "+f"(o_acc1[(24) + 2]), "+f"(o_acc1[(24) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[24]), "+f"(o_acc1[(24) + 1]), "+f"(o_acc1[(24) + 2]), "+f"(o_acc1[(24) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_5_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_6_1[2];
                    #pragma unroll
                    for (int _lr_23 = 0; _lr_23 < 2; _lr_23++) {
                        smem_regs_6_1[_lr_23] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((56 + (lane >> 2)) * 2) + (unsigned int)_lr_23];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[28]), "+f"(o_acc0[(28) + 1]), "+f"(o_acc0[(28) + 2]), "+f"(o_acc0[(28) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[28]), "+f"(o_acc0[(28) + 1]), "+f"(o_acc0[(28) + 2]), "+f"(o_acc0[(28) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[28]), "+f"(o_acc1[(28) + 1]), "+f"(o_acc1[(28) + 2]), "+f"(o_acc1[(28) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[28]), "+f"(o_acc1[(28) + 1]), "+f"(o_acc1[(28) + 2]), "+f"(o_acc1[(28) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_6_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((64 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (64 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_7_1[2];
                    #pragma unroll
                    for (int _lr_24 = 0; _lr_24 < 2; _lr_24++) {
                        smem_regs_7_1[_lr_24] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((64 + (lane >> 2)) * 2) + (unsigned int)_lr_24];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[32]), "+f"(o_acc0[(32) + 1]), "+f"(o_acc0[(32) + 2]), "+f"(o_acc0[(32) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[32]), "+f"(o_acc0[(32) + 1]), "+f"(o_acc0[(32) + 2]), "+f"(o_acc0[(32) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[32]), "+f"(o_acc1[(32) + 1]), "+f"(o_acc1[(32) + 2]), "+f"(o_acc1[(32) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[32]), "+f"(o_acc1[(32) + 1]), "+f"(o_acc1[(32) + 2]), "+f"(o_acc1[(32) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_7_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_8_1[2];
                    #pragma unroll
                    for (int _lr_25 = 0; _lr_25 < 2; _lr_25++) {
                        smem_regs_8_1[_lr_25] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((72 + (lane >> 2)) * 2) + (unsigned int)_lr_25];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[36]), "+f"(o_acc0[(36) + 1]), "+f"(o_acc0[(36) + 2]), "+f"(o_acc0[(36) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[36]), "+f"(o_acc0[(36) + 1]), "+f"(o_acc0[(36) + 2]), "+f"(o_acc0[(36) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[36]), "+f"(o_acc1[(36) + 1]), "+f"(o_acc1[(36) + 2]), "+f"(o_acc1[(36) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[36]), "+f"(o_acc1[(36) + 1]), "+f"(o_acc1[(36) + 2]), "+f"(o_acc1[(36) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_8_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((80 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (80 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_9_1[2];
                    #pragma unroll
                    for (int _lr_26 = 0; _lr_26 < 2; _lr_26++) {
                        smem_regs_9_1[_lr_26] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((80 + (lane >> 2)) * 2) + (unsigned int)_lr_26];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[40]), "+f"(o_acc0[(40) + 1]), "+f"(o_acc0[(40) + 2]), "+f"(o_acc0[(40) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[40]), "+f"(o_acc0[(40) + 1]), "+f"(o_acc0[(40) + 2]), "+f"(o_acc0[(40) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[40]), "+f"(o_acc1[(40) + 1]), "+f"(o_acc1[(40) + 2]), "+f"(o_acc1[(40) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[40]), "+f"(o_acc1[(40) + 1]), "+f"(o_acc1[(40) + 2]), "+f"(o_acc1[(40) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_9_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_10_1[2];
                    #pragma unroll
                    for (int _lr_27 = 0; _lr_27 < 2; _lr_27++) {
                        smem_regs_10_1[_lr_27] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((88 + (lane >> 2)) * 2) + (unsigned int)_lr_27];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[44]), "+f"(o_acc0[(44) + 1]), "+f"(o_acc0[(44) + 2]), "+f"(o_acc0[(44) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[44]), "+f"(o_acc0[(44) + 1]), "+f"(o_acc0[(44) + 2]), "+f"(o_acc0[(44) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[44]), "+f"(o_acc1[(44) + 1]), "+f"(o_acc1[(44) + 2]), "+f"(o_acc1[(44) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[44]), "+f"(o_acc1[(44) + 1]), "+f"(o_acc1[(44) + 2]), "+f"(o_acc1[(44) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_10_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((96 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (96 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_11_1[2];
                    #pragma unroll
                    for (int _lr_28 = 0; _lr_28 < 2; _lr_28++) {
                        smem_regs_11_1[_lr_28] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((96 + (lane >> 2)) * 2) + (unsigned int)_lr_28];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[48]), "+f"(o_acc0[(48) + 1]), "+f"(o_acc0[(48) + 2]), "+f"(o_acc0[(48) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[48]), "+f"(o_acc0[(48) + 1]), "+f"(o_acc0[(48) + 2]), "+f"(o_acc0[(48) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[48]), "+f"(o_acc1[(48) + 1]), "+f"(o_acc1[(48) + 2]), "+f"(o_acc1[(48) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[48]), "+f"(o_acc1[(48) + 1]), "+f"(o_acc1[(48) + 2]), "+f"(o_acc1[(48) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_11_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_12_1[2];
                    #pragma unroll
                    for (int _lr_29 = 0; _lr_29 < 2; _lr_29++) {
                        smem_regs_12_1[_lr_29] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((104 + (lane >> 2)) * 2) + (unsigned int)_lr_29];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[52]), "+f"(o_acc0[(52) + 1]), "+f"(o_acc0[(52) + 2]), "+f"(o_acc0[(52) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[52]), "+f"(o_acc0[(52) + 1]), "+f"(o_acc0[(52) + 2]), "+f"(o_acc0[(52) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[52]), "+f"(o_acc1[(52) + 1]), "+f"(o_acc1[(52) + 2]), "+f"(o_acc1[(52) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[52]), "+f"(o_acc1[(52) + 1]), "+f"(o_acc1[(52) + 2]), "+f"(o_acc1[(52) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_12_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[0]), "=r"(v_frag[1]), "=r"(v_frag[2]), "=r"(v_frag[3])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)((lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(v_frag[4]), "=r"(v_frag[5]), "=r"(v_frag[6]), "=r"(v_frag[7])
                        : "r"(VT_stage_addr + kv_stage_1 * 8192 + (unsigned int)((112 + (lane >> 4 & 1) * 8 + (lane & 7)) * 64) + (unsigned int)(32 + (lane >> 3 & 1) * 16 ^ (112 + (lane >> 4 & 1) * 8 + (lane & 7) >> 1 & 3) << 4))
                        : "memory");
                    unsigned int smem_regs_13_1[2];
                    #pragma unroll
                    for (int _lr_30 = 0; _lr_30 < 2; _lr_30++) {
                        smem_regs_13_1[_lr_30] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((112 + (lane >> 2)) * 2) + (unsigned int)_lr_30];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[56]), "+f"(o_acc0[(56) + 1]), "+f"(o_acc0[(56) + 2]), "+f"(o_acc0[(56) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[56]), "+f"(o_acc0[(56) + 1]), "+f"(o_acc0[(56) + 2]), "+f"(o_acc0[(56) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[56]), "+f"(o_acc1[(56) + 1]), "+f"(o_acc1[(56) + 2]), "+f"(o_acc1[(56) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[0]), "r"(v_frag[1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[56]), "+f"(o_acc1[(56) + 1]), "+f"(o_acc1[(56) + 2]), "+f"(o_acc1[(56) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[4]), "r"(v_frag[(4) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_13_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    unsigned int smem_regs_14_1[2];
                    #pragma unroll
                    for (int _lr_31 = 0; _lr_31 < 2; _lr_31++) {
                        smem_regs_14_1[_lr_31] = VSF_stage[kv_stage_1 * 256 + (unsigned int)((120 + (lane >> 2)) * 2) + (unsigned int)_lr_31];
                    }
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[60]), "+f"(o_acc0[(60) + 1]), "+f"(o_acc0[(60) + 2]), "+f"(o_acc0[(60) + 3])
                        : "r"(pf0[0]), "r"(pf0[1]), "r"(pf0[2]), "r"(pf0[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc0[60]), "+f"(o_acc0[(60) + 1]), "+f"(o_acc0[(60) + 2]), "+f"(o_acc0[(60) + 3])
                        : "r"(pf0[4]), "r"(pf0[(4) + 1]), "r"(pf0[(4) + 2]), "r"(pf0[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf0[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[60]), "+f"(o_acc1[(60) + 1]), "+f"(o_acc1[(60) + 2]), "+f"(o_acc1[(60) + 3])
                        : "r"(pf1[0]), "r"(pf1[1]), "r"(pf1[2]), "r"(pf1[3]), "r"(v_frag[2]), "r"(v_frag[(2) + 1]), "r"((uint32_t)(psf1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(o_acc1[60]), "+f"(o_acc1[(60) + 1]), "+f"(o_acc1[(60) + 2]), "+f"(o_acc1[(60) + 3])
                        : "r"(pf1[4]), "r"(pf1[(4) + 1]), "r"(pf1[(4) + 2]), "r"(pf1[(4) + 3]), "r"(v_frag[6]), "r"(v_frag[(6) + 1]), "r"((uint32_t)(psf1[1])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(smem_regs_14_1[1])), "h"((uint16_t)0), "h"((uint16_t)0));
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(v_empty_addr + (kv_stage_1) * 8);
                    }
                    if (warp == 8) {
                        mbarrier_wait(v_empty_addr + (kv_stage_1) * 8, kv_phase_1);
                    }
                    if (kb_last_1 >= kb_1 + 2) {
                        if (warp == 8) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(v_full_addr + (kv_stage_1) * 8, 9216);
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(VT_stage_addr + kv_stage_1 * 8192), "l"((&VT_map)), "r"((kb_1 + 2) * 64), "r"(0), "r"(head_1),
                                       "r"(v_full_addr + (kv_stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(VSF_stage_addr + kv_stage_1 * 1024), "l"((&VSF_map)), "r"(0), "r"((head_1 * num_kblocks + (kb_1 + 2)) * 64),
                                       "r"(v_full_addr + (kv_stage_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            }
                        }
                    }
                    kv_stage_1 += 1;
                    if (kv_stage_1 == 2) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
                    p_slot_1 += 1;
                    if (p_slot_1 == 2) { p_slot_1 = 0; p_phase_1 ^= 1; }
                }
                int rows0[2];
                int rows1[2];
                float lred[4];
                float invs[4];
                rows0[0] = q_row0_1 + pvw * 2 * 16 + (lane >> 2);
                rows1[0] = rows0[0] + 8;
                lred[0] = lsum0[0];
                lred[1] = lsum0[1];
                float _shfl_xor_sync_3_1 = __shfl_xor_sync(0xFFFFFFFF, lred[0], 1);
                lred[0] = lred[0] + _shfl_xor_sync_3_1;
                float _shfl_xor_sync_4_1 = __shfl_xor_sync(0xFFFFFFFF, lred[0], 2);
                lred[0] = lred[0] + _shfl_xor_sync_4_1;
                float _shfl_xor_sync_5_1 = __shfl_xor_sync(0xFFFFFFFF, lred[1], 1);
                lred[1] = lred[1] + _shfl_xor_sync_5_1;
                float _shfl_xor_sync_6_1 = __shfl_xor_sync(0xFFFFFFFF, lred[1], 2);
                lred[1] = lred[1] + _shfl_xor_sync_6_1;
                float _rcp_7 = approx_rcp(lred[0]);
                invs[0] = _rcp_7;
                float _rcp_8 = approx_rcp(lred[1]);
                invs[1] = _rcp_8;
                o_tmp[0] = o_acc0[0] * invs[0];
                o_tmp[1] = o_acc0[1] * invs[0];
                o_tmp[2] = o_acc0[2] * invs[1];
                o_tmp[3] = o_acc0[3] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_0[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2))[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
                }
                o_tmp[0] = o_acc0[4] * invs[0];
                o_tmp[1] = o_acc0[5] * invs[0];
                o_tmp[2] = o_acc0[6] * invs[1];
                o_tmp[3] = o_acc0[7] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_2[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 8))[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_3[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 8))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
                }
                o_tmp[0] = o_acc0[8] * invs[0];
                o_tmp[1] = o_acc0[9] * invs[0];
                o_tmp[2] = o_acc0[10] * invs[1];
                o_tmp[3] = o_acc0[11] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_4[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 16))[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_5[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 16))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
                }
                o_tmp[0] = o_acc0[12] * invs[0];
                o_tmp[1] = o_acc0[13] * invs[0];
                o_tmp[2] = o_acc0[14] * invs[1];
                o_tmp[3] = o_acc0[15] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_6[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 24))[0] = reinterpret_cast<unsigned*>(_packed_6)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_7[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 24))[0] = reinterpret_cast<unsigned*>(_packed_7)[0];
                }
                o_tmp[0] = o_acc0[16] * invs[0];
                o_tmp[1] = o_acc0[17] * invs[0];
                o_tmp[2] = o_acc0[18] * invs[1];
                o_tmp[3] = o_acc0[19] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_8[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_8[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_8)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_9[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_9[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_9)[0];
                }
                o_tmp[0] = o_acc0[20] * invs[0];
                o_tmp[1] = o_acc0[21] * invs[0];
                o_tmp[2] = o_acc0[22] * invs[1];
                o_tmp[3] = o_acc0[23] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_10[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_10[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 40))[0] = reinterpret_cast<unsigned*>(_packed_10)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_11[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_11[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 40))[0] = reinterpret_cast<unsigned*>(_packed_11)[0];
                }
                o_tmp[0] = o_acc0[24] * invs[0];
                o_tmp[1] = o_acc0[25] * invs[0];
                o_tmp[2] = o_acc0[26] * invs[1];
                o_tmp[3] = o_acc0[27] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_12[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_12[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_12)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_13[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_13[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_13)[0];
                }
                o_tmp[0] = o_acc0[28] * invs[0];
                o_tmp[1] = o_acc0[29] * invs[0];
                o_tmp[2] = o_acc0[30] * invs[1];
                o_tmp[3] = o_acc0[31] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_14[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_14[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 56))[0] = reinterpret_cast<unsigned*>(_packed_14)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_15[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_15[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 56))[0] = reinterpret_cast<unsigned*>(_packed_15)[0];
                }
                o_tmp[0] = o_acc0[32] * invs[0];
                o_tmp[1] = o_acc0[33] * invs[0];
                o_tmp[2] = o_acc0[34] * invs[1];
                o_tmp[3] = o_acc0[35] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_16[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_16[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 64))[0] = reinterpret_cast<unsigned*>(_packed_16)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_17[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_17[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 64))[0] = reinterpret_cast<unsigned*>(_packed_17)[0];
                }
                o_tmp[0] = o_acc0[36] * invs[0];
                o_tmp[1] = o_acc0[37] * invs[0];
                o_tmp[2] = o_acc0[38] * invs[1];
                o_tmp[3] = o_acc0[39] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_18[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_18[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 72))[0] = reinterpret_cast<unsigned*>(_packed_18)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_19[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_19[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 72))[0] = reinterpret_cast<unsigned*>(_packed_19)[0];
                }
                o_tmp[0] = o_acc0[40] * invs[0];
                o_tmp[1] = o_acc0[41] * invs[0];
                o_tmp[2] = o_acc0[42] * invs[1];
                o_tmp[3] = o_acc0[43] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_20[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_20[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 80))[0] = reinterpret_cast<unsigned*>(_packed_20)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_21[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_21[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 80))[0] = reinterpret_cast<unsigned*>(_packed_21)[0];
                }
                o_tmp[0] = o_acc0[44] * invs[0];
                o_tmp[1] = o_acc0[45] * invs[0];
                o_tmp[2] = o_acc0[46] * invs[1];
                o_tmp[3] = o_acc0[47] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_22[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_22[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 88))[0] = reinterpret_cast<unsigned*>(_packed_22)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_23[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_23[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 88))[0] = reinterpret_cast<unsigned*>(_packed_23)[0];
                }
                o_tmp[0] = o_acc0[48] * invs[0];
                o_tmp[1] = o_acc0[49] * invs[0];
                o_tmp[2] = o_acc0[50] * invs[1];
                o_tmp[3] = o_acc0[51] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_24[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_24[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 96))[0] = reinterpret_cast<unsigned*>(_packed_24)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_25[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_25[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 96))[0] = reinterpret_cast<unsigned*>(_packed_25)[0];
                }
                o_tmp[0] = o_acc0[52] * invs[0];
                o_tmp[1] = o_acc0[53] * invs[0];
                o_tmp[2] = o_acc0[54] * invs[1];
                o_tmp[3] = o_acc0[55] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_26[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_26[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 104))[0] = reinterpret_cast<unsigned*>(_packed_26)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_27[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_27[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 104))[0] = reinterpret_cast<unsigned*>(_packed_27)[0];
                }
                o_tmp[0] = o_acc0[56] * invs[0];
                o_tmp[1] = o_acc0[57] * invs[0];
                o_tmp[2] = o_acc0[58] * invs[1];
                o_tmp[3] = o_acc0[59] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_28[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_28[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 112))[0] = reinterpret_cast<unsigned*>(_packed_28)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_29[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_29[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 112))[0] = reinterpret_cast<unsigned*>(_packed_29)[0];
                }
                o_tmp[0] = o_acc0[60] * invs[0];
                o_tmp[1] = o_acc0[61] * invs[0];
                o_tmp[2] = o_acc0[62] * invs[1];
                o_tmp[3] = o_acc0[63] * invs[1];
                if (rows0[0] < seg_end_v_1) {
                    unsigned int _packed_30[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_30[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 120))[0] = reinterpret_cast<unsigned*>(_packed_30)[0];
                }
                if (rows1[0] < seg_end_v_1) {
                    unsigned int _packed_31[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_31[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[0] * num_heads + head_1) * 128 + (lane & 3) * 2 + 120))[0] = reinterpret_cast<unsigned*>(_packed_31)[0];
                }
                rows0[1] = q_row0_1 + (pvw * 2 + 1) * 16 + (lane >> 2);
                rows1[1] = rows0[1] + 8;
                lred[2] = lsum1[0];
                lred[3] = lsum1[1];
                float _shfl_xor_sync_9_1 = __shfl_xor_sync(0xFFFFFFFF, lred[2], 1);
                lred[2] = lred[2] + _shfl_xor_sync_9_1;
                float _shfl_xor_sync_10_1 = __shfl_xor_sync(0xFFFFFFFF, lred[2], 2);
                lred[2] = lred[2] + _shfl_xor_sync_10_1;
                float _shfl_xor_sync_11_1 = __shfl_xor_sync(0xFFFFFFFF, lred[3], 1);
                lred[3] = lred[3] + _shfl_xor_sync_11_1;
                float _shfl_xor_sync_12_1 = __shfl_xor_sync(0xFFFFFFFF, lred[3], 2);
                lred[3] = lred[3] + _shfl_xor_sync_12_1;
                float _rcp_13 = approx_rcp(lred[2]);
                invs[2] = _rcp_13;
                float _rcp_14 = approx_rcp(lred[3]);
                invs[3] = _rcp_14;
                o_tmp[0] = o_acc1[0] * invs[2];
                o_tmp[1] = o_acc1[1] * invs[2];
                o_tmp[2] = o_acc1[2] * invs[3];
                o_tmp[3] = o_acc1[3] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_32[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_32[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2))[0] = reinterpret_cast<unsigned*>(_packed_32)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_33[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_33[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2))[0] = reinterpret_cast<unsigned*>(_packed_33)[0];
                }
                o_tmp[0] = o_acc1[4] * invs[2];
                o_tmp[1] = o_acc1[5] * invs[2];
                o_tmp[2] = o_acc1[6] * invs[3];
                o_tmp[3] = o_acc1[7] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_34[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_34[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 8))[0] = reinterpret_cast<unsigned*>(_packed_34)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_35[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_35[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 8))[0] = reinterpret_cast<unsigned*>(_packed_35)[0];
                }
                o_tmp[0] = o_acc1[8] * invs[2];
                o_tmp[1] = o_acc1[9] * invs[2];
                o_tmp[2] = o_acc1[10] * invs[3];
                o_tmp[3] = o_acc1[11] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_36[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_36[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 16))[0] = reinterpret_cast<unsigned*>(_packed_36)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_37[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_37[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 16))[0] = reinterpret_cast<unsigned*>(_packed_37)[0];
                }
                o_tmp[0] = o_acc1[12] * invs[2];
                o_tmp[1] = o_acc1[13] * invs[2];
                o_tmp[2] = o_acc1[14] * invs[3];
                o_tmp[3] = o_acc1[15] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_38[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_38[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 24))[0] = reinterpret_cast<unsigned*>(_packed_38)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_39[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_39[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 24))[0] = reinterpret_cast<unsigned*>(_packed_39)[0];
                }
                o_tmp[0] = o_acc1[16] * invs[2];
                o_tmp[1] = o_acc1[17] * invs[2];
                o_tmp[2] = o_acc1[18] * invs[3];
                o_tmp[3] = o_acc1[19] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_40[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_40[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_40)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_41[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_41[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_41)[0];
                }
                o_tmp[0] = o_acc1[20] * invs[2];
                o_tmp[1] = o_acc1[21] * invs[2];
                o_tmp[2] = o_acc1[22] * invs[3];
                o_tmp[3] = o_acc1[23] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_42[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_42[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 40))[0] = reinterpret_cast<unsigned*>(_packed_42)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_43[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_43[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 40))[0] = reinterpret_cast<unsigned*>(_packed_43)[0];
                }
                o_tmp[0] = o_acc1[24] * invs[2];
                o_tmp[1] = o_acc1[25] * invs[2];
                o_tmp[2] = o_acc1[26] * invs[3];
                o_tmp[3] = o_acc1[27] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_44[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_44[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_44)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_45[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_45[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_45)[0];
                }
                o_tmp[0] = o_acc1[28] * invs[2];
                o_tmp[1] = o_acc1[29] * invs[2];
                o_tmp[2] = o_acc1[30] * invs[3];
                o_tmp[3] = o_acc1[31] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_46[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_46[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 56))[0] = reinterpret_cast<unsigned*>(_packed_46)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_47[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_47[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 56))[0] = reinterpret_cast<unsigned*>(_packed_47)[0];
                }
                o_tmp[0] = o_acc1[32] * invs[2];
                o_tmp[1] = o_acc1[33] * invs[2];
                o_tmp[2] = o_acc1[34] * invs[3];
                o_tmp[3] = o_acc1[35] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_48[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_48[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 64))[0] = reinterpret_cast<unsigned*>(_packed_48)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_49[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_49[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 64))[0] = reinterpret_cast<unsigned*>(_packed_49)[0];
                }
                o_tmp[0] = o_acc1[36] * invs[2];
                o_tmp[1] = o_acc1[37] * invs[2];
                o_tmp[2] = o_acc1[38] * invs[3];
                o_tmp[3] = o_acc1[39] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_50[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_50[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 72))[0] = reinterpret_cast<unsigned*>(_packed_50)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_51[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_51[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 72))[0] = reinterpret_cast<unsigned*>(_packed_51)[0];
                }
                o_tmp[0] = o_acc1[40] * invs[2];
                o_tmp[1] = o_acc1[41] * invs[2];
                o_tmp[2] = o_acc1[42] * invs[3];
                o_tmp[3] = o_acc1[43] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_52[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_52[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 80))[0] = reinterpret_cast<unsigned*>(_packed_52)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_53[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_53[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 80))[0] = reinterpret_cast<unsigned*>(_packed_53)[0];
                }
                o_tmp[0] = o_acc1[44] * invs[2];
                o_tmp[1] = o_acc1[45] * invs[2];
                o_tmp[2] = o_acc1[46] * invs[3];
                o_tmp[3] = o_acc1[47] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_54[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_54[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 88))[0] = reinterpret_cast<unsigned*>(_packed_54)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_55[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_55[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 88))[0] = reinterpret_cast<unsigned*>(_packed_55)[0];
                }
                o_tmp[0] = o_acc1[48] * invs[2];
                o_tmp[1] = o_acc1[49] * invs[2];
                o_tmp[2] = o_acc1[50] * invs[3];
                o_tmp[3] = o_acc1[51] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_56[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_56[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 96))[0] = reinterpret_cast<unsigned*>(_packed_56)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_57[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_57[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 96))[0] = reinterpret_cast<unsigned*>(_packed_57)[0];
                }
                o_tmp[0] = o_acc1[52] * invs[2];
                o_tmp[1] = o_acc1[53] * invs[2];
                o_tmp[2] = o_acc1[54] * invs[3];
                o_tmp[3] = o_acc1[55] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_58[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_58[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 104))[0] = reinterpret_cast<unsigned*>(_packed_58)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_59[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_59[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 104))[0] = reinterpret_cast<unsigned*>(_packed_59)[0];
                }
                o_tmp[0] = o_acc1[56] * invs[2];
                o_tmp[1] = o_acc1[57] * invs[2];
                o_tmp[2] = o_acc1[58] * invs[3];
                o_tmp[3] = o_acc1[59] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_60[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_60[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 112))[0] = reinterpret_cast<unsigned*>(_packed_60)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_61[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_61[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 112))[0] = reinterpret_cast<unsigned*>(_packed_61)[0];
                }
                o_tmp[0] = o_acc1[60] * invs[2];
                o_tmp[1] = o_acc1[61] * invs[2];
                o_tmp[2] = o_acc1[62] * invs[3];
                o_tmp[3] = o_acc1[63] * invs[3];
                if (rows0[1] < seg_end_v_1) {
                    unsigned int _packed_62[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
                        _packed_62[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows0[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 120))[0] = reinterpret_cast<unsigned*>(_packed_62)[0];
                }
                if (rows1[1] < seg_end_v_1) {
                    unsigned int _packed_63[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
                        _packed_63[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(O + ((rows1[1] * num_heads + head_1) * 128 + (lane & 3) * 2 + 120))[0] = reinterpret_cast<unsigned*>(_packed_63)[0];
                }
                asm volatile("barrier.sync 6, 128;" ::: "memory");
            }
        }
    }

    // Cleanup
}

} // extern "C"
