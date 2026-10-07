/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define NUM_KV_RING_STAGES 3
#define NUM_XCHG_STAGES 2
#define SMEM_KV_U8_OFF 128
#define SMEM_KV_U8_STAGE_BYTES 35328
#define SMEM_KV_U8_STRIDE 35328
#define SMEM_Q_I32_OFF 35456
#define SMEM_Q_I32_STAGE_BYTES 9216
#define SMEM_Q_I32_STRIDE 9216
#define SMEM_Q_U32_OFF 35456
#define SMEM_Q_U32_STAGE_BYTES 9216
#define SMEM_Q_U32_STRIDE 9216
#define SMEM_S_U8_OFF 44672
#define SMEM_S_U8_STAGE_BYTES 36864
#define SMEM_S_U8_STRIDE 36864
#define SMEM_P_U8_OFF 81536
#define SMEM_P_U8_STAGE_BYTES 3840
#define SMEM_P_U8_STRIDE 3840
#define SMEM_P_U32_OFF 81536
#define SMEM_P_U32_STAGE_BYTES 3840
#define SMEM_P_U32_STRIDE 3840
#define SMEM_ALPHA_F32_OFF 85376
#define SMEM_ALPHA_F32_STAGE_BYTES 192
#define SMEM_ALPHA_F32_STRIDE 192
#define SMEM_ML_F32_OFF 85568
#define SMEM_ML_F32_STAGE_BYTES 128
#define SMEM_ML_F32_STRIDE 128
#define SMEM_ITAB_OFF 85696
#define SMEM_ITAB_STAGE_BYTES 8192
#define SMEM_ITAB_STRIDE 8192
#define SMEM_ML_U32_OFF 85568
#define SMEM_ML_U32_STAGE_BYTES 128
#define SMEM_ML_U32_STRIDE 128
#define SMEM_RING_U32_OFF 128
#define SMEM_RING_U32_STAGE_BYTES 35328
#define SMEM_RING_U32_STRIDE 35328
#define SMEM_VT_U8_OFF 93888
#define SMEM_VT_U8_STAGE_BYTES 98304
#define SMEM_VT_U8_STRIDE 98304
#define SMEM_ZONE_U8_OFF 192192
#define SMEM_ZONE_U8_STAGE_BYTES 34560
#define SMEM_ZONE_U8_STRIDE 34560
#define SMEM_ZONE_U32_OFF 192192
#define SMEM_ZONE_U32_STAGE_BYTES 34560
#define SMEM_ZONE_U32_STRIDE 34560
#define SMEM_TOTAL 226816
#define THREADS 448

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

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
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


__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


extern "C" {

__global__ __launch_bounds__(448, 1) __cluster_dims__(1,8,1) void
kernel_cake_nvfp4_sparse_mla_decode_788a6d133a5a217a83c1(int* __restrict__ q, uint8_t* __restrict__ kv, int* __restrict__ indices, __nv_bfloat16* __restrict__ out, int topk, int keys_per_cta, float qk_scale, float out_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define kv_full_addr (mbar_base + 0)
    #define kv_empty_addr (mbar_base + 24)
    #define s_ready_addr (mbar_base + 48)
    #define p_ready_addr (mbar_base + 64)
    #define zone_ready_addr (mbar_base + 80)
    #define exit_ready_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = ((blockIdx.z * (gridDim.y / 8) + (blockIdx.y / 8)) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * (gridDim.y / 8) * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* kv_u8 = reinterpret_cast<uint8_t*>(smem_raw + 128);
    const int kv_u8_addr = smem + 128;
    int* q_i32 = reinterpret_cast<int*>(smem_raw + 35456);
    const int q_i32_addr = smem + 35456;
    unsigned int* q_u32 = reinterpret_cast<unsigned int*>(smem_raw + 35456);
    const int q_u32_addr = smem + 35456;
    uint8_t* s_u8 = reinterpret_cast<uint8_t*>(smem_raw + 44672);
    const int s_u8_addr = smem + 44672;
    uint8_t* p_u8 = reinterpret_cast<uint8_t*>(smem_raw + 81536);
    const int p_u8_addr = smem + 81536;
    unsigned int* p_u32 = reinterpret_cast<unsigned int*>(smem_raw + 81536);
    const int p_u32_addr = smem + 81536;
    float* alpha_f32 = reinterpret_cast<float*>(smem_raw + 85376);
    const int alpha_f32_addr = smem + 85376;
    float* ml_f32 = reinterpret_cast<float*>(smem_raw + 85568);
    const int ml_f32_addr = smem + 85568;
    int* itab = reinterpret_cast<int*>(smem_raw + 85696);
    const int itab_addr = smem + 85696;
    unsigned int* ml_u32 = reinterpret_cast<unsigned int*>(smem_raw + 85568);
    const int ml_u32_addr = smem + 85568;
    unsigned int* ring_u32 = reinterpret_cast<unsigned int*>(smem_raw + 128);
    const int ring_u32_addr = smem + 128;
    uint8_t* vt_u8 = reinterpret_cast<uint8_t*>(smem_raw + 93888);
    const int vt_u8_addr = smem + 93888;
    uint8_t* zone_u8 = reinterpret_cast<uint8_t*>(smem_raw + 192192);
    const int zone_u8_addr = smem + 192192;
    unsigned int* zone_u32 = reinterpret_cast<unsigned int*>(smem_raw + 192192);
    const int zone_u32_addr = smem + 192192;
    int token = blockIdx.x;
    int cta = blockIdx.y;
    int key_lo = cta * keys_per_cta;
    int _min_0 = ((key_lo + keys_per_cta) < (topk) ? (key_lo + keys_per_cta) : (topk));
    int key_hi = _min_0;
    int _max_0 = ((key_hi - key_lo) > (0) ? (key_hi - key_lo) : (0));
    int n_keys = _max_0;
    int num_stages = n_keys + 31 >> 5;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kv_ring' ---
            // kv_full: 3 barriers, init_count=64
            mbarrier_init(smem + 0, 64);
            mbarrier_init(smem + 8, 64);
            mbarrier_init(smem + 16, 64);
            // kv_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 24, 8);
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            // --- pipeline 'xchg' ---
            // s_ready: 2 barriers, init_count=8
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            // p_ready: 2 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            // zone_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // exit_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // ---- Role: math ----
    if (warp <= 7) {
        { // math_main
            int lane_0 = lane;
            int warp_1 = warp;
            int g = lane_0 >> 2;
            int c = lane_0 & 3;
            int thread = tid;
            int piece_q = 0;
            #pragma unroll
            for (int r = 0; r < 3; r++) {
                piece_q = thread + r * 256;
                if (piece_q < 576) {
                    int _vec_load_1[4];
                    {
                        const int4* _ivptr_0 = reinterpret_cast<const int4*>(q + token * 2304 + piece_q * 4);
                        int4 _ivld_0;
                        _ivld_0 = *_ivptr_0;
                        _vec_load_1[0 + 0] = _ivld_0.x;
                        _vec_load_1[0 + 1] = _ivld_0.y;
                        _vec_load_1[0 + 2] = _ivld_0.z;
                        _vec_load_1[0 + 3] = _ivld_0.w;
                    }
                    int* _sv_ptr_0 = reinterpret_cast<int*>(q_i32 + (piece_q * 4));
                    reinterpret_cast<int4*>(_sv_ptr_0 + 0)[0] = reinterpret_cast<int4*>(_vec_load_1)[0];
                }
            }
            asm volatile("barrier.sync 2, 448;" ::: "memory");
            unsigned int qa[16];
            unsigned int qr[4];
            unsigned int _q_u32_reg_0[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(q_u32);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _q_u32_reg_0[_lr] = _smem_ptr[(g * 144 + (16 * warp_1 + 2 * c)) + _lr];
            }
            unsigned int _q_u32_reg_1[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(q_u32);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _q_u32_reg_1[_lr] = _smem_ptr[((g + 8) * 144 + (16 * warp_1 + 2 * c)) + _lr];
            }
            uint32_t _e4m3x2_to_f16x2_0;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)((uint16_t)(_q_u32_reg_0[0] & 65535))));
            qa[0] = _e4m3x2_to_f16x2_0;
            uint32_t _e4m3x2_to_f16x2_1;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)((uint16_t)(_q_u32_reg_1[0] & 65535))));
            qa[1] = _e4m3x2_to_f16x2_1;
            uint32_t _e4m3x2_to_f16x2_2;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)((uint16_t)(_q_u32_reg_0[0] >> 16))));
            qa[2] = _e4m3x2_to_f16x2_2;
            uint32_t _e4m3x2_to_f16x2_3;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)((uint16_t)(_q_u32_reg_1[0] >> 16))));
            qa[3] = _e4m3x2_to_f16x2_3;
            uint32_t _e4m3x2_to_f16x2_4;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)((uint16_t)(_q_u32_reg_0[1] & 65535))));
            qa[4] = _e4m3x2_to_f16x2_4;
            uint32_t _e4m3x2_to_f16x2_5;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)((uint16_t)(_q_u32_reg_1[1] & 65535))));
            qa[5] = _e4m3x2_to_f16x2_5;
            uint32_t _e4m3x2_to_f16x2_6;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)((uint16_t)(_q_u32_reg_0[1] >> 16))));
            qa[6] = _e4m3x2_to_f16x2_6;
            uint32_t _e4m3x2_to_f16x2_7;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)((uint16_t)(_q_u32_reg_1[1] >> 16))));
            qa[7] = _e4m3x2_to_f16x2_7;
            unsigned int _q_u32_reg_2[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(q_u32);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _q_u32_reg_2[_lr] = _smem_ptr[(g * 144 + (16 * warp_1 + 8 + 2 * c)) + _lr];
            }
            unsigned int _q_u32_reg_3[2];
            {
                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(q_u32);
                #pragma unroll
                for (int _lr = 0; _lr < 2; _lr++)
                    _q_u32_reg_3[_lr] = _smem_ptr[((g + 8) * 144 + (16 * warp_1 + 8 + 2 * c)) + _lr];
            }
            uint32_t _e4m3x2_to_f16x2_8;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)((uint16_t)(_q_u32_reg_2[0] & 65535))));
            qa[8] = _e4m3x2_to_f16x2_8;
            uint32_t _e4m3x2_to_f16x2_9;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)((uint16_t)(_q_u32_reg_3[0] & 65535))));
            qa[9] = _e4m3x2_to_f16x2_9;
            uint32_t _e4m3x2_to_f16x2_10;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)((uint16_t)(_q_u32_reg_2[0] >> 16))));
            qa[10] = _e4m3x2_to_f16x2_10;
            uint32_t _e4m3x2_to_f16x2_11;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)((uint16_t)(_q_u32_reg_3[0] >> 16))));
            qa[11] = _e4m3x2_to_f16x2_11;
            uint32_t _e4m3x2_to_f16x2_12;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_12) : "h"((uint16_t)((uint16_t)(_q_u32_reg_2[1] & 65535))));
            qa[12] = _e4m3x2_to_f16x2_12;
            uint32_t _e4m3x2_to_f16x2_13;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_13) : "h"((uint16_t)((uint16_t)(_q_u32_reg_3[1] & 65535))));
            qa[13] = _e4m3x2_to_f16x2_13;
            uint32_t _e4m3x2_to_f16x2_14;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_14) : "h"((uint16_t)((uint16_t)(_q_u32_reg_2[1] >> 16))));
            qa[14] = _e4m3x2_to_f16x2_14;
            uint32_t _e4m3x2_to_f16x2_15;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_15) : "h"((uint16_t)((uint16_t)(_q_u32_reg_3[1] >> 16))));
            qa[15] = _e4m3x2_to_f16x2_15;
            if (warp_1 < 4) {
                unsigned int _q_u32_reg_4[1];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(q_u32);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _q_u32_reg_4[_lr] = _smem_ptr[(g * 144 + (128 + 4 * warp_1 + c)) + _lr];
                }
                unsigned int _q_u32_reg_5[1];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(q_u32);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _q_u32_reg_5[_lr] = _smem_ptr[((g + 8) * 144 + (128 + 4 * warp_1 + c)) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_16;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_16) : "h"((uint16_t)((uint16_t)(_q_u32_reg_4[0] & 65535))));
                qr[0] = _e4m3x2_to_f16x2_16;
                uint32_t _e4m3x2_to_f16x2_17;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_17) : "h"((uint16_t)((uint16_t)(_q_u32_reg_5[0] & 65535))));
                qr[1] = _e4m3x2_to_f16x2_17;
                uint32_t _e4m3x2_to_f16x2_18;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_18) : "h"((uint16_t)((uint16_t)(_q_u32_reg_4[0] >> 16))));
                qr[2] = _e4m3x2_to_f16x2_18;
                uint32_t _e4m3x2_to_f16x2_19;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_19) : "h"((uint16_t)((uint16_t)(_q_u32_reg_5[0] >> 16))));
                qr[3] = _e4m3x2_to_f16x2_19;
            }
            unsigned int raw[8];
            unsigned int rraw[4];
            unsigned int sc[8];
            unsigned int words[32];
            unsigned int rwords[8];
            float s[16];
            float o[32];
            unsigned int pa[8];
            unsigned int vb[4];
            unsigned int vb1[4];
            unsigned int sacc[16];
            float out8[8];
            float wts[16];
            o[0] = 0.0f;
            o[1] = 0.0f;
            o[2] = 0.0f;
            o[3] = 0.0f;
            o[4] = 0.0f;
            o[5] = 0.0f;
            o[6] = 0.0f;
            o[7] = 0.0f;
            o[8] = 0.0f;
            o[9] = 0.0f;
            o[10] = 0.0f;
            o[11] = 0.0f;
            o[12] = 0.0f;
            o[13] = 0.0f;
            o[14] = 0.0f;
            o[15] = 0.0f;
            o[16] = 0.0f;
            o[17] = 0.0f;
            o[18] = 0.0f;
            o[19] = 0.0f;
            o[20] = 0.0f;
            o[21] = 0.0f;
            o[22] = 0.0f;
            o[23] = 0.0f;
            o[24] = 0.0f;
            o[25] = 0.0f;
            o[26] = 0.0f;
            o[27] = 0.0f;
            o[28] = 0.0f;
            o[29] = 0.0f;
            o[30] = 0.0f;
            o[31] = 0.0f;
            unsigned int m_stage = 0;
            unsigned int m_phase = 0;
            int ring_addr = kv_u8_addr;
            int s_addr = s_u8_addr;
            int p_addr = p_u8_addr;
            int vt_addr = vt_u8_addr + (unsigned int)(warp_1 * 12288);
            int vs_row = (lane_0 & 7) * 128;
            int vsx = (lane_0 >> 3 ^ lane_0 & 7) << 4;
            int vrow = (8 * (lane_0 >> 3 & 1) + (lane_0 & 7)) * 128;
            int vx = lane_0 >> 4 << 4 ^ (lane_0 & 7) << 4;
            int par = 0;
            int vpar = 0;
            int ppar = 0;
            unsigned int xs = 0;
            unsigned int pxs = 0;
            unsigned int pxp = 0;
            int slot_base = 0;
            int _min_1 = ((num_stages) < (2) ? (num_stages) : (2));
            int n_pro = _min_1;
            for (int st = 0; st < n_pro; st++) {
                par = st & 1;
                xs = (unsigned int)par;
                vpar = st % 3;
                mbarrier_wait(kv_full_addr + (m_stage) * 8, m_phase);
                slot_base = (int)m_stage * 11776;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
                    : "r"(ring_addr + slot_base + (8 * (lane_0 >> 3 >> 1) + (lane_0 & 7)) * 368 + 32 * warp_1 + 16 * (lane_0 >> 3 & 1))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(raw[4]), "=r"(raw[5]), "=r"(raw[6]), "=r"(raw[7])
                    : "r"(ring_addr + slot_base + (16 + 8 * (lane_0 >> 3 >> 1) + (lane_0 & 7)) * 368 + 32 * warp_1 + 16 * (lane_0 >> 3 & 1))
                    : "memory");
                if (warp_1 < 4) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(rraw[0]), "=r"(rraw[1]), "=r"(rraw[2]), "=r"(rraw[3])
                        : "r"(ring_addr + slot_base + (8 * (lane_0 >> 3) + (lane_0 & 7)) * 368 + 256 + 16 * warp_1)
                        : "memory");
                }
                uint8_t _kv_u8_reg_0[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_0[_lr] = _smem_ptr[(slot_base + g * 368 + 320 + warp_1 + 8 * (c >> 1)) + _lr];
                }
                uint8_t _kv_u8_reg_1[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_1[_lr] = _smem_ptr[(slot_base + g * 368 + 320 + warp_1 + 8 * (c >> 1) + 16) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_20;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_20) : "h"((uint16_t)((uint16_t)((unsigned int)_kv_u8_reg_0[0] | (unsigned int)_kv_u8_reg_1[0] << 8))));
                uint32_t _prmt_b32_0;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_0) : "r"(_e4m3x2_to_f16x2_20), "r"(_e4m3x2_to_f16x2_20));
                sc[0] = _prmt_b32_0;
                uint32_t _prmt_b32_1;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_1) : "r"(_e4m3x2_to_f16x2_20), "r"(_e4m3x2_to_f16x2_20));
                sc[1] = _prmt_b32_1;
                uint8_t _kv_u8_reg_2[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_2[_lr] = _smem_ptr[(slot_base + (8 + g) * 368 + 320 + warp_1 + 8 * (c >> 1)) + _lr];
                }
                uint8_t _kv_u8_reg_3[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_3[_lr] = _smem_ptr[(slot_base + (8 + g) * 368 + 320 + warp_1 + 8 * (c >> 1) + 16) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_21;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_21) : "h"((uint16_t)((uint16_t)((unsigned int)_kv_u8_reg_2[0] | (unsigned int)_kv_u8_reg_3[0] << 8))));
                uint32_t _prmt_b32_2;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_2) : "r"(_e4m3x2_to_f16x2_21), "r"(_e4m3x2_to_f16x2_21));
                sc[2] = _prmt_b32_2;
                uint32_t _prmt_b32_3;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_3) : "r"(_e4m3x2_to_f16x2_21), "r"(_e4m3x2_to_f16x2_21));
                sc[3] = _prmt_b32_3;
                uint8_t _kv_u8_reg_4[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_4[_lr] = _smem_ptr[(slot_base + (16 + g) * 368 + 320 + warp_1 + 8 * (c >> 1)) + _lr];
                }
                uint8_t _kv_u8_reg_5[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_5[_lr] = _smem_ptr[(slot_base + (16 + g) * 368 + 320 + warp_1 + 8 * (c >> 1) + 16) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_22;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_22) : "h"((uint16_t)((uint16_t)((unsigned int)_kv_u8_reg_4[0] | (unsigned int)_kv_u8_reg_5[0] << 8))));
                uint32_t _prmt_b32_4;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_4) : "r"(_e4m3x2_to_f16x2_22), "r"(_e4m3x2_to_f16x2_22));
                sc[4] = _prmt_b32_4;
                uint32_t _prmt_b32_5;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_5) : "r"(_e4m3x2_to_f16x2_22), "r"(_e4m3x2_to_f16x2_22));
                sc[5] = _prmt_b32_5;
                uint8_t _kv_u8_reg_6[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_6[_lr] = _smem_ptr[(slot_base + (24 + g) * 368 + 320 + warp_1 + 8 * (c >> 1)) + _lr];
                }
                uint8_t _kv_u8_reg_7[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_7[_lr] = _smem_ptr[(slot_base + (24 + g) * 368 + 320 + warp_1 + 8 * (c >> 1) + 16) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_23;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_23) : "h"((uint16_t)((uint16_t)((unsigned int)_kv_u8_reg_6[0] | (unsigned int)_kv_u8_reg_7[0] << 8))));
                uint32_t _prmt_b32_6;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_6) : "r"(_e4m3x2_to_f16x2_23), "r"(_e4m3x2_to_f16x2_23));
                sc[6] = _prmt_b32_6;
                uint32_t _prmt_b32_7;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_7) : "r"(_e4m3x2_to_f16x2_23), "r"(_e4m3x2_to_f16x2_23));
                sc[7] = _prmt_b32_7;
                if (warp_1 < 4) {
                    uint32_t _e4m3x2_to_f16x2_24;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_24) : "h"((uint16_t)((uint16_t)(rraw[0] & 65535))));
                    rwords[0] = _e4m3x2_to_f16x2_24;
                    uint32_t _e4m3x2_to_f16x2_25;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_25) : "h"((uint16_t)((uint16_t)(rraw[0] >> 16))));
                    rwords[1] = _e4m3x2_to_f16x2_25;
                    uint32_t _e4m3x2_to_f16x2_26;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_26) : "h"((uint16_t)((uint16_t)(rraw[1] & 65535))));
                    rwords[2] = _e4m3x2_to_f16x2_26;
                    uint32_t _e4m3x2_to_f16x2_27;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_27) : "h"((uint16_t)((uint16_t)(rraw[1] >> 16))));
                    rwords[3] = _e4m3x2_to_f16x2_27;
                    uint32_t _e4m3x2_to_f16x2_28;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_28) : "h"((uint16_t)((uint16_t)(rraw[2] & 65535))));
                    rwords[4] = _e4m3x2_to_f16x2_28;
                    uint32_t _e4m3x2_to_f16x2_29;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_29) : "h"((uint16_t)((uint16_t)(rraw[2] >> 16))));
                    rwords[5] = _e4m3x2_to_f16x2_29;
                    uint32_t _e4m3x2_to_f16x2_30;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_30) : "h"((uint16_t)((uint16_t)(rraw[3] & 65535))));
                    rwords[6] = _e4m3x2_to_f16x2_30;
                    uint32_t _e4m3x2_to_f16x2_31;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_31) : "h"((uint16_t)((uint16_t)(rraw[3] >> 16))));
                    rwords[7] = _e4m3x2_to_f16x2_31;
                }
                uint32_t _e2m1x2_to_f16x2_0;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_0) : "r"((uint32_t)(raw[0])));
                uint32_t _f16x2_mul_0;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_0) : "r"(_e2m1x2_to_f16x2_0), "r"(sc[0]));
                words[0] = _f16x2_mul_0;
                uint32_t _e2m1x2_to_f16x2_1;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_1) : "r"((uint32_t)(raw[0] >> 8)));
                uint32_t _f16x2_mul_1;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_1) : "r"(_e2m1x2_to_f16x2_1), "r"(sc[0]));
                words[1] = _f16x2_mul_1;
                uint32_t _e2m1x2_to_f16x2_2;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_2) : "r"((uint32_t)(raw[0] >> 16)));
                uint32_t _f16x2_mul_2;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_2) : "r"(_e2m1x2_to_f16x2_2), "r"(sc[0]));
                words[2] = _f16x2_mul_2;
                uint32_t _e2m1x2_to_f16x2_3;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_3) : "r"((uint32_t)(raw[0] >> 24)));
                uint32_t _f16x2_mul_3;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_3) : "r"(_e2m1x2_to_f16x2_3), "r"(sc[0]));
                words[3] = _f16x2_mul_3;
                uint32_t _e2m1x2_to_f16x2_4;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_4) : "r"((uint32_t)(raw[1])));
                uint32_t _f16x2_mul_4;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_4) : "r"(_e2m1x2_to_f16x2_4), "r"(sc[1]));
                words[4] = _f16x2_mul_4;
                uint32_t _e2m1x2_to_f16x2_5;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_5) : "r"((uint32_t)(raw[1] >> 8)));
                uint32_t _f16x2_mul_5;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_5) : "r"(_e2m1x2_to_f16x2_5), "r"(sc[1]));
                words[5] = _f16x2_mul_5;
                uint32_t _e2m1x2_to_f16x2_6;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_6) : "r"((uint32_t)(raw[1] >> 16)));
                uint32_t _f16x2_mul_6;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_6) : "r"(_e2m1x2_to_f16x2_6), "r"(sc[1]));
                words[6] = _f16x2_mul_6;
                uint32_t _e2m1x2_to_f16x2_7;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_7) : "r"((uint32_t)(raw[1] >> 24)));
                uint32_t _f16x2_mul_7;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_7) : "r"(_e2m1x2_to_f16x2_7), "r"(sc[1]));
                words[7] = _f16x2_mul_7;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(s[0]), "=f"(s[1]), "=f"(s[2]), "=f"(s[3])
                    : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(words[0]), "r"(words[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
                    : "r"(qa[4]), "r"(qa[(4) + 1]), "r"(qa[(4) + 2]), "r"(qa[(4) + 3]), "r"(words[2]), "r"(words[(2) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
                    : "r"(qa[8]), "r"(qa[(8) + 1]), "r"(qa[(8) + 2]), "r"(qa[(8) + 3]), "r"(words[4]), "r"(words[(4) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
                    : "r"(qa[12]), "r"(qa[(12) + 1]), "r"(qa[(12) + 2]), "r"(qa[(12) + 3]), "r"(words[6]), "r"(words[(6) + 1]));
                if (warp_1 < 4) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
                        : "r"(qr[0]), "r"(qr[1]), "r"(qr[2]), "r"(qr[3]), "r"(rwords[0]), "r"(rwords[1]));
                }
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(vt_addr + vpar * 4096 + vs_row + (0 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&words[0])), "r"(*reinterpret_cast<const uint32_t*>(&words[1])), "r"(*reinterpret_cast<const uint32_t*>(&words[2])), "r"(*reinterpret_cast<const uint32_t*>(&words[3]))
                    : "memory");
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(vt_addr + vpar * 4096 + vs_row + (64 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&words[4])), "r"(*reinterpret_cast<const uint32_t*>(&words[5])), "r"(*reinterpret_cast<const uint32_t*>(&words[6])), "r"(*reinterpret_cast<const uint32_t*>(&words[7]))
                    : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_addr + par * 18432 + warp_1 * 2304 + lane_0 * 16), "r"(*reinterpret_cast<uint32_t*>(&s[0])), "r"(*reinterpret_cast<uint32_t*>(&s[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s[(0) + 3])));
                uint32_t _e2m1x2_to_f16x2_8;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_8) : "r"((uint32_t)(raw[2])));
                uint32_t _f16x2_mul_8;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_8) : "r"(_e2m1x2_to_f16x2_8), "r"(sc[2]));
                words[8] = _f16x2_mul_8;
                uint32_t _e2m1x2_to_f16x2_9;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_9) : "r"((uint32_t)(raw[2] >> 8)));
                uint32_t _f16x2_mul_9;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_9) : "r"(_e2m1x2_to_f16x2_9), "r"(sc[2]));
                words[9] = _f16x2_mul_9;
                uint32_t _e2m1x2_to_f16x2_10;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_10) : "r"((uint32_t)(raw[2] >> 16)));
                uint32_t _f16x2_mul_10;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_10) : "r"(_e2m1x2_to_f16x2_10), "r"(sc[2]));
                words[10] = _f16x2_mul_10;
                uint32_t _e2m1x2_to_f16x2_11;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_11) : "r"((uint32_t)(raw[2] >> 24)));
                uint32_t _f16x2_mul_11;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_11) : "r"(_e2m1x2_to_f16x2_11), "r"(sc[2]));
                words[11] = _f16x2_mul_11;
                uint32_t _e2m1x2_to_f16x2_12;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_12) : "r"((uint32_t)(raw[3])));
                uint32_t _f16x2_mul_12;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_12) : "r"(_e2m1x2_to_f16x2_12), "r"(sc[3]));
                words[12] = _f16x2_mul_12;
                uint32_t _e2m1x2_to_f16x2_13;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_13) : "r"((uint32_t)(raw[3] >> 8)));
                uint32_t _f16x2_mul_13;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_13) : "r"(_e2m1x2_to_f16x2_13), "r"(sc[3]));
                words[13] = _f16x2_mul_13;
                uint32_t _e2m1x2_to_f16x2_14;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_14) : "r"((uint32_t)(raw[3] >> 16)));
                uint32_t _f16x2_mul_14;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_14) : "r"(_e2m1x2_to_f16x2_14), "r"(sc[3]));
                words[14] = _f16x2_mul_14;
                uint32_t _e2m1x2_to_f16x2_15;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_15) : "r"((uint32_t)(raw[3] >> 24)));
                uint32_t _f16x2_mul_15;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_15) : "r"(_e2m1x2_to_f16x2_15), "r"(sc[3]));
                words[15] = _f16x2_mul_15;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(s[4]), "=f"(s[(4) + 1]), "=f"(s[(4) + 2]), "=f"(s[(4) + 3])
                    : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(words[8]), "r"(words[(8) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
                    : "r"(qa[4]), "r"(qa[(4) + 1]), "r"(qa[(4) + 2]), "r"(qa[(4) + 3]), "r"(words[10]), "r"(words[(10) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
                    : "r"(qa[8]), "r"(qa[(8) + 1]), "r"(qa[(8) + 2]), "r"(qa[(8) + 3]), "r"(words[12]), "r"(words[(12) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
                    : "r"(qa[12]), "r"(qa[(12) + 1]), "r"(qa[(12) + 2]), "r"(qa[(12) + 3]), "r"(words[14]), "r"(words[(14) + 1]));
                if (warp_1 < 4) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
                        : "r"(qr[0]), "r"(qr[1]), "r"(qr[2]), "r"(qr[3]), "r"(rwords[2]), "r"(rwords[(2) + 1]));
                }
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 1024 + vs_row + (0 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&words[8])), "r"(*reinterpret_cast<const uint32_t*>(&words[9])), "r"(*reinterpret_cast<const uint32_t*>(&words[10])), "r"(*reinterpret_cast<const uint32_t*>(&words[11]))
                    : "memory");
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 1024 + vs_row + (64 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&words[12])), "r"(*reinterpret_cast<const uint32_t*>(&words[13])), "r"(*reinterpret_cast<const uint32_t*>(&words[14])), "r"(*reinterpret_cast<const uint32_t*>(&words[15]))
                    : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_addr + par * 18432 + warp_1 * 2304 + 576 + lane_0 * 16), "r"(*reinterpret_cast<uint32_t*>(&s[4])), "r"(*reinterpret_cast<uint32_t*>(&s[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s[(4) + 3])));
                uint32_t _e2m1x2_to_f16x2_16;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_16) : "r"((uint32_t)(raw[4])));
                uint32_t _f16x2_mul_16;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_16) : "r"(_e2m1x2_to_f16x2_16), "r"(sc[4]));
                words[16] = _f16x2_mul_16;
                uint32_t _e2m1x2_to_f16x2_17;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_17) : "r"((uint32_t)(raw[4] >> 8)));
                uint32_t _f16x2_mul_17;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_17) : "r"(_e2m1x2_to_f16x2_17), "r"(sc[4]));
                words[17] = _f16x2_mul_17;
                uint32_t _e2m1x2_to_f16x2_18;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_18) : "r"((uint32_t)(raw[4] >> 16)));
                uint32_t _f16x2_mul_18;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_18) : "r"(_e2m1x2_to_f16x2_18), "r"(sc[4]));
                words[18] = _f16x2_mul_18;
                uint32_t _e2m1x2_to_f16x2_19;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_19) : "r"((uint32_t)(raw[4] >> 24)));
                uint32_t _f16x2_mul_19;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_19) : "r"(_e2m1x2_to_f16x2_19), "r"(sc[4]));
                words[19] = _f16x2_mul_19;
                uint32_t _e2m1x2_to_f16x2_20;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_20) : "r"((uint32_t)(raw[5])));
                uint32_t _f16x2_mul_20;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_20) : "r"(_e2m1x2_to_f16x2_20), "r"(sc[5]));
                words[20] = _f16x2_mul_20;
                uint32_t _e2m1x2_to_f16x2_21;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_21) : "r"((uint32_t)(raw[5] >> 8)));
                uint32_t _f16x2_mul_21;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_21) : "r"(_e2m1x2_to_f16x2_21), "r"(sc[5]));
                words[21] = _f16x2_mul_21;
                uint32_t _e2m1x2_to_f16x2_22;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_22) : "r"((uint32_t)(raw[5] >> 16)));
                uint32_t _f16x2_mul_22;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_22) : "r"(_e2m1x2_to_f16x2_22), "r"(sc[5]));
                words[22] = _f16x2_mul_22;
                uint32_t _e2m1x2_to_f16x2_23;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_23) : "r"((uint32_t)(raw[5] >> 24)));
                uint32_t _f16x2_mul_23;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_23) : "r"(_e2m1x2_to_f16x2_23), "r"(sc[5]));
                words[23] = _f16x2_mul_23;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(s[8]), "=f"(s[(8) + 1]), "=f"(s[(8) + 2]), "=f"(s[(8) + 3])
                    : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(words[16]), "r"(words[(16) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[8]), "+f"(s[(8) + 1]), "+f"(s[(8) + 2]), "+f"(s[(8) + 3])
                    : "r"(qa[4]), "r"(qa[(4) + 1]), "r"(qa[(4) + 2]), "r"(qa[(4) + 3]), "r"(words[18]), "r"(words[(18) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[8]), "+f"(s[(8) + 1]), "+f"(s[(8) + 2]), "+f"(s[(8) + 3])
                    : "r"(qa[8]), "r"(qa[(8) + 1]), "r"(qa[(8) + 2]), "r"(qa[(8) + 3]), "r"(words[20]), "r"(words[(20) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[8]), "+f"(s[(8) + 1]), "+f"(s[(8) + 2]), "+f"(s[(8) + 3])
                    : "r"(qa[12]), "r"(qa[(12) + 1]), "r"(qa[(12) + 2]), "r"(qa[(12) + 3]), "r"(words[22]), "r"(words[(22) + 1]));
                if (warp_1 < 4) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(s[8]), "+f"(s[(8) + 1]), "+f"(s[(8) + 2]), "+f"(s[(8) + 3])
                        : "r"(qr[0]), "r"(qr[1]), "r"(qr[2]), "r"(qr[3]), "r"(rwords[4]), "r"(rwords[(4) + 1]));
                }
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 2048 + vs_row + (0 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&words[16])), "r"(*reinterpret_cast<const uint32_t*>(&words[17])), "r"(*reinterpret_cast<const uint32_t*>(&words[18])), "r"(*reinterpret_cast<const uint32_t*>(&words[19]))
                    : "memory");
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 2048 + vs_row + (64 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&words[20])), "r"(*reinterpret_cast<const uint32_t*>(&words[21])), "r"(*reinterpret_cast<const uint32_t*>(&words[22])), "r"(*reinterpret_cast<const uint32_t*>(&words[23]))
                    : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_addr + par * 18432 + warp_1 * 2304 + 1152 + lane_0 * 16), "r"(*reinterpret_cast<uint32_t*>(&s[8])), "r"(*reinterpret_cast<uint32_t*>(&s[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s[(8) + 3])));
                uint32_t _e2m1x2_to_f16x2_24;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_24) : "r"((uint32_t)(raw[6])));
                uint32_t _f16x2_mul_24;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_24) : "r"(_e2m1x2_to_f16x2_24), "r"(sc[6]));
                words[24] = _f16x2_mul_24;
                uint32_t _e2m1x2_to_f16x2_25;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_25) : "r"((uint32_t)(raw[6] >> 8)));
                uint32_t _f16x2_mul_25;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_25) : "r"(_e2m1x2_to_f16x2_25), "r"(sc[6]));
                words[25] = _f16x2_mul_25;
                uint32_t _e2m1x2_to_f16x2_26;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_26) : "r"((uint32_t)(raw[6] >> 16)));
                uint32_t _f16x2_mul_26;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_26) : "r"(_e2m1x2_to_f16x2_26), "r"(sc[6]));
                words[26] = _f16x2_mul_26;
                uint32_t _e2m1x2_to_f16x2_27;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_27) : "r"((uint32_t)(raw[6] >> 24)));
                uint32_t _f16x2_mul_27;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_27) : "r"(_e2m1x2_to_f16x2_27), "r"(sc[6]));
                words[27] = _f16x2_mul_27;
                uint32_t _e2m1x2_to_f16x2_28;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_28) : "r"((uint32_t)(raw[7])));
                uint32_t _f16x2_mul_28;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_28) : "r"(_e2m1x2_to_f16x2_28), "r"(sc[7]));
                words[28] = _f16x2_mul_28;
                uint32_t _e2m1x2_to_f16x2_29;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_29) : "r"((uint32_t)(raw[7] >> 8)));
                uint32_t _f16x2_mul_29;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_29) : "r"(_e2m1x2_to_f16x2_29), "r"(sc[7]));
                words[29] = _f16x2_mul_29;
                uint32_t _e2m1x2_to_f16x2_30;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_30) : "r"((uint32_t)(raw[7] >> 16)));
                uint32_t _f16x2_mul_30;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_30) : "r"(_e2m1x2_to_f16x2_30), "r"(sc[7]));
                words[30] = _f16x2_mul_30;
                uint32_t _e2m1x2_to_f16x2_31;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_31) : "r"((uint32_t)(raw[7] >> 24)));
                uint32_t _f16x2_mul_31;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_31) : "r"(_e2m1x2_to_f16x2_31), "r"(sc[7]));
                words[31] = _f16x2_mul_31;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(s[12]), "=f"(s[(12) + 1]), "=f"(s[(12) + 2]), "=f"(s[(12) + 3])
                    : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(words[24]), "r"(words[(24) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[12]), "+f"(s[(12) + 1]), "+f"(s[(12) + 2]), "+f"(s[(12) + 3])
                    : "r"(qa[4]), "r"(qa[(4) + 1]), "r"(qa[(4) + 2]), "r"(qa[(4) + 3]), "r"(words[26]), "r"(words[(26) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[12]), "+f"(s[(12) + 1]), "+f"(s[(12) + 2]), "+f"(s[(12) + 3])
                    : "r"(qa[8]), "r"(qa[(8) + 1]), "r"(qa[(8) + 2]), "r"(qa[(8) + 3]), "r"(words[28]), "r"(words[(28) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[12]), "+f"(s[(12) + 1]), "+f"(s[(12) + 2]), "+f"(s[(12) + 3])
                    : "r"(qa[12]), "r"(qa[(12) + 1]), "r"(qa[(12) + 2]), "r"(qa[(12) + 3]), "r"(words[30]), "r"(words[(30) + 1]));
                if (warp_1 < 4) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(s[12]), "+f"(s[(12) + 1]), "+f"(s[(12) + 2]), "+f"(s[(12) + 3])
                        : "r"(qr[0]), "r"(qr[1]), "r"(qr[2]), "r"(qr[3]), "r"(rwords[6]), "r"(rwords[(6) + 1]));
                }
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 3072 + vs_row + (0 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&words[24])), "r"(*reinterpret_cast<const uint32_t*>(&words[25])), "r"(*reinterpret_cast<const uint32_t*>(&words[26])), "r"(*reinterpret_cast<const uint32_t*>(&words[27]))
                    : "memory");
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 3072 + vs_row + (64 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&words[28])), "r"(*reinterpret_cast<const uint32_t*>(&words[29])), "r"(*reinterpret_cast<const uint32_t*>(&words[30])), "r"(*reinterpret_cast<const uint32_t*>(&words[31]))
                    : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_addr + par * 18432 + warp_1 * 2304 + 1728 + lane_0 * 16), "r"(*reinterpret_cast<uint32_t*>(&s[12])), "r"(*reinterpret_cast<uint32_t*>(&s[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s[(12) + 3])));
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(kv_empty_addr + (m_stage) * 8);
                    mbarrier_arrive(s_ready_addr + (xs) * 8);
                }
                m_stage += 1;
                if (m_stage == 3) { m_stage = 0; m_phase ^= 1; }
            }
            for (int st_1 = 2; st_1 < num_stages; st_1++) {
                par = st_1 & 1;
                xs = (unsigned int)par;
                vpar = st_1 % 3;
                ppar = (st_1 - 2) % 3;
                pxs = (unsigned int)(st_1 - 2 & 1);
                pxp = (unsigned int)(st_1 - 2 >> 1 & 1);
                mbarrier_wait(p_ready_addr + (pxs) * 8, pxp);
                mbarrier_wait(kv_full_addr + (m_stage) * 8, m_phase);
                slot_base = (int)m_stage * 11776;
                float _alpha_f32_reg_0[1];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(alpha_f32);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _alpha_f32_reg_0[_lr] = _smem_ptr[(ppar * 16 + g) + _lr];
                }
                float _alpha_f32_reg_1[1];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(alpha_f32);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _alpha_f32_reg_1[_lr] = _smem_ptr[(ppar * 16 + g + 8) + _lr];
                }
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
                    : "r"(ring_addr + slot_base + (8 * (lane_0 >> 3 >> 1) + (lane_0 & 7)) * 368 + 32 * warp_1 + 16 * (lane_0 >> 3 & 1))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(raw[4]), "=r"(raw[5]), "=r"(raw[6]), "=r"(raw[7])
                    : "r"(ring_addr + slot_base + (16 + 8 * (lane_0 >> 3 >> 1) + (lane_0 & 7)) * 368 + 32 * warp_1 + 16 * (lane_0 >> 3 & 1))
                    : "memory");
                if (warp_1 < 4) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(rraw[0]), "=r"(rraw[1]), "=r"(rraw[2]), "=r"(rraw[3])
                        : "r"(ring_addr + slot_base + (8 * (lane_0 >> 3) + (lane_0 & 7)) * 368 + 256 + 16 * warp_1)
                        : "memory");
                }
                uint8_t _kv_u8_reg_8[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_8[_lr] = _smem_ptr[(slot_base + g * 368 + 320 + warp_1 + 8 * (c >> 1)) + _lr];
                }
                uint8_t _kv_u8_reg_9[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_9[_lr] = _smem_ptr[(slot_base + g * 368 + 320 + warp_1 + 8 * (c >> 1) + 16) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_32;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_32) : "h"((uint16_t)((uint16_t)((unsigned int)_kv_u8_reg_8[0] | (unsigned int)_kv_u8_reg_9[0] << 8))));
                uint32_t _prmt_b32_8;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_8) : "r"(_e4m3x2_to_f16x2_32), "r"(_e4m3x2_to_f16x2_32));
                sc[0] = _prmt_b32_8;
                uint32_t _prmt_b32_9;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_9) : "r"(_e4m3x2_to_f16x2_32), "r"(_e4m3x2_to_f16x2_32));
                sc[1] = _prmt_b32_9;
                uint8_t _kv_u8_reg_10[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_10[_lr] = _smem_ptr[(slot_base + (8 + g) * 368 + 320 + warp_1 + 8 * (c >> 1)) + _lr];
                }
                uint8_t _kv_u8_reg_11[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_11[_lr] = _smem_ptr[(slot_base + (8 + g) * 368 + 320 + warp_1 + 8 * (c >> 1) + 16) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_33;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_33) : "h"((uint16_t)((uint16_t)((unsigned int)_kv_u8_reg_10[0] | (unsigned int)_kv_u8_reg_11[0] << 8))));
                uint32_t _prmt_b32_10;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_10) : "r"(_e4m3x2_to_f16x2_33), "r"(_e4m3x2_to_f16x2_33));
                sc[2] = _prmt_b32_10;
                uint32_t _prmt_b32_11;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_11) : "r"(_e4m3x2_to_f16x2_33), "r"(_e4m3x2_to_f16x2_33));
                sc[3] = _prmt_b32_11;
                uint8_t _kv_u8_reg_12[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_12[_lr] = _smem_ptr[(slot_base + (16 + g) * 368 + 320 + warp_1 + 8 * (c >> 1)) + _lr];
                }
                uint8_t _kv_u8_reg_13[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_13[_lr] = _smem_ptr[(slot_base + (16 + g) * 368 + 320 + warp_1 + 8 * (c >> 1) + 16) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_34;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_34) : "h"((uint16_t)((uint16_t)((unsigned int)_kv_u8_reg_12[0] | (unsigned int)_kv_u8_reg_13[0] << 8))));
                uint32_t _prmt_b32_12;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_12) : "r"(_e4m3x2_to_f16x2_34), "r"(_e4m3x2_to_f16x2_34));
                sc[4] = _prmt_b32_12;
                uint32_t _prmt_b32_13;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_13) : "r"(_e4m3x2_to_f16x2_34), "r"(_e4m3x2_to_f16x2_34));
                sc[5] = _prmt_b32_13;
                uint8_t _kv_u8_reg_14[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_14[_lr] = _smem_ptr[(slot_base + (24 + g) * 368 + 320 + warp_1 + 8 * (c >> 1)) + _lr];
                }
                uint8_t _kv_u8_reg_15[1];
                {
                    const uint8_t* _smem_ptr = reinterpret_cast<const uint8_t*>(kv_u8);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _kv_u8_reg_15[_lr] = _smem_ptr[(slot_base + (24 + g) * 368 + 320 + warp_1 + 8 * (c >> 1) + 16) + _lr];
                }
                uint32_t _e4m3x2_to_f16x2_35;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_35) : "h"((uint16_t)((uint16_t)((unsigned int)_kv_u8_reg_14[0] | (unsigned int)_kv_u8_reg_15[0] << 8))));
                uint32_t _prmt_b32_14;
                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_14) : "r"(_e4m3x2_to_f16x2_35), "r"(_e4m3x2_to_f16x2_35));
                sc[6] = _prmt_b32_14;
                uint32_t _prmt_b32_15;
                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_15) : "r"(_e4m3x2_to_f16x2_35), "r"(_e4m3x2_to_f16x2_35));
                sc[7] = _prmt_b32_15;
                o[0] = o[0] * _alpha_f32_reg_0[0];
                o[1] = o[1] * _alpha_f32_reg_0[0];
                o[2] = o[2] * _alpha_f32_reg_1[0];
                o[3] = o[3] * _alpha_f32_reg_1[0];
                o[4] = o[4] * _alpha_f32_reg_0[0];
                o[5] = o[5] * _alpha_f32_reg_0[0];
                o[6] = o[6] * _alpha_f32_reg_1[0];
                o[7] = o[7] * _alpha_f32_reg_1[0];
                o[8] = o[8] * _alpha_f32_reg_0[0];
                o[9] = o[9] * _alpha_f32_reg_0[0];
                o[10] = o[10] * _alpha_f32_reg_1[0];
                o[11] = o[11] * _alpha_f32_reg_1[0];
                o[12] = o[12] * _alpha_f32_reg_0[0];
                o[13] = o[13] * _alpha_f32_reg_0[0];
                o[14] = o[14] * _alpha_f32_reg_1[0];
                o[15] = o[15] * _alpha_f32_reg_1[0];
                o[16] = o[16] * _alpha_f32_reg_0[0];
                o[17] = o[17] * _alpha_f32_reg_0[0];
                o[18] = o[18] * _alpha_f32_reg_1[0];
                o[19] = o[19] * _alpha_f32_reg_1[0];
                o[20] = o[20] * _alpha_f32_reg_0[0];
                o[21] = o[21] * _alpha_f32_reg_0[0];
                o[22] = o[22] * _alpha_f32_reg_1[0];
                o[23] = o[23] * _alpha_f32_reg_1[0];
                o[24] = o[24] * _alpha_f32_reg_0[0];
                o[25] = o[25] * _alpha_f32_reg_0[0];
                o[26] = o[26] * _alpha_f32_reg_1[0];
                o[27] = o[27] * _alpha_f32_reg_1[0];
                o[28] = o[28] * _alpha_f32_reg_0[0];
                o[29] = o[29] * _alpha_f32_reg_0[0];
                o[30] = o[30] * _alpha_f32_reg_1[0];
                o[31] = o[31] * _alpha_f32_reg_1[0];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(pa[0]), "=r"(pa[1]), "=r"(pa[2]), "=r"(pa[3])
                    : "r"(p_addr + ppar * 1280 + (lane_0 & 15) * 80 + ((lane_0 >> 4 & 1) * 16 ^ (lane_0 >> 3 & 1) << 5))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(pa[4]), "=r"(pa[5]), "=r"(pa[6]), "=r"(pa[7])
                    : "r"(p_addr + ppar * 1280 + (lane_0 & 15) * 80 + (32 + (lane_0 >> 4 & 1) * 16 ^ (lane_0 >> 3 & 1) << 5))
                    : "memory");
                if (warp_1 < 4) {
                    uint32_t _e4m3x2_to_f16x2_36;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_36) : "h"((uint16_t)((uint16_t)(rraw[0] & 65535))));
                    rwords[0] = _e4m3x2_to_f16x2_36;
                    uint32_t _e4m3x2_to_f16x2_37;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_37) : "h"((uint16_t)((uint16_t)(rraw[0] >> 16))));
                    rwords[1] = _e4m3x2_to_f16x2_37;
                    uint32_t _e4m3x2_to_f16x2_38;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_38) : "h"((uint16_t)((uint16_t)(rraw[1] & 65535))));
                    rwords[2] = _e4m3x2_to_f16x2_38;
                    uint32_t _e4m3x2_to_f16x2_39;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_39) : "h"((uint16_t)((uint16_t)(rraw[1] >> 16))));
                    rwords[3] = _e4m3x2_to_f16x2_39;
                    uint32_t _e4m3x2_to_f16x2_40;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_40) : "h"((uint16_t)((uint16_t)(rraw[2] & 65535))));
                    rwords[4] = _e4m3x2_to_f16x2_40;
                    uint32_t _e4m3x2_to_f16x2_41;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_41) : "h"((uint16_t)((uint16_t)(rraw[2] >> 16))));
                    rwords[5] = _e4m3x2_to_f16x2_41;
                    uint32_t _e4m3x2_to_f16x2_42;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_42) : "h"((uint16_t)((uint16_t)(rraw[3] & 65535))));
                    rwords[6] = _e4m3x2_to_f16x2_42;
                    uint32_t _e4m3x2_to_f16x2_43;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_43) : "h"((uint16_t)((uint16_t)(rraw[3] >> 16))));
                    rwords[7] = _e4m3x2_to_f16x2_43;
                }
                uint32_t _e2m1x2_to_f16x2_32;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_32) : "r"((uint32_t)(raw[0])));
                uint32_t _f16x2_mul_32;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_32) : "r"(_e2m1x2_to_f16x2_32), "r"(sc[0]));
                words[0] = _f16x2_mul_32;
                uint32_t _e2m1x2_to_f16x2_33;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_33) : "r"((uint32_t)(raw[0] >> 8)));
                uint32_t _f16x2_mul_33;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_33) : "r"(_e2m1x2_to_f16x2_33), "r"(sc[0]));
                words[1] = _f16x2_mul_33;
                uint32_t _e2m1x2_to_f16x2_34;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_34) : "r"((uint32_t)(raw[0] >> 16)));
                uint32_t _f16x2_mul_34;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_34) : "r"(_e2m1x2_to_f16x2_34), "r"(sc[0]));
                words[2] = _f16x2_mul_34;
                uint32_t _e2m1x2_to_f16x2_35;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_35) : "r"((uint32_t)(raw[0] >> 24)));
                uint32_t _f16x2_mul_35;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_35) : "r"(_e2m1x2_to_f16x2_35), "r"(sc[0]));
                words[3] = _f16x2_mul_35;
                uint32_t _e2m1x2_to_f16x2_36;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_36) : "r"((uint32_t)(raw[1])));
                uint32_t _f16x2_mul_36;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_36) : "r"(_e2m1x2_to_f16x2_36), "r"(sc[1]));
                words[4] = _f16x2_mul_36;
                uint32_t _e2m1x2_to_f16x2_37;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_37) : "r"((uint32_t)(raw[1] >> 8)));
                uint32_t _f16x2_mul_37;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_37) : "r"(_e2m1x2_to_f16x2_37), "r"(sc[1]));
                words[5] = _f16x2_mul_37;
                uint32_t _e2m1x2_to_f16x2_38;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_38) : "r"((uint32_t)(raw[1] >> 16)));
                uint32_t _f16x2_mul_38;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_38) : "r"(_e2m1x2_to_f16x2_38), "r"(sc[1]));
                words[6] = _f16x2_mul_38;
                uint32_t _e2m1x2_to_f16x2_39;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_39) : "r"((uint32_t)(raw[1] >> 24)));
                uint32_t _f16x2_mul_39;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_39) : "r"(_e2m1x2_to_f16x2_39), "r"(sc[1]));
                words[7] = _f16x2_mul_39;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + vrow + (0 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(s[0]), "=f"(s[1]), "=f"(s[2]), "=f"(s[3])
                    : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(words[0]), "r"(words[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
                    : "r"(qa[4]), "r"(qa[(4) + 1]), "r"(qa[(4) + 2]), "r"(qa[(4) + 3]), "r"(words[2]), "r"(words[(2) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
                    : "r"(qa[8]), "r"(qa[(8) + 1]), "r"(qa[(8) + 2]), "r"(qa[(8) + 3]), "r"(words[4]), "r"(words[(4) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
                    : "r"(qa[12]), "r"(qa[(12) + 1]), "r"(qa[(12) + 2]), "r"(qa[(12) + 3]), "r"(words[6]), "r"(words[(6) + 1]));
                if (warp_1 < 4) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3])
                        : "r"(qr[0]), "r"(qr[1]), "r"(qr[2]), "r"(qr[3]), "r"(rwords[0]), "r"(rwords[1]));
                }
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb1[0]), "=r"(vb1[1]), "=r"(vb1[2]), "=r"(vb1[3])
                    : "r"(vt_addr + ppar * 4096 + vrow + (32 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb1[0]), "r"(vb1[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb1[2]), "r"(vb1[(2) + 1]));
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(vt_addr + vpar * 4096 + vs_row + (0 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&words[0])), "r"(*reinterpret_cast<const uint32_t*>(&words[1])), "r"(*reinterpret_cast<const uint32_t*>(&words[2])), "r"(*reinterpret_cast<const uint32_t*>(&words[3]))
                    : "memory");
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(vt_addr + vpar * 4096 + vs_row + (64 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&words[4])), "r"(*reinterpret_cast<const uint32_t*>(&words[5])), "r"(*reinterpret_cast<const uint32_t*>(&words[6])), "r"(*reinterpret_cast<const uint32_t*>(&words[7]))
                    : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_addr + par * 18432 + warp_1 * 2304 + lane_0 * 16), "r"(*reinterpret_cast<uint32_t*>(&s[0])), "r"(*reinterpret_cast<uint32_t*>(&s[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s[(0) + 3])));
                uint32_t _e2m1x2_to_f16x2_40;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_40) : "r"((uint32_t)(raw[2])));
                uint32_t _f16x2_mul_40;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_40) : "r"(_e2m1x2_to_f16x2_40), "r"(sc[2]));
                words[8] = _f16x2_mul_40;
                uint32_t _e2m1x2_to_f16x2_41;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_41) : "r"((uint32_t)(raw[2] >> 8)));
                uint32_t _f16x2_mul_41;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_41) : "r"(_e2m1x2_to_f16x2_41), "r"(sc[2]));
                words[9] = _f16x2_mul_41;
                uint32_t _e2m1x2_to_f16x2_42;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_42) : "r"((uint32_t)(raw[2] >> 16)));
                uint32_t _f16x2_mul_42;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_42) : "r"(_e2m1x2_to_f16x2_42), "r"(sc[2]));
                words[10] = _f16x2_mul_42;
                uint32_t _e2m1x2_to_f16x2_43;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_43) : "r"((uint32_t)(raw[2] >> 24)));
                uint32_t _f16x2_mul_43;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_43) : "r"(_e2m1x2_to_f16x2_43), "r"(sc[2]));
                words[11] = _f16x2_mul_43;
                uint32_t _e2m1x2_to_f16x2_44;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_44) : "r"((uint32_t)(raw[3])));
                uint32_t _f16x2_mul_44;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_44) : "r"(_e2m1x2_to_f16x2_44), "r"(sc[3]));
                words[12] = _f16x2_mul_44;
                uint32_t _e2m1x2_to_f16x2_45;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_45) : "r"((uint32_t)(raw[3] >> 8)));
                uint32_t _f16x2_mul_45;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_45) : "r"(_e2m1x2_to_f16x2_45), "r"(sc[3]));
                words[13] = _f16x2_mul_45;
                uint32_t _e2m1x2_to_f16x2_46;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_46) : "r"((uint32_t)(raw[3] >> 16)));
                uint32_t _f16x2_mul_46;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_46) : "r"(_e2m1x2_to_f16x2_46), "r"(sc[3]));
                words[14] = _f16x2_mul_46;
                uint32_t _e2m1x2_to_f16x2_47;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_47) : "r"((uint32_t)(raw[3] >> 24)));
                uint32_t _f16x2_mul_47;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_47) : "r"(_e2m1x2_to_f16x2_47), "r"(sc[3]));
                words[15] = _f16x2_mul_47;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + vrow + (64 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(s[4]), "=f"(s[(4) + 1]), "=f"(s[(4) + 2]), "=f"(s[(4) + 3])
                    : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(words[8]), "r"(words[(8) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
                    : "r"(qa[4]), "r"(qa[(4) + 1]), "r"(qa[(4) + 2]), "r"(qa[(4) + 3]), "r"(words[10]), "r"(words[(10) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
                    : "r"(qa[8]), "r"(qa[(8) + 1]), "r"(qa[(8) + 2]), "r"(qa[(8) + 3]), "r"(words[12]), "r"(words[(12) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
                    : "r"(qa[12]), "r"(qa[(12) + 1]), "r"(qa[(12) + 2]), "r"(qa[(12) + 3]), "r"(words[14]), "r"(words[(14) + 1]));
                if (warp_1 < 4) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(s[4]), "+f"(s[(4) + 1]), "+f"(s[(4) + 2]), "+f"(s[(4) + 3])
                        : "r"(qr[0]), "r"(qr[1]), "r"(qr[2]), "r"(qr[3]), "r"(rwords[2]), "r"(rwords[(2) + 1]));
                }
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb1[0]), "=r"(vb1[1]), "=r"(vb1[2]), "=r"(vb1[3])
                    : "r"(vt_addr + ppar * 4096 + vrow + (96 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb1[0]), "r"(vb1[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb1[2]), "r"(vb1[(2) + 1]));
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 1024 + vs_row + (0 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&words[8])), "r"(*reinterpret_cast<const uint32_t*>(&words[9])), "r"(*reinterpret_cast<const uint32_t*>(&words[10])), "r"(*reinterpret_cast<const uint32_t*>(&words[11]))
                    : "memory");
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 1024 + vs_row + (64 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&words[12])), "r"(*reinterpret_cast<const uint32_t*>(&words[13])), "r"(*reinterpret_cast<const uint32_t*>(&words[14])), "r"(*reinterpret_cast<const uint32_t*>(&words[15]))
                    : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_addr + par * 18432 + warp_1 * 2304 + 576 + lane_0 * 16), "r"(*reinterpret_cast<uint32_t*>(&s[4])), "r"(*reinterpret_cast<uint32_t*>(&s[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s[(4) + 3])));
                uint32_t _e2m1x2_to_f16x2_48;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_48) : "r"((uint32_t)(raw[4])));
                uint32_t _f16x2_mul_48;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_48) : "r"(_e2m1x2_to_f16x2_48), "r"(sc[4]));
                words[16] = _f16x2_mul_48;
                uint32_t _e2m1x2_to_f16x2_49;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_49) : "r"((uint32_t)(raw[4] >> 8)));
                uint32_t _f16x2_mul_49;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_49) : "r"(_e2m1x2_to_f16x2_49), "r"(sc[4]));
                words[17] = _f16x2_mul_49;
                uint32_t _e2m1x2_to_f16x2_50;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_50) : "r"((uint32_t)(raw[4] >> 16)));
                uint32_t _f16x2_mul_50;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_50) : "r"(_e2m1x2_to_f16x2_50), "r"(sc[4]));
                words[18] = _f16x2_mul_50;
                uint32_t _e2m1x2_to_f16x2_51;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_51) : "r"((uint32_t)(raw[4] >> 24)));
                uint32_t _f16x2_mul_51;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_51) : "r"(_e2m1x2_to_f16x2_51), "r"(sc[4]));
                words[19] = _f16x2_mul_51;
                uint32_t _e2m1x2_to_f16x2_52;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_52) : "r"((uint32_t)(raw[5])));
                uint32_t _f16x2_mul_52;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_52) : "r"(_e2m1x2_to_f16x2_52), "r"(sc[5]));
                words[20] = _f16x2_mul_52;
                uint32_t _e2m1x2_to_f16x2_53;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_53) : "r"((uint32_t)(raw[5] >> 8)));
                uint32_t _f16x2_mul_53;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_53) : "r"(_e2m1x2_to_f16x2_53), "r"(sc[5]));
                words[21] = _f16x2_mul_53;
                uint32_t _e2m1x2_to_f16x2_54;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_54) : "r"((uint32_t)(raw[5] >> 16)));
                uint32_t _f16x2_mul_54;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_54) : "r"(_e2m1x2_to_f16x2_54), "r"(sc[5]));
                words[22] = _f16x2_mul_54;
                uint32_t _e2m1x2_to_f16x2_55;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_55) : "r"((uint32_t)(raw[5] >> 24)));
                uint32_t _f16x2_mul_55;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_55) : "r"(_e2m1x2_to_f16x2_55), "r"(sc[5]));
                words[23] = _f16x2_mul_55;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (0 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(s[8]), "=f"(s[(8) + 1]), "=f"(s[(8) + 2]), "=f"(s[(8) + 3])
                    : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(words[16]), "r"(words[(16) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[8]), "+f"(s[(8) + 1]), "+f"(s[(8) + 2]), "+f"(s[(8) + 3])
                    : "r"(qa[4]), "r"(qa[(4) + 1]), "r"(qa[(4) + 2]), "r"(qa[(4) + 3]), "r"(words[18]), "r"(words[(18) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[8]), "+f"(s[(8) + 1]), "+f"(s[(8) + 2]), "+f"(s[(8) + 3])
                    : "r"(qa[8]), "r"(qa[(8) + 1]), "r"(qa[(8) + 2]), "r"(qa[(8) + 3]), "r"(words[20]), "r"(words[(20) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[8]), "+f"(s[(8) + 1]), "+f"(s[(8) + 2]), "+f"(s[(8) + 3])
                    : "r"(qa[12]), "r"(qa[(12) + 1]), "r"(qa[(12) + 2]), "r"(qa[(12) + 3]), "r"(words[22]), "r"(words[(22) + 1]));
                if (warp_1 < 4) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(s[8]), "+f"(s[(8) + 1]), "+f"(s[(8) + 2]), "+f"(s[(8) + 3])
                        : "r"(qr[0]), "r"(qr[1]), "r"(qr[2]), "r"(qr[3]), "r"(rwords[4]), "r"(rwords[(4) + 1]));
                }
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb1[0]), "=r"(vb1[1]), "=r"(vb1[2]), "=r"(vb1[3])
                    : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (32 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb1[0]), "r"(vb1[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb1[2]), "r"(vb1[(2) + 1]));
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 2048 + vs_row + (0 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&words[16])), "r"(*reinterpret_cast<const uint32_t*>(&words[17])), "r"(*reinterpret_cast<const uint32_t*>(&words[18])), "r"(*reinterpret_cast<const uint32_t*>(&words[19]))
                    : "memory");
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 2048 + vs_row + (64 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&words[20])), "r"(*reinterpret_cast<const uint32_t*>(&words[21])), "r"(*reinterpret_cast<const uint32_t*>(&words[22])), "r"(*reinterpret_cast<const uint32_t*>(&words[23]))
                    : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_addr + par * 18432 + warp_1 * 2304 + 1152 + lane_0 * 16), "r"(*reinterpret_cast<uint32_t*>(&s[8])), "r"(*reinterpret_cast<uint32_t*>(&s[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s[(8) + 3])));
                uint32_t _e2m1x2_to_f16x2_56;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_56) : "r"((uint32_t)(raw[6])));
                uint32_t _f16x2_mul_56;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_56) : "r"(_e2m1x2_to_f16x2_56), "r"(sc[6]));
                words[24] = _f16x2_mul_56;
                uint32_t _e2m1x2_to_f16x2_57;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_57) : "r"((uint32_t)(raw[6] >> 8)));
                uint32_t _f16x2_mul_57;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_57) : "r"(_e2m1x2_to_f16x2_57), "r"(sc[6]));
                words[25] = _f16x2_mul_57;
                uint32_t _e2m1x2_to_f16x2_58;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_58) : "r"((uint32_t)(raw[6] >> 16)));
                uint32_t _f16x2_mul_58;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_58) : "r"(_e2m1x2_to_f16x2_58), "r"(sc[6]));
                words[26] = _f16x2_mul_58;
                uint32_t _e2m1x2_to_f16x2_59;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_59) : "r"((uint32_t)(raw[6] >> 24)));
                uint32_t _f16x2_mul_59;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_59) : "r"(_e2m1x2_to_f16x2_59), "r"(sc[6]));
                words[27] = _f16x2_mul_59;
                uint32_t _e2m1x2_to_f16x2_60;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_60) : "r"((uint32_t)(raw[7])));
                uint32_t _f16x2_mul_60;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_60) : "r"(_e2m1x2_to_f16x2_60), "r"(sc[7]));
                words[28] = _f16x2_mul_60;
                uint32_t _e2m1x2_to_f16x2_61;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_61) : "r"((uint32_t)(raw[7] >> 8)));
                uint32_t _f16x2_mul_61;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_61) : "r"(_e2m1x2_to_f16x2_61), "r"(sc[7]));
                words[29] = _f16x2_mul_61;
                uint32_t _e2m1x2_to_f16x2_62;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_62) : "r"((uint32_t)(raw[7] >> 16)));
                uint32_t _f16x2_mul_62;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_62) : "r"(_e2m1x2_to_f16x2_62), "r"(sc[7]));
                words[30] = _f16x2_mul_62;
                uint32_t _e2m1x2_to_f16x2_63;
                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1x2_to_f16x2_63) : "r"((uint32_t)(raw[7] >> 24)));
                uint32_t _f16x2_mul_63;
                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_63) : "r"(_e2m1x2_to_f16x2_63), "r"(sc[7]));
                words[31] = _f16x2_mul_63;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (64 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(s[12]), "=f"(s[(12) + 1]), "=f"(s[(12) + 2]), "=f"(s[(12) + 3])
                    : "r"(qa[0]), "r"(qa[1]), "r"(qa[2]), "r"(qa[3]), "r"(words[24]), "r"(words[(24) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[12]), "+f"(s[(12) + 1]), "+f"(s[(12) + 2]), "+f"(s[(12) + 3])
                    : "r"(qa[4]), "r"(qa[(4) + 1]), "r"(qa[(4) + 2]), "r"(qa[(4) + 3]), "r"(words[26]), "r"(words[(26) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[12]), "+f"(s[(12) + 1]), "+f"(s[(12) + 2]), "+f"(s[(12) + 3])
                    : "r"(qa[8]), "r"(qa[(8) + 1]), "r"(qa[(8) + 2]), "r"(qa[(8) + 3]), "r"(words[28]), "r"(words[(28) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(s[12]), "+f"(s[(12) + 1]), "+f"(s[(12) + 2]), "+f"(s[(12) + 3])
                    : "r"(qa[12]), "r"(qa[(12) + 1]), "r"(qa[(12) + 2]), "r"(qa[(12) + 3]), "r"(words[30]), "r"(words[(30) + 1]));
                if (warp_1 < 4) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(s[12]), "+f"(s[(12) + 1]), "+f"(s[(12) + 2]), "+f"(s[(12) + 3])
                        : "r"(qr[0]), "r"(qr[1]), "r"(qr[2]), "r"(qr[3]), "r"(rwords[6]), "r"(rwords[(6) + 1]));
                }
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb1[0]), "=r"(vb1[1]), "=r"(vb1[2]), "=r"(vb1[3])
                    : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (96 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb1[0]), "r"(vb1[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb1[2]), "r"(vb1[(2) + 1]));
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 3072 + vs_row + (0 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&words[24])), "r"(*reinterpret_cast<const uint32_t*>(&words[25])), "r"(*reinterpret_cast<const uint32_t*>(&words[26])), "r"(*reinterpret_cast<const uint32_t*>(&words[27]))
                    : "memory");
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(vt_addr + vpar * 4096 + 3072 + vs_row + (64 ^ vsx));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&words[28])), "r"(*reinterpret_cast<const uint32_t*>(&words[29])), "r"(*reinterpret_cast<const uint32_t*>(&words[30])), "r"(*reinterpret_cast<const uint32_t*>(&words[31]))
                    : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_addr + par * 18432 + warp_1 * 2304 + 1728 + lane_0 * 16), "r"(*reinterpret_cast<uint32_t*>(&s[12])), "r"(*reinterpret_cast<uint32_t*>(&s[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s[(12) + 3])));
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(kv_empty_addr + (m_stage) * 8);
                    mbarrier_arrive(s_ready_addr + (xs) * 8);
                }
                m_stage += 1;
                if (m_stage == 3) { m_stage = 0; m_phase ^= 1; }
            }
            if (num_stages > 1) {
                ppar = (num_stages - 2) % 3;
                pxs = (unsigned int)(num_stages - 2 & 1);
                pxp = (unsigned int)(num_stages - 2 >> 1 & 1);
                mbarrier_wait(p_ready_addr + (pxs) * 8, pxp);
                float _alpha_f32_reg_2[1];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(alpha_f32);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _alpha_f32_reg_2[_lr] = _smem_ptr[(ppar * 16 + g) + _lr];
                }
                float _alpha_f32_reg_3[1];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(alpha_f32);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _alpha_f32_reg_3[_lr] = _smem_ptr[(ppar * 16 + g + 8) + _lr];
                }
                o[0] = o[0] * _alpha_f32_reg_2[0];
                o[1] = o[1] * _alpha_f32_reg_2[0];
                o[2] = o[2] * _alpha_f32_reg_3[0];
                o[3] = o[3] * _alpha_f32_reg_3[0];
                o[4] = o[4] * _alpha_f32_reg_2[0];
                o[5] = o[5] * _alpha_f32_reg_2[0];
                o[6] = o[6] * _alpha_f32_reg_3[0];
                o[7] = o[7] * _alpha_f32_reg_3[0];
                o[8] = o[8] * _alpha_f32_reg_2[0];
                o[9] = o[9] * _alpha_f32_reg_2[0];
                o[10] = o[10] * _alpha_f32_reg_3[0];
                o[11] = o[11] * _alpha_f32_reg_3[0];
                o[12] = o[12] * _alpha_f32_reg_2[0];
                o[13] = o[13] * _alpha_f32_reg_2[0];
                o[14] = o[14] * _alpha_f32_reg_3[0];
                o[15] = o[15] * _alpha_f32_reg_3[0];
                o[16] = o[16] * _alpha_f32_reg_2[0];
                o[17] = o[17] * _alpha_f32_reg_2[0];
                o[18] = o[18] * _alpha_f32_reg_3[0];
                o[19] = o[19] * _alpha_f32_reg_3[0];
                o[20] = o[20] * _alpha_f32_reg_2[0];
                o[21] = o[21] * _alpha_f32_reg_2[0];
                o[22] = o[22] * _alpha_f32_reg_3[0];
                o[23] = o[23] * _alpha_f32_reg_3[0];
                o[24] = o[24] * _alpha_f32_reg_2[0];
                o[25] = o[25] * _alpha_f32_reg_2[0];
                o[26] = o[26] * _alpha_f32_reg_3[0];
                o[27] = o[27] * _alpha_f32_reg_3[0];
                o[28] = o[28] * _alpha_f32_reg_2[0];
                o[29] = o[29] * _alpha_f32_reg_2[0];
                o[30] = o[30] * _alpha_f32_reg_3[0];
                o[31] = o[31] * _alpha_f32_reg_3[0];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(pa[0]), "=r"(pa[1]), "=r"(pa[2]), "=r"(pa[3])
                    : "r"(p_addr + ppar * 1280 + (lane_0 & 15) * 80 + ((lane_0 >> 4 & 1) * 16 ^ (lane_0 >> 3 & 1) << 5))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(pa[4]), "=r"(pa[5]), "=r"(pa[6]), "=r"(pa[7])
                    : "r"(p_addr + ppar * 1280 + (lane_0 & 15) * 80 + (32 + (lane_0 >> 4 & 1) * 16 ^ (lane_0 >> 3 & 1) << 5))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + vrow + (0 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + vrow + (32 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + vrow + (64 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + vrow + (96 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
                    : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (0 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (32 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (64 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                    : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (96 ^ vx))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
                    : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            }
            ppar = (num_stages - 1) % 3;
            pxs = (unsigned int)(num_stages - 1 & 1);
            pxp = (unsigned int)(num_stages - 1 >> 1 & 1);
            mbarrier_wait(p_ready_addr + (pxs) * 8, pxp);
            float _alpha_f32_reg_4[1];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(alpha_f32);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _alpha_f32_reg_4[_lr] = _smem_ptr[(ppar * 16 + g) + _lr];
            }
            float _alpha_f32_reg_5[1];
            {
                const float* _smem_ptr = reinterpret_cast<const float*>(alpha_f32);
                #pragma unroll
                for (int _lr = 0; _lr < 1; _lr++)
                    _alpha_f32_reg_5[_lr] = _smem_ptr[(ppar * 16 + g + 8) + _lr];
            }
            o[0] = o[0] * _alpha_f32_reg_4[0];
            o[1] = o[1] * _alpha_f32_reg_4[0];
            o[2] = o[2] * _alpha_f32_reg_5[0];
            o[3] = o[3] * _alpha_f32_reg_5[0];
            o[4] = o[4] * _alpha_f32_reg_4[0];
            o[5] = o[5] * _alpha_f32_reg_4[0];
            o[6] = o[6] * _alpha_f32_reg_5[0];
            o[7] = o[7] * _alpha_f32_reg_5[0];
            o[8] = o[8] * _alpha_f32_reg_4[0];
            o[9] = o[9] * _alpha_f32_reg_4[0];
            o[10] = o[10] * _alpha_f32_reg_5[0];
            o[11] = o[11] * _alpha_f32_reg_5[0];
            o[12] = o[12] * _alpha_f32_reg_4[0];
            o[13] = o[13] * _alpha_f32_reg_4[0];
            o[14] = o[14] * _alpha_f32_reg_5[0];
            o[15] = o[15] * _alpha_f32_reg_5[0];
            o[16] = o[16] * _alpha_f32_reg_4[0];
            o[17] = o[17] * _alpha_f32_reg_4[0];
            o[18] = o[18] * _alpha_f32_reg_5[0];
            o[19] = o[19] * _alpha_f32_reg_5[0];
            o[20] = o[20] * _alpha_f32_reg_4[0];
            o[21] = o[21] * _alpha_f32_reg_4[0];
            o[22] = o[22] * _alpha_f32_reg_5[0];
            o[23] = o[23] * _alpha_f32_reg_5[0];
            o[24] = o[24] * _alpha_f32_reg_4[0];
            o[25] = o[25] * _alpha_f32_reg_4[0];
            o[26] = o[26] * _alpha_f32_reg_5[0];
            o[27] = o[27] * _alpha_f32_reg_5[0];
            o[28] = o[28] * _alpha_f32_reg_4[0];
            o[29] = o[29] * _alpha_f32_reg_4[0];
            o[30] = o[30] * _alpha_f32_reg_5[0];
            o[31] = o[31] * _alpha_f32_reg_5[0];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(pa[0]), "=r"(pa[1]), "=r"(pa[2]), "=r"(pa[3])
                : "r"(p_addr + ppar * 1280 + (lane_0 & 15) * 80 + ((lane_0 >> 4 & 1) * 16 ^ (lane_0 >> 3 & 1) << 5))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(pa[4]), "=r"(pa[5]), "=r"(pa[6]), "=r"(pa[7])
                : "r"(p_addr + ppar * 1280 + (lane_0 & 15) * 80 + (32 + (lane_0 >> 4 & 1) * 16 ^ (lane_0 >> 3 & 1) << 5))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                : "r"(vt_addr + ppar * 4096 + vrow + (0 ^ vx))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
                : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
                : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                : "r"(vt_addr + ppar * 4096 + vrow + (32 ^ vx))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
                : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
                : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                : "r"(vt_addr + ppar * 4096 + vrow + (64 ^ vx))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
                : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
                : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                : "r"(vt_addr + ppar * 4096 + vrow + (96 ^ vx))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
                : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[0]), "r"(vb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
                : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (0 ^ vx))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3])
                : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[4]), "+f"(o[(4) + 1]), "+f"(o[(4) + 2]), "+f"(o[(4) + 3])
                : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (32 ^ vx))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[8]), "+f"(o[(8) + 1]), "+f"(o[(8) + 2]), "+f"(o[(8) + 3])
                : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[12]), "+f"(o[(12) + 1]), "+f"(o[(12) + 2]), "+f"(o[(12) + 3])
                : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (64 ^ vx))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[16]), "+f"(o[(16) + 1]), "+f"(o[(16) + 2]), "+f"(o[(16) + 3])
                : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[20]), "+f"(o[(20) + 1]), "+f"(o[(20) + 2]), "+f"(o[(20) + 3])
                : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                : "r"(vt_addr + ppar * 4096 + 2048 + vrow + (96 ^ vx))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[24]), "+f"(o[(24) + 1]), "+f"(o[(24) + 2]), "+f"(o[(24) + 3])
                : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[0]), "r"(vb[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(o[28]), "+f"(o[(28) + 1]), "+f"(o[(28) + 2]), "+f"(o[(28) + 3])
                : "r"(pa[4]), "r"(pa[(4) + 1]), "r"(pa[(4) + 2]), "r"(pa[(4) + 3]), "r"(vb[2]), "r"(vb[(2) + 1]));
            asm volatile("barrier.sync 3, 384;" ::: "memory");
            int rank = cta_rank;
            unsigned int exit_flag = 1;
            out8[0] = o[0];
            out8[1] = o[1];
            out8[2] = o[4];
            out8[3] = o[5];
            out8[4] = o[8];
            out8[5] = o[9];
            out8[6] = o[12];
            out8[7] = o[13];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(((rank == (8 * warp_1 + c) % 8) ? (int)zone_u8_addr : ring_addr) + (8 * warp_1 + c) % 8 * 4224 + ((8 * warp_1 + c) / 8 * 16 + (g ^ c)) * 32 + ((g >> 2 & 1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&out8[0])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(((rank == (8 * warp_1 + c) % 8) ? (int)zone_u8_addr : ring_addr) + (8 * warp_1 + c) % 8 * 4224 + ((8 * warp_1 + c) / 8 * 16 + (g ^ c)) * 32 + ((g >> 2 & 1) << 4 ^ 16)), "r"(*reinterpret_cast<uint32_t*>(&out8[4])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 3])));
            out8[0] = o[2];
            out8[1] = o[3];
            out8[2] = o[6];
            out8[3] = o[7];
            out8[4] = o[10];
            out8[5] = o[11];
            out8[6] = o[14];
            out8[7] = o[15];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(((rank == (8 * warp_1 + c) % 8) ? (int)zone_u8_addr : ring_addr) + (8 * warp_1 + c) % 8 * 4224 + ((8 * warp_1 + c) / 8 * 16 + (g + 8 ^ c)) * 32 + ((g >> 2 & 1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&out8[0])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(((rank == (8 * warp_1 + c) % 8) ? (int)zone_u8_addr : ring_addr) + (8 * warp_1 + c) % 8 * 4224 + ((8 * warp_1 + c) / 8 * 16 + (g + 8 ^ c)) * 32 + ((g >> 2 & 1) << 4 ^ 16)), "r"(*reinterpret_cast<uint32_t*>(&out8[4])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 3])));
            out8[0] = o[16];
            out8[1] = o[17];
            out8[2] = o[20];
            out8[3] = o[21];
            out8[4] = o[24];
            out8[5] = o[25];
            out8[6] = o[28];
            out8[7] = o[29];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(((rank == (8 * warp_1 + 4 + c) % 8) ? (int)zone_u8_addr : ring_addr) + (8 * warp_1 + 4 + c) % 8 * 4224 + ((8 * warp_1 + 4 + c) / 8 * 16 + (g ^ c)) * 32 + ((g >> 2 & 1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&out8[0])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(((rank == (8 * warp_1 + 4 + c) % 8) ? (int)zone_u8_addr : ring_addr) + (8 * warp_1 + 4 + c) % 8 * 4224 + ((8 * warp_1 + 4 + c) / 8 * 16 + (g ^ c)) * 32 + ((g >> 2 & 1) << 4 ^ 16)), "r"(*reinterpret_cast<uint32_t*>(&out8[4])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 3])));
            out8[0] = o[18];
            out8[1] = o[19];
            out8[2] = o[22];
            out8[3] = o[23];
            out8[4] = o[26];
            out8[5] = o[27];
            out8[6] = o[30];
            out8[7] = o[31];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(((rank == (8 * warp_1 + 4 + c) % 8) ? (int)zone_u8_addr : ring_addr) + (8 * warp_1 + 4 + c) % 8 * 4224 + ((8 * warp_1 + 4 + c) / 8 * 16 + (g + 8 ^ c)) * 32 + ((g >> 2 & 1) << 4)), "r"(*reinterpret_cast<uint32_t*>(&out8[0])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&out8[(0) + 3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(((rank == (8 * warp_1 + 4 + c) % 8) ? (int)zone_u8_addr : ring_addr) + (8 * warp_1 + 4 + c) % 8 * 4224 + ((8 * warp_1 + 4 + c) / 8 * 16 + (g + 8 ^ c)) * 32 + ((g >> 2 & 1) << 4 ^ 16)), "r"(*reinterpret_cast<uint32_t*>(&out8[4])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&out8[(4) + 3])));
            if (warp_1 == 0) {
                unsigned int _ml_u32_reg_0[1];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(ml_u32);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _ml_u32_reg_0[_lr] = _smem_ptr[(lane_0) + _lr];
                }
                if (rank == 0) {
                    zone_u32[1024 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank != 0) {
                    ring_u32[1024 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank == 1) {
                    zone_u32[2080 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank != 1) {
                    ring_u32[2080 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank == 2) {
                    zone_u32[3136 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank != 2) {
                    ring_u32[3136 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank == 3) {
                    zone_u32[4192 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank != 3) {
                    ring_u32[4192 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank == 4) {
                    zone_u32[5248 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank != 4) {
                    ring_u32[5248 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank == 5) {
                    zone_u32[6304 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank != 5) {
                    ring_u32[6304 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank == 6) {
                    zone_u32[7360 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank != 6) {
                    ring_u32[7360 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank == 7) {
                    zone_u32[8416 + lane_0] = _ml_u32_reg_0[0];
                }
                if (rank != 7) {
                    ring_u32[8416 + lane_0] = _ml_u32_reg_0[0];
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (thread == 0) {
                mbarrier_arrive_expect_tx(zone_ready_addr, 29568);
            }
            if (warp_1 < 8) {
                if (warp_1 != rank) {
                    if (elect_sync()) {
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(zone_u8_addr + (unsigned int)(rank * 4224)), "r"(warp_1));
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(zone_ready_addr), "r"(warp_1));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_0), "r"(ring_addr + warp_1 * 4224), "r"((uint32_t)(4224)), "r"(_mapa_1)
                            : "memory");
                    }
                }
            }
            mbarrier_wait_cluster(zone_ready_addr, 0);
            if (thread == 0) {
                mbarrier_arrive_expect_tx(exit_ready_addr, 28);
            }
            if (warp_1 == 0) {
                if (lane_0 < 8) {
                    if (lane_0 != rank) {
                        uint32_t _mapa_2;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_2) : "r"(ml_u32_addr + (unsigned int)(rank * 4)), "r"(lane_0));
                        uint32_t _mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_3) : "r"(exit_ready_addr), "r"(lane_0));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_mapa_2), "r"(exit_flag), "r"(_mapa_3) : "memory");
                    }
                }
            }
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1]))
                : "r"(zone_u8_addr + 4096 + (unsigned int)((thread & 15) * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&sacc[2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(2) + 1]))
                : "r"(zone_u8_addr + 4224 + 4096 + (unsigned int)((thread & 15) * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1]))
                : "r"(zone_u8_addr + 8448 + 4096 + (unsigned int)((thread & 15) * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&sacc[6])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(6) + 1]))
                : "r"(zone_u8_addr + 12672 + 4096 + (unsigned int)((thread & 15) * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&sacc[8])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(8) + 1]))
                : "r"(zone_u8_addr + 16896 + 4096 + (unsigned int)((thread & 15) * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&sacc[10])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(10) + 1]))
                : "r"(zone_u8_addr + 21120 + 4096 + (unsigned int)((thread & 15) * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&sacc[12])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(12) + 1]))
                : "r"(zone_u8_addr + 25344 + 4096 + (unsigned int)((thread & 15) * 8)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&sacc[14])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(14) + 1]))
                : "r"(zone_u8_addr + 29568 + 4096 + (unsigned int)((thread & 15) * 8)));
            wts[8] = __uint_as_float(sacc[0]);
            float _fmax_0 = fmaxf(wts[8], __uint_as_float(sacc[2]));
            wts[8] = _fmax_0;
            float _fmax_1 = fmaxf(wts[8], __uint_as_float(sacc[4]));
            wts[8] = _fmax_1;
            float _fmax_2 = fmaxf(wts[8], __uint_as_float(sacc[6]));
            wts[8] = _fmax_2;
            float _fmax_3 = fmaxf(wts[8], __uint_as_float(sacc[8]));
            wts[8] = _fmax_3;
            float _fmax_4 = fmaxf(wts[8], __uint_as_float(sacc[10]));
            wts[8] = _fmax_4;
            float _fmax_5 = fmaxf(wts[8], __uint_as_float(sacc[12]));
            wts[8] = _fmax_5;
            float _fmax_6 = fmaxf(wts[8], __uint_as_float(sacc[14]));
            wts[8] = _fmax_6;
            wts[9] = 0.0f;
            float _exp2_0 = approx_exp2(__uint_as_float(sacc[0]) - wts[8]);
            wts[0] = ((__uint_as_float(sacc[0]) > -1e+29f) ? _exp2_0 : 0.0f);
            float _fma_0 = __fmaf_rn(wts[0], __uint_as_float(sacc[1]), wts[9]);
            wts[9] = _fma_0;
            float _exp2_1 = approx_exp2(__uint_as_float(sacc[2]) - wts[8]);
            wts[1] = ((__uint_as_float(sacc[2]) > -1e+29f) ? _exp2_1 : 0.0f);
            float _fma_1 = __fmaf_rn(wts[1], __uint_as_float(sacc[3]), wts[9]);
            wts[9] = _fma_1;
            float _exp2_2 = approx_exp2(__uint_as_float(sacc[4]) - wts[8]);
            wts[2] = ((__uint_as_float(sacc[4]) > -1e+29f) ? _exp2_2 : 0.0f);
            float _fma_2 = __fmaf_rn(wts[2], __uint_as_float(sacc[5]), wts[9]);
            wts[9] = _fma_2;
            float _exp2_3 = approx_exp2(__uint_as_float(sacc[6]) - wts[8]);
            wts[3] = ((__uint_as_float(sacc[6]) > -1e+29f) ? _exp2_3 : 0.0f);
            float _fma_3 = __fmaf_rn(wts[3], __uint_as_float(sacc[7]), wts[9]);
            wts[9] = _fma_3;
            float _exp2_4 = approx_exp2(__uint_as_float(sacc[8]) - wts[8]);
            wts[4] = ((__uint_as_float(sacc[8]) > -1e+29f) ? _exp2_4 : 0.0f);
            float _fma_4 = __fmaf_rn(wts[4], __uint_as_float(sacc[9]), wts[9]);
            wts[9] = _fma_4;
            float _exp2_5 = approx_exp2(__uint_as_float(sacc[10]) - wts[8]);
            wts[5] = ((__uint_as_float(sacc[10]) > -1e+29f) ? _exp2_5 : 0.0f);
            float _fma_5 = __fmaf_rn(wts[5], __uint_as_float(sacc[11]), wts[9]);
            wts[9] = _fma_5;
            float _exp2_6 = approx_exp2(__uint_as_float(sacc[12]) - wts[8]);
            wts[6] = ((__uint_as_float(sacc[12]) > -1e+29f) ? _exp2_6 : 0.0f);
            float _fma_6 = __fmaf_rn(wts[6], __uint_as_float(sacc[13]), wts[9]);
            wts[9] = _fma_6;
            float _exp2_7 = approx_exp2(__uint_as_float(sacc[14]) - wts[8]);
            wts[7] = ((__uint_as_float(sacc[14]) > -1e+29f) ? _exp2_7 : 0.0f);
            float _fma_7 = __fmaf_rn(wts[7], __uint_as_float(sacc[15]), wts[9]);
            wts[9] = _fma_7;
            wts[10] = ((wts[9] > 0.0f) ? out_scale / wts[9] : 0.0f);
            if (thread >> 4 < 8) {
                if ((thread >> 4) * 8 + rank < 64) {
                    out8[0] = 0.0f;
                    out8[1] = 0.0f;
                    out8[2] = 0.0f;
                    out8[3] = 0.0f;
                    out8[4] = 0.0f;
                    out8[5] = 0.0f;
                    out8[6] = 0.0f;
                    out8[7] = 0.0f;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 3]))
                        : "r"(zone_u8_addr + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 3]))
                        : "r"(zone_u8_addr + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4 ^ 16)));
                    float _fma_8 = __fmaf_rn(__uint_as_float(sacc[0]), wts[0], out8[0]);
                    out8[0] = _fma_8;
                    float _fma_9 = __fmaf_rn(__uint_as_float(sacc[1]), wts[0], out8[1]);
                    out8[1] = _fma_9;
                    float _fma_10 = __fmaf_rn(__uint_as_float(sacc[2]), wts[0], out8[2]);
                    out8[2] = _fma_10;
                    float _fma_11 = __fmaf_rn(__uint_as_float(sacc[3]), wts[0], out8[3]);
                    out8[3] = _fma_11;
                    float _fma_12 = __fmaf_rn(__uint_as_float(sacc[4]), wts[0], out8[4]);
                    out8[4] = _fma_12;
                    float _fma_13 = __fmaf_rn(__uint_as_float(sacc[5]), wts[0], out8[5]);
                    out8[5] = _fma_13;
                    float _fma_14 = __fmaf_rn(__uint_as_float(sacc[6]), wts[0], out8[6]);
                    out8[6] = _fma_14;
                    float _fma_15 = __fmaf_rn(__uint_as_float(sacc[7]), wts[0], out8[7]);
                    out8[7] = _fma_15;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 3]))
                        : "r"(zone_u8_addr + 4224 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 3]))
                        : "r"(zone_u8_addr + 4224 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4 ^ 16)));
                    float _fma_16 = __fmaf_rn(__uint_as_float(sacc[0]), wts[1], out8[0]);
                    out8[0] = _fma_16;
                    float _fma_17 = __fmaf_rn(__uint_as_float(sacc[1]), wts[1], out8[1]);
                    out8[1] = _fma_17;
                    float _fma_18 = __fmaf_rn(__uint_as_float(sacc[2]), wts[1], out8[2]);
                    out8[2] = _fma_18;
                    float _fma_19 = __fmaf_rn(__uint_as_float(sacc[3]), wts[1], out8[3]);
                    out8[3] = _fma_19;
                    float _fma_20 = __fmaf_rn(__uint_as_float(sacc[4]), wts[1], out8[4]);
                    out8[4] = _fma_20;
                    float _fma_21 = __fmaf_rn(__uint_as_float(sacc[5]), wts[1], out8[5]);
                    out8[5] = _fma_21;
                    float _fma_22 = __fmaf_rn(__uint_as_float(sacc[6]), wts[1], out8[6]);
                    out8[6] = _fma_22;
                    float _fma_23 = __fmaf_rn(__uint_as_float(sacc[7]), wts[1], out8[7]);
                    out8[7] = _fma_23;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 3]))
                        : "r"(zone_u8_addr + 8448 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 3]))
                        : "r"(zone_u8_addr + 8448 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4 ^ 16)));
                    float _fma_24 = __fmaf_rn(__uint_as_float(sacc[0]), wts[2], out8[0]);
                    out8[0] = _fma_24;
                    float _fma_25 = __fmaf_rn(__uint_as_float(sacc[1]), wts[2], out8[1]);
                    out8[1] = _fma_25;
                    float _fma_26 = __fmaf_rn(__uint_as_float(sacc[2]), wts[2], out8[2]);
                    out8[2] = _fma_26;
                    float _fma_27 = __fmaf_rn(__uint_as_float(sacc[3]), wts[2], out8[3]);
                    out8[3] = _fma_27;
                    float _fma_28 = __fmaf_rn(__uint_as_float(sacc[4]), wts[2], out8[4]);
                    out8[4] = _fma_28;
                    float _fma_29 = __fmaf_rn(__uint_as_float(sacc[5]), wts[2], out8[5]);
                    out8[5] = _fma_29;
                    float _fma_30 = __fmaf_rn(__uint_as_float(sacc[6]), wts[2], out8[6]);
                    out8[6] = _fma_30;
                    float _fma_31 = __fmaf_rn(__uint_as_float(sacc[7]), wts[2], out8[7]);
                    out8[7] = _fma_31;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 3]))
                        : "r"(zone_u8_addr + 12672 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 3]))
                        : "r"(zone_u8_addr + 12672 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4 ^ 16)));
                    float _fma_32 = __fmaf_rn(__uint_as_float(sacc[0]), wts[3], out8[0]);
                    out8[0] = _fma_32;
                    float _fma_33 = __fmaf_rn(__uint_as_float(sacc[1]), wts[3], out8[1]);
                    out8[1] = _fma_33;
                    float _fma_34 = __fmaf_rn(__uint_as_float(sacc[2]), wts[3], out8[2]);
                    out8[2] = _fma_34;
                    float _fma_35 = __fmaf_rn(__uint_as_float(sacc[3]), wts[3], out8[3]);
                    out8[3] = _fma_35;
                    float _fma_36 = __fmaf_rn(__uint_as_float(sacc[4]), wts[3], out8[4]);
                    out8[4] = _fma_36;
                    float _fma_37 = __fmaf_rn(__uint_as_float(sacc[5]), wts[3], out8[5]);
                    out8[5] = _fma_37;
                    float _fma_38 = __fmaf_rn(__uint_as_float(sacc[6]), wts[3], out8[6]);
                    out8[6] = _fma_38;
                    float _fma_39 = __fmaf_rn(__uint_as_float(sacc[7]), wts[3], out8[7]);
                    out8[7] = _fma_39;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 3]))
                        : "r"(zone_u8_addr + 16896 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 3]))
                        : "r"(zone_u8_addr + 16896 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4 ^ 16)));
                    float _fma_40 = __fmaf_rn(__uint_as_float(sacc[0]), wts[4], out8[0]);
                    out8[0] = _fma_40;
                    float _fma_41 = __fmaf_rn(__uint_as_float(sacc[1]), wts[4], out8[1]);
                    out8[1] = _fma_41;
                    float _fma_42 = __fmaf_rn(__uint_as_float(sacc[2]), wts[4], out8[2]);
                    out8[2] = _fma_42;
                    float _fma_43 = __fmaf_rn(__uint_as_float(sacc[3]), wts[4], out8[3]);
                    out8[3] = _fma_43;
                    float _fma_44 = __fmaf_rn(__uint_as_float(sacc[4]), wts[4], out8[4]);
                    out8[4] = _fma_44;
                    float _fma_45 = __fmaf_rn(__uint_as_float(sacc[5]), wts[4], out8[5]);
                    out8[5] = _fma_45;
                    float _fma_46 = __fmaf_rn(__uint_as_float(sacc[6]), wts[4], out8[6]);
                    out8[6] = _fma_46;
                    float _fma_47 = __fmaf_rn(__uint_as_float(sacc[7]), wts[4], out8[7]);
                    out8[7] = _fma_47;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 3]))
                        : "r"(zone_u8_addr + 21120 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 3]))
                        : "r"(zone_u8_addr + 21120 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4 ^ 16)));
                    float _fma_48 = __fmaf_rn(__uint_as_float(sacc[0]), wts[5], out8[0]);
                    out8[0] = _fma_48;
                    float _fma_49 = __fmaf_rn(__uint_as_float(sacc[1]), wts[5], out8[1]);
                    out8[1] = _fma_49;
                    float _fma_50 = __fmaf_rn(__uint_as_float(sacc[2]), wts[5], out8[2]);
                    out8[2] = _fma_50;
                    float _fma_51 = __fmaf_rn(__uint_as_float(sacc[3]), wts[5], out8[3]);
                    out8[3] = _fma_51;
                    float _fma_52 = __fmaf_rn(__uint_as_float(sacc[4]), wts[5], out8[4]);
                    out8[4] = _fma_52;
                    float _fma_53 = __fmaf_rn(__uint_as_float(sacc[5]), wts[5], out8[5]);
                    out8[5] = _fma_53;
                    float _fma_54 = __fmaf_rn(__uint_as_float(sacc[6]), wts[5], out8[6]);
                    out8[6] = _fma_54;
                    float _fma_55 = __fmaf_rn(__uint_as_float(sacc[7]), wts[5], out8[7]);
                    out8[7] = _fma_55;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 3]))
                        : "r"(zone_u8_addr + 25344 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 3]))
                        : "r"(zone_u8_addr + 25344 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4 ^ 16)));
                    float _fma_56 = __fmaf_rn(__uint_as_float(sacc[0]), wts[6], out8[0]);
                    out8[0] = _fma_56;
                    float _fma_57 = __fmaf_rn(__uint_as_float(sacc[1]), wts[6], out8[1]);
                    out8[1] = _fma_57;
                    float _fma_58 = __fmaf_rn(__uint_as_float(sacc[2]), wts[6], out8[2]);
                    out8[2] = _fma_58;
                    float _fma_59 = __fmaf_rn(__uint_as_float(sacc[3]), wts[6], out8[3]);
                    out8[3] = _fma_59;
                    float _fma_60 = __fmaf_rn(__uint_as_float(sacc[4]), wts[6], out8[4]);
                    out8[4] = _fma_60;
                    float _fma_61 = __fmaf_rn(__uint_as_float(sacc[5]), wts[6], out8[5]);
                    out8[5] = _fma_61;
                    float _fma_62 = __fmaf_rn(__uint_as_float(sacc[6]), wts[6], out8[6]);
                    out8[6] = _fma_62;
                    float _fma_63 = __fmaf_rn(__uint_as_float(sacc[7]), wts[6], out8[7]);
                    out8[7] = _fma_63;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(0) + 3]))
                        : "r"(zone_u8_addr + 29568 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sacc[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc[(4) + 3]))
                        : "r"(zone_u8_addr + 29568 + (unsigned int)(((thread >> 4) * 16 + (thread & 15 ^ (thread >> 4) * 8 + rank & 3)) * 32) + (unsigned int)(((thread & 15) >> 2 & 1) << 4 ^ 16)));
                    float _fma_64 = __fmaf_rn(__uint_as_float(sacc[0]), wts[7], out8[0]);
                    out8[0] = _fma_64;
                    float _fma_65 = __fmaf_rn(__uint_as_float(sacc[1]), wts[7], out8[1]);
                    out8[1] = _fma_65;
                    float _fma_66 = __fmaf_rn(__uint_as_float(sacc[2]), wts[7], out8[2]);
                    out8[2] = _fma_66;
                    float _fma_67 = __fmaf_rn(__uint_as_float(sacc[3]), wts[7], out8[3]);
                    out8[3] = _fma_67;
                    float _fma_68 = __fmaf_rn(__uint_as_float(sacc[4]), wts[7], out8[4]);
                    out8[4] = _fma_68;
                    float _fma_69 = __fmaf_rn(__uint_as_float(sacc[5]), wts[7], out8[5]);
                    out8[5] = _fma_69;
                    float _fma_70 = __fmaf_rn(__uint_as_float(sacc[6]), wts[7], out8[6]);
                    out8[6] = _fma_70;
                    float _fma_71 = __fmaf_rn(__uint_as_float(sacc[7]), wts[7], out8[7]);
                    out8[7] = _fma_71;
                    {
                        const float2 _prescale2_17 = {wts[10], wts[10]};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&out8[0])[_ps], _prescale2_17);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            out8[0 + _ps] *= wts[10];
                        #endif
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(out8[0 + 0], out8[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(out8[0 + 2], out8[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(out8[0 + 4], out8[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(out8[0 + 6], out8[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((token * 16 + (thread & 15)) * 512 + 8 * ((thread >> 4) * 8 + rank))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
            mbarrier_wait_cluster(exit_ready_addr, 0);
        }
    }
    // ---- Role: sm ----
    if (warp >= 8 && warp <= 11) {
        { // sm_main
            int lane_s = lane;
            int sw = warp - 8;
            asm volatile("barrier.sync 2, 448;" ::: "memory");
            unsigned int sacc_s[16];
            float pvals[2];
            unsigned int pw[1];
            float pvals2[2];
            unsigned int pw2[1];
            float m_run = -1e+30f;
            float l_run = 0.0f;
            int shalf = lane_s >> 3 & 1;
            int sg = sw + 4 * (lane_s >> 4);
            int sh = sg + 8 * shalf;
            int sq = lane_s >> 2 & 1;
            int scol = lane_s & 3;
            int kqa = 8 * sq + 2 * scol;
            int rd_off = sq * 576 + (4 * sg + scol) * 16 + shalf * 8;
            int s_addr_s = s_u8_addr;
            int spar = 0;
            int spar3 = 0;
            unsigned int sxs = 0;
            unsigned int sxp = 0;
            int rd_base = 0;
            int v0 = 0;
            int v1 = 0;
            int v2 = 0;
            int v3 = 0;
            float sv0 = 0.0f;
            float sv1 = 0.0f;
            float sv2 = 0.0f;
            float sv3 = 0.0f;
            float mx = 0.0f;
            float m_new = 0.0f;
            float alpha = 0.0f;
            float p0 = 0.0f;
            float p1 = 0.0f;
            float p2 = 0.0f;
            float p3 = 0.0f;
            for (int st_2 = 0; st_2 < num_stages; st_2++) {
                spar = st_2 & 1;
                spar3 = st_2 % 3;
                sxs = (unsigned int)spar;
                sxp = (unsigned int)(st_2 >> 1 & 1);
                int _itab_reg_11[2];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _itab_reg_11[_lr] = _smem_ptr[(st_2 * 32 + kqa) + _lr];
                }
                v0 = _itab_reg_11[0];
                v1 = _itab_reg_11[1];
                int _itab_reg_12[2];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _itab_reg_12[_lr] = _smem_ptr[(st_2 * 32 + kqa + 16) + _lr];
                }
                v2 = _itab_reg_12[0];
                v3 = _itab_reg_12[1];
                mbarrier_wait(s_ready_addr + (sxs) * 8, sxp);
                rd_base = s_addr_s + spar * 18432 + rd_off;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(0) + 1]))
                    : "r"(rd_base));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(2) + 1]))
                    : "r"(rd_base + 2304));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(4) + 1]))
                    : "r"(rd_base + 4608));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(6) + 1]))
                    : "r"(rd_base + 6912));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(8) + 1]))
                    : "r"(rd_base + 9216));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(10) + 1]))
                    : "r"(rd_base + 11520));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(12) + 1]))
                    : "r"(rd_base + 13824));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(14) + 1]))
                    : "r"(rd_base + 16128));
                sv0 = __uint_as_float(sacc_s[0]) + __uint_as_float(sacc_s[2]) + __uint_as_float(sacc_s[4]) + __uint_as_float(sacc_s[6]) + __uint_as_float(sacc_s[8]) + __uint_as_float(sacc_s[10]) + __uint_as_float(sacc_s[12]) + __uint_as_float(sacc_s[14]);
                sv1 = __uint_as_float(sacc_s[1]) + __uint_as_float(sacc_s[3]) + __uint_as_float(sacc_s[5]) + __uint_as_float(sacc_s[7]) + __uint_as_float(sacc_s[9]) + __uint_as_float(sacc_s[11]) + __uint_as_float(sacc_s[13]) + __uint_as_float(sacc_s[15]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(0) + 1]))
                    : "r"(rd_base + 1152));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[2])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(2) + 1]))
                    : "r"(rd_base + 1152 + 2304));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(4) + 1]))
                    : "r"(rd_base + 1152 + 4608));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[6])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(6) + 1]))
                    : "r"(rd_base + 1152 + 6912));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(8) + 1]))
                    : "r"(rd_base + 1152 + 9216));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[10])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(10) + 1]))
                    : "r"(rd_base + 1152 + 11520));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(12) + 1]))
                    : "r"(rd_base + 1152 + 13824));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[14])), "=r"(*reinterpret_cast<uint32_t*>(&sacc_s[(14) + 1]))
                    : "r"(rd_base + 1152 + 16128));
                sv2 = __uint_as_float(sacc_s[0]) + __uint_as_float(sacc_s[2]) + __uint_as_float(sacc_s[4]) + __uint_as_float(sacc_s[6]) + __uint_as_float(sacc_s[8]) + __uint_as_float(sacc_s[10]) + __uint_as_float(sacc_s[12]) + __uint_as_float(sacc_s[14]);
                sv3 = __uint_as_float(sacc_s[1]) + __uint_as_float(sacc_s[3]) + __uint_as_float(sacc_s[5]) + __uint_as_float(sacc_s[7]) + __uint_as_float(sacc_s[9]) + __uint_as_float(sacc_s[11]) + __uint_as_float(sacc_s[13]) + __uint_as_float(sacc_s[15]);
                sv0 = ((v0 >= 0) ? sv0 * qk_scale : -1e+30f);
                sv1 = ((v1 >= 0) ? sv1 * qk_scale : -1e+30f);
                sv2 = ((v2 >= 0) ? sv2 * qk_scale : -1e+30f);
                sv3 = ((v3 >= 0) ? sv3 * qk_scale : -1e+30f);
                float _fmax_7 = fmaxf(sv0, sv1);
                float _fmax_8 = fmaxf(sv2, sv3);
                float _fmax_9 = fmaxf(_fmax_7, _fmax_8);
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, _fmax_9, 4);
                float _fmax_10 = fmaxf(_fmax_9, _shfl_xor_0);
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _fmax_10, 2);
                float _fmax_11 = fmaxf(_fmax_10, _shfl_xor_1);
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, _fmax_11, 1);
                float _fmax_12 = fmaxf(_fmax_11, _shfl_xor_2);
                mx = _fmax_12;
                float _fmax_13 = fmaxf(m_run, mx);
                m_new = _fmax_13;
                float _exp2_8 = approx_exp2(m_run - m_new);
                alpha = _exp2_8;
                float _exp2_9 = approx_exp2(sv0 - m_new);
                p0 = ((v0 >= 0) ? _exp2_9 : 0.0f);
                float _exp2_10 = approx_exp2(sv1 - m_new);
                p1 = ((v1 >= 0) ? _exp2_10 : 0.0f);
                float _exp2_11 = approx_exp2(sv2 - m_new);
                p2 = ((v2 >= 0) ? _exp2_11 : 0.0f);
                float _exp2_12 = approx_exp2(sv3 - m_new);
                p3 = ((v3 >= 0) ? _exp2_12 : 0.0f);
                l_run = l_run * alpha + (p0 + p1 + (p2 + p3));
                m_run = m_new;
                pvals[0] = p0;
                pvals[1] = p1;
                pvals2[0] = p2;
                pvals2[1] = p3;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(pvals[_lp*2 + 0], pvals[_lp*2+1 + 0]));
                    pw[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(pvals2[_lp*2 + 0], pvals2[_lp*2+1 + 0]));
                    pw2[_lp] = *(uint32_t*)&_h2;
                }
                p_u32[spar3 * 320 + sh * 20 + (4 * sq + scol ^ (sh >> 3 & 1) << 3)] = pw[0];
                p_u32[spar3 * 320 + sh * 20 + (4 * sq + 8 + scol ^ (sh >> 3 & 1) << 3)] = pw2[0];
                if ((lane_s & 7) == 0) {
                    alpha_f32[spar3 * 16 + sh] = alpha;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(p_ready_addr + (sxs) * 8);
                }
            }
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, l_run, 4);
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, l_run + _shfl_xor_3, 2);
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, l_run + _shfl_xor_3 + _shfl_xor_4, 1);
            l_run = l_run + _shfl_xor_3 + _shfl_xor_4 + _shfl_xor_5;
            if ((lane_s & 7) == 0) {
                ml_f32[sh * 2] = m_run;
                ml_f32[sh * 2 + 1] = l_run;
            }
            asm volatile("barrier.sync 3, 384;" ::: "memory");
        }
    }
    // ---- Role: io ----
    if (warp >= 12 && warp <= 13) {
        { // io_main
            int io_tid = tid - 384;
            int e_i = 0;
            int idxs[32];
            #pragma unroll
            for (int r_1 = 0; r_1 < 32; r_1++) {
                e_i = io_tid + r_1 * 64;
                idxs[r_1] = -1;
                if (e_i < n_keys) {
                    int _vec_load_0[1];
                    {
                        uint32_t _scalar_bits_0;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_0) : "l"((const void*)(indices + (token * topk + key_lo + e_i))) : "memory");
                        _vec_load_0[0] = (int32_t)_scalar_bits_0;
                    }
                    idxs[r_1] = _vec_load_0[0];
                }
            }
            #pragma unroll
            for (int r_2 = 0; r_2 < 32; r_2++) {
                e_i = io_tid + r_2 * 64;
                itab[e_i] = idxs[r_2];
            }
            asm volatile("barrier.sync 2, 448;" ::: "memory");
            unsigned int io_stage = 0;
            unsigned int io_phase = 0;
            unsigned long long kv_u64 = (unsigned long long)kv;
            for (int st_3 = 0; st_3 < num_stages; st_3++) {
                mbarrier_wait(kv_empty_addr + (io_stage) * 8, io_phase ^ 1);
                int slot_byte = (int)io_stage * 11776;
                int _itab_reg_0[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_0[_lr] = _smem_ptr[(st_3 * 32 + io_tid / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + io_tid / 22 * 368 + (io_tid - io_tid / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_0[0] >= 0) ? _itab_reg_0[0] : 0)) * 352 + (unsigned long long)((io_tid - io_tid / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_0[0] >= 0) ? 16 : 0))));
                int _itab_reg_1[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_1[_lr] = _smem_ptr[(st_3 * 32 + (64 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (64 + io_tid) / 22 * 368 + (64 + io_tid - (64 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_1[0] >= 0) ? _itab_reg_1[0] : 0)) * 352 + (unsigned long long)((64 + io_tid - (64 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_1[0] >= 0) ? 16 : 0))));
                int _itab_reg_2[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_2[_lr] = _smem_ptr[(st_3 * 32 + (128 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (128 + io_tid) / 22 * 368 + (128 + io_tid - (128 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_2[0] >= 0) ? _itab_reg_2[0] : 0)) * 352 + (unsigned long long)((128 + io_tid - (128 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_2[0] >= 0) ? 16 : 0))));
                int _itab_reg_3[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_3[_lr] = _smem_ptr[(st_3 * 32 + (192 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (192 + io_tid) / 22 * 368 + (192 + io_tid - (192 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_3[0] >= 0) ? _itab_reg_3[0] : 0)) * 352 + (unsigned long long)((192 + io_tid - (192 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_3[0] >= 0) ? 16 : 0))));
                int _itab_reg_4[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_4[_lr] = _smem_ptr[(st_3 * 32 + (256 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (256 + io_tid) / 22 * 368 + (256 + io_tid - (256 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_4[0] >= 0) ? _itab_reg_4[0] : 0)) * 352 + (unsigned long long)((256 + io_tid - (256 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_4[0] >= 0) ? 16 : 0))));
                int _itab_reg_5[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_5[_lr] = _smem_ptr[(st_3 * 32 + (320 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (320 + io_tid) / 22 * 368 + (320 + io_tid - (320 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_5[0] >= 0) ? _itab_reg_5[0] : 0)) * 352 + (unsigned long long)((320 + io_tid - (320 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_5[0] >= 0) ? 16 : 0))));
                int _itab_reg_6[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_6[_lr] = _smem_ptr[(st_3 * 32 + (384 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (384 + io_tid) / 22 * 368 + (384 + io_tid - (384 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_6[0] >= 0) ? _itab_reg_6[0] : 0)) * 352 + (unsigned long long)((384 + io_tid - (384 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_6[0] >= 0) ? 16 : 0))));
                int _itab_reg_7[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_7[_lr] = _smem_ptr[(st_3 * 32 + (448 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (448 + io_tid) / 22 * 368 + (448 + io_tid - (448 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_7[0] >= 0) ? _itab_reg_7[0] : 0)) * 352 + (unsigned long long)((448 + io_tid - (448 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_7[0] >= 0) ? 16 : 0))));
                int _itab_reg_8[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_8[_lr] = _smem_ptr[(st_3 * 32 + (512 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (512 + io_tid) / 22 * 368 + (512 + io_tid - (512 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_8[0] >= 0) ? _itab_reg_8[0] : 0)) * 352 + (unsigned long long)((512 + io_tid - (512 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_8[0] >= 0) ? 16 : 0))));
                int _itab_reg_9[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_9[_lr] = _smem_ptr[(st_3 * 32 + (576 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (576 + io_tid) / 22 * 368 + (576 + io_tid - (576 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_9[0] >= 0) ? _itab_reg_9[0] : 0)) * 352 + (unsigned long long)((576 + io_tid - (576 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_9[0] >= 0) ? 16 : 0))));
                int _itab_reg_10[1];
                {
                    const int* _smem_ptr = reinterpret_cast<const int*>(itab);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _itab_reg_10[_lr] = _smem_ptr[(st_3 * 32 + (640 + io_tid) / 22) + _lr];
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(kv_u8_addr + (unsigned int)(slot_byte + (640 + io_tid) / 22 * 368 + (640 + io_tid - (640 + io_tid) / 22 * 22) * 16)), "l"(kv_u64 + (unsigned long long)(((_itab_reg_10[0] >= 0) ? _itab_reg_10[0] : 0)) * 352 + (unsigned long long)((640 + io_tid - (640 + io_tid) / 22 * 22) * 16)), "r"((unsigned int)(((_itab_reg_10[0] >= 0) ? 16 : 0))));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(kv_full_addr + (io_stage) * 8) : "memory");
                io_stage += 1;
                if (io_stage == 3) { io_stage = 0; io_phase ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
