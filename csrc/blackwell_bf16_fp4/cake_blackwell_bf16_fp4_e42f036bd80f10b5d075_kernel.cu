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
#define TMEM_NCOLS 256
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 5
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 16384
#define SMEM_SMEM_ACT_STRIDE 27648
#define SMEM_SMEM_PACKED_OFF 17408
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 27648
#define SMEM_SMEM_SCALE_OFF 19456
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 27648
#define SMEM_SMEM_WEIGHT_OFF 20480
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 27648
#define SMEM_TOTAL 139264
#define HAS_ALPHA 0
#define ENABLE_PDL 0

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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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




__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_blackwell_bf16_fp4_e42f036bd80f10b5d075(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define act_full_addr (mbar_base + 0)
    #define act_done_addr (mbar_base + 40)
    #define packed_full_addr (mbar_base + 80)
    #define packed_done_addr (mbar_base + 120)
    #define weight_full_addr (mbar_base + 160)
    #define weight_done_addr (mbar_base + 200)
    #define output_full_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    uint8_t* smem_packed = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_packed_addr = smem + 17408;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_scale_addr = smem + 19456;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 20480);
    const int smem_weight_addr = smem + 20480;

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 31 barriers)
    // Mbarriers at smem_raw[0..248)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // act_done: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // packed_full: 5 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // packed_done: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // weight_full: 5 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // weight_done: 5 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 248);
    if (warp == 0) {
        int _tmem_hold = smem + 248;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int tile_m = blockIdx.y;
            int tile_n = blockIdx.x;
            int off_m = tile_m * 128;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            unsigned int _phase_output_full_0 = 0;
            mbarrier_wait(output_full_addr, _phase_output_full_0);
            _phase_output_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            long long output_plane = 0;
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local = lane_pair * 2;
            int n_local = row_base + ((0) ? 8 : 0);
            int m_global = off_m + m_local;
            int n_global = off_n + n_local;
            if (m_global < M && n_global < N) {
                long long output_linear = output_plane + (long long)m_global * (long long)N + (long long)n_global;
                float value = _tmem_load_0[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear) + (0)) = __float2bfloat16_rn(value);
            }
            int m_local_0 = lane_pair * 2 + 1;
            int n_local_1 = row_base + ((0) ? 8 : 0);
            int m_global_2 = off_m + m_local_0;
            int n_global_3 = off_n + n_local_1;
            if (m_global_2 < M && n_global_3 < N) {
                long long output_linear_1 = output_plane + (long long)m_global_2 * (long long)N + (long long)n_global_3;
                float value_1 = _tmem_load_0[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_1) + (0)) = __float2bfloat16_rn(value_1);
            }
            int m_local_4 = lane_pair * 2;
            int n_local_5 = row_base + ((1) ? 8 : 0);
            int m_global_6 = off_m + m_local_4;
            int n_global_7 = off_n + n_local_5;
            if (m_global_6 < M && n_global_7 < N) {
                long long output_linear_2 = output_plane + (long long)m_global_6 * (long long)N + (long long)n_global_7;
                float value_2 = _tmem_load_0[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_2) + (0)) = __float2bfloat16_rn(value_2);
            }
            int m_local_8 = lane_pair * 2 + 1;
            int n_local_9 = row_base + ((1) ? 8 : 0);
            int m_global_10 = off_m + m_local_8;
            int n_global_11 = off_n + n_local_9;
            if (m_global_10 < M && n_global_11 < N) {
                long long output_linear_3 = output_plane + (long long)m_global_10 * (long long)N + (long long)n_global_11;
                float value_3 = _tmem_load_0[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_3) + (0)) = __float2bfloat16_rn(value_3);
            }
            int m_local_12 = 8 + lane_pair * 2;
            int n_local_13 = row_base + ((0) ? 8 : 0);
            int m_global_14 = off_m + m_local_12;
            int n_global_15 = off_n + n_local_13;
            if (m_global_14 < M && n_global_15 < N) {
                long long output_linear_4 = output_plane + (long long)m_global_14 * (long long)N + (long long)n_global_15;
                float value_4 = _tmem_load_0[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_4) + (0)) = __float2bfloat16_rn(value_4);
            }
            int m_local_16 = 8 + lane_pair * 2 + 1;
            int n_local_17 = row_base + ((0) ? 8 : 0);
            int m_global_18 = off_m + m_local_16;
            int n_global_19 = off_n + n_local_17;
            if (m_global_18 < M && n_global_19 < N) {
                long long output_linear_5 = output_plane + (long long)m_global_18 * (long long)N + (long long)n_global_19;
                float value_5 = _tmem_load_0[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_5) + (0)) = __float2bfloat16_rn(value_5);
            }
            int m_local_20 = 8 + lane_pair * 2;
            int n_local_21 = row_base + ((1) ? 8 : 0);
            int m_global_22 = off_m + m_local_20;
            int n_global_23 = off_n + n_local_21;
            if (m_global_22 < M && n_global_23 < N) {
                long long output_linear_6 = output_plane + (long long)m_global_22 * (long long)N + (long long)n_global_23;
                float value_6 = _tmem_load_0[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_6) + (0)) = __float2bfloat16_rn(value_6);
            }
            int m_local_24 = 8 + lane_pair * 2 + 1;
            int n_local_25 = row_base + ((1) ? 8 : 0);
            int m_global_26 = off_m + m_local_24;
            int n_global_27 = off_n + n_local_25;
            if (m_global_26 < M && n_global_27 < N) {
                long long output_linear_7 = output_plane + (long long)m_global_26 * (long long)N + (long long)n_global_27;
                float value_7 = _tmem_load_0[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_7) + (0)) = __float2bfloat16_rn(value_7);
            }
            float _tmem_load_1[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7]))
                : "r"(taddr + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local_28 = 16 + lane_pair * 2;
            int n_local_29 = row_base + ((0) ? 8 : 0);
            int m_global_30 = off_m + m_local_28;
            int n_global_31 = off_n + n_local_29;
            if (m_global_30 < M && n_global_31 < N) {
                long long output_linear_8 = output_plane + (long long)m_global_30 * (long long)N + (long long)n_global_31;
                float value_8 = _tmem_load_1[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_8) + (0)) = __float2bfloat16_rn(value_8);
            }
            int m_local_32 = 16 + lane_pair * 2 + 1;
            int n_local_33 = row_base + ((0) ? 8 : 0);
            int m_global_34 = off_m + m_local_32;
            int n_global_35 = off_n + n_local_33;
            if (m_global_34 < M && n_global_35 < N) {
                long long output_linear_9 = output_plane + (long long)m_global_34 * (long long)N + (long long)n_global_35;
                float value_9 = _tmem_load_1[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_9) + (0)) = __float2bfloat16_rn(value_9);
            }
            int m_local_36 = 16 + lane_pair * 2;
            int n_local_37 = row_base + ((1) ? 8 : 0);
            int m_global_38 = off_m + m_local_36;
            int n_global_39 = off_n + n_local_37;
            if (m_global_38 < M && n_global_39 < N) {
                long long output_linear_10 = output_plane + (long long)m_global_38 * (long long)N + (long long)n_global_39;
                float value_10 = _tmem_load_1[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_10) + (0)) = __float2bfloat16_rn(value_10);
            }
            int m_local_40 = 16 + lane_pair * 2 + 1;
            int n_local_41 = row_base + ((1) ? 8 : 0);
            int m_global_42 = off_m + m_local_40;
            int n_global_43 = off_n + n_local_41;
            if (m_global_42 < M && n_global_43 < N) {
                long long output_linear_11 = output_plane + (long long)m_global_42 * (long long)N + (long long)n_global_43;
                float value_11 = _tmem_load_1[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_11) + (0)) = __float2bfloat16_rn(value_11);
            }
            int m_local_44 = 24 + lane_pair * 2;
            int n_local_45 = row_base + ((0) ? 8 : 0);
            int m_global_46 = off_m + m_local_44;
            int n_global_47 = off_n + n_local_45;
            if (m_global_46 < M && n_global_47 < N) {
                long long output_linear_12 = output_plane + (long long)m_global_46 * (long long)N + (long long)n_global_47;
                float value_12 = _tmem_load_1[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_12) + (0)) = __float2bfloat16_rn(value_12);
            }
            int m_local_48 = 24 + lane_pair * 2 + 1;
            int n_local_49 = row_base + ((0) ? 8 : 0);
            int m_global_50 = off_m + m_local_48;
            int n_global_51 = off_n + n_local_49;
            if (m_global_50 < M && n_global_51 < N) {
                long long output_linear_13 = output_plane + (long long)m_global_50 * (long long)N + (long long)n_global_51;
                float value_13 = _tmem_load_1[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_13) + (0)) = __float2bfloat16_rn(value_13);
            }
            int m_local_52 = 24 + lane_pair * 2;
            int n_local_53 = row_base + ((1) ? 8 : 0);
            int m_global_54 = off_m + m_local_52;
            int n_global_55 = off_n + n_local_53;
            if (m_global_54 < M && n_global_55 < N) {
                long long output_linear_14 = output_plane + (long long)m_global_54 * (long long)N + (long long)n_global_55;
                float value_14 = _tmem_load_1[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_14) + (0)) = __float2bfloat16_rn(value_14);
            }
            int m_local_56 = 24 + lane_pair * 2 + 1;
            int n_local_57 = row_base + ((1) ? 8 : 0);
            int m_global_58 = off_m + m_local_56;
            int n_global_59 = off_n + n_local_57;
            if (m_global_58 < M && n_global_59 < N) {
                long long output_linear_15 = output_plane + (long long)m_global_58 * (long long)N + (long long)n_global_59;
                float value_15 = _tmem_load_1[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_15) + (0)) = __float2bfloat16_rn(value_15);
            }
            float _tmem_load_2[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7]))
                : "r"(taddr + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local_60 = 32 + lane_pair * 2;
            int n_local_61 = row_base + ((0) ? 8 : 0);
            int m_global_62 = off_m + m_local_60;
            int n_global_63 = off_n + n_local_61;
            if (m_global_62 < M && n_global_63 < N) {
                long long output_linear_16 = output_plane + (long long)m_global_62 * (long long)N + (long long)n_global_63;
                float value_16 = _tmem_load_2[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_16) + (0)) = __float2bfloat16_rn(value_16);
            }
            int m_local_64 = 32 + lane_pair * 2 + 1;
            int n_local_65 = row_base + ((0) ? 8 : 0);
            int m_global_66 = off_m + m_local_64;
            int n_global_67 = off_n + n_local_65;
            if (m_global_66 < M && n_global_67 < N) {
                long long output_linear_17 = output_plane + (long long)m_global_66 * (long long)N + (long long)n_global_67;
                float value_17 = _tmem_load_2[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_17) + (0)) = __float2bfloat16_rn(value_17);
            }
            int m_local_68 = 32 + lane_pair * 2;
            int n_local_69 = row_base + ((1) ? 8 : 0);
            int m_global_70 = off_m + m_local_68;
            int n_global_71 = off_n + n_local_69;
            if (m_global_70 < M && n_global_71 < N) {
                long long output_linear_18 = output_plane + (long long)m_global_70 * (long long)N + (long long)n_global_71;
                float value_18 = _tmem_load_2[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_18) + (0)) = __float2bfloat16_rn(value_18);
            }
            int m_local_72 = 32 + lane_pair * 2 + 1;
            int n_local_73 = row_base + ((1) ? 8 : 0);
            int m_global_74 = off_m + m_local_72;
            int n_global_75 = off_n + n_local_73;
            if (m_global_74 < M && n_global_75 < N) {
                long long output_linear_19 = output_plane + (long long)m_global_74 * (long long)N + (long long)n_global_75;
                float value_19 = _tmem_load_2[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_19) + (0)) = __float2bfloat16_rn(value_19);
            }
            int m_local_76 = 40 + lane_pair * 2;
            int n_local_77 = row_base + ((0) ? 8 : 0);
            int m_global_78 = off_m + m_local_76;
            int n_global_79 = off_n + n_local_77;
            if (m_global_78 < M && n_global_79 < N) {
                long long output_linear_20 = output_plane + (long long)m_global_78 * (long long)N + (long long)n_global_79;
                float value_20 = _tmem_load_2[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_20) + (0)) = __float2bfloat16_rn(value_20);
            }
            int m_local_80 = 40 + lane_pair * 2 + 1;
            int n_local_81 = row_base + ((0) ? 8 : 0);
            int m_global_82 = off_m + m_local_80;
            int n_global_83 = off_n + n_local_81;
            if (m_global_82 < M && n_global_83 < N) {
                long long output_linear_21 = output_plane + (long long)m_global_82 * (long long)N + (long long)n_global_83;
                float value_21 = _tmem_load_2[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_21) + (0)) = __float2bfloat16_rn(value_21);
            }
            int m_local_84 = 40 + lane_pair * 2;
            int n_local_85 = row_base + ((1) ? 8 : 0);
            int m_global_86 = off_m + m_local_84;
            int n_global_87 = off_n + n_local_85;
            if (m_global_86 < M && n_global_87 < N) {
                long long output_linear_22 = output_plane + (long long)m_global_86 * (long long)N + (long long)n_global_87;
                float value_22 = _tmem_load_2[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_22) + (0)) = __float2bfloat16_rn(value_22);
            }
            int m_local_88 = 40 + lane_pair * 2 + 1;
            int n_local_89 = row_base + ((1) ? 8 : 0);
            int m_global_90 = off_m + m_local_88;
            int n_global_91 = off_n + n_local_89;
            if (m_global_90 < M && n_global_91 < N) {
                long long output_linear_23 = output_plane + (long long)m_global_90 * (long long)N + (long long)n_global_91;
                float value_23 = _tmem_load_2[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_23) + (0)) = __float2bfloat16_rn(value_23);
            }
            float _tmem_load_3[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7]))
                : "r"(taddr + 96));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local_92 = 48 + lane_pair * 2;
            int n_local_93 = row_base + ((0) ? 8 : 0);
            int m_global_94 = off_m + m_local_92;
            int n_global_95 = off_n + n_local_93;
            if (m_global_94 < M && n_global_95 < N) {
                long long output_linear_24 = output_plane + (long long)m_global_94 * (long long)N + (long long)n_global_95;
                float value_24 = _tmem_load_3[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_24) + (0)) = __float2bfloat16_rn(value_24);
            }
            int m_local_96 = 48 + lane_pair * 2 + 1;
            int n_local_97 = row_base + ((0) ? 8 : 0);
            int m_global_98 = off_m + m_local_96;
            int n_global_99 = off_n + n_local_97;
            if (m_global_98 < M && n_global_99 < N) {
                long long output_linear_25 = output_plane + (long long)m_global_98 * (long long)N + (long long)n_global_99;
                float value_25 = _tmem_load_3[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_25) + (0)) = __float2bfloat16_rn(value_25);
            }
            int m_local_100 = 48 + lane_pair * 2;
            int n_local_101 = row_base + ((1) ? 8 : 0);
            int m_global_102 = off_m + m_local_100;
            int n_global_103 = off_n + n_local_101;
            if (m_global_102 < M && n_global_103 < N) {
                long long output_linear_26 = output_plane + (long long)m_global_102 * (long long)N + (long long)n_global_103;
                float value_26 = _tmem_load_3[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_26) + (0)) = __float2bfloat16_rn(value_26);
            }
            int m_local_104 = 48 + lane_pair * 2 + 1;
            int n_local_105 = row_base + ((1) ? 8 : 0);
            int m_global_106 = off_m + m_local_104;
            int n_global_107 = off_n + n_local_105;
            if (m_global_106 < M && n_global_107 < N) {
                long long output_linear_27 = output_plane + (long long)m_global_106 * (long long)N + (long long)n_global_107;
                float value_27 = _tmem_load_3[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_27) + (0)) = __float2bfloat16_rn(value_27);
            }
            int m_local_108 = 56 + lane_pair * 2;
            int n_local_109 = row_base + ((0) ? 8 : 0);
            int m_global_110 = off_m + m_local_108;
            int n_global_111 = off_n + n_local_109;
            if (m_global_110 < M && n_global_111 < N) {
                long long output_linear_28 = output_plane + (long long)m_global_110 * (long long)N + (long long)n_global_111;
                float value_28 = _tmem_load_3[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_28) + (0)) = __float2bfloat16_rn(value_28);
            }
            int m_local_112 = 56 + lane_pair * 2 + 1;
            int n_local_113 = row_base + ((0) ? 8 : 0);
            int m_global_114 = off_m + m_local_112;
            int n_global_115 = off_n + n_local_113;
            if (m_global_114 < M && n_global_115 < N) {
                long long output_linear_29 = output_plane + (long long)m_global_114 * (long long)N + (long long)n_global_115;
                float value_29 = _tmem_load_3[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_29) + (0)) = __float2bfloat16_rn(value_29);
            }
            int m_local_116 = 56 + lane_pair * 2;
            int n_local_117 = row_base + ((1) ? 8 : 0);
            int m_global_118 = off_m + m_local_116;
            int n_global_119 = off_n + n_local_117;
            if (m_global_118 < M && n_global_119 < N) {
                long long output_linear_30 = output_plane + (long long)m_global_118 * (long long)N + (long long)n_global_119;
                float value_30 = _tmem_load_3[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_30) + (0)) = __float2bfloat16_rn(value_30);
            }
            int m_local_120 = 56 + lane_pair * 2 + 1;
            int n_local_121 = row_base + ((1) ? 8 : 0);
            int m_global_122 = off_m + m_local_120;
            int n_global_123 = off_n + n_local_121;
            if (m_global_122 < M && n_global_123 < N) {
                long long output_linear_31 = output_plane + (long long)m_global_122 * (long long)N + (long long)n_global_123;
                float value_31 = _tmem_load_3[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_31) + (0)) = __float2bfloat16_rn(value_31);
            }
            float _tmem_load_4[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
                : "r"(taddr + 128));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local_124 = 64 + lane_pair * 2;
            int n_local_125 = row_base + ((0) ? 8 : 0);
            int m_global_126 = off_m + m_local_124;
            int n_global_127 = off_n + n_local_125;
            if (m_global_126 < M && n_global_127 < N) {
                long long output_linear_32 = output_plane + (long long)m_global_126 * (long long)N + (long long)n_global_127;
                float value_32 = _tmem_load_4[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_32) + (0)) = __float2bfloat16_rn(value_32);
            }
            int m_local_128 = 64 + lane_pair * 2 + 1;
            int n_local_129 = row_base + ((0) ? 8 : 0);
            int m_global_130 = off_m + m_local_128;
            int n_global_131 = off_n + n_local_129;
            if (m_global_130 < M && n_global_131 < N) {
                long long output_linear_33 = output_plane + (long long)m_global_130 * (long long)N + (long long)n_global_131;
                float value_33 = _tmem_load_4[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_33) + (0)) = __float2bfloat16_rn(value_33);
            }
            int m_local_132 = 64 + lane_pair * 2;
            int n_local_133 = row_base + ((1) ? 8 : 0);
            int m_global_134 = off_m + m_local_132;
            int n_global_135 = off_n + n_local_133;
            if (m_global_134 < M && n_global_135 < N) {
                long long output_linear_34 = output_plane + (long long)m_global_134 * (long long)N + (long long)n_global_135;
                float value_34 = _tmem_load_4[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_34) + (0)) = __float2bfloat16_rn(value_34);
            }
            int m_local_136 = 64 + lane_pair * 2 + 1;
            int n_local_137 = row_base + ((1) ? 8 : 0);
            int m_global_138 = off_m + m_local_136;
            int n_global_139 = off_n + n_local_137;
            if (m_global_138 < M && n_global_139 < N) {
                long long output_linear_35 = output_plane + (long long)m_global_138 * (long long)N + (long long)n_global_139;
                float value_35 = _tmem_load_4[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_35) + (0)) = __float2bfloat16_rn(value_35);
            }
            int m_local_140 = 72 + lane_pair * 2;
            int n_local_141 = row_base + ((0) ? 8 : 0);
            int m_global_142 = off_m + m_local_140;
            int n_global_143 = off_n + n_local_141;
            if (m_global_142 < M && n_global_143 < N) {
                long long output_linear_36 = output_plane + (long long)m_global_142 * (long long)N + (long long)n_global_143;
                float value_36 = _tmem_load_4[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_36) + (0)) = __float2bfloat16_rn(value_36);
            }
            int m_local_144 = 72 + lane_pair * 2 + 1;
            int n_local_145 = row_base + ((0) ? 8 : 0);
            int m_global_146 = off_m + m_local_144;
            int n_global_147 = off_n + n_local_145;
            if (m_global_146 < M && n_global_147 < N) {
                long long output_linear_37 = output_plane + (long long)m_global_146 * (long long)N + (long long)n_global_147;
                float value_37 = _tmem_load_4[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_37) + (0)) = __float2bfloat16_rn(value_37);
            }
            int m_local_148 = 72 + lane_pair * 2;
            int n_local_149 = row_base + ((1) ? 8 : 0);
            int m_global_150 = off_m + m_local_148;
            int n_global_151 = off_n + n_local_149;
            if (m_global_150 < M && n_global_151 < N) {
                long long output_linear_38 = output_plane + (long long)m_global_150 * (long long)N + (long long)n_global_151;
                float value_38 = _tmem_load_4[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_38) + (0)) = __float2bfloat16_rn(value_38);
            }
            int m_local_152 = 72 + lane_pair * 2 + 1;
            int n_local_153 = row_base + ((1) ? 8 : 0);
            int m_global_154 = off_m + m_local_152;
            int n_global_155 = off_n + n_local_153;
            if (m_global_154 < M && n_global_155 < N) {
                long long output_linear_39 = output_plane + (long long)m_global_154 * (long long)N + (long long)n_global_155;
                float value_39 = _tmem_load_4[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_39) + (0)) = __float2bfloat16_rn(value_39);
            }
            float _tmem_load_5[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7]))
                : "r"(taddr + 160));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local_156 = 80 + lane_pair * 2;
            int n_local_157 = row_base + ((0) ? 8 : 0);
            int m_global_158 = off_m + m_local_156;
            int n_global_159 = off_n + n_local_157;
            if (m_global_158 < M && n_global_159 < N) {
                long long output_linear_40 = output_plane + (long long)m_global_158 * (long long)N + (long long)n_global_159;
                float value_40 = _tmem_load_5[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_40) + (0)) = __float2bfloat16_rn(value_40);
            }
            int m_local_160 = 80 + lane_pair * 2 + 1;
            int n_local_161 = row_base + ((0) ? 8 : 0);
            int m_global_162 = off_m + m_local_160;
            int n_global_163 = off_n + n_local_161;
            if (m_global_162 < M && n_global_163 < N) {
                long long output_linear_41 = output_plane + (long long)m_global_162 * (long long)N + (long long)n_global_163;
                float value_41 = _tmem_load_5[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_41) + (0)) = __float2bfloat16_rn(value_41);
            }
            int m_local_164 = 80 + lane_pair * 2;
            int n_local_165 = row_base + ((1) ? 8 : 0);
            int m_global_166 = off_m + m_local_164;
            int n_global_167 = off_n + n_local_165;
            if (m_global_166 < M && n_global_167 < N) {
                long long output_linear_42 = output_plane + (long long)m_global_166 * (long long)N + (long long)n_global_167;
                float value_42 = _tmem_load_5[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_42) + (0)) = __float2bfloat16_rn(value_42);
            }
            int m_local_168 = 80 + lane_pair * 2 + 1;
            int n_local_169 = row_base + ((1) ? 8 : 0);
            int m_global_170 = off_m + m_local_168;
            int n_global_171 = off_n + n_local_169;
            if (m_global_170 < M && n_global_171 < N) {
                long long output_linear_43 = output_plane + (long long)m_global_170 * (long long)N + (long long)n_global_171;
                float value_43 = _tmem_load_5[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_43) + (0)) = __float2bfloat16_rn(value_43);
            }
            int m_local_172 = 88 + lane_pair * 2;
            int n_local_173 = row_base + ((0) ? 8 : 0);
            int m_global_174 = off_m + m_local_172;
            int n_global_175 = off_n + n_local_173;
            if (m_global_174 < M && n_global_175 < N) {
                long long output_linear_44 = output_plane + (long long)m_global_174 * (long long)N + (long long)n_global_175;
                float value_44 = _tmem_load_5[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_44) + (0)) = __float2bfloat16_rn(value_44);
            }
            int m_local_176 = 88 + lane_pair * 2 + 1;
            int n_local_177 = row_base + ((0) ? 8 : 0);
            int m_global_178 = off_m + m_local_176;
            int n_global_179 = off_n + n_local_177;
            if (m_global_178 < M && n_global_179 < N) {
                long long output_linear_45 = output_plane + (long long)m_global_178 * (long long)N + (long long)n_global_179;
                float value_45 = _tmem_load_5[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_45) + (0)) = __float2bfloat16_rn(value_45);
            }
            int m_local_180 = 88 + lane_pair * 2;
            int n_local_181 = row_base + ((1) ? 8 : 0);
            int m_global_182 = off_m + m_local_180;
            int n_global_183 = off_n + n_local_181;
            if (m_global_182 < M && n_global_183 < N) {
                long long output_linear_46 = output_plane + (long long)m_global_182 * (long long)N + (long long)n_global_183;
                float value_46 = _tmem_load_5[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_46) + (0)) = __float2bfloat16_rn(value_46);
            }
            int m_local_184 = 88 + lane_pair * 2 + 1;
            int n_local_185 = row_base + ((1) ? 8 : 0);
            int m_global_186 = off_m + m_local_184;
            int n_global_187 = off_n + n_local_185;
            if (m_global_186 < M && n_global_187 < N) {
                long long output_linear_47 = output_plane + (long long)m_global_186 * (long long)N + (long long)n_global_187;
                float value_47 = _tmem_load_5[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_47) + (0)) = __float2bfloat16_rn(value_47);
            }
            float _tmem_load_6[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7]))
                : "r"(taddr + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local_188 = 96 + lane_pair * 2;
            int n_local_189 = row_base + ((0) ? 8 : 0);
            int m_global_190 = off_m + m_local_188;
            int n_global_191 = off_n + n_local_189;
            if (m_global_190 < M && n_global_191 < N) {
                long long output_linear_48 = output_plane + (long long)m_global_190 * (long long)N + (long long)n_global_191;
                float value_48 = _tmem_load_6[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_48) + (0)) = __float2bfloat16_rn(value_48);
            }
            int m_local_192 = 96 + lane_pair * 2 + 1;
            int n_local_193 = row_base + ((0) ? 8 : 0);
            int m_global_194 = off_m + m_local_192;
            int n_global_195 = off_n + n_local_193;
            if (m_global_194 < M && n_global_195 < N) {
                long long output_linear_49 = output_plane + (long long)m_global_194 * (long long)N + (long long)n_global_195;
                float value_49 = _tmem_load_6[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_49) + (0)) = __float2bfloat16_rn(value_49);
            }
            int m_local_196 = 96 + lane_pair * 2;
            int n_local_197 = row_base + ((1) ? 8 : 0);
            int m_global_198 = off_m + m_local_196;
            int n_global_199 = off_n + n_local_197;
            if (m_global_198 < M && n_global_199 < N) {
                long long output_linear_50 = output_plane + (long long)m_global_198 * (long long)N + (long long)n_global_199;
                float value_50 = _tmem_load_6[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_50) + (0)) = __float2bfloat16_rn(value_50);
            }
            int m_local_200 = 96 + lane_pair * 2 + 1;
            int n_local_201 = row_base + ((1) ? 8 : 0);
            int m_global_202 = off_m + m_local_200;
            int n_global_203 = off_n + n_local_201;
            if (m_global_202 < M && n_global_203 < N) {
                long long output_linear_51 = output_plane + (long long)m_global_202 * (long long)N + (long long)n_global_203;
                float value_51 = _tmem_load_6[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_51) + (0)) = __float2bfloat16_rn(value_51);
            }
            int m_local_204 = 104 + lane_pair * 2;
            int n_local_205 = row_base + ((0) ? 8 : 0);
            int m_global_206 = off_m + m_local_204;
            int n_global_207 = off_n + n_local_205;
            if (m_global_206 < M && n_global_207 < N) {
                long long output_linear_52 = output_plane + (long long)m_global_206 * (long long)N + (long long)n_global_207;
                float value_52 = _tmem_load_6[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_52) + (0)) = __float2bfloat16_rn(value_52);
            }
            int m_local_208 = 104 + lane_pair * 2 + 1;
            int n_local_209 = row_base + ((0) ? 8 : 0);
            int m_global_210 = off_m + m_local_208;
            int n_global_211 = off_n + n_local_209;
            if (m_global_210 < M && n_global_211 < N) {
                long long output_linear_53 = output_plane + (long long)m_global_210 * (long long)N + (long long)n_global_211;
                float value_53 = _tmem_load_6[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_53) + (0)) = __float2bfloat16_rn(value_53);
            }
            int m_local_212 = 104 + lane_pair * 2;
            int n_local_213 = row_base + ((1) ? 8 : 0);
            int m_global_214 = off_m + m_local_212;
            int n_global_215 = off_n + n_local_213;
            if (m_global_214 < M && n_global_215 < N) {
                long long output_linear_54 = output_plane + (long long)m_global_214 * (long long)N + (long long)n_global_215;
                float value_54 = _tmem_load_6[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_54) + (0)) = __float2bfloat16_rn(value_54);
            }
            int m_local_216 = 104 + lane_pair * 2 + 1;
            int n_local_217 = row_base + ((1) ? 8 : 0);
            int m_global_218 = off_m + m_local_216;
            int n_global_219 = off_n + n_local_217;
            if (m_global_218 < M && n_global_219 < N) {
                long long output_linear_55 = output_plane + (long long)m_global_218 * (long long)N + (long long)n_global_219;
                float value_55 = _tmem_load_6[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_55) + (0)) = __float2bfloat16_rn(value_55);
            }
            float _tmem_load_7[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7]))
                : "r"(taddr + 224));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int m_local_220 = 112 + lane_pair * 2;
            int n_local_221 = row_base + ((0) ? 8 : 0);
            int m_global_222 = off_m + m_local_220;
            int n_global_223 = off_n + n_local_221;
            if (m_global_222 < M && n_global_223 < N) {
                long long output_linear_56 = output_plane + (long long)m_global_222 * (long long)N + (long long)n_global_223;
                float value_56 = _tmem_load_7[0] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_56) + (0)) = __float2bfloat16_rn(value_56);
            }
            int m_local_224 = 112 + lane_pair * 2 + 1;
            int n_local_225 = row_base + ((0) ? 8 : 0);
            int m_global_226 = off_m + m_local_224;
            int n_global_227 = off_n + n_local_225;
            if (m_global_226 < M && n_global_227 < N) {
                long long output_linear_57 = output_plane + (long long)m_global_226 * (long long)N + (long long)n_global_227;
                float value_57 = _tmem_load_7[1] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_57) + (0)) = __float2bfloat16_rn(value_57);
            }
            int m_local_228 = 112 + lane_pair * 2;
            int n_local_229 = row_base + ((1) ? 8 : 0);
            int m_global_230 = off_m + m_local_228;
            int n_global_231 = off_n + n_local_229;
            if (m_global_230 < M && n_global_231 < N) {
                long long output_linear_58 = output_plane + (long long)m_global_230 * (long long)N + (long long)n_global_231;
                float value_58 = _tmem_load_7[2] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_58) + (0)) = __float2bfloat16_rn(value_58);
            }
            int m_local_232 = 112 + lane_pair * 2 + 1;
            int n_local_233 = row_base + ((1) ? 8 : 0);
            int m_global_234 = off_m + m_local_232;
            int n_global_235 = off_n + n_local_233;
            if (m_global_234 < M && n_global_235 < N) {
                long long output_linear_59 = output_plane + (long long)m_global_234 * (long long)N + (long long)n_global_235;
                float value_59 = _tmem_load_7[3] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_59) + (0)) = __float2bfloat16_rn(value_59);
            }
            int m_local_236 = 120 + lane_pair * 2;
            int n_local_237 = row_base + ((0) ? 8 : 0);
            int m_global_238 = off_m + m_local_236;
            int n_global_239 = off_n + n_local_237;
            if (m_global_238 < M && n_global_239 < N) {
                long long output_linear_60 = output_plane + (long long)m_global_238 * (long long)N + (long long)n_global_239;
                float value_60 = _tmem_load_7[4] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_60) + (0)) = __float2bfloat16_rn(value_60);
            }
            int m_local_240 = 120 + lane_pair * 2 + 1;
            int n_local_241 = row_base + ((0) ? 8 : 0);
            int m_global_242 = off_m + m_local_240;
            int n_global_243 = off_n + n_local_241;
            if (m_global_242 < M && n_global_243 < N) {
                long long output_linear_61 = output_plane + (long long)m_global_242 * (long long)N + (long long)n_global_243;
                float value_61 = _tmem_load_7[5] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_61) + (0)) = __float2bfloat16_rn(value_61);
            }
            int m_local_244 = 120 + lane_pair * 2;
            int n_local_245 = row_base + ((1) ? 8 : 0);
            int m_global_246 = off_m + m_local_244;
            int n_global_247 = off_n + n_local_245;
            if (m_global_246 < M && n_global_247 < N) {
                long long output_linear_62 = output_plane + (long long)m_global_246 * (long long)N + (long long)n_global_247;
                float value_62 = _tmem_load_7[6] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_62) + (0)) = __float2bfloat16_rn(value_62);
            }
            int m_local_248 = 120 + lane_pair * 2 + 1;
            int n_local_249 = row_base + ((1) ? 8 : 0);
            int m_global_250 = off_m + m_local_248;
            int n_global_251 = off_n + n_local_249;
            if (m_global_250 < M && n_global_251 < N) {
                long long output_linear_63 = output_plane + (long long)m_global_250 * (long long)N + (long long)n_global_251;
                float value_63 = _tmem_load_7[7] * alpha_value;
                *(reinterpret_cast<__nv_bfloat16*>(C + output_linear_63) + (0)) = __float2bfloat16_rn(value_63);
            }
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int k_extent = K;
            int k_tiles = (k_extent + 64 - 1) / 64;
            unsigned int mma_stage = 0;
            unsigned int _phase_act_full = 0;
            unsigned int _phase_weight_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(act_full_addr + (mma_stage) * 8, _phase_act_full);
                    mbarrier_wait(weight_full_addr + (mma_stage) * 8, _phase_weight_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((kt == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                    int _mma_a_lo_1 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    int _mma_b_lo_1 = (((smem_act_addr + 2048) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (32))), "r"(((init_flag) ? 0 : 1)));
                    int _mma_a_lo_2 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    int _mma_b_lo_2 = (((smem_act_addr + 4096) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_accum + (64))), "r"(((init_flag) ? 0 : 1)));
                    int _mma_a_lo_3 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    int _mma_b_lo_3 = (((smem_act_addr + 6144) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_accum + (96))), "r"(((init_flag) ? 0 : 1)));
                    int _mma_a_lo_4 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    int _mma_b_lo_4 = (((smem_act_addr + 8192) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_accum + (128))), "r"(((init_flag) ? 0 : 1)));
                    int _mma_a_lo_5 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    int _mma_b_lo_5 = (((smem_act_addr + 10240) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_accum + (160))), "r"(((init_flag) ? 0 : 1)));
                    int _mma_a_lo_6 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    int _mma_b_lo_6 = (((smem_act_addr + 12288) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_accum + (192))), "r"(((init_flag) ? 0 : 1)));
                    int _mma_a_lo_7 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    int _mma_b_lo_7 = (((smem_act_addr + 14336) >> 4) & 0x3FFF) + (mma_stage) * 1728;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_accum + (224))), "r"(((init_flag) ? 0 : 1)));
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 5) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int tile_m_1 = blockIdx.y;
            int off_m_1 = tile_m_1 * 128;
            int k_begin = 0;
            int k_extent_1 = K;
            int k_tiles_1 = (k_extent_1 + 64 - 1) / 64;
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 27648, (&A), k_begin + kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 16384);
                    act_stage += 1;
                    if (act_stage == 5) { act_stage = 0; _phase_act_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_weight ----
    if (warp == 6) {
        { // load_weight_main
            int tile_n_1 = blockIdx.x;
            int off_n_1 = tile_n_1 * 64;
            int k_begin_1 = 0;
            int k_extent_2 = K;
            int k_tiles_2 = (k_extent_2 + 64 - 1) / 64;
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 27648, (&B), k_begin_1 / 2 + kt_2 * 32, off_n_1, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 27648, (&B_descale), k_begin_1 / 16 + kt_2 / 4 * 16, off_n_1, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((0) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 5) { packed_stage = 0; _phase_packed_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 7) {
        // idle — no tasks assigned
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int k_extent_3 = K;
            int k_tiles_3 = (k_extent_3 + 64 - 1) / 64;
            unsigned int convert_stage = 0;
            int warp_id_in_role = (warp - 8);
            int convert_tid = warp_id_in_role * 32 + lane;
            unsigned int raw_words[2];
            unsigned int scale_word[1];
            unsigned int _phase_packed_full = 0;
            unsigned int _phase_weight_done = 1;
            #pragma unroll 1
            for (int kt_3 = 0; kt_3 < k_tiles_3; kt_3++) {
                mbarrier_wait(packed_full_addr + (convert_stage) * 8, _phase_packed_full);
                mbarrier_wait(weight_done_addr + (convert_stage) * 8, _phase_weight_done);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int packed_base = smem_packed_addr + convert_stage * 27648;
                int scale_base = smem_scale_addr + convert_stage * 27648;
                int weight_row = convert_tid / 4;
                int k_block = convert_tid - weight_row * 4;
                int pair_col_base = k_block * 8;
                unsigned int packed0 = 0;
                unsigned int packed1 = 0;
                uint8_t scale_byte = (uint8_t)0;
                int word_linear = weight_row * 8 + k_block * 2;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(0) + 1]))
                    : "r"(packed_base + word_linear * 4));
                packed0 = raw_words[0];
                packed1 = raw_words[1];
                int scale_group_offset = 0;
                {
                    scale_group_offset = kt_3 % 4 * 4;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + weight_row * ((0) ? 4 : 16) + scale_group_offset));
                scale_byte = (uint8_t)(scale_word[0] >> (unsigned int)(k_block * 8) & 255);
                uint32_t _fp4_dequant_block16_bf16_0[8];
                {
                    uint32_t _scale_byte = ((uint32_t)(scale_byte)) & 0xFFu;
                    uint16_t _scale_e4m3x2 = (uint16_t)(_scale_byte | (_scale_byte << 8));
                    uint32_t _scale_bf16x2;
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_scale_bf16x2) : "h"(_scale_e4m3x2));
                    #else
                    uint32_t _f16x2;
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_scale_e4m3x2));
                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                    float _f0;
                    float _f1;
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_f1), "f"(_f0));
                    #endif
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(packed0)) >> 0))) & 0xFFu);
                        uint32_t _fp4_bf16x2;
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                            : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                        #else
                        uint32_t _fp4_f16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                        uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                        uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                        float _fp4_f0;
                        float _fp4_f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                        #endif
                        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[0]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                    }
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(packed0)) >> 8))) & 0xFFu);
                        uint32_t _fp4_bf16x2;
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                            : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                        #else
                        uint32_t _fp4_f16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                        uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                        uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                        float _fp4_f0;
                        float _fp4_f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                        #endif
                        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[1]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                    }
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(packed0)) >> 16))) & 0xFFu);
                        uint32_t _fp4_bf16x2;
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                            : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                        #else
                        uint32_t _fp4_f16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                        uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                        uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                        float _fp4_f0;
                        float _fp4_f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                        #endif
                        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[2]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                    }
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(packed0)) >> 24))) & 0xFFu);
                        uint32_t _fp4_bf16x2;
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                            : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                        #else
                        uint32_t _fp4_f16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                        uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                        uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                        float _fp4_f0;
                        float _fp4_f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                        #endif
                        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[3]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                    }
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(packed1)) >> 0))) & 0xFFu);
                        uint32_t _fp4_bf16x2;
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                            : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                        #else
                        uint32_t _fp4_f16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                        uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                        uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                        float _fp4_f0;
                        float _fp4_f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                        #endif
                        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[4]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                    }
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(packed1)) >> 8))) & 0xFFu);
                        uint32_t _fp4_bf16x2;
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                            : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                        #else
                        uint32_t _fp4_f16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                        uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                        uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                        float _fp4_f0;
                        float _fp4_f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                        #endif
                        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[5]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                    }
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(packed1)) >> 16))) & 0xFFu);
                        uint32_t _fp4_bf16x2;
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                            : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                        #else
                        uint32_t _fp4_f16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                        uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                        uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                        float _fp4_f0;
                        float _fp4_f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                        #endif
                        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[6]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                    }
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)((((uint32_t)(packed1)) >> 24))) & 0xFFu);
                        uint32_t _fp4_bf16x2;
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
                            : "=r"(_fp4_bf16x2) : "h"(_fp4_u16));
                        #else
                        uint32_t _fp4_f16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_f16x2) : "h"(_fp4_u16));
                        uint16_t _fp4_h0 = (uint16_t)(_fp4_f16x2 & 0xFFFFu);
                        uint16_t _fp4_h1 = (uint16_t)((_fp4_f16x2 >> 16) & 0xFFFFu);
                        float _fp4_f0;
                        float _fp4_f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f0) : "h"(_fp4_h0));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp4_f1) : "h"(_fp4_h1));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x2) : "f"(_fp4_f1), "f"(_fp4_f0));
                        #endif
                        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_block16_bf16_0[7]) : "r"(_fp4_bf16x2), "r"(_scale_bf16x2));
                    }
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 27648 + (unsigned int)(pair_col_base * 2 / 64 * 8192 + weight_row * 128 + pair_col_base * 2 % 64 * 2 ^ (pair_col_base * 2 / 64 * 8192 + weight_row * 128 + pair_col_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 27648 + (unsigned int)((pair_col_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 1) * 2 % 64 * 2 ^ ((pair_col_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 27648 + (unsigned int)((pair_col_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 2) * 2 % 64 * 2 ^ ((pair_col_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 27648 + (unsigned int)((pair_col_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 3) * 2 % 64 * 2 ^ ((pair_col_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 27648 + (unsigned int)((pair_col_base + 4) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 4) * 2 % 64 * 2 ^ ((pair_col_base + 4) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 4) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 27648 + (unsigned int)((pair_col_base + 5) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 5) * 2 % 64 * 2 ^ ((pair_col_base + 5) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 5) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 27648 + (unsigned int)((pair_col_base + 6) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 6) * 2 % 64 * 2 ^ ((pair_col_base + 6) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 6) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 27648 + (unsigned int)((pair_col_base + 7) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 7) * 2 % 64 * 2 ^ ((pair_col_base + 7) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 7) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[7]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 5) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
