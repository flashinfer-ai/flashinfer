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
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_MAIN_PIPE_STAGES 8
#define NUM_OUTPUT_PIPE_STAGES 1
#define SMEM_SMEM_ACT_OFF 1024
#define SMEM_SMEM_ACT_STAGE_BYTES 2048
#define SMEM_SMEM_ACT_STRIDE 13312
#define SMEM_SMEM_PACKED_OFF 3072
#define SMEM_SMEM_PACKED_STAGE_BYTES 2048
#define SMEM_SMEM_PACKED_STRIDE 13312
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 256
#define SMEM_SMEM_SCALE_STRIDE 13312
#define SMEM_SMEM_WEIGHT_OFF 6144
#define SMEM_SMEM_WEIGHT_STAGE_BYTES 8192
#define SMEM_SMEM_WEIGHT_STRIDE 13312
#define SMEM_TOTAL 107520
#define HAS_ALPHA 1
#define ENABLE_PDL 1
#define FLAT_GRID 0

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

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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
kernel_cake_blackwell_bf16_fp4_329b223fccd643796bdf(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
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
    #define act_done_addr (mbar_base + 64)
    #define packed_full_addr (mbar_base + 128)
    #define packed_done_addr (mbar_base + 192)
    #define weight_full_addr (mbar_base + 256)
    #define weight_done_addr (mbar_base + 320)
    #define output_full_addr (mbar_base + 384)
    #define tmem_release_addr (mbar_base + 392)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_act_addr = smem + 1024;
    int* smem_packed = reinterpret_cast<int*>(smem_raw + 3072);
    const int smem_packed_addr = smem + 3072;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    __nv_bfloat16* smem_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_weight_addr = smem + 6144;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 50 barriers)
    // Mbarriers at smem_raw[0..400)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // act_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // act_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // packed_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // packed_done: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // weight_full: 8 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // weight_done: 8 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'output_pipe' ---
            // output_full: 1 barriers, init_count=1
            mbarrier_init(smem + 384, 1);
            // tmem_release: 1 barriers, init_count=128
            mbarrier_init(smem + 392, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 400);
    if (warp == 0) {
        int _tmem_hold = smem + 400;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
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
            int off_m = tile_m * 16;
            int off_n = tile_n * 64;
            int epi_warp = warp % 4;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            float alpha_value = 1.0f;
            {
                {
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                }
                alpha_value = alpha[0];
            }
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
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_release_addr);
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
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
                    int _mma_a_lo_0 = (((smem_weight_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
                    int _mma_b_lo_0 = (((smem_act_addr) >> 4) & 0x3FFF) + (mma_stage) * 832;
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
                    tcgen05_commit(act_done_addr + (mma_stage) * 8);
                    tcgen05_commit(weight_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 8) { mma_stage = 0; _phase_act_full ^= 1; _phase_weight_full ^= 1; }
                }
                tcgen05_commit(output_full_addr);
            }
            unsigned int _phase_tmem_release_0 = 0;
            mbarrier_wait_hint(tmem_release_addr, _phase_tmem_release_0, 10000000);
            _phase_tmem_release_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(32));
        }
    }
    // ---- Role: load_act ----
    if (warp == 5) {
        { // load_act_main
            int tile_m_1 = blockIdx.y;
            int off_m_1 = tile_m_1 * 16;
            int k_begin = 0;
            int k_extent_1 = K;
            int k_tiles_1 = (k_extent_1 + 64 - 1) / 64;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int act_stage = 0;
            unsigned int _phase_act_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                    mbarrier_wait(act_done_addr + (act_stage) * 8, _phase_act_done);
                    tma_2d_gmem2smem(smem_act_addr + act_stage * 13312, (&A), k_begin + kt_1 * 64, off_m_1, act_full_addr + (act_stage) * 8);
                    mbarrier_arrive_expect_tx(act_full_addr + (act_stage) * 8, 2048);
                    act_stage += 1;
                    if (act_stage == 8) { act_stage = 0; _phase_act_done ^= 1; }
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
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int packed_stage = 0;
            unsigned int _phase_packed_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < k_tiles_2; kt_2++) {
                    mbarrier_wait(packed_done_addr + (packed_stage) * 8, _phase_packed_done);
                    tma_2d_gmem2smem(smem_packed_addr + packed_stage * 13312, (&B), off_n_1 * 2, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    tma_2d_gmem2smem(smem_scale_addr + packed_stage * 13312, (&B_descale), off_n_1, kt_2 * 4, packed_full_addr + (packed_stage) * 8);
                    mbarrier_arrive_expect_tx(packed_full_addr + (packed_stage) * 8, 2048 + ((1) ? 256 : 1024));
                    packed_stage += 1;
                    if (packed_stage == 8) { packed_stage = 0; _phase_packed_done ^= 1; }
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
                int packed_base = smem_packed_addr + convert_stage * 13312;
                int scale_base = smem_scale_addr + convert_stage * 13312;
                int weight_row = convert_tid / 4;
                int k_block = convert_tid - weight_row * 4;
                int pair_col_base = k_block * 8;
                unsigned int packed0 = 0;
                unsigned int packed1 = 0;
                uint8_t scale_byte = (uint8_t)0;
                int base_n = weight_row % 16;
                int n_warp = base_n / 8;
                int tc_col = base_n & 7;
                int u32_local = weight_row / 32;
                int row_half = weight_row % 32 / 16;
                int packed0_shift = row_half * 16;
                int packed1_shift = packed0_shift + 8;
                int lane_0 = tc_col * 4;
                int u32_pos = n_warp * 64 + lane_0 * 2 + u32_local;
                int word_linear = k_block * 128 + u32_pos;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[0])) : "r"(packed_base + word_linear * 4));
                packed0 = packed0 | raw_words[0] >> (unsigned int)packed0_shift & 255;
                packed1 = packed1 | raw_words[0] >> (unsigned int)packed1_shift & 255;
                int lane_1 = tc_col * 4 + 1;
                int u32_pos_2 = n_warp * 64 + lane_1 * 2 + u32_local;
                int word_linear_3 = k_block * 128 + u32_pos_2;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[0])) : "r"(packed_base + word_linear_3 * 4));
                packed0 = packed0 | (raw_words[0] >> (unsigned int)packed0_shift & 255) << 8;
                packed1 = packed1 | (raw_words[0] >> (unsigned int)packed1_shift & 255) << 8;
                int lane_4 = tc_col * 4 + 2;
                int u32_pos_5 = n_warp * 64 + lane_4 * 2 + u32_local;
                int word_linear_6 = k_block * 128 + u32_pos_5;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[0])) : "r"(packed_base + word_linear_6 * 4));
                packed0 = packed0 | (raw_words[0] >> (unsigned int)packed0_shift & 255) << 16;
                packed1 = packed1 | (raw_words[0] >> (unsigned int)packed1_shift & 255) << 16;
                int lane_7 = tc_col * 4 + 3;
                int u32_pos_8 = n_warp * 64 + lane_7 * 2 + u32_local;
                int word_linear_9 = k_block * 128 + u32_pos_8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[0])) : "r"(packed_base + word_linear_9 * 4));
                packed0 = packed0 | (raw_words[0] >> (unsigned int)packed0_shift & 255) << 24;
                packed1 = packed1 | (raw_words[0] >> (unsigned int)packed1_shift & 255) << 24;
                int scale_linear = k_block * 64 + weight_row;
                int scale_aligned = scale_linear / 4 * 4;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(scale_base + scale_aligned));
                int scale_shift = (scale_linear & 3) * 8;
                scale_byte = (uint8_t)(scale_word[0] >> (unsigned int)scale_shift & 255);
                uint32_t _fp4_dequant_block16_bf16_0[8];
                {
                    uint32_t _scale_byte = ((uint32_t)(scale_byte)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_bf16x2;
                    uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
                    uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
                    float _scale_f0;
                    float _scale_f1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_bf16x2) : "f"(_scale_f1), "f"(_scale_f0));
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
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)(pair_col_base * 2 / 64 * 8192 + weight_row * 128 + pair_col_base * 2 % 64 * 2 ^ (pair_col_base * 2 / 64 * 8192 + weight_row * 128 + pair_col_base * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[0]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_col_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 1) * 2 % 64 * 2 ^ ((pair_col_base + 1) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 1) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[1]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_col_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 2) * 2 % 64 * 2 ^ ((pair_col_base + 2) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 2) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[2]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_col_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 3) * 2 % 64 * 2 ^ ((pair_col_base + 3) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 3) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[3]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_col_base + 4) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 4) * 2 % 64 * 2 ^ ((pair_col_base + 4) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 4) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[4]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_col_base + 5) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 5) * 2 % 64 * 2 ^ ((pair_col_base + 5) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 5) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[5]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_col_base + 6) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 6) * 2 % 64 * 2 ^ ((pair_col_base + 6) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 6) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[6]) : "memory");
                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_weight_addr + convert_stage * 13312 + (unsigned int)((pair_col_base + 7) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 7) * 2 % 64 * 2 ^ ((pair_col_base + 7) * 2 / 64 * 8192 + weight_row * 128 + (pair_col_base + 7) * 2 % 64 * 2 >> 7 & 7) << 4))), "r"(_fp4_dequant_block16_bf16_0[7]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(packed_done_addr + (convert_stage) * 8);
                        mbarrier_arrive(weight_full_addr + (convert_stage) * 8);
                    }
                }
                convert_stage += 1;
                if (convert_stage == 8) { convert_stage = 0; _phase_packed_full ^= 1; _phase_weight_done ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
