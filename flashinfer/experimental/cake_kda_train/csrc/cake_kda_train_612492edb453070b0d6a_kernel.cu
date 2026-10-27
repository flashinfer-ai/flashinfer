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
#define TMEM_NCOLS 128
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 64
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_W_PANEL_OFF 1024
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_KG_PANEL_OFF 33792
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_W_OP_OFF 1024
#define SMEM_W_OP_STAGE_BYTES 16384
#define SMEM_W_OP_STRIDE 16384
#define SMEM_KG_MN_OFF 33792
#define SMEM_KG_MN_STAGE_BYTES 16384
#define SMEM_KG_MN_STRIDE 16384
#define SMEM_HS_OFF 66560
#define SMEM_HS_STAGE_BYTES 16384
#define SMEM_HS_STRIDE 16384
#define SMEM_HS_MN_OFF 66560
#define SMEM_HS_MN_STAGE_BYTES 16384
#define SMEM_HS_MN_STRIDE 16384
#define SMEM_BV_OFF 82944
#define SMEM_BV_STAGE_BYTES 8192
#define SMEM_BV_STRIDE 8192
#define SMEM_BV_MN_OFF 82944
#define SMEM_BV_MN_STAGE_BYTES 8192
#define SMEM_BV_MN_STRIDE 8192
#define SMEM_TOTAL 91136
#define THREADS 320
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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


extern "C" {

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_612492edb453070b0d6a(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, const __grid_constant__ CUtensorMap h_out, const __grid_constant__ CUtensorMap v_new, int num_heads, int* __restrict__ seq_chunk_start, int* __restrict__ seq_order)
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
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define hs_ready_addr (mbar_base + 32)
    #define vp_ready_addr (mbar_base + 40)
    #define bv_ready_addr (mbar_base + 48)
    #define dec_done_addr (mbar_base + 56)
    #define h_ready_addr (mbar_base + 64)
    #define h_tape_read_addr (mbar_base + 72)
    #define bv_tape_read_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_panel_addr = smem + 1024;
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kg_panel_addr = smem + 33792;
    __nv_bfloat16* w_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_op_addr = smem + 1024;
    __nv_bfloat16* kg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kg_mn_addr = smem + 33792;
    __nv_bfloat16* hs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_addr = smem + 66560;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn_addr = smem + 66560;
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_addr = smem + 82944;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_mn_addr = smem + 82944;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[0..88)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'one' ---
            // hs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // bv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // dec_done: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            // h_ready: 1 barriers, init_count=2
            mbarrier_init(smem + 64, 2);
            // h_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // bv_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 88);
    if (warp == 0) {
        int _tmem_hold = smem + 88;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 64;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int half = blockIdx.x % 2;
            int head = blockIdx.x / 2 % num_heads;
            int rank = blockIdx.x / (2 * num_heads);
            int seq = seq_order[rank];
            int c0 = seq_chunk_start[seq];
            int nc = seq_chunk_start[seq + 1] - c0;
            int row0 = c0 * 64;
            int vbase = half * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            float zeros16[16];
            zeros16[0] = 0.0f;
            zeros16[1] = 0.0f;
            zeros16[2] = 0.0f;
            zeros16[3] = 0.0f;
            zeros16[4] = 0.0f;
            zeros16[5] = 0.0f;
            zeros16[6] = 0.0f;
            zeros16[7] = 0.0f;
            zeros16[8] = 0.0f;
            zeros16[9] = 0.0f;
            zeros16[10] = 0.0f;
            zeros16[11] = 0.0f;
            zeros16[12] = 0.0f;
            zeros16[13] = 0.0f;
            zeros16[14] = 0.0f;
            zeros16[15] = 0.0f;
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, zeros16);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int u_pk[16];
            u_pk[0] = 0;
            u_pk[1] = 0;
            u_pk[2] = 0;
            u_pk[3] = 0;
            u_pk[4] = 0;
            u_pk[5] = 0;
            u_pk[6] = 0;
            u_pk[7] = 0;
            u_pk[8] = 0;
            u_pk[9] = 0;
            u_pk[10] = 0;
            u_pk[11] = 0;
            u_pk[12] = 0;
            u_pk[13] = 0;
            u_pk[14] = 0;
            u_pk[15] = 0;
            if (nc > 0) {
                long long u_next = ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128 + (long long)vbase;
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    {
                        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(lane_half * 16 + q * 8));
                        u_pk[q * 4 + 0] = _uv4_0.x;
                        u_pk[q * 4 + 1] = _uv4_0.y;
                        u_pk[q * 4 + 2] = _uv4_0.z;
                        u_pk[q * 4 + 3] = _uv4_0.w;
                    }
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    {
                        uint4 _uv4_1 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(32 + lane_half * 16 + q_1 * 8));
                        u_pk[8 + q_1 * 4 + 0] = _uv4_1.x;
                        u_pk[8 + q_1 * 4 + 1] = _uv4_1.y;
                        u_pk[8 + q_1 * 4 + 2] = _uv4_1.z;
                        u_pk[8 + q_1 * 4 + 3] = _uv4_1.w;
                    }
                }
            }
            unsigned int _phase_bv_tape_read_0 = 1;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_tape_read_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q_2 * 8 / 64 * 16384 + k_row * 128 + q_2 * 8 % 64 * 2 ^ (q_2 * 8 / 64 * 16384 + k_row * 128 + q_2 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[q_2 * 4]), "r"(_tmem_load_4_bf16[q_2 * 4 + 1]), "r"(_tmem_load_4_bf16[q_2 * 4 + 2]), "r"(_tmem_load_4_bf16[q_2 * 4 + 3]) : "memory");
                }
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                    _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_3 * 8) / 64 * 16384 + k_row * 128 + (16 + q_3 * 8) % 64 * 2 ^ ((16 + q_3 * 8) / 64 * 16384 + k_row * 128 + (16 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[q_3 * 4]), "r"(_tmem_load_5_bf16[q_3 * 4 + 1]), "r"(_tmem_load_5_bf16[q_3 * 4 + 2]), "r"(_tmem_load_5_bf16[q_3 * 4 + 3]) : "memory");
                }
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_4 * 8) / 64 * 16384 + k_row * 128 + (32 + q_4 * 8) % 64 * 2 ^ ((32 + q_4 * 8) / 64 * 16384 + k_row * 128 + (32 + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[q_4 * 4]), "r"(_tmem_load_6_bf16[q_4 * 4 + 1]), "r"(_tmem_load_6_bf16[q_4 * 4 + 2]), "r"(_tmem_load_6_bf16[q_4 * 4 + 3]) : "memory");
                }
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_7_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                    _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_5 = 0; q_5 < 2; q_5++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_5 * 8) / 64 * 16384 + k_row * 128 + (48 + q_5 * 8) % 64 * 2 ^ ((48 + q_5 * 8) / 64 * 16384 + k_row * 128 + (48 + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[q_5 * 4]), "r"(_tmem_load_7_bf16[q_5 * 4 + 1]), "r"(_tmem_load_7_bf16[q_5 * 4 + 2]), "r"(_tmem_load_7_bf16[q_5 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(bv_tape_read_addr, _phase_bv_tape_read_0);
                _phase_bv_tape_read_0 ^= 1;
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long u_base = ((long long)trow * (long long)num_heads + (long long)head) * 128 + (long long)vbase;
                float _tmem_load_8[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col = lane_half * 16;
                float _unpack_bf16_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) & 0xffff0000u);
                }
                float bvv[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    bvv[j] = _unpack_bf16_0[j] - _tmem_load_8[j];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_6 = 0; q_6 < 2; q_6++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col + q_6 * 8) / 64 * 8192 + t_row * 128 + (col + q_6 * 8) % 64 * 2 ^ ((col + q_6 * 8) / 64 * 8192 + t_row * 128 + (col + q_6 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[q_6 * 4]), "r"(bvv_bf16[q_6 * 4 + 1]), "r"(bvv_bf16[q_6 * 4 + 2]), "r"(bvv_bf16[q_6 * 4 + 3]) : "memory");
                }
                float _tmem_load_9[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_0 = 32 + lane_half * 16;
                float _unpack_bf16_1[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) & 0xffff0000u);
                }
                float bvv_1[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    bvv_1[j_1] = _unpack_bf16_1[j_1] - _tmem_load_9[j_1];
                }
                unsigned int bvv_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1[_lp*2 + 0], bvv_1[_lp*2+1 + 0]));
                    bvv_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_7 = 0; q_7 < 2; q_7++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_0 + q_7 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_7 * 8) % 64 * 2 ^ ((col_0 + q_7 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_1_bf16[q_7 * 4]), "r"(bvv_1_bf16[q_7 * 4 + 1]), "r"(bvv_1_bf16[q_7 * 4 + 2]), "r"(bvv_1_bf16[q_7 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    long long u_next_1 = u_base + (long long)(64 * num_heads * 128);
                    #pragma unroll
                    for (int q_8 = 0; q_8 < 2; q_8++) {
                        {
                            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(lane_half * 16 + q_8 * 8));
                            u_pk[q_8 * 4 + 0] = _uv4_2.x;
                            u_pk[q_8 * 4 + 1] = _uv4_2.y;
                            u_pk[q_8 * 4 + 2] = _uv4_2.z;
                            u_pk[q_8 * 4 + 3] = _uv4_2.w;
                        }
                    }
                    #pragma unroll
                    for (int q_9 = 0; q_9 < 2; q_9++) {
                        {
                            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(32 + lane_half * 16 + q_9 * 8));
                            u_pk[8 + q_9 * 4 + 0] = _uv4_3.x;
                            u_pk[8 + q_9 * 4 + 1] = _uv4_3.y;
                            u_pk[8 + q_9 * 4 + 2] = _uv4_3.z;
                            u_pk[8 + q_9 * 4 + 3] = _uv4_3.w;
                        }
                    }
                }
                mbarrier_wait(h_tape_read_addr, _phase_h_tape_read_0);
                _phase_h_tape_read_0 ^= 1;
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
            }
        }
    }
    // ---- Role: decayw ----
    if (warp >= 4 && warp <= 7) {
        { // decayw_main
            int half_1 = blockIdx.x % 2;
            int head_1 = blockIdx.x / 2 % num_heads;
            int rank_1 = blockIdx.x / (2 * num_heads);
            int seq_1 = seq_order[rank_1];
            int c0_1 = seq_chunk_start[seq_1];
            int nc_1 = seq_chunk_start[seq_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int warp_in_wg_1 = warp % 4;
            int k_row_1 = warp_in_wg_1 * 32 + lane;
            int h_row_base_1 = warp_in_wg_1 * 32 << 16;
            float gk_pre = 0.0f;
            if (nc_1 > 0) {
                long long gn_next = ((long long)(row0_1 + 64 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                gk_pre = gk[gn_next];
            }
            unsigned int _phase_hs_ready_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float decay = approx_exp2(gk_pre);
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)h_row_base_1);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    hs2[j_2] = _tmem_load_0[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1, hs2);
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_0[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    hs2_0[j_3] = _tmem_load_1[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 16, hs2_0);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_1[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    hs2_1[j_4] = _tmem_load_2[j_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 32, hs2_1);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_2[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    hs2_2[j_5] = _tmem_load_3[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 48, hs2_2);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dec_done_addr);
                if (nc_1 > c_1 + 1) {
                    long long gn_next_1 = ((long long)(row0_1 + c_1 * 64 + 128 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                    gk_pre = gk[gn_next_1];
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int half_2 = blockIdx.x % 2;
            int head_2 = blockIdx.x / 2 % num_heads;
            int rank_2 = blockIdx.x / (2 * num_heads);
            int seq_2 = seq_order[rank_2];
            int c0_2 = seq_chunk_start[seq_2];
            int nc_2 = seq_chunk_start[seq_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            int vbase_1 = half_2 * 64;
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_hs_ready_0_1 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                if (nc_2 > 0) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                    int trow_1 = row0_2;
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&w_tma), seg * 64, head_2, trow_1, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&kg_tma), seg * 64, head_2, trow_1, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                }
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                        int trow_2 = row0_2 + (c_2 + 1) * 64;
                        #pragma unroll
                        for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                            tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)seg_1) * 8192, (&w_tma), seg_1 * 64, head_2, trow_2, tiles_full_addr + (stage) * 8);
                            tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg_1) * 8192, (&kg_tma), seg_1 * 64, head_2, trow_2, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    int chunk = (c0_2 + c_2) * num_heads + head_2;
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                    _phase_hs_ready_0_1 ^= 1;
                    tma_store_3d((&h_out), vbase_1, 0, chunk, hs_addr);
                    asm volatile("cp.async.bulk.commit_group;");
                    int trow_3 = row0_2 + c_2 * 64;
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    tma_store_3d((&v_new), vbase_1, head_2, trow_3, bv_addr);
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                    mbarrier_arrive(h_tape_read_addr);
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(bv_tape_read_addr);
                }
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int half_3 = blockIdx.x % 2;
            int head_3 = blockIdx.x / 2 % num_heads;
            int rank_3 = blockIdx.x / (2 * num_heads);
            int seq_3 = seq_order[rank_3];
            int c0_3 = seq_chunk_start[seq_3];
            int nc_3 = seq_chunk_start[seq_3 + 1] - c0_3;
            int row0_3 = c0_3 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hs_ready_0_2 = 0;
            unsigned int _phase_bv_ready_0_1 = 0;
            unsigned int _phase_dec_done_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_2);
                    _phase_hs_ready_0_2 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x7FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
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
                    "mov.b32 id, 68224144;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0_1);
                    _phase_bv_ready_0_1 ^= 1;
                    mbarrier_wait(dec_done_addr, _phase_dec_done_0);
                    _phase_dec_done_0 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(h_ready_addr);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_2 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
