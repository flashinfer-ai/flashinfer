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
#define TMEM_NCOLS 384
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 128
#define TMEM_ACC_DX_OFFSET 256
#define NUM_RING_PIPE_STAGES 2
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
#define SMEM_HS0_OFF 66560
#define SMEM_HS0_STAGE_BYTES 32768
#define SMEM_HS0_STRIDE 32768
#define SMEM_HS1_OFF 99328
#define SMEM_HS1_STAGE_BYTES 32768
#define SMEM_HS1_STRIDE 32768
#define SMEM_HS2_OFF 132096
#define SMEM_HS2_STAGE_BYTES 32768
#define SMEM_HS2_STRIDE 32768
#define SMEM_HS_MN0_OFF 66560
#define SMEM_HS_MN0_STAGE_BYTES 32768
#define SMEM_HS_MN0_STRIDE 32768
#define SMEM_HS_MN1_OFF 99328
#define SMEM_HS_MN1_STAGE_BYTES 32768
#define SMEM_HS_MN1_STRIDE 32768
#define SMEM_HS_MN2_OFF 132096
#define SMEM_HS_MN2_STAGE_BYTES 32768
#define SMEM_HS_MN2_STRIDE 32768
#define SMEM_BV0_OFF 164864
#define SMEM_BV0_STAGE_BYTES 16384
#define SMEM_BV0_STRIDE 16384
#define SMEM_BV1_OFF 181248
#define SMEM_BV1_STAGE_BYTES 16384
#define SMEM_BV1_STRIDE 16384
#define SMEM_BV2_OFF 197632
#define SMEM_BV2_STAGE_BYTES 16384
#define SMEM_BV2_STRIDE 16384
#define SMEM_BV_MN0_OFF 164864
#define SMEM_BV_MN0_STAGE_BYTES 16384
#define SMEM_BV_MN0_STRIDE 16384
#define SMEM_BV_MN1_OFF 181248
#define SMEM_BV_MN1_STAGE_BYTES 16384
#define SMEM_BV_MN1_STRIDE 16384
#define SMEM_BV_MN2_OFF 197632
#define SMEM_BV_MN2_STAGE_BYTES 16384
#define SMEM_BV_MN2_STRIDE 16384
#define SMEM_TOTAL 214016
#define THREADS 192
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_b1a95a8ee8110e47558e_kcp(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, int init_mode, int use_u, int write_transposed)
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
    #define h_ready_addr (mbar_base + 56)

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
    __nv_bfloat16* hs0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs0_addr = smem + 66560;
    __nv_bfloat16* hs1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int hs1_addr = smem + 99328;
    __nv_bfloat16* hs2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int hs2_addr = smem + 132096;
    __nv_bfloat16* hs_mn0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn0_addr = smem + 66560;
    __nv_bfloat16* hs_mn1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int hs_mn1_addr = smem + 99328;
    __nv_bfloat16* hs_mn2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int hs_mn2_addr = smem + 132096;
    __nv_bfloat16* bv0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int bv0_addr = smem + 164864;
    __nv_bfloat16* bv1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int bv1_addr = smem + 181248;
    __nv_bfloat16* bv2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int bv2_addr = smem + 197632;
    __nv_bfloat16* bv_mn0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int bv_mn0_addr = smem + 164864;
    __nv_bfloat16* bv_mn1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int bv_mn1_addr = smem + 181248;
    __nv_bfloat16* bv_mn2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int bv_mn2_addr = smem + 197632;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring_pipe' ---
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
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 128;
    const int tmem_acc_dx = taddr + 256;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int vslice = 0;
            int head = blockIdx.x % num_heads;
            int seg = walk_seg[blockIdx.x / num_heads];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            int v0 = vslice * 128;
            unsigned int mask_hi = 4294901760u;
            float init_row[16];
            #pragma unroll
            for (int j_1_0 = 0; j_1_0 < 16; j_1_0++) {
                float one_here_1_0 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + j_1_0 == k_row) {
                        one_here_1_0 = 1.0f;
                    }
                }
                init_row[j_1_0] = one_here_1_0;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
            #pragma unroll
            for (int j_1_1 = 0; j_1_1 < 16; j_1_1++) {
                float one_here_1_1 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 16 + j_1_1 == k_row) {
                        one_here_1_1 = 1.0f;
                    }
                }
                init_row[j_1_1] = one_here_1_1;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
            #pragma unroll
            for (int j_1_2 = 0; j_1_2 < 16; j_1_2++) {
                float one_here_1_2 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 32 + j_1_2 == k_row) {
                        one_here_1_2 = 1.0f;
                    }
                }
                init_row[j_1_2] = one_here_1_2;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
            #pragma unroll
            for (int j_1_3 = 0; j_1_3 < 16; j_1_3++) {
                float one_here_1_3 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 48 + j_1_3 == k_row) {
                        one_here_1_3 = 1.0f;
                    }
                }
                init_row[j_1_3] = one_here_1_3;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
            #pragma unroll
            for (int j_1_4 = 0; j_1_4 < 16; j_1_4++) {
                float one_here_1_4 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 64 + j_1_4 == k_row) {
                        one_here_1_4 = 1.0f;
                    }
                }
                init_row[j_1_4] = one_here_1_4;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
            #pragma unroll
            for (int j_1_5 = 0; j_1_5 < 16; j_1_5++) {
                float one_here_1_5 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 80 + j_1_5 == k_row) {
                        one_here_1_5 = 1.0f;
                    }
                }
                init_row[j_1_5] = one_here_1_5;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
            #pragma unroll
            for (int j_1_6 = 0; j_1_6 < 16; j_1_6++) {
                float one_here_1_6 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 96 + j_1_6 == k_row) {
                        one_here_1_6 = 1.0f;
                    }
                }
                init_row[j_1_6] = one_here_1_6;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
            #pragma unroll
            for (int j_1_7 = 0; j_1_7 < 16; j_1_7++) {
                float one_here_1_7 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 112 + j_1_7 == k_row) {
                        one_here_1_7 = 1.0f;
                    }
                }
                init_row[j_1_7] = one_here_1_7;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
            float zeros[16];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            zeros[8] = 0.0f;
            zeros[9] = 0.0f;
            zeros[10] = 0.0f;
            zeros[11] = 0.0f;
            zeros[12] = 0.0f;
            zeros[13] = 0.0f;
            zeros[14] = 0.0f;
            zeros[15] = 0.0f;
            tmem_st_x16_f32(taddr + 256 + (unsigned int)h_row_base, zeros);
            tmem_st_x16_f32(taddr + 256 + (unsigned int)h_row_base + 16, zeros);
            tmem_st_x16_f32(taddr + 256 + (unsigned int)h_row_base + 32, zeros);
            tmem_st_x16_f32(taddr + 256 + (unsigned int)h_row_base + 48, zeros);
            tmem_st_x16_f32(taddr + 256 + (unsigned int)h_row_base + 64, zeros);
            tmem_st_x16_f32(taddr + 256 + (unsigned int)h_row_base + 80, zeros);
            tmem_st_x16_f32(taddr + 256 + (unsigned int)h_row_base + 96, zeros);
            tmem_st_x16_f32(taddr + 256 + (unsigned int)h_row_base + 112, zeros);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                long long gn_index = ((long long)(row0 + c * 64 + 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                float decay = approx_exp2(gk[gn_index]);
                float _tmem_load_0_0[16];
                tmem_ld_x16(&_tmem_load_0_0[0], taddr + (unsigned int)h_row_base);
                float _tmem_load_1_0[16];
                tmem_ld_x16(&_tmem_load_1_0[0], taddr + 256 + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_0 = 0; j_0 < 16; j_0++) {
                    _tmem_load_0_0[j_0] = _tmem_load_0_0[j_0] + _tmem_load_1_0[j_0];
                }
                unsigned int tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0_0[_lp*2 + 0], _tmem_load_0_0[_lp*2+1 + 0]));
                    tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1[16];
                r1[0] = _tmem_load_0_0[0] - __uint_as_float(tmem_load_0_bf16[0] << 16);
                r1[1] = _tmem_load_0_0[1] - __uint_as_float(tmem_load_0_bf16[0] & mask_hi);
                r1[2] = _tmem_load_0_0[2] - __uint_as_float(tmem_load_0_bf16[1] << 16);
                r1[3] = _tmem_load_0_0[3] - __uint_as_float(tmem_load_0_bf16[1] & mask_hi);
                r1[4] = _tmem_load_0_0[4] - __uint_as_float(tmem_load_0_bf16[2] << 16);
                r1[5] = _tmem_load_0_0[5] - __uint_as_float(tmem_load_0_bf16[2] & mask_hi);
                r1[6] = _tmem_load_0_0[6] - __uint_as_float(tmem_load_0_bf16[3] << 16);
                r1[7] = _tmem_load_0_0[7] - __uint_as_float(tmem_load_0_bf16[3] & mask_hi);
                r1[8] = _tmem_load_0_0[8] - __uint_as_float(tmem_load_0_bf16[4] << 16);
                r1[9] = _tmem_load_0_0[9] - __uint_as_float(tmem_load_0_bf16[4] & mask_hi);
                r1[10] = _tmem_load_0_0[10] - __uint_as_float(tmem_load_0_bf16[5] << 16);
                r1[11] = _tmem_load_0_0[11] - __uint_as_float(tmem_load_0_bf16[5] & mask_hi);
                r1[12] = _tmem_load_0_0[12] - __uint_as_float(tmem_load_0_bf16[6] << 16);
                r1[13] = _tmem_load_0_0[13] - __uint_as_float(tmem_load_0_bf16[6] & mask_hi);
                r1[14] = _tmem_load_0_0[14] - __uint_as_float(tmem_load_0_bf16[7] << 16);
                r1[15] = _tmem_load_0_0[15] - __uint_as_float(tmem_load_0_bf16[7] & mask_hi);
                unsigned int r1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1[_lp*2 + 0], r1[_lp*2+1 + 0]));
                    r1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2[16];
                r2[0] = r1[0] - __uint_as_float(r1_bf16[0] << 16);
                r2[1] = r1[1] - __uint_as_float(r1_bf16[0] & mask_hi);
                r2[2] = r1[2] - __uint_as_float(r1_bf16[1] << 16);
                r2[3] = r1[3] - __uint_as_float(r1_bf16[1] & mask_hi);
                r2[4] = r1[4] - __uint_as_float(r1_bf16[2] << 16);
                r2[5] = r1[5] - __uint_as_float(r1_bf16[2] & mask_hi);
                r2[6] = r1[6] - __uint_as_float(r1_bf16[3] << 16);
                r2[7] = r1[7] - __uint_as_float(r1_bf16[3] & mask_hi);
                r2[8] = r1[8] - __uint_as_float(r1_bf16[4] << 16);
                r2[9] = r1[9] - __uint_as_float(r1_bf16[4] & mask_hi);
                r2[10] = r1[10] - __uint_as_float(r1_bf16[5] << 16);
                r2[11] = r1[11] - __uint_as_float(r1_bf16[5] & mask_hi);
                r2[12] = r1[12] - __uint_as_float(r1_bf16[6] << 16);
                r2[13] = r1[13] - __uint_as_float(r1_bf16[6] & mask_hi);
                r2[14] = r1[14] - __uint_as_float(r1_bf16[7] << 16);
                r2[15] = r1[15] - __uint_as_float(r1_bf16[7] & mask_hi);
                unsigned int r2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2[_lp*2 + 0], r2[_lp*2+1 + 0]));
                    r2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16[0]), "r"(tmem_load_0_bf16[1]), "r"(tmem_load_0_bf16[2]), "r"(tmem_load_0_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16[4]), "r"(tmem_load_0_bf16[5]), "r"(tmem_load_0_bf16[6]), "r"(tmem_load_0_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(r1_bf16[0]), "r"(r1_bf16[1]), "r"(r1_bf16[2]), "r"(r1_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r1_bf16[4]), "r"(r1_bf16[5]), "r"(r1_bf16[6]), "r"(r1_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(r2_bf16[0]), "r"(r2_bf16[1]), "r"(r2_bf16[2]), "r"(r2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r2_bf16[4]), "r"(r2_bf16[5]), "r"(r2_bf16[6]), "r"(r2_bf16[7]) : "memory");
                float hs2_0_0[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    hs2_0_0[j_2] = _tmem_load_0_0[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, hs2_0_0);
                float _tmem_load_0_1[16];
                tmem_ld_x16(&_tmem_load_0_1[0], taddr + (unsigned int)h_row_base + 16);
                float _tmem_load_1_1[16];
                tmem_ld_x16(&_tmem_load_1_1[0], taddr + 256 + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    _tmem_load_0_1[j_3] = _tmem_load_0_1[j_3] + _tmem_load_1_1[j_3];
                }
                unsigned int tmem_load_0_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0_1[_lp*2 + 0], _tmem_load_0_1[_lp*2+1 + 0]));
                    tmem_load_0_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_0[16];
                r1_0[0] = _tmem_load_0_1[0] - __uint_as_float(tmem_load_0_bf16_0[0] << 16);
                r1_0[1] = _tmem_load_0_1[1] - __uint_as_float(tmem_load_0_bf16_0[0] & mask_hi);
                r1_0[2] = _tmem_load_0_1[2] - __uint_as_float(tmem_load_0_bf16_0[1] << 16);
                r1_0[3] = _tmem_load_0_1[3] - __uint_as_float(tmem_load_0_bf16_0[1] & mask_hi);
                r1_0[4] = _tmem_load_0_1[4] - __uint_as_float(tmem_load_0_bf16_0[2] << 16);
                r1_0[5] = _tmem_load_0_1[5] - __uint_as_float(tmem_load_0_bf16_0[2] & mask_hi);
                r1_0[6] = _tmem_load_0_1[6] - __uint_as_float(tmem_load_0_bf16_0[3] << 16);
                r1_0[7] = _tmem_load_0_1[7] - __uint_as_float(tmem_load_0_bf16_0[3] & mask_hi);
                r1_0[8] = _tmem_load_0_1[8] - __uint_as_float(tmem_load_0_bf16_0[4] << 16);
                r1_0[9] = _tmem_load_0_1[9] - __uint_as_float(tmem_load_0_bf16_0[4] & mask_hi);
                r1_0[10] = _tmem_load_0_1[10] - __uint_as_float(tmem_load_0_bf16_0[5] << 16);
                r1_0[11] = _tmem_load_0_1[11] - __uint_as_float(tmem_load_0_bf16_0[5] & mask_hi);
                r1_0[12] = _tmem_load_0_1[12] - __uint_as_float(tmem_load_0_bf16_0[6] << 16);
                r1_0[13] = _tmem_load_0_1[13] - __uint_as_float(tmem_load_0_bf16_0[6] & mask_hi);
                r1_0[14] = _tmem_load_0_1[14] - __uint_as_float(tmem_load_0_bf16_0[7] << 16);
                r1_0[15] = _tmem_load_0_1[15] - __uint_as_float(tmem_load_0_bf16_0[7] & mask_hi);
                unsigned int r1_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_0[_lp*2 + 0], r1_0[_lp*2+1 + 0]));
                    r1_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_0[16];
                r2_0[0] = r1_0[0] - __uint_as_float(r1_bf16_0[0] << 16);
                r2_0[1] = r1_0[1] - __uint_as_float(r1_bf16_0[0] & mask_hi);
                r2_0[2] = r1_0[2] - __uint_as_float(r1_bf16_0[1] << 16);
                r2_0[3] = r1_0[3] - __uint_as_float(r1_bf16_0[1] & mask_hi);
                r2_0[4] = r1_0[4] - __uint_as_float(r1_bf16_0[2] << 16);
                r2_0[5] = r1_0[5] - __uint_as_float(r1_bf16_0[2] & mask_hi);
                r2_0[6] = r1_0[6] - __uint_as_float(r1_bf16_0[3] << 16);
                r2_0[7] = r1_0[7] - __uint_as_float(r1_bf16_0[3] & mask_hi);
                r2_0[8] = r1_0[8] - __uint_as_float(r1_bf16_0[4] << 16);
                r2_0[9] = r1_0[9] - __uint_as_float(r1_bf16_0[4] & mask_hi);
                r2_0[10] = r1_0[10] - __uint_as_float(r1_bf16_0[5] << 16);
                r2_0[11] = r1_0[11] - __uint_as_float(r1_bf16_0[5] & mask_hi);
                r2_0[12] = r1_0[12] - __uint_as_float(r1_bf16_0[6] << 16);
                r2_0[13] = r1_0[13] - __uint_as_float(r1_bf16_0[6] & mask_hi);
                r2_0[14] = r1_0[14] - __uint_as_float(r1_bf16_0[7] << 16);
                r2_0[15] = r1_0[15] - __uint_as_float(r1_bf16_0[7] & mask_hi);
                unsigned int r2_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_0[_lp*2 + 0], r2_0[_lp*2+1 + 0]));
                    r2_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_0[0]), "r"(tmem_load_0_bf16_0[1]), "r"(tmem_load_0_bf16_0[2]), "r"(tmem_load_0_bf16_0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_0[4]), "r"(tmem_load_0_bf16_0[5]), "r"(tmem_load_0_bf16_0[6]), "r"(tmem_load_0_bf16_0[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r1_bf16_0[0]), "r"(r1_bf16_0[1]), "r"(r1_bf16_0[2]), "r"(r1_bf16_0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r1_bf16_0[4]), "r"(r1_bf16_0[5]), "r"(r1_bf16_0[6]), "r"(r1_bf16_0[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r2_bf16_0[0]), "r"(r2_bf16_0[1]), "r"(r2_bf16_0[2]), "r"(r2_bf16_0[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r2_bf16_0[4]), "r"(r2_bf16_0[5]), "r"(r2_bf16_0[6]), "r"(r2_bf16_0[7]) : "memory");
                float hs2_0_1[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    hs2_0_1[j_4] = _tmem_load_0_1[j_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, hs2_0_1);
                float _tmem_load_0_2[16];
                tmem_ld_x16(&_tmem_load_0_2[0], taddr + (unsigned int)h_row_base + 32);
                float _tmem_load_1_2[16];
                tmem_ld_x16(&_tmem_load_1_2[0], taddr + 256 + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    _tmem_load_0_2[j_5] = _tmem_load_0_2[j_5] + _tmem_load_1_2[j_5];
                }
                unsigned int tmem_load_0_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0_2[_lp*2 + 0], _tmem_load_0_2[_lp*2+1 + 0]));
                    tmem_load_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_1[16];
                r1_1[0] = _tmem_load_0_2[0] - __uint_as_float(tmem_load_0_bf16_1[0] << 16);
                r1_1[1] = _tmem_load_0_2[1] - __uint_as_float(tmem_load_0_bf16_1[0] & mask_hi);
                r1_1[2] = _tmem_load_0_2[2] - __uint_as_float(tmem_load_0_bf16_1[1] << 16);
                r1_1[3] = _tmem_load_0_2[3] - __uint_as_float(tmem_load_0_bf16_1[1] & mask_hi);
                r1_1[4] = _tmem_load_0_2[4] - __uint_as_float(tmem_load_0_bf16_1[2] << 16);
                r1_1[5] = _tmem_load_0_2[5] - __uint_as_float(tmem_load_0_bf16_1[2] & mask_hi);
                r1_1[6] = _tmem_load_0_2[6] - __uint_as_float(tmem_load_0_bf16_1[3] << 16);
                r1_1[7] = _tmem_load_0_2[7] - __uint_as_float(tmem_load_0_bf16_1[3] & mask_hi);
                r1_1[8] = _tmem_load_0_2[8] - __uint_as_float(tmem_load_0_bf16_1[4] << 16);
                r1_1[9] = _tmem_load_0_2[9] - __uint_as_float(tmem_load_0_bf16_1[4] & mask_hi);
                r1_1[10] = _tmem_load_0_2[10] - __uint_as_float(tmem_load_0_bf16_1[5] << 16);
                r1_1[11] = _tmem_load_0_2[11] - __uint_as_float(tmem_load_0_bf16_1[5] & mask_hi);
                r1_1[12] = _tmem_load_0_2[12] - __uint_as_float(tmem_load_0_bf16_1[6] << 16);
                r1_1[13] = _tmem_load_0_2[13] - __uint_as_float(tmem_load_0_bf16_1[6] & mask_hi);
                r1_1[14] = _tmem_load_0_2[14] - __uint_as_float(tmem_load_0_bf16_1[7] << 16);
                r1_1[15] = _tmem_load_0_2[15] - __uint_as_float(tmem_load_0_bf16_1[7] & mask_hi);
                unsigned int r1_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_1[_lp*2 + 0], r1_1[_lp*2+1 + 0]));
                    r1_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_1[16];
                r2_1[0] = r1_1[0] - __uint_as_float(r1_bf16_2[0] << 16);
                r2_1[1] = r1_1[1] - __uint_as_float(r1_bf16_2[0] & mask_hi);
                r2_1[2] = r1_1[2] - __uint_as_float(r1_bf16_2[1] << 16);
                r2_1[3] = r1_1[3] - __uint_as_float(r1_bf16_2[1] & mask_hi);
                r2_1[4] = r1_1[4] - __uint_as_float(r1_bf16_2[2] << 16);
                r2_1[5] = r1_1[5] - __uint_as_float(r1_bf16_2[2] & mask_hi);
                r2_1[6] = r1_1[6] - __uint_as_float(r1_bf16_2[3] << 16);
                r2_1[7] = r1_1[7] - __uint_as_float(r1_bf16_2[3] & mask_hi);
                r2_1[8] = r1_1[8] - __uint_as_float(r1_bf16_2[4] << 16);
                r2_1[9] = r1_1[9] - __uint_as_float(r1_bf16_2[4] & mask_hi);
                r2_1[10] = r1_1[10] - __uint_as_float(r1_bf16_2[5] << 16);
                r2_1[11] = r1_1[11] - __uint_as_float(r1_bf16_2[5] & mask_hi);
                r2_1[12] = r1_1[12] - __uint_as_float(r1_bf16_2[6] << 16);
                r2_1[13] = r1_1[13] - __uint_as_float(r1_bf16_2[6] & mask_hi);
                r2_1[14] = r1_1[14] - __uint_as_float(r1_bf16_2[7] << 16);
                r2_1[15] = r1_1[15] - __uint_as_float(r1_bf16_2[7] & mask_hi);
                unsigned int r2_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_1[_lp*2 + 0], r2_1[_lp*2+1 + 0]));
                    r2_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_1[0]), "r"(tmem_load_0_bf16_1[1]), "r"(tmem_load_0_bf16_1[2]), "r"(tmem_load_0_bf16_1[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_1[4]), "r"(tmem_load_0_bf16_1[5]), "r"(tmem_load_0_bf16_1[6]), "r"(tmem_load_0_bf16_1[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r1_bf16_2[0]), "r"(r1_bf16_2[1]), "r"(r1_bf16_2[2]), "r"(r1_bf16_2[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r1_bf16_2[4]), "r"(r1_bf16_2[5]), "r"(r1_bf16_2[6]), "r"(r1_bf16_2[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r2_bf16_2[0]), "r"(r2_bf16_2[1]), "r"(r2_bf16_2[2]), "r"(r2_bf16_2[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r2_bf16_2[4]), "r"(r2_bf16_2[5]), "r"(r2_bf16_2[6]), "r"(r2_bf16_2[7]) : "memory");
                float hs2_0_2[16];
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    hs2_0_2[j_6] = _tmem_load_0_2[j_6] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, hs2_0_2);
                float _tmem_load_0_3[16];
                tmem_ld_x16(&_tmem_load_0_3[0], taddr + (unsigned int)h_row_base + 48);
                float _tmem_load_1_3[16];
                tmem_ld_x16(&_tmem_load_1_3[0], taddr + 256 + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    _tmem_load_0_3[j_7] = _tmem_load_0_3[j_7] + _tmem_load_1_3[j_7];
                }
                unsigned int tmem_load_0_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0_3[_lp*2 + 0], _tmem_load_0_3[_lp*2+1 + 0]));
                    tmem_load_0_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_2[16];
                r1_2[0] = _tmem_load_0_3[0] - __uint_as_float(tmem_load_0_bf16_2[0] << 16);
                r1_2[1] = _tmem_load_0_3[1] - __uint_as_float(tmem_load_0_bf16_2[0] & mask_hi);
                r1_2[2] = _tmem_load_0_3[2] - __uint_as_float(tmem_load_0_bf16_2[1] << 16);
                r1_2[3] = _tmem_load_0_3[3] - __uint_as_float(tmem_load_0_bf16_2[1] & mask_hi);
                r1_2[4] = _tmem_load_0_3[4] - __uint_as_float(tmem_load_0_bf16_2[2] << 16);
                r1_2[5] = _tmem_load_0_3[5] - __uint_as_float(tmem_load_0_bf16_2[2] & mask_hi);
                r1_2[6] = _tmem_load_0_3[6] - __uint_as_float(tmem_load_0_bf16_2[3] << 16);
                r1_2[7] = _tmem_load_0_3[7] - __uint_as_float(tmem_load_0_bf16_2[3] & mask_hi);
                r1_2[8] = _tmem_load_0_3[8] - __uint_as_float(tmem_load_0_bf16_2[4] << 16);
                r1_2[9] = _tmem_load_0_3[9] - __uint_as_float(tmem_load_0_bf16_2[4] & mask_hi);
                r1_2[10] = _tmem_load_0_3[10] - __uint_as_float(tmem_load_0_bf16_2[5] << 16);
                r1_2[11] = _tmem_load_0_3[11] - __uint_as_float(tmem_load_0_bf16_2[5] & mask_hi);
                r1_2[12] = _tmem_load_0_3[12] - __uint_as_float(tmem_load_0_bf16_2[6] << 16);
                r1_2[13] = _tmem_load_0_3[13] - __uint_as_float(tmem_load_0_bf16_2[6] & mask_hi);
                r1_2[14] = _tmem_load_0_3[14] - __uint_as_float(tmem_load_0_bf16_2[7] << 16);
                r1_2[15] = _tmem_load_0_3[15] - __uint_as_float(tmem_load_0_bf16_2[7] & mask_hi);
                unsigned int r1_bf16_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_2[_lp*2 + 0], r1_2[_lp*2+1 + 0]));
                    r1_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_2[16];
                r2_2[0] = r1_2[0] - __uint_as_float(r1_bf16_3[0] << 16);
                r2_2[1] = r1_2[1] - __uint_as_float(r1_bf16_3[0] & mask_hi);
                r2_2[2] = r1_2[2] - __uint_as_float(r1_bf16_3[1] << 16);
                r2_2[3] = r1_2[3] - __uint_as_float(r1_bf16_3[1] & mask_hi);
                r2_2[4] = r1_2[4] - __uint_as_float(r1_bf16_3[2] << 16);
                r2_2[5] = r1_2[5] - __uint_as_float(r1_bf16_3[2] & mask_hi);
                r2_2[6] = r1_2[6] - __uint_as_float(r1_bf16_3[3] << 16);
                r2_2[7] = r1_2[7] - __uint_as_float(r1_bf16_3[3] & mask_hi);
                r2_2[8] = r1_2[8] - __uint_as_float(r1_bf16_3[4] << 16);
                r2_2[9] = r1_2[9] - __uint_as_float(r1_bf16_3[4] & mask_hi);
                r2_2[10] = r1_2[10] - __uint_as_float(r1_bf16_3[5] << 16);
                r2_2[11] = r1_2[11] - __uint_as_float(r1_bf16_3[5] & mask_hi);
                r2_2[12] = r1_2[12] - __uint_as_float(r1_bf16_3[6] << 16);
                r2_2[13] = r1_2[13] - __uint_as_float(r1_bf16_3[6] & mask_hi);
                r2_2[14] = r1_2[14] - __uint_as_float(r1_bf16_3[7] << 16);
                r2_2[15] = r1_2[15] - __uint_as_float(r1_bf16_3[7] & mask_hi);
                unsigned int r2_bf16_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_2[_lp*2 + 0], r2_2[_lp*2+1 + 0]));
                    r2_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_2[0]), "r"(tmem_load_0_bf16_2[1]), "r"(tmem_load_0_bf16_2[2]), "r"(tmem_load_0_bf16_2[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_2[4]), "r"(tmem_load_0_bf16_2[5]), "r"(tmem_load_0_bf16_2[6]), "r"(tmem_load_0_bf16_2[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r1_bf16_3[0]), "r"(r1_bf16_3[1]), "r"(r1_bf16_3[2]), "r"(r1_bf16_3[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r1_bf16_3[4]), "r"(r1_bf16_3[5]), "r"(r1_bf16_3[6]), "r"(r1_bf16_3[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r2_bf16_3[0]), "r"(r2_bf16_3[1]), "r"(r2_bf16_3[2]), "r"(r2_bf16_3[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r2_bf16_3[4]), "r"(r2_bf16_3[5]), "r"(r2_bf16_3[6]), "r"(r2_bf16_3[7]) : "memory");
                float hs2_0_3[16];
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    hs2_0_3[j_8] = _tmem_load_0_3[j_8] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, hs2_0_3);
                float _tmem_load_0_4[16];
                tmem_ld_x16(&_tmem_load_0_4[0], taddr + (unsigned int)h_row_base + 64);
                float _tmem_load_1_4[16];
                tmem_ld_x16(&_tmem_load_1_4[0], taddr + 256 + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    _tmem_load_0_4[j_9] = _tmem_load_0_4[j_9] + _tmem_load_1_4[j_9];
                }
                unsigned int tmem_load_0_bf16_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0_4[_lp*2 + 0], _tmem_load_0_4[_lp*2+1 + 0]));
                    tmem_load_0_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_3[16];
                r1_3[0] = _tmem_load_0_4[0] - __uint_as_float(tmem_load_0_bf16_3[0] << 16);
                r1_3[1] = _tmem_load_0_4[1] - __uint_as_float(tmem_load_0_bf16_3[0] & mask_hi);
                r1_3[2] = _tmem_load_0_4[2] - __uint_as_float(tmem_load_0_bf16_3[1] << 16);
                r1_3[3] = _tmem_load_0_4[3] - __uint_as_float(tmem_load_0_bf16_3[1] & mask_hi);
                r1_3[4] = _tmem_load_0_4[4] - __uint_as_float(tmem_load_0_bf16_3[2] << 16);
                r1_3[5] = _tmem_load_0_4[5] - __uint_as_float(tmem_load_0_bf16_3[2] & mask_hi);
                r1_3[6] = _tmem_load_0_4[6] - __uint_as_float(tmem_load_0_bf16_3[3] << 16);
                r1_3[7] = _tmem_load_0_4[7] - __uint_as_float(tmem_load_0_bf16_3[3] & mask_hi);
                r1_3[8] = _tmem_load_0_4[8] - __uint_as_float(tmem_load_0_bf16_3[4] << 16);
                r1_3[9] = _tmem_load_0_4[9] - __uint_as_float(tmem_load_0_bf16_3[4] & mask_hi);
                r1_3[10] = _tmem_load_0_4[10] - __uint_as_float(tmem_load_0_bf16_3[5] << 16);
                r1_3[11] = _tmem_load_0_4[11] - __uint_as_float(tmem_load_0_bf16_3[5] & mask_hi);
                r1_3[12] = _tmem_load_0_4[12] - __uint_as_float(tmem_load_0_bf16_3[6] << 16);
                r1_3[13] = _tmem_load_0_4[13] - __uint_as_float(tmem_load_0_bf16_3[6] & mask_hi);
                r1_3[14] = _tmem_load_0_4[14] - __uint_as_float(tmem_load_0_bf16_3[7] << 16);
                r1_3[15] = _tmem_load_0_4[15] - __uint_as_float(tmem_load_0_bf16_3[7] & mask_hi);
                unsigned int r1_bf16_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_3[_lp*2 + 0], r1_3[_lp*2+1 + 0]));
                    r1_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_3[16];
                r2_3[0] = r1_3[0] - __uint_as_float(r1_bf16_4[0] << 16);
                r2_3[1] = r1_3[1] - __uint_as_float(r1_bf16_4[0] & mask_hi);
                r2_3[2] = r1_3[2] - __uint_as_float(r1_bf16_4[1] << 16);
                r2_3[3] = r1_3[3] - __uint_as_float(r1_bf16_4[1] & mask_hi);
                r2_3[4] = r1_3[4] - __uint_as_float(r1_bf16_4[2] << 16);
                r2_3[5] = r1_3[5] - __uint_as_float(r1_bf16_4[2] & mask_hi);
                r2_3[6] = r1_3[6] - __uint_as_float(r1_bf16_4[3] << 16);
                r2_3[7] = r1_3[7] - __uint_as_float(r1_bf16_4[3] & mask_hi);
                r2_3[8] = r1_3[8] - __uint_as_float(r1_bf16_4[4] << 16);
                r2_3[9] = r1_3[9] - __uint_as_float(r1_bf16_4[4] & mask_hi);
                r2_3[10] = r1_3[10] - __uint_as_float(r1_bf16_4[5] << 16);
                r2_3[11] = r1_3[11] - __uint_as_float(r1_bf16_4[5] & mask_hi);
                r2_3[12] = r1_3[12] - __uint_as_float(r1_bf16_4[6] << 16);
                r2_3[13] = r1_3[13] - __uint_as_float(r1_bf16_4[6] & mask_hi);
                r2_3[14] = r1_3[14] - __uint_as_float(r1_bf16_4[7] << 16);
                r2_3[15] = r1_3[15] - __uint_as_float(r1_bf16_4[7] & mask_hi);
                unsigned int r2_bf16_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_3[_lp*2 + 0], r2_3[_lp*2+1 + 0]));
                    r2_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 ^ (16384 + k_row * 128 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_3[0]), "r"(tmem_load_0_bf16_3[1]), "r"(tmem_load_0_bf16_3[2]), "r"(tmem_load_0_bf16_3[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 16 ^ (16384 + k_row * 128 + 16 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_3[4]), "r"(tmem_load_0_bf16_3[5]), "r"(tmem_load_0_bf16_3[6]), "r"(tmem_load_0_bf16_3[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 ^ (16384 + k_row * 128 >> 7 & 7) << 4))), "r"(r1_bf16_4[0]), "r"(r1_bf16_4[1]), "r"(r1_bf16_4[2]), "r"(r1_bf16_4[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 16 ^ (16384 + k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r1_bf16_4[4]), "r"(r1_bf16_4[5]), "r"(r1_bf16_4[6]), "r"(r1_bf16_4[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 ^ (16384 + k_row * 128 >> 7 & 7) << 4))), "r"(r2_bf16_4[0]), "r"(r2_bf16_4[1]), "r"(r2_bf16_4[2]), "r"(r2_bf16_4[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 16 ^ (16384 + k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r2_bf16_4[4]), "r"(r2_bf16_4[5]), "r"(r2_bf16_4[6]), "r"(r2_bf16_4[7]) : "memory");
                float hs2_0_4[16];
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    hs2_0_4[j_10] = _tmem_load_0_4[j_10] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, hs2_0_4);
                float _tmem_load_0_5[16];
                tmem_ld_x16(&_tmem_load_0_5[0], taddr + (unsigned int)h_row_base + 80);
                float _tmem_load_1_5[16];
                tmem_ld_x16(&_tmem_load_1_5[0], taddr + 256 + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_11 = 0; j_11 < 16; j_11++) {
                    _tmem_load_0_5[j_11] = _tmem_load_0_5[j_11] + _tmem_load_1_5[j_11];
                }
                unsigned int tmem_load_0_bf16_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0_5[_lp*2 + 0], _tmem_load_0_5[_lp*2+1 + 0]));
                    tmem_load_0_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_4[16];
                r1_4[0] = _tmem_load_0_5[0] - __uint_as_float(tmem_load_0_bf16_4[0] << 16);
                r1_4[1] = _tmem_load_0_5[1] - __uint_as_float(tmem_load_0_bf16_4[0] & mask_hi);
                r1_4[2] = _tmem_load_0_5[2] - __uint_as_float(tmem_load_0_bf16_4[1] << 16);
                r1_4[3] = _tmem_load_0_5[3] - __uint_as_float(tmem_load_0_bf16_4[1] & mask_hi);
                r1_4[4] = _tmem_load_0_5[4] - __uint_as_float(tmem_load_0_bf16_4[2] << 16);
                r1_4[5] = _tmem_load_0_5[5] - __uint_as_float(tmem_load_0_bf16_4[2] & mask_hi);
                r1_4[6] = _tmem_load_0_5[6] - __uint_as_float(tmem_load_0_bf16_4[3] << 16);
                r1_4[7] = _tmem_load_0_5[7] - __uint_as_float(tmem_load_0_bf16_4[3] & mask_hi);
                r1_4[8] = _tmem_load_0_5[8] - __uint_as_float(tmem_load_0_bf16_4[4] << 16);
                r1_4[9] = _tmem_load_0_5[9] - __uint_as_float(tmem_load_0_bf16_4[4] & mask_hi);
                r1_4[10] = _tmem_load_0_5[10] - __uint_as_float(tmem_load_0_bf16_4[5] << 16);
                r1_4[11] = _tmem_load_0_5[11] - __uint_as_float(tmem_load_0_bf16_4[5] & mask_hi);
                r1_4[12] = _tmem_load_0_5[12] - __uint_as_float(tmem_load_0_bf16_4[6] << 16);
                r1_4[13] = _tmem_load_0_5[13] - __uint_as_float(tmem_load_0_bf16_4[6] & mask_hi);
                r1_4[14] = _tmem_load_0_5[14] - __uint_as_float(tmem_load_0_bf16_4[7] << 16);
                r1_4[15] = _tmem_load_0_5[15] - __uint_as_float(tmem_load_0_bf16_4[7] & mask_hi);
                unsigned int r1_bf16_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_4[_lp*2 + 0], r1_4[_lp*2+1 + 0]));
                    r1_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_4[16];
                r2_4[0] = r1_4[0] - __uint_as_float(r1_bf16_5[0] << 16);
                r2_4[1] = r1_4[1] - __uint_as_float(r1_bf16_5[0] & mask_hi);
                r2_4[2] = r1_4[2] - __uint_as_float(r1_bf16_5[1] << 16);
                r2_4[3] = r1_4[3] - __uint_as_float(r1_bf16_5[1] & mask_hi);
                r2_4[4] = r1_4[4] - __uint_as_float(r1_bf16_5[2] << 16);
                r2_4[5] = r1_4[5] - __uint_as_float(r1_bf16_5[2] & mask_hi);
                r2_4[6] = r1_4[6] - __uint_as_float(r1_bf16_5[3] << 16);
                r2_4[7] = r1_4[7] - __uint_as_float(r1_bf16_5[3] & mask_hi);
                r2_4[8] = r1_4[8] - __uint_as_float(r1_bf16_5[4] << 16);
                r2_4[9] = r1_4[9] - __uint_as_float(r1_bf16_5[4] & mask_hi);
                r2_4[10] = r1_4[10] - __uint_as_float(r1_bf16_5[5] << 16);
                r2_4[11] = r1_4[11] - __uint_as_float(r1_bf16_5[5] & mask_hi);
                r2_4[12] = r1_4[12] - __uint_as_float(r1_bf16_5[6] << 16);
                r2_4[13] = r1_4[13] - __uint_as_float(r1_bf16_5[6] & mask_hi);
                r2_4[14] = r1_4[14] - __uint_as_float(r1_bf16_5[7] << 16);
                r2_4[15] = r1_4[15] - __uint_as_float(r1_bf16_5[7] & mask_hi);
                unsigned int r2_bf16_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_4[_lp*2 + 0], r2_4[_lp*2+1 + 0]));
                    r2_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 32 ^ (16384 + k_row * 128 + 32 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_4[0]), "r"(tmem_load_0_bf16_4[1]), "r"(tmem_load_0_bf16_4[2]), "r"(tmem_load_0_bf16_4[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 48 ^ (16384 + k_row * 128 + 48 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_4[4]), "r"(tmem_load_0_bf16_4[5]), "r"(tmem_load_0_bf16_4[6]), "r"(tmem_load_0_bf16_4[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 32 ^ (16384 + k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r1_bf16_5[0]), "r"(r1_bf16_5[1]), "r"(r1_bf16_5[2]), "r"(r1_bf16_5[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 48 ^ (16384 + k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r1_bf16_5[4]), "r"(r1_bf16_5[5]), "r"(r1_bf16_5[6]), "r"(r1_bf16_5[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 32 ^ (16384 + k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r2_bf16_5[0]), "r"(r2_bf16_5[1]), "r"(r2_bf16_5[2]), "r"(r2_bf16_5[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 48 ^ (16384 + k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r2_bf16_5[4]), "r"(r2_bf16_5[5]), "r"(r2_bf16_5[6]), "r"(r2_bf16_5[7]) : "memory");
                float hs2_0_5[16];
                #pragma unroll
                for (int j_12 = 0; j_12 < 16; j_12++) {
                    hs2_0_5[j_12] = _tmem_load_0_5[j_12] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, hs2_0_5);
                float _tmem_load_0_6[16];
                tmem_ld_x16(&_tmem_load_0_6[0], taddr + (unsigned int)h_row_base + 96);
                float _tmem_load_1_6[16];
                tmem_ld_x16(&_tmem_load_1_6[0], taddr + 256 + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_13 = 0; j_13 < 16; j_13++) {
                    _tmem_load_0_6[j_13] = _tmem_load_0_6[j_13] + _tmem_load_1_6[j_13];
                }
                unsigned int tmem_load_0_bf16_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0_6[_lp*2 + 0], _tmem_load_0_6[_lp*2+1 + 0]));
                    tmem_load_0_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_5[16];
                r1_5[0] = _tmem_load_0_6[0] - __uint_as_float(tmem_load_0_bf16_5[0] << 16);
                r1_5[1] = _tmem_load_0_6[1] - __uint_as_float(tmem_load_0_bf16_5[0] & mask_hi);
                r1_5[2] = _tmem_load_0_6[2] - __uint_as_float(tmem_load_0_bf16_5[1] << 16);
                r1_5[3] = _tmem_load_0_6[3] - __uint_as_float(tmem_load_0_bf16_5[1] & mask_hi);
                r1_5[4] = _tmem_load_0_6[4] - __uint_as_float(tmem_load_0_bf16_5[2] << 16);
                r1_5[5] = _tmem_load_0_6[5] - __uint_as_float(tmem_load_0_bf16_5[2] & mask_hi);
                r1_5[6] = _tmem_load_0_6[6] - __uint_as_float(tmem_load_0_bf16_5[3] << 16);
                r1_5[7] = _tmem_load_0_6[7] - __uint_as_float(tmem_load_0_bf16_5[3] & mask_hi);
                r1_5[8] = _tmem_load_0_6[8] - __uint_as_float(tmem_load_0_bf16_5[4] << 16);
                r1_5[9] = _tmem_load_0_6[9] - __uint_as_float(tmem_load_0_bf16_5[4] & mask_hi);
                r1_5[10] = _tmem_load_0_6[10] - __uint_as_float(tmem_load_0_bf16_5[5] << 16);
                r1_5[11] = _tmem_load_0_6[11] - __uint_as_float(tmem_load_0_bf16_5[5] & mask_hi);
                r1_5[12] = _tmem_load_0_6[12] - __uint_as_float(tmem_load_0_bf16_5[6] << 16);
                r1_5[13] = _tmem_load_0_6[13] - __uint_as_float(tmem_load_0_bf16_5[6] & mask_hi);
                r1_5[14] = _tmem_load_0_6[14] - __uint_as_float(tmem_load_0_bf16_5[7] << 16);
                r1_5[15] = _tmem_load_0_6[15] - __uint_as_float(tmem_load_0_bf16_5[7] & mask_hi);
                unsigned int r1_bf16_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_5[_lp*2 + 0], r1_5[_lp*2+1 + 0]));
                    r1_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_5[16];
                r2_5[0] = r1_5[0] - __uint_as_float(r1_bf16_6[0] << 16);
                r2_5[1] = r1_5[1] - __uint_as_float(r1_bf16_6[0] & mask_hi);
                r2_5[2] = r1_5[2] - __uint_as_float(r1_bf16_6[1] << 16);
                r2_5[3] = r1_5[3] - __uint_as_float(r1_bf16_6[1] & mask_hi);
                r2_5[4] = r1_5[4] - __uint_as_float(r1_bf16_6[2] << 16);
                r2_5[5] = r1_5[5] - __uint_as_float(r1_bf16_6[2] & mask_hi);
                r2_5[6] = r1_5[6] - __uint_as_float(r1_bf16_6[3] << 16);
                r2_5[7] = r1_5[7] - __uint_as_float(r1_bf16_6[3] & mask_hi);
                r2_5[8] = r1_5[8] - __uint_as_float(r1_bf16_6[4] << 16);
                r2_5[9] = r1_5[9] - __uint_as_float(r1_bf16_6[4] & mask_hi);
                r2_5[10] = r1_5[10] - __uint_as_float(r1_bf16_6[5] << 16);
                r2_5[11] = r1_5[11] - __uint_as_float(r1_bf16_6[5] & mask_hi);
                r2_5[12] = r1_5[12] - __uint_as_float(r1_bf16_6[6] << 16);
                r2_5[13] = r1_5[13] - __uint_as_float(r1_bf16_6[6] & mask_hi);
                r2_5[14] = r1_5[14] - __uint_as_float(r1_bf16_6[7] << 16);
                r2_5[15] = r1_5[15] - __uint_as_float(r1_bf16_6[7] & mask_hi);
                unsigned int r2_bf16_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_5[_lp*2 + 0], r2_5[_lp*2+1 + 0]));
                    r2_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 64 ^ (16384 + k_row * 128 + 64 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_5[0]), "r"(tmem_load_0_bf16_5[1]), "r"(tmem_load_0_bf16_5[2]), "r"(tmem_load_0_bf16_5[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 80 ^ (16384 + k_row * 128 + 80 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_5[4]), "r"(tmem_load_0_bf16_5[5]), "r"(tmem_load_0_bf16_5[6]), "r"(tmem_load_0_bf16_5[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 64 ^ (16384 + k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r1_bf16_6[0]), "r"(r1_bf16_6[1]), "r"(r1_bf16_6[2]), "r"(r1_bf16_6[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 80 ^ (16384 + k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r1_bf16_6[4]), "r"(r1_bf16_6[5]), "r"(r1_bf16_6[6]), "r"(r1_bf16_6[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 64 ^ (16384 + k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r2_bf16_6[0]), "r"(r2_bf16_6[1]), "r"(r2_bf16_6[2]), "r"(r2_bf16_6[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 80 ^ (16384 + k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r2_bf16_6[4]), "r"(r2_bf16_6[5]), "r"(r2_bf16_6[6]), "r"(r2_bf16_6[7]) : "memory");
                float hs2_0_6[16];
                #pragma unroll
                for (int j_14 = 0; j_14 < 16; j_14++) {
                    hs2_0_6[j_14] = _tmem_load_0_6[j_14] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, hs2_0_6);
                float _tmem_load_0_7[16];
                tmem_ld_x16(&_tmem_load_0_7[0], taddr + (unsigned int)h_row_base + 112);
                float _tmem_load_1_7[16];
                tmem_ld_x16(&_tmem_load_1_7[0], taddr + 256 + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_15 = 0; j_15 < 16; j_15++) {
                    _tmem_load_0_7[j_15] = _tmem_load_0_7[j_15] + _tmem_load_1_7[j_15];
                }
                unsigned int tmem_load_0_bf16_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0_7[_lp*2 + 0], _tmem_load_0_7[_lp*2+1 + 0]));
                    tmem_load_0_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_6[16];
                r1_6[0] = _tmem_load_0_7[0] - __uint_as_float(tmem_load_0_bf16_6[0] << 16);
                r1_6[1] = _tmem_load_0_7[1] - __uint_as_float(tmem_load_0_bf16_6[0] & mask_hi);
                r1_6[2] = _tmem_load_0_7[2] - __uint_as_float(tmem_load_0_bf16_6[1] << 16);
                r1_6[3] = _tmem_load_0_7[3] - __uint_as_float(tmem_load_0_bf16_6[1] & mask_hi);
                r1_6[4] = _tmem_load_0_7[4] - __uint_as_float(tmem_load_0_bf16_6[2] << 16);
                r1_6[5] = _tmem_load_0_7[5] - __uint_as_float(tmem_load_0_bf16_6[2] & mask_hi);
                r1_6[6] = _tmem_load_0_7[6] - __uint_as_float(tmem_load_0_bf16_6[3] << 16);
                r1_6[7] = _tmem_load_0_7[7] - __uint_as_float(tmem_load_0_bf16_6[3] & mask_hi);
                r1_6[8] = _tmem_load_0_7[8] - __uint_as_float(tmem_load_0_bf16_6[4] << 16);
                r1_6[9] = _tmem_load_0_7[9] - __uint_as_float(tmem_load_0_bf16_6[4] & mask_hi);
                r1_6[10] = _tmem_load_0_7[10] - __uint_as_float(tmem_load_0_bf16_6[5] << 16);
                r1_6[11] = _tmem_load_0_7[11] - __uint_as_float(tmem_load_0_bf16_6[5] & mask_hi);
                r1_6[12] = _tmem_load_0_7[12] - __uint_as_float(tmem_load_0_bf16_6[6] << 16);
                r1_6[13] = _tmem_load_0_7[13] - __uint_as_float(tmem_load_0_bf16_6[6] & mask_hi);
                r1_6[14] = _tmem_load_0_7[14] - __uint_as_float(tmem_load_0_bf16_6[7] << 16);
                r1_6[15] = _tmem_load_0_7[15] - __uint_as_float(tmem_load_0_bf16_6[7] & mask_hi);
                unsigned int r1_bf16_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_6[_lp*2 + 0], r1_6[_lp*2+1 + 0]));
                    r1_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_6[16];
                r2_6[0] = r1_6[0] - __uint_as_float(r1_bf16_7[0] << 16);
                r2_6[1] = r1_6[1] - __uint_as_float(r1_bf16_7[0] & mask_hi);
                r2_6[2] = r1_6[2] - __uint_as_float(r1_bf16_7[1] << 16);
                r2_6[3] = r1_6[3] - __uint_as_float(r1_bf16_7[1] & mask_hi);
                r2_6[4] = r1_6[4] - __uint_as_float(r1_bf16_7[2] << 16);
                r2_6[5] = r1_6[5] - __uint_as_float(r1_bf16_7[2] & mask_hi);
                r2_6[6] = r1_6[6] - __uint_as_float(r1_bf16_7[3] << 16);
                r2_6[7] = r1_6[7] - __uint_as_float(r1_bf16_7[3] & mask_hi);
                r2_6[8] = r1_6[8] - __uint_as_float(r1_bf16_7[4] << 16);
                r2_6[9] = r1_6[9] - __uint_as_float(r1_bf16_7[4] & mask_hi);
                r2_6[10] = r1_6[10] - __uint_as_float(r1_bf16_7[5] << 16);
                r2_6[11] = r1_6[11] - __uint_as_float(r1_bf16_7[5] & mask_hi);
                r2_6[12] = r1_6[12] - __uint_as_float(r1_bf16_7[6] << 16);
                r2_6[13] = r1_6[13] - __uint_as_float(r1_bf16_7[6] & mask_hi);
                r2_6[14] = r1_6[14] - __uint_as_float(r1_bf16_7[7] << 16);
                r2_6[15] = r1_6[15] - __uint_as_float(r1_bf16_7[7] & mask_hi);
                unsigned int r2_bf16_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_6[_lp*2 + 0], r2_6[_lp*2+1 + 0]));
                    r2_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 96 ^ (16384 + k_row * 128 + 96 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_6[0]), "r"(tmem_load_0_bf16_6[1]), "r"(tmem_load_0_bf16_6[2]), "r"(tmem_load_0_bf16_6[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 112 ^ (16384 + k_row * 128 + 112 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_6[4]), "r"(tmem_load_0_bf16_6[5]), "r"(tmem_load_0_bf16_6[6]), "r"(tmem_load_0_bf16_6[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 96 ^ (16384 + k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r1_bf16_7[0]), "r"(r1_bf16_7[1]), "r"(r1_bf16_7[2]), "r"(r1_bf16_7[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 112 ^ (16384 + k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r1_bf16_7[4]), "r"(r1_bf16_7[5]), "r"(r1_bf16_7[6]), "r"(r1_bf16_7[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 96 ^ (16384 + k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r2_bf16_7[0]), "r"(r2_bf16_7[1]), "r"(r2_bf16_7[2]), "r"(r2_bf16_7[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 112 ^ (16384 + k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r2_bf16_7[4]), "r"(r2_bf16_7[5]), "r"(r2_bf16_7[6]), "r"(r2_bf16_7[7]) : "memory");
                float hs2_0_7[16];
                #pragma unroll
                for (int j_16 = 0; j_16 < 16; j_16++) {
                    hs2_0_7[j_16] = _tmem_load_0_7[j_16] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, hs2_0_7);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long u_base = ((long long)trow * (long long)num_heads + (long long)head) * 128 + (long long)v0;
                float _tmem_load_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_1_0 = lane_half * 16;
                float uv_1_0[16];
                uv_1_0[0] = 0.0f;
                uv_1_0[1] = 0.0f;
                uv_1_0[2] = 0.0f;
                uv_1_0[3] = 0.0f;
                uv_1_0[4] = 0.0f;
                uv_1_0[5] = 0.0f;
                uv_1_0[6] = 0.0f;
                uv_1_0[7] = 0.0f;
                uv_1_0[8] = 0.0f;
                uv_1_0[9] = 0.0f;
                uv_1_0[10] = 0.0f;
                uv_1_0[11] = 0.0f;
                uv_1_0[12] = 0.0f;
                uv_1_0[13] = 0.0f;
                uv_1_0[14] = 0.0f;
                uv_1_0[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_1_0);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_1_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&uv_1_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_1_0[16];
                #pragma unroll
                for (int j_1_8 = 0; j_1_8 < 16; j_1_8++) {
                    bvv_1_0[j_1_8] = uv_1_0[j_1_8] - _tmem_load_2[j_1_8];
                }
                unsigned int tmem_load_0_bf16_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1_0[_lp*2 + 0], bvv_1_0[_lp*2+1 + 0]));
                    tmem_load_0_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_7[16];
                r1_7[0] = bvv_1_0[0] - __uint_as_float(tmem_load_0_bf16_7[0] << 16);
                r1_7[1] = bvv_1_0[1] - __uint_as_float(tmem_load_0_bf16_7[0] & mask_hi);
                r1_7[2] = bvv_1_0[2] - __uint_as_float(tmem_load_0_bf16_7[1] << 16);
                r1_7[3] = bvv_1_0[3] - __uint_as_float(tmem_load_0_bf16_7[1] & mask_hi);
                r1_7[4] = bvv_1_0[4] - __uint_as_float(tmem_load_0_bf16_7[2] << 16);
                r1_7[5] = bvv_1_0[5] - __uint_as_float(tmem_load_0_bf16_7[2] & mask_hi);
                r1_7[6] = bvv_1_0[6] - __uint_as_float(tmem_load_0_bf16_7[3] << 16);
                r1_7[7] = bvv_1_0[7] - __uint_as_float(tmem_load_0_bf16_7[3] & mask_hi);
                r1_7[8] = bvv_1_0[8] - __uint_as_float(tmem_load_0_bf16_7[4] << 16);
                r1_7[9] = bvv_1_0[9] - __uint_as_float(tmem_load_0_bf16_7[4] & mask_hi);
                r1_7[10] = bvv_1_0[10] - __uint_as_float(tmem_load_0_bf16_7[5] << 16);
                r1_7[11] = bvv_1_0[11] - __uint_as_float(tmem_load_0_bf16_7[5] & mask_hi);
                r1_7[12] = bvv_1_0[12] - __uint_as_float(tmem_load_0_bf16_7[6] << 16);
                r1_7[13] = bvv_1_0[13] - __uint_as_float(tmem_load_0_bf16_7[6] & mask_hi);
                r1_7[14] = bvv_1_0[14] - __uint_as_float(tmem_load_0_bf16_7[7] << 16);
                r1_7[15] = bvv_1_0[15] - __uint_as_float(tmem_load_0_bf16_7[7] & mask_hi);
                unsigned int r1_bf16_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_7[_lp*2 + 0], r1_7[_lp*2+1 + 0]));
                    r1_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_7[16];
                r2_7[0] = r1_7[0] - __uint_as_float(r1_bf16_8[0] << 16);
                r2_7[1] = r1_7[1] - __uint_as_float(r1_bf16_8[0] & mask_hi);
                r2_7[2] = r1_7[2] - __uint_as_float(r1_bf16_8[1] << 16);
                r2_7[3] = r1_7[3] - __uint_as_float(r1_bf16_8[1] & mask_hi);
                r2_7[4] = r1_7[4] - __uint_as_float(r1_bf16_8[2] << 16);
                r2_7[5] = r1_7[5] - __uint_as_float(r1_bf16_8[2] & mask_hi);
                r2_7[6] = r1_7[6] - __uint_as_float(r1_bf16_8[3] << 16);
                r2_7[7] = r1_7[7] - __uint_as_float(r1_bf16_8[3] & mask_hi);
                r2_7[8] = r1_7[8] - __uint_as_float(r1_bf16_8[4] << 16);
                r2_7[9] = r1_7[9] - __uint_as_float(r1_bf16_8[4] & mask_hi);
                r2_7[10] = r1_7[10] - __uint_as_float(r1_bf16_8[5] << 16);
                r2_7[11] = r1_7[11] - __uint_as_float(r1_bf16_8[5] & mask_hi);
                r2_7[12] = r1_7[12] - __uint_as_float(r1_bf16_8[6] << 16);
                r2_7[13] = r1_7[13] - __uint_as_float(r1_bf16_8[6] & mask_hi);
                r2_7[14] = r1_7[14] - __uint_as_float(r1_bf16_8[7] << 16);
                r2_7[15] = r1_7[15] - __uint_as_float(r1_bf16_8[7] & mask_hi);
                unsigned int r2_bf16_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_7[_lp*2 + 0], r2_7[_lp*2+1 + 0]));
                    r2_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_1_0 / 64 * 8192 + t_row * 128 + col_1_0 % 64 * 2 ^ (col_1_0 / 64 * 8192 + t_row * 128 + col_1_0 % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_7[0]), "r"(tmem_load_0_bf16_7[1]), "r"(tmem_load_0_bf16_7[2]), "r"(tmem_load_0_bf16_7[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_1_0 + 8) / 64 * 8192 + t_row * 128 + (col_1_0 + 8) % 64 * 2 ^ ((col_1_0 + 8) / 64 * 8192 + t_row * 128 + (col_1_0 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_7[4]), "r"(tmem_load_0_bf16_7[5]), "r"(tmem_load_0_bf16_7[6]), "r"(tmem_load_0_bf16_7[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_1_0 / 64 * 8192 + t_row * 128 + col_1_0 % 64 * 2 ^ (col_1_0 / 64 * 8192 + t_row * 128 + col_1_0 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_8[0]), "r"(r1_bf16_8[1]), "r"(r1_bf16_8[2]), "r"(r1_bf16_8[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_1_0 + 8) / 64 * 8192 + t_row * 128 + (col_1_0 + 8) % 64 * 2 ^ ((col_1_0 + 8) / 64 * 8192 + t_row * 128 + (col_1_0 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_8[4]), "r"(r1_bf16_8[5]), "r"(r1_bf16_8[6]), "r"(r1_bf16_8[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_1_0 / 64 * 8192 + t_row * 128 + col_1_0 % 64 * 2 ^ (col_1_0 / 64 * 8192 + t_row * 128 + col_1_0 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_8[0]), "r"(r2_bf16_8[1]), "r"(r2_bf16_8[2]), "r"(r2_bf16_8[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_1_0 + 8) / 64 * 8192 + t_row * 128 + (col_1_0 + 8) % 64 * 2 ^ ((col_1_0 + 8) / 64 * 8192 + t_row * 128 + (col_1_0 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_8[4]), "r"(r2_bf16_8[5]), "r"(r2_bf16_8[6]), "r"(r2_bf16_8[7]) : "memory");
                float _tmem_load_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_1_1 = 32 + lane_half * 16;
                float uv_1_1[16];
                uv_1_1[0] = 0.0f;
                uv_1_1[1] = 0.0f;
                uv_1_1[2] = 0.0f;
                uv_1_1[3] = 0.0f;
                uv_1_1[4] = 0.0f;
                uv_1_1[5] = 0.0f;
                uv_1_1[6] = 0.0f;
                uv_1_1[7] = 0.0f;
                uv_1_1[8] = 0.0f;
                uv_1_1[9] = 0.0f;
                uv_1_1[10] = 0.0f;
                uv_1_1[11] = 0.0f;
                uv_1_1[12] = 0.0f;
                uv_1_1[13] = 0.0f;
                uv_1_1[14] = 0.0f;
                uv_1_1[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_1_1);
                        uint4 _vld_1[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                                (&uv_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_1_1[16];
                #pragma unroll
                for (int j_1_9 = 0; j_1_9 < 16; j_1_9++) {
                    bvv_1_1[j_1_9] = uv_1_1[j_1_9] - _tmem_load_3[j_1_9];
                }
                unsigned int tmem_load_0_bf16_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1_1[_lp*2 + 0], bvv_1_1[_lp*2+1 + 0]));
                    tmem_load_0_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_8[16];
                r1_8[0] = bvv_1_1[0] - __uint_as_float(tmem_load_0_bf16_8[0] << 16);
                r1_8[1] = bvv_1_1[1] - __uint_as_float(tmem_load_0_bf16_8[0] & mask_hi);
                r1_8[2] = bvv_1_1[2] - __uint_as_float(tmem_load_0_bf16_8[1] << 16);
                r1_8[3] = bvv_1_1[3] - __uint_as_float(tmem_load_0_bf16_8[1] & mask_hi);
                r1_8[4] = bvv_1_1[4] - __uint_as_float(tmem_load_0_bf16_8[2] << 16);
                r1_8[5] = bvv_1_1[5] - __uint_as_float(tmem_load_0_bf16_8[2] & mask_hi);
                r1_8[6] = bvv_1_1[6] - __uint_as_float(tmem_load_0_bf16_8[3] << 16);
                r1_8[7] = bvv_1_1[7] - __uint_as_float(tmem_load_0_bf16_8[3] & mask_hi);
                r1_8[8] = bvv_1_1[8] - __uint_as_float(tmem_load_0_bf16_8[4] << 16);
                r1_8[9] = bvv_1_1[9] - __uint_as_float(tmem_load_0_bf16_8[4] & mask_hi);
                r1_8[10] = bvv_1_1[10] - __uint_as_float(tmem_load_0_bf16_8[5] << 16);
                r1_8[11] = bvv_1_1[11] - __uint_as_float(tmem_load_0_bf16_8[5] & mask_hi);
                r1_8[12] = bvv_1_1[12] - __uint_as_float(tmem_load_0_bf16_8[6] << 16);
                r1_8[13] = bvv_1_1[13] - __uint_as_float(tmem_load_0_bf16_8[6] & mask_hi);
                r1_8[14] = bvv_1_1[14] - __uint_as_float(tmem_load_0_bf16_8[7] << 16);
                r1_8[15] = bvv_1_1[15] - __uint_as_float(tmem_load_0_bf16_8[7] & mask_hi);
                unsigned int r1_bf16_9[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_8[_lp*2 + 0], r1_8[_lp*2+1 + 0]));
                    r1_bf16_9[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_8[16];
                r2_8[0] = r1_8[0] - __uint_as_float(r1_bf16_9[0] << 16);
                r2_8[1] = r1_8[1] - __uint_as_float(r1_bf16_9[0] & mask_hi);
                r2_8[2] = r1_8[2] - __uint_as_float(r1_bf16_9[1] << 16);
                r2_8[3] = r1_8[3] - __uint_as_float(r1_bf16_9[1] & mask_hi);
                r2_8[4] = r1_8[4] - __uint_as_float(r1_bf16_9[2] << 16);
                r2_8[5] = r1_8[5] - __uint_as_float(r1_bf16_9[2] & mask_hi);
                r2_8[6] = r1_8[6] - __uint_as_float(r1_bf16_9[3] << 16);
                r2_8[7] = r1_8[7] - __uint_as_float(r1_bf16_9[3] & mask_hi);
                r2_8[8] = r1_8[8] - __uint_as_float(r1_bf16_9[4] << 16);
                r2_8[9] = r1_8[9] - __uint_as_float(r1_bf16_9[4] & mask_hi);
                r2_8[10] = r1_8[10] - __uint_as_float(r1_bf16_9[5] << 16);
                r2_8[11] = r1_8[11] - __uint_as_float(r1_bf16_9[5] & mask_hi);
                r2_8[12] = r1_8[12] - __uint_as_float(r1_bf16_9[6] << 16);
                r2_8[13] = r1_8[13] - __uint_as_float(r1_bf16_9[6] & mask_hi);
                r2_8[14] = r1_8[14] - __uint_as_float(r1_bf16_9[7] << 16);
                r2_8[15] = r1_8[15] - __uint_as_float(r1_bf16_9[7] & mask_hi);
                unsigned int r2_bf16_9[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_8[_lp*2 + 0], r2_8[_lp*2+1 + 0]));
                    r2_bf16_9[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_1_1 / 64 * 8192 + t_row * 128 + col_1_1 % 64 * 2 ^ (col_1_1 / 64 * 8192 + t_row * 128 + col_1_1 % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_8[0]), "r"(tmem_load_0_bf16_8[1]), "r"(tmem_load_0_bf16_8[2]), "r"(tmem_load_0_bf16_8[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_1_1 + 8) / 64 * 8192 + t_row * 128 + (col_1_1 + 8) % 64 * 2 ^ ((col_1_1 + 8) / 64 * 8192 + t_row * 128 + (col_1_1 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_8[4]), "r"(tmem_load_0_bf16_8[5]), "r"(tmem_load_0_bf16_8[6]), "r"(tmem_load_0_bf16_8[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_1_1 / 64 * 8192 + t_row * 128 + col_1_1 % 64 * 2 ^ (col_1_1 / 64 * 8192 + t_row * 128 + col_1_1 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_9[0]), "r"(r1_bf16_9[1]), "r"(r1_bf16_9[2]), "r"(r1_bf16_9[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_1_1 + 8) / 64 * 8192 + t_row * 128 + (col_1_1 + 8) % 64 * 2 ^ ((col_1_1 + 8) / 64 * 8192 + t_row * 128 + (col_1_1 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_9[4]), "r"(r1_bf16_9[5]), "r"(r1_bf16_9[6]), "r"(r1_bf16_9[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_1_1 / 64 * 8192 + t_row * 128 + col_1_1 % 64 * 2 ^ (col_1_1 / 64 * 8192 + t_row * 128 + col_1_1 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_9[0]), "r"(r2_bf16_9[1]), "r"(r2_bf16_9[2]), "r"(r2_bf16_9[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_1_1 + 8) / 64 * 8192 + t_row * 128 + (col_1_1 + 8) % 64 * 2 ^ ((col_1_1 + 8) / 64 * 8192 + t_row * 128 + (col_1_1 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_9[4]), "r"(r2_bf16_9[5]), "r"(r2_bf16_9[6]), "r"(r2_bf16_9[7]) : "memory");
                float _tmem_load_4[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_1_2 = 64 + lane_half * 16;
                float uv_1_2[16];
                uv_1_2[0] = 0.0f;
                uv_1_2[1] = 0.0f;
                uv_1_2[2] = 0.0f;
                uv_1_2[3] = 0.0f;
                uv_1_2[4] = 0.0f;
                uv_1_2[5] = 0.0f;
                uv_1_2[6] = 0.0f;
                uv_1_2[7] = 0.0f;
                uv_1_2[8] = 0.0f;
                uv_1_2[9] = 0.0f;
                uv_1_2[10] = 0.0f;
                uv_1_2[11] = 0.0f;
                uv_1_2[12] = 0.0f;
                uv_1_2[13] = 0.0f;
                uv_1_2[14] = 0.0f;
                uv_1_2[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_1_2);
                        uint4 _vld_2[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_1_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                (&uv_1_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_1_2[16];
                #pragma unroll
                for (int j_1_10 = 0; j_1_10 < 16; j_1_10++) {
                    bvv_1_2[j_1_10] = uv_1_2[j_1_10] - _tmem_load_4[j_1_10];
                }
                unsigned int tmem_load_0_bf16_9[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1_2[_lp*2 + 0], bvv_1_2[_lp*2+1 + 0]));
                    tmem_load_0_bf16_9[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_9[16];
                r1_9[0] = bvv_1_2[0] - __uint_as_float(tmem_load_0_bf16_9[0] << 16);
                r1_9[1] = bvv_1_2[1] - __uint_as_float(tmem_load_0_bf16_9[0] & mask_hi);
                r1_9[2] = bvv_1_2[2] - __uint_as_float(tmem_load_0_bf16_9[1] << 16);
                r1_9[3] = bvv_1_2[3] - __uint_as_float(tmem_load_0_bf16_9[1] & mask_hi);
                r1_9[4] = bvv_1_2[4] - __uint_as_float(tmem_load_0_bf16_9[2] << 16);
                r1_9[5] = bvv_1_2[5] - __uint_as_float(tmem_load_0_bf16_9[2] & mask_hi);
                r1_9[6] = bvv_1_2[6] - __uint_as_float(tmem_load_0_bf16_9[3] << 16);
                r1_9[7] = bvv_1_2[7] - __uint_as_float(tmem_load_0_bf16_9[3] & mask_hi);
                r1_9[8] = bvv_1_2[8] - __uint_as_float(tmem_load_0_bf16_9[4] << 16);
                r1_9[9] = bvv_1_2[9] - __uint_as_float(tmem_load_0_bf16_9[4] & mask_hi);
                r1_9[10] = bvv_1_2[10] - __uint_as_float(tmem_load_0_bf16_9[5] << 16);
                r1_9[11] = bvv_1_2[11] - __uint_as_float(tmem_load_0_bf16_9[5] & mask_hi);
                r1_9[12] = bvv_1_2[12] - __uint_as_float(tmem_load_0_bf16_9[6] << 16);
                r1_9[13] = bvv_1_2[13] - __uint_as_float(tmem_load_0_bf16_9[6] & mask_hi);
                r1_9[14] = bvv_1_2[14] - __uint_as_float(tmem_load_0_bf16_9[7] << 16);
                r1_9[15] = bvv_1_2[15] - __uint_as_float(tmem_load_0_bf16_9[7] & mask_hi);
                unsigned int r1_bf16_10[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_9[_lp*2 + 0], r1_9[_lp*2+1 + 0]));
                    r1_bf16_10[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_9[16];
                r2_9[0] = r1_9[0] - __uint_as_float(r1_bf16_10[0] << 16);
                r2_9[1] = r1_9[1] - __uint_as_float(r1_bf16_10[0] & mask_hi);
                r2_9[2] = r1_9[2] - __uint_as_float(r1_bf16_10[1] << 16);
                r2_9[3] = r1_9[3] - __uint_as_float(r1_bf16_10[1] & mask_hi);
                r2_9[4] = r1_9[4] - __uint_as_float(r1_bf16_10[2] << 16);
                r2_9[5] = r1_9[5] - __uint_as_float(r1_bf16_10[2] & mask_hi);
                r2_9[6] = r1_9[6] - __uint_as_float(r1_bf16_10[3] << 16);
                r2_9[7] = r1_9[7] - __uint_as_float(r1_bf16_10[3] & mask_hi);
                r2_9[8] = r1_9[8] - __uint_as_float(r1_bf16_10[4] << 16);
                r2_9[9] = r1_9[9] - __uint_as_float(r1_bf16_10[4] & mask_hi);
                r2_9[10] = r1_9[10] - __uint_as_float(r1_bf16_10[5] << 16);
                r2_9[11] = r1_9[11] - __uint_as_float(r1_bf16_10[5] & mask_hi);
                r2_9[12] = r1_9[12] - __uint_as_float(r1_bf16_10[6] << 16);
                r2_9[13] = r1_9[13] - __uint_as_float(r1_bf16_10[6] & mask_hi);
                r2_9[14] = r1_9[14] - __uint_as_float(r1_bf16_10[7] << 16);
                r2_9[15] = r1_9[15] - __uint_as_float(r1_bf16_10[7] & mask_hi);
                unsigned int r2_bf16_10[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_9[_lp*2 + 0], r2_9[_lp*2+1 + 0]));
                    r2_bf16_10[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_1_2 / 64 * 8192 + t_row * 128 + col_1_2 % 64 * 2 ^ (col_1_2 / 64 * 8192 + t_row * 128 + col_1_2 % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_9[0]), "r"(tmem_load_0_bf16_9[1]), "r"(tmem_load_0_bf16_9[2]), "r"(tmem_load_0_bf16_9[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_1_2 + 8) / 64 * 8192 + t_row * 128 + (col_1_2 + 8) % 64 * 2 ^ ((col_1_2 + 8) / 64 * 8192 + t_row * 128 + (col_1_2 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_9[4]), "r"(tmem_load_0_bf16_9[5]), "r"(tmem_load_0_bf16_9[6]), "r"(tmem_load_0_bf16_9[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_1_2 / 64 * 8192 + t_row * 128 + col_1_2 % 64 * 2 ^ (col_1_2 / 64 * 8192 + t_row * 128 + col_1_2 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_10[0]), "r"(r1_bf16_10[1]), "r"(r1_bf16_10[2]), "r"(r1_bf16_10[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_1_2 + 8) / 64 * 8192 + t_row * 128 + (col_1_2 + 8) % 64 * 2 ^ ((col_1_2 + 8) / 64 * 8192 + t_row * 128 + (col_1_2 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_10[4]), "r"(r1_bf16_10[5]), "r"(r1_bf16_10[6]), "r"(r1_bf16_10[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_1_2 / 64 * 8192 + t_row * 128 + col_1_2 % 64 * 2 ^ (col_1_2 / 64 * 8192 + t_row * 128 + col_1_2 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_10[0]), "r"(r2_bf16_10[1]), "r"(r2_bf16_10[2]), "r"(r2_bf16_10[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_1_2 + 8) / 64 * 8192 + t_row * 128 + (col_1_2 + 8) % 64 * 2 ^ ((col_1_2 + 8) / 64 * 8192 + t_row * 128 + (col_1_2 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_10[4]), "r"(r2_bf16_10[5]), "r"(r2_bf16_10[6]), "r"(r2_bf16_10[7]) : "memory");
                float _tmem_load_5[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_1_3 = 96 + lane_half * 16;
                float uv_1_3[16];
                uv_1_3[0] = 0.0f;
                uv_1_3[1] = 0.0f;
                uv_1_3[2] = 0.0f;
                uv_1_3[3] = 0.0f;
                uv_1_3[4] = 0.0f;
                uv_1_3[5] = 0.0f;
                uv_1_3[6] = 0.0f;
                uv_1_3[7] = 0.0f;
                uv_1_3[8] = 0.0f;
                uv_1_3[9] = 0.0f;
                uv_1_3[10] = 0.0f;
                uv_1_3[11] = 0.0f;
                uv_1_3[12] = 0.0f;
                uv_1_3[13] = 0.0f;
                uv_1_3[14] = 0.0f;
                uv_1_3[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_1_3);
                        uint4 _vld_3[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_3[_blk] = _vptr_3[_blk];
                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_1_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                (&uv_1_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_1_3[16];
                #pragma unroll
                for (int j_1_11 = 0; j_1_11 < 16; j_1_11++) {
                    bvv_1_3[j_1_11] = uv_1_3[j_1_11] - _tmem_load_5[j_1_11];
                }
                unsigned int tmem_load_0_bf16_10[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1_3[_lp*2 + 0], bvv_1_3[_lp*2+1 + 0]));
                    tmem_load_0_bf16_10[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_10[16];
                r1_10[0] = bvv_1_3[0] - __uint_as_float(tmem_load_0_bf16_10[0] << 16);
                r1_10[1] = bvv_1_3[1] - __uint_as_float(tmem_load_0_bf16_10[0] & mask_hi);
                r1_10[2] = bvv_1_3[2] - __uint_as_float(tmem_load_0_bf16_10[1] << 16);
                r1_10[3] = bvv_1_3[3] - __uint_as_float(tmem_load_0_bf16_10[1] & mask_hi);
                r1_10[4] = bvv_1_3[4] - __uint_as_float(tmem_load_0_bf16_10[2] << 16);
                r1_10[5] = bvv_1_3[5] - __uint_as_float(tmem_load_0_bf16_10[2] & mask_hi);
                r1_10[6] = bvv_1_3[6] - __uint_as_float(tmem_load_0_bf16_10[3] << 16);
                r1_10[7] = bvv_1_3[7] - __uint_as_float(tmem_load_0_bf16_10[3] & mask_hi);
                r1_10[8] = bvv_1_3[8] - __uint_as_float(tmem_load_0_bf16_10[4] << 16);
                r1_10[9] = bvv_1_3[9] - __uint_as_float(tmem_load_0_bf16_10[4] & mask_hi);
                r1_10[10] = bvv_1_3[10] - __uint_as_float(tmem_load_0_bf16_10[5] << 16);
                r1_10[11] = bvv_1_3[11] - __uint_as_float(tmem_load_0_bf16_10[5] & mask_hi);
                r1_10[12] = bvv_1_3[12] - __uint_as_float(tmem_load_0_bf16_10[6] << 16);
                r1_10[13] = bvv_1_3[13] - __uint_as_float(tmem_load_0_bf16_10[6] & mask_hi);
                r1_10[14] = bvv_1_3[14] - __uint_as_float(tmem_load_0_bf16_10[7] << 16);
                r1_10[15] = bvv_1_3[15] - __uint_as_float(tmem_load_0_bf16_10[7] & mask_hi);
                unsigned int r1_bf16_11[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_10[_lp*2 + 0], r1_10[_lp*2+1 + 0]));
                    r1_bf16_11[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_10[16];
                r2_10[0] = r1_10[0] - __uint_as_float(r1_bf16_11[0] << 16);
                r2_10[1] = r1_10[1] - __uint_as_float(r1_bf16_11[0] & mask_hi);
                r2_10[2] = r1_10[2] - __uint_as_float(r1_bf16_11[1] << 16);
                r2_10[3] = r1_10[3] - __uint_as_float(r1_bf16_11[1] & mask_hi);
                r2_10[4] = r1_10[4] - __uint_as_float(r1_bf16_11[2] << 16);
                r2_10[5] = r1_10[5] - __uint_as_float(r1_bf16_11[2] & mask_hi);
                r2_10[6] = r1_10[6] - __uint_as_float(r1_bf16_11[3] << 16);
                r2_10[7] = r1_10[7] - __uint_as_float(r1_bf16_11[3] & mask_hi);
                r2_10[8] = r1_10[8] - __uint_as_float(r1_bf16_11[4] << 16);
                r2_10[9] = r1_10[9] - __uint_as_float(r1_bf16_11[4] & mask_hi);
                r2_10[10] = r1_10[10] - __uint_as_float(r1_bf16_11[5] << 16);
                r2_10[11] = r1_10[11] - __uint_as_float(r1_bf16_11[5] & mask_hi);
                r2_10[12] = r1_10[12] - __uint_as_float(r1_bf16_11[6] << 16);
                r2_10[13] = r1_10[13] - __uint_as_float(r1_bf16_11[6] & mask_hi);
                r2_10[14] = r1_10[14] - __uint_as_float(r1_bf16_11[7] << 16);
                r2_10[15] = r1_10[15] - __uint_as_float(r1_bf16_11[7] & mask_hi);
                unsigned int r2_bf16_11[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_10[_lp*2 + 0], r2_10[_lp*2+1 + 0]));
                    r2_bf16_11[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_1_3 / 64 * 8192 + t_row * 128 + col_1_3 % 64 * 2 ^ (col_1_3 / 64 * 8192 + t_row * 128 + col_1_3 % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_10[0]), "r"(tmem_load_0_bf16_10[1]), "r"(tmem_load_0_bf16_10[2]), "r"(tmem_load_0_bf16_10[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_1_3 + 8) / 64 * 8192 + t_row * 128 + (col_1_3 + 8) % 64 * 2 ^ ((col_1_3 + 8) / 64 * 8192 + t_row * 128 + (col_1_3 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_10[4]), "r"(tmem_load_0_bf16_10[5]), "r"(tmem_load_0_bf16_10[6]), "r"(tmem_load_0_bf16_10[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_1_3 / 64 * 8192 + t_row * 128 + col_1_3 % 64 * 2 ^ (col_1_3 / 64 * 8192 + t_row * 128 + col_1_3 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_11[0]), "r"(r1_bf16_11[1]), "r"(r1_bf16_11[2]), "r"(r1_bf16_11[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_1_3 + 8) / 64 * 8192 + t_row * 128 + (col_1_3 + 8) % 64 * 2 ^ ((col_1_3 + 8) / 64 * 8192 + t_row * 128 + (col_1_3 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_bf16_11[4]), "r"(r1_bf16_11[5]), "r"(r1_bf16_11[6]), "r"(r1_bf16_11[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_1_3 / 64 * 8192 + t_row * 128 + col_1_3 % 64 * 2 ^ (col_1_3 / 64 * 8192 + t_row * 128 + col_1_3 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_11[0]), "r"(r2_bf16_11[1]), "r"(r2_bf16_11[2]), "r"(r2_bf16_11[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_1_3 + 8) / 64 * 8192 + t_row * 128 + (col_1_3 + 8) % 64 * 2 ^ ((col_1_3 + 8) / 64 * 8192 + t_row * 128 + (col_1_3 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_bf16_11[4]), "r"(r2_bf16_11[5]), "r"(r2_bf16_11[6]), "r"(r2_bf16_11[7]) : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int slot = out_index[seg];
            if (slot >= 0) {
                long long tile = (long long)((head / out_hg * out_n + slot) * out_hg + head % out_hg) * (long long)tile_stride * 16384;
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)h_row_base);
                float _tmem_load_36_0[16];
                tmem_ld_x16(&_tmem_load_36_0[0], taddr + 256 + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_12 = 0; j_1_12 < 16; j_1_12++) {
                    _tmem_load_6[j_1_12] = _tmem_load_6[j_1_12] + _tmem_load_36_0[j_1_12];
                }
                if (write_transposed == 1) {
                    long long row_at_1_0 = tile + (long long)(k_row * 128 + v0);
                    {
                        float4 _v4 = make_float4(_tmem_load_6[0 + 0], _tmem_load_6[0 + 1], _tmem_load_6[0 + 2], _tmem_load_6[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_6[4 + 0], _tmem_load_6[4 + 1], _tmem_load_6[4 + 2], _tmem_load_6[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_6[8 + 0], _tmem_load_6[8 + 1], _tmem_load_6[8 + 2], _tmem_load_6[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_6[12 + 0], _tmem_load_6[12 + 1], _tmem_load_6[12 + 2], _tmem_load_6[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_0 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_13 = 0; j_1_13 < 16; j_1_13++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + j_1_13) * 128 + k_row))) = _tmem_load_6[j_1_13];
                    }
                }
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)h_row_base + 16);
                float _tmem_load_36_1[16];
                tmem_ld_x16(&_tmem_load_36_1[0], taddr + 256 + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_14 = 0; j_1_14 < 16; j_1_14++) {
                    _tmem_load_7[j_1_14] = _tmem_load_7[j_1_14] + _tmem_load_36_1[j_1_14];
                }
                if (write_transposed == 1) {
                    long long row_at_1_1 = tile + (long long)(k_row * 128 + v0 + 16);
                    {
                        float4 _v4 = make_float4(_tmem_load_7[0 + 0], _tmem_load_7[0 + 1], _tmem_load_7[0 + 2], _tmem_load_7[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[4 + 0], _tmem_load_7[4 + 1], _tmem_load_7[4 + 2], _tmem_load_7[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[8 + 0], _tmem_load_7[8 + 1], _tmem_load_7[8 + 2], _tmem_load_7[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_7[12 + 0], _tmem_load_7[12 + 1], _tmem_load_7[12 + 2], _tmem_load_7[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_1 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_15 = 0; j_1_15 < 16; j_1_15++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 16 + j_1_15) * 128 + k_row))) = _tmem_load_7[j_1_15];
                    }
                }
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], taddr + (unsigned int)h_row_base + 32);
                float _tmem_load_36_2[16];
                tmem_ld_x16(&_tmem_load_36_2[0], taddr + 256 + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_16 = 0; j_1_16 < 16; j_1_16++) {
                    _tmem_load_8[j_1_16] = _tmem_load_8[j_1_16] + _tmem_load_36_2[j_1_16];
                }
                if (write_transposed == 1) {
                    long long row_at_1_2 = tile + (long long)(k_row * 128 + v0 + 32);
                    {
                        float4 _v4 = make_float4(_tmem_load_8[0 + 0], _tmem_load_8[0 + 1], _tmem_load_8[0 + 2], _tmem_load_8[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_2) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_8[4 + 0], _tmem_load_8[4 + 1], _tmem_load_8[4 + 2], _tmem_load_8[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_2 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_8[8 + 0], _tmem_load_8[8 + 1], _tmem_load_8[8 + 2], _tmem_load_8[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_2 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_8[12 + 0], _tmem_load_8[12 + 1], _tmem_load_8[12 + 2], _tmem_load_8[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_2 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_17 = 0; j_1_17 < 16; j_1_17++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 32 + j_1_17) * 128 + k_row))) = _tmem_load_8[j_1_17];
                    }
                }
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], taddr + (unsigned int)h_row_base + 48);
                float _tmem_load_36_3[16];
                tmem_ld_x16(&_tmem_load_36_3[0], taddr + 256 + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_18 = 0; j_1_18 < 16; j_1_18++) {
                    _tmem_load_9[j_1_18] = _tmem_load_9[j_1_18] + _tmem_load_36_3[j_1_18];
                }
                if (write_transposed == 1) {
                    long long row_at_1_3 = tile + (long long)(k_row * 128 + v0 + 48);
                    {
                        float4 _v4 = make_float4(_tmem_load_9[0 + 0], _tmem_load_9[0 + 1], _tmem_load_9[0 + 2], _tmem_load_9[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_3) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[4 + 0], _tmem_load_9[4 + 1], _tmem_load_9[4 + 2], _tmem_load_9[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_3 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[8 + 0], _tmem_load_9[8 + 1], _tmem_load_9[8 + 2], _tmem_load_9[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_3 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_9[12 + 0], _tmem_load_9[12 + 1], _tmem_load_9[12 + 2], _tmem_load_9[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_3 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_19 = 0; j_1_19 < 16; j_1_19++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 48 + j_1_19) * 128 + k_row))) = _tmem_load_9[j_1_19];
                    }
                }
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], taddr + (unsigned int)h_row_base + 64);
                float _tmem_load_36_4[16];
                tmem_ld_x16(&_tmem_load_36_4[0], taddr + 256 + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_20 = 0; j_1_20 < 16; j_1_20++) {
                    _tmem_load_10[j_1_20] = _tmem_load_10[j_1_20] + _tmem_load_36_4[j_1_20];
                }
                if (write_transposed == 1) {
                    long long row_at_1_4 = tile + (long long)(k_row * 128 + v0 + 64);
                    {
                        float4 _v4 = make_float4(_tmem_load_10[0 + 0], _tmem_load_10[0 + 1], _tmem_load_10[0 + 2], _tmem_load_10[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_10[4 + 0], _tmem_load_10[4 + 1], _tmem_load_10[4 + 2], _tmem_load_10[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_4 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_10[8 + 0], _tmem_load_10[8 + 1], _tmem_load_10[8 + 2], _tmem_load_10[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_4 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_10[12 + 0], _tmem_load_10[12 + 1], _tmem_load_10[12 + 2], _tmem_load_10[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_4 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_21 = 0; j_1_21 < 16; j_1_21++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 64 + j_1_21) * 128 + k_row))) = _tmem_load_10[j_1_21];
                    }
                }
                float _tmem_load_11[16];
                tmem_ld_x16(&_tmem_load_11[0], taddr + (unsigned int)h_row_base + 80);
                float _tmem_load_36_5[16];
                tmem_ld_x16(&_tmem_load_36_5[0], taddr + 256 + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_22 = 0; j_1_22 < 16; j_1_22++) {
                    _tmem_load_11[j_1_22] = _tmem_load_11[j_1_22] + _tmem_load_36_5[j_1_22];
                }
                if (write_transposed == 1) {
                    long long row_at_1_5 = tile + (long long)(k_row * 128 + v0 + 80);
                    {
                        float4 _v4 = make_float4(_tmem_load_11[0 + 0], _tmem_load_11[0 + 1], _tmem_load_11[0 + 2], _tmem_load_11[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_5) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[4 + 0], _tmem_load_11[4 + 1], _tmem_load_11[4 + 2], _tmem_load_11[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_5 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[8 + 0], _tmem_load_11[8 + 1], _tmem_load_11[8 + 2], _tmem_load_11[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_5 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_11[12 + 0], _tmem_load_11[12 + 1], _tmem_load_11[12 + 2], _tmem_load_11[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_5 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_23 = 0; j_1_23 < 16; j_1_23++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 80 + j_1_23) * 128 + k_row))) = _tmem_load_11[j_1_23];
                    }
                }
                float _tmem_load_12[16];
                tmem_ld_x16(&_tmem_load_12[0], taddr + (unsigned int)h_row_base + 96);
                float _tmem_load_36_6[16];
                tmem_ld_x16(&_tmem_load_36_6[0], taddr + 256 + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_24 = 0; j_1_24 < 16; j_1_24++) {
                    _tmem_load_12[j_1_24] = _tmem_load_12[j_1_24] + _tmem_load_36_6[j_1_24];
                }
                if (write_transposed == 1) {
                    long long row_at_1_6 = tile + (long long)(k_row * 128 + v0 + 96);
                    {
                        float4 _v4 = make_float4(_tmem_load_12[0 + 0], _tmem_load_12[0 + 1], _tmem_load_12[0 + 2], _tmem_load_12[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_6) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_12[4 + 0], _tmem_load_12[4 + 1], _tmem_load_12[4 + 2], _tmem_load_12[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_6 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_12[8 + 0], _tmem_load_12[8 + 1], _tmem_load_12[8 + 2], _tmem_load_12[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_6 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_12[12 + 0], _tmem_load_12[12 + 1], _tmem_load_12[12 + 2], _tmem_load_12[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_6 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_25 = 0; j_1_25 < 16; j_1_25++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 96 + j_1_25) * 128 + k_row))) = _tmem_load_12[j_1_25];
                    }
                }
                float _tmem_load_13[16];
                tmem_ld_x16(&_tmem_load_13[0], taddr + (unsigned int)h_row_base + 112);
                float _tmem_load_36_7[16];
                tmem_ld_x16(&_tmem_load_36_7[0], taddr + 256 + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1_26 = 0; j_1_26 < 16; j_1_26++) {
                    _tmem_load_13[j_1_26] = _tmem_load_13[j_1_26] + _tmem_load_36_7[j_1_26];
                }
                if (write_transposed == 1) {
                    long long row_at_1_7 = tile + (long long)(k_row * 128 + v0 + 112);
                    {
                        float4 _v4 = make_float4(_tmem_load_13[0 + 0], _tmem_load_13[0 + 1], _tmem_load_13[0 + 2], _tmem_load_13[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_7) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_13[4 + 0], _tmem_load_13[4 + 1], _tmem_load_13[4 + 2], _tmem_load_13[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_7 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_13[8 + 0], _tmem_load_13[8 + 1], _tmem_load_13[8 + 2], _tmem_load_13[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_7 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_13[12 + 0], _tmem_load_13[12 + 1], _tmem_load_13[12 + 2], _tmem_load_13[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1_7 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_1_27 = 0; j_1_27 < 16; j_1_27++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 112 + j_1_27) * 128 + k_row))) = _tmem_load_13[j_1_27];
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int vslice_1 = 0;
            int head_1 = blockIdx.x % num_heads;
            int seg_1 = walk_seg[blockIdx.x / num_heads];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nc_1; c_1++) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                    int trow_1 = row0_1 + c_1 * 64;
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)half) * 8192, (&w_tma), half * 64, head_1, trow_1, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)half) * 8192, (&kg_tma), half * 64, head_1, trow_1, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int vslice_2 = 0;
            int head_2 = blockIdx.x % num_heads;
            int seg_2 = walk_seg[blockIdx.x / num_heads];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                    _phase_hs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x7FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn2_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn2_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 69272720;\n\t"
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
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((w_op_addr) >> 4) & 0x7FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((hs_mn1_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_2 = (((w_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((hs_mn1_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 69272720;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_vp), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((w_op_addr) >> 4) & 0x7FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_5 = (((hs_mn0_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_4 = (((w_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_5 = (((hs_mn0_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 69272720;\n\t"
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc_vp), "r"(1) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_7 = (((bv_mn2_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_6 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_7 = (((bv_mn2_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 136414352;\n\t"
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_dx), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_9 = (((bv_mn1_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_8 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_9 = (((bv_mn1_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 136414352;\n\t"
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
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tmem_acc_dx), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_11 = (((bv_mn0_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_10 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_11 = (((bv_mn0_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 136414352;\n\t"
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
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_acc_dx), "r"(1) : "memory");
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
