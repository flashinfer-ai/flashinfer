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
kernel_cake_kda_train_b1a95a8ee8110e47558e(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, int init_mode, int use_u, int write_transposed)
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
            for (int j = 0; j < 16; j++) {
                float one_here = 0.0f;
                if (init_mode == 0) {
                    if (v0 + j == k_row) {
                        one_here = 1.0f;
                    }
                }
                init_row[j] = one_here;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
            #pragma unroll
            for (int j_1 = 0; j_1 < 16; j_1++) {
                float one_here_1 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 16 + j_1 == k_row) {
                        one_here_1 = 1.0f;
                    }
                }
                init_row[j_1] = one_here_1;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
            #pragma unroll
            for (int j_2 = 0; j_2 < 16; j_2++) {
                float one_here_2 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 32 + j_2 == k_row) {
                        one_here_2 = 1.0f;
                    }
                }
                init_row[j_2] = one_here_2;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
            #pragma unroll
            for (int j_3 = 0; j_3 < 16; j_3++) {
                float one_here_3 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 48 + j_3 == k_row) {
                        one_here_3 = 1.0f;
                    }
                }
                init_row[j_3] = one_here_3;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
            #pragma unroll
            for (int j_4 = 0; j_4 < 16; j_4++) {
                float one_here_4 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 64 + j_4 == k_row) {
                        one_here_4 = 1.0f;
                    }
                }
                init_row[j_4] = one_here_4;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
            #pragma unroll
            for (int j_5 = 0; j_5 < 16; j_5++) {
                float one_here_5 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 80 + j_5 == k_row) {
                        one_here_5 = 1.0f;
                    }
                }
                init_row[j_5] = one_here_5;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
            #pragma unroll
            for (int j_6 = 0; j_6 < 16; j_6++) {
                float one_here_6 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 96 + j_6 == k_row) {
                        one_here_6 = 1.0f;
                    }
                }
                init_row[j_6] = one_here_6;
            }
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
            #pragma unroll
            for (int j_7 = 0; j_7 < 16; j_7++) {
                float one_here_7 = 0.0f;
                if (init_mode == 0) {
                    if (v0 + 112 + j_7 == k_row) {
                        one_here_7 = 1.0f;
                    }
                }
                init_row[j_7] = one_here_7;
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
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)h_row_base);
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + 256 + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    _tmem_load_0[j_8] = _tmem_load_0[j_8] + _tmem_load_1[j_8];
                }
                unsigned int _tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1[16];
                r1[0] = _tmem_load_0[0] - __uint_as_float(_tmem_load_0_bf16[0] << 16);
                r1[1] = _tmem_load_0[1] - __uint_as_float(_tmem_load_0_bf16[0] & mask_hi);
                r1[2] = _tmem_load_0[2] - __uint_as_float(_tmem_load_0_bf16[1] << 16);
                r1[3] = _tmem_load_0[3] - __uint_as_float(_tmem_load_0_bf16[1] & mask_hi);
                r1[4] = _tmem_load_0[4] - __uint_as_float(_tmem_load_0_bf16[2] << 16);
                r1[5] = _tmem_load_0[5] - __uint_as_float(_tmem_load_0_bf16[2] & mask_hi);
                r1[6] = _tmem_load_0[6] - __uint_as_float(_tmem_load_0_bf16[3] << 16);
                r1[7] = _tmem_load_0[7] - __uint_as_float(_tmem_load_0_bf16[3] & mask_hi);
                r1[8] = _tmem_load_0[8] - __uint_as_float(_tmem_load_0_bf16[4] << 16);
                r1[9] = _tmem_load_0[9] - __uint_as_float(_tmem_load_0_bf16[4] & mask_hi);
                r1[10] = _tmem_load_0[10] - __uint_as_float(_tmem_load_0_bf16[5] << 16);
                r1[11] = _tmem_load_0[11] - __uint_as_float(_tmem_load_0_bf16[5] & mask_hi);
                r1[12] = _tmem_load_0[12] - __uint_as_float(_tmem_load_0_bf16[6] << 16);
                r1[13] = _tmem_load_0[13] - __uint_as_float(_tmem_load_0_bf16[6] & mask_hi);
                r1[14] = _tmem_load_0[14] - __uint_as_float(_tmem_load_0_bf16[7] << 16);
                r1[15] = _tmem_load_0[15] - __uint_as_float(_tmem_load_0_bf16[7] & mask_hi);
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
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(r1_bf16[0]), "r"(r1_bf16[1]), "r"(r1_bf16[2]), "r"(r1_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r1_bf16[4]), "r"(r1_bf16[5]), "r"(r1_bf16[6]), "r"(r1_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 ^ (k_row * 128 >> 7 & 7) << 4))), "r"(r2_bf16[0]), "r"(r2_bf16[1]), "r"(r2_bf16[2]), "r"(r2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 16 ^ (k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r2_bf16[4]), "r"(r2_bf16[5]), "r"(r2_bf16[6]), "r"(r2_bf16[7]) : "memory");
                float hs2_0[16];
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    hs2_0[j_9] = _tmem_load_0[j_9] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, hs2_0);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)h_row_base + 16);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + 256 + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    _tmem_load_2[j_10] = _tmem_load_2[j_10] + _tmem_load_3[j_10];
                }
                unsigned int _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_1[16];
                r1_1[0] = _tmem_load_2[0] - __uint_as_float(_tmem_load_2_bf16[0] << 16);
                r1_1[1] = _tmem_load_2[1] - __uint_as_float(_tmem_load_2_bf16[0] & mask_hi);
                r1_1[2] = _tmem_load_2[2] - __uint_as_float(_tmem_load_2_bf16[1] << 16);
                r1_1[3] = _tmem_load_2[3] - __uint_as_float(_tmem_load_2_bf16[1] & mask_hi);
                r1_1[4] = _tmem_load_2[4] - __uint_as_float(_tmem_load_2_bf16[2] << 16);
                r1_1[5] = _tmem_load_2[5] - __uint_as_float(_tmem_load_2_bf16[2] & mask_hi);
                r1_1[6] = _tmem_load_2[6] - __uint_as_float(_tmem_load_2_bf16[3] << 16);
                r1_1[7] = _tmem_load_2[7] - __uint_as_float(_tmem_load_2_bf16[3] & mask_hi);
                r1_1[8] = _tmem_load_2[8] - __uint_as_float(_tmem_load_2_bf16[4] << 16);
                r1_1[9] = _tmem_load_2[9] - __uint_as_float(_tmem_load_2_bf16[4] & mask_hi);
                r1_1[10] = _tmem_load_2[10] - __uint_as_float(_tmem_load_2_bf16[5] << 16);
                r1_1[11] = _tmem_load_2[11] - __uint_as_float(_tmem_load_2_bf16[5] & mask_hi);
                r1_1[12] = _tmem_load_2[12] - __uint_as_float(_tmem_load_2_bf16[6] << 16);
                r1_1[13] = _tmem_load_2[13] - __uint_as_float(_tmem_load_2_bf16[6] & mask_hi);
                r1_1[14] = _tmem_load_2[14] - __uint_as_float(_tmem_load_2_bf16[7] << 16);
                r1_1[15] = _tmem_load_2[15] - __uint_as_float(_tmem_load_2_bf16[7] & mask_hi);
                unsigned int r1_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_1[_lp*2 + 0], r1_1[_lp*2+1 + 0]));
                    r1_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_2[16];
                r2_2[0] = r1_1[0] - __uint_as_float(r1_1_bf16[0] << 16);
                r2_2[1] = r1_1[1] - __uint_as_float(r1_1_bf16[0] & mask_hi);
                r2_2[2] = r1_1[2] - __uint_as_float(r1_1_bf16[1] << 16);
                r2_2[3] = r1_1[3] - __uint_as_float(r1_1_bf16[1] & mask_hi);
                r2_2[4] = r1_1[4] - __uint_as_float(r1_1_bf16[2] << 16);
                r2_2[5] = r1_1[5] - __uint_as_float(r1_1_bf16[2] & mask_hi);
                r2_2[6] = r1_1[6] - __uint_as_float(r1_1_bf16[3] << 16);
                r2_2[7] = r1_1[7] - __uint_as_float(r1_1_bf16[3] & mask_hi);
                r2_2[8] = r1_1[8] - __uint_as_float(r1_1_bf16[4] << 16);
                r2_2[9] = r1_1[9] - __uint_as_float(r1_1_bf16[4] & mask_hi);
                r2_2[10] = r1_1[10] - __uint_as_float(r1_1_bf16[5] << 16);
                r2_2[11] = r1_1[11] - __uint_as_float(r1_1_bf16[5] & mask_hi);
                r2_2[12] = r1_1[12] - __uint_as_float(r1_1_bf16[6] << 16);
                r2_2[13] = r1_1[13] - __uint_as_float(r1_1_bf16[6] & mask_hi);
                r2_2[14] = r1_1[14] - __uint_as_float(r1_1_bf16[7] << 16);
                r2_2[15] = r1_1[15] - __uint_as_float(r1_1_bf16[7] & mask_hi);
                unsigned int r2_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_2[_lp*2 + 0], r2_2[_lp*2+1 + 0]));
                    r2_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r1_1_bf16[0]), "r"(r1_1_bf16[1]), "r"(r1_1_bf16[2]), "r"(r1_1_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r1_1_bf16[4]), "r"(r1_1_bf16[5]), "r"(r1_1_bf16[6]), "r"(r1_1_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 32 ^ (k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r2_2_bf16[0]), "r"(r2_2_bf16[1]), "r"(r2_2_bf16[2]), "r"(r2_2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 48 ^ (k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r2_2_bf16[4]), "r"(r2_2_bf16[5]), "r"(r2_2_bf16[6]), "r"(r2_2_bf16[7]) : "memory");
                float hs2_3[16];
                #pragma unroll
                for (int j_11 = 0; j_11 < 16; j_11++) {
                    hs2_3[j_11] = _tmem_load_2[j_11] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, hs2_3);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)h_row_base + 32);
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + 256 + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_12 = 0; j_12 < 16; j_12++) {
                    _tmem_load_4[j_12] = _tmem_load_4[j_12] + _tmem_load_5[j_12];
                }
                unsigned int _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_4[16];
                r1_4[0] = _tmem_load_4[0] - __uint_as_float(_tmem_load_4_bf16[0] << 16);
                r1_4[1] = _tmem_load_4[1] - __uint_as_float(_tmem_load_4_bf16[0] & mask_hi);
                r1_4[2] = _tmem_load_4[2] - __uint_as_float(_tmem_load_4_bf16[1] << 16);
                r1_4[3] = _tmem_load_4[3] - __uint_as_float(_tmem_load_4_bf16[1] & mask_hi);
                r1_4[4] = _tmem_load_4[4] - __uint_as_float(_tmem_load_4_bf16[2] << 16);
                r1_4[5] = _tmem_load_4[5] - __uint_as_float(_tmem_load_4_bf16[2] & mask_hi);
                r1_4[6] = _tmem_load_4[6] - __uint_as_float(_tmem_load_4_bf16[3] << 16);
                r1_4[7] = _tmem_load_4[7] - __uint_as_float(_tmem_load_4_bf16[3] & mask_hi);
                r1_4[8] = _tmem_load_4[8] - __uint_as_float(_tmem_load_4_bf16[4] << 16);
                r1_4[9] = _tmem_load_4[9] - __uint_as_float(_tmem_load_4_bf16[4] & mask_hi);
                r1_4[10] = _tmem_load_4[10] - __uint_as_float(_tmem_load_4_bf16[5] << 16);
                r1_4[11] = _tmem_load_4[11] - __uint_as_float(_tmem_load_4_bf16[5] & mask_hi);
                r1_4[12] = _tmem_load_4[12] - __uint_as_float(_tmem_load_4_bf16[6] << 16);
                r1_4[13] = _tmem_load_4[13] - __uint_as_float(_tmem_load_4_bf16[6] & mask_hi);
                r1_4[14] = _tmem_load_4[14] - __uint_as_float(_tmem_load_4_bf16[7] << 16);
                r1_4[15] = _tmem_load_4[15] - __uint_as_float(_tmem_load_4_bf16[7] & mask_hi);
                unsigned int r1_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_4[_lp*2 + 0], r1_4[_lp*2+1 + 0]));
                    r1_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_5[16];
                r2_5[0] = r1_4[0] - __uint_as_float(r1_4_bf16[0] << 16);
                r2_5[1] = r1_4[1] - __uint_as_float(r1_4_bf16[0] & mask_hi);
                r2_5[2] = r1_4[2] - __uint_as_float(r1_4_bf16[1] << 16);
                r2_5[3] = r1_4[3] - __uint_as_float(r1_4_bf16[1] & mask_hi);
                r2_5[4] = r1_4[4] - __uint_as_float(r1_4_bf16[2] << 16);
                r2_5[5] = r1_4[5] - __uint_as_float(r1_4_bf16[2] & mask_hi);
                r2_5[6] = r1_4[6] - __uint_as_float(r1_4_bf16[3] << 16);
                r2_5[7] = r1_4[7] - __uint_as_float(r1_4_bf16[3] & mask_hi);
                r2_5[8] = r1_4[8] - __uint_as_float(r1_4_bf16[4] << 16);
                r2_5[9] = r1_4[9] - __uint_as_float(r1_4_bf16[4] & mask_hi);
                r2_5[10] = r1_4[10] - __uint_as_float(r1_4_bf16[5] << 16);
                r2_5[11] = r1_4[11] - __uint_as_float(r1_4_bf16[5] & mask_hi);
                r2_5[12] = r1_4[12] - __uint_as_float(r1_4_bf16[6] << 16);
                r2_5[13] = r1_4[13] - __uint_as_float(r1_4_bf16[6] & mask_hi);
                r2_5[14] = r1_4[14] - __uint_as_float(r1_4_bf16[7] << 16);
                r2_5[15] = r1_4[15] - __uint_as_float(r1_4_bf16[7] & mask_hi);
                unsigned int r2_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_5[_lp*2 + 0], r2_5[_lp*2+1 + 0]));
                    r2_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[0]), "r"(_tmem_load_4_bf16[1]), "r"(_tmem_load_4_bf16[2]), "r"(_tmem_load_4_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[4]), "r"(_tmem_load_4_bf16[5]), "r"(_tmem_load_4_bf16[6]), "r"(_tmem_load_4_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r1_4_bf16[0]), "r"(r1_4_bf16[1]), "r"(r1_4_bf16[2]), "r"(r1_4_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r1_4_bf16[4]), "r"(r1_4_bf16[5]), "r"(r1_4_bf16[6]), "r"(r1_4_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 64 ^ (k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r2_5_bf16[0]), "r"(r2_5_bf16[1]), "r"(r2_5_bf16[2]), "r"(r2_5_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 80 ^ (k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r2_5_bf16[4]), "r"(r2_5_bf16[5]), "r"(r2_5_bf16[6]), "r"(r2_5_bf16[7]) : "memory");
                float hs2_6[16];
                #pragma unroll
                for (int j_13 = 0; j_13 < 16; j_13++) {
                    hs2_6[j_13] = _tmem_load_4[j_13] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, hs2_6);
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)h_row_base + 48);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + 256 + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_14 = 0; j_14 < 16; j_14++) {
                    _tmem_load_6[j_14] = _tmem_load_6[j_14] + _tmem_load_7[j_14];
                }
                unsigned int _tmem_load_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_7[16];
                r1_7[0] = _tmem_load_6[0] - __uint_as_float(_tmem_load_6_bf16[0] << 16);
                r1_7[1] = _tmem_load_6[1] - __uint_as_float(_tmem_load_6_bf16[0] & mask_hi);
                r1_7[2] = _tmem_load_6[2] - __uint_as_float(_tmem_load_6_bf16[1] << 16);
                r1_7[3] = _tmem_load_6[3] - __uint_as_float(_tmem_load_6_bf16[1] & mask_hi);
                r1_7[4] = _tmem_load_6[4] - __uint_as_float(_tmem_load_6_bf16[2] << 16);
                r1_7[5] = _tmem_load_6[5] - __uint_as_float(_tmem_load_6_bf16[2] & mask_hi);
                r1_7[6] = _tmem_load_6[6] - __uint_as_float(_tmem_load_6_bf16[3] << 16);
                r1_7[7] = _tmem_load_6[7] - __uint_as_float(_tmem_load_6_bf16[3] & mask_hi);
                r1_7[8] = _tmem_load_6[8] - __uint_as_float(_tmem_load_6_bf16[4] << 16);
                r1_7[9] = _tmem_load_6[9] - __uint_as_float(_tmem_load_6_bf16[4] & mask_hi);
                r1_7[10] = _tmem_load_6[10] - __uint_as_float(_tmem_load_6_bf16[5] << 16);
                r1_7[11] = _tmem_load_6[11] - __uint_as_float(_tmem_load_6_bf16[5] & mask_hi);
                r1_7[12] = _tmem_load_6[12] - __uint_as_float(_tmem_load_6_bf16[6] << 16);
                r1_7[13] = _tmem_load_6[13] - __uint_as_float(_tmem_load_6_bf16[6] & mask_hi);
                r1_7[14] = _tmem_load_6[14] - __uint_as_float(_tmem_load_6_bf16[7] << 16);
                r1_7[15] = _tmem_load_6[15] - __uint_as_float(_tmem_load_6_bf16[7] & mask_hi);
                unsigned int r1_7_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_7[_lp*2 + 0], r1_7[_lp*2+1 + 0]));
                    r1_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_8[16];
                r2_8[0] = r1_7[0] - __uint_as_float(r1_7_bf16[0] << 16);
                r2_8[1] = r1_7[1] - __uint_as_float(r1_7_bf16[0] & mask_hi);
                r2_8[2] = r1_7[2] - __uint_as_float(r1_7_bf16[1] << 16);
                r2_8[3] = r1_7[3] - __uint_as_float(r1_7_bf16[1] & mask_hi);
                r2_8[4] = r1_7[4] - __uint_as_float(r1_7_bf16[2] << 16);
                r2_8[5] = r1_7[5] - __uint_as_float(r1_7_bf16[2] & mask_hi);
                r2_8[6] = r1_7[6] - __uint_as_float(r1_7_bf16[3] << 16);
                r2_8[7] = r1_7[7] - __uint_as_float(r1_7_bf16[3] & mask_hi);
                r2_8[8] = r1_7[8] - __uint_as_float(r1_7_bf16[4] << 16);
                r2_8[9] = r1_7[9] - __uint_as_float(r1_7_bf16[4] & mask_hi);
                r2_8[10] = r1_7[10] - __uint_as_float(r1_7_bf16[5] << 16);
                r2_8[11] = r1_7[11] - __uint_as_float(r1_7_bf16[5] & mask_hi);
                r2_8[12] = r1_7[12] - __uint_as_float(r1_7_bf16[6] << 16);
                r2_8[13] = r1_7[13] - __uint_as_float(r1_7_bf16[6] & mask_hi);
                r2_8[14] = r1_7[14] - __uint_as_float(r1_7_bf16[7] << 16);
                r2_8[15] = r1_7[15] - __uint_as_float(r1_7_bf16[7] & mask_hi);
                unsigned int r2_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_8[_lp*2 + 0], r2_8[_lp*2+1 + 0]));
                    r2_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[0]), "r"(_tmem_load_6_bf16[1]), "r"(_tmem_load_6_bf16[2]), "r"(_tmem_load_6_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[4]), "r"(_tmem_load_6_bf16[5]), "r"(_tmem_load_6_bf16[6]), "r"(_tmem_load_6_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r1_7_bf16[0]), "r"(r1_7_bf16[1]), "r"(r1_7_bf16[2]), "r"(r1_7_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r1_7_bf16[4]), "r"(r1_7_bf16[5]), "r"(r1_7_bf16[6]), "r"(r1_7_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 96 ^ (k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r2_8_bf16[0]), "r"(r2_8_bf16[1]), "r"(r2_8_bf16[2]), "r"(r2_8_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(k_row * 128 + 112 ^ (k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r2_8_bf16[4]), "r"(r2_8_bf16[5]), "r"(r2_8_bf16[6]), "r"(r2_8_bf16[7]) : "memory");
                float hs2_9[16];
                #pragma unroll
                for (int j_15 = 0; j_15 < 16; j_15++) {
                    hs2_9[j_15] = _tmem_load_6[j_15] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, hs2_9);
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], taddr + (unsigned int)h_row_base + 64);
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], taddr + 256 + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_16 = 0; j_16 < 16; j_16++) {
                    _tmem_load_8[j_16] = _tmem_load_8[j_16] + _tmem_load_9[j_16];
                }
                unsigned int _tmem_load_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                    _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_10[16];
                r1_10[0] = _tmem_load_8[0] - __uint_as_float(_tmem_load_8_bf16[0] << 16);
                r1_10[1] = _tmem_load_8[1] - __uint_as_float(_tmem_load_8_bf16[0] & mask_hi);
                r1_10[2] = _tmem_load_8[2] - __uint_as_float(_tmem_load_8_bf16[1] << 16);
                r1_10[3] = _tmem_load_8[3] - __uint_as_float(_tmem_load_8_bf16[1] & mask_hi);
                r1_10[4] = _tmem_load_8[4] - __uint_as_float(_tmem_load_8_bf16[2] << 16);
                r1_10[5] = _tmem_load_8[5] - __uint_as_float(_tmem_load_8_bf16[2] & mask_hi);
                r1_10[6] = _tmem_load_8[6] - __uint_as_float(_tmem_load_8_bf16[3] << 16);
                r1_10[7] = _tmem_load_8[7] - __uint_as_float(_tmem_load_8_bf16[3] & mask_hi);
                r1_10[8] = _tmem_load_8[8] - __uint_as_float(_tmem_load_8_bf16[4] << 16);
                r1_10[9] = _tmem_load_8[9] - __uint_as_float(_tmem_load_8_bf16[4] & mask_hi);
                r1_10[10] = _tmem_load_8[10] - __uint_as_float(_tmem_load_8_bf16[5] << 16);
                r1_10[11] = _tmem_load_8[11] - __uint_as_float(_tmem_load_8_bf16[5] & mask_hi);
                r1_10[12] = _tmem_load_8[12] - __uint_as_float(_tmem_load_8_bf16[6] << 16);
                r1_10[13] = _tmem_load_8[13] - __uint_as_float(_tmem_load_8_bf16[6] & mask_hi);
                r1_10[14] = _tmem_load_8[14] - __uint_as_float(_tmem_load_8_bf16[7] << 16);
                r1_10[15] = _tmem_load_8[15] - __uint_as_float(_tmem_load_8_bf16[7] & mask_hi);
                unsigned int r1_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_10[_lp*2 + 0], r1_10[_lp*2+1 + 0]));
                    r1_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_11[16];
                r2_11[0] = r1_10[0] - __uint_as_float(r1_10_bf16[0] << 16);
                r2_11[1] = r1_10[1] - __uint_as_float(r1_10_bf16[0] & mask_hi);
                r2_11[2] = r1_10[2] - __uint_as_float(r1_10_bf16[1] << 16);
                r2_11[3] = r1_10[3] - __uint_as_float(r1_10_bf16[1] & mask_hi);
                r2_11[4] = r1_10[4] - __uint_as_float(r1_10_bf16[2] << 16);
                r2_11[5] = r1_10[5] - __uint_as_float(r1_10_bf16[2] & mask_hi);
                r2_11[6] = r1_10[6] - __uint_as_float(r1_10_bf16[3] << 16);
                r2_11[7] = r1_10[7] - __uint_as_float(r1_10_bf16[3] & mask_hi);
                r2_11[8] = r1_10[8] - __uint_as_float(r1_10_bf16[4] << 16);
                r2_11[9] = r1_10[9] - __uint_as_float(r1_10_bf16[4] & mask_hi);
                r2_11[10] = r1_10[10] - __uint_as_float(r1_10_bf16[5] << 16);
                r2_11[11] = r1_10[11] - __uint_as_float(r1_10_bf16[5] & mask_hi);
                r2_11[12] = r1_10[12] - __uint_as_float(r1_10_bf16[6] << 16);
                r2_11[13] = r1_10[13] - __uint_as_float(r1_10_bf16[6] & mask_hi);
                r2_11[14] = r1_10[14] - __uint_as_float(r1_10_bf16[7] << 16);
                r2_11[15] = r1_10[15] - __uint_as_float(r1_10_bf16[7] & mask_hi);
                unsigned int r2_11_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_11[_lp*2 + 0], r2_11[_lp*2+1 + 0]));
                    r2_11_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 ^ (16384 + k_row * 128 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[0]), "r"(_tmem_load_8_bf16[1]), "r"(_tmem_load_8_bf16[2]), "r"(_tmem_load_8_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 16 ^ (16384 + k_row * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[4]), "r"(_tmem_load_8_bf16[5]), "r"(_tmem_load_8_bf16[6]), "r"(_tmem_load_8_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 ^ (16384 + k_row * 128 >> 7 & 7) << 4))), "r"(r1_10_bf16[0]), "r"(r1_10_bf16[1]), "r"(r1_10_bf16[2]), "r"(r1_10_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 16 ^ (16384 + k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r1_10_bf16[4]), "r"(r1_10_bf16[5]), "r"(r1_10_bf16[6]), "r"(r1_10_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 ^ (16384 + k_row * 128 >> 7 & 7) << 4))), "r"(r2_11_bf16[0]), "r"(r2_11_bf16[1]), "r"(r2_11_bf16[2]), "r"(r2_11_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 16 ^ (16384 + k_row * 128 + 16 >> 7 & 7) << 4))), "r"(r2_11_bf16[4]), "r"(r2_11_bf16[5]), "r"(r2_11_bf16[6]), "r"(r2_11_bf16[7]) : "memory");
                float hs2_12[16];
                #pragma unroll
                for (int j_17 = 0; j_17 < 16; j_17++) {
                    hs2_12[j_17] = _tmem_load_8[j_17] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, hs2_12);
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], taddr + (unsigned int)h_row_base + 80);
                float _tmem_load_11[16];
                tmem_ld_x16(&_tmem_load_11[0], taddr + 256 + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_18 = 0; j_18 < 16; j_18++) {
                    _tmem_load_10[j_18] = _tmem_load_10[j_18] + _tmem_load_11[j_18];
                }
                unsigned int _tmem_load_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                    _tmem_load_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_13[16];
                r1_13[0] = _tmem_load_10[0] - __uint_as_float(_tmem_load_10_bf16[0] << 16);
                r1_13[1] = _tmem_load_10[1] - __uint_as_float(_tmem_load_10_bf16[0] & mask_hi);
                r1_13[2] = _tmem_load_10[2] - __uint_as_float(_tmem_load_10_bf16[1] << 16);
                r1_13[3] = _tmem_load_10[3] - __uint_as_float(_tmem_load_10_bf16[1] & mask_hi);
                r1_13[4] = _tmem_load_10[4] - __uint_as_float(_tmem_load_10_bf16[2] << 16);
                r1_13[5] = _tmem_load_10[5] - __uint_as_float(_tmem_load_10_bf16[2] & mask_hi);
                r1_13[6] = _tmem_load_10[6] - __uint_as_float(_tmem_load_10_bf16[3] << 16);
                r1_13[7] = _tmem_load_10[7] - __uint_as_float(_tmem_load_10_bf16[3] & mask_hi);
                r1_13[8] = _tmem_load_10[8] - __uint_as_float(_tmem_load_10_bf16[4] << 16);
                r1_13[9] = _tmem_load_10[9] - __uint_as_float(_tmem_load_10_bf16[4] & mask_hi);
                r1_13[10] = _tmem_load_10[10] - __uint_as_float(_tmem_load_10_bf16[5] << 16);
                r1_13[11] = _tmem_load_10[11] - __uint_as_float(_tmem_load_10_bf16[5] & mask_hi);
                r1_13[12] = _tmem_load_10[12] - __uint_as_float(_tmem_load_10_bf16[6] << 16);
                r1_13[13] = _tmem_load_10[13] - __uint_as_float(_tmem_load_10_bf16[6] & mask_hi);
                r1_13[14] = _tmem_load_10[14] - __uint_as_float(_tmem_load_10_bf16[7] << 16);
                r1_13[15] = _tmem_load_10[15] - __uint_as_float(_tmem_load_10_bf16[7] & mask_hi);
                unsigned int r1_13_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_13[_lp*2 + 0], r1_13[_lp*2+1 + 0]));
                    r1_13_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_14[16];
                r2_14[0] = r1_13[0] - __uint_as_float(r1_13_bf16[0] << 16);
                r2_14[1] = r1_13[1] - __uint_as_float(r1_13_bf16[0] & mask_hi);
                r2_14[2] = r1_13[2] - __uint_as_float(r1_13_bf16[1] << 16);
                r2_14[3] = r1_13[3] - __uint_as_float(r1_13_bf16[1] & mask_hi);
                r2_14[4] = r1_13[4] - __uint_as_float(r1_13_bf16[2] << 16);
                r2_14[5] = r1_13[5] - __uint_as_float(r1_13_bf16[2] & mask_hi);
                r2_14[6] = r1_13[6] - __uint_as_float(r1_13_bf16[3] << 16);
                r2_14[7] = r1_13[7] - __uint_as_float(r1_13_bf16[3] & mask_hi);
                r2_14[8] = r1_13[8] - __uint_as_float(r1_13_bf16[4] << 16);
                r2_14[9] = r1_13[9] - __uint_as_float(r1_13_bf16[4] & mask_hi);
                r2_14[10] = r1_13[10] - __uint_as_float(r1_13_bf16[5] << 16);
                r2_14[11] = r1_13[11] - __uint_as_float(r1_13_bf16[5] & mask_hi);
                r2_14[12] = r1_13[12] - __uint_as_float(r1_13_bf16[6] << 16);
                r2_14[13] = r1_13[13] - __uint_as_float(r1_13_bf16[6] & mask_hi);
                r2_14[14] = r1_13[14] - __uint_as_float(r1_13_bf16[7] << 16);
                r2_14[15] = r1_13[15] - __uint_as_float(r1_13_bf16[7] & mask_hi);
                unsigned int r2_14_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_14[_lp*2 + 0], r2_14[_lp*2+1 + 0]));
                    r2_14_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 32 ^ (16384 + k_row * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_10_bf16[0]), "r"(_tmem_load_10_bf16[1]), "r"(_tmem_load_10_bf16[2]), "r"(_tmem_load_10_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 48 ^ (16384 + k_row * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_10_bf16[4]), "r"(_tmem_load_10_bf16[5]), "r"(_tmem_load_10_bf16[6]), "r"(_tmem_load_10_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 32 ^ (16384 + k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r1_13_bf16[0]), "r"(r1_13_bf16[1]), "r"(r1_13_bf16[2]), "r"(r1_13_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 48 ^ (16384 + k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r1_13_bf16[4]), "r"(r1_13_bf16[5]), "r"(r1_13_bf16[6]), "r"(r1_13_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 32 ^ (16384 + k_row * 128 + 32 >> 7 & 7) << 4))), "r"(r2_14_bf16[0]), "r"(r2_14_bf16[1]), "r"(r2_14_bf16[2]), "r"(r2_14_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 48 ^ (16384 + k_row * 128 + 48 >> 7 & 7) << 4))), "r"(r2_14_bf16[4]), "r"(r2_14_bf16[5]), "r"(r2_14_bf16[6]), "r"(r2_14_bf16[7]) : "memory");
                float hs2_15[16];
                #pragma unroll
                for (int j_19 = 0; j_19 < 16; j_19++) {
                    hs2_15[j_19] = _tmem_load_10[j_19] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, hs2_15);
                float _tmem_load_12[16];
                tmem_ld_x16(&_tmem_load_12[0], taddr + (unsigned int)h_row_base + 96);
                float _tmem_load_13[16];
                tmem_ld_x16(&_tmem_load_13[0], taddr + 256 + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_20 = 0; j_20 < 16; j_20++) {
                    _tmem_load_12[j_20] = _tmem_load_12[j_20] + _tmem_load_13[j_20];
                }
                unsigned int _tmem_load_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 0], _tmem_load_12[_lp*2+1 + 0]));
                    _tmem_load_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_16[16];
                r1_16[0] = _tmem_load_12[0] - __uint_as_float(_tmem_load_12_bf16[0] << 16);
                r1_16[1] = _tmem_load_12[1] - __uint_as_float(_tmem_load_12_bf16[0] & mask_hi);
                r1_16[2] = _tmem_load_12[2] - __uint_as_float(_tmem_load_12_bf16[1] << 16);
                r1_16[3] = _tmem_load_12[3] - __uint_as_float(_tmem_load_12_bf16[1] & mask_hi);
                r1_16[4] = _tmem_load_12[4] - __uint_as_float(_tmem_load_12_bf16[2] << 16);
                r1_16[5] = _tmem_load_12[5] - __uint_as_float(_tmem_load_12_bf16[2] & mask_hi);
                r1_16[6] = _tmem_load_12[6] - __uint_as_float(_tmem_load_12_bf16[3] << 16);
                r1_16[7] = _tmem_load_12[7] - __uint_as_float(_tmem_load_12_bf16[3] & mask_hi);
                r1_16[8] = _tmem_load_12[8] - __uint_as_float(_tmem_load_12_bf16[4] << 16);
                r1_16[9] = _tmem_load_12[9] - __uint_as_float(_tmem_load_12_bf16[4] & mask_hi);
                r1_16[10] = _tmem_load_12[10] - __uint_as_float(_tmem_load_12_bf16[5] << 16);
                r1_16[11] = _tmem_load_12[11] - __uint_as_float(_tmem_load_12_bf16[5] & mask_hi);
                r1_16[12] = _tmem_load_12[12] - __uint_as_float(_tmem_load_12_bf16[6] << 16);
                r1_16[13] = _tmem_load_12[13] - __uint_as_float(_tmem_load_12_bf16[6] & mask_hi);
                r1_16[14] = _tmem_load_12[14] - __uint_as_float(_tmem_load_12_bf16[7] << 16);
                r1_16[15] = _tmem_load_12[15] - __uint_as_float(_tmem_load_12_bf16[7] & mask_hi);
                unsigned int r1_16_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_16[_lp*2 + 0], r1_16[_lp*2+1 + 0]));
                    r1_16_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_17[16];
                r2_17[0] = r1_16[0] - __uint_as_float(r1_16_bf16[0] << 16);
                r2_17[1] = r1_16[1] - __uint_as_float(r1_16_bf16[0] & mask_hi);
                r2_17[2] = r1_16[2] - __uint_as_float(r1_16_bf16[1] << 16);
                r2_17[3] = r1_16[3] - __uint_as_float(r1_16_bf16[1] & mask_hi);
                r2_17[4] = r1_16[4] - __uint_as_float(r1_16_bf16[2] << 16);
                r2_17[5] = r1_16[5] - __uint_as_float(r1_16_bf16[2] & mask_hi);
                r2_17[6] = r1_16[6] - __uint_as_float(r1_16_bf16[3] << 16);
                r2_17[7] = r1_16[7] - __uint_as_float(r1_16_bf16[3] & mask_hi);
                r2_17[8] = r1_16[8] - __uint_as_float(r1_16_bf16[4] << 16);
                r2_17[9] = r1_16[9] - __uint_as_float(r1_16_bf16[4] & mask_hi);
                r2_17[10] = r1_16[10] - __uint_as_float(r1_16_bf16[5] << 16);
                r2_17[11] = r1_16[11] - __uint_as_float(r1_16_bf16[5] & mask_hi);
                r2_17[12] = r1_16[12] - __uint_as_float(r1_16_bf16[6] << 16);
                r2_17[13] = r1_16[13] - __uint_as_float(r1_16_bf16[6] & mask_hi);
                r2_17[14] = r1_16[14] - __uint_as_float(r1_16_bf16[7] << 16);
                r2_17[15] = r1_16[15] - __uint_as_float(r1_16_bf16[7] & mask_hi);
                unsigned int r2_17_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_17[_lp*2 + 0], r2_17[_lp*2+1 + 0]));
                    r2_17_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 64 ^ (16384 + k_row * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_12_bf16[0]), "r"(_tmem_load_12_bf16[1]), "r"(_tmem_load_12_bf16[2]), "r"(_tmem_load_12_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 80 ^ (16384 + k_row * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_12_bf16[4]), "r"(_tmem_load_12_bf16[5]), "r"(_tmem_load_12_bf16[6]), "r"(_tmem_load_12_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 64 ^ (16384 + k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r1_16_bf16[0]), "r"(r1_16_bf16[1]), "r"(r1_16_bf16[2]), "r"(r1_16_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 80 ^ (16384 + k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r1_16_bf16[4]), "r"(r1_16_bf16[5]), "r"(r1_16_bf16[6]), "r"(r1_16_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 64 ^ (16384 + k_row * 128 + 64 >> 7 & 7) << 4))), "r"(r2_17_bf16[0]), "r"(r2_17_bf16[1]), "r"(r2_17_bf16[2]), "r"(r2_17_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 80 ^ (16384 + k_row * 128 + 80 >> 7 & 7) << 4))), "r"(r2_17_bf16[4]), "r"(r2_17_bf16[5]), "r"(r2_17_bf16[6]), "r"(r2_17_bf16[7]) : "memory");
                float hs2_18[16];
                #pragma unroll
                for (int j_21 = 0; j_21 < 16; j_21++) {
                    hs2_18[j_21] = _tmem_load_12[j_21] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, hs2_18);
                float _tmem_load_14[16];
                tmem_ld_x16(&_tmem_load_14[0], taddr + (unsigned int)h_row_base + 112);
                float _tmem_load_15[16];
                tmem_ld_x16(&_tmem_load_15[0], taddr + 256 + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_22 = 0; j_22 < 16; j_22++) {
                    _tmem_load_14[j_22] = _tmem_load_14[j_22] + _tmem_load_15[j_22];
                }
                unsigned int _tmem_load_14_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 0], _tmem_load_14[_lp*2+1 + 0]));
                    _tmem_load_14_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_19[16];
                r1_19[0] = _tmem_load_14[0] - __uint_as_float(_tmem_load_14_bf16[0] << 16);
                r1_19[1] = _tmem_load_14[1] - __uint_as_float(_tmem_load_14_bf16[0] & mask_hi);
                r1_19[2] = _tmem_load_14[2] - __uint_as_float(_tmem_load_14_bf16[1] << 16);
                r1_19[3] = _tmem_load_14[3] - __uint_as_float(_tmem_load_14_bf16[1] & mask_hi);
                r1_19[4] = _tmem_load_14[4] - __uint_as_float(_tmem_load_14_bf16[2] << 16);
                r1_19[5] = _tmem_load_14[5] - __uint_as_float(_tmem_load_14_bf16[2] & mask_hi);
                r1_19[6] = _tmem_load_14[6] - __uint_as_float(_tmem_load_14_bf16[3] << 16);
                r1_19[7] = _tmem_load_14[7] - __uint_as_float(_tmem_load_14_bf16[3] & mask_hi);
                r1_19[8] = _tmem_load_14[8] - __uint_as_float(_tmem_load_14_bf16[4] << 16);
                r1_19[9] = _tmem_load_14[9] - __uint_as_float(_tmem_load_14_bf16[4] & mask_hi);
                r1_19[10] = _tmem_load_14[10] - __uint_as_float(_tmem_load_14_bf16[5] << 16);
                r1_19[11] = _tmem_load_14[11] - __uint_as_float(_tmem_load_14_bf16[5] & mask_hi);
                r1_19[12] = _tmem_load_14[12] - __uint_as_float(_tmem_load_14_bf16[6] << 16);
                r1_19[13] = _tmem_load_14[13] - __uint_as_float(_tmem_load_14_bf16[6] & mask_hi);
                r1_19[14] = _tmem_load_14[14] - __uint_as_float(_tmem_load_14_bf16[7] << 16);
                r1_19[15] = _tmem_load_14[15] - __uint_as_float(_tmem_load_14_bf16[7] & mask_hi);
                unsigned int r1_19_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_19[_lp*2 + 0], r1_19[_lp*2+1 + 0]));
                    r1_19_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_20[16];
                r2_20[0] = r1_19[0] - __uint_as_float(r1_19_bf16[0] << 16);
                r2_20[1] = r1_19[1] - __uint_as_float(r1_19_bf16[0] & mask_hi);
                r2_20[2] = r1_19[2] - __uint_as_float(r1_19_bf16[1] << 16);
                r2_20[3] = r1_19[3] - __uint_as_float(r1_19_bf16[1] & mask_hi);
                r2_20[4] = r1_19[4] - __uint_as_float(r1_19_bf16[2] << 16);
                r2_20[5] = r1_19[5] - __uint_as_float(r1_19_bf16[2] & mask_hi);
                r2_20[6] = r1_19[6] - __uint_as_float(r1_19_bf16[3] << 16);
                r2_20[7] = r1_19[7] - __uint_as_float(r1_19_bf16[3] & mask_hi);
                r2_20[8] = r1_19[8] - __uint_as_float(r1_19_bf16[4] << 16);
                r2_20[9] = r1_19[9] - __uint_as_float(r1_19_bf16[4] & mask_hi);
                r2_20[10] = r1_19[10] - __uint_as_float(r1_19_bf16[5] << 16);
                r2_20[11] = r1_19[11] - __uint_as_float(r1_19_bf16[5] & mask_hi);
                r2_20[12] = r1_19[12] - __uint_as_float(r1_19_bf16[6] << 16);
                r2_20[13] = r1_19[13] - __uint_as_float(r1_19_bf16[6] & mask_hi);
                r2_20[14] = r1_19[14] - __uint_as_float(r1_19_bf16[7] << 16);
                r2_20[15] = r1_19[15] - __uint_as_float(r1_19_bf16[7] & mask_hi);
                unsigned int r2_20_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_20[_lp*2 + 0], r2_20[_lp*2+1 + 0]));
                    r2_20_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 96 ^ (16384 + k_row * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_14_bf16[0]), "r"(_tmem_load_14_bf16[1]), "r"(_tmem_load_14_bf16[2]), "r"(_tmem_load_14_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs0_addr + (unsigned int)(16384 + k_row * 128 + 112 ^ (16384 + k_row * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_14_bf16[4]), "r"(_tmem_load_14_bf16[5]), "r"(_tmem_load_14_bf16[6]), "r"(_tmem_load_14_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 96 ^ (16384 + k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r1_19_bf16[0]), "r"(r1_19_bf16[1]), "r"(r1_19_bf16[2]), "r"(r1_19_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs1_addr + (unsigned int)(16384 + k_row * 128 + 112 ^ (16384 + k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r1_19_bf16[4]), "r"(r1_19_bf16[5]), "r"(r1_19_bf16[6]), "r"(r1_19_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 96 ^ (16384 + k_row * 128 + 96 >> 7 & 7) << 4))), "r"(r2_20_bf16[0]), "r"(r2_20_bf16[1]), "r"(r2_20_bf16[2]), "r"(r2_20_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs2_addr + (unsigned int)(16384 + k_row * 128 + 112 ^ (16384 + k_row * 128 + 112 >> 7 & 7) << 4))), "r"(r2_20_bf16[4]), "r"(r2_20_bf16[5]), "r"(r2_20_bf16[6]), "r"(r2_20_bf16[7]) : "memory");
                float hs2_21[16];
                #pragma unroll
                for (int j_23 = 0; j_23 < 16; j_23++) {
                    hs2_21[j_23] = _tmem_load_14[j_23] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, hs2_21);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long u_base = ((long long)trow * (long long)num_heads + (long long)head) * 128 + (long long)v0;
                float _tmem_load_16[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col = lane_half * 16;
                float uv[16];
                uv[0] = 0.0f;
                uv[1] = 0.0f;
                uv[2] = 0.0f;
                uv[3] = 0.0f;
                uv[4] = 0.0f;
                uv[5] = 0.0f;
                uv[6] = 0.0f;
                uv[7] = 0.0f;
                uv[8] = 0.0f;
                uv[9] = 0.0f;
                uv[10] = 0.0f;
                uv[11] = 0.0f;
                uv[12] = 0.0f;
                uv[13] = 0.0f;
                uv[14] = 0.0f;
                uv[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(u + u_base + (long long)col);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&uv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv[16];
                #pragma unroll
                for (int j_24 = 0; j_24 < 16; j_24++) {
                    bvv[j_24] = uv[j_24] - _tmem_load_16[j_24];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_22[16];
                r1_22[0] = bvv[0] - __uint_as_float(bvv_bf16[0] << 16);
                r1_22[1] = bvv[1] - __uint_as_float(bvv_bf16[0] & mask_hi);
                r1_22[2] = bvv[2] - __uint_as_float(bvv_bf16[1] << 16);
                r1_22[3] = bvv[3] - __uint_as_float(bvv_bf16[1] & mask_hi);
                r1_22[4] = bvv[4] - __uint_as_float(bvv_bf16[2] << 16);
                r1_22[5] = bvv[5] - __uint_as_float(bvv_bf16[2] & mask_hi);
                r1_22[6] = bvv[6] - __uint_as_float(bvv_bf16[3] << 16);
                r1_22[7] = bvv[7] - __uint_as_float(bvv_bf16[3] & mask_hi);
                r1_22[8] = bvv[8] - __uint_as_float(bvv_bf16[4] << 16);
                r1_22[9] = bvv[9] - __uint_as_float(bvv_bf16[4] & mask_hi);
                r1_22[10] = bvv[10] - __uint_as_float(bvv_bf16[5] << 16);
                r1_22[11] = bvv[11] - __uint_as_float(bvv_bf16[5] & mask_hi);
                r1_22[12] = bvv[12] - __uint_as_float(bvv_bf16[6] << 16);
                r1_22[13] = bvv[13] - __uint_as_float(bvv_bf16[6] & mask_hi);
                r1_22[14] = bvv[14] - __uint_as_float(bvv_bf16[7] << 16);
                r1_22[15] = bvv[15] - __uint_as_float(bvv_bf16[7] & mask_hi);
                unsigned int r1_22_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_22[_lp*2 + 0], r1_22[_lp*2+1 + 0]));
                    r1_22_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_23[16];
                r2_23[0] = r1_22[0] - __uint_as_float(r1_22_bf16[0] << 16);
                r2_23[1] = r1_22[1] - __uint_as_float(r1_22_bf16[0] & mask_hi);
                r2_23[2] = r1_22[2] - __uint_as_float(r1_22_bf16[1] << 16);
                r2_23[3] = r1_22[3] - __uint_as_float(r1_22_bf16[1] & mask_hi);
                r2_23[4] = r1_22[4] - __uint_as_float(r1_22_bf16[2] << 16);
                r2_23[5] = r1_22[5] - __uint_as_float(r1_22_bf16[2] & mask_hi);
                r2_23[6] = r1_22[6] - __uint_as_float(r1_22_bf16[3] << 16);
                r2_23[7] = r1_22[7] - __uint_as_float(r1_22_bf16[3] & mask_hi);
                r2_23[8] = r1_22[8] - __uint_as_float(r1_22_bf16[4] << 16);
                r2_23[9] = r1_22[9] - __uint_as_float(r1_22_bf16[4] & mask_hi);
                r2_23[10] = r1_22[10] - __uint_as_float(r1_22_bf16[5] << 16);
                r2_23[11] = r1_22[11] - __uint_as_float(r1_22_bf16[5] & mask_hi);
                r2_23[12] = r1_22[12] - __uint_as_float(r1_22_bf16[6] << 16);
                r2_23[13] = r1_22[13] - __uint_as_float(r1_22_bf16[6] & mask_hi);
                r2_23[14] = r1_22[14] - __uint_as_float(r1_22_bf16[7] << 16);
                r2_23[15] = r1_22[15] - __uint_as_float(r1_22_bf16[7] & mask_hi);
                unsigned int r2_23_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_23[_lp*2 + 0], r2_23[_lp*2+1 + 0]));
                    r2_23_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col / 64 * 8192 + t_row * 128 + col % 64 * 2 ^ (col / 64 * 8192 + t_row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[0]), "r"(bvv_bf16[1]), "r"(bvv_bf16[2]), "r"(bvv_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 ^ ((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[4]), "r"(bvv_bf16[5]), "r"(bvv_bf16[6]), "r"(bvv_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col / 64 * 8192 + t_row * 128 + col % 64 * 2 ^ (col / 64 * 8192 + t_row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(r1_22_bf16[0]), "r"(r1_22_bf16[1]), "r"(r1_22_bf16[2]), "r"(r1_22_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 ^ ((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_22_bf16[4]), "r"(r1_22_bf16[5]), "r"(r1_22_bf16[6]), "r"(r1_22_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col / 64 * 8192 + t_row * 128 + col % 64 * 2 ^ (col / 64 * 8192 + t_row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(r2_23_bf16[0]), "r"(r2_23_bf16[1]), "r"(r2_23_bf16[2]), "r"(r2_23_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 ^ ((col + 8) / 64 * 8192 + t_row * 128 + (col + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_23_bf16[4]), "r"(r2_23_bf16[5]), "r"(r2_23_bf16[6]), "r"(r2_23_bf16[7]) : "memory");
                float _tmem_load_17[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_24 = 32 + lane_half * 16;
                float uv_25[16];
                uv_25[0] = 0.0f;
                uv_25[1] = 0.0f;
                uv_25[2] = 0.0f;
                uv_25[3] = 0.0f;
                uv_25[4] = 0.0f;
                uv_25[5] = 0.0f;
                uv_25[6] = 0.0f;
                uv_25[7] = 0.0f;
                uv_25[8] = 0.0f;
                uv_25[9] = 0.0f;
                uv_25[10] = 0.0f;
                uv_25[11] = 0.0f;
                uv_25[12] = 0.0f;
                uv_25[13] = 0.0f;
                uv_25[14] = 0.0f;
                uv_25[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_24);
                        uint4 _vld_1[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_25[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                                (&uv_25[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_26[16];
                #pragma unroll
                for (int j_25 = 0; j_25 < 16; j_25++) {
                    bvv_26[j_25] = uv_25[j_25] - _tmem_load_17[j_25];
                }
                unsigned int bvv_26_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_26[_lp*2 + 0], bvv_26[_lp*2+1 + 0]));
                    bvv_26_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_27[16];
                r1_27[0] = bvv_26[0] - __uint_as_float(bvv_26_bf16[0] << 16);
                r1_27[1] = bvv_26[1] - __uint_as_float(bvv_26_bf16[0] & mask_hi);
                r1_27[2] = bvv_26[2] - __uint_as_float(bvv_26_bf16[1] << 16);
                r1_27[3] = bvv_26[3] - __uint_as_float(bvv_26_bf16[1] & mask_hi);
                r1_27[4] = bvv_26[4] - __uint_as_float(bvv_26_bf16[2] << 16);
                r1_27[5] = bvv_26[5] - __uint_as_float(bvv_26_bf16[2] & mask_hi);
                r1_27[6] = bvv_26[6] - __uint_as_float(bvv_26_bf16[3] << 16);
                r1_27[7] = bvv_26[7] - __uint_as_float(bvv_26_bf16[3] & mask_hi);
                r1_27[8] = bvv_26[8] - __uint_as_float(bvv_26_bf16[4] << 16);
                r1_27[9] = bvv_26[9] - __uint_as_float(bvv_26_bf16[4] & mask_hi);
                r1_27[10] = bvv_26[10] - __uint_as_float(bvv_26_bf16[5] << 16);
                r1_27[11] = bvv_26[11] - __uint_as_float(bvv_26_bf16[5] & mask_hi);
                r1_27[12] = bvv_26[12] - __uint_as_float(bvv_26_bf16[6] << 16);
                r1_27[13] = bvv_26[13] - __uint_as_float(bvv_26_bf16[6] & mask_hi);
                r1_27[14] = bvv_26[14] - __uint_as_float(bvv_26_bf16[7] << 16);
                r1_27[15] = bvv_26[15] - __uint_as_float(bvv_26_bf16[7] & mask_hi);
                unsigned int r1_27_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_27[_lp*2 + 0], r1_27[_lp*2+1 + 0]));
                    r1_27_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_28[16];
                r2_28[0] = r1_27[0] - __uint_as_float(r1_27_bf16[0] << 16);
                r2_28[1] = r1_27[1] - __uint_as_float(r1_27_bf16[0] & mask_hi);
                r2_28[2] = r1_27[2] - __uint_as_float(r1_27_bf16[1] << 16);
                r2_28[3] = r1_27[3] - __uint_as_float(r1_27_bf16[1] & mask_hi);
                r2_28[4] = r1_27[4] - __uint_as_float(r1_27_bf16[2] << 16);
                r2_28[5] = r1_27[5] - __uint_as_float(r1_27_bf16[2] & mask_hi);
                r2_28[6] = r1_27[6] - __uint_as_float(r1_27_bf16[3] << 16);
                r2_28[7] = r1_27[7] - __uint_as_float(r1_27_bf16[3] & mask_hi);
                r2_28[8] = r1_27[8] - __uint_as_float(r1_27_bf16[4] << 16);
                r2_28[9] = r1_27[9] - __uint_as_float(r1_27_bf16[4] & mask_hi);
                r2_28[10] = r1_27[10] - __uint_as_float(r1_27_bf16[5] << 16);
                r2_28[11] = r1_27[11] - __uint_as_float(r1_27_bf16[5] & mask_hi);
                r2_28[12] = r1_27[12] - __uint_as_float(r1_27_bf16[6] << 16);
                r2_28[13] = r1_27[13] - __uint_as_float(r1_27_bf16[6] & mask_hi);
                r2_28[14] = r1_27[14] - __uint_as_float(r1_27_bf16[7] << 16);
                r2_28[15] = r1_27[15] - __uint_as_float(r1_27_bf16[7] & mask_hi);
                unsigned int r2_28_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_28[_lp*2 + 0], r2_28[_lp*2+1 + 0]));
                    r2_28_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_24 / 64 * 8192 + t_row * 128 + col_24 % 64 * 2 ^ (col_24 / 64 * 8192 + t_row * 128 + col_24 % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_26_bf16[0]), "r"(bvv_26_bf16[1]), "r"(bvv_26_bf16[2]), "r"(bvv_26_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_24 + 8) / 64 * 8192 + t_row * 128 + (col_24 + 8) % 64 * 2 ^ ((col_24 + 8) / 64 * 8192 + t_row * 128 + (col_24 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_26_bf16[4]), "r"(bvv_26_bf16[5]), "r"(bvv_26_bf16[6]), "r"(bvv_26_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_24 / 64 * 8192 + t_row * 128 + col_24 % 64 * 2 ^ (col_24 / 64 * 8192 + t_row * 128 + col_24 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_27_bf16[0]), "r"(r1_27_bf16[1]), "r"(r1_27_bf16[2]), "r"(r1_27_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_24 + 8) / 64 * 8192 + t_row * 128 + (col_24 + 8) % 64 * 2 ^ ((col_24 + 8) / 64 * 8192 + t_row * 128 + (col_24 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_27_bf16[4]), "r"(r1_27_bf16[5]), "r"(r1_27_bf16[6]), "r"(r1_27_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_24 / 64 * 8192 + t_row * 128 + col_24 % 64 * 2 ^ (col_24 / 64 * 8192 + t_row * 128 + col_24 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_28_bf16[0]), "r"(r2_28_bf16[1]), "r"(r2_28_bf16[2]), "r"(r2_28_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_24 + 8) / 64 * 8192 + t_row * 128 + (col_24 + 8) % 64 * 2 ^ ((col_24 + 8) / 64 * 8192 + t_row * 128 + (col_24 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_28_bf16[4]), "r"(r2_28_bf16[5]), "r"(r2_28_bf16[6]), "r"(r2_28_bf16[7]) : "memory");
                float _tmem_load_18[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_29 = 64 + lane_half * 16;
                float uv_30[16];
                uv_30[0] = 0.0f;
                uv_30[1] = 0.0f;
                uv_30[2] = 0.0f;
                uv_30[3] = 0.0f;
                uv_30[4] = 0.0f;
                uv_30[5] = 0.0f;
                uv_30[6] = 0.0f;
                uv_30[7] = 0.0f;
                uv_30[8] = 0.0f;
                uv_30[9] = 0.0f;
                uv_30[10] = 0.0f;
                uv_30[11] = 0.0f;
                uv_30[12] = 0.0f;
                uv_30[13] = 0.0f;
                uv_30[14] = 0.0f;
                uv_30[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_29);
                        uint4 _vld_2[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_30[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                (&uv_30[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_31[16];
                #pragma unroll
                for (int j_26 = 0; j_26 < 16; j_26++) {
                    bvv_31[j_26] = uv_30[j_26] - _tmem_load_18[j_26];
                }
                unsigned int bvv_31_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_31[_lp*2 + 0], bvv_31[_lp*2+1 + 0]));
                    bvv_31_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_32[16];
                r1_32[0] = bvv_31[0] - __uint_as_float(bvv_31_bf16[0] << 16);
                r1_32[1] = bvv_31[1] - __uint_as_float(bvv_31_bf16[0] & mask_hi);
                r1_32[2] = bvv_31[2] - __uint_as_float(bvv_31_bf16[1] << 16);
                r1_32[3] = bvv_31[3] - __uint_as_float(bvv_31_bf16[1] & mask_hi);
                r1_32[4] = bvv_31[4] - __uint_as_float(bvv_31_bf16[2] << 16);
                r1_32[5] = bvv_31[5] - __uint_as_float(bvv_31_bf16[2] & mask_hi);
                r1_32[6] = bvv_31[6] - __uint_as_float(bvv_31_bf16[3] << 16);
                r1_32[7] = bvv_31[7] - __uint_as_float(bvv_31_bf16[3] & mask_hi);
                r1_32[8] = bvv_31[8] - __uint_as_float(bvv_31_bf16[4] << 16);
                r1_32[9] = bvv_31[9] - __uint_as_float(bvv_31_bf16[4] & mask_hi);
                r1_32[10] = bvv_31[10] - __uint_as_float(bvv_31_bf16[5] << 16);
                r1_32[11] = bvv_31[11] - __uint_as_float(bvv_31_bf16[5] & mask_hi);
                r1_32[12] = bvv_31[12] - __uint_as_float(bvv_31_bf16[6] << 16);
                r1_32[13] = bvv_31[13] - __uint_as_float(bvv_31_bf16[6] & mask_hi);
                r1_32[14] = bvv_31[14] - __uint_as_float(bvv_31_bf16[7] << 16);
                r1_32[15] = bvv_31[15] - __uint_as_float(bvv_31_bf16[7] & mask_hi);
                unsigned int r1_32_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_32[_lp*2 + 0], r1_32[_lp*2+1 + 0]));
                    r1_32_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_33[16];
                r2_33[0] = r1_32[0] - __uint_as_float(r1_32_bf16[0] << 16);
                r2_33[1] = r1_32[1] - __uint_as_float(r1_32_bf16[0] & mask_hi);
                r2_33[2] = r1_32[2] - __uint_as_float(r1_32_bf16[1] << 16);
                r2_33[3] = r1_32[3] - __uint_as_float(r1_32_bf16[1] & mask_hi);
                r2_33[4] = r1_32[4] - __uint_as_float(r1_32_bf16[2] << 16);
                r2_33[5] = r1_32[5] - __uint_as_float(r1_32_bf16[2] & mask_hi);
                r2_33[6] = r1_32[6] - __uint_as_float(r1_32_bf16[3] << 16);
                r2_33[7] = r1_32[7] - __uint_as_float(r1_32_bf16[3] & mask_hi);
                r2_33[8] = r1_32[8] - __uint_as_float(r1_32_bf16[4] << 16);
                r2_33[9] = r1_32[9] - __uint_as_float(r1_32_bf16[4] & mask_hi);
                r2_33[10] = r1_32[10] - __uint_as_float(r1_32_bf16[5] << 16);
                r2_33[11] = r1_32[11] - __uint_as_float(r1_32_bf16[5] & mask_hi);
                r2_33[12] = r1_32[12] - __uint_as_float(r1_32_bf16[6] << 16);
                r2_33[13] = r1_32[13] - __uint_as_float(r1_32_bf16[6] & mask_hi);
                r2_33[14] = r1_32[14] - __uint_as_float(r1_32_bf16[7] << 16);
                r2_33[15] = r1_32[15] - __uint_as_float(r1_32_bf16[7] & mask_hi);
                unsigned int r2_33_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_33[_lp*2 + 0], r2_33[_lp*2+1 + 0]));
                    r2_33_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_29 / 64 * 8192 + t_row * 128 + col_29 % 64 * 2 ^ (col_29 / 64 * 8192 + t_row * 128 + col_29 % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_31_bf16[0]), "r"(bvv_31_bf16[1]), "r"(bvv_31_bf16[2]), "r"(bvv_31_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_29 + 8) / 64 * 8192 + t_row * 128 + (col_29 + 8) % 64 * 2 ^ ((col_29 + 8) / 64 * 8192 + t_row * 128 + (col_29 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_31_bf16[4]), "r"(bvv_31_bf16[5]), "r"(bvv_31_bf16[6]), "r"(bvv_31_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_29 / 64 * 8192 + t_row * 128 + col_29 % 64 * 2 ^ (col_29 / 64 * 8192 + t_row * 128 + col_29 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_32_bf16[0]), "r"(r1_32_bf16[1]), "r"(r1_32_bf16[2]), "r"(r1_32_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_29 + 8) / 64 * 8192 + t_row * 128 + (col_29 + 8) % 64 * 2 ^ ((col_29 + 8) / 64 * 8192 + t_row * 128 + (col_29 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_32_bf16[4]), "r"(r1_32_bf16[5]), "r"(r1_32_bf16[6]), "r"(r1_32_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_29 / 64 * 8192 + t_row * 128 + col_29 % 64 * 2 ^ (col_29 / 64 * 8192 + t_row * 128 + col_29 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_33_bf16[0]), "r"(r2_33_bf16[1]), "r"(r2_33_bf16[2]), "r"(r2_33_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_29 + 8) / 64 * 8192 + t_row * 128 + (col_29 + 8) % 64 * 2 ^ ((col_29 + 8) / 64 * 8192 + t_row * 128 + (col_29 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_33_bf16[4]), "r"(r2_33_bf16[5]), "r"(r2_33_bf16[6]), "r"(r2_33_bf16[7]) : "memory");
                float _tmem_load_19[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_34 = 96 + lane_half * 16;
                float uv_35[16];
                uv_35[0] = 0.0f;
                uv_35[1] = 0.0f;
                uv_35[2] = 0.0f;
                uv_35[3] = 0.0f;
                uv_35[4] = 0.0f;
                uv_35[5] = 0.0f;
                uv_35[6] = 0.0f;
                uv_35[7] = 0.0f;
                uv_35[8] = 0.0f;
                uv_35[9] = 0.0f;
                uv_35[10] = 0.0f;
                uv_35[11] = 0.0f;
                uv_35[12] = 0.0f;
                uv_35[13] = 0.0f;
                uv_35[14] = 0.0f;
                uv_35[15] = 0.0f;
                if (use_u == 1) {
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_34);
                        uint4 _vld_3[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_3[_blk] = _vptr_3[_blk];
                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&uv_35[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                (&uv_35[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
                float bvv_36[16];
                #pragma unroll
                for (int j_27 = 0; j_27 < 16; j_27++) {
                    bvv_36[j_27] = uv_35[j_27] - _tmem_load_19[j_27];
                }
                unsigned int bvv_36_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_36[_lp*2 + 0], bvv_36[_lp*2+1 + 0]));
                    bvv_36_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r1_37[16];
                r1_37[0] = bvv_36[0] - __uint_as_float(bvv_36_bf16[0] << 16);
                r1_37[1] = bvv_36[1] - __uint_as_float(bvv_36_bf16[0] & mask_hi);
                r1_37[2] = bvv_36[2] - __uint_as_float(bvv_36_bf16[1] << 16);
                r1_37[3] = bvv_36[3] - __uint_as_float(bvv_36_bf16[1] & mask_hi);
                r1_37[4] = bvv_36[4] - __uint_as_float(bvv_36_bf16[2] << 16);
                r1_37[5] = bvv_36[5] - __uint_as_float(bvv_36_bf16[2] & mask_hi);
                r1_37[6] = bvv_36[6] - __uint_as_float(bvv_36_bf16[3] << 16);
                r1_37[7] = bvv_36[7] - __uint_as_float(bvv_36_bf16[3] & mask_hi);
                r1_37[8] = bvv_36[8] - __uint_as_float(bvv_36_bf16[4] << 16);
                r1_37[9] = bvv_36[9] - __uint_as_float(bvv_36_bf16[4] & mask_hi);
                r1_37[10] = bvv_36[10] - __uint_as_float(bvv_36_bf16[5] << 16);
                r1_37[11] = bvv_36[11] - __uint_as_float(bvv_36_bf16[5] & mask_hi);
                r1_37[12] = bvv_36[12] - __uint_as_float(bvv_36_bf16[6] << 16);
                r1_37[13] = bvv_36[13] - __uint_as_float(bvv_36_bf16[6] & mask_hi);
                r1_37[14] = bvv_36[14] - __uint_as_float(bvv_36_bf16[7] << 16);
                r1_37[15] = bvv_36[15] - __uint_as_float(bvv_36_bf16[7] & mask_hi);
                unsigned int r1_37_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r1_37[_lp*2 + 0], r1_37[_lp*2+1 + 0]));
                    r1_37_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float r2_38[16];
                r2_38[0] = r1_37[0] - __uint_as_float(r1_37_bf16[0] << 16);
                r2_38[1] = r1_37[1] - __uint_as_float(r1_37_bf16[0] & mask_hi);
                r2_38[2] = r1_37[2] - __uint_as_float(r1_37_bf16[1] << 16);
                r2_38[3] = r1_37[3] - __uint_as_float(r1_37_bf16[1] & mask_hi);
                r2_38[4] = r1_37[4] - __uint_as_float(r1_37_bf16[2] << 16);
                r2_38[5] = r1_37[5] - __uint_as_float(r1_37_bf16[2] & mask_hi);
                r2_38[6] = r1_37[6] - __uint_as_float(r1_37_bf16[3] << 16);
                r2_38[7] = r1_37[7] - __uint_as_float(r1_37_bf16[3] & mask_hi);
                r2_38[8] = r1_37[8] - __uint_as_float(r1_37_bf16[4] << 16);
                r2_38[9] = r1_37[9] - __uint_as_float(r1_37_bf16[4] & mask_hi);
                r2_38[10] = r1_37[10] - __uint_as_float(r1_37_bf16[5] << 16);
                r2_38[11] = r1_37[11] - __uint_as_float(r1_37_bf16[5] & mask_hi);
                r2_38[12] = r1_37[12] - __uint_as_float(r1_37_bf16[6] << 16);
                r2_38[13] = r1_37[13] - __uint_as_float(r1_37_bf16[6] & mask_hi);
                r2_38[14] = r1_37[14] - __uint_as_float(r1_37_bf16[7] << 16);
                r2_38[15] = r1_37[15] - __uint_as_float(r1_37_bf16[7] & mask_hi);
                unsigned int r2_38_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r2_38[_lp*2 + 0], r2_38[_lp*2+1 + 0]));
                    r2_38_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)(col_34 / 64 * 8192 + t_row * 128 + col_34 % 64 * 2 ^ (col_34 / 64 * 8192 + t_row * 128 + col_34 % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_36_bf16[0]), "r"(bvv_36_bf16[1]), "r"(bvv_36_bf16[2]), "r"(bvv_36_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv0_addr + (unsigned int)((col_34 + 8) / 64 * 8192 + t_row * 128 + (col_34 + 8) % 64 * 2 ^ ((col_34 + 8) / 64 * 8192 + t_row * 128 + (col_34 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_36_bf16[4]), "r"(bvv_36_bf16[5]), "r"(bvv_36_bf16[6]), "r"(bvv_36_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)(col_34 / 64 * 8192 + t_row * 128 + col_34 % 64 * 2 ^ (col_34 / 64 * 8192 + t_row * 128 + col_34 % 64 * 2 >> 7 & 7) << 4))), "r"(r1_37_bf16[0]), "r"(r1_37_bf16[1]), "r"(r1_37_bf16[2]), "r"(r1_37_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv1_addr + (unsigned int)((col_34 + 8) / 64 * 8192 + t_row * 128 + (col_34 + 8) % 64 * 2 ^ ((col_34 + 8) / 64 * 8192 + t_row * 128 + (col_34 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r1_37_bf16[4]), "r"(r1_37_bf16[5]), "r"(r1_37_bf16[6]), "r"(r1_37_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)(col_34 / 64 * 8192 + t_row * 128 + col_34 % 64 * 2 ^ (col_34 / 64 * 8192 + t_row * 128 + col_34 % 64 * 2 >> 7 & 7) << 4))), "r"(r2_38_bf16[0]), "r"(r2_38_bf16[1]), "r"(r2_38_bf16[2]), "r"(r2_38_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv2_addr + (unsigned int)((col_34 + 8) / 64 * 8192 + t_row * 128 + (col_34 + 8) % 64 * 2 ^ ((col_34 + 8) / 64 * 8192 + t_row * 128 + (col_34 + 8) % 64 * 2 >> 7 & 7) << 4))), "r"(r2_38_bf16[4]), "r"(r2_38_bf16[5]), "r"(r2_38_bf16[6]), "r"(r2_38_bf16[7]) : "memory");
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
                float _tmem_load_20[16];
                tmem_ld_x16(&_tmem_load_20[0], taddr + (unsigned int)h_row_base);
                float _tmem_load_21[16];
                tmem_ld_x16(&_tmem_load_21[0], taddr + 256 + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_28 = 0; j_28 < 16; j_28++) {
                    _tmem_load_20[j_28] = _tmem_load_20[j_28] + _tmem_load_21[j_28];
                }
                if (write_transposed == 1) {
                    long long row_at = tile + (long long)(k_row * 128 + v0);
                    {
                        float4 _v4 = make_float4(_tmem_load_20[0 + 0], _tmem_load_20[0 + 1], _tmem_load_20[0 + 2], _tmem_load_20[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_20[4 + 0], _tmem_load_20[4 + 1], _tmem_load_20[4 + 2], _tmem_load_20[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_20[8 + 0], _tmem_load_20[8 + 1], _tmem_load_20[8 + 2], _tmem_load_20[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_20[12 + 0], _tmem_load_20[12 + 1], _tmem_load_20[12 + 2], _tmem_load_20[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_29 = 0; j_29 < 16; j_29++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + j_29) * 128 + k_row))) = _tmem_load_20[j_29];
                    }
                }
                float _tmem_load_22[16];
                tmem_ld_x16(&_tmem_load_22[0], taddr + (unsigned int)h_row_base + 16);
                float _tmem_load_23[16];
                tmem_ld_x16(&_tmem_load_23[0], taddr + 256 + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_30 = 0; j_30 < 16; j_30++) {
                    _tmem_load_22[j_30] = _tmem_load_22[j_30] + _tmem_load_23[j_30];
                }
                if (write_transposed == 1) {
                    long long row_at_1 = tile + (long long)(k_row * 128 + v0 + 16);
                    {
                        float4 _v4 = make_float4(_tmem_load_22[0 + 0], _tmem_load_22[0 + 1], _tmem_load_22[0 + 2], _tmem_load_22[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_22[4 + 0], _tmem_load_22[4 + 1], _tmem_load_22[4 + 2], _tmem_load_22[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_22[8 + 0], _tmem_load_22[8 + 1], _tmem_load_22[8 + 2], _tmem_load_22[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_22[12 + 0], _tmem_load_22[12 + 1], _tmem_load_22[12 + 2], _tmem_load_22[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_1 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_31 = 0; j_31 < 16; j_31++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 16 + j_31) * 128 + k_row))) = _tmem_load_22[j_31];
                    }
                }
                float _tmem_load_24[16];
                tmem_ld_x16(&_tmem_load_24[0], taddr + (unsigned int)h_row_base + 32);
                float _tmem_load_25[16];
                tmem_ld_x16(&_tmem_load_25[0], taddr + 256 + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_32 = 0; j_32 < 16; j_32++) {
                    _tmem_load_24[j_32] = _tmem_load_24[j_32] + _tmem_load_25[j_32];
                }
                if (write_transposed == 1) {
                    long long row_at_2 = tile + (long long)(k_row * 128 + v0 + 32);
                    {
                        float4 _v4 = make_float4(_tmem_load_24[0 + 0], _tmem_load_24[0 + 1], _tmem_load_24[0 + 2], _tmem_load_24[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_24[4 + 0], _tmem_load_24[4 + 1], _tmem_load_24[4 + 2], _tmem_load_24[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_24[8 + 0], _tmem_load_24[8 + 1], _tmem_load_24[8 + 2], _tmem_load_24[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_24[12 + 0], _tmem_load_24[12 + 1], _tmem_load_24[12 + 2], _tmem_load_24[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_2 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_33 = 0; j_33 < 16; j_33++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 32 + j_33) * 128 + k_row))) = _tmem_load_24[j_33];
                    }
                }
                float _tmem_load_26[16];
                tmem_ld_x16(&_tmem_load_26[0], taddr + (unsigned int)h_row_base + 48);
                float _tmem_load_27[16];
                tmem_ld_x16(&_tmem_load_27[0], taddr + 256 + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_34 = 0; j_34 < 16; j_34++) {
                    _tmem_load_26[j_34] = _tmem_load_26[j_34] + _tmem_load_27[j_34];
                }
                if (write_transposed == 1) {
                    long long row_at_3 = tile + (long long)(k_row * 128 + v0 + 48);
                    {
                        float4 _v4 = make_float4(_tmem_load_26[0 + 0], _tmem_load_26[0 + 1], _tmem_load_26[0 + 2], _tmem_load_26[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_26[4 + 0], _tmem_load_26[4 + 1], _tmem_load_26[4 + 2], _tmem_load_26[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_26[8 + 0], _tmem_load_26[8 + 1], _tmem_load_26[8 + 2], _tmem_load_26[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_26[12 + 0], _tmem_load_26[12 + 1], _tmem_load_26[12 + 2], _tmem_load_26[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_3 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_35 = 0; j_35 < 16; j_35++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 48 + j_35) * 128 + k_row))) = _tmem_load_26[j_35];
                    }
                }
                float _tmem_load_28[16];
                tmem_ld_x16(&_tmem_load_28[0], taddr + (unsigned int)h_row_base + 64);
                float _tmem_load_29[16];
                tmem_ld_x16(&_tmem_load_29[0], taddr + 256 + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_36 = 0; j_36 < 16; j_36++) {
                    _tmem_load_28[j_36] = _tmem_load_28[j_36] + _tmem_load_29[j_36];
                }
                if (write_transposed == 1) {
                    long long row_at_4 = tile + (long long)(k_row * 128 + v0 + 64);
                    {
                        float4 _v4 = make_float4(_tmem_load_28[0 + 0], _tmem_load_28[0 + 1], _tmem_load_28[0 + 2], _tmem_load_28[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_28[4 + 0], _tmem_load_28[4 + 1], _tmem_load_28[4 + 2], _tmem_load_28[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_4 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_28[8 + 0], _tmem_load_28[8 + 1], _tmem_load_28[8 + 2], _tmem_load_28[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_4 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_28[12 + 0], _tmem_load_28[12 + 1], _tmem_load_28[12 + 2], _tmem_load_28[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_4 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_37 = 0; j_37 < 16; j_37++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 64 + j_37) * 128 + k_row))) = _tmem_load_28[j_37];
                    }
                }
                float _tmem_load_30[16];
                tmem_ld_x16(&_tmem_load_30[0], taddr + (unsigned int)h_row_base + 80);
                float _tmem_load_31[16];
                tmem_ld_x16(&_tmem_load_31[0], taddr + 256 + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_38 = 0; j_38 < 16; j_38++) {
                    _tmem_load_30[j_38] = _tmem_load_30[j_38] + _tmem_load_31[j_38];
                }
                if (write_transposed == 1) {
                    long long row_at_5 = tile + (long long)(k_row * 128 + v0 + 80);
                    {
                        float4 _v4 = make_float4(_tmem_load_30[0 + 0], _tmem_load_30[0 + 1], _tmem_load_30[0 + 2], _tmem_load_30[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_5) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_30[4 + 0], _tmem_load_30[4 + 1], _tmem_load_30[4 + 2], _tmem_load_30[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_5 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_30[8 + 0], _tmem_load_30[8 + 1], _tmem_load_30[8 + 2], _tmem_load_30[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_5 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_30[12 + 0], _tmem_load_30[12 + 1], _tmem_load_30[12 + 2], _tmem_load_30[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_5 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_39 = 0; j_39 < 16; j_39++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 80 + j_39) * 128 + k_row))) = _tmem_load_30[j_39];
                    }
                }
                float _tmem_load_32[16];
                tmem_ld_x16(&_tmem_load_32[0], taddr + (unsigned int)h_row_base + 96);
                float _tmem_load_33[16];
                tmem_ld_x16(&_tmem_load_33[0], taddr + 256 + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_40 = 0; j_40 < 16; j_40++) {
                    _tmem_load_32[j_40] = _tmem_load_32[j_40] + _tmem_load_33[j_40];
                }
                if (write_transposed == 1) {
                    long long row_at_6 = tile + (long long)(k_row * 128 + v0 + 96);
                    {
                        float4 _v4 = make_float4(_tmem_load_32[0 + 0], _tmem_load_32[0 + 1], _tmem_load_32[0 + 2], _tmem_load_32[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_6) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_32[4 + 0], _tmem_load_32[4 + 1], _tmem_load_32[4 + 2], _tmem_load_32[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_6 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_32[8 + 0], _tmem_load_32[8 + 1], _tmem_load_32[8 + 2], _tmem_load_32[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_6 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_32[12 + 0], _tmem_load_32[12 + 1], _tmem_load_32[12 + 2], _tmem_load_32[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_6 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_41 = 0; j_41 < 16; j_41++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 96 + j_41) * 128 + k_row))) = _tmem_load_32[j_41];
                    }
                }
                float _tmem_load_34[16];
                tmem_ld_x16(&_tmem_load_34[0], taddr + (unsigned int)h_row_base + 112);
                float _tmem_load_35[16];
                tmem_ld_x16(&_tmem_load_35[0], taddr + 256 + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_42 = 0; j_42 < 16; j_42++) {
                    _tmem_load_34[j_42] = _tmem_load_34[j_42] + _tmem_load_35[j_42];
                }
                if (write_transposed == 1) {
                    long long row_at_7 = tile + (long long)(k_row * 128 + v0 + 112);
                    {
                        float4 _v4 = make_float4(_tmem_load_34[0 + 0], _tmem_load_34[0 + 1], _tmem_load_34[0 + 2], _tmem_load_34[0 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_7) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_34[4 + 0], _tmem_load_34[4 + 1], _tmem_load_34[4 + 2], _tmem_load_34[4 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_7 + 4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_34[8 + 0], _tmem_load_34[8 + 1], _tmem_load_34[8 + 2], _tmem_load_34[8 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_7 + 8) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_34[12 + 0], _tmem_load_34[12 + 1], _tmem_load_34[12 + 2], _tmem_load_34[12 + 3]);
                        *reinterpret_cast<float4*>(out + row_at_7 + 12) = _v4;
                    }
                } else {
                    #pragma unroll
                    for (int j_43 = 0; j_43 < 16; j_43++) {
                        *(reinterpret_cast<float*>(out) + (tile + (long long)((v0 + 112 + j_43) * 128 + k_row))) = _tmem_load_34[j_43];
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
