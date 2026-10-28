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
#define TMEM_NCOLS 256
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 128
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_W_PANEL_OFF 1024
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_K_PANEL_OFF 33792
#define SMEM_K_PANEL_STAGE_BYTES 8192
#define SMEM_K_PANEL_STRIDE 8192
#define SMEM_W_OP_OFF 1024
#define SMEM_W_OP_STAGE_BYTES 16384
#define SMEM_W_OP_STRIDE 16384
#define SMEM_K_MN_OFF 33792
#define SMEM_K_MN_STAGE_BYTES 16384
#define SMEM_K_MN_STRIDE 16384
#define SMEM_HS_OFF 66560
#define SMEM_HS_STAGE_BYTES 32768
#define SMEM_HS_STRIDE 32768
#define SMEM_HS_MN_OFF 66560
#define SMEM_HS_MN_STAGE_BYTES 32768
#define SMEM_HS_MN_STRIDE 32768
#define SMEM_BV_OFF 99328
#define SMEM_BV_STAGE_BYTES 16384
#define SMEM_BV_STRIDE 16384
#define SMEM_BV_MN_OFF 99328
#define SMEM_BV_MN_STAGE_BYTES 16384
#define SMEM_BV_MN_STRIDE 16384
#define SMEM_HS_PANEL_OFF 66560
#define SMEM_HS_PANEL_STAGE_BYTES 16384
#define SMEM_HS_PANEL_STRIDE 16384
#define SMEM_VN_OFF 66560
#define SMEM_VN_STAGE_BYTES 16384
#define SMEM_VN_STRIDE 16384
#define SMEM_VN_PANEL_OFF 66560
#define SMEM_VN_PANEL_STAGE_BYTES 8192
#define SMEM_VN_PANEL_STRIDE 8192
#define SMEM_TOTAL 115712
#define THREADS 192
#define LAUNCH_MIN_BLOCKS 2

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

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_2a0326b75a49ed13cfde(const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap w_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ rowscale, float* __restrict__ gamma, int num_heads, int num_qk_heads, int* __restrict__ chunk_bos, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, float* __restrict__ h0, int* __restrict__ h0_index, int h0_hg, int h0_n, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, const __grid_constant__ CUtensorMap h_tape, const __grid_constant__ CUtensorMap v_new)
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
    #define hs_free_addr (mbar_base + 64)
    #define vn_free_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_panel_addr = smem + 1024;
    __nv_bfloat16* k_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_panel_addr = smem + 33792;
    __nv_bfloat16* w_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_op_addr = smem + 1024;
    __nv_bfloat16* k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_mn_addr = smem + 33792;
    __nv_bfloat16* hs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_addr = smem + 66560;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn_addr = smem + 66560;
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_addr = smem + 99328;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_mn_addr = smem + 99328;
    __nv_bfloat16* hs_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_panel_addr = smem + 66560;
    __nv_bfloat16* vn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int vn_addr = smem + 66560;
    __nv_bfloat16* vn_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int vn_panel_addr = smem + 66560;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

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
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // hs_free: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // vn_free: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int hb = blockIdx.x;
            int vbase = 0;
            int head = hb % num_heads;
            int walk = hb / num_heads;
            int seg = walk_seg[walk];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            float init_row[16];
            int h0_slot = h0_index[seg];
            if (h0_slot >= 0) {
                int h0_tile = (head / h0_hg * h0_n + h0_slot) * h0_hg + head % h0_hg;
                long long h0_base = (long long)h0_tile * (long long)tile_stride * 16384 + (long long)(k_row * 128 + vbase);
                #pragma unroll
                for (int blk = 0; blk < 8; blk++) {
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(blk * 16 + q * 4));
                            init_row[q * 4 + 0] = _v4.x;
                            init_row[q * 4 + 1] = _v4.y;
                            init_row[q * 4 + 2] = _v4.z;
                            init_row[q * 4 + 3] = _v4.w;
                        }
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)h_row_base + (unsigned int)(blk * 16), init_row);
                }
            } else {
                init_row[0] = 0.0f;
                init_row[1] = 0.0f;
                init_row[2] = 0.0f;
                init_row[3] = 0.0f;
                init_row[4] = 0.0f;
                init_row[5] = 0.0f;
                init_row[6] = 0.0f;
                init_row[7] = 0.0f;
                init_row[8] = 0.0f;
                init_row[9] = 0.0f;
                init_row[10] = 0.0f;
                init_row[11] = 0.0f;
                init_row[12] = 0.0f;
                init_row[13] = 0.0f;
                init_row[14] = 0.0f;
                init_row[15] = 0.0f;
                #pragma unroll
                for (int blk_1 = 0; blk_1 < 8; blk_1++) {
                    tmem_st_x16_f32(taddr + (unsigned int)h_row_base + (unsigned int)(blk_1 * 16), init_row);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float decay = 0.0f;
            float rs = 0.0f;
            long long g_next = (long long)c0 * (long long)num_heads + (long long)head;
            long long rs_next = (long long)(row0 + t_row) * (long long)num_heads + (long long)head;
            if (nc > 0) {
                decay = gamma[g_next];
                rs = rowscale[rs_next];
            }
            unsigned int _phase_vn_free_0 = 1;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_hs_free_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                mbarrier_wait(vn_free_addr, _phase_vn_free_0);
                _phase_vn_free_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int _tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q_1 * 8 / 64 * 16384 + k_row * 128 + q_1 * 8 % 64 * 2 ^ (q_1 * 8 / 64 * 16384 + k_row * 128 + q_1 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[q_1 * 4]), "r"(_tmem_load_0_bf16[q_1 * 4 + 1]), "r"(_tmem_load_0_bf16[q_1 * 4 + 2]), "r"(_tmem_load_0_bf16[q_1 * 4 + 3]) : "memory");
                }
                float hd[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    hd[j] = _tmem_load_0[j] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, hd);
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int _tmem_load_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_2 * 8) / 64 * 16384 + k_row * 128 + (16 + q_2 * 8) % 64 * 2 ^ ((16 + q_2 * 8) / 64 * 16384 + k_row * 128 + (16 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[q_2 * 4]), "r"(_tmem_load_1_bf16[q_2 * 4 + 1]), "r"(_tmem_load_1_bf16[q_2 * 4 + 2]), "r"(_tmem_load_1_bf16[q_2 * 4 + 3]) : "memory");
                }
                float hd_0[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    hd_0[j_1] = _tmem_load_1[j_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, hd_0);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_3 * 8) / 64 * 16384 + k_row * 128 + (32 + q_3 * 8) % 64 * 2 ^ ((32 + q_3 * 8) / 64 * 16384 + k_row * 128 + (32 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[q_3 * 4]), "r"(_tmem_load_2_bf16[q_3 * 4 + 1]), "r"(_tmem_load_2_bf16[q_3 * 4 + 2]), "r"(_tmem_load_2_bf16[q_3 * 4 + 3]) : "memory");
                }
                float hd_1[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    hd_1[j_2] = _tmem_load_2[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, hd_1);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_4 * 8) / 64 * 16384 + k_row * 128 + (48 + q_4 * 8) % 64 * 2 ^ ((48 + q_4 * 8) / 64 * 16384 + k_row * 128 + (48 + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[q_4 * 4]), "r"(_tmem_load_3_bf16[q_4 * 4 + 1]), "r"(_tmem_load_3_bf16[q_4 * 4 + 2]), "r"(_tmem_load_3_bf16[q_4 * 4 + 3]) : "memory");
                }
                float hd_2[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    hd_2[j_3] = _tmem_load_3[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, hd_2);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_5 = 0; q_5 < 2; q_5++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((64 + q_5 * 8) / 64 * 16384 + k_row * 128 + (64 + q_5 * 8) % 64 * 2 ^ ((64 + q_5 * 8) / 64 * 16384 + k_row * 128 + (64 + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[q_5 * 4]), "r"(_tmem_load_4_bf16[q_5 * 4 + 1]), "r"(_tmem_load_4_bf16[q_5 * 4 + 2]), "r"(_tmem_load_4_bf16[q_5 * 4 + 3]) : "memory");
                }
                float hd_3[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    hd_3[j_4] = _tmem_load_4[j_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, hd_3);
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int _tmem_load_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                    _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_6 = 0; q_6 < 2; q_6++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((80 + q_6 * 8) / 64 * 16384 + k_row * 128 + (80 + q_6 * 8) % 64 * 2 ^ ((80 + q_6 * 8) / 64 * 16384 + k_row * 128 + (80 + q_6 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[q_6 * 4]), "r"(_tmem_load_5_bf16[q_6 * 4 + 1]), "r"(_tmem_load_5_bf16[q_6 * 4 + 2]), "r"(_tmem_load_5_bf16[q_6 * 4 + 3]) : "memory");
                }
                float hd_4[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    hd_4[j_5] = _tmem_load_5[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, hd_4);
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int _tmem_load_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_7 = 0; q_7 < 2; q_7++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((96 + q_7 * 8) / 64 * 16384 + k_row * 128 + (96 + q_7 * 8) % 64 * 2 ^ ((96 + q_7 * 8) / 64 * 16384 + k_row * 128 + (96 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[q_7 * 4]), "r"(_tmem_load_6_bf16[q_7 * 4 + 1]), "r"(_tmem_load_6_bf16[q_7 * 4 + 2]), "r"(_tmem_load_6_bf16[q_7 * 4 + 3]) : "memory");
                }
                float hd_5[16];
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    hd_5[j_6] = _tmem_load_6[j_6] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, hd_5);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int _tmem_load_7_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                    _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_8 = 0; q_8 < 2; q_8++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((112 + q_8 * 8) / 64 * 16384 + k_row * 128 + (112 + q_8 * 8) % 64 * 2 ^ ((112 + q_8 * 8) / 64 * 16384 + k_row * 128 + (112 + q_8 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[q_8 * 4]), "r"(_tmem_load_7_bf16[q_8 * 4 + 1]), "r"(_tmem_load_7_bf16[q_8 * 4 + 2]), "r"(_tmem_load_7_bf16[q_8 * 4 + 3]) : "memory");
                }
                float hd_6[16];
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    hd_6[j_7] = _tmem_load_7[j_7] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, hd_6);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                long long u_base = ((long long)(row0 + c * 64 + t_row) * (long long)num_heads + (long long)head) * 128 + (long long)vbase;
                int col = lane_half * 16;
                float uv[16];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(u + u_base + (long long)col);
                    uint4 _vld_1[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&uv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&uv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                mbarrier_wait(hs_free_addr, _phase_hs_free_0);
                _phase_hs_free_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_8[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base));
                int col_0 = 32 + lane_half * 16;
                float uv_0[16];
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_0);
                    uint4 _vld_2[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&uv_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&uv_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float bvv[16];
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    bvv[j_8] = uv[j_8] - _tmem_load_8[j_8];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_9 = 0; q_9 < 2; q_9++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((vn_addr + (unsigned int)((col + q_9 * 8) / 64 * 8192 + t_row * 128 + (col + q_9 * 8) % 64 * 2 ^ ((col + q_9 * 8) / 64 * 8192 + t_row * 128 + (col + q_9 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[q_9 * 4]), "r"(bvv_bf16[q_9 * 4 + 1]), "r"(bvv_bf16[q_9 * 4 + 2]), "r"(bvv_bf16[q_9 * 4 + 3]) : "memory");
                }
                float bo[16];
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    bo[j_9] = bvv[j_9] * rs;
                }
                unsigned int bo_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bo[_lp*2 + 0], bo[_lp*2+1 + 0]));
                    bo_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_10 = 0; q_10 < 2; q_10++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col + q_10 * 8) / 64 * 8192 + t_row * 128 + (col + q_10 * 8) % 64 * 2 ^ ((col + q_10 * 8) / 64 * 8192 + t_row * 128 + (col + q_10 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bo_bf16[q_10 * 4]), "r"(bo_bf16[q_10 * 4 + 1]), "r"(bo_bf16[q_10 * 4 + 2]), "r"(bo_bf16[q_10 * 4 + 3]) : "memory");
                }
                float _tmem_load_9[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                int col_1 = 64 + lane_half * 16;
                float uv_1[16];
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_1);
                    uint4 _vld_3[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&uv_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&uv_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float bvv_11[16];
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    bvv_11[j_10] = uv_0[j_10] - _tmem_load_9[j_10];
                }
                unsigned int bvv_11_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_11[_lp*2 + 0], bvv_11[_lp*2+1 + 0]));
                    bvv_11_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_11 = 0; q_11 < 2; q_11++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((vn_addr + (unsigned int)((col_0 + q_11 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_11 * 8) % 64 * 2 ^ ((col_0 + q_11 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_11_bf16[q_11 * 4]), "r"(bvv_11_bf16[q_11 * 4 + 1]), "r"(bvv_11_bf16[q_11 * 4 + 2]), "r"(bvv_11_bf16[q_11 * 4 + 3]) : "memory");
                }
                float bo_12[16];
                #pragma unroll
                for (int j_11 = 0; j_11 < 16; j_11++) {
                    bo_12[j_11] = bvv_11[j_11] * rs;
                }
                unsigned int bo_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bo_12[_lp*2 + 0], bo_12[_lp*2+1 + 0]));
                    bo_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_12 = 0; q_12 < 2; q_12++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_0 + q_12 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_12 * 8) % 64 * 2 ^ ((col_0 + q_12 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_12 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bo_12_bf16[q_12 * 4]), "r"(bo_12_bf16[q_12 * 4 + 1]), "r"(bo_12_bf16[q_12 * 4 + 2]), "r"(bo_12_bf16[q_12 * 4 + 3]) : "memory");
                }
                float _tmem_load_10[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 64));
                int col_2 = 96 + lane_half * 16;
                float uv_2[16];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(u + u_base + (long long)col_2);
                    uint4 _vld_4[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&uv_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&uv_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float bvv_15[16];
                #pragma unroll
                for (int j_12 = 0; j_12 < 16; j_12++) {
                    bvv_15[j_12] = uv_1[j_12] - _tmem_load_10[j_12];
                }
                unsigned int bvv_15_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_15[_lp*2 + 0], bvv_15[_lp*2+1 + 0]));
                    bvv_15_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_13 = 0; q_13 < 2; q_13++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((vn_addr + (unsigned int)((col_1 + q_13 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_13 * 8) % 64 * 2 ^ ((col_1 + q_13 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_13 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_15_bf16[q_13 * 4]), "r"(bvv_15_bf16[q_13 * 4 + 1]), "r"(bvv_15_bf16[q_13 * 4 + 2]), "r"(bvv_15_bf16[q_13 * 4 + 3]) : "memory");
                }
                float bo_16[16];
                #pragma unroll
                for (int j_13 = 0; j_13 < 16; j_13++) {
                    bo_16[j_13] = bvv_15[j_13] * rs;
                }
                unsigned int bo_16_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bo_16[_lp*2 + 0], bo_16[_lp*2+1 + 0]));
                    bo_16_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_14 = 0; q_14 < 2; q_14++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_1 + q_14 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_14 * 8) % 64 * 2 ^ ((col_1 + q_14 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_14 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bo_16_bf16[q_14 * 4]), "r"(bo_16_bf16[q_14 * 4 + 1]), "r"(bo_16_bf16[q_14 * 4 + 2]), "r"(bo_16_bf16[q_14 * 4 + 3]) : "memory");
                }
                float _tmem_load_11[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float bvv_17[16];
                #pragma unroll
                for (int j_14 = 0; j_14 < 16; j_14++) {
                    bvv_17[j_14] = uv_2[j_14] - _tmem_load_11[j_14];
                }
                unsigned int bvv_17_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_17[_lp*2 + 0], bvv_17[_lp*2+1 + 0]));
                    bvv_17_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_15 = 0; q_15 < 2; q_15++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((vn_addr + (unsigned int)((col_2 + q_15 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_15 * 8) % 64 * 2 ^ ((col_2 + q_15 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_15 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_17_bf16[q_15 * 4]), "r"(bvv_17_bf16[q_15 * 4 + 1]), "r"(bvv_17_bf16[q_15 * 4 + 2]), "r"(bvv_17_bf16[q_15 * 4 + 3]) : "memory");
                }
                float bo_18[16];
                #pragma unroll
                for (int j_15 = 0; j_15 < 16; j_15++) {
                    bo_18[j_15] = bvv_17[j_15] * rs;
                }
                unsigned int bo_18_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bo_18[_lp*2 + 0], bo_18[_lp*2+1 + 0]));
                    bo_18_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_16 = 0; q_16 < 2; q_16++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_2 + q_16 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_16 * 8) % 64 * 2 ^ ((col_2 + q_16 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_16 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bo_18_bf16[q_16 * 4]), "r"(bo_18_bf16[q_16 * 4 + 1]), "r"(bo_18_bf16[q_16 * 4 + 2]), "r"(bo_18_bf16[q_16 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    g_next = (long long)(c0 + c + 1) * (long long)num_heads + (long long)head;
                    rs_next = (long long)(row0 + (c + 1) * 64 + t_row) * (long long)num_heads + (long long)head;
                    decay = gamma[g_next];
                    rs = rowscale[rs_next];
                }
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int hb_1 = blockIdx.x;
            int vbase_1 = 0;
            int head_1 = hb_1 % num_heads;
            int walk_1 = hb_1 / num_heads;
            int seg_1 = walk_seg[walk_1];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int k_head = head_1 / (num_heads / num_qk_heads);
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nc_1; c_1++) {
                    if (c_1 == 0) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                        int trow = row0_1 + c_1 * 64;
                        int ktok = chunk_bos[c0_1 + c_1];
                        #pragma unroll
                        for (int panel = 0; panel < 2; panel++) {
                            tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)panel) * 8192, (&w_tma), panel * 64, head_1, trow, tiles_full_addr + (stage) * 8);
                            tma_3d_gmem2smem(k_panel_addr + (stage * 2 + (unsigned int)panel) * 8192, (&k_tma), panel * 64, k_head, ktok, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    if (nc_1 > c_1 + 1) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                        int trow_1 = row0_1 + (c_1 + 1) * 64;
                        int ktok_1 = chunk_bos[c0_1 + (c_1 + 1)];
                        #pragma unroll
                        for (int panel_1 = 0; panel_1 < 2; panel_1++) {
                            tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)panel_1) * 8192, (&w_tma), panel_1 * 64, head_1, trow_1, tiles_full_addr + (stage) * 8);
                            tma_3d_gmem2smem(k_panel_addr + (stage * 2 + (unsigned int)panel_1) * 8192, (&k_tma), panel_1 * 64, k_head, ktok_1, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                    _phase_hs_ready_0 ^= 1;
                    int chunk_head = (c0_1 + c_1) * num_heads + head_1;
                    #pragma unroll
                    for (int panel_2 = 0; panel_2 < 2; panel_2++) {
                        tma_store_3d((&h_tape), vbase_1 + panel_2 * 64, 0, chunk_head, hs_panel_addr + (unsigned int)panel_2 * 16384);
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(hs_free_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    int vrow = row0_1 + c_1 * 64;
                    #pragma unroll
                    for (int panel_3 = 0; panel_3 < 2; panel_3++) {
                        tma_store_3d((&v_new), vbase_1 + panel_3 * 64, head_1, vrow, vn_panel_addr + (unsigned int)panel_3 * 8192);
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(vn_free_addr);
                }
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int hb_2 = blockIdx.x;
            int vbase_2 = 0;
            int head_2 = hb_2 % num_heads;
            int walk_2 = hb_2 / num_heads;
            int seg_2 = walk_seg[walk_2];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hs_ready_0_1 = 0;
            unsigned int _phase_bv_ready_0_1 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                    _phase_hs_ready_0_1 ^= 1;
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
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0_1);
                    _phase_bv_ready_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((((k_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_2 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
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
